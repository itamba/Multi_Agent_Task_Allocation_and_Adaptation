"""
Full Training Integration Script - MAPPO + BLADE + MATCH-AOU
=============================================================

Trains RL agents using MAPPO (Multi-Agent PPO) with Centralized Training,
Decentralized Execution (CTDE) in a real BLADE military simulation.

Architecture:
    Actor (shared weights, decentralized):
        local_obs [30] → 128 → 64 → action distribution [5]
    Critic (centralized, padded to MAX_AGENTS):
        global_state [30 * MAX_AGENTS] → 128 → 64 → V(s) [1]

Training approach:
    1. Load BLADE scenario (or generate a variation via ScenarioGenerator)
    2. Extract agents + tasks from scenario
    3. Split tasks: partial (2/3) vs full (all)
    4. Solve MATCH-AOU on both sets
    5. Run BLADE with partial plan (BladeExecutorMinimal)
    6. Event-driven RL decisions (NO periodic — only on trigger events):
       - Discovery: agent sees a target not in its partial plan
       - Fuel damage: agent's fuel is reduced mid-mission
       - On trigger: build local obs + global obs (padded to MAX_AGENTS)
       - Actor samples action from policy π(a|o)
       - Critic estimates state value V(s) from global state
       - Oracle provides ground truth from full solution
       - Compute imitation reward, store in rollout buffer
    7. After episode: compute GAE advantages → PPO update (K epochs)

Usage:
    # Fixed scenario (original behavior):
    python train_full.py --scenario data/scenarios/strike_training_4v5.json --episodes 50

    # Varied scenarios (default):
    python train_full.py --scenario data/scenarios/strike_training_4v5.json \\
        --vary-scenarios --min-aircraft 2 --max-aircraft 3 \\
        --min-facilities 2 --max-facilities 4 --max-target-dist 500 \\
        --vary-base --episodes 100
"""

from __future__ import annotations

import argparse
import copy
import glob
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# Ensure `src/` is importable when running this script directly (so the
# user can just hit "Run" in their IDE without setting PYTHONPATH).
_SRC_DIR = str(Path(__file__).resolve().parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# --- BLADE imports ---
import gymnasium
from blade.Game import Game
from blade.Scenario import Scenario

# Override BLADE's 10MB recording file-size limit to avoid splitting
# recordings into multiple files mid-episode.
import blade.utils.PlaybackRecorder as _pbr
_pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # 500MB

# --- MATCH-AOU imports ---
from match_aou.solvers import MatchAou, round_trip_cost
from match_aou.models import Agent, Task
from match_aou.utils.scheduling_utils import post_solve_filter_and_level
from match_aou.utils.blade_utils import create_agents_from_scenario
from match_aou.utils.blade_utils.blade_executor_minimal import BladeExecutorMinimal
from match_aou.utils.blade_utils.scenario_factory import (
    _normalize_side_color,
    generate_all_enemy_tasks,
)

# --- RL imports ---
from match_aou.rl.agent import ActorCriticNetwork
from match_aou.rl.observation import build_observation_vector, ObservationConfig
from match_aou.rl.observation.observation_utils import extract_target_id_from_action
from match_aou.rl.training import PPOTrainer, PPOConfig
from match_aou.rl.training.reward import (
    compute_step_reward, compute_episode_reward,
    RewardConfig, RewardTracker,
    build_target_utility_map, get_action_utility,
    compute_oracle_total_utility,
)
from match_aou.rl.plan_editor import plan_edit_to_blade_action

# --- Scenario generation ---
from match_aou.utils.blade_utils.scenario_generator import (
    ScenarioGenerator, VariationConfig,
)

# --- Fuel damage events ---
from match_aou.rl.training.fuel_damage import FuelDamageManager, FuelDamageConfig

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent / "src"))

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_full")

# --- Constants ---
ATTACKING_SIDE_COLOR = "blue"
SOLVER_NAME = "bonmin"
MAX_SIM_TICKS = 14400
# RL is event-driven: decisions only on discovery or fuel damage (no periodic)
DISCOVERY_SCAN_INTERVAL = 50  # Check for new targets every N ticks (not every tick)
PROGRESS_LOG_INTERVAL = 1000  # Print simulation progress every N ticks
PARTIAL_RATIO = 2 / 3         # Fraction of tasks in partial set
VARY_SCENARIOS = True          # Toggle scenario variation (or use --vary-scenarios flag)
VARY_BASE = False              # Toggle blue base position randomization
INCLUDE_SAMS = False           # When False, scenarios have only RED airbases (no SAM interception)
FUEL_DAMAGE_ENABLED = False     # Toggle fuel damage surprise events
VALIDATE_EVERY = 100           # Run oracle-only validation every N episodes (0=disabled)
OUTPUT_DIR = "training_output"  # Directory for logs and recordings
MAX_AGENTS = 5                 # Max agents for critic padding (fixed network size)


def all_agents_returned_to_base(observation, agent_ids: List[str]) -> bool:
    """
    Check if ALL our agents have returned to base (no longer airborne).

    In BLADE, when an aircraft lands it's removed from scenario.aircraft
    and placed back into airbase.aircraft. So if none of our agent IDs
    appear in scenario.aircraft, they've all landed.

    Args:
        observation: BLADE Scenario observation
        agent_ids: List of our agent IDs

    Returns:
        True if no agent is airborne
    """
    airborne_ids = set()
    for ac in getattr(observation, "aircraft", []) or []:
        airborne_ids.add(str(getattr(ac, "id", "")))

    for aid in agent_ids:
        if str(aid) in airborne_ids:
            return False  # At least one agent still flying

    return True


# =============================================================================
# 1. BLADE Environment Setup
# =============================================================================

def setup_blade_env(scenario_path: str, max_steps: int = MAX_SIM_TICKS, recording_dir: str = None):
    """
    Load BLADE scenario and create Gym environment.

    Returns:
        (game, env, observation) tuple
    """
    rec_path = recording_dir or str(Path(scenario_path).parent)
    game = Game(
        current_scenario=Scenario(),
        record_every_seconds=10,
        recording_export_path=rec_path,
    )
    with open(scenario_path, "r", encoding="utf-8") as f:
        game.load_scenario(f.read())

    # Log the registered default so we can verify it
    try:
        spec = gymnasium.spec("blade/BLADE-v0")
        logger.info(f"BLADE registered max_episode_steps: {spec.max_episode_steps}")
    except Exception:
        pass

    # Pass max_episode_steps directly to gymnasium.make.
    # This OVERRIDES any default registered with the env spec,
    # preventing a hidden inner TimeLimit from cutting episodes short.
    duration_from_scenario = 0
    env = gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=max_steps)
    observation, info = env.reset()

    duration_from_scenario = int(getattr(observation, "duration", 0) or 0)
    if duration_from_scenario > max_steps:
        logger.warning(
            f"Scenario duration ({duration_from_scenario}) > max_steps ({max_steps}). "
            f"Consider increasing --max-ticks."
        )

    logger.info(
        f"BLADE env ready: duration={duration_from_scenario}, "
        f"max_episode_steps={max_steps}, "
        f"start_time={getattr(observation, 'start_time', '?')}, "
        f"current_time={getattr(observation, 'current_time', '?')}"
    )

    return game, env, observation


def reload_scenario(game: Game, scenario_path: str) -> None:
    """Load a new scenario JSON into an existing Game.

    After this call, the next env.reset() will start the new scenario.
    This avoids recreating the Game and Gym env from scratch each episode.

    Args:
        game: Existing Game instance
        scenario_path: Path to the new scenario JSON
    """
    with open(scenario_path, "r", encoding="utf-8") as f:
        game.load_scenario(f.read())
    logger.debug(f"Reloaded scenario from {scenario_path}")


# =============================================================================
# 3. MATCH-AOU Solving
# =============================================================================

def solve_match_aou(
    agents: List[Agent],
    tasks: List[Task],
    solver_name: str = SOLVER_NAME,
) -> Tuple[Dict, List[Task], List[int]]:
    """
    Solve MATCH-AOU and return post-processed solution.

    Args:
        agents: List of Agent objects
        tasks: List of Task objects
        solver_name: Solver to use (default: bonmin)

    Returns:
        (solution, filtered_tasks, unselected_task_indices)
        solution: {agent_id: [(task_idx, step_idx, level), ...]}
        filtered_tasks: Tasks after filtering unselected
        unselected_task_indices: Indices of tasks not selected by solver
    """
    if not tasks or not agents:
        logger.warning("No tasks or agents to solve")
        return {}, tasks, []

    # risk_factor=0.0: the movement-budget constraint now charges an explicit round-trip
    # per target, so no reserve margin is needed (Phase 2 may revisit with sigma > 0).
    model = MatchAou(agents=agents, tasks=tasks, precedence_relations=[], risk_factor=0.0)
    solution, results, unselected = model.solve(solver_name=solver_name)

    if not solution:
        logger.warning("MATCH-AOU returned empty solution")
        return {}, tasks, list(range(len(tasks)))

    # Post-process: filter unselected tasks, compute topological levels
    artifacts = post_solve_filter_and_level(
        tasks=tasks,
        solution=solution,
        precedence_relations=[],
        unselected_tasks=unselected,
    )

    logger.debug(f"  → {sum(len(v) for v in artifacts.solution.values())} assignments, {len(unselected)} unselected")

    return artifacts.solution, artifacts.tasks, unselected


def _collect_blue_aircraft(observation) -> List:
    """Pull blue-side aircraft from a BLADE observation (top-level + airbases)."""
    out = []
    for ac in getattr(observation, "aircraft", []) or []:
        if _normalize_side_color(getattr(ac, "side_color", "")) == "blue":
            out.append(ac)
    for ab in getattr(observation, "airbases", []) or []:
        if _normalize_side_color(getattr(ab, "side_color", "")) == "blue":
            for ac in getattr(ab, "aircraft", []) or []:
                out.append(ac)
    return out


def _min_fleet_radar_km(observation) -> Optional[float]:
    """Minimum radar detection range across the blue fleet, in km."""
    rngs_nm = []
    for ac in _collect_blue_aircraft(observation):
        r = float(getattr(ac, "range", 0) or 0)
        if r > 0:
            rngs_nm.append(r)
    if not rngs_nm:
        return None
    return min(rngs_nm) * 1.852


def split_tasks(
    all_tasks: List[Task],
    partial_ratio: float = PARTIAL_RATIO,
    observation=None,
    max_attempts: int = 20,
) -> Tuple[List[Task], List[Task], Dict]:
    """Split tasks into partial (known) and full (oracle) sets.

    Discovery-chain aware: ensures every hidden task has at least one
    known task within the minimum fleet radar range, so a masked target
    can in principle be discovered when an aircraft visits a known
    same-zone neighbour. Algorithm:

    1. Build the radar-adjacency graph between targets using
       `min_fleet_radar_km` and pairwise great-circle distance between
       `task.steps[0].location` points.
    2. Pin "isolated" targets (no radar neighbour at all) into the
       known set — there is no other path to discover them. Layer 1
       in `scenario_generator._ensure_discovery_chain` already minimises
       these via per-zone connectivity, but they can remain in
       single-target zones or pathological geometries.
    3. Random-sample the rest up to `max_attempts` times. A draw is
       valid iff every hidden target has at least one known radar
       neighbour. Log `clean` on first success, `resampled (attempt N)`
       on later success, or a warning on exhaustion (last draw kept).

    Args:
        all_tasks: All generated tasks (each with `steps[0].location`).
        partial_ratio: Fraction for partial set (default 2/3).
        observation: BLADE Scenario observation. When None, falls back to
            the original mask-blind random sample (for callers that don't
            yet pass it; preserves backwards compatibility).
        max_attempts: Cap on rejection-sampling retries before giving up.

    Returns:
        (partial_tasks, full_tasks, split_meta)
        split_meta keys:
            outcome: "clean" | "resampled" | "exhaust" | "warn-fallback" | "no-chain"
            attempt: 1-based attempt index when outcome is clean/resampled
            hidden, known, isolated_pinned, partial, full: counts
    """
    full_tasks = list(all_tasks)
    n = len(all_tasks)
    num_partial = max(1, int(n * partial_ratio))

    # Backwards-compatible fallback: no observation → no connectivity check.
    if observation is None or n < 2:
        partial_tasks = random.sample(all_tasks, num_partial) if n else []
        hidden = [t for t in full_tasks if t not in partial_tasks]
        logger.debug(
            f"Task split: {len(partial_tasks)} partial, {len(full_tasks)} full, "
            f"{len(hidden)} hidden"
        )
        meta = {
            "outcome": "no-chain", "attempt": 1,
            "hidden": len(hidden), "known": len(partial_tasks),
            "isolated_pinned": 0,
            "partial": len(partial_tasks), "full": len(full_tasks),
        }
        return partial_tasks, full_tasks, meta

    min_radar_km = _min_fleet_radar_km(observation)
    if min_radar_km is None:
        # Fleet has no radar range data; fall back to plain random.
        partial_tasks = random.sample(all_tasks, num_partial)
        hidden = [t for t in full_tasks if t not in partial_tasks]
        logger.debug(
            f"Task split: {len(partial_tasks)} partial, {len(full_tasks)} full, "
            f"{len(hidden)} hidden (chain check skipped: no radar range)"
        )
        meta = {
            "outcome": "no-chain", "attempt": 1,
            "hidden": len(hidden), "known": len(partial_tasks),
            "isolated_pinned": 0,
            "partial": len(partial_tasks), "full": len(full_tasks),
        }
        return partial_tasks, full_tasks, meta

    # Build radar adjacency between tasks (by index).
    locs = [task.steps[0].location for task in all_tasks]
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if locs[i].distance_to(locs[j]) <= min_radar_km:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Pin isolated tasks (no radar neighbour) to the known set.
    isolated = {i for i in range(n) if not neighbors[i]}

    if len(isolated) > num_partial:
        # Too many isolated to fit in the partial budget; force as many
        # as fit into known and hide the rest (will be undiscoverable).
        forced = list(isolated)[:num_partial]
        partial_tasks = [all_tasks[i] for i in forced]
        hidden = [t for t in full_tasks if t not in partial_tasks]
        logger.warning(
            f"Discovery chain (split): isolated={len(isolated)} exceeds partial "
            f"budget={num_partial}; {len(isolated) - num_partial} isolated "
            f"target(s) will be hidden and undiscoverable"
        )
        logger.debug(
            f"Task split: {len(partial_tasks)} partial, {len(full_tasks)} full, "
            f"{len(hidden)} hidden"
        )
        meta = {
            "outcome": "exhaust", "attempt": 0,
            "hidden": len(hidden), "known": len(partial_tasks),
            "isolated_pinned": min(len(isolated), num_partial),
            "partial": len(partial_tasks), "full": len(full_tasks),
        }
        return partial_tasks, full_tasks, meta

    pool = [i for i in range(n) if i not in isolated]
    extra_needed = num_partial - len(isolated)

    last_known_set: Set[int] = set()
    for attempt in range(1, max_attempts + 1):
        sampled = random.sample(pool, extra_needed) if extra_needed > 0 else []
        known_set = isolated | set(sampled)
        hidden_set = set(range(n)) - known_set

        valid = all(bool(neighbors[h] & known_set) for h in hidden_set)
        last_known_set = known_set
        if valid:
            partial_tasks = [all_tasks[i] for i in known_set]
            hidden = [t for t in full_tasks if t not in partial_tasks]
            tag = "clean" if attempt == 1 else f"resampled (attempt {attempt})"
            logger.debug(
                f"Discovery chain (split): {tag} (hidden={len(hidden_set)}, "
                f"known={len(known_set)}, isolated_pinned={len(isolated)}, "
                f"min_radar={min_radar_km:.0f} km)"
            )
            logger.debug(
                f"Task split: {len(partial_tasks)} partial, {len(full_tasks)} full, "
                f"{len(hidden)} hidden"
            )
            meta = {
                "outcome": "clean" if attempt == 1 else "resampled",
                "attempt": attempt,
                "hidden": len(hidden_set), "known": len(known_set),
                "isolated_pinned": len(isolated),
                "partial": len(partial_tasks), "full": len(full_tasks),
            }
            return partial_tasks, full_tasks, meta

    # Exhausted retries — keep last draw and warn.
    partial_tasks = [all_tasks[i] for i in last_known_set]
    hidden = [t for t in full_tasks if t not in partial_tasks]
    logger.warning(
        f"Discovery chain (split): no valid split after {max_attempts} attempts; "
        f"some hidden targets may have no known radar neighbour "
        f"(min_radar={min_radar_km:.0f} km)"
    )
    logger.debug(
        f"Task split: {len(partial_tasks)} partial, {len(full_tasks)} full, "
        f"{len(hidden)} hidden"
    )
    meta = {
        "outcome": "warn-fallback", "attempt": max_attempts,
        "hidden": len(hidden), "known": len(partial_tasks),
        "isolated_pinned": len(isolated),
        "partial": len(partial_tasks), "full": len(full_tasks),
    }
    return partial_tasks, full_tasks, meta


# =============================================================================
# 4. Oracle Logic
# =============================================================================

def extract_target_ids_from_solution(
    solution: Dict[str, List[Tuple]],
    tasks: List[Task],
) -> Dict[str, Set[str]]:
    """
    Extract target IDs assigned to each agent in a solution.

    Args:
        solution: {agent_id: [(task_idx, step_idx, level), ...]}
        tasks: Task objects (indices reference into this list)

    Returns:
        {agent_id: {target_id_1, target_id_2, ...}}
    """
    agent_targets: Dict[str, Set[str]] = {}

    for agent_id, assignments in solution.items():
        targets = set()
        for task_idx, step_idx, _level in assignments:
            if 0 <= task_idx < len(tasks):
                task = tasks[task_idx]
                if 0 <= step_idx < len(task.steps):
                    action = getattr(task.steps[step_idx], "action", "") or ""
                    target_id = extract_target_id_from_action(action)
                    if target_id:
                        targets.add(target_id)
        agent_targets[agent_id] = targets

    return agent_targets


def get_oracle_action(
    observation,         # ObservationOutput from build_observation_vector
    agent_id: str,
    full_agent_targets: Dict[str, Set[str]],  # From extract_target_ids_from_solution
) -> int:
    """
    Determine what the oracle would do given full knowledge.

    The oracle knows the full MATCH-AOU solution. It checks:
    - For each visible target NOT in the partial plan (is_in_plan == False)
    - If that target IS assigned to this agent in the full solution
    - → Oracle says INSERT_ATTACK on that target slot

    Args:
        observation: ObservationOutput with targets info
        agent_id: Current agent ID
        full_agent_targets: {agent_id: {target_ids}} from full solution

    Returns:
        Action index: 0=NOOP, 1-3=INSERT_ATTACK(slot), 4=FORCE_RTB
    """
    # Get targets assigned to this agent in the FULL solution
    my_full_targets = full_agent_targets.get(agent_id, set())

    for slot_idx, target in enumerate(observation.targets):
        if not target.exists:
            continue
        if target.is_in_plan:
            continue  # Already being handled by partial plan

        # This target is visible but NOT in partial plan.
        # Is it assigned to this agent in the full solution?
        if target.id in my_full_targets:
            return slot_idx + 1  # INSERT_ATTACK(slot_idx)

    return 0  # NOOP — everything is fine, continue partial plan


# =============================================================================
# 5. Discovery Detection
# =============================================================================

def check_discovery(
    observation,         # ObservationOutput
    partial_target_ids: Set[str],
) -> bool:
    """
    Check if agent sees a target that wasn't in the partial plan.

    This is the "surprise" — the agent discovers a target it didn't know about.

    Args:
        observation: ObservationOutput with visible targets
        partial_target_ids: Set of target IDs in the partial solution

    Returns:
        True if a new (undiscovered) target is visible
    """
    for target in observation.targets:
        if target.exists and not target.is_in_plan:
            # This target is visible but not in partial plan
            if target.id not in partial_target_ids:
                return True
    return False


# =============================================================================
# 6. Action Mask Helper
# =============================================================================

def get_simple_action_mask(observation, action_dim: int = 5) -> np.ndarray:
    """
    Build a simple action mask from observation.

    Rules:
    - NOOP (0): Always valid
    - INSERT_ATTACK(k) (1-3): Valid if target slot k exists and has a real target
    - FORCE_RTB (4): Always valid (agent can always RTB)

    For full validation (weapon checks, cooldown, etc.), use the action module's
    compute_action_mask. This simplified version avoids needing the BLADE scenario
    object in the action validator.

    Args:
        observation: ObservationOutput
        action_dim: Action space size (default 5)

    Returns:
        Boolean numpy array of shape [action_dim]
    """
    mask = np.zeros(action_dim, dtype=bool)
    mask[0] = True  # NOOP always valid
    mask[4] = True  # RTB always valid

    # Attack actions valid if target exists and agent has weapons
    has_weapon = observation.self_state.has_weapon > 0.5
    for slot_idx, target in enumerate(observation.targets):
        if slot_idx >= 3:
            break
        if target.exists and has_weapon:
            mask[slot_idx + 1] = True

    return mask


# =============================================================================
# 6b. Simulation Logging Helpers
# =============================================================================

# Pre-compiled regex patterns for parsing BLADE action strings
_RE_ATTACK = re.compile(r"handle_aircraft_attack\('([^']*)'[^']*'([^']*)'")
_RE_MOVE = re.compile(r"move_aircraft\('([^']*)',\s*\[\[([^\]]+)\]\]")
_RE_LAUNCH = re.compile(r"launch_aircraft_from_airbase\('([^']*)'\)")
_RE_RTB = re.compile(r"return_to_base\('([^']*)'\)")


def _build_name_lookup(scenario) -> Dict[str, str]:
    """Build a {uuid → human-readable name} map from a BLADE scenario observation.

    Sweeps top-level aircraft, airbases (and aircraft nested inside them),
    and facilities. Used at episode start to resolve UUIDs in debug logs
    without reaching into BLADE objects on every log line.

    Defensive: missing `.name` or `.id` attributes are skipped silently;
    UUIDs that don't appear in the scenario simply won't be in the map,
    and `_id_label` falls back to UUID-only formatting in that case.
    """
    lookup: Dict[str, str] = {}

    def _add(obj):
        oid = getattr(obj, "id", None)
        name = getattr(obj, "name", None)
        if oid and name:
            lookup[str(oid)] = str(name)

    for ac in getattr(scenario, "aircraft", []) or []:
        _add(ac)
    for ab in getattr(scenario, "airbases", []) or []:
        _add(ab)
        for ac in getattr(ab, "aircraft", []) or []:
            _add(ac)
    for f in getattr(scenario, "facilities", []) or []:
        _add(f)

    return lookup


def _id_label(uuid: str, name_lookup: Optional[Dict[str, str]] = None) -> str:
    """Format a UUID as '8charpfx.. (Name)' or just '8charpfx..' if not found.

    `name_lookup=None` (or no match) preserves the legacy UUID-only format
    so callers that don't have a lookup keep working unchanged.
    """
    short = str(uuid)[:8]
    if name_lookup:
        name = name_lookup.get(str(uuid))
        if name:
            return f"{short}.. ({name})"
    return f"{short}.."


def _log_blade_action(
    tick: int,
    action: str,
    source: str,
    name_lookup: Optional[Dict[str, str]] = None,
) -> None:
    """
    Parse a BLADE action string and log it in a human-readable format.

    Args:
        tick: Current simulation tick
        action: BLADE action string (e.g., "handle_aircraft_attack(...)")
        source: "EXEC" for executor, "RL" for RL override
        name_lookup: Optional {uuid → name} map for inline naming.
    """
    if not action:
        return

    m = _RE_ATTACK.search(action)
    if m:
        agent_id, target_id = m.group(1), m.group(2)
        logger.debug(
            f"  Tick {tick:5d} [{source:4s}] ATTACK: "
            f"agent {_id_label(agent_id, name_lookup)} → "
            f"target {_id_label(target_id, name_lookup)}"
        )
        return

    m = _RE_MOVE.search(action)
    if m:
        agent_id, coords = m.group(1), m.group(2)
        logger.debug(
            f"  Tick {tick:5d} [{source:4s}] MOVE:   "
            f"agent {_id_label(agent_id, name_lookup)} → ({coords})"
        )
        return

    m = _RE_LAUNCH.search(action)
    if m:
        logger.debug(
            f"  Tick {tick:5d} [{source:4s}] LAUNCH: "
            f"from airbase {_id_label(m.group(1), name_lookup)}"
        )
        return

    m = _RE_RTB.search(action)
    if m:
        logger.debug(
            f"  Tick {tick:5d} [{source:4s}] RTB:    "
            f"agent {_id_label(m.group(1), name_lookup)}"
        )
        return

    # Fallback: unknown action format
    logger.debug(f"  Tick {tick:5d} [{source:4s}] ACTION: {action[:80]}")


def _log_progress(
    tick: int,
    n_agents: int,
    returned_agents: Set[str],
    decisions: int,
    episode_reward: float,
    rl_attacked_target_ids: Set[str],
    n_tasks: int,
) -> None:
    """Log a periodic progress summary during the simulation."""
    airborne = n_agents - len(returned_agents)
    logger.debug(
        f"  ── Tick {tick:5d} ── "
        f"airborne: {airborne}/{n_agents} | "
        f"RL decisions: {decisions} | "
        f"reward: {episode_reward:+.2f} | "
        f"targets attacked: {len(rl_attacked_target_ids)}/{n_tasks}"
    )


# =============================================================================
# 7. Validation Episode (oracle-only, no RL)
# =============================================================================

def run_validation_episode(
    game: Game,
    env,
    scenario_path: str,
    episode_num: int,
    max_ticks: int = MAX_SIM_TICKS,
    record: bool = True,
) -> Optional[Dict]:
    """
    Run the full MATCH-AOU solution through BLADE without RL intervention.

    Purpose: produce a recording where the oracle plan executes cleanly,
    so we can visually verify that MATCH-AOU assignments are correct and
    aircraft actually reach and attack their targets.

    Flow:
        1. Reset BLADE
        2. Create agents + ALL tasks (no partial split)
        3. Solve MATCH-AOU on full task set
        4. Launch aircraft
        5. Run BladeExecutorMinimal until all agents RTB or max ticks
        6. Export recording as episode_XXX_validation.jsonl
    """
    logger.debug("--- Validation run (oracle only, no RL) ---")

    # --- Reset ---
    observation, info = env.reset()

    blue_side = None
    for side in observation.sides:
        if str(getattr(side, "name", "")).upper() == "BLUE":
            blue_side = side
            break
    if blue_side:
        game.current_side_id = blue_side.id

    # --- Create agents and tasks ---
    agents_by_side = create_agents_from_scenario(observation)
    attacking_agents = agents_by_side.get(ATTACKING_SIDE_COLOR, [])
    if not attacking_agents:
        logger.warning("Validation: no agents found, skipping")
        return None

    all_tasks = generate_all_enemy_tasks(observation, ATTACKING_SIDE_COLOR)
    if not all_tasks:
        logger.warning("Validation: no tasks found, skipping")
        return None

    logger.debug(f"Validation: {len(attacking_agents)} agents, {len(all_tasks)} tasks")

    # --- Reachability audit (silent data collection) ---
    RISK = 0.0  # must match MatchAou(risk_factor=...) in solve_match_aou()
    # target_cost[tid][agent_id] = one_way move cost (fuel units) — informational (cheapest= display only)
    # target_rt[tid][agent_id]   = round-trip cost via round_trip_cost — the SAME number the
    #                              solver's movement-budget constraint charges; used for reach + used.
    target_cost: Dict[str, Dict[str, float]] = {}
    target_rt: Dict[str, Dict[str, float]] = {}
    target_reach: Dict[str, Set[str]] = {}
    target_short: Dict[str, str] = {}
    for task in all_tasks:
        tid = extract_target_id_from_action(task.steps[0].action) or "?"
        target_short[tid] = tid[:8]
        target_cost[tid] = {}
        target_rt[tid] = {}
        target_reach[tid] = set()
        step_loc = task.steps[0].location
        for agent in attacking_agents:
            try:
                c = float(agent.move_cost(destination=step_loc, source=agent.location))
            except Exception:
                c = float("inf")
            target_cost[tid][str(agent.id)] = c
            try:
                rt = float(round_trip_cost(agent, step_loc))
            except Exception:
                rt = float("inf")
            target_rt[tid][str(agent.id)] = rt
            if rt <= agent.budget:  # full round-trip fits the budget (matches the solver constraint)
                target_reach[tid].add(str(agent.id))
    unreachable_tids = {tid for tid, r in target_reach.items() if not r}

    # --- Solve MATCH-AOU (full) ---
    solution, tasks_filtered, _ = solve_match_aou(
        attacking_agents, all_tasks, SOLVER_NAME
    )
    if not solution:
        logger.warning("Validation: solver returned empty solution, skipping")
        return None

    # --- Collect oracle plan per agent (silent) ---
    oracle_plan: Dict[str, List[str]] = {str(a.id): [] for a in attacking_agents}
    oracle_assigned_tids: Set[str] = set()
    oracle_violations = 0
    for agent_id, assignments in solution.items():
        aid = str(agent_id)
        oracle_plan.setdefault(aid, [])
        for task_idx, _step_idx, _level in assignments:
            if task_idx >= len(tasks_filtered):
                continue
            tsk = tasks_filtered[task_idx]
            tid = extract_target_id_from_action(tsk.steps[0].action) or "?"
            oracle_plan[aid].append(tid)
            oracle_assigned_tids.add(tid)
            if aid not in target_reach.get(tid, set()):
                oracle_violations += 1

    # Name lookup for human-readable log lines (UUID → "B-2 Spirit #698").
    # Built once here, after env.reset has populated airbases (still pre-launch),
    # so blue aircraft (in airbase.aircraft) and red facilities/airbases are
    # all captured in one sweep.
    name_lookup = _build_name_lookup(observation)

    # --- Per-agent oracle plan dump (intent before simulation) ---
    # Combined with the [VAL ] action stream below, this lets a triage
    # parser reconstruct exactly when each planned step happened (or
    # didn't) for the two validation-truncate cases. The 8-char prefix
    # is preserved so existing parsers still match; names are appended
    # in parens for human readers.
    for aid in [str(a.id) for a in attacking_agents]:
        plan_tids = oracle_plan.get(aid, [])
        plan_labels = [_id_label(t, name_lookup) for t in plan_tids] if plan_tids else ["-"]
        logger.debug(
            f"  VAL plan: agent={_id_label(aid, name_lookup)} → tasks={plan_labels}"
        )

    # --- Launch aircraft ---
    game.current_scenario.name = f"ep{episode_num + 1:03d}_validation"
    rec_step = game.record_step if record else (lambda *a, **kw: None)
    if record:
        game.start_recording()
    rec_step()

    for _ in range(5):
        observation, _, _, _, _ = env.step("")
        rec_step(force=True)

    for airbase in getattr(observation, "airbases", []) or []:
        ab_side = _normalize_side_color(getattr(airbase, "side_color", ""))
        if ab_side != ATTACKING_SIDE_COLOR:
            continue
        ab_id = str(airbase.id)
        for ac in list(getattr(airbase, "aircraft", []) or []):
            ac_name = getattr(ac, "name", str(ac.id))
            logger.debug(
                f"  Validation LAUNCH: {ac_name} (id={str(ac.id)[:8]}..) "
                f"from airbase {_id_label(ab_id, name_lookup)}"
            )
            observation, _, _, _, _ = env.step(
                f"launch_aircraft_from_airbase('{ab_id}')"
            )
            rec_step()

    for _ in range(10):
        observation, _, _, _, _ = env.step("")
        rec_step()

    # --- Setup executor with FULL plan ---
    executor = BladeExecutorMinimal(
        tasks=tasks_filtered,
        solution=solution,
        agents=attacking_agents,
        add_return_to_base=True,
        arrival_threshold_km=50.0,
    )

    # --- Simulation loop (executor only) ---
    agent_ids = [str(a.id) for a in attacking_agents]
    returned: set = set()
    attacked_tids: Set[str] = set()

    for tick in range(max_ticks):
        try:
            action = executor.next_action(observation, fallback_tick=tick) or ""
        except ValueError:
            action = ""

        if action and "handle_aircraft_attack" in action:
            tid = extract_target_id_from_action(action)
            if tid:
                attacked_tids.add(tid)

        # Per-tick action log (DEBUG → per-episode file only). Source tag
        # "VAL " distinguishes validation-phase actions from RL-phase
        # [EXEC]/[RL  ] in mixed-phase episode files. Same parser path
        # used by the RL loop, so RTB issuance is captured here too via
        # the substring match in _RE_RTB.
        if action:
            _log_blade_action(tick, action, "VAL", name_lookup)

        observation, _, terminated, truncated, _ = env.step(action)
        rec_step()

        # Check RTB
        airborne_ids = {
            str(getattr(ac, "id", ""))
            for ac in getattr(observation, "aircraft", []) or []
        }
        for aid in agent_ids:
            if aid not in returned and aid not in airborne_ids:
                returned.add(aid)
                logger.debug(
                    f"  Tick {tick:5d} VAL RTB: agent {_id_label(aid, name_lookup)} landed"
                )

        # End-zone diagnostic block (mirror of the RL loop): every 10th
        # tick in the last 100 before max_ticks, dump per-aircraft state
        # so a truncation can be diagnosed without a recording.
        ticks_remaining = max_ticks - tick
        if ticks_remaining <= 100 and ticks_remaining % 10 == 0:
            all_aircraft = getattr(observation, "aircraft", []) or []
            logger.debug(
                f"  ── Tick {tick:5d} [VAL END-ZONE] ── "
                f"remaining={ticks_remaining} | "
                f"airborne={len(all_aircraft)} | "
                f"returned={len(returned)}/{len(agent_ids)} | "
                f"terminated={terminated} | truncated={truncated}"
            )
            for ac in all_aircraft:
                ac_id = str(getattr(ac, "id", ""))[:8]
                ac_name = getattr(ac, "name", ac_id)
                fuel = getattr(ac, "current_fuel", 0)
                rtb = getattr(ac, "rtb", False)
                lat = getattr(ac, "latitude", 0)
                lon = getattr(ac, "longitude", 0)
                route_len = len(getattr(ac, "route", []) or [])
                logger.debug(
                    f"    {ac_name} (id={ac_id}..): "
                    f"pos=({lat:.2f},{lon:.2f}) fuel={fuel:.0f} "
                    f"rtb={rtb} route_pts={route_len}"
                )

        if tick > 100 and len(returned) == len(agent_ids):
            logger.debug(f"  Validation: all agents RTB at tick {tick}")
            break
        if terminated or truncated:
            logger.warning(
                f"  Validation ended at tick {tick}: "
                f"terminated={terminated}, truncated={truncated}"
            )
            break

    # --- Validation audit block (one compact summary) ---
    reachable_tids = {tid for tid, r in target_reach.items() if r}
    reachable_hit = len(attacked_tids & reachable_tids)
    unreachable_hit = len(attacked_tids & unreachable_tids)
    plan_hit = len(attacked_tids & oracle_assigned_tids)
    dropped = reachable_tids - oracle_assigned_tids  # reachable but not in oracle plan

    logger.info("  --- Validation audit ---")
    # Per-target line: [short] reach=[A1,A2] plan=A1 hit=Y cheapest=A1:cost
    for tid in target_short:
        reach_ids = sorted(target_reach[tid])
        reach_tag = ",".join(a[:4] for a in reach_ids) if reach_ids else "-"
        planned_by = [aid[:4] for aid, tids in oracle_plan.items() if tid in tids]
        plan_tag = ",".join(planned_by) if planned_by else "-"
        hit_tag = "Y" if tid in attacked_tids else "N"
        costs = target_cost[tid]
        if costs:
            cheapest_aid = min(costs, key=costs.get)
            cheap_tag = f"{cheapest_aid[:4]}:{costs[cheapest_aid]:.0f}"
        else:
            cheap_tag = "-"
        logger.info(
            f"    t={target_short[tid]} reach=[{reach_tag}] plan=[{plan_tag}] hit={hit_tag} cheapest={cheap_tag}"
        )
    # Per-agent line: [short] budget cap used/cap plan=[t1,t2]
    for agent in attacking_agents:
        aid = str(agent.id)
        cap = agent.budget * (1.0 - RISK)
        plan_tids = oracle_plan.get(aid, [])
        # Round-trip per assigned target — the same number the solver constraint charges.
        used = sum(target_rt[tid][aid] for tid in plan_tids if tid in target_rt and aid in target_rt[tid])
        plan_tag = ",".join(target_short[tid] for tid in plan_tids) if plan_tids else "-"
        logger.info(
            f"    agent={aid[:4]} budget={agent.budget:.0f} cap={cap:.0f} "
            f"used={used:.0f}/{cap:.0f} plan=[{plan_tag}]"
        )
    # Headline summary
    logger.info(
        f"  Hit: plan={plan_hit}/{len(oracle_assigned_tids)} "
        f"reachable={reachable_hit}/{len(reachable_tids)} "
        f"unreachable={unreachable_hit}/{len(unreachable_tids)} "
        f"dropped_reachable={len(dropped)} oracle_violations={oracle_violations}"
    )
    if dropped:
        logger.info(
            f"  Dropped reachable targets (oracle chose not to plan): "
            f"{[target_short[t] for t in dropped]}"
        )
    if unreachable_hit > 0:
        logger.error(
            f"  ANOMALY: unreachable target(s) attacked: "
            f"{sorted(attacked_tids & unreachable_tids)}"
        )
    if plan_hit < len(oracle_assigned_tids):
        missed = oracle_assigned_tids - attacked_tids
        logger.warning(
            f"  Oracle plan incomplete in execution — missed: "
            f"{[target_short[t] for t in missed]}"
        )

    # --- Export recording ---
    if record:
        try:
            game.export_recording()
            logger.debug(f"  Validation recording exported: ep{episode_num + 1:03d}_validation")
        except Exception as e:
            logger.warning(f"  Failed to export validation recording: {e}")

    # Return audit summary so the main loop can detect !ANOMALY and tally
    # validation outcomes for run_summary.txt.
    return {
        "oracle_violations": oracle_violations,
        "unreachable_hit": unreachable_hit,
        "plan_hit": plan_hit,
        "plan_total": len(oracle_assigned_tids),
        "reachable_hit": reachable_hit,
        "reachable_total": len(reachable_tids),
        "dropped_reachable": len(dropped),
    }


# =============================================================================
# 8. Training Episode
# =============================================================================

def train_episode(
    trainer: PPOTrainer,
    game: Game,
    env,
    scenario_path: str,
    obs_config: ObservationConfig,
    episode_num: int,
    max_ticks: int = MAX_SIM_TICKS,
    record: bool = True,
    fuel_damage_enabled: bool = FUEL_DAMAGE_ENABLED,
    debug_force_flags: Optional[Set[str]] = None,
    record_name: Optional[str] = None,
    replay_only: bool = False,
) -> Dict:
    """
    Run a single training episode with MAPPO (PPO + CTDE).

    Episode 0 prints full diagnostic info. Episodes 1+ print compact runtime info.

    Event-driven RL: decisions happen ONLY when a trigger event occurs
    for a specific agent (discovery or fuel damage). No periodic decisions.
    This avoids polluting the rollout buffer with meaningless NOOP transitions.

    MAPPO flow per triggered agent:
        1. Build local obs for EACH agent
        2. Concatenate all → global_obs (for centralized critic)
        3. Actor(local_obs) → action sample + log_prob
        4. Critic(global_obs) → value estimate
        5. Store (local_obs, global_obs, action, log_prob, reward, value, done, mask)

    After episode ends:
        6. Compute GAE advantages
        7. PPO update (K epochs over collected data)
        8. Clear buffer
    """
    # The diagnostic dumps below were originally gated on `episode_num == 0`
    # so subsequent episodes wouldn't flood the console. Now every episode
    # writes them to its own per-episode DEBUG file (training_output/logs/
    # episode_NNNN.log), so we always run the dumps and rely on log-level
    # filtering at the handler boundary to keep the console quiet.
    verbose = True
    # Per-episode aggregates exposed in the returned metrics dict so the
    # main loop can build the compact summary line.
    action_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}  # NOOP, ATK_0, ATK_1, ATK_2, RTB
    timeout_flag = False
    all_rtb_flag = False

    # --- Step 1: Reset BLADE ---
    observation, info = env.reset()

    blue_side = None
    for side in observation.sides:
        if str(getattr(side, "name", "")).upper() == "BLUE":
            blue_side = side
            break
    if blue_side:
        game.current_side_id = blue_side.id

    # --- Step 2: Create agents and tasks ---
    agents_by_side = create_agents_from_scenario(observation)
    attacking_agents = agents_by_side.get(ATTACKING_SIDE_COLOR, [])
    if not attacking_agents:
        logger.error("No attacking agents found!")
        return _empty_metrics()

    all_tasks = generate_all_enemy_tasks(observation, ATTACKING_SIDE_COLOR)
    if not all_tasks:
        logger.error("No tasks generated!")
        return _empty_metrics()

    # Sort agents to match airbase FIFO order
    airbase_fifo_order = []
    for airbase in getattr(observation, "airbases", []) or []:
        for ac in getattr(airbase, "aircraft", []) or []:
            airbase_fifo_order.append(str(getattr(ac, "id", "")))
    if airbase_fifo_order:
        def _fifo_sort_key(agent):
            aid = str(agent.id)
            return airbase_fifo_order.index(aid) if aid in airbase_fifo_order else 9999
        attacking_agents = sorted(attacking_agents, key=_fifo_sort_key)

    # --- Compact scenario summary (every episode) ---
    _blue_base = None
    _ac_types = []
    for ab in getattr(observation, "airbases", []) or []:
        if _normalize_side_color(getattr(ab, "side_color", "")) == ATTACKING_SIDE_COLOR:
            _blue_base = ab
            _ac_types = [getattr(ac, "class_name", "?") for ac in getattr(ab, "aircraft", [])]
            break

    _base_loc = f"({_blue_base.latitude:.2f}, {_blue_base.longitude:.2f})" if _blue_base else "?"

    _target_parts = []
    for fac in getattr(observation, "facilities", []) or []:
        if _normalize_side_color(getattr(fac, "side_color", "")) != ATTACKING_SIDE_COLOR:
            _target_parts.append(
                f"{getattr(fac, 'class_name', '?')} ({fac.latitude:.2f}, {fac.longitude:.2f})"
            )
    for ab in getattr(observation, "airbases", []) or []:
        if _normalize_side_color(getattr(ab, "side_color", "")) != ATTACKING_SIDE_COLOR:
            _target_parts.append(
                f"Red Airbase ({ab.latitude:.2f}, {ab.longitude:.2f})"
            )

    logger.debug(
        f"Scenario: {len(_ac_types)} agents {_ac_types} | Blue base: {_base_loc}"
    )
    logger.debug(
        f"  Targets ({len(_target_parts)}): {', '.join(_target_parts)}"
    )

    # --- Verbose: print agents and tasks (always; level filtering hides
    # this on console in long-run mode but keeps it in episode_*.log) ---
    if verbose:
        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("AGENTS")
        logger.debug("=" * 60)
        for i, a in enumerate(attacking_agents):
            logger.debug(f"  Agent {i}: {a.id}")
            logger.debug(f"    Name:      (from scenario)")
            logger.debug(f"    Location:  ({a.location.latitude:.4f}, {a.location.longitude:.4f})")
            logger.debug(f"    Budget:    {a.budget:.0f}")
            logger.debug(f"    Weapon ID: {a.weapon_id}")
            logger.debug(f"    Home base: {a.home_base_id}")
            caps = [c.name for c in a.capabilities] if a.capabilities else []
            logger.debug(f"    Capabilities: {caps}")

        logger.debug("")
        logger.debug("=" * 60)
        logger.debug(f"ALL TASKS ({len(all_tasks)} total)")
        logger.debug("=" * 60)
        for i, t in enumerate(all_tasks):
            action_str = t.steps[0].action
            target_id = extract_target_id_from_action(action_str) or "?"
            loc = t.steps[0].location
            logger.debug(f"  Task {i}:")
            logger.debug(f"    Target ID: {target_id}")
            logger.debug(f"    Utility:   {t.utility}")
            logger.debug(f"    Location:  ({loc.latitude:.4f}, {loc.longitude:.4f})")
            logger.debug(f"    Action:    {action_str}")

    # --- Step 3: Split tasks ---
    partial_tasks, full_tasks, split_meta = split_tasks(
        all_tasks, PARTIAL_RATIO, observation=observation,
    )

    if verbose:
        partial_ids = set()
        for t in partial_tasks:
            tid = extract_target_id_from_action(t.steps[0].action)
            if tid:
                partial_ids.add(tid)

        full_ids = set()
        for t in full_tasks:
            tid = extract_target_id_from_action(t.steps[0].action)
            if tid:
                full_ids.add(tid)

        hidden_ids = full_ids - partial_ids

        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("TASK SPLIT")
        logger.debug("=" * 60)
        logger.debug(f"  Partial tasks ({len(partial_tasks)}):")
        for i, t in enumerate(partial_tasks):
            tid = extract_target_id_from_action(t.steps[0].action) or "?"
            logger.debug(f"    [{i}] target={tid}, utility={t.utility}")
        logger.debug(f"  Full tasks ({len(full_tasks)}):")
        for i, t in enumerate(full_tasks):
            tid = extract_target_id_from_action(t.steps[0].action) or "?"
            marker = " *** HIDDEN ***" if tid in hidden_ids else ""
            logger.debug(f"    [{i}] target={tid}, utility={t.utility}{marker}")
        logger.debug(f"  Hidden targets: {hidden_ids}")

    # --- Step 4: Solve MATCH-AOU twice ---
    if verbose:
        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("MATCH-AOU SOLUTIONS")
        logger.debug("=" * 60)

    logger.debug("Solving MATCH-AOU (partial)...")
    partial_solution, partial_tasks_filtered, _ = solve_match_aou(
        attacking_agents, partial_tasks, SOLVER_NAME
    )

    if verbose:
        logger.debug("  --- Partial Solution ---")
        _log_solution_details(partial_solution, partial_tasks_filtered)

    logger.debug("Solving MATCH-AOU (full / oracle)...")
    full_solution, full_tasks_filtered, _ = solve_match_aou(
        attacking_agents, full_tasks, SOLVER_NAME
    )

    if verbose:
        logger.debug("  --- Full (Oracle) Solution ---")
        _log_solution_details(full_solution, full_tasks_filtered)

        # Show the diff
        partial_all_targets = _extract_all_target_ids(partial_solution, partial_tasks_filtered)
        full_all_targets = _extract_all_target_ids(full_solution, full_tasks_filtered)
        new_in_full = full_all_targets - partial_all_targets
        logger.debug(f"  --- Comparison ---")
        logger.debug(f"  Targets in partial: {partial_all_targets}")
        logger.debug(f"  Targets in full:    {full_all_targets}")
        logger.debug(f"  NEW in full (what RL should learn to attack): {new_in_full}")

    if not partial_solution:
        logger.warning("Partial solution empty, skipping episode")
        m = _empty_metrics()
        m["split_meta"] = split_meta
        m["empty_partial"] = True
        return m

    # --- Step 5: Pre-launch all aircraft from airbases ---
    # The recording filename derives from `current_scenario.name` set
    # before `start_recording()`. For flagged-episode replays the main
    # loop passes a tagged record_name like "ep0042_flagged_TIMEOUT_rl".
    game.current_scenario.name = record_name or f"ep{episode_num + 1:03d}_rl"
    rec_step = game.record_step if record else (lambda *a, **kw: None)
    if record:
        game.start_recording()
    rec_step()

    # Buffer: run a few empty ticks so recording shows aircraft at base before launch
    PRE_LAUNCH_BUFFER = 5
    for _ in range(PRE_LAUNCH_BUFFER):
        observation, _, _, _, _ = env.step("")
        rec_step(force=True)

    if verbose:
        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("PRE-LAUNCH")
        logger.debug("=" * 60)

    # Name lookup for human-readable log lines (UUID → "B-2 Spirit #698").
    # Built once here, after airbases are populated but before launch, so
    # both blue aircraft (still in airbase.aircraft) and red facilities/
    # airbases are captured in a single sweep.
    name_lookup = _build_name_lookup(observation)

    for airbase in getattr(observation, "airbases", []) or []:
        ab_side = _normalize_side_color(getattr(airbase, "side_color", ""))
        if ab_side != ATTACKING_SIDE_COLOR:
            continue
        ab_id = str(airbase.id)
        aircraft_in_base = list(getattr(airbase, "aircraft", []) or [])
        for ac in aircraft_in_base:
            launch_cmd = f"launch_aircraft_from_airbase('{ab_id}')"
            ac_name = getattr(ac, 'name', ac.id)
            logger.debug(
                f"  LAUNCH: {ac_name} (id={str(ac.id)[:8]}..) "
                f"from airbase {_id_label(ab_id, name_lookup)}"
            )
            observation, _, _, _, _ = env.step(launch_cmd)
            rec_step()

    for _ in range(10):
        observation, _, _, _, _ = env.step("")
        rec_step()

    airborne = [getattr(ac, 'name', ac.id) for ac in getattr(observation, 'aircraft', [])]
    logger.debug(f"  Airborne after launch: {len(airborne)} aircraft — {airborne}")

    # --- Step 6: Setup executor with partial plan ---
    executor = BladeExecutorMinimal(
        tasks=partial_tasks_filtered,
        solution=partial_solution,
        agents=attacking_agents,
        add_return_to_base=True,
        arrival_threshold_km=50.0,
    )

    if verbose:
        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("EXECUTOR QUEUE")
        logger.debug("=" * 60)
        for aid, q in executor.queue.items():
            logger.debug(f"  Agent {aid}: {len(q)} assignments")
            for task_idx, step_idx, level in q:
                target_id = "?"
                if 0 <= task_idx < len(partial_tasks_filtered):
                    action = getattr(partial_tasks_filtered[task_idx].steps[step_idx], "action", "")
                    target_id = extract_target_id_from_action(action) or "?"
                logger.debug(f"    task={task_idx}, step={step_idx}, level={level}, target={target_id}")

    # Pre-compute oracle data
    full_agent_targets = extract_target_ids_from_solution(
        full_solution, full_tasks_filtered
    )
    partial_target_ids = set()
    for assignments in partial_solution.values():
        for task_idx, step_idx, _level in assignments:
            if 0 <= task_idx < len(partial_tasks_filtered):
                action = getattr(
                    partial_tasks_filtered[task_idx].steps[step_idx], "action", ""
                ) or ""
                tid = extract_target_id_from_action(action)
                if tid:
                    partial_target_ids.add(tid)

    if verbose:
        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("ORACLE SETUP")
        logger.debug("=" * 60)
        logger.debug(f"  Partial target IDs (known): {partial_target_ids}")
        for aid, targets in full_agent_targets.items():
            logger.debug(f"  Full targets for {aid}: {targets}")
            new = targets - partial_target_ids
            if new:
                logger.debug(f"    → Agent should learn to attack: {new}")
        logger.debug("")
        logger.debug("=" * 60)
        logger.debug("SIMULATION START")
        logger.debug("=" * 60)

    # --- Step 7: Simulation loop ---
    episode_reward = 0.0
    decisions = 0
    matches = 0
    fuel_eff_count = 0  # real-fuel hits actually applied (vs. fuel_dmg.events scheduled)
    returned_agents: Set[str] = set()
    n_agents = len(attacking_agents)
    obs_dim = obs_config.top_k * 6 + 6 + 6  # 30

    # --- Utility-based reward setup ---
    # Build target_id → utility mapping from ALL tasks (for reward computation)
    target_utility_map = build_target_utility_map(all_tasks, extract_target_id_from_action)
    max_utility = max((t.utility for t in all_tasks), default=1.0)
    oracle_total_utility = compute_oracle_total_utility(
        full_solution, full_tasks_filtered, extract_target_id_from_action,
    )

    # Hidden targets = in full plan but NOT in partial plan (RL's job to discover)
    partial_agent_targets = extract_target_ids_from_solution(
        partial_solution, partial_tasks_filtered
    )
    all_partial_target_ids: Set[str] = set()
    for tids in partial_agent_targets.values():
        all_partial_target_ids.update(tids)

    hidden_targets_per_agent: Dict[str, Set[str]] = {
        aid: targets - all_partial_target_ids
        for aid, targets in full_agent_targets.items()
    }

    all_hidden_target_ids: Set[str] = set()
    for tids in hidden_targets_per_agent.values():
        all_hidden_target_ids.update(tids)

    hidden_target_utility = sum(
        target_utility_map.get(tid, 0.0) for tid in all_hidden_target_ids
    )
    # Track which targets RL agents successfully attacked (for episode-end reward)
    rl_attacked_target_ids: Set[str] = set()
    blade_attacks: Dict[str, str] = {}  # target_id → agent_id of REAL BLADE hits

    if verbose:
        logger.debug(f"  Utility map: {target_utility_map}")
        logger.debug(f"  Max utility: {max_utility}")
        logger.debug(f"  Oracle total utility: {oracle_total_utility}")

    # --- Fuel damage setup ---
    fuel_dmg = FuelDamageManager(FuelDamageConfig(enabled=fuel_damage_enabled))
    fuel_dmg.plan_episode(
        agent_ids=[a.id for a in attacking_agents],
        max_ticks=max_ticks,
        seed=episode_num,
    )

    if n_agents > MAX_AGENTS:
        logger.warning(
            f"Scenario has {n_agents} agents but MAX_AGENTS={MAX_AGENTS}. "
            f"Only the first {MAX_AGENTS} will be used for the critic."
        )

    # Track which (agent, target) discoveries have already triggered RL.
    # Without this, the same hidden target would re-trigger every tick
    # because partial_target_ids never changes during the episode.
    processed_discoveries: Dict[str, Set[str]] = {
        str(a.id): set() for a in attacking_agents
    }

    for tick in range(max_ticks):
        # Executor decides action for this tick (partial plan)
        try:
            executor_action = executor.next_action(observation, fallback_tick=tick) or ""
        except ValueError as e:
            logger.debug(f"Tick {tick}: Executor error (skipping): {e}")
            executor_action = ""

        # Check for fuel damage activation — capture newly damaged agents
        newly_damaged = fuel_dmg.check_and_activate(tick)

        # Apply the one-time real-fuel hit BEFORE building any observation
        # this tick, so the obs vector naturally reads the reduced fuel.
        # apply_real_damage returns None when the aircraft is no longer in
        # observation.aircraft (already landed/crashed) — that's "scheduled
        # but not effective", so only count non-None returns.
        for aid in newly_damaged:
            if aid not in returned_agents:
                if fuel_dmg.apply_real_damage(observation, aid) is not None:
                    fuel_eff_count += 1

        rl_override_action = ""

        # === Event detection ===
        # Two triggers: (1) discovery (checked every DISCOVERY_SCAN_INTERVAL),
        #               (2) fuel damage (checked every tick, triggers immediately).
        triggered_agents: Dict[str, str] = {}  # agent_id → trigger reason

        # Fuel damage: immediate trigger (no waiting for scan tick)
        for aid in newly_damaged:
            if aid not in returned_agents:
                triggered_agents[aid] = "fuel_damage"

        # Discovery scan: only every N ticks (avoid building obs every tick)
        is_scan_tick = (tick > 0 and tick % DISCOVERY_SCAN_INTERVAL == 0)

        # Build observations when we have a reason:
        # - Scan tick → check for discoveries
        # - Fuel damage → need obs for RL decision + global obs for critic
        needs_obs = is_scan_tick or bool(triggered_agents)

        agent_obs_map: Dict[str, object] = {}
        if needs_obs:
            for agent_obj in attacking_agents:
                agent_id = agent_obj.id
                if agent_id in returned_agents:
                    continue
                try:
                    agent_plan = partial_solution.get(agent_id, [])
                    obs = build_observation_vector(
                        scenario=observation,
                        agent_id=agent_id,
                        current_plan=agent_plan,
                        current_time=tick,
                        config=obs_config,
                        tasks=partial_tasks_filtered,
                        solution=partial_solution,
                    )
                    agent_obs_map[agent_id] = obs
                except (ValueError, Exception) as e:
                    logger.debug(f"Tick {tick}: Can't observe {agent_id}: {e}")

            # On scan ticks, check for NEW (unprocessed) discoveries
            if is_scan_tick:
                for agent_id, obs in agent_obs_map.items():
                    for target in obs.targets:
                        if (target.exists
                                and not target.is_in_plan
                                and target.id not in partial_target_ids
                                and target.id not in processed_discoveries[agent_id]):
                            processed_discoveries[agent_id].add(target.id)
                            triggered_agents.setdefault(agent_id, "discovery")
                            logger.debug(
                                f"  Tick {tick:5d} DISCOVERY: "
                                f"agent {_id_label(agent_id, name_lookup)} "
                                f"sees target {_id_label(target.id, name_lookup)}"
                            )

        # === RL decisions (ONLY when triggered) ===
        if triggered_agents:
            # Construct global observation (padded to MAX_AGENTS)
            global_obs_parts = []
            for i in range(MAX_AGENTS):
                if i < len(attacking_agents):
                    aid = attacking_agents[i].id
                    if aid in agent_obs_map:
                        global_obs_parts.append(agent_obs_map[aid].vector)
                    else:
                        global_obs_parts.append(np.zeros(obs_dim, dtype=np.float32))
                else:
                    global_obs_parts.append(np.zeros(obs_dim, dtype=np.float32))
            global_obs = np.concatenate(global_obs_parts)

            # Per-agent decisions (only for triggered agents)
            for agent_obj in attacking_agents:
                agent_id = agent_obj.id
                if agent_id not in triggered_agents:
                    continue
                if agent_id not in agent_obs_map:
                    continue

                trigger = triggered_agents[agent_id]
                obs = agent_obs_map[agent_id]
                local_obs = obs.vector

                action_mask = get_simple_action_mask(obs)

                # PPO: actor samples action, critic estimates value
                rl_action, log_prob, value = trainer.get_action(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    action_mask=action_mask,
                )

                oracle_action = get_oracle_action(
                    obs, agent_id, full_agent_targets
                )

                # Utilities are kept for the tracker / log fields only —
                # they no longer feed reward (pure episode-end utility now).
                rl_utility = get_action_utility(rl_action, obs, target_utility_map)
                oracle_utility = get_action_utility(oracle_action, obs, target_utility_map)

                is_valid = bool(action_mask[rl_action]) if rl_action < len(action_mask) else False
                reward = 0.0 if is_valid else trainer.config.reward_config.invalid_action_penalty

                episode_reward += reward
                decisions += 1
                if 0 <= rl_action <= 4:
                    action_hist[rl_action] += 1
                is_match = (rl_action == oracle_action)
                if is_match:
                    matches += 1

                # Track attacked targets for episode-end utility
                if 1 <= rl_action <= 3:
                    slot_idx = rl_action - 1
                    if slot_idx < len(obs.targets) and obs.targets[slot_idx].exists:
                        rl_attacked_target_ids.add(obs.targets[slot_idx].id)

                trainer.reward_tracker.add_step(
                    reward=reward,
                    is_match=is_match,
                    rl_utility=rl_utility,
                    oracle_utility=oracle_utility,
                )

                # Log every event-driven RL decision (these are rare and
                # meaningful — but rare per *tick*, not per *episode*; over
                # 1000 episodes the volume swamps the console).
                action_names = {0: "NOOP", 1: "ATTACK_0", 2: "ATTACK_1", 3: "ATTACK_2", 4: "RTB"}
                logger.debug(
                    f"  Tick {tick:5d} RL DECISION: {_id_label(agent_id, name_lookup)} | "
                    f"trigger={trigger} | "
                    f"RL={action_names.get(rl_action, '?')} "
                    f"Oracle={action_names.get(oracle_action, '?')} "
                    f"Match={'✓' if rl_action == oracle_action else '✗'} "
                    f"Reward={reward:+.2f} "
                    f"(rl_u={rl_utility:.0f}, oracle_u={oracle_utility:.0f})"
                )

                # RL override: any trigger can produce an override action
                # (discovery → attack new target, fuel damage → RTB, etc.)
                if rl_action != 0:
                    try:
                        rl_override_action = plan_edit_to_blade_action(
                            action_token=rl_action,
                            observation_output=obs,
                            scenario=observation,
                            agent_id=agent_id,
                        )
                    except (ValueError, Exception) as e:
                        logger.debug(f"  RL action {rl_action} invalid for {agent_id}: {e}")
                        rl_override_action = ""

                # Store in rollout buffer (PPO collects, trains AFTER episode).
                # Skipped on replay_only re-runs so flagged-episode recordings
                # don't double-update the policy on the same scenario.
                done = (tick >= max_ticks - 1)
                if not replay_only:
                    trainer.buffer.store(
                        local_obs=local_obs,
                        global_obs=global_obs,
                        action=rl_action,
                        log_prob=log_prob,
                        reward=reward,
                        value=value,
                        done=done,
                        action_mask=action_mask.astype(np.float32),
                        oracle_action=oracle_action,
                    )

        # Decide what action to send to BLADE
        final_action = rl_override_action if rl_override_action else executor_action

        # Log BLADE actions with parsed, human-readable format
        if final_action:
            source = "RL" if rl_override_action else "EXEC"
            _log_blade_action(tick, final_action, source, name_lookup)

            # Capture real BLADE attack execution for utility crediting
            m = _RE_ATTACK.search(final_action)
            if m:
                attacker_id, attacked_target_id = m.group(1), m.group(2)
                blade_attacks[attacked_target_id] = attacker_id

        observation, _reward, terminated, truncated, info = env.step(final_action)
        rec_step()

        # Check which agents have returned to base (no longer airborne)
        airborne_ids = {str(getattr(ac, "id", "")) for ac in getattr(observation, "aircraft", []) or []}
        for agent_obj in attacking_agents:
            aid = str(agent_obj.id)
            if aid not in returned_agents and aid not in airborne_ids:
                returned_agents.add(aid)
                logger.debug(
                    f"  Tick {tick:5d} RTB:     agent {_id_label(aid, name_lookup)} landed"
                )

        # --- Detailed logging near end of episode ---
        # Log every tick in the last 100 before max_ticks, or when
        # terminated/truncated is about to fire, so we can see exactly
        # what's happening when the episode cuts off.
        ticks_remaining = max_ticks - tick
        if ticks_remaining <= 100 and ticks_remaining % 10 == 0:
            all_aircraft = getattr(observation, "aircraft", []) or []
            logger.debug(
                f"  ── Tick {tick:5d} [END-ZONE] ── "
                f"remaining={ticks_remaining} | "
                f"airborne={len(all_aircraft)} | "
                f"returned={len(returned_agents)}/{n_agents} | "
                f"terminated={terminated} | truncated={truncated}"
            )
            for ac in all_aircraft:
                ac_id = str(getattr(ac, "id", ""))[:8]
                ac_name = getattr(ac, "name", ac_id)
                fuel = getattr(ac, "current_fuel", 0)
                rtb = getattr(ac, "rtb", False)
                lat = getattr(ac, "latitude", 0)
                lon = getattr(ac, "longitude", 0)
                route_len = len(getattr(ac, "route", []) or [])
                logger.debug(
                    f"    {ac_name} (id={ac_id}..): "
                    f"pos=({lat:.2f},{lon:.2f}) fuel={fuel:.0f} "
                    f"rtb={rtb} route_pts={route_len}"
                )

        # End episode when all agents have returned to base
        if tick > 100 and len(returned_agents) == len(attacking_agents):
            logger.debug(f"  All agents returned to base at tick {tick} — ending episode")
            all_rtb_flag = True
            break

        if terminated or truncated:
            logger.debug(
                f"  Episode ended at tick {tick}: "
                f"terminated={terminated}, truncated={truncated} "
                f"(env step count ≈ {tick + PRE_LAUNCH_BUFFER + n_agents + 10})"
            )
            if truncated:
                timeout_flag = True
            break

        # Periodic progress summary
        if tick > 0 and tick % PROGRESS_LOG_INTERVAL == 0:
            _log_progress(
                tick, n_agents, returned_agents, decisions,
                episode_reward, rl_attacked_target_ids, len(all_tasks),
            )

    # === End of episode: pure utility-based reward ===
    # Achieved utility credits the TEAM for every target actually destroyed
    # by BLADE this episode (regardless of partial/hidden split). Aircraft
    # losses fold into the same formula via compute_episode_reward — there
    # is no longer a separate crash-penalty term.
    achieved_utility = sum(
        target_utility_map.get(target_id, 0.0)
        for target_id in blade_attacks.keys()
    )

    crashed_agent_ids = identify_crashed_agents(
        observation, [a.id for a in attacking_agents],
    )
    if crashed_agent_ids:
        logger.debug(f"  Crashed agents: {sorted(crashed_agent_ids)}")

    max_target_utility = max(target_utility_map.values()) if target_utility_map else 0.0

    ep_reward = compute_episode_reward(
        achieved_utility=achieved_utility,
        max_total_utility=oracle_total_utility,
        lost_aircraft_count=len(crashed_agent_ids),
        max_target_utility=max_target_utility,
        config=trainer.config.reward_config,
    )

    # Add episode reward to the LAST transition in buffer.
    # GAE propagates this backward through advantages.
    if trainer.buffer.size > 0:
        trainer.buffer.rewards[trainer.buffer.size - 1] += ep_reward
        episode_reward += ep_reward

    trainer.reward_tracker.set_episode_utilities(achieved_utility, oracle_total_utility)
    trainer.reward_tracker.set_crashes(len(crashed_agent_ids))

    logger.debug(
        f"  Episode utility: achieved={achieved_utility:.0f} / "
        f"total={oracle_total_utility:.0f} "
        f"(ratio={achieved_utility / max(oracle_total_utility, 1):.2f})  "
        f"hidden_achieved={sum(target_utility_map.get(tid, 0.0) for tid, aid in blade_attacks.items() if aid in hidden_targets_per_agent and tid in hidden_targets_per_agent[aid]):.0f} / {hidden_target_utility:.0f}"
    )
    logger.debug(
        f"  Reward: ep_reward={ep_reward:+.3f}  crashes={len(crashed_agent_ids)}"
    )

    # === PPO update ===
    # Compute GAE advantages using collected trajectory.
    # Skipped on replay_only re-runs (no buffer entries → nothing to update).
    last_value = 0.0  # Terminal state → value = 0
    no_ppo_flag = False
    if replay_only:
        update_metrics = {}
    elif trainer.buffer.size > 0:
        trainer.buffer.compute_returns_and_advantages(last_value)
        update_metrics = trainer.update()
        logger.debug(
            f"  PPO update: policy_loss={update_metrics.get('policy_loss', 0):.4f}, "
            f"value_loss={update_metrics.get('value_loss', 0):.4f}, "
            f"entropy={update_metrics.get('entropy', 0):.4f}, "
            f"clip_frac={update_metrics.get('clip_fraction', 0):.3f}"
        )
    else:
        update_metrics = {}
        logger.warning("  No transitions collected, skipping PPO update")
        no_ppo_flag = True

    if not replay_only:
        trainer.episode_count += 1
        trainer.buffer.reset()
    else:
        # Drop any transitions we may have inadvertently appended; we never
        # called .store() under replay_only, so this is just a safety reset.
        trainer.buffer.reset()

    if record:
        try:
            game.export_recording()
            export_label = record_name or f"ep{episode_num + 1:03d}_rl"
            logger.debug(f"  Recording exported: {export_label}")
        except Exception as e:
            logger.warning(f"  Failed to export recording: {e}")

    accuracy = matches / max(decisions, 1)

    # Compute partial-vs-full oracle utilities for the summary line.
    partial_oracle_utility = compute_oracle_total_utility(
        partial_solution, partial_tasks_filtered, extract_target_id_from_action,
    )

    # Apply debug flag injection (test-only path used in smoke tests to
    # verify !TIMEOUT and !L2-fallback prefixes render correctly).
    forced = debug_force_flags or set()
    if "timeout" in forced:
        timeout_flag = True
    if "l2-fallback" in forced and split_meta.get("outcome") not in ("warn-fallback", "exhaust"):
        split_meta = dict(split_meta)
        split_meta["outcome"] = "warn-fallback"

    fuel_planned = len(fuel_dmg.events)

    return {
        "episode_reward": episode_reward,
        "decisions": decisions,
        "accuracy": accuracy,
        "matches": matches,
        "ticks": tick + 1,
        "n_agents": n_agents,
        "n_tasks": len(all_tasks),
        "policy_loss": update_metrics.get("policy_loss", 0.0),
        "value_loss": update_metrics.get("value_loss", 0.0),
        "entropy": update_metrics.get("entropy", 0.0),
        "achieved_utility": achieved_utility,
        "oracle_utility": oracle_total_utility,
        "max_target_utility": max_target_utility,
        "partial_oracle_utility": partial_oracle_utility,
        "utility_ratio": achieved_utility / max(oracle_total_utility, 1),
        "crashes": len(crashed_agent_ids),
        "targets_hit_total": len(blade_attacks),
        "fuel_damage_planned": fuel_planned,
        "fuel_damage_effective": fuel_eff_count,
        "action_hist": action_hist,
        "all_rtb": all_rtb_flag,
        "timeout": timeout_flag,
        "no_ppo": no_ppo_flag,
        "split_meta": split_meta,
        "targets_attacked": sum(
            1 for tid, aid in blade_attacks.items()
            if aid in hidden_targets_per_agent
            and tid in hidden_targets_per_agent[aid]
        ),
        "n_hidden": len(all_hidden_target_ids),
    }


# =============================================================================
# Logging Helpers
# =============================================================================

def identify_crashed_agents(observation, agent_ids: List[str]) -> Set[str]:
    """
    Identify which of our agents crashed (removed by BLADE due to fuel-zero etc).

    An agent is crashed iff at episode end it is neither airborne
    (in `observation.aircraft`) nor at any airbase
    (`airbase.aircraft` for some airbase in `observation.airbases`).
    BLADE moves landed aircraft into the airbase inventory and removes
    fuel-zero aircraft entirely via `remove_aircraft`, so absence from
    both lists means the aircraft was destroyed.
    """
    airborne_ids = {
        str(getattr(ac, "id", ""))
        for ac in getattr(observation, "aircraft", []) or []
    }
    in_airbase_ids: Set[str] = set()
    for ab in getattr(observation, "airbases", []) or []:
        for ac in getattr(ab, "aircraft", []) or []:
            in_airbase_ids.add(str(getattr(ac, "id", "")))

    crashed: Set[str] = set()
    for aid in agent_ids:
        sid = str(aid)
        if sid not in airborne_ids and sid not in in_airbase_ids:
            crashed.add(sid)
    return crashed


def _empty_metrics() -> Dict:
    """Return empty metrics dict for skipped episodes.

    Includes all keys the per-episode summary formatter expects so a
    skipped/crashed episode still renders cleanly (with `?`-padded
    fields rather than KeyErrors on top of the original failure).
    """
    return {
        "episode_reward": 0.0, "decisions": 0, "accuracy": 0.0,
        "matches": 0, "ticks": 0, "n_agents": 0, "n_tasks": 0,
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "achieved_utility": 0.0, "oracle_utility": 0.0,
        "max_target_utility": 0.0,
        "partial_oracle_utility": 0.0, "utility_ratio": 0.0,
        "fuel_damage_planned": 0, "fuel_damage_effective": 0,
        "action_hist": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "all_rtb": False, "timeout": False, "no_ppo": True,
        "split_meta": {"outcome": "-", "partial": 0, "full": 0},
        "targets_attacked": 0,
        "targets_hit_total": 0,
        "n_hidden": 0,
        "crashes": 0,
    }


def _log_solution_details(solution: Dict, tasks: List[Task]):
    """Pretty-print a MATCH-AOU solution at DEBUG level (per-episode files)."""
    total = sum(len(v) for v in solution.values())
    logger.debug(f"  Total assignments: {total}")
    for agent_id, assignments in solution.items():
        logger.debug(f"  Agent {agent_id}:")
        for task_idx, step_idx, level in assignments:
            target_id = "?"
            action = ""
            if 0 <= task_idx < len(tasks):
                action = getattr(tasks[task_idx].steps[step_idx], "action", "") or ""
                target_id = extract_target_id_from_action(action) or "?"
            logger.debug(
                f"    task={task_idx} step={step_idx} level={level} → target={target_id}"
            )


def _extract_all_target_ids(solution: Dict, tasks: List[Task]) -> Set[str]:
    """Get all unique target IDs from a solution."""
    ids = set()
    for assignments in solution.values():
        for task_idx, step_idx, _level in assignments:
            if 0 <= task_idx < len(tasks):
                action = getattr(tasks[task_idx].steps[step_idx], "action", "") or ""
                tid = extract_target_id_from_action(action)
                if tid:
                    ids.add(tid)
    return ids


# =============================================================================
# 7b. Compact summary line / run summary builder
# =============================================================================

# Severity ordering used to pick the leading flag prefix when an episode
# triggers multiple categories (e.g. timeout + L2-fallback). Ordered most
# severe first; the leading prefix is the highest-severity flag, with any
# remaining flags appended comma-separated.
_FLAG_SEVERITY = ["CRASH", "ANOMALY", "TIMEOUT", "L2-exhaust", "L2-fallback", "noPPO"]


def _format_episode_summary(
    episode_num: int,
    metrics: Dict,
    gen_stats: Optional[Dict],
    is_validation: bool,
    audit: Optional[Dict],
    fuel_damage_enabled: bool,
) -> Tuple[List[str], List[str]]:
    """Build the compact 2-line per-episode summary.

    Returns:
        (lines, flags) where:
          lines is the list of lines to emit at INFO level (usually 2;
            on warn-fallback / exhaust / etc. flags are folded into the
            line prefix, not extra lines).
          flags is the list of !FLAG categories raised for this episode,
            ordered by severity. The main loop forwards this to
            RunSummaryBuilder for the end-of-run report.

    NOTE: u%% denominator = hidden_target_utility (sum of utilities of
    targets in the full plan but NOT in the partial plan). Matches the
    episode-end reward in train_episode, where compute_episode_reward is
    fed hidden_target_utility as oracle_total_utility — so display and
    learning signal are on the same scale (perfect hidden play → u=100%,
    r=+5.00). crash_penalty still uses the full-plan utility share.
    """
    flags: List[str] = []

    # Detect flags from metrics/audit
    if metrics.get("timeout"):
        flags.append("TIMEOUT")
    split_meta = metrics.get("split_meta") or {}
    outcome = split_meta.get("outcome", "")
    if outcome == "exhaust":
        flags.append("L2-exhaust")
    elif outcome == "warn-fallback":
        flags.append("L2-fallback")
    if metrics.get("no_ppo") or metrics.get("empty_partial"):
        if "noPPO" not in flags:
            flags.append("noPPO")
    if audit:
        if audit.get("oracle_violations", 0) > 0 or audit.get("unreachable_hit", 0) > 0:
            flags.append("ANOMALY")

    # Sort by severity
    flags = sorted(set(flags), key=lambda f: _FLAG_SEVERITY.index(f) if f in _FLAG_SEVERITY else 999)

    if flags:
        prefix = "!" + ",".join(flags) + " "
    else:
        prefix = ""

    val_tag = "[VAL] " if is_validation else "      "
    ep_tag = f"ep{episode_num + 1:04d}"

    # Scenario stats
    n_ag = metrics.get("n_agents", "?")
    n_tg = metrics.get("n_tasks", "?")
    if gen_stats:
        n_easy = gen_stats.get("n_easy", 0)
        n_stretch = gen_stats.get("n_stretch", 0)
        zone_str = f"[{n_easy}e+{n_stretch}s]"
        e_r = gen_stats.get("easy_relocated", 0)
        e_t = gen_stats.get("easy_total", 0)
        e_i = gen_stats.get("easy_isolated", 0)
        s_r = gen_stats.get("stretch_relocated", 0)
        s_t = gen_stats.get("stretch_total", 0)
        s_i = gen_stats.get("stretch_isolated", 0)
        l1_str = f"L1:e={e_r}/{e_t}+{e_i}iso s={s_r}/{s_t}+{s_i}iso"
    else:
        zone_str = "[-]"
        l1_str = "L1:-"

    # L2 outcome string
    if outcome == "clean":
        l2_str = "L2:clean".ljust(16)
    elif outcome == "resampled":
        l2_str = f"L2:resamp@{split_meta.get('attempt', '?')}".ljust(16)
    elif outcome == "warn-fallback":
        l2_str = "L2:warn-fallback".ljust(16)
    elif outcome == "exhaust":
        l2_str = "L2:exhaust".ljust(16)
    elif outcome == "no-chain":
        l2_str = "L2:no-chain".ljust(16)
    else:
        l2_str = f"L2:{outcome or '-'}".ljust(16)

    partial_count = split_meta.get("partial", "?")
    full_count = split_meta.get("full", "?")
    split_str = f"split={partial_count}/{full_count}"

    u_partial = metrics.get("partial_oracle_utility", 0.0)
    u_full = metrics.get("oracle_utility", 0.0)
    ou_str = f"ou={u_partial:.0f}/{u_full:.0f}"

    line1_extra = ""
    if audit and "ANOMALY" in flags:
        line1_extra = f"  audit_violations={audit.get('oracle_violations', 0) + audit.get('unreachable_hit', 0)}"

    line1 = (
        f"{prefix}{ep_tag} {val_tag} ag={n_ag} tg={n_tg}{zone_str}  "
        f"{l1_str}  {l2_str} {split_str}  {ou_str}{line1_extra}"
    )

    # Line 2 — execution results
    decisions = metrics.get("decisions", 0)
    matches = metrics.get("matches", 0)
    hist = metrics.get("action_hist") or {}
    n_attack = (hist.get(1, 0) + hist.get(2, 0) + hist.get(3, 0))
    n_rtb = hist.get(4, 0)
    n_noop = hist.get(0, 0)
    rl_str = f"RL={decisions}d[A{n_attack} R{n_rtb} N{n_noop}] m={matches}/{decisions}"

    targets_attacked = metrics.get("targets_attacked", 0)
    n_hidden = metrics.get("n_hidden", 0)
    hit_str = f"hit={targets_attacked}/{n_hidden}"
    rtb_str = f"RTB={'Y' if metrics.get('all_rtb') else 'N'}"
    crashes_str = f"crashes={metrics.get('crashes', 0)}"

    fd_str = ""
    if fuel_damage_enabled:
        fd_str = f"fd={metrics.get('fuel_damage_effective', 0)}eff/{metrics.get('fuel_damage_planned', 0)}pln  "

    ticks = metrics.get("ticks", 0)
    reward = metrics.get("episode_reward", 0.0)
    u_pct = metrics.get("utility_ratio", 0.0) * 100

    line2 = (
        f"{prefix}{ep_tag} {val_tag} {rl_str}  {hit_str}  {rtb_str}  {crashes_str}  "
        f"{fd_str}t={ticks:5d}  r={reward:+6.2f}  u={u_pct:3.0f}%"
    )

    return [line1, line2], flags


def _aggregate_window(metrics_list: List[Dict]) -> Dict:
    """Aggregate per-episode metrics over a rolling window.

    Used by the learning-progress block to render trend signals during
    a long training run. Returns averages / sums / accuracy / action
    distribution / PPO-loss averages / window flag counts.
    """
    if not metrics_list:
        return {}
    n = len(metrics_list)
    total_dec = sum(m.get("decisions", 0) for m in metrics_list)
    total_match = sum(m.get("matches", 0) for m in metrics_list)

    # Action histogram counts (NOOP=0, ATTACK 0/1/2=1-3, RTB=4)
    a_sum = r_sum = noop_sum = 0
    for m in metrics_list:
        h = m.get("action_hist") or {}
        a_sum += h.get(1, 0) + h.get(2, 0) + h.get(3, 0)
        r_sum += h.get(4, 0)
        noop_sum += h.get(0, 0)
    total_actions = a_sum + r_sum + noop_sum

    # PPO losses averaged over episodes that ran an update (skip noPPO)
    loss_eps = [m for m in metrics_list if not m.get("no_ppo")]
    n_loss = max(len(loss_eps), 1)
    pl = sum(m.get("policy_loss", 0.0) for m in loss_eps) / n_loss
    vl = sum(m.get("value_loss", 0.0) for m in loss_eps) / n_loss
    ent = sum(m.get("entropy", 0.0) for m in loss_eps) / n_loss

    # Window flag tally — read from per-episode flag list captured by the
    # main loop and stashed onto each metrics dict (key: "_flags").
    flag_counts: Dict[str, int] = {}
    for m in metrics_list:
        for f in m.get("_flags", []):
            flag_counts[f] = flag_counts.get(f, 0) + 1

    return {
        "n": n,
        "reward_mean": sum(m.get("episode_reward", 0.0) for m in metrics_list) / n,
        "utility_mean_pct": sum(m.get("utility_ratio", 0.0) for m in metrics_list) / n * 100.0,
        "accuracy_pct": (total_match / total_dec * 100.0) if total_dec else 0.0,
        "decisions_per_ep": total_dec / n,
        "ticks_mean": sum(m.get("ticks", 0) for m in metrics_list) / n,
        "action_attack_pct": (a_sum / total_actions * 100.0) if total_actions else 0.0,
        "action_rtb_pct": (r_sum / total_actions * 100.0) if total_actions else 0.0,
        "action_noop_pct": (noop_sum / total_actions * 100.0) if total_actions else 0.0,
        "policy_loss": pl, "value_loss": vl, "entropy": ent,
        "flag_counts": flag_counts,
    }


def _format_progress_block(
    ep_num: int, current: Dict, previous: Optional[Dict],
    checkpoint_saved: bool, window: int,
) -> List[str]:
    """Build the multi-line learning-progress block.

    Returns a list of lines; the caller emits each at INFO level. The
    block compares the current window's aggregates against the previous
    one and appends an ASCII trend arrow (↑/↓/·) to highlight whether
    the RL is learning. Uses ASCII-only delta markers so it survives
    PyCharm/SSH/grep cleanly. Layout pads numeric fields to keep the
    blocks visually aligned across many windows.
    """
    arrow = lambda d, eps=0.01: ("↑" if d > eps else ("↓" if d < -eps else "·"))
    # The caller's `fmt` already contains the sign flag (e.g. "+5.2f"),
    # so we don't prepend another `+` here — that would produce the
    # invalid format specifier "++5.2f".
    fmt_delta = lambda d, fmt: (f"Δ {d:{fmt}} " + arrow(d))
    fmt_delta_pct = lambda d, fmt: (f"Δ {d:{fmt}}% " + arrow(d))

    head = (
        f"========== Progress @ ep{ep_num:04d}"
        + (" | checkpoint saved" if checkpoint_saved else "")
        + f" | rolling {window}ep =========="
    )
    tail = "=" * len(head)

    if previous is None:
        # First block — no delta column
        l1 = (
            f"  Reward   : {current['reward_mean']:+6.2f}            "
            f"Utility : {current['utility_mean_pct']:5.1f}%        "
            f"Accuracy: {current['accuracy_pct']:5.1f}%"
        )
        l2 = (
            f"  Ticks/ep : {current['ticks_mean']:6.0f}             "
            f"Actions : A:{current['action_attack_pct']:4.1f}% "
            f"R:{current['action_rtb_pct']:4.1f}% N:{current['action_noop_pct']:4.1f}%   "
            f"Decisions: {current['decisions_per_ep']:.2f}/ep"
        )
    else:
        d_r = current["reward_mean"] - previous["reward_mean"]
        d_u = current["utility_mean_pct"] - previous["utility_mean_pct"]
        d_a = current["accuracy_pct"] - previous["accuracy_pct"]
        d_t = current["ticks_mean"] - previous["ticks_mean"]
        l1 = (
            f"  Reward   : {current['reward_mean']:+6.2f}  {fmt_delta(d_r, '+5.2f')}    "
            f"Utility : {current['utility_mean_pct']:5.1f}%  {fmt_delta_pct(d_u, '+5.1f')}  "
            f"Accuracy: {current['accuracy_pct']:5.1f}%  {fmt_delta_pct(d_a, '+5.1f')}"
        )
        l2 = (
            f"  Ticks/ep : {current['ticks_mean']:6.0f}  {fmt_delta(d_t, '+6.0f')}     "
            f"Actions : A:{current['action_attack_pct']:4.1f}% "
            f"R:{current['action_rtb_pct']:4.1f}% N:{current['action_noop_pct']:4.1f}%   "
            f"Decisions: {current['decisions_per_ep']:.2f}/ep"
        )
    flags_str = (" ".join(f"!{k}={v}" for k, v in sorted(current["flag_counts"].items()))
                 or "(none)")
    l3 = (
        f"  PPO loss : π={current['policy_loss']:+7.4f}  V={current['value_loss']:6.2f}  "
        f"H={current['entropy']:.3f}    Flags(window): {flags_str}"
    )
    return [head, l1, l2, l3, tail]


def _find_recording(recordings_dir: Path, ep_num: int, kind: str = "rl") -> str:
    """Return the recording filename for ep_num (1-based) of given kind.

    BLADE's PlaybackRecorder appends a timestamp suffix to the configured
    name (e.g. "ep0042_rl Recording 081234 - 091045.jsonl"), so we glob
    by prefix. Tries the flagged variant first (3 or 4 digit form), then
    the regular variant. Returns "" when no matching file exists.
    """
    candidates = [
        f"ep{ep_num:04d}_flagged_*_{kind}",  # tagged replay
        f"ep{ep_num:03d}_flagged_*_{kind}",  # tagged replay (legacy 3-digit)
        f"ep{ep_num:04d}_{kind}",
        f"ep{ep_num:03d}_{kind}",
    ]
    for prefix in candidates:
        for f in recordings_dir.glob(f"{prefix}*.jsonl"):
            return f.name
    return ""


def _write_highlights(
    out_path: Path,
    all_metrics: List[Dict],
    flag_episodes: Dict[str, List[int]],
    recordings_dir: Path,
) -> None:
    """Generate highlights.txt — curated 'watch this' index of episodes
    worth opening in Panopticon, organised by training-time window and
    by behavioural pattern (perfect-match, significant-mismatch,
    learning-trend, flagged).

    Filenames are pulled from disk via _find_recording, so flagged-episode
    replays show their tagged filename and the user can copy-paste it
    directly into Panopticon's open dialog.
    """
    n = len(all_metrics)
    lines: List[str] = []

    def episode_to_dict(ep_idx: int) -> Optional[Dict]:
        if 0 <= ep_idx < n:
            d = dict(all_metrics[ep_idx])
            d["_ep"] = ep_idx + 1  # 1-based for display
            return d
        return None

    def fmt_row(d: Dict, kind: str = "rl") -> str:
        ep_num = d["_ep"]
        rec = _find_recording(recordings_dir, ep_num, kind=kind) or "(no recording)"
        match = d.get("matches", 0)
        dec = d.get("decisions", 0)
        rew = d.get("episode_reward", 0.0)
        u = d.get("utility_ratio", 0.0) * 100.0
        return (
            f"  ep{ep_num:04d}  m={match}/{dec}  r={rew:+6.2f}  u={u:5.1f}%   {rec}"
        )

    def windows() -> List[Tuple[str, range]]:
        """Late / mid / early as user requested. For runs <3000 ep, fall
        back to thirds to keep the section non-empty on small smoke tests.
        """
        if n >= 3000:
            return [
                ("Late training (last 1000 eps)", range(n - 1000, n)),
                ("Mid training (eps 2000-3000)", range(2000, 3000)),
                ("Early training (first 1000 eps)", range(0, 1000)),
            ]
        # Small-run fallback — split into thirds (late / mid / early).
        third = max(1, n // 3)
        return [
            ("Late training", range(2 * third, n)),
            ("Mid training", range(third, 2 * third)),
            ("Early training", range(0, third)),
        ]

    win_specs = windows()

    # === Section 1: PERFECT-MATCH episodes ===
    lines.append("=" * 78)
    lines.append("PERFECT-MATCH RL EPISODES (m=N/N with N>=2)")
    lines.append("=" * 78)
    for label, rng in win_specs:
        candidates = [
            episode_to_dict(i) for i in rng
            if (d := episode_to_dict(i)) is not None
            and d.get("decisions", 0) >= 2
            and d.get("matches", 0) == d.get("decisions", 0)
        ]
        candidates.sort(key=lambda d: d.get("episode_reward", 0.0), reverse=True)
        lines.append("")
        lines.append(f"{label}, top 5 by reward:")
        if not candidates:
            lines.append("  (no perfect-match episodes in this window)")
        else:
            for d in candidates[:5]:
                lines.append(fmt_row(d))
    lines.append("")

    # === Section 2: SIGNIFICANT-MISMATCH episodes ===
    lines.append("=" * 78)
    lines.append("SIGNIFICANT-MISMATCH EPISODES (m <= 50% with >=2 decisions)")
    lines.append("=" * 78)
    for label, rng in win_specs:
        candidates = [
            d for i in rng if (d := episode_to_dict(i)) is not None
            and d.get("decisions", 0) >= 2
            and (d.get("matches", 0) / max(d.get("decisions", 1), 1)) <= 0.5
        ]
        candidates.sort(
            key=lambda d: (d.get("decisions", 0) - d.get("matches", 0)),
            reverse=True,
        )
        lines.append("")
        lines.append(f"{label}, top 5 by mismatch count:")
        if not candidates:
            lines.append("  (no significant-mismatch episodes in this window)")
        else:
            for d in candidates[:5]:
                lines.append(fmt_row(d))
    lines.append("")

    # === Section 3: LEARNING-TREND samples ===
    lines.append("=" * 78)
    lines.append("LEARNING TREND SAMPLES (one episode per 1000-ep window,")
    lines.append("                        episode closest to that window's mean reward)")
    lines.append("=" * 78)
    lines.append("")
    bucket = 1000
    for start in range(0, n, bucket):
        chunk = all_metrics[start:start + bucket]
        if not chunk:
            continue
        rewards = [m.get("episode_reward", 0.0) for m in chunk]
        mean_r = sum(rewards) / len(rewards)
        # Find the episode closest to the window mean
        best_idx = min(range(len(chunk)),
                       key=lambda i: abs(rewards[i] - mean_r))
        d = episode_to_dict(start + best_idx)
        ep_lo = start + 1
        ep_hi = start + len(chunk)
        lines.append(f"Window ep{ep_lo:04d}-ep{ep_hi:04d}  (mean reward = {mean_r:+.2f}):")
        if d is not None:
            lines.append(fmt_row(d))
        lines.append("")

    # === Section 4: FLAGGED EPISODE INDEX ===
    lines.append("=" * 78)
    lines.append("FLAGGED EPISODE INDEX (recap from run_summary.txt)")
    lines.append("=" * 78)
    flag_order = ["CRASH", "ANOMALY", "TIMEOUT",
                  "L2-exhaust", "L2-fallback", "noPPO"]
    for f in flag_order:
        eps = flag_episodes.get(f, [])
        lines.append("")
        lines.append(f"!{f}  ({len(eps)} episode{'s' if len(eps) != 1 else ''}):")
        if not eps:
            lines.append("  (none)")
            continue
        for ep_num in eps:
            d = episode_to_dict(ep_num - 1)
            if d is None:
                lines.append(f"  ep{ep_num:04d}  (metrics unavailable)")
                continue
            lines.append(fmt_row(d))
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class RunSummaryBuilder:
    """Accumulate per-episode flag info and write run_summary.txt at end of training.

    Tracks: flag counts, episode index lists per flag, rolling reward /
    utility / RL-accuracy in fixed windows, and cluster alerts (≥3
    flagged episodes within a 5-episode span).
    """

    def __init__(self, total_episodes: int, validate_every: int, window: int = 50):
        self.total_episodes = total_episodes
        self.validate_every = validate_every
        self.window = window
        self.flag_counts: Dict[str, int] = {}
        self.flag_episodes: Dict[str, List[int]] = {}
        self.per_episode: List[Dict] = []   # per-ep summary fields
        self.audits: List[Dict] = []        # validation audit summaries
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def record(self, episode_num: int, metrics: Dict, flags: List[str], audit: Optional[Dict]):
        for f in flags:
            self.flag_counts[f] = self.flag_counts.get(f, 0) + 1
            self.flag_episodes.setdefault(f, []).append(episode_num + 1)
        self.per_episode.append({
            "ep": episode_num + 1,
            "reward": metrics.get("episode_reward", 0.0),
            "utility_ratio": metrics.get("utility_ratio", 0.0),
            "matches": metrics.get("matches", 0),
            "decisions": metrics.get("decisions", 0),
            "flags": list(flags),
        })
        if audit is not None:
            self.audits.append({"ep": episode_num + 1, **audit})

    def _detect_clusters(self) -> List[str]:
        """Find runs of ≥3 flagged episodes within any 5-episode span."""
        flagged = [e["ep"] for e in self.per_episode if e["flags"]]
        out: List[str] = []
        if len(flagged) < 3:
            return out
        # Sliding window over consecutive flagged episodes
        i = 0
        while i < len(flagged):
            j = i
            while j + 1 < len(flagged) and flagged[j + 1] - flagged[i] <= 4:
                j += 1
            if j - i + 1 >= 3:
                window_eps = flagged[i:j + 1]
                # collect flags per ep in this window
                per = []
                tally: Dict[str, int] = {}
                for e in self.per_episode:
                    if e["ep"] in window_eps:
                        for f in e["flags"]:
                            tally[f] = tally.get(f, 0) + 1
                tally_str = ", ".join(f"!{k} ×{v}" for k, v in tally.items())
                out.append(
                    f"ep{window_eps[0]:04d}-ep{window_eps[-1]:04d}: "
                    f"{j - i + 1} episodes flagged ({tally_str})"
                )
                i = j + 1
            else:
                i += 1
        return out

    def write(self, out_path: Path) -> None:
        lines: List[str] = []
        lines.append("Multi-Agent Task Allocation - run summary")
        lines.append("=" * 50)
        lines.append(f"Episodes:       {self.total_episodes}")
        if self.start_time is not None and self.end_time is not None:
            wall = self.end_time - self.start_time
            mins, secs = divmod(int(wall), 60)
            per_ep = wall / max(self.total_episodes, 1)
            lines.append(f"Wall time:      {mins}m {secs:02d}s   ({per_ep:.2f}s/episode avg)")
        lines.append(f"Validation:     every {self.validate_every} episodes ({len(self.audits)} audits)")
        lines.append("")
        lines.append("Flag counts")
        lines.append("-" * 11)
        for f in _FLAG_SEVERITY:
            n = self.flag_counts.get(f, 0)
            lines.append(f"!{f:<14} : {n:>4} episodes")
        lines.append("")
        lines.append("Episode lists (open these in Panopticon for inspection)")
        lines.append("-" * 56)
        for f in _FLAG_SEVERITY:
            eps = self.flag_episodes.get(f, [])
            if not eps:
                continue
            shown = ", ".join(str(e) for e in eps[:50])
            tail = f" ... [{len(eps)} total]" if len(eps) > 50 else ""
            lines.append(f"!{f:<14} : {shown}{tail}")
        lines.append("")
        lines.append("Cluster alerts (>=3 flagged episodes within a 5-episode span)")
        lines.append("-" * 60)
        clusters = self._detect_clusters()
        if clusters:
            for c in clusters:
                lines.append(c)
        else:
            lines.append("(none)")
        lines.append("")
        lines.append(f"Rolling {self.window}-episode windows (reward / utility-ratio / RL-accuracy)")
        lines.append("-" * 64)
        for start in range(0, len(self.per_episode), self.window):
            chunk = self.per_episode[start:start + self.window]
            if not chunk:
                continue
            r = sum(c["reward"] for c in chunk) / len(chunk)
            u = sum(c["utility_ratio"] for c in chunk) / len(chunk) * 100
            tot_dec = sum(c["decisions"] for c in chunk)
            tot_match = sum(c["matches"] for c in chunk)
            m = (tot_match / tot_dec * 100) if tot_dec else 0.0
            ep_lo = chunk[0]["ep"]
            ep_hi = chunk[-1]["ep"]
            lines.append(
                f"ep{ep_lo:04d}-ep{ep_hi:04d}   r={r:+6.2f}  u={u:3.0f}%  m={m:3.0f}%"
            )
        lines.append("")
        lines.append("Validation audit summary")
        lines.append("-" * 24)
        n_audits = len(self.audits)
        with_viol = sum(1 for a in self.audits
                        if a.get("oracle_violations", 0) > 0
                        or a.get("unreachable_hit", 0) > 0)
        total_viol = sum(a.get("oracle_violations", 0) for a in self.audits)
        lines.append(
            f"Episodes audited: {n_audits}  |  audits with violations: {with_viol}  "
            f"|  total oracle_violations: {total_viol}"
        )
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# 8. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full RL Training with BLADE + MATCH-AOU")
    parser.add_argument(
        "--scenario",
        default="data/scenarios/strike_training_4v5.json",
        help="Path to base scenario JSON (used as template for pools)",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max-ticks", type=int, default=MAX_SIM_TICKS,
                        help="Max simulation ticks per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (PPO default: 3e-4)")
    parser.add_argument("--save-freq", type=int, default=100, help="Save checkpoint every N episodes")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Emit a multi-line learning-progress block every N "
                             "episodes (rolling-window aggregates with trend deltas "
                             "vs the previous window). Default 100 = ~50 blocks "
                             "across a 5000-ep run. Set to 0 to disable. The "
                             "first block has no delta; subsequent ones compare "
                             "against the immediately preceding window.")
    parser.add_argument("--validate-every", type=int, default=VALIDATE_EVERY,
                        help="Run oracle-only validation every N episodes (0=disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for logs/recordings")

    # --- Scenario variation args ---
    parser.add_argument("--vary-scenarios", action="store_true", default=VARY_SCENARIOS,
                        help="Enable scenario variation between episodes")
    parser.add_argument("--min-aircraft", type=int, default=2,
                        help="Min aircraft per episode (when --vary-scenarios)")
    parser.add_argument("--max-aircraft", type=int, default=3,
                        help="Max aircraft per episode (when --vary-scenarios)")
    parser.add_argument("--min-facilities", type=int, default=2,
                        help="Min facilities per episode (when --vary-scenarios)")
    parser.add_argument("--max-facilities", type=int, default=4,
                        help="Max facilities per episode (when --vary-scenarios)")
    parser.add_argument("--max-target-dist", type=float, default=2500.0,
                        help="Max target distance from base in km")
    parser.add_argument("--min-red-airbases", type=int, default=3,
                        help="Min RED airbases per episode (when --vary-scenarios)")
    parser.add_argument("--max-red-airbases", type=int, default=5,
                        help="Max RED airbases per episode (when --vary-scenarios)")
    parser.add_argument("--vary-base", action="store_true", default=VARY_BASE,
                        help="Also randomize blue base position")
    parser.add_argument("--include-sams", action="store_true", default=INCLUDE_SAMS,
                        help="Include SAM facilities as targets (default: False, airbases only)")
    parser.add_argument("--base-shift-km", type=float, default=150.0,
                        help="Max base shift radius in km")
    parser.add_argument("--allowed-aircraft", nargs="+", default=None,
                        help="Aircraft classes to use (e.g. 'F-35A Lightning II'). "
                             "Default: all types from pool.")
    parser.add_argument("--stretch-ratio", type=float, default=0.5,
                        help="Fraction of targets in stretch zone (only reachable by "
                             "some agents). 0.0=all easy, 0.5=50%% stretch. "
                             "No effect with homogeneous fleets. (default: 0.5)")
    parser.add_argument("--time-feasible-max-km", type=float, default=None,
                        help="Cap stretch_max at this one-way distance (km) so the "
                             "slowest aircraft in the eligible pool can round-trip a "
                             "stretch target within --max-ticks at cruise speed with "
                             "30%% safety. Default = auto-compute from the pool. "
                             "Set to a very large value (e.g. 99999) to disable the "
                             "cap for ablation runs (preserves pre-fix behaviour and "
                             "reproduces !TIMEOUT category (c) cascades).")
    parser.add_argument("--record-every", type=int, default=50,
                        help="Export BLADE recordings every N episodes. "
                             "1=every episode, 50=every 50th (default), 0=never. "
                             "Validation runs still follow --validate-every and are "
                             "recorded when they fire.")
    # --- Logging / debug flags ---
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Emit full per-tick / per-decision logs on the console "
                             "(equivalent to long-run console mode at DEBUG). "
                             "Auto-enabled when --episodes 1.")
    parser.add_argument("--fuel-damage", action="store_true", default=FUEL_DAMAGE_ENABLED,
                        help="Override FUEL_DAMAGE_ENABLED for this run only "
                             "(does not change canonical default).")
    parser.add_argument("--debug-force-flags", default="",
                        help="Comma-separated list of flag categories to force-inject "
                             "for smoke-test verification of the !FLAG prefix path. "
                             "Supported: timeout, l2-fallback. Each flag fires on a "
                             "different episode (1=timeout, 2=l2-fallback). Use only "
                             "for testing the logging refactor.")
    args = parser.parse_args()

    # Single-episode runs are always interactive debugging — flip verbose on.
    if args.episodes == 1:
        args.verbose = True

    debug_force_flags_set: Set[str] = {
        s.strip().lower() for s in args.debug_force_flags.split(",") if s.strip()
    }

    # --- Setup output directory ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    recordings_dir = output_dir / "recordings"
    recordings_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    scenarios_dir = output_dir / "scenarios"
    scenarios_dir.mkdir(exist_ok=True)

    # Clean old recordings, logs, and generated scenarios
    for old_file in recordings_dir.glob("*"):
        old_file.unlink()
        logger.debug(f"Removed old recording: {old_file}")
    for old_file in logs_dir.glob("episode_*.log"):
        old_file.unlink()
    for old_file in scenarios_dir.glob("*.json"):
        old_file.unlink()
        logger.debug(f"Removed old scenario: {old_file}")

    # --- Setup file logging ---
    # Three handlers, attached to the ROOT logger so all module loggers
    # (train_full, scenario_generator, scenario_factory, ppo_trainer,
    # fuel_damage, ...) feed through them uniformly:
    #   - Console: INFO when compact, DEBUG when --verbose.
    #   - Master training.log: mirrors the console (same level).
    #   - Per-episode episode_NNNN.log: always DEBUG (full firehose).
    # The per-episode handler is attached at the start of each episode and
    # detached at the end, in the main loop below.
    console_level = logging.DEBUG if args.verbose else logging.INFO
    master_log_path = logs_dir / "training.log"

    root = logging.getLogger()
    # Strip any handlers installed by basicConfig at module-import time.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    master_handler = logging.FileHandler(master_log_path, mode="w", encoding="utf-8")
    master_handler.setLevel(console_level)
    master_handler.setFormatter(fmt)
    root.addHandler(master_handler)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("=" * 70)
    logger.info("Full RL Training — MAPPO + BLADE + MATCH-AOU")
    logger.info("=" * 70)
    logger.info(f"Base scenario:     {args.scenario}")
    logger.info(f"Vary scenarios:    {args.vary_scenarios}")
    logger.info(f"Episodes:          {args.episodes}")
    logger.info(f"RL trigger:        event-driven (discovery + fuel damage)")
    logger.info(f"Discovery scan:    every {DISCOVERY_SCAN_INTERVAL} ticks")
    logger.info(f"Max ticks:         {args.max_ticks}")
    logger.info(f"Max agents:        {MAX_AGENTS}")
    logger.info(f"Learning rate:     {args.lr}")
    logger.info(f"Seed:              {args.seed}")
    logger.info(f"Fuel damage:       {args.fuel_damage}")
    logger.info(f"Include SAMs:      {args.include_sams}")
    logger.info(f"Allowed aircraft:  {args.allowed_aircraft or 'all (from pool)'}")
    logger.info(f"Stretch ratio:     {args.stretch_ratio}")
    logger.info(f"Validate every:    {args.validate_every} episodes")
    logger.info(f"Record every:      {args.record_every} episodes (0=never)")
    logger.info(f"Verbose console:   {args.verbose}")
    if debug_force_flags_set:
        logger.info(f"DEBUG force flags: {sorted(debug_force_flags_set)}")
    logger.info(f"Output dir:        {output_dir.resolve()}")

    # --- Setup scenario generator ---
    scenario_gen = None
    if args.vary_scenarios:
        scenario_gen = ScenarioGenerator(
            base_scenario_path=args.scenario,
            output_dir=str(scenarios_dir),
            max_sim_ticks=args.max_ticks,
        )
        # Auto-compute the time-feasibility cap from the eligible pool
        # (post `--allowed-aircraft` filter). The user can override via
        # --time-feasible-max-km.
        scenario_gen.recompute_time_feasible_cap(
            allowed_classes=args.allowed_aircraft,
        )
        if args.time_feasible_max_km is None:
            tfm_inputs = scenario_gen.time_feasible_inputs
            if tfm_inputs:
                logger.info(
                    f"Time-feasibility cap: {tfm_inputs['cap_km']:.0f} km one-way "
                    f"(slowest={tfm_inputs['slowest_class']} "
                    f"{tfm_inputs['slowest_kmh']:.0f} km/h, "
                    f"ticks={tfm_inputs['max_ticks']}, "
                    f"safety={tfm_inputs['safety']:.1f}) [auto]"
                )
            else:
                logger.info("Time-feasibility cap: not computed (empty pool)")
        else:
            logger.info(
                f"Time-feasibility cap: {args.time_feasible_max_km:.0f} km one-way "
                f"[manual override via --time-feasible-max-km]"
            )
        logger.info(
            f"ScenarioGenerator: aircraft_pool={scenario_gen.aircraft_pool.class_names}, "
            f"facility_pool={scenario_gen.facility_pool.class_names}, "
            f"aircraft=({args.min_aircraft}-{args.max_aircraft}), "
            f"facilities=({args.min_facilities}-{args.max_facilities}), "
            f"red_airbases=({args.min_red_airbases}-{args.max_red_airbases}), "
            f"max_dist={args.max_target_dist}km, vary_base={args.vary_base}"
        )

    # --- Setup BLADE ---
    logger.info("\n--- Setting up BLADE environment ---")
    game, env, initial_obs = setup_blade_env(
        args.scenario, args.max_ticks, recording_dir=str(recordings_dir)
    )

    # --- Create RL components (MAPPO) ---
    logger.info("\n--- Creating RL components (MAPPO) ---")

    obs_config = ObservationConfig(top_k=3)
    obs_dim = 6 + (3 * 6) + 6  # 30: self(6) + targets(18) + plan_context(6)
    action_dim = 5  # NOOP + 3 attacks + RTB

    network = ActorCriticNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=MAX_AGENTS,
        hidden_size=128,
    )
    actor_params = sum(p.numel() for p in network.actor.parameters())
    critic_params = sum(p.numel() for p in network.critic.parameters())
    logger.info(f"ActorCriticNetwork: actor={actor_params:,} params, critic={critic_params:,} params")
    logger.info(f"  Actor:  obs[{obs_dim}] → 128 → 64 → logits[{action_dim}]")
    logger.info(f"  Critic: global[{obs_dim * MAX_AGENTS}] → 128 → 64 → V(s)[1]")

    config = PPOConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=MAX_AGENTS,
        hidden_size=128,
        learning_rate=args.lr,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=4,
        batch_size=64,
        max_grad_norm=0.5,
        value_coef=0.5,
        entropy_coef=0.01,
        buffer_capacity=2048,
        reward_config=RewardConfig(),
        model_dir=str(models_dir),
    )
    trainer = PPOTrainer(network, config)
    logger.info("PPOTrainer ready")

    # --- Training loop ---
    logger.info("\n" + "=" * 70)
    logger.info("Starting Training")
    logger.info("=" * 70)

    all_metrics = []
    run_summary = RunSummaryBuilder(
        total_episodes=args.episodes,
        validate_every=args.validate_every,
        window=50,
    )
    import time as _time
    run_summary.start_time = _time.time()

    for episode in range(args.episodes):
        # In long-run mode the per-episode separator banner is replaced
        # by the compact summary line emitted at episode end. In verbose
        # mode the banner survives at DEBUG so it still appears on console.
        logger.debug(f"\n{'='*50}")
        logger.debug(f"Episode {episode + 1}/{args.episodes}")
        logger.debug(f"{'='*50}")

        # --- Generate or reuse scenario ---
        gen_stats: Optional[Dict] = None
        if scenario_gen is not None:
            # When SAMs are excluded, ensure at least 1 RED airbase as target
            min_rab = args.min_red_airbases
            if not args.include_sams and min_rab < 1:
                min_rab = 1

            ep_config = VariationConfig(
                include_sams=args.include_sams,
                num_aircraft=(args.min_aircraft, args.max_aircraft),
                allowed_aircraft_classes=args.allowed_aircraft,
                num_facilities=(args.min_facilities, args.max_facilities),
                num_red_airbases=(min_rab, args.max_red_airbases),
                randomize_facility_positions=True,
                randomize_red_airbase_positions=True,
                max_target_distance_km=args.max_target_dist,
                stretch_target_ratio=args.stretch_ratio,
                # None → ScenarioGenerator.generate() injects the
                # auto-computed cap. Explicit value → use as-is.
                time_feasible_max_km=args.time_feasible_max_km,
                randomize_base_position=args.vary_base,
                base_shift_radius_km=args.base_shift_km,
                seed=args.seed + episode,  # Deterministic per episode
            )
            ep_scenario_path = str(scenario_gen.generate(
                episode=episode, config=ep_config,
            ))
            reload_scenario(game, ep_scenario_path)
            logger.debug(f"  Generated scenario: {Path(ep_scenario_path).name}")
            gen_stats = dict(scenario_gen.last_generation_stats)
        else:
            ep_scenario_path = args.scenario

        # Decide whether this episode gets a recording.
        should_record = (args.record_every > 0 and episode % args.record_every == 0)
        is_validation_episode = (
            args.validate_every > 0 and episode % args.validate_every == 0
        )

        # Per-episode log file (DEBUG firehose). Attach BEFORE the
        # validation run so the audit block lands in the per-episode
        # file as well as on console.
        ep_log_path = logs_dir / f"episode_{episode + 1:04d}.log"
        ep_handler = logging.FileHandler(ep_log_path, mode="w", encoding="utf-8")
        ep_handler.setLevel(logging.DEBUG)
        ep_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        ))
        root.addHandler(ep_handler)

        # --- Per-episode injection of debug-only forced flags ---
        ep_force = set()
        if "timeout" in debug_force_flags_set and (episode % 5 == 0):
            ep_force.add("timeout")
        if "l2-fallback" in debug_force_flags_set and (episode % 5 == 1):
            ep_force.add("l2-fallback")

        # --- Validation run (oracle only, every N episodes) ---
        audit: Optional[Dict] = None
        if is_validation_episode:
            audit = run_validation_episode(
                game=game,
                env=env,
                scenario_path=ep_scenario_path,
                episode_num=episode,
                max_ticks=args.max_ticks,
                record=should_record,
            )
            # Reload same scenario fresh for the RL episode
            reload_scenario(game, ep_scenario_path)

        # --- RL training episode (wrapped to surface CRASH cleanly) ---
        try:
            metrics = train_episode(
                trainer=trainer,
                game=game,
                env=env,
                scenario_path=ep_scenario_path,
                obs_config=obs_config,
                episode_num=episode,
                max_ticks=args.max_ticks,
                record=should_record,
                fuel_damage_enabled=args.fuel_damage,
                debug_force_flags=ep_force,
            )
        except Exception as e:
            # Console-visible !CRASH banner + full traceback. Record into
            # run_summary and continue to the next episode so a one-off
            # failure doesn't kill a 1000-episode run.
            logger.exception(
                f"!CRASH ep{episode + 1:04d}  {type(e).__name__}: {e}"
            )
            logger.error(f"!CRASH ep{episode + 1:04d}  (episode aborted, continuing)")
            run_summary.record(episode, _empty_metrics(), ["CRASH"], None)
            root.removeHandler(ep_handler)
            ep_handler.close()
            continue

        # Detach per-episode handler before we emit the summary line, so
        # the line lands on console + master + ep_NNNN file (the ep file
        # already captured everything from the RL episode).
        root.removeHandler(ep_handler)
        ep_handler.close()

        all_metrics.append(metrics)

        # --- Compact 2-line per-episode summary ---
        lines, flags = _format_episode_summary(
            episode_num=episode,
            metrics=metrics,
            gen_stats=gen_stats,
            is_validation=is_validation_episode,
            audit=audit,
            fuel_damage_enabled=args.fuel_damage,
        )
        for ln in lines:
            if any(f in flags for f in ("CRASH", "ANOMALY", "TIMEOUT",
                                        "L2-exhaust", "L2-fallback", "noPPO")):
                logger.warning(ln)
            else:
                logger.info(ln)
        # Stash flags onto the metrics dict so the progress-block
        # aggregator can tally per-window flag counts later.
        metrics["_flags"] = list(flags)
        run_summary.record(episode, metrics, flags, audit)

        # --- Record-on-flag: replay any flagged episode that wasn't
        # already recorded by --record-every cadence. The replay reuses
        # the same scenario file (deterministic per seed=episode_num) but
        # the policy has been updated since the original run, so the
        # replay shows current-policy behaviour on this scenario. We
        # skip CRASH (no clean recovery point) and any flag that means
        # the original recording window already covered the episode. ---
        recordable = {"TIMEOUT", "ANOMALY", "noPPO",
                      "L2-fallback", "L2-exhaust"}
        flag_replay = [f for f in flags if f in recordable]
        if flag_replay and not should_record:
            tag = "_".join(flag_replay)
            replay_name = f"ep{episode + 1:04d}_flagged_{tag}_rl"
            logger.info(f"  → Replaying flagged episode for recording: {replay_name}")
            try:
                reload_scenario(game, ep_scenario_path)
                train_episode(
                    trainer=trainer,
                    game=game,
                    env=env,
                    scenario_path=ep_scenario_path,
                    obs_config=obs_config,
                    episode_num=episode,
                    max_ticks=args.max_ticks,
                    record=True,
                    record_name=replay_name,
                    fuel_damage_enabled=args.fuel_damage,
                    debug_force_flags=ep_force,
                    replay_only=True,
                )
            except Exception as e:
                logger.warning(f"  Flagged-episode replay failed: {e}")

        # --- Save checkpoint (model serialisation) ---
        ckpt_due = (episode + 1) % args.save_freq == 0
        if ckpt_due:
            ckpt_name = f"checkpoint_ep{episode + 1}.pt"
            trainer.save_checkpoint(ckpt_name)

        # --- Learning-progress block (rolling-window aggregates with
        # trend deltas vs the previous window — the primary signal for
        # "is the RL actually learning?" during a long run). ---
        progress_due = (
            args.progress_every > 0
            and (episode + 1) % args.progress_every == 0
        )
        if progress_due:
            window = args.progress_every
            current_window = all_metrics[-window:]
            prev_window = (
                all_metrics[-2 * window:-window]
                if len(all_metrics) >= 2 * window else None
            )
            cur_agg = _aggregate_window(current_window)
            prev_agg = _aggregate_window(prev_window) if prev_window else None
            for ln in _format_progress_block(
                ep_num=episode + 1, current=cur_agg, previous=prev_agg,
                checkpoint_saved=ckpt_due, window=window,
            ):
                logger.info(ln)
        elif ckpt_due:
            # Checkpoint without progress block (only happens if user set
            # --progress-every 0 explicitly; keep the lightweight banner).
            recent = all_metrics[-args.save_freq:]
            avg_r = (sum(m["episode_reward"] for m in recent) / len(recent)
                     if recent else 0.0)
            logger.info(
                f"=== Checkpoint saved (ep{episode + 1:04d}) | "
                f"rolling avg reward (last {len(recent)}): {avg_r:+.2f} ==="
            )

    # --- Final summary ---
    run_summary.end_time = _time.time()
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)

    trainer.save_checkpoint("final_model.pt")
    network.save(str(models_dir / "actor_critic_final.pt"))

    summary = trainer.get_metrics_summary()
    logger.info(f"Total episodes:      {summary['episode_count']}")
    logger.info(f"Total PPO updates:   {summary['total_updates']}")
    logger.info(f"Avg policy loss:     {summary.get('avg_policy_loss', 0):.4f}")
    logger.info(f"Avg value loss:      {summary.get('avg_value_loss', 0):.4f}")

    if all_metrics:
        avg_reward = np.mean([m["episode_reward"] for m in all_metrics[-10:]])
        avg_accuracy = np.mean([m["accuracy"] for m in all_metrics[-10:]])
        avg_utility = np.mean([m.get("utility_ratio", 0) for m in all_metrics[-10:]])
        logger.info(f"Avg reward (last 10): {avg_reward:.2f}")
        logger.info(f"Avg accuracy (last 10): {avg_accuracy:.1%}")
        logger.info(f"Avg utility ratio (last 10): {avg_utility:.1%}")

    # Write the categorised run summary so problematic episodes can be
    # found and inspected in Panopticon without re-grepping training.log.
    run_summary_path = logs_dir / "run_summary.txt"
    try:
        run_summary.write(run_summary_path)
        logger.info(f"Run summary written to: {run_summary_path}")
    except Exception as e:
        logger.warning(f"Failed to write run_summary.txt: {e}")

    # highlights.txt — curated index for Panopticon viewing. Sections:
    # perfect-match RL eps, significant-mismatch eps, learning-trend
    # samples, and flagged-episode index with recording filenames.
    highlights_path = logs_dir / "highlights.txt"
    try:
        _write_highlights(
            out_path=highlights_path,
            all_metrics=all_metrics,
            flag_episodes=run_summary.flag_episodes,
            recordings_dir=recordings_dir,
        )
        logger.info(f"Highlights written to:  {highlights_path}")
    except Exception as e:
        logger.warning(f"Failed to write highlights.txt: {e}")

    logger.info(f"\nOutputs saved to: {output_dir.resolve()}")
    logger.info(f"  Logs:       {logs_dir}/")
    logger.info(f"  Recordings: {recordings_dir}/")
    logger.info(f"  Models:     {models_dir}/")
    logger.info(f"  Scenarios:  {scenarios_dir}/")


if __name__ == "__main__":
    main()