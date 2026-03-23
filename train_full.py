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

# --- BLADE imports ---
import gymnasium
from blade.Game import Game
from blade.Scenario import Scenario

# Override BLADE's 10MB recording file-size limit to avoid splitting
# recordings into multiple files mid-episode.
import blade.utils.PlaybackRecorder as _pbr
_pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # 500MB

# --- MATCH-AOU imports ---
from match_aou.solvers import MatchAou
from match_aou.models import Agent, Capability, Location, Step, StepType, Task
from match_aou.utils.scheduling_utils import post_solve_filter_and_level
from match_aou.utils.blade_utils import create_agents_from_scenario
from match_aou.utils.blade_utils.blade_executor_minimal import BladeExecutorMinimal
from match_aou.utils.blade_utils.scenario_factory import _normalize_side_color

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
FUEL_DAMAGE_ENABLED = True     # Toggle fuel damage surprise events
VALIDATE_EVERY = 10            # Run oracle-only validation every N episodes (0=disabled)
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
# 1. Task Generation
# =============================================================================

def generate_all_enemy_tasks(scenario, attacking_side_color: str) -> List[Task]:
    """
    Generate one attack Task per enemy target (facility / airbase / ship).

    Scans all units in the scenario and creates a Task for each one whose
    side_color differs from ours.

    Args:
        scenario: BLADE Scenario observation object
        attacking_side_color: Our side color (e.g. "blue")

    Returns:
        List of Task objects, one per enemy target
    """
    tasks: List[Task] = []
    our_side = _normalize_side_color(attacking_side_color)

    attack_capability = Capability(name="attack", properties={"Quantity": 2})
    attack_step_type = StepType(name="attack", base_cost=1)

    def _make_task(unit, utility: float) -> Task:
        target_loc = Location(
            unit.latitude,
            unit.longitude,
            getattr(unit, "altitude", 0) or 0,
        )
        step = Step(
            location=target_loc,
            capabilities=[attack_capability],
            step_type=attack_step_type,
            effort=2,
            probability=0.6,
            action=f"handle_aircraft_attack('AGENT_ID', '{unit.id}', 'WEAPON_ID', 2)",
        )
        return Task(steps=[step], utility=utility)

    # Enemy facilities
    for facility in getattr(scenario, "facilities", []) or []:
        side = _normalize_side_color(getattr(facility, "side_color", ""))
        if side and side != our_side:
            tasks.append(_make_task(facility, utility=100))

    # Enemy airbases
    for airbase in getattr(scenario, "airbases", []) or []:
        side = _normalize_side_color(getattr(airbase, "side_color", ""))
        if side and side != our_side:
            tasks.append(_make_task(airbase, utility=80))

    # Enemy ships
    for ship in getattr(scenario, "ships", []) or []:
        side = _normalize_side_color(getattr(ship, "side_color", ""))
        if side and side != our_side:
            tasks.append(_make_task(ship, utility=95))

    logger.info(f"Generated {len(tasks)} enemy tasks")

    return tasks


# =============================================================================
# 2. BLADE Environment Setup
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

    model = MatchAou(agents=agents, tasks=tasks, precedence_relations=[])
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

    logger.info(f"  → {sum(len(v) for v in artifacts.solution.values())} assignments, {len(unselected)} unselected")

    return artifacts.solution, artifacts.tasks, unselected


def split_tasks(
    all_tasks: List[Task],
    partial_ratio: float = PARTIAL_RATIO,
) -> Tuple[List[Task], List[Task]]:
    """
    Split tasks into partial (known) and full (oracle) sets.

    The partial set is what the agents initially know about.
    The full set is what the oracle knows (ground truth).

    Args:
        all_tasks: All generated tasks
        partial_ratio: Fraction for partial set (default 2/3)

    Returns:
        (partial_tasks, full_tasks) where full_tasks == all_tasks
    """
    full_tasks = list(all_tasks)  # Oracle knows everything

    num_partial = max(1, int(len(all_tasks) * partial_ratio))
    partial_tasks = random.sample(all_tasks, num_partial)

    hidden = [t for t in full_tasks if t not in partial_tasks]
    logger.info(f"Task split: {len(partial_tasks)} partial, {len(full_tasks)} full, {len(hidden)} hidden")

    return partial_tasks, full_tasks


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


def _log_blade_action(tick: int, action: str, source: str) -> None:
    """
    Parse a BLADE action string and log it in a human-readable format.

    Args:
        tick: Current simulation tick
        action: BLADE action string (e.g., "handle_aircraft_attack(...)")
        source: "EXEC" for executor, "RL" for RL override
    """
    if not action:
        return

    m = _RE_ATTACK.search(action)
    if m:
        agent_id, target_id = m.group(1), m.group(2)
        logger.info(
            f"  Tick {tick:5d} [{source:4s}] ATTACK: "
            f"agent {agent_id[:8]}.. → target {target_id[:8]}.."
        )
        return

    m = _RE_MOVE.search(action)
    if m:
        agent_id, coords = m.group(1), m.group(2)
        logger.info(
            f"  Tick {tick:5d} [{source:4s}] MOVE:   "
            f"agent {agent_id[:8]}.. → ({coords})"
        )
        return

    m = _RE_LAUNCH.search(action)
    if m:
        logger.info(f"  Tick {tick:5d} [{source:4s}] LAUNCH: from airbase {m.group(1)[:8]}..")
        return

    m = _RE_RTB.search(action)
    if m:
        logger.info(f"  Tick {tick:5d} [{source:4s}] RTB:    agent {m.group(1)[:8]}..")
        return

    # Fallback: unknown action format
    logger.info(f"  Tick {tick:5d} [{source:4s}] ACTION: {action[:80]}")


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
    logger.info(
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
) -> None:
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
    logger.info("--- Validation run (oracle only, no RL) ---")

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
        return

    all_tasks = generate_all_enemy_tasks(observation, ATTACKING_SIDE_COLOR)
    if not all_tasks:
        logger.warning("Validation: no tasks found, skipping")
        return

    logger.info(f"Validation: {len(attacking_agents)} agents, {len(all_tasks)} tasks")

    # --- Solve MATCH-AOU (full) ---
    solution, tasks_filtered, _ = solve_match_aou(
        attacking_agents, all_tasks, SOLVER_NAME
    )
    if not solution:
        logger.warning("Validation: solver returned empty solution, skipping")
        return

    # --- Launch aircraft ---
    game.current_scenario.name = f"ep{episode_num + 1:03d}_validation"
    game.start_recording()
    game.record_step()

    for _ in range(5):
        observation, _, _, _, _ = env.step("")
        game.record_step(force=True)

    for airbase in getattr(observation, "airbases", []) or []:
        ab_side = _normalize_side_color(getattr(airbase, "side_color", ""))
        if ab_side != ATTACKING_SIDE_COLOR:
            continue
        ab_id = str(airbase.id)
        for ac in list(getattr(airbase, "aircraft", []) or []):
            observation, _, _, _, _ = env.step(
                f"launch_aircraft_from_airbase('{ab_id}')"
            )
            game.record_step()

    for _ in range(10):
        observation, _, _, _, _ = env.step("")
        game.record_step()

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

    for tick in range(max_ticks):
        try:
            action = executor.next_action(observation, fallback_tick=tick) or ""
        except ValueError:
            action = ""

        observation, _, terminated, truncated, _ = env.step(action)
        game.record_step()

        # Check RTB
        airborne_ids = {
            str(getattr(ac, "id", ""))
            for ac in getattr(observation, "aircraft", []) or []
        }
        for aid in agent_ids:
            if aid not in returned and aid not in airborne_ids:
                returned.add(aid)

        if tick > 100 and len(returned) == len(agent_ids):
            logger.info(f"  Validation: all agents RTB at tick {tick}")
            break
        if terminated or truncated:
            logger.info(
                f"  Validation ended at tick {tick}: "
                f"terminated={terminated}, truncated={truncated}"
            )
            break

    # --- Export recording ---
    try:
        game.export_recording()
        logger.info(f"  Validation recording exported: ep{episode_num + 1:03d}_validation")
    except Exception as e:
        logger.warning(f"  Failed to export validation recording: {e}")


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
    verbose = (episode_num == 0)  # Full diagnostics only for first episode

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

    logger.info(
        f"Scenario: {len(_ac_types)} agents {_ac_types} | Blue base: {_base_loc}"
    )
    logger.info(
        f"  Targets ({len(_target_parts)}): {', '.join(_target_parts)}"
    )

    # --- Verbose: print agents and tasks ---
    if verbose:
        logger.info("")
        logger.info("=" * 60)
        logger.info("AGENTS")
        logger.info("=" * 60)
        for i, a in enumerate(attacking_agents):
            logger.info(f"  Agent {i}: {a.id}")
            logger.info(f"    Name:      (from scenario)")
            logger.info(f"    Location:  ({a.location.latitude:.4f}, {a.location.longitude:.4f})")
            logger.info(f"    Budget:    {a.budget:.0f}")
            logger.info(f"    Weapon ID: {a.weapon_id}")
            logger.info(f"    Home base: {a.home_base_id}")
            caps = [c.name for c in a.capabilities] if a.capabilities else []
            logger.info(f"    Capabilities: {caps}")

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"ALL TASKS ({len(all_tasks)} total)")
        logger.info("=" * 60)
        for i, t in enumerate(all_tasks):
            action_str = t.steps[0].action
            target_id = extract_target_id_from_action(action_str) or "?"
            loc = t.steps[0].location
            logger.info(f"  Task {i}:")
            logger.info(f"    Target ID: {target_id}")
            logger.info(f"    Utility:   {t.utility}")
            logger.info(f"    Location:  ({loc.latitude:.4f}, {loc.longitude:.4f})")
            logger.info(f"    Action:    {action_str}")

    # --- Step 3: Split tasks ---
    partial_tasks, full_tasks = split_tasks(all_tasks, PARTIAL_RATIO)

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

        logger.info("")
        logger.info("=" * 60)
        logger.info("TASK SPLIT")
        logger.info("=" * 60)
        logger.info(f"  Partial tasks ({len(partial_tasks)}):")
        for i, t in enumerate(partial_tasks):
            tid = extract_target_id_from_action(t.steps[0].action) or "?"
            logger.info(f"    [{i}] target={tid}, utility={t.utility}")
        logger.info(f"  Full tasks ({len(full_tasks)}):")
        for i, t in enumerate(full_tasks):
            tid = extract_target_id_from_action(t.steps[0].action) or "?"
            marker = " *** HIDDEN ***" if tid in hidden_ids else ""
            logger.info(f"    [{i}] target={tid}, utility={t.utility}{marker}")
        logger.info(f"  Hidden targets: {hidden_ids}")

    # --- Step 4: Solve MATCH-AOU twice ---
    if verbose:
        logger.info("")
        logger.info("=" * 60)
        logger.info("MATCH-AOU SOLUTIONS")
        logger.info("=" * 60)

    logger.info("Solving MATCH-AOU (partial)...")
    partial_solution, partial_tasks_filtered, _ = solve_match_aou(
        attacking_agents, partial_tasks, SOLVER_NAME
    )

    if verbose:
        logger.info("  --- Partial Solution ---")
        _log_solution_details(partial_solution, partial_tasks_filtered)

    logger.info("Solving MATCH-AOU (full / oracle)...")
    full_solution, full_tasks_filtered, _ = solve_match_aou(
        attacking_agents, full_tasks, SOLVER_NAME
    )

    if verbose:
        logger.info("  --- Full (Oracle) Solution ---")
        _log_solution_details(full_solution, full_tasks_filtered)

        # Show the diff
        partial_all_targets = _extract_all_target_ids(partial_solution, partial_tasks_filtered)
        full_all_targets = _extract_all_target_ids(full_solution, full_tasks_filtered)
        new_in_full = full_all_targets - partial_all_targets
        logger.info(f"  --- Comparison ---")
        logger.info(f"  Targets in partial: {partial_all_targets}")
        logger.info(f"  Targets in full:    {full_all_targets}")
        logger.info(f"  NEW in full (what RL should learn to attack): {new_in_full}")

    if not partial_solution:
        logger.warning("Partial solution empty, skipping episode")
        return _empty_metrics()

    # --- Step 5: Pre-launch all aircraft from airbases ---
    game.current_scenario.name = f"ep{episode_num + 1:03d}_rl"
    game.start_recording()
    game.record_step()

    # Buffer: run a few empty ticks so recording shows aircraft at base before launch
    PRE_LAUNCH_BUFFER = 5
    for _ in range(PRE_LAUNCH_BUFFER):
        observation, _, _, _, _ = env.step("")
        game.record_step(force=True)

    if verbose:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PRE-LAUNCH")
        logger.info("=" * 60)

    for airbase in getattr(observation, "airbases", []) or []:
        ab_side = _normalize_side_color(getattr(airbase, "side_color", ""))
        if ab_side != ATTACKING_SIDE_COLOR:
            continue
        ab_id = str(airbase.id)
        aircraft_in_base = list(getattr(airbase, "aircraft", []) or [])
        for ac in aircraft_in_base:
            launch_cmd = f"launch_aircraft_from_airbase('{ab_id}')"
            ac_name = getattr(ac, 'name', ac.id)
            logger.info(f"  LAUNCH: {ac_name} (id={str(ac.id)[:8]}..) from airbase {ab_id[:8]}..")
            observation, _, _, _, _ = env.step(launch_cmd)
            game.record_step()

    for _ in range(10):
        observation, _, _, _, _ = env.step("")
        game.record_step()

    airborne = [getattr(ac, 'name', ac.id) for ac in getattr(observation, 'aircraft', [])]
    logger.info(f"  Airborne after launch: {len(airborne)} aircraft — {airborne}")

    # --- Step 6: Setup executor with partial plan ---
    executor = BladeExecutorMinimal(
        tasks=partial_tasks_filtered,
        solution=partial_solution,
        agents=attacking_agents,
        add_return_to_base=True,
        arrival_threshold_km=50.0,
    )

    if verbose:
        logger.info("")
        logger.info("=" * 60)
        logger.info("EXECUTOR QUEUE")
        logger.info("=" * 60)
        for aid, q in executor.queue.items():
            logger.info(f"  Agent {aid}: {len(q)} assignments")
            for task_idx, step_idx, level in q:
                target_id = "?"
                if 0 <= task_idx < len(partial_tasks_filtered):
                    action = getattr(partial_tasks_filtered[task_idx].steps[step_idx], "action", "")
                    target_id = extract_target_id_from_action(action) or "?"
                logger.info(f"    task={task_idx}, step={step_idx}, level={level}, target={target_id}")

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
        logger.info("")
        logger.info("=" * 60)
        logger.info("ORACLE SETUP")
        logger.info("=" * 60)
        logger.info(f"  Partial target IDs (known): {partial_target_ids}")
        for aid, targets in full_agent_targets.items():
            logger.info(f"  Full targets for {aid}: {targets}")
            new = targets - partial_target_ids
            if new:
                logger.info(f"    → Agent should learn to attack: {new}")
        logger.info("")
        logger.info("=" * 60)
        logger.info("SIMULATION START")
        logger.info("=" * 60)

    # --- Step 7: Simulation loop ---
    episode_reward = 0.0
    decisions = 0
    matches = 0
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
    # Track which targets RL agents successfully attacked (for episode-end reward)
    rl_attacked_target_ids: Set[str] = set()

    if verbose:
        logger.info(f"  Utility map: {target_utility_map}")
        logger.info(f"  Max utility: {max_utility}")
        logger.info(f"  Oracle total utility: {oracle_total_utility}")

    # --- Fuel damage setup ---
    fuel_dmg = FuelDamageManager(FuelDamageConfig(enabled=FUEL_DAMAGE_ENABLED))
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
                    # Apply fuel damage to observation (if active)
                    if fuel_dmg.is_damaged(agent_id):
                        obs.vector[0] = fuel_dmg.apply_damage(agent_id, obs.vector[0])
                        obs.self_state.fuel_norm = obs.vector[0]
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
                            logger.info(
                                f"  Tick {tick:5d} DISCOVERY: "
                                f"agent {agent_id[:8]}.. sees target {target.id[:8]}.."
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

                # Compute utilities for reward
                rl_utility = get_action_utility(rl_action, obs, target_utility_map)
                oracle_utility = get_action_utility(oracle_action, obs, target_utility_map)

                is_valid = bool(action_mask[rl_action]) if rl_action < len(action_mask) else False
                reward = compute_step_reward(
                    rl_action=rl_action,
                    oracle_action=oracle_action,
                    rl_utility=rl_utility,
                    oracle_utility=oracle_utility,
                    max_utility=max_utility,
                    is_valid=is_valid,
                    config=trainer.config.reward_config,
                )

                episode_reward += reward
                decisions += 1
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

                # Log every event-driven RL decision (these are rare and meaningful)
                action_names = {0: "NOOP", 1: "ATTACK_0", 2: "ATTACK_1", 3: "ATTACK_2", 4: "RTB"}
                logger.info(
                    f"  Tick {tick:5d} RL DECISION: {agent_id[:8]}.. | "
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

                # Store in rollout buffer (PPO collects, trains AFTER episode)
                done = (tick >= max_ticks - 1)
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
            _log_blade_action(tick, final_action, source)

        observation, _reward, terminated, truncated, info = env.step(final_action)
        game.record_step()

        # Check which agents have returned to base (no longer airborne)
        airborne_ids = {str(getattr(ac, "id", "")) for ac in getattr(observation, "aircraft", []) or []}
        for agent_obj in attacking_agents:
            aid = str(agent_obj.id)
            if aid not in returned_agents and aid not in airborne_ids:
                returned_agents.add(aid)
                logger.info(f"  Tick {tick:5d} RTB:     agent {aid[:8]}.. landed")

        # --- Detailed logging near end of episode ---
        # Log every tick in the last 100 before max_ticks, or when
        # terminated/truncated is about to fire, so we can see exactly
        # what's happening when the episode cuts off.
        ticks_remaining = max_ticks - tick
        if ticks_remaining <= 100 and ticks_remaining % 10 == 0:
            all_aircraft = getattr(observation, "aircraft", []) or []
            logger.info(
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
                logger.info(
                    f"    {ac_name} (id={ac_id}..): "
                    f"pos=({lat:.2f},{lon:.2f}) fuel={fuel:.0f} "
                    f"rtb={rtb} route_pts={route_len}"
                )

        # End episode when all agents have returned to base
        if tick > 100 and len(returned_agents) == len(attacking_agents):
            logger.info(f"  All agents returned to base at tick {tick} — ending episode")
            break

        if terminated or truncated:
            logger.info(
                f"  Episode ended at tick {tick}: "
                f"terminated={terminated}, truncated={truncated} "
                f"(env step count ≈ {tick + PRE_LAUNCH_BUFFER + n_agents + 10})"
            )
            break

        # Periodic progress summary
        if tick > 0 and tick % PROGRESS_LOG_INTERVAL == 0:
            _log_progress(
                tick, n_agents, returned_agents, decisions,
                episode_reward, rl_attacked_target_ids, len(all_tasks),
            )

    # === End of episode: compute episode-end utility reward ===
    # Sum utility of targets RL successfully attacked
    achieved_utility = sum(
        target_utility_map.get(tid, 0.0) for tid in rl_attacked_target_ids
    )
    ep_reward = compute_episode_reward(
        achieved_utility=achieved_utility,
        oracle_total_utility=oracle_total_utility,
        config=trainer.config.reward_config,
    )
    # Add episode reward to the LAST transition in buffer
    # GAE will propagate this backward through advantages
    if trainer.buffer.size > 0:
        trainer.buffer.rewards[trainer.buffer.size - 1] += ep_reward
        episode_reward += ep_reward

    trainer.reward_tracker.set_episode_utilities(achieved_utility, oracle_total_utility)

    logger.info(
        f"  Episode utility: achieved={achieved_utility:.0f} / "
        f"oracle={oracle_total_utility:.0f} "
        f"(ratio={achieved_utility / max(oracle_total_utility, 1):.2f}) "
        f"→ ep_reward={ep_reward:+.2f}"
    )

    # === PPO update ===
    # Compute GAE advantages using collected trajectory
    last_value = 0.0  # Terminal state → value = 0
    if trainer.buffer.size > 0:
        trainer.buffer.compute_returns_and_advantages(last_value)
        update_metrics = trainer.update()
        logger.info(
            f"  PPO update: policy_loss={update_metrics.get('policy_loss', 0):.4f}, "
            f"value_loss={update_metrics.get('value_loss', 0):.4f}, "
            f"entropy={update_metrics.get('entropy', 0):.4f}, "
            f"clip_frac={update_metrics.get('clip_fraction', 0):.3f}"
        )
    else:
        update_metrics = {}
        logger.warning("  No transitions collected, skipping PPO update")

    trainer.episode_count += 1
    trainer.buffer.reset()

    try:
        game.export_recording()
        logger.info(f"  Recording exported: ep{episode_num + 1:03d}_rl")
    except Exception as e:
        logger.warning(f"  Failed to export recording: {e}")

    accuracy = matches / max(decisions, 1)

    return {
        "episode_reward": episode_reward,
        "decisions": decisions,
        "accuracy": accuracy,
        "ticks": tick + 1,
        "n_agents": n_agents,
        "n_tasks": len(all_tasks),
        "policy_loss": update_metrics.get("policy_loss", 0.0),
        "value_loss": update_metrics.get("value_loss", 0.0),
        "entropy": update_metrics.get("entropy", 0.0),
        "achieved_utility": achieved_utility,
        "oracle_utility": oracle_total_utility,
        "utility_ratio": achieved_utility / max(oracle_total_utility, 1),
        "fuel_damage_events": len(fuel_dmg.events),
    }


# =============================================================================
# Logging Helpers
# =============================================================================

def _empty_metrics() -> Dict:
    """Return empty metrics dict for skipped episodes."""
    return {"episode_reward": 0, "decisions": 0, "accuracy": 0, "ticks": 0,
            "policy_loss": 0, "value_loss": 0, "entropy": 0}


def _log_solution_details(solution: Dict, tasks: List[Task]):
    """Pretty-print a MATCH-AOU solution."""
    total = sum(len(v) for v in solution.values())
    logger.info(f"  Total assignments: {total}")
    for agent_id, assignments in solution.items():
        logger.info(f"  Agent {agent_id}:")
        for task_idx, step_idx, level in assignments:
            target_id = "?"
            action = ""
            if 0 <= task_idx < len(tasks):
                action = getattr(tasks[task_idx].steps[step_idx], "action", "") or ""
                target_id = extract_target_id_from_action(action) or "?"
            logger.info(
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
# 8. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full RL Training with BLADE + MATCH-AOU")
    parser.add_argument(
        "--scenario",
        default="data/scenarios/strike_training_4v5.json",
        help="Path to base scenario JSON (used as template for pools)",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--max-ticks", type=int, default=MAX_SIM_TICKS,
                        help="Max simulation ticks per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (PPO default: 3e-4)")
    parser.add_argument("--save-freq", type=int, default=10, help="Save checkpoint every N episodes")
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
    parser.add_argument("--max-target-dist", type=float, default=500.0,
                        help="Max target distance from base in km")
    parser.add_argument("--min-red-airbases", type=int, default=2,
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
    args = parser.parse_args()

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
    # Console: INFO level (compact)
    # Per-episode files: DEBUG level (full detail)
    # Master log: everything
    master_log_path = logs_dir / "training.log"
    master_handler = logging.FileHandler(master_log_path, mode="w", encoding="utf-8")
    master_handler.setLevel(logging.DEBUG)
    master_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s"
    ))
    logger.addHandler(master_handler)

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
    logger.info(f"Fuel damage:       {FUEL_DAMAGE_ENABLED}")
    logger.info(f"Include SAMs:      {args.include_sams}")
    logger.info(f"Allowed aircraft:  {args.allowed_aircraft or 'all (from pool)'}")
    logger.info(f"Validate every:    {args.validate_every} episodes")
    logger.info(f"Output dir:        {output_dir.resolve()}")

    # --- Setup scenario generator ---
    scenario_gen = None
    if args.vary_scenarios:
        scenario_gen = ScenarioGenerator(
            base_scenario_path=args.scenario,
            output_dir=str(scenarios_dir),
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

    for episode in range(args.episodes):
        logger.info(f"\n{'='*50}")
        logger.info(f"Episode {episode + 1}/{args.episodes}")
        logger.info(f"{'='*50}")

        # --- Generate or reuse scenario ---
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
                randomize_base_position=args.vary_base,
                base_shift_radius_km=args.base_shift_km,
                seed=args.seed + episode,  # Deterministic per episode
            )
            ep_scenario_path = str(scenario_gen.generate(
                episode=episode, config=ep_config,
            ))
            reload_scenario(game, ep_scenario_path)
            logger.info(f"  Generated scenario: {Path(ep_scenario_path).name}")
        else:
            ep_scenario_path = args.scenario

        # --- Validation run (oracle only, every N episodes) ---
        if args.validate_every > 0 and episode % args.validate_every == 0:
            run_validation_episode(
                game=game,
                env=env,
                scenario_path=ep_scenario_path,
                episode_num=episode,
                max_ticks=args.max_ticks,
            )
            # Reload same scenario fresh for the RL episode
            reload_scenario(game, ep_scenario_path)

        # Per-episode log file
        ep_log_path = logs_dir / f"episode_{episode + 1:03d}.log"
        ep_handler = logging.FileHandler(ep_log_path, mode="w", encoding="utf-8")
        ep_handler.setLevel(logging.DEBUG)
        ep_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s"
        ))
        logger.addHandler(ep_handler)

        metrics = train_episode(
            trainer=trainer,
            game=game,
            env=env,
            scenario_path=ep_scenario_path,
            obs_config=obs_config,
            episode_num=episode,
            max_ticks=args.max_ticks,
        )

        # Remove per-episode handler
        logger.removeHandler(ep_handler)
        ep_handler.close()

        all_metrics.append(metrics)

        logger.info(
            f"  → Reward: {metrics['episode_reward']:7.2f} | "
            f"Decisions: {metrics['decisions']:3d} | "
            f"Accuracy: {metrics['accuracy']:5.1%} | "
            f"Utility: {metrics.get('utility_ratio', 0):5.1%} | "
            f"π_loss: {metrics['policy_loss']:.4f} | "
            f"V_loss: {metrics['value_loss']:.4f} | "
            f"Ticks: {metrics['ticks']} | "
            f"Agents: {metrics.get('n_agents', '?')} | "
            f"Tasks: {metrics.get('n_tasks', '?')}"
        )

        # Save checkpoint
        if (episode + 1) % args.save_freq == 0:
            ckpt_name = f"checkpoint_ep{episode + 1}.pt"
            trainer.save_checkpoint(ckpt_name)
            logger.info(f"  → Saved checkpoint: {models_dir / ckpt_name}")

    # --- Final summary ---
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

    logger.info(f"\nOutputs saved to: {output_dir.resolve()}")
    logger.info(f"  Logs:       {logs_dir}/")
    logger.info(f"  Recordings: {recordings_dir}/")
    logger.info(f"  Models:     {models_dir}/")
    logger.info(f"  Scenarios:  {scenarios_dir}/")


if __name__ == "__main__":
    main()