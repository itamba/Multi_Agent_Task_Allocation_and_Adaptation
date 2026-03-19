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
    6. At decision points (every N ticks + discovery events):
       - Build local obs per agent + global obs (padded to MAX_AGENTS)
       - Actor samples action from policy π(a|o)
       - Critic estimates state value V(s) from global state
       - Oracle provides ground truth from full solution
       - Compute imitation reward, store in rollout buffer
    7. After episode: compute GAE advantages → PPO update (K epochs)

Usage:
    # Fixed scenario (original behavior):
    python train_full.py --scenario data/scenarios/strike_training_2v3.json --episodes 50

    # Varied scenarios (new):
    python train_full.py --scenario data/scenarios/strike_training_2v3.json \\
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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

# --- BLADE imports ---
import gymnasium
from gymnasium.wrappers.common import TimeLimit
from blade.Game import Game
from blade.Scenario import Scenario

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
DECISION_INTERVAL = 100       # RL decides every N ticks
PARTIAL_RATIO = 2 / 3         # Fraction of tasks in partial set
VARY_SCENARIOS = True          # Toggle scenario variation (or use --vary-scenarios flag)
FUEL_DAMAGE_ENABLED = True     # Toggle fuel damage surprise events
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
        record_every_seconds=1,
        recording_export_path=rec_path,
    )
    with open(scenario_path, "r", encoding="utf-8") as f:
        game.load_scenario(f.read())

    env = gymnasium.make("blade/BLADE-v0", game=game)
    observation, info = env.reset()

    # Ensure episode horizon is long enough
    duration = int(getattr(observation, "duration", 0) or 0)
    desired_steps = max(max_steps, duration + 5 if duration > 0 else max_steps)
    env = TimeLimit(env, max_episode_steps=desired_steps)

    logger.info(
        f"BLADE env ready: duration={duration}, "
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
# 7. Training Episode
# =============================================================================

def train_episode(
    trainer: PPOTrainer,
    game: Game,
    env,
    scenario_path: str,
    obs_config: ObservationConfig,
    episode_num: int,
    decision_interval: int = DECISION_INTERVAL,
    max_ticks: int = MAX_SIM_TICKS,
    recordings_dir: str = None,
) -> Dict:
    """
    Run a single training episode with MAPPO (PPO + CTDE).

    Episode 0 prints full diagnostic info. Episodes 1+ print compact runtime info.

    MAPPO flow per decision point:
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
            if verbose:
                logger.info(f"  LAUNCH: {getattr(ac, 'name', ac.id)} from airbase {ab_id}")
            observation, _, _, _, _ = env.step(launch_cmd)
            game.record_step()

    for _ in range(10):
        observation, _, _, _, _ = env.step("")
        game.record_step()

    if verbose:
        airborne = [getattr(ac, 'name', ac.id) for ac in getattr(observation, 'aircraft', [])]
        logger.info(f"  Airborne after launch: {airborne}")

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

    for tick in range(max_ticks):
        # Executor decides action for this tick
        try:
            executor_action = executor.next_action(observation, fallback_tick=tick) or ""
        except ValueError as e:
            logger.debug(f"Tick {tick}: Executor error (skipping): {e}")
            executor_action = ""

        # Check for fuel damage activation
        fuel_dmg.check_and_activate(tick)

        # Check if this is a decision point
        is_decision_tick = (tick % decision_interval == 0) and tick > 0
        rl_override_action = ""

        if is_decision_tick or tick == 0:
            # === MAPPO: Build observations for ALL agents first ===
            agent_obs_map: Dict[str, object] = {}  # agent_id → ObservationOutput
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

            # === Construct global observation (padded to MAX_AGENTS) ===
            # First N slots are actual agents (in attacking_agents order).
            # Remaining slots (up to MAX_AGENTS) are zero-padded.
            # This keeps critic input size fixed regardless of actual agent count.
            global_obs_parts = []
            for i in range(MAX_AGENTS):
                if i < len(attacking_agents):
                    aid = attacking_agents[i].id
                    if aid in agent_obs_map:
                        global_obs_parts.append(agent_obs_map[aid].vector)
                    else:
                        global_obs_parts.append(np.zeros(obs_dim, dtype=np.float32))
                else:
                    # Padding slot — no agent here
                    global_obs_parts.append(np.zeros(obs_dim, dtype=np.float32))
            global_obs = np.concatenate(global_obs_parts)

            # === Per-agent decisions ===
            for agent_obj in attacking_agents:
                agent_id = agent_obj.id
                if agent_id in returned_agents:
                    continue
                if agent_id not in agent_obs_map:
                    continue

                obs = agent_obs_map[agent_id]
                local_obs = obs.vector

                is_discovery = check_discovery(obs, partial_target_ids)
                if not is_decision_tick and not is_discovery:
                    continue

                if is_discovery:
                    logger.info(f"  Tick {tick}: DISCOVERY for {agent_id}!")

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

                # Log RL decisions
                if is_discovery or verbose:
                    action_names = {0: "NOOP", 1: "ATTACK_0", 2: "ATTACK_1", 3: "ATTACK_2", 4: "RTB"}
                    logger.info(
                        f"  Tick {tick} | {agent_id[:8]}.. | "
                        f"RL={action_names.get(rl_action, '?')} "
                        f"Oracle={action_names.get(oracle_action, '?')} "
                        f"Match={'✓' if rl_action == oracle_action else '✗'} "
                        f"Reward={reward:+.2f} "
                        f"(rl_u={rl_utility:.0f}, oracle_u={oracle_utility:.0f})"
                    )

                # RL override: ONLY on discovery events
                if is_discovery and rl_action != 0:
                    try:
                        rl_override_action = plan_edit_to_blade_action(
                            action_token=rl_action,
                            observation_output=obs,
                            scenario=observation,
                            agent_id=agent_id,
                        )
                        logger.info(
                            f"  Tick {tick}: RL OVERRIDE → '{rl_override_action[:80]}'"
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

        # Log BLADE actions
        if final_action:
            source = "RL" if rl_override_action else "EXEC"
            logger.info(f"  Tick {tick} [{source}]: '{final_action}'")

        observation, _reward, terminated, truncated, info = env.step(final_action)
        game.record_step()

        # Check which agents have returned to base (no longer airborne)
        airborne_ids = {str(getattr(ac, "id", "")) for ac in getattr(observation, "aircraft", []) or []}
        for agent_obj in attacking_agents:
            aid = str(agent_obj.id)
            if aid not in returned_agents and aid not in airborne_ids:
                returned_agents.add(aid)
                logger.info(f"  Tick {tick}: Agent {aid[:8]}.. returned to base")

        # End episode when all agents have returned to base
        if tick > 100 and len(returned_agents) == len(attacking_agents):
            logger.info(f"  All agents returned to base at tick {tick} — ending episode")
            break

        if terminated or truncated:
            logger.info(f"  Episode ended at tick {tick}: terminated={terminated}")
            break

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

        # Rename recording to include episode number
        if recordings_dir:
            rec_dir = Path(recordings_dir)
            jsonl_files = sorted(rec_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
            if jsonl_files:
                latest = jsonl_files[-1]
                new_name = rec_dir / f"episode_{episode_num + 1:03d}.jsonl"
                if latest != new_name:
                    # Don't rename if already named correctly (re-run)
                    if new_name.exists():
                        new_name.unlink()
                    latest.rename(new_name)
                logger.info(f"  Recording saved: {new_name.name}")
            else:
                logger.info(f"  Recording exported (no .jsonl found to rename)")
        else:
            logger.info(f"  Recording exported for episode {episode_num}")
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
        help="Path to scenario JSON",
    )
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--decision-interval", type=int, default=DECISION_INTERVAL,
                        help="Ticks between RL decisions")
    parser.add_argument("--max-ticks", type=int, default=MAX_SIM_TICKS,
                        help="Max simulation ticks per episode")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (PPO default: 3e-4)")
    parser.add_argument("--save-freq", type=int, default=10, help="Save checkpoint every N episodes")
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
    parser.add_argument("--min-red-airbases", type=int, default=0,
                        help="Min RED airbases per episode (when --vary-scenarios)")
    parser.add_argument("--max-red-airbases", type=int, default=3,
                        help="Max RED airbases per episode (when --vary-scenarios)")
    parser.add_argument("--vary-base", action="store_true",
                        help="Also randomize blue base position")
    parser.add_argument("--base-shift-km", type=float, default=150.0,
                        help="Max base shift radius in km")
    parser.add_argument("--extra-templates", nargs="*", default=[],
                        help="Additional scenario JSONs to extract aircraft templates from")

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
    logger.info(f"Decision interval: {args.decision_interval} ticks")
    logger.info(f"Max ticks:         {args.max_ticks}")
    logger.info(f"Max agents:        {MAX_AGENTS}")
    logger.info(f"Learning rate:     {args.lr}")
    logger.info(f"Seed:              {args.seed}")
    logger.info(f"Fuel damage:       {FUEL_DAMAGE_ENABLED}")
    logger.info(f"Output dir:        {output_dir.resolve()}")

    # --- Setup scenario generator ---
    scenario_gen = None
    if args.vary_scenarios:
        scenario_gen = ScenarioGenerator(
            base_scenario_path=args.scenario,
            extra_template_paths=args.extra_templates or None,
            output_dir=str(scenarios_dir),
        )
        logger.info(
            f"ScenarioGenerator: aircraft_pool={scenario_gen.aircraft_pool.class_names}, "
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
            ep_config = VariationConfig(
                num_aircraft=(args.min_aircraft, args.max_aircraft),
                num_facilities=(args.min_facilities, args.max_facilities),
                num_red_airbases=(args.min_red_airbases, args.max_red_airbases),
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
            decision_interval=args.decision_interval,
            max_ticks=args.max_ticks,
            recordings_dir=str(recordings_dir),
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