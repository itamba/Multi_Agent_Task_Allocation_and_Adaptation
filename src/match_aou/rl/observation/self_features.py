"""
Self Features Extraction

Extract 6 self features from agent state:
    1. fuel_norm - Current fuel / max fuel
    2. has_weapon - Binary flag for available weapons
    3. dist_to_next_step_norm - Distance to next waypoint / weapon range
    4. next_step_is_attack - Is next action an attack?
    5. rtb_possible - Can safely return to base?
    6. plan_progress - Steps completed / total steps

Note: Uses BLADE's built-in methods:
- aircraft.current_fuel / aircraft.max_fuel
- aircraft.get_weapon_with_highest_engagement_range()
- scenario.get_airbase() for home base lookup

Future enhancement: Could use game.get_fuel_needed_to_return_to_base()
for more accurate RTB calculation when Game object is available.
"""

from typing import Dict, List
import re

from .observation_types import SelfState
from .config import ObservationConfig
from ..shared_utils import clip_to_01, nm_to_km, haversine_distance
from .observation_utils import calculate_fuel_needed, is_attack_action

def compute_self_features(
    aircraft,  # BLADE Aircraft object
    current_plan,  # List of (task_idx, step_idx, level) tuples
    current_time: int,
    scenario,  # BLADE Scenario object
    config: ObservationConfig,
    tasks=None  # Optional: List[Task] for extracting action info
) -> SelfState:
    """
    Extract 6 self features from agent state.

    Args:
        aircraft: BLADE Aircraft object
        current_plan: Agent's task assignments (list of tuples)
            Format: [(task_idx, step_idx, level_order), ...]
        current_time: Current simulation time (tick)
        scenario: BLADE Scenario object
        config: ObservationConfig

    Returns:
        SelfState with 6 normalized features
    """
    # Feature 1: Fuel
    fuel_norm = _compute_fuel_norm(aircraft)

    # Feature 2: Has weapon
    has_weapon = _check_has_weapon(aircraft)

    # Get weapon range for normalization
    weapon_range_km = _get_weapon_range_km(aircraft, config)

    # Feature 3: Distance to next step
    dist_to_next_step_norm = _compute_next_step_distance_norm(
        aircraft, weapon_range_km
    )

    # Feature 4: Next step is attack?
    # Check if current task/step for this agent is an attack
    next_step_is_attack = _is_current_step_attack(
        aircraft_id=aircraft.id,
        current_plan=current_plan,
        tasks=tasks
    )

    # Feature 5: RTB possible?
    rtb_possible = _check_rtb_feasible(aircraft, scenario, config)

    # Feature 6: Plan progress
    # Simplified: based on number of assigned tasks vs total
    plan_progress = _compute_plan_progress_simple(current_plan)

    return SelfState(
        fuel_norm=clip_to_01(fuel_norm),
        has_weapon=float(has_weapon),
        dist_to_next_step_norm=clip_to_01(dist_to_next_step_norm),
        next_step_is_attack=float(next_step_is_attack),
        rtb_possible=float(rtb_possible),
        plan_progress=clip_to_01(plan_progress)
    )


def _compute_fuel_norm(aircraft) -> float:
    """
    Compute normalized fuel: current_fuel / max_fuel.

    Args:
        aircraft: BLADE Aircraft object

    Returns:
        Fuel ratio [0, 1]
    """
    if aircraft.max_fuel <= 0:
        return 0.0

    return aircraft.current_fuel / aircraft.max_fuel


def _check_has_weapon(aircraft) -> bool:
    """
    Check if aircraft has available weapons.

    Args:
        aircraft: BLADE Aircraft object

    Returns:
        True if has weapons with quantity > 0
    """
    if not hasattr(aircraft, 'weapons') or not aircraft.weapons:
        return False

    return any(w.current_quantity > 0 for w in aircraft.weapons)


def _get_weapon_range_km(aircraft, config: ObservationConfig) -> float:
    """
    Get weapon engagement range in kilometers.

    Args:
        aircraft: BLADE Aircraft object
        config: ObservationConfig (for fallback range)

    Returns:
        Weapon range in km
    """
    if not hasattr(aircraft, 'get_weapon_with_highest_engagement_range'):
        return config.min_weapon_range_km

    weapon = aircraft.get_weapon_with_highest_engagement_range()

    if weapon is None:
        return config.min_weapon_range_km

    range_nm = weapon.get_engagement_range()
    return nm_to_km(range_nm)


def _compute_next_step_distance_norm(
    aircraft,
    weapon_range_km: float
) -> float:
    """
    Compute normalized distance to next waypoint.

    Uses aircraft.route (current waypoint list) instead of plan dict.

    Args:
        aircraft: BLADE Aircraft object
        weapon_range_km: For normalization

    Returns:
        Normalized distance [0, 1]
    """
    # Get next waypoint from route
    if hasattr(aircraft, 'route') and aircraft.route:
        next_waypoint = aircraft.route[0]

        # Calculate distance using shared utility
        current_pos = (aircraft.latitude, aircraft.longitude)
        next_pos = (next_waypoint[0], next_waypoint[1])

        distance_km = haversine_distance(current_pos, next_pos)

        # Normalize by weapon range
        if weapon_range_km > 0:
            return distance_km / weapon_range_km
        else:
            return 0.0

    # No route - return 0
    return 0.0


def _is_current_step_attack(
    aircraft_id: str,
    current_plan,  # List of (task_idx, step_idx, level) tuples
    tasks  # List[Task]
) -> bool:
    """
    Check if this agent's current/next task step is an attack.

    Looks at the agent's assigned steps and checks if any contain
    attack actions (handle_aircraft_attack, handle_ship_attack, etc.)

    This is agent-specific: only checks this agent's plan, not others.

    Args:
        aircraft_id: This agent's ID
        current_plan: This agent's task assignments
        tasks: All task objects (to extract action strings)

    Returns:
        True if any assigned step is an attack
    """
    if not current_plan or not tasks:
        return False

    # Check each assigned step for this agent
    for task_idx, step_idx, level in current_plan:
        # Validate indices
        if not (0 <= task_idx < len(tasks)):
            continue

        task = tasks[task_idx]

        if not (0 <= step_idx < len(task.steps)):
            continue

        step = task.steps[step_idx]

        # Check if step has an attack action using shared utility
        action = getattr(step, 'action', None)
        if action and is_attack_action(action):
            return True

    return False


def _check_rtb_feasible(
    aircraft,
    scenario,
    config: ObservationConfig
) -> bool:
    """
    Check if aircraft can safely return to base.
    
    Simplified calculation:
    - Find home base
    - Calculate distance
    - Check if fuel > distance_fuel_cost * margin
    
    Args:
        aircraft: BLADE Aircraft object
        scenario: BLADE Scenario object
        config: ObservationConfig (for fuel margin)
    
    Returns:
        True if RTB is feasible
    """
    # Get home base
    home_base_id = getattr(aircraft, 'home_base_id', None)
    
    if not home_base_id:
        return False
    
    # Find home base
    home_base = None
    if hasattr(scenario, 'get_airbase'):
        home_base = scenario.get_airbase(home_base_id)
    
    if home_base is None and hasattr(scenario, 'get_ship'):
        home_base = scenario.get_ship(home_base_id)
    
    if home_base is None:
        # Can't find home base - assume RTB not possible
        return False
    
    # Calculate distance to home base using shared utility
    current_pos = (aircraft.latitude, aircraft.longitude)
    base_pos = (home_base.latitude, home_base.longitude)
    
    distance_km = haversine_distance(current_pos, base_pos)
    
    # Calculate fuel needed
    fuel_needed = calculate_fuel_needed(
        distance_km,
        aircraft.speed if aircraft.speed > 0 else 1.0,  # Avoid division by zero
        aircraft.fuel_rate
    )
    
    # Check with safety margin
    return aircraft.current_fuel >= fuel_needed * config.rtb_fuel_margin


def _compute_plan_progress_simple(current_plan) -> float:
    """
    Compute a simple progress indicator.
    
    Since we don't have timestep execution info, we use a heuristic:
    - If plan is empty -> 1.0 (complete/no plan)
    - If plan has tasks -> 0.0 (not complete, still executing)
    
    This is a placeholder. In a real implementation, the executor
    could track which tasks have been completed.
    
    Args:
        current_plan: List of task assignments
    
    Returns:
        Progress ratio [0, 1]
    """
    if not current_plan or len(current_plan) == 0:
        # No plan or empty plan -> consider complete
        return 1.0
    
    # Has tasks -> assume still executing
    # This is conservative but safe
    return 0.0
