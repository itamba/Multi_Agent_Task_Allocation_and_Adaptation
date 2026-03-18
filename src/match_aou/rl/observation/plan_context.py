"""
Plan Context Features
=====================

Extract 6 features about the agent's plan and execution state.

All features use ONLY local information:
- Original plan (known at start)
- Agent's current state (radar, fuel, position)
- Agent's own execution progress

NO global information (other agents' states, target destruction status).

Features:
1. fuel_margin_for_plan - Can I afford to deviate? (fuel surplus/deficit)
2. unassigned_targets_nearby - Are there new opportunities?
3. my_remaining_tasks_ratio - How much work do I have left?
4. next_target_visible - Is my next planned target still there?
5. target_coordination - Am I coordinating with others?
6. time_until_next_attack - When is my next attack?
"""

from typing import List, Tuple, Optional, Set
import numpy as np

from ..shared_utils import nm_to_km, clip_to_01, haversine_distance
from .observation_utils import extract_target_id_from_action, is_attack_action, calculate_fuel_needed

# Type aliases
Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level)
Solution = dict  # {agent_id: [Assignment, ...]}


def compute_plan_context_features(
    aircraft,                        # BLADE Aircraft object
    agent_id: str,                   # This agent's ID
    current_plan: List[Assignment],  # This agent's assignments
    solution: Solution,              # Full solution (all agents)
    tasks: List,                     # All Task objects
    scenario,                        # BLADE Scenario object
    visible_targets: List,           # TargetInfo list from observation
    current_time: int                # Current simulation tick
) -> np.ndarray:
    """
    Compute 6 plan context features.
    
    All features are normalized to [0, 1] range.
    
    Args:
        aircraft: BLADE Aircraft object
        agent_id: This agent's ID
        current_plan: This agent's task assignments [(task_idx, step_idx, level), ...]
        solution: Full solution dict {agent_id: [assignments]}
        tasks: List of all Task objects
        scenario: BLADE Scenario object
        visible_targets: List of TargetInfo from observation builder
        current_time: Current simulation time (tick)
    
    Returns:
        np.ndarray of shape (6,) with values in [0, 1]
        [fuel_margin, unassigned_ratio, remaining_ratio, 
         next_visible, coordination, time_to_attack]
    """
    
    # Feature 1: Fuel margin for completing plan
    fuel_margin = _compute_fuel_margin(
        aircraft=aircraft,
        current_plan=current_plan,
        tasks=tasks,
        scenario=scenario
    )
    
    # Feature 2: Unassigned targets nearby
    unassigned_ratio = _compute_unassigned_targets_ratio(
        visible_targets=visible_targets,
        solution=solution,
        tasks=tasks
    )
    
    # Feature 3: My remaining tasks ratio
    remaining_ratio = _compute_remaining_tasks_ratio(
        current_plan=current_plan,
        current_time=current_time
    )
    
    # Feature 4: Next planned target visible
    next_visible = _check_next_target_visible(
        aircraft=aircraft,
        current_plan=current_plan,
        tasks=tasks,
        visible_targets=visible_targets
    )
    
    # Feature 5: Target coordination (shared targets)
    coordination = _compute_coordination_load(
        agent_id=agent_id,
        current_plan=current_plan,
        solution=solution,
        tasks=tasks
    )
    
    # Feature 6: Time until next attack
    time_to_attack = _compute_time_until_attack(
        current_plan=current_plan,
        tasks=tasks,
        current_time=current_time
    )
    
    # Build feature vector
    features = np.array([
        fuel_margin,
        unassigned_ratio,
        remaining_ratio,
        next_visible,
        coordination,
        time_to_attack
    ], dtype=np.float32)
    
    # Validate: all should be in [0, 1]
    # Note: fuel_margin can be negative, we'll handle this specially
    features = np.clip(features, 0.0, 1.0)
    
    return features


# =============================================================================
# Feature 1: Fuel Margin for Plan
# =============================================================================

def _compute_fuel_margin(
    aircraft,
    current_plan: List[Assignment],
    tasks: List,
    scenario
) -> float:
    """
    Compute fuel margin: (current_fuel - fuel_needed_for_plan) / max_fuel
    
    Negative value = can't complete plan without refueling
    Positive value = have surplus fuel
    
    Returns:
        Float in range (can be negative), then normalized to [0, 1]
        where 0.5 = exactly enough fuel, <0.5 = deficit, >0.5 = surplus
    """
    if not current_plan or not tasks:
        return 0.5  # No plan = neutral
    
    if aircraft.max_fuel <= 0:
        return 0.0  # No fuel capacity = problem
    
    # Calculate fuel needed for remaining plan
    fuel_needed = _estimate_fuel_for_plan(
        aircraft=aircraft,
        plan=current_plan,
        tasks=tasks,
        scenario=scenario
    )
    
    # Compute margin
    current_fuel = aircraft.current_fuel
    margin = (current_fuel - fuel_needed) / aircraft.max_fuel
    
    # Normalize: margin of -1.0 to +1.0 → map to [0, 1]
    # -1.0 → 0.0 (big deficit)
    #  0.0 → 0.5 (exactly enough)
    # +1.0 → 1.0 (lots of surplus)
    normalized_margin = (margin + 1.0) / 2.0
    
    return clip_to_01(normalized_margin)


def _estimate_fuel_for_plan(
    aircraft,
    plan: List[Assignment],
    tasks: List,
    scenario
) -> float:
    """
    Estimate total fuel needed to complete remaining plan.
    
    This is approximate since we don't know exact execution state.
    We sum distances between waypoints and multiply by fuel rate.
    
    Returns:
        Estimated fuel needed (in same units as aircraft.current_fuel)
    """
    if not plan:
        return 0.0
    
    total_fuel = 0.0
    current_pos = (aircraft.latitude, aircraft.longitude)
    
    # Get aircraft fuel consumption rate
    fuel_rate = getattr(aircraft, 'fuel_rate', 1.0)
    speed = getattr(aircraft, 'speed', 1.0)
    
    if speed <= 0:
        speed = 1.0  # Avoid division by zero
    
    # Iterate through plan steps
    for task_idx, step_idx, level in plan:
        # Validate indices
        if not (0 <= task_idx < len(tasks)):
            continue
        
        task = tasks[task_idx]
        
        if not (0 <= step_idx < len(task.steps)):
            continue
        
        step = task.steps[step_idx]
        
        # Get step location
        step_location = getattr(step, 'location', None)
        if step_location is None:
            continue
        
        next_pos = (step_location.latitude, step_location.longitude)
        
        # Calculate distance using shared utility
        distance_km = haversine_distance(current_pos, next_pos)
        
        # Calculate fuel for this leg
        leg_fuel = calculate_fuel_needed(distance_km, speed, fuel_rate)
        total_fuel += leg_fuel
        
        # Update position for next iteration
        current_pos = next_pos
    
    # Add fuel for return to base (safety margin)
    home_base = _get_home_base(aircraft, scenario)
    if home_base:
        base_pos = (home_base.latitude, home_base.longitude)
        rtb_distance = haversine_distance(current_pos, base_pos)
        rtb_fuel = calculate_fuel_needed(rtb_distance, speed, fuel_rate)
        total_fuel += rtb_fuel
    
    return total_fuel


# =============================================================================
# Feature 2: Unassigned Targets Nearby
# =============================================================================

def _compute_unassigned_targets_ratio(
    visible_targets: List,
    solution: Solution,
    tasks: List
) -> float:
    """
    Ratio of visible targets NOT in any agent's plan.
    
    High value = many opportunities
    Low value = all targets assigned
    
    Returns:
        Float in [0, 1]: unassigned_count / total_visible_count
    """
    if not visible_targets:
        return 0.0  # No targets visible
    
    # Count existing targets (not padding)
    existing_targets = [t for t in visible_targets if t.exists]
    
    if not existing_targets:
        return 0.0
    
    # Get all target IDs in solution
    planned_target_ids = _extract_all_planned_targets(solution, tasks)
    
    # Count unassigned visible targets
    unassigned_count = 0
    for target in existing_targets:
        if target.id not in planned_target_ids:
            unassigned_count += 1
    
    # Return ratio
    return unassigned_count / len(existing_targets)


def _extract_all_planned_targets(solution: Solution, tasks: List) -> Set[str]:
    """
    Extract all target IDs that appear in ANY agent's plan.
    
    Returns:
        Set of target IDs
    """
    target_ids = set()
    
    if not solution or not tasks:
        return target_ids
    
    for agent_id, assignments in solution.items():
        for task_idx, step_idx, level in assignments:
            # Validate indices
            if not (0 <= task_idx < len(tasks)):
                continue
            
            task = tasks[task_idx]
            
            if not (0 <= step_idx < len(task.steps)):
                continue
            
            step = task.steps[step_idx]
            
            # Extract target ID from action string
            action = getattr(step, 'action', None)
            if action:
                target_id = extract_target_id_from_action(action)
                if target_id:
                    target_ids.add(target_id)
    
    return target_ids


# =============================================================================
# Feature 3: My Remaining Tasks Ratio
# =============================================================================

def _compute_remaining_tasks_ratio(
    current_plan: List[Assignment],
    current_time: int
) -> float:
    """
    Estimate fraction of plan remaining.
    
    Since we don't have global completion state, we use a simple heuristic:
    - If plan is empty → 0.0 (nothing remaining)
    - If plan has steps → assume proportional to time (very rough!)
    
    Better approach would track local completion, but requires executor state.
    
    Returns:
        Float in [0, 1]: 1.0 = full plan ahead, 0.0 = plan complete
    """
    if not current_plan:
        return 0.0  # No plan = nothing remaining
    
    # Heuristic: assume we complete tasks at constant rate
    # This is a placeholder - ideally we'd track which tasks are done
    
    # For now, return 0.5 (middle ground) since we can't accurately estimate
    # TODO: Improve by tracking task completion locally
    return 0.5


# =============================================================================
# Feature 4: Next Target Visible
# =============================================================================

def _check_next_target_visible(
    aircraft,
    current_plan: List[Assignment],
    tasks: List,
    visible_targets: List
) -> float:
    """
    Check if next planned target is visible in radar.
    
    Returns:
        1.0 if visible, 0.0 if not visible or no next target
    """
    if not current_plan or not tasks:
        return 0.0  # No plan
    
    # Find next attack action in plan
    next_target_id = None
    
    for task_idx, step_idx, level in current_plan:
        # Validate indices
        if not (0 <= task_idx < len(tasks)):
            continue
        
        task = tasks[task_idx]
        
        if not (0 <= step_idx < len(task.steps)):
            continue
        
        step = task.steps[step_idx]
        
        # Check if this is an attack action
        action = getattr(step, 'action', None)
        if action and is_attack_action(action):
            next_target_id = extract_target_id_from_action(action)
            if next_target_id:
                break  # Found first attack
    
    if not next_target_id:
        return 0.0  # No attack in plan
    
    # Check if target is visible
    for target in visible_targets:
        if target.exists and target.id == next_target_id:
            return 1.0  # Target visible!
    
    return 0.0  # Target not visible


# =============================================================================
# Feature 5: Target Coordination
# =============================================================================

def _compute_coordination_load(
    agent_id: str,
    current_plan: List[Assignment],
    solution: Solution,
    tasks: List
) -> float:
    """
    Ratio of my targets that are shared with other agents.
    
    High value = coordinating with others
    Low value = solo mission
    
    Returns:
        Float in [0, 1]: shared_targets / my_total_targets
    """
    if not current_plan or not solution or not tasks:
        return 0.0
    
    # Extract my target IDs
    my_targets = _extract_targets_from_plan(current_plan, tasks)
    
    if not my_targets:
        return 0.0
    
    # Count how many are shared
    shared_count = 0
    
    for target_id in my_targets:
        # Count agents assigned to this target
        agent_count = _count_agents_for_target(target_id, solution, tasks)
        
        if agent_count > 1:  # Shared if 2+ agents
            shared_count += 1
    
    return shared_count / len(my_targets)


def _extract_targets_from_plan(
    plan: List[Assignment],
    tasks: List
) -> Set[str]:
    """Extract target IDs from a plan."""
    target_ids = set()
    
    for task_idx, step_idx, level in plan:
        if not (0 <= task_idx < len(tasks)):
            continue
        
        task = tasks[task_idx]
        
        if not (0 <= step_idx < len(task.steps)):
            continue
        
        step = task.steps[step_idx]
        
        action = getattr(step, 'action', None)
        if action:
            target_id = extract_target_id_from_action(action)
            if target_id:
                target_ids.add(target_id)
    
    return target_ids


def _count_agents_for_target(
    target_id: str,
    solution: Solution,
    tasks: List
) -> int:
    """Count how many agents are assigned to a specific target."""
    agent_count = 0
    
    for agent_id, assignments in solution.items():
        for task_idx, step_idx, level in assignments:
            if not (0 <= task_idx < len(tasks)):
                continue
            
            task = tasks[task_idx]
            
            if not (0 <= step_idx < len(task.steps)):
                continue
            
            step = task.steps[step_idx]
            
            action = getattr(step, 'action', None)
            if action:
                tid = extract_target_id_from_action(action)
                if tid == target_id:
                    agent_count += 1
                    break  # Count each agent once per target
    
    return agent_count


# =============================================================================
# Feature 6: Time Until Next Attack
# =============================================================================

def _compute_time_until_attack(
    current_plan: List[Assignment],
    tasks: List,
    current_time: int
) -> float:
    """
    Estimate steps until next attack action.
    
    Returns:
        Float in [0, 1]: 0.0 = attacking now, 1.0 = far in future
    """
    if not current_plan or not tasks:
        return 1.0  # No plan = no attack soon
    
    # Count steps until first attack
    steps_to_attack = 0
    found_attack = False
    
    for task_idx, step_idx, level in current_plan:
        if not (0 <= task_idx < len(tasks)):
            steps_to_attack += 1
            continue
        
        task = tasks[task_idx]
        
        if not (0 <= step_idx < len(task.steps)):
            steps_to_attack += 1
            continue
        
        step = task.steps[step_idx]
        
        action = getattr(step, 'action', None)
        if action and is_attack_action(action):
            found_attack = True
            break
        
        steps_to_attack += 1
    
    if not found_attack:
        return 1.0  # No attack in plan
    
    # Normalize by max expected steps (e.g., 10)
    max_steps = 10
    normalized = steps_to_attack / max_steps
    
    return clip_to_01(normalized)


# =============================================================================
# Helper Functions - Home Base Lookup
# =============================================================================

def _get_home_base(aircraft, scenario):
    """Get aircraft's home base from scenario."""
    home_base_id = getattr(aircraft, 'home_base_id', None)
    
    if not home_base_id:
        return None
    
    # Try airbase first
    if hasattr(scenario, 'get_airbase'):
        base = scenario.get_airbase(home_base_id)
        if base:
            return base
    
    # Try ship (carrier)
    if hasattr(scenario, 'get_ship'):
        base = scenario.get_ship(home_base_id)
        if base:
            return base
    
    return None
