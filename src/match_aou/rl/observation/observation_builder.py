"""
Observation Builder - Main API

This is the main entry point for extracting observations from BLADE scenarios.

Usage:
    from match_aou.rl.observation import build_observation_vector, ObservationConfig
    
    obs = build_observation_vector(
        scenario=blade_scenario,
        agent_id="f16_01",
        current_plan=execution_time_to_actions,
        current_time=100
    )
    
    # obs.vector is np.array of shape (21,)
"""

from typing import Dict, List
import numpy as np

from .observation_types import ObservationOutput
from .config import ObservationConfig
from .self_features import compute_self_features
from .target_extraction import extract_visible_targets, select_topk_targets
from .plan_parsing import mark_targets_in_plan
from ..shared_utils import nm_to_km

# Import Location from project (assuming it's in the path)
# If not available during testing, we'll use haversine directly
try:
    from match_aou.models import Location
except ImportError:
    # Fallback: create simple Location class
    class Location:
        def __init__(self, latitude, longitude, altitude=0):
            self.latitude = latitude
            self.longitude = longitude
            self.altitude = altitude


def build_observation_vector(
    scenario,  # BLADE Scenario object
    agent_id: str,
    current_plan,  # List of (task_idx, step_idx, level) tuples from solution
    current_time: int,
    config: ObservationConfig = None,
    tasks=None,  # Optional: List[Task] for extracting target IDs from actions
    solution=None  # Optional: Full solution dict for checking other agents' plans
) -> ObservationOutput:
    """
    Build complete observation vector from BLADE scenario.

    This is the main entry point for observation extraction.

    Args:
        scenario: BLADE Scenario object (from Game._get_observation())
        agent_id: Agent's unique ID (e.g., "f16_01")
        current_plan: Agent's task assignments from MATCH-AOU solution
            Format: List of (task_idx, step_idx, level_order) tuples
            Example: [(0, 0, 0), (1, 0, 1), (1, 1, 1)]
        current_time: Current simulation time (tick)
            Can be obtained from: scenario.current_time - scenario.start_time
        config: Optional ObservationConfig (uses default if None)
        tasks: Optional List[Task] - enables extracting target IDs from action strings
            If provided, improves accuracy of is_in_plan and next_step_is_attack
        solution: Optional full solution dict {agent_id: [(task, step, level), ...]}
            If provided, enables checking if other agents are attacking same targets

    Returns:
        ObservationOutput with:
            - vector: np.array of shape (30,) with values in [0, 1]
                [24 existing features + 6 plan context features]
            - self_state: SelfState object (6 features)
            - targets: List of TargetInfo objects (up to K)
            - agent_id: Agent ID
            - current_time: Simulation time

    Raises:
        ValueError: If agent not found in scenario
        ValueError: If inputs are invalid

    Example:
        >>> config = ObservationConfig(top_k=3)
        >>> plan = [(0, 0, 0), (1, 0, 1)]  # Task assignments
        >>> obs = build_observation_vector(
        ...     scenario, "f16_01", plan, 100, config,
        ...     tasks=tasks, solution=solution
        ... )
        >>> print(obs.vector.shape)
        (30,)
        >>> print(obs.self_state.fuel_norm)
        0.75
        >>> print(len(obs.targets))
        3
    """
    # Use default config if not provided
    if config is None:
        config = ObservationConfig()

    # Validate inputs
    if not agent_id:
        raise ValueError("agent_id cannot be empty")

    if current_plan is None:
        current_plan = []

    # Get agent from scenario
    if not hasattr(scenario, 'get_aircraft'):
        raise ValueError("Scenario does not have get_aircraft method")

    aircraft = scenario.get_aircraft(agent_id)

    if aircraft is None:
        raise ValueError(f"Aircraft {agent_id} not found in scenario")

    # === Step 1: Extract self features (6) ===
    self_state = compute_self_features(
        aircraft=aircraft,
        current_plan=current_plan,
        current_time=current_time,
        scenario=scenario,
        config=config,
        tasks=tasks  # Pass tasks for better attack detection
    )

    # === Step 2: Get aircraft radar range for target detection ===
    # Use aircraft's detection range (radar) instead of weapon range
    # This avoids detecting targets 3000 NM away with long-range missiles
    # when aircraft radar is only 120 NM
    aircraft_radar_range_nm = getattr(aircraft, 'range', None)

    if aircraft_radar_range_nm is not None and aircraft_radar_range_nm > 0:
        detection_range_km = nm_to_km(aircraft_radar_range_nm)
    else:
        # Fallback to weapon range if radar range not available
        weapon = None
        if hasattr(aircraft, 'get_weapon_with_highest_engagement_range'):
            weapon = aircraft.get_weapon_with_highest_engagement_range()

        detection_range_km = (
            nm_to_km(weapon.get_engagement_range())
            if weapon else config.min_weapon_range_km
        )

    # For normalization purposes, we still need weapon range
    weapon = None
    if hasattr(aircraft, 'get_weapon_with_highest_engagement_range'):
        weapon = aircraft.get_weapon_with_highest_engagement_range()

    weapon_range_km = (
        nm_to_km(weapon.get_engagement_range())
        if weapon else config.min_weapon_range_km
    )

    # === Step 3: Extract visible targets ===
    agent_location = Location(
        aircraft.latitude,
        aircraft.longitude,
        getattr(aircraft, 'altitude', 0)
    )

    agent_side = getattr(aircraft, 'side_id', None)
    if not agent_side:
        raise ValueError(f"Aircraft {agent_id} has no side_id")

    visible_targets = extract_visible_targets(
        scenario=scenario,
        agent_location=agent_location,
        agent_side=agent_side,
        weapon_range_km=detection_range_km  # Use radar range for detection
    )

    # === Step 4: Mark which targets are in plan ===
    mark_targets_in_plan(
        targets=visible_targets,
        plan=current_plan,
        agent_id=agent_id,
        tasks=tasks,
        solution=solution
    )
    
    # === Step 5: Select top-K and pad ===
    topk_targets = select_topk_targets(visible_targets, config)
    
    # === Step 6: Build observation vector ===
    vector_parts = [self_state.to_array()]
    
    for target in topk_targets:
        vector_parts.append(target.to_array(weapon_range_km))
    
    # === Step 6.5: Add plan context features (NEW) ===
    # Only compute if we have tasks and solution (full context available)
    if tasks is not None and solution is not None:
        from .plan_context import compute_plan_context_features
        
        try:
            plan_context = compute_plan_context_features(
                aircraft=aircraft,
                agent_id=agent_id,
                current_plan=current_plan,
                solution=solution,
                tasks=tasks,
                scenario=scenario,
                visible_targets=topk_targets,
                current_time=current_time
            )
            vector_parts.append(plan_context)  # Add [6] features
        except Exception as e:
            # Fallback: if plan context computation fails, use zeros
            import warnings
            warnings.warn(f"Plan context computation failed: {e}. Using zeros.")
            vector_parts.append(np.zeros(6, dtype=np.float32))
    else:
        # No tasks/solution provided - use zeros for plan context
        vector_parts.append(np.zeros(6, dtype=np.float32))
    
    observation_vector = np.concatenate(vector_parts, dtype=np.float32)
    
    # === Step 7: Create output ===
    output = ObservationOutput(
        vector=observation_vector,
        self_state=self_state,
        targets=topk_targets,
        agent_id=agent_id,
        current_time=current_time
    )
    
    # === Step 8: Validate ===
    output.validate(max_targets=config.top_k)
    
    return output
