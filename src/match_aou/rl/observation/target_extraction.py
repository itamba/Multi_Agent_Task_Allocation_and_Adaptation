"""
Target Extraction and Filtering Module

Find all enemy targets within weapon range and select top-K closest.

Note: Uses BLADE's built-in methods where available:
- scenario.is_hostile() for enemy detection
- unit.get_weapon_with_highest_engagement_range() for threat assessment
- weapon.get_engagement_range() for range calculations
- scenario.weapons list for engagement counting

This module intentionally uses simple distance/range checks instead of
BLADE's engine functions (is_threat_detected, weapon_can_engage_target)
for performance and modularity reasons.
"""

from typing import List

from .observation_types import TargetInfo
from .config import ObservationConfig
from ..shared_utils import nm_to_km, haversine_distance


def extract_visible_targets(
    scenario,  # BLADE Scenario
    agent_location,  # Location object
    agent_side: str,
    weapon_range_km: float
) -> List[TargetInfo]:
    """
    Find all enemy targets within weapon range.
    
    Searches:
        - Aircraft
        - Ships
        - Facilities (SAMs, command centers)
        - Airbases
    
    Args:
        scenario: BLADE Scenario object
        agent_location: Agent's current position (Location object)
        agent_side: Agent's side_id (for filtering enemies)
        weapon_range_km: Max engagement range
    
    Returns:
        List of TargetInfo objects (unsorted, unfiltered by K)
    """
    targets = []
    
    # Search aircraft
    if hasattr(scenario, 'aircraft'):
        for ac in scenario.aircraft:
            if _is_enemy(ac.side_id, agent_side, scenario):
                info = _create_target_info(
                    unit=ac,
                    agent_location=agent_location,
                    weapon_range_km=weapon_range_km,
                    is_dynamic=True,
                    scenario=scenario  # NEW
                )
                if info.distance_km <= weapon_range_km:
                    targets.append(info)
    
    # Search ships
    if hasattr(scenario, 'ships'):
        for ship in scenario.ships:
            if _is_enemy(ship.side_id, agent_side, scenario):
                info = _create_target_info(
                    unit=ship,
                    agent_location=agent_location,
                    weapon_range_km=weapon_range_km,
                    is_dynamic=True,
                    scenario=scenario  # NEW
                )
                if info.distance_km <= weapon_range_km:
                    targets.append(info)
    
    # Search facilities (SAMs, command centers)
    if hasattr(scenario, 'facilities'):
        for facility in scenario.facilities:
            if _is_enemy(facility.side_id, agent_side, scenario):
                info = _create_target_info(
                    unit=facility,
                    agent_location=agent_location,
                    weapon_range_km=weapon_range_km,
                    is_dynamic=False,
                    scenario=scenario  # NEW
                )
                if info.distance_km <= weapon_range_km:
                    targets.append(info)
    
    # Search airbases
    if hasattr(scenario, 'airbases'):
        for airbase in scenario.airbases:
            if _is_enemy(airbase.side_id, agent_side, scenario):
                info = _create_target_info(
                    unit=airbase,
                    agent_location=agent_location,
                    weapon_range_km=weapon_range_km,
                    is_dynamic=False,
                    scenario=scenario  # NEW
                )
                if info.distance_km <= weapon_range_km:
                    targets.append(info)
    
    return targets


def _is_enemy(target_side: str, agent_side: str, scenario) -> bool:
    """
    Check if target is hostile to agent.
    
    Args:
        target_side: Target's side_id
        agent_side: Agent's side_id
        scenario: BLADE Scenario
    
    Returns:
        True if hostile
    """
    if not hasattr(scenario, 'is_hostile'):
        # Fallback: different side = enemy
        return target_side != agent_side
    
    try:
        return scenario.is_hostile(agent_side, target_side)
    except:
        # Fallback
        return target_side != agent_side


def _create_target_info(
    unit,
    agent_location,
    weapon_range_km: float,
    is_dynamic: bool,
    scenario  # NEW: needed for engagement counting
) -> TargetInfo:
    """
    Create TargetInfo from BLADE unit.
    
    Args:
        unit: BLADE unit (Aircraft, Ship, Facility, Airbase)
        agent_location: Agent's Location object
        weapon_range_km: For range checking
        is_dynamic: True if unit is mobile (aircraft/ship)
        scenario: BLADE Scenario (for engagement counting)
    
    Returns:
        TargetInfo object
    """
    # Calculate distance using shared utility
    target_pos = (unit.latitude, unit.longitude)
    agent_pos = (agent_location.latitude, agent_location.longitude)
    
    distance_km = haversine_distance(agent_pos, target_pos)
    
    # Check if target is a threat
    is_threat = _check_is_threat(unit, distance_km)
    
    # Count weapons already targeting this unit
    already_engaged = _count_engaged_weapons(scenario, unit.id)
    
    return TargetInfo(
        exists=True,
        id=unit.id,
        latitude=unit.latitude,
        longitude=unit.longitude,
        distance_km=distance_km,
        is_threat=is_threat,
        is_dynamic=is_dynamic,
        is_in_plan=False,  # Will be set by plan_parsing
        side_id=unit.side_id,
        already_engaged=already_engaged  # NEW
    )


def _check_is_threat(unit, distance_km: float) -> bool:
    """
    Check if unit can engage agent at current distance.
    
    Simple range check: does unit have weapons that can reach agent?
    
    Args:
        unit: BLADE unit
        distance_km: Distance to agent
    
    Returns:
        True if unit is a threat
    """
    # Check if unit has get_weapon_with_highest_engagement_range method
    if not hasattr(unit, 'get_weapon_with_highest_engagement_range'):
        return False
    
    try:
        weapon = unit.get_weapon_with_highest_engagement_range()
        
        if weapon is None:
            return False
        
        # Get weapon range in km
        threat_range_nm = weapon.get_engagement_range()
        threat_range_km = nm_to_km(threat_range_nm)
        
        # Check if we're in range
        return distance_km <= threat_range_km
    
    except:
        return False


def _count_engaged_weapons(scenario, target_id: str) -> int:
    """
    Count how many weapons are currently targeting this unit.
    
    Uses BLADE's scenario.weapons list to count active engagements.
    This helps avoid over-assigning resources to already-engaged targets.
    
    Args:
        scenario: BLADE Scenario object
        target_id: Target unit ID
    
    Returns:
        Number of weapons currently targeting this unit (0 if none)
    """
    if not hasattr(scenario, 'weapons'):
        return 0
    
    count = 0
    try:
        for weapon in scenario.weapons:
            if hasattr(weapon, 'target_id') and weapon.target_id == target_id:
                count += 1
    except:
        pass
    
    return count


# =============================================================================
# Target Filtering
# =============================================================================

def select_topk_targets(
    targets: List[TargetInfo],
    config: ObservationConfig
) -> List[TargetInfo]:
    """
    Sort targets by distance and select top-K closest.
    
    If fewer than K targets exist, pad with empty targets.
    This ensures the observation vector always has a fixed size.
    
    Args:
        targets: List of visible targets (from extract_visible_targets)
        config: ObservationConfig (for top_k parameter)
    
    Returns:
        List of exactly K targets (padded if needed)
    
    Example:
        >>> targets = extract_visible_targets(scenario, agent_loc, "blue", 100.0)
        >>> config = ObservationConfig(top_k=3)
        >>> topk = select_topk_targets(targets, config)
        >>> len(topk)
        3
    """
    # Sort by distance (closest first)
    sorted_targets = sorted(targets, key=lambda t: t.distance_km)
    
    # Take top-K
    topk = sorted_targets[:config.top_k]
    
    # Pad if needed
    while len(topk) < config.top_k:
        topk.append(TargetInfo.create_empty())
    
    return topk
