"""
Observation-Specific Utilities

Helper functions specific to observation extraction.

For common utilities (distance, normalization), see ../shared_utils.py

Functions:
    - extract_target_id_from_action: Parse target ID from action string
    - is_attack_action: Identify attack actions in action strings
    - calculate_travel_time_hours: Calculate travel time
    - calculate_fuel_needed: Calculate fuel consumption for distance
"""

import re
from typing import Optional, Tuple


def extract_target_id_from_action(action: str) -> Optional[str]:
    """
    Parse action string to extract target ID.
    
    Assumes format: func('agent_id', 'target_id', ...)
    
    Args:
        action: Action string (e.g., "handle_aircraft_attack('f16_01', 'target_01', 'aim_120', 2)")
    
    Returns:
        Target ID or None if not found
    
    Examples:
        >>> extract_target_id_from_action("handle_aircraft_attack('f16_01', 'target_01', 'aim_120', 2)")
        'target_01'
        >>> extract_target_id_from_action("move_aircraft('f16_01', [[35.0, 40.0]])")
        None
    """
    # Match pattern: func('arg1', 'arg2', ...)
    # Target is the second quoted argument
    pattern = r"['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]"
    match = re.search(pattern, action)
    
    if match:
        return match.group(2)  # Second argument is target
    
    return None


def calculate_travel_time_hours(
    distance_km: float,
    speed_knots: float
) -> float:
    """
    Calculate travel time in hours.
    
    Args:
        distance_km: Distance in kilometers
        speed_knots: Speed in knots
    
    Returns:
        Travel time in hours
    """
    if speed_knots <= 0:
        return float('inf')
    
    # Convert speed to km/h
    speed_kmh = speed_knots * 1.852
    
    return distance_km / speed_kmh


def calculate_fuel_needed(
    distance_km: float,
    speed_knots: float,
    fuel_rate_lbs_per_hour: float
) -> float:
    """
    Calculate fuel needed for a distance.
    
    Args:
        distance_km: Distance in kilometers
        speed_knots: Speed in knots
        fuel_rate_lbs_per_hour: Fuel consumption rate (lbs/hr)
    
    Returns:
        Fuel needed in pounds
    """
    travel_time_hours = calculate_travel_time_hours(distance_km, speed_knots)
    
    if travel_time_hours == float('inf'):
        return float('inf')
    
    return travel_time_hours * fuel_rate_lbs_per_hour


def is_attack_action(action: str) -> bool:
    """
    Check if action string represents an attack action.
    
    Identifies attack actions by keywords:
    - handle_aircraft_attack
    - handle_ship_attack
    - launch_weapon
    
    This is the shared implementation used across all observation modules.
    
    Args:
        action: Action string (e.g., "handle_aircraft_attack('f16_01', 'target_01', ...)")
    
    Returns:
        True if action is an attack, False otherwise
    
    Example:
        >>> is_attack_action("handle_aircraft_attack('f16_01', 'target_01', 'aim_120', 2)")
        True
        >>> is_attack_action("move_aircraft('f16_01', [[35.0, 40.0]])")
        False
    """
    attack_keywords = [
        "handle_aircraft_attack",
        "handle_ship_attack",
        "launch_weapon"
    ]
    
    return any(keyword in action for keyword in attack_keywords)
