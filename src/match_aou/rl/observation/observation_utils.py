"""
Observation-Specific Utilities

Helper functions specific to observation extraction.

For common utilities (distance, normalization), see ../shared_utils.py

Functions:
    - calculate_travel_time_hours: Calculate travel time
    - calculate_fuel_needed: Calculate fuel consumption for distance
"""


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
