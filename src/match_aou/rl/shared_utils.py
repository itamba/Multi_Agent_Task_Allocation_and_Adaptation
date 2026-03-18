"""
Shared Utilities for RL Module

Common functions used across observation, action, and plan_edit modules.

This module provides utilities that are needed by multiple RL components:
- Distance calculations (haversine)
- Unit conversions (nautical miles to kilometers)
- Value normalization and clipping

All RL submodules should import from here rather than duplicating code.
"""

import math
from typing import Tuple


def haversine_distance(
    coord1: Tuple[float, float],
    coord2: Tuple[float, float]
) -> float:
    """
    Calculate great-circle distance between two (lat, lon) points.
    
    Uses haversine library if available, otherwise falls back to
    manual calculation. This is the single source of truth for
    distance calculations across the RL module.
    
    Args:
        coord1: (latitude, longitude) tuple for point 1
        coord2: (latitude, longitude) tuple for point 2
    
    Returns:
        Distance in kilometers
    
    Example:
        >>> haversine_distance((35.0, 40.0), (35.1, 40.1))
        13.47
    """
    try:
        from haversine import haversine
        return haversine(coord1, coord2)
    except ImportError:
        # Fallback: manual haversine implementation
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c


def nm_to_km(nautical_miles: float) -> float:
    """
    Convert nautical miles to kilometers.
    
    Standard conversion: 1 NM = 1.852 km
    
    Args:
        nautical_miles: Distance in nautical miles
    
    Returns:
        Distance in kilometers
    
    Example:
        >>> nm_to_km(100)
        185.2
    """
    return nautical_miles * 1.852


def clip_to_01(value: float) -> float:
    """
    Clip value to [0, 1] range.
    
    Useful for ensuring normalized values stay in bounds.
    
    Args:
        value: Value to clip
    
    Returns:
        Value clipped to [0, 1]
    
    Example:
        >>> clip_to_01(1.5)
        1.0
        >>> clip_to_01(-0.3)
        0.0
    """
    return max(0.0, min(1.0, float(value)))


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to [0, 1] range.
    
    Maps a value from [min_val, max_val] to [0, 1].
    Automatically clips result to [0, 1].
    
    Args:
        value: Value to normalize
        min_val: Minimum possible value
        max_val: Maximum possible value
    
    Returns:
        Normalized value in [0, 1]
    
    Example:
        >>> normalize_value(50, 0, 100)
        0.5
        >>> normalize_value(150, 0, 100)
        1.0
    """
    if max_val <= min_val:
        return 0.0
    
    normalized = (value - min_val) / (max_val - min_val)
    return clip_to_01(normalized)
