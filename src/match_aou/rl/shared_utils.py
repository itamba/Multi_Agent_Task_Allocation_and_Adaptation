"""
Shared numeric helpers for the graph-RL modules.

Two small pure functions, kept here so the layers that need them do not each carry
their own copy:

- ``haversine_distance((lat, lon), (lat, lon))`` -- great-circle distance in kilometres,
  via the ``haversine`` package when it is installed and an inline fallback otherwise.
- ``clip_to_01(value)`` -- clamp a value into ``[0, 1]``.

Both are consumed by ``rl/observation/graph_builder.py`` when it normalizes task and
agent features. Nothing here imports BLADE, gymnasium, torch or the solver.
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
