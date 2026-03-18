"""
Action Space Utilities

Helper functions specific to action space operations.

For common utilities (distance, normalization), see ../shared_utils.py
"""

from typing import Optional


def action_index_to_target_slot(action: int) -> Optional[int]:
    """Extract target slot from INSERT_ATTACK action.
    
    Args:
        action: Action index
        
    Returns:
        Target slot k (0, 1, 2) or None if not INSERT_ATTACK
        
    Example:
        >>> action_index_to_target_slot(1)  # INSERT_ATTACK_0
        0
        >>> action_index_to_target_slot(2)  # INSERT_ATTACK_1
        1
        >>> action_index_to_target_slot(0)  # NOOP
        None
    """
    from .action_config import ActionType
    
    if ActionType.INSERT_ATTACK_0 <= action <= ActionType.INSERT_ATTACK_2:
        return action - ActionType.INSERT_ATTACK_0
    
    return None


def is_attack_action(action: int) -> bool:
    """Check if action is an INSERT_ATTACK action.
    
    Args:
        action: Action index
        
    Returns:
        True if INSERT_ATTACK
        
    Example:
        >>> is_attack_action(1)
        True
        >>> is_attack_action(0)  # NOOP
        False
    """
    from .action_config import ActionType
    return ActionType.INSERT_ATTACK_0 <= action <= ActionType.INSERT_ATTACK_2


def is_noop_action(action: int) -> bool:
    """Check if action is NOOP.
    
    Args:
        action: Action index
        
    Returns:
        True if NOOP
        
    Example:
        >>> is_noop_action(0)
        True
    """
    from .action_config import ActionType
    return action == ActionType.NOOP


def is_rtb_action(action: int) -> bool:
    """Check if action is FORCE_RTB.
    
    Args:
        action: Action index
        
    Returns:
        True if FORCE_RTB
        
    Example:
        >>> is_rtb_action(4)
        True
    """
    from .action_config import ActionType
    return action == ActionType.FORCE_RTB

