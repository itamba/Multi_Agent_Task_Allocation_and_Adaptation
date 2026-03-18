"""
Training Utilities
==================

Shared helper functions for training module.
"""

import numpy as np
from typing import Optional


def get_action_mask_array(
    observation,
    scenario,
    agent_id: str,
    last_attack_tick: Optional[int] = None
) -> np.ndarray:
    """
    Get action mask as boolean numpy array.
    
    This is a shared helper used by Oracle classes to get valid actions
    from the action space module.
    
    Args:
        observation: ObservationOutput from observation module
        scenario: BLADE Scenario object
        agent_id: Agent ID
        last_attack_tick: Tick of last attack (for cooldown check)
    
    Returns:
        Boolean numpy array [action_dim] where True = valid action
        
    Example:
        >>> mask = get_action_mask_array(obs, scenario, "f16_01")
        >>> print(mask)
        [True, True, False, True, True]  # Action 2 invalid
    """
    from ..action import compute_action_mask, ActionSpaceConfig
    
    config = ActionSpaceConfig(top_k=3)
    
    mask_obj = compute_action_mask(
        observation_output=observation,
        scenario=scenario,
        agent_id=agent_id,
        config=config,
        last_attack_tick=last_attack_tick if last_attack_tick is not None else -9999
    )
    
    # Convert to boolean numpy array
    valid_actions = mask_obj.get_valid_actions()
    
    mask = np.zeros(5, dtype=bool)
    for action in valid_actions:
        mask[action] = True
    
    return mask
