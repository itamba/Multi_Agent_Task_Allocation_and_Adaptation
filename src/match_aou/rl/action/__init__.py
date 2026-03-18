"""
Action Space Module

Provides discrete action space for RL agents in the MATCH-AOU system.

Main API:
    compute_action_mask: Get valid actions for current state
    validate_action: Check if a specific action is legal
    ActionSpaceConfig: Configuration for action space

Action Space Structure:
    Total actions: 1 (NOOP) + K (attacks) + 1 (RTB) where K = config.top_k
    
    Actions:
    - NOOP (0): Continue current plan without changes
    - INSERT_ATTACK(k) (1-3): Insert attack on target slot k (k=0,1,2)
    - FORCE_RTB (4): Force return to base
    
    Default (top_k=3, enable_rtb=True): 5 total actions [0,1,2,3,4]
    
    IMPORTANT LIMITATION:
    Currently top_k is LIMITED to 1-3 due to hard-coded ActionType enum.
    See action_config.py for details on extending this limitation.

Usage:
    from match_aou.rl.action import (
        ActionSpaceConfig,
        compute_action_mask,
        validate_action
    )
    
    # Configure action space (top_k must match observation space)
    config = ActionSpaceConfig(top_k=3, enable_rtb=True)
    
    # Get valid actions for current state
    mask = compute_action_mask(obs, scenario, agent_id, config)
    valid_actions = mask.get_valid_actions()
    
    # Sample from valid actions (or use policy)
    action = np.random.choice(valid_actions)
    
    # Validate before execution
    is_valid, reason = validate_action(action, obs, scenario, agent_id, config)
    if not is_valid:
        print(f"Invalid action: {reason}")

Phase: 2 of 6 (Action Space)
Version: 1.0.0
"""

from typing import Optional

from .action_config import ActionSpaceConfig, ActionType, DEFAULT_ACTION_CONFIG
from .action_validation import ActionValidator, ActionMask
from .action_utils import (
    action_index_to_target_slot,
    is_attack_action,
    is_noop_action,
    is_rtb_action
)


# ============================================================================
# Public API Functions
# ============================================================================

def compute_action_mask(
    observation_output,
    scenario,
    agent_id: str,
    config: Optional[ActionSpaceConfig] = None,
    last_attack_tick: Optional[int] = None
) -> ActionMask:
    """Compute action mask for current state.
    
    This is the main function for determining which actions are legal.
    
    Args:
        observation_output: ObservationOutput from build_observation_vector
        scenario: BLADE Scenario object
        agent_id: Agent ID
        config: Optional ActionSpaceConfig (uses default if None)
        last_attack_tick: Tick of last attack (for cooldown check)
        
    Returns:
        ActionMask with binary mask and invalid reasons
        
    Example:
        >>> mask = compute_action_mask(obs, scenario, "f16_01")
        >>> valid_actions = mask.get_valid_actions()
        >>> print(valid_actions)
        [0, 1, 4]  # NOOP, INSERT_ATTACK(0), FORCE_RTB
        >>> print(mask.get_invalid_reason(2))
        'Target 1 does not exist'
    """
    if config is None:
        config = DEFAULT_ACTION_CONFIG
    
    validator = ActionValidator(config)
    return validator.validate_action_mask(
        observation_output,
        scenario,
        agent_id,
        last_attack_tick
    )


def validate_action(
    action: int,
    observation_output,
    scenario,
    agent_id: str,
    config: Optional[ActionSpaceConfig] = None,
    last_attack_tick: Optional[int] = None
) -> tuple[bool, Optional[str]]:
    """Validate a single action.
    
    Args:
        action: Action index to validate
        observation_output: ObservationOutput
        scenario: BLADE Scenario
        agent_id: Agent ID
        config: Optional ActionSpaceConfig
        last_attack_tick: Last attack tick
        
    Returns:
        (is_valid, reason) tuple where reason is None if valid
        
    Example:
        >>> is_valid, reason = validate_action(1, obs, scenario, "f16_01")
        >>> if not is_valid:
        ...     print(f"Action invalid: {reason}")
    """
    if config is None:
        config = DEFAULT_ACTION_CONFIG
    
    validator = ActionValidator(config)
    return validator.validate_single_action(
        action,
        observation_output,
        scenario,
        agent_id,
        last_attack_tick
    )


def action_to_string(action: int, config: Optional[ActionSpaceConfig] = None) -> str:
    """Convert action index to human-readable string.
    
    Args:
        action: Action index
        config: Optional ActionSpaceConfig
        
    Returns:
        String representation (e.g., "INSERT_ATTACK(1)")
        
    Example:
        >>> action_to_string(0)
        'NOOP'
        >>> action_to_string(2)
        'INSERT_ATTACK(1)'
        >>> action_to_string(4)
        'FORCE_RTB'
    """
    if config is None:
        config = DEFAULT_ACTION_CONFIG
    
    return config.action_to_string(action)


def string_to_action(action_str: str, config: Optional[ActionSpaceConfig] = None) -> Optional[int]:
    """Convert string to action index.
    
    Args:
        action_str: String like "NOOP" or "INSERT_ATTACK(1)"
        config: Optional ActionSpaceConfig
        
    Returns:
        Action index or None if invalid
        
    Example:
        >>> string_to_action("NOOP")
        0
        >>> string_to_action("INSERT_ATTACK(1)")
        2
        >>> string_to_action("INVALID")
        None
    """
    if config is None:
        config = DEFAULT_ACTION_CONFIG
    
    return config.string_to_action(action_str)


def get_action_space_size(config: Optional[ActionSpaceConfig] = None) -> int:
    """Get total number of discrete actions.
    
    Args:
        config: Optional ActionSpaceConfig
        
    Returns:
        Total action count
        
    Example:
        >>> get_action_space_size()  # Default: NOOP + 3 attacks + RTB
        5
        >>> get_action_space_size(ActionSpaceConfig(top_k=2, enable_rtb=False))
        3  # NOOP + 2 attacks
    """
    if config is None:
        config = DEFAULT_ACTION_CONFIG
    
    return config.get_action_space_size()


def get_valid_action_indices(config: Optional[ActionSpaceConfig] = None) -> list[int]:
    """Get list of all possible action indices.
    
    Args:
        config: Optional ActionSpaceConfig
        
    Returns:
        List of ActionType values
        
    Example:
        >>> get_valid_action_indices()
        [0, 1, 2, 3, 4]  # NOOP, INSERT_ATTACK(0-2), FORCE_RTB
    """
    if config is None:
        config = DEFAULT_ACTION_CONFIG
    
    return config.get_valid_action_indices()


# ============================================================================
# Re-export key classes and functions
# ============================================================================

__all__ = [
    # Config
    'ActionSpaceConfig',
    'ActionType',
    'DEFAULT_ACTION_CONFIG',
    
    # Validation
    'ActionMask',
    'compute_action_mask',
    'validate_action',
    
    # Conversion
    'action_to_string',
    'string_to_action',
    
    # Queries
    'get_action_space_size',
    'get_valid_action_indices',
    'action_index_to_target_slot',
    'is_attack_action',
    'is_noop_action',
    'is_rtb_action',
]
