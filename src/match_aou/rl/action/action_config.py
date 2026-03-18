"""
Action Space Configuration

Defines the discrete action space for the RL agent.

Action Types:
- NOOP: Continue as planned (no changes)
- INSERT_ATTACK(k): Insert attack on target k from Top-K
- FORCE_RTB: Force return to base

Configuration Parameters:
- top_k: Number of target slots (matches observation space)
- enable_rtb: Whether RTB action is enabled
- enable_noop: Whether NOOP is allowed

IMPORTANT - Current Limitation:
    top_k is currently LIMITED to 1-3 due to hard-coded ActionType enum.
    ActionType defines INSERT_ATTACK_0/1/2 explicitly (3 attack actions).
    
    If you need top_k > 3, you must:
    1. Extend ActionType enum (add INSERT_ATTACK_3, INSERT_ATTACK_4, etc.)
    2. Update action_config.py validation and get_valid_action_indices()
    3. Update action_utils.py range checks
    
    For now, config validation will raise ValueError if top_k > 3.
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import IntEnum


class ActionType(IntEnum):
    """Discrete action types."""
    NOOP = 0
    INSERT_ATTACK_0 = 1  # Attack target slot 0
    INSERT_ATTACK_1 = 2  # Attack target slot 1
    INSERT_ATTACK_2 = 3  # Attack target slot 2
    FORCE_RTB = 4


@dataclass(frozen=True)
class ActionSpaceConfig:
    """Configuration for action space.
    
    Attributes:
        top_k: Number of target slots (must match observation space)
        enable_rtb: Whether FORCE_RTB action is available
        enable_noop: Whether NOOP action is available
        min_attack_fuel_margin: Minimum fuel margin to allow attack (0-1)
        min_rtb_distance_km: Minimum distance from base to allow RTB
        attack_cooldown_ticks: Minimum ticks between consecutive attacks
    """
    top_k: int = 3
    enable_rtb: bool = True
    enable_noop: bool = True
    min_attack_fuel_margin: float = 0.3  # Need 30% fuel to attack
    min_rtb_distance_km: float = 10.0    # Must be >10km from base to RTB
    attack_cooldown_ticks: int = 50      # Wait 50 ticks between attacks
    
    def __post_init__(self):
        """Validate configuration."""
        # IMPORTANT: top_k limited to 1-3 due to hard-coded ActionType enum
        # See module docstring for details on extending this limitation
        if self.top_k not in [1, 2, 3]:
            raise ValueError(
                f"top_k must be 1-3 (current ActionType limitation), got {self.top_k}. "
                f"See action_config.py docstring for how to extend."
            )
        
        if not (0.0 <= self.min_attack_fuel_margin <= 1.0):
            raise ValueError(
                f"min_attack_fuel_margin must be in [0,1], "
                f"got {self.min_attack_fuel_margin}"
            )
        
        if self.min_rtb_distance_km < 0:
            raise ValueError(
                f"min_rtb_distance_km must be >= 0, "
                f"got {self.min_rtb_distance_km}"
            )
        
        if self.attack_cooldown_ticks < 0:
            raise ValueError(
                f"attack_cooldown_ticks must be >= 0, "
                f"got {self.attack_cooldown_ticks}"
            )
    
    def get_action_space_size(self) -> int:
        """Get total number of discrete actions.
        
        Returns:
            Total action count (1 + top_k + rtb_enabled)
        """
        size = 0
        
        if self.enable_noop:
            size += 1
        
        size += self.top_k  # INSERT_ATTACK for each target slot
        
        if self.enable_rtb:
            size += 1
        
        return size
    
    def get_valid_action_indices(self) -> List[int]:
        """Get list of valid action indices.
        
        Returns:
            List of ActionType values that are enabled
        """
        valid = []
        
        if self.enable_noop:
            valid.append(ActionType.NOOP)
        
        # Add INSERT_ATTACK actions for each target slot
        attack_actions = [
            ActionType.INSERT_ATTACK_0,
            ActionType.INSERT_ATTACK_1,
            ActionType.INSERT_ATTACK_2,
        ]
        valid.extend(attack_actions[:self.top_k])
        
        if self.enable_rtb:
            valid.append(ActionType.FORCE_RTB)
        
        return valid
    
    def action_to_string(self, action: int) -> str:
        """Convert action index to human-readable string.
        
        Args:
            action: Action index
            
        Returns:
            String representation (e.g., "NOOP", "INSERT_ATTACK(0)")
        """
        if action == ActionType.NOOP:
            return "NOOP"
        elif action == ActionType.FORCE_RTB:
            return "FORCE_RTB"
        elif ActionType.INSERT_ATTACK_0 <= action <= ActionType.INSERT_ATTACK_2:
            k = action - ActionType.INSERT_ATTACK_0
            return f"INSERT_ATTACK({k})"
        else:
            return f"UNKNOWN({action})"
    
    def string_to_action(self, action_str: str) -> Optional[int]:
        """Convert string to action index.
        
        Args:
            action_str: String like "NOOP" or "INSERT_ATTACK(1)"
            
        Returns:
            Action index or None if invalid
        """
        action_str = action_str.strip().upper()
        
        if action_str == "NOOP":
            return ActionType.NOOP
        
        if action_str == "FORCE_RTB":
            return ActionType.FORCE_RTB
        
        # Parse INSERT_ATTACK(k)
        if action_str.startswith("INSERT_ATTACK(") and action_str.endswith(")"):
            try:
                k_str = action_str[14:-1]  # Extract number
                k = int(k_str)
                if 0 <= k < self.top_k:
                    return ActionType.INSERT_ATTACK_0 + k
            except ValueError:
                pass
        
        return None


# Default configuration (matches observation space defaults)
DEFAULT_ACTION_CONFIG = ActionSpaceConfig(
    top_k=3,
    enable_rtb=True,
    enable_noop=True,
    min_attack_fuel_margin=0.3,
    min_rtb_distance_km=10.0,
    attack_cooldown_ticks=50
)
