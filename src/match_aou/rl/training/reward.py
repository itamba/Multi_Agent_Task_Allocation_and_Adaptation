"""
Reward Function - Imitation Learning from MATCH-AOU Oracle
===========================================================

Reward function for training RL agent to imitate MATCH-AOU's optimal decisions.

Approach:
- MATCH-AOU solves the full problem (knows all targets)
- RL agent sees partial information (discovers targets gradually)
- Reward = how similar is RL action to what oracle would do?

Reward Components:
1. Imitation reward - main signal (did we match oracle?)
2. Fuel efficiency - bonus for conserving fuel
3. Target coverage - bonus for attacking unassigned targets
4. Invalid action penalty - strong negative for illegal actions

Usage:
    from match_aou.rl.reward import compute_reward, RewardConfig
    
    reward = compute_reward(
        rl_action=1,           # RL chose ATTACK_TARGET_0
        oracle_action=1,       # Oracle also chose ATTACK_TARGET_0
        observation=obs,
        config=RewardConfig()
    )
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RewardConfig:
    """
    Configuration for reward computation.
    
    Attributes:
        imitation_reward: Reward for matching oracle (+1.0)
        imitation_penalty: Penalty for not matching oracle (-0.5)
        
        fuel_efficiency_bonus: Bonus for having fuel surplus (0.1)
        target_coverage_bonus: Bonus for attacking unassigned targets (0.2)
        
        invalid_action_penalty: Penalty for invalid actions (-5.0)
        
        use_shaping: Whether to use reward shaping (True)
            If False, only imitation reward is used (simpler)
    """
    
    # Core imitation learning
    imitation_reward: float = 1.0
    imitation_penalty: float = -0.5
    
    # Reward shaping (optional)
    fuel_efficiency_bonus: float = 0.1
    target_coverage_bonus: float = 0.2
    
    # Safety
    invalid_action_penalty: float = -5.0
    
    # Toggle reward shaping
    use_shaping: bool = True


def compute_reward(
    rl_action: int,
    oracle_action: int,
    observation,  # ObservationOutput
    is_valid: bool = True,
    config: Optional[RewardConfig] = None
) -> float:
    """
    Compute reward for RL agent's action.
    
    Main signal: Does RL action match oracle action?
    
    Args:
        rl_action: Action selected by RL agent (0-4)
        oracle_action: Action selected by MATCH-AOU oracle (0-4)
        observation: ObservationOutput with current state
        is_valid: Whether rl_action is valid (passed action masking)
        config: RewardConfig (uses default if None)
    
    Returns:
        Float reward value
        
    Example:
        >>> reward = compute_reward(
        ...     rl_action=1,      # RL: Attack target 0
        ...     oracle_action=1,  # Oracle: Attack target 0
        ...     observation=obs,
        ...     is_valid=True
        ... )
        >>> print(reward)
        1.0  # Perfect match!
    """
    if config is None:
        config = RewardConfig()
    
    # Invalid action penalty (overrides everything else)
    if not is_valid:
        return config.invalid_action_penalty
    
    # Core imitation reward
    if rl_action == oracle_action:
        reward = config.imitation_reward
    else:
        reward = config.imitation_penalty
    
    # Optional reward shaping
    if config.use_shaping:
        # Fuel efficiency bonus
        if _has_fuel_surplus(observation):
            reward += config.fuel_efficiency_bonus
        
        # Target coverage bonus (attacking unassigned targets)
        if _is_attacking_unassigned_target(rl_action, observation):
            reward += config.target_coverage_bonus
    
    return reward


def compute_reward_batch(
    rl_actions: list,
    oracle_actions: list,
    observations: list,
    is_valid_list: list,
    config: Optional[RewardConfig] = None
) -> np.ndarray:
    """
    Compute rewards for a batch of actions (vectorized).
    
    Args:
        rl_actions: List of RL actions
        oracle_actions: List of oracle actions
        observations: List of ObservationOutputs
        is_valid_list: List of validity flags
        config: RewardConfig
    
    Returns:
        np.array of rewards (same length as input lists)
    
    Example:
        >>> rewards = compute_reward_batch(
        ...     rl_actions=[1, 2, 0],
        ...     oracle_actions=[1, 1, 0],
        ...     observations=[obs1, obs2, obs3],
        ...     is_valid_list=[True, True, True]
        ... )
        >>> print(rewards)
        [1.0, -0.5, 1.0]  # Match, mismatch, match
    """
    rewards = []
    
    for rl_act, oracle_act, obs, is_valid in zip(
        rl_actions, oracle_actions, observations, is_valid_list
    ):
        r = compute_reward(rl_act, oracle_act, obs, is_valid, config)
        rewards.append(r)
    
    return np.array(rewards, dtype=np.float32)


# =============================================================================
# Reward Shaping Helpers
# =============================================================================

def _has_fuel_surplus(observation) -> bool:
    """
    Check if agent has fuel surplus (from plan_context features).
    
    Args:
        observation: ObservationOutput
    
    Returns:
        True if agent has extra fuel beyond plan requirements
    """
    # Check if observation has plan context (30 features)
    if len(observation.vector) < 30:
        return False
    
    # Plan context starts at index 24
    # Feature 0 (index 24): fuel_margin_for_plan
    fuel_margin = observation.vector[24]
    
    # Margin > 0.5 means surplus (see plan_context.py)
    return fuel_margin > 0.5


def _is_attacking_unassigned_target(action: int, observation) -> bool:
    """
    Check if action attacks an unassigned target.
    
    Unassigned targets are opportunities that should be encouraged.
    
    Args:
        action: Action index (0-4)
        observation: ObservationOutput
    
    Returns:
        True if action is attacking an unassigned target
    """
    # Check if action is an attack (1-3)
    if not (1 <= action <= 3):
        return False
    
    # Get target slot (0-2)
    target_slot = action - 1
    
    # Check if target exists and is unassigned
    if target_slot >= len(observation.targets):
        return False
    
    target = observation.targets[target_slot]
    
    if not target.exists:
        return False
    
    # Target is unassigned if is_in_plan == False
    return not target.is_in_plan


# =============================================================================
# Alternative Reward Functions
# =============================================================================

def compute_simple_imitation_reward(
    rl_action: int,
    oracle_action: int
) -> float:
    """
    Simplest possible reward: +1 for match, 0 for mismatch.
    
    Use this for initial debugging/testing.
    
    Args:
        rl_action: RL agent's action
        oracle_action: Oracle's action
    
    Returns:
        1.0 if match, 0.0 if mismatch
    """
    return 1.0 if rl_action == oracle_action else 0.0


def compute_soft_imitation_reward(
        rl_action: int,
        oracle_action: int,
        action_similarities: Optional[dict] = None
) -> float:
    """
    Soft imitation reward with partial credit for similar actions.

    Example:
        Oracle chooses ATTACK_0
        RL chooses ATTACK_1
        â†’ Give partial credit (e.g., 0.5) instead of 0

    Args:
        rl_action: RL agent's action
        oracle_action: Oracle's action
        action_similarities: Dict mapping (rl_action, oracle_action) â†’ similarity
            If None, uses default similarities

    Returns:
        Reward in [0, 1] based on action similarity
    """
    # Perfect match
    if rl_action == oracle_action:
        return 1.0

    # Default similarities
    if action_similarities is None:
        # All attack actions are somewhat similar
        # NOOP vs attack = less similar
        # RTB vs anything = least similar
        action_similarities = {
            # (rl, oracle): similarity
            (0, 0): 1.0,  # NOOP = NOOP
            (0, 1): 0.3, (0, 2): 0.3, (0, 3): 0.3,  # NOOP vs attack
            (0, 4): 0.1,  # NOOP vs RTB

            (1, 0): 0.3, (1, 1): 1.0, (1, 2): 0.7, (1, 3): 0.7,  # ATTACK_0
            (1, 4): 0.1,

            (2, 0): 0.3, (2, 1): 0.7, (2, 2): 1.0, (2, 3): 0.7,  # ATTACK_1
            (2, 4): 0.1,

            (3, 0): 0.3, (3, 1): 0.7, (3, 2): 0.7, (3, 3): 1.0,  # ATTACK_2
            (3, 4): 0.1,

            (4, 0): 0.1, (4, 1): 0.1, (4, 2): 0.1, (4, 3): 0.1,  # RTB
            (4, 4): 1.0,
        }

    # Get similarity
    similarity = action_similarities.get((rl_action, oracle_action), 0.0)

    return similarity


# =============================================================================
# Reward Statistics
# =============================================================================

class RewardTracker:
    """
    Track reward statistics during training.

    Useful for monitoring learning progress.

    Example:
        >>> tracker = RewardTracker()
        >>> tracker.add_reward(1.0)
        >>> tracker.add_reward(-0.5)
        >>> print(tracker.get_stats())
        {'mean': 0.25, 'std': 0.75, 'count': 2}
    """

    def __init__(self):
        self.rewards = []
        self.imitation_matches = 0
        self.total_actions = 0

    def add_reward(self, reward: float, is_match: bool = None):
        """Add a reward to tracker."""
        self.rewards.append(reward)
        self.total_actions += 1

        if is_match is not None and is_match:
            self.imitation_matches += 1

    def get_stats(self) -> dict:
        """Get reward statistics."""
        empty = {
            'mean': 0.0, 'std': 0.0, 'min': 0.0,
            'max': 0.0, 'count': 0, 'accuracy': 0.0,
        }

        if not self.rewards or len(self.rewards) == 0:
            return empty

        try:
            # Use Python builtins for min/max to avoid numpy version issues
            reward_floats = [float(r) for r in self.rewards]
            n = len(reward_floats)
            mean_val = sum(reward_floats) / n
            var_val = sum((r - mean_val) ** 2 for r in reward_floats) / n

            return {
                'mean': mean_val,
                'std': var_val ** 0.5,
                'min': min(reward_floats),
                'max': max(reward_floats),
                'count': n,
                'accuracy': self.imitation_matches / self.total_actions if self.total_actions > 0 else 0.0,
            }
        except (TypeError, ValueError):
            return empty

    def reset(self):
        """Reset tracker."""
        self.rewards = []
        self.imitation_matches = 0
        self.total_actions = 0