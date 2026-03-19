"""
Reward Function - Utility-Based Imitation Learning from MATCH-AOU Oracle
=========================================================================

Reward function for training RL agents to imitate MATCH-AOU's optimal decisions,
using task utility values as the reward signal.

Design principles (per advisor guidance):
- NO ad-hoc bonuses (fuel efficiency, target coverage, etc.)
- ALL reward signal derives from MATCH-AOU utility comparison
- Two components:
    1. Per-step: utility-proportional reward based on oracle comparison
    2. Episode-end: ratio of achieved utility vs oracle total utility

Per-step logic:
    Match (RL == Oracle):
        - Attack match  → +oracle_utility / max_utility
        - NOOP match    → +noop_match_reward (small positive)
    Mismatch (RL != Oracle):
        - (rl_utility - oracle_utility) / max_utility
        - e.g., RL picks NOOP when oracle says ATTACK(100): (0-100)/100 = -1.0
        - e.g., RL picks ATTACK(80) when oracle says ATTACK(100): (80-100)/100 = -0.2

Episode-end logic:
    achieved_utility / oracle_total_utility * episode_reward_scale

References:
    - IRAT (Wang et al., 2022): per-step individual + episodic team reward
    - Differentiated reward (2025): utility-proportional signals accelerate MAPPO
    - HERO (Tao et al., 2025): hybrid sparse+dense outperforms either alone

Usage:
    from match_aou.rl.training import compute_step_reward, compute_episode_reward, RewardConfig

    reward = compute_step_reward(
        rl_action=1, oracle_action=2,
        rl_utility=80.0, oracle_utility=100.0,
        max_utility=100.0, config=RewardConfig()
    )
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RewardConfig:
    """
    Configuration for utility-based reward computation.

    Per-step parameters:
        noop_match_reward: Reward when both RL and oracle choose NOOP (+0.1)
        invalid_action_penalty: Penalty for invalid actions (-2.0)

    Episode-end parameters:
        episode_reward_scale: Multiplier for utility ratio at episode end (5.0)

    Weighting (for combined total, used externally):
        step_weight: Relative weight of per-step component (0.3)
        episode_weight: Relative weight of episode-end component (0.7)
    """

    # Per-step
    noop_match_reward: float = 0.1
    invalid_action_penalty: float = -2.0

    # Episode-end
    episode_reward_scale: float = 5.0

    # Weighting guidance (used by train_full.py, not enforced here)
    step_weight: float = 0.3
    episode_weight: float = 0.7


# =============================================================================
# Per-Step Reward
# =============================================================================

def compute_step_reward(
    rl_action: int,
    oracle_action: int,
    rl_utility: float,
    oracle_utility: float,
    max_utility: float,
    is_valid: bool = True,
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Compute per-step reward based on utility comparison with oracle.

    Args:
        rl_action: Action selected by RL agent (0-4)
        oracle_action: Action selected by oracle (0-4)
        rl_utility: Utility of the target RL chose (0.0 if NOOP/RTB)
        oracle_utility: Utility of the target oracle chose (0.0 if NOOP)
        max_utility: Maximum utility across all tasks (for normalization)
        is_valid: Whether rl_action passed action masking
        config: RewardConfig (uses default if None)

    Returns:
        Float reward value

    Examples:
        >>> # RL matches oracle on high-value target
        >>> compute_step_reward(1, 1, 100.0, 100.0, 100.0)
        1.0
        >>> # RL picks NOOP when oracle says attack
        >>> compute_step_reward(0, 1, 0.0, 100.0, 100.0)
        -1.0
        >>> # Both NOOP — small positive
        >>> compute_step_reward(0, 0, 0.0, 0.0, 100.0)
        0.1
    """
    if config is None:
        config = RewardConfig()

    # Invalid action overrides everything
    if not is_valid:
        return config.invalid_action_penalty

    # Avoid division by zero
    if max_utility <= 0:
        max_utility = 1.0

    # Match
    if rl_action == oracle_action:
        if oracle_utility > 0:
            # Attack match — proportional to target value
            return oracle_utility / max_utility
        else:
            # NOOP=NOOP or RTB=RTB — small positive
            return config.noop_match_reward

    # Mismatch — utility differential
    return (rl_utility - oracle_utility) / max_utility


# =============================================================================
# Episode-End Reward
# =============================================================================

def compute_episode_reward(
    achieved_utility: float,
    oracle_total_utility: float,
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Compute episode-end reward as utility ratio.

    This is the "big picture" signal: how much of the oracle's total
    utility did the RL agents actually achieve?

    Args:
        achieved_utility: Sum of utilities of targets successfully attacked by RL
        oracle_total_utility: Sum of utilities in the full oracle solution
        config: RewardConfig (uses default if None)

    Returns:
        Scaled utility ratio (0 to episode_reward_scale, typically 0-5)

    Examples:
        >>> # Perfect execution
        >>> compute_episode_reward(280.0, 280.0)
        5.0
        >>> # Half the utility achieved
        >>> compute_episode_reward(140.0, 280.0)
        2.5
        >>> # Nothing achieved
        >>> compute_episode_reward(0.0, 280.0)
        0.0
    """
    if config is None:
        config = RewardConfig()

    if oracle_total_utility <= 0:
        return 0.0

    ratio = achieved_utility / oracle_total_utility
    # Clamp to [0, 1.5] — allow slight over-performance but cap it
    ratio = min(ratio, 1.5)

    return ratio * config.episode_reward_scale


# =============================================================================
# Batch Computation
# =============================================================================

def compute_step_reward_batch(
    rl_actions: list,
    oracle_actions: list,
    rl_utilities: list,
    oracle_utilities: list,
    max_utility: float,
    is_valid_list: list,
    config: Optional[RewardConfig] = None,
) -> np.ndarray:
    """
    Compute per-step rewards for a batch of actions.

    Args:
        rl_actions: List of RL actions
        oracle_actions: List of oracle actions
        rl_utilities: List of RL target utilities
        oracle_utilities: List of oracle target utilities
        max_utility: Max utility for normalization
        is_valid_list: List of validity flags
        config: RewardConfig

    Returns:
        np.array of rewards
    """
    rewards = []
    for rl_act, oracle_act, rl_u, oracle_u, is_valid in zip(
        rl_actions, oracle_actions, rl_utilities, oracle_utilities, is_valid_list
    ):
        r = compute_step_reward(rl_act, oracle_act, rl_u, oracle_u, max_utility, is_valid, config)
        rewards.append(r)

    return np.array(rewards, dtype=np.float32)


# =============================================================================
# Utility Helpers
# =============================================================================

def build_target_utility_map(tasks: list, extract_target_id_fn) -> dict:
    """
    Build a mapping from target_id → utility from Task objects.

    This is called once per episode to create the lookup table used
    by the reward function.

    Args:
        tasks: List of Task objects (with .utility and .steps[].action)
        extract_target_id_fn: Function that extracts target_id from action string
            Signature: (action_str: str) -> Optional[str]

    Returns:
        Dict mapping target_id (str) → utility (float)

    Example:
        >>> from match_aou.rl.observation.observation_utils import extract_target_id_from_action
        >>> utility_map = build_target_utility_map(all_tasks, extract_target_id_from_action)
        >>> utility_map
        {'facility-1': 100.0, 'facility-2': 100.0, 'airbase-1': 80.0}
    """
    target_utility = {}
    for task in tasks:
        for step in task.steps:
            action_str = getattr(step, "action", "") or ""
            target_id = extract_target_id_fn(action_str)
            if target_id:
                target_utility[target_id] = task.utility
    return target_utility


def get_action_utility(
    action: int,
    observation,  # ObservationOutput
    target_utility_map: dict,
) -> float:
    """
    Get the utility of the target associated with an action.

    Args:
        action: Action index (0=NOOP, 1-3=ATTACK slot, 4=RTB)
        observation: ObservationOutput with targets info
        target_utility_map: Mapping target_id → utility

    Returns:
        Utility of the target (0.0 for NOOP/RTB or missing target)
    """
    # NOOP or RTB — no target utility
    if action == 0 or action == 4:
        return 0.0

    # ATTACK slot (1-3) → target slot (0-2)
    slot_idx = action - 1
    if slot_idx >= len(observation.targets):
        return 0.0

    target = observation.targets[slot_idx]
    if not target.exists:
        return 0.0

    return target_utility_map.get(target.id, 0.0)


def compute_oracle_total_utility(
    full_solution: dict,
    tasks: list,
    extract_target_id_fn,
) -> float:
    """
    Compute total utility of the full oracle solution.

    Sums utilities of all unique tasks assigned in the oracle solution.

    Args:
        full_solution: {agent_id: [(task_idx, step_idx, level), ...]}
        tasks: Task list (indexed by task_idx)
        extract_target_id_fn: Target ID extractor function

    Returns:
        Total utility (float)
    """
    selected_tasks = set()
    for assignments in full_solution.values():
        for task_idx, step_idx, _level in assignments:
            selected_tasks.add(task_idx)

    total = 0.0
    for task_idx in selected_tasks:
        if 0 <= task_idx < len(tasks):
            total += tasks[task_idx].utility

    return total


# =============================================================================
# Reward Tracker
# =============================================================================

class RewardTracker:
    """
    Track reward statistics during training.

    Tracks both per-step rewards and utility-based metrics.

    Example:
        >>> tracker = RewardTracker()
        >>> tracker.add_step(reward=0.8, is_match=True, rl_utility=100, oracle_utility=100)
        >>> tracker.set_episode_utilities(achieved=180, oracle_total=280)
        >>> print(tracker.get_stats())
    """

    def __init__(self):
        self.rewards = []
        self.imitation_matches = 0
        self.total_actions = 0

        # Utility tracking
        self.rl_utilities = []
        self.oracle_utilities = []
        self.episode_achieved_utility = 0.0
        self.episode_oracle_utility = 0.0

    def add_step(
        self,
        reward: float,
        is_match: bool = False,
        rl_utility: float = 0.0,
        oracle_utility: float = 0.0,
    ):
        """Record a single decision step."""
        self.rewards.append(reward)
        self.total_actions += 1
        if is_match:
            self.imitation_matches += 1
        self.rl_utilities.append(rl_utility)
        self.oracle_utilities.append(oracle_utility)

    def set_episode_utilities(self, achieved: float, oracle_total: float):
        """Record episode-level utility totals."""
        self.episode_achieved_utility = achieved
        self.episode_oracle_utility = oracle_total

    def get_stats(self) -> dict:
        """Get reward and utility statistics."""
        if not self.rewards:
            return {
                "mean_reward": 0.0, "std_reward": 0.0,
                "min_reward": 0.0, "max_reward": 0.0,
                "count": 0, "accuracy": 0.0,
                "utility_ratio": 0.0,
                "mean_rl_utility": 0.0, "mean_oracle_utility": 0.0,
            }

        reward_floats = [float(r) for r in self.rewards]
        n = len(reward_floats)
        mean_val = sum(reward_floats) / n
        var_val = sum((r - mean_val) ** 2 for r in reward_floats) / n

        utility_ratio = (
            self.episode_achieved_utility / self.episode_oracle_utility
            if self.episode_oracle_utility > 0 else 0.0
        )

        return {
            "mean_reward": mean_val,
            "std_reward": var_val ** 0.5,
            "min_reward": min(reward_floats),
            "max_reward": max(reward_floats),
            "count": n,
            "accuracy": self.imitation_matches / self.total_actions if self.total_actions > 0 else 0.0,
            "utility_ratio": utility_ratio,
            "mean_rl_utility": sum(self.rl_utilities) / n if n > 0 else 0.0,
            "mean_oracle_utility": sum(self.oracle_utilities) / n if n > 0 else 0.0,
        }

    def reset(self):
        """Reset tracker for next episode."""
        self.rewards = []
        self.imitation_matches = 0
        self.total_actions = 0
        self.rl_utilities = []
        self.oracle_utilities = []
        self.episode_achieved_utility = 0.0
        self.episode_oracle_utility = 0.0