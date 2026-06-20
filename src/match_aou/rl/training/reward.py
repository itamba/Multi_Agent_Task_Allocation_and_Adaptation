"""
Reward Function — Pure Utility-Based, with Aircraft Loss as Negative Utility
=============================================================================

Per advisor guidance, the reward is now a single sparse signal delivered at
episode end:

    aircraft_value = aircraft_value_alpha * max_target_utility
    net_utility    = achieved_utility - lost_aircraft_count * aircraft_value
    reward         = (net_utility - max_total_utility) / max_total_utility
                   * episode_reward_scale

`achieved_utility` is the sum of utilities of targets the team actually
destroyed (regardless of whether they were in the partial plan or hidden).
`max_total_utility` is the sum of all target utilities in the scenario —
the team's reward is 0 when it destroys everything and loses no aircraft.

Per-step rewards are gone. The only step-time signal is an
`invalid_action_penalty` for sanity (kept off the imitation path).

References:
    - Reward shaping in cooperative MARL (Schroeder de Witt et al., 2020)
    - Sparse-reward MAPPO best practices (Yu et al., 2022)
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RewardConfig:
    """
    Configuration for utility-based reward computation.

    Fields:
        aircraft_value_alpha: Multiplier mapping max-target-utility to
            per-aircraft asset value. At alpha=1.0 a lost aircraft costs
            exactly one max-utility target.
        invalid_action_penalty: Per-step penalty when the actor samples an
            invalid action (kept for sanity; unrelated to imitation).
        episode_reward_scale: Multiplier applied to the normalized episode
            reward. Default 1.0 — the new normalization already lives near
            [-2, 0], so the old 5x scaling is no longer needed.
    """

    aircraft_value_alpha: float = 1.0
    invalid_action_penalty: float = -2.0
    episode_reward_scale: float = 1.0


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
    Per-decision reward — invalid-action penalty only.

    DEPRECATED imitation signature: `oracle_action`, `rl_utility`,
    `oracle_utility`, `max_utility` are ignored. They are kept so callers
    that still log Match=✓/✗ and (rl_u=, oracle_u=) don't break, but the
    function returns either the invalid-action penalty or 0.0.

    The real learning signal arrives at episode end via
    `compute_episode_reward`.
    """
    if config is None:
        config = RewardConfig()
    if not is_valid:
        return config.invalid_action_penalty
    return 0.0


# =============================================================================
# Episode-End Reward
# =============================================================================

def compute_episode_reward(
    achieved_utility: float,
    max_total_utility: float,
    lost_aircraft_count: int = 0,
    max_target_utility: float = 0.0,
    config: Optional[RewardConfig] = None,
) -> float:
    """
    Compute episode-end reward as normalized net utility shortfall.

    Formula:
        aircraft_value = config.aircraft_value_alpha * max_target_utility
        net_utility    = achieved_utility - lost_aircraft_count * aircraft_value
        reward         = (net_utility - max_total_utility) / max_total_utility
                       * config.episode_reward_scale

    Reward is 0 when the team destroys every target and loses no aircraft.
    It is -1 when nothing is achieved and no aircraft is lost. It can go
    below -1 when aircraft are lost — there is no lower clamp.

    Args:
        achieved_utility:    Sum of utilities of targets the team destroyed.
        max_total_utility:   Sum of all target utilities in the scenario
                             (i.e. the upper bound on achievable utility).
        lost_aircraft_count: Number of our aircraft removed by BLADE
                             (fuel-zero crashes etc).
        max_target_utility:  Maximum utility of any single target — used to
                             scale per-aircraft asset value.
        config:              RewardConfig (uses defaults if None).

    Returns:
        Scaled reward, typically in [-2, 0].

    Examples:
        >>> # All targets hit, 0 crashes
        >>> compute_episode_reward(300.0, 300.0, 0, 100.0)
        0.0
        >>> # 0 targets hit, 0 crashes
        >>> compute_episode_reward(0.0, 300.0, 0, 100.0)
        -1.0
        >>> # 0 targets hit, 1 crash (alpha=1.0, max_util=100, total=300)
        >>> round(compute_episode_reward(0.0, 300.0, 1, 100.0), 4)
        -1.3333
        >>> # All targets hit, 1 crash
        >>> round(compute_episode_reward(300.0, 300.0, 1, 100.0), 4)
        -0.3333
    """
    if config is None:
        config = RewardConfig()

    if max_total_utility <= 0:
        return 0.0

    aircraft_value = config.aircraft_value_alpha * max_target_utility
    net_utility = achieved_utility - lost_aircraft_count * aircraft_value
    reward = (net_utility - max_total_utility) / max_total_utility
    return reward * config.episode_reward_scale


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
    Batched form of compute_step_reward. Same deprecation note applies:
    imitation arguments are ignored; only `is_valid_list` and `config`
    affect the result.
    """
    rewards = [
        compute_step_reward(
            rl_act, oracle_act, rl_u, oracle_u, max_utility, is_valid, config,
        )
        for rl_act, oracle_act, rl_u, oracle_u, is_valid in zip(
            rl_actions, oracle_actions, rl_utilities, oracle_utilities, is_valid_list,
        )
    ]
    return np.array(rewards, dtype=np.float32)


# =============================================================================
# Utility Helpers
# =============================================================================

def build_target_utility_map(tasks: list) -> dict:
    """
    Build a mapping from target_id → utility from Task objects.

    Reads each step's explicit semantic `target_id` field (no action-string parsing).
    Called once per episode for use by the reward function.
    """
    target_utility = {}
    for task in tasks:
        for step in task.steps:
            target_id = getattr(step, "target_id", None)
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
        Utility of the target (0.0 for NOOP/RTB or missing target).
    """
    if action == 0 or action == 4:
        return 0.0

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
) -> float:
    """
    Sum utilities of all unique tasks assigned in the oracle solution.
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

    `imitation_matches` / `accuracy` are kept as informational counters —
    they no longer affect reward, but they are still useful for monitoring
    how often the policy agrees with the oracle.

    `episode_crashes` records the per-episode aircraft loss count.
    """

    def __init__(self):
        self.rewards = []
        self.imitation_matches = 0
        self.total_actions = 0

        self.rl_utilities = []
        self.oracle_utilities = []
        self.episode_achieved_utility = 0.0
        self.episode_oracle_utility = 0.0
        self.episode_crashes = 0

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

    def set_crashes(self, n: int):
        """Record number of aircraft lost this episode."""
        self.episode_crashes = int(n)

    def get_stats(self) -> dict:
        """Get reward and utility statistics."""
        if not self.rewards:
            return {
                "mean_reward": 0.0, "std_reward": 0.0,
                "min_reward": 0.0, "max_reward": 0.0,
                "count": 0, "accuracy": 0.0,
                "utility_ratio": 0.0,
                "mean_rl_utility": 0.0, "mean_oracle_utility": 0.0,
                "crashes": self.episode_crashes,
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
            "crashes": self.episode_crashes,
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
        self.episode_crashes = 0
