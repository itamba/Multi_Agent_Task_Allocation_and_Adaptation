"""
MAPPO Rollout Buffer
====================

On-policy trajectory storage for PPO training.

Unlike DQN's replay buffer (stores random past experiences),
this buffer stores ONE episode of transitions, computes advantages
using GAE (Generalized Advantage Estimation), then gets cleared.

Flow:
    1. Collect transitions during episode:
       store(local_obs, global_obs, action, log_prob, reward, value, done, mask)
    2. After episode ends:
       compute_returns_and_advantages(last_value)
    3. PPO update loop:
       for epoch in range(K):
           for batch in buffer.get_batches(batch_size):
               update_network(batch)
    4. Clear buffer for next episode:
       reset()

GAE (Generalized Advantage Estimation):
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)                    # TD error
    A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...      # GAE

    λ=0: A_t = δ_t (high bias, low variance — like TD)
    λ=1: A_t = Σ γ^k r_{t+k} - V(s_t) (low bias, high variance — like MC)
    λ=0.95: Good balance (standard in PPO)

Multi-Agent Note:
    Each agent contributes transitions independently (parameter sharing).
    The buffer doesn't distinguish between agents — all transitions are
    from "the shared policy" seeing different local observations.
    Global observations are stored for the centralized critic.

Usage:
    buffer = RolloutBuffer(obs_dim=30, global_obs_dim=60, action_dim=5)

    # During episode
    buffer.store(local_obs, global_obs, action, log_prob, reward, value, done, mask)

    # After episode
    buffer.compute_returns_and_advantages(last_value=0.0)

    # Training
    for batch in buffer.get_batches(batch_size=64):
        # batch is a dict with: obs, global_obs, actions, log_probs, etc.
        pass

    buffer.reset()
"""

import numpy as np
from typing import Dict, Generator, Optional


class RolloutBuffer:
    """
    Fixed-size buffer for one episode of on-policy data.

    Stores transitions from ALL agents (with parameter sharing, they're
    all samples from the same policy).

    Attributes:
        capacity: Maximum transitions per episode
        obs_dim: Local observation dimension (30)
        global_obs_dim: Global observation dimension (60)
        action_dim: Number of actions (5)
        gamma: Discount factor for returns
        gae_lambda: GAE lambda for advantage estimation
    """

    def __init__(
        self,
        obs_dim: int = 30,
        global_obs_dim: int = 60,
        action_dim: int = 5,
        capacity: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize rollout buffer.

        Args:
            obs_dim: Local observation dimension per agent
            global_obs_dim: Global observation dimension (obs_dim * n_agents)
            action_dim: Number of discrete actions
            capacity: Max transitions to store
            gamma: Discount factor (0.99)
            gae_lambda: GAE lambda (0.95)
        """
        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Pre-allocate arrays
        self.local_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.global_obs = np.zeros((capacity, global_obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.action_masks = np.zeros((capacity, action_dim), dtype=np.float32)

        # Computed after episode
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        # Oracle actions for imitation learning metrics
        self.oracle_actions = np.zeros(capacity, dtype=np.int64)

        self.ptr = 0  # Next write position
        self.path_start_idx = 0  # Start of current trajectory
        self.size = 0  # Total stored transitions

    def store(
        self,
        local_obs: np.ndarray,
        global_obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        action_mask: np.ndarray,
        oracle_action: int = -1,
    ):
        """
        Store a single transition.

        Called once per agent per decision point. With parameter sharing,
        transitions from all agents go into the same buffer.

        Args:
            local_obs: Agent's local observation [obs_dim]
            global_obs: Global state [global_obs_dim]
            action: Action taken by agent
            log_prob: Log probability of action under current policy
            reward: Reward received (imitation reward)
            value: Value estimate from critic V(s)
            done: Whether episode ended
            action_mask: Valid actions [action_dim]
            oracle_action: What oracle would do (-1 if unknown)
        """
        assert self.ptr < self.capacity, (
            f"Buffer overflow: {self.ptr} >= {self.capacity}. "
            f"Increase capacity or call reset() between episodes."
        )

        self.local_obs[self.ptr] = local_obs
        self.global_obs[self.ptr] = global_obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.action_masks[self.ptr] = action_mask
        self.oracle_actions[self.ptr] = oracle_action

        self.ptr += 1
        self.size = self.ptr

    def compute_returns_and_advantages(self, last_value: float = 0.0):
        """
        Compute GAE advantages and discounted returns.

        Must be called after collecting all transitions for the episode,
        BEFORE training.

        GAE formula:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            A_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}
            R_t = A_t + V(s_t)

        Args:
            last_value: V(s_T) — value of the final state.
                0.0 if episode ended (done=True).
                V(s) if episode was truncated (time limit).
        """
        # Work backwards from the last transition
        last_gae = 0.0

        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE: A_t = δ_t + γλ * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

            self.advantages[t] = last_gae

        # Returns = advantages + values
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]

    def get_batches(self, batch_size: int) -> Generator[Dict[str, np.ndarray], None, None]:
        """
        Yield mini-batches for PPO training.

        Shuffles data and yields batches. Used in the inner PPO loop:
            for epoch in range(K):
                for batch in buffer.get_batches(batch_size):
                    loss = update(batch)

        Args:
            batch_size: Size of each mini-batch

        Yields:
            Dict with keys: obs, global_obs, actions, log_probs,
            advantages, returns, action_masks, oracle_actions
        """
        indices = np.arange(self.size)
        np.random.shuffle(indices)

        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_indices = indices[start:end]

            yield {
                "obs": self.local_obs[batch_indices],
                "global_obs": self.global_obs[batch_indices],
                "actions": self.actions[batch_indices],
                "log_probs": self.log_probs[batch_indices],
                "advantages": self.advantages[batch_indices],
                "returns": self.returns[batch_indices],
                "action_masks": self.action_masks[batch_indices],
                "oracle_actions": self.oracle_actions[batch_indices],
            }

    def get_all(self) -> Dict[str, np.ndarray]:
        """Get all data as a single batch (for small datasets)."""
        return {
            "obs": self.local_obs[:self.size],
            "global_obs": self.global_obs[:self.size],
            "actions": self.actions[:self.size],
            "log_probs": self.log_probs[:self.size],
            "advantages": self.advantages[:self.size],
            "returns": self.returns[:self.size],
            "action_masks": self.action_masks[:self.size],
            "oracle_actions": self.oracle_actions[:self.size],
        }

    def reset(self):
        """Clear buffer for next episode."""
        self.ptr = 0
        self.path_start_idx = 0
        self.size = 0
        # No need to zero arrays — ptr/size control what's valid

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return self.size

    def get_imitation_accuracy(self) -> float:
        """Fraction of actions that matched oracle (for logging)."""
        if self.size == 0:
            return 0.0
        valid = self.oracle_actions[:self.size] >= 0
        if not valid.any():
            return 0.0
        matches = (self.actions[:self.size][valid] == self.oracle_actions[:self.size][valid])
        return float(matches.mean())
