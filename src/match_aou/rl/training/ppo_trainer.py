"""
MAPPO Trainer — PPO with Centralized Training, Decentralized Execution
========================================================================

Trains an ActorCriticNetwork using Proximal Policy Optimization (PPO)
with centralized critic and decentralized actor.

PPO Algorithm:
    1. Collect trajectory using current policy (actor)
    2. Compute advantages using GAE with centralized critic
    3. For K epochs:
        a. Compute new log_probs and values
        b. Policy loss: clipped surrogate objective
        c. Value loss: MSE between V(s) and returns
        d. Entropy bonus: encourages exploration
        e. Total loss = policy_loss + c1 * value_loss - c2 * entropy

PPO Clipped Objective:
    ratio = π_new(a|s) / π_old(a|s)
    L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    This prevents the policy from changing too much in one update.
    ε = 0.2 is standard.

Imitation Learning Integration:
    The reward signal comes from comparing RL actions to oracle (MATCH-AOU
    with full information). PPO then optimizes this reward through the
    standard RL objective — the actor learns to take actions that match
    the oracle, while the critic learns to predict expected imitation reward.

Usage:
    from match_aou.rl.agent import ActorCriticNetwork
    from match_aou.rl.training import PPOTrainer, PPOConfig

    net = ActorCriticNetwork(obs_dim=30, action_dim=5, n_agents=2)
    config = PPOConfig()
    trainer = PPOTrainer(net, config)

    # Collect data
    trainer.buffer.store(obs, global_obs, action, log_prob, reward, value, done, mask)
    ...

    # Train
    metrics = trainer.update()

    # Next episode
    trainer.buffer.reset()
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

from .rollout_buffer import RolloutBuffer
from .reward import RewardConfig, RewardTracker
from ..agent.network import ActorCriticNetwork

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """
    PPO hyperparameters.

    Grouped by function:
        Network:    obs_dim, action_dim, n_agents, hidden_size
        PPO core:   clip_eps, gamma, gae_lambda
        Training:   lr, epochs, batch_size, max_grad_norm
        Loss:       value_coef, entropy_coef
        Buffer:     buffer_capacity
        Reward:     reward_config
        Output:     model_dir
    """

    # --- Network ---
    obs_dim: int = 30
    action_dim: int = 5
    n_agents: int = 2
    hidden_size: int = 128

    # --- PPO core ---
    clip_eps: float = 0.2       # Clipping parameter ε (standard: 0.2)
    gamma: float = 0.99         # Discount factor
    gae_lambda: float = 0.95    # GAE λ (0.95 is standard balance)

    # --- Training ---
    learning_rate: float = 3e-4    # Standard PPO lr (lower than DQN)
    ppo_epochs: int = 4            # K epochs per update (4-10 is typical)
    batch_size: int = 64           # Mini-batch size
    max_grad_norm: float = 0.5     # Gradient clipping (PPO standard)

    # --- Loss coefficients ---
    value_coef: float = 0.5        # c1: weight of value loss
    entropy_coef: float = 0.01     # c2: weight of entropy bonus

    # --- Buffer ---
    buffer_capacity: int = 2048    # Max transitions per episode

    # --- Reward ---
    reward_config: Optional[RewardConfig] = None

    # --- Output ---
    model_dir: str = "models"

    def __post_init__(self):
        if self.reward_config is None:
            self.reward_config = RewardConfig()


class PPOTrainer:
    """
    MAPPO Trainer with centralized critic and decentralized actor.

    Lifecycle per episode:
        1. trainer.get_action(local_obs, mask) → action, log_prob, value
        2. trainer.buffer.store(...)           → accumulate transitions
        3. trainer.update()                    → PPO update (K epochs)
        4. trainer.buffer.reset()              → clear for next episode

    Attributes:
        network: ActorCriticNetwork (actor + critic)
        optimizer: Adam optimizer (shared for actor and critic)
        buffer: RolloutBuffer for trajectory storage
        config: PPOConfig hyperparameters
    """

    def __init__(
        self,
        network: ActorCriticNetwork,
        config: Optional[PPOConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize PPO trainer.

        Args:
            network: ActorCriticNetwork to train
            config: PPO hyperparameters
            device: 'cpu' or 'cuda'
        """
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.network = network.to(self.device)

        # Single optimizer for both actor and critic
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,  # PPO standard
        )

        # Rollout buffer
        global_obs_dim = self.config.obs_dim * self.config.n_agents
        self.buffer = RolloutBuffer(
            obs_dim=self.config.obs_dim,
            global_obs_dim=global_obs_dim,
            action_dim=self.config.action_dim,
            capacity=self.config.buffer_capacity,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Tracking
        self.reward_tracker = RewardTracker()
        self.episode_count = 0
        self.total_updates = 0
        self.metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        # Model directory
        self.model_dir = Path(self.config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_action(
        self,
        local_obs: np.ndarray,
        global_obs: np.ndarray,
        action_mask: np.ndarray,
    ) -> tuple:
        """
        Select action using current policy (for data collection).

        This runs the ACTOR only (decentralized). The critic value is
        also computed for storing in the buffer.

        Args:
            local_obs: Agent's local observation [obs_dim]
            global_obs: Global state [global_obs_dim]
            action_mask: Valid actions [action_dim]

        Returns:
            (action, log_prob, value) tuple
            - action: int, selected action
            - log_prob: float, log probability
            - value: float, critic's value estimate
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(local_obs).to(self.device)
            global_t = torch.FloatTensor(global_obs).to(self.device)
            mask_t = torch.BoolTensor(action_mask).to(self.device)

            action, log_prob, entropy, value = self.network.get_action_and_value(
                obs_t, global_t, mask_t
            )

        return (
            action.item(),
            log_prob.item(),
            value.squeeze().item(),
        )

    def update(self) -> Dict[str, float]:
        """
        Run PPO update on collected trajectory.

        Must be called after buffer.compute_returns_and_advantages().

        Process:
            For K epochs:
                For each mini-batch:
                    1. Recompute log_probs and values with current network
                    2. Compute ratio = new_prob / old_prob
                    3. Policy loss = -min(ratio*A, clip(ratio)*A)
                    4. Value loss = MSE(V, returns)
                    5. Entropy bonus = mean entropy
                    6. Backprop total loss

        Returns:
            Dict with training metrics
        """
        if self.buffer.size == 0:
            logger.warning("Buffer empty, skipping update")
            return {}

        # Normalize advantages (standard in PPO, helps training)
        advantages = self.buffer.advantages[:self.buffer.size]
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        self.buffer.advantages[:self.buffer.size] = (advantages - adv_mean) / adv_std

        # Track metrics across epochs
        epoch_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        for epoch in range(self.config.ppo_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                # Convert batch to tensors
                obs = torch.FloatTensor(batch["obs"]).to(self.device)
                global_obs = torch.FloatTensor(batch["global_obs"]).to(self.device)
                actions = torch.LongTensor(batch["actions"]).to(self.device)
                old_log_probs = torch.FloatTensor(batch["log_probs"]).to(self.device)
                adv = torch.FloatTensor(batch["advantages"]).to(self.device)
                returns = torch.FloatTensor(batch["returns"]).to(self.device)
                masks = torch.FloatTensor(batch["action_masks"]).to(self.device)

                # --- Forward pass with current network ---
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    local_obs=obs,
                    global_obs=global_obs,
                    action_mask=masks,
                    action=actions,
                )
                new_values = new_values.squeeze(-1)

                # --- Policy loss (clipped surrogate) ---
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value loss (MSE) ---
                value_loss = nn.functional.mse_loss(new_values, returns)

                # --- Entropy bonus ---
                entropy_loss = entropy.mean()

                # --- Total loss ---
                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                # --- Optimize ---
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                # --- Metrics ---
                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_eps).float().mean().item()

                epoch_metrics["policy_loss"].append(policy_loss.item())
                epoch_metrics["value_loss"].append(value_loss.item())
                epoch_metrics["entropy"].append(entropy_loss.item())
                epoch_metrics["total_loss"].append(total_loss.item())
                epoch_metrics["approx_kl"].append(approx_kl)
                epoch_metrics["clip_fraction"].append(clip_frac)

        self.total_updates += 1

        # Aggregate metrics
        result = {}
        for key, values in epoch_metrics.items():
            mean_val = np.mean(values) if values else 0.0
            result[key] = mean_val
            self.metrics[key].append(mean_val)

        # Add imitation accuracy
        result["imitation_accuracy"] = self.buffer.get_imitation_accuracy()

        return result

    def save_checkpoint(self, filename: str):
        """Save training state."""
        path = Path(filename)
        if not path.is_absolute():
            path = self.model_dir / filename

        torch.save(
            {
                "network_state": self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "episode_count": self.episode_count,
                "total_updates": self.total_updates,
                "metrics": self.metrics,
                "config": self.config,
            },
            path,
        )
        logger.info(f"Saved PPO checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.episode_count = checkpoint["episode_count"]
        self.total_updates = checkpoint["total_updates"]
        self.metrics = checkpoint["metrics"]
        logger.info(f"Loaded PPO checkpoint: {path} (episode {self.episode_count})")

    def get_metrics_summary(self) -> Dict:
        """Get summary of training metrics."""
        summary = {
            "episode_count": self.episode_count,
            "total_updates": self.total_updates,
        }
        for key, values in self.metrics.items():
            summary[f"avg_{key}"] = np.mean(values[-100:]) if values else 0.0
        summary["reward_stats"] = self.reward_tracker.get_stats()
        return summary
