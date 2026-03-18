"""
Training Module
===============

Components for training RL agents.

Primary (MAPPO):
- PPOTrainer: PPO with centralized critic, decentralized actor
- PPOConfig: PPO hyperparameters
- RolloutBuffer: On-policy trajectory storage with GAE

Legacy (DQN):
- DQNTrainer, TrainingConfig, ReplayBuffer

Shared:
- Reward functions, Oracle, EpisodeInitializer
"""

# PPO (primary)
from .ppo_trainer import PPOTrainer, PPOConfig
from .rollout_buffer import RolloutBuffer

# DQN (legacy)
# from .buffer import ReplayBuffer
# from .trainer import DQNTrainer, TrainingConfig, train_episode

# Shared
from .oracle import MatchAOUOracle, SimpleOracle
from .episode_initializer import EpisodeInitializer
from .reward import (
    compute_reward,
    compute_reward_batch,
    compute_simple_imitation_reward,
    compute_soft_imitation_reward,
    RewardConfig,
    RewardTracker,
)

__all__ = [
    # PPO (primary)
    'PPOTrainer',
    'PPOConfig',
    'RolloutBuffer',

    # DQN (legacy)
    # 'ReplayBuffer',
    # 'DQNTrainer',
    # 'TrainingConfig',
    # 'train_episode',

    # Oracle
    'MatchAOUOracle',
    'SimpleOracle',

    # Episode
    'EpisodeInitializer',

    # Reward
    'compute_reward',
    'compute_reward_batch',
    'compute_simple_imitation_reward',
    'compute_soft_imitation_reward',
    'RewardConfig',
    'RewardTracker',
]
