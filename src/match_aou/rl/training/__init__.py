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
- Reward functions, Oracle, EpisodeInitializer, FuelDamage
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
    compute_step_reward,
    compute_episode_reward,
    compute_step_reward_batch,
    RewardConfig,
    RewardTracker,
    build_target_utility_map,
    get_action_utility,
    compute_oracle_total_utility,
)
from .fuel_damage import FuelDamageManager, FuelDamageConfig

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
    'compute_step_reward',
    'compute_episode_reward',
    'compute_step_reward_batch',
    'RewardConfig',
    'RewardTracker',
    'build_target_utility_map',
    'get_action_utility',
    'compute_oracle_total_utility',

    # Fuel damage
    'FuelDamageManager',
    'FuelDamageConfig',
]