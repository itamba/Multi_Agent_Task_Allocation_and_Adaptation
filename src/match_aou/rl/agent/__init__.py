"""
Agent Module — Neural network architectures for RL agents.

Primary:    ActorCriticNetwork  (MAPPO / CTDE)
Legacy:     EnhancedMLPQNetwork (DQN, kept for backward compatibility)
"""

from .network import (
    ActorCriticNetwork,
    EnhancedMLPQNetwork,
    create_target_network,
    soft_update_target_network,
    hard_update_target_network,
)

__all__ = [
    "ActorCriticNetwork",
    "EnhancedMLPQNetwork",
    "create_target_network",
    "soft_update_target_network",
    "hard_update_target_network",
]
