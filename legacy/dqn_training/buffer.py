"""
Replay Buffer
=============

Store and sample experience tuples for DQN training.

Experience tuple: (state, action, reward, next_state, done, mask, next_mask)

Features:
- Fixed capacity (FIFO)
- Random sampling
- Efficient numpy arrays
- Action mask storage

Usage:
    from match_aou.rl.training import ReplayBuffer
    
    buffer = ReplayBuffer(capacity=10000)
    
    # Add experience
    buffer.add(state, action, reward, next_state, done, mask, next_mask)
    
    # Sample batch
    batch = buffer.sample(batch_size=32)
    
Note: PrioritizedReplayBuffer removed - can be added later if needed.
"""

from collections import deque
import numpy as np
import random
from typing import Tuple, Optional


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    
    Uses deque for efficient FIFO operations.
    
    Attributes:
        capacity: Maximum number of experiences to store
        buffer: Deque containing experience tuples
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray
    ):
        """
        Add experience to buffer.
        
        Args:
            state: Current state observation [obs_dim]
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Next state observation [obs_dim]
            done: Episode terminated? (bool)
            action_mask: Valid actions in current state [action_dim]
            next_action_mask: Valid actions in next state [action_dim]
        """
        experience = (
            state,
            action,
            reward,
            next_state,
            done,
            action_mask,
            next_action_mask
        )
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Optional[Tuple]:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, masks, next_masks)
            Each as numpy array with batch dimension
            
            Returns None if buffer doesn't have enough samples
        
        Example:
            >>> batch = buffer.sample(32)
            >>> states, actions, rewards, next_states, dones, masks, next_masks = batch
            >>> print(states.shape)
            (32, 30)
        """
        if len(self.buffer) < batch_size:
            return None
        
        # Sample random experiences
        experiences = random.sample(self.buffer, batch_size)
        
        # Unzip into separate arrays
        states = np.array([e[0] for e in experiences], dtype=np.float32)
        actions = np.array([e[1] for e in experiences], dtype=np.int64)
        rewards = np.array([e[2] for e in experiences], dtype=np.float32)
        next_states = np.array([e[3] for e in experiences], dtype=np.float32)
        dones = np.array([e[4] for e in experiences], dtype=np.float32)
        action_masks = np.array([e[5] for e in experiences], dtype=np.float32)
        next_action_masks = np.array([e[6] for e in experiences], dtype=np.float32)
        
        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            action_masks,
            next_action_masks
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum samples needed
        
        Returns:
            True if buffer has at least min_size samples
        """
        return len(self.buffer) >= min_size

