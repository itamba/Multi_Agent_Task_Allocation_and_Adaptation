"""
DQN Trainer - Imitation Learning from MATCH-AOU
================================================

Trains Q-Network to imitate MATCH-AOU's optimal decisions.

Training approach:
1. Behavioral Cloning: Learn from oracle demonstrations
2. Q-Learning: Improve through self-play (optional)
3. Action Masking: Ensure safe actions only

Usage:
    from match_aou.rl.training import DQNTrainer
    from match_aou.rl.agent import EnhancedMLPQNetwork
    
    q_network = EnhancedMLPQNetwork()
    trainer = DQNTrainer(q_network)
    
    # Training loop
    for episode in range(100):
        trainer.train_episode(env, oracle)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from .buffer import ReplayBuffer
from ..agent.network import (
    EnhancedMLPQNetwork,
    create_target_network,
    soft_update_target_network,
    hard_update_target_network
)
from .reward import compute_reward, RewardConfig, RewardTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.
    
    Attributes:
        # Network
        obs_dim: Observation dimension (30)
        action_dim: Number of actions (5)
        hidden_sizes: Hidden layer sizes [128, 64]
        
        # Optimization
        learning_rate: Learning rate for Adam (0.001)
        gamma: Discount factor (0.99)
        batch_size: Batch size for training (32)
        
        # Experience replay
        buffer_size: Replay buffer capacity (10000)
        min_buffer_size: Min samples before training (1000)
        
        # Target network
        target_update_freq: Hard update frequency (100 steps)
        tau: Soft update rate (0.005, if using soft updates)
        use_soft_updates: Use soft vs hard updates (False)
        
        # Exploration
        epsilon_start: Initial exploration rate (1.0)
        epsilon_end: Final exploration rate (0.01)
        epsilon_decay: Decay rate (0.995)
        
        # Training
        train_freq: Train every N steps (1)
        max_grad_norm: Gradient clipping (10.0)
        
        # Reward
        reward_config: RewardConfig instance
        
        # Logging
        log_freq: Log every N steps (100)
        save_freq: Save model every N steps (1000)
        model_dir: Directory for saving models
    """
    
    # Network
    obs_dim: int = 30
    action_dim: int = 5
    hidden_sizes: List[int] = None
    
    # Optimization
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 32
    
    # Experience replay
    buffer_size: int = 10000
    min_buffer_size: int = 1000
    
    # Target network
    target_update_freq: int = 100
    tau: float = 0.005
    use_soft_updates: bool = False
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training
    train_freq: int = 1
    max_grad_norm: float = 10.0
    
    # Reward
    reward_config: Optional[RewardConfig] = None
    
    # Logging
    log_freq: int = 100
    save_freq: int = 1000
    model_dir: str = "models"
    
    def __post_init__(self):
        """Set defaults."""
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64]
        
        if self.reward_config is None:
            self.reward_config = RewardConfig()


class DQNTrainer:
    """
    DQN trainer with imitation learning from oracle.
    
    Combines:
    - Behavioral cloning (learn from oracle)
    - Q-learning (improve through experience)
    - Action masking (safety)
    
    Attributes:
        q_network: Main Q-network
        target_network: Target network for stable learning
        optimizer: Adam optimizer
        loss_fn: MSE loss
        buffer: Experience replay buffer
        config: Training configuration
    """
    
    def __init__(
        self,
        q_network: EnhancedMLPQNetwork,
        config: Optional[TrainingConfig] = None,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            q_network: Q-network to train
            config: Training configuration
            device: 'cpu' or 'cuda'
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(device)
        
        # Networks
        self.q_network = q_network.to(self.device)
        self.target_network = create_target_network(q_network).to(self.device)
        
        # Optimization
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate
        )
        self.loss_fn = nn.MSELoss()
        
        # Experience replay
        self.buffer = ReplayBuffer(capacity=self.config.buffer_size)
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.epsilon = self.config.epsilon_start
        
        # Metrics
        self.reward_tracker = RewardTracker()
        self.metrics = {
            'loss': [],
            'q_values': [],
            'epsilon': [],
            'imitation_accuracy': []
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup model directory."""
        self.model_dir = Path(self.config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(self) -> Optional[float]:
        """
        Single training step (sample batch and update network).
        
        Returns:
            Loss value or None if buffer not ready
        """
        # Check if buffer has enough samples
        if not self.buffer.is_ready(self.config.min_buffer_size):
            return None
        
        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)
        if batch is None:
            return None
        
        # Unpack batch
        states, actions, rewards, next_states, dones, masks, next_masks = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device).bool()
        next_masks = torch.FloatTensor(next_masks).to(self.device).bool()
        
        # Compute current Q-values
        current_q = self.q_network(states, masks)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            # Get next Q-values from target network
            next_q = self.target_network(next_states, next_masks)
            
            # Max Q-value for next state
            next_q_max = next_q.max(dim=1)[0]
            
            # TD target: r + gamma * max_a' Q(s', a')
            target_q = rewards + (1 - dones) * self.config.gamma * next_q_max
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        
        if self.config.use_soft_updates:
            soft_update_target_network(
                self.q_network,
                self.target_network,
                self.config.tau
            )
        elif self.step_count % self.config.target_update_freq == 0:
            hard_update_target_network(self.q_network, self.target_network)
            logger.debug(f"Updated target network at step {self.step_count}")
        
        # Log metrics
        self.metrics['loss'].append(loss.item())
        
        return loss.item()
    
    def add_experience(
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
        Add experience to replay buffer.
        
        Args:
            state: Current state [obs_dim]
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done?
            action_mask: Valid actions in current state
            next_action_mask: Valid actions in next state
        """
        self.buffer.add(
            state, action, reward, next_state, done,
            action_mask, next_action_mask
        )
    
    def get_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            action_mask: Valid actions
            training: If True, use epsilon-greedy; else greedy
        
        Returns:
            Selected action index
        """
        epsilon = self.epsilon if training else 0.0
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        mask_tensor = torch.BoolTensor(action_mask).to(self.device)
        
        # Get action from network
        with torch.no_grad():
            action = self.q_network.get_action(
                state_tensor,
                mask_tensor,
                epsilon=epsilon
            )
        
        return action
    
    def update_epsilon(self):
        """Decay epsilon (exploration rate)."""
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        self.metrics['epsilon'].append(self.epsilon)
    
    def log_metrics(self, additional_metrics: Optional[Dict] = None):
        """
        Log training metrics.
        
        Args:
            additional_metrics: Extra metrics to log
        """
        if len(self.metrics['loss']) == 0:
            return
        
        # Compute statistics
        recent_loss = np.mean(self.metrics['loss'][-self.config.log_freq:])
        
        reward_stats = self.reward_tracker.get_stats()
        
        logger.info(
            f"Step {self.step_count} | "
            f"Episode {self.episode_count} | "
            f"Loss: {recent_loss:.4f} | "
            f"Epsilon: {self.epsilon:.3f} | "
            f"Accuracy: {reward_stats['accuracy']:.2%} | "
            f"Avg Reward: {reward_stats['mean']:.3f}"
        )
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                logger.info(f"  {key}: {value}")
    
    def save_checkpoint(self, filename: Optional[str] = None):
        """
        Save training checkpoint.
        
        Args:
            filename: Checkpoint filename (auto-generated if None)
        """
        if filename is None:
            filename = f"checkpoint_step_{self.step_count}.pt"
        
        checkpoint_path = self.model_dir / filename
        
        torch.save({
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'q_network_state': self.q_network.state_dict(),
            'target_network_state': self.target_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.epsilon = checkpoint['epsilon']
        
        self.q_network.load_state_dict(checkpoint['q_network_state'])
        self.target_network.load_state_dict(checkpoint['target_network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        self.metrics = checkpoint['metrics']
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        logger.info(f"Resumed at step {self.step_count}, episode {self.episode_count}")
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of training metrics."""
        return {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.metrics['loss'][-100:]) if self.metrics['loss'] else 0.0,
            'reward_stats': self.reward_tracker.get_stats()
        }


# =============================================================================
# Training Loop Helpers
# =============================================================================

def train_episode(
    trainer: DQNTrainer,
    env,  # BLADE environment
    oracle,  # Oracle that provides optimal actions
    max_steps: int = 1000
) -> Dict:
    """
    Train for one episode.
    
    Args:
        trainer: DQNTrainer instance
        env: BLADE environment
        oracle: Function that returns optimal action
        max_steps: Maximum steps per episode
    
    Returns:
        Episode statistics
    """
    state = env.reset()  # Initial observation
    episode_reward = 0.0
    episode_steps = 0
    
    for step in range(max_steps):
        # Get action mask
        action_mask = oracle.get_action_mask(state)
        
        # Get RL action
        rl_action = trainer.get_action(
            state=state.vector,
            action_mask=action_mask,
            training=True
        )
        
        # Get oracle action (optimal)
        oracle_action = oracle.get_action(state)
        
        # Execute action in environment
        next_state, _, done, truncated, _ = env.step(rl_action)
        
        # Compute reward (compare to oracle)
        reward = compute_reward(
            rl_action=rl_action,
            oracle_action=oracle_action,
            observation=state,
            is_valid=(action_mask[rl_action] if rl_action < len(action_mask) else False),
            config=trainer.config.reward_config
        )
        
        # Get next action mask
        next_action_mask = oracle.get_action_mask(next_state)
        
        # Add to replay buffer
        trainer.add_experience(
            state=state.vector,
            action=rl_action,
            reward=reward,
            next_state=next_state.vector,
            done=done or truncated,
            action_mask=action_mask,
            next_action_mask=next_action_mask
        )
        
        # Train
        if step % trainer.config.train_freq == 0:
            loss = trainer.train_step()
        
        # Track rewards
        trainer.reward_tracker.add_reward(
            reward,
            is_match=(rl_action == oracle_action)
        )
        
        episode_reward += reward
        episode_steps += 1
        state = next_state
        
        if done or truncated:
            break
    
    # Update epsilon
    trainer.update_epsilon()
    trainer.episode_count += 1
    
    return {
        'episode_reward': episode_reward,
        'episode_steps': episode_steps,
        'imitation_accuracy': trainer.reward_tracker.get_stats()['accuracy']
    }

