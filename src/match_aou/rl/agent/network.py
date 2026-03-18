"""
MAPPO Actor-Critic Network (CTDE)
==================================

Multi-Agent PPO with Centralized Training, Decentralized Execution.

Architecture:
    Actor (shared weights, decentralized):
        local_obs [30] → 128 → tanh → 64 → tanh → logits [5]
        + action masking (invalid → -inf before softmax)

    Critic (centralized):
        global_state [30 * N_agents] → 128 → tanh → 64 → tanh → V(s) [1]
        Sees all agents' observations concatenated.

Design Choices:
    - Parameter sharing: All agents use same actor weights (homogeneous F-16s)
    - tanh activations: Standard for PPO, prevents exploding activations
    - Orthogonal initialization: Proven to help PPO convergence
    - Action masking via logits: Set invalid logits to -inf → 0 probability
    - Separate actor/critic: Critic gets global info, actor only local

CTDE Explained:
    Training:  Critic sees global_state = [obs_agent_0 || obs_agent_1 || ...]
               This helps it evaluate "should agent A attack, given agent B's state?"
    Execution: Each agent runs the Actor with only its local observation.
               No communication needed between agents at deployment time.

Usage:
    from match_aou.rl.agent import ActorCriticNetwork

    net = ActorCriticNetwork(obs_dim=30, action_dim=5, n_agents=2)

    # Actor (each agent independently):
    obs_local = torch.randn(1, 30)
    mask = torch.tensor([[True, True, False, True, True]])
    dist = net.get_distribution(obs_local, mask)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    # Critic (centralized, sees all agents):
    obs_global = torch.randn(1, 60)  # [obs_A || obs_B]
    value = net.get_value(obs_global)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Optional, Tuple


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    """
    Orthogonal initialization for a linear layer.

    Standard in PPO implementations (CleanRL, MAPPO paper).
    Helps with training stability and convergence.

    Args:
        layer: Linear layer to initialize
        std: Standard deviation for orthogonal init (sqrt(2) for hidden, 0.01 for output)
        bias_const: Constant for bias initialization
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNetwork(nn.Module):
    """
    MAPPO Actor-Critic with Centralized Training, Decentralized Execution.

    The actor and critic are separate networks (no shared layers).
    This is the standard MAPPO approach — shared layers can cause
    conflicting gradients between policy and value objectives.

    Attributes:
        obs_dim: Local observation dimension per agent (30)
        action_dim: Number of discrete actions (5)
        n_agents: Number of agents in the scenario (2)
        global_obs_dim: Critic input size = obs_dim * n_agents (60)
    """

    def __init__(
        self,
        obs_dim: int = 30,
        action_dim: int = 5,
        n_agents: int = 2,
        hidden_size: int = 128,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            obs_dim: Local observation dimension per agent (30)
            action_dim: Number of discrete actions (5)
            n_agents: Number of agents (for critic global state)
            hidden_size: Hidden layer size (128)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.global_obs_dim = obs_dim * n_agents
        self.hidden_size = hidden_size

        # --- Actor (decentralized) ---
        # Input: local observation [obs_dim]
        # Output: action logits [action_dim]
        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size // 2)),
            nn.Tanh(),
            # Small std (0.01) for output layer → initial policy close to uniform
            _layer_init(nn.Linear(hidden_size // 2, action_dim), std=0.01),
        )

        # --- Critic (centralized) ---
        # Input: global state [obs_dim * n_agents]
        # Output: state value V(s) [1]
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(self.global_obs_dim, hidden_size)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_size, hidden_size // 2)),
            nn.Tanh(),
            # std=1.0 for value output (larger range)
            _layer_init(nn.Linear(hidden_size // 2, 1), std=1.0),
        )

    def get_distribution(
        self,
        local_obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """
        Get action probability distribution from local observation.

        This is what each agent runs independently during execution.

        Process:
            1. Forward through actor → raw logits
            2. Apply action mask (invalid → -inf)
            3. Softmax → probabilities
            4. Return Categorical distribution

        Args:
            local_obs: Agent's local observation [batch, obs_dim] or [obs_dim]
            action_mask: Boolean mask [batch, action_dim] or [action_dim]
                True = valid action, False = invalid

        Returns:
            Categorical distribution over actions
        """
        if local_obs.dim() == 1:
            local_obs = local_obs.unsqueeze(0)

        logits = self.actor(local_obs)

        # Apply action masking: invalid actions get -inf logits → 0 probability
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            if action_mask.dtype != torch.bool:
                action_mask = action_mask.bool()
            logits = logits.masked_fill(~action_mask, float("-inf"))

        return Categorical(logits=logits)

    def get_value(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Get state value from centralized (global) observation.

        This is only used during TRAINING — the critic sees all agents'
        observations concatenated: [obs_agent_0 || obs_agent_1 || ...]

        Args:
            global_obs: Global state [batch, obs_dim * n_agents] or [obs_dim * n_agents]

        Returns:
            State value [batch, 1]
        """
        if global_obs.dim() == 1:
            global_obs = global_obs.unsqueeze(0)

        return self.critic(global_obs)

    def get_action_and_value(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value in one call.

        Used during training to compute all quantities needed for PPO loss.

        Args:
            local_obs: Local observation [batch, obs_dim]
            global_obs: Global observation [batch, global_obs_dim]
            action_mask: Action mask [batch, action_dim]
            action: If provided, compute log_prob for this action (for PPO update).
                    If None, sample a new action.

        Returns:
            (action, log_prob, entropy, value) tuple:
            - action: Selected action [batch]
            - log_prob: Log probability of selected action [batch]
            - entropy: Distribution entropy [batch] (for exploration bonus)
            - value: State value from critic [batch, 1]
        """
        dist = self.get_distribution(local_obs, action_mask)
        value = self.get_value(global_obs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_greedy_action(
        self,
        local_obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Select best action deterministically (for evaluation / deployment).

        Args:
            local_obs: Local observation [obs_dim]
            action_mask: Action mask [action_dim]

        Returns:
            Action index (int)
        """
        with torch.no_grad():
            dist = self.get_distribution(local_obs, action_mask)
            # Pick action with highest probability
            return dist.probs.argmax(dim=-1).item()

    def save(self, path: str):
        """Save network to file."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "n_agents": self.n_agents,
                "hidden_size": self.hidden_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ActorCriticNetwork":
        """Load network from file."""
        checkpoint = torch.load(path, weights_only=False)
        network = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            n_agents=checkpoint["n_agents"],
            hidden_size=checkpoint["hidden_size"],
        )
        network.load_state_dict(checkpoint["state_dict"])
        return network


# =============================================================================
# Backward Compatibility — keep DQN network available
# =============================================================================

class EnhancedMLPQNetwork(nn.Module):
    """Legacy DQN Q-Network. Kept for backward compatibility.
    For new work, use ActorCriticNetwork with PPOTrainer."""

    def __init__(self, obs_dim=30, action_dim=5, hidden_sizes=None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes or [128, 64]
        layers = []
        layers.append(nn.Linear(obs_dim, self.hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(self.hidden_sizes) - 1):
            layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_sizes[-1], action_dim))
        self.network = nn.Sequential(*layers)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs, action_mask=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        q = self.network(obs)
        if action_mask is not None:
            if action_mask.dtype != torch.bool:
                action_mask = action_mask.bool()
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            q = q.masked_fill(~action_mask, float("-inf"))
        return q

    def get_action(self, obs, action_mask, epsilon=0.0):
        if np.random.random() < epsilon:
            mask = action_mask.squeeze(0) if action_mask.dim() == 2 else action_mask
            valid = torch.where(mask)[0].cpu().numpy()
            return np.random.choice(valid) if len(valid) > 0 else 0
        with torch.no_grad():
            return self.forward(obs, action_mask).argmax(dim=-1).item()

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "obs_dim": self.obs_dim,
                     "action_dim": self.action_dim, "hidden_sizes": self.hidden_sizes}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, weights_only=False)
        net = cls(obs_dim=ckpt["obs_dim"], action_dim=ckpt["action_dim"],
                  hidden_sizes=ckpt["hidden_sizes"])
        net.load_state_dict(ckpt["state_dict"])
        return net


def create_target_network(q_network):
    target = EnhancedMLPQNetwork(q_network.obs_dim, q_network.action_dim, q_network.hidden_sizes)
    target.load_state_dict(q_network.state_dict())
    target.eval()
    return target


def soft_update_target_network(q_net, target_net, tau=0.005):
    for tp, qp in zip(target_net.parameters(), q_net.parameters()):
        tp.data.copy_(tau * qp.data + (1.0 - tau) * tp.data)


def hard_update_target_network(q_net, target_net):
    target_net.load_state_dict(q_net.state_dict())
