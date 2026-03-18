# RL Module Documentation

**Reinforcement Learning system for MATCH-AOU tactical decision-making**

Version: 2.0.0
Last Updated: March 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Structure](#module-structure)
4. [Quick Start](#quick-start)
5. [Components Reference](#components-reference)
6. [Training Pipeline](#training-pipeline)
7. [Integration with MATCH-AOU & BLADE](#integration)
8. [Configuration](#configuration)
9. [Examples](#examples)

---

## Overview

The RL module enables agents to learn tactical decision-making through imitation of MATCH-AOU's optimal solutions. The system bridges three key components:

- **MATCH-AOU**: MINLP solver providing optimal task allocation (the "oracle")
- **BLADE**: Physics-based simulation environment (Panopticon fork)
- **RL Agent**: MAPPO neural network learning to adapt plans in real-time

### Key Features

- **MAPPO (CTDE)**: Multi-Agent PPO with Centralized Training, Decentralized Execution
- **Shared actor weights**: All F-16 agents use the same policy (homogeneous agents)
- **Centralized critic**: Critic sees global state (all agents concatenated) during training
- **Partial observability**: Agent discovers targets gradually (unlike omniscient MATCH-AOU)
- **Online adaptation**: Real-time plan modifications during mission execution
- **Action masking**: Ensures only valid/safe actions are selected
- **Imitation learning**: Reward signal from comparing RL decisions to oracle solutions

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                       MAPPO RL SYSTEM (CTDE)                      │
│                                                                   │
│  ┌────────────┐    ┌──────────────┐    ┌────────────┐           │
│  │            │    │    Actor     │    │            │           │
│  │ Observation│───▶│  (shared,    │───▶│Plan Editor │           │
│  │  Builder   │    │ decentralized│    │            │           │
│  │            │    │   per agent) │    │            │           │
│  └────────────┘    └──────────────┘    └────────────┘           │
│        │                  │                  │                    │
│        │           ┌──────────────┐          │                   │
│        │           │   Critic     │          │                   │
│        └──────────▶│(centralized, │          │                   │
│         (global)   │ training only│          │                   │
│                    └──────────────┘          │                   │
│        ▼                  ▼                  ▼                    │
│   [30 features]      [5 actions]    [BLADE command]              │
│    per agent          + V(s)                                     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
         │                                         │
         ▼                                         ▼
  ┌─────────────┐                          ┌─────────────┐
  │    BLADE    │                          │  MATCH-AOU  │
  │  Scenario   │                          │   Oracle    │
  └─────────────┘                          └─────────────┘
```

### CTDE Explained

- **Training**: The critic sees `global_state = [obs_agent_0 || obs_agent_1]` (concatenation of all agents' observations). This helps evaluate joint outcomes.
- **Execution**: Each agent runs the actor independently with only its local observation. No communication between agents at deployment time.
- **Parameter sharing**: All F-16 agents share the same actor weights (homogeneous).

### Pipeline Flow

1. **Observation**: Extract 30-feature vector from BLADE state (per agent)
2. **Global state**: Concatenate all agents' observations for the critic
3. **Action Selection**: Actor samples action from policy distribution
4. **Value Estimation**: Critic estimates state value from global state
5. **Plan Editing**: Convert action to BLADE execution command
6. **Execution**: BLADE simulates one step
7. **Reward**: Compare RL decision to oracle (imitation learning)
8. **PPO Update**: After episode, compute GAE advantages and update network

---

## Module Structure

```
src/match_aou/rl/
├── shared_utils.py              # Common utilities (haversine, normalize, etc.)
│
├── observation/                 # Phase 1: State Extraction
│   ├── __init__.py             # Public API: build_observation_vector()
│   ├── config.py               # ObservationConfig
│   ├── observation_types.py    # SelfState, TargetInfo, ObservationOutput
│   ├── observation_builder.py  # Main builder
│   ├── observation_utils.py    # Observation-specific helpers
│   ├── self_features.py        # Agent's own state (6 features)
│   ├── target_extraction.py    # Enemy detection (K×6 features)
│   ├── plan_parsing.py         # Plan analysis
│   └── plan_context.py         # Plan context (6 features)
│
├── action/                      # Phase 2: Action Space
│   ├── __init__.py             # Public API: compute_action_mask()
│   ├── action_config.py        # ActionType, ActionSpaceConfig
│   ├── action_validation.py    # ActionValidator, ActionMask
│   └── action_utils.py         # Action-specific helpers
│
├── plan_editor.py              # Phase 3: Action → BLADE Command
│
├── agent/                       # Phase 4: Neural Network
│   ├── __init__.py             # Exports ActorCriticNetwork
│   └── network.py              # ActorCriticNetwork (MAPPO), legacy EnhancedMLPQNetwork
│
└── training/                    # Phase 5: Training System (MAPPO)
    ├── __init__.py             # Training API exports
    ├── ppo_trainer.py          # PPOTrainer + PPOConfig
    ├── rollout_buffer.py       # On-policy trajectory storage with GAE
    ├── reward.py               # Reward functions (imitation + shaping)
    ├── oracle.py               # MATCH-AOU oracle wrappers
    └── episode_initializer.py  # Episode setup (task split, solving)
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt

# Required: torch>=2.0.0, numpy>=1.24.0, gymnasium>=0.28.0
```

### Basic Usage

```python
from match_aou.rl.agent import ActorCriticNetwork
from match_aou.rl.observation import build_observation_vector, ObservationConfig
from match_aou.rl.action import compute_action_mask
from match_aou.rl.plan_editor import plan_edit_to_blade_action

# 1. Extract observation
obs = build_observation_vector(
    scenario=blade_scenario,
    agent_id="f16_01",
    current_plan=plan,
    current_time=100,
    config=ObservationConfig(top_k=3),
    tasks=tasks,
    solution=solution,
)

# 2. Select action (using trained network)
network = ActorCriticNetwork.load('trained_model.pt')
action = network.get_greedy_action(
    local_obs=torch.tensor(obs.vector),
    action_mask=torch.tensor(mask),
)

# 3. Convert to BLADE command
blade_action = plan_edit_to_blade_action(
    action_token=action,
    observation_output=obs,
    scenario=blade_scenario,
    agent_id="f16_01",
)

# 4. Execute in BLADE
next_obs, reward, done, _, _ = blade_env.step(blade_action)
```

### Training

```python
from match_aou.rl.agent import ActorCriticNetwork
from match_aou.rl.training import PPOTrainer, PPOConfig, RewardConfig

# Setup MAPPO
network = ActorCriticNetwork(obs_dim=30, action_dim=5, n_agents=2)
config = PPOConfig(
    obs_dim=30, action_dim=5, n_agents=2,
    learning_rate=3e-4,
    reward_config=RewardConfig(use_shaping=True),
)
trainer = PPOTrainer(network, config)

# Training is done via train_full.py which handles the full pipeline
# See: python train_full.py --scenario data/scenarios/strike_training_2v3.json
```

---

## Components Reference

### 1. Observation Space

**Purpose**: Extract structured features from BLADE scenario

**API**:
```python
from match_aou.rl.observation import build_observation_vector, ObservationConfig

config = ObservationConfig(
    top_k=3,              # Track top-3 nearest targets
    max_plan_length=10,   # Max tasks in plan
    max_distance_km=500.0 # Normalization bound
)

obs = build_observation_vector(
    scenario=blade_scenario,
    agent_id="f16_01",
    current_plan=plan,
    current_time=100,
    config=config,
    tasks=tasks,
    solution=solution,
)

# Returns: ObservationOutput
# - obs.vector: np.ndarray [30]  (default top_k=3)
# - obs.self_state: SelfState (fuel, weapons, etc.)
# - obs.targets: List[TargetInfo] (K targets)
# - obs.agent_id: str
# - obs.current_time: int
```

**Feature Vector** (30 features with top_k=3):
```
[0-5]    Self features (6):
         - fuel_norm
         - has_weapon
         - dist_to_next_step_norm
         - next_step_is_attack
         - rtb_possible
         - plan_progress

[6-23]   Target features (3 x 6 = 18):
         For each of top-3 targets:
         - exists
         - distance_norm
         - is_threat
         - is_dynamic
         - is_in_plan
         - already_engaged

[24-29]  Plan context features (6):
         - fuel_margin_for_plan
         - num_unassigned_targets_norm
         - remaining_tasks_norm
         - next_target_visible
         - coordination_bonus
         - time_to_next_attack_norm
```

**Configuration**:
- `top_k`: Number of targets to track (1-3, limited by ActionType)
- Vector size = 6 + (K x 6) + 6 = 12 + 6K
- Default: top_k=3 -> 30 features

---

### 2. Action Space

**Purpose**: Define discrete actions and validate legality

**API**:
```python
from match_aou.rl.action import (
    compute_action_mask,
    validate_action,
    ActionSpaceConfig,
    ActionType,
)

config = ActionSpaceConfig(
    top_k=3,                      # Must match observation
    enable_rtb=True,
    enable_noop=True,
    min_attack_fuel_margin=0.3,  # Need 30% fuel to attack
    min_rtb_distance_km=10.0,    # Must be >10km from base
    attack_cooldown_ticks=50,    # Wait 50 ticks between attacks
)

# Get action mask
mask = compute_action_mask(
    observation_output=obs,
    scenario=blade_scenario,
    agent_id="f16_01",
    config=config,
    last_attack_tick=None
)

# Check validity
valid_actions = mask.get_valid_actions()  # e.g., [0, 1, 4]
is_valid = mask.is_valid(2)               # False
reason = mask.get_invalid_reason(2)       # "Target 1 does not exist"
```

**Actions** (5 total with top_k=3):
```
0: NOOP           - Continue current plan (no changes)
1: INSERT_ATTACK_0 - Attack target in slot 0
2: INSERT_ATTACK_1 - Attack target in slot 1
3: INSERT_ATTACK_2 - Attack target in slot 2
4: FORCE_RTB      - Return to base immediately
```

**Validation Rules**:
- `NOOP`: Always valid
- `INSERT_ATTACK(k)`: Valid if:
  - Target k exists
  - Target k not already in plan
  - Agent has weapons
  - Fuel >= min_attack_fuel_margin
  - Time since last attack >= attack_cooldown_ticks
- `FORCE_RTB`: Valid if:
  - Not already RTB
  - RTB is possible (enough fuel)
  - Distance from base > min_rtb_distance_km

**IMPORTANT LIMITATION**:
```python
# top_k currently LIMITED to 1-3 due to hard-coded ActionType enum
# To extend to top_k > 3, you must:
# 1. Add INSERT_ATTACK_3, INSERT_ATTACK_4 to ActionType (action_config.py)
# 2. Update validation logic
# 3. Update action_utils.py range checks
```

---

### 3. Plan Editor

**Purpose**: Convert RL action tokens to BLADE execution commands

**API**:
```python
from match_aou.rl import plan_edit_to_blade_action, preview_blade_action

# Convert action to BLADE command
blade_action = plan_edit_to_blade_action(
    action_token=1,          # INSERT_ATTACK_0
    observation_output=obs,
    scenario=blade_scenario,
    agent_id="f16_01",
    weapon_quantity=2
)
# Returns: "handle_aircraft_attack('f16_01', 'sam_05', 'aim_120', 2)"

# Preview without execution (for debugging)
preview = preview_blade_action(
    action_token=1,
    observation_output=obs,
    scenario=blade_scenario,
    agent_id="f16_01"
)
# Returns: {
#   'action_type': 'INSERT_ATTACK(0)',
#   'blade_action': "handle_aircraft_attack(...)",
#   'target_id': 'sam_05',
#   'weapon_id': 'aim_120',
#   'valid': True,
#   'error': None
# }
```

**Action Mapping**:
```
NOOP           -> ""                              (empty = continue plan)
INSERT_ATTACK_k -> "handle_aircraft_attack(...)" (insert attack in plan)
FORCE_RTB      -> "aircraft_return_to_base(...)" (immediate RTB)
```

**Weapon Selection**:
- Automatically selects best available weapon
- Uses `aircraft.get_weapon_with_highest_engagement_range()`
- Falls back to first available weapon

---

### 4. Neural Network (MAPPO Actor-Critic)

**Purpose**: Policy network (actor) and value network (critic) for MAPPO

**API**:
```python
from match_aou.rl.agent import ActorCriticNetwork

# Create network
network = ActorCriticNetwork(
    obs_dim=30,       # Local observation dimension
    action_dim=5,     # Number of discrete actions
    n_agents=2,       # Number of agents (for critic input size)
    hidden_size=128,  # Hidden layer width
)

# Actor: get action distribution (decentralized, per agent)
local_obs = torch.randn(1, 30)
mask = torch.tensor([[True, True, False, True, True]])
dist = network.get_distribution(local_obs, mask)
action = dist.sample()
log_prob = dist.log_prob(action)

# Critic: get state value (centralized, sees all agents)
global_obs = torch.randn(1, 60)  # [obs_agent_0 || obs_agent_1]
value = network.get_value(global_obs)

# Combined (for training):
action, log_prob, entropy, value = network.get_action_and_value(
    local_obs, global_obs, mask
)

# Greedy action (for evaluation / deployment)
best_action = network.get_greedy_action(local_obs, mask)

# Save/Load
network.save('model.pt')
loaded = ActorCriticNetwork.load('model.pt')
```

**Architecture**:
```
Actor (decentralized):
  local_obs [30] -> FC(128) -> Tanh -> FC(64) -> Tanh -> logits [5]
  + action masking (invalid logits -> -inf before softmax)

Critic (centralized):
  global_state [60] -> FC(128) -> Tanh -> FC(64) -> Tanh -> V(s) [1]
```

**Design choices**:
- Orthogonal initialization (standard for PPO, helps convergence)
- Tanh activations (prevents exploding activations in PPO)
- Separate actor/critic (no shared layers; avoids conflicting gradients)
- Small std (0.01) for actor output layer (initial policy close to uniform)
- Parameter sharing: all agents use the same actor weights

---

### 5. Training System (MAPPO)

#### Rollout Buffer

On-policy trajectory storage. Unlike DQN's replay buffer, this stores one episode, computes GAE advantages, then gets cleared.

```python
from match_aou.rl.training import RolloutBuffer

buffer = RolloutBuffer(
    obs_dim=30,
    global_obs_dim=60,
    action_dim=5,
    capacity=2048,
    gamma=0.99,
    gae_lambda=0.95,
)

# During episode: store transitions
buffer.store(
    local_obs=obs_vector,       # [30]
    global_obs=global_vector,   # [60]
    action=1,                   # int
    log_prob=-1.2,              # float
    reward=1.0,                 # float
    value=0.5,                  # float (from critic)
    done=False,                 # bool
    action_mask=mask,           # [5]
    oracle_action=1,            # int (optional, for tracking)
)

# After episode: compute advantages
buffer.compute_returns_and_advantages(last_value=0.0)

# Training: iterate in mini-batches
for batch in buffer.get_batches(batch_size=64):
    # batch dict: obs, global_obs, actions, old_log_probs,
    #             advantages, returns, action_masks
    pass

# Reset for next episode
buffer.reset()
```

**GAE (Generalized Advantage Estimation)**:
```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)       # TD error
A_t = delta_t + (gamma * lambda) * A_{t+1}          # GAE (recursive)
```
- lambda=0: Pure TD (high bias, low variance)
- lambda=1: Pure MC (low bias, high variance)
- lambda=0.95: Standard PPO balance

#### Reward Function

```python
from match_aou.rl.training import compute_reward, RewardConfig

config = RewardConfig(
    imitation_reward=1.0,           # Match oracle
    imitation_penalty=-0.5,         # Mismatch oracle
    fuel_efficiency_bonus=0.1,      # Bonus for fuel surplus
    target_coverage_bonus=0.2,      # Bonus for attacking unassigned
    invalid_action_penalty=-5.0,    # Strong penalty
    use_shaping=True                # Enable reward shaping
)

reward = compute_reward(
    rl_action=1,          # What RL chose
    oracle_action=1,      # What oracle chose
    observation=obs,
    is_valid=True,
    config=config
)
# Returns: 1.0 (perfect match) + bonuses
```

**Reward Components**:
1. **Imitation**: +1.0 for match, -0.5 for mismatch
2. **Fuel efficiency**: +0.1 if fuel_margin > 0.5
3. **Target coverage**: +0.2 if attacking unassigned target
4. **Invalid action**: -5.0 (safety penalty)

#### Oracle

```python
from match_aou.rl.training import MatchAOUOracle, SimpleOracle

# Full oracle (uses MATCH-AOU solver)
oracle = MatchAOUOracle(solver_name='bonmin')
action = oracle.get_action(
    observation=obs,
    agent_id="f16_01",
    tasks=all_tasks,
    current_solution=current_sol,
    scenario=blade_scenario,
    new_targets=discovered_targets
)

# Simple oracle (heuristic-based, for testing)
simple_oracle = SimpleOracle()
action = simple_oracle.get_action(obs, "f16_01")
# Rules: Low fuel -> RTB, Unassigned nearby -> Attack, Else -> NOOP
```

#### PPO Trainer

```python
from match_aou.rl.agent import ActorCriticNetwork
from match_aou.rl.training import PPOTrainer, PPOConfig, RewardConfig

config = PPOConfig(
    # Network
    obs_dim=30,
    action_dim=5,
    n_agents=2,
    hidden_size=128,

    # PPO core
    clip_eps=0.2,           # Clipping parameter epsilon
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda

    # Training
    learning_rate=3e-4,     # Standard PPO lr
    ppo_epochs=4,           # K epochs per update
    batch_size=64,          # Mini-batch size
    max_grad_norm=0.5,      # Gradient clipping

    # Loss coefficients
    value_coef=0.5,         # Weight of value loss
    entropy_coef=0.01,      # Weight of entropy bonus

    # Buffer
    buffer_capacity=2048,

    # Reward
    reward_config=RewardConfig(use_shaping=True),

    # Output
    model_dir="training_output/models",
)

network = ActorCriticNetwork(
    obs_dim=config.obs_dim,
    action_dim=config.action_dim,
    n_agents=config.n_agents,
)
trainer = PPOTrainer(network, config)

# Per decision point:
action, log_prob, value = trainer.get_action(
    local_obs=obs_vector,       # [30] numpy
    global_obs=global_vector,   # [60] numpy
    action_mask=mask,           # [5] numpy bool
)

# Store transition:
trainer.buffer.store(
    local_obs=obs_vector,
    global_obs=global_vector,
    action=action,
    log_prob=log_prob,
    reward=reward,
    value=value,
    done=False,
    action_mask=mask,
)

# After episode: PPO update
trainer.buffer.compute_returns_and_advantages(last_value=0.0)
update_metrics = trainer.update()
# Returns: {policy_loss, value_loss, entropy, clip_fraction, approx_kl, total_loss}

trainer.buffer.reset()

# Save
trainer.save_checkpoint("checkpoint_ep50.pt")
```

**PPO Update Process**:
1. Compute GAE advantages from collected trajectory
2. For K epochs (default 4):
   a. Sample mini-batches from buffer
   b. Compute new log_probs and values from current network
   c. Policy loss: clipped surrogate objective
   d. Value loss: MSE between V(s) and returns
   e. Entropy bonus: encourages exploration
   f. Total loss = policy_loss + c1 * value_loss - c2 * entropy
3. Gradient clipping (max_grad_norm=0.5)
4. Adam optimizer step

---

## Training Pipeline

The main training script is `train_full.py` at the project root.

### Full Pipeline

```bash
python train_full.py --scenario data/scenarios/strike_training_2v3.json --episodes 50
```

**What happens per episode:**

1. **Reset BLADE** environment
2. **Create agents** from scenario (2 F-16s)
3. **Generate tasks** from enemy targets (3 hostile facilities)
4. **Split tasks**: partial (67%) vs full (all)
5. **Solve MATCH-AOU** twice: partial solution (agent's plan) + full solution (oracle)
6. **Launch aircraft** from airbases
7. **Setup executor** with partial plan (BladeExecutorMinimal)
8. **Simulation loop** (up to 14,400 ticks):
   - Executor issues BLADE commands from partial plan
   - At decision points (every 100 ticks + discovery events):
     - Build local observation per agent (30 features)
     - Concatenate all agents -> global observation (60 features)
     - Actor samples action from policy
     - Critic estimates state value
     - Oracle provides ground truth from full solution
     - Compute imitation reward
     - Store transition in rollout buffer
   - If RL decides to attack on discovery, override executor action
9. **PPO update**: compute GAE, run K epochs of PPO
10. **Export recording** and save checkpoint

### Episode Initialization

The task split simulates partial observability:
```python
# Agent knows 67% of targets
partial_tasks = random.sample(all_tasks, int(len(all_tasks) * 2/3))
full_tasks = all_tasks  # Oracle knows everything

# Solve both
partial_solution = solve_match_aou(agents, partial_tasks)
full_solution = solve_match_aou(agents, full_tasks)
```

The RL agent must learn to handle the "hidden" targets when they are discovered during execution.

---

## Integration with MATCH-AOU & BLADE

### Data Flow

```
┌─────────────┐
│  MATCH-AOU  │  Solves full problem (oracle)
│   Solver    │  Input: All agents, all tasks
└──────┬──────┘  Output: Optimal allocation
       │
       ▼
┌─────────────┐
│ Episode Init│  Creates partial/full solutions
└──────┬──────┘  Launches agents
       │
       ▼
┌─────────────┐
│    BLADE    │◄─┐
│  Scenario   │  │  Executor issues plan commands
└──────┬──────┘  │  RL may override on discovery
       │         │
       ▼         │
┌─────────────┐  │
│ Observation │  │  Extract 30 features per agent
│   Builder   │  │  Concatenate -> 60 global features
└──────┬──────┘  │
       │         │
       ▼         │
┌─────────────┐  │
│ MAPPO Actor │──┘  Sample action from policy pi(a|o)
│ + Critic    │     Critic estimates V(s) from global state
└─────────────┘     (PPO update after episode)
```

### BLADE Integration Points

**1. Scenario Access**:
```python
# Get aircraft
aircraft = scenario.get_aircraft(agent_id)
# Access: .latitude, .longitude, .weapons, .fuel, etc.

# Get targets
targets = scenario.get_visible_enemies(agent_id)
# Returns: List of enemy units visible to agent
```

**2. Plan Access**:
```python
# Current plan from MATCH-AOU
plan = current_solution[agent_id]  # [(task, step, level), ...]

# Parse plan
from match_aou.rl.observation import parse_plan_for_agent
tasks, attack_tasks = parse_plan_for_agent(plan, agent_id)
```

**3. Action Execution**:
```python
# BLADE expects string commands
blade_action = "handle_aircraft_attack('f16_01', 'sam_05', 'aim_120', 2)"

# Execute
next_state, reward, done, truncated, info = blade_env.step(blade_action)
```

---

## Configuration

### Observation Configuration

```python
from match_aou.rl.observation import ObservationConfig

config = ObservationConfig(
    top_k=3,                    # Track top-3 targets (LIMITED to 1-3)
    max_plan_length=10,         # Max tasks in plan
    max_distance_km=500.0,      # Distance normalization
    max_speed_kmh=2000.0,       # Speed normalization
    fuel_reserve_fraction=0.15, # Reserve 15% for safety
)
```

### Action Configuration

```python
from match_aou.rl.action import ActionSpaceConfig

config = ActionSpaceConfig(
    top_k=3,                      # MUST match observation.top_k
    enable_rtb=True,              # Allow RTB actions
    enable_noop=True,             # Allow NOOP
    min_attack_fuel_margin=0.3,  # Need 30% fuel to attack
    min_rtb_distance_km=10.0,    # Must be >10km to RTB
    attack_cooldown_ticks=50,    # Cooldown between attacks
)
```

### PPO Training Configuration

```python
from match_aou.rl.training import PPOConfig, RewardConfig

ppo_config = PPOConfig(
    # Network
    obs_dim=30,
    action_dim=5,
    n_agents=2,
    hidden_size=128,

    # PPO core
    clip_eps=0.2,           # Clipping parameter (standard: 0.2)
    gamma=0.99,             # Discount factor
    gae_lambda=0.95,        # GAE lambda (0.95 is standard)

    # Training
    learning_rate=3e-4,     # PPO standard (lower than DQN's 0.001)
    ppo_epochs=4,           # K epochs per update (4-10 typical)
    batch_size=64,          # Mini-batch size
    max_grad_norm=0.5,      # Gradient clipping (PPO standard)

    # Loss coefficients
    value_coef=0.5,         # c1: weight of value loss
    entropy_coef=0.01,      # c2: weight of entropy bonus

    # Buffer
    buffer_capacity=2048,   # Max transitions per episode

    # Reward
    reward_config=RewardConfig(use_shaping=True),
)

reward_config = RewardConfig(
    imitation_reward=1.0,
    imitation_penalty=-0.5,
    fuel_efficiency_bonus=0.1,
    target_coverage_bonus=0.2,
    invalid_action_penalty=-5.0,
    use_shaping=True,
)
```

---

## Examples

### Example 1: Inference (Using Trained Model)

```python
import torch
from match_aou.rl.agent import ActorCriticNetwork
from match_aou.rl.observation import build_observation_vector, ObservationConfig
from match_aou.rl.plan_editor import plan_edit_to_blade_action

# Load trained model
network = ActorCriticNetwork.load('training_output/models/actor_critic_final.pt')
network.eval()

# Game loop
while not done:
    # 1. Observe
    obs = build_observation_vector(
        scenario=blade_scenario,
        agent_id="f16_01",
        current_plan=current_plan,
        current_time=current_tick,
        config=ObservationConfig(top_k=3),
        tasks=tasks,
        solution=solution,
    )

    # 2. Select action (greedy, no sampling)
    action = network.get_greedy_action(
        local_obs=torch.tensor(obs.vector, dtype=torch.float32),
        action_mask=torch.tensor(action_mask, dtype=torch.bool),
    )

    # 3. Convert to BLADE command
    if action != 0:  # Not NOOP
        blade_action = plan_edit_to_blade_action(
            action_token=action,
            observation_output=obs,
            scenario=blade_scenario,
            agent_id="f16_01",
        )
    else:
        blade_action = ""  # Continue plan

    # 4. Execute
    next_obs, reward, done, _, info = blade_env.step(blade_action)
```

### Example 2: Custom Reward Function

```python
from match_aou.rl.training import RewardConfig

# Conservative agent (prioritize fuel)
conservative_config = RewardConfig(
    imitation_reward=1.0,
    imitation_penalty=-0.5,
    fuel_efficiency_bonus=0.5,  # Higher fuel bonus
    target_coverage_bonus=0.1,  # Lower attack bonus
    use_shaping=True,
)

# Aggressive agent (prioritize attacks)
aggressive_config = RewardConfig(
    imitation_reward=1.0,
    imitation_penalty=-0.5,
    fuel_efficiency_bonus=0.05,  # Lower fuel bonus
    target_coverage_bonus=0.3,   # Higher attack bonus
    use_shaping=True,
)

# Simple (no shaping, only imitation signal)
simple_config = RewardConfig(
    imitation_reward=1.0,
    imitation_penalty=-0.5,
    use_shaping=False,
)
```

### Example 3: Evaluation Metrics

```python
from match_aou.rl.training import RewardTracker

tracker = RewardTracker()

# During evaluation
for episode in range(num_eval_episodes):
    total_reward = 0

    while not done:
        # ... run episode ...
        tracker.add_reward(reward, is_match=(rl_action == oracle_action))
        total_reward += reward

    print(f"Episode {episode}: {total_reward:.2f}")

# Get statistics
stats = tracker.get_stats()
print(f"Mean reward: {stats['mean']:.2f}")
print(f"Std: {stats['std']:.2f}")
print(f"Imitation accuracy: {stats['accuracy']:.1%}")
```

---

## Troubleshooting

### Common Issues

**1. Action mask is all False**
```python
# Check if agent exists
aircraft = scenario.get_aircraft(agent_id)
if aircraft is None:
    print(f"Agent {agent_id} not found!")

# Check observation
print(f"Fuel: {obs.self_state.fuel_norm}")
print(f"Has weapon: {obs.self_state.has_weapon}")
print(f"Targets: {[t.exists for t in obs.targets]}")
```

**2. Network outputs NaN**
```python
# Check for NaN in observation
if np.isnan(obs.vector).any():
    print("NaN in observation!")

# Check learning rate (PPO default 3e-4 is usually stable)
# Reduce if loss is unstable
config.learning_rate = 1e-4
```

**3. Policy not improving**
```python
# Check imitation accuracy
accuracy = matches / total_actions
print(f"Oracle accuracy: {accuracy:.1%}")  # Should increase

# Check entropy (should decrease slowly, not collapse to 0)
print(f"Entropy: {update_metrics['entropy']:.4f}")

# Check clip fraction (should be 0.1-0.3, not 0 or 1)
print(f"Clip fraction: {update_metrics['clip_fraction']:.3f}")
```

**4. top_k mismatch**
```python
# CRITICAL: observation.top_k MUST equal action.top_k
obs_config = ObservationConfig(top_k=3)
action_config = ActionSpaceConfig(top_k=3)  # MUST MATCH

# Check vector size
expected_size = 6 + (top_k * 6) + 6
assert len(obs.vector) == expected_size
```

---

## API Summary

### Main Imports

```python
# Observation
from match_aou.rl.observation import (
    build_observation_vector,
    ObservationConfig,
    ObservationOutput,
)

# Action
from match_aou.rl.action import (
    compute_action_mask,
    validate_action,
    ActionSpaceConfig,
    ActionType,
)

# Plan Editor
from match_aou.rl.plan_editor import (
    plan_edit_to_blade_action,
    preview_blade_action,
)

# Network (MAPPO)
from match_aou.rl.agent import ActorCriticNetwork

# Training (MAPPO)
from match_aou.rl.training import (
    PPOTrainer,
    PPOConfig,
    RolloutBuffer,
    compute_reward,
    RewardConfig,
    RewardTracker,
    MatchAOUOracle,
    SimpleOracle,
    EpisodeInitializer,
)
```

---

## Future Extensions

### Planned

1. **Extend top_k > 3**:
   - Add INSERT_ATTACK_3, INSERT_ATTACK_4 to ActionType
   - Update validation logic
   - Increase observation vector size

2. **Heterogeneous agents**:
   - Remove parameter sharing
   - Agent-specific actor networks
   - Agent ID embedding in observations

3. **Communication channels**:
   - Message passing between agents
   - Attention-based communication

### Possible

4. **Curriculum learning**:
   - Start with simple scenarios (1 agent, 1 target)
   - Gradually increase complexity

5. **Self-play evaluation**:
   - Compare trained policy against hand-crafted heuristics
   - Ablation studies (with/without reward shaping)

---

## References

- **MAPPO**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (2022)
- **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
- **CTDE**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (2017)
- **Imitation Learning**: Schaal, "Is imitation learning the route to humanoid robots?"

---

**Last Updated**: March 2026
**Version**: 2.0.0
**Maintainer**: Itamar, MSc Research — Ben-Gurion University of the Negev