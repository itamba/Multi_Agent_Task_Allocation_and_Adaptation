# RL Module Documentation

**Reinforcement Learning system for MATCH-AOU tactical decision-making**

Version: 1.0.0  
Last Updated: February 2026

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
- **RL Agent**: Neural network learning to adapt plans in real-time

### Key Features

- **Partial observability**: Agent discovers targets gradually (unlike omniscient MATCH-AOU)
- **Online adaptation**: Real-time plan modifications during mission execution
- **Action masking**: Ensures only valid/safe actions are selected
- **Imitation learning**: Learn from MATCH-AOU demonstrations
- **Modular design**: Clean separation between observation, action, and training

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         RL SYSTEM                                │
│                                                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
│  │            │    │            │    │            │           │
│  │ Observation│───▶│   Policy   │───▶│Plan Editor │           │
│  │  Builder   │    │  Network   │    │            │           │
│  │            │    │            │    │            │           │
│  └────────────┘    └────────────┘    └────────────┘           │
│        │                  │                  │                  │
│        │                  │                  │                  │
│        ▼                  ▼                  ▼                  │
│   [30 features]      [5 actions]    [BLADE command]           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
         │                                         │
         │                                         │
         ▼                                         ▼
  ┌─────────────┐                          ┌─────────────┐
  │    BLADE    │                          │  MATCH-AOU  │
  │  Scenario   │                          │   Oracle    │
  └─────────────┘                          └─────────────┘
```

### Pipeline Flow

1. **Observation**: Extract 30-feature vector from BLADE state
2. **Action Selection**: Neural network chooses discrete action (0-4)
3. **Plan Editing**: Convert action to BLADE execution command
4. **Execution**: BLADE simulates one step
5. **Reward**: Compare RL decision to oracle (imitation learning)

---

## Module Structure

```
src/match_aou/rl/
├── __init__.py                  # Main RL API
├── shared_utils.py              # Common utilities (haversine, normalize, etc.)
│
├── observation/                 # Phase 1: State Extraction
│   ├── __init__.py             # Public API: build_observation_vector()
│   ├── config.py               # ObservationConfig
│   ├── types.py                # SelfState, TargetInfo, ObservationOutput
│   ├── observation_builder.py # Main builder
│   ├── self_features.py        # Agent's own state (6 features)
│   ├── target_extraction.py   # Enemy detection (K×6 features)
│   ├── plan_parsing.py         # Plan analysis
│   ├── plan_context.py         # Plan context (6 features)
│   └── utils.py                # Observation-specific helpers
│
├── action/                      # Phase 2: Action Space
│   ├── __init__.py             # Public API: compute_action_mask()
│   ├── action_config.py        # ActionType, ActionSpaceConfig
│   ├── action_validation.py   # ActionValidator, ActionMask
│   └── action_utils.py         # Action-specific helpers
│
├── plan_editor.py              # Phase 3: Action → BLADE Command
│
├── network.py                  # Phase 4: Neural Network
│
└── training/                   # Phase 5: Training System
    ├── __init__.py             # Training API
    ├── buffer.py               # ReplayBuffer
    ├── reward.py               # Reward functions
    ├── oracle.py               # MATCH-AOU oracle wrappers
    ├── episode_initializer.py # Episode setup
    ├── trainer.py              # DQNTrainer
    └── training_utils.py       # Training helpers
```

---

## Quick Start

### Installation

```bash
# Install project
pip install -e .

# Required dependencies
torch>=2.0.0
numpy>=1.24.0
```

### Basic Usage

```python
from match_aou.rl import (
    build_observation_vector,
    compute_action_mask,
    plan_edit_to_blade_action,
    EnhancedMLPQNetwork,
)

# 1. Extract observation
obs = build_observation_vector(
    scenario=blade_scenario,
    agent_id="f16_01",
    current_plan=plan,
    current_time=100
)

# 2. Get valid actions
mask = compute_action_mask(obs, blade_scenario, "f16_01")
valid_actions = mask.get_valid_actions()

# 3. Select action (using trained network)
network = EnhancedMLPQNetwork.load('trained_model.pt')
action = network.get_action(
    obs=torch.tensor(obs.vector),
    action_mask=torch.tensor(mask.mask),
    epsilon=0.0  # No exploration (greedy)
)

# 4. Convert to BLADE command
blade_action = plan_edit_to_blade_action(
    action_token=action,
    observation_output=obs,
    scenario=blade_scenario,
    agent_id="f16_01"
)

# 5. Execute in BLADE
next_obs, reward, done, _, _ = blade_env.step(blade_action)
```

### Training

```python
from match_aou.rl.training import (
    DQNTrainer,
    TrainingConfig,
    MatchAOUOracle,
    ReplayBuffer,
)

# Setup
config = TrainingConfig(
    obs_dim=30,
    action_dim=5,
    learning_rate=0.001,
    buffer_size=10000,
)

network = EnhancedMLPQNetwork()
buffer = ReplayBuffer(capacity=10000)
oracle = MatchAOUOracle(solver_name='bonmin')
trainer = DQNTrainer(network, buffer, oracle, config)

# Train
for episode in range(100):
    metrics = trainer.train_episode(blade_env, scenario)
    print(f"Episode {episode}: reward={metrics['total_reward']:.2f}")
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
    config=config
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

[6-23]   Target features (3 × 6 = 18):
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
- Vector size = 6 + (K × 6) + 6 = 12 + 6K
- Default: top_k=3 → 30 features

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
  - Fuel ≥ min_attack_fuel_margin
  - Time since last attack ≥ attack_cooldown_ticks
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
NOOP           → ""                              (empty = continue plan)
INSERT_ATTACK_k → "handle_aircraft_attack(...)" (insert attack in plan)
FORCE_RTB      → "aircraft_return_to_base(...)" (immediate RTB)
```

**Weapon Selection**:
- Automatically selects best available weapon
- Uses `aircraft.get_weapon_with_highest_engagement_range()`
- Falls back to first available weapon

---

### 4. Neural Network

**Purpose**: Q-Network for action-value estimation

**API**:
```python
from match_aou.rl import EnhancedMLPQNetwork

# Create network
network = EnhancedMLPQNetwork(
    obs_dim=30,
    action_dim=5,
    hidden_sizes=[128, 64]  # Default
)

# Forward pass
obs = torch.randn(1, 30)
mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.bool)
q_values = network(obs, mask)  # Shape: (1, 5)
# Invalid actions have Q = -inf

# Action selection
action = network.get_action(
    obs=obs.squeeze(0),
    action_mask=mask.squeeze(0),
    epsilon=0.1  # 10% exploration
)

# Save/Load
network.save('model.pt')
loaded_network = EnhancedMLPQNetwork.load('model.pt')
```

**Architecture**:
```
Input [30] → FC(128) → ReLU → FC(64) → ReLU → FC(5) → Q-values

Parameters: ~10k
Initialization: Xavier uniform
```

**Action Masking**:
- Invalid actions receive Q-value = `-inf`
- Ensures policy only selects valid actions
- Applies in both training and inference

---

### 5. Training System

#### Replay Buffer

```python
from match_aou.rl.training import ReplayBuffer

buffer = ReplayBuffer(capacity=10000)

# Add experience
buffer.add(
    state=obs.vector,            # [30]
    action=1,                    # int
    reward=1.0,                  # float
    next_state=next_obs.vector,  # [30]
    done=False,                  # bool
    action_mask=mask.mask,       # [5]
    next_action_mask=next_mask.mask
)

# Sample batch
batch = buffer.sample(batch_size=32)
if batch is not None:
    states, actions, rewards, next_states, dones, masks, next_masks = batch
```

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
# Rules: Low fuel → RTB, Unassigned nearby → Attack, Else → NOOP
```

#### DQN Trainer

```python
from match_aou.rl.training import DQNTrainer, TrainingConfig

config = TrainingConfig(
    # Network
    obs_dim=30,
    action_dim=5,
    hidden_sizes=[128, 64],
    
    # Optimization
    learning_rate=0.001,
    gamma=0.99,
    batch_size=32,
    
    # Experience replay
    buffer_size=10000,
    min_buffer_size=1000,
    
    # Target network
    target_update_freq=100,
    
    # Exploration
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    
    # Training
    train_freq=1,
    max_grad_norm=10.0,
)

trainer = DQNTrainer(
    q_network=network,
    buffer=buffer,
    oracle=oracle,
    config=config
)

# Train one episode
metrics = trainer.train_episode(blade_env, scenario)
# Returns: {
#   'total_reward': float,
#   'steps': int,
#   'epsilon': float,
#   'loss': float,
#   'accuracy': float,  # % matching oracle
# }
```

---

## Training Pipeline

### Full Training Loop

```python
import torch
from match_aou.rl import EnhancedMLPQNetwork
from match_aou.rl.training import (
    DQNTrainer,
    TrainingConfig,
    ReplayBuffer,
    MatchAOUOracle,
    EpisodeInitializer,
)

# 1. Setup
network = EnhancedMLPQNetwork(obs_dim=30, action_dim=5)
buffer = ReplayBuffer(capacity=10000)
oracle = MatchAOUOracle(solver_name='bonmin')
config = TrainingConfig()

trainer = DQNTrainer(network, buffer, oracle, config)
initializer = EpisodeInitializer(blade_env, oracle)

# 2. Training loop
for episode in range(num_episodes):
    # Initialize episode
    obs_dict, partial_sol, full_sol = initializer.initialize_episode(
        scenario=blade_scenario,
        agents=agent_list,
        all_tasks=task_list,
        partial_ratio=0.67  # 67% of tasks known initially
    )
    
    # Run episode
    metrics = trainer.train_episode(
        env=blade_env,
        scenario=blade_scenario,
        initial_observations=obs_dict,
        partial_solution=partial_sol,
        full_solution=full_sol  # Oracle's complete plan
    )
    
    # Log
    print(f"Episode {episode}:")
    print(f"  Reward: {metrics['total_reward']:.2f}")
    print(f"  Steps: {metrics['steps']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Loss: {metrics['loss']:.4f}")
    
    # Save checkpoints
    if episode % 100 == 0:
        network.save(f'checkpoints/model_ep{episode}.pt')

# 3. Final model
network.save('final_model.pt')
```

### Episode Initialization

The `EpisodeInitializer` handles:
1. **Task partitioning**: Split tasks into partial (known) and full (oracle)
2. **MATCH-AOU solving**: Solve both partial and full problems
3. **Auto-launch**: Launch agents with initial assignments
4. **Observation extraction**: Get initial state for all agents

```python
initializer = EpisodeInitializer(blade_env, oracle)

obs_dict, partial_sol, full_sol = initializer.initialize_episode(
    scenario=blade_scenario,
    agents=agent_list,
    all_tasks=task_list,
    partial_ratio=0.67  # Agent knows 67% of targets initially
)

# obs_dict: {agent_id: ObservationOutput}
# partial_sol: {agent_id: [(task, step, level), ...]}  # Agent's plan
# full_sol: {agent_id: [(task, step, level), ...]}     # Oracle's plan
```

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
│  Scenario   │  │  RL decides: NOOP / ATTACK / RTB
└──────┬──────┘  │  BLADE executes one step
       │         │
       ▼         │
┌─────────────┐  │
│ Observation │  │  Extract 30 features
│   Builder   │  │
└──────┬──────┘  │
       │         │
       ▼         │
┌─────────────┐  │
│ RL Network  │──┘  Select action (Q-learning + mask)
└─────────────┘
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

### MATCH-AOU Integration

**Task Format**:
```python
from match_aou import Task, Agent

# Define agents
agents = [
    Agent(
        id='f16_01',
        location=(35.0, 40.0),
        fuel=1000.0,
        weapons=['aim_120', 'aim_9']
    )
]

# Define tasks
tasks = [
    Task(
        id='attack_sam_01',
        target_id='sam_01',
        location=(35.5, 40.5),
        weapon_required='aim_120',
        priority=1.0
    )
]

# Solve
solution = oracle.solve_full_problem(agents, tasks, precedence_relations=[])
# Returns: {agent_id: [(task, step, level), ...]}
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

### Training Configuration

```python
from match_aou.rl.training import TrainingConfig, RewardConfig

train_config = TrainingConfig(
    # Network architecture
    obs_dim=30,
    action_dim=5,
    hidden_sizes=[128, 64],
    
    # Optimizer
    learning_rate=0.001,
    gamma=0.99,              # Discount factor
    batch_size=32,
    
    # Replay buffer
    buffer_size=10000,
    min_buffer_size=1000,    # Start training after 1k samples
    
    # Target network
    target_update_freq=100,  # Hard update every 100 steps
    
    # Exploration
    epsilon_start=1.0,       # 100% random initially
    epsilon_end=0.01,        # 1% random finally
    epsilon_decay=0.995,     # Decay per episode
    
    # Training
    train_freq=1,            # Train every step
    max_grad_norm=10.0,      # Gradient clipping
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
from match_aou.rl import (
    build_observation_vector,
    compute_action_mask,
    plan_edit_to_blade_action,
    EnhancedMLPQNetwork,
)

# Load trained model
network = EnhancedMLPQNetwork.load('trained_model.pt')
network.eval()

# Game loop
while not done:
    # 1. Observe
    obs = build_observation_vector(
        scenario=blade_scenario,
        agent_id="f16_01",
        current_plan=current_plan,
        current_time=blade_env.current_tick
    )
    
    # 2. Get valid actions
    mask = compute_action_mask(
        observation_output=obs,
        scenario=blade_scenario,
        agent_id="f16_01",
        last_attack_tick=last_attack_tick
    )
    
    # 3. Select action (greedy, no exploration)
    with torch.no_grad():
        action = network.get_action(
            obs=torch.tensor(obs.vector, dtype=torch.float32),
            action_mask=torch.tensor(mask.mask, dtype=torch.bool),
            epsilon=0.0  # Greedy
        )
    
    # 4. Convert to BLADE command
    blade_action = plan_edit_to_blade_action(
        action_token=action,
        observation_output=obs,
        scenario=blade_scenario,
        agent_id="f16_01"
    )
    
    # 5. Execute
    next_obs, reward, done, _, info = blade_env.step(blade_action)
    
    # Update state
    if action in [1, 2, 3]:  # Attack actions
        last_attack_tick = blade_env.current_tick
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
    use_shaping=True
)

# Aggressive agent (prioritize attacks)
aggressive_config = RewardConfig(
    imitation_reward=1.0,
    imitation_penalty=-0.5,
    fuel_efficiency_bonus=0.05,  # Lower fuel bonus
    target_coverage_bonus=0.3,   # Higher attack bonus
    use_shaping=True
)

# Simple (no shaping)
simple_config = RewardConfig(
    imitation_reward=1.0,
    imitation_penalty=-0.5,
    use_shaping=False  # Only imitation signal
)
```

### Example 3: Multi-Agent Training

```python
# Train multiple agents simultaneously
agents = ["f16_01", "f16_02", "f16_03"]

# Separate networks (decentralized)
networks = {
    agent_id: EnhancedMLPQNetwork()
    for agent_id in agents
}

# Shared buffer (optional)
buffer = ReplayBuffer(capacity=30000)

# Training loop
for episode in range(num_episodes):
    # Initialize
    obs_dict, partial_sol, full_sol = initializer.initialize_episode(
        scenario=scenario,
        agents=agent_list,
        all_tasks=task_list
    )
    
    done = False
    while not done:
        actions = {}
        
        # Each agent selects action
        for agent_id in agents:
            obs = obs_dict[agent_id]
            mask = compute_action_mask(obs, scenario, agent_id)
            
            action = networks[agent_id].get_action(
                obs=torch.tensor(obs.vector),
                action_mask=torch.tensor(mask.mask),
                epsilon=epsilon
            )
            actions[agent_id] = action
        
        # Execute all actions
        # ... (step through BLADE)
        
        # Store experiences for all agents
        for agent_id in agents:
            buffer.add(...)
        
        # Train all networks
        if buffer.is_ready(min_size=1000):
            for agent_id in agents:
                batch = buffer.sample(32)
                loss = networks[agent_id].update(batch)
```

### Example 4: Evaluation Metrics

```python
from match_aou.rl.training import RewardTracker

tracker = RewardTracker()

# During evaluation
for episode in range(num_eval_episodes):
    total_reward = 0
    
    while not done:
        # ... (run episode)
        
        oracle_action = oracle.get_action(...)
        rl_action = network.get_action(...)
        
        reward = compute_reward(rl_action, oracle_action, obs)
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
    
# Check learning rate
if loss > 1000:
    print("Loss exploding - reduce learning rate")
    
# Enable gradient clipping
config.max_grad_norm = 1.0  # More aggressive clipping
```

**3. Agent not learning**
```python
# Check exploration
print(f"Epsilon: {epsilon:.3f}")  # Should decay over time

# Check if matching oracle
accuracy = matches / total_actions
print(f"Oracle accuracy: {accuracy:.1%}")  # Should increase

# Visualize Q-values
q_values = network(obs, mask)
print(f"Q-values: {q_values}")
print(f"Selected: {q_values.argmax()}")
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

## Performance Tips

### 1. Hyperparameter Tuning

```python
# Start with these proven defaults
config = TrainingConfig(
    learning_rate=0.001,     # Lower if unstable
    gamma=0.99,              # Higher for long-term planning
    batch_size=32,           # Increase if GPU available
    epsilon_decay=0.995,     # Slower decay = more exploration
    target_update_freq=100,  # More frequent = more stable
)
```

### 2. Training Stability

```python
# Use gradient clipping
config.max_grad_norm = 10.0

# Use target network
config.use_soft_updates = False  # Hard updates more stable
config.target_update_freq = 100

# Start with simple rewards
reward_config.use_shaping = False  # Just imitation
```

### 3. Faster Training

```python
# Larger batch size (if GPU memory allows)
config.batch_size = 64

# Larger buffer
config.buffer_size = 50000

# Train less frequently
config.train_freq = 4  # Every 4 steps instead of every step
```

### 4. Better Exploration

```python
# Slower epsilon decay
config.epsilon_start = 1.0
config.epsilon_end = 0.05    # Don't go too low
config.epsilon_decay = 0.998  # Very slow decay

# Or use epsilon scheduling
def get_epsilon(episode, total_episodes):
    # Linear decay
    return max(0.01, 1.0 - (episode / total_episodes))
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
from match_aou.rl import (
    plan_edit_to_blade_action,
    preview_blade_action,
)

# Network
from match_aou.rl import EnhancedMLPQNetwork

# Training
from match_aou.rl.training import (
    DQNTrainer,
    TrainingConfig,
    ReplayBuffer,
    compute_reward,
    RewardConfig,
    MatchAOUOracle,
    SimpleOracle,
    EpisodeInitializer,
)
```

---

## Future Extensions

### Planned Features

1. **Extend top_k > 3**:
   - Add INSERT_ATTACK_3, INSERT_ATTACK_4 to ActionType
   - Update validation logic
   - Increase observation vector size

2. **Multi-agent coordination**:
   - Shared replay buffer
   - Centralized critic
   - Communication channels

3. **Advanced algorithms**:
   - Double DQN (already prepared in network.py)
   - Dueling DQN
   - Rainbow DQN

4. **Better exploration**:
   - Noisy Networks
   - Curiosity-driven exploration
   - Parameter space noise

5. **Prioritized replay**:
   - Re-implement PrioritizedReplayBuffer
   - Sample important experiences more often

---

## Contributing

When adding new components:

1. **Follow existing patterns**: Use dataclasses for config, type hints everywhere
2. **Update documentation**: Add to this file and docstrings
3. **No test code in production**: Tests go in `tests/`, not in module files
4. **Use shared_utils**: Don't duplicate haversine, normalize, etc.
5. **Validate top_k compatibility**: Observation and action must match

---

## References

- **MATCH-AOU Paper**: [Link to paper]
- **BLADE Documentation**: See `/mnt/project/BLADE_API_DOCUMENTATION.md`
- **DQN Paper**: Mnih et al., "Human-level control through deep reinforcement learning"
- **Imitation Learning**: Schaal, "Is imitation learning the route to humanoid robots?"

---

**Last Updated**: February 2026  
**Version**: 1.0.0  
**Maintainer**: Multi-Agent Task Allocation Team
