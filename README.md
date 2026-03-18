# Multi-Agent Task Allocation and Adaptation

MSc research project combining optimization-based task allocation with reinforcement learning for dynamic plan adaptation in multi-agent military simulations.

## Overview

The system has three integrated components:

1. **MATCH-AOU** — A Mixed-Integer Nonlinear Programming (MINLP) solver that generates optimal task allocations for heterogeneous agents operating without communication.

2. **BLADE** — A physics-based military simulation environment (Panopticon fork), providing realistic aircraft dynamics, weapon engagement, and scenario execution.

3. **RL Layer (MAPPO)** — A Multi-Agent PPO system with Centralized Training, Decentralized Execution (CTDE) that enables agents to adapt their plans mid-mission when new targets are discovered.

### Research Problem

Agents start a mission with a MATCH-AOU plan based on **partial** target information (~67% of targets). During execution, they discover new targets and must decide in real-time whether to deviate from the original plan. The RL layer learns this adaptation by imitating an oracle that has access to the **full** target set.

## Project Structure

```
Multi_Agent_Task_Allocation_and_Adaptation/
├── train_full.py                    # Main training script (MAPPO + BLADE + MATCH-AOU)
├── requirements.txt
├── data/
│   └── scenarios/                   # BLADE scenario JSON files
│       └── strike_training_2v3.json # Primary training scenario (2 F-16s, 3 targets)
├── src/match_aou/
│   ├── models/                      # Domain objects (Agent, Task, Step, Location, ...)
│   ├── solvers/                     # MATCH-AOU MINLP solver
│   ├── utils/
│   │   ├── blade_utils/             # BLADE integration (executor, plan utils, scenario factory)
│   │   ├── scheduling_utils.py      # Post-solve scheduling
│   │   ├── topology_utils.py        # Topological ordering
│   │   └── match_aou_parser.py      # Output parsing
│   ├── rl/
│   │   ├── observation/             # Observation builder (30-dim vector)
│   │   ├── action/                  # Action space (5 discrete: NOOP, ATTACK×3, RTB)
│   │   ├── agent/                   # ActorCriticNetwork (shared actor, centralized critic)
│   │   ├── training/                # PPOTrainer, RolloutBuffer, Reward, Oracle
│   │   ├── plan_editor.py           # Converts RL actions → BLADE commands
│   │   └── shared_utils.py          # Common utilities (haversine, normalization)
│   └── integrations/
│       └── panopticon-main/         # BLADE engine (frozen Panopticon copy)
│           └── gym/blade/           # Python API (Game, Scenario, units, Gym env)
├── tests/                           # Unit tests (observation, action space)
├── legacy/                          # Archived code (DQN-era)
└── docs/                            # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.10+
- A Pyomo-compatible solver (bonmin recommended)

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train_full.py --scenario data/scenarios/strike_training_2v3.json --episodes 50
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--scenario` | `data/scenarios/strike_training_2v3.json` | Scenario file |
| `--episodes` | 50 | Training episodes |
| `--decision-interval` | 100 | Ticks between RL decisions |
| `--lr` | 3e-4 | Learning rate |
| `--save-freq` | 10 | Checkpoint save frequency |

Training outputs are saved to `training_output/` (models, logs, BLADE recordings).

## Architecture

### Observation Space (30 features)

| Component | Features | Description |
|---|---|---|
| Self-state | 6 | fuel, has_weapon, dist_to_next, next_is_attack, heading, rtb_possible |
| Targets (×3) | 18 | exists, dist_norm, is_threat, is_dynamic, is_in_plan, already_engaged |
| Plan context | 6 | Remaining assignments, progress, distances |

### Action Space (5 discrete)

| Token | Action | Description |
|---|---|---|
| 0 | NOOP | Continue current plan |
| 1-3 | INSERT_ATTACK(k) | Insert attack on target slot k |
| 4 | FORCE_RTB | Return to base |

### MAPPO (CTDE)

- **Actor** (shared weights, decentralized): `local_obs[30] → 128 → 64 → logits[5]`
- **Critic** (centralized): `global_state[60] → 128 → 64 → V(s)[1]`
- **Training**: PPO with GAE, clipped surrogate objective, imitation reward from oracle

## Documentation

- [BLADE API Reference](docs/BLADE_API_DOCUMENTATION.md)
- [MATCH-AOU API Reference](docs/MATCH_AOU_API.md)
- [Integration Guide](docs/INTEGRATION_GUIDE.md)
- [RL Module Documentation](src/match_aou/rl/RL_MODULE_DOCUMENTATION.md)

## License

This project is part of MSc research at Ben-Gurion University of the Negev, Department of Software and Information Systems Engineering.
