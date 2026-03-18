# Quick Start Guide: Project Setup & Understanding
================================================================================

## Project Overview

This is a **Multi-Agent Task Allocation and Adaptation** project that combines:
- **MATCH-AOU**: An optimization solver for task allocation
- **BLADE**: A simulation engine for multi-agent scenarios  
- **RL Integration** (in development): Reinforcement Learning for dynamic plan adaptation

---

## Project Structure at a Glance

```
Multi_Agent_Task_Allocation_and_Adaptation/
│
├── src/match_aou/                          # Main project code
│   ├── models/                             # Data structures
│   │   ├── agent.py                        # Agent definition
│   │   ├── task.py                         # Task definition
│   │   ├── step.py                         # Plan step definition
│   │   ├── step_type.py                    # Step type enum
│   │   ├── capability.py                   # Capability definition
│   │   └── location.py                     # Location/coordinates
│   │
│   ├── solvers/                            # Optimization solvers
│   │   └── match_aou_MINLP_solver.py      # MINLP solver
│   │
│   ├── utils/                              # Utilities
│   │   ├── blade_utils/                    # BLADE integration utilities
│   │   │   ├── blade_executor_minimal.py  # Execute plans in BLADE
│   │   │   ├── blade_plan_utils.py        # Plan conversion utilities
│   │   │   ├── observation_utils.py       # Extract observations
│   │   │   └── scenario_factory.py        # Create scenarios
│   │   ├── scheduling_utils.py            # Scheduling helpers
│   │   ├── topology_utils.py              # Topology calculations
│   │   └── match_aou_parser.py            # Parse MATCH-AOU output
│   │
│   └── integrations/                       # Third-party integrations
│       └── panopticon-main/               # BLADE engine (Panopticon)
│           └── gym/blade/                  # BLADE as Gym environment
│               ├── units/                  # Aircraft, Ships, etc.
│               ├── envs/blade.py          # Gym environment wrapper
│               ├── Game.py                # Main game loop
│               ├── Scenario.py            # Scenario management
│               ├── mission/               # Mission types
│               ├── engine/                # Weapon engagement
│               └── db/                    # Unit database
│
├── scripts/                                # Demo and test scripts
│
├── data/                                   # Data files
│   ├── scenarios/                         # Scenario definitions (JSON)
│   └── agents_from_csv/                   # Agent data
│
└── BLADE_API_REFERENCE.md                 # API documentation
```

---

## Key Components Explained

### 1. MATCH-AOU (Solver)

**Location:** `src/match_aou/solvers/match_aou_MINLP_solver.py`

**What it does:**
- Takes a scenario with agents and tasks
- Solves optimization problem to assign tasks to agents
- Outputs a **plan** for each agent (sequence of steps)

**Key Classes:**
- `Agent`: Represents an agent with capabilities
- `Task`: Represents a task requiring certain capabilities  
- `Step`: A single action in an agent's plan
- `StepType`: Enum of step types (ATTACK, MOVE, RTB, etc.)

### 2. BLADE (Simulation Engine)

**Location:** `src/match_aou/integrations/panopticon-main/gym/blade/`

**What it does:**
- Simulates military scenarios with aircraft, ships, facilities
- Executes agent plans in a physics-based environment
- Provides observations (what agents can see)

**Key Files:**
- `Game.py`: Main simulation loop
- `Scenario.py`: Loads and manages scenarios
- `units/Aircraft.py`: Aircraft behavior
- `envs/blade.py`: Gym environment wrapper

### 3. Integration Utilities

**Location:** `src/match_aou/utils/blade_utils/`

**What they do:**
- Bridge between MATCH-AOU and BLADE
- Convert MATCH-AOU plans to BLADE actions
- Extract observations from BLADE for RL

**Key Files:**
- `blade_executor_minimal.py`: Execute MATCH-AOU plans in BLADE
- `blade_plan_utils.py`: Convert between plan formats
- `observation_utils.py`: Extract agent observations
- `scenario_factory.py`: Create BLADE scenarios

---

## How the Components Connect

```
┌─────────────────┐
│   MATCH-AOU     │  (Generates initial plan)
│   (Solver)      │
└────────┬────────┘
         │
         │ Plan (sequence of Steps)
         ▼
┌─────────────────┐
│  blade_utils    │  (Converts plan to BLADE format)
│  (Integration)  │
└────────┬────────┘
         │
         │ BLADE Actions
         ▼
┌─────────────────┐
│     BLADE       │  (Executes simulation)
│  (Simulation)   │
└────────┬────────┘
         │
         │ Observations
         ▼
┌─────────────────┐
│       RL        │  (Adapts plan - IN DEVELOPMENT)
│   (Learning)    │
└─────────────────┘
```

---

## Getting Started

### Prerequisites

1. **Python 3.8+**
2. **Required packages** (see `requirements.txt`):
   - numpy, scipy (scientific computing)
   - pyomo (optimization)
   - gymnasium (RL environment)
   - stable-baselines3 (RL algorithms)
   - shapely, haversine (geospatial)

### Installation

```bash
# Clone/navigate to project
cd Multi_Agent_Task_Allocation_and_Adaptation

# Install dependencies
pip install -r requirements.txt

# Install BLADE as package (if available)
cd src/match_aou/integrations/panopticon-main/gym
pip install -e .
cd ../../../../..
```

### Running a Demo

```bash
# From project root
python scripts/match_aou_demo_main.py
```

---

## Next Steps for RL Integration

### What's Needed:

1. **Gym Environment Wrapper**
   - Wrap BLADE with standard Gym interface
   - Define observation space (what agent sees)
   - Define action space (what agent can do)

2. **Plan-Edit API**
   - Allow RL to modify agent plans
   - Actions: INSERT_ATTACK, FORCE_RTB, NOOP, etc.

3. **Reward Function**
   - Define what makes a good plan adaptation
   - Based on task completion, efficiency, etc.

4. **Training Loop**
   - Use algorithms like PPO, DQN, A3C
   - Train agents to adapt plans dynamically

---

## Important Files to Review

### For understanding BLADE:
1. `src/match_aou/integrations/panopticon-main/gym/blade/Game.py`
2. `src/match_aou/integrations/panopticon-main/gym/blade/Scenario.py`
3. `src/match_aou/integrations/panopticon-main/gym/blade/units/Aircraft.py`
4. `BLADE_API_REFERENCE.md` (in project root)

### For understanding MATCH-AOU:
1. `src/match_aou/solvers/match_aou_MINLP_solver.py`
2. `src/match_aou/models/agent.py`
3. `src/match_aou/models/task.py`
4. `src/match_aou/models/step.py`

### For understanding integration:
1. `src/match_aou/utils/blade_utils/blade_executor_minimal.py`
2. `src/match_aou/utils/blade_utils/blade_plan_utils.py`
3. `src/match_aou/utils/blade_utils/observation_utils.py`

---

## Questions to Answer Before Starting RL

1. **What is the exact observation space?**
   - What information does each agent see?
   - How is it represented (vector, dict, etc.)?

2. **What is the exact action space?**
   - What actions can RL take?
   - How are they parameterized?

3. **What is the reward function?**
   - How do we measure success?
   - What do we want to optimize?

4. **How does Plan-Edit work?**
   - How does RL modify existing plans?
   - What constraints exist?

---

## Project Statistics

- **Total Python Files:** 49
- **Total Packages:** 8
- **Main Components:**
  - MATCH-AOU: 7 model files + 1 solver
  - BLADE: Full simulation engine
  - Integration: 4 utility files
  - RL: In development

---

## Contact & References

- **BLADE Documentation**: Check `BLADE_API_REFERENCE.md` in project root
- **BLADE Source**: `src/match_aou/integrations/panopticon-main/gym/README.md`
- **MATCH-AOU**: Review solver code and model definitions

---

**Last Updated:** 2026-02-02  
**Project Status:** MATCH-AOU and BLADE integrated, RL layer in development
