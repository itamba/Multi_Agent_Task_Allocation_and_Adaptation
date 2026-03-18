# MATCH-AOU ↔ BLADE Integration Guide
================================================================================

This guide explains how to connect MATCH-AOU (task allocation) with BLADE
(simulation) using the integration utilities.

**What's Covered:** Integration workflow, `blade_utils`, conversion functions
**Not Covered:** MATCH-AOU API (see `MATCH_AOU_API.md`), BLADE API (see `BLADE_API_DOCUMENTATION.md`)

**See Also:**
- `MATCH_AOU_API.md` for solver usage
- `BLADE_API_DOCUMENTATION.md` for simulation details

---

## Table of Contents

1. [Overview](#overview)
2. [Integration Architecture](#integration-architecture)
3. [blade_utils Module](#blade_utils-module)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Action Template System](#action-template-system)
6. [Complete Example](#complete-example)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### The Integration Flow

```
┌─────────────────┐
│   MATCH-AOU     │  1. Solve optimization problem
│   (Solver)      │     → Get agent assignments
└────────┬────────┘
         │
         │ solution: {agent_id: [(task, step), ...]}
         ▼
┌─────────────────┐
│  blade_utils    │  2. Convert to BLADE format
│  (Integration)  │     → Add timing, resolve actions
└────────┬────────┘
         │
         │ timestep → [actions]
         ▼
┌─────────────────┐
│     BLADE       │  3. Execute in simulation
│  (Simulation)   │     → Get observations
└────────┬────────┘
         │
         │ observations
         ▼
┌─────────────────┐
│       RL        │  4. Adapt plans (future)
│   (Learning)    │
└─────────────────┘
```

---

## Integration Architecture

### Key Concepts

1. **MATCH-AOU produces abstract plans** - which agents do which steps
2. **blade_utils converts to concrete actions** - actual BLADE commands with timing
3. **BLADE executes actions** - simulates physics and returns observations

### Why We Need Integration Utilities

MATCH-AOU and BLADE speak different languages:

| MATCH-AOU Output | BLADE Input |
|------------------|-------------|
| Abstract steps | Concrete actions |
| No timing | Explicit timesteps |
| Placeholders (AGENT_ID) | Actual IDs |
| Logical assignments | Physical commands |

**blade_utils bridges this gap.**

---

## blade_utils Module

**Location:** `src/match_aou/utils/blade_utils/`

### Module Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `blade_plan_utils.py` | Convert plans to BLADE actions | Always |
| `blade_executor_minimal.py` | Execute plans in BLADE | When running full simulation |
| `scenario_factory.py` | Create BLADE scenarios | When setting up |
| `observation_utils.py` | Extract RL observations | For RL training |

### blade_plan_utils.py

**Main Functions:**

#### `assign_execution_times_and_resolve_actions()`
Converts MATCH-AOU solution to timed BLADE actions.

```python
from match_aou.utils.blade_utils.blade_plan_utils import (
    assign_execution_times_and_resolve_actions
)

blade_artifacts = assign_execution_times_and_resolve_actions(
    agents=agents,              # List[Agent]
    tasks=tasks,                # List[Task]
    solution=solution,          # Dict from solver
    precedence_levels=levels    # Optional: pre-computed levels
)

# Returns: BladePlanArtifacts
# - execution_time_to_actions: Dict[int, List[str]]
# - levels: List[int]
# - level_start_time: Dict[int, int]
```

**What it does:**
1. Assigns timestep to each step (considering travel time)
2. Resolves action templates (AGENT_ID → 'f16_01')
3. Ensures one action per timestep (BLADE constraint)
4. Handles multi-agent coordination

### scenario_factory.py

Creates BLADE scenarios from MATCH-AOU data.

```python
from match_aou.utils.blade_utils.scenario_factory import create_scenario

blade_scenario = create_scenario(
    agents=agents,
    tasks=tasks,
    scenario_name='My Scenario'
)
```

### blade_executor_minimal.py

Executes MATCH-AOU plans in BLADE simulation.

```python
from match_aou.utils.blade_utils.blade_executor_minimal import execute_plan

final_scenario = execute_plan(
    initial_scenario=blade_scenario,
    execution_time_to_actions=blade_artifacts.execution_time_to_actions,
    max_steps=1000
)
```

### observation_utils.py

Extracts observations for RL agent.

```python
from match_aou.utils.blade_utils.observation_utils import extract_observations

obs = extract_observations(
    scenario=current_scenario,
    agent_id='f16_01'
)
```

---

## Step-by-Step Workflow

### Complete Integration Pipeline

```python
# ===== STEP 1: Define Problem (MATCH-AOU) =====
from match_aou.models import Agent, Task, Step, StepType, Capability, Location
from match_aou.solvers.match_aou_MINLP_solver import MatchAou

# Define agents with BLADE-compatible properties
agent1 = Agent(
    location=Location(lat=35.0, lon=40.0),
    capabilities=[Capability('attack')],
    budget=1000.0,
    move_cost_function=distance_func,
    agent_id='f16_01',          # ← Will be used in BLADE
    weapon_id='aim_120',        # ← For action templates
    home_base_id='base_01'
)

# Define tasks with BLADE action templates
attack_step = Step(
    location=Location(lat=35.5, lon=40.5),
    capabilities=[Capability('attack')],
    step_type=StepType('attack', 100),
    effort=1,
    probability=0.9,
    action='handle_aircraft_attack(AGENT_ID, "target_01", WEAPON_ID, 2)'
    #       ^^^^^^^^^^^^^^^^^^^^^^ Template with placeholders
)

task1 = Task(steps=[attack_step], utility=500)

# Solve
solver = MatchAou(agents=[agent1], tasks=[task1])
solution, results, unselected = solver.solve()

# ===== STEP 2: Convert to BLADE (Integration) =====
from match_aou.utils.blade_utils.blade_plan_utils import (
    assign_execution_times_and_resolve_actions
)

blade_artifacts = assign_execution_times_and_resolve_actions(
    agents=[agent1],
    tasks=[task1],
    solution=solution
)

# Now we have:
# blade_artifacts.execution_time_to_actions = {
#     0: ['move_aircraft("f16_01", [[35.5, 40.5]])'],
#     50: ['handle_aircraft_attack("f16_01", "target_01", "aim_120", 2)']
# }

# ===== STEP 3: Create BLADE Scenario =====
from match_aou.utils.blade_utils.scenario_factory import create_scenario

blade_scenario = create_scenario(
    agents=[agent1],
    tasks=[task1],
    scenario_name='Strike Mission'
)

# ===== STEP 4: Execute in BLADE =====
from blade import Game

game = Game(blade_scenario)

# Execute actions at scheduled times
for timestep in range(1000):
    # Check if there are actions for this timestep
    if timestep in blade_artifacts.execution_time_to_actions:
        actions = blade_artifacts.execution_time_to_actions[timestep]
        for action in actions:
            # Parse and execute action
            eval(f'game.{action}')  # Note: Use proper parsing in production!
    
    # Step simulation
    obs, reward, done, truncated, info = game.step(None)
    
    if done or truncated:
        break

# ===== STEP 5: Extract Observations (for RL) =====
from match_aou.utils.blade_utils.observation_utils import extract_observations

agent_obs = extract_observations(
    scenario=game.current_scenario,
    agent_id='f16_01'
)
```

---

## Action Template System

### How Templates Work

MATCH-AOU steps use **placeholders** that get replaced with actual values:

| Placeholder | Replaced With | Source |
|-------------|---------------|--------|
| `AGENT_ID` | Agent's unique ID | `agent.id` |
| `WEAPON_ID` | Weapon type ID | `agent.weapon_id` |
| `TARGET_ID` | Target unit ID | Specified in template |

### Template Examples

#### Move Template
```python
move_step = Step(
    location=Location(lat=35.5, lon=40.5),
    capabilities=[Capability('move')],
    step_type=move_type,
    effort=1,
    probability=1.0,
    action='move_aircraft(AGENT_ID, [[35.5, 40.5]])'
)

# After resolution:
# 'move_aircraft("f16_01", [[35.5, 40.5]])'
```

#### Attack Template
```python
attack_step = Step(
    location=target_location,
    capabilities=[Capability('attack')],
    step_type=attack_type,
    effort=2,
    probability=0.85,
    action='handle_aircraft_attack(AGENT_ID, "sam_05", WEAPON_ID, 2)'
)

# After resolution:
# 'handle_aircraft_attack("f16_01", "sam_05", "agm_88", 2)'
```

#### RTB Template
```python
rtb_step = Step(
    location=None,  # RTB doesn't need location
    capabilities=[Capability('move')],
    step_type=move_type,
    effort=0,
    probability=1.0,
    action='aircraft_return_to_base(AGENT_ID)'
)

# After resolution:
# 'aircraft_return_to_base("f16_01")'
```

### Multi-Step Action (Macro)

Some actions expand into multiple BLADE commands:

```python
# Single MATCH-AOU step
complex_attack = Step(
    location=target_loc,
    capabilities=[Capability('attack')],
    step_type=attack_type,
    effort=1,
    probability=0.8,
    action='MACRO_ATTACK'  # Special keyword
)

# Expands to:
# 1. launch_aircraft_from_airbase(AGENT_ID)
# 2. move_aircraft(AGENT_ID, [[target_lat, target_lon]])
# 3. handle_aircraft_attack(AGENT_ID, TARGET_ID, WEAPON_ID, qty)
```

---

## Complete Example

### Scenario: Coordinated Strike

```python
# ===== Setup =====
from match_aou.models import Agent, Task, Step, StepType, Capability, Location
from match_aou.solvers.match_aou_MINLP_solver import MatchAou
from match_aou.utils.blade_utils.blade_plan_utils import (
    assign_execution_times_and_resolve_actions
)
from match_aou.utils.blade_utils.scenario_factory import create_scenario
from blade import Game

# Distance function
def distance_cost(loc1, loc2):
    return ((loc1.lat - loc2.lat)**2 + (loc1.lon - loc2.lon)**2)**0.5 * 10

# Step types
attack_type = StepType('attack', base_cost=100)

# ===== Define Agents =====
agents = [
    Agent(
        location=Location(lat=35.0, lon=40.0),
        capabilities=[Capability('attack')],
        budget=1500.0,
        move_cost_function=distance_cost,
        agent_id='f16_01',
        weapon_id='agm_88',
        home_base_id='base_alpha'
    ),
    Agent(
        location=Location(lat=35.2, lon=40.1),
        capabilities=[Capability('attack')],
        budget=1500.0,
        move_cost_function=distance_cost,
        agent_id='f16_02',
        weapon_id='agm_88',
        home_base_id='base_alpha'
    )
]

# ===== Define Tasks =====
tasks = [
    Task(
        steps=[
            Step(
                location=Location(lat=36.0, lon=41.0),
                capabilities=[Capability('attack')],
                step_type=attack_type,
                effort=1,
                probability=0.7,  # Low probability → may assign 2 agents
                action='handle_aircraft_attack(AGENT_ID, "sam_01", WEAPON_ID, 2)'
            )
        ],
        utility=800
    )
]

# ===== Solve MATCH-AOU =====
solver = MatchAou(agents=agents, tasks=tasks)
solution, results, unselected = solver.solve()

print("\n=== MATCH-AOU Solution ===")
solver.display_solution(solution)

# ===== Convert to BLADE =====
blade_artifacts = assign_execution_times_and_resolve_actions(
    agents=agents,
    tasks=tasks,
    solution=solution
)

print("\n=== BLADE Action Schedule ===")
for timestep, actions in sorted(blade_artifacts.execution_time_to_actions.items()):
    print(f"t={timestep}: {actions}")

# ===== Create and Run BLADE Scenario =====
blade_scenario = create_scenario(agents=agents, tasks=tasks)
game = Game(blade_scenario)

print("\n=== Executing in BLADE ===")
for t in range(200):
    # Execute scheduled actions
    if t in blade_artifacts.execution_time_to_actions:
        for action_str in blade_artifacts.execution_time_to_actions[t]:
            print(f"t={t}: Executing {action_str}")
            # Execute action (parse properly in production)
            eval(f'game.{action_str}')
    
    # Step simulation
    obs, reward, done, truncated, info = game.step(None)
    
    if done or truncated:
        print(f"\nSimulation complete at t={t}")
        break

print("\n=== Final State ===")
print(f"Aircraft in scenario: {len(game.current_scenario.aircraft)}")
print(f"Facilities: {len(game.current_scenario.facilities)}")
```

---

## Troubleshooting

### Problem: Actions Not Executing

**Symptoms:** BLADE simulation runs but nothing happens

**Possible Causes:**
1. Action templates not resolved (still contain AGENT_ID)
2. Timestep mismatch
3. Action string format incorrect

**Solution:**
```python
# Debug: Print resolved actions
for t, actions in blade_artifacts.execution_time_to_actions.items():
    print(f"t={t}:")
    for action in actions:
        print(f"  {action}")
        # Check: Should NOT contain 'AGENT_ID' or 'WEAPON_ID'
```

### Problem: Wrong Agent IDs

**Symptoms:** `Aircraft with id 'AGENT_ID' not found`

**Cause:** Template placeholders not replaced

**Solution:**
```python
# Ensure agents have IDs set
agent = Agent(..., agent_id='f16_01')  # ← Must set this!

# Ensure weapon_id is set if using WEAPON_ID placeholder
agent = Agent(..., weapon_id='agm_88')  # ← Required for weapon templates
```

### Problem: Timing Issues

**Symptoms:** Actions execute in wrong order or too close together

**Cause:** Insufficient travel time between actions

**Solution:**
```python
# blade_plan_utils automatically adds travel time
# But you can check/adjust:
print(blade_artifacts.level_start_time)  # When each level starts

# If needed, manually adjust agent speed:
agent = Agent(..., speed=400.0)  # knots
```

### Problem: Missing Capabilities

**Symptoms:** Solver returns no solution or unassigned tasks

**Cause:** Agent capabilities don't match task requirements

**Solution:**
```python
# Debug: Check capability matching
for agent in agents:
    print(f"Agent {agent.id}: {[c.name for c in agent.capabilities]}")

for task_idx, task in enumerate(tasks):
    for step_idx, step in enumerate(task.steps):
        required = [c.name for c in step.capabilities]
        print(f"Task {task_idx} Step {step_idx} needs: {required}")
```

### Problem: Budget Exceeded

**Symptoms:** Some agents not assigned despite having capabilities

**Solution:**
```python
# Increase agent budgets
agent = Agent(..., budget=2000.0)  # ← Increase

# OR use risk factor to see if budget is the issue
solver = MatchAou(..., risk_factor=0.0)  # No safety margin

# Check actual costs:
for agent in agents:
    total_cost = 0
    if agent.id in solution:
        for task_idx, step_idx in solution[agent.id]:
            step = tasks[task_idx].steps[step_idx]
            move_cost = agent.move_cost(step.location)
            step_cost = agent.step_cost(step)
            total_cost += move_cost + step_cost
    print(f"Agent {agent.id}: {total_cost} / {agent.budget}")
```

---

## Quick Reference

### Conversion Checklist

- [ ] Agents have `agent_id` set
- [ ] Agents have `weapon_id` if using weapon actions
- [ ] Steps have `action` templates with placeholders
- [ ] Called `assign_execution_times_and_resolve_actions()`
- [ ] Created BLADE scenario with `create_scenario()`
- [ ] Execute actions at scheduled timesteps

### Common Action Templates

```python
# Move
action='move_aircraft(AGENT_ID, [[lat, lon]])'

# Attack
action='handle_aircraft_attack(AGENT_ID, "target_id", WEAPON_ID, qty)'

# RTB
action='aircraft_return_to_base(AGENT_ID)'

# Launch
action='launch_aircraft_from_airbase("base_id")'
```

---

**Last Updated:** February 2026  
**For MATCH-AOU API:** See `MATCH_AOU_API.md`  
**For BLADE API:** See `BLADE_API_DOCUMENTATION.md`