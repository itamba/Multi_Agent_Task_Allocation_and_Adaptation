# MATCH-AOU API Reference
================================================================================

**MATCH-AOU** (Multi-Agent Task Allocation, Ordering, and Utilization) is an
optimization solver that assigns tasks to agents while maximizing utility.

**What's Covered:** MATCH-AOU solver, data structures, usage examples
**Not Covered:** BLADE simulation (see `BLADE_API_DOCUMENTATION.md`), 
Integration details (see `INTEGRATION_GUIDE.md`)

**See Also:** `INTEGRATION_GUIDE.md` for connecting MATCH-AOU with BLADE

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Data Structures](#data-structures)
4. [Solver Usage](#solver-usage)
5. [Solution Format](#solution-format)
6. [Complete Example](#complete-example)
7. [Tips & Best Practices](#tips--best-practices)

---

## Overview

MATCH-AOU solves the problem of:
- **Assigning** tasks (with multiple steps) to agents
- **Maximizing** utility while respecting constraints
- **Ensuring** agents have required capabilities
- **Managing** budgets (fuel, time, etc.)
- **Respecting** task precedence relations

**Problem Formulation:** Mixed-Integer Nonlinear Program (MINLP)
**Solver:** Pyomo with backend solvers (GLPK, CBC, IPOPT)

---

## Quick Start

```python
from match_aou.models import Agent, Task, Step, StepType, Capability, Location
from match_aou.solvers.match_aou_MINLP_solver import MatchAou

# 1. Define agents
agent1 = Agent(
    location=Location(lat=35.0, lon=40.0),
    capabilities=[Capability('attack'), Capability('surveillance')],
    budget=1000.0,  # Fuel budget
    move_cost_function=lambda src, dst: calculate_distance(src, dst),
    agent_id='f16_01'
)

# 2. Define tasks
task1 = Task(
    steps=[
        Step(
            location=Location(lat=35.5, lon=40.5),
            capabilities=[Capability('attack')],
            step_type=StepType('attack', base_cost=100),
            effort=1,
            probability=0.9
        )
    ],
    utility=500
)

# 3. Solve
solver = MatchAou(agents=[agent1], tasks=[task1])
solution, results, unselected = solver.solve()

# 4. View results
solver.display_solution(solution)
```

---

## Data Structures

### Agent

**Location:** `match_aou/models/agent.py`

Represents an agent that can perform tasks.

**Constructor:**
```python
Agent(
    location: Location,              # Starting position
    capabilities: List[Capability],  # What agent can do
    budget: float,                   # Resource constraint (e.g., fuel)
    move_cost_function: Callable,    # Function(src, dst) -> cost
    speed: float = None,             # Agent speed
    return_location: Location = None, # Where to return (RTB)
    agent_id: str = None,            # Unique identifier
    side_color: str = None,          # Team color
    weapon_id: str = None,           # Weapon type
    home_base_id: str = None,        # Home base ID
    target_id: str = None            # Current target
)
```

**Key Methods:**
```python
agent.move_cost(destination: Location) -> float
    # Calculate cost of moving to destination

agent.has_capabilities(required: List[Capability]) -> bool
    # Check if agent has all required capabilities

agent.step_cost(step: Step) -> float
    # Calculate cost of performing a step
```

### Task

**Location:** `match_aou/models/task.py`

Represents a task composed of multiple steps.

**Constructor:**
```python
Task(
    steps: List[Step],                      # Ordered list of steps
    utility: float,                         # Value of completing task
    precedence_relations: List[Tuple] = []  # (step_k1, step_k2) constraints
)
```

**Properties:**
- `task.steps`: List of Step objects
- `task.utility`: Reward for completion
- `task.precedence_relations`: Ordering constraints between steps

### Step

**Location:** `match_aou/models/step.py`

Represents a single action in a task.

**Constructor:**
```python
Step(
    location: Optional[Location],    # Where step happens
    capabilities: List[Capability],  # Required capabilities
    step_type: StepType,             # Type of step
    effort: int,                     # Amount of work required
    probability: float,              # Success probability (0-1)
    action: Optional[str] = None     # BLADE action template
)
```

**Key Methods:**
```python
step.compute_step_cost() -> float
    # Calculate cost of this step

step.get_action(agent_id: str) -> str
    # Get action string with agent ID filled in
    # Example: 'move_aircraft(AGENT_ID, ...)' → 'move_aircraft("f16_01", ...)'

step.get_execution_time(agent_id: str) -> int
    # Get when this step executes for specific agent
```

**Important Properties:**
- `step.execution_times`: Dict[agent_id -> timestep] (filled by post-processing)
- `step.action`: Template string with placeholders (AGENT_ID, WEAPON_ID)

### StepType

**Location:** `match_aou/models/step_type.py`

Defines a category of steps with associated costs.

**Constructor:**
```python
StepType(
    name: str,                       # e.g., 'attack', 'surveillance'
    base_cost: float,                # Cost per unit of effort
    custom_cost_function: Callable = None  # Optional custom cost
)
```

**Method:**
```python
step_type.compute_cost(effort: int) -> float
    # Returns: base_cost * effort (or custom function result)
```

**Common Examples:**
```python
attack_type = StepType('attack', base_cost=100)
surveillance_type = StepType('surveillance', base_cost=50)
move_type = StepType('move', base_cost=10)
```

### Capability

**Location:** `match_aou/models/capability.py`

```python
Capability(name: str)
```

**Examples:**
```python
attack_cap = Capability('attack')
surveillance_cap = Capability('surveillance')
electronic_warfare = Capability('ew')
```

### Location

**Location:** `match_aou/models/location.py`

```python
Location(lat: float, lon: float)
```

Represents geographic coordinates.

---

## Solver Usage

### MatchAou Class

**Location:** `match_aou/solvers/match_aou_MINLP_solver.py`

**Constructor:**
```python
MatchAou(
    agents: List[Agent],
    tasks: List[Task],
    precedence_relations: List[Tuple[int, int]] = None,  # (task_j1, task_j2)
    risk_factor: float = 0.0  # Budget safety margin (0-1)
)
```

**Parameters:**
- `agents`: List of Agent objects
- `tasks`: List of Task objects
- `precedence_relations`: Task-level precedence (j1 must complete before j2)
- `risk_factor`: Conservative budget margin (e.g., 0.1 = keep 10% reserve)

### Solving the Problem

```python
solution, results, unselected_tasks = solver.solve(solver_name='glpk')
```

**Returns:**
1. `solution`: Dict[agent_id -> List[Tuple[task_idx, step_idx]]]
2. `results`: Pyomo solver results object
3. `unselected_tasks`: List of task indices not assigned

**Solver Options:**
- `'glpk'`: Open-source, good for small-medium problems
- `'cbc'`: Coin-OR solver, good for larger problems
- `'ipopt'`: Nonlinear solver (if you have MINLP problems)
- `'gurobi'`: Commercial, very fast (requires license)

### Displaying Results

```python
solver.display_solution(solution)
```

**Output Example:**
```
Assigned Tasks:
Agent f16_01 assigned to steps:
  Task 0, Step 0
  Task 1, Step 0

Unassigned Tasks:
Task 2 is unassigned (Utility: 300)
```

---

## Solution Format

### Solution Dictionary

The `solution` returned by `solve()` has this structure:

```python
{
    'agent_id_1': [(task_idx, step_idx), (task_idx, step_idx), ...],
    'agent_id_2': [(task_idx, step_idx), ...],
    ...
}
```

**Example:**
```python
{
    'f16_01': [(0, 0), (1, 0), (1, 1)],  # Task 0 step 0, Task 1 steps 0&1
    'f16_02': [(2, 0)]                   # Task 2 step 0
}
```

**Interpreting the Solution:**
- Each tuple `(task_idx, step_idx)` means agent is assigned to that step
- Task and step indices start at 0
- Steps are NOT necessarily in execution order (use precedence)

### Accessing Task/Step Details

```python
# Get assigned steps for an agent
agent_assignments = solution['f16_01']

# Access the actual Step object
for task_idx, step_idx in agent_assignments:
    task = tasks[task_idx]
    step = task.steps[step_idx]
    
    print(f"Agent assigned to: {step.step_type.name}")
    print(f"Location: {step.location}")
    print(f"Effort: {step.effort}")
```

---

## Complete Example

### Scenario: Two Aircraft Strike Mission

```python
from match_aou.models import Agent, Task, Step, StepType, Capability, Location
from match_aou.solvers.match_aou_MINLP_solver import MatchAou

# Helper: Simple distance function
def distance_cost(loc1, loc2):
    lat_diff = abs(loc1.lat - loc2.lat)
    lon_diff = abs(loc1.lon - loc2.lon)
    return (lat_diff**2 + lon_diff**2)**0.5 * 10  # Fuel cost

# Step types
attack_type = StepType('attack', base_cost=100)
surveillance_type = StepType('surveillance', base_cost=50)

# Agents
f16_01 = Agent(
    location=Location(lat=35.0, lon=40.0),
    capabilities=[Capability('attack'), Capability('surveillance')],
    budget=1000.0,
    move_cost_function=distance_cost,
    agent_id='f16_01',
    weapon_id='aim_120'
)

f16_02 = Agent(
    location=Location(lat=35.2, lon=40.1),
    capabilities=[Capability('attack')],
    budget=800.0,
    move_cost_function=distance_cost,
    agent_id='f16_02',
    weapon_id='aim_120'
)

# Tasks
task_1 = Task(
    steps=[
        Step(
            location=Location(lat=35.5, lon=40.5),
            capabilities=[Capability('surveillance')],
            step_type=surveillance_type,
            effort=1,
            probability=0.95
        ),
        Step(
            location=Location(lat=35.6, lon=40.6),
            capabilities=[Capability('attack')],
            step_type=attack_type,
            effort=2,
            probability=0.85
        )
    ],
    utility=500
)

task_2 = Task(
    steps=[
        Step(
            location=Location(lat=36.0, lon=41.0),
            capabilities=[Capability('attack')],
            step_type=attack_type,
            effort=1,
            probability=0.9
        )
    ],
    utility=300
)

# Solve
solver = MatchAou(
    agents=[f16_01, f16_02],
    tasks=[task_1, task_2],
    risk_factor=0.1  # Keep 10% fuel reserve
)

solution, results, unselected = solver.solve(solver_name='glpk')

# Display
solver.display_solution(solution)

# Process solution
if solution:
    for agent_id, assignments in solution.items():
        print(f"\n{agent_id} plan:")
        for task_idx, step_idx in assignments:
            step = [task_1, task_2][task_idx].steps[step_idx]
            print(f"  - {step.step_type.name} at {step.location}")
```

---

## Tips & Best Practices

### 1. Budget Management
- **Always include movement costs** in agent budgets
- **Use risk_factor** to keep a safety margin (0.1-0.2 recommended)
- **Account for return trip** by setting `return_location`

### 2. Capabilities
- **Match capability names exactly** (case-sensitive)
- **Keep capability granularity reasonable** (not too many)
- **Use descriptive names** ('air_to_air', 'ground_attack', etc.)

### 3. Task Design
- **Break complex tasks into steps** for flexibility
- **Set realistic probabilities** (0.7-0.95 typical)
- **Use precedence** when step order matters

### 4. Solver Selection
- **Start with GLPK** for prototyping
- **Use CBC** for larger problems (>20 agents)
- **Consider Gurobi** if you have license and need speed

### 5. Performance Tips
- **Limit task steps** to 3-5 when possible
- **Pre-filter impossible assignments** before solving
- **Batch similar tasks** to reduce problem size
- **Use location clustering** to reduce distance calculations

### 6. Debugging
```python
# Check if problem is feasible
if solution is None:
    print("No solution found. Check:")
    print("- Agent budgets sufficient?")
    print("- Agents have required capabilities?")
    print("- Task precedence not circular?")

# Examine unselected tasks
for task_idx in unselected:
    print(f"Task {task_idx} not selected: {tasks[task_idx].utility}")
```

### 7. Integration with BLADE
**Do NOT manually execute plans** - use the integration utilities!

See `INTEGRATION_GUIDE.md` for details on:
- Converting MATCH-AOU solution to BLADE actions
- Timing and execution order
- Action template resolution

---

## Common Patterns

### Pattern 1: Multi-Agent Coordination
```python
# Task requires 2 agents for higher success probability
coordinated_attack = Task(
    steps=[
        Step(  # Can be done by multiple agents
            location=target_loc,
            capabilities=[Capability('attack')],
            step_type=attack_type,
            effort=1,
            probability=0.6  # Low per-agent, high if both assigned
        )
    ],
    utility=800  # High reward for coordination
)
```

### Pattern 2: Sequential Task Chain
```python
# Task 0 must complete before Task 1
solver = MatchAou(
    agents=agents,
    tasks=[recon_task, strike_task],
    precedence_relations=[(0, 1)]  # Task 0 before Task 1
)
```

### Pattern 3: Fallback Assignments
```python
# Create tasks with different utility/difficulty
high_value = Task(steps=[difficult_step], utility=1000)
medium_value = Task(steps=[medium_step], utility=500)
low_value = Task(steps=[easy_step], utility=200)

# Solver will pick best combination given constraints
```

---

**Last Updated:** February 2026  
**MATCH-AOU Version:** MINLP Solver  
**For Integration:** See `INTEGRATION_GUIDE.md`