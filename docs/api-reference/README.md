# API Reference

Complete API documentation for all project components.

## 📖 Available APIs

### [MATCH-AOU API](MATCH_AOU_API.md)
**Task allocation solver**

Learn how to:
- Define agents, tasks, and steps
- Configure the MINLP solver
- Interpret solution results
- Optimize task allocation

**When to use:** When you need to assign tasks to agents optimally.

---

### [BLADE API](BLADE_API_DOCUMENTATION.md)
**Simulation engine**

Learn how to:
- Control the simulation (Game class)
- Manage units (Aircraft, Ships, Facilities)
- Execute actions (move, attack, RTB)
- Work with scenarios and missions

**When to use:** When you need to simulate multi-agent operations.

---

### [Integration Guide](INTEGRATION_GUIDE.md)
**Connecting MATCH-AOU with BLADE**

Learn how to:
- Convert MATCH-AOU plans to BLADE actions
- Use blade_utils module
- Handle action templates (AGENT_ID, WEAPON_ID)
- Execute plans in simulation
- Extract observations for RL

**When to use:** When connecting the solver output to the simulator.

---

## 🎯 Quick Reference

### Common Tasks

**Solve task allocation:**
```python
from match_aou.solvers import MatchAou
solver = MatchAou(agents, tasks)
solution, results, unselected = solver.solve()
```
→ See [MATCH-AOU API](MATCH_AOU_API.md)

**Run BLADE simulation:**
```python
from blade import Game, Scenario
game = Game(scenario)
obs, reward, done, truncated, info = game.step(action)
```
→ See [BLADE API](BLADE_API_DOCUMENTATION.md)

**Convert and execute plan:**
```python
from match_aou.utils.blade_utils import assign_execution_times_and_resolve_actions
blade_artifacts = assign_execution_times_and_resolve_actions(agents, tasks, solution)
```
→ See [Integration Guide](INTEGRATION_GUIDE.md)

---

## 📊 API Coverage

| Component | Classes | Key Methods | Examples |
|-----------|---------|-------------|----------|
| **MATCH-AOU** | 7 | solve(), display_solution() | 3 complete |
| **BLADE** | 10+ | step(), move_aircraft(), handle_attack() | 4 complete |
| **Integration** | - | assign_execution_times(), create_scenario() | 1 end-to-end |

---

[← Back to Main Documentation](../README.md)
