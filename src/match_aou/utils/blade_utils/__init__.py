"""match_aou.utils.blade_utils

Blade-specific utilities used by demos / scenario glue code.

Design goals
- Keep planning (solver) independent of the simulator.
- Group Blade-dependent parsing and scheduling in one place.
- Provide small, testable functions.

Typical flow
1) Build MATCH-AOU inputs (Agents/Tasks) from a Blade Scenario observation.
2) Solve MATCH-AOU (planning).
3) Post-process the solution to produce an execution order / schedule (simulation).
4) (Optional) Generate scenario variations for diverse RL training.
"""

from .scenario_factory import (
    create_agents_from_scenario,
    generate_all_enemy_tasks,
    _normalize_side_color,
)
from .scenario_generator import ScenarioGenerator, VariationConfig

__all__ = [
    "create_agents_from_scenario",
    "generate_all_enemy_tasks",
    "_normalize_side_color",
    "ScenarioGenerator",
    "VariationConfig",
]