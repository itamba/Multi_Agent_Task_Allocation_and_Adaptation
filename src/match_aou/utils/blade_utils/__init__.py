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

from .observation_utils import update_agents_from_observation
from .scenario_factory import (
    create_agents_from_scenario,
    generate_attack_base_task,
    generate_attack_ship_task,
    _normalize_side_color,
)
from .blade_plan_utils import populate_blade_fields, BladePlanArtifacts
from .scenario_generator import ScenarioGenerator, VariationConfig

__all__ = [
    "create_agents_from_scenario",
    "generate_attack_base_task",
    "generate_attack_ship_task",
    "update_agents_from_observation",
    "populate_blade_fields",
    "BladePlanArtifacts",
    "_normalize_side_color",
    "ScenarioGenerator",
    "VariationConfig",
]