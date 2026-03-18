# Multi-Agent Task Allocation & Adaptation Project Structure
================================================================================

**Project Root:** `/home/claude/Multi_Agent_Task_Allocation_and_Adaptation`
**Total Python Files:** 49
**Total Packages:** 8

## Table of Contents
1. [Project Directory Tree](#project-directory-tree)
2. [Package Structure](#package-structure)
3. [Core Modules](#core-modules)
4. [Dependencies Map](#dependencies-map)
5. [File Inventory](#file-inventory)

## Project Directory Tree
--------------------------------------------------------------------------------

```
Multi_Agent_Task_Allocation_and_Adaptation/
├── data/                           # Data files
│   ├── agents_from_csv/            # Agent definitions
│   └── scenarios/                  # Scenario JSON files
├── scripts/                        # Demo and test scripts
├── src/match_aou/                  # Main source code
│   ├── models/                     # Data structures
│   │   ├── agent.py
│   │   ├── task.py
│   │   ├── step.py
│   │   ├── step_type.py
│   │   ├── capability.py
│   │   └── location.py
│   ├── solvers/                    # Optimization solvers
│   │   └── match_aou_MINLP_solver.py
│   ├── utils/                      # Utilities
│   │   ├── blade_utils/            # BLADE integration
│   │   │   ├── blade_executor_minimal.py
│   │   │   ├── blade_plan_utils.py
│   │   │   ├── observation_utils.py
│   │   │   └── scenario_factory.py
│   │   ├── scheduling_utils.py
│   │   ├── topology_utils.py
│   │   └── match_aou_parser.py
│   └── integrations/               # External integrations
│       └── panopticon-main/       # BLADE engine
│           └── gym/blade/
│               ├── units/          # Aircraft, Ships, etc.
│               ├── envs/           # Gym environment
│               ├── engine/         # Weapon engagement
│               ├── mission/        # Mission types
│               ├── db/             # Unit database
│               └── utils/          # BLADE utilities
├── BLADE_API_REFERENCE.md
└── [Generated Documentation]
```

## Package Structure
--------------------------------------------------------------------------------

Python packages (directories with `__init__.py`):

- **`src/match_aou/`**
- **`src/match_aou/integrations/`**
- **`src/match_aou/integrations/panopticon-main/gym/blade/`**
- **`src/match_aou/integrations/panopticon-main/gym/blade/envs/`**
- **`src/match_aou/models/`**
- **`src/match_aou/solvers/`**
- **`src/match_aou/utils/`**
- **`src/match_aou/utils/blade_utils/`**

## Core Modules
--------------------------------------------------------------------------------

### src/match_aou/models/

#### `agent.py`
- **Path:** `src/match_aou/models/agent.py`
- **Size:** 3.64 KB
- **Classes:** Agent
- **Functions:** __init__, move_cost, step_cost, has_capabilities, __repr__
- **Imports:** 1 modules

#### `capability.py`
- **Path:** `src/match_aou/models/capability.py`
- **Size:** 1.00 KB
- **Classes:** Capability
- **Functions:** __init__, matches_requirement, __repr__

#### `location.py`
- **Path:** `src/match_aou/models/location.py`
- **Size:** 1.15 KB
- **Classes:** Location
- **Functions:** __init__, distance_to, __repr__
- **Imports:** 1 modules

#### `step.py`
- **Path:** `src/match_aou/models/step.py`
- **Size:** 3.13 KB
- **Classes:** Step
- **Functions:** __init__, compute_step_cost, get_action, get_execution_time, __repr__
- **Imports:** 6 modules

#### `step_type.py`
- **Path:** `src/match_aou/models/step_type.py`
- **Size:** 1.20 KB
- **Classes:** StepType
- **Functions:** __init__, compute_cost, __repr__

#### `task.py`
- **Path:** `src/match_aou/models/task.py`
- **Size:** 1.92 KB
- **Classes:** Task
- **Functions:** __init__, __repr__

### src/match_aou/solvers/

#### `match_aou_MINLP_solver.py`
- **Path:** `src/match_aou/solvers/match_aou_MINLP_solver.py`
- **Size:** 9.74 KB
- **Classes:** MatchAou
- **Functions:** __init__, _add_objective, _add_constraints, solve, display_solution (+5 more)
- **Imports:** 13 modules

### src/match_aou/utils/blade_utils/

#### `blade_executor_minimal.py`
- **Path:** `src/match_aou/utils/blade_utils/blade_executor_minimal.py`
- **Size:** 15.35 KB
- **Classes:** _AgentExec, Candidate, BladeExecutorMinimal
- **Functions:** _get_sim_tick, _find_aircraft_obj, _get_aircraft_location, _find_airbase_and_inventory, _infer_airbase_id_for_aircraft (+13 more)
- **Imports:** 11 modules

#### `blade_plan_utils.py`
- **Path:** `src/match_aou/utils/blade_utils/blade_plan_utils.py`
- **Size:** 18.01 KB
- **Classes:** BladePlanArtifacts
- **Functions:** _ensure_execution_times, _ensure_actions_by_agent, _replace_action_placeholders, _serialize_events_one_action_per_tick, populate_blade_fields (+3 more)
- **Imports:** 12 modules

#### `observation_utils.py`
- **Path:** `src/match_aou/utils/blade_utils/observation_utils.py`
- **Size:** 2.63 KB
- **Functions:** update_agents_from_observation
- **Imports:** 6 modules

#### `scenario_factory.py`
- **Path:** `src/match_aou/utils/blade_utils/scenario_factory.py`
- **Size:** 6.96 KB
- **Functions:** _normalize_side_color, create_agents_from_scenario, generate_attack_base_task, generate_attack_ship_task, convert_unit_to_agent (+2 more)
- **Imports:** 11 modules

### src/match_aou/utils/

#### `match_aou_parser.py`
- **Path:** `src/match_aou/utils/match_aou_parser.py`
- **Size:** 3.14 KB
- **Functions:** load_data, load_json, load_csv, parse_agent, parse_task (+1 more)
- **Imports:** 8 modules

#### `scheduling_utils.py`
- **Path:** `src/match_aou/utils/scheduling_utils.py`
- **Size:** 6.57 KB
- **Classes:** PostSolveArtifacts
- **Functions:** _selected_tasks_from_solution, post_solve_filter_and_level, _add_level_to_solution_no_reindex
- **Imports:** 10 modules

#### `topology_utils.py`
- **Path:** `src/match_aou/utils/topology_utils.py`
- **Size:** 2.82 KB
- **Functions:** compute_topological_levels_selected, levels_to_layers
- **Imports:** 10 modules

### src/match_aou/integrations/panopticon-main/gym/blade/

#### `Doctrine.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/Doctrine.py`
- **Size:** 0.78 KB
- **Classes:** DoctrineType, SideDoctrine
- **Imports:** 3 modules

#### `Game.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/Game.py`
- **Size:** 51.51 KB
- **Classes:** Game
- **Functions:** __init__, remove_aircraft, land_aicraft, add_reference_point, remove_reference_point (+35 more)
- **Imports:** 31 modules

#### `Relationships.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/Relationships.py`
- **Size:** 2.38 KB
- **Classes:** Relationships
- **Functions:** __init__, add_hostile, remove_hostile, add_ally, remove_ally (+7 more)
- **Imports:** 3 modules

#### `Scenario.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/Scenario.py`
- **Size:** 10.92 KB
- **Classes:** Scenario
- **Functions:** __init__, get_default_doctrine, get_default_side_doctrine, get_side_doctrine, check_side_doctrine (+28 more)
- **Imports:** 16 modules

#### `Side.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/Side.py`
- **Size:** 0.67 KB
- **Classes:** Side
- **Functions:** __init__, to_dict
- **Imports:** 3 modules

#### `UnitDb.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/db/UnitDb.py`
- **Size:** 20.40 KB

#### `weaponEngagement.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/engine/weaponEngagement.py`
- **Size:** 7.82 KB
- **Functions:** is_threat_detected, weapon_can_engage_target, check_target_tracked_by_count, weapon_endgame, launch_weapon (+3 more)
- **Imports:** 15 modules

#### `blade.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/envs/blade.py`
- **Size:** 2.78 KB
- **Classes:** BLADE
- **Functions:** __init__, _get_obs, _get_info, reset, step (+2 more)
- **Imports:** 7 modules

#### `PatrolMission.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/mission/PatrolMission.py`
- **Size:** 1.88 KB
- **Classes:** PatrolMission
- **Functions:** __init__, update_patrol_area_geometry, check_if_coordinates_is_within_patrol_area, generate_random_coordinates_within_patrol_area, to_dict
- **Imports:** 6 modules

#### `StrikeMission.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/mission/StrikeMission.py`
- **Size:** 0.80 KB
- **Classes:** StrikeMission
- **Functions:** __init__, to_dict
- **Imports:** 2 modules

#### `Airbase.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/units/Airbase.py`
- **Size:** 1.39 KB
- **Classes:** Airbase
- **Functions:** __init__, to_dict
- **Imports:** 6 modules

#### `Aircraft.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/units/Aircraft.py`
- **Size:** 5.30 KB
- **Classes:** BlackBox, Aircraft
- **Functions:** __init__, log, get_logs, get_last_log_pp, get_last_log (+7 more)
- **Imports:** 7 modules

#### `Facility.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/units/Facility.py`
- **Size:** 1.89 KB
- **Classes:** Facility
- **Functions:** __init__, get_total_weapon_quantity, get_weapon_with_highest_engagement_range, get_detection_range, to_dict
- **Imports:** 6 modules

#### `ReferencePoint.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/units/ReferencePoint.py`
- **Size:** 1.04 KB
- **Classes:** ReferencePoint
- **Functions:** __init__, to_dict
- **Imports:** 4 modules

#### `Ship.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/units/Ship.py`
- **Size:** 3.16 KB
- **Classes:** Ship
- **Functions:** __init__, get_total_weapon_quantity, get_weapon_with_highest_engagement_range, get_detection_range, get_weapon (+1 more)
- **Imports:** 7 modules

#### `Weapon.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/units/Weapon.py`
- **Size:** 2.50 KB
- **Classes:** Weapon
- **Functions:** __init__, get_engagement_range, to_dict
- **Imports:** 5 modules

#### `PlaybackRecorder.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/utils/PlaybackRecorder.py`
- **Size:** 3.33 KB
- **Classes:** PlaybackRecorder
- **Functions:** __init__, should_record, reset, start_recording, record_step (+1 more)
- **Imports:** 3 modules

#### `colors.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/utils/colors.py`
- **Size:** 1.07 KB
- **Classes:** SIDE_COLOR
- **Functions:** convert_color_name_to_side_color, upper
- **Imports:** 1 modules

#### `constants.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/utils/constants.py`
- **Size:** 0.39 KB

#### `utils.py`
- **Path:** `src/match_aou/integrations/panopticon-main/gym/blade/utils/utils.py`
- **Size:** 4.20 KB
- **Functions:** to_radians, to_degrees, get_bearing_between_two_points, get_distance_between_two_points, get_terminal_coordinates_from_distance_and_bearing (+5 more)
- **Imports:** 7 modules

### scripts/

#### `match_aou_demo_main.py`
- **Path:** `scripts/match_aou_demo_main.py`
- **Size:** 16.95 KB
- **Classes:** PlanStats
- **Functions:** _setup_logging, _header, _ensure_episode_horizon, _cleanup_previous_exports, _get_sim_tick (+5 more)
- **Imports:** 21 modules

## Dependencies Map
--------------------------------------------------------------------------------

### External Dependencies

```
__future__
agent
argparse
blade_plan_utils
capability
gymnasium
haversine
location
match_aou_MINLP_solver
models
numpy
observation_utils
pyomo
random
scenario_factory
scheduling_utils
setuptools
shapely
solvers
stable_baselines3
step
step_type
task
topology_utils
uuid
```

## File Inventory
--------------------------------------------------------------------------------

| File Path | Size (KB) | Classes | Functions | Imports |
|-----------|-----------|---------|-----------|---------|
| `analyze_project.py` | 3.6 | 0 | 3 | 5 |
| `scripts/match_aou_demo_main.py` | 17.0 | 1 | 10 | 21 |
| `src/match_aou/__init__.py` | 0.2 | 0 | 0 | 7 |
| `src/match_aou/integrations/__init__.py` | 0.0 | 0 | 0 | 0 |
| `src/match_aou/integrations/panopticon-main/docs/conf.py` | 1.8 | 0 | 0 | 2 |
| `src/match_aou/integrations/panopticon-main/gym/blade/Doctrine.py` | 0.8 | 2 | 0 | 3 |
| `src/match_aou/integrations/panopticon-main/gym/blade/Game.py` | 51.5 | 1 | 40 | 31 |
| `src/match_aou/integrations/panopticon-main/gym/blade/Relationships.py` | 2.4 | 1 | 12 | 3 |
| `src/match_aou/integrations/panopticon-main/gym/blade/Scenario.py` | 10.9 | 1 | 33 | 16 |
| `src/match_aou/integrations/panopticon-main/gym/blade/Side.py` | 0.7 | 1 | 2 | 3 |
| `src/match_aou/integrations/panopticon-main/gym/blade/__init__.py` | 0.1 | 0 | 0 | 1 |
| `src/match_aou/integrations/panopticon-main/gym/blade/db/UnitDb.py` | 20.4 | 0 | 0 | 0 |
| `src/match_aou/integrations/panopticon-main/gym/blade/engine/weaponEngagement.py` | 7.8 | 0 | 8 | 15 |
| `src/match_aou/integrations/panopticon-main/gym/blade/envs/__init__.py` | 0.0 | 0 | 0 | 1 |
| `src/match_aou/integrations/panopticon-main/gym/blade/envs/blade.py` | 2.8 | 1 | 7 | 7 |
| `src/match_aou/integrations/panopticon-main/gym/blade/mission/PatrolMission.py` | 1.9 | 1 | 5 | 6 |
| `src/match_aou/integrations/panopticon-main/gym/blade/mission/StrikeMission.py` | 0.8 | 1 | 2 | 2 |
| `src/match_aou/integrations/panopticon-main/gym/blade/units/Airbase.py` | 1.4 | 1 | 2 | 6 |
| `src/match_aou/integrations/panopticon-main/gym/blade/units/Aircraft.py` | 5.3 | 2 | 12 | 7 |
| `src/match_aou/integrations/panopticon-main/gym/blade/units/Facility.py` | 1.9 | 1 | 5 | 6 |
| `src/match_aou/integrations/panopticon-main/gym/blade/units/ReferencePoint.py` | 1.0 | 1 | 2 | 4 |
| `src/match_aou/integrations/panopticon-main/gym/blade/units/Ship.py` | 3.2 | 1 | 6 | 7 |
| `src/match_aou/integrations/panopticon-main/gym/blade/units/Weapon.py` | 2.5 | 1 | 3 | 5 |
| `src/match_aou/integrations/panopticon-main/gym/blade/utils/PlaybackRecorder.py` | 3.3 | 1 | 6 | 3 |
| `src/match_aou/integrations/panopticon-main/gym/blade/utils/colors.py` | 1.1 | 1 | 2 | 1 |
| `src/match_aou/integrations/panopticon-main/gym/blade/utils/constants.py` | 0.4 | 0 | 0 | 0 |
| `src/match_aou/integrations/panopticon-main/gym/blade/utils/utils.py` | 4.2 | 0 | 10 | 7 |
| `src/match_aou/integrations/panopticon-main/gym/scripts/generate_load_test_scenario.py` | 7.0 | 0 | 3 | 5 |
| `src/match_aou/integrations/panopticon-main/gym/scripts/simple_demo/demo.py` | 3.5 | 0 | 1 | 5 |
| `src/match_aou/integrations/panopticon-main/gym/scripts/stable_baselines/train.py` | 7.7 | 0 | 8 | 8 |
| `src/match_aou/integrations/panopticon-main/gym/setup.py` | 0.7 | 0 | 0 | 1 |
| `src/match_aou/models/__init__.py` | 0.2 | 0 | 0 | 6 |
| `src/match_aou/models/agent.py` | 3.6 | 1 | 5 | 1 |
| `src/match_aou/models/capability.py` | 1.0 | 1 | 3 | 0 |
| `src/match_aou/models/location.py` | 1.2 | 1 | 3 | 1 |
| `src/match_aou/models/step.py` | 3.1 | 1 | 5 | 6 |
| `src/match_aou/models/step_type.py` | 1.2 | 1 | 3 | 0 |
| `src/match_aou/models/task.py` | 1.9 | 1 | 2 | 0 |
| `src/match_aou/solvers/__init__.py` | 0.1 | 0 | 0 | 1 |
| `src/match_aou/solvers/match_aou_MINLP_solver.py` | 9.7 | 1 | 10 | 13 |
| `src/match_aou/utils/__init__.py` | 0.3 | 0 | 0 | 4 |
| `src/match_aou/utils/blade_utils/__init__.py` | 0.9 | 0 | 0 | 6 |
| `src/match_aou/utils/blade_utils/blade_executor_minimal.py` | 15.3 | 3 | 18 | 11 |
| `src/match_aou/utils/blade_utils/blade_plan_utils.py` | 18.0 | 1 | 8 | 12 |
| `src/match_aou/utils/blade_utils/observation_utils.py` | 2.6 | 0 | 1 | 6 |
| `src/match_aou/utils/blade_utils/scenario_factory.py` | 7.0 | 0 | 7 | 11 |
| `src/match_aou/utils/match_aou_parser.py` | 3.1 | 0 | 6 | 8 |
| `src/match_aou/utils/scheduling_utils.py` | 6.6 | 1 | 3 | 10 |
| `src/match_aou/utils/topology_utils.py` | 2.8 | 0 | 2 | 10 |
