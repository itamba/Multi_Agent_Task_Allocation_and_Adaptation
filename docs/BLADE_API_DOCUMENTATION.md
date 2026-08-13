# BLADE API Reference — this project's vendored fork

**Scope.** This document describes the BLADE (Panopticon) engine **as vendored in this
repository**, at:

```
src/match_aou/integrations/panopticon-main/gym/blade/
```

It is a *fork* reference, not a generic BLADE manual. Every symbol, signature and default
below was read from that source tree. The engine is **frozen** for this project — do not
refactor it; see `CLAUDE.md` §2.

The fork is editable-installed into the project environment, so `import blade` resolves to
the tree above:

```bash
pip install -e src/match_aou/integrations/panopticon-main/gym
```

---

## Table of contents

1. [Importing — read this first](#1-importing--read-this-first)
2. [Loading a scenario](#2-loading-a-scenario)
3. [The Gym environment](#3-the-gym-environment)
4. [Game — the simulation controller](#4-game--the-simulation-controller)
5. [Project-specific additive APIs](#5-project-specific-additive-apis)
6. [Scenario — state container and lookups](#6-scenario--state-container-and-lookups)
7. [Unit classes](#7-unit-classes)
8. [Missions](#8-missions)
9. [Recording and playback](#9-recording-and-playback)
10. [Utility and engine functions](#10-utility-and-engine-functions)
11. [Units of measure](#11-units-of-measure)
12. [Integration with this project](#12-integration-with-this-project)
13. [Gotchas](#13-gotchas)

---

## 1. Importing — read this first

`blade/__init__.py` contains **only** a Gymnasium environment registration. It exports no
classes:

```python
# blade/__init__.py  (complete)
from gymnasium.envs.registration import register

register(
    id="blade/BLADE-v0",
    entry_point="blade.envs:BLADE",
    max_episode_steps=2000,
)
```

Therefore:

```python
# CORRECT — import the class from its module
from blade.Game import Game
from blade.Scenario import Scenario
from blade.units.Aircraft import Aircraft
```

```python
# WRONG — and it fails LATE, not at import time
from blade import Game
game = Game(...)   # TypeError: 'module' object is not callable
```

`from blade import Game` does **not** raise `ImportError`. Python falls back to importing
the *submodule* `blade.Game`, so the name binds to a module object and the failure only
surfaces at the call site. Always import from the module path.

| Symbol | Module |
|---|---|
| `Game` | `blade.Game` |
| `Scenario` | `blade.Scenario` |
| `Side` | `blade.Side` |
| `Relationships` | `blade.Relationships` |
| `DoctrineType`, `SideDoctrine` | `blade.Doctrine` |
| `Aircraft`, `Ship`, `Facility`, `Airbase`, `Weapon`, `ReferencePoint` | `blade.units.<Name>` |
| `StrikeMission`, `PatrolMission` | `blade.mission.<Name>` |
| `BLADE` (the Gym env) | `blade.envs.blade` |
| `PlaybackRecorder` | `blade.utils.PlaybackRecorder` |

---

## 2. Loading a scenario

There is **no** `Scenario.load_from_file(...)` and **no** `Scenario.from_json(...)` in this
fork. A scenario is loaded by constructing an empty `Scenario`, wrapping it in a `Game`,
and handing the `Game` a **JSON string**:

```python
from blade.Game import Game
from blade.Scenario import Scenario

game = Game(
    current_scenario=Scenario(),
    record_every_seconds=None,      # None disables periodic recording
    recording_export_path=".",
)

scenario_json = open("data/scenarios/strike_training_4v5.json", encoding="utf-8").read()
game.load_scenario(scenario_json)   # takes the JSON STRING, not a path
```

`Game.__init__` signature:

```python
Game(
    current_scenario: Scenario,
    record_every_seconds: Optional[int] = None,
    recording_export_path: Optional[str] = ".",
)
```

`Game.load_scenario(scenario_string: str) -> None` parses the string and expects the
top-level keys `currentSideId`, `mapView` and `currentScenario`. It also sets
`game.current_side_id` from the file.

This is exactly the path the project uses — see
`match_aou.rl.training.graph_episode_setup._build_env`.

---

## 3. The Gym environment

The registered id is `blade/BLADE-v0`, and the environment class is named **`BLADE`**
(there is no `BladeEnv`). It wraps an **already-constructed `Game`** — it does not load a
scenario file itself:

```python
import gymnasium
from blade.Game import Game
from blade.Scenario import Scenario

game = Game(current_scenario=Scenario())
game.load_scenario(scenario_json)

env = gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=2000)
obs, info = env.reset()                       # obs IS a Scenario object
obs, reward, terminated, truncated, info = env.step("<command string>")
env.close()
```

`BLADE.__init__` signature:

```python
BLADE(
    render_mode=None,
    game: Game = None,
    observation_space=None,
    action_space=None,
    action_transform_fnc=None,
    observation_filter_fnc=None,
    reward_filter_fnc=None,
    termination_filter_fnc=None,
)
```

The observation is the live `Scenario` object, not an array. The optional `*_fnc` hooks let
a caller project it into a real space; this project does not use them — it reads the
`Scenario` directly.

---

## 4. `Game` — the simulation controller

### Simulation control

```python
game.step(action) -> Tuple[Scenario, float, bool, bool, dict]
game.reset() -> None
game.update_game_state() -> None
game.check_game_ended() -> bool
game.export_scenario() -> dict
```

`step()` calls `handle_action(action)` then `update_game_state()`, and returns
`(observation, reward, terminated, truncated, info)`.

**Important, and easy to get wrong:** in this fork `step()` always returns `reward = 0` and
`terminated = False`, `truncated` comes from `check_game_ended()` which **always returns
`False`**, and `info` is always `{}` (`_get_info` returns an empty dict). Episode
termination therefore comes from the Gymnasium `max_episode_steps` TimeLimit wrapper, or
from the caller's own logic — never from the engine's reward or terminated flags.

`reset()` deep-copies `initial_scenario` back into `current_scenario` and re-selects the
first side.

### Actions

`Game.handle_action(action: list | str) -> None` accepts **either a single command string
or a list of them**, executing each. That is what lets this project issue one command per
agent in a single tick.

```python
game.move_aircraft(aircraft_id: str, new_coordinates: list) -> Aircraft | None
game.aircraft_return_to_base(aircraft_id: str) -> Aircraft | None
game.handle_aircraft_attack(aircraft_id, target_id, weapon_id=None, weapon_quantity=2) -> None
game.land_aicraft(aircraft_id: str) -> None          # NOTE the engine's spelling
game.remove_aircraft(aircraft_id: str) -> None

game.move_ship(ship_id: str, new_coordinates: list) -> Ship | None
game.handle_ship_attack(ship_id, target_id, weapon_id, weapon_quantity) -> None

game.launch_aircraft_from_airbase(airbase_id: str, aircraft_id: str | None = None) -> Aircraft | None
game.launch_aircraft_from_ship(ship_id: str) -> Aircraft | None
```

`move_aircraft` takes a list of `[lat, lon]` waypoints and **clears the existing route on
every call**, so re-issuing an identical move resets progress.

### Missions

The creation methods are `create_*`, not `add_*`, and they return `None`:

```python
game.create_strike_mission(mission_name: str, assigned_attackers: list[str], assigned_targets: list[str]) -> None
game.create_patrol_mission(mission_name: str, assigned_units: list[str], assigned_area: list[ReferencePoint]) -> None
game.update_strike_mission(mission_id, mission_name, assigned_attackers, assigned_targets) -> None
game.update_patrol_mission(mission_id, mission_name, assigned_units, assigned_area) -> None
game.delete_mission(mission_id: str) -> None
game.clear_completed_strike_missions() -> None
```

`create_patrol_mission` silently returns if `len(assigned_area) < 3`.

### Fuel and state updates

```python
game.get_fuel_needed_to_return_to_base(aircraft: Aircraft) -> float
game.update_all_aircraft_position() -> None
game.update_all_ship_position() -> None
game.update_onboard_weapon_positions() -> None
game.aircraft_air_to_air_engagement() -> None
game.facility_auto_defense() -> None
game.ship_auto_defense() -> None
```

`update_all_aircraft_position` burns `fuel_rate / 3600` per tick for **every** airborne
aircraft, including one with no route.

### Reference points

```python
game.add_reference_point(reference_point_name: str, latitude: float, longitude: float) -> ReferencePoint
game.remove_reference_point(reference_point_id: str) -> None
```

---

## 5. Project-specific additive APIs

Two `Game` methods carry **additive edits made for this project**. They are load-bearing —
the graph executor depends on them (`CLAUDE.md` §2).

### Targeted launch

```python
game.launch_aircraft_from_airbase(airbase_id: str, aircraft_id: str | None = None) -> Aircraft | None
```

- **Omitting `aircraft_id`** preserves the original FIFO behaviour: `airbase.aircraft.pop(0)`.
- **Passing `aircraft_id`** launches that specific aircraft, matched by `str(ac.id)`. If it
  is not in the airbase inventory the method returns `None` — it never launches a different
  aircraft as a fallback.
- Returns `None` immediately when `game.current_side_id` is unset.

The launched aircraft is moved from `airbase.aircraft` into `scenario.aircraft`. **Its
coordinates are not changed**, so the aircraft goes airborne exactly where its record says
it is. This project relies on that: each aircraft record carries its own airbase's
coordinates, so every agent launches over its base.

### Two-argument attack

```python
game.handle_aircraft_attack(
    aircraft_id: str,
    target_id: str,
    weapon_id: str | None = None,
    weapon_quantity: int = 2,
) -> None
```

- `weapon_id=None` (the default) selects
  `aircraft.get_weapon_with_highest_engagement_range()`.
- `weapon_quantity` defaults to **2**, which is what makes "one ATTACK step ⇒ target
  destroyed" hold in the current cell.
- Four-argument callers are unaffected.
- The call is a no-op unless the target exists, the aircraft exists, they are on different
  sides, and the target is not the aircraft itself. It also returns immediately if
  `weapon_quantity <= 0`.

```python
# both forms are valid
game.handle_aircraft_attack("f16_01", "airbase_07")                       # auto weapon, qty 2
game.handle_aircraft_attack("f16_01", "airbase_07", "aim120_id", 2)       # explicit
```

---

## 6. `Scenario` — state container and lookups

Constructed with defaults for everything, so `Scenario()` is a valid empty scenario.

**Key attributes**

```python
scenario.id: str
scenario.name: str                       # see the recording note in §9
scenario.start_time: int                 # unix seconds
scenario.current_time: int               # unix seconds
scenario.duration: int
scenario.time_compression: int
scenario.sides: list[Side]
scenario.aircraft: list[Aircraft]        # AIRBORNE aircraft only
scenario.ships: list[Ship]
scenario.facilities: list[Facility]
scenario.airbases: list[Airbase]         # grounded aircraft live in airbase.aircraft
scenario.weapons: list[Weapon]           # weapons IN FLIGHT
scenario.reference_points: list[ReferencePoint]
scenario.missions: list[PatrolMission | StrikeMission]
scenario.relationships: Relationships
scenario.doctrine: Dict[str, SideDoctrine]
```

**Lookups** (all return `None` when not found)

```python
scenario.get_aircraft(aircraft_id) -> Aircraft | None
scenario.get_ship(ship_id) -> Ship | None
scenario.get_facility(facility_id) -> Facility | None
scenario.get_airbase(airbase_id) -> Airbase | None
scenario.get_weapon(weapon_id) -> Weapon | None
scenario.get_reference_point(reference_point_id) -> ReferencePoint | None
scenario.get_target(target_id) -> Aircraft | Facility | Weapon | Airbase | Ship | None
scenario.get_side(side_id) -> Side | None
scenario.get_side_name(side_id) -> str
scenario.get_side_color(side_id) -> SIDE_COLOR
scenario.get_aircraft_homebase(aircraft_id) -> Airbase | Ship | None
scenario.get_closest_base_to_aircraft(aircraft_id) -> Airbase | Ship | None
scenario.get_all_targets_from_enemy_sides(side_id)
scenario.get_strike_mission(mission_id) / get_patrol_mission(mission_id)
scenario.get_all_strike_missions() / get_all_patrol_missions()
```

`get_target` is the general "does this unit still exist?" probe — this project uses
`get_target(id) is None` as its kill-confirmation signal.

**Relationships and doctrine**

```python
scenario.is_hostile(side_id: str, target_id: str) -> bool
scenario.check_side_doctrine(side_id: str, doctrine_type: DoctrineType) -> bool
scenario.get_side_doctrine(side_id) -> SideDoctrine
scenario.update_side_doctrine(side_id, side_doctrine=None) -> None
scenario.remove_side_doctrine(side_id) -> None
scenario.get_default_doctrine() / get_default_side_doctrine()
```

`is_hostile` takes a **side id and a target id** — not two side ids.

`DoctrineType` members (`blade/Doctrine.py`), all string-valued:

```python
DoctrineType.AIRCRAFT_ATTACK_HOSTILE
DoctrineType.AIRCRAFT_CHASE_HOSTILE
DoctrineType.AIRCRAFT_RTB_WHEN_OUT_OF_RANGE
DoctrineType.AIRCRAFT_RTB_WHEN_STRIKE_MISSION_COMPLETE
DoctrineType.SAM_ATTACK_HOSTILE
DoctrineType.SHIP_ATTACK_HOSTILE
```

**Serialization**

```python
scenario.to_dict() -> dict
scenario.toJson() -> str
game.export_scenario() -> dict        # toJson() + camelCase key conversion
```

---

## 7. Unit classes

### `Aircraft` (`blade/units/Aircraft.py`)

```python
Aircraft(id, name, side_id, class_name, latitude, longitude, altitude, heading,
         speed, current_fuel, max_fuel, fuel_rate, range, route=None, selected=False,
         side_color=None, weapons=None, home_base_id='', rtb=False, target_id='',
         desired_route=None)
```

| Attribute | Meaning |
|---|---|
| `speed` | knots |
| `current_fuel`, `max_fuel` | lbs |
| `fuel_rate` | lbs/hour |
| `range` | **detection range in nautical miles** (see §11) |
| `route` | `[[lat, lon], ...]` remaining waypoints |
| `rtb` | return-to-base flag — a **toggle**, see §13 |
| `home_base_id`, `target_id` | ids |

Methods:

```python
aircraft.get_detection_range() -> float                      # returns self.range
aircraft.get_weapon(weapon_id) -> Weapon | None
aircraft.get_weapon_with_highest_engagement_range() -> Weapon | None
aircraft.get_total_weapon_quantity() -> int
aircraft.to_dict() -> dict
```

### `Ship` (`blade/units/Ship.py`)

Same constructor shape as `Aircraft`, plus an `aircraft` inventory list. Same four methods.

### `Facility` (`blade/units/Facility.py`)

```python
Facility(id, name, side_id, class_name, latitude=0.0, longitude=0.0, altitude=0.0,
         range=250.0, side_color=None, weapons=None)
```

Methods: `get_detection_range()`, `get_weapon_with_highest_engagement_range()`,
`get_total_weapon_quantity()`, `to_dict()`. Note it has **no** `get_weapon(weapon_id)`.

### `Airbase` (`blade/units/Airbase.py`)

```python
Airbase(id, name, side_id, class_name, latitude, longitude, altitude,
        side_color=None, aircraft=None)
```

`airbase.aircraft` is the grounded inventory. The only method is `to_dict()` — an airbase
has **no** range, no weapons and no detection method.

### `Weapon` (`blade/units/Weapon.py`)

```python
Weapon(id, name, side_id, class_name, latitude, longitude, altitude, heading, speed,
       current_fuel, max_fuel, fuel_rate, range, route=None, side_color=None,
       target_id=None, lethality=0.0, max_quantity=0, current_quantity=0)
```

```python
weapon.get_engagement_range() -> float     # speed * (current_fuel / fuel_rate), NAUTICAL MILES
weapon.to_dict() -> dict
```

The engagement range is **derived, not stored** — it is how far the weapon can fly on its
remaining fuel, so it is not `weapon.range`.

### `ReferencePoint` (`blade/units/ReferencePoint.py`)

```python
ReferencePoint(id, name, side_id, latitude, longitude, altitude, side_color=None)
```

### `Side` (`blade/Side.py`)

```python
Side(id: str, name: str, total_score: int = 0, color: str | SIDE_COLOR | None = None)
```

Only method: `to_dict()`. This project selects its side by `side.name == "BLUE"`.

---

## 8. Missions

```python
StrikeMission(id, name, side_id, assigned_unit_ids: List[str],
              assigned_target_ids: List[str], active: bool)

PatrolMission(id, name, side_id, assigned_unit_ids: List[str],
              assigned_area: List[ReferencePoint], active: bool)
```

A `PatrolMission` also carries `patrol_area_geometry`. The area attribute is
**`assigned_area`** (a list of `ReferencePoint`), not `patrol_area` and not raw coordinates.

This project does not use BLADE missions: it issues per-tick commands through
`GraphPlanExecutor` instead.

---

## 9. Recording and playback

```python
game.start_recording()
game.record_step(force: bool = False)
game.export_recording()
```

`record_step` writes a frame only when `recorder.should_record(current_time)` is true, or
when `force=True`.

**`game.current_scenario.name` must be set BEFORE `start_recording()`.**
`PlaybackRecorder.start_recording(scenario)` snapshots `scenario.name` at that moment and
uses it for the exported filename. Setting the name afterwards has no effect, and the
recording is written as *"New Scenario"*.

```python
game.current_scenario.name = "my-episode-tag"   # FIRST
game.start_recording()                          # then
```

`PlaybackRecorder` (`blade/utils/PlaybackRecorder.py`):

```python
PlaybackRecorder(record_every_seconds: Optional[int] = None,
                 recording_export_path: Optional[str] = '.')

recorder.start_recording(scenario)
recorder.should_record(current_scenario_time) -> bool
recorder.record_step(current_step: str, current_scenario_time: int)
recorder.export_recording(recording_end_time_unix, recording_start_time_unix=None)
recorder.reset()
```

The recorder buffers frames in memory and **auto-splits** when the buffer exceeds the
module-level `CHARACTER_LIMIT`, exporting a chunk and starting a new one. A long episode
can therefore produce several files; consumers must handle more than one chunk. Exported
files are named `{export_path}/{scenario_name} Recording {start} - {end}.jsonl`.

---

## 10. Utility and engine functions

### `blade/utils/utils.py`

```python
get_distance_between_two_points(start_lat, start_lon, dest_lat, dest_lon) -> float   # KILOMETRES
get_bearing_between_two_points(start_lat, start_lon, dest_lat, dest_lon) -> float    # degrees
get_terminal_coordinates_from_distance_and_bearing(start_lat, start_lon, distance, bearing) -> List[float]
get_next_coordinates(origin_lat, origin_lon, dest_lat, dest_lon, platform_speed) -> List[float]
to_radians(degrees) -> float
to_degrees(radians) -> float
unix_to_local_time(unix_timestamp, separator=':') -> str
to_camelcase(s)
random_float(min_value, max_value) -> float
random_int(min_value, max_value) -> int
```

`get_next_coordinates` takes **origin, destination and speed** — it advances one tick along
the great circle toward the destination. It does *not* take a bearing and a distance; that
is `get_terminal_coordinates_from_distance_and_bearing`.

### `blade/engine/weaponEngagement.py`

```python
is_threat_detected(threat: Aircraft | Weapon, detector: Facility | Ship | Aircraft) -> bool
weapon_can_engage_target(target, weapon: Weapon) -> bool
launch_weapon(scenario, origin, target, launched_weapon, launched_weapon_quantity) -> None
weapon_engagement(scenario, weapon) -> None
weapon_endgame(scenario, weapon, target) -> bool
aircraft_pursuit(scenario, aircraft) -> None
route_aircraft_to_strike_position(scenario, aircraft, target_id, strike_radius_nm) -> None
check_target_tracked_by_count(scenario, target) -> int
```

`is_threat_detected` builds a Shapely buffer of `detector.get_detection_range() / 60`,
treating the detection range as nautical miles converted to degrees — a deliberately rough
approximation. `weapon_can_engage_target` converts the km distance to nautical miles and
compares it against `weapon.get_engagement_range()`.

### `blade/utils/constants.py`

```python
EARTH_RADIUS_KM = 6371
EARTH_RADIUS_M = 6371008.8
KILOMETERS_TO_NAUTICAL_MILES = 0.539957
NAUTICAL_MILES_TO_METERS = 1852
DEFAULT_OL_PROJECTION_CODE = 'EPSG:3857'
GAME_SPEED_DELAY_MS = {1: 1000, 2: 500, 4: 250, 8: 125, 100: 0}
BLADE_ENV_ACTION_SPACE_MAX_CHARACTERS = 100000
BLADE_ENV_OBSERVATION_SPACE_MAX_CHARACTERS = 100000
```

---

## 11. Units of measure

Mixing these up is the most common source of silent errors in this engine.

| Quantity | Unit |
|---|---|
| `get_distance_between_two_points(...)` | **kilometres** |
| `aircraft.range` / `get_detection_range()` | **nautical miles** |
| `weapon.get_engagement_range()` | **nautical miles** |
| `aircraft.speed` | knots |
| `current_fuel`, `max_fuel` | lbs |
| `fuel_rate` | lbs/hour |
| `scenario.current_time`, `start_time` | unix seconds |
| `altitude` | metres |
| bearings/headings | degrees |

Fuel for a distance is therefore
`km → nm (× KILOMETERS_TO_NAUTICAL_MILES) → hours (÷ speed in knots) → lbs (× fuel_rate)`,
which is what `Game.get_fuel_needed_to_return_to_base` computes.

---

## 12. Integration with this project

The **only** layer that turns plans into BLADE commands is:

```
src/match_aou/utils/blade_utils/blade_graph_executor.py   ->  GraphPlanExecutor
```

Supporting modules:

| Module | Role |
|---|---|
| `utils/blade_utils/scenario_factory.py` | BLADE scenario → MATCH-AOU `Agent` / `Task` objects |
| `utils/blade_utils/scenario_generator.py` | seeded scenario variations from the template |
| `rl/training/graph_episode_setup.py` | builds the env (`_build_env`), solves, patches and reloads |
| `rl/training/graph_tick_loop.py` | the two-phase tick; also drives recording |

Typical command strings emitted by the executor:

```python
f"move_aircraft('{ego_id}', [[{lat}, {lon}]])"
f"launch_aircraft_from_airbase('{base_id}', '{ego_id}')"
f"handle_aircraft_attack('{ego_id}', '{target_id}')"
f"aircraft_return_to_base('{ego_id}')"
```

`Game.handle_action` accepts a **list** of such strings, which is how all agents act within
one tick.

---

## 13. Gotchas

- **`from blade import Game` binds a module, not the class.** §1.
- **`aircraft_return_to_base` is a TOGGLE.** Issuing it twice cancels the RTB. This project
  latches it to a single issue per agent — which is only safe because the BLUE side has the
  `AIRCRAFT_RTB_WHEN_OUT_OF_RANGE` doctrine **off** in `strike_training_4v5.json`. With
  that doctrine on, the engine issues its own RTB on bingo fuel and a later
  executor-issued call would toggle it back off.
- **`move_aircraft` clears the route on every call**, so re-issuing an identical move
  restarts the leg. Check the live route before re-issuing.
- **`step()` never terminates on its own**: `reward = 0`, `terminated = False`,
  `check_game_ended()` returns `False`, `info` is `{}`. Termination comes from the
  Gymnasium TimeLimit or from caller logic.
- **`land_aicraft` is spelled that way in the engine** (missing the `r`). It is not a typo
  in this document.
- **Grounded aircraft are not in `scenario.aircraft`.** They live in `airbase.aircraft`
  until launched, so "is this aircraft airborne?" is `scenario.get_aircraft(id) is not None`.
- **Set `current_scenario.name` before `start_recording()`.** §9.
- **`Facility` has no `get_weapon(weapon_id)`** even though `Aircraft` and `Ship` do.
- **Fuel burns every tick for every airborne aircraft**, including ones with no route.

---

**Source of truth:** the vendored tree at
`src/match_aou/integrations/panopticon-main/gym/blade/`. If this document and that code
disagree, the code is correct — please correct the document.
