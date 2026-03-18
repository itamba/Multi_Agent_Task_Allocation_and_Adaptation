# BLADE API Reference
================================================================================

**BLADE** (Panopticon) is a military simulation engine for multi-agent scenarios.
This document provides a comprehensive API reference for working with BLADE.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Classes](#core-classes)
3. [Game Class - Main API](#game-class---main-api)
4. [Scenario Management](#scenario-management)
5. [Unit Classes](#unit-classes)
6. [Mission Types](#mission-types)
7. [Gym Environment](#gym-environment)
8. [Common Workflows](#common-workflows)
9. [Utility Functions](#utility-functions)

---

## Quick Start

```python
from blade import Game, Scenario

# Load a scenario
scenario = Scenario.load_from_file('scenario.json')

# Create game instance
game = Game(scenario)

# Step the simulation
observation, reward, terminated, truncated, info = game.step(action)

# Access units
aircraft = game.current_scenario.get_aircraft(aircraft_id)
```

---

## Core Classes

### Game
**Location:** `blade/Game.py`

Main simulation controller. Manages the scenario, handles actions, and updates game state.

**Key Responsibilities:**
- Simulation stepping (time advancement)
- Action handling (move, attack, RTB)
- Unit management
- Auto-defense and AI behaviors

### Scenario
**Location:** `blade/Scenario.py`

Container for all scenario data - units, sides, missions, time.

**Key Properties:**
- `aircraft: List[Aircraft]` - All aircraft in scenario
- `ships: List[Ship]` - All ships
- `facilities: List[Facility]` - All facilities (SAMs, bases)
- `airbases: List[Airbase]` - All airbases
- `sides: List[Side]` - Factions/teams
- `missions: List[Mission]` - Active missions
- `current_time: int` - Simulation time (seconds)

### Side
**Location:** `blade/Side.py`

Represents a faction/team in the simulation.

**Properties:**
- `id: str` - Unique identifier
- `name: str` - Display name
- `color: str` - Team color
- `total_score: int` - Side's score

---

## Game Class - Main API

### Initialization

```python
game = Game(
    current_scenario: Scenario,
    record_every_seconds: Optional[int] = None,
    recording_export_path: Optional[str] = '.'
)
```

### Simulation Control

#### `step(action) -> Tuple[Scenario, float, bool, bool, dict]`
Advance simulation by one time step.

**Returns:**
- `observation` (Scenario): Current state
- `reward` (float): Reward value
- `terminated` (bool): Episode ended normally
- `truncated` (bool): Episode cut short
- `info` (dict): Additional information

```python
obs, reward, terminated, truncated, info = game.step(action)
```

#### `reset() -> None`
Reset simulation to initial state.

```python
game.reset()
```

#### `update_game_state() -> None`
Update all units, weapons, missions. Called internally by `step()`.

### Aircraft Control

#### `move_aircraft(aircraft_id: str, new_coordinates: list) -> Aircraft`
Set aircraft route to waypoints.

**Parameters:**
- `aircraft_id`: Aircraft unique ID
- `new_coordinates`: List of [lat, lon] waypoints

```python
aircraft = game.move_aircraft(
    'aircraft_1',
    [[35.0, 40.0], [35.5, 40.5]]  # waypoints
)
```

#### `handle_aircraft_attack(aircraft_id, target_id, weapon_id, weapon_quantity)`
Launch weapons from aircraft at target.

**Parameters:**
- `aircraft_id`: Attacking aircraft ID
- `target_id`: Target unit ID
- `weapon_id`: Weapon type ID
- `weapon_quantity`: Number of weapons to launch

```python
game.handle_aircraft_attack(
    aircraft_id='f16_01',
    target_id='sam_site_03',
    weapon_id='agm_88',
    weapon_quantity=2
)
```

#### `aircraft_return_to_base(aircraft_id: str) -> Aircraft`
Command aircraft to return to base (RTB).

```python
aircraft = game.aircraft_return_to_base('f16_01')
```

### Ship Control

#### `move_ship(ship_id: str, new_coordinates: list) -> Ship`
Set ship route to waypoints.

#### `handle_ship_attack(ship_id, target_id, weapon_id, weapon_quantity)`
Launch weapons from ship at target.

### Mission Management

#### `add_strike_mission(...) -> StrikeMission`
Create a new strike mission.

**Parameters:**
- `side_id`: Owning side
- `mission_name`: Mission name
- `assigned_attackers`: List of attacker IDs
- `assigned_targets`: List of target IDs

#### `delete_mission(mission_id: str)`
Remove a mission.

---

## Scenario Management

### Loading Scenarios

```python
# From JSON file
scenario = Scenario.load_from_file('my_scenario.json')

# From JSON string
scenario = Scenario.from_json(json_string)
```

### Accessing Units

```python
# Get specific unit
aircraft = scenario.get_aircraft(aircraft_id)
ship = scenario.get_ship(ship_id)
facility = scenario.get_facility(facility_id)

# Get any target (aircraft, ship, or facility)
target = scenario.get_target(target_id)

# Get all units of a side
side_aircraft = [a for a in scenario.aircraft if a.side_id == side_id]
```

### Checking Relationships

```python
# Check if two sides are hostile
is_hostile = scenario.is_hostile(side_id_1, side_id_2)

# Check doctrine settings
auto_engage = scenario.check_side_doctrine(side_id, DoctrineType.ATTACK_HOSTILE)
```

---

## Unit Classes

### Aircraft
**Location:** `blade/units/Aircraft.py`

**Key Properties:**
```python
aircraft.id: str                    # Unique identifier
aircraft.name: str                  # Display name
aircraft.side_id: str               # Owning side
aircraft.class_name: str            # Unit type (F-16C, etc.)
aircraft.latitude: float            # Position
aircraft.longitude: float
aircraft.altitude: float            # Meters
aircraft.heading: float             # Degrees (0-360)
aircraft.speed: float               # Knots
aircraft.current_fuel: float        # lbs
aircraft.max_fuel: float
aircraft.fuel_rate: float           # lbs/hr
aircraft.range: float               # Detection range (meters)
aircraft.route: List[List[float]]   # Waypoints [[lat, lon], ...]
aircraft.weapons: List[Weapon]      # Available weapons
aircraft.home_base_id: str          # Home airbase ID
aircraft.rtb: bool                  # Return-to-base flag
aircraft.target_id: str             # Current target
```

**Key Methods:**
```python
aircraft.get_weapon(weapon_id) -> Weapon  # Get specific weapon
aircraft.to_dict() -> dict                # Serialize to dict
```

### Ship
**Location:** `blade/units/Ship.py`

Similar to Aircraft, but for naval units.

**Key Properties:**
```python
ship.id: str
ship.name: str
ship.side_id: str
ship.latitude: float
ship.longitude: float
ship.heading: float
ship.speed: float
ship.route: List[List[float]]
ship.weapons: List[Weapon]
```

### Facility
**Location:** `blade/units/Facility.py`

Static ground units (SAM sites, command centers, etc.)

**Key Properties:**
```python
facility.id: str
facility.name: str
facility.side_id: str
facility.latitude: float
facility.longitude: float
facility.range: float               # Detection/engagement range
facility.weapons: List[Weapon]
```

**Key Methods:**
```python
facility.get_weapon_with_highest_engagement_range() -> Weapon
```

### Airbase
**Location:** `blade/units/Airbase.py`

Base that can launch/recover aircraft.

**Key Properties:**
```python
airbase.id: str
airbase.name: str
airbase.side_id: str
airbase.latitude: float
airbase.longitude: float
airbase.aircraft: List[Aircraft]    # Aircraft on ground
```

### Weapon
**Location:** `blade/units/Weapon.py`

Represents weapons (missiles, bombs) both on units and in flight.

**Key Properties:**
```python
weapon.id: str
weapon.name: str
weapon.side_id: str
weapon.latitude: float
weapon.longitude: float
weapon.speed: float
weapon.range: float
weapon.target_id: str               # What it's targeting
weapon.lethality: float             # Damage potential
weapon.current_quantity: int        # How many on unit
weapon.max_quantity: int
```

---

## Mission Types

### StrikeMission
**Location:** `blade/mission/StrikeMission.py`

Attack mission against specific targets.

**Properties:**
```python
mission.id: str
mission.name: str
mission.side_id: str
mission.assigned_unit_ids: List[str]    # Attackers
mission.assigned_target_ids: List[str]  # Targets
```

### PatrolMission
**Location:** `blade/mission/PatrolMission.py`

Patrol/CAP mission in an area.

**Properties:**
```python
mission.id: str
mission.name: str
mission.side_id: str
mission.assigned_unit_ids: List[str]
mission.patrol_area: List[List[float]]  # Area waypoints
```

---

## Gym Environment

**Location:** `blade/envs/blade.py`

Gymnasium-compatible wrapper for BLADE.

```python
import gymnasium as gym
from blade.envs.blade import BladeEnv

# Create environment
env = BladeEnv(scenario_file='scenario.json')

# Standard gym loop
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

**Note:** You'll need to define observation and action spaces based on your use case.

---

## Common Workflows

### 1. Execute a Pre-Planned Mission

```python
# Load scenario with planned routes
scenario = Scenario.load_from_file('planned_scenario.json')
game = Game(scenario)

# Step simulation until missions complete
for step in range(1000):
    obs, reward, done, truncated, info = game.step(None)
    if done or truncated:
        break
```

### 2. Manual Control - Move and Attack

```python
# Move aircraft to position
waypoints = [[35.5, 40.2], [35.8, 40.5]]
game.move_aircraft('f16_01', waypoints)

# Attack target when in range
game.handle_aircraft_attack(
    aircraft_id='f16_01',
    target_id='sam_05',
    weapon_id='agm_88_harm',
    weapon_quantity=2
)

# Return to base
game.aircraft_return_to_base('f16_01')

# Step simulation
for _ in range(100):
    game.step(None)
```

### 3. Monitor Unit Status

```python
aircraft = game.current_scenario.get_aircraft('f16_01')

print(f"Position: ({aircraft.latitude}, {aircraft.longitude})")
print(f"Fuel: {aircraft.current_fuel}/{aircraft.max_fuel} lbs")
print(f"Speed: {aircraft.speed} knots")
print(f"Heading: {aircraft.heading}°")
print(f"RTB: {aircraft.rtb}")

# Check weapons
for weapon in aircraft.weapons:
    print(f"{weapon.name}: {weapon.current_quantity}/{weapon.max_quantity}")
```

### 4. Get All Units of a Side

```python
side_id = 'blue_team'

# Get all aircraft
aircraft_list = [
    a for a in game.current_scenario.aircraft
    if a.side_id == side_id
]

# Get all facilities
facilities = [
    f for f in game.current_scenario.facilities
    if f.side_id == side_id
]

print(f"Side {side_id} has:")
print(f"  - {len(aircraft_list)} aircraft")
print(f"  - {len(facilities)} facilities")
```

---

## Utility Functions

**Location:** `blade/utils/utils.py`

### Coordinate/Distance Functions

```python
from blade.utils.utils import (
    get_distance_between_two_points,
    get_bearing_between_two_points,
    get_next_coordinates
)

# Calculate distance (km)
distance_km = get_distance_between_two_points(
    lat1, lon1, lat2, lon2
)

# Calculate bearing (degrees)
bearing = get_bearing_between_two_points(
    lat1, lon1, lat2, lon2
)

# Get next position given bearing and distance
next_lat, next_lon = get_next_coordinates(
    current_lat, current_lon, bearing, distance_km
)
```

### Weapon Engagement Functions

**Location:** `blade/engine/weaponEngagement.py`

```python
from blade.engine.weaponEngagement import (
    is_threat_detected,
    weapon_can_engage_target,
    launch_weapon
)

# Check if unit can detect threat
detected = is_threat_detected(target, sensor_unit)

# Check if weapon can engage target
can_engage = weapon_can_engage_target(target, weapon)
```

### Constants

**Location:** `blade/utils/constants.py`

```python
from blade.utils.constants import NAUTICAL_MILES_TO_METERS

distance_meters = distance_nm * NAUTICAL_MILES_TO_METERS
```

---

## Integration with MATCH-AOU

### Converting MATCH-AOU Plans to BLADE Actions

See `src/match_aou/utils/blade_utils/` for integration utilities:

- **`blade_plan_utils.py`** - Convert MATCH-AOU steps to BLADE actions
- **`blade_executor_minimal.py`** - Execute plans in BLADE
- **`observation_utils.py`** - Extract observations for RL
- **`scenario_factory.py`** - Create BLADE scenarios from MATCH-AOU data

```python
from match_aou.utils.blade_utils.blade_executor_minimal import execute_plan
from match_aou.utils.blade_utils.blade_plan_utils import convert_step_to_action

# Convert MATCH-AOU plan to BLADE actions
for step in agent_plan:
    action = convert_step_to_action(step)
    game.step(action)
```

---

## Tips & Best Practices

### 1. Time Management
- `scenario.current_time` is in **seconds**
- Each `step()` advances time based on `time_compression`
- Use `scenario.duration` to set mission length

### 2. Fuel Management
- Monitor `aircraft.current_fuel`
- Use `game.get_fuel_needed_to_return_to_base(aircraft)` to check RTB feasibility
- Aircraft automatically consume fuel based on `fuel_rate`

### 3. Weapon Ranges
- Check `weapon.range` before launching
- Distance calculations are in **kilometers** or **meters** (check function)
- Use `get_distance_between_two_points()` for range checks

### 4. Side Management
- Always check `scenario.is_hostile(side1, side2)` before attacking
- Use `scenario.sides` to iterate through all teams

### 5. Performance
- Large scenarios (100+ units) can be slow
- Consider limiting detection ranges for better performance
- Use `time_compression` to speed up simulation

---

## Quick Reference Cheat Sheet

```python
# === INITIALIZATION ===
from blade import Game, Scenario
scenario = Scenario.load_from_file('scenario.json')
game = Game(scenario)

# === UNIT ACCESS ===
aircraft = game.current_scenario.get_aircraft(id)
ship = game.current_scenario.get_ship(id)
facility = game.current_scenario.get_facility(id)
target = game.current_scenario.get_target(id)  # Any unit

# === ACTIONS ===
game.move_aircraft(id, [[lat, lon], ...])
game.handle_aircraft_attack(aircraft_id, target_id, weapon_id, qty)
game.aircraft_return_to_base(id)

# === SIMULATION ===
obs, reward, done, trunc, info = game.step(action)
game.reset()

# === QUERIES ===
is_hostile = scenario.is_hostile(side1, side2)
distance_km = get_distance_between_two_points(lat1, lon1, lat2, lon2)
bearing = get_bearing_between_two_points(lat1, lon1, lat2, lon2)
```

---

**Last Updated:** February 2026  
**BLADE Version:** Panopticon fork  
**For Questions:** See `src/match_aou/integrations/panopticon-main/gym/README.md`