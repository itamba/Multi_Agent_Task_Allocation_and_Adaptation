"""scenario_factory.py

Same as scenario_factory_fixed.py plus:
- Better side_color normalization (string vs enum).
- When aircraft are stored inside an Airbase, we ensure their Agent.home_base_id is set
  to that airbase id (needed for launch_aircraft_from_airbase planning).
- Uses a more sensible minimum planning speed for aircraft that start on the ground
  with speed=0, to avoid absurd costs/travel times in MATCH-AOU.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...models import Agent, Capability, Location, Step, StepType, Task


def _normalize_side_color(side_color: Any) -> str:
    if side_color is None:
        return "unknown"
    try:
        val = getattr(side_color, "value", side_color)
    except Exception:
        val = side_color
    return str(val).lower()


def create_agents_from_scenario(scenario: Any) -> Dict[str, List[Agent]]:
    """Convert a Scenario observation to `Agent` objects grouped by side_color.

    Includes units from:
    - scenario.aircraft
    - scenario.ships
    - aircraft stored inside scenario.airbases (if present)

    Returns:
        Dict[str, List[Agent]]: mapping of normalized side_color -> list of Agents
    """

    agents_by_side: Dict[str, List[Agent]] = {}

    def convert_unit_to_agent(unit: Any, *, home_base_id_override: Optional[str] = None) -> Agent:
        # If a unit starts on the ground (speed=0 at reset), planning still needs a
        # reasonable cruise-speed estimate. This is demo-only.
        MIN_SPEED_KTS = 250.0

        location = Location(unit.latitude, unit.longitude, getattr(unit, "altitude", 0))

        capabilities: List[Capability] = []
        for w in getattr(unit, "weapons", []) or []:
            weapon_name = str(getattr(w, "class_name", "weapon"))
            cap = Capability(
                name="attack",
                properties={weapon_name: int(getattr(w, "current_quantity", 0) or 0)},
            )
            capabilities.append(cap)

        budget = float(getattr(unit, "current_fuel", 0) or 0)

        raw_speed = float(getattr(unit, "speed", 0) or 0)
        speed_knots = raw_speed if raw_speed > 1e-6 else MIN_SPEED_KTS

        agent_id = getattr(unit, "id", None)
        side_color = getattr(unit, "side_color", None)

        weapon_id = None
        if hasattr(unit, "get_weapon_with_highest_engagement_range"):
            best = unit.get_weapon_with_highest_engagement_range()
            weapon_id = getattr(best, "id", None)

        home_base_id = home_base_id_override or getattr(unit, "home_base_id", None)
        target_id = getattr(unit, "target_id", None)

        def move_cost_function(src: Any, dest: Any) -> float:
            if not isinstance(src, Location):
                src = Location(*src)
            if not isinstance(dest, Location):
                dest = Location(*dest)

            dist_km = src.distance_to(dest)
            speed_kmh = (speed_knots * 1.852)
            fuel_rate = float(getattr(unit, "fuel_rate", 0) or 0)

            time_hr = dist_km / speed_kmh if speed_kmh > 0 else 0.0
            return float(time_hr * fuel_rate)

        # Optional return location (home base coordinates)
        return_location = None
        home_base = None
        if home_base_id:
            if hasattr(scenario, "get_airbase"):
                home_base = scenario.get_airbase(home_base_id)
            if not home_base and hasattr(scenario, "get_ship"):
                home_base = scenario.get_ship(home_base_id)

        if home_base:
            return_location = Location(
                home_base.latitude,
                home_base.longitude,
                getattr(home_base, "altitude", 0),
            )

        return Agent(
            location=location,
            capabilities=capabilities,
            budget=budget,
            move_cost_function=move_cost_function,
            speed=speed_knots,
            return_location=return_location,
            agent_id=agent_id,
            side_color=side_color,
            weapon_id=weapon_id,
            home_base_id=home_base_id,
            target_id=target_id,
        )

    def add_agent(agent: Agent) -> None:
        side_key = _normalize_side_color(getattr(agent, "side_color", None))
        agents_by_side.setdefault(str(side_key), []).append(agent)

    for ac in getattr(scenario, "aircraft", []) or []:
        add_agent(convert_unit_to_agent(ac))

    for ship in getattr(scenario, "ships", []) or []:
        add_agent(convert_unit_to_agent(ship))

    # Aircraft stored in Airbases: ensure home_base_id is set to that airbase
    for base in getattr(scenario, "airbases", []) or []:
        base_id = getattr(base, "id", None)
        for ac in getattr(base, "aircraft", []) or []:
            add_agent(convert_unit_to_agent(ac, home_base_id_override=base_id))

    return agents_by_side


def generate_attack_base_task(
    scenario: Any,
    attacking_agent_side_color: str,
    agent_id_placeholder: str = "AGENT_ID",
) -> Optional[Task]:
    enemy_facilities = [
        facility
        for facility in (getattr(scenario, "facilities", []) or [])
        if _normalize_side_color(getattr(facility, "side_color", None))
        != _normalize_side_color(attacking_agent_side_color)
    ]
    if not enemy_facilities:
        return None

    target_facility = enemy_facilities[0]

    attack_capability = Capability(name="attack", properties={"Quantity": 2})
    attack_step_type = StepType(name="attack", base_cost=1)

    target_lat = target_facility.latitude - 2
    target_lon = target_facility.longitude - 4
    target_alt = getattr(target_facility, "altitude", 0) or 0

    step_attack = Step(
        location=Location(target_lat, target_lon, target_alt),
        capabilities=[attack_capability],
        step_type=attack_step_type,
        effort=2,
        probability=0.6,
        action=f"handle_aircraft_attack('{agent_id_placeholder}', '{target_facility.id}', 'WEAPON_ID', 2)",
    )

    return Task(steps=[step_attack], utility=100)


def generate_attack_ship_task(
    scenario: Any,
    attacking_agent_side_color: str,
    agent_id_placeholder: str = "AGENT_ID",
) -> Optional[Task]:
    enemy_ships = [
        ship
        for ship in (getattr(scenario, "ships", []) or [])
        if _normalize_side_color(getattr(ship, "side_color", None))
        != _normalize_side_color(attacking_agent_side_color)
    ]
    if not enemy_ships:
        return None

    target_ship = enemy_ships[0]

    attack_capability = Capability(name="attack", properties={"Quantity": 2})
    attack_step_type = StepType(name="attack", base_cost=1)

    target_lat = target_ship.latitude - 3.6
    target_lon = target_ship.longitude + 2.5
    target_alt = 10000

    step_attack = Step(
        location=Location(target_lat, target_lon, target_alt),
        capabilities=[attack_capability],
        step_type=attack_step_type,
        effort=2,
        probability=0.6,
        action=f"handle_aircraft_attack('{agent_id_placeholder}', '{target_ship.id}', 'WEAPON_ID', 2)",
    )

    return Task(steps=[step_attack], utility=95)
