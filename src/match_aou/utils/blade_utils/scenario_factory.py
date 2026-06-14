"""scenario_factory.py

Converts BLADE Scenario observations into MATCH-AOU model objects (Agents, Tasks).

Responsibilities:
- create_agents_from_scenario: BLADE units → MATCH-AOU Agent objects
- generate_all_enemy_tasks: BLADE enemy units → MATCH-AOU Task objects (one per target)

Design notes:
- When aircraft are stored inside an Airbase, we ensure their Agent.home_base_id is set
  to that airbase id (needed for launch_aircraft_from_airbase planning).
- Uses a more sensible minimum planning speed for aircraft that start on the ground
  with speed=0, to avoid absurd costs/travel times in MATCH-AOU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...models import Agent, Capability, Location, Step, StepType, Task

logger = logging.getLogger(__name__)


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


def generate_all_enemy_tasks(
    scenario: Any,
    attacking_side_color: str,
    probability: float = 1.0,
) -> List[Task]:
    """Generate one attack Task per enemy target (facility / airbase / ship).

    Scans all units in the scenario and creates a Task for each one whose
    side_color differs from ours.

    Args:
        scenario: BLADE Scenario observation object
        attacking_side_color: Our side color (e.g. "blue")
        probability: Success probability per attack step. Use 1.0 for targets
            destroyed in a single hit (e.g. airbases). Use < 1.0 for defended
            targets where multi-agent redundancy may be beneficial.

    Returns:
        List of Task objects, one per enemy target
    """
    tasks: List[Task] = []
    our_side = _normalize_side_color(attacking_side_color)

    attack_capability = Capability(name="attack", properties={"Quantity": 2})
    attack_step_type = StepType(name="attack", base_cost=1)

    def _make_task(unit: Any, utility: float) -> Task:
        target_loc = Location(
            unit.latitude,
            unit.longitude,
            getattr(unit, "altitude", 0) or 0,
        )
        step = Step(
            location=target_loc,
            capabilities=[attack_capability],
            step_type=attack_step_type,
            effort=2,
            probability=probability,
            action=f"handle_aircraft_attack('AGENT_ID', '{unit.id}', 'WEAPON_ID', 2)",
        )
        return Task(steps=[step], utility=utility)

    # Enemy facilities
    for facility in getattr(scenario, "facilities", []) or []:
        side = _normalize_side_color(getattr(facility, "side_color", ""))
        if side and side != our_side:
            tasks.append(_make_task(facility, utility=100))

    # Enemy airbases
    for airbase in getattr(scenario, "airbases", []) or []:
        side = _normalize_side_color(getattr(airbase, "side_color", ""))
        if side and side != our_side:
            tasks.append(_make_task(airbase, utility=80))

    # Enemy ships
    for ship in getattr(scenario, "ships", []) or []:
        side = _normalize_side_color(getattr(ship, "side_color", ""))
        if side and side != our_side:
            tasks.append(_make_task(ship, utility=95))

    logger.debug(f"Generated {len(tasks)} enemy tasks")

    return tasks