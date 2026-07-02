"""scenario_factory.py

Converts BLADE Scenario observations into MATCH-AOU model objects (Agents, Tasks).

Responsibilities:
- create_agents_from_scenario: BLADE units → MATCH-AOU Agent objects
- generate_all_enemy_tasks: BLADE enemy units → MATCH-AOU Task objects (one per target)
- iter_enemy_targets: shared enemy-enumeration ((target_id, Location) per enemy unit),
  consumed by both generate_all_enemy_tasks and the executor's sensing exposure.
- make_attack_task: single-ATTACK-step Task for one enemy unit, consumed by both
  generate_all_enemy_tasks (static plan) and the trigger layer's pop-up path.

Design notes:
- When aircraft are stored inside an Airbase, we ensure their Agent.home_base_id is set
  to that airbase id (needed for launch_aircraft_from_airbase planning).
- Uses a more sensible minimum planning speed for aircraft that start on the ground
  with speed=0, to avoid absurd costs/travel times in MATCH-AOU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ...models import Agent, Capability, Location, Step, StepKind, Task

logger = logging.getLogger(__name__)

# Per-target utility keyed by BLADE unit class name. Mirrors the utilities the
# original generate_all_enemy_tasks assigned per collection (facilities 100 /
# airbases 80 / ships 95). The BLADE unit classes are a frozen vendored copy, so
# these class names are stable; type(unit).__name__ keeps this module free of a
# hard import dependency on the engine (it stays duck-typed, like the rest of the file).
_UTILITY_BY_UNIT_TYPE: Dict[str, int] = {"Facility": 100, "Airbase": 80, "Ship": 95}
# Fallback for an unexpected unit type. In practice unreachable: iter_enemy_targets
# only yields the three types above, and the trigger's pop-up path builds tasks only
# from ids that came through iter_enemy_targets. Airbase-tier (the default target
# class when INCLUDE_SAMS is False) is the safe default.
_DEFAULT_TARGET_UTILITY: int = 80


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


def iter_enemy_targets(
    scenario: Any,
    our_side_color: Any,
) -> Iterator[Tuple[str, Location]]:
    """Yield ``(target_id, Location)`` for every enemy target unit in the scenario.

    Enemy = a facility / airbase / ship whose normalized ``side_color`` differs from
    ours (and is non-empty). This is the SINGLE shared enemy-enumeration used by:
      * :func:`generate_all_enemy_tasks` — to build the static task set, and
      * ``GraphPlanExecutor.sensed_target_ids`` — to test which enemies are within the
        ego's own sensor range.

    Yield order is facilities, then airbases, then ships — matching the original
    inlined order in ``generate_all_enemy_tasks`` so ``task_idx`` assignment is
    unchanged. The yielded ``Location`` is what the sensing consumer needs (geometry);
    the task-building consumer re-resolves the unit (see ``generate_all_enemy_tasks``)
    because :func:`make_attack_task` derives per-type utility from the unit itself.

    Args:
        scenario: BLADE Scenario observation object.
        our_side_color: our side color (raw or normalized; normalized here).

    Yields:
        ``(target_id, Location)`` per enemy target, in facilities→airbases→ships order.
    """
    our_side = _normalize_side_color(our_side_color)
    for collection in (
        getattr(scenario, "facilities", None),
        getattr(scenario, "airbases", None),
        getattr(scenario, "ships", None),
    ):
        for unit in collection or []:
            side = _normalize_side_color(getattr(unit, "side_color", ""))
            if side and side != our_side:
                loc = Location(
                    unit.latitude,
                    unit.longitude,
                    getattr(unit, "altitude", 0) or 0,
                )
                yield str(unit.id), loc


def make_attack_task(unit: Any, probability: float = 1.0) -> Task:
    """Build a single-ATTACK-step :class:`Task` for one enemy target unit.

    Module-level twin of the former nested ``_make_task`` closure. Used by BOTH
    :func:`generate_all_enemy_tasks` (statically-known targets) AND the trigger
    layer's pop-up path (a discovered target), so a pop-up Task is byte-for-byte
    the same shape as a statically-known one.

    Utility is derived from the unit's BLADE class name (``Facility`` 100 /
    ``Airbase`` 80 / ``Ship`` 95, per :data:`_UTILITY_BY_UNIT_TYPE`); unknown types
    fall back to :data:`_DEFAULT_TARGET_UTILITY`.

    Note: a FRESH ``attack`` Capability instance is created per task (the original
    shared one instance across all tasks). Content is identical, so solver behavior
    is unchanged; fresh instances additionally avoid cross-task aliasing.

    Args:
        unit: a BLADE facility / airbase / ship (needs ``id`` / ``latitude`` /
            ``longitude`` / optional ``altitude``).
        probability: ATTACK-step success probability (default 1.0).

    Returns:
        A Task with one ATTACK Step targeting ``unit``.
    """
    target_loc = Location(
        unit.latitude,
        unit.longitude,
        getattr(unit, "altitude", 0) or 0,
    )
    utility = _UTILITY_BY_UNIT_TYPE.get(type(unit).__name__, _DEFAULT_TARGET_UTILITY)
    step = Step(
        location=target_loc,
        target_id=str(unit.id),
        capabilities=[Capability(name="attack", properties={"Quantity": 2})],
        probability=probability,
        effort=2,
        step_kind=StepKind.ATTACK,
    )
    return Task(steps=[step], utility=utility)


def generate_all_enemy_tasks(
    scenario: Any,
    attacking_side_color: str,
    probability: float = 1.0,
) -> List[Task]:
    """Generate one attack Task per enemy target (facility / airbase / ship).

    Scans all units in the scenario and creates a Task for each one whose
    side_color differs from ours.

    Now a thin composition of the two shared helpers: :func:`iter_enemy_targets`
    (the enemy enumeration) and :func:`make_attack_task` (per-type task build). The
    enumeration yields only ``(target_id, Location)``, so the unit is re-resolved via
    ``scenario.get_target`` to feed ``make_attack_task`` (frozen BLADE ``get_target``
    is the same liveness probe used elsewhere). Task order / count / content are
    unchanged from the previous inlined form.

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
    for target_id, _target_loc in iter_enemy_targets(scenario, attacking_side_color):
        # Re-resolve the unit: iter_enemy_targets yields only (id, Location) for the
        # sensing consumer; make_attack_task derives per-type utility + geometry from
        # the unit itself. Same unit, same order -> identical task list.
        unit = scenario.get_target(target_id)
        if unit is None:
            continue
        tasks.append(make_attack_task(unit, probability=probability))

    logger.debug(f"Generated {len(tasks)} enemy tasks")

    return tasks


# =============================================================================
# Self-test (hand-built stubs; NO BLADE, NO solver)
# =============================================================================

def _selftest() -> None:
    """Assert the extracted helpers + generate_all_enemy_tasks behavior preservation.

    Run under nlp_env from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.utils.blade_utils.scenario_factory
    """
    # Stub BLADE units. Class NAMES matter: make_attack_task derives utility from
    # type(unit).__name__, so these must be named exactly Facility / Airbase / Ship.
    class Facility:
        def __init__(self, id, lat, lon, side, alt=0):
            self.id, self.latitude, self.longitude = id, lat, lon
            self.side_color, self.altitude = side, alt

    class Airbase:
        def __init__(self, id, lat, lon, side, alt=0):
            self.id, self.latitude, self.longitude = id, lat, lon
            self.side_color, self.altitude = side, alt

    class Ship:
        def __init__(self, id, lat, lon, side, alt=0):
            self.id, self.latitude, self.longitude = id, lat, lon
            self.side_color, self.altitude = side, alt

    class StubScenario:
        def __init__(self, facilities, airbases, ships):
            self.facilities, self.airbases, self.ships = facilities, airbases, ships

        def get_target(self, target_id):
            for u in (self.facilities + self.airbases + self.ships):
                if str(u.id) == str(target_id):
                    return u
            return None

    # blue = ours; red = enemy. A friendly (blue) unit of each type must be excluded.
    facilities = [Facility("f_red", 32.0, 35.0, "red", alt=10),
                  Facility("f_blue", 32.1, 35.1, "blue")]
    airbases = [Airbase("ab_red", 33.0, 36.0, "red")]
    ships = [Ship("s_red", 34.0, 37.0, "red"), Ship("s_blue", 34.1, 37.1, "blue")]
    scenario = StubScenario(facilities, airbases, ships)

    print("=" * 72)
    print("scenario_factory self-test")
    print("=" * 72)

    # (1) iter_enemy_targets: enemy-only, facilities→airbases→ships order, (id, Location).
    enum = list(iter_enemy_targets(scenario, "blue"))
    assert [tid for tid, _ in enum] == ["f_red", "ab_red", "s_red"], enum
    for _tid, loc in enum:
        assert isinstance(loc, Location)
    # Geometry + altitude preserved for the first (facility) entry.
    assert (enum[0][1].latitude, enum[0][1].longitude, enum[0][1].altitude) == (32.0, 35.0, 10)
    print("[1] iter_enemy_targets: enemy-only, ordered, (id, Location) w/ altitude   OK")

    # (2) make_attack_task: utility-by-type, target_id, single ATTACK step, attack cap.
    t_fac = make_attack_task(facilities[0])
    t_air = make_attack_task(airbases[0])
    t_ship = make_attack_task(ships[0])
    assert (t_fac.utility, t_air.utility, t_ship.utility) == (100, 80, 95)
    assert t_fac.steps[0].target_id == "f_red"
    assert t_fac.steps[0].step_kind is StepKind.ATTACK
    assert t_fac.steps[0].probability == 1.0 and t_fac.steps[0].effort == 2
    caps = t_fac.steps[0].capabilities
    assert len(caps) == 1 and caps[0].name == "attack" and caps[0].properties == {"Quantity": 2}
    # Fresh (non-shared) Capability instance per task.
    assert t_air.steps[0].capabilities[0] is not t_fac.steps[0].capabilities[0]
    # probability override is honored.
    assert make_attack_task(facilities[0], probability=0.7).steps[0].probability == 0.7
    print("[2] make_attack_task: utility-by-type, target_id, ATTACK step, fresh cap   OK")

    # (3) generate_all_enemy_tasks behavior preservation: count / order / utilities /
    #     target_ids exactly as the previous inlined implementation would produce.
    tasks = generate_all_enemy_tasks(scenario, "blue")
    assert [t.utility for t in tasks] == [100, 80, 95], [t.utility for t in tasks]
    assert [t.steps[0].target_id for t in tasks] == ["f_red", "ab_red", "s_red"]
    assert all(t.steps[0].step_kind is StepKind.ATTACK for t in tasks)
    # Friendly (blue) units excluded.
    assert "f_blue" not in {t.steps[0].target_id for t in tasks}
    assert "s_blue" not in {t.steps[0].target_id for t in tasks}
    # probability propagates end-to-end.
    tasks_p = generate_all_enemy_tasks(scenario, "blue", probability=0.6)
    assert all(t.steps[0].probability == 0.6 for t in tasks_p)
    print("[3] generate_all_enemy_tasks: order/utilities/target_ids preserved, blue excluded   OK")

    # (4) Empty / no-enemy scenarios degrade gracefully.
    assert list(iter_enemy_targets(StubScenario([], [], []), "blue")) == []
    assert generate_all_enemy_tasks(StubScenario([Facility("f2", 1, 1, "blue")], [], []), "blue") == []
    print("[4] empty / all-friendly scenario -> no tasks   OK")

    # (5) Unknown unit type falls back to the default utility.
    class Radar:
        def __init__(self):
            self.id, self.latitude, self.longitude, self.altitude = "rdr", 1, 2, 0
    assert make_attack_task(Radar()).utility == _DEFAULT_TARGET_UTILITY
    print("[5] unknown unit type -> default utility   OK")

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()