"""
Unit tests for FD-BASELINE-v1 -- the deterministic, ego-local fuel-damage difficulty.

SOLVER-FREE AND ENGINE-FREE BY CONSTRUCTION. Nothing here calls bonmin, builds a BLADE
`Game`, or resets a gymnasium env: the fuel-damage component is pure arithmetic over
duck-typed objects, so the whole factor -- the rng domain, the window, the mutation, the
trigger, the tick-loop ordering and the trainer's accounting -- can be proven with plain
stubs. The one place a real engine is unavoidable (a live episode end to end) is not
proven here and is not claimed to be.

The three proof obligations of the task, and where each is discharged:

PO1  DETERMINISTIC PHYSICAL EVENT AND ACCOUNTING
     P1.1  the derived rng domain is a pure, stable function of the episode seed, and is
           unreachable from global `random` / torch state;
     P1.2  the same seed and mode yield the same condition, the same selected ego and a
           field-identical plan, no matter what consumed RNG first;
     P1.3  forced-damaged selects the SAME ego the seeded mixture would -- what makes a
           matched pair comparable;
     P1.4  the window is BLADE's own arithmetic (`Game.get_fuel_needed_to_return_to_base`
           transcribed), the midpoint really lies inside `[floor, requirement)`, and the
           whole window is strict;
     P1.5  the live aircraft's `current_fuel` is mutated exactly ONCE, to the planned
           value, at the first tick past the threshold -- never re-applied;
     P1.6  a damaged episode with no valid strict window RAISES and lands as an accounted
           `setup` failure, and is never downgraded to a clean episode;
     P1.7  a failure record carries its scheduled condition, so failed counts by
           condition are answerable.

PO2  LOCALITY AND SAME-SNAPSHOT NO-COMMUNICATION
     P2.1  the event is applied BEFORE the per-ego Phase-1 loop -- every ego, the first
           one included, sees the post-damage fuel;
     P2.2  exactly one ego receives `fuel_damage=True`; every peer receives False;
     P2.3  Phase-1 ego ITERATION ORDER cannot change what any ego observed;
     P2.4  the damaged ego's graph at that same wake carries the POST-damage `fuel_norm`
           while every peer row stays featureless (0.0);
     P2.5  a clean controller touches no aircraft and wakes nobody.

PO3  BEHAVIOURAL AND MEASUREMENT INTEGRATION
     P3.1  a controlled SELF_PRESERVATION_ABORT empties only the damaged ego's plan and
           produces exactly ONE existing RTB command (the executor's latch, unchanged);
     P3.2  `RewardConfig(aircraft_penalty_coeff=2.25)` really reaches
           `compute_episode_reward` from BOTH harnesses;
     P3.3  matched eval members reuse the identical seed while taking disjoint scenario
           tags, and the paired delta is computed only over complete pairs and always
           reported with its own denominator.

Run: python -m pytest tests/test_graph_fuel_damage.py -v
     python tests/test_graph_fuel_damage.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import torch

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so match_aou.* imports resolve

from match_aou.models.agent import Agent  # noqa: E402
from match_aou.models.location import Location  # noqa: E402
from match_aou.models.step import Step, StepKind  # noqa: E402
from match_aou.models.task import Task  # noqa: E402
from match_aou.rl.action.graph_action import MetaAction  # noqa: E402
from match_aou.rl.action.graph_effect import apply_meta_action  # noqa: E402
from match_aou.rl.action.graph_trigger import (  # noqa: E402
    NO_TASK_INDEX,
    TriggerKind,
    decide_triggers,
)
from match_aou.rl.observation.graph_builder import (  # noqa: E402
    GraphObservationConfig,
    build_graph_observation,
)
from match_aou.rl.training import graph_tick_loop, graph_train  # noqa: E402
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    CONDITION_CLEAN,
    CONDITION_DAMAGED,
    NAUTICAL_MILES_TO_METERS,
    FuelDamageController,
    FuelDamageError,
    FuelDamageMode,
    FuelDamageParameters,
    FuelDamagePlan,
    build_fuel_damage_controller,
    build_fuel_damage_plan,
    derive_fuel_damage_seed,
    fuel_for_distance_km,
    interpolate_great_circle,
    plan_fuel_damage,
    resolve_condition,
)
from match_aou.rl.training.graph_rollout import RolloutConfig  # noqa: E402
from match_aou.rl.training.graph_train import (  # noqa: E402
    EpisodeAttemptError,
    TrainConfig,
    eval_member_tag,
)
from match_aou.utils.blade_utils.blade_graph_executor import (  # noqa: E402
    GraphPlanExecutor,
)


# =============================================================================
# Stubs -- the duck-typed shapes the component and its consumers actually read
# =============================================================================

_BLUE_SIDE = "side-blue"
_RED_SIDE = "side-red"
_BASE = Location(32.85416264197241, 35.3124013096915)   # the real template's BLUE base


def _point_at(origin: Location, distance_km: float, bearing_deg: float) -> Location:
    """A point ``distance_km`` from ``origin`` on ``bearing_deg`` (great circle)."""
    radius = 6371.0088
    brg = math.radians(bearing_deg)
    lat1 = math.radians(origin.latitude)
    lon1 = math.radians(origin.longitude)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_km / radius)
        + math.cos(lat1) * math.sin(distance_km / radius) * math.cos(brg)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(distance_km / radius) * math.cos(lat1),
        math.cos(distance_km / radius) - math.sin(lat1) * math.sin(lat2),
    )
    return Location(math.degrees(lat2), math.degrees(lon2))


class _StubAircraft:
    """The BLADE ``Aircraft`` fields this feature and the graph builder touch."""

    def __init__(self, aid, lat, lon, *, speed=1303.0, fuel=12000.0, rate=6700.0,
                 home_base_id="base-blue", side_id=_BLUE_SIDE, side_color="blue"):
        self.id = aid
        self.name = "AC %s" % aid
        self.side_id = side_id
        self.side_color = side_color
        self.class_name = "F-16 Fighting Falcon"
        self.latitude = lat
        self.longitude = lon
        self.altitude = 10000
        self.heading = 90.0
        self.speed = speed
        self.current_fuel = fuel
        self.max_fuel = fuel
        self.fuel_rate = rate
        self.range = 100
        self.weapons = []
        self.home_base_id = home_base_id
        self.target_id = None
        self.route = []
        self.rtb = False

    def get_weapon_with_highest_engagement_range(self):
        return None


class _StubAirbase:
    def __init__(self, bid, lat, lon, *, side_id=_BLUE_SIDE, side_color="blue",
                 aircraft=None, name=None):
        self.id = bid
        self.name = name or ("Base %s" % bid)
        self.side_id = side_id
        self.side_color = side_color
        self.latitude = lat
        self.longitude = lon
        self.altitude = 0
        self.aircraft = list(aircraft or [])


class _StubScenario:
    """A minimal live scenario: the lookups the component and the builder call."""

    def __init__(self, aircraft=None, airbases=None):
        self.aircraft = list(aircraft or [])
        self.airbases = list(airbases or [])
        self.ships = []
        self.facilities = []
        self.current_time = 0

    def get_aircraft(self, aircraft_id):
        return next(
            (a for a in self.aircraft if str(a.id) == str(aircraft_id)), None
        )

    def get_airbase(self, airbase_id):
        return next(
            (b for b in self.airbases if str(b.id) == str(airbase_id)), None
        )

    def get_ship(self, _ship_id):
        return None

    def get_target(self, target_id):
        for unit in self.airbases + self.facilities + self.ships:
            if str(unit.id) == str(target_id):
                return unit
        return None


class _StubBelief:
    def __init__(self, tasks, solution=None):
        self.tasks = list(tasks)
        self.solution = dict(solution or {})


class _StubExecutor:
    def __init__(self, agent_ids, *, done=None, rtb=None):
        self.dead = set()
        self.done = set(done or set())
        self.rtb_issued = dict(rtb or {})
        self.arrival_threshold_km = 50.0
        self.agent_ids = list(agent_ids)

    def sensed_target_ids(self, _observation, _ego_id):
        return {}

    def next_actions(self, _observation):
        return []

    def resync(self, *_a, **_k):
        return None

    def is_done(self):
        return False


class _StubEnv:
    """Advances every airborne aircraft toward its first target by a fixed step."""

    def __init__(self, scenario, *, step_km=10.0, targets=None):
        self.scenario = scenario
        self.step_km = float(step_km)
        self.targets = dict(targets or {})
        self.closed = False
        self.n_steps = 0

    def step(self, _action):
        self.n_steps += 1
        for aircraft in self.scenario.aircraft:
            target = self.targets.get(str(aircraft.id))
            if target is None:
                continue
            here = Location(aircraft.latitude, aircraft.longitude)
            remaining = here.distance_to(target)
            if remaining <= self.step_km:
                aircraft.latitude, aircraft.longitude = target.latitude, target.longitude
                continue
            bearing = _bearing(here, target)
            moved = _point_at(here, self.step_km, bearing)
            aircraft.latitude, aircraft.longitude = moved.latitude, moved.longitude
        return self.scenario, 0.0, False, False, {}

    def close(self):
        self.closed = True


def _bearing(a: Location, b: Location) -> float:
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlon = math.radians(b.longitude - a.longitude)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _attack_task(target_id: str, loc: Location, utility: float = 80.0) -> Task:
    return Task(
        steps=[Step(loc, target_id, [], 1.0, 1, StepKind.ATTACK)], utility=utility
    )


class _StubGame:
    def __init__(self, scenario):
        self.current_scenario = scenario


class _FuelDamageCtx:
    """The six `EpisodeContext` attributes the fuel-damage adapter reads, plus a
    tick-loop-compatible surface so ONE fixture serves the plan tests and the loop tests.
    """

    def __init__(self, *, n_agents=3, target_distance_km=250.0, fuel=12000.0,
                 speed=1303.0, rate=6700.0):
        self.n_agents = n_agents
        self.agent_ids = ["ego%d" % i for i in range(n_agents)]
        base = _StubAirbase("base-blue", _BASE.latitude, _BASE.longitude)
        aircraft = [
            _StubAircraft(aid, _BASE.latitude, _BASE.longitude,
                          speed=speed, fuel=fuel, rate=rate)
            for aid in self.agent_ids
        ]
        # Airborne over the base: launch point == the BLUE airbase (CLAUDE.md section 3).
        self.scenario = _StubScenario(aircraft=aircraft, airbases=[base])
        self.targets = {
            aid: _point_at(_BASE, target_distance_km, 30.0 + 40.0 * i)
            for i, aid in enumerate(self.agent_ids)
        }
        for i, aid in enumerate(self.agent_ids):
            loc = self.targets[aid]
            self.scenario.airbases.append(
                _StubAirbase("tgt%d" % i, loc.latitude, loc.longitude,
                             side_id=_RED_SIDE, side_color="red",
                             name="Floridistan AFB #%d" % i)
            )

        tasks = [
            _attack_task("tgt%d" % i, self.targets[aid])
            for i, aid in enumerate(self.agent_ids)
        ]
        self.a_init = {aid: [(i, 0, 0)] for i, aid in enumerate(self.agent_ids)}
        self.beliefs = {
            aid: _StubBelief(list(tasks), {k: list(v) for k, v in self.a_init.items()})
            for aid in self.agent_ids
        }
        self.agents = [
            Agent(location=Location(_BASE.latitude, _BASE.longitude),
                  capabilities=[], budget=fuel, move_cost_function=lambda s, d: 0.0,
                  speed=speed, return_location=Location(_BASE.latitude, _BASE.longitude),
                  agent_id=aid, side_color="blue", home_base_id="base-blue")
            for aid in self.agent_ids
        ]
        self.oracle_tasks = list(tasks)
        self.oracle_solution = dict(self.a_init)
        self.game = _StubGame(self.scenario)
        self.executor = _StubExecutor(self.agent_ids)
        self.env = _StubEnv(self.scenario, targets=self.targets)
        self.observation = self.scenario
        self.record = False
        self.split_meta = {"known": n_agents, "hidden": 0, "full": n_agents,
                           "partial": n_agents}
        self.placements = ()


_PARAMS = FuelDamageParameters()


# =============================================================================
# PO1 -- deterministic physical event and accounting
# =============================================================================

def test_p1_1_the_rng_domain_is_stable_and_unreachable_from_other_rng_state() -> None:
    """P1.1. The derived seed is a pure function of the episode seed, and only of it.

    The three properties the module claims: reproducible across processes (so a run can
    be re-derived later), distinct per seed (so consecutive training seeds do not share
    a condition), and untouched by any other RNG consumer (so which episode is damaged
    cannot depend on how many placements the generator happened to reject).
    """
    first = [derive_fuel_damage_seed(s) for s in range(16)]

    random.seed(999)
    torch.manual_seed(999)
    [random.random() for _ in range(101)]
    torch.rand(37)
    second = [derive_fuel_damage_seed(s) for s in range(16)]

    assert first == second, "the derived seed moved with unrelated RNG consumption"
    assert len(set(first)) == len(first), "consecutive episode seeds collide"

    # Reproducible in a FRESH interpreter too -- `hash()` would not be (PYTHONHASHSEED).
    child = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "from match_aou.rl.training.graph_fuel_damage import derive_fuel_damage_seed;"
         "import json; print(json.dumps([derive_fuel_damage_seed(s) for s in range(16)]))"
         % str(SRC)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.strip().splitlines()[-1]) == first


def test_p1_2_condition_ego_and_window_are_a_pure_function_of_the_seed() -> None:
    """P1.2. Same seed + same mode -> same condition, same ego, field-identical plan.

    Deliberately run with DIFFERENT prior RNG consumption between the two builds: that is
    the failure mode a module-global draw would have, and it would not show up in a test
    that built both plans from a clean interpreter state.
    """
    seed = 7
    random.seed(1)
    torch.manual_seed(1)
    plan_a = build_fuel_damage_plan(
        _FuelDamageCtx(), episode_seed=seed, params=_PARAMS
    )

    random.seed(424242)
    [random.random() for _ in range(53)]
    torch.manual_seed(424242)
    torch.rand(11)
    plan_b = build_fuel_damage_plan(
        _FuelDamageCtx(), episode_seed=seed, params=_PARAMS
    )

    assert plan_a == plan_b, "the plan is not a pure function of the episode seed"
    assert plan_a.condition == CONDITION_DAMAGED  # seed 7 is damaged (see P1.1's domain)
    assert plan_a.ego_id in plan_a.eligible_ego_ids

    # Different seeds really do produce different plans (the draw is not a constant).
    conditions = {
        resolve_condition(episode_seed=s, params=_PARAMS) for s in range(24)
    }
    assert conditions == {CONDITION_CLEAN, CONDITION_DAMAGED}, conditions
    egos = {
        build_fuel_damage_plan(_FuelDamageCtx(), episode_seed=s, params=_PARAMS).ego_id
        for s in range(24)
        if resolve_condition(episode_seed=s, params=_PARAMS) == CONDITION_DAMAGED
    }
    assert len(egos) > 1, "every damaged episode selected the same ego: %r" % egos


def test_p1_3_forced_damaged_picks_the_ego_the_mixture_would_have() -> None:
    """P1.3. The forced modes share the mixture's rng stream position.

    This is what makes a matched pair a PAIR: member B must be the damaged version of the
    very episode member A ran clean, which means the same ego, the same window and the
    same event point -- not merely the same scenario.
    """
    forced_clean = _PARAMS.__class__(mode=FuelDamageMode.FORCED_CLEAN)
    forced_damaged = _PARAMS.__class__(mode=FuelDamageMode.FORCED_DAMAGED)

    for seed in range(12):
        assert resolve_condition(episode_seed=seed, params=forced_clean) == CONDITION_CLEAN
        assert (resolve_condition(episode_seed=seed, params=forced_damaged)
                == CONDITION_DAMAGED)

        forced = build_fuel_damage_plan(
            _FuelDamageCtx(), episode_seed=seed, params=forced_damaged
        )
        if resolve_condition(episode_seed=seed, params=_PARAMS) == CONDITION_DAMAGED:
            mixed = build_fuel_damage_plan(
                _FuelDamageCtx(), episode_seed=seed, params=_PARAMS
            )
            assert forced.ego_id == mixed.ego_id, seed
            assert forced.post_damage_fuel == mixed.post_damage_fuel, seed
            assert (forced.event_latitude, forced.event_longitude) == (
                mixed.event_latitude, mixed.event_longitude), seed

        # A forced-CLEAN member computes no window at all, so it cannot fail for
        # window reasons -- the two members of a pair fail independently or not at all.
        clean = build_fuel_damage_plan(
            _FuelDamageCtx(), episode_seed=seed, params=forced_clean
        )
        assert clean.condition == CONDITION_CLEAN
        assert clean.ego_id is None and clean.post_damage_fuel is None


def test_p1_4_the_window_is_blades_own_fuel_arithmetic_and_is_strict() -> None:
    """P1.4. `fuel_for_distance_km` IS `Game.get_fuel_needed_to_return_to_base`.

    Transcribed independently here from the engine source rather than imported, so the
    test fails if the module ever drifts toward a convenient-but-different fuel model
    (the specific thing the task forbids).
    """
    for distance, speed, rate in ((75.0, 1303.0, 6700.0), (425.0, 461.0, 33500.0),
                                  (1.0, 565.0, 33500.0), (0.0, 1043.0, 6700.0)):
        nm = (distance * 1000.0) / 1852.0        # NAUTICAL_MILES_TO_METERS
        expected = (nm / speed) * rate
        got = fuel_for_distance_km(distance, speed_knots=speed, fuel_rate=rate)
        assert abs(got - expected) < 1e-9, (distance, speed, rate, got, expected)
    assert NAUTICAL_MILES_TO_METERS == 1852.0

    # Degenerate inputs fail loudly rather than producing an infinite window.
    for kwargs in ({"speed_knots": 0.0, "fuel_rate": 6700.0},
                   {"speed_knots": 1303.0, "fuel_rate": 0.0},
                   {"speed_knots": float("nan"), "fuel_rate": 6700.0}):
        try:
            fuel_for_distance_km(10.0, **kwargs)
        except FuelDamageError:
            pass
        else:
            raise AssertionError("fuel_for_distance_km accepted %r" % kwargs)

    # The window itself: both ends are `margin * fuel(distance)`, the chosen value is the
    # midpoint, the interval is non-empty and half-open, and the choice is a real LOSS.
    plan = build_fuel_damage_plan(
        _FuelDamageCtx(), episode_seed=1, params=_PARAMS
    )
    assert plan.condition == CONDITION_DAMAGED
    margin = _PARAMS.rtb_safety_margin
    assert abs(plan.rtb_fuel_floor - margin * fuel_for_distance_km(
        plan.rtb_distance_km, speed_knots=plan.speed_knots,
        fuel_rate=plan.fuel_rate)) < 1e-9
    assert abs(plan.continue_fuel_requirement - margin * fuel_for_distance_km(
        plan.continue_distance_km, speed_knots=plan.speed_knots,
        fuel_rate=plan.fuel_rate)) < 1e-9
    assert plan.rtb_fuel_floor < plan.continue_fuel_requirement
    assert (plan.rtb_fuel_floor <= plan.post_damage_fuel
            < plan.continue_fuel_requirement)
    assert abs(plan.post_damage_fuel
               - 0.5 * (plan.rtb_fuel_floor + plan.continue_fuel_requirement)) < 1e-9
    assert plan.post_damage_fuel < plan.projected_fuel_at_event < plan.fuel_at_launch

    # The event point really sits at the requested fraction ALONG the leg (great circle,
    # not a flat lat/lon lerp), which is what makes the RTB floor measured from it right.
    event = plan.event_location
    assert abs(Location(_BASE.latitude, _BASE.longitude).distance_to(event)
               - _PARAMS.leg_progress_threshold * plan.leg_length_km) < 1e-6
    assert abs(event.distance_to(plan.first_target_location)
               - (1.0 - _PARAMS.leg_progress_threshold) * plan.leg_length_km) < 1e-6


def test_p1_4b_great_circle_interpolation_is_proportional() -> None:
    """P1.4. Interpolation is along-track proportional at every fraction."""
    end = _point_at(_BASE, 400.0, 77.0)
    total = _BASE.distance_to(end)
    for fraction in (0.0, 0.1, 0.3, 0.5, 0.85, 1.0):
        point = interpolate_great_circle(_BASE, end, fraction)
        assert abs(_BASE.distance_to(point) - fraction * total) < 1e-6, fraction
    # Degenerate endpoints degrade to the start rather than raising or producing NaN.
    same = interpolate_great_circle(_BASE, Location(_BASE.latitude, _BASE.longitude), 0.3)
    assert abs(same.latitude - _BASE.latitude) < 1e-12


def test_p1_5_the_live_aircraft_is_mutated_exactly_once() -> None:
    """P1.5. ONE physical mutation of the real `current_fuel`, at the right tick.

    Driven over far more ticks than it takes to cross the threshold, so a controller that
    re-applied every tick (or re-fired after the ego passed the target) fails here.
    """
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    plan = controller.plan
    assert plan.condition == CONDITION_DAMAGED
    ego = plan.ego_id
    aircraft = ctx.scenario.get_aircraft(ego)
    launch_fuel = aircraft.current_fuel

    fires, fuels = [], []
    for tick in range(200):
        fired = controller.maybe_apply(ctx.scenario, tick)
        if fired is not None:
            fires.append((tick, fired))
        fuels.append(aircraft.current_fuel)
        ctx.env.step([])

    assert len(fires) == 1, "the event fired %d time(s): %r" % (len(fires), fires)
    fire_tick, fired_ego = fires[0]
    assert fired_ego == ego

    # Exactly ONE change of the real value over 200 ticks, on the event tick, to the
    # planned quantity. (The stub env moves aircraft but burns no fuel, so the event is
    # the only possible source of a change -- which is what makes this count exact.)
    changes = [i for i in range(1, len(fuels)) if fuels[i] != fuels[i - 1]]
    assert changes == [fire_tick], (changes, fire_tick)
    assert aircraft.current_fuel == plan.post_damage_fuel
    assert aircraft.current_fuel < launch_fuel

    # It fired at the FIRST tick at or past the threshold, not later.
    outcome = controller.outcome
    assert outcome.fired and outcome.event_tick == fire_tick
    assert outcome.observed_progress >= plan.progress_threshold
    assert outcome.fuel_before == launch_fuel
    assert outcome.fuel_after == plan.post_damage_fuel
    assert abs(outcome.damage_factor - plan.post_damage_fuel / launch_fuel) < 1e-12
    assert controller.fired

    # And no peer aircraft was touched at all.
    for peer in ctx.agent_ids:
        if peer == ego:
            continue
        assert ctx.scenario.get_aircraft(peer).current_fuel == launch_fuel


def test_p1_5b_the_event_cannot_fire_before_the_threshold_or_while_grounded() -> None:
    """P1.5. Below the threshold, and with no airborne aircraft, nothing happens."""
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    ego = controller.plan.ego_id
    aircraft = ctx.scenario.get_aircraft(ego)

    # At the launch point progress is 0 -- well below 0.30.
    assert controller.maybe_apply(ctx.scenario, 0) is None
    assert aircraft.current_fuel == controller.plan.fuel_at_launch
    assert abs(controller.observed_progress(ctx.scenario)) < 1e-9

    # Grounded / removed: not in `scenario.aircraft` at all.
    grounded = _StubScenario(aircraft=[], airbases=ctx.scenario.airbases)
    assert controller.maybe_apply(grounded, 500) is None
    assert controller.observed_progress(grounded) is None
    assert not controller.fired


def test_p1_5c_an_upward_mutation_is_refused_loudly() -> None:
    """P1.5. If the live fuel is already at or below the target, it RAISES.

    That state means the pre-run projection did not describe the run. Clamping it would
    turn a broken window into a silently different experiment; the loud failure lands as
    an accounted `run`-stage attempt failure instead.
    """
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    ego = controller.plan.ego_id
    aircraft = ctx.scenario.get_aircraft(ego)
    # Fly to the threshold, then drain the tank below the planned post-damage value.
    for _ in range(200):
        if (controller.observed_progress(ctx.scenario) or 0.0) >= 0.30:
            break
        ctx.env.step([])
    aircraft.current_fuel = controller.plan.post_damage_fuel * 0.5

    try:
        controller.maybe_apply(ctx.scenario, 123)
    except FuelDamageError as exc:
        assert "not above" in str(exc), str(exc)
    else:
        raise AssertionError("an upward 'damage' mutation was accepted")
    assert not controller.fired


def test_p1_6_no_valid_window_raises_and_never_becomes_a_clean_episode() -> None:
    """P1.6. Each of the three window preconditions fails LOUDLY and separately."""
    launch = Location(_BASE.latitude, _BASE.longitude)
    target = _point_at(_BASE, 250.0, 45.0)
    ok = dict(condition=CONDITION_DAMAGED, mode=FuelDamageMode.SEEDED_MIXTURE,
              derived_seed=1, eligible_ego_ids=("a",), ego_id="a",
              launch_point=launch, home_base=launch, route_points=[target],
              speed_knots=1303.0, fuel_rate=6700.0, max_fuel=12000.0,
              fuel_at_launch=12000.0, params=_PARAMS)
    # Control: the reference case really does produce a plan.
    assert plan_fuel_damage(**ok).condition == CONDITION_DAMAGED

    # (a) pre-damage fuel already below the continue requirement -> no decision exists.
    starved = dict(ok, fuel_at_launch=900.0)
    try:
        plan_fuel_damage(**starved)
    except FuelDamageError as exc:
        assert "already below the continue requirement" in str(exc), str(exc)
    else:
        raise AssertionError("a pre-infeasible plan produced a window")

    # (b) zero-length first leg -> no leg to place an event on.
    degenerate = dict(ok, route_points=[Location(launch.latitude, launch.longitude)])
    try:
        plan_fuel_damage(**degenerate)
    except FuelDamageError as exc:
        assert "zero-length leg" in str(exc), str(exc)
    else:
        raise AssertionError("a zero-length leg produced a window")

    # (c) an empty predicted route -> the ego was not eligible after all.
    try:
        plan_fuel_damage(**dict(ok, route_points=[]))
    except FuelDamageError:
        pass
    else:
        raise AssertionError("an empty route produced a window")

    # (d) a damaged episode with NO eligible ego is a failure, not a clean episode.
    ctx = _FuelDamageCtx()
    ctx.a_init = {aid: [] for aid in ctx.agent_ids}
    try:
        build_fuel_damage_plan(ctx, episode_seed=1, params=_PARAMS)
    except FuelDamageError as exc:
        assert "not silently downgraded to clean" in str(exc), str(exc)
    else:
        raise AssertionError("a damaged episode with no eligible ego was downgraded")


def test_p1_6b_a_window_failure_is_an_accounted_setup_stage_failure() -> None:
    """P1.6. The real `_run_one_episode` attributes it to `setup`, not a new stage.

    That attribution is what routes it into `skip_and_account_v1`: recorded once, never
    retried, never replaced by another seed, and never entering a reward aggregate.
    """
    ctx = _FuelDamageCtx()
    # Give the selected ego a tank too small for its own route: the window premise fails.
    for aircraft in ctx.scenario.aircraft:
        aircraft.current_fuel = 500.0
        aircraft.max_fuel = 500.0

    try:
        _run_one_episode_against(ctx, seed=1)
    except EpisodeAttemptError as exc:
        assert exc.stage == "setup", exc.stage
        assert exc.stage in graph_train._PIPELINE_STAGES
        assert isinstance(exc.original, FuelDamageError), type(exc.original)
    else:
        raise AssertionError("a missing fuel window produced a successful measurement")
    assert ctx.env.closed, "the env must still be closed on the failing path"


def test_p1_7_failure_records_carry_the_scheduled_condition(tmp_path: Path) -> None:
    """P1.7. `failed counts by condition` is answerable for attempts that never ran.

    The condition is resolvable from the seed and the mode alone, so it is recorded even
    though the attempt produced no episode -- otherwise a per-condition mean could be
    reported next to a denominator that quietly excluded its own failures.
    """
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=6, base_seed=0,
                      output_dir=tmp_path / "run", eval_every=0, eval_episodes=0,
                      checkpoint_every=0)
    failing = {s for s in range(6)}
    summary, _events = _run_stub_training(cfg, failing_seeds=failing)

    ledger = [json.loads(line) for line in
              (tmp_path / "run" / "episode_failures.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    assert len(ledger) == 6
    for record in ledger:
        expected = resolve_condition(
            episode_seed=record["seed"], params=cfg.fuel_damage_parameters()
        )
        assert record["condition"] == expected, record
    assert summary["failures_by_condition"] == {
        CONDITION_CLEAN: sum(
            1 for s in failing
            if resolve_condition(episode_seed=s, params=cfg.fuel_damage_parameters())
            == CONDITION_CLEAN),
        CONDITION_DAMAGED: sum(
            1 for s in failing
            if resolve_condition(episode_seed=s, params=cfg.fuel_damage_parameters())
            == CONDITION_DAMAGED),
    }
    # An all-failed batch reports null, never the oracle optimum 0.0.
    record = summary["train_records"][0]
    assert record["train_reward_mean"] is None
    assert record["reward_mean_clean"] is None and record["reward_mean_damaged"] is None
    assert (record["n_clean_attempted"] + record["n_damaged_attempted"]
            == record["n_attempted"] == 6)


# =============================================================================
# PO2 -- locality and same-snapshot no-communication
# =============================================================================

def _drive_loop(ctx, *, seed=1, agent_ids=None, params=_PARAMS):
    """Run the REAL `run_episode` over a stub world, recording every trigger call.

    `_wake_decision` is replaced by a recorder: the encoder and the head are irrelevant to
    what this proves (WHEN the event lands and WHO is told), and stubbing them keeps the
    test torch-light and deterministic. Everything else -- the tick structure, the
    controller call site, the per-ego flag -- is production code.
    """
    if agent_ids is not None:
        ctx.agent_ids = list(agent_ids)
    controller = build_fuel_damage_controller(ctx, episode_seed=seed, params=params)
    seen = []

    real_decide = graph_tick_loop.decide_triggers

    def spy_decide(belief_tasks, belief_solution, sensed, eta=None, *,
                   ego_id, clock, fuel_damage=False):
        # Snapshot what THIS ego observes about EVERY aircraft's fuel at its own turn.
        seen.append({
            "tick": clock,
            "ego": str(ego_id),
            "fuel_damage": bool(fuel_damage),
            "fuels": {str(a.id): a.current_fuel for a in ctx.scenario.aircraft},
        })
        return real_decide(belief_tasks, belief_solution, sensed,
                           ego_id=ego_id, clock=clock, fuel_damage=fuel_damage)

    wakes = []

    def spy_wake(_policy, ego_id, _obs, _belief, _executor, _cfg, tick, **_kw):
        wakes.append((str(ego_id), int(tick)))
        return graph_tick_loop.Transition(
            gobs=None, ego_id=str(ego_id), tick=int(tick),
            meta_action=int(MetaAction.SELF_PRESERVATION_ABORT), node_v=0,
            log_prob=0.0, entropy=0.0,
        )

    saved = (graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision)
    graph_tick_loop.decide_triggers = spy_decide
    graph_tick_loop._wake_decision = spy_wake
    try:
        result = graph_tick_loop.run_episode(
            None, ctx, GraphObservationConfig(detection_range_km=50.0),
            max_ticks=200, fuel_damage=controller,
        )
    finally:
        graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision = saved
    return controller, seen, wakes, result


def test_p2_1_and_p2_2_the_event_precedes_phase_1_and_reaches_one_ego_only() -> None:
    """P2.1 + P2.2. Applied before the ego loop; exactly one ego is told.

    The load-bearing assertion is on the FIRST ego processed on the event tick: if the
    mutation happened inside the loop, the ego that ran before the damaged one would have
    observed the pre-damage fuel, and the episode would depend on iteration order.
    """
    ctx = _FuelDamageCtx()
    controller, seen, wakes, _result = _drive_loop(ctx)
    plan = controller.plan
    assert plan.condition == CONDITION_DAMAGED
    ego = plan.ego_id
    assert controller.outcome.fired, "the event never fired in the loop"
    event_tick = controller.outcome.event_tick

    on_event_tick = [row for row in seen if row["tick"] == event_tick]
    assert len(on_event_tick) == len(ctx.agent_ids), on_event_tick
    # P2.1: EVERY ego on that tick -- including the first processed -- saw post-damage.
    for row in on_event_tick:
        assert row["fuels"][ego] == plan.post_damage_fuel, (row["ego"], row["fuels"])
    # And the tick before the event, every ego saw the pre-damage value.
    before = [row for row in seen if row["tick"] == event_tick - 1]
    assert before, "no tick preceded the event"
    for row in before:
        assert row["fuels"][ego] == plan.fuel_at_launch

    # P2.2: exactly one ego was flagged, on exactly one tick.
    flagged = [(row["tick"], row["ego"]) for row in seen if row["fuel_damage"]]
    assert flagged == [(event_tick, ego)], flagged
    assert (ego, event_tick) in wakes

    # The wake was attributed back to the event, with the meta-action it produced.
    outcome = controller.outcome
    assert outcome.wake_occurred
    assert outcome.wake_meta_action == int(MetaAction.SELF_PRESERVATION_ABORT)

    # And no peer's fuel was ever observed to change.
    for row in seen:
        for peer in ctx.agent_ids:
            if peer == ego:
                continue
            assert row["fuels"][peer] == plan.fuel_at_launch, (row["tick"], peer)


def test_p2_3_phase_1_ego_iteration_order_cannot_change_the_result() -> None:
    """P2.3. Reversing the ego order changes nothing any ego observed.

    The structural no-communication property, re-proven for the new exogenous input:
    because the event lands before the loop and BLADE advances only in Phase 2, two runs
    that differ ONLY in iteration order must agree on the event, on the flags and on
    every fuel snapshot.
    """
    forward_ctx = _FuelDamageCtx()
    forward = _drive_loop(forward_ctx, agent_ids=forward_ctx.agent_ids)
    reverse_ctx = _FuelDamageCtx()
    reverse = _drive_loop(reverse_ctx, agent_ids=list(reversed(reverse_ctx.agent_ids)))

    assert forward[0].plan == reverse[0].plan
    assert forward[0].outcome == reverse[0].outcome

    def by_tick(rows):
        out = {}
        for row in rows:
            out.setdefault(row["tick"], {})[row["ego"]] = (row["fuel_damage"],
                                                           row["fuels"])
        return out

    assert by_tick(forward[1]) == by_tick(reverse[1]), (
        "an ego observed something different under a different iteration order"
    )
    assert sorted(forward[2]) == sorted(reverse[2])


def test_p2_4_the_damaged_graph_carries_post_damage_fuel_and_peers_stay_featureless() -> None:
    """P2.4. The same-wake graph reads the mutated aircraft; peer rows are still 0.0.

    Uses the REAL `build_graph_observation` -- the point is precisely that the builder is
    unchanged and that the event reaches it through the live aircraft rather than through
    any new feature or channel.
    """
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    plan = controller.plan
    ego = plan.ego_id
    cfg = GraphObservationConfig(detection_range_km=50.0)

    # Fly to the threshold and fire.
    for tick in range(200):
        if controller.maybe_apply(ctx.scenario, tick) is not None:
            break
        ctx.env.step([])
    assert controller.fired

    belief = ctx.beliefs[ego]
    gobs = build_graph_observation(
        scenario=ctx.scenario, agent_id=ego, current_plan=belief.solution.get(ego),
        current_time=controller.outcome.event_tick, tasks=belief.tasks,
        solution=belief.solution, precedence_relations=[], config=cfg,
    )
    aircraft = ctx.scenario.get_aircraft(ego)
    expected = plan.post_damage_fuel / aircraft.max_fuel
    # Row 0 is the ego by the builder's "ego FIRST" node ordering (graph_builder).
    assert abs(float(gobs.agent_features[0, 0]) - expected) < 1e-6, gobs.agent_features
    # A peer's graph, built from the SAME scenario, cannot see the damaged fuel: its own
    # ego row is its own (undamaged) fuel and every peer row is featureless.
    for peer in ctx.agent_ids:
        peer_belief = ctx.beliefs[peer]
        peer_gobs = build_graph_observation(
            scenario=ctx.scenario, agent_id=peer,
            current_plan=peer_belief.solution.get(peer),
            current_time=controller.outcome.event_tick, tasks=peer_belief.tasks,
            solution=peer_belief.solution, precedence_relations=[], config=cfg,
        )
        rows = peer_gobs.agent_features
        assert all(abs(float(rows[i, 0])) < 1e-12 for i in range(1, rows.shape[0])), (
            "a peer agent row is not featureless: %r" % (rows,)
        )
        if peer != ego:
            assert abs(float(rows[0, 0]) - 1.0) < 1e-6, (
                "peer %s's own fuel_norm changed: it must not see the event" % peer
            )


def test_p2_5_a_clean_controller_is_inert() -> None:
    """P2.5. Clean: no mutation, no wake, no flag -- and the plan still records that."""
    ctx = _FuelDamageCtx()
    clean = FuelDamageParameters(mode=FuelDamageMode.FORCED_CLEAN)
    controller, seen, wakes, _result = _drive_loop(ctx, seed=1, params=clean)

    assert controller.plan.condition == CONDITION_CLEAN
    assert controller.plan.ego_id is None
    assert not controller.fired and not controller.outcome.wake_occurred
    assert not any(row["fuel_damage"] for row in seen)
    assert wakes == []
    fuels = {a.current_fuel for a in ctx.scenario.aircraft}
    assert fuels == {controller.plan.fuel_at_launch or 12000.0}, fuels

    # `note_wake` on a clean controller is a no-op too (it cannot invent an event).
    controller.note_wake(ego_id="ego0", meta_action=int(MetaAction.PLAN_COMPLIANCE))
    assert not controller.outcome.wake_occurred


def test_p2_5b_the_trigger_seam_stays_pure_and_edits_nothing() -> None:
    """P2. FUEL_DAMAGE wakes without touching the belief -- the layer stays pure."""
    tasks = [_attack_task("t0", _point_at(_BASE, 250.0, 30.0))]
    solution = {"ego": [(0, 0, 0)], "peer": []}
    new_tasks, new_solution, wake, events = decide_triggers(
        tasks, solution, {}, ego_id="ego", clock=42, fuel_damage=True,
    )
    assert wake is True
    assert events == [(TriggerKind.FUEL_DAMAGE, NO_TASK_INDEX)]
    assert new_tasks == tasks and new_tasks is not tasks
    assert new_solution == solution and new_solution is not solution
    # NO_TASK_INDEX is a sentinel, not a usable task index.
    assert NO_TASK_INDEX < 0


# =============================================================================
# PO3 -- behavioural and measurement integration
# =============================================================================

def test_p3_1_abort_empties_only_the_damaged_ego_and_issues_exactly_one_rtb() -> None:
    """P3.1. The existing effect + executor path, exercised with a controlled abort.

    Nothing new is built for RTB: the abort drops the ego's assignment, the ego's plan
    becomes empty, and `GraphPlanExecutor`'s pre-existing empty-plan branch issues its
    single latched `aircraft_return_to_base`. This test locks that the fuel-damage
    decision reaches BLADE through exactly that path and through no other.
    """
    ego, peer = "ego0", "ego1"
    t_ego = _point_at(_BASE, 250.0, 30.0)
    t_peer = _point_at(_BASE, 250.0, 90.0)
    tasks = [_attack_task("tgt0", t_ego), _attack_task("tgt1", t_peer)]
    solution = {ego: [(0, 0, 0)], peer: [(1, 0, 0)]}

    def _agent(aid):
        return Agent(location=Location(_BASE.latitude, _BASE.longitude), capabilities=[],
                     budget=1.0, move_cost_function=lambda s, d: 0.0, agent_id=aid,
                     side_color="blue", home_base_id="base-blue",
                     return_location=Location(_BASE.latitude, _BASE.longitude))

    executor = GraphPlanExecutor(
        tasks=tasks, solution=solution, agents=[_agent(ego), _agent(peer)],
        arrival_threshold_km=50.0,
    )
    # Both egos airborne, mid-leg, far from their targets (so the plan is not "done").
    mid = _point_at(_BASE, 75.0, 30.0)
    scenario = _StubScenario(
        aircraft=[_StubAircraft(ego, mid.latitude, mid.longitude),
                  _StubAircraft(peer, mid.latitude, mid.longitude)],
        airbases=[_StubAirbase("base-blue", _BASE.latitude, _BASE.longitude)],
    )

    # Baseline: neither ego is RTB-ing; both are flying their plan.
    before = executor.next_actions(scenario)
    assert not any("aircraft_return_to_base" in c for c in before), before

    # The abort itself: the SAME pure effect layer a wake would call, on the damaged
    # ego's private belief only.
    class _Gobs:
        task_target_ids = ["tgt0", "tgt1"]

    new_solution = apply_meta_action(
        solution, _Gobs(), ego, int(MetaAction.SELF_PRESERVATION_ABORT), 0, tasks
    )
    assert new_solution[ego] == [], new_solution
    assert new_solution[peer] == [(1, 0, 0)], "the peer's plan was edited"
    assert solution[ego] == [(0, 0, 0)], "apply_meta_action mutated its input"

    executor.resync(new_solution, ego_id=ego, tasks=tasks)
    assert executor.plans[peer] == [(1, 0, 0)], "the peer's executor slice moved"

    commands = executor.next_actions(scenario)
    rtbs = [c for c in commands if "aircraft_return_to_base" in c]
    assert rtbs == ["aircraft_return_to_base('%s')" % ego], commands
    assert executor.rtb_issued.get(ego) is True
    assert executor.rtb_issued.get(peer) is not True

    # EXACTLY one: the single-issue latch means a second tick emits no second RTB
    # (`aircraft_return_to_base` is a TOGGLE in BLADE -- a second one would cancel it).
    again = executor.next_actions(scenario)
    assert not any("aircraft_return_to_base" in c for c in again), again


def test_p3_2_the_penalty_coefficient_reaches_both_harnesses() -> None:
    """P3.2. `RewardConfig(aircraft_penalty_coeff=2.25)` is what the reward is CALLED with.

    Both halves matter: the two configs must agree (a rollout that scored deaths as free
    would not be diagnosing a training episode), and the value must actually arrive at
    `compute_episode_reward` rather than being recorded and then dropped.
    """
    train_cfg = TrainConfig(n_iterations=1)
    rollout_cfg = RolloutConfig()
    assert train_cfg.aircraft_penalty_coeff == 2.25
    assert rollout_cfg.aircraft_penalty_coeff == 2.25
    assert train_cfg.reward_config().aircraft_penalty_coeff == 2.25
    assert rollout_cfg.reward_config().aircraft_penalty_coeff == 2.25
    # The formula's other knob is untouched -- graph_reward stays frozen.
    assert train_cfg.reward_config().regret_epsilon == 1e-5

    seen = {}
    ctx = _FuelDamageCtx()

    def spy_reward(_ctx, _result, cfg=None):
        seen["cfg"] = cfg

        class _R:
            reward = -0.25
        return _R()

    _run_one_episode_against(ctx, seed=1, reward_spy=spy_reward)
    assert seen["cfg"] is not None, "compute_episode_reward was called with no RewardConfig"
    assert seen["cfg"].aircraft_penalty_coeff == 2.25, seen["cfg"]

    # And the resolved coefficient is recorded, not merely used.
    payload = json.loads(_write_run_config_to_string(train_cfg))
    assert payload["difficulty"]["reward"]["aircraft_penalty_coeff"] == 2.25
    assert payload["difficulty"]["reward"]["formula_changed"] is False
    assert payload["difficulty"]["fuel_damage"]["mode"] == FuelDamageMode.SEEDED_MIXTURE
    assert payload["difficulty"]["fuel_damage"]["probability"] == 0.5
    assert payload["difficulty"]["fuel_damage"]["leg_progress_threshold"] == 0.30
    assert payload["difficulty"]["fuel_damage"]["rtb_safety_margin"] == 1.10


def test_p3_3_eval_pairs_share_the_seed_and_split_the_tags(tmp_path: Path) -> None:
    """P3.3. Two members per held-out seed: same seed, forced conditions, disjoint tags."""
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                      output_dir=tmp_path / "run", eval_every=1, eval_episodes=3,
                      eval_base_seed=1_000_000, checkpoint_every=0)
    _summary, events = _run_stub_training(cfg)

    evals = [e for e in events if e[0] == "episode" and e[1] == "eval"]
    # One PRE-UPDATE round and one post-update round, each 3 seeds x 2 members.
    assert len(evals) == 2 * 3 * 2, len(evals)

    first_round = evals[:6]
    seeds = [e[2] for e in first_round]
    tags = [e[3] for e in first_round]
    modes = [e[4] for e in first_round]

    # Each held-out seed appears exactly twice, adjacently -- one pair.
    assert seeds == [1_000_000, 1_000_000, 1_000_001, 1_000_001,
                     1_000_002, 1_000_002], seeds
    # ...once forced clean and once forced damaged.
    assert modes == [FuelDamageMode.FORCED_CLEAN, FuelDamageMode.FORCED_DAMAGED] * 3
    # ...and the two members write to DIFFERENT scenario files.
    assert len(set(tags)) == 6, tags
    assert tags == [eval_member_tag(round_ordinal=0, e=e, member=m)
                    for e in range(3) for m in (0, 1)]
    # The second round's tags are disjoint from the first's (rounds never overwrite).
    assert not (set(tags) & set(e[3] for e in evals[6:]))

    # The scenario files really coexist on disk.
    scen = sorted(p.name for p in (tmp_path / "run" / "scenarios").glob("*.json"))
    assert len(scen) == len(set(scen)) >= 12, scen


def test_p3_3b_the_paired_delta_is_taken_only_over_complete_pairs(tmp_path: Path) -> None:
    """P3.3. A pair with a failed member contributes to no delta -- and says so.

    The tempting repair (use the surviving member, or treat the gap as 0) would report a
    within-seed difference that was never measured. What is locked here is that the delta
    population and the attempt population are reported separately and honestly.
    """
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                      output_dir=tmp_path / "run", eval_every=1, eval_episodes=3,
                      eval_base_seed=1_000_000, checkpoint_every=0)
    # Fail the DAMAGED member of the middle held-out seed, in every round.
    summary, _events = _run_stub_training(
        cfg, failing_eval=(1_000_001, FuelDamageMode.FORCED_DAMAGED)
    )

    rounds = summary["eval_records"]
    assert rounds, "no eval round was recorded"
    for ev in rounds:
        assert ev["n_pairs_attempted"] == 3
        assert ev["n_pairs_successful"] == 2, ev
        assert ev["pair_success_fraction"] == 2 / 3
        assert ev["n_attempted"] == 6 and ev["n_successful"] == 5 and ev["n_failed"] == 1
        assert ev["eval_n_clean_attempted"] == 3 and ev["eval_n_clean_successful"] == 3
        assert ev["eval_n_damaged_attempted"] == 3
        assert ev["eval_n_damaged_successful"] == 2
        assert ev["eval_n_damaged_failed"] == 1
        assert ev["eval_paired_reward_delta"] is not None
        assert ev["paired_delta_over"] == "pairs_with_both_members_successful"
        # The held-out SEED band is 3 wide however many attempts each seed took.
        assert ev["seed_band"]["stop"] - ev["seed_band"]["start"] == 3

    # Every failure is in the ledger exactly once, under its own condition.
    ledger = [json.loads(line) for line in
              (tmp_path / "run" / "episode_failures.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    assert all(r["condition"] == CONDITION_DAMAGED for r in ledger), ledger
    assert all(r["phase"] == "eval" for r in ledger)
    assert summary["accounting_reconciled"] is True


def test_p3_3c_an_empty_pair_population_is_null_not_zero(tmp_path: Path) -> None:
    """P3.3. No complete pair -> `null`, never 0.0 (which would mean 'no effect')."""
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                      output_dir=tmp_path / "run", eval_every=1, eval_episodes=2,
                      eval_base_seed=1_000_000, checkpoint_every=0)
    summary, _events = _run_stub_training(
        cfg, failing_eval=("*", FuelDamageMode.FORCED_DAMAGED)
    )
    for ev in summary["eval_records"]:
        assert ev["n_pairs_successful"] == 0
        assert ev["eval_paired_reward_delta"] is None
        assert ev["eval_reward_mean_damaged"] is None
        assert ev["eval_reward_mean_clean"] is not None   # the clean members did run
    assert summary["final_eval_paired_reward_delta"] is None
    assert summary["final_eval_pairs_successful"] == 0


def test_p3_3d_the_config_refuses_an_eval_band_that_cannot_hold_its_pairs() -> None:
    """P3.3. The doubled tag namespace is validated up front, not discovered mid-run."""
    stride = graph_train._EVAL_ROUND_TAG_STRIDE
    ok = TrainConfig(n_iterations=1, episodes_per_iteration=1,
                     eval_every=1, eval_episodes=stride // 2,
                     eval_base_seed=5_000_000)
    ok.validate()   # exactly fits
    too_big = TrainConfig(n_iterations=1, episodes_per_iteration=1,
                          eval_every=1, eval_episodes=stride // 2 + 1,
                          eval_base_seed=5_000_000)
    try:
        too_big.validate()
    except ValueError as exc:
        assert "matched pair members" in str(exc), str(exc)
    else:
        raise AssertionError("an eval band that overflows its tag namespace was accepted")


# =============================================================================
# Config surface + purity
# =============================================================================

def test_rollout_config_mirrors_the_train_difficulty_cell() -> None:
    """ANTI-DRIFT: the two harnesses agree on the difficulty factor, field for field."""
    t = TrainConfig(n_iterations=1)
    r = RolloutConfig()
    for name in ("fuel_damage_mode", "fuel_damage_probability",
                 "fuel_damage_leg_progress", "fuel_damage_rtb_margin",
                 "aircraft_penalty_coeff"):
        assert getattr(t, name) == getattr(r, name), (
            "RolloutConfig.%s (%r) drifted from TrainConfig.%s (%r)"
            % (name, getattr(r, name), name, getattr(t, name))
        )
    assert t.fuel_damage_parameters() == r.fuel_damage_parameters()
    assert t.reward_config() == r.reward_config()


def test_the_configs_refuse_a_forced_mode_and_bad_parameters() -> None:
    """A forced mode is an EVAL pair member; it is not a training mixture."""
    for cfg in (TrainConfig(n_iterations=1, fuel_damage_mode=FuelDamageMode.FORCED_DAMAGED),
                RolloutConfig(fuel_damage_mode=FuelDamageMode.FORCED_CLEAN)):
        try:
            cfg.validate()
        except ValueError as exc:
            assert "evaluation pair member" in str(exc) or "pair member" in str(exc)
        else:
            raise AssertionError("a forced mode was accepted as a training mode")

    for kwargs in ({"fuel_damage_probability": 1.5},
                   {"fuel_damage_leg_progress": 0.0},
                   {"fuel_damage_leg_progress": 1.0},
                   {"fuel_damage_rtb_margin": 0.9},
                   {"aircraft_penalty_coeff": -1.0}):
        for cfg in (TrainConfig(n_iterations=1, **kwargs), RolloutConfig(**kwargs)):
            try:
                cfg.validate()
            except ValueError:
                pass
            else:
                raise AssertionError("%s accepted %r" % (type(cfg).__name__, kwargs))


def test_degenerate_difficulty_settings_warn_but_do_not_raise() -> None:
    """A researcher may deliberately probe a factor-free cell; it must be visible."""
    for kwargs, needle in (({"fuel_damage_mode": FuelDamageMode.OFF}, "DISABLED"),
                           ({"fuel_damage_probability": 0.0}, "SAME condition"),
                           ({"aircraft_penalty_coeff": 0.0}, "costs NOTHING")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            TrainConfig(n_iterations=1, **kwargs).validate()     # must NOT raise
        out = buf.getvalue()
        assert "[WARN]" in out and needle in out, (kwargs, out)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TrainConfig(n_iterations=1).validate()
    assert "[WARN]" not in buf.getvalue(), "the approved cell must be quiet"


def test_the_component_has_no_blade_torch_or_solver_dependency() -> None:
    """The layer stays hand-testable: no engine, no torch, no solver in its closure.

    `pyomo` is exempt and the exemption is CONTROLLED below: importing any `match_aou.*`
    module eagerly pulls in the root package's `from .solvers import MatchAou`, so pyomo
    is present for every module in this project (CLAUDE.md section 8). The control proves
    that is what happened here, so this test starts failing if the root package is ever
    made lazy and this module then acquires a real solver dependency of its own.
    """
    child = (
        "import sys, json, importlib\n"
        "importlib.import_module('match_aou.rl.training.graph_fuel_damage')\n"
        "banned = [m for m in ('blade', 'gymnasium', 'gym', 'torch')\n"
        "          if m in sys.modules or any(k.startswith(m + '.') for k in sys.modules)]\n"
        "print('RESULT:' + json.dumps({'banned': banned,\n"
        "                              'pyomo': any(k.startswith('pyomo') for k in sys.modules)}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True,
        cwd=str(ROOT), env=_child_env(),
    )
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT:"))
    result = json.loads(line[len("RESULT:"):])
    assert result["banned"] == [], (
        "graph_fuel_damage pulled in %r -- the layer must stay pure" % result["banned"]
    )

    # CONTROL: plain `match_aou` already brings pyomo, so its presence above is inherited
    # root-package behaviour and not a dependency this module introduced.
    control = subprocess.run(
        [sys.executable, "-c",
         "import sys, match_aou, json;"
         "print('RESULT:' + json.dumps(any(k.startswith('pyomo') for k in sys.modules)))"],
        capture_output=True, text=True, cwd=str(ROOT), env=_child_env(),
    )
    assert control.returncode == 0, control.stderr
    control_line = next(
        l for l in control.stdout.splitlines() if l.startswith("RESULT:")
    )
    assert json.loads(control_line[len("RESULT:"):]) is True, (
        "the root package no longer imports pyomo -- re-check this module's exemption"
    )


# =============================================================================
# Local drivers (kept below the tests they serve)
# =============================================================================

def _child_env() -> dict:
    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return env


def _write_run_config_to_string(cfg: TrainConfig) -> str:
    """`write_run_config` into a temp dir, returned as text (provenance stubbed out)."""
    import tempfile
    saved = graph_train._git_provenance
    graph_train._git_provenance = lambda repo_root: {
        "repo_root": str(repo_root), "available": True, "commit": "0" * 40,
        "branch": "test", "dirty": False, "dirty_path_count": 0, "reason": None,
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = graph_train.write_run_config(Path(tmp), cfg)
            return path.read_text(encoding="utf-8")
    finally:
        graph_train._git_provenance = saved


def _run_one_episode_against(ctx, *, seed: int, reward_spy=None):
    """Drive the REAL `_run_one_episode` against a stub context.

    Only the BLADE / solver seams are replaced (`setup_episode`, `run_episode`, the
    generator, and optionally the reward). The fuel-damage plan construction, the stage
    attribution and the outcome assembly are production code.
    """
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                      eval_every=0, checkpoint_every=0)

    class _Result:
        trajectory = []
        ticks = 11
        ended = "done"
        n_wakes = 0
        confirmed_kills = 0
        n_dead = 0

    class _Reward:
        reward = -0.25

    class _Path:
        @staticmethod
        def read_text(encoding=None):
            return "{}"

    class _Gen:
        @staticmethod
        def generate(episode, config):
            return _Path()

    saved = {
        "setup_episode": graph_train.setup_episode,
        "run_episode": graph_train.run_episode,
        "compute_episode_reward": graph_train.compute_episode_reward,
    }
    graph_train.setup_episode = lambda *a, **k: ctx
    graph_train.run_episode = lambda *a, **k: _Result()
    graph_train.compute_episode_reward = (
        reward_spy if reward_spy is not None else (lambda *a, **k: _Reward())
    )
    try:
        return graph_train._run_one_episode(
            None, _Gen(), cfg, seed=seed, episode_tag=0, deterministic=False,
        )
    finally:
        for name, original in saved.items():
            setattr(graph_train, name, original)


class _StubTransition:
    def __init__(self, ego_id="ego0", meta_action=0):
        self.ego_id = ego_id
        self.tick = 1
        self.meta_action = int(meta_action)
        self.node_v = 0
        self.log_prob = 0.0
        self.entropy = 0.0
        self.gobs = None
        self.reward = None


class _StubUpdater:
    def __init__(self, policy, ppo):
        self.cfg = ppo
        self.optimizer = torch.optim.Adam(policy.encoder.parameters(), lr=1e-4)

    def update(self, buf):
        n_eps = len(getattr(buf, "episodes", []) or [])
        return {
            "policy_loss": 0.0, "total_loss": 0.0, "entropy": 0.0, "mean_ratio": 1.0,
            "clip_fraction": 0.0, "approx_kl": 0.0, "max_ratio_dev": 0.0,
            "grad_norm": 0.0, "adv_std_raw": 0.0, "n_transitions": 0,
            "n_episodes": n_eps, "episodes_with_wakes": 0,
            "n_epochs_run": 1 if n_eps else 0, "baseline": -0.5,
        }


def _run_stub_training(cfg: TrainConfig, *, failing_seeds=(), failing_eval=None):
    """Drive the REAL `train()` with the BLADE + solver episode body stubbed.

    Everything under test stays real: the loop, the seed and pair schedule, the tag
    allocation, the ledger, the condition accounting and the record writers.
    ``failing_eval`` is ``(seed_or_"*", mode)`` -- the eval member that raises.
    """
    events = []
    failing_seeds = set(failing_seeds)
    fail_seed, fail_mode = failing_eval if failing_eval else (None, None)

    saved = {
        "_run_one_episode": graph_train._run_one_episode,
        "_build_generator": graph_train._build_generator,
        "PPOUpdater": graph_train.PPOUpdater,
        "_git_provenance": graph_train._git_provenance,
    }
    state = {"scen_dir": None}

    def fake_build_generator(scen_dir):
        state["scen_dir"] = Path(scen_dir)
        return object()

    def fake_run_one_episode(policy, gen, cfg_, *, seed, episode_tag, deterministic,
                             fuel_damage_mode=None):
        phase = "eval" if deterministic else "train"
        events.append(("episode", phase, int(seed), int(episode_tag), fuel_damage_mode))
        if state["scen_dir"] is not None:
            state["scen_dir"].mkdir(parents=True, exist_ok=True)
            (state["scen_dir"] / ("episode_%04d_scenario.json" % int(episode_tag))
             ).write_text(json.dumps({"tag": int(episode_tag)}), encoding="utf-8")
        if phase == "train" and seed in failing_seeds:
            raise EpisodeAttemptError("setup", FuelDamageError("no strict window"))
        if phase == "eval" and fail_mode is not None and fuel_damage_mode == fail_mode \
                and (fail_seed == "*" or seed == fail_seed):
            raise EpisodeAttemptError("setup", FuelDamageError("no strict window"))

        params = cfg_.fuel_damage_parameters(fuel_damage_mode)
        condition = resolve_condition(episode_seed=seed, params=params)
        damaged = condition == CONDITION_DAMAGED
        return graph_train._EpisodeOutcome(
            trajectory=[_StubTransition()], reward=-0.5 + (0.1 if damaged else 0.0),
            ticks=42, ended="done", n_wakes=1, confirmed_kills=1, n_dead=0,
            seconds=0.01, targets_confirmed_unique=1, targets_total=6,
            known_target_names=("A",), hidden_target_names=("B",),
            known_confirmed_names=("A",), hidden_confirmed_names=(),
            fuel_damage_plan={"condition": condition, "ego_id": "ego0" if damaged else None},
            fuel_damage_outcome={"condition": condition, "fired": damaged,
                                 "wake_occurred": damaged, "wake_meta_action": None},
            selected_ego_rtb_issued=True if damaged else None,
        )

    graph_train._git_provenance = lambda repo_root: {
        "repo_root": str(repo_root), "available": True, "commit": "0" * 40,
        "branch": "test", "dirty": False, "dirty_path_count": 0, "reason": None,
    }
    graph_train._run_one_episode = fake_run_one_episode
    graph_train._build_generator = fake_build_generator
    graph_train.PPOUpdater = _StubUpdater
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            summary = graph_train.train(cfg)
    finally:
        for name, original in saved.items():
            setattr(graph_train, name, original)
    return summary, events


# =============================================================================
# Standalone runner (pytest is absent in nlp_env)
# =============================================================================

if __name__ == "__main__":
    import inspect
    import tempfile

    tests = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        needs_tmp = "tmp_path" in inspect.signature(fn).parameters
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as tmp:
                    fn(Path(tmp))
            else:
                fn()
        except Exception as exc:  # noqa: BLE001 - a standalone runner reports, not raises
            failed += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
            import traceback
            traceback.print_exc()
        else:
            print("OK   %s" % name)
    print("-" * 72)
    print("%d passed, %d failed (of %d)" % (len(tests) - failed, failed, len(tests)))
    sys.exit(1 if failed else 0)
