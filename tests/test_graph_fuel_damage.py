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

F1   THE RTB MEASUREMENT IS COMMAND HISTORY, NOT THE EXECUTOR LIFECYCLE LATCH
     `rtb_command_for` is proven byte-equal to what a real `GraphPlanExecutor` emits; a
     controlled abort produces exactly one such command and reports True; and an ego that
     dies flying PLAN_COMPLIANCE reports False WHILE the executor's `rtb_issued` latch is
     True for it -- which is what proves the measurement does not reuse that field.

F2   THE WINDOW IS RE-VALIDATED AGAINST THE LIVE EVENT STATE
     The stub environment is a faithful miniature of
     `Game.update_all_aircraft_position` -- per-tick `fuel_rate / 3600` burn including
     route-less launch ticks, movement derived from the aircraft's own knots speed, and
     removal at empty -- so live fuel at the event really is below
     `projected_fuel_at_event`. All three live checks are exercised, and each refusal is
     proven to leave `current_fuel` untouched and the event unfired.

PO3  BEHAVIOURAL AND MEASUREMENT INTEGRATION
     P3.1  a controlled SELF_PRESERVATION_ABORT is EGO-GLOBAL -- it empties the damaged
           ego's WHOLE remaining plan (never only the selected node's assignment) and
           only that ego's, producing exactly ONE existing RTB command (the executor's
           latch, unchanged). P3.1b proves the pure effect-layer scope and isolation;
           P3.1c drives the real `_wake_decision` -> resync -> RTB chain end to end;
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
import hashlib
import inspect
import io
import json
import math
import random
import subprocess
import sys
from dataclasses import replace
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
from match_aou.rl.action.graph_action import (  # noqa: E402
    NUM_META_ACTIONS,
    MetaAction,
    build_action_mask,
)
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
from match_aou.rl.observation import graph_builder  # noqa: E402
from match_aou.rl.observation.graph_builder import (  # noqa: E402
    TASK_FEATURE_DIM,
)
from match_aou.rl.training import (  # noqa: E402
    graph_fuel_damage,
    graph_tick_loop,
    graph_train,
)
from match_aou.rl.training import graph_hidden_placement  # noqa: E402
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    CERTIFICATE_TICK_TOLERANCE,
    CONDITION_CLEAN,
    CONDITION_DAMAGED,
    FD_ELIGIBILITY_CERTIFIED_V1,
    FD_ELIGIBILITY_LEGACY_V1,
    FD_ELIGIBILITY_POLICIES,
    FD_ELIGIBILITY_REJECTION_REASONS,
    FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN,
    KILOMETERS_TO_NAUTICAL_MILES,
    NAUTICAL_MILES_TO_METERS,
    NO_FD_ELIGIBLE_EGO,
    POST_FD_DEACTIVATED_DEAD,
    POST_FD_DEACTIVATED_RTB,
    POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
    POST_FD_WAKE_POLICIES,
    POST_FD_WAKE_SINGLE_V1,
    REASON_INVALID_BAND,
    REASON_NO_ROUTE,
    REASON_PRE_EVENT_ASSIGNMENT_BOUNDARY,
    REASON_PRE_EVENT_POPUP_RISK,
    WAYPOINT_SNAP_KM,
    FuelDamageIntegrityError,
    certify_fd_candidate,
    derive_fuel_damage_eligibility_seed,
    eligibility_ordinal_permutation,
    engine_leg_distance_km,
    predict_leg_states,
    SEVERITIES,
    SEVERITY_MILD,
    SEVERITY_SEVERE,
    TARGET_POLICY_LIVE_SEVERITY_MIDPOINT,
    TARGET_POLICY_PLANNED_MIDPOINT,
    FuelDamageController,
    FuelDamageError,
    FuelDamageMode,
    FuelDamageParameters,
    FuelDamagePlan,
    build_fuel_damage_controller,
    build_fuel_damage_plan,
    derive_fuel_damage_seed,
    derive_fuel_damage_severity_seed,
    fuel_for_distance_km,
    interpolate_great_circle,
    measure_window,
    plan_fuel_damage,
    resolve_condition,
    resolve_severity,
    rtb_command_for,
    severity_band,
)
from match_aou.rl.training.graph_rollout import RolloutConfig  # noqa: E402
from match_aou.rl.training.graph_train import (  # noqa: E402
    EpisodeAttemptError,
    MeasurementIntegrityError,
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

    def is_done(self, _observation):
        # Mirrors the production signature: completion is decided from the live
        # observation, never from `rtb_issued`. This stub never completes, so these
        # fixtures always run their full tick budget.
        return False


class _StubEnv:
    """A faithful miniature of ``Game.update_all_aircraft_position``.

    Two properties are modelled deliberately, because the fuel-window logic depends on
    both and a convenient stub would hide them:

      * FUEL IS BURNED EVERY TICK, `fuel_rate / 3600`, whether or not the aircraft moved
        -- the engine decrements unconditionally, and `hold_ticks` reproduces the real
        launch tick, where the aircraft is already airborne but has no route yet. This is
        exactly why live fuel at the event is BELOW `projected_fuel_at_event`, which
        subtracts only the fuel for distance flown.
      * AN AIRCRAFT AT `current_fuel <= 0` IS REMOVED from `scenario.aircraft`, as
        `Game.remove_aircraft` does -- which is how the executor learns an ego is dead.

    DISTANCE PER TICK IS DERIVED FROM THE AIRCRAFT'S OWN KNOTS SPEED, exactly as
    `get_next_coordinates` does: `speed * NAUTICAL_MILES_TO_METERS / 3600` metres per
    tick. That is what makes the burn and the movement CONSISTENT -- flying d km costs
    exactly `fuel_for_distance_km(d)` -- which in turn is what lets a test state the true
    relationship between live fuel and `projected_fuel_at_event` instead of a relationship
    manufactured by a compressed timeline.
    """

    def __init__(self, scenario, *, step_km=None, targets=None, hold_ticks=0,
                 burn_fuel=True):
        self.scenario = scenario
        self.step_km = None if step_km is None else float(step_km)
        self.targets = dict(targets or {})
        self.hold_ticks = int(hold_ticks)
        self.burn_fuel = bool(burn_fuel)
        self.closed = False
        self.n_steps = 0
        self.removed = []

    def _step_km_of(self, aircraft) -> float:
        if self.step_km is not None:
            return self.step_km
        return float(aircraft.speed) * NAUTICAL_MILES_TO_METERS / 3600.0 / 1000.0

    def step(self, _action):
        moving = self.n_steps >= self.hold_ticks
        self.n_steps += 1
        for aircraft in list(self.scenario.aircraft):
            target = self.targets.get(str(aircraft.id))
            step_km = self._step_km_of(aircraft)
            if moving and target is not None:
                here = Location(aircraft.latitude, aircraft.longitude)
                if here.distance_to(target) <= step_km:
                    aircraft.latitude = target.latitude
                    aircraft.longitude = target.longitude
                else:
                    moved = _point_at(here, step_km, _bearing(here, target))
                    aircraft.latitude, aircraft.longitude = moved.latitude, moved.longitude
            if self.burn_fuel:
                aircraft.current_fuel -= aircraft.fuel_rate / 3600.0
                if aircraft.current_fuel <= 0:
                    self.scenario.aircraft.remove(aircraft)
                    self.removed.append(str(aircraft.id))
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
        # The RAW t=0 world snapshots setup takes BEFORE either solve. This fixture's
        # world is entirely known -- `n_agents` targets, no constructed hidden half -- so
        # the two snapshots coincide here. They are still kept separate from
        # `oracle_tasks`, which is an ALLOCATION and only happens to cover the same set.
        self.known_target_ids = tuple("tgt%d" % i for i in range(n_agents))
        self.executed_target_ids = tuple(self.known_target_ids)
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

    burn = aircraft.fuel_rate / 3600.0
    n_ticks = 250          # the faithful timeline reaches 30% of a 250 km leg at ~t=112
    fires, fuels = [], []
    for tick in range(n_ticks):
        fired = controller.maybe_apply(ctx.scenario, tick)
        if fired is not None:
            fires.append((tick, fired))
        fuels.append(aircraft.current_fuel)
        ctx.env.step([])

    assert len(fires) == 1, "the event fired %d time(s): %r" % (len(fires), fires)
    fire_tick, fired_ego = fires[0]
    assert fired_ego == ego

    # Exactly ONE non-burn change of the real value over 120 ticks. The env burns
    # `fuel_rate / 3600` every tick like the engine does, so the test separates the two:
    # every step is the burn, EXCEPT the event tick, which is the jump to the planned
    # quantity. A controller that re-applied would show a second jump.
    jumps = [
        i for i in range(1, len(fuels))
        if abs(fuels[i] - (fuels[i - 1] - burn)) > 1e-9
    ]
    assert jumps == [fire_tick], (jumps, fire_tick)
    assert fire_tick > 1, "the event fired before the ego had flown anywhere"
    assert fuels[fire_tick] == plan.post_damage_fuel
    assert fuels[fire_tick - 1] > plan.post_damage_fuel
    assert aircraft.current_fuel < plan.post_damage_fuel  # still burning afterwards
    assert aircraft.current_fuel < launch_fuel

    # It fired at the FIRST tick at or past the threshold, not later.
    outcome = controller.outcome
    assert outcome.fired and outcome.event_tick == fire_tick
    assert outcome.observed_progress >= plan.progress_threshold
    # `fuels[i]` is sampled after `maybe_apply(i)` but before `env.step(i)`, so the fuel
    # the event saw is the previous sample minus that tick's burn.
    assert abs(outcome.fuel_before - (fuels[fire_tick - 1] - burn)) < 1e-9
    assert outcome.fuel_after == plan.post_damage_fuel
    assert abs(outcome.damage_factor
               - plan.post_damage_fuel / outcome.fuel_before) < 1e-12
    assert controller.fired

    # And no peer aircraft was damaged: every peer's fuel is the pure burn schedule.
    for peer in ctx.agent_ids:
        if peer == ego:
            continue
        peer_ac = ctx.scenario.get_aircraft(peer)
        expected = launch_fuel - n_ticks * (peer_ac.fuel_rate / 3600.0)
        assert abs(peer_ac.current_fuel - expected) < 1e-6, peer


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
            max_ticks=300, fuel_damage=controller,
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
    # And on the tick before, every ego saw the SAME pre-damage value -- far above the
    # post-damage level (it is only down by the per-tick burn every aircraft pays).
    before = [row for row in seen if row["tick"] == event_tick - 1]
    assert before, "no tick preceded the event"
    assert len({row["fuels"][ego] for row in before}) == 1, before
    assert before[0]["fuels"][ego] > 10.0 * plan.post_damage_fuel, before[0]["fuels"][ego]

    # P2.2: exactly one ego was flagged, on exactly one tick.
    flagged = [(row["tick"], row["ego"]) for row in seen if row["fuel_damage"]]
    assert flagged == [(event_tick, ego)], flagged
    assert (ego, event_tick) in wakes

    # The wake was attributed back to the event, with the meta-action it produced.
    outcome = controller.outcome
    assert outcome.wake_occurred
    assert outcome.wake_meta_action == int(MetaAction.SELF_PRESERVATION_ABORT)

    # And no peer's fuel was ever JUMPED: each follows the pure per-tick burn schedule
    # every aircraft pays, with no step the event could account for.
    per_tick = {}
    for row in seen:
        per_tick.setdefault(row["tick"], []).append(row["fuels"])
    for peer in ctx.agent_ids:
        if peer == ego:
            continue
        burn = ctx.scenario.get_aircraft(peer).fuel_rate / 3600.0
        series = [snapshots[0][peer] for _tick, snapshots in sorted(per_tick.items())]
        # Every ego on a given tick agreed on this peer's fuel...
        for _tick, snapshots in per_tick.items():
            assert len({snap[peer] for snap in snapshots}) == 1, (peer, _tick)
        # ...and it only ever moved by the burn.
        for i in range(1, len(series)):
            assert abs(series[i] - (series[i - 1] - burn)) < 1e-9, (peer, i)


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
    for tick in range(400):
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
            # A peer's own row is its OWN live fuel (down only by the shared per-tick
            # burn), nowhere near the damaged ego's post-damage level.
            peer_ac = ctx.scenario.get_aircraft(peer)
            assert abs(float(rows[0, 0])
                       - peer_ac.current_fuel / peer_ac.max_fuel) < 1e-6, peer
            assert float(rows[0, 0]) > 10.0 * expected, (
                "peer %s's fuel_norm moved toward the damaged value" % peer
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
    # Inert means no ego was singled out: every aircraft is on the identical per-tick
    # burn schedule and none was set to a damaged level.
    fuels = {round(a.current_fuel, 9) for a in ctx.scenario.aircraft}
    assert len(fuels) == 1, fuels
    only = next(iter(fuels))
    assert 0.0 < only < 12000.0, only

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

# The ABORT fixture, shared by P3.1 and P3.1b/c: an actor holding TWO assignments at
# DIFFERENT levels plus an independent peer. The multi-assignment actor is the whole
# point -- a node-scoped abort would clear only the selected tuple, leave the ego's plan
# non-empty and therefore emit a MOVE instead of the RTB.
_ABORT_EGO, _ABORT_PEER = "ego0", "ego1"


def _abort_tasks():
    """Three attack tasks: nodes 0 and 2 belong to the actor, node 1 to the peer."""
    return [
        _attack_task("tgt0", _point_at(_BASE, 250.0, 30.0)),   # actor, level 0
        _attack_task("tgt1", _point_at(_BASE, 250.0, 90.0)),   # peer
        _attack_task("tgt2", _point_at(_BASE, 260.0, 45.0)),   # actor, level 1
    ]


def _abort_solution():
    """The actor's mission spans two levels; the peer flies its own single task."""
    return {_ABORT_EGO: [(0, 0, 0), (2, 0, 1)], _ABORT_PEER: [(1, 0, 0)]}


def _abort_agent(aid):
    return Agent(location=Location(_BASE.latitude, _BASE.longitude), capabilities=[],
                 budget=1.0, move_cost_function=lambda s, d: 0.0, agent_id=aid,
                 side_color="blue", home_base_id="base-blue",
                 return_location=Location(_BASE.latitude, _BASE.longitude))


def _abort_world(tasks, solution):
    """A real `GraphPlanExecutor` plus both egos airborne, mid-leg, far from target."""
    executor = GraphPlanExecutor(
        tasks=tasks, solution=solution,
        agents=[_abort_agent(_ABORT_EGO), _abort_agent(_ABORT_PEER)],
        arrival_threshold_km=50.0,
    )
    mid = _point_at(_BASE, 75.0, 30.0)
    scenario = _StubScenario(
        aircraft=[_StubAircraft(_ABORT_EGO, mid.latitude, mid.longitude),
                  _StubAircraft(_ABORT_PEER, mid.latitude, mid.longitude)],
        airbases=[_StubAirbase("base-blue", _BASE.latitude, _BASE.longitude)],
    )
    return executor, scenario


class _AbortGobs:
    """The single attribute `apply_meta_action` reads off the observation."""

    task_target_ids = ["tgt0", "tgt1", "tgt2"]


def test_p3_1_abort_empties_only_the_damaged_ego_and_issues_exactly_one_rtb() -> None:
    """P3.1. The existing effect + executor path, exercised with a controlled abort.

    Nothing new is built for RTB: the abort clears the ego's WHOLE remaining plan, the
    ego's plan becomes empty, and `GraphPlanExecutor`'s pre-existing empty-plan branch
    issues its single latched `aircraft_return_to_base`. This test locks that the
    fuel-damage decision reaches BLADE through exactly that path and through no other.

    THE REGRESSION: the actor holds TWO assignments and the loop selects each of its
    legal SELF_PRESERVATION_ABORT cells in turn -- each names only ONE of them. A
    node-scoped abort leaves the other tuple in place, so the plan stays non-empty, the
    executor emits a move and NO RTB is ever issued. Both cells must produce the same
    empty plan and the same single RTB.
    """
    ego, peer = _ABORT_EGO, _ABORT_PEER

    for selected_node in (0, 2):   # both of the actor's legal SPA cells
        tasks = _abort_tasks()
        solution = _abort_solution()
        executor, scenario = _abort_world(tasks, solution)

        # Baseline: neither ego is RTB-ing; both are flying their plan.
        before = executor.next_actions(scenario)
        assert not any("aircraft_return_to_base" in c for c in before), before

        # The abort itself: the SAME pure effect layer a wake would call, on the damaged
        # ego's private belief only.
        new_solution = apply_meta_action(
            solution, _AbortGobs(), ego,
            int(MetaAction.SELF_PRESERVATION_ABORT), selected_node, tasks,
        )
        assert new_solution[ego] == [], (
            "abort on node %d left %r: the effect is EGO-GLOBAL, not node-scoped"
            % (selected_node, new_solution[ego])
        )
        assert new_solution[peer] == [(1, 0, 0)], "the peer's plan was edited"
        assert solution[ego] == [(0, 0, 0), (2, 0, 1)], "apply_meta_action mutated its input"

        executor.resync(new_solution, ego_id=ego, tasks=tasks)
        assert executor.plans[ego] == [], executor.plans[ego]
        assert executor.plans[peer] == [(1, 0, 0)], "the peer's executor slice moved"

        commands = executor.next_actions(scenario)
        rtbs = [c for c in commands if "aircraft_return_to_base" in c]
        assert rtbs == ["aircraft_return_to_base('%s')" % ego], (selected_node, commands)
        # ...and nothing stale is still being flown for the aborted ego.
        assert not any(c.startswith("move_aircraft('%s'" % ego)
                       or c.startswith("handle_aircraft_attack('%s'" % ego)
                       for c in commands), commands
        assert executor.rtb_issued.get(ego) is True
        assert executor.rtb_issued.get(peer) is not True

        # EXACTLY one: the single-issue latch means a second tick emits no second RTB
        # (`aircraft_return_to_base` is a TOGGLE in BLADE -- a second one would cancel it).
        again = executor.next_actions(scenario)
        assert not any("aircraft_return_to_base" in c for c in again), again


def test_p3_1b_abort_is_ego_global_and_every_peer_slice_is_untouched() -> None:
    """P3.1b (PO1). The pure effect layer: ego-global scope + private isolation.

    Two peers, one of them holding several assignments of its own, so "the acting ego's
    whole slice and nothing else" is a real claim rather than an artefact of a two-entry
    dict. The `k x 3` selection surface is asserted UNCHANGED alongside it: the actor
    still has one legal abort CELL PER assigned node -- which is exactly why selected-node
    independence has to be proven rather than assumed.
    """
    ego = _ABORT_EGO
    tasks = _abort_tasks()
    tasks_snapshot = list(tasks)
    solution = {
        ego: [(0, 0, 0), (2, 0, 1)],
        "peer_a": [(1, 0, 0)],
        "peer_b": [(1, 0, 2), (2, 0, 2)],
    }
    snapshot = {k: list(v) for k, v in solution.items()}

    # --- the k x 3 selection surface, from the REAL builder + REAL mask -------------
    mid = _point_at(_BASE, 75.0, 30.0)
    scenario = _StubScenario(
        aircraft=[_StubAircraft(ego, mid.latitude, mid.longitude),
                  _StubAircraft("peer_a", mid.latitude, mid.longitude),
                  _StubAircraft("peer_b", mid.latitude, mid.longitude)],
        airbases=[_StubAirbase("base-blue", _BASE.latitude, _BASE.longitude)],
    )
    gobs = build_graph_observation(
        scenario=scenario, agent_id=ego, current_plan=solution[ego], current_time=0,
        tasks=tasks, solution=solution, precedence_relations=[],
        config=GraphObservationConfig(detection_range_km=50.0),
    )
    mask = build_action_mask(gobs)
    assert NUM_META_ACTIONS == 3, NUM_META_ACTIONS
    assert mask.shape == (len(tasks), NUM_META_ACTIONS), mask.shape
    spa = int(MetaAction.SELF_PRESERVATION_ABORT)
    legal_cells = [v for v in range(len(tasks)) if math.isfinite(float(mask[v, spa]))]
    assert legal_cells == [0, 2], (
        "abort must stay legal on exactly the ego's own assigned nodes: %r" % (legal_cells,)
    )
    assert all(math.isfinite(float(mask[v, int(MetaAction.PLAN_COMPLIANCE)]))
               for v in range(len(tasks))), mask

    # --- every legal cell produces the SAME ego-global result -----------------------
    results = [
        apply_meta_action(solution, gobs, ego,
                          int(MetaAction.SELF_PRESERVATION_ABORT), v, tasks)
        for v in legal_cells
    ]
    for v, out in zip(legal_cells, results):
        assert out[ego] == [], (
            "abort on node %d left %r; the effect must be ego-global" % (v, out[ego])
        )
        assert out["peer_a"] == [(1, 0, 0)], (v, out["peer_a"])
        assert out["peer_b"] == [(1, 0, 2), (2, 0, 2)], (v, out["peer_b"])
    assert results[0] == results[1], (results[0], results[1])

    # --- purity: the input solution and the task list are untouched -----------------
    assert solution == snapshot, solution
    assert tasks == tasks_snapshot and all(a is b for a, b in zip(tasks, tasks_snapshot))

    # --- the bounds guard is unchanged ---------------------------------------------
    for bad_v in (-1, len(tasks)):
        try:
            apply_meta_action(solution, gobs, ego,
                              int(MetaAction.SELF_PRESERVATION_ABORT), bad_v, tasks)
        except ValueError:
            pass
        else:
            raise AssertionError("out-of-range node_v=%r must still raise" % (bad_v,))


def test_p3_1c_a_real_wake_decision_aborts_the_whole_ego_and_yields_one_rtb() -> None:
    """P3.1c (PO2). The REAL wake -> effect -> resync -> single-RTB chain.

    `graph_tick_loop._wake_decision` is the production function under test: it builds the
    real `GraphObservation`, applies the real `build_action_mask`, decodes the cell
    through the real `sample_action`, calls the real `apply_meta_action` and the real
    `GraphPlanExecutor.resync`. ONLY the encoder and the head are stubbed, and only to
    make the sampled cell deterministic -- they contribute nothing to WHAT the chosen
    cell does. Both of the actor's legal abort cells are driven, and the selected cell
    names only one of its two assignments.
    """
    ego, peer = _ABORT_EGO, _ABORT_PEER
    cfg = GraphObservationConfig(detection_range_km=50.0)

    class _ForcedPolicy:
        """Puts all the mass on ONE (node, meta) cell; `deterministic=True` takes it."""

        def __init__(self, node_v, meta_action):
            self.node_v, self.meta_action = int(node_v), int(meta_action)

        def encoder(self, gobs):
            return torch.zeros((int(gobs.task_features.shape[0]), 4), dtype=torch.float32)

        def head(self, emb):
            logits = torch.zeros((int(emb.shape[0]), NUM_META_ACTIONS), dtype=torch.float32)
            logits[self.node_v, self.meta_action] = 10.0
            return logits

    for selected_node in (0, 2):
        tasks = _abort_tasks()
        solution = _abort_solution()
        executor, scenario = _abort_world(tasks, solution)
        beliefs = {aid: _StubBelief(list(tasks), {k: list(v) for k, v in solution.items()})
                   for aid in (ego, peer)}
        peer_solution_before = {k: list(v) for k, v in beliefs[peer].solution.items()}

        # Pre-wake: the actor is flying its plan, nobody is RTB-ing.
        before = executor.next_actions(scenario)
        assert not any("aircraft_return_to_base" in c for c in before), before

        transition = graph_tick_loop._wake_decision(
            _ForcedPolicy(selected_node, int(MetaAction.SELF_PRESERVATION_ABORT)),
            ego, scenario, beliefs[ego], executor, cfg, tick=7, deterministic=True,
        )
        # The real `sample_action` really decoded the forced abort cell.
        assert transition.meta_action == int(MetaAction.SELF_PRESERVATION_ABORT), transition
        assert transition.node_v == selected_node, transition
        assert transition.ego_id == ego and transition.tick == 7

        # BEFORE Phase 2: belief and executor slice are both empty for the actor only.
        assert beliefs[ego].solution[ego] == [], beliefs[ego].solution
        assert executor.plans[ego] == [], executor.plans[ego]
        assert beliefs[ego].solution[peer] == [(1, 0, 0)], beliefs[ego].solution
        assert executor.plans[peer] == [(1, 0, 0)], executor.plans
        assert beliefs[peer].solution == peer_solution_before, beliefs[peer].solution
        assert executor.tasks[peer] == tasks, "the peer's task-view moved"

        # Phase 2: exactly one RTB, for the actor, and nothing stale for it.
        commands = executor.next_actions(scenario)
        assert [c for c in commands if "aircraft_return_to_base" in c] == [
            "aircraft_return_to_base('%s')" % ego], (selected_node, commands)
        assert not any(c.startswith("move_aircraft('%s'" % ego)
                       or c.startswith("handle_aircraft_attack('%s'" % ego)
                       for c in commands), commands
        # The peer was NOT sent home by the actor's abort -- it is still flying its plan.
        assert any(c.startswith("move_aircraft('%s'" % peer) for c in commands), commands
        assert executor.rtb_issued.get(peer) is not True

        # The single-issue latch: no second RTB toggle on the next call.
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
# F1 -- the RTB measurement is COMMAND HISTORY, not the executor lifecycle latch
# =============================================================================
#
# The defect: `selected_ego_rtb_issued` was read off `GraphPlanExecutor.rtb_issued`.
# That field is not a record of commands. `_command_for_ego`'s dead branch sets it True
# for an ego that is neither airborne nor in an airbase -- precisely BECAUSE no command
# was emitted -- so an ego that flew its plan into the ground registered as an RTB *and*
# as a death, in the same episode, in the aggregate that is supposed to show whether the
# fuel-damage event produced an abort.


def _executor_ctx(*, meta_action, target_distance_km=250.0, kill_selected_at=None,
                  seed=1):
    """A context whose executor is a REAL `GraphPlanExecutor`, driven by `run_episode`.

    Only `_wake_decision` is replaced, and the replacement does what the real one does
    for the plan: `apply_meta_action` on the ego's own belief, then `executor.resync` of
    that ego's slice. The encoder and head are irrelevant to whether a command is emitted,
    and stubbing them keeps this solver-free and deterministic.

    `kill_selected_at` removes the selected ego's aircraft from `scenario.aircraft` after
    that tick -- the engine's own `remove_aircraft` behaviour on an empty tank -- which is
    how the "died without ever issuing an RTB" case is produced.
    """
    ctx = _FuelDamageCtx(target_distance_km=target_distance_km)
    agents = ctx.agents
    tasks = ctx.beliefs[ctx.agent_ids[0]].tasks
    ctx.executor = GraphPlanExecutor(
        tasks=tasks, solution=ctx.a_init, agents=agents, arrival_threshold_km=50.0,
    )
    # No fuel burn: this fixture is about COMMANDS, and a mid-episode fuel-out would
    # confound "the ego died before it could RTB" with the case being constructed.
    ctx.env = _StubEnv(ctx.scenario, targets=ctx.targets, burn_fuel=False)
    controller = build_fuel_damage_controller(ctx, episode_seed=seed, params=_PARAMS)
    selected = controller.plan.ego_id

    if kill_selected_at is not None:
        real_step = ctx.env.step

        def killing_step(action):
            result = real_step(action)
            if ctx.env.n_steps == int(kill_selected_at):
                victim = ctx.scenario.get_aircraft(selected)
                if victim is not None:
                    ctx.scenario.aircraft.remove(victim)
            return result

        ctx.env.step = killing_step

    def acting_wake(_policy, ego_id, _obs, belief, executor, _cfg, tick, **_kw):
        class _Gobs:
            task_target_ids = [
                str(t.steps[0].target_id) for t in belief.tasks
            ]
        node_v = int(belief.solution.get(str(ego_id), [(0, 0, 0)])[0][0])
        belief.solution = apply_meta_action(
            belief.solution, _Gobs(), str(ego_id), int(meta_action), node_v, belief.tasks
        )
        executor.resync(belief.solution, ego_id=str(ego_id), tasks=belief.tasks)
        return graph_tick_loop.Transition(
            gobs=None, ego_id=str(ego_id), tick=int(tick),
            meta_action=int(meta_action), node_v=node_v, log_prob=0.0, entropy=0.0,
        )

    issued = []
    real_next = ctx.executor.next_actions

    def spy_next(observation):
        commands = real_next(observation)
        issued.extend(commands)
        return commands

    ctx.executor.next_actions = spy_next

    saved = graph_tick_loop._wake_decision
    graph_tick_loop._wake_decision = acting_wake
    try:
        result = graph_tick_loop.run_episode(
            None, ctx, GraphObservationConfig(detection_range_km=50.0),
            max_ticks=250, fuel_damage=controller,
        )
    finally:
        graph_tick_loop._wake_decision = saved
    return ctx, controller, issued, result


def test_f1_the_rtb_command_string_mirrors_the_real_executor() -> None:
    """F1. `rtb_command_for` is byte-equal to what a real executor emits.

    It is a deliberate second copy of the executor's format string (importing the BLADE
    translation layer would cost the component its purity), so the equivalence is
    test-enforced here -- the same discipline `derived_split` uses against `split_tasks`.
    """
    ego = "ego0"
    agent = Agent(location=Location(_BASE.latitude, _BASE.longitude), capabilities=[],
                  budget=1.0, move_cost_function=lambda s, d: 0.0, agent_id=ego,
                  side_color="blue", home_base_id="base-blue",
                  return_location=Location(_BASE.latitude, _BASE.longitude))
    executor = GraphPlanExecutor(tasks=[], solution={ego: []}, agents=[agent],
                                arrival_threshold_km=50.0)
    mid = _point_at(_BASE, 75.0, 30.0)
    scenario = _StubScenario(
        aircraft=[_StubAircraft(ego, mid.latitude, mid.longitude)],
        airbases=[_StubAirbase("base-blue", _BASE.latitude, _BASE.longitude)],
    )
    commands = executor.next_actions(scenario)   # empty plan -> RTB
    assert commands == [rtb_command_for(ego)], (commands, rtb_command_for(ego))


def test_f1_a_controlled_abort_reports_one_real_rtb_command() -> None:
    """F1 (1). Abort -> exactly one emitted RTB command -> `rtb_command_issued is True`."""
    ctx, controller, issued, _result = _executor_ctx(
        meta_action=MetaAction.SELF_PRESERVATION_ABORT
    )
    plan = controller.plan
    assert plan.condition == CONDITION_DAMAGED
    ego = plan.ego_id
    assert controller.outcome.fired, "the event never fired"

    wanted = rtb_command_for(ego)
    assert issued.count(wanted) == 1, (
        "expected exactly one RTB command for the selected ego, got %d in %r"
        % (issued.count(wanted), [c for c in issued if "return_to_base" in c])
    )
    assert controller.outcome.rtb_command_issued is True
    assert controller.outcome.wake_meta_action == int(
        MetaAction.SELF_PRESERVATION_ABORT)
    # Only the damaged ego's plan was emptied; the peers still fly theirs.
    assert ctx.executor.plans[ego] == []
    for peer in ctx.agent_ids:
        if peer != ego:
            assert ctx.executor.plans[peer], peer


def test_f1_a_dead_ego_reports_no_rtb_even_though_the_latch_is_set() -> None:
    """F1 (2) + (3). THE regression: death without a command is not an RTB.

    The selected ego flies PLAN_COMPLIANCE and is removed from the scenario shortly after
    the event -- the engine's behaviour on an empty tank. It therefore never emits an
    `aircraft_return_to_base`, but the executor's dead branch DOES latch
    `rtb_issued[ego] = True`. The measurement must report False, and the latch being True
    at the same time is what proves it is not the source.
    """
    ctx, controller, issued, _result = _executor_ctx(
        meta_action=MetaAction.PLAN_COMPLIANCE, kill_selected_at=150
    )
    plan = controller.plan
    ego = plan.ego_id
    assert controller.outcome.fired, "the event never fired"

    # (2) no RTB command was ever emitted for this ego, and the measurement says so.
    assert rtb_command_for(ego) not in issued, [
        c for c in issued if "return_to_base" in c
    ]
    assert controller.outcome.rtb_command_issued is False

    # (3) ...while the executor latch IS True, for the dead ego, having emitted nothing.
    assert ego in ctx.executor.dead, ctx.executor.dead
    assert ctx.executor.rtb_issued.get(ego) is True, (
        "the fixture no longer reproduces the latch state the defect depended on"
    )
    # Which is exactly the contradiction the old derivation produced: RTB *and* death.
    assert ctx.executor.rtb_issued.get(ego) != controller.outcome.rtb_command_issued


def test_f1_the_trainer_reports_the_command_not_the_latch() -> None:
    """F1. The value that reaches `_EpisodeOutcome` and the aggregates is the command.

    Driven through the REAL `_run_one_episode` against a context whose executor latch is
    True for an ego that emitted nothing -- the exact state the old code misread.
    """
    ctx = _FuelDamageCtx()
    plan = build_fuel_damage_plan(ctx, episode_seed=1, params=_PARAMS)
    ctx.executor.rtb_issued = {str(plan.ego_id): True}   # latched, no command emitted

    out = _run_one_episode_against(ctx, seed=1)
    assert out.fuel_damage_plan["condition"] == CONDITION_DAMAGED
    assert out.selected_ego_rtb_issued is False, (
        "the trainer read the executor latch instead of the emitted command"
    )
    assert out.fuel_damage_outcome["rtb_command_issued"] is False

    # And the tally counts commands, so this episode contributes no RTB. The episode was
    # SCHEDULED damaged and executed damaged, so it passes the scheduled-vs-executed
    # guard and is folded in normally.
    tally = graph_train._ConditionTally()
    tally.attempt(CONDITION_DAMAGED)
    tally.success(out, expected_cell=CONDITION_DAMAGED)
    assert tally.to_record()["fuel_damage_rtb_issued"] == 0


# =============================================================================
# F2 -- the window is re-validated against the LIVE event state
# =============================================================================
#
# The defect: the preflight projection subtracts fuel for distance FLOWN, but
# `Game.update_all_aircraft_position` burns `fuel_rate / 3600` every tick regardless --
# including the launch tick, where the aircraft is airborne with no route. Live fuel at
# the event is therefore always below `projected_fuel_at_event`, and `maybe_apply` only
# checked `fuel_before > post_damage_fuel`. An ego could be below its LIVE continue
# requirement -- i.e. already unable to finish and get home -- and the event would still
# fire, producing an episode that looks like a decision and is not one.


def _fly_to_threshold(ctx, controller, *, max_ticks=400):
    """Step the faithful env until the selected ego crosses the progress threshold."""
    for _ in range(max_ticks):
        if (controller.observed_progress(ctx.scenario) or 0.0) >= float(
                controller.plan.progress_threshold):
            return True
        ctx.env.step([])
    return False


def test_f2_live_fuel_at_the_event_is_below_the_projection() -> None:
    """F2. The discrepancy the fix is about is real, and the live bounds are recorded.

    The env burns exactly what the engine burns, including three hold ticks that stand in
    for the real launch tick (airborne, no route). The event still fires -- the live
    window holds with margin on this cell -- and what is locked here is that the numbers
    the mutation was validated against are the LIVE ones, recorded apart from the planned
    ones so the two can never be confused.
    """
    ctx = _FuelDamageCtx()
    ctx.env = _StubEnv(ctx.scenario, targets=ctx.targets, hold_ticks=3)
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    plan = controller.plan
    aircraft = ctx.scenario.get_aircraft(plan.ego_id)

    tick = 0
    fired = None
    while fired is None and tick < 200:
        fired = controller.maybe_apply(ctx.scenario, tick)
        if fired is None:
            ctx.env.step([])
            tick += 1
    assert fired == plan.ego_id, "the event never fired"

    outcome = controller.outcome
    # THE defect's premise: live fuel is strictly below the preflight projection, because
    # the projection charges distance only and the engine also charged the hold ticks.
    assert outcome.fuel_before < plan.projected_fuel_at_event, (
        outcome.fuel_before, plan.projected_fuel_at_event
    )

    # The live bounds are recorded, are real numbers, and are the ones the four checks
    # were applied to.
    assert outcome.live_rtb_fuel_floor is not None
    assert outcome.live_continue_fuel_requirement is not None
    assert outcome.live_rtb_fuel_floor <= plan.post_damage_fuel
    assert plan.post_damage_fuel < outcome.live_continue_fuel_requirement
    assert outcome.fuel_before >= outcome.live_continue_fuel_requirement
    # Planned and live are DIFFERENT quantities and are reported under different names.
    assert outcome.live_rtb_distance_km != plan.rtb_distance_km
    assert aircraft.current_fuel == plan.post_damage_fuel


def test_f2_the_event_refuses_a_state_already_below_the_live_continue_requirement() -> None:
    """F2. THE required case: above `post_damage_fuel`, below the live continue need.

    That state means the ego could no longer have finished its route and come home even
    without the damage, so the event would create no decision. The old check
    (`fuel_before > post_damage_fuel`) passes here and would have fired. The event must
    refuse it, and must refuse it WITHOUT MUTATING anything.
    """
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    plan = controller.plan
    aircraft = ctx.scenario.get_aircraft(plan.ego_id)
    assert _fly_to_threshold(ctx, controller)

    live = controller.live_bounds(aircraft)
    # Strictly inside (post_damage_fuel, live continue requirement): the old guard is
    # satisfied, the new one is not.
    starved = 0.5 * (plan.post_damage_fuel + live.continue_fuel_requirement)
    assert plan.post_damage_fuel < starved < live.continue_fuel_requirement
    aircraft.current_fuel = starved

    try:
        controller.maybe_apply(ctx.scenario, 99)
    except FuelDamageError as exc:
        assert "below the LIVE continue requirement" in str(exc), str(exc)
    else:
        raise AssertionError(
            "the event fired from a state that was already infeasible to continue from"
        )
    # NOTHING was mutated, and the event remains unfired (so it is not silently spent).
    assert aircraft.current_fuel == starved
    assert not controller.fired
    assert controller.outcome.fired is False
    assert controller.outcome.live_rtb_fuel_floor is None


def test_f2_the_event_refuses_a_state_where_rtb_is_no_longer_affordable() -> None:
    """F2. The target must still cover the LIVE RTB floor, measured where the ego IS.

    Constructed faithfully: the ego overshoots far past its target (progress is well past
    the threshold, but it is now much further from home than the plan assumed), so the
    live RTB floor rises above the planned post-damage quantity. Applying the event there
    would be a kill, not a decision.
    """
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    plan = controller.plan
    aircraft = ctx.scenario.get_aircraft(plan.ego_id)

    # Far BEYOND the target on the same bearing: progress is comfortably past the
    # threshold, but the distance home is now ~412 km instead of the planned 75 km.
    target = plan.first_target_location
    beyond = _point_at(target, 0.65 * plan.leg_length_km,
                       _bearing(Location(_BASE.latitude, _BASE.longitude), target))
    aircraft.latitude, aircraft.longitude = beyond.latitude, beyond.longitude
    before = aircraft.current_fuel
    assert (controller.observed_progress(ctx.scenario) or 0.0) >= plan.progress_threshold

    live = controller.live_bounds(aircraft)
    assert live.rtb_fuel_floor > plan.post_damage_fuel, live

    try:
        controller.maybe_apply(ctx.scenario, 77)
    except FuelDamageError as exc:
        assert "below the LIVE RTB floor" in str(exc), str(exc)
    else:
        raise AssertionError("the event fired where the ego could no longer fly home")
    assert aircraft.current_fuel == before
    assert not controller.fired


def test_f2_the_event_refuses_a_target_that_still_affords_continuing() -> None:
    """F2. If the target no longer sits below the LIVE continue requirement, refuse.

    A plan whose `post_damage_fuel` is above the live requirement describes a "damage"
    the ego can simply absorb -- it would fly its route and come home regardless, and the
    episode would carry a damaged label with no decision in it. Built by hand because the
    reference geometry cannot reach this state naturally: with a single-target route the
    live continue distance is bounded below by the return leg, so the planned midpoint is
    always under it.
    """
    ctx = _FuelDamageCtx()
    base_plan = build_fuel_damage_plan(ctx, episode_seed=1, params=_PARAMS)
    inflated = replace(
        base_plan,
        post_damage_fuel=base_plan.continue_fuel_requirement * 10.0,
    )
    controller = FuelDamageController(inflated)
    aircraft = ctx.scenario.get_aircraft(inflated.ego_id)
    assert _fly_to_threshold(ctx, controller)
    aircraft.current_fuel = inflated.post_damage_fuel * 2.0   # clears checks (1) and (2)
    before = aircraft.current_fuel

    try:
        controller.maybe_apply(ctx.scenario, 55)
    except FuelDamageError as exc:
        assert "not below the LIVE continue requirement" in str(exc), str(exc)
    else:
        raise AssertionError("the event fired with a target that changes nothing")
    assert aircraft.current_fuel == before
    assert not controller.fired


def test_f2_a_refused_live_window_is_an_accounted_run_stage_failure() -> None:
    """F2. The refusal surfaces through `run_episode` and is attributed to `run`.

    Not `setup`: the preflight window WAS valid, and what failed is the live state at the
    event. `skip_and_account_v1` records it once either way, but the stage is the finding.
    """
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_PARAMS)
    plan = controller.plan
    aircraft = ctx.scenario.get_aircraft(plan.ego_id)
    assert _fly_to_threshold(ctx, controller)
    live = controller.live_bounds(aircraft)
    aircraft.current_fuel = 0.5 * (plan.post_damage_fuel + live.continue_fuel_requirement)

    saved = graph_tick_loop._wake_decision
    graph_tick_loop._wake_decision = lambda *a, **k: None
    try:
        graph_tick_loop.run_episode(
            None, ctx, GraphObservationConfig(detection_range_km=50.0),
            max_ticks=5, fuel_damage=controller,
        )
    except FuelDamageError as exc:
        assert "LIVE continue requirement" in str(exc), str(exc)
    else:
        raise AssertionError("run_episode swallowed the refused event")
    finally:
        graph_tick_loop._wake_decision = saved

    # `_run_one_episode` wraps whatever `run_episode` raises as the `run` stage.
    assert "run" in graph_train._PIPELINE_STAGES


# =============================================================================
# Config surface + purity
# =============================================================================

# =============================================================================
# PO-C -- an ego that has committed to return makes no further decisions (Defect C)
# =============================================================================
#
# Physical completion means an episode keeps ticking while aircraft fly home. Those
# extra ticks exist ONLY so the engine can resolve the lifecycle -- they must not become
# extra decision points. Once `rtb_issued` is latched, the tick loop skips that ego's
# whole Phase-1 chain: no sensing, no trigger, no wake, no policy call, no belief edit
# and no resync. Peers are untouched.


# The unassigned pop-up: 80 km out at t=0, closing 5 km per tick along the shared leg,
# so it crosses the 50 km detection radius partway through the run -- on the SAME tick
# for every ego, because they all fly the same leg at the same step. The crossing tick
# is DERIVED from what the peers actually did rather than pinned to a literal, so the
# proof is "the peers woke and this one did not", not an arithmetic coincidence.
_POPUP_TARGET_ID = "popup-late"
_POPUP_DISTANCE_KM = 80.0
_POPUP_STEP_KM = 5.0
_COMMON_BEARING_DEG = 30.0


def _returning_ego_run(*, max_ticks=14, target_distance_km=30.0):
    """Drive the REAL `run_episode` with a REAL `GraphPlanExecutor` over the stub world.

    THE SYMMETRY THAT MAKES THIS FALSIFIABLE. Ego 0 starts with an EMPTY plan, so the
    executor's pre-existing empty-plan branch orders it home on the very first tick. All
    three egos then fly the IDENTICAL path at the IDENTICAL speed, and an UNASSIGNED red
    airbase -- in no ego's `belief_tasks`, so a genuine pop-up -- sits just beyond the
    50 km detection radius and drifts into range partway through the run. Every ego is
    therefore at the same distance from the same pop-up at the same tick. The peers wake
    on it; the returning ego, and only the returning ego, does not. Nothing about the
    fixture prevents it from waking -- the guard does.

    Only `_wake_decision` is replaced, by a pure RECORDER that edits nothing: the
    encoder and the head decide nothing about the lifecycle, and stubbing them keeps
    this deterministic and torch-light. Everything else is production code.
    """
    ctx = _FuelDamageCtx(target_distance_km=target_distance_km, n_agents=3)
    returning, peers = ctx.agent_ids[0], ctx.agent_ids[1:]
    solution = {aid: list(v) for aid, v in ctx.a_init.items()}
    solution[returning] = []  # nothing left to do -> the executor orders it home
    for aid in ctx.agent_ids:
        ctx.beliefs[aid].solution = {k: list(v) for k, v in solution.items()}

    # The pop-up: an enemy airbase nobody has a task for, beyond detection range at t=0.
    popup_at = _point_at(_BASE, _POPUP_DISTANCE_KM, _COMMON_BEARING_DEG)
    ctx.scenario.airbases.append(
        _StubAirbase(_POPUP_TARGET_ID, popup_at.latitude, popup_at.longitude,
                     side_id=_RED_SIDE, side_color="red", name="Floridistan Pop-up")
    )

    tasks = ctx.beliefs[returning].tasks
    ctx.executor = GraphPlanExecutor(
        tasks=tasks, solution=solution, agents=ctx.agents, arrival_threshold_km=50.0,
    )
    # Every ego flies the SAME leg at the SAME fixed step, so their distance to the
    # pop-up is identical on every tick. No fuel burn: this fixture is about DECISIONS,
    # and a mid-run death would end the very Phase-1 processing whose absence is proven.
    ctx.env = _StubEnv(
        ctx.scenario, targets={aid: popup_at for aid in ctx.agent_ids},
        burn_fuel=False, step_km=_POPUP_STEP_KM,
    )

    sensed_calls, trigger_calls, wake_calls, resync_calls = [], [], [], []
    issued = []
    frozen = {}

    real_sensed = ctx.executor.sensed_target_ids
    real_resync = ctx.executor.resync
    real_next = ctx.executor.next_actions
    real_decide = graph_tick_loop.decide_triggers

    def spy_sensed(observation, ego_id):
        sensed_calls.append((str(ego_id), ctx.env.n_steps))
        return real_sensed(observation, ego_id)

    def spy_resync(new_solution, *, ego_id, tasks=None):
        resync_calls.append((str(ego_id), ctx.env.n_steps))
        return real_resync(new_solution, ego_id=ego_id, tasks=tasks)

    def spy_next(observation):
        commands = real_next(observation)
        issued.append((ctx.env.n_steps, list(commands)))
        # Phase 2 has run, so the latch is now whatever this tick made it. Freeze the
        # returning ego's belief the moment it commits, and compare at the end.
        if returning not in frozen and ctx.executor.rtb_issued.get(returning, False):
            frozen[returning] = _belief_fingerprint(ctx.beliefs[returning])
        return commands

    def spy_decide(belief_tasks, belief_solution, sensed, eta=None, *,
                   ego_id, clock, fuel_damage=False):
        trigger_calls.append((str(ego_id), int(clock)))
        return real_decide(belief_tasks, belief_solution, sensed,
                           ego_id=ego_id, clock=clock, fuel_damage=fuel_damage)

    def spy_wake(_policy, ego_id, _obs, _belief, _executor, _cfg, tick, **_kw):
        wake_calls.append((str(ego_id), int(tick)))
        return graph_tick_loop.Transition(
            gobs=None, ego_id=str(ego_id), tick=int(tick),
            meta_action=int(MetaAction.PLAN_COMPLIANCE), node_v=0,
            log_prob=0.0, entropy=0.0,
        )

    ctx.executor.sensed_target_ids = spy_sensed
    ctx.executor.resync = spy_resync
    ctx.executor.next_actions = spy_next
    saved = (graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision)
    graph_tick_loop.decide_triggers = spy_decide
    graph_tick_loop._wake_decision = spy_wake
    try:
        result = graph_tick_loop.run_episode(
            None, ctx, GraphObservationConfig(detection_range_km=50.0),
            max_ticks=max_ticks,
        )
    finally:
        graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision = saved

    return {
        "ctx": ctx, "result": result, "returning": returning, "peers": peers,
        "sensed": sensed_calls, "triggers": trigger_calls, "wakes": wake_calls,
        "resyncs": resync_calls, "issued": issued, "frozen": frozen,
        "max_ticks": max_ticks,
    }


def _belief_fingerprint(belief):
    return (
        [str(t.steps[0].target_id) for t in belief.tasks],
        {k: [tuple(a) for a in v] for k, v in belief.solution.items()},
    )


def test_poc_1_a_returning_ego_leaves_phase_1_entirely() -> None:
    """PO-C.1. After the RTB order the ego is never sensed, triggered, woken or resynced.

    The world is built so a still-processed ego WOULD wake every tick (its target is in
    range from t=0 and unassigned in its belief), so the silence below is the guard, not
    an accident of the fixture.
    """
    out = _returning_ego_run()
    ego, ctx = out["returning"], out["ctx"]

    rtb = [t for t, cmds in out["issued"] if "aircraft_return_to_base('%s')" % ego in cmds]
    assert rtb == [0], rtb
    assert ctx.executor.rtb_issued.get(ego) is True

    # It WAS processed on the tick it committed (Phase 1 ran before Phase 2 issued the
    # order), and the pop-up really did wake the peers later -- flying the same leg, at
    # the same distance, on the same tick. Without those two facts the silence below
    # would prove nothing.
    assert (ego, 0) in out["triggers"], out["triggers"]
    peer_wakes = sorted((e, t) for e, t in out["wakes"] if e in out["peers"])
    assert peer_wakes, "no peer ever woke; the fixture proves nothing"
    crossing = {t for _e, t in peer_wakes}
    assert len(crossing) == 1, ("the peers did not cross together", peer_wakes)
    sensed_tick = crossing.pop()
    assert sensed_tick > 0, sensed_tick
    assert peer_wakes == sorted((p, sensed_tick) for p in out["peers"]), peer_wakes

    late = lambda calls: [(e, t) for e, t in calls if e == ego and t > 0]
    assert late(out["triggers"]) == [], late(out["triggers"])
    assert late(out["sensed"]) == [], late(out["sensed"])
    assert late(out["wakes"]) == [], late(out["wakes"])
    assert late(out["resyncs"]) == [], late(out["resyncs"])

    # No transition is appended for it during the return.
    assert [tr for tr in out["result"].trajectory
            if tr.ego_id == ego and tr.tick > 0] == []


def test_poc_2_the_returning_egos_belief_is_frozen_from_the_moment_it_commits() -> None:
    """PO-C.2. The belief it carried into the return leg is the belief it ends with."""
    out = _returning_ego_run()
    ego, ctx = out["returning"], out["ctx"]

    assert ego in out["frozen"], "the ego never committed to return"
    assert _belief_fingerprint(ctx.beliefs[ego]) == out["frozen"][ego], (
        "the returning ego's belief was edited after it committed to return"
    )
    # Concretely: the pop-up every peer discovered is absent from its task list.
    returning_ids = [str(t.steps[0].target_id) for t in ctx.beliefs[ego].tasks]
    assert _POPUP_TARGET_ID not in returning_ids, returning_ids
    for peer in out["peers"]:
        peer_ids = [str(t.steps[0].target_id) for t in ctx.beliefs[peer].tasks]
        assert _POPUP_TARGET_ID in peer_ids, (peer, peer_ids)
    # And its executor slice stayed the empty mission that produced the RTB.
    assert ctx.executor.plans[ego] == []


def test_poc_3_peers_continue_normally_while_a_peer_returns() -> None:
    """PO-C.3. The guard is per-ego: everyone else keeps the full Phase-1 + Phase-2 path.

    Also pins the two-phase backbone: exactly one `env.step` and one `next_actions` per
    tick, unchanged by the guard.
    """
    out = _returning_ego_run()
    ctx, ticks = out["ctx"], out["max_ticks"]

    for peer in out["peers"]:
        seen = sorted({t for e, t in out["triggers"] if e == peer})
        assert seen == list(range(ticks)), (peer, seen)
        assert [t for e, t in out["sensed"] if e == peer] != []

    # Two-phase backbone: ONE snapshot, ONE step, ONE command pass per tick.
    assert ctx.env.n_steps == ticks
    assert len(out["issued"]) == ticks
    assert out["result"].ticks == ticks


def test_poc_4_no_peer_can_decide_whether_the_returning_ego_is_home_or_lost() -> None:
    """PO-C.4. Classification reads the ego's OWN entries only (no-communication).

    The same final world is re-presented with the peers moved through every lifecycle
    state -- still flying, landed, removed -- and the returning ego's own verdict must
    not move, nor may a peer's disappearance mark the returning ego dead.
    """
    out = _returning_ego_run()
    ctx, ego, peers = out["ctx"], out["returning"], out["peers"]
    executor, scenario = ctx.executor, ctx.scenario
    home = scenario.get_airbase("base-blue")

    baseline = executor._physical_state(ego, scenario)
    assert baseline == "airborne", baseline

    verdicts = set()
    for where in ("air", "landed", "gone"):
        peer_units = [scenario.get_aircraft(p) for p in peers]
        try:
            for unit in peer_units:
                if unit is not None and where != "air":
                    scenario.aircraft.remove(unit)
                    if where == "landed":
                        home.aircraft.append(unit)
            verdicts.add(executor._physical_state(ego, scenario))
        finally:  # restore, so each variant is judged against the same ego state
            for unit in peer_units:
                if unit is not None and unit not in scenario.aircraft:
                    scenario.aircraft.append(unit)
                if unit is not None and unit in home.aircraft:
                    home.aircraft.remove(unit)

    assert verdicts == {baseline}, verdicts
    assert ego not in executor.dead, "a peer's lifecycle marked the returning ego dead"


def test_rollout_config_mirrors_the_train_difficulty_cell() -> None:
    """ANTI-DRIFT: the two harnesses agree on the difficulty factor, field for field."""
    t = TrainConfig(n_iterations=1)
    r = RolloutConfig()
    for name in ("fuel_damage_mode", "fuel_damage_probability",
                 "fuel_damage_leg_progress", "fuel_damage_rtb_margin",
                 "fuel_damage_mild_probability", "aircraft_penalty_coeff"):
        assert getattr(t, name) == getattr(r, name), (
            "RolloutConfig.%s (%r) drifted from TrainConfig.%s (%r)"
            % (name, getattr(r, name), name, getattr(t, name))
        )
    assert t.fuel_damage_parameters() == r.fuel_damage_parameters()
    assert t.reward_config() == r.reward_config()


def test_the_configs_refuse_a_forced_mode_and_bad_parameters() -> None:
    """A forced mode is an EVAL group member; it is not a training mixture.

    All FOUR forced modes are refused by both harnesses -- the two severity ones are
    evaluation members exactly as the original two are, and a training run configured
    with one would condition every episode identically.
    """
    for mode in (FuelDamageMode.FORCED_DAMAGED, FuelDamageMode.FORCED_CLEAN,
                 FuelDamageMode.FORCED_MILD, FuelDamageMode.FORCED_SEVERE):
        for cfg in (TrainConfig(n_iterations=1, fuel_damage_mode=mode),
                    RolloutConfig(fuel_damage_mode=mode)):
            try:
                cfg.validate()
            except ValueError as exc:
                assert "group member" in str(exc), str(exc)
            else:
                raise AssertionError(
                    "%s accepted the forced mode %r as a training mode"
                    % (type(cfg).__name__, mode)
                )

    for kwargs in ({"fuel_damage_probability": 1.5},
                   {"fuel_damage_mild_probability": 1.5},
                   {"fuel_damage_mild_probability": -0.1},
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
# FD-VARIABLE-SEVERITY-v1 -- PO1: legacy preservation + deterministic severity
# =============================================================================

_VARIABLE = FuelDamageParameters(mode=FuelDamageMode.SEEDED_VARIABLE)
_MILD = FuelDamageParameters(mode=FuelDamageMode.FORCED_MILD)
_SEVERE = FuelDamageParameters(mode=FuelDamageMode.FORCED_SEVERE)


def test_vs_po1_the_legacy_v1_rng_domain_and_draw_order_are_untouched() -> None:
    """PO1. The severity factor cannot move ANY legacy FD-v1 decision.

    THE LOAD-BEARING CLAIM OF THE WHOLE TASK. An approved long-baseline measurement
    exists on the legacy design, so if adding severity had shifted the v1 stream -- which
    taking the mild/severe bit from it would have done -- every legacy seed would select
    a different ego and that measurement would be irreproducible rather than extended.

    Proven three ways: the domain string and derived seeds are unchanged, the seeded
    mixture assigns the identical condition under both designs, and the two domains are
    genuinely different streams.
    """
    assert graph_fuel_damage.FUEL_DAMAGE_RNG_DOMAIN == "fuel_damage_v1"
    assert graph_fuel_damage.FUEL_DAMAGE_SEVERITY_RNG_DOMAIN == "fuel_damage_severity_v1"

    for seed in range(64):
        # (a) the v1 derivation is EXACTLY the documented digest, recomputed here from
        # the spec rather than imported, so a change to the function fails this test.
        payload = ("fuel_damage_v1:%d" % seed).encode("ascii")
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        assert derive_fuel_damage_seed(seed) == expected, seed

        # (b) the SEVERITY domain is a different digest over a different string, so it
        # cannot be a re-labelled read of the same stream.
        sev_payload = ("fuel_damage_severity_v1:%d" % seed).encode("ascii")
        sev_expected = int.from_bytes(hashlib.sha256(sev_payload).digest()[:8], "big")
        assert derive_fuel_damage_severity_seed(seed) == sev_expected, seed
        assert derive_fuel_damage_severity_seed(seed) != derive_fuel_damage_seed(seed)

        # (c) the clean/damaged assignment is IDENTICAL under both seeded designs.
        assert (resolve_condition(episode_seed=seed, params=_PARAMS)
                == resolve_condition(episode_seed=seed, params=_VARIABLE)), seed

    # (d) a LEGACY mode never acquires a severity, and never consults that domain.
    for params in (_PARAMS,
                   FuelDamageParameters(mode=FuelDamageMode.FORCED_DAMAGED),
                   FuelDamageParameters(mode=FuelDamageMode.FORCED_CLEAN),
                   FuelDamageParameters(mode=FuelDamageMode.OFF)):
        assert params.variable_severity is False
        assert params.target_policy == TARGET_POLICY_PLANNED_MIDPOINT
        for seed in range(16):
            assert resolve_severity(episode_seed=seed, params=params) is None


def test_vs_po1_the_legacy_damaged_plan_is_field_identical_to_the_formula() -> None:
    """PO1. A legacy damaged plan's PHYSICS is unchanged, field by field.

    The legacy band and the severe band are the same interval, so the risk is that
    generalizing the arithmetic silently moved the legacy value. This recomputes the
    pre-severity formula independently -- the midpoint of `[rtb_floor, continue_req)`
    against the PROJECTED fuel -- and requires the plan to match it exactly.
    """
    for seed in range(24):
        if resolve_condition(episode_seed=seed, params=_PARAMS) != CONDITION_DAMAGED:
            continue
        plan = build_fuel_damage_plan(
            _FuelDamageCtx(), episode_seed=seed, params=_PARAMS
        )
        assert plan.severity is None, "a legacy plan must carry NO severity label"
        assert plan.target_policy == TARGET_POLICY_PLANNED_MIDPOINT
        assert plan.severity_derived_seed is None
        assert plan.mild_probability is None
        # The pre-severity formula, recomputed from the plan's own recorded bounds.
        expected = 0.5 * (plan.rtb_fuel_floor + plan.continue_fuel_requirement)
        assert plan.post_damage_fuel == expected, seed
        assert (plan.rtb_fuel_floor <= plan.post_damage_fuel
                < plan.continue_fuel_requirement)
        assert plan.post_damage_fuel < plan.projected_fuel_at_event


def test_vs_po1_severity_is_deterministic_and_matches_the_distribution() -> None:
    """PO1. Severity is a pure function of the seed, at 50% clean / 25% / 25%.

    Determinism is proven against deliberately hostile RNG state: global `random` and
    torch are re-seeded differently between the two reads, so a severity that leaked out
    of its private domain would diverge here.
    """
    first = {}
    for seed in range(512):
        random.seed(seed * 7 + 1)
        torch.manual_seed(seed * 13 + 5)
        first[seed] = resolve_severity(episode_seed=seed, params=_VARIABLE)
    for seed in range(512):
        random.seed(999 - seed)
        torch.manual_seed(seed + 4242)
        assert resolve_severity(episode_seed=seed, params=_VARIABLE) == first[seed], seed

    # The approved contract: P(clean) = 0.50, P(mild) = 0.25, P(severe) = 0.25. Sampled
    # over 512 seeds, so the tolerance is a sanity band, not an exact frequency claim.
    counts = {None: 0, SEVERITY_MILD: 0, SEVERITY_SEVERE: 0}
    for seed in range(512):
        counts[first[seed]] += 1
    assert counts[None] + counts[SEVERITY_MILD] + counts[SEVERITY_SEVERE] == 512
    for label, want in ((None, 0.50), (SEVERITY_MILD, 0.25), (SEVERITY_SEVERE, 0.25)):
        got = counts[label] / 512.0
        assert abs(got - want) < 0.07, (label, got, want, counts)

    # Every damaged episode has a severity, and every clean one has none -- a severity is
    # a refinement of `damaged`, never a third condition.
    for seed in range(128):
        damaged = (resolve_condition(episode_seed=seed, params=_VARIABLE)
                   == CONDITION_DAMAGED)
        severity = resolve_severity(episode_seed=seed, params=_VARIABLE)
        assert damaged == (severity in SEVERITIES), seed

    # The conditional knob is REAL: at P(mild|damaged) = 0 every damaged episode is
    # severe, at 1 every one is mild, and the clean/damaged split is unmoved either way.
    for p_mild, expected in ((0.0, SEVERITY_SEVERE), (1.0, SEVERITY_MILD)):
        params = FuelDamageParameters(mode=FuelDamageMode.SEEDED_VARIABLE,
                                      mild_probability=p_mild)
        for seed in range(64):
            assert (resolve_condition(episode_seed=seed, params=params)
                    == resolve_condition(episode_seed=seed, params=_PARAMS)), seed
            if resolve_condition(episode_seed=seed, params=params) == CONDITION_DAMAGED:
                assert resolve_severity(episode_seed=seed, params=params) == expected


def test_vs_po1_forced_mild_and_forced_severe_select_the_same_ego() -> None:
    """PO1. The triad's two damaged members are the SAME episode, damaged differently.

    This is what makes a matched triad a TRIAD: mild and severe must differ in the fuel
    band alone -- same ego, same route, same event point, same measured window -- so
    their reward difference cannot be attributed to which aircraft was hit. It also holds
    against the SEEDED episode and against LEGACY `forced_damaged`, which is the property
    that keeps the new design comparable with the approved baseline.
    """
    for seed in range(16):
        mild = build_fuel_damage_plan(_FuelDamageCtx(), episode_seed=seed, params=_MILD)
        severe = build_fuel_damage_plan(
            _FuelDamageCtx(), episode_seed=seed, params=_SEVERE
        )
        assert mild.severity == SEVERITY_MILD and severe.severity == SEVERITY_SEVERE
        # Same ego, same predicted route, same event point, same measured window.
        assert mild.ego_id == severe.ego_id, seed
        assert mild.route_points == severe.route_points, seed
        assert (mild.event_latitude, mild.event_longitude) == (
            severe.event_latitude, severe.event_longitude), seed
        assert mild.rtb_fuel_floor == severe.rtb_fuel_floor, seed
        assert mild.continue_fuel_requirement == severe.continue_fuel_requirement, seed
        assert mild.derived_seed == severe.derived_seed, seed
        assert (mild.severity_derived_seed == severe.severity_derived_seed
                == derive_fuel_damage_severity_seed(seed))
        # ...and only the BAND differs.
        assert mild.planned_band_low != severe.planned_band_low, seed
        assert mild.post_damage_fuel > severe.post_damage_fuel, seed

        # The same ego the SEEDED variable episode and the LEGACY design would pick.
        for params in (_VARIABLE, _PARAMS,
                       FuelDamageParameters(mode=FuelDamageMode.FORCED_DAMAGED)):
            if resolve_condition(episode_seed=seed, params=params) != CONDITION_DAMAGED:
                continue
            other = build_fuel_damage_plan(
                _FuelDamageCtx(), episode_seed=seed, params=params
            )
            assert other.ego_id == mild.ego_id, (seed, params.mode)


def test_vs_po1_rerunning_a_seed_reproduces_the_plan_and_the_outcome() -> None:
    """PO1. Same seed, same mode -> field-equivalent plan AND field-equivalent event."""
    for mode in (FuelDamageMode.SEEDED_VARIABLE, FuelDamageMode.FORCED_MILD,
                 FuelDamageMode.FORCED_SEVERE):
        params = FuelDamageParameters(mode=mode)
        for seed in (1, 5, 11):
            if resolve_condition(episode_seed=seed, params=params) != CONDITION_DAMAGED:
                continue
            outcomes, plans = [], []
            for repeat in range(2):
                random.seed(repeat * 31 + 7)   # hostile: different RNG state each time
                torch.manual_seed(repeat * 17 + 3)
                ctx = _FuelDamageCtx()
                controller = build_fuel_damage_controller(
                    ctx, episode_seed=seed, params=params
                )
                plans.append(controller.plan.to_record())
                for tick in range(4000):
                    if controller.maybe_apply(ctx.scenario, tick):
                        break
                    ctx.env.step([])
                outcomes.append(controller.outcome.to_record())
            assert plans[0] == plans[1], (mode, seed)
            assert outcomes[0] == outcomes[1], (mode, seed)
            assert outcomes[0]["fired"] is True, (mode, seed)


# =============================================================================
# FD-VARIABLE-SEVERITY-v1 -- PO2: physical validity + no-communication
# =============================================================================

def _fire(params, *, seed=1, ctx=None):
    """Run one episode's event to completion; returns (controller, ctx, tick)."""
    ctx = ctx or _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=seed, params=params)
    for tick in range(4000):
        if controller.maybe_apply(ctx.scenario, tick):
            return controller, ctx, tick
        ctx.env.step([])
    raise AssertionError("the event never fired")


def test_vs_po2_the_live_severity_inequalities_hold_exactly() -> None:
    """PO2. THE PHYSICS. Mild and severe land in their declared LIVE intervals.

        MILD    F_rtb < F_cont < F_after < F_before
        SEVERE  F_rtb <= F_after < F_cont <= F_before

    Measured at the LIVE event state -- the aircraft's real position and real fuel -- and
    against the bounds the mutation was really validated with, not the pre-run
    projection. The two are different numbers by construction (the engine burns fuel on
    route-less ticks too), which is exactly why the live measurement is the one that
    defines the severity.
    """
    for seed in (1, 3, 5, 7, 11):
        mild, _mild_ctx, _t = _fire(_MILD, seed=seed)
        out = mild.outcome
        assert out.severity == SEVERITY_MILD
        assert (out.live_rtb_fuel_floor
                < out.live_continue_fuel_requirement
                < out.fuel_after
                < out.fuel_before), (seed, out)
        # A mild loss leaves continuation feasible -- a POSITIVE margin -- which is the
        # single number that separates the two severities physically.
        assert out.continuation_margin > 0.0, seed
        assert 0.0 < out.fuel_after_fraction_of_max <= 1.0

        severe, _severe_ctx, _t2 = _fire(_SEVERE, seed=seed)
        sout = severe.outcome
        assert sout.severity == SEVERITY_SEVERE
        assert (sout.live_rtb_fuel_floor
                <= sout.fuel_after
                < sout.live_continue_fuel_requirement
                <= sout.fuel_before), (seed, sout)
        assert sout.continuation_margin < 0.0, seed

        # BOTH severities keep flying home FEASIBLE -- the reserve contract is never
        # traded away for difficulty; a severe event is a decision, not a kill.
        for o in (out, sout):
            assert o.fuel_after >= o.live_rtb_fuel_floor, seed
        # ...and severe really is the harsher of the two, on the same world.
        assert sout.fuel_after < out.fuel_after, seed
        # The two members hit the SAME ego at the SAME window -- only the band differs.
        assert mild.plan.ego_id == severe.plan.ego_id
        assert abs(out.live_continue_fuel_requirement
                   - sout.live_continue_fuel_requirement) < 1e-6

        # The applied value is the LIVE band's midpoint, not the planned one: the live
        # window is measured from a position the projection only approximated.
        assert abs(out.fuel_after
                   - 0.5 * (out.live_band_low + out.live_band_high)) < 1e-9
        assert mild.plan.post_damage_fuel != out.fuel_after, (
            "a variable-severity event must derive its target from the LIVE band"
        )


def test_vs_po2_the_severity_band_helper_is_the_one_arithmetic_site() -> None:
    """PO2. The band intervals are exactly as declared, including their inclusivity."""
    window = measure_window(
        position=_point_at(_BASE, 60.0, 30.0),
        route=[_point_at(_BASE, 250.0, 30.0)],
        home_base=_BASE, speed_knots=1303.0, fuel_rate=6700.0, margin=1.10,
    )
    fuel_before = window.continue_fuel_requirement * 4.0

    mild = severity_band(window=window, fuel_before=fuel_before,
                         severity=SEVERITY_MILD)
    assert (mild.low, mild.high) == (window.continue_fuel_requirement, fuel_before)
    assert mild.low_inclusive is False and mild.high_inclusive is False
    assert mild.target == 0.5 * (mild.low + mild.high)
    assert mild.contains(mild.target)
    assert not mild.contains(mild.low) and not mild.contains(mild.high)

    severe = severity_band(window=window, fuel_before=fuel_before,
                           severity=SEVERITY_SEVERE)
    assert (severe.low, severe.high) == (window.rtb_fuel_floor,
                                         window.continue_fuel_requirement)
    assert severe.low_inclusive is True and severe.high_inclusive is False
    assert severe.contains(severe.low) and not severe.contains(severe.high)

    # `None` IS the legacy interval -- the same interval as severe. That identity is what
    # makes "severe reproduces the legacy physics" checkable rather than merely asserted.
    legacy = severity_band(window=window, fuel_before=fuel_before, severity=None)
    assert (legacy.low, legacy.high) == (severe.low, severe.high)
    assert legacy.target == severe.target

    try:
        severity_band(window=window, fuel_before=fuel_before, severity="catastrophic")
    except FuelDamageError:
        pass
    else:
        raise AssertionError("an unknown severity was accepted")


def test_vs_po2_an_invalid_band_raises_before_the_mutation() -> None:
    """PO2. A refused event leaves `current_fuel` untouched and stays unfired.

    Nothing is clamped, downgraded to the other severity, or converted to clean: the
    approved policy is that a physically impossible band is an accounted `run`-stage
    FAILURE, not a quietly different episode.
    """
    # MILD needs live fuel STRICTLY above the continue requirement. Drain the tank below
    # it and the interval collapses.
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(ctx, episode_seed=1, params=_MILD)
    ego = str(controller.plan.ego_id)
    aircraft = ctx.scenario.get_aircraft(ego)
    for _ in range(4000):
        if (controller.observed_progress(ctx.scenario) or 0.0) >= 0.30:
            break
        ctx.env.step([])
    live = controller.live_bounds(aircraft)
    aircraft.current_fuel = live.continue_fuel_requirement * 0.5
    before = aircraft.current_fuel
    try:
        controller.maybe_apply(ctx.scenario, 123)
    except FuelDamageError as exc:
        assert "continue requirement" in str(exc), str(exc)
    else:
        raise AssertionError("an impossible MILD band was accepted")
    assert aircraft.current_fuel == before, "the engine was mutated before the refusal"
    assert controller.fired is False
    assert controller.outcome.fired is False
    assert controller.outcome.fuel_after is None

    # A degenerate window (continue requirement == RTB floor) leaves SEVERE no interior.
    flat = measure_window(
        position=_BASE, route=[_BASE], home_base=_BASE,
        speed_knots=1303.0, fuel_rate=6700.0, margin=1.10,
    )
    band = severity_band(window=flat, fuel_before=1000.0, severity=SEVERITY_SEVERE)
    try:
        graph_fuel_damage._require_valid_band(
            band, ego_id="ego0", fuel_before=1000.0, window=flat, where="live"
        )
    except FuelDamageError as exc:
        assert "empty" in str(exc), str(exc)
    else:
        raise AssertionError("a zero-width band was accepted")


def test_vs_po2_the_event_stays_top_of_tick_ego_local_and_order_independent() -> None:
    """PO2. The no-communication properties are UNCHANGED by the severity split.

    Re-proves, for the new modes, exactly what the legacy factor is proven to guarantee:
    ONE mutation, applied before any ego senses, visible only in the damaged ego's own
    fuel, with Phase-1 ego order irrelevant and peers untouched.
    """
    for params in (_MILD, _SEVERE):
        ctx = _FuelDamageCtx()
        controller = build_fuel_damage_controller(ctx, episode_seed=1, params=params)
        ego = str(controller.plan.ego_id)
        peers = [a for a in ctx.agent_ids if a != ego]
        peer_fuel_before = {
            p: ctx.scenario.get_aircraft(p).current_fuel for p in peers
        }

        woken, mutations = [], []
        for tick in range(4000):
            fired = controller.maybe_apply(ctx.scenario, tick)
            if fired is not None:
                mutations.append((tick, fired))
                # EVERY ego -- the first one included -- reads the post-event world,
                # because the mutation precedes the per-ego loop.
                snapshot = {
                    a: ctx.scenario.get_aircraft(a).current_fuel
                    for a in ctx.agent_ids
                    if ctx.scenario.get_aircraft(a) is not None
                }
                assert snapshot[ego] == controller.outcome.fuel_after
                for order in (ctx.agent_ids, list(reversed(ctx.agent_ids))):
                    seen = {a: snapshot[a] for a in order if a in snapshot}
                    assert seen == {a: snapshot[a] for a in snapshot}, "order mattered"
                woken = [a for a in ctx.agent_ids if a == fired]
            ctx.env.step([])
            if mutations and tick > mutations[0][0] + 50:
                break

        # ONE mutation, ONE woken ego, and it is the selected one.
        assert len(mutations) == 1, (params.mode, mutations)
        assert woken == [ego]
        # Peers lost only what the engine burns; the damage never reached them. The
        # damaged ego is strictly worse off than any peer's ordinary burn.
        for p in peers:
            live = ctx.scenario.get_aircraft(p)
            if live is None:
                continue
            burned = peer_fuel_before[p] - live.current_fuel
            assert burned >= 0.0
            assert burned < abs(controller.outcome.fuel_before
                                - controller.outcome.fuel_after), p


def test_vs_po2_no_severity_label_reaches_the_observation() -> None:
    """PO2. The policy is told NOTHING about severity -- only its own live fuel.

    The anti-shortcut premise of the whole experiment. If a severity label reached the
    graph, the actor could read the answer instead of inferring it, and the measurement
    would be of a labelled classifier rather than of an adaptive policy.
    """
    # The graph's feature width is untouched, and the builder knows nothing about the
    # fuel-damage layer at all.
    assert TASK_FEATURE_DIM == 6
    builder_source = Path(
        inspect.getsourcefile(graph_builder)
    ).read_text(encoding="utf-8")
    for forbidden in ("severity", "mild", "graph_fuel_damage"):
        assert forbidden not in builder_source.lower(), forbidden

    # What DOES change is the damaged ego's own fuel_norm, and it changes by the
    # severity: a mild event leaves a fuller tank than a severe one on the same world.
    norms = {}
    for params in (_MILD, _SEVERE):
        controller, ctx, _tick = _fire(params, seed=1)
        ego = str(controller.plan.ego_id)
        gobs = build_graph_observation(
            ctx.scenario, ego,
            tasks=ctx.beliefs[ego].tasks, solution=ctx.beliefs[ego].solution,
            config=GraphObservationConfig(detection_range_km=50.0),
        )
        row = gobs.agent_ids.index(ego)
        norms[params.mode] = float(gobs.agent_features[row, 0])
        # Peers stay FEATURELESS, so the damaged value is unreachable from any peer row.
        for i, aid in enumerate(gobs.agent_ids):
            if aid != ego:
                assert float(gobs.agent_features[i, 0]) == 0.0, aid
    assert norms[FuelDamageMode.FORCED_MILD] > norms[FuelDamageMode.FORCED_SEVERE], norms


# =============================================================================
# FD-VARIABLE-SEVERITY-v1 -- PO3: matched triad + durable measurement
# =============================================================================

def _variable_cfg(tmp_path: Path, **kwargs) -> TrainConfig:
    """A minimal variable-severity training config for the stub-driven trainer tests."""
    base = dict(n_iterations=1, episodes_per_iteration=4, base_seed=0,
                output_dir=tmp_path / "run", eval_every=1, eval_episodes=3,
                eval_base_seed=1_000_000, checkpoint_every=0,
                fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE)
    base.update(kwargs)
    return TrainConfig(**base)


def _jsonl(path: Path) -> list:
    """Read a jsonl artifact into a list of dicts (missing file -> empty list)."""
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _skip_plotting() -> bool:
    """True when matplotlib is unavailable (plots are optional, never a hard dep)."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return True
    return False


def test_vs_po3_a_held_out_seed_is_evaluated_as_a_clean_mild_severe_triad(
    tmp_path: Path,
) -> None:
    """PO3. Three members per held-out seed: same seed, forced cells, disjoint tags."""
    cfg = _variable_cfg(tmp_path)
    assert cfg.eval_group_kind == "triad" and cfg.eval_group_size == 3
    assert cfg.reported_cells == (CONDITION_CLEAN, SEVERITY_MILD, SEVERITY_SEVERE)
    _summary, events = _run_stub_training(cfg)

    evals = [e for e in events if e[0] == "episode" and e[1] == "eval"]
    # One PRE-UPDATE round and one post-update round, each 3 seeds x 3 members.
    assert len(evals) == 2 * 3 * 3, len(evals)

    first_round = evals[:9]
    seeds = [e[2] for e in first_round]
    tags = [e[3] for e in first_round]
    modes = [e[4] for e in first_round]

    # Each held-out seed appears exactly three times, adjacently -- one triad.
    assert seeds == [1_000_000] * 3 + [1_000_001] * 3 + [1_000_002] * 3, seeds
    # ...once forced clean, once forced mild and once forced severe.
    assert modes == [FuelDamageMode.FORCED_CLEAN, FuelDamageMode.FORCED_MILD,
                     FuelDamageMode.FORCED_SEVERE] * 3, modes
    # ...and every member writes to a DIFFERENT scenario file.
    assert len(set(tags)) == 9, tags
    assert tags == [eval_member_tag(round_ordinal=0, e=e, member=m, group_size=3)
                    for e in range(3) for m in (0, 1, 2)]
    # The second round's tags are disjoint from the first's (rounds never overwrite).
    assert not (set(tags) & set(e[3] for e in evals[9:]))

    # The scenario files really coexist on disk: 9 per round, none overwritten.
    scen = sorted(p.name for p in (tmp_path / "run" / "scenarios").glob("*.json"))
    assert len(scen) == len(set(scen)) >= 18, scen

    # LEGACY EVALUATION IS UNCHANGED: a seeded_mixture run still runs PAIRS.
    legacy_cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                             output_dir=tmp_path / "legacy", eval_every=1,
                             eval_episodes=3, eval_base_seed=1_000_000,
                             checkpoint_every=0)
    assert legacy_cfg.eval_group_kind == "pair" and legacy_cfg.eval_group_size == 2
    _s2, legacy_events = _run_stub_training(legacy_cfg)
    legacy_evals = [e for e in legacy_events if e[1] == "eval"]
    assert len(legacy_evals) == 2 * 3 * 2, "a legacy run silently became a triad"
    assert [e[4] for e in legacy_evals[:6]] == [
        FuelDamageMode.FORCED_CLEAN, FuelDamageMode.FORCED_DAMAGED] * 3


def test_vs_po3_every_delta_and_rate_travels_with_its_denominator(
    tmp_path: Path,
) -> None:
    """PO3. The three deltas come from COMPLETE triads only, and say so.

    The tempting repair -- differencing the two members that survived when the third
    failed -- would report a within-seed comparison that was never measured. What is
    locked here is that an incomplete triad contributes to NO delta, and that every rate
    and every mean is reported next to the population it was taken over.
    """
    cfg = _variable_cfg(tmp_path)
    # Fail the SEVERE member of the middle held-out seed, in every round.
    summary, _events = _run_stub_training(
        cfg, failing_eval=(1_000_001, FuelDamageMode.FORCED_SEVERE)
    )

    for ev in summary["eval_records"]:
        assert ev["eval_group_kind"] == "triad" and ev["eval_group_size"] == 3
        assert ev["n_groups_attempted"] == 3
        # Two complete triads: the middle seed lost a member and contributes to none of
        # the deltas, even though its clean and mild members both succeeded.
        assert ev["n_groups_successful"] == 2, ev
        assert ev["group_success_fraction"] == 2 / 3
        assert ev["n_attempted"] == 9 and ev["n_successful"] == 8 and ev["n_failed"] == 1
        # Per-cell denominators, including the one that lost a member.
        assert ev["eval_n_clean_attempted"] == 3 and ev["eval_n_clean_successful"] == 3
        assert ev["eval_n_mild_attempted"] == 3 and ev["eval_n_mild_successful"] == 3
        assert ev["eval_n_severe_attempted"] == 3
        assert ev["eval_n_severe_successful"] == 2
        assert ev["eval_n_severe_failed"] == 1
        # `damaged` still pools the two severities -- its meaning is unchanged.
        assert ev["eval_n_damaged_attempted"] == 6
        assert ev["eval_n_damaged_successful"] == 5
        # All three deltas exist, are named, and are over the SAME complete population.
        assert ev["eval_delta_keys"] == [
            "eval_delta_mild_minus_clean", "eval_delta_severe_minus_clean",
            "eval_delta_severe_minus_mild",
        ]
        for key in ev["eval_delta_keys"]:
            assert ev[key] is not None, key
        assert ev["eval_delta_over"] == "groups_with_all_members_successful"
        # The stub's rewards are clean -0.5 / mild -0.3 / severe -0.7, so the deltas are
        # exact and the arithmetic is checkable rather than merely present.
        assert abs(ev["eval_delta_mild_minus_clean"] - 0.2) < 1e-9
        assert abs(ev["eval_delta_severe_minus_clean"] + 0.2) < 1e-9
        assert abs(ev["eval_delta_severe_minus_mild"] + 0.4) < 1e-9
        # The LEGACY damaged-minus-clean key is null under a triad: there is no single
        # damaged member to difference, and inventing one would be a false measurement.
        assert ev["eval_paired_reward_delta"] is None
        # THE PRIMARY BEHAVIOURAL MEASUREMENT, per severity, over FD WAKES.
        assert ev["eval_n_mild_fd_wakes"] == 3
        assert ev["eval_n_severe_fd_wakes"] == 2
        assert ev["eval_fd_meta_action_counts_mild"]["PLAN_COMPLIANCE"] == 3
        assert ev["eval_fd_meta_action_counts_severe"]["SELF_PRESERVATION_ABORT"] == 2
        assert ev["eval_fd_meta_action_rates_mild"]["SELF_PRESERVATION_ABORT"] == 0.0
        assert ev["eval_fd_meta_action_rates_severe"]["SELF_PRESERVATION_ABORT"] == 1.0
        # The held-out SEED band is 3 wide however many attempts each seed took.
        assert ev["seed_band"]["stop"] - ev["seed_band"]["start"] == 3

    # Every failure is in the ledger exactly once, under its own CONDITION.
    ledger = [json.loads(line) for line in
              (tmp_path / "run" / "episode_failures.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    assert all(r["condition"] == CONDITION_DAMAGED for r in ledger), ledger
    assert summary["accounting_reconciled"] is True


def test_vs_po3_an_empty_triad_population_is_null_not_zero(tmp_path: Path) -> None:
    """PO3. No complete triad -> `null` for every delta, never 0.0 ('no effect')."""
    cfg = _variable_cfg(tmp_path, eval_episodes=2)
    summary, _events = _run_stub_training(
        cfg, failing_eval=("*", FuelDamageMode.FORCED_SEVERE)
    )
    for ev in summary["eval_records"]:
        assert ev["n_groups_successful"] == 0
        for key in ev["eval_delta_keys"]:
            assert ev[key] is None, key
        assert ev["eval_reward_mean_severe"] is None
        assert ev["eval_n_severe_fd_wakes"] == 0
        # A rate with no denominator is MISSING, not zero.
        for rate in ev["eval_fd_meta_action_rates_severe"].values():
            assert rate is None
        # The members that did run are still reported.
        assert ev["eval_reward_mean_clean"] is not None
        assert ev["eval_reward_mean_mild"] is not None
    assert summary["final_eval_groups_successful"] == 0
    assert all(v is None for v in summary["final_eval_group_deltas"].values())


def test_vs_po3_the_durable_stream_exposes_every_per_attempt_measurement(
    tmp_path: Path,
) -> None:
    """PO3. `episode_outcomes.jsonl` carries one auditable row per SUCCESSFUL attempt.

    The aggregate records cannot be un-averaged, so the distributional question this
    experiment exists to answer needs a per-episode stream. What is locked here is that
    it is written for every successful attempt, carries the identity / event / outcome
    fields the analysis needs, and does NOT duplicate the failure ledger.
    """
    cfg = _variable_cfg(tmp_path)
    summary, _events = _run_stub_training(
        cfg, failing_eval=(1_000_001, FuelDamageMode.FORCED_SEVERE)
    )
    path = tmp_path / "run" / graph_train._EPISODE_OUTCOMES_FILENAME
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # ONE row per successful attempt, and NOTHING for the failed one -- the two files
    # are disjoint, so reading both cannot double-count an attempt.
    n_success = (summary["train_episodes_successful"]
                 + summary["eval_episodes_successful"])
    assert len(rows) == n_success, (len(rows), n_success)
    assert summary["episode_outcomes_recorded"] == len(rows)
    ledger = [json.loads(line) for line in
              (tmp_path / "run" / "episode_failures.jsonl").read_text(
                  encoding="utf-8").splitlines() if line.strip()]
    assert ledger, "the fixture must actually fail something"
    assert summary["eval_episodes_failed"] == len(ledger)

    required = (
        "schema", "phase", "iteration", "updates_completed", "attempt_ordinal",
        "seed", "episode_tag", "fuel_damage_mode", "cell", "condition", "severity",
        "fd_derived_seed", "fd_severity_derived_seed", "fd_ego_id", "fd_fired",
        "fd_event_tick", "fd_observed_progress", "fd_fuel_before", "fd_fuel_after",
        "fd_damage_factor", "fd_fuel_after_fraction_of_max",
        "fd_live_rtb_fuel_floor", "fd_live_continue_fuel_requirement",
        "fd_continuation_margin", "fd_wake_occurred", "fd_wake_meta_action",
        "fd_wake_meta_action_name", "fd_rtb_command_issued",
        "reward", "n_dead", "targets_confirmed_unique", "targets_total",
        "ended", "ticks",
    )
    for row in rows:
        for key in required:
            assert key in row, key
        assert row["cell"] in (CONDITION_CLEAN, SEVERITY_MILD, SEVERITY_SEVERE)
        # A clean row states ABSENCE as null, never as a zero that would read as an
        # empty tank or as tick zero.
        if row["cell"] == CONDITION_CLEAN:
            assert row["severity"] is None
            assert row["fd_fuel_after"] is None
            assert row["fd_event_tick"] is None
            assert row["fd_wake_meta_action_name"] is None
        else:
            assert row["severity"] == row["cell"]
            assert row["condition"] == CONDITION_DAMAGED
            assert row["fd_wake_meta_action_name"] in (
                "PLAN_COMPLIANCE", "SELF_PRESERVATION_ABORT")

    # Both phases are present, and eval rows carry their group coordinates.
    phases = {r["phase"] for r in rows}
    assert {"train", "pre_update", "post_update"} <= phases, phases
    for row in rows:
        if row["phase"] == "train":
            assert row["episode_index"] is not None
            assert row["eval_group_member"] is None
        else:
            assert row["eval_group_member"] in (0, 1, 2)
            assert row["eval_episode_index"] is not None


def test_vs_po3_the_summary_is_rebuilt_from_the_durable_files_alone(
    tmp_path: Path,
) -> None:
    """PO3. Summary and plots are derivable from the run directory, with no run state.

    `build_run_summary` re-reads the jsonl artifacts, so the severity-response table it
    publishes cannot describe a run the files do not -- the one-metric-path discipline,
    extended to the new stream.
    """
    cfg = _variable_cfg(tmp_path)
    summary, _events = _run_stub_training(cfg)

    rebuilt = graph_train.build_run_summary(tmp_path / "run", cfg=cfg)
    for key in ("severity_response", "episode_outcomes_recorded",
                "final_eval_group_deltas", "eval_group_cells", "difficulty_factor",
                "fuel_damage_totals"):
        assert rebuilt[key] == summary[key], key
    assert rebuilt["difficulty_factor"] == "fuel_damage_variable_severity_v1"
    assert rebuilt["eval_group_kind"] == "triad"

    # The severity-response table is per phase and per cell, and its RATES are over FD
    # WAKES -- a smaller population than the episode count, because an event can fire
    # without ever waking the policy.
    response = rebuilt["severity_response"]
    assert set(response) >= {"train", "pre_update", "post_update"}
    for phase, cells in response.items():
        for cell, stats in cells.items():
            assert stats["rates_over"] == "fd_wakes"
            assert stats["n_fd_wakes"] <= stats["n_episodes"]
            for name, rate in stats["meta_action_rates"].items():
                if stats["n_fd_wakes"] == 0:
                    assert rate is None, (phase, cell, name)
                else:
                    assert abs(rate - stats["meta_action_counts"][name]
                               / stats["n_fd_wakes"]) < 1e-9
    # The behavioural split the stub encoded is visible in the table: mild complies,
    # severe aborts.
    post = response["post_update"]
    assert post[SEVERITY_MILD]["meta_action_rates"]["PLAN_COMPLIANCE"] == 1.0
    assert post[SEVERITY_SEVERE]["meta_action_rates"]["SELF_PRESERVATION_ABORT"] == 1.0

    # The summary is persisted WITHOUT the embedded record lists -- the jsonl files stay
    # the single record.
    written = json.loads(
        (tmp_path / "run" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert "episode_outcome_records" not in written
    assert written["severity_response"] == rebuilt["severity_response"]

    # And the figures render from those files alone, with no policy and no torch.
    if not _skip_plotting():
        written_paths = graph_train.plot_training_subprocess(tmp_path / "run")
        # The three REQUIRED figures, in order, plus -- because this run really
        # records per-wake diagnostics (episode-outcome schema v3) -- the OPTIONAL
        # `fd_policy_sensitivity.png`. A pre-v3 run produces the three alone.
        names = [p.name for p in written_paths]
        assert names[:3] == list(graph_train._PLOT_FILENAMES)
        assert set(names[3:]) <= set(graph_train._PLOT_OPTIONAL_FILENAMES)
        for p in written_paths:
            assert p.exists() and p.stat().st_size > 1000, p


def test_vs_po3_a_cell_the_run_does_not_report_aborts_as_data_integrity() -> None:
    """PO3. An episode whose plan disagrees with its schedule can never vanish quietly.

    Storage is per cell, and `to_record` emits only the run's declared cells -- so an
    outcome carrying an undeclared cell would keep its reward inside the round totals
    while disappearing from every per-cell mean and denominator. That is a measurement
    disagreeing with its own accounting, so it is INFRASTRUCTURE and it aborts, exactly
    as a roster fault does.
    """
    tally = graph_train._ConditionTally((CONDITION_CLEAN, CONDITION_DAMAGED))
    out = graph_train._EpisodeOutcome(
        trajectory=[], reward=-0.5, ticks=1, ended="done", n_wakes=0,
        confirmed_kills=0, n_dead=0, seconds=0.0, targets_confirmed_unique=0,
        targets_total=6, known_target_names=(), hidden_target_names=(),
        known_confirmed_names=(), hidden_confirmed_names=(),
        fuel_damage_plan={"condition": CONDITION_DAMAGED, "severity": SEVERITY_MILD},
        fuel_damage_outcome={"fired": True, "wake_occurred": False},
        selected_ego_rtb_issued=None,
    )
    try:
        tally.success(out, expected_cell=CONDITION_DAMAGED)
    except MeasurementIntegrityError as exc:
        assert "mild" in str(exc), str(exc)
    else:
        raise AssertionError("an undeclared cell was silently absorbed")
    # It is INFRASTRUCTURE, never an accounted episode failure.
    assert not issubclass(MeasurementIntegrityError, EpisodeAttemptError)
    # The same guard protects the DENOMINATOR side.
    try:
        tally.attempt(SEVERITY_SEVERE)
    except MeasurementIntegrityError:
        pass
    else:
        raise AssertionError("an undeclared cell was scheduled without complaint")
    # ...and a SCHEDULE naming a cell this tally cannot report is refused too, so the
    # two sides of the comparison are both required to be reportable before they are
    # compared.
    try:
        tally.success(out, expected_cell=SEVERITY_SEVERE)
    except MeasurementIntegrityError as exc:
        assert "scheduled under cell" in str(exc), str(exc)
    else:
        raise AssertionError("an undeclared SCHEDULED cell was accepted")


def test_vs_po3_a_scheduled_cell_that_executes_as_another_aborts(tmp_path: Path) -> None:
    """PO3. SCHEDULED cell != EXECUTED cell is a measurement-integrity abort.

    THE FAULT MEMBERSHIP ALONE CANNOT CATCH. Under FD-VARIABLE-SEVERITY-v1 a scheduled
    `mild` that executed as `severe` names a cell the run legitimately reports, so a
    membership test accepts it -- and then books the ATTEMPT in mild's denominator and
    the REWARD in severe's. Both cells are corrupted at once: mild reads as a failure
    that never happened, severe as a success that was never scheduled, and a matched
    triad's within-seed delta would be taken between two members the schedule never
    paired. Equality with the scheduled cell is the only check that sees it.

    Proven at three levels: the tally in isolation, the real TRAINING call site, and the
    real EVALUATION call site -- the last two because the guard is only worth anything
    if both production sites actually pass their scheduled cell into it.
    """
    cells = (CONDITION_CLEAN, SEVERITY_MILD, SEVERITY_SEVERE)

    def _damaged_outcome(severity):
        return graph_train._EpisodeOutcome(
            trajectory=[_StubTransition()], reward=-0.9, ticks=7, ended="done",
            n_wakes=1, confirmed_kills=1, n_dead=2, seconds=0.01,
            targets_confirmed_unique=1, targets_total=6,
            known_target_names=("A",), hidden_target_names=("B",),
            known_confirmed_names=("A",), hidden_confirmed_names=(),
            fuel_damage_plan={"condition": CONDITION_DAMAGED, "severity": severity},
            fuel_damage_outcome={"condition": CONDITION_DAMAGED, "severity": severity,
                                 "fired": True, "wake_occurred": True,
                                 "wake_meta_action":
                                     MetaAction.SELF_PRESERVATION_ABORT.value},
            selected_ego_rtb_issued=True,
        )

    # ---- (1) the tally in isolation: rejected, and NOTHING is mutated ----
    tally = graph_train._ConditionTally(cells)
    tally.attempt(SEVERITY_MILD)

    def snapshot(t):
        return {
            "attempted": dict(t.attempted), "failed": dict(t.failed),
            "rewards": {k: list(v) for k, v in t.rewards.items()},
            "fd_fired": dict(t.fd_fired), "fd_wakes": dict(t.fd_wakes),
            "fd_meta": {k: dict(v) for k, v in t.fd_meta.items()},
            "events_applied": t.events_applied, "wakes": t.wakes,
            "rtb_issued": t.rtb_issued, "deaths": t.deaths,
        }

    before = snapshot(tally)
    try:
        tally.success(_damaged_outcome(SEVERITY_SEVERE), expected_cell=SEVERITY_MILD)
    except MeasurementIntegrityError as exc:
        message = str(exc)
        assert "SCHEDULED as 'mild'" in message, message
        assert "EXECUTED as 'severe'" in message, message
    else:
        raise AssertionError(
            "a mild-scheduled attempt was folded in as a severe success"
        )
    # The rejected episode left NO trace: not a reward, not an FD counter, not a death.
    assert snapshot(tally) == before, "a rejected episode mutated the tally"
    assert tally.successful(SEVERITY_MILD) == 0
    assert tally.successful(SEVERITY_SEVERE) == 0
    assert tally.mean(SEVERITY_SEVERE) is None
    assert (tally.events_applied, tally.wakes, tally.rtb_issued, tally.deaths) == (
        0, 0, 0, 0)
    # ...and the matching case is still accepted, so the guard rejects the fault rather
    # than the feature.
    assert tally.success(
        _damaged_outcome(SEVERITY_MILD), expected_cell=SEVERITY_MILD) == SEVERITY_MILD
    assert tally.successful(SEVERITY_MILD) == 1

    # ---- (2) the real TRAINING call site ----
    # Seed 1 is a scheduled MILD training episode; the stub executes it as SEVERE.
    cfg = _variable_cfg(tmp_path / "train_site")
    assert resolve_severity(
        episode_seed=1, params=cfg.fuel_damage_parameters()) == SEVERITY_MILD
    try:
        _run_stub_training(cfg, mislabel_train_seed=1)
    except MeasurementIntegrityError as exc:
        assert "SCHEDULED as 'mild'" in str(exc), str(exc)
    else:
        raise AssertionError("the TRAINING call site accepted a mismatched episode")
    run_dir = tmp_path / "train_site" / "run"
    # It ABORTED: it is never an accounted scientific failure, and the rejected attempt
    # left no successful outcome record behind.
    ledger = _jsonl(run_dir / "episode_failures.jsonl")
    assert ledger == [], ledger
    outcomes = _jsonl(run_dir / graph_train._EPISODE_OUTCOMES_FILENAME)
    assert all(row["seed"] != 1 for row in outcomes), outcomes
    assert all(row["phase"] != "train" or row["cell"] == CONDITION_CLEAN
               for row in outcomes), "a mismatched training episode was recorded"

    # ---- (3) the real EVALUATION call site ----
    # The forced-MILD member of a held-out triad, executed as severe.
    cfg2 = _variable_cfg(tmp_path / "eval_site")
    try:
        _run_stub_training(
            cfg2, mislabel_eval=(1_000_000, FuelDamageMode.FORCED_MILD)
        )
    except MeasurementIntegrityError as exc:
        assert "SCHEDULED as 'mild'" in str(exc), str(exc)
    else:
        raise AssertionError("the EVALUATION call site accepted a mismatched member")
    run_dir2 = tmp_path / "eval_site" / "run"
    assert _jsonl(run_dir2 / "episode_failures.jsonl") == []
    for row in _jsonl(run_dir2 / graph_train._EPISODE_OUTCOMES_FILENAME):
        # The clean member of that seed ran first and is legitimately recorded; the
        # mismatched mild member must not be, and the severe member never ran.
        assert not (row["seed"] == 1_000_000 and row["cell"] != CONDITION_CLEAN), row
    # No eval round was ever completed, so no matched-group delta was computed from a
    # group whose member was rejected.
    assert _jsonl(run_dir2 / "eval_records.jsonl") == []


def test_vs_po3_the_run_config_records_the_severity_contract(tmp_path: Path) -> None:
    """PO3. A run states its design in its own config: modes, split, group and deltas."""
    payload = json.loads(_write_run_config_to_string(_variable_cfg(tmp_path)))
    difficulty = payload["difficulty"]
    assert difficulty["factor"] == "fuel_damage_variable_severity_v1"
    fd = difficulty["fuel_damage"]
    assert fd["mode"] == FuelDamageMode.SEEDED_VARIABLE
    assert fd["variable_severity"] is True
    assert fd["mild_probability"] == 0.5
    assert fd["severity_rng_domain"] == "fuel_damage_severity_v1"
    assert fd["rng_domain"] == "fuel_damage_v1", "the legacy domain must be recorded too"
    assert fd["target_policy"] == TARGET_POLICY_LIVE_SEVERITY_MIDPOINT
    assert fd["severities"] == [SEVERITY_MILD, SEVERITY_SEVERE]
    # The approved three-way distribution, written out rather than left to be derived.
    assert difficulty["scheduled_cell_probabilities"] == {
        CONDITION_CLEAN: 0.5, SEVERITY_MILD: 0.25, SEVERITY_SEVERE: 0.25,
    }
    assert difficulty["eval_group_kind"] == "triad"
    assert difficulty["eval_group_cells"] == [
        CONDITION_CLEAN, SEVERITY_MILD, SEVERITY_SEVERE]
    assert difficulty["eval_group_modes"] == [
        FuelDamageMode.FORCED_CLEAN, FuelDamageMode.FORCED_MILD,
        FuelDamageMode.FORCED_SEVERE]
    assert difficulty["eval_group_members_per_seed"] == 3
    assert difficulty["eval_group_deltas"] == [
        "eval_delta_mild_minus_clean", "eval_delta_severe_minus_clean",
        "eval_delta_severe_minus_mild"]
    # The reward is UNCHANGED -- the factor changes the world, never the objective.
    assert difficulty["reward"]["formula_changed"] is False
    assert difficulty["reward"]["aircraft_penalty_coeff"] == 2.25

    # A LEGACY run's config still says exactly what it always said.
    legacy = json.loads(_write_run_config_to_string(TrainConfig(n_iterations=1)))
    assert legacy["difficulty"]["factor"] == "fuel_damage_baseline_v1"
    assert legacy["difficulty"]["fuel_damage"]["variable_severity"] is False
    assert legacy["difficulty"]["scheduled_cell_probabilities"] is None
    assert legacy["difficulty"]["eval_group_kind"] == "pair"
    assert legacy["difficulty"]["eval_pair_conditions"] == [
        CONDITION_CLEAN, CONDITION_DAMAGED]


def test_vs_po3_the_severity_knob_reaches_both_harnesses_and_the_cli() -> None:
    """PO3. Configuration parity: trainer, rollout and CLI agree on the new factor."""
    t = TrainConfig(n_iterations=1, fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
                    fuel_damage_mild_probability=0.25)
    r = RolloutConfig(fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
                      fuel_damage_mild_probability=0.25)
    assert t.fuel_damage_parameters() == r.fuel_damage_parameters()
    assert t.fuel_damage_parameters().mild_probability == 0.25
    t.validate()
    r.validate()

    # The CLI exposes the mode and the conditional, and its defaults are read off the
    # dataclass -- so the two cannot drift.
    parser = graph_train._build_arg_parser()
    args = parser.parse_args(["--fuel-damage-mode", FuelDamageMode.SEEDED_VARIABLE,
                              "--fuel-damage-mild-probability", "0.25"])
    assert args.fuel_damage_mode == FuelDamageMode.SEEDED_VARIABLE
    assert args.fuel_damage_mild_probability == 0.25
    default = TrainConfig(n_iterations=1)
    assert (parser.parse_args([]).fuel_damage_mild_probability
            == default.fuel_damage_mild_probability)
    # The dest -> field mapping covers the new knob, so a JSON preset and the flag reach
    # the same field.
    assert (graph_train._CLI_FIELD_BY_DEST["fuel_damage_mild_probability"]
            == "fuel_damage_mild_probability")

    # A degenerate conditional WARNS (a researcher may probe it) and never raises.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        TrainConfig(n_iterations=1,
                    fuel_damage_mode=FuelDamageMode.SEEDED_VARIABLE,
                    fuel_damage_mild_probability=1.0).validate()
    assert "[WARN]" in buf.getvalue() and "same severity" in buf.getvalue()


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
    # The cell is declared to match the world this fixture actually builds: `n_agents`
    # known targets and no constructed hidden half. `_run_one_episode` now refuses a
    # roster that does not describe its configured cell, so a config claiming 3 + 3 = 6
    # against a 3-target stub world would be a (correctly) rejected measurement.
    n_known = len(getattr(ctx, "known_target_ids", ()) or ())
    cfg = TrainConfig(n_iterations=1, episodes_per_iteration=1, base_seed=0,
                      eval_every=0, checkpoint_every=0,
                      num_agents=n_known, n_known=n_known, n_hidden=0)

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


def _run_stub_training(cfg: TrainConfig, *, failing_seeds=(), failing_eval=None,
                       mislabel_train_seed=None, mislabel_eval=None):
    """Drive the REAL `train()` with the BLADE + solver episode body stubbed.

    Everything under test stays real: the loop, the seed and group schedule, the tag
    allocation, the ledger, the cell accounting and the record writers.
    ``failing_eval`` is ``(seed_or_"*", mode)`` -- the eval member that raises.

    ``mislabel_train_seed`` / ``mislabel_eval`` make one attempt return an outcome whose
    executed plan reports the OTHER severity than the one its schedule resolved. That is
    not a situation the production pipeline can currently produce -- which is exactly why
    it has to be injected to be testable at all: the guard it exercises exists so that if
    the schedule and the executed plan ever DO diverge, the run stops instead of booking
    the attempt in one cell and the reward in another. ``mislabel_eval`` is
    ``(seed, mode)``.
    """
    events = []
    failing_seeds = set(failing_seeds)
    fail_seed, fail_mode = failing_eval if failing_eval else (None, None)
    mislabel_seed, mislabel_mode = mislabel_eval if mislabel_eval else (None, None)

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
        # The stub plan must report the SAME cell the schedule resolved -- the tally
        # refuses an episode whose plan disagrees with its own scheduling, and a stub
        # that omitted the severity would be exercising that refusal instead of the
        # feature. `resolve_severity` is the production function, so the stub cannot
        # drift from it.
        severity = resolve_severity(episode_seed=seed, params=params)
        damaged = condition == CONDITION_DAMAGED
        # INJECTED DIVERGENCE: report the other severity than the one scheduled, for the
        # one attempt under test. Everything else about the outcome stays consistent, so
        # what the guard sees is exactly a scheduled/executed cell disagreement.
        mislabelled = (
            (phase == "train" and seed == mislabel_train_seed)
            or (phase == "eval" and mislabel_mode is not None
                and fuel_damage_mode == mislabel_mode
                and (mislabel_seed == "*" or seed == mislabel_seed))
        )
        if mislabelled and severity in SEVERITIES:
            severity = (SEVERITY_SEVERE if severity == SEVERITY_MILD
                        else SEVERITY_MILD)
        # A severity-dependent reward, so a matched triad has a real within-seed
        # structure to difference: severe costs more than mild, and both cost more than
        # clean. Arbitrary magnitudes -- what is under test is the ARITHMETIC over
        # complete groups, not the values.
        reward = -0.5 + {None: 0.0, SEVERITY_MILD: 0.2, SEVERITY_SEVERE: -0.2}.get(
            severity, 0.1 if damaged else 0.0
        )
        meta = (None if not damaged
                else (MetaAction.PLAN_COMPLIANCE.value if severity == SEVERITY_MILD
                      else MetaAction.SELF_PRESERVATION_ABORT.value))
        return graph_train._EpisodeOutcome(
            trajectory=[_StubTransition()], reward=reward,
            ticks=42, ended="done", n_wakes=1, confirmed_kills=1, n_dead=0,
            seconds=0.01, targets_confirmed_unique=1, targets_total=6,
            known_target_names=("A",), hidden_target_names=("B",),
            known_confirmed_names=("A",), hidden_confirmed_names=(),
            fuel_damage_plan={"condition": condition, "severity": severity,
                              "ego_id": "ego0" if damaged else None},
            fuel_damage_outcome={"condition": condition, "severity": severity,
                                 "fired": damaged, "wake_occurred": damaged,
                                 "wake_meta_action": meta},
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
# GENERALIZED-V1 step 2 -- G1: CERTIFIED FD ELIGIBILITY (PO1)
# =============================================================================

def _certified(mode=FuelDamageMode.FORCED_SEVERE, **kwargs) -> FuelDamageParameters:
    """The approved knobs with the CERTIFIED eligibility policy selected."""
    return FuelDamageParameters(
        mode=mode, eligibility_policy=FD_ELIGIBILITY_CERTIFIED_V1, **kwargs
    )


def test_g1_1_the_eligibility_rng_domain_is_private_versioned_and_stable() -> None:
    """G1.1. A THIRD domain, derived from the episode seed alone and from nothing else.

    The same three properties the other two domains claim -- reproducible across
    processes, well mixed over consecutive seeds, and unreachable from any other RNG
    consumer -- plus the one that only matters for a third domain: it must not COLLIDE
    with either existing one, or the "eligibility cannot move the ego" separation would
    be a coincidence rather than a construction.
    """
    first = [derive_fuel_damage_eligibility_seed(s) for s in range(16)]

    random.seed(4242)
    torch.manual_seed(4242)
    [random.random() for _ in range(97)]
    torch.rand(23)
    second = [derive_fuel_damage_eligibility_seed(s) for s in range(16)]
    assert first == second, "the eligibility seed moved with unrelated RNG consumption"
    assert len(set(first)) == len(first), "consecutive episode seeds collide"

    for seed in range(16):
        assert derive_fuel_damage_eligibility_seed(seed) != derive_fuel_damage_seed(seed)
        assert (derive_fuel_damage_eligibility_seed(seed)
                != derive_fuel_damage_severity_seed(seed))

    # The domain string is versioned and is what the digest is actually taken over.
    expected = int.from_bytes(
        hashlib.sha256(
            ("%s:%d" % (FUEL_DAMAGE_ELIGIBILITY_RNG_DOMAIN, 7)).encode("ascii")
        ).digest()[:8],
        "big",
    )
    assert derive_fuel_damage_eligibility_seed(7) == expected

    # Reproducible in a FRESH interpreter -- `hash()` would not be (PYTHONHASHSEED).
    child = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);\n"
         "from match_aou.rl.training.graph_fuel_damage import "
         "derive_fuel_damage_eligibility_seed as d;\n"
         "print([d(s) for s in range(16)])" % str(SRC)],
        capture_output=True, text=True, check=True,
    )
    assert eval(child.stdout.strip()) == first  # noqa: S307 - our own literal list


def test_g1_2_the_certified_policy_does_not_move_the_historical_streams() -> None:
    """G1.2. Selecting the certified policy changes NO legacy draw.

    The approved FD-BASELINE-v1 and FD-VARIABLE-SEVERITY-v1 measurements were taken on
    the legacy streams. If the new policy took its permutation from `fuel_damage_v1` it
    would shift the position the ego draw reads from, and every historical damaged
    episode would pick a different ego -- invalidating those measurements rather than
    extending them. So: same condition, same severity, same legacy ego, seed for seed.
    """
    for mode in (FuelDamageMode.SEEDED_MIXTURE, FuelDamageMode.SEEDED_VARIABLE):
        legacy = FuelDamageParameters(mode=mode)
        certified = _certified(mode=mode)
        for seed in range(64):
            assert (resolve_condition(episode_seed=seed, params=certified)
                    == resolve_condition(episode_seed=seed, params=legacy))
            assert (resolve_severity(episode_seed=seed, params=certified)
                    == resolve_severity(episode_seed=seed, params=legacy))

    # And the LEGACY ego draw itself is byte-unchanged: the documented two-draw order
    # (mixture bit, then `choice` over the sorted non-empty routes) still reproduces it.
    ctx = _FuelDamageCtx()
    for seed in (0, 1, 2, 3, 5, 8, 13):
        plan = build_fuel_damage_plan(
            ctx, episode_seed=seed,
            params=FuelDamageParameters(mode=FuelDamageMode.FORCED_DAMAGED),
        )
        rng = random.Random(derive_fuel_damage_seed(seed))
        rng.random()
        assert plan.ego_id == rng.choice(sorted(ctx.a_init)), seed
        assert plan.eligibility_policy == FD_ELIGIBILITY_LEGACY_V1
        assert plan.eligibility_audit is None and plan.certificate is None


def test_g1_3_candidate_order_is_ordinal_based_and_mirrors_the_placement_permutation() -> None:
    """G1.3. STABLE ORDINALS, never id text -- and the same Fisher-Yates as B2's.

    Handoff 3l.2: generated ids are not seed-derived, so an ordering keyed on their text
    would make an episode's damaged ego irreproducible across runs of the same seed. The
    permutation is therefore over `range(count)`, and it is the deliberate MIRROR of
    `graph_hidden_placement._ordinal_permutation` -- written separately so a change to
    hidden placement's ordering cannot silently move which ego is damaged, with the
    equivalence pinned HERE rather than assumed.
    """
    for count in range(1, 8):
        mine = eligibility_ordinal_permutation(count, random.Random(count * 977 + 5))
        theirs = graph_hidden_placement._ordinal_permutation(
            count, random.Random(count * 977 + 5)
        )
        assert mine == theirs, (count, mine, theirs)
        assert sorted(mine) == list(range(count)), mine

    # Renaming every ego cannot move the walk: the audit's ORDER is ordinals, and the
    # selected ORDINAL is what a cross-run comparison keys on.
    plain = _FuelDamageCtx()
    plan_a = build_fuel_damage_plan(plain, episode_seed=11, params=_certified())
    renamed = _FuelDamageCtx()
    remap = {old: "zzz-%02d" % i for i, old in enumerate(renamed.agent_ids)}
    renamed.agent_ids = [remap[a] for a in renamed.agent_ids]
    renamed.a_init = {remap[k]: v for k, v in renamed.a_init.items()}
    renamed.beliefs = {
        remap[k]: _StubBelief(b.tasks, {remap[i]: list(v) for i, v in b.solution.items()})
        for k, b in renamed.beliefs.items()
    }
    for agent, old_id in zip(renamed.agents, sorted(remap)):
        agent.id = remap[old_id]
    for aircraft, old_id in zip(renamed.scenario.aircraft, sorted(remap)):
        aircraft.id = remap[old_id]
    plan_b = build_fuel_damage_plan(renamed, episode_seed=11, params=_certified())

    assert (plan_a.eligibility_audit.candidate_order
            == plan_b.eligibility_audit.candidate_order)
    assert (plan_a.eligibility_audit.selected_ordinal
            == plan_b.eligibility_audit.selected_ordinal)


def test_g1_4_clean_mild_and_severe_certify_the_same_ego_and_certificate() -> None:
    """G1.4 + A2. Eligibility is a WORLD precondition, so CLEAN walks it too.

    This is the whole point of the policy (handoff 3l.3): the three members of a matched
    group must share one accepted-world support, which they only do if the clean member
    performs the SAME complete walk and identifies the SAME counterfactual ego. The clean
    plan still damages nobody -- `ego_id` keeps its historical meaning and stays None --
    so the counterfactual lives in the audit rather than overloading that field.
    """
    ctx = _FuelDamageCtx()
    plans = {
        mode: build_fuel_damage_plan(ctx, episode_seed=3, params=_certified(mode=mode))
        for mode in (FuelDamageMode.FORCED_CLEAN, FuelDamageMode.FORCED_MILD,
                     FuelDamageMode.FORCED_SEVERE)
    }
    audits = {m: p.eligibility_audit for m, p in plans.items()}
    reference = audits[FuelDamageMode.FORCED_CLEAN]
    for mode, audit in audits.items():
        assert audit.policy == FD_ELIGIBILITY_CERTIFIED_V1
        assert audit.candidate_order == reference.candidate_order, mode
        assert audit.considered_ordinals == reference.considered_ordinals, mode
        assert audit.selected_ordinal == reference.selected_ordinal, mode
        assert audit.selected_ego_id == reference.selected_ego_id, mode
        assert (audit.certificate.to_record()
                == reference.certificate.to_record()), mode

    clean = plans[FuelDamageMode.FORCED_CLEAN]
    assert clean.condition == CONDITION_CLEAN and clean.ego_id is None
    assert clean.eligibility_audit.selected_ego_id is not None
    for mode in (FuelDamageMode.FORCED_MILD, FuelDamageMode.FORCED_SEVERE):
        assert plans[mode].ego_id == reference.selected_ego_id
        assert plans[mode].certificate is not None
        # The damaged plan's event geometry IS the certificate's, not a re-projection.
        assert plans[mode].event_latitude == reference.certificate.latitude
        assert plans[mode].projected_fuel_at_event == reference.certificate.fuel_before


def test_g1_5_every_scheduled_ego_is_a_candidate_and_exhaustion_is_a_normal_rejection() -> None:
    """G1.5. Egos the allocated-only A_init omitted are CANDIDATES, rejected truthfully.

    An ego the solver left out has no route and cannot carry the event -- but it is a
    finding, not an absence: a population that quietly excluded it could never report
    how often that happens. And when the bounded walk exhausts every candidate the world
    is rejected at SETUP with a stable machine-readable reason, which is ordinary
    accounted attrition, never the integrity exception.
    """
    ctx = _FuelDamageCtx()
    ctx.a_init = {aid: [] for aid in ctx.agent_ids}  # allocated-only, and it allocated none

    raised = None
    try:
        build_fuel_damage_plan(ctx, episode_seed=5, params=_certified())
    except FuelDamageError as exc:
        raised = exc
    assert raised is not None, "a world with no routed ego must be rejected"

    message = str(raised)
    assert NO_FD_ELIGIBLE_EGO in message, message
    assert REASON_NO_ROUTE in message, message
    assert not isinstance(raised, FuelDamageIntegrityError), (
        "an ineligible SETUP is ordinary attrition, never an instrument fault"
    )

    # Every ordinal was visited exactly once -- BOUNDED, and complete.
    ctx_one = _FuelDamageCtx()
    keep = ctx_one.agent_ids[1]
    ctx_one.a_init = {
        aid: (list(v) if aid == keep else []) for aid, v in ctx_one.a_init.items()
    }
    plan = build_fuel_damage_plan(ctx_one, episode_seed=5, params=_certified())
    audit = plan.eligibility_audit
    assert audit.selected_ego_id == keep, audit.to_record()
    assert len(audit.considered_ordinals) == len(set(audit.considered_ordinals))
    rejected = [c for c in audit.candidates if not c.accepted]
    assert all(c.reason == REASON_NO_ROUTE for c in rejected), rejected
    assert all(c.reason in FD_ELIGIBILITY_REJECTION_REASONS for c in rejected)
    assert audit.candidate_count == len(ctx_one.agent_ids)


def test_g1_6_a_one_assignment_ego_is_certified_when_physically_valid() -> None:
    """G1.6. No >= 2-assignment requirement -- that would bias the generalized sample.

    Requiring two assignments would quietly restrict the population to solver-STACKED
    allocations, which is a research choice, not a physical one. A one-assignment ego is
    a perfectly valid FD candidate; it simply has no later completion boundary to reach.
    """
    ctx = _FuelDamageCtx()
    assert all(len(v) == 1 for v in ctx.a_init.values()), "fixture must be one-assignment"
    plan = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    assert plan.certificate is not None and plan.certificate.route_length == 1


def test_g1_7_a_pre_event_popup_risk_rejects_the_candidate() -> None:
    """G1.7 / A5.1. A sensable target absent from the ego's OWN t=0 belief is fatal.

    Such a target would be classified POP_UP by `decide_triggers` and could wake the
    actor -- and therefore move the route -- BEFORE the certified event state exists. The
    certificate is not defended by suppressing that trigger or by changing actor
    behaviour; the CANDIDATE is rejected instead, which is the only option that leaves
    the runtime semantics untouched. It applies to ANY world target the ego's belief does
    not hold, not merely to the construction path's "hidden" half.
    """
    ctx = _FuelDamageCtx()
    # A live enemy airbase 20 km off the base, in NOBODY's belief task list.
    near = _point_at(_BASE, 20.0, 0.0)
    ctx.scenario.airbases.append(
        _StubAirbase("unbelieved", near.latitude, near.longitude,
                     side_id=_RED_SIDE, side_color="red", name="Unbelieved AFB")
    )
    try:
        build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    except FuelDamageError as exc:
        assert REASON_PRE_EVENT_POPUP_RISK in str(exc), str(exc)
        assert NO_FD_ELIGIBLE_EGO in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a pre-event pop-up risk must reject every candidate")

    # CONTROL: the same geometry with that target IN every ego's belief certifies fine,
    # so the rejection above is the belief membership and not the extra unit.
    ctx2 = _FuelDamageCtx()
    ctx2.scenario.airbases.append(
        _StubAirbase("unbelieved", near.latitude, near.longitude,
                     side_id=_RED_SIDE, side_color="red", name="Unbelieved AFB")
    )
    for belief in ctx2.beliefs.values():
        belief.tasks = list(belief.tasks) + [_attack_task("unbelieved", near)]
    assert build_fuel_damage_plan(
        ctx2, episode_seed=3, params=_certified()
    ).certificate is not None


def test_g1_8_a_pre_event_assignment_boundary_rejects_the_candidate() -> None:
    """G1.8 / A5.2. Already inside the arrival radius before the event is fatal.

    Phase 2 would be free to attack, confirm and advance the plan before the certified
    state exists, so the fuel window the certificate promises would be measured against a
    route the ego has already left.
    """
    # 60 km legs: at 30 % the ego is still 42 km out, INSIDE the 50 km radius, so an
    # earlier pre-event tick already sits inside it.
    ctx = _FuelDamageCtx(target_distance_km=60.0)
    try:
        build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    except FuelDamageError as exc:
        assert REASON_PRE_EVENT_ASSIGNMENT_BOUNDARY in str(exc), str(exc)
    else:  # pragma: no cover
        raise AssertionError("a pre-event arrival boundary must reject every candidate")


def test_g1_9_the_certificate_requires_both_bands_with_a_one_tick_margin() -> None:
    """G1.9 / A3. F_rtb < F_continue < F_before, each interval wider than ONE TICK.

    Both severities must be constructible ON THE SAME EGO, or the matched group is not
    one world with one factor varied. The margin is the engine's own quantum -- a band
    narrower than a single tick of burn could be crossed by the very quantization the
    certificate already tolerates -- and it is DERIVED (`fuel_rate / 3600`), never tuned.
    """
    ctx = _FuelDamageCtx()
    cert = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified()).certificate

    assert cert.rtb_fuel_floor < cert.continue_fuel_requirement < cert.fuel_before
    assert cert.required_band_margin_fuel == _approx(6700.0 / 3600.0)
    assert cert.band_margin_fuel > cert.required_band_margin_fuel

    # The recorded bands ARE `severity_band` at the certified state -- one arithmetic
    # site, not a second copy.
    window = measure_window(
        position=cert.event_location,
        route=[ctx.targets[cert.ego_id]],
        home_base=Location(_BASE.latitude, _BASE.longitude),
        speed_knots=1303.0, fuel_rate=6700.0, margin=1.10,
    )
    mild = severity_band(window=window, fuel_before=cert.fuel_before,
                         severity=SEVERITY_MILD)
    severe = severity_band(window=window, fuel_before=cert.fuel_before,
                           severity=SEVERITY_SEVERE)
    assert cert.mild_band_low == _approx(mild.low)
    assert cert.mild_target == _approx(mild.target)
    assert cert.severe_band_high == _approx(severe.high)
    assert cert.severe_target == _approx(severe.target)

    # A tank that barely covers the continue requirement leaves no MILD band, so the
    # candidate is rejected rather than certified for severe alone.
    starved = _FuelDamageCtx(fuel=1350.0)
    try:
        build_fuel_damage_plan(starved, episode_seed=3, params=_certified())
    except FuelDamageError as exc:
        assert REASON_INVALID_BAND in str(exc), str(exc)
    else:  # pragma: no cover
        raise AssertionError("a world with no mild band must not be certified")


def test_g1_10_the_event_state_is_derived_from_the_engine_tick_and_fuel_semantics() -> None:
    """G1.10 / A4. The DISCRETE event state, from the engine's own one-second model.

    Three separate claims, none of which the legacy distance projection makes:
      * the per-tick leg is the engine's `get_next_coordinates` arithmetic, floor and all;
      * a state after `m` movements is observed at tick `m + 1`, because the launch tick
        is airborne, route-less and still burns;
      * `fuel_before` is exactly `launch - tick * fuel_rate / 3600`, with NO invented
        reserve of any kind added on top.
    """
    # (1) the engine's own leg arithmetic, transcribed rather than approximated.
    for distance_km, speed in ((250.0, 1303.0), (17.5, 450.0), (3.0, 900.0)):
        seconds = max(math.floor(distance_km * KILOMETERS_TO_NAUTICAL_MILES
                                 / speed * 3600.0), 0.0001)
        expected = distance_km / seconds
        expected = distance_km if distance_km < expected else expected
        assert engine_leg_distance_km(distance_km, speed_knots=speed) == expected
    # A negative speed is NORMALIZED, exactly as the engine normalizes it.
    assert (engine_leg_distance_km(250.0, speed_knots=-1303.0)
            == engine_leg_distance_km(250.0, speed_knots=1303.0))

    # (2) tick == movements + 1, and progress is the executor's own quantity.
    states = predict_leg_states(leg_length_km=250.0, speed_knots=1303.0)
    assert states[0].movements == 0 and states[0].tick == 1
    assert states[0].progress == 0.0, "no movement has happened at the top of tick 1"
    for st in states:
        assert st.tick == st.movements + 1
        assert st.progress == _approx((250.0 - st.remaining_km) / 250.0)

    # (3) the certificate's own numbers, with no reserve term.
    ctx = _FuelDamageCtx()
    cert = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified()).certificate
    assert cert.event_tick == cert.movement_count + 1
    assert cert.fuel_per_tick == _approx(6700.0 / 3600.0)
    assert cert.fuel_before == _approx(12000.0 - cert.event_tick * 6700.0 / 3600.0)
    assert cert.progress >= 0.30
    earlier = states[cert.movement_count - 1]
    assert earlier.progress < 0.30, "the certified tick must be the FIRST crossing"
    # The tolerated bracket is exactly +/- the one engine quantum, and both tolerances
    # are that quantum plus a documented float epsilon -- not a round number.
    assert cert.tick_tolerance == CERTIFICATE_TICK_TOLERANCE == 1
    assert set(cert.bracket_ticks) == {
        cert.event_tick - 1, cert.event_tick, cert.event_tick + 1
    }
    assert cert.fuel_tolerance > cert.fuel_per_tick
    assert cert.fuel_tolerance < 2.0 * cert.fuel_per_tick
    assert cert.position_tolerance_km > cert.leg_km_per_tick


def test_g1_11_a_contradicted_certificate_raises_the_integrity_error() -> None:
    """G1.11 / A7. A certified world that contradicts itself is an INSTRUMENT fault.

    Handoff 3l.3: under the certified policy live validation keeps its defensive role but
    changes its MEANING. A world proven FD-capable before a tick was paid for cannot then
    be "an episode that did not work out" -- if it arrives somewhere the certificate does
    not describe, the certifier does not describe the simulator. It must abort, and it
    must not be a `FuelDamageError`, because anything catching one would swallow it.
    """
    assert not issubclass(FuelDamageIntegrityError, FuelDamageError)
    assert not issubclass(FuelDamageError, FuelDamageIntegrityError)

    ctx = _FuelDamageCtx()
    plan = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    cert = plan.certificate
    ego = plan.ego_id
    aircraft = next(a for a in ctx.scenario.aircraft if str(a.id) == ego)

    def _at_certified_state():
        aircraft.latitude, aircraft.longitude = cert.latitude, cert.longitude
        aircraft.current_fuel = cert.fuel_before

    # CONTROL: at the certified state the event fires normally.
    _at_certified_state()
    controller = FuelDamageController(plan)
    assert controller.maybe_apply(ctx.scenario, cert.event_tick) == ego
    assert controller.outcome.fired

    # (a) A PHYSICALLY contradicted certificate aborts. The ABSOLUTE OUTER TICK is
    # deliberately NOT in this loop any more: frozen BLADE can skip an airborne ego's
    # whole update, so the outer tick is diagnostic only. Its own case is G1.11b, and
    # the two physical invariants it replaced as the binding ones are unchanged here.
    for corrupt, needle in (
        (replace(cert, fuel_before=cert.fuel_before + 500.0), "FUEL FAILED"),
        (replace(cert, latitude=cert.latitude + 1.0), "POSITION FAILED"),
    ):
        _at_certified_state()
        fuel_snapshot = aircraft.current_fuel
        bad = FuelDamageController(replace(plan, certificate=corrupt))
        try:
            bad.maybe_apply(ctx.scenario, cert.event_tick)
        except FuelDamageIntegrityError as exc:
            assert "CERTIFICATE CONTRADICTED" in str(exc), str(exc)
            assert needle in str(exc), str(exc)
        else:  # pragma: no cover
            raise AssertionError("a contradicted certificate must abort: %s" % needle)
        assert aircraft.current_fuel == fuel_snapshot, (
            "the mutation must be refused BEFORE the engine is touched"
        )
        assert not bad.outcome.fired

    # (b) A certified plan whose LIVE PHYSICS refuses the event is the SAME fault, and
    # it is worth exercising separately because the three checks above cover the state
    # while this covers the window derived FROM that state. The reserve is inflated on
    # the plan alone -- position, tick and fuel still match the certificate exactly -- so
    # the state checks pass and the physics is what refuses.
    _at_certified_state()
    fuel_snapshot = aircraft.current_fuel
    inconsistent = FuelDamageController(replace(plan, rtb_safety_margin=1000.0))
    try:
        inconsistent.maybe_apply(ctx.scenario, cert.event_tick)
    except FuelDamageIntegrityError as exc:
        assert "CERTIFICATE CONTRADICTED" in str(exc), str(exc)
    else:  # pragma: no cover
        raise AssertionError("a certified world whose physics refuses must abort")
    assert aircraft.current_fuel == fuel_snapshot, "refused BEFORE the engine is touched"

    # A LEGACY plan in the same situation stays ordinary accounted attrition.
    legacy_plan = build_fuel_damage_plan(
        ctx, episode_seed=3, params=FuelDamageParameters(mode=FuelDamageMode.FORCED_DAMAGED)
    )
    legacy_ego = legacy_plan.ego_id
    legacy_ac = next(a for a in ctx.scenario.aircraft if str(a.id) == legacy_ego)
    legacy_ac.current_fuel = 1.0  # far below every live bound
    legacy_ac.latitude = ctx.targets[legacy_ego].latitude
    legacy_ac.longitude = ctx.targets[legacy_ego].longitude
    legacy_controller = FuelDamageController(legacy_plan)
    try:
        legacy_controller.maybe_apply(ctx.scenario, 10)
    except FuelDamageIntegrityError:  # pragma: no cover
        raise AssertionError("the LEGACY policy must not raise the integrity error")
    except FuelDamageError:
        pass


def test_g1_11b_the_absolute_outer_tick_is_diagnostic_never_binding() -> None:
    """G1.11b. THE PROVEN FROZEN-BLADE SKIP IS ACCEPTED, on its PHYSICAL state alone.

    `Game.update_all_aircraft_position` iterates `self.current_scenario.aircraft` while
    its own `land_aicraft` -> `remove_aircraft` path calls `.remove()` on that SAME list.
    Removing an element shifts the tail left under the live iterator, so the entry that
    followed the departing aircraft is skipped ENTIRELY for that outer tick -- losing
    BOTH its movement leg and its `fuel_rate / 3600` burn. An ego whose peers land is
    therefore physically EARLIER than the outer tick count implies.

    THE EXACT DIAGNOSED SIGNATURE is reproduced here: certified event tick 914 (movement
    count 913, bracket 913..915), two lost engine visits, the crossing observed at outer
    tick 916 -- with the ego AT the certified physical state. That is +2 ticks, strictly
    outside `tick_tolerance`, so the retired check would have aborted; the physical
    promise held exactly, so the repaired check must not.

    WHAT IS PROVEN, and nothing weaker:
      * `_require_certificate_holds` does not raise on the +2 signature;
      * `maybe_apply` goes on to the ORDINARY live severity/window checks and mutates;
      * NO tolerance was widened -- both are still exactly the certificate's own quanta.
    """
    ctx = _FuelDamageCtx()
    plan = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    issued = plan.certificate
    ego = plan.ego_id
    aircraft = next(a for a in ctx.scenario.aircraft if str(a.id) == ego)

    # The certificate the diagnosed run carried, in its own tick coordinates. Only the
    # tick bookkeeping is restated; every PHYSICAL field is the one setup really issued.
    cert = replace(issued, event_tick=914, movement_count=913,
                   bracket_ticks=(913, 914, 915))
    plan = replace(plan, certificate=cert)
    observed_tick = 916

    # The retired check's own condition, stated so this regression is falsifiable: the
    # observed tick really is outside the bracket, so a tick-binding check WOULD abort.
    assert abs(observed_tick - cert.event_tick) > cert.tick_tolerance
    assert observed_tick not in cert.bracket_ticks

    # The ego is at the certified PHYSICAL event state -- the state the diagnosed run
    # matched to ~7e-11 km and ~6e-9 lbs.
    aircraft.latitude, aircraft.longitude = cert.latitude, cert.longitude
    aircraft.current_fuel = cert.fuel_before

    controller = FuelDamageController(plan)
    # (1) the live check itself is silent.
    controller._require_certificate_holds(
        aircraft=aircraft, tick=observed_tick, fuel_before=float(aircraft.current_fuel)
    )
    # (2) and the whole event proceeds: ordinary live physics, then the real mutation.
    fuel_before = float(aircraft.current_fuel)
    assert controller.maybe_apply(ctx.scenario, observed_tick) == ego
    out = controller.outcome
    assert out.fired and out.event_tick == observed_tick
    assert aircraft.current_fuel < fuel_before, "the mutation is a real LOSS"
    assert aircraft.current_fuel == _approx(out.fuel_after)
    # The live severity/window checks really ran -- a severe event leaves continuation
    # infeasible and a safe RTB affordable, which is what they exist to guarantee.
    assert out.live_rtb_fuel_floor is not None
    assert out.fuel_after >= out.live_rtb_fuel_floor

    # (3) NOTHING WAS WIDENED. Both tolerances are still the certificate's own quanta,
    # and a state just BEYOND either still aborts at this very tick.
    assert cert.tick_tolerance == CERTIFICATE_TICK_TOLERANCE == 1
    assert cert.fuel_tolerance > cert.fuel_per_tick
    assert cert.fuel_tolerance < 2.0 * cert.fuel_per_tick
    assert cert.position_tolerance_km > cert.leg_km_per_tick
    assert cert.position_tolerance_km < 2.0 * cert.leg_km_per_tick
    fresh = FuelDamageController(plan)
    bad_fuel = cert.fuel_before + 2.0 * cert.fuel_tolerance
    aircraft.latitude, aircraft.longitude = cert.latitude, cert.longitude
    aircraft.current_fuel = bad_fuel
    try:
        fresh._require_certificate_holds(
            aircraft=aircraft, tick=observed_tick, fuel_before=bad_fuel
        )
    except FuelDamageIntegrityError:
        pass
    else:  # pragma: no cover
        raise AssertionError("the fuel quantum must still be binding at any tick")


def test_g1_11c_a_physical_contradiction_aborts_and_reports_all_three_deltas() -> None:
    """G1.11c. The two PHYSICAL invariants are binding, and a failure reports everything.

    Four cases, and the fourth is the point of the repair:
      * position beyond its own quantum while fuel is exact          -> ABORT
      * fuel beyond its own quantum while position is exact          -> ABORT
      * BOTH beyond their quanta                                     -> ABORT
      * the outer tick far outside the old bracket, BOTH physical
        fields exact                                                 -> ACCEPTED

    Every abort is raised BEFORE the mutation, and every abort message carries the
    position offset, the fuel offset AND the tick discrepancy -- one contradicted
    quantity must never hide the state of the other, because the message is what a
    preserved failure is diagnosed from without a replay.
    """
    ctx = _FuelDamageCtx()
    plan = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    cert = plan.certificate
    ego = plan.ego_id
    aircraft = next(a for a in ctx.scenario.aircraft if str(a.id) == ego)

    # A position offset comfortably beyond one tick of travel, expressed as a real
    # displacement from the certified point rather than as an invented number.
    far = _point_at(cert.event_location, 5.0 * cert.position_tolerance_km, 90.0)
    bad_fuel = cert.fuel_before + 5.0 * cert.fuel_tolerance

    cases = (
        ("position only", far.latitude, far.longitude, cert.fuel_before, True, False),
        ("fuel only", cert.latitude, cert.longitude, bad_fuel, False, True),
        ("both", far.latitude, far.longitude, bad_fuel, True, True),
    )
    for label, lat, lon, fuel, want_pos_fail, want_fuel_fail in cases:
        aircraft.latitude, aircraft.longitude = lat, lon
        aircraft.current_fuel = fuel
        snapshot = float(aircraft.current_fuel)
        controller = FuelDamageController(plan)
        try:
            controller._require_certificate_holds(
                aircraft=aircraft, tick=cert.event_tick, fuel_before=fuel
            )
        except FuelDamageIntegrityError as exc:
            msg = str(exc)
            assert "CERTIFICATE CONTRADICTED" in msg, msg
            # BOTH physical verdicts are always present, whichever one failed ...
            assert ("POSITION FAILED" if want_pos_fail else "POSITION passed") in msg, msg
            assert ("FUEL FAILED" if want_fuel_fail else "FUEL passed") in msg, msg
            # ... and so is the tick discrepancy, labelled as the diagnostic it is.
            assert "TICK (DIAGNOSTIC ONLY, never binding" in msg, msg
            assert "certified %d, observed %d, delta +0" % (
                cert.event_tick, cert.event_tick) in msg, msg
            # Both offsets are reported against their own tolerance.
            assert "km tolerance" in msg and "lbs/tick" in msg, msg
        else:  # pragma: no cover
            raise AssertionError("a physical contradiction must abort: %s" % label)
        assert aircraft.current_fuel == snapshot, (
            "%s: refused BEFORE the engine is touched" % label
        )
        assert not controller.outcome.fired

    # THE FOURTH CASE. A wildly wrong outer tick, with BOTH physical fields exact, is
    # accepted -- the whole content of this repair.
    aircraft.latitude, aircraft.longitude = cert.latitude, cert.longitude
    aircraft.current_fuel = cert.fuel_before
    accepted = FuelDamageController(plan)
    for wild in (cert.event_tick + 2, cert.event_tick + 40, cert.event_tick + 500):
        accepted._require_certificate_holds(
            aircraft=aircraft, tick=wild, fuel_before=float(cert.fuel_before)
        )
    assert accepted.maybe_apply(ctx.scenario, cert.event_tick + 500) == ego

    # AND THE +/- ONE-QUANTUM PHYSICAL ACCEPTANCE IS UNCHANGED: a state INSIDE either
    # tolerance is still accepted, at the certified tick exactly as before.
    edge = FuelDamageController(plan)
    near = _point_at(cert.event_location, 0.5 * cert.position_tolerance_km, 90.0)
    aircraft.latitude, aircraft.longitude = near.latitude, near.longitude
    edge._require_certificate_holds(
        aircraft=aircraft, tick=cert.event_tick,
        fuel_before=float(cert.fuel_before) - 0.5 * cert.fuel_tolerance,
    )


def test_g1_11d_the_certificate_construction_and_world_acceptance_are_unchanged() -> None:
    """G1.11d. The repair touches LIVE VALIDATION ONLY -- never setup-time acceptance.

    Certificate CONSTRUCTION, the tick arithmetic it rests on, the bracket it is issued
    over and the quantum both physical tolerances are derived from are all exactly as
    they were, and so is which worlds the certified walk accepts. A repair that quietly
    moved world acceptance would change the POPULATION rather than repair a check on it.
    """
    ctx = _FuelDamageCtx()
    plan = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    cert = plan.certificate

    # The quantum, the bracket and the tick arithmetic: unchanged, and still derived.
    assert CERTIFICATE_TICK_TOLERANCE == 1
    assert cert.tick_tolerance == CERTIFICATE_TICK_TOLERANCE
    assert cert.event_tick == cert.movement_count + 1
    assert set(cert.bracket_ticks) == {
        cert.event_tick - 1, cert.event_tick, cert.event_tick + 1
    }
    assert cert.fuel_before == _approx(12000.0 - cert.event_tick * 6700.0 / 3600.0)
    # Both physical tolerances are still ONE quantum plus a documented float epsilon --
    # the live check binds these two, and this is where they come from.
    assert cert.fuel_tolerance > CERTIFICATE_TICK_TOLERANCE * cert.fuel_per_tick
    assert cert.fuel_tolerance < 2.0 * cert.fuel_per_tick
    assert cert.position_tolerance_km > CERTIFICATE_TICK_TOLERANCE * cert.leg_km_per_tick
    assert cert.position_tolerance_km < 2.0 * cert.leg_km_per_tick

    # Setup-time acceptance is unchanged: this world is still certified, the audit still
    # names the same ordinal, and the certificate is still field-identical across builds.
    again = build_fuel_damage_plan(ctx, episode_seed=3, params=_certified())
    assert again.certificate.to_record() == cert.to_record()
    assert (again.eligibility_audit.selected_ordinal
            == plan.eligibility_audit.selected_ordinal)
    assert plan.eligibility_audit.selected_ego_id == plan.ego_id

    # And a world that is NOT FD-capable is still REFUSED at setup, as ordinary accounted
    # attrition -- never as the integrity fault the live check raises.
    starved = _FuelDamageCtx(fuel=1350.0)
    try:
        build_fuel_damage_plan(starved, episode_seed=3, params=_certified())
    except FuelDamageIntegrityError:  # pragma: no cover
        raise AssertionError("setup ineligibility is NOT an instrument fault")
    except FuelDamageError as exc:
        assert NO_FD_ELIGIBLE_EGO in str(exc) or REASON_INVALID_BAND in str(exc), str(exc)
    else:  # pragma: no cover
        raise AssertionError("an FD-incapable world must still be refused at setup")

def test_g1_12_the_legacy_defaults_are_pinned_at_every_construction_site() -> None:
    """G1.12. Every existing construction site obtains the merged behaviour, unasked.

    The two new policies are OPT-IN, and "opt-in" has to be true at the sites that
    already exist -- `FuelDamageParameters()` itself, and the ONE place the trainer turns
    a `TrainConfig` into one -- or an approved measurement would silently change design
    the next time it was re-derived.
    """
    default = FuelDamageParameters()
    assert default.eligibility_policy == FD_ELIGIBILITY_LEGACY_V1
    assert default.post_fd_wake_policy == POST_FD_WAKE_SINGLE_V1
    assert default.certified_eligibility is False
    assert default.completion_boundary_wakes is False

    from_trainer = TrainConfig(n_iterations=1).fuel_damage_parameters()
    assert from_trainer.eligibility_policy == FD_ELIGIBILITY_LEGACY_V1
    assert from_trainer.post_fd_wake_policy == POST_FD_WAKE_SINGLE_V1
    from_rollout = RolloutConfig().fuel_damage_parameters()
    assert from_rollout.eligibility_policy == FD_ELIGIBILITY_LEGACY_V1
    assert from_rollout.post_fd_wake_policy == POST_FD_WAKE_SINGLE_V1

    # Both are validated as CLOSED sets, so a typo cannot silently mean "legacy".
    for bad in ("certified", "", None, "legacy_selected_ego_v2"):
        try:
            FuelDamageParameters(eligibility_policy=bad).validate()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("eligibility policy %r must be rejected" % (bad,))
        try:
            FuelDamageParameters(post_fd_wake_policy=bad).validate()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("wake policy %r must be rejected" % (bad,))

    assert set(FD_ELIGIBILITY_POLICIES) == {
        FD_ELIGIBILITY_LEGACY_V1, FD_ELIGIBILITY_CERTIFIED_V1
    }
    assert set(POST_FD_WAKE_POLICIES) == {
        POST_FD_WAKE_SINGLE_V1, POST_FD_WAKE_COMPLETION_BOUNDARY_V1
    }
    # The parameter record names both, so a run artifact says which design it used.
    record = default.to_record()
    assert record["eligibility_policy"] == FD_ELIGIBILITY_LEGACY_V1
    assert record["post_fd_wake_policy"] == POST_FD_WAKE_SINGLE_V1
    assert record["certified_eligibility"] is False


def test_g1_13_the_transcribed_engine_constants_and_leg_match_the_frozen_engine() -> None:
    """G1.13 / A4. The transcription is checked against the ENGINE, not against a literal.

    This layer stays BLADE-free at import time (that is what makes it hand-testable and
    safe inside the tick loop's closure), so the movement constant it needs is
    TRANSCRIBED. A transcription is only as good as the check on it, and comparing it to
    another literal would check nothing -- so the frozen engine is imported HERE, in the
    test, and asked directly.

    THE STRONGER CLAIM is the second half: the whole per-tick leg, floor and all, is
    compared against `blade.utils.utils.get_next_coordinates` itself -- and then the whole
    multi-tick recursion is compared against repeatedly calling it, because the certified
    event TICK is what the certificate promises and a per-call agreement that drifted over
    hundreds of ticks would not be enough.
    """
    try:
        import blade.utils.constants as engine_constants
        import blade.utils.utils as engine_utils
    except ImportError:  # pragma: no cover - BLADE is absent in some environments
        if pytest is not None:
            pytest.skip("the vendored BLADE engine is not importable here")
        return

    assert KILOMETERS_TO_NAUTICAL_MILES == engine_constants.KILOMETERS_TO_NAUTICAL_MILES
    assert NAUTICAL_MILES_TO_METERS == float(engine_constants.NAUTICAL_MILES_TO_METERS)
    # They are DIFFERENT constants used for DIFFERENT questions, and this layer must not
    # quietly substitute one for the other.
    assert KILOMETERS_TO_NAUTICAL_MILES != 1000.0 / NAUTICAL_MILES_TO_METERS

    speed = 1303.0
    start = (32.0, 35.0)
    dest = (34.0, 35.0)

    # (1) ONE tick: the engine's own displacement is this module's own leg.
    remaining = engine_utils.get_distance_between_two_points(*start, *dest)
    moved_to = engine_utils.get_next_coordinates(*start, *dest, speed)
    engine_leg = engine_utils.get_distance_between_two_points(*start, *moved_to)
    mine = engine_leg_distance_km(remaining, speed_knots=speed)
    assert abs(mine - engine_leg) <= 1e-9 * engine_leg, (mine, engine_leg)

    # (2) THE WHOLE FLIGHT: drive the engine tick by tick and compare the crossing.
    states = predict_leg_states(leg_length_km=remaining, speed_knots=speed)
    predicted = next(i for i, st in enumerate(states) if st.progress >= 0.30)

    lat, lon = start
    engine_crossing = None
    for movements in range(len(states) + 2):
        left = engine_utils.get_distance_between_two_points(lat, lon, *dest)
        if (remaining - left) / remaining >= 0.30:
            engine_crossing = movements
            break
        if left < WAYPOINT_SNAP_KM:  # pragma: no cover - not reached before 30 %
            break
        lat, lon = engine_utils.get_next_coordinates(lat, lon, *dest, speed)

    assert engine_crossing is not None, "the engine never crossed the threshold"
    assert abs(engine_crossing - predicted) <= CERTIFICATE_TICK_TOLERANCE, (
        "the tick-aware prediction (%d movements) and the engine's own flight (%d) "
        "differ by more than the certified bracket" % (predicted, engine_crossing)
    )
    # In practice they agree EXACTLY; the bracket exists for the boundary case, not as
    # slack the prediction routinely uses.
    assert engine_crossing == predicted, (engine_crossing, predicted)
    # The remaining distance the engine really has at the crossing matches the recursion
    # to well under one metre, so the certified event POSITION is the engine's own.
    engine_left = engine_utils.get_distance_between_two_points(lat, lon, *dest)
    assert abs(engine_left - states[predicted].remaining_km) < 1e-3, (
        engine_left, states[predicted].remaining_km
    )


def _approx(value, tol=1e-9):
    """Tiny local tolerance helper (this file also runs without pytest)."""
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - float(value)) <= tol * max(1.0, abs(float(value)))

        def __repr__(self):  # pragma: no cover - only used in a failure message
            return "approx(%r)" % (value,)

    return _Approx()


# --- G1.14: the TERMINAL half of the certified promise ------------------------
# `maybe_apply` guards the promise from the inside; this guards it from the outside. The
# four cells are enumerated once, here, because the whole risk is that the invariant
# leaks onto the LEGACY policy (which an approved measurement was taken on) or onto a
# CLEAN episode (which is supposed to finish without firing).

def _stalled_run(params, *, seed=3, max_ticks=4, burn_fuel=False):
    """Run a REAL `run_episode` on a world where the event CANNOT fire.

    Nothing moves (``step_km=0``), so the selected ego never reaches 30 % of its first
    leg and the episode truncates with the event unfired. Nothing burns either, so any
    change to any aircraft's fuel afterwards could only have come from the terminal check
    itself -- which is exactly what one of the assertions below is for.
    """
    ctx = _FuelDamageCtx()
    ctx.env = _StubEnv(ctx.scenario, step_km=0.0, burn_fuel=burn_fuel)
    controller = build_fuel_damage_controller(ctx, episode_seed=seed, params=params)
    fuel_before = {str(a.id): a.current_fuel for a in ctx.scenario.aircraft}
    raised = None
    result = None
    try:
        result = graph_tick_loop.run_episode(
            None, ctx, GraphObservationConfig(detection_range_km=50.0),
            max_ticks=max_ticks, fuel_damage=controller,
        )
    except BaseException as exc:  # noqa: BLE001 - the abort itself is under test
        raised = exc
    return {
        "ctx": ctx, "controller": controller, "result": result, "raised": raised,
        "fuel_before": fuel_before,
        "fuel_after": {str(a.id): a.current_fuel for a in ctx.scenario.aircraft},
    }


def test_g1_14_a_certified_damaged_event_that_never_fires_is_an_integrity_fault() -> None:
    """G1.14 (cell 2). CERTIFIED + DAMAGED + never fired -> `FuelDamageIntegrityError`.

    The other half of the certified promise. A world proven FD-capable before a tick was
    paid for that then ends without delivering its own event has not had an uneventful
    episode -- its certificate did not materialize, and returning it would admit a world
    whose promise failed into a scientific population as a SUCCESSFUL damaged episode.

    The fixture makes the non-fire structural rather than lucky: nothing moves, so the
    threshold is unreachable by construction.
    """
    for mode in (FuelDamageMode.FORCED_MILD, FuelDamageMode.FORCED_SEVERE,
                 FuelDamageMode.FORCED_DAMAGED):
        out = _stalled_run(_certified(mode=mode))
        raised = out["raised"]
        assert isinstance(raised, FuelDamageIntegrityError), (mode, raised)
        assert not isinstance(raised, FuelDamageError), (
            "it must not be catchable as ordinary attrition"
        )
        assert "CERTIFICATE NOT REALIZED" in str(raised), str(raised)
        assert out["result"] is None, "no EpisodeResult may be returned"
        assert out["controller"].outcome.fired is False


def test_g1_15_the_terminal_check_mutates_nothing_to_satisfy_itself() -> None:
    """G1.15 (cell 2, purity). A certificate that did not materialize is REPORTED.

    The one repair a terminal invariant must never make is applying the event late so it
    can pass. Nothing burns in this fixture, so EVERY aircraft's fuel must be exactly
    what it was at launch -- the selected ego's included -- and the controller must still
    report the event as unfired with no live measurement attached.
    """
    out = _stalled_run(_certified())
    assert isinstance(out["raised"], FuelDamageIntegrityError), out["raised"]
    assert out["fuel_after"] == out["fuel_before"], (
        "the terminal check applied fuel damage to satisfy itself: %r -> %r"
        % (out["fuel_before"], out["fuel_after"])
    )
    outcome = out["controller"].outcome
    assert outcome.fired is False
    assert outcome.event_tick is None and outcome.fuel_after is None
    assert outcome.live_rtb_fuel_floor is None and outcome.observed_progress is None
    # Calling it again is still a pure raise -- no state was latched by the first call.
    try:
        out["controller"].require_certified_event_realized()
    except FuelDamageIntegrityError:
        pass
    assert out["controller"].outcome.fired is False


def test_g1_16_a_certified_clean_episode_may_finish_without_firing() -> None:
    """G1.16 (cell 3). CLEAN + certified: `fired == False` is the CORRECT outcome.

    The clean member runs the whole eligibility walk and carries a certificate, but that
    certificate is a COUNTERFACTUAL -- it names the ego its matched mild and severe
    siblings will damage. Nothing was scheduled to fire here, so the terminal invariant
    must stay silent, and this is the cell most at risk from a check written as
    "certified => must have fired".
    """
    out = _stalled_run(_certified(mode=FuelDamageMode.FORCED_CLEAN))
    assert out["raised"] is None, out["raised"]
    assert out["result"] is not None and out["result"].ended == "truncated"
    plan = out["controller"].plan
    assert plan.condition == CONDITION_CLEAN and plan.ego_id is None
    assert plan.is_certified and plan.certificate is not None
    assert plan.eligibility_audit.selected_ego_id is not None, (
        "the clean member still identifies the counterfactual ego"
    )
    assert out["controller"].outcome.fired is False


def test_g1_17_the_legacy_policy_keeps_its_historical_non_fire_behaviour() -> None:
    """G1.17 (cell 4). LEGACY + DAMAGED + never fired: unchanged, and NOT a fault.

    The approved FD-VARIABLE-SEVERITY-v1 measurement contains exactly one such episode
    (successful damaged training seed 424, whose selected ego returned before reaching
    the 0.30 leg-progress trigger). It was a recorded observation there, and it must stay
    one: putting a terminal requirement on the legacy policy would change the behaviour
    that measurement was taken on rather than extend it.
    """
    for mode in (FuelDamageMode.FORCED_DAMAGED, FuelDamageMode.FORCED_SEVERE,
                 FuelDamageMode.FORCED_CLEAN, FuelDamageMode.SEEDED_MIXTURE):
        out = _stalled_run(FuelDamageParameters(mode=mode))
        assert out["raised"] is None, (mode, out["raised"])
        assert out["result"] is not None, mode
        assert out["controller"].plan.is_certified is False, mode
        assert out["controller"].outcome.fired is False, mode

    # The predicate itself is a no-op for a legacy plan even when called directly, so the
    # invariance does not depend on the tick loop's own branching.
    ctx = _FuelDamageCtx()
    legacy = build_fuel_damage_controller(
        ctx, episode_seed=3,
        params=FuelDamageParameters(mode=FuelDamageMode.FORCED_DAMAGED),
    )
    assert legacy.plan.is_damaged and not legacy.outcome.fired
    legacy.require_certified_event_realized(scenario=ctx.scenario, ticks=10)


def test_g1_18_a_realized_certified_event_passes_the_terminal_check() -> None:
    """G1.18 (cell 1). The invariant must not reject the VALID fired path.

    A guard that also failed the healthy case would be worse than no guard: it would
    abort every certified damaged run. The event is fired here through the production
    path at the certified state, and the terminal check then returns silently.
    """
    ctx = _FuelDamageCtx()
    controller = build_fuel_damage_controller(
        ctx, episode_seed=3, params=_certified(mode=FuelDamageMode.FORCED_SEVERE)
    )
    cert = controller.plan.certificate
    aircraft = next(a for a in ctx.scenario.aircraft
                    if str(a.id) == str(controller.plan.ego_id))
    aircraft.latitude, aircraft.longitude = cert.latitude, cert.longitude
    aircraft.current_fuel = cert.fuel_before

    assert controller.maybe_apply(ctx.scenario, cert.event_tick) == controller.plan.ego_id
    assert controller.outcome.fired is True
    controller.require_certified_event_realized(
        scenario=ctx.scenario, ticks=cert.event_tick + 1
    )  # must not raise


def test_g1_19_the_terminal_check_lives_at_exactly_one_seam() -> None:
    """G1.19. ONE call site, at the place that owns episode exit.

    The packet's requirement, and the reason it matters: a predicate duplicated across
    `graph_train` and `graph_rollout` would be two chances to disagree about when a
    certified world counts as having failed. `run_episode` is the single path every
    scientific consumer goes through, so it is the only caller.
    """
    tick_loop_src = inspect.getsource(graph_tick_loop)
    assert tick_loop_src.count("require_certified_event_realized(") == 1, (
        "the terminal invariant must be invoked exactly once in the tick loop"
    )
    assert "require_certified_event_realized" not in inspect.getsource(graph_train), (
        "the trainer must not duplicate the predicate -- it only ROUTES the exception"
    )
    from match_aou.rl.training import graph_rollout
    assert "require_certified_event_realized" not in inspect.getsource(graph_rollout), (
        "the rollout harness must not duplicate the predicate either"
    )
    # And it runs BEFORE the recording export, so an aborted episode never leaves a
    # playback file that no manifest will list (the roster-integrity lesson).
    check_at = tick_loop_src.index("require_certified_event_realized(")
    export_at = tick_loop_src.index("ctx.game.export_recording()")
    assert check_at < export_at, (
        "the terminal check must precede the playback export"
    )


# =============================================================================
# GENERALIZED-V1 step 2 -- G2: POST-FD COMPLETION BOUNDARIES (PO2 / PO3)
# =============================================================================

_TA = _point_at(_BASE, 120.0, 45.0)     # the damaged ego's first assignment
_TB = _point_at(_BASE, 200.0, 45.0)     # its second -- 80 km beyond the first
_TPEER = _point_at(_BASE, 250.0, 200.0)  # the peer's, far off the ego's route


class _BoundaryEnv:
    """Nothing moves; a scripted kill removes ONE target after a chosen step.

    Movement is deliberately absent: these tests are about WHEN a decision is offered,
    and a fixture that also flew the aircraft would make the tick a boundary landed on an
    accident of the step size rather than a statement about the seam.
    """

    def __init__(self, scenario, *, kill_after=None, target_id=None, popup=None):
        self.scenario = scenario
        self.kill_after = kill_after
        self.target_id = target_id
        self.popup = popup
        self.n_steps = 0
        self.closed = False

    def step(self, _action):
        self.n_steps += 1
        if self.kill_after is not None and self.n_steps == self.kill_after:
            self.scenario.airbases = [
                b for b in self.scenario.airbases if str(b.id) != str(self.target_id)
            ]
            if self.popup is not None:
                self.scenario.airbases.append(self.popup)
        return self.scenario, 0.0, False, False, {}

    def close(self):
        self.closed = True


class _BoundaryWorld:
    """One damaged ego with TWO assignments, one peer, and a real `GraphPlanExecutor`.

    The ego starts ON its first target, so which assignment it confirms is decided by
    the LIVENESS of that target and never by the proximity gate. The peer sits at the
    base with its own far assignment, so every "peers are untouched" assertion is about
    an ego that really had something to do.
    """

    def __init__(self, *, ego_at=None, peer_assignment=True):
        self.ego, self.peer = "ego-damaged", "peer-quiet"
        self.agent_ids = [self.ego, self.peer]
        here = ego_at or _TA
        self.ego_ac = _StubAircraft(self.ego, here.latitude, here.longitude)
        self.peer_ac = _StubAircraft(self.peer, _BASE.latitude, _BASE.longitude)
        base = _StubAirbase("base-blue", _BASE.latitude, _BASE.longitude)
        self.scenario = _StubScenario(
            aircraft=[self.ego_ac, self.peer_ac], airbases=[base]
        )
        for tid, loc in (("tA", _TA), ("tB", _TB), ("tPeer", _TPEER)):
            self.scenario.airbases.append(
                _StubAirbase(tid, loc.latitude, loc.longitude, side_id=_RED_SIDE,
                             side_color="red", name="Red %s" % tid)
            )
        self.tasks = [
            _attack_task("tA", _TA), _attack_task("tB", _TB), _attack_task("tPeer", _TPEER)
        ]
        self.a_init = {
            self.ego: [(0, 0, 0), (1, 0, 0)],
            self.peer: [(2, 0, 0)] if peer_assignment else [],
        }
        self.beliefs = {
            aid: _StubBelief(list(self.tasks),
                             {k: list(v) for k, v in self.a_init.items()})
            for aid in self.agent_ids
        }
        self.agents = [
            Agent(location=Location(_BASE.latitude, _BASE.longitude), capabilities=[],
                  budget=12000.0, move_cost_function=lambda s, d: 0.0, speed=1303.0,
                  return_location=Location(_BASE.latitude, _BASE.longitude),
                  agent_id=aid, side_color="blue", home_base_id="base-blue")
            for aid in self.agent_ids
        ]
        self.executor = GraphPlanExecutor(
            tasks=self.tasks, solution=self.a_init, agents=self.agents,
            arrival_threshold_km=50.0,
        )
        self.game = _StubGame(self.scenario)
        self.observation = self.scenario
        self.env = _BoundaryEnv(self.scenario)
        self.record = False
        self.oracle_tasks = list(self.tasks)
        self.oracle_solution = dict(self.a_init)
        self.known_target_ids = ("tA", "tB", "tPeer")
        self.executed_target_ids = ("tA", "tB", "tPeer")
        self.split_meta = {}
        self.placements = ()

    # -- the fuel-damage side --------------------------------------------------

    def armed_controller(self, *, policy=POST_FD_WAKE_COMPLETION_BOUNDARY_V1):
        """A controller whose event has ALREADY fired on the ego, deterministically.

        The plan is built through the PURE `plan_fuel_damage` with the ego named
        explicitly, so the fixture does not depend on which ego a seeded draw happens to
        pick, and the event is then applied at the ego's real position. Everything after
        that -- arming, the boundary walk, the wake -- is production code.
        """
        params = FuelDamageParameters(
            mode=FuelDamageMode.FORCED_DAMAGED, post_fd_wake_policy=policy
        )
        plan = plan_fuel_damage(
            condition=CONDITION_DAMAGED, mode=params.mode, derived_seed=0,
            eligible_ego_ids=(self.ego,), ego_id=self.ego,
            launch_point=Location(_BASE.latitude, _BASE.longitude),
            home_base=Location(_BASE.latitude, _BASE.longitude),
            route_points=[_TA, _TB], speed_knots=1303.0, fuel_rate=6700.0,
            max_fuel=12000.0, fuel_at_launch=12000.0, params=params,
        )
        controller = FuelDamageController(plan)
        fired = controller.maybe_apply(self.scenario, 0)
        assert fired == self.ego, "the fixture's event must fire on the damaged ego"
        return controller

    def belief_fingerprint(self, ego_id):
        return _belief_fingerprint(self.beliefs[ego_id])


def _run_boundary(world, controller, *, max_ticks=3):
    """Drive the REAL `run_episode`, recording wakes, trigger flags and commands."""
    wakes, flags, commands = [], [], []
    real_decide = graph_tick_loop.decide_triggers
    real_next = world.executor.next_actions

    def spy_decide(belief_tasks, belief_solution, sensed, eta=None, *,
                   ego_id, clock, fuel_damage=False, post_fd_completion=False):
        flags.append((int(clock), str(ego_id), bool(fuel_damage),
                      bool(post_fd_completion)))
        return real_decide(belief_tasks, belief_solution, sensed, ego_id=ego_id,
                           clock=clock, fuel_damage=fuel_damage,
                           post_fd_completion=post_fd_completion)

    def spy_next(observation):
        issued = real_next(observation)
        commands.append(list(issued))
        return issued

    def spy_wake(_policy, ego_id, _obs, _belief, _executor, _cfg, tick, **_kw):
        wakes.append((int(tick), str(ego_id)))
        return graph_tick_loop.Transition(
            gobs=None, ego_id=str(ego_id), tick=int(tick),
            meta_action=int(MetaAction.PLAN_COMPLIANCE), node_v=0,
            log_prob=0.0, entropy=0.0,
        )

    world.executor.next_actions = spy_next
    saved = (graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision)
    graph_tick_loop.decide_triggers = spy_decide
    graph_tick_loop._wake_decision = spy_wake
    try:
        result = graph_tick_loop.run_episode(
            None, world, GraphObservationConfig(detection_range_km=50.0),
            max_ticks=max_ticks, fuel_damage=controller,
        )
    finally:
        graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision = saved
        world.executor.next_actions = real_next
    return {"result": result, "wakes": wakes, "flags": flags, "commands": commands}


def test_g2_1_only_the_actually_damaged_ego_enters_post_fd_adaptation() -> None:
    """G2.1 / B1. Armed by the REAL mutation, never by a schedule or a clean plan."""
    world = _BoundaryWorld()
    params = FuelDamageParameters(
        mode=FuelDamageMode.FORCED_DAMAGED,
        post_fd_wake_policy=POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
    )
    plan = plan_fuel_damage(
        condition=CONDITION_DAMAGED, mode=params.mode, derived_seed=0,
        eligible_ego_ids=(world.ego,), ego_id=world.ego,
        launch_point=Location(_BASE.latitude, _BASE.longitude),
        home_base=Location(_BASE.latitude, _BASE.longitude),
        route_points=[_TA, _TB], speed_knots=1303.0, fuel_rate=6700.0,
        max_fuel=12000.0, fuel_at_launch=12000.0, params=params,
    )
    controller = FuelDamageController(plan)
    assert controller.boundary_wakes_enabled, "the policy grants boundary wakes"
    assert controller.post_fd_ego is None, "nothing is armed before the event fires"
    controller.maybe_apply(world.scenario, 0)
    assert controller.post_fd_ego == world.ego

    # A CLEAN controller -- even under the same policy -- arms nothing at all.
    clean = FuelDamageController(plan_fuel_damage(
        condition=CONDITION_CLEAN, mode=FuelDamageMode.FORCED_CLEAN, derived_seed=0,
        eligible_ego_ids=(world.ego,), ego_id=None, launch_point=None, home_base=None,
        route_points=None, speed_knots=None, fuel_rate=None, max_fuel=None,
        fuel_at_launch=None,
        params=FuelDamageParameters(
            mode=FuelDamageMode.FORCED_CLEAN,
            post_fd_wake_policy=POST_FD_WAKE_COMPLETION_BOUNDARY_V1,
        ),
    ))
    assert clean.boundary_wakes_enabled is False and clean.post_fd_ego is None
    assert clean.maybe_apply(world.scenario, 0) is None
    assert clean.post_fd_outcome.armed is False

    # And the DEFAULT policy never arms it, however damaged the ego is. A FRESH world:
    # the event above already mutated this one's fuel, and a second application would be
    # refused for that reason rather than for the reason under test.
    fresh = _BoundaryWorld()
    legacy = FuelDamageController(replace(
        plan, post_fd_wake_policy=POST_FD_WAKE_SINGLE_V1
    ))
    assert legacy.boundary_wakes_enabled is False
    assert legacy.maybe_apply(fresh.scenario, 0) == fresh.ego
    assert legacy.post_fd_ego is None and legacy.post_fd_outcome.armed is False


def test_g2_2_an_alive_target_or_a_bare_attack_produces_no_boundary() -> None:
    """G2.2 / B5. Emitting a salvo is not a completion; a live target is not a boundary."""
    world = _BoundaryWorld()
    controller = world.armed_controller()
    out = _run_boundary(world, controller, max_ticks=2)

    assert any("handle_aircraft_attack('%s', 'tA')" % world.ego in c
               for tick_cmds in out["commands"] for c in tick_cmds), out["commands"]
    assert controller.post_fd_outcome.boundaries_confirmed == 0
    assert out["wakes"] == [], "a live target must wake nobody"
    assert all(not post_fd for _t, _e, _fd, post_fd in out["flags"]), out["flags"]


def test_g2_3_a_peer_kill_far_away_produces_no_boundary() -> None:
    """G2.3 / B5. A target that dies while the damaged ego is far off is NOT a boundary.

    This is the no-communication rule restated for this seam: the boundary is the ego's
    OWN proximity-gated confirmation, never a peer's outcome learned some other way.
    """
    # 48 km along the 120 km first leg: PAST the 30 % threshold (so the event really
    # fires) and still 72 km from tA, well outside the 50 km radius (so the kill below is
    # one the ego cannot confirm from where it is).
    world = _BoundaryWorld(ego_at=_point_at(_BASE, 48.0, 45.0))
    controller = world.armed_controller()
    world.env = _BoundaryEnv(world.scenario, kill_after=1, target_id="tA")

    out = _run_boundary(world, controller, max_ticks=3)

    assert world.scenario.get_target("tA") is None, "the fixture must really kill tA"
    assert controller.post_fd_outcome.boundaries_confirmed == 0, (
        "a kill the ego could not confirm from where it is must not be a boundary"
    )
    assert out["wakes"] == [], out["wakes"]
    assert (world.ego, "tA") not in world.executor.done


def test_g2_4_a_local_confirmation_wakes_once_before_the_next_movement() -> None:
    """G2.4 / B5. ONE wake, at the boundary, BEFORE the ego commits to the next leg.

    The load-bearing ordering assertion is the last one: the move toward the SECOND
    assignment must not appear on any tick before the decision. If the reconciliation had
    stayed in Phase 2, the ego would have flown first and been asked afterwards.
    """
    world = _BoundaryWorld()
    controller = world.armed_controller()
    world.env = _BoundaryEnv(world.scenario, kill_after=1, target_id="tA")

    out = _run_boundary(world, controller, max_ticks=3)

    assert (world.ego, "tA") in world.executor.done, "the ego confirmed its own kill"
    outcome = controller.post_fd_outcome
    assert outcome.boundaries_confirmed == 1, outcome.to_record()
    assert outcome.boundaries_with_remaining_mission == 1
    assert outcome.boundary_wakes == 1
    assert outcome.boundaries[0].confirmed_target_ids == ("tA",)
    assert outcome.boundaries[0].remaining_mission is True
    assert outcome.boundary_meta_actions == (int(MetaAction.PLAN_COMPLIANCE),)

    assert out["wakes"] == [(1, world.ego)], out["wakes"]
    boundary_flags = [f for f in out["flags"] if f[3]]
    assert boundary_flags == [(1, world.ego, False, True)], out["flags"]

    # The IMMEDIATE fuel-damage record keeps its own, separate meaning.
    assert controller.outcome.wake_occurred is False
    assert controller.outcome.wake_meta_action is None

    move_ticks = [
        tick for tick, cmds in enumerate(out["commands"])
        if any(c.startswith("move_aircraft('%s'" % world.ego) for c in cmds)
    ]
    assert move_ticks and min(move_ticks) == 1, (
        "the ego must not commit movement toward its next assignment before the "
        "boundary decision (moves at %r)" % (move_ticks,)
    )


def test_g2_5_the_final_assignment_completes_without_a_useless_wake() -> None:
    """G2.5 / B3. A completion that ends the mission is recorded, and wakes nobody."""
    world = _BoundaryWorld()
    world.a_init[world.ego] = [(0, 0, 0)]  # ONE assignment: tA is the whole mission
    for belief in world.beliefs.values():
        belief.solution = {k: list(v) for k, v in world.a_init.items()}
    world.executor = GraphPlanExecutor(
        tasks=world.tasks, solution=world.a_init, agents=world.agents,
        arrival_threshold_km=50.0,
    )
    controller = world.armed_controller()
    world.env = _BoundaryEnv(world.scenario, kill_after=1, target_id="tA")

    out = _run_boundary(world, controller, max_ticks=3)

    outcome = controller.post_fd_outcome
    assert outcome.boundaries_confirmed == 1, outcome.to_record()
    assert outcome.boundaries_terminal == 1
    assert outcome.boundaries_with_remaining_mission == 0
    assert outcome.boundary_wakes == 0, "nothing was left to decide"
    assert out["wakes"] == [], out["wakes"]
    # Normal RTB behaviour proceeds through the executor's existing empty-plan path.
    assert any("aircraft_return_to_base('%s')" % world.ego in c
               for cmds in out["commands"] for c in cmds), out["commands"]


def test_g2_6_several_confirmations_in_one_reconciliation_coalesce_to_one_decision() -> None:
    """G2.6 / B3. Two heads retired in one pass are ONE boundary and ONE wake."""
    world = _BoundaryWorld(ego_at=_TA)
    # A third assignment the ego can also confirm from here: tC sits ON tA's position.
    world.tasks.append(_attack_task("tC", _TA))
    world.a_init[world.ego] = [(0, 0, 0), (3, 0, 0), (1, 0, 0)]
    world.scenario.airbases.append(
        _StubAirbase("tC", _TA.latitude, _TA.longitude, side_id=_RED_SIDE,
                     side_color="red", name="Red tC")
    )
    for belief in world.beliefs.values():
        belief.tasks = list(world.tasks)
        belief.solution = {k: list(v) for k, v in world.a_init.items()}
    world.executor = GraphPlanExecutor(
        tasks=world.tasks, solution=world.a_init, agents=world.agents,
        arrival_threshold_km=50.0,
    )
    controller = world.armed_controller()

    class _DoubleKillEnv(_BoundaryEnv):
        def step(self, action):
            self.n_steps += 1
            if self.n_steps == 1:
                self.scenario.airbases = [
                    b for b in self.scenario.airbases
                    if str(b.id) not in ("tA", "tC")
                ]
            return self.scenario, 0.0, False, False, {}

    world.env = _DoubleKillEnv(world.scenario)
    out = _run_boundary(world, controller, max_ticks=3)

    outcome = controller.post_fd_outcome
    assert outcome.boundaries_confirmed == 1, outcome.to_record()
    assert set(outcome.boundaries[0].confirmed_target_ids) == {"tA", "tC"}
    assert outcome.boundary_wakes == 1
    assert out["wakes"] == [(1, world.ego)], out["wakes"]
    assert world.beliefs[world.ego].solution[world.ego] == [(1, 0, 0)], (
        "both confirmed assignments must leave the ego's own belief"
    )


def test_g2_7_a_boundary_and_a_popup_coalesce_into_one_transition() -> None:
    """G2.7 / B4. Simultaneous triggers share ONE `wake` boolean, so ONE decision."""
    world = _BoundaryWorld()
    controller = world.armed_controller()
    popup_at = _point_at(_TA, 10.0, 90.0)  # inside the ego's radius, in nobody's belief
    world.env = _BoundaryEnv(
        world.scenario, kill_after=1, target_id="tA",
        popup=_StubAirbase("popup", popup_at.latitude, popup_at.longitude,
                           side_id=_RED_SIDE, side_color="red", name="Red popup"),
    )

    out = _run_boundary(world, controller, max_ticks=3)

    at_boundary = [w for w in out["wakes"] if w[0] == 1]
    assert at_boundary == [(1, world.ego)], (
        "a boundary and a pop-up on one snapshot must produce exactly ONE decision, "
        "got %r" % (out["wakes"],)
    )
    # The pop-up really was there: the trigger appended it to the ego's own belief.
    ego_targets = [str(t.steps[0].target_id) for t in world.beliefs[world.ego].tasks]
    assert "popup" in ego_targets, ego_targets
    assert controller.post_fd_outcome.boundary_wakes == 1


def test_g2_8_adaptation_deactivates_on_rtb_or_death_with_a_recorded_reason() -> None:
    """G2.8 / B1. The state ends when the ego can no longer reach a boundary."""
    world = _BoundaryWorld()
    controller = world.armed_controller()
    world.executor.rtb_issued[world.ego] = True
    graph_tick_loop._post_fd_boundary(controller, world, world.scenario, 4)
    assert controller.post_fd_ego is None
    assert controller.post_fd_outcome.deactivation_reason == POST_FD_DEACTIVATED_RTB
    assert controller.post_fd_outcome.active is False

    dead_world = _BoundaryWorld()
    dead_controller = dead_world.armed_controller()
    dead_world.executor.dead.add(dead_world.ego)
    graph_tick_loop._post_fd_boundary(
        dead_controller, dead_world, dead_world.scenario, 9
    )
    assert dead_controller.post_fd_ego is None
    assert (dead_controller.post_fd_outcome.deactivation_reason
            == POST_FD_DEACTIVATED_DEAD)


def test_g2_9_no_target_is_re_executed_through_stale_done_state() -> None:
    """G2.9 / B2. Reconciling early cannot resurrect work, and is IDEMPOTENT."""
    world = _BoundaryWorld()
    controller = world.armed_controller()
    world.env = _BoundaryEnv(world.scenario, kill_after=1, target_id="tA")
    _run_boundary(world, controller, max_ticks=4)

    assert (world.ego, "tA") in world.executor.done
    # A confirmed assignment is gone from the ego's belief AND from its executor slice,
    # and no later tick re-issues an attack on it.
    assert world.beliefs[world.ego].solution[world.ego] == [(1, 0, 0)]
    assert world.executor.plans[world.ego] == [(1, 0, 0)]
    later = world.executor.next_actions(world.scenario)
    assert not any("'tA'" in c for c in later), later
    # A second reconciliation confirms nothing further.
    assert world.executor.reconcile_confirmed_for_ego(world.ego, world.scenario) == ()


# =============================================================================
# GENERALIZED-V1 step 2 -- G3: NO-COMMUNICATION AND THE UNCHANGED ACTOR (PO3)
# =============================================================================

def test_g3_1_only_the_damaged_egos_belief_and_slice_change_at_a_boundary() -> None:
    """G3.1 / PO3. Peer beliefs and peer executor slices stay byte-identical."""
    world = _BoundaryWorld()
    controller = world.armed_controller()
    world.env = _BoundaryEnv(world.scenario, kill_after=1, target_id="tA")

    peer_before = world.belief_fingerprint(world.peer)
    peer_plan_before = list(world.executor.plans[world.peer])
    ego_before = world.belief_fingerprint(world.ego)

    _run_boundary(world, controller, max_ticks=3)

    assert world.belief_fingerprint(world.peer) == peer_before, (
        "the peer's private belief must be untouched by the damaged ego's boundary"
    )
    assert world.executor.plans[world.peer] == peer_plan_before
    assert world.belief_fingerprint(world.ego) != ego_before, (
        "control: the DAMAGED ego's own belief really did change, so the assertion "
        "above is not vacuous"
    )
    # And the peer's own confirmations are untouched: `done` is keyed per (ego, target).
    assert not any(pair[0] == world.peer for pair in world.executor.done), (
        world.executor.done
    )


def test_g3_2_no_generalized_v1_quantity_reaches_the_graph_observation() -> None:
    """G3.2 / PO3. No certificate, severity, policy or post-FD flag is observable.

    Handoff 3l.2 / CLAUDE.md Sec 3: the acting path still reads only the ego's own
    sensing and its own fuel. The feature widths are pinned, and the observation's own
    field names are checked against the merged set -- a new privileged column could not
    be added without this failing.
    """
    ctx = _FuelDamageCtx()
    ego = ctx.agent_ids[0]
    gobs = build_graph_observation(
        scenario=ctx.scenario, agent_id=ego,
        current_plan=ctx.a_init[ego], current_time=0,
        tasks=ctx.beliefs[ego].tasks, solution=ctx.a_init,
        precedence_relations=[], config=GraphObservationConfig(detection_range_km=50.0),
    )
    assert TASK_FEATURE_DIM == 6, "the task feature width is a locked contract"
    assert int(gobs.task_features.shape[1]) == TASK_FEATURE_DIM
    assert int(gobs.agent_features.shape[1]) == 1, "agents carry fuel_norm and nothing else"

    banned = ("certificate", "severity", "eligibility", "post_fd", "boundary",
              "policy", "hidden", "ordinal")
    for field_name in vars(gobs):
        assert not any(word in field_name.lower() for word in banned), field_name

    # Peers stay FEATURELESS, so no peer fuel or peer completion can leak.
    peer_rows = [
        i for i, aid in enumerate(gobs.agent_ids) if str(aid) != str(ego)
    ]
    assert peer_rows, "the fixture must have peers for this to mean anything"
    for row in peer_rows:
        assert float(gobs.agent_features[row, 0]) == 0.0


def test_g3_3_the_action_set_is_unchanged() -> None:
    """G3.3 / B4. No trim-tail, no new meta-action, no new selection surface."""
    assert NUM_META_ACTIONS == 3
    assert [m.name for m in MetaAction] == [
        "PLAN_COMPLIANCE", "OPPORTUNISTIC_ENGAGEMENT", "SELF_PRESERVATION_ABORT"
    ]
    # The new trigger kind is APPEND-ONLY and carries the non-task sentinel.
    assert int(TriggerKind.POP_UP) == 0 and int(TriggerKind.PEER_OVERDUE) == 1
    assert int(TriggerKind.FUEL_DAMAGE) == 2
    assert int(TriggerKind.POST_FD_COMPLETION) == 3
    _t, _s, wake, events = decide_triggers(
        [], {}, {}, ego_id="ego", clock=0.0, post_fd_completion=True
    )
    assert wake is True
    assert events == [(TriggerKind.POST_FD_COMPLETION, NO_TASK_INDEX)]


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
