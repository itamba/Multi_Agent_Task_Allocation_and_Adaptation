"""
GENERALIZED-V1 task 3 -- the event-conditioned MATCH-AOU continuation reference.

Three proof obligations, in three sections:

  PO1  HISTORICAL PRESERVATION AND SOLVE BUDGET. The DEFAULT ``static_t0_v1`` policy is
       structurally untouched: setup still performs the full t=0 reference solve, the
       tick loop produces NO reference, and the reward runs the byte-for-byte historical
       arithmetic. Under the opt-in policy an accepted episode still costs EXACTLY TWO
       MATCH-AOU solves -- the second one MOVES, it is never added -- and no harness
       selects the policy.

  PO2  EVENT-STATE CORRECTNESS AND ISOLATION. The continuation agents carry the REAL
       post-mutation fuel and live positions; the checkpoint runs before the post-FD
       boundary, before every trigger, before ``central.capture`` and before the actor
       decision; simulation time does not advance across it; it mutates nothing; dead and
       RTB-committed egos are excluded with recorded reasons; and a FAILED solve is
       distinguishable from an accepted solve that allocated nothing.

  PO3  REWARD / ACCOUNTING CORRECTNESS. Hand-computed fixtures for
       ``U_ref = U_prefix + U_cont_ref`` and ``U_achieved = U_prefix + U_post``; a frozen
       prefix; ``U_post`` scored only over continuation-allocated tasks; a kill outside
       that set moving accounting and not reward; no clamping; the death penalty on the
       reference universe; unchanged terminal credit assignment; and identical
       reward/reference semantics for actor-only and CTDE.

Solver-free and BLADE-free: the MATCH-AOU model is replaced by a recording stub, and
every scenario is a hand-built duck-typed fixture -- so this file runs under the base-env
``pytest`` AND standalone under ``nlp_env``. ``post_solve_filter_and_level``, the tick
loop, the executor, the fuel-damage layer, the graph builder and the central builder are
all the REAL implementations.

Run: python -m pytest tests/test_graph_reference_continuation.py -v
     python tests/test_graph_reference_continuation.py
"""

from __future__ import annotations

import copy
import inspect
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.models import Agent, Capability, Location, Step, StepKind, Task  # noqa: E402
from match_aou.rl.observation.central_graph_builder import (  # noqa: E402
    CentralStateRecorder,
)
from match_aou.rl.observation.graph_builder import (  # noqa: E402
    GraphObservationConfig,
    build_graph_observation,
)
from match_aou.rl.action.graph_action import MetaAction  # noqa: E402
from match_aou.rl.training import graph_episode_setup as ges  # noqa: E402
from match_aou.rl.training import graph_tick_loop  # noqa: E402
from match_aou.rl.training.graph_episode_setup import (  # noqa: E402
    SOLVE_NOT_ATTEMPTED,
    SolveAudit,
    build_continuation_reference,
    build_t0_reference,
    setup_episode,
    solve_and_normalize,
    solve_and_normalize_audited,
)
from match_aou.rl.training.graph_fuel_damage import (  # noqa: E402
    CONDITION_CLEAN,
    CONDITION_DAMAGED,
    FuelDamageController,
    FuelDamageMode,
    FuelDamageParameters,
    plan_fuel_damage,
)
from match_aou.rl.training.graph_reward import (  # noqa: E402
    CONTINUATION_EXCLUSION_DEAD,
    CONTINUATION_EXCLUSION_NOT_AIRBORNE,
    CONTINUATION_EXCLUSION_RTB,
    REFERENCE_KIND_CLEAN_T0,
    REFERENCE_KIND_DAMAGED_EVENT,
    REFERENCE_KIND_DAMAGED_EVENT_UNREALIZED,
    REFERENCE_KINDS,
    REFERENCE_POLICIES,
    REFERENCE_POLICY_EVENT_CONDITIONED_V1,
    REFERENCE_POLICY_STATIC_T0_V1,
    EpisodeReference,
    EpisodeReward,
    ReferenceIntegrityError,
    RewardConfig,
    compute_episode_reward,
    plan_value,
    realized_task_indices,
    realized_utility,
    task_target_ids,
    uses_event_conditioned_reference,
)
from match_aou.solvers.match_aou_MINLP_solver import EPSILON  # noqa: E402
from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor  # noqa: E402

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore


# =============================================================================
# Stubs -- exactly the duck-typed surface the reference seam reads, and no more
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


_TA = _point_at(_BASE, 120.0, 45.0)      # the damaged ego's first assignment
_TB = _point_at(_BASE, 200.0, 45.0)      # its second -- 80 km beyond the first
_TPEER = _point_at(_BASE, 250.0, 200.0)  # the peer's, far off the ego's route


class _Weapon:
    def __init__(self):
        self.id = "w1"
        self.class_name = "AIM-120"
        self.current_quantity = 4
        self.speed = 2000.0

    def get_engagement_range(self):
        return 30.0


class _Aircraft:
    """The BLADE ``Aircraft`` fields the reference seam and the builders touch."""

    def __init__(self, aid, loc, *, speed=1303.0, fuel=12000.0, rate=6700.0,
                 home_base_id="base-blue"):
        self.id = aid
        self.name = "AC %s" % aid
        self.side_id = _BLUE_SIDE
        self.side_color = "blue"
        self.class_name = "F-16 Fighting Falcon"
        self.latitude = loc.latitude
        self.longitude = loc.longitude
        self.altitude = 10000
        self.heading = 90.0
        self.speed = speed
        self.current_fuel = fuel
        self.max_fuel = fuel
        self.fuel_rate = rate
        self.range = 100
        self.weapons = [_Weapon()]
        self.home_base_id = home_base_id
        self.target_id = None
        self.route = []
        self.rtb = False

    def get_weapon_with_highest_engagement_range(self):
        return self.weapons[0] if self.weapons else None


class Airbase:
    """An airbase. The class NAME matters: ``make_attack_task`` reads it for utility."""

    def __init__(self, bid, loc, *, side_id=_BLUE_SIDE, side_color="blue",
                 aircraft=None, name=None):
        self.id = bid
        self.name = name or ("Base %s" % bid)
        self.side_id = side_id
        self.side_color = side_color
        self.latitude = loc.latitude
        self.longitude = loc.longitude
        self.altitude = 0
        self.aircraft = list(aircraft or [])


class _Scenario:
    def __init__(self, aircraft=None, airbases=None):
        self.aircraft = list(aircraft or [])
        self.airbases = list(airbases or [])
        self.ships = []
        self.facilities = []
        self.current_time = 0

    def get_aircraft(self, aircraft_id):
        return next((a for a in self.aircraft if str(a.id) == str(aircraft_id)), None)

    def get_airbase(self, airbase_id):
        return next((b for b in self.airbases if str(b.id) == str(airbase_id)), None)

    def get_ship(self, _ship_id):
        return None

    def get_target(self, target_id):
        for unit in self.airbases + self.facilities + self.ships:
            if str(unit.id) == str(target_id):
                return unit
        return None


class _Env:
    """Nothing moves; a scripted kill removes ONE target after a chosen step.

    Movement is deliberately absent: these tests are about WHEN the reference is taken
    and WHAT it is taken over, and a fixture that also flew the aircraft would make the
    checkpoint an accident of the step size rather than a statement about the seam.
    """

    def __init__(self, scenario, *, kill_after=None, target_id=None):
        self.scenario = scenario
        self.kill_after = kill_after
        self.target_id = target_id
        self.n_steps = 0
        self.closed = False

    def step(self, _action):
        self.n_steps += 1
        if self.kill_after is not None and self.n_steps == self.kill_after:
            self.scenario.airbases = [
                b for b in self.scenario.airbases if str(b.id) != str(self.target_id)
            ]
        return self.scenario, 0.0, False, False, {}

    def close(self):
        self.closed = True


class _Game:
    def __init__(self, scenario):
        self.current_scenario = scenario


class _Belief:
    def __init__(self, tasks, solution=None):
        self.tasks = list(tasks)
        self.solution = {k: [tuple(t) for t in v] for k, v in (solution or {}).items()}


def _task(target_id: str, loc: Location, utility: float = 80.0,
          probability: float = 1.0) -> Task:
    return Task(
        steps=[Step(loc, target_id, [], probability, 1, StepKind.ATTACK)],
        utility=utility,
    )


def _agent(aid: str, loc: Location = _BASE, budget: float = 12000.0) -> Agent:
    return Agent(
        location=loc,
        capabilities=[Capability(name="attack", properties={"Quantity": 2})],
        budget=budget,
        move_cost_function=lambda s, d: 0.0,
        speed=1303.0,
        return_location=_BASE,
        agent_id=aid,
        side_color="blue",
        home_base_id="base-blue",
    )


# =============================================================================
# The MATCH-AOU stub: records every solve, and can answer three different ways
# =============================================================================

_TERMINATION_OK = "optimal"
_TERMINATION_BAD = "maxIterations"


class _SolveRecord:
    """What ONE MATCH-AOU construction was handed -- captured at construction time.

    The agent budgets and locations are snapshotted as plain numbers HERE, so a later
    engine mutation cannot retroactively change what the test believes the solver saw.
    """

    def __init__(self, agents, tasks, observed=None):
        self.agent_ids = [str(a.id) for a in agents]
        self.budgets = {str(a.id): float(a.budget) for a in agents}
        self.locations = {
            str(a.id): (float(a.location.latitude), float(a.location.longitude))
            for a in agents
        }
        self.target_ids = list(task_target_ids(tasks))
        self.n_agents = len(agents)
        self.n_tasks = len(tasks)
        #: Whatever the caller's `observe()` reported AT SOLVE TIME -- used to pin WHEN
        #: a solve happened (e.g. how many env.steps the world had taken by then).
        self.observed = observed


class _SolverStub:
    """Replaces ``graph_episode_setup.MatchAou``. Deterministic, and it counts calls.

    ``decide(agents, tasks)`` returns either the string ``"fail"`` (the solver did NOT
    reach acceptable optimality -> ``MatchAou.solve`` returns ``None``) or the list of
    task indices to SELECT (an empty list is an accepted solve that allocated nothing).
    Selected tasks are handed round-robin to the agents, which is enough structure for
    ``post_solve_filter_and_level`` -- the REAL one -- to do its remap.
    """

    def __init__(self, decide=None, observe=None):
        self.calls = []
        self._decide = decide or (lambda agents, tasks: list(range(len(tasks))))
        self._observe = observe
        self._saved = None

    def __enter__(self):
        outer = self

        class _Model:
            def __init__(self, agents, tasks, precedence_relations=None,
                         risk_factor=0.0):
                self._agents = list(agents)
                self._tasks = list(tasks)
                outer.calls.append(_SolveRecord(
                    self._agents, self._tasks,
                    observed=None if outer._observe is None else outer._observe(),
                ))

            def solve(self, solver_name="bonmin"):
                verdict = outer._decide(self._agents, self._tasks)
                if verdict == "fail":
                    return None, _results(_TERMINATION_BAD), []
                selected = [int(j) for j in verdict]
                unselected = [j for j in range(len(self._tasks)) if j not in selected]
                solution = {}
                for n, j in enumerate(selected):
                    if not self._agents:
                        continue
                    aid = str(self._agents[n % len(self._agents)].id)
                    solution.setdefault(aid, []).append((j, 0))
                return solution, _results(_TERMINATION_OK), unselected

        self._saved = ges.MatchAou
        ges.MatchAou = _Model
        return self

    def __exit__(self, *_exc):
        ges.MatchAou = self._saved
        return False

    @property
    def n_solves(self):
        return len(self.calls)


def _results(termination):
    return SimpleNamespace(solver=SimpleNamespace(termination_condition=termination))


# =============================================================================
# The world: one damaged ego with TWO assignments, one peer with its own
# =============================================================================

class _World:
    """A duck-typed ``EpisodeContext`` plus the tick-loop surface, in ONE fixture.

    The ego starts ON its first target, so which assignment it confirms is decided by
    the LIVENESS of that target and never by the proximity gate. The executor is the REAL
    :class:`GraphPlanExecutor`, so every lifecycle fact the reference reads (``dead`` /
    ``rtb_issued`` / ``done``) is the production one.
    """

    def __init__(self, *, reference_policy=REFERENCE_POLICY_EVENT_CONDITIONED_V1,
                 ego_at=None, kill_after=None, kill_target=None, done=()):
        self.ego, self.peer = "ego-damaged", "peer-quiet"
        self.agent_ids = [self.ego, self.peer]
        here = ego_at or _TA
        self.base = Airbase("base-blue", _BASE)
        self.ego_ac = _Aircraft(self.ego, here)
        self.peer_ac = _Aircraft(self.peer, _BASE)
        self.scenario = _Scenario(
            aircraft=[self.ego_ac, self.peer_ac], airbases=[self.base]
        )
        for tid, loc in (("tA", _TA), ("tB", _TB), ("tPeer", _TPEER)):
            self.scenario.airbases.append(
                Airbase(tid, loc, side_id=_RED_SIDE, side_color="red",
                        name="Red %s" % tid)
            )
        # The RAW t=0 executed-world universe -- every target, allocated or not.
        self.t0_reference_tasks = (
            _task("tA", _TA), _task("tB", _TB), _task("tPeer", _TPEER)
        )
        self.tasks = list(self.t0_reference_tasks)
        self.a_init = {self.ego: [(0, 0, 0), (1, 0, 0)], self.peer: [(2, 0, 0)]}
        self.beliefs = {
            aid: _Belief(self.tasks, self.a_init) for aid in self.agent_ids
        }
        self.agents = [_agent(self.ego, here), _agent(self.peer, _BASE)]
        self.executor = GraphPlanExecutor(
            tasks=self.tasks, solution=self.a_init, agents=self.agents,
            arrival_threshold_km=50.0,
        )
        self.executor.done.update(done)
        self.game = _Game(self.scenario)
        self.observation = self.scenario
        self.env = _Env(self.scenario, kill_after=kill_after, target_id=kill_target)
        self.record = False
        self.reference_policy = reference_policy
        # DEFERRED under the event-conditioned policy, exactly as setup leaves it.
        if reference_policy == REFERENCE_POLICY_EVENT_CONDITIONED_V1:
            self.oracle_solution, self.oracle_tasks = {}, []
        else:
            self.oracle_solution = dict(self.a_init)
            self.oracle_tasks = list(self.tasks)
            self.t0_reference_tasks = ()
        self.known_target_ids = ("tA", "tB", "tPeer")
        self.executed_target_ids = ("tA", "tB", "tPeer")
        self.split_meta = {}
        self.placements = ()

    def damaged_controller(self, **params_kwargs):
        """A DAMAGED controller whose event fires on the ego's first `maybe_apply`.

        The plan is built through the PURE ``plan_fuel_damage`` with the ego named
        explicitly, so the fixture never depends on which ego a seeded draw would pick.
        """
        params = FuelDamageParameters(
            mode=FuelDamageMode.FORCED_DAMAGED, **params_kwargs
        )
        plan = plan_fuel_damage(
            condition=CONDITION_DAMAGED, mode=params.mode, derived_seed=0,
            eligible_ego_ids=(self.ego,), ego_id=self.ego,
            launch_point=_BASE, home_base=_BASE, route_points=[_TA, _TB],
            speed_knots=1303.0, fuel_rate=6700.0, max_fuel=12000.0,
            fuel_at_launch=12000.0, params=params,
        )
        return FuelDamageController(plan)

    def clean_controller(self):
        params = FuelDamageParameters(mode=FuelDamageMode.FORCED_CLEAN)
        plan = plan_fuel_damage(
            condition=CONDITION_CLEAN, mode=params.mode, derived_seed=0,
            eligible_ego_ids=(self.ego,), ego_id=None, launch_point=None,
            home_base=None, route_points=None, speed_knots=None, fuel_rate=None,
            max_fuel=None, fuel_at_launch=None, params=params,
        )
        return FuelDamageController(plan)


def _result_stub(n_transitions=1):
    """The EpisodeResult surface `compute_episode_reward` reads."""
    return SimpleNamespace(
        trajectory=[SimpleNamespace(reward=None) for _ in range(n_transitions)],
        reference=None,
    )


def _run(world, controller=None, *, max_ticks=3, central=None, spy=None):
    """Drive the REAL `run_episode`, with `_wake_decision` replaced by a recorder.

    The encoder / head are deliberately absent: these tests are about the reference seam
    and the ORDER of the tick's steps, and a real policy would add sampling noise without
    adding evidence. Everything else -- the trigger layer, the executor, the fuel-damage
    controller, the central recorder and the reference builders -- is production code.
    """
    events = [] if spy is None else spy
    real_decide = graph_tick_loop.decide_triggers
    real_boundary = graph_tick_loop._post_fd_boundary
    real_continuation = graph_tick_loop.build_continuation_reference
    real_next = world.executor.next_actions
    real_capture = None if central is None else central.capture

    def spy_decide(belief_tasks, belief_solution, sensed, eta=None, *, ego_id, clock,
                   fuel_damage=False, post_fd_completion=False):
        events.append(("trigger", int(clock), str(ego_id)))
        return real_decide(belief_tasks, belief_solution, sensed, ego_id=ego_id,
                           clock=clock, fuel_damage=fuel_damage,
                           post_fd_completion=post_fd_completion)

    def spy_boundary(fd, ctx, obs, tick):
        events.append(("boundary", int(tick), ""))
        return real_boundary(fd, ctx, obs, tick)

    def spy_continuation(ctx, *, scenario, tick, damaged_ego_id, **kw):
        events.append(("checkpoint", int(tick), str(damaged_ego_id)))
        return real_continuation(ctx, scenario=scenario, tick=tick,
                                 damaged_ego_id=damaged_ego_id, **kw)

    def spy_next(observation):
        events.append(("step", -1, ""))
        return real_next(observation)

    def spy_wake(_policy, ego_id, obs, belief, _executor, cfg, tick, **_kw):
        events.append(("decision", int(tick), str(ego_id)))
        gobs = build_graph_observation(
            scenario=obs, agent_id=str(ego_id),
            current_plan=belief.solution.get(str(ego_id)), current_time=tick,
            tasks=belief.tasks, solution=belief.solution,
            precedence_relations=[], config=cfg,
        )
        return graph_tick_loop.Transition(
            gobs=gobs, ego_id=str(ego_id), tick=int(tick),
            meta_action=int(MetaAction.PLAN_COMPLIANCE), node_v=0,
            log_prob=0.0, entropy=0.0,
        )

    def spy_capture(**kwargs):
        events.append(("capture", int(kwargs.get("current_time", -1)), ""))
        return real_capture(**kwargs)

    saved = (graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision,
             graph_tick_loop._post_fd_boundary,
             graph_tick_loop.build_continuation_reference)
    graph_tick_loop.decide_triggers = spy_decide
    graph_tick_loop._wake_decision = spy_wake
    graph_tick_loop._post_fd_boundary = spy_boundary
    graph_tick_loop.build_continuation_reference = spy_continuation
    world.executor.next_actions = spy_next
    if central is not None:
        central.capture = spy_capture
    try:
        result = graph_tick_loop.run_episode(
            None, world, GraphObservationConfig(detection_range_km=50.0),
            max_ticks=max_ticks, fuel_damage=controller,
            **({} if central is None else {"central": central}),
        )
    finally:
        (graph_tick_loop.decide_triggers, graph_tick_loop._wake_decision,
         graph_tick_loop._post_fd_boundary,
         graph_tick_loop.build_continuation_reference) = saved
        world.executor.next_actions = real_next
        if central is not None:
            central.capture = real_capture
    return result, events


def _expect_raises(exc_type, label, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as other:  # pragma: no cover - a wrong exception type is a failure
        raise AssertionError(
            "%s: expected %s, got %s: %s"
            % (label, exc_type.__name__, type(other).__name__, other)
        )
    raise AssertionError("%s: expected %s, nothing raised" % (label, exc_type.__name__))


def _fingerprint(gobs):
    """A comparable, fully-expanded view of a GraphObservation."""
    return (
        gobs.task_features.tolist(), gobs.agent_features.tolist(),
        int(gobs.ego_index), gobs.edge_index.tolist(), gobs.edge_type.tolist(),
        list(gobs.task_target_ids), list(gobs.agent_ids), str(gobs.agent_id),
        int(gobs.current_time), float(gobs.time_norm),
    )


def _central_fingerprint(cobs):
    return (
        cobs.task_features.tolist(), cobs.agent_features.tolist(),
        int(cobs.ego_index), cobs.edge_index.tolist(), cobs.edge_type.tolist(),
        cobs.edge_attr.tolist(), list(cobs.task_target_ids), list(cobs.agent_ids),
        int(cobs.current_time), float(cobs.time_norm),
    )


def _executor_fingerprint(ex):
    """Comparable ACROSS separately-constructed worlds, so task slices are named by
    target id rather than by object identity."""
    return (
        {k: [tuple(t) for t in v] for k, v in ex.plans.items()},
        {k: list(task_target_ids(v)) for k, v in ex.tasks.items()},
        sorted(ex.done), sorted(ex.dead), dict(ex.rtb_issued),
    )


def _belief_fingerprint(beliefs):
    return {
        aid: ([[str(s.target_id) for s in t.steps] for t in b.tasks],
              {k: [tuple(t) for t in v] for k, v in b.solution.items()})
        for aid, b in beliefs.items()
    }


def _scenario_fingerprint(scen):
    return (
        [(str(a.id), a.latitude, a.longitude, a.current_fuel, a.rtb, list(a.route))
         for a in scen.aircraft],
        [(str(b.id), b.latitude, b.longitude, [str(a.id) for a in b.aircraft])
         for b in scen.airbases],
        scen.current_time,
    )


# =============================================================================
# PO1 -- HISTORICAL PRESERVATION AND SOLVE BUDGET
# =============================================================================

def test_po1_policy_set_is_closed_and_the_default_is_the_historical_one() -> None:
    """ONE closed set, ONE default, and an unknown id RAISES before any BLADE object."""
    assert REFERENCE_POLICIES == (
        REFERENCE_POLICY_STATIC_T0_V1, REFERENCE_POLICY_EVENT_CONDITIONED_V1
    )
    assert inspect.signature(setup_episode).parameters["reference_policy"].default == \
        REFERENCE_POLICY_STATIC_T0_V1

    # The predicate resolves an ABSENT declaration to the preserved path, never the
    # opt-in one -- so a duck-typed context can only ever fall back historically.
    assert uses_event_conditioned_reference(SimpleNamespace()) is False
    assert uses_event_conditioned_reference(
        SimpleNamespace(reference_policy=REFERENCE_POLICY_STATIC_T0_V1)) is False
    assert uses_event_conditioned_reference(
        SimpleNamespace(reference_policy=REFERENCE_POLICY_EVENT_CONDITIONED_V1)) is True

    # An unknown id is REFUSED rather than coerced -- and refused before anything is
    # built, so a rejected episode manufactures no solve at all.
    with _SolverStub() as stub:
        _expect_raises(ValueError, "unknown reference policy",
                       ges._resolve_reference_policy, "event_conditioned_v2")
        _expect_raises(ValueError, "unknown reference policy",
                       ges._resolve_reference_policy, "STATIC_T0_V1")
        assert stub.n_solves == 0


def test_po1_setup_solves_the_t0_reference_only_under_the_historical_policy() -> None:
    """`_t0_reference_or_deferred`: the SITE that keeps the budget at two solves."""
    agents = [_agent("ego0"), _agent("ego1")]
    tasks = [_task("tA", _TA), _task("tB", _TB)]

    with _SolverStub() as stub:
        solution, allocated = ges._t0_reference_or_deferred(
            agents, tasks, reference_policy=REFERENCE_POLICY_STATIC_T0_V1
        )
        assert stub.n_solves == 1, "the historical policy still solves its reference here"
        assert allocated and solution, (solution, allocated)

    with _SolverStub() as stub:
        solution, allocated = ges._t0_reference_or_deferred(
            agents, tasks, reference_policy=REFERENCE_POLICY_EVENT_CONDITIONED_V1
        )
        assert stub.n_solves == 0, "the opt-in policy DEFERS -- it must not solve here"
        assert (solution, allocated) == ({}, []), (solution, allocated)


def test_po1_default_policy_produces_no_reference_and_no_solve_in_the_loop() -> None:
    """The historical path through the REAL tick loop: nothing is computed, nothing solved."""
    world = _World(reference_policy=REFERENCE_POLICY_STATIC_T0_V1)
    controller = world.damaged_controller()
    with _SolverStub() as stub:
        result, events = _run(world, controller, max_ticks=2)
    assert result.reference is None, "the historical policy produces no reference"
    assert stub.n_solves == 0, "the historical loop performs no MATCH-AOU solve"
    assert controller.fired, "the fixture's event must still fire"
    assert not any(kind == "checkpoint" for kind, _t, _e in events)


def test_po1_default_policy_reward_is_the_historical_arithmetic() -> None:
    """The static breakdown, recomputed by hand, plus every new field left ABSENT."""
    world = _World(reference_policy=REFERENCE_POLICY_STATIC_T0_V1)
    world.executor.done.update({(world.ego, "tA")})
    world.executor.dead.add(world.peer)
    cfg = RewardConfig(aircraft_penalty_coeff=2.25)

    result = _result_stub(3)
    breakdown = compute_episode_reward(world, result, cfg)

    u_oracle = plan_value(world.oracle_solution, world.oracle_tasks)
    u_achieved = realized_utility(world.oracle_tasks, world.executor.done)
    denom = abs(u_oracle) + cfg.regret_epsilon
    expected = (u_achieved - u_oracle) / denom - (2.25 * 80.0 * 1) / denom

    assert breakdown.u_oracle == u_oracle
    assert breakdown.u_achieved == u_achieved
    assert breakdown.reward == expected
    assert breakdown.reference_policy == REFERENCE_POLICY_STATIC_T0_V1
    assert breakdown.u_ref == u_oracle, "u_ref names the denominator under BOTH policies"
    # MISSING IS NULL, NEVER ZERO: a 0.0 on a normalized regret scale is the OPTIMUM,
    # and a 0 count reads as a measurement of nothing rather than an absent measurement.
    for absent in ("reference_kind", "checkpoint_tick", "u_prefix", "u_cont_ref",
                   "u_post", "unique_completed_targets", "scored_completed_targets",
                   "unscored_completed_targets"):
        assert getattr(breakdown, absent) is None, absent
    assert breakdown.unscored_completed_target_ids == ()
    # Terminal-on-last placement, unchanged.
    assert [t.reward for t in result.trajectory] == [0.0, 0.0, expected]


def test_po1_clean_event_conditioned_costs_exactly_one_loop_solve_at_t0() -> None:
    """CLEAN: solve #2 is the FULL t=0 reference, taken BEFORE the first tick."""
    world = _World()
    controller = world.clean_controller()
    with _SolverStub(observe=lambda: world.env.n_steps) as stub:
        result, events = _run(world, controller, max_ticks=3)

    assert stub.n_solves == 1, "exactly ONE reference solve for the whole episode"
    call = stub.calls[0]
    assert call.target_ids == ["tA", "tB", "tPeer"], "the FULL t=0 world"
    assert sorted(call.agent_ids) == sorted(world.agent_ids)
    ref = result.reference
    assert ref is not None and ref.kind == REFERENCE_KIND_CLEAN_T0
    assert ref.checkpoint_tick is None, "a t=0 reference has no checkpoint tick"
    assert ref.u_prefix == 0.0
    assert ref.u_ref == ref.u_cont_ref
    assert ref.prefix_target_ids == () and ref.excluded_agents == ()
    # MEASURED, not inferred: no BLADE state had advanced when the solve ran.
    assert call.observed == 0, "the t=0 reference was solved before the first env.step"
    assert not any(k == "checkpoint" for k, _t, _e in events)
    assert world.env.n_steps == 3, "the loop still ran normally afterwards"


def test_po1_damaged_event_conditioned_costs_exactly_one_loop_solve_at_the_event() -> None:
    """DAMAGED: solve #2 is the CONTINUATION reference, and there is no t=0 call."""
    world = _World()
    controller = world.damaged_controller()
    with _SolverStub() as stub:
        result, events = _run(world, controller, max_ticks=3)

    assert stub.n_solves == 1, "exactly ONE reference solve -- the continuation one"
    assert controller.fired
    ref = result.reference
    assert ref is not None and ref.kind == REFERENCE_KIND_DAMAGED_EVENT
    assert ref.checkpoint_tick == controller.outcome.event_tick
    # NO full t=0 reference call was made: the one solve saw only the OPEN tasks.
    assert stub.calls[0].target_ids == ["tA", "tB", "tPeer"], (
        "nothing was confirmed yet, so the open universe IS the whole world -- but it "
        "was solved at the checkpoint, not at t=0"
    )
    assert [k for k, _t, _e in events if k == "checkpoint"] == ["checkpoint"]


def test_po1_a_refused_event_manufactures_no_reference_solve() -> None:
    """A controller that never fires leaves the checkpoint unreached.

    The damaged-but-unfired episode still gets its ONE solve, at the episode-exit seam,
    as the t=0 reference it physically deserves -- never a second one.
    """
    # The ego sits AT THE BASE, so it never crosses the leg-progress threshold.
    world = _World(ego_at=_BASE)
    controller = world.damaged_controller()
    with _SolverStub() as stub:
        result, events = _run(world, controller, max_ticks=3)

    assert not controller.fired, "the fixture's event must NOT fire"
    assert not any(k == "checkpoint" for k, _t, _e in events)
    assert stub.n_solves == 1, "still exactly one solve, never two"
    ref = result.reference
    assert ref is not None
    assert ref.kind == REFERENCE_KIND_DAMAGED_EVENT_UNREALIZED, (
        "recorded under its own kind -- 'the event did not fire' and 'no event was "
        "scheduled' are different facts"
    )
    assert ref.checkpoint_tick is None and ref.u_prefix == 0.0
    assert stub.calls[0].target_ids == ["tA", "tB", "tPeer"]


def test_po1_no_harness_selects_the_new_policy() -> None:
    """Task 4 owns harness exposure. Today NOTHING reaches the opt-in path by default."""
    for name in ("graph_train.py", "graph_rollout.py"):
        src = (SRC / "match_aou" / "rl" / "training" / name).read_text(encoding="utf-8")
        assert "reference_policy" not in src, (
            "%s must not select or expose the reference policy in task 3" % name
        )
        assert "event_conditioned_continuation_v1" not in src, name
        assert "build_continuation_reference" not in src, name
    # And a caller that passes nothing gets the historical field value.
    assert ges.EpisodeContext.reference_policy == REFERENCE_POLICY_STATIC_T0_V1
    assert ges.EpisodeContext.t0_reference_tasks == ()


# =============================================================================
# PO2 -- EVENT-STATE CORRECTNESS AND ISOLATION
# =============================================================================

def test_po2_continuation_agents_carry_live_post_mutation_fuel_and_position() -> None:
    """The solver sees the fuel the ego REALLY holds, where it REALLY is."""
    world = _World()
    controller = world.damaged_controller()
    launch_fuel = world.ego_ac.current_fuel
    with _SolverStub() as stub:
        result, _events = _run(world, controller, max_ticks=2)

    assert controller.fired
    call = stub.calls[0]
    fuel_after = controller.outcome.fuel_after
    assert fuel_after is not None and fuel_after < launch_fuel, "a real loss"
    assert call.budgets[world.ego] == world.ego_ac.current_fuel == fuel_after, (
        "the continuation agent's budget IS the live post-mutation current_fuel"
    )
    assert call.budgets[world.peer] == world.peer_ac.current_fuel, "peers are live too"
    assert call.locations[world.ego] == (
        world.ego_ac.latitude, world.ego_ac.longitude
    ), "and its location is the live event position, not the t=0 launch point"
    assert call.locations[world.ego] != (_BASE.latitude, _BASE.longitude)
    assert result.reference.continuation_agent_ids == tuple(world.agent_ids)


def test_po2_checkpoint_precedes_boundary_trigger_capture_and_decision() -> None:
    """ORDER: mutation -> checkpoint -> post-FD boundary -> triggers -> capture -> act -> step."""
    world = _World()
    controller = world.damaged_controller(
        post_fd_wake_policy="completion_boundary_v1"
    )
    central = CentralStateRecorder()
    with _SolverStub():
        _result, events = _run(world, controller, max_ticks=1, central=central)

    kinds = [k for k, _t, _e in events]
    assert kinds.count("checkpoint") == 1
    at = kinds.index("checkpoint")
    assert at == 0, "the checkpoint is the FIRST thing after the mutation"
    # Every one of these really occurs on this tick -- the fuel-damage wake is what makes
    # the decision and the capture unconditional, so the ordering claim is not vacuous.
    for later in ("boundary", "trigger", "capture", "decision", "step"):
        assert later in kinds, "%s never happened -- the ordering claim would be vacuous"
        assert kinds.index(later) > at, "%s must follow the checkpoint" % later
    assert kinds.index("boundary") < kinds.index("trigger") < kinds.index("capture")
    assert kinds.index("capture") < kinds.index("decision") < kinds.index("step")
    assert central.samples, "the CTDE recorder really captured the decision"


def test_po2_simulation_time_does_not_advance_across_the_checkpoint() -> None:
    """Wall-clock time may pass in the solver; the SIMULATION clock may not."""
    world = _World()
    seen = {}

    real_builder = ges.build_continuation_reference

    def watching(ctx, *, scenario, tick, damaged_ego_id, **kw):
        seen["steps_before"] = world.env.n_steps
        seen["clock_before"] = scenario.current_time
        out = real_builder(ctx, scenario=scenario, tick=tick,
                           damaged_ego_id=damaged_ego_id, **kw)
        seen["steps_after"] = world.env.n_steps
        seen["clock_after"] = scenario.current_time
        return out

    controller = world.damaged_controller()
    saved = ges.build_continuation_reference
    graph_tick_loop.build_continuation_reference = watching
    ges.build_continuation_reference = watching
    try:
        with _SolverStub():
            _run(world, controller, max_ticks=1)
    finally:
        graph_tick_loop.build_continuation_reference = saved
        ges.build_continuation_reference = saved

    assert seen, "the checkpoint must have run"
    assert seen["steps_before"] == seen["steps_after"] == 0, (
        "the checkpoint issues no env.step, so it cannot advance the world"
    )
    assert seen["clock_before"] == seen["clock_after"]


def test_po2_checkpoint_mutates_nothing() -> None:
    """Measurement only: BLADE, beliefs, executor state and the plans are byte-unchanged."""
    world = _World()
    controller = world.damaged_controller()
    # Fire the event first, so the snapshot below isolates the CHECKPOINT and not the
    # mutation the fuel-damage layer legitimately performs.
    assert controller.maybe_apply(world.scenario, 0) == world.ego

    before = (
        _scenario_fingerprint(world.scenario),
        _belief_fingerprint(world.beliefs),
        _executor_fingerprint(world.executor),
        copy.deepcopy(world.a_init),
        [id(t) for t in world.t0_reference_tasks],
    )
    with _SolverStub():
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=0, damaged_ego_id=world.ego
        )
    after = (
        _scenario_fingerprint(world.scenario),
        _belief_fingerprint(world.beliefs),
        _executor_fingerprint(world.executor),
        copy.deepcopy(world.a_init),
        [id(t) for t in world.t0_reference_tasks],
    )
    assert before == after, "the checkpoint mutated episode state"
    assert isinstance(ref, EpisodeReference)


def test_po2_dead_and_rtb_committed_egos_cannot_be_reallocated() -> None:
    """A dead / returning / grounded ego is EXCLUDED, with a stable recorded reason."""
    # (a) DEAD.
    world = _World()
    world.executor.dead.add(world.ego)
    with _SolverStub() as stub:
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=5, damaged_ego_id=world.ego)
    assert world.ego not in ref.continuation_agent_ids
    assert (world.ego, CONTINUATION_EXCLUSION_DEAD) in ref.excluded_agents
    assert world.ego not in stub.calls[0].agent_ids, "the solver never saw it"

    # (b) RTB-COMMITTED -- the latch is set, so Phase 1 no longer processes it and the
    #     execution layer could not honour a fresh allocation.
    world = _World()
    world.executor.rtb_issued[world.ego] = True
    with _SolverStub() as stub:
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=5, damaged_ego_id=world.ego)
    assert (world.ego, CONTINUATION_EXCLUSION_RTB) in ref.excluded_agents
    assert world.ego not in stub.calls[0].agent_ids

    # (c) NOT AIRBORNE -- the engine has landed it into an airbase inventory.
    world = _World()
    world.scenario.aircraft = [world.peer_ac]
    world.base.aircraft = [world.ego_ac]
    with _SolverStub() as stub:
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=5, damaged_ego_id=world.ego)
    assert (world.ego, CONTINUATION_EXCLUSION_NOT_AIRBORNE) in ref.excluded_agents
    assert world.ego not in stub.calls[0].agent_ids

    # (d) EVERY ego excluded -> a legitimate ZERO reference, and NO solver call at all.
    world = _World()
    world.executor.dead.update(world.agent_ids)
    with _SolverStub() as stub:
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=5, damaged_ego_id=world.ego)
    assert stub.n_solves == 0, "nothing to allocate -> no bonmin call"
    assert ref.continuation_agent_ids == () and ref.u_cont_ref == 0.0
    assert ref.solver_invoked is False
    assert ref.solver_termination == SOLVE_NOT_ATTEMPTED


def test_po2_solver_failure_is_distinguishable_from_a_valid_zero_allocation() -> None:
    """The audited seam: `None` (unanswered) vs `{}` (answered zero) -- and the SAME triple."""
    agents = [_agent("ego0")]
    tasks = [_task("tA", _TA)]

    # (a) The PUBLIC triple is byte-identical in both cases -- historical behaviour.
    with _SolverStub(decide=lambda a, t: "fail"):
        failed = solve_and_normalize(agents, tasks)
        failed_audited = solve_and_normalize_audited(agents, tasks)
    with _SolverStub(decide=lambda a, t: []):
        empty = solve_and_normalize(agents, tasks)
        empty_audited = solve_and_normalize_audited(agents, tasks)
    assert failed == empty == ({}, [], [0]), (failed, empty)
    assert failed_audited[:3] == empty_audited[:3] == failed

    # (b) The AUDIT is what tells them apart.
    assert failed_audited[3].invoked and failed_audited[3].accepted is False
    assert failed_audited[3].termination_condition == _TERMINATION_BAD
    assert empty_audited[3].invoked and empty_audited[3].accepted is True
    assert empty_audited[3].termination_condition == _TERMINATION_OK
    # A skipped solve is neither: nothing was asked.
    with _SolverStub() as stub:
        skipped = solve_and_normalize_audited(agents, [])
        assert stub.n_solves == 0
    assert skipped[3] == SolveAudit(
        invoked=False, accepted=True,
        termination_condition=SOLVE_NOT_ATTEMPTED, allocated_task_count=0, seconds=0.0,
    )

    # (c) The reference REFUSES the failure and ACCEPTS the zero.
    world = _World()
    with _SolverStub(decide=lambda a, t: "fail"):
        _expect_raises(
            ReferenceIntegrityError, "a failed reference solve must not become a zero",
            build_continuation_reference,
            world, scenario=world.scenario, tick=7, damaged_ego_id=world.ego,
        )
        _expect_raises(ReferenceIntegrityError, "the t=0 reference refuses it too",
                       build_t0_reference, world, kind=REFERENCE_KIND_CLEAN_T0)
    with _SolverStub(decide=lambda a, t: []) as stub:
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=7, damaged_ego_id=world.ego)
    assert stub.n_solves == 1 and ref.solver_invoked and ref.solver_accepted
    assert ref.u_cont_ref == 0.0 and ref.tasks == () and ref.u_ref == 0.0
    assert ref.candidate_task_count == 3, "three tasks were OFFERED and none selected"


def test_po2_the_reference_never_reaches_the_actor_or_the_critic() -> None:
    """Two identical episodes, one with the reference computed -- same inputs everywhere.

    The strongest available statement of the isolation: if any privileged reference value
    reached an actor observation or a central observation, the two runs' recorded inputs
    would differ. They are compared element by element.
    """
    fingerprints = {}
    for policy in (REFERENCE_POLICY_STATIC_T0_V1, REFERENCE_POLICY_EVENT_CONDITIONED_V1):
        world = _World(reference_policy=policy)
        controller = world.damaged_controller()
        central = CentralStateRecorder()
        with _SolverStub():
            result, _events = _run(world, controller, max_ticks=3, central=central)
        fingerprints[policy] = (
            [_fingerprint(tr.gobs) for tr in result.trajectory],
            [_central_fingerprint(c) for c in central.samples],
            _scenario_fingerprint(world.scenario),
            _belief_fingerprint(world.beliefs),
            _executor_fingerprint(world.executor),
        )
        if policy == REFERENCE_POLICY_EVENT_CONDITIONED_V1:
            assert result.reference is not None, "the opt-in run really built one"
        else:
            assert result.reference is None

    static, event = (fingerprints[REFERENCE_POLICY_STATIC_T0_V1],
                     fingerprints[REFERENCE_POLICY_EVENT_CONDITIONED_V1])
    assert static[0] == event[0], "actor observations differ"
    assert static[1] == event[1], "central observations differ"
    assert static[2] == event[2], "BLADE state differs"
    assert static[3] == event[3], "beliefs differ"
    assert static[4] == event[4], "executor state differs"
    # And the samples are still 1:1 with the decisions, boundary or not.
    assert len(event[1]) == len(event[0])


def test_po2_the_reference_universe_is_the_world_and_never_a_private_belief() -> None:
    """The continuation universe comes from the t=0 EXECUTED world, not from any ego."""
    world = _World()
    # Poison every belief: if the reference read one, the solver would see "ghost".
    for belief in world.beliefs.values():
        belief.tasks = [_task("ghost", _TPEER)]
        belief.solution = {}
    with _SolverStub() as stub:
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=2, damaged_ego_id=world.ego)
    assert stub.calls[0].target_ids == ["tA", "tB", "tPeer"]
    assert "ghost" not in ref.reference_target_ids

    # A retained universe is REQUIRED under this policy -- an empty one is refused
    # rather than silently scored as zero.
    world.t0_reference_tasks = ()
    with _SolverStub():
        _expect_raises(ReferenceIntegrityError, "no retained universe",
                       build_continuation_reference,
                       world, scenario=world.scenario, tick=2,
                       damaged_ego_id=world.ego)
        _expect_raises(ReferenceIntegrityError, "no retained universe (t0)",
                       build_t0_reference, world, kind=REFERENCE_KIND_CLEAN_T0)


# =============================================================================
# PO3 -- REWARD / ACCOUNTING CORRECTNESS
# =============================================================================

def _hand_reference(*, u_prefix, allocated, prefix_ids, solution=None,
                    kind=REFERENCE_KIND_DAMAGED_EVENT, tick=11):
    """A hand-built reference, so the reward arithmetic is checked against numbers."""
    sol = solution if solution is not None else {
        "ego-damaged": [(j, 0, 0) for j in range(len(allocated))]
    }
    u_cont = plan_value(sol, allocated)
    return EpisodeReference(
        policy=REFERENCE_POLICY_EVENT_CONDITIONED_V1,
        kind=kind,
        checkpoint_tick=tick,
        u_prefix=float(u_prefix),
        u_cont_ref=float(u_cont),
        u_ref=float(u_prefix) + float(u_cont),
        u_aircraft=80.0,
        solution=sol,
        tasks=tuple(allocated),
        reference_target_ids=task_target_ids(allocated),
        prefix_target_ids=tuple(prefix_ids),
        candidate_task_count=len(allocated),
        continuation_agent_ids=("ego-damaged",),
        excluded_agents=(),
        solver_invoked=True,
        solver_accepted=True,
        solver_termination=_TERMINATION_OK,
        solver_seconds=0.25,
    )


def test_po3_reference_arithmetic_is_verified_not_asserted() -> None:
    """`U_ref = U_prefix + U_cont_ref` is CHECKED on construction; a bad kind is refused."""
    allocated = [_task("tB", _TB)]
    ref = _hand_reference(u_prefix=80.0, allocated=allocated, prefix_ids=("tA",))
    assert ref.u_ref == 80.0 + ref.u_cont_ref
    assert ref.u_cont_ref == 80.0 * (1.0 - EPSILON), ref.u_cont_ref
    assert ref.is_event_checkpoint

    bad = dict(
        policy=REFERENCE_POLICY_EVENT_CONDITIONED_V1, kind=REFERENCE_KIND_DAMAGED_EVENT,
        checkpoint_tick=1, u_prefix=80.0, u_cont_ref=10.0, u_ref=999.0, u_aircraft=80.0,
        solution={}, tasks=(), reference_target_ids=(), prefix_target_ids=(),
        candidate_task_count=0, continuation_agent_ids=(), excluded_agents=(),
        solver_invoked=True, solver_accepted=True, solver_termination=_TERMINATION_OK,
        solver_seconds=0.0,
    )
    _expect_raises(ReferenceIntegrityError, "u_ref must reconcile",
                   EpisodeReference, **bad)
    _expect_raises(ReferenceIntegrityError, "unknown kind",
                   EpisodeReference, **{**bad, "u_ref": 90.0, "kind": "made_up"})

    record = ref.to_record()
    assert set(REFERENCE_KINDS) >= {record["kind"]}
    assert record["u_ref"] == ref.u_ref and record["allocated_task_count"] == 1
    assert record["prefix_target_ids"] == ["tA"]
    assert "tasks" not in record and "solution" not in record


def test_po3_u_ref_and_u_achieved_decompose_by_hand() -> None:
    """The two identities, on numbers, with the prefix genuinely non-zero."""
    world = _World()
    # tA was confirmed BEFORE the checkpoint; tB and tPeer are still open.
    world.executor.done.update({(world.ego, "tA"), (world.peer, "tB")})
    ref = _hand_reference(
        u_prefix=80.0, allocated=[_task("tB", _TB), _task("tPeer", _TPEER)],
        prefix_ids=("tA",),
        solution={"ego-damaged": [(0, 0, 0)], "peer-quiet": [(1, 0, 0)]},
    )
    result = _result_stub(2)
    result.reference = ref
    breakdown = compute_episode_reward(world, result, RewardConfig())

    u_post = realized_utility(ref.tasks, world.executor.done)
    assert u_post == 80.0, "only tB is confirmed among the allocated tasks"
    assert breakdown.u_prefix == 80.0
    assert breakdown.u_cont_ref == ref.u_cont_ref
    assert breakdown.u_ref == 80.0 + ref.u_cont_ref
    assert breakdown.u_post == u_post
    assert breakdown.u_achieved == 80.0 + u_post == 160.0
    denom = abs(breakdown.u_ref) + 1e-5
    assert breakdown.ratio == (breakdown.u_achieved - breakdown.u_ref) / denom
    assert breakdown.reward == breakdown.ratio  # c == 0.0 here
    assert breakdown.u_oracle is None, "no static oracle exists under this policy"
    assert breakdown.reference_policy == REFERENCE_POLICY_EVENT_CONDITIONED_V1
    assert breakdown.reference_kind == REFERENCE_KIND_DAMAGED_EVENT
    assert breakdown.checkpoint_tick == 11


def test_po3_u_prefix_is_frozen_at_the_checkpoint() -> None:
    """It is taken off the reference, never recomputed from the larger final `done`."""
    world = _World(kill_after=1, kill_target="tA")
    with _SolverStub():
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=0, damaged_ego_id=world.ego)
    assert ref.u_prefix == 0.0 and ref.prefix_target_ids == ()
    assert set(ref.reference_target_ids) == {"tA", "tB", "tPeer"}

    # Now confirm EVERYTHING, long after the checkpoint.
    world.executor.done.update(
        {(world.ego, "tA"), (world.ego, "tB"), (world.peer, "tPeer")}
    )
    result = _result_stub(1)
    result.reference = ref
    breakdown = compute_episode_reward(world, result, RewardConfig())
    assert breakdown.u_prefix == 0.0, (
        "the prefix is frozen; recomputing it here would credit post-checkpoint kills "
        "to it AND leave them scorable in U_post"
    )
    assert breakdown.u_post == 240.0
    assert breakdown.u_achieved == 240.0

    # And the mirror case: a NON-zero prefix stays exactly what the checkpoint froze.
    world2 = _World(done={("ego-damaged", "tA")})
    with _SolverStub():
        ref2 = build_continuation_reference(
            world2, scenario=world2.scenario, tick=4, damaged_ego_id=world2.ego)
    assert ref2.u_prefix == 80.0 and ref2.prefix_target_ids == ("tA",)
    assert "tA" not in ref2.reference_target_ids, (
        "a prefix task is EXCLUDED from the continuation universe, so it can never be "
        "counted twice"
    )
    world2.executor.done.add((world2.ego, "tB"))
    result2 = _result_stub(1)
    result2.reference = ref2
    b2 = compute_episode_reward(world2, result2, RewardConfig())
    assert b2.u_prefix == 80.0 and b2.u_post == 80.0 and b2.u_achieved == 160.0


def test_po3_a_kill_outside_the_reference_moves_accounting_but_not_reward() -> None:
    """Kills outside the reward-bearing set are ACCOUNTING-ONLY, exactly as specified."""
    world = _World()
    ref = _hand_reference(
        u_prefix=80.0, allocated=[_task("tB", _TB)], prefix_ids=("tA",)
    )
    world.executor.done.update({(world.ego, "tA"), (world.ego, "tB")})
    result = _result_stub(1)
    result.reference = ref
    baseline = compute_episode_reward(world, result, RewardConfig())
    assert baseline.scored_completed_targets == 2
    assert baseline.unscored_completed_targets == 0
    assert baseline.unique_completed_targets == 2

    # tPeer is neither in the prefix nor allocated by the continuation reference.
    world.executor.done.add((world.peer, "tPeer"))
    result2 = _result_stub(1)
    result2.reference = ref
    after = compute_episode_reward(world, result2, RewardConfig())
    assert after.reward == baseline.reward, "an unscored kill must not move the reward"
    assert after.u_post == baseline.u_post == 80.0
    assert after.u_achieved == baseline.u_achieved
    assert after.unique_completed_targets == 3
    assert after.scored_completed_targets == 2
    assert after.unscored_completed_targets == 1
    assert after.unscored_completed_target_ids == ("tPeer",)


def test_po3_the_reward_is_not_clamped() -> None:
    """With a real airframe loss the event-conditioned reward legitimately drops below -1."""
    world = _World()
    world.executor.dead.update(world.agent_ids)  # two lost airframes
    ref = _hand_reference(u_prefix=0.0, allocated=[_task("tB", _TB)], prefix_ids=())
    result = _result_stub(1)
    result.reference = ref
    breakdown = compute_episode_reward(
        world, result, RewardConfig(aircraft_penalty_coeff=2.25)
    )
    denom = abs(ref.u_ref) + 1e-5
    assert breakdown.n_lost == 2
    assert breakdown.penalty == (2.25 * 80.0 * 2) / denom
    assert breakdown.reward == breakdown.ratio - breakdown.penalty
    assert breakdown.reward < -1.0, breakdown.reward


def test_po3_aircraft_penalty_uses_the_reference_universe() -> None:
    """`U_aircraft` is the reference universe's best target -- prefix INCLUDED."""
    # The prefix holds the 100-utility target; the continuation allocation holds an 80.
    prefix_task = _task("tA", _TA, utility=100.0)
    open_task = _task("tB", _TB, utility=80.0)
    world = _World()
    world.t0_reference_tasks = (prefix_task, open_task)
    world.executor.done.add((world.ego, "tA"))
    with _SolverStub():
        ref = build_continuation_reference(
            world, scenario=world.scenario, tick=3, damaged_ego_id=world.ego)
    assert ref.u_prefix == 100.0
    assert ref.u_aircraft == 100.0, (
        "the prefix target is part of the reward-bearing universe, so it sets the scale"
    )

    # With NO prefix the scale comes from the allocated tasks alone.
    world2 = _World()
    world2.t0_reference_tasks = (open_task,)
    with _SolverStub():
        ref2 = build_continuation_reference(
            world2, scenario=world2.scenario, tick=3, damaged_ego_id=world2.ego)
    assert ref2.u_prefix == 0.0 and ref2.u_aircraft == 80.0

    # With NOTHING open, the whole universe is the PREFIX -- which is still
    # reward-bearing, so the scale comes from it. It is a MAX, not a sum.
    world3 = _World()
    world3.executor.done.update(
        {(world3.ego, "tA"), (world3.ego, "tB"), (world3.ego, "tPeer")}
    )
    with _SolverStub() as stub:
        ref3 = build_continuation_reference(
            world3, scenario=world3.scenario, tick=3, damaged_ego_id=world3.ego)
    assert stub.n_solves == 0, "no open task -> no bonmin call"
    assert ref3.tasks == () and ref3.u_prefix == 240.0
    assert ref3.u_aircraft == 80.0, "the MAX of the universe, never its sum"

    # A genuinely empty reward-bearing universe scores 0.0, matching the historical
    # `max(..., default=0.0)`. (The builders cannot reach this -- a retained universe is
    # required -- so the helper is exercised directly.)
    assert ges._reference_aircraft_utility((), ()) == 0.0


def test_po3_terminal_credit_assignment_is_unchanged() -> None:
    """Non-terminal transitions get 0.0, the last gets R, and zero-wake attaches nothing."""
    world = _World()
    ref = _hand_reference(u_prefix=0.0, allocated=[_task("tB", _TB)], prefix_ids=())

    result = _result_stub(4)
    result.reference = ref
    breakdown = compute_episode_reward(world, result, RewardConfig())
    assert [t.reward for t in result.trajectory] == [0.0, 0.0, 0.0, breakdown.reward]

    empty = _result_stub(0)
    empty.reference = ref
    zero_wake = compute_episode_reward(world, empty, RewardConfig())
    assert empty.trajectory == []
    assert isinstance(zero_wake, EpisodeReward), "a breakdown is still returned"

    # RED LINE: the ONLY mutation is Transition.reward.
    world2 = _World()
    before = (_belief_fingerprint(world2.beliefs),
              _executor_fingerprint(world2.executor),
              _scenario_fingerprint(world2.scenario))
    r = _result_stub(2)
    r.reference = ref
    compute_episode_reward(world2, r, RewardConfig())
    after = (_belief_fingerprint(world2.beliefs),
             _executor_fingerprint(world2.executor),
             _scenario_fingerprint(world2.scenario))
    assert before == after


def test_po3_a_clean_event_conditioned_episode_matches_the_static_reward() -> None:
    """The checkable collapse: CLEAN under the opt-in policy == the historical number.

    Same world, same confirmations, same losses. The static run reads setup's t=0
    reference; the opt-in run solves the identical t=0 reference in the loop. The two
    rewards must be the same number, or the opt-in path has silently moved the clean
    condition it is supposed to preserve.
    """
    rewards = {}
    for policy in (REFERENCE_POLICY_STATIC_T0_V1, REFERENCE_POLICY_EVENT_CONDITIONED_V1):
        world = _World(reference_policy=policy)
        controller = world.clean_controller()
        with _SolverStub():
            result, _events = _run(world, controller, max_ticks=2)
        world.executor.done.update({(world.ego, "tA"), (world.peer, "tPeer")})
        world.executor.dead.add(world.peer)
        if policy == REFERENCE_POLICY_STATIC_T0_V1:
            # The historical path's reference is setup's; rebuild the SAME allocation
            # the stub solver would have produced there, so the two arms differ only in
            # WHERE the reference was solved.
            with _SolverStub():
                world.oracle_solution, world.oracle_tasks, _ = solve_and_normalize(
                    world.agents, list(world.tasks)
                )
        rewards[policy] = compute_episode_reward(
            world, result, RewardConfig(aircraft_penalty_coeff=2.25)
        )

    static = rewards[REFERENCE_POLICY_STATIC_T0_V1]
    event = rewards[REFERENCE_POLICY_EVENT_CONDITIONED_V1]
    assert event.u_prefix == 0.0
    assert event.u_ref == static.u_ref == static.u_oracle
    assert event.u_achieved == static.u_achieved
    assert event.reward == static.reward, (static.reward, event.reward)
    assert event.reference_kind == REFERENCE_KIND_CLEAN_T0


def test_po3_actor_only_and_ctde_get_identical_reference_and_reward() -> None:
    """The critic changes nothing about the reference or the reward for the same episode."""
    outcomes = {}
    for mode in ("actor_only", "ctde"):
        world = _World()
        controller = world.damaged_controller()
        central = CentralStateRecorder() if mode == "ctde" else None
        with _SolverStub():
            result, _events = _run(world, controller, max_ticks=3, central=central)
        world.executor.done.add((world.ego, "tB"))
        breakdown = compute_episode_reward(
            world, result, RewardConfig(aircraft_penalty_coeff=2.25)
        )
        outcomes[mode] = (result.reference.to_record(), breakdown)
        if mode == "ctde":
            assert len(central.samples) == len(result.trajectory), (
                "central samples stay 1:1 with actor decisions"
            )

    actor_record, actor_reward = outcomes["actor_only"]
    ctde_record, ctde_reward = outcomes["ctde"]
    # `solver_seconds` is a wall clock and legitimately differs; everything else is the
    # same reference computed from the same state.
    actor_record.pop("solver_seconds"), ctde_record.pop("solver_seconds")
    assert actor_record == ctde_record
    assert actor_reward.reward == ctde_reward.reward
    assert actor_reward.u_ref == ctde_reward.u_ref
    assert actor_reward.u_post == ctde_reward.u_post


def test_po3_the_policy_refuses_to_fall_back_on_an_unsolved_oracle() -> None:
    """An event-conditioned episode without a reference RAISES rather than scoring zero."""
    world = _World()
    assert (world.oracle_solution, world.oracle_tasks) == ({}, []), (
        "setup deliberately leaves the static pair EMPTY under this policy"
    )
    result = _result_stub(1)
    assert result.reference is None
    _expect_raises(ReferenceIntegrityError, "no reference reached the reward",
                   compute_episode_reward, world, result, RewardConfig())


def test_po3_the_realized_split_has_one_rule_site() -> None:
    """`realized_utility` and the prefix/continuation split apply the SAME all-steps rule."""
    tasks = [
        _task("t0", _TA, utility=10.0),
        Task(steps=[Step(_TA, "t1a", [], 1.0, 1, StepKind.ATTACK),
                    Step(_TB, "t1b", [], 1.0, 1, StepKind.ATTACK)], utility=20.0),
        _task("t2", _TB, utility=30.0),
    ]
    done = {("A", "t0"), ("A", "t1a"), ("A", "t2"), ("B", "t2")}
    assert realized_task_indices(tasks, done) == (0, 2)
    assert realized_utility(tasks, done) == 40.0, "the partial multi-step task pays 0"
    prefix_idx, u_prefix, prefix_ids, open_tasks = ges._reference_universe(tasks, done)
    assert prefix_idx == (0, 2) and u_prefix == 40.0
    assert prefix_ids == ("t0", "t2")
    assert [t.utility for t in open_tasks] == [20.0], (
        "the two halves PARTITION the universe -- nothing is in both, nothing is lost"
    )
    # Completing the remaining step moves the task across the split, consistently.
    done2 = done | {("C", "t1b")}
    assert realized_utility(tasks, done2) == 60.0
    assert ges._reference_universe(tasks, done2)[3] == []


# =============================================================================
# Standalone runner (nlp_env has no pytest)
# =============================================================================

def _all_tests():
    return [(name, obj) for name, obj in sorted(globals().items())
            if name.startswith("test_") and callable(obj)]


if __name__ == "__main__":
    failures = 0
    for name, fn in _all_tests():
        try:
            fn()
            print("PASS  %s" % name)
        except Exception as exc:  # pragma: no cover - standalone reporting
            failures += 1
            print("FAIL  %s: %s: %s" % (name, type(exc).__name__, exc))
            import traceback

            traceback.print_exc()
    total = len(_all_tests())
    print("-" * 72)
    print("%d passed, %d failed (of %d)" % (total - failures, failures, total))
    sys.exit(1 if failures else 0)
