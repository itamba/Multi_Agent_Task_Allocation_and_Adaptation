"""PO1 / PO2 proofs for the EXPLICIT MATCH-AOU backend selector.

WHAT IS BEING PROVED, AND WHAT IS DELIBERATELY NOT
--------------------------------------------------
The deterministic ``p = 1`` MILP is now reachable through the GENERALIZED-V1 runtime as an
EXPLICIT alternative allocation objective. It is **NOT** a transparent performance swap:
it removes the legacy ``EPSILON`` stacking incentive, so it can select a different optimal
allocation, and on the construction path ``A_init`` is what route-relative hidden placement
predicts routes from. **NO equivalence between the two objectives is claimed anywhere in
this file**, and none may be inferred from a passing run.

What IS proved here:

  * **PO1 -- explicit routing + historical preservation.** The default is the legacy
    backend everywhere; the legacy path still builds the frozen ``MatchAou`` and asks
    BONMIN; the P1 path uses ``MatchAouP1MILP`` and never reaches BONMIN; there is no
    ``auto`` and no fallback in either direction; ONE episode uses ONE backend for every
    solve it performs, including the deferred reference solves; and a backend fault ABORTS
    rather than being spent as ordinary episode attrition.
  * **PO2 -- objective / reference coherence.** Legacy ``plan_value`` is unchanged against
    hand-computed historical values; P1 ``plan_value`` is exact covered utility; a redundant
    P1 assignment is worth exactly zero while the SAME redundancy keeps its legacy EPSILON
    premium; the static and event-conditioned reference rewards are normalized by the
    episode's own objective; and ``U_prefix`` / ``U_post`` / the penalty / the regret
    denominator are untouched.

These are ENGINEERING proofs. They measure nothing scientific, and no reward, learning or
performance claim follows from them.

Run: python -m pytest tests/test_match_aou_backend_integration.py -v
     python tests/test_match_aou_backend_integration.py      (nlp_env has no pytest)
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from match_aou.models import (  # noqa: E402
    Agent,
    Capability,
    Location,
    Step,
    StepKind,
    Task,
)
from match_aou.solvers.match_aou_MINLP_solver import EPSILON  # noqa: E402
from match_aou.solvers.match_aou_backend import (  # noqa: E402
    DEFAULT_MATCH_AOU_BACKEND,
    MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
    MATCH_AOU_BACKEND_P1_MILP_V1,
    MATCH_AOU_BACKENDS,
    MatchAouBackendError,
    load_p1_milp_solver,
    resolve_match_aou_backend,
    uses_p1_milp,
)
from match_aou.rl.training import graph_episode_setup as _setup  # noqa: E402
from match_aou.rl.training import graph_train as _train  # noqa: E402
from match_aou.rl.training.graph_episode_setup import (  # noqa: E402
    SOLVE_NOT_ATTEMPTED,
    _backend_kwargs,
    build_continuation_reference,
    build_t0_reference,
    setup_episode,
    solve_and_normalize,
    solve_and_normalize_audited,
)
from match_aou.rl.training.graph_reward import (  # noqa: E402
    REFERENCE_KIND_CLEAN_T0,
    REFERENCE_POLICY_EVENT_CONDITIONED_V1,
    REFERENCE_POLICY_STATIC_T0_V1,
    RewardConfig,
    compute_episode_reward,
    episode_match_aou_backend,
    plan_value,
    realized_utility,
)
from match_aou.rl.training.graph_rollout import RolloutConfig  # noqa: E402
from match_aou.utils.blade_utils.blade_graph_executor import (  # noqa: E402
    GraphPlanExecutor,
)


# =============================================================================
# Fixtures -- the smallest duck-typed surface the seams actually read
# =============================================================================

_BASE = Location(32.85416264197241, 35.3124013096915)
_BLUE_SIDE = "side-blue"
_RED_SIDE = "side-red"


def _point_at(origin: Location, distance_km: float, bearing_deg: float) -> Location:
    radius = 6371.0088
    brg = math.radians(bearing_deg)
    lat1, lon1 = math.radians(origin.latitude), math.radians(origin.longitude)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_km / radius)
        + math.cos(lat1) * math.sin(distance_km / radius) * math.cos(brg)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brg) * math.sin(distance_km / radius) * math.cos(lat1),
        math.cos(distance_km / radius) - math.sin(lat1) * math.sin(lat2),
    )
    return Location(math.degrees(lat2), math.degrees(lon2))


_TA = _point_at(_BASE, 120.0, 45.0)
_TB = _point_at(_BASE, 200.0, 45.0)


def _task(target_id: str, loc: Location = _TA, utility: float = 80.0,
          probability: float = 1.0, n_steps: int = 1) -> Task:
    return Task(
        steps=[Step(loc, target_id, [], probability, 1, StepKind.ATTACK)
               for _ in range(n_steps)],
        utility=utility,
    )


def _agent(aid: str, loc: Location = _BASE, budget: float = 1e9) -> Agent:
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


class _Aircraft:
    def __init__(self, aid: str, loc: Location, fuel: float = 12000.0):
        self.id = aid
        self.name = "AC %s" % aid
        self.side_id = _BLUE_SIDE
        self.side_color = "blue"
        self.class_name = "F-16 Fighting Falcon"
        self.latitude, self.longitude = loc.latitude, loc.longitude
        self.altitude, self.heading, self.speed = 10000, 90.0, 1303.0
        self.current_fuel = self.max_fuel = fuel
        self.fuel_rate = 6700.0
        self.range = 100
        self.weapons: List[Any] = []
        self.home_base_id = "base-blue"
        self.target_id = None
        self.route: List[Any] = []
        self.rtb = False

    def get_weapon_with_highest_engagement_range(self) -> Any:
        return None


class _Airbase:
    def __init__(self, bid: str, loc: Location, *, side_id: str = _BLUE_SIDE,
                 side_color: str = "blue"):
        self.id = bid
        self.name = "Base %s" % bid
        self.side_id, self.side_color = side_id, side_color
        self.latitude, self.longitude = loc.latitude, loc.longitude
        self.altitude = 0
        self.aircraft: List[Any] = []


class _Scenario:
    def __init__(self, aircraft: Sequence[Any], airbases: Sequence[Any]):
        self.aircraft = list(aircraft)
        self.airbases = list(airbases)
        self.ships: List[Any] = []
        self.facilities: List[Any] = []
        self.current_time = 0

    def get_target(self, target_id: str) -> Any:
        for unit in self.airbases + self.facilities + self.ships:
            if str(unit.id) == str(target_id):
                return unit
        return None


class _Ctx:
    """A duck-typed ``EpisodeContext`` carrying exactly what the reference seam reads."""

    def __init__(self, *, backend: str = DEFAULT_MATCH_AOU_BACKEND,
                 reference_policy: str = REFERENCE_POLICY_EVENT_CONDITIONED_V1,
                 done: Sequence[Tuple[str, str]] = (),
                 declare_backend: bool = True):
        self.ego, self.peer = "ego-0", "ego-1"
        self.agent_ids = [self.ego, self.peer]
        self.agents = [_agent(self.ego, _TA), _agent(self.peer, _BASE)]
        self.t0_reference_tasks = (_task("tA", _TA), _task("tB", _TB))
        self.tasks = list(self.t0_reference_tasks)
        self.a_init = {self.ego: [(0, 0, 0)], self.peer: [(1, 0, 0)]}
        self.executor = GraphPlanExecutor(
            tasks=self.tasks, solution=self.a_init, agents=self.agents,
            arrival_threshold_km=50.0,
        )
        self.executor.done.update(done)
        base = _Airbase("base-blue", _BASE)
        self.scenario = _Scenario(
            aircraft=[_Aircraft(self.ego, _TA), _Aircraft(self.peer, _BASE)],
            airbases=[base,
                      _Airbase("tA", _TA, side_id=_RED_SIDE, side_color="red"),
                      _Airbase("tB", _TB, side_id=_RED_SIDE, side_color="red")],
        )
        self.reference_policy = reference_policy
        if reference_policy == REFERENCE_POLICY_EVENT_CONDITIONED_V1:
            self.oracle_solution, self.oracle_tasks = {}, []
        else:
            self.oracle_solution = dict(self.a_init)
            self.oracle_tasks = list(self.tasks)
            self.t0_reference_tasks = ()
        # `declare_backend=False` models a context from BEFORE this field existed, which
        # `episode_match_aou_backend` must resolve to the PRESERVED objective.
        if declare_backend:
            self.match_aou_backend = backend


class _SolverSpy:
    """Replaces BOTH backend entry points in ``graph_episode_setup`` and records use.

    Patching the two SITES rather than the solver classes is what makes "the P1 path never
    reaches BONMIN" checkable: the legacy stand-in RAISES if it is entered at all under a
    P1 selection, so a fallback could not pass silently.
    """

    def __init__(self, *, selected: Optional[Sequence[int]] = None,
                 legacy_forbidden: bool = False, p1_forbidden: bool = False,
                 p1_raises: Optional[BaseException] = None,
                 termination: str = "optimal", fail: bool = False):
        self.legacy_calls: List[Dict[str, Any]] = []
        self.p1_calls: List[Dict[str, Any]] = []
        self._selected = selected
        self._legacy_forbidden = legacy_forbidden
        self._p1_forbidden = p1_forbidden
        self._p1_raises = p1_raises
        self._termination = termination
        self._fail = fail
        self._saved: Dict[str, Any] = {}

    def _answer(self, agents, tasks, record):
        if self._fail:
            return None, [], SimpleNamespace(
                solver=SimpleNamespace(termination_condition=self._termination))
        chosen = (list(range(len(tasks))) if self._selected is None
                  else list(self._selected))
        solution: Dict[str, List[Tuple[int, int]]] = {}
        for n, j in enumerate(chosen):
            solution.setdefault(str(agents[n % len(agents)].id), []).append((j, 0))
        unselected = [j for j in range(len(tasks)) if j not in set(chosen)]
        return solution, unselected, SimpleNamespace(
            solver=SimpleNamespace(termination_condition=self._termination))

    def __enter__(self) -> "_SolverSpy":
        outer = self

        def legacy(agents, tasks, precedence):
            if outer._legacy_forbidden:
                raise AssertionError(
                    "the LEGACY MINLP / BONMIN path was entered under a P1 selection"
                )
            outer.legacy_calls.append({"n_agents": len(agents), "n_tasks": len(tasks)})
            sol, unsel, results = outer._answer(agents, tasks, outer.legacy_calls)
            return sol, unsel, results, 0.0

        def p1(agents, tasks, precedence):
            if outer._p1_forbidden:
                raise AssertionError("the P1 path was entered under a legacy selection")
            outer.p1_calls.append({"n_agents": len(agents), "n_tasks": len(tasks)})
            if outer._p1_raises is not None:
                raise outer._p1_raises
            sol, unsel, results = outer._answer(agents, tasks, outer.p1_calls)
            return sol, unsel, results, 0.0

        for name, fn in (("_solve_legacy_minlp", legacy), ("_solve_p1_milp", p1)):
            self._saved[name] = getattr(_setup, name)
            setattr(_setup, name, fn)
        return self

    def __exit__(self, *exc: Any) -> None:
        for name, original in self._saved.items():
            setattr(_setup, name, original)


def _cfg(**overrides: Any) -> Any:
    base: Dict[str, Any] = dict(
        n_iterations=1, episodes_per_iteration=1, eval_every=0, eval_episodes=0,
    )
    base.update(overrides)
    return _train.TrainConfig(**base)


def _raises(exc_type: Any, fn: Any, *args: Any, **kwargs: Any) -> BaseException:
    try:
        fn(*args, **kwargs)
    except exc_type as exc:  # noqa: PERF203 - the assertion IS the point
        return exc
    raise AssertionError("expected %s, nothing raised" % exc_type.__name__)


# =============================================================================
# PO1 -- explicit routing + historical preservation
# =============================================================================

def test_po1_every_default_is_the_historical_legacy_backend() -> None:
    """Nothing opts in by accident: every default resolves to the frozen MINLP."""
    assert DEFAULT_MATCH_AOU_BACKEND == MATCH_AOU_BACKEND_LEGACY_MINLP_V1
    assert set(MATCH_AOU_BACKENDS) == {
        MATCH_AOU_BACKEND_LEGACY_MINLP_V1, MATCH_AOU_BACKEND_P1_MILP_V1
    }
    assert _cfg().match_aou_backend == MATCH_AOU_BACKEND_LEGACY_MINLP_V1
    assert RolloutConfig().match_aou_backend == MATCH_AOU_BACKEND_LEGACY_MINLP_V1
    assert resolve_match_aou_backend(None) == MATCH_AOU_BACKEND_LEGACY_MINLP_V1
    # A context from before the field existed is a HISTORICAL episode, never an opt-in one.
    assert episode_match_aou_backend(SimpleNamespace()) == \
        MATCH_AOU_BACKEND_LEGACY_MINLP_V1
    assert uses_p1_milp(DEFAULT_MATCH_AOU_BACKEND) is False


def test_po1_legacy_backend_routes_to_the_frozen_minlp_and_asks_bonmin() -> None:
    """The historical path still builds the frozen ``MatchAou`` and names BONMIN."""
    seen: Dict[str, Any] = {}

    class _Model:
        def __init__(self, agents, tasks, precedence_relations=None, risk_factor=0.0):
            seen["constructed"] = True
            seen["risk_factor"] = risk_factor
            seen["n_agents"] = len(agents)
            seen["n_tasks"] = len(tasks)

        def solve(self, solver_name="bonmin"):
            seen["solver_name"] = solver_name
            return ({"e0": [(0, 0)]}, SimpleNamespace(
                solver=SimpleNamespace(termination_condition="optimal")), [])

    original = _setup.MatchAou
    _setup.MatchAou = _Model  # type: ignore[assignment]
    try:
        solution, allocated, _unsel, audit = solve_and_normalize_audited(
            [_agent("e0")], [_task("tA")]
        )
    finally:
        _setup.MatchAou = original  # type: ignore[assignment]

    assert seen["constructed"] is True
    # `SOLVER_NAME` is the module's single source of truth; it must be what is asked for.
    assert seen["solver_name"] == _setup.SOLVER_NAME == "bonmin"
    assert seen["risk_factor"] == 0.0
    assert audit.invoked is True and audit.accepted is True
    assert solution and allocated


def test_po1_p1_backend_uses_the_p1_solver_and_never_touches_bonmin() -> None:
    """Selecting P1 reaches ``MatchAouP1MILP`` and CANNOT reach the BONMIN path."""
    poisoned: Dict[str, Any] = {}

    class _Poison:
        def __init__(self, *a: Any, **k: Any):
            poisoned["legacy_entered"] = True
            raise AssertionError("the frozen MINLP must not be constructed under P1")

    original = _setup.MatchAou
    _setup.MatchAou = _Poison  # type: ignore[assignment]
    try:
        solution, allocated, _unsel, audit = solve_and_normalize_audited(
            [_agent("e0"), _agent("e1")],
            [_task("tA", _TA), _task("tB", _TB)],
            backend=MATCH_AOU_BACKEND_P1_MILP_V1,
        )
    finally:
        _setup.MatchAou = original  # type: ignore[assignment]

    assert "legacy_entered" not in poisoned
    assert audit.invoked is True and audit.accepted is True
    # A REAL HiGHS solve ran: both targets are covered and the audit says optimal.
    assert audit.termination_condition == "optimal"
    assert len(allocated) == 2
    assert sum(len(v) for v in solution.values()) >= 2


def test_po1_the_two_backends_are_reached_through_their_own_sites_only() -> None:
    """Each selection enters exactly one solve site; the other one raises if entered."""
    agents, tasks = [_agent("e0")], [_task("tA")]

    with _SolverSpy(p1_forbidden=True) as spy:
        solve_and_normalize(agents, tasks)
        solve_and_normalize(agents, tasks, backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1)
    assert len(spy.legacy_calls) == 2 and spy.p1_calls == []

    with _SolverSpy(legacy_forbidden=True) as spy:
        solve_and_normalize(agents, tasks, backend=MATCH_AOU_BACKEND_P1_MILP_V1)
    assert len(spy.p1_calls) == 1 and spy.legacy_calls == []


def test_po1_there_is_no_auto_and_no_fallback_in_either_direction() -> None:
    """An unknown id is refused, and a P1 refusal never re-asks the legacy solver."""
    for bad in ("auto", "", "P1_MILP_V1", "p1", "legacy", 0):
        _raises(MatchAouBackendError, resolve_match_aou_backend, bad)
    assert "auto" not in MATCH_AOU_BACKENDS

    # P1 refuses the problem -> the legacy path must NOT be tried as a rescue.
    # The REAL `_solve_p1_milp` runs here (only the LEGACY site is replaced), so this
    # proves the refusal, its wrapping, and the absence of a rescue in one pass.
    poisoned: Dict[str, Any] = {}

    class _Poison:
        def __init__(self, *a: Any, **k: Any):
            poisoned["legacy_entered"] = True
            raise AssertionError("the legacy MINLP must not rescue a refused P1 solve")

    original = _setup.MatchAou
    _setup.MatchAou = _Poison  # type: ignore[assignment]
    try:
        exc = _raises(
            MatchAouBackendError, solve_and_normalize,
            [_agent("e0")], [_task("tA", n_steps=2)],
            backend=MATCH_AOU_BACKEND_P1_MILP_V1,
        )
    finally:
        _setup.MatchAou = original  # type: ignore[assignment]
    assert "legacy_entered" not in poisoned
    assert "never answered by falling back" in str(exc)


def test_po1_a_p1_contract_violation_is_a_backend_fault_not_a_solver_failure() -> None:
    """Multi-step / p != 1 / precedence are CONFIGURATION faults, and they say so."""
    agents = [_agent("e0")]
    for tasks, precedence, label in (
        ([_task("tA", n_steps=2)], None, "multi-step"),
        ([_task("tA", probability=0.5)], None, "p != 1"),
        # A LEGAL precedence relation (task 0 before task 1), so the legacy comparison
        # below is a real answer rather than a degenerate self-cycle.
        ([_task("tA", _TA), _task("tB", _TB)], [(0, 1)], "precedence"),
    ):
        exc = _raises(
            MatchAouBackendError, solve_and_normalize_audited,
            agents, tasks, precedence, backend=MATCH_AOU_BACKEND_P1_MILP_V1,
        )
        assert MATCH_AOU_BACKEND_P1_MILP_V1 in str(exc), label
        # It is NOT an ordinary attempt failure: the trainer's accounted wrapper would
        # have swallowed it into a pipeline stage.
        assert not isinstance(exc, _train.EpisodeAttemptError), label
        # The LEGACY backend still ACCEPTS the very same input -- so this is a statement
        # about the selected objective, not about the world. Stubbed at the legacy solve
        # site so the check needs no BONMIN.
        with _SolverSpy(p1_forbidden=True) as spy:
            legacy = solve_and_normalize_audited(agents, tasks, precedence)
        assert len(spy.legacy_calls) == 1, label
        assert legacy[3].invoked is True, label


def test_po1_a_solver_that_did_not_answer_keeps_the_existing_failure_semantics() -> None:
    """A non-optimal termination stays an ordinary solve failure under BOTH backends."""
    agents, tasks = [_agent("e0")], [_task("tA")]
    for backend in MATCH_AOU_BACKENDS:
        with _SolverSpy(fail=True, termination="maxIterations"):
            solution, allocated, unselected, audit = solve_and_normalize_audited(
                agents, tasks, **_backend_kwargs(backend)
            )
        assert solution == {} and allocated == [] and unselected == [0]
        assert audit.invoked is True and audit.accepted is False
        assert audit.termination_condition == "maxIterations"


def test_po1_the_degenerate_short_circuit_is_backend_independent() -> None:
    """No tasks / no agents is a SKIPPED solve under either backend -- and costs none."""
    for backend in MATCH_AOU_BACKENDS:
        with _SolverSpy(legacy_forbidden=True, p1_forbidden=True):
            for agents, tasks in (([_agent("e0")], []), ([], [_task("tA")])):
                _sol, _alloc, _unsel, audit = solve_and_normalize_audited(
                    agents, tasks, **_backend_kwargs(backend)
                )
                assert audit.invoked is False and audit.accepted is True
                assert audit.termination_condition == SOLVE_NOT_ATTEMPTED


def test_po1_setup_episode_refuses_an_unknown_backend_before_any_blade_object() -> None:
    """The id is validated before an environment exists, so a refusal costs nothing."""
    def _forbidden(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("_build_env must not run for an unknown backend")

    original = _setup._build_env
    _setup._build_env = _forbidden  # type: ignore[assignment]
    try:
        _raises(MatchAouBackendError, setup_episode, "{}", match_aou_backend="auto")
    finally:
        _setup._build_env = original  # type: ignore[assignment]


def test_po1_every_deferred_reference_solve_uses_the_episodes_stored_backend() -> None:
    """One episode, one backend: the deferred solves READ it rather than re-deciding."""
    for backend in MATCH_AOU_BACKENDS:
        other_forbidden = dict(
            legacy_forbidden=(backend == MATCH_AOU_BACKEND_P1_MILP_V1),
            p1_forbidden=(backend == MATCH_AOU_BACKEND_LEGACY_MINLP_V1),
        )
        ctx = _Ctx(backend=backend)
        assert episode_match_aou_backend(ctx) == backend

        with _SolverSpy(**other_forbidden) as spy:
            clean = build_t0_reference(ctx, kind=REFERENCE_KIND_CLEAN_T0)
        used = spy.p1_calls if backend == MATCH_AOU_BACKEND_P1_MILP_V1 else spy.legacy_calls
        assert len(used) == 1, backend
        assert clean.solver_accepted is True

        with _SolverSpy(**other_forbidden) as spy:
            cont = build_continuation_reference(
                ctx, scenario=ctx.scenario, tick=7, damaged_ego_id=ctx.ego,
            )
        used = spy.p1_calls if backend == MATCH_AOU_BACKEND_P1_MILP_V1 else spy.legacy_calls
        assert len(used) == 1, backend
        assert cont.checkpoint_tick == 7


def test_po1_a_backend_fault_aborts_and_is_never_accounted_attrition() -> None:
    """``_run_one_episode`` re-raises it AHEAD of every accounted-stage wrapper."""
    def _boom(*_a: Any, **_k: Any) -> Any:
        raise MatchAouBackendError("configured against a domain it does not model")

    original = _train.setup_episode
    _train.setup_episode = _boom  # type: ignore[assignment]
    try:
        exc = _raises(
            MatchAouBackendError, _train._run_one_episode,
            SimpleNamespace(), _StubGenerator(), _cfg(),
            seed=0, episode_tag=0, deterministic=True,
        )
    finally:
        _train.setup_episode = original  # type: ignore[assignment]

    # It escaped as ITSELF -- not wrapped, and so never written to the ledger with a
    # pipeline stage, never tallied, and never entered into `skip_and_account_v1`.
    assert isinstance(exc, MatchAouBackendError)
    assert not isinstance(exc, _train.EpisodeAttemptError)
    assert not isinstance(exc, _train.MeasurementIntegrityError)


class _StubGenerator:
    """The generator surface ``_run_one_episode`` touches before setup is reached.

    It writes a real (empty) scenario file, because ``_run_one_episode`` READS the
    generated path to build ``setup_episode``'s argument -- so a non-existent path would
    fail earlier, at generation, and would never reach the seam under test.
    """

    def __init__(self) -> None:
        self._dir = tempfile.mkdtemp(prefix="backend_integration_")

    def generate(self, episode: int, config: Any) -> Path:
        path = Path(self._dir) / ("scenario_%d.json" % int(episode))
        path.write_text("{}", encoding="utf-8")
        return path


def test_po1_the_harnesses_omit_the_keyword_on_the_historical_backend() -> None:
    """Keyword OMISSION, so the historical call is the pre-integration call itself."""
    assert _train._backend_setup_kwargs(_cfg()) == {}
    assert _train._backend_setup_kwargs(
        _cfg(match_aou_backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1)) == {}
    assert _train._backend_setup_kwargs(
        _cfg(match_aou_backend=MATCH_AOU_BACKEND_P1_MILP_V1)
    ) == {"match_aou_backend": MATCH_AOU_BACKEND_P1_MILP_V1}
    # The same discipline one level down, at the solve seam.
    assert _backend_kwargs(MATCH_AOU_BACKEND_LEGACY_MINLP_V1) == {}
    assert _backend_kwargs(MATCH_AOU_BACKEND_P1_MILP_V1) == \
        {"backend": MATCH_AOU_BACKEND_P1_MILP_V1}


def test_po1_the_backend_is_independent_of_episode_design() -> None:
    """Neither selector constrains or resolves the other."""
    from match_aou.rl.training.graph_generalized import (
        EPISODE_DESIGN_FIXED_CELL_V1,
        EpisodeDesign,
        resolve_episode_design,
    )
    # `EpisodeDesign` carries FOUR low-level policy ids and no backend.
    design = resolve_episode_design(EPISODE_DESIGN_FIXED_CELL_V1)
    assert not hasattr(design, "match_aou_backend")
    assert "match_aou_backend" not in design.to_record()
    assert "match_aou_backend" not in {f for f in EpisodeDesign.__dataclass_fields__}

    # BOTH designs accept BOTH backends; validate() refuses neither combination.
    for backend in MATCH_AOU_BACKENDS:
        cfg = _cfg(match_aou_backend=backend)
        cfg.validate()
        assert cfg.design.design == EPISODE_DESIGN_FIXED_CELL_V1
        assert cfg.match_aou_backend == backend


def test_po1_the_backend_module_does_not_eagerly_load_the_p1_solver() -> None:
    """Naming a backend must not drag the MILP stack in; the loader is the only door."""
    import subprocess
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; import match_aou.solvers.match_aou_backend as b; "
         "import match_aou.rl.training.graph_episode_setup; "
         "print('match_aou.solvers.match_aou_p1_milp_solver' in sys.modules)"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "False", (
        "importing the backend contract (or the solve seam) must not load the P1 module; "
        "probe said %r" % probe.stdout.strip()
    )
    # And the lazy door really does open onto the approved class.
    assert load_p1_milp_solver().__name__ == "MatchAouP1MILP"


def test_po1_benchmark_preflight_probes_under_the_runs_own_backend() -> None:
    """Population SELECTION and later evaluation must use the SAME objective."""
    from match_aou.rl.training import graph_benchmark_preflight as bp

    # It reuses the trainer's own keyword-omission helper, so the two cannot diverge.
    assert bp._backend_setup_kwargs is _train._backend_setup_kwargs
    # And an unknown backend is refused before any candidate is attempted.
    cfg = _cfg(
        episode_design="generalized_v1",
        fuel_damage_mode="seeded_variable",
        generalized_max_attempts_per_iteration=1,
    )
    object.__setattr__(cfg, "match_aou_backend", "auto")
    _raises(MatchAouBackendError, bp._require_preflight_config, cfg)


# =============================================================================
# PO2 -- objective / reference coherence
# =============================================================================

def _legacy_term(utility: float, m: int, probability: float = 1.0) -> float:
    """The historical per-task term, written out independently of the implementation."""
    return utility * (1.0 - (1.0 - probability + EPSILON) ** m)


def test_po2_legacy_plan_value_is_unchanged_against_hand_computed_values() -> None:
    """Pinned against the EPSILON formula transcribed here, not against the code."""
    tasks = [_task("tA", _TA, utility=80.0), _task("tB", _TB, utility=90.0)]

    one_each = {"e0": [(0, 0, 0)], "e1": [(1, 0, 0)]}
    assert plan_value(one_each, tasks) == \
        _legacy_term(80.0, 1) + _legacy_term(90.0, 1)

    stacked = {"e0": [(0, 0, 0)], "e1": [(0, 0, 0)]}
    assert plan_value(stacked, tasks) == _legacy_term(80.0, 2)

    # An unselected task contributes exactly zero (m == 0 -> factor 0).
    assert plan_value({}, tasks) == 0.0
    # The DEFAULT is the legacy arithmetic, so a pre-integration caller is unaffected.
    assert plan_value(one_each, tasks) == \
        plan_value(one_each, tasks, backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1)
    # ...and it is genuinely EPSILON-discounted, i.e. NOT already exact utility.
    assert plan_value(one_each, tasks) < 170.0


def test_po2_p1_plan_value_is_exactly_covered_utility() -> None:
    """Covered -> exactly ``task.utility``; uncovered -> exactly 0. No EPSILON."""
    tasks = [_task("tA", _TA, utility=80.0), _task("tB", _TB, utility=90.0)]
    p1 = MATCH_AOU_BACKEND_P1_MILP_V1

    assert plan_value({"e0": [(0, 0, 0)], "e1": [(1, 0, 0)]}, tasks, backend=p1) == 170.0
    assert plan_value({"e0": [(0, 0, 0)]}, tasks, backend=p1) == 80.0
    assert plan_value({"e0": [(1, 0, 0)]}, tasks, backend=p1) == 90.0
    assert plan_value({}, tasks, backend=p1) == 0.0
    # EXACT equality, not approximate: no epsilon participates anywhere.
    assert plan_value({"e0": [(0, 0, 0)]}, tasks, backend=p1) == float(tasks[0].utility)


def test_po2_a_redundant_p1_assignment_is_worth_exactly_zero() -> None:
    """The stacking incentive is GONE under P1 -- and still THERE under legacy."""
    tasks = [_task("tA", _TA, utility=80.0)]
    single = {"e0": [(0, 0, 0)]}
    double = {"e0": [(0, 0, 0)], "e1": [(0, 0, 0)]}
    triple = {"e0": [(0, 0, 0)], "e1": [(0, 0, 0)], "e2": [(0, 0, 0)]}
    p1, legacy = MATCH_AOU_BACKEND_P1_MILP_V1, MATCH_AOU_BACKEND_LEGACY_MINLP_V1

    p1_single = plan_value(single, tasks, backend=p1)
    assert plan_value(double, tasks, backend=p1) - p1_single == 0.0
    assert plan_value(triple, tasks, backend=p1) - p1_single == 0.0

    # THE SAME redundancy keeps its legacy premium -- that is why the two objectives do
    # not share an optimal allocation set, and it is preserved rather than "cleaned up".
    legacy_single = plan_value(single, tasks, backend=legacy)
    assert plan_value(double, tasks, backend=legacy) - legacy_single > 0.0
    assert plan_value(double, tasks, backend=legacy) == _legacy_term(80.0, 2)

    # A single agent listing the same cell twice is still ONE agent under both.
    dup = {"e0": [(0, 0, 0), (0, 0, 0)]}
    assert plan_value(dup, tasks, backend=p1) == p1_single
    assert plan_value(dup, tasks, backend=legacy) == legacy_single


def test_po2_p1_plan_value_refuses_inputs_outside_its_contract() -> None:
    """A confidently wrong denominator is refused; the legacy valuation still answers."""
    p1 = MATCH_AOU_BACKEND_P1_MILP_V1
    sol = {"e0": [(0, 0, 0)]}
    for tasks, label in (
        ([_task("tA", n_steps=2)], "multi-step"),
        ([_task("tA", probability=0.5)], "p != 1"),
    ):
        _raises(MatchAouBackendError, plan_value, sol, tasks, backend=p1)
        plan_value(sol, tasks)  # the legacy valuation is unchanged and still answers
        assert label  # keeps the label meaningful in a failure message


def test_po2_the_static_reference_reward_uses_the_episodes_own_objective() -> None:
    """``U_oracle`` is measured under the backend the oracle was solved with."""
    result = SimpleNamespace(trajectory=[SimpleNamespace(reward=None)], reference=None)
    cfg = RewardConfig(aircraft_penalty_coeff=2.25)

    legacy_ctx = _Ctx(backend=MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
                      reference_policy=REFERENCE_POLICY_STATIC_T0_V1)
    p1_ctx = _Ctx(backend=MATCH_AOU_BACKEND_P1_MILP_V1,
                  reference_policy=REFERENCE_POLICY_STATIC_T0_V1)
    legacy_reward = compute_episode_reward(legacy_ctx, result, cfg)
    p1_reward = compute_episode_reward(p1_ctx, result, cfg)

    tasks = list(legacy_ctx.oracle_tasks)
    assert legacy_reward.u_oracle == plan_value(legacy_ctx.oracle_solution, tasks)
    assert p1_reward.u_oracle == plan_value(
        p1_ctx.oracle_solution, tasks, backend=MATCH_AOU_BACKEND_P1_MILP_V1)
    # Exact covered utility under P1; strictly EPSILON-discounted under legacy.
    assert p1_reward.u_oracle == 160.0
    assert legacy_reward.u_oracle < p1_reward.u_oracle
    # `u_ref` is the denominator source under BOTH, and equals `u_oracle` on this path.
    assert legacy_reward.u_ref == legacy_reward.u_oracle
    assert p1_reward.u_ref == p1_reward.u_oracle
    assert p1_reward.reference_policy == REFERENCE_POLICY_STATIC_T0_V1
    # A context that predates the field is scored on the PRESERVED objective.
    undeclared = _Ctx(reference_policy=REFERENCE_POLICY_STATIC_T0_V1,
                      declare_backend=False)
    assert compute_episode_reward(undeclared, result, cfg).u_oracle == \
        legacy_reward.u_oracle


def test_po2_event_conditioned_references_use_the_episodes_own_objective() -> None:
    """CLEAN t=0 and DAMAGED continuation references alike are valued under it."""
    for backend, expected_full in (
        (MATCH_AOU_BACKEND_P1_MILP_V1, 160.0),
        (MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
         _legacy_term(80.0, 1) + _legacy_term(80.0, 1)),
    ):
        ctx = _Ctx(backend=backend)
        with _SolverSpy():
            clean = build_t0_reference(ctx, kind=REFERENCE_KIND_CLEAN_T0)
        assert clean.u_prefix == 0.0
        assert clean.u_cont_ref == expected_full
        assert clean.u_ref == expected_full

        # DAMAGED: one target already realized, so the prefix is split out of the
        # continuation universe and `U_ref = U_prefix + U_cont_ref` closes.
        damaged_ctx = _Ctx(backend=backend, done=(("ego-0", "tA"),))
        with _SolverSpy():
            cont = build_continuation_reference(
                damaged_ctx, scenario=damaged_ctx.scenario, tick=11,
                damaged_ego_id=damaged_ctx.ego,
            )
        # `U_prefix` is EXACT covered utility under BOTH objectives -- it is a plain sum
        # of `task.utility` over realized tasks and carries no EPSILON either way.
        assert cont.u_prefix == 80.0
        assert cont.u_ref == cont.u_prefix + cont.u_cont_ref
        assert cont.checkpoint_tick == 11
        assert "tA" in cont.prefix_target_ids


def test_po2_prefix_post_penalty_and_denominator_semantics_are_unchanged() -> None:
    """Only the reference VALUATION is backend-aware; the arithmetic around it is not."""
    ctx = _Ctx(backend=MATCH_AOU_BACKEND_P1_MILP_V1, done=(("ego-0", "tA"),))
    with _SolverSpy():
        reference = build_continuation_reference(
            ctx, scenario=ctx.scenario, tick=5, damaged_ego_id=ctx.ego)

    result = SimpleNamespace(
        trajectory=[SimpleNamespace(reward=None) for _ in range(3)],
        reference=reference,
    )
    cfg = RewardConfig(aircraft_penalty_coeff=2.25)
    reward = compute_episode_reward(ctx, result, cfg)

    # `realized_utility` is UNTOUCHED by the backend: it is exact utility either way.
    u_post = realized_utility(list(reference.tasks), ctx.executor.done)
    assert reward.u_post == u_post
    assert reward.u_prefix == reference.u_prefix          # frozen at the checkpoint
    assert reward.u_achieved == reward.u_prefix + reward.u_post

    denom = abs(reference.u_ref) + cfg.regret_epsilon      # the SAME division guard
    assert reward.ratio == (reward.u_achieved - reference.u_ref) / denom
    assert reward.penalty == (
        cfg.aircraft_penalty_coeff * reference.u_aircraft * reward.n_lost) / denom
    assert reward.reward == reward.ratio - reward.penalty  # never clamped
    assert reward.u_oracle is None                         # no static optimum exists
    # Terminal-on-last credit placement is unchanged under both backends.
    assert [t.reward for t in result.trajectory[:-1]] == [0.0, 0.0]
    assert result.trajectory[-1].reward == reward.reward


def test_po2_the_resolved_backend_is_recorded_durably_in_the_run_config() -> None:
    """A reader can always answer "which objective produced this run?"."""
    from dataclasses import asdict
    for backend in MATCH_AOU_BACKENDS:
        cfg = _cfg(match_aou_backend=backend)
        assert asdict(cfg)["match_aou_backend"] == backend
        provenance = _train.collect_provenance(cfg)
        assert provenance["solver"]["match_aou_backend"] == backend
        # BONMIN's own probe is still recorded on both, so the environment fact survives.
        assert "bonmin" in provenance["solver"]


# =============================================================================
# Standalone runner (CLAUDE.md section 1: nlp_env has no pytest)
# =============================================================================

if __name__ == "__main__":
    import traceback

    failures = passed = 0
    for _name, _fn in sorted(globals().items()):
        if not _name.startswith("test_") or not callable(_fn):
            continue
        try:
            _fn()
            passed += 1
            print("OK   %s" % _name)
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            failures += 1
            print("FAIL %s: %s: %s" % (_name, type(exc).__name__, exc))
            traceback.print_exc()

    print("")
    print("=" * 72)
    print("%d passed, %d failed" % (passed, failures))
    print("=" * 72)
    sys.exit(1 if failures else 0)
