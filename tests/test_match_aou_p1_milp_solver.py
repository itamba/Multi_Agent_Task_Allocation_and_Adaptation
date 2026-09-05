"""Proof tests for the deterministic ``p = 1`` MILP solver (``MatchAouP1MILP``).

Organized by the task's three proof obligations:

  PO1 -- specialized-domain mathematical correctness (exact 0 / exact full utility, no
         EPSILON, multi-agent feasible with zero marginal gain, capability + round-trip
         fuel rules identical to the frozen solver).
  PO2 -- legacy isolation and output compatibility (frozen file byte-identical, no active
         runtime file touched, ``(task_idx, 0)`` assignments, exact unselected semantics,
         a readable ``results.solver.termination_condition``, and no way for a
         non-optimal outcome to be reported as an optimal empty allocation).
  PO3 -- same-input engineering comparison against the legacy BONMIN MINLP. The full
         matrix lives in ``tools/benchmark_match_aou_p1_milp.py``; the tests here pin the
         same-input agreement on small hand-built worlds and are SKIPPED when no bonmin
         executable is reachable.

Most PO1 checks are deliberately hand-evaluated against the built model
(``c``, ``bounds``, ``constraint_matrix``) rather than asserted only through solver
output: a solver-output-only test cannot distinguish "the formulation is right" from
"the solver happened to agree".

BONMIN discovery for the PO3 tier: ``bonmin`` on PATH, else the
``MATCH_AOU_BONMIN_EXECUTABLE`` environment variable. Nothing is hardcoded to one
machine, and the tier skips rather than failing when neither is present.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pytest is optional: absent in nlp_env, so keep the __main__ runner usable.
    import pytest
except ImportError:  # pragma: no cover - standalone mode
    pytest = None  # type: ignore[assignment]


# =============================================================================
# Minimal pytest compatibility surface (standalone mode ONLY)
# =============================================================================
# Built ONLY when real pytest is missing. With pytest installed this whole block
# is skipped and `python -m pytest` behaviour and collection are untouched.
#
# Scope is deliberately this file's needs and nothing more -- `mark.parametrize`,
# `mark.skipif`, `raises`, `approx` (scalars) and `skip`. This is NOT a pytest
# clone, and anything it does not implement must fail loudly rather than quietly
# passing.
#
# The mark objects expose `.name` / `.args` / `.kwargs` and attach through
# `pytestmark`, which is exactly the surface real pytest's `MarkDecorator`
# exposes and exactly what `_cases()` / `_skip_reason()` already read -- so the
# standalone runner expands the SAME cases in both modes.

if pytest is None:  # pragma: no cover - exercised only where pytest is absent

    class Skipped(Exception):
        """Raised by the shim's ``skip``. Named to match pytest's own outcome.

        The runner classifies by ``type(exc).__name__ == "Skipped"``, so a skip
        is counted as SKIP in both modes rather than being mistaken for a pass.
        """

    class _Mark:
        """A pytest-compatible mark: ``.name`` / ``.args`` / ``.kwargs``."""

        def __init__(self, name: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            self.name = name
            self.args = args
            self.kwargs = kwargs

        def __call__(self, func: Any) -> Any:
            # Append, mirroring pytest's accumulation of stacked decorators.
            func.pytestmark = [*getattr(func, "pytestmark", []), self]
            return func

    class _MarkFactory:
        """``pytest.mark.*`` -- only the two marks this file actually uses."""

        @staticmethod
        def parametrize(argnames: Any, argvalues: Any, **kwargs: Any) -> _Mark:
            return _Mark("parametrize", (argnames, argvalues), kwargs)

        @staticmethod
        def skipif(condition: Any, *, reason: str = "skipif") -> _Mark:
            return _Mark("skipif", (condition,), {"reason": reason})

        def __getattr__(self, name: str) -> Any:
            raise NotImplementedError(
                f"standalone pytest shim does not implement pytest.mark.{name}; "
                "run this file under real pytest, or extend the shim deliberately"
            )

    class _RaisesContext:
        """``pytest.raises(Expected, match=...)`` with the same failure modes."""

        def __init__(self, expected: Any, match: Optional[str]) -> None:
            self.expected = expected
            self.match = match
            self.value: Optional[BaseException] = None

        def __enter__(self) -> "_RaisesContext":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            if exc_type is None:
                raise AssertionError(f"DID NOT RAISE {self.expected!r}")
            if not issubclass(exc_type, self.expected):
                return False  # wrong exception type: let it propagate as a failure
            if self.match is not None and re.search(self.match, str(exc)) is None:
                raise AssertionError(
                    f"pattern {self.match!r} not found in {str(exc)!r}"
                )
            self.value = exc
            return True  # matched: suppress

    class _Approx:
        """Scalar-only ``pytest.approx``. Non-scalars fail loudly, never silently."""

        # Stop numpy scalars from trying to handle the comparison themselves, so
        # Python falls back to this object's reflected __eq__.
        __array_ufunc__ = None

        def __init__(self, expected: Any, rel: float = 1e-6, abs: float = 1e-12) -> None:
            try:
                self.expected = float(expected)
            except (TypeError, ValueError) as exc:
                raise NotImplementedError(
                    "standalone pytest shim implements approx for SCALARS only; "
                    f"got {expected!r}"
                ) from exc
            self.rel = rel
            self.abs = abs

        def __eq__(self, other: Any) -> bool:
            try:
                value = float(other)
            except (TypeError, ValueError):
                return NotImplemented
            tolerance = max(self.abs, self.rel * max(abs(self.expected), abs(value)))
            return abs(value - self.expected) <= tolerance

        def __repr__(self) -> str:
            return f"approx({self.expected!r} +- rel={self.rel} abs={self.abs})"

    class _PytestShim:
        """The tiny ``pytest`` stand-in this module binds when pytest is absent."""

        mark = _MarkFactory()
        # Only ever referenced inside postponed annotations, which are never
        # evaluated; the standalone runner supplies its own `_MonkeyPatch`.
        MonkeyPatch = object

        @staticmethod
        def raises(expected: Any, match: Optional[str] = None) -> _RaisesContext:
            return _RaisesContext(expected, match)

        @staticmethod
        def approx(expected: Any, rel: float = 1e-6, abs: float = 1e-12) -> _Approx:
            return _Approx(expected, rel=rel, abs=abs)

        @staticmethod
        def skip(reason: str = "") -> None:
            raise Skipped(reason)

        def __getattr__(self, name: str) -> Any:
            raise NotImplementedError(
                f"standalone pytest shim does not implement pytest.{name}; run "
                "this file under real pytest, or extend the shim deliberately"
            )

    pytest = _PytestShim()  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from match_aou.models.agent import Agent  # noqa: E402
from match_aou.models.capability import Capability  # noqa: E402
from match_aou.models.location import Location  # noqa: E402
from match_aou.models.step import Step, StepKind  # noqa: E402
from match_aou.models.task import Task  # noqa: E402
from match_aou.solvers.match_aou_MINLP_solver import EPSILON as LEGACY_EPSILON  # noqa: E402
from match_aou.solvers.match_aou_MINLP_solver import round_trip_cost  # noqa: E402
from match_aou.solvers.match_aou_p1_milp_solver import (  # noqa: E402
    TERMINATION_OPTIMAL,
    MatchAouP1MILP,
    P1MilpUnsupportedInputError,
)
import match_aou.solvers.match_aou_p1_milp_solver as p1_module  # noqa: E402


# =============================================================================
# Hand-built world helpers -- deterministic, BLADE-free, solver-free
# =============================================================================

def flat_cost(source: Location, destination: Location) -> float:
    """Manhattan cost in degrees: trivially hand-checkable, unlike haversine."""
    return abs(source.latitude - destination.latitude) + abs(
        source.longitude - destination.longitude
    )


def mk_agent(
    agent_id: str,
    *,
    lat: float = 0.0,
    lon: float = 0.0,
    budget: float = 1_000.0,
    capabilities: Sequence[str] = ("attack",),
    location: Optional[Location] = "unset",  # type: ignore[assignment]
    return_location: Optional[Location] = "unset",  # type: ignore[assignment]
) -> Agent:
    """An agent whose round trip to (lat_t, lon_t) costs exactly 2*|lat_t| + 2*|lon_t|.

    ``location`` / ``return_location`` accept an explicit ``None`` (the sentinel default
    means "derive from lat/lon"), so the missing-location branches are testable.
    """
    home = Location(lat, lon)
    loc = home if location == "unset" else location
    ret = home if return_location == "unset" else return_location
    return Agent(
        location=loc,
        capabilities=[Capability(name=name) for name in capabilities],
        budget=budget,
        move_cost_function=flat_cost,
        return_location=ret,
        agent_id=agent_id,
    )


def mk_task(
    utility: float,
    *,
    lat: float = 1.0,
    lon: float = 0.0,
    capabilities: Sequence[str] = ("attack",),
    probability: float = 1.0,
    location: Optional[Location] = "unset",  # type: ignore[assignment]
    n_steps: int = 1,
) -> Task:
    """A single-ATTACK-step task, matching ``scenario_factory.make_attack_task``'s shape."""
    loc = Location(lat, lon) if location == "unset" else location
    steps = [
        Step(
            location=loc,
            target_id=f"target_{lat}_{lon}_{index}",
            capabilities=[Capability(name=name) for name in capabilities],
            probability=probability,
            effort=2,
            step_kind=StepKind.ATTACK,
        )
        for index in range(n_steps)
    ]
    return Task(steps=steps, utility=utility)


def point(
    model: MatchAouP1MILP, assignments: Dict[int, Sequence[int]]
) -> np.ndarray:
    """Build a full variable vector for a hand-specified assignment set.

    ``assignments`` maps agent index -> task indices assigned. ``y`` is filled in by the
    OR semantics the formulation is supposed to enforce, so feeding this point back
    through the constraint rows tests exactly that.
    """
    vector = np.zeros(model.n_variables, dtype=float)
    covered = set()
    for agent_index, task_indices in assignments.items():
        for task_index in task_indices:
            vector[model._x_index(agent_index, task_index)] = 1.0
            covered.add(task_index)
    for task_index in covered:
        vector[model._y_index(task_index)] = 1.0
    return vector


def satisfies(model: MatchAouP1MILP, vector: np.ndarray) -> bool:
    """True when ``vector`` satisfies every constraint row AND every variable bound."""
    lhs = model.constraint_matrix @ vector
    within_rows = bool(
        np.all(lhs <= model.constraint_upper + 1e-9)
        and np.all(lhs >= model.constraint_lower - 1e-9)
    )
    within_bounds = bool(
        np.all(vector >= model.bounds.lb - 1e-9)
        and np.all(vector <= model.bounds.ub + 1e-9)
    )
    return within_rows and within_bounds


def maximization_value(model: MatchAouP1MILP, vector: np.ndarray) -> float:
    """Objective in MAXIMIZATION sense (the solver minimizes the negated vector)."""
    return float(-(model.c @ vector))


# =============================================================================
# PO1 -- specialized-domain mathematical correctness
# =============================================================================

def test_po1_uncovered_task_is_exactly_zero_and_covered_is_exactly_full_utility() -> None:
    """The two halves of the deterministic p=1 rule, as exact float equalities."""
    model = MatchAouP1MILP([mk_agent("A")], [mk_task(80.0, lat=1.0)])

    empty = point(model, {})
    assert maximization_value(model, empty) == 0.0, "uncovered task must contribute 0"

    covered = point(model, {0: [0]})
    assert maximization_value(model, covered) == 80.0, "covered task must contribute full utility"

    solution, results, unselected = model.solve()
    assert results.solver.termination_condition == TERMINATION_OPTIMAL
    assert solution == {"A": [(0, 0)]}
    assert unselected == []
    # EXACT equality, not approximate: this is the whole point of dropping EPSILON.
    assert results.objective_value == 80.0


def test_po1_objective_vector_is_zero_on_x_and_negated_utility_on_y() -> None:
    """An assignment is worth nothing by itself; only the coverage it creates is."""
    utilities = [80.0, 100.0, 95.0]
    model = MatchAouP1MILP(
        [mk_agent("A"), mk_agent("B")],
        [mk_task(u, lat=index + 1.0) for index, u in enumerate(utilities)],
    )
    for i in range(model.n_agents):
        for j in range(model.n_tasks):
            assert model.c[model._x_index(i, j)] == 0.0, "x must carry no objective weight"
    for j, utility in enumerate(utilities):
        assert model.c[model._y_index(j)] == -utility, "y carries the negated utility"


def test_po1_no_epsilon_participates_in_the_new_objective() -> None:
    """Structurally absent from the module, and numerically absent from the value.

    The legacy objective scores a covered task at ``utility * (1 - EPSILON**m)``. This
    asserts the new value is EXACTLY ``utility`` -- i.e. it differs from the legacy
    expression by exactly the EPSILON residue, and carries none of it.
    """
    # Strip comments AND string literals via `tokenize`, so the module's own prose
    # explaining why EPSILON is gone cannot be mistaken for a use of it.
    import io
    import tokenize

    source = Path(p1_module.__file__).read_text(encoding="utf-8")
    executable_tokens = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
    ]
    code = " ".join(executable_tokens)
    assert "EPSILON" not in code, "the P1 formulation must not reference EPSILON"
    assert "1e-6" not in code, "the P1 formulation must not carry the EPSILON literal"
    # Positive control: the frozen legacy module DOES carry it, so this really discriminates.
    legacy_source = Path(
        REPO_ROOT / "src/match_aou/solvers/match_aou_MINLP_solver.py"
    ).read_text(encoding="utf-8")
    legacy_code = " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(legacy_source).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING)
    )
    assert "EPSILON" in legacy_code, "sanity: the legacy solver really does use EPSILON"

    model = MatchAouP1MILP([mk_agent("A")], [mk_task(80.0, lat=1.0)])
    _solution, results, _unselected = model.solve()

    legacy_scored = 80.0 * (1.0 - (0.0 + LEGACY_EPSILON) ** 1)
    assert results.objective_value == 80.0
    assert legacy_scored != 80.0, "sanity: the legacy expression really does carry a residue"


def test_po1_module_does_not_overclaim_objective_equivalence() -> None:
    """Pin the corrected claim so the retracted overclaim cannot silently return.

    An earlier revision of this module asserted that the P1 and legacy objectives "agree
    on which allocations are optimal in this domain". The benchmark contradicted that:
    at p = 1 the legacy objective pays a strictly positive marginal reward for a
    REDUNDANT agent, and the measured legacy solver stacks where the P1 MILP does not.
    Prose is the only place that claim can live, so prose is what this test guards.
    """
    doc = p1_module.__doc__ or ""
    lowered = doc.lower()

    assert "agree on which allocations are optimal" not in lowered, (
        "the retracted objective-equivalence overclaim is back in the module docstring"
    )

    # The narrower supported claim, and the explicit denial of the broader one.
    assert "covered-task set" in lowered
    assert "not** generally have the same optimal allocation set" in lowered or (
        "same optimal allocation set" in lowered and "not" in lowered
    ), "the docstring must deny a shared optimal ALLOCATION set"
    assert "intentionally changes the allocation objective" in lowered, (
        "the docstring must state that removing EPSILON changes the allocation objective"
    )

    # And the pre-existing statement must NOT have been weakened away.
    assert "not a claim of identical arithmetic" in lowered, (
        "the explicit 'not identical arithmetic' statement must be preserved"
    )


def test_po1_multiple_agents_on_one_task_remain_feasible() -> None:
    """Stacking is permitted. There must be no `sum_i x[i,j] <= 1` anywhere."""
    agents = [mk_agent("A"), mk_agent("B"), mk_agent("C")]
    model = MatchAouP1MILP(agents, [mk_task(80.0, lat=1.0)])

    for stack in ([0], [0, 1], [0, 1, 2]):
        vector = point(model, {index: [0] for index in stack})
        assert satisfies(model, vector), f"{len(stack)} agent(s) on one task must be feasible"


def test_po1_extra_agents_give_exactly_zero_marginal_objective_gain() -> None:
    """Agents 2/3/4 on a covered task add exactly nothing -- hand-evaluated."""
    agents = [mk_agent(name) for name in ("A", "B", "C", "D")]
    model = MatchAouP1MILP(agents, [mk_task(80.0, lat=1.0)])

    values = [
        maximization_value(model, point(model, {index: [0] for index in range(1, count + 1)}))
        for count in range(1, 5)
    ]
    assert values == [80.0, 80.0, 80.0, 80.0], f"marginal gain must be exactly 0, got {values}"


def test_po1_coverage_linking_pins_y_from_both_sides() -> None:
    """`x <= y` forces coverage ON; `y <= sum_i x` forces it OFF. Both are required."""
    model = MatchAouP1MILP([mk_agent("A")], [mk_task(80.0, lat=1.0)])

    # Claiming coverage with no assignment must be INFEASIBLE (else free utility).
    free_utility = np.zeros(model.n_variables, dtype=float)
    free_utility[model._y_index(0)] = 1.0
    assert not satisfies(model, free_utility), "y=1 with no assignment must be infeasible"

    # Assigning while denying coverage must be INFEASIBLE (else y is not pinned up).
    denied = np.zeros(model.n_variables, dtype=float)
    denied[model._x_index(0, 0)] = 1.0
    assert not satisfies(model, denied), "x=1 with y=0 must be infeasible"

    assert satisfies(model, point(model, {0: [0]}))
    assert satisfies(model, point(model, {}))


def test_po1_capability_rule_matches_the_legacy_predicate() -> None:
    """Blocked pairs are exactly those the frozen solver's predicate rejects."""
    agents = [
        mk_agent("attacker", capabilities=("attack",)),
        mk_agent("scout", capabilities=("recon",)),
        mk_agent("multi", capabilities=("attack", "recon")),
    ]
    tasks = [
        mk_task(80.0, lat=1.0, capabilities=("attack",)),
        mk_task(90.0, lat=2.0, capabilities=("recon",)),
        mk_task(70.0, lat=3.0, capabilities=("attack", "recon")),
    ]
    model = MatchAouP1MILP(agents, tasks)

    expected_blocked = {
        (i, j)
        for i, agent in enumerate(agents)
        for j, task in enumerate(tasks)
        # Literally the frozen solver's capability_constraint predicate.
        if not agent.has_capabilities(task.steps[0].capabilities)
    }
    assert set(model.capability_blocked) == expected_blocked
    assert expected_blocked, "sanity: the fixture must actually block something"

    # Blocking is enforced by an upper bound of 0, not by a constraint row.
    for i, j in expected_blocked:
        assert model.bounds.ub[model._x_index(i, j)] == 0.0
    for i in range(model.n_agents):
        for j in range(model.n_tasks):
            if (i, j) not in expected_blocked:
                assert model.bounds.ub[model._x_index(i, j)] == 1.0

    # An incapable agent can never be assigned, even when it would raise utility.
    solution, _results, _unselected = model.solve()
    for agent_id, assignments in (solution or {}).items():
        index = [a.id for a in agents].index(agent_id)
        for task_index, _step in assignments:
            assert (index, task_index) not in expected_blocked


def test_po1_movement_budget_row_matches_the_legacy_round_trip_rule() -> None:
    """Coefficients are `round_trip_cost`; RHS is `budget * (1 - risk_factor)`."""
    agent = mk_agent("A", lat=0.0, lon=0.0, budget=10.0)
    tasks = [mk_task(80.0, lat=1.0), mk_task(90.0, lat=2.5)]
    model = MatchAouP1MILP([agent], tasks, risk_factor=0.25)

    row_index = model.budget_rows_by_agent[0]
    row = model.constraint_matrix[row_index]
    for j, task in enumerate(tasks):
        expected = round_trip_cost(agent, task.steps[0].location)
        assert row[model._x_index(0, j)] == pytest.approx(expected)
    # Hand-check the geometry itself: out-and-back over 1.0 and 2.5 degrees.
    assert row[model._x_index(0, 0)] == pytest.approx(2.0)
    assert row[model._x_index(0, 1)] == pytest.approx(5.0)
    assert model.constraint_upper[row_index] == pytest.approx(10.0 * (1 - 0.25))

    # y columns are never charged movement.
    for j in range(model.n_tasks):
        assert row[model._y_index(j)] == 0.0


def test_po1_budget_actually_binds_and_prefers_the_affordable_optimum() -> None:
    """A round trip the agent cannot afford is excluded, not merely penalised."""
    # Round trip to lat=2.0 costs 4.0; budget 3.0 forbids it. lat=1.0 costs 2.0.
    agent = mk_agent("A", budget=3.0)
    tasks = [mk_task(80.0, lat=1.0), mk_task(1_000.0, lat=2.0)]
    solution, results, unselected = MatchAouP1MILP([agent], tasks).solve()

    assert results.solver.termination_condition == TERMINATION_OPTIMAL
    assert solution == {"A": [(0, 0)]}, "the unaffordable high-utility task must be excluded"
    assert unselected == [1]
    assert results.objective_value == 80.0


def test_po1_missing_locations_follow_the_legacy_skip_semantics() -> None:
    """Agent without a location: NO budget row. Step without one: no cost term."""
    located = mk_agent("located", budget=10.0)
    unlocated = mk_agent("unlocated", budget=10.0, location=None, return_location=None)
    tasks = [mk_task(80.0, lat=1.0), mk_task(90.0, location=None)]
    model = MatchAouP1MILP([located, unlocated], tasks)

    # Legacy `Constraint.Skip` for a location-less agent.
    assert 1 not in model.budget_rows_by_agent
    assert 0 in model.budget_rows_by_agent

    # Legacy `continue` for a location-less step: zero coefficient, not a dropped row.
    row = model.constraint_matrix[model.budget_rows_by_agent[0]]
    assert row[model._x_index(0, 1)] == 0.0
    assert row[model._x_index(0, 0)] == pytest.approx(2.0)


def test_po1_input_validation_rejects_everything_outside_the_contract() -> None:
    """Multi-step, p != 1, precedence and degenerate inputs are refused, not coerced."""
    agent = mk_agent("A")
    ok_task = mk_task(80.0, lat=1.0)

    with pytest.raises(P1MilpUnsupportedInputError, match="one step per task"):
        MatchAouP1MILP([agent], [mk_task(80.0, lat=1.0, n_steps=2)])

    with pytest.raises(P1MilpUnsupportedInputError, match="deterministic p = 1"):
        MatchAouP1MILP([agent], [mk_task(80.0, lat=1.0, probability=0.7)])

    with pytest.raises(P1MilpUnsupportedInputError, match="no precedence"):
        MatchAouP1MILP([agent], [ok_task, mk_task(90.0, lat=2.0)], precedence_relations=[(0, 1)])

    task_with_precedence = mk_task(80.0, lat=1.0)
    task_with_precedence.precedence_relations = [(0, 1)]
    with pytest.raises(P1MilpUnsupportedInputError, match="no precedence"):
        MatchAouP1MILP([agent], [task_with_precedence])

    with pytest.raises(P1MilpUnsupportedInputError, match="no tasks"):
        MatchAouP1MILP([agent], [])
    with pytest.raises(P1MilpUnsupportedInputError, match="no agents"):
        MatchAouP1MILP([], [ok_task])

    # p = 1 expressed as an int is still p = 1.
    MatchAouP1MILP([agent], [mk_task(80.0, lat=1.0, probability=1)])


def test_po1_model_has_no_step_dimension_and_no_stacking_bound() -> None:
    """Variable count is exactly `A*T + T`, and no row caps assignments per task."""
    model = MatchAouP1MILP(
        [mk_agent("A"), mk_agent("B"), mk_agent("C")],
        [mk_task(80.0, lat=index + 1.0) for index in range(4)],
    )
    assert model.n_variables == 3 * 4 + 4, "no step dimension may appear in the variables"

    # Row block sizes: A*T linking-upper, T linking-lower, and one budget row per agent.
    assert model.n_constraint_rows == 3 * 4 + 4 + 3

    # A row that caps a task's assignments would have all-positive x coefficients on one
    # task column group and a positive upper bound. None may exist.
    for row_index in range(model.n_constraint_rows):
        row = model.constraint_matrix[row_index]
        for j in range(model.n_tasks):
            task_column_values = [row[model._x_index(i, j)] for i in range(model.n_agents)]
            caps_all_agents = all(value > 0 for value in task_column_values)
            if caps_all_agents and model.constraint_upper[row_index] < np.inf:
                # Legitimate only if it is a budget row (single agent, movement costs).
                assert row_index in model.budget_rows_by_agent.values(), (
                    f"row {row_index} looks like a per-task stacking bound"
                )

    # y is continuous: pinned by the links, so branching on it would be pure waste.
    for j in range(model.n_tasks):
        assert model.integrality[model._y_index(j)] == 0
    for i in range(model.n_agents):
        for j in range(model.n_tasks):
            assert model.integrality[model._x_index(i, j)] == 1


# =============================================================================
# PO2 -- legacy isolation and output compatibility
# =============================================================================

TASK_BASE_SHA = "fd0d668d5031adef1f3b6af612e584f9ab56454b"
FROZEN_SOLVER_RELPATH = "src/match_aou/solvers/match_aou_MINLP_solver.py"
#: The approved P1 FORMULATION, pinned like the frozen legacy solver: the integration
#: commit consumes it and must not re-tune it. `P1_APPROVED_SHA` is the GPT-approved
#: isolated-solver candidate this branch's integration commit was built on top of.
P1_SOLVER_RELPATH = "src/match_aou/solvers/match_aou_p1_milp_solver.py"
P1_APPROVED_SHA = "1462163277322a3ef29eec28c782766edb8ea73b"

#: Files this branch is allowed to differ from `main` in.
#:
#: THE BRANCH IS TWO STAGES, and the guard tracks both rather than being deleted when the
#: second arrives. Stage 1 added the ISOLATED solver, its tests and its benchmark tool --
#: at that point the guard's whole content was "no active runtime file was touched", which
#: was the correct claim for an opt-in module nothing imported. Stage 2 is the reviewed
#: INTEGRATION, which necessarily touches the runtime: a backend cannot be selectable
#: without a selector, a solve seam, a reward valuation and a harness field.
#:
#: So the claim it now enforces is the one that is still checkable and still worth
#: enforcing: EXACTLY the declared integration surface changed, and nothing else did. The
#: two solver formulations are pinned separately and byte-for-byte by the two tests below,
#: which is the guarantee that actually matters -- neither approved objective was edited
#: to make the integration fit.
_STAGE1_ISOLATED_SOLVER_FILES = {
    "src/match_aou/solvers/match_aou_p1_milp_solver.py",
    "tests/test_match_aou_p1_milp_solver.py",
    "tools/benchmark_match_aou_p1_milp.py",
}
_STAGE2_INTEGRATION_FILES = {
    # the backend contract: ids, validation, the lazy P1 loader
    "src/match_aou/solvers/match_aou_backend.py",
    # the ONE solve seam + the stored per-episode backend
    "src/match_aou/rl/training/graph_episode_setup.py",
    # backend-aware reference valuation (`plan_value`)
    "src/match_aou/rl/training/graph_reward.py",
    # harness field, CLI/preset surface, provenance, abort routing
    "src/match_aou/rl/training/graph_train.py",
    "src/match_aou/rl/training/graph_rollout.py",
    # the population selector must use the SAME backend the run will evaluate under
    "src/match_aou/rl/training/graph_benchmark_preflight.py",
    # tests
    "tests/test_graph_setup_seam.py",
    "tests/test_match_aou_backend_integration.py",
}
ALLOWED_CHANGED_FILES = _STAGE1_ISOLATED_SOLVER_FILES | _STAGE2_INTEGRATION_FILES


def _git(*args: str) -> Optional[str]:
    """Run a git command in the repo, or return None when git is unusable here."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def test_po2_frozen_legacy_solver_is_byte_for_byte_unchanged() -> None:
    """The frozen MINLP file must hash identically to its blob at the task base."""
    base_blob = _git("rev-parse", f"{TASK_BASE_SHA}:{FROZEN_SOLVER_RELPATH}")
    if base_blob is None:
        pytest.skip("git unavailable or base commit not present in this checkout")

    working_blob = _git("hash-object", FROZEN_SOLVER_RELPATH)
    assert working_blob == base_blob, (
        "the frozen MATCH-AOU MINLP solver was modified "
        f"(base blob {base_blob}, working blob {working_blob})"
    )


def test_po2_approved_p1_formulation_is_byte_for_byte_unchanged() -> None:
    """The APPROVED P1 solver must hash identically to its reviewed blob.

    The integration CONSUMES this formulation; it does not get to adjust it. Changing the
    objective to make an integration fit would silently re-open a reviewed decision, so it
    is pinned exactly as the frozen legacy solver is -- and a genuine defect found here is
    a STOP, not an edit.
    """
    approved_blob = _git("rev-parse", f"{P1_APPROVED_SHA}:{P1_SOLVER_RELPATH}")
    if approved_blob is None:
        pytest.skip("git unavailable or approved commit not present in this checkout")

    working_blob = _git("hash-object", P1_SOLVER_RELPATH)
    assert working_blob == approved_blob, (
        "the approved P1 MILP formulation was modified "
        f"(approved blob {approved_blob}, working blob {working_blob})"
    )


def test_po2_only_the_declared_surface_was_modified() -> None:
    """Only the declared stage-1 + stage-2 files may differ from the task base."""
    changed = _git("diff", "--name-only", TASK_BASE_SHA)
    if changed is None:
        pytest.skip("git unavailable or base commit not present in this checkout")

    names = {line.strip() for line in changed.splitlines() if line.strip()}
    unexpected = names - ALLOWED_CHANGED_FILES
    assert not unexpected, f"files outside the allowed set were modified: {sorted(unexpected)}"


def test_po2_the_new_solver_is_not_exported_from_the_package() -> None:
    """Opt-in by explicit module import: importing the package must not reach it."""
    import match_aou.solvers as solvers_package

    assert set(solvers_package.__all__) == {"MatchAou", "round_trip_cost"}
    assert not hasattr(solvers_package, "MatchAouP1MILP"), "must not be re-exported"

    # `hasattr(package, "match_aou_p1_milp_solver")` is TRUE in this process purely
    # because this test module imported the submodule, and Python binds a submodule onto
    # its parent package. The real claim -- that importing the package does not REACH the
    # new module -- can only be made in a fresh interpreter.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import match_aou.solvers; "
            "print('match_aou.solvers.match_aou_p1_milp_solver' in sys.modules)",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert probe.returncode == 0, f"probe failed: {probe.stderr}"
    assert probe.stdout.strip() == "False", (
        "importing match_aou.solvers must not load the opt-in P1 module "
        f"(probe said {probe.stdout.strip()!r})"
    )


def test_po2_solution_shape_uses_task_index_and_step_zero() -> None:
    """`{agent_id: [(task_idx, 0), ...]}`, keyed by agent id, step index literally 0."""
    agents = [mk_agent("alpha"), mk_agent("beta")]
    tasks = [mk_task(80.0, lat=1.0), mk_task(90.0, lat=2.0)]
    solution, _results, _unselected = MatchAouP1MILP(agents, tasks).solve()

    assert solution is not None
    assert set(solution) <= {"alpha", "beta"}
    for agent_id, assignments in solution.items():
        assert isinstance(agent_id, str)
        for entry in assignments:
            assert isinstance(entry, tuple) and len(entry) == 2
            task_index, step_index = entry
            assert step_index == 0, "one step per task means the step index is always 0"
            assert isinstance(task_index, int) and 0 <= task_index < len(tasks)


def test_po2_unselected_is_exactly_the_zero_assignment_tasks() -> None:
    """Derived from assignment counts, and consistent with the returned solution."""
    # Only one task is reachable within budget; the other two are not.
    agent = mk_agent("A", budget=3.0)
    tasks = [mk_task(80.0, lat=1.0), mk_task(90.0, lat=9.0), mk_task(70.0, lat=8.0)]
    solution, _results, unselected = MatchAouP1MILP([agent], tasks).solve()

    assigned = {task_index for entries in (solution or {}).values() for task_index, _ in entries}
    assert unselected == sorted(set(range(len(tasks))) - assigned)
    assert unselected == [1, 2]

    # And the all-covered case leaves it empty.
    _s, _r, none_unselected = MatchAouP1MILP(
        [mk_agent("A"), mk_agent("B")], [mk_task(80.0, lat=1.0), mk_task(90.0, lat=2.0)]
    ).solve()
    assert none_unselected == []


def test_po2_results_expose_a_readable_termination_condition() -> None:
    """The real audit reader must be able to name this solver's termination."""
    from match_aou.rl.training.graph_episode_setup import _termination_name

    _solution, results, _unselected = MatchAouP1MILP(
        [mk_agent("A")], [mk_task(80.0, lat=1.0)]
    ).solve()

    assert results.solver.termination_condition == TERMINATION_OPTIMAL
    assert _termination_name(results) == "optimal"


class _FakeMilpResult:
    """Stand-in for a `scipy.optimize.milp` return value."""

    def __init__(self, status: int, x: Optional[Sequence[float]], message: str = "stub") -> None:
        self.status = status
        self.x = None if x is None else np.asarray(x, dtype=float)
        self.message = message
        self.fun = None


@pytest.mark.parametrize(
    ("status", "expected_condition"),
    [(1, "maxTimeLimit"), (2, "infeasible"), (3, "unbounded"), (4, "error"), (99, "unknown")],
)
def test_po2_non_optimal_outcomes_are_never_an_optimal_empty_allocation(
    monkeypatch: pytest.MonkeyPatch, status: int, expected_condition: str
) -> None:
    """A failed question must return None -- never a confident empty answer.

    Status 1 is the dangerous one: HiGHS can return a FEASIBLE incumbent alongside it.
    The stub therefore supplies a full, plausible solution vector, so a permissive
    implementation would happily report assignments that were never proven optimal.
    """
    model = MatchAouP1MILP([mk_agent("A")], [mk_task(80.0, lat=1.0)])
    feasible_looking = point(model, {0: [0]})

    monkeypatch.setattr(
        p1_module,
        "milp",
        lambda **_kwargs: _FakeMilpResult(status, feasible_looking),
    )
    solution, results, unselected = model.solve()

    assert solution is None, "a non-optimal termination must not yield assignments"
    assert unselected == [], "the legacy failure branch returns an empty unselected list"
    assert results.objective_value is None, "no objective may be claimed for a failed solve"
    assert results.solver.termination_condition == expected_condition
    assert results.solver.termination_condition != TERMINATION_OPTIMAL


def test_po2_optimal_status_without_a_solution_vector_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`status == 0` but `x is None` is a malformed result, not an empty allocation."""
    model = MatchAouP1MILP([mk_agent("A")], [mk_task(80.0, lat=1.0)])
    monkeypatch.setattr(p1_module, "milp", lambda **_kwargs: _FakeMilpResult(0, None))

    solution, results, unselected = model.solve()
    assert solution is None
    assert unselected == []
    assert results.objective_value is None


def test_po2_zero_mip_gap_is_requested_by_default() -> None:
    """A covered-set comparison needs a proven optimum, so the default gap is exactly 0."""
    captured: Dict[str, Any] = {}
    model = MatchAouP1MILP([mk_agent("A")], [mk_task(80.0, lat=1.0)])
    assert model.mip_rel_gap == 0.0

    real_milp = p1_module.milp

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_milp(**kwargs)

    p1_module.milp = _capture  # type: ignore[assignment]
    try:
        _solution, results, _unselected = model.solve()
    finally:
        p1_module.milp = real_milp  # type: ignore[assignment]

    assert captured["options"]["mip_rel_gap"] == 0.0
    assert results.mip_rel_gap == 0.0
    # HiGHS must have accepted the option rather than erroring on it.
    assert results.solver.termination_condition == TERMINATION_OPTIMAL


# =============================================================================
# PO3 -- same-input comparison against the frozen BONMIN MINLP
# =============================================================================

def _benchmark_module() -> Any:
    """Import the benchmark tool (it lives in `tools/`, not on the default path)."""
    tools_dir = REPO_ROOT / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import benchmark_match_aou_p1_milp  # noqa: PLC0415

    return benchmark_match_aou_p1_milp


def test_po3_evidence_labels_never_self_certify_a_validated_environment() -> None:
    """The benchmark records environment FACTS; it must never grade its own run.

    An earlier revision emitted ``validated_same_environment`` whenever
    ``--bonmin-executable`` was absent. That was too strong: bonmin being on PATH proves
    only that both solver arms can run from one process. It proves nothing about whether
    the interpreter is one of the repository's validated execution contexts, nor -- on
    the cluster -- that ``PYTHONNOUSERSITE=1`` was set. Grade-A validation is an
    orchestrator decision over the recorded facts, so no label may assert it.
    """
    tool = _benchmark_module()

    assert tool.classify_evidence(bonmin_executable_overridden=True) == (
        "diagnostic_cross_environment"
    )
    assert tool.classify_evidence(bonmin_executable_overridden=False) == (
        "same_environment_unverified"
    )

    # The closed set, and the anti-self-certification property that motivates it.
    assert set(tool.EVIDENCE_CLASSES) == {
        "diagnostic_cross_environment",
        "same_environment_unverified",
    }
    for label in tool.EVIDENCE_CLASSES:
        assert "validated" not in label.lower(), (
            f"evidence label {label!r} asserts validation the tool cannot establish"
        )

    # The environment facts an orchestrator judges must all still be recorded.
    for fact in ("python", "scipy", "pyomo", "bonmin", "CONDA_DEFAULT_ENV", "PYTHONNOUSERSITE"):
        assert fact.lower() in (tool.__doc__ or "").lower(), (
            f"the tool must document that it records {fact}"
        )


def _bonmin_executable() -> Optional[str]:
    """`bonmin` on PATH, else `MATCH_AOU_BONMIN_EXECUTABLE`, else None."""
    found = shutil.which("bonmin")
    if found:
        return found
    configured = os.environ.get("MATCH_AOU_BONMIN_EXECUTABLE")
    if configured and Path(configured).exists():
        return configured
    return None


requires_bonmin = pytest.mark.skipif(
    _bonmin_executable() is None,
    reason="no bonmin executable on PATH or in MATCH_AOU_BONMIN_EXECUTABLE",
)


def _solve_legacy(
    agents: Sequence[Agent], tasks: Sequence[Task]
) -> Tuple[Optional[Dict[Any, List[Tuple[int, int]]]], List[int]]:
    """Solve the SAME objects with the frozen MINLP under bonmin."""
    from pyomo.environ import SolverFactory

    from match_aou.solvers.match_aou_MINLP_solver import MatchAou

    executable = _bonmin_executable()
    model = MatchAou(agents=list(agents), tasks=list(tasks), precedence_relations=[], risk_factor=0.0)

    original_factory = SolverFactory

    def _factory(name: str, **kwargs: Any) -> Any:
        return original_factory(name, executable=executable, **kwargs)

    import match_aou.solvers.match_aou_MINLP_solver as legacy_module

    legacy_module.SolverFactory = _factory  # type: ignore[assignment]
    try:
        solution, _results, unselected = model.solve(solver_name="bonmin")
    finally:
        legacy_module.SolverFactory = original_factory  # type: ignore[assignment]
    return solution, unselected


def _covered_utility(
    tasks: Sequence[Task], solution: Optional[Dict[Any, List[Tuple[int, int]]]]
) -> float:
    """Exact-P1 covered utility of an allocation, for either solver's output."""
    covered = {task_index for entries in (solution or {}).values() for task_index, _ in entries}
    return sum(float(tasks[index].utility) for index in covered)


@requires_bonmin
@pytest.mark.parametrize("n_agents,n_tasks", [(2, 2), (2, 3), (3, 3), (3, 4)])
def test_po3_same_inputs_reach_the_same_covered_utility(n_agents: int, n_tasks: int) -> None:
    """Legacy and P1 must agree on covered utility and on the covered task set.

    Raw assignments may legitimately differ, and on these inputs the difference is
    EXPECTED and SYSTEMATIC rather than incidental tie-breaking: at ``p = 1`` the legacy
    objective still pays ``utility * (EPSILON - EPSILON**2)`` for a redundant agent, so
    stacking is optimal for THAT objective and not for this one. What must NOT differ is
    the covered task set and therefore the exact-P1 covered utility.

    This asserts agreement on the covered set / covered utility ONLY. It deliberately
    does not assert a shared optimal ALLOCATION set, because the two objectives do not
    have one.
    """
    agents = [mk_agent(f"agent{i}", budget=500.0) for i in range(n_agents)]
    tasks = [mk_task(80.0 + 5 * j, lat=j + 1.0) for j in range(n_tasks)]

    legacy_solution, _legacy_unselected = _solve_legacy(agents, tasks)
    p1_solution, p1_results, _p1_unselected = MatchAouP1MILP(agents, tasks).solve()

    assert legacy_solution is not None, "legacy solve did not reach acceptable optimality"
    assert p1_solution is not None
    assert p1_results.solver.termination_condition == TERMINATION_OPTIMAL

    legacy_utility = _covered_utility(tasks, legacy_solution)
    p1_utility = _covered_utility(tasks, p1_solution)
    assert p1_utility == pytest.approx(legacy_utility), (
        f"covered utility differs: legacy {legacy_utility} vs P1 {p1_utility}"
    )
    assert p1_results.objective_value == pytest.approx(p1_utility)


@requires_bonmin
def test_po3_p1_respects_the_same_capability_and_fuel_feasibility_as_legacy() -> None:
    """Every P1 assignment is capability-legal and inside the agent's fuel budget."""
    agents = [
        mk_agent("attacker", budget=12.0, capabilities=("attack",)),
        mk_agent("scout", budget=500.0, capabilities=("recon",)),
    ]
    tasks = [
        mk_task(80.0, lat=1.0, capabilities=("attack",)),
        mk_task(90.0, lat=2.0, capabilities=("recon",)),
        mk_task(120.0, lat=40.0, capabilities=("attack",)),  # far: unaffordable
    ]
    p1_solution, _results, _unselected = MatchAouP1MILP(agents, tasks).solve()
    legacy_solution, _legacy_unselected = _solve_legacy(agents, tasks)
    assert legacy_solution is not None

    by_id = {agent.id: agent for agent in agents}
    for solution, label in ((p1_solution, "P1"), (legacy_solution, "legacy")):
        for agent_id, entries in (solution or {}).items():
            agent = by_id[agent_id]
            spent = 0.0
            for task_index, _step in entries:
                task = tasks[task_index]
                assert agent.has_capabilities(task.steps[0].capabilities), (
                    f"{label}: capability-illegal assignment {agent_id} -> {task_index}"
                )
                spent += round_trip_cost(agent, task.steps[0].location)
            assert spent <= agent.budget + 1e-6, f"{label}: {agent_id} overspent fuel"


# =============================================================================
# Degeneracy OBSERVATION -- reported, never repaired
# =============================================================================

def test_observe_redundant_stacking_when_agents_outnumber_tasks() -> None:
    """Record what HiGHS naturally does when redundancy is free.

    With p = 1 a second agent on a task has zero marginal utility, so both a stacked and
    a non-stacked allocation can be optimal. This test asserts only that the optimum is
    reached and covered utility is right; it deliberately does NOT require a
    non-redundant allocation, because enforcing that would need a tie-breaker the task
    forbids. The count is printed so the behaviour is observable.
    """
    agents = [mk_agent(f"agent{i}") for i in range(4)]
    tasks = [mk_task(80.0, lat=1.0), mk_task(90.0, lat=2.0)]
    solution, results, unselected = MatchAouP1MILP(agents, tasks).solve()

    assert results.solver.termination_condition == TERMINATION_OPTIMAL
    assert unselected == []
    assert results.objective_value == 170.0

    assigned_per_task = [0, 0]
    for entries in (solution or {}).values():
        for task_index, _step in entries:
            assigned_per_task[task_index] += 1
    redundant = sum(max(0, count - 1) for count in assigned_per_task)
    print(
        f"[observation] 4 agents / 2 tasks -> per-task assignment counts "
        f"{assigned_per_task}, redundant assignments {redundant}"
    )
    assert sum(assigned_per_task) >= 2, "both tasks must be covered"


# =============================================================================
# Standalone runner (CLAUDE.md section 1: nlp_env has no pytest)
# =============================================================================

if __name__ == "__main__":
    import inspect
    import traceback

    class _MonkeyPatch:
        """Minimal `monkeypatch` stand-in for the pytest-free runner."""

        def __init__(self) -> None:
            self._undo: List[Tuple[Any, str, Any]] = []

        def setattr(self, target: Any, name: str, value: Any) -> None:
            self._undo.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self) -> None:
            for target, name, original in reversed(self._undo):
                setattr(target, name, original)
            self._undo.clear()

    def _wants_monkeypatch(function: Any) -> bool:
        try:
            return "monkeypatch" in inspect.signature(function).parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins only
            return False

    def _skip_reason(marks: Sequence[Any]) -> Optional[str]:
        """Honour `pytest.mark.skipif` so the bonmin tier degrades like it does in pytest."""
        for mark in marks:
            if mark.name != "skipif":
                continue
            condition = mark.args[0] if mark.args else False
            if condition:
                return str(mark.kwargs.get("reason", "skipif"))
        return None

    def _cases() -> List[Tuple[str, Any, Dict[str, Any], Optional[str]]]:
        collected: List[Tuple[str, Any, Dict[str, Any], Optional[str]]] = []
        for name, function in sorted(globals().items()):
            if not name.startswith("test_") or not callable(function):
                continue
            marks = list(getattr(function, "pytestmark", []))
            skip = _skip_reason(marks)
            params = [mark for mark in marks if mark.name == "parametrize"]
            if not params:
                collected.append((name, function, {}, skip))
                continue
            raw_argnames = params[0].args[0]
            if isinstance(raw_argnames, str):
                argnames = [a.strip() for a in raw_argnames.split(",") if a.strip()]
            else:
                argnames = [str(a).strip() for a in raw_argnames]
            for argvalues in params[0].args[1]:
                values = argvalues if isinstance(argvalues, tuple) else (argvalues,)
                kwargs = dict(zip(argnames, values))
                label = ",".join(f"{key}={value}" for key, value in kwargs.items())
                collected.append((f"{name}[{label}]", function, kwargs, skip))
        return collected

    failures = 0
    skipped = 0
    passed = 0
    for case_name, case_function, case_kwargs, case_skip in _cases():
        if case_skip:
            skipped += 1
            print(f"SKIP {case_name}: {case_skip}")
            continue
        patcher = _MonkeyPatch()
        call_kwargs = dict(case_kwargs)
        if _wants_monkeypatch(case_function):
            call_kwargs["monkeypatch"] = patcher
        try:
            case_function(**call_kwargs)
            passed += 1
            print(f"OK   {case_name}")
        # BaseException, not Exception: real pytest's `Skipped` derives from
        # BaseException, so `pytest.skip()` inside a test would otherwise escape this
        # handler and abort the whole standalone run instead of counting as a SKIP.
        # The shim's own `Skipped` is an ordinary Exception; both are handled here.
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise  # never swallow an interrupt or an explicit exit
            if type(exc).__name__ in {"Skipped", "OutcomeException"}:
                skipped += 1
                print(f"SKIP {case_name}: {exc}")
            else:
                failures += 1
                print(f"FAIL {case_name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
        finally:
            patcher.undo()

    print("")
    print(
        f"MATCH_AOU_P1_MILP TESTS: {passed} passed, {skipped} skipped, "
        f"{failures} failed"
    )
    if failures:
        sys.exit(1)
