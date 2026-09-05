"""MATCH-AOU allocation specialized to the deterministic ``p = 1`` domain, as a MILP.

WHY THIS EXISTS
---------------
Every task this project actually generates is a **single-ATTACK-step task with
``probability = 1.0``** and no precedence relations: see
``scenario_factory.make_attack_task`` (one ``Step``, ``probability: float = 1.0``) and
``scenario_factory.generate_all_enemy_tasks`` (same default), which
``graph_episode_setup._extract_world`` calls with an explicit ``probability=1.0``.

The frozen legacy solver (``match_aou_MINLP_solver.MatchAou``, ``CLAUDE.md`` section 2 —
``MATCH-AOU solver — FROZEN (advisor-approved form)``) is a general MINLP whose objective
carries the nonlinear per-step success expression

    prod_k [ 1 - (1 - p_k + EPSILON) ** sum_i x[i,j,k] ]

with ``EPSILON = 1e-6``. That expression has a **variable exponent** (the assignment
count), which is what makes the model nonlinear and forces a MINLP solver. ``EPSILON``
exists purely to keep that expression numerically evaluable around ``p = 1, m = 0``; it
is a numerical workaround, not a modelling intent.

At ``p = 1`` the whole expression collapses to a pure OR over assignments, so the
nonlinearity is unnecessary in this domain. This module therefore removes it outright
rather than carrying the workaround forward:

  * no assigned agent  -> task value exactly ``0``
  * one or more agents -> task value exactly ``task.utility``
  * agents 2, 3, 4 ... -> exactly ZERO marginal task utility, but still FEASIBLE

**This is deliberately NOT a claim of identical arithmetic with the legacy MINLP.** The
legacy objective evaluates a covered task at ``utility * (1 - EPSILON ** m)`` rather than
at ``utility``. The legacy formulation needs ``EPSILON``; this one has no variable
exponent and therefore does not.

**NOR is it a claim that the two objectives share an optimal allocation set.** Stating
exactly what is and is not supported:

  * In the exercised domain the two objectives CAN agree on the optimal COVERED-TASK SET
    and therefore on the exact-P1 covered utility. That is what the engineering
    comparison in ``tools/benchmark_match_aou_p1_milp.py`` observed on the cells it ran.
  * They do **NOT** generally have the same optimal ALLOCATION set, and the difference is
    systematic rather than incidental. At ``p = 1`` the legacy objective still pays a
    strictly positive ``utility * (EPSILON - EPSILON ** 2)`` for each REDUNDANT agent on
    an already-covered task, so stacking is genuinely optimal FOR THAT objective. The
    measured legacy solver stacks in exactly this way where the P1 MILP does not.
  * This formulation therefore **INTENTIONALLY CHANGES THE ALLOCATION OBJECTIVE** by
    removing that numerical stacking incentive. Removing ``EPSILON`` is not a
    presentation detail: it changes which allocations are optimal.

A consequence worth stating once, plainly: swapping this solver in changes ALLOCATIONS,
not merely runtime. That is why its integration was taken as a separate reviewed decision
rather than as a drop-in performance change -- see SCOPE below for the seam that resulted
and for what selecting it does to the rest of the pipeline.

THE FORMULATION
---------------
Decision variables:

  ``x[i, j] in {0, 1}``   agent ``i`` is assigned to task ``j``
  ``0 <= y[j] <= 1``      task ``j`` is covered  (integrality NOT required, see below)

Constraints:

  ``x[i, j] <= y[j]``        for every (i, j)   -- any assignment forces coverage up
  ``y[j] <= sum_i x[i, j]``                     -- no assignment forces coverage down

Together these are exact OR semantics, which is also why ``y`` needs no integrality: the
two links pin ``y[j]`` to ``0`` or ``1`` from both sides for any integral ``x``, so
declaring ``y`` binary would only enlarge the branch-and-bound tree for nothing.

Objective (maximize):  ``sum_j utility[j] * y[j]``

SciPy's ``milp`` minimizes, so the implementation loads the algebraically equivalent
negated linear objective and negates the reported optimum back.

There is deliberately **no step dimension** in the optimization variables, no nonlinear
term, no exponent, no ``EPSILON``, no precedence variable, no legacy ``y``/``x`` Big-M,
and no assignment-count table.

WHAT IS PRESERVED EXACTLY FROM THE FROZEN SOLVER
------------------------------------------------
* **Capability**: ``agent.has_capabilities(task.steps[0].capabilities)``. An incapable
  pairing is made impossible by pinning that variable's upper bound to ``0`` rather than
  by adding a constraint row -- the same effect as the legacy ``x[i,j,k] == 0``, at no
  cost to the constraint matrix.
* **Movement budget**: ``sum_j round_trip_cost(agent_i, step_loc_j) * x[i, j]
  <= agent_i.budget * (1 - risk_factor)``, importing ``round_trip_cost`` from the frozen
  module rather than re-deriving its geometry.
* **Missing-location handling**: an agent with ``location is None`` gets NO budget row at
  all (the legacy ``Constraint.Skip``); a step with ``location is None`` contributes no
  cost term (the legacy ``continue``).

DEGENERACY IS DELIBERATELY NOT BROKEN
-------------------------------------
With ``p = 1`` a redundant second agent on a task has exactly zero marginal utility, so
some optimal solutions may stack agents redundantly. This module adds **no**
``sum_i x[i,j] <= 1`` bound, **no** assignment or fuel penalty, **no** epsilon tie-breaker
and **no** lexicographic second objective. Whether HiGHS stacks redundantly is something
to OBSERVE first; inventing a tie-break here would silently change the allocation
semantics that the rest of the pipeline was measured against.

Observed so far, and recorded as an OBSERVATION rather than a guarantee: across the
engineering matrix run to date HiGHS returned no redundant assignment at all, so no
tie-breaker was needed. That was measured on LOCAL diagnostic evidence only (see the
benchmark tool's environment caveat) and it is not a proof that HiGHS never stacks.

SCOPE
-----
**THE FORMULATION ABOVE IS THE APPROVED DETERMINISTIC-P1 FORMULATION AND IS UNCHANGED.**
What follows describes only its RELATIONSHIP to the rest of the repository, which the
reviewed backend-selector integration changed.

  * **Still not exported from** ``match_aou.solvers.__init__``. Importing the package does
    not reach this module, and the package's import surface is exactly what it was.
  * **The runtime MAY now select this solver, but only EXPLICITLY.** The integration added
    a reviewed backend seam -- ``match_aou.solvers.match_aou_backend`` -- and
    ``graph_episode_setup.solve_and_normalize_audited`` routes to this formulation when,
    and only when, the caller asked for
    :data:`~match_aou.solvers.match_aou_backend.MATCH_AOU_BACKEND_P1_MILP_V1`. That seam
    imports this module LAZILY, so naming a backend still costs nothing.
  * **The legacy MINLP remains the DEFAULT.** A caller that says nothing gets the frozen
    ``MatchAou`` through BONMIN -- the objective every approved measurement was taken on.
  * **There is no** ``auto`` **and no fallback in either direction.** A refused P1 solve is
    never rescued by the legacy solver, and one episode never mixes backends.
  * **Selecting it is NOT a transparent performance swap.** Removing the ``EPSILON``
    stacking incentive changes which allocations are optimal; because ``A_init`` is what
    route-relative hidden placement predicts routes from, it can change the hidden
    geometry, episode feasibility and what a policy learns. **No equivalence with the
    legacy objective is claimed anywhere.**

``p < 1``, multi-step tasks, precedence support and any tie-breaking rule remain
explicitly OUT OF SCOPE for this formulation, exactly as before -- an input outside that
contract is REFUSED rather than answered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse the frozen geometry rather than reinterpreting it. The frozen module imports
# pyomo at module scope, but that is already unavoidable in this package: importing any
# `match_aou.*` module runs `match_aou/__init__.py`, which does
# `from .solvers import MatchAou` (recorded in CLAUDE.md section 8). So this import adds
# no dependency that the process did not already have.
from match_aou.solvers.match_aou_MINLP_solver import round_trip_cost

try:  # SciPy >= 1.9 provides `milp` (HiGHS). Absence is reported at construction time.
    from scipy.optimize import Bounds, LinearConstraint, milp

    _SCIPY_MILP_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _exc:  # pragma: no cover - exercised only on a SciPy-less install
    Bounds = LinearConstraint = milp = None  # type: ignore[assignment]
    _SCIPY_MILP_IMPORT_ERROR = _exc


__all__ = [
    "MatchAouP1MILP",
    "P1MilpResults",
    "P1MilpSolverSection",
    "TERMINATION_OPTIMAL",
    "P1MilpUnsupportedInputError",
    "P1MilpBackendUnavailableError",
]


# --------------------------------------------------------------------------------------
# Termination vocabulary
# --------------------------------------------------------------------------------------
# Spelled to match Pyomo's `TerminationCondition` value strings so that a downstream
# audit reads the same way whichever solver produced it. `graph_episode_setup.
# _termination_name` records `str(getattr(condition, "value", condition))`, and a plain
# `str` has no `.value`, so these constants pass through verbatim.
TERMINATION_OPTIMAL: str = "optimal"
TERMINATION_INFEASIBLE: str = "infeasible"
TERMINATION_UNBOUNDED: str = "unbounded"
TERMINATION_MAX_TIME_LIMIT: str = "maxTimeLimit"
TERMINATION_ERROR: str = "error"
TERMINATION_UNKNOWN: str = "unknown"

#: `scipy.optimize.milp` status code -> termination name. Status 1 ("iteration or time
#: limit reached") may still carry a FEASIBLE incumbent; it is deliberately mapped to a
#: non-accepted condition, because an unproven incumbent must never be reported as an
#: optimal allocation.
_STATUS_TO_TERMINATION: Dict[int, str] = {
    0: TERMINATION_OPTIMAL,
    1: TERMINATION_MAX_TIME_LIMIT,
    2: TERMINATION_INFEASIBLE,
    3: TERMINATION_UNBOUNDED,
    4: TERMINATION_ERROR,
}

#: The ONLY accepted outcome. The legacy solver additionally accepts `locallyOptimal`,
#: which is a nonlinear-solver concept: HiGHS solved to a zero relative gap either proves
#: global optimality or does not terminate optimally at all.
_ACCEPTED_TERMINATIONS = frozenset({TERMINATION_OPTIMAL})

#: Values below/above this are read as binary 0/1, matching the legacy `> 0.5` test.
_BINARY_THRESHOLD: float = 0.5


class P1MilpUnsupportedInputError(ValueError):
    """Input lies outside the deterministic single-step ``p = 1`` contract.

    Raised instead of silently coercing, because every one of these cases means the
    caller's problem is NOT the problem this formulation solves, and answering it anyway
    would return a confidently wrong allocation.
    """


class P1MilpBackendUnavailableError(RuntimeError):
    """``scipy.optimize.milp`` (SciPy >= 1.9) is not importable in this interpreter."""


@dataclass(frozen=True)
class P1MilpSolverSection:
    """The ``results.solver`` section, shaped for the existing audit reader.

    ``graph_episode_setup._termination_name`` reads exactly
    ``results.solver.termination_condition``; this type provides that attribute without
    building a Pyomo model or a Pyomo ``SolverResults`` object.
    """

    termination_condition: str
    status: int
    message: str


@dataclass(frozen=True)
class P1MilpResults:
    """Result envelope mirroring the parts of Pyomo's ``SolverResults`` that are read.

    Attributes:
        solver: the section carrying ``termination_condition``.
        objective_value: covered utility of the returned allocation in MAXIMIZATION
            sense (``None`` when no solution was accepted). This is
            ``sum_j utility[j] * y[j]`` -- exact-P1 covered utility, with no ``EPSILON``
            anywhere in it.
        build_seconds: wall-clock time spent constructing the model in ``__init__``.
        solve_seconds: wall-clock time spent inside ``scipy.optimize.milp``.
        mip_rel_gap: the relative gap actually requested of HiGHS.
        n_variables / n_constraint_rows: model size, for engineering comparison.
    """

    solver: P1MilpSolverSection
    objective_value: Optional[float]
    build_seconds: float
    solve_seconds: float
    mip_rel_gap: Optional[float]
    n_variables: int
    n_constraint_rows: int


class MatchAouP1MILP:
    """MATCH-AOU allocation for the deterministic single-step ``p = 1`` domain.

    Constructor shape deliberately mirrors
    :class:`~match_aou.solvers.match_aou_MINLP_solver.MatchAou` so the two are
    substitutable in a same-input comparison.

    The model is built in ``__init__`` (as the legacy class builds its Pyomo model
    there), so ``build_seconds`` and ``solve_seconds`` can be reported separately.
    """

    def __init__(
        self,
        agents: Sequence[Any],
        tasks: Sequence[Any],
        precedence_relations: Optional[Sequence[Tuple[int, int]]] = None,
        risk_factor: float = 0.0,
        *,
        mip_rel_gap: Optional[float] = 0.0,
    ) -> None:
        """Validate the inputs against the P1 contract and build the MILP.

        Args:
            agents: agent objects (need ``id`` / ``location`` / ``budget`` /
                ``has_capabilities`` / ``move_cost``).
            tasks: task objects, each with EXACTLY ONE step whose ``probability`` is
                exactly ``1.0``.
            precedence_relations: must be empty or ``None``. Precedence is out of scope
                for this formulation and is REJECTED rather than ignored.
            risk_factor: conservative budget margin, applied exactly as the frozen
                solver applies it.
            mip_rel_gap: relative MIP gap handed to HiGHS. Defaults to ``0.0`` so the
                returned allocation is a proven optimum -- which is what a
                covered-set / covered-utility comparison against the legacy solver
                requires. It does not make the two objectives equivalent.

        Raises:
            P1MilpBackendUnavailableError: ``scipy.optimize.milp`` is not importable.
            P1MilpUnsupportedInputError: an input violates the P1 contract.
        """
        if milp is None:  # pragma: no cover - depends on the installed SciPy
            raise P1MilpBackendUnavailableError(
                "MatchAouP1MILP requires `scipy.optimize.milp` (SciPy >= 1.9), which "
                f"could not be imported: {_SCIPY_MILP_IMPORT_ERROR!r}"
            )

        self.agents: List[Any] = list(agents)
        self.tasks: List[Any] = list(tasks)
        self.precedence_relations: List[Tuple[int, int]] = list(precedence_relations or [])
        self.risk_factor = risk_factor
        self.mip_rel_gap = mip_rel_gap

        self._validate_inputs()

        started = time.perf_counter()
        self._build_model()
        self.build_seconds = time.perf_counter() - started

    # ---------------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------------
    def _validate_inputs(self) -> None:
        """Reject anything outside the deterministic single-step ``p = 1`` contract."""
        # Degenerate inputs. The production caller
        # (`graph_episode_setup.solve_and_normalize_audited`) short-circuits both of
        # these BEFORE constructing a solver and records `SOLVE_NOT_ATTEMPTED`, so this
        # is unreachable on the live path. It raises rather than returning an empty
        # allocation precisely so "there was nobody to allocate" can never be recorded
        # as "solved, and the answer is allocate nothing".
        if not self.tasks:
            raise P1MilpUnsupportedInputError(
                "MatchAouP1MILP: no tasks. A degenerate solve is the caller's to "
                "short-circuit; it is refused here so it cannot be read as an "
                "answered empty allocation."
            )
        if not self.agents:
            raise P1MilpUnsupportedInputError(
                "MatchAouP1MILP: no agents. A degenerate solve is the caller's to "
                "short-circuit; it is refused here so it cannot be read as an "
                "answered empty allocation."
            )

        if self.precedence_relations:
            raise P1MilpUnsupportedInputError(
                "MatchAouP1MILP models no precedence: "
                f"{len(self.precedence_relations)} relation(s) supplied "
                f"({self.precedence_relations[:4]}...). Use the legacy MINLP solver for "
                "problems with precedence relations."
            )

        for j, task in enumerate(self.tasks):
            steps = getattr(task, "steps", None)
            if steps is None:
                raise P1MilpUnsupportedInputError(
                    f"MatchAouP1MILP: task {j} has no `steps`."
                )
            if len(steps) != 1:
                raise P1MilpUnsupportedInputError(
                    f"MatchAouP1MILP models exactly one step per task: task {j} has "
                    f"{len(steps)}. Use the legacy MINLP solver for multi-step tasks."
                )
            probability = steps[0].probability
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise P1MilpUnsupportedInputError(
                    f"MatchAouP1MILP: task {j} step probability is not a real number "
                    f"({probability!r})."
                )
            if float(probability) != 1.0:
                raise P1MilpUnsupportedInputError(
                    f"MatchAouP1MILP models deterministic p = 1 only: task {j} step "
                    f"probability is {probability!r}. Use the legacy MINLP solver for "
                    "p < 1."
                )
            # A task-level precedence list is the same statement as a problem-level one.
            task_precedence = getattr(task, "precedence_relations", None)
            if task_precedence:
                raise P1MilpUnsupportedInputError(
                    f"MatchAouP1MILP models no precedence: task {j} carries "
                    f"{task_precedence!r}."
                )

    # ---------------------------------------------------------------------------------
    # Model construction
    # ---------------------------------------------------------------------------------
    def _x_index(self, i: int, j: int) -> int:
        """Flat column index of ``x[i, j]`` (row-major over agents, then tasks)."""
        return i * self.n_tasks + j

    def _y_index(self, j: int) -> int:
        """Flat column index of ``y[j]``, which follows the whole ``x`` block."""
        return self.n_agents * self.n_tasks + j

    def _build_model(self) -> None:
        """Assemble objective, bounds, integrality and the constraint rows.

        The matrix is built dense on purpose: at the cardinalities this project actually
        runs (agents 2-4, targets up to ~8) it is a few hundred floats, and a dense
        matrix is far easier to audit against the formulation above than an assembled
        sparse triple. Should the cell grow by orders of magnitude, this is the one place
        to switch to `scipy.sparse`.
        """
        self.n_agents = len(self.agents)
        self.n_tasks = len(self.tasks)
        n_x = self.n_agents * self.n_tasks
        n_vars = n_x + self.n_tasks
        self.n_variables = n_vars

        # --- Objective: minimize -sum_j utility[j] * y[j]  ==  maximize covered utility.
        #     x carries NO objective coefficient: an assignment is worth nothing by
        #     itself, only the coverage it creates is.
        c = np.zeros(n_vars, dtype=float)
        for j, task in enumerate(self.tasks):
            c[self._y_index(j)] = -float(task.utility)
        self.c = c

        # --- Bounds. x is binary; capability-infeasible pairs are pinned to 0 here
        #     rather than via a constraint row (the legacy `x[i,j,k] == 0`).
        lower = np.zeros(n_vars, dtype=float)
        upper = np.ones(n_vars, dtype=float)
        self.capability_blocked: List[Tuple[int, int]] = []
        for i, agent in enumerate(self.agents):
            for j, task in enumerate(self.tasks):
                if not agent.has_capabilities(task.steps[0].capabilities):
                    upper[self._x_index(i, j)] = 0.0
                    self.capability_blocked.append((i, j))
        self.bounds = Bounds(lb=lower, ub=upper)

        # --- Integrality: x binary (1), y continuous (0). y is pinned to {0, 1} by the
        #     two linking constraints for any integral x, so branching on it is waste.
        integrality = np.zeros(n_vars, dtype=int)
        integrality[:n_x] = 1
        self.integrality = integrality

        rows: List[np.ndarray] = []
        row_lower: List[float] = []
        row_upper: List[float] = []

        # --- (1) x[i, j] - y[j] <= 0     : any assignment forces coverage ON.
        for i in range(self.n_agents):
            for j in range(self.n_tasks):
                row = np.zeros(n_vars, dtype=float)
                row[self._x_index(i, j)] = 1.0
                row[self._y_index(j)] = -1.0
                rows.append(row)
                row_lower.append(-np.inf)
                row_upper.append(0.0)

        # --- (2) y[j] - sum_i x[i, j] <= 0 : no assignment forces coverage OFF.
        for j in range(self.n_tasks):
            row = np.zeros(n_vars, dtype=float)
            row[self._y_index(j)] = 1.0
            for i in range(self.n_agents):
                row[self._x_index(i, j)] = -1.0
            rows.append(row)
            row_lower.append(-np.inf)
            row_upper.append(0.0)

        # --- (3) movement budget, identical in form and in missing-location handling to
        #     the frozen `movement_budget_constraint`.
        self.budget_rows_by_agent: Dict[int, int] = {}
        for i, agent in enumerate(self.agents):
            start_loc = getattr(agent, "location", None)
            if start_loc is None:
                # Legacy `Constraint.Skip`: this agent gets no budget row at all.
                continue
            row = np.zeros(n_vars, dtype=float)
            for j, task in enumerate(self.tasks):
                step_loc = getattr(task.steps[0], "location", None)
                if step_loc is None:
                    # Legacy `continue`: this step contributes no movement cost.
                    continue
                row[self._x_index(i, j)] = float(round_trip_cost(agent, step_loc))
            self.budget_rows_by_agent[i] = len(rows)
            rows.append(row)
            row_lower.append(-np.inf)
            row_upper.append(float(agent.budget) * (1.0 - self.risk_factor))

        self.n_constraint_rows = len(rows)
        if rows:
            matrix = np.vstack(rows)
            self.constraints = [LinearConstraint(matrix, row_lower, row_upper)]
            self.constraint_matrix = matrix
            self.constraint_lower = np.asarray(row_lower, dtype=float)
            self.constraint_upper = np.asarray(row_upper, dtype=float)
        else:  # pragma: no cover - unreachable: block (1) is non-empty for any input
            self.constraints = []
            self.constraint_matrix = np.zeros((0, n_vars), dtype=float)
            self.constraint_lower = np.zeros(0, dtype=float)
            self.constraint_upper = np.zeros(0, dtype=float)

    # ---------------------------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------------------------
    def solve(self) -> Tuple[Optional[Dict[Any, List[Tuple[int, int]]]], P1MilpResults, List[int]]:
        """Solve the MILP.

        Returns:
            ``(solution, results, unselected_tasks)`` -- the same high-level triple the
            frozen solver returns.

            ``solution`` maps ``agent_id -> [(task_idx, 0), ...]``. The step index is
            literally ``0`` because the contract is one step per task; it is emitted so
            downstream consumers (which expect ``(task_idx, step_idx)`` pairs) need no
            redesign. Agents with no assignment do not appear as keys, matching the
            legacy `setdefault` behaviour. ``None`` when the solve did not reach proven
            optimality.

            ``unselected_tasks`` is exactly the task indices receiving ZERO assignments.
        """
        options: Dict[str, Any] = {}
        if self.mip_rel_gap is not None:
            options["mip_rel_gap"] = float(self.mip_rel_gap)

        started = time.perf_counter()
        raw = milp(
            c=self.c,
            constraints=self.constraints,
            bounds=self.bounds,
            integrality=self.integrality,
            options=options or None,
        )
        solve_seconds = time.perf_counter() - started

        status = int(getattr(raw, "status", 4))
        termination = _STATUS_TO_TERMINATION.get(status, TERMINATION_UNKNOWN)
        message = str(getattr(raw, "message", ""))

        def _results(objective_value: Optional[float]) -> P1MilpResults:
            return P1MilpResults(
                solver=P1MilpSolverSection(
                    termination_condition=termination,
                    status=status,
                    message=message,
                ),
                objective_value=objective_value,
                build_seconds=self.build_seconds,
                solve_seconds=solve_seconds,
                mip_rel_gap=self.mip_rel_gap,
                n_variables=self.n_variables,
                n_constraint_rows=self.n_constraint_rows,
            )

        # A non-optimal termination is a FAILED question, never an empty answer. Note
        # that HiGHS can return a feasible incumbent alongside status 1; refusing it here
        # is what stops an unproven allocation being recorded as an optimal one.
        if termination not in _ACCEPTED_TERMINATIONS or getattr(raw, "x", None) is None:
            return None, _results(None), []

        values = np.asarray(raw.x, dtype=float)

        solution: Dict[Any, List[Tuple[int, int]]] = {}
        assigned_count = [0] * self.n_tasks
        # Agent-major then task-major, mirroring the frozen solver's extraction order so
        # the two produce comparably ordered assignment lists.
        for i, agent in enumerate(self.agents):
            for j in range(self.n_tasks):
                if values[self._x_index(i, j)] > _BINARY_THRESHOLD:
                    solution.setdefault(agent.id, []).append((j, 0))
                    assigned_count[j] += 1

        unselected_tasks = [j for j in range(self.n_tasks) if assigned_count[j] == 0]

        # Report covered utility from the ALLOCATION rather than from `raw.fun`, so the
        # number describes the assignments actually returned. They coincide at an optimum;
        # deriving it from the allocation means the reported value cannot drift from the
        # solution the caller receives.
        covered_utility = sum(
            float(self.tasks[j].utility)
            for j in range(self.n_tasks)
            if assigned_count[j] > 0
        )

        return solution, _results(covered_utility), unselected_tasks
