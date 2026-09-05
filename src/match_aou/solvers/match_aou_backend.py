"""WHICH MATCH-AOU allocation objective a run solves — the explicit backend selector.

WHY THIS MODULE EXISTS, AND WHY IT IS SEPARATE
----------------------------------------------
Two approved MATCH-AOU formulations now exist in this repository:

  * :data:`MATCH_AOU_BACKEND_LEGACY_MINLP_V1` -- the FROZEN general MINLP
    (``match_aou_MINLP_solver.MatchAou``, ``CLAUDE.md`` section 2), solved through
    BONMIN. Its objective carries the ``EPSILON = 1e-6`` numerical guard.
  * :data:`MATCH_AOU_BACKEND_P1_MILP_V1` -- the deterministic ``p = 1`` MILP
    (``match_aou_p1_milp_solver.MatchAouP1MILP``), solved through SciPy/HiGHS, with no
    ``EPSILON`` anywhere in it.

**THE TWO ARE NOT INTERCHANGEABLE AND SWAPPING THEM IS NOT A PERFORMANCE CHANGE.** At
``p = 1`` the legacy objective still pays ``utility * (EPSILON - EPSILON ** 2)`` for each
REDUNDANT agent on an already-covered task, so stacking is genuinely optimal FOR THAT
objective; the P1 formulation removes that incentive outright. Removing it changes WHICH
ALLOCATIONS ARE OPTIMAL, and therefore can change ``A_init``, the route-relative hidden
geometry built from ``A_init``, episode feasibility and what a policy learns. No
equivalence between the two objectives is claimed here, and none may be inferred from
this module's existence.

THE SELECTION RULES, ALL OF THEM DELIBERATE
-------------------------------------------
* :data:`DEFAULT_MATCH_AOU_BACKEND` is the LEGACY backend. Every approved measurement to
  date was taken on it, so a caller that says nothing keeps exactly the historical solver.
* The backend is an INDEPENDENT EXPLICIT selector. It is never inferred from
  ``episode_design``, from task probabilities, from whether BONMIN or SciPy happens to be
  importable, or from any other state -- a run that reached a different allocation
  objective because of what was installed on the machine would be a measurement nobody
  chose.
* There is deliberately **no** ``auto``, **no** fallback, **no** "try P1 then BONMIN", and
  **no** per-solve switching. One episode uses ONE backend for every MATCH-AOU solve it
  performs: the known-world ``A_init``, the static t=0 reference, the event-conditioned
  clean t=0 reference, the damaged continuation reference and the unrealized-event
  compatibility reference alike.
* An UNKNOWN id RAISES rather than falling back on the default. A run that quietly solved
  the legacy objective while its config said ``p1_milp_v1`` is a mislabelled measurement,
  which is worse than a crash.

IMPORT WEIGHT
-------------
This module holds ids, validation and a LAZY loader, and nothing else. It imports neither
the P1 solver nor SciPy at module scope, so exposing the policy ids costs a caller
nothing: :func:`load_p1_milp_solver` is the only thing that reaches the MILP stack, and it
runs only when ``p1_milp_v1`` was actually selected. (Importing any ``match_aou.*`` module
already pulls in Pyomo through the package ``__init__``; that is pre-existing and recorded
in ``CLAUDE.md`` section 8, and nothing here adds to it.)

This module is intentionally NOT re-exported from ``match_aou.solvers.__init__``: the
package's import surface stays exactly what it was.
"""

from __future__ import annotations

from typing import Any, Tuple

__all__ = [
    "MATCH_AOU_BACKEND_LEGACY_MINLP_V1",
    "MATCH_AOU_BACKEND_P1_MILP_V1",
    "MATCH_AOU_BACKENDS",
    "DEFAULT_MATCH_AOU_BACKEND",
    "MatchAouBackendError",
    "resolve_match_aou_backend",
    "uses_p1_milp",
    "load_p1_milp_solver",
]


#: The frozen general MINLP through BONMIN -- the historical objective, EPSILON and all.
MATCH_AOU_BACKEND_LEGACY_MINLP_V1: str = "legacy_minlp_v1"
#: The deterministic single-step ``p = 1`` MILP through SciPy/HiGHS -- no EPSILON.
MATCH_AOU_BACKEND_P1_MILP_V1: str = "p1_milp_v1"

#: The CLOSED set of backend ids. There are exactly two, and there is no ``auto``.
MATCH_AOU_BACKENDS: Tuple[str, ...] = (
    MATCH_AOU_BACKEND_LEGACY_MINLP_V1,
    MATCH_AOU_BACKEND_P1_MILP_V1,
)

#: The HISTORICAL default. Changing it would silently reinterpret every existing config.
DEFAULT_MATCH_AOU_BACKEND: str = MATCH_AOU_BACKEND_LEGACY_MINLP_V1


class MatchAouBackendError(RuntimeError):
    """A BACKEND / CONFIGURATION fault — never ordinary world attrition.

    Raised for the cases where the question the caller asked is not the question the
    selected backend answers, or where the selected backend cannot be reached at all:

      * an unknown backend id;
      * the P1 backend selected but its SciPy/HiGHS stack is not importable;
      * an input outside the P1 contract reaching a P1-selected runtime (a multi-step
        task, ``p != 1``, or precedence).

    **THE ROUTING IS THE POINT.** The active GENERALIZED-V1 domain is expected to satisfy
    the P1 assumptions, so none of these is a world that merely turned out infeasible.
    Every one of them says the INSTRUMENT is configured against a domain it does not
    model, which implicates every episode it touched. Training, the diagnostic rollout and
    the benchmark preflight therefore ABORT on it: it is never written to
    ``episode_failures.jsonl``, never counted against a condition or stratum tally, never
    entered into ``skip_and_account_v1``, never replaced by another training seed, never
    turned into a rejected benchmark candidate -- and it NEVER falls back on the legacy
    solver.

    It is deliberately a ``RuntimeError`` and NOT a ``ValueError``, matching the
    integrity-abort family this project already routes that way
    (``graph_fuel_damage.FuelDamageIntegrityError``,
    ``graph_reward.ReferenceIntegrityError``) rather than the ordinary config-validation
    family. It lives HERE, beside the ids it classifies, because the solver layer must not
    import the trainer -- the trainer imports this and routes it, exactly as it does for
    the fuel-damage and reference faults.

    A solver invocation that simply did not reach acceptable optimality is NOT this
    exception: that keeps the existing solve-failure / reference-failure semantics.
    """


def resolve_match_aou_backend(backend: Any) -> str:
    """Validate a backend id and return it as a plain ``str``.

    ``None`` resolves to :data:`DEFAULT_MATCH_AOU_BACKEND`, so an omitted selector means
    "the historical backend" rather than "no backend". Anything else must already be one
    of :data:`MATCH_AOU_BACKENDS`: no case-folding, no prefix matching and no fallback,
    because every one of those would let a typo run a different objective than the record
    claims.

    Raises:
        MatchAouBackendError: the id is not one of the two.
    """
    if backend is None:
        return DEFAULT_MATCH_AOU_BACKEND
    resolved = str(backend)
    if resolved not in MATCH_AOU_BACKENDS:
        raise MatchAouBackendError(
            "unknown MATCH-AOU backend %r: expected one of %s. There is deliberately no "
            "'auto' and no fallback -- a run that quietly solved a different allocation "
            "objective than its record claims is a mislabelled measurement."
            % (backend, list(MATCH_AOU_BACKENDS))
        )
    return resolved


def uses_p1_milp(backend: Any) -> bool:
    """``True`` iff ``backend`` selects the deterministic P1 MILP. Validates first."""
    return resolve_match_aou_backend(backend) == MATCH_AOU_BACKEND_P1_MILP_V1


def load_p1_milp_solver() -> Any:
    """Import and return ``MatchAouP1MILP`` — LAZILY, and only when P1 was selected.

    Keeping this behind a function is what lets the ids above be imported by the runtime
    (and by a config validator) without dragging SciPy's MILP stack into the import
    closure of every module that merely needs to name a backend.

    Raises:
        MatchAouBackendError: the P1 module (or the SciPy stack it needs) is not
            importable. This is a CONFIGURATION fault and it ABORTS -- it must never be
            answered by silently solving the legacy objective instead.
    """
    try:
        from .match_aou_p1_milp_solver import MatchAouP1MILP
    except Exception as exc:  # pragma: no cover - depends on the installed SciPy
        raise MatchAouBackendError(
            "MATCH-AOU backend %r was selected but its solver could not be imported "
            "(%s: %s). This is a configuration fault, not an episode outcome: it is "
            "never answered by falling back on the legacy MINLP."
            % (MATCH_AOU_BACKEND_P1_MILP_V1, type(exc).__name__, exc)
        ) from exc
    return MatchAouP1MILP
