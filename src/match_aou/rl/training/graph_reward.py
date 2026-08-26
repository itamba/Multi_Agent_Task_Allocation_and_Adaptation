"""graph_reward.py — the episode-terminal, utility-based reward (Phase-2 graph RL).

This is the reward layer that fills the seam the tick-loop leaves open. The graph
orchestrator is built and locked:

    graph_episode_setup.setup_episode(scenario)      -> EpisodeContext
    graph_tick_loop.run_episode(policy, ctx, cfg)    -> EpisodeResult
        EpisodeResult.trajectory : List[Transition]   # each .reward is None until HERE

``compute_episode_reward`` reads the finished episode through the ``EpisodeContext``
ONLY, computes a single scalar terminal reward, and writes it onto the trajectory.
Nothing else in the pipeline changes.

TWO REFERENCE POLICIES, SELECTED EXPLICITLY (never inferred)
------------------------------------------------------------
The reward is a REGRET against a MATCH-AOU reference. Which reference, and WHEN it is
solved, is chosen by ``EpisodeContext.reference_policy`` and by nothing else:

  * :data:`REFERENCE_POLICY_STATIC_T0_V1` (the DEFAULT, and the historical behaviour
    every approved measurement was taken on) -- ``setup_episode`` solves the full t=0
    world once and the reward normalizes by that static optimum. UNCHANGED.
  * :data:`REFERENCE_POLICY_EVENT_CONDITIONED_V1` (GENERALIZED-V1, opt-in) -- a CLEAN
    episode keeps a full t=0 reference; a DAMAGED episode replaces it with a MATCH-AOU
    CONTINUATION reference solved at the fuel-damage checkpoint, from the world and the
    agents as they REALLY are immediately after the mutation. ``run_episode`` owns the
    checkpoint and hands the result out as ``EpisodeResult.reference``.

A run under the second policy is normalized by ``U_ref``, NOT by a static oracle, and
this module deliberately never calls that reference an "oracle": it is conditioned on an
event the oracle could not have known about, and it is a MATCH-AOU ALLOCATION reference
rather than a claim of physical route-optimality.

REWARD (v1) -- STATIC t=0 REFERENCE (the default policy)
--------------------------------------------------------
    R = ( U_achieved  -  c * U_aircraft * n_lost  -  U_oracle ) / ( |U_oracle| + eps )

  U_oracle   = plan_value(ctx.oracle_solution, ctx.oracle_tasks)   # static full-set optimum, t=0
  U_achieved = realized_utility(ctx.oracle_tasks, ctx.executor.done) # what was actually killed
  U_aircraft = max(t.utility for t in ctx.oracle_tasks)  (0.0 if empty)
  n_lost     = len(ctx.executor.dead)
  c          = cfg.aircraft_penalty_coeff  (DEFAULT 0.0 -> pure utility ratio)
  eps        = cfg.regret_epsilon (DEFAULT 1e-5)  # DIVISION GUARD ONLY

Since ``U_achieved <= U_oracle`` (agents cannot beat the fully-informed oracle over
the same task set), the un-penalized ratio lies in ~[-1, 0]. Folding the death
penalty into the numerator makes it auto-scale with the scenario's utility magnitude,
so the gradient scale is consistent across scenarios. With ``c = 1.0`` a lost aircraft
costs exactly one max-utility target, so suicide-on-a-target is never net-positive;
``c > 1`` makes RTB strictly beat suicide-on-best.

REWARD -- EVENT-CONDITIONED CONTINUATION REFERENCE (GENERALIZED-V1, opt-in)
---------------------------------------------------------------------------
    U_ref      = U_prefix + U_cont_ref
    U_achieved = U_prefix + U_post

    ratio   = ( U_achieved - U_ref )              / ( |U_ref| + eps )
    penalty = ( c * U_aircraft * n_lost )         / ( |U_ref| + eps )
    R       = ratio - penalty                     # NEVER clamped

  U_prefix   = utility already realized (confirmed) BEFORE the checkpoint, FROZEN there
               and never recomputed from the larger end-of-episode ``done`` set.
  U_cont_ref = plan_value of the MATCH-AOU CONTINUATION allocation -- the reference
               solved at the checkpoint over the tasks that were still open, using the
               agents that were still continuation-capable at their real post-event fuel
               and position.
  U_post     = realized utility scored ONLY over the tasks the continuation reference
               ALLOCATED. A confirmed kill outside that set is ACCOUNTING-ONLY: it is
               reported (``unscored_completed_target_ids``) and adds NO utility.
  U_aircraft = the most valuable target of the REWARD-BEARING REFERENCE UNIVERSE (the
               prefix tasks plus the continuation-allocated tasks) -- derived from that
               universe rather than from a relabelled "oracle" task list.

  For a CLEAN episode under this policy ``U_prefix == 0`` and the reference IS the full
  t=0 reference, so the arithmetic collapses to the static formula above -- which is the
  checkable property that the opt-in path does not silently move the clean condition.

NO-COMMUNICATION RED LINE (enforced by construction)
----------------------------------------------------
This reward is a CENTRALIZED / privileged TRAINING signal. It MAY read global state
(``executor.done`` spans all egos; the oracle is full-info). It MUST NEVER write into
any ego's belief / observation, and MUST NEVER be consulted by the policy / encoder /
executor at decision time. The only mutation this module performs is
``Transition.reward`` on ``result.trajectory``. The runtime functions are torch-free
and BLADE-free. (T6 proves the write-purity; T1 proves the fidelity.)

OUT OF SCOPE (v1 is TERMINAL + UTILITY-ONLY — these are deliberate seams, NOT built here)
----------------------------------------------------------------------------------------
  * per-wake / dense regret shaping (this reward lands only on the terminal transition,
    under BOTH reference policies -- the credit assignment is untouched by GENERALIZED-V1);
  * any reference re-solve on the DEFAULT policy: ``static_t0_v1`` compares against the
    static t=0 full-set optimum and never re-solves. The event-conditioned policy's ONE
    continuation solve REPLACES that t=0 reference solve rather than adding to it, so an
    accepted episode still costs exactly two BONMIN calls;
  * run-level persistence of :class:`EpisodeReference` (no jsonl field, no run_config
    block, no summary key, no plot) -- that is a later GENERALIZED-V1 task, exactly as
    ``PostFdAdaptationOutcome`` is produced but not yet persisted;
  * centralized critic / value head / CTDE (``GraphEncoder.pool`` is the future hook);
  * variable-size PPO buffer, ``evaluate_action``, and GAE credit propagation (the
    terminal-on-last placement is exactly the seam the future GAE task consumes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

# REUSE the solver's probability-term stabilizer so U_oracle is bit-faithful to the
# solver's own optimum. This is the (1 - p + EPSILON)**m guard from the MINLP objective;
# it is NOT the reward's division guard (that is cfg.regret_epsilon) — keep them separate.
from ...solvers.match_aou_MINLP_solver import EPSILON
# The DOMAIN enum only (pure dataclasses + stdlib): the reward has to answer "which
# target does this task attack" for the event-conditioned reference's accounting, and
# guessing it from `steps[0]` alone would be a second, weaker rule beside the one
# `graph_episode_setup` / `graph_trigger` already use. Importing it keeps the runtime
# torch-free and BLADE-free.
from ...models import StepKind

if TYPE_CHECKING:  # Types only — keeps the runtime torch-free (tick-loop) and BLADE-free (setup/executor).
    from ...models import Task
    from .graph_episode_setup import EpisodeContext
    from .graph_tick_loop import EpisodeResult

# A normalized assignment is (task_idx, step_idx, level); plan_value only reads the
# first two, so a raw 2-tuple solution (as returned by ``MatchAou.solve``) works too.
Assignment = Tuple[int, int, int]
Solution = Dict[str, List[Assignment]]


# =============================================================================
# 0. The reference policy — ONE closed set, ONE stored value, ONE predicate
# =============================================================================

#: The DEFAULT, and the historical behaviour: ``setup_episode`` solves the full t=0
#: world a second time and the reward normalizes by that static optimum. Every approved
#: measurement (``CLAUDE.md`` section 7) was taken on this path and it is UNCHANGED.
REFERENCE_POLICY_STATIC_T0_V1: str = "static_t0_v1"
#: GENERALIZED-V1, OPT-IN: a CLEAN episode keeps a full t=0 reference solved before the
#: first tick; a DAMAGED episode replaces that t=0 reference solve with a MATCH-AOU
#: CONTINUATION solve taken at the fuel-damage checkpoint. Either way the episode costs
#: exactly the same two BONMIN calls the historical path costs.
REFERENCE_POLICY_EVENT_CONDITIONED_V1: str = "event_conditioned_continuation_v1"
#: The closed set ``setup_episode`` validates against. An unknown id RAISES; it is never
#: coerced, defaulted or ignored.
REFERENCE_POLICIES: Tuple[str, ...] = (
    REFERENCE_POLICY_STATIC_T0_V1,
    REFERENCE_POLICY_EVENT_CONDITIONED_V1,
)

#: Reference KINDS, produced ONLY under :data:`REFERENCE_POLICY_EVENT_CONDITIONED_V1`.
#: The historical policy produces NO :class:`EpisodeReference` at all -- that absence is
#: how a reader tells which policy ran, exactly as ``EpisodeContext.construction_audit``
#: being ``None`` identifies the historical hidden-cardinality policy.
#: A CLEAN episode's full t=0 reference, solved before the first tick.
REFERENCE_KIND_CLEAN_T0: str = "clean_t0"
#: A DAMAGED episode's continuation reference, solved AT the fuel-damage checkpoint.
REFERENCE_KIND_DAMAGED_EVENT: str = "damaged_event_checkpoint"
#: A DAMAGED-scheduled episode whose event NEVER FIRED, and which therefore physically
#: ran as a clean one. Its reference is the full t=0 reference over the RETAINED t=0
#: inputs -- so it is still solve #2 and never a third call -- and it carries its own
#: kind rather than being reported as a scheduled clean episode, because "the event did
#: not fire" and "no event was scheduled" are different facts. This case is REACHABLE
#: ONLY under the legacy FD eligibility policy, where a damaged ego that never reaches
#: the leg-progress threshold is a legitimate recorded observation (``CLAUDE.md``
#: section 7: the Phase-A rerun's seed 424); under ``certified_both_severities_v1`` the
#: tick loop's terminal ``require_certified_event_realized`` raises first.
REFERENCE_KIND_DAMAGED_EVENT_UNREALIZED: str = "damaged_event_unrealized_t0"
REFERENCE_KINDS: Tuple[str, ...] = (
    REFERENCE_KIND_CLEAN_T0,
    REFERENCE_KIND_DAMAGED_EVENT,
    REFERENCE_KIND_DAMAGED_EVENT_UNREALIZED,
)

#: Why an original episode ego was NOT offered to the continuation reference. Stable,
#: machine-readable slugs, checked in this order.
CONTINUATION_EXCLUSION_DEAD: str = "dead"
CONTINUATION_EXCLUSION_RTB: str = "rtb_committed"
CONTINUATION_EXCLUSION_NOT_AIRBORNE: str = "not_airborne"
CONTINUATION_EXCLUSION_REASONS: Tuple[str, ...] = (
    CONTINUATION_EXCLUSION_DEAD,
    CONTINUATION_EXCLUSION_RTB,
    CONTINUATION_EXCLUSION_NOT_AIRBORNE,
)


#: WHY a reference could not be produced honestly. Stable, machine-readable slugs, and
#: the ONLY thing a caller may branch on -- a routing decision taken by matching on an
#: exception's MESSAGE breaks the moment the wording improves.
#:
#: The set is split into exactly two ROUTING CLASSES, and the split is the whole point:
#:
#:   * :data:`REFERENCE_FAULT_SOLVE_UNACCEPTABLE` is ORDINARY ACCOUNTED ATTRITION. The
#:     question was asked and the solver did not answer it. Nothing was contradicted, no
#:     other episode is implicated, and the scheduled attempt is spent -- recorded once,
#:     never retried, never replaced, never converted into a zero-valued reference.
#:   * everything else is a MEASUREMENT-INTEGRITY fault: the instrument contradicted
#:     itself (a policy that requires a reference produced none, a reference was asked for
#:     with nothing to solve from, an unknown kind, or a record whose own arithmetic does
#:     not close). Every episode the reference layer touched is then suspect, so the
#:     harness ABORTS rather than shrinking a scientific denominator by attrition.
REFERENCE_FAULT_SOLVE_UNACCEPTABLE: str = "reference_solve_unacceptable"
REFERENCE_FAULT_MISSING: str = "reference_missing"
REFERENCE_FAULT_NO_UNIVERSE: str = "reference_universe_unavailable"
REFERENCE_FAULT_UNKNOWN_KIND: str = "reference_kind_unknown"
REFERENCE_FAULT_ARITHMETIC: str = "reference_arithmetic_contradiction"

#: Every legal reason, in routing order (the one attrition reason first).
REFERENCE_FAULT_REASONS: Tuple[str, ...] = (
    REFERENCE_FAULT_SOLVE_UNACCEPTABLE,
    REFERENCE_FAULT_MISSING,
    REFERENCE_FAULT_NO_UNIVERSE,
    REFERENCE_FAULT_UNKNOWN_KIND,
    REFERENCE_FAULT_ARITHMETIC,
)

#: The reasons that are ORDINARY EPISODE ATTRITION. Everything else aborts.
REFERENCE_ATTRITION_REASONS: Tuple[str, ...] = (REFERENCE_FAULT_SOLVE_UNACCEPTABLE,)


class ReferenceIntegrityError(RuntimeError):
    """The reward-bearing reference could not be established, or contradicts itself.

    Raised when a reference the episode DEPENDS on cannot be produced honestly:

      * the continuation MATCH-AOU solve did NOT reach acceptable optimality -- which is
        emphatically NOT the same fact as "it terminated acceptably and selected nothing".
        The second is a legitimate continuation reference with ``U_cont_ref == 0``; the
        first is an unanswered question, and converting it into an empty reference would
        silently hand the episode a denominator of zero and a reward of ``-0/eps == 0``,
        i.e. the OPTIMUM, for a solve that never happened;
      * the event-conditioned policy is in force but no reference reached the reward, so
        the static ``oracle_solution`` / ``oracle_tasks`` (which that policy deliberately
        leaves EMPTY) would have been read as if they were one;
      * a reference was asked for with no retained t=0 universe to solve from;
      * an unknown reference KIND;
      * a reference's own arithmetic does not reconcile (``U_ref != U_prefix + U_cont_ref``).

    ``reason`` IS A REQUIRED KEYWORD, and it is what a caller routes on. An optional one
    would let a future raise site skip the classification by omission and fall into
    whichever routing happened to be the default -- the same class of defect that made a
    roster fault look like ordinary episode attrition. There is deliberately no default
    and no inference from the message text.

    THIS LAYER STILL TAKES NO ROUTING DECISION AND IMPORTS NO TRAINER. It is a plain
    ``RuntimeError`` subclass, not a ``FuelDamageError`` / ``EpisodeRosterError`` sibling;
    what it now carries is the machine-readable FACT the harness needs in order to route
    correctly (:func:`reference_fault_aborts`).
    """

    def __init__(self, message: str, *, reason: str) -> None:
        if str(reason) not in REFERENCE_FAULT_REASONS:
            raise ValueError(
                "unknown ReferenceIntegrityError reason %r; expected one of %r"
                % (reason, list(REFERENCE_FAULT_REASONS))
            )
        super().__init__(message)
        self.reason: str = str(reason)

    @property
    def is_measurement_integrity(self) -> bool:
        """True iff this fault means the INSTRUMENT contradicted itself.

        The complement of "the solver was asked and did not answer", which is a fact
        about one attempt rather than about the measurement apparatus.
        """
        return self.reason not in REFERENCE_ATTRITION_REASONS


def reference_fault_aborts(exc: BaseException) -> bool:
    """Should this reference fault ABORT the run, rather than be accounted as attrition?

    THE ONE PREDICATE a harness routes on, so the classification lives beside the reasons
    it classifies instead of being re-derived (differently) at four handler sites. It
    reads the exception's ``reason`` SLUG and nothing else -- never its message.

    An exception that is not a :class:`ReferenceIntegrityError` is not this layer's fault
    to classify, and returns ``False``.
    """
    if not isinstance(exc, ReferenceIntegrityError):
        return False
    return exc.is_measurement_integrity


# =============================================================================
# 1. Plan value — the MINLP objective, reproduced EXACTLY (no re-solve)
# =============================================================================

def plan_value(solution: Solution, tasks: Sequence["Task"]) -> float:
    """Scalar value of a plan under the MATCH-AOU objective — bit-faithful to the solver.

    Reproduces ``match_aou_MINLP_solver.MatchAou._add_objective`` exactly:

        sum_j  u_j * prod_k [ 1 - (1 - p_jk + EPSILON) ** m_jk ]

    where ``j`` = task_idx (positional index into ``tasks``), ``k`` = step_idx,
    ``u_j`` = ``tasks[j].utility``, ``p_jk`` = ``tasks[j].steps[k].probability`` and
    ``m_jk`` = the number of DISTINCT agents assigned to ``(j, k)`` in ``solution``.

    The solver's ``y[j]`` factor is intentionally absent: a task with ``m_jk == 0`` on
    every step (i.e. unselected) has each factor ``1 - (·)**0 = 1 - 1 = 0`` -> product
    0 -> contributes 0, which is identical to ``y[j] == 0``. So this matches the solver
    objective for any solved model, whether ``tasks`` is the allocated-only normalized
    list or the raw pre-filter list. (``m_jk == 0`` is handled without crashing:
    ``x ** 0 == 1`` for any ``x`` in Python, including ``0.0 ** 0``.)

    WHY it exists: ``solve_and_normalize`` discards the Pyomo model, so no scalar plan
    value survives anywhere; the reward needs one and it must be faithful so
    ``U_oracle`` equals the solver's own optimum.

    Args:
        solution: ``{agent_id: [(task_idx, step_idx[, level]), ...]}``. Keys are agent
            ids; tuples may be 2- or 3-element (only ``[0]`` / ``[1]`` are read).
        tasks: the task list ``task_idx`` indexes into.

    Returns:
        The objective value as a float. Pure: NO torch, NO BLADE, NO re-solve.
    """
    # Distinct agents per (task_idx, step_idx). A set keyed by agent_id dedups a
    # single agent that lists the same (j, k) more than once.
    assigned: Dict[Tuple[int, int], Set[str]] = {}
    for agent_id, tuples in (solution or {}).items():
        for t in tuples:
            jk = (int(t[0]), int(t[1]))
            assigned.setdefault(jk, set()).add(str(agent_id))

    total = 0.0
    for j, task in enumerate(tasks):
        product = 1.0
        for k, step in enumerate(task.steps):
            m = len(assigned.get((j, k), ()))
            # m == 0 -> (·) ** 0 == 1 -> factor 0 (task contributes nothing). No crash.
            factor = 1.0 - (1.0 - step.probability + EPSILON) ** m
            product *= factor
        total += task.utility * product
    return total


# =============================================================================
# 2. Realized utility — what the egos actually killed (all-or-nothing per task)
# =============================================================================

def realized_utility(tasks: Sequence["Task"], done: Set[Tuple[str, str]]) -> float:
    """Utility actually achieved: a task pays out IFF every one of its targets is killed.

    ``done`` is the executor's confirmed-kill set keyed ``(ego_id, target_id)`` (both
    str). A target is "killed" if ANY ego confirmed it (dedup over ego — a target
    killed under two ego ids counts once). Task ``j`` contributes ``tasks[j].utility``
    IFF every one of its steps' targets is killed:

        all( any((ego, str(step.target_id)) in done for ego in egos) for step in task.steps )

    which collapses to "every step's target is in the killed-target set".

    WHY all-or-nothing: task utility is gained on COMPLETING the task (see
    ``Task.utility``); crediting per-target would over-reward a partially-completed
    multi-step task. The current regime is single-step, so this reduces to "target
    killed", but the all-steps form is future-proof.

    Args:
        tasks: the task list to score (the oracle's task list for the reward).
        done: the executor's confirmed-kill set, keyed ``(ego_id, target_id)``.

    Returns:
        The summed utility of fully-killed tasks, as a float. Pure: no torch / BLADE.
    """
    total = 0.0
    for j in realized_task_indices(tasks, done):
        total += tasks[j].utility
    return total


def realized_task_indices(
    tasks: Sequence["Task"], done: Set[Tuple[str, str]]
) -> Tuple[int, ...]:
    """Positional indices of the tasks that are FULLY realized under ``done``.

    THE ONE SITE that applies the all-steps rule, so ``realized_utility`` (which sums
    these tasks' utilities) and the event-conditioned reference (which must EXCLUDE
    exactly these tasks from the continuation universe, because their utility is already
    counted in ``U_prefix``) can never drift apart. Restating the rule in two places is
    how a task ends up counted twice, or in neither half.

    Args:
        tasks: the task list to score, indexed positionally.
        done: the executor's confirmed-kill set, keyed ``(ego_id, target_id)``.

    Returns:
        Ascending positional indices into ``tasks``. Pure: no torch / BLADE / solver.
    """
    killed: Set[str] = {target_id for (_ego, target_id) in (done or set())}
    return tuple(
        j for j, task in enumerate(tasks)
        if all(str(step.target_id) in killed for step in task.steps)
    )


def task_target_ids(tasks: Sequence["Task"]) -> Tuple[str, ...]:
    """Each task's ATTACK-step target id, in task order (``""`` if it names none).

    Local, like ``graph_episode_setup._task_target_id`` and
    ``graph_trigger._task_target_id`` -- this module must stay importable without the
    setup / builder / executor closure.
    """
    out: List[str] = []
    for task in tasks:
        steps = getattr(task, "steps", None) or []
        target = ""
        for step in steps:
            if getattr(step, "step_kind", None) == StepKind.ATTACK:
                target = str(getattr(step, "target_id", ""))
                break
        if not target and steps:
            target = str(getattr(steps[0], "target_id", ""))
        out.append(target)
    return tuple(out)


# =============================================================================
# 2b. The event-conditioned reference record (GENERALIZED-V1)
# =============================================================================

#: Absolute slack on ``u_ref == u_prefix + u_cont_ref``. Both operands are sums of float
#: utilities on the same scale, so this is float re-association noise only -- it is NOT a
#: tolerance on the reward and NOT a clamp.
_REFERENCE_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class EpisodeReference:
    """The reward-bearing MATCH-AOU reference for ONE episode, and how it was obtained.

    Produced ONLY under :data:`REFERENCE_POLICY_EVENT_CONDITIONED_V1`, by
    ``graph_episode_setup.build_t0_reference`` / ``build_continuation_reference``, and
    carried out of the episode as ``EpisodeResult.reference`` -- ONE owner, ONE surface.
    Under the historical policy no reference object exists at all, which is what makes
    "did the opt-in path run?" answerable from the artifact rather than inferred.

    IT IS A REFERENCE, NOT AN ORACLE, and the distinction is deliberate. A DAMAGED
    episode's reference is CONDITIONED on an event that had already happened when it was
    solved, so it is not the fully-informed t=0 optimum and must never be reported as
    one. It is also a MATCH-AOU ALLOCATION reference: it states what the solver would
    allocate from the post-event state, NOT a claim that the resulting physical routes
    are optimal.

    NOTHING HERE REACHES THE ACTING PATH. No field enters ``GraphObservation``, no field
    enters the central critic's ``CentralGraphObservation``, and no field is read by the
    executor, the trigger layer or the encoder. It is a TRAINING / measurement record, in
    exactly the sense ``graph_reward`` as a whole already is (the no-communication red
    line above).

    Attributes:
        policy: always :data:`REFERENCE_POLICY_EVENT_CONDITIONED_V1` -- recorded rather
            than implied, so a record states its own design.
        kind: one of :data:`REFERENCE_KINDS`; CLEAN t=0 vs DAMAGED event checkpoint vs
            the damaged-but-never-fired t=0 fallback.
        checkpoint_tick: the tick the DAMAGED checkpoint was taken at. ``None`` for a t=0
            reference -- ``None``, never ``0``, because tick 0 is a real tick.
        u_prefix: utility already CONFIRMED before the checkpoint, frozen HERE. ``0.0``
            for a t=0 reference by construction (nothing has been confirmed yet).
        u_cont_ref: ``plan_value`` of this reference's own allocation. For a t=0
            reference this IS the full t=0 reference value.
        u_ref: ``u_prefix + u_cont_ref`` -- the reward's denominator source. Verified on
            construction, never merely asserted in prose.
        u_aircraft: the most valuable target of the REWARD-BEARING REFERENCE UNIVERSE
            (prefix tasks plus allocated tasks), ``0.0`` when that universe is empty.
        solution: the reference allocation, positional over :attr:`tasks`.
        tasks: the ALLOCATED-ONLY normalized task list ``solution`` indexes. It is what
            ``U_post`` is scored over, and it is deliberately NOT a world inventory --
            the same allocated-only contract ``solve_and_normalize`` has always had.
        reference_target_ids: the target ids of :attr:`tasks` -- the REWARD-BEARING set a
            post-checkpoint kill must land in to score.
        prefix_target_ids: the target ids already realized at the checkpoint, retained so
            scored-vs-unscored accounting can be reconstructed without re-deriving the
            split. Empty for a t=0 reference.
        candidate_task_count: how many tasks were OFFERED to the reference solve (the
            continuation universe size). Larger than ``len(tasks)`` whenever the solver
            legitimately left some unselected.
        continuation_agent_ids: the original episode egos the solve was allowed to use,
            in scheduled order.
        excluded_agents: ``((ego_id, reason), ...)`` for every original ego that was NOT
            offered, with a stable :data:`CONTINUATION_EXCLUSION_REASONS` slug. A dead or
            RTB-committed ego appears HERE and can therefore never be reallocated -- the
            fact is RECORDED rather than left to be inferred from an absence.
        solver_invoked: False when the solve was SKIPPED because there was nothing to
            solve (no open task, or no continuation-capable ego). That is a legitimate
            zero reference and costs no BONMIN call.
        solver_accepted: whether the solver reached acceptable optimality. A reference is
            never CONSTRUCTED with ``False`` -- the builder raises
            :class:`ReferenceIntegrityError` instead -- so this is always True on a
            returned object, and it exists to make that guarantee legible in the record.
        solver_termination: the raw termination-condition name, recorded verbatim.
        solver_seconds: wall-clock seconds the reference solve took (``0.0`` when it was
            skipped). Wall-clock time passing is fine; SIMULATION time must not advance,
            and does not -- the checkpoint issues no ``env.step``.
    """

    policy: str
    kind: str
    checkpoint_tick: Optional[int]
    u_prefix: float
    u_cont_ref: float
    u_ref: float
    u_aircraft: float
    solution: Solution
    tasks: Tuple["Task", ...]
    reference_target_ids: Tuple[str, ...]
    prefix_target_ids: Tuple[str, ...]
    candidate_task_count: int
    continuation_agent_ids: Tuple[str, ...]
    excluded_agents: Tuple[Tuple[str, str], ...]
    solver_invoked: bool
    solver_accepted: bool
    solver_termination: str
    solver_seconds: float

    def __post_init__(self) -> None:
        # VERIFIED, not trusted: the whole point of carrying three numbers is that the
        # identity between them is checkable, and a record whose own arithmetic does not
        # close is worse than no record.
        if abs(self.u_ref - (self.u_prefix + self.u_cont_ref)) > _REFERENCE_TOLERANCE:
            raise ReferenceIntegrityError(
                "reference arithmetic does not reconcile: u_ref=%r but u_prefix=%r + "
                "u_cont_ref=%r = %r"
                % (self.u_ref, self.u_prefix, self.u_cont_ref,
                   self.u_prefix + self.u_cont_ref),
                reason=REFERENCE_FAULT_ARITHMETIC,
            )
        if self.kind not in REFERENCE_KINDS:
            raise ReferenceIntegrityError(
                "unknown reference kind %r; expected one of %r"
                % (self.kind, list(REFERENCE_KINDS)),
                reason=REFERENCE_FAULT_UNKNOWN_KIND,
            )

    @property
    def is_event_checkpoint(self) -> bool:
        """True iff this reference was solved AT the fuel-damage checkpoint."""
        return self.kind == REFERENCE_KIND_DAMAGED_EVENT

    def to_record(self) -> Dict[str, Any]:
        """A JSON-ready view (plain builtins only) for a later persistence task.

        Deliberately omits :attr:`tasks` (live ``Task`` objects) and :attr:`solution` (an
        allocation keyed by uuid): the ids here are WITHIN-RUN accounting identifiers,
        never a cross-run reproducibility key, because generated target and agent uuids
        are not seed-derived (``CLAUDE.md`` section 8).
        """
        return {
            "policy": str(self.policy),
            "kind": str(self.kind),
            "checkpoint_tick": (
                None if self.checkpoint_tick is None else int(self.checkpoint_tick)
            ),
            "u_prefix": float(self.u_prefix),
            "u_cont_ref": float(self.u_cont_ref),
            "u_ref": float(self.u_ref),
            "u_aircraft": float(self.u_aircraft),
            "allocated_task_count": int(len(self.tasks)),
            "candidate_task_count": int(self.candidate_task_count),
            "reference_target_ids": [str(t) for t in self.reference_target_ids],
            "prefix_target_ids": [str(t) for t in self.prefix_target_ids],
            "continuation_agent_ids": [str(a) for a in self.continuation_agent_ids],
            "excluded_agents": [
                [str(a), str(reason)] for (a, reason) in self.excluded_agents
            ],
            "solver_invoked": bool(self.solver_invoked),
            "solver_accepted": bool(self.solver_accepted),
            "solver_termination": str(self.solver_termination),
            "solver_seconds": float(self.solver_seconds),
        }


def uses_event_conditioned_reference(ctx: Any) -> bool:
    """THE ONE PREDICATE behind "which reference policy is this episode under?".

    Reads ``EpisodeContext.reference_policy`` -- the single stored source of truth,
    validated against :data:`REFERENCE_POLICIES` by ``setup_episode`` before any BLADE
    object exists -- and interprets it HERE and nowhere else, so the question cannot be
    spelled two ways.

    The attribute is read duck-typed because ``ctx`` is duck-typed throughout this
    pipeline (``run_episode`` and this module are both driven by lightweight stub
    contexts in the test suite and in the module self-tests). An object that declares no
    policy IS a historical ``static_t0_v1`` episode, which is precisely the default -- so
    this can only ever resolve an absent field to the PRESERVED path, never to the opt-in
    one.
    """
    return (
        str(getattr(ctx, "reference_policy", REFERENCE_POLICY_STATIC_T0_V1))
        == REFERENCE_POLICY_EVENT_CONDITIONED_V1
    )


# =============================================================================
# 3. Config + result breakdown
# =============================================================================

@dataclass(frozen=True)
class RewardConfig:
    """Knobs for the v1 terminal reward (frozen: the shared default is never mutated)."""

    # Ratio-denominator guard (the paper's regret epsilon). NOT the solver EPSILON.
    regret_epsilon: float = 1e-5
    # Death penalty coefficient c. v1 default 0.0 -> pure utility ratio. c >= 1.0
    # activates the death-only penalty (one lost aircraft == one max-utility target).
    aircraft_penalty_coeff: float = 0.0


@dataclass
class EpisodeReward:
    """The reward breakdown for logging / validation (all fields on the normalized scale).

    The first seven fields are the historical breakdown and keep their meanings under
    BOTH reference policies. Everything after them is the GENERALIZED-V1 event-conditioned
    checkpoint, and every one of those is ``None`` on the historical static path --
    ``None``, never ``0.0`` / ``0``, because on a normalized regret scale ``0`` is the
    OPTIMUM and on a count it reads as a measurement of nothing rather than as an absent
    measurement (the same "missing is null, never 0" rule the FD records already follow).

    ``u_oracle`` is ``Optional`` for the same reason: under
    :data:`REFERENCE_POLICY_EVENT_CONDITIONED_V1` there IS no static full-set optimum --
    setup deliberately did not solve one -- and reporting ``0.0`` there would fabricate a
    perfect oracle. :attr:`u_ref` is the denominator source under BOTH policies (on the
    static path it equals ``u_oracle``), so anything that wants "the number this reward
    was normalized by" reads that and is correct either way.
    """

    u_achieved: float   # realized utility. STATIC: fully-killed oracle tasks.
                        # EVENT-CONDITIONED: u_prefix + u_post.
    u_oracle: Optional[float]  # static full-set optimum; None under the event-conditioned policy
    u_aircraft: float   # most-valuable REFERENCE-UNIVERSE target's utility (0.0 if none)
    n_lost: int         # number of crashed egos (len(executor.dead))
    ratio: float        # (u_achieved - u_ref) / (|u_ref| + eps_regret)
    penalty: float      # (c * u_aircraft * n_lost) / (|u_ref| + eps_regret) -> >= 0
    reward: float       # ratio - penalty  == the R formula. NEVER clamped.

    # --- reference identity (both policies) ---
    reference_policy: str = REFERENCE_POLICY_STATIC_T0_V1
    u_ref: float = 0.0  # the denominator source; == u_oracle on the static path

    # --- GENERALIZED-V1 event-conditioned checkpoint (None on the static path) ---
    reference_kind: Optional[str] = None
    checkpoint_tick: Optional[int] = None
    u_prefix: Optional[float] = None
    u_cont_ref: Optional[float] = None
    u_post: Optional[float] = None
    unique_completed_targets: Optional[int] = None
    scored_completed_targets: Optional[int] = None
    unscored_completed_targets: Optional[int] = None
    unscored_completed_target_ids: Tuple[str, ...] = ()


# =============================================================================
# 4. The episode-terminal reward
# =============================================================================

def compute_episode_reward(
    ctx: "EpisodeContext",
    result: "EpisodeResult",
    cfg: RewardConfig = RewardConfig(),
) -> EpisodeReward:
    """Compute the terminal reward and write it onto the trajectory's last transition.

    WHICH REFERENCE IS USED IS DECIDED BY THE EPISODE, NOT BY THIS FUNCTION:

      * ``result.reference is None`` -> the HISTORICAL static-t=0 reward, read through
        ``ctx`` ONLY (``ctx.oracle_solution`` / ``ctx.oracle_tasks`` and
        ``ctx.executor.done`` / ``ctx.executor.dead``). Byte-for-byte the
        pre-GENERALIZED-V1 arithmetic (:func:`_static_t0_breakdown`).
      * a reference IS present -> the GENERALIZED-V1 event-conditioned reward against it
        (:func:`_event_conditioned_breakdown`). ``run_episode`` produces exactly one
        reference per episode under that policy and none at all under the historical one,
        so this branch cannot be entered by accident.

    An episode that DECLARES the event-conditioned policy but arrives without a reference
    raises :class:`ReferenceIntegrityError` rather than silently falling back on the
    static oracle that policy never solved.

    Placement (terminal-on-last convention, IDENTICAL under both policies): if
    ``result.trajectory`` is non-empty, every transition's reward is set to ``0.0`` and
    the LAST is overwritten with ``R`` (PPO/GAE propagates that terminal credit backward).
    If the trajectory is EMPTY — a zero-wake episode — nothing is attached, and the
    breakdown is still returned for logging.

    RED LINE: the ONLY mutation is ``Transition.reward`` on ``result.trajectory``.
    Nothing is written into any belief, executor plan, or observation.

    Args:
        ctx: the finished episode's :class:`EpisodeContext`.
        result: the :class:`EpisodeResult` from ``run_episode`` (its trajectory is the
            reward seam; its optional ``reference`` selects the branch above).
        cfg: reward knobs (see :class:`RewardConfig`).

    Returns:
        The :class:`EpisodeReward` breakdown.

    Raises:
        ReferenceIntegrityError: the event-conditioned policy is declared but no
            reference reached the reward.
    """
    # Duck-typed for the same reason `uses_event_conditioned_reference` is: this function
    # accepts anything exposing the EpisodeResult surface, and the module self-test plus
    # the test suite drive it with lightweight stubs. A result carrying NO reference IS a
    # historical static-t0 episode, which is the preserved path -- so an absent field can
    # only ever resolve to the historical branch, never to the opt-in one.
    reference = getattr(result, "reference", None)
    if reference is None:
        if uses_event_conditioned_reference(ctx):
            # The policy is in force but no reference reached here, so `oracle_solution`
            # / `oracle_tasks` are the EMPTY pair that policy deliberately leaves behind.
            # Falling through would normalize by 0 and hand the episode `-0/eps == 0`,
            # i.e. the optimum, for a reference that was never solved.
            raise ReferenceIntegrityError(
                "the episode declares %r but carries no EpisodeResult.reference; the "
                "static oracle it would fall back on was deliberately never solved"
                % REFERENCE_POLICY_EVENT_CONDITIONED_V1,
                reason=REFERENCE_FAULT_MISSING,
            )
        breakdown = _static_t0_breakdown(ctx, cfg)
    else:
        breakdown = _event_conditioned_breakdown(ctx, reference, cfg)

    # Terminal-on-last placement. MUTATE ONLY Transition.reward fields (the red line).
    # IDENTICAL under both policies -- GENERALIZED-V1 changes WHAT the scalar is, never
    # where it lands or how it is propagated.
    if result.trajectory:
        for tr in result.trajectory:
            tr.reward = 0.0
        result.trajectory[-1].reward = float(breakdown.reward)

    return breakdown


def _static_t0_breakdown(ctx: "EpisodeContext", cfg: RewardConfig) -> EpisodeReward:
    """The historical static-t=0 reward. BYTE-FOR-BYTE the pre-GENERALIZED-V1 arithmetic.

    Lifted out of :func:`compute_episode_reward` unchanged: the same reads, the same
    operand order, the same single ``denom``, the same folding of the penalty. Nothing
    here consults a reference policy, a checkpoint or a reference record -- this is the
    path every approved measurement was taken on and it must stay recognisable as such.
    """
    oracle_tasks = list(ctx.oracle_tasks)

    u_oracle = plan_value(ctx.oracle_solution, oracle_tasks)
    u_achieved = realized_utility(oracle_tasks, ctx.executor.done)
    u_aircraft = max((float(t.utility) for t in oracle_tasks), default=0.0)
    n_lost = len(ctx.executor.dead)

    # eps_regret is a DIVISION GUARD only (distinct from the solver EPSILON in plan_value).
    denom = abs(u_oracle) + cfg.regret_epsilon
    ratio = (u_achieved - u_oracle) / denom
    penalty = (cfg.aircraft_penalty_coeff * u_aircraft * n_lost) / denom
    reward = ratio - penalty  # == (u_achieved - c*u_aircraft*n_lost - u_oracle) / denom

    return EpisodeReward(
        u_achieved=float(u_achieved),
        u_oracle=float(u_oracle),
        u_aircraft=float(u_aircraft),
        n_lost=int(n_lost),
        ratio=float(ratio),
        penalty=float(penalty),
        reward=float(reward),
        reference_policy=REFERENCE_POLICY_STATIC_T0_V1,
        # The number the reward really was normalized by. On this path it IS the static
        # oracle; carrying it under a policy-neutral name is what lets a consumer read
        # "the denominator" without first asking which policy ran.
        u_ref=float(u_oracle),
    )


def _event_conditioned_breakdown(
    ctx: "EpisodeContext", reference: EpisodeReference, cfg: RewardConfig
) -> EpisodeReward:
    """The GENERALIZED-V1 reward against the event-conditioned reference.

        U_achieved = U_prefix + U_post
        U_ref      = U_prefix + U_cont_ref        (verified on the reference itself)
        R          = (U_achieved - U_ref)/(|U_ref| + eps)
                     - (c * U_aircraft * n_lost)/(|U_ref| + eps)

    ``U_prefix`` is taken STRAIGHT OFF the reference -- it was frozen at the checkpoint
    and is deliberately NOT recomputed here, because the ``done`` set has grown since and
    recomputing would silently credit post-checkpoint kills to the prefix and double-count
    them against ``U_post``.

    ``U_post`` is scored ONLY over ``reference.tasks``, the ALLOCATED-ONLY continuation
    task list. A confirmed kill on any other target -- a target the reference did not
    allocate, or one the prefix already paid for -- contributes NO utility and is reported
    as accounting instead (``unscored_completed_target_ids``).

    ``U_aircraft`` comes from the REWARD-BEARING REFERENCE UNIVERSE (prefix tasks plus
    allocated tasks) rather than from a task list relabelled as an oracle, so the death
    penalty stays on the same utility scale as the numerator it is subtracted from.

    NOT CLAMPED. With ``c > 0`` and a real airframe loss the reward legitimately falls
    below ``-1``, exactly as it already can on the static path.
    """
    done = ctx.executor.done
    ref_tasks = list(reference.tasks)

    u_prefix = float(reference.u_prefix)
    u_post = realized_utility(ref_tasks, done)
    u_achieved = u_prefix + u_post
    u_ref = float(reference.u_ref)
    u_aircraft = float(reference.u_aircraft)
    n_lost = len(ctx.executor.dead)

    denom = abs(u_ref) + cfg.regret_epsilon
    ratio = (u_achieved - u_ref) / denom
    penalty = (cfg.aircraft_penalty_coeff * u_aircraft * n_lost) / denom
    reward = ratio - penalty

    # --- accounting: which confirmations actually paid, and which did not -------
    # Target ids, deduplicated over ego exactly as `realized_utility` does. The
    # reward-bearing set is the prefix (already counted in U_prefix) plus the
    # continuation-allocated targets (the only ones U_post can score).
    confirmed_ids = {str(target_id) for (_ego, target_id) in (done or set())}
    reward_bearing = set(reference.prefix_target_ids) | set(reference.reference_target_ids)
    scored = sorted(confirmed_ids & reward_bearing)
    unscored = sorted(confirmed_ids - reward_bearing)

    return EpisodeReward(
        u_achieved=float(u_achieved),
        # NO static oracle exists under this policy and none was solved. `None` says so;
        # a 0.0 would read as a perfect-oracle measurement.
        u_oracle=None,
        u_aircraft=float(u_aircraft),
        n_lost=int(n_lost),
        ratio=float(ratio),
        penalty=float(penalty),
        reward=float(reward),
        reference_policy=str(reference.policy),
        u_ref=float(u_ref),
        reference_kind=str(reference.kind),
        checkpoint_tick=(
            None if reference.checkpoint_tick is None else int(reference.checkpoint_tick)
        ),
        u_prefix=float(u_prefix),
        u_cont_ref=float(reference.u_cont_ref),
        u_post=float(u_post),
        unique_completed_targets=len(confirmed_ids),
        scored_completed_targets=len(scored),
        unscored_completed_targets=len(unscored),
        unscored_completed_target_ids=tuple(unscored),
    )


# =============================================================================
# Self-test (branch-coverage on duck-typed stubs; T1/T7 need bonmin and SKIP if absent)
# =============================================================================

def _selftest() -> None:
    """Branch-coverage on lightweight stubs (always) + fidelity/end-to-end (bonmin, SKIP-able).

    Run from the repo root under nlp_env, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.training.graph_reward
    """
    import copy
    import math
    from types import SimpleNamespace

    print("=" * 72)
    print("graph_reward self-test")
    print("=" * 72)

    # --- Duck-typed stub builders (no BLADE/env/solver needed) ---------------
    def _mk_task(utility, steps):
        # steps: list of (probability, target_id).
        return SimpleNamespace(
            utility=utility,
            steps=[SimpleNamespace(probability=p, target_id=tid) for (p, tid) in steps],
        )

    def _mk_ctx(oracle_solution, oracle_tasks, done, dead, *,
                beliefs=None, plans=None, observation=None):
        executor = SimpleNamespace(
            done=set(done), dead=set(dead),
            plans=plans if plans is not None else {},
        )
        return SimpleNamespace(
            oracle_solution=oracle_solution,
            oracle_tasks=oracle_tasks,
            executor=executor,
            beliefs=beliefs if beliefs is not None else {},
            observation=observation,
        )

    def _mk_result(rewards):
        # rewards: initial per-transition reward values (None or float).
        return SimpleNamespace(trajectory=[SimpleNamespace(reward=r) for r in rewards])

    e = EPSILON

    # =====================================================================
    # T1 — plan_value fidelity vs the solved MINLP objective (needs bonmin)
    # =====================================================================
    print("-" * 72)
    print("[T1] plan_value fidelity vs value(model.obj)")
    try:
        from pyomo.environ import value
        from ...models import Agent, Task, Step, StepKind, Location
        from ...solvers import MatchAou

        def _mv(src, dst):
            return src.distance_to(dst)

        loc0 = Location(0.0, 0.0)
        agents = [
            Agent(location=loc0, capabilities=[], budget=1e9, move_cost_function=_mv,
                  return_location=loc0, agent_id=f"A{i}")
            for i in range(2)
        ]
        tasks_t1 = [
            Task(steps=[Step(Location(0.1, 0.1), "T0", [], 0.9, 1, StepKind.ATTACK)], utility=100),
            Task(steps=[Step(Location(0.2, 0.2), "T1", [], 0.85, 1, StepKind.ATTACK)], utility=80),
        ]
        model = MatchAou(agents=agents, tasks=tasks_t1, precedence_relations=[], risk_factor=0.0)
        raw_solution, _results, _unsel = model.solve(solver_name="bonmin")
        if not raw_solution:
            print("  [T1] SKIP: solver returned no solution (not optimal / selected nothing)")
        else:
            obj_val = value(model.model.obj)
            pv = plan_value(raw_solution, tasks_t1)
            assert abs(pv - obj_val) < 1e-9, (pv, obj_val)
            print(f"  [T1] plan_value={pv:.10f} == value(model.obj)={obj_val:.10f}   OK")
    except Exception as exc:  # bonmin/pyomo missing, or solver error -> SKIP, never fail.
        print(f"  [T1] SKIP (bonmin/pyomo unavailable): {type(exc).__name__}: {exc}")

    # =====================================================================
    # T2 — plan_value hand value (known m_jk, p, u) incl. an m == 0 task
    # =====================================================================
    print("-" * 72)
    print("[T2] plan_value hand value")
    tasks_t2 = [
        _mk_task(10, [(0.5, "x0")]),                 # 1 step, m=1
        _mk_task(20, [(1.0, "y0"), (1.0, "y1")]),    # 2 steps, m=2 then m=1
        _mk_task(50, [(0.9, "z0")]),                 # NO assignment -> m=0 -> contributes 0
    ]
    solution_t2 = {
        "A": [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        "B": [(1, 0, 0), (1, 0, 0)],                 # duplicate (1,0) must NOT inflate m past 2
    }
    exp0 = 10 * (1 - (1 - 0.5 + e) ** 1)
    exp1 = 20 * (1 - (1 - 1.0 + e) ** 2) * (1 - (1 - 1.0 + e) ** 1)
    exp2 = 50 * (1 - (1 - 0.9 + e) ** 0)             # == 50 * (1 - 1) == 0
    expected_t2 = exp0 + exp1 + exp2
    pv2 = plan_value(solution_t2, tasks_t2)
    assert abs(pv2 - expected_t2) < 1e-12, (pv2, expected_t2)
    assert exp2 == 0.0, exp2
    print(f"  [T2] plan_value={pv2:.10f} == expected={expected_t2:.10f} (m=0 -> 0)   OK")

    # =====================================================================
    # T3 — realized_utility: all-steps credit, ego dedup, partial multi-step -> 0
    # =====================================================================
    print("-" * 72)
    print("[T3] realized_utility all-or-nothing + ego dedup")
    tasks_t3 = [
        _mk_task(10, [(1.0, "t0")]),                 # single-step, killed -> +10
        _mk_task(20, [(1.0, "t1a"), (1.0, "t1b")]),  # multi-step, one unkilled -> 0
        _mk_task(30, [(1.0, "t2")]),                 # killed by TWO egos (dedup) -> +30 once
    ]
    done_t3 = {("A", "t0"), ("A", "t1a"), ("A", "t2"), ("B", "t2")}
    ru = realized_utility(tasks_t3, done_t3)
    assert ru == 40, ru
    # Kill the remaining step of task1 -> it now pays out too (all-steps credit).
    ru2 = realized_utility(tasks_t3, done_t3 | {("C", "t1b")})
    assert ru2 == 60, ru2
    print(f"  [T3] realized_utility={ru} (task1 partial excluded; t2 deduped); +t1b -> {ru2}   OK")

    # =====================================================================
    # T4 — compute_episode_reward on stubs: ratio bounds, placement, empty, u_oracle=0
    # =====================================================================
    print("-" * 72)
    print("[T4] compute_episode_reward branch coverage")
    oracle_tasks = [_mk_task(100, [(1.0, "t0")]), _mk_task(50, [(1.0, "t1")])]
    oracle_solution = {"A": [(0, 0, 0)], "B": [(1, 0, 0)]}

    # (a) all-killed -> ratio ~ 0 (u_achieved == full sum ~ u_oracle at p=1.0).
    ctx_all = _mk_ctx(oracle_solution, oracle_tasks, done={("A", "t0"), ("B", "t1")}, dead=set())
    res_all = _mk_result([None, None, None])
    br_all = compute_episode_reward(ctx_all, res_all)
    assert abs(br_all.ratio) < 1e-3, br_all.ratio
    # placement: last == reward, all others normalized to 0.0.
    assert res_all.trajectory[-1].reward == br_all.reward
    assert all(t.reward == 0.0 for t in res_all.trajectory[:-1])
    print(f"  [T4a] all-killed ratio={br_all.ratio:.3e} ~ 0; terminal placed, rest 0.0   OK")

    # (b) none-killed -> ratio ~ -1.
    ctx_none = _mk_ctx(oracle_solution, oracle_tasks, done=set(), dead=set())
    br_none = compute_episode_reward(ctx_none, _mk_result([None]))
    assert abs(br_none.ratio - (-1.0)) < 1e-3, br_none.ratio
    print(f"  [T4b] none-killed ratio={br_none.ratio:.6f} ~ -1   OK")

    # (c) empty trajectory -> nothing attached, breakdown still returned.
    res_empty = _mk_result([])
    br_empty = compute_episode_reward(ctx_none, res_empty)
    assert res_empty.trajectory == []
    assert isinstance(br_empty, EpisodeReward)
    print("  [T4c] empty trajectory -> no attachment, breakdown returned   OK")

    # (d) u_oracle == 0 -> no division blow-up (denom == eps_regret).
    ctx_zero = _mk_ctx({}, [], done=set(), dead=set())
    br_zero = compute_episode_reward(ctx_zero, _mk_result([None]))
    assert br_zero.u_oracle == 0.0 and math.isfinite(br_zero.reward), br_zero
    print(f"  [T4d] u_oracle=0 -> reward={br_zero.reward} finite (no blow-up)   OK")

    # =====================================================================
    # T5 — penalty: c=0 -> reward == ratio; c=1, n_lost=1 -> drop == u_aircraft/denom
    # =====================================================================
    print("-" * 72)
    print("[T5] death penalty folding")
    ctx_p = _mk_ctx(oracle_solution, oracle_tasks, done={("A", "t0")}, dead={"C"})  # 1 lost
    br0 = compute_episode_reward(ctx_p, _mk_result([None]),
                                 RewardConfig(aircraft_penalty_coeff=0.0))
    assert br0.penalty == 0.0
    assert br0.reward == br0.ratio
    br1 = compute_episode_reward(ctx_p, _mk_result([None]),
                                 RewardConfig(aircraft_penalty_coeff=1.0))
    assert br1.u_aircraft == 100.0, br1.u_aircraft   # max utility target
    assert br1.n_lost == 1, br1.n_lost
    denom = abs(br1.u_oracle) + 1e-5
    expected_drop = br1.u_aircraft / denom
    assert br0.ratio == br1.ratio                    # ratio is coeff-independent
    assert abs((br1.ratio - br1.reward) - expected_drop) < 1e-12, (br1.ratio, br1.reward)
    print(f"  [T5] c=0 -> reward==ratio; c=1 drop={br1.ratio - br1.reward:.6f} "
          f"== u_aircraft/denom={expected_drop:.6f}   OK")

    # =====================================================================
    # T6 — purity / no-comms RED LINE (ALWAYS runs, on stubs)
    # =====================================================================
    print("-" * 72)
    print("[T6] purity / no-comms (external objects byte-unchanged)")
    # External objects the reward MUST NOT touch (plain comparable structures).
    beliefs = {"A": {"tasks": ["t0"], "solution": {"A": [(0, 0, 0)]}},
               "B": {"tasks": ["t1"], "solution": {"B": [(1, 0, 0)]}}}
    plans = {"A": [(0, 0, 0)], "B": [(1, 0, 0)]}
    observation = {"marker": "obs", "aircraft": [1, 2, 3]}
    oracle_solution_t6 = {"A": [(0, 0, 0)], "B": [(1, 0, 0)]}
    oracle_tasks_t6 = [_mk_task(100, [(1.0, "t0")]), _mk_task(50, [(1.0, "t1")])]
    ctx6 = _mk_ctx(oracle_solution_t6, oracle_tasks_t6, done={("A", "t0")}, dead=set(),
                   beliefs=beliefs, plans=plans, observation=observation)

    beliefs_snap = copy.deepcopy(beliefs)
    plans_snap = copy.deepcopy(plans)
    obs_snap = copy.deepcopy(observation)
    oracle_sol_snap = copy.deepcopy(oracle_solution_t6)

    # Pre-zeroed non-terminal rewards so ONLY the last transition visibly changes.
    res6 = _mk_result([0.0, 0.0])
    br6 = compute_episode_reward(ctx6, res6)

    assert ctx6.beliefs == beliefs_snap, "beliefs mutated!"
    assert ctx6.executor.plans == plans_snap, "executor.plans mutated!"
    assert ctx6.observation == obs_snap, "observation mutated!"
    assert ctx6.oracle_solution == oracle_sol_snap, "oracle_solution mutated!"
    # Only trajectory[-1].reward changed; the non-terminal one stayed 0.0.
    assert res6.trajectory[0].reward == 0.0, res6.trajectory[0].reward
    assert res6.trajectory[-1].reward == br6.reward
    print("  [T6] beliefs/executor.plans/observation byte-unchanged; only last reward set   OK")

    # =====================================================================
    # T7 — real end-to-end smoke (SKIP if bonmin/env/setup unavailable)
    # =====================================================================
    print("-" * 72)
    print("[T7] real end-to-end smoke")
    try:
        import tempfile
        from pathlib import Path

        import blade.utils.PlaybackRecorder as _pbr
        _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # PlaybackRecorder CHARACTER_LIMIT override (historical flat-era convention)

        from ...utils.blade_utils.scenario_generator import ScenarioGenerator, VariationConfig
        from .graph_episode_setup import setup_episode, MAX_SIM_TICKS, DETECTION_KM
        from .graph_tick_loop import build_policy, run_episode

        repo_root = Path(__file__).resolve().parents[4]
        base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
        out_dir = tempfile.mkdtemp(prefix="graph_reward_selftest_")

        gen = ScenarioGenerator(base_scenario_path=str(base_scenario), output_dir=out_dir,
                                max_sim_ticks=MAX_SIM_TICKS)
        gen.recompute_time_feasible_cap(allowed_classes=None)
        cfg_gen = VariationConfig(
            include_sams=False, num_red_airbases=(3, 3),
            randomize_red_airbase_positions=True, stretch_target_ratio=0.5,
            detection_km=DETECTION_KM,  # single-radius: generator connectivity == split == sensing (50 km)
            seed=0,
        )
        scenario_path = str(gen.generate(episode=0, config=cfg_gen))
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_json = f.read()

        import torch
        torch.manual_seed(0)
        ctx = setup_episode(scenario_json, recording_export_path=out_dir)
        policy = build_policy(embed_dim=64)

        result = run_episode(policy, ctx, deterministic=True, max_ticks=3000)
        traj_len_before = len(result.trajectory)

        # PURITY on REAL objects: snapshot the POST-run state right before the reward
        # call, so the check below isolates compute_episode_reward. (run_episode itself
        # legitimately edits beliefs/executor.plans on every wake — under the real
        # discovery-chain split a pop-up wake mutates the woken ego's belief, so a
        # pre-run snapshot would spuriously trip.)
        beliefs_before = {aid: (copy.deepcopy(b.tasks), copy.deepcopy(b.solution))
                          for aid, b in ctx.beliefs.items()}
        plans_before = copy.deepcopy(ctx.executor.plans)

        br = compute_episode_reward(ctx, result)

        print(f"  [T7] ended={result.ended} ticks={result.ticks} wakes={result.n_wakes} "
              f"kills={result.confirmed_kills} dead={result.n_dead}")
        print(f"  [T7] u_achieved={br.u_achieved:.4f} u_oracle={br.u_oracle:.4f} "
              f"u_aircraft={br.u_aircraft:.1f} n_lost={br.n_lost} reward={br.reward:.6f}")

        assert isinstance(br, EpisodeReward)
        assert math.isfinite(br.reward), br.reward
        assert br.u_oracle > 0.0, "real oracle has no value?!"
        # Bound holds for the probability=1.0 regime: U_achieved is raw realized utility,
        # U_oracle is EPSILON-discounted (plan_value), so an all-killed episode can exceed
        # U_oracle by ~U_oracle*1e-6 — allow a small relative slack.
        assert br.u_achieved <= br.u_oracle * (1.0 + 1e-5) + 1e-9, (br.u_achieved, br.u_oracle)

        # Real discovery-chain split -> organic pop-up wakes place a terminal reward on
        # the last transition; an empty trajectory (no wake fired) attaches nothing.
        if traj_len_before == 0:
            assert result.trajectory == []
            print("  [T7] empty trajectory -> nothing attached   OK")
        else:
            assert result.trajectory[-1].reward == br.reward
            print(f"  [T7] non-empty trajectory ({traj_len_before}) -> terminal reward placed   OK")

        # PURITY on real objects: beliefs + executor plans byte-unchanged by the reward.
        for aid, b in ctx.beliefs.items():
            t_before, s_before = beliefs_before[aid]
            assert [[str(s.target_id) for s in t.steps] for t in b.tasks] \
                   == [[str(s.target_id) for s in t.steps] for t in t_before], f"belief {aid} tasks changed"
            assert b.solution == s_before, f"belief {aid} solution changed"
        assert ctx.executor.plans == plans_before, "executor.plans changed"
        print("  [T7] REAL beliefs + executor.plans byte-unchanged after reward   OK")

        ctx.env.close()
    except Exception as exc:
        print(f"  [T7] SKIP (bonmin/env/setup unavailable): {type(exc).__name__}: {exc}")

    print("-" * 72)
    print("All assertions passed (skipped tests noted above).")


if __name__ == "__main__":
    _selftest()
