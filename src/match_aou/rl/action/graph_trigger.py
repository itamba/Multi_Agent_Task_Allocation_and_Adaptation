"""
Graph Trigger Module (Phase-2 RL layer)
=======================================

Decides WHEN to wake the policy. The RL is NOT run on a periodic scan: agents fly
the static plan A_init "blind" via the executor until an EVENT wakes the policy.
This module is that gate — upstream of the decision core:

    sensed_target_ids  ->  [THIS: decide_triggers]  ->  graph_builder  ->  graph_action
       (executor eyes)        (WHEN + belief edit)       (observation)      (decision)

Its orchestrator is the graph tick-loop (``training/graph_tick_loop.py``). Like
``graph_action`` / ``graph_effect`` it is a pure,
hand-testable module (no BLADE engine, no torch).

Two event kinds
---------------
- POP-UP: the ego senses an enemy target that is NOT in its plan (a target it did
  not know about). It becomes a new task node the policy MAY engage.
- PEER-OVERDUE: the ego senses a target that A_init assigned to a PEER, and the peer's
  expected completion time (an ETA model) has passed. The ego INFERS the peer may have
  failed (from A_init + clock, NEVER from observing the peer) and may take it over.

NO-COMMUNICATION (research-critical, structurally enforced here)
----------------------------------------------------------------
1. PER-EGO PRIVATE BELIEF. ``decide_triggers`` operates on ONE ego's private
   ``(belief_tasks, belief_solution)`` and is PURE: it returns NEW copies, never
   mutates its inputs, and never reaches a global. Editing ego A's belief cannot
   touch ego B's separate belief (mirrors ``graph_effect.apply_meta_action``'s purity).
2. SENSING IS THE EGO'S OWN SENSOR ONLY. ``sensed_targets`` comes from the executor's
   ``sensed_target_ids`` (the ego's own radius); this module never derives sensing
   from a peer's state.
3. PEER-OVERDUE IS INFERRED, NOT SENSED. Overdue-ness comes from ``eta`` + ``clock``
   (an A_init-based model), never from observing the peer. It is a deterministic GATE
   on WHEN to ask the policy — the policy still freely decides whether to engage
   (via ``graph_action`` / ``graph_effect``).

Belief edits (what a trigger DOES, before the policy runs)
----------------------------------------------------------
- POP-UP:       APPEND a new pop-up Task to ``belief_tasks`` (``make_attack_task``).
                ``task_idx`` is POSITIONAL and indexes ``solution`` tuples, so tasks are
                APPEND-ONLY — existing tasks are NEVER removed or reordered. The pop-up
                is NOT added to ``belief_solution`` (that only happens if the policy later
                chooses OPPORTUNISTIC_ENGAGEMENT — see ``graph_effect``).
- PEER-OVERDUE: REMOVE the overdue peer's ``(task_idx, ...)`` tuple(s) from THIS ego's
                ``belief_solution`` copy, so the sensed target reads as unassigned+sensed
                — identical to a pop-up from the policy's view. The ego's own tuples and
                every other peer's tuples are untouched.

Relationship to ``graph_effect``
--------------------------------
This layer prepares the BELIEF the policy reasons over; ``graph_effect`` applies the
policy's CHOICE. Both are pure plan editors on the same ``solution`` dict. A
PEER-OVERDUE edit here makes the task look like a pop-up; if the policy then picks
OPPORTUNISTIC_ENGAGEMENT, ``graph_effect`` adds the ego's assignment — so this module
deliberately does the "remove peer edge" half and leaves the "add ego edge" half to
the effect layer, exactly as the removed Cooperative-Recovery meta-action would have.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from math import inf
from typing import Any, Callable, Dict, List, Tuple

from ...models import StepKind, Task
from ...utils.blade_utils.scenario_factory import make_attack_task

logger = logging.getLogger(__name__)

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level)


# =============================================================================
# Trigger kinds
# =============================================================================

class TriggerKind(IntEnum):
    """Kind of event that woke the policy, paired with a task-node index in ``events``."""

    POP_UP = 0
    PEER_OVERDUE = 1


# =============================================================================
# Injected ETA seam (placeholder — DORMANT by default)
# =============================================================================

def never_overdue(peer_id: str, task_idx: int) -> float:
    """Default ETA stub: expected completion is ``+inf`` for every peer/task.

    Because a peer is overdue only when ``clock > eta(peer, task)`` and ``+inf`` is
    never exceeded, this keeps the PEER-OVERDUE branch structurally present but
    DORMANT. That is deliberate: it lets the seam ship and be unit-tested without the
    real model.

    TODO (later effort, NOT built here): the real ETA — derived from haversine
    distance / aircraft cruise speed / task ordering / assignment levels — replaces
    this stub. Only the seam (an injected ``eta(peer_id, task_idx) -> float``) is built
    now.
    """
    return inf


# =============================================================================
# Local helpers (kept tiny/dependency-free, like graph_effect's private helpers)
# =============================================================================

def _task_target_id(task: Task) -> str:
    """Target-id of a task == its ATTACK step's ``target_id`` (mirrors graph_builder).

    Reimplemented locally rather than importing ``graph_builder._attack_step`` (a
    private symbol on a module we only consume), matching ``graph_effect``'s choice to
    reimplement ``_attack_step_index``.
    """
    steps = getattr(task, "steps", None) or []
    for step in steps:
        if getattr(step, "step_kind", None) == StepKind.ATTACK:
            return str(getattr(step, "target_id", ""))
    return str(getattr(steps[0], "target_id", "")) if steps else ""  # defensive fallback


def _copy_solution(
    solution: Dict[str, List[Assignment]],
) -> Dict[str, List[Assignment]]:
    """NEW dict, str keys, fresh lists of (immutable) tuples. Mirrors graph_effect._copy_solution.

    Tuples are immutable and the lists are freshly built, so the result is fully
    decoupled from the input — an edit below can never mutate the caller's plan.
    """
    return {str(aid): [tuple(t) for t in (tuples or [])] for aid, tuples in solution.items()}


# =============================================================================
# The trigger decision (pure function of ONE ego's private belief)
# =============================================================================

def decide_triggers(
    belief_tasks: List[Task],
    belief_solution: Dict[str, List[Assignment]],
    sensed_targets: Dict[str, Any],
    eta: Callable[[str, int], float] = never_overdue,
    *,
    ego_id: str,
    clock: float,
) -> Tuple[List[Task], Dict[str, List[Assignment]], bool, List[Tuple[TriggerKind, int]]]:
    """Decide whether the ego's sensing warrants waking the policy, editing its belief.

    PURE: operates on ONE ego's private belief, returns NEW copies, never mutates the
    inputs, never reaches a global (mirrors ``graph_effect.apply_meta_action``).

    ``sensed_targets`` is the ``{id: unit}`` map the executor's ``sensed_target_ids``
    returns directly: the keys are the ego's own sensed enemy target-ids and the values
    are the live BLADE units (``get_target`` results, already resolved there for the
    liveness check). The map form is needed because the POP-UP branch builds a Task via
    ``make_attack_task(unit)`` — which derives per-type utility + geometry from the unit —
    while the membership / peer-overdue tests use only the keys. The caller (a future
    orchestrator) passes the ``sensed_target_ids`` result straight through.

    Args:
        belief_tasks: the ego's private task list (``task_idx`` == list index). Never
            mutated; a pop-up is appended to a COPY (append-only — existing indices stay).
        belief_solution: the ego's private allocation ``{agent_id: [(task_idx, step_idx,
            level), ...]}``. Never mutated; a peer-overdue removal is on a COPY.
        sensed_targets: ``{target_id: unit}`` for enemies within the ego's OWN sensor
            range right now (the ego's own sensing only).
        eta: injected ``eta(peer_id, task_idx) -> float`` expected-completion tick.
            Defaults to :func:`never_overdue` (+inf -> PEER-OVERDUE dormant).
        ego_id: the deciding ego's id (keyword-only).
        clock: current simulation tick (keyword-only). A peer is overdue iff
            ``clock > eta(peer_id, task_idx)``.

    Returns:
        ``(new_belief_tasks, new_belief_solution, wake, events)`` where ``events`` is a
        list of ``(TriggerKind, task_idx)``. When nothing triggers, ``wake`` is False,
        ``events`` is empty, and the returned belief is an equal-but-not-same copy of the
        inputs.
    """
    # Equal-but-not-same copies up front: append-only for tasks, deep-ish for solution.
    new_tasks: List[Task] = list(belief_tasks)          # never reorder/remove existing
    new_solution: Dict[str, List[Assignment]] = _copy_solution(belief_solution)
    events: List[Tuple[TriggerKind, int]] = []
    wake = False

    ego_key = str(ego_id)

    # Existing belief target-id -> task_idx (first occurrence). Doubles as the
    # "already known?" membership test for pop-up classification.
    target_id_to_idx: Dict[str, int] = {}
    for idx, task in enumerate(belief_tasks):
        target_id_to_idx.setdefault(_task_target_id(task), idx)

    # Deterministic order over sensed ids (str-normalized keys).
    sensed_map = {str(k): v for k, v in sensed_targets.items()}
    for sid in sorted(sensed_map.keys()):
        if sid not in target_id_to_idx:
            # --- POP-UP: unknown sensed target -> APPEND a new task node -----------
            new_idx = len(new_tasks)
            new_tasks.append(make_attack_task(sensed_map[sid]))
            target_id_to_idx[sid] = new_idx  # keep the map consistent (ids are unique)
            events.append((TriggerKind.POP_UP, new_idx))
            wake = True
            logger.debug("POP_UP ego=%s target=%s -> task_idx=%d", ego_key, sid, new_idx)
            continue

        # --- PEER-OVERDUE: known sensed target assigned to an overdue peer ---------
        task_idx = target_id_to_idx[sid]
        removed_any = False
        # list(...) so reassigning a value mid-iteration is unambiguous.
        for aid, tuples in list(new_solution.items()):
            if aid == ego_key:
                continue  # only PEER assignments can be "taken over"
            if not any(int(t[0]) == task_idx for t in tuples):
                continue  # this peer is not assigned to task_idx
            if clock > eta(aid, task_idx):
                # Overdue: drop this peer's tuple(s) for task_idx from THIS ego's belief
                # ONLY (the no-comms point — no global, no other ego's plan touched).
                kept = [t for t in tuples if int(t[0]) != task_idx]
                if len(kept) != len(tuples):
                    new_solution[aid] = kept  # key retained (may be empty), like graph_effect ABORT
                    removed_any = True
        if removed_any:
            events.append((TriggerKind.PEER_OVERDUE, task_idx))
            wake = True
            logger.debug("PEER_OVERDUE ego=%s target=%s task_idx=%d", ego_key, sid, task_idx)

    return new_tasks, new_solution, wake, events


# =============================================================================
# Self-test (hand-built; NO BLADE engine, NO torch, NO solver)
# =============================================================================

def _selftest() -> None:
    """Assert pop-up / peer-overdue / dormant-eta + the no-comms & append-only red lines.

    Run under nlp_env from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.action.graph_trigger
    """
    import copy
    from types import SimpleNamespace

    from ...models import Location, Step

    # Named stub units so make_attack_task derives utility by type(unit).__name__.
    class Airbase:
        def __init__(self, uid, lat, lon):
            self.id, self.latitude, self.longitude, self.altitude = uid, lat, lon, 0

    def known_task(target_id: str, lat: float, lon: float) -> Task:
        step = Step(Location(lat, lon), target_id, [], 1.0, 2, StepKind.ATTACK)
        return Task(steps=[step], utility=80)

    # A_init: two known targets. Ego assigned t0; peer p1 assigned t1.
    belief_tasks = [known_task("t0", 32.0, 35.0), known_task("t1", 33.0, 36.0)]
    belief_solution = {"ego": [(0, 0, 0)], "p1": [(1, 0, 0)]}

    # Deep snapshots to prove the inputs are never mutated.
    tasks_snapshot = list(belief_tasks)
    solution_snapshot = copy.deepcopy(belief_solution)

    print("=" * 72)
    print("graph_trigger self-test")
    print("=" * 72)

    # (1) POP-UP: a sensed id absent from belief_tasks -> append exactly one task.
    popup_unit = Airbase("pop99", 32.05, 35.05)
    nt, ns, wake, events = decide_triggers(
        belief_tasks, belief_solution, {"pop99": popup_unit}, ego_id="ego", clock=100.0,
    )
    assert wake is True and events == [(TriggerKind.POP_UP, 2)], events
    assert len(nt) == 3 and _task_target_id(nt[2]) == "pop99"
    assert nt[2].utility == 80  # make_attack_task derived Airbase utility
    # Append-only: existing task_idx unchanged (same objects at 0/1).
    assert nt[0] is belief_tasks[0] and nt[1] is belief_tasks[1]
    # Pop-up is NOT added to the solution.
    assert ns == belief_solution and ns is not belief_solution
    # INPUTS NOT MUTATED.
    assert belief_tasks == tasks_snapshot and len(belief_tasks) == 2
    assert belief_solution == solution_snapshot
    print("[1] POP-UP: +1 task at idx 2, wake, event, solution unchanged, inputs intact   OK")

    # (2) PEER-OVERDUE with a synthetic overdue eta -> remove the peer's tuple only.
    overdue_eta = lambda peer, tidx: 50.0  # 50 < clock(100) -> overdue
    nt2, ns2, wake2, events2 = decide_triggers(
        belief_tasks, belief_solution, {"t1": Airbase("t1", 33.0, 36.0)},
        overdue_eta, ego_id="ego", clock=100.0,
    )
    assert wake2 is True and events2 == [(TriggerKind.PEER_OVERDUE, 1)], events2
    assert ns2["p1"] == []            # peer p1's (1,0,0) removed (key retained)
    assert ns2["ego"] == [(0, 0, 0)]  # a second ego's entry UNTOUCHED
    assert nt2 == belief_tasks        # no task added by peer-overdue
    # INPUTS NOT MUTATED.
    assert belief_solution == solution_snapshot
    print("[2] PEER-OVERDUE: peer tuple removed, ego entry untouched, inputs intact   OK")

    # (3) DORMANT default eta (+inf): same sensed peer-target -> NO trigger.
    nt3, ns3, wake3, events3 = decide_triggers(
        belief_tasks, belief_solution, {"t1": Airbase("t1", 33.0, 36.0)},
        ego_id="ego", clock=10_000_000.0,
    )
    assert wake3 is False and events3 == []
    assert ns3 == belief_solution and ns3 is not belief_solution
    print("[3] DORMANT eta (+inf default): sensed peer-target -> no trigger (wake False)   OK")

    # (4) Nothing sensed -> wake False, equal-but-not-same copies.
    nt4, ns4, wake4, events4 = decide_triggers(
        belief_tasks, belief_solution, {}, ego_id="ego", clock=100.0,
    )
    assert wake4 is False and events4 == []
    assert nt4 == belief_tasks and nt4 is not belief_tasks
    assert ns4 == belief_solution and ns4 is not belief_solution
    print("[4] nothing sensed -> wake False, equal-but-not-same copies   OK")

    # (5) NO-COMMS ISOLATION: a separately-held ego-B belief is byte-identical after
    #     ego A edits its own belief. Proves per-ego privacy via pure copies (no global).
    b_tasks = [known_task("t0", 32.0, 35.0)]
    b_solution = {"egoB": [(0, 0, 0)]}
    b_tasks_snap, b_solution_snap = list(b_tasks), copy.deepcopy(b_solution)
    _ = decide_triggers(
        belief_tasks, belief_solution, {"pop_a": Airbase("pop_a", 1.0, 1.0)},
        ego_id="ego", clock=100.0,
    )
    assert b_tasks == b_tasks_snap and b_solution == b_solution_snap
    print("[5] NO-COMMS ISOLATION: ego B's separate belief untouched by ego A's edit   OK")

    # (6) APPEND-ONLY across TWO pop-ups: earlier task_idx are stable.
    two = decide_triggers(
        belief_tasks, belief_solution,
        {"popA": Airbase("popA", 1.0, 1.0), "popB": Airbase("popB", 2.0, 2.0)},
        ego_id="ego", clock=100.0,
    )
    nt6, _ns6, wake6, events6 = two
    assert wake6 is True and sorted(e[1] for e in events6) == [2, 3]
    assert _task_target_id(nt6[0]) == "t0" and _task_target_id(nt6[1]) == "t1"  # unchanged
    assert {_task_target_id(nt6[2]), _task_target_id(nt6[3])} == {"popA", "popB"}
    print("[6] APPEND-ONLY: two pop-ups -> idx 2,3; earlier idx 0,1 unchanged   OK")

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
