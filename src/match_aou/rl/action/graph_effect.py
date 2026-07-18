"""
Graph Effect Module (Phase-2 RL layer)
======================================

The semantic twin of ``graph_action.py``. Where ``build_action_mask`` defines
*when* each meta-action is legal, this module defines *what* each meta-action
DOES to the plan. Together they are the two halves of the decision core:

    build_graph_observation  ->  encoder  ->  graph_action (mask + head + sample)
                                                     |  emits <meta_action m, node v>
                                                     v
                                              graph_effect.apply_meta_action
                                                     |  edits the PLAN
                                                     v
                                          new `solution`  ->  (future) executor

Design invariant (OUR CHOICE): the static plan is the ``solution`` dict
``{agent_id: [(task_idx, step_idx, level), ...]}`` and is the SINGLE source of
truth. The graph is a pure projection of ``(solution, world)``, rebuilt fresh at
every trigger — there is NO in-place graph mutation. The effect layer therefore
edits the PLAN, not the graph; the next ``build_graph_observation`` re-derives
the edges from the updated ``solution``.

Purity (OUR CHOICE): ``apply_meta_action`` is a pure function — ``solution`` in,
new ``solution`` out. No torch, no BLADE, no simulation access, independently
unit-testable (see ``_selftest``), exactly like ``graph_action``'s mask. It
returns a NEW dict with copied lists and never mutates its input, so PPO/rollout
code can keep the pre-decision plan.

Meta-action -> plan edit (these map onto ASSIGNMENT-edge deltas, expressed on the
``solution`` dict, NOT on edges):

- PLAN_COMPLIANCE          : no edit. Return the plan unchanged.
- OPPORTUNISTIC_ENGAGEMENT : add ASSIGNMENT ego -> task v  (== append a tuple).
- SELF_PRESERVATION_ABORT  : remove ASSIGNMENT ego -> task v (== drop the tuple(s)).

OUR CHOICE: OPPORTUNISTIC_ENGAGEMENT is the sole "add assignment" meta-action.
Peer-failure recovery is handled upstream by the trigger layer (a peer-overdue
sensed target is converted to a pop-up), so there is no separate Cooperative
Recovery meta-action; the plan-level effect is "the ego now also attacks v".

OUR CHOICE: "push to front" is expressed purely through the plan. The inserted
level is ``min(existing ego levels) - 1`` (or ``0`` if the ego is unassigned).
The executor orders by ``level_order`` first, so a lower level puts the new
target at the FRONT of the ego's queue (attack-now) while the ego's original
assignments stay in ``solution`` at their original levels and resume naturally
afterward. No queue object is mutated; the temporal priority lives in the level.

EXTENDS / OUR CHOICE: SELF_PRESERVATION_ABORT only removes the assignment. In the
single-target regime this empties the ego's queue, which the FUTURE executor
turns into RTB — but the effect layer does NOT issue RTB; it only edits the plan.

This module replaced the retired flat ``plan_editor.py``. The executor that consumes
the updated ``solution`` is ``GraphPlanExecutor``; this layer stays decoupled from
it (no import of, or dependency on, any BLADE executor).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from ...models import StepKind, Task
from ..observation.graph_builder import GraphObservation
from .graph_action import MetaAction

logger = logging.getLogger(__name__)


# =============================================================================
# Local helpers (kept tiny and dependency-free, like graph_action's predicates)
# =============================================================================

def _attack_step_index(task: Task) -> int:
    """Index of the task's ATTACK step within ``task.steps``.

    Mirrors ``graph_builder._attack_step`` (find the first ``StepKind.ATTACK``,
    else fall back to step 0), but returns the INDEX rather than the step object
    and is reimplemented locally — ``graph_builder._attack_step`` is private on a
    module we must not import private symbols from.
    """
    steps = getattr(task, "steps", None) or []
    for idx, step in enumerate(steps):
        if getattr(step, "step_kind", None) == StepKind.ATTACK:
            return idx
    return 0  # defensive fallback (mirrors graph_builder's steps[0])


def _copy_solution(
    solution: Dict[str, List[Tuple[int, int, int]]],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Return a NEW solution dict with str keys and fresh lists of (immutable) tuples.

    Keys are normalized to ``str`` to match the rest of the codebase (the executor
    stores ``{str(k): list(v)}`` and graph_builder ``str()``-es every solution key),
    so ``setdefault(str(ego_id))`` and the ABORT membership test below see the same
    key type as the stored keys — and an upstream non-str key cannot produce a
    duplicate ego entry.

    Tuples are immutable, so copying the list and normalizing each entry to a
    tuple fully decouples the result from the input — no shared mutable state,
    so the input ``solution`` can never be mutated by an edit below.
    """
    return {str(aid): [tuple(t) for t in (tuples or [])] for aid, tuples in solution.items()}


# =============================================================================
# Effect: meta-action -> plan edit
# =============================================================================

def apply_meta_action(
    solution: Dict[str, List[Tuple[int, int, int]]],
    obs: GraphObservation,
    ego_id: str,
    meta_action: int,          # a MetaAction value (int or MetaAction member)
    node_v: int,               # task-node global index == task_idx
    tasks: List[Task],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Apply one ``<meta_action, node_v>`` decision to ``solution``, returning a new plan.

    Pure function: returns a NEW ``solution`` dict (copied lists, str keys) and
    never mutates the input. See the module docstring for the per-meta-action
    semantics.

    ``node_v`` is a GLOBAL node index, but task nodes use ``task_idx == node index``
    (graph_builder's canonical order), so ``node_v`` IS the ``task_idx`` and indexes
    directly into ``tasks``.

    Args:
        solution: the current allocation ``{agent_id: [(task_idx, step_idx, level), ...]}``.
        obs: the :class:`GraphObservation` the decision was made on (used only for
            logging / target-id traceability — the edit itself is in terms of
            ``task_idx == node_v``).
        ego_id: the deciding agent's id (may NOT yet be a key in ``solution`` — a
            pop-up engagement by an unassigned ego is handled by creating the entry).
        meta_action: a :class:`MetaAction` value (accepts the int or the member).
        node_v: the chosen task node ``== task_idx``; guarded to ``[0, len(tasks))``.
        tasks: the stable Task list (``task_idx`` indexes into it), used to locate
            the ATTACK step index for an inserted assignment.

    Returns:
        A NEW ``solution`` dict reflecting the edit.

    Raises:
        ValueError: if ``node_v`` is out of range, or ``meta_action`` is not a
            valid :class:`MetaAction` value.
    """
    if not (0 <= node_v < len(tasks)):
        raise ValueError(
            f"node_v={node_v} out of range for {len(tasks)} task node(s) "
            f"(must satisfy 0 <= node_v < len(tasks))"
        )

    try:
        action = MetaAction(meta_action)
    except ValueError as exc:
        raise ValueError(
            f"unknown meta_action {meta_action!r}; expected a MetaAction value"
        ) from exc

    ego_key = str(ego_id)
    new_solution = _copy_solution(solution)

    # Target id is for logging/traceability only; the plan edit is by task_idx.
    target_id = (
        obs.task_target_ids[node_v]
        if 0 <= node_v < len(obs.task_target_ids) else "?"
    )

    # --- PLAN_COMPLIANCE: no edit (equal-but-not-same copy) ------------------
    if action is MetaAction.PLAN_COMPLIANCE:
        logger.debug("PLAN_COMPLIANCE ego=%s node=%d (target %s): no-op",
                     ego_key, node_v, target_id)
        return new_solution

    # --- OE: add an ego -> task v assignment --------------------------------
    if action is MetaAction.OPPORTUNISTIC_ENGAGEMENT:
        ego_assignments = new_solution.setdefault(ego_key, [])

        # Idempotent / anti-duplicate: keyed on task_idx, NOT the full tuple. A
        # second insert would compute a different level (min - 1 shifts), so the
        # dedup must be "is the ego already assigned to task v?".
        if any(int(t[0]) == node_v for t in ego_assignments):
            logger.debug("%s ego=%s already assigned to task %d (target %s): idempotent no-op",
                         action.name, ego_key, node_v, target_id)
            return new_solution

        step_idx = _attack_step_index(tasks[node_v])
        existing_levels = [int(t[2]) for t in ego_assignments]
        new_level = (min(existing_levels) - 1) if existing_levels else 0

        ego_assignments.append((int(node_v), int(step_idx), int(new_level)))
        logger.debug("%s ego=%s += (task=%d, step=%d, level=%d) (target %s)",
                     action.name, ego_key, node_v, step_idx, new_level, target_id)
        return new_solution

    # --- SELF_PRESERVATION_ABORT: drop the ego's assignment(s) to task v -----
    if action is MetaAction.SELF_PRESERVATION_ABORT:
        if ego_key in new_solution:
            kept = [t for t in new_solution[ego_key] if int(t[0]) != node_v]
            new_solution[ego_key] = kept  # key retained (possibly empty -> future RTB)
        logger.debug("SELF_PRESERVATION_ABORT ego=%s -= task %d (target %s)",
                     ego_key, node_v, target_id)
        return new_solution

    # Unreachable: every MetaAction member is handled above.
    raise ValueError(f"unhandled MetaAction {action}")  # pragma: no cover


# =============================================================================
# Self-test
# =============================================================================

def _selftest() -> None:
    """Hand-crafted solution + minimal tasks (no solver/BLADE) asserting every effect.

    Run under nlp_env from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.action.graph_effect
    """
    import numpy as np
    from types import SimpleNamespace

    from ...models import Step

    # --- Minimal tasks: task 2 has a non-ATTACK step BEFORE its ATTACK step, so
    #     its attack-step index is 1 (exercises the step_idx lookup, not just 0). ---
    attack0 = Step(None, "t0", [], 1.0, 1, StepKind.ATTACK)
    attack1 = Step(None, "t1", [], 1.0, 1, StepKind.ATTACK)
    pre_step = SimpleNamespace(step_kind=None, target_id="t2-pre")  # not an ATTACK
    attack2 = Step(None, "t2", [], 1.0, 1, StepKind.ATTACK)
    tasks = [
        Task(steps=[attack0], utility=80),
        Task(steps=[attack1], utility=60),
        Task(steps=[pre_step, attack2], utility=50),  # ATTACK at index 1
    ]

    assert _attack_step_index(tasks[0]) == 0
    assert _attack_step_index(tasks[2]) == 1, "ATTACK step must be found at index 1"

    # ego assigned to task 0 (level 0); peer p1 assigned to task 1 (level 0).
    base = {"ego": [(0, 0, 0)], "p1": [(1, 0, 0)]}
    base_snapshot = {"ego": [(0, 0, 0)], "p1": [(1, 0, 0)]}

    # Minimal GraphObservation (only task_target_ids / agent_id are read here).
    obs = GraphObservation(
        task_features=np.zeros((3, 6), dtype=np.float32),
        agent_features=np.zeros((2, 3), dtype=np.float32),
        ego_index=3,
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_type=np.zeros((0,), dtype=np.int64),
        task_target_ids=["t0", "t1", "t2"],
        agent_ids=["ego", "p1"],
        agent_id="ego",
        current_time=0,
        time_norm=0.0,
    )

    print("=" * 72)
    print("graph_effect self-test")
    print("=" * 72)

    # (1) PLAN_COMPLIANCE: equal-but-not-same, no mutation.
    out = apply_meta_action(base, obs, "ego", MetaAction.PLAN_COMPLIANCE, 0, tasks)
    assert out == base, out
    assert out is not base and out["ego"] is not base["ego"]
    print("[1] PLAN_COMPLIANCE: equal-but-not-same copy   OK")

    # (2) OE inserts (v, attack_step_idx, min_level - 1) at step-0 and step-1 targets.
    out_oe1 = apply_meta_action(base, obs, "ego", MetaAction.OPPORTUNISTIC_ENGAGEMENT, 1, tasks)
    assert out_oe1["ego"] == [(0, 0, 0), (1, 0, -1)], out_oe1["ego"]
    out_oe = apply_meta_action(base, obs, "ego", MetaAction.OPPORTUNISTIC_ENGAGEMENT, 2, tasks)
    assert out_oe["ego"] == [(0, 0, 0), (2, 1, -1)], out_oe["ego"]   # step_idx 1, level -1
    print("[2] OE -> (1,0,-1) [step0] / (2,1,-1) [step1]  (min_level-1, correct step_idx)  OK")

    # (3) ABORT removes the ego's tuple(s) to task v; peer untouched.
    out_ab = apply_meta_action(base, obs, "ego", MetaAction.SELF_PRESERVATION_ABORT, 0, tasks)
    assert out_ab["ego"] == [], out_ab["ego"]
    assert out_ab["p1"] == [(1, 0, 0)], out_ab["p1"]
    print("[3] SELF_PRESERVATION_ABORT removes ego->task0, peer untouched   OK")

    # (4) The input solution was never mutated by any call above.
    assert base == base_snapshot, base
    print("[4] input solution never mutated   OK")

    # (5) Unassigned-ego pop-up: entry is created, level 0 (no existing).
    no_ego = {"p1": [(1, 0, 0)]}
    out_new = apply_meta_action(no_ego, obs, "ego", MetaAction.OPPORTUNISTIC_ENGAGEMENT, 2, tasks)
    assert "ego" in out_new and out_new["ego"] == [(2, 1, 0)], out_new
    assert no_ego == {"p1": [(1, 0, 0)]}
    print("[5] unassigned ego: entry created, level 0   OK")

    # (6) Duplicate insertion is idempotent (dedup keyed on task_idx).
    out1 = apply_meta_action(base, obs, "ego", MetaAction.OPPORTUNISTIC_ENGAGEMENT, 2, tasks)
    out2 = apply_meta_action(out1, obs, "ego", MetaAction.OPPORTUNISTIC_ENGAGEMENT, 2, tasks)
    assert out2["ego"] == out1["ego"], (out1["ego"], out2["ego"])
    assert sum(1 for t in out2["ego"] if t[0] == 2) == 1
    print("[6] duplicate insertion idempotent   OK")

    # Guard: node_v out of range raises.
    for bad_v in (-1, len(tasks)):
        try:
            apply_meta_action(base, obs, "ego", MetaAction.PLAN_COMPLIANCE, bad_v, tasks)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for node_v={bad_v}")
    print("[guard] out-of-range node_v raises ValueError   OK")

    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
