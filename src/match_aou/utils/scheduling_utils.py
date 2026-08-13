"""scheduling_utils.py

Core post-solve utilities for MATCH-AOU (environment-agnostic).

This module is responsible for dependency-aware ordering and filtering after the
solver produces an allocation solution:
- Keep only tasks that were selected by the solver (y[j] == 1).
- Re-index tasks to a dense list [0..n-1] (solver/consumer-friendly).
- Filter/re-map precedence relations to the new indices.
- Compute topological *levels* (layers) from precedence (parallelizable order).
- Add `level_order` to each Task and to each solution tuple.

It also owns `nearest_neighbor_order`, the pure intra-level travel-ordering helper
(see the section at the bottom of this file). It lives here, and not in a simulator
package, because it is plain spherical geometry over assignment tuples: the SHARED
single implementation used by `GraphPlanExecutor._eligible` (online execution) and by
`graph_hidden_placement.predict_route` (offline route prediction). Those two must not
drift apart, so there is exactly one copy and both import it.

Input / Output
--------------
Input solution format (after solve):
    solution = {agent_id: [(task_idx, step_idx), ...], ...}

Output solution format (after post-processing):
    solution = {agent_id: [(task_idx, step_idx, level_order), ...], ...}

Notes
-----
- `level_order` is an *abstract* precedence layer (0,1,2,...), not a physical time.
- If there are no precedence relations among selected tasks, all tasks will have
  level_order == 0.
- We intentionally do not depend on any simulator (BLADE, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import Location
from .topology_utils import compute_topological_levels_selected, levels_to_layers

# Types
Edge = Tuple[int, int]
Assignment2 = Tuple[int, int]          # (task_idx, step_idx)
Assignment3 = Tuple[int, int, int]     # (task_idx, step_idx, level_order)


@dataclass(frozen=True)
class PostSolveArtifacts:
    """Results of MATCH-AOU post-solve processing."""
    tasks: List  # List[Task] (kept untyped here to avoid import cycles)
    solution: Dict[str, List[Assignment3]]
    precedence_relations: List[Edge]
    level_by_task: Dict[int, int]       # new_task_idx -> level
    layers: List[List[int]]             # tasks grouped by level (new indices)
    old_to_new: Dict[int, int]          # original task index -> new compact index


def _selected_tasks_from_solution(solution: Dict[str, List[Assignment2]]) -> List[int]:
    """Infer selected tasks from the assignment solution (fallback method)."""
    selected = set()
    for assigns in solution.values():
        for task_idx, _step_idx in assigns:
            selected.add(int(task_idx))
    return sorted(selected)


def post_solve_filter_and_level(
    tasks: List,
    solution: Dict[str, List[Assignment2]],
    precedence_relations: Optional[Sequence[Edge]] = None,
    *,
    unselected_tasks: Optional[Iterable[int]] = None,
) -> PostSolveArtifacts:
    """Filter unselected tasks and add topological *level_order* to the solution.

    Args:
        tasks: Original task list (indexed like the solver variables).
        solution: Mapping agent_id -> list[(task_idx, step_idx)].
        precedence_relations: Optional precedence edges (parent, child).
        unselected_tasks: Optional list/iterable of tasks that were not selected.
            If provided and empty -> no filtering/reindexing is needed.
            If provided and non-empty -> those tasks will be removed.

            If not provided -> selection will be inferred from `solution`.

    Returns:
        PostSolveArtifacts with filtered tasks, reindexed solution, filtered/reindexed
        precedence relations, and level ordering.

    Raises:
        ValueError: If precedence among selected tasks contains a cycle.
    """
    precedence_relations = list(precedence_relations or [])

    # Decide which tasks are selected
    if unselected_tasks is None:
        selected_old = _selected_tasks_from_solution(solution)
        needs_reindex = (selected_old != list(range(len(tasks))))
    else:
        unselected_set = {int(t) for t in unselected_tasks}
        needs_reindex = bool(unselected_set)
        selected_old = [i for i in range(len(tasks)) if i not in unselected_set]
    if not selected_old:
        # Nothing selected -> empty outputs but stable types
        return PostSolveArtifacts(
            tasks=[],
            solution={agent_id: [] for agent_id in solution.keys()},
            precedence_relations=[],
            level_by_task={},
            layers=[],
            old_to_new={},
        )

    if not needs_reindex and len(selected_old) == len(tasks):
        # No filtering needed; keep original indexing
        old_to_new = {i: i for i in selected_old}
        new_tasks = tasks
        new_precedence = [(int(a), int(b)) for (a, b) in precedence_relations]
        # levels computed on the full set
        level_by_task = compute_topological_levels_selected(selected_old, new_precedence)
        layers = levels_to_layers(level_by_task)
        new_solution: Dict[str, List[Assignment3]] = _add_level_to_solution_no_reindex(
            solution=solution,
            level_by_task=level_by_task,
        )
        return PostSolveArtifacts(
            tasks=new_tasks,
            solution=new_solution,
            precedence_relations=new_precedence,
            level_by_task=level_by_task,
            layers=layers,
            old_to_new=old_to_new,
        )

    # --- Filtering and reindexing ---
    old_to_new = {old: new for new, old in enumerate(selected_old)}
    new_tasks = [tasks[old] for old in selected_old]

    new_precedence: List[Edge] = []
    for a, b in precedence_relations:
        a_i, b_i = int(a), int(b)
        if a_i in old_to_new and b_i in old_to_new:
            new_precedence.append((old_to_new[a_i], old_to_new[b_i]))

    selected_new = list(range(len(new_tasks)))
    level_by_task = compute_topological_levels_selected(selected_new, new_precedence)
    layers = levels_to_layers(level_by_task)

    # Reindex solution and add level_order
    new_solution: Dict[str, List[Assignment3]] = {}
    for agent_id, assigns in solution.items():
        out: List[Assignment3] = []
        for task_idx, step_idx in assigns:
            t_old = int(task_idx)
            if t_old not in old_to_new:
                continue
            t_new = old_to_new[t_old]
            out.append((t_new, int(step_idx), int(level_by_task[t_new])))
        new_solution[agent_id] = out

    return PostSolveArtifacts(
        tasks=new_tasks,
        solution=new_solution,
        precedence_relations=new_precedence,
        level_by_task=level_by_task,
        layers=layers,
        old_to_new=old_to_new,
    )


def _add_level_to_solution_no_reindex(
    solution: Dict[str, List[Assignment2]],
    level_by_task: Dict[int, int],
) -> Dict[str, List[Assignment3]]:
    """Add level_order to a solution without reindexing tasks."""
    out: Dict[str, List[Assignment3]] = {}
    for agent_id, assigns in solution.items():
        out[agent_id] = [(int(t), int(s), int(level_by_task[int(t)])) for (t, s) in assigns]
    return out

# ---------------------------------------------------------------------------
# Intra-level nearest-neighbor ordering
# ---------------------------------------------------------------------------
# Pure travel-ordering over assignment tuples: no simulator, no solver, no torch,
# no module-global randomness. Relocated here from the retired minimal executor,
# BODY UNCHANGED, so that its two live consumers -- `GraphPlanExecutor._eligible`
# (online) and `graph_hidden_placement.predict_route` (offline route prediction) --
# share one implementation and cannot drift apart.

def nearest_neighbor_order(
    assignments: Sequence[Assignment3],
    *,
    location_of: Callable[[Assignment3], Optional[Location]],
    start_location: Optional[Location],
) -> Tuple[List[Assignment3], Optional[Location]]:
    """Reorder same-level assignments by greedy nearest-neighbor over step target locations.

    ``location_of(assignment) -> Location | None`` resolves each assignment's target
    location. Starting at ``start_location``, repeatedly pick the remaining LOCATED
    assignment minimizing ``Location.distance_to(...)`` (haversine) from the current
    position, tie-break by ``(task_idx, step_idx)``, append it, and advance the current
    position to that location. Unlocated assignments (``location_of`` returns ``None``, or
    ``start_location`` is ``None`` so distance cannot be computed) are appended last in
    ``(task_idx, step_idx)`` order and do not move the position.

    Returns ``(ordered_assignments, end_location)`` where ``end_location`` is the position
    after the last located pick (or ``start_location`` if none were located), so callers can
    chain it as the start of the next level.
    """
    located: List[Assignment3] = []
    unlocated: List[Assignment3] = []
    for a in assignments:
        loc = location_of(a)
        if loc is not None:
            located.append(a)
        else:
            unlocated.append(a)

    ordered: List[Assignment3] = []
    current = start_location
    remaining = list(located)

    # If we have no anchor, distance is undefined: keep located steps in deterministic
    # (task_idx, step_idx) order without advancing position.
    if current is None:
        ordered.extend(sorted(remaining, key=lambda x: (int(x[0]), int(x[1]))))
        remaining = []

    while remaining:
        def _key(a: Assignment3) -> Tuple[float, int, int]:
            return (current.distance_to(location_of(a)), int(a[0]), int(a[1]))

        nxt = min(remaining, key=_key)
        ordered.append(nxt)
        current = location_of(nxt)
        remaining.remove(nxt)

    ordered.extend(sorted(unlocated, key=lambda x: (int(x[0]), int(x[1]))))
    return ordered, current
