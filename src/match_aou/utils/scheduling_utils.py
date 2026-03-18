"""scheduling_utils.py

Core post-solve utilities for MATCH-AOU (environment-agnostic).

This module is responsible for dependency-aware ordering and filtering after the
solver produces an allocation solution:
- Keep only tasks that were selected by the solver (y[j] == 1).
- Re-index tasks to a dense list [0..n-1] (solver/consumer-friendly).
- Filter/re-map precedence relations to the new indices.
- Compute topological *levels* (layers) from precedence (parallelizable order).
- Add `level_order` to each Task and to each solution tuple.

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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