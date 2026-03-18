"""topology_utils.py

Utilities for reasoning about precedence relations between tasks.

This module is part of the *core* MATCH-AOU pipeline:
- It operates on task indices and precedence edges (parent -> child).
- It does NOT depend on any simulator (e.g., BLADE).

Key concepts
------------
- A *topological order* is a linear ordering of tasks that respects precedence.
- A *topological level* (a.k.a. layer) groups tasks that can execute in parallel:
    level[v] = 0                           if v has no selected predecessors
    level[v] = 1 + max(level[u] for u->v)  otherwise

tasks with no mutual dependencies may share the same level.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Deque, Dict, Iterable, List, Sequence, Tuple


Edge = Tuple[int, int]


def compute_topological_levels_selected(
    selected_tasks: Iterable[int],
    precedence_relations: Sequence[Edge],
) -> Dict[int, int]:
    """Compute topological *levels* (layers) for selected tasks.

    Tasks with the same level can be executed "in parallel" w.r.t precedence.

    Args:
        selected_tasks: Iterable of task indices that are considered.
        precedence_relations: Edges (u, v) meaning u must precede v.

    Returns:
        Dict task_idx -> level (0, 1, 2, ...)

    Raises:
        ValueError: If there is a cycle among selected tasks.
    """
    selected = set(selected_tasks)

    adj: DefaultDict[int, List[int]] = defaultdict(list)
    indeg: Dict[int, int] = {t: 0 for t in selected}
    level: Dict[int, int] = {t: 0 for t in selected}

    for u, v in precedence_relations:
        if u in selected and v in selected:
            adj[u].append(v)
            indeg[v] += 1

    q: Deque[int] = deque([t for t in selected if indeg[t] == 0])

    processed = 0
    while q:
        u = q.popleft()
        processed += 1
        for v in adj[u]:
            # v cannot be earlier than u+1; take the max across multiple parents.
            if level[v] < level[u] + 1:
                level[v] = level[u] + 1

            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if processed != len(selected):
        remaining = sorted([t for t in selected if indeg[t] > 0])
        raise ValueError(
            "Precedence graph has a cycle among selected tasks. "
            f"Tasks still with indegree>0: {remaining}"
        )

    return level


def levels_to_layers(level_by_task: Dict[int, int]) -> List[List[int]]:
    """Convert a level map into a list of layers (sorted within each layer)."""
    if not level_by_task:
        return []

    max_level = max(level_by_task.values())
    layers: List[List[int]] = [[] for _ in range(max_level + 1)]
    for t, lvl in level_by_task.items():
        layers[lvl].append(t)

    for layer in layers:
        layer.sort()

    return layers
