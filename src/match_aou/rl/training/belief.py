"""belief.py — the per-ego private belief (Phase-2 graph orchestrator).

A ``Belief`` is ONE ego's private, mutable view of the mission:

    tasks     : the ego's Task list (``task_idx`` indexes it) — append-only in
                practice (a pop-up is appended by the trigger layer, never removed).
    solution  : the ego's allocation ``{agent_id: [(task_idx, step_idx, level), ...]}``.

NO-COMMUNICATION FOUNDATION (load-bearing):
  Each ego reasons over its OWN ``Belief``. All N egos start byte-equal to the
  static plan A_init at t=0, but the copies are MUTUALLY INDEPENDENT: appending a
  pop-up to ego A's belief, or editing ego A's ``solution``, must never touch any
  other ego's belief. That independence is what keeps ego A's pop-up / takeover
  invisible to ego B (see ``graph_trigger`` / ``graph_effect``, which are pure and
  edit a single ego's belief copy).

This module is a small leaf on purpose: the reward / PPO layers import ``Belief``
without pulling in the BLADE engine or the solver. The only dependency is the
codebase's canonical solution-copy helper (``graph_effect._copy_solution``), reused
so a belief's ``solution`` copy is byte-identical to the one the effect layer emits
(fresh dict, str keys, fresh lists of immutable 3-tuples).

Extension note: ``Belief`` is a dataclass with exactly the two fields the design
needs now. Later additions (e.g. an ``events`` log or a ``wake_count``) slot in as
new fields WITH DEFAULTS after these two, so existing keyword/positional call sites
(``Belief(tasks=..., solution=...)``) keep working unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ...models import Task
# Canonical solution copy (fresh dict, str keys, fresh lists of immutable tuples).
# Reused so a Belief's solution copy matches graph_effect's output exactly; the task
# spec explicitly sanctions importing this helper here.
from ..action.graph_effect import _copy_solution

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level)


@dataclass
class Belief:
    """One ego's private (tasks, solution) view. See the module docstring."""

    tasks: List[Task]
    solution: Dict[str, List[Assignment]]

    @classmethod
    def independent(
        cls,
        tasks: List[Task],
        solution: Dict[str, List[Assignment]],
    ) -> "Belief":
        """Build a fully-independent belief from a shared (tasks, solution) baseline.

        ``tasks`` are ``copy.deepcopy``-ed (so appending / editing tasks in the
        resulting belief cannot touch the baseline or any sibling belief built from
        it) and ``solution`` goes through :func:`_copy_solution` (fresh dict, str
        keys, fresh lists of immutable tuples). This is THE way ``setup_episode``
        mints the N per-ego beliefs from the single A_init baseline.
        """
        return cls(tasks=copy.deepcopy(list(tasks)), solution=_copy_solution(solution))

    def independent_copy(self) -> "Belief":
        """Return a mutually-independent copy of THIS belief (no shared mutable state).

        Editing the returned belief (append a pop-up task, add/drop an assignment)
        never mutates ``self``, and vice-versa — the no-communication red line.
        """
        return Belief.independent(self.tasks, self.solution)
