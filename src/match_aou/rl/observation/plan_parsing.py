"""
Plan Parsing Module

Extract information from agent's execution plan:
    - Which targets are already in the plan
"""

from typing import List

from .observation_types import TargetInfo


def mark_targets_in_plan(
    targets: List[TargetInfo],
    plan,  # List of (task_idx, step_idx, level) tuples for THIS agent
    agent_id: str,  # Current agent ID
    tasks=None,  # List[Task] to extract target IDs
    solution=None  # Full solution dict to check ALL agents' plans
) -> None:
    """
    Mark which targets are already being attacked by ANY agent in the plan.

    This is important for RL decision making:
    - If is_in_plan=True: target already assigned, less critical to attack again
    - If is_in_plan=False: new target, might be high priority

    The function checks the FULL solution (all agents) to see if any agent
    is already attacking each target.

    Args:
        targets: List of TargetInfo objects to mark
        plan: This agent's task assignments (not used directly, but kept for compatibility)
        agent_id: Current agent ID (not used, but kept for compatibility)
        tasks: All task objects (to extract target IDs from action strings)
        solution: Full solution dict {agent_id: [(task_idx, step_idx, level), ...]}
    """
    if not tasks or not solution:
        # Fallback: mark all as not in plan
        for target in targets:
            target.is_in_plan = False
        return

    # Extract all target IDs being attacked by ANY agent
    planned_target_ids = set()

    for other_agent_id, assignments in solution.items():
        for task_idx, step_idx, level in assignments:
            # Validate indices
            if not (0 <= task_idx < len(tasks)):
                continue

            task = tasks[task_idx]

            if not (0 <= step_idx < len(task.steps)):
                continue

            step = task.steps[step_idx]

            # Target id is now an explicit Step field (no action-string parsing).
            target_id = getattr(step, 'target_id', None)
            if target_id:
                planned_target_ids.add(target_id)

    # Mark targets
    for target in targets:
        if target.exists and target.id in planned_target_ids:
            target.is_in_plan = True
        else:
            target.is_in_plan = False
