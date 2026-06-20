"""
Unit test for WI-4: nearest-neighbor intra-level ordering in BladeExecutorMinimal.

Verifies, on tiny hand-computed synthetic cases (no BLADE, no solver):
  (i)   nn_ordering=False reproduces the legacy (level_order, task_idx, step_idx) sort exactly.
  (ii)  nn_ordering=True yields the hand-computed greedy nearest-neighbor order WITHIN a level,
        ascending-level order BETWEEN levels, and chains the start position level -> level.
  (iii) the SET of issued (task, step, level) assignments is identical between the two modes.
Also exercises the pure `nearest_neighbor_order` helper's unlocated-step handling.

Run: python tests/test_executor_nn_ordering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # so match_aou.* imports resolve

from match_aou.models import Agent, Capability, Location, Step, StepKind, Task  # noqa: E402
from match_aou.utils.blade_utils.blade_executor_minimal import (  # noqa: E402
    BladeExecutorMinimal,
    nearest_neighbor_order,
)


def _step(lat: float, lon: float, target_id: str) -> Step:
    return Step(
        location=Location(lat, lon),
        target_id=target_id,
        capabilities=[Capability("strike")],
        probability=1.0,
        effort=1,
        step_kind=StepKind.ATTACK,
    )


def _agent(aid: str, lat: float, lon: float) -> Agent:
    return Agent(
        location=Location(lat, lon),
        capabilities=[Capability("strike")],
        budget=1e9,
        move_cost_function=lambda a, b: a.distance_to(b),
        agent_id=aid,
    )


def _make_executor(tasks, solution, agents, nn):
    return BladeExecutorMinimal(
        tasks=tasks,
        solution=solution,
        agents=agents,
        add_return_to_base=False,
        nn_ordering=nn,
    )


def test_single_level_three_steps():
    """One agent, 3 located steps, single level: unambiguous NN order."""
    aid = "AG1"
    # Agent at (0,0). Targets along longitude: near=1, mid=3, far=5.
    tasks = [
        Task(steps=[_step(0.0, 5.0, "t0")], utility=1.0),  # task 0: far
        Task(steps=[_step(0.0, 1.0, "t1")], utility=1.0),  # task 1: near
        Task(steps=[_step(0.0, 3.0, "t2")], utility=1.0),  # task 2: mid
    ]
    solution = {aid: [(0, 0, 0), (1, 0, 0), (2, 0, 0)]}
    agents = [_agent(aid, 0.0, 0.0)]

    legacy = _make_executor(tasks, solution, agents, nn=False)
    nn = _make_executor(tasks, solution, agents, nn=True)

    # (i) legacy == (level, task, step) sort
    assert legacy.queue[aid] == [(0, 0, 0), (1, 0, 0), (2, 0, 0)], legacy.queue[aid]

    # (ii) NN from (0,0): near(t1) -> mid(t2) -> far(t0)
    assert nn.queue[aid] == [(1, 0, 0), (2, 0, 0), (0, 0, 0)], nn.queue[aid]

    # (iii) same SET of assignments
    assert set(legacy.queue[aid]) == set(nn.queue[aid])
    print("test_single_level_three_steps: OK")


def test_two_levels_chaining():
    """Two levels: ascending-level between levels, NN within, chained start across levels."""
    aid = "AG2"
    # Agent start (0,0).
    # Level 0: task0 @ lon 4.0 (far), task1 @ lon 1.0 (near).
    #   legacy lvl0 = [t0, t1];  NN from (0,0) = [t1(near), t0(far)], end position = (0,4).
    # Level 1: task2 @ lon 0.5, task3 @ lon 4.2.
    #   legacy lvl1 = [t2, t3].
    #   chained NN from (0,4): t3(4.2, d=0.2) before t2(0.5, d=3.5) -> [t3, t2].
    #   (a NON-chained restart from (0,0) would give [t2, t3] -> so [t3,t2] proves chaining.)
    tasks = [
        Task(steps=[_step(0.0, 4.0, "t0")], utility=1.0),  # task 0, level 0
        Task(steps=[_step(0.0, 1.0, "t1")], utility=1.0),  # task 1, level 0
        Task(steps=[_step(0.0, 0.5, "t2")], utility=1.0),  # task 2, level 1
        Task(steps=[_step(0.0, 4.2, "t3")], utility=1.0),  # task 3, level 1
    ]
    solution = {aid: [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1)]}
    agents = [_agent(aid, 0.0, 0.0)]

    legacy = _make_executor(tasks, solution, agents, nn=False)
    nn = _make_executor(tasks, solution, agents, nn=True)

    # (i) legacy == (level, task, step) sort
    assert legacy.queue[aid] == [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1)], legacy.queue[aid]

    # (ii) NN within + ascending level + chaining: lvl0 [t1,t0], lvl1 chained [t3,t2]
    assert nn.queue[aid] == [(1, 0, 0), (0, 0, 0), (3, 0, 1), (2, 0, 1)], nn.queue[aid]

    # between-level order preserved: all level-0 before all level-1
    levels = [lv for *_, lv in nn.queue[aid]]
    assert levels == sorted(levels), levels

    # (iii) same SET of assignments
    assert set(legacy.queue[aid]) == set(nn.queue[aid])
    print("test_two_levels_chaining: OK")


def test_nearest_neighbor_order_unlocated_last():
    """Pure helper: unlocated steps go last in (task, step) order and don't move position."""
    # Located: a @ lon 2, c @ lon 1. Unlocated: b (None), d (None).
    a = (0, 0, 0)
    b = (1, 0, 0)
    c = (2, 0, 0)
    d = (3, 0, 0)
    locs = {a: Location(0.0, 2.0), c: Location(0.0, 1.0), b: None, d: None}

    ordered, end = nearest_neighbor_order(
        [a, b, c, d],
        location_of=lambda x: locs[x],
        start_location=Location(0.0, 0.0),
    )
    # located NN from (0,0): c(1) then a(2); unlocated b,d appended last in (task,step) order
    assert ordered == [c, a, b, d], ordered
    # end position = last located pick (a) -> lon 2.0
    assert end is not None and abs(end.longitude - 2.0) < 1e-9, end

    # No anchor: located kept in (task, step) order, end stays None
    ordered2, end2 = nearest_neighbor_order(
        [a, c], location_of=lambda x: locs[x], start_location=None
    )
    assert ordered2 == [a, c], ordered2
    assert end2 is None
    print("test_nearest_neighbor_order_unlocated_last: OK")


if __name__ == "__main__":
    test_single_level_three_steps()
    test_two_levels_chaining()
    test_nearest_neighbor_order_unlocated_last()
    print("ALL WI-4 NN-ORDERING UNIT TESTS PASSED")
