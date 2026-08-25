"""
Unit tests for intra-level nearest-neighbor ordering as it exists TODAY:
the shared pure helper `match_aou.utils.scheduling_utils.nearest_neighbor_order`
and its live consumer `GraphPlanExecutor._eligible`.

(Supersedes `tests/test_executor_nn_ordering.py`, which validated the retired
minimal demo executor. The helper moved to the environment-agnostic scheduling
layer unchanged; these tests pin the behaviour the two current consumers -- the
executor and `graph_hidden_placement.predict_route` -- both depend on.)

Covered:

  HELPER (pure, hand-computed, no BLADE / no solver / no torch)
    * greedy nearest-neighbor order from an anchor;
    * deterministic `(task_idx, step_idx)` tie-break at an EXACT distance tie;
    * unlocated assignments appended last in `(task_idx, step_idx)` order,
      without moving the position;
    * no anchor (`start_location is None`) -> located assignments stay in
      deterministic `(task_idx, step_idx)` order and the end location stays the
      input anchor;
    * the returned end location is the last located pick, so callers can chain it.

  EXECUTOR INTEGRATION (`GraphPlanExecutor._eligible`, synthetic observation)
    * `nn_ordering=False` keeps deterministic `(task_idx, step_idx)` order inside
      the current level;
    * `nn_ordering=True` orders greedily from the ACTING EGO'S LIVE POSITION
      (proved by moving only the observation and watching the order change);
    * only the ego's minimum unfinished level is eligible, and consuming it lets
      the next level become eligible;
    * ordering changes the ORDER only -- the SET of executable assignments, and
      the full route consumed level by level, are identical in both modes;
    * a grounded ego (no live position) falls back to deterministic order.

  CONFIRMED-KILL RECONCILIATION (`_reconcile_confirmed` / `reconcile_confirmed_for_ego`
  / `has_open_assignments` -- the GENERALIZED-V1 step-2 extraction, PO2)
    * the extracted helper reproduces the HISTORICAL Phase-2 behaviour exactly: an
      already-gone head inside the ego's own radius is confirmed, its cooldown dropped,
      and the NEXT eligible assignment is acted on in the SAME `next_actions` call;
    * several consecutive already-gone heads are confirmed in ONE call, as before;
    * the public wrapper confirms exactly what Phase 2 would confirm, and is IDEMPOTENT
      -- calling it before Phase 2 leaves the tick's emitted command byte-identical, so
      no ego's command timing changes;
    * the proximity gate and the liveness probe are unchanged: a target that is gone but
      FAR is NOT confirmed (no peer-kill leak), and no peer's `done` is ever touched;
    * a dead or grounded ego confirms nothing;
    * an unexecutable head still blocks the ego's command for the tick;
    * `has_open_assignments` reports the ego's remaining not-done resolvable work.

Run:
    pytest tests/test_graph_executor_nn_ordering.py -q                      (base env)
    conda run -n nlp_env --no-capture-output
        python tests/test_graph_executor_nn_ordering.py                     (nlp_env)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))  # so match_aou.* imports resolve

from match_aou.models import Agent, Capability, Location, Step, StepKind, Task  # noqa: E402
from match_aou.utils.blade_utils.blade_graph_executor import (  # noqa: E402
    GraphPlanExecutor,
)
from match_aou.utils.scheduling_utils import nearest_neighbor_order  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
        return_location=Location(lat, lon),
    )


class _FakeAircraft:
    """The three attributes `_live_location` / `_airborne` read off an observation."""

    def __init__(self, ego_id: str, lat: float, lon: float) -> None:
        self.id = ego_id
        self.latitude = float(lat)
        self.longitude = float(lon)


class _FakeScenario:
    """Minimal observation: airborne egos only. No BLADE, no engine, no stepping."""

    def __init__(self, aircraft) -> None:
        self.aircraft = list(aircraft)
        self.airbases = []

    def moved_to(self, ego_id: str, loc: Location) -> "_FakeScenario":
        """A COPY of this observation with one ego repositioned."""
        return _FakeScenario(
            _FakeAircraft(ac.id, loc.latitude, loc.longitude)
            if str(ac.id) == str(ego_id)
            else _FakeAircraft(ac.id, ac.latitude, ac.longitude)
            for ac in self.aircraft
        )


def _executor(tasks, solution, agents, *, nn: bool) -> GraphPlanExecutor:
    return GraphPlanExecutor(
        tasks=tasks,
        solution=solution,
        agents=agents,
        add_return_to_base=False,
        nn_ordering=nn,
    )


def _consume_route(executor: GraphPlanExecutor, ego_id: str, scenario: _FakeScenario):
    """The order the executor really flies, driven through `_eligible` alone.

    Mirrors `_command_for_ego`: recompute eligibility from the LIVE observation, act
    on the head, confirm that target (the executor's sole `done` signal), advance the
    live position to it, repeat. This exercises the executor's own level-min gating
    and per-ego task resolution -- it is not a second call to the ordering helper.
    """
    order = []
    current = scenario
    for _guard in range(1000):
        eligible = executor._eligible(ego_id, current)
        if not eligible:
            return order
        head = eligible[0]
        step = executor._resolve_step(ego_id, head)
        _assert(step is not None, f"assignment {head} must resolve to a step")
        order.append(tuple(head))
        executor.done.add((ego_id, str(step.target_id)))
        current = current.moved_to(ego_id, step.location)
    raise AssertionError("route consumption did not terminate")


# ---------------------------------------------------------------------------
# The pure helper
# ---------------------------------------------------------------------------

def test_helper_greedy_order_and_end_location() -> None:
    """Greedy nearest-neighbor from the anchor; end location is the last located pick."""
    far, near, mid = (0, 0, 0), (1, 0, 0), (2, 0, 0)
    locs = {far: Location(0.0, 5.0), near: Location(0.0, 1.0), mid: Location(0.0, 3.0)}

    ordered, end = nearest_neighbor_order(
        [far, near, mid],
        location_of=lambda a: locs[a],
        start_location=Location(0.0, 0.0),
    )
    _assert(ordered == [near, mid, far], f"expected near->mid->far, got {ordered}")
    _assert(end is not None and abs(end.longitude - 5.0) < 1e-12,
            f"end location must be the last pick (lon 5.0), got {end}")


def test_helper_end_location_chains_into_the_next_level() -> None:
    """Feeding the returned end location back reproduces a level-to-level chain.

    From (0,0) level 0 is [near(1), far(4)] and ends at lon 4. Level 1 from lon 4
    is [4.2, then 0.5]; a NON-chained restart from (0,0) would give the opposite
    order, so this asserts the chaining itself.
    """
    lvl0 = {(0, 0, 0): Location(0.0, 4.0), (1, 0, 0): Location(0.0, 1.0)}
    lvl1 = {(2, 0, 1): Location(0.0, 0.5), (3, 0, 1): Location(0.0, 4.2)}
    both = {**lvl0, **lvl1}

    first, end = nearest_neighbor_order(
        list(lvl0), location_of=lambda a: both[a], start_location=Location(0.0, 0.0)
    )
    _assert(first == [(1, 0, 0), (0, 0, 0)], f"level 0 order {first}")

    chained, _ = nearest_neighbor_order(
        list(lvl1), location_of=lambda a: both[a], start_location=end
    )
    _assert(chained == [(3, 0, 1), (2, 0, 1)],
            f"level 1 must chain from lon 4.0 and give [t3, t2], got {chained}")

    restarted, _ = nearest_neighbor_order(
        list(lvl1), location_of=lambda a: both[a], start_location=Location(0.0, 0.0)
    )
    _assert(restarted == [(2, 0, 1), (3, 0, 1)],
            "control: an unchained restart must give the OPPOSITE order, else the "
            f"chaining assertion above is vacuous (got {restarted})")


def test_helper_exact_tie_breaks_on_task_then_step() -> None:
    """An EXACT distance tie resolves by `(task_idx, step_idx)`, not input order."""
    same = Location(0.0, 2.0)
    a, b, c = (7, 1, 0), (7, 0, 0), (3, 9, 0)  # all at the identical coordinate
    locs = {a: same, b: same, c: same}

    ordered, _ = nearest_neighbor_order(
        [a, b, c], location_of=lambda x: locs[x], start_location=Location(0.0, 0.0)
    )
    _assert(ordered == [c, b, a], f"tie must sort by (task_idx, step_idx), got {ordered}")

    reversed_input, _ = nearest_neighbor_order(
        [c, b, a], location_of=lambda x: locs[x], start_location=Location(0.0, 0.0)
    )
    _assert(reversed_input == ordered, "the tie-break must not depend on input order")


def test_helper_unlocated_go_last_without_moving_the_position() -> None:
    """Unlocated assignments are appended last in `(task, step)` order."""
    a, b, c, d = (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)
    locs = {a: Location(0.0, 2.0), c: Location(0.0, 1.0), b: None, d: None}

    ordered, end = nearest_neighbor_order(
        [a, b, c, d], location_of=lambda x: locs[x], start_location=Location(0.0, 0.0)
    )
    _assert(ordered == [c, a, b, d], f"expected [c, a, b, d], got {ordered}")
    _assert(end is not None and abs(end.longitude - 2.0) < 1e-12,
            f"unlocated picks must not move the end position, got {end}")


def test_helper_without_anchor_keeps_deterministic_order() -> None:
    """No anchor -> deterministic `(task, step)` order, end location stays the anchor."""
    a, c = (0, 0, 0), (2, 0, 0)
    locs = {a: Location(0.0, 2.0), c: Location(0.0, 1.0)}

    ordered, end = nearest_neighbor_order(
        [c, a], location_of=lambda x: locs[x], start_location=None
    )
    _assert(ordered == [a, c], f"expected deterministic [a, c], got {ordered}")
    _assert(end is None, f"the end location must stay the input anchor (None), got {end}")


# ---------------------------------------------------------------------------
# GraphPlanExecutor integration
# ---------------------------------------------------------------------------

def _single_level_case():
    """One ego at (0,0); three level-0 targets at lon 5 (t0), 1 (t1), 3 (t2)."""
    ego = "AG1"
    tasks = [
        Task(steps=[_step(0.0, 5.0, "t0")], utility=1.0),
        Task(steps=[_step(0.0, 1.0, "t1")], utility=1.0),
        Task(steps=[_step(0.0, 3.0, "t2")], utility=1.0),
    ]
    solution = {ego: [(0, 0, 0), (1, 0, 0), (2, 0, 0)]}
    agents = [_agent(ego, 0.0, 0.0)]
    scenario = _FakeScenario([_FakeAircraft(ego, 0.0, 0.0)])
    return ego, tasks, solution, agents, scenario


def test_executor_legacy_order_is_deterministic_within_the_level() -> None:
    """`nn_ordering=False` -> `(task_idx, step_idx)` order, ignoring the live position."""
    ego, tasks, solution, agents, scenario = _single_level_case()
    executor = _executor(tasks, solution, agents, nn=False)

    _assert(executor._eligible(ego, scenario) == [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            f"legacy order broken: {executor._eligible(ego, scenario)}")
    # Position must be irrelevant in this mode: park the ego on top of the far target.
    far = _FakeScenario([_FakeAircraft(ego, 0.0, 5.0)])
    _assert(executor._eligible(ego, far) == [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            "legacy order must not depend on the live position")


def test_executor_nn_order_follows_the_live_ego_position() -> None:
    """`nn_ordering=True` -> greedy nearest-neighbor seeded from the LIVE position."""
    ego, tasks, solution, agents, scenario = _single_level_case()
    executor = _executor(tasks, solution, agents, nn=True)

    # From (0,0): t1(lon 1) -> t2(lon 3) -> t0(lon 5).
    _assert(executor._eligible(ego, scenario) == [(1, 0, 0), (2, 0, 0), (0, 0, 0)],
            f"NN order from lon 0 broken: {executor._eligible(ego, scenario)}")

    # Move ONLY the observation to lon 4.9: the nearest is now t0, then t2, then t1.
    moved = scenario.moved_to(ego, Location(0.0, 4.9))
    _assert(executor._eligible(ego, moved) == [(0, 0, 0), (2, 0, 0), (1, 0, 0)],
            f"NN order must track the live position: {executor._eligible(ego, moved)}")


def test_executor_grounded_ego_falls_back_to_deterministic_order() -> None:
    """No live position (grounded) -> the helper's no-anchor deterministic order."""
    ego, tasks, solution, agents, _ = _single_level_case()
    executor = _executor(tasks, solution, agents, nn=True)
    grounded = _FakeScenario([])  # ego absent from .aircraft

    _assert(executor._eligible(ego, grounded) == [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
            f"grounded fallback broken: {executor._eligible(ego, grounded)}")


def _two_level_case():
    """Level 0: t0 @ lon 4, t1 @ lon 1. Level 1: t2 @ lon 0.5, t3 @ lon 4.2."""
    ego = "AG2"
    tasks = [
        Task(steps=[_step(0.0, 4.0, "t0")], utility=1.0),
        Task(steps=[_step(0.0, 1.0, "t1")], utility=1.0),
        Task(steps=[_step(0.0, 0.5, "t2")], utility=1.0),
        Task(steps=[_step(0.0, 4.2, "t3")], utility=1.0),
    ]
    solution = {ego: [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1)]}
    agents = [_agent(ego, 0.0, 0.0)]
    scenario = _FakeScenario([_FakeAircraft(ego, 0.0, 0.0)])
    return ego, tasks, solution, agents, scenario


def test_executor_only_the_current_minimum_level_is_eligible() -> None:
    """Level 1 becomes eligible only once every level-0 target is confirmed done."""
    ego, tasks, solution, agents, scenario = _two_level_case()
    executor = _executor(tasks, solution, agents, nn=True)

    first = executor._eligible(ego, scenario)
    _assert({a[2] for a in first} == {0},
            f"only the minimum unfinished level may be eligible, got {first}")
    _assert(len(first) == 2, f"both level-0 assignments must be eligible, got {first}")

    executor.done.add((ego, "t1"))
    still = executor._eligible(ego, scenario)
    _assert(still == [(0, 0, 0)],
            f"one level-0 target left -> level 1 still gated, got {still}")

    executor.done.add((ego, "t0"))
    now = executor._eligible(ego, scenario)
    _assert({a[2] for a in now} == {1},
            f"level 1 must become eligible once level 0 is complete, got {now}")


def test_executor_ordering_changes_order_only_not_the_executed_set() -> None:
    """Both modes fly the SAME assignments; only the sequence differs."""
    ego, tasks, solution, agents, scenario = _two_level_case()
    legacy = _consume_route(_executor(tasks, solution, agents, nn=False), ego, scenario)
    nn = _consume_route(_executor(tasks, solution, agents, nn=True), ego, scenario)

    _assert(legacy == [(0, 0, 0), (1, 0, 0), (2, 0, 1), (3, 0, 1)],
            f"legacy route {legacy}")
    # NN from (0,0): level 0 [t1(1), t0(4)] ends at lon 4; level 1 chains from there,
    # so t3(4.2) precedes t2(0.5) -- an unchained restart would invert it.
    _assert(nn == [(1, 0, 0), (0, 0, 0), (3, 0, 1), (2, 0, 1)], f"NN route {nn}")

    _assert(set(legacy) == set(nn), "ordering must not change the executed SET")
    _assert(sorted(legacy) == sorted(nn), "ordering must not drop or duplicate work")
    _assert([a[2] for a in nn] == sorted(a[2] for a in nn),
            f"levels must stay ascending across the route, got {nn}")


def test_executor_ordering_is_per_ego_and_reads_only_the_acting_ego() -> None:
    """Two egos on one executor order independently from their OWN live positions."""
    tasks = [
        Task(steps=[_step(0.0, 5.0, "t0")], utility=1.0),
        Task(steps=[_step(0.0, 1.0, "t1")], utility=1.0),
    ]
    solution = {"A": [(0, 0, 0), (1, 0, 0)], "B": [(0, 0, 0), (1, 0, 0)]}
    agents = [_agent("A", 0.0, 0.0), _agent("B", 0.0, 6.0)]
    scenario = _FakeScenario([_FakeAircraft("A", 0.0, 0.0), _FakeAircraft("B", 0.0, 6.0)])
    executor = _executor(tasks, solution, agents, nn=True)

    _assert(executor._eligible("A", scenario) == [(1, 0, 0), (0, 0, 0)],
            "A (lon 0) must go to the near target first")
    _assert(executor._eligible("B", scenario) == [(0, 0, 0), (1, 0, 0)],
            "B (lon 6) must go to the far-from-A target first")

    # A's confirmed kill is A's alone: B's eligibility is untouched (no-comms).
    executor.done.add(("A", "t1"))
    _assert(executor._eligible("A", scenario) == [(0, 0, 0)], "A advanced")
    _assert(executor._eligible("B", scenario) == [(0, 0, 0), (1, 0, 0)],
            "a peer's done must not change this ego's eligibility")


# ---------------------------------------------------------------------------
# Confirmed-kill reconciliation (GENERALIZED-V1 step 2, PO2)
# ---------------------------------------------------------------------------

class _KillableScenario(_FakeScenario):
    """`_FakeScenario` plus the liveness probe the confirm-guard calls.

    `get_target` is the ONLY world read `_command_for_ego` makes about a target, and it
    is the fact the executor confirms a kill on. Modelling it as a plain live-id set is
    what lets a test remove a target and watch the confirmation happen.
    """

    def __init__(self, aircraft, targets=None) -> None:
        super().__init__(aircraft)
        # target_id -> unit-ish object; a MISSING key is the engine's "it is gone".
        self.targets = dict(targets or {})

    def get_target(self, target_id):
        return self.targets.get(str(target_id))

    def kill(self, target_id: str) -> None:
        self.targets.pop(str(target_id), None)


def _reconciliation_world(*, nn: bool = False):
    """One ego, THREE assignments in a line, all at level 0, all within its radius.

    The ego sits at (0.0, 0.0) and the targets at 0.1 / 0.2 / 0.3 degrees of longitude
    -- roughly 11 / 22 / 33 km, all comfortably inside the 50 km default radius -- so
    which of them is confirmed is decided by the LIVENESS probe alone and never by the
    proximity gate. Legacy ordering keeps the deterministic `(task_idx, step_idx)` order
    so the head is predictable.
    """
    tasks = [
        Task(steps=[_step(0.0, 0.1, "t0")], utility=10.0),
        Task(steps=[_step(0.0, 0.2, "t1")], utility=10.0),
        Task(steps=[_step(0.0, 0.3, "t2")], utility=10.0),
    ]
    solution = {"ego": [(0, 0, 0), (1, 0, 0), (2, 0, 0)], "peer": [(0, 0, 0)]}
    agents = [_agent("ego", 0.0, 0.0), _agent("peer", 0.0, 0.0)]
    scenario = _KillableScenario(
        [_FakeAircraft("ego", 0.0, 0.0), _FakeAircraft("peer", 0.0, 0.0)],
        targets={"t0": object(), "t1": object(), "t2": object()},
    )
    executor = _executor(tasks, solution, agents, nn=nn)
    return executor, scenario


def test_reconcile_reproduces_the_historical_confirm_and_advance_in_one_call() -> None:
    """PO2. A gone head is confirmed and the NEXT assignment acted on, same Phase-2 call.

    This is the behaviour the extraction had to preserve: `_command_for_ego` confirms
    first, recomputes eligibility, and then issues the command for whatever is now the
    head -- all inside ONE `next_actions`. If the extraction had turned the confirmation
    into a separate pass, the ego would idle for a tick here.
    """
    executor, scenario = _reconciliation_world()
    executor.attack_cooldown[("ego", "t0")] = 17  # a salvo was in flight
    scenario.kill("t0")

    commands = executor.next_actions(scenario)

    _assert(("ego", "t0") in executor.done, "the gone head must be confirmed done")
    _assert(("ego", "t0") not in executor.attack_cooldown,
            "confirming a kill must drop that pair's re-fire cooldown")
    ego_cmds = [c for c in commands if "'ego'" in c]
    _assert(len(ego_cmds) == 1, f"exactly one command per ego per tick, got {ego_cmds}")
    _assert("handle_aircraft_attack('ego', 't1')" == ego_cmds[0],
            f"the ego must act on t1 in the SAME call, got {ego_cmds[0]}")


def test_reconcile_confirms_several_consecutive_gone_heads_in_one_call() -> None:
    """PO2. The confirmation LOOP is preserved: two gone heads retire in one call."""
    executor, scenario = _reconciliation_world()
    scenario.kill("t0")
    scenario.kill("t1")

    commands = executor.next_actions(scenario)

    _assert(("ego", "t0") in executor.done and ("ego", "t1") in executor.done,
            f"both gone heads must be confirmed, got {executor.done}")
    ego_cmds = [c for c in commands if "'ego'" in c]
    _assert(ego_cmds == ["handle_aircraft_attack('ego', 't2')"],
            f"the ego must skip straight to t2, got {ego_cmds}")


def test_reconcile_public_helper_leaves_the_tick_command_byte_identical() -> None:
    """PO2. Calling the helper first changes NOTHING about what the tick emits.

    This is the property the whole post-FD seam rests on: the tick loop may expose a
    damaged ego's confirmations BEFORE Phase 1, and because `done` is monotone the
    Phase-2 pass then finds nothing further -- so the ego emits exactly the command it
    would have emitted anyway, on exactly the same tick, and no other ego is touched.
    """
    control_ex, control_sc = _reconciliation_world()
    control_sc.kill("t0")
    control_commands = control_ex.next_actions(control_sc)

    early_ex, early_sc = _reconciliation_world()
    early_sc.kill("t0")
    confirmed = early_ex.reconcile_confirmed_for_ego("ego", early_sc)
    _assert(confirmed == ("t0",), f"the helper must report exactly t0, got {confirmed}")
    again = early_ex.reconcile_confirmed_for_ego("ego", early_sc)
    _assert(again == (), f"IDEMPOTENT: a second call confirms nothing, got {again}")

    early_commands = early_ex.next_actions(early_sc)
    _assert(early_commands == control_commands,
            f"early reconciliation changed the tick's commands: {early_commands} vs "
            f"{control_commands}")
    _assert(early_ex.done == control_ex.done,
            f"early reconciliation changed `done`: {early_ex.done} vs {control_ex.done}")


def test_reconcile_is_proximity_gated_and_never_reads_peer_state() -> None:
    """PO2 / PO3. A gone target the ego is FAR from is not confirmed, and peers are inert.

    The far target stands in for "a peer killed it while this ego was elsewhere". The
    executor must not learn that, which is exactly why the guard is distance-gated -- and
    it is why a peer's kill can never manufacture a post-FD completion boundary.
    """
    tasks = [Task(steps=[_step(0.0, 5.0, "far")], utility=10.0)]  # ~556 km away
    solution = {"ego": [(0, 0, 0)], "peer": [(0, 0, 0)]}
    agents = [_agent("ego", 0.0, 0.0), _agent("peer", 0.0, 0.0)]
    scenario = _KillableScenario(
        [_FakeAircraft("ego", 0.0, 0.0), _FakeAircraft("peer", 0.0, 0.0)], targets={}
    )  # `far` is ALREADY gone, and out of range
    executor = _executor(tasks, solution, agents, nn=False)

    confirmed = executor.reconcile_confirmed_for_ego("ego", scenario)

    _assert(confirmed == (), f"an out-of-range kill must not be confirmed, got {confirmed}")
    _assert(executor.done == set(), f"nothing may enter `done`, got {executor.done}")
    _assert(executor.has_open_assignments("ego", scenario),
            "the ego still has its (unconfirmed) assignment")
    _assert(executor.has_open_assignments("peer", scenario),
            "the PEER's own state must be untouched by the ego's reconciliation")


def test_reconcile_ignores_a_dead_or_grounded_ego() -> None:
    """PO2. No live position (grounded / removed) and no dead ego confirms anything."""
    executor, scenario = _reconciliation_world()
    scenario.kill("t0")

    grounded = _KillableScenario([_FakeAircraft("peer", 0.0, 0.0)], targets={})
    _assert(executor.reconcile_confirmed_for_ego("ego", grounded) == (),
            "an ego with no live position confirms nothing")

    executor.dead.add("ego")
    _assert(executor.reconcile_confirmed_for_ego("ego", scenario) == (),
            "a dead ego confirms nothing")
    _assert(executor.done == set(), "neither path may mutate `done`")


def test_reconcile_unexecutable_head_still_blocks_the_command() -> None:
    """PO2. The `blocked` path is the original `return None`, unchanged.

    An assignment whose task index does not resolve stopped `_command_for_ego` before the
    extraction, and must still stop it: an ego with an unexecutable head emits nothing
    this tick rather than silently skipping ahead.
    """
    tasks = [Task(steps=[_step(0.0, 0.1, "t0")], utility=10.0)]
    solution = {"ego": [(0, 0, 0), (99, 0, 0)]}
    agents = [_agent("ego", 0.0, 0.0)]
    scenario = _KillableScenario([_FakeAircraft("ego", 0.0, 0.0)], targets={})
    executor = _executor(tasks, solution, agents, nn=False)

    # t0 is gone and in range -> confirmed; the next head (99) does not resolve, so the
    # executor treats it as implicitly satisfied and the ego has no work left.
    commands = executor.next_actions(scenario)
    _assert(("ego", "t0") in executor.done, "the resolvable head is still confirmed")
    _assert(commands == [], f"no command may be emitted this tick, got {commands}")


def test_has_open_assignments_tracks_not_done_resolvable_work() -> None:
    """PO2. The remaining-mission predicate the boundary wake is gated on."""
    executor, scenario = _reconciliation_world()
    _assert(executor.has_open_assignments("ego", scenario), "three assignments open")

    for target_id in ("t0", "t1"):
        scenario.kill(target_id)
        executor.reconcile_confirmed_for_ego("ego", scenario)
    _assert(executor.has_open_assignments("ego", scenario),
            "t2 is still open, so the mission is not over")

    scenario.kill("t2")
    executor.reconcile_confirmed_for_ego("ego", scenario)
    _assert(not executor.has_open_assignments("ego", scenario),
            "every assignment confirmed -> no remaining mission")
    _assert(executor.has_open_assignments("peer", scenario),
            "the peer's own mission is unaffected by the ego finishing")


# ---------------------------------------------------------------------------
# __main__ runner (pytest is absent from nlp_env -- CLAUDE.md Sec 1)
# ---------------------------------------------------------------------------

TESTS = [
    ("helper_greedy_order_and_end_location",
     test_helper_greedy_order_and_end_location),
    ("helper_end_location_chains_into_the_next_level",
     test_helper_end_location_chains_into_the_next_level),
    ("helper_exact_tie_breaks_on_task_then_step",
     test_helper_exact_tie_breaks_on_task_then_step),
    ("helper_unlocated_go_last_without_moving_the_position",
     test_helper_unlocated_go_last_without_moving_the_position),
    ("helper_without_anchor_keeps_deterministic_order",
     test_helper_without_anchor_keeps_deterministic_order),
    ("executor_legacy_order_is_deterministic_within_the_level",
     test_executor_legacy_order_is_deterministic_within_the_level),
    ("executor_nn_order_follows_the_live_ego_position",
     test_executor_nn_order_follows_the_live_ego_position),
    ("executor_grounded_ego_falls_back_to_deterministic_order",
     test_executor_grounded_ego_falls_back_to_deterministic_order),
    ("executor_only_the_current_minimum_level_is_eligible",
     test_executor_only_the_current_minimum_level_is_eligible),
    ("executor_ordering_changes_order_only_not_the_executed_set",
     test_executor_ordering_changes_order_only_not_the_executed_set),
    ("executor_ordering_is_per_ego_and_reads_only_the_acting_ego",
     test_executor_ordering_is_per_ego_and_reads_only_the_acting_ego),
    ("reconcile_reproduces_the_historical_confirm_and_advance_in_one_call",
     test_reconcile_reproduces_the_historical_confirm_and_advance_in_one_call),
    ("reconcile_confirms_several_consecutive_gone_heads_in_one_call",
     test_reconcile_confirms_several_consecutive_gone_heads_in_one_call),
    ("reconcile_public_helper_leaves_the_tick_command_byte_identical",
     test_reconcile_public_helper_leaves_the_tick_command_byte_identical),
    ("reconcile_is_proximity_gated_and_never_reads_peer_state",
     test_reconcile_is_proximity_gated_and_never_reads_peer_state),
    ("reconcile_ignores_a_dead_or_grounded_ego",
     test_reconcile_ignores_a_dead_or_grounded_ego),
    ("reconcile_unexecutable_head_still_blocks_the_command",
     test_reconcile_unexecutable_head_still_blocks_the_command),
    ("has_open_assignments_tracks_not_done_resolvable_work",
     test_has_open_assignments_tracks_not_done_resolvable_work),
]

if __name__ == "__main__":
    failures = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - standalone runner reports every failure
            failures += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)
