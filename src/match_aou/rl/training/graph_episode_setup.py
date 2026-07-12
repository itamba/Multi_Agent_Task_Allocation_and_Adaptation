"""graph_episode_setup.py — Stage 0 of the graph RL orchestrator (episode init).

This is the graph-native pre-loop setup: build the BLADE env, solve MATCH-AOU for
the static plan the egos start from, mint the N per-ego private beliefs, and stand
up the ONE executor. It hands the tick-loop (next task) a single ``EpisodeContext``.

Pipeline position:

    [THIS]  ->  tick-loop  ->  (per-trigger)  graph_builder -> encoder
      setup       execute                       -> graph_action -> graph_effect
                                                 -> executor.resync

This module is graph-native and does NOT import from the old flat ``train_full.py``
(which is being deleted). It reuses only the independent domain helpers:
``create_agents_from_scenario`` / ``generate_all_enemy_tasks`` (scenario_factory),
``MatchAou`` + ``post_solve_filter_and_level`` (solver + post-processing),
``GraphPlanExecutor`` (the sole BLADE translation layer), and ``Belief``.

NO-COMMUNICATION FOUNDATION (load-bearing) — enforced by construction here:
  * The N beliefs are MUTUALLY INDEPENDENT (deepcopy tasks + ``_copy_solution``);
    an edit to one ego's belief can never leak into another's.
  * Beliefs / executor see only the ALLOCATED-ONLY normalized task list
    (``solve_and_normalize`` output). Passing the raw ``all_tasks`` would seed an
    unallocated task with no ASSIGNMENT edge that the graph would misread as a pop-up.
  * The partial and full sets are solved TWICE, independently, so the oracle is
    never an alias of A_init (holds even under the identity split stub).
  * ONE ``DETECTION_KM`` feeds the executor now and (later) the builder's
    ``detection_range_km`` — sensing == attack == arrival is a single radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...models import Agent, Task
from ...solvers import MatchAou
from ...utils.scheduling_utils import post_solve_filter_and_level
from ...utils.blade_utils.scenario_factory import (
    create_agents_from_scenario,
    generate_all_enemy_tasks,
)
from ...utils.blade_utils.blade_graph_executor import GraphPlanExecutor
from .belief import Belief

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level)

# =============================================================================
# Module constants (single source of truth; NOT imported from train_full.py).
# =============================================================================
# Sensing == attack == arrival is a SINGLE physical radius (see CLAUDE.md). This
# feeds the executor's arrival_threshold_km now and will feed the graph builder's
# detection_range_km when the orchestrator wires the builder — keep them equal.
DETECTION_KM: float = 50.0
# Fraction of tasks the egos start knowing (the partial set). Mirrors train_full's
# PARTIAL_RATIO default; consumed by split_tasks (a stub for now — identity split).
PARTIAL_RATIO: float = 2.0 / 3.0
MAX_SIM_TICKS: int = 14400
SOLVER_NAME: str = "bonmin"
ATTACKING_SIDE_COLOR: str = "blue"


# =============================================================================
# 1. Solve + normalize (clean rewrite of the old solve wrapper)
# =============================================================================

def solve_and_normalize(
    agents: Sequence[Agent],
    tasks: List[Task],
    precedence_relations: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[Dict[str, List[Assignment]], List[Task], List[int]]:
    """Solve MATCH-AOU and return THE normalized (allocated-only) baseline.

    Runs the MINLP (``risk_factor=0.0`` — the movement budget already charges an
    explicit round-trip, so no reserve margin) then ``post_solve_filter_and_level``,
    which drops unselected tasks, remaps ``task_idx`` to a dense ``[0..n-1]``, and
    stamps a topological ``level`` onto every 3-tuple.

    Returns:
        ``(solution, belief_tasks, unselected)`` where
        ``solution`` is the normalized allocation ``{agent_id: [(task_idx, step_idx,
        level), ...]}`` over ``belief_tasks``, ``belief_tasks`` is the ALLOCATED-ONLY
        task list (never the raw pre-filter list), and ``unselected`` is the raw
        solver-index list of tasks ``y[j] == 0`` (pre-remap; informational).

    Degenerate inputs (no tasks or no agents) and a solver that selects nothing both
    return ``({}, [], all-indices-unselected)`` — an empty normalized baseline, never
    a partially-allocated one.
    """
    precedence = list(precedence_relations or [])
    if not tasks or not agents:
        return {}, [], list(range(len(tasks)))

    model = MatchAou(
        agents=list(agents),
        tasks=tasks,
        precedence_relations=precedence,
        risk_factor=0.0,
    )
    raw_solution, _results, unselected = model.solve(solver_name=SOLVER_NAME)
    if not raw_solution:
        # Solver failed to reach acceptable optimality, or selected nothing.
        return {}, [], list(range(len(tasks)))

    artifacts = post_solve_filter_and_level(
        tasks=tasks,
        solution=raw_solution,
        precedence_relations=precedence,
        unselected_tasks=unselected,
    )
    # artifacts.solution is allocated-only + remapped; artifacts.tasks is the
    # allocated-only task list. This pair is THE normalized baseline.
    return artifacts.solution, artifacts.tasks, unselected


# =============================================================================
# 2. Partial / full split (STUB — real discovery-chain split is a separate task)
# =============================================================================

def split_tasks(
    all_tasks: List[Task],
    partial_ratio: float,
    observation: Any,
    **kw: Any,
) -> Tuple[List[Task], List[Task], Dict[str, Any]]:
    """Split the full task set into a partial (known) set and the full (oracle) set.

    STUB: returns ``(all_tasks, all_tasks, {"stub": True})`` — identity, so partial
    == full and there are no pop-ups. The return SHAPE is final: the real
    rejection-sampling / discovery-chain split slots in behind this signature without
    touching any caller.

    Args:
        all_tasks: every enemy task in the scenario.
        partial_ratio: fraction of tasks in the partial set (ignored by the stub).
        observation: the BLADE observation (needed by the real split for radar
            adjacency; ignored by the stub).
        **kw: forward-compat knobs for the real split (e.g. ``max_attempts``).

    Returns:
        ``(partial_tasks, full_tasks, split_meta)``.
    """
    # TODO(split): real rejection-sampling split — separate task. Must ensure every
    # hidden target has a known same-zone radar neighbour (two-layer discovery chain);
    # return the same (partial, full, meta) shape so callers stay untouched.
    return all_tasks, all_tasks, {"stub": True}


# =============================================================================
# 3. Episode context (the handoff object the tick-loop consumes)
# =============================================================================

@dataclass
class EpisodeContext:
    """Everything the tick-loop needs after Stage-0 setup.

    ``a_init`` is the static plan the egos start from (the seed the beliefs and the
    executor were built from). It is exposed for the tick-loop / reward and so the
    two-independent-solves invariant (oracle is NOT A_init) is observable; the live
    authoritative plans are the per-ego ``beliefs`` and ``executor.plans``.
    """

    env: Any
    game: Any
    agents: List[Agent]
    agent_ids: List[str]
    beliefs: Dict[str, Belief]
    executor: GraphPlanExecutor
    a_init: Dict[str, List[Assignment]]
    oracle_solution: Dict[str, List[Assignment]]
    oracle_tasks: List[Task]
    split_meta: Dict[str, Any]


# =============================================================================
# 4. Episode setup (env + solve + belief/executor construction)
# =============================================================================

def setup_episode(
    scenario_json: str,
    *,
    partial_ratio: float = PARTIAL_RATIO,
    max_episode_steps: int = MAX_SIM_TICKS,
    attacking_side_color: str = ATTACKING_SIDE_COLOR,
    detection_km: float = DETECTION_KM,
    record_every_seconds: Optional[int] = 10,
    recording_export_path: Optional[str] = None,
) -> EpisodeContext:
    """Stand up one episode: BLADE env -> solve -> beliefs + executor.

    Args:
        scenario_json: the scenario JSON *content* (as ``load_scenario`` expects).
        partial_ratio: fraction of tasks the egos start knowing (fed to ``split_tasks``).
        max_episode_steps: BLADE ``max_episode_steps`` (per-episode tick cap).
        attacking_side_color: our side (blue); selects agents and the blue side id.
        detection_km: the unified sensing/attack/arrival radius fed to the executor.
        record_every_seconds / recording_export_path: passed to ``Game`` (setup does
            NOT start recording — the tick-loop owns that).

    Returns:
        An :class:`EpisodeContext` handoff object.

    Raises:
        RuntimeError: if the scenario yields no blue agents or no enemy tasks.
    """
    # BLADE / gymnasium imported lazily (engine boundary): importing Belief or the
    # solve helpers elsewhere must not drag in the engine.
    import gymnasium
    from blade.Game import Game
    from blade.Scenario import Scenario

    # --- 1. Build env exactly as the frozen integration does ------------------
    game = Game(
        current_scenario=Scenario(),
        record_every_seconds=record_every_seconds,
        recording_export_path=recording_export_path or ".",
    )
    game.load_scenario(scenario_json)
    env = gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=max_episode_steps)
    obs, _info = env.reset()

    # --- 2. Identify the blue side; set current_side_id -----------------------
    side_name = attacking_side_color.upper()
    blue_side = None
    for side in getattr(obs, "sides", []) or []:
        if str(getattr(side, "name", "")).upper() == side_name:
            blue_side = side
            break
    if blue_side is not None:
        game.current_side_id = blue_side.id

    # --- 3. Extract blue agents + all enemy tasks -----------------------------
    agents_by_side = create_agents_from_scenario(obs)
    agents = agents_by_side.get(attacking_side_color.lower(), [])
    if not agents:
        raise RuntimeError(
            f"setup_episode: no {attacking_side_color!r} agents in the scenario"
        )
    # probability=1.0: the anti-stacking task-construction default (see CLAUDE.md).
    all_tasks = generate_all_enemy_tasks(
        obs, attacking_side_color=attacking_side_color, probability=1.0
    )
    if not all_tasks:
        raise RuntimeError("setup_episode: no enemy tasks in the scenario")

    # --- 4. Partial / full split (identity stub for now) ----------------------
    partial, full, split_meta = split_tasks(all_tasks, partial_ratio, obs)

    # --- 5. Solve the PARTIAL set -> A_init (the static plan egos start from) --
    a_init, belief_tasks, _ = solve_and_normalize(agents, partial)

    # --- 6. Solve the FULL set -> oracle (for the reward chat) ----------------
    # A SEPARATE, independent solve: oracle must never be an alias of a_init, even
    # under the identity split stub (partial == full).
    oracle_solution, oracle_tasks, _ = solve_and_normalize(agents, full)

    # --- 7. N mutually-independent beliefs, one per ego id --------------------
    # All egos start byte-equal to a_init at t=0, but each belief is a fresh deepcopy
    # of the tasks + a fresh _copy_solution of a_init, so a per-ego edit never leaks.
    beliefs: Dict[str, Belief] = {
        str(a.id): Belief.independent(belief_tasks, a_init) for a in agents
    }

    # --- 8. ONE executor over the normalized (allocated-only) baseline --------
    # The executor fans belief_tasks out to per-ego private task lists internally.
    # arrival_threshold_km == detection_km == the single unified radius.
    executor = GraphPlanExecutor(
        solution=a_init,
        tasks=belief_tasks,
        agents=agents,
        arrival_threshold_km=detection_km,
    )

    agent_ids = [str(a.id) for a in agents]
    return EpisodeContext(
        env=env,
        game=game,
        agents=agents,
        agent_ids=agent_ids,
        beliefs=beliefs,
        executor=executor,
        a_init=a_init,
        oracle_solution=oracle_solution,
        oracle_tasks=oracle_tasks,
        split_meta=split_meta,
    )


# =============================================================================
# Self-test (bonmin path; generates one real scenario, like graph_builder's)
# =============================================================================

def _selftest() -> None:
    """Run ``setup_episode`` on one generated scenario and assert the invariants.

    Run under nlp_env (needs bonmin) from the repo, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.training.graph_episode_setup
    """
    import copy
    import tempfile
    from pathlib import Path

    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # match train_full's deliberate override

    from match_aou.utils.blade_utils.scenario_generator import (
        ScenarioGenerator, VariationConfig,
    )
    from match_aou.models import Location, Step, StepKind, Task as _Task

    repo_root = Path(__file__).resolve().parents[4]
    base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
    out_dir = tempfile.mkdtemp(prefix="graph_setup_selftest_")

    print("=" * 72)
    print("graph_episode_setup self-test")
    print("=" * 72)

    # --- Generate one scenario variation (RED airbases only, no SAMs) ---
    gen = ScenarioGenerator(
        base_scenario_path=str(base_scenario),
        output_dir=out_dir,
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    cfg = VariationConfig(
        include_sams=False,
        num_red_airbases=(3, 3),
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.5,
        seed=0,
    )
    scenario_path = str(gen.generate(episode=0, config=cfg))
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario_json = f.read()

    # --- Independent two-solve check on the helper itself (object non-aliasing) ---
    # setup_episode relies on solve_and_normalize returning a FRESH object per call;
    # prove that here at the unit level before trusting the wired oracle-vs-A_init.
    ctx = setup_episode(
        scenario_json,
        recording_export_path=out_dir,
    )

    agent_ids = ctx.agent_ids
    n_blue = len(agent_ids)
    print(f"[env] blue agents: {n_blue} ({[a[:8] for a in agent_ids]})")
    print(f"[solve] belief tasks (allocated-only): {len(ctx.executor.tasks[agent_ids[0]])}, "
          f"assignments in A_init: {sum(len(v) for v in ctx.a_init.values())}")
    assert n_blue >= 1, "expected at least one blue agent"

    # (1) N beliefs exist (one per blue agent id) and are ALL EQUAL at t=0.
    assert set(ctx.beliefs.keys()) == set(agent_ids), (ctx.beliefs.keys(), agent_ids)
    assert len(ctx.beliefs) == n_blue

    def _tids(tasks: List[Task]) -> List[List[str]]:
        return [[str(s.target_id) for s in t.steps] for t in tasks]

    ref_sol = ctx.beliefs[agent_ids[0]].solution
    ref_tids = _tids(ctx.beliefs[agent_ids[0]].tasks)
    for aid in agent_ids:
        assert ctx.beliefs[aid].solution == ref_sol, f"belief {aid} solution differs at t=0"
        assert _tids(ctx.beliefs[aid].tasks) == ref_tids, f"belief {aid} tasks differ at t=0"
    print(f"[1] {n_blue} beliefs, one per blue agent, all EQUAL at t=0   OK")

    # (2) INDEPENDENCE: mutate belief[A]; every other belief must be byte-unchanged.
    a_id = agent_ids[0]
    # Snapshot every OTHER ego's belief (deep) BEFORE mutating A.
    snap_sol = {aid: copy.deepcopy(ctx.beliefs[aid].solution) for aid in agent_ids}
    snap_len = {aid: len(ctx.beliefs[aid].tasks) for aid in agent_ids}
    snap_tids = {aid: _tids(ctx.beliefs[aid].tasks) for aid in agent_ids}

    # Distinct underlying objects (no shared mutable state between beliefs).
    for other in agent_ids[1:]:
        assert ctx.beliefs[a_id].tasks is not ctx.beliefs[other].tasks
        assert ctx.beliefs[a_id].solution is not ctx.beliefs[other].solution

    # deepcopy defense-in-depth: the shared t=0 Task OBJECTS must be distinct per ego,
    # not shared references (else an in-place Task edit would leak across egos).
    for other in agent_ids[1:]:
        for i in range(len(ctx.beliefs[a_id].tasks)):
            assert ctx.beliefs[a_id].tasks[i] is not ctx.beliefs[other].tasks[i], (
                f"Task object {i} is shared between {a_id} and {other} — deepcopy regressed"
            )

    # Edit A: append a dummy pop-up task AND add an assignment to A's solution.
    dummy = _Task(
        steps=[Step(Location(0.0, 0.0), "DUMMY_POPUP", [], 1.0, 1, StepKind.ATTACK)],
        utility=1,
    )
    ctx.beliefs[a_id].tasks.append(dummy)
    ctx.beliefs[a_id].solution.setdefault(a_id, []).append((999, 0, -1))

    for other in agent_ids[1:]:
        assert ctx.beliefs[other].solution == snap_sol[other], \
            f"INDEPENDENCE VIOLATED: belief {other} solution changed after editing {a_id}"
        assert len(ctx.beliefs[other].tasks) == snap_len[other], \
            f"INDEPENDENCE VIOLATED: belief {other} task count changed after editing {a_id}"
        assert _tids(ctx.beliefs[other].tasks) == snap_tids[other], \
            f"INDEPENDENCE VIOLATED: belief {other} tasks changed after editing {a_id}"
    # A itself did change (sanity: the mutation actually happened).
    assert len(ctx.beliefs[a_id].tasks) == snap_len[a_id] + 1
    assert (999, 0, -1) in ctx.beliefs[a_id].solution[a_id]
    print(f"[2] object-distinct tasks; editing belief[{a_id[:8]}] left all "
          f"{n_blue - 1} peer beliefs byte-identical   OK")

    # (3) belief_tasks is ALLOCATED-ONLY: every task_idx in A_init is a valid index,
    #     and the set of referenced task_idx == the whole belief_tasks range (no orphan
    #     selected-but-unassigned task). Uses the executor's per-ego belief list.
    belief_tasks = ctx.executor.tasks[a_id]  # fanned-out copy of the normalized list
    n_tasks = len(belief_tasks)
    referenced = {int(t[0]) for tuples in ctx.a_init.values() for t in tuples}
    assert n_tasks >= 1, "solver selected no tasks — scenario too hard for a meaningful test"
    assert all(0 <= i < n_tasks for i in referenced), (referenced, n_tasks)
    assert referenced == set(range(n_tasks)), \
        f"allocated-only violated: referenced={sorted(referenced)} vs range(0,{n_tasks})"
    print(f"[3] belief_tasks allocated-only: {n_tasks} tasks, all referenced by A_init   OK")

    # (4) oracle_solution produced and NOT the same object as A_init.
    assert ctx.oracle_solution, "oracle_solution is empty"
    assert ctx.oracle_solution is not ctx.a_init, "oracle is aliased to A_init!"
    # Also object-distinct from every belief's / the executor's live plan dict.
    for aid in agent_ids:
        assert ctx.oracle_solution is not ctx.beliefs[aid].solution
    assert ctx.oracle_solution is not ctx.executor.plans
    print(f"[4] oracle_solution produced, distinct object from A_init "
          f"({sum(len(v) for v in ctx.oracle_solution.values())} assignments)   OK")

    # (5) Executor constructed; is_done() is False at t=0 (work remains, no RTB yet).
    assert isinstance(ctx.executor, GraphPlanExecutor)
    assert ctx.executor.is_done() is False, "executor claims done at t=0 (no work?!)"
    print("[5] executor constructed; is_done() is False at t=0   OK")

    # --- Bonus: solve_and_normalize non-aliasing at the unit level -------------
    # Two calls on the same inputs must return DISTINCT objects (setup relies on this
    # for the two-independent-solves invariant above).
    s1, _t1, _ = solve_and_normalize(ctx.agents, belief_tasks)
    s2, _t2, _ = solve_and_normalize(ctx.agents, belief_tasks)
    assert s1 is not s2, "solve_and_normalize returned an aliased solution object"
    print("[bonus] solve_and_normalize returns a fresh solution object per call   OK")

    ctx.env.close()
    print("-" * 72)
    print("All assertions passed.")


if __name__ == "__main__":
    _selftest()
