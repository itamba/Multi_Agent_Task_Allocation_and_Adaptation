"""graph_episode_setup.py — Stage 0 of the graph RL orchestrator (episode init).

This is the graph-native pre-loop setup: build the BLADE env, solve MATCH-AOU for
the static plan the egos start from, mint the N per-ego private beliefs, and stand
up the ONE executor. It hands the tick-loop (next task) a single ``EpisodeContext``.

Pipeline position:

    [THIS]  ->  tick-loop  ->  (per-trigger)  graph_builder -> encoder
      setup       execute                       -> graph_action -> graph_effect
                                                 -> executor.resync

This module is graph-native. It reuses only the independent domain helpers:
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
    never an alias of A_init (holds even when the split leaves partial == full).
  * ONE ``DETECTION_KM`` feeds the executor now and (later) the builder's
    ``detection_range_km`` — sensing == attack == arrival is a single radius.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ...models import Agent, Task
from ...solvers import MatchAou
from ...utils.scheduling_utils import post_solve_filter_and_level
from ...utils.blade_utils.scenario_factory import (
    create_agents_from_scenario,
    generate_all_enemy_tasks,
)
from ...utils.blade_utils.blade_graph_executor import GraphPlanExecutor
from .belief import Belief

logger = logging.getLogger(__name__)

Assignment = Tuple[int, int, int]  # (task_idx, step_idx, level)

# =============================================================================
# Module constants (single source of truth).
# =============================================================================
# Sensing == attack == arrival is a SINGLE physical radius (see CLAUDE.md). This
# feeds both the executor's arrival_threshold_km and the graph builder's
# detection_range_km — keep them equal.
DETECTION_KM: float = 50.0
# Fraction of tasks the egos start knowing (the partial set); consumed by
# split_tasks (the discovery-chain sampler).
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
# 2. Partial / full split (discovery-chain aware, single-radius)
# =============================================================================

def split_tasks(
    all_tasks: List[Task],
    partial_ratio: float = PARTIAL_RATIO,
    *,
    detection_km: float = DETECTION_KM,
    max_attempts: int = 20,
) -> Tuple[List[Task], List[Task], Dict[str, Any]]:
    """Split the full task set into a partial (known) set and the full (oracle) set.

    Discovery-chain aware: every HIDDEN target keeps at least one KNOWN target
    within ``detection_km`` (great-circle between ``task.steps[0].location``
    points), so a masked target can in principle be discovered once an ego reaches
    a known neighbour and senses it. ``detection_km`` is the SAME unified
    sensing/attack/arrival radius the executor senses at (``arrival_threshold_km``)
    and the generator now builds connectivity at — it is NOT BLADE ``aircraft.range``.
    This is the fix for the old flat split, which measured adjacency at the (larger)
    fleet *radar* range and so could mark a target discoverable at a radius the ego
    never actually senses at, leaving hidden targets silently undiscoverable.

    Algorithm (Layer 2 of the two-layer discovery chain; Layer 1 lives in
    ``scenario_generator._ensure_discovery_chain``):

    1. Build undirected adjacency between tasks: ``i`` and ``j`` are neighbours iff
       ``locs[i].distance_to(locs[j]) <= detection_km``.
    2. Pin "isolated" targets (no neighbour at all) into the known set — there is no
       other path to discover them.
    3. Rejection-sample the remaining known slots up to ``max_attempts`` times. A
       draw is valid iff every hidden target has ≥1 known neighbour. ``clean`` on the
       first attempt, ``resampled`` on a later one, ``warn-fallback`` on exhaustion
       (last draw kept).

    Args:
        all_tasks: every enemy task (each with a ``steps[0].location``).
        partial_ratio: fraction of tasks in the partial (known) set.
        detection_km: the unified discovery radius (== the executor sensing radius).
        max_attempts: cap on rejection-sampling retries before giving up.

    Returns:
        ``(partial_tasks, full_tasks, split_meta)`` where ``full_tasks`` is a copy of
        ``all_tasks`` and ``partial_tasks ⊆ full_tasks``. ``split_meta`` keys:
        ``outcome`` (``"clean" | "resampled" | "exhaust" | "warn-fallback" |
        "no-chain"``), ``attempt``, and the counts ``hidden, known, isolated_pinned,
        partial, full``.
    """
    full_tasks = list(all_tasks)
    n = len(all_tasks)

    # Degenerate: 0 or 1 task — nothing to hide, everything is known.
    if n < 2:
        meta = {
            "outcome": "no-chain", "attempt": 1,
            "hidden": 0, "known": n, "isolated_pinned": 0,
            "partial": n, "full": n,
        }
        return full_tasks, full_tasks, meta

    num_partial = max(1, int(n * partial_ratio))

    # Build undirected adjacency between tasks (by index) at the discovery radius.
    locs = [task.steps[0].location for task in all_tasks]
    neighbors: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if locs[i].distance_to(locs[j]) <= detection_km:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Pin isolated tasks (no neighbour within the discovery radius) to the known set.
    isolated = {i for i in range(n) if not neighbors[i]}

    if len(isolated) > num_partial:
        # More isolated targets than the partial budget: force as many as fit into
        # known; the rest stay hidden and undiscoverable (there is no better draw).
        forced = sorted(isolated)[:num_partial]
        partial_tasks = [all_tasks[i] for i in forced]
        logger.warning(
            "Discovery chain (split): isolated=%d exceeds partial budget=%d; "
            "%d isolated target(s) will be hidden and undiscoverable",
            len(isolated), num_partial, len(isolated) - num_partial,
        )
        meta = {
            "outcome": "exhaust", "attempt": 0,
            "hidden": n - len(partial_tasks), "known": len(partial_tasks),
            "isolated_pinned": min(len(isolated), num_partial),
            "partial": len(partial_tasks), "full": n,
        }
        return partial_tasks, full_tasks, meta

    pool = [i for i in range(n) if i not in isolated]
    extra_needed = num_partial - len(isolated)

    last_known_set: Set[int] = set()
    for attempt in range(1, max_attempts + 1):
        sampled = random.sample(pool, extra_needed) if extra_needed > 0 else []
        known_set = isolated | set(sampled)
        hidden_set = set(range(n)) - known_set

        valid = all(bool(neighbors[h] & known_set) for h in hidden_set)
        last_known_set = known_set
        if valid:
            partial_tasks = [all_tasks[i] for i in sorted(known_set)]
            tag = "clean" if attempt == 1 else f"resampled (attempt {attempt})"
            logger.debug(
                "Discovery chain (split): %s (hidden=%d, known=%d, "
                "isolated_pinned=%d, detection=%.0f km)",
                tag, len(hidden_set), len(known_set), len(isolated), detection_km,
            )
            meta = {
                "outcome": "clean" if attempt == 1 else "resampled",
                "attempt": attempt,
                "hidden": len(hidden_set), "known": len(known_set),
                "isolated_pinned": len(isolated),
                "partial": len(partial_tasks), "full": n,
            }
            return partial_tasks, full_tasks, meta

    # Exhausted retries — keep the last draw and warn.
    partial_tasks = [all_tasks[i] for i in sorted(last_known_set)]
    logger.warning(
        "Discovery chain (split): no valid split after %d attempts; some hidden "
        "target(s) may have no known neighbour (detection=%.0f km)",
        max_attempts, detection_km,
    )
    meta = {
        "outcome": "warn-fallback", "attempt": max_attempts,
        "hidden": n - len(partial_tasks), "known": len(partial_tasks),
        "isolated_pinned": len(isolated),
        "partial": len(partial_tasks), "full": n,
    }
    return partial_tasks, full_tasks, meta


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

    ``observation`` is the ``env.reset()`` observation captured here — the SEED the
    tick-loop needs for its very first per-ego sense (before it has stepped BLADE even
    once). The loop advances its own local ``obs`` from ``env.step`` thereafter and must
    NEVER call ``env.reset()`` again (that would restart the episode and invalidate the
    solve this context is built around); this field exists so it doesn't have to.
    """

    env: Any
    game: Any
    observation: Any
    agents: List[Agent]
    agent_ids: List[str]
    beliefs: Dict[str, Belief]
    executor: GraphPlanExecutor
    a_init: Dict[str, List[Assignment]]
    oracle_solution: Dict[str, List[Assignment]]
    oracle_tasks: List[Task]
    split_meta: Dict[str, Any]
    record: bool = False
    """True iff recording was armed at setup (a ``recording_export_path`` was given);
    the tick-loop drives the recorder (start / step / export) iff this is True."""


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
        record_every_seconds / recording_export_path: passed to ``Game``. Passing a
            ``recording_export_path`` ARMS recording (sets ``EpisodeContext.record``);
            setup itself does NOT start recording — the tick-loop starts / steps /
            exports it. ``record_every_seconds`` throttles the per-tick frame cadence.

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

    # --- 4. Partial / full split (discovery-chain aware) ----------------------
    # detection_km is the SAME radius fed to the executor below (arrival_threshold_km)
    # and the generator's connectivity — so split-adjacency == runtime-sensing by
    # construction. With a real split, partial ⊊ full: A_init covers only the known
    # targets and the hidden ones become discoverable pop-ups.
    partial, full, split_meta = split_tasks(
        all_tasks, partial_ratio, detection_km=detection_km
    )

    # --- 5. Solve the PARTIAL set -> A_init (the static plan egos start from) --
    a_init, belief_tasks, _ = solve_and_normalize(agents, partial)

    # --- 6. Solve the FULL set -> oracle (for the reward chat) ----------------
    # A SEPARATE, independent solve: oracle must never be an alias of a_init, even
    # in the degenerate case where the split leaves partial == full.
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
        observation=obs,  # seed for the tick-loop's first sense; loop never re-resets
        agents=agents,
        agent_ids=agent_ids,
        beliefs=beliefs,
        executor=executor,
        a_init=a_init,
        oracle_solution=oracle_solution,
        oracle_tasks=oracle_tasks,
        split_meta=split_meta,
        # Single source of truth: recording is armed iff an export path was given.
        record=recording_export_path is not None,
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
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # PlaybackRecorder CHARACTER_LIMIT override (historical flat-era convention)

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
    # detection_km=DETECTION_KM: the generator builds discovery connectivity at the
    # SAME radius the split checks and the runtime senses at (single-radius invariant).
    cfg = VariationConfig(
        include_sams=False,
        num_red_airbases=(3, 3),
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.5,
        detection_km=DETECTION_KM,
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

    # (6) REAL SPLIT (Test 2): partial ⊊ full, and A_init covers only KNOWN targets.
    #     The full enemy set is recomputed from the SAME reset observation the setup
    #     split ran on; the belief/A_init task universe must exclude every hidden id.
    meta = ctx.split_meta
    print(f"[split] meta: {meta}")
    assert meta.get("outcome") in {"clean", "resampled", "exhaust", "warn-fallback",
                                    "no-chain"}, meta
    # counts are self-consistent: known + hidden == full == partial + hidden.
    assert meta["known"] + meta["hidden"] == meta["full"], meta
    assert meta["partial"] == meta["known"], meta
    # With 3 airbases + stretch 0.5 the geometry always hides ≥1 target (num_partial=2).
    assert meta["hidden"] >= 1, f"expected hidden>0 with this geometry, got {meta}"
    assert meta["partial"] < meta["full"], f"partial not a strict subset of full: {meta}"

    # A_init covers only known targets: belief/A_init target ids ⊊ the full enemy set.
    full_ids = {
        str(s.target_id)
        for t in generate_all_enemy_tasks(
            ctx.observation, attacking_side_color=ATTACKING_SIDE_COLOR, probability=1.0
        )
        for s in t.steps
    }
    belief_ids = {str(s.target_id) for t in belief_tasks for s in t.steps}
    assert len(full_ids) == meta["full"], (len(full_ids), meta["full"])
    assert belief_ids <= full_ids, (belief_ids - full_ids)
    # allocated-only A_init sees at most the known targets, never all of them.
    assert len(belief_ids) <= meta["known"], (len(belief_ids), meta["known"])
    assert belief_ids != full_ids, "A_init covers ALL targets - hidden ones leaked in!"
    print(f"[6] real split: partial={meta['partial']} < full={meta['full']} "
          f"(hidden={meta['hidden']}); A_init sees {len(belief_ids)}/{len(full_ids)} "
          f"targets, none hidden   OK")

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


def _selftest_split() -> None:
    """Test 1: the discovery-chain split in isolation (no BLADE, no bonmin).

    Hand-builds tasks with known great-circle spacing and asserts the split's red
    lines directly: hidden targets keep a known neighbour, isolated targets are
    pinned, partial ⊆ full with consistent counts, and a tight radius that isolates
    everything falls into the ``exhaust`` path.
    """
    from match_aou.models import Location, Step, StepKind, Task as _Task

    print("=" * 72)
    print("split_tasks unit test (Test 1 - no BLADE/bonmin)")
    print("=" * 72)

    def _mk(lon: float) -> _Task:
        # lat=0 everywhere; at the equator Δlon deg ≈ Δlon * 111.32 km great-circle.
        return _Task(
            steps=[Step(Location(0.0, lon), f"T{lon:g}", [], 1.0, 1, StepKind.ATTACK)],
            utility=80,
        )

    DET = 50.0

    # Two well-separated pairs. Within a pair ≈33 km (≤DET); pairs ≈556 km apart.
    # A naive random draw can hide a whole pair (both hidden, no known neighbour) —
    # the rejection sampler must reject those draws.
    two_pairs = [_mk(0.0), _mk(0.3), _mk(5.0), _mk(5.3)]  # P1={0,1}, P2={2,3}
    L = [t.steps[0].location for t in two_pairs]
    d01, d23, d02 = L[0].distance_to(L[1]), L[2].distance_to(L[3]), L[0].distance_to(L[2])
    assert d01 <= DET and d23 <= DET and d02 > DET, (d01, d23, d02)

    # (a) every hidden target keeps a known neighbour within DET — across many seeds.
    last_meta = None
    for seed in range(50):
        random.seed(seed)
        partial, full, meta = split_tasks(two_pairs, 2.0 / 3.0, detection_km=DET)
        last_meta = meta
        assert full == list(two_pairs) and len(full) == 4
        assert all(any(p is f for f in full) for p in partial)  # partial subset-of full
        known = {i for i, t in enumerate(two_pairs) if any(t is p for p in partial)}
        hidden = set(range(4)) - known
        for h in hidden:
            hl = two_pairs[h].steps[0].location
            assert any(two_pairs[k].steps[0].location.distance_to(hl) <= DET
                       for k in known), \
                f"seed {seed}: hidden {h} has no known neighbour within {DET} km"
        assert meta["known"] == len(partial) == meta["partial"]
        assert meta["hidden"] == 4 - len(partial)
        assert meta["known"] + meta["hidden"] == meta["full"] == 4
        assert meta["outcome"] in ("clean", "resampled")
    print(f"[1a] hidden targets always keep a known neighbour within DET (50 seeds)  OK")
    print(f"     last meta: {last_meta}")

    # (b) isolated target is PINNED to known. Close pair + one far isolated (~2226 km).
    with_isolated = [_mk(0.0), _mk(0.3), _mk(20.0)]
    far = with_isolated[2]
    for seed in range(20):
        random.seed(seed)
        partial, full, meta = split_tasks(with_isolated, 2.0 / 3.0, detection_km=DET)
        assert any(far is p for p in partial), f"seed {seed}: isolated target not pinned"
        assert meta["isolated_pinned"] == 1
    print("[1b] isolated target pinned into the known set (20 seeds)  OK")

    # (c) partial ⊆ full and meta counts consistent.
    random.seed(0)
    partial, full, meta = split_tasks(two_pairs, 2.0 / 3.0, detection_km=DET)
    assert {id(t) for t in partial} <= {id(t) for t in full}
    assert meta["partial"] + meta["hidden"] == meta["full"] == len(full)
    print("[1c] partial subset-of full, meta counts consistent  OK")

    # (d) a tight radius isolates EVERYTHING → exhaust path (isolated > partial budget).
    random.seed(0)
    partial, full, meta = split_tasks(two_pairs, 2.0 / 3.0, detection_km=1.0)
    n = len(two_pairs)
    num_partial = max(1, int(n * 2.0 / 3.0))
    assert meta["outcome"] == "exhaust", meta
    assert meta["isolated_pinned"] == num_partial == len(partial)
    assert meta["hidden"] == n - num_partial
    assert {id(t) for t in partial} <= {id(t) for t in full}
    print(f"[1d] tight radius -> all isolated -> 'exhaust', {num_partial} pinned  OK")

    # (e) degenerate n<2 → 'no-chain', nothing hidden.
    p1, f1, m1 = split_tasks([_mk(0.0)], 2.0 / 3.0, detection_km=DET)
    assert m1["outcome"] == "no-chain" and m1["hidden"] == 0 and len(p1) == 1
    p0, f0, m0 = split_tasks([], 2.0 / 3.0, detection_km=DET)
    assert m0["hidden"] == 0 and p0 == [] and f0 == []
    print("[1e] degenerate n<2 -> 'no-chain', nothing hidden  OK")

    print("-" * 72)
    print("Test 1 (split unit) passed.")


def _selftest_generator() -> None:
    """Test 3: the generator builds discovery connectivity at DETECTION_KM (no bonmin).

    Proves the connectivity-radius SOURCE switched: with ``detection_km`` set the
    stat is exactly that radius; with it ``None`` the legacy ``aircraft.range`` value
    is used (and differs). A geometric spot-check confirms real ≤DETECTION_KM pairs.
    """
    import json
    import tempfile
    from pathlib import Path

    from match_aou.utils.blade_utils.scenario_generator import (
        ScenarioGenerator, VariationConfig,
    )
    from match_aou.models import Location

    print("=" * 72)
    print("generator connectivity radius test (Test 3 - no bonmin)")
    print("=" * 72)

    repo_root = Path(__file__).resolve().parents[4]
    base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
    out_dir = tempfile.mkdtemp(prefix="graph_gen_selftest_")

    gen = ScenarioGenerator(
        base_scenario_path=str(base_scenario), output_dir=out_dir,
        max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    common = dict(include_sams=False, num_red_airbases=(4, 4),
                  randomize_red_airbase_positions=True, stretch_target_ratio=0.5, seed=7)

    # (a) detection_km=DETECTION_KM → connectivity built at exactly that radius.
    gen.generate(episode=0, config=VariationConfig(detection_km=DETECTION_KM, **common))
    stat50 = gen.last_generation_stats["min_radar_km"]
    assert stat50 == DETECTION_KM, stat50
    print(f"[3a] detection_km={DETECTION_KM} -> connectivity radius stat == {stat50}  OK")

    # (b) detection_km=None → legacy aircraft.range-derived radius (and ≠ DETECTION_KM).
    gen.generate(episode=1, config=VariationConfig(detection_km=None, **common))
    legacy = gen.last_generation_stats["min_radar_km"]
    assert legacy > 0 and legacy != DETECTION_KM, legacy
    print(f"[3b] detection_km=None -> legacy radius {legacy:.1f} km (aircraft.range, != 50)  OK")

    # (c) geometric spot-check: with detection_km=50 the generated same-zone targets
    #     actually sit within 50 km of a neighbour (connectivity produced ≤50 km pairs).
    path50 = gen.generate(episode=2, config=VariationConfig(detection_km=DETECTION_KM, **common))
    with open(path50, "r", encoding="utf-8") as f:
        sc = json.load(f)["currentScenario"]
    airbases = gen._get_red_airbases(sc)
    locs = [Location(ab["latitude"], ab["longitude"]) for ab in airbases]
    assert len(locs) >= 2, len(locs)
    # At least one ≤50 km pair must exist (2+2 zones ⇒ each zone is a connected pair).
    close_pairs = sum(
        1 for i in range(len(locs)) for j in range(i + 1, len(locs))
        if locs[i].distance_to(locs[j]) <= DETECTION_KM
    )
    assert close_pairs >= 1, f"no <={DETECTION_KM} km neighbour pair among {len(locs)} targets"
    print(f"[3c] generated targets have {close_pairs} neighbour pair(s) <={DETECTION_KM} km  OK")

    print("-" * 72)
    print("Test 3 (generator connectivity) passed.")


if __name__ == "__main__":
    _selftest_split()       # Test 1 — pure, no BLADE/bonmin
    _selftest_generator()   # Test 3 — generation only, no bonmin
    _selftest()             # existing self-test + Test 2 (bonmin)
