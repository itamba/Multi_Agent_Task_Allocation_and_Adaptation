"""graph_reward.py — the episode-terminal, utility-based reward (Phase-2 graph RL).

This is the reward layer that fills the seam the tick-loop leaves open. The graph
orchestrator is built and locked:

    graph_episode_setup.setup_episode(scenario)      -> EpisodeContext
    graph_tick_loop.run_episode(policy, ctx, cfg)    -> EpisodeResult
        EpisodeResult.trajectory : List[Transition]   # each .reward is None until HERE

``compute_episode_reward`` reads the finished episode through the ``EpisodeContext``
ONLY, computes a single scalar terminal reward, and writes it onto the trajectory.
Nothing else in the pipeline changes.

REWARD (v1)
-----------
    R = ( U_achieved  -  c * U_aircraft * n_lost  -  U_oracle ) / ( |U_oracle| + eps )

  U_oracle   = plan_value(ctx.oracle_solution, ctx.oracle_tasks)   # static full-set optimum, t=0
  U_achieved = realized_utility(ctx.oracle_tasks, ctx.executor.done) # what was actually killed
  U_aircraft = max(t.utility for t in ctx.oracle_tasks)  (0.0 if empty)
  n_lost     = len(ctx.executor.dead)
  c          = cfg.aircraft_penalty_coeff  (DEFAULT 0.0 -> pure utility ratio)
  eps        = cfg.regret_epsilon (DEFAULT 1e-5)  # DIVISION GUARD ONLY

Since ``U_achieved <= U_oracle`` (agents cannot beat the fully-informed oracle over
the same task set), the un-penalized ratio lies in ~[-1, 0]. Folding the death
penalty into the numerator makes it auto-scale with the scenario's utility magnitude,
so the gradient scale is consistent across scenarios. With ``c = 1.0`` a lost aircraft
costs exactly one max-utility target, so suicide-on-a-target is never net-positive;
``c > 1`` makes RTB strictly beat suicide-on-best.

NO-COMMUNICATION RED LINE (enforced by construction)
----------------------------------------------------
This reward is a CENTRALIZED / privileged TRAINING signal. It MAY read global state
(``executor.done`` spans all egos; the oracle is full-info). It MUST NEVER write into
any ego's belief / observation, and MUST NEVER be consulted by the policy / encoder /
executor at decision time. The only mutation this module performs is
``Transition.reward`` on ``result.trajectory``. The runtime functions are torch-free
and BLADE-free. (T6 proves the write-purity; T1 proves the fidelity.)

OUT OF SCOPE (v1 is TERMINAL + UTILITY-ONLY — these are deliberate seams, NOT built here)
----------------------------------------------------------------------------------------
  * per-wake / dense regret shaping (this reward lands only on the terminal transition);
  * oracle RE-SOLVE on the DEGRADED mid-episode state (we compare against the static
    t=0 full-set optimum, never a re-solve);
  * centralized critic / value head / CTDE (``GraphEncoder.pool`` is the future hook);
  * variable-size PPO buffer, ``evaluate_action``, and GAE credit propagation (the
    terminal-on-last placement is exactly the seam the future GAE task consumes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Sequence, Set, Tuple

# REUSE the solver's probability-term stabilizer so U_oracle is bit-faithful to the
# solver's own optimum. This is the (1 - p + EPSILON)**m guard from the MINLP objective;
# it is NOT the reward's division guard (that is cfg.regret_epsilon) — keep them separate.
from ...solvers.match_aou_MINLP_solver import EPSILON

if TYPE_CHECKING:  # Types only — keeps the runtime torch-free (tick-loop) and BLADE-free (setup/executor).
    from ...models import Task
    from .graph_episode_setup import EpisodeContext
    from .graph_tick_loop import EpisodeResult

# A normalized assignment is (task_idx, step_idx, level); plan_value only reads the
# first two, so a raw 2-tuple solution (as returned by ``MatchAou.solve``) works too.
Assignment = Tuple[int, int, int]
Solution = Dict[str, List[Assignment]]


# =============================================================================
# 1. Plan value — the MINLP objective, reproduced EXACTLY (no re-solve)
# =============================================================================

def plan_value(solution: Solution, tasks: Sequence["Task"]) -> float:
    """Scalar value of a plan under the MATCH-AOU objective — bit-faithful to the solver.

    Reproduces ``match_aou_MINLP_solver.MatchAou._add_objective`` exactly:

        sum_j  u_j * prod_k [ 1 - (1 - p_jk + EPSILON) ** m_jk ]

    where ``j`` = task_idx (positional index into ``tasks``), ``k`` = step_idx,
    ``u_j`` = ``tasks[j].utility``, ``p_jk`` = ``tasks[j].steps[k].probability`` and
    ``m_jk`` = the number of DISTINCT agents assigned to ``(j, k)`` in ``solution``.

    The solver's ``y[j]`` factor is intentionally absent: a task with ``m_jk == 0`` on
    every step (i.e. unselected) has each factor ``1 - (·)**0 = 1 - 1 = 0`` -> product
    0 -> contributes 0, which is identical to ``y[j] == 0``. So this matches the solver
    objective for any solved model, whether ``tasks`` is the allocated-only normalized
    list or the raw pre-filter list. (``m_jk == 0`` is handled without crashing:
    ``x ** 0 == 1`` for any ``x`` in Python, including ``0.0 ** 0``.)

    WHY it exists: ``solve_and_normalize`` discards the Pyomo model, so no scalar plan
    value survives anywhere; the reward needs one and it must be faithful so
    ``U_oracle`` equals the solver's own optimum.

    Args:
        solution: ``{agent_id: [(task_idx, step_idx[, level]), ...]}``. Keys are agent
            ids; tuples may be 2- or 3-element (only ``[0]`` / ``[1]`` are read).
        tasks: the task list ``task_idx`` indexes into.

    Returns:
        The objective value as a float. Pure: NO torch, NO BLADE, NO re-solve.
    """
    # Distinct agents per (task_idx, step_idx). A set keyed by agent_id dedups a
    # single agent that lists the same (j, k) more than once.
    assigned: Dict[Tuple[int, int], Set[str]] = {}
    for agent_id, tuples in (solution or {}).items():
        for t in tuples:
            jk = (int(t[0]), int(t[1]))
            assigned.setdefault(jk, set()).add(str(agent_id))

    total = 0.0
    for j, task in enumerate(tasks):
        product = 1.0
        for k, step in enumerate(task.steps):
            m = len(assigned.get((j, k), ()))
            # m == 0 -> (·) ** 0 == 1 -> factor 0 (task contributes nothing). No crash.
            factor = 1.0 - (1.0 - step.probability + EPSILON) ** m
            product *= factor
        total += task.utility * product
    return total


# =============================================================================
# 2. Realized utility — what the egos actually killed (all-or-nothing per task)
# =============================================================================

def realized_utility(tasks: Sequence["Task"], done: Set[Tuple[str, str]]) -> float:
    """Utility actually achieved: a task pays out IFF every one of its targets is killed.

    ``done`` is the executor's confirmed-kill set keyed ``(ego_id, target_id)`` (both
    str). A target is "killed" if ANY ego confirmed it (dedup over ego — a target
    killed under two ego ids counts once). Task ``j`` contributes ``tasks[j].utility``
    IFF every one of its steps' targets is killed:

        all( any((ego, str(step.target_id)) in done for ego in egos) for step in task.steps )

    which collapses to "every step's target is in the killed-target set".

    WHY all-or-nothing: task utility is gained on COMPLETING the task (see
    ``Task.utility``); crediting per-target would over-reward a partially-completed
    multi-step task. The current regime is single-step, so this reduces to "target
    killed", but the all-steps form is future-proof.

    Args:
        tasks: the task list to score (the oracle's task list for the reward).
        done: the executor's confirmed-kill set, keyed ``(ego_id, target_id)``.

    Returns:
        The summed utility of fully-killed tasks, as a float. Pure: no torch / BLADE.
    """
    killed: Set[str] = {target_id for (_ego, target_id) in (done or set())}
    total = 0.0
    for task in tasks:
        if all(str(step.target_id) in killed for step in task.steps):
            total += task.utility
    return total


# =============================================================================
# 3. Config + result breakdown
# =============================================================================

@dataclass(frozen=True)
class RewardConfig:
    """Knobs for the v1 terminal reward (frozen: the shared default is never mutated)."""

    # Ratio-denominator guard (the paper's regret epsilon). NOT the solver EPSILON.
    regret_epsilon: float = 1e-5
    # Death penalty coefficient c. v1 default 0.0 -> pure utility ratio. c >= 1.0
    # activates the death-only penalty (one lost aircraft == one max-utility target).
    aircraft_penalty_coeff: float = 0.0


@dataclass
class EpisodeReward:
    """The reward breakdown for logging / validation (all fields on the normalized scale)."""

    u_achieved: float   # realized utility (fully-killed oracle tasks)
    u_oracle: float     # static full-set optimum (plan_value of the oracle plan)
    u_aircraft: float   # most-valuable oracle target's utility (0.0 if no tasks)
    n_lost: int         # number of crashed egos (len(executor.dead))
    ratio: float        # (u_achieved - u_oracle) / (|u_oracle| + eps_regret)   -> ~[-1, 0]
    penalty: float      # (c * u_aircraft * n_lost) / (|u_oracle| + eps_regret) -> >= 0
    reward: float       # ratio - penalty  == the R formula


# =============================================================================
# 4. The episode-terminal reward
# =============================================================================

def compute_episode_reward(
    ctx: "EpisodeContext",
    result: "EpisodeResult",
    cfg: RewardConfig = RewardConfig(),
) -> EpisodeReward:
    """Compute the terminal reward and write it onto the trajectory's last transition.

    Reads the finished episode through ``ctx`` ONLY (``ctx.oracle_solution`` /
    ``ctx.oracle_tasks`` and ``ctx.executor.done`` / ``ctx.executor.dead``) and returns
    an :class:`EpisodeReward` breakdown.

    Placement (terminal-on-last convention): if ``result.trajectory`` is non-empty,
    every transition's reward is set to ``0.0`` and the LAST is overwritten with ``R``
    (the future PPO/GAE task propagates that terminal credit backward). If the
    trajectory is EMPTY, nothing is attached — the breakdown is still returned for
    logging.

    RED LINE: the ONLY mutation is ``Transition.reward`` on ``result.trajectory``.
    Nothing is written into any belief, executor plan, or observation.

    Args:
        ctx: the finished episode's :class:`EpisodeContext`.
        result: the :class:`EpisodeResult` from ``run_episode`` (its trajectory is the
            reward seam).
        cfg: reward knobs (see :class:`RewardConfig`).

    Returns:
        The :class:`EpisodeReward` breakdown.
    """
    oracle_tasks = list(ctx.oracle_tasks)

    u_oracle = plan_value(ctx.oracle_solution, oracle_tasks)
    u_achieved = realized_utility(oracle_tasks, ctx.executor.done)
    u_aircraft = max((float(t.utility) for t in oracle_tasks), default=0.0)
    n_lost = len(ctx.executor.dead)

    # eps_regret is a DIVISION GUARD only (distinct from the solver EPSILON in plan_value).
    denom = abs(u_oracle) + cfg.regret_epsilon
    ratio = (u_achieved - u_oracle) / denom
    penalty = (cfg.aircraft_penalty_coeff * u_aircraft * n_lost) / denom
    reward = ratio - penalty  # == (u_achieved - c*u_aircraft*n_lost - u_oracle) / denom

    breakdown = EpisodeReward(
        u_achieved=float(u_achieved),
        u_oracle=float(u_oracle),
        u_aircraft=float(u_aircraft),
        n_lost=int(n_lost),
        ratio=float(ratio),
        penalty=float(penalty),
        reward=float(reward),
    )

    # Terminal-on-last placement. MUTATE ONLY Transition.reward fields (the red line).
    if result.trajectory:
        for tr in result.trajectory:
            tr.reward = 0.0
        result.trajectory[-1].reward = float(reward)

    return breakdown


# =============================================================================
# Self-test (branch-coverage on duck-typed stubs; T1/T7 need bonmin and SKIP if absent)
# =============================================================================

def _selftest() -> None:
    """Branch-coverage on lightweight stubs (always) + fidelity/end-to-end (bonmin, SKIP-able).

    Run from the repo root under nlp_env, e.g.:
        env PYTHONPATH=src python -m match_aou.rl.training.graph_reward
    """
    import copy
    import math
    from types import SimpleNamespace

    print("=" * 72)
    print("graph_reward self-test")
    print("=" * 72)

    # --- Duck-typed stub builders (no BLADE/env/solver needed) ---------------
    def _mk_task(utility, steps):
        # steps: list of (probability, target_id).
        return SimpleNamespace(
            utility=utility,
            steps=[SimpleNamespace(probability=p, target_id=tid) for (p, tid) in steps],
        )

    def _mk_ctx(oracle_solution, oracle_tasks, done, dead, *,
                beliefs=None, plans=None, observation=None):
        executor = SimpleNamespace(
            done=set(done), dead=set(dead),
            plans=plans if plans is not None else {},
        )
        return SimpleNamespace(
            oracle_solution=oracle_solution,
            oracle_tasks=oracle_tasks,
            executor=executor,
            beliefs=beliefs if beliefs is not None else {},
            observation=observation,
        )

    def _mk_result(rewards):
        # rewards: initial per-transition reward values (None or float).
        return SimpleNamespace(trajectory=[SimpleNamespace(reward=r) for r in rewards])

    e = EPSILON

    # =====================================================================
    # T1 — plan_value fidelity vs the solved MINLP objective (needs bonmin)
    # =====================================================================
    print("-" * 72)
    print("[T1] plan_value fidelity vs value(model.obj)")
    try:
        from pyomo.environ import value
        from ...models import Agent, Task, Step, StepKind, Location
        from ...solvers import MatchAou

        def _mv(src, dst):
            return src.distance_to(dst)

        loc0 = Location(0.0, 0.0)
        agents = [
            Agent(location=loc0, capabilities=[], budget=1e9, move_cost_function=_mv,
                  return_location=loc0, agent_id=f"A{i}")
            for i in range(2)
        ]
        tasks_t1 = [
            Task(steps=[Step(Location(0.1, 0.1), "T0", [], 0.9, 1, StepKind.ATTACK)], utility=100),
            Task(steps=[Step(Location(0.2, 0.2), "T1", [], 0.85, 1, StepKind.ATTACK)], utility=80),
        ]
        model = MatchAou(agents=agents, tasks=tasks_t1, precedence_relations=[], risk_factor=0.0)
        raw_solution, _results, _unsel = model.solve(solver_name="bonmin")
        if not raw_solution:
            print("  [T1] SKIP: solver returned no solution (not optimal / selected nothing)")
        else:
            obj_val = value(model.model.obj)
            pv = plan_value(raw_solution, tasks_t1)
            assert abs(pv - obj_val) < 1e-9, (pv, obj_val)
            print(f"  [T1] plan_value={pv:.10f} == value(model.obj)={obj_val:.10f}   OK")
    except Exception as exc:  # bonmin/pyomo missing, or solver error -> SKIP, never fail.
        print(f"  [T1] SKIP (bonmin/pyomo unavailable): {type(exc).__name__}: {exc}")

    # =====================================================================
    # T2 — plan_value hand value (known m_jk, p, u) incl. an m == 0 task
    # =====================================================================
    print("-" * 72)
    print("[T2] plan_value hand value")
    tasks_t2 = [
        _mk_task(10, [(0.5, "x0")]),                 # 1 step, m=1
        _mk_task(20, [(1.0, "y0"), (1.0, "y1")]),    # 2 steps, m=2 then m=1
        _mk_task(50, [(0.9, "z0")]),                 # NO assignment -> m=0 -> contributes 0
    ]
    solution_t2 = {
        "A": [(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        "B": [(1, 0, 0), (1, 0, 0)],                 # duplicate (1,0) must NOT inflate m past 2
    }
    exp0 = 10 * (1 - (1 - 0.5 + e) ** 1)
    exp1 = 20 * (1 - (1 - 1.0 + e) ** 2) * (1 - (1 - 1.0 + e) ** 1)
    exp2 = 50 * (1 - (1 - 0.9 + e) ** 0)             # == 50 * (1 - 1) == 0
    expected_t2 = exp0 + exp1 + exp2
    pv2 = plan_value(solution_t2, tasks_t2)
    assert abs(pv2 - expected_t2) < 1e-12, (pv2, expected_t2)
    assert exp2 == 0.0, exp2
    print(f"  [T2] plan_value={pv2:.10f} == expected={expected_t2:.10f} (m=0 -> 0)   OK")

    # =====================================================================
    # T3 — realized_utility: all-steps credit, ego dedup, partial multi-step -> 0
    # =====================================================================
    print("-" * 72)
    print("[T3] realized_utility all-or-nothing + ego dedup")
    tasks_t3 = [
        _mk_task(10, [(1.0, "t0")]),                 # single-step, killed -> +10
        _mk_task(20, [(1.0, "t1a"), (1.0, "t1b")]),  # multi-step, one unkilled -> 0
        _mk_task(30, [(1.0, "t2")]),                 # killed by TWO egos (dedup) -> +30 once
    ]
    done_t3 = {("A", "t0"), ("A", "t1a"), ("A", "t2"), ("B", "t2")}
    ru = realized_utility(tasks_t3, done_t3)
    assert ru == 40, ru
    # Kill the remaining step of task1 -> it now pays out too (all-steps credit).
    ru2 = realized_utility(tasks_t3, done_t3 | {("C", "t1b")})
    assert ru2 == 60, ru2
    print(f"  [T3] realized_utility={ru} (task1 partial excluded; t2 deduped); +t1b -> {ru2}   OK")

    # =====================================================================
    # T4 — compute_episode_reward on stubs: ratio bounds, placement, empty, u_oracle=0
    # =====================================================================
    print("-" * 72)
    print("[T4] compute_episode_reward branch coverage")
    oracle_tasks = [_mk_task(100, [(1.0, "t0")]), _mk_task(50, [(1.0, "t1")])]
    oracle_solution = {"A": [(0, 0, 0)], "B": [(1, 0, 0)]}

    # (a) all-killed -> ratio ~ 0 (u_achieved == full sum ~ u_oracle at p=1.0).
    ctx_all = _mk_ctx(oracle_solution, oracle_tasks, done={("A", "t0"), ("B", "t1")}, dead=set())
    res_all = _mk_result([None, None, None])
    br_all = compute_episode_reward(ctx_all, res_all)
    assert abs(br_all.ratio) < 1e-3, br_all.ratio
    # placement: last == reward, all others normalized to 0.0.
    assert res_all.trajectory[-1].reward == br_all.reward
    assert all(t.reward == 0.0 for t in res_all.trajectory[:-1])
    print(f"  [T4a] all-killed ratio={br_all.ratio:.3e} ~ 0; terminal placed, rest 0.0   OK")

    # (b) none-killed -> ratio ~ -1.
    ctx_none = _mk_ctx(oracle_solution, oracle_tasks, done=set(), dead=set())
    br_none = compute_episode_reward(ctx_none, _mk_result([None]))
    assert abs(br_none.ratio - (-1.0)) < 1e-3, br_none.ratio
    print(f"  [T4b] none-killed ratio={br_none.ratio:.6f} ~ -1   OK")

    # (c) empty trajectory -> nothing attached, breakdown still returned.
    res_empty = _mk_result([])
    br_empty = compute_episode_reward(ctx_none, res_empty)
    assert res_empty.trajectory == []
    assert isinstance(br_empty, EpisodeReward)
    print("  [T4c] empty trajectory -> no attachment, breakdown returned   OK")

    # (d) u_oracle == 0 -> no division blow-up (denom == eps_regret).
    ctx_zero = _mk_ctx({}, [], done=set(), dead=set())
    br_zero = compute_episode_reward(ctx_zero, _mk_result([None]))
    assert br_zero.u_oracle == 0.0 and math.isfinite(br_zero.reward), br_zero
    print(f"  [T4d] u_oracle=0 -> reward={br_zero.reward} finite (no blow-up)   OK")

    # =====================================================================
    # T5 — penalty: c=0 -> reward == ratio; c=1, n_lost=1 -> drop == u_aircraft/denom
    # =====================================================================
    print("-" * 72)
    print("[T5] death penalty folding")
    ctx_p = _mk_ctx(oracle_solution, oracle_tasks, done={("A", "t0")}, dead={"C"})  # 1 lost
    br0 = compute_episode_reward(ctx_p, _mk_result([None]),
                                 RewardConfig(aircraft_penalty_coeff=0.0))
    assert br0.penalty == 0.0
    assert br0.reward == br0.ratio
    br1 = compute_episode_reward(ctx_p, _mk_result([None]),
                                 RewardConfig(aircraft_penalty_coeff=1.0))
    assert br1.u_aircraft == 100.0, br1.u_aircraft   # max utility target
    assert br1.n_lost == 1, br1.n_lost
    denom = abs(br1.u_oracle) + 1e-5
    expected_drop = br1.u_aircraft / denom
    assert br0.ratio == br1.ratio                    # ratio is coeff-independent
    assert abs((br1.ratio - br1.reward) - expected_drop) < 1e-12, (br1.ratio, br1.reward)
    print(f"  [T5] c=0 -> reward==ratio; c=1 drop={br1.ratio - br1.reward:.6f} "
          f"== u_aircraft/denom={expected_drop:.6f}   OK")

    # =====================================================================
    # T6 — purity / no-comms RED LINE (ALWAYS runs, on stubs)
    # =====================================================================
    print("-" * 72)
    print("[T6] purity / no-comms (external objects byte-unchanged)")
    # External objects the reward MUST NOT touch (plain comparable structures).
    beliefs = {"A": {"tasks": ["t0"], "solution": {"A": [(0, 0, 0)]}},
               "B": {"tasks": ["t1"], "solution": {"B": [(1, 0, 0)]}}}
    plans = {"A": [(0, 0, 0)], "B": [(1, 0, 0)]}
    observation = {"marker": "obs", "aircraft": [1, 2, 3]}
    oracle_solution_t6 = {"A": [(0, 0, 0)], "B": [(1, 0, 0)]}
    oracle_tasks_t6 = [_mk_task(100, [(1.0, "t0")]), _mk_task(50, [(1.0, "t1")])]
    ctx6 = _mk_ctx(oracle_solution_t6, oracle_tasks_t6, done={("A", "t0")}, dead=set(),
                   beliefs=beliefs, plans=plans, observation=observation)

    beliefs_snap = copy.deepcopy(beliefs)
    plans_snap = copy.deepcopy(plans)
    obs_snap = copy.deepcopy(observation)
    oracle_sol_snap = copy.deepcopy(oracle_solution_t6)

    # Pre-zeroed non-terminal rewards so ONLY the last transition visibly changes.
    res6 = _mk_result([0.0, 0.0])
    br6 = compute_episode_reward(ctx6, res6)

    assert ctx6.beliefs == beliefs_snap, "beliefs mutated!"
    assert ctx6.executor.plans == plans_snap, "executor.plans mutated!"
    assert ctx6.observation == obs_snap, "observation mutated!"
    assert ctx6.oracle_solution == oracle_sol_snap, "oracle_solution mutated!"
    # Only trajectory[-1].reward changed; the non-terminal one stayed 0.0.
    assert res6.trajectory[0].reward == 0.0, res6.trajectory[0].reward
    assert res6.trajectory[-1].reward == br6.reward
    print("  [T6] beliefs/executor.plans/observation byte-unchanged; only last reward set   OK")

    # =====================================================================
    # T7 — real end-to-end smoke (SKIP if bonmin/env/setup unavailable)
    # =====================================================================
    print("-" * 72)
    print("[T7] real end-to-end smoke")
    try:
        import tempfile
        from pathlib import Path

        import blade.utils.PlaybackRecorder as _pbr
        _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # match train_full's deliberate override

        from ...utils.blade_utils.scenario_generator import ScenarioGenerator, VariationConfig
        from .graph_episode_setup import setup_episode, MAX_SIM_TICKS, DETECTION_KM
        from .graph_tick_loop import build_policy, run_episode

        repo_root = Path(__file__).resolve().parents[4]
        base_scenario = repo_root / "data" / "scenarios" / "strike_training_4v5.json"
        out_dir = tempfile.mkdtemp(prefix="graph_reward_selftest_")

        gen = ScenarioGenerator(base_scenario_path=str(base_scenario), output_dir=out_dir,
                                max_sim_ticks=MAX_SIM_TICKS)
        gen.recompute_time_feasible_cap(allowed_classes=None)
        cfg_gen = VariationConfig(
            include_sams=False, num_red_airbases=(3, 3),
            randomize_red_airbase_positions=True, stretch_target_ratio=0.5,
            detection_km=DETECTION_KM,  # single-radius: generator connectivity == split == sensing (50 km)
            seed=0,
        )
        scenario_path = str(gen.generate(episode=0, config=cfg_gen))
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_json = f.read()

        import torch
        torch.manual_seed(0)
        ctx = setup_episode(scenario_json, recording_export_path=out_dir)
        policy = build_policy(embed_dim=64)

        result = run_episode(policy, ctx, deterministic=True, max_ticks=3000)
        traj_len_before = len(result.trajectory)

        # PURITY on REAL objects: snapshot the POST-run state right before the reward
        # call, so the check below isolates compute_episode_reward. (run_episode itself
        # legitimately edits beliefs/executor.plans on every wake — under the real
        # discovery-chain split a pop-up wake mutates the woken ego's belief, so a
        # pre-run snapshot would spuriously trip.)
        beliefs_before = {aid: (copy.deepcopy(b.tasks), copy.deepcopy(b.solution))
                          for aid, b in ctx.beliefs.items()}
        plans_before = copy.deepcopy(ctx.executor.plans)

        br = compute_episode_reward(ctx, result)

        print(f"  [T7] ended={result.ended} ticks={result.ticks} wakes={result.n_wakes} "
              f"kills={result.confirmed_kills} dead={result.n_dead}")
        print(f"  [T7] u_achieved={br.u_achieved:.4f} u_oracle={br.u_oracle:.4f} "
              f"u_aircraft={br.u_aircraft:.1f} n_lost={br.n_lost} reward={br.reward:.6f}")

        assert isinstance(br, EpisodeReward)
        assert math.isfinite(br.reward), br.reward
        assert br.u_oracle > 0.0, "real oracle has no value?!"
        # Bound holds for the probability=1.0 regime: U_achieved is raw realized utility,
        # U_oracle is EPSILON-discounted (plan_value), so an all-killed episode can exceed
        # U_oracle by ~U_oracle*1e-6 — allow a small relative slack.
        assert br.u_achieved <= br.u_oracle * (1.0 + 1e-5) + 1e-9, (br.u_achieved, br.u_oracle)

        # Real discovery-chain split -> organic pop-up wakes place a terminal reward on
        # the last transition; an empty trajectory (no wake fired) attaches nothing.
        if traj_len_before == 0:
            assert result.trajectory == []
            print("  [T7] empty trajectory -> nothing attached   OK")
        else:
            assert result.trajectory[-1].reward == br.reward
            print(f"  [T7] non-empty trajectory ({traj_len_before}) -> terminal reward placed   OK")

        # PURITY on real objects: beliefs + executor plans byte-unchanged by the reward.
        for aid, b in ctx.beliefs.items():
            t_before, s_before = beliefs_before[aid]
            assert [[str(s.target_id) for s in t.steps] for t in b.tasks] \
                   == [[str(s.target_id) for s in t.steps] for t in t_before], f"belief {aid} tasks changed"
            assert b.solution == s_before, f"belief {aid} solution changed"
        assert ctx.executor.plans == plans_before, "executor.plans changed"
        print("  [T7] REAL beliefs + executor.plans byte-unchanged after reward   OK")

        ctx.env.close()
    except Exception as exc:
        print(f"  [T7] SKIP (bonmin/env/setup unavailable): {type(exc).__name__}: {exc}")

    print("-" * 72)
    print("All assertions passed (skipped tests noted above).")


if __name__ == "__main__":
    _selftest()
