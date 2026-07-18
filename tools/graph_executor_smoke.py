"""graph_executor_smoke.py — standalone smoke test for GraphPlanExecutor.

Drives the NEW Phase-2 executor (blade_graph_executor.GraphPlanExecutor) on ONE
real, solved 4v5 scenario, end-to-end in BLADE:

    reset (blue aircraft in airbases)
      -> create agents + tasks  -> hand-built 1-1 plan (isolates the executor)
        -> GraphPlanExecutor(tasks, solution, agents)
          -> loop: actions = executor.next_actions(obs); obs,_ = env.step(actions)

This is the FIRST real exercise of the two new BLADE edits' new forms:
  * targeted launch        launch_aircraft_from_airbase(base, ego)
  * 2-arg attack           handle_aircraft_attack(ego, target)

Asserts:
  * every assigned target ends up destroyed (get_target -> None);
  * each agent launches (becomes airborne) then RTBs (lands into an airbase);
  * executor.is_done() becomes True;
  * no exception / Traceback.

Also instruments the per-(ego,target) kill-confirm latency (ticks from attack-emit
to get_target -> None), which calibrates GraphPlanExecutor.kill_confirm_ticks.

NOTE: `blade` is the editable-installed vendored engine (the two committed edits
are live in nlp_env), so there is NO sys.path hack — one engine for the executor
and every downstream consumer.

RUN (from repo root, under nlp_env so bonmin is on PATH):
    conda run -n nlp_env python tools/graph_executor_smoke.py
"""

from __future__ import annotations

import re
import sys
import tempfile
import traceback
from pathlib import Path

# --- Make src/ importable when run as a plain script. ---
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import gymnasium
from blade.Game import Game
from blade.Scenario import Scenario
import blade.utils.PlaybackRecorder as _pbr
_pbr.CHARACTER_LIMIT = 500 * 1024 * 1024  # PlaybackRecorder CHARACTER_LIMIT override (historical flat-era convention)

from match_aou.models import StepKind
from match_aou.utils.blade_utils.scenario_factory import (
    generate_all_enemy_tasks,
    create_agents_from_scenario,
    _normalize_side_color,
)
from match_aou.utils.blade_utils.scenario_generator import (
    ScenarioGenerator, VariationConfig,
)
from match_aou.utils.blade_utils.blade_graph_executor import GraphPlanExecutor

MAX_SIM_TICKS = 14400
ATTACKING = "blue"


def _build_scenario_env():
    """Generate a RED-airbases-only 4v5 variation, load it, reset the env."""
    base = _REPO_ROOT / "data" / "scenarios" / "strike_training_4v5.json"
    out_dir = tempfile.mkdtemp(prefix="graph_executor_smoke_")
    gen = ScenarioGenerator(
        base_scenario_path=str(base), output_dir=out_dir, max_sim_ticks=MAX_SIM_TICKS,
    )
    gen.recompute_time_feasible_cap(allowed_classes=None)
    cfg = VariationConfig(
        include_sams=False,            # RED airbases only -> no SAM return fire
        num_red_airbases=(3, 3),
        randomize_red_airbase_positions=True,
        stretch_target_ratio=0.5,
        seed=0,
    )
    scenario_path = str(gen.generate(episode=0, config=cfg))

    game = Game(current_scenario=Scenario(), record_every_seconds=10,
                recording_export_path=out_dir)
    with open(scenario_path, "r", encoding="utf-8") as f:
        game.load_scenario(f.read())
    # disable_env_checker: the executor emits a LIST action; the Text action_space
    # checker is irrelevant to this harness (handle_action accepts lists).
    env = gymnasium.make(
        "blade/BLADE-v0", game=game,
        max_episode_steps=MAX_SIM_TICKS, disable_env_checker=True,
    )
    obs, _ = env.reset()

    # Act as BLUE (needed for launch_aircraft_from_airbase's side guard).
    for side in getattr(obs, "sides", []) or []:
        if str(getattr(side, "name", "")).upper() == "BLUE":
            game.current_side_id = side.id
            break
    return game, env, obs


def _assigned_target_ids(solution, tasks):
    tids = set()
    for assignments in solution.values():
        for t_idx, s_idx, _lv in assignments:
            if 0 <= t_idx < len(tasks) and 0 <= s_idx < len(tasks[t_idx].steps):
                tids.add(str(tasks[t_idx].steps[s_idx].target_id))
    return tids


def main() -> int:
    # Diagnostics: prove WHICH blade engine is loaded and that it has the new forms.
    import blade
    import inspect
    print(f"blade engine: {blade.__file__}")
    launch_params = inspect.signature(Game.launch_aircraft_from_airbase).parameters
    if "aircraft_id" not in launch_params:
        print("SMOKE FAIL: loaded blade engine lacks the targeted-launch edit "
              "(launch_aircraft_from_airbase has no aircraft_id param). "
              "The installed nlp_env package is stale — re-sync the vendored engine.")
        return 1
    print("blade engine has the 2-arg targeted launch + 2-arg attack edits. OK\n")

    game, env, obs = _build_scenario_env()

    # --- Build agents + tasks on the RESET observation (aircraft still in airbases,
    #     so the executor exercises the launch path). ---
    blue_agents = create_agents_from_scenario(obs).get(ATTACKING, [])
    tasks_g = generate_all_enemy_tasks(obs, attacking_side_color=ATTACKING)
    if not blue_agents or not tasks_g:
        print(f"SMOKE FAIL: no agents ({len(blue_agents)}) or tasks ({len(tasks_g)})")
        return 1

    # --- Trivial hand-built 1-1 plan (spec-permitted; isolates the executor). ---
    # We deliberately do NOT use the MINLP solver here: on this scenario it produces
    # a redundant/stacked allocation (multiple agents per target). Under no-comms,
    # the redundant agents fly across the map chasing targets a peer already
    # destroyed (confirming the kill only on arrival) and run out of fuel — a
    # SOLVER/scenario "stacking" artifact, not an executor fault (the executor is
    # correctly dumb on fuel by design). A 1-1 reachable assignment is the right
    # executor-isolating input.
    def _attack_idx(task):
        for i, s in enumerate(task.steps):
            if getattr(s, "step_kind", None) == StepKind.ATTACK:
                return i
        return 0

    def _round_trip(agent, loc):
        home = agent.return_location or agent.location
        return agent.move_cost(loc, agent.location) + agent.move_cost(home, loc)

    # Greedy nearest-distinct, fuel-feasible (0.9 budget margin; attacking at the
    # 50 km threshold means actual burn is below this full-round-trip estimate).
    solution_g: dict = {}
    taken: set[int] = set()
    for agent in sorted(blue_agents, key=lambda a: str(a.id)):
        best = None
        for t_idx, task in enumerate(tasks_g):
            if t_idx in taken:
                continue
            loc = task.steps[_attack_idx(task)].location
            rt = _round_trip(agent, loc)
            if rt <= 0.9 * float(agent.budget) and (best is None or rt < best[1]):
                best = (t_idx, rt)
        if best is not None:
            t_idx = best[0]
            taken.add(t_idx)
            solution_g[str(agent.id)] = [(t_idx, _attack_idx(tasks_g[t_idx]), 0)]

    if not solution_g:
        print("SMOKE FAIL: no fuel-feasible 1-1 assignment found.")
        return 1

    assigned_tids = _assigned_target_ids(solution_g, tasks_g)
    assigned_agent_ids = sorted(solution_g.keys())
    print(f"Setup: {len(blue_agents)} blue agents, {len(tasks_g)} tasks, "
          f"hand-built 1-1 plan: {len(assigned_agent_ids)} agents assigned, "
          f"{len(assigned_tids)} distinct targets.")
    for aid, assigns in solution_g.items():
        labels = [str(tasks_g[t].steps[s].target_id)[:8] for t, s, _ in assigns]
        print(f"  plan {str(aid)[:8]} -> {labels}")

    # --- Build the executor and drive the loop. ---
    executor = GraphPlanExecutor(
        tasks=tasks_g, solution=solution_g, agents=blue_agents,
        add_return_to_base=True, arrival_threshold_km=50.0, nn_ordering=True,
    )
    # Launch / RTB assertions apply only to agents that received a target.
    agent_ids = assigned_agent_ids

    # Kill-confirm latency instrumentation (DECISION 2): record the FIRST attack-emit
    # tick per (ego, target) and the tick the target is first observed gone, so we can
    # report the emit->confirm latency that calibrates kill_confirm_ticks.
    _ATTACK_RE = re.compile(r"handle_aircraft_attack\('([^']+)', '([^']+)'\)")
    emit_tick: dict[tuple[str, str], int] = {}
    kill_tick: dict[tuple[str, str], int] = {}

    ever_airborne: set[str] = set()
    last_tick = -1
    for tick in range(MAX_SIM_TICKS):
        last_tick = tick
        actions = executor.next_actions(obs)
        airborne_now = {str(getattr(ac, "id", "")) for ac in getattr(obs, "aircraft", []) or []}
        ever_airborne |= (airborne_now & set(agent_ids))

        # Record first attack-emit per (ego, target).
        for act in actions:
            m = _ATTACK_RE.fullmatch(act)
            if m:
                key = (m.group(1), m.group(2))
                emit_tick.setdefault(key, tick)

        obs, _, terminated, truncated, _ = env.step(actions)

        # Record first confirmed kill per emitted (ego, target).
        for key, et in emit_tick.items():
            if key not in kill_tick and obs.get_target(key[1]) is None:
                kill_tick[key] = tick

        # Stop once: plan done (is_done), in-flight weapons resolved the kills, AND
        # every assigned agent has physically left the air (landed or crashed).
        # NOTE: is_done() latches at RTB *issue*, not landing — so we additionally
        # wait for the agents to actually clear scenario.aircraft (fly home + land).
        airborne_ids = {str(getattr(ac, "id", "")) for ac in getattr(obs, "aircraft", []) or []}
        targets_gone = all(obs.get_target(t) is None for t in assigned_tids)
        all_cleared_air = all(a not in airborne_ids for a in agent_ids)
        if executor.is_done() and targets_gone and all_cleared_air:
            print(f"Loop finished at tick {tick}: is_done=True, all assigned targets "
                  f"destroyed, all assigned agents cleared the air.")
            break
        if terminated or truncated:
            print(f"Env ended at tick {tick}: terminated={terminated}, truncated={truncated}")
            break

    # --- Assertions ---
    ok = True

    targets_gone = {t: (obs.get_target(t) is None) for t in assigned_tids}
    survivors = [t for t, gone in targets_gone.items() if not gone]
    if survivors:
        ok = False
        print(f"ASSERT FAIL: {len(survivors)} assigned target(s) NOT destroyed: "
              f"{[s[:8] for s in survivors]}")
    else:
        print(f"OK: all {len(assigned_tids)} assigned targets destroyed.")

    not_launched = [a for a in agent_ids if a not in ever_airborne]
    if not_launched:
        ok = False
        print(f"ASSERT FAIL: agent(s) never airborne: {[a[:8] for a in not_launched]}")
    else:
        print(f"OK: all {len(agent_ids)} agents launched.")

    # RTB == left scenario.aircraft (no longer airborne) AND now in an airbase inventory.
    final_airborne = {str(getattr(ac, "id", "")) for ac in getattr(obs, "aircraft", []) or []}
    in_airbase = set()
    for base in getattr(obs, "airbases", []) or []:
        for ac in getattr(base, "aircraft", []) or []:
            in_airbase.add(str(getattr(ac, "id", "")))
    not_rtb = [a for a in agent_ids if a in final_airborne or a not in in_airbase]
    if not_rtb:
        ok = False
        print(f"ASSERT FAIL: agent(s) did not RTB into an airbase: {[a[:8] for a in not_rtb]}")
    else:
        print(f"OK: all {len(agent_ids)} agents RTB'd into an airbase.")

    if not executor.is_done():
        ok = False
        print("ASSERT FAIL: executor.is_done() is False at end.")
    else:
        print("OK: executor.is_done() is True.")

    # --- Kill-confirm latency report (calibrates kill_confirm_ticks) ---
    print("\nKill-confirm latency (ticks from attack-emit to get_target -> None):")
    latencies = []
    for key in sorted(emit_tick):
        ego, tid = key
        if key in kill_tick:
            lat = kill_tick[key] - emit_tick[key]
            latencies.append(lat)
            print(f"  agent {ego[:8]} -> target {tid[:8]}: "
                  f"emit@{emit_tick[key]} confirm@{kill_tick[key]} = {lat} ticks")
        else:
            print(f"  agent {ego[:8]} -> target {tid[:8]}: emit@{emit_tick[key]} "
                  f"confirm@NEVER (target not confirmed gone)")
    if latencies:
        mx = max(latencies)
        print(f"  MAX kill-confirm latency = {mx} ticks "
              f"(set kill_confirm_ticks just above this; current default "
              f"= {executor.kill_confirm_ticks}).")

    print(f"\n{'SMOKE PASS' if ok else 'SMOKE FAIL'} (ended at tick {last_tick}).")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\nSMOKE FAIL: exception raised (see traceback above).")
        sys.exit(1)
