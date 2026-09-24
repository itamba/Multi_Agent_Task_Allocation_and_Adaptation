"""PO5 engineering check: the new actor builder vs the LEGACY builder at the base SHA.

For IDENTICAL supplied private inputs, every legacy output -- the six task columns, the
fuel column, edges, ids, time and the action mask -- must be equal; only the new agent
column 1 (``mission_fuel_slack_norm``) may differ. The legacy builder is loaded verbatim
from Git (``git show <base>:src/match_aou/rl/observation/graph_builder.py``) into its own
module inside the ``match_aou.rl.observation`` package, so its relative imports resolve
against the SAME shared modules the new builder uses.

Two populations:
  1. randomized duck-typed stub worlds (variable k, a, positions, fuels, plans), and
  2. one REAL BLADE world (generated, launched, solved with BONMIN, flown), every blue
     airborne ego, with and without a synthetic own-confirmation set.

Engineering evidence only (no training, no rollout, no benchmark). Run from the repo root
under nlp_env with PYTHONPATH=src:

    python research_evidence/generalized_v2/actor_mission_slack_dev_r1/engineering_checks/legacy_equivalence_check.py --base <sha> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from match_aou.models.location import Location  # noqa: E402
from match_aou.rl.action.graph_action import build_action_mask  # noqa: E402
from match_aou.rl.observation import graph_builder as NEW  # noqa: E402


def _load_legacy(base_sha: str):
    src = subprocess.run(
        ["git", "show", "%s:src/match_aou/rl/observation/graph_builder.py" % base_sha],
        cwd=str(ROOT), capture_output=True, check=True,
    ).stdout
    path = Path(tempfile.mkdtemp(prefix="legacy_builder_")) / "graph_builder_legacy.py"
    path.write_bytes(src)
    name = "match_aou.rl.observation.graph_builder_legacy_%s" % base_sha[:7]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "match_aou.rl.observation"
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    import hashlib
    return mod, hashlib.sha256(src).hexdigest()


def _compare(legacy_obs, new_obs) -> list:
    bad = []
    if not np.array_equal(legacy_obs.task_features, new_obs.task_features):
        bad.append("task_features")
    if legacy_obs.agent_features.shape[1] != 1 or new_obs.agent_features.shape[1] != 2:
        bad.append("agent widths")
    elif not np.array_equal(legacy_obs.agent_features[:, 0], new_obs.agent_features[:, 0]):
        bad.append("fuel column")
    if new_obs.agent_features[1:, :].any():
        bad.append("peer rows not featureless")
    for name in ("edge_index", "edge_type"):
        if not np.array_equal(getattr(legacy_obs, name), getattr(new_obs, name)):
            bad.append(name)
    for name in ("ego_index", "task_target_ids", "agent_ids", "agent_id",
                 "current_time", "time_norm"):
        if getattr(legacy_obs, name) != getattr(new_obs, name):
            bad.append(name)
    if not np.array_equal(build_action_mask(legacy_obs), build_action_mask(new_obs)):
        bad.append("action mask")
    return bad


def _stub_population(LEG, n_cases: int, seed: int) -> dict:
    import test_graph_fuel_damage as FD
    rng = random.Random(seed)
    n_ok, failures, slack = 0, [], []
    for case in range(n_cases):
        n_agents = rng.randint(1, 4)
        ctx = FD._FuelDamageCtx(n_agents=n_agents,
                                target_distance_km=rng.uniform(80.0, 400.0),
                                fuel=rng.uniform(4000.0, 16000.0))
        k_extra = rng.randint(0, 4)
        for j in range(k_extra):
            loc = FD._point_at(FD._BASE, rng.uniform(30.0, 450.0), rng.uniform(0, 360))
            for aid in ctx.agent_ids:
                ctx.beliefs[aid].tasks.append(FD._attack_task("x%d" % j, loc))
        for aid in ctx.agent_ids:
            ac = ctx.scenario.get_aircraft(aid)
            here = FD._point_at(FD._BASE, rng.uniform(0.0, 250.0), rng.uniform(0, 360))
            ac.latitude, ac.longitude = here.latitude, here.longitude
            ac.current_fuel = rng.uniform(0.0, ac.max_fuel)
        ego = rng.choice(ctx.agent_ids)
        belief = ctx.beliefs[ego]
        k = len(belief.tasks)
        sol = {a: list(v) for a, v in belief.solution.items()}
        sol[ego] = sol.get(ego, []) + [
            (j, 0, rng.randint(-1, 2)) for j in range(k) if rng.random() < 0.3]
        confirmed = frozenset(
            FD.graph_tick_loop._assignment_target_id(t, belief.tasks)
            for t in sol[ego] if rng.random() < 0.3)
        kwargs = dict(scenario=ctx.scenario, agent_id=ego, current_plan=sol.get(ego),
                      current_time=rng.randint(0, 5000), tasks=belief.tasks,
                      solution=sol, precedence_relations=[],
                      config=NEW.GraphObservationConfig(
                          detection_range_km=50.0, max_sim_ticks=14400))
        legacy = LEG.build_graph_observation(**kwargs)
        new = NEW.build_graph_observation(**kwargs, mission=NEW.EgoMissionInputs(
            home_base=FD._BASE, confirmed_target_ids=confirmed))
        bad = _compare(legacy, new)
        slack.append(float(new.agent_features[0, 1]))
        if bad:
            failures.append({"case": case, "mismatch": bad})
        else:
            n_ok += 1
    return {"n_cases": n_cases, "n_equal": n_ok, "failures": failures,
            "slack_min": min(slack), "slack_max": max(slack),
            "n_negative_slack": sum(1 for s in slack if s < 0),
            "all_slack_finite": all(math.isfinite(s) for s in slack)}


def _real_blade(LEG) -> dict:
    """One real generated world: launch, BONMIN allocation, fly, compare every blue ego."""
    import gymnasium
    from blade.Game import Game
    from blade.Scenario import Scenario
    import blade.utils.PlaybackRecorder as _pbr
    _pbr.CHARACTER_LIMIT = 500 * 1024 * 1024
    from match_aou.solvers import MatchAou
    from match_aou.utils.scheduling_utils import post_solve_filter_and_level
    from match_aou.utils.blade_utils.scenario_factory import (
        generate_all_enemy_tasks, create_agents_from_scenario, _normalize_side_color)
    from match_aou.utils.blade_utils.scenario_generator import (
        ScenarioGenerator, VariationConfig)

    out_dir = tempfile.mkdtemp(prefix="legacy_eq_blade_")
    gen = ScenarioGenerator(
        base_scenario_path=str(ROOT / "data" / "scenarios" / "strike_training_4v5.json"),
        output_dir=out_dir, max_sim_ticks=14400)
    gen.recompute_time_feasible_cap(allowed_classes=None)
    cfg = VariationConfig(include_sams=False, num_red_airbases=(3, 3),
                          randomize_red_airbase_positions=True,
                          stretch_target_ratio=0.5, seed=0)
    scenario_path = str(gen.generate(episode=0, config=cfg))
    game = Game(current_scenario=Scenario(), record_every_seconds=10,
                recording_export_path=out_dir)
    with open(scenario_path, "r", encoding="utf-8") as f:
        game.load_scenario(f.read())
    env = gymnasium.make("blade/BLADE-v0", game=game, max_episode_steps=14400)
    obs, _ = env.reset()
    agents0 = create_agents_from_scenario(obs).get("blue", [])
    homes = {str(a.id): a.return_location for a in agents0}
    for _ in range(5):
        obs, *_ = env.step("")
    for base in getattr(obs, "airbases", []) or []:
        if _normalize_side_color(getattr(base, "side_color", "")) != "blue":
            continue
        for _ac in list(getattr(base, "aircraft", []) or []):
            obs, *_ = env.step(f"launch_aircraft_from_airbase('{base.id}')")
    for _ in range(10):
        obs, *_ = env.step("")
    tasks = generate_all_enemy_tasks(obs, attacking_side_color="blue")
    blue = create_agents_from_scenario(obs).get("blue", [])
    model = MatchAou(agents=blue, tasks=tasks, precedence_relations=[], risk_factor=0.0)
    raw, results, unselected = model.solve(solver_name="bonmin")
    art = post_solve_filter_and_level(tasks=tasks, solution=raw, precedence_relations=[],
                                      unselected_tasks=unselected)
    tasks_g, sol_g = art.tasks, art.solution
    far = NEW._attack_step(tasks_g[0]).location
    egos = [str(ac.id) for ac in obs.aircraft
            if _normalize_side_color(getattr(ac, "side_color", "")) == "blue"]
    obs, *_ = env.step(f"move_aircraft('{egos[0]}', [[{far.latitude}, {far.longitude}]])")
    for _ in range(120):
        obs, *_ = env.step("")
    rows, failures = [], []
    for ego in [str(ac.id) for ac in obs.aircraft
                if _normalize_side_color(getattr(ac, "side_color", "")) == "blue"]:
        for label, confirmed in (("none", frozenset()),
                                 ("first_own", frozenset(
                                     NEW._attack_step(tasks_g[t]).target_id
                                     for (t, _s, _l) in (sol_g.get(ego) or [])[:1]))):
            kwargs = dict(scenario=obs, agent_id=ego, current_plan=sol_g.get(ego),
                          current_time=135, tasks=tasks_g, solution=sol_g,
                          precedence_relations=[],
                          config=NEW.GraphObservationConfig(detection_range_km=50.0))
            legacy = LEG.build_graph_observation(**kwargs)
            new = NEW.build_graph_observation(**kwargs, mission=NEW.EgoMissionInputs(
                home_base=homes[ego], confirmed_target_ids=confirmed))
            bad = _compare(legacy, new)
            rec = new.mission_slack.as_record()
            rows.append({"ego": ego, "confirmed": label, "k": len(tasks_g),
                         "equal": not bad, "mismatch": bad,
                         "mission_fuel_slack_norm": rec["mission_fuel_slack_norm"],
                         "n_route_stops": rec["n_route_stops"],
                         "route_distance_km": rec["route_distance_km"]})
            if bad:
                failures.append(rows[-1])
    env.close()
    term = getattr(getattr(results, "solver", None), "termination_condition", None)
    return {"solver_termination": str(term if term is not None else type(results).__name__),
            "n_solution_assignments": sum(len(v or []) for v in sol_g.values()),
            "n_egos_compared": len(rows), "rows": rows, "failures": failures}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--no-blade", action="store_true")
    args = ap.parse_args()
    LEG, legacy_sha256 = _load_legacy(args.base)
    report = {"check": "legacy_equivalence_po5", "base_sha": args.base,
              "legacy_builder_blob_sha256": legacy_sha256,
              "stub": _stub_population(LEG, args.cases, seed=20260924)}
    if not args.no_blade:
        report["real_blade"] = _real_blade(LEG)
    ok = (not report["stub"]["failures"]
          and report["stub"]["n_equal"] == report["stub"]["n_cases"]
          and (args.no_blade or (not report["real_blade"]["failures"]
                                 and report["real_blade"]["n_egos_compared"] > 0)))
    report["pass"] = bool(ok)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "real_blade"}, indent=1)[:3000])
    if "real_blade" in report:
        rb = report["real_blade"]
        print("real_blade:", rb["solver_termination"], rb["n_egos_compared"],
              "compared,", len(rb["failures"]), "failures")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
