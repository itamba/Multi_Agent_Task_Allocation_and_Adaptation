"""Pre-launch verification for the REWARD-01 coefficient comparison (both arms).

Run from the FROZEN measured worktree under nlp_env with PYTHONPATH=<worktree>/src and
PYTHONNOUSERSITE=1. It trains, evaluates, solves, replays and preflights NOTHING. It checks:

  * source identity -- HEAD == the measured SHA, clean tree, ``match_aou`` / ``graph_train`` /
    ``graph_reward`` import from this worktree, the imported ``blade`` tree digest (the
    Git-blob identity is checked separately by ``blade_blob_identity.py``);
  * each arm's configuration -- the planned argv resolved by the trainer's OWN parser and
    ``resolve_train_config``: validates, ``config_source`` is ``config_file`` with no CLI
    override, every planned value holds, ``reward_config()`` carries the arm's coefficient;
  * the single experimental factor -- the two resolved configs differ exactly in
    ``aircraft_penalty_coeff`` and ``output_dir``; each differs from the section-17
    ``train_config`` only in the declared keys;
  * the coefficient wiring -- ``compute_episode_reward`` is called exactly once in
    ``graph_train``, inside ``_run_one_episode``, with ``cfg.reward_config()``;
    ``reward_config`` passes the field through; the observation / representation ids;
  * reward algebra -- the REAL ``compute_episode_reward`` / ``_event_conditioned_breakdown``
    on synthetic fixed references and outcomes at c = 2.25 and 4.5: doubling ``c`` doubles
    the penalty exactly, leaves the ratio ``q`` unchanged, leaves a zero-loss outcome
    unchanged, and ``R(c) = q - c * p`` holds (also the extractor's rescoring identity);
  * benchmark identity and held-outness -- file SHA-256, ``manifest_id`` recomputed by the V2
    loader, 20 development worlds (identity metadata only), no manifest seed inside
    ``[3000000, 3004500)``;
  * both fresh output paths (and their ``__launcher`` siblings) are absent.

Writes ``--out`` (JSON) and exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

PKG = "research_evidence/generalized_v2/reward_coefficient_dev_r1"
ARMS = {"A": ("configs/arm_a_c2p25.json", 2.25, "reward_c225_r1_s3000000_%s"),
        "B": ("configs/arm_b_c4p5.json", 4.5, "reward_c450_r1_s3000000_%s")}
S17_RUN_CONFIG = "research_evidence/generalized_v2/actor_mission_slack_dev_r1/run_artifacts/run_config.json"
MANIFEST = r"C:\gra\benchmarks\v2_preflight_seed2000000_ae42cb0\benchmark_manifest.json"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"
ACTOR_OBSERVATION_ID = "actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1"
ACTION_REPRESENTATION_ID = "semantic_k_plus_2_logmeanexp_v1"
PLANNED = {"n_iterations": 375, "episodes_per_iteration": 8, "base_seed": 3000000,
           "training_mode": "actor_only", "episode_design": "generalized_v2",
           "match_aou_backend": "p1_milp_v1", "generalized_max_attempts_per_iteration": 12,
           "fuel_damage_mode": "seeded_variable", "fuel_damage_probability": 0.5,
           "fuel_damage_mild_probability": 0.5, "fuel_damage_leg_progress": 0.3,
           "fuel_damage_rtb_margin": 1.1, "early_stopping": False, "max_ticks": None,
           "eval_every": 25, "checkpoint_every": 25, "benchmark_profile": "development",
           "benchmark_manifest": MANIFEST, "visual_artifacts": False,
           "actor_gradient_diagnostics": False, "actor_step_diagnostics": False,
           "actor_step_vector_iterations": []}
PLANNED_PPO = {"lr": 0.0003, "clip_ratio": 0.2, "entropy_coeff": 0.01, "n_epochs": 4,
               "gamma": 1.0, "max_grad_norm": 0.5, "adv_norm_eps": 1e-08}
S17_DIFF = {"A": {"output_dir", "actor_step_diagnostics", "actor_step_vector_iterations"},
            "B": {"output_dir", "actor_step_diagnostics", "actor_step_vector_iterations",
                  "aircraft_penalty_coeff"}}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_digest(root: Path) -> str:
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            lines.append("%s %s\n" % (p.relative_to(root).as_posix(), _sha(p)))
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _jsonable(cfg) -> dict:
    return json.loads(json.dumps(asdict(cfg), default=list))


def reward_algebra():
    """The REAL reward functions on synthetic fixed references; returns (cases, ok)."""
    from match_aou.rl.training import graph_reward as gr

    def task(u, tid):
        return SimpleNamespace(utility=float(u),
                               steps=[SimpleNamespace(probability=1.0, target_id=tid)])

    def reference(u_prefix, tasks, prefix_ids):
        u_cont = sum(t.utility for t in tasks)
        return SimpleNamespace(
            policy=gr.REFERENCE_POLICY_EVENT_CONDITIONED_V1, kind="damaged_event_checkpoint",
            checkpoint_tick=392, u_prefix=float(u_prefix), u_cont_ref=float(u_cont),
            u_ref=float(u_prefix + u_cont),
            u_aircraft=max([t.utility for t in tasks] + [80.0 if u_prefix else 0.0]),
            tasks=tasks, reference_target_ids=tuple(t.steps[0].target_id for t in tasks),
            prefix_target_ids=tuple(prefix_ids))

    ref = reference(80.0, [task(80.0, "t1"), task(40.0, "t2")], ["t0"])
    outcomes = {
        "one_loss_partial": ({("e1", "t1"), ("e2", "tX")}, {"e1"}),
        "two_losses_full": ({("e1", "t1"), ("e2", "t2")}, {"e1", "e2"}),
        "zero_loss_partial": ({("e2", "t2")}, set()),
        "zero_loss_nothing": (set(), set()),
    }
    cases, ok = [], True
    for name, (done, dead) in outcomes.items():
        rows = {}
        for c in (2.25, 4.5):
            ctx = SimpleNamespace(executor=SimpleNamespace(done=set(done), dead=set(dead)),
                                  reference_policy=gr.REFERENCE_POLICY_EVENT_CONDITIONED_V1)
            traj = [SimpleNamespace(reward=None), SimpleNamespace(reward=None)]
            result = SimpleNamespace(reference=ref, trajectory=traj)
            er = gr.compute_episode_reward(ctx, result, gr.RewardConfig(aircraft_penalty_coeff=c))
            direct = gr._event_conditioned_breakdown(ctx, ref,
                                                     gr.RewardConfig(aircraft_penalty_coeff=c))
            denom = abs(er.u_ref) + 1e-5
            p = er.u_aircraft * er.n_lost / denom
            rows[c] = {"ratio_q": er.ratio, "penalty": er.penalty, "reward": er.reward,
                       "n_lost": er.n_lost, "u_aircraft": er.u_aircraft, "u_ref": er.u_ref,
                       "p": p, "terminal_reward_on_last_transition": traj[-1].reward,
                       "other_transition_rewards": [t.reward for t in traj[:-1]],
                       "direct_breakdown_equal": direct == er}
        a, b = rows[2.25], rows[4.5]
        checks = {
            "q_unchanged": a["ratio_q"] == b["ratio_q"],
            "penalty_doubles_exactly": b["penalty"] == 2.0 * a["penalty"],
            "reward_is_q_minus_penalty": all(abs(r["reward"] - (r["ratio_q"] - r["penalty"]))
                                             <= 1e-15 for r in (a, b)),
            "reward_is_q_minus_c_p": all(abs(rows[c]["reward"] - (rows[c]["ratio_q"] - c *
                                                                  rows[c]["p"])) <= 1e-12
                                         for c in rows),
            "rescore_identity_R45_eq_R225_minus_2p25p": abs(
                b["reward"] - (a["reward"] - 2.25 * a["p"])) <= 1e-12,
            "zero_loss_reward_unchanged": (a["n_lost"] > 0) or (a["reward"] == b["reward"]
                                                                and a["penalty"] == 0.0),
            "loss_penalty_positive": (a["n_lost"] == 0) or (a["penalty"] > 0.0),
            "terminal_placement": all(r["terminal_reward_on_last_transition"] == r["reward"]
                                      and r["other_transition_rewards"] == [0.0]
                                      for r in (a, b)),
            "direct_breakdown_equal": a["direct_breakdown_equal"] and b["direct_breakdown_equal"],
        }
        ok = ok and all(checks.values())
        cases.append({"case": name, "c_2p25": a, "c_4p5": b, "checks": checks})
    return cases, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured-sha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    wt = Path.cwd().resolve()
    short = args.measured_sha[:7]
    checks = []
    report = {"record": "preflight", "record_version": 1, "development_only": True,
              "confirmatory_profile_untouched": True, "scientific_execution": "none"}

    def check(name, ok, detail=None):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # --- source identity -----------------------------------------------------
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True,
                                    cwd=str(wt)).stdout.strip()
    head, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=all")
    check("head_is_measured_sha", head == args.measured_sha, head)
    check("worktree_clean", dirty == "", dirty or None)
    import match_aou
    from match_aou.rl.training import graph_reward, graph_train
    from match_aou.rl.action import graph_action
    from match_aou.rl.observation import graph_builder
    paths = {m.__name__: str(Path(m.__file__).resolve())
             for m in (match_aou, graph_train, graph_reward, graph_action, graph_builder)}
    check("modules_import_from_worktree",
          all(p.lower().startswith(str(wt).lower()) for p in paths.values()), paths)
    import blade
    bpath = Path(blade.__file__).resolve().parent
    wt_blade = wt / "src/match_aou/integrations/panopticon-main/gym/blade"
    d_import, d_wt = _tree_digest(bpath), _tree_digest(wt_blade)
    report["code"] = {"measured_code_sha": args.measured_sha, "head": head,
                      "working_tree_clean": dirty == "", "worktree": str(wt),
                      "module_paths": paths, "blade_imported_from": str(bpath),
                      "blade_imported_tree_digest": d_import,
                      "blade_worktree_tree_digest": d_wt,
                      "blade_byte_digest_equal": d_import == d_wt,
                      "blade_note": ("byte digests may differ by line endings only "
                                     "(core.autocrlf); Git-blob identity is the gate, "
                                     "prelaunch/preflight_blade_blob_identity.json")}
    check("actor_observation_id", graph_builder.ACTOR_OBSERVATION_ID == ACTOR_OBSERVATION_ID,
          graph_builder.ACTOR_OBSERVATION_ID)
    check("action_representation_id",
          graph_action.ACTION_REPRESENTATION_ID == ACTION_REPRESENTATION_ID,
          graph_action.ACTION_REPRESENTATION_ID)

    # --- coefficient wiring (static, no episode) ------------------------------
    src = Path(graph_train.__file__).read_text(encoding="utf-8")
    run_one = inspect.getsource(graph_train._run_one_episode)
    rc_src = inspect.getsource(graph_train.TrainConfig.reward_config)
    wiring = {"compute_episode_reward_call_sites_in_graph_train": src.count("compute_episode_reward("),
              "call_inside__run_one_episode":
                  "compute_episode_reward(ctx, result, cfg.reward_config())" in run_one,
              "reward_config_passes_field":
                  "RewardConfig(aircraft_penalty_coeff=float(self.aircraft_penalty_coeff))"
                  in rc_src,
              "cli_dest_maps_to_field":
                  graph_train._CLI_FIELD_BY_DEST.get("aircraft_penalty_coeff")
                  == "aircraft_penalty_coeff"}
    check("coefficient_wiring",
          wiring["compute_episode_reward_call_sites_in_graph_train"] == 1
          and all(v for k, v in wiring.items() if isinstance(v, bool)), wiring)

    # --- per-arm resolved configuration ---------------------------------------
    s17 = json.loads((wt / S17_RUN_CONFIG).read_text(encoding="utf-8"))["train_config"]
    resolved, arms = {}, {}
    for arm, (preset, coeff, out_pat) in ARMS.items():
        preset_abs = str(wt / PKG / preset)
        out_dir = "C:\\gruns\\" + (out_pat % short)
        argv = ["--config", preset_abs, "--out", out_dir]
        parsed = graph_train._build_arg_parser().parse_args(argv)
        values = graph_train.load_config_file(preset_abs)
        cfg, source = graph_train.resolve_train_config(
            parsed, explicit=graph_train._explicit_cli_dests(argv),
            config_values=values, config_path=preset_abs)
        cfg.validate()
        tc = _jsonable(cfg)
        resolved[arm] = tc
        rcfg = cfg.reward_config()
        bad = {k: (tc.get(k), v) for k, v in PLANNED.items() if tc.get(k) != v}
        bad.update({"ppo." + k: (tc["ppo"].get(k), v) for k, v in PLANNED_PPO.items()
                    if tc["ppo"].get(k) != v})
        check("arm_%s_planned_values" % arm, not bad, bad or None)
        check("arm_%s_reward_config_coefficient" % arm,
              rcfg.aircraft_penalty_coeff == coeff and cfg.aircraft_penalty_coeff == coeff
              and rcfg.regret_epsilon == 1e-5,
              {"reward_config": asdict(rcfg), "train_config": cfg.aircraft_penalty_coeff})
        check("arm_%s_config_source_file_no_override" % arm,
              source["resolved_from"] == "config_file" and source["cli_overrides"] == []
              and "output_dir" not in source["config_fields"], source)
        check("arm_%s_output_dir" % arm, str(cfg.output_dir) == out_dir, str(cfg.output_dir))
        d17 = sorted(k for k in set(tc) | set(s17) if tc.get(k, "<absent>") != s17.get(k, "<absent>"))
        check("arm_%s_diff_vs_section17_train_config" % arm, set(d17) == S17_DIFF[arm],
              {k: {"arm": tc.get(k, "<absent>"), "section17": s17.get(k, "<absent>")}
               for k in d17})
        arms[arm] = {"preset": PKG + "/" + preset, "preset_sha256": _sha(Path(preset_abs)),
                     "argv_after_python": ["-m", "match_aou.rl.training.graph_train"] + argv,
                     "output_dir": out_dir, "launcher_dir": out_dir + "__launcher",
                     "train_config": tc, "config_source": source,
                     "reward_config": asdict(rcfg)}
    between = sorted(k for k in set(resolved["A"]) | set(resolved["B"])
                     if resolved["A"].get(k) != resolved["B"].get(k))
    check("arms_differ_only_in_coefficient_and_output",
          between == ["aircraft_penalty_coeff", "output_dir"],
          {k: {"A": resolved["A"].get(k), "B": resolved["B"].get(k)} for k in between})
    report["arms"] = arms
    report["config_diff_between_arms"] = {k: {"A": resolved["A"].get(k),
                                              "B": resolved["B"].get(k)} for k in between}

    # --- reward algebra on the real functions ---------------------------------
    cases, ok = reward_algebra()
    check("reward_algebra_event_conditioned", ok, [c["checks"] for c in cases])
    report["reward_algebra"] = cases

    # --- benchmark identity and held-outness ----------------------------------
    from match_aou.rl.training.graph_generalized import (
        load_v2_benchmark_manifest, manifest_seed_overlap)
    mf = Path(MANIFEST)
    fsha = _sha(mf)
    check("manifest_file_sha256", fsha == MANIFEST_SHA256, fsha)
    manifest = load_v2_benchmark_manifest(mf)
    check("manifest_id", manifest.manifest_id == MANIFEST_ID, manifest.manifest_id)
    dev = manifest.profile_worlds("development")
    dev_rows = [{"world_ordinal": int(getattr(w, "world_ordinal", -1)), "seed": int(w.seed),
                 "key": str(getattr(w, "key", ""))} for w in dev]
    check("development_profile_worlds", len(dev_rows) == 20, len(dev_rows))
    lo = int(PLANNED["base_seed"])
    hi = lo + int(PLANNED["n_iterations"]) * int(PLANNED["generalized_max_attempts_per_iteration"])
    overlap = manifest_seed_overlap(manifest, start=lo, stop=hi)
    seeds = sorted(int(s) for s in manifest.seeds())
    check("held_out_full_manifest", not overlap and (lo, hi) == (3000000, 3004500),
          {"train_band_half_open": [lo, hi], "manifest_n_seeds": len(seeds),
           "manifest_seed_min": seeds[0], "manifest_seed_max": seeds[-1],
           "overlap": list(overlap)})
    report["benchmark"] = {"manifest_path": str(mf), "manifest_file_sha256": fsha,
                           "manifest_id": manifest.manifest_id, "profile": "development",
                           "development_worlds": dev_rows, "regenerated_or_mutated": False,
                           "train_band_half_open": [lo, hi]}

    # --- fresh outputs ---------------------------------------------------------
    for arm, a in arms.items():
        check("arm_%s_output_absent" % arm, not Path(a["output_dir"]).exists()
              and not Path(a["launcher_dir"]).exists(), a["output_dir"])

    report["checks"] = checks
    report["pass"] = all(c["pass"] for c in checks)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for c in checks:
        print("%s  %s" % ("PASS" if c["pass"] else "FAIL", c["check"]))
    print("OVERALL", "PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
