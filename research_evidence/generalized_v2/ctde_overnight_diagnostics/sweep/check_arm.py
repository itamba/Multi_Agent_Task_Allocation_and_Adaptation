"""Post-arm accounting / identity check for the GENERALIZED-V2 CTDE dev diagnostic sweep.

Read-only over a completed run directory. Exit 0 = PASS, 2 = HARD FAIL (stop sweep).
Usage: python check_arm.py <run_dir> <iterations> <episodes> <max_attempts> <eval_every> <fd_prob>
"""
import json, sys, os

SHA = "ae42cb01677f94868b2873008d87be677e31f0c8"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"

run_dir = sys.argv[1]
iters, eps, max_att, eval_every = (int(x) for x in sys.argv[2:6])
fd_prob = float(sys.argv[6])
hard, warn = [], []


def need(cond, msg):
    (None if cond else hard.append(msg))


def jl(name):
    p = os.path.join(run_dir, name)
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


cfg = json.load(open(os.path.join(run_dir, "run_config.json"), encoding="utf-8"))
tc = cfg["train_config"]
g = cfg["provenance"]["git"]
need(g["commit"] == SHA and g["available"] and g["dirty"] is False, f"git provenance {g}")
need(cfg["provenance"]["platform"]["system"] == "Windows", "platform")
need("nlp_env" in cfg["provenance"]["invocation"]["python_executable"], "python env")
expect = {
    "n_iterations": iters, "episodes_per_iteration": eps, "base_seed": 3000000,
    "training_mode": "ctde", "episode_design": "generalized_v2", "match_aou_backend": "p1_milp_v1",
    "benchmark_profile": "development", "generalized_max_attempts_per_iteration": max_att,
    "early_stopping": False, "fuel_damage_mode": "seeded_variable",
    "fuel_damage_probability": fd_prob, "fuel_damage_mild_probability": 0.5,
    "eval_every": eval_every, "checkpoint_every": eval_every, "eval_episodes": 8,
    "visual_artifacts": False, "ctde": {"critic_lr": 0.0003, "value_coeff": 0.5, "gae_lambda": 0.95},
    "ppo": {"clip_ratio": 0.2, "entropy_coeff": 0.01, "lr": 0.0003, "n_epochs": 4, "gamma": 1.0,
            "max_grad_norm": 0.5, "adv_norm_eps": 1e-08},
}
for k, v in expect.items():
    need(tc.get(k) == v, f"train_config.{k}={tc.get(k)!r} expected {v!r}")
be = cfg["provenance"]["seeds"]["benchmark_evaluation"]
need(be["manifest_id"] == MANIFEST_ID, "manifest id")
need(be["evaluation_profile"]["profile"] == "development", "profile")
need(be["evaluation_profile"]["world_ordinals"] == [0, 1], "world ordinals")
need(be["held_out_verified"] is True, "held out")
ap = cfg["training"]["attempt_policy"]
need(ap["successful_episode_quota_total"] == 3000, "quota total")
need(ap["max_possible_training_attempts"] == 4500, "max attempts")

s = json.load(open(os.path.join(run_dir, "run_summary.json"), encoding="utf-8"))
need(s["accounting_reconciled"] is True, "accounting_reconciled")
need(s["train_episodes_successful"] == 3000, f"train successful {s['train_episodes_successful']}")
need(s["train_episodes_attempted"] <= 4500, "attempted <= 4500")
need(s["updates_completed"] == iters, f"updates_completed {s['updates_completed']}")
need(s["n_eval_rounds"] == 16, f"n_eval_rounds {s['n_eval_rounds']}")
need(s["train_iterations_at_full_quota"] == iters, "iterations at full quota")

tr = jl("train_records.jsonl")
need(len(tr) == iters, f"train records {len(tr)}")
need(all(r["n_successful"] == eps and r["n_attempted"] <= max_att and r["n_epochs_run"] > 0 for r in tr),
     "per-iteration quota / attempts / productive update")
ev = jl("eval_records.jsonl")
stages = [r["evaluation_stage"] for r in ev]
need(len(ev) == 16 and stages.count("pre_update") == 1 and stages.count("post_update") == 15, f"eval stages {stages}")
need(all(r["benchmark_profile"] == "development" and r["benchmark_manifest_id"] == MANIFEST_ID for r in ev),
     "eval profile/manifest")
for r in ev:
    if r["n_successful"] != 60 or r["n_attempted"] != 60:
        warn.append(f"eval round {r['eval_round_ordinal']} {r['n_successful']}/{r['n_attempted']}")

n_out = sum(1 for _ in open(os.path.join(run_dir, "episode_outcomes.jsonl"), encoding="utf-8"))
n_fail = sum(1 for _ in open(os.path.join(run_dir, "episode_failures.jsonl"), encoding="utf-8"))
need(n_out == s["train_episodes_successful"] + s["eval_episodes_successful"], f"outcomes {n_out}")
need(n_fail == s["train_episodes_failed"] + s["eval_episodes_failed"], f"failures {n_fail}")

print(json.dumps({
    "run_dir": run_dir, "train_attempted": s["train_episodes_attempted"],
    "train_successful": s["train_episodes_successful"], "train_failed": s["train_episodes_failed"],
    "replacement_attempts": s["train_replacement_attempts"], "updates": s["updates_completed"],
    "eval_rounds": s["n_eval_rounds"], "eval_attempted": s["eval_episodes_attempted"],
    "eval_successful": s["eval_episodes_successful"], "outcome_lines": n_out, "failure_lines": n_fail,
    "failures_by_error_type": s["failures_by_error_type"], "run_seconds": s["run_seconds"],
    "hard": hard, "warn": warn, "verdict": "PASS" if not hard else "HARD_FAIL"}, indent=2))
sys.exit(0 if not hard else 2)
