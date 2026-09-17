"""Read-only extractor for the two CTDE actor-gradient development diagnostics.

Standard library only. Reads the ORIGINAL external run directories (never writes to
them) and produces the two committed package files:

  artifact_manifest.json   SHA-256 and byte size of the run artifacts
  diagnostic_summary.json  completion counts, evaluation trajectory, gradient windows
                           and credit quantities cited in docs/history/measurements.md §11

Usage (from the repository root):

  python research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/scripts/extract_summary.py
  python research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/scripts/extract_summary.py --check

``--write`` (default) rewrites the two package files; ``--check`` regenerates them in memory
and fails unless they are byte-identical to the committed files. ``--runs-root`` overrides the
parent directory of the two run directories.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RUNS_ROOT = "C:\\gruns"
MEASURED_SHA = "6ed964a1abd09de2130aee3d0d314c8f32165056"
RUNS = (
    ("A_p050", "graph_rl_v2_semantic_ctde_grad_diag_r1_seed3000000_6ed964a"),
    ("B_fd100", "graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a"),
)
TOP_LEVEL_FILES = (
    "authorized_plan.json",
    "preflight.json",
    "launch.cmd",
    "launch_record.json",
    "run_config.json",
    "run_summary.json",
    "train_records.jsonl",
    "eval_records.jsonl",
    "episode_failures.jsonl",
    "episode_outcomes.jsonl",
    "train_actor_gradient_diagnostics.jsonl",
    "train_credit_diagnostics.jsonl",
    "training_console.log",
    "invocation_start_local.txt",
    "native_exit_code.txt",
)
TREE_DIRS = ("checkpoints", "plots", "scenarios")
WINDOW = 25
ABORT = "SELF_PRESERVATION_ABORT"
PLAN = "PLAN_COMPLIANCE"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def manifest_for(run_dir):
    files = {}
    for name in TOP_LEVEL_FILES:
        p = os.path.join(run_dir, name)
        files[name] = {"sha256": _sha256(p), "bytes": os.path.getsize(p)}
    trees = {}
    for d in TREE_DIRS:
        entries = []
        root = os.path.join(run_dir, d)
        for dirpath, _, names in os.walk(root):
            for n in names:
                p = os.path.join(dirpath, n)
                rel = os.path.relpath(p, run_dir).replace("\\", "/")
                entries.append((rel, _sha256(p), os.path.getsize(p)))
        entries.sort()
        tree = {
            "n_files": len(entries),
            "total_bytes": sum(e[2] for e in entries),
            "tree_digest_sha256": hashlib.sha256(
                "".join("%s %s %d\n" % e for e in entries).encode("utf-8")
            ).hexdigest(),
            "tree_digest_definition": "sha256 over sorted lines '<relpath> <sha256> <bytes>\\n'",
        }
        if d == "checkpoints":
            tree["files"] = {e[0]: {"sha256": e[1], "bytes": e[2]} for e in entries}
        trees[d] = tree
    return {"files": files, "trees": trees}


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, prefix + "/" + str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flatten(v, prefix + "/" + str(i)))
    else:
        out[prefix] = obj
    return out


def completion(run_dir):
    s = _json(os.path.join(run_dir, "run_summary.json"))
    c = _json(os.path.join(run_dir, "run_config.json"))
    tr = _jsonl(os.path.join(run_dir, "train_records.jsonl"))
    fd = s["fuel_damage_totals"]
    with open(os.path.join(run_dir, "native_exit_code.txt"), encoding="utf-8") as f:
        exit_code = f.read().strip()
    keys = ("updates_completed", "n_productive_iterations", "train_episodes_attempted",
            "train_episodes_successful", "train_episodes_failed", "train_replacement_attempts",
            "total_transitions", "n_eval_rounds", "eval_episodes_attempted",
            "eval_episodes_successful", "eval_episodes_failed", "failures_recorded",
            "accounting_reconciled")
    fd_keys = ("train_clean_successful", "train_damaged_successful",
               "train_fuel_damage_events_applied", "train_fuel_damage_wakes",
               "train_mild_successful", "train_severe_successful",
               "eval_clean_successful", "eval_mild_successful", "eval_severe_successful")
    return {
        "provenance_git": {k: c["provenance"]["git"][k] for k in ("commit", "branch", "dirty", "dirty_path_count")},
        "fuel_damage_probability": c["train_config"]["fuel_damage_probability"],
        "fuel_damage_mild_probability": c["train_config"]["fuel_damage_mild_probability"],
        "run_summary": {k: s[k] for k in keys},
        "fuel_damage_totals": {k: fd[k] for k in fd_keys},
        "train_records_rows": len(tr),
        "train_records_sum_n_transitions": sum(r["n_transitions"] for r in tr),
        "episode_failures_rows": len(_jsonl(os.path.join(run_dir, "episode_failures.jsonl"))),
        "native_exit_code": exit_code,
    }


def evaluation(run_dir):
    rounds = []
    for e in _jsonl(os.path.join(run_dir, "eval_records.jsonl")):
        b = e["v2_behaviour"]
        rounds.append({
            "evaluation_stage": e["evaluation_stage"],
            "updates_completed": e["updates_completed"],
            "n_attempted": e["n_attempted"],
            "n_successful": e["n_successful"],
            "macro_severe_minus_mild_p_abort": b["macro_mean_over_base_cells"],
            "macro_n_base_cells_defined": b["macro_n_base_cells_defined"],
            "n_groups_complete": b["n_groups_complete"],
            "n_groups_metric_eligible": b["n_groups_metric_eligible"],
            "directional_switch_count": b["directional_switch_count"],
            "reverse_switch_count": b["reverse_switch_count"],
        })
    return rounds


def gradient(run_dir):
    rows = _jsonl(os.path.join(run_dir, "train_actor_gradient_diagnostics.jsonl"))
    defined = [r for r in rows if r["separation_contrast"] is not None]
    fd = lambda r: r["derived"]["fd"]["separation_pressure"]
    nfd = lambda r: r["derived"]["non_fd"]["separation_pressure"]
    tot = lambda r: r["total_policy_surrogate_separation_pressure"]
    act = lambda r: r["total_actor_loss_separation_pressure"]
    windows = []
    n_iter = max(r["iteration"] for r in rows) + 1
    for lo in range(0, n_iter, WINDOW):
        w = [r for r in defined if lo <= r["iteration"] < lo + WINDOW]
        windows.append({
            "iterations": [lo, min(lo + WINDOW, n_iter) - 1],
            "n_defined": len(w),
            "n_fd_pressure_positive": sum(fd(r) > 0 for r in w),
            "n_fd_pressure_negative": sum(fd(r) < 0 for r in w),
            "fraction_fd_pressure_positive": sum(fd(r) > 0 for r in w) / len(w) if w else None,
            "median_fd_pressure": statistics.median(map(fd, w)) if w else None,
            "median_non_fd_pressure": statistics.median(map(nfd, w)) if w else None,
            "median_total_policy_surrogate_pressure": statistics.median(map(tot, w)) if w else None,
        })
    cos_def = [r["cosine_policy_surrogate_vs_actor_loss"] for r in defined]
    cos_all = [r["cosine_policy_surrogate_vs_actor_loss"] for r in rows]
    return {
        "n_rows": len(rows),
        "n_distinct_iterations": len({r["iteration"] for r in rows}),
        "n_defined_contrast": len(defined),
        "n_fd_pressure_positive": sum(fd(r) > 0 for r in defined),
        "n_fd_positive_and_non_fd_negative": sum(fd(r) > 0 and nfd(r) < 0 for r in defined),
        "n_fd_positive_and_total_surrogate_negative": sum(fd(r) > 0 and tot(r) < 0 for r in defined),
        "windows": windows,
        "entropy": {
            "n_defined_where_actor_loss_pressure_sign_differs_from_surrogate": sum(
                (tot(r) > 0) != (act(r) > 0) for r in defined),
            "mean_cosine_surrogate_vs_actor_loss_over_defined": statistics.mean(cos_def),
            "mean_cosine_surrogate_vs_actor_loss_over_all_rows": statistics.mean(cos_all),
            "min_cosine_surrogate_vs_actor_loss_over_all_rows": min(cos_all),
        },
        "max_reconstruction_relative_error": max(r["reconstruction_relative_error"] for r in rows),
        "denominator_note": "windows and signed counts use only updates whose separation_contrast is defined (both MILD and SEVERE immediate-FD transitions present); medians are statistics.median over those updates",
    }


def credit(run_dir):
    rows = _jsonl(os.path.join(run_dir, "train_credit_diagnostics.jsonl"))
    tr = _jsonl(os.path.join(run_dir, "train_records.jsonl"))
    per_iter = Counter(r["iteration"] for r in rows)
    covered = all(per_iter.get(t["iteration"], 0) == t["n_transitions"] for t in tr)
    pop = [r for r in rows if r["wake_kind"] == "immediate_fuel_damage"
           and r["measurement_join"]["is_fd_selected_ego"]]
    by_sev = {}
    for sev in ("mild", "severe"):
        rs = [r for r in pop if r["measurement_join"]["severity"] == sev]
        a = [r["normalized_advantage"] for r in rs if r["selected_meta_action_name"] == ABORT]
        p = [r["normalized_advantage"] for r in rs if r["selected_meta_action_name"] == PLAN]
        nz = [r for r in rs if r["raw_advantage"] != 0]
        by_sev[sev] = {
            "n": len(rs),
            "selected_action_counts": dict(sorted(Counter(r["selected_meta_action_name"] for r in rs).items())),
            "mean_normalized_advantage_abort": statistics.mean(a),
            "mean_normalized_advantage_plan": statistics.mean(p),
            "mean_normalized_advantage_abort_minus_plan": statistics.mean(a) - statistics.mean(p),
            "median_abs_td_residual_over_abs_raw_advantage": statistics.median(
                abs(r["td_residual"]) / abs(r["raw_advantage"]) for r in nz),
            "n_rows_in_ratio": len(nz),
            "mean_abs_td_residual_over_mean_abs_raw_advantage": statistics.mean(
                abs(r["td_residual"]) for r in rs) / statistics.mean(abs(r["raw_advantage"]) for r in rs),
        }
    grouped = defaultdict(lambda: defaultdict(list))
    for r in pop:
        grouped[r["iteration"]][r["measurement_join"]["severity"]].append(r)
    within = {}
    for k in ("value_old", "value_target", "raw_advantage", "td_residual"):
        diffs = [statistics.mean(x[k] for x in g["severe"]) - statistics.mean(x[k] for x in g["mild"])
                 for g in grouped.values() if g["severe"] and g["mild"]]
        within[k] = {"n_updates": len(diffs), "median_severe_minus_mild": statistics.median(diffs)}
    return {
        "n_rows": len(rows),
        "rows_per_update_equal_train_records_n_transitions": covered,
        "wake_kind_counts": dict(sorted(Counter(r["wake_kind"] for r in rows).items())),
        "population": "immediate_fuel_damage rows with measurement_join.is_fd_selected_ego = true",
        "population_n": len(pop),
        "by_severity": by_sev,
        "within_update_severe_minus_mild": within,
        "definitions": {
            "abort_minus_plan": "mean normalized_advantage of rows selecting SELF_PRESERVATION_ABORT minus that of rows selecting PLAN_COMPLIANCE, within severity, pooled over updates; non-counterfactual",
            "within_update": "per update holding both severities: mean(severe) - mean(mild); median over those updates",
            "td_ratio": "per-row |td_residual| / |raw_advantage| (rows with raw_advantage != 0), median; descriptive only, not a causal credit decomposition",
        },
    }


def build(runs_root):
    manifest = {"measured_code_sha": MEASURED_SHA, "runs": {}}
    summary = {"measured_code_sha": MEASURED_SHA, "runs": {}}
    configs = {}
    for key, run_id in RUNS:
        run_dir = os.path.join(runs_root, run_id)
        manifest["runs"][key] = dict(run_id=run_id, run_dir=run_dir, **manifest_for(run_dir))
        summary["runs"][key] = {
            "run_id": run_id,
            "run_dir": run_dir,
            "completion": completion(run_dir),
            "evaluation_trajectory": evaluation(run_dir),
            "gradient": gradient(run_dir),
            "credit": credit(run_dir),
        }
        configs[key] = _flatten(_json(os.path.join(run_dir, "run_config.json")))
    a, b = configs["A_p050"], configs["B_fd100"]
    summary["run_config_paths_differing_B_vs_A"] = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return manifest, summary


def _dump(obj):
    return (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    manifest, summary = build(args.runs_root)
    outputs = {"artifact_manifest.json": _dump(manifest), "diagnostic_summary.json": _dump(summary)}
    ok = True
    for name, data in outputs.items():
        path = os.path.join(PACKAGE_DIR, name)
        if args.check:
            with open(path, "rb") as f:
                same = f.read() == data
            print("%s %s" % ("MATCH" if same else "DIFFER", name))
            ok = ok and same
        else:
            with open(path, "wb") as f:
                f.write(data)
            print("wrote %s" % name)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
