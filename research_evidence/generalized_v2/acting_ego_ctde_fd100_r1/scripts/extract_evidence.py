"""Read-only extractor / checker for the acting-ego CTDE FD100 development diagnostic.

Standard library only. Reads the ORIGINAL external run directories -- the new acting-ego
run and the historical FD100 comparator -- and never writes to them. It produces every
generated file of this package:

  artifact_manifest.json          SHA-256 and bytes of both runs' artifacts (+ tree digests)
  evidence_summary.json           identity, validity, accounting, config comparison,
                                  held-out trajectory, credit and actor-gradient quantities
  derived/*.jsonl                 per-row projections sufficient to recompute every credit
                                  and gradient quantity in the summary
  originals/*                     byte-identical copies of the new run's small identity,
                                  configuration, summary and record files

Usage (from the repository root):

  python research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/scripts/extract_evidence.py
  python research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/scripts/extract_evidence.py --check
  python research_evidence/generalized_v2/acting_ego_ctde_fd100_r1/scripts/extract_evidence.py --from-derived

``--write`` (default) regenerates the package files. ``--check`` regenerates them in memory,
fails unless every one is byte-identical to the committed file, and runs the integrity
checks. ``--from-derived`` needs NO original run directory: it recomputes the credit,
gradient and trajectory blocks from the committed ``derived/`` rows and ``originals/`` copies
alone and compares them with ``evidence_summary.json``.
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
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_DIR)))

NEW_KEY, HIST_KEY = "acting_ego", "historical_fd100"
RUNS = {
    NEW_KEY: {
        "run_id": "graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3",
        "measured_code_sha": "68055e39768d5fa601e5960a9f08823b9e65c08f",
        "branch": "task/ctde-acting-ego-conditioning",
    },
    HIST_KEY: {
        "run_id": "graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a",
        "measured_code_sha": "6ed964a1abd09de2130aee3d0d314c8f32165056",
        "branch": "task/v2-ctde-gradient-pressure-diagnostics",
    },
}
MANIFEST_PATH = "C:\\gra\\benchmarks\\v2_preflight_seed2000000_ae42cb0\\benchmark_manifest.json"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"
# The PR #74 compact index recorded the historical run's bytes on 2026-09-17; matching it
# proves the comparator originals are unchanged since that review.
PR74_MANIFEST = os.path.join(
    REPO_ROOT, "research_evidence", "generalized_v2", "semantic_ctde_grad_diag_r1",
    "artifact_manifest.json")

TOP_LEVEL_FILES = (
    "authorized_plan.json", "preflight.json", "launch.cmd", "launch_record.json",
    "run_config.json", "run_summary.json", "train_records.jsonl", "eval_records.jsonl",
    "episode_failures.jsonl", "episode_outcomes.jsonl",
    "train_actor_gradient_diagnostics.jsonl", "train_credit_diagnostics.jsonl",
    "training_console.log", "invocation_start_local.txt", "native_exit_code.txt",
)
TREE_DIRS = ("checkpoints", "plots", "scenarios")
# Byte-identical copies committed under originals/ (new run only; all small).
ORIGINAL_COPIES = (
    "authorized_plan.json", "preflight.json", "launch.cmd", "launch_record.json",
    "run_config.json", "run_summary.json", "train_records.jsonl", "eval_records.jsonl",
    "train_actor_gradient_diagnostics.jsonl", "episode_failures.jsonl",
    "invocation_start_local.txt", "native_exit_code.txt",
)
COMMON_MAX_ITERATION = 100      # iterations 0..99; evaluation rounds updates_completed <= 100
EVAL_POINTS = (0, 25, 50, 75, 100)
WINDOWS = ((0, 25), (25, 50), (50, 75), (75, 100))
ABORT = "SELF_PRESERVATION_ABORT"
PLAN = "PLAN_COMPLIANCE"
AUTHORIZED_TRAIN_CONFIG_DIFFS = ("/n_iterations", "/output_dir")


# --------------------------------------------------------------------------- io
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
    # utf-8-sig: launch_record.json was written by PowerShell with a UTF-8 BOM (recorded, not normalized)
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _dump(obj):
    return (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _dump_rows(rows):
    return "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n"
                   for r in rows).encode("utf-8")


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


def _med(x):
    return statistics.median(x) if x else None


def _mean(x):
    return statistics.mean(x) if x else None


# --------------------------------------------------------------------- manifest
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
                "".join("%s %s %d\n" % e for e in entries).encode("utf-8")).hexdigest(),
            "tree_digest_definition": "sha256 over sorted lines '<relpath> <sha256> <bytes>\\n'",
        }
        if d == "checkpoints":
            tree["files"] = {e[0]: {"sha256": e[1], "bytes": e[2]} for e in entries}
        trees[d] = tree
    return {"files": files, "trees": trees}


# ------------------------------------------------------------------ projections
def project_credit(run_dir, max_iteration):
    """Immediate-FD rows of the FD-selected ego, iterations < max_iteration.

    ``is_episode_terminal`` is computed over ALL of the episode's credit rows (every wake
    kind): the row is the episode's last decision, so its GAE bootstrap V_next is zero.
    """
    rows = [r for r in _jsonl(os.path.join(run_dir, "train_credit_diagnostics.jsonl"))
            if r["iteration"] < max_iteration]
    n_dec = Counter((r["iteration"], r["episode_index"]) for r in rows)
    last = defaultdict(int)
    for r in rows:
        k = (r["iteration"], r["episode_index"])
        last[k] = max(last[k], r["episode_decision_ordinal"])
    out = []
    for r in rows:
        j = r["measurement_join"]
        if r["wake_kind"] != "immediate_fuel_damage" or not j["is_fd_selected_ego"]:
            continue
        k = (r["iteration"], r["episode_index"])
        out.append({
            "iteration": r["iteration"],
            "episode_index": r["episode_index"],
            "episode_seed": r["episode_seed"],
            "episode_decision_ordinal": r["episode_decision_ordinal"],
            "episode_n_decisions": n_dec[k],
            "is_episode_terminal": r["episode_decision_ordinal"] == last[k],
            "severity": j["severity"],
            "selected_meta_action_name": r["selected_meta_action_name"],
            "value_old": r["value_old"],
            "value_target": r["value_target"],
            "raw_advantage": r["raw_advantage"],
            "td_residual": r["td_residual"],
            "normalized_advantage": r["normalized_advantage"],
            "gamma": r["gamma"],
            "gae_lambda": r["gae_lambda"],
        })
    return out, len(rows)


def project_gradient(run_dir, max_iteration):
    out = []
    for r in _jsonl(os.path.join(run_dir, "train_actor_gradient_diagnostics.jsonl")):
        if r["iteration"] >= max_iteration:
            continue
        g = r["groups"]
        out.append({
            "iteration": r["iteration"],
            "batch_n_transitions": r["batch_n_transitions"],
            "n_immediate_fd_mild": g.get("immediate_fd_mild", {}).get("n_transitions", 0),
            "n_immediate_fd_severe": g.get("immediate_fd_severe", {}).get("n_transitions", 0),
            "separation_contrast": r["separation_contrast"],
            "fd_separation_pressure": r["derived"]["fd"]["separation_pressure"],
            "non_fd_separation_pressure": r["derived"]["non_fd"]["separation_pressure"],
            "total_policy_surrogate_separation_pressure": r["total_policy_surrogate_separation_pressure"],
            "total_actor_loss_separation_pressure": r["total_actor_loss_separation_pressure"],
            "cosine_policy_surrogate_vs_actor_loss": r["cosine_policy_surrogate_vs_actor_loss"],
            "reconstruction_relative_error": r["reconstruction_relative_error"],
        })
    return out


def project_eval(eval_rows):
    out = []
    for e in eval_rows:
        if e["updates_completed"] > COMMON_MAX_ITERATION:
            continue
        b = e["v2_behaviour"]
        out.append({
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
            "switch_denominator_matched_groups": b["n_groups_metric_eligible"],
        })
    return out


# ----------------------------------------------------------------- computations
def _within(rows, key, lo=0, hi=COMMON_MAX_ITERATION):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if lo <= r["iteration"] < hi:
            g[r["iteration"]][r["severity"]].append(key(r))
    d = [statistics.mean(v["severe"]) - statistics.mean(v["mild"])
         for v in g.values() if v["severe"] and v["mild"]]
    return {"n_updates_with_both_severities": len(d),
            "median_severe_minus_mild": _med(d), "mean_severe_minus_mild": _mean(d)}


def credit_block(rows):
    f = {k: (lambda r, k=k: r[k]) for k in ("value_old", "value_target", "raw_advantage", "td_residual")}
    future = lambda r: r["raw_advantage"] - r["td_residual"]
    nt = [r for r in rows if not r["is_episode_terminal"]]
    nz = [r for r in nt if r["raw_advantage"] != 0]
    by_sev = {}
    for sev in ("mild", "severe"):
        rs = [r for r in rows if r["severity"] == sev]
        a = [r["normalized_advantage"] for r in rs if r["selected_meta_action_name"] == ABORT]
        p = [r["normalized_advantage"] for r in rs if r["selected_meta_action_name"] == PLAN]
        by_sev[sev] = {
            "n": len(rs),
            "selected_action_counts": dict(sorted(Counter(r["selected_meta_action_name"] for r in rs).items())),
            "mean_normalized_advantage_abort_minus_plan": (_mean(a) - _mean(p)) if a and p else None,
        }
    return {
        "population": "immediate_fuel_damage credit rows with measurement_join.is_fd_selected_ego = true, iterations 0..99",
        "population_n": len(rows),
        "nonterminal_n": len(nt),
        "terminal_n": len(rows) - len(nt),
        "by_severity": by_sev,
        "within_update_severe_minus_mild_0_99": {k: _within(rows, fn) for k, fn in f.items()},
        "within_update_severe_minus_mild_50_99": {k: _within(rows, fn, 50, 100) for k, fn in f.items()},
        "value_old_within_update_by_window": {
            "%d-%d" % (lo, hi - 1): _within(rows, f["value_old"], lo, hi) for lo, hi in WINDOWS},
        "nonterminal_td_ratio": {
            "median_abs_td_residual_over_abs_raw_advantage": _med(
                [abs(r["td_residual"]) / abs(r["raw_advantage"]) for r in nz]),
            "n_rows_in_ratio": len(nz),
            "n_nonterminal_rows_excluded_raw_advantage_zero": len(nt) - len(nz),
        },
        "nonterminal_local_vs_future_gae": {
            "median_abs_local_td_residual": _med([abs(r["td_residual"]) for r in nt]),
            "median_abs_future_gamma_lambda_A_next": _med([abs(future(r)) for r in nt]),
            "mean_abs_local_over_mean_abs_future": (
                _mean([abs(r["td_residual"]) for r in nt]) / _mean([abs(future(r)) for r in nt])) if nt else None,
            "fraction_abs_future_gt_abs_local": (
                sum(abs(future(r)) > abs(r["td_residual"]) for r in nt) / len(nt)) if nt else None,
            "within_update_local_0_99": _within(nt, f["td_residual"]),
            "within_update_future_0_99": _within(nt, future),
            "within_update_local_50_99": _within(nt, f["td_residual"], 50, 100),
            "within_update_future_50_99": _within(nt, future, 50, 100),
        },
    }


def gradient_block(rows):
    defined = [r for r in rows if r["separation_contrast"] is not None]
    fd = lambda r: r["fd_separation_pressure"]
    nfd = lambda r: r["non_fd_separation_pressure"]
    tot = lambda r: r["total_policy_surrogate_separation_pressure"]
    act = lambda r: r["total_actor_loss_separation_pressure"]
    windows = {}
    for lo, hi in WINDOWS:
        w = [r for r in defined if lo <= r["iteration"] < hi]
        windows["%d-%d" % (lo, hi - 1)] = {
            "n_updates_in_window": sum(lo <= r["iteration"] < hi for r in rows),
            "n_defined": len(w),
            "n_fd_positive": sum(fd(r) > 0 for r in w),
            "n_non_fd_positive": sum(nfd(r) > 0 for r in w),
            "n_total_surrogate_positive": sum(tot(r) > 0 for r in w),
            "n_fd_positive_total_surrogate_negative": sum(fd(r) > 0 and tot(r) < 0 for r in w),
            "median_fd": _med([fd(r) for r in w]),
            "mean_fd": _mean([fd(r) for r in w]),
            "median_non_fd": _med([nfd(r) for r in w]),
            "median_total_surrogate": _med([tot(r) for r in w]),
        }
    return {
        "n_rows": len(rows),
        "n_distinct_iterations": len({r["iteration"] for r in rows}),
        "n_defined_contrast": len(defined),
        "n_fd_positive": sum(fd(r) > 0 for r in defined),
        "n_fd_positive_non_fd_negative": sum(fd(r) > 0 and nfd(r) < 0 for r in defined),
        "n_fd_positive_total_surrogate_negative": sum(fd(r) > 0 and tot(r) < 0 for r in defined),
        "windows": windows,
        "entropy_sign_flip": {
            "n_defined_where_actor_loss_pressure_sign_differs_from_surrogate": sum(
                (tot(r) > 0) != (act(r) > 0) for r in defined),
            "denominator_n_defined": len(defined),
            "min_cosine_surrogate_vs_actor_loss_over_all_rows": min(
                r["cosine_policy_surrogate_vs_actor_loss"] for r in rows),
        },
        "max_reconstruction_relative_error": max(r["reconstruction_relative_error"] for r in rows),
        "denominator_note": "windows and signed counts use only updates whose separation_contrast is defined (both MILD and SEVERE immediate-FD transitions present)",
    }


def mechanism_blocks(credit_rows, grad_rows, eval_traj):
    return {
        "evaluation_trajectory": eval_traj,
        "credit": credit_block(credit_rows),
        "actor_gradient": gradient_block(grad_rows),
    }


# --------------------------------------------------------------- identity, checks
def console_completion_block(run_dir):
    with open(os.path.join(run_dir, "training_console.log"), "rb") as f:
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("TRAINING SUMMARY ("))
    block = lines[start:]
    return {
        "summary_block_first_line_number_1_based": start + 1,
        "summary_block": block,
        "n_lines_containing_Traceback": sum("Traceback" in l for l in lines),
        "last_iteration_line": [l for l in lines if l.startswith("[iter ")][-1],
    }


def identity_and_validity(new_dir, hist_dir, checks):
    rc = _json(os.path.join(new_dir, "run_config.json"))
    pf = _json(os.path.join(new_dir, "preflight.json"))
    plan = _json(os.path.join(new_dir, "authorized_plan.json"))
    lr = _json(os.path.join(new_dir, "launch_record.json"))
    s = _json(os.path.join(new_dir, "run_summary.json"))
    tr = _jsonl(os.path.join(new_dir, "train_records.jsonl"))
    hist_rc = _json(os.path.join(hist_dir, "run_config.json"))
    with open(os.path.join(new_dir, "native_exit_code.txt"), encoding="utf-8") as f:
        exit_code = f.read().strip()
    with open(os.path.join(new_dir, "invocation_start_local.txt"), encoding="utf-8") as f:
        start_end = f.read().splitlines()
    git = rc["provenance"]["git"]
    be = rc["provenance"]["seeds"]["benchmark_evaluation"]
    fd = s["fuel_damage_totals"]
    n_credit = sum(1 for _ in open(os.path.join(new_dir, "train_credit_diagnostics.jsonl"), encoding="utf-8"))

    tc_new, tc_hist = rc["train_config"], hist_rc["train_config"]
    tc_diff = sorted(k for k in set(_flatten(tc_new)) | set(_flatten(tc_hist))
                     if _flatten(tc_new).get(k) != _flatten(tc_hist).get(k))
    tc_diff_top = sorted({"/" + k.split("/")[1] for k in tc_diff})
    full_new, full_hist = _flatten(rc), _flatten(hist_rc)
    full_diff = sorted(k for k in set(full_new) | set(full_hist) if full_new.get(k) != full_hist.get(k))

    checks["measured_sha_recorded_in_run_config"] = git["commit"] == RUNS[NEW_KEY]["measured_code_sha"]
    checks["clean_tree_recorded_in_run_config"] = git["dirty"] is False and git["dirty_path_count"] == 0
    checks["preflight_checkout_head_equals_measured_sha"] = pf["code"]["checkout_head"] == RUNS[NEW_KEY]["measured_code_sha"]
    checks["preflight_working_tree_clean"] = pf["code"]["working_tree_clean"] is True and pf["code"]["porcelain_untracked_all"] == ""
    checks["run_config_train_config_equals_preflight"] = tc_new == pf["resolved_configuration"]["train_config"]
    checks["plan_measured_sha_equals"] = plan["measured_code_sha"] == RUNS[NEW_KEY]["measured_code_sha"]
    checks["launch_record_hashes_match_files"] = (
        lr["launch_cmd_sha256"] == _sha256(os.path.join(new_dir, "launch.cmd"))
        and lr["authorized_plan_json_sha256"] == _sha256(os.path.join(new_dir, "authorized_plan.json"))
        and lr["preflight_json_sha256"] == _sha256(os.path.join(new_dir, "preflight.json")))
    checks["train_config_differs_from_historical_fd100_only_in_authorized_keys"] = tc_diff_top == list(AUTHORIZED_TRAIN_CONFIG_DIFFS)
    checks["manifest_id_in_run_config"] = (rc["episode_design"]["benchmark_manifest"]["manifest_id"] == MANIFEST_ID
                                           and be["manifest_id"] == MANIFEST_ID)
    checks["manifest_file_sha256_at_preflight"] = pf["manifest_check"]["file_sha256"] == MANIFEST_SHA256
    checks["development_profile_only"] = (tc_new["benchmark_profile"] == "development"
                                          and be["evaluation_profile"]["profile"] == "development")
    checks["held_out_verified_zero_overlap"] = be["held_out_verified"] is True and be["held_out_overlap_count"] == 0
    checks["updates_100_of_100"] = s["updates_completed"] == 100 and s["n_productive_iterations"] == 100
    checks["successful_quota_800"] = s["train_episodes_successful"] == 800
    checks["attempts_within_1200"] = s["train_episodes_attempted"] <= 1200
    checks["accounting_reconciled"] = s["accounting_reconciled"] is True
    checks["no_failures"] = (s["train_episodes_failed"] == 0 and s["eval_episodes_failed"] == 0
                             and s["failures_recorded"] == 0
                             and os.path.getsize(os.path.join(new_dir, "episode_failures.jsonl")) == 0)
    checks["five_eval_rounds"] = s["n_eval_rounds"] == 5
    checks["transitions_reconcile"] = (sum(t["n_transitions"] for t in tr) == s["total_transitions"] == n_credit)
    checks["fd100_every_training_episode_damaged"] = (
        fd["train_clean_successful"] == 0 and fd["train_damaged_successful"] == 800
        and fd["train_fuel_damage_events_applied"] == 800 and fd["train_fuel_damage_wakes"] == 800)
    checks["native_exit_code_zero"] = exit_code == "0"

    return {
        "run_config_provenance_git": git,
        "preflight_code": pf["code"],
        "authorized_plan_authorization": plan["authorization"],
        "benchmark": {
            "manifest_path": MANIFEST_PATH,
            "manifest_id": be["manifest_id"],
            "manifest_file_sha256_at_preflight": pf["manifest_check"]["file_sha256"],
            "profile": be["evaluation_profile"]["profile"],
            "world_ordinals": be["evaluation_profile"]["world_ordinals"],
            "n_worlds": be["evaluation_profile"]["n_worlds"],
            "n_members": be["evaluation_profile"]["n_members"],
            "group_keys_sha256": be["evaluation_profile"]["group_keys_sha256"],
            "seed_list_sha256": be["evaluation_profile"]["seed_list_sha256"],
            "held_out_against_train_band": be["held_out_against_train_band"],
            "held_out_overlap_count": be["held_out_overlap_count"],
            "held_out_verified": be["held_out_verified"],
            "held_out_checked_over": be["held_out_checked_over"],
        },
        "invocation_argv": rc["provenance"]["invocation"]["argv"],
        "launch_record_has_utf8_bom": open(os.path.join(new_dir, "launch_record.json"), "rb").read(3) == bytes([0xEF, 0xBB, 0xBF]),
        "resolved_train_config": tc_new,
        "config_comparison_vs_historical_fd100": {
            "method": "flatten both run_config.json files; list every differing path",
            "train_config_differing_top_level_keys": tc_diff_top,
            "authorized_train_config_differences": list(AUTHORIZED_TRAIN_CONFIG_DIFFS),
            "full_run_config_differing_paths": full_diff,
            "note": "cross-version: the measured code differs (the acting-ego critic state of PR #75 changes the critic input); the resolved configuration differs only in n_iterations 150 -> 100 and output_dir; every other differing run_config path is either derived from n_iterations (seed-band stop/count, attempt budget, quota) or provenance (git commit, git branch, the argv --iterations / --out values, provenance collected_at)",
        },
        "completion": {
            "run_summary": {k: s[k] for k in (
                "updates_completed", "n_productive_iterations", "train_episodes_attempted",
                "train_episodes_successful", "train_episodes_failed", "train_replacement_attempts",
                "total_transitions", "n_eval_rounds", "eval_episodes_attempted",
                "eval_episodes_successful", "eval_episodes_failed", "failures_recorded",
                "accounting_reconciled")},
            "fuel_damage_totals": {k: fd[k] for k in (
                "train_clean_successful", "train_damaged_successful",
                "train_fuel_damage_events_applied", "train_fuel_damage_wakes",
                "train_mild_successful", "train_severe_successful",
                "eval_clean_successful", "eval_mild_successful", "eval_severe_successful")},
            "train_records_rows": len(tr),
            "train_records_sum_n_transitions": sum(t["n_transitions"] for t in tr),
            "train_credit_diagnostics_rows": n_credit,
            "episode_failures_bytes": os.path.getsize(os.path.join(new_dir, "episode_failures.jsonl")),
            "native_exit_code": exit_code,
            "invocation_start_end_local": start_end,
            "console": console_completion_block(new_dir),
        },
    }


def historical_unchanged_vs_pr74(hist_manifest, checks):
    if not os.path.exists(PR74_MANIFEST):
        checks["historical_fd100_bytes_equal_pr74_index"] = False
        return {"pr74_manifest_found": False}
    ref = _json(PR74_MANIFEST)["runs"]["B_fd100"]
    same_files = all(ref["files"][n] == hist_manifest["files"][n] for n in TOP_LEVEL_FILES)
    same_trees = all(ref["trees"][d]["tree_digest_sha256"] == hist_manifest["trees"][d]["tree_digest_sha256"]
                     for d in TREE_DIRS)
    checks["historical_fd100_bytes_equal_pr74_index"] = same_files and same_trees
    return {"pr74_manifest": "research_evidence/generalized_v2/semantic_ctde_grad_diag_r1/artifact_manifest.json",
            "all_top_level_files_equal": same_files, "all_tree_digests_equal": same_trees}


# ------------------------------------------------------------------------ build
def build(runs_root):
    new_dir = os.path.join(runs_root, RUNS[NEW_KEY]["run_id"])
    hist_dir = os.path.join(runs_root, RUNS[HIST_KEY]["run_id"])
    checks = {}
    outputs = {}

    manifest = {"runs": {}}
    for key, d in ((NEW_KEY, new_dir), (HIST_KEY, hist_dir)):
        manifest["runs"][key] = dict(RUNS[key], run_dir=d, **manifest_for(d))
    manifest["external_benchmark_manifest"] = {
        "path": MANIFEST_PATH, "manifest_id": MANIFEST_ID, "file_sha256": MANIFEST_SHA256,
        "note": "hash verified by the run's preflight and by this extractor when the file is reachable"}
    if os.path.exists(MANIFEST_PATH):
        checks["benchmark_manifest_file_sha256_now"] = _sha256(MANIFEST_PATH) == MANIFEST_SHA256

    credit_new, _ = project_credit(new_dir, COMMON_MAX_ITERATION)
    credit_hist, _ = project_credit(hist_dir, COMMON_MAX_ITERATION)
    grad_new = project_gradient(new_dir, COMMON_MAX_ITERATION)
    grad_hist = project_gradient(hist_dir, COMMON_MAX_ITERATION)
    eval_new = project_eval(_jsonl(os.path.join(new_dir, "eval_records.jsonl")))
    eval_hist = project_eval(_jsonl(os.path.join(hist_dir, "eval_records.jsonl")))
    checks["eval_points_new"] = [e["updates_completed"] for e in eval_new] == list(EVAL_POINTS)
    checks["eval_points_historical_common_range"] = [e["updates_completed"] for e in eval_hist] == list(EVAL_POINTS)
    checks["gradient_rows_new_one_per_update"] = sorted(r["iteration"] for r in grad_new) == list(range(100))
    checks["gradient_rows_hist_one_per_update_0_99"] = sorted(r["iteration"] for r in grad_hist) == list(range(100))
    checks["credit_population_new_equals_fd_wakes"] = len(credit_new) == 800

    summary = {
        "package": "acting-ego CTDE FD100 DEVELOPMENT diagnostic -- compact evidence package",
        "formal_status": {
            "implementation_candidate": "68055e39768d5fa601e5960a9f08823b9e65c08f GPT-approved for measurement (PR #75)",
            "run": "completed",
            "scientific_measurement_verdict": "PENDING GPT review of this evidence package",
            "merge_authorized": False,
            "confirmatory_profile_used": False,
            "new_scientific_execution_authorized_by_preservation": False,
            "evidence_class": "DEVELOPMENT diagnostic, cross-version relative to the historical FD100 comparator; not confirmatory",
        },
        "measured_code_sha": {k: RUNS[k]["measured_code_sha"] for k in RUNS},
        "runs": {k: {"run_id": RUNS[k]["run_id"], "run_dir": os.path.join(runs_root, RUNS[k]["run_id"])} for k in RUNS},
        "common_range": "training iterations 0..99 (updates 1..100); evaluation rounds with updates_completed in {0,25,50,75,100}",
        NEW_KEY: dict(identity_and_validity(new_dir, hist_dir, checks),
                      **mechanism_blocks(credit_new, grad_new, eval_new)),
        HIST_KEY: dict(mechanism_blocks(credit_hist, grad_hist, eval_hist),
                       unchanged_since_pr74_index=historical_unchanged_vs_pr74(manifest["runs"][HIST_KEY], checks)),
        "definitions": {
            "within_update": "per update holding both severities: mean(severe) - mean(mild) over the population rows; median / mean over those updates",
            "nonterminal": "row is not its episode's last decision over all wake kinds (its GAE V_next is not the terminal zero)",
            "local": "td_residual = r_t + gamma * V_next - V_old (the one-step TD term of GAE)",
            "future": "raw_advantage - td_residual = gamma * gae_lambda * A_{t+1}",
            "td_ratio": "per-row |td_residual| / |raw_advantage| over nonterminal rows with raw_advantage != 0; median",
            "gradient_windows": "updates are training iterations; windows are half-open [lo, hi) over the iteration index",
            "descriptive_only": "all quantities are descriptive and non-counterfactual",
        },
    }
    summary["integrity_checks"] = dict(sorted(checks.items()))
    outputs["artifact_manifest.json"] = _dump(manifest)
    outputs["evidence_summary.json"] = _dump(summary)
    outputs["derived/immediate_fd_credit_rows_acting_ego.jsonl"] = _dump_rows(credit_new)
    outputs["derived/immediate_fd_credit_rows_historical_fd100_u0_99.jsonl"] = _dump_rows(credit_hist)
    outputs["derived/actor_gradient_rows_acting_ego.jsonl"] = _dump_rows(grad_new)
    outputs["derived/actor_gradient_rows_historical_fd100_u0_99.jsonl"] = _dump_rows(grad_hist)
    outputs["derived/evaluation_trajectory_historical_fd100_u0_100.jsonl"] = _dump_rows(eval_hist)
    for name in ORIGINAL_COPIES:
        with open(os.path.join(new_dir, name), "rb") as f:
            outputs["originals/" + name] = f.read()
    return outputs, checks


def from_derived():
    """Recompute the mechanism blocks from committed files only (no run directory)."""
    summary = _json(os.path.join(PACKAGE_DIR, "evidence_summary.json"))
    d = lambda n: _jsonl(os.path.join(PACKAGE_DIR, "derived", n))
    eval_new = project_eval(_jsonl(os.path.join(PACKAGE_DIR, "originals", "eval_records.jsonl")))
    new = mechanism_blocks(d("immediate_fd_credit_rows_acting_ego.jsonl"),
                           d("actor_gradient_rows_acting_ego.jsonl"), eval_new)
    hist = mechanism_blocks(d("immediate_fd_credit_rows_historical_fd100_u0_99.jsonl"),
                            d("actor_gradient_rows_historical_fd100_u0_99.jsonl"),
                            d("evaluation_trajectory_historical_fd100_u0_100.jsonl"))
    ok = True
    for key, blocks in ((NEW_KEY, new), (HIST_KEY, hist)):
        for b, v in blocks.items():
            same = _dump(v) == _dump(summary[key][b])
            print("%s %s.%s (from derived rows)" % ("MATCH" if same else "DIFFER", key, b))
            ok = ok and same
    # the originals copies of the gradient stream reproduce the derived projection
    same = _dump_rows(project_gradient(os.path.join(PACKAGE_DIR, "originals"), COMMON_MAX_ITERATION)) == \
        open(os.path.join(PACKAGE_DIR, "derived", "actor_gradient_rows_acting_ego.jsonl"), "rb").read()
    print("%s derived/actor_gradient_rows_acting_ego.jsonl reprojected from originals/" % ("MATCH" if same else "DIFFER"))
    return 0 if ok and same else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--from-derived", action="store_true")
    args = ap.parse_args()
    if args.from_derived:
        return from_derived()
    outputs, checks = build(args.runs_root)
    ok = True
    for name, data in sorted(outputs.items()):
        path = os.path.join(PACKAGE_DIR, *name.split("/"))
        if args.check:
            with open(path, "rb") as f:
                same = f.read() == data      # package is -text: bytes are exact
            print("%s %s" % ("MATCH" if same else "DIFFER", name))
            ok = ok and same
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            print("wrote %s" % name)
    for k, v in sorted(checks.items()):
        print("%s %s" % ("PASS" if v else "FAIL", k))
        ok = ok and bool(v)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
