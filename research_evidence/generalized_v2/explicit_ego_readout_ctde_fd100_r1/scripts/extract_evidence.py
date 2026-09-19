"""Read-only extractor / checker for the explicit acting-ego readout CTDE FD100 diagnostic.

Standard library only. Reads the ORIGINAL external run directories of three arms and never
writes to them:

  explicit_readout  graph_rl_v2_explicit_ego_readout_ctde_fd100_r1_seed3000000_1a1e0c9  (new)
  role_only         graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3           (primary comparator)
  symmetric         graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a   (secondary context)

and produces every generated file of this package:

  artifact_manifest.json      SHA-256 and bytes of all three runs' artifacts (+ tree digests)
  evidence_summary.json       identity, validity, accounting, benchmark and config boundaries,
                              held-out trajectories, credit, actor-gradient and the
                              OWNER-TRANSITION credit audit for all three arms
  derived/<arm>/*.jsonl       per-row projections sufficient to recompute every mechanism and
                              owner-transition quantity in the summary
  originals/*                 byte-identical copies of the explicit run's small identity,
                              configuration, summary and record files

Every mechanism quantity uses the common range: training iterations 0..99 and evaluation
rounds with updates_completed <= 100.

Usage (from the repository root):

  python research_evidence/generalized_v2/explicit_ego_readout_ctde_fd100_r1/scripts/extract_evidence.py
  python research_evidence/generalized_v2/explicit_ego_readout_ctde_fd100_r1/scripts/extract_evidence.py --check
  python research_evidence/generalized_v2/explicit_ego_readout_ctde_fd100_r1/scripts/extract_evidence.py --from-derived

``--write`` (default) regenerates the package files. ``--check`` regenerates them in memory,
fails unless every one is byte-identical to the committed file, and runs the integrity checks.
``--from-derived`` needs NO run directory: it recomputes the trajectory, credit, gradient and
owner-transition blocks from the committed ``derived/`` rows alone and compares them with
``evidence_summary.json``.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PACKAGE_DIR)))
DEFAULT_RUNS_ROOT = "C:\\gruns"

NEW, ROLE, SYM = "explicit_readout", "role_only", "symmetric"
ARMS = (NEW, ROLE, SYM)
RUNS = {
    NEW: {"run_id": "graph_rl_v2_explicit_ego_readout_ctde_fd100_r1_seed3000000_1a1e0c9",
          "measured_code_sha": "1a1e0c953c54e9d3f46c871158d5ab6bdd881f24",
          "branch": "task/ctde-acting-ego-conditioning",
          "critic": "role-only acting-ego central state + explicit readout [mean pool ; acting-ego embedding]",
          "n_iterations": 100, "n_successful": 800, "n_eval_rounds": 5},
    ROLE: {"run_id": "graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3",
           "measured_code_sha": "68055e39768d5fa601e5960a9f08823b9e65c08f",
           "branch": "task/ctde-acting-ego-conditioning",
           "critic": "role-only acting-ego central state, mean-pool readout",
           "n_iterations": 100, "n_successful": 800, "n_eval_rounds": 5},
    SYM: {"run_id": "graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a",
          "measured_code_sha": "6ed964a1abd09de2130aee3d0d314c8f32165056",
          "branch": "task/v2-ctde-gradient-pressure-diagnostics",
          "critic": "symmetric central state (no distinguished agent), mean-pool readout",
          "n_iterations": 150, "n_successful": 1200, "n_eval_rounds": 7},
}
READOUT_ID = "mean_pool_plus_acting_ego_v1"
MANIFEST_PATH = "C:\\gra\\benchmarks\\v2_preflight_seed2000000_ae42cb0\\benchmark_manifest.json"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"
# Earlier committed indexes of the two comparator runs; byte equality with them proves the
# comparator originals are unchanged since those reviews.
PRIOR_INDEX = {
    ROLE: (os.path.join(REPO_ROOT, "research_evidence", "generalized_v2", "acting_ego_ctde_fd100_r1",
                        "artifact_manifest.json"), "acting_ego"),
    SYM: (os.path.join(REPO_ROOT, "research_evidence", "generalized_v2", "semantic_ctde_grad_diag_r1",
                       "artifact_manifest.json"), "B_fd100"),
}
TOP_LEVEL_FILES = (
    "authorized_plan.json", "preflight.json", "launch.cmd", "launch_record.json",
    "run_config.json", "run_summary.json", "train_records.jsonl", "eval_records.jsonl",
    "episode_failures.jsonl", "episode_outcomes.jsonl",
    "train_actor_gradient_diagnostics.jsonl", "train_credit_diagnostics.jsonl",
    "training_console.log", "invocation_start_local.txt", "native_exit_code.txt",
)
TREE_DIRS = ("checkpoints", "plots", "scenarios")
ORIGINAL_COPIES = (
    "authorized_plan.json", "preflight.json", "launch.cmd", "launch_record.json",
    "run_config.json", "run_summary.json", "train_records.jsonl", "eval_records.jsonl",
    "train_actor_gradient_diagnostics.jsonl", "episode_failures.jsonl",
    "invocation_start_local.txt", "native_exit_code.txt",
)
MAX_IT = 100
EVAL_POINTS = (0, 25, 50, 75, 100)
WINDOWS = ((0, 25), (25, 50), (50, 75), (75, 100))
ABORT, PLAN = "SELF_PRESERVATION_ABORT", "PLAN_COMPLIANCE"
TD_TOL = 1e-5


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
    # utf-8-sig: the role-only run's launch_record.json carries a UTF-8 BOM (read, never rewritten)
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _dump(obj):
    return (json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _dump_rows(rows):
    return "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows).encode("utf-8")


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


def _sign(x):
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------- manifest
def manifest_for(run_dir):
    files = {}
    for name in TOP_LEVEL_FILES:
        p = os.path.join(run_dir, name)
        files[name] = {"sha256": _sha256(p), "bytes": os.path.getsize(p)}
    trees = {}
    for d in TREE_DIRS:
        entries = []
        for dirpath, _, names in os.walk(os.path.join(run_dir, d)):
            for n in names:
                p = os.path.join(dirpath, n)
                entries.append((os.path.relpath(p, run_dir).replace("\\", "/"), _sha256(p), os.path.getsize(p)))
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
def load_credit(run_dir):
    return [r for r in _jsonl(os.path.join(run_dir, "train_credit_diagnostics.jsonl"))
            if r["iteration"] < MAX_IT]


def episode_sequences(rows, checks, arm):
    """Global decision sequence per episode, keyed (iteration, episode_index)."""
    ep = defaultdict(list)
    for r in rows:
        ep[(r["iteration"], r["episode_index"])].append(r)
    ok_contig = ok_seed = True
    for key, rs in ep.items():
        rs.sort(key=lambda r: r["episode_decision_ordinal"])
        ords = [r["episode_decision_ordinal"] for r in rs]
        if ords != list(range(len(rs))):
            ok_contig = False
        if len({r["episode_seed"] for r in rs}) != 1:
            ok_seed = False
    checks["%s_decision_ordinals_contiguous_no_duplicates_no_gaps" % arm] = ok_contig
    checks["%s_one_seed_per_episode" % arm] = ok_seed
    return ep


def project_credit(ep):
    out = []
    for (it, ei), rs in sorted(ep.items()):
        last = len(rs) - 1
        for r in rs:
            j = r["measurement_join"]
            if r["wake_kind"] != "immediate_fuel_damage" or not j["is_fd_selected_ego"]:
                continue
            out.append({
                "iteration": it, "episode_index": ei, "episode_seed": r["episode_seed"],
                "episode_decision_ordinal": r["episode_decision_ordinal"],
                "episode_n_decisions": len(rs),
                "is_episode_terminal": r["episode_decision_ordinal"] == last,
                "severity": j["severity"],
                "selected_meta_action_name": r["selected_meta_action_name"],
                "value_old": r["value_old"], "value_target": r["value_target"],
                "raw_advantage": r["raw_advantage"], "td_residual": r["td_residual"],
                "normalized_advantage": r["normalized_advantage"],
                "gamma": r["gamma"], "gae_lambda": r["gae_lambda"],
            })
    return out


def project_owner_transitions(ep):
    """Every NONTERMINAL immediate-FD FD-selected-ego row joined to the NEXT global decision."""
    out = []
    for (it, ei), rs in sorted(ep.items()):
        for idx, r in enumerate(rs):
            j = r["measurement_join"]
            if r["wake_kind"] != "immediate_fuel_damage" or not j["is_fd_selected_ego"]:
                continue
            if idx == len(rs) - 1:
                continue                      # terminal: no next decision, excluded by definition
            n = rs[idx + 1]
            out.append({
                "iteration": it, "episode_index": ei, "episode_seed": r["episode_seed"],
                "episode_decision_ordinal": r["episode_decision_ordinal"],
                "next_episode_decision_ordinal": n["episode_decision_ordinal"],
                "severity": j["severity"],
                "ego_id": str(r["ego_id"]), "next_ego_id": str(n["ego_id"]),
                "same_ego_next": str(r["ego_id"]) == str(n["ego_id"]),
                "next_wake_kind": n["wake_kind"],
                "value_old": r["value_old"], "next_value_old": n["value_old"],
                "delta_value": n["value_old"] - r["value_old"],
                "transition_reward": r["transition_reward"],
                "gamma": r["gamma"], "gae_lambda": r["gae_lambda"],
                "td_residual": r["td_residual"], "raw_advantage": r["raw_advantage"],
                "next_raw_advantage": n["raw_advantage"],
                "future_component": r["raw_advantage"] - r["td_residual"],
            })
    return out


def project_gradient(run_dir):
    out = []
    for r in _jsonl(os.path.join(run_dir, "train_actor_gradient_diagnostics.jsonl")):
        if r["iteration"] >= MAX_IT:
            continue
        g = r["groups"]
        out.append({
            "iteration": r["iteration"], "batch_n_transitions": r["batch_n_transitions"],
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


def project_eval(run_dir):
    out = []
    for e in _jsonl(os.path.join(run_dir, "eval_records.jsonl")):
        if e["updates_completed"] > MAX_IT:
            continue
        b = e["v2_behaviour"]
        out.append({
            "evaluation_stage": e["evaluation_stage"], "updates_completed": e["updates_completed"],
            "n_attempted": e["n_attempted"], "n_successful": e["n_successful"],
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
def _within(rows, key, lo=0, hi=MAX_IT):
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
        "population_n": len(rows), "nonterminal_n": len(nt), "terminal_n": len(rows) - len(nt),
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
            "median_fd": _med([fd(r) for r in w]), "mean_fd": _mean([fd(r) for r in w]),
            "median_non_fd": _med([nfd(r) for r in w]),
            "median_total_surrogate": _med([tot(r) for r in w]),
        }
    return {
        "n_rows": len(rows), "n_distinct_iterations": len({r["iteration"] for r in rows}),
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


def _split_stats(rs, n_total):
    d = [r["delta_value"] for r in rs]
    q = statistics.quantiles(d, n=4, method="inclusive") if len(d) >= 2 else None
    md = _med(d)
    nz = [r for r in rs if r["raw_advantage"] != 0]
    return {
        "n": len(rs),
        "fraction_of_nonterminal": len(rs) / n_total if n_total else None,
        "median_abs_td_residual": _med([abs(r["td_residual"]) for r in rs]),
        "mean_abs_td_residual": _mean([abs(r["td_residual"]) for r in rs]),
        "median_abs_td_over_abs_raw_advantage": _med([abs(r["td_residual"]) / abs(r["raw_advantage"]) for r in nz]),
        "n_rows_in_td_ratio": len(nz),
        "median_delta_value": md, "mean_delta_value": _mean(d),
        "pstdev_delta_value": statistics.pstdev(d) if d else None,
        "iqr_delta_value": (q[2] - q[0]) if q else None,
        "median_abs_deviation_delta_value": _med([abs(x - md) for x in d]) if d else None,
        "median_abs_delta_value": _med([abs(x) for x in d]),
        "fraction_sign_td_differs_from_sign_raw_advantage": (
            sum(_sign(r["td_residual"]) != _sign(r["raw_advantage"]) for r in rs) / len(rs)) if rs else None,
        "fraction_abs_future_gt_abs_local": (
            sum(abs(r["future_component"]) > abs(r["td_residual"]) for r in rs) / len(rs)) if rs else None,
    }


def _within_block(rs):
    keys = {"V_t": lambda r: r["value_old"], "V_next": lambda r: r["next_value_old"],
            "delta_value": lambda r: r["delta_value"], "td_residual": lambda r: r["td_residual"],
            "future_component": lambda r: r["future_component"]}
    return {k: _within(rs, fn) for k, fn in keys.items()}


def owner_transition_block(rows):
    n = len(rows)
    same = [r for r in rows if r["same_ego_next"]]
    diff = [r for r in rows if not r["same_ego_next"]]
    cells = defaultdict(list)
    for r in rows:
        cells[("same_ego_next" if r["same_ego_next"] else "different_ego_next", r["next_wake_kind"])].append(r)
    return {
        "population": "NONTERMINAL immediate_fuel_damage rows of the FD-selected ego, iterations 0..99, joined to the next global decision of the same episode",
        "nonterminal_n": n,
        "all_nonterminal": _split_stats(rows, n),
        "same_ego_next": _split_stats(same, n),
        "different_ego_next": _split_stats(diff, n),
        "within_update_severe_minus_mild": {
            "all_nonterminal": _within_block(rows),
            "same_ego_next": _within_block(same),
            "different_ego_next": _within_block(diff),
        },
        "structure": {
            "n_nonzero_transition_reward": sum(r["transition_reward"] != 0 for r in rows),
            "gamma_values": sorted({r["gamma"] for r in rows}),
            "gae_lambda_values": sorted({r["gae_lambda"] for r in rows}),
            "n_td_residual_not_equal_delta_value": sum(
                abs(r["td_residual"] - r["delta_value"]) > TD_TOL for r in rows),
            "note": "with a terminal-only reward and gamma = 1, a nonterminal td_residual is exactly V_{t+1} - V_t",
        },
        "next_wake_kind_counts": dict(sorted(Counter(r["next_wake_kind"] for r in rows).items())),
        "by_owner_and_next_wake_kind": {
            "%s|%s" % k: _split_stats(v, n) for k, v in sorted(cells.items())},
        "severity_by_owner": {
            "same_ego_next": dict(sorted(Counter(r["severity"] for r in same).items())),
            "different_ego_next": dict(sorted(Counter(r["severity"] for r in diff).items())),
        },
        "note": "only observed (owner, next-wake) cells are listed; no empty category is created",
    }


def owner_transition_checks(rows, credit_rows, arm, checks):
    td_err = max((abs(r["transition_reward"] + r["gamma"] * r["next_value_old"] - r["value_old"]
                      - r["td_residual"]) for r in rows), default=0.0)
    gae_err = max((abs(r["future_component"] - r["gamma"] * r["gae_lambda"] * r["next_raw_advantage"])
                   for r in rows), default=0.0)
    checks["%s_td_residual_equals_r_plus_gamma_Vnext_minus_V" % arm] = td_err <= TD_TOL
    checks["%s_future_equals_gamma_lambda_A_next" % arm] = gae_err <= TD_TOL
    checks["%s_next_ordinal_is_current_plus_one" % arm] = all(
        r["next_episode_decision_ordinal"] == r["episode_decision_ordinal"] + 1 for r in rows)
    checks["%s_owner_transition_n_equals_credit_nonterminal_n" % arm] = (
        len(rows) == sum(not r["is_episode_terminal"] for r in credit_rows))
    return {"max_abs_td_reconstruction_error": td_err, "max_abs_gae_recurrence_error": gae_err,
            "tolerance": TD_TOL}


def mechanism_blocks(credit_rows, grad_rows, eval_rows, ot_rows):
    return {"evaluation_trajectory": eval_rows, "credit": credit_block(credit_rows),
            "actor_gradient": gradient_block(grad_rows),
            "owner_transition_audit": owner_transition_block(ot_rows)}


# ------------------------------------------------------------------ identity
def validity(arm, run_dir, checks):
    spec = RUNS[arm]
    rc = _json(os.path.join(run_dir, "run_config.json"))
    s = _json(os.path.join(run_dir, "run_summary.json"))
    tr = _jsonl(os.path.join(run_dir, "train_records.jsonl"))
    git = rc["provenance"]["git"]
    be = rc["provenance"]["seeds"]["benchmark_evaluation"]
    fd = s["fuel_damage_totals"]
    with open(os.path.join(run_dir, "native_exit_code.txt"), encoding="utf-8") as f:
        exit_code = f.read().strip()
    n_credit = sum(1 for _ in open(os.path.join(run_dir, "train_credit_diagnostics.jsonl"), encoding="utf-8"))
    a = arm + "_"
    checks[a + "measured_sha"] = git["commit"] == spec["measured_code_sha"]
    checks[a + "clean_tree"] = git["dirty"] is False and git["dirty_path_count"] == 0
    checks[a + "updates"] = s["updates_completed"] == spec["n_iterations"] == s["n_productive_iterations"]
    checks[a + "successful_quota"] = s["train_episodes_successful"] == spec["n_successful"]
    checks[a + "accounting_reconciled"] = s["accounting_reconciled"] is True
    checks[a + "no_failures"] = (s["train_episodes_failed"] == 0 and s["eval_episodes_failed"] == 0
                                 and s["failures_recorded"] == 0
                                 and os.path.getsize(os.path.join(run_dir, "episode_failures.jsonl")) == 0)
    checks[a + "eval_rounds"] = s["n_eval_rounds"] == spec["n_eval_rounds"]
    checks[a + "transitions_reconcile"] = sum(t["n_transitions"] for t in tr) == s["total_transitions"] == n_credit
    checks[a + "fd100_all_training_damaged"] = (
        fd["train_clean_successful"] == 0 and fd["train_damaged_successful"] == spec["n_successful"]
        and fd["train_fuel_damage_events_applied"] == spec["n_successful"]
        and fd["train_fuel_damage_wakes"] == spec["n_successful"])
    checks[a + "native_exit_code_zero"] = exit_code == "0"
    checks[a + "manifest_id"] = (be["manifest_id"] == MANIFEST_ID
                                 and rc["episode_design"]["benchmark_manifest"]["manifest_id"] == MANIFEST_ID)
    checks[a + "development_profile_only"] = (rc["train_config"]["benchmark_profile"] == "development"
                                              and be["evaluation_profile"]["profile"] == "development")
    checks[a + "held_out_zero_overlap"] = be["held_out_verified"] is True and be["held_out_overlap_count"] == 0
    return {
        "run_id": spec["run_id"], "run_dir": run_dir, "measured_code_sha": spec["measured_code_sha"],
        "critic": spec["critic"], "provenance_git": git,
        "run_summary": {k: s[k] for k in (
            "updates_completed", "n_productive_iterations", "train_episodes_attempted",
            "train_episodes_successful", "train_episodes_failed", "train_replacement_attempts",
            "total_transitions", "n_eval_rounds", "eval_episodes_attempted",
            "eval_episodes_successful", "eval_episodes_failed", "failures_recorded",
            "accounting_reconciled")},
        "fuel_damage_totals": {k: fd[k] for k in (
            "train_clean_successful", "train_damaged_successful", "train_fuel_damage_events_applied",
            "train_fuel_damage_wakes", "train_mild_successful", "train_severe_successful")},
        "train_credit_diagnostics_rows": n_credit,
        "native_exit_code": exit_code,
        "benchmark": {"manifest_id": be["manifest_id"], "profile": be["evaluation_profile"]["profile"],
                      "held_out_against_train_band": be["held_out_against_train_band"],
                      "held_out_overlap_count": be["held_out_overlap_count"],
                      "held_out_verified": be["held_out_verified"]},
    }


def config_boundary(new_dir, other_dir):
    a = _json(os.path.join(new_dir, "run_config.json"))["train_config"]
    b = _json(os.path.join(other_dir, "run_config.json"))["train_config"]
    return sorted("/" + k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


def explicit_specifics(new_dir, checks):
    pf = _json(os.path.join(new_dir, "preflight.json"))
    plan = _json(os.path.join(new_dir, "authorized_plan.json"))
    lr = _json(os.path.join(new_dir, "launch_record.json"))
    rc = _json(os.path.join(new_dir, "run_config.json"))
    with open(os.path.join(new_dir, "training_console.log"), "rb") as f:
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("TRAINING SUMMARY ("))
    with open(os.path.join(new_dir, "invocation_start_local.txt"), encoding="utf-8") as f:
        start_end = f.read().splitlines()
    checks["explicit_readout_preflight_head_equals_measured_sha"] = pf["code"]["checkout_head"] == RUNS[NEW]["measured_code_sha"]
    checks["explicit_readout_preflight_pr75_head_equals_measured_sha"] = pf["code"]["pr75_head"] == RUNS[NEW]["measured_code_sha"]
    checks["explicit_readout_preflight_tree_clean"] = pf["code"]["working_tree_clean"] is True and pf["code"]["porcelain_untracked_all"] == ""
    checks["explicit_readout_run_config_equals_preflight"] = rc["train_config"] == pf["resolved_configuration"]["train_config"]
    checks["explicit_readout_preflight_critic_readout_id"] = pf["critic_readout"]["critic_readout_id"] == READOUT_ID
    checks["explicit_readout_value_head_input_2x_embed"] = pf["critic_readout"]["value_head_first_layer_shape"] == [64, 128]
    checks["explicit_readout_launch_record_hashes_match"] = (
        lr["launch_cmd_sha256"] == _sha256(os.path.join(new_dir, "launch.cmd"))
        and lr["authorized_plan_json_sha256"] == _sha256(os.path.join(new_dir, "authorized_plan.json"))
        and lr["preflight_json_sha256"] == _sha256(os.path.join(new_dir, "preflight.json")))
    checks["explicit_readout_preflight_manifest_sha256"] = pf["manifest_check"]["file_sha256"] == MANIFEST_SHA256
    checks["explicit_readout_console_no_traceback"] = not any("Traceback" in l for l in lines)
    return {
        "preflight_code": pf["code"], "critic_readout": pf["critic_readout"],
        "authorized_plan_authorization": plan["authorization"],
        "invocation_argv": rc["provenance"]["invocation"]["argv"],
        "resolved_train_config": rc["train_config"],
        "invocation_start_end_local": start_end,
        "console_summary_block": lines[start:],
        "console_n_lines_containing_Traceback": sum("Traceback" in l for l in lines),
    }


# ------------------------------------------------------------------------ build
def build(runs_root):
    checks, outputs = {}, {}
    dirs = {arm: os.path.join(runs_root, RUNS[arm]["run_id"]) for arm in ARMS}

    manifest = {"runs": {arm: dict(RUNS[arm], run_dir=dirs[arm], **manifest_for(dirs[arm])) for arm in ARMS},
                "external_benchmark_manifest": {"path": MANIFEST_PATH, "manifest_id": MANIFEST_ID,
                                                "file_sha256": MANIFEST_SHA256}}
    if os.path.exists(MANIFEST_PATH):
        checks["benchmark_manifest_file_sha256_now"] = _sha256(MANIFEST_PATH) == MANIFEST_SHA256
    prior = {}
    for arm, (path, key) in PRIOR_INDEX.items():
        ref = _json(path)["runs"][key]
        m = manifest["runs"][arm]
        same = (all(ref["files"][n] == m["files"][n] for n in TOP_LEVEL_FILES)
                and all(ref["trees"][d]["tree_digest_sha256"] == m["trees"][d]["tree_digest_sha256"] for d in TREE_DIRS))
        checks["%s_bytes_equal_prior_committed_index" % arm] = same
        prior[arm] = {"index": os.path.relpath(path, REPO_ROOT).replace("\\", "/"), "entry": key, "equal": same}

    cfg = {"vs_role_only": config_boundary(dirs[NEW], dirs[ROLE]),
           "vs_symmetric": config_boundary(dirs[NEW], dirs[SYM])}
    checks["config_vs_role_only_differs_only_in_output_dir"] = cfg["vs_role_only"] == ["/output_dir"]
    checks["config_vs_symmetric_differs_only_in_n_iterations_output_dir"] = cfg["vs_symmetric"] == ["/n_iterations", "/output_dir"]

    arms_summary, ot_integrity = {}, {}
    for arm in ARMS:
        ep = episode_sequences(load_credit(dirs[arm]), checks, arm)
        credit = project_credit(ep)
        ot = project_owner_transitions(ep)
        grad = project_gradient(dirs[arm])
        ev = project_eval(dirs[arm])
        ot_integrity[arm] = owner_transition_checks(ot, credit, arm, checks)
        checks["%s_eval_points_common_range" % arm] = [e["updates_completed"] for e in ev] == list(EVAL_POINTS)
        checks["%s_gradient_one_row_per_update_0_99" % arm] = sorted(r["iteration"] for r in grad) == list(range(MAX_IT))
        arms_summary[arm] = dict(validity(arm, dirs[arm], checks), **mechanism_blocks(credit, grad, ev, ot))
        arms_summary[arm]["owner_transition_integrity"] = ot_integrity[arm]
        base = "derived/%s/" % arm
        outputs[base + "immediate_fd_credit_rows.jsonl"] = _dump_rows(credit)
        outputs[base + "owner_transition_rows.jsonl"] = _dump_rows(ot)
        outputs[base + "actor_gradient_rows.jsonl"] = _dump_rows(grad)
        outputs[base + "evaluation_trajectory.jsonl"] = _dump_rows(ev)

    arms_summary[NEW]["explicit_run_specifics"] = explicit_specifics(dirs[NEW], checks)
    summary = {
        "package": "explicit acting-ego readout CTDE FD100 DEVELOPMENT diagnostic -- compact evidence package",
        "formal_status": {
            "implementation_candidate": "1a1e0c953c54e9d3f46c871158d5ab6bdd881f24 GPT-approved for measurement (PR #75)",
            "run": "completed",
            "scientific_measurement_verdict": "APPROVE -- VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT (GPT review of this evidence package, 2026-09-19; recorded in docs/history/measurements.md section 13)",
            "evidence_class": "DEVELOPMENT only; not randomized; not confirmatory",
            "primary_comparator": "role_only (68055e3); the intended code-level difference is the critic readout",
            "secondary_context": "symmetric FD100 (6ed964a), common range through update 100",
            "confirmatory_profile_used": False,
            "new_run_authorized": False, "merge_authorized": False,
        },
        "measured_code_sha": {arm: RUNS[arm]["measured_code_sha"] for arm in ARMS},
        "common_range": "training iterations 0..99; evaluation rounds updates_completed in {0,25,50,75,100}",
        "config_boundaries": dict(cfg, note="train_config keys that differ; code-level differences (critic state / readout) are stated per arm under 'critic', not hidden as configuration equality"),
        "comparator_bytes_unchanged_since_prior_index": prior,
        "arms": arms_summary,
        "definitions": {
            "within_update": "per update holding both severities: mean(severe) - mean(mild); median / mean over those updates",
            "nonterminal": "row is not its episode's last global decision (over all wake kinds)",
            "next_decision": "the row with episode_decision_ordinal + 1 in the same (iteration, episode_index) global decision sequence",
            "local": "td_residual = transition_reward + gamma * V_next - V_t",
            "future": "future_component = raw_advantage - td_residual = gamma * gae_lambda * A_{t+1}",
            "delta_value": "V_{t+1} - V_t with V = value_old of the respective rows (same update, same V_old evaluation)",
            "sign": "sign(x) in {-1, 0, +1}; 'differs' compares these three-valued signs",
            "spread": "pstdev (population), IQR (statistics.quantiles n=4, inclusive), and median absolute deviation about the median",
            "descriptive_only": "all quantities are descriptive and non-counterfactual",
        },
    }
    summary["integrity_checks"] = dict(sorted(checks.items()))
    outputs["artifact_manifest.json"] = _dump(manifest)
    outputs["evidence_summary.json"] = _dump(summary)
    for name in ORIGINAL_COPIES:
        with open(os.path.join(dirs[NEW], name), "rb") as f:
            outputs["originals/" + name] = f.read()
    return outputs, checks


def from_derived():
    summary = _json(os.path.join(PACKAGE_DIR, "evidence_summary.json"))
    ok = True
    for arm in ARMS:
        d = lambda n: _jsonl(os.path.join(PACKAGE_DIR, "derived", arm, n))
        blocks = mechanism_blocks(d("immediate_fd_credit_rows.jsonl"), d("actor_gradient_rows.jsonl"),
                                  d("evaluation_trajectory.jsonl"), d("owner_transition_rows.jsonl"))
        for b, v in blocks.items():
            same = _dump(v) == _dump(summary["arms"][arm][b])
            print("%s %s.%s (from derived rows)" % ("MATCH" if same else "DIFFER", arm, b))
            ok = ok and same
        rows = d("owner_transition_rows.jsonl")
        td_ok = all(abs(r["transition_reward"] + r["gamma"] * r["next_value_old"] - r["value_old"] - r["td_residual"]) <= TD_TOL
                    for r in rows)
        print("%s %s owner-transition TD reconstruction from derived rows" % ("PASS" if td_ok else "FAIL", arm))
        ok = ok and td_ok
    same = _dump_rows(project_eval(os.path.join(PACKAGE_DIR, "originals"))) == \
        open(os.path.join(PACKAGE_DIR, "derived", NEW, "evaluation_trajectory.jsonl"), "rb").read()
    print("%s derived/%s/evaluation_trajectory.jsonl reprojected from originals/" % ("MATCH" if same else "DIFFER", NEW))
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
