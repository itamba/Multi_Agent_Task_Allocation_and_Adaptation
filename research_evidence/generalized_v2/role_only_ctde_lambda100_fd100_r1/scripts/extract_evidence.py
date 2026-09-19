"""Read-only extractor / checker for the role-only CTDE gae_lambda = 1.0 FD100 diagnostic.

Standard library only. Reads the ORIGINAL external run directories of four arms and never
writes to them:

  lambda100          graph_rl_v2_role_only_ctde_lambda100_fd100_r1_seed3000000_68055e3  (new)
  role_only_lambda095 graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3         (PRIMARY comparator, same code)
  symmetric          graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a   (secondary, cross-version)
  explicit_readout   graph_rl_v2_explicit_ego_readout_ctde_fd100_r1_seed3000000_1a1e0c9 (secondary, cross-version)

and produces every generated file of this package:

  artifact_manifest.json      SHA-256 and bytes of all four runs' artifacts (+ tree digests)
  evidence_summary.json       identity, validity, accounting, benchmark and config boundaries,
                              held-out trajectories, the lambda = 1 structural identities,
                              credit, action-credit, actor-gradient and owner-transition
                              blocks for all four arms, integrity checks
  derived/<arm>/*.jsonl       per-row projections sufficient to recompute every mechanism,
                              structural-identity and owner-transition quantity in the summary
  originals/*                 byte-identical copies of the new run's small identity,
                              configuration, summary and record files

Every mechanism quantity uses the common range: training iterations 0..99 and evaluation
rounds with updates_completed <= 100.

Usage (from the repository root):

  python research_evidence/generalized_v2/role_only_ctde_lambda100_fd100_r1/scripts/extract_evidence.py
  python research_evidence/generalized_v2/role_only_ctde_lambda100_fd100_r1/scripts/extract_evidence.py --check
  python research_evidence/generalized_v2/role_only_ctde_lambda100_fd100_r1/scripts/extract_evidence.py --from-derived

``--write`` (default) regenerates the package files. ``--check`` regenerates them in memory,
fails unless every one is byte-identical to the committed file, and runs the integrity checks.
``--from-derived`` needs NO run directory: it recomputes the trajectory, structural-identity,
credit, action-credit, gradient and owner-transition blocks from the committed ``derived/``
rows alone, compares them with ``evidence_summary.json`` and re-runs the row-level identity
checks.
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

NEW, ROLE, SYM, EXP = "lambda100", "role_only_lambda095", "symmetric", "explicit_readout"
ARMS = (NEW, ROLE, SYM, EXP)
LAMBDA_ARMS = (NEW, ROLE)
ROLE_SHA = "68055e39768d5fa601e5960a9f08823b9e65c08f"
RUNS = {
    NEW: {"run_id": "graph_rl_v2_role_only_ctde_lambda100_fd100_r1_seed3000000_68055e3",
          "measured_code_sha": ROLE_SHA,
          "checkout": "isolated detached worktree C:\\grolelambda1 (no branch)",
          "critic": "role-only acting-ego central state, mean-pool readout",
          "gae_lambda": 1.0, "role": "NEW run (intervention: ctde.gae_lambda 0.95 -> 1.0)",
          "n_iterations": 100, "n_successful": 800, "n_eval_rounds": 5},
    ROLE: {"run_id": "graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3",
           "measured_code_sha": ROLE_SHA,
           "checkout": "branch task/ctde-acting-ego-conditioning",
           "critic": "role-only acting-ego central state, mean-pool readout",
           "gae_lambda": 0.95, "role": "PRIMARY comparator (same measured code SHA)",
           "n_iterations": 100, "n_successful": 800, "n_eval_rounds": 5},
    SYM: {"run_id": "graph_rl_v2_semantic_ctde_grad_diag_fd100_r1_seed3000000_6ed964a",
          "measured_code_sha": "6ed964a1abd09de2130aee3d0d314c8f32165056",
          "checkout": "branch task/v2-ctde-gradient-pressure-diagnostics",
          "critic": "symmetric central state (no distinguished agent), mean-pool readout",
          "gae_lambda": 0.95, "role": "SECONDARY context, cross-version; 150 updates, compared through update 100",
          "n_iterations": 150, "n_successful": 1200, "n_eval_rounds": 7},
    EXP: {"run_id": "graph_rl_v2_explicit_ego_readout_ctde_fd100_r1_seed3000000_1a1e0c9",
          "measured_code_sha": "1a1e0c953c54e9d3f46c871158d5ab6bdd881f24",
          "checkout": "branch task/ctde-acting-ego-conditioning",
          "critic": "role-only acting-ego central state + explicit readout [mean pool ; acting-ego embedding]",
          "gae_lambda": 0.95, "role": "SECONDARY context, cross-version",
          "n_iterations": 100, "n_successful": 800, "n_eval_rounds": 5},
}
MANIFEST_PATH = "C:\\gra\\benchmarks\\v2_preflight_seed2000000_ae42cb0\\benchmark_manifest.json"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"
WORKTREE = "C:\\grolelambda1"
# Earlier committed indexes of the three comparator runs; byte equality with them proves the
# comparator originals are unchanged since those reviews.
_RE = os.path.join(REPO_ROOT, "research_evidence", "generalized_v2")
PRIOR_INDEX = {
    ROLE: (os.path.join(_RE, "acting_ego_ctde_fd100_r1", "artifact_manifest.json"), "acting_ego"),
    SYM: (os.path.join(_RE, "semantic_ctde_grad_diag_r1", "artifact_manifest.json"), "B_fd100"),
    EXP: (os.path.join(_RE, "explicit_ego_readout_ctde_fd100_r1", "artifact_manifest.json"), "explicit_readout"),
}
TOP_LEVEL_FILES = (
    "authorized_plan.json", "preflight.json", "launch.cmd", "launch_record.json",
    "run_config.json", "run_summary.json", "train_records.jsonl", "eval_records.jsonl",
    "episode_failures.jsonl", "episode_outcomes.jsonl",
    "train_actor_gradient_diagnostics.jsonl", "train_credit_diagnostics.jsonl",
    "training_console.log", "invocation_start_local.txt", "native_exit_code.txt",
)
NEW_EXTRA_FILES = ("train_config_preset.json",)
TREE_DIRS = ("checkpoints", "plots", "scenarios")
ORIGINAL_COPIES = (
    "authorized_plan.json", "preflight.json", "train_config_preset.json", "launch.cmd",
    "launch_record.json", "run_config.json", "run_summary.json", "train_records.jsonl",
    "eval_records.jsonl", "train_actor_gradient_diagnostics.jsonl", "episode_failures.jsonl",
    "invocation_start_local.txt", "native_exit_code.txt",
)
AUTHORIZED_DIFF_VS_ROLE = ["/ctde/gae_lambda", "/output_dir"]
EXPECTED_DIFF = {ROLE: AUTHORIZED_DIFF_VS_ROLE, EXP: AUTHORIZED_DIFF_VS_ROLE,
                 SYM: ["/ctde/gae_lambda", "/n_iterations", "/output_dir"]}
PREFLIGHT_CHECKS = (
    "output_dir_fresh", "head_is_measured_sha", "working_tree_clean_incl_untracked",
    "detached_head", "match_aou_from_worktree", "blade_vendored_engine_identical_to_measured_tree",
    "comparator_config_loaded", "config_diff_only_authorized", "config_source_is_preset_unmodified",
    "role_only_acting_ego_present", "explicit_readout_absent", "benchmark_identity",
    "held_out_entire_manifest", "development_profile_matches_comparator", "solver_environment",
    "seed_schedule_matches_comparator",
)
MAX_IT = 100
EVAL_POINTS = (0, 25, 50, 75, 100)
WINDOWS = ((0, 25), (25, 50), (50, 75), (75, 100))
ACTION_WINDOWS = (("0-99", 0, 100), ("50-74", 50, 75), ("75-99", 75, 100))
ABORT, PLAN = "SELF_PRESERVATION_ABORT", "PLAN_COMPLIANCE"
TD_TOL = 1e-5            # owner-transition / GAE recurrence tolerance (as in the prior packages)
IDENTITY_TOL = 1e-9      # lambda = 1 telescoping identities (float64 values in the artifact)


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
    # utf-8-sig: the role-only lambda 0.95 run's launch_record.json carries a UTF-8 BOM (read, never rewritten)
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


def _diff_paths(a, b):
    fa, fb = _flatten(a), _flatten(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>"))


def _med(x):
    return statistics.median(x) if x else None


def _mean(x):
    return statistics.mean(x) if x else None


def _sign(x):
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------- manifest
def manifest_for(run_dir, extra=()):
    files = {}
    for name in TOP_LEVEL_FILES + tuple(extra):
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
    for rs in ep.values():
        rs.sort(key=lambda r: r["episode_decision_ordinal"])
        if [r["episode_decision_ordinal"] for r in rs] != list(range(len(rs))):
            ok_contig = False
        if len({r["episode_seed"] for r in rs}) != 1:
            ok_seed = False
    checks["%s_decision_ordinals_contiguous_no_duplicates_no_gaps" % arm] = ok_contig
    checks["%s_one_seed_per_episode" % arm] = ok_seed
    return ep


def project_all_transitions(ep):
    """EVERY CTDE transition (all wake kinds), with the next decision's value / advantage."""
    out = []
    for (it, ei), rs in sorted(ep.items()):
        for idx, r in enumerate(rs):
            n = rs[idx + 1] if idx + 1 < len(rs) else None
            out.append({
                "iteration": it, "episode_index": ei,
                "episode_decision_ordinal": r["episode_decision_ordinal"],
                "is_episode_terminal": n is None,
                "wake_kind": r["wake_kind"],
                "episode_reward": r["episode_reward"], "transition_reward": r["transition_reward"],
                "gamma": r["gamma"], "gae_lambda": r["gae_lambda"],
                "value_old": r["value_old"], "value_target": r["value_target"],
                "raw_advantage": r["raw_advantage"], "td_residual": r["td_residual"],
                "next_value_old": None if n is None else n["value_old"],
                "next_raw_advantage": None if n is None else n["raw_advantage"],
            })
    return out


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
                "episode_reward": r["episode_reward"],
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
def structural_block(rows):
    """Telescoping identities over EVERY transition, and the TD / GAE recurrences."""
    nt = [r for r in rows if not r["is_episode_terminal"]]
    te = [r for r in rows if r["is_episode_terminal"]]
    e_adv = [abs(r["raw_advantage"] - (r["episode_reward"] - r["value_old"])) for r in rows]
    e_tgt = [abs(r["value_target"] - r["episode_reward"]) for r in rows]
    return {
        "population": "EVERY CTDE transition (all wake kinds), iterations 0..99",
        "n_transitions": len(rows), "n_nonterminal": len(nt), "n_terminal": len(te),
        "n_episodes": len({(r["iteration"], r["episode_index"]) for r in rows}),
        "gamma_values": sorted({r["gamma"] for r in rows}),
        "gae_lambda_values": sorted({r["gae_lambda"] for r in rows}),
        "telescoping_identities": {
            "max_abs_err_raw_advantage_minus_episode_reward_minus_value_old": max(e_adv),
            "median_abs_err_raw_advantage_minus_episode_reward_minus_value_old": _med(e_adv),
            "n_rows_err_gt_identity_tol_raw_advantage": sum(e > IDENTITY_TOL for e in e_adv),
            "max_abs_err_value_target_minus_episode_reward": max(e_tgt),
            "median_abs_err_value_target_minus_episode_reward": _med(e_tgt),
            "n_rows_err_gt_identity_tol_value_target": sum(e > IDENTITY_TOL for e in e_tgt),
            "identity_tol": IDENTITY_TOL,
            "note": "the identities hold by construction only when gamma = 1, lambda = 1 and reward is terminal-only; for a lambda = 0.95 arm the errors are reported as the size of the retained intermediate bootstrap, not as a check",
        },
        "value_target_definition": {
            "max_abs_err_value_target_minus_raw_advantage_minus_value_old": max(
                abs(r["value_target"] - (r["raw_advantage"] + r["value_old"])) for r in rows)},
        "reward_structure": {
            "max_abs_nonterminal_transition_reward": max((abs(r["transition_reward"]) for r in nt), default=0.0),
            "max_abs_terminal_transition_reward_minus_episode_reward": max(
                abs(r["transition_reward"] - r["episode_reward"]) for r in te),
            "n_episodes_with_nonconstant_episode_reward": sum(
                len(v) > 1 for v in _group(rows, lambda r: r["episode_reward"]).values()),
        },
        "td_recurrence": {
            "max_abs_err_nonterminal_td_minus_r_plus_gamma_Vnext_minus_V": max(
                (abs(r["td_residual"] - (r["transition_reward"] + r["gamma"] * r["next_value_old"] - r["value_old"]))
                 for r in nt), default=0.0),
            "max_abs_err_terminal_td_minus_r_minus_V": max(
                abs(r["td_residual"] - (r["transition_reward"] - r["value_old"])) for r in te),
            "denominators": {"nonterminal": len(nt), "terminal": len(te)},
        },
        "gae_recurrence": {
            "max_abs_err_nonterminal_A_minus_td_minus_gamma_lambda_A_next": max(
                (abs(r["raw_advantage"] - r["td_residual"] - r["gamma"] * r["gae_lambda"] * r["next_raw_advantage"])
                 for r in nt), default=0.0),
            "max_abs_err_terminal_A_minus_td": max(abs(r["raw_advantage"] - r["td_residual"]) for r in te),
        },
    }


def _group(rows, key):
    g = defaultdict(set)
    for r in rows:
        g[(r["iteration"], r["episode_index"])].add(key(r))
    return g


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
    return {
        "population": "immediate_fuel_damage credit rows with measurement_join.is_fd_selected_ego = true, iterations 0..99",
        "population_n": len(rows), "nonterminal_n": len(nt), "terminal_n": len(rows) - len(nt),
        "severity_counts": dict(sorted(Counter(r["severity"] for r in rows).items())),
        "within_update_severe_minus_mild_0_99": {k: _within(rows, fn) for k, fn in f.items()},
        "within_update_severe_minus_mild_50_99": {k: _within(rows, fn, 50, 100) for k, fn in f.items()},
        "value_old_within_update_by_window": {
            "%d-%d" % (lo, hi - 1): _within(rows, f["value_old"], lo, hi) for lo, hi in WINDOWS},
        "nonterminal_local_vs_total": {
            "median_abs_td_residual_over_abs_raw_advantage": _med(
                [abs(r["td_residual"]) / abs(r["raw_advantage"]) for r in nz]),
            "n_rows_in_ratio": len(nz),
            "n_nonterminal_rows_excluded_raw_advantage_zero": len(nt) - len(nz),
            "median_abs_local_td_residual": _med([abs(r["td_residual"]) for r in nt]),
            "median_abs_future_gamma_lambda_A_next": _med([abs(future(r)) for r in nt]),
            "mean_abs_local_over_mean_abs_future": (
                _mean([abs(r["td_residual"]) for r in nt]) / _mean([abs(future(r)) for r in nt])) if nt else None,
            "fraction_sign_td_differs_from_sign_raw_advantage": (
                sum(_sign(r["td_residual"]) != _sign(r["raw_advantage"]) for r in nt) / len(nt)) if nt else None,
            "fraction_abs_future_gt_abs_local": (
                sum(abs(future(r)) > abs(r["td_residual"]) for r in nt) / len(nt)) if nt else None,
            "within_update_local_0_99": _within(nt, f["td_residual"]),
            "within_update_future_0_99": _within(nt, future),
            "within_update_local_50_99": _within(nt, f["td_residual"], 50, 100),
            "within_update_future_50_99": _within(nt, future, 50, 100),
        },
    }


def action_credit_block(rows):
    out = {"definition": "mean normalized_advantage over sampled ABORT rows minus mean over sampled PLAN rows, per severity and window; descriptive sampled-action association, NOT a counterfactual Q value"}
    for name, lo, hi in ACTION_WINDOWS:
        out[name] = {}
        for sev in ("mild", "severe"):
            rs = [r for r in rows if r["severity"] == sev and lo <= r["iteration"] < hi]
            a = [r["normalized_advantage"] for r in rs if r["selected_meta_action_name"] == ABORT]
            p = [r["normalized_advantage"] for r in rs if r["selected_meta_action_name"] == PLAN]
            out[name][sev] = {
                "n_rows": len(rs), "n_abort": len(a), "n_plan": len(p), "n_other": len(rs) - len(a) - len(p),
                "selected_action_counts": dict(sorted(Counter(r["selected_meta_action_name"] for r in rs).items())),
                "mean_normalized_advantage_abort": _mean(a), "mean_normalized_advantage_plan": _mean(p),
                "abort_minus_plan": (_mean(a) - _mean(p)) if a and p else None,
            }
    return out


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
    by_window = {}
    for lo, hi in ((0, 50), (50, 100)):
        by_window["%d-%d" % (lo, hi - 1)] = {
            "same_ego_next": _split_stats([r for r in same if lo <= r["iteration"] < hi], n),
            "different_ego_next": _split_stats([r for r in diff if lo <= r["iteration"] < hi], n)}
    return {
        "population": "NONTERMINAL immediate_fuel_damage rows of the FD-selected ego, iterations 0..99, joined to the next global decision of the same episode",
        "nonterminal_n": n,
        "all_nonterminal": _split_stats(rows, n),
        "same_ego_next": _split_stats(same, n),
        "different_ego_next": _split_stats(diff, n),
        "by_iteration_window": by_window,
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
            "note": "with a terminal-only reward and gamma = 1, a nonterminal td_residual is exactly V_{t+1} - V_t. Under lambda = 1 it remains a DIAGNOSTIC of consecutive values but is NOT the actor's total advantage: raw_advantage telescopes to R - V_t, and future_component = A_{t+1} = R - V_{t+1}",
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


def row_checks(arm, all_rows, credit_rows, ot_rows, checks):
    """Row-level identity checks, run identically from originals and from derived rows."""
    a = arm + "_"
    nt = [r for r in all_rows if not r["is_episode_terminal"]]
    te = [r for r in all_rows if r["is_episode_terminal"]]
    checks[a + "td_recurrence_nonterminal"] = all(
        abs(r["td_residual"] - (r["transition_reward"] + r["gamma"] * r["next_value_old"] - r["value_old"])) <= TD_TOL for r in nt)
    checks[a + "td_recurrence_terminal"] = all(
        abs(r["td_residual"] - (r["transition_reward"] - r["value_old"])) <= TD_TOL for r in te)
    checks[a + "gae_recurrence_all_transitions"] = all(
        abs(r["raw_advantage"] - r["td_residual"] - r["gamma"] * r["gae_lambda"] * r["next_raw_advantage"]) <= TD_TOL
        for r in nt) and all(abs(r["raw_advantage"] - r["td_residual"]) <= TD_TOL for r in te)
    checks[a + "terminal_only_reward"] = (all(r["transition_reward"] == 0 for r in nt)
                                          and all(r["transition_reward"] == r["episode_reward"] for r in te))
    checks[a + "gamma_is_one"] = {r["gamma"] for r in all_rows} == {1.0}
    checks[a + "gae_lambda_as_configured"] = {r["gae_lambda"] for r in all_rows} == {RUNS[arm]["gae_lambda"]}
    checks[a + "one_terminal_per_episode"] = len(te) == len({(r["iteration"], r["episode_index"]) for r in all_rows})
    checks[a + "credit_population_800"] = len(credit_rows) == 800
    checks[a + "owner_transition_n_equals_credit_nonterminal_n"] = (
        len(ot_rows) == sum(not r["is_episode_terminal"] for r in credit_rows))
    checks[a + "owner_next_ordinal_is_current_plus_one"] = all(
        r["next_episode_decision_ordinal"] == r["episode_decision_ordinal"] + 1 for r in ot_rows)
    checks[a + "owner_td_equals_delta_value"] = all(abs(r["td_residual"] - r["delta_value"]) <= TD_TOL for r in ot_rows)
    if arm == NEW:
        checks[a + "lambda1_raw_advantage_equals_R_minus_V_every_transition"] = all(
            abs(r["raw_advantage"] - (r["episode_reward"] - r["value_old"])) <= IDENTITY_TOL for r in all_rows)
        checks[a + "lambda1_value_target_equals_R_every_transition"] = all(
            abs(r["value_target"] - r["episode_reward"]) <= IDENTITY_TOL for r in all_rows)
        checks[a + "lambda1_credit_rows_raw_advantage_equals_R_minus_V"] = all(
            abs(r["raw_advantage"] - (r["episode_reward"] - r["value_old"])) <= IDENTITY_TOL for r in credit_rows)


def mechanism_blocks(all_rows, credit_rows, grad_rows, eval_rows, ot_rows):
    out = {"evaluation_trajectory": eval_rows, "credit": credit_block(credit_rows),
           "action_credit": action_credit_block(credit_rows),
           "actor_gradient": gradient_block(grad_rows),
           "owner_transition_audit": owner_transition_block(ot_rows)}
    if all_rows is not None:
        out["structural_identity"] = structural_block(all_rows)
    return out


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
    checks[a + "train_config_gae_lambda"] = rc["train_config"]["ctde"]["gae_lambda"] == spec["gae_lambda"]
    checks[a + "updates"] = s["updates_completed"] == spec["n_iterations"] == s["n_productive_iterations"]
    checks[a + "successful_quota"] = s["train_episodes_successful"] == spec["n_successful"]
    checks[a + "attempts_within_budget"] = s["train_episodes_attempted"] <= 12 * spec["n_iterations"]
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
    checks[a + "held_out_zero_overlap_entire_manifest"] = (
        be["held_out_verified"] is True and be["held_out_overlap_count"] == 0
        and be.get("held_out_checked_over") == "entire_manifest_all_profiles")
    return {
        "run_id": spec["run_id"], "run_dir": run_dir, "measured_code_sha": spec["measured_code_sha"],
        "checkout": spec["checkout"], "critic": spec["critic"], "role": spec["role"],
        "gae_lambda": spec["gae_lambda"], "provenance_git": git,
        "config_source": rc.get("config_source"),
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
                      "n_worlds": be["evaluation_profile"]["n_worlds"],
                      "n_members": be["evaluation_profile"]["n_members"],
                      "group_keys_sha256": be["evaluation_profile"]["group_keys_sha256"],
                      "held_out_against_train_band": be["held_out_against_train_band"],
                      "held_out_overlap_count": be["held_out_overlap_count"],
                      "held_out_verified": be["held_out_verified"],
                      "held_out_checked_over": be.get("held_out_checked_over")},
    }


def new_run_specifics(new_dir, role_dir, checks):
    pf = _json(os.path.join(new_dir, "preflight.json"))
    plan = _json(os.path.join(new_dir, "authorized_plan.json"))
    lr = _json(os.path.join(new_dir, "launch_record.json"))
    rc = _json(os.path.join(new_dir, "run_config.json"))
    preset = _json(os.path.join(new_dir, "train_config_preset.json"))
    role_rc = _json(os.path.join(role_dir, "run_config.json"))
    with open(os.path.join(new_dir, "training_console.log"), "rb") as f:
        lines = f.read().decode("utf-8", errors="replace").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("TRAINING SUMMARY ("))
    with open(os.path.join(new_dir, "invocation_start_local.txt"), encoding="utf-8") as f:
        start_end = f.read().splitlines()
    with open(os.path.join(new_dir, "launch.cmd"), encoding="ascii") as f:
        launch_cmd = f.read()
    n = NEW + "_"
    git = rc["provenance"]["git"]
    # ---- measured checkout identity
    checks[n + "preflight_passed_all_checks"] = (pf["passed"] is True and pf["failed_checks"] == []
                                                  and sorted(pf["checks"]) == sorted(PREFLIGHT_CHECKS)
                                                  and all(v["passed"] for v in pf["checks"].values()))
    checks[n + "preflight_head_equals_measured_sha"] = pf["code"]["checkout_head"] == ROLE_SHA
    checks[n + "preflight_tree_clean_incl_untracked"] = pf["code"]["porcelain_untracked_all"] == ""
    checks[n + "preflight_detached_worktree"] = (pf["code"]["detached"] is True and pf["code"]["worktree"] == WORKTREE
                                                  and pf["checks"]["detached_head"]["detail"] == "HEAD")
    checks[n + "run_config_repo_root_is_worktree"] = git["repo_root"] == WORKTREE and git["branch"] == "HEAD"
    checks[n + "match_aou_imported_from_worktree"] = rc["provenance"]["packages"]["match_aou"]["path"].lower().startswith(WORKTREE.lower())
    bl = pf["checks"]["blade_vendored_engine_identical_to_measured_tree"]["detail"]
    checks[n + "blade_engine_tree_equals_measured_tree"] = (
        bl["measured_tree"] == bl["loaded_checkout_tree"] and bl["loaded_checkout_blade_dirty"] == "")
    checks[n + "preflight_role_only_present_explicit_absent"] = (
        pf["checks"]["role_only_acting_ego_present"]["passed"] and pf["checks"]["explicit_readout_absent"]["passed"])
    checks[n + "preflight_bonmin_probe_ok"] = (pf["trainer_provenance_preview"]["solver"]["bonmin"]["probe"] == "ok"
                                               and rc["provenance"]["solver"]["bonmin"]["probe"] == "ok")
    # ---- configuration
    preset_tc = {k: v for k, v in preset.items() if not k.startswith("_")}
    exp_preset = json.loads(json.dumps(role_rc["train_config"]))
    exp_preset["ctde"]["gae_lambda"] = 1.0
    exp_preset["output_dir"] = rc["train_config"]["output_dir"]
    checks[n + "preset_equals_comparator_train_config_with_two_paths_replaced"] = preset_tc == exp_preset
    checks[n + "run_config_train_config_equals_preflight"] = rc["train_config"] == pf["resolved_configuration"]["train_config"]
    checks[n + "run_config_train_config_equals_preset"] = rc["train_config"] == preset_tc
    checks[n + "preflight_unauthorized_differences_empty"] = (
        pf["config_comparison_vs_primary_comparator"]["unauthorized_differences"] == [])
    checks[n + "config_source_config_file_no_cli_overrides"] = (
        rc["config_source"]["resolved_from"] == "config_file" and rc["config_source"]["cli_overrides"] == []
        and os.path.basename(rc["config_source"]["path"]) == "train_config_preset.json")
    checks[n + "comparator_config_source_cli_defaults"] = role_rc["config_source"]["resolved_from"] == "cli_defaults"
    checks[n + "preflight_comparator_run_config_sha256"] = (
        pf["primary_comparator"]["run_config_sha256"] == _sha256(os.path.join(role_dir, "run_config.json")))
    checks[n + "preflight_preset_sha256"] = (
        pf["resolved_configuration"]["preset_sha256"] == _sha256(os.path.join(new_dir, "train_config_preset.json")))
    checks[n + "seed_schedule_equals_comparator"] = rc["provenance"]["seeds"] == role_rc["provenance"]["seeds"]
    # ---- launch / benchmark / console
    checks[n + "launch_record_hashes_match"] = (
        lr["launch_cmd_sha256"] == _sha256(os.path.join(new_dir, "launch.cmd"))
        and lr["authorized_plan_json_sha256"] == _sha256(os.path.join(new_dir, "authorized_plan.json"))
        and lr["preflight_json_sha256"] == _sha256(os.path.join(new_dir, "preflight.json"))
        and lr["train_config_preset_json_sha256"] == _sha256(os.path.join(new_dir, "train_config_preset.json")))
    checks[n + "launch_cmd_runs_worktree_nlp_env_preset_only"] = (
        'set "REPO=C:\\grolelambda1"' in launch_cmd and 'set "PYTHONPATH=src"' in launch_cmd
        and "conda run -n nlp_env --no-capture-output python -m match_aou.rl.training.graph_train" in launch_cmd
        and '--config "%RUN_DIR%\\train_config_preset.json"' in launch_cmd)
    checks[n + "invocation_argv_is_config_only"] = rc["provenance"]["invocation"]["argv"][1:] == [
        "--config", rc["config_source"]["path"]]
    checks[n + "plan_measured_sha"] = plan["measured_code_sha"] == ROLE_SHA
    checks[n + "preflight_manifest_sha256_and_id"] = (pf["manifest_check"]["file_sha256"] == MANIFEST_SHA256
                                                      and pf["manifest_check"]["manifest_id"] == MANIFEST_ID)
    checks[n + "console_no_traceback_no_crash"] = not any(("Traceback" in l) or ("CRASH" in l) for l in lines)
    return {
        "preflight_code": pf["code"], "preflight_checks": pf["checks"],
        "authorized_plan_authorization": plan["authorization"], "authorized_plan_budget": plan["budget"],
        "invocation_argv": rc["provenance"]["invocation"]["argv"],
        "invocation_cwd": rc["provenance"]["invocation"]["cwd"],
        "python_executable": rc["provenance"]["invocation"]["python_executable"],
        "resolved_train_config": rc["train_config"],
        "config_source": rc["config_source"],
        "comparator_config_source": role_rc["config_source"],
        "invocation_start_end_local": start_end,
        "console_summary_block": lines[start:],
        "console_n_lines_containing_Traceback": sum("Traceback" in l for l in lines),
        "console_n_lines_containing_CRASH": sum("CRASH" in l for l in lines),
        "launch_record_process_tree": lr["process_tree"],
    }


# ------------------------------------------------------------------------ build
def build(runs_root):
    checks, outputs = {}, {}
    dirs = {arm: os.path.join(runs_root, RUNS[arm]["run_id"]) for arm in ARMS}

    manifest = {"runs": {arm: dict(RUNS[arm], run_dir=dirs[arm],
                                   **manifest_for(dirs[arm], NEW_EXTRA_FILES if arm == NEW else ()))
                         for arm in ARMS},
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

    tcs = {arm: _json(os.path.join(dirs[arm], "run_config.json"))["train_config"] for arm in ARMS}
    cfg = {"vs_" + arm: _diff_paths(tcs[arm], tcs[NEW]) for arm in (ROLE, SYM, EXP)}
    checks["config_vs_role_only_lambda095_differs_only_in_gae_lambda_and_output_dir"] = cfg["vs_" + ROLE] == EXPECTED_DIFF[ROLE]
    checks["config_vs_explicit_readout_differs_only_in_gae_lambda_and_output_dir"] = cfg["vs_" + EXP] == EXPECTED_DIFF[EXP]
    checks["config_vs_symmetric_differs_only_in_gae_lambda_n_iterations_output_dir"] = cfg["vs_" + SYM] == EXPECTED_DIFF[SYM]
    checks["config_vs_role_only_values"] = (tcs[ROLE]["ctde"]["gae_lambda"] == 0.95 and tcs[NEW]["ctde"]["gae_lambda"] == 1.0
                                            and tcs[NEW]["output_dir"] == dirs[NEW])
    full_new = _json(os.path.join(dirs[NEW], "run_config.json"))
    full_role = _json(os.path.join(dirs[ROLE], "run_config.json"))
    full_diff = _diff_paths(full_role, full_new)
    checks["full_run_config_differences_vs_role_only_outside_train_config_are_provenance_or_derived"] = all(
        p.startswith(("/train_config/ctde/gae_lambda", "/train_config/output_dir", "/provenance/", "/config_source/",
                      "/training/ctde/gae_lambda"))
        for p in full_diff)

    arms_summary = {}
    for arm in ARMS:
        ep = episode_sequences(load_credit(dirs[arm]), checks, arm)
        all_rows = project_all_transitions(ep) if arm in LAMBDA_ARMS else None
        credit = project_credit(ep)
        ot = project_owner_transitions(ep)
        grad = project_gradient(dirs[arm])
        ev = project_eval(dirs[arm])
        if arm in LAMBDA_ARMS:
            row_checks(arm, all_rows, credit, ot, checks)
        checks["%s_eval_points_common_range" % arm] = [e["updates_completed"] for e in ev] == list(EVAL_POINTS)
        checks["%s_eval_rounds_complete_denominators" % arm] = all(
            e["n_successful"] == e["n_attempted"] == 60 and e["macro_n_base_cells_defined"] == 10
            and e["n_groups_metric_eligible"] == e["n_groups_complete"] == 20 for e in ev)
        checks["%s_gradient_one_row_per_update_0_99" % arm] = sorted(r["iteration"] for r in grad) == list(range(MAX_IT))
        arms_summary[arm] = dict(validity(arm, dirs[arm], checks), **mechanism_blocks(all_rows, credit, grad, ev, ot))
        base = "derived/%s/" % arm
        if all_rows is not None:
            outputs[base + "all_transition_rows.jsonl"] = _dump_rows(all_rows)
        outputs[base + "immediate_fd_credit_rows.jsonl"] = _dump_rows(credit)
        outputs[base + "owner_transition_rows.jsonl"] = _dump_rows(ot)
        outputs[base + "actor_gradient_rows.jsonl"] = _dump_rows(grad)
        outputs[base + "evaluation_trajectory.jsonl"] = _dump_rows(ev)

    arms_summary[NEW]["new_run_specifics"] = new_run_specifics(dirs[NEW], dirs[ROLE], checks)
    summary = {
        "package": "role-only acting-ego CTDE gae_lambda = 1.0 FD100 DEVELOPMENT diagnostic -- compact evidence package",
        "formal_status": {
            "measured_code_sha": ROLE_SHA,
            "measured_code_note": "the role-only acting-ego implementation measured by the primary comparator; run from an isolated detached worktree, NOT from PR #75's later explicit-readout head",
            "intervention": "ctde.gae_lambda 0.95 -> 1.0 (plus a fresh output_dir); nothing else in the resolved train_config",
            "run": "completed",
            "scientific_measurement_verdict": "APPROVE -- VALID DEVELOPMENT DIAGNOSTIC MEASUREMENT (GPT review of this evidence package, 2026-09-19; recorded in docs/history/measurements.md section 14)",
            "evidence_class": "DEVELOPMENT only; one seed stream; repeated evaluation on the same frozen development worlds; not randomized; not confirmatory",
            "primary_comparison": "same-code role-only lambda = 0.95 (graph_rl_v2_acting_ego_ctde_fd100_r1_seed3000000_68055e3) vs lambda = 1.0",
            "secondary_context": "symmetric FD100 (6ed964a) and explicit readout (1a1e0c9): CROSS-VERSION context only, compared through update 100",
            "bitwise_identity": "not claimed: BLADE run-to-run timing nondeterminism; the two lambda arms are separate executions",
            "confirmatory_profile_used": False,
            "new_run_authorized": False, "merge_authorized": False,
        },
        "measured_code_sha": {arm: RUNS[arm]["measured_code_sha"] for arm in ARMS},
        "common_range": "training iterations 0..99; evaluation rounds updates_completed in {0,25,50,75,100}",
        "config_boundaries": dict(
            cfg,
            method="flatten both run_config.json:/train_config objects; list every differing JSON path",
            authorized_vs_role_only=AUTHORIZED_DIFF_VS_ROLE,
            full_run_config_differing_paths_vs_role_only=full_diff,
            config_source_note="config_source.resolved_from differs (lambda100: config_file, the run-local train_config_preset.json with no CLI overrides; role_only_lambda095: cli_defaults). That is PROVENANCE of how the identical train_config was resolved, not a scientific train_config difference; every other differing run_config path outside /train_config is provenance (git repo_root / branch, invocation argv / cwd, collected_at) or the mirrored /training/ctde/gae_lambda",
            code_note="lambda arms share measured code; symmetric and explicit-readout differ at code level (critic state / readout), stated per arm under 'critic'"),
        "comparator_bytes_unchanged_since_prior_index": prior,
        "arms": arms_summary,
        "definitions": {
            "within_update": "per update holding both severities: mean(severe) - mean(mild); median / mean over those updates",
            "nonterminal": "row is not its episode's last global decision (over all wake kinds)",
            "next_decision": "the row with episode_decision_ordinal + 1 in the same (iteration, episode_index) global decision sequence",
            "local": "td_residual = transition_reward + gamma * V_next - V_t (terminal: transition_reward - V_t)",
            "future": "future_component = raw_advantage - td_residual = gamma * gae_lambda * A_{t+1}; with lambda = 1 this is A_{t+1} = R - V_{t+1}",
            "lambda1_telescoping": "with gamma = 1, lambda = 1 and terminal-only reward, A_t = sum of TD residuals = R_episode - V_t and value_target = A_t + V_t = R_episode: no intermediate V_{t+k} (including a value conditioned on a different acting ego) survives in the actor's total advantage; one-step TD residuals still exist and are reported only as diagnostics",
            "delta_value": "V_{t+1} - V_t with V = value_old of the respective rows (same update, same V_old evaluation)",
            "sign": "sign(x) in {-1, 0, +1}; 'differs' compares these three-valued signs",
            "spread": "pstdev (population), IQR (statistics.quantiles n=4, inclusive), and median absolute deviation about the median",
            "action_credit": "descriptive sampled-action association of normalized advantage, NOT a counterfactual Q value",
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
        all_rows = d("all_transition_rows.jsonl") if arm in LAMBDA_ARMS else None
        credit, ot = d("immediate_fd_credit_rows.jsonl"), d("owner_transition_rows.jsonl")
        blocks = mechanism_blocks(all_rows, credit, d("actor_gradient_rows.jsonl"),
                                  d("evaluation_trajectory.jsonl"), ot)
        for b, v in blocks.items():
            same = _dump(v) == _dump(summary["arms"][arm][b])
            print("%s %s.%s (from derived rows)" % ("MATCH" if same else "DIFFER", arm, b))
            ok = ok and same
        if arm in LAMBDA_ARMS:
            chk = {}
            row_checks(arm, all_rows, credit, ot, chk)
            for k, v in sorted(chk.items()):
                agree = v and summary["integrity_checks"].get(k) is True
                print("%s %s (from derived rows)" % ("PASS" if agree else "FAIL", k))
                ok = ok and agree
    same = _dump_rows(project_eval(os.path.join(PACKAGE_DIR, "originals"))) == \
        open(os.path.join(PACKAGE_DIR, "derived", NEW, "evaluation_trajectory.jsonl"), "rb").read()
    print("%s derived/%s/evaluation_trajectory.jsonl reprojected from originals/" % ("MATCH" if same else "DIFFER", NEW))
    same_g = _dump_rows(project_gradient(os.path.join(PACKAGE_DIR, "originals"))) == \
        open(os.path.join(PACKAGE_DIR, "derived", NEW, "actor_gradient_rows.jsonl"), "rb").read()
    print("%s derived/%s/actor_gradient_rows.jsonl reprojected from originals/" % ("MATCH" if same_g else "DIFFER", NEW))
    return 0 if ok and same and same_g else 1


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
