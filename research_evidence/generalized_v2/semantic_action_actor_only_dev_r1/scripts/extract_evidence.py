"""Deterministic, read-only evidence extraction for the GENERALIZED-V2 semantic-action
actor-only DEVELOPMENT run R1 (measured code SHA d4e9f3721e6d151c00be3fe93c3d149df9d31965).

STANDARD LIBRARY ONLY. It imports nothing from ``match_aou``, starts no subprocess, and runs
no training, evaluation, replay, checkpoint inference or benchmark preflight. Source artifacts
are opened read-only; their SHA-256 is taken before and after extraction and the run fails if
any byte changed.

Every reproduced quantity is cross-checked against the value the trainer itself persisted
(per-round ``v2_behaviour`` in ``eval_records.jsonl``, ``run_summary.json``,
``train_records.jsonl``). Any schema, provenance, accounting or arithmetic mismatch raises
:class:`EvidenceError` and writes nothing.

Usage (from anywhere; paths default to the completed run and the archived comparator)::

    python extract_evidence.py --out <package dir> [--copy-run-artifacts]
                               [--verify-against <package dir>/artifact_sha256.txt]

Outputs (all byte-deterministic for identical sources)::

    <out>/artifact_sha256.txt            hashes of every source artifact (in / out of Git)
    <out>/source_manifest.json           path, size, SHA-256, role, disposition per source
    <out>/review_precheck.json           machine-readable validity pre-check (no verdict)
    <out>/run_artifacts/*                byte-identical copies (with --copy-run-artifacts)
    <out>/extracted/eval_immediate_fd_wakes.jsonl
    <out>/extracted/behaviour_summary.json
    <out>/extracted/comparator_r1_behaviour.json
    <out>/extracted/collapse_timeline.json
    <out>/extracted/fd_selected_ego_credit_rows.jsonl
    <out>/extracted/credit_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------- identities
MEASURED_SHA = "d4e9f3721e6d151c00be3fe93c3d149df9d31965"
COMPARATOR_SHA = "ae42cb01677f94868b2873008d87be677e31f0c8"
REPRESENTATION = "semantic_k_plus_2_logmeanexp_v1"
LEGACY_REPRESENTATION = "legacy_node_indexed_joint_k_x_3"  # reader label only
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"

DEFAULT_RUN = Path(r"C:\Users\Itama\PycharmProjects"
                   r"\graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37")
DEFAULT_R1 = Path(r"C:\gra\runs\development\v2_actor_only_r1_seed3000000_ae42cb0")
DEFAULT_ARCHIVE_INDEX = Path(r"C:\gra\metadata\ARTIFACT_INDEX.jsonl")
R1_ARCHIVE_KEY = "v2_actor_only_r1_seed3000000_ae42cb0"

ABORT = "SELF_PRESERVATION_ABORT"
FD_WAKE = "immediate_fuel_damage"
SEVERITIES = ("mild", "severe")
EVAL_PHASES = ("pre_update", "post_update")
N_ROUNDS = 16
N_EVAL_MEMBERS_PER_ROUND = 60
N_BASE_CELLS = 10
N_ITERATIONS = 375
WINDOW = 25
TOL = 1e-9

# role, disposition for each source artifact of the new run
RUN_ARTIFACTS = [
    ("authorized_plan.json", "pre-run authorized bounded plan, written before training", "copied"),
    ("preflight.json", "pre-run technical preflight, written before training", "copied"),
    ("run_config.json", "trainer-resolved configuration and provenance", "copied"),
    ("run_summary.json", "trainer run summary: accounting, final round, schema observation",
     "copied"),
    ("train_records.jsonl", "one record per training update (baseline, n_transitions)", "copied"),
    ("eval_records.jsonl", "one record per evaluation round, incl. per-round v2_behaviour",
     "copied"),
    ("episode_failures.jsonl", "failure ledger (taxonomy, failed seeds)", "copied"),
    ("episode_outcomes.jsonl", "per-episode outcomes incl. wake diagnostics (train + eval)",
     "external_only; eval immediate-FD wakes extracted"),
    ("train_credit_diagnostics.jsonl", "per-transition credit of every productive update",
     "external_only; FD-selected-ego immediate-FD rows extracted"),
    ("checkpoints/ckpt_iter0374.pt", "final actor checkpoint (not loaded or executed)",
     "external_only; hash reference"),
    ("training_console.log", "trainer console output (*.log is git-ignored)",
     "external_only; hash reference"),
    ("native_exit_code.txt", "launcher exit-code file; EMPTY due to the reported cmd redirect "
     "defect (`echo %RC%> file` parses a one-digit code as a handle redirect); not repaired",
     "external_only; hash reference"),
    ("launch.cmd", "the detached launcher that produced the run", "external_only; hash reference"),
    ("invocation_start_local.txt", "launcher start/end local timestamps",
     "external_only; hash reference"),
]
COPIED = [name for name, _, disp in RUN_ARTIFACTS if disp == "copied"]
R1_ARTIFACTS = [
    ("run_config.json", "comparator resolved configuration"),
    ("run_summary.json", "comparator final-round behaviour"),
    ("eval_records.jsonl", "comparator per-round v2_behaviour"),
    ("episode_outcomes.jsonl", "comparator eval immediate-FD wakes (trajectory recomputation)"),
    ("episode_failures.jsonl", "comparator failure ledger (failed-seed comparison)"),
]


class EvidenceError(RuntimeError):
    """A schema, provenance, accounting or arithmetic check failed."""


def require(cond, msg, *args):
    if not cond:
        raise EvidenceError(msg % args if args else msg)


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


# ----------------------------------------------------------------------------- io helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                raise EvidenceError("%s line %d is not valid JSON: %s" % (path, n, exc))
    return out


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise EvidenceError("%s line %d is not valid JSON: %s" % (path, n, exc))


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def write_jsonl(path: Path, rows):
    write_text(path, "".join(json.dumps(r, sort_keys=True, ensure_ascii=True, allow_nan=False)
                             + "\n" for r in rows))


# ----------------------------------------------------------------------------- statistics
def mean(xs):
    xs = [float(x) for x in xs]
    return math.fsum(xs) / len(xs) if xs else None


def desc(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return {"n": 0, "mean": None, "median": None, "std_pop": None, "min": None, "max": None}
    m = math.fsum(xs) / len(xs)
    return {"n": len(xs), "mean": m, "median": statistics.median(xs),
            "std_pop": math.sqrt(math.fsum((x - m) ** 2 for x in xs) / len(xs)),
            "min": min(xs), "max": max(xs)}


def rate(k, n):
    return {"count": k, "denominator": n, "rate": (k / n) if n else None}


# ----------------------------------------------------------------------------- hashing
def hash_sources(run_dir: Path, r1_dir: Path, manifest_path: Path, archive_index: Path):
    entries = []
    for name, role, disp in RUN_ARTIFACTS:
        p = run_dir / name
        require(p.exists(), "missing run artifact %s", p)
        entries.append({"group": "run", "name": name, "absolute_path": str(p),
                        "bytes": p.stat().st_size, "sha256": sha256_file(p), "role": role,
                        "disposition": disp,
                        "git_path": ("run_artifacts/" + name) if disp == "copied" else None})
    for name, role in R1_ARTIFACTS:
        p = r1_dir / name
        require(p.exists(), "missing comparator artifact %s", p)
        entries.append({"group": "comparator_r1", "name": name, "absolute_path": str(p),
                        "bytes": p.stat().st_size, "sha256": sha256_file(p), "role": role,
                        "disposition": "external_only; read for comparator recomputation",
                        "git_path": None})
    entries.append({"group": "benchmark", "name": "benchmark_manifest.json",
                    "absolute_path": str(manifest_path), "bytes": manifest_path.stat().st_size,
                    "sha256": sha256_file(manifest_path),
                    "role": "frozen GENERALIZED-V2 benchmark manifest consumed by the run",
                    "disposition": "external_only; hash reference", "git_path": None})
    entries.append({"group": "archive", "name": "ARTIFACT_INDEX.jsonl",
                    "absolute_path": str(archive_index), "bytes": archive_index.stat().st_size,
                    "sha256": sha256_file(archive_index),
                    "role": "archive index: independent record of comparator key hashes",
                    "disposition": "external_only; read for comparator hash verification",
                    "git_path": None})
    return entries


def render_sha_file(entries) -> str:
    lines = ["# sha256  bytes  group  absolute_path",
             "# Source artifacts of the evidence package. Generated by scripts/extract_evidence.py.",
             "# native_exit_code.txt is EMPTY (0 bytes) because of the launcher redirect defect;",
             "# it is recorded as-is and was not repaired."]
    for e in entries:
        lines.append("%s  %d  %s  %s" % (e["sha256"], e["bytes"], e["group"], e["absolute_path"]))
    return "\n".join(lines) + "\n"


def parse_sha_file(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, size, group, abspath = line.split("  ", 3)
        out[abspath] = (sha, int(size))
    return out


# ----------------------------------------------------------------------------- precheck
def precheck(run_dir, entries, rc, plan, pre, summary, train_records, eval_records, failures,
             r1_rc, r1_failures, archive_entry, manifest_path):
    tc = rc["train_config"]
    prov = rc["provenance"]
    git = prov["git"]
    hashes = {e["name"]: e for e in entries if e["group"] == "run"}
    anomalies = []

    require(git["commit"] == MEASURED_SHA, "run commit %s != %s", git["commit"], MEASURED_SHA)
    require(git["dirty"] is False and int(git["dirty_path_count"]) == 0, "run recorded dirty tree")
    rep = rc.get("training", {}).get("action_representation_id")
    require(rep == REPRESENTATION, "run_config representation %r", rep)
    require(pre["code"]["measured_code_sha"] == MEASURED_SHA, "preflight SHA mismatch")
    require(pre["action_representation"]["id"] == REPRESENTATION, "preflight representation")

    # plan vs resolved configuration
    design = plan["training_design_frozen_to_comparator"]
    ap = rc["training"]["attempt_policy"]
    actual = {
        "episode_design": tc["episode_design"], "training_mode": tc["training_mode"],
        "match_aou_backend": tc["match_aou_backend"],
        "benchmark_profile": tc["benchmark_profile"], "base_seed": tc["base_seed"],
        "n_iterations": tc["n_iterations"],
        "successful_episodes_per_iteration": tc["episodes_per_iteration"],
        "successful_episode_quota_total": ap["successful_episode_quota_total"],
        "generalized_max_attempts_per_iteration": tc["generalized_max_attempts_per_iteration"],
        "max_possible_training_attempts": ap["max_possible_training_attempts"],
        "fuel_damage_mode": tc["fuel_damage_mode"],
        "fuel_damage_probability": tc["fuel_damage_probability"],
        "fuel_damage_mild_probability": tc["fuel_damage_mild_probability"],
        "eval_every": tc["eval_every"], "checkpoint_every": tc["checkpoint_every"],
        "early_stopping": tc["early_stopping"],
        "solver_timeout": None,
    }
    timeout_keys = sorted(k for k in tc if "timeout" in k.lower())
    require(not timeout_keys, "unexpected timeout keys in train_config: %s", timeout_keys)
    plan_vs_actual = {}
    for k, v in design.items():
        if k == "solver_timeout_note":
            continue
        require(k in actual, "plan key %s has no mapped resolved value", k)
        plan_vs_actual[k] = {"plan": v, "resolved": actual[k], "match": v == actual[k]}
        require(v == actual[k], "plan %s=%r but resolved %r", k, v, actual[k])
    require(tc["ppo"] == pre["ppo_configuration"], "PPO config differs from preflight")
    intended = pre["resolved_training_configuration_intended"]
    for k, v in intended.items():
        require(tc.get(k) == v, "train_config %s=%r differs from preflight intended %r",
                k, tc.get(k), v)
    argv_plan = plan["invocation"]["argv"][3:]
    argv_run = prov["invocation"]["argv"][1:]
    require(argv_plan == argv_run, "run argv differs from authorized plan argv")
    require(rc["config_source"]["resolved_from"] == "cli_defaults"
            and rc["config_source"]["cli_overrides"] == [], "config_source not cli_defaults")
    r1_tc = r1_rc["train_config"]
    r1_diff = sorted(k for k in set(tc) | set(r1_tc) if tc.get(k, "<absent>") != r1_tc.get(k, "<absent>"))
    require(r1_diff == ["benchmark_manifest", "output_dir"],
            "train_config differs from comparator beyond output_dir/manifest path: %s", r1_diff)

    # manifest
    bm = rc["episode_design"]["benchmark_manifest"]
    require(bm["manifest_id"] == MANIFEST_ID, "manifest id %s", bm["manifest_id"])
    require(Path(bm["absolute_path"]) == manifest_path, "manifest path differs")
    man_sha = sha256_file(manifest_path)
    require(man_sha == MANIFEST_SHA256, "manifest sha256 %s", man_sha)
    require(bm["evaluation_profile"]["profile"] == "development", "profile not development")
    r1_bm = r1_rc["episode_design"]["benchmark_manifest"]
    require(bm["evaluation_profile"]["group_keys_sha256"]
            == r1_bm["evaluation_profile"]["group_keys_sha256"]
            and bm["evaluation_profile"]["seed_list_sha256"]
            == r1_bm["evaluation_profile"]["seed_list_sha256"]
            and bm["seed_list_sha256"] == r1_bm["seed_list_sha256"],
            "development population identity differs from comparator")
    ho = prov["seeds"]["benchmark_evaluation"]
    require(ho["held_out_verified"] is True and int(ho["held_out_overlap_count"]) == 0,
            "held-out check failed")

    # completion and accounting
    s = summary
    require(s["updates_completed"] == N_ITERATIONS and s["n_iterations"] == N_ITERATIONS,
            "updates %s", s["updates_completed"])
    require(len(train_records) == N_ITERATIONS, "train_records rows %d", len(train_records))
    require(s["train_episodes_successful"] == 3000, "successful %s", s["train_episodes_successful"])
    require(s["train_episodes_attempted"] == s["train_episodes_successful"] + s["train_episodes_failed"],
            "train attempted != ok + failed")
    require(s["train_episodes_attempted"] <= 4500, "attempts exceed budget")
    require(s["train_iterations_at_full_quota"] == N_ITERATIONS, "not all iterations at full quota")
    require(s["accounting_reconciled"] is True, "accounting not reconciled")
    require(sum(int(t["n_successful"]) for t in train_records) == 3000, "train_records successes")
    require(sum(int(t["n_attempted"]) for t in train_records) == s["train_episodes_attempted"],
            "train_records attempts")
    require(len(eval_records) == N_ROUNDS and s["n_eval_rounds"] == N_ROUNDS, "eval rounds")
    require(s["eval_episodes_attempted"] == N_ROUNDS * N_EVAL_MEMBERS_PER_ROUND
            and s["eval_episodes_failed"] == 0, "eval counts")
    require(len(failures) == s["failures_recorded"] == s["train_episodes_failed"], "failure ledger")
    fail_rows = sorted(({"seed": f.get("seed"), "phase": f.get("phase"),
                         "pipeline_stage": f.get("pipeline_stage") or f.get("stage"),
                         "error_type": f.get("error_type")} for f in failures),
                       key=lambda r: r["seed"])
    taxonomy = {"generation", "setup", "run", "reward"}
    require(all(r["pipeline_stage"] in taxonomy for r in fail_rows), "failure outside taxonomy")
    r1_seeds = sorted(f.get("seed") for f in r1_failures)
    es = s["early_stopping"]
    require(es["enabled"] is False and es["triggered"] is False, "early stopping active")
    fin = s["final_eval_selection"]
    require(fin["selected"] is True and fin["identity"]["updates_completed"] == N_ITERATIONS,
            "final round selection")
    fb = s["generalized"]["v2_benchmark"]["final_round_behaviour"]
    require(fb["macro_n_base_cells_defined"] == N_BASE_CELLS and fb["macro_undefined_base_cells"] == [],
            "final macro not defined over 10/10 base cells")

    # schema observations
    oas = s["observed_artifact_schema"]
    require(oas["episode_outcome_schema_versions_observed"] == [4]
            and oas["wake_diagnostics_schema_versions_observed"] == [2]
            and oas["wake_action_representations_observed"] == [REPRESENTATION],
            "observed artifact schema not uniform")
    ocd = s["observed_credit_diagnostics"]
    require(ocd["schema_versions_observed"] == [1]
            and ocd["action_representation_ids_observed"] == [REPRESENTATION]
            and ocd["training_modes_observed"] == ["actor_only"], "credit schema observation")
    require(ocd["n_rows"] == s["total_transitions"], "credit rows != total transitions")

    # launcher exit-code defect and label anomalies
    exit_bytes = hashes["native_exit_code.txt"]["bytes"]
    anomalies.append({
        "id": "launcher_exit_code_file_empty",
        "observed": "native_exit_code.txt is %d bytes" % exit_bytes,
        "cause": "cmd parses `echo %RC%> file` with a one-digit code as a handle redirect",
        "handling": "recorded as-is; not repaired retrospectively; completion is established "
                    "from run_summary / train_records / final checkpoint instead"})
    require(exit_bytes == 0, "native_exit_code.txt unexpectedly non-empty")
    if fb.get("metric") == "severe_minus_mild_aggregate_abort_mass":
        anomalies.append({
            "id": "legacy_metric_label_in_v2_behaviour",
            "observed": "v2_behaviour.metric = 'severe_minus_mild_aggregate_abort_mass'",
            "context": "the same block states abort_probability_definition = the one semantic "
                       "SELF_PRESERVATION_ABORT leaf and aggregate_mass_is_not_selected_action_"
                       "probability = false; the value is the semantic leaf, the label is legacy",
            "handling": "reporting-label defect only; values preserved unnormalized"})
    anomalies.append({
        "id": "manifest_path_differs_from_comparator",
        "observed": "run consumed %s; comparator recorded %s" % (bm["absolute_path"],
                                                                  r1_bm["absolute_path"]),
        "context": "authorized 2026-09-15 archival relocation; SHA-256 and manifest_id verified",
        "handling": "not a population change"})
    anomalies.append({
        "id": "expected_setup_failures",
        "observed": "%d training setup failures (%s)" % (
            len(fail_rows), sorted({r["error_type"] for r in fail_rows})),
        "context": "certified-FD eligibility refusals (no_fd_eligible_ego); accounted and "
                   "replaced under the quota contract; failed seeds identical to comparator R1: %s"
                   % ([r["seed"] for r in fail_rows] == r1_seeds),
        "handling": "expected attrition class, not an integrity abort"})

    return {
        "schema": "semantic_action_actor_only_dev_r1_review_precheck", "schema_version": 1,
        "is_scientific_verdict": False,
        "evidence_class": "DEVELOPMENT (not confirmatory)",
        "confirmatory_profile_used": False,
        "measured_code_sha": git["commit"],
        "repository_state_recorded_by_run": {"branch": git["branch"], "dirty": git["dirty"],
                                             "dirty_path_count": git["dirty_path_count"]},
        "action_representation": {"run_config_training": rep,
                                  "episode_outcomes_observed": oas["wake_action_representations_observed"],
                                  "credit_rows_observed": ocd["action_representation_ids_observed"]},
        "authorized_plan": {"sha256": hashes["authorized_plan.json"]["sha256"],
                            "bytes": hashes["authorized_plan.json"]["bytes"],
                            "written_at": plan["written_at"],
                            "preflight_sha256": hashes["preflight.json"]["sha256"],
                            "preflight_timestamp": pre["preflight_timestamp"],
                            "run_config_collected_at": prov["collected_at"],
                            "plan_written_before_run_config": plan["written_at"][:19] <= prov["collected_at"]},
        "resolved_config_vs_authorized_plan": plan_vs_actual,
        "ppo_resolved": tc["ppo"],
        "invocation_argv_matches_plan": True,
        "config_source": rc["config_source"],
        "train_config_keys_differing_from_comparator_r1": r1_diff,
        "manifest": {"manifest_id": bm["manifest_id"], "file_sha256": man_sha,
                     "path_consumed": bm["absolute_path"],
                     "profile": bm["evaluation_profile"]["profile"],
                     "development_group_keys_sha256": bm["evaluation_profile"]["group_keys_sha256"],
                     "development_seed_list_sha256": bm["evaluation_profile"]["seed_list_sha256"],
                     "same_development_population_as_comparator": True,
                     "held_out_verified": True, "held_out_overlap_count": 0},
        "completion": {"updates_completed": s["updates_completed"],
                       "n_iterations_planned": N_ITERATIONS,
                       "train_records_rows": len(train_records),
                       "final_checkpoint_sha256": hashes["checkpoints/ckpt_iter0374.pt"]["sha256"],
                       "early_stopping": {"enabled": es["enabled"],
                                          "termination_reason": es["termination_reason"]}},
        "training_counts": {"attempted": s["train_episodes_attempted"],
                            "successful": s["train_episodes_successful"],
                            "failed": s["train_episodes_failed"],
                            "replacement_attempts": s["train_replacement_attempts"],
                            "iterations_at_full_quota": s["train_iterations_at_full_quota"],
                            "max_possible_attempts": ap["max_possible_training_attempts"]},
        "evaluation_counts": {"rounds": s["n_eval_rounds"],
                              "attempted": s["eval_episodes_attempted"],
                              "successful": s["eval_episodes_successful"],
                              "failed": s["eval_episodes_failed"]},
        "accounting_reconciled": s["accounting_reconciled"],
        "failures": {"by_phase": s["failures_by_phase"],
                     "by_pipeline_stage": s["failures_by_pipeline_stage"],
                     "by_error_type": s["failures_by_error_type"],
                     "rows": fail_rows,
                     "comparator_r1_failed_seeds": r1_seeds,
                     "failed_seeds_identical_to_comparator_r1": [r["seed"] for r in fail_rows] == r1_seeds},
        "final_round_identity": fin["identity"],
        "primary_endpoint_coverage": {
            "n_groups_attempted": fb["n_groups_attempted"],
            "n_groups_complete": fb["n_groups_complete"],
            "n_groups_metric_eligible": fb["n_groups_metric_eligible"],
            "macro_n_base_cells_required": fb["macro_n_base_cells_required"],
            "macro_n_base_cells_defined": fb["macro_n_base_cells_defined"],
            "macro_undefined_base_cells": fb["macro_undefined_base_cells"]},
        "schema_versions_observed": {
            "episode_outcome": oas["episode_outcome_schema_versions_observed"],
            "wake_diagnostics": oas["wake_diagnostics_schema_versions_observed"],
            "credit_diagnostics": ocd["schema_versions_observed"]},
        "comparator_r1": {"measured_code_sha": r1_rc["provenance"]["git"]["commit"],
                          "archive_key": R1_ARCHIVE_KEY,
                          "archive_original_path": archive_entry["original_path"],
                          "archive_current_path": archive_entry["current_path"],
                          "archive_evidence_ref_status_text": archive_entry["evidence_ref_status"],
                          "run_config_sha256": next(e["sha256"] for e in entries
                                                    if e["group"] == "comparator_r1"
                                                    and e["name"] == "run_config.json")},
        "launcher_exit_file_defect": anomalies[0],
        "anomalies": anomalies,
    }


# ----------------------------------------------------------------------------- behaviour
def wake_p_abort(wake, legacy: bool):
    if legacy:
        v = (wake.get("aggregate_probability_per_meta_action") or {}).get(ABORT)
    else:
        v = (wake.get("semantic_probability_per_meta_action") or {}).get(ABORT)
        leaves = [l for l in wake.get("semantic_leaves") or () if l.get("meta_action_name") == ABORT]
        require(len(leaves) == 1, "semantic wake without exactly one ABORT leaf")
        require(close(leaves[0]["probability"], v, 1e-12),
                "ABORT leaf probability %r != semantic_probability_per_meta_action %r",
                leaves[0]["probability"], v)
    require(isinstance(v, (int, float)) and not isinstance(v, bool), "P(ABORT) missing")
    return float(v)


WAKE_FIELDS = ("tick", "ego_id", "selected_meta_action", "selected_meta_action_name",
               "selected_node", "selected_leaf", "selected_action_probability",
               "deterministic_argmax_meta_action_name", "semantic_probability_per_meta_action",
               "semantic_entropy_raw", "semantic_entropy_normalized", "n_task_nodes",
               "n_agent_nodes", "n_semantic_leaves", "n_valid_semantic_leaves",
               "n_abort_legal_nodes", "n_engage_legal_leaves", "ego_fuel_norm",
               "reachable_by_ego", "task_distance_norm", "n_task_distance_clipped",
               "fraction_task_distance_clipped", "time_norm", "top_two_probability_margin")


def extract_eval_wakes(outcomes_path: Path, legacy: bool):
    """(rows for mild/severe members, per-round member counts, per-round complete-group sets)."""
    rows = []
    members = defaultdict(lambda: defaultdict(set))  # round -> group -> member cells
    round_updates = {}
    n_eval = 0
    for r in iter_jsonl(outcomes_path):
        if r.get("phase") not in EVAL_PHASES:
            continue
        n_eval += 1
        ro = int(r["eval_round_ordinal"])
        upd = int(r["updates_completed"])
        require(round_updates.setdefault(ro, upd) == upd, "round %d mixes update counts", ro)
        bv = r.get("benchmark_v2") or {}
        gk = r["benchmark_group_key"]
        cell = bv.get("member_cell")
        require(cell in ("clean",) + SEVERITIES, "unknown member cell %r", cell)
        require(cell not in members[ro][gk], "duplicate member %s/%s round %d", gk, cell, ro)
        members[ro][gk].add(cell)
        if not legacy:
            require(r.get("action_representation_id") == REPRESENTATION,
                    "outcome representation %r", r.get("action_representation_id"))
            require(r.get("schema_version") == 4 and r.get("wake_diagnostics_schema_version") == 2,
                    "outcome schema versions")
        if cell == "clean":
            continue
        require(r.get("severity") == cell, "severity %r != member cell %r", r.get("severity"), cell)
        fd = [w for w in r.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        require(len(fd) == 1, "round %d %s %s has %d immediate-FD wakes", ro, gk, cell, len(fd))
        w = fd[0]
        require(w.get("ego_id") == r.get("fd_ego_id"), "FD wake ego != fd_ego_id (%s)", gk)
        if legacy:
            require("action_representation_id" not in w, "legacy wake carries a representation id")
        else:
            require(w.get("action_representation_id") == REPRESENTATION,
                    "wake representation %r", w.get("action_representation_id"))
        row = {
            "evaluation_stage": r["phase"], "eval_round_ordinal": ro, "updates_completed": upd,
            "benchmark_group_key": gk, "base_cell": bv.get("base_cell"),
            "benchmark_world_ordinal": r.get("benchmark_world_ordinal"),
            "benchmark_profile": bv.get("profile"), "member_cell": cell,
            "severity": r.get("severity"), "condition": r.get("condition"),
            "seed": r.get("seed"), "episode_tag": r.get("episode_tag"),
            "eval_episode_index": r.get("eval_episode_index"),
            "eval_group_member": r.get("eval_group_member"),
            "fd_ego_id": r.get("fd_ego_id"), "fd_event_tick": r.get("fd_event_tick"),
            "fd_fuel_after_fraction_of_max": r.get("fd_fuel_after_fraction_of_max"),
            "episode_reward": r.get("reward"), "episode_ended": r.get("ended"),
            "wake_kind": w["wake_kind"],
            "action_representation_id": LEGACY_REPRESENTATION if legacy else w["action_representation_id"],
            "p_abort": wake_p_abort(w, legacy),
            "p_abort_source": ("aggregate_probability_per_meta_action (legacy alias mass)"
                               if legacy else "semantic SELF_PRESERVATION_ABORT leaf"),
        }
        for k in WAKE_FIELDS:
            if k in w:
                row["wake_" + k] = w[k]
        if legacy:
            row["wake_selected_meta_action_name"] = w.get("selected_meta_action_name")
            row["wake_selected_node"] = w.get("selected_node")
        rows.append(row)
    rows.sort(key=lambda x: (x["eval_round_ordinal"], x["benchmark_group_key"], x["member_cell"]))
    return rows, members, round_updates, n_eval


def behaviour_by_round(rows, members, round_updates, eval_records, label):
    require(sorted(round_updates) == list(range(N_ROUNDS)), "%s: rounds %s", label, sorted(round_updates))
    by_round = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_round[r["eval_round_ordinal"]][r["benchmark_group_key"]][r["member_cell"]] = r
    ev_by_round = {int(e["eval_round_ordinal"]): e for e in eval_records}
    require(sorted(ev_by_round) == list(range(N_ROUNDS)), "%s: eval_records rounds", label)
    out = []
    for ro in range(N_ROUNDS):
        ev = ev_by_round[ro]
        require(int(ev["updates_completed"]) == round_updates[ro], "%s round %d updates", label, ro)
        vb = ev["v2_behaviour"]
        n_members = sum(len(c) for c in members[ro].values())
        require(n_members == N_EVAL_MEMBERS_PER_ROUND, "%s round %d has %d members", label, ro, n_members)
        groups, cell_deltas = [], defaultdict(list)
        sw = rev = 0
        for gk in sorted(members[ro]):
            m = by_round[ro][gk]
            complete = members[ro][gk] == {"clean", "mild", "severe"}
            eligible = complete and "mild" in m and "severe" in m
            g = {"group_key": gk, "complete": complete, "metric_eligible": eligible}
            if eligible:
                d = m["severe"]["p_abort"] - m["mild"]["p_abort"]
                ms, ss = m["mild"]["wake_selected_meta_action_name"], m["severe"]["wake_selected_meta_action_name"]
                g.update({"base_cell": m["mild"]["base_cell"], "p_abort_mild": m["mild"]["p_abort"],
                          "p_abort_severe": m["severe"]["p_abort"], "severe_minus_mild": d,
                          "mild_selected": ms, "severe_selected": ss,
                          "directional_switch": ms != ABORT and ss == ABORT,
                          "reverse_switch": ms == ABORT and ss != ABORT})
                sw += g["directional_switch"]
                rev += g["reverse_switch"]
                cell_deltas[g["base_cell"]].append(d)
            groups.append(g)
        cells = {c: {"n_metric_eligible_groups": len(v), "severe_minus_mild_mean": mean(v)}
                 for c, v in sorted(cell_deltas.items())}
        defined = [c for c in cells if cells[c]["n_metric_eligible_groups"] > 0]
        macro = mean(cells[c]["severe_minus_mild_mean"] for c in defined) if len(defined) == N_BASE_CELLS else None
        elig = [g for g in groups if g["metric_eligible"]]
        pooled = mean(g["severe_minus_mild"] for g in elig)
        mild_p = [by_round[ro][gk]["mild"]["p_abort"] for gk in sorted(by_round[ro]) if "mild" in by_round[ro][gk]]
        sev_p = [by_round[ro][gk]["severe"]["p_abort"] for gk in sorted(by_round[ro]) if "severe" in by_round[ro][gk]]

        # --- cross-check against the trainer's own per-round record ---
        require(vb["n_groups_metric_eligible"] == len(elig), "%s r%d eligible %s vs %d", label, ro, vb["n_groups_metric_eligible"], len(elig))
        require(vb["directional_switch_count"] == sw and vb["reverse_switch_count"] == rev,
                "%s r%d switch counts differ from trainer", label, ro)
        require(close(vb["pooled_mean_over_groups"], pooled), "%s r%d pooled %r vs %r", label, ro, vb["pooled_mean_over_groups"], pooled)
        require((vb["macro_mean_over_base_cells"] is None and macro is None)
                or close(vb["macro_mean_over_base_cells"], macro), "%s r%d macro %r vs %r", label, ro, vb["macro_mean_over_base_cells"], macro)
        for c, cv in cells.items():
            require(close(vb["by_base_cell"][c]["severe_minus_mild_abort_mass_mean"], cv["severe_minus_mild_mean"]),
                    "%s r%d cell %s differs", label, ro, c)
        tg = {g["group_key"]: g for g in vb["groups"]}
        for g in elig:
            t = tg[g["group_key"]]
            require(close(t["p_abort_mild"], g["p_abort_mild"], 1e-12)
                    and close(t["p_abort_severe"], g["p_abort_severe"], 1e-12)
                    and t["mild_selected_meta_action"] == g["mild_selected"]
                    and t["severe_selected_meta_action"] == g["severe_selected"],
                    "%s r%d group %s differs from trainer", label, ro, g["group_key"])
        if label == "new":
            require(vb.get("action_representation_ids_observed") == [REPRESENTATION],
                    "round %d representations %s", ro, vb.get("action_representation_ids_observed"))
            require(vb.get("aggregate_mass_is_not_selected_action_probability") is False,
                    "round %d aggregate-mass flag", ro)
        out.append({
            "eval_round_ordinal": ro, "evaluation_stage": ev["evaluation_stage"],
            "updates_completed": round_updates[ro],
            "n_members": n_members, "n_groups": len(groups), "n_groups_complete": sum(g["complete"] for g in groups),
            "n_groups_metric_eligible": len(elig),
            "p_abort_mild_mean": mean(mild_p), "n_mild_members": len(mild_p),
            "p_abort_severe_mean": mean(sev_p), "n_severe_members": len(sev_p),
            "severe_minus_mild_pooled_mean": pooled,
            "severe_minus_mild_macro_over_base_cells": macro,
            "macro_n_base_cells_defined": len(defined),
            "directional_switches": rate(sw, len(elig)),
            "reverse_switches": rate(rev, len(elig)),
            "by_base_cell": cells,
            "groups": groups,
            "trainer_cross_check": "passed (macro, pooled, per-cell, per-group, switch counts)",
        })
    return out


# ----------------------------------------------------------------------------- credit
def credit_extraction(credit_path: Path, train_records, rc, summary, outcomes_path: Path):
    rows = read_jsonl(credit_path)
    tr_by_iter = {int(t["iteration"]): t for t in train_records}
    require(sorted(tr_by_iter) == list(range(N_ITERATIONS)), "train_records iterations")
    gamma_cfg = rc["train_config"]["ppo"]["gamma"]
    eps_cfg = rc["train_config"]["ppo"]["adv_norm_eps"]
    require(gamma_cfg == 1.0, "configured gamma %r", gamma_cfg)
    require(rc["train_config"]["training_mode"] == "actor_only", "training mode")

    by_iter = defaultdict(list)
    for i, r in enumerate(rows):
        require(r.get("schema") == "graph_train_credit_diagnostics" and r.get("schema_version") == 1,
                "credit row %d schema", i)
        require(r.get("action_representation_id") == REPRESENTATION, "credit row %d representation", i)
        require(r.get("training_mode") == "actor_only", "credit row %d mode", i)
        require(r.get("gamma") == 1.0 and r.get("adv_norm_eps") == eps_cfg, "credit row %d gamma/eps", i)
        require((r.get("measurement_join") or {}).get("joined") is True, "credit row %d not joined", i)
        by_iter[int(r["iteration"])].append(r)
    require(sorted(by_iter) == list(range(N_ITERATIONS)), "credit rows do not cover iterations 0..374")
    require(len(rows) == summary["total_transitions"] == summary["observed_credit_diagnostics"]["n_rows"],
            "credit rows %d vs summary", len(rows))

    coverage = []
    baseline_checked = baseline_skipped = 0
    for it in range(N_ITERATIONS):
        rs = by_iter[it]
        t = tr_by_iter[it]
        n_tr = int(t["n_transitions"]) if "n_transitions" in t else None
        require(n_tr == len(rs), "iteration %d: %d credit rows vs n_transitions %r", it, len(rs), n_tr)
        require(int(t["updates_completed_before"]) == it and all(int(r["updates_completed_before"]) == it for r in rs),
                "iteration %d updates_completed_before", it)
        require(sorted(int(r["batch_transition_ordinal"]) for r in rs) == list(range(len(rs))),
                "iteration %d batch ordinals", it)
        for k in ("batch_raw_advantage_mean", "batch_raw_advantage_std", "batch_n_transitions",
                  "batch_n_episodes", "batch_n_episodes_with_wakes", "actor_only_episode_baseline"):
            require(len({r[k] for r in rs}) == 1, "iteration %d field %s not batch-constant", it, k)
        r0 = rs[0]
        require(r0["batch_n_transitions"] == len(rs), "iteration %d batch_n_transitions", it)
        require(close(r0["actor_only_episode_baseline"], t["baseline"], 1e-12), "iteration %d baseline vs train_records", it)
        raw = [float(r["raw_advantage"]) for r in rs]
        m = math.fsum(raw) / len(raw)
        sd = math.sqrt(math.fsum((x - m) ** 2 for x in raw) / len(raw))
        require(close(m, r0["batch_raw_advantage_mean"]) and close(sd, r0["batch_raw_advantage_std"]),
                "iteration %d batch moments do not recompute", it)
        for r in rs:
            require(close(r["return"], r["episode_reward"], 1e-12), "return != episode reward (gamma=1)")
            require(close(r["raw_advantage"], r["return"] - r["actor_only_episode_baseline"], 1e-12),
                    "raw advantage != return - baseline")
            require(close(r["normalized_advantage"],
                          (r["raw_advantage"] - r["batch_raw_advantage_mean"])
                          / (r["batch_raw_advantage_std"] + r["adv_norm_eps"]), 1e-7),
                    "normalized advantage does not recompute")
        eps_rewards = {}
        for r in rs:
            eps_rewards.setdefault((r["episode_seed"], r["episode_index"]), r["episode_reward"])
        if len(eps_rewards) == r0["batch_n_episodes"]:
            require(close(mean(eps_rewards.values()), r0["actor_only_episode_baseline"], 1e-12),
                    "iteration %d episode-mean baseline does not recompute", it)
            baseline_checked += 1
        else:
            baseline_skipped += 1
        coverage.append(len(rs))

    # structural constancy
    chains, episodes = defaultdict(set), defaultdict(set)
    chain_len, episode_len = defaultdict(int), defaultdict(int)
    for r in rows:
        ck = (r["iteration"], r["episode_seed"], r["episode_index"], r["ego_id"])
        ek = (r["iteration"], r["episode_seed"], r["episode_index"])
        chains[ck].add(r["raw_advantage"])
        episodes[ek].add(r["raw_advantage"])
        chain_len[ck] += 1
        episode_len[ek] += 1
    # terminal-only reward: per episode, stored transition rewards sum to R and all of R sits on
    # the single transition at the episode's latest tick (the last one of its ego chain)
    ep_rows = defaultdict(list)
    for r in rows:
        ep_rows[(r["iteration"], r["episode_seed"], r["episode_index"])].append(r)
    n_r_nonzero = 0
    for ek, rs in ep_rows.items():
        R = rs[0]["episode_reward"]
        require(all(x["episode_reward"] == R for x in rs), "episode %s mixes episode_reward", ek)
        require(all(x.get("transition_reward") is not None for x in rs), "episode %s null transition_reward", ek)
        require(close(math.fsum(x["transition_reward"] for x in rs), R, 1e-12),
                "episode %s transition rewards do not sum to R", ek)
        nz = [x for x in rs if x["transition_reward"] != 0.0]
        if R != 0.0:
            n_r_nonzero += 1
            require(len(nz) == 1, "episode %s has %d nonzero transition rewards", ek, len(nz))
            z = nz[0]
            require(z["tick"] == max(x["tick"] for x in rs), "episode %s reward not at latest tick", ek)
            require(z["ego_chain_ordinal"] == max(x["ego_chain_ordinal"] for x in rs
                                                  if x["ego_id"] == z["ego_id"]),
                    "episode %s reward not on the last transition of its ego chain", ek)
        else:
            require(not nz, "episode %s has R = 0 but a nonzero transition reward", ek)
    tr_nonzero = [r for r in rows if r["transition_reward"] != 0.0]
    structural = {
        "configured_gamma": gamma_cfg, "rows_gamma_values": sorted({r["gamma"] for r in rows}),
        "n_chains_checked": len(chains),
        "n_chains_with_varying_raw_advantage": sum(1 for v in chains.values() if len(v) > 1),
        "n_chains_with_more_than_one_transition": sum(1 for v in chain_len.values() if v > 1),
        "n_episodes_checked": len(episodes),
        "n_episodes_with_varying_raw_advantage": sum(1 for v in episodes.values() if len(v) > 1),
        "n_episodes_with_more_than_one_transition": sum(1 for v in episode_len.values() if v > 1),
        "equality_test": "exact float equality of persisted raw_advantage values",
        "per_row_identities_verified": [
            "return == episode_reward (gamma = 1.0)",
            "raw_advantage == return - actor_only_episode_baseline",
            "normalized_advantage == (raw - batch_raw_mean) / (batch_raw_std + adv_norm_eps)",
            "batch_raw_advantage_mean/std recompute from the batch's persisted rows",
            "actor_only_episode_baseline == train_records.baseline for the iteration"],
        "episode_mean_baseline_recomputed_batches": baseline_checked,
        "episode_mean_baseline_not_recomputable_batches": baseline_skipped,
        "episode_mean_baseline_note": ("a batch whose zero-wake episodes contribute no credit row "
                                       "cannot have its episode-mean baseline recomputed from rows alone"),
        "terminal_only_reward_verification": {
            "n_episodes_checked": len(ep_rows),
            "n_episodes_transition_rewards_sum_to_episode_reward": len(ep_rows),
            "n_episodes_with_nonzero_episode_reward": n_r_nonzero,
            "n_rows_with_nonzero_transition_reward": len(tr_nonzero),
            "placement": "every episode with R != 0 carries all of R on exactly one transition, "
                         "at the episode's latest tick and last in its ego chain; every other "
                         "stored transition reward is 0.0",
            "note": "transition_reward is the stored Transition.reward; the actor-only return "
                    "does not read it -- it assigns the episode's terminal R to every "
                    "transition of every chain (verified: return == episode_reward per row)"},
    }

    # FD-selected-ego immediate-FD rows
    fd_rows = [r for r in rows if r["wake_kind"] == FD_WAKE
               and r["measurement_join"].get("is_fd_selected_ego") is True]
    fd_other = [r for r in rows if r["wake_kind"] == FD_WAKE
                and r["measurement_join"].get("is_fd_selected_ego") is not True]
    require(not fd_other, "%d immediate-FD credit rows are not the FD-selected ego", len(fd_other))
    keys = [(r["iteration"], r["episode_seed"], r["episode_index"]) for r in fd_rows]
    require(len(keys) == len(set(keys)), "an episode has more than one FD-selected immediate-FD row")
    require(all(r["measurement_join"]["severity"] in SEVERITIES for r in fd_rows), "FD row severity")
    require(all(r["measurement_join"]["fd_selected_ego_id"] == r["ego_id"] for r in fd_rows), "FD ego id join")
    fdt = summary["fuel_damage_totals"]
    train_events = {"events_applied": fdt["train_fuel_damage_events_applied"],
                    "wakes": fdt["train_fuel_damage_wakes"],
                    "mild_successful": fdt["train_mild_successful"],
                    "severe_successful": fdt["train_severe_successful"]}
    n_sev = {s: sum(1 for r in fd_rows if r["measurement_join"]["severity"] == s) for s in SEVERITIES}
    require(len(fd_rows) == train_events["events_applied"] == train_events["wakes"],
            "FD credit rows %d vs summary train FD events/wakes %r", len(fd_rows), train_events)
    require(n_sev["mild"] == train_events["mild_successful"]
            and n_sev["severe"] == train_events["severe_successful"],
            "FD credit rows by severity %r vs summary %r", n_sev, train_events)

    # join to training outcomes: selected action and the unbiased P(ABORT) at the same wake
    train_fd = {}
    for o in iter_jsonl(outcomes_path):
        if o.get("phase") != "train":
            continue
        fd = [w for w in o.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        if not fd:
            continue
        require(len(fd) == 1, "training outcome with %d immediate-FD wakes", len(fd))
        key = (int(o["iteration"]), int(o["seed"]))
        require(key not in train_fd, "duplicate training outcome %s", key)
        train_fd[key] = (o, fd[0])
    require(len(train_fd) == len(fd_rows), "training FD outcomes %d vs FD credit rows %d", len(train_fd), len(fd_rows))
    joined = []
    for r in fd_rows:
        o, w = train_fd[(int(r["iteration"]), int(r["episode_seed"]))]
        require(o.get("severity") == r["measurement_join"]["severity"], "severity join")
        require(w.get("ego_id") == r["ego_id"] and w.get("tick") == r["tick"], "wake identity join")
        require(w.get("selected_meta_action_name") == r["selected_meta_action_name"]
                and w.get("selected_node") == r["selected_node"], "selected action join")
        require(close(o.get("reward"), r["episode_reward"], 1e-12), "episode reward join")
        joined.append((r, wake_p_abort(w, legacy=False)))
    return rows, fd_rows, joined, coverage, structural, train_events


def credit_summary(fd_joined, coverage, structural, train_events):
    def block(pairs):
        rs = [r for r, _ in pairs]
        return {"n": len(rs),
                "raw_advantage": desc(r["raw_advantage"] for r in rs),
                "normalized_advantage": desc(r["normalized_advantage"] for r in rs),
                "episode_reward": desc(r["episode_reward"] for r in rs),
                "abort_selected": rate(sum(r["selected_meta_action_name"] == ABORT for r in rs), len(rs)),
                "p_abort_at_wake_mean": mean(p for _, p in pairs)}

    def abort_gap(pairs):
        a = [r for r, _ in pairs if r["selected_meta_action_name"] == ABORT]
        b = [r for r, _ in pairs if r["selected_meta_action_name"] != ABORT]
        ga = lambda key: (mean(x[key] for x in a) - mean(x[key] for x in b)) if a and b else None
        return {"n_abort": len(a), "n_not_abort": len(b),
                "abort": {"raw_advantage_mean": mean(x["raw_advantage"] for x in a),
                          "normalized_advantage_mean": mean(x["normalized_advantage"] for x in a),
                          "episode_reward_mean": mean(x["episode_reward"] for x in a)},
                "not_abort": {"raw_advantage_mean": mean(x["raw_advantage"] for x in b),
                              "normalized_advantage_mean": mean(x["normalized_advantage"] for x in b),
                              "episode_reward_mean": mean(x["episode_reward"] for x in b)},
                "abort_minus_not_abort_raw_advantage": ga("raw_advantage"),
                "abort_minus_not_abort_normalized_advantage": ga("normalized_advantage")}

    sev = {s: [(r, p) for r, p in fd_joined if r["measurement_join"]["severity"] == s] for s in SEVERITIES}
    per_batch = defaultdict(lambda: defaultdict(list))
    for r, p in fd_joined:
        per_batch[r["iteration"]][r["measurement_join"]["severity"]].append(r)
    within = []
    for it in sorted(per_batch):
        d = per_batch[it]
        if d["mild"] and d["severe"]:
            within.append({
                "iteration": it, "n_mild": len(d["mild"]), "n_severe": len(d["severe"]),
                "severe_minus_mild_raw": mean(x["raw_advantage"] for x in d["severe"]) - mean(x["raw_advantage"] for x in d["mild"]),
                "severe_minus_mild_normalized": mean(x["normalized_advantage"] for x in d["severe"]) - mean(x["normalized_advantage"] for x in d["mild"])})
    windows = []
    for w0 in range(0, N_ITERATIONS, WINDOW):
        entry = {"iterations": [w0, w0 + WINDOW - 1]}
        for s in SEVERITIES:
            pairs = [(r, p) for r, p in sev[s] if w0 <= r["iteration"] < w0 + WINDOW]
            entry[s] = {"descriptive": block(pairs), "abort_vs_not_non_counterfactual": abort_gap(pairs)}
        windows.append(entry)
    return {
        "schema": "semantic_action_actor_only_dev_r1_credit_summary", "schema_version": 1,
        "is_scientific_verdict": False,
        "population": "training credit rows with wake_kind = immediate_fuel_damage AND "
                      "measurement_join.is_fd_selected_ego = true",
        "counts": {"total": len(fd_joined), "mild": len(sev["mild"]), "severe": len(sev["severe"]),
                   "summary_train_fd_events": train_events},
        "coverage": {"n_productive_updates": len(coverage), "rows_per_update_min": min(coverage),
                     "rows_per_update_max": max(coverage), "rows_total": sum(coverage),
                     "every_update_rows_equal_train_records_n_transitions": True},
        "analysis_classes": {
            "descriptive_association": "per-severity summaries over all training FD rows; different "
                                       "episodes, worlds and policy states are pooled",
            "matched_within_batch": "severe minus mild means within the SAME update batch (same "
                                    "baseline and normalization); still different episodes/worlds",
            "abort_vs_not_non_counterfactual": "rows that selected ABORT versus rows that did not, "
                                              "within a severity; different episodes and worlds, "
                                              "NOT a counterfactual action-value contrast"},
        "descriptive_association": {s: block(sev[s]) for s in SEVERITIES},
        "matched_within_batch": {
            "n_batches_with_both_severities": len(within),
            "severe_minus_mild_raw_advantage": desc(x["severe_minus_mild_raw"] for x in within),
            "severe_minus_mild_normalized_advantage": desc(x["severe_minus_mild_normalized"] for x in within),
            "batches_severe_below_mild_raw": rate(sum(x["severe_minus_mild_raw"] < 0 for x in within), len(within)),
            "per_batch": within},
        "abort_vs_not_non_counterfactual": {s: abort_gap(sev[s]) for s in SEVERITIES},
        "windows_25_iterations": windows,
        "structural_credit_verification": structural,
        "p_abort_at_wake_source": "semantic_probability_per_meta_action of the same wake in the "
                                  "training episode outcome (joined on iteration + seed; selected "
                                  "action, ego, tick and reward verified equal)",
    }


# ----------------------------------------------------------------------------- main
def run(args):
    run_dir, r1_dir, out = Path(args.run_dir), Path(args.r1_dir), Path(args.out)
    archive_index = Path(args.archive_index)
    rc = read_json(run_dir / "run_config.json")
    manifest_path = Path(rc["episode_design"]["benchmark_manifest"]["absolute_path"])

    entries = hash_sources(run_dir, r1_dir, manifest_path, archive_index)
    if args.verify_against:
        expected = parse_sha_file(Path(args.verify_against))
        for e in entries:
            require(e["absolute_path"] in expected, "artifact %s absent from expected hashes", e["absolute_path"])
            require(expected[e["absolute_path"]] == (e["sha256"], e["bytes"]),
                    "HASH MISMATCH for %s: expected %s, found %s", e["absolute_path"],
                    expected[e["absolute_path"]], (e["sha256"], e["bytes"]))
        require(len(expected) == len(entries), "expected hash file lists %d artifacts, found %d", len(expected), len(entries))
    before = {e["absolute_path"]: e["sha256"] for e in entries}

    # comparator identity from the archive index (independent of the run's own claims)
    archive_entry = None
    for rec in iter_jsonl(archive_index):
        if rec.get("artifact_id") == R1_ARCHIVE_KEY:
            archive_entry = rec
    require(archive_entry is not None, "comparator absent from archive index")
    require(archive_entry["measured_code_sha"] == COMPARATOR_SHA, "archive comparator SHA")
    for e in entries:
        if e["group"] == "comparator_r1" and e["name"] in archive_entry["key_sha256"]:
            require(archive_entry["key_sha256"][e["name"]] == e["sha256"],
                    "comparator %s hash differs from archive index", e["name"])

    plan = read_json(run_dir / "authorized_plan.json")
    pre = read_json(run_dir / "preflight.json")
    summary = read_json(run_dir / "run_summary.json")
    train_records = read_jsonl(run_dir / "train_records.jsonl")
    eval_records = read_jsonl(run_dir / "eval_records.jsonl")
    failures = read_jsonl(run_dir / "episode_failures.jsonl")
    r1_rc = read_json(r1_dir / "run_config.json")
    require(r1_rc["provenance"]["git"]["commit"] == COMPARATOR_SHA, "comparator run_config SHA")
    r1_summary = read_json(r1_dir / "run_summary.json")
    r1_eval = read_jsonl(r1_dir / "eval_records.jsonl")
    r1_failures = read_jsonl(r1_dir / "episode_failures.jsonl")

    pc = precheck(run_dir, entries, rc, plan, pre, summary, train_records, eval_records, failures,
                  r1_rc, r1_failures, archive_entry, manifest_path)

    # behaviour (new) and comparator trajectory (R1)
    wakes, members, rupd, n_eval = extract_eval_wakes(run_dir / "episode_outcomes.jsonl", legacy=False)
    require(n_eval == summary["eval_episodes_attempted"], "eval outcome rows %d", n_eval)
    require(len(wakes) == N_ROUNDS * 40, "eval immediate-FD wake rows %d != 640", len(wakes))
    rounds = behaviour_by_round(wakes, members, rupd, eval_records, "new")
    fb = summary["generalized"]["v2_benchmark"]["final_round_behaviour"]
    final = rounds[-1]
    require(final["updates_completed"] == summary["final_eval_selection"]["identity"]["updates_completed"]
            and final["eval_round_ordinal"] == summary["final_eval_selection"]["identity"]["eval_round_ordinal"],
            "final round identity")
    require(final["macro_n_base_cells_defined"] == N_BASE_CELLS, "final macro not over 10/10 cells")
    require(close(final["severe_minus_mild_macro_over_base_cells"], fb["macro_mean_over_base_cells"]),
            "final macro differs from run_summary")

    r1_wakes, r1_members, r1_rupd, r1_n = extract_eval_wakes(r1_dir / "episode_outcomes.jsonl", legacy=True)
    r1_rounds = behaviour_by_round(r1_wakes, r1_members, r1_rupd, r1_eval, "r1")
    r1_fb = r1_summary["generalized"]["v2_benchmark"]["final_round_behaviour"]
    require(close(r1_rounds[-1]["severe_minus_mild_macro_over_base_cells"], r1_fb["macro_mean_over_base_cells"]),
            "comparator final macro differs from its run_summary")

    compact = lambda rs: [{k: v for k, v in r.items() if k != "groups"} for r in rs]
    behaviour = {
        "schema": "semantic_action_actor_only_dev_r1_behaviour_summary", "schema_version": 1,
        "is_scientific_verdict": False,
        "endpoint": "SEVERE - MILD P(SELF_PRESERVATION_ABORT) at the certified ego's "
                    "immediate_fuel_damage wake; pair within frozen world group, mean per base "
                    "cell, equal-weight macro over ten base cells",
        "p_abort_definition": "the one semantic SELF_PRESERVATION_ABORT leaf (%s)" % REPRESENTATION,
        "repeated_measures_note": ("every evaluation round re-measures the SAME 20 frozen development "
                                   "worlds (60 members); rounds are repeated measures of one "
                                   "population, not independent samples, and cross-round totals "
                                   "describe a trajectory only"),
        "verifications": ["representation uniformly %s on every outcome and wake" % REPRESENTATION,
                          "every MILD / SEVERE member has exactly one immediate-FD wake",
                          "no mixed representations in any round",
                          "per-round macro, pooled mean, per-cell means, per-group P(ABORT) and "
                          "selected actions equal the trainer's persisted v2_behaviour",
                          "final macro defined over 10/10 base cells and equal to run_summary"],
        "n_rounds": len(rounds),
        "round_identities": [{"eval_round_ordinal": r["eval_round_ordinal"],
                              "evaluation_stage": r["evaluation_stage"],
                              "updates_completed": r["updates_completed"]} for r in rounds],
        "rounds": rounds,
        "final_round": {"identity": summary["final_eval_selection"]["identity"],
                        "macro_over_base_cells": final["severe_minus_mild_macro_over_base_cells"],
                        "macro_n_base_cells_defined": final["macro_n_base_cells_defined"],
                        "pooled_mean_over_groups": final["severe_minus_mild_pooled_mean"],
                        "n_groups_metric_eligible": final["n_groups_metric_eligible"],
                        "directional_switches": final["directional_switches"],
                        "reverse_switches": final["reverse_switches"],
                        "by_base_cell": final["by_base_cell"]},
    }
    comparator = {
        "schema": "semantic_action_actor_only_dev_r1_comparator_r1_behaviour", "schema_version": 1,
        "is_scientific_verdict": False,
        "comparator": pc["comparator_r1"],
        "source": "recomputed from the archived comparator episode_outcomes.jsonl and cross-checked "
                  "against its eval_records.jsonl v2_behaviour and run_summary.json; no prose used",
        "p_abort_definition": "aggregate mass over the node-indexed abort cells (historical "
                              "representation; no representation id recorded)",
        "cross_version_note": "different measured code SHA and action representation; same frozen "
                              "development population and training seed stream",
        "rounds": compact(r1_rounds),
        "max_over_rounds": {
            "severe_minus_mild_pooled_mean": max(r["severe_minus_mild_pooled_mean"] for r in r1_rounds),
            "severe_minus_mild_macro_over_base_cells": max(r["severe_minus_mild_macro_over_base_cells"] for r in r1_rounds),
            "abs_severe_minus_mild_pooled_mean": max(abs(r["severe_minus_mild_pooled_mean"]) for r in r1_rounds),
            "directional_switch_count": max(r["directional_switches"]["count"] for r in r1_rounds),
            "reverse_switch_count": max(r["reverse_switches"]["count"] for r in r1_rounds)},
        "final_round_macro": r1_rounds[-1]["severe_minus_mild_macro_over_base_cells"],
    }

    credit_rows, fd_rows, fd_joined, coverage, structural, train_events = credit_extraction(
        run_dir / "train_credit_diagnostics.jsonl", train_records, rc, summary,
        run_dir / "episode_outcomes.jsonl")
    csum = credit_summary(fd_joined, coverage, structural, train_events)

    timeline = {
        "schema": "semantic_action_actor_only_dev_r1_collapse_timeline", "schema_version": 1,
        "is_scientific_verdict": False,
        "note": "ALL 16 evaluation rounds are included (no selection); training windows cover all "
                "375 iterations. Evaluation rounds are repeated measures of the same 20 worlds.",
        "evaluation_rounds": [{
            "eval_round_ordinal": r["eval_round_ordinal"], "updates_completed": r["updates_completed"],
            "evaluation_stage": r["evaluation_stage"],
            "p_abort_mild_mean": r["p_abort_mild_mean"], "p_abort_severe_mean": r["p_abort_severe_mean"],
            "severe_minus_mild_pooled_mean": r["severe_minus_mild_pooled_mean"],
            "severe_minus_mild_macro_over_base_cells": r["severe_minus_mild_macro_over_base_cells"],
            "n_groups_metric_eligible": r["n_groups_metric_eligible"],
            "directional_switches": r["directional_switches"]["count"],
            "reverse_switches": r["reverse_switches"]["count"],
            "comparator_r1_macro": q["severe_minus_mild_macro_over_base_cells"],
            "comparator_r1_pooled": q["severe_minus_mild_pooled_mean"],
            "comparator_r1_directional_switches": q["directional_switches"]["count"],
        } for r, q in zip(rounds, r1_rounds)],
        "highlighted_updates_requested": [75, 100, 125, 150, 175],
        "training_fd_wake_windows": [{
            "iterations": w["iterations"],
            "severe_n": w["severe"]["descriptive"]["n"],
            "severe_p_abort_at_wake_mean": w["severe"]["descriptive"]["p_abort_at_wake_mean"],
            "severe_abort_selected_rate": w["severe"]["descriptive"]["abort_selected"]["rate"],
            "mild_n": w["mild"]["descriptive"]["n"],
            "mild_p_abort_at_wake_mean": w["mild"]["descriptive"]["p_abort_at_wake_mean"],
            "mild_abort_selected_rate": w["mild"]["descriptive"]["abort_selected"]["rate"],
            "severe_abort_minus_not_raw_advantage_non_counterfactual":
                w["severe"]["abort_vs_not_non_counterfactual"]["abort_minus_not_abort_raw_advantage"],
            "mild_abort_minus_not_raw_advantage_non_counterfactual":
                w["mild"]["abort_vs_not_non_counterfactual"]["abort_minus_not_abort_raw_advantage"],
        } for w in csum["windows_25_iterations"]],
    }

    # every source still byte-identical
    after = {e["absolute_path"]: sha256_file(Path(e["absolute_path"])) for e in entries}
    require(after == before, "a source artifact changed during extraction")

    source_manifest = {
        "schema": "semantic_action_actor_only_dev_r1_source_manifest", "schema_version": 1,
        "measured_code_sha": MEASURED_SHA, "comparator_code_sha": COMPARATOR_SHA,
        "hash_algorithm": "sha256", "artifacts": entries,
        "sources_unchanged_during_extraction": True,
        "extraction_invoked_no_scientific_execution": True,
    }

    out.mkdir(parents=True, exist_ok=True)
    if args.copy_run_artifacts:
        for e in entries:
            if e["disposition"] == "copied":
                dst = out / e["git_path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(e["absolute_path"], dst)
                require(sha256_file(dst) == e["sha256"], "copy of %s not byte-identical", e["name"])
    write_text(out / "artifact_sha256.txt", render_sha_file(entries))
    write_text(out / "source_manifest.json", dumps(source_manifest))
    write_text(out / "review_precheck.json", dumps(pc))
    ex = out / "extracted"
    write_jsonl(ex / "eval_immediate_fd_wakes.jsonl", wakes)
    write_text(ex / "behaviour_summary.json", dumps(behaviour))
    write_text(ex / "comparator_r1_behaviour.json", dumps(comparator))
    write_text(ex / "collapse_timeline.json", dumps(timeline))
    write_jsonl(ex / "fd_selected_ego_credit_rows.jsonl", fd_rows)
    write_text(ex / "credit_summary.json", dumps(csum))

    # parse-back of every output
    for p in sorted(out.rglob("*")):
        if p.suffix == ".json":
            read_json(p)
        elif p.suffix == ".jsonl":
            read_jsonl(p)
    print("OK: %d source artifacts verified unchanged; %d eval FD wake rows; %d FD credit rows "
          "(mild %d / severe %d); final macro %.12g over %d/10 cells"
          % (len(entries), len(wakes), len(fd_rows), csum["counts"]["mild"], csum["counts"]["severe"],
             final["severe_minus_mild_macro_over_base_cells"], final["macro_n_base_cells_defined"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--r1-dir", default=str(DEFAULT_R1))
    ap.add_argument("--archive-index", default=str(DEFAULT_ARCHIVE_INDEX))
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-run-artifacts", action="store_true")
    ap.add_argument("--verify-against", default=None,
                    help="an artifact_sha256.txt whose hashes every source must match")
    args = ap.parse_args(argv)
    try:
        run(args)
    except EvidenceError as exc:
        print("EVIDENCE CHECK FAILED: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
