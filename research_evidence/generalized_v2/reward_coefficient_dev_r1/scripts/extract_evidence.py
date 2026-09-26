"""Deterministic, read-only evidence extraction for the REWARD-01 coefficient comparison R1.

Arms (both at measured code SHA ``2b570194dea3612f3796999fc589b73d7082ae31``):
  A (control)       aircraft_penalty_coeff = 2.25   C:\\gruns\\reward_c225_r1_s3000000_2b57019
  B (intervention)  aircraft_penalty_coeff = 4.5    C:\\gruns\\reward_c450_r1_s3000000_2b57019

STANDARD LIBRARY ONLY. It imports nothing from ``match_aou``, starts no subprocess and runs no
training, evaluation, replay, checkpoint load or benchmark preflight. Every source is hashed
before and after extraction and the run fails if a byte changed. Every reproduced evaluation
quantity is cross-checked against the trainer's own per-round ``v2_behaviour``; every
evaluation reward is re-derived from its own recorded terms before it is rescored.

Written and committed BEFORE either arm's results were inspected; the endpoints and the
rescoring rule are those of ``authorized_plan.json``.

Rescoring (algebra only, never a replay or a counterfactual outcome): for each saved evaluation
outcome, with its OWN recorded reference,
    D = |u_ref| + 1e-5,  q = reward_ratio = (u_achieved - u_ref) / D,  p = u_aircraft * n_dead / D,
    R(c) = q - c * p,    checked: reward_ratio == (u_achieved - u_ref)/D,
                                  reward_penalty == c_native * p,  reward == q - reward_penalty.

Usage::

    python extract_evidence.py --out <package dir> [--copy-run-artifacts]
                               [--verify-against <package dir>/artifact_sha256.txt]

Any schema, provenance, accounting or arithmetic mismatch exits 2 with EVIDENCE CHECK FAILED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pre_update  # noqa: E402

# ----------------------------------------------------------------------------- identities
MEASURED_SHA = "2b570194dea3612f3796999fc589b73d7082ae31"
SHORT = MEASURED_SHA[:7]
WORKTREE = r"C:\grc1src"
PKG = "research_evidence/generalized_v2/reward_coefficient_dev_r1"
ARMS = {
    "A": {"coeff": 2.25, "label": "control", "preset": "arm_a_c2p25.json",
          "run": Path(r"C:\gruns") / ("reward_c225_r1_s3000000_" + SHORT)},
    "B": {"coeff": 4.5, "label": "intervention", "preset": "arm_b_c4p5.json",
          "run": Path(r"C:\gruns") / ("reward_c450_r1_s3000000_" + SHORT)},
}
COMMON_C = (2.25, 4.5)
REPRESENTATION = "semantic_k_plus_2_logmeanexp_v1"
ACTOR_OBSERVATION_ID = "actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"
DEFAULT_MANIFEST = Path(r"C:\gra\benchmarks\v2_preflight_seed2000000_ae42cb0\benchmark_manifest.json")
REGRET_EPS = 1e-5

ABORT = "SELF_PRESERVATION_ABORT"
FD_WAKE = "immediate_fuel_damage"
WAKE_KINDS = ("ordinary", FD_WAKE, "post_fd_boundary")
SEVERITIES = ("mild", "severe")
CELLS = ("clean",) + SEVERITIES
EVAL_PHASES = ("pre_update", "post_update")
N_ROUNDS = 16
N_EVAL_MEMBERS_PER_ROUND = 60
N_BASE_CELLS = 10
N_ITERATIONS = 375
WINDOW = 25
CAP_SECONDS = 4 * 3600
TOL = 1e-9

RUN_ARTIFACTS = [
    ("run_config.json", "trainer-resolved configuration and provenance", "copied"),
    ("run_summary.json", "trainer run summary: accounting, final round", "copied"),
    ("train_records.jsonl", "one record per training update", "copied"),
    ("eval_records.jsonl", "one record per evaluation round, incl. v2_behaviour", "copied"),
    ("episode_failures.jsonl", "failure ledger", "copied"),
    ("episode_outcomes.jsonl", "per-episode outcomes incl. wake diagnostics (train + eval)",
     "external_only; evaluation outcomes and wakes, training summaries extracted"),
    ("train_credit_diagnostics.jsonl", "per-transition credit of every productive update",
     "external_only; counts and FD-selected-ego immediate-FD rows extracted"),
    ("checkpoints/ckpt_iter0374.pt", "final actor checkpoint (never loaded)",
     "external_only; hash reference"),
]
LAUNCHER_ARTIFACTS = [
    ("launcher_record.json", "launcher / watchdog record", "copied"),
    ("native_exit_code.txt", "exit code written by the Python launcher", "copied"),
    ("invocation_start_local.txt", "launcher start timestamp", "copied"),
    ("env_probe.json", "environment probe run just before the training process", "copied"),
    ("training_console.log", "trainer console output (*.log is git-ignored)",
     "external_only; hash reference; scanned for Traceback / CRASH"),
]
LAUNCHER_OPTIONAL = [("pre_update_identity.json", "arm-B live pre-update identity check",
                      "copied")]


class EvidenceError(RuntimeError):
    """A schema, provenance, accounting or arithmetic check failed."""


def require(cond, msg, *args):
    if not cond:
        raise EvidenceError(msg % args if args else msg)


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def rel_close(a, b, rel=1e-12):
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(a)), abs(float(b)))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


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


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise EvidenceError("%s line %d is not valid JSON: %s" % (path, n, exc))


def read_jsonl(path: Path):
    return list(iter_jsonl(path))


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
    xs = sorted(float(x) for x in xs)
    if not xs:
        return {"n": 0, "mean": None, "median": None, "std_pop": None, "min": None, "max": None}
    m = math.fsum(xs) / len(xs)
    return {"n": len(xs), "mean": m, "median": statistics.median(xs),
            "std_pop": math.sqrt(math.fsum((x - m) ** 2 for x in xs) / len(xs)),
            "min": xs[0], "max": xs[-1]}


def rate(k, n):
    return {"count": k, "denominator": n, "rate": (k / n) if n else None}


def total(xs):
    return math.fsum(float(x) for x in xs)


# ----------------------------------------------------------------------------- hashing
def hash_sources(manifest: Path, package_dir: Path):
    entries = []

    def add(group, name, p, role, disp, git_path=None, required=True):
        if not p.exists():
            require(not required, "missing artifact %s", p)
            return
        entries.append({"group": group, "name": name, "absolute_path": str(p),
                        "bytes": p.stat().st_size, "sha256": sha256_file(p), "role": role,
                        "disposition": disp, "git_path": git_path})
    for arm, a in ARMS.items():
        run_dir, side = a["run"], Path(str(a["run"]) + "__launcher")
        tag = "arm_" + arm.lower()
        for name, role, disp in RUN_ARTIFACTS:
            add(tag + "_run", name, run_dir / name, role, disp,
                ("run_artifacts/%s/%s" % (tag, name)) if disp == "copied" else None)
        for name, role, disp in LAUNCHER_ARTIFACTS:
            add(tag + "_launcher", name, side / name, role, disp,
                ("run_artifacts/%s/launcher/%s" % (tag, name)) if disp == "copied" else None)
        for name, role, disp in LAUNCHER_OPTIONAL:
            add(tag + "_launcher", name, side / name, role, disp,
                "run_artifacts/%s/launcher/%s" % (tag, name), required=(arm == "B"))
    add("benchmark", "benchmark_manifest.json", manifest,
        "frozen GENERALIZED-V2 benchmark manifest consumed by both arms",
        "external_only; hash reference")
    for p in sorted((package_dir / "prelaunch").glob("*")):
        add("prelaunch", p.name, p, "pre-launch verification record", "committed",
            "prelaunch/" + p.name)
    for name in ("authorized_plan.json", "configs/arm_a_c2p25.json", "configs/arm_b_c4p5.json"):
        add("plan", name, package_dir / name, "committed before launch", "committed", name)
    return entries


def render_sha_file(entries) -> str:
    lines = ["# sha256  bytes  group  absolute_path",
             "# Source artifacts of the evidence package. Generated by scripts/extract_evidence.py."]
    for e in entries:
        lines.append("%s  %d  %s  %s" % (e["sha256"], e["bytes"], e["group"], e["absolute_path"]))
    return "\n".join(lines) + "\n"


def parse_sha_file(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, size, _group, abspath = line.split("  ", 3)
        out[abspath] = (sha, int(size))
    return out


# ----------------------------------------------------------------------------- precheck
def precheck(arm, a, rc, summary, train_records, eval_records, failures, launcher, preflight):
    run_dir, side = a["run"], Path(str(a["run"]) + "__launcher")
    tc, prov = rc["train_config"], rc["provenance"]
    git = prov["git"]
    require(git["commit"] == MEASURED_SHA, "%s: run commit %s", arm, git["commit"])
    require(git["dirty"] is False and git["dirty_path_count"] == 0, "%s: run not clean", arm)
    require(prov["packages"]["match_aou"]["path"].lower().startswith(WORKTREE.lower()),
            "%s: match_aou not imported from the measured worktree", arm)
    require(rc["training"]["actor_observation_id"] == ACTOR_OBSERVATION_ID, "%s: obs id", arm)
    require(rc["training"]["action_representation_id"] == REPRESENTATION, "%s: repr", arm)
    require(rc["training"]["mode"] == "actor_only", "%s: training mode", arm)
    pf = preflight["arms"][arm]
    require(tc == pf["train_config"], "%s: train_config differs from the pre-launch resolution",
            arm)
    require(rc["config_source"] == pf["config_source"], "%s: config_source differs", arm)
    argv = prov["invocation"]["argv"][1:]
    require(argv == pf["argv_after_python"][2:], "%s: argv %s differs from the plan %s", arm,
            argv, pf["argv_after_python"][2:])
    require(Path(tc["output_dir"]) == run_dir, "%s: output_dir", arm)
    require(rc["difficulty"]["reward"]["aircraft_penalty_coeff"] == a["coeff"]
            and rc["difficulty"]["reward"]["formula_changed"] is False, "%s: reward block", arm)
    require(all(float(t["aircraft_penalty_coeff"]) == a["coeff"] for t in train_records),
            "%s: train record coefficient", arm)
    # completion + accounting
    require(len(train_records) == N_ITERATIONS, "%s: %d train records", arm, len(train_records))
    require(summary["updates_completed"] == N_ITERATIONS, "%s: updates_completed", arm)
    require(summary["accounting_reconciled"] is True, "%s: accounting not reconciled", arm)
    require((run_dir / "checkpoints" / "ckpt_iter0374.pt").exists(), "%s: final ckpt", arm)
    require(summary["train_episodes_successful"] == 3000, "%s: successful training", arm)
    require(summary["train_episodes_attempted"] <= 4500, "%s: attempt budget", arm)
    require(summary["eval_episodes_attempted"] == N_ROUNDS * N_EVAL_MEMBERS_PER_ROUND,
            "%s: eval episodes attempted", arm)
    require(summary["n_eval_rounds"] == N_ROUNDS, "%s: eval rounds", arm)
    nonfinite = [(int(t["iteration"]), k) for t in train_records
                 for k in ("policy_loss", "total_loss", "grad_norm", "approx_kl", "entropy",
                           "mean_ratio", "adv_std_raw", "train_reward_mean")
                 if t.get(k) is not None and not finite(t.get(k))]
    require(not nonfinite, "%s: non-finite optimisation values %s", arm, nonfinite[:5])
    fail_types = Counter((f.get("phase"), f.get("pipeline_stage"), f.get("error_type"))
                         for f in failures)
    for (ph, st, et) in fail_types:
        require(ph == "train" and st == "setup" and et == "FuelDamageError",
                "%s: unexpected failure class %s/%s/%s", arm, ph, st, et)
    require(all("no_fd_eligible_ego" in (f.get("error_message") or "") for f in failures),
            "%s: non-eligibility FuelDamageError", arm)
    require(launcher["status"] == "finished" and launcher["termination_reason"] == "exited",
            "%s: launcher did not finish normally", arm)
    require(launcher["exit_code"] == 0, "%s: exit code %r", arm, launcher["exit_code"])
    require(launcher["walltime_seconds"] < CAP_SECONDS, "%s: walltime cap", arm)
    console = (side / "training_console.log").read_text(encoding="utf-8", errors="replace")
    n_tb, n_crash = console.count("Traceback"), console.count("CRASH")
    require(n_crash == 0, "%s: console carries CRASH: %d", arm, n_crash)
    chains = console.count('raise EpisodeAttemptError("setup", exc) from exc')
    ends = console.count("EpisodeAttemptError: setup failed: FuelDamageError: no_fd_eligible_ego")
    require(n_tb == 2 * len(failures) and chains == len(failures) and ends == len(failures),
            "%s: console tracebacks (%d) are not exactly the %d accounted setup failures",
            arm, n_tb, len(failures))
    fs = summary["final_eval_selection"]
    require(fs["identity"] == {"evaluation_stage": "post_update", "updates_completed": 375,
                               "eval_round_ordinal": 15}, "%s: trainer final selection", arm)
    return {
        "arm": arm, "aircraft_penalty_coeff": a["coeff"], "run_dir": str(run_dir),
        "provenance": {"measured_code_sha": git["commit"], "dirty": git["dirty"],
                       "repo_root": git["repo_root"],
                       "match_aou_path": prov["packages"]["match_aou"]["path"],
                       "blade_path": prov["packages"]["blade"]["path"],
                       "python": prov["python"]["version"],
                       "torch": prov["packages"]["torch"]["version"],
                       "bonmin_available": prov["solver"]["bonmin"]["available"],
                       "argv": argv, "argv_matches_plan": True,
                       "train_config_equals_prelaunch_resolution": True,
                       "config_source": rc["config_source"]},
        "completion": {"updates_completed": summary["updates_completed"],
                       "train_records": len(train_records),
                       "final_checkpoint": "checkpoints/ckpt_iter0374.pt",
                       "launcher": {k: launcher.get(k) for k in (
                           "started_at", "ended_at", "walltime_seconds", "exit_code",
                           "termination_reason", "launcher_pid", "child_pid",
                           "pre_update_check", "monitor_errors")},
                       "run_seconds_harness": summary.get("run_seconds"),
                       "console_traceback_count": n_tb, "console_crash_count": n_crash,
                       "console_tracebacks_classification": (
                           "all belong to the accounted no_fd_eligible_ego setup failures "
                           "(two chained tracebacks per ledger row)"),
                       "non_finite_optimisation_values": 0},
        "accounting": {k: summary.get(k) for k in (
            "accounting_reconciled", "train_episodes_attempted", "train_episodes_successful",
            "train_episodes_failed", "train_replacement_attempts", "eval_episodes_attempted",
            "eval_episodes_successful", "eval_episodes_failed", "failures_by_phase",
            "failures_by_pipeline_stage", "failures_by_error_type", "n_eval_rounds")},
        "failed_training_seeds": sorted(int(f["seed"]) for f in failures),
        "final_eval_selection": fs,
        "schemas": summary.get("observed_artifact_schema"),
    }


def cross_arm_identity(rcs, probes):
    diffs = {}
    for k in ("construction", "episode_design", "derived_split", "base_scenario"):
        if rcs["A"].get(k) != rcs["B"].get(k):
            diffs[k] = "differs"
    da, db = dict(rcs["A"]["difficulty"]), dict(rcs["B"]["difficulty"])
    ra, rb = dict(da.pop("reward")), dict(db.pop("reward"))
    if da != db:
        diffs["difficulty (excluding reward)"] = "differs"
    ra.pop("aircraft_penalty_coeff"), rb.pop("aircraft_penalty_coeff")
    if ra != rb:
        diffs["difficulty.reward (excluding coefficient)"] = "differs"
    pa, pb = rcs["A"]["provenance"], rcs["B"]["provenance"]
    for k in ("python", "platform", "packages"):
        if pa.get(k) != pb.get(k):
            diffs["provenance." + k] = "differs"
    if pa["solver"]["bonmin"]["executable"] != pb["solver"]["bonmin"]["executable"]:
        diffs["provenance.solver"] = "differs"
    ta = {k: v for k, v in rcs["A"]["train_config"].items()}
    tb = {k: v for k, v in rcs["B"]["train_config"].items()}
    tdiff = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
    ea, eb = dict(probes["A"]), dict(probes["B"])
    probe_diff = sorted(k for k in set(ea) | set(eb) if ea.get(k) != eb.get(k))
    require(tdiff == ["aircraft_penalty_coeff", "output_dir"], "train_config diff %s", tdiff)
    require(not diffs, "run_config identity differs between arms: %s", diffs)
    require(probe_diff == [], "environment probes differ between arms: %s", probe_diff)
    return {"train_config_diff_between_arms": {k: {"A": ta.get(k), "B": tb.get(k)}
                                               for k in tdiff},
            "run_config_blocks_identical": ["construction", "episode_design", "derived_split",
                                            "base_scenario", "difficulty (except the "
                                            "coefficient)", "provenance python / platform / "
                                            "packages / solver"],
            "env_probe_identical": True,
            "env_probe": {k: probes["A"].get(k) for k in ("python", "executable", "platform",
                                                           "cpu_count", "packages", "torch",
                                                           "device", "env")}}


# ----------------------------------------------------------------------------- evaluation
def wake_p_abort(wake):
    v = (wake.get("semantic_probability_per_meta_action") or {}).get(ABORT)
    leaves = [l for l in wake.get("semantic_leaves") or () if l.get("meta_action_name") == ABORT]
    require(len(leaves) == 1, "semantic wake without exactly one ABORT leaf")
    require(close(leaves[0]["probability"], v, 1e-12), "ABORT leaf probability mismatch")
    require(finite(v), "P(ABORT) missing")
    return float(v)


def outcome_row(arm, coeff, r):
    """One evaluation episode: identity, outcome, reference terms and exact rescoring."""
    bv = r.get("benchmark_v2") or {}
    for f in ("u_achieved", "u_ref", "u_prefix", "u_cont_ref", "u_post", "u_aircraft",
              "reward", "reward_ratio", "reward_penalty"):
        require(finite(r.get(f)), "%s: outcome field %s not finite (%s)", arm, f,
                r.get("benchmark_group_key"))
    require(r.get("reference_policy") == "event_conditioned_continuation_v1", "reference policy")
    require(r.get("u_oracle") is None, "u_oracle present under the event-conditioned policy")
    n_dead, agents = int(r["n_dead"]), int(r["agent_count"])
    D = abs(float(r["u_ref"])) + REGRET_EPS
    q = float(r["reward_ratio"])
    p = float(r["u_aircraft"]) * n_dead / D
    require(rel_close(q, (float(r["u_achieved"]) - float(r["u_ref"])) / D), "%s: q identity", arm)
    require(rel_close(float(r["u_achieved"]), float(r["u_prefix"]) + float(r["u_post"])),
            "%s: u_achieved != u_prefix + u_post", arm)
    require(rel_close(float(r["u_ref"]), float(r["u_prefix"]) + float(r["u_cont_ref"])),
            "%s: u_ref != u_prefix + u_cont_ref", arm)
    require(rel_close(float(r["reward_penalty"]), coeff * p), "%s: penalty != c*p", arm)
    require(rel_close(float(r["reward"]), q - float(r["reward_penalty"])), "%s: R != q - pen", arm)
    fd = [w for w in r.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
    return {
        "arm": arm, "native_coeff": coeff,
        "evaluation_stage": r["phase"], "eval_round_ordinal": int(r["eval_round_ordinal"]),
        "updates_completed": int(r["updates_completed"]),
        "benchmark_group_key": r["benchmark_group_key"], "base_cell": bv.get("base_cell"),
        "member_cell": bv.get("member_cell"), "seed": r.get("seed"),
        "episode_tag": r.get("episode_tag"), "agent_count": agents,
        "ended": r.get("ended"), "ticks": r.get("ticks"), "n_wakes": r.get("n_wakes"),
        "n_dead": n_dead, "survivors": agents - n_dead,
        "survivors_at_physical_completion": (agents - n_dead) if r.get("ended") == "done" else None,
        "u_achieved": float(r["u_achieved"]), "u_prefix": float(r["u_prefix"]),
        "u_post": float(r["u_post"]), "u_ref": float(r["u_ref"]),
        "u_cont_ref": float(r["u_cont_ref"]), "u_aircraft": float(r["u_aircraft"]),
        "reference_kind": r.get("reference_kind"),
        "reference_checkpoint_tick": r.get("reference_checkpoint_tick"),
        "reference_allocated_task_count": r.get("reference_allocated_task_count"),
        "reference_continuation_agent_count": r.get("reference_continuation_agent_count"),
        "targets_confirmed_unique": r.get("targets_confirmed_unique"),
        "targets_realized": r.get("targets_realized"),
        "unique_completed_targets": r.get("unique_completed_targets"),
        "scored_completed_targets": r.get("scored_completed_targets"),
        "unscored_completed_targets": r.get("unscored_completed_targets"),
        "fd_fired": r.get("fd_fired"), "fd_ego_id": r.get("fd_ego_id"),
        "fd_event_tick": r.get("fd_event_tick"),
        "fd_wake_selected_meta_action_name": fd[0]["selected_meta_action_name"] if fd else None,
        "fd_wake_p_abort": wake_p_abort(fd[0]) if fd else None,
        "fd_rtb_command_issued": r.get("fd_rtb_command_issued"),
        "denominator_D": D, "q_ratio": q, "p_loss_per_unit_c": p,
        "reward_native": float(r["reward"]), "reward_penalty_native": float(r["reward_penalty"]),
        "R_c2p25": q - 2.25 * p, "R_c4p5": q - 4.5 * p,
    }


def extract_eval(outcomes_path: Path, arm: str, coeff: float):
    rows, fd_rows = [], []
    members = defaultdict(lambda: defaultdict(set))
    round_updates = {}
    for r in iter_jsonl(outcomes_path):
        if r.get("phase") not in EVAL_PHASES:
            continue
        ro, upd = int(r["eval_round_ordinal"]), int(r["updates_completed"])
        require(round_updates.setdefault(ro, upd) == upd, "%s round %d mixes updates", arm, ro)
        require(r.get("benchmark_manifest_id") == MANIFEST_ID, "%s: manifest id", arm)
        require((r.get("benchmark_v2") or {}).get("profile") == "development", "%s profile", arm)
        require(r.get("action_representation_id") == REPRESENTATION, "%s: representation", arm)
        require(r.get("actor_observation_id") == ACTOR_OBSERVATION_ID, "%s: observation", arm)
        require(r.get("schema_version") == 5 and r.get("wake_diagnostics_schema_version") == 3,
                "%s: outcome schema versions", arm)
        cell = (r.get("benchmark_v2") or {}).get("member_cell")
        gk = r["benchmark_group_key"]
        require(cell in CELLS, "%s: unknown member cell %r", arm, cell)
        require(cell not in members[ro][gk], "%s: duplicate member %s/%s r%d", arm, gk, cell, ro)
        members[ro][gk].add(cell)
        row = outcome_row(arm, coeff, r)
        rows.append(row)
        if cell == "clean":
            continue
        require(r.get("severity") == cell, "%s: severity != member cell", arm)
        fd = [w for w in r.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        require(len(fd) == 1, "%s r%d %s %s: %d immediate-FD wakes", arm, ro, gk, cell, len(fd))
        w = fd[0]
        require(w.get("ego_id") == r.get("fd_ego_id"), "%s: FD wake ego != fd_ego_id", arm)
        fd_rows.append({
            "arm": arm, "eval_round_ordinal": ro, "updates_completed": upd,
            "evaluation_stage": r["phase"], "benchmark_group_key": gk,
            "base_cell": row["base_cell"], "member_cell": cell, "seed": r.get("seed"),
            "fd_ego_id": r.get("fd_ego_id"), "tick": w.get("tick"),
            "p_abort": wake_p_abort(w),
            "semantic_probability_per_meta_action": w.get("semantic_probability_per_meta_action"),
            "selected_meta_action_name": w.get("selected_meta_action_name"),
            "deterministic_argmax_meta_action_name": w.get("deterministic_argmax_meta_action_name"),
            "selected_leaf": w.get("selected_leaf"),
            "n_valid_semantic_leaves": w.get("n_valid_semantic_leaves"),
            "n_abort_legal_nodes": w.get("n_abort_legal_nodes"),
            "ego_fuel_norm": w.get("ego_fuel_norm"),
            "ego_mission_fuel_slack_norm": w.get("ego_mission_fuel_slack_norm")})
    key = lambda x: (x["eval_round_ordinal"], x["benchmark_group_key"], x["member_cell"])
    rows.sort(key=key)
    fd_rows.sort(key=key)
    return rows, fd_rows, members, round_updates


def behaviour_by_round(fd_rows, members, round_updates, eval_records, arm):
    require(sorted(round_updates) == list(range(N_ROUNDS)), "%s: rounds %s", arm,
            sorted(round_updates))
    by_round = defaultdict(lambda: defaultdict(dict))
    for r in fd_rows:
        by_round[r["eval_round_ordinal"]][r["benchmark_group_key"]][r["member_cell"]] = r
    ev_by_round = {int(e["eval_round_ordinal"]): e for e in eval_records}
    require(sorted(ev_by_round) == list(range(N_ROUNDS)), "%s: eval_records rounds", arm)
    out = []
    for ro in range(N_ROUNDS):
        ev = ev_by_round[ro]
        require(int(ev["updates_completed"]) == round_updates[ro], "%s r%d updates", arm, ro)
        require(ev["evaluation_stage"] == ("pre_update" if ro == 0 else "post_update"),
                "%s r%d stage", arm, ro)
        require(round_updates[ro] == 25 * ro, "%s r%d is not at update %d", arm, ro, 25 * ro)
        vb = ev["v2_behaviour"]
        n_members = sum(len(c) for c in members[ro].values())
        require(n_members == N_EVAL_MEMBERS_PER_ROUND, "%s r%d: %d members", arm, ro, n_members)
        groups, cell_deltas = [], defaultdict(list)
        sw = rev = both_abort = both_non = 0
        for gk in sorted(members[ro]):
            m = by_round[ro][gk]
            complete = members[ro][gk] == set(CELLS)
            eligible = complete and "mild" in m and "severe" in m
            g = {"group_key": gk, "complete": complete, "metric_eligible": eligible}
            if eligible:
                d = m["severe"]["p_abort"] - m["mild"]["p_abort"]
                ms = m["mild"]["selected_meta_action_name"]
                ss = m["severe"]["selected_meta_action_name"]
                g.update({"base_cell": m["mild"]["base_cell"],
                          "p_abort_mild": m["mild"]["p_abort"],
                          "p_abort_severe": m["severe"]["p_abort"], "severe_minus_mild": d,
                          "mild_selected": ms, "severe_selected": ss,
                          "directional_switch": ms != ABORT and ss == ABORT,
                          "reverse_switch": ms == ABORT and ss != ABORT,
                          "both_abort": ms == ABORT and ss == ABORT,
                          "both_non_abort": ms != ABORT and ss != ABORT})
                sw += g["directional_switch"]
                rev += g["reverse_switch"]
                both_abort += g["both_abort"]
                both_non += g["both_non_abort"]
                cell_deltas[g["base_cell"]].append(d)
            groups.append(g)
        cells = {c: {"n_metric_eligible_groups": len(v), "severe_minus_mild_mean": mean(v)}
                 for c, v in sorted(cell_deltas.items())}
        defined = [c for c in cells if cells[c]["n_metric_eligible_groups"] > 0]
        macro = (mean(cells[c]["severe_minus_mild_mean"] for c in defined)
                 if len(defined) == N_BASE_CELLS else None)
        elig = [g for g in groups if g["metric_eligible"]]
        pooled = mean(g["severe_minus_mild"] for g in elig)
        mild_p = [by_round[ro][gk]["mild"]["p_abort"] for gk in sorted(by_round[ro])
                  if "mild" in by_round[ro][gk]]
        sev_p = [by_round[ro][gk]["severe"]["p_abort"] for gk in sorted(by_round[ro])
                 if "severe" in by_round[ro][gk]]
        mild_abort = sum(1 for gk in by_round[ro] if "mild" in by_round[ro][gk]
                         and by_round[ro][gk]["mild"]["selected_meta_action_name"] == ABORT)
        sev_abort = sum(1 for gk in by_round[ro] if "severe" in by_round[ro][gk]
                        and by_round[ro][gk]["severe"]["selected_meta_action_name"] == ABORT)
        # --- cross-check against the trainer's own per-round record ---
        require(vb["n_groups_metric_eligible"] == len(elig), "%s r%d eligible", arm, ro)
        require(vb["directional_switch_count"] == sw and vb["reverse_switch_count"] == rev,
                "%s r%d switch counts differ from trainer", arm, ro)
        require(close(vb["pooled_mean_over_groups"], pooled), "%s r%d pooled", arm, ro)
        require((vb["macro_mean_over_base_cells"] is None and macro is None)
                or close(vb["macro_mean_over_base_cells"], macro), "%s r%d macro", arm, ro)
        for c, cv in cells.items():
            require(close(vb["by_base_cell"][c]["severe_minus_mild_abort_mass_mean"],
                          cv["severe_minus_mild_mean"]), "%s r%d cell %s", arm, ro, c)
        tg = {g["group_key"]: g for g in vb["groups"]}
        for g in elig:
            t = tg[g["group_key"]]
            require(close(t["p_abort_mild"], g["p_abort_mild"], 1e-12)
                    and close(t["p_abort_severe"], g["p_abort_severe"], 1e-12)
                    and t["mild_selected_meta_action"] == g["mild_selected"]
                    and t["severe_selected_meta_action"] == g["severe_selected"],
                    "%s r%d group %s differs from trainer", arm, ro, g["group_key"])
        require(vb.get("action_representation_ids_observed") == [REPRESENTATION],
                "%s r%d representations", arm, ro)
        out.append({
            "eval_round_ordinal": ro, "evaluation_stage": ev["evaluation_stage"],
            "updates_completed": round_updates[ro],
            "n_members": n_members, "n_groups": len(groups),
            "n_groups_complete": sum(g["complete"] for g in groups),
            "n_groups_metric_eligible": len(elig),
            "p_abort_mild_mean": mean(mild_p), "n_mild_members": len(mild_p),
            "p_abort_severe_mean": mean(sev_p), "n_severe_members": len(sev_p),
            "selected_abort_mild": rate(mild_abort, len(mild_p)),
            "selected_abort_severe": rate(sev_abort, len(sev_p)),
            "severe_minus_mild_pooled_mean": pooled,
            "severe_minus_mild_macro_over_base_cells": macro,
            "macro_n_base_cells_defined": len(defined),
            "directional_switches": rate(sw, len(elig)),
            "reverse_switches": rate(rev, len(elig)),
            "both_abort": rate(both_abort, len(elig)),
            "both_non_abort": rate(both_non, len(elig)),
            "by_base_cell": cells, "groups": groups,
            "trainer_cross_check": "passed (macro, pooled, per-cell, per-group, switch counts)",
        })
    return out


def common_group_contrast(beh):
    """Descriptive only: final-round macro over groups metric-eligible in BOTH arms."""
    fa, fb = beh["A"][-1], beh["B"][-1]
    ga = {g["group_key"]: g for g in fa["groups"] if g["metric_eligible"]}
    gb = {g["group_key"]: g for g in fb["groups"] if g["metric_eligible"]}
    common = sorted(set(ga) & set(gb))
    out = {"n_common_groups": len(common), "eligible_sets_identical": set(ga) == set(gb),
           "only_in_A": sorted(set(ga) - set(gb)), "only_in_B": sorted(set(gb) - set(ga)),
           "label": "COMMON-GROUP DESCRIPTIVE CONTRAST - not the primary endpoint"}
    for arm, gs in (("A", ga), ("B", gb)):
        cells = defaultdict(list)
        for k in common:
            cells[gs[k]["base_cell"]].append(gs[k]["severe_minus_mild"])
        out[arm] = {"n_cells": len(cells),
                    "macro": mean(mean(v) for v in cells.values())
                    if len(cells) == N_BASE_CELLS else None}
    return out


# ----------------------------------------------------------------------------- outcomes
def group_summary(rows):
    n = len(rows)
    ended = Counter(r["ended"] for r in rows)
    sev = [r for r in rows if r["member_cell"] in SEVERITIES]
    return {
        "n_episodes": n,
        "u_achieved_sum": total(r["u_achieved"] for r in rows),
        "u_achieved_mean": mean(r["u_achieved"] for r in rows),
        "u_post_mean": mean(r["u_post"] for r in rows),
        "u_ref_mean": mean(r["u_ref"] for r in rows),
        "unique_completed_targets_sum": sum(int(r["unique_completed_targets"] or 0) for r in rows),
        "scored_completed_targets_sum": sum(int(r["scored_completed_targets"] or 0) for r in rows),
        "unscored_completed_targets_sum": sum(int(r["unscored_completed_targets"] or 0)
                                              for r in rows),
        "targets_realized_sum": sum(int(r["targets_realized"] or 0) for r in rows),
        "deaths_sum": sum(r["n_dead"] for r in rows),
        "episodes_with_any_death": rate(sum(1 for r in rows if r["n_dead"] > 0), n),
        "agents_sum": sum(r["agent_count"] for r in rows),
        "survivors_sum": sum(r["survivors"] for r in rows),
        "survivors_at_physical_completion_sum": sum(r["survivors_at_physical_completion"] or 0
                                                    for r in rows),
        "ended_counts": dict(sorted(ended.items())),
        "fd_wake_selected_abort": rate(sum(1 for r in sev if r["fd_wake_selected_meta_action_name"]
                                           == ABORT), len(sev)),
        "fd_rtb_command_issued": rate(sum(1 for r in sev if r["fd_rtb_command_issued"]), len(sev)),
        "reward_native_mean": mean(r["reward_native"] for r in rows),
        "q_ratio_mean": mean(r["q_ratio"] for r in rows),
        "p_loss_per_unit_c_mean": mean(r["p_loss_per_unit_c"] for r in rows),
        "R_c2p25_mean": mean(r["R_c2p25"] for r in rows),
        "R_c4p5_mean": mean(r["R_c4p5"] for r in rows),
    }


def outcome_summaries(rows_by_arm):
    out = {}
    for arm, rows in rows_by_arm.items():
        by_round = defaultdict(list)
        by_rc = defaultdict(list)
        for r in rows:
            by_round[r["eval_round_ordinal"]].append(r)
            by_rc[(r["eval_round_ordinal"], r["member_cell"])].append(r)
        out[arm] = [{"eval_round_ordinal": ro, "updates_completed": rs[0]["updates_completed"],
                     "all_members": group_summary(rs),
                     "by_member_cell": {c: group_summary(by_rc[(ro, c)]) for c in CELLS}}
                    for ro, rs in sorted(by_round.items())]
    return out


REF_TERMS = ("u_ref", "u_prefix", "u_cont_ref", "u_aircraft", "reference_kind",
             "reference_checkpoint_tick", "reference_allocated_task_count",
             "reference_continuation_agent_count")


def paired_contrasts(rows_by_arm):
    ia = {(r["eval_round_ordinal"], r["benchmark_group_key"], r["member_cell"]): r
          for r in rows_by_arm["A"]}
    ib = {(r["eval_round_ordinal"], r["benchmark_group_key"], r["member_cell"]): r
          for r in rows_by_arm["B"]}
    require(set(ia) == set(ib), "the two arms' evaluation member sets differ")
    pairs = []
    for k in sorted(ia):
        a, b = ia[k], ib[k]
        for f in ("seed", "episode_tag", "base_cell", "agent_count", "updates_completed"):
            require(a[f] == b[f], "frozen world identity differs at %s: %s", k, f)
        row = {"eval_round_ordinal": k[0], "updates_completed": a["updates_completed"],
               "benchmark_group_key": k[1], "member_cell": k[2], "base_cell": a["base_cell"],
               "seed": a["seed"], "reference_terms_identical": all(a[f] == b[f]
                                                                   for f in REF_TERMS)}
        for f in ("u_achieved", "u_post", "n_dead", "survivors", "ended", "ticks",
                  "unique_completed_targets", "scored_completed_targets",
                  "unscored_completed_targets", "fd_wake_selected_meta_action_name",
                  "fd_wake_p_abort", "fd_rtb_command_issued", "q_ratio", "p_loss_per_unit_c",
                  "R_c2p25", "R_c4p5", "reward_native") + REF_TERMS:
            row["A_" + f], row["B_" + f] = a[f], b[f]
        for f in ("u_achieved", "n_dead", "R_c2p25", "R_c4p5"):
            row["B_minus_A_" + f] = b[f] - a[f]
        pairs.append(row)
    by_round = defaultdict(list)
    for p in pairs:
        by_round[(p["eval_round_ordinal"], p["member_cell"])].append(p)
        by_round[(p["eval_round_ordinal"], "all")].append(p)
    summary = []
    for (ro, cell), ps in sorted(by_round.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        s = {"eval_round_ordinal": ro, "member_cell": cell, "n_worlds": len(ps),
             "reference_terms_identical": rate(sum(p["reference_terms_identical"] for p in ps),
                                               len(ps)),
             "fd_action_identical": rate(sum(p["A_fd_wake_selected_meta_action_name"] ==
                                             p["B_fd_wake_selected_meta_action_name"]
                                             for p in ps), len(ps))}
        for f in ("u_achieved", "n_dead", "R_c2p25", "R_c4p5"):
            d = [p["B_minus_A_" + f] for p in ps]
            s[f] = {"B_minus_A_mean": mean(d), "n_B_greater": sum(x > 0 for x in d),
                    "n_B_less": sum(x < 0 for x in d), "n_equal": sum(x == 0 for x in d)}
        summary.append(s)
    return pairs, summary


# ----------------------------------------------------------------------------- training
def training_extraction(arm, train_records, credit_path: Path, outcomes_path: Path, coeff):
    per_update = defaultdict(Counter)
    fd_rows = []
    for row in iter_jsonl(credit_path):
        it = int(row["iteration"])
        per_update[it][row["wake_kind"]] += 1
        require(row["wake_kind"] in WAKE_KINDS, "%s: unknown wake kind", arm)
        j = row.get("measurement_join") or {}
        if row["wake_kind"] == FD_WAKE and j.get("is_fd_selected_ego"):
            fd_rows.append({"arm": arm, "iteration": it, "episode_seed": row["episode_seed"],
                            "ego_id": row["ego_id"], "tick": row["tick"],
                            "severity": j.get("severity"),
                            "selected_meta_action_name": row["selected_meta_action_name"],
                            "episode_reward_native": row["episode_reward"],
                            "raw_advantage": row["raw_advantage"],
                            "normalized_advantage": row["normalized_advantage"],
                            "batch_n_transitions": row["batch_n_transitions"]})
    tr_by_it = {int(t["iteration"]): t for t in train_records}
    for it, t in tr_by_it.items():
        if int(t.get("n_epochs_run") or 0) > 0:
            require(sum(per_update[it].values()) == int(t["n_transitions"]),
                    "%s: credit rows do not cover update %d", arm, it)
    # training-episode outcomes, rescored (same algebra as evaluation; stochastic actor)
    tro = defaultdict(list)
    for r in iter_jsonl(outcomes_path):
        if r.get("phase") != "train":
            continue
        D = abs(float(r["u_ref"])) + REGRET_EPS
        p = float(r["u_aircraft"]) * int(r["n_dead"]) / D
        q = float(r["reward_ratio"])
        require(rel_close(float(r["reward"]), q - coeff * p), "%s: train reward algebra", arm)
        tro[(int(r["iteration"]) // WINDOW, r.get("cell"))].append(
            (float(r["u_achieved"]), int(r["n_dead"]), q - 2.25 * p, q - 4.5 * p))
    per_iteration = []
    for it in sorted(tr_by_it):
        t = tr_by_it[it]
        per_iteration.append({
            "iteration": it, "updates_completed": t["updates_completed"],
            "n_attempted": t["n_attempted"], "n_successful": t["n_successful"],
            "n_failed": t["n_failed"], "n_transitions": t["n_transitions"],
            "n_epochs_run": t["n_epochs_run"],
            "transitions_by_wake_kind": {k: per_update[it].get(k, 0) for k in WAKE_KINDS},
            "meta_action_counts": t.get("meta_action_counts"),
            "n_mild_fd_wakes": t.get("n_mild_fd_wakes"),
            "n_severe_fd_wakes": t.get("n_severe_fd_wakes"),
            "fd_meta_action_counts_mild": t.get("fd_meta_action_counts_mild"),
            "fd_meta_action_counts_severe": t.get("fd_meta_action_counts_severe"),
            "deaths": t.get("deaths"), "train_reward_mean_native_c": t.get("train_reward_mean"),
            "adv_std_raw": t.get("adv_std_raw"), "grad_norm": t.get("grad_norm"),
            "approx_kl": t.get("approx_kl"), "clip_fraction": t.get("clip_fraction"),
            "entropy": t.get("entropy")})
    windows = []
    for lo in range(0, N_ITERATIONS, WINDOW):
        w = [r for r in fd_rows if lo <= r["iteration"] < lo + WINDOW]
        blk = {"iterations": [lo, lo + WINDOW - 1],
               "transitions_by_wake_kind": {k: sum(per_update[i].get(k, 0)
                                                   for i in range(lo, lo + WINDOW))
                                            for k in WAKE_KINDS}}
        for s in SEVERITIES:
            ws = [r for r in w if r["severity"] == s]
            ab = [r for r in ws if r["selected_meta_action_name"] == ABORT]
            na = [r for r in ws if r["selected_meta_action_name"] != ABORT]
            blk[s] = {"n_fd_selected_ego_immediate_fd": len(ws),
                      "selected_abort": rate(len(ab), len(ws)),
                      "raw_advantage_abort": desc(r["raw_advantage"] for r in ab),
                      "raw_advantage_non_abort": desc(r["raw_advantage"] for r in na),
                      "normalized_advantage_abort": desc(r["normalized_advantage"] for r in ab),
                      "normalized_advantage_non_abort": desc(r["normalized_advantage"]
                                                             for r in na)}
        for cell in CELLS:
            xs = tro.get((lo // WINDOW, cell), [])
            blk["train_outcomes_" + cell] = {
                "n_episodes": len(xs), "u_achieved_mean": mean(x[0] for x in xs),
                "deaths_sum": sum(x[1] for x in xs),
                "episodes_with_any_death": sum(1 for x in xs if x[1] > 0),
                "R_c2p25_mean": mean(x[2] for x in xs), "R_c4p5_mean": mean(x[3] for x in xs)}
        windows.append(blk)
    totals = {k: sum(per_update[i].get(k, 0) for i in per_update) for k in WAKE_KINDS}
    fd_tot = {s: {"n": len([r for r in fd_rows if r["severity"] == s]),
                  "selected_abort": rate(sum(1 for r in fd_rows if r["severity"] == s and
                                             r["selected_meta_action_name"] == ABORT),
                                         len([r for r in fd_rows if r["severity"] == s]))}
              for s in SEVERITIES}
    return fd_rows, {
        "arm": arm, "native_coeff": coeff, "transitions_by_wake_kind_total": totals,
        "fd_selected_ego_immediate_fd_total": fd_tot, "windows_25_updates": windows,
        "per_iteration": per_iteration,
        "limitations": ("actor-only credit with gamma = 1 and a terminal reward is episode / "
                        "chain-level; ABORT / non-ABORT advantage groups pool different "
                        "episodes and updates and are NOT local action values; raw and "
                        "normalized advantages are kept separate; training rewards are at "
                        "the arm's native c and only the R_c columns share a scale")}


# ----------------------------------------------------------------------------- tables
def acquisition_retention(beh):
    out = {}
    for arm, rounds in beh.items():
        with_sw = [r["eval_round_ordinal"] for r in rounds
                   if r["directional_switches"]["count"] > 0]
        post = [r for r in rounds if r["evaluation_stage"] == "post_update"]
        best = max(post, key=lambda r: (r["severe_minus_mild_macro_over_base_cells"]
                                        if r["severe_minus_mild_macro_over_base_cells"]
                                        is not None else -math.inf))
        fin = rounds[-1]
        out[arm] = {
            "acquisition_exploratory": {
                "rounds_with_any_directional_switch": with_sw,
                "first_post_update_round_with_directional_switch": next(
                    (r["eval_round_ordinal"] for r in post
                     if r["directional_switches"]["count"] > 0), None),
                "max_post_update_macro": best["severe_minus_mild_macro_over_base_cells"],
                "max_post_update_macro_round": best["eval_round_ordinal"],
                "max_post_update_macro_updates": best["updates_completed"],
                "max_directional_switches": max(r["directional_switches"]["count"]
                                                for r in post)},
            "retention_final_round": {
                "updates_completed": fin["updates_completed"],
                "macro": fin["severe_minus_mild_macro_over_base_cells"],
                "directional_switches": fin["directional_switches"],
                "reverse_switches": fin["reverse_switches"],
                "both_abort": fin["both_abort"], "both_non_abort": fin["both_non_abort"]},
            "note": "peaks and first rounds are exploratory, never a replacement endpoint"}
    return out


def fmt(x, nd=4):
    return "null" if x is None else ("%+.*f" % (nd, x))


def trajectory_markdown(beh, outs) -> str:
    out = ["# REWARD-01 evaluation trajectory — arm A (c = 2.25) vs arm B (c = 4.5)", "",
           "Ten-base-cell macro `P(ABORT | SEVERE) − P(ABORT | MILD)` at the certified ego's "
           "immediate-FD wake; `dir / rev / bothA / bothN` are directional switches, reverse "
           "switches, both-ABORT and both-non-ABORT groups out of the metric-eligible groups. "
           "Both arms at measured SHA `2b57019`; one training seed; every round re-measures the "
           "same 20 frozen development worlds (repeated measures). Generated by "
           "`scripts/extract_evidence.py`.", "",
           "| round | updates | A macro | A P(A\\|MILD) | A P(A\\|SEV) | A dir/rev/bothA/bothN | "
           "B macro | B P(A\\|MILD) | B P(A\\|SEV) | B dir/rev/bothA/bothN |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b in zip(beh["A"], beh["B"]):
        def sw(r):
            return "%d/%d/%d/%d of %d" % (r["directional_switches"]["count"],
                                         r["reverse_switches"]["count"],
                                         r["both_abort"]["count"], r["both_non_abort"]["count"],
                                         r["directional_switches"]["denominator"])
        out.append("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            a["eval_round_ordinal"], a["updates_completed"],
            fmt(a["severe_minus_mild_macro_over_base_cells"]), fmt(a["p_abort_mild_mean"]),
            fmt(a["p_abort_severe_mean"]), sw(a),
            fmt(b["severe_minus_mild_macro_over_base_cells"]), fmt(b["p_abort_mild_mean"]),
            fmt(b["p_abort_severe_mean"]), sw(b)))
    out += ["", "## Outcomes per round (all 60 members; SEVERE members in brackets)", "",
            "Utility = mean scored `U_prefix + U_post`; deaths = total airframes lost; "
            "`R(2.25)` / `R(4.5)` = mean algebraic rescoring `q − c·p` of the SAME saved "
            "trajectories (no replay).", "",
            "| round | A utility | A deaths [SEV] | A R(2.25) | A R(4.5) | B utility | "
            "B deaths [SEV] | B R(2.25) | B R(4.5) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b in zip(outs["A"], outs["B"]):
        def cells(r):
            m, s = r["all_members"], r["by_member_cell"]["severe"]
            return "%.2f | %d [%d] | %s | %s" % (m["u_achieved_mean"], m["deaths_sum"],
                                                 s["deaths_sum"], fmt(m["R_c2p25_mean"]),
                                                 fmt(m["R_c4p5_mean"]))
        out.append("| %d | %s | %s |" % (a["eval_round_ordinal"], cells(a), cells(b)))
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- main
def run(args):
    out, manifest = Path(args.out), Path(args.manifest)
    entries = hash_sources(manifest, out)
    by = {(e["group"], e["name"]): e for e in entries}
    require(by[("benchmark", "benchmark_manifest.json")]["sha256"] == MANIFEST_SHA256, "manifest")
    if args.verify_against:
        ref = parse_sha_file(Path(args.verify_against))
        for e in entries:
            if e["absolute_path"] in ref:
                require(ref[e["absolute_path"]] == (e["sha256"], e["bytes"]),
                        "source changed since the package: %s", e["absolute_path"])
    preflight = read_json(out / "prelaunch" / "preflight.json")
    require(preflight["pass"] is True and
            preflight["code"]["measured_code_sha"] == MEASURED_SHA, "preflight identity")

    rcs, probes, pre, beh, rows_by_arm, fd_by_arm, train_by_arm, credit_by_arm = (
        {}, {}, {}, {}, {}, {}, {}, {})
    for arm, a in ARMS.items():
        run_dir, side = a["run"], Path(str(a["run"]) + "__launcher")
        rc = read_json(run_dir / "run_config.json")
        summary = read_json(run_dir / "run_summary.json")
        train_records = read_jsonl(run_dir / "train_records.jsonl")
        eval_records = read_jsonl(run_dir / "eval_records.jsonl")
        failures = read_jsonl(run_dir / "episode_failures.jsonl")
        launcher = read_json(side / "launcher_record.json")
        rcs[arm], probes[arm] = rc, read_json(side / "env_probe.json")
        pre[arm] = precheck(arm, a, rc, summary, train_records, eval_records, failures,
                            launcher, preflight)
        rows, fd_rows, members, rounds = extract_eval(run_dir / "episode_outcomes.jsonl", arm,
                                                      a["coeff"])
        require(len(rows) == N_ROUNDS * N_EVAL_MEMBERS_PER_ROUND, "%s: eval rows %d", arm,
                len(rows))
        beh[arm] = behaviour_by_round(fd_rows, members, rounds, eval_records, arm)
        rows_by_arm[arm], fd_by_arm[arm] = rows, fd_rows
        credit_by_arm[arm], train_by_arm[arm] = training_extraction(
            arm, train_records, run_dir / "train_credit_diagnostics.jsonl",
            run_dir / "episode_outcomes.jsonl", a["coeff"])
    identity = cross_arm_identity(rcs, probes)
    pre_update = compare_pre_update.compare(ARMS["B"]["run"], ARMS["A"]["run"])
    live = read_json(Path(str(ARMS["B"]["run"]) + "__launcher") / "pre_update_identity.json")
    require(live["stop"] is False and pre_update["stop"] is False, "pre-update identity STOP")
    require({k: v for k, v in pre_update.items() if k not in ("checked_at",)}
            == {k: v for k, v in live.items()
                if k not in ("checked_at", "elapsed_seconds_at_check")},
            "the extractor's pre-update comparison differs from the launcher's live one")

    fa, fb = beh["A"][-1], beh["B"][-1]
    for f in (fa, fb):
        require(f["updates_completed"] == N_ITERATIONS and f["evaluation_stage"] == "post_update",
                "final round identity")
    ma, mb = (fa["severe_minus_mild_macro_over_base_cells"],
              fb["severe_minus_mild_macro_over_base_cells"])
    elig_same = ({g["group_key"] for g in fa["groups"] if g["metric_eligible"]} ==
                 {g["group_key"] for g in fb["groups"] if g["metric_eligible"]})
    endpoint = {
        "primary_endpoint": ("per arm: final-round ten-base-cell macro P(ABORT|SEVERE) - "
                             "P(ABORT|MILD) at the certified ego's immediate-FD wake; B minus A"),
        "final_round_selection": "semantic: evaluation_stage post_update, updates_completed 375 "
                                 "(cross-checked against each trainer's final_eval_selection)",
        "arms": {arm: {"aircraft_penalty_coeff": ARMS[arm]["coeff"], **{k: f[k] for k in (
            "eval_round_ordinal", "evaluation_stage", "updates_completed",
            "severe_minus_mild_macro_over_base_cells", "macro_n_base_cells_defined",
            "n_groups_metric_eligible", "n_groups_complete", "p_abort_mild_mean",
            "p_abort_severe_mean", "directional_switches", "reverse_switches", "both_abort",
            "both_non_abort", "selected_abort_mild", "selected_abort_severe", "by_base_cell")}}
            for arm, f in (("A", fa), ("B", fb))},
        "B_minus_A_macro": (mb - ma) if (ma is not None and mb is not None) else None,
        "eligibility_identical_between_arms": elig_same,
        "common_group_contrast": None if elig_same else common_group_contrast(beh),
        "missing_is_never_zero": True,
    }
    rescoring = {
        "rule": "R(c) = q - c*p with each trajectory's own recorded reference; algebra only",
        "native_reward_reproduced_exactly": True,
        "final_round": {arm: {c: outs_c for c, outs_c in (
            ("R_c2p25_mean", mean(r["R_c2p25"] for r in rows_by_arm[arm]
                                  if r["updates_completed"] == N_ITERATIONS)),
            ("R_c4p5_mean", mean(r["R_c4p5"] for r in rows_by_arm[arm]
                                 if r["updates_completed"] == N_ITERATIONS)))}
            for arm in ARMS},
        "note": ("native rewards at different c are NOT one scale; policies are compared only "
                 "under a common c; references can differ between policies (see paired "
                 "contrasts reference_terms_identical)"),
    }
    outs = outcome_summaries(rows_by_arm)
    pairs, pair_summary = paired_contrasts(rows_by_arm)
    acq = acquisition_retention(beh)

    # --- write -------------------------------------------------------------------------
    write_text(out / "artifact_sha256.txt", render_sha_file(entries))
    write_text(out / "source_manifest.json", dumps({"measured_code_sha": MEASURED_SHA,
                                                    "arms": {k: str(v["run"]) for k, v in
                                                             ARMS.items()},
                                                    "sources": entries}))
    write_text(out / "review_precheck.json", dumps({
        "record": "review_precheck", "verdict": None,
        "note": "Validity pre-check only; NOT a verdict. Review order: experiments.md 4.1.",
        "arms": pre, "cross_arm_identity": identity,
        "failed_training_seeds_identical": (pre["A"]["failed_training_seeds"] ==
                                            pre["B"]["failed_training_seeds"]),
        "pre_update_identity": pre_update}))
    ex = out / "extracted"
    write_text(ex / "primary_endpoint.json", dumps(endpoint))
    write_text(ex / "behaviour_by_round.json", dumps(beh))
    write_text(ex / "acquisition_retention.json", dumps(acq))
    write_jsonl(ex / "eval_immediate_fd_wakes.jsonl", fd_by_arm["A"] + fd_by_arm["B"])
    write_jsonl(ex / "eval_outcome_rows.jsonl", rows_by_arm["A"] + rows_by_arm["B"])
    write_text(ex / "outcome_summary_by_round.json", dumps(outs))
    write_jsonl(ex / "paired_world_contrasts.jsonl", pairs)
    write_text(ex / "paired_world_contrast_summary.json", dumps(pair_summary))
    write_text(ex / "rescoring_summary.json", dumps(rescoring))
    write_text(ex / "training_summary.json", dumps(train_by_arm))
    write_jsonl(ex / "fd_selected_ego_credit_rows.jsonl", credit_by_arm["A"] + credit_by_arm["B"])
    write_text(ex / "trajectory_comparison.md", trajectory_markdown(beh, outs))
    if args.copy_run_artifacts:
        for arm, a in ARMS.items():
            tag = "arm_" + arm.lower()
            dst = out / "run_artifacts" / tag
            (dst / "launcher").mkdir(parents=True, exist_ok=True)
            for name, _r, disp in RUN_ARTIFACTS:
                if disp == "copied":
                    shutil.copyfile(a["run"] / name, dst / name)
            side = Path(str(a["run"]) + "__launcher")
            for name, _r, disp in LAUNCHER_ARTIFACTS + LAUNCHER_OPTIONAL:
                if disp == "copied" and (side / name).exists():
                    shutil.copyfile(side / name, dst / "launcher" / name)
    for e in entries:
        if e["group"] not in ("prelaunch", "plan"):
            require(sha256_file(Path(e["absolute_path"])) == e["sha256"],
                    "source changed during extraction: %s", e["absolute_path"])
    print("final macro A", ma, "B", mb, "B-A", endpoint["B_minus_A_macro"])
    print("EVIDENCE CHECK PASSED")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-run-artifacts", action="store_true")
    ap.add_argument("--verify-against")
    args = ap.parse_args(argv)
    try:
        run(args)
    except EvidenceError as exc:
        print("EVIDENCE CHECK FAILED:", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
