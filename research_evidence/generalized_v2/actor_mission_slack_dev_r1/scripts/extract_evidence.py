"""Deterministic, read-only evidence extraction for the mission-fuel-slack development run.

Run: ``graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441`` (measured code SHA
``3bc944119da08af8e25268c9ee83fc63a8d1e533``), compared cross-version with the reviewed
semantic actor-only R1 ``graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37``.

STANDARD LIBRARY ONLY. It imports nothing from ``match_aou``, starts no subprocess and runs
no training, evaluation, replay, checkpoint inference or benchmark preflight. Every source is
hashed before and after extraction and the run fails if a byte changed. Every reproduced
evaluation quantity is cross-checked against the trainer's own per-round ``v2_behaviour`` in
``eval_records.jsonl`` (both runs). The mission-fuel-slack feature is RE-CHECKED from the
audit captured at each wake (arithmetic, route order, exclusions) -- never recomputed by a
second control path or by replaying the policy.

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
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------- identities
MEASURED_SHA = "3bc944119da08af8e25268c9ee83fc63a8d1e533"
COMPARATOR_SHA = "d4e9f3721e6d151c00be3fe93c3d149df9d31965"
RUN_ID = "graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441"
COMPARATOR_RUN_ID = "graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37"
REPRESENTATION = "semantic_k_plus_2_logmeanexp_v1"
ACTOR_OBSERVATION_ID = "actor_graph_task6_agent2_fuel_norm_mission_fuel_slack_v1"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"
COMPARATOR_ARCHIVE_KEY = "v2_semantic_actor_only_r1_seed3000000_d4e9f37"

DEFAULT_RUN = Path(r"C:\gruns") / RUN_ID
DEFAULT_LAUNCHER = Path(r"C:\gruns") / (RUN_ID + "__launcher")
DEFAULT_COMPARATOR = Path(r"C:\gra\runs\development") / COMPARATOR_ARCHIVE_KEY
DEFAULT_MANIFEST = Path(r"C:\gra\benchmarks\v2_preflight_seed2000000_ae42cb0\benchmark_manifest.json")
DEFAULT_ARCHIVE_INDEX = Path(r"C:\gra\metadata\ARTIFACT_INDEX.jsonl")

ABORT = "SELF_PRESERVATION_ABORT"
FD_WAKE = "immediate_fuel_damage"
BOUNDARY_WAKE = "post_fd_boundary"
ORDINARY_WAKE = "ordinary"
WAKE_KINDS = (ORDINARY_WAKE, FD_WAKE, BOUNDARY_WAKE)
SEVERITIES = ("mild", "severe")
EVAL_PHASES = ("pre_update", "post_update")
PHASES = ("train",) + EVAL_PHASES
N_ROUNDS = 16
N_EVAL_MEMBERS_PER_ROUND = 60
N_BASE_CELLS = 10
N_ITERATIONS = 375
WINDOW = 25
TOL = 1e-9
R_KM = 6371.0088          # the haversine package's mean earth radius (Location.distance_to)

RUN_ARTIFACTS = [
    ("run_config.json", "trainer-resolved configuration and provenance", "copied"),
    ("run_summary.json", "trainer run summary: accounting, final round, schema observation",
     "copied"),
    ("train_records.jsonl", "one record per training update", "copied"),
    ("eval_records.jsonl", "one record per evaluation round, incl. per-round v2_behaviour",
     "copied"),
    ("episode_failures.jsonl", "failure ledger (taxonomy, failed seeds)", "copied"),
    ("episode_outcomes.jsonl", "per-episode outcomes incl. wake diagnostics and mission-slack "
     "audits (train + eval)", "external_only; eval wakes, train FD wakes and summaries "
     "extracted"),
    ("train_credit_diagnostics.jsonl", "per-transition credit of every productive update",
     "external_only; FD-selected-ego immediate-FD rows extracted"),
    ("checkpoints/ckpt_iter0374.pt", "final actor checkpoint (not loaded or executed)",
     "external_only; hash reference"),
]
LAUNCHER_ARTIFACTS = [
    ("launcher_record.json", "launcher / watchdog record: process ids, start / end, walltime, "
     "exit code, termination reason", "copied"),
    ("native_exit_code.txt", "exit code written by the Python launcher", "copied"),
    ("invocation_start_local.txt", "launcher start timestamp", "copied"),
    ("training_console.log", "trainer console output (*.log is git-ignored)",
     "external_only; hash reference; scanned for Traceback / CRASH"),
]
COMPARATOR_ARTIFACTS = [
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


def rel_close(a, b, rel=1e-12):
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(a)), abs(float(b)))


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
    return list(iter_jsonl(path))


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
    xs = sorted(float(x) for x in xs)
    if not xs:
        return {"n": 0, "mean": None, "median": None, "std_pop": None, "min": None,
                "max": None, "p05": None, "p25": None, "p75": None, "p95": None}
    m = math.fsum(xs) / len(xs)

    def q(p):
        k = (len(xs) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    return {"n": len(xs), "mean": m, "median": statistics.median(xs),
            "std_pop": math.sqrt(math.fsum((x - m) ** 2 for x in xs) / len(xs)),
            "min": xs[0], "max": xs[-1], "p05": q(0.05), "p25": q(0.25), "p75": q(0.75),
            "p95": q(0.95)}


def rate(k, n):
    return {"count": k, "denominator": n, "rate": (k / n) if n else None}


def f32(x: float) -> float:
    return struct.unpack("f", struct.pack("f", float(x)))[0]


def hav(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R_KM * math.asin(math.sqrt(h))


# ----------------------------------------------------------------------------- hashing
def hash_sources(run_dir, launcher_dir, comp_dir, manifest, archive_index, prelaunch_dir,
                 package_dir):
    entries = []

    def add(group, name, p, role, disp, git_path=None):
        require(p.exists(), "missing artifact %s", p)
        entries.append({"group": group, "name": name, "absolute_path": str(p),
                        "bytes": p.stat().st_size, "sha256": sha256_file(p), "role": role,
                        "disposition": disp, "git_path": git_path})
    for name, role, disp in RUN_ARTIFACTS:
        add("run", name, run_dir / name, role, disp,
            ("run_artifacts/" + name) if disp == "copied" else None)
    for name, role, disp in LAUNCHER_ARTIFACTS:
        add("launcher", name, launcher_dir / name, role, disp,
            ("run_artifacts/launcher/" + name) if disp == "copied" else None)
    for name, role in COMPARATOR_ARTIFACTS:
        add("comparator", name, comp_dir / name, role,
            "external_only; read for comparator recomputation")
    add("benchmark", "benchmark_manifest.json", manifest,
        "frozen GENERALIZED-V2 benchmark manifest consumed by the run",
        "external_only; hash reference")
    add("archive", "ARTIFACT_INDEX.jsonl", archive_index,
        "archive index: comparator location and key hashes", "external_only")
    for p in sorted(prelaunch_dir.glob("*")):
        add("prelaunch", p.name, p, "pre-launch engineering check / verification record",
            "committed", "prelaunch/" + p.name)
    add("plan", "authorized_plan.json", package_dir / "authorized_plan.json",
        "exact bounded plan, committed before launch", "committed", "authorized_plan.json")
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


# ----------------------------------------------------------------------------- eval wakes
def wake_p_abort(wake):
    v = (wake.get("semantic_probability_per_meta_action") or {}).get(ABORT)
    leaves = [l for l in wake.get("semantic_leaves") or () if l.get("meta_action_name") == ABORT]
    require(len(leaves) == 1, "semantic wake without exactly one ABORT leaf")
    require(close(leaves[0]["probability"], v, 1e-12), "ABORT leaf probability mismatch")
    require(isinstance(v, (int, float)) and not isinstance(v, bool), "P(ABORT) missing")
    return float(v)


def audit_summary(w):
    a = w.get("mission_slack_audit") or {}
    return {
        "ego_fuel_norm": w.get("ego_fuel_norm"),
        "ego_mission_fuel_slack_norm": w.get("ego_mission_fuel_slack_norm"),
        "audit_current_fuel": a.get("current_fuel"), "audit_max_fuel": a.get("max_fuel"),
        "audit_required_fuel": a.get("required_fuel"),
        "audit_route_distance_km": a.get("route_distance_km"),
        "audit_return_leg_km": a.get("return_leg_km"),
        "audit_n_route_stops": a.get("n_route_stops"),
        "audit_n_confirmed": len(a.get("confirmed_target_ids") or ()),
        "audit_n_excluded_confirmed": len(a.get("excluded_confirmed_assignments") or ()),
    }


WAKE_FIELDS = ("tick", "ego_id", "selected_meta_action", "selected_meta_action_name",
               "selected_node", "selected_leaf", "selected_action_probability",
               "deterministic_argmax_meta_action_name", "semantic_probability_per_meta_action",
               "semantic_entropy_raw", "semantic_entropy_normalized", "n_task_nodes",
               "n_agent_nodes", "n_semantic_leaves", "n_valid_semantic_leaves",
               "n_abort_legal_nodes", "n_engage_legal_leaves", "ego_fuel_norm",
               "ego_mission_fuel_slack_norm", "reachable_by_ego", "task_distance_norm",
               "n_task_distance_clipped", "fraction_task_distance_clipped", "time_norm",
               "top_two_probability_margin")


def extract_eval_wakes(outcomes_path: Path, new: bool):
    """Immediate-FD rows (mild/severe members), boundary rows, member sets, round updates."""
    rows, boundary = [], []
    members = defaultdict(lambda: defaultdict(set))
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
        require(r.get("action_representation_id") == REPRESENTATION, "representation")
        if new:
            require(r.get("schema_version") == 5 and r.get("wake_diagnostics_schema_version") == 3,
                    "new-run outcome schema versions")
            require(r.get("actor_observation_id") == ACTOR_OBSERVATION_ID, "actor observation id")
        else:
            require(r.get("schema_version") == 4 and r.get("wake_diagnostics_schema_version") == 2,
                    "comparator outcome schema versions")
        ident = {"evaluation_stage": r["phase"], "eval_round_ordinal": ro, "updates_completed": upd,
                 "benchmark_group_key": gk, "base_cell": bv.get("base_cell"),
                 "benchmark_world_ordinal": r.get("benchmark_world_ordinal"),
                 "member_cell": cell, "severity": r.get("severity"),
                 "condition": r.get("condition"), "seed": r.get("seed"),
                 "episode_tag": r.get("episode_tag"), "fd_ego_id": r.get("fd_ego_id"),
                 "fd_event_tick": r.get("fd_event_tick"),
                 "fd_fuel_after_fraction_of_max": r.get("fd_fuel_after_fraction_of_max"),
                 "episode_reward": r.get("reward"), "episode_ended": r.get("ended")}
        if new:
            for w in r.get("wake_decisions") or ():
                if w.get("wake_kind") == BOUNDARY_WAKE:
                    b = dict(ident)
                    b.update({"wake_" + k: w[k] for k in WAKE_FIELDS if k in w})
                    b["p_abort"] = wake_p_abort(w)
                    b.update(audit_summary(w))
                    boundary.append(b)
        if cell == "clean":
            continue
        require(r.get("severity") == cell, "severity %r != member cell %r", r.get("severity"), cell)
        fd = [w for w in r.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        require(len(fd) == 1, "round %d %s %s has %d immediate-FD wakes", ro, gk, cell, len(fd))
        w = fd[0]
        require(w.get("ego_id") == r.get("fd_ego_id"), "FD wake ego != fd_ego_id (%s)", gk)
        require(w.get("action_representation_id") == REPRESENTATION, "wake representation")
        row = dict(ident)
        row.update({"wake_kind": w["wake_kind"], "p_abort": wake_p_abort(w),
                    "p_abort_source": "semantic SELF_PRESERVATION_ABORT leaf"})
        row.update({"wake_" + k: w[k] for k in WAKE_FIELDS if k in w})
        if new:
            require(w.get("actor_observation_id") == ACTOR_OBSERVATION_ID, "wake observation id")
            row.update(audit_summary(w))
        rows.append(row)
    key = lambda x: (x["eval_round_ordinal"], x["benchmark_group_key"], x["member_cell"])
    rows.sort(key=key)
    boundary.sort(key=lambda x: key(x) + (x.get("wake_tick") or 0,))
    return rows, boundary, members, round_updates, n_eval


def behaviour_by_round(rows, members, round_updates, eval_records, label):
    require(sorted(round_updates) == list(range(N_ROUNDS)), "%s: rounds %s", label,
            sorted(round_updates))
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
        require(n_members == N_EVAL_MEMBERS_PER_ROUND, "%s round %d has %d members", label, ro,
                n_members)
        groups, cell_deltas = [], defaultdict(list)
        sw = rev = 0
        for gk in sorted(members[ro]):
            m = by_round[ro][gk]
            complete = members[ro][gk] == {"clean", "mild", "severe"}
            eligible = complete and "mild" in m and "severe" in m
            g = {"group_key": gk, "complete": complete, "metric_eligible": eligible}
            if eligible:
                d = m["severe"]["p_abort"] - m["mild"]["p_abort"]
                ms = m["mild"]["wake_selected_meta_action_name"]
                ss = m["severe"]["wake_selected_meta_action_name"]
                g.update({"base_cell": m["mild"]["base_cell"], "p_abort_mild": m["mild"]["p_abort"],
                          "p_abort_severe": m["severe"]["p_abort"], "severe_minus_mild": d,
                          "mild_selected": ms, "severe_selected": ss,
                          "directional_switch": ms != ABORT and ss == ABORT,
                          "reverse_switch": ms == ABORT and ss != ABORT})
                if "ego_mission_fuel_slack_norm" in m["mild"]:
                    g.update({"slack_mild": m["mild"]["ego_mission_fuel_slack_norm"],
                              "slack_severe": m["severe"]["ego_mission_fuel_slack_norm"],
                              "fuel_norm_mild": m["mild"]["ego_fuel_norm"],
                              "fuel_norm_severe": m["severe"]["ego_fuel_norm"]})
                sw += g["directional_switch"]
                rev += g["reverse_switch"]
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
                         and by_round[ro][gk]["mild"]["wake_selected_meta_action_name"] == ABORT)
        sev_abort = sum(1 for gk in by_round[ro] if "severe" in by_round[ro][gk]
                        and by_round[ro][gk]["severe"]["wake_selected_meta_action_name"] == ABORT)
        # --- cross-check against the trainer's own per-round record ---
        require(vb["n_groups_metric_eligible"] == len(elig), "%s r%d eligible", label, ro)
        require(vb["directional_switch_count"] == sw and vb["reverse_switch_count"] == rev,
                "%s r%d switch counts differ from trainer", label, ro)
        require(close(vb["pooled_mean_over_groups"], pooled), "%s r%d pooled", label, ro)
        require((vb["macro_mean_over_base_cells"] is None and macro is None)
                or close(vb["macro_mean_over_base_cells"], macro), "%s r%d macro", label, ro)
        for c, cv in cells.items():
            require(close(vb["by_base_cell"][c]["severe_minus_mild_abort_mass_mean"],
                          cv["severe_minus_mild_mean"]), "%s r%d cell %s differs", label, ro, c)
        tg = {g["group_key"]: g for g in vb["groups"]}
        for g in elig:
            t = tg[g["group_key"]]
            require(close(t["p_abort_mild"], g["p_abort_mild"], 1e-12)
                    and close(t["p_abort_severe"], g["p_abort_severe"], 1e-12)
                    and t["mild_selected_meta_action"] == g["mild_selected"]
                    and t["severe_selected_meta_action"] == g["severe_selected"],
                    "%s r%d group %s differs from trainer", label, ro, g["group_key"])
        require(vb.get("action_representation_ids_observed") == [REPRESENTATION],
                "%s round %d representations", label, ro)
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
            "by_base_cell": cells, "groups": groups,
            "trainer_cross_check": "passed (macro, pooled, per-cell, per-group, switch counts)",
        })
    return out


# ----------------------------------------------------------------------------- feature audit
def recheck_audit(w, counters):
    """Independent re-check of ONE wake's mission-slack value from its own audit record."""
    a = w.get("mission_slack_audit")
    if a is None:
        counters["missing_audit"] += 1
        return
    v = w.get("ego_mission_fuel_slack_norm")
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        counters["nonfinite_value"] += 1
        return
    ok = True
    if f32(a["mission_fuel_slack_norm"]) != float(v):
        counters["value_vs_audit_float32"] += 1
        ok = False
    route = a["route"]
    legs = math.fsum(s["leg_km"] for s in route) + a["return_leg_km"]
    if not rel_close(legs, a["route_distance_km"], 1e-9):
        counters["leg_sum"] += 1
        ok = False
    req = a["route_distance_km"] * 1000.0 / 1852.0 / a["speed_knots"] * a["fuel_rate"]
    if not rel_close(req, a["required_fuel"], 1e-12):
        counters["fuel_convention"] += 1
        ok = False
    if not rel_close((a["current_fuel"] - a["required_fuel"]) / a["max_fuel"],
                     a["mission_fuel_slack_norm"], 1e-12):
        counters["slack_formula"] += 1
        ok = False
    confirmed = set(a["confirmed_target_ids"])
    if any(s["target_id"] in confirmed for s in route):
        counters["confirmed_in_route"] += 1
        ok = False
    if any(e["target_id"] not in confirmed for e in a["excluded_confirmed_assignments"]):
        counters["excluded_not_confirmed"] += 1
        ok = False
    if len(route) != len(a["remaining_assignments"]):
        counters["route_vs_remaining"] += 1
        ok = False
    levels = [s["level"] for s in route]
    if levels != sorted(levels):
        counters["level_order"] += 1
        ok = False
    # nearest-neighbour + geometric tie rule, per level, from the chained point
    cur = tuple(a["position"])
    first = hav(cur, tuple(route[0]["target"])) if route else hav(cur, tuple(a["home_base"]))
    if route and not rel_close(first, route[0]["leg_km"], 1e-6):
        counters["first_leg_geometry"] += 1
        ok = False
    for i, s in enumerate(route):
        pending = [t for t in route[i:] if t["level"] == s["level"]]
        best = min(pending, key=lambda t: (hav(cur, tuple(t["target"])), t["target"][0],
                                           t["target"][1], t["target_id"], t["step_idx"]))
        if hav(cur, tuple(best["target"])) < hav(cur, tuple(s["target"])) - 1e-6:
            counters["nearest_neighbour"] += 1
            ok = False
            break
        cur = tuple(s["target"])
    if not rel_close(hav(cur, tuple(a["home_base"])), a["return_leg_km"], 1e-6):
        counters["return_leg_geometry"] += 1
        ok = False
    counters["checked"] += 1
    counters["passed"] += int(ok)


def feature_extraction(outcomes_path: Path, fd_ego_by_episode_ok=True):
    """Distributions by phase x wake kind, audit re-checks, and compact row extracts."""
    values = defaultdict(list)          # (phase, kind) -> slack values
    fuel = defaultdict(list)
    fd_by_sev = defaultdict(list)       # (phase, severity) -> immediate-FD slack
    fd_abort = defaultdict(lambda: [0, 0])
    counters = Counter()
    nonfinite = Counter()
    eval_audits, train_fd_rows = [], []
    transitions = defaultdict(Counter)  # iteration -> wake kind counts (train)
    for r in iter_jsonl(outcomes_path):
        ph = r.get("phase")
        require(ph in PHASES, "unknown phase %r", ph)
        require(r.get("actor_observation_id") == ACTOR_OBSERVATION_ID, "outcome observation id")
        for w in r.get("wake_decisions") or ():
            kind = w.get("wake_kind")
            require(kind in WAKE_KINDS, "unknown wake kind %r", kind)
            require(w.get("actor_observation_id") == ACTOR_OBSERVATION_ID, "wake observation id")
            v = w.get("ego_mission_fuel_slack_norm")
            if not (isinstance(v, (int, float)) and math.isfinite(v)):
                nonfinite[(ph, kind)] += 1
            else:
                values[(ph, kind)].append(float(v))
                fuel[(ph, kind)].append(float(w["ego_fuel_norm"]))
            recheck_audit(w, counters)
            if ph == "train":
                transitions[int(r["iteration"])][kind] += 1
            if kind == FD_WAKE and r.get("severity") in SEVERITIES and \
                    w.get("ego_id") == r.get("fd_ego_id"):
                fd_by_sev[(ph, r["severity"])].append(float(v))
                c = fd_abort[(ph, r["severity"])]
                c[0] += int(w["selected_meta_action_name"] == ABORT)
                c[1] += 1
                if ph == "train":
                    tr = {"iteration": r.get("iteration"), "seed": r.get("seed"),
                          "episode_index": r.get("episode_index"),
                          "severity": r["severity"], "tick": w.get("tick"),
                          "ego_id": w.get("ego_id"),
                          "selected_meta_action_name": w["selected_meta_action_name"],
                          "p_abort": wake_p_abort(w), "episode_reward": r.get("reward")}
                    tr.update(audit_summary(w))
                    train_fd_rows.append(tr)
            if ph in EVAL_PHASES:
                eval_audits.append({
                    "evaluation_stage": ph, "eval_round_ordinal": r.get("eval_round_ordinal"),
                    "updates_completed": r.get("updates_completed"),
                    "benchmark_group_key": r.get("benchmark_group_key"),
                    "member_cell": (r.get("benchmark_v2") or {}).get("member_cell"),
                    "seed": r.get("seed"), "wake_kind": kind, "tick": w.get("tick"),
                    "ego_id": w.get("ego_id"),
                    "selected_meta_action_name": w.get("selected_meta_action_name"),
                    "ego_fuel_norm": w.get("ego_fuel_norm"),
                    "ego_mission_fuel_slack_norm": v,
                    "mission_slack_audit": w.get("mission_slack_audit")})

    def block(d):
        out = {}
        for (ph, k), xs in sorted(d.items()):
            s = desc(xs)
            s.update({"n_negative": sum(1 for x in xs if x < 0),
                      "n_zero": sum(1 for x in xs if x == 0),
                      "n_positive": sum(1 for x in xs if x > 0)})
            out.setdefault(ph, {})[k] = s
        return out
    summary = {
        "note": ("Distributions of the actor input as the encoder received it, by phase and wake "
                 "kind; severity is REPORTING-ONLY (joined from the outcome record, never an "
                 "input). Train rows are a stochastic actor on a sampled population; eval rows "
                 "re-measure 20 frozen worlds every round (repeated measures)."),
        "mission_fuel_slack_norm_by_phase_and_wake_kind": block(values),
        "fuel_norm_by_phase_and_wake_kind": block(fuel),
        "nonfinite_by_phase_and_wake_kind": {"%s/%s" % k: n for k, n in sorted(nonfinite.items())},
        "immediate_fd_slack_by_phase_and_severity": block(fd_by_sev),
        "immediate_fd_selected_abort_by_phase_and_severity": {
            "%s/%s" % k: rate(c[0], c[1]) for k, c in sorted(fd_abort.items())},
        "audit_recheck": dict(sorted(counters.items())),
        "audit_recheck_definition": (
            "per wake: float32(audit slack) == recorded actor input; legs sum to distance; "
            "required fuel == km*1000/1852/knots*rate; slack == (current - required)/max_fuel; "
            "no confirmed target on the route; every excluded assignment confirmed; route "
            "covers every remaining assignment; levels non-decreasing; each stop is a nearest "
            "neighbour of the chained point within its level (independent haversine, 1e-6 km); "
            "first and return legs match the geometry"),
    }
    require(counters["checked"] > 0 and counters["passed"] == counters["checked"],
            "mission-slack audit re-check failed: %s", dict(counters))
    require(not nonfinite, "non-finite actor inputs: %s", dict(nonfinite))
    train_transitions = {str(k): dict(v) for k, v in sorted(transitions.items())}
    return summary, eval_audits, train_fd_rows, train_transitions


# ----------------------------------------------------------------------------- secondary
def secondary_outcomes(outcomes_path: Path):
    by = defaultdict(lambda: defaultdict(list))
    for r in iter_jsonl(outcomes_path):
        if r.get("phase") not in EVAL_PHASES:
            continue
        cell = (r.get("benchmark_v2") or {}).get("member_cell")
        key = (int(r["eval_round_ordinal"]), int(r["updates_completed"]), cell)
        by[key]["reward"].append(float(r["reward"]))
        for f in ("u_achieved", "u_ref", "reward_penalty", "n_dead",
                  "targets_confirmed_unique"):
            if isinstance(r.get(f), (int, float)) and not isinstance(r.get(f), bool):
                by[key][f].append(float(r[f]))
        by[key]["any_death"].append(1.0 if (r.get("n_dead") or 0) > 0 else 0.0)
        if cell in SEVERITIES:
            by[key]["fd_rtb_command_issued"].append(1.0 if r.get("fd_rtb_command_issued") else 0.0)
            by[key]["post_fd_boundary_wakes"].append(float(r.get("post_fd_boundary_wakes") or 0))
    out = []
    for (ro, upd, cell), d in sorted(by.items()):
        out.append({"eval_round_ordinal": ro, "updates_completed": upd, "member_cell": cell,
                    "n_episodes": len(d["reward"]),
                    **{f + "_mean": mean(xs) for f, xs in sorted(d.items())}})
    return out


# ----------------------------------------------------------------------------- credit
def credit_extraction(credit_path: Path, train_records, outcomes_path: Path):
    slack_by_key = {}
    for r in iter_jsonl(outcomes_path):
        if r.get("phase") != "train":
            continue
        for w in r.get("wake_decisions") or ():
            k = (int(r["seed"]), str(w["ego_id"]), int(w["tick"]), w["wake_kind"])
            slack_by_key[k] = (w.get("ego_mission_fuel_slack_norm"), w.get("ego_fuel_norm"))
    per_update = Counter()
    fd_rows, joined, missing = [], 0, 0
    for row in iter_jsonl(credit_path):
        per_update[int(row["iteration"])] += 1
        j = row.get("measurement_join") or {}
        if row["wake_kind"] == FD_WAKE and j.get("is_fd_selected_ego"):
            k = (int(row["episode_seed"]), str(row["ego_id"]), int(row["tick"]), row["wake_kind"])
            s = slack_by_key.get(k)
            r2 = dict(row)
            r2["joined_mission_fuel_slack_norm"] = None if s is None else s[0]
            r2["joined_fuel_norm"] = None if s is None else s[1]
            joined += s is not None
            missing += s is None
            fd_rows.append(r2)
    tr_by_it = {int(t["iteration"]): t for t in train_records}
    cover_ok = all(per_update.get(it, 0) == int(t["n_transitions"])
                   for it, t in tr_by_it.items() if int(t.get("n_epochs_run") or 0) > 0)
    require(cover_ok, "credit rows do not cover every productive update's transitions")
    require(missing == 0, "%d FD credit rows could not be joined to their wake", missing)

    def blk(rows):
        return {"n": len(rows),
                "raw_advantage": desc(r["raw_advantage"] for r in rows),
                "normalized_advantage": desc(r["normalized_advantage"] for r in rows),
                "selected_abort": rate(sum(r["selected_meta_action_name"] == ABORT for r in rows),
                                       len(rows)),
                "mission_fuel_slack_norm": desc(r["joined_mission_fuel_slack_norm"] for r in rows)}
    by_sev = {s: blk([r for r in fd_rows if (r.get("measurement_join") or {}).get("severity") == s])
              for s in SEVERITIES}
    windows = []
    for lo in range(0, N_ITERATIONS, WINDOW):
        ws = [r for r in fd_rows if lo <= int(r["iteration"]) < lo + WINDOW]
        windows.append({"iterations": [lo, lo + WINDOW - 1],
                        **{s: {"n": len([r for r in ws if r["measurement_join"]["severity"] == s]),
                               "selected_abort": rate(
                                   sum(r["selected_meta_action_name"] == ABORT for r in ws
                                       if r["measurement_join"]["severity"] == s),
                                   len([r for r in ws if r["measurement_join"]["severity"] == s]))}
                           for s in SEVERITIES}})
    summary = {"n_credit_rows": sum(per_update.values()),
               "coverage": "every productive update's row count equals train_records.n_transitions",
               "fd_selected_ego_immediate_fd_rows": len(fd_rows),
               "joined_to_wake_feature": joined,
               "by_severity": by_sev, "training_fd_windows": windows,
               "limitation": ("actor-only credit with gamma = 1 and a terminal reward is episode / "
                              "chain-level; ABORT-vs-not and severity gaps are observational "
                              "associations between different episodes, never action values")}
    return fd_rows, summary


# ----------------------------------------------------------------------------- precheck
def precheck(run_dir, launcher_dir, rc, plan, summary, train_records, eval_records, failures,
             comp_rc, comp_failures, launcher, console_path):
    tc = rc["train_config"]
    prov = rc["provenance"]
    git = prov["git"]
    anomalies = []
    require(git["commit"] == MEASURED_SHA, "run commit %s", git["commit"])
    require(git["dirty"] is False and git["dirty_path_count"] == 0, "run not clean")
    require(prov["packages"]["match_aou"]["path"].lower().startswith(r"c:\gms1src"),
            "match_aou not imported from the measured worktree")
    require(rc["training"]["actor_observation_id"] == ACTOR_OBSERVATION_ID, "run observation id")
    require(rc["training"]["action_representation_id"] == REPRESENTATION, "run representation")
    require(rc["config_source"]["resolved_from"] == "cli_defaults", "config source")
    argv = prov["invocation"]["argv"][1:]
    plan_argv = [a for a in plan["execution"]["argv_after_python"][2:]]
    plan_argv = [str(run_dir) if a == "<resolved output dir>" else a for a in plan_argv]
    require(argv == plan_argv, "invocation argv differs from the plan: %s vs %s", argv, plan_argv)
    diffs = {k: (tc.get(k), comp_rc["train_config"].get(k))
             for k in sorted(set(tc) | set(comp_rc["train_config"]))
             if tc.get(k) != comp_rc["train_config"].get(k)}
    require(set(diffs) <= {"output_dir", "actor_gradient_diagnostics"},
            "resolved configuration differs from the comparator beyond output_dir: %s", diffs)
    if "actor_gradient_diagnostics" in diffs:
        anomalies.append("train_config key actor_gradient_diagnostics=false is absent from the "
                         "comparator's run_config (added later by PR #74, default off); not a "
                         "consequential difference")
    for k in ("construction", "difficulty", "episode_design", "derived_split", "base_scenario"):
        require(rc.get(k) == comp_rc.get(k), "run_config block %s differs from comparator", k)
    require(tc["benchmark_manifest"].endswith("benchmark_manifest.json"), "manifest path")
    # completion + accounting
    require(len(train_records) == N_ITERATIONS, "train records %d", len(train_records))
    require(summary["updates_completed"] == N_ITERATIONS, "updates_completed")
    require(summary["accounting_reconciled"] is True, "accounting not reconciled")
    require((run_dir / "checkpoints" / "ckpt_iter0374.pt").exists(), "final checkpoint missing")
    require(summary["train_episodes_successful"] == 3000, "successful training episodes")
    require(summary["train_episodes_attempted"] <= 4500, "attempt budget")
    require(summary["eval_episodes_attempted"] == N_ROUNDS * N_EVAL_MEMBERS_PER_ROUND,
            "eval episodes attempted")
    require(summary["n_eval_rounds"] == N_ROUNDS, "eval rounds")
    fail_types = Counter((f.get("phase"), f.get("pipeline_stage"), f.get("error_type"))
                         for f in failures)
    for (ph, st, et), n in fail_types.items():
        require(ph == "train" and st == "setup" and et == "FuelDamageError",
                "unexpected failure class %s/%s/%s", ph, st, et)
        require(all("no_fd_eligible_ego" in (f.get("error_message") or "") for f in failures),
                "non-eligibility FuelDamageError")
    failed_seeds = sorted(int(f["seed"]) for f in failures)
    comp_seeds = sorted(int(f["seed"]) for f in comp_failures)
    # launcher / process
    require(launcher["status"] == "finished" and launcher["termination_reason"] == "exited",
            "launcher did not finish normally")
    require(launcher["exit_code"] == 0, "training exit code %r", launcher["exit_code"])
    require(launcher["walltime_seconds"] < 24 * 3600, "walltime cap")
    console = console_path.read_text(encoding="utf-8", errors="replace")
    n_tb, n_crash = console.count("Traceback"), console.count("CRASH")
    require(n_crash == 0, "console carries CRASH: %d", n_crash)
    # Every Traceback must belong to an ACCOUNTED ledger failure: the trainer prints each
    # skipped attempt as a two-traceback chain (FuelDamageError -> EpisodeAttemptError).
    chains = console.count('raise EpisodeAttemptError("setup", exc) from exc')
    ends = console.count("EpisodeAttemptError: setup failed: FuelDamageError: "
                         "no_fd_eligible_ego")
    require(n_tb == 2 * len(failures) and chains == len(failures) and ends == len(failures),
            "console tracebacks (%d) are not exactly the %d accounted setup failures",
            n_tb, len(failures))
    fs = summary["final_eval_selection"]
    return {
        "record": "review_precheck", "verdict": None,
        "note": "Validity pre-check only; NOT a verdict. Review order: experiments.md section 4.1.",
        "provenance": {"measured_code_sha": git["commit"], "dirty": git["dirty"],
                       "repo_root": git["repo_root"], "match_aou_path":
                       prov["packages"]["match_aou"]["path"],
                       "blade_path": prov["packages"]["blade"]["path"],
                       "python": prov["python"]["version"], "torch":
                       prov["packages"]["torch"]["version"], "bonmin": prov["solver"]["bonmin"]["available"],
                       "argv_matches_plan": True, "config_source": rc["config_source"],
                       "train_config_diff_vs_comparator": diffs},
        "completion": {"updates_completed": summary["updates_completed"],
                       "train_records": len(train_records),
                       "final_checkpoint": "checkpoints/ckpt_iter0374.pt",
                       "launcher": {k: launcher.get(k) for k in (
                           "started_at", "ended_at", "walltime_seconds", "exit_code",
                           "termination_reason", "launcher_pid", "child_pid")},
                       "run_seconds_harness": summary.get("run_seconds"),
                       "console_traceback_count": n_tb, "console_crash_count": n_crash,
                       "console_tracebacks_classification": (
                           "all belong to the accounted no_fd_eligible_ego setup failures "
                           "(two chained tracebacks per ledger row)")},
        "accounting": {k: summary.get(k) for k in (
            "accounting_reconciled", "train_episodes_attempted", "train_episodes_successful",
            "train_episodes_failed", "train_replacement_attempts", "eval_episodes_attempted",
            "eval_episodes_successful", "eval_episodes_failed", "failures_by_phase",
            "failures_by_pipeline_stage", "failures_by_error_type", "n_eval_rounds")},
        "failed_training_seeds": failed_seeds,
        "comparator_failed_training_seeds": comp_seeds,
        "failed_seeds_equal_comparator": failed_seeds == comp_seeds,
        "final_eval_selection": fs,
        "schemas": summary.get("observed_artifact_schema"),
        "anomalies": anomalies,
    }


# ----------------------------------------------------------------------------- tables
def trajectory_table(new_b, comp_b):
    rows = []
    for a, b in zip(new_b, comp_b):
        require(a["updates_completed"] == b["updates_completed"], "round alignment")
        rows.append({
            "eval_round_ordinal": a["eval_round_ordinal"],
            "updates_completed": a["updates_completed"],
            "new_macro": a["severe_minus_mild_macro_over_base_cells"],
            "new_cells_defined": a["macro_n_base_cells_defined"],
            "new_p_abort_mild": a["p_abort_mild_mean"], "new_p_abort_severe": a["p_abort_severe_mean"],
            "new_directional": a["directional_switches"], "new_reverse": a["reverse_switches"],
            "new_selected_abort_mild": a["selected_abort_mild"],
            "new_selected_abort_severe": a["selected_abort_severe"],
            "comparator_macro": b["severe_minus_mild_macro_over_base_cells"],
            "comparator_cells_defined": b["macro_n_base_cells_defined"],
            "comparator_p_abort_mild": b["p_abort_mild_mean"],
            "comparator_p_abort_severe": b["p_abort_severe_mean"],
            "comparator_directional": b["directional_switches"],
            "comparator_reverse": b["reverse_switches"]})
    return rows


def fmt(x, nd=4):
    return "null" if x is None else ("%+.*f" % (nd, x))


def trajectory_markdown(rows) -> str:
    out = ["# Evaluation trajectory — mission-slack run vs semantic actor-only R1 (comparator)", "",
           "Ten-base-cell macro `P(ABORT | SEVERE) − P(ABORT | MILD)` at the immediate-FD wake of the "
           "certified ego; switches are out of metric-eligible groups (all rounds: 20). Cross-version, "
           "single-training-seed development comparison; every round re-measures the same 20 frozen "
           "worlds (repeated measures). Generated by `scripts/extract_evidence.py`.", "",
           "| round | updates | new macro | new P(A\\|MILD) | new P(A\\|SEVERE) | new dir / rev | "
           "comp macro | comp P(A\\|MILD) | comp P(A\\|SEVERE) | comp dir / rev |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append("| %d | %d | %s | %s | %s | %d / %d of %d | %s | %s | %s | %d / %d of %d |" % (
            r["eval_round_ordinal"], r["updates_completed"], fmt(r["new_macro"]),
            fmt(r["new_p_abort_mild"]), fmt(r["new_p_abort_severe"]),
            r["new_directional"]["count"], r["new_reverse"]["count"],
            r["new_directional"]["denominator"], fmt(r["comparator_macro"]),
            fmt(r["comparator_p_abort_mild"]), fmt(r["comparator_p_abort_severe"]),
            r["comparator_directional"]["count"], r["comparator_reverse"]["count"],
            r["comparator_directional"]["denominator"]))
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- main
def run(args):
    run_dir, launcher_dir = Path(args.run_dir), Path(args.launcher_dir)
    comp_dir, out = Path(args.comparator_dir), Path(args.out)
    manifest, archive_index = Path(args.manifest), Path(args.archive_index)
    prelaunch = out / "prelaunch"
    entries = hash_sources(run_dir, launcher_dir, comp_dir, manifest, archive_index, prelaunch, out)
    by = {(e["group"], e["name"]): e for e in entries}
    require(by[("benchmark", "benchmark_manifest.json")]["sha256"] == MANIFEST_SHA256,
            "manifest hash")
    if args.verify_against:
        ref = parse_sha_file(Path(args.verify_against))
        for e in entries:
            if e["absolute_path"] in ref:
                require(ref[e["absolute_path"]] == (e["sha256"], e["bytes"]),
                        "source changed since the package: %s", e["absolute_path"])
    # archive index -> comparator location
    idx = [json.loads(l) for l in archive_index.read_text(encoding="utf-8").splitlines() if l.strip()]
    row = [r for r in idx if r.get("artifact_id") == COMPARATOR_ARCHIVE_KEY]
    require(len(row) == 1 and Path(row[0]["current_path"]).resolve() == comp_dir.resolve(),
            "comparator archive location")

    rc = read_json(run_dir / "run_config.json")
    summary = read_json(run_dir / "run_summary.json")
    train_records = read_jsonl(run_dir / "train_records.jsonl")
    eval_records = read_jsonl(run_dir / "eval_records.jsonl")
    failures = read_jsonl(run_dir / "episode_failures.jsonl")
    plan = read_json(out / "authorized_plan.json")
    launcher = read_json(launcher_dir / "launcher_record.json")
    comp_rc = read_json(comp_dir / "run_config.json")
    require(comp_rc["provenance"]["git"]["commit"] == COMPARATOR_SHA, "comparator SHA")
    comp_eval = read_jsonl(comp_dir / "eval_records.jsonl")
    comp_failures = read_jsonl(comp_dir / "episode_failures.jsonl")

    pre = precheck(run_dir, launcher_dir, rc, plan, summary, train_records, eval_records,
                   failures, comp_rc, comp_failures, launcher,
                   launcher_dir / "training_console.log")

    rows, boundary, members, rounds, n_eval = extract_eval_wakes(
        run_dir / "episode_outcomes.jsonl", new=True)
    require(n_eval == N_ROUNDS * N_EVAL_MEMBERS_PER_ROUND, "eval outcome count %d", n_eval)
    new_b = behaviour_by_round(rows, members, rounds, eval_records, "new")
    c_rows, _cb, c_members, c_rounds, c_n = extract_eval_wakes(
        comp_dir / "episode_outcomes.jsonl", new=False)
    comp_b = behaviour_by_round(c_rows, c_members, c_rounds, comp_eval, "comparator")
    final = new_b[-1]
    fs = summary["final_eval_selection"]
    require(final["updates_completed"] == N_ITERATIONS and
            final["evaluation_stage"] == "post_update", "final round identity")
    require(final["macro_n_base_cells_defined"] == N_BASE_CELLS, "final macro undefined")

    feat, eval_audits, train_fd_rows, train_transitions = feature_extraction(
        run_dir / "episode_outcomes.jsonl")
    secondary = secondary_outcomes(run_dir / "episode_outcomes.jsonl")
    comp_secondary = secondary_outcomes(comp_dir / "episode_outcomes.jsonl")
    fd_credit, credit = credit_extraction(run_dir / "train_credit_diagnostics.jsonl",
                                          train_records, run_dir / "episode_outcomes.jsonl")
    table = trajectory_table(new_b, comp_b)
    endpoint = {
        "primary_endpoint": "ten-base-cell macro P(ABORT|SEVERE) - P(ABORT|MILD), final round",
        "final_round": {k: final[k] for k in (
            "eval_round_ordinal", "evaluation_stage", "updates_completed",
            "severe_minus_mild_macro_over_base_cells", "macro_n_base_cells_defined",
            "n_groups_metric_eligible", "p_abort_mild_mean", "p_abort_severe_mean",
            "directional_switches", "reverse_switches", "selected_abort_mild",
            "selected_abort_severe", "by_base_cell")},
        "trainer_final_eval_selection": fs,
        "comparator_final_round": {k: comp_b[-1][k] for k in (
            "severe_minus_mild_macro_over_base_cells", "macro_n_base_cells_defined",
            "n_groups_metric_eligible", "p_abort_mild_mean", "p_abort_severe_mean",
            "directional_switches", "reverse_switches")},
        "exploratory_only": {
            "new_max_macro": max((r["new_macro"] for r in table if r["new_macro"] is not None),
                                 default=None),
            "new_max_directional": max(r["new_directional"]["count"] for r in table),
            "comparator_max_macro": max(r["comparator_macro"] for r in table
                                        if r["comparator_macro"] is not None),
            "comparator_max_directional": max(r["comparator_directional"]["count"] for r in table),
            "note": "peak / first-crossing observations are exploratory and never a replacement "
                    "endpoint"},
    }
    # --- write -----------------------------------------------------------------------
    write_text(out / "artifact_sha256.txt", render_sha_file(entries))
    write_text(out / "source_manifest.json", dumps({"run_id": RUN_ID,
                                                    "measured_code_sha": MEASURED_SHA,
                                                    "sources": entries}))
    write_text(out / "review_precheck.json", dumps(pre))
    ex = out / "extracted"
    write_jsonl(ex / "eval_immediate_fd_wakes.jsonl", rows)
    write_jsonl(ex / "eval_post_fd_boundary_wakes.jsonl", boundary)
    write_text(ex / "behaviour_summary.json", dumps({"run_id": RUN_ID, "rounds": new_b}))
    write_text(ex / "comparator_behaviour.json", dumps({"run_id": COMPARATOR_RUN_ID,
                                                        "measured_code_sha": COMPARATOR_SHA,
                                                        "rounds": comp_b}))
    write_text(ex / "primary_endpoint.json", dumps(endpoint))
    write_text(ex / "trajectory_comparison.json", dumps({"rows": table}))
    write_text(ex / "trajectory_comparison.md", trajectory_markdown(table))
    write_text(ex / "feature_summary.json", dumps(feat))
    write_jsonl(ex / "eval_wake_feature_audits.jsonl", eval_audits)
    write_jsonl(ex / "train_immediate_fd_feature_rows.jsonl", train_fd_rows)
    write_text(ex / "training_wake_counts_by_iteration.json", dumps(train_transitions))
    write_text(ex / "secondary_outcomes.json", dumps({"new": secondary,
                                                      "comparator": comp_secondary}))
    write_jsonl(ex / "fd_selected_ego_credit_rows.jsonl", fd_credit)
    write_text(ex / "credit_summary.json", dumps(credit))
    if args.copy_run_artifacts:
        for name, _r, disp in RUN_ARTIFACTS:
            if disp == "copied":
                (out / "run_artifacts").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(run_dir / name, out / "run_artifacts" / name)
        for name, _r, disp in LAUNCHER_ARTIFACTS:
            if disp == "copied":
                (out / "run_artifacts" / "launcher").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(launcher_dir / name, out / "run_artifacts" / "launcher" / name)
    # sources unchanged during extraction
    for e in entries:
        if e["group"] in ("run", "launcher", "comparator", "benchmark", "archive"):
            require(sha256_file(Path(e["absolute_path"])) == e["sha256"],
                    "source changed during extraction: %s", e["absolute_path"])
    print("final macro", final["severe_minus_mild_macro_over_base_cells"],
          "switches", final["directional_switches"], "comparator final",
          comp_b[-1]["severe_minus_mild_macro_over_base_cells"])
    print("EVIDENCE CHECK PASSED")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--launcher-dir", default=str(DEFAULT_LAUNCHER))
    ap.add_argument("--comparator-dir", default=str(DEFAULT_COMPARATOR))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--archive-index", default=str(DEFAULT_ARCHIVE_INDEX))
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
