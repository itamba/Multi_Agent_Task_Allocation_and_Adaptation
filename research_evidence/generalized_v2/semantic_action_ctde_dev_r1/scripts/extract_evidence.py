"""Deterministic, read-only evidence extraction for the GENERALIZED-V2 semantic-action CTDE
DEVELOPMENT run R1 (measured code SHA 8056266cff89f677911462b29970346bed0a57c1).

STANDARD LIBRARY ONLY. It imports nothing from ``match_aou``, imports no torch, starts no
subprocess, and runs no training, evaluation, replay, checkpoint inference or benchmark
preflight. Source artifacts are opened read-only; their SHA-256 is taken before and after
extraction and the run fails if any byte changed.

Inputs:

* the completed CTDE run directory;
* the semantic actor-only comparator run directory (measured SHA d4e9f37) -- its files are
  verified against the hashes preserved in evidence PR #71;
* a raw-blob copy of the PR #71 evidence package at exact head 0d136fa -- every file is
  verified by its Git blob id (computed here, no Git invocation) against constants recorded
  from ``git ls-tree`` of that head;
* the archived historical CTDE R1 (measured SHA ae42cb0; node-indexed representation) --
  secondary context only, verified against the archive index and its evidence commit b2bbe7a;
* the frozen benchmark manifest and the archive index (hash references).

Every reproduced evaluation quantity is cross-checked against the trainer's own persisted
values (per-round ``v2_behaviour`` in ``eval_records.jsonl``, ``run_summary.json``,
``train_records.jsonl``). Every CTDE credit row's GAE arithmetic is reconstructed from the
persisted ``value_old`` / ``transition_reward`` sequence. Any schema, provenance, accounting or
arithmetic mismatch raises :class:`EvidenceError` and writes nothing.

Usage::

    python extract_evidence.py --out <package dir> --comparator-evidence-dir <PR #71 blobs>
                               [--copy-run-artifacts] [--verify-against <artifact_sha256.txt>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------- identities
MEASURED_SHA = "8056266cff89f677911462b29970346bed0a57c1"
RUN_ID = "graph_rl_v2_semantic_action_ctde_dev_r1_seed3000000_8056266"
COMPARATOR_RUN_ID = "graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37"
COMPARATOR_SHA = "d4e9f3721e6d151c00be3fe93c3d149df9d31965"
COMPARATOR_EVIDENCE_PR = 71
COMPARATOR_EVIDENCE_HEAD = "0d136fa89286c4bbd9e89dfb6bd0a3326c70b670"
HIST_CTDE_SHA = "ae42cb01677f94868b2873008d87be677e31f0c8"
HIST_CTDE_EVIDENCE_COMMIT = "b2bbe7a6235c3b9255106826cfb268af7e73f72d"
HIST_CTDE_RUN_CONFIG_BLOB = "83d50ce7ec120df3701f9dccfd291e4fc69ea9cf"
HIST_CTDE_ARCHIVE_KEY = "v2_ctde_r1_seed3000000_ae42cb0"
REPRESENTATION = "semantic_k_plus_2_logmeanexp_v1"
MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
MANIFEST_SHA256 = "dd72afc9cc0d2d1fe494ddbebe53734dc36bd5890997125d3e96a2a59641a103"
FROZEN_CTDE = {"critic_lr": 0.0003, "value_coeff": 0.5, "gae_lambda": 0.95}
FROZEN_PPO = {"clip_ratio": 0.2, "entropy_coeff": 0.01, "lr": 0.0003, "n_epochs": 4,
              "gamma": 1.0, "max_grad_norm": 0.5, "adv_norm_eps": 1e-08}

# Git blob ids of the PR #71 evidence package at exact head 0d136fa (from `git ls-tree -r`).
PR71_BLOBS = {
    "README.md": "21fdeed6f262ad15ceea842a1f019b776c5d9825",
    "artifact_sha256.txt": "b5df34ecf7e94ab838ef682999b6fee53274cbed",
    "extracted/behaviour_summary.json": "b662a9ed3bdd14386bfb190ee9be593f30e2b7ad",
    "extracted/collapse_timeline.json": "64a94368dbba85f88be79fba0c6fa744b3cbe254",
    "extracted/comparator_r1_behaviour.json": "e75d178a2ef97483ed9849eb96ae0eb7d370121c",
    "extracted/credit_summary.json": "291438296a04767149464e249b6c5b8d9d5730d7",
    "extracted/eval_immediate_fd_wakes.jsonl": "ee4be8e1ec06a7d2aaaeb5e824d21dee2673310e",
    "extracted/fd_selected_ego_credit_rows.jsonl": "f8c1da21dc79b64ce9ff743b85521341ca981183",
    "review_precheck.json": "e24f213e3f89f85296564a27cddc6033e6ced7c8",
    "run_artifacts/authorized_plan.json": "1e6591d0024bce12f022b30d51f20112b30953c5",
    "run_artifacts/episode_failures.jsonl": "af423246ec8fd459e44769c2542eadebcbb2384d",
    "run_artifacts/eval_records.jsonl": "9d4c7b51dc9e54ab8615dcbf97f7f822203d17bf",
    "run_artifacts/preflight.json": "3416903186fcc2ec04a988ad42f1ab0b48e67c6e",
    "run_artifacts/run_config.json": "d3715460fcba2a03084bf01d2202f5bddbd78807",
    "run_artifacts/run_summary.json": "fc568b59a20a41b732195017af2962f6ebd15207",
    "run_artifacts/train_records.jsonl": "8df094b153eb93be89b86036c212446ecfab1a05",
    "scripts/extract_evidence.py": "472e992b220eddc61259661e4bef16330f6e33ff",
    "source_manifest.json": "9ffa0c980d770f76517409bda3ac7e7966726962",
}
PR71_USED = ["README.md", "artifact_sha256.txt", "extracted/behaviour_summary.json",
             "extracted/credit_summary.json", "extracted/collapse_timeline.json",
             "review_precheck.json", "source_manifest.json", "run_artifacts/run_config.json"]

DEFAULT_RUN = Path(r"C:\Users\Itama\PycharmProjects") / RUN_ID
DEFAULT_AO = Path(r"C:\Users\Itama\PycharmProjects") / COMPARATOR_RUN_ID
DEFAULT_HIST = Path(r"C:\gra\runs\development\v2_ctde_r1_seed3000000_ae42cb0")
DEFAULT_ARCHIVE_INDEX = Path(r"C:\gra\metadata\ARTIFACT_INDEX.jsonl")

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
GAE_TOL = 1e-9

RUN_ARTIFACTS = [
    ("authorized_plan.json", "pre-run authorized bounded plan, written before training", "copied"),
    ("preflight.json", "pre-run technical preflight incl. actor-initialization identity check", "copied"),
    ("pre_update_comparability_observation.json",
     "post-launch read-only observation of pre-update comparability vs actor-only", "copied"),
    ("launch_record.json", "post-launch launcher / process identity record", "copied"),
    ("run_config.json", "trainer-resolved configuration and provenance", "copied"),
    ("run_summary.json", "trainer run summary: accounting, final round, schema observation", "copied"),
    ("train_records.jsonl", "one record per training update incl. critic diagnostics", "copied"),
    ("eval_records.jsonl", "one record per evaluation round, incl. per-round v2_behaviour", "copied"),
    ("episode_failures.jsonl", "failure ledger (taxonomy, failed seeds)", "copied"),
    ("native_exit_code.txt", "launcher-recorded native exit code (corrected prefix redirect)", "copied"),
    ("invocation_start_local.txt", "launcher start/end local timestamps", "copied"),
    ("launch.cmd", "the detached launcher that produced the run (CRLF batch file)",
     "external_only; hash reference"),
    ("training_console.log", "trainer console output (*.log is git-ignored)",
     "external_only; hash reference"),
    ("episode_outcomes.jsonl", "per-episode outcomes incl. wake diagnostics (train + eval)",
     "extracted; eval immediate-FD wakes and training FD-wake P(ABORT) joined"),
    ("train_credit_diagnostics.jsonl", "per-transition CTDE credit of every productive update",
     "extracted; FD-selected-ego immediate-FD rows copied verbatim + structural verification"),
    ("checkpoints/ckpt_iter0374.pt", "final actor+critic checkpoint (not loaded or executed)",
     "external_only; hash reference"),
]
AO_ARTIFACTS = [
    ("run_config.json", "comparator resolved configuration"),
    ("run_summary.json", "comparator accounting and final-round behaviour"),
    ("eval_records.jsonl", "comparator per-round v2_behaviour"),
    ("episode_outcomes.jsonl", "comparator eval immediate-FD wakes (trajectory reproduction), "
                               "pre-update comparability and training seed stream"),
    ("episode_failures.jsonl", "comparator failure ledger (failed-seed comparison)"),
    ("train_records.jsonl", "comparator per-update records (training counts)"),
    ("train_credit_diagnostics.jsonl", "comparator credit artifact (hash reference only)"),
    ("checkpoints/ckpt_iter0374.pt", "comparator final checkpoint (hash reference only)"),
]
HIST_ARTIFACTS = [
    ("run_config.json", "historical CTDE R1 configuration (CTDE hyperparameter source)"),
    ("run_summary.json", "historical CTDE R1 final-round behaviour"),
    ("eval_records.jsonl", "historical CTDE R1 per-round v2_behaviour (legacy aggregate mass)"),
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


def git_blob_id(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


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


def median(xs):
    xs = [float(x) for x in xs]
    return statistics.median(xs) if xs else None


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


def diff_or_none(a, b):
    return None if a is None or b is None else a - b


def signs(xs):
    xs = list(xs)
    return {"negative": sum(1 for x in xs if x < 0), "zero": sum(1 for x in xs if x == 0),
            "positive": sum(1 for x in xs if x > 0), "n": len(xs)}


# ----------------------------------------------------------------------------- hashing
def entry(group, name, path: Path, role, disposition, git_path=None, measured_sha=None):
    require(path.exists(), "missing source artifact %s", path)
    return {"group": group, "name": name, "absolute_path": str(path),
            "bytes": path.stat().st_size, "sha256": sha256_file(path), "role": role,
            "disposition": disposition, "git_path": git_path,
            "measured_code_sha": measured_sha}


def hash_sources(run_dir, ao_dir, c71_dir, hist_dir, manifest_path, archive_index):
    entries = []
    for name, role, disp in RUN_ARTIFACTS:
        entries.append(entry("run", name, run_dir / name, role, disp,
                             ("run_artifacts/" + name) if disp == "copied" else None, MEASURED_SHA))
    for name, role in AO_ARTIFACTS:
        entries.append(entry("comparator_actor_only_run", name, ao_dir / name, role,
                             "external_only; read for comparator reproduction / hash reference",
                             None, COMPARATOR_SHA))
    for rel in sorted(PR71_BLOBS):
        e = entry("comparator_actor_only_evidence_pr71", rel, c71_dir / rel,
                  "PR #71 evidence file (raw Git blob at %s)" % COMPARATOR_EVIDENCE_HEAD,
                  "external_only; read" if rel in PR71_USED else "external_only; blob-verified",
                  None, COMPARATOR_SHA)
        e["git_blob_id"] = git_blob_id(c71_dir / rel)
        e["git_blob_id_expected"] = PR71_BLOBS[rel]
        require(e["git_blob_id"] == PR71_BLOBS[rel], "PR #71 blob mismatch for %s", rel)
        e["absolute_path"] = "<comparator-evidence-dir>/" + rel
        entries.append(e)
    for name, role in HIST_ARTIFACTS:
        entries.append(entry("historical_ctde_r1", name, hist_dir / name, role,
                             "external_only; secondary context", None, HIST_CTDE_SHA))
    entries.append(entry("benchmark", "benchmark_manifest.json", manifest_path,
                         "frozen GENERALIZED-V2 benchmark manifest consumed by both semantic arms",
                         "external_only; hash reference"))
    entries.append(entry("archive", "ARTIFACT_INDEX.jsonl", archive_index,
                         "archive index: independent record of historical CTDE R1 key hashes",
                         "external_only; read for hash verification"))
    return entries


def render_sha_file(entries) -> str:
    lines = ["# sha256  bytes  group  absolute_path",
             "# Source artifacts of the evidence package. Generated by scripts/extract_evidence.py.",
             "# PR #71 files are addressed relative to --comparator-evidence-dir and additionally",
             "# verified by Git blob id against exact head %s." % COMPARATOR_EVIDENCE_HEAD]
    for e in entries:
        lines.append("%s  %d  %s  %s" % (e["sha256"], e["bytes"], e["group"], e["absolute_path"]))
    return "\n".join(lines) + "\n"


def parse_sha_file(path: Path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sha, size, group, abspath = line.split("  ", 3)
        out[(group, abspath)] = (sha, int(size))
    return out


def resolve_entry_path(e, c71_dir: Path) -> Path:
    if e["absolute_path"].startswith("<comparator-evidence-dir>/"):
        return c71_dir / e["absolute_path"].split("/", 1)[1]
    return Path(e["absolute_path"])


# ----------------------------------------------------------------------------- behaviour
def wake_p_abort(wake):
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
               "deterministic_argmax_leaf", "deterministic_argmax_meta_action_name",
               "semantic_probability_per_meta_action", "semantic_entropy_raw",
               "semantic_entropy_normalized", "n_task_nodes", "n_agent_nodes",
               "n_semantic_leaves", "n_valid_semantic_leaves", "n_abort_legal_nodes",
               "n_engage_legal_leaves", "ego_fuel_norm", "reachable_by_ego", "task_distance_norm",
               "n_task_distance_clipped", "fraction_task_distance_clipped", "time_norm",
               "top_two_probability_margin")


def extract_eval_wakes(outcomes_path: Path, label: str):
    rows = []
    members = defaultdict(lambda: defaultdict(set))
    round_updates = {}
    n_eval = 0
    for r in iter_jsonl(outcomes_path):
        if r.get("phase") not in EVAL_PHASES:
            continue
        n_eval += 1
        ro = int(r["eval_round_ordinal"])
        upd = int(r["updates_completed"])
        require(round_updates.setdefault(ro, upd) == upd, "%s: round %d mixes update counts", label, ro)
        bv = r.get("benchmark_v2") or {}
        gk = r["benchmark_group_key"]
        require(isinstance(gk, str) and re.match(r"^A[2-6]-D[02]-w00[01]$", gk),
                "%s: malformed or non-development group key %r", label, gk)
        require(bv.get("profile") == "development", "%s: non-development member %s", label, gk)
        require(int(r.get("benchmark_world_ordinal")) in (0, 1), "%s: world ordinal outside [0,1]", label)
        require(r.get("benchmark_manifest_id") == MANIFEST_ID, "%s: manifest id on outcome", label)
        cell = bv.get("member_cell")
        require(cell in ("clean",) + SEVERITIES, "%s: unknown member cell %r", label, cell)
        require(cell not in members[ro][gk], "%s: duplicate member %s/%s round %d", label, gk, cell, ro)
        require(bv.get("base_cell") == gk[:5], "%s: base cell %r inconsistent with group %s",
                label, bv.get("base_cell"), gk)
        members[ro][gk].add(cell)
        require(r.get("action_representation_id") == REPRESENTATION,
                "%s: outcome representation %r", label, r.get("action_representation_id"))
        require(r.get("schema_version") == 4 and r.get("wake_diagnostics_schema_version") == 2,
                "%s: outcome schema versions", label)
        for w in r.get("wake_decisions") or ():
            require(w.get("action_representation_id") == REPRESENTATION,
                    "%s: mixed wake representation %r", label, w.get("action_representation_id"))
        if cell == "clean":
            continue
        require(r.get("severity") == cell, "%s: severity %r != member cell %r", label, r.get("severity"), cell)
        fd = [w for w in r.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        require(len(fd) == 1, "%s: round %d %s %s has %d immediate-FD wakes", label, ro, gk, cell, len(fd))
        w = fd[0]
        require(w.get("ego_id") == r.get("fd_ego_id"), "%s: FD wake ego != fd_ego_id (%s)", label, gk)
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
            "wake_kind": w["wake_kind"], "action_representation_id": w["action_representation_id"],
            "p_abort": wake_p_abort(w), "p_abort_source": "semantic SELF_PRESERVATION_ABORT leaf",
        }
        for k in WAKE_FIELDS:
            if k in w:
                row["wake_" + k] = w[k]
        rows.append(row)
    rows.sort(key=lambda x: (x["eval_round_ordinal"], x["benchmark_group_key"], x["member_cell"]))
    return rows, members, round_updates, n_eval


def behaviour_by_round(rows, members, round_updates, eval_records, label):
    require(sorted(round_updates) == list(range(N_ROUNDS)), "%s: rounds %s", label, sorted(round_updates))
    by_round = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_round[r["eval_round_ordinal"]][r["benchmark_group_key"]][r["member_cell"]] = r
    ev_by_round = {int(e["eval_round_ordinal"]): e for e in eval_records}
    require(sorted(ev_by_round) == list(range(N_ROUNDS)) and len(eval_records) == N_ROUNDS,
            "%s: eval_records rounds", label)
    out = []
    for ro in range(N_ROUNDS):
        ev = ev_by_round[ro]
        require(int(ev["updates_completed"]) == round_updates[ro] == ro * WINDOW,
                "%s round %d updates %r", label, ro, round_updates[ro])
        require(ev["evaluation_stage"] == ("pre_update" if ro == 0 else "post_update"),
                "%s round %d stage %r", label, ro, ev["evaluation_stage"])
        require(ev.get("benchmark_profile") == "development" and ev.get("benchmark_manifest_id") == MANIFEST_ID,
                "%s round %d benchmark identity", label, ro)
        vb = ev["v2_behaviour"]
        n_members = sum(len(c) for c in members[ro].values())
        require(n_members == N_EVAL_MEMBERS_PER_ROUND, "%s round %d has %d members", label, ro, n_members)
        require(len(members[ro]) == 20, "%s round %d has %d groups", label, ro, len(members[ro]))
        groups, cell_deltas = [], defaultdict(list)
        sw = rev = 0
        for gk in sorted(members[ro]):
            m = by_round[ro][gk]
            complete = members[ro][gk] == {"clean", "mild", "severe"}
            eligible = complete and "mild" in m and "severe" in m
            g = {"group_key": gk, "complete": complete, "metric_eligible": eligible}
            if eligible:
                require(m["mild"]["seed"] == m["severe"]["seed"], "%s r%d %s pair seeds differ", label, ro, gk)
                d = m["severe"]["p_abort"] - m["mild"]["p_abort"]
                ms, ss = m["mild"]["wake_selected_meta_action_name"], m["severe"]["wake_selected_meta_action_name"]
                g.update({"base_cell": m["mild"]["base_cell"], "seed": m["mild"]["seed"],
                          "p_abort_mild": m["mild"]["p_abort"], "p_abort_severe": m["severe"]["p_abort"],
                          "severe_minus_mild": d, "mild_selected": ms, "severe_selected": ss,
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
        require(sorted(vb["by_base_cell"]) == sorted(cells), "%s r%d base-cell set differs", label, ro)
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
        require(vb.get("action_representation_ids_observed") == [REPRESENTATION],
                "%s round %d representations %s", label, ro, vb.get("action_representation_ids_observed"))
        require(vb.get("aggregate_mass_is_not_selected_action_probability") is False,
                "%s round %d aggregate-mass flag", label, ro)
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


def retention_descriptors(rounds):
    upd = [r["updates_completed"] for r in rounds]
    sw = [r["directional_switches"]["count"] for r in rounds]
    rev = [r["reverse_switches"]["count"] for r in rounds]
    mac = [r["severe_minus_mild_macro_over_base_cells"] for r in rounds]
    nz = [u for u, s in zip(upd, sw) if s > 0]
    best_run, cur, best_rounds, cur_rounds = 0, 0, [], []
    for u, s in zip(upd, sw):
        if s > 0:
            cur += 1
            cur_rounds.append(u)
            if cur > best_run:
                best_run, best_rounds = cur, list(cur_rounds)
        else:
            cur, cur_rounds = 0, []
    imax = max(range(len(mac)), key=lambda i: (mac[i], -i))
    smax = max(range(len(sw)), key=lambda i: (sw[i], -i))
    at = dict(zip(upd, sw))
    return {
        "label": "DESCRIPTIVE repeated-measures quantities over the same 20 development worlds; "
                 "no success threshold; not the primary endpoint",
        "first_update_with_any_directional_switch": nz[0] if nz else None,
        "updates_with_nonzero_directional_switches": nz,
        "max_macro": mac[imax], "max_macro_update_first_occurrence": upd[imax],
        "min_macro": min(mac),
        "max_directional_switch_count": sw[smax], "max_switch_update_first_occurrence": upd[smax],
        "longest_consecutive_rounds_with_nonzero_directional_switches": best_run,
        "longest_consecutive_run_updates": best_rounds,
        "directional_switches_at_150": at.get(150), "directional_switches_at_175": at.get(175),
        "separation_present_at_150_persists_at_175": (at.get(150, 0) > 0 and at.get(175, 0) > 0),
        "any_directional_switch_at_final_round": sw[-1] > 0,
        "any_directional_switch_after_175": any(s > 0 for u, s in zip(upd, sw) if u > 175),
        "updates_with_reverse_switches": [u for u, s in zip(upd, rev) if s > 0],
        "final_update": upd[-1], "final_macro": mac[-1], "final_directional_switches": sw[-1],
        "final_reverse_switches": rev[-1],
    }


# ----------------------------------------------------------------------------- pre-update
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
TIMING = re.compile(r"seconds|elapsed|wall|timestamp|_at$|duration")


def _norm(o):
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items() if not TIMING.search(k)}
    if isinstance(o, list):
        return [_norm(v) for v in o]
    if isinstance(o, str) and UUID.match(o):
        return "<uuid>"
    return o


def _diffs(a, b, p=""):
    res = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                res.append(p + "/" + k)
            else:
                res += _diffs(a[k], b[k], p + "/" + k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            res += _diffs(x, y, "%s[%d]" % (p, i))
    elif a != b:
        res.append(p)
    return res


def pre_update_comparability(ctde_outcomes, ao_outcomes, ctde_eval, ao_eval, observation):
    def rows(path):
        out = {}
        for r in iter_jsonl(path):
            if r.get("phase") == "pre_update":
                out[r["episode_tag"]] = r
        return out
    c, a = rows(ctde_outcomes), rows(ao_outcomes)
    require(sorted(c) == sorted(a) and len(c) == N_EVAL_MEMBERS_PER_ROUND, "pre-update tag sets differ")
    n_ident = n_wake_ident = 0
    sel_diff = rew_diff = seed_diff = nwake_diff = 0
    tick_diff_tags, classes = [], defaultdict(int)
    max_abs_tick = max_ep_tick = max_wake_tick = 0
    fd_n = fd_same_tick = fd_same_p = fd_same_sel = fd_event_same = 0
    fd_max_abs_p = 0.0
    for tag in sorted(a):
        x, y = a[tag], c[tag]
        fx = [w for w in x.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        fy = [w for w in y.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        require(len(fx) == len(fy), "pre-update FD wake count differs for tag %s", tag)
        fd_event_same += x.get("fd_event_tick") == y.get("fd_event_tick")
        for w1, w2 in zip(fx, fy):
            fd_n += 1
            fd_same_tick += w1["tick"] == w2["tick"]
            p1, p2 = wake_p_abort(w1), wake_p_abort(w2)
            fd_same_p += p1 == p2
            fd_max_abs_p = max(fd_max_abs_p, abs(p1 - p2))
            fd_same_sel += (w1.get("selected_meta_action_name"), w1.get("selected_node")) == \
                           (w2.get("selected_meta_action_name"), w2.get("selected_node"))
        max_ep_tick = max(max_ep_tick, abs(int(x["ticks"]) - int(y["ticks"])))
        for w1, w2 in zip(x.get("wake_decisions") or (), y.get("wake_decisions") or ()):
            max_wake_tick = max(max_wake_tick, abs(int(w1["tick"]) - int(w2["tick"])))
        seed_diff += x["seed"] != y["seed"]
        rew_diff += x["reward"] != y["reward"]
        sa = [(w["wake_kind"], w.get("selected_meta_action_name"), w.get("selected_node")) for w in x.get("wake_decisions") or ()]
        sb = [(w["wake_kind"], w.get("selected_meta_action_name"), w.get("selected_node")) for w in y.get("wake_decisions") or ()]
        nwake_diff += len(sa) != len(sb)
        sel_diff += sa != sb
        nx, ny = _norm(x), _norm(y)
        d = _diffs(nx, ny)
        n_ident += not d
        n_wake_ident += not _diffs(nx.get("wake_decisions"), ny.get("wake_decisions"))
        if x.get("ticks") != y.get("ticks"):
            tick_diff_tags.append(tag)
            max_abs_tick = max(max_abs_tick, abs(int(x["ticks"]) - int(y["ticks"])))
        for w1, w2 in zip(x.get("wake_decisions") or (), y.get("wake_decisions") or ()):
            max_abs_tick = max(max_abs_tick, abs(int(w1["tick"]) - int(w2["tick"])))
        for p in d:
            classes[re.sub(r"\[\d+\]", "[]", p)] += 1
    ec = [e for e in ctde_eval if e["evaluation_stage"] == "pre_update"][0]
    ea = [e for e in ao_eval if e["evaluation_stage"] == "pre_update"][0]
    eval_diffs = _diffs(_norm(ea), _norm(ec))
    res = {
        "rows": len(a), "identical_normalized_rows": n_ident,
        "identical_normalized_wake_decisions": n_wake_ident,
        "episodes_with_selected_action_sequence_difference": sel_diff,
        "episodes_with_wake_count_difference": nwake_diff,
        "episodes_with_reward_difference": rew_diff, "episodes_with_seed_difference": seed_diff,
        "episodes_with_episode_tick_difference": len(tick_diff_tags),
        "episode_tick_difference_tags": tick_diff_tags,
        "max_abs_tick_difference_episode_or_wake": max_abs_tick,
        "max_abs_episode_length_tick_difference": max_ep_tick,
        "max_abs_wake_tick_difference": max_wake_tick,
        "episodes_with_identical_fd_event_tick": fd_event_same,
        "immediate_fd_wakes": {"n": fd_n, "identical_tick": fd_same_tick,
                               "bit_identical_p_abort": fd_same_p, "max_abs_p_abort_difference": fd_max_abs_p,
                               "identical_selected_action_and_node": fd_same_sel},
        "launch_time_note_correction": "the launch-time observation described episode/wake timing as shifted "
                                       "by exactly +/-1 tick; recomputation shows wake ticks differ by at most "
                                       "%d and episode length by at most %d" % (max_wake_tick, max_ep_tick),
        "difference_paths_count_by_class": dict(sorted(classes.items())),
        "pre_update_eval_record_normalized_differences": len(eval_diffs),
        "pre_update_v2_behaviour_identical": _norm(ea["v2_behaviour"]) == _norm(ec["v2_behaviour"]),
        "normalization": "timing keys dropped; UUID-valued strings replaced; UUIDs inside free-text "
                         "strings not normalized",
    }
    obs = observation["result"]
    res["observation_file_consistent"] = {
        "identical_normalized_rows": obs["identical_normalized_rows"] == n_ident,
        "identical_wake_decisions": obs["identical_wake_decisions"] == n_wake_ident,
        "eval_record_differences": obs["pre_update_eval_record_normalized_differences"] == len(eval_diffs),
        "selected_actions_differ": obs["selected_actions_differ"] == (sel_diff > 0),
        "rewards_differ": obs["rewards_differ"] == (rew_diff > 0),
    }
    require(all(res["observation_file_consistent"].values()),
            "pre_update_comparability_observation.json inconsistent with recomputation: %r",
            res["observation_file_consistent"])
    return res


# ----------------------------------------------------------------------------- seeds
def training_seed_stream(outcomes_path, failures):
    att = {}
    for r in iter_jsonl(outcomes_path):
        if r.get("phase") == "train":
            k = int(r["seed"])
            require(k not in att, "duplicate training seed %d", k)
            att[k] = {"iteration": int(r["iteration"]), "attempt_ordinal": int(r["attempt_ordinal"]),
                      "status": "successful"}
    for f in failures:
        if f.get("phase") == "train":
            k = int(f["seed"])
            require(k not in att, "failed seed %d also successful", k)
            att[k] = {"iteration": int(f["iteration"]), "attempt_ordinal": int(f["attempt_ordinal"]),
                      "status": "failed", "pipeline_stage": f.get("pipeline_stage"),
                      "error_type": f.get("error_type")}
    return att


def compare_seed_streams(c, a):
    cs, as_ = sorted(c), sorted(a)
    common = sorted(set(cs) & set(as_))
    same_assign = sum(1 for s in common if (c[s]["iteration"], c[s]["attempt_ordinal"], c[s]["status"])
                      == (a[s]["iteration"], a[s]["attempt_ordinal"], a[s]["status"]))
    first_diff = None
    for s in sorted(set(cs) | set(as_)):
        if s not in c or s not in a or (c[s]["iteration"], c[s]["attempt_ordinal"], c[s]["status"]) \
                != (a[s]["iteration"], a[s]["attempt_ordinal"], a[s]["status"]):
            first_diff = {"seed": s, "ctde": c.get(s), "actor_only": a.get(s)}
            break
    cf = sorted(s for s in cs if c[s]["status"] == "failed")
    af = sorted(s for s in as_ if a[s]["status"] == "failed")
    return {
        "ctde_attempted_seed_count": len(cs), "actor_only_attempted_seed_count": len(as_),
        "ctde_seed_range": [cs[0], cs[-1]], "actor_only_seed_range": [as_[0], as_[-1]],
        "ctde_seeds_contiguous_from_base": cs == list(range(3000000, 3000000 + len(cs))),
        "attempted_seed_sets_identical": cs == as_,
        "seeds_with_identical_iteration_attempt_and_status": same_assign,
        "first_difference": first_diff,
        "ctde_failed_seeds": cf, "actor_only_failed_seeds": af,
        "failed_seed_sets_identical": cf == af,
        "ctde_failures": [dict(seed=s, **c[s]) for s in cf],
    }


# ----------------------------------------------------------------------------- credit
def credit_extraction(credit_path: Path, train_records, rc, summary, outcomes_path: Path):
    rows = read_jsonl(credit_path)
    tr_by_iter = {int(t["iteration"]): t for t in train_records}
    require(sorted(tr_by_iter) == list(range(N_ITERATIONS)) and len(train_records) == N_ITERATIONS,
            "train_records iterations")
    tc = rc["train_config"]
    require(tc["training_mode"] == "ctde", "training mode %r", tc["training_mode"])
    gamma = tc["ppo"]["gamma"]
    lam = tc["ctde"]["gae_lambda"]
    eps = tc["ppo"]["adv_norm_eps"]
    require(gamma == 1.0 and lam == 0.95, "gamma/lambda %r %r", gamma, lam)

    by_iter = defaultdict(list)
    for i, r in enumerate(rows):
        require(r.get("schema") == "graph_train_credit_diagnostics" and r.get("schema_version") == 1,
                "credit row %d schema", i)
        require(r.get("action_representation_id") == REPRESENTATION, "credit row %d representation", i)
        require(r.get("training_mode") == "ctde", "credit row %d mode %r", i, r.get("training_mode"))
        require(r.get("gamma") == gamma and r.get("adv_norm_eps") == eps and r.get("gae_lambda") == lam,
                "credit row %d gamma/eps/lambda", i)
        require(r.get("return") is None and r.get("actor_only_episode_baseline") is None
                and r.get("ego_chain_ordinal") is None, "credit row %d carries actor-only fields", i)
        for k in ("value_old", "td_residual", "value_target", "transition_reward", "raw_advantage",
                  "normalized_advantage"):
            require(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]), "credit row %d %s", i, k)
        require((r.get("measurement_join") or {}).get("joined") is True, "credit row %d not joined", i)
        by_iter[int(r["iteration"])].append(r)
    require(sorted(by_iter) == list(range(N_ITERATIONS)), "credit rows do not cover iterations 0..374")
    require(len(rows) == summary["total_transitions"] == summary["observed_credit_diagnostics"]["n_rows"],
            "credit rows %d vs summary", len(rows))

    coverage = []
    gae_checked = 0
    max_err = {"td_residual": 0.0, "raw_advantage": 0.0, "value_target": 0.0, "normalized_advantage": 0.0}
    tr_cross = {"value_target_mean_matches": 0, "value_mean_equals_mean_value_old": 0,
                "baseline_equals_episode_reward_mean_recomputable": 0, "baseline_not_recomputable": 0}
    ordering_ok = True
    for it in range(N_ITERATIONS):
        rs = sorted(by_iter[it], key=lambda r: int(r["batch_transition_ordinal"]))
        t = tr_by_iter[it]
        require(int(t["n_transitions"]) == len(rs), "iteration %d: %d credit rows vs n_transitions %r",
                it, len(rs), t["n_transitions"])
        require(int(t["updates_completed_before"]) == it and all(int(r["updates_completed_before"]) == it for r in rs),
                "iteration %d updates_completed_before", it)
        require([int(r["batch_transition_ordinal"]) for r in rs] == list(range(len(rs))),
                "iteration %d batch ordinals", it)
        for k in ("batch_raw_advantage_mean", "batch_raw_advantage_std", "batch_n_transitions",
                  "batch_n_episodes", "batch_n_episodes_with_wakes"):
            require(len({r[k] for r in rs}) == 1, "iteration %d field %s not batch-constant", it, k)
        r0 = rs[0]
        require(r0["batch_n_transitions"] == len(rs), "iteration %d batch_n_transitions", it)
        require(r0["batch_n_episodes"] == int(t["n_episodes"]) and
                r0["batch_n_episodes_with_wakes"] == int(t["episodes_with_wakes"]),
                "iteration %d batch episode counts vs train_records", it)
        require(close(r0["batch_raw_advantage_std"], t["adv_std_raw"], 1e-12), "iteration %d adv_std_raw", it)
        raw = [float(r["raw_advantage"]) for r in rs]
        m = math.fsum(raw) / len(raw)
        sd = math.sqrt(math.fsum((x - m) ** 2 for x in raw) / len(raw))
        require(close(m, r0["batch_raw_advantage_mean"]) and close(sd, r0["batch_raw_advantage_std"]),
                "iteration %d batch moments do not recompute", it)
        for r in rs:
            e = abs(r["normalized_advantage"] - (r["raw_advantage"] - r0["batch_raw_advantage_mean"])
                    / (r0["batch_raw_advantage_std"] + eps))
            max_err["normalized_advantage"] = max(max_err["normalized_advantage"], e)
            require(e <= 1e-7, "iteration %d normalized advantage does not recompute", it)
        # GAE per episode over the global decision sequence
        eps_rows = defaultdict(list)
        for r in rs:
            eps_rows[(r["episode_seed"], r["episode_index"])].append(r)
        require(len(eps_rows) == r0["batch_n_episodes_with_wakes"], "iteration %d wake-episode count", it)
        prev_last = -1
        for key in sorted(eps_rows, key=lambda k: min(int(x["batch_transition_ordinal"]) for x in eps_rows[k])):
            seq = sorted(eps_rows[key], key=lambda r: int(r["episode_decision_ordinal"]))
            require([int(x["episode_decision_ordinal"]) for x in seq] == list(range(len(seq))),
                    "iteration %d episode %s decision ordinals", it, key)
            bto = [int(x["batch_transition_ordinal"]) for x in seq]
            if bto != list(range(bto[0], bto[0] + len(seq))) or bto[0] != prev_last + 1:
                ordering_ok = False
            prev_last = bto[-1]
            ticks = [int(x["tick"]) for x in seq]
            require(ticks == sorted(ticks), "iteration %d episode %s ticks not nondecreasing", it, key)
            R = seq[0]["episode_reward"]
            require(all(x["episode_reward"] == R for x in seq), "episode %s mixes episode_reward", key)
            require(close(math.fsum(x["transition_reward"] for x in seq), R, 1e-12),
                    "episode %s transition rewards do not sum to episode reward", key)
            running = 0.0
            for idx in range(len(seq) - 1, -1, -1):
                x = seq[idx]
                v_next = float(seq[idx + 1]["value_old"]) if idx + 1 < len(seq) else 0.0
                delta = float(x["transition_reward"]) + gamma * v_next - float(x["value_old"])
                running = delta + gamma * lam * running
                target = running + float(x["value_old"])
                for k, v in (("td_residual", delta), ("raw_advantage", running), ("value_target", target)):
                    err = abs(float(x[k]) - v)
                    max_err[k] = max(max_err[k], err)
                    require(err <= GAE_TOL, "iteration %d episode %s ordinal %d %s: persisted %r vs "
                            "reconstructed %r", it, key, idx, k, x[k], v)
                gae_checked += 1
        vt_mean = mean(r["value_target"] for r in rs)
        if close(vt_mean, t.get("value_target_mean"), 1e-9):
            tr_cross["value_target_mean_matches"] += 1
        if close(mean(r["value_old"] for r in rs), t.get("value_mean"), 1e-9):
            tr_cross["value_mean_equals_mean_value_old"] += 1
        if r0["batch_n_episodes"] == r0["batch_n_episodes_with_wakes"]:
            ep_r = [eps_rows[k][0]["episode_reward"] for k in eps_rows]
            require(close(mean(ep_r), t["baseline"], 1e-12), "iteration %d baseline vs episode reward mean", it)
            tr_cross["baseline_equals_episode_reward_mean_recomputable"] += 1
        else:
            tr_cross["baseline_not_recomputable"] += 1
        coverage.append(len(rs))
    require(tr_cross["value_target_mean_matches"] == N_ITERATIONS,
            "train_records.value_target_mean matches only %d/%d batches", tr_cross["value_target_mean_matches"], N_ITERATIONS)

    # structural variation
    def groupvar(keyf):
        g = defaultdict(list)
        for r in rows:
            g[keyf(r)].append(r)
        res = {"n_checked": len(g), "n_with_more_than_one_transition": sum(1 for v in g.values() if len(v) > 1)}
        for f in ("raw_advantage", "normalized_advantage", "value_old", "td_residual", "value_target",
                  "transition_reward"):
            res["n_with_varying_" + f] = sum(1 for v in g.values() if len({x[f] for x in v}) > 1)
        res["n_multi_transition_with_constant_raw_advantage"] = sum(
            1 for v in g.values() if len(v) > 1 and len({x["raw_advantage"] for x in v}) == 1)
        return res
    chains = groupvar(lambda r: (r["iteration"], r["episode_seed"], r["episode_index"], r["ego_id"]))
    episodes = groupvar(lambda r: (r["iteration"], r["episode_seed"], r["episode_index"]))
    n_tr_nonzero = sum(1 for r in rows if r["transition_reward"] != 0.0)
    ep_nonzero_placement = 0
    ep_R_nonzero = 0
    epg = defaultdict(list)
    for r in rows:
        epg[(r["iteration"], r["episode_seed"], r["episode_index"])].append(r)
    for k, v in epg.items():
        R = v[0]["episode_reward"]
        nz = [x for x in v if x["transition_reward"] != 0.0]
        if R != 0.0:
            ep_R_nonzero += 1
            if len(nz) == 1 and nz[0]["episode_decision_ordinal"] == max(x["episode_decision_ordinal"] for x in v):
                ep_nonzero_placement += 1
        else:
            require(not nz, "episode %s R = 0 but nonzero transition reward", k)
    structural = {
        "gamma": gamma, "gae_lambda": lam, "adv_norm_eps": eps,
        "gae_definition_verified": [
            "per episode, rows ordered by episode_decision_ordinal (GLOBAL decision sequence, not per ego)",
            "delta_t = transition_reward_t + gamma * value_old_{t+1} - value_old_t, value_old_N = 0",
            "A_t = delta_t + gamma * gae_lambda * A_{t+1}, A_N = 0; raw_advantage == A_t",
            "value_target == A_t + value_old_t",
            "normalized_advantage == (raw - batch_raw_mean) / (batch_raw_std + adv_norm_eps)",
            "batch_raw_advantage_mean / std recompute from the batch rows (population std)",
            "train_records.value_target_mean == mean(value_target) of the batch"],
        "gae_rows_reconstructed": gae_checked, "gae_tolerance_abs": GAE_TOL,
        "max_abs_reconstruction_error": max_err,
        "decision_sequence_ordering": {
            "episode_decision_ordinals_contiguous": True,
            "ticks_nondecreasing_with_decision_ordinal": True,
            "episodes_occupy_contiguous_batch_ordinal_blocks": ordering_ok},
        "ego_chains": chains, "episodes": episodes,
        "reward_placement": {"n_episodes": len(epg), "n_episodes_with_nonzero_reward": ep_R_nonzero,
                             "n_nonzero_reward_episodes_with_single_final_decision_reward": ep_nonzero_placement,
                             "n_rows_with_nonzero_transition_reward": n_tr_nonzero},
        "train_records_cross_checks": tr_cross,
        "train_records_value_mean_note": "value_mean is compared with mean(value_old) and reported as "
                                         "a count only; it is not required to match",
        "equality_test": "exact float equality of persisted values for 'varying'",
        "interpretation_note": "varying credit within chains/episodes establishes state- and time-"
                               "dependent credit assignment by the critic/GAE path; it does not "
                               "establish causal correctness of that credit",
        "actor_only_reference_pr71": None,  # filled by caller
    }

    # FD-selected-ego immediate-FD rows
    fd_rows = [r for r in rows if r["wake_kind"] == FD_WAKE and r["measurement_join"].get("is_fd_selected_ego") is True]
    fd_other = [r for r in rows if r["wake_kind"] == FD_WAKE and r["measurement_join"].get("is_fd_selected_ego") is not True]
    require(not fd_other, "%d immediate-FD credit rows are not the FD-selected ego", len(fd_other))
    keys = [(r["iteration"], r["episode_seed"], r["episode_index"]) for r in fd_rows]
    require(len(keys) == len(set(keys)), "an episode has more than one FD-selected immediate-FD row")
    require(all(r["measurement_join"]["severity"] in SEVERITIES for r in fd_rows), "FD row severity")
    require(all(r["measurement_join"]["fd_selected_ego_id"] == r["ego_id"] for r in fd_rows), "FD ego id join")
    fdt = summary["fuel_damage_totals"]
    train_events = {"events_applied": fdt["train_fuel_damage_events_applied"], "wakes": fdt["train_fuel_damage_wakes"],
                    "mild_successful": fdt["train_mild_successful"], "severe_successful": fdt["train_severe_successful"]}
    n_sev = {s: sum(1 for r in fd_rows if r["measurement_join"]["severity"] == s) for s in SEVERITIES}
    require(len(fd_rows) == train_events["events_applied"] == train_events["wakes"],
            "FD credit rows %d vs summary train FD events/wakes %r", len(fd_rows), train_events)
    require(n_sev["mild"] == train_events["mild_successful"] and n_sev["severe"] == train_events["severe_successful"],
            "FD credit rows by severity %r vs summary %r", n_sev, train_events)

    train_fd = {}
    for o in iter_jsonl(outcomes_path):
        if o.get("phase") != "train":
            continue
        for w in o.get("wake_decisions") or ():
            require(w.get("action_representation_id") == REPRESENTATION, "training wake representation")
        fd = [w for w in o.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
        if not fd:
            continue
        require(len(fd) == 1, "training outcome with %d immediate-FD wakes", len(fd))
        key = (int(o["iteration"]), int(o["seed"]))
        require(key not in train_fd, "duplicate training outcome %s", key)
        train_fd[key] = (o, fd[0])
    require(len(train_fd) == len(fd_rows), "training FD outcomes %d vs FD credit rows %d", len(train_fd), len(fd_rows))
    joined, out_rows = [], []
    for r in fd_rows:
        k = (int(r["iteration"]), int(r["episode_seed"]))
        require(k in train_fd, "FD credit row %s has no training outcome", k)
        o, w = train_fd[k]
        require(int(o["episode_index"]) == int(r["episode_index"]), "episode index join")
        require(o.get("severity") == r["measurement_join"]["severity"], "severity join")
        require(o.get("fd_ego_id") == r["ego_id"], "fd ego join")
        require(w.get("ego_id") == r["ego_id"] and w.get("tick") == r["tick"], "wake identity join")
        require(w.get("selected_meta_action_name") == r["selected_meta_action_name"]
                and w.get("selected_node") == r["selected_node"], "selected action join")
        require(close(o.get("reward"), r["episode_reward"], 1e-12), "episode reward join")
        p = wake_p_abort(w)
        ep = epg[(r["iteration"], r["episode_seed"], r["episode_index"])]
        n_dec = len(ep)
        row = dict(r)
        row["x_join"] = {
            "source": "episode_outcomes.jsonl phase=train, joined on (iteration, seed)",
            "verified_equal": ["iteration", "seed", "episode_index", "severity", "fd_ego_id", "ego_id",
                               "tick", "selected_meta_action_name", "selected_node", "episode_reward"],
            "p_abort_at_wake": p,
            "deterministic_argmax_meta_action_name": w.get("deterministic_argmax_meta_action_name"),
            "episode_decision_count": n_dec,
            "decisions_after_fd_wake_in_episode": n_dec - 1 - int(r["episode_decision_ordinal"]),
            "fd_event_tick_outcome": o.get("fd_event_tick"),
        }
        joined.append((r, p))
        out_rows.append(row)
    out_rows.sort(key=lambda x: (x["iteration"], x["batch_transition_ordinal"]))
    return rows, out_rows, joined, coverage, structural, train_events


FIELDS = ("raw_advantage", "normalized_advantage", "value_old", "value_target", "td_residual", "episode_reward")


def sev_block(pairs):
    rs = [r for r, _ in pairs]
    b = {"n": len(rs)}
    for f in FIELDS:
        b[f] = desc(r[f] for r in rs)
    b["transition_reward"] = desc(r["transition_reward"] for r in rs)
    b["p_abort_at_wake_mean"] = mean(p for _, p in pairs)
    b["abort_selected"] = rate(sum(r["selected_meta_action_name"] == ABORT for r in rs), len(rs))
    b["episode_decision_ordinal"] = desc(r["episode_decision_ordinal"] for r in rs)
    return b


def abort_gap(pairs):
    a = [r for r, _ in pairs if r["selected_meta_action_name"] == ABORT]
    b = [r for r, _ in pairs if r["selected_meta_action_name"] != ABORT]
    out = {"label": "NON-COUNTERFACTUAL descriptive association (different episodes/worlds/policy states)",
           "n_abort": len(a), "n_not_abort": len(b), "abort_means": {}, "not_abort_means": {},
           "abort_minus_not_abort": {}}
    for f in FIELDS:
        ma, mb = mean(x[f] for x in a), mean(x[f] for x in b)
        out["abort_means"][f], out["not_abort_means"][f] = ma, mb
        out["abort_minus_not_abort"][f] = diff_or_none(ma, mb)
    return out


def same_batch(pairs):
    per = defaultdict(lambda: defaultdict(list))
    for r, p in pairs:
        per[r["iteration"]][r["measurement_join"]["severity"]].append((r, p))
    within = []
    for it in sorted(per):
        d = per[it]
        if d["mild"] and d["severe"]:
            e = {"iteration": it, "n_mild": len(d["mild"]), "n_severe": len(d["severe"])}
            for f in FIELDS[:5]:
                e["severe_minus_mild_" + f] = mean(x[f] for x, _ in d["severe"]) - mean(x[f] for x, _ in d["mild"])
            e["severe_minus_mild_p_abort"] = mean(p for _, p in d["severe"]) - mean(p for _, p in d["mild"])
            within.append(e)
    res = {"label": "matched by update batch (same critic/normalization state), NOT counterfactual "
                    "episode pairs", "n_batches_with_both_severities": len(within)}
    for f in list(FIELDS[:5]) + ["p_abort"]:
        xs = [x["severe_minus_mild_" + f] for x in within]
        res["severe_minus_mild_" + f] = desc(xs)
        res["severe_minus_mild_" + f + "_signs"] = signs(xs)
    return res, within


def credit_summary(fd_joined, coverage, structural, train_events):
    sev = {s: [(r, p) for r, p in fd_joined if r["measurement_join"]["severity"] == s] for s in SEVERITIES}
    sb, within = same_batch(fd_joined)
    windows = []
    for w0 in range(0, N_ITERATIONS, WINDOW):
        inw = [(r, p) for r, p in fd_joined if w0 <= r["iteration"] < w0 + WINDOW]
        e = {"iterations": [w0, w0 + WINDOW - 1]}
        for s in SEVERITIES:
            pairs = [(r, p) for r, p in inw if r["measurement_join"]["severity"] == s]
            e[s] = {"descriptive": sev_block(pairs), "abort_vs_not_non_counterfactual": abort_gap(pairs)}
        sbw, _ = same_batch(inw)
        e["same_update_batch"] = sbw
        e["severe_minus_mild_pooled_means"] = {
            f: diff_or_none(e["severe"]["descriptive"][f]["mean"], e["mild"]["descriptive"][f]["mean"])
            for f in FIELDS}
        windows.append(e)
    return {
        "schema": "semantic_action_ctde_dev_r1_credit_summary", "schema_version": 1,
        "is_scientific_verdict": False,
        "population": "CTDE training credit rows with wake_kind = immediate_fuel_damage AND "
                      "measurement_join.is_fd_selected_ego = true (ALL rows, no sampling)",
        "counts": {"total": len(fd_joined), "mild": len(sev["mild"]), "severe": len(sev["severe"]),
                   "summary_train_fd_events": train_events},
        "coverage": {"n_productive_updates": len(coverage), "rows_per_update_min": min(coverage),
                     "rows_per_update_max": max(coverage), "rows_total": sum(coverage),
                     "every_update_rows_equal_train_records_n_transitions": True},
        "analysis_classes": {
            "descriptive_association": "per-severity summaries pooled over different episodes, worlds and policy/critic states",
            "same_update_batch": "severe minus mild means within the SAME update batch; still different episodes/worlds",
            "abort_vs_not_non_counterfactual": "rows that selected ABORT versus rows that did not, within a severity; NOT a counterfactual action-value contrast"},
        "measurement_tag_note": "severity / FD tags come from measurement_join and are observational only; "
                                "the credit values were computed by the trainer before the join",
        "descriptive_association": {s: sev_block(sev[s]) for s in SEVERITIES},
        "same_update_batch": dict(sb, per_batch=within),
        "abort_vs_not_non_counterfactual": {s: abort_gap(sev[s]) for s in SEVERITIES},
        "windows_25_iterations": windows,
        "structural_credit_verification": structural,
        "p_abort_at_wake_source": "semantic_probability_per_meta_action of the same wake in the training "
                                  "episode outcome (joined on iteration + seed; ego, tick, severity, selected "
                                  "action/node and reward verified equal)",
    }


# ----------------------------------------------------------------------------- precheck
def precheck(entries, rc, plan, pre, summary, train_records, eval_records, failures, ao_rc, ao_summary,
             launch_record, exit_text, console_scan, seedcmp, preupd, manifest_path, credit_rows_n):
    tc = rc["train_config"]
    prov = rc["provenance"]
    git = prov["git"]
    hashes = {e["name"]: e for e in entries if e["group"] == "run"}
    anomalies = []

    require(git["commit"] == MEASURED_SHA, "run commit %s != %s", git["commit"], MEASURED_SHA)
    require(git["branch"] == "main" and git["dirty"] is False and int(git["dirty_path_count"]) == 0, "run recorded dirty tree")
    tr = rc.get("training", {})
    require(tr.get("action_representation_id") == REPRESENTATION, "run_config representation %r", tr.get("action_representation_id"))
    require(tr.get("mode") == "ctde" and tr.get("ctde_enabled") is True and tr.get("ctde") == FROZEN_CTDE,
            "run_config training block %r", tr)
    require(tc["ctde"] == FROZEN_CTDE and tc["ppo"] == FROZEN_PPO, "train_config CTDE/PPO not frozen values")
    require(pre["code"]["measured_code_sha"] == MEASURED_SHA and plan["measured_code_sha"] == MEASURED_SHA, "plan/preflight SHA")
    require(pre["actor_initialization_identity_check"]["actor_state_identical_actor_only_vs_ctde"] is True,
            "preflight actor-init identity not PASS")

    design = plan["training_design_frozen"]
    ap = rc["training"]["attempt_policy"]
    actual = {
        "episode_design": tc["episode_design"], "training_mode": tc["training_mode"],
        "action_representation_id": tr["action_representation_id"],
        "match_aou_backend": tc["match_aou_backend"], "benchmark_profile": tc["benchmark_profile"],
        "base_seed": tc["base_seed"], "n_iterations": tc["n_iterations"],
        "successful_episodes_per_iteration": tc["episodes_per_iteration"],
        "successful_episode_quota_total": ap["successful_episode_quota_total"],
        "generalized_max_attempts_per_iteration": tc["generalized_max_attempts_per_iteration"],
        "max_possible_training_attempts": ap["max_possible_training_attempts"],
        "fuel_damage_mode": tc["fuel_damage_mode"], "fuel_damage_probability": tc["fuel_damage_probability"],
        "fuel_damage_mild_probability": tc["fuel_damage_mild_probability"],
        "eval_every": tc["eval_every"], "checkpoint_every": tc["checkpoint_every"],
        "expected_evaluation_rounds": summary["n_eval_rounds"],
        "early_stopping": tc["early_stopping"], "solver_timeout": None,
        "resume_or_warm_start": False, "actor_only_second_arm": False, "hyperparameter_sweep": False,
        "ppo": tc["ppo"], "ctde": tc["ctde"], "unchanged": design.get("unchanged"),
    }
    timeout_keys = sorted(k for k in tc if "timeout" in k.lower())
    require(not timeout_keys, "unexpected timeout keys in train_config: %s", timeout_keys)
    plan_vs_actual = {}
    for k, v in design.items():
        require(k in actual, "plan key %s has no mapped resolved value", k)
        plan_vs_actual[k] = {"plan": v, "resolved": actual[k], "match": v == actual[k]}
        require(v == actual[k], "plan %s=%r but resolved %r", k, v, actual[k])
    require(tc == pre["resolved_configuration"]["train_config"], "train_config differs from preflight resolved config")
    argv_plan = plan["invocation"]["argv"][3:]
    argv_run = prov["invocation"]["argv"][1:]
    require(argv_plan == argv_run, "run argv differs from authorized plan argv")
    require(rc["config_source"]["resolved_from"] == "cli_defaults" and rc["config_source"]["cli_overrides"] == [],
            "config_source not cli_defaults")
    ao_tc = ao_rc["train_config"]
    ao_diff = sorted(k for k in set(tc) | set(ao_tc) if tc.get(k, "<absent>") != ao_tc.get(k, "<absent>"))
    require(ao_diff == ["output_dir", "training_mode"], "train_config differs from actor-only beyond output_dir/training_mode: %s", ao_diff)
    require(ao_rc["provenance"]["git"]["commit"] == COMPARATOR_SHA, "comparator SHA")
    require(ao_rc["training"]["action_representation_id"] == REPRESENTATION, "comparator representation")
    env_keys = ("python", "platform", "packages")
    env_same = {k: prov[k] == ao_rc["provenance"][k] for k in env_keys}

    bm = rc["episode_design"]["benchmark_manifest"]
    require(bm["manifest_id"] == MANIFEST_ID and Path(bm["absolute_path"]) == manifest_path, "manifest identity/path")
    man_sha = sha256_file(manifest_path)
    require(man_sha == MANIFEST_SHA256 == pre["benchmark_manifest"]["manifest_file_sha256"], "manifest sha256 %s", man_sha)
    require(bm["evaluation_profile"]["profile"] == "development" and bm["evaluation_profile"]["world_ordinals"] == [0, 1],
            "profile not development [0,1]")
    ao_bm = ao_rc["episode_design"]["benchmark_manifest"]
    require(bm["evaluation_profile"] == ao_bm["evaluation_profile"] and bm["seed_list_sha256"] == ao_bm["seed_list_sha256"],
            "development population identity differs from comparator")
    ho = prov["seeds"]["benchmark_evaluation"]
    require(ho["held_out_verified"] is True and int(ho["held_out_overlap_count"]) == 0, "held-out check failed")

    s = summary
    require(s["updates_completed"] == N_ITERATIONS and s["n_iterations"] == N_ITERATIONS, "updates %s", s["updates_completed"])
    require(s["n_productive_iterations"] == N_ITERATIONS, "productive iterations %s", s["n_productive_iterations"])
    require(s["train_episodes_successful"] == 3000, "successful %s", s["train_episodes_successful"])
    require(s["train_episodes_attempted"] == s["train_episodes_successful"] + s["train_episodes_failed"], "attempted != ok + failed")
    require(s["train_episodes_attempted"] <= 4500, "attempts exceed budget")
    require(s["train_iterations_at_full_quota"] == N_ITERATIONS, "not all iterations at full quota")
    require(s["accounting_reconciled"] is True, "accounting not reconciled")
    require(sum(int(t["n_successful"]) for t in train_records) == 3000, "train_records successes")
    require(sum(int(t["n_attempted"]) for t in train_records) == s["train_episodes_attempted"], "train_records attempts")
    require(len(eval_records) == N_ROUNDS and s["n_eval_rounds"] == N_ROUNDS, "eval rounds")
    require(s["eval_episodes_attempted"] == N_ROUNDS * N_EVAL_MEMBERS_PER_ROUND and s["eval_episodes_failed"] == 0, "eval counts")
    require(len(failures) == s["failures_recorded"] == s["train_episodes_failed"], "failure ledger")
    fail_rows = sorted(({"seed": f.get("seed"), "iteration": f.get("iteration"), "phase": f.get("phase"),
                         "pipeline_stage": f.get("pipeline_stage"), "error_type": f.get("error_type"),
                         "error_message_head": (f.get("error_message") or "")[:60]} for f in failures),
                       key=lambda r: r["seed"])
    require(all(r["pipeline_stage"] in {"generation", "setup", "run", "reward"} for r in fail_rows), "failure outside taxonomy")
    es = s["early_stopping"]
    require(es["enabled"] is False and es["triggered"] is False, "early stopping active")
    fin = s["final_eval_selection"]
    require(fin["selected"] is True and fin["identity"]["updates_completed"] == N_ITERATIONS
            and fin["identity"]["eval_round_ordinal"] == N_ROUNDS - 1, "final round selection")
    fb = s["generalized"]["v2_benchmark"]["final_round_behaviour"]
    require(fb["macro_n_base_cells_defined"] == N_BASE_CELLS and fb["macro_undefined_base_cells"] == []
            and fb["n_groups_metric_eligible"] == 20 and fb["n_groups_attempted"] == 20, "final endpoint coverage")
    oas = s["observed_artifact_schema"]
    require(oas["episode_outcome_schema_versions_observed"] == [4] and oas["wake_diagnostics_schema_versions_observed"] == [2]
            and oas["wake_action_representations_observed"] == [REPRESENTATION], "observed artifact schema not uniform")
    ocd = s["observed_credit_diagnostics"]
    require(ocd["schema_versions_observed"] == [1] and ocd["action_representation_ids_observed"] == [REPRESENTATION]
            and ocd["training_modes_observed"] == ["ctde"], "credit schema observation %r", ocd)
    require(ocd["n_rows"] == s["total_transitions"] == credit_rows_n, "credit rows != total transitions")
    require(sum(int(t["n_transitions"]) for t in train_records) == credit_rows_n, "train_records transitions != credit rows")

    # native exit code
    require(exit_text.strip() == "0", "native_exit_code.txt content %r", exit_text)
    require(console_scan["unexplained_traceback_lines"] == 0 and console_scan["crash_lines"] == 0,
            "console log contains unexplained Traceback/CRASH: %r", console_scan)
    ckpt = hashes["checkpoints/ckpt_iter0374.pt"]

    if fb.get("metric") == "severe_minus_mild_aggregate_abort_mass":
        anomalies.append({"id": "legacy_metric_label_in_v2_behaviour",
                          "observed": "v2_behaviour.metric = 'severe_minus_mild_aggregate_abort_mass'",
                          "context": "same block defines P(ABORT) as the semantic ABORT leaf and sets "
                                     "aggregate_mass_is_not_selected_action_probability = false",
                          "handling": "reporting-label only; values preserved unnormalized (same as PR #71)"})
    anomalies.append({"id": "pre_update_run_to_run_timing_nondeterminism",
                      "observed": "%d/60 pre-update episodes differ from actor-only after normalization; "
                                  "max |wake tick difference| %d; max |episode length difference| %d ticks; immediate-FD wakes with "
                                  "identical tick %d/%d; selected-action sequences differ in %d; rewards differ in %d; "
                                  "pre-update eval record normalized differences %d"
                                  % (60 - preupd["identical_normalized_rows"], preupd["max_abs_wake_tick_difference"],
                                     preupd["max_abs_episode_length_tick_difference"],
                                     preupd["immediate_fd_wakes"]["identical_tick"], preupd["immediate_fd_wakes"]["n"],
                                     preupd["episodes_with_selected_action_sequence_difference"],
                                     preupd["episodes_with_reward_difference"],
                                     preupd["pre_update_eval_record_normalized_differences"]),
                      "context": "same difference class appeared between historical actor-only R1 and CTDE R1 at one "
                                 "code SHA (pre_update_comparability_observation.json); actor weights at update 0 proven "
                                 "identical by preflight",
                      "handling": "recorded limitation; per-episode trajectories across arms are NOT bit-identical; "
                                  "aggregate pre-update endpoint identical"})
    anomalies.append({"id": "expected_setup_failures",
                      "observed": "%d training failures (%s)" % (len(fail_rows), sorted({r["error_type"] for r in fail_rows})),
                      "context": "failed seeds identical to semantic actor-only: %s" % seedcmp["failed_seed_sets_identical"],
                      "handling": "expected certified-FD eligibility attrition, accounted and replaced"})
    if not seedcmp["attempted_seed_sets_identical"]:
        anomalies.append({"id": "training_seed_stream_differs", "observed": seedcmp["first_difference"]})

    return {
        "schema": "semantic_action_ctde_dev_r1_review_precheck", "schema_version": 1,
        "is_scientific_verdict": False, "evidence_class": "DEVELOPMENT (not confirmatory)",
        "confirmatory_profile_used": False, "run_id": RUN_ID, "measured_code_sha": git["commit"],
        "repository_state_recorded_by_run": {"branch": git["branch"], "dirty": git["dirty"], "dirty_path_count": git["dirty_path_count"]},
        "authorized_plan": {"sha256": hashes["authorized_plan.json"]["sha256"], "written_at": plan["written_at"],
                            "preflight_sha256": hashes["preflight.json"]["sha256"],
                            "preflight_timestamp": pre["preflight_timestamp"],
                            "run_config_collected_at": prov["collected_at"],
                            "plan_written_before_run_config": plan["written_at"][:19] <= prov["collected_at"],
                            "launch_record_mtime_ns": launch_record.get("mtime_ns")},
        "resolved_config_vs_authorized_plan": plan_vs_actual,
        "ctde_resolved": tc["ctde"], "ppo_resolved": tc["ppo"],
        "action_representation": {"run_config_training": tr["action_representation_id"],
                                  "episode_outcomes_observed": oas["wake_action_representations_observed"],
                                  "credit_rows_observed": ocd["action_representation_ids_observed"]},
        "training_modes_observed_in_credit_rows": ocd["training_modes_observed"],
        "invocation_argv_matches_plan": True, "config_source": rc["config_source"],
        "train_config_keys_differing_from_actor_only": ao_diff,
        "environment_provenance_identical_to_actor_only": env_same,
        "manifest": {"manifest_id": bm["manifest_id"], "file_sha256": man_sha, "path_consumed": bm["absolute_path"],
                     "profile": bm["evaluation_profile"]["profile"], "world_ordinals": bm["evaluation_profile"]["world_ordinals"],
                     "development_group_keys_sha256": bm["evaluation_profile"]["group_keys_sha256"],
                     "development_seed_list_sha256": bm["evaluation_profile"]["seed_list_sha256"],
                     "same_development_population_as_actor_only": True, "held_out_verified": True},
        "actor_initialization_identity_check_recorded_by_preflight": {
            k: pre["actor_initialization_identity_check"][k] for k in (
                "result", "actor_state_identical_actor_only_vs_ctde", "ctde_init_deterministic",
                "post_episode_reseed_draws_identical", "run_one_episode_reseeds_first")},
        "completion": {"native_exit_code": int(exit_text.strip()), "updates_completed": s["updates_completed"],
                       "n_productive_iterations": s["n_productive_iterations"], "train_records_rows": len(train_records),
                       "final_checkpoint": {"path": ckpt["absolute_path"], "sha256": ckpt["sha256"], "bytes": ckpt["bytes"]},
                       "early_stopping": {"enabled": es["enabled"], "termination_reason": es["termination_reason"]},
                       "run_seconds": s.get("run_seconds"), "console_scan": console_scan},
        "training_counts": {"attempted": s["train_episodes_attempted"], "successful": s["train_episodes_successful"],
                            "failed": s["train_episodes_failed"], "replacement_attempts": s["train_replacement_attempts"],
                            "iterations_at_full_quota": s["train_iterations_at_full_quota"],
                            "max_possible_attempts": ap["max_possible_training_attempts"],
                            "zero_wake_episodes": s["train_zero_wake_episodes"],
                            "actor_only_attempted": ao_summary["train_episodes_attempted"],
                            "actor_only_failed": ao_summary["train_episodes_failed"]},
        "evaluation_counts": {"rounds": s["n_eval_rounds"], "attempted": s["eval_episodes_attempted"],
                              "successful": s["eval_episodes_successful"], "failed": s["eval_episodes_failed"]},
        "accounting_reconciled": s["accounting_reconciled"],
        "failures": {"by_phase": s["failures_by_phase"], "by_pipeline_stage": s["failures_by_pipeline_stage"],
                     "by_error_type": s["failures_by_error_type"], "rows": fail_rows},
        "training_seed_comparison_vs_actor_only": seedcmp,
        "final_round_identity": fin["identity"],
        "primary_endpoint_coverage": {k: fb[k] for k in ("n_groups_attempted", "n_groups_complete", "n_groups_metric_eligible",
                                                          "macro_n_base_cells_required", "macro_n_base_cells_defined",
                                                          "macro_undefined_base_cells")},
        "credit_coverage": {"rows": credit_rows_n, "summary_total_transitions": s["total_transitions"],
                            "every_productive_update_rows_equal_n_transitions": True},
        "schema_versions_observed": {"episode_outcome": oas["episode_outcome_schema_versions_observed"],
                                     "wake_diagnostics": oas["wake_diagnostics_schema_versions_observed"],
                                     "credit_diagnostics": ocd["schema_versions_observed"]},
        "pre_update_comparability_recomputed": preupd,
        "comparator": {"run_id": COMPARATOR_RUN_ID, "measured_code_sha": COMPARATOR_SHA,
                       "evidence_pr": COMPARATOR_EVIDENCE_PR, "evidence_head": COMPARATOR_EVIDENCE_HEAD},
        "anomalies": anomalies,
    }


# ----------------------------------------------------------------------------- main
def scan_console(path: Path, failures):
    """Classify every console Traceback by its terminal exception line; reconcile FAILED lines."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    crash = sum(1 for l in lines if "CRASH" in l)
    warn = sum(1 for l in lines if "UserWarning" in l)
    blocks = []
    for n, l in enumerate(lines):
        if l.startswith("Traceback"):
            term = None
            for m in range(n + 1, len(lines)):
                x = lines[m]
                if x and not x[0].isspace():
                    term = x
                    break
            blocks.append((n + 1, term))
    explained = [b for b in blocks if b[1] and "no_fd_eligible_ego" in b[1]]
    failed_lines = [l for l in lines if "] FAILED (seed=" in l]
    failed_seeds = sorted(int(re.search(r"seed=(\d+)", l).group(1)) for l in failed_lines)
    ledger = sorted(int(f["seed"]) for f in failures)
    ledger_types = sorted({(f.get("pipeline_stage"), f.get("error_type")) for f in failures})
    all_expected = ledger_types in ([], [("setup", "FuelDamageError")]) and all(
        "no_fd_eligible_ego" in (f.get("error_message") or "") for f in failures)
    return {"traceback_lines": len(blocks), "crash_lines": crash, "gymnasium_userwarning_lines": warn,
            "traceback_blocks_ending_in_no_fd_eligible_ego": len(explained),
            "unexplained_traceback_lines": len(blocks) - len(explained) if all_expected else len(blocks),
            "unexplained_traceback_line_numbers": [b[0] for b in blocks if b not in explained][:20],
            "console_failed_seeds": failed_seeds, "ledger_failed_seeds": ledger,
            "console_failed_lines_match_ledger": failed_seeds == ledger,
            "classification": "each expected setup failure prints a chained FuelDamageError -> EpisodeAttemptError "
                              "traceback pair ending in no_fd_eligible_ego; all tracebacks are explained only if every "
                              "block ends that way and every ledger failure is that class"}


def pr71_rounds(c71: Path):
    b = read_json(c71 / "extracted/behaviour_summary.json")
    return b


def run(args):
    run_dir, ao_dir, c71, hist_dir = Path(args.run_dir), Path(args.ao_dir), Path(args.comparator_evidence_dir), Path(args.hist_dir)
    out, archive_index = Path(args.out), Path(args.archive_index)
    rc = read_json(run_dir / "run_config.json")
    manifest_path = Path(rc["episode_design"]["benchmark_manifest"]["absolute_path"])

    entries = hash_sources(run_dir, ao_dir, c71, hist_dir, manifest_path, archive_index)
    if args.verify_against:
        expected = parse_sha_file(Path(args.verify_against))
        for e in entries:
            k = (e["group"], e["absolute_path"])
            require(k in expected, "artifact %s absent from expected hashes", k)
            require(expected[k] == (e["sha256"], e["bytes"]), "HASH MISMATCH for %s: expected %s, found %s",
                    k, expected[k], (e["sha256"], e["bytes"]))
        require(len(expected) == len(entries), "expected hash file lists %d artifacts, found %d", len(expected), len(entries))
    before = {(e["group"], e["absolute_path"]): e["sha256"] for e in entries}

    # comparator run files must equal what PR #71 preserved
    pr71_hashes = {}
    for line in (c71 / "artifact_sha256.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            sha, size, group, p = line.split("  ", 3)
            pr71_hashes[p] = (sha, int(size))
    ao_verified = []
    for e in entries:
        if e["group"] == "comparator_actor_only_run":
            require(e["absolute_path"] in pr71_hashes, "comparator file %s not in PR #71 hashes", e["absolute_path"])
            require(pr71_hashes[e["absolute_path"]] == (e["sha256"], e["bytes"]),
                    "comparator file %s differs from PR #71 preserved hash", e["absolute_path"])
            ao_verified.append(e["name"])
    for rel in ("run_artifacts/run_config.json",):
        require(sha256_file(c71 / rel) == sha256_file(ao_dir / "run_config.json"), "PR #71 run_config copy differs")

    # historical CTDE identity from archive index + evidence blob
    arch = None
    for rec in iter_jsonl(archive_index):
        if rec.get("artifact_id") == HIST_CTDE_ARCHIVE_KEY:
            arch = rec
    require(arch is not None and arch["measured_code_sha"] == HIST_CTDE_SHA, "historical CTDE archive entry")
    for e in entries:
        if e["group"] == "historical_ctde_r1":
            require(arch["key_sha256"][e["name"]] == e["sha256"], "historical %s differs from archive index", e["name"])
    require(git_blob_id(hist_dir / "run_config.json") == HIST_CTDE_RUN_CONFIG_BLOB,
            "historical CTDE run_config differs from evidence commit %s blob", HIST_CTDE_EVIDENCE_COMMIT)

    plan = read_json(run_dir / "authorized_plan.json")
    pre = read_json(run_dir / "preflight.json")
    obs = read_json(run_dir / "pre_update_comparability_observation.json")
    launch_record = read_json(run_dir / "launch_record.json")
    summary = read_json(run_dir / "run_summary.json")
    train_records = read_jsonl(run_dir / "train_records.jsonl")
    eval_records = read_jsonl(run_dir / "eval_records.jsonl")
    failures = read_jsonl(run_dir / "episode_failures.jsonl")
    exit_text = (run_dir / "native_exit_code.txt").read_text(encoding="ascii")
    console = scan_console(run_dir / "training_console.log", failures)
    require(console["console_failed_lines_match_ledger"], "console FAILED lines differ from failure ledger")
    ao_rc = read_json(ao_dir / "run_config.json")
    ao_summary = read_json(ao_dir / "run_summary.json")
    ao_eval = read_jsonl(ao_dir / "eval_records.jsonl")
    ao_failures = read_jsonl(ao_dir / "episode_failures.jsonl")
    ao_train = read_jsonl(ao_dir / "train_records.jsonl")
    hist_rc = read_json(hist_dir / "run_config.json")
    hist_summary = read_json(hist_dir / "run_summary.json")
    hist_eval = read_jsonl(hist_dir / "eval_records.jsonl")

    # --- behaviour, CTDE ---
    wakes, members, rupd, n_eval = extract_eval_wakes(run_dir / "episode_outcomes.jsonl", "ctde")
    require(n_eval == summary["eval_episodes_attempted"], "eval outcome rows %d", n_eval)
    require(len(wakes) == N_ROUNDS * 40, "eval immediate-FD wake rows %d != 640", len(wakes))
    rounds = behaviour_by_round(wakes, members, rupd, eval_records, "ctde")
    fb = summary["generalized"]["v2_benchmark"]["final_round_behaviour"]
    final = rounds[-1]
    require(final["updates_completed"] == summary["final_eval_selection"]["identity"]["updates_completed"]
            and final["eval_round_ordinal"] == summary["final_eval_selection"]["identity"]["eval_round_ordinal"],
            "final round identity")
    require(final["macro_n_base_cells_defined"] == N_BASE_CELLS, "final macro not over 10/10 cells")
    require(close(final["severe_minus_mild_macro_over_base_cells"], fb["macro_mean_over_base_cells"]), "final macro differs from run_summary")
    require(final["directional_switches"]["count"] == fb["directional_switch_count"]
            and final["reverse_switches"]["count"] == fb["reverse_switch_count"], "final switches differ from run_summary")

    # --- behaviour, actor-only reproduced from its run artifacts, checked against PR #71 ---
    ao_wakes, ao_members, ao_rupd, ao_n = extract_eval_wakes(ao_dir / "episode_outcomes.jsonl", "actor_only")
    ao_rounds = behaviour_by_round(ao_wakes, ao_members, ao_rupd, ao_eval, "actor_only")
    pr71b = read_json(c71 / "extracted/behaviour_summary.json")
    require(len(pr71b["rounds"]) == N_ROUNDS, "PR #71 rounds")
    for mine, theirs in zip(ao_rounds, pr71b["rounds"]):
        for k in ("updates_completed", "n_groups_metric_eligible", "directional_switches", "reverse_switches"):
            require(mine[k] == theirs[k], "actor-only round %d %s differs from PR #71", mine["eval_round_ordinal"], k)
        for k in ("p_abort_mild_mean", "p_abort_severe_mean", "severe_minus_mild_macro_over_base_cells",
                  "severe_minus_mild_pooled_mean"):
            require(close(mine[k], theirs[k], 1e-12), "actor-only round %d %s differs from PR #71", mine["eval_round_ordinal"], k)
    ao_fb = ao_summary["generalized"]["v2_benchmark"]["final_round_behaviour"]
    require(close(ao_rounds[-1]["severe_minus_mild_macro_over_base_cells"], ao_fb["macro_mean_over_base_cells"]),
            "actor-only final macro differs from its run_summary")
    # same frozen benchmark members: identical (round, group, cell, seed, tag) sets
    mem = lambda ws: sorted((w["eval_round_ordinal"], w["benchmark_group_key"], w["member_cell"], w["seed"], w["episode_tag"]) for w in ws)
    require(mem(wakes) == mem(ao_wakes), "benchmark member identities differ between arms")

    pr71c = read_json(c71 / "extracted/credit_summary.json")

    # --- pre-update comparability, seeds ---
    preupd = pre_update_comparability(run_dir / "episode_outcomes.jsonl", ao_dir / "episode_outcomes.jsonl",
                                      eval_records, ao_eval, obs)
    seedcmp = compare_seed_streams(training_seed_stream(run_dir / "episode_outcomes.jsonl", failures),
                                   training_seed_stream(ao_dir / "episode_outcomes.jsonl", ao_failures))

    # --- credit ---
    credit_rows, fd_rows, fd_joined, coverage, structural, train_events = credit_extraction(
        run_dir / "train_credit_diagnostics.jsonl", train_records, rc, summary, run_dir / "episode_outcomes.jsonl")
    ao_struct = pr71c["structural_credit_verification"]
    structural["actor_only_reference_pr71"] = {
        "source": "PR #71 extracted/credit_summary.json structural_credit_verification (blob-verified)",
        "n_chains_checked": ao_struct["n_chains_checked"],
        "n_chains_with_more_than_one_transition": ao_struct["n_chains_with_more_than_one_transition"],
        "n_chains_with_varying_raw_advantage": ao_struct["n_chains_with_varying_raw_advantage"],
        "n_episodes_checked": ao_struct["n_episodes_checked"],
        "n_episodes_with_more_than_one_transition": ao_struct["n_episodes_with_more_than_one_transition"],
        "n_episodes_with_varying_raw_advantage": ao_struct["n_episodes_with_varying_raw_advantage"],
        "chain_definition_note": "actor-only chains are (iteration, episode, ego) with per-ego chain ordinals; "
                                 "CTDE chains use the same (iteration, episode, ego) key; CTDE GAE runs over the "
                                 "episode's global decision sequence"}
    csum = credit_summary(fd_joined, coverage, structural, train_events)

    pc = precheck(entries, rc, plan, pre, summary, train_records, eval_records, failures, ao_rc, ao_summary,
                  launch_record, exit_text, console, seedcmp, preupd, manifest_path, len(credit_rows))

    compact = lambda rs: [{k: v for k, v in r.items() if k != "groups"} for r in rs]
    behaviour = {
        "schema": "semantic_action_ctde_dev_r1_behaviour_summary", "schema_version": 1,
        "is_scientific_verdict": False, "run_id": RUN_ID, "measured_code_sha": MEASURED_SHA,
        "endpoint": "SEVERE - MILD P(SELF_PRESERVATION_ABORT) at the certified ego's immediate_fuel_damage wake; "
                    "pair within frozen world group, mean per base cell, equal-weight macro over ten base cells",
        "p_abort_definition": "the one semantic SELF_PRESERVATION_ABORT leaf (%s)" % REPRESENTATION,
        "repeated_measures_note": "every round re-measures the SAME 20 frozen development worlds (60 members); "
                                  "rounds are repeated measures, not independent samples",
        "verifications": ["representation uniformly %s on every outcome and wake" % REPRESENTATION,
                          "every MILD / SEVERE member has exactly one immediate-FD wake",
                          "group keys, base cells, profile, world ordinals and manifest id well-formed",
                          "per-round macro, pooled mean, per-cell means, per-group P(ABORT) and selected actions "
                          "equal the trainer's persisted v2_behaviour",
                          "final macro and switch counts equal run_summary final_round_behaviour"],
        "n_rounds": len(rounds),
        "round_identities": [{"eval_round_ordinal": r["eval_round_ordinal"], "evaluation_stage": r["evaluation_stage"],
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
        "retention_descriptors": retention_descriptors(rounds),
    }
    ao_rc_ppo = ao_rc["train_config"]["ppo"]
    comparator = {
        "schema": "semantic_action_ctde_dev_r1_actor_only_comparator", "schema_version": 1,
        "is_scientific_verdict": False,
        "provenance": {"run_id": COMPARATOR_RUN_ID, "measured_code_sha": COMPARATOR_SHA,
                       "evidence_pr": COMPARATOR_EVIDENCE_PR, "evidence_exact_head": COMPARATOR_EVIDENCE_HEAD,
                       "verdict_as_transferred": "APPROVE — VALID DEVELOPMENT MEASUREMENT (GPT, 2026-09-16)",
                       "training_mode": ao_rc["train_config"]["training_mode"],
                       "action_representation": ao_rc["training"]["action_representation_id"],
                       "manifest_id": ao_rc["episode_design"]["benchmark_manifest"]["manifest_id"],
                       "profile": ao_rc["episode_design"]["benchmark_manifest"]["evaluation_profile"]["profile"],
                       "budget": {"n_iterations": ao_rc["train_config"]["n_iterations"],
                                  "episodes_per_iteration": ao_rc["train_config"]["episodes_per_iteration"],
                                  "generalized_max_attempts_per_iteration": ao_rc["train_config"]["generalized_max_attempts_per_iteration"],
                                  "base_seed": ao_rc["train_config"]["base_seed"]},
                       "ppo": ao_rc_ppo,
                       "run_files_verified_against_pr71_hashes": sorted(ao_verified),
                       "pr71_files_verified_by_git_blob_id": sorted(PR71_BLOBS)},
        "source": "recomputed from the comparator run's episode_outcomes.jsonl / eval_records.jsonl (hash-verified "
                  "against PR #71) and cross-checked field-by-field against PR #71 extracted/behaviour_summary.json",
        "not_pooled_with_ctde": True,
        "rounds": ao_rounds,
        "retention_descriptors": retention_descriptors(ao_rounds),
        "training_fd_wake_windows_from_pr71": [{
            "iterations": w["iterations"],
            "mild_n": w["mild"]["descriptive"]["n"], "severe_n": w["severe"]["descriptive"]["n"],
            "mild_p_abort_at_wake_mean": w["mild"]["descriptive"]["p_abort_at_wake_mean"],
            "severe_p_abort_at_wake_mean": w["severe"]["descriptive"]["p_abort_at_wake_mean"],
            "mild_abort_selected_rate": w["mild"]["descriptive"]["abort_selected"]["rate"],
            "severe_abort_selected_rate": w["severe"]["descriptive"]["abort_selected"]["rate"],
            "severe_minus_mild_raw_advantage_pooled": diff_or_none(w["severe"]["descriptive"]["raw_advantage"]["mean"],
                                                                   w["mild"]["descriptive"]["raw_advantage"]["mean"]),
            "severe_abort_minus_not_raw_advantage_non_counterfactual": w["severe"]["abort_vs_not_non_counterfactual"]["abort_minus_not_abort_raw_advantage"],
        } for w in pr71c["windows_25_iterations"]],
        "credit_structure_from_pr71": structural["actor_only_reference_pr71"],
    }

    side = []
    for c, a in zip(rounds, ao_rounds):
        require(c["updates_completed"] == a["updates_completed"], "round alignment")
        side.append({
            "eval_round_ordinal": c["eval_round_ordinal"], "updates_completed": c["updates_completed"],
            "role": "PRIMARY ENDPOINT (final round)" if c["updates_completed"] == N_ITERATIONS
                    else "secondary / descriptive repeated measure",
            "actor_only": {"p_abort_mild_mean": a["p_abort_mild_mean"], "p_abort_severe_mean": a["p_abort_severe_mean"],
                           "macro": a["severe_minus_mild_macro_over_base_cells"],
                           "directional_switches": a["directional_switches"]["count"],
                           "reverse_switches": a["reverse_switches"]["count"],
                           "by_base_cell": {k: v["severe_minus_mild_mean"] for k, v in a["by_base_cell"].items()}},
            "ctde": {"p_abort_mild_mean": c["p_abort_mild_mean"], "p_abort_severe_mean": c["p_abort_severe_mean"],
                     "macro": c["severe_minus_mild_macro_over_base_cells"],
                     "directional_switches": c["directional_switches"]["count"],
                     "reverse_switches": c["reverse_switches"]["count"],
                     "by_base_cell": {k: v["severe_minus_mild_mean"] for k, v in c["by_base_cell"].items()}},
            "ctde_minus_actor_only_macro": c["severe_minus_mild_macro_over_base_cells"] - a["severe_minus_mild_macro_over_base_cells"],
            "ctde_minus_actor_only_directional_switches": c["directional_switches"]["count"] - a["directional_switches"]["count"],
            "denominator_groups": {"actor_only": a["n_groups_metric_eligible"], "ctde": c["n_groups_metric_eligible"]},
        })
    vs = {
        "schema": "semantic_action_ctde_dev_r1_ctde_vs_actor_only", "schema_version": 1,
        "is_scientific_verdict": False,
        "primary_endpoint": {"definition": "final round (updates_completed = 375) ten-base-cell equal-weight macro "
                                           "SEVERE - MILD P(SELF_PRESERVATION_ABORT)",
                             "ctde": final["severe_minus_mild_macro_over_base_cells"],
                             "actor_only": ao_rounds[-1]["severe_minus_mild_macro_over_base_cells"],
                             "ctde_minus_actor_only": final["severe_minus_mild_macro_over_base_cells"] - ao_rounds[-1]["severe_minus_mild_macro_over_base_cells"],
                             "ctde_directional_switches": final["directional_switches"]["count"],
                             "actor_only_directional_switches": ao_rounds[-1]["directional_switches"]["count"],
                             "ctde_reverse_switches": final["reverse_switches"]["count"],
                             "actor_only_reverse_switches": ao_rounds[-1]["reverse_switches"]["count"],
                             "base_cells_defined": {"ctde": final["macro_n_base_cells_defined"],
                                                    "actor_only": ao_rounds[-1]["macro_n_base_cells_defined"]},
                             "success_threshold": None},
        "trajectory_role": "secondary / descriptive repeated measures over the same 20 development worlds; "
                           "no best-checkpoint selection",
        "arms_not_pooled": True,
        "only_material_training_difference": "training_mode actor_only -> ctde (centralized critic + GAE credit)",
        "rounds": side,
        "retention_descriptors": {"ctde": behaviour["retention_descriptors"],
                                  "actor_only": comparator["retention_descriptors"]},
    }

    # training-side windows and timeline
    tr_by = {int(t["iteration"]): t for t in train_records}
    ao_tr_by = {int(t["iteration"]): t for t in ao_train}
    timeline_rows = []
    for k, w in enumerate(csum["windows_25_iterations"]):
        its = range(w["iterations"][0], w["iterations"][1] + 1)
        r_before, r_after = rounds[k], rounds[k + 1]
        aw = comparator["training_fd_wake_windows_from_pr71"][k]
        timeline_rows.append({
            "window_iterations": w["iterations"],
            "eval_before_window": {"updates_completed": r_before["updates_completed"],
                                   "macro": r_before["severe_minus_mild_macro_over_base_cells"],
                                   "directional_switches": r_before["directional_switches"]["count"]},
            "eval_after_window": {"updates_completed": r_after["updates_completed"],
                                  "macro": r_after["severe_minus_mild_macro_over_base_cells"],
                                  "directional_switches": r_after["directional_switches"]["count"],
                                  "reverse_switches": r_after["reverse_switches"]["count"],
                                  "p_abort_mild_mean": r_after["p_abort_mild_mean"],
                                  "p_abort_severe_mean": r_after["p_abort_severe_mean"]},
            "actor_only_eval_after_window": {"macro": ao_rounds[k + 1]["severe_minus_mild_macro_over_base_cells"],
                                             "directional_switches": ao_rounds[k + 1]["directional_switches"]["count"]},
            "training_fd_wakes": {
                "mild_n": w["mild"]["descriptive"]["n"], "severe_n": w["severe"]["descriptive"]["n"],
                "mild_p_abort_mean": w["mild"]["descriptive"]["p_abort_at_wake_mean"],
                "severe_p_abort_mean": w["severe"]["descriptive"]["p_abort_at_wake_mean"],
                "mild_abort_selected_rate": w["mild"]["descriptive"]["abort_selected"]["rate"],
                "severe_abort_selected_rate": w["severe"]["descriptive"]["abort_selected"]["rate"]},
            "actor_only_training_fd_wakes": {k2: aw[k2] for k2 in ("mild_p_abort_at_wake_mean", "severe_p_abort_at_wake_mean",
                                                                  "mild_abort_selected_rate", "severe_abort_selected_rate")},
            "credit_severe_minus_mild_pooled": w["severe_minus_mild_pooled_means"],
            "credit_severe_minus_mild_same_batch_means": {
                f: w["same_update_batch"]["severe_minus_mild_" + f]["mean"] for f in FIELDS[:5]},
            "credit_same_batch_n": w["same_update_batch"]["n_batches_with_both_severities"],
            "severe_abort_minus_not_non_counterfactual": {
                f: w["severe"]["abort_vs_not_non_counterfactual"]["abort_minus_not_abort"][f]
                for f in ("raw_advantage", "td_residual", "value_target", "value_old")},
            "mild_abort_minus_not_non_counterfactual": {
                f: w["mild"]["abort_vs_not_non_counterfactual"]["abort_minus_not_abort"][f]
                for f in ("raw_advantage", "td_residual", "value_target", "value_old")},
            "critic_and_update_diagnostics_mean_over_window": {
                f: mean(tr_by[i][f] for i in its) for f in ("value_loss", "value_mean", "value_target_mean",
                                                             "critic_grad_norm", "grad_norm", "approx_kl",
                                                             "clip_fraction", "entropy", "train_reward_mean")},
            "fd_wake_value_old_mean": {s: w[s]["descriptive"]["value_old"]["mean"] for s in SEVERITIES},
            "actor_only_update_diagnostics_mean_over_window": {
                f: mean(ao_tr_by[i][f] for i in its) for f in ("approx_kl", "clip_fraction", "entropy", "train_reward_mean")},
            "actor_only_credit_severe_minus_mild_raw_advantage_pooled": aw["severe_minus_mild_raw_advantage_pooled"],
        })
    timeline = {
        "schema": "semantic_action_ctde_dev_r1_behaviour_credit_timeline", "schema_version": 1,
        "is_scientific_verdict": False,
        "note": "DESCRIPTIVE temporal alignment only, not causal analysis. Window k covers iterations "
                "[25k, 25k+24]; its evaluation round 'after' is at updates_completed = 25(k+1). All 15 windows "
                "and all 16 rounds included; evaluation rounds re-measure the same 20 worlds.",
        "evaluation_round_0": {"updates_completed": 0, "ctde_macro": rounds[0]["severe_minus_mild_macro_over_base_cells"],
                               "actor_only_macro": ao_rounds[0]["severe_minus_mild_macro_over_base_cells"]},
        "windows": timeline_rows,
    }

    hist_rounds = []
    for e in sorted(hist_eval, key=lambda x: int(x["eval_round_ordinal"])):
        vb = e["v2_behaviour"]
        el = [g for g in vb["groups"] if g["metric_eligible"]]
        hist_rounds.append({"eval_round_ordinal": e["eval_round_ordinal"], "updates_completed": e["updates_completed"],
                            "macro_legacy_aggregate_mass": vb["macro_mean_over_base_cells"],
                            "p_abort_mild_mean_legacy": mean(g["p_abort_mild"] for g in el),
                            "p_abort_severe_mean_legacy": mean(g["p_abort_severe"] for g in el),
                            "directional_switches": vb["directional_switch_count"],
                            "reverse_switches": vb["reverse_switch_count"],
                            "n_groups_metric_eligible": vb["n_groups_metric_eligible"]})
    hist = {
        "schema": "semantic_action_ctde_dev_r1_historical_ctde_r1_context", "schema_version": 1,
        "is_scientific_verdict": False,
        "role": "SECONDARY CONTEXT ONLY; NOT the primary comparator (historical node-indexed joint k x 3 action "
                "representation; P(ABORT) is aggregate mass over node-indexed abort cells)",
        "measured_code_sha": hist_rc["provenance"]["git"]["commit"],
        "evidence_commit": HIST_CTDE_EVIDENCE_COMMIT, "archive_key": HIST_CTDE_ARCHIVE_KEY,
        "run_config_git_blob_id": HIST_CTDE_RUN_CONFIG_BLOB,
        "ctde": hist_rc["train_config"]["ctde"], "ppo": hist_rc["train_config"]["ppo"],
        "ctde_and_ppo_equal_this_run": hist_rc["train_config"]["ctde"] == rc["train_config"]["ctde"]
                                       and hist_rc["train_config"]["ppo"] == rc["train_config"]["ppo"],
        "train_config_keys_differing_from_this_run": sorted(k for k in set(hist_rc["train_config"]) | set(rc["train_config"])
                                                            if hist_rc["train_config"].get(k) != rc["train_config"].get(k)),
        "source": "trainer-persisted per-round v2_behaviour of the archived run (hash-verified against the archive index); "
                  "not recomputed from episode outcomes",
        "rounds": hist_rounds,
        "final_round_macro_run_summary": hist_summary["generalized"]["v2_benchmark"]["final_round_behaviour"]["macro_mean_over_base_cells"],
    }
    require(len(hist_rounds) == N_ROUNDS, "historical rounds")

    # every source still byte-identical
    after = {(e["group"], e["absolute_path"]): sha256_file(resolve_entry_path(e, c71)) for e in entries}
    require(after == before, "a source artifact changed during extraction: %s",
            sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)))

    source_manifest = {
        "schema": "semantic_action_ctde_dev_r1_source_manifest", "schema_version": 1,
        "measured_code_sha": MEASURED_SHA, "comparator_code_sha": COMPARATOR_SHA,
        "historical_ctde_code_sha": HIST_CTDE_SHA, "comparator_evidence_head": COMPARATOR_EVIDENCE_HEAD,
        "hash_algorithm": "sha256", "artifacts": entries,
        "sources_unchanged_during_extraction": True, "extraction_invoked_no_scientific_execution": True,
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
    write_text(ex / "actor_only_comparator.json", dumps(comparator))
    write_text(ex / "ctde_vs_actor_only.json", dumps(vs))
    write_jsonl(ex / "fd_selected_ego_credit_rows.jsonl", fd_rows)
    write_text(ex / "credit_summary.json", dumps(csum))
    write_text(ex / "behaviour_credit_timeline.json", dumps(timeline))
    write_text(ex / "historical_ctde_r1_context.json", dumps(hist))

    for p in sorted(out.rglob("*")):
        if p.suffix == ".json":
            read_json(p)
        elif p.suffix == ".jsonl":
            read_jsonl(p)
    print("OK: %d source artifacts verified unchanged; %d eval FD wake rows; %d credit rows; %d FD credit rows "
          "(mild %d / severe %d); GAE rows reconstructed %d; final macro %.12g over %d/10 cells; switches %d/%d"
          % (len(entries), len(wakes), len(credit_rows), len(fd_rows), csum["counts"]["mild"], csum["counts"]["severe"],
             structural["gae_rows_reconstructed"], final["severe_minus_mild_macro_over_base_cells"],
             final["macro_n_base_cells_defined"], final["directional_switches"]["count"], final["reverse_switches"]["count"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--ao-dir", default=str(DEFAULT_AO))
    ap.add_argument("--hist-dir", default=str(DEFAULT_HIST))
    ap.add_argument("--archive-index", default=str(DEFAULT_ARCHIVE_INDEX))
    ap.add_argument("--comparator-evidence-dir", required=True,
                    help="raw Git-blob copy of PR #71's research_evidence/generalized_v2/"
                         "semantic_action_actor_only_dev_r1/ at %s" % COMPARATOR_EVIDENCE_HEAD)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy-run-artifacts", action="store_true")
    ap.add_argument("--verify-against", default=None)
    args = ap.parse_args(argv)
    try:
        run(args)
    except EvidenceError as exc:
        print("EVIDENCE CHECK FAILED: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
