"""Read-only extraction of recorded evaluation immediate-FD wakes and matched MILD/SEVERE pairs.

Temporary review package (not a permanent evidence archive). This script:

* parses existing JSON / JSONL run artifacts only, using the Python standard library;
* imports nothing from ``match_aou`` and runs no policy, BLADE, solver, environment,
  training, evaluation, replay or checkpoint code;
* recomputes no policy decision: every per-wake value is copied from the recorded
  ``wake_decisions`` entry; the only derived quantities are the transparent arithmetic
  comparisons in ``pair_record`` and the descriptive statistics in ``round_summary``;
* never coerces a missing value: an absent field stays ``None`` and every derived
  quantity that depends on it is ``None``.

Usage (writes the four outputs next to this file):

    python extract_wake_pairs.py
"""

import hashlib
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = r"C:\Users\Itama\PycharmProjects"

EXPECTED_MEASURED_SHA = "ae42cb01677f94868b2873008d87be677e31f0c8"
EXPECTED_MANIFEST_ID = "ef17a68a1d41b04cf6cb9b4ed92d91f3a687b600376ff1dc7bd5b83b21a46ea8"
EXPECTED_PROFILE = "development"
EXPECTED_ROUNDS = 16  # 1 pre_update + 15 post_update
EXPECTED_GROUPS_PER_ROUND = 20  # development profile: world ordinals 0..1 in each of 10 base cells

ABORT = "SELF_PRESERVATION_ABORT"
PLAN = "PLAN_COMPLIANCE"
ENGAGE = "OPPORTUNISTIC_ENGAGEMENT"
META_NAMES = (PLAN, ENGAGE, ABORT)
IMMEDIATE_FD = "immediate_fuel_damage"
SEVERITIES = ("mild", "severe")

# Preserved SHA-256 identities, copied from the evidence ledgers:
#   PR #61 research_evidence/generalized_v2/actor_only_dev_r1/artifact_sha256.txt @ 1375a881637a9a32721a1630f598adc571422a47
#   PR #62 research_evidence/generalized_v2/ctde_dev_r1/artifact_sha256.txt       @ b2bbe7a6235c3b9255106826cfb268af7e73f72d
#   PR #64 research_evidence/generalized_v2/ctde_overnight_diagnostics/artifact_sha256.txt @ 90516d51beeddacded2b89a321d14291e411f2b0
RUNS = [
    {
        "variant": "actor_only_r1",
        "run_id": "graph_rl_v2_actor_only_dev_r1_seed3000000_ae42cb0",
        "ledger": "PR #61 @ 1375a881637a9a32721a1630f598adc571422a47",
        "expected_training_mode": "actor_only",
        "sha256": {
            "episode_outcomes.jsonl": "b1bcc0647c45932e1ad6c939d0c740100291dc60a6a83371efb6485d8ebb9da3",
            "run_config.json": "8e104fc012eb1b69543b2d9145ceaf40d3bbe722d74404389b3028f7c2935801",
            "run_summary.json": "db97794b3fec4875afb870ea84d8d2fa2338e497891fbafc68def12649faebc9",
            "eval_records.jsonl": "2987bf78539eddfbf1acc054ce12717d737bfc965e09c970af38aed61960d03c",
        },
    },
    {
        "variant": "ctde_r1",
        "run_id": "graph_rl_v2_ctde_dev_r1_seed3000000_ae42cb0",
        "ledger": "PR #62 @ b2bbe7a6235c3b9255106826cfb268af7e73f72d",
        "expected_training_mode": "ctde",
        "sha256": {
            "episode_outcomes.jsonl": "fbb18848eef09f187f9335eaed1aa97d686b1ae4494484e06d333afc27d6327b",
            "run_config.json": "beac06d463639b25ebe4e00b42b7f1a370ea9fe2500c2cfb4d65137e11030b27",
            "run_summary.json": "714fe0cc10189380cce553af8eacd898e28a745bc610842008b013f4b4cfebaf",
            "eval_records.jsonl": "76951246c1a7be7942ab4216421840a6add2dd499f238c276a0bd5728690f1b0",
        },
    },
    {
        "variant": "smallbatch",
        "run_id": "graph_rl_v2_ctde_dev_diag_smallbatch_seed3000000_ae42cb0",
        "ledger": "PR #64 @ 90516d51beeddacded2b89a321d14291e411f2b0",
        "expected_training_mode": "ctde",
        "sha256": {
            "episode_outcomes.jsonl": "b763d588fac327dc68cd4ce1c982aabb3307e19bc62cba55cfe020ff6e59d052",
            "run_config.json": "676e384b0c9984da62fae3b7a097afa8b18e40eceadfe72c745e0edf2ddc3077",
            "run_summary.json": "71c907cf46af1a18d7d5de3810b8e54a053027bc27a20ae2d8a2ccc868b9161d",
            "eval_records.jsonl": "98dc5967ea9768f7da7a52af9dd8e098f4cc7b4e5518755ec5b8931c8acef9a2",
        },
    },
    {
        "variant": "largebatch",
        "run_id": "graph_rl_v2_ctde_dev_diag_largebatch_seed3000000_ae42cb0",
        "ledger": "PR #64 @ 90516d51beeddacded2b89a321d14291e411f2b0",
        "expected_training_mode": "ctde",
        "sha256": {
            "episode_outcomes.jsonl": "0307c03f7a37f8de5ee7be53390c7a631eba3c8323e82c0bd34584179ee68d68",
            "run_config.json": "f36baadccbc07bfcf7d636be942a3f24fd33533c4251e488a6b1998878259149",
            "run_summary.json": "09bdda4342f4d255d354ac915055b86d93e236e1566dae8988082e737724827e",
            "eval_records.jsonl": "997c7b30739aff35ecef277aa147d1b06b2ce2479cc2e90eab5494ea12cc0435",
        },
    },
    {
        "variant": "fd80",
        "run_id": "graph_rl_v2_ctde_dev_diag_fd80_seed3000000_ae42cb0",
        "ledger": "PR #64 @ 90516d51beeddacded2b89a321d14291e411f2b0",
        "expected_training_mode": "ctde",
        "sha256": {
            "episode_outcomes.jsonl": "865bca5c730cccd6420033bd51ee4da87e81ec5443fa3b3b6a20f7a079892d5b",
            "run_config.json": "8323148ba2c74bee417c6ba89bce18103986bc294fe802fdebbfdabefd3b6a6b",
            "run_summary.json": "87b07700296c86031589c3c6b0b61897454472ba8544e2d3fb06f9db083c359c",
            "eval_records.jsonl": "57e7b85bec8b193c56241ca0e71dfe4313c224677d07ea70d069ade648150c03",
        },
    },
]

# Rounds named in the task so a reviewer can locate them (identified, not interpreted).
LANDMARK_UPDATES = {"smallbatch": [200, 250], "largebatch": [30]}

# Episode-level context copied verbatim from the outcome record when the key is present.
EPISODE_CONTEXT_KEYS = (
    "reward", "n_dead", "ended", "ticks", "fd_fired", "fd_event_tick", "fd_damage_factor",
    "fd_fuel_after_fraction_of_max", "fd_wake_occurred", "fd_wake_meta_action_name",
    "fd_rtb_command_issued", "targets_confirmed_unique", "targets_total", "n_wakes",
    "agent_count", "known_requested", "known_realized", "hidden_requested", "hidden_realized",
    "targets_requested", "targets_realized", "hidden_short_realized", "u_achieved", "u_ref",
    "post_fd_deactivation_reason", "post_fd_boundary_wakes",
)


# ---------------------------------------------------------------- source integrity

def file_identity(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    st = os.stat(path)
    return {"sha256": h.hexdigest(), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def source_identities():
    out = {}
    for run in RUNS:
        for name in run["sha256"]:
            out[(run["run_id"], name)] = file_identity(os.path.join(SOURCE_ROOT, run["run_id"], name))
    return out


# ---------------------------------------------------------------- helpers (no coercion)

def num(x):
    """A real number, or None. bool is not a number here."""
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def sub(a, b):
    a, b = num(a), num(b)
    return None if a is None or b is None else a - b


def stats(values):
    vals = [v for v in values if num(v) is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(vals), "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}


def agg_p(wake, name):
    agg = wake.get("aggregate_probability_per_meta_action")
    return num(agg.get(name)) if isinstance(agg, dict) else None


def round_identity(rec):
    """The contract's round identity: (evaluation_stage, updates_completed, eval_round_ordinal, manifest_id)."""
    return (rec.get("evaluation_stage", rec.get("phase")), rec.get("updates_completed"),
            rec.get("eval_round_ordinal"), rec.get("benchmark_manifest_id"))


def identity_complete(ident):
    stage, upd, ordinal, mid = ident
    return (stage in ("pre_update", "post_update") and isinstance(upd, int) and not isinstance(upd, bool)
            and isinstance(ordinal, int) and not isinstance(ordinal, bool) and isinstance(mid, str) and mid)


def ident_dict(ident):
    return dict(zip(("evaluation_stage", "updates_completed", "eval_round_ordinal", "benchmark_manifest_id"), ident))


# ---------------------------------------------------------------- per-run reading

def read_run(run, anomalies):
    rdir = os.path.join(SOURCE_ROOT, run["run_id"])
    cfg = json.load(open(os.path.join(rdir, "run_config.json"), encoding="utf-8"))
    git = cfg.get("provenance", {}).get("git", {})
    measured_sha = git.get("commit")
    training_mode = cfg.get("train_config", {}).get("training_mode")
    if measured_sha != EXPECTED_MEASURED_SHA or git.get("dirty") is not False:
        anomalies.append({"run_id": run["run_id"], "kind": "measured_sha_unexpected",
                          "commit": measured_sha, "dirty": git.get("dirty")})
    if training_mode != run["expected_training_mode"]:
        anomalies.append({"run_id": run["run_id"], "kind": "training_mode_unexpected", "value": training_mode})
    cfg_profile = cfg.get("train_config", {}).get("benchmark_profile")
    cfg_manifest = cfg.get("provenance", {}).get("seeds", {}).get("benchmark_evaluation", {}).get("manifest_id")
    if cfg_profile != EXPECTED_PROFILE or cfg_manifest != EXPECTED_MANIFEST_ID:
        anomalies.append({"run_id": run["run_id"], "kind": "config_manifest_or_profile_unexpected",
                          "profile": cfg_profile, "manifest_id": cfg_manifest})

    eval_rounds = []
    with open(os.path.join(rdir, "eval_records.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_rounds.append(json.loads(line))

    # Stream the outcome file; keep only evaluation-phase members (training rows are skipped).
    members = []
    with open(os.path.join(rdir, "episode_outcomes.jsonl"), encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("phase") not in ("pre_update", "post_update"):
                continue
            rec["_line"] = line_no
            members.append(rec)
    return {"measured_sha": measured_sha, "training_mode": training_mode, "eval_rounds": eval_rounds,
            "members": members}


def member_condition(rec):
    bv2 = rec.get("benchmark_v2") if isinstance(rec.get("benchmark_v2"), dict) else {}
    return bv2.get("member_cell")


def wake_row(run, info, rec, wake, wake_ordinal, n_fd_wakes):
    bv2 = rec.get("benchmark_v2") if isinstance(rec.get("benchmark_v2"), dict) else {}
    ident = round_identity(rec)
    sel = wake.get("selected_meta_action_name")
    return {
        "run_id": run["run_id"],
        "variant": run["variant"],
        "measured_code_sha": info["measured_sha"],
        "training_mode": info["training_mode"],
        "evaluation_stage": ident[0],
        "updates_completed": ident[1],
        "eval_round_ordinal": ident[2],
        "benchmark_manifest_id": ident[3],
        "benchmark_profile": bv2.get("profile"),
        "benchmark_base_cell": bv2.get("base_cell"),
        "benchmark_group_key": rec.get("benchmark_group_key"),
        "benchmark_world_ordinal": rec.get("benchmark_world_ordinal"),
        "member_condition": member_condition(rec),
        "record_cell": rec.get("cell"),
        "record_severity": rec.get("severity"),
        "seed": rec.get("seed"),
        "episode_tag": rec.get("episode_tag"),
        "eval_episode_index": rec.get("eval_episode_index"),
        "reconstructed_identity_verified": bv2.get("reconstructed_identity_verified"),
        "frozen_route_count": bv2.get("frozen_route_count"),
        "source_line": rec["_line"],
        "wake_kind": wake.get("wake_kind"),
        "immediate_fd_wakes_in_episode": n_fd_wakes,
        "immediate_fd_wake_ordinal": wake_ordinal,
        # actor-input diagnostics (recorded values)
        "tick": wake.get("tick"),
        "n_task_nodes": wake.get("n_task_nodes"),
        "n_agent_nodes": wake.get("n_agent_nodes"),
        "ego_fuel_norm": wake.get("ego_fuel_norm"),
        "reachable_by_ego": wake.get("reachable_by_ego"),
        "task_distance_norm": wake.get("task_distance_norm"),  # the dist_to_ego_norm column
        "n_task_distance_clipped": wake.get("n_task_distance_clipped"),
        "fraction_task_distance_clipped": wake.get("fraction_task_distance_clipped"),
        # action / distribution diagnostics (recorded values)
        "selected_meta_action_name": sel,
        "selected_node": wake.get("selected_node"),
        "selected_node_ownership": wake.get("selected_node_ownership"),
        "aggregate_p_plan": agg_p(wake, PLAN),
        "aggregate_p_engage": agg_p(wake, ENGAGE),
        "aggregate_p_abort": agg_p(wake, ABORT),
        # per-wake indicator whose run-level mean is selected_joint_cell_abort_fraction
        "selected_joint_cell_is_abort": (None if sel is None else sel == ABORT),
        "joint_argmax_meta_action_name": wake.get("joint_argmax_meta_action_name"),
        "aggregate_argmax_meta_action_name": wake.get("aggregate_argmax_meta_action_name"),
        "joint_vs_aggregate_disagree": wake.get("joint_vs_aggregate_disagree"),
        "joint_entropy_raw": wake.get("joint_entropy_raw"),
        "joint_entropy_normalized": wake.get("joint_entropy_normalized"),
        "aggregate_meta_action_entropy": wake.get("aggregate_meta_action_entropy"),
        "n_valid_cells": wake.get("n_valid_cells"),
        "episode": {k: rec[k] for k in EPISODE_CONTEXT_KEYS if k in rec},
        "wake_decision": wake,
    }


# ---------------------------------------------------------------- pairing arithmetic

def vector_compare(a, b, exact):
    """Elementwise comparison when both are equal-length numeric lists; otherwise a stated reason."""
    if not isinstance(a, list) or not isinstance(b, list):
        return {"comparable": False, "reason": "vector_not_recorded"}
    if len(a) != len(b):
        return {"comparable": False, "reason": "length_mismatch_%d_vs_%d" % (len(a), len(b))}
    if any(num(x) is None for x in a + b):
        return {"comparable": False, "reason": "non_numeric_entry"}
    n = len(a)
    diffs = [abs(y - x) for x, y in zip(a, b)]
    n_differ = sum(1 for d in diffs if d != 0)
    out = {"comparable": True, "length": n, "n_entries_differ": n_differ,
           "fraction_entries_differ": (n_differ / n) if n else None}
    if not exact:
        out.update({"mean_abs_diff": (sum(diffs) / n) if n else None,
                    "max_abs_diff": max(diffs) if n else None})
    return out


def pair_record(run, info, ident, group_key, mild_rows, severe_rows, reasons):
    base = {"run_id": run["run_id"], "variant": run["variant"], "measured_code_sha": info["measured_sha"],
            "training_mode": info["training_mode"], **ident_dict(ident), "benchmark_group_key": group_key}
    if reasons:
        return {**base, "pair_formed": False, "reasons": reasons,
                "mild_source_lines": [r["source_line"] for r in mild_rows],
                "severe_source_lines": [r["source_line"] for r in severe_rows]}
    m, s = mild_rows[0], severe_rows[0]
    me, se = m["episode"], s["episode"]
    sel_m, sel_s = m["selected_meta_action_name"], s["selected_meta_action_name"]
    same = None if sel_m is None or sel_s is None else sel_m == sel_s

    def both(key):
        return {"mild": m[key], "severe": s[key], "severe_minus_mild": sub(s[key], m[key])}

    return {
        **base,
        "pair_formed": True,
        "reasons": [],
        "benchmark_profile": m["benchmark_profile"],
        "benchmark_base_cell": m["benchmark_base_cell"],
        "mild_source": {"seed": m["seed"], "episode_tag": m["episode_tag"], "source_line": m["source_line"],
                        "tick": m["tick"]},
        "severe_source": {"seed": s["seed"], "episode_tag": s["episode_tag"], "source_line": s["source_line"],
                          "tick": s["tick"]},
        "same_seed": m["seed"] == s["seed"],
        "same_base_cell": m["benchmark_base_cell"] == s["benchmark_base_cell"],
        "same_wake_tick": m["tick"] == s["tick"],
        "ego_fuel_norm": both("ego_fuel_norm"),
        "n_task_nodes": both("n_task_nodes"),
        "reachable_by_ego": {"mild": m["reachable_by_ego"], "severe": s["reachable_by_ego"],
                             "compare": vector_compare(m["reachable_by_ego"], s["reachable_by_ego"], exact=True)},
        "task_distance_norm": {"mild": m["task_distance_norm"], "severe": s["task_distance_norm"],
                               "compare": vector_compare(m["task_distance_norm"], s["task_distance_norm"],
                                                         exact=False)},
        "n_task_distance_clipped": both("n_task_distance_clipped"),
        "fraction_task_distance_clipped": both("fraction_task_distance_clipped"),
        "aggregate_p_abort": both("aggregate_p_abort"),
        "aggregate_p_plan": both("aggregate_p_plan"),
        "aggregate_p_engage": both("aggregate_p_engage"),
        "selected_meta_action": {"mild": sel_m, "severe": sel_s, "match": same,
                                 "change": (None if same in (None, True) else "%s->%s" % (sel_m, sel_s))},
        "selected_node_ownership": {"mild": m["selected_node_ownership"], "severe": s["selected_node_ownership"]},
        "joint_vs_aggregate_disagree": {"mild": m["joint_vs_aggregate_disagree"],
                                        "severe": s["joint_vs_aggregate_disagree"]},
        "joint_entropy_raw": both("joint_entropy_raw"),
        "joint_entropy_normalized": both("joint_entropy_normalized"),
        "aggregate_meta_action_entropy": both("aggregate_meta_action_entropy"),
        "n_valid_cells": both("n_valid_cells"),
        "episode_context": {
            key: {"mild": me.get(key), "severe": se.get(key)}
            for key in ("reward", "n_dead", "fd_rtb_command_issued", "targets_confirmed_unique", "ended",
                        "fd_wake_meta_action_name")
        },
        "reward_severe_minus_mild": sub(se.get("reward"), me.get("reward")),
    }


# ---------------------------------------------------------------- main

def main():
    before = source_identities()
    anomalies = []
    ledger_mismatch = []
    for run in RUNS:
        for name, expected in run["sha256"].items():
            got = before[(run["run_id"], name)]["sha256"]
            if got != expected:
                ledger_mismatch.append({"run_id": run["run_id"], "file": name, "expected": expected, "actual": got})
    if ledger_mismatch:
        print(json.dumps({"STOP": "source identity does not match preserved ledger", "mismatch": ledger_mismatch},
                         indent=1))
        sys.exit(2)

    wake_rows, pairs, run_summaries = [], [], []
    for run in RUNS:
        info = read_run(run, anomalies)
        rounds = info["eval_rounds"]
        if len(rounds) != EXPECTED_ROUNDS:
            anomalies.append({"run_id": run["run_id"], "kind": "eval_round_count", "value": len(rounds)})
        round_ids = [round_identity(r) for r in rounds]
        if len(set(round_ids)) != len(round_ids):
            anomalies.append({"run_id": run["run_id"], "kind": "duplicate_eval_round_identity"})

        # Index evaluation members by (round identity, group key, member condition).
        by_key = {}
        for rec in info["members"]:
            ident = round_identity(rec)
            if not identity_complete(ident):
                anomalies.append({"run_id": run["run_id"], "kind": "incomplete_member_round_identity",
                                  "source_line": rec["_line"], "identity": list(ident)})
                continue
            if ident not in round_ids:
                anomalies.append({"run_id": run["run_id"], "kind": "member_round_not_in_eval_records",
                                  "source_line": rec["_line"], "identity": list(ident)})
            cond = member_condition(rec)
            if cond in SEVERITIES and (rec.get("cell") != cond or rec.get("severity") != cond):
                anomalies.append({"run_id": run["run_id"], "kind": "member_condition_field_disagreement",
                                  "source_line": rec["_line"], "member_cell": cond, "cell": rec.get("cell"),
                                  "severity": rec.get("severity")})
            by_key.setdefault((ident, rec.get("benchmark_group_key"), cond), []).append(rec)

        per_round = []
        final_ordinal = max((i[2] for i in round_ids if isinstance(i[2], int)), default=None)
        for er, ident in zip(rounds, round_ids):
            groups = er.get("v2_benchmark_groups") or {}
            prof = groups.get("benchmark_profile_identity") or {}
            group_keys = prof.get("group_keys") or []
            if not identity_complete(ident):
                anomalies.append({"run_id": run["run_id"], "kind": "incomplete_eval_round_identity",
                                  "identity": list(ident)})
            if er.get("benchmark_manifest_id") != EXPECTED_MANIFEST_ID or er.get("benchmark_profile") != EXPECTED_PROFILE:
                anomalies.append({"run_id": run["run_id"], "kind": "eval_round_manifest_or_profile_unexpected",
                                  "identity": list(ident), "profile": er.get("benchmark_profile")})
            if len(group_keys) != EXPECTED_GROUPS_PER_ROUND:
                anomalies.append({"run_id": run["run_id"], "kind": "group_key_count", "identity": list(ident),
                                  "value": len(group_keys)})

            round_rows = {sev: [] for sev in SEVERITIES}
            fd_wake_count_hist = {sev: Counter() for sev in SEVERITIES}
            round_pairs = []
            seen_pair_ids = set()
            member_counts = Counter()
            for gk in group_keys:
                pair_id = (run["run_id"],) + ident + (gk,)
                if pair_id in seen_pair_ids:
                    anomalies.append({"run_id": run["run_id"], "kind": "duplicate_pair_identity",
                                      "identity": list(ident), "group_key": gk})
                seen_pair_ids.add(pair_id)
                reasons, rows_by_sev = [], {}
                for cond in ("clean",) + SEVERITIES:
                    member_counts[cond] += len(by_key.get((ident, gk, cond), []))
                for sev in SEVERITIES:
                    recs = by_key.get((ident, gk, sev), [])
                    rows_by_sev[sev] = []
                    if len(recs) == 0:
                        reasons.append("missing_%s_member" % sev)
                        continue
                    if len(recs) > 1:
                        reasons.append("duplicate_%s_member_%d" % (sev, len(recs)))
                    for rec in recs:
                        bv2 = rec.get("benchmark_v2") or {}
                        if rec.get("benchmark_manifest_id") != EXPECTED_MANIFEST_ID or bv2.get("profile") != EXPECTED_PROFILE:
                            reasons.append("%s_member_manifest_or_profile_mismatch" % sev)
                        wakes = rec.get("wake_decisions")
                        if not isinstance(wakes, list):
                            reasons.append("%s_wake_decisions_not_recorded" % sev)
                            fd_wake_count_hist[sev]["not_recorded"] += 1
                            continue
                        fd = [w for w in wakes if isinstance(w, dict) and w.get("wake_kind") == IMMEDIATE_FD]
                        fd_wake_count_hist[sev][len(fd)] += 1
                        if len(fd) != 1:
                            reasons.append("%s_immediate_fd_wake_count_%d" % (sev, len(fd)))
                        for j, w in enumerate(fd):
                            row = wake_row(run, info, rec, w, j, len(fd))
                            wake_rows.append(row)
                            round_rows[sev].append(row)
                            rows_by_sev[sev].append(row)
                pr = pair_record(run, info, ident, gk, rows_by_sev["mild"], rows_by_sev["severe"], reasons)
                pairs.append(pr)
                round_pairs.append(pr)
            # members present in outcomes for this round but outside the profile's group keys
            stray = sorted({k[1] for k in by_key if k[0] == ident and k[1] not in group_keys})
            if stray:
                anomalies.append({"run_id": run["run_id"], "kind": "members_outside_profile_group_keys",
                                  "identity": list(ident), "group_keys": stray})
            per_round.append(round_summary(run, ident, er, group_keys, member_counts, round_rows,
                                           fd_wake_count_hist, round_pairs, final_ordinal))
        run_summaries.append({"run_id": run["run_id"], "variant": run["variant"],
                              "measured_code_sha": info["measured_sha"], "training_mode": info["training_mode"],
                              "n_eval_rounds": len(rounds), "rounds": per_round})

    after = source_identities()
    unchanged = all(before[k] == after[k] for k in before)
    write_outputs(before, after, unchanged, wake_rows, pairs, run_summaries, anomalies)


def round_summary(run, ident, er, group_keys, member_counts, round_rows, hist, round_pairs, final_ordinal):
    formed = [p for p in round_pairs if p["pair_formed"]]
    landmarks = []
    if ident[0] == "post_update" and ident[1] in LANDMARK_UPDATES.get(run["variant"], []):
        landmarks.append("task_named_round_%s_update_%d" % (run["variant"], ident[1]))
    if ident[2] == final_ordinal and ident[0] == "post_update":
        landmarks.append("final_post_update_round")
    sel_counts = {sev: {n: sum(1 for r in round_rows[sev] if r["selected_meta_action_name"] == n) for n in META_NAMES}
                  for sev in SEVERITIES}
    recorded = {sev: er.get("eval_fd_meta_action_counts_%s" % sev) for sev in SEVERITIES}
    return {
        **ident_dict(ident),
        "landmarks": landmarks,
        "n_benchmark_groups": len(group_keys),
        "eval_record_complete_groups": len((er.get("v2_benchmark_groups") or {}).get("complete_group_keys") or []),
        "eval_record_incomplete_groups": len((er.get("v2_benchmark_groups") or {}).get("incomplete_group_keys") or []),
        "outcome_members_by_condition": dict(member_counts),
        "immediate_fd_wakes_per_member_histogram": {sev: {str(k): v for k, v in hist[sev].items()} for sev in SEVERITIES},
        "wake_rows": {sev: len(round_rows[sev]) for sev in SEVERITIES},
        "complete_mild_severe_pairs": len(formed),
        "pairs_not_formed": [{"group_key": p["benchmark_group_key"], "reasons": p["reasons"]}
                             for p in round_pairs if not p["pair_formed"]],
        "selected_meta_action_counts": sel_counts,
        "eval_record_fd_meta_action_counts": recorded,
        "selected_counts_equal_eval_record": {sev: (recorded[sev] == sel_counts[sev]) if isinstance(recorded[sev], dict)
                                              else None for sev in SEVERITIES},
        "n_pairs_selected_action_differs": sum(1 for p in formed if p["selected_meta_action"]["match"] is False),
        "selected_action_change_counts": dict(Counter(p["selected_meta_action"]["change"] for p in formed
                                                      if p["selected_meta_action"]["change"])),
        "n_joint_vs_aggregate_disagree": {sev: sum(1 for r in round_rows[sev] if r["joint_vs_aggregate_disagree"] is True)
                                          for sev in SEVERITIES},
        "n_pairs_joint_vs_aggregate_disagree_either": sum(
            1 for p in formed if True in (p["joint_vs_aggregate_disagree"]["mild"], p["joint_vs_aggregate_disagree"]["severe"])),
        "aggregate_p_abort_severe_minus_mild": stats([p["aggregate_p_abort"]["severe_minus_mild"] for p in formed]),
        "ego_fuel_norm_severe_minus_mild": stats([p["ego_fuel_norm"]["severe_minus_mild"] for p in formed]),
        "fraction_task_distance_clipped": {sev: stats([r["fraction_task_distance_clipped"] for r in round_rows[sev]])
                                           for sev in SEVERITIES},
        "fraction_task_distance_clipped_severe_minus_mild": stats(
            [p["fraction_task_distance_clipped"]["severe_minus_mild"] for p in formed]),
        "n_wakes_all_task_distances_clipped": {
            sev: sum(1 for r in round_rows[sev] if r["fraction_task_distance_clipped"] == 1.0) for sev in SEVERITIES},
    }


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def write_outputs(before, after, unchanged, wake_rows, pairs, run_summaries, anomalies):
    expected_rows = len(RUNS) * EXPECTED_ROUNDS * EXPECTED_GROUPS_PER_ROUND * len(SEVERITIES)
    expected_pairs = len(RUNS) * EXPECTED_ROUNDS * EXPECTED_GROUPS_PER_ROUND
    n_formed = sum(1 for p in pairs if p["pair_formed"])
    with open(os.path.join(HERE, "immediate_fd_wakes.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in wake_rows:
            f.write(dumps(r) + "\n")
    with open(os.path.join(HERE, "matched_mild_severe_pairs.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for p in pairs:
            f.write(dumps(p) + "\n")
    sources = [{"run_id": rid, "file": name, **before[(rid, name)],
                "unchanged_after_extraction": before[(rid, name)] == after[(rid, name)]}
               for (rid, name) in before]
    summary = {
        "package": "v2_wake_pair_diagnostics",
        "status": "temporary review-only extraction; factual counts, no interpretation",
        "extraction_rule": ("evaluation phases pre_update and post_update; all rounds in eval_records.jsonl; "
                            "benchmark_v2.member_cell in {mild, severe}; wake_decisions entries with "
                            "wake_kind == immediate_fuel_damage; pairs keyed by run_id + "
                            "(evaluation_stage, updates_completed, eval_round_ordinal, benchmark_manifest_id) "
                            "+ benchmark_group_key over the round's recorded profile group keys"),
        "expected_wake_rows": expected_rows,
        "actual_wake_rows": len(wake_rows),
        "expected_pair_records": expected_pairs,
        "actual_pair_records": len(pairs),
        "formed_pairs": n_formed,
        "pairs_not_formed": len(pairs) - n_formed,
        "sources_unchanged_by_extraction": unchanged,
        "anomalies": anomalies,
        "sources": sources,
        "runs": run_summaries,
    }
    with open(os.path.join(HERE, "extraction_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("expected_wake_rows", "actual_wake_rows", "expected_pair_records",
                                               "actual_pair_records", "formed_pairs", "pairs_not_formed",
                                               "sources_unchanged_by_extraction")}, indent=1))
    print("anomalies:", len(anomalies))


if __name__ == "__main__":
    main()
