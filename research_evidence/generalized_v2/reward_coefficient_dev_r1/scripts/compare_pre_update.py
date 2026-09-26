"""Pre-declared pre-update identity check between the two REWARD-01 arms (standard library).

Compares the ALREADY RECORDED ``pre_update`` round of arm B with arm A's (the first 60
``episode_outcomes.jsonl`` records of each run). It runs no evaluation, replay or checkpoint
load. The rule is fixed in ``authorized_plan.json:/pre_update_mismatch_rule``:

  STOP (``stop = True``) if
    * the member sets differ, or any frozen-world identity field differs for a member;
    * any member's wake sequence differs in count, wake kind, ego, tick, selected meta-action
      or selected leaf;
    * any recorded semantic meta-action probability differs by more than ``PROB_TOL``.

  RECORD ONLY: episode outcomes (utility, deaths, ticks, end reason, FD fields), the reward
  ratio, and whether arm B's penalty is exactly twice arm A's (4.5 / 2.25).

``compare(b_run, a_run)`` returns ``None`` while arm B's pre-update round is incomplete, and
otherwise a report dict. It never writes into either run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

N_PRE_UPDATE = 60
PROB_TOL = 1e-6
PENALTY_RATIO = 4.5 / 2.25
IDENTITY_FIELDS = ("seed", "episode_tag", "benchmark_group_key", "benchmark_manifest_id",
                   "benchmark_stratum", "benchmark_world_ordinal", "benchmark_world_identity",
                   "fuel_damage_mode", "cell", "severity")
WAKE_FIELDS = ("wake_kind", "ego_id", "tick", "selected_meta_action_name", "selected_leaf")
OUTCOME_FIELDS = ("u_achieved", "u_ref", "u_prefix", "u_cont_ref", "u_post", "u_aircraft",
                  "n_dead", "ticks", "ended", "n_wakes", "targets_confirmed_unique",
                  "unique_completed_targets", "scored_completed_targets",
                  "unscored_completed_targets", "reward_ratio", "fd_ego_id", "fd_event_tick",
                  "fd_fuel_after", "fd_rtb_command_issued", "fd_certificate_fingerprint",
                  "reference_kind", "reference_checkpoint_tick")


def load_pre_update(run_dir: Path):
    """The pre_update records keyed by (group, member cell); None while incomplete.

    Reads line by line and stops at the first non-pre_update record, so it never parses the
    whole stream. A trailing partial line (the trainer mid-write) ends the read.
    """
    path = Path(run_dir) / "episode_outcomes.jsonl"
    if not path.exists():
        return None
    recs = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.endswith("\n"):
                break
            r = json.loads(line)
            if r.get("phase") != "pre_update":
                break
            key = (r["benchmark_group_key"], (r.get("benchmark_v2") or {}).get("member_cell"))
            if key in recs:
                raise ValueError("duplicate pre_update member %r in %s" % (key, run_dir))
            recs[key] = r
    return recs if len(recs) >= N_PRE_UPDATE else None


def _probs(w):
    return w.get("semantic_probability_per_meta_action") or {}


def compare(b_run: Path, a_run: Path):
    b = load_pre_update(b_run)
    if b is None:
        return None
    a = load_pre_update(a_run)
    if a is None:
        raise ValueError("arm A's pre_update round is incomplete: %s" % a_run)
    reasons, identity_mismatch, wake_mismatch = [], [], []
    max_dp, n_wakes, outcome_diff, penalty_ratio_bad = 0.0, 0, [], []
    if set(a) != set(b):
        reasons.append("member sets differ")
    for key in sorted(set(a) & set(b)):
        ra, rb = a[key], b[key]
        for f in IDENTITY_FIELDS:
            if ra.get(f) != rb.get(f):
                identity_mismatch.append({"member": list(key), "field": f})
        wa, wb = ra.get("wake_decisions") or [], rb.get("wake_decisions") or []
        if len(wa) != len(wb):
            wake_mismatch.append({"member": list(key), "field": "n_wake_decisions",
                                  "a": len(wa), "b": len(wb)})
        for i, (x, y) in enumerate(zip(wa, wb)):
            n_wakes += 1
            for f in WAKE_FIELDS:
                if x.get(f) != y.get(f):
                    wake_mismatch.append({"member": list(key), "wake": i, "field": f,
                                          "a": x.get(f), "b": y.get(f)})
            px, py = _probs(x), _probs(y)
            if set(px) != set(py):
                wake_mismatch.append({"member": list(key), "wake": i,
                                      "field": "probability keys"})
            for m in set(px) & set(py):
                max_dp = max(max_dp, abs(float(px[m]) - float(py[m])))
        for f in OUTCOME_FIELDS:
            if ra.get(f) != rb.get(f):
                outcome_diff.append({"member": list(key), "field": f, "a": ra.get(f),
                                     "b": rb.get(f)})
        pa, pb = float(ra.get("reward_penalty") or 0.0), float(rb.get("reward_penalty") or 0.0)
        if abs(pb - PENALTY_RATIO * pa) > 1e-12 * max(1.0, abs(pb)):
            penalty_ratio_bad.append({"member": list(key), "a": pa, "b": pb})
    if identity_mismatch:
        reasons.append("frozen-world identity differs")
    if wake_mismatch:
        reasons.append("wake sequence / selected action differs")
    if max_dp > PROB_TOL:
        reasons.append("semantic probability differs by %.3g > %.0e" % (max_dp, PROB_TOL))
    return {
        "record": "pre_update_identity", "record_version": 1,
        "rule": "authorized_plan.json:/pre_update_mismatch_rule",
        "arm_a_run": str(a_run), "arm_b_run": str(b_run),
        "n_members_a": len(a), "n_members_b": len(b), "n_wakes_compared": n_wakes,
        "identity_mismatches": identity_mismatch, "wake_mismatches": wake_mismatch,
        "max_abs_semantic_probability_difference": max_dp, "probability_tolerance": PROB_TOL,
        "stop": bool(reasons), "stop_reasons": reasons,
        "record_only": {
            "outcome_field_differences": outcome_diff,
            "n_members_with_outcome_differences": len({tuple(d["member"])
                                                       for d in outcome_diff}),
            "penalty_b_equals_2x_a_all_members": not penalty_ratio_bad,
            "penalty_ratio_exceptions": penalty_ratio_bad,
        },
    }


if __name__ == "__main__":
    import sys
    rep = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(rep, indent=1))
