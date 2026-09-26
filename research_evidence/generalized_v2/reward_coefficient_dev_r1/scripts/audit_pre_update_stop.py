"""Read-only audit of the arm-B pre-update STOP (standard library; no training, replay or load).

The launcher's pre-declared comparison (``compare_pre_update.py``) stopped arm B at 120 s with
``pre_update_mismatch_stop``. This script re-runs that comparison on the recorded outcomes and
classifies every mismatch:

  * per field (ego_id / tick / selected action / leaf / wake kind / probability);
  * whether each mismatching ego id is even stable WITHIN arm A across A's own 16 evaluation
    rounds of the same frozen member (if not, the id is per-episode and cannot identify an
    ego across runs);
  * tick deltas, and whether frozen-world identity, selected actions, probabilities and the
    recorded outcomes other than ``ticks`` agree.

It reads the two runs' ``episode_outcomes.jsonl`` and the launcher's
``pre_update_identity.json``, hashes them before and after, and writes one JSON record.

Usage: python audit_pre_update_stop.py <out.json>
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_pre_update as C  # noqa: E402

A = Path(r"C:\gruns\reward_c225_r1_s3000000_2b57019")
B = Path(r"C:\gruns\reward_c450_r1_s3000000_2b57019")
LIVE = Path(str(B) + "__launcher") / "pre_update_identity.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    srcs = [A / "episode_outcomes.jsonl", B / "episode_outcomes.jsonl", LIVE]
    before = {str(p): sha(p) for p in srcs}
    live = json.loads(LIVE.read_text(encoding="utf-8"))
    rep = C.compare(B, A)
    same_as_live = ({k: v for k, v in rep.items()} ==
                    {k: v for k, v in live.items()
                     if k not in ("checked_at", "elapsed_seconds_at_check")})
    a, b = C.load_pre_update(A), C.load_pre_update(B)
    n_action = sum(1 for k in a for x, y in zip(a[k]["wake_decisions"], b[k]["wake_decisions"])
                   if (x["wake_kind"], x["selected_meta_action_name"], x["selected_leaf"])
                   != (y["wake_kind"], y["selected_meta_action_name"], y["selected_leaf"]))
    ego_members = sorted({tuple(m["member"]) for m in rep["wake_mismatches"]
                          if m["field"] == "ego_id"})
    own = defaultdict(set)
    with open(A / "episode_outcomes.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["phase"] not in ("pre_update", "post_update"):
                continue
            key = (r["benchmark_group_key"], r["benchmark_v2"]["member_cell"])
            if key in ego_members:
                own[key].add(tuple(sorted({w["ego_id"] for w in r["wake_decisions"]})))
    fd_ego_equal = all(a[k].get("fd_ego_id") == b[k].get("fd_ego_id") for k in a)
    ticks = Counter(m["b"] - m["a"] for m in rep["wake_mismatches"] if m["field"] == "tick")
    out = {
        "record": "pre_update_stop_audit", "record_version": 1,
        "scope": "read-only classification of recorded outputs; no training, evaluation, "
                 "replay or checkpoint load",
        "live_report_reproduced_exactly": same_as_live,
        "stop": rep["stop"], "stop_reasons": rep["stop_reasons"],
        "n_members": {"A": rep["n_members_a"], "B": rep["n_members_b"]},
        "n_wakes_compared": rep["n_wakes_compared"],
        "frozen_world_identity_mismatches": len(rep["identity_mismatches"]),
        "max_abs_semantic_probability_difference": rep["max_abs_semantic_probability_difference"],
        "probability_tolerance": rep["probability_tolerance"],
        "wakes_with_different_wake_kind_or_selected_action_or_leaf": n_action,
        "wake_mismatches_by_field": dict(Counter(m["field"] for m in rep["wake_mismatches"])),
        "tick_deltas_b_minus_a": {str(k): v for k, v in sorted(ticks.items())},
        "mismatch_members": sorted({"%s/%s" % tuple(m["member"])
                                    for m in rep["wake_mismatches"]}),
        "ego_id_mismatch_members": ["%s/%s" % k for k in ego_members],
        "ego_id_stability_within_arm_A": {
            "%s/%s" % k: {"distinct_ego_id_sets_over_A_rounds": len(v), "rounds": 16}
            for k, v in sorted(own.items())},
        "fd_ego_id_equal_all_members": fd_ego_equal,
        "record_only": {
            "members_with_outcome_differences": rep["record_only"]
            ["n_members_with_outcome_differences"],
            "outcome_difference_fields": dict(Counter(
                x["field"] for x in rep["record_only"]["outcome_field_differences"])),
            "penalty_b_equals_2x_a_all_members": rep["record_only"]
            ["penalty_b_equals_2x_a_all_members"]},
        "wake_mismatches": rep["wake_mismatches"],
        "outcome_field_differences": rep["record_only"]["outcome_field_differences"],
        "sources_sha256": before,
    }
    after = {str(p): sha(p) for p in srcs}
    assert before == after, "a source changed during the audit"
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n",
                                 encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in (
        "live_report_reproduced_exactly", "frozen_world_identity_mismatches",
        "max_abs_semantic_probability_difference",
        "wakes_with_different_wake_kind_or_selected_action_or_leaf",
        "wake_mismatches_by_field", "tick_deltas_b_minus_a", "fd_ego_id_equal_all_members",
        "record_only")}, indent=1))


if __name__ == "__main__":
    main()
