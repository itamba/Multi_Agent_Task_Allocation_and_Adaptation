"""DESCRIPTIVE per-iteration training-stream comparison (after the declared first divergence,
records are compared only descriptively; plan: prefix_comparison). Uses compare_prefix's
normalization and classifier unchanged. For each training iteration: episodes identical /
policy-side / input-side / absent, credit rows identical, the largest absolute difference of the
first differing field, and whether seeds and cells match; plus the first TRAINING-stream
divergence (evaluation episodes do not feed the learner).

Usage: python training_stream_divergence.py <new run> <original run> <out json>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_prefix as C  # noqa: E402


def main() -> int:
    new_dir, orig_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    orig = C.load_original(orig_dir)
    new = C.read_complete_lines(new_dir / "episode_outcomes.jsonl")
    cred = C.read_complete_lines(new_dir / "train_credit_diagnostics.jsonl")
    per = defaultdict(lambda: {"episodes": 0, "identical": 0, "policy_side": 0,
                               "input_side": 0, "absent": 0, "seed_cell_match": 0,
                               "max_first_field_abs_diff": 0.0, "credit_rows": 0,
                               "credit_identical": 0})
    first = None
    for r in new:
        if r.get("phase") != "train":
            continue
        p = per[r["iteration"]]
        p["episodes"] += 1
        o = orig["out"].get(C._outcome_key(r))
        if o is None:
            p["absent"] += 1
            continue
        p["seed_cell_match"] += int((r["seed"], r.get("cell"), r.get("severity"))
                                    == (o["seed"], o.get("cell"), o.get("severity")))
        if C.first_difference(C.normalize(r), C.normalize(o)) is None:
            p["identical"] += 1
            continue
        c = C.classify_episode(r, o)
        p[c["side"]] += 1
        diff = (c.get("difference") or {}).get("abs_diff")
        if isinstance(diff, (int, float)):
            p["max_first_field_abs_diff"] = max(p["max_first_field_abs_diff"], float(diff))
        if first is None:
            first = dict(c, key=list(C._outcome_key(r)))
    for r in cred:
        p = per[r["iteration"]]
        p["credit_rows"] += 1
        o = orig["cred"].get((r["iteration"], r["batch_transition_ordinal"]))
        p["credit_identical"] += int(o is not None and C.first_difference(
            C.normalize(r), C.normalize(o)) is None)
    rows = [dict(iteration=k, **v) for k, v in sorted(per.items())]
    rep = {"record": "training_stream_divergence", "record_version": 1,
           "descriptive_only": True,
           "first_training_stream_divergence": first,
           "iterations_fully_identical": [r["iteration"] for r in rows
                                          if r["identical"] == r["episodes"]
                                          and r["credit_identical"] == r["credit_rows"]],
           "all_seeds_and_cells_match": all(r["seed_cell_match"] == r["episodes"] - r["absent"]
                                            and r["absent"] == 0 for r in rows),
           "per_iteration": rows}
    out.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("first_training_stream_divergence",
                                          "iterations_fully_identical",
                                          "all_seeds_and_cells_match")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
