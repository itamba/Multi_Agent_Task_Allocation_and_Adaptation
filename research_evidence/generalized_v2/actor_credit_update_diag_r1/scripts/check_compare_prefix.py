"""Pre-launch self-check of ``compare_prefix`` (standard library only; reads, never writes, the
original run). Declared before launch: the comparator must report the original's first-100-update
prefix as identical to itself, and must classify planted perturbations as declared.

It builds a temporary PREFIX COPY of the original's streams (the pre-update round, iterations
0..99 and the rounds at 25..100) in a scratch directory, compares it with the original, then
plants one perturbation at a time in a copy and checks the reported side and location.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_prefix as C  # noqa: E402


def _prefix(rows, *, iteration_key="iteration", limit=99):
    out = []
    for r in rows:
        it = r.get(iteration_key)
        uc = r.get("updates_completed")
        if r.get("phase") == "train" or "batch_transition_ordinal" in r or "n_epochs_run" in r:
            if it is not None and it > limit:
                continue
        elif uc is not None and uc > limit + 1:
            continue
        out.append(r)
    return out


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def main() -> int:
    orig = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    tmp = Path(tempfile.mkdtemp(prefix="cmpchk_"))
    try:
        streams = {}
        for name in ("episode_outcomes.jsonl", "train_credit_diagnostics.jsonl",
                     "train_records.jsonl", "eval_records.jsonl", "episode_failures.jsonl"):
            streams[name] = _prefix(C.read_complete_lines(orig / name))
        base = tmp / "prefix"
        base.mkdir()
        for name, rows in streams.items():
            _write(base / name, rows)
        results = {"identity": C.compare(base, orig)}
        ok = (results["identity"]["first_divergence"] is None
              and not results["identity"]["stop"]
              and all(e["identical"] for e in results["identity"]["eval_records"]))

        outcomes = streams["episode_outcomes.jsonl"]
        # a train episode with a wake, deep in the prefix
        idx = next(i for i, r in enumerate(outcomes)
                   if r.get("phase") == "train" and r.get("iteration") == 40
                   and r.get("wake_decisions"))
        cases = {}

        def plant(label, mutate, expect_side):
            d = tmp / label
            d.mkdir()
            for name, rows in streams.items():
                rows = copy.deepcopy(rows) if name == "episode_outcomes.jsonl" else rows
                if name == "episode_outcomes.jsonl":
                    mutate(rows[idx])
                _write(d / name, rows)
            rep = C.compare(d, orig)
            fd = rep["first_divergence"] or {}
            cases[label] = {"side": fd.get("side"), "key": fd.get("key"),
                            "difference": fd.get("difference"), "stop": rep["stop"]}
            return fd.get("side") == expect_side and rep["stop"] == (expect_side == "policy_side")

        def policy(r):
            w = r["wake_decisions"][0]
            w["semantic_probabilities"][0] = float(w["semantic_probabilities"][0]) + 1e-12

        def tick(r):
            r["wake_decisions"][0]["tick"] = int(r["wake_decisions"][0]["tick"]) + 1

        def reward(r):
            r["reward"] = float(r["reward"]) + 1e-9

        def uuid_only(r):
            r["wake_decisions"][0]["ego_id"] = "00000000-0000-0000-0000-000000000000"

        ok &= plant("policy_output", policy, "policy_side")
        ok &= plant("wake_tick", tick, "input_side")
        ok &= plant("episode_reward", reward, "input_side")
        # a UUID-only change is normalized away: no divergence at all
        d = tmp / "uuid_only"
        d.mkdir()
        for name, rows in streams.items():
            rows = copy.deepcopy(rows) if name == "episode_outcomes.jsonl" else rows
            if name == "episode_outcomes.jsonl":
                uuid_only(rows[idx])
            _write(d / name, rows)
        rep = C.compare(d, orig)
        cases["uuid_only"] = {"first_divergence": rep["first_divergence"]}
        ok &= rep["first_divergence"] is None
        results["planted"] = cases
        results["pass"] = bool(ok)
        results["prefix_counts"] = {k: len(v) for k, v in streams.items()}
        out_path.write_text(json.dumps(results, indent=1, sort_keys=True, default=str),
                            encoding="utf-8")
        print("PASS" if ok else "FAIL", json.dumps(results["prefix_counts"]))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
