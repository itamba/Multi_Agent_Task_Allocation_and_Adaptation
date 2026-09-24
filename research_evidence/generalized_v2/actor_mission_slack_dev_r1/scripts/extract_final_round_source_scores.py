"""Deterministic extract of the FINAL-ROUND immediate-FD per-node source scores (review fix F3).

Standard library only; reads the two runs' original ``episode_outcomes.jsonl`` files (hash
pinned, verified before and after), runs nothing, infers nothing. It preserves the recorded
per-node source scores that the evidence README's per-node remark rests on, so that remark is
inspectable from the package.

SELECTION RULE (per run): records with ``phase == "post_update"``,
``eval_round_ordinal == 15`` and ``updates_completed == 375``, whose benchmark member cell is
``mild`` or ``severe``; within each, the ONE ``immediate_fuel_damage`` wake, whose ego must be the
record's ``fd_ego_id``. Exactly 40 wakes per run (20 groups x MILD/SEVERE) are required.

NODE CLASSES come from the recorded legality alone: ``source_cell_legal[v][2]`` (the ABORT cell)
is 1 exactly when task node ``v`` is assigned to the ego (``graph_action.build_action_mask``), so
a node is ``ego_assigned`` or ``not_ego_assigned``. The record does NOT distinguish a
peer-assigned node from an unassigned one, and this extract does not pretend to.

Score columns follow ``MetaAction``: 0 = PLAN_COMPLIANCE, 1 = OPPORTUNISTIC_ENGAGEMENT,
2 = SELF_PRESERVATION_ABORT. Ranges are exact ``max - min`` over the unrounded recorded float
values (float32 logits written as JSON floats); no tolerance or rounding is applied.

    python extract_final_round_source_scores.py --out <package dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RUNS = {
    "new": {
        "run_id": "graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441",
        "measured_code_sha": "3bc944119da08af8e25268c9ee83fc63a8d1e533",
        "path": r"C:\gruns\graph_rl_v2_actor_mission_slack_dev_r1_seed3000000_3bc9441"
                r"\episode_outcomes.jsonl",
    },
    "comparator": {
        "run_id": "graph_rl_v2_semantic_action_actor_only_dev_r1_seed3000000_d4e9f37",
        "measured_code_sha": "d4e9f3721e6d151c00be3fe93c3d149df9d31965",
        "path": r"C:\gra\runs\development\v2_semantic_actor_only_r1_seed3000000_d4e9f37"
                r"\episode_outcomes.jsonl",
    },
}
FD_WAKE = "immediate_fuel_damage"
ROUND, UPDATES, N_WAKES = 15, 375, 40
COLUMNS = ("PLAN_COMPLIANCE", "OPPORTUNISTIC_ENGAGEMENT", "SELF_PRESERVATION_ABORT")
ABORT_COL = 2


class ExtractError(RuntimeError):
    pass


def require(cond, msg, *args):
    if not cond:
        raise ExtractError(msg % args if args else msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_hashes(package: Path):
    """The original outcomes' hashes as recorded by the ORIGINAL evidence ledger."""
    out = {}
    for line in (package / "artifact_sha256.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        sha, size, _g, path = line.split("  ", 3)
        out[path] = (sha, int(size))
    return out


def select(path: Path, label: str):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("phase") != "post_update" or r.get("eval_round_ordinal") != ROUND):
                continue
            require(r.get("updates_completed") == UPDATES, "%s: round %d updates", label, ROUND)
            cell = (r.get("benchmark_v2") or {}).get("member_cell")
            if cell not in ("mild", "severe"):
                continue
            fd = [w for w in r.get("wake_decisions") or () if w.get("wake_kind") == FD_WAKE]
            require(len(fd) == 1, "%s: %s/%s has %d FD wakes", label,
                    r.get("benchmark_group_key"), cell, len(fd))
            w = fd[0]
            require(w.get("ego_id") == r.get("fd_ego_id"), "%s: FD wake ego", label)
            scores, legal = w["source_scores"], w["source_cell_legal"]
            k = int(w["n_task_nodes"])
            require(len(scores) == k and len(legal) == k, "%s: node count", label)
            nodes = [{"node": v,
                      "ownership_class": ("ego_assigned" if int(legal[v][ABORT_COL]) == 1
                                          else "not_ego_assigned"),
                      "scores": [float(x) for x in scores[v]],
                      "cell_legal": [int(x) for x in legal[v]]} for v in range(k)]
            p = w["semantic_probability_per_meta_action"]
            rows.append({
                "run": label, "evaluation_stage": r["phase"], "eval_round_ordinal": ROUND,
                "updates_completed": UPDATES, "benchmark_group_key": r["benchmark_group_key"],
                "base_cell": (r.get("benchmark_v2") or {}).get("base_cell"),
                "member_cell": cell, "seed": r.get("seed"), "ego_id": w["ego_id"],
                "tick": w["tick"], "n_task_nodes": k,
                "n_abort_legal_nodes": w["n_abort_legal_nodes"],
                "selected_meta_action_name": w["selected_meta_action_name"],
                "p_abort": float(p["SELF_PRESERVATION_ABORT"]),
                "p_plan": float(p["PLAN_COMPLIANCE"]),
                "nodes": nodes})
    rows.sort(key=lambda x: (x["benchmark_group_key"], x["member_cell"]))
    require(len(rows) == N_WAKES, "%s: %d final-round FD wakes, expected %d", label, len(rows),
            N_WAKES)
    return rows


def spread(values):
    return None if not values else {"n": len(values), "min": min(values), "max": max(values),
                                    "range": max(values) - min(values)}


def ranges(rows):
    out = {"n_wakes": len(rows),
           "p_abort_across_wakes": spread([r["p_abort"] for r in rows])}
    for cls in ("ego_assigned", "not_ego_assigned", None):
        name = cls or "all_nodes"
        nodes = [n for r in rows for n in r["nodes"] if cls is None or n["ownership_class"] == cls]
        out[name] = {"n_nodes": len(nodes),
                     "n_wakes_with_class": sum(1 for r in rows if any(
                         cls is None or n["ownership_class"] == cls for n in r["nodes"])),
                     "score_range_across_all_nodes_and_wakes": {
                         COLUMNS[c]: spread([n["scores"][c] for n in nodes]) for c in range(3)}}
    within = []
    for r in rows:
        within.append(max(max(n["scores"][c] for n in r["nodes"])
                          - min(n["scores"][c] for n in r["nodes"]) for c in range(3)))
    out["max_within_wake_node_score_range"] = max(within)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the evidence package directory")
    args = ap.parse_args(argv)
    package = Path(args.out)
    pins = pinned_hashes(package)
    sources, all_rows, result = {}, [], {}
    try:
        for label, spec in RUNS.items():
            p = Path(spec["path"])
            require(spec["path"] in pins, "%s outcomes not pinned in artifact_sha256.txt", label)
            before = sha256_file(p)
            require((before, p.stat().st_size) == pins[spec["path"]],
                    "%s outcomes differ from the pinned original", label)
            rows = select(p, label)
            require(sha256_file(p) == before, "%s outcomes changed during extraction", label)
            sources[label] = {"run_id": spec["run_id"],
                              "measured_code_sha": spec["measured_code_sha"],
                              "path": spec["path"], "sha256": before,
                              "bytes": p.stat().st_size}
            all_rows.extend(rows)
            result[label] = ranges(rows)
    except ExtractError as exc:
        print("EXTRACT FAILED:", exc)
        return 2
    summary = {
        "record": "final_round_immediate_fd_source_scores", "review_fix": "F3",
        "selection_rule": ("post_update, eval_round_ordinal 15, updates_completed 375; mild and "
                           "severe members; the one immediate_fuel_damage wake of fd_ego_id; "
                           "40 wakes per run"),
        "node_classes": ("ego_assigned = ABORT cell legal; not_ego_assigned = ABORT cell illegal "
                         "(peer-assigned OR unassigned: the record does not distinguish them)"),
        "score_columns": list(COLUMNS),
        "tolerance": "none: exact max - min over the recorded (unrounded) values",
        "scope_note": ("describes the recorded final-round outputs of one training seed per run; "
                       "not a test of feature dependence and not a causal diagnosis"),
        "sources": sources, "ranges": result,
    }
    ex = package / "extracted"
    with open(ex / "final_round_fd_source_scores.jsonl", "w", encoding="utf-8",
              newline="\n") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    with open(ex / "final_round_fd_source_score_ranges.json", "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(json.dumps(summary, indent=1, sort_keys=True, ensure_ascii=True,
                            allow_nan=False) + "\n")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
