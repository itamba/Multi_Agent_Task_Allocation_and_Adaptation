"""Same-seed PREFIX consistency comparison: new instrumented run vs the original mission-slack run.

Standard library only. It never writes into either run directory. The comparison, its
normalization and its stop rule were DECLARED BEFORE LAUNCH in ``authorized_plan.json``
(``prefix_comparison``) and are not adapted after any data is seen.

WHAT IS COMPARED (keyed, never by line position):
  * ``episode_outcomes.jsonl`` -- key (phase, iteration, attempt_ordinal, eval_round_ordinal,
    eval_episode_index, seed); every field after NORMALIZATION;
  * ``episode_failures.jsonl`` -- key (phase, iteration, attempt_ordinal, seed); stage, error
    class and message head;
  * ``train_credit_diagnostics.jsonl`` -- key (iteration, batch_transition_ordinal);
  * ``train_records.jsonl`` -- key iteration; ``eval_records.jsonl`` -- key round ordinal;
  * in the final report only: the checkpoints at iterations 24 / 49 / 74 / 99 (tensor equality).

NORMALIZATION (the only exclusions): keys naming wall-clock time (``seconds``, ``_at``,
``elapsed``, ``timestamp``, ``wall``), filesystem paths and tracebacks (``output_dir``, ``path``,
``run_dir``, ``scenario``, ``traceback``; absolute Windows paths inside strings become
``<path>``), and run-random identities -- UUIDs (generated ids are not seed-stable; artifacts
and metrics §6.3), wherever they occur: whole values, substrings, and dict keys (replaced by
their ordinal). Floats are compared EXACTLY (tolerance 0). The new diagnostic stream
is not compared (it has no original).

FIRST DIVERGENCE AND ITS CLASS. Records are ordered by their position in the NEW run's own
streams (outcome stream order, with each update's credit rows and train record placed after
that iteration's episodes). The first record with any normalized difference is the first
divergence. Inside an episode, wakes are scanned in order: at the first wake with a difference,
if every ACTOR-INPUT field (``_WAKE_INPUT_FIELDS``) and every earlier wake is identical but a
policy output differs, the divergence is ``policy_side``; any other first difference (world,
construction, tick, wake kind, actor input, episode-level physics) is ``input_side``.

STOP RULE (declared): a ``policy_side`` first divergence means the actor produced a different
output from identical recorded inputs -- different parameters -- which the observational
diagnostic must never cause; it is a MATERIAL UNEXPLAINED DISCREPANCY and stops the run. An
``input_side`` first divergence is the pre-declared known class (BLADE run-to-run execution
nondeterminism, measurements §16.3): it is reported with its location and the run continues;
records after it are compared only descriptively.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_UUID_IN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_WIN_PATH_IN = re.compile(r"[A-Za-z]:[\\/][^\s\"',)]*")
_TIME_KEYS = ("seconds", "_at", "elapsed", "timestamp", "wall")
_PATH_KEYS = {"output_dir", "path", "run_dir", "scenario", "scenario_path", "scenario_file",
              "absolute_path", "traceback"}
_WAKE_INPUT_FIELDS = (
    "wake_kind", "tick", "n_task_nodes", "n_agent_nodes", "ego_fuel_norm",
    "actor_observation_id", "ego_mission_fuel_slack_norm", "mission_slack_audit",
    "reachable_by_ego", "task_distance_norm", "n_task_distance_clipped",
    "fraction_task_distance_clipped", "time_norm", "n_abort_legal_nodes",
    "n_engage_legal_leaves", "source_cell_legal", "n_valid_semantic_leaves",
    "n_semantic_leaves", "n_meta_actions", "action_representation_id",
)


def _is_time_key(k: str) -> bool:
    return any(t in k for t in _TIME_KEYS)


def normalize(obj: Any) -> Any:
    """The declared normalization (module docstring)."""
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if _is_time_key(k) or k in _PATH_KEYS:
                continue
            key = "<uuid#%d>" % i if isinstance(k, str) and _UUID.match(k) else k
            out[key] = normalize(v)
        return out
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, str):
        if _UUID.match(obj):
            return "<uuid>"
        return _WIN_PATH_IN.sub("<path>", _UUID_IN.sub("<uuid>", obj))
    return obj


def first_difference(a: Any, b: Any, where: str = "") -> Optional[Dict[str, Any]]:
    """The first (depth-first, key order of ``a``) differing leaf, or ``None``."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in list(a) + [k for k in b if k not in a]:
            if k not in a or k not in b:
                return {"path": where + "/" + str(k), "new": a.get(k, "<absent>"),
                        "original": b.get(k, "<absent>")}
            d = first_difference(a[k], b[k], where + "/" + str(k))
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, "%s[%d]" % (where, i))
            if d:
                return d
        if len(a) != len(b):
            return {"path": where + "#len", "new": len(a), "original": len(b)}
        return None
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                       and not isinstance(a, bool) and not isinstance(b, bool)):
        return {"path": where, "new": a, "original": b}
    if a != b:
        out = {"path": where, "new": a, "original": b}
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out["abs_diff"] = abs(float(a) - float(b))
        return out
    return None


def read_complete_lines(path: Path) -> List[Dict[str, Any]]:
    """Every COMPLETE json line (a trailing partial line of a live run is ignored)."""
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n")
    if end < 0:
        return []
    return [json.loads(line) for line in data[:end].decode("utf-8").splitlines() if line]


def _outcome_key(r: Dict[str, Any]) -> Tuple:
    return (r.get("phase"), r.get("iteration"), r.get("attempt_ordinal"),
            r.get("eval_round_ordinal"), r.get("eval_episode_index"), r.get("seed"))


def classify_episode(new: Dict[str, Any], orig: Dict[str, Any]) -> Dict[str, Any]:
    """The first difference of one episode outcome pair, with its side."""
    n, o = normalize(new), normalize(orig)
    wn, wo = n.get("wake_decisions") or [], o.get("wake_decisions") or []
    for i, (a, b) in enumerate(zip(wn, wo)):
        d = first_difference(a, b, "/wake_decisions[%d]" % i)
        if d is None:
            continue
        inputs_equal = all(first_difference(a.get(f), b.get(f)) is None
                           for f in _WAKE_INPUT_FIELDS)
        return {"side": "policy_side" if inputs_equal else "input_side",
                "wake_index": i, "tick": a.get("tick"), "wake_kind": a.get("wake_kind"),
                "difference": d}
    # no wake-level difference among the common wakes: episode level (physics / world / count)
    rest_n = {k: v for k, v in n.items() if k != "wake_decisions"}
    rest_o = {k: v for k, v in o.items() if k != "wake_decisions"}
    d = first_difference(rest_n, rest_o) or first_difference(wn, wo, "/wake_decisions")
    return {"side": "input_side", "wake_index": None, "difference": d}


def load_original(orig_dir: Path) -> Dict[str, Any]:
    """The original run's streams, keyed; loaded once and reusable across comparisons."""
    return {
        "out": {_outcome_key(r): r
                for r in read_complete_lines(orig_dir / "episode_outcomes.jsonl")},
        "cred": {(r["iteration"], r["batch_transition_ordinal"]): r
                 for r in read_complete_lines(orig_dir / "train_credit_diagnostics.jsonl")},
        "train": {r["iteration"]: r
                  for r in read_complete_lines(orig_dir / "train_records.jsonl")},
        "eval": {r.get("round_ordinal", r.get("eval_round_ordinal", i)): r
                 for i, r in enumerate(read_complete_lines(orig_dir / "eval_records.jsonl"))},
        "fail": {(r.get("phase"), r.get("iteration"), r.get("attempt_ordinal"), r.get("seed")): r
                 for r in read_complete_lines(orig_dir / "episode_failures.jsonl")},
    }


def compare(new_dir: Path, orig_dir: Path, *, final: bool = False,
            original: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    o = original if original is not None else load_original(orig_dir)
    orig_out, orig_cred, orig_train, orig_eval, orig_fail = (
        o["out"], o["cred"], o["train"], o["eval"], o["fail"])
    new_out = read_complete_lines(new_dir / "episode_outcomes.jsonl")
    new_cred = read_complete_lines(new_dir / "train_credit_diagnostics.jsonl")
    new_train = read_complete_lines(new_dir / "train_records.jsonl")
    new_eval = read_complete_lines(new_dir / "eval_records.jsonl")
    new_fail = read_complete_lines(new_dir / "episode_failures.jsonl")

    # ordered stream of comparable items in the NEW run's order
    cred_by_iter: Dict[int, List[Dict[str, Any]]] = {}
    for r in new_cred:
        cred_by_iter.setdefault(int(r["iteration"]), []).append(r)
    train_by_iter = {int(r["iteration"]): r for r in new_train}
    items: List[Tuple[str, Any, Dict[str, Any], Optional[Dict[str, Any]]]] = []
    last_iter = None
    for r in new_out:
        it = r.get("iteration")
        if r.get("phase") != "train" and last_iter is not None:
            # an evaluation round after training iteration `last_iter`: that update came first
            items.extend(_update_items(last_iter, cred_by_iter, orig_cred, train_by_iter,
                                       orig_train))
            last_iter = None
        if r.get("phase") == "train":
            if last_iter is not None and it != last_iter:
                items.extend(_update_items(last_iter, cred_by_iter, orig_cred, train_by_iter,
                                           orig_train))
            last_iter = it
        items.append(("outcome", _outcome_key(r), r, orig_out.get(_outcome_key(r))))
    if last_iter is not None and last_iter in train_by_iter:
        items.extend(_update_items(last_iter, cred_by_iter, orig_cred, train_by_iter, orig_train))

    counts = {"outcome": [0, 0], "credit": [0, 0], "train_record": [0, 0]}
    first = None
    for kind, key, new, orig in items:
        counts[kind][0] += 1
        if orig is None:
            d = {"side": "input_side", "difference": {"path": "<record absent in original>"}}
        elif kind == "outcome":
            same = first_difference(normalize(new), normalize(orig)) is None
            d = None if same else classify_episode(new, orig)
        else:
            diff = first_difference(normalize(new), normalize(orig))
            d = None if diff is None else {
                "side": ("policy_side" if kind == "credit" and diff["path"] in (
                    "/stored_log_prob", "/selected_meta_action", "/selected_meta_action_name",
                    "/selected_node") else "input_side"),
                "difference": diff}
        if d is None:
            counts[kind][1] += 1
        elif first is None:
            first = dict(d, record_kind=kind, key=list(key) if isinstance(key, tuple) else key)
    failures_compared = sum(1 for r in new_fail)
    failures_equal = sum(
        1 for r in new_fail
        if first_difference(normalize(r), normalize(orig_fail.get(
            (r.get("phase"), r.get("iteration"), r.get("attempt_ordinal"), r.get("seed")), {}))
            ) is None)
    evals = []
    for i, r in enumerate(new_eval):
        key = r.get("round_ordinal", r.get("eval_round_ordinal", i))
        o = orig_eval.get(key)
        diff = None if o is None else first_difference(normalize(r), normalize(o))
        evals.append({"round": key, "updates_completed": r.get("updates_completed"),
                      "identical": o is not None and diff is None,
                      "first_difference": diff if o is not None else "<absent in original>"})
    report = {
        "record": "prefix_comparison", "record_version": 1, "final": bool(final),
        "tolerance": 0.0,
        "counts_compared_identical": {k: {"compared": v[0], "identical": v[1]}
                                      for k, v in counts.items()},
        "failures": {"compared": failures_compared, "identical": failures_equal},
        "eval_records": evals,
        "first_divergence": first,
        "stop": bool(first and first.get("side") == "policy_side"),
    }
    return report


def _update_items(it, cred_by_iter, orig_cred, train_by_iter, orig_train):
    out = []
    for r in cred_by_iter.get(int(it), []):
        k = (r["iteration"], r["batch_transition_ordinal"])
        out.append(("credit", k, r, orig_cred.get(k)))
    if int(it) in train_by_iter:
        out.append(("train_record", int(it), train_by_iter[int(it)], orig_train.get(int(it))))
    return out


def compare_checkpoints(new_dir: Path, orig_dir: Path, iterations: Iterable[int]) -> List[Dict]:
    """Tensor equality of the saved actor / optimizer state (final report only; needs torch)."""
    import torch
    rows = []
    for it in iterations:
        name = "ckpt_iter%04d.pt" % it
        a_p, b_p = new_dir / "checkpoints" / name, orig_dir / "checkpoints" / name
        if not (a_p.exists() and b_p.exists()):
            rows.append({"checkpoint": name, "present": [a_p.exists(), b_p.exists()]})
            continue
        a = torch.load(a_p, map_location="cpu", weights_only=False)
        b = torch.load(b_p, map_location="cpu", weights_only=False)
        eq = {}
        for part in ("encoder", "head"):
            eq[part] = all(torch.equal(a[part][k], b[part][k]) for k in a[part]) and \
                a[part].keys() == b[part].keys()
        sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
        eq["optimizer_state"] = sa.keys() == sb.keys() and all(
            torch.equal(sa[k][f], sb[k][f]) if torch.is_tensor(sa[k][f]) else sa[k][f] == sb[k][f]
            for k in sa for f in sa[k])
        max_abs = 0.0
        for part in ("encoder", "head"):
            for k in a[part]:
                max_abs = max(max_abs, float((a[part][k].double() - b[part][k].double())
                                             .abs().max()))
        rows.append({"checkpoint": name, "identical": eq, "max_abs_parameter_diff": max_abs})
    return rows


if __name__ == "__main__":  # final report
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoints", default="24,49,74,99")
    args = ap.parse_args()
    rep = compare(Path(args.new), Path(args.original), final=True)
    rep["checkpoints"] = compare_checkpoints(
        Path(args.new), Path(args.original),
        [int(x) for x in args.checkpoints.split(",") if x])
    Path(args.out).write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("counts_compared_identical", "first_divergence",
                                          "stop")}, indent=1)[:4000])
