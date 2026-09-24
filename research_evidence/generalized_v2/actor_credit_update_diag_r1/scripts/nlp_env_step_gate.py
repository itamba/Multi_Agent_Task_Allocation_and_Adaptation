"""Engineering gate of the actor-step instrumentation UNDER THE SCIENTIFIC ENVIRONMENT (nlp_env).

``pytest`` is absent from ``nlp_env`` (environments_cleanup.md §1), and nlp_env's torch differs
from the base env's, so the load-bearing checks of ``tests/test_graph_actor_step_diagnostics.py``
are repeated here without pytest, on synthetic data only (no BLADE episode, no solver):

  G1 BASE (``PPOUpdater`` compiled from the verified base commit) == OFF == ON over two
     consecutive updates with both advantage signs and later-epoch clipping: parameters,
     Adam state, updater outputs, torch / numpy RNG;
  G2 every epoch: ``_actor_step_record`` accepts the report (its residual / attribution /
     finiteness checks are the declared ones), the displacement equals the parameter change
     around the real ``Adam.step`` and the clipped gradient equals what Adam consumed, and the
     epoch deltas telescope;
  G3 the real trainer (stubbed episodes with real immediate-FD wakes) ON vs OFF: identical
     training records (timing removed), credit rows, outcomes and checkpoints; one step record
     per productive update, four epochs each.

Usage (from the measured worktree):
    conda run -n nlp_env --no-capture-output python <this> --base-sha <sha> --out <json>
"""

from __future__ import annotations

import __future__
import argparse
import ast
import copy
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from match_aou.rl.action.graph_action import (MetaAction, build_action_mask, evaluate_action,
                                              sample_action)
from match_aou.rl.observation.graph_builder import EdgeType, GraphObservation
from match_aou.rl.training import graph_ppo as GP
from match_aou.rl.training import graph_tick_loop as TL
from match_aou.rl.training import graph_train as GT
from match_aou.rl.training.graph_fuel_damage import resolve_condition, resolve_severity
from match_aou.rl.training.graph_ppo import EpisodeRecord, PPOConfig, PPOUpdater

IMM, POSTK, ORDK = TL.WAKE_KIND_IMMEDIATE_FD, TL.WAKE_KIND_POST_FD_BOUNDARY, TL.WAKE_KIND_ORDINARY
ABORT, PLAN = int(MetaAction.SELF_PRESERVATION_ABORT), int(MetaAction.PLAN_COMPLIANCE)
HIGH_LR = PPOConfig(lr=0.03, n_epochs=4)


def _row(u):
    return [u, 0.3, 1.0, 1.0, 1.0, 1.0]


def _obs(rows, ego_nodes=(0, 1)):
    k = len(rows)
    return GraphObservation(
        task_features=np.asarray(rows, dtype=np.float32),
        agent_features=np.array([[0.9, 0.1], [0.0, 0.0]], dtype=np.float32),
        ego_index=k,
        edge_index=np.array([[k] * len(ego_nodes), list(ego_nodes)], dtype=np.int64),
        edge_type=np.full((len(ego_nodes),), int(EdgeType.ASSIGNMENT), dtype=np.int64),
        task_target_ids=["t%d" % i for i in range(k)], agent_ids=["ego", "peer"],
        agent_id="ego", current_time=5, time_norm=0.05)


def _transition(policy, gobs, ego, tick, seed, force=None):
    mask = build_action_mask(gobs)
    with torch.no_grad():
        logits = policy.head(policy.encoder(gobs))
        if force is None:
            torch.manual_seed(seed)
            meta, node, lp, ent = sample_action(logits, mask)
        else:
            meta, node = force, None
            lp, ent = evaluate_action(logits, mask, meta, node)
    return TL.Transition(gobs=gobs, ego_id=ego, tick=tick, meta_action=int(meta), node_v=node,
                         log_prob=float(lp.item()), entropy=float(ent.item()))


SPECS = [
    (0, 500, -0.15, [("egoA", ORDK, None), ("egoB", ORDK, None), ("egoA", IMM, ABORT),
                     ("egoB", ORDK, None), ("egoA", POSTK, None)]),
    (1, 501, -0.9, []),
    (2, 502, -0.7, [("egoB", ORDK, None), ("egoA", IMM, PLAN), ("egoB", ORDK, None),
                    ("egoA", POSTK, None)]),
    (3, 503, -0.05, [("egoA", ORDK, None), ("egoB", ORDK, None)]),
    (4, 504, -0.55, [("egoB", IMM, PLAN), ("egoA", ORDK, None)]),
]
TAGS = {
    (0, 500): {"severity": "severe", "fd_selected_ego_id": "egoA"},
    (1, 501): {"severity": None, "fd_selected_ego_id": None},
    (2, 502): {"severity": "mild", "fd_selected_ego_id": "egoA"},
    (3, 503): {"severity": None, "fd_selected_ego_id": None},
    (4, 504): {"severity": "mild", "fd_selected_ego_id": "egoB"},
}


def _records(policy):
    out = []
    for e, seed, reward, wakes in SPECS:
        traj = []
        for t, (ego, kind, force) in enumerate(wakes):
            gobs = _obs([_row(0.8 - 0.07 * e - 0.02 * t), _row(0.6 + 0.03 * t),
                         _row(0.5 + 0.05 * e), _row(0.7)])
            tr = _transition(policy, gobs, ego, t + 1, 19 * e + t, force)
            tr.wake_kind = kind
            traj.append(tr)
        if traj:
            traj[-1].reward = reward
        out.append(EpisodeRecord.from_trajectory(traj, reward, seed=seed, episode_index=e))
    return out


def _base_class(base_sha):
    src = subprocess.run(["git", "show", "%s:src/match_aou/rl/training/graph_ppo.py" % base_sha],
                         capture_output=True, text=True, check=True).stdout
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.ClassDef) and n.name == "PPOUpdater")
    ns = dict(vars(GP))
    exec(compile(ast.get_source_segment(src, node), "<base PPOUpdater>", "exec",
                 flags=__future__.annotations.compiler_flag, dont_inherit=True), ns)
    return ns["PPOUpdater"]


def _state_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _optim_equal(a, b):
    sa, sb = a.state_dict()["state"], b.state_dict()["state"]
    return sa.keys() == sb.keys() and all(
        torch.equal(sa[k][f], sb[k][f]) for k in sa for f in sa[k])


def _flat(policy):
    return np.concatenate([p.detach().reshape(-1).double().numpy()
                           for p in list(policy.encoder.parameters())
                           + list(policy.head.parameters())])


def gate_g1_g2(base_sha):
    torch.manual_seed(31)
    policy0 = TL.build_policy()
    records = _records(policy0)
    ids = GT._actor_step_group_ids(records, TAGS)
    batch = GP.compute_returns_and_advantages(records, HIGH_LR)
    runs = {}
    observed = []
    real_step = torch.optim.Adam.step
    for name, cls, on in (("base", _base_class(base_sha), False), ("off", PPOUpdater, False),
                          ("on", PPOUpdater, True)):
        policy = copy.deepcopy(policy0)
        upd = cls(policy, HIGH_LR)
        reports, diags = [], []
        if on:
            def spy(self, *a, **k):
                before = _flat(policy)
                g = np.concatenate([np.zeros(p.numel()) if p.grad is None else
                                    p.grad.detach().reshape(-1).double().numpy()
                                    for p in upd.parameters])
                r = real_step(self, *a, **k)
                observed.append((g, _flat(policy) - before))
                return r
            torch.optim.Adam.step = spy
        torch.manual_seed(11)
        np.random.seed(11)
        try:
            for _ in range(2):
                kw = ({"step_group_ids": ids, "step_sink": reports.append,
                       "step_contrast_ids": GT._actor_gradient_contrast_ids()} if on else {})
                diags.append(upd.update(records, **kw))
        finally:
            torch.optim.Adam.step = real_step
        runs[name] = dict(policy=policy, upd=upd, diags=diags, reports=reports,
                          rng=torch.get_rng_state().clone(),
                          np_rng=np.random.get_state()[1].copy())
    b = runs["base"]
    g1 = {"advantages_both_signs": bool((batch.advantages > 0).any()
                                        and (batch.advantages < 0).any()),
          "later_epoch_clip_fraction_max": max(b["diags"][0]["per_epoch"]["clip_fraction"][1:])}
    for name in ("off", "on"):
        r = runs[name]
        g1[name] = {
            "diagnostics_equal": r["diags"] == b["diags"],
            "encoder_equal": _state_equal(r["policy"].encoder, b["policy"].encoder),
            "head_equal": _state_equal(r["policy"].head, b["policy"].head),
            "adam_state_equal": _optim_equal(r["upd"].optimizer, b["upd"].optimizer),
            "torch_rng_equal": bool(torch.equal(r["rng"], b["rng"])),
            "numpy_rng_equal": bool(np.array_equal(r["np_rng"], b["np_rng"])),
        }
    g1["pass"] = (g1["advantages_both_signs"] and g1["later_epoch_clip_fraction_max"] > 0
                  and all(all(v.values()) for k, v in g1.items() if k in ("off", "on")))
    on = runs["on"]
    epochs = [e for rep in on["reports"] for e in rep.epochs]
    g2 = {"n_reports": len(on["reports"]), "n_epochs": len(epochs),
          "n_adam_steps_observed": len(observed)}
    g2["delta_is_actual_adam_step"] = len(observed) == len(epochs) and all(
        np.array_equal(e.delta_theta, d) and np.array_equal(e.clipped_grad, g)
        for e, (g, d) in zip(epochs, observed))
    summaries = []
    for rep, diag in zip(on["reports"], on["diags"]):
        rec = GT._actor_step_record(rep, diag, iteration=0, updates_completed_before=0,
                                    measurement_tags=TAGS)   # raises on any declared fault
        s = rec["update_summary"]
        summaries.append({"telescoping_residual": s["telescoping_residual"],
                          "max_epoch_boundary_mismatch": s["max_epoch_boundary_mismatch"],
                          "max_group_sum_relative_residual": max(
                              ep["gradient"]["group_sum_relative_residual"]
                              for ep in rec["epochs"]),
                          "max_total_vs_backward_relative_residual": max(
                              ep["gradient"]["total_loss_vs_backward_relative_residual"]
                              for ep in rec["epochs"])})
        json.dumps(rec, allow_nan=False)
    g2["records"] = summaries
    g2["pass"] = (g2["delta_is_actual_adam_step"] and len(on["reports"]) == 2
                  and all(abs(x["telescoping_residual"]) <= 1e-12
                          and x["max_epoch_boundary_mismatch"] <= 1e-7 for x in summaries))
    return g1, g2


def _fd_stub(policy_unused=None):
    def fake(policy, gen, cfg_, *, seed, episode_tag, deterministic, fuel_damage_mode=None,
             **kwargs):
        params = cfg_.fuel_damage_parameters(fuel_damage_mode)
        condition = resolve_condition(episode_seed=seed, params=params)
        severity = resolve_severity(episode_seed=seed, params=params)
        damaged = condition == GT.CONDITION_DAMAGED
        kinds = [ORDK, IMM, POSTK] if damaged else [ORDK, ORDK]
        traj = []
        for t, kind in enumerate(kinds):
            gobs = _obs([_row(0.8 - 0.01 * (seed % 11)), _row(0.6 + 0.02 * t), _row(0.5),
                         _row(0.7)])
            tr = _transition(policy, gobs, "a0", t + 1, seed * 7 + t)
            tr.wake_kind = kind
            traj.append(tr)
        reward = -0.1 - 0.05 * (seed % 9)
        traj[-1].reward = reward
        return GT._EpisodeOutcome(
            trajectory=traj, reward=reward, ticks=10, ended="done", n_wakes=len(traj),
            confirmed_kills=1, n_dead=0, seconds=0.01, targets_confirmed_unique=1,
            targets_total=6, known_target_names=("A",), hidden_target_names=("B",),
            known_confirmed_names=("A",), hidden_confirmed_names=(),
            fuel_damage_plan={"condition": condition, "severity": severity,
                              "ego_id": "a0" if damaged else None},
            fuel_damage_outcome={"condition": condition, "severity": severity,
                                 "fired": damaged, "wake_occurred": damaged,
                                 "wake_meta_action": None,
                                 "event_tick": 2 if damaged else None},
            selected_ego_rtb_issued=None)
    return fake


def _lines(p):
    return [json.loads(x) for x in p.read_text("utf-8").splitlines() if x]


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items()
                if not any(t in k for t in ("seconds", "elapsed", "timestamp", "_at", "wall"))}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def gate_g3():
    tmp = Path(tempfile.mkdtemp(prefix="stepgate_"))
    saved = {n: getattr(GT, n) for n in ("_run_one_episode", "_build_generator",
                                         "_git_provenance")}
    GT._git_provenance = lambda repo_root: {
        "repo_root": str(repo_root), "available": True, "commit": "0" * 40, "branch": "gate",
        "dirty": False, "dirty_path_count": 0, "reason": None}
    GT._run_one_episode = _fd_stub()
    GT._build_generator = lambda _d: object()
    dirs = {}
    try:
        for on in (False, True):
            cfg = GT.TrainConfig(n_iterations=3, episodes_per_iteration=4, eval_every=0,
                                 eval_episodes=0, output_dir=str(tmp / str(on)),
                                 fuel_damage_mode="seeded_variable", checkpoint_every=1,
                                 actor_step_diagnostics=on,
                                 actor_step_vector_iterations=(0, 2) if on else ())
            GT.train(cfg)
            dirs[on] = Path(cfg.output_dir)
    finally:
        for n, v in saved.items():
            setattr(GT, n, v)
    off, on = dirs[False], dirs[True]
    res = {
        "train_records_equal": _strip(_lines(off / "train_records.jsonl"))
        == _strip(_lines(on / "train_records.jsonl")),
        "credit_rows_equal": _lines(off / "train_credit_diagnostics.jsonl")
        == _lines(on / "train_credit_diagnostics.jsonl"),
        "outcomes_equal": _strip(_lines(off / "episode_outcomes.jsonl"))
        == _strip(_lines(on / "episode_outcomes.jsonl")),
        "off_wrote_no_step_file": not (off / "train_actor_step_diagnostics.jsonl").exists(),
    }
    ck = True
    for name in sorted(p.name for p in (off / "checkpoints").iterdir()):
        a = torch.load(off / "checkpoints" / name, weights_only=False)
        b = torch.load(on / "checkpoints" / name, weights_only=False)
        ck &= all(torch.equal(a[m][k], b[m][k]) for m in ("encoder", "head") for k in a[m])
        sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
        ck &= all(torch.equal(sa[k][f], sb[k][f]) for k in sa for f in sa[k])
    res["checkpoints_equal"] = bool(ck)
    recs = _lines(on / "train_actor_step_diagnostics.jsonl")
    res["n_step_records"] = len(recs)
    res["epochs_per_record"] = [len(r["epochs"]) for r in recs]
    res["contrast_defined_updates"] = sum(1 for r in recs
                                          if r["update_summary"]["contrast_defined"])
    res["vector_files"] = [r["vector_file"]["path"] if r["vector_file"] else None for r in recs]
    res["pass"] = (all(v for k, v in res.items() if isinstance(v, bool))
                   and res["n_step_records"] == 3 and res["epochs_per_record"] == [4, 4, 4]
                   and res["contrast_defined_updates"] >= 1
                   and res["vector_files"][1] is None and None not in res["vector_files"][::2])
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sha", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    g1, g2 = gate_g1_g2(args.base_sha)
    g3 = gate_g3()
    report = {"record": "nlp_env_step_gate", "record_version": 1,
              "python": sys.version.split()[0], "platform": platform.platform(),
              "torch": torch.__version__, "numpy": np.__version__,
              "torch_num_threads": torch.get_num_threads(),
              "graph_ppo_file": GP.__file__, "graph_train_file": GT.__file__,
              "base_sha": args.base_sha, "G1": g1, "G2": g2, "G3": g3,
              "pass": bool(g1["pass"] and g2["pass"] and g3["pass"])}
    Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("PASS" if report["pass"] else "FAIL")
    print(json.dumps({"G1": g1["pass"], "G2": g2["pass"], "G3": g3["pass"]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
