"""Deterministic evidence extraction for the credit-to-update diagnostic R1.

Reads the NEW run directory, its launcher directory and the ORIGINAL mission-slack run (read
only) and writes, under the evidence package:

  run_artifacts/                byte-identical copies of the small run / accounting artifacts,
                                the COMPLETE scalar diagnostic stream, the five epoch-0 vector
                                files and the launcher files (never normalized)
  extracted/per_update_diagnostic.jsonl   one compact row per productive update (EVERY update)
  extracted/primary_diagnostic.json       the declared window summaries (plan: primary_diagnostic)
  extracted/per_epoch_summary.json        the same quantities by epoch index 0..3
  extracted/secondary_behaviour.json      the five matched development rounds, new vs original
  extracted/accounting.json               completion and attempt accounting
  extracted/vector_spot_checks.json       scalar re-computation from the unrounded vectors
  extracted/per_update_table.md           a readable table of every update
  artifact_sha256.txt                     every source artifact: SHA-256, bytes, path

``--check`` re-runs the extraction into a scratch directory and requires byte-identical
extracted files. numpy is used only to read the .npz vectors, elementwise (no BLAS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

NEG = 1e-6          # the plan's numerically-negligible reporting scale (not a threshold)
WINDOWS = [(0, 24), (25, 49), (50, 74), (75, 99)]
GROUPS = ("immediate_fd_mild", "immediate_fd_severe", "post_fd", "ordinary")
SMALL_RUN_FILES = ("run_config.json", "run_summary.json", "train_records.jsonl",
                   "eval_records.jsonl", "episode_failures.jsonl",
                   "train_actor_step_diagnostics.jsonl", "train_credit_diagnostics.jsonl")
LAUNCHER_FILES = ("launcher_record.json", "native_exit_code.txt", "invocation_start_local.txt",
                  "prefix_monitor.json", "prefix_first_divergence.json",
                  "prefix_monitor_errors.log", "training_console.log")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lines(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def dist(values):
    values = list(values)
    v = sorted(float(x) for x in values if x is not None)
    if not v:
        return None if not values else {"n": 0, "n_null": len(values)}

    def q(p):
        pos = (len(v) - 1) * p
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)
    return {"n": len(v), "mean": math.fsum(v) / len(v), "q10": q(0.1), "median": q(0.5),
            "q90": q(0.9), "min": v[0], "max": v[-1],
            "mean_abs": math.fsum(abs(x) for x in v) / len(v),
            "n_positive": sum(1 for x in v if x >= NEG),
            "n_negative": sum(1 for x in v if x <= -NEG),
            "n_negligible": sum(1 for x in v if abs(x) < NEG),
            "n_null": len(values) - len(v)}


def sgn(x):
    return 0 if abs(x) < NEG else (1 if x > 0 else -1)


def per_update_rows(records):
    rows = []
    for r in records:
        s = r["update_summary"]
        eps = []
        for ep in r["epochs"]:
            c = ep["contrast"]
            st = ep["step"]
            row = {"epoch": ep["epoch"], "pre_clip_grad_norm": st["pre_clip_grad_norm"],
                   "post_clip_grad_norm": st["post_clip_grad_norm"], "clipped": st["clipped"],
                   "delta_theta_norm": st["delta_theta_norm"],
                   "cosine_delta_vs_neg_total_loss_grad": st["cosine_delta_vs_neg_total_loss_grad"],
                   "fd_clip_binding": (ep["ratio"]["derived"]["fd"] or {}).get(
                       "fraction_clip_binding"),
                   "all_clip_binding": ep["ratio"]["all"]["fraction_clip_binding"]}
            if c is not None:
                p = c["pressure"]
                row.update({
                    "contrast_before": c["before"], "contrast_after": c["after"],
                    "p_abort_severe_before": c["p_abort_mean_immediate_fd_severe_before"],
                    "p_abort_mild_before": c["p_abort_mean_immediate_fd_mild_before"],
                    "h_norm": c["grad_norm"],
                    **{"pressure_" + g: p["groups"][g]["pressure"] for g in GROUPS},
                    "pressure_fd": p["derived"]["fd"]["pressure"],
                    "pressure_non_fd": p["derived"]["non_fd"]["pressure"],
                    "pressure_entropy": p["entropy_component"]["pressure"],
                    "pressure_policy_surrogate": p["policy_surrogate"]["pressure"],
                    "pressure_total": p["total_loss"]["pressure"],
                    "alignment_fd": p["derived"]["fd"]["alignment"],
                    "alignment_total": p["total_loss"]["alignment"],
                    "predicted_delta": c["predicted_delta_from_displacement"],
                    "actual_delta": c["actual_delta"],
                    "linearization_residual": c["linearization_residual"],
                    "cosine_delta_vs_h": c["cosine_delta_vs_contrast_grad"]})
            eps.append(row)
        rows.append({
            "iteration": r["iteration"], "n_transitions": r["batch_n_transitions"],
            "counts": {g: r["groups"][g]["n_transitions"] for g in GROUPS},
            "fd_actions": {g: r["groups"][g]["selected_meta_action_counts"]
                           for g in GROUPS[:2]},
            "fd_normalized_advantage_mean": {
                g: (r["groups"][g]["normalized_advantage"] or {}).get("mean")
                for g in GROUPS[:2]},
            "contrast_defined": s["contrast_defined"],
            "contrast_undefined_reason": s["contrast_undefined_reason"],
            "contrast_before_update": s["contrast_before_update"],
            "contrast_after_update": s["contrast_after_update"],
            "full_update_delta": s["full_update_delta"],
            "epoch_deltas": s["epoch_deltas"],
            "sum_first_order_predictions": s["sum_first_order_predictions"],
            "sum_linearization_residuals": s["sum_linearization_residuals"],
            "telescoping_residual": s["telescoping_residual"],
            "max_epoch_boundary_mismatch": s["max_epoch_boundary_mismatch"],
            "full_displacement_norm": s["full_displacement_norm"],
            "epochs": eps})
    return rows


def window_summary(rows, lo, hi):
    win = [r for r in rows if lo <= r["iteration"] <= hi]
    d = [r for r in win if r["contrast_defined"]]
    e0 = [r["epochs"][0] for r in d]
    allep = [ep for r in d for ep in r["epochs"]]
    out = {
        "iterations": [lo, hi], "n_updates": len(win), "n_defined": len(d),
        "n_undefined": len(win) - len(d),
        "undefined_reasons": sorted({r["contrast_undefined_reason"] for r in win
                                     if not r["contrast_defined"]}),
        "full_update_delta": dist([r["full_update_delta"] for r in d]),
        "sum_first_order_predictions": dist([r["sum_first_order_predictions"] for r in d]),
        "sum_linearization_residuals": dist([r["sum_linearization_residuals"] for r in d]),
        "contrast_before_update": dist([r["contrast_before_update"] for r in d]),
        "epoch0_pressure": {k: dist([ep["pressure_" + k] for ep in e0])
                            for k in ("fd", "non_fd", "entropy", "policy_surrogate", "total")
                            + GROUPS},
        "all_epoch_pressure": {k: dist([ep["pressure_" + k] for ep in allep])
                               for k in ("fd", "non_fd", "entropy", "total")},
        "epoch_predicted_delta": dist([ep["predicted_delta"] for ep in allep]),
        "epoch_actual_delta": dist([ep["actual_delta"] for ep in allep]),
        "h_norm_epoch0": dist([ep["h_norm"] for ep in e0]),
        "counts": {
            "epoch0_fd_positive": sum(1 for ep in e0 if sgn(ep["pressure_fd"]) > 0),
            "epoch0_fd_positive_non_fd_negative": sum(
                1 for ep in e0 if sgn(ep["pressure_fd"]) > 0 and sgn(ep["pressure_non_fd"]) < 0),
            "epoch0_fd_positive_total_negative": sum(
                1 for ep in e0 if sgn(ep["pressure_fd"]) > 0 and sgn(ep["pressure_total"]) < 0),
            "epoch0_entropy_flips_total_sign": sum(
                1 for ep in e0 if sgn(ep["pressure_policy_surrogate"]) != 0
                and sgn(ep["pressure_total"]) != sgn(ep["pressure_policy_surrogate"])),
            "epochs_raw_total_vs_adam_prediction_sign_agree": sum(
                1 for ep in allep if sgn(ep["pressure_total"]) != 0
                and sgn(ep["pressure_total"]) == sgn(ep["predicted_delta"])),
            "epochs_raw_total_nonzero": sum(1 for ep in allep if sgn(ep["pressure_total"]) != 0),
            "epochs_prediction_vs_actual_sign_agree": sum(
                1 for ep in allep if sgn(ep["actual_delta"]) != 0
                and sgn(ep["actual_delta"]) == sgn(ep["predicted_delta"])),
            "epochs_actual_nonzero": sum(1 for ep in allep if sgn(ep["actual_delta"]) != 0),
            "epochs_clipped": sum(1 for ep in allep if ep["clipped"]),
            "n_epochs": len(allep),
            "full_update_delta_positive_negative_negligible": [
                sum(1 for r in d if sgn(r["full_update_delta"]) > 0),
                sum(1 for r in d if sgn(r["full_update_delta"]) < 0),
                sum(1 for r in d if sgn(r["full_update_delta"]) == 0)],
        },
    }
    return out


def secondary(new_eval, orig_eval):
    orig_by = {r["eval_round_ordinal"]: r for r in orig_eval}
    rows = []
    for r in new_eval:
        o = orig_by.get(r["eval_round_ordinal"])

        def pick(ev):
            vb = ev["v2_behaviour"]
            elig = [g for g in vb["groups"] if g["metric_eligible"]]
            return {"updates_completed": ev["updates_completed"],
                    "evaluation_stage": ev["evaluation_stage"],
                    "macro_severe_minus_mild": vb["macro_mean_over_base_cells"],
                    "macro_n_base_cells_defined": vb["macro_n_base_cells_defined"],
                    "n_groups_metric_eligible": vb["n_groups_metric_eligible"],
                    "n_groups_attempted": vb["n_groups_attempted"],
                    "directional_switches": vb["directional_switch_count"],
                    "reverse_switches": vb["reverse_switch_count"],
                    "mean_p_abort_mild": (math.fsum(g["p_abort_mild"] for g in elig) / len(elig)
                                          if elig else None),
                    "mean_p_abort_severe": (math.fsum(g["p_abort_severe"] for g in elig)
                                            / len(elig) if elig else None),
                    "n_attempted": ev["n_attempted"], "n_successful": ev["n_successful"],
                    "n_failed": ev["n_failed"], "eval_reward_mean": ev["eval_reward_mean"],
                    "eval_reward_mean_clean": ev.get("eval_reward_mean_clean"),
                    "eval_reward_mean_mild": ev.get("eval_reward_mean_mild"),
                    "eval_reward_mean_severe": ev.get("eval_reward_mean_severe"),
                    "eval_deaths": ev.get("eval_deaths"),
                    "by_base_cell": {k: v["severe_minus_mild_abort_mass_mean"]
                                     for k, v in vb["by_base_cell"].items()}}
        a, b = pick(r), (pick(o) if o else None)
        rows.append({"eval_round_ordinal": r["eval_round_ordinal"], "new": a, "original": b,
                     "macro_new_minus_original": (None if b is None else
                                                  a["macro_severe_minus_mild"]
                                                  - b["macro_severe_minus_mild"])})
    return rows


def vector_spot_checks(run_dir, records):
    import numpy as np
    out = []
    for r in records:
        vf = r["vector_file"]
        if not vf:
            continue
        p = run_dir / vf["path"]
        row = {"iteration": r["iteration"], "path": vf["path"], "sha256_recorded": vf["sha256"],
               "sha256_actual": sha(p), "bytes": p.stat().st_size}
        ep = r["epochs"][0]
        with np.load(p, allow_pickle=False) as z:
            layout = json.loads(str(z["layout_json"]))
            n = sum(int(np.prod(e["shape"])) for e in layout)
            row["layout_entries"] = len(layout)
            row["n_parameters"] = n
            row["layout_matches_record"] = n == r["n_actor_parameters"]

            def dot(a, b):
                return float(np.sum(a * b))
            g = {k: z["group_" + k] for k in GROUPS}
            tot, pol, delta = z["total_loss_grad"], z["policy_surrogate_grad"], z["delta_theta"]
            row["recomputed"] = {
                "delta_theta_norm": [dot(delta, delta) ** 0.5, ep["step"]["delta_theta_norm"]],
                "total_loss_grad_norm": [dot(tot, tot) ** 0.5,
                                         ep["gradient"]["total_loss_grad_norm"]],
                "group_sum_residual_norm": [dot(pol - sum(g.values()), pol - sum(g.values()))
                                            ** 0.5, ep["gradient"]["group_sum_residual_norm"]],
            }
            if "contrast_grad" in z.files:
                h = z["contrast_grad"]
                c = ep["contrast"]
                row["recomputed"].update({
                    "h_norm": [dot(h, h) ** 0.5, c["grad_norm"]],
                    "pressure_total": [-dot(h, tot), c["pressure"]["total_loss"]["pressure"]],
                    "pressure_fd": [-dot(h, g["immediate_fd_mild"] + g["immediate_fd_severe"]),
                                    c["pressure"]["derived"]["fd"]["pressure"]],
                    "predicted_delta": [dot(h, delta), c["predicted_delta_from_displacement"]],
                })
            else:
                row["contrast_grad_absent_reason"] = vf["contrast_grad_absent_reason"]
        row["max_relative_mismatch"] = max(
            abs(a - b) / max(abs(b), 1e-300) if b != 0 else abs(a)
            for a, b in row["recomputed"].values())
        row["pass"] = (row["sha256_recorded"] == row["sha256_actual"]
                       and row["layout_matches_record"] and row["max_relative_mismatch"] < 1e-9)
        out.append(row)
    return out


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    return "%.*e" % (nd, x)


def table_md(rows):
    head = ("| it | n | MILD / SEVERE | C before | full ΔC | Σ pred | Σ resid | e0 FD press "
            "| e0 non-FD press | e0 entropy press | e0 total press | clipped epochs |\n"
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    body = []
    for r in rows:
        e0 = r["epochs"][0]
        body.append("| %d | %d | %d / %d | %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
            r["iteration"], r["n_transitions"], r["counts"]["immediate_fd_mild"],
            r["counts"]["immediate_fd_severe"], fmt(r["contrast_before_update"]),
            fmt(r["full_update_delta"]), fmt(r["sum_first_order_predictions"]),
            fmt(r["sum_linearization_residuals"]), fmt(e0.get("pressure_fd")),
            fmt(e0.get("pressure_non_fd")), fmt(e0.get("pressure_entropy")),
            fmt(e0.get("pressure_total")), sum(1 for ep in r["epochs"] if ep["clipped"])))
    return ("# Every update — actor-step diagnostic (values unrounded in "
            "per_update_diagnostic.jsonl)\n\n'—' = undefined (a severity absent from the batch).\n\n"
            + head + "\n".join(body) + "\n")


def extract(run_dir: Path, launcher_dir: Path, original: Path, pkg: Path, copy: bool) -> None:
    ext = pkg / "extracted"
    ext.mkdir(parents=True, exist_ok=True)
    records = lines(run_dir / "train_actor_step_diagnostics.jsonl")
    train = lines(run_dir / "train_records.jsonl")
    rows = per_update_rows(records)
    (ext / "per_update_diagnostic.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    productive = [t["iteration"] for t in train if t["n_epochs_run"]]
    primary = {
        "definition": "full-update actual delta C_B on updates holding both FD severities; "
                      "negligible = |x| < 1e-6 (reporting scale only); sign counts are never "
                      "read without the magnitudes",
        "n_productive_updates": len(productive),
        "n_step_records": len(records),
        "every_productive_update_has_one_record": [r["iteration"] for r in records]
        == productive,
        "n_defined": sum(1 for r in rows if r["contrast_defined"]),
        "undefined_iterations": [r["iteration"] for r in rows if not r["contrast_defined"]],
        "windows": [window_summary(rows, lo, hi) for lo, hi in WINDOWS],
        "all_updates": window_summary(rows, 0, 99),
        "integrity": {
            "max_abs_telescoping_residual": max(
                (abs(r["telescoping_residual"]) for r in rows if r["contrast_defined"]),
                default=None),
            "max_epoch_boundary_mismatch": max(
                (r["max_epoch_boundary_mismatch"] for r in rows if r["contrast_defined"]),
                default=None),
            "max_group_sum_relative_residual": max(
                ep["gradient"]["group_sum_relative_residual"] or 0.0
                for r in records for ep in r["epochs"]),
            "max_total_vs_backward_relative_residual": max(
                ep["gradient"]["total_loss_vs_backward_relative_residual"] or 0.0
                for r in records for ep in r["epochs"]),
        },
    }
    (ext / "primary_diagnostic.json").write_text(json.dumps(primary, indent=1, sort_keys=True),
                                                 encoding="utf-8")
    d = [r for r in rows if r["contrast_defined"]]
    per_epoch = {str(k): {
        "actual_delta": dist([r["epochs"][k]["actual_delta"] for r in d]),
        "predicted_delta": dist([r["epochs"][k]["predicted_delta"] for r in d]),
        "linearization_residual": dist([r["epochs"][k]["linearization_residual"] for r in d]),
        "pressure_fd": dist([r["epochs"][k]["pressure_fd"] for r in d]),
        "pressure_non_fd": dist([r["epochs"][k]["pressure_non_fd"] for r in d]),
        "pressure_total": dist([r["epochs"][k]["pressure_total"] for r in d]),
        "fd_clip_binding": dist([r["epochs"][k]["fd_clip_binding"] for r in d
                                 if r["epochs"][k]["fd_clip_binding"] is not None]),
        "n_clipped": sum(1 for r in rows if r["epochs"][k]["clipped"])}
        for k in range(4)}
    (ext / "per_epoch_summary.json").write_text(json.dumps(per_epoch, indent=1, sort_keys=True),
                                                encoding="utf-8")
    new_eval = lines(run_dir / "eval_records.jsonl")
    orig_eval = lines(original / "eval_records.jsonl")
    sec = secondary(new_eval, orig_eval)
    (ext / "secondary_behaviour.json").write_text(json.dumps(sec, indent=1, sort_keys=True),
                                                  encoding="utf-8")
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    fails = lines(run_dir / "episode_failures.jsonl")
    acc = {
        "n_train_records": len(train), "n_productive_updates": len(productive),
        "train_attempted": sum(t["n_attempted"] for t in train),
        "train_successful": sum(t["n_successful"] for t in train),
        "train_failed": sum(t["n_failed"] for t in train),
        "n_eval_rounds": len(new_eval),
        "eval_attempted": sum(e["n_attempted"] for e in new_eval),
        "eval_successful": sum(e["n_successful"] for e in new_eval),
        "failures": [{k: f.get(k) for k in ("phase", "iteration", "attempt_ordinal", "seed",
                                            "pipeline_stage", "error_type")} for f in fails],
        "accounting_reconciled": summary.get("accounting_reconciled",
                                             summary.get("accounting", {}).get("reconciled")),
        "optimizer_steps": sum(t["n_epochs_run"] for t in train),
    }
    (ext / "accounting.json").write_text(json.dumps(acc, indent=1, sort_keys=True),
                                         encoding="utf-8")
    (ext / "vector_spot_checks.json").write_text(
        json.dumps(vector_spot_checks(run_dir, records), indent=1, sort_keys=True),
        encoding="utf-8")
    (ext / "per_update_table.md").write_text(table_md(rows), encoding="utf-8")

    if not copy:
        return
    ledger = ["# sha256  bytes  group  absolute_path",
              "# Source artifacts of the evidence package; generated by scripts/extract_evidence.py"]
    dest = pkg / "run_artifacts"
    (dest / "launcher").mkdir(parents=True, exist_ok=True)
    (dest / "train_actor_step_vectors").mkdir(parents=True, exist_ok=True)
    for name in SMALL_RUN_FILES:
        src = run_dir / name
        shutil.copyfile(src, dest / name)
    for vf in sorted((run_dir / "train_actor_step_vectors").iterdir()):
        shutil.copyfile(vf, dest / "train_actor_step_vectors" / vf.name)
    for name in LAUNCHER_FILES:
        if (launcher_dir / name).exists():
            shutil.copyfile(launcher_dir / name, dest / "launcher" / name)
    for f in sorted(p for p in run_dir.rglob("*") if p.is_file()
                    and "scenarios" not in p.relative_to(run_dir).parts):
        ledger.append("%s  %d  run  %s" % (sha(f), f.stat().st_size, f))
    n_scen = sum(1 for p in (run_dir / "scenarios").rglob("*") if p.is_file()) \
        if (run_dir / "scenarios").exists() else 0
    ledger.append("# scenarios/: %d generated scenario files, not hashed individually" % n_scen)
    for f in sorted(p for p in launcher_dir.iterdir() if p.is_file()):
        ledger.append("%s  %d  launcher  %s" % (sha(f), f.stat().st_size, f))
    for extra in (Path(str(launcher_dir).replace("__launcher", "__launcher.stdout.txt")),
                  Path(str(launcher_dir).replace("__launcher", "__launcher.stderr.txt"))):
        if extra.exists():
            ledger.append("%s  %d  launcher  %s" % (sha(extra), extra.stat().st_size, extra))
    for name in ("run_config.json", "eval_records.jsonl", "train_records.jsonl",
                 "train_credit_diagnostics.jsonl", "episode_outcomes.jsonl"):
        f = original / name
        ledger.append("%s  %d  original  %s" % (sha(f), f.stat().st_size, f))
    for it in (24, 49, 74, 99):
        f = original / "checkpoints" / ("ckpt_iter%04d.pt" % it)
        ledger.append("%s  %d  original  %s" % (sha(f), f.stat().st_size, f))
    (pkg / "artifact_sha256.txt").write_text("\n".join(ledger) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--launcher", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    run, launcher, original, pkg = map(Path, (args.run, args.launcher, args.original,
                                              args.package))
    if args.check:
        tmp = Path(tempfile.mkdtemp(prefix="extract_check_"))
        try:
            extract(run, launcher, original, tmp, copy=False)
            bad = [p.name for p in sorted((tmp / "extracted").iterdir())
                   if p.read_bytes() != (pkg / "extracted" / p.name).read_bytes()]
            print("CHECK", "PASS" if not bad else "FAIL %s" % bad)
            return 0 if not bad else 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    extract(run, launcher, original, pkg, copy=True)
    print("extracted to", pkg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
