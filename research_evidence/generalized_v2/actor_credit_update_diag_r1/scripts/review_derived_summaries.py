"""REVIEW-DERIVED arithmetic over existing committed scalar records (review fix F3).

Not a predeclared endpoint. Standard library only; reads ``extracted/per_update_diagnostic.jsonl``
and ``extracted/secondary_behaviour.json`` (both committed) and writes
``extracted/review_derived_summaries.json``. Signs use the extractor's existing numerically-
negligible convention |x| < 1e-6. Quantiles are linear interpolation between order statistics.

Usage: python review_derived_summaries.py <package>          (write)
       python review_derived_summaries.py <package> --check  (rebuild and require byte identity)
"""

import json
import math
import sys
from pathlib import Path

NEG = 1e-6
WINDOWS = [(0, 24), (25, 49), (50, 74), (75, 99), (0, 99)]


def sgn(x):
    return 0 if abs(x) < NEG else (1 if x > 0 else -1)


def q(v, p):
    v = sorted(v)
    pos = (len(v) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def stats(v):
    v = [float(x) for x in v]
    if not v:
        return None
    return {"n": len(v), "mean": math.fsum(v) / len(v), "median": q(v, 0.5),
            "q10": q(v, 0.1), "q90": q(v, 0.9), "max": max(v)}


def build(pkg: Path) -> bytes:
    rows = [json.loads(x) for x in (pkg / "extracted" / "per_update_diagnostic.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    sec = json.loads((pkg / "extracted" / "secondary_behaviour.json").read_text(encoding="utf-8"))
    d = [r for r in rows if r["contrast_defined"]]
    name = {1: "positive", -1: "negative", 0: "negligible"}
    pairs = {}
    for r in d:
        e0 = r["epochs"][0]
        k = "fd_%s__non_fd_%s" % (name[sgn(e0["pressure_fd"])], name[sgn(e0["pressure_non_fd"])])
        pairs[k] = pairs.get(k, 0) + 1
    nonneg = [r["epochs"][0] for r in d if sgn(r["epochs"][0]["pressure_fd"]) != 0
              and sgn(r["epochs"][0]["pressure_non_fd"]) != 0]
    out = {
        "record": "review_derived_summaries", "record_version": 1,
        "status": ("REVIEW-DERIVED arithmetic over committed scalar records (PR #80 review fix "
                   "F3); NOT a predeclared endpoint"),
        "negligible_convention": "|x| < 1e-6",
        "quantiles": "linear interpolation between order statistics",
        "n_defined_updates": len(d), "n_defined_epochs": 4 * len(d),
        "A_epoch0_fd_vs_non_fd_pressure_sign_pairs": dict(sorted(pairs.items())),
        "A_non_negligible_pairs": {
            "n": len(nonneg),
            "opposing_sign": sum(1 for e in nonneg
                                 if sgn(e["pressure_fd"]) != sgn(e["pressure_non_fd"])),
            "same_sign": sum(1 for e in nonneg
                             if sgn(e["pressure_fd"]) == sgn(e["pressure_non_fd"]))},
        "A_non_fd_positive_toward_contrast": sum(
            1 for r in d if sgn(r["epochs"][0]["pressure_non_fd"]) > 0),
        "windows": [],
    }
    for lo, hi in WINDOWS:
        w = [r for r in d if lo <= r["iteration"] <= hi]
        eps = [e for r in w for e in r["epochs"]]
        cos = [e["cosine_delta_vs_h"] for e in eps if e["cosine_delta_vs_h"] is not None]
        out["windows"].append({
            "iterations": [lo, hi], "n_defined_updates": len(w),
            "B_abs_cosine_delta_theta_vs_h": stats([abs(c) for c in cos]),
            "B_signed_cosine_delta_theta_vs_h": stats(cos),
            "C_abs_of_update_summed_linearization_residual": stats(
                [abs(r["sum_linearization_residuals"]) for r in w]),
            "C_update_sum_of_abs_epoch_linearization_residuals": stats(
                [math.fsum(abs(e["linearization_residual"]) for e in r["epochs"]) for r in w]),
            "C_abs_epoch_linearization_residual": stats(
                [abs(e["linearization_residual"]) for e in eps]),
            "C_abs_epoch_actual_delta": stats([abs(e["actual_delta"]) for e in eps]),
            "D_abs_full_update_delta": stats([abs(r["full_update_delta"]) for r in w]),
        })
    out["D_E_fixed_world_evaluation_rounds"] = [{
        "updates_completed": r["new"]["updates_completed"],
        "directional_switches": r["new"]["directional_switches"],
        "reverse_switches": r["new"]["reverse_switches"],
        "n_groups_metric_eligible": r["new"]["n_groups_metric_eligible"],
        "macro_severe_minus_mild": r["new"]["macro_severe_minus_mild"]} for r in sec]
    return (json.dumps(out, indent=1, sort_keys=True) + "\n").encode("utf-8")


def main() -> int:
    pkg = Path(sys.argv[1])
    target = pkg / "extracted" / "review_derived_summaries.json"
    data = build(pkg)
    if "--check" in sys.argv[2:]:
        ok = target.read_bytes() == data
        print("CHECK", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    target.write_bytes(data)
    print("wrote", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
