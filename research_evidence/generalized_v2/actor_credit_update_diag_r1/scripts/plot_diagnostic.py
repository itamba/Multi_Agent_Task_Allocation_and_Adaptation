"""Readable figure of the primary diagnostic (reads extracted/per_update_diagnostic.jsonl only).

Top: per update, the full-update actual delta C_B and the summed first-order predictions
dot(h, delta_theta) of its four actual Adam steps. Middle: epoch-0 raw descent pressures
(FD, non-FD, entropy, total). Bottom: the batch contrast C_B before each update. Undefined
updates (a severity absent) are marked on the x axis. A symlog y scale keeps near-zero values
visible as near-zero. Run in a process without torch.

Usage: python plot_diagnostic.py <package dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    pkg = Path(sys.argv[1])
    rows = [json.loads(x) for x in (pkg / "extracted" / "per_update_diagnostic.jsonl")
            .read_text(encoding="utf-8").splitlines() if x]
    d = [r for r in rows if r["contrast_defined"]]
    u = [r["iteration"] for r in rows if not r["contrast_defined"]]
    it = [r["iteration"] for r in d]
    fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    ax[0].plot(it, [r["full_update_delta"] for r in d], "o-", ms=3, label="actual full-update ΔC_B")
    ax[0].plot(it, [r["sum_first_order_predictions"] for r in d], "x", ms=4,
               label="Σ epochs dot(h, Δθ) (first-order, actual Adam steps)")
    ax[0].set_ylabel("ΔC_B")
    ax[0].set_yscale("symlog", linthresh=1e-6)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].legend(fontsize=8)
    for key, lab in (("pressure_fd", "FD"), ("pressure_non_fd", "non-FD"),
                     ("pressure_entropy", "entropy"), ("pressure_total", "total")):
        ax[1].plot(it, [r["epochs"][0][key] for r in d], ".-", ms=3, lw=0.8, label=lab)
    ax[1].set_yscale("symlog", linthresh=1e-6)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("epoch-0 pressure −h·g")
    ax[1].legend(fontsize=8, ncol=4)
    ax[2].plot(it, [r["contrast_before_update"] for r in d], "o-", ms=3)
    ax[2].set_ylabel("C_B before update")
    ax[2].set_xlabel("iteration (update index); ticks below = undefined (a severity absent)")
    for a in ax:
        for w in (24.5, 49.5, 74.5):
            a.axvline(w, color="grey", lw=0.5, ls=":")
    ax[2].plot(u, [ax[2].get_ylim()[0]] * len(u), "|", color="red", ms=12)
    fig.suptitle("Actor-only credit-to-update diagnostic R1 — every update (development only)")
    fig.tight_layout()
    fig.savefig(pkg / "extracted" / "primary_diagnostic.png", dpi=110)
    print("wrote", pkg / "extracted" / "primary_diagnostic.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
