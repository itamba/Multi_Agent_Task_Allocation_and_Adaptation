"""Render extracted/trajectory_comparison.json as a readable small-multiples PNG.

Reads ONLY the extractor's output (no run artifact, no checkpoint). Three panels, each with
its own single y-axis: the ten-cell macro endpoint, P(ABORT) by severity, and directional
switches. Color = run (validated categorical slots 1 and 2), line style = severity.

    python plot_trajectory.py --package <package dir>
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NEW, COMP = "#2a78d6", "#eb6834"      # categorical slots 1 (blue) and 2 (orange)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    args = ap.parse_args()
    pkg = Path(args.package)
    rows = json.loads((pkg / "extracted" / "trajectory_comparison.json").read_text())["rows"]
    x = [r["updates_completed"] for r in rows]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    series = [("mission-slack run (3bc9441)", NEW, "new"),
              ("semantic actor-only R1 (d4e9f37)", COMP, "comparator")]

    ax = axes[0]
    for label, col, key in series:
        ax.plot(x, [r[key + "_macro"] for r in rows], color=col, lw=2, marker="o", ms=4,
                label=label)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("macro P(A|SEVERE) - P(A|MILD)")
    ax.set_title("Primary endpoint trajectory (ten base cells; final round = 375 updates)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    for label, col, key in series:
        ax.plot(x, [r[key + "_p_abort_severe"] for r in rows], color=col, lw=2,
                label=label + " — SEVERE")
        ax.plot(x, [r[key + "_p_abort_mild"] for r in rows], color=col, lw=2, ls="--",
                label=label + " — MILD")
    ax.set_ylabel("mean P(ABORT) at immediate-FD wake")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    ax = axes[2]
    for label, col, key in series:
        ax.plot(x, [r[key + "_directional"]["count"] for r in rows], color=col, lw=2,
                marker="o", ms=4, label=label)
    ax.set_ylabel("directional switches (of 20 groups)")
    ax.set_ylim(-0.5, 20.5)
    ax.set_xlabel("PPO updates completed")
    ax.legend(frameon=False, loc="upper right")

    for a in axes:
        a.grid(True, color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.text(0.01, 0.005, "Cross-version, single-training-seed development comparison; every "
             "round re-measures the same 20 frozen worlds.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = pkg / "extracted" / "trajectory_comparison.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
