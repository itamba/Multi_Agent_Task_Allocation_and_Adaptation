"""Plot the REWARD-01 arm comparison from the committed extracted JSON (needs matplotlib).

Reads only ``extracted/behaviour_by_round.json`` and ``extracted/outcome_summary_by_round.json``
and writes ``extracted/trajectory_comparison.png``. Six small multiples, each with ONE y-axis:
P(ABORT) by severity, the macro gap, switch counts, mean scored utility, airframes lost, and
the mean algebraic rescoring under each common coefficient. Arm A (c = 2.25) is series slot 1
(blue), arm B (c = 4.5) slot 2 (orange); identity is also carried by the legend and line style.

Usage: python plot_comparison.py <package dir>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"A": "#2a78d6", "B": "#eb6834"}
LABELS = {"A": "arm A  c = 2.25", "B": "arm B  c = 4.5"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.set_xlabel("updates completed", fontsize=8, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def main():
    pkg = Path(sys.argv[1])
    ex = pkg / "extracted"
    beh = json.loads((ex / "behaviour_by_round.json").read_text(encoding="utf-8"))
    outs = json.loads((ex / "outcome_summary_by_round.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True)
    kw = dict(linewidth=2, marker="o", markersize=4)
    for arm in ("A", "B"):
        c, x = COLORS[arm], [r["updates_completed"] for r in beh[arm]]
        axes[0, 0].plot(x, [r["p_abort_severe_mean"] for r in beh[arm]], color=c,
                        label=LABELS[arm] + "  SEVERE", **kw)
        axes[0, 0].plot(x, [r["p_abort_mild_mean"] for r in beh[arm]], color=c, linestyle="--",
                        label=LABELS[arm] + "  MILD", **kw)
        axes[0, 1].plot(x, [r["severe_minus_mild_macro_over_base_cells"] for r in beh[arm]],
                        color=c, label=LABELS[arm], **kw)
        axes[1, 0].plot(x, [r["directional_switches"]["count"] for r in beh[arm]], color=c,
                        label=LABELS[arm] + "  directional", **kw)
        axes[1, 0].plot(x, [r["both_abort"]["count"] for r in beh[arm]], color=c,
                        linestyle=":", label=LABELS[arm] + "  both ABORT", **kw)
        xo = [r["updates_completed"] for r in outs[arm]]
        axes[1, 1].plot(xo, [r["all_members"]["u_achieved_mean"] for r in outs[arm]], color=c,
                        label=LABELS[arm], **kw)
        axes[2, 0].plot(xo, [r["all_members"]["deaths_sum"] for r in outs[arm]], color=c,
                        label=LABELS[arm] + "  all members", **kw)
        axes[2, 0].plot(xo, [r["by_member_cell"]["severe"]["deaths_sum"] for r in outs[arm]],
                        color=c, linestyle="--", label=LABELS[arm] + "  SEVERE members", **kw)
        axes[2, 1].plot(xo, [r["all_members"]["R_c2p25_mean"] for r in outs[arm]], color=c,
                        label=LABELS[arm] + "  rescored at c = 2.25", **kw)
        axes[2, 1].plot(xo, [r["all_members"]["R_c4p5_mean"] for r in outs[arm]], color=c,
                        linestyle="--", label=LABELS[arm] + "  rescored at c = 4.5", **kw)
    style(axes[0, 0], "P(ABORT) at the immediate-FD wake (mean over 20 worlds)", "probability")
    style(axes[0, 1], "Primary quantity: macro P(ABORT|SEVERE) − P(ABORT|MILD)", "difference")
    axes[0, 1].axhline(0, color=MUTED, linewidth=0.8)
    style(axes[1, 0], "Selected-action groups (of 20 matched worlds)", "groups")
    style(axes[1, 1], "Mean scored utility U_prefix + U_post (60 members)", "utility")
    style(axes[2, 0], "Airframes lost per round", "aircraft")
    style(axes[2, 1], "Mean reward rescored q − c·p (same saved trajectories)", "reward")
    for ax in axes.flat:
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("REWARD-01 development comparison R1 — one training seed, 20 frozen "
                 "development worlds re-measured each round (repeated measures)",
                 fontsize=10, color=INK)
    fig.savefig(ex / "trajectory_comparison.png", dpi=110, facecolor="white")
    print("wrote", ex / "trajectory_comparison.png")


if __name__ == "__main__":
    main()
