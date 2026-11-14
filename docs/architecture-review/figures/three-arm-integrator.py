"""Three-arm architecture test at 64x32. Regenerate: python figures/three-arm-integrator.py"""

import csv
from pathlib import Path

import numpy as np
from orx_figstyle import (
    BASELINE,
    MUTED,
    PALETTE,
    WIDE,
    figure_grid,
    panel_labels,
    save,
    use_style,
)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "three_arm_cpu.csv"
SYSTEMS = [
    "advection",
    "burgers",
    "incompressible_ns",
    "sine_gordon",
    "wave",
    "thermal_convection",
]
LABELS = ["Advection", "Burgers", "Incomp. NS", "Sine-Gordon", "Wave*", "Thermal*"]


def load():
    values = {}
    with DATA.open() as handle:
        for row in csv.DictReader(handle):
            if int(row["horizon"]) != 24:
                continue
            values[(row["arm"], row["system"])] = float(row["nmse_median"])
    return values


def main():
    use_style()
    values = load()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.36, sharey=True)
    ax_a, ax_b = axes

    y = np.arange(len(SYSTEMS))[::-1]
    for yi, system in zip(y, SYSTEMS):
        left = values[("residual", system)]
        right = values[("derivative_heun", system)]
        ax_a.plot([left, right], [yi, yi], color=MUTED, linewidth=1.2, zorder=1)
        ax_a.plot(left, yi, "o", color=BASELINE, markeredgecolor="black",
                  markeredgewidth=0.3, zorder=3, label="Residual map" if yi == y[0] else None)
        ax_a.plot(right, yi, "s", color=PALETTE["blue"], markeredgecolor="black",
                  markeredgewidth=0.3, zorder=3, label="Derivative + Heun" if yi == y[0] else None)
        direct = values[("direct_state", system)]
        ax_b.plot(direct, yi, "D", color=PALETTE["orange"], markeredgecolor="black",
                  markeredgewidth=0.3, zorder=3, label="Direct state" if yi == y[0] else None)

    ax_a.set_xscale("log")
    ax_a.set_xlim(0.02, 3.0)
    ax_a.set_xticks([0.03, 0.1, 0.3, 1, 3])
    ax_a.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax_a.axvline(1.0, color=BASELINE, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax_a.set_xlabel("h24 NMSE (log scale)")
    ax_a.set_yticks(y, LABELS)
    ax_a.set_ylabel("System")
    ax_a.tick_params(axis="y", length=0)
    ax_a.grid(axis="x")
    ax_a.grid(axis="y", visible=False)
    ax_a.legend(loc="upper left", fontsize=6.5, frameon=True, framealpha=1.0,
                edgecolor="black", borderpad=0.4)

    ax_b.set_xscale("log")
    ax_b.set_xlim(1e3, 3e5)
    ax_b.set_xticks([1e3, 1e4, 1e5])
    ax_b.set_xticklabels(["$10^3$", "$10^4$", "$10^5$"])
    ax_b.set_xlabel("Direct-state h24 NMSE (log scale)")
    ax_b.tick_params(axis="y", length=0)
    ax_b.grid(axis="x")
    ax_b.grid(axis="y", visible=False)

    for ax in (ax_a, ax_b):
        ax.spines["left"].set_visible(False)

    panel_labels([ax_a, ax_b])
    save(fig, str(HERE / "three-arm-integrator"), formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    main()
