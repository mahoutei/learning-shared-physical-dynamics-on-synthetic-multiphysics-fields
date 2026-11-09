"""Median rollout NMSE for the reduced-scale GPhyT-S ablations.

Regenerate with:
    uv run --no-project --with matplotlib --with numpy python docs/figures/results.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from orx_figstyle import BASELINE, PALETTE, WIDE, figure_grid, panel_labels, save, use_style

DATA = Path(__file__).resolve().parent.parent / "data" / "rollout_nmse.csv"
OUT = Path(__file__).resolve().parent / "rollout_nmse"

STYLES = {
    "GPhyT-S baseline": dict(color=BASELINE, lw=1.8, zorder=4),
    "No derivatives": dict(color=PALETTE["blue"], lw=1.2, zorder=3),
    "UNet-S": dict(color=PALETTE["orange"], lw=1.2, zorder=2),
    "Context 1": dict(color=PALETTE["green"], lw=1.2, zorder=3),
}


def load():
    series = defaultdict(dict)
    with DATA.open(encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            variant = row["variant"]
            if variant not in STYLES:
                continue
            split = row["split"]
            series[(variant, split)][int(row["horizon"])] = float(row["nmse_median"])
    return series


def main():
    use_style()
    series = load()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.38, sharey=True)

    for ax, split in zip(axes, ("train", "heldout")):
        for variant, style in STYLES.items():
            points = series[(variant, split)]
            horizons = sorted(points)
            values = [points[h] for h in horizons]
            ax.plot(horizons, values, marker="o", ms=2.0, label=variant, **style)
        ax.set_xlabel("Autoregressive rollout horizon (steps)")
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="major", color="0.92", lw=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Median NMSE (normalized, log scale)")
    panel_labels(axes, ["(a) Train systems", "(b) Held-out systems"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    save(fig, str(OUT), formats=("svg", "pdf"))


if __name__ == "__main__":
    main()
