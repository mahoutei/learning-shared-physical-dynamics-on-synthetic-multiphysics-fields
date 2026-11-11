"""Per-system held-out rollout NMSE for the reduced-scale GPhyT-S ablations.

Build the CSV first with build_data.py, then regenerate the figure with:
    uv run --no-project --with matplotlib --with numpy python docs/figures/heldout_systems.py
"""

import csv
from collections import defaultdict
from pathlib import Path

from orx_figstyle import WIDE, figure_grid, panel_labels, save, use_style
from results import STYLES

DATA = Path(__file__).resolve().parent.parent / "data" / "rollout_nmse_t4_per_system.csv"
OUT = Path(__file__).resolve().parent / "heldout_systems"

SYSTEMS = [("wave", "(a) Wave"), ("thermal_convection", "(b) Thermal convection")]
LEGEND_ORDER = [
    "GPhyT-S baseline",
    "Context 1",
    "Context 2",
    "Context 8",
    "No derivatives",
    "UNet-S",
]


def load():
    series = defaultdict(lambda: defaultdict(dict))
    with DATA.open(encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "heldout":
                continue
            series[row["system"]][row["variant"]][int(row["horizon"])] = float(
                row["nmse_median"]
            )
    return series


def main():
    use_style()
    series = load()
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.38, sharey=True)

    for ax, (system, _label) in zip(axes, SYSTEMS):
        for variant in LEGEND_ORDER:
            points = series[system][variant]
            horizons = sorted(points)
            values = [points[h] for h in horizons]
            ax.plot(
                horizons,
                values,
                marker="o",
                ms=2.0,
                label=variant,
                **STYLES[variant],
            )
        ax.set_xlabel("Autoregressive rollout horizon (steps)")
        ax.set_xlim(1, 24)
        ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 5.0)
        ax.grid(True, axis="y", which="major", color="0.92", lw=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Median NMSE (normalized, log scale)")
    panel_labels(axes, [label for _system, label in SYSTEMS])
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        [by_label[name] for name in LEGEND_ORDER],
        LEGEND_ORDER,
        loc="outside lower center",
        ncol=6,
        frameon=False,
    )
    save(fig, str(OUT), formats=("svg", "pdf", "png"))


if __name__ == "__main__":
    main()
