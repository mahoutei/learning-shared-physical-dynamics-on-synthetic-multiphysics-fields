"""Scale helps on average but not on every physical system.

Data: 256x128 zero-shot medians from the four released GPhyT checkpoints.
Inputs are four frames; errors are medians over 24 test trajectories overall
and 4 per system. No confidence intervals are available.
"""

import csv
from pathlib import Path

from matplotlib.ticker import NullFormatter

from orx_figstyle import PALETTE, TEXT, figure_grid, panel_labels, save, use_style

DATA = Path(__file__).resolve().parent.parent / "data" / "released_scaling_256.csv"
OUT = Path(__file__).resolve().parent / "released-scaling-heterogeneity"

MODELS = ["GPhyT-S", "GPhyT-M", "GPhyT-L", "GPhyT-XL"]
COLORS = {
    "GPhyT-S": PALETTE["blue"],
    "GPhyT-M": PALETTE["orange"],
    "GPhyT-L": PALETTE["green"],
    "GPhyT-XL": PALETTE["red"],
}
MARKERS = {"GPhyT-S": "o", "GPhyT-M": "s", "GPhyT-L": "^", "GPhyT-XL": "D"}
SYSTEMS = [
    ("advection", "Advection"),
    ("burgers", "Burgers"),
    ("incompressible_ns", "Incompressible NS"),
    ("sine_gordon", "Sine-Gordon"),
    ("wave", "Wave"),
    ("thermal_convection", "Thermal convection"),
]
HORIZONS = [1, 8, 24]


def load():
    rows = []
    with DATA.open(encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "model": row["model"],
                    "parameters": int(row["parameters"]),
                    "scope": row["scope"],
                    "horizon": int(row["horizon"]),
                    "nmse": float(row["nmse_median"]),
                }
            )
    return rows


def main():
    use_style()
    rows = load()
    overall = {
        (r["model"], r["horizon"]): r["nmse"] for r in rows if r["scope"] == "overall"
    }
    per_system = {
        (r["model"], r["scope"]): r["nmse"]
        for r in rows
        if r["scope"] != "overall" and r["horizon"] == 24
    }
    params = {r["model"]: r["parameters"] / 1e6 for r in rows}

    fig, axes = figure_grid(1, 2, width=TEXT, ratio=0.44)

    ax = axes[0]
    for model in MODELS:
        values = [overall[(model, h)] for h in HORIZONS]
        ax.plot(
            HORIZONS,
            values,
            color=COLORS[model],
            marker=MARKERS[model],
            ms=3.2,
            lw=1.1,
            label=model.replace("GPhyT-", ""),
        )
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 3.0)
    ax.set_xlim(0.7, 26)
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Autoregressive rollout horizon (steps)")
    ax.set_ylabel("Median NMSE (log scale)")
    ax.grid(True, axis="y", which="major", color="0.92", lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    for model in MODELS:
        ratios = [
            per_system[(model, system)] / per_system[("GPhyT-S", system)]
            for system, _label in SYSTEMS
        ]
        ax.scatter(
            ratios,
            range(len(SYSTEMS)),
            color=COLORS[model],
            marker=MARKERS[model],
            s=18,
            label=model.replace("GPhyT-", ""),
            zorder=3,
        )
    ax.axvline(1.0, color="0.45", lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 1.6)
    ax.set_xticks([0.25, 0.5, 1.0])
    ax.set_xticklabels(["0.25", "0.5", "1.0"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(range(len(SYSTEMS)))
    ax.set_yticklabels([label for _system, label in SYSTEMS])
    ax.invert_yaxis()
    ax.set_xlabel("H24 error relative to GPhyT-S (log scale)")
    ax.grid(True, axis="x", which="major", color="0.92", lw=0.5)
    ax.set_axisbelow(True)

    handle, label = axes[0].get_legend_handles_labels()
    fig.legend(
        handle,
        label,
        title="Released checkpoint",
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    panel_labels(axes)
    save(fig, str(OUT), formats=("pdf", "svg", "png"))


if __name__ == "__main__":
    main()
