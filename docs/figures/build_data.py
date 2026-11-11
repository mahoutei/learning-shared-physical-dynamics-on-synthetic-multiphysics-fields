"""Build the per-system T4 held-out metric table from result JSONs.

Usage:
    uv run --no-project python docs/figures/build_data.py \
        --metrics-dir <artifacts>/colab-runs/data \
        --out docs/data/rollout_nmse_t4_per_system.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VARIANTS = [
    ("gphyt_s_baseline_colab", "GPhyT-S baseline"),
    ("gphyt_s_context_1", "Context 1"),
    ("gphyt_s_context_2", "Context 2"),
    ("gphyt_s_context_8", "Context 8"),
    ("gphyt_s_no_derivatives", "No derivatives"),
    ("unet_s_baseline", "UNet-S"),
    ("gphyt_s_direct_prediction", "Direct prediction"),
]
VARIANT_ORDER = {name: index for index, (_, name) in enumerate(VARIANTS)}
VARIANT_NAMES = {run_name: name for run_name, name in VARIANTS}

SYSTEMS = [
    ("advection", "train"),
    ("burgers", "train"),
    ("incompressible_ns", "train"),
    ("sine_gordon", "train"),
    ("wave", "heldout"),
    ("thermal_convection", "heldout"),
]
SYSTEM_ORDER = {name: index for index, (name, _) in enumerate(SYSTEMS)}
SYSTEM_SPLIT = dict(SYSTEMS)

FIELDS = [
    "variant",
    "run_name",
    "system",
    "split",
    "horizon",
    "nmse_median",
    "nmse_mean",
    "n_samples",
]


def load_rows(metrics_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in sorted(metrics_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_name = payload.get("run_name")
        if run_name not in VARIANT_NAMES:
            raise SystemExit(f"{path.name}: unexpected run_name {run_name!r}")
        if run_name in seen:
            raise SystemExit(f"duplicate run_name {run_name!r}")
        seen.add(run_name)

        groups = {
            "train": payload["eval"]["rollout"]["per_system"],
            "heldout": payload["eval"]["heldout_rollout"]["per_system"],
        }
        for system, split in SYSTEMS:
            data = groups[split][system]
            horizons = data["horizons"]
            medians = data["nmse_median"]
            means = data["nmse_mean"]
            if not (len(horizons) == len(medians) == len(means)):
                raise SystemExit(f"{path.name}: mismatched arrays for {system}")
            for horizon, median, mean in zip(horizons, medians, means):
                rows.append(
                    {
                        "variant": VARIANT_NAMES[run_name],
                        "run_name": run_name,
                        "system": system,
                        "split": SYSTEM_SPLIT[system],
                        "horizon": int(horizon),
                        "nmse_median": median,
                        "nmse_mean": mean,
                        "n_samples": int(data["n_samples"]),
                    }
                )

    missing = set(VARIANT_NAMES) - seen
    if missing:
        raise SystemExit(f"missing result JSONs for: {sorted(missing)}")
    rows.sort(
        key=lambda row: (
            VARIANT_ORDER[str(row["variant"])],
            SYSTEM_ORDER[str(row["system"])],
            int(row["horizon"]),
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/data/rollout_nmse_t4_per_system.csv"),
    )
    args = parser.parse_args()

    rows = load_rows(args.metrics_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
