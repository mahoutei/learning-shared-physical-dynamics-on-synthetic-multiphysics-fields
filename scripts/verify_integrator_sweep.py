import csv
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

root = Path(sys.argv[1])
variants = ("residual", "whole-window", "last-frame")
reference = None
recipe = None
scores = {}
times = {}
for seed in range(3):
    for variant in variants:
        path = root / f"{variant}-seed{seed}"
        m = json.loads((path / "metrics.json").read_text())
        config = m["config"]
        model = config["model"].copy()
        assert model.pop("prediction_mode") == ("residual" if variant == "residual" else "derivative")
        assert model.pop("last_frame_only") == (variant == "last-frame")
        signature = (config["data"], config["training"], config["eval"], model, m["provenance"])
        if recipe is None:
            recipe = signature
        assert signature == recipe
        assert m["seed"] == seed and m["steps"] == 2000
        assert m["device"] == "cuda" and "T4" in m["gpu"]
        times[seed, variant] = m["train_time_s"]
        with (path / "aligned_trajectory_nmse.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        targets = {}
        groups = {}
        for r in rows:
            h = int(r["horizon"])
            value = float(r["nmse"])
            key = r["split"], r["system"], r["trajectory"], h
            assert key not in targets and math.isfinite(value)
            targets[key] = r["target_sha256"]
            groups.setdefault((r["split"], r["system"], h), []).append(value)
            groups.setdefault((r["split"], "overall", h), []).append(value)
        assert len(targets) == 576 and len(set(targets.values())) == 24
        if reference is None:
            reference = targets
        assert reference == targets
        with (path / "aligned_median_nmse.csv").open(newline="") as f:
            reported = list(csv.DictReader(f))
        assert len(reported) == 48
        for row in reported:
            key = row["split"], row["system"], int(row["horizon"])
            assert len(groups[key]) == int(row["n_samples"])
            assert math.isclose(float(row["nmse_median"]), median(groups[key]), rel_tol=1e-12)
        scores[seed, variant] = {key: median(values) for key, values in groups.items()}
with (root / "paired_nmse.csv").open(newline="") as f:
    paired = list(csv.DictReader(f))
assert len(paired) == 216
for row in paired:
    seed, variant = int(row["seed"]), row["variant"]
    key = row["split"], row["system"], int(row["horizon"])
    assert math.isclose(float(row["delta_vs_residual"]), scores[seed, variant][key] - scores[seed, "residual"][key], abs_tol=1e-12)

print("Verified nine matched configurations, 5,184 finite scores with identical targets, and 216 paired rows.")
for split in ("train", "heldout"):
    for h in (1, 8, 24):
        print(split, f"h{h}", [[round(scores[s, v][split, "overall", h], 6)
                                  for v in variants] for s in range(3)])
for split, system, h in scores[0, "residual"]:
    if h == 24 and system != "overall":
        print(system, [[round(scores[s, v][split, system, h] - scores[s, "residual"][split, system, h], 6)
                        for s in range(3)] for v in variants[1:]])
for v in variants:
    print(v, "average train minutes", round(mean(times[s, v] for s in range(3)) / 60, 2))
