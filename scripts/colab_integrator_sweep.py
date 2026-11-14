"""Drive the matched T4 sweep and download each completed run via Colab CLI."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
import zipfile

VARIANTS = ("residual", "whole-window", "last-frame")
ARTIFACTS = ("metrics.json", "model.pt", "aligned_trajectory_nmse.csv", "aligned_median_nmse.csv")
WORK = "/content/integrator-comparison"


def colab(cli, *args):
    result = subprocess.run([cli, *args], capture_output=True, text=True)
    diagnostic = re.sub(r"(colab-runtime-proxy-token=)[^&\s]+", r"\1[redacted]",
                        result.stdout + result.stderr)
    print(diagnostic.strip(), flush=True)
    result.check_returncode()


def validate_run(directory, variant, seed, smoke, provenance):
    metrics = json.loads((directory / "metrics.json").read_text())
    expected_calls = 1 if variant == "residual" else 2
    if (metrics["variant"] != variant or metrics["seed"] != seed
            or metrics["steps"] != (5 if smoke else 2000)
            or metrics["calls_per_forecast"] != expected_calls
            or "T4" not in metrics["gpu"] or metrics["device"] != "cuda"
            or metrics["provenance"] != provenance
            or not math.isfinite(metrics["train_time_s"])):
        raise ValueError(f"Run metadata mismatch: {directory}")
    with zipfile.ZipFile(directory / "model.pt") as checkpoint:
        if checkpoint.testzip() is not None:
            raise ValueError(f"Checkpoint archive failed its CRC check: {directory}")
    with (directory / "aligned_trajectory_nmse.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = {(r["split"], r["system"], int(r["trajectory"]), int(r["horizon"])):
               r["target_sha256"] for r in rows}
    if (len(rows) != 576 or len(targets) != 576
            or len(set(targets.values())) != 24
            or not all(math.isfinite(float(r["nmse"])) for r in rows)):
        raise ValueError(f"Invalid trajectory scores: {directory}")
    return metrics, targets


def write_report(output):
    runs = {}
    for seed in range(3):
        for variant in VARIANTS:
            directory = output / f"{variant}-seed{seed}"
            if not directory.is_dir():
                continue
            metrics = json.loads((directory / "metrics.json").read_text())
            with (directory / "aligned_median_nmse.csv").open(newline="") as handle:
                values = {(r["split"], r["system"], r["horizon"]): float(r["nmse_median"])
                          for r in csv.DictReader(handle) if r["horizon"] in ("1", "8", "24")}
            if len(values) != 24:
                raise ValueError(f"Incomplete aligned medians: {directory}")
            runs[seed, variant] = metrics, values
    with (output / "paired_nmse.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "variant", "split", "system", "horizon", "median_nmse",
                         "residual_median_nmse", "delta_vs_residual", "train_time_s",
                         "network_calls_per_forecast"))
        for (seed, variant), (metrics, values) in runs.items():
            baseline = runs[seed, "residual"][1]
            for (split, system, horizon), value in values.items():
                control = baseline[split, system, horizon]
                writer.writerow((seed, variant, split, system, horizon, value, control,
                                 value - control, metrics["train_time_s"],
                                 metrics["calls_per_forecast"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colab", default="colab")
    parser.add_argument("--session", default="gphyt-integrators")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    provenance = {name: entry["sha256"] for name, entry in manifest.items()}
    jobs = [(0, v, True) for v in VARIANTS]
    jobs += [(s, v, False) for s in range(3) for v in VARIANTS]
    reference = None
    for seed, variant, smoke in jobs:
        tag = f"{variant}-seed{seed}" + ("-smoke" if smoke else "")
        destination = args.output / tag
        with TemporaryDirectory(dir=args.output) as scratch:
            if not destination.exists():
                print(f"Starting {tag}", flush=True)
                stage = Path(scratch) / tag
                stage.mkdir()
                payload = Path(scratch) / "remote_job.py"
                command = [f"{WORK}/.venv/bin/python", "-u", "-m", "gphyt.repro.integrator_trial",
                           variant, str(seed), "--data-dir", f"{WORK}/corpus",
                           "--output-dir", f"{WORK}/results"]
                if smoke:
                    command.append("--smoke")
                remote_dir = f"{WORK}/results/{tag}"
                payload.write_text(
                    "import json, subprocess\nfrom pathlib import Path\n"
                    f"run = Path({remote_dir!r})\n"
                    f"complete = all((run / name).is_file() for name in {ARTIFACTS!r})\n"
                    "if complete:\n"
                    "    complete = 'provenance' in json.loads((run / 'metrics.json').read_text())\n"
                    "if not complete:\n"
                    f"    subprocess.run({command!r}, cwd={WORK!r}, check=True)\n"
                )
                colab(args.colab, "exec", "-s", args.session, "--timeout", "3600", "-f", str(payload))
                for filename in ARTIFACTS:
                    colab(args.colab, "download", "-s", args.session,
                          f"{WORK}/results/{tag}/{filename}", str(stage / filename))
                _, targets = validate_run(stage, variant, seed, smoke, provenance)
                if reference is not None and reference != targets:
                    raise ValueError(f"Forecast targets differ for {tag}")
                stage.rename(destination)
            metrics, targets = validate_run(destination, variant, seed, smoke, provenance)
            if reference is not None and reference != targets:
                raise ValueError(f"Forecast targets differ for {tag}")
            reference = targets
            print(f"Verified {tag}, {metrics['train_time_s']:.1f}s training", flush=True)
        if not smoke:
            write_report(args.output)


if __name__ == "__main__":
    main()
