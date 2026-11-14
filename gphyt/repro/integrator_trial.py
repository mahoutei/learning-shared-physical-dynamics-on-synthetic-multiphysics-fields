"""Run and validate one matched integrator job on a Colab T4."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from gphyt.repro.eval_aligned import evaluate_checkpoint, load_samples, write_csv

VARIANTS = ("residual", "whole-window", "last-frame")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("seed", type=int, choices=range(3))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(device)
    if "T4" not in gpu:
        raise RuntimeError(f"Expected a T4, found {gpu}")
    tag = f"{args.variant}-seed{args.seed}" + ("-smoke" if args.smoke else "")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path("gphyt/repro/config.yaml").read_text())
    config["run_name"] = tag
    config["seed"] = args.seed
    config["model"]["prediction_mode"] = "residual" if args.variant == "residual" else "derivative"
    config["model"]["last_frame_only"] = args.variant == "last-frame"
    config["eval"]["single_step"] = False
    config_path = output / f"{tag}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    subprocess.run([
        sys.executable, "-u", "-m", "gphyt.repro.run", "--config", str(config_path),
        "--device", "cuda", "--data-dir", str(args.data_dir.resolve()),
        "--output-dir", str(output), "--tag", tag,
        *(["--smoke"] if args.smoke else []),
    ], check=True)
    run = output / tag
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    if (metrics["steps"] != (5 if args.smoke else 2000)
            or metrics["seed"] != args.seed or metrics["device"] != "cuda"
            or metrics["config"] != checkpoint["config"]
            or checkpoint["config"]["model"] != config["model"]
            or checkpoint["config"]["data"]["n_steps_input"] != 4
            or not math.isfinite(metrics["train_loss"])):
        raise ValueError(f"Unexpected or invalid run metadata for {tag}")
    torch.set_num_threads(4)
    samples = load_samples(args.data_dir, checkpoint["config"]["data"])
    errors, medians = evaluate_checkpoint(Path(tag), checkpoint, samples, device)
    write_csv(run / "aligned_trajectory_nmse.csv",
              ("checkpoint", "context", "split", "system", "trajectory", "target_sha256",
               "horizon", "nmse"), errors)
    write_csv(run / "aligned_median_nmse.csv",
              ("checkpoint", "context", "split", "system", "horizon", "n_samples",
               "nmse_median"), medians)
    metrics.update(gpu=gpu, torch=str(torch.__version__), variant=args.variant,
                   calls_per_forecast=1 if args.variant == "residual" else 2,
                   provenance=json.loads(Path("provenance.json").read_text()))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(f"Validated {tag}", flush=True)


if __name__ == "__main__":
    main()
