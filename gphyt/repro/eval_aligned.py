"""Evaluate saved reproduction checkpoints against identical forecast targets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gphyt.repro.run import build_model, resolve_device, rollout, system_dataset

HISTORY = 8
HORIZON = 24
REPORT_HORIZONS = (1, 2, 4, 8, 16, 24)


def load_samples(root, data):
    samples = []
    for split, key in (("train", "datasets"), ("heldout", "heldout_datasets")):
        for system in data[key]:
            dataset = system_dataset(
                root, system, "test", HISTORY, HORIZON,
                full_trajectory=True, max_rollout_steps=HORIZON,
                include_fields=data.get("include_fields"),
            )
            if dataset is None or len(dataset) != 4:
                raise ValueError(f"Expected four test trajectories for {system}")
            for index in range(len(dataset)):
                x, y = dataset[index]
                if len(x) != HISTORY or len(y) != HORIZON:
                    raise ValueError(f"Unexpected history or target length for {system}/{index}")
                digest = hashlib.sha256(y.contiguous().numpy().tobytes()).hexdigest()
                samples.append((split, system, index, x, y, digest))
    if len(samples) != 24:
        raise ValueError(f"Expected 24 test trajectories, got {len(samples)}")
    return samples


@torch.inference_mode()
def evaluate_checkpoint(path, saved, samples, device):
    config = saved["config"]
    data = config["data"]
    context = data["n_steps_input"]
    if context not in (1, 2, 4, 8):
        raise ValueError(f"Unsupported context length {context} in {path}")
    model, count = build_model(config["model"], context, data["height"], data["width"])
    if count != saved["param_count"]:
        raise ValueError(f"Parameter count mismatch for {path}")
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    groups = {}
    errors = []
    for split, system, index, x, y, digest in samples:
        pred = rollout(model, x[-context:].unsqueeze(0).to(device), HORIZON, context)[0].cpu()
        mse = (pred - y).square().mean(dim=(1, 2, 3))
        nmse = mse / (y.var(dim=(1, 2, 3), unbiased=False) + 1e-8)
        if not torch.isfinite(nmse).all():
            raise ValueError(f"Nonfinite NMSE for {path.name} on {system}/{index}")
        values = nmse.tolist()
        groups.setdefault((split, system), []).append(values)
        groups.setdefault((split, "overall"), []).append(values)
        errors.extend((path.stem, context, split, system, index, digest, h, value)
                      for h, value in enumerate(values, 1))
    medians = []
    for (split, system), values in groups.items():
        median = np.median(values, axis=0)
        medians.extend((path.stem, context, split, system, h, len(values), median[h - 1])
                       for h in REPORT_HORIZONS)
    return errors, medians


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.set_num_threads(4)
    device = resolve_device(args.device)
    checkpoints = sorted(args.checkpoint_dir.glob("*-t4.pt"))
    if len(checkpoints) != 7:
        raise ValueError(f"Expected seven T4 checkpoints, got {len(checkpoints)}")
    samples = None
    data_signature = None
    all_errors, all_medians = [], []
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        data = saved["config"]["data"]
        signature = {k: data.get(k) for k in
                     ("datasets", "heldout_datasets", "include_fields", "height", "width")}
        if samples is None:
            samples = load_samples(args.data_dir, data)
            data_signature = signature
        if signature != data_signature:
            raise ValueError(f"Corpus configuration differs for {path}")
        errors, medians = evaluate_checkpoint(path, saved, samples, device)
        all_errors.extend(errors)
        all_medians.extend(medians)
        print(f"{path.name}: scored {len(samples)} trajectories", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "aligned_trajectory_nmse.csv",
              ("checkpoint", "context", "split", "system", "trajectory", "target_sha256",
               "horizon", "nmse"), all_errors)
    write_csv(args.output_dir / "aligned_median_nmse.csv",
              ("checkpoint", "context", "split", "system", "horizon", "n_samples",
               "nmse_median"), all_medians)
    metadata = {"device": str(device), "torch": str(torch.__version__),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "history": HISTORY, "horizon": HORIZON}
    (args.output_dir / "aligned_runtime.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
