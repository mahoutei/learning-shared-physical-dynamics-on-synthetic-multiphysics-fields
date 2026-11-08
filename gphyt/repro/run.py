"""Reduced-scale reproduction of "Towards a Physics Foundation Model".

The entry point is deliberately a single fixed command::

    uv run --locked python -m gphyt.repro.run

All experiment variation lives in ``gphyt/repro/config.yaml`` (model size,
ablations, data systems, horizon, seed).  The harness:

1. builds a deterministic multi-physics corpus in The Well HDF5 format,
2. trains GPhyT (or a UNet baseline) with autoregressive multi-step loss,
3. evaluates single-step and autoregressive rollout NMSE on the training
   systems and on held-out physical systems, and
4. writes ``metrics.json`` plus the checkpoint to the run directory.

This is a reduced surrogate, not the paper's 1.8 TB / 4xH100 training run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from gphyt.data.dataset_utils import get_dt_datasets
from gphyt.data.phys_dataset import SuperDataset, get_phys_dataset
from gphyt.models.model_utils import get_model
from gphyt.repro import corpus, systems

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


class DirectPrediction(nn.Module):
    """Ablation: predict the next state directly instead of integrating a derivative."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.differentiate(x)[:, -1, ...].unsqueeze(1)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def prepare_data(config: dict, smoke: bool, data_dir_override: str | None) -> Path:
    data_root = Path(data_dir_override or config["data"]["data_dir"]).expanduser()
    if not data_root.is_absolute():
        data_root = (REPO_ROOT / data_root).resolve()
    if smoke:
        corpus.SPLIT_TRAJECTORIES = {"train": 2, "valid": 1, "test": 1}
    if not corpus.check_corpus(data_root):
        print(f"[data] corpus missing at {data_root}; generating")
        corpus.build_corpus(data_root, overwrite=False)
    return data_root


def build_model(model_config: dict, n_steps_input: int, h: int, w: int) -> tuple[nn.Module, int]:
    cfg = copy.deepcopy(model_config)
    cfg["img_size"] = (n_steps_input, h, w)
    architecture = cfg.get("architecture", "gphyt")
    if architecture == "unet":
        cfg.setdefault("n_time_steps", n_steps_input)
    model = get_model(cfg)
    if architecture == "gphyt" and not cfg.get("residual", True):
        model = DirectPrediction(model)
    param_count = sum(p.numel() for p in model.parameters())
    return model, param_count


def make_train_loader(data_config: dict, seed: int) -> tuple[DataLoader, SuperDataset]:
    datasets = get_dt_datasets(data_config, split="train")
    super_ds = SuperDataset(datasets, max_samples_per_ds=None, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        super_ds,
        batch_size=data_config["batch_size"],
        sampler=RandomSampler(super_ds, generator=generator),
        num_workers=0,
        drop_last=True,
    )
    return loader, super_ds


def infinite(loader: DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def system_dataset(
    data_root: Path,
    system: str,
    split: str,
    n_steps_input: int,
    n_steps_output: int,
    *,
    full_trajectory: bool = False,
    max_rollout_steps: int = 1,
    include_fields: dict | None = None,
):
    path = data_root / system / "data" / split
    if not path.exists():
        return None
    return get_phys_dataset(
        path.resolve(),
        n_steps_input=n_steps_input,
        n_steps_output=n_steps_output,
        dt_stride=1,
        full_trajectory_mode=full_trajectory,
        max_rollout_steps=max_rollout_steps,
        use_normalization=True,
        include_fields=include_fields,
    )


@torch.no_grad()
def evaluate_single_step(
    model: nn.Module,
    data_root: Path,
    system_names: list[str],
    data_config: dict,
    batch_size: int,
    device: torch.device,
) -> dict:
    n_in = data_config["n_steps_input"]
    include_fields = data_config.get("include_fields")
    per_system: dict[str, dict] = {}
    pooled: list[float] = []
    model.eval()
    for system in system_names:
        ds = system_dataset(
            data_root,
            system,
            "test",
            n_in,
            1,
            include_fields=include_fields,
        )
        if ds is None or len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        scores: list[float] = []
        mses: list[float] = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mse = ((pred - y) ** 2).mean(dim=(1, 2, 3, 4))
            var = y.var(dim=(1, 2, 3, 4), unbiased=False)
            scores.extend((mse / (var + 1e-8)).cpu().tolist())
            mses.extend(mse.cpu().tolist())
        per_system[system] = {
            "nmse_median": float(np.median(scores)),
            "nmse_mean": float(np.mean(scores)),
            "mse_median": float(np.median(mses)),
            "n_samples": len(scores),
        }
        pooled.extend(scores)
    return {
        "per_system": per_system,
        "nmse_median_all": float(np.median(pooled)) if pooled else None,
    }


@torch.no_grad()
def rollout(
    model: nn.Module,
    x: torch.Tensor,
    horizon: int,
    n_steps_input: int,
) -> torch.Tensor:
    window = x
    outputs = []
    for _ in range(horizon):
        out = model(window)
        outputs.append(out[:, 0, ...])
        if n_steps_input == 1:
            window = out
        else:
            window = torch.cat([window[:, 1:, ...], out], dim=1)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def evaluate_rollout(
    model: nn.Module,
    data_root: Path,
    system_names: list[str],
    data_config: dict,
    horizon: int,
    device: torch.device,
    max_samples: int | None = None,
) -> dict:
    n_in = data_config["n_steps_input"]
    include_fields = data_config.get("include_fields")
    per_system: dict[str, dict] = {}
    pooled: list[list[float]] = []
    model.eval()
    for system in system_names:
        ds = system_dataset(
            data_root,
            system,
            "test",
            n_in,
            horizon,
            full_trajectory=True,
            max_rollout_steps=horizon,
            include_fields=include_fields,
        )
        if ds is None or len(ds) == 0:
            continue
        n = len(ds) if max_samples is None else min(len(ds), max_samples)
        scores: list[list[float]] = []
        for i in range(n):
            x, y = ds[i]
            pred = rollout(model, x.unsqueeze(0).to(device), horizon, n_in)[0].cpu()
            mse = ((pred - y) ** 2).mean(dim=(1, 2, 3))
            var = y.var(dim=(1, 2, 3), unbiased=False)
            scores.append((mse / (var + 1e-8)).tolist())
        arr = np.asarray(scores)
        per_system[system] = {
            "horizons": list(range(1, horizon + 1)),
            "nmse_median": np.median(arr, axis=0).tolist(),
            "nmse_mean": arr.mean(axis=0).tolist(),
            "n_samples": n,
        }
        pooled.extend(scores)
        print(
            f"[eval] {system}: median NMSE @ h={horizon} "
            f"{per_system[system]['nmse_median'][-1]:.4f}"
        )
    pooled_arr = np.asarray(pooled) if pooled else np.empty((0, horizon))
    return {
        "per_system": per_system,
        "overall": {
            "nmse_median": np.median(pooled_arr, axis=0).tolist() if len(pooled) else [],
            "nmse_mean": pooled_arr.mean(axis=0).tolist() if len(pooled) else [],
        },
    }


def evaluate(
    model: nn.Module,
    data_root: Path,
    config: dict,
    device: torch.device,
    *,
    max_samples: int | None = None,
) -> dict:
    data_config = config["data"]
    train_systems = list(data_config["datasets"])
    heldout = list(data_config.get("heldout_datasets", []))
    horizon = int(config["eval"]["max_rollout_steps"])
    results: dict[str, object] = {}
    if config["eval"].get("single_step", True):
        results["single_step"] = evaluate_single_step(
            model,
            data_root,
            train_systems,
            data_config,
            int(config["eval"]["batch_size"]),
            device,
        )
    results["rollout"] = evaluate_rollout(
        model,
        data_root,
        train_systems,
        data_config,
        horizon,
        device,
        max_samples=max_samples,
    )
    results["heldout_rollout"] = evaluate_rollout(
        model,
        data_root,
        heldout,
        data_config,
        horizon,
        device,
        max_samples=max_samples,
    )
    return results


def train(
    model: nn.Module,
    loader: DataLoader,
    config: dict,
    device: torch.device,
    steps: int,
) -> tuple[float, float]:
    training = config["training"]
    n_in = int(config["data"]["n_steps_input"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        weight_decay=float(training["weight_decay"]),
        betas=tuple(training["betas"]),
    )

    warmup = int(training.get("warmup_steps", 100))
    min_factor = 0.1

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return max(1e-3, step / max(1, warmup))
        progress = (step - warmup) / max(1, steps - warmup)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    model.train()
    model.to(device)

    running: list[float] = []
    start = time.time()
    iterator = infinite(loader)
    for step in range(1, steps + 1):
        x, y = next(iterator)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)

        loss = torch.zeros((), device=device)
        x_ar = x
        n_out = y.shape[1]
        for k in range(n_out):
            out = model(x_ar)
            loss = loss + criterion(out, y[:, k : k + 1]) / n_out
            x_ar = (
                torch.cat([x_ar[:, 1:, ...], out.detach()], dim=1)
                if n_in > 1
                else out.detach()
            )
        loss.backward()
        if training.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
        optimizer.step()
        scheduler.step()

        running.append(float(loss.detach().cpu()))
        if step % int(training.get("log_every", 50)) == 0 or step == 1:
            recent = float(np.mean(running[-int(training.get("log_every", 50)) :]))
            elapsed = time.time() - start
            print(
                f"[train] step {step:6d}/{steps} loss {recent:.5f} "
                f"lr {scheduler.get_last_lr()[0]:.2e} "
                f"{step / max(elapsed, 1e-6):.2f} steps/s"
            )
    return float(np.mean(running[-100:])), time.time() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf8"))
    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.smoke:
        config["training"]["steps"] = args.steps or 5
        config["data"]["batch_size"] = 2
        config["eval"]["max_rollout_steps"] = 2
        config["eval"]["horizons"] = [1, 2]
        config["training"]["log_every"] = 1

    steps = int(config["training"]["steps"])
    seed = int(config.get("seed", 0))
    set_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    data_root = prepare_data(config, args.smoke, args.data_dir)
    config["data"]["data_dir"] = str(data_root)

    device = resolve_device(config["training"].get("device", "auto"))
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "gphyt")
    size = model_cfg.get("model_size") or model_cfg.get("transformer", {}).get("model_size")
    ablation = []
    if arch == "gphyt":
        if not model_cfg.get("residual", True):
            ablation.append("direct-prediction")
        if not model_cfg.get("transformer", {}).get("use_derivatives", True):
            ablation.append("no-derivatives")
    print(
        f"[config] arch={arch} size={size} systems={data_config['datasets']} "
        f"heldout={data_config.get('heldout_datasets', [])} "
        f"n_in={data_config['n_steps_input']} n_out={data_config['n_steps_output']} "
        f"batch={data_config['batch_size']} steps={steps} "
        f"ablation={','.join(ablation) or 'none'}"
    )
    print(f"[setup] device={device} steps={steps} data={data_root}")

    data_config = copy.deepcopy(config["data"])
    data_config["data_dir"] = str(data_root)
    train_loader, train_ds = make_train_loader(data_config, seed)
    print(f"[data] train windows={len(train_ds)} batches/epoch={len(train_loader)}")

    model, param_count = build_model(
        config["model"],
        int(data_config["n_steps_input"]),
        int(data_config["height"]),
        int(data_config["width"]),
    )
    print(f"[model] parameters={param_count:,}")

    train_loss, train_time = train(model, train_loader, config, device, steps)
    print(f"[train] done loss={train_loss:.5f} time={train_time:.1f}s")

    max_samples = 1 if args.smoke else None
    eval_results = evaluate(model, data_root, config, device, max_samples=max_samples)

    tag = args.tag or config.get("run_name", "run")
    out_dir = Path(config["output_dir"]) / tag
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "run_name": tag,
        "steps": steps,
        "seed": seed,
        "device": str(device),
        "parameter_count": param_count,
        "train_loss": train_loss,
        "train_time_s": train_time,
        "config": config,
        "eval": eval_results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "param_count": param_count,
        },
        out_dir / "model.pt",
    )
    print(f"[done] wrote {out_dir / 'metrics.json'}")
    if "rollout" in eval_results:
        overall = eval_results["rollout"]["overall"]["nmse_median"]
        if overall:
            print(
                "[done] train-system rollout median NMSE "
                + " ".join(f"h{i + 1}={v:.4f}" for i, v in enumerate(overall))
            )
    if "heldout_rollout" in eval_results:
        overall = eval_results["heldout_rollout"]["overall"]["nmse_median"]
        if overall:
            print(
                "[done] held-out rollout median NMSE "
                + " ".join(f"h{i + 1}={v:.4f}" for i, v in enumerate(overall))
            )


if __name__ == "__main__":
    main()
