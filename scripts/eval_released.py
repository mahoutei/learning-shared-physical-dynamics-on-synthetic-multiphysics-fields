"""Evaluate released GPhyT checkpoints on the native 256x128 grid.

Released checkpoints use absolute positional encodings for 4 input frames at
256x128 (512 tokens), so they cannot be loaded against the 64x32 reduced
corpus.  This script builds a small 256x128 evaluation corpus, loads a released
checkpoint strictly, and evaluates autoregressive rollout NMSE zero-shot on all
six synthetic systems.

Example::

    uv run --locked python scripts/eval_released.py --model-size M --device cuda

On Colab, ``scripts/colab_eval_released.py`` wraps this script and downloads
the checkpoint automatically.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time
import urllib.request
from pathlib import Path

import torch
import yaml
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from eval_corpus import (
    DEFAULT_FRAMES,
    build_eval_corpus,
    check_eval_corpus,
    corpus_info,
)
from gphyt.repro import systems
from gphyt.repro.run import REPO_ROOT, build_model, evaluate, resolve_device, set_seed

DEFAULT_CONFIG = REPO_ROOT / "gphyt" / "repro" / "config.yaml"
HF_BASE = "https://huggingface.co/flwi/Physics-Foundation-Model/resolve/main"
MODEL_CHOICES = ("S", "M", "L", "XL")


def download_checkpoint(size: str, target: Path, base_url: str) -> Path:
    """Download a released checkpoint with a simple progress indicator."""
    target = Path(target)
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"[checkpoint] using {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/gphyt-{size}.pth"
    part = target.with_suffix(target.suffix + ".part")
    print(f"[checkpoint] downloading {url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "gphyt-eval/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        with part.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024 * 8)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100.0 * downloaded / total
                    print(
                        f"\r[checkpoint] {downloaded / 1e6:.0f}/"
                        f"{total / 1e6:.0f} MB ({pct:.0f}%)",
                        end="",
                        flush=True,
                    )
    print(flush=True)
    if part.stat().st_size < 1_000_000:
        raise RuntimeError(f"downloaded checkpoint is unexpectedly small: {part}")
    part.replace(target)
    print(f"[checkpoint] saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def load_released_state(path: Path) -> dict:
    """Load a possibly DDP/compiled checkpoint and strip key prefixes."""
    torch.serialization.add_safe_globals(
        [Path, pathlib.PosixPath, pathlib.WindowsPath]
    )
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {type(obj)}")
    state = dict(state)
    consume_prefix_in_state_dict_if_present(state, "module.")
    consume_prefix_in_state_dict_if_present(state, "_orig_mod.")
    state.pop("_metadata", None)
    if not state:
        raise ValueError("checkpoint state dict is empty")
    return state


def make_config(args: argparse.Namespace, data_dir: Path, out_dir: Path) -> dict:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf8"))
    config = dict(config)
    config["run_name"] = args.tag
    config["seed"] = 0
    config["data"] = dict(config["data"])
    config["data"]["data_dir"] = str(data_dir)
    config["data"]["height"] = 256
    config["data"]["width"] = 128
    config["data"]["n_steps_input"] = 4
    config["data"]["n_steps_output"] = 2
    config["data"]["batch_size"] = 1
    config["data"]["datasets"] = list(systems.TRAIN_SYSTEMS + systems.HELDOUT_SYSTEMS)
    config["data"]["heldout_datasets"] = []
    config["model"] = dict(config["model"])
    config["model"]["transformer"] = dict(config["model"]["transformer"])
    config["model"]["transformer"]["model_size"] = f"GPT_{args.model_size}"
    config["eval"] = dict(config["eval"])
    config["eval"]["max_rollout_steps"] = int(args.horizon)
    config["eval"]["batch_size"] = 1
    config["eval"]["single_step"] = False
    config["output_dir"] = str(out_dir)
    return config


def ensure_corpus(
    args: argparse.Namespace,
    data_dir: Path,
    system_names: tuple[str, ...] | None,
) -> None:
    if args.no_build_corpus:
        if not check_eval_corpus(
            data_dir, n_frames=args.n_frames, system_names=system_names
        ):
            raise SystemExit(
                f"no compatible evaluation corpus at {data_dir}; "
                "remove --no-build-corpus or build it first"
            )
        return
    build_eval_corpus(
        data_dir,
        n_frames=args.n_frames,
        base_seed=args.corpus_seed,
        overwrite=args.rebuild_corpus,
        system_names=system_names,
    )


def summarize(eval_results: dict, horizon: int) -> dict:
    per_system: dict[str, dict] = {}
    for name, values in eval_results.get("rollout", {}).get("per_system", {}).items():
        medians = values["nmse_median"]
        per_system[name] = {
            "h1": float(medians[0]),
            "h8": float(medians[min(7, horizon - 1)]),
            "h24": float(medians[-1]),
            "n_samples": int(values["n_samples"]),
        }
    overall = eval_results.get("rollout", {}).get("overall", {})
    overall_median = overall.get("nmse_median") or []
    overall_summary = None
    if overall_median:
        overall_summary = {
            "h1": float(overall_median[0]),
            "h8": float(overall_median[min(7, horizon - 1)]),
            "h24": float(overall_median[-1]),
        }
    return {"per_system": per_system, "overall": overall_summary}


def print_summary(summary: dict) -> None:
    for name, values in summary["per_system"].items():
        print(
            f"[summary] {name:20s} h1={values['h1']:.6g} "
            f"h8={values['h8']:.6g} h24={values['h24']:.6g} "
            f"n={values['n_samples']}"
        )
    overall = summary.get("overall")
    if overall:
        print(
            f"[summary] {'all systems':20s} h1={overall['h1']:.6g} "
            f"h8={overall['h8']:.6g} h24={overall['h24']:.6g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CHOICES, default="M")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("~/.cache/gphyt_repro/eval_corpus_256"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/released"))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--n-frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--corpus-seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--hf-base", default=HF_BASE)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-build-corpus", action="store_true")
    parser.add_argument("--rebuild-corpus", action="store_true")
    parser.add_argument(
        "--corpus-systems",
        default=None,
        help="comma-separated corpus subset for smoke runs (default: all six)",
    )
    args = parser.parse_args()

    if args.horizon < 1 or args.horizon > args.n_frames - 4:
        parser.error("horizon must fit in n_frames - 4")
    if args.tag is None:
        args.tag = f"released_{args.model_size.lower()}_256"

    data_dir = args.data_dir.expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    out_dir = args.output_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir = out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but no GPU is available")
    print(f"[eval] device={device} model=GPT_{args.model_size} tag={args.tag}")

    set_seed(0)
    system_names = (
        None if args.corpus_systems is None else tuple(args.corpus_systems.split(","))
    )
    ensure_corpus(args, data_dir, system_names)
    config = make_config(args, data_dir, out_dir)

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = REPO_ROOT / "checkpoints" / f"gphyt-{args.model_size}.pth"
        if args.no_download:
            raise SystemExit(f"missing checkpoint {checkpoint} and --no-download is set")
        download_checkpoint(args.model_size, checkpoint, args.hf_base)
    else:
        checkpoint = checkpoint.expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (REPO_ROOT / checkpoint).resolve()
        if not checkpoint.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint}")

    model, parameter_count = build_model(
        config["model"],
        int(config["data"]["n_steps_input"]),
        int(config["data"]["height"]),
        int(config["data"]["width"]),
    )
    state = load_released_state(checkpoint)
    state_len = len(state)
    state_tensors = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    print(
        f"[model] params={parameter_count:,} state_tensors={len(state)} "
        f"state_values={state_tensors:,}"
    )
    result = model.load_state_dict(state, strict=True)
    print(f"[model] strict load: {result}")
    del state
    gc.collect()

    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    eval_results = evaluate(
        model,
        data_dir,
        config,
        device,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - started
    peak_gpu_gb = None
    if device.type == "cuda":
        peak_gpu_gb = torch.cuda.max_memory_allocated(device) / 1e9

    summary = summarize(eval_results, args.horizon)
    metrics = {
        "run_name": args.tag,
        "kind": "released_checkpoint_eval",
        "model_size": f"GPT_{args.model_size}",
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameter_count": parameter_count,
        "state_tensors": state_len,
        "device": str(device),
        "eval_time_s": elapsed,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "corpus": corpus_info(data_dir),
        "summary": summary,
        "config": config,
        "eval": eval_results,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf8"
    )
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf8"
    )
    print_summary(summary)
    print(f"[eval] metrics={out_dir / 'metrics.json'}")
    if peak_gpu_gb is not None:
        print(f"[eval] peak_gpu_memory_gb={peak_gpu_gb:.2f}")


if __name__ == "__main__":
    main()
