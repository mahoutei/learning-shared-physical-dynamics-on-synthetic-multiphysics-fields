"""Run released-checkpoint evaluation on a Google Colab T4 runtime.

The official Colab CLI sends this file to the VM.  It clones the evaluation
branch, installs the locked environment with uv, downloads the requested
released checkpoint, builds/uses the 256x128 evaluation corpus, and runs
``scripts/eval_released.py``.

Example::

    colab new -s gphyt-eval --gpu T4
    colab exec -s gphyt-eval -f scripts/colab_eval_released.py --timeout 7200 \
      --env GPHYT_MODEL_SIZE=M
    colab download -s gphyt-eval \
      /content/gphyt-repro/results/released/released_m_256/metrics.json \
      colab-runs/data/released_m_256-t4.json
    colab stop -s gphyt-eval

Environment variables:
- ``GPHYT_REPO_URL``: repository to clone.
- ``GPHYT_BRANCH``: branch to run (default: ``orx/released-scaling-eval``).
- ``GPHYT_REPO_DIR``: checkout directory (default: ``/content/gphyt-repro``).
- ``GPHYT_MODEL_SIZE``: S, M, L, or XL (default: M).
- ``GPHYT_DATA_DIR``: evaluation corpus directory.
- ``GPHYT_OUTPUT_DIR``: results directory (default: ``results/released``).
- ``GPHYT_TAG``: output tag (default: ``released_<size>_256``).
- ``GPHYT_DEVICE``: torch device (default: cuda).
- ``GPHYT_MAX_SAMPLES``: cap test trajectories per system (smoke runs).
- ``GPHYT_HORIZON``: rollout horizon (default: 24).
- ``GPHYT_CHECKPOINT``: local checkpoint path, bypassing download.
- ``GPHYT_REBUILD_CORPUS``: set to 1 to regenerate the evaluation corpus.
- ``GPHYT_NO_BUILD_CORPUS``: set to 1 to require an existing corpus.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = os.environ.get(
    "GPHYT_REPO_URL",
    "https://github.com/mahoutei/learning-shared-physical-dynamics-on-synthetic-multiphysics-fields.git",
)
BRANCH = os.environ.get("GPHYT_BRANCH", "orx/released-scaling-eval")
REPO_DIR = Path(os.environ.get("GPHYT_REPO_DIR", "/content/gphyt-repro"))
MODEL_SIZE = os.environ.get("GPHYT_MODEL_SIZE", "M").upper()
DATA_DIR = os.environ.get("GPHYT_DATA_DIR", "/content/gphyt-eval-corpus-256")
OUTPUT_DIR = os.environ.get("GPHYT_OUTPUT_DIR", "results/released")
TAG = os.environ.get("GPHYT_TAG")
DEVICE = os.environ.get("GPHYT_DEVICE", "cuda")
MAX_SAMPLES = os.environ.get("GPHYT_MAX_SAMPLES")
HORIZON = os.environ.get("GPHYT_HORIZON", "24")
CHECKPOINT = os.environ.get("GPHYT_CHECKPOINT")
REBUILD_CORPUS = os.environ.get("GPHYT_REBUILD_CORPUS", "0") == "1"
NO_BUILD_CORPUS = os.environ.get("GPHYT_NO_BUILD_CORPUS", "0") == "1"


def run(args: list[str], **kwargs) -> None:
    print("+", " ".join(str(arg) for arg in args), flush=True)
    subprocess.run(args, check=True, **kwargs)


def ensure_checkout() -> None:
    if REPO_DIR.exists():
        run(["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", BRANCH])
        run(["git", "-C", str(REPO_DIR), "checkout", "-B", BRANCH, "FETCH_HEAD"])
        run(["git", "-C", str(REPO_DIR), "reset", "--hard", "FETCH_HEAD"])
    else:
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                BRANCH,
                REPO_URL,
                str(REPO_DIR),
            ]
        )


def resolve_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv = shutil.which("uv")
    if uv:
        return uv
    fallback = Path.home() / ".local" / "bin" / "uv"
    if fallback.exists():
        return str(fallback)
    raise SystemExit("uv is not available on the Colab runtime")


def main() -> None:
    import torch

    print(
        f"[colab] python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"[colab] gpu={torch.cuda.get_device_name(0)}", flush=True)
    elif DEVICE == "cuda":
        raise SystemExit("CUDA requested but no GPU is available")
    if MODEL_SIZE not in {"S", "M", "L", "XL"}:
        raise SystemExit(f"invalid GPHYT_MODEL_SIZE={MODEL_SIZE!r}")

    ensure_checkout()
    uv = resolve_uv()
    tag = TAG or f"released_{MODEL_SIZE.lower()}_256"

    cmd = [
        uv,
        "run",
        "--locked",
        "python",
        "scripts/eval_released.py",
        "--model-size",
        MODEL_SIZE,
        "--device",
        DEVICE,
        "--data-dir",
        DATA_DIR,
        "--output-dir",
        OUTPUT_DIR,
        "--tag",
        tag,
        "--horizon",
        HORIZON,
    ]
    if MAX_SAMPLES:
        cmd += ["--max-samples", MAX_SAMPLES]
    if CHECKPOINT:
        cmd += ["--checkpoint", CHECKPOINT]
    if REBUILD_CORPUS:
        cmd += ["--rebuild-corpus"]
    if NO_BUILD_CORPUS:
        cmd += ["--no-build-corpus"]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    run(cmd, cwd=REPO_DIR, env=env)

    metrics = REPO_DIR / OUTPUT_DIR / tag / "metrics.json"
    if not metrics.exists():
        raise SystemExit(f"evaluation finished but no metrics.json at {metrics}")
    print(f"[colab] metrics_path={metrics}", flush=True)
    print(metrics.read_text(encoding="utf8"), flush=True)


if __name__ == "__main__":
    main()
