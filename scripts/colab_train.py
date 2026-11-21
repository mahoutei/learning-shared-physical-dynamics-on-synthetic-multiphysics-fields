"""Run the reduced GPhyT reproduction on a Google Colab VM.

This script is intended to be sent to a Colab runtime with the official
Colab CLI.  It clones the public reproduction branch, installs the locked
environment with uv, runs training, and prints the resulting metrics.

Example::

    colab new -s gphyt --gpu T4
    colab exec -s gphyt -f scripts/colab_train.py --timeout 7200
    colab download -s gphyt /content/gphyt-repro/results/repro/gphyt_s_baseline/metrics.json ./metrics.json
    colab stop -s gphyt

Environment variables:

- ``GPHYT_REPO_URL``: repository to clone (default: public reproduction repo).
- ``GPHYT_BRANCH``: branch to run (default: ``repro``).
- ``GPHYT_REPO_DIR``: checkout directory (default: ``/content/gphyt-repro``).
- ``GPHYT_SMOKE``: set to ``1`` for the five-step smoke run.
- ``GPHYT_STEPS``: override the configured training steps.
- ``GPHYT_TAG``: override the output directory name.
- ``GPHYT_DATA_DIR``: override the corpus directory.
- ``GPHYT_DEVICE``: torch device (default: ``cuda``).
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
BRANCH = os.environ.get("GPHYT_BRANCH", "repro")
REPO_DIR = Path(os.environ.get("GPHYT_REPO_DIR", "/content/gphyt-repro"))
SMOKE = os.environ.get("GPHYT_SMOKE", "0") == "1"
STEPS = os.environ.get("GPHYT_STEPS")
TAG = os.environ.get("GPHYT_TAG")
DATA_DIR = os.environ.get("GPHYT_DATA_DIR")
DEVICE = os.environ.get("GPHYT_DEVICE", "cuda")


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

    ensure_checkout()
    uv = resolve_uv()

    cmd = [uv, "run", "--locked", "python", "-m", "gphyt.repro.run"]
    cmd += ["--device", DEVICE]
    if SMOKE:
        cmd.append("--smoke")
    if STEPS:
        cmd += ["--steps", STEPS]
    if TAG:
        cmd += ["--tag", TAG]
    if DATA_DIR:
        cmd += ["--data-dir", DATA_DIR]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    run(cmd, cwd=REPO_DIR, env=env)

    results = sorted((REPO_DIR / "results" / "repro").glob("*/metrics.json"))
    if not results:
        raise SystemExit("training finished but no metrics.json was written")
    latest = max(results, key=lambda path: path.stat().st_mtime)
    print(f"[colab] metrics_path={latest}", flush=True)
    print(latest.read_text(encoding="utf8"), flush=True)


if __name__ == "__main__":
    main()
