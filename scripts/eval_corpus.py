"""Build a native-resolution evaluation corpus for released checkpoints.

The reduced reproduction corpus is 64x32.  Released checkpoints were trained
on 256x128 fields and cannot be evaluated on the reduced grid without changing
their positional encoding.  This module builds a small, deterministic 256x128
corpus for zero-shot evaluation only: four training trajectories per system
(for normalization statistics), one validation trajectory, and four test
trajectories.  It is not a replacement for the paper's training corpus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py

from gphyt.data.normalize import compute_statistics
from gphyt.repro import corpus, systems

EVAL_SPLITS = {"train": 4, "valid": 1, "test": 4}
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 128
DEFAULT_FRAMES = 29
META_NAME = ".eval_corpus.json"


def _selected_systems(system_names: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    selected = tuple(systems.TRAIN_SYSTEMS + systems.HELDOUT_SYSTEMS) if system_names is None else tuple(system_names)
    unknown = [name for name in selected if name not in systems.GENERATORS]
    if unknown:
        raise ValueError(f"unknown systems: {unknown}")
    return selected


def meta_path(root: Path) -> Path:
    return Path(root) / META_NAME


def _shape(path: Path) -> tuple[int, int, int, int]:
    with h5py.File(path, "r") as handle:
        pressure = handle["t0_fields"]["pressure"]
        n_traj, n_frames, height, width = pressure.shape
    return n_traj, n_frames, height, width


def check_eval_corpus(
    root: Path,
    *,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    n_frames: int = DEFAULT_FRAMES,
    system_names: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Return True when a compatible evaluation corpus is already present."""
    root = Path(root)
    meta = meta_path(root)
    if not meta.exists():
        return False
    try:
        info = json.loads(meta.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        int(info.get("height", -1)) != height
        or int(info.get("width", -1)) != width
        or int(info.get("n_frames", -1)) != n_frames
    ):
        return False
    for system in _selected_systems(system_names):
        system_root = root / system
        if not (system_root / "stats.yaml").exists():
            return False
        for split, expected in EVAL_SPLITS.items():
            path = system_root / "data" / split / "data.h5"
            if not path.exists():
                return False
            n_traj, frames, h, w = _shape(path)
            if (n_traj, frames, h, w) != (expected, n_frames, height, width):
                return False
    return True


def _existing_is_compatible(
    path: Path,
    *,
    expected_trajectories: int,
    height: int,
    width: int,
    n_frames: int,
) -> bool:
    if not path.exists():
        return False
    n_traj, frames, h, w = _shape(path)
    return (n_traj, h, w) == (expected_trajectories, height, width) and frames >= n_frames


def build_eval_corpus(
    root: Path,
    *,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    n_frames: int = DEFAULT_FRAMES,
    base_seed: int = 0,
    overwrite: bool = False,
    system_names: tuple[str, ...] | list[str] | None = None,
) -> Path:
    """Generate the 256x128 evaluation corpus and its normalization stats."""
    if n_frames < 28:
        raise ValueError("n_frames must be at least 28 for a 24-step rollout")
    root = Path(root).expanduser().resolve()
    selected = _selected_systems(system_names)
    if check_eval_corpus(
        root, height=height, width=width, n_frames=n_frames, system_names=selected
    ):
        print(f"[eval-corpus] compatible corpus already exists at {root}")
        return root

    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for system in selected:
        system_root = root / system
        regenerated = False
        for split, n_traj in EVAL_SPLITS.items():
            path = system_root / "data" / split / "data.h5"
            if not overwrite and _existing_is_compatible(
                path,
                expected_trajectories=n_traj,
                height=height,
                width=width,
                n_frames=n_frames,
            ):
                print(f"[eval-corpus] {system}/{split}: exists, skipping")
                continue
            print(
                f"[eval-corpus] {system}/{split}: generating "
                f"{n_traj} trajectories at {height}x{width}, {n_frames} frames",
                flush=True,
            )
            split_started = time.time()
            data = systems.generate(
                system,
                n_traj=n_traj,
                n_frames=n_frames,
                h=height,
                w=width,
                seed=corpus.system_seed(base_seed, system, split),
            ).numpy()
            corpus.write_well_file(path, data, dataset_name=system, h=height, w=width)
            regenerated = True
            print(
                f"[eval-corpus] {system}/{split}: "
                f"{time.time() - split_started:.1f}s",
                flush=True,
            )

        stats_path = system_root / "stats.yaml"
        if overwrite or regenerated or not stats_path.exists():
            if stats_path.exists():
                stats_path.unlink()
            print(f"[eval-corpus] {system}: computing stats", flush=True)
            compute_statistics(system_root / "data" / "train", stats_path)
        print(f"[eval-corpus] {system}: done", flush=True)

    info = {
        "height": height,
        "width": width,
        "n_frames": n_frames,
        "base_seed": base_seed,
        "splits": EVAL_SPLITS,
        "systems": list(selected),
        "purpose": "zero-shot released-checkpoint evaluation",
    }
    meta_path(root).write_text(json.dumps(info, indent=2) + "\n", encoding="utf8")
    total_bytes = sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )
    print(
        f"[eval-corpus] ready at {root} "
        f"({total_bytes / 1e9:.2f} GB, {time.time() - started:.1f}s)",
        flush=True,
    )
    return root


def corpus_info(root: Path) -> dict:
    root = Path(root)
    meta = meta_path(root)
    if not meta.exists():
        return {"path": str(root), "exists": False}
    info = json.loads(meta.read_text(encoding="utf8"))
    info["path"] = str(root)
    info["exists"] = True
    return info
