"""Build the reduced multi-physics corpus in The Well HDF5 layout.

The corpus has one directory per physical system::

    corpus/<system>/data/train/data.h5
    corpus/<system>/data/valid/data.h5
    corpus/<system>/data/test/data.h5
    corpus/<system>/stats.yaml

Fields are stored exactly as in The Well: ``t0_fields`` for scalars and
``t1_fields`` for the two velocity components.  Normalization statistics are
computed with the repository's own ``compute_statistics`` so that
``PhysicsDataset`` sees the same format as the paper's datasets.
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import torch

from gphyt.data.normalize import compute_statistics
from gphyt.repro import systems

SPLIT_TRAJECTORIES = {"train": 24, "valid": 4, "test": 4}
SCALAR_FIELDS = ("pressure", "density", "temperature")
VECTOR_FIELD = "velocity"


def _write_dimension(group: h5py.Group, name: str, values: np.ndarray) -> None:
    ds = group.create_dataset(name, data=values.astype(np.float32))
    ds.attrs["sample_varying"] = False
    ds.attrs["time_varying"] = False


def _write_field(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    dim_varying: list[bool],
) -> None:
    ds = group.create_dataset(name, data=values.astype(np.float32))
    ds.attrs["dim_varying"] = dim_varying
    ds.attrs["sample_varying"] = True
    ds.attrs["time_varying"] = True


def write_well_file(
    path: Path,
    data: np.ndarray,
    dataset_name: str,
    h: int,
    w: int,
) -> None:
    """Write ``data`` of shape (n_traj, n_frames, h, w, 5) in The Well layout."""
    n_traj, n_frames, _, _, n_channels = data.shape
    assert n_channels == systems.N_CHANNELS, data.shape
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = dataset_name
        f.attrs["n_trajectories"] = n_traj
        f.attrs["n_spatial_dims"] = 2
        f.attrs["grid_type"] = "cartesian"
        f.attrs["simulation_parameters"] = []

        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        _write_dimension(dims, "time", np.arange(n_frames))
        _write_dimension(dims, "x", np.arange(h) / h)
        _write_dimension(dims, "y", np.arange(w) / w)

        bcs = f.create_group("boundary_conditions")
        for dim, size in (("x", h), ("y", w)):
            bc = bcs.create_group(f"{dim}_periodic")
            bc.attrs["bc_type"] = "PERIODIC"
            bc.attrs["associated_dims"] = [dim]
            bc.create_dataset("mask", data=np.ones(size, dtype=bool))

        scalars = f.create_group("scalars")
        scalars.attrs["field_names"] = []

        t0 = f.create_group("t0_fields")
        t0.attrs["field_names"] = list(SCALAR_FIELDS)
        for i, name in enumerate(SCALAR_FIELDS):
            _write_field(t0, name, data[..., i], [True, True])

        t1 = f.create_group("t1_fields")
        t1.attrs["field_names"] = [VECTOR_FIELD]
        _write_field(t1, VECTOR_FIELD, data[..., 3:5], [True, True])

        t2 = f.create_group("t2_fields")
        t2.attrs["field_names"] = []


def system_seed(base_seed: int, system: str, split: str) -> int:
    split_idx = {"train": 0, "valid": 1, "test": 2}[split]
    return base_seed + 1000 * split_idx + 17 * sum(map(ord, system))


def build_system(
    root: Path,
    system: str,
    *,
    n_frames: int,
    h: int,
    w: int,
    base_seed: int,
    overwrite: bool = False,
) -> None:
    system_root = root / system
    if system_root.exists() and not overwrite:
        stats = system_root / "stats.yaml"
        if stats.exists():
            print(f"[corpus] {system}: exists, skipping")
            return
    for split, n_traj in SPLIT_TRAJECTORIES.items():
        path = system_root / "data" / split / "data.h5"
        if path.exists() and not overwrite:
            print(f"[corpus] {system}/{split}: exists, skipping")
            continue
        print(f"[corpus] {system}/{split}: generating {n_traj} trajectories")
        data = systems.generate(
            system,
            n_traj=n_traj,
            n_frames=n_frames,
            h=h,
            w=w,
            seed=system_seed(base_seed, system, split),
        ).numpy()
        write_well_file(path, data, dataset_name=system, h=h, w=w)
    stats_path = system_root / "stats.yaml"
    if not stats_path.exists() or overwrite:
        if stats_path.exists():
            stats_path.unlink()
        compute_statistics(system_root / "data" / "train", stats_path)
    print(f"[corpus] {system}: done")


def build_corpus(
    root: Path,
    *,
    n_frames: int = 49,
    h: int = 64,
    w: int = 32,
    base_seed: int = 0,
    overwrite: bool = False,
) -> Path:
    root = Path(root)
    for system in systems.TRAIN_SYSTEMS + systems.HELDOUT_SYSTEMS:
        build_system(
            root,
            system,
            n_frames=n_frames,
            h=h,
            w=w,
            base_seed=base_seed,
            overwrite=overwrite,
        )
    return root


def check_corpus(root: Path) -> bool:
    root = Path(root)
    for system in systems.TRAIN_SYSTEMS + systems.HELDOUT_SYSTEMS:
        if not (root / system / "stats.yaml").exists():
            return False
        for split in SPLIT_TRAJECTORIES:
            if not (root / system / "data" / split / "data.h5").exists():
                return False
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("corpus"))
    parser.add_argument("--n-frames", type=int, default=49)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--w", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_corpus(
        args.root,
        n_frames=args.n_frames,
        h=args.h,
        w=args.w,
        base_seed=args.seed,
        overwrite=args.overwrite,
    )
