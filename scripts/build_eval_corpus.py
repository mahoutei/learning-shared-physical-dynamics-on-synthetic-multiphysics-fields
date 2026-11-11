"""Build the 256x128 zero-shot evaluation corpus used by released models."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_corpus import DEFAULT_FRAMES, DEFAULT_HEIGHT, DEFAULT_WIDTH, build_eval_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("~/.cache/gphyt_repro/eval_corpus_256"),
        help="corpus directory",
    )
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--n-frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--systems",
        default=None,
        help="comma-separated subset (default: all six)",
    )
    args = parser.parse_args()
    build_eval_corpus(
        args.root,
        height=args.height,
        width=args.width,
        n_frames=args.n_frames,
        base_seed=args.seed,
        overwrite=args.overwrite,
        system_names=None if args.systems is None else tuple(args.systems.split(",")),
    )


if __name__ == "__main__":
    main()
