#!/usr/bin/env bash
# Evaluate released GPhyT checkpoints sequentially on one Colab T4 session.
#
# Prerequisites:
#   colab whoami
#   colab new -s gphyt-eval --gpu T4
#
# Usage:
#   bash scripts/colab_eval_all.sh gphyt-eval colab-runs
#
# Environment:
#   GPHYT_BRANCH  evaluation branch (default: orx/released-scaling-eval)
#   GPHYT_HORIZON rollout horizon (default: 24)

set -euo pipefail

SESSION="${1:-gphyt-eval}"
OUT_DIR="${2:-colab-runs}"
BRANCH="${GPHYT_BRANCH:-orx/released-scaling-eval}"
HORIZON="${GPHYT_HORIZON:-24}"

if ! command -v colab >/dev/null 2>&1; then
  echo "error: the official colab CLI is not on PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR/data"

for SIZE in S M L XL; do
  TAG="released_${SIZE,,}_256"
  echo "===== START ${TAG} ====="

  colab exec -s "$SESSION" -f scripts/colab_eval_released.py --timeout 7200 \
    --env GPHYT_BRANCH="$BRANCH" \
    --env GPHYT_MODEL_SIZE="$SIZE" \
    --env GPHYT_DEVICE=cuda \
    --env GPHYT_HORIZON="$HORIZON" \
    --env GPHYT_DATA_DIR=/content/gphyt-eval-corpus-256 \
    --env GPHYT_OUTPUT_DIR=results/released

  colab download -s "$SESSION" \
    "/content/gphyt-repro/results/released/${TAG}/metrics.json" \
    "${OUT_DIR}/data/${TAG}-t4.json"

  echo "===== DONE ${TAG} ====="
done

echo "All released checkpoints complete."
