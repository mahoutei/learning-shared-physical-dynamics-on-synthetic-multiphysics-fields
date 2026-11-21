#!/usr/bin/env bash
# Run every GPhyT reproduction variant sequentially on one Colab session.
#
# Prerequisites:
#   colab whoami
#   colab new -s gphyt-all --gpu T4
#
# Usage:
#   bash scripts/colab_all.sh gphyt-all
#
# Environment:
#   GPHYT_STEPS  training steps (default: 2000)
#   OUT_DIR      output directory (default: colab-runs)

set -euo pipefail

SESSION="${1:-gphyt-all}"
OUT_DIR="${2:-colab-runs}"
STEPS="${GPHYT_STEPS:-2000}"

if ! command -v colab >/dev/null 2>&1; then
  echo "error: the official colab CLI is not on PATH" >&2
  exit 1
fi

VARIANTS=(
  "repro|gphyt_s_baseline_colab"
  "orx/context-length-1-frame|gphyt_s_context_1"
  "orx/context-length-2-frames|gphyt_s_context_2"
  "orx/context-length-8-frames|gphyt_s_context_8"
  "orx/ablation-direct-prediction-no-residual|gphyt_s_direct_prediction"
  "orx/ablation-no-derivative-channels|gphyt_s_no_derivatives"
  "orx/baseline-unet-s-direct-predictor|unet_s_baseline"
)

mkdir -p "$OUT_DIR/data" "$OUT_DIR/models"

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r branch tag <<< "$entry"
  echo "===== START ${tag} (${branch}) ====="

  colab exec -s "$SESSION" -f scripts/colab_train.py --timeout 7200 \
    --env GPHYT_BRANCH="$branch" \
    --env GPHYT_DEVICE=cuda \
    --env GPHYT_STEPS="$STEPS" \
    --env GPHYT_TAG="$tag" \
    --env GPHYT_SMOKE=0 \
    --env GPHYT_DATA_DIR=/content/gphyt-corpus

  colab download -s "$SESSION" \
    "/content/gphyt-repro/results/repro/${tag}/metrics.json" \
    "${OUT_DIR}/data/${tag}-t4.json"
  colab download -s "$SESSION" \
    "/content/gphyt-repro/results/repro/${tag}/model.pt" \
    "${OUT_DIR}/models/${tag}-t4.pt"

  echo "===== DONE ${tag} ====="
done

echo "All variants complete."
