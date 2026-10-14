#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
PY="${PY:-/home/desc/Conda/envs/torcs_RL/bin/python}"
OUT="$PKG_DIR/rerun/metric2_standard_h20"

mkdir -p "$OUT"
cd "$REPO_ROOT"

PYTHONUNBUFFERED=1 "$PY" scripts/eval_steptrace.py \
  --model-type snn_ours \
  --checkpoint "$PKG_DIR/models/exp012_ours_t4_best.pth" \
  --episodes 1 \
  --max-steps 1000 \
  --run-mode fast_train \
  --center-coef 20.0 \
  --fixed-accel 0.15 \
  --device cuda:0 \
  --quiet \
  --report "$OUT/baseline_t0.15_steptrace.json"

PYTHONUNBUFFERED=1 "$PY" scripts/summarize_decode_control_validity.py \
  --trace "$OUT/baseline_t0.15_steptrace.json" \
  --report "$OUT/baseline_t0.15_H20.json" \
  --horizon 20

echo "Saved: $OUT/baseline_t0.15_H20.json"

