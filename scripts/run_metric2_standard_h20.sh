#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
# Python interpreter: export PY=... to override; otherwise auto-resolve the
# torcs_RL conda env, falling back to the active interpreter.
if [ -z "${PY:-}" ]; then
    for _c in "$HOME/Conda/envs/torcs_RL/bin/python" \
              "$HOME/miniconda3/envs/torcs_RL/bin/python" \
              "$HOME/anaconda3/envs/torcs_RL/bin/python" \
              "/opt/conda/envs/torcs_RL/bin/python"; do
        [ -x "$_c" ] && PY="$_c" && break
    done
fi
PY="${PY:-python3}"
# Compute device: export DEVICE=cpu on GPU-less boxes.
DEVICE="${DEVICE:-cuda:0}"
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
  --device "$DEVICE" \
  --quiet \
  --report "$OUT/baseline_t0.15_steptrace.json"

PYTHONUNBUFFERED=1 "$PY" scripts/summarize_decode_control_validity.py \
  --trace "$OUT/baseline_t0.15_steptrace.json" \
  --report "$OUT/baseline_t0.15_H20.json" \
  --horizon 20

echo "Saved: $OUT/baseline_t0.15_H20.json"

