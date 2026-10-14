#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
# Python interpreter: export PY=... to override; otherwise auto-resolve the
# torcs_RL conda env (common install prefixes incl. docker /opt/conda),
# falling back to the active interpreter (activate torcs_RL first).
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
DEVICE="${DEVICE:-cuda}"
OUT="$PKG_DIR/rerun/metric1_latency"

mkdir -p "$OUT"
cd "$REPO_ROOT"

PYTHONUNBUFFERED=1 "$PY" scripts/benchmark_policy_latency.py \
  --baseline-ckpt "$PKG_DIR/models/exp013_baseline_t8_best.pth" \
  --ours-ckpt "$PKG_DIR/models/exp012_ours_t4_best.pth" \
  --report-dir "$OUT" \
  --device "$DEVICE"

echo "Saved: $OUT/latency_benchmark.json"

