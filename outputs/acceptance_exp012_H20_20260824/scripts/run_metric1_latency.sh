#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd)"
PY="${PY:-/home/desc/Conda/envs/torcs_RL/bin/python}"
OUT="$PKG_DIR/rerun/metric1_latency"

mkdir -p "$OUT"
cd "$REPO_ROOT"

PYTHONUNBUFFERED=1 "$PY" scripts/benchmark_policy_latency.py \
  --baseline-ckpt "$PKG_DIR/models/exp013_baseline_t8_best.pth" \
  --ours-ckpt "$PKG_DIR/models/exp012_ours_t4_best.pth" \
  --report-dir "$OUT" \
  --device cuda

echo "Saved: $OUT/latency_benchmark.json"

