#!/usr/bin/env bash
# G3 acceptance demo: launch TORCS (headless, Xvfb) + load the baseline SNN
# (exp013, T=8) + 200 closed-loop steps + per-step telemetry, and regenerate
# data/replay_example.json (sample input for the display-software replay mode).
#
# Acceptance: per-step lines + final "TORCS terminated." + fresh JSON = PASS.
# Requires: TORCS installed (see DEPLOY_TORCS.md) + conda env torcs_RL.
#   Full lap replay data: change --max-steps to 1000.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"   # cwd must be repo root: gym_torcs launches `bash autostart.sh` relative to cwd

# --- activate conda env torcs_RL (conda on PATH first, then common prefixes) ---
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/Conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/Conda/etc/profile.d/conda.sh"
else
    echo "[run.sh] conda not found; assuming current 'python' already resolves to torcs_RL." >&2
fi
conda activate torcs_RL 2>/dev/null || echo "[run.sh] env torcs_RL not found; using current python" >&2

PYTHON_BIN="${PY:-python}"

PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/eval_steptrace.py \
    --model-type snn_baseline \
    --checkpoint outputs/acceptance_exp012_H20_20260824/models/exp013_baseline_t8_best.pth \
    --config configs/default.yaml \
    --episodes 1 \
    --max-steps 200 \
    --run-mode fast_train \
    --track g-track-1 \
    --report data/replay_example.json

echo
echo "[run.sh] done. Replay data: data/replay_example.json"
