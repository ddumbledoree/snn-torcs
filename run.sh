#!/usr/bin/env bash
# Quick demo entry: runs the handoff acceptance loop (launches TORCS headless,
# loads model/exp013_best.pth, 200 closed-loop steps, per-step telemetry,
# regenerates snn_torcs_handoff/data/replay_example.json).
# Full stack: TORCS installed + conda env torcs_RL active (see DEPLOY_TORCS.md).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/snn_torcs_handoff/run.sh"
