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
# NOTE: requires snn_torcs/condition.py (placeholder in this repo — see
# docs/condition_module_spec.md); every condition routes through eval_condition.py.
DEVICE="${DEVICE:-cuda:0}"
OUT="$PKG_DIR/rerun/condition_matrix_h20"
COMMON=(--model-type snn_ours --checkpoint "$PKG_DIR/models/exp012_ours_t4_best.pth" --episodes 1 --run-mode fast_train --center-coef 20.0 --quiet --device "$DEVICE")

mkdir -p "$OUT"
cd "$REPO_ROOT"

run_one() {
  local condition="$1"
  local accel="$2"
  local label="$3"
  local max_steps="$4"
  local trace="$OUT/${label}_steptrace.json"
  local h20="$OUT/${label}_H20.json"
  local log="$OUT/${label}.log"

  echo "=== $label max_steps=$max_steps ==="
  set +e
  PYTHONUNBUFFERED=1 timeout 900 "$PY" scripts/eval_condition.py \
    --condition "$condition" \
    --fixed-accel "$accel" \
    "${COMMON[@]}" \
    --max-steps "$max_steps" \
    --report "$trace" \
    > "$log" 2>&1
  local rc=$?
  set -e

  tail -30 "$log" || true
  if [[ $rc -eq 0 && -f "$trace" ]]; then
    PYTHONUNBUFFERED=1 "$PY" scripts/summarize_decode_control_validity.py \
      --trace "$trace" \
      --report "$h20" \
      --horizon 20
  else
    echo "NO_COMPLETE_TRACE label=$label rc=$rc" | tee -a "$log"
  fi
}

run_one baseline 0.10 baseline_t0.10 1000
run_one baseline 0.15 baseline_t0.15 1000
run_one baseline 0.20 baseline_t0.20 1000
run_one baseline 0.25 baseline_t0.25 1000
run_one wet-all  0.15 wet-all_t0.15 1000
run_one alt-car  0.15 alt-car_t0.15 1000

# Full 1000-step alt-track may trigger TORCS CommTimeout near step 556.
# Keep the full attempt log, then run the pre-timeout stopping point so the
# trace is auditable rather than lost on socket timeout.
run_one alt-track 0.15 alt-track_t0.15_full1000 1000
run_one alt-track 0.15 alt-track_t0.15_pretimeout556 556

PYTHONUNBUFFERED=1 "$PY" - "$OUT" <<'PY'
import csv
import json
import os
import sys

out = sys.argv[1]
labels = [
    "baseline_t0.10",
    "baseline_t0.15",
    "baseline_t0.20",
    "baseline_t0.25",
    "wet-all_t0.15",
    "alt-car_t0.15",
    "alt-track_t0.15_full1000",
    "alt-track_t0.15_pretimeout556",
]
rows = []
for label in labels:
    trace_path = os.path.join(out, f"{label}_steptrace.json")
    h20_path = os.path.join(out, f"{label}_H20.json")
    if not (os.path.exists(trace_path) and os.path.exists(h20_path)):
        rows.append({
            "condition": label,
            "steps": "NA",
            "done_reason": "no_complete_trace",
            "H20": "NA",
            "pass": "NA",
        })
        continue
    with open(trace_path) as f:
        trace = json.load(f)
    with open(h20_path) as f:
        h20 = json.load(f)
    run = trace["runs"][0]
    rows.append({
        "condition": label,
        "steps": run["steps_count"],
        "done_reason": run["done_reason"],
        "H20": f"{h20['decode_precision'] * 100:.2f}%",
        "pass": str(h20["metric2_pass"]).lower(),
    })

with open(os.path.join(out, "summary_H20.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["condition", "steps", "done_reason", "H20", "pass"])
    writer.writeheader()
    writer.writerows(rows)

print(open(os.path.join(out, "summary_H20.csv")).read())
PY

echo "Saved: $OUT/summary_H20.csv"

