#!/usr/bin/env python
"""Summarize finalG metric 2 from eval_steptrace JSON.

Metric: decoded-control valid sample rate.

Input is produced by scripts/eval_steptrace.py. Each step record is the result
of applying the SNN-decoded action to TORCS. A decision sample is valid when the
following H-step window keeps the car on track, moving forward, not backward,
and does not show persistent growth in lateral/heading error.
"""
import argparse
import json
import math
import os
import statistics


FAIL_REASONS = {"off_track", "backward", "slow_progress"}


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def _percentile(values, pct):
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (len(xs) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - rank) + xs[hi] * (rank - lo))


def _window_has_terminal_failure(run, start, end):
    reason = run.get("done_reason") or "max_steps"
    if reason not in FAIL_REASONS:
        return None
    final_idx = int(run.get("steps_count", len(run.get("steps", [])))) - 1
    if start <= final_idx < end:
        return reason
    return None


def classify_step(run, idx, *, horizon=10, track_limit=1.0,
                  angle_weight=1.0, worsening_tol=1e-6):
    steps = run.get("steps", [])
    end = min(len(steps), idx + horizon)
    window = steps[idx:end]
    if not window:
        return False, "empty_window"

    terminal_reason = _window_has_terminal_failure(run, idx, end)
    if terminal_reason:
        return False, terminal_reason

    for rec in window:
        if rec.get("done_reason") in FAIL_REASONS:
            return False, rec.get("done_reason")
        track_pos = float(rec.get("trackPos", 0.0))
        angle = float(rec.get("angle", 0.0))
        if abs(track_pos) >= track_limit:
            return False, "off_track"
        if math.cos(angle) < 0.0:
            return False, "backward"

    progresses = []
    for prev, cur in zip(window, window[1:]):
        progresses.append(float(cur.get("distRaced", 0.0)) -
                          float(prev.get("distRaced", 0.0)))
    if progresses and min(progresses) <= 0.0:
        return False, "non_positive_progress"
    if not progresses and float(window[0].get("speedX", 0.0)) <= 0.0:
        return False, "non_positive_speed"

    errors = [
        abs(float(rec.get("trackPos", 0.0))) +
        angle_weight * abs(float(rec.get("angle", 0.0)))
        for rec in window
    ]
    if len(errors) >= 3:
        deltas = [b - a for a, b in zip(errors, errors[1:])]
        if all(d > worsening_tol for d in deltas):
            return False, "persistent_error_worsening"

    return True, "valid"


def region_for_step(rec):
    track_pos = abs(float(rec.get("trackPos", 0.0)))
    angle = abs(float(rec.get("angle", 0.0)))
    if track_pos < 0.3:
        lateral = "center_region"
    else:
        lateral = "edge_region"
    if track_pos >= 0.6 or angle >= 0.25:
        critical = "critical_state"
    else:
        critical = "nominal_state"
    return lateral, critical


def summarize(trace, *, horizon=10, track_limit=1.0, angle_weight=1.0):
    total_steps = 0
    valid_steps = 0
    invalid_reasons = {}
    region_counts = {}
    progress_values = []
    reward_values = []
    infer_values = []
    truncated_windows = 0

    for run in trace.get("runs", []):
        steps = run.get("steps", [])
        for idx, rec in enumerate(steps):
            total_steps += 1
            if idx + horizon > len(steps):
                truncated_windows += 1
            ok, reason = classify_step(
                run, idx, horizon=horizon, track_limit=track_limit,
                angle_weight=angle_weight)
            if ok:
                valid_steps += 1
            else:
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

            for region in region_for_step(rec):
                bucket = region_counts.setdefault(region, {"total": 0, "valid": 0})
                bucket["total"] += 1
                if ok:
                    bucket["valid"] += 1

            reward_values.append(float(rec.get("reward", 0.0)))
            if rec.get("infer_ms") is not None:
                infer_values.append(float(rec["infer_ms"]))

        for prev, cur in zip(steps, steps[1:]):
            progress_values.append(float(cur.get("distRaced", 0.0)) -
                                   float(prev.get("distRaced", 0.0)))

    valid_rate = valid_steps / total_steps if total_steps else 0.0
    valid_rate_by_region = {
        name: {
            "valid_steps": data["valid"],
            "total_steps": data["total"],
            "valid_rate": data["valid"] / data["total"] if data["total"] else 0.0,
        }
        for name, data in sorted(region_counts.items())
    }
    return {
        "metric": "finalG_decoded_control_validity",
        "horizon": horizon,
        "track_limit": track_limit,
        "angle_weight": angle_weight,
        "total_steps": total_steps,
        "valid_steps": valid_steps,
        "valid_rate": valid_rate,
        "decode_precision": valid_rate,
        "metric2_pass": valid_rate >= 0.90,
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "valid_rate_by_region": valid_rate_by_region,
        "truncated_windows": truncated_windows,
        "mean_progress": _mean(progress_values),
        "min_progress": min(progress_values) if progress_values else None,
        "mean_reward": _mean(reward_values),
        "mean_infer_ms": _mean(infer_values),
        "p95_infer_ms": _percentile(infer_values, 0.95),
        "source_trace": {
            "model_type": trace.get("model_type"),
            "checkpoint": trace.get("checkpoint", {}),
            "episodes": trace.get("episodes"),
            "track": trace.get("track"),
            "config": trace.get("config", {}),
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="Summarize finalG decoded-control validity")
    p.add_argument("--trace", required=True, help="Input JSON from eval_steptrace.py")
    p.add_argument("--report", default=None,
                   help="Output JSON. Default: outputs/reports/decode_control_validity.json")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--track-limit", type=float, default=1.0)
    p.add_argument("--angle-weight", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.trace) as f:
        trace = json.load(f)

    report = summarize(
        trace, horizon=args.horizon, track_limit=args.track_limit,
        angle_weight=args.angle_weight)
    report["trace_path"] = os.path.abspath(args.trace)

    out = args.report or os.path.join(
        os.path.dirname(args.trace) or ".", "decode_control_validity.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("=== finalG metric2: decoded-control validity ===")
    print(f"  valid_steps / total_steps = {report['valid_steps']} / {report['total_steps']}")
    print(f"  decode_precision          = {report['decode_precision']:.2%}")
    print(f"  metric2_pass              = {report['metric2_pass']}")
    print(f"  invalid_reasons           = {report['invalid_reasons']}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
