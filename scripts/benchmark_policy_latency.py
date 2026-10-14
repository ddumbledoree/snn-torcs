#!/usr/bin/env python
"""Benchmark single-forward latency of baseline vs ours policy.

Measures ONLY model forward (obs → Q-values → action).
No env.step, no UDP, no rendering, no checkpoint, no reference policy.
CUDA timing uses synchronize. Reports mean/std/p50/p95.
"""
import argparse
import os
import sys
import time
import json

import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from snn_torcs.models import FCANN, FCSNN
from snn_torcs.trainer import load_state_dict
from snn_torcs.utils import set_seed, get_device, save_report


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark policy latency")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--baseline-ckpt", default=None)
    p.add_argument("--ours-ckpt", default=None)
    p.add_argument("--baseline-exp-dir", default=None,
                   help="Baseline experiment dir (uses best.pth)")
    p.add_argument("--ours-exp-dir", default=None,
                   help="Ours experiment dir (uses best.pth)")
    p.add_argument("--report-dir", default=None)
    p.add_argument("--warmup-iters", type=int, default=None)
    p.add_argument("--test-iters", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def benchmark_model(model, device, warmup, iters):
    """Return list of latencies in ms."""
    model.eval()
    dummy = torch.randn(1, 68, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    return latencies


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = get_device(args.device)
    report_dir = args.report_dir or cfg["paths"]["report_dir"]
    warmup = args.warmup_iters or cfg["benchmark"]["warmup_iters"]
    iters = args.test_iters or cfg["benchmark"]["test_iters"]
    m = cfg["model"]
    s = cfg["snn"]
    ckpt_dir = cfg["paths"]["ckpt_dir"]

    print(f"[Benchmark] device={device}, warmup={warmup}, iters={iters}")

    results = {}

    # --- Baseline SNN T=8 ---
    baseline_ckpt = args.baseline_ckpt
    ours_ckpt = args.ours_ckpt
    # resolve from exp-dir if provided
    if args.baseline_exp_dir:
        baseline_ckpt = os.path.join(args.baseline_exp_dir, "best.pth")
    if args.ours_exp_dir:
        ours_ckpt = os.path.join(args.ours_exp_dir, "best.pth")
    # fallback defaults
    if not baseline_ckpt:
        baseline_ckpt = os.path.join(ckpt_dir, "baseline_snn_best.pth")
    if not ours_ckpt:
        ours_ckpt = os.path.join(ckpt_dir, "ours_snn_best.pth")
    if os.path.isfile(baseline_ckpt):
        baseline = FCSNN(m["input_dim"], m["hidden_dim"], m["output_dim"],
                         time_window=s["time_window_baseline"],
                         thresh=s["thresh"], lens=s["lens"], decay=s["decay"])
        baseline.load_state_dict(load_state_dict(baseline_ckpt, map_location=device))
        baseline.to(device)
        lat = benchmark_model(baseline, device, warmup, iters)
        results["baseline"] = {
            "latencies_ms": lat,
            "mean": float(np.mean(lat)),
            "std": float(np.std(lat)),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
        }
        print(f"  Baseline (T={s['time_window_baseline']}): "
              f"mean={results['baseline']['mean']:.3f}ms, "
              f"p50={results['baseline']['p50']:.3f}ms")
    else:
        # benchmark untrained model
        print(f"  No baseline checkpoint at {baseline_ckpt}, using random init")
        baseline = FCSNN(m["input_dim"], m["hidden_dim"], m["output_dim"],
                         time_window=s["time_window_baseline"],
                         thresh=s["thresh"], lens=s["lens"], decay=s["decay"])
        baseline.to(device)
        lat = benchmark_model(baseline, device, warmup, iters)
        results["baseline"] = {
            "latencies_ms": lat,
            "mean": float(np.mean(lat)),
            "std": float(np.std(lat)),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
        }
        print(f"  Baseline (T={s['time_window_baseline']} random): "
              f"mean={results['baseline']['mean']:.3f}ms")

    # --- Ours SNN T-4 ---
    # ours_ckpt already resolved above from --ours-exp-dir or --ours-ckpt or default
    if os.path.isfile(ours_ckpt):
        ours = FCSNN(m["input_dim"], m["hidden_dim"], m["output_dim"],
                     time_window=s["time_window_ours"],
                     thresh=s["thresh"], lens=s["lens"], decay=s["decay"])
        ours.load_state_dict(load_state_dict(ours_ckpt, map_location=device))
        ours.to(device)
    else:
        print(f"  No ours checkpoint at {ours_ckpt}, using random init")
        ours = FCSNN(m["input_dim"], m["hidden_dim"], m["output_dim"],
                     time_window=s["time_window_ours"],
                     thresh=s["thresh"], lens=s["lens"], decay=s["decay"])
        ours.to(device)

    lat = benchmark_model(ours, device, warmup, iters)
    results["ours"] = {
        "latencies_ms": lat,
        "mean": float(np.mean(lat)),
        "std": float(np.std(lat)),
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
    }
    print(f"  Ours     (T={s['time_window_ours']}): "
          f"mean={results['ours']['mean']:.3f}ms, "
          f"p50={results['ours']['p50']:.3f}ms")

    # reduction
    b_mean = results["baseline"]["mean"]
    o_mean = results["ours"]["mean"]
    reduction = (b_mean - o_mean) / b_mean if b_mean > 0 else 0.0
    results["latency_reduction"] = reduction
    print(f"  Latency reduction: {reduction:.1%}")

    report = {
        "baseline_policy_latency_ms": results["baseline"]["mean"],
        "baseline_p50_ms": results["baseline"]["p50"],
        "baseline_p95_ms": results["baseline"]["p95"],
        "ours_policy_latency_ms": results["ours"]["mean"],
        "ours_p50_ms": results["ours"]["p50"],
        "ours_p95_ms": results["ours"]["p95"],
        "latency_reduction": reduction,
    }

    report_path = os.path.join(report_dir, "latency_benchmark.json")
    save_report(report, report_path)
    print(f"  Report saved: {report_path}")

    return report


if __name__ == "__main__":
    main()
