#!/usr/bin/env python
"""Step-level (per-frame) evaluation trace — for replay mode + live demo.

Loads a checkpoint, runs TORCS closed-loop (fast_train = headless by default),
and at every step records exactly the fields the display software needs:

    step / action / steer / Q(7) / reward / speedX / trackPos / angle / distRaced / infer_ms

It does two things at once:
  1. writes a step-level JSON  — input for replay mode (per-frame trajectory)
  2. prints one line per step  — the run.sh acceptance demo (per-step telemetry)

Self-contained: ships its own build_model / checkpoint_identity so the handoff
package need not include eval_policy.py.
"""
import argparse
import hashlib
import os
import sys
import time
import json
from datetime import datetime

import yaml
import numpy as np
import torch

# Support both layouts: dev (script in scripts/, snn_torcs at repo root) and
# handoff (script next to snn_torcs/ in app/).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from snn_torcs.models import FCANN, FCSNN
from snn_torcs.io import load_state_dict, load_checkpoint_metadata
from snn_torcs.env_wrapper import TorcsDiscreteWrapper, STEER_VALUES, STEER_VALUES_BY_DIM
from snn_torcs.utils import set_seed, get_device
from snakeoil3_gym import CommTimeout


# --------------------------------------------------------------------------- #
# self-contained helpers (mirror scripts/eval_policy.py)
# --------------------------------------------------------------------------- #
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_identity(path):
    """Stable identity fields so reports stay auditable after overwrite."""
    abs_path = os.path.abspath(path)
    st = os.stat(abs_path)
    return {
        "path": abs_path,
        "basename": os.path.basename(abs_path),
        "parent": os.path.basename(os.path.dirname(abs_path)),
        "size_bytes": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "sha256": _sha256_file(abs_path),
        "metadata": load_checkpoint_metadata(abs_path, map_location="cpu"),
    }


def _detect_output_dim(checkpoint_path, model_type):
    sd = load_state_dict(checkpoint_path)
    key = "net.4.weight" if model_type == "ann" else "fc3.weight"
    if key in sd:
        return sd[key].shape[0]
    return None


def build_model(model_type, cfg, output_dim=None):
    m, s = cfg["model"], cfg["snn"]
    od = output_dim or m["output_dim"]
    if model_type == "ann":
        return FCANN(m["input_dim"], m["hidden_dim"], od)
    if model_type == "snn_baseline":
        tw = s["time_window_baseline"]
    else:  # snn_ours
        tw = s["time_window_ours"]
    return FCSNN(m["input_dim"], m["hidden_dim"], od,
                 time_window=tw, thresh=s["thresh"], lens=s["lens"], decay=s["decay"])


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Step-level eval trace for replay + demo")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model-type", required=True, choices=["ann", "snn_baseline", "snn_ours"])
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--exp-dir", default=None, help="Experiment dir (uses best.pth inside)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run-mode", choices=["fast_train", "demo_eval"], default="fast_train",
                   help="fast_train=headless(Xvfb); demo_eval=GUI(needs real DISPLAY)")
    p.add_argument("--report", default=None, help="Output step-level JSON path")
    p.add_argument("--track", default="g-track-1", help="Track name (for replay geometry)")
    p.add_argument("--device", default=None)
    p.add_argument("--center-coef", type=float, default=0.1)
    p.add_argument("--fixed-accel", type=float, default=None,
                   help="Override config env.fixed_accel (constant throttle opening 0-1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true", help="Suppress per-step stdout")
    args = p.parse_args()
    if args.exp_dir:
        args.checkpoint = os.path.join(args.exp_dir, "best.pth")
    if not args.checkpoint:
        p.error("Either --checkpoint or --exp-dir is required")
    return args


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = get_device(args.device)
    max_steps = args.max_steps or cfg["eval"]["max_steps"]

    output_dim = _detect_output_dim(args.checkpoint, args.model_type) or cfg["model"]["output_dim"]
    steer_values = STEER_VALUES_BY_DIM.get(output_dim, STEER_VALUES)

    fixed_accel = args.fixed_accel if args.fixed_accel is not None else cfg["env"]["fixed_accel"]

    print(f"[steptrace] type={args.model_type} ckpt={args.checkpoint}")
    print(f"            device={device} output_dim={output_dim} "
          f"steer={steer_values.tolist()} run_mode={args.run_mode} fixed_accel={fixed_accel}")

    env = TorcsDiscreteWrapper(
        vision=cfg["env"]["vision"], throttle=cfg["env"]["throttle"],
        fixed_accel=fixed_accel, max_episode_steps=max_steps,
        run_mode=args.run_mode, steer_values=steer_values,
        center_coef=args.center_coef,
    )

    model = build_model(args.model_type, cfg, output_dim=output_dim)
    model.load_state_dict(load_state_dict(args.checkpoint, map_location=device))
    model.to(device).eval()

    runs = []
    try:
        for ep in range(args.episodes):
            relaunch = (ep > 0 and ep % cfg["env"]["relaunch_interval"] == 0)
            obs = env.reset(relaunch=relaunch)
            state = torch.FloatTensor(obs).unsqueeze(0).to(device)
            steps, total_reward, st, done, info = [], 0.0, 0, False, {}

            while st < max_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    q = model(state)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                infer_ms = (time.perf_counter() - t0) * 1000.0

                q_cpu = q.detach().cpu().numpy()[0]
                action = int(q_cpu.argmax())
                steer = float(steer_values[action])

                next_obs, reward, done, info = env.step(action)
                rec = {
                    "step": st, "action": action, "steer": steer,
                    "q": [round(float(x), 4) for x in q_cpu],
                    "q_max": round(float(q_cpu.max()), 4),
                    "q_gap": round(float(q_cpu.max() - q_cpu.min()), 4),
                    "reward": round(float(reward), 4),
                    "speedX": round(float(info.get("speedX", 0.0)), 4),
                    "trackPos": round(float(info.get("trackPos", 0.0)), 4),
                    "angle": round(float(info.get("angle", 0.0)), 4),
                    "distRaced": round(float(info.get("distRaced", 0.0)), 2),
                    "infer_ms": round(float(infer_ms), 4),
                    "done": bool(done),
                    "done_reason": info.get("done_reason") if done else None,
                }
                steps.append(rec)
                total_reward += reward

                if not args.quiet:
                    print(f"ep{ep} s={st:4d} act={action} st={steer:+.2f} "
                          f"Q={rec['q_max']:6.2f} gap={rec['q_gap']:.2f} "
                          f"v={rec['speedX']:5.1f} tp={rec['trackPos']:+.3f} "
                          f"ang={rec['angle']:+.3f} dist={rec['distRaced']:7.1f} "
                          f"r={reward:+6.3f} {infer_ms:.3f}ms", flush=True)

                state = torch.FloatTensor(next_obs).unsqueeze(0).to(device)
                st += 1
                if done:
                    break

            runs.append({
                "episode": ep, "steps_count": st,
                "done_reason": info.get("done_reason") or "max_steps",
                "reward": round(float(total_reward), 2),
                "final_speedX": steps[-1]["speedX"] if steps else None,
                "final_trackPos": steps[-1]["trackPos"] if steps else None,
                "final_distRaced": steps[-1]["distRaced"] if steps else None,
                "steps": steps,
            })
            print(f"--- episode {ep} done: steps={st} reward={total_reward:.1f} "
                  f"reason={runs[-1]['done_reason']}", flush=True)

        report = {
            "model_type": args.model_type,
            "checkpoint": checkpoint_identity(args.checkpoint),
            "config": {
                "path": os.path.abspath(args.config),
                "seed": args.seed, "device": str(device),
                "run_mode": args.run_mode, "fixed_accel": fixed_accel,
                "max_steps": max_steps, "output_dim": output_dim,
                "steer_values": [float(x) for x in steer_values.tolist()],
            },
            "track": args.track,
            "episodes": args.episodes,
            "runs": runs,
        }
        report_path = args.report or os.path.join(
            cfg["paths"]["report_dir"],
            f"steptrace_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        n_steps = sum(r["steps_count"] for r in runs)
        print(f"\n[steptrace] wrote {report_path}  ({n_steps} steps, "
              f"{os.path.getsize(report_path)/1024:.0f} KB)")

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    except CommTimeout as e:
        print(f"\n[CommTimeout] {e}")
    finally:
        try:
            env.end()
        except Exception:
            pass
        print("TORCS terminated.")


if __name__ == "__main__":
    main()
