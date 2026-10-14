"""
Backward-compat re-exports for scripts that import from snn_torcs.trainer.

This repo ships the eval/inference-only handoff package. Checkpoint IO was
extracted into snn_torcs/io.py when the handoff was cut (see io.py docstring:
"trainer.py re-exports these for backward compatibility"). No training code
(optimizer / replay buffer / train loop) is included — deployment only runs
eval on the shipped checkpoints.

scripts/benchmark_policy_latency.py still does `from snn_torcs.trainer import
load_state_dict`, so we keep that import path alive with a thin re-export.
"""
from snn_torcs.io import load_state_dict, load_checkpoint_metadata  # noqa: F401

__all__ = ["load_state_dict", "load_checkpoint_metadata"]
