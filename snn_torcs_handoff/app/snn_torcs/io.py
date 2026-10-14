"""
Checkpoint IO — load model weights and metadata, independent of training logic.

Kept dependency-light (torch only) so the handoff package can ship this module
without the trainer / replay buffer / optimizer code. trainer.py re-exports
these for backward compatibility.
"""
import torch


def load_state_dict(path, map_location="cpu"):
    """Load a model state_dict from any checkpoint format.

    Handles both new (dict with 'policy_net') and legacy (plain state_dict)
    checkpoints.  Use this wherever you need model weights — not the full
    trainer state — e.g. eval scripts, ANN→SNN weight transfer, benchmarks.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and 'policy_net' in ckpt:
        return ckpt['policy_net']
    return ckpt


def load_checkpoint_metadata(path, map_location="cpu"):
    """Return non-weight checkpoint metadata, or {} for legacy state_dict files."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or 'policy_net' not in ckpt:
        return {}
    return {
        k: ckpt.get(k)
        for k in ('episode', 'best_reward', 'stage', 'train_step', 'epsilon')
        if k in ckpt
    }
