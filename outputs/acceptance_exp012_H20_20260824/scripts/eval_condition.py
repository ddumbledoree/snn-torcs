#!/usr/bin/env python3
"""Eval driver: run eval_steptrace under a TORCS condition preset
(track / car / surface-friction), zero-stock-write via the cwd symlink-farm.

Usage:
  python scripts/eval_condition.py --condition wet \
      --model-type snn_baseline --checkpoint model/exp013_best.pth \
      --episodes 1 --max-steps 200 --run-mode fast_train --report data/wet.json

Every arg except --condition is passed through to eval_steptrace. The 'baseline'
preset reproduces stock exactly, so it doubles as the regression anchor:
    diff(eval_condition --condition baseline) ≈ eval_steptrace (stock).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from snn_torcs import condition as cond_mod


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--condition', required=True,
                   help='preset name from conditions/conditions.yaml')
    args, remaining = p.parse_known_args()

    condition = cond_mod.load_condition(args.condition)
    before = cond_mod.snapshot_stock()
    condition.apply()
    cond_mod.install_signal_handlers(condition, before)
    print(f'[condition] applied "{args.condition}": '
          f'{condition.spec.get("track")}/{condition.spec.get("category")} '
          f'car={condition.spec.get("car")} '
          f'friction={[e["value"] for e in condition.spec.get("friction", [])]}')

    sys.argv = ['eval_steptrace'] + remaining
    try:
        import eval_steptrace  # scripts/eval_steptrace.py (on sys.path)
        eval_steptrace.main()
    finally:
        try:
            condition.restore()
        finally:
            # Pollution gate: stock must be byte-identical (semantic sha256) to
            # the pre-run snapshot. Raises if anything drifted.
            cond_mod.assert_stock_unchanged(before)
            print('[condition] pollution gate PASSED — stock unchanged')


if __name__ == '__main__':
    main()
