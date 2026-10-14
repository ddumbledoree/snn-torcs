"""
Condition-preset engine (track / car / surface-friction switching) — PLACEHOLDER.

The original implementation lives in the private training repo
(/home/desc/gym_torcs/snn_torcs/condition.py) and was NOT part of the two
handoff packages. It is the engine behind `scripts/eval_condition.py` and the
robustness matrix `scripts/run_7_condition_h20_matrix.sh`, switching TORCS
conditions defined in conditions/conditions.yaml:

    baseline / wet / icy / wet-all / alt-car / alt-track

What it did (per conditions.yaml header + eval_condition.py call sites):
  - practice.xml  : <Tracks> name+category scoped edit        (track select)
  - scr_server.xml: <attstr name="car name"> global edit      (car select)
  - track XML     : <section name=...> friction scoped edit   (grip select)
  - zero-stock-write via a cwd symlink-farm + restore + pollution gate
    (assert_stock_unchanged: stock files byte-identical after the run)

To restore matrix reruns, either
  1. copy the original condition.py from the training machine over this file, or
  2. reimplement it following docs/condition_module_spec.md.

Everything else — metric1 latency benchmark, metric2 standard H20, and the
run.sh acceptance demo — does NOT need this module.
"""


def _unavailable(*_args, **_kwargs):
    raise NotImplementedError(
        "snn_torcs/condition.py is a placeholder: the condition-switching "
        "engine was not included in the handoff packages. Copy the original "
        "file from the training repo (/home/desc/gym_torcs/snn_torcs/"
        "condition.py) onto this path, or reimplement it per "
        "docs/condition_module_spec.md. Note: metric1 / metric2-standard / "
        "run.sh do not require this module."
    )


def load_condition(name):
    """Return a condition object with .spec / .apply() / .restore()."""
    _unavailable()


def snapshot_stock():
    """Snapshot stock TORCS files (semantic sha256) before a condition run."""
    _unavailable()


def install_signal_handlers(condition, before):
    """Restore stock + condition on SIGINT/SIGTERM."""
    _unavailable()


def assert_stock_unchanged(before):
    """Pollution gate: raise if any stock file drifted."""
    _unavailable()
