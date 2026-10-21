"""One-line progress reports the GUI turns into a single bar and a time estimate.

A driver first declares its phases and their rough share of the run time,
``plan(preprocess=1, filter=1, backproject=10, save=2)``, which prints
``progress-plan: preprocess=1 filter=1 backproject=10 save=2``. It then calls
``report(phase, done, total)`` from the loops it already has (projection
chunks, iterations, slabs), printing ``progress: <phase> <done>/<total>``.
The GUI's log scanner maps each phase onto its slice of one cumulative
fraction, so the bar never resets between phases, and measures the pace of
that fraction over the last few minutes for the estimate. The weights only
have to be in the right ballpark: the measured pace corrects the estimate as
the run goes on. Prints are throttled to one per second per phase, except
the first and the last.
"""
from __future__ import annotations

import sys
import time

_last: dict[str, float] = {}
MIN_INTERVAL = 1.0


def plan(**phases: float) -> None:
    """Declare the phases of this run, in order, with their relative weights."""
    print('progress-plan: ' + ' '.join(f'{k}={float(v):g}' for k, v in phases.items()), flush=True)
    sys.stdout.flush()


def report(phase: str, done: int, total: int) -> None:
    now = time.monotonic()
    if 0 < done < total and now - _last.get(phase, 0.0) < MIN_INTERVAL:
        return
    _last[phase] = now
    print(f'progress: {phase} {int(done)}/{int(total)}', flush=True)
    sys.stdout.flush()
