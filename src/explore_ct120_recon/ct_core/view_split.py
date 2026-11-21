"""Test views: projections that NO step of a reconstruction ever sees.

Every driver already has ONE evaluation projection (``--holdout-index`` /
``--withhold-eval``). That view does a job: it is the validation signal the
learned backends stop on and the LR reducer listens to, and TIGRE's crossval
picks its best iterate with it. Once a view has chosen the iterate it is no
longer an unbiased test of it. A comparison across reconstruction methods
needs a second set — views reconstructed from by nobody, used for nothing,
scored once at the end — and this module is that set.

THE SPLIT HAPPENS ON THE DATA, NOT IN THE SAMPLER. The drivers replace the
``ScanContext``'s projections (and angles, acquisition groups, breathing
phases) with the TRAINING subset before any backend is built
(``subset_context``). A backend then cannot use a test view for anything — the
ray sampler, the noise-model fit, air normalisation's column statistics, ring
correction, an FDK warm start, a TIGRE crossval sweep — because it never had
it. Excluding the views in the sampler instead would leave every one of those
data-dependent steps to be audited separately, and the next one added would
leak silently.

Indices are always in the ORIGINAL numbering of the loaded projection list
(what ``--holdout-index`` means and what a sidecar records). ``ViewSplit``
maps between that numbering and the subset's.

Test views and the holdout never coincide: the holdout keeps its role, and a
test view that also chose the iterate would not be a test view.

With several acquisition groups loaded (a gated scan with both breath-holds),
a test ANGLE is withheld in every group: the same ray path measured in the
other breath-hold differs only by the breathing state, so leaving it in
training would let the model see most of what the test view is supposed to
hold back. Explicit ``--test-indices`` are taken literally; the run warns
when a same-angle counterpart stays in training.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewSplit:
    """Which loaded views train and which are only ever scored.

    ``train`` and ``test`` are sorted ORIGINAL indices; ``holdout`` is the
    evaluation view in the original numbering (or None when the driver has no
    evaluation view), ``mode`` how the test set was chosen.
    """
    n_views: int
    train: np.ndarray
    test: np.ndarray
    holdout: int | None
    mode: str

    def to_train(self, index: int) -> int:
        """Original index -> position in the training subset."""
        pos = int(np.searchsorted(self.train, int(index)))
        if pos >= self.train.size or int(self.train[pos]) != int(index):
            raise ValueError(f"view {index} is not a training view")
        return pos

    def record(self) -> dict:
        """The sidecar / run-config block: enough to score exactly these views
        later (``run_projection_report --test-views-from``) and to check that
        a scored view was never trained on."""
        return {
            'n_views': int(self.n_views),
            'test_indices': [int(i) for i in self.test],
            'train_indices': [int(i) for i in self.train],
            'holdout_index': None if self.holdout is None else int(self.holdout),
            'test_mode': self.mode,
        }


def evenly_spaced_test_indices(n_views: int, count: int, *,
                               reserved=(), view_groups=None) -> np.ndarray:
    """``count`` test views spread evenly over the arc, deterministically.

    The centres of ``count`` equal segments of the (first) group's views, so
    the set is symmetric about the middle of the scan and NEVER includes the
    first or last view: a short scan's terminal views carry the Parker ramps
    and bound the angular span the FDK normalises by, and withholding one
    would change the scan, not just thin it.

    A position that lands on a ``reserved`` view (the holdout) moves to the
    nearest free interior view, alternating +1, -1, +2, ... so the shift is
    the smallest available and its direction deterministic.
    """
    n, k = int(n_views), int(count)
    groups = _groups(view_groups, n)
    # Positions are chosen within ONE group's angle list and then applied to
    # every group, so a test angle is withheld from each breath-hold alike.
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    m = int(min(len(s) for s in members))
    if k < 1:
        raise ValueError(f"--test-count must be >= 1, got {k}")
    if k > max(0, m - 2) // 2:
        raise ValueError(
            f"--test-count {k} is too many for {m} views per acquisition "
            f"group: at most {max(0, m - 2) // 2} keeps a training view "
            f"between every two test views and leaves both ends in training")
    reserved_pos = set()
    for r in reserved:
        if r is None:
            continue
        for s in members:
            hit = np.flatnonzero(s == int(r))
            if hit.size:
                reserved_pos.add(int(hit[0]))
    taken: set = set()
    out_pos = []
    for i in range(k):
        p = int(np.floor((i + 0.5) * m / k))
        for step in _offsets(m):
            q = p + step
            if 0 < q < m - 1 and q not in reserved_pos and q not in taken:
                break
        else:                                       # pragma: no cover - guarded above
            raise ValueError("no free interior view for a test position")
        taken.add(q)
        out_pos.append(q)
    idx = sorted(int(s[q]) for s in members for q in out_pos)
    return np.asarray(idx, dtype=np.int64)


def _offsets(m: int):
    yield 0
    for d in range(1, m):
        yield d
        yield -d


def _groups(view_groups, n: int) -> np.ndarray:
    if view_groups is None:
        return np.zeros(n, dtype=np.int64)
    g = np.asarray(view_groups, dtype=np.int64)
    if g.shape != (n,):
        raise ValueError(f"view_groups has shape {g.shape} for {n} views")
    return g


def resolve_view_split(n_views: int, *, test_indices=None, test_count=None,
                       holdout: int | None = None,
                       view_groups=None) -> ViewSplit | None:
    """The split, or None when no test views were asked for.

    None is the whole of the default path: a driver that gets None changes
    nothing, so runs without ``--test-indices``/``--test-count`` are the runs
    they always were.
    """
    if test_indices and test_count:
        raise ValueError("give --test-indices or --test-count, not both")
    n = int(n_views)
    if test_indices:
        idx = np.asarray(sorted({int(i) for i in test_indices}), dtype=np.int64)
        bad = idx[(idx < 0) | (idx >= n)]
        if bad.size:
            raise ValueError(f"--test-indices {bad.tolist()} out of range "
                             f"[0, {n})")
        if holdout is not None and int(holdout) in set(idx.tolist()):
            raise ValueError(
                f"--test-indices contains the evaluation/holdout view "
                f"{holdout}. It chooses the iterate, so it cannot also be a "
                f"test view; pick another index or move --holdout-index.")
        if idx.size >= n - 1:
            raise ValueError("--test-indices would leave fewer than two "
                             "training views")
        mode = 'indices'
    elif test_count:
        idx = evenly_spaced_test_indices(n, int(test_count),
                                         reserved=(holdout,),
                                         view_groups=view_groups)
        mode = f'count={int(test_count)}'
    else:
        return None
    train = np.setdiff1d(np.arange(n, dtype=np.int64), idx)
    return ViewSplit(n_views=n, train=train, test=idx,
                     holdout=None if holdout is None else int(holdout),
                     mode=mode)


def same_angle_counterparts(split: ViewSplit, view_groups) -> list:
    """Training views that repeat a test view's angle in another acquisition
    group — ``(test, counterpart)`` pairs, empty for a single group."""
    if view_groups is None:
        return []
    g = _groups(view_groups, split.n_views)
    groups = np.unique(g)
    if groups.size < 2:
        return []
    members = {int(k): np.flatnonzero(g == k) for k in groups}
    train = set(split.train.tolist())
    out = []
    for t in split.test.tolist():
        gk = int(g[t])
        pos = int(np.flatnonzero(members[gk] == t)[0])
        for k, s in members.items():
            if k != gk and pos < s.size and int(s[pos]) in train:
                out.append((int(t), int(s[pos])))
    return out


def subset_context(ctx, split: ViewSplit):
    """A copy of ``ctx`` holding only the TRAINING views.

    Every per-view array is cut the same way — projections, angles,
    acquisition groups and breathing phases — so nothing downstream can pair a
    projection with the wrong angle. The original context is untouched; the
    drivers keep it for the test views and for diagnostics that address views
    by their original index.
    """
    keep = split.train
    changes = {'projections': np.ascontiguousarray(ctx.projections[keep]),
               'angles': np.asarray(ctx.angles)[keep]}
    if getattr(ctx, 'view_groups', None) is not None:
        changes['view_groups'] = np.asarray(ctx.view_groups)[keep]
    if getattr(ctx, 'view_phase', None) is not None:
        changes['view_phase'] = np.asarray(ctx.view_phase)[keep]
    return dataclasses.replace(ctx, **changes)


def add_view_split_args(parser) -> None:
    """``--test-indices`` / ``--test-count``, shared by every driver."""
    parser.add_argument(
        '--test-indices', type=int, nargs='+', default=None, metavar='N',
        help='Projection indices (original numbering) to withhold as TEST '
             'views: removed from the data before the backend is built, so '
             'they reach no training batch, stopping rule, LR schedule, '
             'noise or air-normalisation fit, FDK warm start or crossval '
             'sweep, and are scored once at the end. Distinct from '
             '--holdout-index, which stays the validation view. Default: '
             'none.')
    parser.add_argument(
        '--test-count', type=int, default=None, metavar='N',
        help='Withhold N test views evenly spaced over the arc '
             '(deterministic; never the first or last view, never the '
             'holdout; with several acquisition groups loaded, the same '
             'angles are withheld in every group). Alternative to '
             '--test-indices. Default: none.')


def view_split_from_args(args, n_views: int, *, holdout=None,
                         view_groups=None, verbose: bool = True):
    """``resolve_view_split`` from the parsed flags, with the one-line account
    every driver prints. None when neither flag was given."""
    split = resolve_view_split(
        n_views, test_indices=getattr(args, 'test_indices', None),
        test_count=getattr(args, 'test_count', None), holdout=holdout,
        view_groups=view_groups)
    if split is not None and verbose:
        shown = split.test.tolist()
        print(f"\n  Test views ({split.mode}): {len(shown)} of {n_views} "
              f"withheld from the reconstruction and scored only at the end — "
              f"{shown if len(shown) <= 24 else shown[:24] + ['...']}"
              + (f"; holdout/eval view {split.holdout} stays the validation "
                 f"view" if split.holdout is not None else ""))
        pairs = same_angle_counterparts(split, view_groups)
        if pairs and split.mode == 'indices':
            print(f"  WARNING: {len(pairs)} test view(s) have a same-angle "
                  f"counterpart from another acquisition group in TRAINING "
                  f"(e.g. {pairs[:3]}). --test-count withholds those too.")
    return split
