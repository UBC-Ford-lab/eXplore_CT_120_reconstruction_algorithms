"""The detector's noise model for a log-attenuation sinogram, and where it
comes from.

A weighted data term needs the variance of every measured line integral
``p_i = -ln(T_i)``. This module owns that variance: the MODEL (which two
constants describe it), the ESTIMATOR (how a scan measures them from its own
projections, with no repeat acquisition), and the POLICY (what to do when a
scan cannot measure one of them — pinned values, a shared calibration file
keyed by the detector, or a loud package default).

It lives in ``ct_core`` because the noise belongs to the DETECTOR and the DATA,
not to any one representation: the ``wls`` data term of every learned backend
(``learning_based_iterative.losses``, built by ``LearnedReconstructor.
_build_wls_weights``) consumes it, and any other likelihood-weighted data term
should consume the same one rather than grow its own.

The model
---------
For a pixel at binning ``ds`` (the MEAN of ``ds^2`` raw pixels, which is what
``preprocessing.downsample_projections`` produces) with ``c`` counts above
dark, on the same (mean) scale,

    Var(p) = A_ds / c + (sigma_r / ds)^2 / c^2

``A_ds`` is the Poisson slope AT THAT BINNING — ADU per detected quantum,
Swank excess and the pooling's own correlation included; the electron gain
and the units cancel, which is why it can be measured with no gain
calibration. ``sigma_r`` is the read noise in counts PER RAW PIXEL. Read noise
is constant in the COUNTS domain, ``Var(c) = A c + sigma_r^2``, and division
by ``c^2`` on the way into the log domain is what puts it at ``1/c^2``.

This replaces ``A / c + B`` with a constant ``B``, which is what the model was
until 2026-09-09. The two are indistinguishable over a short range of counts —
and every scan measured until then had a short range (see "identifiability"
below) — but they are different physics: a constant term in ``Var(p)`` is a
fixed FRACTIONAL fluctuation, not read noise. MEASURED on Scan_1988 (phantom,
counts 721..3201 at ds3, a 4.4x lever, curvature-corrected as below), fitted
three ways:

    A/c + B           R2 0.99294   B = -2.3e-5 (NEGATIVE; clamped to 0 before)
    A/c + B'/c^2      R2 0.99631   B' = 43.6 at ds3 -> sigma_r = 19.8 counts
    A/c + B + B'/c^2  R2 0.99660   the extra parameter does not pay (AIC)

The constant-floor form also forces the slope up by 74 % on the same data,
because a constant cannot absorb a ``1/c^2`` misfit and the Poisson term
takes it up instead. The ``1/c^2`` form finds the same read noise at every
binning (next paragraph), which a wrong shape would not.

Pooling — the two terms pool DIFFERENTLY, and this is measured, not assumed.
Free fits of the same Scan_1988 views at ds1, ds2, ds3, ds4 return a read
noise of 20.2, 19.9, 21.5, 19.8 counts per raw pixel once the per-binning
value is multiplied back by ``ds``: the read noise is independent per pixel
and averages down by ``ds^2`` in variance, exactly. The Poisson term does NOT:
its per-raw-pixel equivalent climbs 0.43 -> 0.60 -> 0.74 -> 0.99 from ds1 to
ds4, i.e. a ``ds x ds`` mean keeps 1.4x (ds2), 1.7x (ds3), 2.3x (ds4) more
photon variance than independent pixels would, because the scintillator
spreads one quantum's light over neighbouring pixels. So: ``sigma_r`` is
stored per raw pixel and transfers across binnings by ``1/ds``; ``A_ds`` is
never transferred — it is refit on every run at the run's own binning, which
needs no lever and costs seconds. (The previous convention converted the
slope by ``ds^2`` too; that was wrong by the factors above. Uniform within a
run, so no run's weight SHAPE was affected — only the absolute chi-square.)

The estimator
-------------
Pixels that see the same thing in consecutive views differ only by noise, so
a difference between views measures the variance with the fixed pattern and
the flat-field reference cancelled exactly (same pixel, same reference) and
the source's gain drift removed (a difference). Air pixels satisfy that
trivially, but their counts come from flat-field vignetting alone — a 1.2x
(Scan_1988) to 1.5x (Scan_1510) range, over which ``A/c`` and ``1/c^2`` are
collinear. To fit the second term at all the estimator must reach the rays
THROUGH the object, and there consecutive views differ by real anatomy.

So it uses the SECOND difference over a view stride ``s``,

    d2 = p[k+s] - 2 p[k] + p[k-s],      E[d2^2] / 6 = sigma^2 + C s^2 + O(s^4)

which kills any locally linear trend and leaves the object's curvature — and
that curvature scales as ``s^2`` while independent noise is stride-blind.
Measuring at ``s = 1..4`` and solving each count bin for ``(sigma^2, C)`` gives
the noise with the anatomy removed and no noise model assumed anywhere.
Validated on Scan_1988: air pixels (nothing moves) give ``v(2)/v(1) = 1.002``;
object bins ran 1.1-2.4x before the correction; the ``s^2`` law held to 1-3 %
in 20-22 of 24 bins, and the ones it failed in (the phantom's edge, where the
projection's curvature is too strong for the quadratic term) are excluded by
the residual gate rather than fitted. Samples are binned PER RAY, not per
pixel: a mouse's pixel sees air in some views and body in others, and a
per-pixel mean smears those onto one intermediate count.

The per-bin statistic is a sigma-clipped MEAN of ``d2^2``, not a median: the
mean is exactly ``sigma^2 + C s^2`` (a median of a squared Gaussian is 0.455x
its mean, and the factor moves once the curvature adds a non-central term),
and the clip is what makes it survive dead pixels and zingers.

Identifiability. ``sigma_r`` is DETECTED when the fitted ``sigma_r^2`` is
positive, its t-statistic clears ``T_DETECT`` and the count lever clears
``MIN_LEVER``. Calibrated on the two scans in hand: Scan_1988 (lever 4.7x)
gives ``t = 10-15`` at every binning; Scan_1510 (a mouse, max line integral
0.71, lever 2.0x) gives a NEGATIVE ``sigma_r^2`` at ``t = -8``, i.e. the two
terms are not separable there and a free fit goes unphysical — which is why
the lever gate exists alongside the t-test. A scan that cannot detect it
still measures its own Poisson slope: that needs no lever, and it is refitted
with the read noise PINNED to whatever the policy supplies, so the slope is
always the one consistent with the model actually used. The R2 of that
as-used model on the scan's own bins is logged: on Scan_1510 it is poor
(~0.4) because the mouse's variance curve is FLATTER than ``1/c`` — a
deviation neither form captures, seen independently with the acquisition
pair, and left as an open question rather than papered over.

The policy (``resolve_noise_model``)
------------------------------------
For each constant, first match wins:

  poisson_slope   pinned by the caller  >  measured on this scan
                  >  (estimator failed) the package default for the binning
  read_noise      pinned by the caller  >  measured on this scan (if detected)
                  >  the detector's calibration file  >  the package default

The calibration file, ``data/calibration/detector_noise_<serial>.json``, is
what a scan with a long lever leaves behind for the scans that have none — a
mouse on the same detector inherits the phantom's read noise. It is written
only from a DETECTED measurement, and replaced only by a better-conditioned
one (higher t). Every run also writes ``detector_noise_<serial>_<scan>.json``
with its own measurement, detected or not, so the record of what each scan
could and could not determine survives the run. The same directory and the
same serial key as the detector-psi calibration; see ``paths.calibration_dir``.

The package default is the last resort and it says so loudly: it is this
lab's detector measured on one phantom scan, and it is wrong by an unknown
factor on any other hardware. ``read_noise=0`` is the explicit way to run
slope-only (pure Poisson) if that is what a detector warrants.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

#: Fallback read noise, counts PER RAW PIXEL. Measured 2026-09-09 on Scan_1988
#: (the phantom) on this lab's flat panel at four binnings: 20.2, 19.9, 21.5,
#: 19.8. Right for that detector; wrong by an unknown factor anywhere else —
#: the policy says so whenever it uses it.
DEFAULT_READ_NOISE = 20.0

#: Fallback Poisson slope AT EACH BINNING (see "pooling" above for why it is
#: not one number). Same measurement as the read noise. Used only when the
#: estimator cannot run on a scan at all; the slope is otherwise always the
#: scan's own.
DEFAULT_POISSON_SLOPE_BY_DS = {1: 0.43, 2: 0.149, 3: 0.082, 4: 0.062}

#: Detection gates for the read noise (see "identifiability" above).
T_DETECT = 4.0
MIN_LEVER = 3.0

#: Estimator defaults. Four strides over-determine the two-parameter
#: curvature solve; 24 quantile bins put ~equal samples in each; the residual
#: gate drops bins where the s^2 law fails (object edges).
DEFAULT_STRIDES = (1, 2, 3, 4)
DEFAULT_N_BINS = 24
MAX_CURVATURE_ERR = 0.10
CLIP_SIGMA = 5.0
MIN_BIN_SAMPLES = 1000
MIN_COUNT = 5.0

#: Source codes, so a run's record says WHERE each constant came from as a
#: number W&B can plot, next to the value.
SOURCE_PINNED = 0
SOURCE_MEASURED = 1
SOURCE_CALIBRATION = 2
SOURCE_DEFAULT = 3
SOURCE_NAMES = {SOURCE_PINNED: "pinned", SOURCE_MEASURED: "measured",
                SOURCE_CALIBRATION: "calibration", SOURCE_DEFAULT: "default"}

_METHOD = "stride-curvature"
_FILE_PREFIX = "detector_noise"


def default_poisson_slope(downsample: int) -> float:
    """The fallback slope for a binning; the nearest measured one otherwise."""
    ds = int(downsample)
    if ds in DEFAULT_POISSON_SLOPE_BY_DS:
        return DEFAULT_POISSON_SLOPE_BY_DS[ds]
    nearest = min(DEFAULT_POISSON_SLOPE_BY_DS, key=lambda k: abs(k - ds))
    # scale the nearest by the independent-pixel rule as a rough bridge
    return DEFAULT_POISSON_SLOPE_BY_DS[nearest] * (nearest / ds) ** 2


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

def log_variance(counts, *, poisson_slope: float, read_noise: float,
                 downsample: int = 1):
    """``Var(p)`` for pixels at binning ``downsample`` whose counts above dark
    (per-pixel MEAN over the pooled raw pixels) are ``counts``.

    ``poisson_slope`` is the slope AT THIS BINNING; ``read_noise`` is per raw
    pixel and pools as ``1/ds`` (measured, module docstring).
    """
    c = np.asarray(counts, dtype=np.float64)
    n = float(int(downsample)) ** 2
    return float(poisson_slope) / c + (float(read_noise) ** 2 / n) / c ** 2


def auto_or_float(v):
    """``'auto'`` (case-insensitive) or a float — the type of every wls constant
    at the CLI and the constructor."""
    if isinstance(v, str) and v.strip().lower() == 'auto':
        return 'auto'
    return float(v)


def detect_seam_columns(sinogram, *, t_min: float = 6.0, halfwidth: int = 3,
                        dilate: int = 1) -> np.ndarray:
    """Detector columns whose offset against their neighbours is the SAME in
    every view — a static detector artefact, not anatomy.

    Per view, each column's mean over rows is compared with the median of its
    ``2 halfwidth + 1``-column neighbourhood; anatomy sweeps across columns as
    the gantry turns and averages out, a seam does not. The mean of that
    offset over views divided by its standard error is a t-statistic; columns
    with ``|t| >= t_min`` are seams, widened by ``dilate`` on each side to
    cover the dip that flanks a tiled-panel seam. MEASURED on Scan_1510 (ds3,
    220 views): t = 10.6 at column 980 and 6.7 at 72, both with a flanking
    negative lobe, offsets of 0.02-0.06 pixel sigma per view — invisible in a
    single view, a full-height red stripe in the held-out SSIM map once the
    fit has converged, because no volume can render a per-column step. The
    ``wls`` data term gives them ZERO weight.
    Returns sorted column indices (empty when ``t_min <= 0``).
    """
    if t_min <= 0:
        return np.zeros(0, dtype=int)
    from scipy.ndimage import median_filter
    P = np.asarray(sinogram, dtype=np.float32)
    col = P.mean(axis=1)                                   # (views, cols)
    resid = col - median_filter(col, size=(1, 2 * int(halfwidth) + 1),
                                mode='nearest')
    se = resid.std(axis=0) / np.sqrt(P.shape[0])
    t = resid.mean(axis=0) / np.maximum(se, 1e-12)
    seed = np.abs(t) >= float(t_min)
    mask = seed.copy()
    for k in range(1, int(dilate) + 1):
        mask[k:] |= seed[:-k]
        mask[:-k] |= seed[k:]
    return np.nonzero(mask)[0]


def inverse_variance_weights(counts, dark=None, *, poisson_slope: float,
                             read_noise: float, min_counts: float = 1.0,
                             downsample: int = 1):
    """Per-pixel inverse variance of the log-attenuation, from the counts.

    ``counts`` is the detector reading for the SAME binned pixel the log value
    was formed from, so the per-pixel flat field (a 4x range on Scan_1510)
    is in the weight — which ``exp(-p)`` weighting cannot see. ``dark`` is
    subtracted here when given (a 2-D field or a per-view stack). Counts are
    floored at ``min_counts`` so a dead pixel gets a large finite variance
    rather than an infinite weight.
    """
    c = np.asarray(counts, dtype=np.float32)
    if dark is not None:
        d = np.asarray(dark, dtype=np.float32)
        if d.ndim == c.ndim - 1:
            d = d[None]
        c = c - d
    c = np.maximum(c, float(min_counts))
    var = log_variance(c, poisson_slope=poisson_slope, read_noise=read_noise,
                       downsample=downsample)
    return (1.0 / var).astype(np.float32)


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

@dataclass
class NoiseFit:
    """One scan's measurement.

    ``poisson_slope`` is AT the fit's binning ``downsample``; ``read_noise``
    (and ``read_noise_var``) are PER RAW PIXEL.
    """
    poisson_slope: float
    poisson_slope_se: float
    read_noise_var: float          # sigma_r^2 per raw pixel; <= 0 when undetected
    read_noise_var_se: float
    t_read_noise: float
    r2: float
    n_bins_used: int
    n_bins: int
    count_min: float
    count_max: float
    lever: float
    downsample: int
    n_samples: int
    #: (count, var_s1, sigma2, curvature, s2_err, used) per bin, at the
    #: run's binning — kept so a run can plot the curve it fitted.
    bins: np.ndarray = field(repr=False)
    reasons: list = field(default_factory=list)

    @property
    def read_noise(self) -> float:
        return math.sqrt(max(self.read_noise_var, 0.0))

    @property
    def read_noise_detected(self) -> bool:
        return not self.reasons


def _binned_stride_variance(sinogram, counts, *, strides, n_bins, clip_sigma,
                            min_count, setup_view_step=4):
    """Per count bin and stride: sigma-clipped mean of ``d2^2 / 6``.

    Streams over views so memory is O(rows x cols) whatever the view count —
    at ds1 a materialised ``d2`` stack is the size of the sinogram itself.
    Bin edges come from a view subsample (quantiles do not need every sample),
    the clip threshold from a robust L1 start on the same subsample refined
    by two clipped passes, and the final sums from every usable view.
    """
    P = np.asarray(sinogram, dtype=np.float32)
    C = np.asarray(counts, dtype=np.float32)
    if P.shape != C.shape or P.ndim != 3:
        raise ValueError(f"need matching (views, rows, cols) arrays, got "
                         f"{P.shape} and {C.shape}")
    smax = int(max(strides))
    n_views = P.shape[0]
    if n_views < 2 * smax + 2:
        raise ValueError(f"need at least {2 * smax + 2} views for strides up to "
                         f"{smax}, got {n_views}")
    ks = np.arange(smax, n_views - smax)
    n_s = len(strides)

    # pass 0 — count-quantile edges from a view subsample
    sub = C[ks[::setup_view_step]].ravel()
    sub = sub[sub > min_count]
    if sub.size < n_bins * 10:
        raise ValueError("too few pixels above the count floor to bin")
    edges = np.quantile(sub, np.linspace(0.0, 1.0, n_bins + 1))
    inner = edges[1:-1]

    def bin_index(ck):
        return np.searchsorted(inner, ck, side="right")

    def d2_of(k, s):
        return (P[k + s] - 2.0 * P[k] + P[k - s]).ravel()

    # pass 1 — clip threshold per (stride, bin): L1 start, then two clipped
    # passes. E|x| = sigma sqrt(2/pi) for a Gaussian; outliers enter the L1
    # mean linearly rather than quadratically, which is what makes it a safe
    # starting point.
    sub_ks = ks[::setup_view_step]
    l1 = np.zeros((n_s, n_bins)); cnt = np.zeros((n_s, n_bins))
    for k in sub_ks:
        ck = C[k].ravel(); valid = ck > min_count; b = bin_index(ck)
        for i, s in enumerate(strides):
            d2 = d2_of(k, s)
            l1[i] += np.bincount(b[valid], weights=np.abs(d2[valid]), minlength=n_bins)
            cnt[i] += np.bincount(b[valid], minlength=n_bins)
    sigma = (l1 / np.maximum(cnt, 1.0)) * math.sqrt(math.pi / 2.0)
    for _ in range(2):
        ss = np.zeros((n_s, n_bins)); cnt = np.zeros((n_s, n_bins))
        for k in sub_ks:
            ck = C[k].ravel(); valid = ck > min_count; b = bin_index(ck)
            for i, s in enumerate(strides):
                d2 = d2_of(k, s)
                keep = valid & (np.abs(d2) <= clip_sigma * sigma[i][b])
                ss[i] += np.bincount(b[keep], weights=d2[keep] ** 2, minlength=n_bins)
                cnt[i] += np.bincount(b[keep], minlength=n_bins)
        sigma = np.sqrt(ss / np.maximum(cnt, 1.0))
    sigma = np.where(cnt > 0, sigma, np.inf)

    # pass 2 — every usable view, clipped
    ss = np.zeros((n_s, n_bins)); nn = np.zeros((n_s, n_bins))
    csum = np.zeros(n_bins); cn = np.zeros(n_bins)
    for k in ks:
        ck = C[k].ravel(); valid = ck > min_count; b = bin_index(ck)
        csum += np.bincount(b[valid], weights=ck[valid], minlength=n_bins)
        cn += np.bincount(b[valid], minlength=n_bins)
        for i, s in enumerate(strides):
            d2 = d2_of(k, s)
            keep = valid & (np.abs(d2) <= clip_sigma * sigma[i][b])
            ss[i] += np.bincount(b[keep], weights=d2[keep] ** 2, minlength=n_bins)
            nn[i] += np.bincount(b[keep], minlength=n_bins)
    v = ss / np.maximum(nn, 1.0) / 6.0                   # (strides, bins)
    cmid = csum / np.maximum(cn, 1.0)
    return cmid, v, nn.min(axis=0), int(cn.sum())


def _weighted_fit(xs, ys, n, *, fix_slope=None, fix_read_var=None):
    """Fit ``y = A x1 + B x2`` with ``x1 = 1/c`` (slope at this binning) and
    ``x2 = 1/(n c^2)`` (read-noise variance per raw pixel) in RELATIVE least
    squares (weights ``1/y``): the per-bin means have standard errors
    proportional to their values, so an absolute fit would let the noisiest,
    lowest-count bins dictate everything. Returns ``(A, se_A, B, se_B, r2)``
    with the standard errors of the free parameters (0 for a pinned one)."""
    X = np.stack([1.0 / xs, 1.0 / (n * xs ** 2)], axis=1)
    w = 1.0 / ys
    pins = (fix_slope, fix_read_var)
    free = [i for i, fx in enumerate(pins) if fx is None]
    fixed = np.array([0.0 if fx is None else float(fx) for fx in pins])
    target = ys - X @ fixed
    beta = fixed.copy(); se = np.zeros(2)
    if free:
        Xf = X[:, free] * w[:, None]
        yw = target * w
        coef, *_ = np.linalg.lstsq(Xf, yw, rcond=None)
        beta[free] = coef
        res = yw - Xf @ coef
        dof = max(len(ys) - len(free), 1)
        cov = np.linalg.pinv(Xf.T @ Xf) * float(res @ res) / dof
        se[free] = np.sqrt(np.maximum(np.diag(cov), 0.0))
    pred = X @ beta
    ss = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - float(((ys - pred) ** 2).sum()) / ss if ss > 0 else 1.0
    return float(beta[0]), float(se[0]), float(beta[1]), float(se[1]), r2


def estimate_noise_model(sinogram, counts, *, downsample: int = 1,
                         strides=DEFAULT_STRIDES, n_bins: int = DEFAULT_N_BINS,
                         clip_sigma: float = CLIP_SIGMA,
                         max_curvature_err: float = MAX_CURVATURE_ERR,
                         min_bin_samples: int = MIN_BIN_SAMPLES,
                         min_count: float = MIN_COUNT,
                         pinned_slope: Optional[float] = None,
                         pinned_read_noise: Optional[float] = None,
                         t_detect: float = T_DETECT,
                         min_lever: float = MIN_LEVER) -> NoiseFit:
    """Measure ``A_ds`` and ``sigma_r`` from a scan's own views (module docstring).

    ``sinogram`` is the preprocessed log-attenuation the loss will see and
    ``counts`` the matching reading above dark, both ``(views, rows, cols)``
    at the run's binning ``downsample``. The slope comes back AT that binning,
    the read noise PER RAW PIXEL. ``pinned_slope`` / ``pinned_read_noise``
    hold one constant fixed and fit the other — the policy uses the second to
    refit the slope under a supplied read noise. Raises ``ValueError`` when
    the scan's view-to-view change defeats the curvature correction (fewer
    than four count bins survive the gate).
    """
    strides = tuple(int(s) for s in strides)
    n = float(int(downsample)) ** 2
    cmid, v, nmin, n_samples = _binned_stride_variance(
        sinogram, counts, strides=strides, n_bins=n_bins,
        clip_sigma=clip_sigma, min_count=min_count)

    # per bin: v(s) = sigma^2 + C s^2, solved over the strides at once
    S = np.stack([np.ones(len(strides)), np.array(strides, float) ** 2], axis=1)
    sol = np.linalg.pinv(S) @ v                         # (2, bins)
    sigma2, curv = sol[0], sol[1]
    resid = v - S @ sol
    err = np.abs(resid).max(axis=0) / np.maximum(v[0], 1e-30)
    used = (sigma2 > 0) & (err < max_curvature_err) & (nmin >= min_bin_samples)
    bins = np.stack([cmid, v[0], sigma2, curv, err, used.astype(float)], axis=1)

    if used.sum() < 4:
        raise ValueError(
            f"only {int(used.sum())} of {n_bins} count bins survive the "
            f"curvature gate (s^2 residual < {max_curvature_err:.0%}, "
            f"sigma^2 > 0, >= {min_bin_samples} samples); the view-to-view "
            f"change of this scan is not described by a local curvature, so "
            f"its noise cannot be separated from it this way.")
    xs, ys = cmid[used], sigma2[used]

    fix_b = None if pinned_read_noise is None else float(pinned_read_noise) ** 2
    A, seA, B, seB, r2 = _weighted_fit(xs, ys, n, fix_slope=pinned_slope,
                                       fix_read_var=fix_b)
    t = (B / seB) if seB > 0 else (math.inf if B > 0 else -math.inf)
    lever = float(xs.max() / max(xs.min(), 1e-9))

    reasons = []
    if pinned_read_noise is None:
        if B <= 0:
            reasons.append(f"fitted sigma_r^2 is not positive ({B:.3g}): the "
                           f"curve is not steeper than 1/c at low counts")
        elif t < t_detect:
            reasons.append(f"t = {t:.1f} < {t_detect:g}: the 1/c^2 term is not "
                           f"distinguished from the 1/c term on this range")
        if lever < min_lever:
            reasons.append(f"count lever {lever:.2f}x < {min_lever:g}x: too "
                           f"short a range for the two terms to separate")
    return NoiseFit(poisson_slope=A, poisson_slope_se=seA, read_noise_var=B,
                    read_noise_var_se=seB, t_read_noise=float(t), r2=float(r2),
                    n_bins_used=int(used.sum()), n_bins=int(n_bins),
                    count_min=float(xs.min()), count_max=float(xs.max()),
                    lever=lever, downsample=int(downsample),
                    n_samples=int(n_samples), bins=bins, reasons=reasons)


# --------------------------------------------------------------------------
# the calibration cache
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseCalibrationKey:
    """What identifies the detector (and the scan) for the cache.

    ``serial`` is the detector serial from the acquisition headers (None when
    the headers carry none: then nothing is cached), ``scan_tag`` the scan
    folder's basename, ``calib_dir`` an explicit directory or None for
    ``paths.calibration_dir()``.
    """
    serial: Optional[str]
    scan_tag: Optional[str] = None
    calib_dir: Optional[Path] = None

    def directory(self) -> Path:
        if self.calib_dir is not None:
            return Path(self.calib_dir)
        from .paths import calibration_dir
        return calibration_dir()

    def detector_path(self) -> Optional[Path]:
        if not self.serial:
            return None
        return self.directory() / f"{_FILE_PREFIX}_{self.serial}.json"

    def scan_path(self) -> Optional[Path]:
        if not self.serial or not self.scan_tag:
            return None
        return self.directory() / f"{_FILE_PREFIX}_{self.serial}_{self.scan_tag}.json"


def fit_record(fit: NoiseFit, key: NoiseCalibrationKey, elapsed_s: float) -> dict:
    """The JSON record for one measurement; the same shape for the per-scan
    and the per-detector file, so either can be read as the other."""
    return {
        "read_noise": fit.read_noise,
        "read_noise_units": "counts per raw detector pixel (pools as 1/ds)",
        "read_noise_var": fit.read_noise_var,
        "read_noise_var_se": fit.read_noise_var_se,
        "read_noise_detected": fit.read_noise_detected,
        "t_read_noise": fit.t_read_noise,
        "poisson_slope": fit.poisson_slope,
        "poisson_slope_se": fit.poisson_slope_se,
        "poisson_slope_units": f"at downsample {fit.downsample} (binning-specific; "
                               f"never transferred)",
        "r2": fit.r2,
        "lever": fit.lever,
        "count_min": fit.count_min,
        "count_max": fit.count_max,
        "n_bins_used": fit.n_bins_used,
        "n_bins": fit.n_bins,
        "n_samples": fit.n_samples,
        "reasons": list(fit.reasons),
        "downsample": fit.downsample,
        "method": _METHOD,
        "detector_serial": key.serial,
        "scan": key.scan_tag,
        "measured_on": date.today().isoformat(),
        "elapsed_s": elapsed_s,
    }


def read_record(path: Optional[Path]) -> Optional[dict]:
    if path is None or not Path(path).exists():
        return None
    try:
        rec = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or "read_noise" not in rec:
        return None
    rec["_path"] = str(path)
    return rec


def write_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in record.items()
                                if not k.startswith("_")}, indent=2))


def _better_conditioned(new: dict, old: Optional[dict]) -> bool:
    """Replace the detector record only with a measurement that determines
    the read noise at least as sharply (higher t)."""
    if old is None or not old.get("read_noise_detected", False):
        return True
    return float(new.get("t_read_noise", 0.0)) >= float(old.get("t_read_noise", 0.0))


# --------------------------------------------------------------------------
# the policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseModel:
    """The constants a run actually uses, with provenance.

    ``poisson_slope`` is at the run's binning ``downsample``; ``read_noise``
    is counts per raw pixel.
    """
    poisson_slope: float
    read_noise: float
    downsample: int
    slope_source: int
    read_noise_source: int
    fit: Optional[NoiseFit] = field(default=None, repr=False)
    calibration: Optional[dict] = field(default=None, repr=False)
    #: R2 of the model as USED (read noise from wherever it came) on this
    #: scan's own bins — the honest adequacy number, distinct from the free
    #: fit's r2 when the read noise was not this scan's own.
    r2_as_used: float = float("nan")

    def describe(self) -> str:
        return (f"Var(p) = {self.poisson_slope:.4g}/count + "
                f"({self.read_noise:.3g}/{self.downsample})^2/count^2 at "
                f"ds{self.downsample}  [slope {SOURCE_NAMES[self.slope_source]}, "
                f"read noise {SOURCE_NAMES[self.read_noise_source]}]")

    def scalars(self, prefix: str = "wls/") -> dict:
        """What goes into the run record. Numbers only."""
        out = {
            f"{prefix}poisson_slope": float(self.poisson_slope),
            f"{prefix}read_noise_counts": float(self.read_noise),
            f"{prefix}noise_downsample": int(self.downsample),
            f"{prefix}slope_source": int(self.slope_source),
            f"{prefix}read_noise_source": int(self.read_noise_source),
            f"{prefix}noise_fit_ok": int(self.fit is not None),
        }
        if self.fit is not None:
            f = self.fit
            out.update({
                f"{prefix}read_noise_t": float(f.t_read_noise),
                f"{prefix}read_noise_detected": int(f.read_noise_detected),
                f"{prefix}noise_fit_r2": float(f.r2),
                f"{prefix}noise_r2_as_used": float(self.r2_as_used),
                f"{prefix}noise_lever": float(f.lever),
                f"{prefix}noise_bins_used": int(f.n_bins_used),
                f"{prefix}noise_count_min": float(f.count_min),
                f"{prefix}noise_count_max": float(f.count_max),
            })
        return out


def _auto(v) -> bool:
    return isinstance(v, str) and v.strip().lower() == "auto"


def resolve_noise_model(sinogram, counts, *, downsample: int = 1,
                        poisson_slope="auto", read_noise="auto",
                        calibration: Optional[NoiseCalibrationKey] = None,
                        verbose: bool = True, log_fn=None,
                        **fit_kwargs) -> NoiseModel:
    """Decide the two constants for this run (policy in the module docstring).

    ``poisson_slope`` (at the run's binning) and ``read_noise`` (counts per
    raw pixel) are ``'auto'`` or a number. ``calibration`` names the detector
    for the shared file; None disables the cache in both directions. Prints
    the decision and, through ``log_fn`` if given, logs its scalars at step 0.
    """
    say = print if verbose else (lambda *a, **k: None)
    ds = int(downsample)
    pin_a = None if _auto(poisson_slope) else float(poisson_slope)
    pin_r = None if _auto(read_noise) else float(read_noise)
    if pin_r is not None and pin_r < 0:
        raise ValueError(f"read_noise must be >= 0 counts, got {pin_r}")

    if pin_a is not None and pin_r is not None:
        model = NoiseModel(pin_a, pin_r, ds, SOURCE_PINNED, SOURCE_PINNED)
        say(f"    noise model: pinned by the caller — {model.describe()}")
        if log_fn is not None:
            log_fn(model.scalars(), 0)
        return model

    # ---- measure this scan ------------------------------------------------
    t0 = time.time()
    try:
        fit = estimate_noise_model(sinogram, counts, downsample=ds,
                                   pinned_slope=pin_a, pinned_read_noise=pin_r,
                                   **fit_kwargs)
    except ValueError as e:
        fit = None
        say(f"    noise model: WARNING — could not be measured on this scan "
            f"({e})")
    elapsed = time.time() - t0
    if fit is not None:
        say(f"    noise model: measured from {fit.n_samples:,} rays over "
            f"{fit.n_bins_used}/{fit.n_bins} count bins (counts "
            f"{fit.count_min:.0f}..{fit.count_max:.0f} at ds{ds}, lever "
            f"{fit.lever:.2f}x, {elapsed:.1f} s): slope {fit.poisson_slope:.4g} "
            f"+/- {fit.poisson_slope_se:.2g}, sigma_r^2 {fit.read_noise_var:.4g} "
            f"+/- {fit.read_noise_var_se:.2g} per raw pixel (t = "
            f"{fit.t_read_noise:.1f}), R2 {fit.r2:.4f}")

    # ---- read noise ---------------------------------------------------------
    cal = None
    if pin_r is not None:
        r_src, r_val = SOURCE_PINNED, pin_r
    elif fit is not None and fit.read_noise_detected:
        r_src, r_val = SOURCE_MEASURED, fit.read_noise
        say(f"      read noise DETECTED on this scan: sigma_r = {r_val:.2f} "
            f"counts per raw pixel")
    else:
        for why in (fit.reasons if fit is not None else []):
            say(f"      read noise not identifiable here: {why}")
        cal = read_record(calibration.detector_path()) if calibration else None
        if cal is not None:
            r_src, r_val = SOURCE_CALIBRATION, float(cal["read_noise"])
            say(f"      using the detector's calibration "
                f"{Path(cal['_path']).name}: sigma_r = {r_val:.2f} counts per "
                f"raw pixel (measured on {cal.get('scan', '?')}, "
                f"{cal.get('measured_on', '?')}, t = "
                f"{float(cal.get('t_read_noise', float('nan'))):.1f})")
        else:
            r_src, r_val = SOURCE_DEFAULT, DEFAULT_READ_NOISE
            no_serial = calibration is None or not calibration.serial
            say(f"      WARNING — no detector calibration"
                f"{' (no detector serial to look one up by)' if no_serial else ''}"
                f"; using the package default sigma_r = {r_val:g} counts, which "
                f"describes one detector measured on one phantom scan and is "
                f"wrong by an unknown factor anywhere else. Run a scan with a "
                f"count range >= {MIN_LEVER:g}x on this detector once (a dense "
                f"phantom) to calibrate it, or pin the read noise explicitly "
                f"(0 = pure Poisson).")

    # ---- slope: refit under the read noise actually used ------------------
    final = fit
    if pin_a is not None:
        a_src, a_val = SOURCE_PINNED, pin_a
    elif fit is None:
        a_src, a_val = SOURCE_DEFAULT, default_poisson_slope(ds)
        say(f"      WARNING — using the package default slope {a_val:g} at "
            f"ds{ds}, measured on one detector; the chi-square scale of this "
            f"run is not to be trusted.")
    else:
        a_src = SOURCE_MEASURED
        if r_src != SOURCE_MEASURED:
            final = estimate_noise_model(sinogram, counts, downsample=ds,
                                         pinned_read_noise=r_val, **fit_kwargs)
            say(f"      slope refitted with sigma_r pinned at {r_val:.2f}: "
                f"{final.poisson_slope:.4g} +/- {final.poisson_slope_se:.2g}, "
                f"R2 as used {final.r2:.4f}")
        a_val = final.poisson_slope
        if (cal is not None and cal.get("poisson_slope")
                and int(cal.get("downsample", -1)) == ds):
            ratio = a_val / float(cal["poisson_slope"])
            if not 0.75 <= ratio <= 1.33:
                say(f"      WARNING — this scan's Poisson slope ({a_val:.3g}) "
                    f"is {ratio:.2f}x the calibration's at the same binning "
                    f"({float(cal['poisson_slope']):.3g}). The slope is a "
                    f"detector constant; a mismatch this large means a "
                    f"different gain, kV or detector, and the calibration's "
                    f"read noise may not transfer either.")

    model = NoiseModel(a_val, r_val, ds, a_src, r_src, fit=fit, calibration=cal,
                       r2_as_used=(final.r2 if final is not None else float("nan")))
    say(f"    noise model: {model.describe()}")

    # ---- cache --------------------------------------------------------------
    if fit is not None and calibration is not None and calibration.serial:
        rec = fit_record(fit, calibration, elapsed)
        sp = calibration.scan_path()
        if sp is not None:
            try:
                write_record(sp, rec)
            except OSError as e:
                say(f"      (could not write {sp.name}: {e})")
        dp = calibration.detector_path()
        if fit.read_noise_detected and dp is not None:
            old = read_record(dp)
            if _better_conditioned(rec, old):
                try:
                    write_record(dp, rec)
                    say(f"      saved {dp.name} — every scan on this detector "
                        f"now inherits this read noise")
                except OSError as e:
                    say(f"      (could not write {dp.name}: {e})")
            else:
                say(f"      kept {dp.name} (t = "
                    f"{float(old.get('t_read_noise', 0.0)):.1f} from "
                    f"{old.get('scan', '?')} is better conditioned)")
    if log_fn is not None:
        log_fn(model.scalars(), 0)
    return model
