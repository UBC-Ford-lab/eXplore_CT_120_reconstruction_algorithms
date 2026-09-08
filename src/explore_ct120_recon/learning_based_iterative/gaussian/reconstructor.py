"""Gaussian splatting as a first-class ``--algorithm``.

Every other learned backend answers three hooks (model / domain / export) and
lets ``LearnedReconstructor.reconstruct`` drive: sample rays, integrate along
them, score, step. That loop is a good fit for anything queried pointwise, and
a bad fit for a rasteriser. Splatting inverts the data access — a step consumes
one WHOLE VIEW, because the cost of a splat is paid per primitive-tile pair and
sampling scattered rays from it would throw that away — and it changes the
model between steps, which no ray-sampling loop expects.

So this subclass overrides ``reconstruct`` outright. It keeps everything else:
the base constructor's data, preprocessing and stopping configuration; the
driver's noise ceiling, HU calibration, ROI crop, VFF export and W&B panels;
the submodule's ``EarlyStopper``, ``LCurve``, ``StoppingRules``,
``PlateauLRReducer``, ``HoldoutScorer`` and covered-detector window, so a
Gaussian run anneals, stops and is scored exactly as ``voxel``, ``fdk`` and the
iterative backends do. What is different is the loop, which is what was
supposed to be different.

WHAT AN OVERRIDE HAS TO CARRY, AND WHAT THIS ONE DROPPED
--------------------------------------------------------
Replacing the loop means replacing the scaffolding around it too, and the
first version did not. It built a bare Adam and never touched its rate again —
while the driver went on passing ``lr_plateau=`` and recording
``lr_plateau: true`` in the run's own W&B config, so a long run trained at the
peak LR from first step to last and its record said otherwise. It scored with
a bare ``EarlyStopper``, dropping the per-sinogram-visit tolerance that makes
``--stop-min-gain`` mean one thing on any cadence, and the L-curve entirely.
It logged no training curve at all, and it called ``diag_fn`` without gating
``figures``, which defaults True — a heatmap and a power spectrum uploaded at
every evaluation instead of one in ``figure_every_evals``.

All three are fixed below, by mirroring the base loop rather than
reimplementing it: same schedule helper, same reducer hand-over, same
``StoppingRules``, same figure gate. The rule the pipeline already had — a
second IMPLEMENTATION is never allowed, a second LOOP sometimes is — applies
inside an override as much as across the repo boundary.

THE CONTRACT WITH THE DRIVER, IN FULL
-------------------------------------
    reconstructed_volume  (Nx, Ny, Nz) mu in mm^-1 over geometry['vol_shape']
    crossval_history      [{'iteration', 'ssim', 'psnr', 'mse'}, ...]
    n_measurements        pixels the fit actually saw
    iterations_run        after early stopping
    rays_per_batch        pixels consumed per step (one full view, here)

UNITS, IN ONE PLACE
-------------------
Two scales stand between the kernel and the pipeline, and both are undone here
and nowhere else:

    world_scale   mm -> [-1,1]^3, so the kernel's absolute radius caps apply
    signal_scale  lifts line integrals to O(1), off the alpha cutoff (`kernel`)

A rendered pixel is ``mu_path_mm * world_scale * signal_scale``; a queried
voxel is ``mu_mm * signal_scale``. Both divisions happen at the boundary.
"""
from __future__ import annotations

import math
import os
import tempfile
import time

import numpy as np
import torch

from ...ct_core.early_stop import (EarlyStopper, HoldoutScorer,
                                               LCurve, StoppingRules,
                                               resolve_holdout_index,
                                               resolve_min_iter,
                                               resolve_patience, solution_norm)
from ...ct_core.projection_diag import (covered_detector_window,
                                                    ssim_2d)
from ...ct_core.support import export_grid_geometry
from ..trainer import LearnedReconstructor
from ..training import lr_multiplier

from . import camera as cam_mod
from . import kernel, seeding
from .kernel import DEFAULT_MAX_VOXELS
from .model import GaussianCloud


def inverse_variance_weights(counts, dark=None, *, var_per_count: float,
                             var_floor: float, min_counts: float = 1.0,
                             downsample: int = 1):
    """Per-pixel inverse variance of the log-attenuation, from the counts.

    Noise model, MEASURED on Scan_1510 (acq-00/acq-01 pair, air pixels, at
    ds3 = mean of 9 raw pixels: ``Var = 0.0934 / count + 8e-6``) — Gaussian,
    nearly white, 82 % pure Poisson at the median count. The constants here
    are PER RAW DETECTOR PIXEL: ``var_per_count`` = 0.8406 and ``var_floor``
    = 7.2e-5, i.e. nine times the ds3 values, because a pooled pixel is the
    mean of ds^2 raw ones, its count is on the same scale (a mean), and both
    the Poisson term and the white electronic floor average down by ds^2:
    ``Var_ds(p) = var_per_count / (ds^2 count) + var_floor / ds^2``. Pass the
    sinogram's ``downsample`` and the same constants hold at any binning.

    ``count`` is the detector reading above dark for the SAME binned pixel
    the log value was formed from, so the per-pixel flat field (a 4x range on
    this scan) is in the weight, which ``exp(-p)`` weighting cannot see.
    Counts are floored at ``min_counts`` so a dead pixel gets a finite
    (large) variance rather than an infinite weight.
    """
    c = np.asarray(counts, dtype=np.float32)
    if dark is not None:
        d = np.asarray(dark, dtype=np.float32)
        if d.ndim == c.ndim - 1:
            d = d[None]
        c = c - d
    c = np.maximum(c, float(min_counts))
    n = float(int(downsample)) ** 2
    var = (float(var_per_count) / n) / c + float(var_floor) / n
    return (1.0 / var).astype(np.float32)


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
    fit has converged, because no volume can render a per-column step.
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


#: Scan_1510's constants (acq-00/acq-01 pair, air pixels, per RAW pixel). The
#: fallback when a scan offers no air pixels to measure its own noise on.
DEFAULT_WLS_VAR = (0.8406, 7.2e-5)


def _auto_or_float(v):
    if isinstance(v, str) and v.strip().lower() == 'auto':
        return 'auto'
    return float(v)


def estimate_noise_model(sinogram, counts, *, air_max: float = 0.05,
                         min_pixels: int = 5000, n_bins: int = 24,
                         min_lever: float = 1.25) -> dict | None:
    """Fit ``Var(log-attenuation) = a / count + b`` from the scan's OWN air
    pixels, with no repeat acquisition and no pinned constants.

    An air pixel sees the same thing in every view, so the difference between
    consecutive views is pure noise there: ``Var = mean(diff^2) / 2`` per
    pixel, immune to the slow gain drift (it is a difference) and to the fixed
    pattern (same pixel both times). The flat field gives those pixels a wide
    range of counts — 1.5x on Scan_1510 — which is the lever arm that
    separates the Poisson slope ``a`` from the white floor ``b``: the fit is
    an ordinary least squares of the per-bin median variance on 1/count over
    ``n_bins`` count-quantile bins.

    MEASURED on Scan_1510 (ds3, 220 views, 87,739 air pixels): 0.0921/count
    + 5.0e-6, against 0.0897/count + 6.4e-6 from the acq-00/acq-01 pair on
    the same pixels and 0.0934/count + 8e-6 pinned before — the slope agrees
    to 3 %, and sigma at the median count is identical (0.00716). The air
    threshold matters: ``air_max`` 0.02 leaves too short a lever (R2 0.88,
    the split of a and b unstable), 0.1 admits object-edge pixels whose
    content changes between views (slope 0.067, i.e. anatomy counted as a
    floor). Object pixels themselves are NOT usable: between two
    acquisitions they differ by 1.7x the photon variance (motion, gating),
    which is a residual the volume cannot explain, not detector noise.

    ``sinogram`` is the preprocessed log-attenuation the loss will see,
    ``counts`` the matching detector reading above dark, both ``(views, rows,
    cols)`` at the run's binning; the returned ``var_per_count`` and
    ``var_floor`` are at that binning too. Returns None when fewer than
    ``min_pixels`` are air (a phantom filling the field of view); when the
    count lever ``p95/p5`` is below ``min_lever`` the floor is pinned at zero
    and only the slope is fitted, and ``lever_ok`` says so.
    """
    P = np.asarray(sinogram, dtype=np.float32)
    C = np.asarray(counts, dtype=np.float32)
    if P.shape != C.shape or P.ndim != 3 or P.shape[0] < 3:
        raise ValueError(f"need matching (views, rows, cols) arrays with >= 3 "
                         f"views, got {P.shape} and {C.shape}")
    air = (P.max(axis=0) < float(air_max)) & (C.mean(axis=0) > 1.0)
    n_air = int(air.sum())
    if n_air < int(min_pixels):
        return None
    c = C.mean(axis=0)[air]
    d = np.diff(P[:, air], axis=0)
    v = (d * d).mean(axis=0) / 2.0
    lo, hi = np.percentile(c, [5, 95])
    lever = float(hi / max(lo, 1e-9))
    lever_ok = lever >= float(min_lever)
    q = np.quantile(c, np.linspace(0.0, 1.0, int(n_bins) + 1))
    xs, ys = [], []
    bin_min = max(10, n_air // (4 * int(n_bins)))
    for a0, a1 in zip(q[:-1], q[1:]):
        m = (c >= a0) & (c <= a1)
        if m.sum() < bin_min:
            continue
        xs.append(float(np.median(1.0 / c[m])))
        ys.append(float(np.median(v[m])))
    xs, ys = np.asarray(xs), np.asarray(ys)
    if lever_ok and len(xs) >= 3:
        A = np.stack([xs, np.ones_like(xs)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, ys, rcond=None)
        b = max(float(b), 0.0)
    else:
        a = float((xs * ys).sum() / max((xs * xs).sum(), 1e-30))
        b = 0.0
    pred = a * xs + b
    ss = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - float(((ys - pred) ** 2).sum()) / ss if ss > 0 else 1.0
    c_med = float(np.median(c))
    return {'var_per_count': float(a), 'var_floor': float(b), 'n_air': n_air,
            'count_p5': float(lo), 'count_p50': c_med, 'count_p95': float(hi),
            'lever': lever, 'lever_ok': lever_ok, 'r2': float(r2),
            'sigma_median': float(np.sqrt(a / c_med + b)),
            'poisson_share': float((a / c_med) / max(a / c_med + b, 1e-30))}


class GaussianReconstructor(LearnedReconstructor):
    """A cloud of anisotropic Gaussians, fitted one view at a time."""

    def __init__(self, projections, angles, geometry, *,
                 n_seed: int = 300_000,
                 max_gaussians: int = 2_000_000,
                 seed_from: str = 'fdk',
                 seed_scale: float = 1.0,
                 seed_floor_quantile: float = 0.5,
                 signal_scale='auto',
                 densify_from: float = 0.05,
                 densify_until: float = 0.5,
                 densify_every: int = 100,
                 densify_fraction: float = 0.02,
                 densify_grad_threshold: float | None = None,
                 min_density_frac: float = 1e-3,
                 max_scale_frac: float = 0.1,
                 dssim_weight: float = 0.2,
                 min_sigma_mm='auto',
                 max_aspect: float = 4.0,
                 max_sigma_spacing: float = 1.0,
                 view_every_evals: int = 0,
                 resume_from: str | None = None,
                 densify_roi_margin_mm: float | None = None,
                 split_mode: str = 'preserving',
                 densify_gate_z: float | None = 3.0,
                 density_control: str | None = None,
                 mcmc: bool = False,
                 mcmc_dead_frac: float = 0.01,
                 mcmc_noise: float = 0.5,
                 density_reg: float = 0.01,
                 scale_reg: float = 0.0,
                 seed_roi_weight: float = 1.0,
                 signed_density: float = 0.0,
                 loss_kind: str = 'l1_dssim',
                 wls_var_per_count='auto',
                 wls_var_floor='auto',
                 wls_min_counts: float = 1.0,
                 wls_seam_t: float = 6.0,
                 lr_multipliers: dict | None = None,
                 checkpoint_path: str | None = None,
                 export_chunk_voxels: int = DEFAULT_MAX_VOXELS,
                 **kwargs):
        self.cfg = dict(
            n_seed=int(n_seed), max_gaussians=int(max_gaussians),
            seed_from=str(seed_from), seed_scale=float(seed_scale),
            seed_floor_quantile=float(seed_floor_quantile),
            densify_from=float(densify_from),
            densify_until=float(densify_until),
            densify_every=int(densify_every),
            densify_fraction=float(densify_fraction),
            densify_grad_threshold=(None if densify_grad_threshold is None
                                    else float(densify_grad_threshold)),
            min_density_frac=float(min_density_frac),
            max_scale_frac=float(max_scale_frac),
            dssim_weight=float(dssim_weight),
            min_sigma_mm=min_sigma_mm,
            max_aspect=float(max_aspect),
            max_sigma_spacing=float(max_sigma_spacing),
            view_every_evals=int(view_every_evals),
            resume_from=(None if resume_from is None else str(resume_from)),
            densify_roi_margin_mm=(None if densify_roi_margin_mm is None
                                   else float(densify_roi_margin_mm)),
            split_mode=str(split_mode),
            densify_gate_z=(None if densify_gate_z is None
                            else float(densify_gate_z)),
            density_control=(str(density_control) if density_control
                             else ('mcmc' if mcmc else 'adaptive')),
            mcmc_dead_frac=float(mcmc_dead_frac),
            mcmc_noise=float(mcmc_noise), density_reg=float(density_reg),
            scale_reg=float(scale_reg), seed_roi_weight=float(seed_roi_weight),
            signed_density=float(signed_density),
            loss_kind=str(loss_kind),
            wls_var_per_count=_auto_or_float(wls_var_per_count),
            wls_var_floor=_auto_or_float(wls_var_floor),
            wls_min_counts=float(wls_min_counts),
            wls_seam_t=float(wls_seam_t),
            lr_multipliers=dict(lr_multipliers or {}))
        if self.cfg['loss_kind'] not in ('l1_dssim', 'wls'):
            raise ValueError(f"loss_kind must be 'l1_dssim' or 'wls', got "
                             f"{loss_kind!r}")
        if self.cfg['density_control'] not in ('adaptive', 'mcmc', 'fixed'):
            raise ValueError(f"density_control must be 'adaptive', 'mcmc' or "
                             f"'fixed', got {density_control!r}")
        # A read-only view of the mode: the loop and the tests ask
        # ``cfg['mcmc']`` and the answer must not drift from the mode.
        self.cfg['mcmc'] = self.cfg['density_control'] == 'mcmc'
        self._noise_fit = None
        self._signal_scale_arg = signal_scale
        self._export_chunk = int(export_chunk_voxels)
        self._checkpoint_path = checkpoint_path
        super().__init__(projections, angles, geometry, **kwargs)
        # One step renders one full view, so the "batch" is a whole projection.
        # Reported through the same key every backend uses so `data/*` stays
        # comparable; what differs is `sampling`, which the driver records.
        self.n_gaussians = 0
        self.world_scale = 1.0
        self.signal_scale = 1.0
        self._wls_w = None
        self._wls_scale = 1.0
        self._seam_cols = np.zeros(0, dtype=int)

    # -- the loop ----------------------------------------------------------
    def reconstruct(self) -> np.ndarray:
        device = self._device()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        sino = np.asarray(self._preprocess(), dtype=np.float32)
        n_views, n_b, n_a = sino.shape
        domain = self.build_domain()

        self.world_scale = cam_mod.world_scale_for(domain)
        self.signal_scale = cam_mod.resolve_signal_scale(
            sino, self.world_scale, self._signal_scale_arg)
        print(f"\n  Gaussian splatting on {n_views} views of {n_b}x{n_a}")
        print(f"    world scale  {self.world_scale:.6f} /mm  "
              f"(domain {(domain.aabb_max - domain.aabb_min).tolist()} mm)")
        print(f"    signal scale {self.signal_scale:.1f}x  — keeps the "
              f"rasteriser's 1e-5 alpha cutoff off the signal; see kernel.py")

        cameras = cam_mod.build_cameras(
            self.angles, geometry=self.geometry, n_b=n_b, n_a=n_a,
            world_scale=self.world_scale, device=device)

        target = torch.from_numpy(sino * self.signal_scale
                                  * self.world_scale).to(device)
        self._wls_w = None
        self._wls_scale = self.signal_scale * self.world_scale
        if self.cfg['loss_kind'] == 'wls':
            # Weighted least squares on the log-attenuation, weights = the
            # measured inverse variances, so the loss IS the reduced
            # chi-square: 1.0 means the residual is at the photon noise.
            ds = int(self.geometry.get('sinogram_downsample', 1) or 1)
            self._resolve_noise_model(sino, ds)
            w = inverse_variance_weights(
                self.projections, self.dark_field,
                var_per_count=self.cfg['wls_var_per_count'],
                var_floor=self.cfg['wls_var_floor'],
                min_counts=self.cfg['wls_min_counts'], downsample=ds)
            if w.shape != sino.shape:
                raise ValueError(f"count array {w.shape} does not match the "
                                 f"sinogram {sino.shape}")
            # Static detector seams get ZERO weight: the loss must not chase
            # a per-column step that no volume can render.
            self._seam_cols = detect_seam_columns(sino, t_min=self.cfg['wls_seam_t'])
            if len(self._seam_cols):
                w[:, :, self._seam_cols] = 0.0
                da = float(self.geometry['da']); cpa = float(self.geometry['central_pixel_a'])
                R_s = float(self.geometry['R_s']); mag = (R_s + float(self.geometry['R_d'])) / R_s
                r_mm = (self._seam_cols - cpa) * da / mag
                print(f"    seam mask: {len(self._seam_cols)} detector columns "
                      f"({100 * len(self._seam_cols) / sino.shape[2]:.2f} %) get zero "
                      f"weight — columns {self._seam_cols.tolist()} = "
                      f"{np.round(r_mm, 1).tolist()} mm from the axis at the isocentre")
                if self.log_fn is not None:
                    self.log_fn({'wls/seam_columns': int(len(self._seam_cols))}, 0)
            self._wls_w = torch.from_numpy(w).to(device)
            sig = 1.0 / np.sqrt(w[w > 0])
            print(f"    loss: weighted least squares (reduced chi-square) on "
                  f"log-attenuation; Var = {self.cfg['wls_var_per_count']:.4g}"
                  f"/count + {self.cfg['wls_var_floor']:.3g} per raw pixel -> "
                  f"at ds{ds}: {self.cfg['wls_var_per_count'] / ds**2:.4g}"
                  f"/count + {self.cfg['wls_var_floor'] / ds**2:.3g}; sigma "
                  f"p5/p50/p95 = {np.percentile(sig, 5):.4f}/"
                  f"{np.percentile(sig, 50):.4f}/{np.percentile(sig, 95):.4f}"
                  f"; no L1, no DSSIM")
        else:
            print(f"    loss: (1 - {self.cfg['dssim_weight']:g}) L1 + "
                  f"{self.cfg['dssim_weight']:g} DSSIM on whole views")

        cloud = self._seed(domain, rng, device)
        if self.cfg['resume_from']:
            self._resume(cloud, self.cfg['resume_from'])
        self._apply_density_floor(cloud)
        self.n_gaussians = len(cloud)
        self.cloud = cloud                  # the live cloud, for inspection

        # ---- shape constraints ---------------------------------------------
        # 'auto' = half the delivered voxel pitch: a Gaussian of sigma = pitch/2
        # has FWHM 1.18 pitch, which is exactly the finest thing the export grid
        # can carry. Anything below that is not identifiable and can only
        # overfit (see `GaussianCloud.project_scales`).
        pitch = float(np.mean([self.geometry['dx'], self.geometry['dx'],
                               self.geometry['dz']]))
        min_sigma_mm = (0.5 * pitch if self.cfg['min_sigma_mm'] == 'auto'
                        else float(self.cfg['min_sigma_mm']))
        self._min_sigma_w = min_sigma_mm * self.world_scale
        self._max_aspect = self.cfg['max_aspect']
        self._sigma_spacing = float(self.cfg['max_sigma_spacing'])
        self._view_every_evals = int(self.cfg['view_every_evals'])
        # Re-measured on the same cadence as densification: those are the two
        # events that change local density (splits raise it, prunes lower it),
        # and the ceiling is a statement about local density.
        self._cap_every = int(self.cfg['densify_every'])
        print(f"    shape constraints: thin axis >= {min_sigma_mm:.4f} mm "
              f"({min_sigma_mm / pitch:.2f} voxel), aspect <= "
              f"{self._max_aspect:g}, width <= "
              + (f"{self._sigma_spacing:g} x local spacing"
                 if self._sigma_spacing > 0 else "unbounded")
              + ("  [OFF]" if (self._max_aspect <= 1 and min_sigma_mm <= 0
                               and self._sigma_spacing <= 0) else ""))
        # Where clone/split may act. The reconstruction domain is ~11x the
        # export ROI on Scan_1510 (the truncated animal and the bed), and it
        # was taking HALF the primitive budget: MEASURED on run ufsqlhpn,
        # 52 % of 1.2 M primitives outside the ROI, at the same width as
        # inside. Outside, the cloud only has to supply line-integral mass,
        # which the seed already does; fineness there is fineness the
        # delivered volume never sees. So growth is confined to the ROI plus
        # a margin, and the ceiling/floor/prune keep acting everywhere.
        # Seed medians: the scale-free references for the MCMC dead line and
        # the amplitude/width regularisers (upstream's thresholds are in
        # opacity units that do not exist here).
        with torch.no_grad():
            self._seed_ref = {
                'density': float(cloud.density.median()),
                'scale': float(cloud.scaling.mean(dim=1).median()),
                'mass': float(cloud.mass().median())}
        if self.cfg['density_control'] == 'fixed':
            print(f"    density control: NONE — a fixed cloud of {len(cloud):,}; "
                  f"no split/clone/prune, no relocation, no noise gate, no "
                  f"Langevin step (the width ceiling and shape projection "
                  f"still apply)")
        elif self.cfg['mcmc']:
            print(f"    MCMC density control: count fixed at {len(cloud):,}; "
                  f"dead below {self.cfg['mcmc_dead_frac']:g} x the seed "
                  f"median mass, relocated onto live primitives by mass every "
                  f"{self.cfg['densify_every']} iterations; Langevin step "
                  f"{self.cfg['mcmc_noise']:g} sigma for the near-dead; "
                  f"amplitude L1 {self.cfg['density_reg']:g}, width L1 "
                  f"{self.cfg['scale_reg']:g}")
        self._densify_box = self._densify_bounds(self.cfg['densify_roi_margin_mm'])
        if self._densify_box is not None:
            lo, hi = self._densify_box
            n_in = int(cloud.inside_box(lo, hi).sum())
            print(f"    densification confined to the export ROI + "
                  f"{self.cfg['densify_roi_margin_mm']:g} mm: "
                  f"{n_in:,} of {len(cloud):,} seed primitives eligible")
        cap = cloud.update_spacing_cap(self._sigma_spacing)
        if self._sigma_spacing > 0:
            print(f"    width ceiling: median {cap['cap_median'] / self.world_scale:.4f} mm, "
                  f"{cap['cap_n']:,} of {len(cloud):,} primitives above it at seed")
        cloud.project_scales(self._min_sigma_w, self._max_aspect)
        optimizer = torch.optim.Adam(
            self._param_groups(cloud),
            eps=1e-15, fused=torch.cuda.is_available())
        print("    lr groups: " + ", ".join(
            f"{g['name']} {g['lr']:.2e}" for g in optimizer.param_groups))

        # ---- LR ownership ----------------------------------------------
        # The same three-way split the base loop uses, because a Gaussian run
        # is configured through the same flags and its W&B config records the
        # same ``lr_plateau: true``. This override used to build a bare Adam
        # and never touch its rate again: every run trained at the peak LR from
        # the first step to the last while its own run record claimed
        # closed-loop control, which is worse than having no schedule — the
        # record was wrong, not just the run.
        #
        # Position is the parameter that most needs the decay. Upstream 3DGS
        # anneals the position rate by ~100x over a run; without it the cloud
        # keeps displacing primitives by the step it needed on iteration one,
        # long after the structure they belong to has been found.
        _schedule = self._lr_schedule_fn or lr_multiplier(
            "cosine", warmup=self.lr_warmup_iters, total=self.iterations,
            hold=self.lr_hold_fraction, floor=self.lr_floor)
        # Captured by INDEX, which stays valid across densification: the
        # optimiser-state surgery in `model` mutates ``group['params'][0]`` in
        # place and never rebuilds the param_groups list, so group i is still
        # group i after a clone/split/prune round.
        base_lrs = [g['lr'] for g in optimizer.param_groups]

        # ---- evaluation projection ---------------------------------------
        holdout = (self._resolve_holdout(n_views)
                   if self.crossval and n_views > 1 else None)
        train_views = [i for i in range(n_views)
                       if not (self.withhold_eval and i == holdout)]
        window = covered_detector_window(self.geometry, n_b, n_a)
        # Resolved BEFORE the loop rather than after it: the stopping tolerance
        # is expressed per sinogram VISIT, and a visit is not defined without
        # them.
        self.rays_per_batch = ((window[1] - window[0])
                               * (window[3] - window[2]))
        self.n_measurements = len(train_views) * self.rays_per_batch

        scorer = stopper = lcurve = rules = None
        _patience = 0
        if holdout is not None:
            scorer = HoldoutScorer(sino[holdout], window=window,
                                   label='held-out' if self.withhold_eval
                                         else 'in-sample')
            print(f"    scoring projection {holdout} "
                  f"({'WITHHELD from training' if self.withhold_eval else 'kept in training'}), "
                  f"rows [{window[0]}, {window[1]}) cols [{window[2]}, {window[3]})")

        # The reducer is DRIVEN by the held-out improvement flag. With no
        # evaluation there is no signal, so it would pin the LR at its
        # post-warmup base for the whole run — strictly worse than the cosine
        # it displaced. Hand the LR back rather than silently doing that.
        if holdout is None and self.lr_plateau is not None:
            print("  lr_plateau: ignored — it is driven by the held-out "
                  "evaluation, and this run has none (--no-crossval). The "
                  "open-loop cosine keeps LR control.")
            self.lr_plateau = None

        if holdout is not None:
            _closed = self.lr_plateau is not None
            # One step consumes one whole view, so a visit is exactly one pass
            # over the training views. Written the base loop's way
            # (iterations x rays / measurements) so the two definitions are
            # visibly the same number and not two conventions.
            _visits_per_eval = (self.eval_every * self.rays_per_batch
                                / max(1, self.n_measurements))
            _patience = resolve_patience(self.iterations, self.eval_every,
                                         self.patience, closed_loop=_closed)
            _min_iter = resolve_min_iter(self.iterations, self.min_stop_iter,
                                         closed_loop=_closed)
            stopper = self._early_stopper or EarlyStopper(
                patience=_patience, min_delta=self.stop_min_delta,
                min_delta_rel=self.stop_min_gain * _visits_per_eval,
                metric=self.stop_metric, min_iter=_min_iter)
            if self.l_curve:
                kind = ("holdout projection" if self.withhold_eval
                        else "eval projection")
                # The gradient seminorm differences a lattice, and a cloud has
                # none. l2 on the density vector is the norm that is defined
                # here; see `_solution_norm` for why it is density alone.
                if self.l_curve_norm != 'l2':
                    print(f"  L-curve: norm {self.l_curve_norm!r} needs a "
                          f"lattice to difference and a point cloud has none; "
                          f"using l2 on the density vector.")
                lcurve = LCurve(patience=_patience, norm='l2',
                                residual_kind=kind)
            rules = StoppingRules(stopper=stopper, lcurve=lcurve,
                                  stop_on=self.stop_on)
            print(f"  Stopping on {' + '.join(self.stop_on)}: "
                  f"{self.stop_metric}, patience {_patience} evals "
                  f"({_patience * self.eval_every} iters)"
                  + ("; no iteration floor — the stop is gated on the LR "
                     "reaching its floor (closed loop)" if _closed else
                     f", no stop before iteration {_min_iter}")
                  + (f", tolerance {100 * self.stop_min_gain:g} % per sinogram "
                     f"visit = {100 * self.stop_min_gain * _visits_per_eval:g} "
                     f"% per evaluation" if self.stop_min_gain > 0 else
                     ", ZERO tolerance (any decrease counts)")
                  + (f", L-curve corner ({lcurve.residual_kind})"
                     if lcurve is not None else ""))

        d_from = int(self.cfg['densify_from'] * self.iterations)
        d_until = int(self.cfg['densify_until'] * self.iterations)
        if self.cfg['density_control'] == 'fixed':
            # No window at all: every ``d_from <= it <= d_until`` below is
            # false, which silences the gate, the split/prune round and the
            # relocation round alike.
            d_from, d_until = 1, 0
        # The noise gate needs a round of null gradients BEFORE the first
        # densification, so its control experiment starts one round early.
        gate_z = self.cfg['densify_gate_z']
        gate_from = max(1, d_from - int(self.cfg['densify_every']))
        self._last_densify = None
        mcmc = bool(self.cfg['mcmc'])
        dead_line = self.cfg['mcmc_dead_frac'] * self._seed_ref['mass']
        if gate_z is not None and not mcmc and d_until >= d_from:
            every = int(self.cfg['densify_every'])
            # z = sum d / sqrt(sum d^2) <= sqrt(n): a round shorter than z^2
            # visits cannot pass the gate whatever the data say.
            note = (f"  WARNING: densify_every={every} < z^2={gate_z ** 2:.0f}, "
                    f"so no primitive can pass; raise it or set the gate to 0"
                    if every < gate_z ** 2 else "")
            print(f"    noise gate: refine only where the data gradient beats "
                  f"the sign-scrambled residual's by z >= {gate_z:g} "
                  f"({every} visits per round, max reachable z "
                  f"{math.sqrt(every):.1f}); split mode "
                  f"'{self.cfg['split_mode']}'{note}")
        min_density = self.cfg['min_density_frac'] * float(
            cloud.density.median())
        max_scale = self.cfg['max_scale_frac'] * 2.0   # of the [-1,1] domain
        split_scale = 2.0 * float(np.mean(
            [self.geometry['dx'], self.geometry['dx'], self.geometry['dz']])
        ) * self.world_scale

        # How many past clouds to keep so the L-curve's corner can be RETURNED
        # and not merely reported. Sized at the primitive CAP rather than the
        # current count, because the cloud grows while these are being kept.
        lcurve_snapshots: dict = {}
        lcurve_depth = 0
        if lcurve is not None and 'lcurve' in self.stop_on:
            want = max(LCurve.MIN_POINTS, lcurve.smooth) + _patience + 1
            state_bytes = 11 * 4 * self.cfg['max_gaussians']
            afford = max(1, int(self.l_curve_snapshot_gib * 2**30)
                         // max(1, state_bytes))
            lcurve_depth = min(want, afford)
            print(f"  L-curve: keeping the last {lcurve_depth} clouds on the "
                  f"host ({lcurve_depth * state_bytes / 2**30:.2f} GiB at the "
                  f"cap) so the corner's iterate can be returned"
                  + ("" if lcurve_depth >= want else
                     f" — WANTED {want}; a corner further back than "
                     f"{lcurve_depth} checkpoints will be reported but not "
                     f"restored (raise l_curve_snapshot_gib)"))

        order, cursor = [], 0
        t0 = time.time()
        self.iterations_run = 0
        stop_reason = 'max iterations'
        for it in range(1, self.iterations + 1):
            staged = False        # a stage frame was drawn this iteration
            # LR ownership, per iteration. The warmup ramp always owns the
            # ramp; after it the reducer takes over for good when there is one,
            # and the cosine owns the whole run when there is not. `_schedule`
            # counts from zero, this loop from one.
            if self.lr_plateau is None or it <= self.lr_warmup_iters:
                scale = float(_schedule(it - 1))
                for group, g0 in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = g0 * scale
            elif not self.lr_plateau.active:
                for group, g0 in zip(optimizer.param_groups, base_lrs):
                    group['lr'] = g0
                self.lr_plateau.activate(optimizer)
                print(f"  lr_plateau: taking over LR control at iteration {it} "
                      f"(warmup done, base "
                      f"{', '.join(f'{g:.2e}' for g in base_lrs)})")
                self._lr_stage(cloud, domain, device, it, 0, base_lrs[0])
                staged = True

            if cursor >= len(order):
                order = list(rng.permutation(train_views))
                cursor = 0
            view = int(order[cursor]); cursor += 1

            out = self._render_train(cloud, cameras, view)
            pred = cam_mod.Camera.to_ours(out['image'])
            loss = self._loss(pred, target[view], view)
            if mcmc:
                # Upstream's opacity_reg / scale_reg, made scale-free by the
                # seed medians. The amplitude term is what retires redundant
                # primitives so relocation has something to move.
                if self.cfg['density_reg'] > 0:
                    loss = loss + self.cfg['density_reg'] * (
                        cloud.density.mean() / self._seed_ref['density'])
                if self.cfg['scale_reg'] > 0:
                    loss = loss + self.cfg['scale_reg'] * (
                        cloud.scaling.mean() / self._seed_ref['scale'])
            loss = self._regularise(loss, cloud, view)

            optimizer.zero_grad(set_to_none=True)
            if gate_z is not None and not mcmc and gate_from <= it <= d_until:
                # The noise gate's control experiment: the same loss, the same
                # view, against a target whose residual has the magnitudes of
                # the real one and random signs. Its gradient is what noise
                # alone produces for every primitive; `record_gradients` pairs
                # it with the real one. One extra backward per iteration while
                # densification can fire (`GaussianCloud.record_null_gradients`).
                with torch.no_grad():
                    r = (pred - target[view]).abs()
                    flip = torch.where(torch.rand_like(r) < 0.5, -1.0, 1.0)
                    null_target = pred.detach() - r * flip
                self._loss(pred, null_target, view).backward(retain_graph=True)
                with torch.no_grad():
                    cloud.record_null_gradients(out['screenspace'],
                                                out['visible'])
                    if out['screenspace'].grad is not None:
                        out['screenspace'].grad.zero_()
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                cloud.record_gradients(out['screenspace'], out['visible'],
                                       out['radii'])
            optimizer.step()
            # A projection, applied where the voxel backend applies its own
            # non-negativity projection: a constraint on the solution set, not
            # a penalty competing with the data term.
            # Positions keep moving after densification stops, so the ceiling
            # keeps being re-measured for the whole run; the densify branch
            # below re-measures on its own when it fires, so skip it here.
            if (self._sigma_spacing > 0 and self._cap_every > 0
                    and it % self._cap_every == 0
                    and not (d_from <= it <= d_until)):
                cloud.update_spacing_cap(self._sigma_spacing)
            clamped = cloud.project_scales(self._min_sigma_w, self._max_aspect)
            if mcmc and self.cfg['mcmc_noise'] > 0:
                lr_ratio = float(optimizer.param_groups[0]['lr']) / base_lrs[0]
                cloud.add_position_noise(
                    amount=self.cfg['mcmc_noise'] * lr_ratio,
                    dead_score=cloud.mass().abs() / dead_line)
            self.iterations_run = it

            # ---- held-out evaluation, BEFORE densification -----------------
            # MEASURED, run rsgwh174 (Scan_1510 ds3/75 um): densification
            # fires at multiples of densify_every (100) and evaluation at
            # multiples of eval_every (250), so every x000/x500 evaluation ran
            # IMMEDIATELY after a clone/split/prune/re-cap round while every
            # x250 one had 50 recovery steps. A split halves each parent's mass
            # by construction (two children at sigma/1.6, same amplitude) and
            # the width re-measure that follows tightens every neighbourhood
            # the new primitives landed in; density trains at 100x the base
            # rate and repairs ~2.5 % of mass in ~50 steps, so the x250 curve
            # was smooth and the x000/x500 curve sat 3-9 dB below it, deepening
            # as the cloud grew. The stopper read three of those dips as "no
            # improvement" and ended the run at 5750 with 63 % of its budget;
            # the LR reducer halved on them too. Evaluating first scores the
            # cloud the optimiser actually produced, and the best-iterate
            # snapshot is then a pre-densify cloud as well.
            if holdout is not None and (it % self.eval_every == 0
                                        or it == self.iterations):
                metrics = self._evaluate(cloud, cameras[holdout], scorer,
                                         view=holdout)
                n_evals = len(self.crossval_history) + 1
                self.crossval_history.append({'iteration': it, **metrics})
                if self.diag_fn is not None:
                    # Figures are GATED. `log_projection_diag` defaults
                    # figures=True, so calling it every evaluation uploaded a
                    # heatmap and a power spectrum every time — a few hundred
                    # per long run, against the base loop's one in four.
                    self.diag_fn(self._line_integral(cloud, cameras[holdout],
                                                     view=holdout),
                                 scorer.crop(sino[holdout]), it,
                                 figures=(n_evals % self.figure_every_evals == 0))
                elif self.log_fn is not None:
                    self.log_fn({f'diag/{k}': v for k, v in metrics.items()},
                                step=it)
                if self.log_fn is not None and torch.cuda.is_available():
                    # The allocator's PEAK, not nvidia-smi's reservation: the
                    # cache grows into whatever is free and says nothing about
                    # need (MEASURED: 15.4 GiB reserved, then trimmed to 7.8,
                    # for the same 2.4 M cloud). This is what the preflight's
                    # size model has to reproduce.
                    self.log_fn({'sys/gpu_peak_alloc_gib':
                                     torch.cuda.max_memory_allocated() / 2**30,
                                 'sys/gpu_reserved_gib':
                                     torch.cuda.memory_reserved() / 2**30},
                                step=it)

                improved = stopper.update(
                    it, metrics,
                    snapshot_fn=((lambda: self._state(cloud))
                                 if self.save_best else None))
                if self.lr_plateau is not None and self.lr_plateau.step(
                        improved, optimizer):
                    # A reduction resets the stopper's patience on purpose: a
                    # noise dip should buy an LR cut, not end the run.
                    stopper.num_bad = 0
                    lrs = ', '.join(f'{g:.2e}' for g in
                                    self.lr_plateau.current_lrs(optimizer))
                    print(f"  lr_plateau: reduction "
                          f"{self.lr_plateau.num_reductions} -> {lrs}")
                    if self.log_fn is not None:
                        self.log_fn(
                            {'lr_plateau/reductions':
                                 self.lr_plateau.num_reductions,
                             'lr_plateau/lr':
                                 self.lr_plateau.current_lrs(optimizer)[0]}, it)
                    self._lr_stage(cloud, domain, device, it,
                                   self.lr_plateau.num_reductions,
                                   self.lr_plateau.current_lrs(optimizer)[0])
                    staged = True
                # The denser record: a frame every K evaluations, on top of
                # the one per LR change. Free for this backend (three planes
                # of the ROI cost ~20 ms, the whole ROI 0.9 s), so the cadence
                # is the caller's choice; a reduction that already drew this
                # evaluation is not drawn twice.
                if (self._view_every_evals > 0 and not staged
                        and n_evals % self._view_every_evals == 0):
                    self._lr_stage(
                        cloud, domain, device, it,
                        (self.lr_plateau.num_reductions
                         if self.lr_plateau is not None else 0),
                        float(optimizer.param_groups[0]['lr']),
                        slug=f"iter{it:06d}")
                if lcurve is not None:
                    lcurve.add(it, float(np.sqrt(metrics['mse'])),
                               self._solution_norm(cloud))
                    if 'lcurve' in self.stop_on and lcurve_depth > 0:
                        lcurve_snapshots[it] = self._state(cloud)
                        for old_it in sorted(lcurve_snapshots)[:-lcurve_depth]:
                            del lcurve_snapshots[old_it]
                    if self.log_fn is not None:
                        c_it, _ = lcurve.corner()
                        self.log_fn({'lcurve/residual': lcurve.residual[-1],
                                     'lcurve/solution': lcurve.solution[-1],
                                     **({'lcurve/corner_iter': c_it}
                                        if c_it is not None else {})}, it)

                if rules.should_stop():
                    if (self.lr_plateau is not None
                            and not self.lr_plateau.at_floor(optimizer)):
                        # Patience ran out but the LR has further to fall.
                        # Stopping here would deliver an un-annealed cloud.
                        stopper.num_bad = 0
                        rules.fired = None
                    else:
                        stop_reason = rules.reason()
                        break

            if (mcmc and d_from <= it <= d_until
                    and it % self.cfg['densify_every'] == 0):
                eligible = (None if self._densify_box is None else
                            cloud.inside_box(*self._densify_box))
                dead = cloud.mass().abs() < dead_line
                stats = cloud.relocate_dead(optimizer, dead=dead,
                                            targets_ok=eligible)
                self._last_densify = {'relocated': stats['relocated'],
                                      'gated': -1, 'eligible': len(cloud)}
                cap = cloud.update_spacing_cap(self._sigma_spacing,
                                               fresh=stats.get('moved', dead))
                cloud.project_scales(self._min_sigma_w, self._max_aspect)
                if self.log_fn is not None:
                    self.log_fn({'model/n_gaussians': len(cloud),
                                 'model/relocated': stats['relocated'],
                                 'model/relocate_targets': stats['targets'],
                                 'model/dead_frac':
                                     stats['dead'] / max(1, len(cloud)),
                                 'model/cap_median_mm':
                                     cap['cap_median'] / self.world_scale,
                                 'model/cap_over': cap['cap_n']}, step=it)
            elif d_from <= it <= d_until and it % self.cfg['densify_every'] == 0:
                eligible = (None if self._densify_box is None else
                            cloud.inside_box(*self._densify_box))
                stats = cloud.densify_and_prune(
                    optimizer,
                    fraction=self.cfg['densify_fraction'],
                    absolute=self.cfg['densify_grad_threshold'],
                    split_scale=split_scale, min_density=min_density,
                    max_scale=max_scale, max_count=self.cfg['max_gaussians'],
                    eligible=eligible, split_mode=self.cfg['split_mode'],
                    gate_z=gate_z, min_sigma=self._min_sigma_w)
                self.n_gaussians = stats['after']
                self._last_densify = stats
                # Fresh children keep their parent's ceiling this round (see
                # `update_spacing_cap`); the random split also needs the
                # projection because it shrinks all three axes alike and can
                # push a child below the floor.
                fresh = None
                if stats['n_new'] > 0:
                    fresh = torch.arange(len(cloud), device=device) >= (
                        len(cloud) - stats['n_new'])
                cap = cloud.update_spacing_cap(self._sigma_spacing, fresh=fresh)
                cloud.project_scales(self._min_sigma_w, self._max_aspect)
                if self.log_fn is not None:
                    self.log_fn({'model/n_gaussians': stats['after'],
                                 'model/densify_eligible': stats['eligible'],
                                 'model/densify_threshold': stats['threshold'],
                                 'model/cloned': stats['cloned'],
                                 'model/split': stats['split'],
                                 'model/pruned': stats['pruned'],
                                 # How many primitives the data still asked to
                                 # refine (-1: gate idle). Zero means the data
                                 # have gone silent and further rounds would
                                 # only fill the null space.
                                 'model/densify_gated': stats['gated'],
                                 'model/densify_floored': stats['floored'],
                                 'model/cap_median_mm':
                                     cap['cap_median'] / self.world_scale,
                                 'model/cap_over': cap['cap_n']}, step=it)

            # The training curve every other backend logs. Without it a
            # Gaussian run has no loss, no LR and no rate in W&B — only the
            # diag/* scalars at each evaluation, which cannot show a schedule
            # doing (or not doing) its job.
            if self.log_every and (it % self.log_every == 0
                                   or it == self.iterations):
                rate = it / max(1e-9, time.time() - t0)
                visits = it / max(1, len(train_views))
                lr_now = float(optimizer.param_groups[0]['lr'])
                last = getattr(self, '_last_densify', None)
                gated = (f"  gated {last['gated']:,}/{last['eligible']:,}"
                         if last is not None and last['gated'] >= 0
                         and d_from <= it <= d_until else "")
                if (last is not None and 'relocated' in last
                        and d_from <= it <= d_until):
                    gated = f"  relocated {last['relocated']:,}"
                print(f"  iter {it:6d}/{self.iterations}  "
                      f"loss {loss.item():.3e}  lr {lr_now:.2e}  "
                      f"{rate:.2f} it/s  {self.n_gaussians:,} gaussians  "
                      f"{visits:.2f} visits/measurement{gated}", flush=True)
                if self.log_fn is not None:
                    self.log_fn({'train/loss': float(loss.item()),
                                 'train/lr': lr_now,
                                 'train/it_per_s': rate,
                                 'train/data_visits': visits,
                                 'train/rays_per_batch': self.rays_per_batch,
                                 'model/n_gaussians': self.n_gaussians,
                                 # How hard the shape constraints are biting.
                                 # A number that stays high means the data term
                                 # is still pushing towards needles and the cap
                                 # is the only thing holding them back.
                                 'model/clamped_thin': clamped['clamped_thin'],
                                 'model/clamped_aspect':
                                     clamped['clamped_aspect'],
                                 'model/clamped_wide':
                                     clamped['clamped_wide']}, it)

        print(f"  {self.iterations_run} iterations in "
              f"{(time.time() - t0) / 60:.2f} min, "
              f"{self.n_gaussians:,} Gaussians — {stop_reason}")
        if holdout is not None and self.crossval_history:
            last = self.crossval_history[-1]
            print(f"  Eval projection: {self.stop_metric} "
                  f"{last[self.stop_metric]:.6g} at the end, best "
                  f"{stopper.best:.6g} at iter {stopper.best_iter}")
            if lcurve is not None:
                c_it, _ = lcurve.corner()
                print(f"  L-curve corner: "
                      f"{c_it if c_it is not None else 'none yet'}"
                      + ("" if c_it is None or stopper.best_iter is None
                         or c_it == stopper.best_iter
                         else f" — DISAGREES with the held-out peak "
                              f"({stopper.best_iter})"))
            # Guarantee final figures even when the last evaluation was not a
            # figure checkpoint, exactly as the base loop does.
            if (self.diag_fn is not None
                    and len(self.crossval_history) % self.figure_every_evals != 0):
                self.diag_fn(self._line_integral(cloud, cameras[holdout],
                                                 view=holdout),
                             scorer.crop(sino[holdout]), self.iterations_run,
                             figures=True, scalars=False)

        # ---- restore the delivered iterate --------------------------------
        # Only when a rule FIRED: a run that reached its iteration limit was
        # asked for every iteration and gets the last one.
        self.stop_iter = int(self.iterations_run)
        self.stopped_by = None if rules is None else rules.fired
        self.delivered_iter = self.stop_iter
        if rules is not None and self.stopped_by is not None:
            keep = rules.best_iter()
            state = (stopper.best_state if self.stopped_by == 'holdout'
                     else lcurve_snapshots.get(keep))
            if keep is not None and keep != self.stop_iter and state is not None:
                print(f"  restoring the iterate from iteration {keep} "
                      f"({self.stopped_by})")
                self._restore_state(cloud, state)
                self.delivered_iter = int(keep)
        self._checkpoint(cloud)
        self.reconstructed_volume = self._export(cloud, domain)
        return self.reconstructed_volume

    # -- pieces ------------------------------------------------------------
    def _loss(self, pred, target, view=None):
        """The data term on one whole view.

        ``loss_kind='wls'``: weighted least squares in LOG-ATTENUATION units,
        ``mean_i w_i (pred_i - target_i)^2`` with ``w_i = 1 / Var_i`` from the
        measured counts (`inverse_variance_weights`), i.e. the reduced
        chi-square of the view; the maximum-likelihood term for the measured
        noise (Gaussian, mildly heteroscedastic), with nothing else added.
        Rendered units are divided out first, so the value is in sigmas.

        ``'l1_dssim'`` (default): L1 + DSSIM, the splatting objective. L1
        rather than L2 because a splat's error is spatially concentrated — a
        misplaced primitive is a large error over a few pixels, and squaring
        it lets one primitive dominate a step. The structural term is the
        submodule's own ``ssim_2d``, not a second implementation, so the
        number in the loss and the number in ``diag/ssim`` mean the same
        thing.
        """
        if self._wls_w is not None:
            if view is None:
                raise ValueError("wls needs the view index for its weights")
            r = (pred - target) / self._wls_scale
            return (self._wls_w[view] * r * r).mean()
        l1 = torch.abs(pred - target).mean()
        w = self.cfg['dssim_weight']
        if w <= 0:
            return l1
        dr = float(target.max() - target.min()) or 1.0
        return (1.0 - w) * l1 + w * (1.0 - ssim_2d(pred, target,
                                                   data_range=dr))

    def _checkpoint(self, cloud) -> str | None:
        """Write the delivered cloud to disk BEFORE the volume is queried.

        The export is the last step of the run and the only one that
        serialises anything, which made it the one step whose failure costs
        everything. MEASURED, run dxphgwxx: 7 750 iterations and 126 minutes of
        training, the held-out peak correctly detected at 6 750 and that
        iterate correctly restored — then the extension OOMed inside
        `voxelize_gaussians` and all of it was gone, because a cloud that has
        not been queried has not been saved.

        A cloud is small (11 floats per primitive: 53 MB at 1.2 M), so there is
        no reason for this to be optional. Everything needed to re-export
        without retraining goes in: the parameters, the two unit scales, and
        the grid they belong to.
        """
        path = self._checkpoint_path or os.path.join(
            tempfile.gettempdir(), f"gaussian_cloud_{os.getpid()}.pt")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save({**self._checkpoint_payload(cloud),
                        'world_scale': self.world_scale,
                        'signal_scale': self.signal_scale,
                        'geometry': {k: self.geometry[k] for k in
                                     ('vol_shape', 'vol_origin', 'dx', 'dz')
                                     if k in self.geometry},
                        'n_gaussians': len(cloud),
                        'delivered_iter': getattr(self, 'delivered_iter', None)},
                       path)
            print(f"  cloud checkpoint -> {path} "
                  f"({os.path.getsize(path) / 2**20:.0f} MiB); the export can "
                  f"be redone from this without retraining")
            return path
        except Exception as e:                     # never lose a run to this
            print(f"  WARNING: could not write the cloud checkpoint ({e}); "
                  f"an export failure would now cost the whole run")
            return None

    def _solution_norm(self, cloud) -> float:
        """||x|| for the L-curve: l2 of the density vector, in mm^-1.

        The base loop concatenates every floating-point parameter, which for a
        voxel grid IS the solution. Doing that here would mix positions and
        unit quaternions into the norm, so the curve would climb whenever
        primitives MIGRATE rather than when the solution grows — the opposite
        of what the ||x|| axis is for, since the L-curve reads a rising ||x||
        as noise being fitted. Density is the parameter that carries mu, and it
        is divided back out of rendered units so the number means the same
        thing at any ``--gauss-signal-scale``.
        """
        d = cloud.density.detach().reshape(-1).float().cpu().numpy()
        return solution_norm(d / max(self.signal_scale, 1e-30), 'l2')

    # -- the representation seams -------------------------------------------
    # Everything a subclass needs to change WHAT is rasterised without touching
    # the loop: the cloud handed to the kernel for a given view, the parameter
    # groups, extra priors, and what a snapshot / checkpoint carries. The base
    # answers are the static cloud's, so this class behaves exactly as before.
    def _render_train(self, cloud, cameras, view: int) -> dict:
        """The training render of ``view``: the kernel on the cloud as is."""
        return kernel.render(cloud, cameras[view], need_screenspace_grad=True)

    def _render_eval(self, cloud, camera, view=None) -> torch.Tensor:
        """The evaluation render, in RENDERED units (see `kernel.render`)."""
        return kernel.render(cloud, camera)['image']

    def _param_groups(self, cloud) -> list:
        return cloud.param_groups(self.lr, self.cfg['lr_multipliers'])

    def _regularise(self, loss, cloud, view: int):
        """Priors added to the data term; none for a static cloud."""
        return loss

    def _state(self, cloud) -> dict:
        """What the best-iterate / L-curve snapshots carry."""
        return cloud.snapshot()

    def _restore_state(self, cloud, state: dict) -> None:
        cloud.restore(state)

    def _checkpoint_payload(self, cloud) -> dict:
        """The representation's part of the checkpoint file (`_checkpoint`)."""
        return {'cloud': cloud.snapshot()}

    @torch.no_grad()
    def _line_integral(self, cloud, camera, view=None) -> np.ndarray:
        """One rendered projection, in mm^-1 * mm, cropped to the scored window."""
        img = cam_mod.Camera.to_ours(self._render_eval(cloud, camera, view))
        arr = img.detach().cpu().numpy() / (self.world_scale * self.signal_scale)
        b0, b1, a0, a1 = covered_detector_window(self.geometry, *arr.shape)
        return np.ascontiguousarray(arr[b0:b1, a0:a1])

    @torch.no_grad()
    def _evaluate(self, cloud, camera, scorer, view=None) -> dict:
        return scorer.score(self._line_integral(cloud, camera, view))

    def _resolve_noise_model(self, sino, ds: int) -> None:
        """Fill 'auto' WLS constants from the scan itself (`estimate_noise_model`).

        The fit is at the run's binning; the config keeps the per-RAW-pixel
        convention `inverse_variance_weights` expects, so both are multiplied
        by ds^2 on the way in. Without enough air pixels the Scan_1510 pair
        constants stand in, loudly: they describe one detector at one
        exposure and are wrong by an unknown factor anywhere else.
        """
        a, b = self.cfg['wls_var_per_count'], self.cfg['wls_var_floor']
        if a != 'auto' and b != 'auto':
            print(f"    noise model: pinned by the caller")
            return
        counts = np.asarray(self.projections, dtype=np.float32)
        if self.dark_field is not None:
            counts = counts - np.asarray(self.dark_field, dtype=np.float32)[None]
        fit = estimate_noise_model(sino, counts, **self.noise_fit_kwargs)
        self._noise_fit = fit
        if fit is None:
            fa, fb = DEFAULT_WLS_VAR
            print(f"    noise model: WARNING — too few air pixels to measure "
                  f"this scan's noise; using Scan_1510's constants "
                  f"({fa:g}/count + {fb:g} per raw pixel), which are only "
                  f"right for that detector at that exposure")
            if self.log_fn is not None:
                self.log_fn({'wls/noise_fit_ok': 0}, 0)
        else:
            fa, fb = fit['var_per_count'] * ds ** 2, fit['var_floor'] * ds ** 2
            print(f"    noise model: measured from {fit['n_air']:,} air pixels "
                  f"over consecutive views: Var = {fit['var_per_count']:.4g}"
                  f"/count + {fit['var_floor']:.3g} at ds{ds} (R2 "
                  f"{fit['r2']:.3f}, count p5/p50/p95 {fit['count_p5']:.0f}/"
                  f"{fit['count_p50']:.0f}/{fit['count_p95']:.0f}, lever "
                  f"{fit['lever']:.2f}"
                  + ("" if fit['lever_ok'] else " — too short, floor pinned at 0")
                  + f"; sigma {fit['sigma_median']:.5f} at the median count, "
                  f"{100 * fit['poisson_share']:.0f} % Poisson)")
            if self.log_fn is not None:
                self.log_fn({'wls/noise_fit_ok': 1,
                             'wls/var_per_count_raw': fa, 'wls/var_floor_raw': fb,
                             'wls/noise_fit_r2': fit['r2'],
                             'wls/noise_air_pixels': fit['n_air'],
                             'wls/noise_sigma_median': fit['sigma_median']}, 0)
        if a == 'auto':
            self.cfg['wls_var_per_count'] = float(fa)
        if b == 'auto':
            self.cfg['wls_var_floor'] = float(fb)

    def _group_views(self, group: int):
        """Indices of the views in acquisition group ``group``, or None when
        the scan has a single group (so callers can keep the whole set)."""
        g = getattr(self, 'view_groups', None)
        if g is None or len(np.unique(g)) < 2:
            return None
        return np.flatnonzero(np.asarray(g) == int(group))

    def _resolve_holdout(self, n_views: int) -> int:
        """The held-out view: the middle of the FIRST group, not of the file
        list — with two gated phases the list's middle is the first view of
        phase 1, the terminal angle of the short scan, the worst possible
        validation view."""
        sel = self._group_views(0)
        if self.holdout_index is None and sel is not None:
            return int(sel[len(sel) // 2])
        return resolve_holdout_index(self.holdout_index, n_views)

    def _roi_box_mm(self, margin_mm):
        """(lo, hi) of the export ROI + margin in mm, or None without a ROI."""
        if margin_mm is None:
            return None
        geom = export_grid_geometry(self.geometry)
        if geom is self.geometry:          # no export_roi: nothing to confine to
            return None
        shape = np.asarray(geom['vol_shape'], dtype=np.float64)
        pitch = np.array([geom['dx'], geom['dx'], geom['dz']], dtype=np.float64)
        centre = np.asarray(geom['vol_origin'], dtype=np.float64)
        half = shape * pitch / 2.0 + float(margin_mm)
        return centre - half, centre + half

    def _densify_bounds(self, margin_mm):
        """(lo, hi) of the export ROI + margin in normalised world units,
        or None when growth is unconfined (no margin given, or no ROI)."""
        box = self._roi_box_mm(margin_mm)
        if box is None:
            return None
        return box[0] * self.world_scale, box[1] * self.world_scale

    def _apply_density_floor(self, cloud) -> None:
        """Signed density: set the floor to ``signed_density`` x the cloud's
        median amplitude (see `GaussianCloud.__init__`), re-parameterising so
        no amplitude moves. Applied to the seed AND to a resumed checkpoint,
        so the run's own setting always wins over the checkpoint's; zero is
        the strictly positive cloud."""
        frac = float(self.cfg['signed_density'])
        with torch.no_grad():
            floor = frac * float(cloud.density.median()) if frac > 0 else 0.0
        if not math.isclose(float(cloud.density_floor), floor, rel_tol=1e-6,
                            abs_tol=1e-12):
            cloud.set_density_floor(floor)
        if floor > 0:
            print(f"    signed density: amplitudes may go down to -{frac:g} x "
                  f"the median ({-floor / self.signal_scale:.3g} mm^-1); air "
                  f"can be a noisy zero and edges can undershoot")
        if self.log_fn is not None:
            self.log_fn({'model/density_floor_mm': floor / self.signal_scale}, 0)

    def _resume(self, cloud, path: str) -> None:
        """Replace the seed with a saved cloud (`--gauss-checkpoint` output).

        The seed is still built first so that everything derived from the
        DATA — world scale, signal scale, the reference volume — is exactly
        what a fresh run would use; only the primitives are swapped. Both
        scales are checked rather than trusted: a checkpoint from a different
        downsample or signal-scale setting is in different units, and the
        densities would be silently wrong by that factor.
        """
        state = torch.load(path, map_location='cpu', weights_only=False)
        for key, mine in (('world_scale', self.world_scale),
                          ('signal_scale', self.signal_scale)):
            theirs = float(state.get(key, mine))
            if not math.isclose(theirs, mine, rel_tol=1e-4):
                raise ValueError(
                    f"resume: {key} of {path} is {theirs:.6g} but this run "
                    f"resolves {mine:.6g}; the checkpoint was made on other "
                    f"data or settings and its units do not transfer")
        cloud.restore(state['cloud'])
        print(f"    resumed {len(cloud):,} primitives from {path}"
              + (f" (delivered at iteration {state['delivered_iter']})"
                 if 'delivered_iter' in state else ""))

    def _seed(self, domain, rng, device) -> GaussianCloud:
        ref = self._reference_volume()
        if ref is None:
            parts = seeding.seed_uniform(
                domain=domain, world_scale=self.world_scale,
                n_points=self.cfg['n_seed'], rng=rng,
                scale_factor=self.cfg['seed_scale'])
        else:
            ref_vol, ref_geom = ref
            parts = seeding.seed_from_volume(
                ref_vol, geometry=ref_geom, domain=domain,
                world_scale=self.world_scale, n_points=self.cfg['n_seed'],
                noise_floor_quantile=self.cfg['seed_floor_quantile'],
                scale_factor=self.cfg['seed_scale'], rng=rng,
                roi_box_mm=self._roi_box_mm(1.0),
                roi_weight=self.cfg['seed_roi_weight'])
        xyz, scaling, rotation, density = parts
        # Report the width against the pitch it will be exported onto. A cloud
        # of width sigma can only represent `G_sigma * (non-negative measure)`,
        # so this ratio is the resolution ceiling AT THE START, and how far the
        # run gets below it is decided by --lr and --iterations (MEASURED on
        # ds6: 0.845 -> 0.627 mm over 5 k iterations at lr 1e-4, -> 0.488 mm at
        # 1e-3; see `seeding._spacing_per_point`). Neither seeding nor the run's
        # geometry knows the other, so the comparison can only be made here.
        sigma_mm = float(np.median(np.asarray(scaling))) / self.world_scale
        pitch = float(np.mean([self.geometry['dx'], self.geometry['dx'],
                               self.geometry['dz']]))
        print(f"    seed width {sigma_mm:.4f} mm = {sigma_mm / pitch:.1f}x the "
              f"{pitch:.4f} mm voxel  (FWHM {2.3548 * sigma_mm:.3f} mm; "
              f"MTF at 1 lp/mm = "
              f"{math.exp(-2 * math.pi ** 2 * sigma_mm ** 2):.3g})")
        # Density is seeded in mm^-1 and the kernel works in rendered units.
        return GaussianCloud(xyz, scaling, rotation,
                             density * self.signal_scale).to(device)

    #: Overrides for `estimate_noise_model` (thresholds); a class attribute
    #: so a harness with a tiny detector can lower `min_pixels`.
    noise_fit_kwargs: dict = {}

    #: Seeds do not need a fine reference. The seed only has to say WHERE mass
    #: is and in what proportion; a 0.15 mm map answers that as well as a
    #: 75 um one, and reconstructing the fine grid first would cost more than
    #: the run it is seeding (124 M voxels against 15 M on Scan_1510).
    SEED_VOXEL_FLOOR_MM = 0.15

    def _reference_volume(self):
        """A cheap volume to seed from — an FDK on this scan's own geometry.

        The submodule's FDK, not a second implementation, and reconstructed on
        the MODEL domain rather than the export ROI so the bed and the
        periphery get primitives. Returns None when seeding is set to
        'uniform' or the FDK is unavailable, which is a slower start and not an
        error.
        """
        if self.cfg['seed_from'] != 'fdk':
            return None
        try:
            from ...fdk.reconstructor import FDKReconstructor
        except ImportError as e:
            print(f"  seeding: FDK unavailable ({e}); using uniform")
            return None

        geom = self._seed_geometry()
        print(f"  seeding: FDK reference on the model domain, "
              f"{geom['vol_shape']} at {geom['dx']:.3f} mm")
        # Several acquisition groups (gated phases) repeat the same gantry
        # positions; an FDK over all of them would weight each angle twice
        # and average the states. The seed is the FIRST group alone.
        sel = self._group_views(0)
        proj = np.asarray(self.projections)
        angs = np.asarray(self.angles, dtype=np.float32)
        if sel is not None:
            print(f"  seeding: from acquisition group 0 only "
                  f"({len(sel)} of {len(angs)} views)")
            proj, angs = proj[sel], angs[sel]
        fdk = FDKReconstructor(
            torch.as_tensor(proj),
            torch.as_tensor(angs),
            geom, folder_name='gaussian-seed', quantitative=True,
            bright_field=self.bright_field, dark_field=self.dark_field,
            soft_clip_sharpness=self.soft_clip_sharpness,
            ring_correction=self.ring_correction,
            ring_median_width=self.ring_median_width,
            air_normalization=self.air_normalization)
        fdk.reconstruct()
        # Every backend in the pipeline leaves its result on this attribute
        # rather than returning it; the FDK is no exception.
        vol = fdk.reconstructed_volume
        if vol is None:
            print("  seeding: FDK produced no volume; using uniform")
            return None
        vol = vol.detach().cpu().numpy() if torch.is_tensor(vol) else np.asarray(vol)
        return np.ascontiguousarray(vol, dtype=np.float32), geom

    def _seed_geometry(self) -> dict:
        """The run's geometry, coarsened to the seeding floor.

        Same physical domain, fewer voxels: the extents are held fixed and the
        counts fall, so positions computed from this grid land in exactly the
        same millimetres as the run's own.
        """
        geom = dict(self.geometry)
        nx, ny, nz = (int(v) for v in geom['vol_shape'])
        dx, dz = float(geom['dx']), float(geom['dz'])
        fx = max(1, int(round(self.SEED_VOXEL_FLOOR_MM / dx)))
        fz = max(1, int(round(self.SEED_VOXEL_FLOOR_MM / dz)))
        if fx == 1 and fz == 1:
            return geom
        geom['vol_shape'] = (max(1, nx // fx), max(1, ny // fx),
                             max(1, nz // fz))
        geom['dx'] = dx * fx
        geom['dz'] = dz * fz
        return geom

    @torch.no_grad()
    def _export(self, cloud, domain) -> np.ndarray:
        """Query the cloud on the reconstruction grid, in mu mm^-1.

        Note this is the grid ``geometry['vol_shape']`` describes — the driver
        crops it to the export ROI afterwards, exactly as it does for every
        other backend, so the delivered volume and its slice indices line up
        with every other algorithm's.
        """
        nx, ny, nz = (int(v) for v in self.geometry['vol_shape'])
        print(f"  exporting {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.1f} M voxels "
              f"at {float(self.geometry['dx']):.4f} mm")
        return self._render_grid(cloud, self.geometry)

    def _render_grid(self, cloud, geom: dict) -> np.ndarray:
        """Query the cloud on the grid ``geom`` describes, in mu mm^-1.

        One place for the lattice arithmetic, shared by the delivered export
        (the whole reconstruction domain) and the mid-training stage views
        (the export ROI only). Under ``no_grad`` because the second caller
        runs while the parameters still require gradients, and a voxelize
        left on the autograd tape keeps its binning buffers alive.
        """
        nx, ny, nz = (int(v) for v in geom['vol_shape'])
        dx, dz = float(geom['dx']), float(geom['dz'])
        origin = geom.get('vol_origin', (0.0, 0.0, 0.0))
        extent_world = (nx * dx * self.world_scale, ny * dx * self.world_scale,
                        nz * dz * self.world_scale)
        center_world = tuple(float(o) * self.world_scale for o in origin)
        with torch.no_grad():
            vol = kernel.chunked_voxelize(
                cloud, n_voxel=(nx, ny, nz), extent_world=extent_world,
                center_world=center_world, max_voxels=self._export_chunk)
        return (vol.detach().cpu().numpy() / self.signal_scale).astype(
            np.float32)

    def _lr_stage(self, cloud, domain, device, it, stage, lr, slug=None):
        """The base hook, minus the part that does not fit a cloud.

        The base loop hands ``on_lr_stage`` the whole reconstruction domain
        and lets the driver crop it, which for a voxel grid is a memory copy.
        Here it would be a 993.6 M-voxel render — about a minute per frame on
        Scan_1510 ds3/75 um — so the export ROI is evaluated directly
        (`export_grid_geometry`, the same lattice arithmetic the crop uses,
        so the two agree voxel for voxel) and handed over WITH its geometry.
        MEASURED: 0.9 s for the 88 M-voxel ROI; the three midplanes alone
        would be ~20 ms, but the driver refits the HU map on every frame from
        the volume's own histogram, and three planes are a thinner histogram
        than the ROI for a saving that no longer matters at this cadence.

        Same contract as the base: a figure is not worth ending a run over.
        """
        if self._on_lr_stage is None:
            return
        try:
            geom = export_grid_geometry(self.geometry)
            self._on_lr_stage(self._render_grid(cloud, geom), int(it),
                              int(stage), float(lr), geometry=geom, slug=slug)
        except Exception as e:                  # never lose a run to a figure
            print(f"  lr_plateau: stage views failed "
                  f"({type(e).__name__}: {e})")
