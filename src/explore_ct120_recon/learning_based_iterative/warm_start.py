"""Warm start from an FDK volume: fit the representation to FDK, then hand
over to the projection-domain objective.

Opt-in (``run_learned_recon --warm-start fdk`` or ``--warm-start-volume``);
nothing here runs otherwise. Two phases, a clean reimplementation of muNeRF's
old ``train.py`` recipe (FDK pretrain -> outside-ROI handoff warmup -> main):

1. **Pretrain** — make the model reproduce the FDK volume. A dense voxel grid
   needs no fitting: its parameters ARE a volume, so the FDK is copied in
   (``LearnedReconstructor._warm_start_direct``). A network is fitted by POINT
   REGRESSION: random points in the FDK grid, target = the FDK value there
   (trilinear), loss = RELATIVE MSE inside the model domain (MSE divided by
   the target's mean square, so 1.0 = an error as large as the signal).
   Stops on a fixed set of
   evaluation points (volume-domain RMSE against the FDK) when the relative
   gain over ``patience`` evaluations falls below ``min_gain``; the best state
   is restored.
2. **Handoff** — a short projection-domain phase with the run's OWN data term
   and sampler (same weights, same excluded holdout), plus an anchor term
   pulling the model toward the FDK at random points, so the switch from a
   volume objective to a projection objective does not scramble what the
   pretrain built. The old recipe needed it because its domain extended past
   the FDK (bed, holder): the anchor held the inside while the projections
   trained the outside. Here the domain usually IS the FDK grid, so the phase
   is a damped transition; ``--warm-start-handoff-iters 0`` skips it.

THE FDK SEES ONLY TRAINING VIEWS. It is computed in-process from the views
the trainer gets (test views are cut from the data before any backend
exists, and the holdout is dropped too when ``--withhold-eval`` makes it a
validation view), or loaded from a file whose sidecar must PROVE the same:
its ``views`` record is checked against this run's withheld views and a
mismatch is refused (``check_volume_views``).

UNITS: the pipeline's own. The in-process FDK returns mu in mm^-1 (absolute
scale since 2026-10-26); a loaded volume is inverted through ITS sidecar's two
anchors, never a fixed mu_water. Air is levelled to mu = 0 with the anchors
the HU calibrator fits (``projection_diag.level_to_air``) — the forward model
assumes air integrates to nothing, and a -0.0008 mm^-1 air offset over a
60 mm chord is a 33 % error in a line integral. NOTHING IS CLIPPED: the FDK's
negative undershoot is kept as a target. (A softplus network cannot reach it
and a voxel grid projects mu >= 0 after its first optimizer step, as it always
does; those are properties of the representations, not of the target.)

WHY RELATIVE MSE, MEASURED: in mm^-1 the squared error is ~4e-5, and its
per-sample gradient (~1e-7 at 131072 points) underflows in tinycudann's
half-precision hash tables. The first Scan_1510 test of this module trained
nothing in 1500 pretrain steps (RMSE and NCC unchanged) while the same model
trained normally on the projection loss. Dividing by the target's mean square
makes the loss O(1) and leaves the minimiser unchanged; the anchor in the
handoff is scored the same way, so its weight means "relative error",
comparable to a reduced chi-square of O(1).

DEVIATIONS FROM THE OLD RECIPE, deliberately:
  * The pretrain stops on volume RMSE at fixed points, not on SSIM of one
    axial slice against the vendor volume.
  * No fixed mu_water conversion and no clamp to zero (the old
    ``load_fdk_mu`` + ``.clamp(min=0)``).
  * The FDK is this pipeline's FDK on the training views, not the vendor
    volume (which saw every view, test views included).
  * The handoff uses the run's registry data term (wls by default) rather
    than the old MSE, and the main batch sampler rather than a separate
    16384-ray one.
  * The pretrain LR is the main run's per-group LRs times
    ``lr_scale`` (default 1.0 = trunk 1e-3, hash tables 1e-2 for the hash
    MLP, which IS the old 1e-3 with the x10 encoder multiplier).

OPTIMISER STATE AT HANDOFF — an explicit choice. Each phase builds its OWN
optimizer, and the main loop starts with a FRESH one (empty Adam moments) and
its own LR warmup (``--lr-warmup-iters``) from zero, exactly like the old
recipe. Carrying the pretrain moments over would apply second-moment
estimates of a volume-domain objective to a projection-domain one; a warmup
from zero is what keeps the first projection-domain steps from undoing the
warm start.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .training import (autocast_ctx, build_optimizer, project_nonneg,
                       unwrap_model)
from .volume import query_mu

#: Old-recipe handoff stopping constants: check the windowed projection loss
#: every CHECK iterations, stop after PATIENCE checks without a REL_GAIN
#: relative improvement, never before MIN_ITERS.
HANDOFF_CHECK_EVERY = 150
HANDOFF_PATIENCE = 3
HANDOFF_REL_GAIN = 0.02
HANDOFF_MIN_ITERS = 600

#: Fixed points the pretrain is scored on. A deterministic functional of the
#: model, so its curve is smooth and the stop is reproducible.
PRETRAIN_EVAL_POINTS = 1 << 20


@dataclass
class WarmStart:
    """What the trainer needs to warm-start, and how.

    ``volume`` is (Nx, Ny, Nz) mu in mm^-1 on ``geometry``'s grid (the
    trainer's own export/parameter grid), already air-levelled. ``record``
    is its provenance, written into recon.json.
    """
    volume: np.ndarray
    geometry: dict
    record: dict = field(default_factory=dict)
    pretrain_iters: int = 100_000
    pretrain_batch: int = 131_072
    pretrain_lr_scale: float = 1.0
    pretrain_eval_every: int = 2000
    pretrain_patience: int = 3
    pretrain_min_gain: float = 0.01
    handoff_iters: int = 3000
    handoff_anchor_weight: float = 1.0
    handoff_anchor_points: int = 65_536

    def settings(self) -> dict:
        return {k: getattr(self, k) for k in (
            'pretrain_iters', 'pretrain_batch', 'pretrain_lr_scale',
            'pretrain_eval_every', 'pretrain_patience', 'pretrain_min_gain',
            'handoff_iters', 'handoff_anchor_weight', 'handoff_anchor_points')}


# --------------------------------------------------------------------------
# The FDK target
# --------------------------------------------------------------------------

def fdk_from_views(projections, angles, geometry, *, bright_field,
                   dark_field, air_normalization=True,
                   soft_clip_sharpness=200.0, ring_correction=False,
                   ring_median_width=51, filter_type='hamming',
                   filter_cutoff='match', parker_weighting=True):
    """FDK of exactly the views passed, on ``geometry``'s grid; mu, mm^-1.

    Same settings as ``run_fdk_recon``'s defaults (Hamming at the matched
    cutoff, Parker, physical normalisation), so the warm start is the volume
    ``ct120-fdk`` would deliver from these views.
    """
    from ..fdk.reconstructor import FDKReconstructor

    geom = dict(geometry)
    geom.setdefault('cor_offset_scale', 1.0)
    if str(filter_cutoff).lower() == 'match':
        cutoff = min(1.0, float(geom['da']) / float(geom['dx']))
    else:
        cutoff = float(filter_cutoff)
    rec = FDKReconstructor(
        projections=projections, angles=torch.as_tensor(np.asarray(angles)),
        geometry=geom, folder_name='', quantitative=True,
        bright_field=bright_field, dark_field=dark_field, clamp_mode='none',
        soft_clip_transmission=True, soft_clip_sharpness=soft_clip_sharpness,
        upper_clamp=True, upper_clamp_value=1.05, physical_normalization=True,
        filter_cutoff=cutoff, filter_type=filter_type,
        parker_weighting=parker_weighting, ring_correction=ring_correction,
        air_normalization=air_normalization,
        ring_median_width=ring_median_width)
    rec.reconstruct()
    vol = rec.reconstructed_volume
    vol = vol.detach().cpu().numpy() if torch.is_tensor(vol) else np.asarray(vol)
    del rec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.ascontiguousarray(vol, dtype=np.float32), {
        'filter_type': filter_type, 'filter_cutoff': float(cutoff),
        'parker_weighting': bool(parker_weighting)}


def level_air(volume, geometry, *, hu_calibration='auto', mu_water=None,
              tissue_hu=None):
    """``(volume - air, air)`` with ``air`` fitted on the export ROI.

    The anchors are read from the voxels the run will deliver (the export
    crop), as ``save_outputs`` reads them, then the offset is removed from the
    whole grid.
    """
    from ..ct_core.hu_calibration import resolve_anchors
    from ..ct_core.projection_diag import level_to_air
    from ..ct_core.support import crop_to_export_roi

    roi, _ = crop_to_export_roi(volume, dict(geometry))
    anchors = resolve_anchors(roi, hu_calibration, mu_water=mu_water,
                              tissue_hu=tissue_hu, verbose=False)
    levelled, air = level_to_air(volume, anchors)
    return np.ascontiguousarray(levelled, dtype=np.float32), float(air)


def resample_to_grid(volume, src: dict, dst: dict) -> np.ndarray:
    """Trilinear resample of an (Nx, Ny, Nz) volume from grid ``src`` onto
    grid ``dst`` (both: vol_shape, vol_origin, dx, dz); zero outside ``src``.
    Identity (returned as-is) when the grids are the same."""
    same = (tuple(int(v) for v in src['vol_shape'])
            == tuple(int(v) for v in dst['vol_shape'])
            and np.allclose(src['vol_origin'], dst['vol_origin'], atol=1e-6)
            and abs(float(src['dx']) - float(dst['dx'])) < 1e-9
            and abs(float(src['dz']) - float(dst['dz'])) < 1e-9)
    if same:
        return np.ascontiguousarray(volume, dtype=np.float32)
    from .volume import grid_coords
    vol = torch.from_numpy(np.ascontiguousarray(
        np.asarray(volume, dtype=np.float32).transpose(2, 1, 0)))[None, None]
    x, y, z = grid_coords(dst)
    sNx, sNy, sNz = (int(v) for v in src['vol_shape'])
    so = [float(v) for v in src['vol_origin']]
    hx, hz = sNx * float(src['dx']) / 2, sNz * float(src['dz']) / 2
    hy = sNy * float(src['dx']) / 2
    out = np.empty(tuple(int(v) for v in dst['vol_shape']), dtype=np.float32)
    gy, gx = torch.meshgrid((y - so[1]) / hy, (x - so[0]) / hx, indexing='ij')
    for k in range(z.numel()):
        gz = torch.full_like(gx, float((z[k] - so[2]) / hz))
        grid = torch.stack([gx, gy, gz], dim=-1)[None, None]   # (1,1,Ny,Nx,3)
        s = F.grid_sample(vol, grid, mode='bilinear', align_corners=False,
                          padding_mode='zeros')[0, 0, 0]       # (Ny, Nx)
        out[:, :, k] = s.numpy().T
    return out


def check_volume_views(sidecar: dict, *, forbidden, n_views: int,
                       name: str) -> dict:
    """Refuse a warm-start volume that may have seen a withheld view.

    ``forbidden`` are ORIGINAL indices this run must not learn from (its test
    views, plus the holdout when it is withheld). The sidecar's ``views``
    record (written by every driver when test views or --withhold-eval are
    active) says which views built the volume. Returns a provenance dict;
    raises ValueError when the volume used a forbidden view or cannot show
    that it did not.
    """
    forbidden = sorted({int(i) for i in forbidden})
    views = (sidecar or {}).get('views') or None
    if views is None:
        if forbidden:
            raise ValueError(
                f"--warm-start-volume {name}: its sidecar has no 'views' "
                f"record, so nothing shows it was reconstructed without this "
                f"run's withheld views {forbidden}. Rebuild it with the same "
                f"--test-indices/--test-count (and --withhold-eval if this run "
                f"withholds its holdout), or use --warm-start fdk to compute "
                f"it in-process from the training views.")
        print(f"  WARNING: --warm-start-volume {name} has no 'views' record; "
              f"accepted only because this run withholds no view.")
        return {'views_verified': False}
    if int(views.get('n_views', -1)) != int(n_views):
        raise ValueError(
            f"--warm-start-volume {name} was reconstructed from a list of "
            f"{views.get('n_views')} views; this run loaded {n_views}. The "
            f"indices cannot be compared.")
    used = set(int(i) for i in views.get('train_indices', []))
    if views.get('holdout_withheld') and views.get('holdout_index') is not None:
        used.discard(int(views['holdout_index']))
    leak = sorted(used & set(forbidden))
    if leak:
        raise ValueError(
            f"--warm-start-volume {name} was reconstructed FROM views {leak}, "
            f"which this run withholds. Warm-starting from it would leak them "
            f"into the model.")
    return {'views_verified': True,
            'volume_views': {k: views.get(k) for k in
                             ('test_indices', 'holdout_index',
                              'holdout_withheld', 'test_mode')}}


def load_warm_start_volume(path, geometry, *, forbidden, n_views,
                           hu_calibration='auto'):
    """A reconstructed volume on disk -> (mu on ``geometry``'s grid, record).

    The HU map is inverted with the file's OWN sidecar anchors; a file with
    no sidecar anchors (the vendor's) is refused, since the only alternative
    is a fixed mu_water.
    """
    from pathlib import Path
    from ..ct_core.hu_calibration import HUAnchors
    from ..ct_core.volume_report import (load_reconstructed_volume,
                                         load_sidecar, resolve_volume_geometry)

    path = Path(path)
    if not path.exists():
        raise ValueError(f"--warm-start-volume: {path} does not exist")
    sidecar = load_sidecar(path, verbose=False)
    record = check_volume_views(sidecar, forbidden=forbidden,
                                n_views=n_views, name=path.name)
    hu = (sidecar or {}).get('hu_calibration') or {}
    if not {'anchor_air_value', 'anchor_tissue_value'} <= set(hu):
        raise ValueError(
            f"--warm-start-volume {path.name}: no two-anchor HU calibration "
            f"in its sidecar, so its mu scale is unknown. Only volumes this "
            f"package wrote can be used.")
    anchors = HUAnchors(value_air=float(hu['anchor_air_value']),
                        value_tissue=float(hu['anchor_tissue_value']),
                        air_hu=float(hu.get('target_air', -1000.0)),
                        tissue_hu=float(hu.get('target_tissue', 120.0)))
    volume_hu, header = load_reconstructed_volume(path, verbose=False)
    src = resolve_volume_geometry(volume_hu, header, sidecar=sidecar,
                                  verbose=False)
    mu = anchors.invert(volume_hu)
    mu = resample_to_grid(mu, src, geometry)
    record.update({'source': f'file:{path.name}',
                   'source_algorithm': sidecar.get('algorithm', 'unknown'),
                   'resampled': not np.array_equal(
                       np.asarray(src['vol_shape']),
                       np.asarray(geometry['vol_shape']))})
    return mu, record


# --------------------------------------------------------------------------
# Phase 1: pretrain a network on the FDK volume
# --------------------------------------------------------------------------

class _VolumeTarget:
    """The FDK as a sampleable field: world mm -> mu (trilinear, zero outside
    its grid), plus the domain normalisation the model expects."""

    def __init__(self, volume, geometry, domain, device):
        self.device = device
        self.vol = torch.from_numpy(np.ascontiguousarray(
            np.asarray(volume, dtype=np.float32).transpose(2, 1, 0))
        ).to(device)[None, None]                               # (1,1,Nz,Ny,Nx)
        Nx, Ny, Nz = (int(v) for v in geometry['vol_shape'])
        o = torch.tensor([float(v) for v in geometry['vol_origin']],
                         device=device)
        half = torch.tensor([Nx * float(geometry['dx']) / 2,
                             Ny * float(geometry['dx']) / 2,
                             Nz * float(geometry['dz']) / 2], device=device)
        self.origin, self.half = o, half
        self.domain = domain

    def sample_points(self, n, gen):
        u = torch.rand((int(n), 3), generator=gen, device=self.device) * 2 - 1
        return self.origin + u * self.half                     # mm

    def value(self, pts_mm):
        g = ((pts_mm - self.origin) / self.half).reshape(1, -1, 1, 1, 3)
        return F.grid_sample(self.vol, g, mode='bilinear', align_corners=False,
                             padding_mode='zeros').reshape(-1)

    def weight(self, pts_mm):
        return self.domain.cylinder_mask(pts_mm)

    def mean_square(self, gen, n=PRETRAIN_EVAL_POINTS) -> float:
        """The target's mean square inside the domain: the scale the
        relative MSE divides by (see the module docstring)."""
        pts = self.sample_points(n, gen)
        w = self.weight(pts) > 0
        v = self.value(pts)[w].double()
        return max(float(torch.mean(v ** 2)), 1e-30)

    def rel_mse(self, model, pts_mm, scale):
        """Relative MSE of the model against the target at ``pts_mm``."""
        w = self.weight(pts_mm)
        err = self.model_mu(model, pts_mm).float() - self.value(pts_mm)
        return torch.sum(w * err ** 2) / torch.clamp(w.sum(), min=1.0) / scale

    def model_mu(self, model, pts_mm):
        return query_mu(model, self.domain.normalize(pts_mm))


def _volume_scores(model, target: _VolumeTarget, pts, chunk=1 << 18):
    """RMSE / relative RMSE / NCC of the model against the FDK at ``pts``."""
    preds, tgts, ws = [], [], []
    m = unwrap_model(model)
    was = m.training
    m.eval()
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            p = pts[i:i + chunk]
            preds.append(target.model_mu(model, p).float())
            tgts.append(target.value(p))
            ws.append(target.weight(p))
    if was:
        m.train()
    p, t, w = torch.cat(preds), torch.cat(tgts), torch.cat(ws) > 0
    p, t = p[w].double(), t[w].double()
    err = p - t
    rmse = float(torch.sqrt(torch.mean(err ** 2)))
    rms_t = float(torch.sqrt(torch.mean(t ** 2))) or 1.0
    pc, tc = p - p.mean(), t - t.mean()
    ncc = float((pc * tc).sum() / (torch.sqrt((pc ** 2).sum() * (tc ** 2).sum())
                                   + 1e-30))
    return {'rmse': rmse, 'rel_rmse': rmse / rms_t, 'ncc': ncc}


def pretrain_on_volume(model, ws: WarmStart, domain, device, *, lr,
                       group_lrs, seed, amp_dtype=None, log_fn=None) -> dict:
    """Point regression of ``model`` onto ``ws.volume``. Returns the report."""
    target = _VolumeTarget(ws.volume, ws.geometry, domain, device)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 31337)
    eval_gen = torch.Generator(device=device).manual_seed(int(seed) + 4242)
    eval_pts = target.sample_points(PRETRAIN_EVAL_POINTS, eval_gen)
    ms = target.mean_square(
        torch.Generator(device=device).manual_seed(int(seed) + 1618))
    scale = float(ws.pretrain_lr_scale)
    opt = build_optimizer(model, lr=float(lr) * scale,
                          group_lrs={k: float(v) * scale
                                     for k, v in (group_lrs or {}).items()},
                          optimizer='adam', verbose=False)
    lrs = ', '.join(f"{g['lr']:.1e}" for g in opt.param_groups)
    print(f"  warm start / pretrain: point regression onto the FDK, up to "
          f"{ws.pretrain_iters} iters x {ws.pretrain_batch} points, Adam lr "
          f"[{lrs}]; scored every {ws.pretrain_eval_every} iters on "
          f"{PRETRAIN_EVAL_POINTS} fixed points, stop after "
          f"{ws.pretrain_patience} evals with < {100 * ws.pretrain_min_gain:g} "
          f"% RMSE gain")
    s0 = _volume_scores(model, target, eval_pts)
    print(f"    target mean square {ms:.3e} (mm^-1)^2 — the relative-MSE scale")
    print(f"    iter      0  rmse {s0['rmse']:.3e} mm^-1  rel "
          f"{s0['rel_rmse']:.3f}  ncc {s0['ncc']:.4f}")
    best, best_it, bad = s0['rmse'], 0, 0
    best_state = {k: v.detach().clone()
                  for k, v in unwrap_model(model).state_dict().items()}
    t0 = time.time()
    it = 0
    for it in range(1, int(ws.pretrain_iters) + 1):
        pts = target.sample_points(ws.pretrain_batch, gen)
        with autocast_ctx(amp_dtype, device.type):
            loss = target.rel_mse(model, pts, ms)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        project_nonneg(model)
        if it % int(ws.pretrain_eval_every) == 0 or it == int(ws.pretrain_iters):
            s = _volume_scores(model, target, eval_pts)
            print(f"    iter {it:6d}  rmse {s['rmse']:.3e}  rel "
                  f"{s['rel_rmse']:.3f}  ncc {s['ncc']:.4f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
            if log_fn is not None:
                log_fn({'warm_start/pretrain_rmse': s['rmse'],
                        'warm_start/pretrain_rel_rmse': s['rel_rmse'],
                        'warm_start/pretrain_ncc': s['ncc']}, 0)
            if s['rmse'] < best * (1.0 - float(ws.pretrain_min_gain)):
                best, best_it, bad = s['rmse'], it, 0
                best_state = {k: v.detach().clone() for k, v in
                              unwrap_model(model).state_dict().items()}
            else:
                bad += 1
                if bad >= int(ws.pretrain_patience):
                    break
    unwrap_model(model).load_state_dict(best_state)
    del best_state, opt
    final = _volume_scores(model, target, eval_pts)
    print(f"  warm start / pretrain: stopped at {it}, restored iter {best_it}: "
          f"rmse {final['rmse']:.3e} mm^-1 (rel {final['rel_rmse']:.3f}, ncc "
          f"{final['ncc']:.4f}) against the FDK, {time.time() - t0:.0f} s")
    return {'mode': 'point_regression', 'loss': 'relative_mse',
            'target_mean_square': ms, 'iters_run': int(it),
            'best_iter': int(best_it), 'initial': s0, 'final': final,
            'seconds': round(time.time() - t0, 1)}


def direct_scores(model, ws: WarmStart, domain, device, seed) -> dict:
    """The same volume-domain score after a direct copy (voxel grid)."""
    target = _VolumeTarget(ws.volume, ws.geometry, domain, device)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 4242)
    return _volume_scores(model, target,
                          target.sample_points(PRETRAIN_EVAL_POINTS, gen))


# --------------------------------------------------------------------------
# Phase 2: projection-domain handoff with an FDK anchor
# --------------------------------------------------------------------------

def handoff(model, ws: WarmStart, domain, device, *, loss_fn, sample_batch,
            render_fn, scene, spp, exclude_angle, lr, group_lrs,
            optimizer_name, seed, amp_dtype=None, grad_scale_fn=None,
            grad_clip=None, log_fn=None) -> dict:
    """``handoff_iters`` steps of data term + anchor; returns the report."""
    if int(ws.handoff_iters) <= 0:
        return {'iters_run': 0, 'skipped': True}
    target = _VolumeTarget(ws.volume, ws.geometry, domain, device)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 271828)
    ms = target.mean_square(
        torch.Generator(device=device).manual_seed(int(seed) + 1618))
    opt = build_optimizer(model, lr=float(lr), group_lrs=group_lrs,
                          optimizer=optimizer_name, verbose=False)
    print(f"  warm start / handoff: up to {ws.handoff_iters} iters of the "
          f"run's data term + {ws.handoff_anchor_weight:g} x FDK anchor "
          f"(relative MSE) at "
          f"{ws.handoff_anchor_points} random points (fresh optimizer; stop "
          f"when the windowed data loss gains < {100 * HANDOFF_REL_GAIN:g} % "
          f"over {HANDOFF_PATIENCE} checks of {HANDOFF_CHECK_EVERY}, not "
          f"before {HANDOFF_MIN_ITERS})")
    window, best, bad = [], None, 0
    t0 = time.time()
    it = 0
    for it in range(1, int(ws.handoff_iters) + 1):
        o, d, tgt = sample_batch(exclude_angle=exclude_angle)
        pts = target.sample_points(ws.handoff_anchor_points, gen)
        with autocast_ctx(amp_dtype, device.type):
            pred = render_fn(o, d, model, scene, num_samples=spp,
                             stratified=True, generator=gen,
                             grad_scale_fn=grad_scale_fn)
            data = loss_fn(pred.reshape(tgt.shape), tgt, it)
            anc = target.rel_mse(model, pts, ms)
            loss = data + float(ws.handoff_anchor_weight) * anc
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            grad_clip(model)
        opt.step()
        project_nonneg(model)
        window.append(float(data.detach()))
        if it % HANDOFF_CHECK_EVERY == 0:
            cur = float(np.mean(window[-HANDOFF_CHECK_EVERY:]))
            print(f"    handoff iter {it:5d}  data {cur:.4e}  anchor (rel "
                  f"mse) {float(anc.detach()):.3e}", flush=True)
            if log_fn is not None:
                log_fn({'warm_start/handoff_data_loss': cur,
                        'warm_start/handoff_anchor': float(anc.detach())}, 0)
            if best is None or cur < best * (1.0 - HANDOFF_REL_GAIN):
                best, bad = cur, 0
            else:
                bad += 1
                if bad >= HANDOFF_PATIENCE and it >= HANDOFF_MIN_ITERS:
                    break
    del opt
    print(f"  warm start / handoff: {it} iters, {time.time() - t0:.0f} s")
    return {'iters_run': int(it), 'skipped': False,
            'final_window_data_loss': (float(np.mean(window[-HANDOFF_CHECK_EVERY:]))
                                       if window else None),
            'seconds': round(time.time() - t0, 1)}
