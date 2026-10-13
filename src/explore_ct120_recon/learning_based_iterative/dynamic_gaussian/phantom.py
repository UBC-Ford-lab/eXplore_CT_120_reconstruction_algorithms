"""A breathing digital phantom with a known ventilation, and the scorer.

WHY. Every number the dynamic model has been judged on so far is a comparison
with another ESTIMATE — the vendor's density reading, an FDK, a registration.
A phantom is the one object whose motion, whose air change and whose two
states are known exactly, because they are constructed here; a reconstruction
run on its simulated scan (with no flag saying it is synthetic) is then
scored against the truth, not against another method's error.

THE OBJECT. A rest volume (any ``.vff`` this pipeline wrote — a static
reconstruction of the animal is the realistic choice, so the anatomy, the
noise texture and the bed are real) is taken as the EXPIRATION state. The
lungs are found in it the way the ventilation report finds them
(``ventilation.lung_mask``) and blurred into a soft indicator ``s(x)``. The
displacement to inspiration is the gradient of a potential whose Laplacian
is the lung's volume source and nothing else:

    div u = 3 a s(x),   u = grad phi,   Laplacian phi = 3 a s,   a = J_lung^(1/3) - 1

Inside the lung the divergence is 3a, i.e. a volume ratio of ``J_lung`` (a
sphere expands homogeneously about its centre, ``u = a (x - c)``); outside
it is ZERO, so the chest wall, heart and liver are pushed aside without
changing volume, the way an incompressible medium moves around an expanding
inclusion (the classic 1/r^2 radial field for a sphere). The pleural surface
itself moves outward. This replaces a first version, ``u = a s (x - c)``,
which pinned the lung surface (``s -> 0`` there) and so squeezed the outer
shell of the lung to J 0.7 while the core expanded — a compression no lung
has (MEASURED on the first Scan_1510 phantom, 2026-09-22). The Poisson
problem is solved on the field grid with zero potential on the box faces
(a discrete sine transform), which lets the displaced volume leave the box.
The truth ``J`` is the determinant of ``I + grad u`` evaluated numerically on
the field's own grid, so whatever the construction, the recorded truth is
what the field really does (the finite-strain ``J`` of the exterior is
``1 - O(|grad u|^2)``, a few tenths of a percent next to the surface).

PATTERNS. A uniform ``J_lung`` tests the amount and the place of the
ventilation, not whether one region can be told from another: the first
Scan_1510 phantom (2026-09-22) came back as the truth smoothed by ~0.9 mm
with its integral kept, and its block correlation measured only the shape of
the boundary ramp. ``pattern='lobes'`` gives the two lungs different ratios
(``J_lung`` on the side of smaller x, ``J2`` on the other); ``pattern='defect'``
puts an unventilated sphere (``J = 1``, radius ``defect_mm``) at the centre of
the larger-x lung. The source becomes ``3 a(x) s(x)`` with ``a`` set per
region before the blur, so the ramp is unchanged; ``truth/regions.nrrd``
labels the regions (1, 2) and the score reports the recovered contrast
between them — the one number a regional claim rests on.

``pattern='texture'`` is the RESOLUTION test. The uniform and lobar truths
are smooth below a millimetre by construction, so against them the
correlation of the recovered J with the truth does not depend on the scale
it is read at (MEASURED 2026-09-25: block r flat from 0.3 to 1.5 mm on both),
while against the vendor densities of the real animal it climbs steadily
(0.42 -> 0.90 over 0.3-3 mm, still 0.56 -> 0.94 after correcting the
reference's own reading noise). A real lung ventilates unevenly below the
scale the field resolves; the phantom had nothing there to lose. The
texture pattern multiplies the lung's source by ``1 + amp * n(x)`` where
``n`` is a unit-variance random field made of ``octaves`` Gaussian-smoothed
white-noise fields at correlation lengths ``texture_mm, 2 texture_mm, ...``
with equal variance each (broadband, so ONE run measures the transfer at
every scale at once). The texture multiplies the source AFTER the boundary
ramp so the ramp is unchanged; the field grid must resolve the finest
scale (``pitch <= texture_mm / 2``). The scorer's ``scale`` block then
reports the block correlation as a function of block size and the
coherence and gain between the recovered and the true J per spatial
frequency — the ventilation transfer function of the whole pipeline — and
the finest scale at which the gain still reaches one half (the map's
MTF-50; a pure blur keeps the coherence at one and only lowers the gain).

MASS. The inspiration state is the PUSHFORWARD of the rest attenuation:
``mu_t(y) = mu_0(x) / J_t(x)`` with ``y = x + t u(x)``. Tissue mass moves and
spreads, air (zero attenuation) fills the room — the same physics the
mass-preserving model and the density-ventilation reading assume, so the
truth is consistent with BOTH readings the pipeline makes: the field's J and
the density ratio ``mu_exp / mu_insp`` agree with ``J_lung`` by construction.

THE SCAN. Each view is rendered at its own breathing state ``t`` in [0, 1]
(1 = inspiration). A gated acquisition writes phase 00 at ``t`` near 1 and
phase 01 near 0 with a per-view jitter (imperfect gating); ``t`` is
quantised to a few levels so the states are rendered once and cached.
The scanner half is ``explore_ct120_recon.simulate`` (the package's own ray
geometry and measured noise model); this module owns only the object.

THE SCORE. ``score`` takes a finished run directory and the phantom's scan
folder and compares the run's ventilation map (``recon_ventilation_J.nrrd``,
expiration frame, its own coarse grid) with the truth J resampled onto that
grid: bias, rms, voxel and block correlations over the lung core, the
boundary ramp by shells of ``s``, the lung's volume change, the smoothing
that best turns the truth into the map (a Gaussian sigma and a scale), and
the run's solid-tissue strain (truth 0). The exported inspiration volume is
scored against the truth state in HU windows. Written next to the run as
``phantom_score.json`` / ``.png``.

The truth the map is held to is the GATE-MEAN one. A gated reconstruction
fits one state per gate to views whose breathing level is jittered, so what
it can measure is the volume ratio between the mean inspiratory and the mean
expiratory state, ``J(t_0) / J(t_1)`` with ``t_0``, ``t_1`` the schedule's
gate means (0.92 / 0.08 at jitter 0.1) — not the peak-to-peak ``J(1)``. The
first Scan_1510 phantom scored against ``J(1)`` read 25 % low in the core
and the displacement 0.8x, across every grid and prior tried (2026-09-22,
five runs); against the gate-mean truth the same runs are within 2 %. The
peak-state numbers stay in the record as ``*_peak``. A run that carries the
per-view breathing levels (``ventilation.breathing_levels``) also has a
PEAK reading, scored against the peak truth as ``peak``, and its tidal
volumes (lung parcels, the dilated band, both states) are scored against
the truth's total under ``tidal_volume_mm3``.

CONVENTIONS. Phase 00 is inspiration (gate 0), phase 01 expiration (gate 1),
as on Scan_1510 — the ventilation report picks the reference by air content
and does not depend on this. Grids and files are the pipeline's (voxeliser
layout ``(nx, ny, nz)``, ``.vff`` + ``.json`` sidecar, ``write_nrrd_3d``).
Nothing here holds a path to anything outside the folders it is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import fft as sfft, ndimage

from ...ct_core.hu_calibration import HUAnchors
from ...ct_core.scan_setup import write_sidecar
from ...ct_core.vff_io import write_vff
from ...ct_core.volume_report import load_reconstructed_volume, load_sidecar
from ...simulate import VolumeSampler, simulate_scan
from ...simulate.forward import sampler_from_volume_file

from . import ventilation as vent
from .export4d import LinearHU, coarse_geometry
from .ventilation import read_nrrd_3d, write_nrrd_3d

TRUTH_DIR = 'truth'
J_FLOOR = 0.2                     # the model's own det floor; a phantom never goes near it


# ------------------------------------------------------------- the object --
def grid_centres(geom: dict) -> np.ndarray:
    """(nx, ny, nz, 3) voxel-centre positions in mm of a ct_core grid."""
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    ox, oy, oz = (float(v) for v in geom['vol_origin'])
    dx, dz = float(geom['dx']), float(geom['dz'])
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx + ox
    y = (np.arange(ny) - (ny - 1) / 2.0) * dx + oy
    z = (np.arange(nz) - (nz - 1) / 2.0) * dz + oz
    return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).astype(np.float32)


def _to_grid_sample(vol_xyz: torch.Tensor) -> torch.Tensor:
    """(nx, ny, nz[, C]) -> (1, C, nz, ny, nx) for ``grid_sample``."""
    if vol_xyz.ndim == 3:
        vol_xyz = vol_xyz.unsqueeze(-1)
    return vol_xyz.permute(3, 2, 1, 0).contiguous().unsqueeze(0)


def jacobian_of(u: np.ndarray, pitch: float, t: float) -> np.ndarray:
    """det(I + t grad u) of a displacement field (..., 3) on its grid."""
    grads = [[np.gradient(u[..., i], pitch, axis=j) for j in range(3)] for i in range(3)]
    Fm = np.empty(u.shape[:3] + (3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            Fm[..., i, j] = (1.0 if i == j else 0.0) + float(t) * grads[i][j]
    return np.linalg.det(Fm).astype(np.float32)


def gate_mean_truth(truth_dir) -> tuple:
    """The truth a gated reconstruction can be held to: ``J(t_0) / J(t_1)``
    on the field grid with ``t_0``, ``t_1`` the schedule's gate means, and
    the two means. ``None`` when the folder has no schedule / displacement."""
    truth_dir = Path(truth_dir)
    if not ((truth_dir / 'schedule.json').exists() and (truth_dir / 'displacement.npz').exists()):
        return None, None, None
    sched = np.asarray(json.load(open(truth_dir / 'schedule.json'))['t'], dtype=np.float64)
    d = np.load(truth_dir / 'displacement.npz')
    u, pitch = d['u_mm'], float(d['pitch_mm'])
    t0, t1 = float(sched[0].mean()), float(sched[1].mean())
    return jacobian_of(u, pitch, t0) / jacobian_of(u, pitch, t1), t0, t1


def smoothing_fit(J_rec, J_true_c, geom_t, X, m, pitch_t,
                  sigmas=(0.0, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0)) -> dict:
    """The Gaussian sigma (mm) and scale k with ``J_rec - 1 ~ k G_sigma (J_true - 1)``
    closest in rms over ``m``: the map's effective resolution and gain."""
    y = J_rec[m] - 1.0
    best = None
    for sig in sigmas:
        Jb = ndimage.gaussian_filter(J_true_c - 1.0, sig / pitch_t) if sig > 0 else J_true_c - 1.0
        xb = vent.sample(Jb, geom_t, X).reshape(J_rec.shape)[m]
        den = float((xb * xb).sum())
        if den <= 0:
            continue
        k = float((xb * y).sum() / den)
        rms = float(np.sqrt(((y - k * xb) ** 2).mean()))
        if best is None or rms < best['rms']:
            best = {'sigma_mm': float(sig), 'k': k, 'rms': rms}
    return best or {}


def potential_field(source: np.ndarray, pitch: float) -> np.ndarray:
    """``u = grad phi`` with ``Laplacian phi = source`` on the grid, zero
    potential on the box faces (DST-I), so ``div u = source`` to second order
    and the flow is irrotational: an incompressible medium displaced by a
    distributed volume source. Returns ``(nx, ny, nz, 3)`` in the source's
    length units (``pitch`` per voxel)."""
    f = np.asarray(source, dtype=np.float64)
    h = float(pitch)
    fh = sfft.dstn(f, type=1, norm='ortho')
    lam = 0.0
    for axis, n in enumerate(f.shape):
        k = np.arange(1, n + 1, dtype=np.float64)
        l1 = (2.0 - 2.0 * np.cos(np.pi * k / (n + 1))) / h ** 2
        lam = lam + l1.reshape([-1 if i == axis else 1 for i in range(f.ndim)])
    phi = sfft.idstn(-fh / lam, type=1, norm='ortho')
    return np.stack(np.gradient(phi, h), axis=-1).astype(np.float32)


class _Field:
    """A (C-channel) field on a ct_core grid, trilinearly sampled at mm points."""

    def __init__(self, values_xyz, geom: dict, device):
        v = torch.as_tensor(np.asarray(values_xyz, dtype=np.float32))
        self.data = _to_grid_sample(v).to(device)
        nx, ny, nz = (int(s) for s in geom['vol_shape'])
        dx, dz = float(geom['dx']), float(geom['dz'])
        ox, oy, oz = (float(s) for s in geom['vol_origin'])
        self.lo = torch.tensor([ox - nx * dx / 2, oy - ny * dx / 2, oz - nz * dz / 2], device=device)
        self.hi = torch.tensor([ox + nx * dx / 2, oy + ny * dx / 2, oz + nz * dz / 2], device=device)

    def __call__(self, xyz_mm: torch.Tensor, padding='zeros') -> torch.Tensor:
        g = (xyz_mm - self.lo) / (self.hi - self.lo) * 2.0 - 1.0
        out = F.grid_sample(self.data, g.reshape(1, -1, 1, 1, 3), mode='bilinear',
                            padding_mode=padding, align_corners=False)
        return out.reshape(self.data.shape[1], -1).T          # (M, C)


def texture_field(shape, sigma_vox: float, octaves: int, seed: int, mask=None) -> np.ndarray:
    """A unit-variance random field: ``octaves`` Gaussian-smoothed white-noise
    fields at correlation lengths ``sigma_vox, 2 sigma_vox, ...`` (the sigma
    of the smoothing kernel, in voxels), each normalised to unit variance
    and summed with equal weight, then normalised again over ``mask`` (the
    whole grid when None). Broadband by construction: equal variance per
    octave, so a transfer function can be read at every scale from one
    field."""
    rng = np.random.default_rng(int(seed))
    out = np.zeros(shape, np.float64)
    for k in range(int(octaves)):
        f = ndimage.gaussian_filter(rng.standard_normal(shape), float(sigma_vox) * 2 ** k)
        out += f / f.std()
    where = np.ones(shape, bool) if mask is None else np.asarray(mask, bool)
    out = (out - out[where].mean()) / out[where].std()
    return out.astype(np.float32)


PATTERNS = ('uniform', 'lobes', 'defect', 'texture')


class BreathingPhantom:
    """The rest volume, the field, the truth J and the states (see module doc).

    ``rest`` is a ``VolumeSampler`` in mu; ``hu`` the ``LinearHU`` of the file
    it came from (the lungs are found in HU). The field lives on a coarse
    grid at ``pitch_mm`` — it is smooth by construction — and is sampled
    trilinearly wherever the full-resolution states need it.
    """

    def __init__(self, rest: VolumeSampler, hu: LinearHU, *, pitch_mm: float = 0.3,
                 J_lung: float = 1.25, blur_mm: float = 0.6,
                 pattern: str = 'uniform', J2: float = None, defect_mm: float = 1.5,
                 texture_mm: float = 0.3, texture_amp: float = 0.3, texture_octaves: int = 4,
                 texture_seed: int = 0,
                 min_lung_mm3: float = vent.MIN_LUNG_MM3, log=print):
        self.rest, self.hu = rest, hu
        self.geom = dict(rest.geom)
        self.device = rest.device
        self.pitch = float(pitch_mm)
        self.J_lung, self.blur_mm = float(J_lung), float(blur_mm)
        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}, not {pattern!r}")
        self.pattern = pattern
        self.J2 = float(J2) if J2 is not None else (1.0 if pattern == 'defect' else self.J_lung)
        self.defect_mm = float(defect_mm)
        self.texture_mm, self.texture_amp = float(texture_mm), float(texture_amp)
        self.texture_octaves, self.texture_seed = int(texture_octaves), int(texture_seed)
        if pattern == 'texture':
            if self.texture_mm < 2.0 * float(pitch_mm):
                raise ValueError(f"texture_mm {self.texture_mm:g} needs pitch_mm <= {self.texture_mm / 2:g} to be resolved")
            if not (0.0 <= self.texture_amp < 1.0):
                raise ValueError("texture_amp must be in [0, 1)")
            if self.texture_octaves < 1:
                raise ValueError("texture_octaves must be >= 1")
        for name, J in (('J_lung', self.J_lung), ('J2', self.J2)):
            if J <= J_FLOOR:
                raise ValueError(f"{name} must exceed {J_FLOOR}")
        self.geom_c = coarse_geometry(self.geom, self.pitch)
        shape_c = tuple(int(v) for v in self.geom_c['vol_shape'])
        # the rest state area-averaged onto the coarse grid (same centre and,
        # to within a coarse voxel, the same extent)
        mu_c = F.interpolate(rest.data, size=shape_c[::-1], mode='area')[0, 0]
        mu_c = mu_c.permute(2, 1, 0).cpu().numpy()
        self.hu_c = hu.apply(mu_c)
        mask = vent.lung_mask(self.hu_c, self.pitch, min_mm3=min_lung_mm3)
        if not mask.any():
            raise ValueError("no lung found in the rest volume (ventilation.lung_mask is empty)")
        self.mask = mask
        sigma = self.blur_mm / self.pitch
        self.s = ndimage.gaussian_filter(mask.astype(np.float32), sigma) if sigma > 0 else mask.astype(np.float32)
        X = grid_centres(self.geom_c)
        self.centroid = X[mask].mean(axis=0).astype(np.float64)
        a = self.J_lung ** (1.0 / 3.0) - 1.0
        self.a = a
        self.regions, a_map, self.defect_centre = self._pattern(X, mask, sigma)
        self.u = potential_field(3.0 * a_map, self.pitch)
        self.J1 = self.jacobian(1.0)
        self._u = _Field(self.u, self.geom_c, self.device)
        self._s = _Field(self.s, self.geom_c, self.device)
        self._J = {}
        self._states = {}
        log(f"  phantom: lung {mask.sum() * self.pitch ** 3:.0f} mm^3 in the rest state, centroid "
            f"({self.centroid[0]:+.1f}, {self.centroid[1]:+.1f}, {self.centroid[2]:+.1f}) mm, "
            f"J_lung {self.J_lung:.3f} (a = {a:.4f}), |u| max {np.linalg.norm(self.u, axis=-1).max():.2f} mm, "
            f"field grid {shape_c} at {self.pitch:g} mm")
        if self.pattern == 'texture':
            t = self.truth_record()['texture']
            log(f"  pattern texture: scales {', '.join(f'{v:g}' for v in t['scales_mm'])} mm (sigma), amplitude {t['amp']:g}, "
                f"seed {t['seed']}; J sd in the core {t['J_core_sd']:.3f}")
        elif self.pattern != 'uniform':
            rec = self.truth_record()['regions']
            log(f"  pattern {self.pattern}: " + ', '.join(
                f"region {k}: {v['n_core'] * self.pitch ** 3:.0f} mm^3 core, J {v['J_true_core_mean']:.3f}" for k, v in rec.items())
                + (f"; defect centre ({self.defect_centre[0]:+.1f}, {self.defect_centre[1]:+.1f}, {self.defect_centre[2]:+.1f}) mm, "
                   f"radius {self.defect_mm:g} mm" if self.defect_centre is not None else ''))

    def _pattern(self, X, mask, sigma):
        """The region labels (int8, 0 outside the lung) and the blurred
        source amplitude ``a(x) s(x)`` for the chosen pattern."""
        a1, a2 = self.a, self.J2 ** (1.0 / 3.0) - 1.0
        regions = np.zeros(mask.shape, np.int8)
        centre = None
        if self.pattern == 'uniform':
            regions[mask] = 1
            a_map = np.where(mask, a1, 0.0)
        elif self.pattern == 'lobes':
            right = X[..., 0] >= self.centroid[0]
            regions[mask & ~right], regions[mask & right] = 1, 2
            a_map = np.where(mask, np.where(right, a2, a1), 0.0)
        elif self.pattern == 'defect':
            right = mask & (X[..., 0] >= self.centroid[0])
            centre = X[right].mean(axis=0).astype(np.float64) if right.any() else self.centroid
            sphere = np.linalg.norm(X - centre, axis=-1) <= self.defect_mm
            regions[mask & sphere], regions[mask & ~sphere] = 1, 2
            a_map = np.where(mask, np.where(sphere, a2, a1), 0.0)
        else:                                                    # texture
            regions[mask] = 1
            a_map = np.where(mask, a1, 0.0)
        a_map = ndimage.gaussian_filter(a_map.astype(np.float32), sigma) if sigma > 0 else a_map.astype(np.float32)
        self.texture = None
        if self.pattern == 'texture':
            self.texture = texture_field(mask.shape, self.texture_mm / self.pitch, self.texture_octaves,
                                         self.texture_seed, mask)
            a_map = a_map * np.clip(1.0 + self.texture_amp * self.texture, 0.05, None).astype(np.float32)
        return regions, a_map, centre

    # -- the truth --
    def jacobian(self, t: float) -> np.ndarray:
        """det(I + t grad u) on the coarse grid, central differences."""
        return jacobian_of(self.u, self.pitch, t)

    def lung_volume_change_mm3(self) -> float:
        """The field's total volume change, sum (J - 1) dV over the grid —
        all of it is the lung's by construction (zero divergence outside)."""
        return float((self.J1 - 1.0).sum() * self.pitch ** 3)

    def divergence(self) -> np.ndarray:
        """div u on the coarse grid (central differences) — the field's own
        volume source, 3a s(x) up to discretisation."""
        return sum(np.gradient(self.u[..., i], self.pitch, axis=i) for i in range(3))

    def truth_record(self) -> dict:
        core = self.s > 0.9
        return {'J_lung': self.J_lung, 'blur_mm': self.blur_mm, 'pitch_mm': self.pitch,
                'pattern': self.pattern, 'J2': self.J2,
                'defect': None if self.defect_centre is None else {'centre_mm': self.defect_centre.tolist(), 'radius_mm': self.defect_mm},
                'texture': None if self.texture is None else {
                    'corr_mm': self.texture_mm, 'amp': self.texture_amp, 'octaves': self.texture_octaves, 'seed': self.texture_seed,
                    'scales_mm': [self.texture_mm * 2 ** k for k in range(self.texture_octaves)],
                    'J_core_sd': float(self.J1[core].std()) if core.any() else None,
                    'definition': 'source x (1 + amp n(x)), n unit-variance, equal variance per octave (sigma of the smoothing)'},
                'regions': {str(int(k)): {'n_core': int((core & (self.regions == k)).sum()),
                                     'J_true_core_mean': float(self.J1[core & (self.regions == k)].mean()) if (core & (self.regions == k)).any() else None}
                            for k in np.unique(self.regions[self.regions > 0])},
                'field': 'grad phi, Laplacian phi = 3 a(x) s (zero divergence outside the lung)', 'field_grid': self.geom_c, 'object_grid': self.geom,
                'lung_mm3_rest': float(self.mask.sum() * self.pitch ** 3),
                'lung_centroid_mm': self.centroid.tolist(),
                'displacement_max_mm': float(np.linalg.norm(self.u, axis=-1).max()),
                'lung_volume_change_mm3': self.lung_volume_change_mm3(),
                'hu_calibration': {'scale': self.hu.scale, 'offset': self.hu.offset},
                'phases': {'00': 'inspiration (t = 1)', '01': 'expiration (t = 0, the rest volume)'}}

    # -- the states --
    def _J_field(self, t: float) -> _Field:
        key = round(float(t), 6)
        if key not in self._J:
            self._J[key] = _Field(self.jacobian(t), self.geom_c, self.device)
        return self._J[key]

    @torch.no_grad()
    def inverse_map(self, y: torch.Tensor, t: float, iterations: int = 8) -> torch.Tensor:
        """x with x + t u(x) = y, by fixed-point iteration (|t grad u| << 1)."""
        x = y.clone()
        for _ in range(int(iterations)):
            x = y - float(t) * self._u(x)
        return x

    @torch.no_grad()
    def state(self, t: float, slab: int = 16) -> VolumeSampler:
        """The object at breathing state ``t``: the pushforward of the rest
        attenuation (module doc). ``t = 0`` is the rest volume itself."""
        key = round(float(t), 6)
        if key in self._states:
            return self._states[key]
        if key == 0.0:
            self._states[key] = self.rest
            return self.rest
        nx, ny, nz = self.rest.shape
        Jf = self._J_field(t)
        out = torch.empty((nz, ny, nx), dtype=torch.float32, device=self.device)
        X = torch.as_tensor(grid_centres(self.geom), device=self.device)      # (nx, ny, nz, 3)
        for z0 in range(0, nz, int(slab)):
            z1 = min(nz, z0 + int(slab))
            y = X[:, :, z0:z1].reshape(-1, 3)
            x = self.inverse_map(y, t)
            mu = self.rest(x)
            J = Jf(x, padding='border').reshape(-1).clamp_min(J_FLOOR)
            out[z0:z1] = (mu / J).reshape(nx, ny, z1 - z0).permute(2, 1, 0)
        vol = out.permute(2, 1, 0).cpu().numpy()
        self._states[key] = VolumeSampler(vol, self.geom, device=self.device)
        return self._states[key]


# ------------------------------------------------------------ the schedule --
def breathing_schedule(n_views: int, phases: int = 2, jitter: float = 0.1, seed: int = 0,
                       levels: int = 21) -> np.ndarray:
    """(phases, n_views) breathing states: phase 0 at t near 1 (inspiration),
    phase 1 near 0, each view off its gate by |N(0, jitter)|, quantised to
    ``levels`` values in [0, 1] so the states can be cached."""
    rng = np.random.default_rng(int(seed))
    t = np.zeros((int(phases), int(n_views)), dtype=np.float64)
    for ph in range(int(phases)):
        off = np.abs(rng.standard_normal(int(n_views))) * float(jitter)
        t[ph] = np.clip(1.0 - off, 0.0, 1.0) if ph == 0 else np.clip(off, 0.0, 1.0)
    q = max(1, int(levels) - 1)
    return np.round(t * q) / q


# ------------------------------------------------------------- the files --
def _anchors_for(hu: LinearHU) -> HUAnchors:
    """An ``HUAnchors`` that reproduces a ``LinearHU`` exactly, for the sidecar."""
    return HUAnchors(value_air=(-1000.0 - hu.offset) / hu.scale,
                     value_tissue=(120.0 - hu.offset) / hu.scale,
                     air_hu=-1000.0, tissue_hu=120.0, method='phantom')


def write_state_vff(path, sampler: VolumeSampler, hu: LinearHU, metadata=None) -> str:
    """A state as ``.vff`` (HU, the pipeline's layout) + sidecar."""
    mu = sampler.data[0, 0].permute(2, 1, 0).cpu().numpy()          # (nx, ny, nz)
    vol_hu = hu.apply(mu)
    write_vff(str(path), {'elementsize': sampler.geom['dx'], 'bits': 16},
              vol_hu.transpose(2, 1, 0)[:, ::-1, :], verbose=False)
    write_sidecar(str(path), sampler.geom, _anchors_for(hu), shape=vol_hu.shape,
                  metadata=dict({'algorithm': 'phantom'}, **(metadata or {})))
    return str(path)


def write_truth(folder, ph: BreathingPhantom, schedule: np.ndarray, log=print) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_state_vff(folder / 'state_insp.vff', ph.state(1.0), ph.hu, {'phantom_state': 'inspiration', 't': 1.0})
    write_state_vff(folder / 'state_exp.vff', ph.state(0.0), ph.hu, {'phantom_state': 'expiration', 't': 0.0})
    fields = {'definition': 'det(I + grad u), expiration -> inspiration, expiration frame'}
    write_nrrd_3d(str(folder / 'J.nrrd'), ph.J1, ph.geom_c, units='volume ratio', extra_fields=fields)
    write_nrrd_3d(str(folder / 'lung.nrrd'), ph.s, ph.geom_c, units='fraction',
                  extra_fields={'definition': 'blurred lung indicator s(x)'})
    write_nrrd_3d(str(folder / 'regions.nrrd'), ph.regions.astype(np.float32), ph.geom_c, units='label',
                  extra_fields={'definition': f'pattern {ph.pattern}: 0 outside the lung, 1 / 2 the regions'})
    np.savez_compressed(folder / 'displacement.npz', u_mm=ph.u, s=ph.s, J=ph.J1,
                        grid_shape=np.asarray(ph.geom_c['vol_shape']),
                        grid_origin=np.asarray(ph.geom_c['vol_origin']), pitch_mm=ph.pitch)
    with open(folder / 'schedule.json', 'w') as f:
        json.dump({'t': schedule.tolist(), 'phase_00': 'inspiration', 'phase_01': 'expiration',
                   'definition': 't in [0, 1]: the object at x + t u(x), 1 = inspiration'}, f, indent=2)
    rec = ph.truth_record()
    rec['files'] = {'J': 'J.nrrd', 'lung': 'lung.nrrd', 'regions': 'regions.nrrd', 'displacement': 'displacement.npz',
                    'state_insp': 'state_insp.vff', 'state_exp': 'state_exp.vff', 'schedule': 'schedule.json'}
    with open(folder / 'truth.json', 'w') as f:
        json.dump(rec, f, indent=2)
    log(f"  truth written to {folder}")
    return folder


def build(rest_vff, template, out, *, J_lung: float = 1.25, blur_mm: float = 0.6,
          pattern: str = 'uniform', J2: float = None, defect_mm: float = 1.5,
          texture_mm: float = 0.3, texture_amp: float = 0.3, texture_octaves: int = 4, texture_seed: int = 0,
          jitter: float = 0.1, pitch_mm: float = 0.3, binning: int = 1,
          n_views=None, noise: bool = True, seed: int = 0, device=None, samples_per_mm=None,
          levels: int = 21, log=print) -> Path:
    """The whole phantom: rest volume -> field -> gated scan folder + truth."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    side = load_sidecar(rest_vff, verbose=False)
    if not side:
        raise FileNotFoundError(f"{rest_vff} has no .json sidecar")
    hu = LinearHU(side['hu_calibration']['scale'], side['hu_calibration']['offset'])
    rest = sampler_from_volume_file(rest_vff, device=device)
    ph = BreathingPhantom(rest, hu, pitch_mm=pitch_mm, J_lung=J_lung, blur_mm=blur_mm,
                          pattern=pattern, J2=J2, defect_mm=defect_mm,
                          texture_mm=texture_mm, texture_amp=texture_amp, texture_octaves=texture_octaves,
                          texture_seed=texture_seed, log=log)
    import xmltodict
    n_views = int(n_views or xmltodict.parse((Path(template) / 'scan.xml').read_text())
                  ['Series']['SeriesParams']['ViewCount'])
    sched = breathing_schedule(n_views, 2, jitter, seed, levels)
    log(f"  schedule: {len(np.unique(sched))} distinct states; phase 00 t in "
        f"[{sched[0].min():.2f}, {sched[0].max():.2f}], phase 01 t in [{sched[1].min():.2f}, {sched[1].max():.2f}]")

    def source(phase, view, angle_deg):
        return ph.state(float(sched[phase, view]))

    folder = simulate_scan(source, template, out, binning=binning, n_views=n_views, phases=2,
                           noise=noise, seed=seed, samples_per_mm=samples_per_mm, device=device,
                           specimen='PHANTOM', extra_record={'phantom': ph.truth_record()}, log=log)
    write_truth(folder / TRUTH_DIR, ph, sched, log=log)
    breathing_gif(ph, folder / TRUTH_DIR / 'breathing.gif', log=log)
    return folder


def write_truth_only(rest_vff, template, out, *, J_lung: float = 1.25, blur_mm: float = 0.6,
                     pattern: str = 'uniform', J2: float = None, defect_mm: float = 1.5,
                     texture_mm: float = 0.3, texture_amp: float = 0.3, texture_octaves: int = 4,
                     texture_seed: int = 0, jitter: float = 0.1, pitch_mm: float = 0.3, n_views=None,
                     seed: int = 0, device=None, levels: int = 21, log=print) -> Path:
    """``build`` without the projections: the same phantom and schedule, only
    ``<out>/truth/`` written. The phantom is deterministic in its parameters,
    so this reproduces the truth of a scan built elsewhere (the cluster) and
    lets ``score`` run where the projections were never copied."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    side = load_sidecar(rest_vff, verbose=False)
    if not side:
        raise FileNotFoundError(f"{rest_vff} has no .json sidecar")
    hu = LinearHU(side['hu_calibration']['scale'], side['hu_calibration']['offset'])
    rest = sampler_from_volume_file(rest_vff, device=device)
    ph = BreathingPhantom(rest, hu, pitch_mm=pitch_mm, J_lung=J_lung, blur_mm=blur_mm,
                          pattern=pattern, J2=J2, defect_mm=defect_mm,
                          texture_mm=texture_mm, texture_amp=texture_amp, texture_octaves=texture_octaves,
                          texture_seed=texture_seed, log=log)
    import xmltodict
    n_views = int(n_views or xmltodict.parse((Path(template) / 'scan.xml').read_text())
                  ['Series']['SeriesParams']['ViewCount'])
    sched = breathing_schedule(n_views, 2, jitter, seed, levels)
    folder = write_truth(Path(out) / TRUTH_DIR, ph, sched, log=log)
    breathing_gif(ph, folder / 'breathing.gif', log=log)
    return folder


# ---------------------------------------------------------------- the gif --
HU_WINDOW = (-1000.0, 400.0)
J_WINDOW = (0.75, 1.5)


def breathing_gif(ph: BreathingPhantom, path, n: int = 8, fps: int = 5, log=print) -> str:
    """``breathing.gif``: the coronal slice through the lung centroid, the HU
    state on the left and the truth ``J_t`` map on the right, ``t`` swept
    0 -> 1 -> 0 with a cosine ease (rounded to the schedule's 0.05 levels so
    the states are the cached ones). A still of the inspiration frame is
    written next to it as ``.png``."""
    import matplotlib
    from .show import write_gif
    cmap = matplotlib.colormaps['RdBu_r']
    nx, ny, nz = ph.rest.shape
    dx = float(ph.geom['dx'])
    iy = int(round((ph.centroid[1] - ph.geom['vol_origin'][1]) / dx + (ny - 1) / 2.0))
    iy = min(max(iy, 0), ny - 1)
    iyc = int(round((ph.centroid[1] - ph.geom_c['vol_origin'][1]) / ph.pitch + (ph.geom_c['vol_shape'][1] - 1) / 2.0))
    iyc = min(max(iyc, 0), ph.geom_c['vol_shape'][1] - 1)
    ts = 0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, 2 * int(n), endpoint=False))
    ts = np.round(ts * 20) / 20
    frames = []
    for t in ts:
        vol = ph.state(float(t)).data[0, 0].permute(2, 1, 0)                 # (nx, ny, nz)
        hu = ph.hu.apply(vol[:, iy, :].cpu().numpy()).T                        # (nz, nx)
        g = np.clip((hu - HU_WINDOW[0]) / (HU_WINDOW[1] - HU_WINDOW[0]), 0, 1)
        grey = np.repeat((g * 255).astype(np.uint8)[..., None], 3, axis=-1)
        J = ph.jacobian(float(t))[:, iyc, :].T                                 # (nzc, nxc)
        k = max(1, int(round(ph.pitch / dx)))
        J = np.repeat(np.repeat(J, k, axis=0), k, axis=1)[:grey.shape[0], :grey.shape[1]]
        pad = [(0, grey.shape[0] - J.shape[0]), (0, grey.shape[1] - J.shape[1])]
        J = np.pad(J, pad, constant_values=1.0)
        rgb = (cmap(np.clip((J - J_WINDOW[0]) / (J_WINDOW[1] - J_WINDOW[0]), 0, 1))[..., :3] * 255).astype(np.uint8)
        frames.append(np.concatenate([grey, rgb], axis=1)[::-1])               # z up
    frames = np.stack(frames)
    write_gif(frames, str(path), fps=fps)
    from PIL import Image
    Image.fromarray(frames[int(n)]).save(str(Path(path).with_suffix('.png')))
    log(f"  breathing gif written to {path} ({len(frames)} frames; HU {HU_WINDOW}, J {J_WINDOW}; "
        f"coronal y = {ph.centroid[1]:+.1f} mm)")
    return str(path)


# -------------------------------------------------------------- the score --
def _ncc(a, b, m):
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    if a.size < 2:
        return float('nan')
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float('nan')


def _block_corr(a, b, m, block: int) -> tuple:
    """Correlation of block means (``block`` voxels a side) over blocks with
    at least half their voxels in ``m``."""
    nx, ny, nz = a.shape
    bx, by, bz = (n // block for n in (nx, ny, nz))
    if min(bx, by, bz) == 0:
        return float('nan'), 0

    def blocks(v):
        v = v[:bx * block, :by * block, :bz * block].reshape(bx, block, by, block, bz, block)
        return v.sum(axis=(1, 3, 5))
    w = blocks(m.astype(np.float64))
    keep = w >= 0.5 * block ** 3
    if keep.sum() < 3:
        return float('nan'), int(keep.sum())
    am = blocks(np.where(m, a, 0.0)) / np.maximum(w, 1)
    bm = blocks(np.where(m, b, 0.0)) / np.maximum(w, 1)
    return float(np.corrcoef(am[keep], bm[keep])[0, 1]), int(keep.sum())


def _block_corr_shifted(a, b, m, block: int, rng, n_origins: int = 8) -> tuple:
    """``_block_corr`` averaged over shifted block-grid origins (a single
    origin makes r at one size a random draw): mean, sd, mean block count."""
    offs = [(0, 0, 0)] if block == 1 else [tuple(int(v) for v in rng.integers(0, block, 3)) for _ in range(int(n_origins))]
    rs, ns = [], []
    for o in offs:
        r, n = _block_corr(a[o[0]:, o[1]:, o[2]:], b[o[0]:, o[1]:, o[2]:], m[o[0]:, o[1]:, o[2]:], block)
        rs.append(r); ns.append(n)
    rs = np.asarray(rs, dtype=np.float64)
    return float(np.nanmean(rs)) if np.isfinite(rs).any() else float('nan'), float(np.nanstd(rs)) if np.isfinite(rs).any() else float('nan'), float(np.mean(ns))


def coherence_spectrum(a, b, w, pitch: float, n_bins: int = 12) -> dict:
    """Coherence and gain of ``a`` against ``b`` per spatial frequency,
    both windowed by ``w`` (the smooth lung indicator, so the boundary
    does not ring) after removing their windowed means. Radial bins are
    log-spaced from the window's extent to the grid's Nyquist. Coherence is
    ``|<A B*>| / sqrt(<|A|^2> <|B|^2>)`` per bin (1 = same map up to a scale
    at that frequency), gain ``Re<A B*> / <|B|^2>`` (the transfer, 1 = the
    truth's amplitude kept). A pure blur leaves the coherence at one (it
    scales each frequency, it does not scramble it) and lowers the gain, so
    ``resolution_mm`` is the finest scale (1 / f) at which the GAIN still
    reaches one half (the MTF-50 of the ventilation map), linearly
    interpolated across bins, None when it never drops below; and
    ``noise_scale_mm`` the same for the coherence (where the map stops
    following the truth at all: noise or misplaced structure). The window's
    own edge leaks into both maps at the finest scales (a CONSTANT map under
    the same window already coheres with the truth there), so each bin also
    carries ``leak`` = that constant map's coherence; the run of bins at the
    fine end with leak >= 0.5 is ``window_limited`` and left out of both
    searches (at coarse scales J follows the lung's shape, so a leak there is
    genuine shared structure, not an artefact)."""
    w = np.asarray(w, np.float64)
    if w.sum() <= 0:
        return {}
    a = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    b = np.asarray(b, np.float64)
    fa = w * (a - (w * a).sum() / w.sum())
    fb = w * (b - (w * b).sum() / w.sum())
    A, B = np.fft.fftn(fa), np.fft.fftn(fb)
    freqs = [np.fft.fftfreq(n, d=float(pitch)) for n in a.shape]
    k = np.sqrt(sum(f.reshape([-1 if i == ax else 1 for i in range(3)]) ** 2 for ax, f in enumerate(freqs)))
    idx = np.argwhere(w > 0.5)
    extent = float(((idx.max(0) - idx.min(0) + 1) * pitch).min()) if len(idx) else float(min(a.shape) * pitch)
    f_lo, f_hi = 1.0 / extent, 0.5 / float(pitch)
    if f_hi <= f_lo * 1.5:
        return {}
    edges = np.geomspace(f_lo, f_hi, int(n_bins) + 1)
    cross, pa, pb = A * np.conj(B), np.abs(A) ** 2, np.abs(B) ** 2
    C = np.fft.fftn(w)                    # the window alone: what a constant map contributes
    leak_cross, pc = C * np.conj(B), np.abs(C) ** 2
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k >= lo) & (k < hi)
        if m.sum() < 8:
            continue
        c, xa, xb = cross[m].sum(), pa[m].sum(), pb[m].sum()
        if xa <= 0 or xb <= 0:
            continue
        lc, xc = leak_cross[m].sum(), pc[m].sum()
        leak = float(abs(lc) / np.sqrt(xc * xb)) if xc > 0 else 0.0
        rows.append((float(np.sqrt(lo * hi)), float(abs(c) / np.sqrt(xa * xb)), float(c.real / xb), int(m.sum()), leak))
    if not rows:
        return {}
    f, coh, gain, n, leak = (np.asarray(v) for v in zip(*rows))
    ok = np.ones(len(f), bool)            # window-limited = the run of bins at the FINE end the window explains
    for j in range(len(f) - 1, -1, -1):   # (at coarse scales J follows the lung's shape, so a leak there is real)
        if leak[j] < 0.5:
            break
        ok[j] = False

    def half_scale(y):
        y = np.where(ok, y, np.nan)
        below = np.nonzero(y < 0.5)[0]
        if not len(below):
            return None
        j = int(below[0])
        if j == 0:
            return float(1.0 / f[0])
        t = (y[j - 1] - 0.5) / max(y[j - 1] - y[j], 1e-12)
        return float(1.0 / (f[j - 1] + t * (f[j] - f[j - 1])))
    return {'freq_per_mm': f.tolist(), 'scale_mm': (1.0 / f).tolist(), 'coherence': coh.tolist(), 'gain': gain.tolist(),
            'n_modes': n.tolist(), 'leak': leak.tolist(), 'window_limited': (~ok).tolist(),
            'resolution_mm': half_scale(gain), 'noise_scale_mm': half_scale(coh),
            'definition': 'lung-windowed J maps; coherence |<A B*>|/sqrt(<|A|^2><|B|^2>) and gain Re<A B*>/<|B|^2> per '
                          'log-spaced radial frequency bin; resolution_mm = finest scale with gain >= 0.5 (MTF-50), '
                          'noise_scale_mm = finest scale with coherence >= 0.5; leak = coherence of the window alone '
                          'with the truth, bins with leak >= 0.5 are window_limited and excluded from both'}


def scale_analysis(J_rec, J_true, s, finite, pitch: float, seed: int = 0) -> dict:
    """The recovered map against the truth as a function of scale (module
    doc, PATTERNS): block correlation vs block size over the core (s > 0.9)
    and the lung (s > 0.5), each over shifted origins, out to the size that
    still holds 8 blocks, and the coherence spectrum over the lung."""
    rng = np.random.default_rng(int(seed))
    Jr = np.where(finite, J_rec, 1.0)
    out = {'block': []}
    masks = {'core': finite & (s > 0.9), 'lung': finite & (s > 0.5)}
    k = 1
    while True:
        row = {'block_mm': float(k * pitch)}
        alive = False
        for name, m in masks.items():
            r, sd, n = _block_corr_shifted(Jr, J_true, m, k, rng)
            row[name] = {'r': r, 'sd_origins': sd, 'n_blocks': n}
            alive = alive or n >= 8
        if not alive:
            break
        out['block'].append(row)
        k = k + 1 if k < 4 else int(round(k * 1.3))
        if k * pitch > 8.0:
            break
    out['coherence'] = coherence_spectrum(Jr, J_true, np.where(finite, s, 0.0), pitch)
    return out


def score(run_dir, scan_dir, out=None, block_mm: float = 1.5, log=print) -> dict:
    """Compare a run on the phantom's scan with the truth (module doc)."""
    run, truth_dir = Path(run_dir), Path(scan_dir) / TRUTH_DIR
    truth = json.load(open(truth_dir / 'truth.json'))
    geom_t = truth['field_grid']
    J_peak_c = read_nrrd_3d(str(truth_dir / 'J.nrrd'))
    s_c = read_nrrd_3d(str(truth_dir / 'lung.nrrd'))
    J_true_c, t0, t1 = gate_mean_truth(truth_dir)
    if J_true_c is None:
        J_true_c, t0, t1 = J_peak_c, 1.0, 0.0

    res = {'run': run.name, 'phantom': {k: truth[k] for k in ('J_lung', 'blur_mm', 'lung_volume_change_mm3')},
           'truth_states': {'t_insp': t0, 't_exp': t1,
                            'definition': 'J(t_insp) / J(t_exp) with the schedule\'s gate-mean levels (what a gated '
                                          'reconstruction measures); *_peak = the J(1) truth'}}
    vj_path = run / 'recon_ventilation.json'
    if vj_path.exists():
        vj = json.load(open(vj_path))
        geom_r = vj['grid']
        J_rec = read_nrrd_3d(str(run / vj['files']['J']))
        X = grid_centres(geom_r).reshape(-1, 3)
        J_true = vent.sample(J_true_c, geom_t, X).reshape(J_rec.shape)
        J_peak = vent.sample(J_peak_c, geom_t, X).reshape(J_rec.shape)
        s = vent.sample(s_c, geom_t, X).reshape(J_rec.shape)
        finite = np.isfinite(J_rec)
        core = finite & (s > 0.9)
        shell = finite & (s > 0.1) & (s <= 0.9)
        p = float(geom_r['dx'])
        block = max(1, int(round(block_mm / p)))
        r_block, n_blocks = _block_corr(J_rec, J_true, core, block)
        err = J_rec[core] - J_true[core]
        shells = []
        for lo, hi in ((0.02, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.95), (0.95, 1.01)):
            m = finite & (s > lo) & (s <= hi)
            if m.sum() >= 10:
                shells.append({'s_range': [lo, hi], 'n': int(m.sum()), 'J_rec_mean': float(J_rec[m].mean()),
                               'J_true_mean': float(J_true[m].mean())})
        lung = finite & (s > 0.5)
        res['ventilation'] = {
            'grid_pitch_mm': p, 'n_core': int(core.sum()), 'n_shell': int(shell.sum()),
            'covered_frac_of_lung': float(finite[s > 0.5].mean()) if (s > 0.5).any() else float('nan'),
            'J_rec_median_core': float(np.median(J_rec[core])) if core.any() else float('nan'),
            'J_true_median_core': float(np.median(J_true[core])) if core.any() else float('nan'),
            'J_true_peak_median_core': float(np.median(J_peak[core])) if core.any() else float('nan'),
            'bias_core': float(np.median(err)) if core.any() else float('nan'),
            'rms_core': float(np.sqrt((err ** 2).mean())) if core.any() else float('nan'),
            'r_voxel_core': _ncc(J_rec, J_true, core),
            'r_block_core': r_block, 'n_blocks': n_blocks, 'block_mm': block * p,
            'shells': shells,
            'lung_volume_change_mm3': {'rec': float(np.nansum((J_rec - 1.0)[lung]) * p ** 3),
                                       'true': float(((J_true - 1.0)[lung]).sum() * p ** 3),
                                       'true_peak': float(((J_peak - 1.0)[lung]).sum() * p ** 3)},
            'smoothing_fit': smoothing_fit(J_rec, J_true_c, geom_t, X, finite & (s > 0.02), float(geom_t['dx'])),
            'solid_tissue_strain_rec': vj.get('solid_tissue_strain'),
            'J_rec_frac_below_1_core': float((J_rec[core] < 1.0).mean()) if core.any() else float('nan'),
            'scale': scale_analysis(J_rec, J_true, s, finite, p)}
        # the tidal volume, three ways, against the truth's TOTAL (all of the
        # field's volume change is the lung's) and against the truth over the
        # same band
        tv = vj.get('tidal_volume_mm3')
        if isinstance(tv, dict):
            n_dil = int(round(float(tv.get('dilate_mm', 0.0)) / p))
            band = ndimage.binary_dilation(finite, iterations=n_dil) if n_dil > 0 else finite
            res['ventilation']['tidal_volume_mm3'] = {
                'true_total': float((J_true_c - 1.0).sum() * float(geom_t['dx']) ** 3),
                'true_total_peak': float((J_peak_c - 1.0).sum() * float(geom_t['dx']) ** 3),
                'true_over_band': float(((J_true - 1.0)[band]).sum() * p ** 3),
                'rec_lung_parcels': tv.get('lung_parcels'), 'rec_band': tv.get('band'),
                'rec_window_all_parcels': tv.get('window_all_parcels'), 'dilate_mm': tv.get('dilate_mm')}
            pk = vj.get('peak') or {}
            if pk:
                res['ventilation']['tidal_volume_mm3'].update({
                    'rec_band_peak': pk['tidal_volume_mm3']['band'], 'rec_lung_parcels_peak': pk['tidal_volume_mm3']['lung_parcels']})
        # the PEAK reading against the peak-state truth J(1) / J(0)
        pk_file = vj.get('files', {}).get('J_peak')
        if pk_file and (run / pk_file).exists():
            J_pk = read_nrrd_3d(str(run / pk_file))
            errp = J_pk[core] - J_peak[core]
            res['ventilation']['peak'] = {
                'J_rec_median_core': float(np.median(J_pk[core])) if core.any() else float('nan'),
                'J_true_median_core': float(np.median(J_peak[core])) if core.any() else float('nan'),
                'bias_core': float(np.median(errp)) if core.any() else float('nan'),
                'rms_core': float(np.sqrt((errp ** 2).mean())) if core.any() else float('nan'),
                'r_voxel_core': _ncc(J_pk, J_peak, core),
                'levels': {k: (vj.get('peak') or {}).get(k) for k in ('level_insp', 'level_exp', 'amplitude_over_gate_mean')},
                'smoothing_fit': smoothing_fit(J_pk, J_peak_c, geom_t, X, finite & (s > 0.02), float(geom_t['dx']))}
        regions_path = truth_dir / truth.get('files', {}).get('regions', 'regions.nrrd')
        if regions_path.exists():
            lab = np.rint(vent.sample(read_nrrd_3d(str(regions_path)), geom_t, X, order=0)).reshape(J_rec.shape).astype(np.int8)
            per = {}
            for k in (1, 2):
                m = core & (lab == k)
                if m.sum() >= 10:
                    per[str(k)] = {'n_core': int(m.sum()), 'J_rec_mean': float(J_rec[m].mean()),
                                   'J_true_mean': float(J_true[m].mean())}
            if len(per) == 2:
                d_rec = per['1']['J_rec_mean'] - per['2']['J_rec_mean']
                d_true = per['1']['J_true_mean'] - per['2']['J_true_mean']
                per['contrast'] = {'rec': d_rec, 'true': d_true,
                                   'recovered': d_rec / d_true if abs(d_true) > 1e-9 else None}
            res['ventilation']['regions'] = {'pattern': truth.get('pattern', 'uniform'), **per}
        if truth.get('texture'):
            res['ventilation']['texture'] = {'truth': truth['texture'],
                                             'J_core_sd_rec': float(J_rec[core].std()) if core.any() else float('nan'),
                                             'J_core_sd_true': float(J_true[core].std()) if core.any() else float('nan')}
        res['_maps'] = {'J_rec': J_rec, 'J_true': J_true, 's': s, 'geom': geom_r}
    else:
        log("  no recon_ventilation.json in the run: ventilation not scored")

    vff = run / 'recon.vff'
    if vff.exists():
        rec_hu, _ = load_reconstructed_volume(str(vff), verbose=False)
        side = load_sidecar(str(vff), verbose=False)
        geom_v = {'vol_shape': tuple(side['volume_shape']), 'vol_origin': tuple(side['vol_origin_mm']),
                  'dx': side['voxel_size_mm']['xy'], 'dz': side['voxel_size_mm']['z']}
        tru_hu, _ = load_reconstructed_volume(str(truth_dir / 'state_insp.vff'), verbose=False)
        geom_i = truth['object_grid']
        same = (tuple(int(v) for v in geom_i['vol_shape']) == tuple(geom_v['vol_shape'])
                and np.allclose(geom_i['vol_origin'], geom_v['vol_origin'], atol=1e-6)
                and abs(float(geom_i['dx']) - float(geom_v['dx'])) < 1e-9)
        inside = np.ones(rec_hu.shape, bool)
        offset_vox = np.zeros(3)
        if not same:
            # The run measures its own domain from the projections, so its grid is usually
            # the truth's shifted by a fraction of a voxel (the first phantom run: half a
            # voxel in every axis). Resampling ONE side blurs that side alone, and the other
            # side's sharp bone edges then land in every HU window (soft NCC 0.65 where an
            # equal treatment gives 0.96). Both sides are therefore sampled on the grid
            # halfway between the two, so each gets the same interpolation.
            h = np.array([geom_v['dx'], geom_v['dx'], geom_v['dz']], dtype=np.float64)
            centres = (np.asarray(geom_v['vol_origin'], np.float64) - np.asarray(geom_i['vol_origin'], np.float64)
                       + (np.asarray(geom_v['vol_shape'], np.float64) - np.asarray(geom_i['vol_shape'], np.float64)) / 2.0 * h)
            offset_vox = (centres / h + 0.5) % 1.0 - 0.5                 # nearest-centre offset, voxels
            X = grid_centres(geom_v).reshape(-1, 3) - 0.5 * offset_vox * h
            inside = ((vent.sample(np.ones(tru_hu.shape, np.float32), geom_i, X, order=0) > 0)
                      & (vent.sample(np.ones(rec_hu.shape, np.float32), geom_v, X, order=0) > 0)).reshape(rec_hu.shape)
            rec_hu = vent.sample(rec_hu, geom_v, X).reshape(inside.shape)
            tru_hu = vent.sample(tru_hu, geom_i, X).reshape(inside.shape)
        windows = {'all': inside, 'soft': inside & (tru_hu > -300) & (tru_hu < 300),
                   'lung': inside & (tru_hu > -800) & (tru_hu < -300), 'bone': inside & (tru_hu > 300)}
        res['image_insp'] = {k: {'ncc': _ncc(rec_hu, tru_hu, m), 'rms_hu': float(np.sqrt(((rec_hu - tru_hu)[m] ** 2).mean())) if m.any() else float('nan'),
                                 'n': int(m.sum())} for k, m in windows.items()}
        res['image_insp']['resampled_truth'] = not same
        res['image_insp']['grid_offset_voxels'] = [float(v) for v in offset_vox]
        res['image_insp']['resampling'] = ('none (same grid)' if same else
                                           'both sides trilinear onto the grid halfway between the two')

    out = Path(out) if out else run / 'phantom_score.json'
    maps = res.pop('_maps', None)
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    log(f"  score written to {out}")
    if maps is not None:
        try:
            _figure(maps, res, out.with_suffix('.png'))
            log(f"  figure written to {out.with_suffix('.png')}")
        except Exception as e:                         # a figure never ends a score
            log(f"  figure skipped ({type(e).__name__}: {e})")
    return res


def _figure(maps: dict, res: dict, path) -> None:
    from matplotlib.figure import Figure
    J_rec, J_true, s = maps['J_rec'], maps['J_true'], maps['s']
    lung = s > 0.5
    idx = np.argwhere(lung)
    cy = int(np.median(idx[:, 1])) if len(idx) else J_rec.shape[1] // 2
    fig = Figure(figsize=(19, 4.4), dpi=110)
    ax = fig.subplots(1, 5)
    for a, (m, title) in zip(ax[:3], ((J_true, 'truth J'), (J_rec, 'reconstructed J'), (J_rec - J_true, 'difference'))):
        v = m[:, cy, :].T
        if title == 'difference':
            im = a.imshow(v, origin='lower', cmap='RdBu_r', vmin=-0.2, vmax=0.2, aspect='equal')
        else:
            im = a.imshow(v, origin='lower', cmap='viridis', vmin=0.9, vmax=1.5, aspect='equal')
        a.set_title(f"{title} (coronal, y = {cy})", fontsize=9)
        fig.colorbar(im, ax=a, shrink=0.8)
    sh = res.get('ventilation', {}).get('shells', [])
    if sh:
        x = [0.5 * (d['s_range'][0] + d['s_range'][1]) for d in sh]
        ax[3].plot(x, [d['J_true_mean'] for d in sh], 'k-o', label='truth')
        ax[3].plot(x, [d['J_rec_mean'] for d in sh], 'r-o', label='reconstruction')
        ax[3].set_xlabel('lung indicator s (0 tissue .. 1 lung core)')
        ax[3].set_ylabel('mean J')
        ax[3].legend(fontsize=8)
        v = res['ventilation']
        ax[3].set_title(f"core: bias {v['bias_core']:+.3f}, rms {v['rms_core']:.3f}, block r {v['r_block_core']:.2f}", fontsize=9)
    sc = res.get('ventilation', {}).get('scale') or {}
    bl = [d for d in sc.get('block', []) if d['lung']['n_blocks'] >= 8]
    co = sc.get('coherence') or {}
    if bl:
        ax[4].plot([d['block_mm'] for d in bl], [d['lung']['r'] for d in bl], 'r-o', ms=3, label='block r, lung')
        cb = [d for d in bl if d['core']['n_blocks'] >= 8]
        ax[4].plot([d['block_mm'] for d in cb], [d['core']['r'] for d in cb], 'r--o', ms=3, label='block r, core')
    if co:
        ax[4].plot(co['scale_mm'], co['coherence'], 'k-s', ms=3, label='coherence')
        ax[4].plot(co['scale_mm'], np.clip(co['gain'], -0.2, 1.5), 'k:s', ms=3, label='gain')
        wl = np.asarray(co.get('window_limited', [False] * len(co['scale_mm'])), bool)
        if wl.any():
            ax[4].axvspan(min(co['scale_mm']), max(np.asarray(co['scale_mm'])[wl]), color='0.85', zorder=0,
                          label='window leakage')
        if co.get('resolution_mm'):
            ax[4].axvline(co['resolution_mm'], color='0.5', lw=0.8)
    if bl or co:
        ax[4].set_xscale('log'); ax[4].set_ylim(-0.2, 1.05); ax[4].axhline(0.5, color='0.8', lw=0.6)
        ax[4].set_xlabel('scale (mm): block size / 1 over frequency'); ax[4].legend(fontsize=7, loc='lower right')
        ax[4].set_title('recovered J vs truth by scale' + (f"; resolution {co['resolution_mm']:.2f} mm" if co.get('resolution_mm') else ''), fontsize=9)
    fig.savefig(path)


# ---------------------------------------------------------------- the CLI --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="breathing digital phantom: build a gated scan with a known "
                                             "ventilation, or score a run on it")
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='rest volume -> gated synthetic scan folder + truth/')
    b.add_argument('rest', help='a .vff this pipeline wrote (with its .json), the expiration state')
    b.add_argument('--template', required=True, help='real scan folder whose acquisition to copy')
    b.add_argument('--output', required=True, help='scan folder to write (data/scans/<name>)')
    b.add_argument('--J', type=float, default=1.25, help='lung volume ratio inspiration/expiration')
    b.add_argument('--blur-mm', type=float, default=0.6, help='width of the lung boundary ramp')
    b.add_argument('--pattern', choices=PATTERNS, default='uniform',
                   help='uniform: --J everywhere; lobes: --J in the lung of smaller x, --J2 in the other; '
                        'defect: an unventilated sphere (--J2, default 1) of --defect-mm at the centre of the larger-x lung; '
                        'texture: --J everywhere times a broadband random texture (--texture-*), the resolution test')
    b.add_argument('--J2', type=float, default=None, help='the second region\'s volume ratio (lobes: required; defect: 1)')
    b.add_argument('--defect-mm', type=float, default=1.5, help='radius of the defect sphere')
    b.add_argument('--texture-mm', type=float, default=0.3,
                   help='texture: finest correlation length (sigma, mm); needs --pitch <= half of it (default 0.3)')
    b.add_argument('--texture-amp', type=float, default=0.3, help='texture: relative sd of the source (default 0.3)')
    b.add_argument('--texture-octaves', type=int, default=4,
                   help='texture: number of octaves above --texture-mm, equal variance each (default 4: 0.3, 0.6, 1.2, 2.4 mm)')
    b.add_argument('--texture-seed', type=int, default=0)
    b.add_argument('--jitter', type=float, default=0.1, help='per-view gating jitter in t (sd)')
    b.add_argument('--pitch', type=float, default=0.3, help='field grid pitch (mm)')
    b.add_argument('--binning', type=int, default=1, help='detector binning of the written frames')
    b.add_argument('--views', type=int, default=None, help='views per phase (default: the template\'s)')
    b.add_argument('--no-noise', action='store_true')
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--samples-per-mm', type=float, default=None)
    b.add_argument('--device', default=None)
    t = sub.add_parser('truth', help='rebuild only truth/ of a phantom scan (no projections): the same '
                                     'deterministic phantom the build wrote, for scoring a cluster run locally')
    t.add_argument('rest', help='the rest .vff the phantom was built from (with its .json)')
    t.add_argument('--template', required=True, help='real scan folder whose view count the build used')
    t.add_argument('--output', required=True, help='scan folder to hold truth/ (data/scans/<name>)')
    for a in ('--J', '--blur-mm', '--pattern', '--J2', '--defect-mm', '--texture-mm', '--texture-amp',
              '--texture-octaves', '--texture-seed', '--jitter', '--pitch', '--views', '--seed', '--device'):
        act = next(x for x in b._actions if a in x.option_strings)
        t.add_argument(a, type=act.type, default=act.default, choices=act.choices, help=act.help)
    s = sub.add_parser('score', help='compare a finished run with the phantom truth')
    s.add_argument('run', help='result directory (recon.vff, recon_ventilation*.{json,nrrd})')
    s.add_argument('scan', help='the phantom scan folder (holds truth/)')
    s.add_argument('--out', default=None, help='score JSON path (default: <run>/phantom_score.json)')
    s.add_argument('--block-mm', type=float, default=1.5)
    args = ap.parse_args(argv)
    if args.cmd == 'build':
        if args.pattern == 'lobes' and args.J2 is None:
            ap.error('--pattern lobes needs --J2')
        if args.pattern == 'texture' and args.pitch > args.texture_mm / 2.0:
            ap.error(f'--pattern texture with --texture-mm {args.texture_mm:g} needs --pitch <= {args.texture_mm / 2:g}')
        folder = build(args.rest, args.template, args.output, J_lung=args.J, blur_mm=args.blur_mm,
                       pattern=args.pattern, J2=args.J2, defect_mm=args.defect_mm,
                       texture_mm=args.texture_mm, texture_amp=args.texture_amp,
                       texture_octaves=args.texture_octaves, texture_seed=args.texture_seed,
                       jitter=args.jitter, pitch_mm=args.pitch,
                       binning=args.binning, n_views=args.views, noise=not args.no_noise, seed=args.seed,
                       device=args.device, samples_per_mm=args.samples_per_mm)
        print(f"Phantom scan written to: {folder}")
        return 0
    if args.cmd == 'truth':
        folder = write_truth_only(args.rest, args.template, args.output, J_lung=args.J, blur_mm=args.blur_mm,
                                  pattern=args.pattern, J2=args.J2, defect_mm=args.defect_mm,
                                  texture_mm=args.texture_mm, texture_amp=args.texture_amp,
                                  texture_octaves=args.texture_octaves, texture_seed=args.texture_seed,
                                  jitter=args.jitter, pitch_mm=args.pitch, n_views=args.views, seed=args.seed,
                                  device=args.device)
        print(f"Phantom truth written to: {folder}")
        return 0
    res = score(args.run, args.scan, out=args.out, block_mm=args.block_mm)
    v = res.get('ventilation')
    if v:
        ts = res['truth_states']
        print(f"  truth = gate-mean states (t {ts['t_insp']:.2f} vs {ts['t_exp']:.2f}); core J peak {v['J_true_peak_median_core']:.3f}")
        print(f"  lung core: J rec {v['J_rec_median_core']:.3f} vs truth {v['J_true_median_core']:.3f} "
              f"(bias {v['bias_core']:+.3f}, rms {v['rms_core']:.3f}); block r {v['r_block_core']:.3f} "
              f"over {v['n_blocks']} blocks of {v['block_mm']:.1f} mm; lung volume change rec "
              f"{v['lung_volume_change_mm3']['rec']:.1f} vs truth {v['lung_volume_change_mm3']['true']:.1f} mm^3")
        sf = v.get('smoothing_fit') or {}
        if sf:
            print(f"  map = truth smoothed by sigma {sf['sigma_mm']:.2f} mm x {sf['k']:.2f} (rms {sf['rms']:.4f})")
        tv = v.get('tidal_volume_mm3')
        if tv:
            print(f"  tidal volume vs the truth's total {tv['true_total']:.1f} mm^3: lung parcels {tv['rec_lung_parcels']:.1f} "
                  f"({100 * tv['rec_lung_parcels'] / tv['true_total']:.0f} %), band {tv['rec_band']:.1f} "
                  f"({100 * tv['rec_band'] / tv['true_total']:.0f} %; truth over the same band {tv['true_over_band']:.1f})"
                  + (f"; peak band {tv['rec_band_peak']:.1f} vs {tv['true_total_peak']:.1f} ({100 * tv['rec_band_peak'] / tv['true_total_peak']:.0f} %)"
                     if 'rec_band_peak' in tv else ''))
        pk = v.get('peak')
        if pk:
            print(f"  peak states: core J rec {pk['J_rec_median_core']:.3f} vs truth J(1)/J(0) {pk['J_true_median_core']:.3f} "
                  f"(bias {pk['bias_core']:+.3f}); levels {pk['levels']['level_insp']:+.2f} / {pk['levels']['level_exp']:+.2f}, "
                  f"amplitude {pk['levels']['amplitude_over_gate_mean']:.2f}x the gate means")
        sc = v.get('scale') or {}
        if sc.get('block'):
            bl = sc['block']
            print("  scale: block r (lung) " + ', '.join(f"{d['block_mm']:.2g} mm {d['lung']['r']:.2f}" for d in bl if d['lung']['n_blocks'] >= 8))
            co = sc.get('coherence') or {}
            if co:
                print("  gain vs the truth: " + ', '.join(f"{sm:.2g} mm {g:.2f}" for sm, g in zip(co['scale_mm'], co['gain']))
                      + (f"; resolution (gain 1/2) {co['resolution_mm']:.2f} mm" if co.get('resolution_mm') else
                         "; gain never below 1/2 down to the grid"))
                wl = np.asarray(co.get('window_limited', []), bool)
                if wl.any():
                    print(f"  (bins at {max(np.asarray(co['scale_mm'])[wl]):.2g} mm and finer are window leakage, not scored)")
                print("  coherence with the truth: " + ', '.join(f"{sm:.2g} mm {c:.2f}" for sm, c in zip(co['scale_mm'], co['coherence']))
                      + (f"; below 1/2 from {co['noise_scale_mm']:.2f} mm" if co.get('noise_scale_mm') else
                         "; never below 1/2 down to the grid"))
        tx = v.get('texture')
        if tx:
            print(f"  texture: J sd in the core rec {tx['J_core_sd_rec']:.3f} vs truth {tx['J_core_sd_true']:.3f} "
                  f"(scales {', '.join(f'{x:g}' for x in tx['truth']['scales_mm'])} mm, amplitude {tx['truth']['amp']:g})")
        reg = v.get('regions')
        if reg and 'contrast' in reg:
            c = reg['contrast']
            print(f"  pattern {reg['pattern']}: region 1 J rec {reg['1']['J_rec_mean']:.3f} vs truth {reg['1']['J_true_mean']:.3f}, "
                  f"region 2 rec {reg['2']['J_rec_mean']:.3f} vs truth {reg['2']['J_true_mean']:.3f}; contrast rec {c['rec']:+.3f} "
                  f"vs truth {c['true']:+.3f} -> {100 * c['recovered']:.0f} % recovered" if c['recovered'] is not None else
                  f"  pattern {reg['pattern']}: regions have no true contrast")
    im = res.get('image_insp')
    if im:
        print("  inspiration image vs truth: " + ', '.join(f"{k} NCC {d['ncc']:.4f}" for k, d in im.items() if isinstance(d, dict)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
