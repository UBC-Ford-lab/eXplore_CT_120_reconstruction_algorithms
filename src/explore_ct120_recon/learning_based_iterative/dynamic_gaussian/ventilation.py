"""Regional ventilation straight from the projections.

WHAT IS MEASURED. Lung parenchyma is tissue plus air. Between expiration
and inspiration the tissue mass of a parcel stays put and air fills the
extra volume, so a parcel that grows by the factor J = det F has its
attenuation divided by J. In 4DCT ventilation imaging that single fact is
read two ways after a deformable registration of the two images — from the
registration's Jacobian, or from the HU change — and the two famously
disagree (Castillo et al. 2010; the VAMPIRE challenge, Kipritidis et al.
2019), because a finite-difference Jacobian of a noisy field is unstable
and nothing forces a registration's field and the images' intensities to
agree. Here there is no registration and no disagreement to have: the
mass-preserving motion model (`motion`) transports every primitive's shape
by F and divides its amplitude by |det F|, so the Jacobian and the density
readings are the same number per primitive by construction, J is analytic
from a smooth trilinear field, the lung carries its own field (the motion
parts) and the field was estimated from the projections together with the
volume. The specific ventilation of primitive i, tissue following, is

    SV_i = (J_i - 1) / f_air,i,      J_i = |det F_i(insp)| / |det F_i(exp)|

with f_air the air fraction at its EXPIRATION position (-HU/1000, the
reference frame of the CT-ventilation literature, so the numbers compare).
The map is the mass-weighted mean of SV over the primitives that cover a
voxel, splatted in the expiration state through the same voxeliser that
renders the volume.

WHAT IS CHECKED, with no ground truth needed:
  global   the field's volume change over the lung parcels, sum (J - 1) dV,
           against those SAME parcels' air-volume change between the two
           rendered gate states, which depends on the densities alone (the
           thorax's total air change is reported beside it; a quarter of it
           sits in boundary parcels whose centres lie outside the lung
           window, and the field's own expansion of those boundary parcels
           is where it errs, MEASURED on 90pu264k); and against the vendor gates
           (a validation script outside this package, which also does the
           independent-data check: density ventilation from the VENDOR
           volumes with this field for correspondence only).
  sign     J > 1 across the lung between expiration and inspiration; the
           fraction of parcels that go the other way is reported.
  gradient the linear trend of SV along each axis, for the posture check.

WHICH GATE IS EXPIRATION is not assumed: the two gate states are rendered
and the one holding LESS air in the thorax is the reference. On Scan_1510
that is gate 1 (MEASURED: lung mean HU -667 at gate 0, -619 at gate 1).

RESOLUTION. J varies at the scale of the control grid (1 mm by default), so
the map is regional. MEASURED on the Scan_1510 digital phantoms: the core of
a lung reads within 2 %, a 0.15 left-right contrast is read at 90 %, the
tissue's spurious strain is 0.02 rms (2026-09-23), and on the texture
phantom (2026-09-26) the map's contrast transfer (gain of the recovered J
against the truth per spatial period) is 0.84 for broad features, one half
at a 2.0 mm period (MTF-50, the Nyquist period of the 1 mm grid) and one
tenth at 1.1 mm (MTF-10); below 1.2 mm what remains is noise (coherence
< 0.5). The map is therefore written at an isotropic 0.5 mm pitch by
default: half the MTF-10, the coarsest grid that still represents every
period the map carries (99.9 % of its coherent signal when sampled; a
0.15 mm export is 37x the storage for noise). A cube average of side b
halves contrast at the period 1.66 b, so the block for a resolution-matched
comparison is 0.6 x MTF-50 = 1.2 mm. The phantom scorer keeps 0.15 mm so
it can sample the truth without adding blur of its own.

GATE MEANS AND PEAKS. The two gate states are the MEAN state of each gate's
views, because one code per gate fits views whose breathing level scatters
within the gate (on the phantom the gate means sit at 0.92 / 0.08 of the
breath, so the gate-mean J is 84 % of peak-to-peak; the vendor's gated
reading is the same mean quantity). The free code of a view, projected onto
the gate field, is that view's own breathing LEVEL (`breathing_levels`;
r 0.93 against the phantom's per-view truth, `breathing_trace` in the
record); the most-inspired / most-expired tenth of each gate's views define
the PEAK states and a second, peak-to-peak reading (`peak`, the `J_peak`
map). Phantom: peak core 1.215 vs the J(1)/J(0) truth 1.233.

TIDAL VOLUME. The lung-parcel integral of J - 1 misses what the smoothing
pushes just outside the lung window (67 % of the phantom's total); the
all-parcel integral over the window dilated by `DILATE_MM` (the band)
encloses the smoothed source: 89 % with the default recipe, 96 % without the
bending prior, 102 % at the 3 mm grid — the prior's cost sits at the edge.
Both are in `tidal_volume_mm3`; the band is the tidal volume.

The lung is selected BY POSITION (a cleaned HU window on the rendered
reference state, sampled at each primitive's position), never by the
primitive's amplitude — a peak value is not a local mu (the trap recorded
in the mass-preservation A/B). With motion parts the overlap between that
selection and the learned lung part is reported as a diagnostic.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import torch
from scipy import ndimage

from . import export4d
from .export4d import (LinearHU, coarse_geometry, gate_code, nrrd_space_positive,
                       _nrrd_vec, read_nrrd_header)

LUNG_HU = (-900.0, -250.0)          # the parenchyma window on the rendered state
MIN_LUNG_MM3 = 20.0                 # components below this are not lung
THORAX_MARGIN_MM = 3.0              # around the union of the two lung masks
F_AIR_FLOOR = 0.05                  # SV is undefined in a parcel with no air
RING_HU = -400.0                    # a lung's surroundings are tissue, not air
MIN_FRAC_OF_LARGEST = 0.2           # gut gas pockets are small next to a lung
CHUNK = 500_000


# ------------------------------------------------------------ the physics --
DILATE_MM = 1.0        # band around the lung window for the all-parcel tidal volume (~ the map's smoothing)
PEAK_FRAC = 0.1        # share of each gate's views that defines a peak state


def specific_ventilation(J, f_air, floor: float = F_AIR_FLOOR):
    """(J - 1) / f_air with the air fraction floored: a parcel that holds no
    air cannot ventilate, and dividing by ~0 would only report noise."""
    J = np.asarray(J, dtype=np.float64)
    f = np.clip(np.asarray(f_air, dtype=np.float64), floor, 1.0)
    return (J - 1.0) / f


def air_fraction(hu):
    """-HU / 1000 clipped to [0, 1]: air -1000, tissue 0."""
    return np.clip(-np.asarray(hu, dtype=np.float64) / 1000.0, 0.0, 1.0)


#: The air gate's ramp (see `motion`): a parcel is fully air-bearing (free
#: to change volume) at the lung window's upper edge and below, fully solid
#: at -100 HU (fat) and above; between, the fraction.
GATE_HU = (-100.0, LUNG_HU[1])


def air_bearing(hu, ramp_hu=GATE_HU) -> np.ndarray:
    """The motion model's per-parcel air-bearing value in [0, 1] from HU
    (`GATE_HU`): 1 inside the lung window, 0 in fat and denser tissue. The
    gate at a control node is the membership-weighted mean of this over the
    parcels in the node's cell (`MotionModel.air_gate_from_cloud`), so
    inside the lung the field is what it was without the gate — the
    constraint is on tissue, not a rescaling of the lung's own expansion
    (which would multiply the lung's effective smoothness prior by
    1/f_air^2)."""
    solid, free = float(ramp_hu[0]), float(ramp_hu[1])
    return np.clip((solid - np.asarray(hu, dtype=np.float64)) / (solid - free), 0.0, 1.0)


def in_box(geom: dict, xyz_mm: np.ndarray) -> np.ndarray:
    """(N,) True where a position lies inside the rendered box of ``geom``."""
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    half = np.array([nx * geom['dx'], ny * geom['dx'], nz * geom['dz']]) / 2.0
    return (np.abs(np.asarray(xyz_mm, dtype=np.float64)
                   - np.asarray(geom['vol_origin'], dtype=np.float64)) <= half).all(axis=1)


def density_ventilation_J(hu_exp, hu_insp):
    """The density reading of the same J, for the independent-data check:
    tissue volume conserved, (1 - f_exp) V_exp = (1 - f_insp) V_insp, so
    J = (1000 + HU_exp) / (1000 + HU_insp)."""
    a = np.clip(1000.0 + np.asarray(hu_exp, dtype=np.float64), 1.0, None)
    b = np.clip(1000.0 + np.asarray(hu_insp, dtype=np.float64), 1.0, None)
    return a / b


# --------------------------------------------------------------- the lung --
def lung_mask(hu: np.ndarray, pitch_mm: float, window=LUNG_HU,
              min_mm3: float = MIN_LUNG_MM3, ring_hu: float = RING_HU,
              min_frac_of_largest: float = MIN_FRAC_OF_LARGEST) -> np.ndarray:
    """Parenchyma in a rendered HU volume: the window, opened by one voxel,
    keeping every connected component above ``min_mm3`` WHOSE SURROUNDINGS
    ARE TISSUE (the mean HU of a two-voxel ring around it above ``ring_hu``)
    and that is at least ``min_frac_of_largest`` of the largest such
    component. The lungs sit inside the chest wall, heart and diaphragm; the
    other things in the window on a mouse scan do not, or are small — the
    partial-volume shell of the body outline is ringed by half air, the bed
    foam by air and plastic (MEASURED on the v2 run: without the ring test
    the "lung" was 15 cm^3, the outline shell plus the foam), and a gut gas
    pocket passes the ring test but is a small fraction of a lung."""
    m = (hu > window[0]) & (hu < window[1])
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1)) * float(pitch_mm) ** 3
    keep, kept_sizes = [], []
    for k in np.flatnonzero(sizes >= min_mm3) + 1:
        comp = lab == k
        ring = ndimage.binary_dilation(comp, iterations=2) & ~comp
        if ring.any() and float(hu[ring].mean()) > ring_hu:
            keep.append(k)
            kept_sizes.append(float(sizes[k - 1]))
    if not keep:
        return np.zeros_like(m)
    biggest = max(kept_sizes)
    keep = [k for k, sz in zip(keep, kept_sizes) if sz >= min_frac_of_largest * biggest]
    return np.isin(lab, keep)


def thorax_region(masks, pitch_mm: float, margin_mm: float = THORAX_MARGIN_MM) -> np.ndarray:
    """The union of the lung masks dilated by ``margin_mm``: where the air is
    counted per state, so a state's air is not missed because its lung moved."""
    u = np.zeros_like(masks[0])
    for m in masks:
        u |= m
    return ndimage.binary_dilation(u, iterations=max(1, int(round(margin_mm / pitch_mm))))


def air_volume_mm3(hu: np.ndarray, region: np.ndarray, pitch_mm: float) -> float:
    return float(air_fraction(hu[region]).sum() * float(pitch_mm) ** 3)


def sample(volume: np.ndarray, geom: dict, xyz_mm: np.ndarray, order: int = 1) -> np.ndarray:
    """Trilinear (order 1) or nearest (0) values of a voxeliser-layout
    (nx, ny, nz) volume at positions in mm; outside the grid: 0."""
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    dx, dz = float(geom['dx']), float(geom['dz'])
    ox, oy, oz = (float(v) for v in geom['vol_origin'])
    idx = np.stack([(xyz_mm[:, 0] - ox) / dx + (nx - 1) / 2.0,
                    (xyz_mm[:, 1] - oy) / dx + (ny - 1) / 2.0,
                    (xyz_mm[:, 2] - oz) / dz + (nz - 1) / 2.0])
    return ndimage.map_coordinates(volume.astype(np.float32), idx, order=order,
                                   mode='constant', cval=0.0)


# ---------------------------------------------------------- the geometry --
class _Replaced:
    """A displaced cloud whose amplitudes are ``density`` (N,) instead of its
    own: the voxeliser then sums that quantity over the primitives' footprints
    — with ones, the footprint count itself (the SPATIAL share of a voxel)."""

    def __init__(self, shown, density: torch.Tensor):
        self._shown = shown
        self._density = density.to(shown.density.dtype)

    @property
    def density(self):
        return self._density.reshape(-1, 1)

    def __len__(self):
        return len(self._shown)

    def __getattr__(self, name):
        if name in ('_shown', '_density'):
            raise AttributeError(name)
        return getattr(self._shown, name)


class _Weighted:
    """A displaced cloud with its amplitudes multiplied by ``weight`` (N,):
    what the voxeliser sees when it splats a per-primitive quantity."""

    def __init__(self, shown, weight: torch.Tensor):
        self._shown = shown
        self._weight = weight.to(shown.density.dtype)

    @property
    def density(self):
        return self._shown.density * self._weight.unsqueeze(-1)

    def __len__(self):
        return len(self._shown)

    def __getattr__(self, name):
        if name in ('_shown', '_weight'):
            raise AttributeError(name)
        return getattr(self._shown, name)


@torch.no_grad()
def jacobian_dets(motion, xyz: torch.Tensor, code: torch.Tensor, chunk: int = CHUNK):
    """(|det F| (N,), displacement (N, 3)) of the rest positions under one
    full code, in chunks, with the model's membership."""
    parts = motion.membership()
    dets, disp = [], []
    for i in range(0, xyz.shape[0], chunk):
        sl = slice(i, i + chunk)
        d, F = motion.displacement(xyz[sl], code, need_jacobian=True,
                                   parts=None if parts is None else parts[sl])
        dets.append(torch.linalg.det(F).abs())
        disp.append(d)
    return torch.cat(dets), torch.cat(disp)


@torch.no_grad()
def _displacement(motion, xyz: torch.Tensor, code: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    parts = motion.membership()
    out = []
    for i in range(0, xyz.shape[0], chunk):
        sl = slice(i, i + chunk)
        out.append(motion.displacement(xyz[sl], code, parts=None if parts is None else parts[sl]))
    return torch.cat(out)


@torch.no_grad()
def breathing_levels(motion, xyz: torch.Tensor, sel: np.ndarray, max_points: int = 40000):
    """Every view's breathing LEVEL on the gate axis, in gate units (-1 =
    the mean state of gate 0, +1 = gate 1's): ``fixed + s . free_code``,
    with ``s`` the projection of each free basis column's displacement onto
    the gate field's over the parcels ``sel`` (the lung). The free code is
    what the model learned about a view beyond its gate; the part of it
    that lies along the breathing direction is the view's own depth of
    breath. MEASURED on the Scan_1510 digital phantom (2026-09-23, three
    runs): r 0.93 against the per-view truth. Returns (levels (n_views,),
    s (n_codes,)); None without a gate axis."""
    if motion.fixed is None or not motion.n_fixed:
        return None
    idx = np.flatnonzero(np.asarray(sel))
    if idx.size == 0:
        idx = np.arange(int(xyz.shape[0]))
    if idx.size > max_points:
        idx = idx[np.linspace(0, idx.size - 1, max_points).astype(int)]
    pts = xyz[torch.as_tensor(idx, device=xyz.device)]
    d0 = _displacement(motion, pts, gate_code(motion, 0.0))
    dg = _displacement(motion, pts, gate_code(motion, 1.0)) - d0
    den = float((dg * dg).sum().clamp_min(1e-12))
    proj = []
    for k in range(int(motion.n_codes)):
        free = torch.zeros(int(motion.n_codes))
        free[k] = 1.0
        dk = _displacement(motion, pts, gate_code(motion, 0.0, free)) - d0
        proj.append(float((dk * dg).sum()) / den)
    fixed = motion.fixed[:, 0].detach().cpu().numpy().astype(np.float64)
    if getattr(motion, 'amp', None) is not None:
        fixed = fixed + motion.amp[:, 0].detach().cpu().numpy()
    codes = motion.codes.detach().cpu().numpy().astype(np.float64)
    return fixed + codes @ np.asarray(proj, dtype=np.float64), np.asarray(proj, dtype=np.float64)


def gradient_fit(values: np.ndarray, xyz_mm: np.ndarray) -> dict:
    """SV ~ a + b . x by least squares: the slope per mm along each axis,
    and the rms of SV the trend explains."""
    if values.size < 8:
        return {'slope_per_mm': [0.0, 0.0, 0.0], 'explained_frac': 0.0}
    A = np.concatenate([np.ones((values.size, 1)), xyz_mm], axis=1)
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    fit = A @ coef
    v = values - values.mean()
    expl = 1.0 - float(((values - fit) ** 2).sum() / max((v * v).sum(), 1e-12))
    return {'slope_per_mm': [float(c) for c in coef[1:]], 'explained_frac': float(expl)}


# -------------------------------------------------------------- the core --
@torch.no_grad()
def compute(cloud, motion, roi_geom: dict, world_scale: float, signal_scale: float,
            anchors, *, pitch_mm: float = 0.3, reference='auto',
            render_fn=None, max_voxels: int | None = None,
            min_lung_mm3: float = MIN_LUNG_MM3, dilate_mm: float = DILATE_MM,
            peak_frac: float = PEAK_FRAC, log=print) -> dict:
    """Everything: the per-primitive J / SV over the lung, the SV and J maps
    on the coarse grid in the expiration state, and the global numbers.
    ``reference``: 'auto' (the gate holding less air), 0 or 1.
    ``dilate_mm``: the band around the lung window over which ALL parcels'
    volume change is summed for the tidal volume (the map's smoothing pushes
    volume change across the boundary). ``peak_frac``: the share of each
    gate's views, by breathing level, that defines the PEAK states.
    ``render_fn(shown, geom, ws, ss, max_voxels)`` -> mu (nx, ny, nz); the
    default is the voxeliser (`export4d.render_grid`, bound at call time)."""
    if motion is None or not motion.n_fixed:
        raise ValueError("ventilation needs a gate axis (two acquisition groups)")
    if render_fn is None:
        render_fn = export4d.render_grid
    geom = coarse_geometry(roi_geom, pitch_mm)
    p = float(geom['dx'])
    ws, ss = float(world_scale), float(signal_scale)
    log(f"  ventilation: rendering the two gate states at {p:.3g} mm "
        f"({np.prod(geom['vol_shape']) / 1e6:.1f} M voxels)")
    hu, mu = {}, {}
    for g, v in ((0, -1.0), (1, 1.0)):
        shown = motion.displaced_with_code(cloud, gate_code(motion, v))
        mu[g] = render_fn(shown, geom, ws, ss, max_voxels)
        hu[g] = anchors.apply(mu[g])
    masks = {g: lung_mask(hu[g], p, min_mm3=min_lung_mm3) for g in (0, 1)}
    region = thorax_region([masks[0], masks[1]], p)
    air = {g: air_volume_mm3(hu[g], region, p) for g in (0, 1)}
    lung_vol = {g: float(masks[g].sum() * p ** 3) for g in (0, 1)}
    if reference == 'auto':
        ref = 0 if air[0] < air[1] else 1
    else:
        ref = int(reference)
    insp = 1 - ref
    log(f"  air in the thorax: gate 0 {air[0]:.1f} mm^3, gate 1 {air[1]:.1f} mm^3 -> "
        f"expiration = gate {ref}" + (" (auto)" if reference == 'auto' else ""))
    code_ref, code_insp = gate_code(motion, -1.0 if ref == 0 else 1.0), gate_code(motion, -1.0 if insp == 0 else 1.0)

    xyz = cloud.xyz.detach()
    det_ref, d_ref = jacobian_dets(motion, xyz, code_ref)
    det_insp, d_insp = jacobian_dets(motion, xyz, code_insp)
    J = (det_insp / det_ref.clamp_min(1e-6)).cpu().numpy().astype(np.float64)
    # The reading is the REALISED det F at the parcel, also under the
    # air-sourced field. A "disaggregated" reading 1 + (J - 1) / g — the
    # cell's volume change assigned to its lung fraction g — was built and
    # FALSIFIED on the air-sourced A/B run (2026-09-20): the median gate over
    # the lung parcels is 0.23, so it pushed the lung's J from 1.228 to 1.31
    # against the vendor density's 1.255, the field's volume change to 1.63x
    # the parcels' own air change (0.90 realised), and the regional
    # correlation with the vendor density from 0.61 to 0.53. The realised J
    # at the lung parcels already agrees with the independent density reading
    # because the node gate scales a dual cell's divergence while a parcel's
    # trilinear J draws on the eight corners around it, which places a mixed
    # cell's volume change at its lung parcels, not uniformly over the cell.
    # The gate is recorded per parcel (``gate``) as a diagnostic only.
    gate_at = motion.gate_at(xyz) if hasattr(motion, 'gate_at') else None
    g_prim = (gate_at.cpu().numpy().astype(np.float64) if gate_at is not None
              else np.ones_like(J))
    x_ref = ((xyz + d_ref) / ws).cpu().numpy()
    x_insp = ((xyz + d_insp) / ws).cpu().numpy()
    # The parcels are SELECTED in the inspiration state, where the lung is
    # most air-filled and the window is unambiguous, and carried to the
    # expiration frame by the field (they are the same parcels): MEASURED on
    # the v2 run, the expiration-state window caught 214 mm^3 of a lung the
    # inspiration state shows as 482 mm^3, because the denser half of the
    # expiration lung sits above -250 HU, and the field's volume change over
    # that half was half the image's.
    lung_i = sample(masks[insp].astype(np.float32), geom, x_insp, order=0) > 0.5
    hu_ref_i = sample(hu[ref], geom, x_ref, order=1)
    f_air = air_fraction(hu_ref_i)
    sv = specific_ventilation(J, f_air)
    n_lung = int(lung_i.sum())
    log(f"  {n_lung:,} lung primitives (by position in the inspiration state, followed to expiration)")

    # the maps: mass-weighted means of J and SV over the lung primitives
    shown_ref = motion.displaced_with_code(cloud, code_ref)
    w_l = torch.as_tensor(lung_i.astype(np.float32), device=xyz.device)
    den = render_fn(_Weighted(shown_ref, w_l), geom, ws, ss, max_voxels)
    numJ = render_fn(_Weighted(shown_ref, w_l * torch.as_tensor(J, dtype=torch.float32, device=xyz.device)),
                     geom, ws, ss, max_voxels)
    numS = render_fn(_Weighted(shown_ref, w_l * torch.as_tensor(sv, dtype=torch.float32, device=xyz.device)),
                     geom, ws, ss, max_voxels)
    valid = den > 0.05 * float(den.max()) if den.size and den.max() > 0 else np.zeros_like(den, bool)
    J_map = np.where(valid, numJ / np.where(valid, den, 1.0), np.nan).astype(np.float32)
    SV_map = np.where(valid, numS / np.where(valid, den, 1.0), np.nan).astype(np.float32)
    covered = valid                          # where the lung parcels are, in the expiration frame
    dV = p ** 3
    # The field's volume change is an integral over the PARCELS, not over the
    # voxels their tails touch: each voxel counts with the parcels' SPATIAL
    # share of it — the lung parcels' footprint sum over every parcel's
    # footprint sum, from two unit-amplitude renders — so a Gaussian's tail
    # does not turn into lung volume (MEASURED: with the covered voxels counted
    # whole, 861 mm^3 of "lung" for a 482 mm^3 lung) and an air-rich parcel is
    # not under-weighted for being faint (the attenuation share gave 84 mm^3
    # where the spatial share gives 91). The SAME share, per state, gives the
    # parcels' OWN air content in each state: the apples-to-apples check for
    # the field is the air change of the parcels it is integrated over, not
    # the air change of the whole thorax, a quarter of which is held by
    # boundary parcels whose centres sit outside the lung window (MEASURED on
    # 90pu264k: parcels 87 mm^3, field 91, thorax 115).
    ones = torch.ones(len(cloud), device=xyz.device)
    unit_all = {}
    unit_lung = {}
    for g_, shown_g in ((ref, shown_ref), (insp, motion.displaced_with_code(cloud, code_insp))):
        unit_all[g_] = render_fn(_Replaced(shown_g, ones), geom, ws, ss, max_voxels)
        unit_lung[g_] = render_fn(_Replaced(shown_g, w_l), geom, ws, ss, max_voxels)
    share = {g_: np.clip(np.where(unit_all[g_] > 1e-9, unit_lung[g_] / np.where(unit_all[g_] > 1e-9, unit_all[g_], 1.0), 0.0), 0.0, 1.0)
             for g_ in (ref, insp)}
    frac = np.where(valid, share[ref], 0.0)
    lung_parcel_mm3 = float(frac.sum() * dV)
    air_parcels = {g_: float((share[g_] * air_fraction(hu[g_])).sum() * dV) for g_ in (ref, insp)}
    dv_parcels = air_parcels[insp] - air_parcels[ref]
    dv_field = float(np.nansum((J_map[covered] - 1.0) * frac[covered]) * dV)
    dv_image = float(air[insp] - air[ref])

    # TIDAL VOLUME over a BAND around the lung window, all parcels: the map
    # is the truth smoothed (~0.9 mm at the 1 mm grid, MEASURED on the
    # digital phantom), which pushes part of the lung's volume change into
    # the parcels just outside the lung window; those are not lung parcels,
    # so the lung-parcel integral above misses it. Every parcel's J,
    # footprint-averaged per voxel, summed over the window dilated by
    # ``dilate_mm``, recovers it (the volume integral of div u over a region
    # that encloses the source).
    J_t = torch.as_tensor(J, dtype=torch.float32, device=xyz.device)
    unit_ref = unit_all[ref]
    J_all_map = np.where(unit_ref > 1e-9, render_fn(_Replaced(shown_ref, J_t), geom, ws, ss, max_voxels)
                         / np.where(unit_ref > 1e-9, unit_ref, 1.0), np.nan).astype(np.float32)
    n_dil = int(round(float(dilate_mm) / p))
    band = ndimage.binary_dilation(covered, iterations=n_dil) if n_dil > 0 else covered.copy()
    band &= np.isfinite(J_all_map)
    dv_band = float(np.nansum(J_all_map[band] - 1.0) * dV)
    dv_window_all = float(np.nansum(J_all_map[covered & np.isfinite(J_all_map)] - 1.0) * dV)

    # THE PEAK STATES from the per-view breathing levels (see
    # `breathing_levels`): the gate states above are the MEAN of each gate's
    # views; the most-inspired / most-expired ``peak_frac`` of the views
    # define the peak states, and their code (the gate value with the mean
    # free code of those views) gives a peak-to-peak reading, which no gated
    # method reads directly. MEASURED on the phantom: the gate means sit at
    # 0.92 / 0.08 of the breath, so the gate-mean J is 84 % of peak-to-peak.
    trace = breathing_levels(motion, xyz, lung_i)
    peak = None
    J_pk = None
    if trace is not None and int(motion.n_codes) > 0:
        levels, proj = trace
        fixed_v = motion.fixed[:, 0].detach().cpu().numpy()
        gate_of = np.where(fixed_v < 0, 0, 1)
        frozen = set(int(v) for v in (getattr(motion, 'frozen_views', None) or ()))
        toward_insp = -1.0 if insp == 0 else 1.0        # gate-axis direction of inspiration

        def extreme(gate, direction):
            views = np.array([v for v in np.flatnonzero(gate_of == gate) if v not in frozen])
            n_top = max(1, int(round(float(peak_frac) * views.size)))
            chosen = views[np.argsort(direction * levels[views])[::-1][:n_top]]
            free = motion.codes[torch.as_tensor(chosen, device=motion.codes.device)].mean(dim=0)
            return gate_code(motion, -1.0 if gate == 0 else 1.0, free), chosen, float(levels[chosen].mean())
        code_insp_pk, v_insp, L_insp = extreme(insp, toward_insp)
        code_ref_pk, v_exp, L_exp = extreme(ref, -toward_insp)
        det_ref_pk, _ = jacobian_dets(motion, xyz, code_ref_pk)
        det_insp_pk, _ = jacobian_dets(motion, xyz, code_insp_pk)
        J_pk = (det_insp_pk / det_ref_pk.clamp_min(1e-6)).cpu().numpy().astype(np.float64)
        J_pk_t = torch.as_tensor(J_pk, dtype=torch.float32, device=xyz.device)
        numJpk = render_fn(_Weighted(shown_ref, w_l * J_pk_t), geom, ws, ss, max_voxels)
        J_pk_map = np.where(valid, numJpk / np.where(valid, den, 1.0), np.nan).astype(np.float32)
        J_pk_all = np.where(unit_ref > 1e-9, render_fn(_Replaced(shown_ref, J_pk_t), geom, ws, ss, max_voxels)
                            / np.where(unit_ref > 1e-9, unit_ref, 1.0), np.nan).astype(np.float32)
        L_mean = {g: float(levels[(gate_of == g) & ~np.isin(np.arange(levels.size), list(frozen))].mean()) for g in (0, 1)}
        lung_jpk = J_pk[lung_i]
        peak = {'frac': float(peak_frac), 'n_views': int(v_insp.size),
                'level_insp': L_insp, 'level_exp': L_exp,
                'level_gate_mean': {f'gate{g}': L_mean[g] for g in (0, 1)},
                'amplitude_over_gate_mean': float((L_insp - L_exp) / (L_mean[insp] - L_mean[ref])) if abs(L_mean[insp] - L_mean[ref]) > 1e-9 else float('nan'),
                'J': {'median': float(np.median(lung_jpk)) if n_lung else float('nan'),
                      'p10': float(np.percentile(lung_jpk, 10)) if n_lung else float('nan'),
                      'p90': float(np.percentile(lung_jpk, 90)) if n_lung else float('nan')},
                'tidal_volume_mm3': {'lung_parcels': float(np.nansum((J_pk_map[covered] - 1.0) * frac[covered]) * dV),
                                     'band': float(np.nansum(J_pk_all[band] - 1.0) * dV)},
                'views': {'insp': [int(v) for v in v_insp], 'exp': [int(v) for v in v_exp]}}
    lung_j, lung_sv = J[lung_i], sv[lung_i]
    out = {
        'definition': {
            'J': 'volume ratio of a parcel between the model\'s two GATE states, mean-inspiratory '
                 'over mean-expiratory. A gated reconstruction fits one state per gate to views whose '
                 'breathing level scatters within the gate, so this is the ratio of the mean states, '
                 'not peak-to-peak (the digital phantom, 2026-09-23: gate means 0.92 / 0.08 of the '
                 'breath at a jitter of 0.1; scored against the peak states the map read 25 % low, '
                 'against the gate means within 2 %). The vendor\'s gated reading is the same quantity.',
            'tidal_volume_mm3': 'lung_parcels: sum of (J - 1) x volume over the LUNG PARCELS (air-bearing '
                                'primitives inside the lung window, = delta_volume_field_mm3). band: every '
                                'parcel\'s J, footprint-averaged per voxel, summed over the lung window '
                                'dilated by dilate_mm — the volume integral that encloses the smoothed source '
                                '(the map is the truth smoothed by ~0.9 mm at the 1 mm grid, MEASURED on the '
                                'Scan_1510 digital phantom).',
            'peak': 'the same readings between the PEAK states: the gate value with the mean free code of '
                    'the most-inspired / most-expired peak.frac of each gate\'s views by breathing level '
                    '(breathing_trace). Peak-to-peak, where the gate states are the gate MEANS.'},
        'tidal_volume_mm3': {'lung_parcels': dv_field, 'band': dv_band, 'window_all_parcels': dv_window_all,
                             'dilate_mm': float(dilate_mm), 'band_mm3': float(band.sum() * dV)},
        'peak': peak,
        'breathing_trace': None if trace is None else {
            'levels': [float(v) for v in trace[0]], 'gate': [int(g) for g in np.where(motion.fixed[:, 0].detach().cpu().numpy() < 0, 0, 1)],
            'projection': [float(v) for v in trace[1]],
            'units': 'gate axis: -1 = the mean state of gate 0, +1 = gate 1; the view\'s free code projected onto the gate field'},
        'reference_gate': ref, 'inspiration_gate': insp,
        'pitch_mm': p, 'grid': geom,
        'n_lung_primitives': n_lung,
        'air_mm3': {f'gate{g}': air[g] for g in (0, 1)},
        'lung_mask_mm3': {f'gate{g}': lung_vol[g] for g in (0, 1)},
        'thorax_region_mm3': float(region.sum() * dV),
        'delta_air_image_mm3': dv_image,
        'delta_air_parcels_mm3': dv_parcels,
        'air_parcels_mm3': {f'gate{g_}': air_parcels[g_] for g_ in (ref, insp)},
        'delta_volume_field_mm3': dv_field,
        'field_over_parcels': (dv_field / dv_parcels) if abs(dv_parcels) > 1e-9 else float('nan'),
        'field_over_image': (dv_field / dv_image) if abs(dv_image) > 1e-9 else float('nan'),
        'covered_lung_mm3': float(covered.sum() * dV),
        'lung_parcel_mm3': lung_parcel_mm3,
        'J': {'median': float(np.median(lung_j)) if n_lung else float('nan'),
              'p10': float(np.percentile(lung_j, 10)) if n_lung else float('nan'),
              'p90': float(np.percentile(lung_j, 90)) if n_lung else float('nan'),
              'frac_below_1': float((lung_j < 1.0).mean()) if n_lung else float('nan')},
        'SV': {'median': float(np.median(lung_sv)) if n_lung else float('nan'),
               'p10': float(np.percentile(lung_sv, 10)) if n_lung else float('nan'),
               'p90': float(np.percentile(lung_sv, 90)) if n_lung else float('nan'),
               'mean_map': float(np.nanmean(SV_map[covered])) if covered.any() else float('nan')},
        'gradient': gradient_fit(lung_sv, x_ref[lung_i]) if n_lung else gradient_fit(np.zeros(0), np.zeros((0, 3))),
        'per_primitive': {'x_ref_mm': x_ref, 'x_insp_mm': x_insp, 'J': J, 'gate': g_prim, 'SV': sv,
                          'f_air': f_air, 'hu_ref': hu_ref_i, 'lung': lung_i,
                          **({} if J_pk is None else {'J_peak': J_pk})},
        'maps': {'J': J_map, 'SV': SV_map, 'valid': valid, 'lung_ref': masks[ref],
                 'hu_ref': hu[ref], 'hu_insp': hu[insp], 'J_all': J_all_map, 'band': band,
                 **({} if J_pk is None else {'J_peak': J_pk_map})},
    }
    # A QUALITY CHECK ON THE FIELD, not on the lung: solid tissue (f_air < 0.1)
    # inside the thorax cannot change volume, so its |J - 1| is the field's
    # error there. MEASURED on 90pu264k: rms 0.03, p99 0.13 — the field
    # over-expands the 0.75 mm shell just outside the lung by ~1.4 % and
    # agrees beyond it. Reported rather than penalised: a prior on det F
    # erases the ventilation signal itself (run 54hxw5ix, J median 1.245 ->
    # 0.999 with every image metric unchanged).
    if gate_at is not None and n_lung:
        out['gate_over_lung'] = {'median': float(np.median(g_prim[lung_i])),
                                 'frac_below_0.5': float((g_prim[lung_i] < 0.5).mean())}
        log(f"  air gate seen by the lung parcels: median {out['gate_over_lung']['median']:.2f}, "
            f"{100 * out['gate_over_lung']['frac_below_0.5']:.0f} % below 0.5 (diagnostic; the reading is the realised J)")
    in_region = sample(region.astype(np.float32), geom, x_ref, order=0) > 0.5
    solid = in_region & (f_air < 0.1)
    out['per_primitive']['thorax'] = in_region      # for the strain figure
    if solid.any():
        js = J[solid] - 1.0
        out['solid_tissue_strain'] = {
            'n': int(solid.sum()), 'rms': float(np.sqrt((js ** 2).mean())),
            'median': float(np.median(js)), 'p99': float(np.percentile(np.abs(js), 99))}
        log(f"  solid tissue in the thorax ({int(solid.sum()):,} parcels, f_air < 0.1): "
            f"|J - 1| rms {out['solid_tissue_strain']['rms']:.3f}, p99 "
            f"{out['solid_tissue_strain']['p99']:.3f} — the field's error where volume cannot change")
    parts = motion.membership()
    if parts is not None and n_lung:
        arg = parts.argmax(dim=1).cpu().numpy()
        out['parts_overlap'] = {f'part{j}': float((arg[lung_i] == j).mean()) for j in range(motion.n_parts)}
    log(f"  J = mean-inspiratory / mean-expiratory gate state; tidal volume {dv_field:.1f} mm^3 over the "
        f"lung parcels, {dv_band:.1f} mm^3 over the window + {float(dilate_mm):g} mm band (all parcels)")
    if peak is not None:
        log(f"  peak states from the breathing levels (top {100 * peak['frac']:.0f} % of each gate's views): "
            f"amplitude {peak['amplitude_over_gate_mean']:.2f}x the gate means; J median {peak['J']['median']:.3f}; "
            f"tidal volume {peak['tidal_volume_mm3']['lung_parcels']:.1f} (lung parcels) / "
            f"{peak['tidal_volume_mm3']['band']:.1f} mm^3 (band)")
    log(f"  J median {out['J']['median']:.3f} (p10-p90 {out['J']['p10']:.3f}-{out['J']['p90']:.3f}, "
        f"{100 * out['J']['frac_below_1']:.1f} % below 1); SV median {out['SV']['median']:.3f}; "
        f"field volume change {dv_field:.1f} mm^3 vs the parcels' own air change {dv_parcels:.1f} mm^3 "
        f"(ratio {out['field_over_parcels']:.2f}; the whole thorax's air change is {dv_image:.1f} mm^3, "
        f"a share of it held by boundary parcels outside the lung window)")
    return out


# ------------------------------------------------------------- the files --
def write_nrrd_3d(path: str, volume: np.ndarray, geom: dict, *, units: str,
                  extra_fields=None) -> str:
    """A float32 gzip NRRD in the delivered files' layout (y flipped like the
    VFF, positive steps, the Imalytics header); NaN where the map is
    undefined."""
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    if volume.shape != (nx, ny, nz):
        raise ValueError(f"volume {volume.shape} does not match the grid {(nx, ny, nz)}")
    D, o = nrrd_space_positive(geom)
    lines = ['NRRD0004', 'type: float', 'dimension: 3', 'space: left-posterior-superior',
             f'sizes: {nx} {ny} {nz}',
             'space directions: ' + ' '.join(_nrrd_vec(d) for d in D),
             'kinds: domain domain domain', 'endian: little', 'encoding: gzip',
             f'space origin: {_nrrd_vec(o)}', 'space units: "mm" "mm" "mm"',
             f'units:={units}']
    for k, v in (extra_fields or {}).items():
        lines.append(f'{k}:={v}')
    header = ('\n'.join(lines) + '\n\n').encode()
    payload = np.ascontiguousarray(volume[:, ::-1, :].astype('<f4').transpose(2, 1, 0)).tobytes()
    with open(path, 'wb') as raw:
        raw.write(header)
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=4, mtime=0) as f:
            f.write(payload)
    return path


def read_nrrd_3d(path: str) -> np.ndarray:
    """The inverse of `write_nrrd_3d`: (nx, ny, nz) in voxeliser layout."""
    fields, offset = read_nrrd_header(path)
    nx, ny, nz = fields['sizes']
    with open(path, 'rb') as f:
        f.seek(offset)
        buf = gzip.decompress(f.read())
    return np.frombuffer(buf, dtype='<f4').reshape(nz, ny, nx).transpose(2, 1, 0)[:, ::-1, :]


def export_ventilation(cloud, motion, roi_geom: dict, world_scale: float,
                       signal_scale: float, anchors, base_path: str, *,
                       pitch_mm: float = 0.3, reference='auto',
                       max_voxels: int | None = None, min_lung_mm3: float = MIN_LUNG_MM3,
                       dilate_mm: float = DILATE_MM, peak_frac: float = PEAK_FRAC,
                       log=print) -> str | None:
    """``<base>_ventilation.nrrd`` (the SV map), ``<base>_ventilation_J.nrrd``,
    ``<base>_ventilation.json`` (the numbers), ``<base>_ventilation.npz``
    (the per-primitive readings for the validation script) and
    ``<base>_ventilation.png``. Returns the map's path, None without a gate
    axis."""
    if motion is None or not motion.n_fixed:
        log("  ventilation skipped: no gate axis (single acquisition group)")
        return None
    res = compute(cloud, motion, roi_geom, world_scale, signal_scale, anchors,
                  pitch_mm=pitch_mm, reference=reference, max_voxels=max_voxels,
                  min_lung_mm3=min_lung_mm3, dilate_mm=dilate_mm, peak_frac=peak_frac, log=log)
    base = base_path[:-4] if base_path.lower().endswith('.vff') else base_path
    geom = res['grid']
    fields = {'reference gate': str(res['reference_gate']),
              'definition': 'specific ventilation (J - 1) / f_air, tissue following, expiration frame'}
    path = write_nrrd_3d(base + '_ventilation.nrrd', res['maps']['SV'], geom, units='fraction', extra_fields=fields)
    write_nrrd_3d(base + '_ventilation_J.nrrd', res['maps']['J'], geom, units='volume ratio', extra_fields=fields)
    side = {k: v for k, v in res.items() if k not in ('per_primitive', 'maps')}
    side['files'] = {'sv': os.path.basename(path), 'J': os.path.basename(base + '_ventilation_J.nrrd'),
                     'primitives': os.path.basename(base + '_ventilation.npz')}
    if res['maps'].get('J_peak') is not None:
        write_nrrd_3d(base + '_ventilation_J_peak.nrrd', res['maps']['J_peak'], geom, units='volume ratio',
                      extra_fields=dict(fields, definition='J between the PEAK states (breathing levels), expiration frame'))
        side['files']['J_peak'] = os.path.basename(base + '_ventilation_J_peak.nrrd')
    with open(base + '_ventilation.json', 'w') as f:
        json.dump(side, f, indent=2)
    np.savez_compressed(base + '_ventilation.npz', **res['per_primitive'],
                        reference_gate=res['reference_gate'], grid_shape=np.asarray(geom['vol_shape']),
                        grid_origin=np.asarray(geom['vol_origin']), pitch_mm=geom['dx'])
    try:
        fig = figure(res)
        fig.savefig(base + '_ventilation.png', dpi=130)
    except Exception as e:                       # a figure never ends a run
        log(f"  ventilation figure skipped ({type(e).__name__}: {e})")
    log(f"  ventilation map saved to: {path}")
    return path


def figure(res: dict):
    """The SV map on the expiration anatomy (coronal and axial mid-lung
    planes) and the SV histogram over the covered lung."""
    from matplotlib.figure import Figure
    from matplotlib import colors
    hu, sv, lung = res['maps']['hu_ref'], res['maps']['J'] - 1.0, res['maps']['lung_ref']
    idx = np.argwhere(lung)
    c = idx.mean(axis=0).astype(int) if idx.size else np.array(hu.shape) // 2
    nx, ny, nz = hu.shape
    fig = Figure(figsize=(15, 5.2), dpi=110)
    ax = fig.subplots(1, 3)
    vmax = float(np.nanpercentile(sv[np.isfinite(sv)], 98)) if np.isfinite(sv).any() else 1.0
    norm = colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-3))
    for a, (img, over, title) in zip(ax[:2], (
            (hu[:, c[1], :].T, sv[:, c[1], :].T, 'coronal, expiration state'),
            (hu[:, :, c[2]].T[::-1], sv[:, :, c[2]].T[::-1], 'axial, expiration state'))):
        a.imshow(img, cmap='gray', vmin=-1000, vmax=600, aspect='equal', origin='lower' if 'coronal' in title else 'upper')
        m = a.imshow(np.ma.masked_invalid(over), cmap='viridis', norm=norm, alpha=0.75, aspect='equal',
                     origin='lower' if 'coronal' in title else 'upper')
        a.set_title(f'volume expansion J - 1, {title}', fontsize=9)
        a.axis('off')
    fig.colorbar(m, ax=ax[:2], fraction=0.03, label='J - 1 (expiration -> inspiration)')
    vals = sv[np.isfinite(sv) & res['maps']['valid']]
    ax[2].hist(vals, bins=60, color='0.35')
    ax[2].axvline(0, color='k', lw=0.6)
    ax[2].set_xlabel('J - 1 over the lung parcels')
    ax[2].set_title(f"median J {res['J']['median']:.3f}; field {res['delta_volume_field_mm3']:.0f} mm$^3$ vs the "
                    f"parcels' air change {res['delta_air_parcels_mm3']:.0f} mm$^3$ (thorax {res['delta_air_image_mm3']:.0f})", fontsize=9)
    return fig


# --------------------------------------------------------------- the CLI --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('result_dir')
    ap.add_argument('--pitch', type=float, default=0.5, help='map pitch in mm (default 0.5, see RESOLUTION)')
    ap.add_argument('--reference', default='auto', help="expiration gate: auto (less air), 0 or 1")
    ap.add_argument('--out', default=None, help='base path (default <result_dir>/recon)')
    ap.add_argument('--dilate-mm', type=float, default=DILATE_MM, help='band around the lung window for the all-parcel tidal volume')
    ap.add_argument('--peak-frac', type=float, default=PEAK_FRAC, help='share of each gate\'s views defining a peak state')
    args = ap.parse_args(argv)
    from .show import load_run
    cloud, motion, geom, ws, ss, hu = load_run(args.result_dir)
    base = args.out or os.path.join(args.result_dir, 'recon')
    path = export_ventilation(cloud, motion, geom, ws, ss, LinearHU(*hu), base,
                              pitch_mm=args.pitch, reference=args.reference,
                              dilate_mm=args.dilate_mm, peak_frac=args.peak_frac)
    return 0 if path else 1


if __name__ == '__main__':
    raise SystemExit(main())
