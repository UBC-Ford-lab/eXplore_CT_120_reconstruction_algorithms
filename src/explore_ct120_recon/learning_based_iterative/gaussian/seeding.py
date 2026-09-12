"""Where the primitives start.

Splatting is far more sensitive to initialisation than a voxel grid is: a voxel
exists everywhere whether or not anything put it there, while a Gaussian only
represents matter someone placed a primitive near. Adaptive density control can
subdivide what exists, but it cannot conjure primitives in a region that got
none — so the seed decides which parts of the specimen the model is even able
to represent.

WHY MASS-PROPORTIONAL, NOT THRESHOLDED
--------------------------------------
The obvious seeding rule — take the voxels above some density threshold and
sample uniformly among them — concentrates the cloud on bone and drops
everything faint. MEASURED on Scan_1510, seeding at a -800 HU threshold:

    discarded 40.5 % of the volume's total attenuation
    of that, 37.1 points were OUTSIDE the export ROI (bed, peripheral tissue)
    seed carried 61.1 % of its mass inside the ROI against a truth of 39.7 %

The scan bed and the specimen's periphery are faint but LONG, so they carry a
large share of every line integral. Starved of primitives there, the optimiser
does the only thing it can: it inflates the interior to explain attenuation
that physically happens outside it. Sampling proportional to mass instead of
above a threshold put the seed at 46.6 % inside — the bed gets primitives in
proportion to what it actually contributes.

The reference volume is whatever the caller passes — an FDK is the natural
choice because it is cheap and already in the pipeline, but nothing here knows
that. With no reference at all, seeding falls back to uniform-in-domain, which
trains but converges much more slowly.

AND WHY THE WIDTH MATTERS AS MUCH AS THE PLACEMENT
---------------------------------------------------
A cloud of Gaussians of width sigma represents exactly ``G_sigma * nu`` for a
non-negative measure nu, so sigma is a modulation transfer function
(``exp(-2 pi^2 sigma^2 f^2)``) and not a soft preference: a frequency it
suppresses is outside the model's range, whatever the loss does. Densities are
softplus-positive, so there is no cancellation to sharpen with either. And
sigma moves SLOWLY: it is optimised in log space at 5x the base LR, ~5e-4 nats
per step at the default 1e-4, so a 600-iteration run moved the median by 0.007
mm while a 5 000-iteration one at 10x the LR moved it 0.845 -> 0.488 mm and grew
the anisotropy from 1.00 to 1.62. The seed width therefore IS the delivered
resolution of a short run and the starting point of a long one;
`local_spacing` is where it is set, and the LR is what decides how far
from it the run gets.
"""
from __future__ import annotations

import math

import numpy as np


def seed_from_volume(volume, *, geometry, domain, world_scale, n_points,
                     noise_floor_quantile: float = 0.5, block: int = 2,
                     scale_factor: float = 1.0, rng=None, verbose: bool = True,
                     roi_box_mm=None, roi_weight: float = 1.0,
                     edge_weight: float = 0.0, edge_sigma_vox: float = 1.0):
    """Mass-proportional seed from a reference volume of mu in mm^-1.

    ``volume`` is (Nx, Ny, Nz) on the grid ``geometry`` describes. Returns
    ``(xyz, scaling, rotation, density)`` in NORMALISED world units, ready for
    `GaussianCloud`.

    ``roi_box_mm`` = (lo, hi) in mm, with ``roi_weight`` > 1, multiplies the
    sampling weight inside that box. The count is the resolution budget (one
    primitive per resolution cell — the blob-basis rule, Matej & Lewitt 1996),
    and a mass-proportional seed spends half of it outside the export ROI on
    the bed and the truncated periphery (MEASURED on Scan_1510: 400 k of
    800 k). Those regions still need primitives, since their shadows are in
    every projection, but not at the ROI's resolution.

    ``edge_weight`` = f in [0, 1] takes that share of the sampling weight from
    the reference's EDGE STRENGTH instead of its mass: |grad| of the volume
    smoothed by ``edge_sigma_vox`` voxels, with its median subtracted (the
    noise gradient of flat regions and air, the same floor idea as the mass
    floor). f = 0 is the mass-proportional seed above. WHY: the count is the
    resolution budget and mass spends it where the attenuation is, not where
    the structure is. MEASURED on Scan_1510 (2.4 M fixed cloud, prod recipe):
    44 primitives per 1,000 voxels in flat soft tissue against 24 at bone
    edges (|grad| 1.6-3.2 kHU/mm), because an edge is thin and carries little
    mass — the seed is densest exactly where nothing needs resolving, which is
    where the speckle comes from, and thinnest where the blur is. Growing the
    cloud did not fix it: gated adaptive densification 2.4 -> 5 M multiplied
    every edge-strength bin by the same 2.1-2.4x. The ROI multiplier applies
    to the mixed weight; the density calibration still conserves the mass.
    """
    rng = np.random.default_rng() if rng is None else rng
    vol = np.asarray(volume, dtype=np.float32)
    nx, ny, nz = vol.shape
    dx = float(geometry['dx'])
    dz = float(geometry['dz'])

    # A noise floor, subtracted before it becomes a sampling weight. Without it
    # every air voxel is a lottery ticket and most of the cloud lands in air —
    # the reconstruction's noise is not a place to put primitives. The quantile
    # is over the whole volume, so it adapts to how much air a scan contains
    # instead of assuming an HU value.
    sub = vol[::3, ::3, ::3].ravel()
    floor = float(np.quantile(sub, noise_floor_quantile))
    weight = np.maximum(vol - floor, 0.0)
    f = float(edge_weight)
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"edge_weight must be in [0, 1], got {f}")
    edge_note = ''
    mass_weight = weight
    if f > 0.0:
        weight, edge_note = _mix_edge_weight(weight, vol, f=f, dx=dx, dz=dz,
                                             sigma_vox=float(edge_sigma_vox))
    if roi_box_mm is not None and float(roi_weight) != 1.0:
        ox, oy, oz = (float(v) for v in geometry.get('vol_origin',
                                                     (0.0, 0.0, 0.0)))
        lo, hi = (np.asarray(v, dtype=np.float64) for v in roi_box_mm)
        cx = (np.arange(nx) - (nx - 1) / 2.0) * dx + ox
        cy = (np.arange(ny) - (ny - 1) / 2.0) * dx + oy
        cz = (np.arange(nz) - (nz - 1) / 2.0) * dz + oz
        inside = ((cx >= lo[0]) & (cx <= hi[0]))[:, None, None] \
            & ((cy >= lo[1]) & (cy <= hi[1]))[None, :, None] \
            & ((cz >= lo[2]) & (cz <= hi[2]))[None, None, :]
        weight = np.where(inside, weight * float(roi_weight), weight)
        if f > 0.0:
            mass_weight = np.where(inside, mass_weight * float(roi_weight),
                                   mass_weight)
    # The density calibration (`_finish`) targets the ROI-WEIGHTED mass of
    # the reference, as it always has (the multiplier was applied before the
    # sum): with an edge term the sampling weight is a unit-sum mixture and
    # says nothing about mass, so the target is kept on the mass weight alone
    # and is the same number whatever f is — the edge term moves primitives,
    # it does not change what they add up to.
    mass_mm = float((mass_weight if f > 0.0 else weight).sum()) * dx * dx * dz

    # Block-reduce before sampling. np.random.choice with an explicit p over
    # every voxel materialises a float64 probability vector: on a 909 M-voxel
    # domain that is 7 GB and it will not run. Reducing by 2^3 first and then
    # jittering uniformly inside the chosen block is the same distribution to
    # within one block, at 1/8 the memory.
    b = int(block)
    nb = [max(1, d // b) for d in weight.shape]
    trimmed = weight[:nb[0] * b, :nb[1] * b, :nb[2] * b]
    blocks = trimmed.reshape(nb[0], b, nb[1], b, nb[2], b).sum(axis=(1, 3, 5))
    flat = blocks.ravel().astype(np.float64)
    total = flat.sum()
    if total <= 0:
        if verbose:
            print("  seeding: reference volume carries no mass above the "
                  "noise floor — falling back to uniform")
        return seed_uniform(domain=domain, world_scale=world_scale,
                            n_points=n_points, rng=rng,
                            scale_factor=scale_factor)
    flat /= total

    pick = rng.choice(flat.size, size=int(n_points), replace=True, p=flat)
    bi = np.stack(np.unravel_index(pick, blocks.shape), axis=-1)
    idx = bi * b + rng.integers(0, b, size=bi.shape)
    idx = np.minimum(idx, np.array([nx - 1, ny - 1, nz - 1]))

    # Voxel index -> mm, with the grid's own origin, then -> normalised world.
    ox, oy, oz = (float(v) for v in geometry.get('vol_origin', (0.0, 0.0, 0.0)))
    pos_mm = np.stack([
        (idx[:, 0] - (nx - 1) / 2.0) * dx + ox,
        (idx[:, 1] - (ny - 1) / 2.0) * dx + oy,
        (idx[:, 2] - (nz - 1) / 2.0) * dz + oz,
    ], axis=-1)
    # Jitter within the voxel so the seed is not a lattice; a lattice biases
    # the first few hundred iterations toward axis-aligned structure.
    pos_mm += (rng.random(pos_mm.shape) - 0.5) * np.array([dx, dx, dz])

    return _finish(pos_mm, mass_mm, world_scale=world_scale,
                   n_points=int(n_points), rng=rng, scale_factor=scale_factor,
                   verbose=verbose,
                   note=f"mass-proportional from a {nx}x{ny}x{nz} reference "
                        f"(noise floor q={noise_floor_quantile:g} -> "
                        f"{floor:.5f} mm^-1){edge_note}")



def _mix_edge_weight(weight, vol, *, f: float, dx: float, dz: float,
                     sigma_vox: float):
    """``(1 - f) * mass + f * edge``, each normalised to unit sum.

    ``edge`` is |grad| of the Gaussian-smoothed reference (mm^-1 per mm,
    anisotropic voxels honoured) with its median over the volume subtracted
    and clipped at zero: the median is the noise gradient of air and flat
    tissue, so what remains is structure. Returns the mixed weight and a note
    for the seeding line. A reference with no edge above the floor keeps the
    mass weight untouched.
    """
    from scipy import ndimage
    sm = ndimage.gaussian_filter(vol, sigma_vox) if sigma_vox > 0 else vol
    g = np.zeros_like(vol)
    for axis, h in enumerate((dx, dx, dz)):
        d = np.gradient(sm, h, axis=axis)
        g += d * d
    g = np.sqrt(g, out=g)
    gfloor = float(np.median(g[::3, ::3, ::3]))
    edge = np.maximum(g - gfloor, 0.0)
    etot = float(edge.sum())
    mtot = float(weight.sum())
    if etot <= 0.0 or mtot <= 0.0:
        return weight, ' (edge weight: no edge above the floor, mass only)'
    mixed = (1.0 - f) * (weight / mtot) + f * (edge / etot)
    return mixed, (f' + edge-weighted f={f:g} (|grad| floor '
                   f'{gfloor:.4g} mm^-2, sigma {sigma_vox:g} vox)')

def seed_uniform(*, domain, world_scale, n_points, rng=None,
                 scale_factor: float = 1.0, mass_mm: float | None = None,
                 verbose: bool = True):
    """Uniform in the model domain. The no-reference fallback."""
    rng = np.random.default_rng() if rng is None else rng
    lo = domain.aabb_min.cpu().numpy().astype(np.float64)
    hi = domain.aabb_max.cpu().numpy().astype(np.float64)
    pos_mm = lo + rng.random((int(n_points), 3)) * (hi - lo)
    if domain.radius_xy is not None:
        cx, cy = domain.center_xy
        keep = ((pos_mm[:, 0] - cx) ** 2 + (pos_mm[:, 1] - cy) ** 2
                <= float(domain.radius_xy) ** 2)
        # Resample the rejected ones rather than returning fewer than asked.
        while not keep.all():
            n_bad = int((~keep).sum())
            pos_mm[~keep] = lo + rng.random((n_bad, 3)) * (hi - lo)
            keep = ((pos_mm[:, 0] - cx) ** 2 + (pos_mm[:, 1] - cy) ** 2
                    <= float(domain.radius_xy) ** 2)
    if mass_mm is None:
        # 2 % of the domain filled with soft tissue: a deliberately weak prior,
        # only there to put the density on the right order of magnitude.
        vol_mm3 = float(np.prod(hi - lo))
        mass_mm = 0.02 * vol_mm3 * 0.022
    return _finish(pos_mm, mass_mm, world_scale=world_scale,
                   n_points=int(n_points), rng=rng, scale_factor=scale_factor,
                   verbose=verbose, note="uniform in the model domain")


#: Mean distance to the 3 nearest neighbours, in units of n^(-1/3), for a
#: homogeneous Poisson process. From E[d_j] = (Gamma(j+1/3)/Gamma(j)) *
#: (4 pi n / 3)^(-1/3) averaged over j = 1, 2, 3; its reciprocal converts a
#: measured kNN distance into the mean spacing the old global formula meant.
#: Verified against 200 k simulated points (1.383 measured against 1.3926
#: closed-form; the gap is the cube's boundary, which has no analogue here
#: because the cloud has no hard edge).
KNN_TO_SPACING = 1.3926

#: kNN neighbours averaged per primitive. Upstream 3DGS uses the same three.
KNN_K = 3


def local_spacing(pos, *, verbose=False):
    """Local mean spacing at each point, or None if it cannot be measured.

    Units are whatever ``pos`` is in — mm at seeding time, normalised world
    units when `GaussianCloud.update_spacing_cap` calls it during training.
    A distance carries its input's frame, so nothing here needs to know
    which frame that is.

    THE BUG THIS REPLACES. The width used to be one global number,
    ``(prod(bounding-box span) / N)^(1/3)``. That IS the mean spacing — for a
    UNIFORM seed. The default seed is mass-proportional, which SPANS the whole
    domain (the scan bed reaches the edges) while OCCUPYING almost none of it,
    so the box is the wrong volume by the concentration ratio. MEASURED on
    Scan_1510's own FDK seed, 80 k primitives: the formula returned 1.737 mm
    where the points' actual density implies 0.874 mm — 1.99x too wide, and
    every primitive got it.

    That number is not a detail. A cloud of Gaussians of width sigma can only
    represent ``G_sigma * (non-negative measure)`` AT ANY ONE MOMENT, so sigma
    is literally a modulation transfer function, ``exp(-2 pi^2 sigma^2 f^2)``.

    How long it stays the answer is a question about the SCHEDULE, and the two
    regimes are far apart. Sigma is optimised in LOG space at 5x the base LR, so
    Adam covers ~5e-4 nats per step at the default 1e-4 and a 2x correction
    needs ~1400 consistently-signed steps. MEASURED on Scan_1510 ds6, same
    config, median narrowest axis:

        600 iterations, lr 1e-4     0.845 -> 0.949 mm   (grew; nothing moved)
        5 000 iterations, lr 1e-4   0.845 -> 0.627 mm   anisotropy 1.00 -> 1.22
        5 000 iterations, lr 1e-3   0.845 -> 0.488 mm   anisotropy 1.00 -> 1.62

    So the seed dominates a SHORT run and is only a starting point in a long
    one — but it is still where a long run starts from, and halving it is worth
    roughly the same as an order of magnitude of learning rate.

    Per-primitive rather than one global value, because the correct answer is
    not global: the bed is sparse and the skeleton is dense, and a single sigma
    is simultaneously too fine for one and too coarse for the other. This is
    also what upstream 3DGS does (its `simple-knn` exists for exactly this);
    the "no kNN dependency" the old docstring claimed as a virtue is what
    introduced the error.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:                       # pragma: no cover - scipy is a dep
        if verbose:
            print("    (scipy unavailable: falling back to the bounding-box "
                  "spacing, which OVER-SIZES a concentrated seed ~2x)")
        return None
    n = len(pos)
    k = min(KNN_K, n - 1)
    if k < 1:
        return None
    d, _ = cKDTree(pos).query(pos, k=k + 1, workers=-1)
    spacing = KNN_TO_SPACING * d[:, 1:].mean(axis=1)
    # Mass-proportional sampling draws WITH replacement, so a heavy block can
    # be picked many times and jitter can leave near-coincident points; their
    # measured spacing tends to zero and would seed a delta function the
    # optimiser cannot recover from. Bound both tails against the cloud's own
    # median rather than an absolute length, so this transfers across scans.
    med = float(np.median(spacing))
    if not np.isfinite(med) or med <= 0:
        return None
    return np.clip(spacing, 0.1 * med, 10.0 * med)


def _finish(pos_mm, mass_mm, *, world_scale, n_points, rng, scale_factor,
            verbose, note):
    """Common tail: sizes, rotations, and a density that conserves total mass.

    Initial width is the LOCAL mean spacing at each primitive, measured by kNN
    over the points actually placed (see `local_spacing`) — so a primitive
    in the skeleton starts narrow and one in the bed starts wide, with no
    scan-specific constant. The scale is calibrated so that a uniform seed
    reproduces the global ``(volume / N)^(1/3)`` this replaced; only a
    CONCENTRATED seed changes, which is the case that was wrong.

    Density is then chosen so the cloud's ANALYTIC mass equals the reference's::

        sum_i a_i (2 pi)^{3/2} |Sigma_i|^{1/2} == mass

    which for one shared ``a`` is still a single division, now over the sum of
    sigma_i^3 rather than N sigma^3. Getting this right at step zero matters
    more than it looks: the optimiser reaches a given total attenuation far
    faster by moving primitives than by re-scaling every density, so a seed that
    is an order of magnitude off spends its first thousand iterations undoing
    that instead of fitting structure.
    """
    sigma_mm = local_spacing(pos_mm, verbose=verbose)
    if sigma_mm is None:
        span = pos_mm.max(axis=0) - pos_mm.min(axis=0)
        occupied = float(np.prod(np.maximum(span, 1e-6)))
        sigma_mm = np.full(n_points,
                           (occupied / max(1, n_points)) ** (1.0 / 3.0))
    sigma_mm = np.asarray(sigma_mm, dtype=np.float64) * float(scale_factor)

    xyz = pos_mm * world_scale
    scaling = np.repeat((sigma_mm * world_scale)[:, None], 3,
                        axis=1).astype(np.float32)
    rotation = np.zeros((n_points, 4), dtype=np.float32)
    rotation[:, 0] = 1.0

    # One shared amplitude, over a cloud whose widths now differ: the analytic
    # mass is a_i summed over (2 pi)^{3/2} sigma_i^3, so the division is over
    # that sum and not over N sigma^3.
    per_unit = (2.0 * math.pi) ** 1.5 * float(np.sum(sigma_mm ** 3))
    a = float(mass_mm) / max(1e-12, per_unit)
    density = np.full((n_points, 1), max(a, 1e-8), dtype=np.float32)

    if verbose:
        q = np.percentile(sigma_mm, [5, 50, 95])
        print(f"  seeding: {n_points:,} Gaussians, {note}")
        print(f"    initial sigma (local kNN spacing) median {q[1]:.4f} mm, "
              f"5-95% {q[0]:.4f}-{q[2]:.4f} mm")
        print(f"    density {a:.5f} mm^-1, total mass {mass_mm:.4g} mm^2")
    return (xyz.astype(np.float32), scaling, rotation, density)
