"""The ONE place the third-party X-ray Gaussian rasteriser is touched.

Requires: xray_gaussian_rasterization_voxelization (optional dependency, built
from source) — the same standing as TIGRE for the iterative backends.

Everything else in this package deals in our own geometry dict, our own mu in
mm^-1, and our own detector convention. This module is the airlock: it holds
the import, the tensor layout the CUDA kernels expect, the two detector flips
that separate their image convention from ours, and — importantly — the one
kernel behaviour that is not a detail.

WHAT WE DEPEND ON, AND WHAT WE DO NOT
-------------------------------------
Only the compiled extension ``xray_gaussian_rasterization_voxelization``, whose
entire surface is four names and two plain-tensor entry points. None of
R2-Gaussian's Python package is imported: its ``Camera``, ``GaussianModel``,
``Scene`` and dataset readers all assume a synthetic-benchmark layout on disk
(``meta_data.json``, per-view ``.npy``, a normalised world cube) and none of
that survives contact with a real scanner. We build the matrices ourselves in
``camera.py`` from the geometry the pipeline already measured, so this backend
works on any scan the rest of the pipeline can load.

THE ALPHA CUTOFF — why this module carries a signal scale
---------------------------------------------------------
``cuda_rasterizer/forward.cu`` drops every per-Gaussian contribution below an
ABSOLUTE threshold::

    const float alpha = con_o.w * mu * exp(power);
    if (alpha < 0.00001f) continue;          // 1e-5

The voxelizer has no matching rule (its own cutoff is 1e-6, ten times lower),
so the projection and volume paths disagree. That threshold was chosen for
benchmark data whose line integrals are O(1). Ours are not: normalising the
volume into [-1, 1]^3 multiplies every line integral by ``world_scale``
(2 / domain extent, ~0.025 /mm on a 80 mm domain), so a typical rendered value
is ~4e-3 and a single primitive's share of it is at or below the cutoff.

MEASURED on Scan_1510 (2026-08-29), rasteriser output over the closed-form line
integral of the same trained mixture, as the density is scaled by k::

    k        0.1     0.3     1.0     3.0    10.0   100.0  1000.0
    ras/exact  0.020   0.407   0.777   0.917   0.969   0.988   0.990

At k=1 the rasteriser silently discards 22 % of the line integral. An optimiser
fitting through it inflates density until the SURVIVING contributions match the
data, so the delivered volume comes out ~1.3x too dense while the projections
"fit" perfectly. Nothing downstream can detect this: the fit is self-consistent.

The fix used here needs no patched kernel. ``signal_scale`` multiplies the
sinogram before training and divides mu back afterwards; the forward model is
linear, so this is exact, and it moves the whole mixture up the curve above.
``camera.resolve_signal_scale`` picks it from the data so that the rendered
image is O(1) — which is precisely the condition the threshold assumes.

If you ever build a patched extension with the threshold lowered (~1e-9 in BOTH
``forward.cu`` and ``backward.cu`` — the backward pass carries the same 1e-5 and
therefore the same bias in the gradients), set ``--gauss-signal-scale 1`` and
this becomes a no-op.
"""
from __future__ import annotations

import math

import torch

# Optional native dependency, guarded exactly as TIGRE and ASTRA are in
# `reconstruction/iterative/{tigre,astra}/reconstructor.py`: import at module
# scope, fall back to None, and let a `_check_*_available()` raise a directed
# ImportError at the point of use. Importing this package must therefore never
# require CUDA or the third-party build — `--help`, the algorithm listing and
# the whole test suite import the registry, and a machine without the extension
# still runs every other backend.
try:
    import xray_gaussian_rasterization_voxelization as _xgrv
except ImportError:
    _xgrv = None

#: The rasteriser's absolute per-contribution threshold, in RENDERED units.
#: Quoted from `cuda_rasterizer/forward.cu`; kept here so `resolve_signal_scale`
#: can reason about it rather than hard-coding a magic factor.
ALPHA_CUTOFF = 1e-5


def available() -> bool:
    """Whether the rasteriser can be used at all. Cheap, and never raises."""
    return _xgrv is not None


def _check_rasterizer_available():
    """Raise ImportError if the X-ray Gaussian rasteriser is not installed."""
    if _xgrv is None:
        raise ImportError(
            "The X-ray Gaussian rasteriser is required for --algorithm "
            "gaussian. It is a compiled extension, not vendored here (Inria/"
            "MPII gaussian-splatting research licence), and is not on PyPI. "
            "Build from source: git clone --recursive "
            "https://github.com/Ruyi-Zha/r2_gaussian && pip install "
            "--no-build-isolation "
            "./r2_gaussian/r2_gaussian/submodules/"
            "xray-gaussian-rasterization-voxelization")


def extension():
    """The compiled extension, or a directed ImportError."""
    _check_rasterizer_available()
    return _xgrv


def render(cloud, camera, *, need_screenspace_grad: bool = False) -> dict:
    """Rasterise the cloud into one projection.

    Returns ``{'image', 'radii', 'visible', 'screenspace'}`` with the image in
    OUR detector convention (see `camera.Camera.to_ours`) and in RENDERED units
    — line integrals scaled by ``world_scale * signal_scale``. Converting back
    to mm^-1 is the caller's job and happens in exactly one place,
    `reconstructor._to_line_integral`.

    ``need_screenspace_grad`` allocates the zero tensor whose gradient the
    densification criterion reads. It is only needed on training steps, and it
    costs an extra (N, 3) buffer plus a graph node, so evaluation renders skip
    it.
    """
    ext = extension()
    settings = ext.GaussianRasterizationSettings(
        image_height=int(camera.height),
        image_width=int(camera.width),
        tanfovx=float(camera.tan_half_fov_x),
        tanfovy=float(camera.tan_half_fov_y),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        campos=camera.camera_center,
        prefiltered=False,
        mode=1,                      # 1 = cone beam; 0 would be parallel
        debug=False,
    )
    rasterizer = ext.GaussianRasterizer(raster_settings=settings)

    means3D = cloud.xyz
    if need_screenspace_grad:
        # The densification criterion is the magnitude of the loss gradient
        # w.r.t. each primitive's SCREEN-SPACE position. The kernel writes that
        # gradient into a tensor it is handed, so the tensor has to exist and
        # be part of the graph even though its value is always zero.
        screenspace = torch.zeros_like(means3D, requires_grad=True) + 0
        try:
            screenspace.retain_grad()
        except RuntimeError:         # not in a grad-enabled context
            pass
    else:
        screenspace = torch.zeros_like(means3D)

    image, radii = rasterizer(
        means3D=means3D,
        means2D=screenspace,
        opacities=cloud.density,
        scales=cloud.scaling,
        rotations=cloud.rotation,
        cov3D_precomp=None,
    )
    return {'image': image[0], 'radii': radii, 'visible': radii > 0,
            'screenspace': screenspace}


def voxelize(cloud, *, n_voxel, extent_world, center_world) -> torch.Tensor:
    """Evaluate the cloud on a regular grid, in WORLD (normalised) units.

    ``n_voxel`` is (nx, ny, nz); ``extent_world`` and ``center_world`` are the
    grid's size and centre in the same normalised coordinates the cloud lives
    in. Returns (nx, ny, nz) of density in RENDERED units per unit length.

    This is the kernel's own voxelizer rather than a hand-rolled evaluation
    loop, and that is a deliberate choice: it applies the same covariance
    convention the rasteriser does, so the exported volume and the fitted
    projections cannot drift apart through two different readings of the same
    parameters. VERIFIED (2026-08-29): grid mass matches the mixture's analytic
    mass ``sum a_i (2 pi)^{3/2} |Sigma_i|^{1/2}`` to 0.5 % at 0.3, 0.15, 0.075
    and 0.0375 mm — no pitch dependence, so the grid is a faithful sampling of
    the field and not a point-sampling artefact.
    """
    ext = extension()
    settings = ext.GaussianVoxelizationSettings(
        scale_modifier=1.0,
        nVoxel_x=int(n_voxel[0]), nVoxel_y=int(n_voxel[1]),
        nVoxel_z=int(n_voxel[2]),
        sVoxel_x=float(extent_world[0]), sVoxel_y=float(extent_world[1]),
        sVoxel_z=float(extent_world[2]),
        center_x=float(center_world[0]), center_y=float(center_world[1]),
        center_z=float(center_world[2]),
        prefiltered=False,
        debug=False,
    )
    voxelizer = ext.GaussianVoxelizer(voxel_settings=settings)
    volume, _radii = voxelizer(
        means3D=cloud.xyz,
        opacities=cloud.density,
        scales=cloud.scaling,
        rotations=cloud.rotation,
        cov3D_precomp=None,
    )
    return volume


#: Voxels per export slab.
#:
#: NOT just the output fp32. MEASURED the hard way (run dxphgwxx, 2026-08-31:
#: 7 750 iterations, best iterate restored, then OOM at the final export):
#: `_C.voxelize_gaussians` asked for 14.95 GiB on a 200 M-voxel slab whose
#: OUTPUT is 0.75 GiB. Reading `cuda_voxelizer/voxelizer_impl.h`, the extension
#: allocates two internal buffers on top of it:
#:
#:   ImageState    12 B per VOXEL  (n_contrib uint32 + ranges uint2)
#:   BinningState  ~24-56 B per (gaussian, voxel-tile) PAIR, i.e. `num_rendered`
#:
#: BinningState is what blew up, and it is not predictable from the slab size
#: alone: it scales with how many tiles each primitive covers, so a cloud that
#: has been trained to high ANISOTROPY (this run reached the 1.2 M cap with
#: pancake-shaped primitives) costs far more per voxel than the seed did. A
#: taller slab is worse roughly linearly, which is the only lever this side of
#: the API — hence a much smaller default AND the halving retry in
#: `chunked_voxelize`, which is what actually makes the export safe.
DEFAULT_MAX_VOXELS = 32_000_000

#: Device bytes per slab voxel, excluding the unpredictable BinningState: 4 for
#: the fp32 output plus 12 for ImageState. `algorithm.footprint` charges this.
BYTES_PER_SLAB_VOXEL = 16


def chunked_voxelize(cloud, *, n_voxel, extent_world, center_world,
                     max_voxels: int = DEFAULT_MAX_VOXELS) -> torch.Tensor:
    """`voxelize`, split along z, assembled ON THE HOST. Returns a CPU tensor.

    A 75 um export of a 25 x 26 x 57 mm ROI is 87 M voxels; the same ROI at
    37.5 um is 700 M, which is 2.8 GiB of fp32 output alone. Slabs keep the
    peak bounded without changing the result — each slab is queried at its own
    centre with the same pitch, and the kernel is position-independent.

    WHY THE OUTPUT LIVES ON THE HOST. Slabbing the QUERY while assembling into
    a device tensor bounds nothing: the full grid is still allocated on the GPU,
    beside a cloud that is still resident because it is what is being queried.
    At Scan_1510's 75 um / ds3 grid (1085 x 1085 x 844 = 993.6 M voxels) that
    output is 3.70 GiB on top of 9.2 GiB of primitives and a 0.73 GiB sinogram,
    which does not fit a 16 GiB card — and it fails at the LAST step of a run
    measured in hours. The caller wants the volume in host memory anyway (it
    goes straight to numpy), so each slab is copied down as it is produced and
    the device never holds more than one. Device peak becomes
    ``max_voxels x 4`` regardless of the export grid.
    """
    nx, ny, nz = (int(v) for v in n_voxel)
    if nx * ny * nz <= max_voxels:
        return voxelize(cloud, n_voxel=(nx, ny, nz),
                        extent_world=extent_world,
                        center_world=center_world).cpu()

    dz = float(extent_world[2]) / nz
    z0 = float(center_world[2]) - float(extent_world[2]) / 2.0
    per = max(1, int(max_voxels // max(1, nx * ny)))
    out = torch.empty((nx, ny, nz), dtype=torch.float32, device='cpu')
    k = 0
    while k < nz:
        m = min(per, nz - k)
        while True:
            try:
                slab = voxelize(
                    cloud, n_voxel=(nx, ny, m),
                    extent_world=(extent_world[0], extent_world[1], m * dz),
                    center_world=(center_world[0], center_world[1],
                                  z0 + (k + m / 2.0) * dz))
                break
            except torch.OutOfMemoryError:
                # The extension's BinningState scales with (gaussian, tile)
                # pairs, which no caller can predict from the slab size — so
                # the honest strategy is to find the working depth by halving
                # rather than to guess a constant. `per` keeps the reduced
                # depth for the remaining slabs, so this costs one failed
                # allocation per export and not one per slab.
                if m <= 1:
                    raise
                m = max(1, m // 2)
                per = m
                torch.cuda.empty_cache()
                print(f"    export: out of memory, halving the slab to {m} "
                      f"z-slices ({nx * ny * m / 1e6:.0f} M voxels) and "
                      f"retrying", flush=True)
        out[:, :, k:k + m] = slab.cpu()
        del slab
        torch.cuda.empty_cache()
        k += m
    return out
