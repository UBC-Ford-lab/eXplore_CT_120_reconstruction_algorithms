"""Scanner geometry -> rasteriser matrices, for ANY scan.

R2-Gaussian's own dataset readers build cameras from a ``meta_data.json``
written by a synthetic-benchmark exporter: a cube-shaped world, a centred
detector, no in-plane detector rotation, no centre-of-rotation policy. A real
cone-beam micro-CT scan has all four, and every one of them moves rays by more
than a pixel. This module builds the same matrices from the geometry dict the
pipeline has already measured and calibrated, so nothing here is scan-specific
and no intermediate JSON exists to drift.

THE THREE THINGS UPSTREAM CANNOT EXPRESS, AND WHERE THEY GO
-----------------------------------------------------------
* **Centre of rotation.** ``prepare_scan`` has already applied the COR policy
  (``ct_core.pipeline.apply_cor_policy`` — geometric centre by default; the
  scan.xml value is deliberately NOT applied because psi already covers the
  alignment). It lands in ``central_pixel_a/b``, which in general is not the
  detector's middle. A symmetric frustum cannot represent that, so it becomes a
  principal-point offset in NDC on ``P[0,2]`` / ``P[1,2]``.
* **Detector in-plane rotation psi.** ``geometry['det_psi_rad']``, measured
  reference-free per scan. It is a roll of the camera about its own optical
  axis, applied to the pose.
* **Physical units.** Positions are normalised into [-1, 1]^3 by
  ``world_scale``; the FoV is an angle and therefore scale-free.

DETECTOR CONVENTION
-------------------
The rasteriser writes its image with both detector axes reversed relative to
ours. Rather than resample anything, `Camera.to_ours` / `Camera.to_theirs`
flip once, in one place. The angle convention differs by pi for the same
reason. Both were verified against the pipeline's own ray geometry (relative
error 0.0025 on the projected position of a calibrated target).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def world_scale_for(domain) -> float:
    """mm -> normalised world units.

    The rasteriser's kernels carry an absolute screen-space radius cap and an
    absolute alpha cutoff, both written for a scene that lives in roughly
    [-1, 1]^3. So the domain — not the export ROI, and not a fixed number — is
    what sets the scale: the largest extent maps to 2.
    """
    extent = (domain.aabb_max - domain.aabb_min).abs().max().item()
    if extent <= 0:
        raise ValueError("model domain has zero extent")
    return 2.0 / float(extent)


def resolve_signal_scale(sinogram, world_scale: float, requested) -> float:
    """How much to multiply the sinogram by before fitting. See `kernel`.

    ``requested`` is a float, or 'auto'. AUTO puts the mean RENDERED line
    integral at 1.0, which is the regime the kernel's absolute thresholds were
    written for: a rendered value is ``mu_path_mm * world_scale * signal_scale``
    and we solve that for a mean of one.

    This is not a fudge factor, it is a change of units, and it is exact: the
    forward model is linear in mu, so fitting ``S * p`` yields ``S * mu`` and
    dividing back at export recovers the same reconstruction the kernel would
    have produced if its thresholds scaled with the data. What it buys is that
    the thresholds stop discarding signal — 22 % of the line integral on
    Scan_1510 at S = 1.
    """
    if requested is not None and str(requested).lower() != 'auto':
        return float(requested)
    mean_path = float(np.mean(np.abs(np.asarray(sinogram, dtype=np.float64))))
    if mean_path <= 0:
        return 1.0
    scale = 1.0 / (mean_path * world_scale)
    # Cheap sanity rails. Below 1 we would be pushing signal INTO the cutoff;
    # far above 1e6 something is wrong with the sinogram's units and a silent
    # enormous scale would be worse than a visible one.
    return float(min(max(scale, 1.0), 1e6))


class Camera:
    """One projection angle, as the rasteriser wants it.

    Cheap to build (a few 4x4s), so the whole set is built once up front and
    kept; the per-view tensors are tiny next to the cloud.
    """

    __slots__ = ('height', 'width', 'tan_half_fov_x', 'tan_half_fov_y',
                 'world_view_transform', 'projection_matrix',
                 'full_proj_transform', 'camera_center', 'index', 'angle')

    def __init__(self, index, angle, *, geometry, n_b, n_a, world_scale,
                 device):
        R_s = float(geometry['R_s'])
        DSD = R_s + float(geometry['R_d'])
        da, db = float(geometry['da']), float(geometry['db'])
        psi = float(geometry.get('det_psi_rad', 0.0) or 0.0)

        self.index = int(index)
        self.angle = float(angle)
        self.height, self.width = int(n_b), int(n_a)

        # Half-angles of the detector as seen from the source. A ratio of two
        # lengths, so world_scale cancels and this is the one quantity that
        # needs no unit conversion.
        self.tan_half_fov_x = math.tan(math.atan2(n_a * da / 2.0, DSD))
        self.tan_half_fov_y = math.tan(math.atan2(n_b * db / 2.0, DSD))

        # Principal point: where the optical axis actually crosses the panel,
        # in NDC. central_pixel_* is the COR-resolved centre, which is only the
        # geometric middle by coincidence.
        du = float(geometry['central_pixel_a']) - (n_a - 1) / 2.0
        dv = float(geometry['central_pixel_b']) - (n_b - 1) / 2.0
        pp_ndc = (-2.0 * du / n_a, -2.0 * dv / n_b)

        c2w = _angle_to_pose(R_s * world_scale, self._their_angle(angle), psi)
        w2c = np.linalg.inv(c2w).astype(np.float32)

        self.world_view_transform = torch.tensor(
            w2c.T, dtype=torch.float32, device=device)
        self.projection_matrix = _projection_matrix(
            self.tan_half_fov_x, self.tan_half_fov_y, pp_ndc).to(device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0)).squeeze(0))
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    # -- conventions -------------------------------------------------------
    @staticmethod
    def _their_angle(angle: float) -> float:
        """Our gantry angle -> theirs. Differs by pi; verified, not assumed."""
        return float((angle + math.pi) % (2.0 * math.pi))

    @staticmethod
    def to_ours(image):
        """Rasteriser image -> our detector indexing (both axes reversed)."""
        return torch.flip(image, dims=(-2, -1))

    @staticmethod
    def to_theirs(image):
        """Our detector indexing -> the rasteriser's. Its own inverse."""
        return torch.flip(image, dims=(-2, -1))


def build_cameras(angles, *, geometry, n_b, n_a, world_scale, device):
    return [Camera(i, float(a), geometry=geometry, n_b=n_b, n_a=n_a,
                   world_scale=world_scale, device=device)
            for i, a in enumerate(np.asarray(angles, dtype=np.float64))]


# --------------------------------------------------------------------------
# The two matrices
# --------------------------------------------------------------------------

def _angle_to_pose(dso_world: float, angle: float, psi_rad: float):
    """Camera-to-world for a source orbiting the z axis at radius ``dso_world``.

    Rotate -90 deg about x, then +90 deg about z, then by the gantry angle
    about z; psi is a further roll about the camera's own optical axis, which
    is what an in-plane detector rotation is.
    """
    phi1 = -math.pi / 2
    R1 = np.array([[1.0, 0.0, 0.0],
                   [0.0, math.cos(phi1), -math.sin(phi1)],
                   [0.0, math.sin(phi1), math.cos(phi1)]])
    phi2 = math.pi / 2
    R2 = np.array([[math.cos(phi2), -math.sin(phi2), 0.0],
                   [math.sin(phi2), math.cos(phi2), 0.0],
                   [0.0, 0.0, 1.0]])
    R3 = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                   [math.sin(angle), math.cos(angle), 0.0],
                   [0.0, 0.0, 1.0]])
    rot = R3 @ R2 @ R1
    if psi_rad:
        cp, sp = math.cos(psi_rad), math.sin(psi_rad)
        rot = rot @ np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = [dso_world * math.cos(angle), dso_world * math.sin(angle), 0.0]
    return pose


def _projection_matrix(tan_half_x: float, tan_half_y: float, pp_ndc) -> torch.Tensor:
    """Perspective projection with a principal-point offset.

    ``x_ndc = P[0,0] * x/z + P[0,2]``, so the offset is a direct NDC shift on
    P[0,2] / P[1,2]. Upstream pins both to zero, which forces a symmetric
    frustum and cannot represent an off-centre centre of rotation.

    znear/zfar only have to bracket the scene; the cloud lives in [-1, 1]^3 and
    the source sits ~10 units out, so these are generous and never bind.
    """
    znear, zfar = 0.01, 100.0
    top = tan_half_y * znear
    right = tan_half_x * znear
    P = torch.zeros(4, 4, dtype=torch.float32)
    P[0, 0] = znear / right
    P[1, 1] = znear / top
    P[0, 2] = float(pp_ndc[0])
    P[1, 2] = float(pp_ndc[1])
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P.transpose(0, 1)
