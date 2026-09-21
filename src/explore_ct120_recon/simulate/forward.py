"""Line integrals of a dense volume through the package's ray geometry, the
detector's counts from them, and the loop that writes a scan.

The geometry is built from the ``scan.xml`` that will be WRITTEN, with the
detector shape that will be written, through the same ``build_scan_geometry``
every driver calls on loading — so the simulated rays are the rays the
reconstruction will assume, to the last pooled pixel. A synthetic scan at
binning 3 is a 3x smaller detector with 3x the pitch, reconstructed at
``--downsample 1``; its noise is the MEASURED noise of a 3x3-pooled detector
(``ct_core.noise_model``: Poisson slope at that binning, read noise per raw
pixel scaled by 1/binning), not the noise of independent pixels.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import xmltodict

from ..ct_core.noise_model import DEFAULT_READ_NOISE, default_poisson_slope
from ..ct_core.pipeline import build_scan_geometry
from ..ct_core.preprocessing import downsample_projections
from ..learning_based_iterative.projection import render_projection
from ..learning_based_iterative.scene import Scene, model_domain_from_bounds
from . import scan_writer


# ------------------------------------------------------------ the object --
class VolumeSampler:
    """A dense attenuation volume (mm^-1) on a ct_core grid, callable at mm
    points by trilinear interpolation; zero outside the grid (air).

    ``volume`` is ``(nx, ny, nz)`` in the package's (x, y, z) order; ``geom``
    carries ``vol_shape``, ``vol_origin``, ``dx``, ``dz`` (the sidecar's
    grid). Storage is ``(1, 1, nz, ny, nx)`` so ``grid_sample``'s (x, y, z)
    grid order maps onto (W, H, D), with ``align_corners=False`` because the
    voxels TILE the box (voxel i is centred at ``(i + 0.5) / n``) — the same
    convention as the voxel backend's ``VoxelGrid``.
    """

    def __init__(self, volume, geom: dict, device='cpu'):
        vol = torch.as_tensor(np.asarray(volume, dtype=np.float32))
        if vol.ndim != 3:
            raise ValueError(f"volume must be (nx, ny, nz), got {tuple(vol.shape)}")
        self.shape = tuple(int(v) for v in vol.shape)
        if tuple(int(v) for v in geom['vol_shape']) != self.shape:
            raise ValueError(f"grid {tuple(geom['vol_shape'])} does not match the volume {self.shape}")
        self.geom = {'vol_shape': self.shape,
                     'vol_origin': tuple(float(v) for v in geom['vol_origin']),
                     'dx': float(geom['dx']), 'dz': float(geom['dz'])}
        self.data = vol.permute(2, 1, 0).contiguous()[None, None].to(device)   # (1,1,nz,ny,nx)
        nx, ny, nz = self.shape
        dx, dz = self.geom['dx'], self.geom['dz']
        ox, oy, oz = self.geom['vol_origin']
        self._lo = torch.tensor([ox - nx * dx / 2, oy - ny * dx / 2, oz - nz * dz / 2],
                                dtype=torch.float32, device=device)
        self._hi = torch.tensor([ox + nx * dx / 2, oy + ny * dx / 2, oz + nz * dz / 2],
                                dtype=torch.float32, device=device)

    @property
    def device(self):
        return self.data.device

    def bounds(self) -> dict:
        lo, hi = self._lo.tolist(), self._hi.tolist()
        return {'x_min': lo[0], 'x_max': hi[0], 'y_min': lo[1], 'y_max': hi[1],
                'z_min': lo[2], 'z_max': hi[2]}

    def __call__(self, xyz_mm: torch.Tensor) -> torch.Tensor:
        """mu at ``(M, 3)`` mm points."""
        lo, hi = self._lo.to(xyz_mm), self._hi.to(xyz_mm)
        g = (xyz_mm - lo) / (hi - lo) * 2.0 - 1.0
        out = torch.nn.functional.grid_sample(
            self.data.to(xyz_mm.dtype), g.reshape(1, -1, 1, 1, 3), mode='bilinear',
            padding_mode='zeros', align_corners=False)
        return out.reshape(-1)

    def volume_fn(self, scene: Scene) -> Callable[[torch.Tensor], torch.Tensor]:
        """The renderer's contract: ``(M, 3)`` in the scene's normalised
        coordinates -> mu."""
        amin, amax = scene.model_domain.aabb_min, scene.model_domain.aabb_max
        center, half = (amin + amax) * 0.5, (amax - amin) * 0.5

        def fn(xyz_norm):
            return self(xyz_norm * half.to(xyz_norm) + center.to(xyz_norm))
        return fn

    def total_attenuation(self) -> float:
        """sum(mu) x voxel volume — the object's 'mass' in mm^2, conserved
        by a mass-preserving deformation."""
        g = self.geom
        return float(self.data.sum().item() * g['dx'] * g['dx'] * g['dz'])


def sampler_from_volume_file(path, device='cpu', verbose=False) -> VolumeSampler:
    """A ``VolumeSampler`` from a ``.vff`` this package wrote (HU, with the
    ``<volume>.json`` sidecar giving the grid and the HU map) — mu is the
    sidecar's calibration inverted."""
    from ..ct_core.volume_report import load_reconstructed_volume, load_sidecar
    side = load_sidecar(path, verbose=verbose)
    if not side:
        raise FileNotFoundError(f"{path} has no .json sidecar; the grid and HU map are unknown")
    hu, _ = load_reconstructed_volume(path, verbose=verbose)
    cal = side['hu_calibration']
    mu = (hu - float(cal['offset'])) / float(cal['scale'])
    vs = side['voxel_size_mm']
    geom = {'vol_shape': tuple(int(v) for v in side['volume_shape']),
            'vol_origin': tuple(float(v) for v in side['vol_origin_mm']),
            'dx': float(vs['xy']), 'dz': float(vs['z'])}
    return VolumeSampler(np.clip(mu, 0.0, None), geom, device=device)


# ----------------------------------------------------------- the scanner --
def scene_for_simulation(xml_header: dict, detector_shape, angles_deg,
                         domain_bounds: dict, psi_deg: float = 0.0,
                         cor_offset_px=(0.0, 0.0)) -> Scene:
    """The scene the loaders will build from the written folder, so the
    simulation renders through it. ``detector_shape`` is ``(n_b, n_a)`` of the
    frames AS WRITTEN. ``psi_deg`` rotates the detector in its plane and
    ``cor_offset_px = (da, db)`` shifts the axis off the panel centre — known
    misalignments for exercising the calibrators; the header stays ideal."""
    n_b, n_a = (int(v) for v in detector_shape)
    geometry = build_scan_geometry(
        xml_header, raw_detector_shape=(n_b, n_a), fov_xy=1.0, fov_z=1.0,
        voxel_xy=1.0, voxel_z=1.0, roi_bounds=None, cor_mode='center',
        downsample=1, verbose=False)
    geometry['central_pixel_a'] += float(cor_offset_px[0])
    geometry['central_pixel_b'] += float(cor_offset_px[1])
    geometry['det_psi_rad'] = math.radians(float(psi_deg))
    geometry['det_psi_deg'] = float(psi_deg)
    angles = torch.deg2rad(torch.as_tensor(np.asarray(angles_deg, dtype=np.float32)))
    # Only indexed for the (unused) targets: a broadcast view costs nothing.
    sinogram = torch.zeros(1, 1, 1, dtype=torch.float32).expand(len(angles), n_b, n_a)
    domain = model_domain_from_bounds(domain_bounds, shape='box')
    return Scene(sinogram=sinogram, angles=angles, geometry=geometry,
                 scan_name='simulation', model_domain=domain)


def samples_for(scene: Scene, samples_per_mm: float) -> int:
    """Quadrature points per ray: the domain's diagonal at ``samples_per_mm``
    (the renderer lays a FIXED count across each ray's chord, so the longest
    chord sets the count)."""
    diag = float((scene.model_domain.aabb_max - scene.model_domain.aabb_min).norm())
    return max(8, int(math.ceil(diag * float(samples_per_mm))))


@torch.no_grad()
def project_view(sampler: VolumeSampler, scene: Scene, view: int, *,
                 samples_per_mm: float, device=None, chunk_size: int = 65536) -> np.ndarray:
    """Line integrals ``(n_b, n_a)`` of ``sampler`` for view ``view``."""
    device = device or sampler.device
    pred, _ = render_projection(sampler.volume_fn(scene), scene, int(view),
                                samples_for(scene, samples_per_mm), device,
                                chunk_size=chunk_size, stratified=False)
    return pred.cpu().numpy().astype(np.float32)


def counts_from_line_integrals(p: np.ndarray, bright: np.ndarray, dark: np.ndarray, *,
                               poisson_slope: float, read_noise: float, binning: int = 1,
                               rng=None) -> np.ndarray:
    """Detector counts for line integrals ``p``: ``dark + (bright - dark)
    exp(-p)`` plus noise of variance ``A c + (sigma_r / binning)^2`` (the
    measured model; Gaussian at these counts). ``rng=None``: noiseless."""
    c = (np.asarray(bright, dtype=np.float64) - np.asarray(dark, dtype=np.float64)) * np.exp(-np.asarray(p, dtype=np.float64))
    counts = np.asarray(dark, dtype=np.float64) + c
    if rng is not None:
        var = float(poisson_slope) * np.clip(c, 0.0, None) + (float(read_noise) / float(binning)) ** 2
        counts = counts + rng.standard_normal(counts.shape) * np.sqrt(var)
    return counts


VolumeSource = Callable[[int, int, float], VolumeSampler]   # (phase, view, angle_deg) -> object


def simulate_scan(source, template_folder, out_folder, *, binning: int = 1,
                  n_views=None, step_deg=None, phases: int = 1, psi_deg: float = 0.0,
                  cor_offset_px=(0.0, 0.0), noise: bool = True, seed: int = 0,
                  samples_per_mm=None, device=None, serial: str = 'SIMULATED',
                  specimen: str = 'SIMULATED', roi_from_template: bool = True,
                  extra_record=None, log=print) -> Path:
    """Simulate and write a scan folder from ``source``.

    ``source`` is a ``VolumeSampler`` (a still object) or a callable
    ``(phase, view, angle_deg) -> VolumeSampler`` (a moving one; every
    sampler must share one grid). The acquisition is the template scan's —
    its views, angular step, bright/dark fields (pooled by ``binning``) and
    header — with ``n_views`` / ``step_deg`` / ``phases`` overriding. The
    written ``simulation.json`` records what was made (no paths).
    """
    tpl = scan_writer.template_header(template_folder)
    xml = tpl['xml']
    sp = xml['Series']['SeriesParams']
    n_views = int(n_views or sp['ViewCount'])
    step_deg = float(step_deg if step_deg is not None else sp['IncrementAngle'])
    B = max(1, int(binning))
    bright = downsample_projections(tpl['bright'], B)
    dark = downsample_projections(tpl['dark'], B)
    pixel_mm = float(xml['Series']['DetectorSpacing']) * B
    gantry = scan_writer.gantry_positions(n_views, step_deg)
    angle_offset = float(xml['Series']['AngleOffset'])
    angles_deg = scan_writer.view_angles_deg(gantry, angle_offset)
    written_xml = xmltodict.parse(scan_writer.scan_xml(
        xml, n_views=n_views, step_deg=step_deg, detector_pixel_mm=pixel_mm,
        detector_shape=bright.shape, phases=phases, specimen=specimen))

    still = isinstance(source, VolumeSampler)
    first = source if still else source(0, 0, float(angles_deg[0]))
    device = device or first.device
    scene = scene_for_simulation(written_xml, bright.shape, angles_deg, first.bounds(),
                                 psi_deg=psi_deg, cor_offset_px=cor_offset_px)
    spm = float(samples_per_mm) if samples_per_mm else 2.0 / min(first.geom['dx'], first.geom['dz'])
    slope, sigma_r = default_poisson_slope(B), DEFAULT_READ_NOISE
    rng = np.random.default_rng(int(seed)) if noise else None
    log(f"  simulating {phases} phase(s) x {n_views} views of {bright.shape[0]}x{bright.shape[1]} "
        f"(binning {B}, pitch {pixel_mm:.4f} mm), {samples_for(scene, spm)} samples/ray, "
        f"noise {'A=%.3g sigma_r=%.3g/%d' % (slope, sigma_r, B) if noise else 'off'}, psi {psi_deg:+.3f} deg")

    def frames():
        for phase in range(int(phases)):
            for view in range(n_views):
                obj = first if still else source(phase, view, float(angles_deg[view]))
                p = project_view(obj, scene, view, samples_per_mm=spm, device=device)
                yield phase, view, counts_from_line_integrals(
                    p, bright, dark, poisson_slope=slope, read_noise=sigma_r, binning=B, rng=rng)
                if (view + 1) % 20 == 0 or view + 1 == n_views:
                    log(f"    phase {phase} view {view + 1}/{n_views}")

    folder = scan_writer.write_scan_folder(
        out_folder, template_xml=xml, frames=frames(), gantry_deg=gantry, step_deg=step_deg,
        detector_pixel_mm=pixel_mm, bright=bright, dark=dark, phases=phases, serial=serial,
        specimen=specimen, roi_xml_from=(template_folder if roi_from_template else None), log=log)
    record = {'template_scan': os.path.basename(str(template_folder).rstrip('/')),
              'binning': B, 'detector_shape': [int(v) for v in bright.shape],
              'detector_pixel_mm': pixel_mm, 'n_views': n_views, 'step_deg': step_deg,
              'phases': int(phases), 'angles_deg': [float(a) for a in angles_deg],
              'psi_deg': float(psi_deg), 'cor_offset_px': [float(v) for v in cor_offset_px],
              'noise': bool(noise), 'poisson_slope': slope, 'read_noise_per_raw_pixel': sigma_r,
              'seed': int(seed), 'samples_per_mm': spm,
              'object_grid': first.geom, 'domain_bounds': first.bounds()}
    record.update(extra_record or {})
    with open(Path(folder) / 'simulation.json', 'w') as f:
        json.dump(record, f, indent=2)
    return Path(folder)
