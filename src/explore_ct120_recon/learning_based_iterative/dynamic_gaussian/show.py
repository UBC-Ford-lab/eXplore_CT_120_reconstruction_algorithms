"""Show the motion a finished dynamic run learned, from its checkpoint.

    ct120-motion-show RESULT_DIR [--wandb-run-id ID]

``RESULT_DIR`` holds ``recon_cloud.pt`` (cloud + motion + geometry) and the
``recon.json`` sidecar (the delivered ROI grid and the HU calibration), i.e.
what a ``gaussian_dynamic`` run leaves behind. Writes into
``RESULT_DIR/motion_plots/``:

    gates.png       mid coronal + axial slices at gate 0 (code -1), the rest
                    state (0) and gate 1 (+1), and the gate-1 minus gate-0
                    difference — the two breathing states the scan sampled
    field.png       the gate field as arrows over the coronal anatomy
    breathing.gif   the coronal slice swept from gate 0 to gate 1 and back
                    (an INTERPOLATION along the learned field: the scan has
                    no frames between the gates)
    codes.png       the free code trajectories (`panels.codes_figure`)

With ``--wandb-run-id`` the same lands on that run's W&B page under
``motion/gates``, ``motion/field_arrows``, ``motion/breathing`` and
``motion/codes_final``.

Slices are oriented as the pipeline's W&B ``recon_slices`` panels and the
delivered VFF (bed at the bottom): the voxeliser's (x, y, z) grid is flipped
in y and transposed to the VFF's (z, y, x), VERIFIED against the delivered
volume of the static two-gate run (see `_vff_orientation`).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from matplotlib.figure import Figure

from ..gaussian import kernel
from ..gaussian.model import GaussianCloud

from . import panels
from .motion import MotionModel

HU_WINDOW = (-1000.0, 2000.0)
DIFF_WINDOW = (-600.0, 600.0)


# ------------------------------------------------------------------ loading --
def load_run(result_dir: str, device='cuda'):
    """(cloud, motion, roi_geometry, world_scale, signal_scale, hu) from a
    result directory. ``motion`` is None for a static run."""
    ck_path = next((os.path.join(result_dir, c) for c in ('recon_cloud.pt', 'cloud.pt')
                    if os.path.exists(os.path.join(result_dir, c))), None)
    if ck_path is None:
        raise FileNotFoundError(f"no recon_cloud.pt / cloud.pt in {result_dir}")
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    c = ck['cloud']
    cloud = GaussianCloud(c['xyz'][:2], torch.ones(2, 3), c['rotation'][:2],
                          torch.ones(2, 1))
    cloud.restore(c)
    cloud = cloud.to(device)
    ws, ss = float(ck['world_scale']), float(ck['signal_scale'])
    motion = None
    if 'motion' in ck:
        mc = ck.get('motion_config', {})
        motion = MotionModel.from_snapshot(
            ck['motion'], n_views=int(ck['motion']['codes'].shape[0]),
            n_codes=int(mc.get('n_codes', ck['motion']['codes'].shape[1])),
            view_groups=ck.get('view_groups'), device=device)
    side = os.path.join(result_dir, 'recon.json')
    if os.path.exists(side):
        meta = json.load(open(side))
        shape = meta['volume_shape']                       # (nx, ny, nz)
        vs = meta['voxel_size_mm']
        geom = {'vol_shape': tuple(int(v) for v in shape),
                'vol_origin': tuple(float(v) for v in meta['vol_origin_mm']),
                'dx': float(vs['xy']), 'dz': float(vs['z'])}
        hu = (float(meta['hu_calibration']['scale']),
              float(meta['hu_calibration']['offset']))
    else:
        geom = dict(ck['geometry'])
        hu = (1.0, 0.0)
    return cloud, motion, geom, ws, ss, hu


# ---------------------------------------------------------------- rendering --
@torch.no_grad()
def render_slab(shown, geom, ws, ss, hu, *, axis: str, index: int,
                thickness: int = 1) -> np.ndarray:
    """A slab of the ROI grid through ``index`` along ``axis`` ('x', 'y' or
    'z'), in HU, as the voxeliser lays it out: (nx, ny, nz) with the slab
    axis reduced to ``thickness`` and averaged."""
    nx, ny, nz = geom['vol_shape']
    dx, dz = float(geom['dx']), float(geom['dz'])
    ox, oy, oz = geom['vol_origin']
    pitch = {'x': dx, 'y': dx, 'z': dz}
    n = {'x': nx, 'y': ny, 'z': nz}
    origin = {'x': ox, 'y': oy, 'z': oz}
    lo = index - thickness // 2
    # centre of the slab in mm: voxel centres run from origin - (n-1)/2 * pitch
    centre = origin[axis] + (lo + (thickness - 1) / 2.0 - (n[axis] - 1) / 2.0) * pitch[axis]
    n_vox = {'x': nx, 'y': ny, 'z': nz}
    n_vox[axis] = thickness
    cen = {'x': ox, 'y': oy, 'z': oz}
    cen[axis] = centre
    extent = (n_vox['x'] * dx * ws, n_vox['y'] * dx * ws, n_vox['z'] * dz * ws)
    vol = kernel.chunked_voxelize(
        shown, n_voxel=(n_vox['x'], n_vox['y'], n_vox['z']),
        extent_world=extent, center_world=(cen['x'] * ws, cen['y'] * ws, cen['z'] * ws))
    mu = vol.detach().cpu().numpy() / ss
    return (mu * hu[0] + hu[1]).astype(np.float32)


def _vff_orientation(slab: np.ndarray) -> np.ndarray:
    """(nx, ny, nz) voxeliser layout -> the VFF's (nz, ny, nx) with the y flip
    the export applies, so a slice indexes like the delivered volume."""
    return np.ascontiguousarray(slab[:, ::-1, :].transpose(2, 1, 0))


def gate_code(motion, value: float, free=None) -> torch.Tensor:
    """A full code for the state ``value`` on the breathing axis (a gate
    code in [-1, 1] for a gated scan, a phase in [0, 1) for a continuous
    one; see `MotionModel.state_code`) with the free part at ``free`` (zeros
    by default). For a single-phase run the fixed part does not exist and
    ``value`` is ignored."""
    return motion.state_code(float(value), free)


def breathing_states(motion) -> dict:
    """The named states of the breathing axis a figure shows, label ->
    value: the two gates and the rest state of a gated scan, or the four
    quarter phases of a continuous cycle (the rest state is the cycle
    mean, not a phase of it)."""
    if motion is None or not motion.n_fixed:
        return {'rest (0)': 0.0}
    if getattr(motion, 'fixed_kind', 'gates') == 'phase':
        return {'phase 0.00': 0.0, 'phase 0.25': 0.25, 'phase 0.50': 0.5,
                'phase 0.75': 0.75}
    return {'gate 0 (-1)': -1.0, 'rest (0)': 0.0, 'gate 1 (+1)': 1.0}


def breathing_extremes(motion) -> tuple:
    """The two states whose difference is 'the breath': (trough, peak) on
    the breathing axis -- the gates (-1, +1), or the phases (0, 0.5) of a
    continuous cycle."""
    if getattr(motion, 'fixed_kind', 'gates') == 'phase':
        return 0.0, 0.5
    return -1.0, 1.0


def state_slices(cloud, motion, geom, ws, ss, hu, value: float):
    """(coronal (nz, nx), axial (ny, nx)) HU slices of the ROI mid-planes in
    the state ``value`` on the gate axis (rest cloud when ``motion`` is None)."""
    shown = cloud if motion is None else motion.displaced_with_code(
        cloud, gate_code(motion, value))
    nx, ny, nz = geom['vol_shape']
    # the export flips y: VFF row j is voxeliser row ny - 1 - j
    cor = _vff_orientation(render_slab(shown, geom, ws, ss, hu, axis='y',
                                       index=ny - 1 - ny // 2, thickness=3))
    axi = _vff_orientation(render_slab(shown, geom, ws, ss, hu, axis='z',
                                       index=nz // 2, thickness=3))
    return cor.mean(axis=1), axi.mean(axis=0)          # (nz, nx), (ny, nx)


# ------------------------------------------------------------------ figures --
def gates_figure(slices: dict, window=HU_WINDOW, diff_window=DIFF_WINDOW) -> Figure:
    """``window`` / ``diff_window`` are HU by default; a live panel during a
    run (before the HU calibration exists) passes windows in mu."""
    HU_WINDOW, DIFF_WINDOW = window, diff_window
    order = ('gate 0 (-1)', 'rest (0)', 'gate 1 (+1)')
    keys = [k for k in order if k in slices] or list(slices)
    diff = (keys[2], keys[0]) if len(keys) == 3 else (
        (keys[2], keys[0]) if len(keys) == 4 else None)   # 4 phases: 0.50 minus 0.00
    fig = Figure(figsize=(5 * (len(keys) + 1), 9.5), dpi=100)
    axes = fig.subplots(2, len(keys) + 1)
    for j, k in enumerate(keys):
        cor, axi = slices[k]
        axes[0, j].imshow(cor, cmap='gray', vmin=HU_WINDOW[0], vmax=HU_WINDOW[1], aspect='auto')
        axes[1, j].imshow(axi, cmap='gray', vmin=HU_WINDOW[0], vmax=HU_WINDOW[1])
        axes[0, j].set_title(f'{k}, coronal mid')
        axes[1, j].set_title(f'{k}, axial mid')
    if diff is not None:
        hi_k, lo_k = diff
        d_cor = slices[hi_k][0] - slices[lo_k][0]
        d_axi = slices[hi_k][1] - slices[lo_k][1]
        axes[0, -1].imshow(d_cor, cmap='gray', vmin=DIFF_WINDOW[0], vmax=DIFF_WINDOW[1], aspect='auto')
        axes[1, -1].imshow(d_axi, cmap='gray', vmin=DIFF_WINDOW[0], vmax=DIFF_WINDOW[1])
        axes[0, -1].set_title(f'{hi_k} minus {lo_k}, coronal')
        axes[1, -1].set_title(f'{hi_k} minus {lo_k}, axial')
    for ax in axes.ravel():
        ax.axis('off')
    fig.tight_layout()
    return fig


@torch.no_grad()
def field_figure(cloud, motion, geom, ws, coronal: np.ndarray, step_mm: float = 1.5) -> Figure:
    """The gate field (gate 1 minus gate 0, in mm) as arrows on the coronal
    mid-slice, sampled on a regular grid of REST positions in that plane."""
    nx, ny, nz = geom['vol_shape']
    dx, dz = float(geom['dx']), float(geom['dz'])
    ox, oy, oz = geom['vol_origin']
    sx = max(1, int(round(step_mm / dx)))
    sz = max(1, int(round(step_mm / dz)))
    ix = np.arange(sx // 2, nx, sx)
    iz = np.arange(sz // 2, nz, sz)
    xs = ox + (ix - (nx - 1) / 2.0) * dx
    zs = oz + (iz - (nz - 1) / 2.0) * dz
    gx, gz = np.meshgrid(xs, zs, indexing='ij')
    pts = np.stack([gx.ravel(), np.full(gx.size, oy), gz.ravel()], axis=1) * ws
    pts_t = torch.as_tensor(pts, dtype=torch.float32, device=motion.codes.device)
    lo_v, hi_v = breathing_extremes(motion)
    d = (motion.displacement(pts_t, gate_code(motion, hi_v))
         - motion.displacement(pts_t, gate_code(motion, lo_v))) / ws   # mm
    d = d.cpu().numpy().reshape(len(ix), len(iz), 3)
    fig = Figure(figsize=(7, 9), dpi=100)
    ax = fig.subplots()
    ax.imshow(coronal, cmap='gray', vmin=HU_WINDOW[0], vmax=HU_WINDOW[1], aspect='auto',
              extent=[0, nx, nz, 0])
    # the coronal image is (nz, nx) in VFF orientation: row = z, column = x
    mag = np.linalg.norm(d[:, :, [0, 2]], axis=2)
    # arrows in voxels; the longest spans ~8 % of the image height
    longest = max(float(np.abs(d[:, :, [0, 2]]).max() / min(dx, dz)), 1e-6)
    scale = longest / (0.08 * nz)
    q = ax.quiver(ix[:, None].repeat(len(iz), 1), iz[None, :].repeat(len(ix), 0),
                  d[:, :, 0] / dx, d[:, :, 2] / dz, mag, cmap='autumn',
                  angles='xy', scale_units='xy', scale=scale, width=0.003)
    fig.colorbar(q, ax=ax, fraction=0.04, label='|gate 1 - gate 0| (mm)')
    ax.set_title(f'learned gate field on the coronal mid-slice '
                 f'(arrows x{1 / scale:.1f}, max {mag.max():.2f} mm)')
    ax.axis('off')
    fig.tight_layout()
    return fig


def breathing_frames(cloud, motion, geom, ws, ss, hu, n: int = 12,
                     window=HU_WINDOW) -> np.ndarray:
    """(T, H, W) uint8 frames of the coronal slice, gate code swept
    -1 -> +1 -> -1 with a cosine ease. ``window`` is the grey window in the
    units ``hu`` maps to (HU after calibration, mu during a run)."""
    if getattr(motion, 'fixed_kind', 'gates') == 'phase':
        vals = np.linspace(0.0, 1.0, 2 * n, endpoint=False)      # one cycle
    else:
        vals = -np.cos(np.linspace(0, 2 * np.pi, 2 * n, endpoint=False))
    frames = []
    for v in vals:
        cor, _ = state_slices(cloud, motion, geom, ws, ss, hu, float(v))
        u = np.clip((cor - window[0]) / (window[1] - window[0]), 0, 1)
        frames.append((u * 255).astype(np.uint8))
    return np.stack(frames)


def write_gif(frames: np.ndarray, path: str, fps: int = 6) -> str:
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], loop=0,
                duration=int(1000 / fps))
    return path


# --------------------------------------------------------------------- main --
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('result_dir')
    ap.add_argument('--wandb-run-id', default=None,
                    help='also log the panels to this W&B run (project from '
                         '$WANDB_PROJECT, entity from $WANDB_ENTITY)')
    ap.add_argument('--frames', type=int, default=12,
                    help='frames per half-cycle of the breathing GIF (default 12)')
    ap.add_argument('--export-4d', '--nifti-4d', type=int, default=0, metavar='N',
                    dest='export_4d',
                    help='also write <result_dir>/recon_4d.nrrd (or .nii.gz, '
                         'see --export-4d-format): N breathing states on the '
                         'ROI in the run\'s HU calibration (what a run writes '
                         'itself with --dyn-export-4d)')
    ap.add_argument('--export-4d-mm', '--nifti-4d-mm', default='auto', metavar='MM',
                    dest='export_4d_mm',
                    help='voxel pitch of those states: "auto" (default) = the '
                         'delivered grid, or an isotropic pitch in mm')
    ap.add_argument('--export-4d-format', choices=('nrrd', 'nrrd-slicer', 'nifti'),
                    default='nrrd', dest='export_4d_format',
                    help='file type of the sequence (default: nrrd, the layout '
                         'Imalytics opens; nrrd-slicer for 3D Slicer sequences)')
    args = ap.parse_args(argv)

    cloud, motion, geom, ws, ss, hu = load_run(args.result_dir)
    if args.export_4d > 0:
        from .export4d import LinearHU, export_4d
        export_4d(cloud, motion, geom, ws, ss, LinearHU(*hu),
                  os.path.join(args.result_dir, 'recon.vff'),
                  n_states=args.export_4d, pitch_mm=args.export_4d_mm,
                  fmt=args.export_4d_format)
    out = os.path.join(args.result_dir, 'motion_plots')
    os.makedirs(out, exist_ok=True)
    print(f"cloud {len(cloud):,} primitives; ROI grid {geom['vol_shape']} at "
          f"{geom['dx']:.4f} mm; motion: "
          f"{'none (static run)' if motion is None else f'{motion.n_fixed} fixed + {motion.n_codes} free codes, grid ' + 'x'.join(map(str, motion.shape)) + (f', {motion.n_parts} parts' if motion.n_parts > 1 else '')}")

    slices = {k: state_slices(cloud, motion, geom, ws, ss, hu, v)
              for k, v in breathing_states(motion).items()}
    figs = {}
    figs['motion/gates'] = gates_figure(slices)
    figs['motion/gates'].savefig(os.path.join(out, 'gates.png'))
    print(f"  wrote {out}/gates.png")
    gif = None
    if motion is not None:
        figs['motion/codes_final'] = panels.codes_figure(motion)
        figs['motion/codes_final'].savefig(os.path.join(out, 'codes.png'))
        print(f"  wrote {out}/codes.png")
        if motion.n_parts > 1:
            figs['motion/parts_final'] = panels.parts_figure(motion, cloud.xyz, geom, ws)
            figs['motion/parts_final'].savefig(os.path.join(out, 'parts.png'))
            print(f"  wrote {out}/parts.png")
        if motion.n_fixed:
            figs['motion/field_arrows'] = field_figure(cloud, motion, geom, ws,
                                                       slices['rest (0)'][0])
            figs['motion/field_arrows'].savefig(os.path.join(out, 'field.png'))
            print(f"  wrote {out}/field.png")
            frames = breathing_frames(cloud, motion, geom, ws, ss, hu, n=args.frames)
            gif = write_gif(frames, os.path.join(out, 'breathing.gif'))
            print(f"  wrote {gif} ({frames.shape[0]} frames)")

    if args.wandb_run_id:
        import wandb
        project = os.environ.get('WANDB_PROJECT')
        if not project:
            raise SystemExit('--wandb-run-id needs the WANDB_PROJECT env var (the project is never hardcoded)')
        run = wandb.init(project=project,
                         entity=os.environ.get('WANDB_ENTITY') or None,
                         id=args.wandb_run_id, resume='must')
        payload = {k: wandb.Image(f) for k, f in figs.items()}
        if gif is not None:
            payload['motion/breathing'] = wandb.Video(gif, format='gif')
        run.log(payload)
        run.finish()
        print(f"  logged {sorted(payload)} to W&B run {args.wandb_run_id}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
