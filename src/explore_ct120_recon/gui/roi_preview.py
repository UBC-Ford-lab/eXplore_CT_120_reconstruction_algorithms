"""A coarse, fast FDK of a scan for the page's ROI picker.

The new-job page lets the user drag a box over the specimen instead of
typing six ``--roi`` numbers. For that it needs to SEE the specimen, so this
module reconstructs the whole attenuating field of a scan at a coarse pitch
(0.4 mm voxels, the detector pooled 4x, every second view): a few seconds on
a GPU, well under a minute on a CPU, and a 5 MB volume rather than 2 GB.

The preview goes through the same front half as every reconstruction
(``ct_core.pipeline.prepare_scan``: unwarp, CoR policy, geometry), so the
box the user draws is in exactly the isocentre-centred millimetre frame the
``--roi`` flag takes, and the bounds it returns rebuild the same lattice on
the real run (see ``scan_setup.grid_bounds_mm``). Two shortcuts, both
behind ``args.preview_fast`` and never taken by a driver: each frame is
pooled and unwarped on the GPU as it is read (the drivers unwarp at full
resolution first, which at this pitch is ten seconds of resampling pixels
that are then averaged away), and the grid is the detector's whole field
rather than the measured support. Nothing is written and nothing is
logged: a preview is not a result.

It runs as a SUBPROCESS of the machine that owns the scan, in that
machine's own python (``python -m explore_ct120_recon.gui.roi_preview
'<json spec>'``), the way the slice reader and the input checks do: the
server process never imports torch, a crash never takes the page down, and
a remote machine answers through its connection like everything else.
Stdout carries one JSON line; the drivers' chatter goes to stderr.

Result (one JSON object)::

    shape        [nx, ny, nz]     preview voxels
    spacing      [dx, dx, dz]     mm
    bounds       {x_min .. z_max} outer extent of the preview grid, mm
    scanner_roi  {x_min .. z_max} the vendor's SubVolumeCoordinates box, or null
    window       [lo, hi]         the mu values mapped to 0 and 255
    n_views      views used
    data         base64 of nz*ny*nx uint8, C order [z][y][x]; index up = mm up
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import proc

VOXEL_MM = 0.4         # preview pitch, both axes
VIEW_STEP = 2          # every second projection
DOWNSAMPLE = 4         # detector pooling factor
TIMEOUT = 900          # s, a CPU-only machine on a cold disk
WINDOW_PCT = (0.5, 99.99)   # mu percentiles mapped to 0 and 255: air black,
                            # soft tissue mid-grey, the densest bone white


def spec_for(data_folder: str, *, cwd: Optional[str] = None, scan_folder: Optional[str] = None,
             phase: str = '00', voxel_mm: float = VOXEL_MM, view_step: int = VIEW_STEP,
             downsample: int = DOWNSAMPLE, window_pct=WINDOW_PCT) -> dict:
    return {'data_folder': data_folder, 'cwd': cwd, 'scan_folder': scan_folder or None,
            'phase': phase or '00', 'voxel_mm': float(voxel_mm), 'view_step': int(view_step),
            'downsample': int(downsample), 'window_pct': [float(v) for v in window_pct]}


def compute_preview(spec: dict) -> dict:
    """The preview volume of ``spec`` (see ``spec_for``), in this process.
    Prints what the pipeline prints; the caller decides where that goes."""
    import numpy as np
    import torch

    from ..ct_core.pipeline import add_common_args, prepare_scan, resolve_detector_psi
    from ..ct_core.scan_setup import grid_bounds_mm, parse_crop_boundary
    from ..fdk import FDKReconstructor

    cwd = spec.get('cwd') or os.getcwd()
    folder = os.path.expanduser(str(spec['data_folder']))
    if not os.path.isabs(folder):
        folder = os.path.join(cwd, folder)
    voxel = float(spec.get('voxel_mm') or VOXEL_MM)
    argv = [folder, '--roi', 'full', '--voxel-xy', str(voxel), '--voxel-z', str(voxel),
            '--downsample', str(int(spec.get('downsample') or DOWNSAMPLE)),
            '--phase', str(spec.get('phase') or '00')]
    scan_folder = spec.get('scan_folder')
    if scan_folder:
        scan_folder = os.path.expanduser(str(scan_folder))
        if not os.path.isabs(scan_folder):
            scan_folder = os.path.join(cwd, scan_folder)
        argv += ['--scan-folder', scan_folder]
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(argv)
    args.view_step = int(spec.get('view_step') or VIEW_STEP)
    args.preview_fast = bool(spec.get('fast', True))

    t0 = time.time()
    ctx = prepare_scan(args)
    geometry = ctx.geometry
    # The cached psi only (never measure for a preview): after a vendor
    # unwarp the axis is the vendor's and psi is not applied, as in the drivers.
    if args.cor_mode != 'vendor':
        record = resolve_detector_psi(ctx.scan_folder, verbose=False)
        if record is not None:
            geometry['det_psi_rad'] = np.radians(float(record['psi_deg']))
    t_load = time.time() - t0

    t0 = time.time()
    recon = FDKReconstructor(
        projections=ctx.projections, angles=torch.as_tensor(ctx.angles), geometry=geometry,
        folder_name='roi-preview', quantitative=True,
        bright_field=ctx.bright_field, dark_field=ctx.dark_field,
        clamp_mode='none', soft_clip_transmission=True,
        soft_clip_sharpness=float(getattr(args, 'soft_clip_sharpness', 200.0)),
        upper_clamp=True, upper_clamp_value=1.05, physical_normalization=True,
        filter_cutoff=min(1.0, float(geometry['da']) / float(geometry['dx'])),
        filter_type='hamming', parker_weighting=True,
        ring_correction=False, air_normalization=bool(ctx.air_normalization))
    recon.reconstruct()
    vol = recon.reconstructed_volume
    vol = vol.detach().cpu().numpy() if hasattr(vol, 'detach') else np.asarray(vol)
    vol = np.asarray(vol, dtype=np.float32)                 # (nx, ny, nz)
    t_recon = time.time() - t0

    # 8-bit for the page. The upper end is nearly the densest bone, not the
    # top of the soft tissue: with the window ending just above water
    # (99.8 %) tissue sat at 210 of 255 and bone saturated, so a slice showed
    # a white body with hardly a bone in it. At 99.99 % tissue is mid-grey
    # and bone has the top half; the page's own window sliders do the rest.
    pct_lo, pct_hi = spec.get('window_pct') or WINDOW_PCT
    lo = float(np.percentile(vol, pct_lo))
    hi = float(np.percentile(vol, pct_hi))
    if not hi > lo:
        hi = lo + 1e-6
    scaled = np.clip((vol - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    data = np.ascontiguousarray(scaled.transpose(2, 1, 0))     # [z][y][x]

    nx, ny, nz = (int(v) for v in geometry['vol_shape'])
    scanner_roi = None
    try:
        scanner_roi = parse_crop_boundary(ctx.scan_folder, ctx.xml_header)
    except Exception:                                           # noqa: BLE001
        scanner_roi = None
    return {
        'shape': [nx, ny, nz],
        'spacing': [float(geometry['dx']), float(geometry['dx']), float(geometry['dz'])],
        'bounds': {k: float(v) for k, v in grid_bounds_mm(geometry).items()},
        'scanner_roi': ({k: float(v) for k, v in scanner_roi.items()} if scanner_roi else None),
        'window': [lo, hi],
        'n_views': int(ctx.projections.shape[0]),
        'voxel_mm': voxel,
        'data_folder': folder,
        'seconds': {'load': round(t_load, 1), 'reconstruct': round(t_recon, 1)},
        'data': base64.b64encode(data.tobytes()).decode(),
    }


def main(argv: Optional[list] = None) -> int:
    """``python -m explore_ct120_recon.gui.roi_preview '<json>'``: the result
    as the only line on stdout; everything the pipeline prints on stderr."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('usage: python -m explore_ct120_recon.gui.roi_preview \'{"data_folder": ...}\'', file=sys.stderr)
        return 2
    spec = json.loads(argv[0])
    with contextlib.redirect_stdout(sys.stderr):
        result = compute_preview(spec)
    sys.stdout.write(json.dumps(result) + '\n')
    sys.stdout.flush()
    return 0


def run_preview(spec: dict, *, conn=None, python: Optional[str] = None, timeout: float = TIMEOUT) -> dict:
    """``compute_preview`` in a subprocess of the machine that owns the scan:
    this machine's ``python`` without ``conn``, the remote one through it."""
    text = json.dumps(spec)
    module = 'explore_ct120_recon.gui.roi_preview'
    if conn is None:
        env = {**os.environ, 'PYTHONUTF8': '1', 'WANDB_MODE': 'disabled', 'MPLBACKEND': 'Agg'}
        r = proc.run([python or sys.executable, '-m', module, text], capture_output=True, text=True,
                     timeout=timeout, env=env, cwd=spec.get('cwd') or None)
        if r.returncode != 0:
            raise RuntimeError(_last_error(r.stderr) or f'exit {r.returncode}')
        return json.loads(r.stdout.strip().splitlines()[-1])
    cmd = (f"cd {shlex.quote(spec.get('cwd') or '.')} && WANDB_MODE=disabled MPLBACKEND=Agg "
           f"{shlex.quote(python or 'python3')} -m {module} {shlex.quote(text)}")
    out = conn.check(cmd, timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def _last_error(stderr: str) -> str:
    lines = [ln for ln in (stderr or '').strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if 'Error' in ln or 'error' in ln:
            return ln.strip()[:400]
    return lines[-1].strip()[:400] if lines else ''


class PreviewCache:
    """Finished previews on disk, keyed by machine, scan and preview
    parameters, so reopening the picker costs one file read. The newest
    ``keep`` files stay; the result is a few MB each."""

    def __init__(self, root: Path, keep: int = 24):
        self.root, self.keep = Path(root), keep
        self._lock = threading.Lock()

    @staticmethod
    def key(machine: str, spec: dict) -> str:
        ident = json.dumps([machine, {k: spec.get(k) for k in
                                      ('data_folder', 'cwd', 'scan_folder', 'phase', 'voxel_mm', 'view_step', 'downsample', 'window_pct')}],
                           sort_keys=True)
        return hashlib.sha1(ident.encode()).hexdigest()[:20]

    def path(self, key: str) -> Path:
        return self.root / f'{key}.json'

    def get(self, key: str) -> Optional[dict]:
        p = self.path(key)
        with self._lock:
            if not p.exists():
                return None
            try:
                d = json.loads(p.read_text())
                os.utime(p, None)
                return d
            except (OSError, ValueError):
                return None

    def put(self, key: str, result: dict) -> None:
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self.path(key).write_text(json.dumps(result))
            files = sorted(self.root.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[self.keep:]:
                try:
                    old.unlink()
                except OSError:
                    pass

    def forget(self, key: str) -> None:
        with self._lock:
            try:
                self.path(key).unlink()
            except OSError:
                pass


if __name__ == '__main__':
    sys.exit(main())
