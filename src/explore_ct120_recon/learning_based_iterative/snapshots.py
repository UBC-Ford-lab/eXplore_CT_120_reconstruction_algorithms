"""Iteration snapshots and model checkpoints for the shared ray loop.

Opt-in (``run_learned_recon --snapshot-dir``); nothing here runs otherwise.
ONE mechanism for every representation trained by
``LearnedReconstructor.reconstruct`` — the voxel grid, the coordinate MLP,
anything registered later — so two representations trained side by side are
captured at the same iterations by the same code.

WHAT IS WRITTEN, per snapshot iteration, in the format muNeRF's
``inr_pipeline/snapshot_voxel`` already uses (which keeps working unchanged):

  * ``iter_%08d_mu.npy`` — the field on the snapshot grid, (Nx, Ny, Nz) mu in
    mm^-1, in the orientation and on the lattice of the delivered volume
    BEFORE HU calibration. Not HU: the HU map is fitted from a volume's own
    histogram, and refitting per frame puts a calibration drift into every
    series (see ``run_learned_recon._lr_stage_views_fn``). Apply the finished
    run's anchors (recon.json ``hu_calibration``) to the whole series.
    With ``slices`` set, ``iter_%08d_slices.npz`` (one 2-D array per chosen
    slice) is written INSTEAD of the volume.
  * ``iter_%08d.json`` — that frame's grid and its record;
  * optionally ``iter_%08d_model.pt`` — the model weights (see checkpoints);
  * ``index.csv`` and ``manifest.json`` — one row per frame: iteration, wall
    time, training seconds so far (snapshot time excluded), the training
    loss, the held-out score and the test-view scores at that iteration, and
    the seconds the snapshot itself cost. Rewritten after every frame, so a
    killed job still has a usable set.

Frames continue to the STOPPING iteration — the over-trained iterates after
the best one are the point of a "converged but wrong" figure — and the
manifest records which iteration was restored and delivered. The delivered
volume is also written once more at the end (``delivered_iter_%08d_mu.npy``)
from the volume the run actually exports; when that iteration also has a
mid-run frame, the largest difference between the two is recorded
(``delivered_vs_snapshot_max_abs``) — the check that a snapshot at the
delivered iterate IS the delivered volume.

THE GRID. The export ROI (``ct_core.support.export_grid_geometry``, the same
lattice arithmetic the shipped crop uses), optionally cut down to ``box``
(isocentre mm) — on the same lattice, so a box frame is a sub-array of the
full frame voxel for voxel. A representation that can be evaluated at
arbitrary points (the default ``export_volume``) renders ONLY the snapshot
grid; one that overrides ``export_volume`` (the voxel grid, whose parameters
are the volume) exports and is cropped.

NON-INTERFERENCE. A snapshot renders without gradients through
``volume.render_volume``, which restores the module's train/eval state;
it draws no random numbers (the held-out and test-view renders are
deterministic, stratified=False), runs outside autocast exactly as the final
export does, and never touches the optimizer, the LR schedule or the
stopper. The training loss it records is the step's own loss, read with
``.item()`` only at a snapshot iteration.

CHECKPOINTS (``save_model_checkpoint``) hold the module's state dict plus
everything needed to rebuild it: the algorithm name and modules, its
constructor options, the reconstruction geometry and the model domain.
``run_render_checkpoint`` re-renders and exports a volume from one without
retraining. Training RESUME is not supported (no optimizer state is saved).
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import torch

#: How the frames are named — shared with ``inr_pipeline/snapshot_voxel``.
FRAME = "iter_{:08d}"


@dataclass
class SnapshotConfig:
    """What to capture and when. Iterations are 1-based (after that many
    optimizer steps), the convention of every iteration the loop reports."""
    directory: str
    iterations: tuple = ()
    every: int = 0
    first: int = 0
    ratio: float = 0.0
    max_frames: int = 200
    box: tuple | None = None
    slices: dict | None = None
    dtype: str = "float16"
    checkpoints: bool = False
    #: Extra metadata a model checkpoint needs to be rebuilt (algorithm name,
    #: modules, constructor options); filled by the driver.
    checkpoint_meta: dict = field(default_factory=dict)

    def settings(self) -> dict:
        return {'directory_name': Path(self.directory).name,
                'iterations': [int(i) for i in self.iterations],
                'every': int(self.every), 'first': int(self.first),
                'ratio': float(self.ratio), 'max_frames': int(self.max_frames),
                'box_mm': None if self.box is None else [float(v) for v in self.box],
                'slices': self.slices, 'dtype': str(self.dtype),
                'checkpoints': bool(self.checkpoints)}


def schedule(cfg: SnapshotConfig, iterations: int) -> list:
    """The union of the explicit list, every N, and the geometric ladder,
    within [1, iterations], sorted, capped at ``max_frames``."""
    last = int(iterations)
    out = {int(i) for i in cfg.iterations if 1 <= int(i) <= last}
    if int(cfg.every) > 0:
        out.update(range(int(cfg.every), last + 1, int(cfg.every)))
    if int(cfg.first) > 0 and float(cfg.ratio) > 1.0:
        x = float(cfg.first)
        while x <= last:
            out.add(int(round(x)))
            x *= float(cfg.ratio)
    out = sorted(out)
    if len(out) > int(cfg.max_frames):
        # keep the ends, thin the middle evenly
        keep = np.unique(np.round(np.linspace(0, len(out) - 1,
                                              int(cfg.max_frames))).astype(int))
        out = [out[k] for k in keep]
    return out


def box_grid(geometry: dict, box) -> dict:
    """``geometry`` cut to ``box`` (x0 x1 y0 y1 z0 z1, mm) on ITS lattice."""
    from ..ct_core.support import export_grid_geometry
    if box is None:
        return geometry
    g = dict(geometry)
    g['export_roi'] = {'x_min': float(box[0]), 'x_max': float(box[1]),
                       'y_min': float(box[2]), 'y_max': float(box[3]),
                       'z_min': float(box[4]), 'z_max': float(box[5])}
    return export_grid_geometry(g)


def subgrid_offsets(outer: dict, inner: dict) -> tuple:
    """Index offsets (i0, j0, k0) of grid ``inner`` inside ``outer``; both on
    the same lattice. Raises if they are not."""
    out = []
    for ax, d in ((0, 'dx'), (1, 'dx'), (2, 'dz')):
        step = float(outer[d])
        lo_o = float(outer['vol_origin'][ax]) - int(outer['vol_shape'][ax]) * step / 2
        lo_i = float(inner['vol_origin'][ax]) - int(inner['vol_shape'][ax]) * step / 2
        f = (lo_i - lo_o) / step
        if abs(f - round(f)) > 1e-3:
            raise ValueError("snapshot grid is not on the reconstruction lattice")
        o = int(round(f))
        if o < 0 or o + int(inner['vol_shape'][ax]) > int(outer['vol_shape'][ax]):
            raise ValueError("snapshot box extends past the reconstruction grid")
        out.append(o)
    return tuple(out)


def slice_grids(grid: dict, slices: dict) -> dict:
    """One single-plane geometry per requested slice of ``grid``.
    ``slices`` = {'axial': [k...], 'coronal': [j...], 'sagittal': [i...]},
    indices into ``grid``."""
    out = {}
    axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    for name, idx in (slices or {}).items():
        ax = axes[name]
        n = int(grid['vol_shape'][ax])
        step = float(grid['dz'] if ax == 2 else grid['dx'])
        for k in idx:
            k = int(k)
            if not 0 <= k < n:
                raise ValueError(f"{name} slice {k} outside 0..{n - 1}")
            g = dict(grid)
            shape = list(int(v) for v in grid['vol_shape'])
            origin = list(float(v) for v in grid['vol_origin'])
            origin[ax] = origin[ax] - n * step / 2 + (k + 0.5) * step
            shape[ax] = 1
            g['vol_shape'], g['vol_origin'] = tuple(shape), tuple(origin)
            out[f"{name}_{k:04d}"] = (g, ax, k)
    return out


def frame_bytes(grid: dict, cfg: SnapshotConfig) -> int:
    """Disk bytes of one frame (volume or slices), for the plan printout."""
    item = np.dtype(cfg.dtype).itemsize
    if cfg.slices:
        n = 0
        for name, (g, _, _) in slice_grids(grid, cfg.slices).items():
            n += int(np.prod(g['vol_shape']))
        return n * item
    return int(np.prod([int(v) for v in grid['vol_shape']])) * item


class SnapshotWriter:
    """Owns the directory, the schedule and the index; the trainer calls
    ``due`` every step and ``write`` when it is."""

    def __init__(self, cfg: SnapshotConfig, geometry: dict, iterations: int,
                 algorithm: str):
        from ..ct_core.support import export_grid_geometry
        self.cfg = cfg
        self.dir = Path(cfg.directory)
        self.dtype = np.dtype(cfg.dtype)
        self.recon_geometry = dict(geometry)
        self.grid = box_grid(export_grid_geometry(dict(geometry)), cfg.box)
        self.offsets = subgrid_offsets(self.recon_geometry, self.grid)
        self.slice_geoms = (slice_grids(self.grid, cfg.slices)
                            if cfg.slices else None)
        self.targets = schedule(cfg, iterations)
        self._next = 0
        self.algorithm = algorithm
        self.rows: list = []
        self.frames: list = []
        self.seconds = 0.0
        self.final: dict = {}
        self.dir.mkdir(parents=True, exist_ok=True)
        nb = frame_bytes(self.grid, cfg)
        shape = tuple(int(v) for v in self.grid['vol_shape'])
        print(f"  Snapshots: {len(self.targets)} planned -> {self.dir} "
              f"({'slices ' + ', '.join(self.slice_geoms) if self.slice_geoms else 'grid ' + 'x'.join(map(str, shape))}, "
              f"{cfg.dtype}, {nb / 2**20:.1f} MiB per frame"
              + (", + model weights per frame" if cfg.checkpoints else "")
              + (f"; first {self.targets[0]}, last {self.targets[-1]}"
                 if self.targets else "") + ")")

    def due(self, it1: int) -> bool:
        if self._next >= len(self.targets) or it1 < self.targets[self._next]:
            return False
        while self._next < len(self.targets) and self.targets[self._next] <= it1:
            self._next += 1
        return True

    # ---------------------------------------------------------------- data --
    def _arrays(self, render_grid, export_full):
        """{name: array} for this frame. ``render_grid(geom)`` evaluates the
        model on a grid; ``export_full()`` returns the whole reconstruction
        grid (used when the representation does not render points)."""
        targets = (self.slice_geoms if self.slice_geoms
                   else {'volume': (self.grid, None, None)})
        out = {}
        full = None
        for name, (g, ax, k) in targets.items():
            if render_grid is not None:
                arr = render_grid(g)
            else:
                if full is None:
                    full = export_full()
                i0, j0, k0 = subgrid_offsets(self.recon_geometry, g)
                nx, ny, nz = (int(v) for v in g['vol_shape'])
                arr = full[i0:i0 + nx, j0:j0 + ny, k0:k0 + nz]
            arr = np.asarray(arr, dtype=np.float32)
            if ax is not None:
                arr = np.squeeze(arr, axis=ax)
            out[name] = np.ascontiguousarray(arr, dtype=self.dtype)
        return out

    def _save(self, stem: str, arrays: dict) -> str:
        if 'volume' in arrays:
            name = f"{stem}_mu.npy"
            np.save(self.dir / name, arrays['volume'])
        else:
            name = f"{stem}_slices.npz"
            np.savez(self.dir / name, **arrays)
        return name

    def write(self, it1: int, *, render_grid, export_full, loss=None,
              train_seconds=None, holdout=None, test=None,
              checkpoint_fn=None) -> None:
        t0 = time.time()
        stem = FRAME.format(int(it1))
        arrays = self._arrays(render_grid, export_full)
        name = self._save(stem, arrays)
        ckpt = None
        if self.cfg.checkpoints and checkpoint_fn is not None:
            ckpt = f"{stem}_model.pt"
            checkpoint_fn(self.dir / ckpt, int(it1))
        cost = time.time() - t0
        self.seconds += cost
        rec = {'file': name, 'iteration': int(it1), 'units': 'mu',
               'dtype': str(self.dtype), **self._grid_record(),
               'model_checkpoint': ckpt}
        (self.dir / f"{stem}.json").write_text(json.dumps(rec, indent=2))
        row = {'iteration': int(it1), 'file': name,
               'wall_time_s': round(time.time(), 3),
               'train_seconds': (None if train_seconds is None
                                 else round(float(train_seconds), 3)),
               'train_loss': None if loss is None else float(loss),
               'snapshot_seconds': round(cost, 3)}
        for k in ('mse', 'psnr', 'ssim'):
            row[f'holdout_{k}'] = None if holdout is None else float(holdout[k])
        if test:
            for k in ('mse', 'psnr', 'ssim'):
                row[f'test_mean_{k}'] = float(test['mean'][k])
            row['test_views'] = test['views']
        self.rows.append(row)
        self.frames.append(rec)
        self._flush()
        print(f"  snapshot: iteration {it1} -> {name} ({cost:.1f} s"
              + (f"; holdout psnr {holdout['psnr']:.2f}" if holdout else "")
              + (f"; test psnr {test['mean']['psnr']:.2f}" if test else "")
              + ")", flush=True)

    def write_delivered(self, volume_full, delivered_iter: int,
                        stop_iter: int, restored: bool) -> None:
        """The volume the run exports, as one more frame, and the check that
        a mid-run frame at the same iteration agrees with it."""
        t0 = time.time()
        g = self.grid
        i0, j0, k0 = self.offsets
        nx, ny, nz = (int(v) for v in g['vol_shape'])
        arrays = {}
        if self.slice_geoms:
            for name, (sg, ax, k) in self.slice_geoms.items():
                a, b, c = subgrid_offsets(self.recon_geometry, sg)
                sx, sy, sz = (int(v) for v in sg['vol_shape'])
                arr = np.squeeze(volume_full[a:a + sx, b:b + sy, c:c + sz], axis=ax)
                arrays[name] = np.ascontiguousarray(arr, dtype=self.dtype)
        else:
            arrays['volume'] = np.ascontiguousarray(
                volume_full[i0:i0 + nx, j0:j0 + ny, k0:k0 + nz], dtype=self.dtype)
        name = self._save(f"delivered_{FRAME.format(int(delivered_iter))}", arrays)
        diff = None
        mid = [r for r in self.rows if r['iteration'] == int(delivered_iter)]
        if mid:
            f = self.dir / mid[0]['file']
            if f.suffix == '.npy':
                prev = {'volume': np.load(f)}
            else:
                with np.load(f) as z:
                    prev = {k: z[k] for k in z.files}
            diff = max(float(np.max(np.abs(prev[k].astype(np.float32)
                                           - arrays[k].astype(np.float32))))
                       for k in arrays)
        self.seconds += time.time() - t0
        self.final = {'delivered_iter': int(delivered_iter),
                      'stop_iter': int(stop_iter), 'restored': bool(restored),
                      'delivered_file': name,
                      'delivered_vs_snapshot_max_abs': diff}
        self._flush()
        print(f"  snapshot: delivered iterate {delivered_iter} -> {name}"
              + (f"; max |delivered - mid-run frame| = {diff:.3g} mm^-1"
                 if diff is not None else ""))

    # -------------------------------------------------------------- record --
    def _grid_record(self) -> dict:
        g = self.grid
        rec = {'volume_shape': [int(v) for v in g['vol_shape']],
               'vol_origin_mm': [float(v) for v in g['vol_origin']],
               'voxel_size_mm': {'xy': float(g['dx']), 'z': float(g['dz'])}}
        if self.slice_geoms:
            rec['slices'] = {k: int(idx) for k, (_, _, idx)
                             in self.slice_geoms.items()}
        return rec

    def summary(self) -> dict:
        return {'directory_name': self.dir.name, 'frames': len(self.rows),
                'snapshot_seconds': round(self.seconds, 1),
                **self.final, 'settings': self.cfg.settings()}

    def _flush(self) -> None:
        (self.dir / 'manifest.json').write_text(json.dumps({
            'created': date.today().isoformat(), 'algorithm': self.algorithm,
            'units': 'mu', 'grid': self._grid_record(),
            'targets': self.targets, 'settings': self.cfg.settings(),
            'snapshot_seconds_total': round(self.seconds, 3),
            **self.final, 'frames': self.rows}, indent=2))
        cols = ['iteration', 'file', 'wall_time_s', 'train_seconds',
                'train_loss', 'holdout_mse', 'holdout_psnr', 'holdout_ssim',
                'test_mean_mse', 'test_mean_psnr', 'test_mean_ssim',
                'snapshot_seconds']
        with open(self.dir / 'index.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in self.rows:
                w.writerow(r)


# --------------------------------------------------------------------------
# Model checkpoints
# --------------------------------------------------------------------------

def domain_record(domain) -> dict:
    return {'shape': domain.shape,
            'aabb_min': [float(v) for v in domain.aabb_min],
            'aabb_max': [float(v) for v in domain.aabb_max],
            'radius_xy': (None if domain.radius_xy is None
                          else float(domain.radius_xy)),
            'center_xy': [float(v) for v in domain.center_xy],
            'ray_clip': getattr(domain, 'ray_clip', 'domain')}


def domain_from_record(rec: dict):
    from .scene import ModelDomain
    return ModelDomain(shape=rec['shape'],
                       aabb_min=torch.tensor(rec['aabb_min'], dtype=torch.float32),
                       aabb_max=torch.tensor(rec['aabb_max'], dtype=torch.float32),
                       radius_xy=rec['radius_xy'],
                       center_xy=tuple(rec['center_xy']),
                       ray_clip=rec.get('ray_clip', 'domain'))


def save_checkpoint(path, model, *, geometry, domain, iteration: int,
                    meta: dict | None = None) -> None:
    """Weights + what is needed to rebuild the module (no optimizer state)."""
    from .training import unwrap_model
    state = {k: v.detach().cpu() for k, v in
             unwrap_model(model).state_dict().items()}
    geom = {k: v for k, v in dict(geometry).items()
            if isinstance(v, (int, float, str, bool, list, tuple, dict,
                              type(None)))}
    torch.save({'format': 'explore_ct120_recon.model_checkpoint/1',
                'iteration': int(iteration), 'state_dict': state,
                'geometry': geom, 'domain': domain_record(domain),
                **(meta or {})}, str(path))
