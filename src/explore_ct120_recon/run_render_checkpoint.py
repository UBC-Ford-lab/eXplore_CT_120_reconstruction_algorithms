"""Re-render and export a volume from a saved model checkpoint, no training.

    ct120-render-checkpoint RUN_model.pt --output rerender.vff
    ct120-render-checkpoint snaps/iter_00020000_model.pt \\
        --algorithm-module inr_pipeline.algorithms --output it20k.vff

A checkpoint (``run_learned_recon --save-checkpoint`` for the delivered
iterate, ``--snapshot-checkpoints`` for every snapshot) holds the module's
weights, the algorithm name and modules, its constructor options, the
reconstruction geometry and the model domain. This rebuilds the module
through the algorithm registry exactly as the run built it, loads the weights,
evaluates it on the export grid (or ``--roi``) and writes VFF + sidecar.

HU: the delivered-iterate checkpoint carries the run's own anchors and they
are applied by default, so the re-render reproduces the shipped calibration;
a snapshot checkpoint has none and is fitted (``--hu-calibration auto``,
or pass ``--anchors-from <run>.json`` to put it on the run's scale).

Re-render only: no optimizer state is saved, so training cannot resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .ct_core.errors import ConfigError, cli_main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('checkpoint', help='a *_model.pt written by run_learned_recon')
    p.add_argument('--output', required=True, help='VFF to write')
    p.add_argument('--algorithm-module', action='append', default=[],
                   metavar='MODULE', help='Module registering the algorithm '
                   '(default: the ones recorded in the checkpoint).')
    p.add_argument('--roi', type=float, nargs=6, default=None,
                   metavar=('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'),
                   help='Render this box (isocentre mm, on the run\'s lattice) '
                        'instead of the run\'s export ROI.')
    p.add_argument('--hu-calibration', choices=('checkpoint', 'auto'),
                   default=None, help='Anchors: the checkpoint\'s own '
                   '(default when it has them) or fitted to this volume.')
    p.add_argument('--anchors-from', default=None, metavar='JSON',
                   help='Apply the hu_calibration anchors of this sidecar.')
    p.add_argument('--save-mu', action='store_true',
                   help='Also write the raw attenuation as <output>_mu.npy.')
    p.add_argument('--gpu-index', type=int, default=0)
    return p.parse_args(argv)


def load_model(ckpt: dict, modules=(), device='cpu'):
    """(reconstructor, model, domain) rebuilt from a checkpoint dict."""
    import importlib
    from .learning_based_iterative import get_algorithm
    from .learning_based_iterative.snapshots import domain_from_record
    for m in list(modules) or list(ckpt.get('algorithm_modules') or []):
        importlib.import_module(m)
    try:
        algorithm = get_algorithm(ckpt['algorithm'])
    except (KeyError, ValueError) as e:
        raise ConfigError(f"cannot rebuild the model: {e}. Pass the module "
                          f"that registers it with --algorithm-module.") from None
    geometry = dict(ckpt['geometry'])
    for k in ('vol_shape', 'vol_origin'):
        geometry[k] = tuple(geometry[k])
    rec = algorithm.reconstructor(projections=None, angles=None,
                                  geometry=geometry, loss='mse',
                                  crossval=False, **dict(ckpt.get('options') or {}))
    domain = domain_from_record(ckpt['domain'])
    device = torch.device(device)
    model = rec.build_model(domain, device).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ckpt['state_dict'].items()})
    model.eval()
    return rec, model, domain


def render(rec, model, domain, device, grid: dict) -> np.ndarray:
    """The field on ``grid`` (same lattice as the run): rendered at the grid's
    points for a representation with the default export, cropped from its
    own export otherwise (a voxel grid)."""
    from .learning_based_iterative.trainer import LearnedReconstructor
    from .learning_based_iterative.volume import render_volume
    from .learning_based_iterative.snapshots import subgrid_offsets
    if type(rec).export_volume is LearnedReconstructor.export_volume:
        return render_volume(model, grid, domain, device=device)
    full = rec.export_volume(model, domain, device)
    i0, j0, k0 = subgrid_offsets(rec.geometry, grid)
    nx, ny, nz = (int(v) for v in grid['vol_shape'])
    return np.ascontiguousarray(full[i0:i0 + nx, j0:j0 + ny, k0:k0 + nz])


def main(argv=None):
    from .ct_core.hu_calibration import HUAnchors
    from .ct_core.scan_setup import postprocess_and_save
    from .ct_core.support import export_grid_geometry
    from .learning_based_iterative.snapshots import box_grid

    args = parse_args(argv)
    path = Path(args.checkpoint)
    if not path.exists():
        raise ConfigError(f"checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    if not str(ckpt.get('format', '')).startswith(
            'explore_ct120_recon.model_checkpoint/'):
        raise ConfigError(f"{path.name} is not a run_learned_recon model checkpoint")
    device = torch.device(f"cuda:{args.gpu_index}" if torch.cuda.is_available()
                          else "cpu")
    rec, model, domain = load_model(ckpt, args.algorithm_module, device)
    geometry = rec.geometry
    grid = (box_grid(dict(geometry, export_roi=None), args.roi) if args.roi
            else export_grid_geometry(dict(geometry)))
    print(f"Re-rendering {ckpt['algorithm']} at iteration {ckpt['iteration']} "
          f"on {tuple(int(v) for v in grid['vol_shape'])}")
    with torch.no_grad():
        mu = render(rec, model, domain, device, grid)

    anchors = None
    mode = args.hu_calibration or ('checkpoint' if ckpt.get('hu_anchors')
                                   else 'auto')
    if args.anchors_from:
        hu = json.loads(Path(args.anchors_from).read_text())['hu_calibration']
        anchors = HUAnchors(value_air=float(hu['anchor_air_value']),
                            value_tissue=float(hu['anchor_tissue_value']),
                            air_hu=float(hu.get('target_air', -1000.0)),
                            tissue_hu=float(hu.get('target_tissue', 120.0)))
    elif mode == 'checkpoint':
        if not ckpt.get('hu_anchors'):
            raise ConfigError("this checkpoint carries no anchors; use "
                              "--hu-calibration auto or --anchors-from")
        anchors = HUAnchors(**ckpt['hu_anchors'])
    meta = {'algorithm': ckpt['algorithm'],
            'rendered_from_checkpoint': path.name,
            'iteration': int(ckpt['iteration'])}
    if ckpt.get('scan'):
        meta['scan'] = ckpt['scan']
    postprocess_and_save(mu, grid, args.output, anchors=anchors,
                         save_mu=args.save_mu, metadata=meta)


def cli() -> None:
    cli_main(main)


if __name__ == '__main__':
    cli()
