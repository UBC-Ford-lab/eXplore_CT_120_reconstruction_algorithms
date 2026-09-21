#!/usr/bin/env python3
"""Simulate a scan of a known volume and write it as a scan folder.

    ct120-simulate-scan VOLUME.vff --template data/scans/Scan_1988 --output data/scans/Sim_1988

The volume is one this package wrote (HU with its ``.json`` sidecar); the
template is a real scan whose acquisition (views, step, bright/dark, header)
the synthetic one copies. The output loads in every driver with no flag —
``ct120-fdk data/scans/Sim_1988`` — and the reconstruction can then be
scored against the very volume it was made from.

For a MOVING object see ``inr_pipeline.dynamic_gaussian.phantom`` in muNeRF,
which feeds ``simulate.simulate_scan`` a volume per view.
"""
import argparse
import sys

from .ct_core.errors import ConfigError
from .simulate import simulate_scan
from .simulate.forward import sampler_from_volume_file


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('volume', help='a .vff with its .json sidecar (HU, grid, calibration)')
    ap.add_argument('--template', required=True, help='real scan folder whose acquisition to copy')
    ap.add_argument('--output', required=True, help='scan folder to write')
    ap.add_argument('--binning', type=int, default=1,
                    help='write the detector pooled by this factor (3: 9x fewer rays; reconstruct '
                         'at --downsample 1). The noise is the measured noise of a pooled detector.')
    ap.add_argument('--views', type=int, default=None, help='number of views (default: the template\'s)')
    ap.add_argument('--step-deg', type=float, default=None, help='angular step (default: the template\'s)')
    ap.add_argument('--phases', type=int, default=1, help='gated phases to write (the same still object in each)')
    ap.add_argument('--no-noise', action='store_true', help='noiseless counts')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--psi-deg', type=float, default=0.0, help='a KNOWN in-plane detector rotation to simulate')
    ap.add_argument('--cor-offset-px', type=float, nargs=2, default=(0.0, 0.0), metavar=('DA', 'DB'),
                    help='a KNOWN centre-of-rotation offset (pixels) to simulate')
    ap.add_argument('--samples-per-mm', type=float, default=None,
                    help='quadrature density along a ray (default: 2 per voxel)')
    ap.add_argument('--device', default=None, help='cuda / cpu (default: cuda if available)')
    ap.add_argument('--serial', default='SIMULATED', help='detector serial written to the frames')
    ap.add_argument('--specimen', default='SIMULATED')
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.binning < 1:
        raise ConfigError("--binning must be >= 1")
    print(f"Loading {args.volume} ...")
    sampler = sampler_from_volume_file(args.volume, device=device, verbose=True)
    nx, ny, nz = sampler.shape
    print(f"  object grid {nx}x{ny}x{nz} at {sampler.geom['dx']:.4f} mm, "
          f"total attenuation {sampler.total_attenuation():.3f} mm^2")
    folder = simulate_scan(
        sampler, args.template, args.output, binning=args.binning, n_views=args.views,
        step_deg=args.step_deg, phases=args.phases, psi_deg=args.psi_deg,
        cor_offset_px=tuple(args.cor_offset_px), noise=not args.no_noise, seed=args.seed,
        samples_per_mm=args.samples_per_mm, device=device, serial=args.serial,
        specimen=args.specimen)
    print(f"Synthetic scan written to: {folder}")
    return 0


def cli() -> None:
    try:
        sys.exit(main())
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    cli()
