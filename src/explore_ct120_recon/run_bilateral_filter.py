"""
Bilateral-filter a finished VFF volume exactly as the MATLAB ``filter_vff.m``.

    filter_vff(infile)        ->  ct120-bilateral-filter INFILE
    filter_vff(infile, sd)    ->  ct120-bilateral-filter INFILE SD

Writes ``<infile minus .vff>_imbilatfilt.vff``: the input's header, every
axial slice through ``imbilatfilt`` with MATLAB's defaults (spatial sigma 1
voxel, 5 x 5 neighbourhood, replicate padding) and ``degreeOfSmoothing =
2 * sd^2``, int16 big-endian. Without SD the noise variance is measured the
MATLAB way, from the fixed patch ``im(20:40,160:180,100)``. See
``ct_core/bilateral.py`` for each step and the one assumption (fwrite's
double-to-int16 rounding).

Differences from the MATLAB function, none of which touch the output VFF:

  * an existing output file is not overwritten without ``--overwrite``, and
    the input file never is;
  * if the input has a ``<volume>.json`` sidecar (every volume this package
    writes does), ``<output>.json`` is written too — the input's, with
    ``vff``/``created`` updated and an ``imbilatfilt`` block recording the
    parameters — so downstream tools still find the grid and the HU map;
  * the two preview figures of slice 150 are not drawn.

No W&B run: this reconstructs nothing. To log the filtered volume, run
``ct120-volume-report`` on the output.

Usage:
    ct120-bilateral-filter data/results/Scan_1988_voxel-adam_ds3_075/recon.vff 229
"""

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

from .ct_core.bilateral import VFFFilterError, filter_vff
from .ct_core.errors import ConfigError, cli_main
from .ct_core.volume_report import load_sidecar


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('infile', help='input .vff volume')
    parser.add_argument('sd', nargs='?', type=float, default=None,
                        help='noise standard deviation in the volume\'s units '
                             '(HU); degreeOfSmoothing = 2*sd^2. Omitted: '
                             'measured from im(20:40,160:180,100) as MATLAB does')
    parser.add_argument('--out', default=None,
                        help='output path (default: <infile minus .vff>'
                             '_imbilatfilt.vff)')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace an existing output file')
    parser.add_argument('--workers', type=int, default=None,
                        help='slices filtered in parallel (default: CPU count, '
                             'at most 32); does not change the result')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.infile)
    if not src.exists():
        raise ConfigError(f"volume not found: {src}")
    if src.suffix.lower() != '.vff':
        raise ConfigError(f"{src.name}: filter_vff reads .vff files only")

    out = Path(args.out) if args.out else Path(str(src)[:-4] + '_imbilatfilt.vff')
    if out.resolve() == src.resolve():
        raise ConfigError("--out is the input file; the input is never "
                          "overwritten")
    sidecar = load_sidecar(src, verbose=False)
    side_out = out.with_suffix('.json')
    targets = [out] + ([side_out] if sidecar else [])
    for p in targets:
        if p.exists() and not args.overwrite:
            raise ConfigError(f"{p} exists (pass --overwrite to replace it)")

    start = time.time()
    print(f"filter_vff: {src}")
    try:
        out, info = filter_vff(src, args.sd, outfile=out, workers=args.workers)
    except VFFFilterError as e:
        raise ConfigError(f"{src.name}: {e}") from e
    print(f"Filtered volume saved to: {out}")

    if sidecar:
        record = dict(sidecar)
        record['vff'] = out.name
        record['created'] = date.today().isoformat()
        record['imbilatfilt'] = {**info, 'source': src.name,
                                 'method': 'filter_vff.m (MATLAB imbilatfilt)'}
        tmp = side_out.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(record, indent=2))
        os.replace(tmp, side_out)
        print(f"Volume metadata saved to: {side_out}")
    print(f"\nDone in {(time.time() - start) / 60:.2f} minutes.")
    return out


def cli() -> None:
    """Console-script entry point (``[project.scripts]`` in pyproject.toml)."""
    cli_main(main)


if __name__ == '__main__':
    cli()
