"""
Measure a scan's noise model — the detector's Poisson slope and read noise —
standalone, and file it for every noise-weighted reconstruction.

Runs ``ct_core.noise_model`` on the scan's own consecutive views (no repeat
acquisition needed) and writes the same calibration records the learned
driver writes on a ``--loss wls`` run (the default of every learned
algorithm):

    data/calibration/detector_noise_<serial>_<scanTag>.json   always
    data/calibration/detector_noise_<serial>.json             when the read
                                                              noise is DETECTED

The second file is the one that matters: the read noise (counts per raw
pixel) can only be separated from the Poisson term on a scan whose counts
span a wide range — a dense phantom (Scan_1988: lever 4.7x, t = 10-15), not
a mouse (Scan_1510: lever 2.0x, unidentifiable) — and once a phantom has
filed it, every scan on that detector inherits it. So the intended use is
once per detector, on the densest object you have, before the mouse runs:

    ct120-noise-calibration data/scans/Scan_1988
    ct120-noise-calibration data/scans/Scan_1988 --downsample 3

The Poisson slope is binning-specific (module docstring of
``ct_core.noise_model``) and is refit on every run regardless, so the
binning chosen here only matters for what the per-scan record says.
``--force`` re-measures and overwrites even when the detector file holds a
better-conditioned measurement.
"""

import argparse
from pathlib import Path

from .ct_core.errors import ScanDataError, cli_main
from .ct_core.noise_model import (MIN_LEVER, NoiseCalibrationKey,
                                  T_DETECT, estimate_noise_model, fit_record,
                                  read_record, write_record)
from .ct_core.pipeline import add_common_args, prepare_scan
from .ct_core.preprocessing import preprocess_sinogram
from .ct_core.vff_io import detector_serial_from_scan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure a scan's noise model (Poisson slope + read noise) "
                    "from its own projections and file it for every "
                    "noise-weighted reconstruction on that detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(parser)
    parser.add_argument(
        '--force', action='store_true',
        help="Overwrite the detector's calibration file even when it holds a "
             "better-conditioned measurement (higher t) from another scan.")
    parser.add_argument(
        '--calibration-dir', default=None, metavar='DIR',
        help="Where to write the records (default: the shared "
             "data/calibration resolved by ct_core.paths).")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Detector Noise Calibration (stride-curvature estimator)")
    print("=" * 60)

    ctx = prepare_scan(args)
    ds = int(ctx.downsample)
    serial = detector_serial_from_scan(ctx.scan_folder)
    key = NoiseCalibrationKey(
        serial=serial, scan_tag=Path(ctx.scan_folder).name,
        calib_dir=None if args.calibration_dir is None else Path(args.calibration_dir))

    # The sinogram the loss would see, and the counts it was formed from.
    sino = preprocess_sinogram(
        ctx.projections, ctx.bright_field, ctx.dark_field,
        ring_correction=args.ring_correction,
        air_normalization=args.air_normalization,
        soft_clip_sharpness=args.soft_clip_sharpness,
        ring_median_width=args.ring_median_width)
    counts = ctx.projections.astype('float32')
    if ctx.dark_field is not None:
        counts = counts - ctx.dark_field.astype('float32')[None]

    import time
    t0 = time.time()
    try:
        fit = estimate_noise_model(sino, counts, downsample=ds)
    except ValueError as e:
        raise ScanDataError(f"noise model could not be measured: {e}")
    elapsed = time.time() - t0

    print(f"\n  {fit.n_samples:,} rays, {fit.n_bins_used}/{fit.n_bins} count "
          f"bins, counts {fit.count_min:.0f}..{fit.count_max:.0f} at ds{ds} "
          f"(lever {fit.lever:.2f}x), {elapsed:.1f} s")
    print(f"  Poisson slope at ds{ds}: {fit.poisson_slope:.4g} +/- "
          f"{fit.poisson_slope_se:.2g}")
    print(f"  read noise: sigma_r^2 = {fit.read_noise_var:.4g} +/- "
          f"{fit.read_noise_var_se:.2g} per raw pixel (t = {fit.t_read_noise:.1f})"
          f"  ->  sigma_r = {fit.read_noise:.2f} counts")
    print(f"  R2 {fit.r2:.4f}")
    if fit.read_noise_detected:
        print(f"  read noise DETECTED (t >= {T_DETECT:g}, lever >= {MIN_LEVER:g}x)")
    else:
        print("  read noise NOT identifiable on this scan:")
        for why in fit.reasons:
            print(f"    - {why}")

    if not serial:
        print("\n  No detector serial in the projection headers — nothing "
              "filed. The numbers above are still valid for pinning "
              "(--wls-slope / --wls-read-noise).")
        return

    rec = fit_record(fit, key, elapsed)
    sp = key.scan_path()
    write_record(sp, rec)
    print(f"\n  wrote {sp}")
    dp = key.detector_path()
    old = read_record(dp)
    if not fit.read_noise_detected:
        if old is not None:
            print(f"  detector file {dp.name} left as is (sigma_r = "
                  f"{float(old['read_noise']):.2f} from {old.get('scan', '?')})")
        else:
            print(f"  no detector file written: this scan cannot determine "
                  f"the read noise. Calibrate on a denser object (count "
                  f"range >= {MIN_LEVER:g}x).")
        return
    if (old is None or args.force
            or float(rec['t_read_noise']) >= float(old.get('t_read_noise', 0.0))
            or not old.get('read_noise_detected', False)):
        write_record(dp, rec)
        print(f"  wrote {dp} — every scan on this detector now inherits "
              f"sigma_r = {fit.read_noise:.2f} counts")
    else:
        print(f"  kept {dp.name}: its measurement from {old.get('scan', '?')} "
              f"(t = {float(old['t_read_noise']):.1f}) is better conditioned "
              f"than this one (t = {fit.t_read_noise:.1f}); --force overrides")


def cli() -> None:
    """Console-script entry point (``[project.scripts]`` in pyproject.toml)."""
    cli_main(main)


if __name__ == '__main__':
    cli()
