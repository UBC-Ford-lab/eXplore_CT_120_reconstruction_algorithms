"""The "detail" control of a Gaussian job: what a run at a given primitive
spacing will cost, before it is submitted.

A Gaussian cloud's resolution is set by the spacing of its primitives
(``--gauss-seed-spacing``; the count follows from it through
``seeding.seed_count_for_spacing``), bounded below by the detector pixel the
run keeps (``--downsample``) and by the voxel it is exported onto
(``--voxel-xy/-z``, which also floors every primitive's width). The page
offers those three as one preset and shows what they imply:

* the primitive count — the seeding rule applied to the ROI picker's own
  coarse preview of the scan, so the number is the specimen's, not a
  constant;
* the VRAM — the algorithm's own preflight footprint on the run's shapes;
* the time — the iteration count times a seconds-per-primitive-iteration
  rate, measured from the last runs on that machine (the
  ``iter N/M ... X it/s  N gaussians`` lines of their logs; a run that was
  killed still measured its pace) and falling back to the rate of the
  machine's GPU class, or the Tesla P100's before any card is known.

The time half serves every learned algorithm: the voxel and INR runs have no
primitive count, but their past runs on a machine give a seconds-per-
iteration figure, which is what `runtime_estimate` scales. On a Slurm machine
the seconds become a ``--time`` suggestion (`walltime`): the estimate with a
margin, rounded up to a quarter hour, with the scheduler's time class it
falls in — a run whose request is shorter than the estimate is the one that
ends in a TIMEOUT.
"""
from __future__ import annotations

import argparse
import base64
import math
import re
from typing import Optional

import numpy as np

from ..ct_core.calibration import PSF_SIGMA_MM

#: Seconds per (primitive x iteration), whole-run average on a Tesla P100
#: at downsample 3 on Scan_1510: static 1500 it at 2.12 M in 24.7 min,
#: dynamic 32000 it at 2.12 M in 7.4 h — both 4.0e-7 once the fixed part is
#: taken off (the dynamic model's per-view rate drifts down with the visit
#: count, its average does not).
DEFAULT_RATE = {'gaussian': 4.0e-7, 'gaussian_dynamic': 4.0e-7}
DEFAULT_RATE_SOURCE = 'measured on a Tesla P100'
#: The same figure per GPU class, where one has been measured. H100: fir job
#: 87868, the dynamic model at 7.09 M primitives, cumulative 0.96 it/s over
#: 16 500 iterations (2026-10-24). Classes not listed fall back to the P100
#: and the estimate says so.
GPU_RATES = {'p100': 4.0e-7, 'h100': 1.5e-7}
#: Tokens that name a GPU class in a card name or a Slurm gres value.
GPU_CLASSES = ('h100', 'a100', 'v100', 'p100', 'l40s', 'a40', 'rtx 5080', 'rtx 4090', 'rtx 3090', 't4')
#: ``--time`` suggestion: estimate x margin + a fixed allowance, rounded up.
TIME_MARGIN = 1.25
TIME_ALLOWANCE_S = 600.0
TIME_ROUND_S = 900.0
#: The Alliance clusters schedule by walltime class; a request is scheduled
#: with the jobs of its class (3 h, 12 h, 24 h, 72 h, 7 d).
TIME_BUCKETS_H = (3, 12, 24, 72, 168)
#: Load, unwarp, seeding FDK and export, outside the iteration loop.
OVERHEAD_S = 180.0
#: Runs the rate is averaged over, newest first.
HISTORY = 5

ALGORITHMS = ('gaussian', 'gaussian_dynamic')

_RATE_LINE = re.compile(
    r'iter\s+(\d+)/(\d+).*?([\d.]+)\s+it/s\s+([\d,]+)\s+gaussians')
#: Any learned trainer's progress line; the printed rate is the cumulative
#: iterations per second since the loop began.
_ITER_LINE = re.compile(r'iter\s+(\d+)/(\d+).*?([\d.]+)\s+it/s')


def progress_from_log(text: str) -> Optional[tuple[int, float, int]]:
    """``(iterations done, it/s, primitives)`` from the LAST progress line of
    a Gaussian run's log, or None when it has none."""
    found = None
    for m in _RATE_LINE.finditer(text):
        found = m
    if found is None:
        return None
    done = int(found.group(1))
    its = float(found.group(3))
    n = int(found.group(4).replace(',', ''))
    if done <= 0 or its <= 0 or n <= 0:
        return None
    return done, its, n


def rate_from_log(text: str, duration: Optional[float] = None) -> Optional[float]:
    """Seconds per primitive-iteration of a finished run: its wall time over
    the iterations it ran and the primitives it held (the per-view rate
    printed late in a run understates the whole-run average), or the printed
    rate when no wall time is known."""
    p = progress_from_log(text)
    if p is None:
        return None
    done, its, n = p
    if duration and duration > OVERHEAD_S:
        return max(float(duration) - OVERHEAD_S, 1.0) / (done * n)
    return 1.0 / (its * n)


#: Runs whose logs measure a pace: a killed run paced itself as well as a
#: finished one (job 87868 on fir: 16 500 iterations before the time limit),
#: and a running one is the freshest measurement there is.
RATE_STATUSES = ('finished', 'failed', 'cancelled', 'running')


def _wall_time(j) -> Optional[float]:
    """The run's own wall time, when the job record measures it: a local job
    finished normally. A Slurm job's ``started`` is its submission, queue
    wait included, and a killed run's end is not its pace."""
    if getattr(j, 'status', '') != 'finished' or getattr(j, 'remote_id', None):
        return None
    started, finished = getattr(j, 'started', None), getattr(j, 'finished', None)
    if started is None or finished is None:
        return None
    return float(finished) - float(started)


def _runs(jobs, algorithm: str, executor: str):
    for j in jobs:
        if getattr(j, 'driver', '') != 'learned' or getattr(j, 'algorithm', None) != algorithm:
            continue
        if getattr(j, 'executor', '') != executor or getattr(j, 'status', '') not in RATE_STATUSES:
            continue
        yield j


def measured_rate(jobs, algorithm: str, executor: str, read_log) -> Optional[tuple[float, int]]:
    """Median seconds per primitive-iteration over the last runs of
    ``algorithm`` on ``executor`` whose log carries a progress line, with
    how many runs that was. ``read_log(job) -> str``. Finished, failed and
    cancelled runs all count; the wall time refines the figure only where
    the record measures it (`_wall_time`)."""
    rates = []
    for j in _runs(jobs, algorithm, executor):
        try:
            r = rate_from_log(read_log(j) or '', _wall_time(j))
        except Exception:                                   # noqa: BLE001
            r = None
        if r:
            rates.append(r)
        if len(rates) >= HISTORY:
            break
    if not rates:
        return None
    return float(np.median(rates)), len(rates)


def seconds_per_iteration_from_log(text: str) -> Optional[float]:
    """1 / the cumulative it/s of the last progress line of any learned run."""
    found = None
    for m in _ITER_LINE.finditer(text):
        found = m
    if found is None:
        return None
    its = float(found.group(3))
    return 1.0 / its if its > 0 else None


def measured_seconds_per_iteration(jobs, algorithm: str, executor: str, read_log) -> Optional[tuple[float, int]]:
    """Median seconds per iteration over the last runs of ``algorithm`` on
    ``executor``: the time model of a representation without a primitive
    count (voxel, INR), where the per-iteration cost is set by the ray batch
    the machine auto-sized, which the next run on it sizes the same way."""
    vals = []
    for j in _runs(jobs, algorithm, executor):
        try:
            v = seconds_per_iteration_from_log(read_log(j) or '')
        except Exception:                                   # noqa: BLE001
            v = None
        if v:
            vals.append(v)
        if len(vals) >= HISTORY:
            break
    if not vals:
        return None
    return float(np.median(vals)), len(vals)


def gpu_class(names) -> Optional[str]:
    """The GPU class named in a card name or a Slurm gres value
    (``'NVIDIA H100 80GB HBM3'``, ``'h100:1'``, ``'gpu:a100:2'``): the first
    of `GPU_CLASSES` found, or None."""
    for name in ([names] if isinstance(names, str) else list(names or [])):
        text = str(name or '').lower().replace('-', ' ').replace('_', ' ')
        for cls in GPU_CLASSES:
            if cls in text:
                return cls
    return None


def default_rate(algorithm: str, gpu: Optional[str]) -> tuple[float, str]:
    """The seconds per primitive-iteration to assume before a run on the
    machine has been measured, and the words for where it comes from."""
    base = DEFAULT_RATE.get(algorithm, DEFAULT_RATE['gaussian'])
    if gpu in GPU_RATES:
        return base * GPU_RATES[gpu] / GPU_RATES['p100'], f'the {gpu.upper()} figure; no run of {algorithm} on this machine yet'
    if gpu:
        article = 'an' if gpu[0] in 'aehilo' else 'a'
        return base, f'the Tesla P100 figure; nothing measured on {article} {gpu.upper()} yet'
    return base, DEFAULT_RATE_SOURCE


def walltime(seconds: float) -> dict:
    """A ``--time`` request for a run estimated at ``seconds``: the estimate
    times `TIME_MARGIN` plus `TIME_ALLOWANCE_S` (the export and the scheduler's
    own slack), rounded up to a quarter hour, with the walltime class it
    falls in and that class's upper bound.
    """
    raw = float(seconds) * TIME_MARGIN + TIME_ALLOWANCE_S
    rounded = math.ceil(raw / TIME_ROUND_S) * TIME_ROUND_S
    bucket_h = next((b for b in TIME_BUCKETS_H if rounded <= b * 3600), TIME_BUCKETS_H[-1])
    return {'estimate_s': float(seconds), 'raw_s': raw, 'suggested_s': rounded,
            'suggested': format_time(rounded), 'bucket_h': bucket_h, 'bucket': format_time(bucket_h * 3600)}


def format_time(seconds: float) -> str:
    """Slurm's ``D-HH:MM:SS`` / ``HH:MM:SS``."""
    s = int(round(seconds))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return (f'{d}-' if d else '') + f'{h:02d}:{m:02d}:{s:02d}'


def parse_time(text) -> Optional[float]:
    """Seconds from a Slurm time string: ``M``, ``M:S``, ``H:M:S``, ``D-H``,
    ``D-H:M``, ``D-H:M:S``; None when it is not one."""
    if text is None:
        return None
    t = str(text).strip()
    if not t or not re.fullmatch(r'(\d+-)?\d+(:\d+){0,2}', t):
        return None
    days = 0
    if '-' in t:
        d, t = t.split('-', 1)
        days = int(d)
        parts = [int(x) for x in t.split(':')]
        while len(parts) < 3:
            parts.append(0)                              # D-H, D-H:M count down from hours
        h, m, s = parts
    else:
        parts = [int(x) for x in t.split(':')]
        if len(parts) == 1:
            h, m, s = 0, parts[0], 0                     # minutes
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]              # minutes:seconds
        else:
            h, m, s = parts
    return float(days * 86400 + h * 3600 + m * 60 + s)


def args_for(parser: argparse.ArgumentParser, command: list) -> argparse.Namespace:
    """The namespace the driver would run with for a GUI command line
    (``build_command`` output): every default the algorithm re-declares is
    in it, so the estimate reads the same flags the run does."""
    argv = list(command)
    for i, tok in enumerate(argv):
        if tok == '-m':
            argv = argv[i + 2:]
            break
    known = {o for a in parser._actions for o in a.option_strings}
    if '--algorithm' not in known:
        while '--algorithm' in argv:
            i = argv.index('--algorithm')
            del argv[i:i + 2]
    ns, _rest = parser.parse_known_args(argv)
    return ns


def roi_box(roi) -> Optional[tuple]:
    """``((x0, y0, z0), (x1, y1, z1))`` from a six-number ``--roi``, else None."""
    if not isinstance(roi, (list, tuple)) or len(roi) != 6:
        return None
    try:
        v = [float(x) for x in roi]
    except (TypeError, ValueError):
        return None
    return ((v[0], v[2], v[4]), (v[1], v[3], v[5]))


def preview_volume(preview: dict) -> np.ndarray:
    """The picker's preview as a float32 (nx, ny, nz) array in its own
    windowed units — an affine map of mu, which the spacing rule (a ratio of
    weights) does not see."""
    nz, ny, nx = (int(v) for v in reversed(preview['shape']))
    raw = np.frombuffer(base64.b64decode(preview['data']), dtype=np.uint8)
    return np.ascontiguousarray(raw.reshape(nz, ny, nx).transpose(2, 1, 0)).astype(np.float32)


def primitive_count(preview: dict, *, spacing_mm: float, roi, roi_weight: float,
                    noise_floor_quantile: float = 0.5) -> int:
    from ..learning_based_iterative.gaussian.seeding import seed_count_for_spacing
    b = preview['bounds']
    sp = preview['spacing']
    geometry = {'dx': float(sp[0]), 'dz': float(sp[2]),
                'vol_origin': ((b['x_min'] + b['x_max']) / 2, (b['y_min'] + b['y_max']) / 2,
                               (b['z_min'] + b['z_max']) / 2)}
    box = roi_box(roi)
    if box is not None:
        # the seed itself widens the box by 1 mm (seed_from_volume's roi_box_mm(1.0))
        box = (tuple(v - 1.0 for v in box[0]), tuple(v + 1.0 for v in box[1]))
    try:
        return int(seed_count_for_spacing(preview_volume(preview), geometry=geometry,
                                          spacing_mm=float(spacing_mm),
                                          noise_floor_quantile=noise_floor_quantile,
                                          roi_box_mm=box, roi_weight=float(roi_weight)))
    except ValueError:
        # an ROI with no mass in the preview (outside the specimen): the box alone
        return int(seed_count_for_spacing(preview_volume(preview), geometry=geometry,
                                          spacing_mm=float(spacing_mm),
                                          noise_floor_quantile=noise_floor_quantile))


def n_views(scan: dict, phase: str) -> int:
    per = int(scan.get('views_per_phase') or 0)
    phases = [str(p) for p in (scan.get('phases') or [])]
    text = str(phase or '00').strip().lower()
    if text == 'all':
        return per * max(1, len(phases))
    wanted = [t.strip() for t in text.split(',') if t.strip()]
    return per * max(1, len([w for w in wanted if not phases or w in phases]))


def estimate(*, algorithm: str, args: argparse.Namespace, preview: dict,
             rate: Optional[float], rate_source: str, gpu: Optional[str] = None) -> dict:
    """Everything the detail row shows. ``preview`` is the picker's cached
    preview of the scan (with its ``scan`` block); ``rate`` the seconds per
    primitive-iteration, or None for the default of the machine's GPU class
    ``gpu`` (`default_rate`)."""
    from ..ct_core.preflight import MachineRequest
    from ..learning_based_iterative import registry

    scan = preview.get('scan') or {}
    spacing = float(getattr(args, 'gauss_seed_spacing', PSF_SIGMA_MM) or PSF_SIGMA_MM)
    ds = max(1, int(getattr(args, 'downsample', 1) or 1))
    vx = float(getattr(args, 'voxel_xy', 0.075) or 0.075)
    vz = float(getattr(args, 'voxel_z', 0.075) or 0.075)
    iterations = int(getattr(args, 'iterations', 20000) or 20000)
    seeds = getattr(args, 'gauss_seeds', 'auto')
    roi = getattr(args, 'roi', None)

    if str(seeds).lower() == 'auto':
        n = primitive_count(preview, spacing_mm=spacing, roi=roi,
                            roi_weight=float(getattr(args, 'gauss_seed_roi_weight', 5.0) or 1.0))
    else:
        n = int(seeds)
    cap = max(int(getattr(args, 'gauss_max', 0) or 0), n)

    # the shapes the preflight sizes from
    rows, cols = (int(v) for v in (scan.get('detector') or (2296, 3500)))
    n_ang = n_views(scan, getattr(args, 'phase', '00')) or int(preview.get('n_views', 220))
    box = roi_box(roi)
    if box is not None:
        ext = [box[1][i] - box[0][i] for i in range(3)]
    else:
        b = preview['bounds']
        ext = [b['x_max'] - b['x_min'], b['y_max'] - b['y_min'], b['z_max'] - b['z_min']]
    vol_shape = (max(1, int(round(ext[0] / vx))), max(1, int(round(ext[1] / vx))),
                 max(1, int(round(ext[2] / vz))))
    req = MachineRequest(n_angles=n_ang, n_b=rows // ds, n_a=cols // ds, vol_shape=vol_shape)
    sized = argparse.Namespace(**vars(args))
    sized.gauss_max = cap
    fp = registry.get(algorithm).footprint(sized, req)

    # the pixel the run keeps, at the isocentre
    R_s = float(scan.get('R_s') or 0.0)
    R_d = float(scan.get('R_d') or 0.0)
    pitch = float(scan.get('pitch_mm') or 0.0)
    pooled = pitch * ds * (R_s / (R_s + R_d) if R_s and R_d else 1.0)

    if rate:
        k = float(rate)
    else:
        k, rate_source = default_rate(algorithm, gpu)
    seconds = OVERHEAD_S + k * n * iterations

    warnings = []
    if pooled and spacing < pooled:
        warnings.append(f'the spacing ({spacing:g} mm) is finer than the detector pixel the run keeps '
                        f'({pooled:.3f} mm at the isocentre at --downsample {ds}): the data cannot resolve it')
    if vx > spacing or vz > spacing:
        warnings.append(f'the export voxel ({max(vx, vz):g} mm) is coarser than the spacing ({spacing:g} mm): '
                        f'the finer primitives are averaged away in the output')
    if spacing < 0.5 * PSF_SIGMA_MM:
        warnings.append(f'the spacing is under half the scanner PSF ({PSF_SIGMA_MM} mm): '
                        f'beyond the PSF every primitive only fits noise')
    return {
        'algorithm': algorithm,
        'spacing_mm': spacing, 'downsample': ds, 'voxel_xy': vx, 'voxel_z': vz,
        'psf_mm': PSF_SIGMA_MM, 'pooled_pixel_mm': pooled or None,
        'primitives': n, 'cap': cap, 'seeds': 'auto' if str(seeds).lower() == 'auto' else int(seeds),
        'n_views': n_ang, 'detector': [rows // ds, cols // ds], 'vol_shape': list(vol_shape),
        'gpu_bytes': int(fp.gpu_bytes()), 'host_bytes': int(fp.host_bytes_safe),
        'iterations': iterations, 'seconds': float(seconds), 'seconds_per_it': float(k * n),
        'rate_source': rate_source, 'gpu': gpu,
        'walltime': walltime(seconds),
        'warnings': warnings,
        'preview_cached': bool(preview.get('cached', True)),
    }


def runtime_estimate(*, algorithm: str, args: argparse.Namespace,
                     seconds_per_it: Optional[float], source: str, gpu: Optional[str] = None) -> dict:
    """The time half alone, for a representation without a primitive count:
    ``seconds_per_it`` from `measured_seconds_per_iteration`, or None when
    nothing on the machine has run this algorithm yet — then there is no
    estimate, only the words for why."""
    iterations = int(getattr(args, 'iterations', 20000) or 20000)
    if seconds_per_it is None:
        return {'algorithm': algorithm, 'iterations': iterations, 'primitives': None,
                'seconds': None, 'seconds_per_it': None, 'gpu': gpu, 'walltime': None,
                'rate_source': f'no run of {algorithm} on this machine to measure from yet', 'warnings': []}
    seconds = OVERHEAD_S + float(seconds_per_it) * iterations
    return {'algorithm': algorithm, 'iterations': iterations, 'primitives': None,
            'seconds': float(seconds), 'seconds_per_it': float(seconds_per_it), 'gpu': gpu,
            'walltime': walltime(seconds), 'rate_source': source, 'warnings': []}
