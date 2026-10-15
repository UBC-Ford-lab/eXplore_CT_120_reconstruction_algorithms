"""The drivers as forms.

The drivers describe their flags once, in argparse. The GUI reads that
description instead of keeping its own, so a new flag or a changed default
appears in the form without touching this package, and the form cannot drift
from the command line.

``capture_parser`` obtains a driver's parser without running it: the driver's
``parse_args`` is called with ``ArgumentParser.parse_args`` temporarily
replaced by a function that hands the parser back. ``parser_to_schema`` turns
it into plain JSON. ``build_command`` turns submitted values back into the
command line the driver would have received.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import shlex
import sys
from dataclasses import dataclass, field
from typing import Any

MODULE_PREFIX = 'explore_ct120_recon.'


@dataclass(frozen=True)
class Driver:
    name: str
    module: str          # python module with ``parse_args``/``build_parser`` and ``main``
    script: str          # the console script, for display
    title: str
    kind: str            # 'reconstruction' | 'report' | 'calibration' | 'simulation'
    primary: str = 'data_folder'   # the positional that names the input
    has_algorithms: bool = False   # the parser depends on ``--algorithm``
    blurb: str = ''      # what the page shows instead of the parser's description

    def algorithms(self) -> list[dict]:
        if not self.has_algorithms:
            return []
        from ..learning_based_iterative import registry
        return [{'name': a.name, 'summary': a.summary} for a in registry.algorithms()]


DRIVERS: dict[str, Driver] = {d.name: d for d in (
    Driver('fdk', MODULE_PREFIX + 'run_fdk_recon', 'ct120-fdk', 'FDK', 'reconstruction',
           blurb='Standard analytic reconstruction and the baseline every other method '
                 'is compared against. Give it a scan folder; it writes the volume in '
                 'HU as <scan>_recon.vff with a .json sidecar and a _plots folder. '
                 'A few minutes on a GPU.'),
    Driver('iterative', MODULE_PREFIX + 'run_iterative_recon', 'ct120-iterative',
           'Classical iterative (ASTRA / TIGRE)', 'reconstruction',
           blurb='Classical iterative reconstruction (SIRT, CGLS, OS-SART and similar) '
                 'through the ASTRA or TIGRE toolbox, which must be installed on the '
                 'machine. Give it a scan folder; it writes <scan>_recon_<algorithm>.vff. '
                 'Slower than FDK, less noise at the same resolution.'),
    Driver('learned', MODULE_PREFIX + 'run_learned_recon', 'ct120-learned',
           'Learning-based', 'reconstruction', has_algorithms=True,
           blurb='Gradient-descent reconstruction with a differentiable projector. '
                 'Pick the model under Algorithm: a voxel grid, a neural field, '
                 'Gaussian splatting, or the time-aware Gaussian model for a breathing '
                 'animal. Give it a scan folder; it writes <scan>_recon_<algorithm>_<N>it.vff. '
                 'Tens of minutes to hours on a GPU, with progress logged to W&B.'),
    Driver('volume-report', MODULE_PREFIX + 'run_volume_report', 'ct120-volume-report',
           'Volume report', 'report', primary='volume',
           blurb='Scores a finished volume: noise, resolution and HU figures, logged '
                 'to W&B. Give it a .vff volume; the figures go into <volume>_report/. '
                 'Works on any reconstruction, including vendor volumes.'),
    Driver('projection-report', MODULE_PREFIX + 'run_projection_report',
           'ct120-projection-report', 'Projection report', 'report',
           blurb='Checks how well one or more volumes explain the projections of the '
                 'scan they came from. Give it the scan folder and the volumes to '
                 'compare; it writes a JSON record and figures next to the first '
                 'volume. Use it to rank reconstructions of the same scan.'),
    Driver('bilateral-filter', MODULE_PREFIX + 'run_bilateral_filter',
           'ct120-bilateral-filter', 'Bilateral filter', 'report', primary='infile',
           blurb="Edge-preserving noise filter for a finished volume, matching the "
                 "lab's MATLAB filter_vff. Give it a .vff volume and optionally the "
                 "noise standard deviation; it writes <volume>_imbilatfilt.vff. "
                 "Reconstructs nothing and does not log to W&B."),
    Driver('geometry-calibration', MODULE_PREFIX + 'run_geometry_calibration',
           'ct120-geometry-calibration', 'Geometry calibration', 'calibration',
           blurb="Measures the detector's in-plane rotation from the scan's own "
                 "projections and stores it in the calibration directory, where every "
                 "reconstruction of that scan picks it up. The reconstruction drivers "
                 "run this on demand, so submit it only to redo the measurement."),
    Driver('noise-calibration', MODULE_PREFIX + 'run_noise_calibration',
           'ct120-noise-calibration', 'Noise calibration', 'calibration',
           blurb="Measures the scan's noise model (photon noise plus read noise) from "
                 "its projections and stores it in the calibration directory. "
                 "Noise-weighted reconstructions on that detector pick it up "
                 "automatically. Run once per scan."),
    Driver('simulate-scan', MODULE_PREFIX + 'run_simulate_scan', 'ct120-simulate-scan',
           'Simulate a scan', 'simulation', primary='volume',
           blurb='Makes a synthetic scan from a known volume. Give it a volume this '
                 'package wrote and a real scan to copy the acquisition from; it writes '
                 'a new scan folder that every driver accepts. Reconstruct that, then '
                 'score the result against the volume it was made from.'),
)}


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def capture_parser(driver: Driver, algorithm: str | None = None) -> argparse.ArgumentParser:
    """The driver's parser, built exactly as the command line would build it."""
    module = importlib.import_module(driver.module)
    if hasattr(module, 'build_parser'):
        return module.build_parser()
    argv = ['--algorithm', algorithm] if (driver.has_algorithms and algorithm) else []

    def _grab(self, args=None, namespace=None):
        raise _Captured(self)

    original = argparse.ArgumentParser.parse_args
    saved_argv = sys.argv
    argparse.ArgumentParser.parse_args = _grab
    sys.argv = [driver.script] + argv
    try:
        if inspect.signature(module.parse_args).parameters:
            module.parse_args(argv)
        else:
            module.parse_args()
    except _Captured as c:
        return c.parser
    finally:
        argparse.ArgumentParser.parse_args = original
        sys.argv = saved_argv
    raise RuntimeError(f'{driver.module}.parse_args returned without parsing')


def _json_default(value):
    if value is argparse.SUPPRESS:
        return None
    if isinstance(value, (list, tuple)):
        return [_json_default(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _help(action) -> str:
    if not action.help or action.help is argparse.SUPPRESS:
        return ''
    params = dict(vars(action))
    params['default'] = action.default
    if action.choices is not None:
        params['choices'] = ', '.join(str(c) for c in action.choices)
    try:
        return (action.help % params).strip()
    except (KeyError, TypeError, ValueError):
        return action.help.strip()


def _type_name(action) -> str:
    t = action.type
    if t is int:
        return 'integer'
    if t is float:
        return 'number'
    if t is None and isinstance(action.default, bool) is False and isinstance(action.default, int):
        return 'integer'
    if t is None and isinstance(action.default, float):
        return 'number'
    return 'text'


def parser_to_schema(parser: argparse.ArgumentParser) -> dict:
    """A JSON description of the parser: positionals, then grouped options.

    Each field: ``dest``, ``kind`` (``flag`` | ``choice`` | ``integer`` |
    ``number`` | ``text`` | ``list``), ``default``, ``help``, ``flags`` and,
    for a ``flag``, the option that switches it on and the one that switches
    it off (``--parker-weighting`` / ``--no-parker`` share one dest and become
    one switch).
    """
    fields: dict[str, dict] = {}
    order: list[str] = []
    for group in parser._action_groups:
        for action in group._group_actions:
            if isinstance(action, argparse._HelpAction):
                continue
            if action.help is argparse.SUPPRESS:
                continue
            dest = action.dest
            is_flag = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
            if dest in fields:
                f = fields[dest]
                if is_flag:
                    key = 'on' if isinstance(action, argparse._StoreTrueAction) else 'off'
                    f[key] = action.option_strings[0]
                    f['help'] = f['help'] or _help(action)
                continue
            f = {
                'dest': dest,
                'group': group.title or '',
                'positional': not action.option_strings,
                'flags': list(action.option_strings),
                'help': _help(action),
                'required': bool(action.required),
                'default': _json_default(action.default),
                'metavar': action.metavar if isinstance(action.metavar, str) else None,
            }
            if is_flag:
                f['kind'] = 'flag'
                f['default'] = bool(action.default) if action.default is not argparse.SUPPRESS else False
                key = 'on' if isinstance(action, argparse._StoreTrueAction) else 'off'
                f[key] = action.option_strings[0]
            elif isinstance(action, argparse._AppendAction) or action.nargs in ('+', '*') \
                    or (isinstance(action.nargs, int) and action.nargs > 1):
                f['kind'] = 'list'
                f['item'] = 'choice' if action.choices is not None else _type_name(action)
                if action.choices is not None:
                    f['choices'] = [_json_default(c) for c in action.choices]
                if f['default'] is None:
                    f['default'] = []
            elif action.choices is not None:
                f['kind'] = 'choice'
                f['choices'] = [_json_default(c) for c in action.choices]
            else:
                f['kind'] = _type_name(action)
            fields[dest] = f
            order.append(dest)
    return {
        'description': (parser.description or '').strip(),
        'fields': [fields[d] for d in order],
    }


def driver_schema(driver: Driver, algorithm: str | None = None) -> dict:
    schema = parser_to_schema(capture_parser(driver, algorithm))
    if driver.blurb:
        schema['description'] = driver.blurb
    schema['driver'] = driver.name
    schema['script'] = driver.script
    schema['algorithm'] = algorithm if driver.has_algorithms else None
    return schema


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return [str(x) for x in a] == [str(x) for x in b]
    if a is None or b is None:
        return a is b
    return str(a) == str(b)


def build_command(driver: Driver, schema: dict, values: dict[str, Any],
                  algorithm: str | None = None, python: str | None = None) -> list[str]:
    """The command line for ``values``: positionals, then every option whose
    value differs from the parser's default, in schema order.

    Only the differences are emitted, so the record reads like a command a
    person would have typed, and a default that moves later in the package
    moves for the recorded command too (as it would for a typed one).
    """
    cmd = [python or sys.executable, '-m', driver.module]
    if driver.has_algorithms and algorithm:
        cmd += ['--algorithm', algorithm]
    for f in schema['fields']:
        dest = f['dest']
        if f['positional']:
            v = values.get(dest)
            if v is None or v == '':
                raise ValueError(f'{dest} is required')
            cmd += [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
            continue
        if dest not in values:
            continue
        v = values[dest]
        if _same(v, f['default']):
            continue
        if f['kind'] == 'flag':
            flag = f.get('on') if v else f.get('off')
            if flag:
                cmd.append(flag)
            continue
        if v is None or v == '' or v == []:
            continue
        flag = f['flags'][0]
        if f['kind'] == 'list':
            cmd.append(flag)
            cmd += [str(x) for x in v]
        else:
            cmd += [flag, str(v)]
    return cmd


def shell_string(cmd: list[str]) -> str:
    return shlex.join(cmd)


# -- where a run will write ------------------------------------------------------
# Mirrors each driver's own rule for a missing --output, so the form can show
# the path before the run starts. The driver remains the authority: the
# record's output_path is read from its log once it prints it.

def _strip_vff(p: str) -> str:
    return p[:-4] if p.lower().endswith('.vff') else p


def _resolved(schema: dict, values: dict) -> dict:
    v = {f['dest']: f['default'] for f in schema['fields']}
    v.update({k: x for k, x in values.items() if x is not None and x != ''})
    return v


def predict_output(driver: Driver, schema: dict, values: dict, algorithm: str | None = None) -> dict | None:
    """``{'dest', 'path', 'writes', 'explicit'}`` or None when the driver has no
    predictable output. ``explicit`` is true when the values already name one."""
    v = _resolved(schema, values)
    dest = {'bilateral-filter': 'out', 'projection-report': 'report'}.get(driver.name, 'output')
    if driver.name in ('geometry-calibration', 'noise-calibration'):
        what = 'detector rotation (psi)' if driver.name == 'geometry-calibration' else 'noise model (Poisson slope, read noise)'
        return {'dest': None, 'path': '', 'explicit': False,
                'writes': [f'the {what} record into the shared calibration directory, keyed by the detector named in scan.xml']}
    if driver.name == 'projection-report':
        vols = v.get('volume') or []
        if v.get('report'):
            return {'dest': dest, 'path': str(v['report']), 'writes': [], 'explicit': True}
        if not vols:
            return None
        first = str(vols[0] if isinstance(vols, (list, tuple)) else vols)
        import os
        return {'dest': dest, 'path': os.path.join(os.path.dirname(first), _strip_vff(os.path.basename(first)) + '_projection_report.json'),
                'writes': ['the JSON record of every row'], 'explicit': False}
    if driver.name == 'simulate-scan':
        return {'dest': dest, 'path': v.get('output') or '', 'writes': ['a scan folder'], 'explicit': bool(v.get('output'))}
    if v.get(dest):
        return {'dest': dest, 'path': str(v[dest]), 'writes': [], 'explicit': True}
    src = str(v.get(driver.primary) or '').rstrip('/')
    if not src:
        return None
    if driver.name == 'fdk':
        path = src + '_recon'
    elif driver.name == 'iterative':
        algo = str(v.get('algorithm', '')).lower().replace('3d_cuda', '').replace('_cuda', '')
        suffix = f'_{algo}'
        if str(v.get('algorithm', '')) != 'FDK_CUDA':
            suffix += f"_{v.get('iterations')}it"
        if v.get('pwls'):
            suffix += '_pwls'
        path = src + '_recon' + suffix
    elif driver.name == 'learned':
        path = f"{src}_recon_{algorithm}_{v.get('iterations')}it"
    elif driver.name == 'volume-report':
        return {'dest': dest, 'path': _strip_vff(src) + '_report', 'writes': ['a report folder'], 'explicit': False}
    elif driver.name == 'bilateral-filter':
        return {'dest': dest, 'path': _strip_vff(src) + '_imbilatfilt', 'writes': ['.vff', '.json'], 'explicit': False}
    else:
        return None
    return {'dest': dest, 'path': path, 'writes': ['.vff', '.json', '_plots/'], 'explicit': False}
