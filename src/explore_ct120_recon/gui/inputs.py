"""Input checks for the New job form: is everything the driver needs there?

The page asks as soon as an input is typed, so the answer has to be cheap:
only directory listings and file headers are read, never a projection or a
volume. The same stdlib-only snippet runs locally and on the remote machine
through its connection, so a scan on the cluster is checked where it lives.

Each driver has a plan (``plan_for``) naming which of its values are scan
folders, volumes, a template scan or the output path; the snippet turns the
plan into groups of items, one group per path, each item
``{'label', 'status', 'detail'}`` with status ``ok`` / ``warn`` / ``fail`` /
``info``. A ``fail`` is something the driver would stop on; ``warn`` is
something it would run with but the user may not want; ``info`` is a fact
with no verdict (a calibration that will be measured on first use).
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys

from .schema import Driver, predict_output

SNIPPET = r'''
import json, os, re, sys, glob
spec = json.loads(sys.argv[1])
cwd = spec.get('cwd')
if cwd and os.path.isdir(os.path.expanduser(cwd)):
    os.chdir(os.path.expanduser(cwd))

def full(p):
    return os.path.abspath(os.path.expanduser(p))

def human(n):
    for unit in ('B', 'kB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return (f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}')
        n /= 1024

def vff_header(path):
    """The key=value header of a VFF file, or None if it is not one."""
    try:
        with open(path, 'rb') as fh:
            head = fh.read(4096)
    except OSError:
        return None
    if not head.startswith(b'ncaa'):
        return None
    raw = head.split(b'\f', 1)[0]
    text = raw.decode('latin-1', errors='replace')
    h = {'_header_bytes': str(len(raw) + 1)}
    for line in text.splitlines()[1:]:
        if '=' in line:
            k, v = line.split('=', 1)
            h[k.strip()] = v.strip().rstrip(';')
    return h

def item(label, status, detail=''):
    return {'label': label, 'status': status, 'detail': detail}

def xml_field(text, name):
    m = re.search(r'<%s>([^<]*)</%s>' % (name, name), text)
    return m.group(1).strip() if m else None

def calibration_dir():
    try:
        from explore_ct120_recon.ct_core.paths import calibration_dir as cd
        return str(cd())
    except Exception:
        return None

def serial_of(path):
    h = vff_header(path) or {}
    return h.get('serialNumber')

def scan_group(path, title, scan_folder=None, want_calibration=True):
    items = []
    p = full(path)
    g = {'title': title, 'path': path, 'items': items}
    if not os.path.isdir(p):
        items.append(item('folder', 'fail', 'not found' if not os.path.exists(p) else 'not a folder'))
        return g
    items.append(item('folder', 'ok', p))
    xml = os.path.join(p, 'scan.xml')
    views = None
    if os.path.isfile(xml):
        try:
            text = open(xml, errors='replace').read()
        except OSError:
            text = ''
        vc, inc = xml_field(text, 'ViewCount'), xml_field(text, 'IncrementAngle')
        if vc and inc:
            views = int(float(vc))
            items.append(item('scan.xml', 'ok', f'{views} views, {float(inc):.4g} deg step'))
        else:
            items.append(item('scan.xml', 'fail', 'present but ViewCount / IncrementAngle missing'))
    else:
        items.append(item('scan.xml', 'fail', 'missing'))
    proj = sorted(glob.glob(os.path.join(p, 'proj-*.vff')))
    pattern = 'proj-*.vff'
    if not proj:
        proj = sorted(glob.glob(os.path.join(p, 'acq*')))
        pattern = 'acq*'
    if not proj:
        items.append(item('projections', 'fail', 'no proj-*.vff or acq-* files'))
    else:
        phases = sorted({m.group(1) for f in proj for m in [re.match(r'acq-(\d+)-', os.path.basename(f))] if m})
        n = len(proj)
        per = n // len(phases) if phases else n
        detail = f'{n} files ({pattern})' + (f', {len(phases)} phases of {per}' if len(phases) > 1 else '')
        if views is not None and per != views:
            items.append(item('projections', 'warn', detail + f'; scan.xml says {views}'))
        else:
            items.append(item('projections', 'ok', detail))
    # bright / dark: here, in --scan-folder, or in a sibling scans/ folder
    cal = None
    if os.path.isfile(os.path.join(p, 'bright.vff')) and os.path.isfile(os.path.join(p, 'dark.vff')):
        cal = p
        items.append(item('bright / dark fields', 'ok', 'next to the projections'))
    elif scan_folder:
        sf = full(scan_folder)
        if os.path.isfile(os.path.join(sf, 'bright.vff')) and os.path.isfile(os.path.join(sf, 'dark.vff')):
            cal = sf
            items.append(item('bright / dark fields', 'ok', f'from --scan-folder {sf}'))
        else:
            items.append(item('bright / dark fields', 'fail', f'not in --scan-folder {sf}'))
    else:
        m = re.search(r'(Scan_\d+)', os.path.basename(p.rstrip('/')))
        found = None
        if m:
            parent = os.path.dirname(p)
            while parent and parent != os.path.dirname(parent):
                cand = os.path.join(parent, 'scans', m.group(1))
                if os.path.isfile(os.path.join(cand, 'bright.vff')) and os.path.isfile(os.path.join(cand, 'dark.vff')):
                    found = cand
                    break
                parent = os.path.dirname(parent)
        if found:
            cal = found
            items.append(item('bright / dark fields', 'ok', f'from {found}'))
        else:
            items.append(item('bright / dark fields', 'fail',
                              'missing; a derived folder needs Scan_XXXX in its name with the scan under a sibling scans/, or --scan-folder'))
    serial = serial_of(proj[0]) if proj else None
    if proj:
        h = vff_header(proj[0])
        if h is None:
            items.append(item('projection header', 'fail', f'{os.path.basename(proj[0])} is not a VFF file'))
        else:
            items.append(item('projection header', 'ok',
                              f"{h.get('size', '?')} px, {h.get('bits', '?')} bit" + (f', detector {serial}' if serial else '')))
    if want_calibration and serial:
        cdir = calibration_dir()
        tag = os.path.basename(p.rstrip('/'))
        if cdir is None:
            items.append(item('calibration directory', 'info', 'package not importable here; cannot look'))
        else:
            for label, prefix in (('geometry calibration (psi)', 'detector_psi'), ('noise calibration', 'detector_noise')):
                scan_file = os.path.join(cdir, f'{prefix}_{serial}_{tag}.json')
                det_file = os.path.join(cdir, f'{prefix}_{serial}.json')
                if os.path.isfile(scan_file):
                    items.append(item(label, 'ok', f'cached for this scan ({os.path.basename(scan_file)})'))
                elif os.path.isfile(det_file):
                    items.append(item(label, 'ok', f'detector-wide record ({os.path.basename(det_file)})'))
                else:
                    items.append(item(label, 'info', 'none cached; measured on first use'))
    return g

def volume_group(path, title, sidecar='warn'):
    items = []
    p = full(path)
    g = {'title': title, 'path': path, 'items': items}
    if not os.path.isfile(p):
        items.append(item('file', 'fail', 'not found' if not os.path.exists(p) else 'not a file'))
        return g
    size = os.path.getsize(p)
    items.append(item('file', 'ok', f'{human(size)}'))
    ext = os.path.splitext(p)[1].lower()
    if ext == '.vff':
        h = vff_header(p)
        if h is None:
            items.append(item('VFF header', 'fail', 'not a VFF file'))
        else:
            dims = h.get('size', '')
            try:
                n = 1
                for d in dims.split():
                    n *= int(d)
                need = n * int(h.get('bits', 16)) // 8
                detail = f"{' x '.join(dims.split())} voxels, {h.get('bits', '?')} bit, {h.get('elementsize', '?')} mm"
                if size - int(h.get('_header_bytes', 0)) < need:
                    items.append(item('VFF header', 'fail', detail + f'; file is {human(size)} but the header needs {human(need)} (truncated?)'))
                else:
                    items.append(item('VFF header', 'ok', detail))
            except ValueError:
                items.append(item('VFF header', 'warn', f'unreadable size/bits ({dims!r})'))
    elif ext in ('.npy', '.npz'):
        items.append(item('format', 'ok', ext[1:] + ' array'))
    else:
        items.append(item('format', 'warn', f'{ext or "no extension"}: expected .vff (or .npy/.npz)'))
    if sidecar != 'none':
        side = os.path.splitext(p)[0] + '.json'
        if os.path.isfile(side):
            try:
                rec = json.load(open(side))
                bits = [str(rec.get('algorithm', ''))]
                if rec.get('scan'):
                    bits.append(f"scan {os.path.basename(str(rec['scan']).rstrip('/'))}")
                if rec.get('created'):
                    bits.append(str(rec['created'])[:10])
                items.append(item('sidecar .json', 'ok', ', '.join(b for b in bits if b)))
            except (OSError, ValueError) as e:
                items.append(item('sidecar .json', 'fail' if sidecar == 'fail' else 'warn', f'unreadable: {e}'))
        else:
            if sidecar == 'fail':
                items.append(item('sidecar .json', 'fail', 'missing; needed for the grid, geometry and HU map'))
            else:
                items.append(item('sidecar .json', 'warn', 'missing; geometry comes from the header, HU map unknown'))
    return g

def output_group(path, title, writes, exists='warn'):
    items = []
    p = full(path)
    g = {'title': title, 'path': path, 'items': items}
    parent = os.path.dirname(p)
    if not os.path.isdir(parent):
        items.append(item('folder', 'fail', f'{parent} does not exist'))
        return g
    if not os.access(parent, os.W_OK):
        items.append(item('folder', 'fail', f'{parent} is not writable'))
        return g
    items.append(item('folder', 'ok', f'{parent} is writable'))
    targets = [p] + [p + w for w in writes if w.startswith('.') or w.startswith('_')]
    present = [t for t in targets if os.path.exists(t.rstrip('/'))]
    if present:
        names = ', '.join(os.path.basename(t.rstrip('/')) for t in present)
        if exists == 'fail':
            items.append(item('already there', 'fail', f'{names}; the driver refuses to overwrite'))
        else:
            items.append(item('already there', 'warn', f'{names} will be overwritten'))
    else:
        items.append(item('name free', 'ok', os.path.basename(p)))
    return g

groups = []
for c in spec['checks']:
    kind = c['kind']
    if kind == 'scan':
        groups.append(scan_group(c['path'], c['title'], c.get('scan_folder'), c.get('calibration', True)))
    elif kind == 'volume':
        groups.append(volume_group(c['path'], c['title'], c.get('sidecar', 'warn')))
    elif kind == 'output':
        groups.append(output_group(c['path'], c['title'], c.get('writes', []), c.get('exists', 'warn')))
print(json.dumps(groups))
'''


def plan_for(driver: Driver, schema: dict, values: dict, algorithm: str | None) -> list[dict]:
    """Which paths in ``values`` to check, and how, for this driver."""
    v = values
    checks: list[dict] = []
    scan_folder = v.get('scan_folder') or None

    def scan(dest: str, title: str, **kw):
        if v.get(dest):
            checks.append({'kind': 'scan', 'path': str(v[dest]), 'title': title,
                           'scan_folder': scan_folder, **kw})

    def volumes(dest: str, title: str, sidecar: str):
        val = v.get(dest)
        if not val:
            return
        paths = val if isinstance(val, list) else [val]
        for i, pth in enumerate(paths):
            if pth:
                checks.append({'kind': 'volume', 'path': str(pth), 'sidecar': sidecar,
                               'title': title if len(paths) == 1 else f'{title} {i + 1}'})

    if driver.name in ('fdk', 'iterative', 'learned'):
        scan('data_folder', 'Scan')
    elif driver.name in ('geometry-calibration', 'noise-calibration'):
        scan('data_folder', 'Scan', calibration=True)
    elif driver.name == 'projection-report':
        scan('data_folder', 'Scan')
        volumes('volume', 'Volume', 'warn')
    elif driver.name == 'volume-report':
        volumes('volume', 'Volume', 'warn')
        if scan_folder:
            checks.append({'kind': 'scan', 'path': scan_folder, 'title': 'Scan folder',
                           'calibration': False})
    elif driver.name == 'bilateral-filter':
        volumes('infile', 'Volume', 'none')
    elif driver.name == 'simulate-scan':
        volumes('volume', 'Volume', 'fail')
        if v.get('template'):
            checks.append({'kind': 'scan', 'path': str(v['template']), 'title': 'Template scan',
                           'calibration': False})

    try:
        pred = predict_output(driver, schema, v, algorithm)
    except Exception:
        pred = None
    if pred and pred.get('path'):
        # What lands next to the base name is a property of the driver, not of
        # whether the path was typed or predicted (a typed path predicts no
        # suffixes), so the clash check must not depend on ``pred['writes']``.
        writes = {'reconstruction': ['.vff', '.json', '_plots'],
                  'volume-report': ['_plots'],
                  'bilateral-filter': ['.vff', '.json']}.get(
            driver.name if driver.kind != 'reconstruction' else 'reconstruction', [])
        exists = 'fail' if (driver.name == 'bilateral-filter' and not v.get('overwrite')) else 'warn'
        checks.append({'kind': 'output', 'path': pred['path'], 'title': 'Output',
                       'writes': writes, 'exists': exists})
    return checks


def run_checks(checks: list[dict], *, cwd: str | None, conn=None, python: str | None = None) -> list[dict]:
    """Run the snippet locally, or on the remote machine through ``conn``."""
    if not checks:
        return []
    spec = json.dumps({'checks': checks, 'cwd': cwd})
    if conn is None:
        r = subprocess.run([python or sys.executable, '-c', SNIPPET, spec],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:500])
        return json.loads(r.stdout.strip().splitlines()[-1])
    out = conn.check(f"{shlex.quote(python or 'python3')} -c {shlex.quote(SNIPPET)} {shlex.quote(spec)}",
                     timeout=60)
    return json.loads(out.strip().splitlines()[-1])
