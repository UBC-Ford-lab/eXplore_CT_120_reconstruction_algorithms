"""The files a run wrote, for the delete-with-data action.

A driver's output path is a base name; the volume is ``<base>.vff`` with its
``.json`` sidecar and ``<base>_plots/``, and some runs add a cloud checkpoint,
a 4-D file and ventilation maps next to it. Only names that follow those
rules are listed, never ``<base>*``, so a second run named ``<base>_...`` by
another job is left alone. The same snippet runs locally and on a remote
machine so both answer identically.
"""
from __future__ import annotations

import json
import subprocess

from . import proc
import sys

SNIPPET = r'''
import json, os, sys, shutil
base = sys.argv[1]; action = sys.argv[2]
d, name = os.path.split(os.path.abspath(os.path.expanduser(base)))
exts = ('.vff', '.json', '.npz', '.png', '.pt', '.nrrd', '.nii', '.nii.gz', '.gif', '.txt', '.log')
def matches(entry):
    if entry == name: return True
    for e in exts:
        if entry == name + e: return True
    for suf in ('_plots', '_cloud.pt', '_4d', '_ventilation', '_report', '_projection_report.json', '_motion', '_evolution'):
        if entry == name + suf or entry.startswith(name + suf + '.') or entry.startswith(name + suf + '_'): return True
    return False
def size(p):
    if os.path.isdir(p):
        t = 0
        for r, _, fs in os.walk(p):
            for f in fs:
                try: t += os.path.getsize(os.path.join(r, f))
                except OSError: pass
        return t
    try: return os.path.getsize(p)
    except OSError: return 0
out = []
if os.path.isdir(d):
    for entry in sorted(os.listdir(d)):
        if matches(entry):
            p = os.path.join(d, entry)
            out.append({'path': p, 'dir': os.path.isdir(p), 'bytes': size(p)})
if action == 'delete':
    for item in out:
        try:
            if item['dir']: shutil.rmtree(item['path'])
            else: os.remove(item['path'])
            item['deleted'] = True
        except OSError as e:
            item['deleted'] = False; item['error'] = str(e)
print(json.dumps(out))
'''


def run_snippet(base: str, action: str, *, conn=None, python: str | None = None) -> list[dict]:
    """``action`` is ``list`` or ``delete``. With ``conn`` the snippet runs on
    the remote machine through its connection."""
    if conn is None:
        r = proc.run([python or sys.executable, '-c', SNIPPET, base, action],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:500])
        return json.loads(r.stdout.strip().splitlines()[-1])
    import shlex
    out = conn.check(f"{shlex.quote(python or 'python3')} -c {shlex.quote(SNIPPET)} {shlex.quote(base)} {action}", timeout=600)
    return json.loads(out.strip().splitlines()[-1])
