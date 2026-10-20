"""Directory listings for the page's file browser, on the local machine or
on a remote one through its connection (same pattern as ``inputs``)."""
from __future__ import annotations

import json
import os
import shlex
import subprocess

from . import proc
import sys

# stdlib only: runs under whatever python the machine has
SNIPPET = r'''
import json, os, sys, stat
spec = json.loads(sys.argv[1])
cwd = spec.get("cwd") or os.getcwd()
raw = spec.get("path") or cwd
p = os.path.expanduser(raw)
if not os.path.isabs(p):
    p = os.path.join(cwd, p)
p = os.path.normpath(p)
# climb to the nearest existing directory (a typed path may be a file or not exist yet)
while p and not os.path.isdir(p):
    q = os.path.dirname(p)
    if q == p:
        break
    p = q
entries = []
try:
    with os.scandir(p) as it:
        for e in it:
            if e.name.startswith(".") and not spec.get("hidden"):
                continue
            try:
                st = e.stat(follow_symlinks=True)
                d = stat.S_ISDIR(st.st_mode)
                entries.append({"name": e.name, "dir": d, "size": None if d else st.st_size, "mtime": st.st_mtime})
            except OSError:
                entries.append({"name": e.name, "dir": False, "size": None, "mtime": None, "broken": True})
    error = None
except OSError as e:
    error = str(e)
entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
print(json.dumps({"path": p, "parent": os.path.dirname(p) if os.path.dirname(p) != p else None,
                  "entries": entries, "error": error,
                  "places": [x for x in [("working dir", cwd), ("home", os.path.expanduser("~")),
                                         ("scans", spec.get("scans_root"))] if x[1]]
                            + ([(c + ":", c + ":\\") for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                if os.path.exists(c + ":\\")] if sys.platform == "win32" else [])}))
'''


def list_dir(path: str | None, *, cwd: str | None, scans_root: str | None = None,
             conn=None, python: str | None = None, hidden: bool = False) -> dict:
    spec = json.dumps({'path': path, 'cwd': cwd, 'scans_root': scans_root, 'hidden': hidden})
    if conn is None:
        r = proc.run([python or sys.executable, '-c', SNIPPET, spec], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:500])
        return json.loads(r.stdout.strip().splitlines()[-1])
    out = conn.check(f"{shlex.quote(python or 'python3')} -c {shlex.quote(SNIPPET)} {shlex.quote(spec)}", timeout=40)
    return json.loads(out.strip().splitlines()[-1])
