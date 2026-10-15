"""Slices of a ``.vff`` volume for the page's viewer, read where the volume
lives: in this process for the local machine, through the connection for a
remote one. Only the requested slice is read (memory-mapped), so a 2 GB
volume costs one disk read per view."""
from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

# stdlib + numpy; runs under the machine's own python (the package is there)
SNIPPET = r'''
import base64, json, os, sys
import numpy as np
spec = json.loads(sys.argv[1])
cwd = spec.get("cwd") or os.getcwd()
p = os.path.expanduser(spec["path"])
if not os.path.isabs(p):
    p = os.path.join(cwd, p)
def header_of(path):
    with open(path, "rb") as f:
        raw = f.read(2000)
    end = raw.find(b"\f")
    text = raw[:end if end >= 0 else len(raw)].decode("latin-1")
    h = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            h[k.strip()] = v.strip().rstrip(";")
    return h, (end + 1 if end >= 0 else len(raw))
h, hlen = header_of(p)
dims = [int(v) for v in h["size"].split()]
if len(dims) == 2:
    dims.append(1)
nx, ny, nz = dims
bits = int(h.get("bits", 16))
dtype = np.dtype(">i1") if bits == 8 else np.dtype(">i2")
nbytes = nx * ny * nz * bits // 8
offset = os.path.getsize(p) - nbytes
vol = np.memmap(p, dtype=dtype, mode="r", offset=offset, shape=(nz, ny, nx))
op = spec.get("op", "info")
if op == "info":
    side = os.path.splitext(p)[0] + ".json"
    spacing = None
    units = None
    if os.path.exists(side):
        try:
            sc = json.load(open(side))
            v = sc.get("voxel_size_mm") or {}
            if "xy" in v:
                spacing = [float(v["xy"]), float(v["xy"]), float(v.get("z", v["xy"]))]
            units = sc.get("units")
        except Exception:
            pass
    if spacing is None:
        e = h.get("elementsize")
        if e:
            parts = [float(x) for x in e.split()]
            spacing = parts if len(parts) == 3 else [parts[0]] * 3
    # a coarse sample of the middle slice for a sensible default window
    mid = np.asarray(vol[nz // 2, ::max(1, ny // 128), ::max(1, nx // 128)]).astype(np.int32)
    lo, hi = (int(np.percentile(mid, 1)), int(np.percentile(mid, 99))) if mid.size else (0, 0)
    print(json.dumps({"path": p, "size": [nx, ny, nz], "bits": bits, "spacing": spacing, "units": units,
                      "sidecar": os.path.exists(side), "bytes": nbytes, "mtime": os.path.getmtime(p),
                      "sample": [lo, hi]}))
else:
    axis, index = spec["axis"], int(spec["index"])
    if axis == "z":
        sl = vol[min(index, nz - 1)]
    elif axis == "y":
        sl = vol[:, min(index, ny - 1), :]
    else:
        sl = vol[:, :, min(index, nx - 1)]
    arr = np.ascontiguousarray(sl, dtype="<i2")
    print(json.dumps({"shape": list(arr.shape), "data": base64.b64encode(arr.tobytes()).decode()}))
'''


def _run(spec: dict, *, conn=None, python: Optional[str] = None, timeout: float = 60) -> dict:
    text = json.dumps(spec)
    if conn is None:
        r = subprocess.run([python or sys.executable, '-c', SNIPPET, text], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip().splitlines()[-1][:400] if r.stderr.strip() else f'exit {r.returncode}')
        return json.loads(r.stdout.strip().splitlines()[-1])
    out = conn.check(f"{shlex.quote(python or 'python3')} -c {shlex.quote(SNIPPET)} {shlex.quote(text)}", timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def volume_info(path: str, *, cwd: Optional[str], conn=None, python: Optional[str] = None) -> dict:
    return _run({'op': 'info', 'path': path, 'cwd': cwd}, conn=conn, python=python)


def volume_slice(path: str, axis: str, index: int, *, cwd: Optional[str], conn=None,
                 python: Optional[str] = None) -> tuple[tuple[int, int], bytes]:
    """(rows, cols) and the slice as little-endian int16 bytes.
    ``z`` gives (y, x), ``y`` gives (z, x), ``x`` gives (z, y): the on-disk index order."""
    if axis not in ('x', 'y', 'z'):
        raise ValueError(f'axis must be x, y or z, not {axis!r}')
    d = _run({'op': 'slice', 'path': path, 'cwd': cwd, 'axis': axis, 'index': int(index)}, conn=conn, python=python)
    return (d['shape'][0], d['shape'][1]), base64.b64decode(d['data'])


class SliceCache:
    """The last few slices, keyed by machine, path, axis and index."""

    def __init__(self, max_bytes: int = 256 * 2**20):
        self.max_bytes = max_bytes
        self._d: OrderedDict[tuple, tuple[tuple[int, int], bytes]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key: tuple, value: tuple[tuple[int, int], bytes]) -> None:
        with self._lock:
            if key in self._d:
                self._bytes -= len(self._d[key][1])
            self._d[key] = value
            self._bytes += len(value[1])
            while self._bytes > self.max_bytes and self._d:
                _, (_, old) = self._d.popitem(last=False)
                self._bytes -= len(old)

    def forget(self, machine: str, path: str) -> None:
        with self._lock:
            for k in [k for k in self._d if k[0] == machine and k[1] == path]:
                self._bytes -= len(self._d.pop(k)[1])
