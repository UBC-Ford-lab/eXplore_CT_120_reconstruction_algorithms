"""The 4-D export of a dynamic Gaussian run: the breathing mouse as one
file — NRRD by default, the 3D Slicer NRRD layout or NIfTI-1 on request
(``--dyn-export-4d-format``) — with the gate axis swept from gate 0 (-1) to
gate 1 (+1) in equal steps. Convert a written file between the layouts
without re-rendering: ``ct120-export-4d
recon_4d.nrrd [--format nrrd|nrrd-slicer]``.

THE GRID. By default the states are on the DELIVERED grid — the same
lattice, pitch and origin as ``recon.vff`` (``--dyn-export-4d-mm auto``), so
state 0 and the volume are the same voxels and a viewer can flip between
them. On Scan_1510 that is 88 M voxels per state, 176 MB of int16, so the
states are rendered and written ONE AT A TIME (never all in memory) and the
file is gzip-compressed. A coarser isotropic pitch can be asked for
(``--dyn-export-4d-mm 0.15``) when a small file matters more than voxel
identity with the volume; the motion lives on a 3 mm control grid, so
nothing of it is lost that way.

WHAT A STATE IS. The gate axis at value g with the free code at zero and the
gate amplitude at zero: the same sweep the breathing GIF shows, on a full
grid instead of one slab. State 0 (g = -1) is the mean state of gate 0 up to
the group's mean amplitude, i.e. what ``--dyn-export-group 0`` delivers.
The frames are HU on the SAME two-anchor calibration the delivered volume
got (the driver hands the anchors over through ``after_save``).

ORIENTATION. The voxeliser lays a volume out as (nx, ny, nz); the VFF
export writes it transposed with y flipped. Every 4-D file keeps (x, y, z)
with the same y flip, so voxel (i, j, k) of a state is voxel (i, j, k) of
``recon.vff``. The NIfTI sform is the scanner frame in RAS (the NIfTI
convention) and records the flip as a negative y step; the Slicer NRRD
writes the SAME placement in LPS (the NRRD convention), which makes the x
step negative.

THREE FILES.
  ``nrrd`` (default) — the layout Imalytics Preclinical opens: state axis
    LAST (``sizes nx ny nz T``, ``kinds domain domain domain list``) and
    every step POSITIVE. Tested 2026-09-15 on a crop of Scan_1510: with the
    state axis first Imalytics reads its ``none`` direction as x and stops
    at "voxelSize.x should be > 0"; with a negative step its reader fails
    (NRRDIO.h:176). A positive-only header carries no axis flips, so a
    viewer draws it by voxel index — exactly how it draws ``recon.vff``
    (origin 0, no directions) — and the voxels keep the VFF's order: state
    0 overlays the volume (checked: 0.2 HU, rounding). The price: the LPS
    placement the header states is the true one mirrored in x within the
    same extent, so an orientation-aware viewer (Slicer, ITK-SNAP) shows it
    mirrored; use ``nrrd-slicer`` or ``nifti`` there. The header has only
    the fields that test opened (the state values and the HU calibration
    are in the sidecar), and the states stream out one at a time.
  ``nrrd-slicer`` — state axis FIRST (``kinds list domain domain domain``,
    state fastest) in the true LPS placement, with the key/value pairs 3D
    Slicer's Sequences module reads; it interleaves the states per voxel,
    so the int16 frames are held together before writing (8 states of
    Scan_1510's ROI = 1.4 GB of host memory). Imalytics cannot open it.
  ``nifti`` — NIfTI-1, state axis last, RAS sform, streamed.

Both writers are by hand (a 348-byte header / a text header, int16, gzip):
the formats are small and fixed, and it keeps nibabel and pynrrd out of the
run's dependencies. Read them back with 3D Slicer, ITK-SNAP, Fiji, nibabel,
pynrrd.
"""
from __future__ import annotations

import gzip
import json
import os
import struct

import numpy as np
import torch

from ..gaussian import kernel

from .show import gate_code

NIFTI_HDR = 348
VOX_OFFSET = 352

#: file format -> the suffix appended to the volume's stem (``recon`` ->
#: ``recon_4d.nrrd``); the sidecar is always ``<stem>_4d.json``.
FORMATS = {'nrrd': '_4d.nrrd', 'nrrd-slicer': '_4d.nrrd', 'nifti': '_4d.nii.gz'}
DEFAULT_FORMAT = 'nrrd'
_ALIASES = {'nii': 'nifti', 'slicer': 'nrrd-slicer', 'nrrd_slicer': 'nrrd-slicer'}


def check_format(fmt) -> str:
    """The canonical format name, or ValueError."""
    f = str(fmt).strip().lower()
    f = _ALIASES.get(f, f)
    if f not in FORMATS:
        raise ValueError(f"4-D export format must be one of {sorted(FORMATS)}, got {fmt!r}")
    return f


def sidecar_path(path: str) -> str:
    """``<stem>_4d.json`` for a 4-D file of either format."""
    for suffix in ('.nii.gz', '.nii', '.nrrd'):
        if path.lower().endswith(suffix):
            return path[:-len(suffix)] + '.json'
    return os.path.splitext(path)[0] + '.json'


# --------------------------------------------------------------- the grid --
def coarse_geometry(geom: dict, pitch_mm) -> dict:
    """The ROI grid ``geom`` (vol_shape, vol_origin, dx, dz) re-gridded at an
    isotropic ``pitch_mm``, same centre, same physical extent (to within a
    voxel). ``'auto'`` / None: the delivered grid itself, untouched."""
    if pitch_mm is None or (isinstance(pitch_mm, str) and pitch_mm.lower() == 'auto'):
        return {'vol_shape': tuple(int(v) for v in geom['vol_shape']),
                'vol_origin': tuple(float(v) for v in geom['vol_origin']),
                'dx': float(geom['dx']), 'dz': float(geom['dz'])}
    p = float(pitch_mm)
    if p <= 0:
        raise ValueError("the 4-D pitch must be > 0 mm")
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    dx, dz = float(geom['dx']), float(geom['dz'])
    shape = tuple(max(1, int(round(e / p))) for e in (nx * dx, ny * dx, nz * dz))
    return {'vol_shape': shape, 'vol_origin': tuple(float(v) for v in geom['vol_origin']),
            'dx': p, 'dz': p}


@torch.no_grad()
def render_grid(shown, geom: dict, world_scale: float, signal_scale: float,
                max_voxels: int | None = None) -> np.ndarray:
    """Voxelise ``shown`` (a cloud or a DisplacedCloud) on ``geom``, in mu
    mm^-1, (nx, ny, nz) — the reconstructor's own lattice arithmetic."""
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    dx, dz = float(geom['dx']), float(geom['dz'])
    ox, oy, oz = (float(v) for v in geom['vol_origin'])
    kw = {} if max_voxels is None else {'max_voxels': int(max_voxels)}
    vol = kernel.chunked_voxelize(
        shown, n_voxel=(nx, ny, nz),
        extent_world=(nx * dx * world_scale, ny * dx * world_scale, nz * dz * world_scale),
        center_world=(ox * world_scale, oy * world_scale, oz * world_scale), **kw)
    return (vol.detach().cpu().numpy() / signal_scale).astype(np.float32)


def gate_values(n_states: int) -> np.ndarray:
    """The gate axis from -1 to +1 in ``n_states`` equal steps (one state
    means the rest state)."""
    n = int(n_states)
    if n < 1:
        raise ValueError("need at least one state")
    return np.zeros(1) if n == 1 else np.linspace(-1.0, 1.0, n)


@torch.no_grad()
def render_states(cloud, motion, geom: dict, world_scale: float, signal_scale: float,
                  values, max_voxels: int | None = None) -> np.ndarray:
    """(T, nx, ny, nz) mu volumes, one per gate value."""
    frames = []
    for v in values:
        shown = motion.displaced_with_code(cloud, gate_code(motion, float(v)))
        frames.append(render_grid(shown, geom, world_scale, signal_scale, max_voxels))
    return np.stack(frames)


# ------------------------------------------------------------- the writer --
def nifti_affine(geom: dict) -> np.ndarray:
    """Voxel (i, j, k) of the y-flipped (nx, ny, nz) array -> mm, as the
    scanner frame the sidecars use (voxel centres, origin = grid centre)."""
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    dx, dz = float(geom['dx']), float(geom['dz'])
    ox, oy, oz = (float(v) for v in geom['vol_origin'])
    A = np.eye(4)
    A[0, 0], A[0, 3] = dx, ox - (nx - 1) / 2.0 * dx
    A[1, 1], A[1, 3] = -dx, oy + (ny - 1) / 2.0 * dx      # flipped y: row 0 is +y
    A[2, 2], A[2, 3] = dz, oz - (nz - 1) / 2.0 * dz
    return A


def _frame_bytes(frame_hu: np.ndarray, shape) -> bytes:
    """One (nx, ny, nz) HU frame -> int16, y flipped, x fastest on disk."""
    frame_hu = np.asarray(frame_hu)
    if frame_hu.shape != tuple(shape):
        raise ValueError(f"frame {frame_hu.shape} does not match the grid {tuple(shape)}")
    data = np.rint(np.clip(frame_hu[:, ::-1, :], -32768, 32767)).astype('<i2')
    return np.ascontiguousarray(data.transpose(2, 1, 0)).tobytes()


def write_nifti_4d(path: str, frames_hu, geom: dict, *, n_states: int | None = None,
                   descrip: str = '', intent_name: str = 'gate -1..+1') -> str:
    """Write HU frames in voxeliser layout as an int16 NIfTI-1 (x, y, z, t),
    y flipped like the VFF. ``frames_hu`` is a (T, nx, ny, nz) array, one
    (nx, ny, nz) frame, or an ITERABLE of frames with ``n_states`` given —
    the streaming form, one state in memory at a time. ``.gz`` compresses."""
    shape = tuple(int(v) for v in geom['vol_shape'])
    nx, ny, nz = shape
    if isinstance(frames_hu, np.ndarray):
        frames = frames_hu[None] if frames_hu.ndim == 3 else frames_hu
        T = int(frames.shape[0])
        frames = iter(frames)
    else:
        if n_states is None:
            raise ValueError("an iterable of frames needs n_states")
        T, frames = int(n_states), iter(frames_hu)
    A = nifti_affine(geom)
    dx, dz = float(geom['dx']), float(geom['dz'])

    h = bytearray(NIFTI_HDR)
    struct.pack_into('<i', h, 0, NIFTI_HDR)
    struct.pack_into('<8h', h, 40, 4, nx, ny, nz, T, 1, 1, 1)
    struct.pack_into('<h', h, 70, 4)                             # DT_SIGNED_SHORT
    struct.pack_into('<h', h, 72, 16)                            # bitpix
    struct.pack_into('<8f', h, 76, 1.0, dx, dx, dz, 1.0, 0, 0, 0)
    struct.pack_into('<f', h, 108, float(VOX_OFFSET))
    struct.pack_into('<ff', h, 112, 1.0, 0.0)                    # scl_slope / inter
    struct.pack_into('<B', h, 123, 2)                            # NIFTI_UNITS_MM
    struct.pack_into('<ff', h, 124, 0.0, 0.0)                    # cal_max/min: unset
    struct.pack_into('<80s', h, 148, descrip.encode()[:79])
    struct.pack_into('<hh', h, 252, 0, 1)                        # qform none, sform scanner
    struct.pack_into('<4f', h, 280, *A[0])
    struct.pack_into('<4f', h, 296, *A[1])
    struct.pack_into('<4f', h, 312, *A[2])
    struct.pack_into('<16s', h, 328, intent_name.encode()[:15])
    struct.pack_into('<4s', h, 344, b'n+1\0')
    opener = ((lambda p: gzip.open(p, 'wb', compresslevel=4))
              if path.endswith('.gz') else (lambda p: open(p, 'wb')))
    written = 0
    with opener(path) as f:
        f.write(bytes(h) + b'\0\0\0\0')
        for frame in frames:
            if written == T:
                raise ValueError(f"more than {T} frames for a {T}-state file")
            f.write(_frame_bytes(frame, shape))
            written += 1
    if written != T:
        raise ValueError(f"{written} frames written for a {T}-state file")
    return path


def read_nifti_header(path: str) -> dict:
    """dims, pixdim, datatype, affine of a NIfTI-1 file (for checks)."""
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        h = f.read(NIFTI_HDR)
    dim = struct.unpack_from('<8h', h, 40)
    return {'dim': dim[1:1 + dim[0]],
            'datatype': struct.unpack_from('<h', h, 70)[0],
            'pixdim': struct.unpack_from('<8f', h, 76)[1:1 + dim[0]],
            'vox_offset': struct.unpack_from('<f', h, 108)[0],
            'sform_code': struct.unpack_from('<h', h, 254)[0],
            'affine': np.array([struct.unpack_from('<4f', h, o) for o in (280, 296, 312)]
                               + [[0, 0, 0, 1]]),
            'magic': struct.unpack_from('<4s', h, 344)[0]}


def read_nifti_data(path: str) -> np.ndarray:
    """The int16 payload back as (T, nx, ny, nz) in voxeliser layout (y
    un-flipped): the inverse of `write_nifti_4d` (for checks)."""
    hdr = read_nifti_header(path)
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rb') as f:
        f.seek(int(hdr['vox_offset']))
        buf = f.read()
    dim = list(hdr['dim']) + [1] * (4 - len(hdr['dim']))
    nx, ny, nz, T = dim[:4]
    arr = np.frombuffer(buf, dtype='<i2').reshape(T, nz, ny, nx).transpose(0, 3, 2, 1)
    return arr[:, :, ::-1, :]



# ---------------------------------------------------------- the NRRD writer --
def nrrd_space(geom: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(directions, origin)`` of the y-flipped (nx, ny, nz) grid in NRRD's
    left-posterior-superior frame: the same placement `nifti_affine` records
    in RAS, so a viewer that reads both files shows one animal. Rows of
    ``directions`` are the per-axis mm steps. The ``nrrd-slicer`` header."""
    A = nifti_affine(geom)
    flip = np.diag([-1.0, -1.0, 1.0])                   # RAS -> LPS
    return (flip @ A[:3, :3]).T.copy(), flip @ A[:3, 3]


def nrrd_space_positive(geom: dict) -> tuple[np.ndarray, np.ndarray]:
    """The ``nrrd`` (Imalytics) header's ``(directions, origin)``: every step
    positive, over the same mm extent as `nrrd_space`. Only x changes sign
    (the y flip is already a positive LPS step), so voxel 0 moves to the
    other end of the x extent: the stated placement is mirrored in x while
    the voxel order stays the VFF's (see the module note)."""
    D, o = nrrd_space(geom)
    nx = int(geom['vol_shape'][0])
    o = o.copy()
    o[0] += D[0, 0] * (nx - 1)
    return np.abs(D), o


def _nrrd_vec(v) -> str:
    return '(' + ','.join(f'{float(x):.9g}' for x in v) + ')'


def nrrd_header(geom: dict, n_states: int, *, values=None, extra_fields=None,
                fmt: str = DEFAULT_FORMAT) -> bytes:
    """The text header of a 4-D NRRD for the grid ``geom``: int16, gzip.
    ``fmt='nrrd'``: state axis last, positive steps and nothing but the
    fields Imalytics was seen to open (``values``/``extra_fields`` are not
    written: the sidecar has them). ``'nrrd-slicer'``: state axis first in
    LPS, plus the key/value pairs 3D Slicer's Sequences module reads."""
    fmt = check_format(fmt)
    nx, ny, nz = (int(v) for v in geom['vol_shape'])
    T = int(n_states)
    values = list(range(T)) if values is None else [float(v) for v in values]
    if len(values) != T:
        raise ValueError(f"{len(values)} state values for {T} states")
    if fmt == 'nrrd':
        D, o = nrrd_space_positive(geom)
        lines = ['NRRD0004',
                 'type: short',
                 'dimension: 4',
                 'space: left-posterior-superior',
                 f'sizes: {nx} {ny} {nz} {T}',
                 'space directions: ' + ' '.join(_nrrd_vec(d) for d in D) + ' none',
                 'kinds: domain domain domain list',
                 'endian: little',
                 'encoding: gzip',
                 f'space origin: {_nrrd_vec(o)}',
                 'space units: "mm" "mm" "mm"']
        return ('\n'.join(lines) + '\n\n').encode()
    if fmt != 'nrrd-slicer':
        raise ValueError(f"not a NRRD format: {fmt!r}")
    D, o = nrrd_space(geom)
    lines = ['NRRD0004',
             '# dynamic Gaussian breathing states: HU, one volume per gate '
             'value, state axis first',
             'type: short',
             'dimension: 4',
             'space: left-posterior-superior',
             f'sizes: {T} {nx} {ny} {nz}',
             'space directions: none ' + ' '.join(_nrrd_vec(d) for d in D),
             'kinds: list domain domain domain',
             'labels: "gate" "x" "y" "z"',
             'endian: little',
             'encoding: gzip',
             f'space origin: {_nrrd_vec(o)}',
             'space units: "mm" "mm" "mm"',
             'axis 0 index type:=numeric',
             'axis 0 index values:=' + ' '.join(f'{v:.9g}' for v in values),
             'DataNodeClassName:=vtkMRMLScalarVolumeNode',
             'units:=HU']
    for k, v in (extra_fields or {}).items():
        lines.append(f'{k}:={v}')
    return ('\n'.join(lines) + '\n\n').encode()


def write_nrrd_4d(path: str, frames_hu, geom: dict, *, n_states: int | None = None,
                  values=None, extra_fields=None, fmt: str = DEFAULT_FORMAT) -> str:
    """Write HU frames in voxeliser layout as an int16 gzip NRRD, y flipped
    like the VFF. ``frames_hu`` as for `write_nifti_4d`. ``fmt='nrrd'``
    (default): state axis last, streamed one state at a time — the payload
    is the NIfTI's, byte for byte. ``'nrrd-slicer'``: state axis first
    (state fastest on disk), so every frame is buffered and the file is
    written one z-slab at a time."""
    fmt = check_format(fmt)
    shape = tuple(int(v) for v in geom['vol_shape'])
    nx, ny, nz = shape
    if isinstance(frames_hu, np.ndarray):
        frames_hu = frames_hu[None] if frames_hu.ndim == 3 else frames_hu
        T = int(frames_hu.shape[0])
    else:
        if n_states is None:
            raise ValueError("an iterable of frames needs n_states")
        T = int(n_states)
    header = nrrd_header(geom, T, values=values, extra_fields=extra_fields, fmt=fmt)
    if fmt == 'nrrd':
        written = 0
        with open(path, 'wb') as raw:
            raw.write(header)
            # no name or mtime in the gzip member: two writes of the same
            # states are byte-identical
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=4,
                               mtime=0) as f:
                for frame in frames_hu:
                    if written == T:
                        raise ValueError(f"more than {T} frames for a {T}-state file")
                    f.write(_frame_bytes(frame, shape))
                    written += 1
        if written != T:
            raise ValueError(f"{written} frames written for a {T}-state file")
        return path
    frames = []
    for frame in frames_hu:
        if len(frames) == T:
            raise ValueError(f"more than {T} frames for a {T}-state file")
        frame = np.asarray(frame)
        if frame.shape != shape:
            raise ValueError(f"frame {frame.shape} does not match the grid {shape}")
        frames.append(np.rint(np.clip(frame[:, ::-1, :], -32768, 32767)).astype('<i2'))
    if len(frames) != T:
        raise ValueError(f"{len(frames)} frames written for a {T}-state file")
    with open(path, 'wb') as raw:
        raw.write(header)
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=4, mtime=0) as f:
            for k in range(nz):
                slab = np.stack([fr[:, :, k] for fr in frames], axis=-1)   # (nx, ny, T)
                f.write(np.ascontiguousarray(slab.transpose(1, 0, 2)).tobytes())
    return path


def read_nrrd_header(path: str) -> tuple[dict, int]:
    """``(fields, data_offset)``: the header's fields and key/value pairs as
    strings (sizes parsed to ints), for checks."""
    fields, offset = {}, 0
    with open(path, 'rb') as f:
        magic = f.readline()
        offset += len(magic)
        if not magic.startswith(b'NRRD'):
            raise ValueError(f"{path} is not a NRRD file")
        fields['magic'] = magic.decode().strip()
        while True:
            line = f.readline()
            offset += len(line)
            if not line or line in (b'\n', b'\r\n'):
                break
            text = line.decode().rstrip('\r\n')
            if text.startswith('#'):
                continue
            if ':=' in text:
                k, v = text.split(':=', 1)
            else:
                k, v = text.split(': ', 1)
            fields[k.strip()] = v.strip()
    fields['sizes'] = [int(v) for v in fields['sizes'].split()]
    return fields, offset


def nrrd_format(fields: dict) -> str:
    """Which of our NRRD layouts a header is: the state axis first
    (``nrrd-slicer``) or not (``nrrd``, also a plain 3-D file)."""
    return 'nrrd-slicer' if fields.get('kinds', '').split()[:1] == ['list'] else 'nrrd'


def read_nrrd_data(path: str) -> np.ndarray:
    """The int16 payload back as (T, nx, ny, nz) in voxeliser layout (y
    un-flipped), either layout: the inverse of `write_nrrd_4d` (for checks
    and `convert_nrrd`)."""
    fields, offset = read_nrrd_header(path)
    with open(path, 'rb') as f:
        f.seek(offset)
        buf = f.read()
    if fields.get('encoding', 'raw') == 'gzip':
        buf = gzip.decompress(buf)
    if nrrd_format(fields) == 'nrrd-slicer':
        T, nx, ny, nz = fields['sizes']
        arr = np.frombuffer(buf, dtype='<i2').reshape(nz, ny, nx, T).transpose(3, 2, 1, 0)
    else:
        nx, ny, nz, T = (list(fields['sizes']) + [1])[:4]
        arr = np.frombuffer(buf, dtype='<i2').reshape(T, nz, ny, nx).transpose(0, 3, 2, 1)
    return arr[:, :, ::-1, :]


WRITERS = {'nrrd': write_nrrd_4d, 'nrrd-slicer': write_nrrd_4d, 'nifti': write_nifti_4d}

class LinearHU:
    """``HU = scale * mu + offset`` — the delivered calibration read back from
    a run's sidecar, in the shape `export_4d` wants (`HUAnchors` has the
    same three members)."""

    def __init__(self, scale: float, offset: float):
        self.scale, self.offset = float(scale), float(offset)

    def apply(self, values, dtype=np.float32) -> np.ndarray:
        return (np.asarray(values) * self.scale + self.offset).astype(dtype)


# ------------------------------------------------------------ the product --
def export_4d(cloud, motion, roi_geom: dict, world_scale: float, signal_scale: float,
              anchors, base_path: str, *, n_states: int, pitch_mm='auto',
              fmt: str = DEFAULT_FORMAT, max_voxels: int | None = None,
              log=print) -> str | None:
    """Render the sweep and write ``<base>_4d.nrrd`` (``fmt='nrrd'``, the
    default, or ``'nrrd-slicer'``) or ``<base>_4d.nii.gz`` (``fmt='nifti'``)
    + the ``<base>_4d.json`` sidecar. Returns the file's path, or None when the model has no gate
    axis (a single-phase run has nothing to sweep)."""
    fmt = check_format(fmt)
    if motion is None or not motion.n_fixed:
        log("  4-D export skipped: no gate axis (single acquisition group)")
        return None
    geom = coarse_geometry(roi_geom, pitch_mm)
    values = gate_values(n_states)
    nx, ny, nz = geom['vol_shape']
    same = geom['vol_shape'] == tuple(int(v) for v in roi_geom['vol_shape'])
    log(f"  exporting {len(values)} breathing states of {nx}x{ny}x{nz} = "
        f"{nx * ny * nz / 1e6:.1f} M voxels at {geom['dx']:.4f} mm"
        + (" (the delivered grid)" if same else "") + ", gate axis -1..+1, one state at a time")

    def states():
        for i, v in enumerate(values):
            shown = motion.displaced_with_code(cloud, gate_code(motion, float(v)))
            mu = render_grid(shown, geom, world_scale, signal_scale, max_voxels)
            log(f"    state {i + 1}/{len(values)} (gate {v:+.2f}) rendered")
            yield anchors.apply(mu)

    base = base_path[:-4] if base_path.lower().endswith('.vff') else base_path
    path = base + FORMATS[fmt]
    if fmt == 'nifti':
        write_nifti_4d(path, states(), geom, n_states=len(values),
                       descrip='dynamic Gaussian breathing states, HU, y flipped like the VFF')
    else:
        write_nrrd_4d(path, states(), geom, n_states=len(values), values=values,
                      extra_fields={'hu calibration': f'{float(anchors.scale):.9g} {float(anchors.offset):.9g}',
                                    'state axis': 'gate code -1 (gate 0) .. +1 (gate 1)'},
                      fmt=fmt)
    side = {'file': os.path.basename(path), 'format': fmt, 'units': 'HU',
            'states': [float(v) for v in values],
            'state_axis': 'gate code -1 (gate 0) .. +1 (gate 1), free code 0, amplitude 0',
            'voxel_size_mm': {'xy': float(geom['dx']), 'z': float(geom['dz'])},
            'grid': 'the delivered volume\'s' if same else f'isotropic {float(pitch_mm):g} mm',
            'shape': [int(nx), int(ny), int(nz), len(values)],
            'vol_origin_mm': [float(v) for v in geom['vol_origin']],
            'hu_calibration': {'scale': float(anchors.scale), 'offset': float(anchors.offset)},
            'affine_ras': nifti_affine(geom).tolist()}
    side.update(_layout_fields(geom, fmt))
    with open(sidecar_path(path), 'w') as f:
        json.dump(side, f, indent=2)
    log(f"  4-D {'NIfTI' if fmt == 'nifti' else 'NRRD (' + fmt + ')'} saved to: {path} "
        f"({os.path.getsize(path) / 2**20:.0f} MiB)")
    return path


def _layout_fields(geom: dict, fmt: str) -> dict:
    """The sidecar entries that depend on the file's layout."""
    if fmt == 'nifti':
        return {'axis_order': 'x, y (flipped like the VFF), z, state'}
    if fmt == 'nrrd-slicer':
        D, o = nrrd_space(geom)
        return {'axis_order': 'state, x, y (flipped like the VFF), z',
                'nrrd_space_directions_lps': D.tolist(), 'nrrd_space_origin_lps': o.tolist()}
    D, o = nrrd_space_positive(geom)
    return {'axis_order': 'x, y (flipped like the VFF), z, state',
            'viewer': 'Imalytics: voxel order of recon.vff, all steps positive; the stated '
                      'LPS placement is mirrored in x (use nrrd-slicer or nifti in Slicer)',
            'nrrd_space_directions_lps': D.tolist(), 'nrrd_space_origin_lps': o.tolist()}


# ------------------------------------------------------------ conversion --
def convert_nrrd(src: str, dst: str | None = None, fmt: str = DEFAULT_FORMAT,
                 log=print) -> str:
    """Rewrite a 4-D NRRD this module wrote in the other NRRD layout, without
    re-rendering (int16 in, int16 out: lossless both ways). The grid comes
    from the ``<stem>_4d.json`` sidecar beside ``src``, which is updated
    (or copied beside ``dst``). ``dst`` None: in place, through a temporary
    file that is read back and compared before it replaces ``src``."""
    fmt = check_format(fmt)
    if fmt == 'nifti':
        raise ValueError("convert_nrrd writes NRRD layouts only (nrrd, nrrd-slicer)")
    with open(sidecar_path(src)) as f:
        side = json.load(f)
    nx, ny, nz, T = (int(v) for v in side['shape'])
    geom = {'vol_shape': (nx, ny, nz), 'vol_origin': tuple(side['vol_origin_mm']),
            'dx': float(side['voxel_size_mm']['xy']), 'dz': float(side['voxel_size_mm']['z'])}
    fields, _ = read_nrrd_header(src)
    have = nrrd_format(fields)
    data = read_nrrd_data(src)
    if data.shape != (T, nx, ny, nz):
        raise ValueError(f"{src} holds {data.shape}, its sidecar says {(T, nx, ny, nz)}")
    target = src if dst is None else dst
    if have == fmt and dst is None:
        log(f"  {src} is already {fmt}: nothing to do")
        return src
    tmp = target + '.converting'
    hu = side.get('hu_calibration', {})
    write_nrrd_4d(tmp, data, geom, values=side.get('states'), fmt=fmt,
                  extra_fields={'hu calibration': f"{float(hu.get('scale', 0)):.9g} "
                                                  f"{float(hu.get('offset', 0)):.9g}",
                                'state axis': 'gate code -1 (gate 0) .. +1 (gate 1)'})
    if not np.array_equal(read_nrrd_data(tmp), data):
        os.remove(tmp)
        raise RuntimeError(f"read-back of {tmp} differs from {src}; {src} left untouched")
    os.replace(tmp, target)
    side = {k: v for k, v in side.items()
            if k not in ('axis_order', 'viewer', 'nrrd_space_directions_lps',
                         'nrrd_space_origin_lps')}
    side.update({'file': os.path.basename(target), 'format': fmt})
    side.update(_layout_fields(geom, fmt))
    with open(sidecar_path(target), 'w') as f:
        json.dump(side, f, indent=2)
    log(f"  {src} ({have}) -> {target} ({fmt}), {T} states of {nx}x{ny}x{nz}, voxels identical")
    return target


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description='Rewrite a dynamic Gaussian 4-D NRRD in another layout, '
                    'no re-rendering (the <stem>_4d.json sidecar must sit beside it).')
    ap.add_argument('src', help='the 4-D .nrrd a run wrote')
    ap.add_argument('dst', nargs='?', default=None,
                    help='output path (default: rewrite src in place, verified first)')
    ap.add_argument('--format', default=DEFAULT_FORMAT, choices=('nrrd', 'nrrd-slicer'),
                    help='nrrd (default): opens in Imalytics, voxel order of recon.vff; '
                         'nrrd-slicer: state axis first for 3D Slicer sequences')
    args = ap.parse_args(argv)
    convert_nrrd(args.src, args.dst, args.format)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
