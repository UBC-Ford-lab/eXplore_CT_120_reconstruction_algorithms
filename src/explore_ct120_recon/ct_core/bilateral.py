"""Bilateral filtering of a VFF volume, as the lab's MATLAB ``filter_vff.m``.

This reproduces, step for step, the MATLAB function

    function [status] = filter_vff(infile, sd)

that has been used to denoise finished micro-CT volumes: copy the header, read
the int16 slices, pick the range kernel from a noise estimate, run
``imbilatfilt`` on every axial slice with its defaults, write int16 back.

What each MATLAB step becomes here, and where the behaviour is pinned:

  * HEADER. The MATLAB loop reads lines with ``fgetl`` until it reads one of
    length <= 1 (normally the form-feed line), and writes each one as
    ``fprintf(fid2, strcat(line, '\\n'))``. ``strcat`` strips TRAILING
    whitespace, form feed included, so the form-feed line comes out as a bare
    newline; ``fprintf(fid2, '\\f\\n')`` then ends the header. The output header
    is therefore the input's lines, one blank line, then ``\\f\\n``.
    ``fprintf`` also treats each line as a FORMAT string, with no data: escape
    sequences are translated, ``%%`` prints ``%``, and output stops at any
    other ``%`` (a conversion with nothing to convert, or an invalid operator,
    which MathWorks documents as "prints all text up to the invalid operator
    ... and discards the rest"). Everything after it on that line is lost,
    INCLUDING the ``\\n`` strcat appended, so the next line's text follows
    directly. The vendor's volumes hit this: their last header line is
    ``cmdLine="... -i .../uwarp-00-%04d.vff ..."``, which MATLAB writes as
    ``cmdLine="... -i .../uwarp-00-`` with no newline, the form-feed line's
    newline then closing it. Octal and hex escapes are not emulated and are
    refused.
  * DATA. ``fread(fid, [A, B], 'int16')`` on a big-endian handle, C times,
    starting right after the last header line. ``im(:, :, i)`` is (x, y);
    the payload slice here is (y, x). The filter below is symmetric under
    that transpose (square kernel, isotropic spatial weight, per-axis
    replicate padding), so filtering the payload slice is the same filter.
  * NOISE. ``nargin == 1``: ``patch = std2(im(20:40, 160:180, 100))^2`` — the
    sample variance (N-1) of a fixed 21 x 21 patch of slice 100, i.e. payload
    ``[99, 159:180, 19:40]``. ``nargin == 2``: ``patch = sd^2``. Then
    ``smooth = 2 * patch``.
  * FILTER. ``imbilatfilt(im(:, :, j), smooth)``: degreeOfSmoothing is the
    VARIANCE of the range Gaussian (MathWorks documentation), spatialSigma
    defaults to 1 pixel, NeighborhoodSize to ``2*ceil(2*spatialSigma)+1`` = 5
    (square), Padding to ``'replicate'``. Computed in float64, as MATLAB does
    on the double array ``fread`` returns. ``imbilatfilt`` rejects a
    non-positive degreeOfSmoothing, so a patch of constant values (e.g. air
    clamped to one value) is an error here too.
  * WRITE. ``fwrite(fid2, fil_im, 'int16', 0, 'ieee-be')``: doubles to int16.
    The ``fwrite`` page does not state the conversion; this uses MATLAB's
    double-to-integer rule (round half away from zero, saturate at the int16
    limits), which is what ``int16()`` does.

The two ``figure``/``imagesc`` previews of slice 150 are not reproduced.
Results agree with MATLAB to floating-point summation order: a filtered value
lying within ~1e-12 of a .5 boundary could round the other way.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# filter_vff.m, nargin == 1: std2(im(20:40,160:180,100)), 1-based (x, y, z).
MATLAB_PATCH_X = (20, 40)
MATLAB_PATCH_Y = (160, 180)
MATLAB_PATCH_SLICE = 100

_INT16 = np.iinfo(np.int16)


class VFFFilterError(ValueError):
    """The input cannot be processed the way filter_vff.m would process it."""


def imbilatfilt(image, degree_of_smoothing, spatial_sigma=1.0,
                neighborhood_size=None):
    """MATLAB ``imbilatfilt`` on a 2-D image, with its default options.

    ``degree_of_smoothing`` is the range kernel's variance, so the range
    weight is ``exp(-(I_q - I_p)^2 / (2 * degree_of_smoothing))``; the spatial
    weight is ``exp(-|q - p|^2 / (2 * spatial_sigma^2))`` over a square
    ``neighborhood_size`` window (default ``2*ceil(2*spatial_sigma)+1``), and
    borders are replicate-padded. Returns float64.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"expected a 2-D image, got {img.ndim}-D")
    if not degree_of_smoothing > 0:
        raise VFFFilterError(f"degreeOfSmoothing must be positive, got "
                             f"{degree_of_smoothing!r}")
    if not spatial_sigma > 0:
        raise ValueError(f"spatial_sigma must be positive, got {spatial_sigma!r}")
    if neighborhood_size is None:
        neighborhood_size = 2 * math.ceil(2 * spatial_sigma) + 1
    n = int(neighborhood_size)
    if n < 1 or n % 2 == 0:
        raise ValueError(f"neighborhood_size must be odd and positive, got {n}")

    r = n // 2
    padded = np.pad(img, r, mode='edge')
    h, w = img.shape
    inv_range = 1.0 / (2.0 * float(degree_of_smoothing))
    inv_space = 1.0 / (2.0 * float(spatial_sigma) ** 2)

    num = np.zeros_like(img)
    den = np.zeros_like(img)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            nb = padded[r + dy:r + dy + h, r + dx:r + dx + w]
            wgt = np.exp(-(dy * dy + dx * dx) * inv_space
                         - (nb - img) ** 2 * inv_range)
            num += wgt * nb
            den += wgt
    return num / den


def matlab_int16(values):
    """Doubles to int16 as MATLAB converts them: round half away from zero,
    saturate at the int16 limits."""
    v = np.asarray(values, dtype=np.float64)
    rounded = np.sign(v) * np.floor(np.abs(v) + 0.5)
    return np.clip(rounded, _INT16.min, _INT16.max).astype(np.int16)


def _strip_trailing_whitespace(line: bytes) -> bytes:
    # strcat on char arrays removes trailing space, \t, \v, \n, \r and \f.
    return line.rstrip(b' \t\v\n\r\f')


_ESCAPES = {b'n': b'\n', b't': b'\t', b'r': b'\r', b'f': b'\f', b'b': b'\b',
            b'a': b'\a', b'v': b'\v', b'\\': b'\\'}


def matlab_fprintf_no_data(fmt: bytes) -> bytes:
    """What ``fprintf(fid, fmt)`` writes when ``fmt`` is given no data.

    Escapes are translated and ``%%`` is a literal ``%``; any other ``%``, or
    an unknown escape, ends the output there.
    """
    out, i, n = bytearray(), 0, len(fmt)
    while i < n:
        c = fmt[i:i + 1]
        if c == b'%':
            if fmt[i + 1:i + 2] == b'%':
                out += b'%'
                i += 2
                continue
            break
        if c == b'\\':
            e = fmt[i + 1:i + 2]
            if e in _ESCAPES:
                out += _ESCAPES[e]
                i += 2
                continue
            if e == b'x' or e.isdigit():
                raise VFFFilterError(f"octal/hex escape in header text {fmt!r}; "
                                     f"not emulated")
            break
        out += c
        i += 1
    return bytes(out)


def read_header_like_matlab(raw: bytes):
    """The header loop of filter_vff.m.

    Returns ``(header_out, data_offset, (A, B, C))``: the header bytes MATLAB
    writes (blank line and ``\\f\\n`` included), where its ``fread`` starts,
    and the three ``size=`` tokens.
    """
    pos, out, size = 0, bytearray(), None
    while True:
        nl = raw.find(b'\n', pos)
        if nl < 0:
            raise VFFFilterError("header ended without a line of length <= 1 "
                                 "(no form-feed line)")
        line = raw[pos:nl]              # fgetl: the line without its newline
        pos = nl + 1
        out += matlab_fprintf_no_data(_strip_trailing_whitespace(line) + b'\\n')
        if line.startswith(b'size='):
            tokens = line[len(b'size='):].replace(b';', b' ').split()
            if len(tokens) < 3:
                raise VFFFilterError(f"size line {line!r} does not hold three "
                                     f"dimensions")
            size = tuple(int(t) for t in tokens[:3])
        if len(line) <= 1:
            break
    if size is None:
        raise VFFFilterError("no 'size=' line in the header")
    out += b'\f\n'
    return bytes(out), pos, size


def matlab_patch_variance(payload) -> float:
    """``std2(im(20:40,160:180,100))^2`` on a (z, y, x) payload."""
    z = MATLAB_PATCH_SLICE - 1
    (x0, x1), (y0, y1) = MATLAB_PATCH_X, MATLAB_PATCH_Y
    nz, ny, nx = payload.shape
    if z >= nz or y1 > ny or x1 > nx:
        raise VFFFilterError(
            f"the MATLAB noise patch im({x0}:{x1},{y0}:{y1},{MATLAB_PATCH_SLICE}) "
            f"is outside a {nx} x {ny} x {nz} volume; pass sd")
    patch = np.asarray(payload[z, y0 - 1:y1, x0 - 1:x1], dtype=np.float64)
    return float(np.std(patch, ddof=1) ** 2)


def filter_vff(infile, sd=None, *, outfile=None, workers=None, verbose=True):
    """filter_vff.m. Writes ``<infile minus .vff>_imbilatfilt.vff`` (or
    ``outfile``) and returns ``(outfile, info)``.

    The whole int16 payload is held in memory twice (input and output), which
    for a 1162 x 1162 x 846 volume is ~4.6 GiB; the float64 work is per slice.
    """
    infile = Path(infile)
    if outfile is None:
        outfile = Path(str(infile)[:-4] + '_imbilatfilt.vff')
    outfile = Path(outfile)

    raw = infile.read_bytes()
    header_out, offset, (A, B, C) = read_header_like_matlab(raw)
    n = A * B * C
    if len(raw) - offset < 2 * n:
        raise VFFFilterError(f"{infile.name}: {len(raw) - offset} bytes after "
                             f"the header, need {2 * n} for {A} x {B} x {C} int16")
    payload = np.frombuffer(raw, dtype='>i2', count=n,
                            offset=offset).reshape(C, B, A)

    if sd is None:
        patch = matlab_patch_variance(payload)
        source = (f"std2(im({MATLAB_PATCH_X[0]}:{MATLAB_PATCH_X[1]},"
                  f"{MATLAB_PATCH_Y[0]}:{MATLAB_PATCH_Y[1]},{MATLAB_PATCH_SLICE}))^2")
    else:
        patch = float(sd) ** 2
        source = f"sd^2 (sd = {float(sd):g})"
    smooth = 2.0 * patch
    if not smooth > 0:
        raise VFFFilterError(f"degreeOfSmoothing = 2 * {source} = {smooth:g}; "
                             f"imbilatfilt requires it to be positive (a "
                             f"constant noise patch?) — pass sd")

    if verbose:
        print(f"  size: {A} x {B} x {C} (int16, big-endian)")
        print(f"  patch = {source} = {patch:.6g}")
        print(f"  degreeOfSmoothing = 2 * patch = {smooth:.6g} "
              f"(range sigma {math.sqrt(smooth):.4g})")
        print(f"  imbilatfilt: spatialSigma 1, 5 x 5 neighbourhood, replicate "
              f"padding, per slice")

    filtered = np.empty((C, B, A), dtype='>i2')

    def one(j):
        filtered[j] = matlab_int16(imbilatfilt(payload[j], smooth))

    workers = workers or min(32, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(one, range(C)):
            pass

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, 'wb') as f:
        f.write(header_out)
        filtered.tofile(f)

    info = {'size': [A, B, C], 'patch': patch, 'patch_source': source,
            'degree_of_smoothing': smooth, 'spatial_sigma_px': 1.0,
            'neighborhood_size': 5, 'padding': 'replicate'}
    return outfile, info
