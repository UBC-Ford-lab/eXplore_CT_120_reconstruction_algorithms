"""Per-pixel detector distortion ("unwarp"), shared by every backend.

Background
----------
The scanner's raw ``acq-*`` frames are geometrically distorted: the detector
pixel grid is not the ideal linear grid a projector assumes. The vendor
removes this before reconstructing (its ``Corrected/uwarp-*`` frames), and
since the 2024 recalibration its scan.xml geometry (pitch, centre of rotation,
distances) describes the detector AFTER that step. Reconstructing the raw
frames with that geometry splits every off-axis structure into two offset
halves in a short scan, and no centre-of-rotation value fixes it at more than
one height (Scan_2005, 2026-10-21): the apparent axis column moves with
detector position because the field does.

The field is measured once per detector epoch by registering the PHOTON
NOISE between paired ``acq``/``uwarp`` frames (signal-domain metrics are
blind to it; a 20 px warp moves whole-frame NCC in the 4th decimal), fitted
to a bicubic (20 coefficients per axis, ~0.5 px rms), and validated on
held-out frames: noise correlation 0.0007 -> 0.58. Fits on different scans of
the same epoch agree to ~0.2 px, so one calibration serves every scan taken
on that hardware between vendor recalibrations.

Two ways to apply it
--------------------
* ``warp_projections`` resamples the frames onto the ideal grid, exactly what
  the vendor does. One bilinear interpolation per frame, done once at load
  time in ``prepare_scan``, so FDK, ASTRA, TIGRE, the learned backends, the
  psi self-calibration and the support measurement all see corrected data.
* ``resolve_detector_warp`` returns the field for a projector that builds a
  ray per detector sample and can correct the sample POSITION instead
  (``learning_based_iterative.ray_sampler``). Exact, no interpolation, but
  only that family can use it, and it must not be combined with the first.

Epochs
------
A calibration is keyed by detector serial AND the scan.xml CentreOfRotation
the vendor wrote for that epoch (it changed 1758.48 -> 1744.96 at the 2024
recalibration, together with the detector distance). A scan whose centre
matches a stored calibration gets the FULL field by default. A scan from
another epoch on the same detector gets nothing unless asked
(``nonaffine``): the non-affine part transfers in principle (2022 scan,
2026 field, vendor-volume NCC 0.688 -> 0.713), but the 2022 scans
reconstruct round as they are and resampling costs sharpness, so it is an
opt-in, not a default.

Convention
----------
The stored field ``d`` satisfies ``ideal(p) = raw(p + d(p))`` in RAW
detector pixels: the ideal frame at pixel ``p`` reads the raw frame at
``p + d(p)``. ``ideal_indices`` solves the inverse for the ray-domain use.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

from .paths import calibration_dir
from .vff_io import detector_serial_from_scan

#: Two calibrations are the same epoch when the vendor centre of rotation
#: they were taken under agrees to this many raw pixels.
EPOCH_TOLERANCE_PX = 0.5

MODES = ('auto', 'off', 'full', 'nonaffine')


def _calib_dir(calib_dir=None) -> Path:
    return Path(calib_dir) if calib_dir else calibration_dir()


# ---------------------------------------------------------------- helpers ---

def poly_terms(deg: int) -> list[tuple[int, int]]:
    """Exponent pairs (i, j) for r^i c^j, matching the fitting basis order."""
    return [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]


def _basis(rn, cn, deg: int):
    """Design matrix for the 2-D monomial basis, numpy or torch."""
    if isinstance(rn, torch.Tensor):
        return torch.stack([rn ** i * cn ** j for i, j in poly_terms(deg)], -1)
    return np.stack([rn ** i * cn ** j for i, j in poly_terms(deg)], -1)


def _moment(k: int) -> float:
    """Mean of x^k over x in [-1, 1]: 1/(k+1) for even k, 0 for odd."""
    return 0.0 if k % 2 else 1.0 / (k + 1)


def strip_affine(coefs: np.ndarray, deg: int) -> np.ndarray:
    """Coefficients of the polynomial minus its least-squares affine fit.

    Projected ANALYTICALLY over the continuous square [-1, 1]^2, where
    {1, r, c} are orthogonal, so the result does not depend on a sampling
    grid's resolution.
    """
    out = np.array(coefs, dtype=np.float64, copy=True)
    terms = poly_terms(deg)
    idx = {t: k for k, t in enumerate(terms)}
    for axis in range(out.shape[0]):
        f = out[axis]
        a0 = sum(f[k] * _moment(i) * _moment(j) for k, (i, j) in enumerate(terms))
        a1 = 3.0 * sum(f[k] * _moment(i + 1) * _moment(j)
                       for k, (i, j) in enumerate(terms))
        a2 = 3.0 * sum(f[k] * _moment(i) * _moment(j + 1)
                       for k, (i, j) in enumerate(terms))
        out[axis][idx[(0, 0)]] -= a0
        out[axis][idx[(1, 0)]] -= a1
        out[axis][idx[(0, 1)]] -= a2
    return out


def xml_centre_of_rotation(xml_header) -> float | None:
    """The vendor's CentreOfRotation from a parsed scan.xml (epoch key)."""
    try:
        return float(xml_header['Series']['CentreOfRotation'])
    except (KeyError, TypeError, ValueError):
        return None


# ------------------------------------------------------------------ model ---

class DetectorWarp:
    """A detector distortion calibration.

    ``mode`` selects what is applied: ``full`` (the measured field),
    ``nonaffine`` (the field minus its affine part), or ``none`` — a stored
    statement that this detector epoch needs NO unwarp (its raw frames were
    verified to reconstruct round), which is as much a calibration as a
    field is: it is what lets an unknown epoch be told apart from a known
    good one.
    """

    def __init__(self, coefs, deg: int, raw_shape, *, mode: str = "nonaffine",
                 detector_serial: str | None = None, source: str | None = None,
                 cor_xml: float | None = None, rfan: float | None = None,
                 scans: tuple[str, ...] = (), measured_on: str | None = None,
                 note: str = ""):
        self.note = str(note or "")
        if mode == "none":
            coefs = np.zeros((2, len(poly_terms(int(deg)))))
        coefs = np.asarray(coefs, dtype=np.float64)
        if coefs.ndim != 2 or coefs.shape[0] != 2:
            raise ValueError(f"coefs must be (2, n_terms); got {coefs.shape}")
        if coefs.shape[1] != len(poly_terms(deg)):
            raise ValueError(
                f"coefs has {coefs.shape[1]} terms but degree {deg} needs "
                f"{len(poly_terms(deg))}")
        if mode not in ("nonaffine", "full", "none"):
            raise ValueError(f"mode must be 'nonaffine', 'full' or 'none'; got {mode!r}")
        self.deg = int(deg)
        self.mode = mode
        self.raw_shape = (int(raw_shape[0]), int(raw_shape[1]))
        self.detector_serial = detector_serial
        self.source = source
        self.cor_xml = None if cor_xml is None else float(cor_xml)
        self.rfan = None if rfan is None else float(rfan)
        self.scans = tuple(scans)
        self.measured_on = measured_on
        self.full_coefs = np.array(coefs, dtype=np.float64, copy=True)
        self.coefs = (strip_affine(coefs, self.deg) if mode == "nonaffine"
                      else self.full_coefs)

        #: optional dense field (2, H, W) float32 in raw px, (d_row, d_col):
        #: when present the projection-domain unwarp samples it directly
        #: (exact, no polynomial residual); the polynomial is still what the
        #: ray-domain inverse and the stored calibrations use.
        self.dense = None

    @property
    def is_null(self) -> bool:
        return self.mode == "none"

    @classmethod
    def from_vendor_table(cls, scan_folder, *, deg: int = 3,
                          detector_serial: str | None = None,
                          cor_xml: float | None = None, rfan: float | None = None,
                          raw_shape=None) -> "DetectorWarp":
        """The field the vendor itself applied, from the scan's ``transform1``.

        Every scan folder carries ``transform1``: a VTK grid transform,
        little-endian float32 (rows, cols, 3) over the detector, rows stored
        bottom-up (VTK's y-up), channel 0 the column displacement, channel 1
        the row displacement, channel 2 zero. Decoded against the field
        measured from paired acq/uwarp frames of the same epoch it agrees to
        R^2 0.998 / 0.986 with 0.4 / 0.6 px residuals (the measurement's own
        noise), and its MD5 is the ``transform_checksum`` the vendor stamps
        into its uwarp frames. It is therefore the exact per-epoch
        calibration, present for epochs that never archived Corrected frames.
        """
        scan_folder = Path(scan_folder)
        path = scan_folder / 'transform1'
        if raw_shape is None:
            from .vff_io import read_vff_header
            frame = next(iter(sorted(scan_folder.glob('acq-*.vff'))), None)
            if frame is None:
                raise FileNotFoundError(f"no acq-*.vff in {scan_folder}")
            h = read_vff_header(str(frame), verbose=False)
            W, H = (int(v) for v in h['size'].split()[:2])
        else:
            H, W = (int(v) for v in raw_shape)
        n = path.stat().st_size
        if n != H * W * 3 * 4:
            raise ValueError(f"{path} is {n} bytes, not a {H}x{W}x3 float32 table")
        t = np.fromfile(str(path), dtype='<f4').reshape(H, W, 3)[::-1]
        dense = np.ascontiguousarray(np.stack([t[..., 1], t[..., 0]]), dtype=np.float32)
        # a compact polynomial for storage / the ray-domain inverse
        rr, cc = np.meshgrid(np.linspace(0, H - 1, min(H, 144)), np.linspace(0, W - 1, min(W, 220)), indexing='ij')
        rr, cc = rr.astype(int), cc.astype(int)
        rn = (rr - (H - 1) / 2) / ((H - 1) / 2); cn = (cc - (W - 1) / 2) / ((W - 1) / 2)
        A = _basis(rn.ravel(), cn.ravel(), deg)
        coefs = np.array([np.linalg.lstsq(A, f[rr, cc].ravel(), rcond=None)[0] for f in dense])
        import hashlib
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        w = cls(coefs, deg, (H, W), mode='full', detector_serial=detector_serial,
                source=str(path), cor_xml=cor_xml, rfan=rfan, scans=(scan_folder.name,),
                note=f"vendor transform1 table (md5 {md5[:8]})")
        w.dense = dense
        w.md5 = md5
        return w

    # -- construction ----------------------------------------------------
    @classmethod
    def load(cls, path, mode: str = "nonaffine") -> "DetectorWarp":
        z = np.load(str(path), allow_pickle=False)

        def opt(key, conv=float):
            if key not in z:
                return None
            v = z[key]
            try:
                return conv(v) if v.shape == () or key != 'scans' else conv(v)
            except (TypeError, ValueError):
                return None

        cor = opt('cor_xml'); rfan = opt('rfan')
        scans = tuple(str(s) for s in z['scans']) if 'scans' in z else ()
        measured = str(z['measured_on']) if 'measured_on' in z else None
        note = str(z['note']) if 'note' in z else ''
        if 'mode' in z and str(z['mode']) == 'none':
            mode = 'none'          # a stored "no unwarp" is not re-interpretable
        return cls(z["coefs"], int(z["deg"]), z["raw_shape"], mode=mode, note=note,
                   detector_serial=(str(z["detector_serial"])
                                    if "detector_serial" in z else None),
                   source=str(path),
                   cor_xml=None if cor is None or not np.isfinite(cor) else cor,
                   rfan=None if rfan is None or not np.isfinite(rfan) else rfan,
                   scans=scans, measured_on=measured)

    @classmethod
    def for_detector(cls, serial: str, mode: str = "nonaffine",
                     calib_dir: Path | None = None,
                     cor_xml: float | None = None) -> "DetectorWarp | None":
        """The calibration for a DETECTOR SERIAL, preferring the one from the
        same epoch as ``cor_xml`` when several exist."""
        cands = find_calibrations(serial, calib_dir)
        if not cands:
            return None
        if cor_xml is not None:
            same = [w for w in cands if w.cor_xml is not None
                    and abs(w.cor_xml - cor_xml) <= EPOCH_TOLERANCE_PX]
            if same:
                cands = same
        w = cands[0]
        return cls.load(w.source, mode=mode)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, coefs=self.full_coefs, deg=self.deg,
                 raw_shape=np.array(self.raw_shape),
                 detector_serial=str(self.detector_serial or ""),
                 mode=self.mode,
                 cor_xml=np.float64(np.nan if self.cor_xml is None else self.cor_xml),
                 rfan=np.float64(np.nan if self.rfan is None else self.rfan),
                 scans=np.array(list(self.scans), dtype=str),
                 measured_on=str(self.measured_on or ""),
                 note=str(self.note or ""))
        return path

    def same_epoch(self, cor_xml: float | None) -> bool:
        return (self.cor_xml is not None and cor_xml is not None
                and abs(self.cor_xml - float(cor_xml)) <= EPOCH_TOLERANCE_PX)

    # -- evaluation ------------------------------------------------------
    def _displacement_raw(self, r_raw, c_raw):
        """Field in RAW detector pixels at raw (possibly fractional) coords."""
        H, W = self.raw_shape
        rn = (r_raw - (H - 1) / 2.0) / ((H - 1) / 2.0)
        cn = (c_raw - (W - 1) / 2.0) / ((W - 1) / 2.0)
        B = _basis(rn, cn, self.deg)
        if isinstance(B, torch.Tensor):
            co = torch.as_tensor(self.coefs, dtype=B.dtype, device=B.device)
            return B @ co[0], B @ co[1]
        return B @ self.coefs[0], B @ self.coefs[1]

    def ideal_indices(self, b_idx, a_idx, downsample: int = 1, iters: int = 2):
        """Map raw pipeline indices to IDEAL detector indices (ray-domain use).

        Solves ``q = p + d(p)`` for ``p`` by fixed-point iteration, in raw
        pixel units, then converts back to pipeline index units. Pipeline
        index k covers raw samples [ds*k, ds*k + ds), centred at
        ``ds*k + (ds-1)/2``.
        """
        ds = int(downsample)
        off = (ds - 1) / 2.0
        b_f = b_idx.to(torch.float64) if torch.is_tensor(b_idx) else np.asarray(b_idx, float)
        a_f = a_idx.to(torch.float64) if torch.is_tensor(a_idx) else np.asarray(a_idx, float)
        q_r = b_f * ds + off
        q_c = a_f * ds + off
        p_r, p_c = q_r, q_c
        for _ in range(max(1, iters)):
            d_r, d_c = self._displacement_raw(p_r, p_c)
            p_r = q_r - d_r
            p_c = q_c - d_c
        return (p_r - off) / ds, (p_c - off) / ds

    def sampling_grid(self, device=None, dtype=torch.float32) -> torch.Tensor:
        """``grid_sample`` coordinates that read ``raw(p + d(p))`` at every
        raw pixel ``p`` — the projection-domain unwarp, (1, H, W, 2). Uses
        the dense vendor table when the field came from one."""
        H, W = self.raw_shape
        rr, cc = torch.meshgrid(
            torch.arange(H, dtype=torch.float64, device=device),
            torch.arange(W, dtype=torch.float64, device=device), indexing="ij")
        if self.dense is not None and self.mode == 'full':
            d = torch.as_tensor(self.dense, dtype=torch.float64, device=device)
            d_r, d_c = d[0], d[1]
        else:
            d_r, d_c = self._displacement_raw(rr, cc)
        gy = 2.0 * (rr + d_r) / (H - 1) - 1.0
        gx = 2.0 * (cc + d_c) / (W - 1) - 1.0
        return torch.stack([gx, gy], -1).unsqueeze(0).to(dtype)

    # -- diagnostics -----------------------------------------------------
    def magnitude(self):
        """(mean, max) displacement in raw px over the detector."""
        H, W = self.raw_shape
        rr, cc = np.meshgrid(np.linspace(0, H - 1, 64), np.linspace(0, W - 1, 64),
                             indexing="ij")
        d_r, d_c = self._displacement_raw(rr, cc)
        mag = np.hypot(d_r, d_c)
        return float(mag.mean()), float(mag.max())

    def summary(self, downsample: int = 1) -> str:
        H, W = self.raw_shape
        if self.is_null:
            return (f"none (raw frames verified correct) serial="
                    f"{self.detector_serial or '?'} epoch CoR {self.cor_xml}"
                    f"{' — ' + self.note if self.note else ''}")
        mean, mx = self.magnitude()
        epoch = (f" epoch CoR {self.cor_xml:.2f}" if self.cor_xml is not None
                 else "")
        return (f"mode={self.mode} deg={self.deg} raw={H}x{W} "
                f"serial={self.detector_serial or '?'}{epoch} | "
                f"|d| mean {mean:.2f} px max {mx:.2f} px "
                f"(= {mean/max(downsample,1):.2f} / "
                f"{mx/max(downsample,1):.2f} pipeline px)")

    def tag(self) -> str:
        """Short provenance string for sidecars and the psi cache."""
        if self.dense is not None:
            return f"{self.mode}:transform1:{getattr(self, 'md5', '')[:8]}"
        src = Path(self.source).name if self.source else 'in-memory'
        return f"{self.mode}:{src}"


# ---------------------------------------------------------------- lookup ---

def calibration_path(serial: str, cor_xml: float | None,
                     calib_dir=None) -> Path:
    """Where a calibration for this detector epoch is stored."""
    suffix = "" if cor_xml is None else f"_cor{float(cor_xml):.2f}"
    return _calib_dir(calib_dir) / f"detector_warp_{serial}{suffix}.npz"


#: Calibrations shipped inside the package (see calibration/README.md); the
#: user's own calibration directory is searched first and wins on a tie.
BUNDLED_DIR = Path(__file__).resolve().parent.parent / 'calibration'


def find_calibrations(serial: str, calib_dir=None) -> list[DetectorWarp]:
    """Every stored calibration for a detector serial, newest epoch first.

    The user's calibration directory first, then the package's bundled ones;
    a bundled file is only listed when no user file has the same name.
    """
    d = _calib_dir(calib_dir)
    paths = sorted(glob.glob(str(d / f"detector_warp_{serial}*.npz")))
    names = {Path(p).name for p in paths}
    if calib_dir is None and BUNDLED_DIR.is_dir():
        paths += [p for p in sorted(glob.glob(str(BUNDLED_DIR / f"detector_warp_{serial}*.npz")))
                  if Path(p).name not in names]
    out = []
    for p in paths:
        try:
            out.append(DetectorWarp.load(p, mode="full"))
        except Exception:
            continue
    out.sort(key=lambda w: (w.cor_xml is None, -(w.cor_xml or 0.0)))
    return out


def vendor_table_checksum_ok(scan_folder) -> bool | None:
    """Whether ``transform1``'s MD5 matches the ``transform_checksum`` the
    vendor wrote into this scan's uwarp frames. None when there is nothing
    to compare against (no Corrected frames, or no checksum in them)."""
    scan_folder = Path(scan_folder)
    u = next(iter(sorted((scan_folder / 'Corrected').glob('uwarp-*.vff'))), None)
    if u is None or not (scan_folder / 'transform1').exists():
        return None
    head = u.open('rb').read(4096).decode('latin-1')
    m = re.search(r'transform_checksum=([0-9a-f]{32})', head)
    if not m:
        return None
    import hashlib
    return hashlib.md5((scan_folder / 'transform1').read_bytes()).hexdigest() == m.group(1)


def has_vendor_table(scan_folder, raw_shape) -> bool:
    p = Path(scan_folder) / 'transform1'
    return p.exists() and p.stat().st_size == int(raw_shape[0]) * int(raw_shape[1]) * 12


def has_paired_frames(scan_folder) -> bool:
    """True when the scan carries the vendor's unwarped frames next to the
    raw ones, i.e. the field can be measured from this scan itself."""
    scan_folder = Path(scan_folder)
    return (any((scan_folder / 'Corrected').glob('uwarp-*.vff'))
            and any(scan_folder.glob('acq-*.vff')))


def select_projection_warp(scan_folder, xml_header, raw_detector_shape, *,
                           mode: str = 'auto', calib_dir=None,
                           verbose: bool = True,
                           allow_measure: bool = True) -> DetectorWarp | None:
    """The field to resample a scan's projections with, by policy.

    The unit of knowledge is a detector EPOCH: one serial under one vendor
    geometry (scan.xml CentreOfRotation, which the vendor rewrites whenever
    it recalibrates — a source change, a detector re-seat). For each epoch
    the calibration directory either holds a measured field, a stored
    "none" (raw frames verified to reconstruct round), or nothing.

    ``auto``
        scan carries the vendor's own ``transform1`` table
                                  -> apply it (exact, per-epoch by construction;
                                     checked against the uwarp checksum when
                                     Corrected frames are present).
        known epoch with a field  -> apply the full field.
        known epoch, "none"       -> raw frames as-is.
        unknown epoch, scan has ``Corrected/uwarp-*`` frames
                                  -> MEASURE the field from this scan, store it
                                     for the epoch, apply it.
        unknown epoch, no paired frames, detector has calibrations
                                  -> STOP: this detector is known to need an
                                     epoch calibration and this epoch has
                                     none; the message says what to archive.
        detector never calibrated -> raw frames, with a warning (nothing is
                                     known either way).
    ``full``       require a same-epoch field (raise otherwise).
    ``nonaffine``  the transferable part of another epoch's field (opt-in).
    ``off``        nothing.
    """
    mode = str(mode or 'auto').lower()
    if mode not in MODES:
        raise ValueError(f"detector warp mode must be one of {MODES}; got {mode!r}")
    if mode == 'off':
        if verbose:
            print("  Detector warp: off (raw frames used as-is).")
        return None
    serial = detector_serial_from_scan(scan_folder)
    cor = xml_centre_of_rotation(xml_header)
    cands = find_calibrations(serial, calib_dir) if serial else []
    same = [w for w in cands if w.same_epoch(cor)]
    fields = [w for w in cands if not w.is_null]
    where = _calib_dir(calib_dir)
    shape = tuple(int(v) for v in raw_detector_shape)

    def _check_shape(w):
        if tuple(w.raw_shape) != shape:
            raise ValueError(
                f"Detector warp calibration {Path(w.source).name} was fitted "
                f"on a {w.raw_shape[0]}x{w.raw_shape[1]} detector but this "
                f"scan is {shape[0]}x{shape[1]}. Refusing to apply a "
                f"mismatched field.")
        return w

    if mode == 'full':
        if not same or same[0].is_null:
            raise FileNotFoundError(
                f"--detector-warp full needs a measured field for detector "
                f"{serial or 'unknown'} at CentreOfRotation {cor} in {where}; "
                f"found {[Path(w.source).name for w in cands] or 'none'}.")
        warp = _check_shape(DetectorWarp.load(same[0].source, mode='full'))
    elif mode == 'nonaffine':
        if not fields:
            raise FileNotFoundError(
                f"--detector-warp nonaffine needs a measured field for "
                f"detector {serial or 'unknown'} in {where}; found none.")
        src = ([w for w in same if not w.is_null] or fields)[0].source
        warp = _check_shape(DetectorWarp.load(src, mode='nonaffine'))
    else:  # auto
        if has_vendor_table(scan_folder, shape):
            warp = DetectorWarp.from_vendor_table(
                scan_folder, detector_serial=serial, cor_xml=cor, raw_shape=shape)
            ok = vendor_table_checksum_ok(scan_folder)
            if verbose:
                print(f"  Detector warp: {warp.summary()}")
                print(f"                 from the scan's own vendor table "
                      f"(transform1, md5 {warp.md5[:8]}"
                      f"{'' if ok is None else ', matches the uwarp frames' if ok else ', DOES NOT MATCH the uwarp frames checksum'})")
            if ok is False:
                raise RuntimeError(
                    f"{Path(scan_folder).name}: transform1 (md5 {warp.md5}) is not "
                    f"the table the vendor applied to this scan's Corrected "
                    f"frames; refusing to use it. Pass --detector-warp off, or "
                    f"remove the file so the stored epoch calibration is used.")
            return warp
        if same and same[0].is_null:
            if verbose:
                print(f"  Detector warp: none — epoch CoR {cor} of detector "
                      f"{serial} is on record as needing no unwarp "
                      f"({Path(same[0].source).name}"
                      f"{': ' + same[0].note if same[0].note else ''}).")
            return None
        if same:
            warp = _check_shape(DetectorWarp.load(same[0].source, mode='full'))
        elif allow_measure and has_paired_frames(scan_folder):
            if verbose:
                print(f"  Detector warp: no calibration for detector "
                      f"{serial or 'unknown'} at CentreOfRotation {cor} "
                      f"(known epochs: {[w.cor_xml for w in cands] or 'none'}), "
                      f"but this scan carries Corrected/uwarp-* frames — "
                      f"measuring the field from them now (one-off, cached "
                      f"per epoch).")
            warp = measure_and_store(scan_folder, xml_header, out_dir=calib_dir,
                                     verbose=verbose)
            warp = _check_shape(DetectorWarp.load(warp.source, mode='full'))
        elif cands:
            raise DetectorEpochUnknown(
                f"Detector {serial} is known to need a per-epoch unwarp "
                f"calibration, but this scan's geometry epoch "
                f"(scan.xml CentreOfRotation {cor}) matches none on record "
                f"({[w.cor_xml for w in cands]}; a source or detector change "
                f"gives a new one). Reconstructing the raw frames would split "
                f"off-axis structures. To fix it once for this epoch: archive "
                f"any scan taken under it WITH its Corrected/uwarp-* frames "
                f"and run it (or `python -m explore_ct120_recon.ct_core."
                f"detector_warp measure <scan>`); the field is then stored in "
                f"{where} and reused. If you have verified that this epoch's "
                f"raw frames reconstruct round, record that with "
                f"`... detector_warp none <scan> --note ...`. To proceed "
                f"anyway, pass --detector-warp off.")
        else:
            if verbose:
                print(f"  Detector warp: nothing known about detector "
                      f"{serial or 'unknown'} (no calibration in {where}) — "
                      f"raw frames used UNCORRECTED. If off-axis structures "
                      f"come out doubled, archive a scan with its "
                      f"Corrected/uwarp-* frames and reconstruct it once.")
            return None
    if verbose:
        why = ("same epoch as the scan" if warp.mode == 'full'
               else f"other epoch (scan CoR {cor}, calibration CoR "
                    f"{warp.cor_xml}); non-affine part only")
        print(f"  Detector warp: {warp.summary()}")
        print(f"                 {why}; from {Path(warp.source).name}")
    return warp


class DetectorEpochUnknown(RuntimeError):
    """A detector known to need unwarping, in a geometry epoch with no
    calibration and no paired frames to measure one from."""


# ------------------------------------------------------------ application ---

def warp_frames(frames, warp: DetectorWarp, *, device=None, grid=None):
    """Resample a (N, H, W) or (H, W) array onto the ideal detector grid.

    Bilinear, border-padded, returned as float32 in the input's layout. The
    grid is built once per call (or passed in) so a stack costs one
    ``grid_sample`` per frame.
    """
    arr = np.asarray(frames)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None]
    if tuple(arr.shape[1:]) != tuple(warp.raw_shape):
        raise ValueError(f"frames are {arr.shape[1:]}, calibration is "
                         f"{warp.raw_shape}")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if grid is None:
        grid = warp.sampling_grid(device=device)
    x = torch.as_tensor(np.asarray(arr, dtype=np.float32), device=device)
    out = torch.nn.functional.grid_sample(
        x.unsqueeze(1), grid.expand(x.shape[0], -1, -1, -1),
        mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
    out = out.cpu().numpy()
    return out[0] if squeeze else out


def pooled_frame_transform(warp, factor: int, *, device=None):
    """A per-frame ``pool, then unwarp`` for a coarse preview.

    The drivers unwarp every frame at full resolution and pool afterwards;
    at a preview pitch (0.4 mm voxels, the detector pooled 4x) that order
    spends ten seconds resampling pixels that are then averaged away. This
    returns a callable ``frame (H, W) -> (H//factor, W//factor) float32``
    that average-pools first and resamples the POOLED frame with the warp
    field pooled the same way (the displacement, in pooled pixels, is the
    block mean of the raw displacement over ``factor``). The field is
    smooth over many pixels, so at the preview pitch the two orders agree;
    this one is not for a delivered reconstruction. ``warp`` may be None
    (pooling only). Everything runs on ``device`` (the GPU if there is one).
    """
    factor = max(1, int(factor))
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid = None
    if warp is not None:
        H, W = warp.raw_shape
        Hp, Wp = H // factor, W // factor
        full = warp.sampling_grid(device=device, dtype=torch.float64)[0]      # (H, W, 2) normalised
        gx, gy = full[..., 0], full[..., 1]
        rr, cc = torch.meshgrid(torch.arange(H, dtype=torch.float64, device=device),
                                torch.arange(W, dtype=torch.float64, device=device), indexing='ij')
        d_c = (gx + 1.0) * (W - 1) / 2.0 - cc                                 # raw px
        d_r = (gy + 1.0) * (H - 1) / 2.0 - rr
        if factor > 1:
            pool = torch.nn.functional.avg_pool2d
            d_c = pool(d_c[None, None, :Hp * factor, :Wp * factor], factor)[0, 0] / factor
            d_r = pool(d_r[None, None, :Hp * factor, :Wp * factor], factor)[0, 0] / factor
        jj, ii = torch.meshgrid(torch.arange(Hp, dtype=torch.float64, device=device),
                                torch.arange(Wp, dtype=torch.float64, device=device), indexing='ij')
        grid = torch.stack([2.0 * (ii + d_c) / max(Wp - 1, 1) - 1.0,
                            2.0 * (jj + d_r) / max(Hp - 1, 1) - 1.0], -1)[None].to(torch.float32)

    def transform(frame):
        arr = np.asarray(frame)
        if warp is not None and tuple(arr.shape) != tuple(warp.raw_shape):
            raise ValueError(f"frame is {arr.shape}, calibration is {warp.raw_shape}")
        x = torch.as_tensor(np.asarray(arr, dtype=np.float32), device=device)[None, None]
        if factor > 1:
            h, w = x.shape[-2:]
            x = torch.nn.functional.avg_pool2d(x[..., :(h // factor) * factor, :(w // factor) * factor], factor)
        if grid is not None:
            x = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border',
                                                align_corners=True)
        return x[0, 0].cpu().numpy()

    return transform


def warp_projections(projections, bright_field, dark_field, warp: DetectorWarp,
                     *, chunk: int = 8, verbose: bool = True):
    """Unwarp a whole scan in place: every projection, plus the flat fields.

    The frames keep their dtype (counts are rounded back), so a 3.5 GB stack
    stays 3.5 GB. Bright and dark are warped with the SAME field so the
    flat-field ratio later compares a pixel with its own gain. Returns
    ``(projections, bright_field, dark_field)``.
    """
    import time
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid = warp.sampling_grid(device=device)
    n = int(projections.shape[0])
    dt = projections.dtype
    is_int = np.issubdtype(dt, np.integer)
    info = np.iinfo(dt) if is_int else None
    for start in range(0, n, max(1, int(chunk))):
        end = min(n, start + chunk)
        w = warp_frames(projections[start:end], warp, device=device, grid=grid)
        if is_int:
            w = np.clip(np.rint(w), info.min, info.max)
        projections[start:end] = w.astype(dt, copy=False)
    out = [projections]
    for f in (bright_field, dark_field):
        if f is None:
            out.append(None)
            continue
        fw = warp_frames(f, warp, device=device, grid=grid)
        out.append(fw.astype(np.float32) if not np.issubdtype(f.dtype, np.integer)
                   else np.clip(np.rint(fw), np.iinfo(f.dtype).min,
                                np.iinfo(f.dtype).max).astype(f.dtype))
    if verbose:
        print(f"  Detector warp: {n} projections + flat fields resampled onto "
              f"the ideal grid in {time.time() - t0:.0f}s.")
    return tuple(out)


# ------------------------------------------------- ray-domain (learned) ---

def resolve_detector_warp(geom_cfg: dict, scan_folder, raw_detector_shape,
                          downsample: int = 1, xml_header=None,
                          cor_xml: float | None = None) -> DetectorWarp | None:
    """The field for a RAY-DOMAIN correction, from ``geometry.detector_warp``.

    Modes ``off`` (default when the config says nothing), ``auto``
    (non-affine if a calibration exists, else a loud warning), ``nonaffine``
    and ``full`` (require one). Only the learning-based projector consumes
    this; a scan whose projections were already resampled must not get it a
    second time — ``prepare_scan`` records that on the geometry.
    """
    cfg = (geom_cfg or {}).get("detector_warp", {}) or {}
    mode = str(cfg.get("mode", "off")).lower()
    if mode in ("off", "none", "false", "disabled"):
        return None
    explicit = cfg.get("path")
    serial = detector_serial_from_scan(scan_folder)
    apply_mode = "nonaffine" if mode in ("auto", "nonaffine") else "full"
    if cor_xml is None and xml_header is not None:
        cor_xml = xml_centre_of_rotation(xml_header)
    if explicit:
        warp = DetectorWarp.load(explicit, mode=apply_mode)
    elif serial:
        warp = DetectorWarp.for_detector(serial, mode=apply_mode,
                                         calib_dir=cfg.get("calib_dir"),
                                         cor_xml=cor_xml)
    else:
        warp = None
    if warp is None:
        msg = (f"  Detector warp: NO CALIBRATION for serial "
               f"{serial or 'unknown'} (looked in {_calib_dir(cfg.get('calib_dir'))}).")
        if mode == "auto":
            print(msg + " Proceeding UNCORRECTED — the projector assumes an "
                        "ideal linear detector grid.")
            return None
        raise FileNotFoundError(
            msg + f" geometry.detector_warp.mode={mode!r} requires one; set "
                  f"mode: auto or off, or supply an explicit path.")
    if tuple(warp.raw_shape) != tuple(raw_detector_shape):
        raise ValueError(
            f"Detector warp calibration was fitted on a "
            f"{warp.raw_shape[0]}x{warp.raw_shape[1]} detector but this scan "
            f"is {raw_detector_shape[0]}x{raw_detector_shape[1]}. Refusing to "
            f"apply a mismatched field.")
    print(f"  Detector warp (ray domain): {warp.summary(downsample)}")
    if warp.source:
        print(f"                 from {warp.source}")
    return warp


# ------------------------------------------------------------- measurement ---

def _highpass(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    sm = torch.nn.functional.avg_pool2d(x[None, None], k, stride=1, padding=k // 2)[0, 0]
    return x - sm


def _hf_peak(a: torch.Tensor, b: torch.Tensor, max_shift: int):
    """FFT cross-correlation peak of two blocks with parabolic sub-pixel
    refinement: (dr, dc, peak, peak/second-peak)."""
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    H, W = a.shape
    c = torch.fft.irfft2(torch.fft.rfft2(b) * torch.fft.rfft2(a).conj(), s=(H, W))
    c = torch.fft.fftshift(c) / denom
    cr, cc = H // 2, W // 2
    win = c[cr - max_shift:cr + max_shift + 1, cc - max_shift:cc + max_shift + 1]
    idx = int(torch.argmax(win)); pr, pc = idx // win.shape[1], idx % win.shape[1]
    peak = float(win[pr, pc])
    w2 = win.clone(); w2[max(0, pr - 2):pr + 3, max(0, pc - 2):pc + 3] = -1e9
    second = float(w2.max())

    def sub(m, o, p):
        d = m - 2 * o + p
        return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (m - p) / d, -1, 1))

    dr, dc = float(pr - max_shift), float(pc - max_shift)
    if 0 < pr < win.shape[0] - 1:
        dr += sub(float(win[pr - 1, pc]), peak, float(win[pr + 1, pc]))
    if 0 < pc < win.shape[1] - 1:
        dc += sub(float(win[pr, pc - 1]), peak, float(win[pr, pc + 1]))
    return -dr, -dc, peak, peak / max(abs(second), 1e-9)


def _shift(img: torch.Tensor, dr: float, dc: float) -> torch.Tensor:
    if dr == 0.0 and dc == 0.0:
        return img
    H, W = img.shape
    rr, cc = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=img.device),
                            torch.arange(W, dtype=torch.float32, device=img.device), indexing="ij")
    grid = torch.stack([2.0 * (cc + dc) / (W - 1) - 1.0, 2.0 * (rr + dr) / (H - 1) - 1.0], -1)[None]
    return torch.nn.functional.grid_sample(img[None, None], grid, mode="bilinear",
                                           padding_mode="border", align_corners=True)[0, 0]


def measure_field(scan_folder, *, n_fit: int = 12, n_test: int = 4,
                  block: int = 256, grid=(11, 17), max_shift: int = 40,
                  min_ratio: float = 1.8, deg: int = 3, device=None,
                  verbose: bool = True) -> dict:
    """Measure the acq -> uwarp detector field from a scan's own paired frames.

    Photon noise is white, so the cross-correlation of the HIGH-PASSED raw
    line integral with the high-passed vendor frame peaks sharply at the
    local displacement even where the image content gives no handle (and
    the image content is what makes whole-frame metrics blind to a 20 px
    warp). Blocks on a grid across the detector, the per-block median over
    frames (the field is static), a robust degree-3 polynomial fit, and a
    validation on frames the fit never saw.

    Returns the fit record (``coefs`` with the global shift folded in,
    ``deg``, ``shape``, ``dr0``/``dc0`` = 0, held-out correlations).
    """
    import time
    from .vff_io import read_vff
    scan_folder = Path(scan_folder)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device(device)

    def load(p):
        return torch.from_numpy(np.asarray(read_vff(str(p), verbose=False)[1])
                                .squeeze().astype(np.float32)).to(dev)
    bright, dark = load(scan_folder / 'bright.vff'), load(scan_folder / 'dark.vff')
    denom = (bright - dark).clamp_min(1.0)
    pairs = []
    for u in sorted((scan_folder / 'Corrected').glob('uwarp-*.vff')):
        a = scan_folder / u.name.replace('uwarp-', 'acq-')
        if a.exists():
            pairs.append((a, u))
    if len(pairs) < n_fit + n_test:
        raise RuntimeError(f"{scan_folder.name}: only {len(pairs)} acq/uwarp "
                           f"pairs, need {n_fit + n_test}")
    pick = [pairs[i] for i in np.linspace(0, len(pairs) - 1, n_fit + n_test).astype(int)]
    test = pick[::(n_fit + n_test) // n_test][:n_test]
    fit = [p for p in pick if p not in test]
    H, W = bright.shape
    t0 = time.time()

    def line_integral(p):
        return -torch.log(((load(p) - dark) / denom).clamp_min(1e-6))

    # global shift first, from the centre block of the first pair, so the
    # per-block search window is centred on the field rather than on zero
    hp_l0 = _highpass(line_integral(fit[0][0])); hp_u0 = _highpass(load(fit[0][1]))
    rc, cc0 = H // 2, W // 2
    dr0, dc0, _, _ = _hf_peak(hp_l0[rc - 256:rc + 256, cc0 - 256:cc0 + 256],
                              hp_u0[rc - 256:rc + 256, cc0 - 256:cc0 + 256], 60)
    if verbose:
        print(f"  measuring detector field on {scan_folder.name}: {len(fit)} "
              f"frames, {grid[0]}x{grid[1]} blocks of {block} px, global shift "
              f"({dr0:+.2f}, {dc0:+.2f}) px")
    rs = np.linspace(block, H - block, grid[0]).astype(int)
    cs = np.linspace(block, W - block, grid[1]).astype(int)
    pts = [(r, c) for r in rs for c in cs]
    samples = {p: [] for p in pts}
    for k, (a, u) in enumerate(fit):
        hp_l = _highpass(_shift(line_integral(a), dr0, dc0)); hp_u = _highpass(load(u))
        for (r, c) in pts:
            sl = (slice(r - block // 2, r + block // 2), slice(c - block // 2, c + block // 2))
            dr, dc, peak, ratio = _hf_peak(hp_l[sl], hp_u[sl], max_shift)
            if ratio >= min_ratio:
                samples[(r, c)].append((dr, dc))
    good = [(p, np.median(np.array(v), axis=0)) for p, v in samples.items() if len(v) >= 3]
    if len(good) < 3 * len(poly_terms(deg)):
        raise RuntimeError(f"only {len(good)}/{len(pts)} blocks registered — the "
                           f"frames do not carry a measurable noise field")
    P = np.array([p for p, _ in good], float); D = np.array([d for _, d in good], float)
    rn = (P[:, 0] - (H - 1) / 2) / ((H - 1) / 2); cn = (P[:, 1] - (W - 1) / 2) / ((W - 1) / 2)
    A = _basis(rn, cn, deg)
    coefs, rms = [], []
    for k in range(2):
        y = D[:, k]; w = np.ones_like(y)
        for _ in range(4):                         # IRLS, Huber-like
            co, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
            res = y - A @ co
            sc = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-9
            w = np.clip(1.5 * sc / np.maximum(np.abs(res), 1e-9), 0, 1) ** 0.5
        coefs.append(co); rms.append(float(res.std()))
    coefs = np.array(coefs)
    idx0 = poly_terms(deg).index((0, 0))
    coefs[0, idx0] += dr0; coefs[1, idx0] += dc0          # self-contained field
    # validation on frames the fit never saw
    warp = DetectorWarp(coefs, deg, (H, W), mode='full')
    g = warp.sampling_grid(device=dev)
    mask = torch.zeros((H, W), dtype=torch.bool, device=dev); mask[150:-150, 150:-150] = True

    def corr(x, y):
        x = x - x.mean(); y = y - y.mean()
        return float((x * y).sum() / (x.norm() * y.norm() + 1e-12))
    held = []
    for a, u in test:
        li = line_integral(a); uw = load(u)
        warped = torch.nn.functional.grid_sample(li[None, None], g, mode='bilinear',
                                                 padding_mode='border', align_corners=True)[0, 0]
        held.append((corr(_highpass(li)[mask], _highpass(uw)[mask]),
                     corr(_highpass(warped)[mask], _highpass(uw)[mask])))
    rec = {'coefs': coefs, 'deg': deg, 'shape': np.array([H, W]), 'dr0': 0.0, 'dc0': 0.0,
           'blocks_used': len(good), 'blocks_total': len(pts), 'fit_rms_px': rms,
           'heldout_corr_raw': [h[0] for h in held], 'heldout_corr_warped': [h[1] for h in held],
           'elapsed_s': time.time() - t0}
    if verbose:
        mean, mx = warp.magnitude()
        print(f"  field: {len(good)}/{len(pts)} blocks, fit rms {rms[0]:.2f}/{rms[1]:.2f} px, "
              f"|d| mean {mean:.1f} px max {mx:.1f} px; held-out noise correlation "
              f"{np.mean([h[0] for h in held]):.4f} -> {np.mean([h[1] for h in held]):.4f} "
              f"({rec['elapsed_s']:.0f}s)")
    if np.mean([h[1] for h in held]) < 0.2:
        raise RuntimeError("the fitted field does not register the held-out "
                           "frames (noise correlation stays below 0.2) — not stored")
    return rec


def measure_and_store(scan_folder, xml_header=None, *, out_dir=None,
                      verbose: bool = True) -> DetectorWarp:
    """Measure the field from a scan with paired frames and store it as the
    calibration of that scan's detector epoch."""
    scan_folder = Path(scan_folder)
    serial, cor, rfan = _scan_epoch(scan_folder)
    if not serial:
        raise RuntimeError(f"no detector serial in {scan_folder}")
    rec = measure_field(scan_folder, verbose=verbose)
    warp = DetectorWarp(rec['coefs'], rec['deg'], tuple(int(v) for v in rec['shape']),
                        mode='full', detector_serial=serial, cor_xml=cor, rfan=rfan,
                        scans=(scan_folder.name,), measured_on=date.today().isoformat(),
                        note=f"measured from {scan_folder.name}; held-out noise corr "
                             f"{np.mean(rec['heldout_corr_warped']):.3f}")
    out = calibration_path(serial, cor, out_dir)
    warp.save(out)
    warp.source = str(out)
    if verbose:
        print(f"  Detector warp: stored {out.name} for detector {serial}, epoch "
              f"CoR {cor} — every later scan of this epoch reuses it.")
    return warp


def record_null_epoch(scan_folder, *, note: str = "", out_dir=None) -> Path:
    """Store that a detector epoch needs NO unwarp (raw frames verified)."""
    scan_folder = Path(scan_folder)
    serial, cor, rfan = _scan_epoch(scan_folder)
    from .vff_io import read_vff_header
    frame = next(iter(sorted(scan_folder.glob('acq-*.vff'))), None)
    if frame is None or not serial:
        raise RuntimeError(f"no acquisition frames / serial in {scan_folder}")
    h = read_vff_header(str(frame), verbose=False)
    W, H = (int(v) for v in h['size'].split()[:2])
    w = DetectorWarp(None, 1, (H, W), mode='none', detector_serial=serial, cor_xml=cor,
                     rfan=rfan, scans=(scan_folder.name,), measured_on=date.today().isoformat(),
                     note=note)
    return w.save(calibration_path(serial, cor, out_dir))


# ------------------------------------------------------------------ export ---

def export_calibration(fit_paths, *, serial: str, cor_xml: float | None,
                       rfan: float | None = None, scans=(), out_dir=None,
                       verbose: bool = True) -> Path:
    """Average fitted fields (``uwarp_field_*.npz`` from the fitting script)
    into one epoch-keyed calibration file.

    Each fit stores a polynomial residual on top of a global (dr0, dc0)
    shift; the shift is folded into the constant term so the stored field is
    self-contained. The disagreement between the input fits is printed: it
    is the calibration's own uncertainty.
    """
    fits = [np.load(str(f)) for f in fit_paths]
    deg = int(fits[0]["deg"])
    H, W = [int(v) for v in fits[0]["shape"]]
    idx0 = poly_terms(deg).index((0, 0))
    stacked = []
    for f, name in zip(fits, fit_paths):
        if int(f["deg"]) != deg or tuple(int(v) for v in f["shape"]) != (H, W):
            raise ValueError(f"{name}: degree/shape mismatch")
        c = np.array(f["coefs"], dtype=np.float64, copy=True)
        c[0][idx0] += float(f["dr0"]) if "dr0" in f else 0.0
        c[1][idx0] += float(f["dc0"]) if "dc0" in f else 0.0
        stacked.append(c)
    stacked = np.stack(stacked)
    coefs = stacked.mean(axis=0)
    rr, cc = np.meshgrid(np.linspace(0, H - 1, 64), np.linspace(0, W - 1, 64),
                         indexing="ij")
    rn = (rr - (H - 1) / 2) / ((H - 1) / 2)
    cn = (cc - (W - 1) / 2) / ((W - 1) / 2)
    B = _basis(rn, cn, deg)
    if verbose:
        print(f"=== {len(fits)} fit(s), degree {deg}, raw detector {H}x{W} ===")
        if len(fits) > 1:
            spread = np.stack([np.hypot(B @ (s[0] - coefs[0]), B @ (s[1] - coefs[1]))
                               for s in stacked])
            print(f"  inter-fit disagreement: mean {spread.mean():.3f} px, "
                  f"max {spread.max():.3f} px")
    warp = DetectorWarp(coefs, deg, (H, W), mode="full", detector_serial=serial,
                        cor_xml=cor_xml, rfan=rfan, scans=tuple(scans),
                        measured_on=date.today().isoformat())
    out = calibration_path(serial, cor_xml, out_dir)
    warp.save(out)
    if verbose:
        print(f"  full      : {warp.summary()}")
        print(f"  nonaffine : {DetectorWarp(coefs, deg, (H, W), mode='nonaffine').summary()}")
        print(f"[saved] {out}")
    return out


def _scan_epoch(scan_folder):
    """(serial, CentreOfRotation, rfan) read from a scan folder."""
    import xmltodict
    scan_folder = Path(scan_folder)
    xml = xmltodict.parse((scan_folder / 'scan.xml').read_text())
    cor = xml_centre_of_rotation(xml)
    rfan = None
    p = scan_folder / 'full.rfan'
    if p.exists():
        try:
            rfan = float(p.read_text().strip())
        except ValueError:
            rfan = None
    return detector_serial_from_scan(scan_folder), cor, rfan


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog='python -m explore_ct120_recon.ct_core.detector_warp',
        description='Publish or inspect detector unwarp calibrations.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    ex = sub.add_parser('export', help='average fitted fields into an '
                                       'epoch-keyed calibration')
    ex.add_argument('--fits', nargs='+', required=True,
                    help='uwarp_field_<scan>.npz files from the fitting script')
    ex.add_argument('--scan-folder', required=True,
                    help='a scan of the epoch the fits were made on (serial, '
                         'CentreOfRotation and full.rfan are read from it)')
    ex.add_argument('--out-dir', default=None)
    sh = sub.add_parser('show', help='list the calibrations for a scan\'s detector')
    sh.add_argument('scan_folder')
    me = sub.add_parser('measure', help='measure the field from a scan that has '
                                        'Corrected/uwarp-* frames and store it '
                                        'for that detector epoch')
    me.add_argument('scan_folder')
    me.add_argument('--out-dir', default=None)
    no = sub.add_parser('none', help='record that this scan\'s detector epoch '
                                     'needs no unwarp (raw frames verified)')
    no.add_argument('scan_folder')
    no.add_argument('--note', default='')
    no.add_argument('--out-dir', default=None)
    args = ap.parse_args(argv)
    if args.cmd == 'measure':
        measure_and_store(args.scan_folder, out_dir=args.out_dir)
        return
    if args.cmd == 'none':
        print(f"[saved] {record_null_epoch(args.scan_folder, note=args.note, out_dir=args.out_dir)}")
        return
    if args.cmd == 'export':
        serial, cor, rfan = _scan_epoch(args.scan_folder)
        if not serial:
            raise SystemExit(f"no detector serial in {args.scan_folder}")
        scans = tuple(re.sub(r'^uwarp_field_|\.npz$', '', Path(f).name)
                      for f in args.fits)
        export_calibration(args.fits, serial=serial, cor_xml=cor, rfan=rfan,
                           scans=scans, out_dir=args.out_dir)
    else:
        serial, cor, rfan = _scan_epoch(args.scan_folder)
        print(f"detector {serial}, scan CentreOfRotation {cor}, rfan {rfan}")
        for w in find_calibrations(serial or ''):
            print(f"  {'SAME EPOCH' if w.same_epoch(cor) else 'other epoch'}: "
                  f"{w.summary()}  scans={list(w.scans)} measured {w.measured_on}")
        if not any(w.same_epoch(cor) for w in find_calibrations(serial or '')):
            print(f"  this epoch is UNKNOWN; paired Corrected/uwarp frames "
                  f"{'present' if has_paired_frames(args.scan_folder) else 'absent'}")


if __name__ == '__main__':
    main()
