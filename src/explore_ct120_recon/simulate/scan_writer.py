"""A scan folder in the dialect the loaders read, from arrays.

The loaders (``ct_core.scan_setup.load_scan_data`` / ``vff_io.VFFDataset``)
take a folder holding ``scan.xml``, ``bright.vff``, ``dark.vff`` and one
``acq-<phase>-<view>.vff`` frame per view and phase, each frame carrying its
``gantryPosition`` (where the angles come from) and ``serialNumber`` (which
keys the detector calibrations). This module writes exactly that, starting
from a REAL scan's header so every field the loaders do not care about is
still there and plausible, and the fields they do care about are set from
what was simulated.

Angles: ``VFFDataset`` turns gantry positions into view angles as
``(AngleOffset + 120 + gantry[0] - gantry[i]) mod 360``. ``view_angles_deg``
is that formula, so the simulator and the loader agree by construction.
"""
from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import xmltodict

from ..ct_core.vff_io import read_vff, write_vff

FRAME_PATTERN = "acq-{phase}-{view:04d}.vff"


def template_header(scan_folder) -> dict:
    """The parsed ``scan.xml`` of a real scan, plus its raw detector shape and
    bright/dark fields: ``{'xml', 'raw_detector_shape', 'bright', 'dark'}``."""
    folder = Path(scan_folder)
    xml = xmltodict.parse((folder / 'scan.xml').read_text())
    _, bright = read_vff(str(folder / 'bright.vff'), verbose=False)
    _, dark = read_vff(str(folder / 'dark.vff'), verbose=False)
    bright = np.asarray(bright, dtype=np.float32).squeeze(0)
    dark = np.asarray(dark, dtype=np.float32).squeeze(0)
    return {'xml': xml, 'raw_detector_shape': tuple(int(v) for v in bright.shape),
            'bright': bright, 'dark': dark}


def view_angles_deg(gantry_deg: np.ndarray, angle_offset_deg: float) -> np.ndarray:
    """The loader's angle for each gantry position (see module docstring)."""
    g = np.asarray(gantry_deg, dtype=np.float64)
    return (float(angle_offset_deg) + 120.0 + g[0] - g) % 360.0


def gantry_positions(n_views: int, step_deg: float, start_deg: float = 0.0) -> np.ndarray:
    """Gantry positions DECREASE during acquisition on this scanner; ``n_views``
    of them ``step_deg`` apart, so the loader's angles increase by the step."""
    return start_deg - np.arange(int(n_views), dtype=np.float64) * float(step_deg)


def scan_xml(template_xml: dict, *, n_views: int, step_deg: float,
             detector_pixel_mm: float, detector_shape, phases: int,
             specimen: str = 'SIMULATED', angle_offset_deg=None) -> str:
    """``scan.xml`` text: the template with the simulated acquisition's values.

    ``detector_shape`` is ``(n_b, n_a)`` of the frames as written (already
    binned, if they are); the centre-of-rotation fields are set to that
    panel's geometric centre, which is also the package's default policy.
    """
    doc = copy.deepcopy(template_xml)
    s = doc['Series']
    sp = s['SeriesParams']
    n_b, n_a = (int(v) for v in detector_shape)
    sp['ViewCount'] = str(int(n_views))
    sp['IncrementAngle'] = f"{float(step_deg):.6f}"
    sp['ScanTechnique'] = 'HalfScan' if n_views * step_deg < 350 else 'FullScan'
    if int(phases) > 1:
        sp['Gates'] = {'@period': '16', '@phases': str(int(phases)),
                       'Gate': [{'@unit': 'ms', '@offset': str(10 + 340 * k)} for k in range(int(phases))]}
    else:
        sp.pop('Gates', None)
    s['Specimen'] = str(specimen)
    s['DetectorSpacing'] = f"{float(detector_pixel_mm):.6f}"
    s['CentreOfRotation'] = f"{(n_a - 1) / 2.0:.6f}"
    s['CentralSlice'] = f"{(n_b - 1) / 2.0:.6f}"
    if angle_offset_deg is not None:
        s['AngleOffset'] = f"{float(angle_offset_deg):.6f}"
    return xmltodict.unparse(doc, pretty=True)


def write_projection_frame(path, counts: np.ndarray, gantry_deg: float,
                           serial: str = 'SIMULATED') -> str:
    """One acquisition frame (``n_b``, ``n_a``) of integer counts in the GE
    ``ncaa`` dialect: big-endian int16, ``gantryPosition`` and ``serialNumber``
    in the header, which ``write_vff`` (a VOLUME writer) does not carry."""
    arr = np.asarray(counts)
    if arr.ndim != 2:
        raise ValueError(f"a frame is 2-D (n_b, n_a), got {arr.shape}")
    n_b, n_a = arr.shape
    lim = np.iinfo(np.int16)
    payload = np.rint(np.clip(arr, lim.min, lim.max)).astype('>i2')
    head = "\n".join([
        "ncaa", "rank=3;", "type=raster;", "modality=CT;",
        f"size={n_a} {n_b} 1;", "bands=1;", "bits=16;", "format=slice;",
        f"gantryPosition={float(gantry_deg):.6f};", f"serialNumber={serial};",
    ]) + "\n\f\n"
    with open(path, 'wb') as f:
        f.write(head.encode('latin-1'))
        payload.tofile(f)
    return str(path)


def write_calibration_fields(folder, bright: np.ndarray, dark: np.ndarray) -> None:
    write_vff(str(Path(folder) / 'bright.vff'), {'bits': 16},
              np.asarray(bright, dtype=np.float32)[None], verbose=False)
    write_vff(str(Path(folder) / 'dark.vff'), {'bits': 16},
              np.asarray(dark, dtype=np.float32)[None], verbose=False)


def write_scan_folder(folder, *, template_xml: dict, frames: Iterable,
                      gantry_deg: np.ndarray, step_deg: float,
                      detector_pixel_mm: float, bright: np.ndarray, dark: np.ndarray,
                      phases: int = 1, serial: str = 'SIMULATED',
                      specimen: str = 'SIMULATED', roi_xml_from=None,
                      angle_offset_deg=None, log=print) -> Path:
    """Write a complete, loadable scan folder and return it.

    ``frames`` yields ``(phase_index, view_index, counts)`` in any order; a
    frame is written as soon as it arrives, so a caller can simulate and
    write one view at a time without holding the scan in memory. ``bright``
    / ``dark`` are the fields at the frames' shape. ``roi_xml_from`` copies
    a real scan's ``Volumes/SubVolumeCoordinates.xml`` so ``--roi auto``
    delivers the same box on the synthetic scan as on the animal it was
    made from.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    gantry = np.asarray(gantry_deg, dtype=np.float64)
    bright = np.asarray(bright, dtype=np.float32)
    dark = np.asarray(dark, dtype=np.float32)
    if bright.shape != dark.shape:
        raise ValueError(f"bright {bright.shape} and dark {dark.shape} differ")
    (folder / 'scan.xml').write_text(scan_xml(
        template_xml, n_views=len(gantry), step_deg=step_deg,
        detector_pixel_mm=detector_pixel_mm, detector_shape=bright.shape,
        phases=phases, specimen=specimen, angle_offset_deg=angle_offset_deg))
    write_calibration_fields(folder, bright, dark)
    if roi_xml_from is not None:
        src = Path(roi_xml_from) / 'Volumes' / 'SubVolumeCoordinates.xml'
        if src.exists():
            (folder / 'Volumes').mkdir(exist_ok=True)
            shutil.copy(src, folder / 'Volumes' / 'SubVolumeCoordinates.xml')
    n_written = 0
    for phase, view, counts in frames:
        counts = np.asarray(counts)
        if counts.shape != bright.shape:
            raise ValueError(f"frame {counts.shape} does not match the fields {bright.shape}")
        name = FRAME_PATTERN.format(phase=f"{int(phase):02d}", view=int(view))
        write_projection_frame(folder / name, counts, gantry[int(view)], serial)
        n_written += 1
    log(f"  wrote {n_written} frames of {bright.shape[0]}x{bright.shape[1]} to {folder}")
    return folder


def remove_frames(folder) -> int:
    """Delete every ``acq-*.vff`` in ``folder`` (a rebuild), return the count."""
    n = 0
    for p in Path(folder).glob('acq-*.vff'):
        os.remove(p)
        n += 1
    return n
