"""Compatibility shim: the detector warp now lives in ``ct_core.detector_warp``
so every backend shares one calibration and one lookup. The ray-domain
application (``ideal_indices`` consumed by ``ray_sampler``) is unchanged."""
from ..ct_core.detector_warp import (  # noqa: F401
    DetectorWarp, _basis, _moment, detector_serial_from_scan, poly_terms,
    resolve_detector_warp, strip_affine,
)
from ..ct_core.paths import calibration_dir

CALIB_DIR = calibration_dir()

__all__ = ['CALIB_DIR', 'DetectorWarp', 'detector_serial_from_scan', 'poly_terms',
           'resolve_detector_warp', 'strip_affine']
