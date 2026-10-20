"""Cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner.

Three algorithm families behind one shared pipeline (scan loading,
preprocessing, geometry and HU calibration, VFF export):

* ``fdk``                        analytic filtered backprojection
* ``iterative``                  classical iterative (ASTRA, TIGRE)
* ``learning_based_iterative``   differentiable projector + gradient descent

Command-line drivers are installed as ``ct120-fdk``, ``ct120-iterative``,
``ct120-learned``, ``ct120-volume-report``, ``ct120-projection-report``,
``ct120-geometry-calibration``, ``ct120-noise-calibration``,
``ct120-bilateral-filter`` and ``ct120-simulate-scan``; the same entry points
are importable as ``explore_ct120_recon.run_*``. The time-aware Gaussian
backend adds ``ct120-motion-show``, ``ct120-export-4d``, ``ct120-ventilation``
and ``ct120-breathing-phantom`` (``learning_based_iterative.dynamic_gaussian``).
``ct120-gui`` (the ``gui`` extra) serves a web page that runs any of them
as jobs on the machine it is started on.

Importing this package pulls in only ``ct_core`` (VFF I/O, preprocessing,
calibration). The backends import their optional toolboxes lazily.
"""

__version__ = "0.2.4"

from . import ct_core
from .ct_core import vff_io
from .ct_core.errors import (
    ConfigError,
    PreflightAbort,
    ReconstructionError,
    ScanDataError,
)
from .ct_core.vff_io import (
    VFFDataset,
    read_vff,
    read_vff_data,
    read_vff_header,
    write_vff,
)

__all__ = [
    "__version__",
    "ct_core",
    "vff_io",
    "ConfigError",
    "PreflightAbort",
    "ReconstructionError",
    "ScanDataError",
    "VFFDataset",
    "read_vff",
    "read_vff_data",
    "read_vff_header",
    "write_vff",
]
