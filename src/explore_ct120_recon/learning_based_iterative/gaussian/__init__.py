"""Gaussian splatting for cone-beam CT, as a pipeline backend.

Optional native dependency, on the same footing as TIGRE and ASTRA: the
compiled ``xray_gaussian_rasterization_voxelization`` extension. Importing this
package never needs it — only running a reconstruction does, and its absence
raises a directed ImportError rather than failing at import time.

``learning_based_iterative`` registers it as ``--algorithm gaussian``::

    python -m explore_ct120_recon.run_learned_recon data/scans/Scan_XXXX \
        --algorithm gaussian --voxel-xy 0.075 --voxel-z 0.075

Everything scan-specific is read from the pipeline's own ``ScanContext``:
geometry, centre-of-rotation policy, detector psi, the measured model domain,
preprocessing, HU calibration, the export ROI and the W&B panels. There is no
intermediate dataset format and no per-scan configuration file.

    kernel.py   the only place the third-party CUDA rasteriser is touched,
                and the alpha-cutoff correction that makes it usable on
                real scanner data
    camera.py   geometry dict -> rasteriser matrices (COR, psi, principal
                point), for any scan
    model.py    the cloud: parameters, activations, adaptive density control
    seeding.py  mass-proportional initialisation from a reference volume
    reconstructor.py  the view-based training loop, honouring the driver's
                contract so the shared back half is reused unchanged
"""
from .algorithm import ALGORITHM  # noqa: F401  (registered by learning_based_iterative)
from .reconstructor import GaussianReconstructor  # noqa: F401

__all__ = ['ALGORITHM', 'GaussianReconstructor']
