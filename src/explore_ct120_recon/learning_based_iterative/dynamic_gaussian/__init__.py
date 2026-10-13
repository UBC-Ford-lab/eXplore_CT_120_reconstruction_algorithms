"""Time-aware Gaussian splatting: the static backend plus a per-view state.

``gaussian`` ships the static representation (cloud, rasteriser binding,
seeding, the per-view training loop). This package subclasses it through the
seams that loop declares and adds the one thing a gated live-animal scan
needs and a static model cannot have: a state per projection.

    motion.py         MotionSupport, ControlGridBasis, DisplacedCloud,
                      MotionModel — the deformation (with the Jacobian shape
                      transport and the mass-preserving amplitude), the
                      gated-phase contrast column, the priors, the
                      air-sourced divergence and the motion PARTS (K fields
                      with a learned per-primitive membership, for sliding
                      interfaces)
    panels.py         the code-trajectory, displacement-field and parts figures
    reconstructor.py  DynamicGaussianReconstructor — the seams filled in
    algorithm.py      ``--algorithm gaussian_dynamic``: the static flags plus
                      ``--dyn-*``, the static driver defaults with every gated
                      phase loaded, the static footprint
    export4d.py       the breathing sequence as a 4-D NRRD / NIfTI
                      (``ct120-export-4d`` converts between the layouts)
    ventilation.py    the regional ventilation map read from the field and
                      its checks (``ct120-ventilation`` re-exports from a run)
    show.py           the gate states, the field and the breathing GIF of a
                      finished run (``ct120-motion-show``)
    phantom.py        a breathing digital phantom with a known ventilation,
                      and the scorer (``ct120-breathing-phantom``)

``learning_based_iterative`` registers it, so it is selectable like the
static one::

    ct120-learned data/scans/Scan_XXXX --phase all --algorithm gaussian_dynamic

(``--phase all`` is the algorithm's own default). Nothing here is a second
implementation of anything in ``gaussian``: the rasteriser, the cloud and the
loop are imported, not copied.
"""
from .algorithm import ALGORITHM  # noqa: F401  (registered by learning_based_iterative)
from .motion import ControlGridBasis, DisplacedCloud, MotionModel, MotionSupport
from .reconstructor import DynamicGaussianReconstructor

__all__ = ['ALGORITHM', 'DynamicGaussianReconstructor', 'MotionModel',
           'MotionSupport', 'ControlGridBasis', 'DisplacedCloud']
