"""Synthetic scans with a known truth.

Everything a reconstruction is compared against on a real animal is another
estimate: the vendor's volume, a registration between two gated images, a
density rule. A simulated scan is the one case where the object, its motion
and the noise are known exactly, because they were made here — so an error
in the reconstruction is measured against the truth, not against another
method's error.

The package owns the two halves that are the SCANNER's:

* ``forward`` — the line integrals of a dense attenuation volume through the
  package's own ray geometry (the same ``render_rays`` every learned backend
  trains on, the same ``build_scan_geometry`` every driver builds), and the
  detector's counts from them under the measured noise model
  (``ct_core.noise_model``: Poisson slope at the binning, read noise per raw
  pixel).
* ``scan_writer`` — a scan FOLDER in the GE dialect the loaders read:
  ``scan.xml``, ``bright.vff`` / ``dark.vff`` and ``acq-<phase>-<view>.vff``
  frames carrying ``gantryPosition``. A driver run on it needs no flag to say
  it is synthetic; that is the point.

What the OBJECT is — a sphere, a reconstructed animal, a breathing phantom —
is the caller's. ``simulate_scan`` takes a ``VolumeSource`` (a function of
the view index and phase returning the attenuation volume at that moment) so
a moving object costs the caller one function, not a fork of this package.

The forward projector is the trilinear interpolation of the voxel grid along
the same quadrature the reconstructions integrate. A voxel phantom rendered
by a ray marcher and reconstructed by a Gaussian mixture or an FDK is not an
inverse crime — the representations differ — but the ray geometry is shared,
so a geometry bug would cancel. Use ``--psi-deg`` / ``--cor-offset`` to put a
KNOWN misalignment in and check that the calibrators find it.
"""
from .forward import (VolumeSampler, counts_from_line_integrals, project_view,
                      scene_for_simulation, simulate_scan)
from .scan_writer import (write_projection_frame, write_scan_folder,
                          template_header)

__all__ = ['VolumeSampler', 'counts_from_line_integrals', 'project_view',
           'scene_for_simulation', 'simulate_scan', 'write_projection_frame',
           'write_scan_folder', 'template_header']
