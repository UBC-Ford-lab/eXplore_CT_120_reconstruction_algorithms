# eXplore CT 120 Reconstruction

[![PyPI](https://img.shields.io/pypi/v/eXplore_CT_120_reconstruction_algorithms.svg)](https://pypi.org/project/eXplore-CT-120-reconstruction-algorithms/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Cone-beam CT reconstruction for the GE eXplore CT 120 micro-CT scanner.
Input is a scan folder as the scanner writes it, output is an HU-calibrated
`.vff` volume.

![Cone-beam reconstruction: a ray from the source through the volume to a detector pixel](https://raw.githubusercontent.com/UBC-Ford-lab/eXplore_CT_120_reconstruction_algorithms/main/assets/voxel_model.png)

*Each detector pixel measures a line integral of the attenuation μ along a
ray from the source. Reconstruction recovers the volume from those
measurements.*

| Family | Algorithms | Needs |
|---|---|---|
| Analytic (`fdk/`) | FDK with ramp, Shepp-Logan, cosine or Hamming window, Parker weighting | PyTorch |
| Classical iterative (`iterative/`) | ASTRA: SIRT, CGLS, SART. TIGRE: OS-SART, SART, SIRT, MLEM, optionally with TV | CUDA, ASTRA or TIGRE |
| Learning-based (`learning_based_iterative/`) | Differentiable projector and gradient descent. `--algorithm voxel` (dense grid, default), `--algorithm gaussian` (3-D Gaussian splatting), `--algorithm gaussian_dynamic` (Gaussian splatting with a deformation field, for gated scans of breathing animals) | CUDA, plus the rasteriser for the two Gaussian algorithms |

The three families share scan loading, preprocessing, geometry calibration,
HU calibration and output format, so their volumes can be compared directly.

## Install

```bash
pip install eXplore_CT_120_reconstruction_algorithms
# or the development version
pip install git+https://github.com/UBC-Ford-lab/eXplore_CT_120_reconstruction_algorithms.git
```

Optional:

```bash
pip install astra-toolbox                                               # ASTRA
pip install "git+https://github.com/CERN/TIGRE.git#subdirectory=Python"  # TIGRE, not on PyPI
pip install wandb                                                       # run logging
./install_rasteriser.sh                                                 # X-ray Gaussian rasteriser for the Gaussian algorithms
```

The rasteriser is the one from [r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian)
(research licence, not on PyPI). The install script clones it at a pinned
commit, builds it and stamps the build with the commit hash so a run's log
records which build it used. Upstream has no releases, and the calibrations
in the Gaussian backend were measured against that commit, so keep the pin.
`./install_rasteriser.sh --help` lists the options, including `--wheel DIR`
for building on a cluster node.

Requires Python 3.10 or newer and PyTorch 2.0 or newer. The import name is
`explore_ct120_recon`.

## Usage

```bash
ct120-fdk data/scans/Scan_1988
ct120-iterative data/scans/Scan_1988 --backend astra --algorithm SIRT3D_CUDA --iterations 100
ct120-iterative data/scans/Scan_1988 --backend tigre --algorithm ossart --tv-lambda 10
ct120-learned data/scans/Scan_1988 --downsample 3
ct120-learned data/scans/Scan_1988 --algorithm gaussian
ct120-learned data/scans/Scan_1510 --algorithm gaussian_dynamic
ct120-volume-report VOLUME.vff
ct120-bilateral-filter VOLUME.vff [SD]
ct120-projection-report data/scans/Scan_1988 --volume a.vff --volume b.vff
ct120-geometry-calibration data/scans/Scan_1988
ct120-noise-calibration data/scans/Scan_1988
ct120-simulate-scan VOLUME.vff --template data/scans/Scan_1988 --output data/scans/Sim_1988
ct120-motion-show RESULT_DIR
ct120-export-4d RESULT_DIR/recon_4d.nrrd --format nrrd-slicer
ct120-ventilation RESULT_DIR
ct120-breathing-phantom build REST.vff --template data/scans/Scan_1510 --output data/scans/Phantom_1510
```

`python -m explore_ct120_recon.run_fdk_recon ...` does the same as the
console script. `--help` on any command lists its flags.

A scan folder holds the projection `.vff` files, `scan.xml` and the
`bright.vff` / `dark.vff` flat-field frames. Each run writes a `.vff` volume
next to the scan folder, a `.json` sidecar with the grid and the fitted HU
map, and a `_plots/` folder with slices, histogram and projection
diagnostics.

## Pipeline

```
raw projections → flat-field + log → air normalization + ring correction
  → detector rotation calibration (measured once per scan, cached)
  → FDK | ASTRA / TIGRE | learned
  → μ (mm⁻¹) → HU calibration → .vff + .json + plots
```

Defaults worth knowing about. Each has an override flag.

- Field of view and ROI are `auto`. The grid is what the detector subtends,
  tightened to the object. The export region is the scanner's own
  `SubVolumeCoordinates.xml` box, so results line up with the vendor volume.
- The model domain is `auto` for the iterative and learned families. They
  reconstruct everything a ray crosses, bed included. Otherwise the missing
  attenuation shows up as a smooth HU bias. `--roi` only crops the output.
- HU calibration fits two anchors from the volume's own histogram: air at
  −1000 HU and bulk soft tissue at `--tissue-hu` (default +120, use 0 for a
  water phantom). No phantom scan is needed.
- Preflight estimates VRAM and RAM before allocating anything.
  `--preflight-only` prints the estimate and exits.
- Weights & Biases logging is on when `--wandb-project` or `$WANDB_PROJECT`
  names a project. Nothing is uploaded otherwise.

## Learning-based reconstruction

Fits a volume model to the line integrals through a differentiable ray
tracer, using Adam with non-negativity, LR warmup, plateau reduction and a
held-out projection for early stopping. Main flags: `--loss`, `--iterations`,
`--rays-per-batch` (sized from free VRAM by default), `--emulate-sart`,
`--compile`.

All representations use the same loss registry
(`learning_based_iterative/losses`), so a loss value means the same thing for
a voxel grid, a network or a Gaussian cloud. The default loss `wls` weights
each ray by the inverse variance of its measured counts,
`Var(p) = A/count + (sigma_r/ds)^2/count^2` (`ct_core/noise_model.py`). The
Poisson slope `A` is measured from consecutive views of the scan on every
run. The read noise `sigma_r` is measured when the scan's count range allows
it, otherwise taken from the detector calibration file, otherwise from a
package default. `--wls-slope` and `--wls-read-noise` pin either value, and
the run log records what was used under `wls/*`. `--loss mse` is the
objective classical SIRT minimises and is the right choice when comparing
against the classical solvers.

To add a representation, subclass `LearnedReconstructor` and implement
`build_model`, `build_domain` and `export_volume`, then register a
`LearnedAlgorithm` descriptor. `voxel/` is the short example, `gaussian/` the
full one with its own training loop, FDK seeding and shape constraints. A
descriptor can change the shared defaults for its representation through
`driver_defaults`; `gaussian` for instance runs at `--downsample 3` with an
SSIM stopper. A representation in another package is loaded with
`--algorithm-module my_package.algorithms`, and may subclass
`GaussianReconstructor` through its hooks (`_render_train`, `_render_eval`,
`_param_groups`, `_regularise`, `_state`, `_restore_state`,
`_checkpoint_payload`).

### Dynamic Gaussian splatting (`--algorithm gaussian_dynamic`)

In a gated step-and-shoot scan of a breathing animal every projection sees a
slightly different state, and the two gates differ by a full breath.
`gaussian_dynamic` (`learning_based_iterative/dynamic_gaussian/`) keeps the
static cloud as the mean anatomy and gives every projection its own state:
a fixed column for the gate plus a small learned code, decoded through a
low-rank displacement basis on a control grid (1 mm by default). The
displacement transports each primitive's shape by the deformation gradient
and divides its amplitude by |det F|, so mass is conserved and compressed
tissue gets denser. The divergence of the field is restricted to cells
containing air, read from the model's own rest state, so solid tissue can
move but keeps its volume. Three motion parts with learned per-primitive
membership let the lung slide along the chest wall. Everything is estimated
from the projections together with the volume. No image registration is
involved.

The run loads all gated phases (`--phase all` is the default for this
algorithm, with a 32000-iteration cap under the usual held-out stopper) and
writes, besides the gate-0 mean state in `recon.vff`:

- `recon_4d.nrrd` and `.json`: the breathing sequence, nine states from
  gate 0 to gate 1 on the output grid. `--dyn-export-4d-format` selects
  `nrrd`, `nrrd-slicer` or `nifti`; `ct120-export-4d` converts between them.
- `recon_ventilation.nrrd`, `_J.nrrd`, `.json`, `.npz`, `.png`: regional
  specific ventilation (J − 1)/f_air per voxel at 0.5 mm in the expiration
  frame. The sidecar compares the field's volume change against the air
  change in the rendered states. `ct120-ventilation` re-exports at another
  pitch.
- `motion/*` panels in the plots folder and in W&B: codes, field, parts,
  gates and a breathing GIF. `ct120-motion-show` redraws them from the
  checkpoint.

`ct120-learned --algorithm gaussian_dynamic --help` lists the `--dyn-*`
flags. `ct120-breathing-phantom` builds a gated scan of a breathing digital
phantom from any volume written by this package, with a known displacement
field and ventilation, using the scanner's ray geometry and noise model via
`ct120-simulate-scan`, and scores a reconstruction against that truth.

## Report tools

`ct120-volume-report` draws slices, the HU histogram and the fitted anchors
for a finished volume, including vendor volumes. `ct120-bilateral-filter` is
a port of the lab's MATLAB `filter_vff.m`: `imbilatfilt` on every axial slice
with its defaults (spatial sigma 1 voxel, 5 x 5 window, replicate padding)
and `degreeOfSmoothing = 2*SD^2`, with SD measured from the patch
`im(20:40,160:180,100)` when not given. It writes `<volume>_imbilatfilt.vff`.
`ct120-projection-report` forward-projects one or more volumes and scores
them against the measured projections on the same angles and detector window.
`ct120-geometry-calibration` measures the detector rotation ahead of time,
for example on a GPU node before submitting long jobs.
`ct120-noise-calibration` measures the detector's Poisson slope and read
noise from consecutive views of a scan and stores the read noise per
detector. A low-attenuation scan such as a mouse cannot separate the read
noise term from the Poisson term, so run it once per detector on a dense
phantom and the `wls` loss will pick the value up from there. Use `--vendor`
with the report tools for a GE volume; it applies the vendor's axis
conventions and locates the ROI from the scan.

Everything is importable:

```python
from explore_ct120_recon.ct_core.vff_io import read_vff, write_vff
from explore_ct120_recon.learning_based_iterative import LearnedReconstructor
```

## Layout

```
src/explore_ct120_recon/
  run_*.py                      drivers, one per family plus the report tools
  ct_core/                      scan loading, preprocessing, geometry and HU
                                calibration, noise model, preflight, VFF I/O
  fdk/                          filtered backprojection
  iterative/{astra,tigre}/      toolbox wrappers
  simulate/                     forward projection of a volume into a scan folder
  learning_based_iterative/     trainer, ray sampler, renderer, losses
    voxel/                      dense grid
    gaussian/                   Gaussian splatting
    dynamic_gaussian/           Gaussian splatting with a deformation field
```

Detector calibrations measured from a scan are cached in `data/calibration/`
when running from a checkout, in the per-user data directory
(`~/.local/share/explore_ct120_recon/calibration` on Linux) for an installed
copy, or in `$CT_CALIBRATION_DIR` if set.

## License

MIT, see [LICENSE](LICENSE).
