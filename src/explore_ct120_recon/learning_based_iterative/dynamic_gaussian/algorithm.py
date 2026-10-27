"""``--algorithm gaussian_dynamic``: the submodule's Gaussian splatting plus a
per-view motion model, as a selectable representation.

Every flag, option, driver default and footprint of ``gaussian`` is
inherited by delegation — this descriptor adds only the ``--dyn-*`` flags
the motion model needs — so an A/B against the static run is "the same
command with ``--algorithm gaussian_dynamic``", and nothing about the static
recipe is restated here.
"""
from __future__ import annotations

import argparse
import os

from ...ct_core.preflight import Footprint, MachineRequest
from ..registry import LearnedAlgorithm
from ..gaussian import algorithm as static

from .motion import MotionModel
from .reconstructor import DynamicGaussianReconstructor


def add_args(group) -> None:
    static.add_args(group)
    g = group
    g.add_argument('--dyn-codes', type=int, default=1, dest='dyn_codes',
                   metavar='K',
                   help='dimension of the per-view state code (default: 1). '
                        'One number per independent motion the data has to '
                        'explain on top of the gate column. On Scan_1510 a '
                        'second code fitted only per-view jitter (A/B '
                        'z708w98r vs t9h4nndo: held-out identical, peak '
                        'displacement 1.5 vs 1.9 mm at a 2 mm bound).')
    g.add_argument('--dyn-grid-mm', type=float, default=1.0, dest='dyn_grid_mm',
                   metavar='MM',
                   help='spacing of the deformation control grid (default: '
                        '1 mm since 2026-09-21; 3 mm before). The basis '
                        'fields are trilinear on this grid, so it is the '
                        'resolution of the motion: at 3 mm the lung boundary '
                        'was a 1.5 mm ramp holding 58 %% of the reported '
                        'ventilation map; at 1 mm the regional agreement '
                        'with the vendor density reading rose from 0.61 to '
                        '0.70 and the boundary shells match it to 0.01. '
                        'A fine grid needs --dyn-reg-basis-order 2.')
    g.add_argument('--dyn-support', default='roi', dest='dyn_support',
                   metavar='roi|domain|X0 X1 Y0 Y1 Z0 Z1',
                   help="where motion is allowed (default: roi = the export "
                        "ROI, so the bed and the truncated periphery stay "
                        "static). 'domain' = everywhere; or six mm bounds.")
    g.add_argument('--dyn-support-margin-mm', type=float, default=2.0,
                   dest='dyn_support_margin_mm', metavar='MM',
                   help='margin added to the support box (default: 2 mm).')
    g.add_argument('--dyn-support-taper-mm', type=float, default=2.0,
                   dest='dyn_support_taper_mm', metavar='MM',
                   help='cosine taper of the support weight outside the box '
                        '(default: 2 mm; 0 = hard box).')
    g.add_argument('--dyn-motion-from', type=float, default=0.25,
                   dest='dyn_motion_from', metavar='F',
                   help='iteration from which the motion trains: a fraction '
                        'of --iterations if < 1, else absolute (default: '
                        '0.25). Before it the run IS the static recipe, so '
                        'the rest cloud becomes the mean anatomy first.')
    g.add_argument('--dyn-motion-hold', type=float, default=0.25,
                   dest='dyn_motion_hold', metavar='F',
                   help='for this long after --dyn-motion-from the motion '
                        'groups keep their base LR through the plateau '
                        'reducer\'s cuts (fraction of --iterations if < 1, '
                        'else absolute; default: 0.25; 0 = no exemption). '
                        'MEASURED (run ffap88ti): a cut drawn by the '
                        'converged cloud landed on the fresh motion '
                        'parameters after 1500 iterations and froze the field.')
    g.add_argument('--dyn-basis-init-vox', type=float, default=0.25,
                   dest='dyn_basis_init_vox', metavar='F',
                   help='std of the random initial basis fields, in voxels '
                        '(default: 0.25). The codes start at zero so the '
                        'initial displacement is exactly zero; a zero basis '
                        'as well would leave both without a gradient.')
    g.add_argument('--dyn-transport-shapes', action='store_true',
                   dest='dyn_transport_shapes', default=True,
                   help='transport each primitive\'s covariance by the '
                        'deformation gradient, Sigma\' = F Sigma F^T (the '
                        'deformation-informed form). Default since run '
                        't9h4nndo: ON. Costs a (N, 3, 3) product per render '
                        '(~25 %% of the iteration at 2.4 M primitives).')
    g.add_argument('--dyn-no-transport-shapes', action='store_false',
                   dest='dyn_transport_shapes',
                   help='keep every primitive\'s rest covariance under '
                        'displacement (the pre-t9h4nndo behaviour).')
    g.add_argument('--dyn-mass-preserving', action='store_true',
                   dest='dyn_mass_preserving', default=True,
                   help='divide a primitive\'s amplitude by |det F| as its '
                        'shape is transported, so its MASS '
                        '(a (2 pi)^1.5 |Sigma|^0.5) is invariant and a '
                        'compressing region gets denser instead of merely '
                        'smaller. Needs --dyn-transport-shapes (without it '
                        'the rest shapes already conserve mass). MEASURED on '
                        'z708w98r: with the amplitude held, transport scales '
                        'each primitive\'s mass by |det F| (mass-weighted '
                        'mean |det F - 1| 2.3 %%, p0.1 0.51) and a uniform '
                        'compression renders IDENTICALLY, so the volumetric '
                        'part of the field is invisible to the data. '
                        'DEFAULT ON since 2026-09-19: the A/B (muqojfqg vs '
                        't805g3zw) tied on held-out and the mass-preserving '
                        'field is the physical one (lung volume ratio 0.84 '
                        'vs 0.60, the amplitude-held model deletes 40 %% of '
                        'the lung\'s mass). --dyn-no-mass-preserving restores '
                        'the DIGS/PhysGaussian convention.')
    g.add_argument('--dyn-no-mass-preserving', action='store_false',
                   dest='dyn_mass_preserving',
                   help='keep the amplitude fixed under transport (the '
                        'pre-2026-09-19 behaviour; needed with '
                        '--dyn-no-transport-shapes).')
    g.add_argument('--dyn-det-floor', type=float, default=0.2,
                   dest='dyn_det_floor', metavar='D',
                   help='|det F| is clamped to this before the amplitude is '
                        'divided by it (default: 0.2 — a primitive keeping a '
                        'fifth of its volume), and --dyn-reg-fold starts '
                        'penalising below it. 1/det is singular at 0 and '
                        'changes sign past it.')
    g.add_argument('--dyn-reg-fold', type=float, default=1.0,
                   dest='dyn_reg_fold', metavar='W',
                   help='weight of mean min(log(det F / --dyn-det-floor), 0)^2 '
                        'over the last render (default: 1; 0 = off): the '
                        'folding guard, one-sided so expansion is free. Only '
                        'with --dyn-mass-preserving, so the recipe without it '
                        'is unchanged — MEASURED on z708w98r, whose field '
                        'reached det F = 0.216 with nothing in the loss aware '
                        'of it, a fifth of the way to a sign flip in 1/det.')
    g.add_argument('--dyn-amplitude', action='store_true',
                   dest='dyn_amplitude', default=False,
                   help='learn a per-view gate AMPLITUDE: the fixed gate '
                        'value of view v becomes +-1 + a_v, one scalar per '
                        'view along the breathing axis the gate field '
                        'already describes (needs --phase with >= 2 gates). '
                        'MEASURED on t9h4nndo: the diaphragm dome with a '
                        'random per-view amount is the largest structure the '
                        'free codes leave in the residual. Default: off.')
    g.add_argument('--dyn-reg-amp', type=float, default=1e-2,
                   dest='dyn_reg_amp', metavar='W',
                   help='weight of mean a_v^2 (default: 1e-2); no smoothness '
                        'term, the amount is random per breath.')
    g.add_argument('--dyn-parts', type=int, default=3, dest='dyn_parts',
                   metavar='K',
                   help='number of independent motion PARTS (DEFAULT 3 since '
                        '2026-09-19: lung / heart + abdomen / rest, MEASURED '
                        'on run 90pu264k — soft-tissue NCC vs FDK gate 0 '
                        'recovers 89 %% of the single-gate gap, the lung '
                        'breathes 0.868 vs 0.893 in volume ratio; 1 = one '
                        'field, the model as before). With K > 1 the grid '
                        'carries K basis sets and every primitive learns a '
                        'soft membership over them, so tissues that touch '
                        'but move differently (lung against chest wall and '
                        'heart) can SLIDE instead of being blended by one '
                        'smooth field. Nothing names the interface: it '
                        'forms where the data gain from one, and with no '
                        'gain the K fields converge to one. Needs a '
                        'fixed-size cloud (--gauss-density-control fixed, '
                        'the default; pass --dyn-parts 1 with adaptive or '
                        'MCMC control). Readouts: motion/parts_* incl. the '
                        'interface slip in mm, and the motion/parts panel. '
                        'Costs ~16 %% per iteration and, through the '
                        'membership LR hold, a later stop (128 vs 90 min).')
    g.add_argument('--dyn-parts-from', type=float, default=0.375,
                   dest='dyn_parts_from', metavar='F',
                   help='iteration at which the parts separate: a fraction of '
                        '--iterations if < 1, else absolute (default: 0.375 = '
                        'half-way through the motion LR hold). Until then the '
                        'parts are TIED (identical fields, frozen uniform '
                        'membership = the single-field model); at it the '
                        'single field\'s local motion (gate displacement + '
                        'Jacobian) is clustered by k-means into K parts and '
                        'the membership set from the labels, fields '
                        'unchanged, so the rendered state is continuous. '
                        'MEASURED (two-shell toy): from a symmetric start '
                        'the parts never separate — identical fields give '
                        'the membership nothing to choose between — so the '
                        'labelling is not optional. Never before '
                        '--dyn-motion-from.')
    g.add_argument('--dyn-parts-volume-weight', type=float, default=6.0,
                   dest='dyn_parts_volume_weight', metavar='W',
                   help='weight of the volume-change feature in the labelling: '
                        'log|det F| x grid cell x W, next to the displacement '
                        '(default: 6 at the 1 mm grid = the same physical '
                        'scale as 2 at 3 mm). This is what separates the LUNG, which '
                        'expands rather than translates: MEASURED on the '
                        'muqojfqg field, 0 or 1 gives liver + diaphragm vs '
                        'everything else (lung with the wall), 2 and 4 give '
                        'lung / heart + abdomen / rest.')
    g.add_argument('--dyn-part-knn', type=int, default=8, dest='dyn_part_knn',
                   metavar='N',
                   help='neighbours per primitive in the membership graph '
                        '(default: 8), built once from the rest positions at '
                        'the first regularised render (one KD-tree query).')
    g.add_argument('--dyn-reg-part-smooth', type=float, default=0.1,
                   dest='dyn_reg_part_smooth', metavar='W',
                   help='weight of the membership total-variation term: mean '
                        'Charbonnier ||p_i - p_j|| over graph edges inside the '
                        'support (default: 0.1, UNTUNED). L1-like on purpose: '
                        'a boundary costs its area, not its sharpness, so the '
                        'partition is piecewise constant with sharp edges.')
    g.add_argument('--dyn-reg-part-commit', type=float, default=1e-2,
                   dest='dyn_reg_part_commit', metavar='W',
                   help='weight of the mean normalised membership entropy '
                        '(default: 1e-2, UNTUNED): pushes every primitive to '
                        'commit to one part.')
    g.add_argument('--dyn-air-sourced', action='store_true', dest='dyn_air_sourced',
                   default=True,
                   help='only air changes volume: every basis field is split '
                        'on the control grid into a divergence-free part and '
                        'a potential part, and the potential part\'s source '
                        'at each node is scaled by the air-bearing fraction '
                        'of the parcels in its cell (each parcel\'s HU in the '
                        'rest state rendered at --dyn-air-gate-mm, ramped '
                        'from solid at -100 HU to free at -250 HU; per part '
                        'once the parts are active; read when the motion '
                        'starts, at activation and at every evaluation). '
                        'Solid tissue keeps its volume by construction, the '
                        'lung keeps the field it had, no weight to tune. Replaces the tissue '
                        'penalty of 2026-09-19 (run 54hxw5ix), which erased '
                        'the lung\'s ventilation together with the '
                        'tissue\'s. DEFAULT ON since 2026-09-20 (A/B vs 90pu264k: '
                        'every thoracic NCC vs FDK up >= 2x the null, soft 0.8245 = '
                        'the static single-gate run, held-out tie, solid-tissue '
                        'strain rms 0.093 -> 0.066, vendor ventilation r unchanged); '
                        '--dyn-no-air-sourced restores the free divergence.')
    g.add_argument('--dyn-no-air-sourced', action='store_false', dest='dyn_air_sourced',
                   help='free divergence everywhere (the model before 2026-09-20).')
    g.add_argument('--dyn-air-gate-mm', type=float, default=0.5, dest='dyn_air_gate_mm',
                   metavar='MM', help='pitch of the rest-state render the air gate '
                                      'is read from (default: 0.5 mm).')
    g.add_argument('--dyn-freeze-views', default='', dest='dyn_freeze_views',
                   metavar='V,V,...',
                   help='views whose renders train the CLOUD and the '
                        'photometric model but NOT the motion model: '
                        'rendered in their gate state with zero free code, '
                        'motion parameters detached, left out of the code '
                        'priors. For views whose excess residual is not '
                        'motion (the residual-vs-view diagnostic names '
                        'them). Comma-separated view indices; default none.')
    g.add_argument('--dyn-max-disp-mm', type=float, default=2.0,
                   dest='dyn_max_disp_mm', metavar='MM',
                   help='displacement bound: a hinge penalty on any primitive '
                        'moving further than this (default: 2 mm — the '
                        'diaphragm excursion between Scan_1510\'s two gates '
                        'MEASURED on the vendor volumes). 0 = no bound.')
    g.add_argument('--dyn-reg-disp', type=float, default=1.0,
                   dest='dyn_reg_disp', metavar='W',
                   help='weight of that hinge, in voxels^2 (default: 1).')
    g.add_argument('--dyn-reg-code', type=float, default=1e-2,
                   dest='dyn_reg_code', metavar='W',
                   help='weight of mean ||z_v||^2 (default: 1e-2): keeps the '
                        'rest cloud the mean state.')
    g.add_argument('--dyn-reg-code-smooth', type=float, default=1e-2,
                   dest='dyn_reg_code_smooth', metavar='W',
                   help='weight of mean ||z_{v+1} - z_v||^2 in acquisition '
                        'order (default: 1e-2).')
    g.add_argument('--dyn-reg-basis', type=float, default=0.01,
                   dest='dyn_reg_basis', metavar='W',
                   help='weight of the basis fields\' squared finite '
                        'differences between neighbouring control points, '
                        'in voxels^2 (default: 0.01 with the bending order '
                        'on the 1 mm grid, chosen 2026-09-21 by vendor-blind '
                        'rules: tissue in lung cells strains no more than '
                        'tissue in solid cells, held-out not below the '
                        'unregularised grid; 0.03 flattened real regional '
                        'contrast, 0 left node-to-node speckle. Was 1e-3 '
                        'with order 1 on the 3 mm grid). The order of the '
                        'differences is --dyn-reg-basis-order.')
    g.add_argument('--dyn-reg-basis-lung-weight', type=float, default=0.0,
                   dest='dyn_reg_basis_lung_weight', metavar='F',
                   help='weight of the basis prior at a fully air-bearing '
                        'control node, relative to a solid one. 0 (DEFAULT '
                        'since 2026-09-24) keeps the bending penalty in '
                        'tissue only, scaled per node by the air gate; 1 is '
                        'the uniform prior of 2026-09-21. MEASURED: uniform '
                        'costs 6 %% of tidal volume at the lung boundary on '
                        'the digital phantom (band 88 %% vs 94 %%) and vendor '
                        'agreement on Scan_1510 (block r 0.705 vs 0.77), '
                        'with the core, the image, held-out and the tissue '
                        'strain inside the thorax unchanged. Reads the air '
                        'gate, rendered for it even with --dyn-no-air-sourced '
                        '(since 2026-09-30); uniform until its first refresh.')
    g.add_argument('--dyn-reg-basis-lung-region', choices=('air', 'thorax', 'box'), default='air',
                   dest='dyn_reg_basis_lung_region', metavar='R',
                   help='where --dyn-reg-basis-lung-weight applies: air '
                        '(default) = every air-bearing control cell, the air '
                        'around the animal included; thorax = only cells '
                        'within 3 mm of the rendered lung; box = the air '
                        'inside the rendered box. MEASURED 2026-09-24: '
                        'thorax frees the lung and changes nothing at its '
                        'boundary (phantom band 88 %% as uniform, animal '
                        'vendor r 0.72), air gives band 94 %% and r 0.77 — '
                        'the field continues into the outside air with real '
                        'curvature and the bending penalty there is paid at '
                        'the lung edge.')
    g.add_argument('--dyn-reg-basis-order', type=int, choices=(1, 2), default=2,
                   dest='dyn_reg_basis_order', metavar='N',
                   help='2 (default since 2026-09-21): second differences = '
                        'the bending energy (Rueckert 1999), zero for any '
                        'uniform strain, positive only where the strain '
                        'varies — the prior for a fine grid. 1: first '
                        'differences = the strain itself, which charges a '
                        'uniform lung expansion 16x what node-to-node '
                        'speckle costs and erased it at the weight that '
                        'removed the speckle (run grid1_rb03).')
    g.add_argument('--dyn-holdout-code-iters', type=int, default=30,
                   dest='dyn_holdout_code_iters', metavar='N',
                   help='Adam steps that fit the withheld view\'s code at each '
                        'evaluation, cloud frozen (default: 30).')
    g.add_argument('--dyn-holdout-code-lr', type=float, default=1e-2,
                   dest='dyn_holdout_code_lr', metavar='LR',
                   help='learning rate of that fit (default: 1e-2).')
    g.add_argument('--dyn-export-view', type=int, default=None,
                   dest='dyn_export_view', metavar='V',
                   help='export the volume in the state of view V instead of '
                        'the rest (mean) state.')
    g.add_argument('--dyn-export-group', type=_export_group, default=0,
                   dest='dyn_export_group', metavar='G|rest',
                   help='with several gated phases loaded: export the MEAN '
                        'state of acquisition group G (default: 0 = the '
                        'first phase, the state the single-gate runs and '
                        'the vendor gate-0 volume are in); "rest" exports '
                        'the midpoint between the gates, a state the '
                        'animal never occupied. Ignored with one phase.')
    g.add_argument('--dyn-phase-harmonics', type=int, default=1,
                   dest='dyn_phase_harmonics', metavar='K',
                   help='with a CONTINUOUS breathing phase per view (a '
                        'view_phase.json in the scan folder): the gate axis '
                        'becomes the Fourier embedding cos/sin(2 pi k phi), '
                        'k = 1..K, one periodic field per column (default: '
                        '1 = a sinusoidal breath; 2 adds the asymmetry '
                        'between inhale and exhale). Ignored for gated scans.')
    g.add_argument('--dyn-intensity-scale', action='store_true', default=False,
                   dest='dyn_intensity_scale',
                   help='fit a per-state INTENSITY SCALE s = 1 + gain . fixed '
                        'columns (two numbers for a sinusoidal phase axis) that '
                        'multiplies every rendered projection: a nuisance for '
                        'data whose calibration drifts with the breathing state '
                        '(the X2GS cases normalise each phase to its own max). '
                        'The anatomy stays mass-conserving. Default: off.')
    g.add_argument('--dyn-intensity-harmonics', type=int, default=1,
                   dest='dyn_intensity_harmonics', metavar='K',
                   help='harmonics of that intensity scale on a continuous phase '
                        'axis (2K numbers; default 1). 4 follows any pattern over '
                        'ten phase bins (the X2GS cases).')
    g.add_argument('--dyn-reg-tv', type=float, default=0.0, dest='dyn_reg_tv', metavar='W',
                   help='weight of a volumetric TV term matched to X2-Gaussian\'s '
                        '(random --dyn-tv-patch^3 patch at the export voxel size, L1 '
                        'of neighbour differences, mean, in image units mu / '
                        '--dyn-tv-mu-ref). For the SSIM comparison only; default 0.')
    g.add_argument('--dyn-tv-patch', type=int, default=32, dest='dyn_tv_patch', metavar='N')
    g.add_argument('--dyn-tv-mu-ref', type=float, default=0.02, dest='dyn_tv_mu_ref', metavar='MU',
                   help='attenuation (/mm) of image value 1 for the TV term (default 0.02 = '
                        'the NAF import\'s mu_per_unit)')
    g.add_argument('--dyn-export-phase', type=float, default=None,
                   dest='dyn_export_phase', metavar='PHI',
                   help='with a continuous breathing phase: export the volume '
                        'at phase PHI in [0, 1) instead of the rest (cycle-'
                        'mean) state; the counterpart of --dyn-export-group.')
    g.add_argument('--dyn-export-4d-states', default=None,
                   dest='dyn_export_4d_states', metavar='V1,V2,...',
                   help='the exact states of the 4-D export along the '
                        'breathing axis (gate codes in [-1, 1] for a gated '
                        'scan, phases in [0, 1) for a continuous one), e.g. '
                        'the phase mid-times a benchmark scores at. Default: '
                        '--dyn-export-4d equal steps.')
    g.add_argument('--dyn-export-4d', type=int, default=9, dest='dyn_export_4d',
                   metavar='N',
                   help='ALSO write the breathing sequence as a 4-D file '
                        '(<output>_4d.nrrd or .nii.gz per --dyn-export-4d-'
                        'format, + <output>_4d.json): N states along the '
                        'gate axis from gate 0 (-1) to gate 1 (+1), on the '
                        'delivered ROI, in the volume\'s HU calibration '
                        '(default: 9 -- odd, so the rest state, code 0, is one '
                        'of them; 0 = off). Needs >= 2 gates.')
    g.add_argument('--dyn-export-4d-format', choices=('nrrd', 'nrrd-slicer', 'nifti'),
                   default='nrrd', dest='dyn_export_4d_format',
                   help='file type of that sequence (default: nrrd). nrrd: '
                        'state axis last, all voxel steps positive, voxel '
                        'order of the .vff: opens in Imalytics Preclinical '
                        '(mirrored in x in orientation-aware viewers); '
                        'nrrd-slicer: state axis first in true LPS placement, '
                        'the layout 3D Slicer loads as a volume sequence; '
                        'nifti: NIfTI-1, state axis last, RAS sform. Same '
                        'voxels and HU in all three; convert a written NRRD '
                        'with ct120-export-4d.')
    g.add_argument('--dyn-export-4d-mm', default='auto',
                   dest='dyn_export_4d_mm', metavar='MM',
                   help='voxel pitch of those states: "auto" (default) = the '
                        'delivered volume\'s own grid, so state 0 and '
                        'recon.vff are the same voxels (88 M per state on '
                        'Scan_1510, streamed one state at a time); or an '
                        'isotropic pitch in mm for a smaller file.')
    g.add_argument('--dyn-export-ventilation', action='store_true',
                   dest='dyn_export_ventilation', default=True,
                   help='ALSO write the regional ventilation map '
                        '(<output>_ventilation.nrrd = specific ventilation '
                        '(J - 1)/f_air per voxel in the expiration frame, '
                        '_ventilation_J.nrrd, _ventilation.json with the '
                        'global check, _ventilation.npz with the per-'
                        'primitive readings, _ventilation.png). Needs >= 2 '
                        'gates; the expiration gate is the one holding less '
                        'air. Default: on.')
    g.add_argument('--dyn-no-export-ventilation', action='store_false',
                   dest='dyn_export_ventilation', help='skip the ventilation map.')
    g.add_argument('--dyn-ventilation-mm', type=float, default=0.5,
                   dest='dyn_ventilation_mm', metavar='MM',
                   help='isotropic pitch of the ventilation map (default: '
                        '0.5 mm = half the map\'s measured MTF-10 of 1.1 mm on '
                        'the texture phantom, the coarsest grid that keeps all '
                        'of its content; the phantom scorer uses 0.15).')
    for name, default in MotionModel.DEFAULT_LR_MULTIPLIERS.items():
        g.add_argument(f'--dyn-lr-{name.replace("motion_", "")}', type=float,
                       default=None, dest=f'dyn_lr_{name}', metavar='MULT',
                       help=f'LR multiplier on the {name} group relative to '
                            f'--lr (default: {default:g}).')


def _export_group(spec) -> int | None:
    """'0' -> 0; 'rest' / 'mid' / '' -> None (the midpoint state)."""
    tok = str(spec if spec is not None else '').strip().lower()
    if tok in ('', 'rest', 'mid', 'midpoint', 'none'):
        return None
    return int(tok)


def _parse_views(spec: str) -> tuple:
    """'3,17,340' -> (3, 17, 340); blanks and repeats ignored."""
    out = set()
    for tok in str(spec or '').replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            out.add(int(tok))
    return tuple(sorted(out))


def options(args) -> dict:
    if int(args.dyn_parts) > 1 and str(getattr(args, 'gauss_density_control', 'fixed')) != 'fixed':
        raise ValueError(
            "--dyn-parts > 1 (the default is 3) needs a fixed-size cloud: the "
            "per-primitive membership is not carried through clone/split/prune "
            "or relocation. Use --gauss-density-control fixed (the default) "
            "or --dyn-parts 1.")
    if int(args.dyn_parts) > 1 and bool(getattr(args, 'gauss_mcmc', False)):
        raise ValueError("--dyn-parts > 1 (the default is 3) needs a fixed-size "
                         "cloud, not MCMC density control (relocation does not "
                         "carry the membership); pass --dyn-parts 1.")
    if args.dyn_mass_preserving and not args.dyn_transport_shapes:
        raise ValueError(
            "mass preservation (the default) needs the Jacobian shape "
            "transport: with --dyn-no-transport-shapes every primitive keeps "
            "its rest covariance, so its mass is already invariant and dividing "
            "the amplitude by |det F| would break it. Add "
            "--dyn-no-mass-preserving.")
    opts = static.options(args)
    opts.update(
        n_codes=int(args.dyn_codes),
        grid_mm=float(args.dyn_grid_mm),
        support=str(args.dyn_support),
        support_margin_mm=float(args.dyn_support_margin_mm),
        support_taper_mm=float(args.dyn_support_taper_mm),
        motion_from=float(args.dyn_motion_from),
        motion_hold=float(args.dyn_motion_hold),
        basis_init_vox=float(args.dyn_basis_init_vox),
        transport_shapes=bool(args.dyn_transport_shapes),
        mass_preserving=bool(args.dyn_mass_preserving),
        det_floor=float(args.dyn_det_floor),
        reg_fold=float(args.dyn_reg_fold),
        amplitude=bool(args.dyn_amplitude),
        frozen_views=_parse_views(args.dyn_freeze_views),
        n_parts=int(args.dyn_parts),
        parts_from=float(args.dyn_parts_from),
        parts_volume_weight=float(args.dyn_parts_volume_weight),
        part_knn=int(args.dyn_part_knn),
        reg_part_smooth=float(args.dyn_reg_part_smooth),
        reg_part_commit=float(args.dyn_reg_part_commit),
        air_sourced=bool(args.dyn_air_sourced),
        air_gate_mm=float(args.dyn_air_gate_mm),
        reg_amp=float(args.dyn_reg_amp),
        max_disp_mm=float(args.dyn_max_disp_mm),
        reg_disp=float(args.dyn_reg_disp),
        reg_code=float(args.dyn_reg_code),
        reg_code_smooth=float(args.dyn_reg_code_smooth),
        reg_basis=float(args.dyn_reg_basis),
        reg_basis_order=int(args.dyn_reg_basis_order),
        reg_basis_lung_weight=float(args.dyn_reg_basis_lung_weight),
        reg_basis_lung_region=str(args.dyn_reg_basis_lung_region),
        holdout_code_iters=int(args.dyn_holdout_code_iters),
        holdout_code_lr=float(args.dyn_holdout_code_lr),
        export_view=args.dyn_export_view,
        export_group=args.dyn_export_group,
        export_phase=args.dyn_export_phase,
        phase_harmonics=int(args.dyn_phase_harmonics),
        intensity_scale=bool(args.dyn_intensity_scale),
        intensity_harmonics=int(args.dyn_intensity_harmonics),
        reg_tv=float(args.dyn_reg_tv), tv_patch=int(args.dyn_tv_patch),
        tv_mu_ref=float(args.dyn_tv_mu_ref),
        export_4d=int(args.dyn_export_4d),
        export_4d_states=_state_list(args.dyn_export_4d_states),
        export_4d_mm=args.dyn_export_4d_mm,
        export_4d_format=args.dyn_export_4d_format,
        export_ventilation=bool(args.dyn_export_ventilation),
        ventilation_mm=float(args.dyn_ventilation_mm),
        motion_lr_multipliers={k: float(getattr(args, f'dyn_lr_{k}'))
                               for k in MotionModel.DEFAULT_LR_MULTIPLIERS
                               if getattr(args, f'dyn_lr_{k}', None) is not None},
    )
    return opts


def _state_list(text) -> tuple | None:
    """``--dyn-export-4d-states``: a comma list of floats, or None."""
    if text is None or str(text).strip() == '':
        return None
    try:
        vals = tuple(float(t) for t in str(text).split(',') if t.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"--dyn-export-4d-states: {e}") from e
    if not vals:
        return None
    return vals


def footprint(args, req: MachineRequest) -> Footprint:
    """The static footprint: the motion model is tens of thousands of numbers
    plus one (N, 3) displacement buffer per render, which the static model's
    per-primitive constant already covers with margin."""
    return static.footprint(args, req)


def after_save(reconstructor, ctx, args, output_path, anchors, logger) -> None:
    """The breathing sequence and the ventilation map, once the volume is
    on disk with its anchors."""
    vent = reconstructor.export_ventilation(ctx.geometry, anchors, output_path)
    if vent and logger is not None and hasattr(logger, 'set_summary'):
        import json as _json
        with open(vent[:-len('.nrrd')] + '.json') as f:
            side = _json.load(f)
        logger.set_summary({'ventilation/file': vent,
                            'ventilation/reference_gate': int(side['reference_gate']),
                            'ventilation/J_median': float(side['J']['median']),
                            'ventilation/J_frac_below_1': float(side['J']['frac_below_1']),
                            'ventilation/SV_median': float(side['SV']['median']),
                            'ventilation/delta_air_image_mm3': float(side['delta_air_image_mm3']),
                            'ventilation/delta_volume_field_mm3': float(side['delta_volume_field_mm3']),
                            'ventilation/field_over_image': float(side['field_over_image'])})
    path = reconstructor.export_4d(ctx.geometry, anchors, output_path)
    if path and logger is not None and hasattr(logger, 'set_summary'):
        # the pitch actually used (the option may be 'auto'): the sidecar
        # the export wrote beside the file records it
        import json
        from .export4d import sidecar_path
        with open(sidecar_path(path)) as f:
            side = json.load(f)
        vs = side.get('voxel_size_mm', {})
        logger.set_summary({'export/4d': path,
                            'export/4d_format': str(side.get('format', '')),
                            'export/4d_states': int(reconstructor.mcfg['export_4d']),
                            'export/4d_mm_xy': float(vs.get('xy', float('nan'))),
                            'export/4d_mm_z': float(vs.get('z', float('nan')))})


ALGORITHM = LearnedAlgorithm(
    name='gaussian_dynamic',
    reconstructor=DynamicGaussianReconstructor,
    summary='Gaussian splatting with a per-view state code and a low-rank '
            'deformation basis (time-aware)',
    default_lr=static.DEFAULT_LR,
    # The static recipe's shared defaults plus two of its own. A motion model
    # wants EVERY gated phase (the gate column is what turns the phase
    # difference into one learned field); with a single phase it degrades to
    # free codes only. And it trains on every loaded gate's views — twice the
    # sinogram on Scan_1510 — so the static cap of 16000 would end the run
    # while the held-out score is still rising: the v4 recipe (z708w98r,
    # 2026-09-11) peaked at 20500 and stopped itself at 21000 under a cap of
    # 32000, which is therefore the cap. The stopping rule still ends the run
    # earlier when the held-out score plateaus; --dyn-motion-from and the
    # --min-stop-iter default are fractions of this number. An explicit
    # --phase or --iterations still wins.
    driver_defaults={**static.DRIVER_DEFAULTS, 'phase': 'all',
                     'iterations': 32000},
    add_args=add_args,
    options=options,
    footprint=footprint,
    after_save=after_save,
)
