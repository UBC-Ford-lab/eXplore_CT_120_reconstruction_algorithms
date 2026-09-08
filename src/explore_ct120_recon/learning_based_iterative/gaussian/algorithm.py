"""Gaussian splatting as a selectable ``--algorithm``.

The three things the driver cannot know, kept with the representation: the
flags it adds, the constructor arguments it takes, and how much of the machine
it needs. Modelled on ``learning_based_iterative/voxel/algorithm.py``, which is
the file this pattern is meant to be copied from.

    python -m explore_ct120_recon.run_learned_recon SCAN \
        --algorithm gaussian
"""
from __future__ import annotations

import os

from ...ct_core.preflight import Footprint, MachineRequest
from ..registry import LearnedAlgorithm

from .kernel import BYTES_PER_SLAB_VOXEL, DEFAULT_MAX_VOXELS
from .model import GaussianCloud
from .reconstructor import GaussianReconstructor

#: VRAM per primitive during TRAINING, in bytes — the allocator's PEAK
#: (`torch.cuda.max_memory_allocated`), NOT nvidia-smi. The previous constant
#: here (4.9 kB) came from nvidia-smi, which reports the caching allocator's
#: reservation: MEASURED on the W&B system metrics of every 2.4 M run, that
#: reservation climbs to 15.4 GiB of the P100's 15.9 in the first three
#: minutes and is then trimmed to 7.8 the moment an allocation fails, for a
#: cloud whose real peak is a fraction of that; it also holds whatever the
#: seeding FDK grabbed (that backend budgets itself to the FREE memory, 8.7
#: GiB at ds3, and releases it). So nvidia-smi over-reads need by the size of
#: the cache and under-reads it whenever the cache happens to be small — it
#: was the source of both the "4.9 kB" and the later "3x over-estimate" claim.
#: Every run now logs the true peak at each evaluation (`sys/gpu_peak_alloc_gib`).
#: MEASURED with `torch.cuda.max_memory_allocated` per stage (2026-09-07,
#: Scan_1510 at ds3, 220 x 765 x 1166 sinogram = 0.73 GiB, fixed 2.4 M cloud,
#: WLS so the weights double the resident sinogram to 1.47 GiB):
#:   seed FDK 6.36 GiB / training 3.06 GiB (steady from the 3rd render) /
#:   export 3.57 GiB / 2.04 after.
#: Training above the resident sinogram is 1.59 GiB for 2.4 M = 0.66 kB per
#: primitive (ds6, 1.2 M: 0.56 kB): the 11 parameters, two Adam moments, the
#: accumulators, and the rasteriser's per-primitive geometry and tile-pair
#: buffers, which grow with a primitive's FOOTPRINT in pixels (so with the
#: binning); the constant is for ds3 and conservative at coarser binnings.
TRAIN_BYTES_PER_GAUSSIAN = 720
#: Per-pixel bytes of one resident view during a step: prediction, target,
#: residual, the null-gate copies and the rasteriser's per-pixel outputs.
TRAIN_BYTES_PER_PIXEL = 48
#: The export (`kernel.chunked_voxelize`): the parameters and optimiser
#: state still alive, the extension's per-primitive binning buffer, beside
#: ONE slab (measured 3.57 - 1.47 sinogram - 0.48 slab = 1.62 GiB for 2.4 M).
EXPORT_BYTES_PER_GAUSSIAN = 720
#: The seeding FDK sizes its own chunks to the FREE memory and releases
#: everything afterwards, so it is charged at what it takes when
#: unconstrained: 6.36 - 1.47 resident - 0.46 seed volume = 4.4 GiB = six
#: sinogram copies (raw, flat-fielded, filtered, padded FFT). With less free
#: it chunks the views and needs less, so this is the conservative end.
SEED_SINO_COPIES = 6

#: Adam LR for the position group; the other three groups are fixed multiples
#: of it (see ``GaussianCloud.param_groups``). NOT the driver's global 1e-4:
#: that value is a step in units of mu for a dense grid, and a Gaussian's
#: dominant parameter is a POSITION in normalised world units, where 1e-4 is a
#: reasonable step and mu's scale is irrelevant.
DEFAULT_LR = 1e-4

#: Production cloud size (seed = cap under fixed density control). MEASURED
#: Scan_1510 ds3: 3.06 GiB resident in training (~0.7 kB per primitive above
#: the sinogram), 188 min to the 16 000-iteration cap on a P100 at ~1.4 it/s.
#: The seed's ROI weight of 5 puts ~2.0 M of these inside the delivered ROI
#: and keeps ~0.4 M on the bed and the truncated periphery, whose shadows are
#: in every projection (see --gauss-seed-roi-weight).
DEFAULT_N_GAUSSIANS = 2_400_000


def add_args(group) -> None:
    """Flags that mean something only to a cloud of primitives.

    The driver hands us the argument group it already created for this
    algorithm, so these attach directly rather than nesting another one.
    """
    g = group
    g.add_argument('--gauss-seeds', type=int, default=DEFAULT_N_GAUSSIANS,
                   dest='gauss_seeds',
                   help='primitives to start from (default: %(default)s, the '
                        'production budget: ~0.7 kB each so VRAM is not the '
                        'limit, wall-clock is — it/s ~ 1/N). Splatting cannot '
                        'create matter where no primitive was seeded, so this '
                        'is a floor on what the model can represent, not just '
                        'a speed knob.')
    g.add_argument('--gauss-max', type=int, default=DEFAULT_N_GAUSSIANS,
                   dest='gauss_max',
                   help='cap on primitives after densification (default: '
                        '%(default)s = the seed count, i.e. no growth under '
                        'the default fixed density control).')
    g.add_argument('--gauss-seed-from', choices=('fdk', 'uniform'),
                   default='fdk', dest='gauss_seed_from',
                   help="'fdk' (default) seeds mass-proportionally from an FDK "
                        "of this scan on the model domain; 'uniform' needs no "
                        "reference and converges more slowly.")
    g.add_argument('--gauss-seed-scale', type=float, default=1.0,
                   dest='gauss_seed_scale',
                   help='multiplier on the initial primitive width, which '
                        'defaults to the mean spacing implied by the seed.')
    g.add_argument('--gauss-seed-floor', type=float, default=0.5,
                   dest='gauss_seed_floor',
                   help='quantile of the reference volume treated as the noise '
                        'floor before it becomes a sampling weight '
                        '(default: 0.5). Without it most of the cloud lands in '
                        'reconstruction noise.')
    g.add_argument('--gauss-signal-scale', default='auto',
                   dest='gauss_signal_scale',
                   help="multiplier on the sinogram, to keep line integrals off "
                        "the rasteriser's absolute 1e-5 alpha cutoff (see "
                        "kernel.py). 'auto' (default) puts the mean rendered "
                        "value at 1.0. Set 1 only with a patched kernel.")
    g.add_argument('--gauss-densify-every', type=int, default=100,
                   dest='gauss_densify_every',
                   help='iterations between clone/split/prune rounds.')
    g.add_argument('--gauss-densify-fraction', type=float, default=0.02,
                   dest='gauss_densify_fraction',
                   help='fraction of primitives densified per round, selected '
                        'by gradient QUANTILE (default: 0.02). Scale-free, so '
                        'it transfers across scans; the absolute threshold it '
                        'replaces silently selected nothing on real data.')
    g.add_argument('--gauss-densify-grad', type=float, default=None,
                   dest='gauss_densify_grad',
                   help='absolute gradient threshold instead of the quantile. '
                        'Only for reproducing an upstream run.')
    g.add_argument('--gauss-densify-window', type=float, nargs=2,
                   default=(0.05, 0.5), metavar=('FROM', 'UNTIL'),
                   dest='gauss_densify_window',
                   help='fractions of the run between which densification is '
                        'active (default: 0.05 0.5). It stops early so the '
                        'last half of the run fits a fixed cloud.')
    g.add_argument('--gauss-split', choices=('preserving', 'random'),
                   default='preserving', dest='gauss_split',
                   help="how a selected primitive is refined (default: "
                        "preserving). 'preserving' replaces it by a "
                        "moment-matched pair that renders what it rendered, so "
                        "a split is invisible to the data until the optimiser "
                        "pulls the children apart; 'random' is upstream's rule "
                        "(children at a random draw from the parent, all axes "
                        "/1.6, half the mass), which changes the volume in ways "
                        "the data never asked for — MEASURED runs "
                        "ufsqlhpn/5eujeyov: rings where ribs were solid and "
                        "filaments off bone, all below the projection noise.")
    g.add_argument('--gauss-densify-gate', type=float, default=3.0,
                   dest='gauss_densify_gate', metavar='Z',
                   help='refine only primitives whose data gradient beats the '
                        'gradient the same view produces against a '
                        'sign-scrambled residual by Z standard errors '
                        '(default: 3). Below noise the data are silent and a '
                        'split can only fill the null space. Costs one extra '
                        'backward per iteration while densification is open. '
                        'Set 0 to disable.')
    g.add_argument('--gauss-density-control',
                   choices=('fixed', 'adaptive', 'mcmc'), default='fixed',
                   dest='gauss_density_control',
                   help="how the primitive count is managed (default: fixed). "
                        "'fixed': the seed cloud IS the cloud — no split/"
                        "clone/prune, no relocation, no noise gate, no "
                        "Langevin step; the count is the resolution budget "
                        "(one primitive per resolution cell) and the width "
                        "ceiling + shape projection still apply. This is the "
                        "production recipe: MEASURED, relocation moved 0 "
                        "primitives in every fixed-count run and the "
                        "amplitude L1 that would feed it costs held-out fit. "
                        "'adaptive': clone/split/prune inside --gauss-densify-"
                        "window, refined where the data gradient beats the "
                        "noise gate. 'mcmc': MCMC density control (Kheradmand "
                        "et al. 2024) — dead primitives (mass below --gauss-"
                        "mcmc-dead x the seed median) are moved onto live ones "
                        "drawn by mass inside the export ROI with the "
                        "amplitude shared, plus a Langevin step for the "
                        "near-dead.")
    g.add_argument('--gauss-mcmc', action='store_true', dest='gauss_mcmc',
                   help='DEPRECATED alias for --gauss-density-control mcmc.')
    g.add_argument('--gauss-mcmc-dead', type=float, default=0.01,
                   dest='gauss_mcmc_dead', metavar='F',
                   help='dead line for MCMC relocation, as a fraction of the '
                        'seed median mass (default: 0.01).')
    g.add_argument('--gauss-mcmc-noise', type=float, default=0.5,
                   dest='gauss_mcmc_noise', metavar='S',
                   help='Langevin step for near-dead primitives, in units of '
                        'their own sigma per iteration at the base LR, decaying '
                        'with the LR schedule (default: 0.5).')
    g.add_argument('--gauss-density-reg', type=float, default=0.01,
                   dest='gauss_density_reg', metavar='W',
                   help='L1 on the mean amplitude relative to the seed median, '
                        'added to the loss in MCMC mode (default: 0.01, '
                        "upstream's opacity_reg). It is what retires redundant "
                        'primitives so relocation has something to move.')
    g.add_argument('--gauss-scale-reg', type=float, default=0.0,
                   dest='gauss_scale_reg', metavar='W',
                   help='L1 on the mean width relative to the seed median '
                        "(default: 0 — upstream uses 0.01; here the width "
                        'ceiling already does that job).')
    g.add_argument('--gauss-seed-roi-weight', type=float, default=5.0,
                   dest='gauss_seed_roi_weight', metavar='W',
                   help='multiply the seed sampling weight inside the export '
                        'ROI + 1 mm by W (default: 5). A mass-proportional '
                        'seed puts half the budget on the bed and the '
                        'truncated periphery. Do NOT starve them: their '
                        'shadows are in every projection, and MEASURED (ds12 '
                        'smoke, 60 k seeds) W=8 left 5 k outside and cost 4.5x '
                        'in held-out MSE whatever the density control. Use W '
                        'to ADD budget inside while keeping the outside count '
                        'of a known-good run: W ~ N_total / N_outside - 1 '
                        '(2.4 M with 400 k outside -> W=5).')
    g.add_argument('--gauss-signed-density', type=float, default=0.0,
                   metavar='F', dest='gauss_signed_density',
                   help='allow NEGATIVE amplitudes down to -F x the seed '
                        'median amplitude (default: 0 = strictly positive). '
                        'A sum of positive Gaussians cannot undershoot at an '
                        'edge and cannot represent air as a noisy zero '
                        '(MEASURED +34 HU air bias); a small bounded floor '
                        'gives the edge its negative lobe without opening the '
                        'cancelling-pair null space. Needs the rasteriser '
                        'built with the |alpha| cutoff patch.')
    g.add_argument('--gauss-loss', choices=('l1_dssim', 'wls'),
                   default='wls', dest='gauss_loss',
                   help="data term (default: wls). 'wls' = weighted least "
                        "squares on the log-attenuation with the MEASURED "
                        "inverse variances (Var = A/count + B, counts above "
                        "dark from the binned raw projections) and NOTHING "
                        "else — the loss value is the reduced chi-square, 1.0 "
                        "= at the photon noise. 'l1_dssim' is the splatting "
                        "literature's objective, kept for comparison.")
    g.add_argument('--gauss-wls-var', nargs=2, default=('auto', 'auto'),
                   metavar=('A', 'B'), dest='gauss_wls_var',
                   help="noise model for --gauss-loss wls, PER RAW DETECTOR "
                        "PIXEL: Var(log-attenuation) = A/count + B, scaled by "
                        "1/ds^2 for a pooled sinogram. Default 'auto auto': "
                        "MEASURED from the scan's own air pixels across "
                        "consecutive views (reconstructor.estimate_noise_"
                        "model; on Scan_1510 it reproduces the acquisition-"
                        "pair value to 3 %%). Falls back to Scan_1510's "
                        "0.8406 7.2e-5 with a warning when a scan has no air "
                        "pixels. Either may be a number to pin it.")
    g.add_argument('--gauss-wls-min-counts', type=float, default=1.0,
                   dest='gauss_wls_min_counts', metavar='C',
                   help='floor on counts above dark before the variance is '
                        'formed, so a dead pixel gets a large finite variance '
                        'instead of an infinite weight (default: 1).')
    g.add_argument('--gauss-wls-seam-t', type=float, default=6.0,
                   dest='gauss_wls_seam_t', metavar='T',
                   help='with --gauss-loss wls: detector columns whose offset '
                        'against their neighbours is static across ALL views '
                        'with |t| >= T get zero weight (default: 6; 0 = no '
                        'mask). A tiled-panel seam is a per-column step no '
                        'volume can render; MEASURED on Scan_1510 at columns '
                        '72 and 978-980 (ds3), t = 6.7 and 10.6.')
    g.add_argument('--gauss-dssim', type=float, default=0.2, dest='gauss_dssim',
                   help='weight of the structural term in L1 + w*DSSIM.')
    g.add_argument('--gauss-max-aspect', type=float, default=2.0,
                   dest='gauss_max_aspect', metavar='K',
                   help='cap on a primitive\'s longest/shortest axis ratio '
                        '(default: 2). Unbounded elongation is a DEGENERATE '
                        'solution: a needle is an angle-selective correction '
                        'that fits projections better than the true object '
                        '(measured run liztmby9 — best projection fit of five '
                        'volumes, worst image). Set 0 to disable.')
    g.add_argument('--gauss-min-sigma', default='auto', dest='gauss_min_sigma',
                   metavar='MM',
                   help="floor on a primitive's shortest axis in mm; 'auto' "
                        "(default) is half the voxel pitch, the finest thing "
                        "the export grid can carry. Structure below it is not "
                        "identifiable from the data. Set 0 to disable.")
    g.add_argument('--gauss-max-sigma-spacing', type=float, default=1.5,
                   dest='gauss_max_sigma_spacing', metavar='K',
                   help='cap on a primitive\'s width as a multiple of its own '
                        'local kNN spacing (default: 1.5). MEASURED run 83m5xh1e: '
                        'the low-contrast population sat 6.4x wider than the '
                        'gap to its own neighbours — a 91x overlapping basis '
                        'whose ENSEMBLE is blurred while no single primitive '
                        'is wrong, so no per-primitive gradient can see it. At '
                        'K=1 this is the seeding rule (sigma = local spacing) '
                        'held for the whole run instead of only at step 0. Set '
                        '0 to disable.')
    g.add_argument('--gauss-view-every-evals', type=int, default=4,
                   dest='gauss_view_every_evals', metavar='K',
                   help='also draw the three midplane views every K held-out '
                        'evaluations (default: 4; 0 = only at LR changes, like '
                        'every other backend). They land on the same '
                        'plots/lr_stage/view_* slider. Free for this backend: '
                        'the export ROI renders in ~1 s and the per-frame HU '
                        'refit costs ~1.6 s, so K=4 on a 250-iteration eval '
                        'cadence adds a few seconds per thousand iterations.')
    g.add_argument('--gauss-checkpoint', default=None, metavar='PATH',
                   dest='gauss_checkpoint',
                   help='where to write the trained cloud before the volume is '
                        'queried (default: <output stem>_cloud.pt next to the '
                        'volume; the system temp dir if there is no output '
                        'path yet, e.g. under a dry run). The export is the '
                        'last step and the only one that serialises anything, '
                        'so without this a failure there costs the entire '
                        'run, and the file is what --gauss-resume and any '
                        're-export read.')
    g.add_argument('--gauss-densify-roi-margin', type=float, default=None,
                   dest='gauss_densify_roi_margin', metavar='MM',
                   help='confine clone/split to the export ROI plus this '
                        'margin in mm (default: unconfined, the whole '
                        'reconstruction domain). MEASURED run ufsqlhpn: 52 %% '
                        'of a 1.2 M budget sat outside the ROI, on the '
                        'truncated animal and the bed, where fineness is '
                        'never delivered. Prune, floor and ceiling still act '
                        'everywhere.')
    g.add_argument('--gauss-resume', default=None, metavar='PATH',
                   dest='gauss_resume',
                   help='start from the cloud a previous run wrote with '
                        '--gauss-checkpoint instead of from the seed. The '
                        'seed is still built (it fixes the units), then '
                        'replaced; the checkpoint must come from the same '
                        'data and signal-scale setting, which is checked. '
                        'Densification, the ceiling and the stopping rules '
                        'all restart as for a fresh run of --iterations.')
    for name, default in GaussianCloud.DEFAULT_LR_MULTIPLIERS.items():
        g.add_argument(f'--gauss-lr-{name}', type=float, default=None,
                       dest=f'gauss_lr_{name}', metavar='MULT',
                       help=f'LR multiplier on the {name} group relative to '
                            f'--lr (default: {default:g}). The four parameter '
                            f'kinds live on different scales, so one rate '
                            f'trains at least three of them wrong; SCALING and '
                            f'ROTATION are the two that set resolution and '
                            f'carry the least rate by default.')


def _auto_or_float(v):
    if isinstance(v, str) and v.strip().lower() == 'auto':
        return 'auto'
    return float(v)


def _checkpoint_default(args) -> str | None:
    """``--gauss-checkpoint``, or ``<output stem>_cloud.pt`` beside the volume.

    The driver resolves ``args.output`` before it asks for our options, so the
    cloud lands where the VFF and its sidecar do. None (temp dir) only when
    there is no output path at all.
    """
    if getattr(args, 'gauss_checkpoint', None):
        return str(args.gauss_checkpoint)
    out = getattr(args, 'output', None)
    if not out:
        return None
    return os.path.splitext(str(out))[0] + '_cloud.pt'


def options(args) -> dict:
    frm, until = args.gauss_densify_window
    return dict(
        n_seed=int(args.gauss_seeds),
        max_gaussians=int(args.gauss_max),
        seed_from=str(args.gauss_seed_from),
        seed_scale=float(args.gauss_seed_scale),
        seed_floor_quantile=float(args.gauss_seed_floor),
        signal_scale=args.gauss_signal_scale,
        densify_from=float(frm),
        densify_until=float(until),
        densify_every=int(args.gauss_densify_every),
        densify_fraction=float(args.gauss_densify_fraction),
        densify_grad_threshold=(None if args.gauss_densify_grad is None
                                else float(args.gauss_densify_grad)),
        dssim_weight=float(args.gauss_dssim),
        checkpoint_path=_checkpoint_default(args),
        max_aspect=float(args.gauss_max_aspect),
        max_sigma_spacing=float(args.gauss_max_sigma_spacing),
        view_every_evals=int(args.gauss_view_every_evals),
        resume_from=args.gauss_resume,
        densify_roi_margin_mm=args.gauss_densify_roi_margin,
        split_mode=str(args.gauss_split),
        density_control=('mcmc' if args.gauss_mcmc
                         else str(args.gauss_density_control)),
        mcmc_dead_frac=float(args.gauss_mcmc_dead),
        mcmc_noise=float(args.gauss_mcmc_noise),
        density_reg=float(args.gauss_density_reg),
        scale_reg=float(args.gauss_scale_reg),
        seed_roi_weight=float(args.gauss_seed_roi_weight),
        signed_density=float(args.gauss_signed_density),
        loss_kind=str(args.gauss_loss),
        wls_var_per_count=_auto_or_float(args.gauss_wls_var[0]),
        wls_var_floor=_auto_or_float(args.gauss_wls_var[1]),
        wls_min_counts=float(args.gauss_wls_min_counts),
        wls_seam_t=float(args.gauss_wls_seam_t),
        densify_gate_z=(None if float(args.gauss_densify_gate) <= 0
                        else float(args.gauss_densify_gate)),
        min_sigma_mm=(args.gauss_min_sigma
                      if str(args.gauss_min_sigma) == 'auto'
                      else float(args.gauss_min_sigma)),
        lr_multipliers={k: float(getattr(args, f'gauss_lr_{k}'))
                        for k in GaussianCloud.DEFAULT_LR_MULTIPLIERS
                        if getattr(args, f'gauss_lr_{k}', None) is not None},
    )


def _n_max(args) -> int:
    """The primitive cap, or the production default when sized by name only
    (``estimate('gaussian', ...)`` hands ``args=None``)."""
    return int(getattr(args, 'gauss_max', DEFAULT_N_GAUSSIANS) or DEFAULT_N_GAUSSIANS)


def footprint_stages(args, req: MachineRequest) -> dict:
    """Peak device bytes of each SEQUENTIAL stage of a Gaussian run.

    Sized from the primitive CAP, not from the export grid: a voxel backend's
    VRAM is ``4 x volume``; a cloud's does not depend on the export grid at
    all until the final query, and depends on how many primitives the run may
    hold. The stages never overlap, so the run's peak is their MAX, not their
    sum — charging the sum is how the old single-term model over-read.
    """
    n = _n_max(args)
    sino = int(req.sino_bytes)
    wls = str(getattr(args, 'gauss_loss', 'wls')) == 'wls'
    # Seeding: the FDK reference on the model domain (see the constant),
    # beside the target and the WLS weights, which are already resident.
    seed = ((2 if wls else 1) * sino + SEED_SINO_COPIES * sino
            + int(req.vol_bytes) // 8)
    # Training: the whole sinogram is resident (a splat step fits a full
    # view, never a ray batch), and WLS keeps a weight per pixel beside it.
    train = ((2 if wls else 1) * sino
             + TRAIN_BYTES_PER_PIXEL * int(req.n_b) * int(req.n_a)
             + n * TRAIN_BYTES_PER_GAUSSIAN)
    # Export: ONE SLAB on the device, the volume assembled on the host
    # (`kernel.chunked_voxelize`), so this is bounded by the slab and does
    # not grow with the export grid. It did not always: this term read the
    # slab while the assembly also allocated the WHOLE grid on the GPU, so at
    # Scan_1510's 75 um / ds3 export the preflight under-read the peak by
    # 3.70 GiB and passed a configuration that OOMs at the final step. The
    # sinogram is still resident then, and the cloud is queried in place.
    slab = (min(int(req.vol_bytes) // 4, DEFAULT_MAX_VOXELS)
            * BYTES_PER_SLAB_VOXEL)
    export = (2 if wls else 1) * sino + slab + n * EXPORT_BYTES_PER_GAUSSIAN
    return {'seed': int(seed), 'train': int(train), 'export': int(export)}


def footprint(args, req: MachineRequest) -> Footprint:
    """The preflight's size model: the largest of the run's stages."""
    n = _n_max(args)
    stages = footprint_stages(args, req)
    peak = max(stages.values())
    return Footprint(
        persistent_gpu_bytes=peak,
        host_bytes=2 * req.sino_bytes + req.vol_bytes,
        gpu_required=True,
        # A splat step consumes a whole view, never a ray batch, so there is no
        # marginal per-sample term for `auto_rays_per_batch` to solve for.
        bytes_per_ray_sample=0,
        notes=(f"Gaussian cloud: up to {n/1e6:.2f} M primitives; peak stage "
               f"'{max(stages, key=stages.get)}' — seed "
               f"{stages['seed']/2**30:.2f} / train "
               f"{stages['train']/2**30:.2f} / export "
               f"{stages['export']/2**30:.2f} GiB (allocator peaks, "
               f"measured; nvidia-smi will show more, that is the cache).",
               "One full view per step; --rays-per-batch is not used.",
               f"Export: {DEFAULT_MAX_VOXELS/1e6:.0f} M-voxel slabs on the "
               f"device, {req.vol_bytes/2**30:.2f} GiB assembled on the host; "
               f"the slab halves itself on OOM (the extension's binning "
               f"buffer is not predictable from the slab size).",
               "A cloud checkpoint is written BEFORE the export, so a failed "
               "export never costs the training.",
               "Needs the xray_gaussian_rasterization_voxelization extension."))


#: The shared driver flags this representation re-defaults (see
#: ``LearnedAlgorithm.driver_defaults``). Together with the flag defaults in
#: `add_args` and `GaussianCloud.DEFAULT_LR_MULTIPLIERS` this IS the
#: production recipe (Scan_1510 run prod_v1, 2026-09-08): a bare
#: ``--algorithm gaussian`` reproduces it, and every entry is one flag away.
#:
#:   downsample 3      cost is per view, not per ray; ds3 holds the 0.075 mm
#:                     export grid and trains a 2.4 M cloud at ~1.4 it/s
#:   iterations 16000  the CAP only; the plateau reducer + SSIM stopper end
#:                     the run (prod_v1 used the cap: held-out SSIM was still
#:                     rising at 10 000 and the floor was never reached)
#:   eval_every 500    one held-out eval per ~6 min; the reducer's patience
#:                     counts evals, so this sets its time constant
#:   stop_metric ssim  the held-out metric a cloud is judged on; MSE turns
#:                     over earlier than the image improves
#:   withhold_eval     the diag/* projection is a true held-out one
#:   plateau 1/3, floor 0.02   levels 1, 1/3, 1/9, 1/27, 0.02 of the base
#:                     LR — three-fold cuts because a cloud's held-out curve
#:                     is flat over a halving; the grid's 0.5 / 0.125 gave
#:                     one visible stage per run
DRIVER_DEFAULTS = dict(
    downsample=3,
    iterations=16000,
    eval_every=500,
    stop_metric='ssim',
    withhold_eval=True,
    lr_plateau_factor=1.0 / 3.0,
    lr_plateau_min_fraction=0.02,
)


ALGORITHM = LearnedAlgorithm(
    name='gaussian',
    reconstructor=GaussianReconstructor,
    summary='anisotropic 3-D Gaussian splatting (X-ray radiative rasteriser)',
    default_lr=DEFAULT_LR,
    driver_defaults=DRIVER_DEFAULTS,
    add_args=add_args,
    options=options,
    footprint=footprint,
)
