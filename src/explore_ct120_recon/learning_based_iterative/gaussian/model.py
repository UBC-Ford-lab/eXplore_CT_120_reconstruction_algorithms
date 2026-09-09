"""The Gaussian cloud: parameters, activations, and adaptive density control.

A cloud of anisotropic 3-D Gaussians standing in for a voxel grid. Each
primitive carries 11 numbers — position (3), log-scale (3), rotation quaternion
(4), density (1) — against a voxel's one, but it goes where the matter is, so
the comparison that matters is primitives-per-object rather than per-domain.

WHY THE ACTIVATIONS ARE NOT NEGOTIABLE
--------------------------------------
``exp`` on scale, ``normalize`` on rotation, ``softplus`` on density: these are
what the CUDA kernels assume when they build a covariance from (scale, rotation)
and read a density. They are matched to the kernel, not chosen by us, and
changing one here silently changes what the rasteriser computes. Note the
consequence for the pipeline: softplus makes mu STRICTLY POSITIVE, so this
backend — like the voxel grid with a softplus head, and unlike FDK/SIRT — never
produces negative mu and needs no non-negativity projection.

ADAPTIVE DENSITY CONTROL, AND THE THRESHOLD THAT DOES NOT TRANSFER
-------------------------------------------------------------------
Splatting only works because the cloud is not fixed: primitives in
under-resolved regions are cloned or split, and ones that stop carrying signal
are pruned. Upstream selects them with an ABSOLUTE gradient threshold
(``densify_grad_threshold``, default 5e-5).

That default is unusable on real scanner data and fails SILENTLY. MEASURED on
Scan_1510: the maximum observed screen-space gradient was 2.04e-5, median
1.04e-7 — the entire distribution sits below the threshold, so densification
never fired once and the run was a fixed-size cloud pretending to be adaptive.
The cause is the same units mismatch as the alpha cutoff in `kernel`: gradients
live in rendered units, which depend on the domain size, the voxel pitch, the
signal scale and the loss — none of which are properties of the scan the
default was tuned on.

So the selection here is a QUANTILE by default: densify the top
``densify_fraction`` of primitives by accumulated gradient. That is invariant to
every one of those scales, transfers across scans without tuning, and cannot
silently select nothing. An absolute threshold remains available for
reproducing an upstream run.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


#: Slack on the "is it over the ceiling?" COUNT only — never on the clamp,
#: which stays exact. At k=1 the ceiling equals the seeding width by
#: construction, so in float32 rounding alone decides the comparison for about
#: half the cloud: MEASURED on a 20 k uniform seed, a bare `>` reported 42 % of
#: primitives clamped where the largest actual width change was 1.9e-6. A
#: diagnostic that reads 42 % when nothing moved is worse than no diagnostic.
CAP_REPORT_TOL = 1e-3

#: The moment-matched split (`GaussianCloud.densify_and_prune`, mode
#: 'preserving'). Children sit at +-SPLIT_OFFSET x sigma along the parent's
#: LONGEST axis, that axis shrunk by SPLIT_SHRINK, each child carrying
#: SPLIT_AMPLITUDE of the parent's amplitude. SHRINK^2 + OFFSET^2 = 1 keeps the
#: second moment along that axis and the amplitude keeps the mass, so the pair
#: renders the parent's profile to within 2.3 % of its peak (tested) — below
#: the projection noise, which is the point: a split must not be visible to the
#: data. Upstream's split is not: it places each child at a random draw from
#: the parent's covariance, shrinks all three axes by 1.6 and keeps the parent's
#: amplitude (so the pair carries half the mass). Every one of those is a
#: change the data never asked for, and where the residual is below noise the
#: data cannot undo it. MEASURED on runs ufsqlhpn/5eujeyov (Scan_1510, ds3):
#: the rings that replaced solid rib cross-sections and the filaments trailing
#: off bone are sub-0.22 mm structure whose projections are 0.4-0.6x the
#: per-pixel noise; they appear only in runs that split heavily.
SPLIT_OFFSET = 0.5
SPLIT_SHRINK = math.sqrt(1.0 - SPLIT_OFFSET ** 2)
SPLIT_AMPLITUDE = 1.0 / (2.0 * SPLIT_SHRINK)


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


class GaussianCloud(nn.Module):
    """Positions/scales/rotations/densities in NORMALISED world units.

    Everything here is in the rasteriser's frame: positions in [-1, 1]^3 and
    density in rendered units. Conversion to mm and mm^-1 happens at the
    package boundary (`reconstructor`), never inside the loop, so there is one
    place where units change and it is not this one.
    """

    def __init__(self, xyz, scaling, rotation, density,
                 density_floor: float = 0.0):
        super().__init__()
        self._xyz = nn.Parameter(torch.as_tensor(xyz, dtype=torch.float32))
        self._scaling = nn.Parameter(torch.log(
            torch.as_tensor(scaling, dtype=torch.float32).clamp_min(1e-9)))
        self._rotation = nn.Parameter(torch.as_tensor(rotation,
                                                      dtype=torch.float32))
        # SIGNED DENSITY. The amplitude is ``softplus(raw) - density_floor``:
        # smooth and monotone as before, but able to go as low as -floor.
        # Zero floor is the strictly positive cloud. WHY A FLOOR AT ALL: a
        # sum of positive Gaussians is a non-negative measure convolved with
        # the blob, so an edge can never undershoot — the tails that leak
        # across it have nothing to cancel them — and air, a noisy zero,
        # can only be represented on one side (MEASURED +34 HU on Scan_1510).
        # WHY BOUNDED: unbounded signs let a +A and a -A primitive cancel in
        # every view for free, which is the null space the noise gate exists
        # to keep out. The floor is a small fraction of a typical amplitude
        # (the edge lobe needs no more), set by the caller from the seed.
        # The rasteriser must ALSO let negatives through: its kernels skip
        # |contribution| < 1e-5 (patched from `alpha < 1e-5`, which dropped
        # every negative in the forward AND the backward pass).
        self.register_buffer('density_floor',
                             torch.tensor(float(density_floor)))
        self._density = nn.Parameter(self._to_raw(
            torch.as_tensor(density, dtype=torch.float32)))
        self.register_buffer('grad_accum', torch.zeros(len(self._xyz)))
        self.register_buffer('grad_denom', torch.zeros(len(self._xyz)))
        self.register_buffer('max_radii', torch.zeros(len(self._xyz)))
        # Per-primitive width ceiling, in the same normalised units as
        # `scaling`. Zero means uncapped, which is what an unconfigured cloud
        # is; `update_spacing_cap` fills it from the cloud's own local density.
        self.register_buffer('max_sigma', torch.zeros(len(self._xyz)))
        # Noise-gate statistics (see `record_null_gradients`): per visit, the
        # data gradient's norm minus the norm the same view produces against a
        # sign-scrambled residual; summed, and summed squared.
        self.register_buffer('gate_sum', torch.zeros(len(self._xyz)))
        self.register_buffer('gate_sq', torch.zeros(len(self._xyz)))
        self._null_norm: torch.Tensor | None = None

    # -- what the kernel reads ---------------------------------------------
    @property
    def xyz(self):
        return self._xyz

    @property
    def scaling(self):
        return torch.exp(self._scaling)

    @property
    def rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def density(self):
        return torch.nn.functional.softplus(self._density) - self.density_floor

    def _to_raw(self, density: torch.Tensor) -> torch.Tensor:
        """The raw parameter that renders ``density`` under the current floor."""
        return _inverse_softplus(
            (density.to(torch.float32) + self.density_floor).clamp_min(1e-8))

    @torch.no_grad()
    def set_density_floor(self, floor: float) -> None:
        """Change the floor WITHOUT changing any amplitude (re-parameterise)."""
        d = self.density.detach().clone()
        self.density_floor.fill_(float(floor))
        self._density.copy_(self._to_raw(d))

    def __len__(self):
        return int(self._xyz.shape[0])

    # -- optimiser ----------------------------------------------------------
    #: Upstream's relative rates, as multiples of ``--lr``. Defaults, not
    #: constants: MEASURED on Scan_1510 ds6, raising all four together by 10x
    #: took the median narrowest axis from 0.627 to 0.488 mm and the anisotropy
    #: from 1.22 to 1.62 (+2.5 dB) — but "all four together" cannot say WHICH
    #: mattered, and the two that set resolution are the two carrying the least
    #: rate. Scale is the width itself; rotation is what lets a primitive lie
    #: flat ON a surface instead of stretching along the world axes, and it sits
    #: at 1.0, the lowest of the four. Both are now reachable on their own.
    #: Production values (Scan_1510 prod_v1): scaling 100 and rotation 30 in
    #: place of upstream's 5 and 1, MEASURED as the two that buy resolution —
    #: the best held-out SSIM, bone FWHM and edge width of every run so far.
    DEFAULT_LR_MULTIPLIERS = {'xyz': 0.16, 'density': 100.0,
                              'scaling': 100.0, 'rotation': 30.0}
    #: The splatting literature's relative rates, for a like-for-like run.
    UPSTREAM_LR_MULTIPLIERS = {'xyz': 0.16, 'density': 100.0,
                               'scaling': 5.0, 'rotation': 1.0}

    def param_groups(self, lr, multipliers: dict | None = None):
        """One group per parameter kind.

        Position, scale, rotation and density are on four different scales and
        a single rate trains at least three of them wrong. ``multipliers``
        overrides `DEFAULT_LR_MULTIPLIERS` per group; unnamed groups keep the
        default, so a caller can raise one without restating the rest.
        """
        mul = dict(self.DEFAULT_LR_MULTIPLIERS)
        unknown = set(multipliers or ()) - set(mul)
        if unknown:
            raise ValueError(f"unknown LR group(s) {sorted(unknown)}; "
                             f"expected {sorted(mul)}")
        mul.update(multipliers or {})
        return [
            {'params': [self._xyz], 'lr': lr * mul['xyz'], 'name': 'xyz'},
            {'params': [self._density], 'lr': lr * mul['density'],
             'name': 'density'},
            {'params': [self._scaling], 'lr': lr * mul['scaling'],
             'name': 'scaling'},
            {'params': [self._rotation], 'lr': lr * mul['rotation'],
             'name': 'rotation'},
        ]

    # -- shape constraints --------------------------------------------------
    @torch.no_grad()
    def update_spacing_cap(self, k: float,
                           fresh: torch.Tensor | None = None) -> dict:
        """Set each primitive's width ceiling to ``k`` x its local spacing.

        ``fresh`` (bool, one per primitive) marks primitives that keep the
        ceiling they already carry instead of being re-measured. Children of a
        moment-matched split are born ON TOP of each other, so the spacing
        measured right after the split counts a sibling as a neighbour and
        would clamp the pair narrower than the profile they were built to
        preserve — hollowing by another route. Their parent's ceiling, which
        they inherit, is the honest value until they have moved.

        WHY A CEILING AT ALL, AND WHY RELATIVE TO SPACING — MEASURED ON RUN
        83m5xh1e (Scan_1510, ds3/75 um), inside the export ROI:

            population            n        spacing    FWHM     FWHM/spacing
            soft tissue     185,708      0.364 mm   2.338 mm       6.42
            mid              95,835      0.255 mm   0.248 mm       0.97
            bone            459,025      0.074 mm   0.140 mm       1.90

        The low-contrast population is not short of primitives. It is spaced
        at 0.364 mm — which supports 0.364 mm resolution — while each member is
        6.4x wider than the gap to its own neighbours. Summed, those Gaussians
        occupy 3,362,414 mm^3 inside a 36,961 mm^3 ROI: a 91x overlapping,
        overcomplete basis whose ENSEMBLE is blurred even though every
        individual primitive could shrink without needing a single new one.

        Nothing in the data term asks them to. Blur in a 91x-redundant basis is
        a COLLECTIVE coordinate: one primitive narrowing while its neighbours
        stay wide does not change the sum, and the sum is all that is measured.
        So the per-primitive gradient cannot see it, and the widths drift
        wherever the data term's mild preference for blur takes them — MEASURED,
        the soft population went from a 0.53 mm seed OUT to 0.99 mm while bone
        went in to 0.059 mm.

        That makes it a constraint on the solution set rather than a penalty,
        which is why it belongs here beside `project_scales` and not in the
        loss. Relative to LOCAL spacing because the correct ceiling is not
        global: the same absolute number is simultaneously too tight for the
        bed and far too loose for the skeleton. At ``k = 1`` this reproduces the
        seeding rule — `seeding.local_spacing` sets sigma to exactly the local
        spacing — so the cap says a primitive may never be wider than the seed
        would have made it AT ITS CURRENT local density. Measured healthy range
        in the cloud above is sigma/spacing 0.4-0.8, so k = 1 passes everything
        already well conditioned and bites only the redundant population.

        Returns a stats dict; a no-op (and a zeroed cap) when ``k <= 0``.
        """
        n = len(self)
        if k <= 0 or n < 2:
            self.max_sigma = torch.zeros(n, device=self._xyz.device)
            return {'cap_k': float(k), 'cap_median': 0.0, 'cap_n': 0}
        from .seeding import local_spacing
        pos = self._xyz.detach().float().cpu().numpy()
        spacing = local_spacing(pos)
        if spacing is None:                    # scipy missing: leave uncapped
            self.max_sigma = torch.zeros(n, device=self._xyz.device)
            return {'cap_k': float(k), 'cap_median': 0.0, 'cap_n': 0}
        cap = torch.as_tensor(spacing, dtype=torch.float32,
                              device=self._xyz.device) * float(k)
        if fresh is not None and self.max_sigma.numel() == n:
            fresh = fresh.to(cap.device)
            cap = torch.where(fresh & (self.max_sigma > 0), self.max_sigma, cap)
        self.max_sigma = cap
        over = int((torch.exp(self._scaling).max(dim=1).values
                    > cap * (1.0 + CAP_REPORT_TOL)).sum())
        return {'cap_k': float(k), 'cap_median': float(cap.median()),
                'cap_n': over}

    @torch.no_grad()
    def project_scales(self, min_sigma: float, max_aspect: float,
                       spacing_cap: bool = True) -> dict:
        """Project the widths onto what the MEASUREMENT can determine.

        Applied after every optimiser step, in the same place and for the same
        reason the voxel backend applies its non-negativity projection: it is a
        constraint on the solution set, not a penalty traded off against the
        data term.

        WHY THIS IS NEEDED — MEASURED ON RUN liztmby9 (Scan_1510, ds3/75 um).
        The projection loss is blind to shape. A round primitive casts the same
        shadow from every angle; a NEEDLE does not — end-on it is a bright dot,
        side-on a faint smear — so it is an angle-selective correction, and
        greedy optimisation reaches for exactly that when squeezing the last dB
        out of a residual that is mostly noise. The result held out at 40.29 dB
        against a 36.24 dB noise ceiling, the best fit any backend has produced
        on this scan, while delivering the WORST of five matched volumes: soft
        tissue flattened to a wash, ribs rendered as slivers, soft-tissue skew
        +0.86 where every other method is negative. The delivered cloud had
        median aspect 7.75, 25.9 % above 20, 6.7 % above 50, longest axis
        pressed against the `max_scale` clamp at 8.00 mm, and needles aligned
        with the ROTATION AXIS at 3.7x chance (49 % within 30 degrees of z) —
        the direction the circular orbit constrains least, since every source
        position lies in the z = 0 plane.

        Two constraints, both statements about identifiability rather than
        taste:

        ``min_sigma``  nothing finer than the grid the volume is delivered on.
            The run put half its cloud an order of magnitude below the scale
            the scan can resolve (p50 thin axis 43 um, p1 1.2 um against a
            ~0.42 mm resolvable sigma). No data exists to determine those, so
            allowing them only buys overfitting.
        ``max_aspect``  bounded elongation. Anisotropy is what makes splatting
            efficient — a pancake gives the edge sharpness of its thin axis at
            the coverage of its wide ones — so this bounds it rather than
            removing it. At 4 a primitive still covers ~16x the area of an
            isotropic one of the same thinness.
        ``max_sigma``   nothing wider than its own neighbourhood, from the
            `max_sigma` buffer when ``spacing_cap`` is set. See
            `update_spacing_cap` for what this is for; briefly, a redundant
            basis can be blurred without any single primitive being wrong, and
            no per-primitive gradient can see that.

        Order matters. Floor first, then cap the ratio against the FLOORED
        minimum, so a needle has its thin axis raised AND its long axis pulled
        in rather than being made merely long and legal. The width ceiling goes
        LAST and only ever lowers the largest axes, so it cannot reintroduce
        the elongation the aspect cap just removed — clamping the top of a
        sorted triple can only shrink its ratio. Against the floor the ceiling
        yields: `min_sigma` is what the SCAN can resolve, so a neighbourhood
        denser than that does not license a primitive finer than the data.

        Note this also breaks a ratchet in `densify_and_prune`: splitting
        divides all three axes by the same factor, so it preserves aspect
        exactly and a needle splits into two smaller needles forever.
        """
        use_ceiling = bool(spacing_cap) and bool((self.max_sigma > 0).any())
        if min_sigma <= 0 and max_aspect <= 1 and not use_ceiling:
            return {'clamped_thin': 0, 'clamped_aspect': 0, 'clamped_wide': 0}
        s = torch.exp(self._scaling)
        n_thin = n_aspect = n_wide = 0
        if min_sigma > 0:
            n_thin = int((s < min_sigma).any(dim=1).sum())
            s = s.clamp_min(float(min_sigma))
        if max_aspect > 1:
            cap = s.min(dim=1, keepdim=True).values * float(max_aspect)
            n_aspect = int((s > cap).any(dim=1).sum())
            s = torch.minimum(s, cap)
        if use_ceiling:
            # A zero entry means "not measured": leave those alone rather than
            # collapsing them, and never push below what the scan can resolve.
            ceil = self.max_sigma.clamp_min(float(max(min_sigma, 0.0)))
            ceil = torch.where(self.max_sigma > 0, ceil,
                               torch.full_like(ceil, float('inf')))
            ceil = ceil.unsqueeze(1)
            n_wide = int((s > ceil * (1.0 + CAP_REPORT_TOL)).any(dim=1).sum())
            s = torch.minimum(s, ceil)
        self._scaling.copy_(torch.log(s))
        return {'clamped_thin': n_thin, 'clamped_aspect': n_aspect,
                'clamped_wide': n_wide}

    # -- best-iterate snapshot ---------------------------------------------
    def snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in (
            ('xyz', self._xyz), ('scaling', self._scaling),
            ('rotation', self._rotation), ('density', self._density),
            ('density_floor', self.density_floor))}

    @torch.no_grad()
    def restore(self, snap: dict) -> None:
        """Reinstate a snapshot, resizing the parameters if the cloud grew.

        Densification means the best iterate may have had a different number of
        primitives than the current one, so this cannot be a copy_.
        """
        dev = self._xyz.device
        self._xyz = nn.Parameter(snap['xyz'].to(dev))
        self._scaling = nn.Parameter(snap['scaling'].to(dev))
        self._rotation = nn.Parameter(snap['rotation'].to(dev))
        self._density = nn.Parameter(snap['density'].to(dev))
        # A snapshot without a floor predates signed density: floor 0.
        self.density_floor = snap.get(
            'density_floor', torch.tensor(0.0)).to(dev).clone()
        n = len(self._xyz)
        self.grad_accum = torch.zeros(n, device=dev)
        self.grad_denom = torch.zeros(n, device=dev)
        self.max_radii = torch.zeros(n, device=dev)
        self.gate_sum = torch.zeros(n, device=dev)
        self.gate_sq = torch.zeros(n, device=dev)
        self._null_norm = None
        # The restored cloud has its own positions, so any ceiling measured on
        # the discarded one describes a different neighbourhood. Zeroing is the
        # honest state — uncapped until re-measured — and matters because
        # restore() runs at the END of training, where a stale per-primitive
        # ceiling would be applied to primitives it was never measured for.
        self.max_sigma = torch.zeros(n, device=dev)

    # -- adaptive density control ------------------------------------------
    @torch.no_grad()
    def record_null_gradients(self, screenspace, visible) -> None:
        """Record what the SAME view's gradient looks like against noise.

        The caller backpropagates the loss against a null target — the current
        residual with its magnitudes kept and its signs scrambled — before the
        real backward, and hands the screen-space gradient here. The next
        `record_gradients` call pairs it with the data gradient of the same
        view.

        WHY. Upstream's densification statistic is the mean NORM of the
        screen-space gradient over visits, and a primitive sitting in noise has
        a norm just like one sitting on a real residual: with an L1 loss the
        gradient is sum_px sign(r_px) J_px, a random walk over its footprint
        whose length is sqrt(sum J^2) whatever the residual is. So a fixed
        quantile keeps splitting noise-driven primitives for as long as the
        window is open. A real residual has a consistent sign across the
        footprint and the sum grows like sum |J| instead — larger by up to
        sqrt(footprint pixels). The null gradient measures the random-walk
        length for THIS primitive in THIS view, so the difference tests the one
        thing that matters: is the data asking for more resolution here, or is
        it silent? See `gate_z`.
        """
        if screenspace.grad is None:
            return
        g = torch.zeros(len(self), device=screenspace.grad.device)
        g[visible] = torch.norm(screenspace.grad[visible, :2], dim=-1)
        self._null_norm = g

    @torch.no_grad()
    def record_gradients(self, screenspace, visible, radii) -> None:
        """Accumulate the densification statistics for one rendered view."""
        if screenspace.grad is None:
            return
        g = torch.norm(screenspace.grad[visible, :2], dim=-1)
        self.grad_accum[visible] += g
        self.grad_denom[visible] += 1
        self.max_radii[visible] = torch.maximum(self.max_radii[visible],
                                                radii[visible].float())
        if self._null_norm is not None:
            d = g - self._null_norm[visible]
            self.gate_sum[visible] += d
            self.gate_sq[visible] += d * d
            self._null_norm = None

    @torch.no_grad()
    def gate_z(self) -> torch.Tensor:
        """Per-primitive z-score of (data gradient - null gradient) over the
        visits since the last densification round.

        With d_i the per-visit difference, ``z = sum d / sqrt(sum d^2)``. Under
        the null (the residual in the primitive's footprint is noise) d has
        zero mean and z ~ N(0, 1) for the ~100 visits of a round; where the
        data consistently pull harder than noise would, z grows like sqrt(n).
        The statistic is in units of the primitive's own noise, so the
        threshold (`densify_and_prune(gate_z=...)`, default 3) carries no
        scale and transfers across scans, losses and signal scales. Zero where
        nothing was recorded.
        """
        z = self.gate_sum / self.gate_sq.clamp_min(1e-30).sqrt()
        return torch.where(self.gate_sq > 0, z, torch.zeros_like(z))

    # -- MCMC density control (Kheradmand et al., NeurIPS 2024) -------------
    # The cloud as SAMPLES from the density it represents. Dead primitives are
    # not pruned but MOVED onto live ones, chosen in proportion to the mass
    # they carry, with the target's amplitude shared out so the render does not
    # change; a Langevin noise term lets near-dead primitives wander until the
    # data claim them. The count is fixed, which makes it what the blob-basis
    # literature says it should be: the resolution budget. Upstream's rules
    # are for alpha-compositing; for an ADDITIVE X-ray model, N copies at one
    # place each carrying 1/N of the amplitude render exactly the original.
    @torch.no_grad()
    def mass(self) -> torch.Tensor:
        """Per-primitive integral of the rendered density, up to (2 pi)^1.5."""
        return self.density.squeeze(-1) * self.scaling.prod(dim=1)

    @torch.no_grad()
    def relocate_dead(self, optimizer, *, dead: torch.Tensor,
                      targets_ok: torch.Tensor | None = None) -> dict:
        """Move every ``dead`` primitive onto a live one.

        Targets are drawn with replacement, probability proportional to mass,
        from the live primitives (and ``targets_ok`` where given, so the
        export ROI can be the only place mass is sent). A target that receives
        k newcomers becomes a group of N = k + 1 members that together render
        what it rendered: the moment-matched spread of `densify_and_prune`'s
        preserving split, generalised — members evenly spaced over
        +-SPLIT_OFFSET sigma along the longest axis, that axis shrunk to keep
        the second moment, amplitude a / N x (sigma / sigma_new) to keep the
        mass. Adam's moments are reset for everyone involved, as upstream does.

        WHY A SPREAD AND NOT A PILE. Upstream stacks the newcomers exactly on
        the target; its rasteriser has no notion of spacing. Ours caps every
        width at k x the kNN spacing, and a pile at 0.1 sigma collapses the
        spacing of the target and, next round, of the newcomers, so the
        ceiling clamps the whole group to a fraction of its width — the group
        is killed by the constraint, dies again, is moved again. MEASURED on
        run 228qo6kn (2.4 M, Scan_1510): dead fraction ramped 0.09 -> 0.7 %
        per round, up to 631 k primitives clamped in ONE iteration as piles
        dragged the median spacing down, held-out error rose from 8.3e-5 at
        3000 to 1.0e-4 at 8000 while the same recipe without MCMC improved.
        Spread at sigma apart, siblings are ordinary neighbours.

        Returns a stats dict including ``moved`` (bool mask: targets and
        newcomers), which the caller should pass to `update_spacing_cap` as
        ``fresh``.
        """
        dead = dead.to(self._xyz.device)
        n_dead = int(dead.sum())
        live = ~dead
        if targets_ok is not None:
            live &= targets_ok.to(live.device)
        n_live = int(live.sum())
        if n_dead == 0 or n_live == 0:
            return {'relocated': 0, 'targets': 0, 'dead': n_dead}
        live_idx = torch.nonzero(live).squeeze(-1)
        probs = self.mass()[live_idx].clamp_min(0.0)
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        pick = torch.multinomial(probs, n_dead, replacement=True)
        tgt = live_idx[pick]                                   # (n_dead,)
        dead_idx = torch.nonzero(dead).squeeze(-1)
        # Group bookkeeping: member j of target t's group of N_t. The target
        # itself is member 0; newcomers are numbered by their order of arrival.
        order = torch.argsort(tgt)
        tgt_sorted = tgt[order]
        dead_sorted = dead_idx[order]
        counts = torch.bincount(tgt, minlength=len(self))          # k per target
        first = torch.searchsorted(tgt_sorted, tgt_sorted)         # first slot
        rank = torch.arange(len(tgt_sorted), device=tgt.device) - first + 1
        N_all = (counts + 1).float()                               # per primitive
        touched = torch.unique(tgt)
        N_t = N_all[tgt_sorted]                                    # per newcomer
        # Even spread over [-OFFSET, +OFFSET] sigma along the longest axis:
        # u_j = (j - (N-1)/2) / ((N-1)/2) * OFFSET, so N = 2 gives +-OFFSET.
        u_new = ((rank.float() - (N_t - 1) / 2) / ((N_t - 1) / 2).clamp_min(1e-6)
                 * SPLIT_OFFSET)
        u_tgt = ((0.0 - (N_all[touched] - 1) / 2)
                 / ((N_all[touched] - 1) / 2).clamp_min(1e-6) * SPLIT_OFFSET)
        # Second moment of the spread, var(u) = OFFSET^2 (N+1) / (3 (N-1)),
        # so the long axis shrinks by sqrt(1 - var) and the amplitude grows
        # back by 1/shrink to conserve the mass.
        def shrink(N):
            var = SPLIT_OFFSET ** 2 * (N + 1) / (3.0 * (N - 1))
            return torch.sqrt((1.0 - var).clamp_min(0.25))
        s_t = self.scaling[touched]
        axis_t = F.one_hot(s_t.argmax(dim=1), 3).to(s_t.dtype)
        long_t = (s_t * axis_t).sum(dim=1)
        sh_t = shrink(N_all[touched])
        amp_t = self.density[touched].squeeze(-1) / N_all[touched] / sh_t
        R_t = _quat_to_rot(self.rotation[touched])
        # A lookup from target index to its row in `touched`.
        row = torch.full((len(self),), -1, dtype=torch.long, device=tgt.device)
        row[touched] = torch.arange(len(touched), device=tgt.device)
        r_new = row[tgt_sorted]
        # Newcomers: the target's shape with the long axis shrunk, spread out.
        xyz_t = self._xyz[touched]
        new_s = s_t * (1.0 - axis_t) + s_t * axis_t * sh_t.unsqueeze(-1)
        self._xyz[dead_sorted] = xyz_t[r_new] + torch.bmm(
            R_t[r_new], (axis_t[r_new] * (u_new * long_t[r_new]).unsqueeze(-1)
                         ).unsqueeze(-1)).squeeze(-1)
        self._scaling[dead_sorted] = torch.log(new_s[r_new])
        self._rotation[dead_sorted] = self._rotation[touched][r_new]
        self._density[dead_sorted] = self._to_raw(amp_t[r_new].unsqueeze(-1))
        self.max_sigma[dead_sorted] = self.max_sigma[touched][r_new]
        # The target: member 0 of its own group.
        self._xyz[touched] = xyz_t + torch.bmm(
            R_t, (axis_t * (u_tgt * long_t).unsqueeze(-1)).unsqueeze(-1)
        ).squeeze(-1)
        self._scaling[touched] = torch.log(new_s)
        self._density[touched] = self._to_raw(amp_t.unsqueeze(-1))
        moved = torch.zeros(len(self), dtype=torch.bool, device=dead.device)
        moved[dead_idx] = True
        moved[touched] = True
        self._reset_moments(optimizer, moved)
        for buf in (self.grad_accum, self.grad_denom, self.max_radii,
                    self.gate_sum, self.gate_sq):
            buf[moved] = 0.0
        return {'relocated': n_dead, 'targets': int(touched.numel()),
                'dead': n_dead, 'moved': moved}

    @torch.no_grad()
    def add_position_noise(self, *, amount: float, dead_score: torch.Tensor,
                           k: float = 100.0) -> int:
        """Langevin exploration for the near-dead: a step of ``amount`` x
        sigma (per axis, in the primitive's own frame) gated by
        ``sigmoid(k * (1 - dead_score))``, so a primitive at or below the dead
        line (``dead_score`` <= 1, its mass over the dead threshold) walks
        and a live one (score >> 1) does not. Upstream gates on opacity with
        the same k. Returns how many primitives received a visible step."""
        if amount <= 0:
            return 0
        gate = torch.sigmoid(float(k) * (1.0 - dead_score.to(self._xyz.device)))
        s = self.scaling
        R = _quat_to_rot(self.rotation)
        eta = torch.randn_like(s) * s * (float(amount) * gate.unsqueeze(-1))
        self._xyz += torch.bmm(R, eta.unsqueeze(-1)).squeeze(-1)
        return int((gate > 0.5).sum())

    @torch.no_grad()
    def _reset_moments(self, optimizer, which: torch.Tensor) -> None:
        for group in optimizer.param_groups:
            if group.get('name') not in self._tensors():
                continue        # non-cloud groups (photometric, motion) have no primitive axis
            p = group['params'][0]
            state = optimizer.state.get(p, None)
            if state is None:
                continue
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key][which] = 0.0

    @torch.no_grad()
    def inside_box(self, lo, hi) -> torch.Tensor:
        """Boolean mask of primitives whose CENTRE lies in the axis-aligned
        box ``[lo, hi]`` (normalised world units, 3-vectors)."""
        lo = torch.as_tensor(lo, dtype=torch.float32, device=self._xyz.device)
        hi = torch.as_tensor(hi, dtype=torch.float32, device=self._xyz.device)
        x = self._xyz.detach()
        return ((x >= lo) & (x <= hi)).all(dim=1)

    def densify_and_prune(self, optimizer, *, fraction, absolute, split_scale,
                          min_density, max_scale, max_count,
                          eligible: torch.Tensor | None = None,
                          split_mode: str = 'preserving',
                          gate_z: float | None = None,
                          min_sigma: float = 0.0) -> dict:
        """One round of clone / split / prune. Returns a small stats dict.

        ``eligible`` (bool, one per primitive) confines clone and split to a
        subset; the gradient quantile is still taken over everything seen, so
        the threshold means the same thing with or without it. Pruning is
        never confined.

        ``gate_z`` vetoes primitives whose data gradient is not distinguishable
        from their noise gradient (`gate_z()` below the threshold), so a round
        can only refine where the data decide. The quantile stays as the RATE
        limit, taken over the excess gradient (data minus null) while the gate
        is live and over the raw norm otherwise; the gate is a veto on top of
        it, and densification ends on its own when nothing passes —
        `stats['gated']` counts what did. Needs `record_null_gradients` to
        have been fed; without it every primitive passes and the ranking is
        upstream's.

        ``split_mode``:
          'preserving'  every selected primitive is replaced by the
              moment-matched pair (see `SPLIT_OFFSET`), whichever size it is.
              The pair renders what the parent rendered, so resolution grows
              only where the optimiser pulls the children apart. A clone that
              preserved the render (two copies at half amplitude) would receive
              identical gradients forever, so there is no clone in this mode.
              Primitives whose longest axis is already at the resolution floor
              (``min_sigma``, same units as the scales) are left alone: a child
              would only be clamped back up, and there is nothing finer to
              find. `stats['floored']` counts them.
          'random'  upstream's rule, kept for reproducing earlier runs. Clone
              vs split by size: a high-gradient primitive that is SMALL is
              under-populated, so copy it; one that is LARGE is under-resolved,
              so break it into two narrower ones drawn from its own
              distribution.
        """
        if split_mode not in ('preserving', 'random'):
            raise ValueError(f"split_mode must be 'preserving' or 'random', "
                             f"got {split_mode!r}")
        n0 = len(self)
        grads = self.grad_accum / self.grad_denom.clamp_min(1.0)
        grads[self.grad_denom == 0] = 0.0
        gate_on = gate_z is not None and bool((self.gate_sq > 0).any())
        if gate_on:
            # Rank by the EXCESS over noise, not by the raw norm. MEASURED on
            # run pgoulwdm (Scan_1510, ds3): the top-2 % by norm were almost
            # entirely noise-inflated bone primitives — 95 % of the cloud
            # passed the gate, yet splits per round fell from 5,840 to 370
            # because the ones the quantile handed to the gate were the ones
            # it vetoed. The norm is sum_px |sign(r) J| and grows with a
            # primitive's contrast whether or not the residual is real; the
            # excess is the part noise does not explain.
            grads = (self.gate_sum / self.grad_denom.clamp_min(1.0)).clamp_min(0.0)
            grads[self.grad_denom == 0] = 0.0

        if absolute is not None:
            thresh = float(absolute)
        else:
            seen = grads[self.grad_denom > 0]
            thresh = (float(torch.quantile(seen.float(),
                                           1.0 - float(fraction)))
                      if seen.numel() else float('inf'))

        selected = grads >= thresh
        if eligible is not None:
            selected &= eligible.to(selected.device)
        n_gated = -1
        if gate_on:
            coherent = self.gate_z() >= float(gate_z)
            seen = coherent & (self.grad_denom > 0)
            if eligible is not None:
                seen &= eligible.to(seen.device)
            n_gated = int(seen.sum())          # of the eligible ones
            selected &= coherent
        scales = self.scaling
        n_floored = 0
        if split_mode == 'preserving':
            if min_sigma > 0:
                floored = (scales.max(dim=1).values * SPLIT_SHRINK
                           < float(min_sigma))
                n_floored = int((selected & floored).sum())
                selected &= ~floored
            big = torch.ones_like(selected)        # everything is a split
        else:
            big = scales.max(dim=1).values > split_scale
        room = max(0, int(max_count) - n0)
        clone_mask = selected & ~big
        split_mask = selected & big
        if room <= 0:
            clone_mask = torch.zeros_like(clone_mask)
            split_mask = torch.zeros_like(split_mask)
        elif int(clone_mask.sum() + split_mask.sum()) > room:
            # Keep the strongest candidates rather than an arbitrary prefix.
            cand = torch.nonzero(clone_mask | split_mask).squeeze(-1)
            keep = cand[torch.argsort(grads[cand], descending=True)[:room]]
            m = torch.zeros_like(clone_mask)
            m[keep] = True
            clone_mask &= m
            split_mask &= m

        new = {'xyz': [], 'scaling': [], 'rotation': [], 'density': []}
        # Ceilings for the primitives about to be created, in the SAME order
        # they are appended. A child sits where its parent sat, so the parent's
        # ceiling is the best estimate available until the next re-measurement;
        # inheriting it keeps every new primitive constrained from its first
        # step instead of running uncapped until then.
        new_cap = []
        if bool(clone_mask.any()):
            new['xyz'].append(self._xyz[clone_mask])
            new['scaling'].append(self._scaling[clone_mask])
            new['rotation'].append(self._rotation[clone_mask])
            new['density'].append(self._density[clone_mask])
            new_cap.append(self.max_sigma[clone_mask])
        if bool(split_mask.any()):
            idx = torch.nonzero(split_mask).squeeze(-1)
            s = scales[idx]
            R = _quat_to_rot(self.rotation[idx])
            if split_mode == 'preserving':
                # Along the longest local axis only: +-OFFSET sigma apart,
                # that axis shrunk, the other two untouched, amplitude split
                # so that the pair sums to the parent (see SPLIT_OFFSET).
                axis = F.one_hot(s.argmax(dim=1), 3).to(s.dtype)
                half = (s * axis).sum(dim=1, keepdim=True) * SPLIT_OFFSET
                offset = torch.bmm(R, (axis * half).unsqueeze(-1)).squeeze(-1)
                child_s = s * (1.0 - axis) + s * axis * SPLIT_SHRINK
                child_d = self._to_raw(self.density[idx] * SPLIT_AMPLITUDE)
            else:
                # Offset each child by a sample from the parent's own
                # covariance, rotated into world frame, and shrink both.
                noise = torch.randn((len(idx), 3), device=s.device) * s
                offset = torch.bmm(R, noise.unsqueeze(-1)).squeeze(-1)
                child_s = s / split_scale_factor()
                child_d = self._density[idx]
            for sign in (1.0, -1.0):
                new['xyz'].append(self._xyz[idx] + sign * offset)
                new['scaling'].append(torch.log(child_s))
                new['rotation'].append(self._rotation[idx])
                new['density'].append(child_d)
                new_cap.append(self.max_sigma[idx])

        if any(new.values()):
            self._append(optimizer, {k: torch.cat(v) for k, v in new.items()
                                     if v})
            if new_cap:
                tail = torch.cat(new_cap)
                self.max_sigma[-len(tail):] = tail
        # Split parents are replaced by their children.
        n_now = len(self)
        drop = torch.zeros(n_now, dtype=torch.bool, device=self._xyz.device)
        drop[:n0] = split_mask
        # A near-zero primitive of EITHER sign is dead weight.
        drop |= (self.density.squeeze(-1).abs() < min_density)
        drop |= (self.scaling.max(dim=1).values > max_scale)
        if bool(drop.any()) and int((~drop).sum()) > 0:
            self._prune(optimizer, ~drop)

        n_new = int(clone_mask.sum()) + 2 * int(split_mask.sum())
        self.grad_accum.zero_()
        self.grad_denom.zero_()
        self.max_radii.zero_()
        self.gate_sum.zero_()
        self.gate_sq.zero_()
        self._null_norm = None
        return {'before': n0, 'after': len(self),
                'cloned': int(clone_mask.sum()), 'split': int(split_mask.sum()),
                'pruned': int(drop.sum()), 'threshold': float(thresh),
                'eligible': (n0 if eligible is None else int(eligible.sum())),
                # -1 = no null gradients were recorded, so the gate was idle.
                'gated': n_gated, 'floored': n_floored,
                # The children are the LAST n_new primitives (pruning keeps
                # order); `update_spacing_cap(fresh=...)` wants to know.
                'n_new': n_new}

    # -- optimiser-state surgery -------------------------------------------
    # Growing or shrinking a parameter invalidates Adam's moments for it. They
    # have to be carried across explicitly: dropping them would restart every
    # surviving primitive's momentum at every densification, and leaving them
    # stale would index the wrong rows.
    def _tensors(self):
        return {'xyz': '_xyz', 'density': '_density', 'scaling': '_scaling',
                'rotation': '_rotation'}

    @torch.no_grad()
    def _append(self, optimizer, extra: dict) -> None:
        for group in optimizer.param_groups:
            name = group['name']
            if name not in extra:
                continue
            old = group['params'][0]
            add = extra[name]
            state = optimizer.state.get(old, None)
            new = nn.Parameter(torch.cat([old.data, add], dim=0))
            if state is not None:
                z = torch.zeros_like(add)
                state['exp_avg'] = torch.cat([state['exp_avg'], z], dim=0)
                state['exp_avg_sq'] = torch.cat([state['exp_avg_sq'], z], dim=0)
                del optimizer.state[old]
                optimizer.state[new] = state
            group['params'][0] = new
            setattr(self, self._tensors()[name], new)
        n = len(self._xyz)
        dev = self._xyz.device
        pad = n - self.grad_accum.numel()
        if pad > 0:
            z = torch.zeros(pad, device=dev)
            self.grad_accum = torch.cat([self.grad_accum, z])
            self.grad_denom = torch.cat([self.grad_denom, z])
            self.max_radii = torch.cat([self.max_radii, z])
            self.gate_sum = torch.cat([self.gate_sum, z])
            self.gate_sq = torch.cat([self.gate_sq, z])
            # Zero reads as "not measured" and so leaves a new primitive
            # uncapped; `densify_and_prune` overwrites this tail with the
            # parents' ceilings straight away, and the next
            # `update_spacing_cap` re-measures all of them.
            self.max_sigma = torch.cat([self.max_sigma, z])

    @torch.no_grad()
    def _prune(self, optimizer, keep: torch.Tensor) -> None:
        for group in optimizer.param_groups:
            name = group['name']
            if name not in self._tensors():
                continue
            old = group['params'][0]
            state = optimizer.state.get(old, None)
            new = nn.Parameter(old.data[keep])
            if state is not None:
                state['exp_avg'] = state['exp_avg'][keep]
                state['exp_avg_sq'] = state['exp_avg_sq'][keep]
                del optimizer.state[old]
                optimizer.state[new] = state
            group['params'][0] = new
            setattr(self, self._tensors()[name], new)
        self.grad_accum = self.grad_accum[keep]
        self.grad_denom = self.grad_denom[keep]
        self.max_radii = self.max_radii[keep]
        self.max_sigma = self.max_sigma[keep]
        self.gate_sum = self.gate_sum[keep]
        self.gate_sq = self.gate_sq[keep]


def split_scale_factor() -> float:
    """How much narrower each child of a split is than its parent."""
    return 1.6


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
