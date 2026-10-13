"""The motion model: a per-view state code and a low-rank deformation basis.

WHY A LATENT STATE AND NOT A CLOCK. The scans this is for are gated
step-and-shoot acquisitions: every projection is shot after its own breath
trigger, so consecutive views are separate breaths and wall-clock time between
them says nothing about the anatomy. What varies per view is the residual
respiratory amplitude inside the gate window and the (unsynchronised) cardiac
phase — a small vector per view, estimated from the data. That vector is the
"time" coordinate here: the free part of ``code(v)`` for view ``v``.

WHY A CONTROL GRID. The displacement of a primitive in view ``v`` is

    d_i(v) = w(x_i) * sum_k code(v)[k] * B_k(x_i)

with ``B_k`` the k-th basis displacement field, stored on a coarse control
grid and trilinearly interpolated to the primitive's REST position ``x_i``,
and ``w`` a support weight that is 1 inside the moving box, 0 outside, with a
cosine taper between. A few thousand control points against millions of
primitives: the motion cannot become a second free cloud, it is smooth by
construction, and it costs a gather of eight corners per primitive per step.
The rest cloud is the mean state because the free codes are penalised toward
zero.

GATED PHASES. When the scan was loaded with several acquisition groups (the
gated phases, ``--phase 00,01``) the code of a view has a FIXED part that
names its group — for two groups a single column at -1 / +1 — followed by the
free part. The basis field paired with that column is therefore the learned
half-difference between the phases, shared by every view of a phase and
determined by all 440 of them; the free columns explain what varies inside a
phase. A single-phase scan has no fixed part.

PER-VIEW GATE AMPLITUDE (``amplitude``). Every breath is a little deeper or
shallower, so the anatomy at the gate's trigger delay is not the same breath
to breath: MEASURED on run t9h4nndo, the largest coherent structure left in
the residual after the gate field and the free codes is the diaphragm dome
with a random per-view amplitude, i.e. the SHAPE of breathing the model knows
and the AMOUNT per view it does not. With ``amplitude`` on, the fixed part
of view v's code is ``fixed[v] + a[v]`` with ``a`` one learned scalar per
view and fixed column, penalised toward zero (``amp_norm``) and with NO
smoothness prior — the amount is random per breath, so a prior that pulls
consecutive views together would fight it. The view moves along the
breathing axis the gate field already describes; nothing new has to be
rediscovered by a free basis field, and the held-out view fits its ``a``
with its free code.

FROZEN VIEWS (``frozen_views``). Views whose renders do NOT train the motion
model: they are rendered in their state (gate value, zero free code and
amplitude) with every motion parameter detached, so their residual drives the
cloud and the photometric model only. For views whose excess residual is not
motion (MEASURED: a burst at the lateral 173-181 deg angles, the thickest
path) this keeps the field from absorbing model error while the views still
count as data. Their codes stay at zero and are left out of the code priors.

SHAPE TRANSPORT. A displaced primitive keeps its rest covariance unless
``transport_shapes`` is on, in which case Sigma' = F Sigma F^T with
F = I + dd/dx from the same trilinear basis (and the taper's gradient), the
"deformation-informed" form. The rasteriser takes Sigma' directly; the
voxeliser cannot (see `kernel._shape_kwargs`) and gets it re-factorised.

MASS (``mass_preserving``). A primitive's mass is a (2 pi)^{3/2} |Sigma|^{1/2}
(the cloud's own ``_integral``), so transporting the shape scales it by
|det F| while the amplitude a stays put: the transported model conserves
DENSITY and destroys MASS. Under a uniform compression the crowding (1/det F)
and the shrinking footprint (det F) cancel exactly and the rendered interior
does not change AT ALL -- MEASURED, a synthetic uniform mixture renders
mu x 1.0000 at det F = 0.512 -- so the volumetric part of the field is
invisible to the data except through boundaries. That is a graphics
convention: PhysGaussian (Xie et al., CVPR 2024, Sec 3.4) transports the
covariance and states "we assume that the opacity ... [is] invariant over
time", which is sound where opacity is appearance and the mass is carried by
the MPM particles, and DIGS (Huang et al., MICCAI 2025, Eq 8) inherited the
pair into CBCT, where a IS the attenuation density ("and rho_n remains
constant over time"). Pushing a density through a material map gives
mu_t(phi(x)) = mu_0(x) / |det F|, so ``mass_preserving`` divides the amplitude
by |det F| (floored at ``det_floor``, below which 1/det blows up and past zero
changes sign). Per-primitive mass is then an exact invariant and the interior
brightness carries div(u). It NEEDS ``transport_shapes``: without it the rest
shapes already conserve the mass and dividing would break the invariant.
`fold_excess` is the companion guard -- nothing else in the model penalises
det F -> 0, and on run z708w98r the minimum over the cloud was already 0.214.
The CLI default is ON since 2026-09-19 (A/B muqojfqg vs t805g3zw: held-out
tie, lung volume ratio 0.84 vs 0.60 with the amplitude held); the class
default stays False so a bare `MotionModel` is the literature model.

PARTS (``n_parts`` > 1): sliding interfaces. One smooth field cannot let
the lung slide along the chest wall or past the heart: a smoothness prior
blends the two motions across the interface, dragging the ribs into the
lung's motion and holding the lung back (MEASURED on t9h4nndo / z708w98r:
the residual left on the gate-0 state is an edge-locked doubled outline at
the heart and lung borders and laterally displaced ribs, exactly where two
tissues touch and move differently). With K parts the model holds K
independent basis sets on the same grid and every primitive learns a soft
membership p_i = softmax(m_i) over them:

    d_i = sum_j p_ij * d^(j)(x_i),    F_i = I + sum_j p_ij * dd^(j)/dx

The membership is a property of the PARTICLE, not of the position (no
dp/dx term): a primitive is a parcel of tissue and which organ it belongs
to does not change as it moves, exactly like its mass. Two priors keep the
partition honest — a commitment term (the entropy of p_i) so that each
primitive picks one part, and a Charbonnier (L1-like, NOT L2) penalty on
||p_i - p_j|| over a k-nearest-neighbour graph of the rest positions, so
the partition is piecewise constant with SHARP boundaries (a Potts model;
an L2 term would smear the boundary over several neighbours, which is the
blending this exists to remove). Nothing says where the interface is: it
appears where the data gain from one, and with no gain the K fields
converge to the same field and the model is the single-field one. The
symmetric start is a SADDLE, though (MEASURED on a two-shell toy): with
identical fields every part receives the same gradient, so the fields
never differ and the membership has nothing to choose between; the
commitment prior then breaks the symmetry in a random direction and the
TV term erases it, and the run ends with every primitive in one part.
Gradient descent, annealing, SGD instead of Adam and a residual-direction
split of the fields all failed the toy. What works is the classic
motion-segmentation order: the parts start TIED (identical fields, frozen
uniform membership, i.e. the single-field model) until ``parts_from``;
then the local motion of that single field at every primitive — the gate
displacement and its Jacobian in displacement units — is clustered by
k-means into K groups and the membership is set from the labels while the
fields stay identical, so the loss is continuous at the switch; from then
on the fields diverge BECAUSE the memberships differ and the priors
sharpen the partition (toy: 99 % of the labels right at the switch, 98 %
after training, the interface slip recovered). There is
no explicit contact (no-gap / no-overlap) constraint because the data term
is one: a gap is missing attenuation along every ray through it and an
overlap is doubled attenuation. Registration methods need such a term
(Preston, Joshi & Whitaker, MICCAI 2016, who also learn the partition —
on reconstructed images); the learned-membership form is linear blend
skinning with learned weights (SC-GS, Huang et al., CVPR 2024, and Shape
of Motion, 2024 — scene splatting, not CT). The cloud must be FIXED in
size (the production default) because the membership is indexed like the
cloud's own tensors and neither clone/split/prune nor relocation knows
about it. The graph is built once, at activation, from the rest positions
of that moment. The CLI default is K = 3 since 2026-09-19 (run 90pu264k:
lung / heart + abdomen / rest, soft-tissue NCC vs FDK gate 0 at 89 % of the
single-gate run's, lung volume ratio 0.868 vs 0.893 with one field); the
class default stays 1.

FINE GRID + BENDING PRIOR (CLI defaults since 2026-09-21: grid 1 mm, parts
volume weight 6, `smoothness(order=2)` at 0.01; the class defaults stay 3 mm /
order 1 so a bare MotionModel is the literature model). At 3 mm the lung
boundary was a 1.5 mm ramp that held 58 % of the reported ventilation map. At
1 mm the field is set by the data at ~20 parcels per cell and speckles unless
regularised; the first-difference prior is the STRAIN and at the weight that
removed the speckle it also removed the lung's real 25 % expansion (run
grid1_rb03: lung J 1.19, vendor r 0.46). The second-difference prior is the
bending energy, zero for any uniform strain, so it removes the speckle (raw
2nd differences 4.25 -> 0.23 voxel^2) and keeps the strain (1st differences
1.0 vs 0.015 under order 1). Its weight is a monotone dial between regional
detail and tissue strain (0 / 0.01 / 0.03: vendor block r 0.74 / 0.70 / 0.65,
tissue strain in lung cells 0.18 / 0.028 / 0.019); 0.01 was chosen by
vendor-blind rules (tissue in lung cells strains no more than tissue in
solid cells, share of lung below J 1 near the 3 mm run's, held-out not
below the unregularised grid) and the vendor comparison was read afterwards
(bias +0.012, deep lung 1.314 vs 1.317, shell 1.194 vs 1.195). Cost: image
agreement with FDK 1-2 null spreads below the 3 mm run in every window,
runtime 154 vs 133 min.

TISSUE-ONLY BENDING (``basis_lung_weight``). The bending prior above is
uniform, and the one place the true field bends sharply is the lung
boundary, where the volume change stops at the pleura. MEASURED on the
Scan_1510 digital phantom (2026-09-23, five runs): the prior changes
nothing in the lung's core (bias -0.017 with it, -0.011 without) and the
tissue's spurious strain barely (0.018 vs 0.020, the phantom has none of the
real scan's model error that made the prior necessary), but it rounds the
boundary: tidal volume over the 1 mm band 89 % of the truth with it, 96 %
without, block correlation of the ramp 0.61 vs 0.79. The prior only exists
for the tissue, so it is weighted per node by the air gate below: 1 at a
solid node, ``basis_lung_weight`` at a fully air-bearing one (per part with
parts), applied to every difference by the mean weight of the nodes it
spans. 0 (the DEFAULT since 2026-09-24, after the animal runs below) frees
the air entirely; 1 is the uniform prior. Before the first air-gate refresh
the prior is uniform. The weight reads the PRIOR gate (``prior_gate``)
when one is set and the air gate otherwise. RUN on the real animal
(..._grid1_bend01_gated, F 0, air gate): vendor block r 0.77 (uniform
0.705), thorax tissue strain unchanged (0.108 vs 0.093) — but the air gate
is 1 in every air-bearing cell, the air around the animal and the empty
cells included, and there the freed field wandered (solid parcels in
air-gated cells outside the thorax |J - 1| rms 0.53, p99 1.05, against
0.03 / 0.12; almost all of them outside the rendered box). The prior gate
was built to limit the lifting to the thorax (the lung mask's 3 mm
neighbourhood, 0 in empty cells; reconstructor option 'thorax'). RUN
2026-09-24: it frees the lung's cells (0.4-0.7 % of the nodes against
79 %) and changes NOTHING at the boundary — phantom band 88 % as the
uniform prior (air gate 94 %), animal vendor r 0.72 (0.77). The bend the
uniform prior charges is not inside the lung: the true field continues
into the air outside the body as a decaying potential with real
curvature, and a bending penalty there is paid by weakening the sources
at the lung edge. So the air gate IS the prior gate ('air', the default);
'thorax' is kept as the measured control.

AIR-SOURCED DIVERGENCE (``air_gate``). Only air changes volume: a parcel
of tissue slides, shears and rotates but keeps its volume, and the field
above lets every grid cell swell or shrink wherever the projections are
ambiguous — an invisible error in the image (a uniform compression renders
identically, see MASS) that lands whole on the ventilation map, which IS the
volumetric part. A penalty on det F in solid tissue (``reg_tissue``, run
54hxw5ix, weight 300 = 5 % of the loss) erased the lung's ventilation
together with the tissue's (J 1.245 -> 0.999) while every image and
held-out metric stayed put: a penalty cannot tell the lung's expansion from
the wall's. So the constraint is STRUCTURAL instead. Every basis field is
split on the control grid into a divergence-free part and a potential part
(the discrete Helmholtz decomposition, with backward-difference divergence
D and forward-difference gradient G so that D G is the 7-point Dirichlet
Laplacian and D = -G^T), and the potential part's source is gated by an
air-bearing fraction g in [0, 1] at every node:

    u' = u - G psi,    G^T G psi = (g - 1) D u    =>    D u' = g D u

exactly, at every node, by one Poisson solve per render. The operator is
the constant-coefficient 7-point Laplacian (reflecting low faces, zero
potential beyond the high faces), separable, so its eigenbasis is a
product of three small 1-D ones and the solve is two separable transforms
and a division (`poisson_solve`): exact, no iterations, the same cost at
any conditioning, autograd for the adjoint.
It replaced a warm-started conjugate-gradient solve (2026-09-23) that was
free on the 12x13x23 grid, ~15 % of a step at 1 mm (80 k nodes) and 24x a
step at 0.5 mm (619 k nodes: ~50 dependent passes twice per step, MEASURED
8.0 vs 0.33 s per iteration). Where g = 0 (solid
tissue) the field is divergence-free at the grid scale, where g = 1 it is
untouched. The gate is a MATERIAL property read from the model's own rest
state, detached — refreshed from the density, never optimised, so the
model cannot turn tissue into air to buy freedom: the reconstructor
renders the rest state at ``air_gate_mm``, reads each parcel's HU at its
rest position and ramps it from solid at -100 HU to free at the lung
window's edge (`ventilation.air_bearing`), and the gate at a node is the
mean of that over the parcels in the node's cell. With PARTS the gate is
PER PART — the membership-weighted mean — so the lung part's field is
free in every cell that holds lung and the wall part's field is
divergence-free in the same cell: a cell that is half lung and half wall
is not a compromise (half the volume change for both) but two fields,
each read only by its own parcels. That is the resolution the parts buy
the constraint; with one field (or in the tied phase) a mixed cell gets
the cell's lung fraction. Trilinear interpolation of a grid-divergence-
free field is divergence-free per cell on average, not pointwise, and the
support taper adds f . grad w at the box faces: the constraint holds at
the resolution of the field, which is the resolution the ventilation is
read at. Incompressibility as a hard structure of the deformation is
known in registration (iLogDemons, Mansi et al., IJCV 2011,
divergence-free stationary velocity fields for the myocardium); gating
the divergence by a jointly reconstructed air fraction, per motion part,
inside a projection-domain reconstruction is not something we found. The class
default is a free field (no gate); the CLI default is ON since 2026-09-20. PREVIEWED post hoc on
run 90pu264k (its free field projected, no re-fit): the solid-tissue
|J - 1| rms fell 0.094 -> 0.045 and the lung's median J 1.243 -> 1.115 —
on the 3 mm grid only 30 % of the air-bearing parcels sit in a cell whose
eight corners are all free, and the checkpoint's "lung" part holds 84 %
of the support at 20 % air-bearing, so the per-part gate was the tied
one; a projection cannot put back at the free nodes what it removed at
the mixed ones, training can (the smoothness prior is 1e-3 of the loss).
What training under the gate gives is the A/B.

Everything here is plain tensors and functions, like `GaussianCloud`: no
nn.Module, so the reconstructor's snapshot / restore / param-group discipline
carries over unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..gaussian import kernel as _kernel
from ..gaussian.model import _quat_to_rot


class MotionSupport:
    """Where motion is allowed: an axis-aligned box with a cosine taper.

    ``lo``, ``hi`` and ``taper`` are in world units (mm x world_scale). The
    weight is 1 inside the box, falls to 0 over ``taper`` outside each face
    (as the product of the three per-axis ramps), and is exactly 0 beyond.
    A zero taper is a hard box.
    """

    def __init__(self, lo, hi, taper: float):
        self.lo = torch.as_tensor(lo, dtype=torch.float32).reshape(3)
        self.hi = torch.as_tensor(hi, dtype=torch.float32).reshape(3)
        if bool((self.hi <= self.lo).any()):
            raise ValueError(f"support box has non-positive extent: "
                             f"lo={self.lo.tolist()} hi={self.hi.tolist()}")
        self.taper = float(taper)
        if self.taper < 0:
            raise ValueError("taper must be >= 0")

    def to(self, device):
        self.lo = self.lo.to(device)
        self.hi = self.hi.to(device)
        return self

    @property
    def outer_lo(self):
        return self.lo - self.taper

    @property
    def outer_hi(self):
        return self.hi + self.taper

    def weight(self, xyz: torch.Tensor) -> torch.Tensor:
        """(N,) in [0, 1] for positions (N, 3)."""
        return self.weight_and_grad(xyz)[0]

    def weight_and_grad(self, xyz: torch.Tensor):
        """The weight (N,) and its spatial gradient (N, 3)."""
        lo, hi = self.lo.to(xyz.device), self.hi.to(xyz.device)
        below, above = (lo - xyz).clamp_min(0.0), (xyz - hi).clamp_min(0.0)
        outside = torch.maximum(below, above)                    # (N, 3)
        if self.taper <= 0:
            w = (outside <= 0).all(dim=1).to(xyz.dtype)
            return w, torch.zeros_like(xyz)
        t = (outside / self.taper).clamp(max=1.0)
        ramp = 0.5 * (1.0 + torch.cos(math.pi * t))              # (N, 3)
        w = ramp.prod(dim=1)
        # d ramp / d x: sign of the excursion times the ramp's slope in t
        sign = torch.where(above > 0, 1.0, torch.where(below > 0, -1.0, 0.0))
        dramp = -0.5 * math.pi * torch.sin(math.pi * t) / self.taper * sign
        others = torch.stack([ramp[:, 1] * ramp[:, 2], ramp[:, 0] * ramp[:, 2],
                              ramp[:, 0] * ramp[:, 1]], dim=1)
        return w, dramp * others


def _shift(x: torch.Tensor, axis: int, step: int) -> torch.Tensor:
    """``x`` read at index i + step along ``axis``, zero outside the grid."""
    n = x.shape[axis]
    pad = torch.zeros_like(x.narrow(axis, 0, abs(step)))
    if step > 0:
        return torch.cat([x.narrow(axis, step, n - step), pad], dim=axis)
    return torch.cat([pad, x.narrow(axis, 0, n + step)], dim=axis)


def grid_grad(psi: torch.Tensor, cell) -> torch.Tensor:
    """Forward differences of scalar grid fields (B, nx, ny, nz) ->
    (B, nx, ny, nz, 3); the potential is zero beyond the last node."""
    return torch.stack([(_shift(psi, a + 1, 1) - psi) / float(cell[a])
                        for a in range(3)], dim=-1)


def grid_div(u: torch.Tensor, cell) -> torch.Tensor:
    """Backward differences of vector grid fields (B, nx, ny, nz, 3) ->
    (B, nx, ny, nz); the field is zero before the first node. Exactly
    -grid_grad^T, so -grid_div(grid_grad(.)) is symmetric positive
    definite (the 7-point Laplacian with a zero potential outside)."""
    out = None
    for a in range(3):
        d = (u[..., a] - _shift(u[..., a], a + 1, -1)) / float(cell[a])
        out = d if out is None else out + d
    return out


def _cg(apply_a, b: torch.Tensor, x0: torch.Tensor, tol: float = 1e-7,
        max_iter: int = 2000) -> torch.Tensor:
    """Batched conjugate gradients on (B, ...) right-hand sides, to a
    relative residual ``tol`` on every member of the batch."""
    b2 = (b * b).flatten(1).sum(1)
    if not bool((b2 > 0).any()):
        return torch.zeros_like(b)                 # exact, whatever the warm start
    x = x0.clone()
    r = b - apply_a(x)
    p = r.clone()
    rs = (r * r).flatten(1).sum(1)
    stop = (tol ** 2) * b2.clamp_min(1e-300)
    view = (-1,) + (1,) * (b.dim() - 1)
    for _ in range(max_iter):
        if bool((rs <= stop).all()):
            break
        ap = apply_a(p)
        alpha = rs / (p * ap).flatten(1).sum(1).clamp_min(1e-300)
        x = x + alpha.view(view) * p
        r = r - alpha.view(view) * ap
        rs_new = (r * r).flatten(1).sum(1)
        p = r + (rs_new / rs.clamp_min(1e-300)).view(view) * p
        rs = rs_new
    return x


class _PoissonSolve(torch.autograd.Function):
    """The conjugate-gradient solve `poisson_solve` replaced (kept as the
    reference the tests compare against): psi = A^{-1} b with A = -div grad
    (SPD), in float64, warm-started from the previous call through
    ``state``; the backward pass is the same solve on the incoming gradient
    because A is symmetric."""

    @staticmethod
    def forward(ctx, b, cell, state):
        def apply_a(v):
            return -grid_div(grid_grad(v, cell), cell)
        b64 = b.detach().double()
        x0 = state.get('forward')
        if x0 is None or x0.shape != b64.shape or x0.device != b64.device:
            x0 = torch.zeros_like(b64)
        x = _cg(apply_a, b64, x0)
        state['forward'] = x
        ctx.cell, ctx.state = cell, state
        return x.to(b.dtype)

    @staticmethod
    def backward(ctx, grad):
        cell, state = ctx.cell, ctx.state

        def apply_a(v):
            return -grid_div(grid_grad(v, cell), cell)
        g64 = grad.detach().double()
        x0 = state.get('backward')
        if x0 is None or x0.shape != g64.shape or x0.device != g64.device:
            x0 = torch.zeros_like(g64)
        x = _cg(apply_a, g64, x0)
        state['backward'] = x
        return x.to(grad.dtype), None, None


_DST_CACHE: dict = {}


def _eigen_factors(shape, cell, device, dtype):
    """Per axis, the orthonormal eigenvectors Q (n x n) of the 1-D operator
    `grid_div(grid_grad(.))` builds — tridiagonal (2, -1)/h^2 with a
    reflecting first node (the field is zero BEFORE the grid, so the flux
    across the low face is zero: diagonal 1) and a zero potential after
    the last node (Dirichlet at the high face) — and the summed eigenvalues
    on the 3-D grid. Cached per (shape, cell, device, dtype). The operator
    is separable and constant-coefficient, so its 3-D eigenbasis is the
    product of the three 1-D ones."""
    key = (tuple(int(n) for n in shape), tuple(float(c) for c in cell), str(device), str(dtype))
    hit = _DST_CACHE.get(key)
    if hit is not None:
        return hit
    mats, lam = [], None
    for a, n in enumerate(key[0]):
        h2 = key[1][a] ** 2
        A = (2.0 * torch.eye(n, dtype=torch.float64) - torch.diag(torch.ones(n - 1, dtype=torch.float64), 1)
             - torch.diag(torch.ones(n - 1, dtype=torch.float64), -1)) / h2
        A[0, 0] = 1.0 / h2
        w, Q = torch.linalg.eigh(A)
        mats.append(Q.to(device=device, dtype=dtype))
        l1 = w.to(device=device, dtype=dtype).reshape([-1 if b == a else 1 for b in range(3)])
        lam = l1 if lam is None else lam + l1
    _DST_CACHE[key] = (mats, lam)
    return mats, lam


def _to_eigen(x: torch.Tensor, mats, inverse: bool = False) -> torch.Tensor:
    """(B, nx, ny, nz) into (or, ``inverse``, out of) the separable
    eigenbasis: Q^T x per axis, Q x on the way back."""
    Qx, Qy, Qz = (m.T if inverse else m for m in mats)     # sum over the axis index: Q^T x, or Q x back
    x = torch.einsum('bxyz,xi->biyz', x, Qx)
    x = torch.einsum('bxyz,yj->bxjz', x, Qy)
    return torch.einsum('bxyz,zk->bxyk', x, Qz)


def poisson_solve(b: torch.Tensor, cell) -> torch.Tensor:
    """psi with -div grad psi = b on the grid (B, nx, ny, nz), the discrete
    operator of `grid_div` / `grid_grad`: the exact solve in the operator's
    separable eigenbasis, in float64. A constant-coefficient operator on a
    regular grid is diagonal there, so the solve is two separable
    transforms (three small matrix products each) and a division — no
    iterations, no warm start, a cost that does not grow with the grid's
    conditioning (MEASURED 2026-09-23: the conjugate-gradient solve it
    replaces cost 200-330 ms per solve at 1 mm and 2.2-3 s at 0.5 mm, this
    0.5 and 3.4 ms). Linear in ``b`` through ordinary matrix products, so
    autograd gives the adjoint."""
    mats, lam = _eigen_factors(b.shape[1:], cell, b.device, torch.float64)
    bh = _to_eigen(b.double(), mats)
    return _to_eigen(bh / lam, mats, inverse=True).to(b.dtype)


def gate_divergence(values: torch.Tensor, shape, cell, gate: torch.Tensor,
                    state: dict | None = None) -> torch.Tensor:
    """Grid fields ``values`` (F, n, 3) with their discrete divergence at
    every node multiplied by ``gate`` — (n,) for all fields or (F, n) one
    per field — in [0, 1]: u' = u - grad psi with -div grad psi =
    (g - 1) div u. Linear in ``values`` (differentiable), the gate is a
    constant; g = 1 everywhere returns ``values`` exactly, g = 0 everywhere
    the divergence-free projection."""
    F = values.shape[0]
    u = values.reshape(F, *shape, 3)
    h = tuple(float(c) for c in cell)
    div = grid_div(u, h)
    g = gate.detach().to(device=values.device, dtype=values.dtype)
    g = g.reshape(1, *shape) if g.dim() == 1 else g.reshape(F, *shape)
    psi = poisson_solve((g - 1.0) * div, h)
    return (u - grid_grad(psi, h)).reshape(F, -1, 3)


class ControlGridBasis:
    """``n_fields`` displacement fields on a regular control grid.

    The grid covers the support's OUTER box (box + taper) at ``spacing``
    world units, at least two control points per axis. Values are the
    displacement in world units.

    ``init_std`` > 0 draws the fields from N(0, init_std). The displacement is
    a PRODUCT of code and basis, so with both at zero neither has a gradient
    (a saddle: MEASURED, the scaffold's first loop test left both at exactly
    zero). Random fields with zero codes keep the initial displacement
    exactly zero — the model still starts as the static cloud — while giving
    the codes a direction to move in, after which the basis trains too.
    """

    def __init__(self, support: MotionSupport, spacing: float, n_fields: int,
                 device=None, init_std: float = 0.0, generator=None,
                 shape=None):
        if spacing <= 0 and shape is None:
            raise ValueError("control grid spacing must be > 0")
        if n_fields < 1:
            raise ValueError("need at least one basis field")
        self.lo = support.outer_lo.clone()
        self.hi = support.outer_hi.clone()
        extent = (self.hi - self.lo)
        self.shape = (tuple(int(v) for v in shape) if shape is not None else
                      tuple(max(2, int(math.ceil(float(e) / spacing)) + 1)
                            for e in extent.tolist()))
        self.n_fields = int(n_fields)
        n = self.shape[0] * self.shape[1] * self.shape[2]
        values = torch.zeros((self.n_fields, n, 3), dtype=torch.float32)
        if init_std > 0:
            values.normal_(0.0, float(init_std), generator=generator)
        self.values = nn.Parameter(values.to(device) if device is not None
                                   else values)
        self._strides = (self.shape[1] * self.shape[2], self.shape[2], 1)
        self._cg_state = {}                       # warm starts of `gated`

    @property
    def n_points(self) -> int:
        return int(self.values.shape[1])

    @property
    def cell(self) -> torch.Tensor:
        """Control spacing per axis, world units (3,)."""
        shape = torch.tensor(self.shape, dtype=torch.float32, device=self.lo.device)
        return (self.hi - self.lo) / (shape - 1.0)

    def to(self, device):
        self.lo = self.lo.to(device)
        self.hi = self.hi.to(device)
        self.values = nn.Parameter(self.values.detach().to(device))
        self._cg_state = {}
        return self

    def node_positions(self) -> torch.Tensor:
        """(n, 3) world positions of the control points, in value order."""
        axes = [torch.linspace(float(self.lo[a]), float(self.hi[a]), self.shape[a],
                               device=self.lo.device) for a in range(3)]
        return torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 3)

    def gated(self, gate: torch.Tensor | None, values: torch.Tensor | None = None):
        """The fields with their divergence gated by ``gate`` (n,) — see
        `gate_divergence`; None passes the values through."""
        values = self.values if values is None else values
        if gate is None:
            return values
        return gate_divergence(values, self.shape, self.cell, gate, self._cg_state)

    def _corners(self, xyz: torch.Tensor):
        dev = xyz.device
        lo, hi = self.lo.to(dev), self.hi.to(dev)
        shape = torch.tensor(self.shape, dtype=torch.float32, device=dev)
        u = (xyz - lo) / (hi - lo) * (shape - 1.0)
        u = u.clamp(min=torch.zeros_like(shape), max=shape - 1.0)
        i0 = u.floor().clamp(max=shape - 2.0)
        f = (u - i0).to(xyz.dtype)                        # (M, 3) in [0, 1]
        return i0.long(), f

    def interpolate(self, xyz: torch.Tensor) -> torch.Tensor:
        """Trilinear values at ``xyz`` (M, 3): (M, n_fields, 3).

        Positions outside the grid are clamped to its faces; the support
        weight is what zeroes them, so this never extrapolates.
        """
        return self.interpolate_with_grad(xyz, need_grad=False)[0]

    def interpolate_with_grad(self, xyz: torch.Tensor, need_grad: bool = True,
                              values: torch.Tensor | None = None):
        """Values (M, K, 3) and, if asked, their spatial gradient
        (M, K, 3, 3) with ``grad[m, k, c, a] = d B_k,c / d x_a``.
        ``values`` replaces the parameter tensor (e.g. a detached copy)."""
        values = self.values if values is None else values
        i0, f = self._corners(xyz)
        sx, sy, sz = self._strides
        cell = self.cell.to(xyz.device)
        val = None
        grad = None
        for cx in (0, 1):
            wx, dwx = (f[:, 0], 1.0) if cx else (1.0 - f[:, 0], -1.0)
            for cy in (0, 1):
                wy, dwy = (f[:, 1], 1.0) if cy else (1.0 - f[:, 1], -1.0)
                for cz in (0, 1):
                    wz, dwz = (f[:, 2], 1.0) if cz else (1.0 - f[:, 2], -1.0)
                    idx = ((i0[:, 0] + cx) * sx + (i0[:, 1] + cy) * sy
                           + (i0[:, 2] + cz) * sz)
                    corner = values[:, idx, :].permute(1, 0, 2)        # (M, K, 3)
                    w = (wx * wy * wz)[:, None, None]
                    val = corner * w if val is None else val + corner * w
                    if need_grad:
                        dw = torch.stack([dwx * wy * wz, wx * dwy * wz,
                                          wx * wy * dwz], dim=1) / cell   # (M, 3)
                        g = corner[:, :, :, None] * dw[:, None, None, :]  # (M,K,3,3)
                        grad = g if grad is None else grad + g
        return val, grad

    def smoothness(self, unit: float = 1.0, order: int = 1,
                   weight: torch.Tensor | None = None) -> torch.Tensor:
        """Mean squared finite difference of the given ``order`` along the
        three grid axes, with the differences expressed in ``unit`` (e.g.
        one voxel in world units) so the number is dimensionless and the
        weight transfers across scans. ``weight`` (n,) or (n_fields, n), in
        [0, 1] per node, scales each difference by the mean weight of the
        nodes it spans (the denominator stays the count of all differences,
        so all-ones reproduces the unweighted value).

        Order 1 penalises the displacement's gradient — the STRAIN — so a
        lung expanding uniformly by 25 % (7.7 % linear, one voxel per 1 mm
        step) pays sixteen times what node-to-node speckle of a quarter
        voxel pays, and at a weight that removes the speckle the expansion
        goes with it (MEASURED, run …_grid1_rb03: solid strain 0.128 ->
        0.044, lung J 1.228 -> 1.194, vendor r 0.74 -> 0.46). Order 2 is
        the bending energy of free-form deformation (Rueckert et al., IEEE
        TMI 1999): second differences, exactly zero for any affine field
        including a uniform expansion, positive only where the strain
        itself varies from node to node — the speckle.
        """
        if int(order) not in (1, 2):
            raise ValueError("smoothness order must be 1 or 2")
        v = self.values.reshape(self.n_fields, *self.shape, 3) / float(unit)
        w = None
        if weight is not None:
            w = torch.as_tensor(weight, dtype=v.dtype, device=v.device).detach()
            w = (w.reshape(1, *self.shape) if w.dim() == 1 else w.reshape(self.n_fields, *self.shape))
        terms = []
        for axis in (1, 2, 3):
            if v.shape[axis] > int(order):
                d = torch.diff(v, n=int(order), dim=axis)
                if w is None:
                    terms.append((d * d).mean())
                else:
                    k = int(order) + 1                              # nodes a difference spans
                    n = w.shape[axis]
                    ww = sum(w.narrow(axis, i, n - k + 1) for i in range(k)) / k
                    terms.append(((d * d) * ww.unsqueeze(-1)).mean())
        return torch.stack(terms).mean() if terms else v.new_zeros(())


class DisplacedCloud:
    """A view of a `GaussianCloud` with its positions displaced.

    Duck-types what the rasteriser and the voxeliser read (``xyz``,
    ``scaling``, ``rotation``, ``density``, optionally ``cov3D_precomp``) and
    ``__len__``; every other attribute is the base cloud's.

    With ``cov6`` (N, 6) the shapes are the TRANSPORTED covariances: the
    rasteriser takes them as they are, the voxeliser gets them re-factorised
    into scale + quaternion on demand (`kernel.covariance_to_scaling_rotation`,
    cached). Without it the rest shapes are used.

    ``density_scale`` (N,) multiplies the base amplitude — the 1/|det F| of
    the mass-preserving transport. None leaves the amplitude alone, which is
    what every transported 4DGS model in the literature does.
    """

    def __init__(self, base, displacement: torch.Tensor, cov6=None,
                 density_scale: torch.Tensor | None = None):
        self.base = base
        self.displacement = displacement
        self.cov3D_precomp = cov6
        self.density_scale = density_scale
        self._factorised = None

    @property
    def xyz(self):
        return self.base.xyz + self.displacement

    def _factorise(self):
        if self._factorised is None:
            with torch.no_grad():
                self._factorised = _kernel.covariance_to_scaling_rotation(
                    self.cov3D_precomp.detach())
        return self._factorised

    @property
    def scaling(self):
        return self.base.scaling if self.cov3D_precomp is None else self._factorise()[0]

    @property
    def rotation(self):
        return self.base.rotation if self.cov3D_precomp is None else self._factorise()[1]

    @property
    def density(self):
        if self.density_scale is None:
            return self.base.density
        return self.base.density * self.density_scale.unsqueeze(-1)

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name in ('base', 'displacement', 'cov3D_precomp', 'density_scale',
                    '_factorised'):
            raise AttributeError(name)
        return getattr(self.base, name)


def fixed_features(view_groups, n_views: int) -> torch.Tensor | None:
    """The per-view FIXED code columns from the acquisition groups.

    G groups give G - 1 contrast columns: column j is +1 on group j + 1, -1 on
    group 0, 0 elsewhere — for two gated phases a single +-1 column, so the
    rest cloud sits midway between the phases. None for a single group.
    """
    if view_groups is None:
        return None
    g = torch.as_tensor(np.asarray(view_groups), dtype=torch.long).reshape(-1)
    if g.numel() != n_views:
        raise ValueError(f"{g.numel()} view groups for {n_views} views")
    n_groups = int(g.max()) + 1 if g.numel() else 1
    if n_groups < 2:
        return None
    cols = []
    for j in range(1, n_groups):
        col = torch.zeros(n_views, dtype=torch.float32)
        col[g == j] = 1.0
        col[g == 0] = -1.0
        cols.append(col)
    return torch.stack(cols, dim=1)


@torch.no_grad()
def _kmeans(x: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """Plain k-means with k-means++ seeding on (M, D); labels (M,) long."""
    m = int(x.shape[0])
    if m == 0:
        return torch.zeros(0, dtype=torch.long, device=x.device)
    k = max(1, min(int(k), m))
    gen = torch.Generator(device='cpu').manual_seed(int(seed))
    cen = x[int(torch.randint(m, (1,), generator=gen))][None]
    for _ in range(1, k):
        d2 = torch.cdist(x, cen).min(dim=1).values.pow(2).cpu().double()
        if float(d2.sum()) > 0:
            nxt = int(torch.multinomial(d2 / d2.sum(), 1, generator=gen))
        else:                                   # every point on a centroid
            nxt = int(torch.randint(m, (1,), generator=gen))
        cen = torch.cat([cen, x[nxt][None]])
    labels = torch.zeros(m, dtype=torch.long, device=x.device)
    for _ in range(int(iters)):
        new = torch.cdist(x, cen).argmin(dim=1)
        if torch.equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            sel = labels == j
            if bool(sel.any()):
                cen[j] = x[sel].mean(dim=0)
    return labels


class MotionModel:
    """State codes + basis + support, with the discipline the loop expects."""

    #: LR multipliers on the two motion groups, relative to ``--lr``. Codes
    #: are O(1) numbers, basis values are displacements in world units
    #: (a voxel is ~1e-3 world units on Scan_1510), hence the split.
    #: motion_parts x100: MEASURED on run fzmn097d, at x10 (1e-3 at the base
    #: LR of 1e-4, one view per step under Adam) the logits moved 2.0 -> 2.16
    #: in 4000 iterations and not one primitive changed part.
    DEFAULT_LR_MULTIPLIERS = {'motion_codes': 10.0, 'motion_basis': 1.0,
                              'motion_amp': 10.0, 'motion_parts': 100.0}

    def __init__(self, n_views: int, n_codes: int, support: MotionSupport,
                 spacing: float, device=None, basis_init_std: float = 0.0,
                 generator=None, view_groups=None,
                 transport_shapes: bool = False, amplitude: bool = False,
                 frozen_views=(), mass_preserving: bool = False,
                 det_floor: float = 0.2, n_parts: int = 1,
                 n_primitives: int | None = None, parts_init_std: float = 0.01,
                 basis_order: int = 1, basis_lung_weight: float = 0.0):
        self.n_views = int(n_views)
        if int(basis_order) not in (1, 2):
            raise ValueError("basis_order must be 1 (strain) or 2 (bending)")
        self.basis_order = int(basis_order)
        if not 0.0 <= float(basis_lung_weight) <= 1.0:
            raise ValueError("basis_lung_weight must be in [0, 1]")
        self.basis_lung_weight = float(basis_lung_weight)
        self.n_codes = int(n_codes)                     # free codes per view
        self.transport_shapes = bool(transport_shapes)
        self._init_mass(bool(mass_preserving), float(det_floor))
        self.support = support.to(device) if device is not None else support
        fixed = fixed_features(view_groups, self.n_views)
        self.fixed = None if fixed is None else fixed.to(device or 'cpu')
        self.n_fixed = 0 if fixed is None else int(fixed.shape[1])
        self.view_groups = (None if view_groups is None
                            else torch.as_tensor(np.asarray(view_groups),
                                                 dtype=torch.long))
        self.n_parts = int(n_parts)
        if self.n_parts < 1:
            raise ValueError("n_parts must be >= 1")
        # K parts = K basis sets stacked along the field axis of ONE grid:
        # field index = part * n_fields + column
        self.basis = ControlGridBasis(self.support, spacing,
                                      self.n_parts * (self.n_fixed + self.n_codes),
                                      device, init_std=basis_init_std,
                                      generator=generator)
        self.codes = nn.Parameter(torch.zeros((self.n_views, self.n_codes),
                                              dtype=torch.float32,
                                              device=device))
        self._init_amplitude(bool(amplitude), device)
        self._init_frozen(frozen_views, device)
        self._init_parts(n_primitives, float(parts_init_std), generator, device)
        self.air_gate = None                     # (n,) per node, or None = free divergence
        self.prior_gate = None                   # (n,) lung-air per node for the bending weight
        self.last_displacement = None            # (M, 3) of the last render
        self.last_det = None                     # (M,) det F, transport only

    def _checked_gate(self, gate, what: str) -> torch.Tensor:
        g = torch.as_tensor(gate, dtype=torch.float32).detach()
        n = self.basis.n_points
        if g.dim() == 1 and g.numel() == n:
            g = g.reshape(n)
        elif g.dim() == 2 and tuple(g.shape) == (self.n_parts, n):
            pass
        else:
            raise ValueError(f"{what} has shape {tuple(g.shape)}; the grid has "
                             f"{n} nodes and the model {self.n_parts} part(s)")
        return g.clamp(0.0, 1.0).to(self.basis.values.device)

    def set_air_gate(self, gate) -> None:
        """The air-bearing fraction per control node, (n,) for every field
        or (n_parts, n) one per part (see the module docstring); None frees
        the divergence again. Stored detached and clamped to [0, 1]; the
        fields are unchanged, only their reading through `displacement`."""
        self.air_gate = None if gate is None else self._checked_gate(gate, "air gate")

    def set_prior_gate(self, gate) -> None:
        """The LUNG-air fraction per control node for the bending weight
        (TISSUE-ONLY BENDING in the module docstring): the air gate's shape,
        but 0 wherever the air is not the lung's. None = the weight reads
        the air gate."""
        self.prior_gate = None if gate is None else self._checked_gate(gate, "prior gate")

    def _field_gate(self, gate=None) -> torch.Tensor | None:
        """The gate per BASIS FIELD (P x n_fields rows): a per-part gate
        repeated over the part's columns. ``gate`` defaults to the air gate."""
        g = self.air_gate if gate is None else gate
        if g is None or g.dim() == 1:
            return g
        return g[:, None, :].expand(self.n_parts, self.n_fields, -1).reshape(-1, g.shape[1])

    @torch.no_grad()
    def gate_at(self, xyz: torch.Tensor, parts: torch.Tensor | None = None) -> torch.Tensor | None:
        """The gate a parcel's own field is under, (N,) in [0, 1]: the node
        gate interpolated trilinearly at ``xyz`` and, with per-part gates,
        mixed by the membership (``parts`` (N, K), default the model's
        own). None without a gate. Outside the grid the corner clamp
        applies, like the field itself."""
        g = self.air_gate
        if g is None:
            return None
        rows = g[None] if g.dim() == 1 else g                        # (F, n)
        vals = rows[:, :, None].expand(-1, -1, 3).contiguous()
        at = self.basis.interpolate_with_grad(xyz, need_grad=False, values=vals)[0][:, :, 0]  # (N, F)
        if g.dim() == 1:
            return at[:, 0].clamp(0.0, 1.0)
        p = self.membership() if parts is None else parts
        if p is None:
            p = torch.full((xyz.shape[0], self.n_parts), 1.0 / self.n_parts, device=xyz.device)
        return (at * p.to(at.dtype)).sum(dim=1).clamp(0.0, 1.0)

    @torch.no_grad()
    def air_gate_from_cloud(self, xyz: torch.Tensor, air_bearing: torch.Tensor,
                            parts: torch.Tensor | None = None, empty: float = 1.0) -> torch.Tensor:
        """The gate from the parcels: for every node, the mean of
        ``air_bearing`` (N,) in [0, 1] over the parcels in its cell — the
        dual cell [node - h, node] the backward-difference divergence
        reads — weighted by ``parts`` (N, K) when given, which makes it one
        gate per part (K, n); (n,) without. A node whose cell holds no
        parcel (of that part) takes ``empty``: FREE (1) for the divergence
        gate, nothing there can be tissue; 0 for the prior gate, nothing
        there is lung."""
        b = self.basis
        lo, cell = b.lo.to(xyz.device), b.cell.to(xyz.device)
        shape = torch.tensor(b.shape, device=xyz.device)
        idx = torch.ceil((xyz - lo) / cell).long().clamp(min=torch.zeros_like(shape), max=shape - 1)
        sx, sy, sz = b._strides
        flat = idx[:, 0] * sx + idx[:, 1] * sy + idx[:, 2] * sz
        a = air_bearing.to(xyz.device, torch.float32).reshape(-1)
        w = (torch.ones((xyz.shape[0], 1), device=xyz.device) if parts is None
             else parts.detach().to(xyz.device, torch.float32))
        K = w.shape[1]
        num = torch.zeros((K, b.n_points), device=xyz.device).index_add_(1, flat, (w * a[:, None]).T)
        den = torch.zeros((K, b.n_points), device=xyz.device).index_add_(1, flat, w.T)
        g = torch.where(den > 1e-6, num / den.clamp_min(1e-6), torch.full_like(num, float(empty)))
        return g.clamp(0.0, 1.0) if parts is not None else g[0].clamp(0.0, 1.0)

    def _init_mass(self, mass_preserving: bool, det_floor: float) -> None:
        """The mass-preserving amplitude and its floor (see the module docstring)."""
        if mass_preserving and not self.transport_shapes:
            raise ValueError(
                "mass_preserving needs transport_shapes: with the rest "
                "covariance kept, a primitive's mass is ALREADY invariant and "
                "dividing the amplitude by |det F| would break it")
        if det_floor <= 0:
            raise ValueError("det_floor must be > 0")
        self.mass_preserving = bool(mass_preserving)
        self.det_floor = float(det_floor)

    def _init_amplitude(self, amplitude: bool, device) -> None:
        if amplitude and self.n_fixed == 0:
            raise ValueError("a per-view gate amplitude needs gated phases "
                             "(several acquisition groups); this scan has one")
        self.amplitude = bool(amplitude)
        self.amp = (nn.Parameter(torch.zeros((self.n_views, self.n_fixed),
                                             dtype=torch.float32, device=device))
                    if self.amplitude else None)

    def _init_frozen(self, frozen_views, device) -> None:
        frozen = torch.zeros(self.n_views, dtype=torch.bool)
        for v in frozen_views or ():
            v = int(v)
            if not 0 <= v < self.n_views:
                raise ValueError(f"frozen view {v} is outside 0..{self.n_views - 1}")
            frozen[v] = True
        self.frozen = frozen.to(device or 'cpu')

    def is_frozen(self, view: int) -> bool:
        return bool(self.frozen[int(view)])

    def _init_parts(self, n_primitives, init_std: float, generator, device) -> None:
        """The per-primitive membership logits (N, K), or None for one part.
        Zero logits are the uniform membership, under which K identical
        random bases give the single-part model; the noise breaks the
        symmetry between the parts."""
        self.parts_logits = None
        self.parts_active = False                # False = tied (single-field) phase
        self.graph = None                        # (N, k) neighbour indices
        self.graph_weight = None                 # (N,) support weight at build
        self._graph_xyz = None                   # (N, 3) rest positions at build
        if self.n_parts == 1:
            return
        if n_primitives is None or int(n_primitives) < 1:
            raise ValueError("n_parts > 1 needs n_primitives (the membership "
                             "is one row per primitive)")
        m = torch.zeros((int(n_primitives), self.n_parts), dtype=torch.float32)
        if init_std > 0:
            m.normal_(0.0, float(init_std), generator=generator)
        self.parts_logits = nn.Parameter(m.to(device) if device is not None else m)
        # tied start: every part carries the SAME field until activation
        with torch.no_grad():
            K, Fn = self.n_parts, self.n_fields
            v = self.basis.values.reshape(K, Fn, -1, 3)
            v[1:] = v[:1]

    def membership(self, tied_ok: bool = False) -> torch.Tensor | None:
        """(N, K) soft membership of every primitive; None with one part and,
        unless ``tied_ok``, None before activation as well (the tied phase
        renders the uniform mix through ``parts=None`` and the logits get no
        gradient — with identical fields their exact gradient is zero and
        Adam would turn the rounding residue into a random walk)."""
        if self.parts_logits is None or not (self.parts_active or tied_ok):
            return None
        return torch.softmax(self.parts_logits, dim=1)

    @torch.no_grad()
    def activate_parts(self, xyz: torch.Tensor, code: torch.Tensor | None = None,
                       logit: float = 2.0, seed: int = 0,
                       volume_weight: float = 2.0) -> dict | None:
        """End the tied phase: label the primitives by k-means on the local
        motion of the (single) field and set the membership from the labels.

        Features per primitive inside the support: the displacement under
        ``code`` (default: the gate contrast, +1 on the first fixed column
        with the free code at zero; without gates the mean state) and the
        VOLUME CHANGE, log|det F| times the grid cell times
        ``volume_weight`` — the displacement difference across one cell
        that would produce that volumetric strain, so it is in displacement
        units — all divided by the rms displacement. The volume term is what
        finds the lung: lung tissue barely translates, it EXPANDS, so on
        displacement alone it lands in the same part as the chest wall
        (MEASURED, run fzmn097d: the partition was liver + diaphragm vs
        everything else). On the muqojfqg field, weight 1 still lost the
        lung to the liver's 0.4 mm translation; weights 2 and 4 both gave
        lung / heart + abdomen / rest. The 9-component strain used before
        diluted the volumetric part among the shears. k-means++ seeded,
        ``K`` clusters, on the device, parts ordered by size. Primitives
        outside the support (where motion is zero anyway) go to the largest
        cluster. The fields are left as they are — identical — so the
        rendered state does not change at the switch. Returns the cluster
        shares, the rms displacement (world units) and each part's mean
        displacement and log det; None if there are no parts or they are
        already active."""
        if self.parts_logits is None or self.parts_active:
            return None
        n = int(xyz.shape[0])
        if n != int(self.parts_logits.shape[0]):
            raise ValueError(f"activate_parts on {n} primitives, membership "
                             f"has {int(self.parts_logits.shape[0])} rows")
        dev = self.parts_logits.device
        xyz = xyz.detach().to(dev)
        if code is None:
            if self.n_fixed:
                code = torch.zeros(self.n_fields, device=dev)
                code[0] = 1.0
            else:
                code = torch.stack([self.full_code(v) for v in range(self.n_views)]).mean(dim=0)
        d, F = self.displacement(xyz, code.to(dev), need_jacobian=True, parts=None)
        w = self.support.weight(xyz)
        inside = (w > 0).nonzero(as_tuple=True)[0]
        cell = float(self.basis.cell.mean())
        # |log det| clamped at 0.5 (det 0.6..1.65, beyond any half-gate
        # physiology): a smooth field fitting a translation JUMP folds in a
        # thin band at the jump (MEASURED on the shift toy: det 0.09 there
        # vs 0.95..1.05 everywhere on the real field), and unclamped that
        # band outweighs every genuine volume change in the clustering.
        logdet = torch.log(torch.linalg.det(F).abs().clamp_min(1e-6)).clamp(-0.5, 0.5)
        x = torch.cat([d, (float(volume_weight) * cell * logdet)[:, None]], dim=1)[inside]
        rms = d[inside].norm(dim=1).pow(2).mean().sqrt() if inside.numel() else d.new_zeros(())
        x = x / rms.clamp_min(1e-12)
        labels = _kmeans(x, self.n_parts, seed=seed)
        # part 0 = the largest cluster, so the label order is reproducible
        counts = torch.bincount(labels, minlength=self.n_parts)
        order = torch.argsort(counts, descending=True)
        remap = torch.empty_like(order)
        remap[order] = torch.arange(self.n_parts, device=dev)
        labels = remap[labels]
        self.parts_logits.zero_()
        self.parts_logits[:, 0] = float(logit)             # outside the support
        self.parts_logits[inside, 0] = 0.0
        self.parts_logits[inside, labels] = float(logit)
        self.parts_active = True
        share = torch.bincount(labels, minlength=self.n_parts).to(torch.float32)
        share = share / share.sum().clamp_min(1.0)
        di, li = d[inside], logdet[inside]
        return {'share': share.cpu().tolist(), 'rms_disp': float(rms),
                'n_inside': int(inside.numel()),
                'part_disp': [float(di[labels == j].norm(dim=1).mean()) if bool((labels == j).any()) else 0.0
                              for j in range(self.n_parts)],
                'part_logdet': [float(li[labels == j].mean()) if bool((labels == j).any()) else 0.0
                                for j in range(self.n_parts)]}

    @torch.no_grad()
    def build_graph(self, xyz: torch.Tensor, k: int = 8) -> None:
        """The k-nearest-neighbour graph of the rest positions ``xyz``
        (N, 3), for the membership smoothness prior, plus the support weight
        at those positions (edges outside the moving box carry no
        information about motion and are weighted out). One scipy KD-tree
        query, on the CPU, once: the rest positions drift by fractions of a
        voxel after the motion starts, and a neighbour in rest space stays
        a neighbour."""
        if self.parts_logits is None:
            return
        n = int(xyz.shape[0])
        if n != int(self.parts_logits.shape[0]):
            raise ValueError(f"graph for {n} primitives, membership has "
                             f"{int(self.parts_logits.shape[0])} rows")
        k = int(k)
        if k < 1:
            raise ValueError("k must be >= 1")
        from scipy.spatial import cKDTree
        pts = xyz.detach().cpu().numpy().astype(np.float64)
        kk = min(k + 1, n)
        _, idx = cKDTree(pts).query(pts, k=kk, workers=-1)
        idx = np.asarray(idx).reshape(n, kk)[:, 1:]        # drop self
        if idx.shape[1] < k:                              # tiny clouds
            idx = np.concatenate([idx] + [idx[:, -1:]] * (k - idx.shape[1]), axis=1)
        dev = self.parts_logits.device
        self.graph = torch.as_tensor(idx, dtype=torch.long, device=dev)
        self._graph_xyz = xyz.detach().to(dev).clone()
        self.graph_weight = self.support.weight(self._graph_xyz)

    def part_priors(self, eps: float = 0.05) -> dict:
        """``part_commit``: mean normalised entropy of the membership over
        the support (1 = uniform, 0 = every primitive committed to one part).
        ``part_smooth``: mean Charbonnier ||p_i - p_j|| over the graph's
        edges, weighted by both ends' support weights — a total-variation
        term on the membership, so a boundary costs its area and not its
        sharpness. Both zero with one part or before the graph exists."""
        p = self.membership()
        if p is None:                       # one part, or the tied phase
            z = self.codes.new_zeros(())
            return {'part_commit': z, 'part_smooth': z}
        w = (self.graph_weight if self.graph_weight is not None
             else torch.ones(p.shape[0], device=p.device, dtype=p.dtype))
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1) / math.log(self.n_parts)
        commit = (ent * w).sum() / w.sum().clamp_min(1.0)
        if self.graph is None:
            return {'part_commit': commit, 'part_smooth': commit.new_zeros(())}
        q = p[self.graph]                                    # (N, k, K)
        diff = (p[:, None, :] - q).norm(dim=2)               # (N, k)
        rho = torch.sqrt(diff * diff + eps * eps) - eps
        ew = w[:, None] * w[self.graph]                      # (N, k)
        smooth = (rho * ew).sum() / ew.sum().clamp_min(1.0)
        return {'part_commit': commit, 'part_smooth': smooth}

    @classmethod
    def from_snapshot(cls, snap: dict, *, n_views: int, n_codes: int,
                      view_groups=None, device=None):
        """Rebuild a model from `snapshot` (as saved in a checkpoint): the
        support, the grid at its stored shape, then the values."""
        support = MotionSupport(snap['support_lo'], snap['support_hi'],
                                float(snap['support_taper']))
        m = cls.__new__(cls)
        m.n_views = int(n_views)
        m.n_codes = int(n_codes)
        m.transport_shapes = bool(snap.get('transport_shapes', False))
        m._init_mass(bool(snap.get('mass_preserving', False)),
                     float(snap.get('det_floor', 0.2)))
        m.support = support.to(device) if device is not None else support
        fixed = fixed_features(view_groups, m.n_views)
        m.fixed = None if fixed is None else fixed.to(device or 'cpu')
        m.n_fixed = 0 if fixed is None else int(fixed.shape[1])
        m.view_groups = (None if view_groups is None
                         else torch.as_tensor(np.asarray(view_groups), dtype=torch.long))
        m.n_parts = int(snap.get('n_parts', 1))
        m.basis_order = 1
        m.basis_lung_weight = float(snap.get('basis_lung_weight', 1.0))
        m.basis = ControlGridBasis(m.support, 0.0,
                                   m.n_parts * (m.n_fixed + m.n_codes), device,
                                   shape=snap['grid_shape'])
        m.codes = nn.Parameter(torch.zeros((m.n_views, m.n_codes),
                                           dtype=torch.float32, device=device))
        m._init_amplitude(bool(snap.get('amplitude', False)), device)
        m._init_frozen(snap.get('frozen_views', ()), device)
        logits = snap.get('parts_logits')
        m._init_parts(None if logits is None else int(logits.shape[0]), 0.0,
                      None, device)
        m.air_gate = None
        m.prior_gate = None
        m.last_displacement = None
        m.last_det = None
        m.restore(snap)
        return m

    # -- codes ---------------------------------------------------------------
    @property
    def n_fields(self) -> int:
        return self.n_fixed + self.n_codes

    def full_code(self, view: int) -> torch.Tensor:
        """Fixed columns (group contrast, plus the view's gate amplitude
        when on) followed by the free code."""
        z = self.codes[int(view)]
        if self.fixed is None:
            return z
        fixed = self.fixed[int(view)].to(z.device)
        if self.amp is not None:
            fixed = fixed + self.amp[int(view)]
        return torch.cat([fixed, z])

    def full_code_from(self, free: torch.Tensor, view: int | None,
                       amp: torch.Tensor | None = None) -> torch.Tensor:
        """A full code from a FREE code and the group of ``view`` (None: no
        group contrast, i.e. the mean of the phases), with ``amp`` added to
        the fixed columns when given (a fitted or a group-mean amplitude)."""
        if self.fixed is None:
            return free
        fixed = (torch.zeros(self.n_fixed, device=free.device) if view is None
                 else self.fixed[int(view)].to(free.device))
        if amp is not None:
            fixed = fixed + amp.to(free.device)
        return torch.cat([fixed, free])

    def group_mean_amplitude(self, views) -> torch.Tensor | None:
        """Mean gate amplitude over ``views`` (None when amplitude is off):
        the fixed part of a group's MEAN state is its contrast plus this."""
        if self.amp is None:
            return None
        idx = torch.as_tensor(np.asarray(views), dtype=torch.long,
                              device=self.amp.device)
        return self.amp.detach()[idx].mean(dim=0)

    # -- the forward model --------------------------------------------------
    def displacement(self, xyz: torch.Tensor, code: torch.Tensor,
                     need_jacobian: bool = False, detach: bool = False,
                     parts: torch.Tensor | None = None):
        """(N, 3) displacement of rest positions under one FULL code, and,
        if asked, the deformation gradient F = I + dd/dx as (N, 3, 3).
        ``detach`` evaluates the basis (and the code) as constants: the
        result still depends on ``xyz``, but no gradient reaches the motion
        parameters (frozen views). ``parts`` (N, K) is the membership of
        each position over the parts; None with several parts is the UNIFORM
        mix (the mean of the part fields — what a field probe at arbitrary
        positions, which have no membership, gets)."""
        w, gw = self.support.weight_and_grad(xyz)                # (N,), (N, 3)
        inside = w > 0
        d = torch.zeros_like(xyz)
        F = None
        if need_jacobian:
            F = torch.eye(3, dtype=xyz.dtype, device=xyz.device).expand(
                xyz.shape[0], 3, 3).clone()
        if bool(inside.any()):
            rows = inside.nonzero(as_tuple=True)[0]
            code = code.to(xyz.dtype)
            if detach:
                code = code.detach()
            values = self.basis.values.detach() if detach else None
            if self.air_gate is not None:
                values = self.basis.gated(self._field_gate(), values)
            fields, grads = self.basis.interpolate_with_grad(
                xyz[rows], need_grad=need_jacobian, values=values)  # (M,PK,3), (M,PK,3,3)
            if self.n_parts == 1:
                f = torch.einsum('mkc,k->mc', fields, code)            # (M, 3)
                df = (torch.einsum('mkca,k->mca', grads, code)         # (M, 3, 3)
                      if need_jacobian else None)
            else:
                P, K = self.n_parts, self.n_fields
                if parts is None:
                    p = torch.full((rows.numel(), P), 1.0 / P,
                                   dtype=xyz.dtype, device=xyz.device)
                else:
                    p = parts[rows].to(xyz.dtype)
                    if detach:
                        p = p.detach()
                fp = torch.einsum('mpkc,k->mpc', fields.reshape(-1, P, K, 3), code)
                f = torch.einsum('mpc,mp->mc', fp, p)
                if need_jacobian:
                    dfp = torch.einsum('mpkca,k->mpca',
                                       grads.reshape(-1, P, K, 3, 3), code)
                    df = torch.einsum('mpca,mp->mca', dfp, p)
                else:
                    df = None
            d = d.index_put((rows,), f * w[rows, None])
            if need_jacobian:
                # d(w f)/dx = w df/dx + f (grad w)^T
                J = w[rows, None, None] * df + f[:, :, None] * gw[rows, None, :]
                F = F.index_put((rows,), F[rows] + J)
        return (d, F) if need_jacobian else d

    def displaced_with_code(self, cloud, code: torch.Tensor,
                            detach: bool = False) -> DisplacedCloud:
        """The cloud in the state of one FULL code (``detach``: see
        `displacement`)."""
        if (self.parts_logits is not None
                and self.parts_logits.shape[0] != cloud.xyz.shape[0]):
            raise ValueError(f"the membership has {self.parts_logits.shape[0]} "
                             f"rows, the cloud {cloud.xyz.shape[0]} primitives: "
                             f"parts need a fixed-size cloud")
        parts = self.membership()
        if not self.transport_shapes:
            d = self.displacement(cloud.xyz, code, detach=detach, parts=parts)
            self.last_displacement = d
            self.last_det = None
            return DisplacedCloud(cloud, d)
        d, F = self.displacement(cloud.xyz, code, need_jacobian=True,
                                 detach=detach, parts=parts)
        self.last_displacement = d
        R = _quat_to_rot(cloud.rotation)
        M = R * cloud.scaling[:, None, :]                          # R diag(s)
        FM = F @ M
        S = FM @ FM.transpose(1, 2)                                # F Sigma F^T
        cov6 = torch.stack([S[:, 0, 0], S[:, 0, 1], S[:, 0, 2],
                            S[:, 1, 1], S[:, 1, 2], S[:, 2, 2]], dim=1)
        self.last_det = torch.linalg.det(F)
        scale = None
        if self.mass_preserving:
            # |Sigma'|^{1/2} = |det F| |Sigma|^{1/2}, so this is the amplitude
            # that leaves a (2 pi)^{3/2} |Sigma|^{1/2} — the primitive's mass —
            # exactly where it was.
            scale = 1.0 / self.last_det.abs().clamp_min(self.det_floor)
        return DisplacedCloud(cloud, d, cov6.contiguous(), density_scale=scale)

    def displaced(self, cloud, view: int, detach: bool | None = None) -> DisplacedCloud:
        """The cloud in the state of ``view``; a frozen view is rendered with
        the motion parameters detached unless ``detach`` says otherwise."""
        if detach is None:
            detach = self.is_frozen(view)
        return self.displaced_with_code(cloud, self.full_code(view), detach=detach)

    # -- priors -------------------------------------------------------------
    def priors(self, unit: float = 1.0, order=None) -> dict:
        """The regularisers, unweighted, on the FREE codes and the basis.

        ``code_norm``    mean ||z_v||^2 — the rest cloud is the mean state
        ``code_smooth``  mean ||z_{v+1} - z_v||^2 over consecutive views of
                         the SAME group in acquisition order (``order``
                         overrides the order) — consecutive views are similar
                         states more often than not (MEASURED ~7-view
                         quasi-periodicity in the thorax residual)
        ``basis_smooth`` mean squared finite difference of the basis fields
                         (first or, with ``basis_order`` 2, second — the
                         bending energy), in units of ``unit`` (one voxel,
                         world units)
        ``amp_norm``     mean a_v^2 over the views (0 with amplitude off) —
                         no smoothness term, the amount is random per breath

        Frozen views are left out of every code term (their codes are zero
        and untrained; a pair straddling one would pull its neighbour to 0).
        """
        z = self.codes
        idx = (torch.arange(z.shape[0], device=z.device) if order is None
               else torch.as_tensor(list(order), device=z.device))
        live = ~self.frozen.to(z.device)[idx]
        z = z[idx]
        lv = live.to(z.dtype)
        code_norm = ((z * z).sum(dim=1) * lv).sum() / lv.sum().clamp_min(1.0)
        if z.shape[0] > 1:
            dz = torch.diff(z, dim=0)
            pair = (dz * dz).sum(dim=1)
            keep = (live[1:] & live[:-1]).to(pair.dtype)
            if self.view_groups is not None:
                g = self.view_groups.to(z.device)[idx]
                keep = keep * (g[1:] == g[:-1]).to(pair.dtype)
            code_smooth = (pair * keep).sum() / keep.sum().clamp_min(1.0)
        else:
            code_smooth = z.new_zeros(())
        if self.amp is not None:
            a = self.amp[idx]
            amp_norm = ((a * a).sum(dim=1) * lv).sum() / lv.sum().clamp_min(1.0)
        else:
            amp_norm = z.new_zeros(())
        return {'code_norm': code_norm, 'code_smooth': code_smooth,
                'basis_smooth': self.basis.smoothness(unit, self.basis_order, self.basis_weight()),
                'amp_norm': amp_norm, **self.part_priors()}

    def basis_weight(self) -> torch.Tensor | None:
        """The per-node weight of the basis prior: 1 in solid tissue, falling
        to ``basis_lung_weight`` where a node is fully LUNG air (the prior
        gate, per part where there are parts; the air gate when no prior
        gate was set). None = uniform (no gate yet, or the weight is 1)."""
        g = self._field_gate(self.air_gate if self.prior_gate is None else self.prior_gate)
        if g is None or self.basis_lung_weight >= 1.0:
            return None
        return 1.0 - (1.0 - self.basis_lung_weight) * g

    def displacement_excess(self, max_world: float) -> torch.Tensor:
        """Mean squared excess of the LAST rendered displacement over
        ``max_world`` (a hinge: zero while every primitive moves less)."""
        d = self.last_displacement
        if d is None or max_world <= 0:
            return self.codes.new_zeros(())
        mag = d.norm(dim=1)
        return ((mag - max_world).clamp_min(0.0) ** 2).mean()

    # -- optimiser / state --------------------------------------------------
    def fold_excess(self, det_min: float | None = None) -> torch.Tensor:
        """Mean squared log-undershoot of the LAST render's det F below
        ``det_min`` (``det_floor`` when None): zero wherever a primitive's
        neighbourhood keeps at least that fraction of its volume, growing as
        log(det/det_min)^2 below it and without bound as det -> 0. One-sided
        on purpose — legitimate expansion is what the model is FOR, folding
        (det <= 0, where the map stops being a deformation and the
        mass-preserving amplitude changes sign) is what has to be impossible.
        """
        d = self.last_det
        if d is None:
            return self.codes.new_zeros(())
        floor = self.det_floor if det_min is None else float(det_min)
        if floor <= 0:
            return self.codes.new_zeros(())
        x = torch.log(d.clamp_min(1e-6) / floor)
        return x.clamp_max(0.0).pow(2).mean()

    def param_groups(self, lr: float, multipliers: dict | None = None) -> list:
        mul = dict(self.DEFAULT_LR_MULTIPLIERS)
        unknown = set(multipliers or ()) - set(mul)
        if unknown:
            raise ValueError(f"unknown motion LR group(s) {sorted(unknown)}; "
                             f"expected {sorted(mul)}")
        mul.update(multipliers or {})
        groups = [
            {'params': [self.codes], 'lr': lr * mul['motion_codes'],
             'name': 'motion_codes'},
            {'params': [self.basis.values], 'lr': lr * mul['motion_basis'],
             'name': 'motion_basis'},
        ]
        if self.amp is not None:
            groups.append({'params': [self.amp], 'lr': lr * mul['motion_amp'],
                           'name': 'motion_amp'})
        if self.parts_logits is not None:
            groups.append({'params': [self.parts_logits],
                           'lr': lr * mul['motion_parts'], 'name': 'motion_parts'})
        return groups

    def parameters(self):
        return ([self.codes, self.basis.values]
                + ([self.amp] if self.amp is not None else [])
                + ([self.parts_logits] if self.parts_logits is not None else []))

    @property
    def shape(self):
        return self.basis.shape

    def snapshot(self) -> dict:
        return {'codes': self.codes.detach().cpu().clone(),
                'basis': self.basis.values.detach().cpu().clone(),
                'fixed': None if self.fixed is None else self.fixed.cpu().clone(),
                'grid_shape': tuple(self.shape),
                'grid_lo': self.basis.lo.detach().cpu().clone(),
                'grid_hi': self.basis.hi.detach().cpu().clone(),
                'support_lo': self.support.lo.detach().cpu().clone(),
                'support_hi': self.support.hi.detach().cpu().clone(),
                'support_taper': self.support.taper,
                'transport_shapes': self.transport_shapes,
                'mass_preserving': self.mass_preserving,
                'det_floor': self.det_floor,
                'amplitude': self.amplitude,
                'amp': None if self.amp is None else self.amp.detach().cpu().clone(),
                'frozen_views': self.frozen.nonzero(as_tuple=True)[0].tolist(),
                'basis_lung_weight': self.basis_lung_weight,
                'n_parts': self.n_parts,
                'parts_active': bool(self.parts_active),
                'parts_logits': (None if self.parts_logits is None
                                 else self.parts_logits.detach().cpu().clone()),
                'air_gate': (None if self.air_gate is None
                             else self.air_gate.detach().cpu().clone()),
                'prior_gate': (None if self.prior_gate is None
                               else self.prior_gate.detach().cpu().clone())}

    @torch.no_grad()
    def restore(self, snap: dict) -> None:
        if int(snap.get('n_parts', 1)) != self.n_parts:
            raise ValueError(f"motion snapshot has {int(snap.get('n_parts', 1))} "
                             f"parts, model has {self.n_parts}")
        if tuple(snap['grid_shape']) != tuple(self.shape):
            raise ValueError(f"motion snapshot grid {tuple(snap['grid_shape'])} "
                             f"!= model grid {tuple(self.shape)}")
        if tuple(snap['codes'].shape) != tuple(self.codes.shape):
            raise ValueError(f"motion snapshot has {tuple(snap['codes'].shape)} "
                             f"codes, model has {tuple(self.codes.shape)}")
        if tuple(snap['basis'].shape) != tuple(self.basis.values.shape):
            raise ValueError(f"motion snapshot has {tuple(snap['basis'].shape)} "
                             f"basis values, model has "
                             f"{tuple(self.basis.values.shape)}")
        self.codes.copy_(snap['codes'].to(self.codes.device))
        self.basis.values.copy_(snap['basis'].to(self.basis.values.device))
        if self.amp is not None and snap.get('amp') is not None:
            if tuple(snap['amp'].shape) != tuple(self.amp.shape):
                raise ValueError(f"motion snapshot has {tuple(snap['amp'].shape)} "
                                 f"amplitudes, model has {tuple(self.amp.shape)}")
            self.amp.copy_(snap['amp'].to(self.amp.device))
        if self.parts_logits is not None:
            logits = snap.get('parts_logits')
            if logits is None or tuple(logits.shape) != tuple(self.parts_logits.shape):
                raise ValueError(f"motion snapshot membership "
                                 f"{None if logits is None else tuple(logits.shape)} "
                                 f"!= model {tuple(self.parts_logits.shape)}")
            self.parts_logits.copy_(logits.to(self.parts_logits.device))
            self.parts_active = bool(snap.get('parts_active', False))
        self.set_air_gate(snap.get('air_gate'))
        self.set_prior_gate(snap.get('prior_gate'))

    # -- diagnostics --------------------------------------------------------
    @torch.no_grad()
    def summary(self, xyz: torch.Tensor, world_scale: float,
                sample: int = 200_000) -> dict:
        """Displacement statistics over a sample of the cloud, in mm."""
        n = xyz.shape[0]
        parts = self.membership()          # None in the tied phase = uniform
        if parts is not None and parts.shape[0] != n:
            parts = None                       # not this cloud's membership
        if n > sample:
            idx = torch.randperm(n, device=xyz.device)[:sample]
            xyz = xyz[idx]
            parts = None if parts is None else parts[idx]
        w = self.support.weight(xyz)
        inside = w > 0
        out = {'motion/code_rms': float(self.codes.pow(2).mean().sqrt()),
               'motion/support_frac': float(inside.float().mean()),
               **({'motion/amp_rms': float(self.amp.pow(2).mean().sqrt()),
                   'motion/amp_max': float(self.amp.abs().max())}
                  if self.amp is not None else {}),
               'motion/basis_rms_mm': float(
                   self.basis.values.pow(2).mean().sqrt() / world_scale)}
        out.update(self.parts_summary(world_scale))
        if self.air_gate is not None:
            out['motion/air_gate_mean'] = float(self.air_gate.mean())
            out['motion/air_gate_solid_frac'] = float((self.air_gate < 0.1).float().mean())
        if bool(inside.any()):
            # Reduced per view, never stacked: at 200 k samples x 440 views a
            # kept (V, M) tensor is 350 MiB per quantity, on a card that is
            # already holding the cloud.
            mags, stats, tail = [], [], []
            p_in = None if parts is None else parts[inside]
            for v in range(self.n_views):
                got = self.displacement(xyz[inside], self.full_code(v),
                                        need_jacobian=self.transport_shapes,
                                        parts=p_in)
                d, F = got if self.transport_shapes else (got, None)
                mags.append(d.norm(dim=1))
                if F is None:
                    continue
                # The volume ratio the transport applies. Without
                # ``mass_preserving`` this IS the factor by which each
                # primitive's mass changes; with it, the factor that had to be
                # divided out, and what is left is only the floor clamp's
                # residue. det <= 0 is a folded field.
                det = torch.linalg.det(F)
                a = det.abs()
                realised = a / a.clamp_min(self.det_floor) if self.mass_preserving else a
                stats.append(torch.stack([det.min(), det.max(),
                                          (a < self.det_floor).to(det.dtype).mean(),
                                          (realised - 1.0).pow(2).mean()]))
                tail.append(det[::max(1, det.numel() // 2048)])
            m = torch.stack(mags) / world_scale                 # (V, M) mm
            out['motion/disp_rms_mm'] = float(m.pow(2).mean().sqrt())
            out['motion/disp_max_mm'] = float(m.max())
            if stats:
                S = torch.stack(stats)
                out['motion/det_min'] = float(S[:, 0].min())
                out['motion/det_max'] = float(S[:, 1].max())
                out['motion/det_below_floor_frac'] = float(S[:, 2].mean())
                out['motion/mass_drift_rms'] = float(S[:, 3].mean().sqrt())
                q = torch.cat(tail)
                out['motion/det_p01'] = float(q.quantile(0.01))
                out['motion/det_p99'] = float(q.quantile(0.99))
            if self.fixed is not None:
                # the fixed (phase) part alone: displacement at +-1 with the
                # free code at zero = the half-difference between the phases
                for j in range(self.n_fixed):
                    code = torch.zeros(self.n_fields, device=xyz.device)
                    code[j] = 1.0
                    dj = self.displacement(xyz[inside], code,
                                           parts=p_in).norm(dim=1) / world_scale
                    out[f'motion/phase{j + 1}_half_diff_rms_mm'] = float(dj.pow(2).mean().sqrt())
                    out[f'motion/phase{j + 1}_half_diff_max_mm'] = float(dj.max())
        return out

    @torch.no_grad()
    def parts_summary(self, world_scale: float) -> dict:
        """The partition, as numbers: normalised entropy, committed share
        (max p > 0.9), each part's share of the support, and the SLIP — the
        rms relative displacement between graph neighbours that belong to
        DIFFERENT parts, at the gate contrast (+1 minus -1 with the free
        code at zero) or, without gates, over the mean state. Neighbours in
        one part move together by construction (same field, one grid cell
        apart), so slip is the sliding the partition buys; zero means the
        parts have converged to one field. Empty with one part; the slip
        needs the graph."""
        p = self.membership(tied_ok=True)
        if p is None:
            return {}
        w = (self.graph_weight if self.graph_weight is not None
             else torch.ones(p.shape[0], device=p.device))
        ws = w.sum().clamp_min(1.0)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1) / math.log(self.n_parts)
        pmax, arg = p.max(dim=1)
        out = {'motion/parts_active': float(self.parts_active),
               'motion/parts_entropy': float((ent * w).sum() / ws),
               'motion/parts_committed_frac': float(((pmax > 0.9).to(w.dtype) * w).sum() / ws)}
        if not self.parts_active:
            return out
        for j in range(self.n_parts):
            out[f'motion/parts_frac_{j}'] = float(((arg == j).to(w.dtype) * w).sum() / ws)
        if self.graph is None or self._graph_xyz is None:
            return out
        xyz = self._graph_xyz
        if self.fixed is not None:
            code = torch.zeros(self.n_fields, device=xyz.device)
            code[0] = 1.0
            d = 2.0 * self.displacement(xyz, code, parts=p)          # +1 minus -1
        else:
            codes = torch.stack([self.full_code(v) for v in range(self.n_views)])
            d = self.displacement(xyz, codes.mean(dim=0), parts=p)
        rel = (d[:, None, :] - d[self.graph]).norm(dim=2) / world_scale   # (N, k) mm
        cross = (arg[:, None] != arg[self.graph]).to(rel.dtype) * w[:, None] * w[self.graph]
        cs = cross.sum()
        out['motion/parts_interface_frac'] = float(cs / (w[:, None] * w[self.graph]).sum().clamp_min(1.0))
        out['motion/parts_slip_rms_mm'] = (float(((rel * rel * cross).sum() / cs).sqrt())
                                           if float(cs) > 0 else 0.0)
        return out
