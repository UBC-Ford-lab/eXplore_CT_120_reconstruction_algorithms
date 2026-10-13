"""Time-aware Gaussian splatting: the static reconstructor plus a motion model.

Everything the submodule's `GaussianReconstructor` does — seeding, the WLS
loss, shape constraints, the LR schedule, stopping, checkpointing, export —
is inherited unchanged. This subclass changes only WHAT is rasterised: for
view ``v`` the kernel sees the rest cloud displaced (and, optionally, its
shapes transported and its amplitude divided by |det F| so the transport
moves mass instead of destroying it) by the motion model's state for that
view (`motion.MotionModel`), and the optimiser sees two more parameter
groups. The seams it uses are the ones the base class declares for exactly
this purpose:
``_render_train``, ``_render_eval``, ``_param_groups``, ``_regularise``,
``_state`` / ``_restore_state``, ``_checkpoint_payload``.

Schedule (the part that decides whether such a model is identifiable):

  1. ``motion_from``: until this iteration the displacement is not applied
     and the motion parameters receive no gradient — the run is the static
     recipe and the rest cloud becomes the mean anatomy.
  2. From then on codes and basis train alongside the cloud, under the
     priors (`MotionModel.priors` + a displacement bound), with their own LR
     multipliers — and, for ``motion_hold`` iterations, EXEMPT from the
     plateau reducer's cuts (``lr_hold_until`` on their param groups). The
     cloud has been training since iteration 1; the motion parameters start
     at zero at ``motion_from``, and a cut drawn by the cloud's plateau
     landed on them after 1500 iterations in run ffap88ti, freezing the
     field at 10000 for the remaining 7000. After the hold they join the
     chain at its current level.

With several acquisition groups loaded (``--phase 00,01``) each view's code
carries its group as a fixed contrast column, so the phase difference is one
shared learned field and the free codes explain the within-phase variation.

With ``n_parts`` > 1 the motion model carries several independent fields
and a learned membership per primitive (sliding interfaces, see
`motion`); that needs a fixed-size cloud, which the constructor checks
against the static recipe's density control, and a neighbour graph, built
once at the first regularised render.

A withheld evaluation view has no trained code: its FREE code is fitted at
evaluation time with the cloud frozen (`_fit_code`), which is honest because
it is a handful of numbers against a million pixels, and it is what makes the
held-out metric comparable to the static run's.

``export_group`` exports a phase's MEAN state (the CLI defaults to group 0,
the state the single-gate runs and the vendor gate-0 volume are in),
``export_view`` one view's state; with neither (``--dyn-export-group rest``,
or a single phase) the REST cloud is exported — with two phases the midpoint
between them, a state the animal never occupied.
"""
from __future__ import annotations

import numpy as np
import torch

from ...ct_core.hu_calibration import resolve_anchors
from ...ct_core.projection_diag import covered_detector_window
from ...ct_core.support import export_grid_geometry
from ..gaussian import camera as cam_mod
from ..gaussian import kernel
from ..gaussian.reconstructor import GaussianReconstructor

from .motion import MotionModel, MotionSupport
from . import panels
from . import show


def _pitch_option(value):
    """'auto' (the delivered grid) or a positive pitch in mm."""
    if value is None or (isinstance(value, str) and value.strip().lower() == 'auto'):
        return 'auto'
    p = float(value)
    if p <= 0:
        raise ValueError("export_4d_mm must be 'auto' or > 0 mm")
    return p


def _format_option(value) -> str:
    """'nrrd' (the default), 'nrrd-slicer' or 'nifti'; validated here so a
    typo fails at option time, not after the training."""
    from .export4d import check_format
    return check_format(value)


class DynamicGaussianReconstructor(GaussianReconstructor):

    def __init__(self, projections, angles, geometry, *,
                 n_codes: int = 1,
                 grid_mm: float = 3.0,
                 support: str = 'roi',
                 support_margin_mm: float = 2.0,
                 support_taper_mm: float = 2.0,
                 motion_from: float = 0.25,
                 motion_hold: float = 0.25,
                 basis_init_vox: float = 0.25,
                 transport_shapes: bool = True,
                 mass_preserving: bool = False,
                 det_floor: float = 0.2,
                 reg_fold: float = 1.0,
                 amplitude: bool = False,
                 frozen_views=(),
                 n_parts: int = 1,
                 parts_from: float = 0.375,
                 parts_volume_weight: float = 2.0,
                 part_knn: int = 8,
                 reg_part_smooth: float = 0.1,
                 reg_part_commit: float = 1e-2,
                 air_sourced: bool = False,
                 air_gate_mm: float = 0.5,
                 reg_code: float = 1e-2,
                 reg_code_smooth: float = 1e-2,
                 reg_amp: float = 1e-2,
                 reg_basis: float = 1e-3,
                 reg_basis_order: int = 1,
                 reg_basis_lung_weight: float = 1.0,   # uniform; the CLI default is 0 (tissue-only)
                 reg_basis_lung_region: str = 'air',
                 max_disp_mm: float = 2.0,
                 reg_disp: float = 1.0,
                 holdout_code_iters: int = 30,
                 holdout_code_lr: float = 1e-2,
                 export_view: int | None = None,
                 export_group: int | None = None,
                 export_4d: int = 9,
                 export_4d_mm='auto',
                 export_4d_format: str = 'nrrd',
                 export_ventilation: bool = True,
                 ventilation_mm: float = 0.5,
                 motion_lr_multipliers: dict | None = None,
                 **kwargs):
        super().__init__(projections, angles, geometry, **kwargs)
        self.mcfg = dict(
            n_codes=int(n_codes), grid_mm=float(grid_mm),
            support=str(support),
            support_margin_mm=float(support_margin_mm),
            support_taper_mm=float(support_taper_mm),
            motion_from=float(motion_from), motion_hold=float(motion_hold),
            basis_init_vox=float(basis_init_vox),
            transport_shapes=bool(transport_shapes),
            mass_preserving=bool(mass_preserving),
            det_floor=float(det_floor), reg_fold=float(reg_fold),
            amplitude=bool(amplitude),
            frozen_views=tuple(sorted({int(v) for v in (frozen_views or ())})),
            n_parts=int(n_parts), parts_from=float(parts_from),
            parts_volume_weight=float(parts_volume_weight),
            part_knn=int(part_knn),
            reg_part_smooth=float(reg_part_smooth),
            reg_part_commit=float(reg_part_commit),
            air_sourced=bool(air_sourced), air_gate_mm=float(air_gate_mm),
            reg_code=float(reg_code), reg_code_smooth=float(reg_code_smooth),
            reg_amp=float(reg_amp), reg_basis=float(reg_basis),
            reg_basis_order=int(reg_basis_order),
            reg_basis_lung_weight=float(reg_basis_lung_weight),
            reg_basis_lung_region=str(reg_basis_lung_region),
            max_disp_mm=float(max_disp_mm), reg_disp=float(reg_disp),
            holdout_code_iters=int(holdout_code_iters),
            holdout_code_lr=float(holdout_code_lr),
            export_view=(None if export_view is None else int(export_view)),
            export_group=(None if export_group is None else int(export_group)),
            export_4d=int(export_4d), export_4d_mm=_pitch_option(export_4d_mm),
            export_4d_format=_format_option(export_4d_format),
            export_ventilation=bool(export_ventilation),
            ventilation_mm=float(ventilation_mm),
            motion_lr_multipliers=dict(motion_lr_multipliers or {}))
        if self.mcfg['ventilation_mm'] <= 0:
            raise ValueError("ventilation_mm must be > 0")
        if self.mcfg['reg_basis_order'] not in (1, 2):
            raise ValueError("reg_basis_order must be 1 (strain) or 2 (bending)")
        if self.mcfg['air_gate_mm'] <= 0:
            raise ValueError("air_gate_mm must be > 0")
        if self.mcfg['n_codes'] < 1:
            raise ValueError("n_codes must be >= 1")
        if self.mcfg['n_parts'] < 1:
            raise ValueError("n_parts must be >= 1")
        if self.mcfg['n_parts'] > 1 and self.cfg['density_control'] != 'fixed':
            raise ValueError(
                f"n_parts > 1 needs a fixed-size cloud: the membership is one "
                f"row per primitive and clone/split/prune/relocation "
                f"(density_control={self.cfg['density_control']!r}) do not "
                f"carry it. Use --gauss-density-control fixed (the production "
                f"default) or one part.")
        if export_view is not None and export_group is not None:
            raise ValueError("export_view and export_group are exclusive")
        self.motion: MotionModel | None = None
        self._motion_from_iter: int = 0
        self._motion_hold_until: int = 0
        self._parts_from_iter: int = 0
        self._parts_hold_until: int = 0
        self._fitted_codes: dict = {}

    # -- construction -------------------------------------------------------
    @property
    def _motion_from(self) -> int:
        return self._motion_from_iter

    def _support_box_world(self, domain):
        """(lo, hi) of the moving region in world units."""
        kind = self.mcfg['support']
        m = self.mcfg['support_margin_mm']
        if kind == 'roi':
            box = self._roi_box_mm(m)
            if box is None:
                kind = 'domain'
            else:
                lo, hi = box
                return (torch.as_tensor(lo, dtype=torch.float32) * self.world_scale,
                        torch.as_tensor(hi, dtype=torch.float32) * self.world_scale)
        if kind == 'domain':
            lo = torch.as_tensor(domain.aabb_min, dtype=torch.float32)
            hi = torch.as_tensor(domain.aabb_max, dtype=torch.float32)
            return lo * self.world_scale, hi * self.world_scale
        vals = [float(v) for v in kind.replace(',', ' ').split()]
        if len(vals) != 6:
            raise ValueError("support must be 'roi', 'domain' or six numbers "
                             "x_min x_max y_min y_max z_min z_max in mm")
        lo = torch.tensor(vals[0::2], dtype=torch.float32) - m
        hi = torch.tensor(vals[1::2], dtype=torch.float32) + m
        return lo * self.world_scale, hi * self.world_scale

    @property
    def _voxel_world(self) -> float:
        """One export voxel in world units — the length unit of the priors."""
        return float(self.geometry['dx']) * self.world_scale

    def _build_motion(self, domain, device, cloud=None) -> MotionModel:
        """``cloud`` sizes the per-primitive membership (parts > 1)."""
        lo, hi = self._support_box_world(domain)
        support = MotionSupport(lo, hi,
                                self.mcfg['support_taper_mm'] * self.world_scale)
        gen = torch.Generator().manual_seed(int(self.seed))
        groups = getattr(self, 'view_groups', None)
        model = MotionModel(n_views=len(self.angles),
                            n_codes=self.mcfg['n_codes'], support=support,
                            spacing=self.mcfg['grid_mm'] * self.world_scale,
                            device=device,
                            basis_init_std=(self.mcfg['basis_init_vox']
                                            * self._voxel_world),
                            generator=gen, view_groups=groups,
                            transport_shapes=self.mcfg['transport_shapes'],
                            mass_preserving=self.mcfg['mass_preserving'],
                            det_floor=self.mcfg['det_floor'],
                            amplitude=self.mcfg['amplitude'],
                            frozen_views=self.mcfg['frozen_views'],
                            n_parts=self.mcfg['n_parts'],
                            n_primitives=(None if cloud is None else len(cloud)),
                            basis_order=self.mcfg['reg_basis_order'],
                            basis_lung_weight=self.mcfg['reg_basis_lung_weight'])
        f = self.mcfg['motion_from']
        self._motion_from_iter = int(round(f * self.iterations)) if f < 1 \
            else int(f)
        h = self.mcfg['motion_hold']
        hold = int(round(h * self.iterations)) if h < 1 else int(h)
        self._motion_hold_until = self._motion_from_iter + max(0, hold)
        pf = self.mcfg['parts_from']
        parts_from = int(round(pf * self.iterations)) if pf < 1 else int(pf)
        # the parts are labelled from the single field, which must exist
        self._parts_from_iter = max(parts_from, self._motion_from_iter)
        # and the membership gets the same LR hold as the other motion
        # groups, counted from ITS start: MEASURED on fzmn097d, activated at
        # 12000 under a hold ending at 16000 after five plateau cuts, the
        # membership froze at the chain's floor 4000 iterations in.
        self._parts_hold_until = self._parts_from_iter + max(0, hold)
        print(f"  motion model: {model.n_fixed} fixed (group) + {model.n_codes} "
              f"free code columns x {len(self.angles)} views"
              + (", per-view gate amplitude" if model.amplitude else "")
              + (f", {int(model.frozen.sum())} frozen views (train the cloud, not the motion): "
                 f"{self.mcfg['frozen_views']}" if bool(model.frozen.any()) else "")
              + ", basis grid "
              f"{'x'.join(str(s) for s in model.shape)} at "
              f"{self.mcfg['grid_mm']:g} mm over "
              f"[{', '.join(f'{v / self.world_scale:.1f}' for v in lo.tolist())}]"
              f"..[{', '.join(f'{v / self.world_scale:.1f}' for v in hi.tolist())}] mm "
              f"(+{self.mcfg['support_taper_mm']:g} mm taper); shapes "
              f"{'transported by the Jacobian' if model.transport_shapes else 'kept'}"
              + (f", amplitude / |det F| (mass preserving, det floor "
                 f"{model.det_floor:g})" if model.mass_preserving else "")
              + (f"; {model.n_parts} PARTS (independent fields, learned "
                 f"per-primitive membership over {len(cloud):,} primitives, "
                 f"{self.mcfg['part_knn']}-NN membership graph; tied until "
                 f"iteration {self._parts_from_iter}, then labelled by k-means "
                 f"on the single field's displacement + {self.mcfg['parts_volume_weight']:g} x "
                 f"cell x log det F, membership LR held until {self._parts_hold_until})"
                 if model.n_parts > 1 else "")
              + (f"; AIR-SOURCED divergence (gate = air-bearing fraction of each "
                 f"control cell's parcels, per part once active, from the rest "
                 f"state at {self.mcfg['air_gate_mm']:g} mm, refreshed at every "
                 f"evaluation)" if self.mcfg['air_sourced'] else "")
              + (f"; bending prior TISSUE-ONLY (lung weight {self.mcfg['reg_basis_lung_weight']:g}, "
                 f"region {self.mcfg['reg_basis_lung_region']})"
                 if float(self.mcfg['reg_basis_lung_weight']) < 1.0 else "")
              + "; "
              f"motion trains from iteration {self._motion_from_iter}"
              + (f", exempt from LR cuts until {self._motion_hold_until}"
                 if self._motion_hold_until > self._motion_from_iter else "")
              + f"; displacement bound {self.mcfg['max_disp_mm']:g} mm; basis prior "
              f"{'BENDING (2nd differences)' if model.basis_order == 2 else 'strain (1st differences)'} "
              f"x{self.mcfg['reg_basis']:g}")
        return model

    def _seed(self, domain, rng, device):
        cloud = super()._seed(domain, rng, device)
        self.motion = self._build_motion(domain, device, cloud)
        return cloud

    # -- the seams ----------------------------------------------------------
    def _motion_active(self) -> bool:
        return (self.motion is not None
                and (self.iterations_run + 1) >= self._motion_from_iter)

    def _render_train(self, cloud, cameras, view: int) -> dict:
        if not self._motion_active():
            return super()._render_train(cloud, cameras, view)
        return kernel.render(self.motion.displaced(cloud, view), cameras[view],
                             need_screenspace_grad=True)

    def _render_eval(self, cloud, camera, view=None):
        if self.motion is None or view is None or not self._motion_active():
            return super()._render_eval(cloud, camera, view)
        code = self._fitted_codes.get(int(view))
        shown = (self.motion.displaced_with_code(cloud, code) if code is not None
                 else self.motion.displaced(cloud, view))
        return kernel.render(shown, camera)['image']

    def _param_groups(self, cloud) -> list:
        groups = super()._param_groups(cloud)
        if self.motion is not None:
            motion_groups = self.motion.param_groups(
                self.lr, self.mcfg['motion_lr_multipliers'])
            if self._motion_hold_until > self._motion_from_iter:
                for g in motion_groups:
                    g['lr_hold_until'] = int(self._motion_hold_until)
            for g in motion_groups:
                if g['name'] == 'motion_parts' and self._parts_hold_until > self._parts_from_iter:
                    g['lr_hold_until'] = int(max(self._parts_hold_until,
                                                 g.get('lr_hold_until', 0)))
            groups += motion_groups
        return groups

    def _regularise(self, loss, cloud, view: int):
        if not self._motion_active():
            return loss
        if self._gate_wanted() and self.motion.air_gate is None \
                and self.motion.prior_gate is None:
            self._refresh_air_gate(cloud)          # once at the start; then per evaluation
        if (self.motion.parts_logits is not None and not self.motion.parts_active
                and (self.iterations_run + 1) >= self._parts_from_iter):
            self._activate_parts(cloud)
        p = self.motion.priors(unit=self._voxel_world)
        loss = (loss + self.mcfg['reg_code'] * p['code_norm']
                + self.mcfg['reg_code_smooth'] * p['code_smooth']
                + self.mcfg['reg_amp'] * p['amp_norm']
                + self.mcfg['reg_basis'] * p['basis_smooth']
                + self.mcfg['reg_part_smooth'] * p['part_smooth']
                + self.mcfg['reg_part_commit'] * p['part_commit'])
        if self.mcfg['reg_disp'] > 0 and self.mcfg['max_disp_mm'] > 0:
            excess = self.motion.displacement_excess(
                self.mcfg['max_disp_mm'] * self.world_scale)
            # in voxels^2 so the weight is dimensionless
            loss = loss + self.mcfg['reg_disp'] * excess / self._voxel_world ** 2
        if self.mcfg['reg_fold'] > 0 and self.motion.mass_preserving:
            # already dimensionless (squared log of a volume ratio)
            loss = loss + self.mcfg['reg_fold'] * self.motion.fold_excess()
        return loss

    def _gate_wanted(self) -> bool:
        """Whether the air gate is rendered at all: for the air-sourced
        divergence, or for the tissue-only bending prior (lung weight < 1),
        or both."""
        return bool(self.mcfg['air_sourced']) or float(self.mcfg['reg_basis_lung_weight']) < 1.0

    def _refresh_air_gate(self, cloud) -> None:
        """The air gate from the REST state rendered at ``air_gate_mm`` and
        calibrated to HU by the shared two-anchor fit (`resolve_anchors`):
        each parcel's HU at its rest position, ramped to its air-bearing
        value (`ventilation.air_bearing`; parcels outside the rendered box
        are free), averaged per control cell — per part once the parts are
        active (`MotionModel.air_gate_from_cloud`) — and handed to the
        motion model detached. The bending weight's PRIOR gate follows
        ``reg_basis_lung_region``: 'air' (default) reads the air gate itself
        — every air-bearing cell, the air around the animal and the empty
        cells included; 'thorax' the same mean over the parcels inside the
        rendered lung mask's 3 mm neighbourhood (`ventilation.thorax_region`)
        alone, 0 in empty cells; 'box' over the parcels inside the rendered
        box. MEASURED 2026-09-23/24 (phantom + animal, lung weight 0):
        'thorax' frees the lung and changes NOTHING at the boundary (band
        88 % as the uniform prior, block r 0.61 -> 0.70; animal vendor r
        0.705 -> 0.72) while 'air' gives band 94 % and vendor r 0.77: the
        true field continues into the air outside the body with real
        curvature (a decaying potential), and a bending penalty there is
        paid by weakening the sources at the lung edge. While the render
        shows no lung yet ('thorax') the prior gate is left unset and the
        weight reads the air gate. A failure (a histogram the fit cannot
        read early in a run) leaves the previous gates in place and says
        so — with no gate yet the divergence stays free until the next
        try. With --dyn-no-air-sourced the gate is still rendered when the
        prior is tissue-only (lung weight < 1) but handed to the PRIOR
        alone: the divergence stays free, the bending weight keeps its
        gate. Before 2026-09-30 the prior silently went uniform in that
        case (the 2026-09-26 ampheld_grid1 / masspres_grid1 runs)."""
        try:
            from . import export4d, ventilation
            geom = export4d.coarse_geometry(export_grid_geometry(self.geometry),
                                            self.mcfg['air_gate_mm'])
            mu = export4d.render_grid(cloud, geom, self.world_scale, self.signal_scale,
                                      self._export_chunk)
            anchors = resolve_anchors(mu, 'auto')
            hu = mu * float(anchors.scale) + float(anchors.offset)
            xyz = cloud.xyz.detach()
            xyz_mm = (xyz / self.world_scale).cpu().numpy()
            a = ventilation.air_bearing(ventilation.sample(hu, geom, xyz_mm, order=1))
            a = np.where(ventilation.in_box(geom, xyz_mm), a, 1.0)
            gate = self.motion.air_gate_from_cloud(
                xyz, torch.as_tensor(a, dtype=torch.float32, device=xyz.device),
                self.motion.membership())
            # the divergence reads the gate only when it is air-sourced; the
            # bending weight reads it either way (prior_gate, or the air gate
            # itself in region 'air')
            self.motion.set_air_gate(gate if self.mcfg['air_sourced'] else None)
            g = gate
            pitch = float(self.mcfg['air_gate_mm'])
            mode = self.mcfg['reg_basis_lung_region']
            prior, region_mm3 = None, None
            if mode == 'thorax':
                region = ventilation.thorax_region([ventilation.lung_mask(hu, pitch)], pitch)
                inside = ventilation.sample(region.astype(np.float32), geom, xyz_mm, order=0) > 0.5
                region_mm3 = float(region.sum()) * pitch ** 3
            elif mode == 'box':
                inside = ventilation.in_box(geom, xyz_mm)
            else:
                inside = None
            if inside is not None and inside.any():
                prior = self.motion.air_gate_from_cloud(
                    xyz, torch.as_tensor(np.where(inside, a, 0.0), dtype=torch.float32,
                                         device=xyz.device),
                    self.motion.membership(), empty=0.0)
            if prior is None and not self.mcfg['air_sourced']:
                prior = gate                       # region 'air' without an air gate to read
            self.motion.set_prior_gate(prior)
            print(f"  air gate refreshed at iteration {self.iterations_run + 1}: "
                  f"{self.motion.basis.n_points:,} nodes"
                  + (f" x {g.shape[0]} parts" if g.dim() == 2 else "")
                  + f", {100 * float((g < 0.1).float().mean()):.1f} % solid (< 0.1), "
                  f"{100 * float((g > 0.5).float().mean()):.1f} % air-bearing (> 0.5); "
                  f"{100 * float(a.mean()):.1f} % of the parcels air-bearing"
                  + (f"; prior ({mode}) lifted at {100 * float((prior > 0.5).float().mean()):.1f} % "
                     f"of the nodes" + (f" (thorax {region_mm3:.0f} mm^3)" if region_mm3 is not None else "")
                     if prior is not None else
                     ("; prior reads the air gate" if mode == 'air'
                      else "; no lung in the render yet, prior gate unset")))
            if self.log_fn is not None:
                self.log_fn({'diag/motion/air_gate_mean': float(g.mean()),
                             'diag/motion/air_gate_solid_frac': float((g < 0.1).float().mean()),
                             'diag/motion/air_gate_aerated_frac': float((g > 0.5).float().mean()),
                             'diag/motion/air_bearing_parcel_frac': float(a.mean())},
                            step=int(self.iterations_run))
        except Exception as e:
            print(f"  air gate not refreshed ({type(e).__name__}: {e})")

    def _activate_parts(self, cloud) -> None:
        """End of the tied phase: the membership graph from the rest
        positions of this moment, then the k-means labelling of the
        single field's local motion (`MotionModel.activate_parts`)."""
        xyz = cloud.xyz.detach()
        if self.motion.graph is None:
            self.motion.build_graph(xyz, k=self.mcfg['part_knn'])
        stats = self.motion.activate_parts(xyz, seed=int(self.seed),
                                           volume_weight=self.mcfg['parts_volume_weight'])
        if stats is None:
            return
        if self._gate_wanted():
            self._refresh_air_gate(cloud)          # the gate becomes per part
        share = '; '.join(f"part {j}: {100 * v:.1f} %, |d| {stats['part_disp'][j] / self.world_scale:.3f} mm, "
                          f"log det {stats['part_logdet'][j]:+.4f}" for j, v in enumerate(stats['share']))
        print(f"  parts activated at iteration {self.iterations_run + 1}: "
              f"{self.motion.n_parts} clusters over {stats['n_inside']:,} "
              f"primitives in the support ({share}); single-field rms "
              f"displacement {stats['rms_disp'] / self.world_scale:.3f} mm"
              + ("  WARNING: the field is ~0, the labels are noise"
                 if stats['rms_disp'] / self.world_scale < 0.02 else ""))
        if self.log_fn is not None:
            self.log_fn({'diag/motion/parts_activated_iter': int(self.iterations_run + 1),
                         **{f'diag/motion/parts_init_share_{j}': float(v)
                            for j, v in enumerate(stats['share'])},
                         **{f'diag/motion/parts_init_logdet_{j}': float(v)
                            for j, v in enumerate(stats['part_logdet'])},
                         **{f'diag/motion/parts_init_disp_mm_{j}': float(v) / self.world_scale
                            for j, v in enumerate(stats['part_disp'])}},
                        step=int(self.iterations_run))

    def _state(self, cloud) -> dict:
        s = super()._state(cloud)
        if self.motion is not None:
            s['motion'] = self.motion.snapshot()
        return s

    def _restore_state(self, cloud, state: dict) -> None:
        super()._restore_state(cloud, state)
        if self.motion is not None and 'motion' in state:
            self.motion.restore(state['motion'])

    def _checkpoint_payload(self, cloud) -> dict:
        payload = super()._checkpoint_payload(cloud)
        if self.motion is not None:
            payload['motion'] = self.motion.snapshot()
            payload['motion_config'] = dict(self.mcfg)
            if getattr(self, 'view_groups', None) is not None:
                payload['view_groups'] = np.asarray(self.view_groups)
        return payload

    # -- evaluation: the withheld view gets a code of its own --------------
    def _evaluate(self, cloud, camera, scorer, view=None) -> dict:
        if (self.motion is not None and view is not None
                and self._motion_active() and self.withhold_eval):
            self._fitted_codes[int(view)] = self._fit_code(cloud, camera,
                                                           scorer.target, view)
        metrics = super()._evaluate(cloud, camera, scorer, view)
        if self.motion is not None and self._motion_active():
            summary = self.motion.summary(cloud.xyz.detach(), self.world_scale)
            metrics.update(summary)
            # Logged HERE, not left to the loop: with a projection-diagnostics
            # callback in place the loop logs the diag/* scalars from that
            # callback's own scoring and never looks at this dict (MEASURED:
            # run ffap88ti had panels but no motion/* scalars).
            if self.log_fn is not None:
                self.log_fn({f'diag/{k}': v for k, v in summary.items()},
                            step=int(self.iterations_run))
            self._motion_panels(cloud)
            if self._gate_wanted():
                self._refresh_air_gate(cloud)
        return metrics

    def _motion_panels(self, cloud) -> None:
        """The code trajectory and the displacement field, on the figure
        cadence of the held-out diagnostics, through `log_fn` (which accepts
        Figures)."""
        if self.log_fn is None:
            return
        n_evals = len(self.crossval_history) + 1
        if n_evals % self.figure_every_evals != 0:
            return
        try:
            figs = {'motion/codes': panels.codes_figure(self.motion),
                    'motion/field': panels.field_figure(
                        self.motion, self.geometry, self.world_scale)}
            if self.motion.n_parts > 1:
                figs['motion/parts'] = panels.parts_figure(
                    self.motion, cloud.xyz.detach(), self.geometry, self.world_scale)
            self.log_fn(figs, step=int(self.iterations_run))
        except Exception as e:                       # a figure never ends a run
            print(f"  motion panels skipped ({type(e).__name__}: {e})")
        if self.motion.n_fixed:
            try:
                self.log_fn(self._breathing_panels(cloud),
                            step=int(self.iterations_run))
            except Exception as e:
                print(f"  breathing panels skipped ({type(e).__name__}: {e})")

    @torch.no_grad()
    def _breathing_panels(self, cloud) -> dict:
        """The two gate states and the sweep between them, LIVE: the ROI's
        mid coronal/axial slices at gate 0, rest and gate 1 (`show.gates_figure`)
        and the coronal slice swept gate 0 -> 1 -> 0 as uint8 frames, which
        the logger turns into a GIF (`motion/breathing`). Rendered in raw mu
        (the HU calibration exists only at export), windowed from the rest
        state's own range so every frame of one GIF shares the window."""
        geom = export_grid_geometry(self.geometry)
        ws, ss = self.world_scale, self.signal_scale
        hu = (1.0, 0.0)
        rest = show.state_slices(cloud, self.motion, geom, ws, ss, hu, 0.0)
        hi = float(np.percentile(rest[0], 99.5))
        window = (0.0, max(hi, 1e-6))
        slices = {'gate 0 (-1)': show.state_slices(cloud, self.motion, geom, ws, ss, hu, -1.0),
                  'rest (0)': rest,
                  'gate 1 (+1)': show.state_slices(cloud, self.motion, geom, ws, ss, hu, 1.0)}
        gates = show.gates_figure(slices, window=window,
                                  diff_window=(-0.3 * window[1], 0.3 * window[1]))
        frames = show.breathing_frames(cloud, self.motion, geom, ws, ss, hu,
                                       n=8, window=window)
        return {'motion/gates': gates, 'motion/breathing': frames}

    def _fit_code(self, cloud, camera, target_mm, view) -> torch.Tensor:
        """The FULL code of a view the model never trained on.

        Cloud and basis frozen; ``holdout_code_iters`` Adam steps on the free
        code alone against the scored window, plain MSE in line-integral
        units. The fixed (group) part is known from the view's group.
        """
        dev = cloud.xyz.device
        free = torch.zeros(self.motion.n_codes, device=dev, requires_grad=True)
        # the gate amplitude of a held-out view is as unknown as its free
        # code: one more scalar per fixed column, fitted with it
        amp = (torch.zeros(self.motion.n_fixed, device=dev, requires_grad=True)
               if self.motion.amp is not None else None)
        t = torch.as_tensor(np.asarray(target_mm, dtype=np.float32), device=dev)
        scale = self.world_scale * self.signal_scale
        opt = torch.optim.Adam([free] + ([amp] if amp is not None else []),
                               lr=self.mcfg['holdout_code_lr'])
        with torch.enable_grad():
            for _ in range(self.mcfg['holdout_code_iters']):
                opt.zero_grad(set_to_none=True)
                code = self.motion.full_code_from(free, view, amp)
                shown = self.motion.displaced_with_code(cloud, code)
                img = cam_mod.Camera.to_ours(kernel.render(shown, camera)['image'])
                b0, b1, a0, a1 = covered_detector_window(self.geometry, *img.shape)
                pred = img[b0:b1, a0:a1] / scale
                loss = ((pred - t) ** 2).mean()
                loss.backward()
                opt.step()
        return self.motion.full_code_from(
            free.detach(), view, None if amp is None else amp.detach())

    # -- export -------------------------------------------------------------
    def export_4d(self, roi_geometry: dict, anchors, output_path: str,
                  log=print) -> str | None:
        """The breathing sequence as a 4-D file next to the volume (see
        `export4d`; NRRD or NIfTI per ``export_4d_format``): ``export_4d``
        states at ``export_4d_mm`` on the delivered ROI grid, in the
        volume's own HU calibration. The driver calls this through the
        algorithm's ``after_save``."""
        n = int(self.mcfg['export_4d'])
        if n <= 0 or self.motion is None:
            return None
        from .export4d import export_4d
        return export_4d(self.cloud, self.motion, roi_geometry, self.world_scale,
                         self.signal_scale, anchors, output_path, n_states=n,
                         pitch_mm=self.mcfg['export_4d_mm'],
                         fmt=self.mcfg['export_4d_format'],
                         max_voxels=self._export_chunk, log=log)

    def export_ventilation(self, roi_geometry: dict, anchors, output_path: str,
                           log=print) -> str | None:
        """The regional ventilation map next to the volume (see
        `ventilation`): the SV and J maps at ``ventilation_mm``, the numbers
        and the per-primitive readings. Through ``after_save`` like the 4-D."""
        if not self.mcfg['export_ventilation'] or self.motion is None:
            return None
        from .ventilation import export_ventilation
        return export_ventilation(self.cloud, self.motion, roi_geometry, self.world_scale,
                                  self.signal_scale, anchors, output_path,
                                  pitch_mm=self.mcfg['ventilation_mm'],
                                  max_voxels=self._export_chunk, log=log)

    def _export(self, cloud, domain) -> np.ndarray:
        if self.motion is None:
            return super()._export(cloud, domain)
        v, g = self.mcfg['export_view'], self.mcfg['export_group']
        if v is not None:
            print(f"  exporting the state of view {v}")
            return super()._export(self.motion.displaced(cloud, v), domain)
        if g is not None and self.motion.fixed is not None:
            sel = self._group_views(g)
            if sel is None or len(sel) == 0:
                raise ValueError(f"no views in acquisition group {g}")
            free = torch.zeros(self.motion.n_codes, device=cloud.xyz.device)
            # the group's MEAN state: its contrast plus its mean amplitude
            code = self.motion.full_code_from(
                free, int(sel[0]), self.motion.group_mean_amplitude(sel))
            print(f"  exporting the mean state of acquisition group {g}")
            return super()._export(self.motion.displaced_with_code(cloud, code),
                                   domain)
        return super()._export(cloud, domain)
