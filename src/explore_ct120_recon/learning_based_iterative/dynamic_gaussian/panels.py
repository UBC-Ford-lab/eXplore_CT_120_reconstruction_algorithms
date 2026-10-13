"""Figures for judging a motion fit by eye: the code trajectory and the field.

Plain `matplotlib.figure.Figure` objects (no pyplot state), handed to the
driver's ``log_fn`` which routes Figures to the local plots folder and W&B.
"""
from __future__ import annotations

import numpy as np
import torch
from matplotlib.figure import Figure


def codes_figure(motion) -> Figure:
    """Free code components against view index, coloured by acquisition group.

    What to look for: a smooth, quasi-periodic trace is a physiological
    state; white noise at the prior's scale is a code that explains nothing;
    a trend across the scan is drift the static part should have absorbed.
    """
    z = motion.codes.detach().cpu().numpy()
    labels = [f'z[{k}]' for k in range(z.shape[1])]
    if getattr(motion, 'amp', None) is not None:
        a = motion.amp.detach().cpu().numpy()
        z = np.concatenate([z, a], axis=1)
        labels += [f'gate amplitude a[{j}]' for j in range(a.shape[1])]
    v = np.arange(z.shape[0])
    groups = (np.zeros(len(v), dtype=int) if motion.view_groups is None
              else motion.view_groups.cpu().numpy())
    fig = Figure(figsize=(9, 2.2 * z.shape[1] + 0.6), dpi=110)
    axes = fig.subplots(z.shape[1], 1, sharex=True, squeeze=False)[:, 0]
    for k, ax in enumerate(axes):
        for g in np.unique(groups):
            m = groups == g
            ax.plot(v[m], z[m, k], '.-', ms=3, lw=0.8, label=f'group {g}')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel(labels[k])
        ax.grid(alpha=0.3)
    axes[0].set_title('per-view state codes (free part'
                      + (' + gate amplitude)' if len(labels) > motion.codes.shape[1] else ')'))
    if len(np.unique(groups)) > 1:
        axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('view index (acquisition order)')
    fig.tight_layout()
    return fig


@torch.no_grad()
def field_figure(motion, geometry: dict, world_scale: float,
                 n: int = 96) -> Figure:
    """Displacement magnitude in mm on the mid-axial plane of the support.

    Left: the phase-contrast field alone (fixed column +1, free part 0) when
    the scan has groups, else the mean field over views. Right: the RMS over
    views of the free part. Both on a regular grid of REST positions inside
    the support's outer box, at its centre z.
    """
    dev = motion.codes.device
    lo = motion.support.outer_lo.to(dev)
    hi = motion.support.outer_hi.to(dev)
    xs = torch.linspace(float(lo[0]), float(hi[0]), n, device=dev)
    ys = torch.linspace(float(lo[1]), float(hi[1]), n, device=dev)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    zc = 0.5 * (lo[2] + hi[2])
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1),
                       torch.full_like(gx.reshape(-1), float(zc))], dim=1)

    def mag(code):
        return (motion.displacement(pts, code).norm(dim=1) / world_scale
                ).reshape(n, n).cpu().numpy()

    if motion.fixed is not None:
        code = torch.zeros(motion.n_fields, device=dev)
        code[0] = 1.0
        left, left_title = mag(code), 'phase contrast field (half-difference), mm'
    else:
        codes = torch.stack([motion.full_code(v) for v in range(motion.n_views)])
        left, left_title = mag(codes.mean(dim=0)), 'mean state field, mm'
    sq = None
    step = max(1, motion.n_views // 64)                 # a sample of views
    views = list(range(0, motion.n_views, step))
    for v in views:
        code = motion.full_code(v).clone()
        if motion.fixed is not None:
            code[:motion.n_fixed] = 0.0                 # free part only
        m = mag(code)
        sq = m ** 2 if sq is None else sq + m ** 2
    right = np.sqrt(sq / len(views))

    extent = [float(lo[1]) / world_scale, float(hi[1]) / world_scale,
              float(lo[0]) / world_scale, float(hi[0]) / world_scale]
    fig = Figure(figsize=(10, 4.6), dpi=110)
    axes = fig.subplots(1, 2)
    for ax, img, title in ((axes[0], left, left_title),
                           (axes[1], right, 'free-state RMS over views, mm')):
        im = ax.imshow(img, origin='lower', extent=extent, cmap='magma')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('x (mm)')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'displacement magnitude at z = {float(zc) / world_scale:.1f} mm',
                 fontsize=10)
    fig.tight_layout()
    return fig


@torch.no_grad()
def parts_figure(motion, xyz: torch.Tensor, geometry: dict, world_scale: float,
                 slab_mm: float = 1.5, max_points: int = 60_000) -> Figure:
    """The learned partition (``n_parts`` > 1): every primitive within
    ``slab_mm`` of the coronal mid-plane of the support, drawn at its REST
    position in the (x, z) plane, coloured by its argmax part and faded by
    how committed it is (alpha = max p); beside it the histogram of the
    membership entropy over the support and each part's gate field
    (half-difference magnitude, mm) on the same plane.

    What to look for: a partition that follows anatomy (lung / wall and
    heart / static) without having been told it, with committed (opaque)
    interiors and a thin uncertain band at the interfaces; a uniform grey
    cloud means the parts have not separated; part fields that differ only
    at the interface are the sliding this buys.
    """
    p = motion.membership(tied_ok=True)          # uniform (grey) while tied
    if p is None:
        raise ValueError("parts_figure needs a model with several parts")
    dev = xyz.device
    lo = motion.support.outer_lo.to(dev)
    hi = motion.support.outer_hi.to(dev)
    yc = 0.5 * (lo[1] + hi[1])
    w = motion.support.weight(xyz)
    sel = ((xyz[:, 1] - yc).abs() <= 0.5 * slab_mm * world_scale) & (w > 0)
    idx = sel.nonzero(as_tuple=True)[0]
    if idx.numel() > max_points:
        idx = idx[torch.randperm(idx.numel(), device=dev)[:max_points]]
    pts = xyz[idx] / world_scale
    pm, arg = p[idx].max(dim=1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1) / np.log(motion.n_parts)
    ent = ent[w > 0].cpu().numpy()

    K = motion.n_parts
    fig = Figure(figsize=(4.2 * (K + 1), 4.6), dpi=110)
    axes = fig.subplots(1, K + 1)
    colours = np.array([[0.00, 0.45, 0.70], [0.84, 0.37, 0.00], [0.00, 0.62, 0.45],
                        [0.80, 0.47, 0.65], [0.90, 0.62, 0.00], [0.34, 0.71, 0.91]])
    rgba = np.zeros((idx.numel(), 4))
    rgba[:, :3] = colours[arg.cpu().numpy() % len(colours)]
    rgba[:, 3] = np.clip(pm.cpu().numpy(), 0.15, 1.0)
    ax = axes[0]
    ax.scatter(pts[:, 0].cpu().numpy(), pts[:, 2].cpu().numpy(), s=1.5, c=rgba,
               linewidths=0)
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    frac = [float((arg == j).float().mean()) for j in range(K)]
    ax.set_title('membership on the coronal mid-slab: '
                 + ', '.join(f'part {j} {100 * f:.0f} %' for j, f in enumerate(frac)),
                 fontsize=8)
    ins = ax.inset_axes([0.60, 0.03, 0.38, 0.20])
    ins.hist(ent, bins=40, range=(0, 1), color='0.3')
    ins.set_xlabel('membership entropy', fontsize=6, labelpad=1)
    ins.set_yticks([])
    ins.tick_params(labelsize=6)

    n = 96
    xs = torch.linspace(float(lo[0]), float(hi[0]), n, device=dev)
    zs = torch.linspace(float(lo[2]), float(hi[2]), n, device=dev)
    gx, gz = torch.meshgrid(xs, zs, indexing='ij')
    grid = torch.stack([gx.reshape(-1), torch.full_like(gx.reshape(-1), float(yc)),
                        gz.reshape(-1)], dim=1)
    code = torch.zeros(motion.n_fields, device=dev)
    if motion.fixed is not None:
        code[0] = 1.0
        title = 'gate half-difference field of part {j}, mm'
    else:
        codes = torch.stack([motion.full_code(v) for v in range(motion.n_views)])
        code = codes.mean(dim=0)
        title = 'mean state field of part {j}, mm'
    extent = [float(lo[0]) / world_scale, float(hi[0]) / world_scale,
              float(lo[2]) / world_scale, float(hi[2]) / world_scale]
    vmax = None
    for j in range(K):
        onehot = torch.zeros((grid.shape[0], K), device=dev)
        onehot[:, j] = 1.0
        mag = (motion.displacement(grid, code, parts=onehot).norm(dim=1)
               / world_scale).reshape(n, n).cpu().numpy()
        vmax = max(float(mag.max()), 1e-6) if vmax is None else vmax
        im = axes[j + 1].imshow(mag.T, origin='lower', extent=extent, cmap='magma',
                                vmin=0.0, vmax=vmax, aspect='equal')
        axes[j + 1].set_title(title.format(j=j), fontsize=8)
        axes[j + 1].set_xlabel('x (mm)')
        fig.colorbar(im, ax=axes[j + 1], fraction=0.046)
    fig.suptitle(f'{K} motion parts, coronal plane y = {float(yc) / world_scale:.1f} mm'
                 + ('' if motion.parts_active else '  [TIED: not yet activated]'),
                 fontsize=10)
    fig.tight_layout()
    return fig
