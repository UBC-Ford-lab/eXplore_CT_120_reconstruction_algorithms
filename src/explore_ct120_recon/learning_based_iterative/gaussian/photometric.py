"""Per-view photometric nuisance parameters for a least-squares data term.

A measured projection is not a pure line integral of one object: the source
output, the detector gain and the beam profile all move a little over a
scan, and a live animal may change too. Per-projection air levelling removes
the part that is constant across the detector; what it leaves behind still
has to go somewhere, and a static volume model can put it in only one place
— the volume. MEASURED on Scan_1510 (prod_v1 residuals): the per-view mean
residual is flat for 160 views and climbs by 0.0085 log-attenuation units
over the last 40; the cloud absorbed that by stretching primitives along
those views' detector direction (the rib-to-lung streaks at 135-170 deg).

This module gives the fit the knob it was missing. For view ``v`` the model's
line integral ``p`` is compared with the data after

    p' = (1 + g_v) * p + o_v + s_v * u

with ``u`` the normalised detector column in [-0.5, 0.5]: a gain deviation
``g_v``, an offset ``o_v`` (log-attenuation units) and, optionally, a lateral
slope ``s_v``. Three numbers against a million pixels per view, so each is
heavily over-determined. Two degeneracies with the volume are closed by
construction, not by priors: the mean gain deviation and the mean offset
over the TRAINING views are subtracted before use, so a common scale stays
in the density and a common offset stays in the air level. A weak L2 prior
in units of ``ref`` (1 % by default) keeps the per-view spread from chasing
noise.

The held-out view never trains its numbers; `fit_view` solves them in closed
form (weighted least squares with the same ridge) at evaluation time, so the
held-out metric is comparable to a run without this model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure

MODES = ('none', 'offset', 'affine', 'affine_lateral')


class PhotometricModel:
    """Per-view (gain, offset, slope) with zero-mean construction."""

    #: LR multiplier on the group relative to ``--lr``: the parameters are
    #: O(0.01) and Adam moves them ~lr per step.
    DEFAULT_LR_MULTIPLIER = 1.0

    def __init__(self, n_views: int, mode: str, train_views, *,
                 ref: float = 0.01, device=None):
        if mode not in MODES or mode == 'none':
            raise ValueError(f"photometric mode must be one of {MODES[1:]}, "
                             f"got {mode!r}")
        self.mode = mode
        self.n_views = int(n_views)
        self.ref = float(ref)
        self.params = nn.Parameter(torch.zeros((self.n_views, 3),
                                               dtype=torch.float32,
                                               device=device))
        self.train_mask = torch.zeros(self.n_views, dtype=torch.bool)
        self.train_mask[list(train_views)] = True
        if device is not None:
            self.train_mask = self.train_mask.to(device)
        self._u_cache: dict = {}

    # -- which columns are live ----------------------------------------------
    @property
    def use_gain(self) -> bool:
        return self.mode in ('affine', 'affine_lateral')

    @property
    def use_slope(self) -> bool:
        return self.mode == 'affine_lateral'

    def effective(self) -> torch.Tensor:
        """(n_views, 3) with the training-view means removed from each live
        column and the dead columns forced to zero."""
        p = self.params
        m = self.train_mask.to(p.device)
        means = (p * m[:, None]).sum(dim=0) / m.sum().clamp_min(1)
        eff = p - means[None, :]
        keep = torch.tensor([self.use_gain, True, self.use_slope],
                            device=p.device, dtype=p.dtype)
        return eff * keep[None, :]

    def _u(self, n_a: int, device) -> torch.Tensor:
        key = (n_a, str(device))
        if key not in self._u_cache:
            self._u_cache[key] = ((torch.arange(n_a, device=device, dtype=torch.float32)
                                   - (n_a - 1) / 2.0) / n_a)
        return self._u_cache[key]

    # -- the forward correction ---------------------------------------------
    def apply(self, pred: torch.Tensor, view: int, scale: float,
              params: torch.Tensor | None = None) -> torch.Tensor:
        """``pred`` (n_b, n_a) in RENDERED units -> corrected prediction.

        ``scale`` converts log-attenuation units to rendered units (offset and
        slope are stored in log-attenuation units). ``params`` overrides the
        view's own triple (used for a held-out view's closed-form fit).
        """
        t = self.effective()[int(view)] if params is None else params
        g, o, s = t[0], t[1], t[2]
        out = pred * (1.0 + g) + o * scale
        if self.use_slope:
            out = out + (s * scale) * self._u(pred.shape[1], pred.device)[None, :]
        return out

    def prior(self) -> torch.Tensor:
        """Mean squared deviation over the training views in units of ``ref``."""
        eff = self.effective()[self.train_mask.to(self.params.device)]
        return ((eff / self.ref) ** 2).sum(dim=1).mean()

    # -- the held-out view ----------------------------------------------------
    @torch.no_grad()
    def fit_view(self, view: int, pred: torch.Tensor, target: torch.Tensor,
                 weights: torch.Tensor | None, scale: float,
                 reg: float = 1e-3) -> torch.Tensor:
        """Closed-form (g, o, s) for one view: weighted least squares of
        ``target - pred`` on [pred, scale, scale * u] with the TRAINING prior
        as the ridge. Writes the triple into the view's row and returns the
        effective correction. Inputs in RENDERED units, (n_b, n_a).

        The ridge is the training objective in these units and nothing else.
        A training view minimises ``mean_pix(w r^2) + reg * |x / ref|^2`` in
        log-attenuation units; times P pixels that is ``sum_pix(w r^2)
        + reg * P * |x / ref|^2``, and with residuals in rendered units
        (``r_rend = scale * r_log``) the ridge becomes
        ``reg * P * scale^2 / ref^2``. MEASURED on Scan_1510 (run m43erxsc):
        the earlier ``ridge * sum(w) / ref^2`` was 462x that, shrinking the
        held-out gain from 0.0197 to 0.0026 while its neighbours sat at 0.016,
        and the held-out PSNR read 39.05 dB for a cloud that scores 40.35.

        The solved triple is the correction the view NEEDS, i.e. what
        `effective` must return for it; `effective` subtracts the training
        means from every row, so the row is stored with those means added
        back. Storing the triple raw applied ``triple - means`` instead
        (an offset error of 0.0023 on the same run).
        """
        n_a = pred.shape[1]
        cols = [pred.reshape(-1)] if self.use_gain else []
        cols.append(torch.full_like(pred.reshape(-1), float(scale)))
        if self.use_slope:
            cols.append((self._u(n_a, pred.device)[None, :] * scale
                         ).expand_as(pred).reshape(-1))
        A = torch.stack(cols, dim=1).double()                      # (P, k)
        y = (target - pred).reshape(-1).double()
        w = (torch.ones_like(y) if weights is None
             else weights.reshape(-1).double())
        AtA = A.T @ (A * w[:, None])
        Aty = A.T @ (w * y)
        lam = float(reg) * A.shape[0] * float(scale) ** 2 / (self.ref ** 2)
        sol = torch.linalg.solve(AtA + lam * torch.eye(A.shape[1], dtype=A.dtype,
                                                       device=A.device), Aty)
        triple = torch.zeros(3, dtype=torch.float32, device=pred.device)
        i = 0
        if self.use_gain:
            triple[0] = sol[i].float(); i += 1
        triple[1] = sol[i].float(); i += 1
        if self.use_slope:
            triple[2] = sol[i].float()
        p = self.params
        m = self.train_mask.to(p.device)
        means = (p * m[:, None]).sum(dim=0) / m.sum().clamp_min(1)
        self.params[int(view)] = triple + means
        return triple

    # -- optimiser / state --------------------------------------------------
    def param_groups(self, lr: float, multiplier: float | None = None) -> list:
        mul = self.DEFAULT_LR_MULTIPLIER if multiplier is None else float(multiplier)
        return [{'params': [self.params], 'lr': lr * mul, 'name': 'photometric'}]

    def snapshot(self) -> dict:
        return {'params': self.params.detach().cpu().clone(), 'mode': self.mode,
                'ref': self.ref, 'train_mask': self.train_mask.cpu().clone()}

    @torch.no_grad()
    def restore(self, snap: dict) -> None:
        if tuple(snap['params'].shape) != tuple(self.params.shape):
            raise ValueError(f"photometric snapshot has {tuple(snap['params'].shape)}, "
                             f"model has {tuple(self.params.shape)}")
        self.params.copy_(snap['params'].to(self.params.device))

    # -- diagnostics --------------------------------------------------------
    @torch.no_grad()
    def summary(self) -> dict:
        eff = self.effective()[self.train_mask.to(self.params.device)].cpu().numpy()
        out = {'photo/offset_rms': float(np.sqrt((eff[:, 1] ** 2).mean())),
               'photo/offset_span': float(eff[:, 1].max() - eff[:, 1].min())}
        if self.use_gain:
            out['photo/gain_rms'] = float(np.sqrt((eff[:, 0] ** 2).mean()))
            out['photo/gain_span'] = float(eff[:, 0].max() - eff[:, 0].min())
        if self.use_slope:
            out['photo/slope_rms'] = float(np.sqrt((eff[:, 2] ** 2).mean()))
        return out

    @torch.no_grad()
    def figure(self, view_groups=None) -> Figure:
        """Gain, offset and slope against view index."""
        eff = self.effective().cpu().numpy()
        mask = self.train_mask.cpu().numpy()
        rows = [('gain deviation', eff[:, 0], self.use_gain),
                ('offset (log-attenuation)', eff[:, 1], True),
                ('lateral slope (per detector width)', eff[:, 2], self.use_slope)]
        rows = [r for r in rows if r[2]]
        fig = Figure(figsize=(9, 2.2 * len(rows) + 0.6), dpi=110)
        axes = fig.subplots(len(rows), 1, sharex=True, squeeze=False)[:, 0]
        v = np.arange(self.n_views)
        groups = (np.zeros(self.n_views, dtype=int) if view_groups is None
                  else np.asarray(view_groups))
        for ax, (label, y, _) in zip(axes, rows):
            for g in np.unique(groups):
                m = (groups == g) & mask
                ax.plot(v[m], y[m], '.-', ms=3, lw=0.8, label=f'group {g}')
            if (~mask).any():
                ax.plot(v[~mask], y[~mask], 'kx', ms=6, label='held-out (fitted)')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.3)
        axes[0].set_title('per-view photometric parameters')
        axes[0].legend(loc='upper left', fontsize=8)
        axes[-1].set_xlabel('view index (acquisition order)')
        fig.tight_layout()
        return fig
