"""Data terms: how a predicted line integral is compared with a measured one.

Every function here maps ``(pred, target)`` — both (N,) line integrals sampled
by the ray sampler — to a scalar, and is selected by name through
``losses.build_data_term``. They are backend-agnostic: nothing below knows
whether the volume behind ``pred`` is a voxel grid, a hash-grid INR or anything
else, only that it was rendered through the shared differentiable projector.

The family, and what distinguishes them:

* ``wls``      — weighted least squares with the MEASURED inverse variance of
                 every ray (``ct_core.noise_model``: Poisson slope at the run's
                 binning + the detector's read noise, from the counts above
                 dark of the very pixel the line integral came from). The
                 maximum-likelihood data term for this detector's noise, and
                 the DEFAULT for every learned backend since 2026-09-09. Its
                 value is the reduced chi-square: 1.0 = at the photon noise.
* ``mse``      — plain L2. The objective classical SIRT descends, so it is the
                 like-for-like comparison against the classical solvers.
* ``weighted`` — L2 weighted by exp(-p), the transmission. What ``wls`` reduces
                 to when every pixel has the same flat-field count and no read
                 noise; blind to the per-pixel flat field (a 4x range across
                 the detector on Scan_1510) and to the read-noise floor. Kept
                 for comparison.
* ``huber``    — L2 near zero, L1 in the tail, with the crossover set from the
                 residual's own robust spread rather than pinned. Bounds the
                 influence of the few rays that are simply wrong (a metal clip,
                 a dead detector column) instead of letting them steer the fit.
* ``filtered`` — L2 on RAMP-FILTERED rows. Flat L2 weights every spatial
                 frequency equally; a ramp weights them proportionally to |f|,
                 which is the weighting the analytic inverse applies, so the
                 high-frequency content that carries edges stops being a
                 rounding error in the objective.
* ``wiener``   — the same idea gated by measured SNR, so the ramp's gain is
                 spent on the band where the signal is actually recoverable
                 rather than on the noise floor above it.
* ``ssim`` / ``msssim``
               — structural similarity on a projection PATCH instead of
                 per-ray error. Sensitive to local contrast and structure,
                 which per-pixel L2 is not.
* ``l1_dssim`` — (1 - w) L1 + w (1 - SSIM) on a patch, the objective of the
                 splatting literature (3DGS, R2-Gaussian). L1 because a
                 primitive's error is spatially concentrated and squaring it
                 lets one primitive dominate a step. Kept for comparison.

All of these are one-line to select and are all evaluated against the same
measured data, which is what makes them comparable.
"""

from __future__ import annotations

import torch

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 weighted by exp(-target), the transmission.

    Under pure Poisson statistics with ONE flat-field count for every pixel,
    Var(p) = 1/(N0 T) so the inverse variance is proportional to exp(-p): thick
    bone paths get down-weighted, air paths up-weighted. That is the limit of
    ``wls`` with no read noise and a flat flat-field; it cannot see the
    per-pixel count (the flat field) or the read-noise floor, because it only
    has the line integral to look at. Normalised at air = 1, so the value is
    not a chi-square.
    """
    w = torch.exp(-target)
    return (w * (pred - target) ** 2).mean()


def wls(pred: torch.Tensor, target: torch.Tensor,
        weights: torch.Tensor) -> torch.Tensor:
    """Weighted least squares in LOG-ATTENUATION units: mean_i w_i r_i^2.

    ``weights`` are the measured inverse variances of THESE rays
    (``ct_core.noise_model.inverse_variance_weights``, gathered by whoever drew
    the batch), so the value is the reduced chi-square of the batch: 1.0 means
    the residual is at the photon noise, and the gradient is the
    maximum-likelihood one for a Gaussian with that variance. A zero weight
    (a masked detector seam) drops the ray while keeping the mean over the
    batch, so the scale of the loss does not depend on how many were masked.

    Mean, not sum, so a learning rate transfers to and from ``mse`` under
    Adam; the classical summed misfit is ``sart``'s business.
    """
    if weights.shape != target.shape:
        raise ValueError(f"wls weights {tuple(weights.shape)} do not match the "
                         f"batch {tuple(target.shape)}")
    r = pred - target
    return (weights * r * r).mean()


def l1_dssim(pred: torch.Tensor, target: torch.Tensor, *, data_range: float,
             dssim_weight: float = 0.2, window_size: int = 11,
             sigma: float = 1.5) -> torch.Tensor:
    """(1 - w) L1 + w (1 - SSIM) on a projection patch — the splatting
    literature's objective. ``dssim_weight`` 0 is plain L1."""
    l1 = torch.abs(pred - target).mean()
    w = float(dssim_weight)
    if w <= 0:
        return l1
    return (1.0 - w) * l1 + w * ssim_loss(pred, target, data_range=data_range,
                                          window_size=window_size, sigma=sigma)


def make_huber_loss(delta=None, sigma_mult=1.345):
    """Robust (Huber) projection data term, SCALE-MATCHED to `mse`.

    Below the crossover δ it is EXACTLY the L2 term (r²), so it is a drop-in
    for `mse` and the `fls_weight` balance is unchanged. Above δ the penalty
    grows LINEARLY (bounded gradient) instead of quadratically, so the largest
    projection residuals — zingers, dead pixels, poly/model-mismatch rays —
    stop dominating the fit (robust regression on the sinogram). Continuity:
    at |r|=δ both pieces equal δ² and both gradients equal 2δ (C¹).

    NOTE this robustifies the PROJECTION residual (a sinogram noise-model
    choice), which is distinct from an image-domain tail prior; here it pairs
    with the frequency-weighting FLS term, which is left as plain L2.

    delta: absolute crossover in line-integral units. If None (default), δ
      ADAPTS per batch to `sigma_mult` × a robust noise-scale estimate
      (MAD = median(|r|)/0.6745), detached from the graph. This gives the
      right schedule: δ is large while residuals are large (early training,
      bone still being learned → ~pure L2, no clipping of signal rays) and
      tightens to the noise scale at convergence (robustifies noise outliers).
    sigma_mult: 1.345 is the classic Huber constant (95% Gaussian efficiency).
    """
    def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r = pred - target
        absr = r.abs()
        if delta is None:
            with torch.no_grad():
                sigma = torch.median(absr) / 0.6745
                d = (sigma_mult * sigma).clamp(min=1e-8)
        else:
            sigma = None
            d = torch.as_tensor(float(delta), dtype=r.dtype, device=r.device)
        quad = r * r                       # == mse's per-element term (r²)
        lin = 2.0 * d * absr - d * d       # scale-matched linear tail
        # Live diagnostics (detached; one compare + two reductions, negligible
        # next to ray marching). Lets the L1/L2 mix be MEASURED during training
        # rather than assumed from a Gaussian model — the assumed 24/76 split
        # holds only for Gaussian residuals, and real ones carry structured
        # model error (bone under-fit) on top of noise, so they are heavier
        # tailed. Read via loss_fn.last_* from the training loop.
        with torch.no_grad():
            _loss.last_delta = float(d)
            _loss.last_frac_quadratic = float((absr <= d).to(r.dtype).mean())
            _loss.last_resid_rms = float(r.pow(2).mean().sqrt())
            _loss.last_mad_sigma = float(sigma) if sigma is not None else None
        return torch.where(absr <= d, quad, lin).mean()

    # Live diagnostics, refreshed every call; None until the first call.
    _loss.last_delta = None            # the crossover actually used this batch
    _loss.last_frac_quadratic = None   # measured fraction with |r| <= delta
    _loss.last_resid_rms = None        # RMS residual (line-integral units)
    _loss.last_mad_sigma = None        # MAD noise-scale estimate (adaptive only)
    return _loss


def _build_ramp_kernel(
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    sqrt: bool = False,
) -> torch.Tensor:
    """Ramp (Ram-Lak) filter kernel in the frequency domain, shape (n//2+1,).

    Matches FDK's ramp: |f| with a cosine window to suppress the noisiest
    high frequencies. Cached by the caller across iterations.

    filtered_mse applies this kernel and then SQUARES the residual, so the
    quadratic form it induces is rᵀ (kernel²) r.
      - sqrt=False (default): kernel = F  ->  penalty rᵀ F² r  (ramp² weighting;
        our historical "mse+filtered" form — steeper than the paper).
      - sqrt=True: kernel = F^{1/2}  ->  penalty rᵀ F r  (ramp¹ weighting). This
        is the EXACT Najaf & Ongie (2025) FLS objective ‖F^{1/2}(Pf−y)‖², whose
        Gram matrix is PᵀFP ≈ FBP. F^{1/2} is also better-scaled (the windowed
        ramp peaks ~0.03, so squaring it made the term ~1e3× smaller and forced
        the huge fls_weight; sqrt keeps it near the ramp's own magnitude).
    """
    freqs = torch.fft.rfftfreq(n, device=device, dtype=dtype)
    ramp = freqs.abs()
    # Cosine window (same as default FDK filter)
    f_max = freqs.abs().max()
    window = torch.cos(torch.pi * freqs / (2.0 * f_max + 1e-8))
    kernel = (ramp * window).clamp(min=0.0)
    if sqrt:
        kernel = kernel.sqrt()
    return kernel


def filtered_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    ramp_kernel: torch.Tensor,
) -> torch.Tensor:
    """Filtered least-squares loss (Najaf & Ongie, 2025).

    Applies a ramp filter to the residual along the last dimension (detector
    column axis) before squaring. This amplifies high-frequency differences,
    giving the optimizer much stronger gradients for edge/detail features
    that plain MSE barely sees (because line integrals average them away).

    Mathematically: L = (residual)^T F (residual), where F is convolution
    with the ramp kernel — equivalent to MSE on ramp-filtered residuals.

    Args:
        pred:   (n_rows, n_cols) rendered projection rows.
        target: (n_rows, n_cols) measured projection rows.
        ramp_kernel: (n_cols//2+1,) frequency-domain ramp from _build_ramp_kernel.
    """
    residual = pred - target
    # Apply ramp filter in frequency domain along columns (dim=-1)
    R = torch.fft.rfft(residual, dim=-1)
    R_filtered = R * ramp_kernel
    filtered = torch.fft.irfft(R_filtered, n=residual.shape[-1], dim=-1)
    return (filtered ** 2).mean()


def build_wiener_kernel(
    sinogram: torch.Tensor,
    angles: torch.Tensor,
    da: float,
    proj_idx: int | None = None,
    downsample: int = 1,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a frequency-domain loss kernel that weights the reducible error.

    Estimates the signal and noise power spectra from the sinogram, then
    returns a Wiener-style weight: high where SNR is large (real signal
    the model should learn), low where SNR is small (noise the model
    should ignore).

        W(f) = SNR(f) / (1 + SNR(f))    where SNR = P_signal / P_noise

    The noise is estimated from the difference of adjacent projections
    (they see nearly the same object, so the difference is dominated by
    photon noise). The signal is the measured power minus the noise.

    Returns a 1-D tensor of shape (n_cols//2+1,) in the rfft convention,
    suitable for element-wise multiplication with rfft output.
    """
    import numpy as np

    sino = sinogram.numpy() if isinstance(sinogram, torch.Tensor) else sinogram
    n_angles, n_b, n_a = sino.shape

    # Pick a representative projection (default: middle angle)
    if proj_idx is None:
        proj_idx = n_angles // 2

    # Downsample to match the resolution used in training rows
    p1 = sino[proj_idx, ::downsample, ::downsample]
    p2_idx = min(proj_idx + 1, n_angles - 1)
    p2 = sino[p2_idx, ::downsample, ::downsample]

    # Row-averaged power spectra
    def _row_ps(img):
        return (np.abs(np.fft.rfft(img, axis=1)) ** 2).mean(axis=0)

    ps_measured = _row_ps(p1)
    ps_noise = _row_ps((p2 - p1) / np.sqrt(2))

    # Signal estimate: measured minus noise (clamp to avoid negative)
    ps_signal = np.maximum(ps_measured - ps_noise, 0.0)

    # SNR and Wiener weight
    snr = ps_signal / np.maximum(ps_noise, 1e-20)
    wiener = snr / (1.0 + snr)

    # Smooth slightly to avoid noisy per-bin oscillations
    from scipy.ndimage import gaussian_filter1d
    wiener = gaussian_filter1d(wiener, sigma=3)

    # Convert frequencies to lp/mm for reporting
    n_cols = p1.shape[1]
    det_px_mm = da * downsample
    freqs_lpmm = np.fft.rfftfreq(n_cols) / det_px_mm

    # Report the effective band
    half_max = wiener.max() / 2
    above_half = np.where(wiener > half_max)[0]
    if len(above_half) > 0:
        f_lo = freqs_lpmm[above_half[0]]
        f_hi = freqs_lpmm[above_half[-1]]
        print(f"  Wiener kernel: half-max band [{f_lo:.2f}, {f_hi:.2f}] lp/mm "
              f"(peak SNR={snr[above_half].max():.1f})")
    else:
        print(f"  Wiener kernel: no significant SNR detected")

    return torch.from_numpy(wiener.astype(np.float32)).to(device)


def build_phase_wiener_gate(
    sino00: torch.Tensor,
    sino01: torch.Tensor,
    da: float,
    device: torch.device | str = "cpu",
    n_angles_sample: int = 40,
    n_rows_sample: int = 256,
) -> torch.Tensor:
    """Wiener gate W(f) from the PHASE-DIFFERENCE noise estimate.

    Like build_wiener_kernel, but the noise power spectrum comes from the
    difference of two breath-phase sinograms (same angles, independent noise
    realizations) instead of adjacent projections, and statistics are pooled
    over many angles/rows. With b00 = s + n, b01 = s + n' (plus a little
    intra-phase motion), P_noise(f) = ½·PS(b00 − b01) and
    P_signal(f) = max(PS(b00) − P_noise, 0), giving

        W(f) = P_signal / (P_signal + P_noise)

    → 1 where the data has real frequency content, → 0 in the pure-noise
    band. Multiplying the FLS ramp by this gate keeps the ramp's whitening
    where the projections carry signal and removes the loss weight that the
    plain ramp puts on noise-dominated bands. (Motion in the phase diff makes
    the noise estimate conservative: the gate closes slightly early.)

    Returns a (n_a//2+1,) tensor in the rfft convention.
    """
    import numpy as np

    A = min(sino00.shape[0], sino01.shape[0])
    n_b = min(sino00.shape[1], sino01.shape[1])
    n_a = sino00.shape[2]

    # Spread of angles, central detector rows (object region)
    ang = np.linspace(0, A - 1, min(n_angles_sample, A)).astype(int)
    r0, r1 = n_b // 4, 3 * n_b // 4
    step = max(1, (r1 - r0) // max(1, n_rows_sample // len(ang)))
    rows = np.arange(r0, r1, step)

    b00 = sino00[ang][:, rows, :].reshape(-1, n_a).float().cpu().numpy()
    b01 = sino01[ang][:, rows, :].reshape(-1, n_a).float().cpu().numpy()

    def _row_ps(x):
        x = x - x.mean(axis=1, keepdims=True)  # detrend DC per row
        return (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(axis=0)

    ps_meas = _row_ps(b00)
    ps_noise = 0.5 * _row_ps(b00 - b01)
    ps_signal = np.maximum(ps_meas - ps_noise, 0.0)
    gate = ps_signal / np.maximum(ps_signal + ps_noise, 1e-30)

    from scipy.ndimage import gaussian_filter1d
    gate = np.clip(gaussian_filter1d(gate, sigma=3), 0.0, 1.0)

    freqs_lpmm = np.fft.rfftfreq(n_a) / da
    above = np.where(gate > 0.5)[0]
    if len(above):
        print(f"  Wiener gate (phase-diff SNR): W>0.5 band "
              f"[{freqs_lpmm[above[0]]:.2f}, {freqs_lpmm[above[-1]]:.2f}] lp/mm, "
              f"pooled over {len(b00)} rows x {len(ang)} angles")
    else:
        print("  Wiener gate (phase-diff SNR): WARNING — no band with W>0.5")

    return torch.from_numpy(gate.astype(np.float32)).to(device)


def wiener_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    wiener_kernel: torch.Tensor,
) -> torch.Tensor:
    """Wiener-weighted MSE: amplifies residuals at frequencies with high SNR.

    Same mechanics as filtered_mse but using the Wiener kernel instead of
    a ramp filter. Focuses the optimizer on frequencies where there's real
    signal to learn, while suppressing noise-dominated frequencies.

    Args:
        pred:   (n_rows, n_cols) rendered projection rows.
        target: (n_rows, n_cols) measured projection rows.
        wiener_kernel: (n_cols//2+1,) from build_wiener_kernel.
    """
    residual = pred - target
    R = torch.fft.rfft(residual, dim=-1)
    R_filtered = R * wiener_kernel
    filtered = torch.fft.irfft(R_filtered, n=residual.shape[-1], dim=-1)
    return (filtered ** 2).mean()


def ssim_loss(pred2d, target2d, data_range: float,
              window_size: int = 11, sigma: float = 1.5,
              ms: bool = False, ms_weights=None) -> torch.Tensor:
    """Differentiable structural loss = 1 - SSIM (or 1 - MS-SSIM).

    ``pred2d``/``target2d``: (H,W) or (B,H,W) projection patch(es). Returns a
    scalar in [0, ~1].

    The SSIM itself is ``ct_core.ssim`` — the same code the projection metric
    (``ct_core.projection_diag.ssim_2d``) runs, so a run's structural LOSS and
    its reported diag/ssim cannot disagree about what SSIM means. That module
    also owns the numerical guards this needs: fp32 under autocast, clamped
    variances, and floored bases for the multi-scale fractional powers.
    """
    from ...ct_core.ssim import msssim as _msssim, ssim as _ssim

    if ms:
        return 1.0 - _msssim(pred2d, target2d, data_range=data_range,
                             window_size=window_size, sigma=sigma,
                             weights=ms_weights).mean()
    return 1.0 - _ssim(pred2d, target2d, data_range=data_range,
                       window_size=window_size, sigma=sigma).mean()
