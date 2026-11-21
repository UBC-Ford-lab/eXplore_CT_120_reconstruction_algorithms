# Changelog

## 0.3.1 (unreleased)

- Fixed the centre of rotation for the ray-based backends (Gaussian, dynamic
  Gaussian, voxel/INR, TIGRE); FDK output is unchanged.
- Gaussian runs save checkpoints and can be resumed.
- Gaussian export only renders the ROI, which uses much less memory.
- Preflight memory estimates are less conservative.
- Dynamic Gaussian: more detailed ventilation / tidal volume output.
- GUI: Slurm fair-share info, re-run and resume buttons, run-time estimates,
  ROI drawing on a preview, editable job notes, and other small improvements.
- Windows fixes.
- Opt-in test views (`--test-indices`, `--test-count`) and FDK warm start
  (`--warm-start`).
- Opt-in iteration snapshots (`--snapshot-dir`) and model checkpoints
  (`--save-checkpoint`, `ct120-render-checkpoint`).

## 0.3.0 (2026-10-21)

- The vendor detector unwarp is now applied when loading projections, which
  corrects the geometry of all reconstructions (including older scans).
- GUI: progress bar with time estimate, version shown on the page.
- Windows fixes.

## 0.2.3 (2026-10-19)

- Windows fixes for the GUI.

## 0.2.2 (2026-10-16)

- GUI runs on Windows.
- `~` paths in machine settings, macOS memory reporting, launcher fixes.

## 0.2.1 (2026-10-16)

- Faster startup (torch is only imported when needed).
- `ct120-gui shortcut` keeps all server options.

## 0.2.0 (2026-10-15)

- Dynamic Gaussian splatting for gated scans, with related tools.
- Scan simulator and digital breathing phantom.
- Web GUI (`ct120-gui`) for running and monitoring jobs and viewing volumes.

## 0.1.0 (2026-09-02)

First release on PyPI: FDK, ASTRA and TIGRE iterative backends, learned
reconstruction with a dense voxel grid or 3-D Gaussian splatting, shared
geometry and HU calibration, volume and projection reports.
