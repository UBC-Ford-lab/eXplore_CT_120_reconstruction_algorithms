# Changelog

## 0.2.0 (2026-10-15)

- Dynamic Gaussian splatting (`ct120-learned --algorithm gaussian_dynamic`):
  Gaussian splatting with a deformation field for gated scans of breathing
  animals, with `ct120-motion-show`, `ct120-export-4d`, `ct120-ventilation`
  and `ct120-breathing-phantom`.
- Scan simulator (`ct120-simulate-scan`) and a digital breathing phantom.
- Web page (`ct120-gui`, the `gui` extra): forms generated from the drivers'
  argparse, input checks, jobs on the local machine, over ssh or through
  Slurm, live logs, a volume viewer that slices `.vff` files where they
  live, and a desktop launcher (`ct120-gui shortcut`).

## 0.1.0 (2026-09-02)

First release on PyPI: FDK, ASTRA and TIGRE iterative backends, learned
reconstruction with a dense voxel grid or 3-D Gaussian splatting, shared
geometry and HU calibration, volume and projection reports.
