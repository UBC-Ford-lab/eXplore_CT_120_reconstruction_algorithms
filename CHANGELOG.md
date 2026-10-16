# Changelog

## 0.2.2 (2026-10-16)

- Windows: the page runs jobs locally (job wrapper, cancel and memory probe
  have Windows branches) and reaches remote machines through a library SSH
  session (paramiko), since Windows' OpenSSH cannot multiplex. The login
  panel relays password and second-factor prompts as before. The backend is
  chosen by platform, or per machine (`connection = "library"`).
- Machine entries written with a tilde (`~/venv/bin/python`, `~/ct_recon`)
  work: the remote home is resolved once and substituted everywhere the
  path is used (shell commands, `sbatch --output`, log reads).
- Machines page: memory is reported on macOS, and the local figure is the
  available memory (reclaimable cache included), as on remote machines.
- `ct120-gui shortcut` without `--cwd` runs jobs in the home directory
  instead of wherever the desktop launched the icon (`/` on macOS).

## 0.2.1 (2026-10-16)

- `import explore_ct120_recon` no longer imports torch or matplotlib: the
  driver pipeline and the self-calibration load on first use, and VFF reading
  is numpy-only. `ct120-gui` starts in about a second instead of paying a
  cold torch import, which runs to minutes on a network file system.
- `ct120-gui shortcut` bakes in every server option (`--host`, `--data-dir`,
  `--cwd`, `--slots` added).

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
