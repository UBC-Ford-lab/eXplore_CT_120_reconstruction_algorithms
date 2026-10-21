# Bundled scanner calibrations

Detector unwarp fields shipped with the package, one per detector serial and
vendor-geometry epoch (`detector_warp_<serial>_cor<scan.xml CentreOfRotation>.npz`).
They are looked up after the user's own calibration directory
(`ct_core.paths.calibration_dir()`), so a locally measured file wins.

| file | detector | epoch (CentreOfRotation / full.rfan) | fitted on | field |
|---|---|---|---|---|
| `detector_warp_CAC91105CP_cor1744.96.npz` | CAC91105CP (eXplore CT 120, UBC) | 1744.96 / 15899.36 (2024 recalibration onward) | Scan_1988, Scan_1989, Scan_2005 (2026) | 13.8 px mean, 28.7 px max |
| `detector_warp_CAC91105CP_cor1758.48.npz` | CAC91105CP | 1758.48 / 15849.40 (2022, before the recalibration) | Scan_1510's own vendor table | 16.4 px mean, 28.9 px max |

These files are FALLBACKS. Every scan folder carries the vendor's own
unwarp table, `transform1` (a VTK grid transform; rows bottom-up, channel 0
column shift, channel 1 row shift, in raw pixels), and the loader applies
that table directly when it is present: it is exact, per-epoch by
construction, and its MD5 is the `transform_checksum` the vendor stamps into
its uwarp frames. The stored records serve scans whose table is missing.
A field can also be measured from paired `acq-*` / `Corrected/uwarp-*` frames
by registering their photon noise, which the package does by itself the first
time it meets a scan of an unknown epoch that has no table but carries those frames
(`python -m explore_ct120_recon.ct_core.detector_warp measure <scan>` does
the same by hand; `export` averages several fits). An epoch whose raw frames
were verified to reconstruct round is recorded with `... none <scan> --note`.
A scan of an epoch with neither record, on a detector that has records, is
refused until one exists (`--detector-warp off` overrides).
