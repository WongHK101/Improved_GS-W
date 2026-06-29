# Metric Pipeline Audit

Unified re-evaluation used the same independent image loader and PSNR/MSE/MAE code for all existing render directories. SSIM and LPIPS were computed through the current installed `kornia` and `lpips` packages when available.

- Torch metric note: `device=cuda:0; lpips=alex; normalize=True; ssim=kornia.metrics.ssim(window=3)`
- Per-view pairs evaluated: `10`.

| Group | Views | PSNR | SSIM | LPIPS | MAE | Status |
|---|---:|---:|---:|---:|---:|---|
| historical_official_3dgs | 2 | 10.983210 | 0.297969 | 0.556248 | 0.212198 | ok |
| historical_gsw_legacy | 2 | 12.385472 | 0.426827 | 0.642460 | 0.175132 | ok |
| clean_gsw_legacy_target_rgb | 2 | 11.150374 | 0.393910 | 0.724205 | 0.214880 | ok |
| clean_gsw_strict_intrinsic | 2 | 11.156010 | 0.399699 | 0.722331 | 0.214480 | ok |
| clean_gsw_strict_nearest_train | 2 | 11.038423 | 0.395382 | 0.719736 | 0.217402 | ok |

## Pipeline checks

- Render and GT images are paired by identical PNG filename inside each method directory.
- Full RGB images are used; no mask or crop is applied.
- Values are decoded through PIL as RGB and scaled to `[0, 1]`.
- PSNR is computed from full-image MSE as `20 * log10(1 / sqrt(MSE))`, matching `utils.image_utils.psnr`.
- Existing `results.json` values are copied into `UNIFIED_REEVALUATION.csv` for comparison.
