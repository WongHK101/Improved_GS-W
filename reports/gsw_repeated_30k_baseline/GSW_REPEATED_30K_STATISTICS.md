# GSW_REPEATED_30K_STATISTICS

- Code/tag: `gsw-strict-baseline-v2` / `ddc6d8702b2e838dc989d612ca23fb311b79f280`
- Scene: `self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Runs: R1/R2/R3, all resolution=1, iterations=30000, frozen manifest, seed 0, no in-training test evaluation.

## Summary Metrics

| mode | PSNR mean/std/range | SSIM mean/std/range | LPIPS mean/std/range |
|---|---:|---:|---:|
| strict_intrinsic | 12.644473 / 0.226971 / 0.448606 | 0.434716 / 0.009397 / 0.018789 | 0.636165 / 0.009725 / 0.019424 |
| strict_nearest_train | 12.485378 / 0.273559 / 0.516712 | 0.427077 / 0.006652 / 0.011620 | 0.637318 / 0.009445 / 0.018534 |
| legacy_target_rgb | 12.555741 / 0.236737 / 0.454119 | 0.418882 / 0.006235 / 0.011742 | 0.648162 / 0.011986 / 0.022029 |

## Key Questions

1. strict_intrinsic PSNR range: `0.448606` dB.
2. strict_intrinsic LPIPS range: `0.019424`.
3. Historical-clean 1.235 dB gap vs clean repeated range: strict_intrinsic range is `0.448606` dB and legacy_target_rgb range is `0.454119` dB, both below 1.235 dB; clean run variance does not explain the full historical gap.
4. Gaussian count vs strict_intrinsic PSNR correlation: `0.997543`; Gaussian count vs strict_intrinsic LPIPS correlation: `0.985315`. With n=3 this is descriptive only.
5. 1000-iteration non-determinism persists into 30k as visible metric variance, but not enough here to cover the old 1.235 dB discrepancy.
6. The earlier single strict value around 11.156 dB is below all three clean strict_intrinsic runs (`12.440201` to `12.888807`), so it appears unusually low relative to this clean repeated benchmark.
7. strict_nearest_train is not stably better than strict_intrinsic. Per-run PSNR differences nearest - intrinsic: `[-0.058661, -0.093155, -0.325471]`.
8. legacy_target_rgb is not a reliable strict advantage. Per-run PSNR differences legacy - strict_intrinsic: `[-0.07284, -0.067327, -0.126028]`.

## Runtime

| run | train min | peak GPU MB | Gaussians | checkpoint bytes |
|---|---:|---:|---:|---:|
| R1 | 91.08 | 11008 | 136793 | 100696283 |
| R2 | 92.60 | 11453 | 157698 | 106382419 |
| R3 | 91.93 | 11123 | 143129 | 102419675 |

## Train View Diagnostic

Train-view conditioned and intrinsic diagnostic metrics are in `GSW_REPEATED_30K_TRAIN_VIEW_STATS.csv`. They use PNG-level PSNR and a global diagnostic SSIM; held-out test metrics remain the authoritative metrics from `metrics.py`.
