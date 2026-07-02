# THREE_SCENE_BASE_SELECTION

- Final choice: `B. official 3DGS`
- Reason: Official matches or exceeds GS-W on PSNR/SSIM in both new scenes; GS-W does not show reproducible cross-scene strict benefit.
- Stability note: the triggered `M-G3` GS-W run completed with return code 0 and produced renders, but its training log shows sustained `Loss=nan` near the final quarter of training and the held-out metrics collapsed to PSNR `7.014448`, SSIM `0.004705`, LPIPS `1.044064`. This is treated as method instability under the frozen strict protocol, not as grounds for a fourth run.

Delta is `GS-W mean - official mean`; positive PSNR/SSIM is better for GS-W, positive LPIPS is worse for GS-W.

| scene | dPSNR | dSSIM | dLPIPS |
|---|---:|---:|---:|
| H | -0.776238 | -0.019291 | 0.199000 |
| M | -5.728619 | -0.280582 | 0.292680 |
| Trackmobile | 0.673809 | 0.083215 | 0.109726 |

## Required discussion points

- PSNR/SSIM/LPIPS are compared by repeated-run means, sample std, min, max and range in `THREE_SCENE_BASELINE_COMPARISON.csv`.
- Per-view consistency is available in `TWO_SCENE_SCREENING_PER_VIEW.csv` and the accepted Trackmobile per-view CSV.
- Training cost, peak memory, checkpoint size and Gaussian count are in `TWO_SCENE_SCREENING_RUN_METADATA.csv` and `TWO_SCENE_SCREENING_RESULTS.csv`.
- Protocol: all new-scene GS-W renders use strict_intrinsic; no test RGB conditioning, no half-image metrics, no crop/resize/mask.
