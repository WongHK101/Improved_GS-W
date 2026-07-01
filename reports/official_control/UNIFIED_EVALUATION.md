# UNIFIED_EVALUATION

- Metrics: PSNR, kornia SSIM with window size 3, LPIPS AlexNet.
- LPIPS call: `lpips.LPIPS(net='alex')(render, gt, normalize=True)`.
- Image protocol: full image, no crop, no resize, RGB tensors in `[0, 1]`.
- Results CSV: `G:\wl3dgs\Improved_GS-W\reports\official_control\UNIFIED_OFFICIAL_GSW_RESULTS.csv`
- Per-view CSV: `G:\wl3dgs\Improved_GS-W\reports\official_control\UNIFIED_OFFICIAL_GSW_PER_VIEW.csv`

| group | label | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| official_3dgs | O1 | 12.226193 | 0.339039 | 0.523305 |
| official_3dgs | O2 | 12.383015 | 0.386731 | 0.530638 |
| official_3dgs | O3 | 11.302783 | 0.328732 | 0.525375 |
| gsw_strict_intrinsic | R1 | 12.440201 | 0.425199 | 0.626745 |
| gsw_strict_intrinsic | R2 | 12.888807 | 0.434959 | 0.646169 |
| gsw_strict_intrinsic | R3 | 12.604411 | 0.443988 | 0.635582 |
