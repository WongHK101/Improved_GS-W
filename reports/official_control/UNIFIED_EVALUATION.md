# UNIFIED_EVALUATION

- Metrics: PSNR, kornia SSIM with window size 3, LPIPS AlexNet.
- LPIPS call: `lpips.LPIPS(net='alex')(render, gt, normalize=True)`.
- Image protocol: full image, no crop, no resize, RGB tensors in `[0, 1]`.
- Results CSV: `G:\wl3dgs\Improved_GS-W\reports\official_control\OFFICIAL_SMOKE_UNIFIED_RESULTS.csv`
- Per-view CSV: `G:\wl3dgs\Improved_GS-W\reports\official_control\OFFICIAL_SMOKE_UNIFIED_PER_VIEW.csv`

| group | label | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| official_smoke | smoke_10 | 9.163085 | 0.342920 | 0.982918 |
| official_smoke | smoke_300 | 12.502751 | 0.413668 | 0.942117 |
