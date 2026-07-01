# FINAL_BASE_SELECTION

Recommendation: `B`

Trackmobile alone should not decide the long-term base because the metric directions are clearly mixed: GS-W strict_intrinsic is better on PSNR/SSIM, while official 3DGS is much better on LPIPS. Verify one stronger appearance-variation scene and one ordinary scenic scene next.

If recommendation is B, the next two scenes should be selected as:

1. A high appearance-variation/self-captured outdoor scene, to stress exposure/shadow robustness.
2. A comparatively ordinary scenic scene, to check whether any GS-W/official trend is scene-specific.

No new algorithm should be implemented before this base-choice uncertainty is resolved.
