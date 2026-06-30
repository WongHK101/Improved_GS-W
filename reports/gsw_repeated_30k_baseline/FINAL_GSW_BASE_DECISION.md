# FINAL_GSW_BASE_DECISION

## Decision

`B. Pause GS-W method development until a clean official 3DGS matched repeated control is run.`

## Basis

- strict GS-W repeated-run variance is measurable but not catastrophic: strict_intrinsic PSNR mean/std/range = `12.644473` / `0.226971` / `0.448606` dB; LPIPS mean/std/range = `0.636165` / `0.009725` / `0.019424`.
- clean repeated-run variance does not explain the old historical-clean `1.235 dB` gap.
- strict_intrinsic is consistently the best GS-W strict mode by PSNR in this run set; strict_nearest_train is not better.
- legacy_target_rgb remains non-strict diagnostic only and must not justify GS-W continuation.
- Existing official 3DGS result is marked `not sufficiently matched`; it has useful provisional values (PSNR `11.76401138305664`, SSIM `0.21798548102378845`, LPIPS `0.5866167545318604`) but lacks frozen split/provenance evidence.

## Practical Recommendation

Do not implement new GS-W-based algorithm modules yet. Run a clean official 3DGS matched repeated control under the same frozen split, resolution, iteration count, metrics, and provenance logging; then decide whether GS-W is a viable base.
