# BASELINE_COMPLETION_AUDIT_SUMMARY

This package is an audit/preflight deliverable. It does not start the remaining 9-scene GS-W long run because the GPT instruction states that execution of the above long run should wait for GPT review.

- Corrected GS-W strict designated successes: `3/12`.
- Corrected GS-W strict missing/pending: `9/12`.
- GS-W successful-scene mean/median PSNR: `17.059891` / `18.879516`.
- GS-W successful-scene mean/median SSIM: `0.601049` / `0.651654`.
- GS-W successful-scene mean/median LPIPS: `0.554201` / `0.534607`.
- Existing official 12-scene classes: A=3, B=9, C=0, D=0, E=0.
- Verified official common scenes currently compared: `3`.
- Splatfacto-W: not strict-main usable. Current evidence corrects the earlier half-image claim: saved 12-scene configs are full-image (`eval_right_half=false`) with average appearance, but no per-view render/GT artifacts exist for unified re-evaluation and split provenance is not frozen-manifest verified.
- Luminance-GS: current local env/adapter state invalid; bounded import/log/adapter diagnostics found pycolmap `SceneManager` API mismatch.
- WildGaussians: current numeric rows invalid/remove; bounded reader diagnostics pass, but render/checkpoint/appearance integration remains invalid/dark.

Next GPT decision needed: approve or reject the remaining single-run GS-W strict 30k schedule for the 9 pending scenes; separately decide whether to repair Luminance-GS/WildGaussians wrappers.
