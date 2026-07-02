# BASELINE_COMPLETION_AUDIT_SUMMARY

This package is an audit/preflight deliverable. It does not start the remaining 9-scene GS-W long run because the GPT instruction states that execution of the above long run should wait for GPT review.

- Corrected GS-W strict designated successes: `3/12`.
- Corrected GS-W strict missing/pending: `9/12`.
- GS-W successful-scene mean/median PSNR: `17.059891` / `18.879516`.
- GS-W successful-scene mean/median SSIM: `0.601049` / `0.651654`.
- GS-W successful-scene mean/median LPIPS: `0.554201` / `0.534607`.
- Existing official 12-scene classes: A=3, B=9, C=0, D=0, E=0.
- Verified official common scenes currently compared: `3`.
- Splatfacto-W: not strict fair; transductive/right-half protocol.
- Luminance-GS: unresolved; bounded diagnostics recommended before full rerun.
- WildGaussians: current numeric rows invalid/remove; likely adapter/config integration issue, not a fair strict result.

Next GPT decision needed: approve or reject the remaining single-run GS-W strict 30k schedule for the 9 pending scenes.
