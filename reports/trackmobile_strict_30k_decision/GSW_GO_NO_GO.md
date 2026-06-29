# GS-W Go/No-Go Decision

## Internal Rules

Continue GS-W if at least one strict mode is comparable to or better than official 3DGS, has no severe visual collapse, and the legacy/strict gap can be explained by a fixable method problem.

Abandon GS-W as base if the best strict mode is worse than official 3DGS by more than about 0.3 dB and SSIM/LPIPS are also worse, or if strict rendering collapses.

## Evidence

- Best strict PSNR: `11.156013`, higher than official 3DGS by about `0.172803` dB.
- Best strict SSIM: `0.399699`, higher than official 3DGS.
- Best strict LPIPS: `0.719736`, worse than official 3DGS (`0.575440`).
- Intrinsic train rendering is weaker than normal train-conditioned rendering, but it does not collapse.
- Zero-feature input is not an extreme OOD case under aggregate feature L2 z-score (`-0.920462`).
- Legacy target-RGB evaluation does not improve this clean strict-trained checkpoint; therefore the historical GS-W advantage cannot be assumed to come entirely from the same effect on this run, but the old historical result remains non-strict and cannot be cited as held-out evidence.

## Decision

No-go as the default main paper base in its vanilla strict form.

Conditional go only as a component/reference if the next method explicitly strengthens canonical appearance learning or introduces a constrained appearance/radiometric decomposition. The current strict GS-W result is not strong enough to justify using GS-W unchanged as the central method base.
