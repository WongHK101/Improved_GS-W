# Intrinsic Feature Distribution Audit

This audit checks whether the all-zero point feature used by `strict_intrinsic` is an obvious out-of-distribution input.

Checkpoint:

`G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover\ckpts_point_cloud\iteration_30000`

Train views evaluated: 13.

Summary:

- Mean feature L2 norm: `1.731735`
- Median feature L2 norm averaged across views: `1.067834`
- Mean feature L2 std: `1.881375`
- Element mean: `-0.002776`
- Element std: `0.369429`
- Near-zero element ratio: `0.144033`
- Near-zero vector ratio: `0.0`
- Zero-feature L2 norm: `0.0`
- Zero-feature L2 z-score: `-0.920462`

Interpretation:

The zero-feature input is lower energy than ordinary train-conditioned features, but it is not an extreme outlier under this aggregate L2 statistic. The stronger concern is semantic: the model was not explicitly trained to make zero point features a canonical neutral condition.

Train-view rendering confirms this: normal train-conditioned rendering is better than intrinsic rendering by about `0.669206` dB PSNR, `0.039548` SSIM, and `0.027605` LPIPS, but intrinsic rendering does not collapse.
