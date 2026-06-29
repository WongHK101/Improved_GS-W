# Baseline Comparison

## Historical Results

| Method | Protocol | PSNR | SSIM | LPIPS |
| --- | --- | ---: | ---: | ---: |
| official 3DGS | clean historical r1 30k | 10.983210 | 0.240177 | 0.575440 |
| GS-W | historical legacy target-RGB r1 30k | 12.385471 | 0.426827 | 0.642460 |
| strict GS-W | clean strict r1 30k, strict_intrinsic | 11.156013 | 0.399699 | 0.722331 |
| strict GS-W | clean strict r1 30k, strict_nearest_train | 11.038421 | 0.395382 | 0.719736 |

## Interpretation

Historical GS-W is non-strict because held-out RGB enters the appearance map generator. It should not be reported as a strict held-out baseline.

Strict GS-W improves PSNR and SSIM over historical official 3DGS on Trackmobile but is worse in LPIPS. The PSNR improvement is only about `+0.17` dB, so it is not a decisive advantage.

Strict-nearest-train uses train RGB only and slightly improves LPIPS over strict-intrinsic, but loses PSNR and SSIM. It is not stable enough to be preferred.
