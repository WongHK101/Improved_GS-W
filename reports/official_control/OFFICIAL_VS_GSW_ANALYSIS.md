# OFFICIAL_VS_GSW_ANALYSIS

## Mean Delta

Delta is `GS-W strict_intrinsic mean - official 3DGS mean`.

- PSNR delta: `0.673809`
- SSIM delta: `0.083215`
- LPIPS delta: `0.109726` (positive means GS-W is worse)

## Range Overlap

- PSNR min/max overlap: `False`
- SSIM min/max overlap: `False`
- LPIPS min/max overlap: `False`

## Notes

- O1/O2/O3 and R1/R2/R3 are repeated runs, not paired deterministic seeds.
- Efficiency comparisons are descriptive only because official and GS-W differ in implementation and extra network components.
- GS-W uses `data_device=cuda`; official uses the official default `data_device=cuda` in this control.
