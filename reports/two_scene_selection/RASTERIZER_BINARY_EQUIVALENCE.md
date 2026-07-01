# RASTERIZER_BINARY_EQUIVALENCE

## Runtime Binary

- diff_gaussian_rasterization module: `D:\anaconda\envs\3dgs\lib\site-packages\diff_gaussian_rasterization\__init__.py`
- diff_gaussian_rasterization _C: `D:\anaconda\envs\3dgs\lib\site-packages\diff_gaussian_rasterization\_C.cp310-win_amd64.pyd`
- _C SHA256: `b6379033dc6466cbea265c3fecbb7f53afa2343fcbd123dc14035a02e0c42cb9`
- settings signature: `(image_height: int, image_width: int, tanfovx: float, tanfovy: float, bg: torch.Tensor, scale_modifier: float, viewmatrix: torch.Tensor, projmatrix: torch.Tensor, sh_degree: int, campos: torch.Tensor, prefiltered: bool, debug: bool, antialiasing: bool)`
- forward signature: `(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None)`
- SparseGaussianAdam present: `False`

## Equivalence Conclusion

Both official 3DGS and GS-W runs execute in the same conda environment (`3dgs`) and import the same installed `diff_gaussian_rasterization`, `simple_knn`, and `fused_ssim` modules recorded in `RASTERIZER_BINARY_CHECKSUMS.csv`.
Therefore official O1/O2/O3 and GS-W R1/R2/R3 use the same runtime rasterizer/simple-knn/fused-ssim binaries on this machine.

## Submodule vs Installed Binary

The clean official clone records submodule commit `9c5c2028...` for diff-gaussian-rasterization, but the actual training imported the installed site-packages binary. Without rebuilding or package provenance metadata tying the binary to a git commit, the exact source commit of the installed binary remains `binary provenance unknown`.
This round does not recompile extensions.

## Risk

Because both methods import the same installed binary, binary differences should not explain official-vs-GS-W quality differences. The unknown binary-to-submodule correspondence remains a provenance limitation, not an inter-method mismatch in this control.
