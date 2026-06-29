# Trackmobile Historical Config Audit

This audit fixes the protocol boundary for the strict GS-W decision run. It separates file-backed facts from inferred defaults and unknowns.

## Evidence Sources

- Historical metrics summary: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\summary\formal_metrics_iter30000.csv`
- Dataset adapter summary: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\summary\dataset_adapters.csv`
- Historical GS-W output: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\gs_w_r1_iter30000\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Historical official 3DGS output: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\official_3dgs_r1_iter30000\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Frozen split manifest in this repo: `splits\TRACKMOBILE_SPLIT.json`

## File-Backed Facts

### Split

- Registered images: 15.
- Train images: 13.
- Test images: 2.
- Frozen test images: `0001.jpg`, `0009.jpg`.
- Frozen train hash: `9ad80ed0ae4c100bb4e261e7989575110e7eeb0d043222b17860fd8e8f4e1b3c`.
- Frozen test hash: `b48466af92f75b2a88c79d296c60786e151baa3511dfee97b0637ee2c0397962`.
- Repo manifest SHA256 previously recorded: `c65cc61677ffb46cd329d57fc72d30ed40d42e0f9a47d75c5a0931950c9f2cd7`.

### Historical Official 3DGS Trackmobile Result

- Source path: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Resolution: `1`
- Iterations: `30000`
- SH degree: `3`
- White background: `False`
- Eval: `True`
- Mean PSNR: `10.9832096099854`
- Mean SSIM: `0.240176811814308`
- Mean LPIPS: `0.575440406799316`
- Per-view files in `per_view.json`:
  - `00000.png`: PSNR `9.990983963012695`, SSIM `0.21918031573295593`, LPIPS `0.6109516024589539`
  - `00001.png`: PSNR `11.975435256958008`, SSIM `0.2611733078956604`, LPIPS `0.5399292707443237`

### Historical GS-W Trackmobile Result

- Source path in `cfg_args`: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\dense`
- Original dataset source recorded by adapter CSV: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Resolution: `1`
- Iterations: `30000`
- SH degree: `3`
- White background: `False`
- Eval: `True`
- `use_colors_precomp=True`
- `use_features_mask=True`
- `use_kmap_pjmap=True`
- `map_num=3`
- `use_lpips_loss=True`
- `position_lr_init=0.00016`
- `position_lr_final=1.6e-07`
- `feature_lr=0.0025`
- `opacity_lr=0.05`
- `scaling_lr=0.005`
- `rotation_lr=0.001`
- `map_generator_lr=0.002`
- `color_net_lr=0.0005`
- `box_coord_lr=1`
- `percent_dense=0.01`
- `lambda_dssim=0.2`
- `densification_interval=100`
- `opacity_reset_interval=3000`
- `densify_from_iter=500`
- `densify_until_iter=15000`
- `densify_grad_threshold=0.0004`
- `opacity_threshold=0.005`
- `random_background=False`
- `lpips_loss_coef=0.005`
- `box_coord_loss_coef=0.001`
- `test_iterations=[30000]`
- `save_iterations=[30000, 30000, 30000]`
- `quiet=True`
- `render_after_train=True`
- `metrics_after_train=True`
- Map generator: `unet`, `resnet18`, `features_dim=48`
- Color net: `naive`, `fin_dim=48`, `pfin_dim=48`, dropout enabled during training.
- Mean PSNR: `12.3854713439941`
- Mean SSIM: `0.42682734131813`
- Mean LPIPS: `0.642460465431213`
- Per-view files in `per_view.json`:
  - `00000.png`: PSNR `11.819730758666992`, SSIM `0.45270779728889465`, LPIPS `0.688982367515564`
  - `00001.png`: PSNR `12.951212882995605`, SSIM `0.40094688534736633`, LPIPS `0.5959386229515076`

## Inferred From Code Defaults

- Random seed is fixed to `0` by `utils/general_utils.py::safe_state`, which calls `random.seed(0)`, `np.random.seed(0)`, and `torch.manual_seed(0)`.
- Current strict decision run uses the same optimizer defaults as historical GS-W because these values match `arguments/__init__.py` and `arguments/args_init.py`.
- Current full-image metric implementation is `metrics.py`, using kornia SSIM, PSNR from `utils/image_utils.py`, and LPIPS AlexNet.

## Unknown Or Not Directly Proven

- The exact shell command used for the historical GS-W run is not preserved as a standalone script. The effective namespace is preserved in `cfg_args` and the log first line.
- CUDA/cuDNN low-level determinism was not explicitly configured in the historical run.
- Historical GS-W used an adapter `dense` path. Current Improved_GS-W can read the clean original `sparse/0` path directly. The strict decision run is therefore labeled a clean strict protocol matched to the historical main parameters, not a byte-for-byte reproduction of the old adapter run.

## Protocol Risk

Historical GS-W rendered held-out views with `legacy_target_rgb`, where the held-out RGB enters the appearance map generator. That result is non-strict and cannot be cited as a strict held-out baseline. It is useful only as an upper-bound-like diagnostic showing the target-conditioned behavior.

The clean strict decision run must train once and evaluate the same 30000-iteration checkpoint under:

- `legacy_target_rgb`
- `strict_intrinsic`
- `strict_nearest_train`

Only `strict_intrinsic` and `strict_nearest_train` are leak-free held-out protocols.
