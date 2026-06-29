# Trackmobile Strict GS-W 30k Decision Report

## Scope

This run evaluates whether strict GS-W is a viable baseline/method base on Trackmobile without held-out RGB leakage. No new algorithm was added and no test-metric tuning was performed.

## Repository State

- Repository: `G:\wl3dgs\Improved_GS-W`
- Remote: `https://github.com/WongHK101/Improved_GS-W.git`
- Launch commit: `4c71cf0780f7fe77c75c1b79f37ce14502bb533a`
- Key prior commits:
  - `d6d4661` all held-out appearance leakage test
  - `804bd47` diagnostic evaluation tools
  - `4c71cf0` strict Trackmobile protocol docs

## Data And Split

- Source data: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Split manifest: `G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json`
- Registered images: 15
- Train images: 13
- Test images: 2 (`0001.jpg`, `0009.jpg`)
- Protocol: frozen LLFF hold-8 full-image evaluation

## Training

- Output path: `G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Command: documented in `TRAIN_COMMAND.md`
- Resolution: `1`
- Iterations: `30000`
- Seed: `0` through `utils/general_utils.py::safe_state`
- Periodic evaluation mode: `strict_intrinsic`
- Final checkpoint: `ckpts_point_cloud\iteration_30000`
- Training elapsed: `5682` seconds
- Final Gaussian count in progress bar/log: `140945`
- Checkpoint files:
  - `point_cloud.ply`: 34,955,900 bytes
  - `map_generator.pth`: 63,263,081 bytes
  - `color_net.pth`: 222,807 bytes
  - `other_atrributes_dict.pth`: 3,383,839 bytes
- OOM/NaN/abnormal exit: not observed
- Peak GPU memory: not captured by the training script; only launch GPU state was recorded.

## Environment

- GPU: NVIDIA GeForce RTX 4090
- Python: 3.10.18
- PyTorch: 2.0.1
- Torch CUDA: 11.8
- torchvision: 0.15.2
- kornia: 0.8.2
- LPIPS package version: unknown

## Held-Out Leakage Verification

All held-out views were tested with RGB perturbations (`zero`, fixed `noise`, `channel_swap`) on the 10-iteration smoke checkpoint.

- `strict_intrinsic`: max/mean difference was exactly `0.0` for both held-out views and all perturbations.
- `strict_nearest_train`: max/mean difference was exactly `0.0` for both held-out views and all perturbations.
- `legacy_target_rgb`: output changed for both held-out views, confirming target RGB conditioning.

Nearest-train mapping:

- `0001.jpg -> 0002.jpg`, camera-center distance `1.5382812023`
- `0009.jpg -> 0010.jpg`, camera-center distance `1.2534600496`

## 30k Appearance Mode Results

All three modes were rendered from the same iteration-30000 checkpoint.

| Mode | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| legacy_target_rgb | 11.150373 | 0.393910 | 0.724205 |
| strict_intrinsic | 11.156013 | 0.399699 | 0.722331 |
| strict_nearest_train | 11.038421 | 0.395382 | 0.719736 |

Differences:

- legacy_target_rgb - strict_intrinsic: PSNR `-0.005639`, SSIM `-0.005788`, LPIPS `+0.001875` (worse)
- legacy_target_rgb - strict_nearest_train: PSNR `+0.111953`, SSIM `-0.001471`, LPIPS `+0.004470` (worse LPIPS)
- strict_nearest_train - strict_intrinsic: PSNR `-0.117592`, SSIM `-0.004317`, LPIPS `-0.002595`

Strict-nearest is not stably better than strict-intrinsic. It slightly improves LPIPS but lowers PSNR and SSIM.

## Train-View Diagnostic

Full 13 train views were evaluated as:

- normal train-conditioned render
- intrinsic zero-feature render

Mean results:

| Mode | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| normal train-conditioned | 17.155774 | 0.615034 | 0.425809 |
| intrinsic zero-feature | 16.486568 | 0.575485 | 0.453413 |

Normal train-conditioned is better by about `0.669206` dB PSNR, `0.039548` SSIM, and `0.027605` LPIPS. The dynamic branch contributes, but intrinsic rendering does not catastrophically collapse.

## Feature Distribution Diagnostic

- Mean point-feature L2 norm across train views: `1.731735`
- Mean point-feature L2 std across train views: `1.881375`
- Zero-feature L2 norm: `0.0`
- Zero-feature z-score against train-view L2 mean: `-0.920462`
- Near-zero element ratio: `0.144033`
- Near-zero vector ratio: `0.0`

Zero features are lower than normal train-conditioned features, but not an extreme distribution outlier under this aggregate statistic.

## Comparison To Historical Baselines

Historical official 3DGS Trackmobile:

- PSNR `10.983210`
- SSIM `0.240177`
- LPIPS `0.575440`

Historical GS-W Trackmobile:

- PSNR `12.385471`
- SSIM `0.426827`
- LPIPS `0.642460`
- This run used legacy target-RGB conditioning and is non-strict.

Clean strict GS-W versus official 3DGS:

- Best strict PSNR is `11.156013`, about `+0.172803` dB over official 3DGS.
- Best strict SSIM is `0.399699`, substantially above official 3DGS.
- Best strict LPIPS is `0.719736`, worse than official 3DGS by about `0.144295`.

Because the protocols differ in implementation family and historical GS-W used an adapter path plus target-conditioned evaluation, direct deltas must be treated carefully.

## Decision

Strict GS-W is not a strong default base for the next paper method on this evidence alone. It is leak-free and does not collapse, but the benefit over clean official 3DGS is mixed: PSNR and SSIM improve, LPIPS worsens sharply, and nearest-train appearance is not consistently superior.

Recommendation: do not continue by simply using vanilla GS-W strict modes as the main method base. If GS-W is used, treat it as a diagnostic or component source. A stronger next route should explicitly fix canonical appearance supervision or use a better constrained appearance decomposition, then validate against official 3DGS with matched protocol.
