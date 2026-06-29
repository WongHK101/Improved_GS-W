# STRICT_APPEARANCE_IMPLEMENTATION

Date: 2026-06-30

## Goal

GS-W upstream conditions test rendering on the target image through `map_generator(test_camera.original_image)`. That is valid for the legacy paper/demo protocol but leaks held-out test RGB under strict novel-view synthesis.

This implementation keeps legacy behavior available and adds two explicit strict-safe test appearance modes.

## Command-Line Interface

```text
--test_appearance_mode legacy_target_rgb
--test_appearance_mode strict_intrinsic
--test_appearance_mode strict_nearest_train
```

Default:

```text
legacy_target_rgb
```

The default preserves upstream compatibility. Strict experiments must explicitly pass `strict_intrinsic` or `strict_nearest_train`.

## Modes

### legacy_target_rgb

Behavior:

- Preserves upstream GS-W behavior.
- Test `Camera.original_image` may enter `map_generator`.
- Runtime prints:

```text
NOT STRICT HELD-OUT: TEST RGB CONDITIONS APPEARANCE
```

Interpretation:

- Non-strict.
- Target-conditioned upper-bound / upstream reproduction mode.
- Must not be ranked as a strict held-out NVS baseline.

### strict_intrinsic

Behavior:

- Does not call `map_generator(test_camera.original_image)`.
- Calls `GaussianModel.forward_intrinsic()`.
- Uses zero dynamic point features and the intrinsic/canonical Gaussian feature branch.
- Does not perform test-time optimization.
- Does not use test RGB to choose parameters.
- Skips GS-W test multiview and intrinsic/dynamic interpolation demo branches because those branches are target-conditioned diagnostics.

Guard:

- Test cameras rendered in strict mode are marked with `forbid_appearance_input=True`.
- If such a camera reaches `GaussianModel.forward()` and therefore would enter `map_generator`, execution raises:

```text
STRICT HELD-OUT VIOLATION: test camera image was passed to map_generator.
```

### strict_nearest_train

Behavior:

- Does not use test RGB.
- Selects an appearance source only from train cameras.
- Selection rule: nearest camera center Euclidean distance; tie-break by training image name.
- Runs `pc.forward(train_camera, store_cache=True)` on the selected train source.
- Runs `pc.forward_cache(test_camera)` for the test pose/view direction.
- Writes mapping CSV:

```text
<model_path>/test/ours_<iteration>/appearance_mapping.csv
```

Trackmobile mapping at iteration 10:

```text
0001.jpg -> 0002.jpg, distance 1.5382812023
0009.jpg -> 0010.jpg, distance 1.2534600496
```

## Code Touch Points

- `arguments/__init__.py`
  - Adds `test_appearance_mode`.
  - Fixes `get_combined_args()` so explicit command-line arguments override checkpoint `cfg_args` during render.

- `scene/cameras.py`
  - Adds `split_role`.

- `scene/__init__.py`
  - Marks loaded train/test cameras with `split_role`.

- `scene/gaussian_model.py`
  - Adds `assert_can_use_camera_appearance()`.
  - Adds `forward_with_point_features()`.
  - Adds `forward_intrinsic()`.
  - Adds `forward_interpolate()`.

- `gaussian_renderer/__init__.py`
  - Adds `appearance_mode` and `appearance_source_camera` to `render()`.
  - Routes legacy, strict intrinsic, and strict nearest-train modes.

- `render.py`
  - Adds nearest-train source selection and mapping CSV.
  - Applies test appearance modes to test rendering.
  - Disables target-conditioned demo branches in strict modes.

- `train.py`
  - Applies test appearance mode during `training_report()`.
  - Applies test appearance mode during post-training test rendering.

## Smoke Evidence

Strict intrinsic 10-iteration Trackmobile command:

```powershell
conda run -n 3dgs python train.py --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_tests\trackmobile_strict_intrinsic_10iter" --resolution 4 --iterations 10 --test_iterations 10 --save_iterations 10 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --test_appearance_mode strict_intrinsic
```

Full-image metrics:

```text
SSIM  = 0.1390864849
PSNR  = 6.8437929153
LPIPS = 0.9891056418
```

Log:

```text
G:\WL3DGS\gpt_review_packages\trackmobile_strict_intrinsic_10iter.log
```

## Status

Implemented and smoke-tested. No new algorithmic modules such as reliability, tone mapping, canonical anchoring, embedding regularization, or densification changes were added.
