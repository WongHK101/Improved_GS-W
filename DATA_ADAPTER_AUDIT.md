# DATA_ADAPTER_AUDIT

Date: 2026-06-29

## Purpose

This commit adds explicit COLMAP sparse model path support for the clean Improved_GS-W baseline. It is a data-layout compatibility change only; it does not change the training loss, renderer math, appearance model, split protocol, or evaluation metrics.

## Sparse Model Resolution

`scene.dataset_readers.resolve_colmap_sparse_model_path()` selects the first valid COLMAP sparse model using this priority:

1. Explicit `--sparse_subdir`, interpreted as absolute when absolute or relative to `--source_path` otherwise.
2. `<source_path>/sparse/0`
3. `<source_path>/sparse`

A valid sparse model must contain either both `images.bin` and `cameras.bin`, or both `images.txt` and `cameras.txt`. If multiple valid candidates exist, the selected path and all candidates are printed.

## Reader Alignment Test

Command:

```powershell
conda run -n 3dgs python tests/test_colmap_reader_smoke.py --scene "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --output-json "G:\WL3DGS\gpt_review_packages\colmap_reader_trackmobile.json"
```

Result:

- Scene: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Selected sparse path: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover\sparse\0`
- Registered cameras: 15
- Initial COLMAP point count: 1920
- Image names: `0001.jpg` through `0015.jpg`
- Compared against official 3DGS reader at `G:\wl3dgs\3dgs_original`
- Camera count, point count, image names, image sizes, uid, FoV, rotation, and translation matched official 3DGS.
- Full JSON report: `G:\WL3DGS\gpt_review_packages\colmap_reader_trackmobile.json`

## Training Smoke Status

Command:

```powershell
conda run -n 3dgs python train.py --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_clean_smoke_20260629\self_Trackmobile_sparse10" --resolution 4 --iterations 10 --test_iterations 10 --save_iterations 10
```

Result:

- The scene loaded successfully through the new sparse model path resolver.
- Training initialized 1920 points and loaded all 15 training cameras.
- The run stopped before the first optimization iteration due to a rasterizer API mismatch:

```text
TypeError: GaussianRasterizationSettings.__new__() missing 1 required positional argument: 'antialiasing'
```

This failure is independent of the COLMAP sparse path change. It indicates that the active `diff_gaussian_rasterization` package in the Python environment exposes a newer `GaussianRasterizationSettings` signature than the GS-W renderer call currently uses. This should be fixed in a separate compatibility commit.

## Source Changes

- `arguments/__init__.py`
  - Adds `--sparse_subdir`.
- `scene/__init__.py`
  - Passes `args.sparse_subdir` into the COLMAP loader.
- `scene/dataset_readers.py`
  - Adds sparse model discovery helpers.
  - Reads cameras and points from the resolved sparse path.
  - Uses `extr.camera_id` for COLMAP intrinsic lookup.
  - Preserves full COLMAP image names such as `0001.jpg`.
- `tests/test_colmap_reader_smoke.py`
  - Adds a reader parity smoke test against official 3DGS.

## Risk Notes

- Preserving the full image filename changes the old GS-W `image_name` convention from stem-only to official 3DGS-style filename. This is intentional because the frozen split manifest will use exact image filenames.
- Existing legacy GS-W TSV split logic must be checked in the split-manifest commit to ensure this image-name convention does not silently break old TSV files.
- The rasterizer `antialiasing` compatibility fix must be reviewed separately and should not be attributed to sparse-path support.
