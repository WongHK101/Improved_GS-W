# SPLIT_IMPLEMENTATION

Date: 2026-06-30

## Goal

This change adds explicit frozen split-manifest support so GS-W can reuse the same train/test image sets as the official 3DGS LLFF hold-8 experiments. It does not let GS-W recompute hold-8 at runtime when `--split_mode frozen_manifest` is selected.

## Command-Line Interface

New loading parameters:

```text
--split_mode legacy
--split_mode frozen_manifest
--split_file <manifest-path>
```

Behavior:

- `legacy`: preserves upstream GS-W behavior. If `--eval` is used, the original TSV path is used. If `--eval` is not used, all registered cameras are training cameras and test is empty.
- `frozen_manifest`: requires `--split_file`. Missing files, missing registered images, extra images, duplicate names, train/test overlap, empty train set, and empty test set are hard errors.

The manifest used by a run is copied to:

```text
<model_path>/split_manifest.json
```

The actual resolved split summary is written to:

```text
<model_path>/split_used.json
```

## Manifest Format

Trackmobile manifest:

```text
G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json
```

Essential fields:

```json
{
  "scene": "...",
  "protocol": "llff_hold_8",
  "ordering_rule": "...",
  "train_images": ["..."],
  "test_images": ["..."]
}
```

Image filenames are the sole split identifiers. Windows paths in a manifest are normalized to basenames before matching, so `C:\tmp\0001.jpg` and `0001.jpg` match the same registered image. Matching is case-sensitive at the normalized basename level to avoid silently merging distinct filenames.

## Trackmobile Split Source

Frozen split source:

```text
G:\wl3dgs\splits\max1600_llffhold8_v1\self_Trackmobile_4650TM_Mobile_Railcar_Mover
```

Source metadata:

```text
G:\wl3dgs\splits\max1600_llffhold8_v1\self_Trackmobile_4650TM_Mobile_Railcar_Mover\split_meta.json
```

Recorded rule:

```text
sorted registered COLMAP image names; test if index % 8 == 0
```

This matches official 3DGS COLMAP reader behavior, where `readColmapSceneInfo()` sorts `cam_infos` by `image_name` before applying LLFF hold-8.

## Trackmobile Frozen Names

Registered count: 15

Train count: 13

Train images:

```text
0002.jpg
0003.jpg
0004.jpg
0005.jpg
0006.jpg
0007.jpg
0008.jpg
0010.jpg
0011.jpg
0012.jpg
0013.jpg
0014.jpg
0015.jpg
```

Test count: 2

Test images:

```text
0001.jpg
0009.jpg
```

Legacy source hashes from `split_meta.json`:

```text
all_sha256   = f3962955c539c64c6a68b3c17133824e147e9ed069e57df2f9e2be5a6e5ea59e
train_sha256 = 9ad80ed0ae4c100bb4e261e7989575110e7eeb0d043222b17860fd8e8f4e1b3c
test_sha256  = b48466af92f75b2a88c79d296c60786e151baa3511dfee97b0637ee2c0397962
```

## Tests

Regression command:

```powershell
conda run -n 3dgs python tests/test_frozen_split_manifest.py --scene "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --manifest "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --output-json "G:\WL3DGS\gpt_review_packages\trackmobile_split_test.json"
```

The test covers:

- normal manifest loading;
- train/test overlap;
- duplicate image names;
- missing registered images;
- extra images not registered by COLMAP;
- empty train set;
- empty test set;
- Windows path normalization;
- reader-order invariance.

## Protocol Note

This split change only establishes full-image train/test separation. Until strict-safe appearance modes are implemented, GS-W test rendering may still be target-conditioned by the upstream appearance path and must be reported as non-strict.
