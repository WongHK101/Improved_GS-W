# STRICT_LEAKAGE_TEST

Date: 2026-06-30

## Test

Script:

```text
tests/test_strict_appearance_invariance.py
```

Command:

```powershell
conda run -n 3dgs python tests/test_strict_appearance_invariance.py --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_tests\trackmobile_split_10iter" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --iteration 10 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --resolution 4 --output-json "G:\WL3DGS\gpt_review_packages\strict_leakage_invariance_trackmobile.json" --mapping-csv "G:\WL3DGS\gpt_review_packages\TEST_APPEARANCE_MAPPING.csv"
```

The test renders the same held-out Trackmobile test camera with:

- original test RGB;
- zero test RGB;
- fixed random noise test RGB;
- RGB channel-swapped test RGB.

It compares the final rendered RGB image under:

- `strict_intrinsic`;
- `strict_nearest_train`;
- `legacy_target_rgb`.

## Expected Behavior

Strict modes:

- Rendered outputs must not change when held-out test RGB is modified.
- Max absolute difference threshold: `1e-7`.

Legacy mode:

- Rendered output should change when held-out test RGB is modified.
- This proves the test actually exercises the target-conditioned appearance path.

## Results

Test camera:

```text
0001.jpg
```

Strict max absolute differences:

```text
strict_intrinsic:
  zero         0.0
  noise        0.0
  channel_swap 0.0

strict_nearest_train:
  zero         0.0
  noise        0.0
  channel_swap 0.0
```

Legacy max absolute differences:

```text
legacy_target_rgb:
  zero         0.20159250497817993
  noise        0.042890965938568115
  channel_swap 0.018136441707611084
```

Nearest-train appearance mapping:

```text
0001.jpg -> 0002.jpg, distance 1.5382812023162842
0009.jpg -> 0010.jpg, distance 1.2534600496292114
```

JSON evidence:

```text
G:\WL3DGS\gpt_review_packages\strict_leakage_invariance_trackmobile.json
```

CSV evidence:

```text
G:\WL3DGS\gpt_review_packages\TEST_APPEARANCE_MAPPING.csv
```

## Conclusion

The strict modes are invariant to held-out test RGB perturbations at the tested tolerance. The legacy mode changes under the same perturbations, confirming that upstream GS-W behavior is target-conditioned and non-strict.
