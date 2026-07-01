# TWO_SCENE_SMOKE_AUDIT

- Overall split preflight: `PASS`
- Overall smoke: `PASS`
- Official vs GS-W GT decoded-pixel checksums: `PASS`
- Run root: `G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean`

## Smoke Summary

| role | scene | method | iter | train min | render min | renders/gt | PSNR | SSIM | LPIPS | finite | black | peak MB |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| H | self_Steam_Locomotive | official_3dgs | 10 | 0.168 | 0.169 | 10/10 | 12.263341 | 0.484009 | 0.949793 | 10 | 0 | 937 |
| H | self_Steam_Locomotive | official_3dgs | 300 | 0.335 | 0.169 | 10/10 | 17.148606 | 0.581488 | 0.829831 | 10 | 0 | 3821 |
| H | self_Steam_Locomotive | gsw_strict_intrinsic | 10 | 0.168 | 0.169 | 10/10 | 9.001109 | 0.371711 | 0.975618 | 10 | 0 | 937 |
| H | self_Steam_Locomotive | gsw_strict_intrinsic | 300 | 1.172 | 0.253 | 10/10 | 16.388797 | 0.573932 | 0.868835 | 10 | 0 | 11429 |
| M | web_Terrestrial | official_3dgs | 10 | 0.168 | 0.505 | 24/24 | 14.560936 | 0.609891 | 0.910161 | 24 | 0 | 937 |
| M | web_Terrestrial | official_3dgs | 300 | 0.335 | 0.505 | 24/24 | 18.080584 | 0.671838 | 0.827551 | 24 | 0 | 7259 |
| M | web_Terrestrial | gsw_strict_intrinsic | 10 | 0.335 | 0.421 | 24/24 | 8.633847 | 0.293644 | 1.035093 | 24 | 0 | 18323 |
| M | web_Terrestrial | gsw_strict_intrinsic | 300 | 1.34 | 0.505 | 24/24 | 16.056575 | 0.643545 | 0.897915 | 24 | 0 | 18325 |

## Leakage Controls

- Scene selection used train-only lighting statistics; no test RGB, historical metrics, method deltas, or qualitative test renders were used.
- GS-W smoke uses `--test_appearance_mode strict_intrinsic` for both training-time test hooks and explicit render.
- GS-W renders use `--render_output_tag strict_intrinsic`; output directories are `ours_<iter>_strict_intrinsic`.
- Official smoke uses clean official 3DGS `--eval`; split equivalence is checked against the frozen manifests.
- `--train_test_exp` is absent, half-image metrics are not run, and evaluation is full-image RGB.
- Official smoke sets `--optimizer_type default`, `--depth_l1_weight_init 0`, and `--depth_l1_weight_final 0`; no antialiasing flag is used.

## Commands

### H official_3dgs iter 10 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\official_3dgs\iter_10\self_Steam_Locomotive" --eval --resolution 1 --iterations 10 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 10 --disable_viewer --quiet
```

### H official_3dgs iter 10 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\official_3dgs\iter_10\self_Steam_Locomotive" --eval --resolution 1 --iteration 10 --skip_train --quiet
```

### H official_3dgs iter 300 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\official_3dgs\iter_300\self_Steam_Locomotive" --eval --resolution 1 --iterations 300 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 300 --disable_viewer --quiet
```

### H official_3dgs iter 300 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\official_3dgs\iter_300\self_Steam_Locomotive" --eval --resolution 1 --iteration 300 --skip_train --quiet
```

### H gsw_strict_intrinsic iter 10 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\gsw_strict_intrinsic\iter_10\self_Steam_Locomotive" --resolution 1 --iterations 10 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 10 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### H gsw_strict_intrinsic iter 10 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\gsw_strict_intrinsic\iter_10\self_Steam_Locomotive" --resolution 1 --iteration 10 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --render_output_tag strict_intrinsic --skip_train --quiet
```

### H gsw_strict_intrinsic iter 300 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\gsw_strict_intrinsic\iter_300\self_Steam_Locomotive" --resolution 1 --iterations 300 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 300 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### H gsw_strict_intrinsic iter 300 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\H\gsw_strict_intrinsic\iter_300\self_Steam_Locomotive" --resolution 1 --iteration 300 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --render_output_tag strict_intrinsic --skip_train --quiet
```

### M official_3dgs iter 10 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\official_3dgs\iter_10\web_Terrestrial" --eval --resolution 1 --iterations 10 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 10 --disable_viewer --quiet
```

### M official_3dgs iter 10 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\official_3dgs\iter_10\web_Terrestrial" --eval --resolution 1 --iteration 10 --skip_train --quiet
```

### M official_3dgs iter 300 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\official_3dgs\iter_300\web_Terrestrial" --eval --resolution 1 --iterations 300 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 300 --disable_viewer --quiet
```

### M official_3dgs iter 300 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\official_3dgs\iter_300\web_Terrestrial" --eval --resolution 1 --iteration 300 --skip_train --quiet
```

### M gsw_strict_intrinsic iter 10 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\gsw_strict_intrinsic\iter_10\web_Terrestrial" --resolution 1 --iterations 10 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 10 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### M gsw_strict_intrinsic iter 10 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\gsw_strict_intrinsic\iter_10\web_Terrestrial" --resolution 1 --iteration 10 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --render_output_tag strict_intrinsic --skip_train --quiet
```

### M gsw_strict_intrinsic iter 300 train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\gsw_strict_intrinsic\iter_300\web_Terrestrial" --resolution 1 --iterations 300 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 300 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### M gsw_strict_intrinsic iter 300 render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean\M\gsw_strict_intrinsic\iter_300\web_Terrestrial" --resolution 1 --iteration 300 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --render_output_tag strict_intrinsic --skip_train --quiet
```

