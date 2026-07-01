# RUN_COMMANDS

## Pilot Commands

### P-H

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\tools\two_scene_benchmark\gsw_pilot_train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\pilot\H\P-H\self_Steam_Locomotive" --resolution 1 --iterations 5000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 1000 3000 5000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet --trace_output "G:\wl3dgs\Improved_GS-W\reports\two_scene_benchmark\pilot_traces\P-H.csv" --checkpoint_audit_output "G:\wl3dgs\Improved_GS-W\reports\two_scene_benchmark\pilot_traces\P-H_checkpoint.csv"
```

### P-M

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\tools\two_scene_benchmark\gsw_pilot_train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\pilot\M\P-M\web_Terrestrial" --resolution 1 --iterations 5000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 1000 3000 5000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet --trace_output "G:\wl3dgs\Improved_GS-W\reports\two_scene_benchmark\pilot_traces\P-M.csv" --checkpoint_audit_output "G:\wl3dgs\Improved_GS-W\reports\two_scene_benchmark\pilot_traces\P-M_checkpoint.csv"
```

## Screening Commands

### H-O1

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\H\H-O1\self_Steam_Locomotive" --eval --resolution 1 --iterations 30000 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

### H-G1

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\H\H-G1\self_Steam_Locomotive" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 30000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### H-O2

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\H\H-O2\self_Steam_Locomotive" --eval --resolution 1 --iterations 30000 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

### H-G2

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\H\H-G2\self_Steam_Locomotive" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 30000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### H-O3

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\H\H-O3\self_Steam_Locomotive" --eval --resolution 1 --iterations 30000 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

### H-G3

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive" --scene_name self_Steam_Locomotive --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\H\H-G3\self_Steam_Locomotive" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 30000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### M-O1

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\M\M-O1\web_Terrestrial" --eval --resolution 1 --iterations 30000 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

### M-G1

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\M\M-G1\web_Terrestrial" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 30000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### M-O2

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\M\M-O2\web_Terrestrial" --eval --resolution 1 --iterations 30000 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

### M-G2

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\M\M-G2\web_Terrestrial" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 30000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```

### M-O3

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\M\M-O3\web_Terrestrial" --eval --resolution 1 --iterations 30000 --optimizer_type default --depth_l1_weight_init 0 --depth_l1_weight_final 0 --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

### M-G3

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Improved_GS-W\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial" --scene_name web_Terrestrial --model_path "G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701\screening\M\M-G3\web_Terrestrial" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json" --test_appearance_mode strict_intrinsic --test_iterations 1000000 --save_iterations 30000 --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --quiet
```
