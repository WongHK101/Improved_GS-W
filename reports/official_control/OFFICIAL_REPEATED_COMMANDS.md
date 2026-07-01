# OFFICIAL_REPEATED_COMMANDS

All commands use the clean official 3DGS worktree and the same Trackmobile LLFF hold-8 split.

## O1 Train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O1\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iterations 30000 --optimizer_type default --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

## O1 Render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O1\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iteration 30000 --quiet --skip_train
```

## O2 Train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O2\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iterations 30000 --optimizer_type default --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

## O2 Render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O2\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iteration 30000 --quiet --skip_train
```

## O3 Train

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O3\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iterations 30000 --optimizer_type default --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

## O3 Render

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O3\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iteration 30000 --quiet --skip_train
```
