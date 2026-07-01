# OFFICIAL_SMOKE_RESULTS

| label | stage | iter | return | renders | gt | duration min | peak MB |
|---|---|---:|---:|---:|---:|---:|---:|
| smoke_10 | train | 10 | 0 |  |  | 0.168 | 889 |
| smoke_10 | render | 10 | 0 | 2 | 2 | 0.085 | 873 |
| smoke_300 | train | 300 | 0 |  |  | 0.168 | 873 |
| smoke_300 | render | 300 | 0 | 2 | 2 | 0.084 | 873 |

## Commands

- `smoke_10 train`: `"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\smoke_10\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iterations 10 --optimizer_type default --test_iterations 1000000 --save_iterations 10 --disable_viewer --quiet`
- `smoke_10 render`: `"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\smoke_10\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iteration 10 --quiet --skip_train`
- `smoke_300 train`: `"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\smoke_300\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iterations 300 --optimizer_type default --test_iterations 1000000 --save_iterations 300 --disable_viewer --quiet`
- `smoke_300 render`: `"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\render.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\smoke_300\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iteration 300 --quiet --skip_train`

No official source files were patched for these smoke tests.
