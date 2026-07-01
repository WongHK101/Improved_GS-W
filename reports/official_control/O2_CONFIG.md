# O2_CONFIG

- Model path: `G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O2\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Iterations: `30000`
- Return code: `0`
- Duration seconds: `1034.864`
- Peak GPU memory MB: `8123`
- Gaussian count: `1849301`
- Checkpoint size bytes: `458628180`

## Train Command

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\O2\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iterations 30000 --optimizer_type default --test_iterations 1000000 --save_iterations 30000 --disable_viewer --quiet
```

## Protocol Flags

- `--eval` is enabled.
- `--train_test_exp` is not present.
- `--antialiasing` is not present.
- `--depths` is not present.
- `--optimizer_type default` is used.
- `--test_iterations 1000000` prevents train-time test evaluation.
