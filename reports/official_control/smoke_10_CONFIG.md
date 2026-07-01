# smoke_10_CONFIG

- Model path: `G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\smoke_10\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Iterations: `10`
- Return code: `0`
- Duration seconds: `10.081`
- Peak GPU memory MB: `889`
- Gaussian count: `1920`
- Checkpoint size bytes: `477689`

## Train Command

```text
"D:\anaconda\envs\3dgs\python.exe" "G:\wl3dgs\Official_3DGS_Strict_Control\train.py" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701\smoke_10\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --eval --resolution 1 --iterations 10 --optimizer_type default --test_iterations 1000000 --save_iterations 10 --disable_viewer --quiet
```

## Protocol Flags

- `--eval` is enabled.
- `--train_test_exp` is not present.
- `--antialiasing` is not present.
- `--depths` is not present.
- `--optimizer_type default` is used.
- `--test_iterations 1000000` prevents train-time test evaluation.
