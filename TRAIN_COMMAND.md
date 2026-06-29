# Trackmobile Strict GS-W 30k Train Command

Repository commit at launch:

`804bd47480470c58cc77e90d83ebd958377dd037`

GPU at launch:

`NVIDIA GeForce RTX 4090, 24564 MiB total, 889 MiB used`

Estimated duration:

Historical Trackmobile GS-W 30k took about 105 minutes. This strict decision run is expected to take roughly 1.5 to 2 hours.

Command:

```powershell
conda run -n 3dgs python train.py `
  --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" `
  --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover `
  --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover" `
  --resolution 1 `
  --iterations 30000 `
  --test_iterations 30000 `
  --save_iterations 30000 `
  --split_mode frozen_manifest `
  --split_file "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" `
  --test_appearance_mode strict_intrinsic `
  --quiet
```

Notes:

- Seed is fixed to `0` by `safe_state()`.
- Train loss only samples train cameras from the frozen manifest.
- Periodic evaluation uses `strict_intrinsic`.
- Final decision uses the fixed iteration-30000 checkpoint; no early stopping or test-metric checkpoint selection is allowed.
- Full-image metrics only; `metrics_half.py` is not used.
