# Short Run Commands

## A1_clean_direct_frozen

```powershell
conda run -n 3dgs --no-capture-output python train.py --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --resolution 1 --iterations 1000 --test_iterations 1000000 --save_iterations 1000000 --disable_save_iterations --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --trace_training_state --quiet --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\A1_clean_direct_frozen_seed0_iter1000" --split_mode frozen_manifest --split_file "G:\WL3DGS\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --trace_output "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\A1_clean_direct_frozen_seed0_iter1000\trace.csv"
```

## A2_clean_direct_frozen_repeat

```powershell
conda run -n 3dgs --no-capture-output python train.py --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --resolution 1 --iterations 1000 --test_iterations 1000000 --save_iterations 1000000 --disable_save_iterations --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --trace_training_state --quiet --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --model_path "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\A2_clean_direct_frozen_seed0_iter1000" --split_mode frozen_manifest --split_file "G:\WL3DGS\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --trace_output "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\A2_clean_direct_frozen_seed0_iter1000\trace.csv"
```

## B_clean_adapter_frozen

```powershell
conda run -n 3dgs --no-capture-output python train.py --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --resolution 1 --iterations 1000 --test_iterations 1000000 --save_iterations 1000000 --disable_save_iterations --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --trace_training_state --quiet --source_path "G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\dense" --model_path "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\B_clean_adapter_frozen_seed0_iter1000" --split_mode frozen_manifest --split_file "G:\WL3DGS\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --trace_output "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\B_clean_adapter_frozen_seed0_iter1000\trace.csv"
```

## C_clean_adapter_legacy_tsv

```powershell
conda run -n 3dgs --no-capture-output python train.py --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --resolution 1 --iterations 1000 --test_iterations 1000000 --save_iterations 1000000 --disable_save_iterations --disable_render_after_train --disable_metrics_after_train --disable_train_temp_images --trace_training_state --quiet --source_path "G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\dense" --model_path "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\C_clean_adapter_legacy_tsv_seed0_iter1000" --eval --split_mode legacy --legacy_tsv_uid_source extrinsic --trace_output "G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630\C_clean_adapter_legacy_tsv_seed0_iter1000\trace.csv"
```

## D_historical_worktree

Not launched by this helper. If run directly in the historical dirty worktree, the command must be separately instrumented there; this audit uses clean-code historical-compatible C as the safe proxy unless D is explicitly approved.
