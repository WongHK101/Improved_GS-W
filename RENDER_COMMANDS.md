# Trackmobile Strict GS-W Render And Evaluation Commands

All commands use the same checkpoint:

`G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover\ckpts_point_cloud\iteration_30000`

Render held-out views with distinct output tags:

```powershell
conda run -n 3dgs python render.py --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --iteration 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --resolution 1 --skip_train --test_appearance_mode legacy_target_rgb --render_output_tag legacy_target_rgb --quiet
conda run -n 3dgs python render.py --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --iteration 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --resolution 1 --skip_train --test_appearance_mode strict_intrinsic --render_output_tag strict_intrinsic --quiet
conda run -n 3dgs python render.py --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --iteration 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --resolution 1 --skip_train --test_appearance_mode strict_nearest_train --render_output_tag strict_nearest_train --quiet
```

Evaluate full-image metrics:

```powershell
conda run -n 3dgs python metrics.py -m "G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover"
```

Summarize metrics into CSV:

```powershell
conda run -n 3dgs python tools\summarize_appearance_metrics.py --model-path "G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene self_Trackmobile_4650TM_Mobile_Railcar_Mover --iteration 30000 --modes legacy_target_rgb strict_intrinsic strict_nearest_train --summary-csv "G:\WL3DGS\gpt_review_packages\TRACKMOBILE_30K_RESULTS.csv" --per-view-csv "G:\WL3DGS\gpt_review_packages\TRACKMOBILE_30K_PER_VIEW.csv"
```

Train-view feature and intrinsic diagnostic:

```powershell
conda run -n 3dgs python tools\diagnose_intrinsic_features.py --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --iteration 30000 --split_mode frozen_manifest --split_file "G:\wl3dgs\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --resolution 1 --feature-csv "G:\WL3DGS\gpt_review_packages\INTRINSIC_FEATURE_STATS.csv" --train-diagnostics-csv "G:\WL3DGS\gpt_review_packages\TRAIN_VIEW_DIAGNOSTICS.csv" --summary-json "G:\WL3DGS\gpt_review_packages\intrinsic_feature_diagnostic_30k.json"
```
