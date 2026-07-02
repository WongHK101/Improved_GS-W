# EXTERNAL_BASELINE_FAIRNESS_AUDIT

| method | class | strict main competitor | key evidence |
|---|---|---:|---|
| GS-W legacy | B | False | reports/gsw_repeated_30k_baseline/EVAL_STATE_AUDIT.md; legacy_target_rgb render mode; previous audit found test RGB into map_generator and BN eval-state risk. |
| corrected GS-W strict_intrinsic | A | True for designated successful rows only | reports/two_scene_benchmark/GSW_STRICT_LEAKAGE_BN_AUDIT.md; reports/baseline_completion/GSW_12SCENE_CODE_FREEZE_AUDIT.md. |
| Splatfacto-W | C | False | external_baselines/splatfacto-w/splatfactow/splatfactow_config.py:43 sets eval_right_half=True; splatfactow_model.py:1261 applies right-half evaluation; datamanager.py:349 masks eval images during training. |
| Luminance-GS | E | False; current local env/adapter state is not reproducible | reports/baseline_completion/EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md; logs/external_bounded_diagnostics/luminance_import_smoke.log shows pycolmap SceneManager import failure; 3dgs_runs/external_baselines_20260620/summary/failures.csv. |
| WildGaussians | E | False | metrics_all_methods.csv shows PSNR mostly 4.94-8.41 except one small-scene row; WILDGAUSSIANS_DIAGNOSTIC_RESULTS.csv; EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md; wild reader smoke logs show COLMAP adapter reads but numeric renders remain invalid. |

Class meanings: A strict held-out comparable; B target/appearance-conditioned; C transductive or test-time optimized/half-image; D incompatible; E adapter/config invalid; F unresolved.
