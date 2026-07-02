# EXTERNAL_BASELINE_FAIRNESS_AUDIT

| method | class | strict main competitor | key evidence |
|---|---|---:|---|
| GS-W legacy | B | False | reports/gsw_repeated_30k_baseline/EVAL_STATE_AUDIT.md; legacy_target_rgb render mode; previous audit found test RGB into map_generator and BN eval-state risk. |
| corrected GS-W strict_intrinsic | A | True for designated successful rows only | reports/two_scene_benchmark/GSW_STRICT_LEAKAGE_BN_AUDIT.md; reports/baseline_completion/GSW_12SCENE_CODE_FREEZE_AUDIT.md. |
| Splatfacto-W | D | False | SPLATFACTO_W_PROTOCOL_AUDIT.md; SPLATFACTO_W_HISTORICAL_RUN_AUDIT.csv shows 12 saved configs with eval_right_half=false, use_avg_appearance=true, eval_mode=interval, eval_interval=8, train_split_fraction=0.9, and zero saved render/GT images for unified re-evaluation. |
| Luminance-GS | E | False; current local env/adapter state is not reproducible | reports/baseline_completion/EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md; logs/external_bounded_diagnostics/luminance_import_smoke.log shows pycolmap SceneManager import failure; 3dgs_runs/external_baselines_20260620/summary/failures.csv. |
| WildGaussians | E | False | metrics_all_methods.csv shows PSNR mostly 4.94-8.41 except one small-scene row; WILDGAUSSIANS_DIAGNOSTIC_RESULTS.csv; EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md; wild reader smoke logs show COLMAP adapter reads but numeric renders remain invalid. |

Class meanings: A strict held-out comparable; B target/appearance-conditioned; C transductive or test-time optimized/half-image; D incompatible; E adapter/config invalid; F unresolved.
