# Eval State Audit

- Model path: `G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover`.
- Confirmed bug before fix: legacy eval simulation leaves `map_generator.training=True` while rendering.
- `map_generator` contains BatchNorm buffers; if it remains in train mode, rendering can update `running_mean`, `running_var`, and `num_batches_tracked`.

## State summary

- Fixed eval mode preserved BN buffers for `3/3` mode/order checks.
- Detailed rows are in `PRE_POST_FIX_RENDER_COMPARISON.csv`.

- legacy_target_rgb / forward: legacy_bn_changed=True, fixed_bn_changed=False, fixed_map_generator_training_inside=False.
- strict_intrinsic / forward: legacy_bn_changed=False, fixed_bn_changed=False, fixed_map_generator_training_inside=False.
- strict_nearest_train / forward: legacy_bn_changed=True, fixed_bn_changed=False, fixed_map_generator_training_inside=False.
