# Eval State Fix Audit

- `GaussianModel.set_eval(True)` now calls both `color_net.eval()` and `map_generator.eval()`.
- `GaussianModel.set_eval(False)` restores the module training states captured before entering eval mode.
- Existing behavior that disables `features_mask` and `color_net.use_drop_out` during eval is preserved, but the original values are restored afterwards.
- The implementation uses a stack so nested eval calls restore in order.

## Render invariance checks

- pre_post_legacy_bug_vs_fixed / legacy_target_rgb: max_abs_pixel_diff=0.8011182546615601, mean_abs_pixel_diff=0.016386751318350434.
- fixed_forward_vs_reverse / legacy_target_rgb: max_abs_pixel_diff=0.0, mean_abs_pixel_diff=0.0.
- fixed_repeat_forward / legacy_target_rgb: max_abs_pixel_diff=0.0, mean_abs_pixel_diff=0.0.
- pre_post_legacy_bug_vs_fixed / strict_intrinsic: max_abs_pixel_diff=0.0, mean_abs_pixel_diff=0.0.
- fixed_forward_vs_reverse / strict_intrinsic: max_abs_pixel_diff=0.0, mean_abs_pixel_diff=0.0.
- fixed_repeat_forward / strict_intrinsic: max_abs_pixel_diff=0.0, mean_abs_pixel_diff=0.0.
- pre_post_legacy_bug_vs_fixed / strict_nearest_train: max_abs_pixel_diff=0.6120296716690063, mean_abs_pixel_diff=0.009112979052588344.
- fixed_forward_vs_reverse / strict_nearest_train: max_abs_pixel_diff=0.0, mean_abs_pixel_diff=0.0.
- fixed_repeat_forward / strict_nearest_train: max_abs_pixel_diff=0.0, mean_abs_pixel_diff=0.0.

## Pre/post metric deltas

- legacy_target_rgb / 0001.jpg: delta_psnr=-0.09621267682591572, delta_ssim=-0.0013473927974700928, delta_lpips=0.0033693313598632812, max_abs_pixel_diff=0.8011182546615601.
- legacy_target_rgb / 0009.jpg: delta_psnr=-0.0066900710094852656, delta_ssim=0.002447664737701416, delta_lpips=-0.0011255741119384766, max_abs_pixel_diff=0.47780466079711914.
- strict_intrinsic / 0001.jpg: delta_psnr=0.0, delta_ssim=0.0, delta_lpips=0.0, max_abs_pixel_diff=0.0.
- strict_intrinsic / 0009.jpg: delta_psnr=0.0, delta_ssim=0.0, delta_lpips=0.0, max_abs_pixel_diff=0.0.
- strict_nearest_train / 0001.jpg: delta_psnr=0.028702210643054116, delta_ssim=0.0019202232360839844, delta_lpips=0.002093791961669922, max_abs_pixel_diff=0.6120296716690063.
- strict_nearest_train / 0009.jpg: delta_psnr=-0.0615784181677661, delta_ssim=0.00048673152923583984, delta_lpips=-0.0015428662300109863, max_abs_pixel_diff=0.43544378876686096.
