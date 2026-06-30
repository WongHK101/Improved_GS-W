# R1_CONFIG

## cfg_args

```text
Namespace(device='cuda:0', sh_degree=3, source_path='G:\\wl3dgs\\3dgs_undistorted\\max1600\\self_Trackmobile_4650TM_Mobile_Railcar_Mover', model_path='G:\\wl3dgs\\3dgs_runs\\gsw_strict_baseline_v2_repeated_30k_20260630\\R1\\self_Trackmobile_4650TM_Mobile_Railcar_Mover', images='images', sparse_subdir='', split_mode='frozen_manifest', split_file='G:\\WL3DGS\\Improved_GS-W\\splits\\TRACKMOBILE_SPLIT.json', legacy_tsv_uid_source='intrinsic', test_appearance_mode='strict_intrinsic', resolution=1, white_background=False, data_device='cuda', eval=False, scene_name='self_Trackmobile_4650TM_Mobile_Railcar_Mover', use_colors_precomp=True, use_decode_with_pos=False, use_indep_mask_branch=False, use_features_mask=True, features_mask_loss_coef=0.15, features_mask_iters=2500, use_okmap=False, use_kmap_pjmap=True, map_num=3, use_wo_adative=0, use_xw_init_box_coord=True, use_color_net=True, use_scaling_loss=False, use_lpips_loss=True, use_box_coord_loss=True, coord_scale=1, iterations=30000, position_lr_init=0.00016, position_lr_final=1.6e-07, position_lr_delay_mult=0.01, position_lr_max_steps=30000, feature_lr=0.0025, opacity_lr=0.05, scaling_lr=0.005, rotation_lr=0.001, map_generator_lr=0.002, color_net_lr=0.0005, box_coord_lr=1, warm_up_iter=0, percent_dense=0.01, lambda_dssim=0.2, densification_interval=100, opacity_reset_interval=3000, densify_from_iter=500, densify_until_iter=15000, densify_grad_threshold=0.0004, opacity_threshold=0.005, random_background=False, scaling_loss_coef=0.005, lpips_loss_coef=0.005, box_coord_loss_coef=0.001, convert_SHs_python=False, compute_cov3D_python=False, debug=False, ip='127.0.0.1', port=6009, debug_from=-1, detect_anomaly=False, test_iterations=[1000000], save_iterations=[30000, 30000, 30000], quiet=True, render_after_train=False, metrics_after_train=False, eval_half_after_train=False, data_perturb=[], trace_training_state=False, trace_output='', disable_render_after_train=True, disable_metrics_after_train=True, disable_save_iterations=False, disable_train_temp_images=True, map_generator_type='unet', feature_maps_dim=16, feature_maps_combine='cat', use_indep_box_coord=True, map_generator_params={'features_dim': 48, 'backbone': 'resnet18', 'use_features_mask': True, 'use_independent_mask_branch': False}, features_dim=48, color_net_type='naive', features_weight_loss_coef=0.01, color_net_params={'fin_dim': 48, 'pin_dim': 3, 'view_dim': 3, 'pfin_dim': 48, 'en_dims': [128, 96, 64], 'de_dims': [48, 48], 'multires': [10, 0], 'pre_compc': True, 'cde_dims': [48], 'use_pencoding': [True, False], 'weight_norm': False, 'weight_xavier': True, 'use_drop_out': True, 'use_decode_with_pos': False})
```

## split_used.json

```json
{
  "split_mode": "frozen_manifest",
  "manifest_path": "G:\\WL3DGS\\Improved_GS-W\\splits\\TRACKMOBILE_SPLIT.json",
  "manifest_sha256": "c65cc61677ffb46cd329d57fc72d30ed40d42e0f9a47d75c5a0931950c9f2cd7",
  "registered_count": 15,
  "train_count": 13,
  "test_count": 2,
  "registered_sha256": "02337975c4d32a4b79cbee6de7c72ec00eed13588d30655c31e6f9874a6d198e",
  "train_sha256": "0f67eb67436be3cb65ad9fa792700fab42a79f627e32960407c06babfb16edd2",
  "test_sha256": "a3bdc279967d8200ccbccb00cc2eac803cd3a9b6db5a61902f85edd5a679bdd3",
  "train_images": [
    "0002.jpg",
    "0003.jpg",
    "0004.jpg",
    "0005.jpg",
    "0006.jpg",
    "0007.jpg",
    "0008.jpg",
    "0010.jpg",
    "0011.jpg",
    "0012.jpg",
    "0013.jpg",
    "0014.jpg",
    "0015.jpg"
  ],
  "test_images": [
    "0001.jpg",
    "0009.jpg"
  ],
  "protocol": "llff_hold_8",
  "ordering_rule": "sort registered COLMAP image names lexicographically; assign test if zero-based sorted index % 8 == 0"
}
```

## runtime

```json
{
  "model_path": "G:\\wl3dgs\\3dgs_runs\\gsw_strict_baseline_v2_repeated_30k_20260630\\R1\\self_Trackmobile_4650TM_Mobile_Railcar_Mover",
  "start_time": "2026-06-30T19:43:45.771061+08:00",
  "end_time": "2026-06-30T21:14:51.475601+08:00",
  "duration_sec": 5465,
  "duration_min": 91.08333333333333,
  "peak_gpu_mem_mb": 11008,
  "gpu_monitor_rows": 545,
  "gaussian_count": 136793,
  "checkpoint_size_bytes": 100696283,
  "split_hash": "c65cc61677ffb46cd329d57fc72d30ed40d42e0f9a47d75c5a0931950c9f2cd7",
  "train_count": 13,
  "test_count": 2
}
```
