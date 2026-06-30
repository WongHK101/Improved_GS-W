# OFFICIAL_3DGS_PROVISIONAL_AUDIT

- Audited path: `G:\wl3dgs\3dgs_runs\official_expcomp_clean_r1_max1600_iter30000_20260622\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- cfg_args: `Namespace(sh_degree=3, source_path='G:\\wl3dgs\\3dgs_undistorted\\max1600\\self_Trackmobile_4650TM_Mobile_Railcar_Mover', model_path='G:\\wl3dgs\\3dgs_runs\\official_expcomp_clean_r1_max1600_iter30000_20260622\\self_Trackmobile_4650TM_Mobile_Railcar_Mover', images='images', depths='', resolution=1, white_background=False, train_test_exp=True, data_device='cpu', eval=True)`
- results.json: `{"ours_30000": {"SSIM": 0.21798548102378845, "PSNR": 11.76401138305664, "LPIPS": 0.5866167545318604}}`
- rendered test images: `2`, GT images: `2`
- final checkpoint exists: `True`
- gaussian count: `1708044`

## Match Checks

- Trackmobile scene: yes, path contains `self_Trackmobile_4650TM_Mobile_Railcar_Mover`.
- Resolution=1: yes, cfg contains `resolution=1`.
- 30000 iterations: checkpoint/results method `ours_30000` exists.
- Full-image evaluation: likely yes from official `metrics.py` output layout, but no explicit protocol manifest is stored.
- 13/2 frozen-equivalent split: not proven. `cfg_args` only has `eval=True`; no `split_manifest.json`/`split_used.json` is present. Render count is 2 test images, but train image list/hash is not recorded.
- No test RGB conditioning: yes for official 3DGS architecture; no GS-W appearance module.
- Same GT/metrics: partly supported by two test image count and same source path, but exact split/hash is not recorded.
- Not old TourLight checkpoint: likely yes by run name and official cfg, but no clean git commit/provenance file is present in this directory.

## Decision

`not sufficiently matched` for final go/no-go. It is useful as a provisional reference only.
