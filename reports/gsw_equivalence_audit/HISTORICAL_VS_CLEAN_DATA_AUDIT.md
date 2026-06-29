# Historical vs Clean Data Audit

- Historical adapter dense path: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\dense`
- Clean source path: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Historical sparse path: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\dense\sparse`
- Clean sparse path: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover\sparse\0`

## Image equivalence

- Registered images audited: `15`.
- File SHA256 all equal: `True`.
- Decoded RGB SHA256 all equal: `True`.
- Maximum absolute pixel difference across audited images: `0`.
- Maximum mean absolute pixel difference across audited images: `0.0`.
- No resize, crop, color conversion, EXIF-orientation discrepancy, or JPEG re-encoding was detected when all SHA256 values are equal.

## Windows link type

- Historical `dense/images` link info: `{'FullName': 'G:\\wl3dgs\\3dgs_runs\\external_baselines_20260620\\adapters\\gs_w\\self_Trackmobile_4650TM_Mobile_Railcar_Mover\\dense\\images', 'Mode': 'd----l', 'LinkType': 'Junction', 'Target': 'G:\\WL3DGS\\3dgs_undistorted\\max1600\\self_Trackmobile_4650TM_Mobile_Railcar_Mover\\images', 'Attributes': 'Directory, ReparsePoint'}`.
- Historical `dense/sparse` link info: `{'FullName': 'G:\\wl3dgs\\3dgs_runs\\external_baselines_20260620\\adapters\\gs_w\\self_Trackmobile_4650TM_Mobile_Railcar_Mover\\dense\\sparse', 'Mode': 'd----l', 'LinkType': 'Junction', 'Target': 'G:\\WL3DGS\\3dgs_undistorted\\max1600\\self_Trackmobile_4650TM_Mobile_Railcar_Mover\\sparse\\0', 'Attributes': 'Directory, ReparsePoint'}`.
- If `LinkType` is populated, the adapter is a Windows link/junction/symlink to the original data rather than a re-encoded image export. If blank, PowerShell reports it as a normal directory.

## COLMAP equivalence

- Camera pose all equal: `True`.
- Historical point count: `1920`.
- Clean point count: `1920`.
- Historical xyz mean: `[0.6614050175, 0.5337651329, 2.5159144044]`.
- Clean xyz mean: `[0.6614050175, 0.5337651329, 2.5159144044]`.

## Adapter side files

- Adapter parent files: `['dense', 'self_Trackmobile_4650TM_Mobile_Railcar_Mover.tsv']`.
- Adapter TSV: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\self_Trackmobile_4650TM_Mobile_Railcar_Mover.tsv`.
- The TSV affects historical GS-W split selection under `eval=True`; image/sparse files are otherwise equivalent to the clean source in this audit.

Detailed tables: `HISTORICAL_VS_CLEAN_IMAGE_CHECKSUMS.csv` and `HISTORICAL_VS_CLEAN_CAMERA_COMPARISON.csv`.
