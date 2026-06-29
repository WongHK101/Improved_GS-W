# Historical Split Membership Audit

- Historical TSV: `G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\self_Trackmobile_4650TM_Mobile_Railcar_Mover.tsv`
- Historical registered images: `15`.
- Historical train images: `13`.
- Historical test images: `2`.
- Historical train list: `['0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0010.jpg', '0011.jpg', '0012.jpg', '0013.jpg', '0014.jpg', '0015.jpg']`.
- Historical test list: `['0001.jpg', '0009.jpg']`.
- Current clean train list: `['0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0010.jpg', '0011.jpg', '0012.jpg', '0013.jpg', '0014.jpg', '0015.jpg']`.
- Current clean test list: `['0001.jpg', '0009.jpg']`.

## Required membership answer

- `0001.jpg` entered historical training: `False`.
- `0009.jpg` entered historical training: `False`.
- `0001.jpg` entered historical test: `True`.
- `0009.jpg` entered historical test: `True`.
- Historical/current membership equal: `True`.

## Interpretation

The historical 30k training membership was 13 images, and the two LLFF hold-8 images were held out from training. However, historical GS-W evaluation still used `legacy_target_rgb`, which conditions the appearance branch on the held-out RGB at render time. Therefore the historical `12.385471 dB` number is not a strict held-out result even though its training split was held-out.

## Leakage taxonomy

A. Training-set leakage: **No evidence of training-set leakage**. `0001.jpg` and `0009.jpg` are TSV `test` rows and are excluded from `train_cam_infos` under historical `eval=True` legacy TSV logic.

B. Test-appearance leakage: **Present for historical legacy evaluation**. Historical GS-W legacy rendering calls `pc.forward(viewpoint_camera)`, so the held-out test camera RGB is passed to `map_generator` for appearance conditioning.

C. Evaluation-protocol leakage: **No split/GT/size/mask leakage found in existing renders**. Unified metric audit uses the same two GT PNGs and full-image filename pairing. Historical and clean protocols still differ in appearance mode and source-path plumbing.
