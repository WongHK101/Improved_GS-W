# Rerun Decision

Primary classification: **D - high-level configuration, data and metric inputs are broadly equivalent, but training-trajectory equivalence is unresolved**.

Independent leakage finding: **test-appearance leakage is present in historical legacy evaluation**. Historical training membership is held-out, but historical legacy rendering is not strict because test RGB conditions the appearance branch. This is not classified as B, because B denotes data/adapter non-equivalence and the adapter data audit found image, camera and point-cloud equivalence.

Configuration, split membership, data pixels, camera poses, and the metric pipeline are broadly equivalent for the purpose of judging training membership and metric comparability. The remaining historical-vs-clean PSNR gap is real, but it is not explained by training-view leakage through the split.

- Unified historical GS-W legacy PSNR: `12.385472`.
- Unified clean GS-W legacy PSNR: `11.150374`.
- Historical minus clean legacy gap: `1.235098 dB`.
- Image RGB equivalence: `True`.
- Camera pose equivalence: `True`.
- Final Gaussian count difference (clean - historical): `7039`.
- Key configuration mismatches recorded: `8`.

## Current gap interpretation

The 1.24 dB gap is real, but the current audit only observes that the completed optimization states differ. Final learned weights and Gaussian counts are post-divergence outcomes, not an identified root cause. The cause of training-trajectory divergence remains unresolved and must be audited through low-cost seed, camera-order, sampling-trace and short-run checks before any 30k reproduction has high information value.

## No-go status

The previous strict GS-W no-go should be **paused/qualified**, not fully retracted. It remains true that strict held-out GS-W on the current clean 30k checkpoint is weak. It is not yet proven that GS-W itself is unsuitable as a method base, because the historical legacy checkpoint was materially better under the same target-conditioned evaluation.

## Corrected 30k status

The previously suggested corrected 30k is **not approved / insufficiently matched**. Switching only to the adapter `source_path` while using clean code and a frozen manifest does not constitute an exact historical reproduction, because the adapter is a junction to the same images and sparse model and data equivalence has already been established.

Important: do not use current clean code with `--eval --split_mode legacy` for this scene unless the historical `uid = extr.id` COLMAP fix is also applied. Current clean frozen-manifest splitting is filename-based and avoids that legacy TSV UID hazard.

Rejected previous command, retained only for traceability:

```powershell
conda run -n 3dgs python train.py --source_path "G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\dense" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --model_path "G:\wl3dgs\3dgs_runs\gsw_matched_historical_adapter_r1_iter30000_REVIEW_ONLY\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\WL3DGS\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --test_appearance_mode legacy_target_rgb --test_iterations 30000 --save_iterations 30000 --render_after_train --metrics_after_train --quiet
```
