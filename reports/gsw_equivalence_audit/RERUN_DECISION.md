# Rerun Decision

Primary classification: **D - configuration, data, code and metrics are broadly equivalent for membership/metric comparability, with residual optimization/code-path uncertainty**.

Secondary findings: **B - test-appearance leakage is present in historical legacy evaluation**. Historical training membership is held-out, but historical legacy rendering is not strict because test RGB conditions the appearance branch.

Configuration, split membership, data pixels, camera poses, and the metric pipeline are broadly equivalent for the purpose of judging training membership and metric comparability. The remaining historical-vs-clean PSNR gap is real, but it is not explained by training-view leakage through the split.

- Unified historical GS-W legacy PSNR: `12.385472`.
- Unified clean GS-W legacy PSNR: `11.150374`.
- Historical minus clean legacy gap: `1.235098 dB`.
- Image RGB equivalence: `True`.
- Camera pose equivalence: `True`.
- Final Gaussian count difference (clean - historical): `7039`.
- Key configuration mismatches recorded: `8`.

## Main gap source

The most concrete source of non-equivalence is the completed optimization state: learned weights differ and final Gaussian count differs. The clean repo also contains strict split/evaluation plumbing and uses the direct source path, while the historical run used the adapter path and a dirty historical worktree. These are enough to require a matched reproduction before treating the 1.24 dB gap as an algorithmic conclusion.

## No-go status

The previous strict GS-W no-go should be **paused/qualified**, not fully retracted. It remains true that strict held-out GS-W on the current clean 30k checkpoint is weak. It is not yet proven that GS-W itself is unsuitable as a method base, because the historical legacy checkpoint was materially better under the same target-conditioned evaluation.

## Corrected 30k requirement

A corrected matched 30k is required if the team wants a definitive historical-vs-clean equivalence answer. This audit does not run it.

Important: do not use current clean code with `--eval --split_mode legacy` for this scene unless the historical `uid = extr.id` COLMAP fix is also applied. Current clean frozen-manifest splitting is filename-based and avoids that legacy TSV UID hazard.

Suggested single command, for GPT review only:

```powershell
conda run -n 3dgs python train.py --source_path "G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w\self_Trackmobile_4650TM_Mobile_Railcar_Mover\dense" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --model_path "G:\wl3dgs\3dgs_runs\gsw_matched_historical_adapter_r1_iter30000_REVIEW_ONLY\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --resolution 1 --iterations 30000 --split_mode frozen_manifest --split_file "G:\WL3DGS\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json" --test_appearance_mode legacy_target_rgb --test_iterations 30000 --save_iterations 30000 --render_after_train --metrics_after_train --quiet
```
