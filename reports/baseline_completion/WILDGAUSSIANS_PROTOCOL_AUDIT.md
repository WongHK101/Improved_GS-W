# WILDGAUSSIANS_PROTOCOL_AUDIT

- Historical metric rows found: `13`.
- PSNR range in historical summary: `4.9407` to `12.1853`.
- Protocol classification: `E. adapter/configuration invalid` pending render/checkpoint/appearance repair.
- Current WildGaussians values should be removed from strict/provisional numeric comparisons.

Evidence:

- `metrics_all_methods.csv` shows broadly collapsed PSNR/SSIM values for WildGaussians, consistent with dark/invalid renders.
- `wildgaussians/utils.py:130-146` performs color-space/background conversion; `utils.py:154-169` saves PNG output.
- `wildgaussians/evaluation.py:332` explicitly blends with black background for metrics.
- Numeric image statistics are in `WILDGAUSSIANS_DIAGNOSTIC_RESULTS.csv` (`16` sampled prediction images).
- `logs/external_bounded_diagnostics/wild_import_smoke.log` passes; `wild_reader_smoke.log` loads both `web_Baalshamin_images` and `web_doss_images` train/test adapters. Thus the remaining failure is downstream of basic COLMAP reader construction.
