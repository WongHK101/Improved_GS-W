# LUMINANCE_GS_PROTOCOL_AUDIT

- Historical metric rows found: `10`.
- Missing final metric scenes: `self_Trackmobile_4650TM_Mobile_Railcar_Mover, web_metopa_images`.
- Protocol classification: `E. adapter/configuration invalid in current local state`.
- Current rows should not enter strict main table: the current 3dgs environment fails to import Luminance-GS because `pycolmap.SceneManager` is unavailable.

Evidence:

- `external_baselines/Luminance-GS/Luminance-GS/README.md` describes LOM low-light/overexposure and MipNeRF360-varying, not this COLMAP tourism protocol.
- `3dgs_runs/external_baselines_20260620/summary/failures.csv` records multiple failed attempts.
- `metrics_all_methods.csv` contains Luminance-GS rows for only a subset/low-confidence set of scenes.
- `reports/baseline_completion/logs/external_bounded_diagnostics/luminance_import_smoke.log` records `ImportError: cannot import name 'SceneManager' from 'pycolmap'`; `luminance_pycolmap_probe.log` records pycolmap `4.0.4` and `has_SceneManager=false`.
- `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_RUNS.csv` and `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_ADAPTER.csv` contain per-scene log/adapter diagnostics.
