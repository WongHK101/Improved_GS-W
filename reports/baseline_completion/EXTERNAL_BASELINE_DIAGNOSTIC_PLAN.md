# EXTERNAL_BASELINE_DIAGNOSTIC_PLAN

This table records bounded diagnostics and remaining follow-up. No full 30k external baseline reruns were launched.

- Luminance-GS: import/API mismatch blocks reader/train smoke in the current environment; no full 30k rerun.
- WildGaussians: import and two-scene reader smoke pass; no full 12-scene rerun until dark-render cause is isolated.
- Any adapter fix must stay in wrapper/adapter code, not in method core source.

| method | scene | diagnostic | status |
|---|---|---|---|
| Luminance-GS | self_3000t_Press | success_scene_reader_render_smoke | blocked_by_import_api_mismatch |
| Luminance-GS | self_3000t_Press | failed_scene_config_plus_300_1000_iter_smoke | blocked_by_import_api_mismatch |
| Luminance-GS | self_Trackmobile_4650TM_Mobile_Railcar_Mover | failed_scene_config_plus_300_1000_iter_smoke | blocked_by_import_api_mismatch |
| Luminance-GS | web_Terrestrial | failed_scene_config_plus_300_1000_iter_smoke | blocked_by_import_api_mismatch |
| Luminance-GS | web_metopa_images | failed_scene_config_plus_300_1000_iter_smoke | blocked_by_import_api_mismatch |
| Luminance-GS | web_statue_images | failed_scene_config_plus_300_1000_iter_smoke | blocked_by_import_api_mismatch |
| WildGaussians | official_or_minimal_public_sample | upstream_sample_validation | import_smoke_passed_sample_not_run |
| WildGaussians | web_Baalshamin_images | simple_scene_adapter_smoke | reader_smoke_passed_render_invalid |
| WildGaussians | web_doss_images | complex_scene_adapter_smoke | reader_smoke_passed_render_invalid |
