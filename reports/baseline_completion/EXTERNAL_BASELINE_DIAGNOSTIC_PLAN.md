# EXTERNAL_BASELINE_DIAGNOSTIC_PLAN

This plan defines bounded diagnostics only. It is not evidence that those diagnostics have been executed.

- Luminance-GS: diagnose current failure scenes with reader/config and 300-1000 iteration smoke only; no full 30k rerun.
- WildGaussians: first validate upstream output and two adapter scenes; no full 12-scene rerun until dark-render cause is isolated.
- Any adapter fix must stay in wrapper/adapter code, not in method core source.

| method | scene | diagnostic | status |
|---|---|---|---|
| Luminance-GS | self_3000t_Press | success_scene_reader_render_smoke | not_executed |
| Luminance-GS | self_3000t_Press | failed_scene_config_plus_300_1000_iter_smoke | not_executed |
| Luminance-GS | self_Trackmobile_4650TM_Mobile_Railcar_Mover | failed_scene_config_plus_300_1000_iter_smoke | not_executed |
| Luminance-GS | web_Terrestrial | failed_scene_config_plus_300_1000_iter_smoke | not_executed |
| Luminance-GS | web_metopa_images | failed_scene_config_plus_300_1000_iter_smoke | not_executed |
| Luminance-GS | web_statue_images | failed_scene_config_plus_300_1000_iter_smoke | not_executed |
| WildGaussians | official_or_minimal_public_sample | upstream_sample_validation | not_executed |
| WildGaussians | web_Baalshamin_images | simple_scene_adapter_smoke | not_executed |
| WildGaussians | web_doss_images | complex_scene_adapter_smoke | not_executed |
