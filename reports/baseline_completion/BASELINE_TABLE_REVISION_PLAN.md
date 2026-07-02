# BASELINE_TABLE_REVISION_PLAN

- Strict main table: retain only class A rows with valid metrics. Currently this means corrected GS-W designated successful rows and verified official class A scenes.
- Legacy GS-W: move to secondary conditioned/non-strict table; use as protocol-risk illustration only.
- Splatfacto-W: do not rank against strict official/GS-W. Historical 30k configs are full-image (`eval_right_half=false`) and use average appearance, but only aggregate Nerfstudio `eval.json` exists; strict use requires per-view render/GT export and unified re-evaluation or a GPT-approved strict rerun/export plan.
- Luminance-GS: mark provisional/unresolved; run bounded diagnostics before any 30k rerun.
- WildGaussians: mark invalid/remove current numeric rows due dark/adapter-invalid renders; diagnose wrapper before considering rerun.
- Old official 12-scene rows: class B unless verified by provenance; do not mix with verified strict table.
