# BASELINE_TABLE_REVISION_PLAN

- Strict main table: retain only class A rows with valid metrics. Currently this means corrected GS-W designated successful rows and verified official class A scenes.
- Legacy GS-W: move to secondary conditioned/non-strict table; use as protocol-risk illustration only.
- Splatfacto-W: move to secondary transductive/half-image table; do not rank against strict official/GS-W.
- Luminance-GS: mark provisional/unresolved; run bounded diagnostics before any 30k rerun.
- WildGaussians: mark invalid/remove current numeric rows due dark/adapter-invalid renders; diagnose wrapper before considering rerun.
- Old official 12-scene rows: class B unless verified by provenance; do not mix with verified strict table.
