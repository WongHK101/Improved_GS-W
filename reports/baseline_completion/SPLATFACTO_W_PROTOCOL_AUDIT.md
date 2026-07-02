# SPLATFACTO_W_PROTOCOL_AUDIT

- Current numeric coverage: `12/12` rows in historical summary.
- Protocol decision: `not strict full-image held-out comparable`.
- Classification: `C. transductive / half-image protocol`.

Evidence:

- `external_baselines/splatfacto-w/splatfactow/splatfactow_config.py:43` sets `SplatfactoWModelConfig(eval_right_half=True)` for the active config.
- `external_baselines/splatfacto-w/splatfactow/splatfactow_model.py:252-255` defines `use_avg_appearance` and `eval_right_half`; `1261-1263` applies hacked right-half evaluation.
- `external_baselines/splatfacto-w/splatfactow/splatfactow_datamanager.py:349-352` detects eval images during training and masks the right half, which means held-out target images are not absent from training.
- The model has appearance embeddings and appearance features (`splatfactow_model.py:226-254`, `345-413`, `860-864`, `937-954`).

Result handling:

- Move current Splatfacto-W rows out of the strict main table.
- They may be retained in a secondary conditioned/transductive table.
- No retraining was started in this round.

- Historical metric rows found: `13`.
