# SPLATFACTO_W_PROTOCOL_AUDIT

- Current 30k numeric coverage: `12/12` rows in historical summary.
- Protocol decision: `not strict full-image held-out comparable`.
- Classification: `D. protocol incompatible with the current strict main table`.

Evidence:

- Source defaults are mixed: `external_baselines/splatfacto-w/splatfactow/splatfactow_config.py:43` enables right-half evaluation for the non-light phototourism config, while the light config at `splatfactow_config.py:126-160` uses `FullImageDatamanagerConfig` and `use_avg_appearance=True` without setting `eval_right_half`.
- The 12 historical 30k saved configs all use `splatfacto-w-light`, `eval_right_half=false`, `use_avg_appearance=true`, `eval_mode=interval`, `eval_interval=8`, `train_split_fraction=0.9`, and `downscale_factor=1`; see `SPLATFACTO_W_HISTORICAL_RUN_AUDIT.csv`.
- `external_baselines/splatfacto-w/splatfactow/splatfactow_model.py:937-956` uses per-camera appearance when `camera.metadata['cam_idx']` exists, otherwise uses average appearance if `use_avg_appearance=True`. The historical eval logs do not show test-time fitting or test RGB appearance optimization.
- `splatfactow_model.py:1261-1265` can crop to the right half only when `eval_right_half=True`; the saved historical configs set it to false, so the previous right-half classification is not supported for these 12 rows.
- Existing Splatfacto-W outputs contain `0` render-like PNG/JPG files across 12 scenes. `ns-eval` wrote aggregate `eval.json` files only, so the requested unified full-image evaluator cannot be run from existing renders.
- The split is controlled by Nerfstudio's COLMAP dataparser command (`--eval-mode interval --eval-interval 8`) and saved config fields, not by a frozen manifest path/hash in the Splatfacto-W artifacts. Even if the interval likely resembles LLFF hold-8, the current evidence is insufficient to treat the aggregate rows as verified strict-main results.

Result handling:

- Move current Splatfacto-W rows out of the strict main table.
- They may be retained only as provisional Nerfstudio-internal aggregate numbers with a clear protocol caveat.
- A fair Splatfacto-W competitor would require either saved per-view render/GT export from the existing checkpoints followed by the unified evaluator, or a new GPT-approved strict run/export plan. This round did not start that work.
- No retraining was started in this round.

- Historical metric rows found: `13` total, including `12` 30k rows.
- Historical configs with `eval_right_half=false`: `12/12`.
- Historical configs with `use_avg_appearance=true`: `12/12`.
