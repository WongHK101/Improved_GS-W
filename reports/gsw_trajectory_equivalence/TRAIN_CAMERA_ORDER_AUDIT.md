# Train Camera Order Audit

- clean_adapter_frozen: `['0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0010.jpg', '0011.jpg', '0012.jpg', '0013.jpg', '0014.jpg', '0015.jpg']`
- clean_adapter_legacy_tsv: `['0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0010.jpg', '0011.jpg', '0012.jpg', '0013.jpg', '0014.jpg', '0015.jpg']`
- clean_direct_frozen: `['0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0010.jpg', '0011.jpg', '0012.jpg', '0013.jpg', '0014.jpg', '0015.jpg']`
- historical_cfg_semantics: `['0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg', '0007.jpg', '0008.jpg', '0010.jpg', '0011.jpg', '0012.jpg', '0013.jpg', '0014.jpg', '0015.jpg']`

## Pairwise order equality

- clean_adapter_frozen vs clean_adapter_legacy_tsv: `True`
- clean_adapter_frozen vs clean_direct_frozen: `True`
- clean_adapter_frozen vs historical_cfg_semantics: `True`
- clean_adapter_legacy_tsv vs clean_direct_frozen: `True`
- clean_adapter_legacy_tsv vs historical_cfg_semantics: `True`
- clean_direct_frozen vs historical_cfg_semantics: `True`

## Interpretation

All probes use `Scene(..., shuffle=False)` as the training script does. Random camera sampling is therefore sensitive to this stored list order. Legacy TSV matching requires `legacy_tsv_uid_source=extrinsic` for this scene because the historical dirty worktree used `uid = extr.id`; the clean default `intrinsic` path is intentionally not treated as historical-compatible.
