# Short Run Pairwise Comparison

- A1_clean_direct_frozen vs A2_clean_direct_frozen_repeat: unavailable; one or both traces missing.
- A1_clean_direct_frozen vs B_clean_adapter_frozen: unavailable; one or both traces missing.
- B_clean_adapter_frozen vs C_clean_adapter_legacy_tsv: unavailable; one or both traces missing.
- C_clean_adapter_legacy_tsv vs D_historical_worktree: unavailable; one or both traces missing.

## Interpretation rules

- A1/A2 tests same-code same-seed repeatability.
- A/B tests direct source vs junction adapter with the same frozen manifest.
- B/C tests frozen manifest vs historical-compatible legacy TSV/eval path.
- C/D can only be answered if a true historical-worktree trace is available; otherwise C is the clean-code historical-compatible proxy.
