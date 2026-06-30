# Short Run Pairwise Comparison

All comparisons below use trace rows at initialization, 1, 10, 100, 499, 500, 501 and 1000 iterations. A checksum difference is treated as a trajectory split, not automatically as a root cause.

## A1_clean_direct_frozen vs A2_clean_direct_frozen_repeat

- Initialization equal: `True`.
- Camera sampling sequence equal: `True`.
- Python/NumPy/Torch RNG hashes at traced points equal: `True`.
- First scalar loss difference: `iteration 10: loss_delta=-0.002312392`.
- First parameter-checksum difference: `iteration 10: xyz_checksum,features_checksum,map_generator_checksum,color_net_checksum,box_coord_checksum`.
- First Gaussian-count difference: `iteration 1000: 6987 vs 6752`.
- Iteration 1000 Gaussian delta (left - right): `235`.
- Iteration 1000 total-loss delta (left - right): `-0.000294596`.

## A1_clean_direct_frozen vs B_clean_adapter_frozen

- Initialization equal: `True`.
- Camera sampling sequence equal: `True`.
- Python/NumPy/Torch RNG hashes at traced points equal: `True`.
- First scalar loss difference: `iteration 10: loss_delta=0.001182228`.
- First parameter-checksum difference: `iteration 10: xyz_checksum,features_checksum,map_generator_checksum,color_net_checksum,box_coord_checksum`.
- First Gaussian-count difference: `iteration 1000: 6987 vs 6948`.
- Iteration 1000 Gaussian delta (left - right): `39`.
- Iteration 1000 total-loss delta (left - right): `-0.004942313`.

## B_clean_adapter_frozen vs C_clean_adapter_legacy_tsv

- Initialization equal: `True`.
- Camera sampling sequence equal: `True`.
- Python/NumPy/Torch RNG hashes at traced points equal: `True`.
- First scalar loss difference: `iteration 10: loss_delta=-0.000491321`.
- First parameter-checksum difference: `iteration 10: xyz_checksum,features_checksum,map_generator_checksum,color_net_checksum,box_coord_checksum`.
- First Gaussian-count difference: `iteration 1000: 6948 vs 6888`.
- Iteration 1000 Gaussian delta (left - right): `60`.
- Iteration 1000 total-loss delta (left - right): `0.007859811`.

- C_clean_adapter_legacy_tsv vs D_historical_worktree: unavailable; one or both traces missing.
## Interpretation rules

- A1/A2 tests same-code same-seed repeatability.
- A/B tests direct source vs junction adapter with the same frozen manifest.
- B/C tests frozen manifest vs historical-compatible legacy TSV/eval path.
- C/D can only be answered if a true historical-worktree trace is available; otherwise C is the clean-code historical-compatible proxy.

## Main interpretation

A1/A2 already diverge by iteration 10 despite identical initialization, camera sampling sequence and traced RNG hashes. Therefore A/B and B/C checksum differences at iteration 10 cannot be attributed to adapter path or split implementation unless they exceed the same-code same-seed variance. In the current short runs, camera order, sampled images and traced RNG hashes are identical across A1/A2/B/C at every traced point; the observed trajectory divergence is most consistent with CUDA/operator non-determinism or untraced kernel-level state.
