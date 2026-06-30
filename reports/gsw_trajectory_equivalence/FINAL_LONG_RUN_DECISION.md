# Final Long-Run Decision

- A1/A2: `init_equal=True; sampling_equal=True; rng_hashes_equal=True; first_loss_diff=iteration 10: loss_delta=-0.002312392; first_param_diff=iteration 10: xyz_checksum,features_checksum,map_generator_checksum,color_net_checksum,box_coord_checksum; first_gaussian_diff=iteration 1000: 6987 vs 6752; iter1000_gaussian_delta=235; iter1000_loss_delta=-0.000294596`.
- A/B: `init_equal=True; sampling_equal=True; rng_hashes_equal=True; first_loss_diff=iteration 10: loss_delta=0.001182228; first_param_diff=iteration 10: xyz_checksum,features_checksum,map_generator_checksum,color_net_checksum,box_coord_checksum; first_gaussian_diff=iteration 1000: 6987 vs 6948; iter1000_gaussian_delta=39; iter1000_loss_delta=-0.004942313`.
- B/C: `init_equal=True; sampling_equal=True; rng_hashes_equal=True; first_loss_diff=iteration 10: loss_delta=-0.000491321; first_param_diff=iteration 10: xyz_checksum,features_checksum,map_generator_checksum,color_net_checksum,box_coord_checksum; first_gaussian_diff=iteration 1000: 6948 vs 6888; iter1000_gaussian_delta=60; iter1000_loss_delta=0.007859811`.

## Direct answers

1. A1/A2 are not byte-identical repeatable beyond iteration 1; they first diverge at iteration 10 while sampled images and traced RNG hashes remain equal.
2. Direct source and junction adapter have identical initialization, camera order and sampling sequence. Their later checksum differences are not distinguishable from A1/A2 same-code variance.
3. Frozen manifest and historical-compatible legacy TSV have identical train camera order and sampling sequence in this scene.
4. Clean vs historical code cannot be fully localized without instrumenting/running the historical dirty worktree. The clean historical-compatible proxy C diverges from B at the same early point already seen in A1/A2.
5. Periodic evaluation has a plausible BatchNorm state side effect because `map_generator.eval()` is not called, but the recorded Trackmobile 30k cfgs only evaluate at 30000, not 7000.

## Most credible current root cause for the 1.24 dB gap

The strongest current evidence is same-code same-seed trajectory non-determinism: A1/A2 split by iteration 10 despite identical initialization, sampled images and traced RNG hashes, and end at 6987 vs 6752 Gaussians by iteration 1000. Camera order, adapter path and legacy TSV are not supported as primary causes by the short-run evidence.

## Unique long-run recommendation

**C - Need clean repeated-seed 30k** if any new 30k is approved. The next long run should estimate strict GS-W seed/operator variance under the clean strict protocol, not chase the historical non-strict legacy result. No historical matched 30k is approved by this audit.
