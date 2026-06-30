# GS-W Historical vs Clean Equivalence Audit Summary

Generated: 2026-06-30T05:59:24

## Key answers

1. Historical training used `13` images.
2. `0001.jpg` and `0009.jpg` did **not** enter historical training; both were historical test images.
3. Historical `12.385471 dB` is **not strict held-out** because historical GS-W legacy rendering conditions the appearance branch on held-out test RGB.
4. Historical/current clean split membership equal: `True`.
5. Historical GS-W legacy unified PSNR: `12.385472`.
6. Current clean GS-W legacy unified PSNR: `11.150374`.
7. Current clean GS-W strict intrinsic unified PSNR: `11.156010`.
8. Historical Gaussian count: `133906`; current clean Gaussian count: `140945`.

## Current clean 30k configuration status

The clean 30k enabled the complete core GS-W switches found in the historical config: `use_colors_precomp=True`, `use_features_mask=True`, `use_kmap_pjmap=True`, `use_lpips_loss=True`, `map_num=3`, `map_generator_type=unet`, `features_dim=48`, and `color_net_type=naive`.

Key argument differences are: `['eval', 'split_source', 'tsv_path', 'source_path', 'sparse_subdir', 'split_mode', 'split_file', 'test_appearance_mode']`.
Most are protocol/data-path differences: historical `source_path` points to the GS-W dense adapter and uses `eval=True` legacy TSV splitting; clean points to the original COLMAP source and uses `split_mode=frozen_manifest`.

## Main interpretation

The 1.24 dB historical-vs-clean legacy gap is real under unified re-evaluation, but current evidence only observes that the final optimization states differ. The cause of the training-trajectory divergence has not yet been localized, so final Gaussian count and learned-weight differences must be treated as outcomes, not root causes.

## Leakage taxonomy

- Training-set leakage: no.
- Test-appearance leakage: yes for historical legacy evaluation. This is an independent leakage finding, not a data/adapter non-equivalence classification.
- Evaluation-protocol leakage: no split/GT/size/mask mismatch found in existing renders; appearance protocol differs.
