# Random State Audit

- Historical cfg contains `seed`: `False`.
- Clean cfg contains `seed`: `False`.
- The previous audit's synthetic `seed=0` equality is not used here as evidence.

## Seed initialization path

- Historical `train.py` calls `safe_state(args.quiet)`: `True`.
- Clean `train.py` calls `safe_state(args.quiet)`: `True`.
- In both codebases, `safe_state()` is called in the `__main__` block after argparse/argument_init and before `training(...)`, `GaussianModel(...)`, and `Scene(...)` creation.
- Historical `safe_state` sets `random.seed(0)`: `True`.
- Historical `safe_state` sets `np.random.seed(0)`: `True`.
- Historical `safe_state` sets `torch.manual_seed(0)`: `True`.
- Clean `safe_state` sets `random.seed(0)`: `True`.
- Clean `safe_state` sets `np.random.seed(0)`: `True`.
- Clean `safe_state` sets `torch.manual_seed(0)`: `True`.
- Explicit `torch.cuda.manual_seed` / `manual_seed_all` found in clean code: `False`.

## Determinism flags

- `torch.backends.cudnn.deterministic` set in clean repo: `False`.
- `torch.backends.cudnn.benchmark` set in clean repo: `False`.
- `torch.use_deterministic_algorithms` set in clean repo: `False`.
- The CUDA rasterizer and PyTorch CUDA reductions may still be non-deterministic; A1/A2 short-run comparison is the empirical check.

## Conclusion

Historical and clean code both call `safe_state()` and both seed Python, NumPy and Torch CPU/CUDA default generator through `torch.manual_seed(0)`. However, neither cfg records a seed value, and no deterministic CUDA/cuDNN flags are set. Therefore the correct conclusion is: both appear to start from seed 0 through code path, but deterministic replay is not guaranteed and must be measured by A1/A2.
