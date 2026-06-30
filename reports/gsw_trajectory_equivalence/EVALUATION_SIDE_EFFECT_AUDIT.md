# Evaluation Side-Effect Audit

- Historical cfg `test_iterations`: `[30000]`.
- Historical cfg `save_iterations`: `[30000, 30000, 30000]`.
- Clean 30k cfg `test_iterations`: `[30000]`.
- Clean 30k cfg `save_iterations`: `[30000, 30000, 30000]`.

## Code path

- Clean train calls `gaussians.set_eval(True)` before `training_report`: `True`.
- Clean train calls `gaussians.set_eval(False)` after `training_report`: `True`.
- Historical train has set_eval bracket around `training_report`: `True`.
- `training_report` renders test and sampled train views only when `iteration in testing_iterations`.
- Historical `training_report` always renders test views with legacy target RGB; clean code can use strict test appearance mode when `args` is passed.

## State risk

`set_eval(True)` disables color-net dropout and feature mask; `set_eval(False)` restores training state. However, evaluation performs rendering through `map_generator` and `color_net`, and may consume CUDA/PyTorch state depending on kernels and dropout behavior. The short runs disable periodic test evaluation by setting `--test_iterations 1000000`, so short-run trajectory differences are not caused by periodic evaluation.

## Conclusion

Periodic evaluation remains a plausible 30k trajectory side-effect after 7000 only if historical and clean evaluation schedules or appearance modes differ. This audit does not run to 7000; a separate one-step evaluation side-effect smoke can be added if GPT requires direct checksum-before/after evidence.
