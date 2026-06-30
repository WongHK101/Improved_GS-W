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

`set_eval(True)` disables color-net dropout and feature mask; `set_eval(False)` restores training state. However, `GaussianModel.set_eval()` does not call `self.map_generator.eval()`. The map generator contains ResNet/UNet BatchNorm layers, so evaluation renders can update BatchNorm running statistics if the module remains in training mode. This is a concrete potential training-trajectory side effect for any iteration where `training_report()` actually renders validation views.

In the inspected Trackmobile 30k configs, both historical and clean runs have `test_iterations=[30000]`, so periodic validation did not run at iteration 7000 for these saved cfgs. The final iteration 30000 evaluation may still affect post-training state if followed by saving/rendering, but it cannot explain training trajectory before 30000. The short runs disable periodic test evaluation with `--test_iterations 1000000`, so their A1/A2/B/C divergence is not caused by periodic evaluation.

## Conclusion

Periodic evaluation has a plausible BatchNorm state side effect in the code, but the recorded Trackmobile 30k cfgs do not show a 7000-iteration evaluation. For the observed 1.24 dB historical-clean gap, periodic evaluation is therefore a lower-priority hypothesis than same-code CUDA/operator non-determinism and historical dirty-code differences.
