# BENCHMARK_RUNNER_AUDIT

## Runner Scope

- New orchestration files live under `tools/two_scene_benchmark/`.
- Reports live under `reports/two_scene_benchmark/`.
- The runner invokes existing official/GS-W training entry points serially and records metadata.
- It does not modify training source files, loss definitions, renderer behavior, split protocol, densification parameters, or appearance mode.

## Resumability Rules

- Each run has an independent output directory.
- If the final checkpoint is complete, the runner records `skipped_existing_complete` and does not rerun.
- If a target directory exists without the complete final checkpoint, it records `interrupted_existing_incomplete` and stops before treating that case as successful.
- Process return codes come directly from `subprocess.Popen(...).returncode`.
- All commands are launched serially by this Python process; no concurrent `conda run` jobs are started.

## Pilot Diagnostics

The GS-W 5000-iteration pilot is launched via `tools/two_scene_benchmark/gsw_pilot_train.py`, a diagnostics-only wrapper. It imports the unchanged `train.py`, expands trace sample iterations, and monkeypatches runtime methods only to count densification events and record CUDA memory. The wrapped functions call the original implementations and do not change parameters, tensors, losses, renderer outputs, optimizer steps, or return values.

## Protocol Freeze

- Required GS-W baseline commit: `ddc6d8702b2e838dc989d612ca23fb311b79f280`
- Required GS-W baseline tag: `gsw-strict-baseline-v2`
- Screening freeze tag: `gsw-two-scene-screening-v1`
- Official worktree: `G:\wl3dgs\Official_3DGS_Strict_Control`
- Official commit: `54c035f7834b564019656c3e3fcc3646292f727d`
- Official flags disable depth regularization with `--depth_l1_weight_init 0 --depth_l1_weight_final 0`.
- Official does not pass `--train_test_exp`; antialiasing remains default false.
- GS-W always uses `--split_mode frozen_manifest` and `--test_appearance_mode strict_intrinsic`.
- All training commands use `--test_iterations 1000000`, so no train-time test evaluation is triggered for 5000 or 30000 iteration runs.

## Environment Probe

```json
{
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_total_memory_mb": 24564,
  "sys_executable": "D:\\anaconda\\envs\\3dgs\\python.exe",
  "python_version": "Python 3.10.18",
  "conda_3dgs_probe": ""
}
```
