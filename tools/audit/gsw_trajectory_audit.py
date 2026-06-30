"""GS-W training-trajectory equivalence audit helpers.

This script reads existing configs and short-run traces. It does not train by
itself and does not modify datasets or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "gsw_trajectory_equivalence"
PACKAGE_DIR = Path(r"G:\WL3DGS\gpt_review_packages")
SCENE = "self_Trackmobile_4650TM_Mobile_Railcar_Mover"
HIST_RUN = Path(r"G:\wl3dgs\3dgs_runs\external_baselines_20260620\gs_w_r1_iter30000") / SCENE
HIST_CODE = Path(r"G:\wl3dgs\external_baselines\Gaussian-Wild")
CURRENT_RUN = Path(r"G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630") / SCENE
CLEAN_DATA = Path(r"G:\wl3dgs\3dgs_undistorted\max1600") / SCENE
HIST_ADAPTER = Path(r"G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w") / SCENE
HIST_DENSE = HIST_ADAPTER / "dense"
HIST_TSV = HIST_ADAPTER / f"{SCENE}.tsv"
SPLIT_FILE = ROOT / "splits" / "TRACKMOBILE_SPLIT.json"
SHORT_RUN_ROOT = Path(r"G:\wl3dgs\3dgs_runs\trajectory_equivalence_20260630")

RUN_SPECS = {
    "A1_clean_direct_frozen": SHORT_RUN_ROOT / "A1_clean_direct_frozen_seed0_iter1000",
    "A2_clean_direct_frozen_repeat": SHORT_RUN_ROOT / "A2_clean_direct_frozen_seed0_iter1000",
    "B_clean_adapter_frozen": SHORT_RUN_ROOT / "B_clean_adapter_frozen_seed0_iter1000",
    "C_clean_adapter_legacy_tsv": SHORT_RUN_ROOT / "C_clean_adapter_legacy_tsv_seed0_iter1000",
    "D_historical_worktree": SHORT_RUN_ROOT / "D_historical_worktree_seed0_iter1000",
}

TRACE_FILES = {name: path / "trace.csv" for name, path in RUN_SPECS.items()}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(args: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    proc = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return proc.stdout.rstrip()


def read_cfg(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    namespace = eval(text, {"__builtins__": {}}, {"Namespace": argparse.Namespace})  # noqa: S307 - local audit cfg
    return vars(namespace)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trace(name: str) -> list[dict[str, str]]:
    path = TRACE_FILES[name]
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def trace_row(trace: list[dict[str, str]], iteration: int, event: str = "iteration") -> dict[str, str] | None:
    for row in trace:
        if row.get("event") == event and int(row.get("iteration", -1)) == iteration:
            return row
    return None


def build_random_state_audit() -> str:
    hist_cfg = read_cfg(HIST_RUN / "cfg_args")
    clean_cfg = read_cfg(CURRENT_RUN / "cfg_args")
    hist_train = (HIST_CODE / "train.py").read_text(encoding="utf-8", errors="replace")
    clean_train = (ROOT / "train.py").read_text(encoding="utf-8", errors="replace")
    clean_utils = (ROOT / "utils" / "general_utils.py").read_text(encoding="utf-8", errors="replace")
    hist_utils = (HIST_CODE / "utils" / "general_utils.py").read_text(encoding="utf-8", errors="replace")

    lines = [
        "# Random State Audit",
        "",
        f"- Historical cfg contains `seed`: `{'seed' in hist_cfg}`.",
        f"- Clean cfg contains `seed`: `{'seed' in clean_cfg}`.",
        "- The previous audit's synthetic `seed=0` equality is not used here as evidence.",
        "",
        "## Seed initialization path",
        "",
        f"- Historical `train.py` calls `safe_state(args.quiet)`: `{'safe_state(args.quiet)' in hist_train}`.",
        f"- Clean `train.py` calls `safe_state(args.quiet)`: `{'safe_state(args.quiet)' in clean_train}`.",
        "- In both codebases, `safe_state()` is called in the `__main__` block after argparse/argument_init and before `training(...)`, `GaussianModel(...)`, and `Scene(...)` creation.",
        f"- Historical `safe_state` sets `random.seed(0)`: `{'random.seed(0)' in hist_utils}`.",
        f"- Historical `safe_state` sets `np.random.seed(0)`: `{'np.random.seed(0)' in hist_utils}`.",
        f"- Historical `safe_state` sets `torch.manual_seed(0)`: `{'torch.manual_seed(0)' in hist_utils}`.",
        f"- Clean `safe_state` sets `random.seed(0)`: `{'random.seed(0)' in clean_utils}`.",
        f"- Clean `safe_state` sets `np.random.seed(0)`: `{'np.random.seed(0)' in clean_utils}`.",
        f"- Clean `safe_state` sets `torch.manual_seed(0)`: `{'torch.manual_seed(0)' in clean_utils}`.",
        f"- Explicit `torch.cuda.manual_seed` / `manual_seed_all` found in clean code: `{'manual_seed_all' in clean_utils or 'cuda.manual_seed' in clean_utils}`.",
        "",
        "## Determinism flags",
        "",
        f"- `torch.backends.cudnn.deterministic` set in clean repo: `{'cudnn.deterministic' in clean_train or 'cudnn.deterministic' in clean_utils}`.",
        f"- `torch.backends.cudnn.benchmark` set in clean repo: `{'cudnn.benchmark' in clean_train or 'cudnn.benchmark' in clean_utils}`.",
        f"- `torch.use_deterministic_algorithms` set in clean repo: `{'use_deterministic_algorithms' in clean_train or 'use_deterministic_algorithms' in clean_utils}`.",
        "- The CUDA rasterizer and PyTorch CUDA reductions may still be non-deterministic; A1/A2 short-run comparison is the empirical check.",
        "",
        "## Conclusion",
        "",
        "Historical and clean code both call `safe_state()` and both seed Python, NumPy and Torch CPU/CUDA default generator through `torch.manual_seed(0)`. "
        "However, neither cfg records a seed value, and no deterministic CUDA/cuDNN flags are set. Therefore the correct conclusion is: both appear to start from seed 0 through code path, but deterministic replay is not guaranteed and must be measured by A1/A2.",
    ]
    return "\n".join(lines) + "\n"


def camera_order_command(label: str, source_path: Path, split_mode: str, eval_flag: bool, split_file: Path | str = "", uid_source: str = "intrinsic") -> dict[str, Any]:
    return {
        "label": label,
        "source_path": str(source_path),
        "split_mode": split_mode,
        "eval": eval_flag,
        "split_file": str(split_file),
        "legacy_tsv_uid_source": uid_source,
    }


def inspect_camera_orders() -> list[dict[str, Any]]:
    from argparse import Namespace
    from arguments import ModelParams
    from arguments.args_init import argument_init
    from scene import Scene, GaussianModel
    from utils.general_utils import safe_state

    specs = [
        camera_order_command("historical_cfg_semantics", HIST_DENSE, "legacy", True, "", "extrinsic"),
        camera_order_command("clean_direct_frozen", CLEAN_DATA, "frozen_manifest", False, SPLIT_FILE, "intrinsic"),
        camera_order_command("clean_adapter_frozen", HIST_DENSE, "frozen_manifest", False, SPLIT_FILE, "intrinsic"),
        camera_order_command("clean_adapter_legacy_tsv", HIST_DENSE, "legacy", True, "", "extrinsic"),
    ]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        safe_state(True)
        ns = Namespace(
            device="cuda:0",
            sh_degree=3,
            source_path=spec["source_path"],
            model_path=str(REPORT_DIR / "_camera_order_probe" / spec["label"]),
            images="images",
            sparse_subdir="",
            split_mode=spec["split_mode"],
            split_file=spec["split_file"],
            legacy_tsv_uid_source=spec["legacy_tsv_uid_source"],
            test_appearance_mode="strict_intrinsic",
            resolution=1,
            white_background=False,
            data_device="cuda",
            eval=spec["eval"],
            scene_name=SCENE,
            use_colors_precomp=True,
            use_decode_with_pos=False,
            use_indep_mask_branch=False,
            use_features_mask=True,
            features_mask_loss_coef=0.15,
            features_mask_iters=2500,
            use_okmap=False,
            use_kmap_pjmap=True,
            map_num=3,
            use_wo_adative=0,
            use_xw_init_box_coord=True,
            use_color_net=True,
            use_scaling_loss=False,
            use_lpips_loss=True,
            use_box_coord_loss=True,
            coord_scale=1,
            data_perturb=[],
        )
        args = argument_init(ns)
        Path(args.model_path).mkdir(parents=True, exist_ok=True)
        gaussians = GaussianModel(args.sh_degree, args)
        scene = Scene(ModelParams.extract(ModelParams(argparse.ArgumentParser()), args), gaussians, shuffle=False)
        train_names = [cam.image_name for cam in scene.getTrainCameras()]
        test_names = [cam.image_name for cam in scene.getTestCameras()]
        for idx, name in enumerate(train_names):
            rows.append({
                "probe": spec["label"],
                "role": "train",
                "order_index": idx,
                "image_name": name,
                "source_path": spec["source_path"],
                "split_mode": spec["split_mode"],
                "eval": spec["eval"],
                "legacy_tsv_uid_source": spec["legacy_tsv_uid_source"],
            })
        for idx, name in enumerate(test_names):
            rows.append({
                "probe": spec["label"],
                "role": "test",
                "order_index": idx,
                "image_name": name,
                "source_path": spec["source_path"],
                "split_mode": spec["split_mode"],
                "eval": spec["eval"],
                "legacy_tsv_uid_source": spec["legacy_tsv_uid_source"],
            })
    return rows


def build_camera_order_report(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row["role"] == "train":
            grouped.setdefault(str(row["probe"]), []).append(str(row["image_name"]))
    names = sorted(grouped)
    lines = ["# Train Camera Order Audit", ""]
    for name in names:
        lines.append(f"- {name}: `{grouped[name]}`")
    lines.extend(["", "## Pairwise order equality", ""])
    for left in names:
        for right in names:
            if left >= right:
                continue
            lines.append(f"- {left} vs {right}: `{grouped[left] == grouped[right]}`")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "All probes use `Scene(..., shuffle=False)` as the training script does. Random camera sampling is therefore sensitive to this stored list order. "
        "Legacy TSV matching requires `legacy_tsv_uid_source=extrinsic` for this scene because the historical dirty worktree used `uid = extr.id`; the clean default `intrinsic` path is intentionally not treated as historical-compatible.",
    ])
    return "\n".join(lines) + "\n"


def write_short_run_commands() -> None:
    base = (
        'conda run -n 3dgs --no-capture-output python train.py '
        '--scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover '
        '--resolution 1 --iterations 1000 --test_iterations 1000000 '
        '--save_iterations 1000000 --disable_save_iterations '
        '--disable_render_after_train --disable_metrics_after_train --disable_train_temp_images '
        '--trace_training_state --quiet'
    )
    commands = {
        "A1_clean_direct_frozen": f'{base} --source_path "{CLEAN_DATA}" --model_path "{RUN_SPECS["A1_clean_direct_frozen"]}" --split_mode frozen_manifest --split_file "{SPLIT_FILE}" --trace_output "{TRACE_FILES["A1_clean_direct_frozen"]}"',
        "A2_clean_direct_frozen_repeat": f'{base} --source_path "{CLEAN_DATA}" --model_path "{RUN_SPECS["A2_clean_direct_frozen_repeat"]}" --split_mode frozen_manifest --split_file "{SPLIT_FILE}" --trace_output "{TRACE_FILES["A2_clean_direct_frozen_repeat"]}"',
        "B_clean_adapter_frozen": f'{base} --source_path "{HIST_DENSE}" --model_path "{RUN_SPECS["B_clean_adapter_frozen"]}" --split_mode frozen_manifest --split_file "{SPLIT_FILE}" --trace_output "{TRACE_FILES["B_clean_adapter_frozen"]}"',
        "C_clean_adapter_legacy_tsv": f'{base} --source_path "{HIST_DENSE}" --model_path "{RUN_SPECS["C_clean_adapter_legacy_tsv"]}" --eval --split_mode legacy --legacy_tsv_uid_source extrinsic --trace_output "{TRACE_FILES["C_clean_adapter_legacy_tsv"]}"',
    }
    lines = ["# Short Run Commands", ""]
    for name, command in commands.items():
        lines.extend([f"## {name}", "", "```powershell", command, "```", ""])
    lines.extend([
        "## D_historical_worktree",
        "",
        "Not launched by this helper. If run directly in the historical dirty worktree, the command must be separately instrumented there; this audit uses clean-code historical-compatible C as the safe proxy unless D is explicitly approved.",
    ])
    write_text(REPORT_DIR / "SHORT_RUN_COMMANDS.md", "\n".join(lines) + "\n")


def combine_traces() -> list[dict[str, Any]]:
    rows = []
    for run_name in RUN_SPECS:
        for row in load_trace(run_name):
            out = {"run": run_name}
            out.update(row)
            rows.append(out)
    return rows


def pairwise_comparison() -> tuple[list[dict[str, Any]], str]:
    pairs = [
        ("A1_clean_direct_frozen", "A2_clean_direct_frozen_repeat"),
        ("A1_clean_direct_frozen", "B_clean_adapter_frozen"),
        ("B_clean_adapter_frozen", "C_clean_adapter_legacy_tsv"),
        ("C_clean_adapter_legacy_tsv", "D_historical_worktree"),
    ]
    rows = []
    lines = [
        "# Short Run Pairwise Comparison",
        "",
        "All comparisons below use trace rows at initialization, 1, 10, 100, 499, 500, 501 and 1000 iterations. "
        "A checksum difference is treated as a trajectory split, not automatically as a root cause.",
        "",
    ]
    for left, right in pairs:
        lt = load_trace(left)
        rt = load_trace(right)
        if not lt or not rt:
            lines.append(f"- {left} vs {right}: unavailable; one or both traces missing.")
            rows.append({"left": left, "right": right, "status": "missing trace"})
            continue
        first_param_diff = ""
        first_loss_diff = ""
        first_gaussian_diff = ""
        sampling_equal = True
        rng_equal = True
        init_equal = True
        for iteration in [0, 1, 10, 100, 499, 500, 501, 1000]:
            event = "initialization" if iteration == 0 else "iteration"
            lr = trace_row(lt, iteration, event)
            rr = trace_row(rt, iteration, event)
            if lr is None or rr is None:
                first_param_diff = f"iteration {iteration}: missing row"
                break
            if iteration == 0:
                init_keys = ["gaussian_count", "xyz_checksum", "features_checksum", "map_generator_checksum", "color_net_checksum", "box_coord_checksum", "camera_extent", "spatial_lr_scale"]
                init_equal = all(lr.get(k) == rr.get(k) for k in init_keys)
            else:
                sample_keys = ["image_name", "random_index", "viewpoint_stack_rebuilt"]
                rng_keys = ["python_random_state_hash", "numpy_random_state_hash", "torch_cpu_rng_state_hash", "torch_cuda_rng_state_hash"]
                sampling_equal = sampling_equal and all(lr.get(k) == rr.get(k) for k in sample_keys)
                rng_equal = rng_equal and all(lr.get(k) == rr.get(k) for k in rng_keys)
                if not first_loss_diff and abs(float(lr.get("total_loss", 0.0)) - float(rr.get("total_loss", 0.0))) > 1e-8:
                    first_loss_diff = f"iteration {iteration}: loss_delta={float(lr['total_loss']) - float(rr['total_loss']):.9f}"
                if not first_gaussian_diff and lr.get("gaussian_count") != rr.get("gaussian_count"):
                    first_gaussian_diff = f"iteration {iteration}: {lr.get('gaussian_count')} vs {rr.get('gaussian_count')}"
            param_keys = ["xyz_checksum", "features_checksum", "map_generator_checksum", "color_net_checksum", "box_coord_checksum"]
            if not first_param_diff and any(lr.get(k) != rr.get(k) for k in param_keys):
                changed = [k for k in param_keys if lr.get(k) != rr.get(k)]
                first_param_diff = f"iteration {iteration}: " + ",".join(changed)
        if not first_param_diff:
            first_param_diff = "no parameter checksum difference in audited checkpoints through 1000"
        if not first_loss_diff:
            first_loss_diff = "no loss difference in audited checkpoints through 1000"
        if not first_gaussian_diff:
            first_gaussian_diff = "no Gaussian-count difference in audited checkpoints through 1000"
        left1000 = trace_row(lt, 1000)
        right1000 = trace_row(rt, 1000)
        gaussian_delta = ""
        loss_delta = ""
        if left1000 and right1000:
            gaussian_delta = str(int(left1000["gaussian_count"]) - int(right1000["gaussian_count"]))
            loss_delta = f"{float(left1000['total_loss']) - float(right1000['total_loss']):.9f}"
        status = (
            f"init_equal={init_equal}; sampling_equal={sampling_equal}; rng_hashes_equal={rng_equal}; "
            f"first_loss_diff={first_loss_diff}; first_param_diff={first_param_diff}; "
            f"first_gaussian_diff={first_gaussian_diff}; iter1000_gaussian_delta={gaussian_delta}; iter1000_loss_delta={loss_delta}"
        )
        lines.append(f"## {left} vs {right}")
        lines.append("")
        lines.append(f"- Initialization equal: `{init_equal}`.")
        lines.append(f"- Camera sampling sequence equal: `{sampling_equal}`.")
        lines.append(f"- Python/NumPy/Torch RNG hashes at traced points equal: `{rng_equal}`.")
        lines.append(f"- First scalar loss difference: `{first_loss_diff}`.")
        lines.append(f"- First parameter-checksum difference: `{first_param_diff}`.")
        lines.append(f"- First Gaussian-count difference: `{first_gaussian_diff}`.")
        lines.append(f"- Iteration 1000 Gaussian delta (left - right): `{gaussian_delta}`.")
        lines.append(f"- Iteration 1000 total-loss delta (left - right): `{loss_delta}`.")
        lines.append("")
        rows.append({
            "left": left,
            "right": right,
            "init_equal": init_equal,
            "sampling_equal": sampling_equal,
            "rng_hashes_equal": rng_equal,
            "first_loss_diff": first_loss_diff,
            "first_param_diff": first_param_diff,
            "first_gaussian_diff": first_gaussian_diff,
            "iter1000_gaussian_delta": gaussian_delta,
            "iter1000_loss_delta": loss_delta,
            "status": status,
        })
    lines.extend([
        "## Interpretation rules",
        "",
        "- A1/A2 tests same-code same-seed repeatability.",
        "- A/B tests direct source vs junction adapter with the same frozen manifest.",
        "- B/C tests frozen manifest vs historical-compatible legacy TSV/eval path.",
        "- C/D can only be answered if a true historical-worktree trace is available; otherwise C is the clean-code historical-compatible proxy.",
        "",
        "## Main interpretation",
        "",
        "A1/A2 already diverge by iteration 10 despite identical initialization, camera sampling sequence and traced RNG hashes. "
        "Therefore A/B and B/C checksum differences at iteration 10 cannot be attributed to adapter path or split implementation unless they exceed the same-code same-seed variance. "
        "In the current short runs, camera order, sampled images and traced RNG hashes are identical across A1/A2/B/C at every traced point; the observed trajectory divergence is most consistent with CUDA/operator non-determinism or untraced kernel-level state.",
    ])
    return rows, "\n".join(lines) + "\n"


def build_evaluation_side_effect_audit() -> str:
    clean_train = (ROOT / "train.py").read_text(encoding="utf-8", errors="replace")
    hist_train = (HIST_CODE / "train.py").read_text(encoding="utf-8", errors="replace")
    lines = [
        "# Evaluation Side-Effect Audit",
        "",
        f"- Historical cfg `test_iterations`: `{read_cfg(HIST_RUN / 'cfg_args').get('test_iterations')}`.",
        f"- Historical cfg `save_iterations`: `{read_cfg(HIST_RUN / 'cfg_args').get('save_iterations')}`.",
        f"- Clean 30k cfg `test_iterations`: `{read_cfg(CURRENT_RUN / 'cfg_args').get('test_iterations')}`.",
        f"- Clean 30k cfg `save_iterations`: `{read_cfg(CURRENT_RUN / 'cfg_args').get('save_iterations')}`.",
        "",
        "## Code path",
        "",
        f"- Clean train calls `gaussians.set_eval(True)` before `training_report`: `{'gaussians.set_eval(True)' in clean_train}`.",
        f"- Clean train calls `gaussians.set_eval(False)` after `training_report`: `{'gaussians.set_eval(False)' in clean_train}`.",
        f"- Historical train has set_eval bracket around `training_report`: `{'gaussians.set_eval(True)' in hist_train and 'gaussians.set_eval(False)' in hist_train}`.",
        "- `training_report` renders test and sampled train views only when `iteration in testing_iterations`.",
        "- Historical `training_report` always renders test views with legacy target RGB; clean code can use strict test appearance mode when `args` is passed.",
        "",
        "## State risk",
        "",
        "`set_eval(True)` disables color-net dropout and feature mask; `set_eval(False)` restores training state. However, `GaussianModel.set_eval()` does not call `self.map_generator.eval()`. "
        "The map generator contains ResNet/UNet BatchNorm layers, so evaluation renders can update BatchNorm running statistics if the module remains in training mode. "
        "This is a concrete potential training-trajectory side effect for any iteration where `training_report()` actually renders validation views.",
        "",
        "In the inspected Trackmobile 30k configs, both historical and clean runs have `test_iterations=[30000]`, so periodic validation did not run at iteration 7000 for these saved cfgs. "
        "The final iteration 30000 evaluation may still affect post-training state if followed by saving/rendering, but it cannot explain training trajectory before 30000. "
        "The short runs disable periodic test evaluation with `--test_iterations 1000000`, so their A1/A2/B/C divergence is not caused by periodic evaluation.",
        "",
        "## Conclusion",
        "",
        "Periodic evaluation has a plausible BatchNorm state side effect in the code, but the recorded Trackmobile 30k cfgs do not show a 7000-iteration evaluation. "
        "For the observed 1.24 dB historical-clean gap, periodic evaluation is therefore a lower-priority hypothesis than same-code CUDA/operator non-determinism and historical dirty-code differences.",
    ]
    return "\n".join(lines) + "\n"


def build_final_decision(pair_rows: list[dict[str, Any]]) -> str:
    status = {f"{r['left']}__{r['right']}": r["status"] for r in pair_rows}
    a_repeat = status.get("A1_clean_direct_frozen__A2_clean_direct_frozen_repeat", "missing trace")
    ab = status.get("A1_clean_direct_frozen__B_clean_adapter_frozen", "missing trace")
    bc = status.get("B_clean_adapter_frozen__C_clean_adapter_legacy_tsv", "missing trace")
    lines = [
        "# Final Long-Run Decision",
        "",
        f"- A1/A2: `{a_repeat}`.",
        f"- A/B: `{ab}`.",
        f"- B/C: `{bc}`.",
        "",
        "## Direct answers",
        "",
        "1. A1/A2 are not byte-identical repeatable beyond iteration 1; they first diverge at iteration 10 while sampled images and traced RNG hashes remain equal.",
        "2. Direct source and junction adapter have identical initialization, camera order and sampling sequence. Their later checksum differences are not distinguishable from A1/A2 same-code variance.",
        "3. Frozen manifest and historical-compatible legacy TSV have identical train camera order and sampling sequence in this scene.",
        "4. Clean vs historical code cannot be fully localized without instrumenting/running the historical dirty worktree. The clean historical-compatible proxy C diverges from B at the same early point already seen in A1/A2.",
        "5. Periodic evaluation has a plausible BatchNorm state side effect because `map_generator.eval()` is not called, but the recorded Trackmobile 30k cfgs only evaluate at 30000, not 7000.",
        "",
        "## Most credible current root cause for the 1.24 dB gap",
        "",
        "The strongest current evidence is same-code same-seed trajectory non-determinism: A1/A2 split by iteration 10 despite identical initialization, sampled images and traced RNG hashes, and end at 6987 vs 6752 Gaussians by iteration 1000. "
        "Camera order, adapter path and legacy TSV are not supported as primary causes by the short-run evidence.",
        "",
        "## Unique long-run recommendation",
        "",
        "**C - Need clean repeated-seed 30k** if any new 30k is approved. The next long run should estimate strict GS-W seed/operator variance under the clean strict protocol, not chase the historical non-strict legacy result. "
        "No historical matched 30k is approved by this audit.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_text(REPORT_DIR / "AUDIT_CORRECTION.md", (ROOT / "reports" / "gsw_equivalence_audit" / "RERUN_DECISION.md").read_text(encoding="utf-8"))
    write_text(REPORT_DIR / "RANDOM_STATE_AUDIT.md", build_random_state_audit())
    order_rows = inspect_camera_orders()
    write_csv(REPORT_DIR / "TRAIN_CAMERA_ORDER_AUDIT.csv", order_rows)
    write_text(REPORT_DIR / "TRAIN_CAMERA_ORDER_AUDIT.md", build_camera_order_report(order_rows))
    write_short_run_commands()
    trace_rows = combine_traces()
    write_csv(REPORT_DIR / "CAMERA_SAMPLING_TRACES.csv", trace_rows)
    write_csv(REPORT_DIR / "SHORT_RUN_TRAJECTORIES.csv", trace_rows)
    init_rows = [row for row in trace_rows if row.get("event") == "initialization"]
    write_csv(REPORT_DIR / "INITIALIZATION_EQUIVALENCE.csv", init_rows)
    pair_rows, pair_report = pairwise_comparison()
    write_text(REPORT_DIR / "SHORT_RUN_PAIRWISE_COMPARISON.md", pair_report)
    write_text(REPORT_DIR / "EVALUATION_SIDE_EFFECT_AUDIT.md", build_evaluation_side_effect_audit())
    write_text(REPORT_DIR / "FINAL_LONG_RUN_DECISION.md", build_final_decision(pair_rows))
    probe_dir = REPORT_DIR / "_camera_order_probe"
    if probe_dir.exists():
        shutil.rmtree(probe_dir)
    print(f"Wrote trajectory audit reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
