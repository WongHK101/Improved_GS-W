"""Audit GS-W appearance module eval state with an existing checkpoint.

The script renders the existing Trackmobile clean 30k checkpoint in memory. It
does not train, save checkpoints, or write full render directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENE = "self_Trackmobile_4650TM_Mobile_Railcar_Mover"
MODEL_PATH = Path(r"G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630") / SCENE
REPORT_DIR = ROOT / "reports" / "gsw_repeated_30k_baseline"
SPLIT_FILE = ROOT / "splits" / "TRACKMOBILE_SPLIT.json"
MODES = ["legacy_target_rgb", "strict_intrinsic", "strict_nearest_train"]


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


def digest_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def digest_module_state(module: torch.nn.Module) -> str:
    pieces = []
    for key, value in module.state_dict().items():
        pieces.append(f"{key}:{digest_tensor(value)}")
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()


def bn_snapshot(module: torch.nn.Module) -> dict[str, dict[str, Any]]:
    snapshot = {}
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            snapshot[name] = {
                "training": child.training,
                "running_mean_checksum": digest_tensor(child.running_mean),
                "running_var_checksum": digest_tensor(child.running_var),
                "num_batches_tracked": int(child.num_batches_tracked.item()),
            }
    return snapshot


def bn_changed(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    changed = []
    for name, row in before.items():
        other = after.get(name, {})
        for key in ["running_mean_checksum", "running_var_checksum", "num_batches_tracked"]:
            if row.get(key) != other.get(key):
                changed.append(f"{name}.{key}")
    return bool(changed), changed


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    data = tensor.detach().float()
    return {
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "mean": float(data.mean().item()),
        "std": float(data.std().item()),
    }


def psnr_value(rendered: torch.Tensor, gt: torch.Tensor) -> float:
    mse = float(((rendered - gt) ** 2).mean().item())
    return float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_value(rendered: torch.Tensor, gt: torch.Tensor) -> float | str:
    try:
        from kornia.metrics import ssim

        return float(ssim(rendered.unsqueeze(0), gt.unsqueeze(0), 3).mean().item())
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"ERROR {type(exc).__name__}: {exc}"


def lpips_value(rendered: torch.Tensor, gt: torch.Tensor, lpips_model: Any) -> float | str:
    if lpips_model is None:
        return ""
    try:
        return float(lpips_model(rendered.unsqueeze(0), gt.unsqueeze(0), normalize=True).mean().item())
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"ERROR {type(exc).__name__}: {exc}"


def read_cfg(path: Path) -> argparse.Namespace:
    text = path.read_text(encoding="utf-8")
    namespace = eval(text, {"__builtins__": {}}, {"Namespace": argparse.Namespace})  # noqa: S307
    return namespace


def build_args(model_path: Path = MODEL_PATH) -> argparse.Namespace:
    from arguments.args_init import argument_init

    args = read_cfg(model_path / "cfg_args")
    args.model_path = str(model_path)
    args.split_file = str(SPLIT_FILE)
    args.split_mode = "frozen_manifest"
    args.legacy_tsv_uid_source = getattr(args, "legacy_tsv_uid_source", "intrinsic")
    args.test_appearance_mode = "strict_intrinsic"
    args.data_perturb = getattr(args, "data_perturb", [])
    return argument_init(args)


def load_scene(args: argparse.Namespace, iteration: int):
    from arguments import ModelParams, PipelineParams
    from scene import GaussianModel, Scene

    model_parser = argparse.ArgumentParser()
    model_params = ModelParams(model_parser)
    pipeline_parser = argparse.ArgumentParser()
    pipeline_params = PipelineParams(pipeline_parser)

    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, args)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    return scene, gaussians, pipeline


@contextmanager
def fixed_eval(gaussians):
    gaussians.set_eval(True)
    try:
        yield
    finally:
        gaussians.set_eval(False)


@contextmanager
def legacy_bug_eval(gaussians):
    state = {
        "eval_mode": gaussians.eval_mode,
        "use_features_mask": getattr(gaussians, "use_features_mask", None),
    }
    if gaussians.use_color_net:
        state["color_net_training"] = gaussians.color_net.training
        state["color_net_use_drop_out"] = getattr(gaussians.color_net, "use_drop_out", None)
    if gaussians.use_kmap_pjmap or gaussians.use_okmap:
        state["map_generator_use_features_mask"] = getattr(gaussians.map_generator, "use_features_mask", None)
    gaussians.eval_mode = True
    if gaussians.use_color_net:
        gaussians.color_net.eval()
        if hasattr(gaussians.color_net, "use_drop_out"):
            gaussians.color_net.use_drop_out = False
    if gaussians.use_kmap_pjmap or gaussians.use_okmap:
        if hasattr(gaussians.map_generator, "use_features_mask"):
            gaussians.map_generator.use_features_mask = False
        gaussians.use_features_mask = False
    try:
        yield
    finally:
        gaussians.eval_mode = state["eval_mode"]
        if gaussians.use_color_net:
            gaussians.color_net.train(state.get("color_net_training", True))
            if state.get("color_net_use_drop_out") is not None:
                gaussians.color_net.use_drop_out = state["color_net_use_drop_out"]
        if gaussians.use_kmap_pjmap or gaussians.use_okmap:
            if state.get("use_features_mask") is not None:
                gaussians.use_features_mask = state["use_features_mask"]
            if state.get("map_generator_use_features_mask") is not None:
                gaussians.map_generator.use_features_mask = state["map_generator_use_features_mask"]


def render_one(view, gaussians, pipeline, background, mode: str, train_views):
    from gaussian_renderer import render
    from render import nearest_train_appearance_sources

    view.forbid_appearance_input = mode in ["strict_intrinsic", "strict_nearest_train"]
    source = None
    if mode == "strict_nearest_train":
        source = nearest_train_appearance_sources([view], train_views)[0][1]
    return render(
        view,
        gaussians,
        pipeline,
        background,
        appearance_mode=mode,
        appearance_source_camera=source,
    )["render"].detach().clamp(0.0, 1.0)


def render_sequence(eval_kind: str, mode: str, order: str, iteration: int, lpips_model: Any = None):
    args = build_args()
    scene, gaussians, pipeline = load_scene(args, iteration)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    train_views = scene.getTrainCameras()
    views = list(scene.getTestCameras())
    if order == "reverse":
        views = list(reversed(views))
    context = fixed_eval if eval_kind == "fixed" else legacy_bug_eval
    before_bn = bn_snapshot(gaussians.map_generator)
    before_state = {
        "model_eval_mode": gaussians.eval_mode,
        "map_generator_training": gaussians.map_generator.training,
        "color_net_training": gaussians.color_net.training,
        "use_features_mask": gaussians.use_features_mask,
        "map_generator_use_features_mask": gaussians.map_generator.use_features_mask,
        "color_net_use_drop_out": gaussians.color_net.use_drop_out,
        "map_generator_state_checksum": digest_module_state(gaussians.map_generator),
        "color_net_state_checksum": digest_module_state(gaussians.color_net),
    }
    outputs = {}
    rows = []
    start = time.time()
    with torch.no_grad():
        with context(gaussians):
            inside_state = {
                "model_eval_mode": gaussians.eval_mode,
                "map_generator_training": gaussians.map_generator.training,
                "color_net_training": gaussians.color_net.training,
                "use_features_mask": gaussians.use_features_mask,
                "map_generator_use_features_mask": gaussians.map_generator.use_features_mask,
                "color_net_use_drop_out": gaussians.color_net.use_drop_out,
            }
            for index, view in enumerate(views):
                rendered = render_one(view, gaussians, pipeline, background, mode, train_views)
                outputs[view.image_name] = rendered.detach().cpu()
                gt = view.original_image[:3].detach().cpu().clamp(0.0, 1.0)
                stats = tensor_stats(rendered.cpu())
                rows.append({
                    "eval_kind": eval_kind,
                    "mode": mode,
                    "order": order,
                    "render_index": index,
                    "image_name": view.image_name,
                    "psnr": psnr_value(rendered.cpu(), gt),
                    "ssim": ssim_value(rendered.cpu(), gt),
                    "lpips": lpips_value(rendered.cpu(), gt, lpips_model),
                    "render_min": stats["min"],
                    "render_max": stats["max"],
                    "render_mean": stats["mean"],
                    "render_std": stats["std"],
                })
    elapsed = time.time() - start
    after_bn = bn_snapshot(gaussians.map_generator)
    after_state = {
        "model_eval_mode": gaussians.eval_mode,
        "map_generator_training": gaussians.map_generator.training,
        "color_net_training": gaussians.color_net.training,
        "use_features_mask": gaussians.use_features_mask,
        "map_generator_use_features_mask": gaussians.map_generator.use_features_mask,
        "color_net_use_drop_out": gaussians.color_net.use_drop_out,
        "map_generator_state_checksum": digest_module_state(gaussians.map_generator),
        "color_net_state_checksum": digest_module_state(gaussians.color_net),
    }
    changed, changed_keys = bn_changed(before_bn, after_bn)
    return {
        "eval_kind": eval_kind,
        "mode": mode,
        "order": order,
        "outputs": outputs,
        "rows": rows,
        "before_bn": before_bn,
        "after_bn": after_bn,
        "bn_changed": changed,
        "bn_changed_keys": changed_keys,
        "before_state": before_state,
        "inside_state": inside_state,
        "after_state": after_state,
        "elapsed_sec": elapsed,
    }


def compare_outputs(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, float]:
    max_diff = 0.0
    mean_diffs = []
    for name, tensor in left.items():
        diff = (tensor - right[name]).abs()
        max_diff = max(max_diff, float(diff.max().item()))
        mean_diffs.append(float(diff.mean().item()))
    return {
        "max_abs_pixel_diff": max_diff,
        "mean_abs_pixel_diff": float(np.mean(mean_diffs)) if mean_diffs else 0.0,
    }


def row_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["image_name"]): row for row in rows}


def pre_post_rows(legacy: dict[str, Any], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    legacy_rows = row_lookup(legacy["rows"])
    fixed_rows = row_lookup(fixed["rows"])
    rows = []
    for image_name, fixed_output in fixed["outputs"].items():
        legacy_output = legacy["outputs"][image_name]
        diff = (legacy_output - fixed_output).abs()
        legacy_stats = tensor_stats(legacy_output)
        fixed_stats = tensor_stats(fixed_output)
        legacy_row = legacy_rows[image_name]
        fixed_row = fixed_rows[image_name]
        rows.append({
            "mode": fixed["mode"],
            "image_name": image_name,
            "legacy_bug_psnr": legacy_row["psnr"],
            "fixed_psnr": fixed_row["psnr"],
            "delta_psnr_fixed_minus_legacy": float(fixed_row["psnr"]) - float(legacy_row["psnr"]),
            "legacy_bug_ssim": legacy_row["ssim"],
            "fixed_ssim": fixed_row["ssim"],
            "delta_ssim_fixed_minus_legacy": float(fixed_row["ssim"]) - float(legacy_row["ssim"]),
            "legacy_bug_lpips": legacy_row["lpips"],
            "fixed_lpips": fixed_row["lpips"],
            "delta_lpips_fixed_minus_legacy": float(fixed_row["lpips"]) - float(legacy_row["lpips"]),
            "max_abs_pixel_diff": float(diff.max().item()),
            "mean_abs_pixel_diff": float(diff.mean().item()),
            "legacy_bug_render_mean": legacy_stats["mean"],
            "fixed_render_mean": fixed_stats["mean"],
            "delta_render_mean_fixed_minus_legacy": fixed_stats["mean"] - legacy_stats["mean"],
            "legacy_bug_render_std": legacy_stats["std"],
            "fixed_render_std": fixed_stats["std"],
            "delta_render_std_fixed_minus_legacy": fixed_stats["std"] - legacy_stats["std"],
            "legacy_bn_changed": legacy["bn_changed"],
            "fixed_bn_changed": fixed["bn_changed"],
        })
    return rows


def build_reports(
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    mode_rows: list[dict[str, Any]],
    prepost_rows: list[dict[str, Any]],
) -> None:
    bn_total = sum(1 for row in summary_rows if row["fixed_bn_changed"] == "False")
    lines = [
        "# Eval State Audit",
        "",
        f"- Model path: `{MODEL_PATH}`.",
        "- Confirmed bug before fix: legacy eval simulation leaves `map_generator.training=True` while rendering.",
        "- `map_generator` contains BatchNorm buffers; if it remains in train mode, rendering can update `running_mean`, `running_var`, and `num_batches_tracked`.",
        "",
        "## State summary",
        "",
        f"- Fixed eval mode preserved BN buffers for `{bn_total}/{len(summary_rows)}` mode/order checks.",
        "- Detailed rows are in `PRE_POST_FIX_RENDER_COMPARISON.csv`.",
        "",
    ]
    for row in summary_rows:
        lines.append(
            f"- {row['mode']} / {row['order']}: legacy_bn_changed={row['legacy_bn_changed']}, "
            f"fixed_bn_changed={row['fixed_bn_changed']}, fixed_map_generator_training_inside={row['fixed_map_generator_training_inside']}."
        )
    write_text(REPORT_DIR / "EVAL_STATE_AUDIT.md", "\n".join(lines) + "\n")

    test_lines = [
        "# Eval State Test Results",
        "",
        "| Mode | Legacy BN Changed | Fixed BN Changed | Fixed Order Max Diff | Fixed Repeat Max Diff | Pre/Post Max Diff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        test_lines.append(
            f"| {row['mode']} | {row['legacy_bn_changed']} | {row['fixed_bn_changed']} | "
            f"{row['fixed_order_max_abs_pixel_diff']} | {row['fixed_repeat_max_abs_pixel_diff']} | "
            f"{row['pre_post_max_abs_pixel_diff']} |"
        )
    test_lines.extend([
        "",
        "A passing fixed path requires `fixed_bn_changed=False`, order/repeat max pixel difference within tolerance, and `map_generator.training=False` inside eval.",
    ])
    write_text(REPORT_DIR / "EVAL_STATE_TEST_RESULTS.md", "\n".join(test_lines) + "\n")

    fix_lines = [
        "# Eval State Fix Audit",
        "",
        "- `GaussianModel.set_eval(True)` now calls both `color_net.eval()` and `map_generator.eval()`.",
        "- `GaussianModel.set_eval(False)` restores the module training states captured before entering eval mode.",
        "- Existing behavior that disables `features_mask` and `color_net.use_drop_out` during eval is preserved, but the original values are restored afterwards.",
        "- The implementation uses a stack so nested eval calls restore in order.",
        "",
        "## Render invariance checks",
        "",
    ]
    for row in comparison_rows:
        fix_lines.append(
            f"- {row['check']} / {row['mode']}: max_abs_pixel_diff={row['max_abs_pixel_diff']}, "
            f"mean_abs_pixel_diff={row['mean_abs_pixel_diff']}."
        )
    fix_lines.extend([
        "",
        "## Pre/post metric deltas",
        "",
    ])
    for row in prepost_rows:
        fix_lines.append(
            f"- {row['mode']} / {row['image_name']}: delta_psnr={row['delta_psnr_fixed_minus_legacy']}, "
            f"delta_ssim={row['delta_ssim_fixed_minus_legacy']}, delta_lpips={row['delta_lpips_fixed_minus_legacy']}, "
            f"max_abs_pixel_diff={row['max_abs_pixel_diff']}."
        )
    write_text(REPORT_DIR / "EVAL_STATE_FIX_AUDIT.md", "\n".join(fix_lines) + "\n")
    write_csv(REPORT_DIR / "PRE_POST_FIX_RENDER_COMPARISON.csv", prepost_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex").to("cpu").eval()
    except Exception:
        lpips_model = None

    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    prepost_rows_all: list[dict[str, Any]] = []
    fixed_results: dict[str, Any] = {}

    for mode in MODES:
        legacy = render_sequence("legacy_bug", mode, "forward", args.iteration, lpips_model)
        fixed = render_sequence("fixed", mode, "forward", args.iteration, lpips_model)
        fixed_reverse = render_sequence("fixed", mode, "reverse", args.iteration, lpips_model)
        fixed_repeat = render_sequence("fixed", mode, "forward", args.iteration, lpips_model)
        fixed_results[mode] = fixed
        prepost_rows_all.extend(pre_post_rows(legacy, fixed))
        prepost = compare_outputs(legacy["outputs"], fixed["outputs"])
        order_cmp = compare_outputs(fixed["outputs"], fixed_reverse["outputs"])
        repeat_cmp = compare_outputs(fixed["outputs"], fixed_repeat["outputs"])
        summary_rows.append({
            "mode": mode,
            "order": "forward",
            "legacy_bn_changed": str(legacy["bn_changed"]),
            "legacy_bn_changed_keys": ";".join(legacy["bn_changed_keys"]),
            "fixed_bn_changed": str(fixed["bn_changed"]),
            "fixed_bn_changed_keys": ";".join(fixed["bn_changed_keys"]),
            "legacy_map_generator_training_inside": str(legacy["inside_state"]["map_generator_training"]),
            "fixed_map_generator_training_inside": str(fixed["inside_state"]["map_generator_training"]),
            "fixed_color_net_training_inside": str(fixed["inside_state"]["color_net_training"]),
            "fixed_map_generator_training_after": str(fixed["after_state"]["map_generator_training"]),
            "fixed_color_net_training_after": str(fixed["after_state"]["color_net_training"]),
            "pre_post_max_abs_pixel_diff": prepost["max_abs_pixel_diff"],
            "pre_post_mean_abs_pixel_diff": prepost["mean_abs_pixel_diff"],
            "fixed_order_max_abs_pixel_diff": order_cmp["max_abs_pixel_diff"],
            "fixed_repeat_max_abs_pixel_diff": repeat_cmp["max_abs_pixel_diff"],
        })
        comparison_rows.extend([
            {"check": "pre_post_legacy_bug_vs_fixed", "mode": mode, **prepost},
            {"check": "fixed_forward_vs_reverse", "mode": mode, **order_cmp},
            {"check": "fixed_repeat_forward", "mode": mode, **repeat_cmp},
        ])
        for result in [legacy, fixed, fixed_reverse, fixed_repeat]:
            for row in result["rows"]:
                base = {
                    **row,
                    "bn_changed": result["bn_changed"],
                    "bn_changed_keys": ";".join(result["bn_changed_keys"]),
                    "map_generator_training_inside": result["inside_state"]["map_generator_training"],
                    "color_net_training_inside": result["inside_state"]["color_net_training"],
                    "map_generator_training_after": result["after_state"]["map_generator_training"],
                    "color_net_training_after": result["after_state"]["color_net_training"],
                }
                mode_rows.append(base)

    write_csv(REPORT_DIR / "EVAL_STATE_TEST_RESULTS.csv", summary_rows)
    write_csv(REPORT_DIR / "EVAL_STATE_RENDER_ROWS.csv", mode_rows)
    write_csv(REPORT_DIR / "EVAL_STATE_RENDER_INVARIANCE.csv", comparison_rows)
    build_reports(summary_rows, comparison_rows, mode_rows, prepost_rows_all)

    failures = []
    for row in summary_rows:
        if row["fixed_bn_changed"] != "False":
            failures.append(f"fixed BN changed for {row['mode']}")
        if row["fixed_map_generator_training_inside"] != "False":
            failures.append(f"map_generator not eval inside fixed eval for {row['mode']}")
        if float(row["fixed_order_max_abs_pixel_diff"]) > args.tolerance:
            failures.append(f"render order changed {row['mode']}: {row['fixed_order_max_abs_pixel_diff']}")
        if float(row["fixed_repeat_max_abs_pixel_diff"]) > args.tolerance:
            failures.append(f"repeat render changed {row['mode']}: {row['fixed_repeat_max_abs_pixel_diff']}")
    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"status": "ok", "report_dir": str(REPORT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
