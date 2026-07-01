from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import deque
from pathlib import Path


IMPROVED_ROOT = Path(r"G:\wl3dgs\Improved_GS-W")
sys.path.insert(0, str(IMPROVED_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as gsw_train  # noqa: E402
from arguments import ModelParams, OptimizationParams, PipelineParams, args_init  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


TRACE_ITERATIONS = {1, 100, 499, 500, 501, 1000, 2000, 3000, 4000, 5000}
TRACE_ITERATIONS.update(range(1000, 5001, 100))
ITER_TIMES_MS: deque[float] = deque(maxlen=100)
DENSIFY_EVENTS: dict[int, dict[str, int]] = {}
ITER_START_WALL = 0.0
GPU_TOTAL_MB = 0


def gpu_total_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    props = torch.cuda.get_device_properties(0)
    return int(props.total_memory / (1024 * 1024))


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def patch_training_trace(trace_csv: Path) -> None:
    gsw_train.TRACE_ITERATIONS = TRACE_ITERATIONS
    original_maybe_record = gsw_train.TrainingTrace.maybe_record_iteration
    original_init = gsw_train.TrainingTrace.maybe_record_initialization
    original_close = gsw_train.TrainingTrace.close

    def maybe_record_initialization(self, scene, gaussians):
        global ITER_START_WALL, GPU_TOTAL_MB
        ITER_START_WALL = time.time()
        GPU_TOTAL_MB = gpu_total_mb()
        torch.cuda.reset_peak_memory_stats()
        return original_init(self, scene, gaussians)

    def maybe_record_iteration(
        self,
        iteration,
        gaussians,
        loss,
        Ll1,
        dssim_loss,
        lpips_loss_value,
        box_coord_loss_value,
        visibility_filter,
        elapsed_ms,
    ):
        ITER_TIMES_MS.append(float(elapsed_ms))
        result = original_maybe_record(
            self,
            iteration,
            gaussians,
            loss,
            Ll1,
            dssim_loss,
            lpips_loss_value,
            box_coord_loss_value,
            visibility_filter,
            elapsed_ms,
        )
        if self.enabled and iteration in TRACE_ITERATIONS and self.rows:
            row = self.rows[-1]
            row["elapsed_wall_sec"] = float(time.time() - ITER_START_WALL)
            row["iter_time_ms"] = float(elapsed_ms)
            row["iter_time_rolling_median_ms"] = float(np.median(list(ITER_TIMES_MS))) if ITER_TIMES_MS else ""
            row["peak_allocated_mb"] = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else ""
            row["peak_reserved_mb"] = float(torch.cuda.max_memory_reserved() / (1024 * 1024)) if torch.cuda.is_available() else ""
            row["gpu_total_memory_mb"] = GPU_TOTAL_MB
            row["loss_finite"] = bool(finite(loss.detach().item()))
            event = DENSIFY_EVENTS.get(int(iteration), {"clone_count": 0, "split_count": 0, "prune_count": 0})
            row["densification_clone_count"] = int(event.get("clone_count", 0))
            row["densification_split_count"] = int(event.get("split_count", 0))
            row["densification_prune_count"] = int(event.get("prune_count", 0))
        return result

    def close(self):
        if not self.enabled or not self.path:
            return original_close(self)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not self.rows:
            return
        fieldnames = []
        for row in self.rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(self.path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)

    gsw_train.TrainingTrace.maybe_record_initialization = maybe_record_initialization
    gsw_train.TrainingTrace.maybe_record_iteration = maybe_record_iteration
    gsw_train.TrainingTrace.close = close


def patch_densification_stats() -> None:
    original_clone = GaussianModel.densify_and_clone
    original_split = GaussianModel.densify_and_split
    original_prune = GaussianModel.prune_points

    def current_iter(self) -> int:
        return int(getattr(self, "_codex_current_densify_iteration", -1))

    def event_for(iteration: int) -> dict[str, int]:
        return DENSIFY_EVENTS.setdefault(iteration, {"clone_count": 0, "split_count": 0, "prune_count": 0})

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        before = int(self.get_xyz.shape[0])
        result = original_clone(self, grads, grad_threshold, scene_extent)
        after = int(self.get_xyz.shape[0])
        event_for(current_iter(self))["clone_count"] += max(0, after - before)
        return result

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        before = int(self.get_xyz.shape[0])
        result = original_split(self, grads, grad_threshold, scene_extent, N=N)
        after = int(self.get_xyz.shape[0])
        # Net increase after split excludes the original selected points pruned inside split.
        event_for(current_iter(self))["split_count"] += max(0, after - before)
        return result

    def prune_points(self, mask):
        try:
            pruned = int(mask.sum().item())
        except Exception:
            pruned = 0
        event_for(current_iter(self))["prune_count"] += max(0, pruned)
        return original_prune(self, mask)

    original_densify_and_prune = GaussianModel.densify_and_prune

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # train.py sets no iteration argument; infer from latest trace by a global marker patched below.
        self._codex_current_densify_iteration = getattr(gsw_train, "_codex_current_iteration", -1)
        return original_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size)

    GaussianModel.densify_and_clone = densify_and_clone
    GaussianModel.densify_and_split = densify_and_split
    GaussianModel.prune_points = prune_points
    GaussianModel.densify_and_prune = densify_and_prune


def patch_iteration_marker() -> None:
    # Lightweight marker via GaussianModel.update_learning_rate, which train.py calls once per iteration.
    original_update_lr = GaussianModel.update_learning_rate

    def update_learning_rate(self, iteration, warm_up_iter=0):
        gsw_train._codex_current_iteration = int(iteration)
        return original_update_lr(self, iteration, warm_up_iter)

    GaussianModel.update_learning_rate = update_learning_rate


def checkpoint_loadable(model_path: Path, iteration: int, args) -> tuple[bool, str]:
    ckpt = model_path / "ckpts_point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ckpt.exists():
        return False, f"missing {ckpt}"
    try:
        model = GaussianModel(args.sh_degree, args)
        model.load_ckpt_ply(str(ckpt))
        return True, ""
    except Exception as exc:
        return False, repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="GS-W pilot trainer with diagnostics only")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000, 5000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_after_train", action="store_true", default=True)
    parser.add_argument("--metrics_after_train", action="store_true", default=True)
    parser.add_argument("--eval_half_after_train", action="store_true", default=False)
    parser.add_argument("--data_perturb", nargs="+", type=str, default=[])
    parser.add_argument("--trace_training_state", action="store_true", default=True)
    parser.add_argument("--trace_output", type=str, required=True)
    parser.add_argument("--checkpoint_audit_output", type=str, required=True)
    parser.add_argument("--disable_render_after_train", action="store_true", default=False)
    parser.add_argument("--disable_metrics_after_train", action="store_true", default=False)
    parser.add_argument("--disable_save_iterations", action="store_true", default=False)
    parser.add_argument("--disable_train_temp_images", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.disable_render_after_train:
        args.render_after_train = False
    if args.disable_metrics_after_train:
        args.metrics_after_train = False
    args.save_iterations.append(args.iterations)
    args = args_init.argument_init(args)

    patch_training_trace(Path(args.trace_output))
    patch_iteration_marker()
    patch_densification_stats()

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    op.position_lr_max_steps = op.iterations

    gsw_train.training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from, args)

    ok, error = checkpoint_loadable(Path(args.model_path), args.iterations, args)
    with open(args.checkpoint_audit_output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_path", "iteration", "checkpoint_loadable", "error"], lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "model_path": args.model_path,
                "iteration": args.iterations,
                "checkpoint_loadable": ok,
                "error": error,
            }
        )
    if not ok:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
