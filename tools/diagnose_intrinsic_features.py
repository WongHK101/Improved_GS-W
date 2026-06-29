import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def tensor_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def image_stats(image):
    return {
        "min": tensor_scalar(image.min()),
        "max": tensor_scalar(image.max()),
        "mean": tensor_scalar(image.mean()),
    }


def feature_stats(image_name, point_features, near_zero_eps):
    point_features = point_features.detach()
    l2 = torch.linalg.vector_norm(point_features, dim=1)
    near_zero_elements = (point_features.abs() <= near_zero_eps).float().mean()
    near_zero_vectors = (l2 <= near_zero_eps).float().mean()
    return {
        "image": image_name,
        "num_gaussians": int(point_features.shape[0]),
        "feature_dim": int(point_features.shape[1]) if point_features.ndim == 2 else 0,
        "l2_mean": tensor_scalar(l2.mean()),
        "l2_median": tensor_scalar(l2.median()),
        "l2_std": tensor_scalar(l2.std(unbiased=False)),
        "element_mean": tensor_scalar(point_features.mean()),
        "element_std": tensor_scalar(point_features.std(unbiased=False)),
        "element_min": tensor_scalar(point_features.min()),
        "element_max": tensor_scalar(point_features.max()),
        "near_zero_element_ratio": tensor_scalar(near_zero_elements),
        "near_zero_vector_ratio": tensor_scalar(near_zero_vectors),
    }


def mean_dict(rows, keys):
    if not rows:
        return {key: None for key in keys}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    from arguments import ModelParams, PipelineParams
    from arguments import args_init
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    from utils.image_utils import psnr
    from kornia.metrics import ssim as kornia_ssim
    import lpips

    parser = argparse.ArgumentParser(description="Diagnose GS-W intrinsic zero-feature behavior.")
    model = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--train-diagnostics-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--max-train-views", default=0, type=int)
    parser.add_argument("--near-zero-eps", default=1e-6, type=float)
    parser.add_argument("--data_perturb", nargs="+", default=[])
    args = parser.parse_args()
    args = args_init.argument_init(args)

    dataset = model.extract(args)
    pipeline = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, args)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.set_eval(True)

    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    train_views = scene.getTrainCameras()
    if args.max_train_views > 0:
        train_views = train_views[:args.max_train_views]

    lpips_alex = lpips.LPIPS(net="alex").to("cuda:0")
    feature_rows = []
    diagnostic_rows = []
    started_at = time.time()

    with torch.no_grad():
        for view in train_views:
            view.split_role = "train"
            normal_render = torch.clamp(
                render(view, gaussians, pipeline, background, appearance_mode="legacy_target_rgb")["render"],
                0.0,
                1.0,
            )
            point_features = gaussians._point_features.detach()
            feature_rows.append(feature_stats(view.image_name, point_features, args.near_zero_eps))

            intrinsic_render = torch.clamp(
                render(view, gaussians, pipeline, background, appearance_mode="strict_intrinsic")["render"],
                0.0,
                1.0,
            )
            gt = view.original_image[0:3, :, :].cuda()
            dynamic_abs = (normal_render - intrinsic_render).abs()
            diagnostic_rows.append({
                "image": view.image_name,
                "normal_psnr": tensor_scalar(psnr(normal_render.unsqueeze(0), gt.unsqueeze(0))),
                "normal_ssim": tensor_scalar(kornia_ssim(normal_render.unsqueeze(0), gt.unsqueeze(0), 3).mean()),
                "normal_lpips": tensor_scalar(lpips_alex(normal_render.unsqueeze(0), gt.unsqueeze(0), normalize=True)),
                "intrinsic_psnr": tensor_scalar(psnr(intrinsic_render.unsqueeze(0), gt.unsqueeze(0))),
                "intrinsic_ssim": tensor_scalar(kornia_ssim(intrinsic_render.unsqueeze(0), gt.unsqueeze(0), 3).mean()),
                "intrinsic_lpips": tensor_scalar(lpips_alex(intrinsic_render.unsqueeze(0), gt.unsqueeze(0), normalize=True)),
                "normal_intrinsic_mae": tensor_scalar(dynamic_abs.mean()),
                "normal_intrinsic_max_abs": tensor_scalar(dynamic_abs.max()),
                "normal_min": image_stats(normal_render)["min"],
                "normal_max": image_stats(normal_render)["max"],
                "normal_mean": image_stats(normal_render)["mean"],
                "intrinsic_min": image_stats(intrinsic_render)["min"],
                "intrinsic_max": image_stats(intrinsic_render)["max"],
                "intrinsic_mean": image_stats(intrinsic_render)["mean"],
            })

    feature_fields = [
        "image",
        "num_gaussians",
        "feature_dim",
        "l2_mean",
        "l2_median",
        "l2_std",
        "element_mean",
        "element_std",
        "element_min",
        "element_max",
        "near_zero_element_ratio",
        "near_zero_vector_ratio",
    ]
    diagnostic_fields = [
        "image",
        "normal_psnr",
        "normal_ssim",
        "normal_lpips",
        "intrinsic_psnr",
        "intrinsic_ssim",
        "intrinsic_lpips",
        "normal_intrinsic_mae",
        "normal_intrinsic_max_abs",
        "normal_min",
        "normal_max",
        "normal_mean",
        "intrinsic_min",
        "intrinsic_max",
        "intrinsic_mean",
    ]
    write_csv(args.feature_csv, feature_rows, feature_fields)
    write_csv(args.train_diagnostics_csv, diagnostic_rows, diagnostic_fields)

    feature_summary_keys = [
        "l2_mean",
        "l2_median",
        "l2_std",
        "element_mean",
        "element_std",
        "near_zero_element_ratio",
        "near_zero_vector_ratio",
    ]
    diagnostic_summary_keys = [
        "normal_psnr",
        "normal_ssim",
        "normal_lpips",
        "intrinsic_psnr",
        "intrinsic_ssim",
        "intrinsic_lpips",
        "normal_intrinsic_mae",
        "normal_intrinsic_max_abs",
    ]
    feature_summary = mean_dict(feature_rows, feature_summary_keys)
    diagnostic_summary = mean_dict(diagnostic_rows, diagnostic_summary_keys)
    zero_l2_z_score = None
    if feature_summary["l2_std"] and feature_summary["l2_std"] > 0:
        zero_l2_z_score = (0.0 - feature_summary["l2_mean"]) / feature_summary["l2_std"]

    summary = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "iteration": scene.loaded_iter,
        "split_file": args.split_file,
        "num_train_views_evaluated": len(train_views),
        "near_zero_eps": args.near_zero_eps,
        "zero_feature_l2_norm": 0.0,
        "zero_l2_z_score_vs_train_view_mean_l2": zero_l2_z_score,
        "feature_summary_mean_across_views": feature_summary,
        "train_render_summary_mean_across_views": diagnostic_summary,
        "elapsed_seconds": time.time() - started_at,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
