import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def mutate_camera_image(camera, mode):
    mutated = copy.deepcopy(camera)
    image = mutated.original_image
    if mode == "zero":
        mutated.original_image = torch.zeros_like(image)
    elif mode == "noise":
        generator = torch.Generator(device=image.device)
        generator.manual_seed(20260630)
        mutated.original_image = torch.rand(image.shape, generator=generator, device=image.device, dtype=image.dtype)
    elif mode == "channel_swap":
        mutated.original_image = image[[2, 1, 0], :, :].clone()
    else:
        raise ValueError(mode)
    mutated.split_role = camera.split_role
    return mutated


def render_once(test_camera, gaussians, pipeline, background, mode, train_views):
    from gaussian_renderer import render
    from render import nearest_train_appearance_sources

    test_camera.forbid_appearance_input = mode in ["strict_intrinsic", "strict_nearest_train"]
    appearance_source = None
    if mode == "strict_nearest_train":
        appearance_source = nearest_train_appearance_sources([test_camera], train_views)[0][1]
    return render(
        test_camera,
        gaussians,
        pipeline,
        background,
        appearance_mode=mode,
        appearance_source_camera=appearance_source,
    )["render"].detach()


def max_abs_diff(a, b):
    return float((a - b).abs().max().item())


def mean_abs_diff(a, b):
    return float((a - b).abs().mean().item())


def main():
    from arguments import ModelParams, PipelineParams
    from arguments import args_init

    parser = argparse.ArgumentParser(description="Strict appearance invariance test")
    model = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=10, type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--mapping-csv", type=Path)
    parser.add_argument("--tolerance", default=1e-7, type=float)
    parser.add_argument("--legacy-min-diff", default=1e-6, type=float)
    parser.add_argument("--data_perturb", nargs="+", default=[])
    args = parser.parse_args()
    args = args_init.argument_init(args)

    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from render import nearest_train_appearance_sources

    dataset = model.extract(args)
    pipeline = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, args)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    gaussians.set_eval(True)

    train_views = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    results = {}
    rows = []
    errors = []
    legacy_unchanged_views = []
    for test_camera in test_cameras:
        variants = {
            "zero": mutate_camera_image(test_camera, "zero"),
            "noise": mutate_camera_image(test_camera, "noise"),
            "channel_swap": mutate_camera_image(test_camera, "channel_swap"),
        }
        view_results = {}
        for mode in ["strict_intrinsic", "strict_nearest_train", "legacy_target_rgb"]:
            base = render_once(test_camera, gaussians, pipeline, background, mode, train_views)
            mode_results = {}
            for variant_name, variant_camera in variants.items():
                rendered = render_once(variant_camera, gaussians, pipeline, background, mode, train_views)
                max_diff = max_abs_diff(base, rendered)
                mean_diff = mean_abs_diff(base, rendered)
                mode_results[variant_name] = {
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                }
                rows.append({
                    "test_image": test_camera.image_name,
                    "mode": mode,
                    "perturbation": variant_name,
                    "max_abs_diff": max_diff,
                    "mean_abs_diff": mean_diff,
                })
            view_results[mode] = mode_results
        results[test_camera.image_name] = view_results

    for test_image, view_results in results.items():
        for mode in ["strict_intrinsic", "strict_nearest_train"]:
            for variant_name, diffs in view_results[mode].items():
                if diffs["max_abs_diff"] > args.tolerance:
                    errors.append(
                        f"{mode} changed on {test_image} under {variant_name}: "
                        f"{diffs['max_abs_diff']}"
                    )
        view_legacy_changed = any(
            diffs["max_abs_diff"] > args.legacy_min_diff
            for diffs in view_results["legacy_target_rgb"].values()
        )
        if not view_legacy_changed:
            legacy_unchanged_views.append(test_image)

    legacy_changed = len(legacy_unchanged_views) < len(test_cameras)
    if not legacy_changed:
        errors.append(f"legacy_target_rgb did not change above {args.legacy_min_diff} for any held-out view")
    if legacy_unchanged_views:
        print(
            "WARNING: legacy_target_rgb did not change above "
            f"{args.legacy_min_diff} for views: {legacy_unchanged_views}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["test_image", "mode", "perturbation", "max_abs_diff", "mean_abs_diff"],
            )
            writer.writeheader()
            writer.writerows(rows)

    mapping = nearest_train_appearance_sources(scene.getTestCameras(), train_views)
    mapping_rows = [
        {
            "test_image": test_view.image_name,
            "train_appearance_source": train_view.image_name,
            "pose_center_distance": distance,
        }
        for test_view, train_view, distance in mapping
    ]
    if args.mapping_csv:
        args.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.mapping_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["test_image", "train_appearance_source", "pose_center_distance"])
            writer.writeheader()
            writer.writerows(mapping_rows)

    report = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "iteration": args.iteration,
        "split_file": args.split_file,
        "test_cameras": [camera.image_name for camera in test_cameras],
        "results": results,
        "strict_tolerance": args.tolerance,
        "legacy_min_diff": args.legacy_min_diff,
        "legacy_changed": legacy_changed,
        "legacy_unchanged_views": legacy_unchanged_views,
        "nearest_train_mapping": mapping_rows,
        "errors": errors,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    print(text)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
