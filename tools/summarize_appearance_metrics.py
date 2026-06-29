import argparse
import csv
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_stats(path):
    image = tf.to_tensor(Image.open(path))[:3]
    return {
        "render_min": float(image.min().item()),
        "render_max": float(image.max().item()),
        "render_mean": float(image.mean().item()),
    }


def load_mapping(method_dir):
    mapping_path = method_dir / "appearance_mapping.csv"
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", newline="", encoding="utf-8") as handle:
        return {row["render_file"]: row for row in csv.DictReader(handle)}


def load_view_mapping(method_dir):
    mapping_path = method_dir / "render_view_mapping.csv"
    if not mapping_path.exists():
        return {}
    with mapping_path.open("r", newline="", encoding="utf-8") as handle:
        return {row["render_file"]: row for row in csv.DictReader(handle)}


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize rendered appearance-mode metrics.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--per-view-csv", type=Path, required=True)
    args = parser.parse_args()

    results = read_json(args.model_path / "results.json")
    per_view = read_json(args.model_path / "per_view.json")
    summary_rows = []
    per_view_rows = []

    for mode in args.modes:
        method = f"ours_{args.iteration}_{mode}"
        method_dir = args.model_path / "test" / method
        mapping = load_mapping(method_dir)
        view_mapping = load_view_mapping(method_dir)
        metrics = results[method]
        summary_rows.append({
            "scene": args.scene,
            "mode": mode,
            "method_dir": str(method_dir),
            "psnr": metrics["PSNR"],
            "ssim": metrics["SSIM"],
            "lpips": metrics["LPIPS"],
        })
        render_dir = method_dir / "renders"
        for render_path in sorted(render_dir.glob("*.png")):
            image_name = render_path.name
            map_row = mapping.get(image_name, {})
            view_row = view_mapping.get(image_name, {})
            stats = image_stats(render_path)
            per_view_rows.append({
                "scene": args.scene,
                "mode": mode,
                "render_file": image_name,
                "image_name": view_row.get("image_name", ""),
                "appearance_source": map_row.get("train_appearance_source", "self_test_rgb" if mode == "legacy_target_rgb" else "zero_point_features"),
                "pose_center_distance": map_row.get("pose_center_distance", ""),
                "psnr": per_view[method]["PSNR"][image_name],
                "ssim": per_view[method]["SSIM"][image_name],
                "lpips": per_view[method]["LPIPS"][image_name],
                **stats,
            })

    write_csv(args.summary_csv, summary_rows, ["scene", "mode", "method_dir", "psnr", "ssim", "lpips"])
    write_csv(
        args.per_view_csv,
        per_view_rows,
        [
            "scene",
            "mode",
            "render_file",
            "image_name",
            "appearance_source",
            "pose_center_distance",
            "psnr",
            "ssim",
            "lpips",
            "render_min",
            "render_max",
            "render_mean",
        ],
    )


if __name__ == "__main__":
    main()
