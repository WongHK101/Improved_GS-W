from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from kornia.metrics import ssim as kornia_ssim
import lpips

from official_control_common import (
    EXPECTED_TEST_IMAGES,
    GSW_RUN_ROOT,
    REPORT_DIR,
    RUN_ROOT,
    SCENE_NAME,
    write_csv,
    write_text,
)


def image_sha256(path: Path) -> str:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        import numpy as np

        return hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()


def tensor_from_image(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        return tf.to_tensor(img.convert("RGB")).unsqueeze(0).cuda()


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((pred - gt) ** 2).view(pred.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def ssim_value(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(kornia_ssim(pred, gt, 3).mean().item())


def mapping_for_method(method_dir: Path) -> dict[str, str]:
    mapping_csv = method_dir / "render_view_mapping.csv"
    if mapping_csv.exists():
        with mapping_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {row["render_file"]: row.get("image_name", "") for row in rows}
    render_files = sorted(path.name for path in (method_dir / "renders").glob("*.png"))
    return {name: EXPECTED_TEST_IMAGES[idx] if idx < len(EXPECTED_TEST_IMAGES) else "" for idx, name in enumerate(render_files)}


def evaluate_method(label: str, group: str, method_dir: Path, lpips_model) -> tuple[dict[str, object], list[dict[str, object]]]:
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not render_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing renders/gt under {method_dir}")
    render_files = sorted(path.name for path in render_dir.glob("*.png"))
    gt_files = sorted(path.name for path in gt_dir.glob("*.png"))
    mapping = mapping_for_method(method_dir)
    if render_files != gt_files:
        raise ValueError(f"Render/GT file mismatch for {label}: {render_files} vs {gt_files}")
    rows = []
    for name in render_files:
        render_path = render_dir / name
        gt_path = gt_dir / name
        render = tensor_from_image(render_path)
        gt = tensor_from_image(gt_path)
        if tuple(render.shape) != tuple(gt.shape):
            raise ValueError(f"Shape mismatch for {label}/{name}: {tuple(render.shape)} vs {tuple(gt.shape)}")
        with torch.no_grad():
            lp = float(lpips_model(render, gt, normalize=True).item())
        row = {
            "group": group,
            "label": label,
            "method_dir": str(method_dir),
            "render_file": name,
            "image_name": mapping.get(name, ""),
            "width": int(render.shape[-1]),
            "height": int(render.shape[-2]),
            "psnr": float(psnr(render, gt).mean().item()),
            "ssim": ssim_value(render, gt),
            "lpips": lp,
            "render_min": float(render.min().item()),
            "render_max": float(render.max().item()),
            "render_mean": float(render.mean().item()),
            "gt_min": float(gt.min().item()),
            "gt_max": float(gt.max().item()),
            "gt_mean": float(gt.mean().item()),
            "gt_rgb_sha256": image_sha256(gt_path),
            "render_rgb_sha256": image_sha256(render_path),
            "lpips_net": "alex",
            "lpips_normalize": True,
            "full_image": True,
        }
        rows.append(row)
    summary = {
        "group": group,
        "label": label,
        "method_dir": str(method_dir),
        "render_count": len(render_files),
        "gt_count": len(gt_files),
        "test_image_names": ";".join(mapping.get(name, "") for name in render_files),
        "psnr": sum(float(row["psnr"]) for row in rows) / len(rows),
        "ssim": sum(float(row["ssim"]) for row in rows) / len(rows),
        "lpips": sum(float(row["lpips"]) for row in rows) / len(rows),
        "lpips_net": "alex",
        "lpips_normalize": True,
        "full_image": True,
    }
    return summary, rows


def default_methods(include_official: bool, include_gsw: bool) -> list[tuple[str, str, Path]]:
    methods: list[tuple[str, str, Path]] = []
    if include_official:
        for run in ["O1", "O2", "O3"]:
            model_path = RUN_ROOT / run / SCENE_NAME
            method_dir = model_path / "test" / "ours_30000"
            if method_dir.exists():
                methods.append((run, "official_3dgs", method_dir))
    if include_gsw:
        for run in ["R1", "R2", "R3"]:
            method_dir = GSW_RUN_ROOT / run / SCENE_NAME / "test" / "ours_30000_strict_intrinsic"
            if method_dir.exists():
                methods.append((run, "gsw_strict_intrinsic", method_dir))
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified official-vs-GS-W full-image evaluation")
    parser.add_argument("--official", action="store_true", help="Include official O1/O2/O3 methods if present")
    parser.add_argument("--gsw", action="store_true", help="Include GS-W R1/R2/R3 strict_intrinsic methods if present")
    parser.add_argument("--label", action="append", default=[], help="Custom label=group=method_dir entry")
    parser.add_argument("--results-csv", type=Path, default=REPORT_DIR / "UNIFIED_OFFICIAL_GSW_RESULTS.csv")
    parser.add_argument("--per-view-csv", type=Path, default=REPORT_DIR / "UNIFIED_OFFICIAL_GSW_PER_VIEW.csv")
    args = parser.parse_args()

    include_official = args.official or not args.label
    include_gsw = args.gsw or not args.label
    methods = default_methods(include_official, include_gsw)
    for item in args.label:
        label, group, method_dir = item.split("=", 2)
        methods.append((label, group, Path(method_dir)))
    if not methods:
        raise RuntimeError("No methods found to evaluate.")

    torch.cuda.set_device(torch.device("cuda:0"))
    lpips_model = lpips.LPIPS(net="alex").to("cuda:0").eval()
    summaries = []
    per_view = []
    for label, group, method_dir in methods:
        summary, rows = evaluate_method(label, group, method_dir, lpips_model)
        summaries.append(summary)
        per_view.extend(rows)

    write_csv(
        args.results_csv,
        summaries,
        [
            "group",
            "label",
            "method_dir",
            "render_count",
            "gt_count",
            "test_image_names",
            "psnr",
            "ssim",
            "lpips",
            "lpips_net",
            "lpips_normalize",
            "full_image",
        ],
    )
    write_csv(
        args.per_view_csv,
        per_view,
        [
            "group",
            "label",
            "method_dir",
            "render_file",
            "image_name",
            "width",
            "height",
            "psnr",
            "ssim",
            "lpips",
            "render_min",
            "render_max",
            "render_mean",
            "gt_min",
            "gt_max",
            "gt_mean",
            "gt_rgb_sha256",
            "render_rgb_sha256",
            "lpips_net",
            "lpips_normalize",
            "full_image",
        ],
    )
    md = [
        "# UNIFIED_EVALUATION",
        "",
        "- Metrics: PSNR, kornia SSIM with window size 3, LPIPS AlexNet.",
        "- LPIPS call: `lpips.LPIPS(net='alex')(render, gt, normalize=True)`.",
        "- Image protocol: full image, no crop, no resize, RGB tensors in `[0, 1]`.",
        f"- Results CSV: `{args.results_csv}`",
        f"- Per-view CSV: `{args.per_view_csv}`",
        "",
        "| group | label | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        md.append(
            f"| {row['group']} | {row['label']} | {float(row['psnr']):.6f} | {float(row['ssim']):.6f} | {float(row['lpips']):.6f} |"
        )
    write_text(REPORT_DIR / "UNIFIED_EVALUATION.md", "\n".join(md) + "\n")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

