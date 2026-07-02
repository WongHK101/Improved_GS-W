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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def image_sha256(path: Path) -> str:
    with Image.open(path) as img:
        import numpy as np

        return hashlib.sha256(np.asarray(img.convert("RGB")).tobytes()).hexdigest()


def tensor_from_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as img:
        return tf.to_tensor(img.convert("RGB")).unsqueeze(0).to(device)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    mse = ((pred - gt) ** 2).view(pred.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_manifest_names(manifest: Path | None) -> list[str]:
    if not manifest:
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return list(payload.get("test_images", []))


def mapping_for_method(method_dir: Path, manifest_names: list[str]) -> dict[str, str]:
    mapping_csv = method_dir / "render_view_mapping.csv"
    if mapping_csv.exists():
        with mapping_csv.open("r", encoding="utf-8", newline="") as handle:
            return {row["render_file"]: row.get("image_name", "") for row in csv.DictReader(handle)}
    render_files = sorted(path.name for path in (method_dir / "renders").glob("*.png"))
    return {name: manifest_names[idx] if idx < len(manifest_names) else "" for idx, name in enumerate(render_files)}


def evaluate(label: str, group: str, scene: str, method_dir: Path, manifest: Path | None, lpips_model, device: torch.device) -> tuple[dict[str, object], list[dict[str, object]]]:
    render_dir = method_dir / "renders"
    gt_dir = method_dir / "gt"
    if not render_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError(f"Missing renders/gt under {method_dir}")
    render_files = sorted(path.name for path in render_dir.glob("*.png"))
    gt_files = sorted(path.name for path in gt_dir.glob("*.png"))
    if render_files != gt_files:
        raise ValueError(f"Render/GT file mismatch for {label}: {render_files} vs {gt_files}")
    manifest_names = load_manifest_names(manifest)
    mapping = mapping_for_method(method_dir, manifest_names)
    rows: list[dict[str, object]] = []
    for name in render_files:
        render_path = render_dir / name
        gt_path = gt_dir / name
        render = tensor_from_image(render_path, device)
        gt = tensor_from_image(gt_path, device)
        if tuple(render.shape) != tuple(gt.shape):
            raise ValueError(f"Shape mismatch for {label}/{name}: {tuple(render.shape)} vs {tuple(gt.shape)}")
        with torch.no_grad():
            lp = float(lpips_model(render, gt, normalize=True).item())
        ps = float(psnr(render, gt).mean().item())
        ss = float(kornia_ssim(render, gt, 3).mean().item())
        rows.append(
            {
                "scene": scene,
                "label": label,
                "group": group,
                "method_dir": str(method_dir),
                "render_file": name,
                "image_name": mapping.get(name, ""),
                "width": int(render.shape[-1]),
                "height": int(render.shape[-2]),
                "psnr": ps,
                "ssim": ss,
                "lpips": lp,
                "render_min": float(render.min().item()),
                "render_max": float(render.max().item()),
                "render_mean": float(render.mean().item()),
                "render_std": float(render.std().item()),
                "gt_min": float(gt.min().item()),
                "gt_max": float(gt.max().item()),
                "gt_mean": float(gt.mean().item()),
                "gt_rgb_sha256": image_sha256(gt_path),
                "render_rgb_sha256": image_sha256(render_path),
                "finite_metrics": math.isfinite(ps) and math.isfinite(ss) and math.isfinite(lp),
                "full_image": True,
                "lpips_net": "alex",
                "lpips_normalize": True,
            }
        )
    summary = {
        "scene": scene,
        "label": label,
        "group": group,
        "method_dir": str(method_dir),
        "render_count": len(render_files),
        "gt_count": len(gt_files),
        "test_image_names": ";".join(mapping.get(name, "") for name in render_files),
        "PSNR": sum(float(row["psnr"]) for row in rows) / len(rows),
        "SSIM": sum(float(row["ssim"]) for row in rows) / len(rows),
        "LPIPS": sum(float(row["lpips"]) for row in rows) / len(rows),
        "full_image": True,
        "lpips_net": "alex",
        "lpips_normalize": True,
    }
    return summary, rows


def parse_label(value: str) -> tuple[str, str, str, Path, Path | None]:
    parts = value.split("=", 4)
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError("label format: label=group=scene=method_dir[=manifest]")
    label, group, scene, method_dir = parts[:4]
    manifest = Path(parts[4]) if len(parts) == 5 and parts[4] else None
    return label, group, scene, Path(method_dir), manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict full-image RGB evaluator for baseline completion runs.")
    parser.add_argument("--label", action="append", type=parse_label, required=True, help="label=group=scene=method_dir[=manifest]")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--per-view-csv", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    summaries: list[dict[str, object]] = []
    per_view: list[dict[str, object]] = []
    for label, group, scene, method_dir, manifest in args.label:
        summary, rows = evaluate(label, group, scene, method_dir, manifest, lpips_model, device)
        summaries.append(summary)
        per_view.extend(rows)

    write_csv(args.results_csv, summaries)
    write_csv(args.per_view_csv, per_view)
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
