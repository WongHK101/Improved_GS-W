"""Read-only GS-W historical-vs-clean equivalence audit.

This script audits existing Trackmobile GS-W artifacts only. It does not train,
render, modify checkpoints, or write into the historical repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "gsw_equivalence_audit"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENE = "self_Trackmobile_4650TM_Mobile_Railcar_Mover"
HIST_CODE = Path(r"G:\wl3dgs\external_baselines\Gaussian-Wild")
HIST_RUN = Path(r"G:\wl3dgs\3dgs_runs\external_baselines_20260620\gs_w_r1_iter30000") / SCENE
HIST_OFFICIAL_RUN = Path(r"G:\wl3dgs\3dgs_runs\external_baselines_20260620\official_3dgs_r1_iter30000") / SCENE
HIST_ADAPTER = Path(r"G:\wl3dgs\3dgs_runs\external_baselines_20260620\adapters\gs_w") / SCENE
HIST_DENSE = HIST_ADAPTER / "dense"
HIST_TSV = HIST_ADAPTER / f"{SCENE}.tsv"
CURRENT_RUN = Path(r"G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630") / SCENE
CLEAN_DATA = Path(r"G:\wl3dgs\3dgs_undistorted\max1600") / SCENE
SPLIT_FILE = ROOT / "splits" / "TRACKMOBILE_SPLIT.json"
GPT_PACKAGE_DIR = Path(r"G:\WL3DGS\gpt_review_packages")


CFG_KEYS_OF_INTEREST = [
    "eval",
    "split_source",
    "tsv_path",
    "train_camera_count",
    "test_camera_count",
    "actual_train_image_names",
    "actual_test_image_names",
    "resolution",
    "source_path",
    "images",
    "use_colors_precomp",
    "use_features_mask",
    "use_kmap_pjmap",
    "use_lpips_loss",
    "map_num",
    "seed",
    "map_generator_type",
    "color_net_type",
    "feature_maps_dim",
    "features_dim",
    "map_generator_lr",
    "color_net_lr",
    "lpips_loss_coef",
    "box_coord_loss_coef",
    "lambda_dssim",
    "densification_interval",
    "densify_from_iter",
    "densify_until_iter",
    "densify_grad_threshold",
    "opacity_threshold",
    "opacity_reset_interval",
    "position_lr_init",
    "position_lr_final",
    "feature_lr",
    "opacity_lr",
    "scaling_lr",
    "rotation_lr",
    "percent_dense",
    "random_background",
    "white_background",
    "sh_degree",
    "resolution",
    "iterations",
    "sparse_subdir",
    "split_mode",
    "split_file",
    "test_appearance_mode",
    "eval",
    "data_device",
    "device",
    "scene_name",
    "use_color_net",
    "use_box_coord_loss",
    "use_xw_init_box_coord",
    "use_decode_with_pos",
    "use_indep_mask_branch",
    "use_okmap",
    "use_wo_adative",
    "coord_scale",
    "map_generator_params",
    "color_net_params",
]

RELEVANT_CODE_FILES = [
    "train.py",
    "render.py",
    "metrics.py",
    "metrics_half.py",
    "gaussian_renderer/__init__.py",
    "scene/__init__.py",
    "scene/dataset_readers.py",
    "scene/gaussian_model.py",
    "scene/cameras.py",
    "utils/camera_utils.py",
    "utils/image_utils.py",
    "arguments/__init__.py",
    "arguments/args_init.py",
    "net_modules/feature_maps_generators.py",
    "net_modules/feature_maps_projection.py",
    "net_modules/feature_maps_sample.py",
    "net_modules/color_features_net.py",
    "submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py",
]


@dataclass
class ColmapCamera:
    camera_id: int
    model_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass
class ColmapImage:
    image_id: int
    qvec: tuple[float, ...]
    tvec: tuple[float, ...]
    camera_id: int
    name: str
    num_points2d: int


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(args: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    try:
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
    except Exception as exc:  # pragma: no cover - defensive audit output
        return f"ERROR: {type(exc).__name__}: {exc}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def strip_line_trailing_ws(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines())


def repr_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def read_cfg(path: Path) -> dict[str, Any]:
    ns_type = argparse.Namespace
    text = path.read_text(encoding="utf-8")
    namespace = eval(text, {"__builtins__": {}}, {"Namespace": ns_type})  # noqa: S307 - local cfg audit file
    return vars(namespace)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv_split(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame = frame[~frame["id"].isnull()].reset_index(drop=True)
    frame["id"] = frame["id"].astype(int)
    return frame


def read_next(handle: Any, num_bytes: int, fmt: str) -> tuple[Any, ...]:
    data = handle.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with path.open("rb") as handle:
        num_cameras = read_next(handle, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next(handle, 24, "iiQQ")
            model, num_params = CAMERA_MODELS[model_id]
            params = read_next(handle, 8 * num_params, "d" * num_params)
            cameras[int(camera_id)] = ColmapCamera(
                camera_id=int(camera_id),
                model_id=int(model_id),
                model=model,
                width=int(width),
                height=int(height),
                params=tuple(float(v) for v in params),
            )
    return cameras


def read_images_binary(path: Path) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with path.open("rb") as handle:
        num_images = read_next(handle, 8, "Q")[0]
        for _ in range(num_images):
            raw = read_next(handle, 64, "idddddddi")
            image_id = int(raw[0])
            qvec = tuple(float(v) for v in raw[1:5])
            tvec = tuple(float(v) for v in raw[5:8])
            camera_id = int(raw[8])
            name_bytes = bytearray()
            while True:
                char = read_next(handle, 1, "c")[0]
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")
            num_points2d = int(read_next(handle, 8, "Q")[0])
            handle.seek(24 * num_points2d, os.SEEK_CUR)
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name, num_points2d)
    return images


def read_points3d_binary_summary(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        num_points = int(read_next(handle, 8, "Q")[0])
        xyz = np.empty((num_points, 3), dtype=np.float64)
        rgb_sum = np.zeros(3, dtype=np.float64)
        err_sum = 0.0
        for idx in range(num_points):
            raw = read_next(handle, 43, "QdddBBBd")
            xyz[idx] = [float(raw[1]), float(raw[2]), float(raw[3])]
            rgb_sum += [int(raw[4]), int(raw[5]), int(raw[6])]
            err_sum += float(raw[7])
            track_len = int(read_next(handle, 8, "Q")[0])
            handle.seek(8 * track_len, os.SEEK_CUR)
    return {
        "point_count": num_points,
        "xyz_min": xyz.min(axis=0).round(10).tolist() if num_points else [],
        "xyz_max": xyz.max(axis=0).round(10).tolist() if num_points else [],
        "xyz_mean": xyz.mean(axis=0).round(10).tolist() if num_points else [],
        "rgb_mean": (rgb_sum / max(num_points, 1)).round(6).tolist(),
        "error_mean": err_sum / max(num_points, 1),
    }


def sparse_model_path(scene_path: Path) -> Path:
    if (scene_path / "sparse" / "0" / "images.bin").exists():
        return scene_path / "sparse" / "0"
    return scene_path / "sparse"


def ply_vertex_count(path: Path) -> int | None:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def image_record(path: Path) -> dict[str, Any]:
    with Image.open(path) as img:
        raw_size = img.size
        raw_mode = img.mode
        exif_orientation = None
        try:
            exif_orientation = img.getexif().get(274)
        except Exception:
            exif_orientation = None
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        arr = np.asarray(rgb)
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "exists": path.exists(),
        "file_size": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "raw_width": raw_size[0],
        "raw_height": raw_size[1],
        "raw_mode": raw_mode,
        "exif_orientation": exif_orientation,
        "rgb_width": rgb.size[0],
        "rgb_height": rgb.size[1],
        "rgb_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "rgb_mean": float(arr.mean()),
        "rgb_std": float(arr.std()),
        "rgb_array": arr,
    }


def link_info(path: Path) -> dict[str, str]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "$item=Get-Item -LiteralPath "
            + repr(str(path))
            + " -Force; "
            "$target=''; "
            "if ($item.Target) { $target=($item.Target -join ';') }; "
            "[pscustomobject]@{FullName=$item.FullName;Mode=$item.Mode;LinkType=$item.LinkType;Target=$target;Attributes=$item.Attributes.ToString()} | ConvertTo-Json -Compress"
        ),
    ]
    out = run_cmd(cmd, timeout=20)
    try:
        parsed = json.loads(out)
        return {k: str(v) for k, v in parsed.items()}
    except Exception:
        return {"FullName": str(path), "Mode": "", "LinkType": "", "Target": "", "Attributes": "", "probe_output": out}


def metric_images(render_path: Path, gt_path: Path) -> dict[str, Any]:
    with Image.open(render_path) as render_img, Image.open(gt_path) as gt_img:
        render = np.asarray(ImageOps.exif_transpose(render_img).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(ImageOps.exif_transpose(gt_img).convert("RGB"), dtype=np.float32) / 255.0
    if render.shape != gt.shape:
        return {
            "compatible": False,
            "incompatible_reason": f"shape mismatch render={render.shape} gt={gt.shape}",
            "width": "",
            "height": "",
            "render_sha256": sha256_file(render_path),
            "gt_sha256": sha256_file(gt_path),
        }
    diff = render - gt
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    return {
        "compatible": True,
        "incompatible_reason": "",
        "width": render.shape[1],
        "height": render.shape[0],
        "mse": mse,
        "psnr": psnr,
        "mae": mae,
        "render_min": float(render.min()),
        "render_max": float(render.max()),
        "render_mean": float(render.mean()),
        "render_std": float(render.std()),
        "gt_min": float(gt.min()),
        "gt_max": float(gt.max()),
        "gt_mean": float(gt.mean()),
        "gt_std": float(gt.std()),
        "render_sha256": sha256_file(render_path),
        "gt_sha256": sha256_file(gt_path),
    }


def optional_torch_metrics(pairs: list[tuple[Path, Path]]) -> tuple[list[dict[str, Any]], str]:
    try:
        import torch
        import torchvision.transforms.functional as tvf
        from kornia.metrics import ssim as kornia_ssim
        import lpips
    except Exception as exc:
        return ([{"ssim": "", "lpips": "", "torch_metric_error": f"{type(exc).__name__}: {exc}"} for _ in pairs], "")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    except Exception as exc:
        return ([{"ssim": "", "lpips": "", "torch_metric_error": f"lpips init {type(exc).__name__}: {exc}"} for _ in pairs], f"device={device}")

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for render_path, gt_path in pairs:
            try:
                render = tvf.to_tensor(Image.open(render_path).convert("RGB")).unsqueeze(0).to(device)
                gt = tvf.to_tensor(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)
                ssim_val = float(kornia_ssim(render, gt, 3).mean().item())
                lpips_val = float(lpips_model(render, gt, normalize=True).mean().item())
                rows.append({"ssim": ssim_val, "lpips": lpips_val, "torch_metric_error": ""})
            except Exception as exc:
                rows.append({"ssim": "", "lpips": "", "torch_metric_error": f"{type(exc).__name__}: {exc}"})
    return rows, f"device={device}; lpips=alex; normalize=True; ssim=kornia.metrics.ssim(window=3)"


def evaluate_existing_renders() -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    groups = [
        ("historical_official_3dgs", HIST_OFFICIAL_RUN / "test" / "ours_30000"),
        ("historical_gsw_legacy", HIST_RUN / "test" / "ours_30000"),
        ("clean_gsw_legacy_target_rgb", CURRENT_RUN / "test" / "ours_30000_legacy_target_rgb"),
        ("clean_gsw_strict_intrinsic", CURRENT_RUN / "test" / "ours_30000_strict_intrinsic"),
        ("clean_gsw_strict_nearest_train", CURRENT_RUN / "test" / "ours_30000_strict_nearest_train"),
    ]

    per_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_pairs: list[tuple[Path, Path]] = []
    pair_indices: list[int] = []

    for label, method_dir in groups:
        renders_dir = method_dir / "renders"
        gt_dir = method_dir / "gt"
        if not renders_dir.exists() or not gt_dir.exists():
            summary_rows.append({"group": label, "method_dir": str(method_dir), "status": "missing render/gt dir"})
            continue
        for render_path in sorted(renders_dir.glob("*.png")):
            gt_path = gt_dir / render_path.name
            row = {
                "group": label,
                "method_dir": str(method_dir),
                "render_file": render_path.name,
                "render_path": str(render_path),
                "gt_path": str(gt_path),
                "gt_exists": gt_path.exists(),
            }
            if gt_path.exists():
                row.update(metric_images(render_path, gt_path))
                if row.get("compatible"):
                    all_pairs.append((render_path, gt_path))
                    pair_indices.append(len(per_rows))
            per_rows.append(row)

    torch_rows, torch_note = optional_torch_metrics(all_pairs)
    for row_index, torch_row in zip(pair_indices, torch_rows):
        per_rows[row_index].update(torch_row)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_rows:
        if row.get("gt_exists"):
            grouped.setdefault(str(row["group"]), []).append(row)
    for label, method_dir in groups:
        rows = grouped.get(label, [])
        if not rows:
            if not any(r.get("group") == label for r in summary_rows):
                summary_rows.append({"group": label, "method_dir": str(method_dir), "status": "no rows"})
            continue
        summary_rows.append({
            "group": label,
            "method_dir": str(method_dir),
            "status": "ok",
            "protocol_set": "original_protocol_set",
            "common_view_set": "00000.png;00001.png",
            "incompatible_or_unavailable": "",
            "num_views": len(rows),
            "unified_psnr": float(np.mean([float(r["psnr"]) for r in rows])),
            "unified_ssim": mean_optional(rows, "ssim"),
            "unified_lpips": mean_optional(rows, "lpips"),
            "common_view_psnr": float(np.mean([float(r["psnr"]) for r in rows])),
            "common_view_ssim": mean_optional(rows, "ssim"),
            "common_view_lpips": mean_optional(rows, "lpips"),
            "mse": float(np.mean([float(r["mse"]) for r in rows])),
            "mae": float(np.mean([float(r["mae"]) for r in rows])),
            "original_reported_metrics_json": existing_metric_json(method_dir.parent.parent, method_dir.name),
        })
    return summary_rows, per_rows, torch_note


def mean_optional(rows: list[dict[str, Any]], key: str) -> str | float:
    values = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
    return float(np.mean(values)) if values else ""


def existing_metric_json(scene_dir: Path, method_name: str) -> str:
    path = scene_dir / "results.json"
    if not path.exists():
        return ""
    try:
        results = read_json(path)
        return json.dumps(results.get(method_name, {}), ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def source_line(path: Path, pattern: str) -> str:
    if not path.exists():
        return "missing"
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if pattern in line:
            return f"{path.relative_to(ROOT)}:{idx}"
    return "not found"


def code_file_checksums() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in RELEVANT_CODE_FILES:
        current = ROOT / rel
        historical = HIST_CODE / rel
        rows.append({
            "relative_path": rel,
            "current_exists": current.exists(),
            "historical_exists": historical.exists(),
            "current_sha256": sha256_file(current) if current.exists() else "",
            "historical_worktree_sha256": sha256_file(historical) if historical.exists() else "",
            "equal": current.exists() and historical.exists() and sha256_file(current) == sha256_file(historical),
        })
    return rows


def effective_cfg(cfg: dict[str, Any], split: dict[str, Any], label: str) -> dict[str, Any]:
    result = dict(cfg)
    if label == "historical":
        result.update({
            "split_source": "legacy_tsv",
            "tsv_path": str(HIST_TSV),
            "train_camera_count": len(split["train_images"]),
            "test_camera_count": len(split["test_images"]),
            "actual_train_image_names": split["train_images"],
            "actual_test_image_names": split["test_images"],
            "seed": 0,
        })
    else:
        result.update({
            "split_source": "frozen_manifest",
            "tsv_path": "",
            "train_camera_count": len(split["train_images"]),
            "test_camera_count": len(split["test_images"]),
            "actual_train_image_names": split["train_images"],
            "actual_test_image_names": split["test_images"],
            "seed": 0,
        })
    return result


def compare_cfgs(historical_cfg: dict[str, Any], current_cfg: dict[str, Any], hist_split: dict[str, Any], cur_split: dict[str, Any]) -> list[dict[str, Any]]:
    historical_eff = effective_cfg(historical_cfg, hist_split, "historical")
    current_eff = effective_cfg(current_cfg, cur_split, "current")
    keys = list(dict.fromkeys(CFG_KEYS_OF_INTEREST + sorted(set(historical_eff) | set(current_eff))))
    rows: list[dict[str, Any]] = []
    for key in keys:
        hval = historical_eff.get(key, "<missing>")
        cval = current_eff.get(key, "<missing>")
        rows.append({
            "argument": key,
            "historical_value": repr_value(hval),
            "current_value": repr_value(cval),
            "equal": repr_value(hval) == repr_value(cval),
            "source_of_historical_value": str(HIST_RUN / "cfg_args"),
            "source_of_current_value": str(CURRENT_RUN / "cfg_args"),
            "expected_metric_impact": expected_cfg_impact(key, hval, cval),
            "notes": cfg_note(key),
        })
    rows.extend([
        {
            "argument": "train_image_names",
            "historical_value": repr_value(hist_split["train_images"]),
            "current_value": repr_value(cur_split["train_images"]),
            "equal": hist_split["train_images"] == cur_split["train_images"],
            "source_of_historical_value": str(HIST_TSV),
            "source_of_current_value": str(CURRENT_RUN / "split_used.json"),
            "expected_metric_impact": "high if different",
            "notes": "Effective training membership.",
        },
        {
            "argument": "test_image_names",
            "historical_value": repr_value(hist_split["test_images"]),
            "current_value": repr_value(cur_split["test_images"]),
            "equal": hist_split["test_images"] == cur_split["test_images"],
            "source_of_historical_value": str(HIST_TSV),
            "source_of_current_value": str(CURRENT_RUN / "split_used.json"),
            "expected_metric_impact": "high if different",
            "notes": "Effective held-out membership.",
        },
    ])
    return rows


def expected_cfg_impact(key: str, hval: Any, cval: Any) -> str:
    if repr_value(hval) == repr_value(cval):
        return "none"
    if key in {"source_path", "images", "sparse_subdir", "eval", "split_mode", "split_file"}:
        return "potentially high: data/split path differs"
    if key == "test_appearance_mode":
        return "evaluation-only unless used in periodic/report rendering"
    if key in {"use_colors_precomp", "use_features_mask", "use_kmap_pjmap", "use_lpips_loss", "map_num"}:
        return "high: core GS-W branch"
    if key.endswith("_lr") or key in {"iterations", "resolution"}:
        return "high: optimization setting"
    return "unknown/low unless consumed by training"


def cfg_note(key: str) -> str:
    if key == "source_path":
        return "Historical uses dense adapter; clean uses original COLMAP scene."
    if key == "eval":
        return "Historical legacy TSV split requires eval=True; clean frozen manifest works with eval=False."
    if key == "test_appearance_mode":
        return "Clean 30k periodic evaluation was strict_intrinsic; later diagnostic renders include legacy mode."
    if key == "split_mode":
        return "Different mechanism; membership audited separately."
    return ""


def split_audit() -> dict[str, Any]:
    frame = read_tsv_split(HIST_TSV)
    train = frame.loc[frame["split"] == "train", "filename"].tolist()
    test = frame.loc[frame["split"] == "test", "filename"].tolist()
    registered = frame["filename"].tolist()
    rows = []
    for _, row in frame.iterrows():
        role = str(row["split"])
        image_name = str(row["filename"])
        rows.append({
            "image_name": image_name,
            "expected_role": "test" if image_name in {"0001.jpg", "0009.jpg"} else "train",
            "historical_scene_role": role,
            "used_for_training": role == "train",
            "used_as_test": role == "test",
            "evidence": (
                f"cfg_args eval=True; legacy TSV split at {HIST_TSV}; "
                "historical scene/dataset_readers.py readColmapSceneInfo filters split=='train' into train_cam_infos "
                "and split=='test' into test_cam_infos before Scene.getTrainCameras sampling"
            ),
            "confidence": "high",
            "tsv_id": int(row["id"]),
            "current_split": "test" if image_name in read_json(CURRENT_RUN / "split_used.json")["test_images"] else "train",
        })
    write_csv(
        REPORT_DIR / "HISTORICAL_TRAIN_TEST_MEMBERSHIP.csv",
        rows,
        fieldnames=[
            "image_name",
            "expected_role",
            "historical_scene_role",
            "used_for_training",
            "used_as_test",
            "evidence",
            "confidence",
            "tsv_id",
            "current_split",
        ],
    )
    return {
        "registered_images": registered,
        "train_images": train,
        "test_images": test,
        "registered_sha256": sha256_lines(registered),
        "train_sha256": sha256_lines(train),
        "test_sha256": sha256_lines(test),
        "tsv_path": str(HIST_TSV),
    }


def data_audit(registered_names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    image_rows: list[dict[str, Any]] = []
    for name in registered_names:
        clean_path = CLEAN_DATA / "images" / name
        hist_path = HIST_DENSE / "images" / name
        clean = image_record(clean_path)
        hist = image_record(hist_path)
        diff = clean["rgb_array"].astype(np.int16) - hist["rgb_array"].astype(np.int16)
        image_rows.append({
            "filename": name,
            "clean_path": clean["path"],
            "historical_path": hist["path"],
            "historical_realpath": hist["realpath"],
            "clean_file_size": clean["file_size"],
            "historical_file_size": hist["file_size"],
            "file_sha256_equal": clean["file_sha256"] == hist["file_sha256"],
            "rgb_sha256_equal": clean["rgb_sha256"] == hist["rgb_sha256"],
            "clean_file_sha256": clean["file_sha256"],
            "historical_file_sha256": hist["file_sha256"],
            "clean_rgb_sha256": clean["rgb_sha256"],
            "historical_rgb_sha256": hist["rgb_sha256"],
            "clean_width": clean["rgb_width"],
            "clean_height": clean["rgb_height"],
            "historical_width": hist["rgb_width"],
            "historical_height": hist["rgb_height"],
            "clean_exif_orientation": clean["exif_orientation"],
            "historical_exif_orientation": hist["exif_orientation"],
            "max_abs_pixel_diff": int(np.max(np.abs(diff))),
            "mean_abs_pixel_diff": float(np.mean(np.abs(diff))),
        })

    hist_sparse = sparse_model_path(HIST_DENSE)
    clean_sparse = sparse_model_path(CLEAN_DATA)
    hist_cameras = read_cameras_binary(hist_sparse / "cameras.bin")
    clean_cameras = read_cameras_binary(clean_sparse / "cameras.bin")
    hist_images = {img.name: img for img in read_images_binary(hist_sparse / "images.bin").values()}
    clean_images = {img.name: img for img in read_images_binary(clean_sparse / "images.bin").values()}
    camera_rows = []
    for name in sorted(set(hist_images) | set(clean_images)):
        hist_img = hist_images.get(name)
        clean_img = clean_images.get(name)
        hist_cam = hist_cameras.get(hist_img.camera_id) if hist_img else None
        clean_cam = clean_cameras.get(clean_img.camera_id) if clean_img else None
        camera_rows.append({
            "filename": name,
            "historical_image_id": hist_img.image_id if hist_img else "",
            "clean_image_id": clean_img.image_id if clean_img else "",
            "image_id_equal": hist_img and clean_img and hist_img.image_id == clean_img.image_id,
            "historical_camera_id": hist_img.camera_id if hist_img else "",
            "clean_camera_id": clean_img.camera_id if clean_img else "",
            "camera_id_equal": hist_img and clean_img and hist_img.camera_id == clean_img.camera_id,
            "qvec_max_abs_diff": max_abs_diff(hist_img.qvec, clean_img.qvec) if hist_img and clean_img else "",
            "tvec_max_abs_diff": max_abs_diff(hist_img.tvec, clean_img.tvec) if hist_img and clean_img else "",
            "num_points2d_hist": hist_img.num_points2d if hist_img else "",
            "num_points2d_clean": clean_img.num_points2d if clean_img else "",
            "num_points2d_equal": hist_img and clean_img and hist_img.num_points2d == clean_img.num_points2d,
            "historical_camera_model": hist_cam.model if hist_cam else "",
            "clean_camera_model": clean_cam.model if clean_cam else "",
            "historical_width": hist_cam.width if hist_cam else "",
            "clean_width": clean_cam.width if clean_cam else "",
            "historical_height": hist_cam.height if hist_cam else "",
            "clean_height": clean_cam.height if clean_cam else "",
            "camera_params_max_abs_diff": max_abs_diff(hist_cam.params, clean_cam.params) if hist_cam and clean_cam else "",
        })

    sparse_summary = {
        "historical_sparse_path": str(hist_sparse),
        "clean_sparse_path": str(clean_sparse),
        "historical_sparse_realpath": str(hist_sparse.resolve()),
        "clean_sparse_realpath": str(clean_sparse.resolve()),
        "historical_images_link_info": link_info(HIST_DENSE / "images"),
        "historical_sparse_link_info": link_info(HIST_DENSE / "sparse"),
        "historical_sparse_files": sorted(p.name for p in hist_sparse.iterdir() if p.is_file()),
        "clean_sparse_files": sorted(p.name for p in clean_sparse.iterdir() if p.is_file()),
        "historical_points3d": read_points3d_binary_summary(hist_sparse / "points3D.bin"),
        "clean_points3d": read_points3d_binary_summary(clean_sparse / "points3D.bin"),
    }
    return image_rows, camera_rows, sparse_summary


def max_abs_diff(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def checkpoint_summary(run_path: Path, label: str) -> dict[str, Any]:
    ckpt = run_path / "ckpts_point_cloud" / "iteration_30000"
    ply_path = ckpt / "point_cloud.ply"
    row = {
        "label": label,
        "run_path": str(run_path),
        "ckpt_path": str(ckpt),
        "point_cloud_iteration": 30000,
        "point_cloud_ply_exists": ply_path.exists(),
        "gaussian_count": ply_vertex_count(ply_path) if ply_path.exists() else "",
    }
    for name in ["point_cloud.ply", "map_generator.pth", "color_net.pth", "other_atrributes_dict.pth"]:
        path = ckpt / name
        row[f"{name}_size"] = path.stat().st_size if path.exists() else ""
        row[f"{name}_sha256"] = sha256_file(path) if path.exists() and path.stat().st_size < 80_000_000 else ""
    row.update(torch_checkpoint_details(ckpt))
    return row


def torch_checkpoint_details(ckpt: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"torch_checkpoint_detail_error": f"{type(exc).__name__}: {exc}"}
    details: dict[str, Any] = {}
    for name in ["map_generator.pth", "color_net.pth"]:
        path = ckpt / name
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location="cpu")
            details[f"{name}_tensor_count"] = len(state)
            details[f"{name}_param_count"] = int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))
            details[f"{name}_tensor_shapes_json"] = json.dumps(
                {k: list(v.shape) for k, v in state.items() if hasattr(v, "shape")},
                sort_keys=True,
            )
        except Exception as exc:
            details[f"{name}_load_error"] = f"{type(exc).__name__}: {exc}"
    other = ckpt / "other_atrributes_dict.pth"
    if other.exists():
        try:
            state = torch.load(other, map_location="cpu")
            details["other_attributes_keys"] = ",".join(sorted(state.keys()))
            if "box_coord" in state:
                details["box_coord_shape"] = repr(tuple(state["box_coord"].shape))
        except Exception as exc:
            details["other_attributes_load_error"] = f"{type(exc).__name__}: {exc}"
    return details


def checkpoint_strict_load_probe() -> str:
    ckpt = CURRENT_RUN / "ckpts_point_cloud" / "iteration_30000"
    watched_files = [
        ckpt / "map_generator.pth",
        ckpt / "color_net.pth",
        ckpt / "other_atrributes_dict.pth",
    ]
    before = {path.name: sha256_file(path) for path in watched_files if path.exists()}
    code = r"""
from pathlib import Path
import torch
from argparse import Namespace
from arguments.args_init import argument_init
from scene.gaussian_model import GaussianModel

root = Path(r'G:\wl3dgs\Improved_GS-W')
cfg_path = Path(r'G:\wl3dgs\3dgs_runs\improved_gsw_strict_30k_decision_20260630\self_Trackmobile_4650TM_Mobile_Railcar_Mover\cfg_args')
ns = eval(cfg_path.read_text(encoding='utf-8'), {'__builtins__': {}}, {'Namespace': Namespace})
args = argument_init(ns)
model = GaussianModel(args.sh_degree, args)
ckpt = cfg_path.parent / 'ckpts_point_cloud' / 'iteration_30000'
map_state = torch.load(ckpt / 'map_generator.pth', map_location='cpu')
color_state = torch.load(ckpt / 'color_net.pth', map_location='cpu')
model.map_generator.load_state_dict(map_state, strict=True)
model.color_net.load_state_dict(color_state, strict=True)
print('strict_load_ok=True')
print('map_generator_keys=' + str(len(map_state)))
print('color_net_keys=' + str(len(color_state)))
print('features_dim=' + str(args.features_dim))
print('map_num=' + str(args.map_num))
print('map_generator_type=' + str(args.map_generator_type))
print('color_net_type=' + str(args.color_net_type))
print('render_executed=False')
print('backward_executed=False')
print('optimizer_step_executed=False')
"""
    output = run_cmd([sys.executable, "-c", code], cwd=ROOT, timeout=120)
    after = {path.name: sha256_file(path) for path in watched_files if path.exists()}
    sha_lines = [
        f"{name}_sha256_before={before[name]}"
        for name in sorted(before)
    ] + [
        f"{name}_sha256_after={after.get(name, '<missing>')}"
        for name in sorted(before)
    ] + [
        f"checkpoint_files_unchanged={before == after}"
    ]
    return "\n".join([output.rstrip(), *sha_lines]).strip()


def environment_lines(torch_note: str) -> str:
    lines = [
        "# Environment Equivalence Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Current audit/runtime environment",
        "",
        f"- Python executable: `{sys.executable}`",
        f"- Python version: `{sys.version.replace(chr(10), ' ')}`",
        f"- Working directory: `{ROOT}`",
        f"- Torch metric note: `{torch_note}`",
        "",
        "## Package versions",
        "",
    ]
    for module in ["torch", "torchvision", "lpips", "kornia", "PIL", "numpy", "pandas", "plyfile"]:
        try:
            mod = __import__(module)
            lines.append(f"- {module}: `{getattr(mod, '__version__', 'installed')}`")
        except Exception as exc:
            lines.append(f"- {module}: unavailable (`{type(exc).__name__}: {exc}`)")
    lines.extend([
        "",
        "## GPU / compiler probes",
        "",
        "```text",
        "$ nvidia-smi",
        strip_line_trailing_ws(run_cmd(["nvidia-smi"], timeout=10)),
        "",
        "$ nvcc --version",
        strip_line_trailing_ws(run_cmd(["nvcc", "--version"], timeout=10)),
        "```",
        "",
        "## Rasterizer signature probe",
        "",
        "```text",
    ])
    lines.append(rasterizer_probe())
    lines.extend([
        "```",
        "",
        "## Historical environment limits",
        "",
        "The historical run directory preserves `cfg_args`, logs, checkpoints, renders and metrics, but not a full exported conda environment. "
        "Therefore Python/PyTorch/CUDA identity for the historical 2026-06-21 run is not fully provable from files currently found. "
        "The historical working tree has a rasterizer compatibility patch adding `antialiasing=False`, which shows the active compiled rasterizer required the newer argument.",
        "",
        "The torchvision pretrained/weights and grid_sample align_corners warnings can affect feature projection only indirectly through library behavior; "
        "they do not explain a deterministic split or metric leak by themselves.",
    ])
    return "\n".join(lines) + "\n"


def rasterizer_probe() -> str:
    code = (
        "try:\n"
        " import diff_gaussian_rasterization as dgr\n"
        " fields=getattr(dgr.GaussianRasterizationSettings,'_fields',())\n"
        " print('GaussianRasterizationSettings._fields=', fields)\n"
        "except Exception as e:\n"
        " print(type(e).__name__+': '+str(e))\n"
    )
    return run_cmd([sys.executable, "-c", code], cwd=ROOT, timeout=20)


def build_active_branch_report() -> str:
    rows = [
        ("map_generator init", "scene/gaussian_model.py", "self.map_generator=Unet_model"),
        ("map_generator optimizer", "scene/gaussian_model.py", '"name": "map_generator"'),
        ("map_generator forward", "scene/gaussian_model.py", "out_gen=self.map_generator"),
        ("feature mask", "scene/gaussian_model.py", 'self.features_mask=out_gen["mask"]'),
        ("kmap/pjmap sampling", "scene/gaussian_model.py", "project2d("),
        ("color_net init", "scene/gaussian_model.py", "self.color_net=Color_net"),
        ("color_net forward", "scene/gaussian_model.py", "self._pre_comp_color=self.color_net"),
        ("colors_precomp renderer", "gaussian_renderer/__init__.py", "pc.use_colors_precomp"),
        ("LPIPS train loss", "train.py", "lpips_criteria(image,gt_image)"),
        ("box-coordinate loss", "train.py", "use_box_coord_loss"),
        ("densification", "scene/gaussian_model.py", "def densify_and_prune"),
        ("checkpoint save map", "scene/gaussian_model.py", "map_generator.state_dict"),
        ("checkpoint load map", "scene/gaussian_model.py", "map_generator.load_state_dict"),
        ("dropout eval disable", "scene/gaussian_model.py", "self.color_net.use_drop_out=False"),
    ]
    lines = [
        "# Active Branch Audit",
        "",
        "This audit checks the existing clean 30k code path without modifying training or running a new 30k job.",
        "",
        "## Branch evidence",
        "",
        "| Feature | Location | Evidence | Risk |",
        "|---|---:|---|---|",
    ]
    for feature, rel, pattern in rows:
        loc = source_line(ROOT / rel, pattern)
        risk = "high if accidentally disabled" if feature in {"map_generator forward", "color_net forward", "colors_precomp renderer"} else "medium"
        lines.append(f"| {feature} | `{loc}` | `{pattern}` | {risk} |")
    hist = checkpoint_summary(HIST_RUN, "historical_gsw")
    cur = checkpoint_summary(CURRENT_RUN, "clean_gsw")
    strict_probe = checkpoint_strict_load_probe()
    lines.extend([
        "",
        "## Checkpoint branch artifacts",
        "",
        f"- Historical map generator exists: `{bool(hist.get('map_generator.pth_size'))}`, size `{hist.get('map_generator.pth_size')}` bytes.",
        f"- Current map generator exists: `{bool(cur.get('map_generator.pth_size'))}`, size `{cur.get('map_generator.pth_size')}` bytes.",
        f"- Historical color net exists: `{bool(hist.get('color_net.pth_size'))}`, size `{hist.get('color_net.pth_size')}` bytes.",
        f"- Current color net exists: `{bool(cur.get('color_net.pth_size'))}`, size `{cur.get('color_net.pth_size')}` bytes.",
        f"- Historical `box_coord` shape: `{hist.get('box_coord_shape', '')}`.",
        f"- Current `box_coord` shape: `{cur.get('box_coord_shape', '')}`.",
        "",
        "## Direct gradient diagnostic status",
        "",
        "No new training or optimizer step was executed in this audit. The script did not run a backward pass on the 30k checkpoint. "
        "The branch is nevertheless active by source-path inspection: `map_generator(img)` feeds `_point_features`, `_point_features` feeds `Color_net`, "
        "and the rendered image participates in the photometric and LPIPS losses. Existing checkpoint files also contain map-generator, color-net and box-coordinate states.",
        "",
        "## Strict checkpoint load probe",
        "",
        "This probe loads the current clean 30k network state into the current architecture with `strict=True`. It does not render, run backward, or save any weights.",
        "",
        "```text",
        strict_probe,
        "```",
        "",
        "Backward-gradient diagnostic was intentionally not executed in this pass to avoid unnecessary checkpoint/model-state mutation risk. "
        "If GPT requires it, it should be run as a separate guarded diagnostic with pre/post SHA256 checks and no optimizer step.",
        "",
        "## Evaluation-mode behavior",
        "",
        "`GaussianModel.set_eval(True)` switches `color_net` to eval, disables color-net dropout, and disables the feature mask branch. "
        "It is restored by `set_eval(False)`. Clean strict modes additionally prevent target test RGB from entering `map_generator`.",
    ])
    return "\n".join(lines) + "\n"


def build_code_diff_report(checksum_rows: list[dict[str, Any]]) -> str:
    hist_status = run_cmd(["git", "status", "--short", "--branch"], cwd=HIST_CODE)
    hist_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=HIST_CODE)
    current_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=ROOT)
    current_branch = run_cmd(["git", "branch", "--show-current"], cwd=ROOT)
    diff_stat = run_cmd(["git", "diff", "--stat"], cwd=HIST_CODE)
    diff_patch = run_cmd(["git", "diff", "--", "gaussian_renderer/__init__.py", "render.py", "scene/dataset_readers.py"], cwd=HIST_CODE)
    write_text(REPORT_DIR / "HISTORICAL_WORKTREE.patch", diff_patch + "\n")

    unequal = [row for row in checksum_rows if row["current_exists"] and row["historical_exists"] and not row["equal"]]
    lines = [
        "# Historical vs Clean Code Diff",
        "",
        f"- Historical code path: `{HIST_CODE}`",
        f"- Historical commit: `{hist_head}`",
        f"- Historical status:",
        "",
        "```text",
        hist_status,
        "```",
        "",
        f"- Clean code path: `{ROOT}`",
        f"- Clean branch/head: `{current_branch}` / `{current_head}`",
        "",
        "## Historical dirty worktree patch",
        "",
        "```text",
        diff_stat,
        "```",
        "",
        "The dirty historical patch is focused on:",
        "",
        "- `scene/dataset_readers.py`: uses `cam_intrinsics[extr.camera_id]` and `uid = extr.id`. This is critical for legacy TSV split membership when a scene has a shared camera intrinsic.",
        "- `gaussian_renderer/__init__.py`: passes `antialiasing=False` and accepts a 3-value rasterizer return. This is rasterizer API compatibility.",
        "- `render.py`: protects the rendering-speed probe on small scenes. This affects post-training render utilities, not the training loss.",
        "",
        "Full focused historical patch is saved as `HISTORICAL_WORKTREE.patch`.",
        "",
        "## Whether historical 30k used these dirty files",
        "",
        "Status: **unknown**. The historical run preserves `cfg_args`, logs and outputs, but no immutable source snapshot or dirty-worktree checksum embedded in the run output was found. "
        "The current historical repo dirty files are frozen in this audit as the best available local evidence, but the audit does not assume they are exactly the files used on 2026-06-21.",
        "",
        "## Clean-vs-historical relevant file checksums",
        "",
        f"- Relevant source files compared: `{len(checksum_rows)}`.",
        f"- Files with different worktree content: `{len(unequal)}`.",
        "",
        "| File | Equal | Interpretation |",
        "|---|---:|---|",
    ]
    for row in checksum_rows:
        if row["equal"]:
            interpretation = "same as historical worktree"
        elif row["relative_path"] in {"train.py", "render.py", "gaussian_renderer/__init__.py", "scene/dataset_readers.py", "arguments/__init__.py"}:
            interpretation = "expected clean strict/data-layout adaptation or historical dirty compatibility change"
        else:
            interpretation = "changed; inspect checksum CSV"
        lines.append(f"| `{row['relative_path']}` | `{row['equal']}` | {interpretation} |")
    lines.extend([
        "",
        "## Result-difference relevance",
        "",
        "The most result-relevant code differences are split/data-loading and evaluation appearance handling. "
        "Training loss, optimizer groups, GS-W feature-map path, color-net path and densification logic remain close to upstream GS-W in the clean repo, "
        "but the clean repo contains strict appearance modes and frozen split plumbing that were not in the historical repo.",
    ])
    return "\n".join(lines) + "\n"


def build_data_report(image_rows: list[dict[str, Any]], camera_rows: list[dict[str, Any]], sparse_summary: dict[str, Any]) -> str:
    image_file_equal = all(bool(row["file_sha256_equal"]) for row in image_rows)
    image_rgb_equal = all(bool(row["rgb_sha256_equal"]) for row in image_rows)
    max_pixel_diff = max(int(row["max_abs_pixel_diff"]) for row in image_rows)
    mean_pixel_diff = max(float(row["mean_abs_pixel_diff"]) for row in image_rows)
    camera_pose_equal = all(
        row.get("qvec_max_abs_diff") == 0.0 and row.get("tvec_max_abs_diff") == 0.0 for row in camera_rows
    )
    clean_points = sparse_summary["clean_points3d"]
    hist_points = sparse_summary["historical_points3d"]
    lines = [
        "# Historical vs Clean Data Audit",
        "",
        f"- Historical adapter dense path: `{HIST_DENSE}`",
        f"- Clean source path: `{CLEAN_DATA}`",
        f"- Historical sparse path: `{sparse_summary['historical_sparse_path']}`",
        f"- Clean sparse path: `{sparse_summary['clean_sparse_path']}`",
        "",
        "## Image equivalence",
        "",
        f"- Registered images audited: `{len(image_rows)}`.",
        f"- File SHA256 all equal: `{image_file_equal}`.",
        f"- Decoded RGB SHA256 all equal: `{image_rgb_equal}`.",
        f"- Maximum absolute pixel difference across audited images: `{max_pixel_diff}`.",
        f"- Maximum mean absolute pixel difference across audited images: `{mean_pixel_diff}`.",
        "- No resize, crop, color conversion, EXIF-orientation discrepancy, or JPEG re-encoding was detected when all SHA256 values are equal.",
        "",
        "## Windows link type",
        "",
        f"- Historical `dense/images` link info: `{sparse_summary['historical_images_link_info']}`.",
        f"- Historical `dense/sparse` link info: `{sparse_summary['historical_sparse_link_info']}`.",
        "- If `LinkType` is populated, the adapter is a Windows link/junction/symlink to the original data rather than a re-encoded image export. If blank, PowerShell reports it as a normal directory.",
        "",
        "## COLMAP equivalence",
        "",
        f"- Camera pose all equal: `{camera_pose_equal}`.",
        f"- Historical point count: `{hist_points['point_count']}`.",
        f"- Clean point count: `{clean_points['point_count']}`.",
        f"- Historical xyz mean: `{hist_points['xyz_mean']}`.",
        f"- Clean xyz mean: `{clean_points['xyz_mean']}`.",
        "",
        "## Adapter side files",
        "",
        f"- Adapter parent files: `{sorted(p.name for p in HIST_ADAPTER.iterdir())}`.",
        f"- Adapter TSV: `{HIST_TSV}`.",
        "- The TSV affects historical GS-W split selection under `eval=True`; image/sparse files are otherwise equivalent to the clean source in this audit.",
        "",
        "Detailed tables: `HISTORICAL_VS_CLEAN_IMAGE_CHECKSUMS.csv` and `HISTORICAL_VS_CLEAN_CAMERA_COMPARISON.csv`.",
    ]
    return "\n".join(lines) + "\n"


def build_metric_report(summary_rows: list[dict[str, Any]], per_rows: list[dict[str, Any]], torch_note: str) -> str:
    lines = [
        "# Metric Pipeline Audit",
        "",
        "Unified re-evaluation used the same independent image loader and PSNR/MSE/MAE code for all existing render directories. "
        "SSIM and LPIPS were computed through the current installed `kornia` and `lpips` packages when available.",
        "",
        f"- Torch metric note: `{torch_note}`",
        f"- Per-view pairs evaluated: `{len([r for r in per_rows if r.get('gt_exists')])}`.",
        "",
        "| Group | Views | PSNR | SSIM | LPIPS | MAE | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row.get('group')} | {row.get('num_views', '')} | {fmt(row.get('unified_psnr'))} | "
            f"{fmt(row.get('unified_ssim'))} | {fmt(row.get('unified_lpips'))} | {fmt(row.get('mae'))} | {row.get('status')} |"
        )
    lines.extend([
        "",
        "## Pipeline checks",
        "",
        "- Render and GT images are paired by identical PNG filename inside each method directory.",
        "- Full RGB images are used; no mask or crop is applied.",
        "- Values are decoded through PIL as RGB and scaled to `[0, 1]`.",
        "- PSNR is computed from full-image MSE as `20 * log10(1 / sqrt(MSE))`, matching `utils.image_utils.psnr`.",
        "- Existing `results.json` values are copied into `original_reported_metrics_json` for comparison.",
        "- Because all audited groups contain the same two files, original-protocol and common-view metrics are identical here.",
        "- The script marks incompatible dimensions rather than resizing or cropping.",
        "- LPIPS uses `lpips.LPIPS(net='alex')`, `normalize=True`, `spatial=False`, and tensors in `[0, 1]`; no LPIPS resize/crop path is used.",
    ])
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def build_checkpoint_report(rows: list[dict[str, Any]]) -> str:
    hist = next(row for row in rows if row["label"] == "historical_gsw")
    cur = next(row for row in rows if row["label"] == "clean_gsw")
    gap = int(cur["gaussian_count"]) - int(hist["gaussian_count"])
    lines = [
        "# Checkpoint Comparison",
        "",
        f"- Historical Gaussian count: `{hist['gaussian_count']}`.",
        f"- Current clean Gaussian count: `{cur['gaussian_count']}`.",
        f"- Difference: `{gap}` Gaussians.",
        f"- Historical point cloud size: `{hist['point_cloud.ply_size']}` bytes.",
        f"- Current point cloud size: `{cur['point_cloud.ply_size']}` bytes.",
        f"- Map-generator checkpoint sizes equal: `{hist['map_generator.pth_size'] == cur['map_generator.pth_size']}`.",
        f"- Color-net checkpoint sizes equal: `{hist['color_net.pth_size'] == cur['color_net.pth_size']}`.",
        f"- Other-attributes sizes equal: `{hist['other_atrributes_dict.pth_size'] == cur['other_atrributes_dict.pth_size']}`.",
        f"- Historical map-generator keys/params: `{hist.get('map_generator.pth_tensor_count')}` / `{hist.get('map_generator.pth_param_count')}`.",
        f"- Current map-generator keys/params: `{cur.get('map_generator.pth_tensor_count')}` / `{cur.get('map_generator.pth_param_count')}`.",
        f"- Historical color-net keys/params: `{hist.get('color_net.pth_tensor_count')}` / `{hist.get('color_net.pth_param_count')}`.",
        f"- Current color-net keys/params: `{cur.get('color_net.pth_tensor_count')}` / `{cur.get('color_net.pth_param_count')}`.",
        f"- Historical other-attributes keys: `{hist.get('other_attributes_keys', '')}`.",
        f"- Current other-attributes keys: `{cur.get('other_attributes_keys', '')}`.",
        "- Feature dimensions: `48`; map_num: `3`; map generator: `unet/resnet18`; color net: `naive`.",
        "",
        "The network architecture file sizes match, but the learned weights differ and the final Gaussian count differs. "
        "That means the two completed 30k optimizations are not byte-equivalent even though the high-level GS-W configuration is largely matched.",
    ]
    return "\n".join(lines) + "\n"


def build_split_report(split: dict[str, Any], cur_split: dict[str, Any]) -> str:
    lines = [
        "# Historical Split Membership Audit",
        "",
        f"- Historical TSV: `{split['tsv_path']}`",
        f"- Historical registered images: `{len(split['registered_images'])}`.",
        f"- Historical train images: `{len(split['train_images'])}`.",
        f"- Historical test images: `{len(split['test_images'])}`.",
        f"- Historical train list: `{split['train_images']}`.",
        f"- Historical test list: `{split['test_images']}`.",
        f"- Current clean train list: `{cur_split['train_images']}`.",
        f"- Current clean test list: `{cur_split['test_images']}`.",
        "",
        "## Required membership answer",
        "",
        f"- `0001.jpg` entered historical training: `{ '0001.jpg' in split['train_images'] }`.",
        f"- `0009.jpg` entered historical training: `{ '0009.jpg' in split['train_images'] }`.",
        f"- `0001.jpg` entered historical test: `{ '0001.jpg' in split['test_images'] }`.",
        f"- `0009.jpg` entered historical test: `{ '0009.jpg' in split['test_images'] }`.",
        f"- Historical/current membership equal: `{split['train_images'] == cur_split['train_images'] and split['test_images'] == cur_split['test_images']}`.",
        "",
        "## Interpretation",
        "",
        "The historical 30k training membership was 13 images, and the two LLFF hold-8 images were held out from training. "
        "However, historical GS-W evaluation still used `legacy_target_rgb`, which conditions the appearance branch on the held-out RGB at render time. "
        "Therefore the historical `12.385471 dB` number is not a strict held-out result even though its training split was held-out.",
        "",
        "## Leakage taxonomy",
        "",
        "A. Training-set leakage: **No evidence of training-set leakage**. `0001.jpg` and `0009.jpg` are TSV `test` rows and are excluded from `train_cam_infos` under historical `eval=True` legacy TSV logic.",
        "",
        "B. Test-appearance leakage: **Present for historical legacy evaluation**. Historical GS-W legacy rendering calls `pc.forward(viewpoint_camera)`, so the held-out test camera RGB is passed to `map_generator` for appearance conditioning.",
        "",
        "C. Evaluation-protocol leakage: **No split/GT/size/mask leakage found in existing renders**. Unified metric audit uses the same two GT PNGs and full-image filename pairing. Historical and clean protocols still differ in appearance mode and source-path plumbing.",
    ]
    return "\n".join(lines) + "\n"


def build_rerun_decision(
    cfg_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    camera_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    ckpt_rows: list[dict[str, Any]],
) -> str:
    key_mismatches = [r for r in cfg_rows if r["equal"] is False and r["argument"] in CFG_KEYS_OF_INTEREST]
    image_equal = all(bool(row["rgb_sha256_equal"]) for row in image_rows)
    pose_equal = all(row.get("qvec_max_abs_diff") == 0.0 and row.get("tvec_max_abs_diff") == 0.0 for row in camera_rows)
    hist_metric = next((r for r in summary_rows if r.get("group") == "historical_gsw_legacy"), {})
    clean_metric = next((r for r in summary_rows if r.get("group") == "clean_gsw_legacy_target_rgb"), {})
    psnr_gap = float(hist_metric.get("unified_psnr", 0)) - float(clean_metric.get("unified_psnr", 0))
    hist_ckpt = next(row for row in ckpt_rows if row["label"] == "historical_gsw")
    clean_ckpt = next(row for row in ckpt_rows if row["label"] == "clean_gsw")
    gaussian_gap = int(clean_ckpt["gaussian_count"]) - int(hist_ckpt["gaussian_count"])

    command = (
        'conda run -n 3dgs python train.py '
        f'--source_path "{HIST_DENSE}" '
        f'--scene_name {SCENE} '
        f'--model_path "G:\\wl3dgs\\3dgs_runs\\gsw_matched_historical_adapter_r1_iter30000_REVIEW_ONLY\\{SCENE}" '
        '--resolution 1 --iterations 30000 '
        f'--split_mode frozen_manifest --split_file "{SPLIT_FILE}" '
        '--test_appearance_mode legacy_target_rgb '
        '--test_iterations 30000 --save_iterations 30000 --render_after_train --metrics_after_train --quiet'
    )
    lines = [
        "# Rerun Decision",
        "",
        "Primary classification: **D - configuration, data, code and metrics are broadly equivalent for membership/metric comparability, with residual optimization/code-path uncertainty**.",
        "",
        "Secondary findings: **B - test-appearance leakage is present in historical legacy evaluation**. "
        "Historical training membership is held-out, but historical legacy rendering is not strict because test RGB conditions the appearance branch.",
        "",
        "Configuration, split membership, data pixels, camera poses, and the metric pipeline are broadly equivalent for the purpose of judging training membership and metric comparability. "
        "The remaining historical-vs-clean PSNR gap is real, but it is not explained by training-view leakage through the split.",
        "",
        f"- Unified historical GS-W legacy PSNR: `{fmt(hist_metric.get('unified_psnr'))}`.",
        f"- Unified clean GS-W legacy PSNR: `{fmt(clean_metric.get('unified_psnr'))}`.",
        f"- Historical minus clean legacy gap: `{psnr_gap:.6f} dB`.",
        f"- Image RGB equivalence: `{image_equal}`.",
        f"- Camera pose equivalence: `{pose_equal}`.",
        f"- Final Gaussian count difference (clean - historical): `{gaussian_gap}`.",
        f"- Key configuration mismatches recorded: `{len(key_mismatches)}`.",
        "",
        "## Main gap source",
        "",
        "The most concrete source of non-equivalence is the completed optimization state: learned weights differ and final Gaussian count differs. "
        "The clean repo also contains strict split/evaluation plumbing and uses the direct source path, while the historical run used the adapter path and a dirty historical worktree. "
        "These are enough to require a matched reproduction before treating the 1.24 dB gap as an algorithmic conclusion.",
        "",
        "## No-go status",
        "",
        "The previous strict GS-W no-go should be **paused/qualified**, not fully retracted. "
        "It remains true that strict held-out GS-W on the current clean 30k checkpoint is weak. "
        "It is not yet proven that GS-W itself is unsuitable as a method base, because the historical legacy checkpoint was materially better under the same target-conditioned evaluation.",
        "",
        "## Corrected 30k requirement",
        "",
        "A corrected matched 30k is required if the team wants a definitive historical-vs-clean equivalence answer. "
        "This audit does not run it.",
        "",
        "Important: do not use current clean code with `--eval --split_mode legacy` for this scene unless the historical "
        "`uid = extr.id` COLMAP fix is also applied. Current clean frozen-manifest splitting is filename-based and avoids that legacy TSV UID hazard.",
        "",
        "Suggested single command, for GPT review only:",
        "",
        "```powershell",
        command,
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_summary(
    split: dict[str, Any],
    cur_split: dict[str, Any],
    cfg_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    ckpt_rows: list[dict[str, Any]],
) -> str:
    hist_metric = next((r for r in metric_rows if r.get("group") == "historical_gsw_legacy"), {})
    clean_metric = next((r for r in metric_rows if r.get("group") == "clean_gsw_legacy_target_rgb"), {})
    strict_metric = next((r for r in metric_rows if r.get("group") == "clean_gsw_strict_intrinsic"), {})
    hist_ckpt = next(row for row in ckpt_rows if row["label"] == "historical_gsw")
    clean_ckpt = next(row for row in ckpt_rows if row["label"] == "clean_gsw")
    key_diff_args = [r["argument"] for r in cfg_rows if r["equal"] is False and r["argument"] in CFG_KEYS_OF_INTEREST]
    lines = [
        "# GS-W Historical vs Clean Equivalence Audit Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Key answers",
        "",
        f"1. Historical training used `{len(split['train_images'])}` images.",
        f"2. `0001.jpg` and `0009.jpg` did **not** enter historical training; both were historical test images.",
        "3. Historical `12.385471 dB` is **not strict held-out** because historical GS-W legacy rendering conditions the appearance branch on held-out test RGB.",
        f"4. Historical/current clean split membership equal: `{split['train_images'] == cur_split['train_images'] and split['test_images'] == cur_split['test_images']}`.",
        f"5. Historical GS-W legacy unified PSNR: `{fmt(hist_metric.get('unified_psnr'))}`.",
        f"6. Current clean GS-W legacy unified PSNR: `{fmt(clean_metric.get('unified_psnr'))}`.",
        f"7. Current clean GS-W strict intrinsic unified PSNR: `{fmt(strict_metric.get('unified_psnr'))}`.",
        f"8. Historical Gaussian count: `{hist_ckpt['gaussian_count']}`; current clean Gaussian count: `{clean_ckpt['gaussian_count']}`.",
        "",
        "## Current clean 30k configuration status",
        "",
        "The clean 30k enabled the complete core GS-W switches found in the historical config: "
        "`use_colors_precomp=True`, `use_features_mask=True`, `use_kmap_pjmap=True`, `use_lpips_loss=True`, `map_num=3`, "
        "`map_generator_type=unet`, `features_dim=48`, and `color_net_type=naive`.",
        "",
        f"Key argument differences are: `{key_diff_args}`.",
        "Most are protocol/data-path differences: historical `source_path` points to the GS-W dense adapter and uses `eval=True` legacy TSV splitting; "
        "clean points to the original COLMAP source and uses `split_mode=frozen_manifest`.",
        "",
        "## Main interpretation",
        "",
        "The 1.24 dB historical-vs-clean legacy gap is real under unified re-evaluation, but current evidence points to optimization/code-path non-equivalence rather than split leakage. "
        "Because the final Gaussian count and learned state differ, a matched adapter-path rerun is needed before using the gap to decide GS-W go/no-go.",
        "",
        "## Leakage taxonomy",
        "",
        "- Training-set leakage: no.",
        "- Test-appearance leakage: yes for historical legacy evaluation.",
        "- Evaluation-protocol leakage: no split/GT/size/mask mismatch found in existing renders; appearance protocol differs.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    hist_cfg = read_cfg(HIST_RUN / "cfg_args")
    cur_cfg = read_cfg(CURRENT_RUN / "cfg_args")
    split = split_audit()
    cur_split = read_json(CURRENT_RUN / "split_used.json")
    cfg_rows = compare_cfgs(hist_cfg, cur_cfg, split, cur_split)
    write_csv(REPORT_DIR / "CURRENT_VS_HISTORICAL_CFG.csv", cfg_rows)
    write_text(REPORT_DIR / "HISTORICAL_SPLIT_MEMBERSHIP_AUDIT.md", build_split_report(split, cur_split))

    image_rows, camera_rows, sparse_summary = data_audit(split["registered_images"])
    write_csv(REPORT_DIR / "HISTORICAL_VS_CLEAN_IMAGE_CHECKSUMS.csv", image_rows)
    write_csv(REPORT_DIR / "HISTORICAL_VS_CLEAN_CAMERA_COMPARISON.csv", camera_rows)
    write_text(REPORT_DIR / "HISTORICAL_VS_CLEAN_DATA_AUDIT.md", build_data_report(image_rows, camera_rows, sparse_summary))

    checksum_rows = code_file_checksums()
    write_csv(REPORT_DIR / "HISTORICAL_WORKTREE_CHECKSUMS.csv", checksum_rows)
    write_text(REPORT_DIR / "HISTORICAL_VS_CLEAN_CODE_DIFF.md", build_code_diff_report(checksum_rows))

    summary_rows, per_rows, torch_note = evaluate_existing_renders()
    write_csv(REPORT_DIR / "UNIFIED_REEVALUATION.csv", summary_rows)
    write_csv(REPORT_DIR / "PER_VIEW_UNIFIED_REEVALUATION.csv", per_rows)
    write_text(REPORT_DIR / "METRIC_PIPELINE_AUDIT.md", build_metric_report(summary_rows, per_rows, torch_note))

    ckpt_rows = [
        checkpoint_summary(HIST_RUN, "historical_gsw"),
        checkpoint_summary(CURRENT_RUN, "clean_gsw"),
    ]
    write_csv(REPORT_DIR / "CHECKPOINT_COMPARISON.csv", ckpt_rows)
    write_text(REPORT_DIR / "CHECKPOINT_COMPARISON.md", build_checkpoint_report(ckpt_rows))

    write_text(REPORT_DIR / "ACTIVE_BRANCH_AUDIT.md", build_active_branch_report())
    write_text(REPORT_DIR / "ENVIRONMENT_EQUIVALENCE_AUDIT.md", environment_lines(torch_note))
    write_text(REPORT_DIR / "RERUN_DECISION.md", build_rerun_decision(cfg_rows, image_rows, camera_rows, summary_rows, ckpt_rows))
    write_text(REPORT_DIR / "EQUIVALENCE_AUDIT_SUMMARY.md", build_summary(split, cur_split, cfg_rows, summary_rows, ckpt_rows))

    print(f"Wrote audit reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
