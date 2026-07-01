from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "scene_selection"))

from scene_selection_common import (  # noqa: E402
    DATA_ROOT,
    IMPROVED_ROOT,
    OFFICIAL_ROOT,
    REPORT_DIR,
    REVIEW_ROOT,
    load_split_names,
    ps_join,
    python_cmd,
    read_json,
    report_manifest_path,
    run_cmd,
    sha256_file,
    write_csv,
    write_report_manifest,
    write_text,
)


SMOKE_RUN_ROOT = Path(r"G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701_clean")
LOG_DIR = REPORT_DIR / "logs"
TRACKMOBILE_OFFICIAL_30K_MIN = 17.416
TRACKMOBILE_GSW_30K_MIN = 91.083
TRACKMOBILE_TRAIN_COUNT = 13


@dataclass
class CommandRecord:
    label: str
    command: list[str]
    cwd: Path
    stdout_log: Path
    stderr_log: Path
    start_time: float
    end_time: float
    returncode: int
    peak_gpu_mem_mb: int | None

    @property
    def duration_sec(self) -> float:
        return self.end_time - self.start_time


def nvidia_smi_memory_mb() -> int | None:
    result = run_cmd(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        timeout=30,
    )
    if result.returncode != 0:
        return None
    values: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            pass
    return max(values) if values else None


def run_logged(
    label: str,
    cmd: list[str],
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    gpu_poll_sec: float = 5.0,
) -> CommandRecord:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    peak = nvidia_smi_memory_mb()
    with stdout_log.open("w", encoding="utf-8", errors="replace") as out, stderr_log.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        out.write(f"# label: {label}\n# cwd: {cwd}\n# command: {ps_join(cmd)}\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
        while proc.poll() is None:
            mem = nvidia_smi_memory_mb()
            if mem is not None:
                peak = mem if peak is None else max(peak, mem)
            time.sleep(gpu_poll_sec)
        returncode = int(proc.returncode)
    end = time.time()
    return CommandRecord(label, cmd, cwd, stdout_log, stderr_log, start, end, returncode, peak)


def parse_last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise ValueError(f"No JSON object found in stdout tail: {stdout[-500:]!r}")


def reader_split(scene: str, method: str, manifest_path: Path) -> dict:
    if method == "official":
        code = (
            "import sys,json;"
            f"sys.path.insert(0,{str(OFFICIAL_ROOT)!r});"
            "from scene.dataset_readers import readColmapSceneInfo;"
            f"info=readColmapSceneInfo({str(DATA_ROOT / scene)!r},'images','',True,False);"
            "print(json.dumps({'train':[c.image_name for c in info.train_cameras],"
            "'test':[c.image_name for c in info.test_cameras]}))"
        )
        result = run_cmd([sys.executable, "-c", code], cwd=OFFICIAL_ROOT, timeout=240)
    elif method == "gsw":
        code = (
            "import sys,json;"
            f"sys.path.insert(0,{str(IMPROVED_ROOT)!r});"
            "from scene.dataset_readers import readColmapSceneInfo;"
            f"info=readColmapSceneInfo({str(DATA_ROOT / scene)!r},'images',False,"
            f"split_mode='frozen_manifest',split_file={str(manifest_path)!r});"
            "print(json.dumps({'train':[c.image_name for c in info.train_cameras],"
            "'test':[c.image_name for c in info.test_cameras],"
            "'summary':getattr(info,'split_summary',{})}))"
        )
        result = run_cmd([sys.executable, "-c", code], cwd=IMPROVED_ROOT, timeout=240)
    else:
        raise ValueError(method)
    if result.returncode != 0:
        raise RuntimeError(f"{method} reader failed for {scene}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    data = parse_last_json(result.stdout)
    return {
        "train": list(data["train"]),
        "test": list(data["test"]),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-5:]),
    }


def decoded_rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()


def split_preflight(scene: str, role: str) -> dict[str, object]:
    manifest_path = write_report_manifest(scene)
    manifest = read_json(manifest_path)
    all_names = list(manifest["registered_images"])
    train = list(manifest["train_images"])
    test = list(manifest["test_images"])
    expected_test = [name for idx, name in enumerate(sorted(all_names)) if idx % 8 == 0]
    expected_train = [name for idx, name in enumerate(sorted(all_names)) if idx % 8 != 0]
    official = reader_split(scene, "official", manifest_path)
    gsw = reader_split(scene, "gsw", manifest_path)

    checks = {
        "manifest_sha256": sha256_file(manifest_path),
        "train_test_disjoint": not set(train).intersection(test),
        "union_covers_registered": set(train).union(test) == set(all_names),
        "hold8_train_match": train == expected_train,
        "hold8_test_match": test == expected_test,
        "official_train_match": official["train"] == train,
        "official_test_match": official["test"] == test,
        "gsw_train_match": gsw["train"] == train,
        "gsw_test_match": gsw["test"] == test,
    }
    rows = []
    for split_name, names in [("train", train), ("test", test)]:
        for idx, name in enumerate(names):
            image_path = DATA_ROOT / scene / "images" / name
            rows.append(
                {
                    "role": role,
                    "scene": scene,
                    "split": split_name,
                    "index": idx,
                    "image_name": name,
                    "image_path": str(image_path),
                    "decoded_rgb_sha256": decoded_rgb_sha256(image_path),
                }
            )
    csv_path = REPORT_DIR / f"{role}_SPLIT_PREFLIGHT.csv"
    write_csv(csv_path, rows, ["role", "scene", "split", "index", "image_name", "image_path", "decoded_rgb_sha256"])
    status = "PASS" if all(v for key, v in checks.items() if key != "manifest_sha256") else "FAIL"
    md = [
        f"# {role}_SPLIT_PREFLIGHT",
        "",
        f"- Scene: `{scene}`",
        f"- Scene path: `{DATA_ROOT / scene}`",
        f"- Manifest: `{manifest_path}`",
        f"- Manifest SHA256: `{checks['manifest_sha256']}`",
        f"- Train/test: `{len(train)}/{len(test)}`",
        f"- Status: `{status}`",
        "",
        "## Checks",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key, value in checks.items():
        md.append(f"| {key} | `{value}` |")
    md.extend(
        [
            "",
            "## Protocol",
            "",
            "- Split rule: sort registered COLMAP image names lexicographically; `index % 8 == 0` is test.",
            "- Official reader path uses clean official 3DGS `--eval` LLFF hold-8 behavior.",
            "- GS-W reader path uses `split_mode=frozen_manifest` with the exact manifest above.",
        ]
    )
    write_text(REPORT_DIR / f"{role}_SPLIT_PREFLIGHT.md", "\n".join(md) + "\n")
    return {
        "role": role,
        "scene": scene,
        "manifest_path": str(manifest_path),
        "manifest_sha256": checks["manifest_sha256"],
        "train_count": len(train),
        "test_count": len(test),
        "status": status,
        **{key: value for key, value in checks.items() if isinstance(value, bool)},
    }


def parse_vertex_count(model_path: Path, iteration: int) -> int | None:
    ply = checkpoint_ply(model_path, iteration)
    if ply is None:
        return None
    with ply.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def checkpoint_ply(model_path: Path, iteration: int) -> Path | None:
    candidates = [
        model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        model_path / "ckpts_point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def checkpoint_dir(model_path: Path, iteration: int) -> Path:
    ply = checkpoint_ply(model_path, iteration)
    if ply is None:
        return model_path / "point_cloud" / f"iteration_{iteration}"
    return ply.parent


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def unique_model_path(base: Path, iteration: int) -> Path:
    if not base.exists() or checkpoint_ply(base, iteration) is not None:
        return base
    for idx in range(1, 100):
        candidate = base.with_name(f"{base.name}_rerun{idx}")
        if not candidate.exists() or checkpoint_ply(candidate, iteration) is not None:
            return candidate
    raise RuntimeError(f"Cannot find non-overwriting model path for {base}")


def official_train_command(scene: str, model_path: Path, iterations: int) -> list[str]:
    return python_cmd(
        OFFICIAL_ROOT / "train.py",
        [
            "--source_path",
            str(DATA_ROOT / scene),
            "--model_path",
            str(model_path),
            "--eval",
            "--resolution",
            "1",
            "--iterations",
            str(iterations),
            "--optimizer_type",
            "default",
            "--depth_l1_weight_init",
            "0",
            "--depth_l1_weight_final",
            "0",
            "--test_iterations",
            "1000000",
            "--save_iterations",
            str(iterations),
            "--disable_viewer",
            "--quiet",
        ],
    )


def official_render_command(scene: str, model_path: Path, iteration: int) -> list[str]:
    return python_cmd(
        OFFICIAL_ROOT / "render.py",
        [
            "--source_path",
            str(DATA_ROOT / scene),
            "--model_path",
            str(model_path),
            "--eval",
            "--resolution",
            "1",
            "--iteration",
            str(iteration),
            "--skip_train",
            "--quiet",
        ],
    )


def gsw_train_command(scene: str, manifest_path: Path, model_path: Path, iterations: int) -> list[str]:
    return python_cmd(
        IMPROVED_ROOT / "train.py",
        [
            "--source_path",
            str(DATA_ROOT / scene),
            "--scene_name",
            scene,
            "--model_path",
            str(model_path),
            "--resolution",
            "1",
            "--iterations",
            str(iterations),
            "--split_mode",
            "frozen_manifest",
            "--split_file",
            str(manifest_path),
            "--test_appearance_mode",
            "strict_intrinsic",
            "--test_iterations",
            "1000000",
            "--save_iterations",
            str(iterations),
            "--disable_render_after_train",
            "--disable_metrics_after_train",
            "--disable_train_temp_images",
            "--quiet",
        ],
    )


def gsw_render_command(scene: str, manifest_path: Path, model_path: Path, iteration: int) -> list[str]:
    return python_cmd(
        IMPROVED_ROOT / "render.py",
        [
            "--source_path",
            str(DATA_ROOT / scene),
            "--scene_name",
            scene,
            "--model_path",
            str(model_path),
            "--resolution",
            "1",
            "--iteration",
            str(iteration),
            "--split_mode",
            "frozen_manifest",
            "--split_file",
            str(manifest_path),
            "--test_appearance_mode",
            "strict_intrinsic",
            "--render_output_tag",
            "strict_intrinsic",
            "--skip_train",
            "--quiet",
        ],
    )


def method_dir(model_path: Path, method: str, iteration: int) -> Path:
    suffix = "_strict_intrinsic" if method == "gsw_strict_intrinsic" else ""
    return model_path / "test" / f"ours_{iteration}{suffix}"


def run_smoke_case(scene: str, role: str, method: str, iteration: int, manifest_path: Path) -> dict[str, object]:
    base_model_path = SMOKE_RUN_ROOT / role / method / f"iter_{iteration}" / scene
    model_path = unique_model_path(base_model_path, iteration)
    model_path.mkdir(parents=True, exist_ok=True)
    log_prefix = f"{role}_{method}_{iteration}"
    checkpoint = checkpoint_ply(model_path, iteration)

    if method == "official_3dgs":
        train_cmd = official_train_command(scene, model_path, iteration)
        render_cmd = official_render_command(scene, model_path, iteration)
        cwd = OFFICIAL_ROOT
    elif method == "gsw_strict_intrinsic":
        train_cmd = gsw_train_command(scene, manifest_path, model_path, iteration)
        render_cmd = gsw_render_command(scene, manifest_path, model_path, iteration)
        cwd = IMPROVED_ROOT
    else:
        raise ValueError(method)

    if checkpoint is not None:
        train_record = None
        train_returncode = 0
        train_duration = 0.0
        train_peak = None
        train_status = "reused_existing_checkpoint"
    else:
        train_record = run_logged(
            f"{log_prefix}_train",
            train_cmd,
            cwd,
            LOG_DIR / f"{log_prefix}_train.stdout.log",
            LOG_DIR / f"{log_prefix}_train.stderr.log",
            gpu_poll_sec=10.0,
        )
        train_returncode = train_record.returncode
        train_duration = train_record.duration_sec
        train_peak = train_record.peak_gpu_mem_mb
        train_status = "ran"
        if train_returncode != 0:
            raise RuntimeError(f"Train failed for {log_prefix}; see {train_record.stderr_log}")

    out_dir = method_dir(model_path, method, iteration)
    render_dir = out_dir / "renders"
    gt_dir = out_dir / "gt"
    expected_test_count = len(load_split_names(scene)[2])
    if render_dir.exists() and len(list(render_dir.glob("*.png"))) == expected_test_count:
        render_record = None
        render_returncode = 0
        render_duration = 0.0
        render_peak = None
        render_status = "reused_existing_render"
    else:
        render_record = run_logged(
            f"{log_prefix}_render",
            render_cmd,
            cwd,
            LOG_DIR / f"{log_prefix}_render.stdout.log",
            LOG_DIR / f"{log_prefix}_render.stderr.log",
            gpu_poll_sec=5.0,
        )
        render_returncode = render_record.returncode
        render_duration = render_record.duration_sec
        render_peak = render_record.peak_gpu_mem_mb
        render_status = "ran"
        if render_returncode != 0:
            raise RuntimeError(f"Render failed for {log_prefix}; see {render_record.stderr_log}")

    return {
        "role": role,
        "scene": scene,
        "method": method,
        "iterations": iteration,
        "model_path": str(model_path),
        "method_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "train_status": train_status,
        "train_returncode": train_returncode,
        "train_duration_sec": round(train_duration, 3),
        "train_duration_min": round(train_duration / 60.0, 3),
        "train_peak_gpu_mem_mb": train_peak,
        "render_status": render_status,
        "render_returncode": render_returncode,
        "render_duration_sec": round(render_duration, 3),
        "render_duration_min": round(render_duration / 60.0, 3),
        "render_peak_gpu_mem_mb": render_peak,
        "gaussian_count": parse_vertex_count(model_path, iteration),
        "checkpoint_size_bytes": dir_size_bytes(checkpoint_dir(model_path, iteration)),
        "render_count": len(list(render_dir.glob("*.png"))) if render_dir.exists() else 0,
        "gt_count": len(list(gt_dir.glob("*.png"))) if gt_dir.exists() else 0,
        "train_command": ps_join(train_cmd),
        "render_command": ps_join(render_cmd),
        "train_stdout_log": str(LOG_DIR / f"{log_prefix}_train.stdout.log"),
        "train_stderr_log": str(LOG_DIR / f"{log_prefix}_train.stderr.log"),
        "render_stdout_log": str(LOG_DIR / f"{log_prefix}_render.stdout.log"),
        "render_stderr_log": str(LOG_DIR / f"{log_prefix}_render.stderr.log"),
    }


def mapping_for_output(scene: str, method_dir_path: Path) -> dict[str, str]:
    mapping_csv = method_dir_path / "render_view_mapping.csv"
    if mapping_csv.exists():
        with mapping_csv.open("r", encoding="utf-8", newline="") as handle:
            return {row["render_file"]: row["image_name"] for row in csv.DictReader(handle)}
    test_names = load_split_names(scene)[2]
    render_files = sorted(path.name for path in (method_dir_path / "renders").glob("*.png"))
    mapping = {name: test_names[idx] if idx < len(test_names) else "" for idx, name in enumerate(render_files)}
    if render_files:
        with (method_dir_path / "render_view_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["render_file", "image_name", "colmap_id", "split_role"])
            for idx, name in enumerate(render_files):
                writer.writerow([name, mapping[name], "", "test"])
    return mapping


def tensor_from_image(path: Path, device: str):
    import torch

    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def psnr_value(pred, gt) -> float:
    import torch

    mse = ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(1)
    return float((20 * torch.log10(1.0 / torch.sqrt(mse))).mean().item())


def evaluate_outputs(smoke_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import torch
    from kornia.metrics import ssim as kornia_ssim
    import lpips

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    summary_rows: list[dict[str, object]] = []
    per_view_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for row in smoke_rows:
            scene = str(row["scene"])
            out_dir = Path(str(row["method_dir"]))
            render_dir = out_dir / "renders"
            gt_dir = out_dir / "gt"
            render_files = sorted(path.name for path in render_dir.glob("*.png"))
            gt_files = sorted(path.name for path in gt_dir.glob("*.png"))
            if render_files != gt_files:
                raise ValueError(f"Render/GT mismatch in {out_dir}: {render_files} vs {gt_files}")
            mapping = mapping_for_output(scene, out_dir)
            psnrs: list[float] = []
            ssims: list[float] = []
            lpipss: list[float] = []
            black_count = 0
            finite_count = 0
            for name in render_files:
                render_path = render_dir / name
                gt_path = gt_dir / name
                pred = tensor_from_image(render_path, device)
                gt = tensor_from_image(gt_path, device)
                if tuple(pred.shape) != tuple(gt.shape):
                    raise ValueError(f"Shape mismatch in {out_dir}/{name}: {tuple(pred.shape)} vs {tuple(gt.shape)}")
                psnr = psnr_value(pred, gt)
                ssim = float(kornia_ssim(pred, gt, 3).mean().item())
                lp = float(lpips_model(pred, gt, normalize=True).item())
                render_np = pred.detach().cpu().numpy()
                gt_np = gt.detach().cpu().numpy()
                render_min = float(render_np.min())
                render_max = float(render_np.max())
                render_mean = float(render_np.mean())
                render_std = float(render_np.std())
                is_black = render_max <= 1e-6 or render_std <= 1e-8
                if is_black:
                    black_count += 1
                if all(math.isfinite(v) for v in [psnr, ssim, lp]):
                    finite_count += 1
                per_view_rows.append(
                    {
                        **{key: row[key] for key in ["role", "scene", "method", "iterations", "method_dir"]},
                        "render_file": name,
                        "image_name": mapping.get(name, ""),
                        "width": int(pred.shape[-1]),
                        "height": int(pred.shape[-2]),
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lp,
                        "render_min": render_min,
                        "render_max": render_max,
                        "render_mean": render_mean,
                        "render_std": render_std,
                        "gt_min": float(gt_np.min()),
                        "gt_max": float(gt_np.max()),
                        "gt_mean": float(gt_np.mean()),
                        "render_rgb_sha256": decoded_rgb_sha256(render_path),
                        "gt_rgb_sha256": decoded_rgb_sha256(gt_path),
                        "is_black_render": is_black,
                        "finite_metrics": all(math.isfinite(v) for v in [psnr, ssim, lp]),
                    }
                )
                psnrs.append(psnr)
                ssims.append(ssim)
                lpipss.append(lp)
            summary_rows.append(
                {
                    **row,
                    "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
                    "ssim": float(np.mean(ssims)) if ssims else float("nan"),
                    "lpips": float(np.mean(lpipss)) if lpipss else float("nan"),
                    "finite_metric_views": finite_count,
                    "black_render_views": black_count,
                    "all_metrics_finite": finite_count == len(render_files),
                    "no_black_images": black_count == 0,
                    "full_image": True,
                    "lpips_net": "alex",
                    "lpips_normalize": True,
                }
            )
    return summary_rows, per_view_rows


def gt_checksum_audit(summary_rows: list[dict[str, object]], per_view_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    lookup: dict[tuple[str, int, str, str], str] = {}
    for row in per_view_rows:
        key = (str(row["scene"]), int(row["iterations"]), str(row["method"]), str(row["image_name"]))
        lookup[key] = str(row["gt_rgb_sha256"])
    scenes = sorted({str(row["scene"]) for row in summary_rows})
    iterations = sorted({int(row["iterations"]) for row in summary_rows})
    for scene in scenes:
        _, _, test_names, _ = load_split_names(scene)
        for iteration in iterations:
            for image_name in test_names:
                off = lookup.get((scene, iteration, "official_3dgs", image_name), "")
                gsw = lookup.get((scene, iteration, "gsw_strict_intrinsic", image_name), "")
                rows.append(
                    {
                        "scene": scene,
                        "iterations": iteration,
                        "image_name": image_name,
                        "official_gt_sha256": off,
                        "gsw_gt_sha256": gsw,
                        "match": bool(off and gsw and off == gsw),
                    }
                )
    write_csv(
        REPORT_DIR / "TWO_SCENE_GT_CHECKSUMS.csv",
        rows,
        ["scene", "iterations", "image_name", "official_gt_sha256", "gsw_gt_sha256", "match"],
    )
    return rows


def selected_scenes() -> dict[str, str]:
    selected = read_json(REPORT_DIR / "SELECTED_SCENES.json")
    return {"H": str(selected["H"]["scene"]), "M": str(selected["M"]["scene"])}


def long_run_budget(summary_rows: list[dict[str, object]]) -> None:
    inventory_rows: dict[str, dict[str, str]] = {}
    with (REPORT_DIR / "SCENE_CANDIDATE_INVENTORY.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            inventory_rows[row["scene"]] = row
    selected = selected_scenes()
    scenes = [selected["H"], selected["M"]]
    budget_rows = []
    for scene in scenes:
        train_count = int(inventory_rows[scene]["frozen_train_count"])
        for method in ["official_3dgs", "gsw_strict_intrinsic"]:
            track_base = TRACKMOBILE_OFFICIAL_30K_MIN if method == "official_3dgs" else TRACKMOBILE_GSW_30K_MIN
            track_scaled = track_base * math.sqrt(train_count / TRACKMOBILE_TRAIN_COUNT)
            smoke_300 = next(
                (
                    float(row["train_duration_min"]) * 100.0
                    for row in summary_rows
                    if row["scene"] == scene and row["method"] == method and int(row["iterations"]) == 300
                ),
                float("nan"),
            )
            budget_rows.append(
                {
                    "scene": scene,
                    "method": method,
                    "train_count": train_count,
                    "trackmobile_scaled_30k_min": round(track_scaled, 2),
                    "smoke300_linear_30k_min": round(smoke_300, 2),
                    "recommended_budget_30k_min": round(max(track_scaled, smoke_300) if math.isfinite(smoke_300) else track_scaled, 2),
                    "estimated_disk_gb_from_inventory": inventory_rows[scene]["estimated_disk_gb"],
                }
            )
    write_csv(
        REPORT_DIR / "LONG_RUN_BUDGET.csv",
        budget_rows,
        [
            "scene",
            "method",
            "train_count",
            "trackmobile_scaled_30k_min",
            "smoke300_linear_30k_min",
            "recommended_budget_30k_min",
            "estimated_disk_gb_from_inventory",
        ],
    )
    full_min = 3 * sum(float(row["recommended_budget_30k_min"]) for row in budget_rows)
    staged_min = 2 * sum(float(row["recommended_budget_30k_min"]) for row in budget_rows)
    md = [
        "# LONG_RUN_BUDGET",
        "",
        "Budget uses two references: measured Trackmobile 30k time scaled by sqrt(train image count), and a conservative linear extrapolation from the 300-iteration smoke. The recommended budget per case is the larger of the two.",
        "",
        "| scene | method | train | Trackmobile-scaled 30k min | smoke300-linear 30k min | recommended 30k min |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in budget_rows:
        md.append(
            f"| {row['scene']} | {row['method']} | {row['train_count']} | {row['trackmobile_scaled_30k_min']} | "
            f"{row['smoke300_linear_30k_min']} | {row['recommended_budget_30k_min']} |"
        )
    md.extend(
        [
            "",
            "## Scheme Cost",
            "",
            f"- Scheme A, complete 3-run x 2 methods x 2 scenes: `{full_min / 60.0:.2f}` GPU hours.",
            f"- Scheme B, preregistered 2-run screening: `{staged_min / 60.0:.2f}` GPU hours before triggered third runs.",
            f"- Worst case for Scheme B, if every scene/method comparison triggers third runs: `{full_min / 60.0:.2f}` GPU hours.",
            "",
            "## Recommendation",
            "",
            "Recommend Scheme B. It preserves a preregistered trigger rule while reducing expected cost, and any third-run trigger applies symmetrically to both methods for the affected scene.",
        ]
    )
    write_text(REPORT_DIR / "LONG_RUN_BUDGET.md", "\n".join(md) + "\n")
    decision = [
        "# NEXT_EXPERIMENT_DECISION",
        "",
        "Recommendation: `B`, preregistered 2-run screening with symmetric third-run trigger.",
        "",
        "Trigger the third run for both methods on a scene if any metric interval overlaps, if the two-run direction is inconsistent, or if either method has two-run PSNR spread greater than 0.3 dB.",
        "",
        "Do not use smoke metrics for method selection; smoke results in this package are only compatibility checks.",
    ]
    write_text(REPORT_DIR / "NEXT_EXPERIMENT_DECISION.md", "\n".join(decision) + "\n")


def smoke_audit_report(
    split_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    checksum_rows: list[dict[str, object]],
) -> None:
    write_csv(REPORT_DIR / "TWO_SCENE_SMOKE_RESULTS.csv", summary_rows)
    all_smoke_pass = all(
        int(row["train_returncode"]) == 0
        and int(row["render_returncode"]) == 0
        and int(row["render_count"]) == int(row["gt_count"])
        and bool(row["all_metrics_finite"])
        and bool(row["no_black_images"])
        for row in summary_rows
    )
    all_split_pass = all(row["status"] == "PASS" for row in split_rows)
    all_gt_match = all(bool(row["match"]) for row in checksum_rows)
    md = [
        "# TWO_SCENE_SMOKE_AUDIT",
        "",
        f"- Overall split preflight: `{'PASS' if all_split_pass else 'FAIL'}`",
        f"- Overall smoke: `{'PASS' if all_smoke_pass else 'FAIL'}`",
        f"- Official vs GS-W GT decoded-pixel checksums: `{'PASS' if all_gt_match else 'FAIL'}`",
        f"- Run root: `{SMOKE_RUN_ROOT}`",
        "",
        "## Smoke Summary",
        "",
        "| role | scene | method | iter | train min | render min | renders/gt | PSNR | SSIM | LPIPS | finite | black | peak MB |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        md.append(
            f"| {row['role']} | {row['scene']} | {row['method']} | {row['iterations']} | "
            f"{row['train_duration_min']} | {row['render_duration_min']} | {row['render_count']}/{row['gt_count']} | "
            f"{float(row['psnr']):.6f} | {float(row['ssim']):.6f} | {float(row['lpips']):.6f} | "
            f"{row['finite_metric_views']} | {row['black_render_views']} | {row['train_peak_gpu_mem_mb']} |"
        )
    md.extend(
        [
            "",
            "## Leakage Controls",
            "",
            "- Scene selection used train-only lighting statistics; no test RGB, historical metrics, method deltas, or qualitative test renders were used.",
            "- GS-W smoke uses `--test_appearance_mode strict_intrinsic` for both training-time test hooks and explicit render.",
            "- GS-W renders use `--render_output_tag strict_intrinsic`; output directories are `ours_<iter>_strict_intrinsic`.",
            "- Official smoke uses clean official 3DGS `--eval`; split equivalence is checked against the frozen manifests.",
            "- `--train_test_exp` is absent, half-image metrics are not run, and evaluation is full-image RGB.",
            "- Official smoke sets `--optimizer_type default`, `--depth_l1_weight_init 0`, and `--depth_l1_weight_final 0`; no antialiasing flag is used.",
            "",
            "## Commands",
            "",
        ]
    )
    for row in summary_rows:
        md.append(f"### {row['role']} {row['method']} iter {row['iterations']} train")
        md.extend(["", "```text", str(row["train_command"]), "```", ""])
        md.append(f"### {row['role']} {row['method']} iter {row['iterations']} render")
        md.extend(["", "```text", str(row["render_command"]), "```", ""])
    write_text(REPORT_DIR / "TWO_SCENE_SMOKE_AUDIT.md", "\n".join(md) + "\n")


def main() -> int:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    selected = selected_scenes()
    split_rows = []
    for role, scene in selected.items():
        split_rows.append(split_preflight(scene, role))
    write_csv(REPORT_DIR / "TWO_SCENE_SPLIT_PREFLIGHT_SUMMARY.csv", split_rows)

    smoke_rows: list[dict[str, object]] = []
    for role, scene in selected.items():
        manifest_path = report_manifest_path(scene)
        for method in ["official_3dgs", "gsw_strict_intrinsic"]:
            for iteration in [10, 300]:
                started = datetime.now().isoformat(timespec="seconds")
                print(json.dumps({"event": "start_smoke", "role": role, "scene": scene, "method": method, "iteration": iteration, "time": started}))
                smoke_rows.append(run_smoke_case(scene, role, method, iteration, manifest_path))
                print(json.dumps({"event": "done_smoke", "role": role, "method": method, "iteration": iteration}))

    summary_rows, per_view_rows = evaluate_outputs(smoke_rows)
    write_csv(REPORT_DIR / "TWO_SCENE_SMOKE_PER_VIEW.csv", per_view_rows)
    checksum_rows = gt_checksum_audit(summary_rows, per_view_rows)
    smoke_audit_report(split_rows, summary_rows, checksum_rows)
    long_run_budget(summary_rows)
    print(json.dumps({"split": split_rows, "smoke_cases": len(summary_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
