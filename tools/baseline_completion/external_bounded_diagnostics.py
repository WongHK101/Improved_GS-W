from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
WL3DGS = REPO.parent
RUN = WL3DGS / "3dgs_runs" / "external_baselines_20260620"
SUMMARY = RUN / "summary"
LOG_DIR = RUN / "logs"
ADAPTERS = RUN / "adapters"
REPORT = REPO / "reports" / "baseline_completion"
LUMINANCE_REPO = WL3DGS / "external_baselines" / "Luminance-GS"
WILD_REPO = WL3DGS / "external_baselines" / "wild-gaussians"


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(path: Path) -> dict[str, str]:
    return {
        "commit": run(["git", "rev-parse", "HEAD"], path).stdout.strip(),
        "status": run(["git", "status", "--short"], path).stdout.strip().replace("\n", "; "),
        "remote": run(["git", "remote", "-v"], path).stdout.strip().replace("\n", "; "),
    }


def tail_text(path: Path, max_bytes: int = 20000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    chunk = data[-max_bytes:]
    if chunk.count(b"\x00") > max(8, len(chunk) // 20):
        text = chunk.decode("utf-16", errors="replace")
    else:
        text = chunk.decode("utf-8-sig", errors="replace")
    return text.replace("\x00", "")


def classify_log(text: str) -> dict[str, Any]:
    low = text.lower()
    patterns = {
        "oom": ["out of memory", "cuda error: out of memory", "oom"],
        "nan": ["nan", "loss=nan"],
        "missing_file": ["no such file", "filenotfounderror", "does not exist", "missing"],
        "assertion": ["assertionerror", "assert "],
        "cuda_compile": ["ninja", "nvcc", "cl.exe", "cuda"],
        "traceback": ["traceback (most recent call last)"],
    }
    row = {key: any(token in low for token in tokens) for key, tokens in patterns.items()}
    last_lines = [line for line in text.splitlines() if line.strip()][-25:]
    row["tail"] = " | ".join(last_lines)[-3000:]
    return row


def image_stats(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return {
            "image_path": str(path),
            "image_min": float(arr.min()),
            "image_max": float(arr.max()),
            "image_mean": float(arr.mean()),
            "image_std": float(arr.std()),
            "black_pixel_ratio": float((arr.max(axis=2) < 0.02).mean()),
            "finite": bool(np.isfinite(arr).all()),
            "image_sha256": file_sha256(path),
        }
    except Exception as exc:
        return {"image_path": str(path), "image_error": str(exc)}


def count_glob(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def luminance_adapter_audit(scene: str) -> dict[str, Any]:
    root = ADAPTERS / "luminance_gs" / scene
    transforms_train = root / "transforms_train.json"
    transforms_test = root / "transforms_test.json"
    row: dict[str, Any] = {
        "method": "Luminance-GS",
        "scene": scene,
        "adapter_path": str(root),
        "adapter_exists": root.exists(),
        "transforms_train_exists": transforms_train.exists(),
        "transforms_test_exists": transforms_test.exists(),
        "colmap_sparse_exists": (root / "colmap_sparse").exists() or (root / "colmap_sparse" / "0").exists(),
        "low_dir_exists": (root / "low").exists(),
        "high_dir_exists": (root / "high").exists(),
        "image_dir_exists": (root / "images").exists(),
    }
    for label, path in [("train", transforms_train), ("test", transforms_test)]:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                frames = payload.get("frames", [])
                row[f"{label}_frames"] = len(frames)
                row[f"{label}_first_file_path"] = frames[0].get("file_path", "") if frames else ""
            except Exception as exc:
                row[f"{label}_json_error"] = str(exc)
    return row


def luminance_run_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    failures = [row for row in read_csv(SUMMARY / "failures.csv") if row.get("method") == "luminance_gs"]
    metrics = [row for row in read_csv(SUMMARY / "metrics_all_methods.csv") if row.get("method") == "luminance_gs"]
    scenes = sorted({row.get("scene", "") for row in failures + metrics if row.get("scene")})
    adapter_rows = [luminance_adapter_audit(scene) for scene in scenes]
    run_rows: list[dict[str, Any]] = []
    for scene in scenes:
        metric = next((row for row in metrics if row.get("scene") == scene), {})
        failure_rows = [row for row in failures if row.get("scene") == scene]
        log_path = Path(failure_rows[-1]["log_path"]) if failure_rows else LOG_DIR / f"luminance_gs_r1_{scene}_train_render_metrics.log"
        text = tail_text(log_path)
        log_class = classify_log(text)
        output = Path(metric.get("output_path") or (RUN / "luminance_gs_r1_iter30000" / scene))
        render_candidates = list((output / "renders").glob("*.png")) if (output / "renders").exists() else []
        stats = image_stats(render_candidates[0]) if render_candidates else {}
        run_rows.append(
            {
                "method": "Luminance-GS",
                "scene": scene,
                "metric_available": bool(metric),
                "psnr": metric.get("psnr", ""),
                "ssim": metric.get("ssim", ""),
                "lpips": metric.get("lpips", ""),
                "failure_count": len(failure_rows),
                "last_failure_note": failure_rows[-1].get("note", "") if failure_rows else "",
                "log_path": str(log_path),
                "output_path": str(output),
                "ckpt_count": count_glob(output / "ckpts", "*.pt") + count_glob(output / "ckpts", "*.pth"),
                "render_count": count_glob(output / "renders", "*.png"),
                "stats_file_count": count_glob(output / "stats", "*"),
                **log_class,
                **stats,
            }
        )
    return adapter_rows, run_rows


def wild_adapter_audit(scene: str) -> dict[str, Any]:
    root = ADAPTERS / "wildgaussians" / scene
    files = {p.name for p in root.iterdir()} if root.exists() else set()
    row: dict[str, Any] = {
        "method": "WildGaussians",
        "scene": scene,
        "adapter_path": str(root),
        "adapter_exists": root.exists(),
        "train_list_exists": "train_list.txt" in files,
        "test_list_exists": "test_list.txt" in files,
        "images_exists": (root / "images").exists(),
        "sparse_exists": (root / "sparse").exists(),
        "sparse_0_exists": (root / "sparse" / "0").exists(),
    }
    for name in ["train_list.txt", "test_list.txt"]:
        path = root / name
        if path.exists():
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            row[name.replace(".txt", "_count")] = len(lines)
            row[name.replace(".txt", "_first")] = lines[0] if lines else ""
    return row


def wild_run_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = [row for row in read_csv(SUMMARY / "metrics_all_methods.csv") if row.get("method") == "wildgaussians"]
    scenes = sorted({row.get("scene", "") for row in metrics if row.get("scene")})
    adapter_rows = [wild_adapter_audit(scene) for scene in scenes]
    run_rows: list[dict[str, Any]] = []
    for metric in metrics:
        scene = metric.get("scene", "")
        output = Path(metric.get("output_path") or (RUN / "wildgaussians_r1_iter30000" / scene))
        log_path = LOG_DIR / f"wildgaussians_r1_{scene}_train_render_metrics.log"
        text = tail_text(log_path)
        log_class = classify_log(text)
        image_candidates = list(output.rglob("*.png"))
        pred_candidates = [p for p in image_candidates if "color" in str(p).lower() or "render" in str(p).lower()]
        chosen = pred_candidates[0] if pred_candidates else (image_candidates[0] if image_candidates else None)
        stats = image_stats(chosen) if chosen else {}
        psnr = float(metric.get("psnr", "nan"))
        run_rows.append(
            {
                "method": "WildGaussians",
                "scene": scene,
                "psnr": metric.get("psnr", ""),
                "ssim": metric.get("ssim", ""),
                "lpips": metric.get("lpips", ""),
                "psnr_lt_10": math.isfinite(psnr) and psnr < 10,
                "log_path": str(log_path),
                "output_path": str(output),
                "checkpoint_count": count_glob(output, "checkpoint-*"),
                "png_count_recursive": len(image_candidates),
                **log_class,
                **stats,
            }
        )
    return adapter_rows, run_rows


def code_evidence() -> list[dict[str, Any]]:
    items = [
        (LUMINANCE_REPO / "Luminance-GS" / "examples" / "simple_trainer_ours.py", ["curve", "valset", "render_traj", "image_ids", "app_opt"]),
        (LUMINANCE_REPO / "Luminance-GS" / "examples" / "datasets" / "colmap.py", ["transforms_train", "transforms_test", "high", "low", "split"]),
        (WILD_REPO / "wildgaussians" / "datasets" / "phototourism.py", ["horizontal_half_dataset", "optimize_embedding", "w//2", "compute_metrics"]),
        (WILD_REPO / "wildgaussians" / "utils.py", ["image_to_srgb", "black background", "linear_to_srgb"]),
        (WILD_REPO / "wildgaussians" / "datasets" / "colmap.py", ["train_list", "test_list", "Indices.every_iters"]),
    ]
    rows = []
    for path, needles in items:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        lines = text.splitlines()
        for needle in needles:
            hits = []
            for idx, line in enumerate(lines, start=1):
                if needle.lower() in line.lower():
                    hits.append(f"{idx}:{line.strip()}")
            rows.append({"file": str(path), "needle": needle, "hits": " | ".join(hits[:8]), "hit_count": len(hits)})
    return rows


def smoke_summary() -> list[dict[str, Any]]:
    log_dir = REPORT / "logs" / "external_bounded_diagnostics"
    rows = []
    for name in ["luminance_import_smoke.log", "luminance_pycolmap_probe.log", "wild_import_smoke.log", "wild_reader_smoke.log"]:
        path = log_dir / name
        text = tail_text(path, 12000)
        rows.append(
            {
                "log": str(path),
                "exists": path.exists(),
                "contains_error": any(token in text.lower() for token in ["traceback", "error", "importerror", "failed"]),
                "contains_success": any(token in text for token in ["wild_import_ok", '"has_SceneManager": false', '"scene": "web_doss_images"', '"scene": "web_Baalshamin_images"']),
                "tail": " | ".join([line for line in text.splitlines() if line.strip()][-20:])[-3000:],
            }
        )
    return rows


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    lum_adapter, lum_runs = luminance_run_audit()
    wild_adapter, wild_runs = wild_run_audit()
    evidence = code_evidence()
    write_csv(REPORT / "LUMINANCE_GS_BOUNDED_DIAGNOSTIC_ADAPTER.csv", lum_adapter)
    write_csv(REPORT / "LUMINANCE_GS_BOUNDED_DIAGNOSTIC_RUNS.csv", lum_runs)
    write_csv(REPORT / "WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_ADAPTER.csv", wild_adapter)
    write_csv(REPORT / "WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_RUNS.csv", wild_runs)
    write_csv(REPORT / "EXTERNAL_BASELINE_CODE_EVIDENCE.csv", evidence)
    lum_info = git_info(LUMINANCE_REPO)
    wild_info = git_info(WILD_REPO)
    smoke_rows = smoke_summary()
    write_csv(REPORT / "EXTERNAL_BASELINE_SMOKE_RESULTS.csv", smoke_rows)
    wild_dark = [row for row in wild_runs if row.get("psnr_lt_10") is True]
    lum_import = next((row for row in smoke_rows if row["log"].endswith("luminance_import_smoke.log")), {})
    pycolmap_probe = next((row for row in smoke_rows if row["log"].endswith("luminance_pycolmap_probe.log")), {})
    wild_import = next((row for row in smoke_rows if row["log"].endswith("wild_import_smoke.log")), {})
    wild_reader = next((row for row in smoke_rows if row["log"].endswith("wild_reader_smoke.log")), {})
    md = [
        "# EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS",
        "",
        "No full 30k reruns were launched by this diagnostic. It audits existing adapters, logs, checkpoints and render samples.",
        "",
        "## Luminance-GS",
        "",
        f"- Commit: `{lum_info['commit']}`.",
        f"- Local status: `{lum_info['status'] or 'clean'}`.",
        f"- Scenes inspected: `{len(lum_adapter)}`.",
        f"- Rows with historical failures: `{sum(1 for row in lum_runs if int(row.get('failure_count') or 0) > 0)}`.",
        f"- Import smoke contains error: `{lum_import.get('contains_error', '')}`.",
        f"- PyCOLMAP probe: `{pycolmap_probe.get('tail', '')}`.",
        "- Adapter evidence: `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_ADAPTER.csv`.",
        "- Run/log evidence: `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_RUNS.csv`.",
        "",
        "Interpretation: Luminance-GS remains unresolved for strict main comparison. Historical logs and adapter state should be treated as wrapper/config evidence, not as a fair complete method result.",
        "",
        "## WildGaussians",
        "",
        f"- Commit: `{wild_info['commit']}`.",
        f"- Local status: `{wild_info['status'] or 'clean'}`.",
        f"- Scenes inspected: `{len(wild_adapter)}`.",
        f"- PSNR<10 rows: `{len(wild_dark)}` / `{len(wild_runs)}`.",
        f"- Import smoke passed: `{wild_import.get('contains_success', '')}`.",
        f"- Reader smoke evidence: `{wild_reader.get('tail', '')}`.",
        "- Adapter evidence: `WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_ADAPTER.csv`.",
        "- Run/log/image-stat evidence: `WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_RUNS.csv`.",
        "",
        "Interpretation: current WildGaussians numeric rows should remain invalid/removed from the strict main table. The evidence supports an integration/protocol problem requiring wrapper-level diagnosis before any full rerun.",
        "",
        "## Code Evidence",
        "",
        "- `EXTERNAL_BASELINE_CODE_EVIDENCE.csv` records source locations for curve/test split handling, phototourism half-image evaluation and color/background conversion.",
        "",
    ]
    write_text(REPORT / "EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md", "\n".join(md))
    print(REPORT / "EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
