from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from official_control_common import (
    GSW_RUN_ROOT,
    REPORT_DIR,
    RUN_ROOT,
    SCENE_NAME,
    git_output,
    write_csv,
    write_text,
)
from run_official_control import checkpoint_size, parse_gaussian_count


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": sample_std(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def by_group(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    return [row for row in rows if row["group"] == group]


def overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return max(a_min, b_min) <= min(a_max, b_max)


def train_meta_for_official(label: str) -> dict[str, object]:
    model_path = RUN_ROOT / label / SCENE_NAME
    raw_rows = []
    raw_path = REPORT_DIR / "OFFICIAL_3DGS_REPEATED_SUMMARY_RAW.csv"
    if raw_path.exists():
        raw_rows = [row for row in read_csv(raw_path) if row.get("label") == label and row.get("stage") == "train"]
    base = raw_rows[0] if raw_rows else {}
    return {
        "model_path": str(model_path),
        "gaussian_count": base.get("gaussian_count") or parse_gaussian_count(model_path, 30000),
        "train_time_sec": base.get("duration_sec", ""),
        "train_time_min": base.get("duration_min", ""),
        "peak_gpu_mem_mb": base.get("peak_gpu_mem_mb", ""),
        "checkpoint_size_bytes": base.get("checkpoint_size_bytes") or checkpoint_size(model_path, 30000),
    }


def gsw_meta(label: str) -> dict[str, object]:
    summary = REPORT_DIR.parent / "gsw_repeated_30k_baseline" / "GSW_REPEATED_30K_SUMMARY.csv"
    if summary.exists():
        for row in read_csv(summary):
            if row.get("run") == label and row.get("mode") == "strict_intrinsic":
                return {
                    "model_path": row.get("model_path", ""),
                    "gaussian_count": row.get("gaussian_count", ""),
                    "train_time_sec": row.get("train_time_sec", ""),
                    "train_time_min": row.get("train_time_min", ""),
                    "peak_gpu_mem_mb": row.get("peak_gpu_mem_mb", ""),
                    "checkpoint_size_bytes": row.get("checkpoint_size_bytes", ""),
                }
    return {}


def main() -> int:
    unified = REPORT_DIR / "UNIFIED_OFFICIAL_GSW_RESULTS.csv"
    if not unified.exists():
        raise FileNotFoundError(unified)
    rows = read_csv(unified)
    official = by_group(rows, "official_3dgs")
    gsw = by_group(rows, "gsw_strict_intrinsic")
    if not official or not gsw:
        raise RuntimeError("Need both official_3dgs and gsw_strict_intrinsic rows.")

    official_summary_rows = []
    for row in official:
        meta = train_meta_for_official(row["label"])
        official_summary_rows.append({**row, **meta})
    write_csv(REPORT_DIR / "OFFICIAL_3DGS_REPEATED_SUMMARY.csv", official_summary_rows)

    stat_rows = []
    for group_name, group_rows in [("official_3dgs", official), ("gsw_strict_intrinsic", gsw)]:
        for metric in ["psnr", "ssim", "lpips"]:
            values = [float(row[metric]) for row in group_rows]
            stat_rows.append({"group": group_name, "metric": metric, **stats(values)})
    write_csv(REPORT_DIR / "OFFICIAL_3DGS_REPEATED_STATISTICS.csv", stat_rows)

    official_stats = {metric: stats([float(row[metric]) for row in official]) for metric in ["psnr", "ssim", "lpips"]}
    gsw_stats = {metric: stats([float(row[metric]) for row in gsw]) for metric in ["psnr", "ssim", "lpips"]}
    delta = {metric: gsw_stats[metric]["mean"] - official_stats[metric]["mean"] for metric in ["psnr", "ssim", "lpips"]}
    covered = {
        metric: overlap(
            official_stats[metric]["min"],
            official_stats[metric]["max"],
            gsw_stats[metric]["min"],
            gsw_stats[metric]["max"],
        )
        for metric in ["psnr", "ssim", "lpips"]
    }

    recommendation = "B"
    recommendation_reason = (
        "Trackmobile alone should not decide the long-term base if metrics are mixed or near run-to-run variance; "
        "verify one stronger appearance-variation scene and one ordinary scenic scene next."
    )
    if (
        delta["psnr"] > max(official_stats["psnr"]["range"], gsw_stats["psnr"]["range"])
        or delta["ssim"] > max(official_stats["ssim"]["range"], gsw_stats["ssim"]["range"])
    ) and not (delta["lpips"] > 0 and abs(delta["lpips"]) > max(official_stats["lpips"]["range"], gsw_stats["lpips"]["range"])):
        recommendation = "A"
        recommendation_reason = "GS-W has a stable quality advantage without a stable LPIPS penalty exceeding run-to-run variation."
    if delta["psnr"] < 0 and delta["ssim"] < 0 and delta["lpips"] > 0:
        recommendation = "C"
        recommendation_reason = "Official is better on PSNR/SSIM and GS-W is worse on LPIPS."

    md_stats = [
        "# OFFICIAL_3DGS_REPEATED_STATISTICS",
        "",
        "| group | metric | mean | sample std | min | max | range |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in stat_rows:
        md_stats.append(
            f"| {row['group']} | {row['metric']} | {row['mean']:.6f} | {row['sample_std']:.6f} | {row['min']:.6f} | {row['max']:.6f} | {row['range']:.6f} |"
        )
    write_text(REPORT_DIR / "OFFICIAL_3DGS_REPEATED_STATISTICS.md", "\n".join(md_stats) + "\n")

    analysis = [
        "# OFFICIAL_VS_GSW_ANALYSIS",
        "",
        "## Mean Delta",
        "",
        "Delta is `GS-W strict_intrinsic mean - official 3DGS mean`.",
        "",
        f"- PSNR delta: `{delta['psnr']:.6f}`",
        f"- SSIM delta: `{delta['ssim']:.6f}`",
        f"- LPIPS delta: `{delta['lpips']:.6f}` (positive means GS-W is worse)",
        "",
        "## Range Overlap",
        "",
        f"- PSNR min/max overlap: `{covered['psnr']}`",
        f"- SSIM min/max overlap: `{covered['ssim']}`",
        f"- LPIPS min/max overlap: `{covered['lpips']}`",
        "",
        "## Notes",
        "",
        "- O1/O2/O3 and R1/R2/R3 are repeated runs, not paired deterministic seeds.",
        "- Efficiency comparisons are descriptive only because official and GS-W differ in implementation and extra network components.",
        "- GS-W uses `data_device=cuda`; official uses the official default `data_device=cuda` in this control.",
    ]
    write_text(REPORT_DIR / "OFFICIAL_VS_GSW_ANALYSIS.md", "\n".join(analysis) + "\n")

    final = [
        "# FINAL_BASE_SELECTION",
        "",
        f"Recommendation: `{recommendation}`",
        "",
        recommendation_reason,
        "",
        "If recommendation is B, the next two scenes should be selected as:",
        "",
        "1. A high appearance-variation/self-captured outdoor scene, to stress exposure/shadow robustness.",
        "2. A comparatively ordinary scenic scene, to check whether any GS-W/official trend is scene-specific.",
        "",
        "No new algorithm should be implemented before this base-choice uncertainty is resolved.",
    ]
    write_text(REPORT_DIR / "FINAL_BASE_SELECTION.md", "\n".join(final) + "\n")

    write_text(
        REPORT_DIR / "OFFICIAL_VS_GSW_ANALYSIS.json",
        json.dumps(
            {
                "official_stats": official_stats,
                "gsw_stats": gsw_stats,
                "delta_gsw_minus_official": delta,
                "range_overlap": covered,
                "recommendation": recommendation,
                "recommendation_reason": recommendation_reason,
            },
            indent=2,
        )
        + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

