from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from scene_selection_common import (
    DATA_ROOT,
    IMPROVED_ROOT,
    OFFICIAL_ROOT,
    REPORT_DIR,
    SPLIT_ROOT,
    TRACKMOBILE,
    ensure_dirs,
    load_colmap_counts,
    load_split_names,
    write_report_manifest,
    run_cmd,
    scene_category,
    scene_names,
    sha256_file,
    source_type,
    split_files,
    write_csv,
    write_text,
)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def official_reader_ok(scene: str) -> tuple[bool, str]:
    code = (
        "import sys,json;"
        f"sys.path.insert(0,{str(OFFICIAL_ROOT)!r});"
        "from scene.dataset_readers import readColmapSceneInfo;"
        f"info=readColmapSceneInfo({str(DATA_ROOT / scene)!r},'images','',True,False);"
        "print(json.dumps({'train':len(info.train_cameras),'test':len(info.test_cameras)}))"
    )
    result = run_cmd([sys.executable, "-c", code], cwd=OFFICIAL_ROOT, timeout=120)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
        return data["train"] > 0 and data["test"] > 0, f"{data['train']}/{data['test']}"
    except Exception as exc:
        return False, f"{repr(exc)} stdout={result.stdout!r}"


def gsw_reader_ok(scene: str) -> tuple[bool, str]:
    manifest_path = write_report_manifest(scene)
    code = (
        "import json;"
        "from scene.dataset_readers import readColmapSceneInfo;"
        f"info=readColmapSceneInfo({str(DATA_ROOT / scene)!r},'images',False,split_mode='frozen_manifest',split_file={str(manifest_path)!r});"
        "print(json.dumps({'train':len(info.train_cameras),'test':len(info.test_cameras)}))"
    )
    result = run_cmd([sys.executable, "-c", code], cwd=IMPROVED_ROOT, timeout=120)
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
        return data["train"] > 0 and data["test"] > 0, f"{data['train']}/{data['test']}"
    except Exception as exc:
        return False, f"{repr(exc)} stdout={result.stdout!r}"


def luminance_stats_for_image(path: Path) -> dict[str, float]:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB").resize((256, max(1, round(img.height * 256 / img.width))), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    log_mean = float(np.log(np.clip(lum, 1e-4, 1.0)).mean())
    sat = float((arr >= 0.98).any(axis=2).mean())
    dark = float((lum <= 0.02).mean())
    denom = np.clip(arr.sum(axis=2), 1e-6, None)
    chroma = arr / denom[..., None]
    mean_r, mean_g, mean_b = [float(chroma[..., idx].mean()) for idx in range(3)]
    # Deterministic low-frequency proxy: resize luminance to 32x32 and use p90-p10.
    low = np.asarray(Image.fromarray(np.clip(lum * 255.0, 0, 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    low_range = float(np.percentile(low, 90) - np.percentile(low, 10))
    return {
        "mean_log_luminance": log_mean,
        "saturation_ratio": sat,
        "dark_ratio": dark,
        "mean_chroma_r": mean_r,
        "mean_chroma_g": mean_g,
        "mean_chroma_b": mean_b,
        "lowfreq_luminance_range": low_range,
    }


def mad(values: list[float]) -> float:
    med = statistics.median(values)
    return statistics.median([abs(v - med) for v in values])


def p90_p10(values: list[float]) -> float:
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def spread(values: list[float]) -> float:
    return p90_p10(values)


def scene_lighting(scene: str, train_names: list[str]) -> dict[str, object]:
    rows = []
    for name in train_names:
        path = DATA_ROOT / scene / "images" / name
        row = {"scene": scene, "image_name": name, **luminance_stats_for_image(path)}
        rows.append(row)
    log_lums = [float(r["mean_log_luminance"]) for r in rows]
    sat = [float(r["saturation_ratio"]) for r in rows]
    dark = [float(r["dark_ratio"]) for r in rows]
    chroma_spreads = []
    for key in ["mean_chroma_r", "mean_chroma_g", "mean_chroma_b"]:
        chroma_spreads.append(p90_p10([float(r[key]) for r in rows]))
    low = [float(r["lowfreq_luminance_range"]) for r in rows]
    return {
        "scene": scene,
        "train_image_count": len(train_names),
        "log_luminance_std": statistics.stdev(log_lums) if len(log_lums) > 1 else 0.0,
        "log_luminance_mad": mad(log_lums),
        "log_luminance_p90_p10": p90_p10(log_lums),
        "saturation_ratio_spread": spread(sat),
        "dark_ratio_spread": spread(dark),
        "chromaticity_spread": float(sum(chroma_spreads) / len(chroma_spreads)),
        "lowfreq_illumination_spread": spread(low),
        "per_image_rows": rows,
    }


def rank_percentiles(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    sorted_rows = sorted(rows, key=lambda row: float(row[key]))
    n = len(sorted_rows)
    out = {}
    for idx, row in enumerate(sorted_rows):
        out[str(row["scene"])] = 0.0 if n == 1 else idx / (n - 1)
    return out


def build_inventory() -> list[dict[str, object]]:
    rows = []
    for scene in scene_names(include_trackmobile=False):
        scene_path = DATA_ROOT / scene
        all_names, train, test, meta = load_split_names(scene)
        first = scene_path / "images" / all_names[0]
        width, height = image_size(first)
        official_ok, official_note = official_reader_ok(scene)
        gsw_ok, gsw_note = gsw_reader_ok(scene)
        colmap = load_colmap_counts(scene_path)
        split_dir = SPLIT_ROOT / scene
        row = {
            "scene": scene,
            "source_type": source_type(scene),
            "scene_category": scene_category(scene),
            "registered_images": len(all_names),
            "frozen_train_count": len(train),
            "frozen_test_count": len(test),
            "original_resolution": f"{width}x{height}",
            "resolution1_actual_size": f"{width}x{height}",
            "colmap_points": colmap.get("colmap_points", ""),
            "data_path": str(scene_path),
            "frozen_split_manifest_exists": split_dir.exists(),
            "split_meta_sha256": sha256_file(split_dir / "split_meta.json") if (split_dir / "split_meta.json").exists() else "",
            "official_reader_loads": official_ok,
            "official_reader_note": official_note,
            "gsw_reader_loads": gsw_ok,
            "gsw_reader_note": gsw_note,
            "copyright_or_integrity_issue": "source category inferred from scene prefix; detailed license file not found in local dataset tree",
            "estimated_official_30k_time_min": round(17.4 * max(1.0, len(train) / 13.0) ** 0.5, 2),
            "estimated_gsw_30k_time_min": round(91.9 * max(1.0, len(train) / 13.0) ** 0.5, 2),
            "estimated_disk_gb": round((0.5 + len(train) * 0.002), 2),
        }
        rows.append(row)
    return rows


def main() -> int:
    ensure_dirs()
    inventory = build_inventory()
    write_csv(REPORT_DIR / "SCENE_CANDIDATE_INVENTORY.csv", inventory)

    diff_rows = []
    per_image_rows = []
    for row in inventory:
        scene = str(row["scene"])
        _, train, _, _ = load_split_names(scene)
        lighting = scene_lighting(scene, train)
        per_image_rows.extend(lighting.pop("per_image_rows"))
        diff_rows.append({**{k: row[k] for k in ["scene", "source_type", "scene_category"]}, **lighting})
    write_csv(REPORT_DIR / "TRAIN_ONLY_LIGHTING_PER_IMAGE.csv", per_image_rows)

    components = [
        "log_luminance_p90_p10",
        "saturation_ratio_spread",
        "dark_ratio_spread",
        "chromaticity_spread",
        "lowfreq_illumination_spread",
    ]
    ranks = {component: rank_percentiles(diff_rows, component) for component in components}
    for row in diff_rows:
        scene = str(row["scene"])
        score = 0.0
        for component in components:
            value = ranks[component][scene]
            row[f"{component}_rank_percentile"] = value
            score += value
        row["lighting_difficulty_score"] = score / len(components)
    diff_rows.sort(key=lambda row: float(row["lighting_difficulty_score"]), reverse=True)
    for idx, row in enumerate(diff_rows, start=1):
        row["difficulty_rank"] = idx
    write_csv(REPORT_DIR / "TRAIN_ONLY_LIGHTING_DIFFICULTY.csv", diff_rows)

    self_candidates = [row for row in diff_rows if row["source_type"] == "self-captured"]
    valid = {row["scene"]: row for row in inventory if row["official_reader_loads"] and row["gsw_reader_loads"]}
    high = next(row for row in self_candidates if row["scene"] in valid)
    web_candidates = [row for row in diff_rows if row["source_type"] == "network/public scenic" and row["scene"] in valid]
    web_sorted = sorted(web_candidates, key=lambda row: float(row["lighting_difficulty_score"]))
    median_score = statistics.median([float(row["lighting_difficulty_score"]) for row in web_sorted])
    medium = min(web_sorted, key=lambda row: abs(float(row["lighting_difficulty_score"]) - median_score))

    selected = {
        "H": high,
        "M": medium,
        "web_median_score": median_score,
    }
    write_text(REPORT_DIR / "SELECTED_SCENES.json", json.dumps(selected, indent=2, ensure_ascii=False) + "\n")

    md = [
        "# TRAIN_ONLY_LIGHTING_DIFFICULTY",
        "",
        "All scores use train images only. No test RGB, test metrics, historical PSNR/SSIM/LPIPS, method deltas, or qualitative test outputs are used.",
        "",
        "Fixed thresholds: saturation if any RGB channel >= 0.98; dark if luminance <= 0.02. Luminance is sRGB relative luminance `Y = 0.2126 R + 0.7152 G + 0.0722 B`.",
        "",
        "| rank | scene | source | score | log-lum range | sat spread | dark spread | chroma spread | lowfreq spread |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diff_rows:
        md.append(
            f"| {row['difficulty_rank']} | {row['scene']} | {row['source_type']} | {float(row['lighting_difficulty_score']):.6f} | "
            f"{float(row['log_luminance_p90_p10']):.6f} | {float(row['saturation_ratio_spread']):.6f} | "
            f"{float(row['dark_ratio_spread']):.6f} | {float(row['chromaticity_spread']):.6f} | {float(row['lowfreq_illumination_spread']):.6f} |"
        )
    write_text(REPORT_DIR / "TRAIN_ONLY_LIGHTING_DIFFICULTY.md", "\n".join(md) + "\n")

    def split_list(scene: str) -> tuple[list[str], list[str]]:
        _, train, test, _ = load_split_names(scene)
        return train, test

    h_train, h_test = split_list(str(high["scene"]))
    m_train, m_test = split_list(str(medium["scene"]))
    selected_md = [
        "# SELECTED_SCENES",
        "",
        "Selection follows the pre-registered train-only rules.",
        "",
        "## Scene H: High Lighting Variation",
        "",
        f"- Scene: `{high['scene']}`",
        f"- Directory: `{DATA_ROOT / str(high['scene'])}`",
        f"- Difficulty rank: `{high['difficulty_rank']}`",
        f"- Score: `{float(high['lighting_difficulty_score']):.6f}`",
        f"- Components: log-lum `{float(high['log_luminance_p90_p10']):.6f}`, saturation `{float(high['saturation_ratio_spread']):.6f}`, dark `{float(high['dark_ratio_spread']):.6f}`, chroma `{float(high['chromaticity_spread']):.6f}`, lowfreq `{float(high['lowfreq_illumination_spread']):.6f}`",
        f"- Train/test counts: `{len(h_train)}/{len(h_test)}`",
        f"- Train names: `{';'.join(h_train)}`",
        f"- Test names: `{';'.join(h_test)}`",
        "",
        "Reason: highest valid self-captured candidate by train-only lighting difficulty, excluding Trackmobile.",
        "",
        "## Scene M: Medium/Public Scenic",
        "",
        f"- Scene: `{medium['scene']}`",
        f"- Directory: `{DATA_ROOT / str(medium['scene'])}`",
        f"- Difficulty rank: `{medium['difficulty_rank']}`",
        f"- Score: `{float(medium['lighting_difficulty_score']):.6f}`",
        f"- Public scenic median score: `{median_score:.6f}`",
        f"- Components: log-lum `{float(medium['log_luminance_p90_p10']):.6f}`, saturation `{float(medium['saturation_ratio_spread']):.6f}`, dark `{float(medium['dark_ratio_spread']):.6f}`, chroma `{float(medium['chromaticity_spread']):.6f}`, lowfreq `{float(medium['lowfreq_illumination_spread']):.6f}`",
        f"- Train/test counts: `{len(m_train)}/{len(m_test)}`",
        f"- Train names: `{';'.join(m_train)}`",
        f"- Test names: `{';'.join(m_test)}`",
        "",
        "Reason: valid network/public scenic scene with score closest to the public-scenic category median, not the easiest scene.",
        "",
        "## Adjacent Candidates",
        "",
        "See `TRAIN_ONLY_LIGHTING_DIFFICULTY.csv` and `SCENE_CANDIDATE_INVENTORY.csv` for adjacent scores and validity checks. No method result was used to replace the selected scenes.",
    ]
    write_text(REPORT_DIR / "SELECTED_SCENES.md", "\n".join(selected_md) + "\n")
    print(json.dumps({"selected_H": high["scene"], "selected_M": medium["scene"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
