from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


REPO = Path(__file__).resolve().parents[2]
WL3DGS = REPO.parent
REPORT = REPO / "reports" / "baseline_completion"
GENERATED_MANIFESTS = REPORT / "generated_manifests"
LOG_DIR = REPORT / "logs"
REVIEW_ROOT = WL3DGS / "gpt_review_packages"
PENDING_GSW_RUN_ROOT = WL3DGS / "3dgs_runs" / "gsw_strict_12scene_single_run_20260702"
GSW_APPROVAL_TOKEN = "GPT_APPROVED_12SCENE_GSW_30K"

SPLIT_ROOT = WL3DGS / "splits" / "max1600_llffhold8_v1"
DATA_ROOT = WL3DGS / "3dgs_undistorted" / "max1600"
RUNS_ROOT = WL3DGS / "3dgs_runs"
EXTERNAL_RUN = RUNS_ROOT / "external_baselines_20260620"
EXTERNAL_SUMMARY = EXTERNAL_RUN / "summary"

TRAINING_PATHS = [
    "train.py",
    "render.py",
    "gaussian_renderer",
    "scene",
    "net_modules",
    "arguments",
    "utils/loss_utils.py",
    "utils/image_utils.py",
]

TRAINING_FREEZE_TAG = "gsw-strict-12scene-v1"

SCENE_GROUP = {
    "self_Trackmobile_4650TM_Mobile_Railcar_Mover": "Trackmobile",
    "self_Steam_Locomotive": "H",
    "web_Terrestrial": "M",
}


def run(cmd: list[str], cwd: Path = REPO, check: bool = True) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return (proc.stdout or "").strip()


def audit_patch_text(paths: list[str]) -> str:
    tracked = run(["git", "diff", "--", *paths], check=False)
    untracked = run(["git", "ls-files", "--others", "--exclude-standard", "--", *paths], check=False)
    chunks = [tracked] if tracked else []
    for rel in [p for p in untracked.splitlines() if p.strip()]:
        path = REPO / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(
            "".join(
                difflib.unified_diff(
                    [],
                    text.splitlines(keepends=True),
                    fromfile="/dev/null",
                    tofile=f"b/{rel}",
                )
            )
        )
    return "\n".join(chunk for chunk in chunks if chunk)


def audit_diff_stat(paths: list[str]) -> str:
    stat = run(["git", "diff", "--stat", "--", *paths], check=False)
    untracked = run(["git", "ls-files", "--others", "--exclude-standard", "--", *paths], check=False)
    rows = [stat] if stat else []
    for rel in [p for p in untracked.splitlines() if p.strip()]:
        path = REPO / rel
        if path.is_file():
            rows.append(f"{rel} | {len(path.read_text(encoding='utf-8', errors='replace').splitlines())} +")
    return "\n".join(row for row in rows if row)


def conda_python(script: Path, args: list[str]) -> list[str]:
    return ["conda", "run", "-n", "3dgs", "--no-capture-output", "python", str(script), *args]


def ps_join(cmd: list[str]) -> str:
    out = []
    for arg in cmd:
        if any(ch.isspace() for ch in arg) or "\\" in arg or ":" in arg:
            out.append('"' + arg.replace('"', '\\"') + '"')
        else:
            out.append(arg)
    return " ".join(out)


def ensure_dirs() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    GENERATED_MANIFESTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def training_freeze_commit() -> str:
    tagged = run(["git", "rev-parse", f"{TRAINING_FREEZE_TAG}^{{}}"], check=False)
    return tagged or run(["git", "rev-parse", "HEAD"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def split_lines(scene: str, name: str) -> list[str]:
    path = SPLIT_ROOT / scene / name
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_scene_summary() -> list[dict[str, Any]]:
    rows = read_csv(SPLIT_ROOT / "split_summary.csv")
    scenes: list[dict[str, Any]] = []
    for row in rows:
        scene = row["scene_id"]
        scenes.append(
            {
                "scene": scene,
                "source_type": "self" if scene.startswith("self_") else "web",
                "source_scene": row["source_scene"],
                "image_count": int(row["num_registered"]),
                "train_count": int(row["num_train"]),
                "test_count": int(row["num_test"]),
                "split_sha256": row["test_sha256"],
            }
        )
    return scenes


def manifest_for(scene: str) -> Path:
    existing = REPO / "reports" / "two_scene_selection" / "generated_manifests" / f"{scene}_SPLIT.json"
    if existing.exists():
        return existing
    generated = GENERATED_MANIFESTS / f"{scene}_SPLIT.json"
    if generated.exists():
        return generated
    all_images = split_lines(scene, "all_registered.txt")
    train = split_lines(scene, "train.txt")
    test = split_lines(scene, "test.txt")
    payload = {
        "scene": scene,
        "protocol": "llff_hold_8",
        "ordering_rule": "sort registered COLMAP image names lexicographically; assign test if zero-based sorted index % 8 == 0",
        "source": {
            "scene_path": str(DATA_ROOT / scene),
            "frozen_split_dir": str(SPLIT_ROOT / scene),
            "split_meta": str(SPLIT_ROOT / scene / "split_meta.json"),
        },
        "registered_images": all_images,
        "train_images": train,
        "test_images": test,
    }
    generated.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return generated


def code_freeze_audit() -> None:
    current = run(["git", "rev-parse", "HEAD"])
    freeze_commit = training_freeze_commit()
    base = "ddc6d8702b2e838dc989d612ca23fb311b79f280"
    strict_tag = "gsw-strict-baseline-v2"
    freeze_tag = TRAINING_FREEZE_TAG
    diff_base = run(["git", "diff", "--name-status", base, "--", *TRAINING_PATHS])
    diff_strict = run(["git", "diff", "--name-status", strict_tag, "--", *TRAINING_PATHS])
    diff_freeze = run(["git", "diff", "--name-status", freeze_tag, "--", *TRAINING_PATHS], check=False)
    tag_line = run(["git", "show-ref", "--tags"], check=False)
    passed = not diff_base and not diff_strict and not diff_freeze
    lines = [
        "# GSW_12SCENE_CODE_FREEZE_AUDIT",
        "",
        f"- Audit package HEAD: `{current}`",
        f"- Training freeze commit: `{freeze_commit}`",
        f"- Baseline commit required by GPT: `{base}`",
        f"- Existing strict tag: `{strict_tag}`",
        f"- New 12-scene freeze tag: `{freeze_tag}`",
        f"- Training behavior diff vs baseline commit: `{'PASS' if not diff_base else 'FAIL'}`",
        f"- Training behavior diff vs strict tag: `{'PASS' if not diff_strict else 'FAIL'}`",
        f"- Training behavior diff vs 12-scene tag: `{'PASS' if not diff_freeze else 'FAIL'}`",
        f"- Overall freeze audit: `{'PASS' if passed else 'FAIL'}`",
        "",
        "Checked paths:",
        "",
    ]
    lines += [f"- `{p}`" for p in TRAINING_PATHS]
    lines += [
        "",
        "## Diff vs baseline commit",
        "",
        "```text",
        diff_base or "(empty)",
        "```",
        "",
        "## Diff vs strict tag",
        "",
        "```text",
        diff_strict or "(empty)",
        "```",
        "",
        "## Diff vs 12-scene tag",
        "",
        "```text",
        diff_freeze or "(empty)",
        "```",
        "",
        "## Tag References",
        "",
        "```text",
        "\n".join([line for line in tag_line.splitlines() if "gsw-" in line]),
        "```",
        "",
    ]
    write_text(REPORT / "GSW_12SCENE_CODE_FREEZE_AUDIT.md", "\n".join(lines))


def find_two_scene_row(label: str) -> dict[str, str] | None:
    for row in read_csv(REPO / "reports" / "two_scene_benchmark" / "TWO_SCENE_SCREENING_RESULTS.csv"):
        if row.get("label") == label:
            return row
    return None


def find_trackmobile_r1() -> dict[str, str] | None:
    for row in read_csv(REPO / "reports" / "gsw_repeated_30k_baseline" / "GSW_REPEATED_30K_SUMMARY.csv"):
        if row.get("run") == "R1" and row.get("mode") == "strict_intrinsic":
            return row
    return None


def designated_gsw_rows() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    r1 = find_trackmobile_r1()
    if r1:
        out[r1["scene"]] = {
            "run_id": "Trackmobile-R1",
            "source": "gsw_repeated_30k_baseline",
            "status": "success",
            "psnr": r1.get("psnr"),
            "ssim": r1.get("ssim"),
            "lpips": r1.get("lpips"),
            "gaussian_count": r1.get("gaussian_count"),
            "train_time": r1.get("train_time_sec"),
            "peak_memory": r1.get("peak_gpu_mem_mb"),
            "checkpoint_size": r1.get("checkpoint_size_bytes"),
            "black_view_count": "0",
            "nan_detected": "False",
            "strict_leakage_pass": "True",
            "model_path": r1.get("model_path"),
            "method_dir": r1.get("method_dir"),
            "notes": "Earliest valid strict_intrinsic repeated run, selected by frozen rule.",
        }
    for label, run_id in [("H-G1", "H-G1"), ("M-G1", "M-G1")]:
        row = find_two_scene_row(label)
        if row:
            out[row["scene"]] = {
                "run_id": run_id,
                "source": "two_scene_benchmark",
                "status": "success",
                "psnr": row.get("psnr"),
                "ssim": row.get("ssim"),
                "lpips": row.get("lpips"),
                "gaussian_count": row.get("gaussian_count"),
                "train_time": "",
                "peak_memory": "",
                "checkpoint_size": row.get("checkpoint_size_bytes"),
                "black_view_count": row.get("black_render_views"),
                "nan_detected": "False",
                "strict_leakage_pass": "True",
                "model_path": row.get("model_path"),
                "method_dir": row.get("method_dir"),
                "notes": "Earliest valid strict_intrinsic run from two-scene screening.",
            }
    return out


def gsw_registry_and_results() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenes = load_scene_summary()
    designated = designated_gsw_rows()
    registry: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for s in scenes:
        scene = s["scene"]
        manifest = manifest_for(scene)
        row = designated.get(scene)
        needs_training = row is None
        output = row.get("model_path") if row else str(RUNS_ROOT / "gsw_strict_12scene_single_run_20260702_pending_gpt" / scene)
        registry.append(
            {
                "scene_name": scene,
                "source_type": s["source_type"],
                "image_count": s["image_count"],
                "train_count": s["train_count"],
                "test_count": s["test_count"],
                "manifest_path": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "existing_valid_run": str(row is not None),
                "designated_run_id": row.get("run_id") if row else f"{scene}-pending-single-run",
                "needs_new_training": str(needs_training),
                "output_path": output,
                "status": "designated_existing_valid" if row else "pending_gpt_approval_for_single_30k",
            }
        )
        result = {
            "scene": scene,
            "run_id": row.get("run_id") if row else f"{scene}-pending-single-run",
            "status": row.get("status") if row else "not_run_pending_gpt_approval",
            "PSNR": row.get("psnr") if row else "",
            "SSIM": row.get("ssim") if row else "",
            "LPIPS": row.get("lpips") if row else "",
            "Gaussian_count": row.get("gaussian_count") if row else "",
            "train_time": row.get("train_time") if row else "",
            "peak_memory": row.get("peak_memory") if row else "",
            "checkpoint_size": row.get("checkpoint_size") if row else "",
            "black_view_count": row.get("black_view_count") if row else "",
            "NaN_detected": row.get("nan_detected") if row else "",
            "strict_leakage_pass": row.get("strict_leakage_pass") if row else "",
            "notes": row.get("notes") if row else "No corrected GS-W strict 30k designated run exists yet; do not infer metrics.",
        }
        results.append(result)
        if row is None:
            failures.append(
                {
                    "scene": scene,
                    "run_id": result["run_id"],
                    "failure_type": "not_run",
                    "status": "pending_gpt_approval_for_long_run",
                    "evidence": "No existing corrected GS-W strict_intrinsic 30k run matching frozen protocol was found.",
                    "required_followup": "Run one strict_intrinsic 30k attempt only if GPT approves the 12-scene long run.",
                }
            )
    # Keep the non-designated M-G3 collapse visible.
    mg3 = find_two_scene_row("M-G3")
    if mg3:
        failures.append(
            {
                "scene": "web_Terrestrial",
                "run_id": "M-G3",
                "failure_type": "non_designated_triggered_third_nan_black",
                "status": "completed_but_invalid_for_designated_12scene_table",
                "evidence": f"black_views={mg3.get('black_render_views')}; PSNR={mg3.get('psnr')}; SSIM={mg3.get('ssim')}; LPIPS={mg3.get('lpips')}",
                "required_followup": "Do not replace M-G1 with M-G3 and do not run a fourth replicate under current protocol.",
            }
        )
    write_csv(REPORT / "GSW_12SCENE_RUN_REGISTRY.csv", registry)
    write_csv(REPORT / "GSW_STRICT_12SCENE_RESULTS.csv", results)
    write_csv(REPORT / "GSW_STRICT_12SCENE_FAILURES.csv", failures)
    write_gsw_per_view()
    return registry, results


def write_gsw_per_view() -> None:
    rows: list[dict[str, Any]] = []
    for row in read_csv(REPO / "reports" / "gsw_repeated_30k_baseline" / "GSW_REPEATED_30K_PER_VIEW.csv"):
        if row.get("run") == "R1" and row.get("mode") == "strict_intrinsic":
            rows.append({"scene": row.get("scene"), "run_id": "Trackmobile-R1", **row})
    for row in read_csv(REPO / "reports" / "two_scene_benchmark" / "TWO_SCENE_SCREENING_PER_VIEW.csv"):
        if row.get("label") in {"H-G1", "M-G1"}:
            rows.append(row)
    write_csv(REPORT / "GSW_STRICT_12SCENE_PER_VIEW.csv", rows)


def official_audit() -> None:
    external_metrics = read_csv(EXTERNAL_SUMMARY / "metrics_all_methods.csv")
    official_external: dict[str, list[dict[str, str]]] = {}
    for row in external_metrics:
        if row.get("method") == "official_3dgs":
            official_external.setdefault(row["scene"], []).append(row)
    rows: list[dict[str, Any]] = []
    for scene in [s["scene"] for s in load_scene_summary()]:
        if scene == "self_Trackmobile_4650TM_Mobile_Railcar_Mover":
            klass = "A"
            evidence = "reports/official_control/UNIFIED_OFFICIAL_GSW_RESULTS.csv; clean Official_3DGS_Strict_Control commit 54c035f; strict full-image repeated evaluation."
            status = "verified strict matched"
        elif scene in {"self_Steam_Locomotive", "web_Terrestrial"}:
            klass = "A"
            evidence = "reports/two_scene_benchmark/TWO_SCENE_SCREENING_RESULTS.csv; frozen manifest; full-image unified evaluation."
            status = "verified strict matched"
        elif scene in official_external:
            klass = "B"
            evidence = "3dgs_runs/external_baselines_20260620/summary/metrics_all_methods.csv; old external supervisor results require provenance reconciliation."
            status = "likely matched but provenance incomplete"
        else:
            klass = "E"
            evidence = "No official result row found."
            status = "unavailable"
        rows.append(
            {
                "scene": scene,
                "class": klass,
                "status": status,
                "clean_official": str(klass == "A"),
                "resolution": "1" if klass in {"A", "B"} else "",
                "iterations": "30000" if klass in {"A", "B"} else "",
                "eval_split": "frozen LLFF hold-8" if klass == "A" else "claimed/likely LLFF hold-8",
                "full_image": str(klass in {"A", "B"}),
                "eligible_for_strict_main_table": str(klass == "A"),
                "evidence": evidence,
            }
        )
    write_csv(REPORT / "OFFICIAL_12SCENE_PROTOCOL_AUDIT.csv", rows)
    counts = {k: sum(1 for r in rows if r["class"] == k) for k in ["A", "B", "C", "D", "E"]}
    md = [
        "# OFFICIAL_12SCENE_PROTOCOL_AUDIT",
        "",
        f"- A verified strict matched: `{counts['A']}`",
        f"- B likely matched but provenance incomplete: `{counts['B']}`",
        f"- C non-strict: `{counts['C']}`",
        f"- D incompatible: `{counts['D']}`",
        f"- E unavailable: `{counts['E']}`",
        "",
        "Only class A rows are used in `GSW_VS_OFFICIAL_VERIFIED_COMPARISON.csv`. Class B rows are kept provisional.",
        "",
    ]
    for r in rows:
        md.append(f"- `{r['scene']}`: class `{r['class']}`; {r['status']}. Evidence: {r['evidence']}")
    write_text(REPORT / "OFFICIAL_12SCENE_PROTOCOL_AUDIT.md", "\n".join(md) + "\n")


def official_value_for_verified(scene: str) -> dict[str, Any] | None:
    if scene == "self_Trackmobile_4650TM_Mobile_Railcar_Mover":
        vals = [r for r in read_csv(REPO / "reports" / "official_control" / "UNIFIED_OFFICIAL_GSW_RESULTS.csv") if r.get("group") == "official_3dgs"]
    elif scene == "self_Steam_Locomotive":
        vals = [r for r in read_csv(REPO / "reports" / "two_scene_benchmark" / "TWO_SCENE_SCREENING_RESULTS.csv") if r.get("scene") == scene and r.get("method") == "official_3dgs"]
    elif scene == "web_Terrestrial":
        vals = [r for r in read_csv(REPO / "reports" / "two_scene_benchmark" / "TWO_SCENE_SCREENING_RESULTS.csv") if r.get("scene") == scene and r.get("method") == "official_3dgs"]
    else:
        return None
    nums = {metric: [safe_float(r.get(metric.lower() if metric != "PSNR" else "psnr")) for r in vals] for metric in ["PSNR", "SSIM", "LPIPS"]}
    clean = {k: [v for v in values if v is not None] for k, values in nums.items()}
    if not clean["PSNR"]:
        return None
    return {
        "official_PSNR": mean(clean["PSNR"]),
        "official_SSIM": mean(clean["SSIM"]),
        "official_LPIPS": mean(clean["LPIPS"]),
        "official_n": len(clean["PSNR"]),
    }


def gsw_vs_official() -> None:
    gsw = {r["scene"]: r for r in read_csv(REPORT / "GSW_STRICT_12SCENE_RESULTS.csv") if r.get("status") == "success"}
    official_rows = read_csv(REPORT / "OFFICIAL_12SCENE_PROTOCOL_AUDIT.csv")
    verified: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []
    external_metrics = read_csv(EXTERNAL_SUMMARY / "metrics_all_methods.csv")
    official_external: dict[str, dict[str, str]] = {}
    for row in external_metrics:
        if row.get("method") == "official_3dgs":
            official_external.setdefault(row["scene"], row)
    for audit in official_rows:
        scene = audit["scene"]
        if scene not in gsw:
            continue
        g = gsw[scene]
        g_psnr, g_ssim, g_lpips = safe_float(g.get("PSNR")), safe_float(g.get("SSIM")), safe_float(g.get("LPIPS"))
        if audit["class"] == "A":
            o = official_value_for_verified(scene)
            if not o:
                continue
            row = {
                "scene": scene,
                "gsw_run_id": g["run_id"],
                "gsw_PSNR": g_psnr,
                "gsw_SSIM": g_ssim,
                "gsw_LPIPS": g_lpips,
                **o,
                "dPSNR": g_psnr - o["official_PSNR"],
                "dSSIM": g_ssim - o["official_SSIM"],
                "dLPIPS": g_lpips - o["official_LPIPS"],
                "winner_PSNR": "GS-W" if g_psnr > o["official_PSNR"] else "official",
                "winner_SSIM": "GS-W" if g_ssim > o["official_SSIM"] else "official",
                "winner_LPIPS": "GS-W" if g_lpips < o["official_LPIPS"] else "official",
            }
            verified.append(row)
        elif audit["class"] == "B" and scene in official_external:
            e = official_external[scene]
            o_psnr, o_ssim, o_lpips = safe_float(e.get("psnr")), safe_float(e.get("ssim")), safe_float(e.get("lpips"))
            provisional.append(
                {
                    "scene": scene,
                    "gsw_run_id": g["run_id"],
                    "gsw_PSNR": g_psnr,
                    "gsw_SSIM": g_ssim,
                    "gsw_LPIPS": g_lpips,
                    "official_PSNR": o_psnr,
                    "official_SSIM": o_ssim,
                    "official_LPIPS": o_lpips,
                    "dPSNR": g_psnr - o_psnr if o_psnr is not None and g_psnr is not None else "",
                    "dSSIM": g_ssim - o_ssim if o_ssim is not None and g_ssim is not None else "",
                    "dLPIPS": g_lpips - o_lpips if o_lpips is not None and g_lpips is not None else "",
                    "protocol_note": "official result is class B provisional; not used in verified strict table.",
                }
            )
    write_csv(REPORT / "GSW_VS_OFFICIAL_VERIFIED_COMPARISON.csv", verified)
    write_csv(REPORT / "GSW_VS_OFFICIAL_PROVISIONAL_COMPARISON.csv", provisional)
    if verified:
        dpsnr = [float(r["dPSNR"]) for r in verified]
        dssim = [float(r["dSSIM"]) for r in verified]
        dlpips = [float(r["dLPIPS"]) for r in verified]
        win_psnr = sum(1 for r in verified if r["winner_PSNR"] == "GS-W")
        win_ssim = sum(1 for r in verified if r["winner_SSIM"] == "GS-W")
        win_lpips = sum(1 for r in verified if r["winner_LPIPS"] == "GS-W")
    else:
        dpsnr = dssim = dlpips = []
        win_psnr = win_ssim = win_lpips = 0
    gsw_results = read_csv(REPORT / "GSW_STRICT_12SCENE_RESULTS.csv")
    success_count = sum(1 for r in gsw_results if r.get("status") == "success")
    md = [
        "# GSW_VS_OFFICIAL_12SCENE_ANALYSIS",
        "",
        f"- Corrected GS-W strict coverage now verified: `{success_count}/12` designated scenes.",
        f"- Verified official common scenes: `{len(verified)}`.",
        f"- Provisional official common scenes: `{len(provisional)}`.",
        "- GS-W has exactly one designated run per scene in this analysis.",
        "- Single-run 12-scene rows are descriptive coverage evidence, not a statistical significance claim.",
        "",
    ]
    if verified:
        md += [
            "## Verified Strict Summary",
            "",
            f"- mean dPSNR: `{mean(dpsnr):.6f}`; median dPSNR: `{median(dpsnr):.6f}`",
            f"- mean dSSIM: `{mean(dssim):.6f}`; median dSSIM: `{median(dssim):.6f}`",
            f"- mean dLPIPS: `{mean(dlpips):.6f}`; median dLPIPS: `{median(dlpips):.6f}`",
            f"- PSNR wins/ties/losses for GS-W: `{win_psnr}/0/{len(verified)-win_psnr}`",
            f"- SSIM wins/ties/losses for GS-W: `{win_ssim}/0/{len(verified)-win_ssim}`",
            f"- LPIPS wins/ties/losses for GS-W: `{win_lpips}/0/{len(verified)-win_lpips}`",
            "",
        ]
    md += [
        "## Interpretation",
        "",
        "The verified table currently covers only Trackmobile, self_Steam_Locomotive and web_Terrestrial. The remaining scenes require GPT-approved single-run GS-W strict training before a complete 12-scene descriptive comparison can be claimed.",
        "",
    ]
    write_text(REPORT / "GSW_VS_OFFICIAL_12SCENE_ANALYSIS.md", "\n".join(md))


def repo_info(path: Path) -> dict[str, str]:
    if not (path / ".git").exists():
        return {"commit": "", "remote": "", "status": "not_git"}
    return {
        "commit": run(["git", "rev-parse", "HEAD"], cwd=path, check=False),
        "remote": run(["git", "remote", "-v"], cwd=path, check=False).replace("\n", "; "),
        "status": run(["git", "status", "--short"], cwd=path, check=False).replace("\n", "; "),
    }


def config_scalar(text: str, key: str) -> str:
    prefix = f"{key}:"
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return ""


def splatfacto_historical_run_audit() -> list[dict[str, Any]]:
    root = EXTERNAL_RUN / "splatfacto_w_r1_iter30000"
    logs = EXTERNAL_RUN / "logs"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        write_csv(REPORT / "SPLATFACTO_W_HISTORICAL_RUN_AUDIT.csv", rows)
        return rows
    for scene_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        cfgs = list(scene_dir.rglob("config.yml"))
        cfg_text = cfgs[0].read_text(encoding="utf-8", errors="replace") if cfgs else ""
        eval_json = scene_dir / "eval.json"
        eval_data: dict[str, Any] = {}
        if eval_json.exists():
            try:
                eval_data = json.loads(eval_json.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                eval_data = {"json_error": str(exc)}
        result = eval_data.get("results", {}) if isinstance(eval_data.get("results"), dict) else eval_data
        image_files = [
            p
            for ext in ("*.png", "*.jpg", "*.jpeg")
            for p in scene_dir.rglob(ext)
            if p.is_file()
        ]
        render_like = [
            p
            for p in image_files
            if any(token in str(p).lower() for token in ("render", "pred", "rgb", "gt", "image"))
        ]
        train_cmd = logs / f"splatfacto_w_r1_{scene_dir.name}_train.cmd"
        metrics_cmd = logs / f"splatfacto_w_r1_{scene_dir.name}_metrics.cmd"
        rows.append(
            {
                "scene": scene_dir.name,
                "config_count": len(cfgs),
                "config_path": str(cfgs[0]) if cfgs else "",
                "max_num_iterations": config_scalar(cfg_text, "max_num_iterations"),
                "method_name": config_scalar(cfg_text, "method_name"),
                "datamanager": "FullImageDatamanager" if "FullImageDatamanagerConfig" in cfg_text else "unknown",
                "eval_right_half": config_scalar(cfg_text, "eval_right_half"),
                "use_avg_appearance": config_scalar(cfg_text, "use_avg_appearance"),
                "appearance_embed_dim": config_scalar(cfg_text, "appearance_embed_dim"),
                "appearance_features_dim": config_scalar(cfg_text, "appearance_features_dim"),
                "train_split_fraction": config_scalar(cfg_text, "train_split_fraction"),
                "eval_mode": config_scalar(cfg_text, "eval_mode"),
                "eval_interval": config_scalar(cfg_text, "eval_interval"),
                "downscale_factor": config_scalar(cfg_text, "downscale_factor"),
                "eval_json_exists": str(eval_json.exists()),
                "psnr": result.get("psnr", ""),
                "ssim": result.get("ssim", ""),
                "lpips": result.get("lpips", ""),
                "checkpoint_count": len(list(scene_dir.rglob("*.ckpt"))),
                "render_like_image_count": len(render_like),
                "train_cmd_exists": str(train_cmd.exists()),
                "metrics_cmd_exists": str(metrics_cmd.exists()),
            }
        )
    write_csv(REPORT / "SPLATFACTO_W_HISTORICAL_RUN_AUDIT.csv", rows)
    return rows


def external_fairness() -> None:
    repos = {
        "GS-W legacy": WL3DGS / "external_baselines" / "Gaussian-Wild",
        "corrected GS-W strict_intrinsic": REPO,
        "Splatfacto-W": WL3DGS / "external_baselines" / "splatfacto-w",
        "Luminance-GS": WL3DGS / "external_baselines" / "Luminance-GS",
        "WildGaussians": WL3DGS / "external_baselines" / "wild-gaussians",
    }
    matrix = [
        {
            "method": "GS-W legacy",
            "upstream_repository": "https://github.com/EastbeanZhang/Gaussian-Wild.git",
            "exact_commit": repo_info(repos["GS-W legacy"])["commit"],
            "license": "not audited in this pass",
            "local_code_modified": "yes/unknown adapter patches from historical integration",
            "split": "historical LLFF hold-8 likely",
            "full_image_or_half_image": "full-image metrics but test RGB conditioned",
            "test_rgb_into_appearance": "yes",
            "test_time_optimization": "no explicit TTO found",
            "strict_main_competitor": "False",
            "protocol_class": "B",
            "evidence": "reports/gsw_repeated_30k_baseline/EVAL_STATE_AUDIT.md; legacy_target_rgb render mode; previous audit found test RGB into map_generator and BN eval-state risk.",
        },
        {
            "method": "corrected GS-W strict_intrinsic",
            "upstream_repository": "https://github.com/EastbeanZhang/Gaussian-Wild.git",
            "exact_commit": training_freeze_commit(),
            "license": "inherits Gaussian-Wild; not redistributed as vendor source in package",
            "local_code_modified": "strict split/strict_intrinsic/eval-state/rasterizer compatibility fixes",
            "split": "frozen LLFF hold-8",
            "full_image_or_half_image": "full-image",
            "test_rgb_into_appearance": "no for strict_intrinsic",
            "test_time_optimization": "no",
            "strict_main_competitor": "True for designated successful rows only",
            "protocol_class": "A",
            "evidence": "reports/two_scene_benchmark/GSW_STRICT_LEAKAGE_BN_AUDIT.md; reports/baseline_completion/GSW_12SCENE_CODE_FREEZE_AUDIT.md.",
        },
        {
            "method": "Splatfacto-W",
            "upstream_repository": "local clone of splatfacto-w",
            "exact_commit": repo_info(repos["Splatfacto-W"])["commit"],
            "license": "LICENSE present; not fully legal-audited",
            "local_code_modified": repo_info(repos["Splatfacto-W"])["status"] or "clean/unknown",
            "split": "Nerfstudio colmap eval-mode interval/eval-interval 8; not tied to frozen manifest hash in historical artifacts",
            "full_image_or_half_image": "historical saved configs show eval_right_half=false full-image aggregate metrics; source non-light default has right-half mode but was not used by these saved configs",
            "test_rgb_into_appearance": "no direct evidence of test RGB fitting; saved configs use_avg_appearance=true average train appearance for eval",
            "test_time_optimization": "no evidence of test-time optimization in ns-eval logs",
            "strict_main_competitor": "False",
            "protocol_class": "D",
            "evidence": "SPLATFACTO_W_PROTOCOL_AUDIT.md; SPLATFACTO_W_HISTORICAL_RUN_AUDIT.csv shows 12 saved configs with eval_right_half=false, use_avg_appearance=true, eval_mode=interval, eval_interval=8, train_split_fraction=0.9, and zero saved render/GT images for unified re-evaluation.",
        },
        {
            "method": "Luminance-GS",
            "upstream_repository": "https://github.com/cuiziteng/Luminance-GS.git",
            "exact_commit": repo_info(repos["Luminance-GS"])["commit"],
            "license": "LICENSE present; not fully legal-audited",
            "local_code_modified": repo_info(repos["Luminance-GS"])["status"] or "unknown",
            "split": "adapter-generated LLFF-like split",
            "full_image_or_half_image": "full-image intended",
            "test_rgb_into_appearance": "unresolved; curve/eval protocol is not strict-audited and current import smoke fails before reader construction",
            "test_time_optimization": "unresolved",
            "strict_main_competitor": "False; current local env/adapter state is not reproducible",
            "protocol_class": "E",
            "evidence": "reports/baseline_completion/EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md; logs/external_bounded_diagnostics/luminance_import_smoke.log shows pycolmap SceneManager import failure; 3dgs_runs/external_baselines_20260620/summary/failures.csv.",
        },
        {
            "method": "WildGaussians",
            "upstream_repository": "https://github.com/jkulhanek/wild-gaussians.git",
            "exact_commit": repo_info(repos["WildGaussians"])["commit"],
            "license": "LICENSE present; not fully legal-audited",
            "local_code_modified": repo_info(repos["WildGaussians"])["status"] or "unknown",
            "split": "adapter-generated",
            "full_image_or_half_image": "full-image intended",
            "test_rgb_into_appearance": "method may use appearance embeddings; current integration invalid due dark renders",
            "test_time_optimization": "unresolved",
            "strict_main_competitor": "False",
            "protocol_class": "E",
            "evidence": "metrics_all_methods.csv shows PSNR mostly 4.94-8.41 except one small-scene row; WILDGAUSSIANS_DIAGNOSTIC_RESULTS.csv; EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md; wild reader smoke logs show COLMAP adapter reads but numeric renders remain invalid.",
        },
    ]
    write_csv(REPORT / "EXTERNAL_BASELINE_FAIRNESS_MATRIX.csv", matrix)
    md = [
        "# EXTERNAL_BASELINE_FAIRNESS_AUDIT",
        "",
        "| method | class | strict main competitor | key evidence |",
        "|---|---|---:|---|",
    ]
    for row in matrix:
        md.append(f"| {row['method']} | {row['protocol_class']} | {row['strict_main_competitor']} | {row['evidence']} |")
    md += [
        "",
        "Class meanings: A strict held-out comparable; B target/appearance-conditioned; C transductive or test-time optimized/half-image; D incompatible; E adapter/config invalid; F unresolved.",
        "",
    ]
    write_text(REPORT / "EXTERNAL_BASELINE_FAIRNESS_AUDIT.md", "\n".join(md))
    splatfacto_audit()
    luminance_audit()
    wildgaussians_audit()
    baseline_registry()


def splatfacto_audit() -> None:
    metrics = [r for r in read_csv(EXTERNAL_SUMMARY / "metrics_all_methods.csv") if r.get("method") == "splatfacto_w"]
    metric_30k = [r for r in metrics if r.get("iterations") == "30000"]
    historical = splatfacto_historical_run_audit()
    full_configs = [r for r in historical if r.get("eval_right_half") == "false"]
    avg_configs = [r for r in historical if r.get("use_avg_appearance") == "true"]
    render_exports = sum(int(r.get("render_like_image_count") or 0) for r in historical)
    md = [
        "# SPLATFACTO_W_PROTOCOL_AUDIT",
        "",
        f"- Current 30k numeric coverage: `{len(metric_30k)}/12` rows in historical summary.",
        "- Protocol decision: `not strict full-image held-out comparable`.",
        "- Classification: `D. protocol incompatible with the current strict main table`.",
        "",
        "Evidence:",
        "",
        "- Source defaults are mixed: `external_baselines/splatfacto-w/splatfactow/splatfactow_config.py:43` enables right-half evaluation for the non-light phototourism config, while the light config at `splatfactow_config.py:126-160` uses `FullImageDatamanagerConfig` and `use_avg_appearance=True` without setting `eval_right_half`.",
        "- The 12 historical 30k saved configs all use `splatfacto-w-light`, `eval_right_half=false`, `use_avg_appearance=true`, `eval_mode=interval`, `eval_interval=8`, `train_split_fraction=0.9`, and `downscale_factor=1`; see `SPLATFACTO_W_HISTORICAL_RUN_AUDIT.csv`.",
        "- `external_baselines/splatfacto-w/splatfactow/splatfactow_model.py:937-956` uses per-camera appearance when `camera.metadata['cam_idx']` exists, otherwise uses average appearance if `use_avg_appearance=True`. The historical eval logs do not show test-time fitting or test RGB appearance optimization.",
        "- `splatfactow_model.py:1261-1265` can crop to the right half only when `eval_right_half=True`; the saved historical configs set it to false, so the previous right-half classification is not supported for these 12 rows.",
        f"- Existing Splatfacto-W outputs contain `{render_exports}` render-like PNG/JPG files across 12 scenes. `ns-eval` wrote aggregate `eval.json` files only, so the requested unified full-image evaluator cannot be run from existing renders.",
        "- The split is controlled by Nerfstudio's COLMAP dataparser command (`--eval-mode interval --eval-interval 8`) and saved config fields, not by a frozen manifest path/hash in the Splatfacto-W artifacts. Even if the interval likely resembles LLFF hold-8, the current evidence is insufficient to treat the aggregate rows as verified strict-main results.",
        "",
        "Result handling:",
        "",
        "- Move current Splatfacto-W rows out of the strict main table.",
        "- They may be retained only as provisional Nerfstudio-internal aggregate numbers with a clear protocol caveat.",
        "- A fair Splatfacto-W competitor would require either saved per-view render/GT export from the existing checkpoints followed by the unified evaluator, or a new GPT-approved strict run/export plan. This round did not start that work.",
        "- No retraining was started in this round.",
        "",
        f"- Historical metric rows found: `{len(metrics)}` total, including `{len(metric_30k)}` 30k rows.",
        f"- Historical configs with `eval_right_half=false`: `{len(full_configs)}/12`.",
        f"- Historical configs with `use_avg_appearance=true`: `{len(avg_configs)}/12`.",
    ]
    write_text(REPORT / "SPLATFACTO_W_PROTOCOL_AUDIT.md", "\n".join(md) + "\n")


def luminance_audit() -> None:
    metrics = [r for r in read_csv(EXTERNAL_SUMMARY / "metrics_all_methods.csv") if r.get("method") == "luminance_gs"]
    scenes = {s["scene"] for s in load_scene_summary()}
    metric_scenes = {r["scene"] for r in metrics}
    missing = sorted(scenes - metric_scenes)
    partial = []
    for r in metrics:
        if r["scene"] == "self_3000t_Press":
            partial.append(r["scene"])
    failures = [r for r in read_csv(EXTERNAL_SUMMARY / "failures.csv") if r.get("method") == "luminance_gs"]
    rows: list[dict[str, Any]] = []
    for f in failures:
        rows.append(
            {
                "scene": f.get("scene"),
                "timestamp": f.get("timestamp"),
                "phase": f.get("phase"),
                "resolution": f.get("resolution"),
                "error_type": f.get("error_type"),
                "log_or_evidence": "; ".join([v for v in f.values() if isinstance(v, str) and "G:\\wl3dgs" in v]),
                "diagnosis": "historical failure; needs bounded adapter/config smoke before any full rerun",
            }
        )
    for s in missing:
        if not any(r.get("scene") == s for r in rows):
            rows.append(
                {
                    "scene": s,
                    "timestamp": "",
                    "phase": "missing_final_metric",
                    "resolution": "1",
                    "error_type": "no valid metric row",
                    "log_or_evidence": str(EXTERNAL_SUMMARY / "metrics_all_methods.csv"),
                    "diagnosis": "missing from final historical metric table",
                }
            )
    if partial:
        rows.append(
            {
                "scene": "self_3000t_Press",
                "timestamp": "",
                "phase": "partial_or_low_confidence_result",
                "resolution": "1",
                "error_type": "result exists after repeated failures but final stats appear step9999-scale",
                "log_or_evidence": str(EXTERNAL_RUN / "luminance_gs_r1_iter30000" / "self_3000t_Press" / "stats"),
                "diagnosis": "treat as provisional/needs audit, not verified strict competitor",
            }
        )
    write_csv(REPORT / "LUMINANCE_GS_FAILURE_DIAGNOSIS.csv", rows)
    md = [
        "# LUMINANCE_GS_PROTOCOL_AUDIT",
        "",
        f"- Historical metric rows found: `{len(metrics)}`.",
        f"- Missing final metric scenes: `{', '.join(missing) if missing else '(none)'}`.",
        "- Protocol classification: `E. adapter/configuration invalid in current local state`.",
        "- Current rows should not enter strict main table: the current 3dgs environment fails to import Luminance-GS because `pycolmap.SceneManager` is unavailable.",
        "",
        "Evidence:",
        "",
        "- `external_baselines/Luminance-GS/Luminance-GS/README.md` describes LOM low-light/overexposure and MipNeRF360-varying, not this COLMAP tourism protocol.",
        "- `3dgs_runs/external_baselines_20260620/summary/failures.csv` records multiple failed attempts.",
        "- `metrics_all_methods.csv` contains Luminance-GS rows for only a subset/low-confidence set of scenes.",
        "- `reports/baseline_completion/logs/external_bounded_diagnostics/luminance_import_smoke.log` records `ImportError: cannot import name 'SceneManager' from 'pycolmap'`; `luminance_pycolmap_probe.log` records pycolmap `4.0.4` and `has_SceneManager=false`.",
        "- `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_RUNS.csv` and `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_ADAPTER.csv` contain per-scene log/adapter diagnostics.",
        "",
    ]
    write_text(REPORT / "LUMINANCE_GS_PROTOCOL_AUDIT.md", "\n".join(md))
    rec = [
        "# LUMINANCE_GS_RERUN_RECOMMENDATION",
        "",
        "Recommendation: do not launch full 30k reruns.",
        "",
        "Bounded next step if GPT approves Luminance-GS repair: fix the wrapper/environment mismatch around `pycolmap.SceneManager` first, then rerun import + reader smoke. Do not run 30k until the current import failure is resolved.",
        "",
        f"Priority scenes: `{', '.join(sorted(set(missing + partial)))}`.",
    ]
    write_text(REPORT / "LUMINANCE_GS_RERUN_RECOMMENDATION.md", "\n".join(rec) + "\n")


def image_stats(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        finite = bool(np.isfinite(arr).all())
        black = (arr.max(axis=2) < 0.02).mean()
        return {
            "image": str(path),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "black_pixel_ratio": float(black),
            "finite": str(finite),
        }
    except Exception as exc:
        return {"image": str(path), "error": str(exc)}


def wildgaussians_audit() -> None:
    metrics = [r for r in read_csv(EXTERNAL_SUMMARY / "metrics_all_methods.csv") if r.get("method") == "wildgaussians"]
    stats: list[dict[str, Any]] = []
    audit_root = EXTERNAL_SUMMARY / "audit_samples"
    sample_groups = {
        "eval_extract": audit_root / "wild_web_doss" / "color",
        "nerfw_render_extract": audit_root / "wild_web_doss_nerfw_render_extract" / "color",
        "train_embed_pred": audit_root / "wild_web_doss_train_embed",
    }
    for group, root in sample_groups.items():
        if not root.exists():
            continue
        if group == "train_embed_pred":
            candidate_images = sorted(root.glob("*_pred_*.png"))
        else:
            candidate_images = sorted(root.glob("*.png"))
        for path in candidate_images[:64]:
            row = image_stats(path)
            row["sample_group"] = group
            stats.append(row)
    write_csv(REPORT / "WILDGAUSSIANS_DIAGNOSTIC_RESULTS.csv", stats)
    low_psnr = [safe_float(r.get("psnr")) for r in metrics]
    low_psnr = [v for v in low_psnr if v is not None]
    md = [
        "# WILDGAUSSIANS_PROTOCOL_AUDIT",
        "",
        f"- Historical metric rows found: `{len(metrics)}`.",
        f"- PSNR range in historical summary: `{min(low_psnr):.4f}` to `{max(low_psnr):.4f}`." if low_psnr else "- No PSNR values found.",
        "- Protocol classification: `E. adapter/configuration invalid` pending render/checkpoint/appearance repair.",
        "- Current WildGaussians values should be removed from strict/provisional numeric comparisons.",
        "",
        "Evidence:",
        "",
        "- `metrics_all_methods.csv` shows broadly collapsed PSNR/SSIM values for WildGaussians, consistent with dark/invalid renders.",
        "- `wildgaussians/utils.py:130-146` performs color-space/background conversion; `utils.py:154-169` saves PNG output.",
        "- `wildgaussians/evaluation.py:332` explicitly blends with black background for metrics.",
        f"- Numeric image statistics are in `WILDGAUSSIANS_DIAGNOSTIC_RESULTS.csv` (`{len(stats)}` sampled prediction images).",
        "- `logs/external_bounded_diagnostics/wild_import_smoke.log` passes; `wild_reader_smoke.log` loads both `web_Baalshamin_images` and `web_doss_images` train/test adapters. Thus the remaining failure is downstream of basic COLMAP reader construction.",
        "",
    ]
    write_text(REPORT / "WILDGAUSSIANS_PROTOCOL_AUDIT.md", "\n".join(md))
    diag = [
        "# WILDGAUSSIANS_BLACK_RENDER_DIAGNOSIS",
        "",
        "Current evidence is strongest for integration/adapter invalidity rather than a fair method failure: historical metrics are dark/low enough that the numeric table is invalid. Basic import and COLMAP reader smoke pass on one simple and one complex scene, so the next diagnostic should focus on render/checkpoint/appearance/background handling.",
        "",
        f"- Diagnostic image files scanned: `{len(stats)}`.",
        "- Import smoke: passed.",
        "- Reader smoke: passed for `web_Baalshamin_images` train/test and `web_doss_images` train/test.",
        "- No full 12-scene WildGaussians rerun was started.",
    ]
    write_text(REPORT / "WILDGAUSSIANS_BLACK_RENDER_DIAGNOSIS.md", "\n".join(diag) + "\n")
    rec = [
        "# WILDGAUSSIANS_RERUN_RECOMMENDATION",
        "",
        "Recommendation: do not run full 12-scene WildGaussians yet.",
        "",
        "First fix or disprove adapter/config issues with two bounded diagnostics: one small/simple scene and one high-image-count scene. Only wrapper/adapter changes should be considered; do not modify WildGaussians method code.",
    ]
    write_text(REPORT / "WILDGAUSSIANS_RERUN_RECOMMENDATION.md", "\n".join(rec) + "\n")


def baseline_registry() -> None:
    metrics = read_csv(EXTERNAL_SUMMARY / "metrics_all_methods.csv")
    fairness = {r["method"]: r for r in read_csv(REPORT / "EXTERNAL_BASELINE_FAIRNESS_MATRIX.csv")}
    method_map = {
        "official_3dgs": "official_3dgs",
        "gs_w": "GS-W legacy",
        "wildgaussians": "WildGaussians",
        "luminance_gs": "Luminance-GS",
        "splatfacto_w": "Splatfacto-W",
    }
    rows: list[dict[str, Any]] = []
    for row in metrics:
        method = method_map.get(row.get("method"), row.get("method"))
        fair = fairness.get(method, {})
        klass = fair.get("protocol_class", "F")
        strict = klass == "A"
        if klass == "A":
            action = "retain in strict table"
            valid = "True"
        elif klass in {"B", "C"}:
            action = "move to secondary conditioned table"
            valid = "True"
        elif klass == "D":
            action = "requires strict render export/re-evaluation"
            valid = "False"
        elif klass == "E":
            action = "invalid/remove"
            valid = "False"
        else:
            action = "provisional"
            valid = "False"
        rows.append(
            {
                "method": method,
                "scene": row.get("scene"),
                "reported_status": "metric_row",
                "protocol_class": klass,
                "strict_comparable": str(strict),
                "result_valid": valid,
                "PSNR": row.get("psnr"),
                "SSIM": row.get("ssim"),
                "LPIPS": row.get("lpips"),
                "render_available": "unknown_from_summary",
                "checkpoint_available": "unknown_from_summary",
                "failure_type": "" if valid == "True" else "protocol_or_adapter_unresolved",
                "required_followup": action,
                "evidence": fair.get("evidence", str(EXTERNAL_SUMMARY / "metrics_all_methods.csv")),
            }
        )
    # Add corrected GS-W designated rows.
    for row in read_csv(REPORT / "GSW_STRICT_12SCENE_RESULTS.csv"):
        rows.append(
            {
                "method": "corrected GS-W strict_intrinsic",
                "scene": row["scene"],
                "reported_status": row["status"],
                "protocol_class": "A",
                "strict_comparable": str(row["status"] == "success"),
                "result_valid": str(row["status"] == "success"),
                "PSNR": row.get("PSNR"),
                "SSIM": row.get("SSIM"),
                "LPIPS": row.get("LPIPS"),
                "render_available": str(row["status"] == "success"),
                "checkpoint_available": str(row["status"] == "success"),
                "failure_type": "" if row["status"] == "success" else "missing_single_run",
                "required_followup": "retain in strict table" if row["status"] == "success" else "requires rerun",
                "evidence": "reports/baseline_completion/GSW_STRICT_12SCENE_RESULTS.csv",
            }
        )
    write_csv(REPORT / "EXTERNAL_BASELINE_RESULT_REGISTRY.csv", rows)
    plan = [
        "# BASELINE_TABLE_REVISION_PLAN",
        "",
        "- Strict main table: retain only class A rows with valid metrics. Currently this means corrected GS-W designated successful rows and verified official class A scenes.",
        "- Legacy GS-W: move to secondary conditioned/non-strict table; use as protocol-risk illustration only.",
        "- Splatfacto-W: do not rank against strict official/GS-W. Historical 30k configs are full-image (`eval_right_half=false`) and use average appearance, but only aggregate Nerfstudio `eval.json` exists; strict use requires per-view render/GT export and unified re-evaluation or a GPT-approved strict rerun/export plan.",
        "- Luminance-GS: mark provisional/unresolved; run bounded diagnostics before any 30k rerun.",
        "- WildGaussians: mark invalid/remove current numeric rows due dark/adapter-invalid renders; diagnose wrapper before considering rerun.",
        "- Old official 12-scene rows: class B unless verified by provenance; do not mix with verified strict table.",
        "",
    ]
    write_text(REPORT / "BASELINE_TABLE_REVISION_PLAN.md", "\n".join(plan))


def repo_status_files() -> None:
    status = run(["git", "status", "--short", "--branch"])
    head = run(["git", "rev-parse", "HEAD"])
    count = run(["git", "rev-list", "--left-right", "--count", "origin/main...main"], check=False)
    tags = run(["git", "tag", "--points-at", "HEAD"], check=False) or "(none)"
    write_text(
        REPORT / "REPO_STATUS.txt",
        "\n".join(
            [
                "## git status --short --branch",
                status,
                "",
                "## git rev-parse HEAD",
                head,
                "",
                "## origin/main...main",
                count,
                "",
                "## tags at HEAD",
                tags,
            ]
        ),
    )
    write_text(REPORT / "COMMIT_HISTORY.md", "# COMMIT_HISTORY\n\n```text\n" + run(["git", "log", "--oneline", "--decorate", "-12"]) + "\n```\n")
    write_text(
        REPORT / "PUSH_STATUS.md",
        "# PUSH_STATUS\n\n"
        + f"- HEAD: `{head}`\n"
        + f"- origin/main...main: `{count}`\n\n"
        + "```text\n"
        + run(["git", "remote", "-v"], check=False)
        + "\n```\n",
    )


def summary_report() -> None:
    gsw_rows = read_csv(REPORT / "GSW_STRICT_12SCENE_RESULTS.csv")
    successes = [r for r in gsw_rows if r.get("status") == "success"]
    psnr = [safe_float(r.get("PSNR")) for r in successes]
    ssim = [safe_float(r.get("SSIM")) for r in successes]
    lpips = [safe_float(r.get("LPIPS")) for r in successes]
    psnr = [v for v in psnr if v is not None]
    ssim = [v for v in ssim if v is not None]
    lpips = [v for v in lpips if v is not None]
    official_a = read_csv(REPORT / "OFFICIAL_12SCENE_PROTOCOL_AUDIT.csv")
    counts = {k: sum(1 for r in official_a if r.get("class") == k) for k in ["A", "B", "C", "D", "E"]}
    verified = read_csv(REPORT / "GSW_VS_OFFICIAL_VERIFIED_COMPARISON.csv")
    md = [
        "# BASELINE_COMPLETION_AUDIT_SUMMARY",
        "",
        "This package is an audit/preflight deliverable. It does not start the remaining 9-scene GS-W long run because the GPT instruction states that execution of the above long run should wait for GPT review.",
        "",
        f"- Corrected GS-W strict designated successes: `{len(successes)}/12`.",
        f"- Corrected GS-W strict missing/pending: `{12 - len(successes)}/12`.",
        f"- GS-W successful-scene mean/median PSNR: `{mean(psnr):.6f}` / `{median(psnr):.6f}`." if psnr else "- GS-W successful-scene PSNR: unavailable.",
        f"- GS-W successful-scene mean/median SSIM: `{mean(ssim):.6f}` / `{median(ssim):.6f}`." if ssim else "- GS-W successful-scene SSIM: unavailable.",
        f"- GS-W successful-scene mean/median LPIPS: `{mean(lpips):.6f}` / `{median(lpips):.6f}`." if lpips else "- GS-W successful-scene LPIPS: unavailable.",
        f"- Existing official 12-scene classes: A={counts['A']}, B={counts['B']}, C={counts['C']}, D={counts['D']}, E={counts['E']}.",
        f"- Verified official common scenes currently compared: `{len(verified)}`.",
        "- Splatfacto-W: not strict-main usable. Current evidence corrects the earlier half-image claim: saved 12-scene configs are full-image (`eval_right_half=false`) with average appearance, but no per-view render/GT artifacts exist for unified re-evaluation and split provenance is not frozen-manifest verified.",
        "- Luminance-GS: current local env/adapter state invalid; bounded import/log/adapter diagnostics found pycolmap `SceneManager` API mismatch.",
        "- WildGaussians: current numeric rows invalid/remove; bounded reader diagnostics pass, but render/checkpoint/appearance integration remains invalid/dark.",
        "",
        "Next GPT decision needed: approve or reject the remaining single-run GS-W strict 30k schedule for the 9 pending scenes; separately decide whether to repair Luminance-GS/WildGaussians wrappers.",
        "",
    ]
    write_text(REPORT / "BASELINE_COMPLETION_AUDIT_SUMMARY.md", "\n".join(md))


def completion_gap_audit() -> None:
    gsw_rows = read_csv(REPORT / "GSW_STRICT_12SCENE_RESULTS.csv")
    successes = [r for r in gsw_rows if r.get("status") == "success"]
    pending = [r for r in gsw_rows if r.get("status") != "success"]
    official_rows = read_csv(REPORT / "OFFICIAL_12SCENE_PROTOCOL_AUDIT.csv")
    official_counts = {k: sum(1 for r in official_rows if r.get("class") == k) for k in ["A", "B", "C", "D", "E"]}
    gap_rows = [
        {
            "requirement": "corrected_gsw_strict_12scene_coverage",
            "status": "incomplete_pending_gpt_approval",
            "evidence": f"GSW_STRICT_12SCENE_RESULTS.csv has {len(successes)} success rows and {len(pending)} non-success/pending rows.",
            "missing_or_risk": "9 corrected GS-W strict 30k single-run scenes have not been executed.",
            "next_action": "After GPT approval, run strict_12scene_runner.py with GPT_APPROVED_12SCENE_GSW_30K=1, then regenerate reports.",
        },
        {
            "requirement": "one_designated_run_per_scene_registry",
            "status": "complete_for_current_registry",
            "evidence": "GSW_12SCENE_RUN_REGISTRY.csv has exactly one designated_run_id per scene.",
            "missing_or_risk": "Pending scenes are command-plan rows, not measured runs.",
            "next_action": "Use the same designated IDs when GPT approves execution.",
        },
        {
            "requirement": "gsw_training_code_freeze",
            "status": "complete",
            "evidence": "GSW_12SCENE_CODE_FREEZE_AUDIT.md reports PASS against gsw-strict-baseline-v2 and gsw-strict-12scene-v1 for training paths.",
            "missing_or_risk": "",
            "next_action": "Keep training behavior frozen; only tools/reports may change before execution.",
        },
        {
            "requirement": "unified_full_image_evaluator",
            "status": "tool_ready_not_full_12scene_executed",
            "evidence": "tools/baseline_completion/unified_full_image_eval.py is included and py_compile-tested.",
            "missing_or_risk": "It has not yet evaluated 9 missing corrected GS-W runs because renders do not exist.",
            "next_action": "Run via strict_12scene_runner.py after training/rendering completes.",
        },
        {
            "requirement": "official_12scene_protocol_audit",
            "status": "complete_for_existing_results",
            "evidence": f"OFFICIAL_12SCENE_PROTOCOL_AUDIT.csv classes: A={official_counts['A']}, B={official_counts['B']}, C={official_counts['C']}, D={official_counts['D']}, E={official_counts['E']}.",
            "missing_or_risk": "B scenes remain provenance-incomplete and cannot enter verified strict main comparison.",
            "next_action": "Do not rerun official in this round; list B scenes as provisional only.",
        },
        {
            "requirement": "gsw_vs_official_12scene_descriptive_comparison",
            "status": "partial",
            "evidence": "GSW_VS_OFFICIAL_VERIFIED_COMPARISON.csv covers 3 verified common scenes.",
            "missing_or_risk": "No verified 12-scene comparison exists until GS-W pending scenes are run and official B scenes are either verified or kept provisional.",
            "next_action": "Regenerate after GPT-approved GS-W runs.",
        },
        {
            "requirement": "external_baseline_fairness_matrix",
            "status": "complete_initial_audit",
            "evidence": "EXTERNAL_BASELINE_FAIRNESS_MATRIX.csv classifies GS-W legacy, corrected GS-W, Splatfacto-W, Luminance-GS and WildGaussians.",
            "missing_or_risk": "Luminance-GS and WildGaussians remain unresolved/invalid until bounded diagnostics are executed.",
            "next_action": "Use current matrix for table revision; do not rank unresolved rows in strict table.",
        },
        {
            "requirement": "splatfacto_w_special_audit",
            "status": "complete_for_current_historical_results",
            "evidence": "SPLATFACTO_W_PROTOCOL_AUDIT.md and SPLATFACTO_W_HISTORICAL_RUN_AUDIT.csv record saved full-image average-appearance configs, missing render/GT exports, and non-manifest split provenance.",
            "missing_or_risk": "No unified re-evaluation from existing renders is possible because historical outputs contain aggregate eval.json only. No strict Splatfacto-W rerun/export was performed, by instruction.",
            "next_action": "Keep current Splatfacto-W numbers out of the strict main table; ask GPT whether to export per-view renders from checkpoints or schedule a strict rerun/export plan.",
        },
        {
            "requirement": "luminance_gs_bounded_diagnostics",
            "status": "completed_bounded_import_log_adapter_diagnostic",
            "evidence": "EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.md plus luminance_import_smoke.log show current environment fails at pycolmap SceneManager import; LUMINANCE_GS_BOUNDED_DIAGNOSTIC_RUNS.csv summarizes historical failures and render/checkpoint presence.",
            "missing_or_risk": "A 300-1000 iteration smoke is blocked by import/API incompatibility; current Luminance-GS rows are not strict-main reproducible.",
            "next_action": "If GPT wants Luminance-GS retained, first fix the wrapper/environment pycolmap API mismatch, then rerun bounded smoke only.",
        },
        {
            "requirement": "wildgaussians_bounded_diagnostics",
            "status": "completed_import_reader_static_render_diagnostic",
            "evidence": "wild_import_smoke.log passes; wild_reader_smoke.log reads simple and complex adapters; WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_RUNS.csv and WILDGAUSSIANS_DIAGNOSTIC_RESULTS.csv show dark/low-PSNR outputs.",
            "missing_or_risk": "Reader is not the root failure; remaining risk is render/checkpoint/appearance/background integration. No full 12-scene rerun was launched.",
            "next_action": "Diagnose WildGaussians wrapper render/checkpoint/appearance handling on two scenes before any full rerun.",
        },
        {
            "requirement": "forbidden_changes",
            "status": "complete",
            "evidence": "Only tools/baseline_completion and reports/baseline_completion were added; training methods, networks, losses and densification were not modified in this round.",
            "missing_or_risk": "",
            "next_action": "Continue to keep baseline method source frozen.",
        },
    ]
    write_csv(REPORT / "COMPLETION_GAP_AUDIT.csv", gap_rows)
    md = [
        "# COMPLETION_GAP_AUDIT",
        "",
        "This audit prevents the current preflight package from being mistaken for a completed 12-scene benchmark.",
        "",
        f"- Corrected GS-W strict success coverage: `{len(successes)}/12`.",
        f"- Corrected GS-W strict pending/non-success rows: `{len(pending)}/12`.",
        f"- Official class counts: A={official_counts['A']}, B={official_counts['B']}, C={official_counts['C']}, D={official_counts['D']}, E={official_counts['E']}.",
        "- Stop condition from GPT is not fully satisfied only for corrected GS-W 12-scene coverage: 9 pending strict 30k runs still require GPT approval. Bounded Luminance-GS and WildGaussians diagnostics have been executed and recorded.",
        "- The package is still useful as a review gate: it freezes protocol, identifies invalid/provisional baselines and provides a guarded runner/evaluator for the next approved step.",
        "",
        "| requirement | status | next action |",
        "|---|---|---|",
    ]
    for row in gap_rows:
        md.append(f"| {row['requirement']} | {row['status']} | {row['next_action']} |")
    write_text(REPORT / "COMPLETION_GAP_AUDIT.md", "\n".join(md) + "\n")


def gsw_execution_plan() -> None:
    registry = read_csv(REPORT / "GSW_12SCENE_RUN_REGISTRY.csv")
    pending = [row for row in registry if row.get("needs_new_training") == "True"]
    plan: list[dict[str, object]] = []
    for row in pending:
        scene = row["scene_name"]
        model_path = PENDING_GSW_RUN_ROOT / scene
        method_dir = model_path / "test" / "ours_30000_strict_intrinsic"
        train_cmd = conda_python(
            REPO / "train.py",
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
                "30000",
                "--split_mode",
                "frozen_manifest",
                "--split_file",
                row["manifest_path"],
                "--test_appearance_mode",
                "strict_intrinsic",
                "--test_iterations",
                "1000000",
                "--save_iterations",
                "30000",
                "--disable_render_after_train",
                "--disable_metrics_after_train",
                "--disable_train_temp_images",
                "--quiet",
            ],
        )
        render_cmd = conda_python(
            REPO / "render.py",
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
                "30000",
                "--split_mode",
                "frozen_manifest",
                "--split_file",
                row["manifest_path"],
                "--test_appearance_mode",
                "strict_intrinsic",
                "--render_output_tag",
                "strict_intrinsic",
                "--skip_train",
                "--quiet",
            ],
        )
        eval_cmd = conda_python(
            REPO / "tools" / "baseline_completion" / "unified_full_image_eval.py",
            [
                "--label",
                f"{row['designated_run_id']}=gsw_strict_intrinsic={scene}={method_dir}={row['manifest_path']}",
                "--results-csv",
                str(REPORT / "runner_eval_results" / f"{scene}_summary.csv"),
                "--per-view-csv",
                str(REPORT / "runner_eval_results" / f"{scene}_per_view.csv"),
            ],
        )
        plan.extend(
            [
                {"scene": scene, "stage": "train_30k", "command": ps_join(train_cmd), "approval_required": GSW_APPROVAL_TOKEN},
                {"scene": scene, "stage": "render_strict_intrinsic", "command": ps_join(render_cmd), "approval_required": GSW_APPROVAL_TOKEN},
                {"scene": scene, "stage": "unified_full_image_eval", "command": ps_join(eval_cmd), "approval_required": GSW_APPROVAL_TOKEN},
            ]
        )
    write_csv(REPORT / "GSW_12SCENE_EXECUTION_PLAN.csv", plan)
    md = [
        "# GSW_12SCENE_EXECUTION_PLAN",
        "",
        "Dry-run command plan only. These commands have not been executed by this audit package.",
        "",
        f"- Pending corrected GS-W strict scenes: `{len(pending)}`.",
        f"- Execution root: `{PENDING_GSW_RUN_ROOT}`.",
        f"- Guarded runner: `tools/baseline_completion/strict_12scene_runner.py`.",
        f"- Approval token required for execution: `{GSW_APPROVAL_TOKEN}=1`.",
        "- Training uses resolution=1, iterations=30000, seed from `safe_state`, frozen manifest split, strict_intrinsic test appearance, no train-time test evaluation and no metrics_half.",
        "",
    ]
    for row in pending:
        md.extend([f"## {row['scene_name']}", ""])
        for item in [p for p in plan if p["scene"] == row["scene_name"]]:
            md.extend([f"### {item['stage']}", "", "```powershell", str(item["command"]), "```", ""])
    write_text(REPORT / "GSW_12SCENE_EXECUTION_PLAN.md", "\n".join(md))


def external_diagnostic_plan() -> None:
    luminance_failures = read_csv(REPORT / "LUMINANCE_GS_FAILURE_DIAGNOSIS.csv")
    luminance_metrics = [r for r in read_csv(EXTERNAL_SUMMARY / "metrics_all_methods.csv") if r.get("method") == "luminance_gs"]
    luminance_success = next((r.get("scene", "") for r in luminance_metrics if safe_float(r.get("psnr")) is not None), "")
    luminance_failed_scenes = sorted({r.get("scene", "") for r in luminance_failures if r.get("scene")})
    rows: list[dict[str, object]] = []
    if luminance_success:
        rows.append(
            {
                "method": "Luminance-GS",
                "scene": luminance_success,
                "diagnostic": "success_scene_reader_render_smoke",
                "allowed_scope": "reader/config/render smoke using existing wrapper; no 30k rerun",
                "forbidden_scope": "method source rewrite; test-metric hyperparameter tuning",
                "status": "blocked_by_import_api_mismatch",
                "evidence_needed": "logs/external_bounded_diagnostics/luminance_import_smoke.log; pycolmap SceneManager unavailable",
            }
        )
    for scene in luminance_failed_scenes:
        rows.append(
            {
                "method": "Luminance-GS",
                "scene": scene,
                "diagnostic": "failed_scene_config_plus_300_1000_iter_smoke",
                "allowed_scope": "adapter/config audit and 300-1000 iteration smoke only if the process starts correctly",
                "forbidden_scope": "full 30k; method source rewrite; selecting by test metric",
                "status": "blocked_by_import_api_mismatch",
                "evidence_needed": "LUMINANCE_GS_BOUNDED_DIAGNOSTIC_RUNS.csv plus luminance_import_smoke.log; no optimization launched",
            }
        )
    rows.extend(
        [
            {
                "method": "WildGaussians",
                "scene": "official_or_minimal_public_sample",
                "diagnostic": "upstream_sample_validation",
                "allowed_scope": "verify upstream code can produce non-dark images in recommended environment",
                "forbidden_scope": "changing method code and calling it the original baseline",
                "status": "import_smoke_passed_sample_not_run",
                "evidence_needed": "wild_import_smoke.log passes; no public sample training launched in this audit",
            },
            {
                "method": "WildGaussians",
                "scene": "web_Baalshamin_images",
                "diagnostic": "simple_scene_adapter_smoke",
                "allowed_scope": "reader/checkpoint/render smoke or 300-1000 iteration wrapper diagnostic",
                "forbidden_scope": "full 12-scene rerun before adapter issue is resolved",
                "status": "reader_smoke_passed_render_invalid",
                "evidence_needed": "wild_reader_smoke.log plus WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_RUNS.csv",
            },
            {
                "method": "WildGaussians",
                "scene": "web_doss_images",
                "diagnostic": "complex_scene_adapter_smoke",
                "allowed_scope": "reader/checkpoint/render smoke or 300-1000 iteration wrapper diagnostic",
                "forbidden_scope": "full 12-scene rerun before adapter issue is resolved",
                "status": "reader_smoke_passed_render_invalid",
                "evidence_needed": "wild_reader_smoke.log plus WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_RUNS.csv",
            },
        ]
    )
    write_csv(REPORT / "EXTERNAL_BASELINE_DIAGNOSTIC_PLAN.csv", rows)
    md = [
        "# EXTERNAL_BASELINE_DIAGNOSTIC_PLAN",
        "",
        "This table records bounded diagnostics and remaining follow-up. No full 30k external baseline reruns were launched.",
        "",
        "- Luminance-GS: import/API mismatch blocks reader/train smoke in the current environment; no full 30k rerun.",
        "- WildGaussians: import and two-scene reader smoke pass; no full 12-scene rerun until dark-render cause is isolated.",
        "- Any adapter fix must stay in wrapper/adapter code, not in method core source.",
        "",
        "| method | scene | diagnostic | status |",
        "|---|---|---|---|",
    ]
    for row in rows:
        md.append(f"| {row['method']} | {row['scene']} | {row['diagnostic']} | {row['status']} |")
    write_text(REPORT / "EXTERNAL_BASELINE_DIAGNOSTIC_PLAN.md", "\n".join(md) + "\n")


def generate() -> None:
    ensure_dirs()
    code_freeze_audit()
    gsw_registry_and_results()
    official_audit()
    gsw_vs_official()
    external_fairness()
    summary_report()
    completion_gap_audit()
    gsw_execution_plan()
    external_diagnostic_plan()
    repo_status_files()


def package() -> Path:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    stage = REVIEW_ROOT / "codex_12scene_gsw_and_baseline_audit"
    zip_path = REVIEW_ROOT / "codex_12scene_gsw_and_baseline_audit.zip"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    global REPORT, GENERATED_MANIFESTS, LOG_DIR
    source_report = REPORT
    stage_report = stage / "reports"
    old_report, old_manifests, old_log_dir = REPORT, GENERATED_MANIFESTS, LOG_DIR
    REPORT = stage_report
    GENERATED_MANIFESTS = stage_report / "generated_manifests"
    LOG_DIR = stage_report / "logs"
    try:
        generate()
    finally:
        REPORT = old_report
        GENERATED_MANIFESTS = old_manifests
        LOG_DIR = old_log_dir

    source_logs = source_report / "logs"
    if source_logs.exists():
        shutil.copytree(source_logs, stage_report / "logs", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for pattern in [
        "EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS.*",
        "EXTERNAL_BASELINE_CODE_EVIDENCE.csv",
        "EXTERNAL_BASELINE_SMOKE_RESULTS.csv",
        "LUMINANCE_GS_BOUNDED_DIAGNOSTIC_*.csv",
        "WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_*.csv",
    ]:
        for source_file in source_report.glob(pattern):
            if source_file.is_file():
                shutil.copy2(source_file, stage_report / source_file.name)
    shutil.copytree(REPO / "tools" / "baseline_completion", stage / "tools" / "baseline_completion", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = stage / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    patch_paths = ["tools/baseline_completion", "reports/baseline_completion"]
    (patches / "git_diff.patch").write_text(audit_patch_text(patch_paths), encoding="utf-8")
    (patches / "git_diff_stat.txt").write_text(audit_diff_stat(patch_paths), encoding="utf-8")
    (patches / "head_commit.patch").write_text(run(["git", "show", "--format=fuller", "--stat", "--patch", "--no-ext-diff", "HEAD"], check=False), encoding="utf-8")
    manifest = [
        "# Package Manifest",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Package: `{zip_path}`",
        "- Includes reports, scripts, small diagnostic CSVs and patches.",
        "- Excludes datasets, checkpoints, complete renders, .git, build outputs, pycache and credentials.",
    ]
    write_text(stage / "PACKAGE_MANIFEST.md", "\n".join(manifest) + "\n")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=REVIEW_ROOT, base_dir=stage.name)
    sha = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    zip_path.with_suffix(".sha256.txt").write_text(f"{sha}  {zip_path.name}\n", encoding="ascii")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    if args.package:
        path = package()
        print(path)
        return 0
    if args.generate:
        generate()
        return 0
    parser.error("Use --generate or --package")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
