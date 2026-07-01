from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


IMPROVED_ROOT = Path(r"G:\wl3dgs\Improved_GS-W")
OFFICIAL_ROOT = Path(r"G:\wl3dgs\Official_3DGS_Strict_Control")
DATA_ROOT = Path(r"G:\wl3dgs\3dgs_undistorted\max1600")
SPLIT_ROOT = Path(r"G:\wl3dgs\splits\max1600_llffhold8_v1")
REPORT_DIR = IMPROVED_ROOT / "reports" / "two_scene_selection"
REVIEW_ROOT = Path(r"G:\WL3DGS\gpt_review_packages")
RUN_ROOT = Path(r"G:\wl3dgs\3dgs_runs\two_scene_smoke_20260701")
TRACKMOBILE = "self_Trackmobile_4650TM_Mobile_Railcar_Mover"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_cmd(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))


def git_output(repo: Path, args: list[str]) -> str:
    result = run_cmd(["git", "-C", str(repo), *args])
    return (result.stdout + result.stderr).strip()


def python_cmd(script: Path, args: list[str] | None = None) -> list[str]:
    if os.environ.get("CONDA_DEFAULT_ENV") == "3dgs":
        return [sys.executable, str(script), *(args or [])]
    return ["conda", "run", "-n", "3dgs", "--no-capture-output", "python", str(script), *(args or [])]


def ps_join(cmd: list[str]) -> str:
    out = []
    for arg in cmd:
        if any(ch.isspace() for ch in arg) or "\\" in arg or ":" in arg:
            out.append('"' + arg.replace('"', '\\"') + '"')
        else:
            out.append(arg)
    return " ".join(out)


def scene_names(include_trackmobile: bool = False) -> list[str]:
    names = sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir())
    if include_trackmobile:
        return names
    return [name for name in names if name != TRACKMOBILE]


def split_files(scene: str) -> dict[str, Path]:
    base = SPLIT_ROOT / scene
    return {
        "dir": base,
        "train": base / "train.txt",
        "test": base / "test.txt",
        "all": base / "all_registered.txt",
        "meta": base / "split_meta.json",
    }


def report_manifest_path(scene: str) -> Path:
    return REPORT_DIR / "generated_manifests" / f"{scene}_SPLIT.json"


def write_report_manifest(scene: str) -> Path:
    all_names, train, test, meta = load_split_names(scene)
    path = report_manifest_path(scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": scene,
        "protocol": "llff_hold_8",
        "ordering_rule": "sort registered COLMAP image names lexicographically; assign test if zero-based sorted index % 8 == 0",
        "source": {
            "scene_path": str(DATA_ROOT / scene),
            "frozen_split_dir": str(SPLIT_ROOT / scene),
            "split_meta": str(SPLIT_ROOT / scene / "split_meta.json"),
        },
        "registered_images": all_names,
        "train_images": train,
        "test_images": test,
        "counts": {"registered": len(all_names), "train": len(train), "test": len(test)},
        "legacy_source_hashes": {
            "all_sha256": meta.get("all_sha256", ""),
            "train_sha256": meta.get("train_sha256", ""),
            "test_sha256": meta.get("test_sha256", ""),
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_split_names(scene: str) -> tuple[list[str], list[str], list[str], dict]:
    files = split_files(scene)
    all_names = [line.strip() for line in files["all"].read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [line.strip() for line in files["train"].read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [line.strip() for line in files["test"].read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = read_json(files["meta"])
    return all_names, train, test, meta


def source_type(scene: str) -> str:
    if scene.startswith("self_"):
        return "self-captured"
    if scene.startswith("web_"):
        return "network/public scenic"
    return "unknown"


def scene_category(scene: str) -> str:
    mapping = {
        "self_3000t_Press": "industrial heritage press",
        "self_CLG899III_Wheel_Loader": "industrial vehicle / loader",
        "self_double-action_press": "industrial heritage press",
        "self_Steam_Locomotive": "industrial heritage locomotive",
        "self_Trackmobile_4650TM_Mobile_Railcar_Mover": "industrial rail vehicle",
        "web_Baalshamin_images": "archaeological temple / heritage ruins",
        "web_cyprus_images": "heritage architecture / scenic landmark",
        "web_doss_images": "public scenic/heritage image collection",
        "web_metopa_images": "heritage relief/sculpture",
        "web_statue_images": "statue/sculpture",
        "web_Terrestrial": "public scenic/heritage image collection",
        "web_Trento_Duomo_images": "cathedral / historic architecture",
    }
    return mapping.get(scene, "unknown")


def load_colmap_counts(scene_path: Path) -> dict[str, int | str]:
    sparse = scene_path / "sparse" / "0"
    out: dict[str, int | str] = {"colmap_points": "", "colmap_images": "", "colmap_cameras": ""}
    try:
        import sys as _sys

        _sys.path.insert(0, str(OFFICIAL_ROOT))
        from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary

        extr = read_extrinsics_binary(str(sparse / "images.bin"))
        intr = read_intrinsics_binary(str(sparse / "cameras.bin"))
        pts, _, _ = read_points3D_binary(str(sparse / "points3D.bin"))
        out["colmap_images"] = len(extr)
        out["colmap_cameras"] = len(intr)
        out["colmap_points"] = int(pts.shape[0])
    except Exception as exc:
        out["colmap_error"] = repr(exc)
    return out
