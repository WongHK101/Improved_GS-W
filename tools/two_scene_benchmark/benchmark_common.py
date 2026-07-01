from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


IMPROVED_ROOT = Path(r"G:\wl3dgs\Improved_GS-W")
OFFICIAL_ROOT = Path(r"G:\wl3dgs\Official_3DGS_Strict_Control")
DATA_ROOT = Path(r"G:\wl3dgs\3dgs_undistorted\max1600")
SELECTION_REPORT_DIR = IMPROVED_ROOT / "reports" / "two_scene_selection"
REPORT_DIR = IMPROVED_ROOT / "reports" / "two_scene_benchmark"
REVIEW_ROOT = Path(r"G:\WL3DGS\gpt_review_packages")
RUN_ROOT = Path(r"G:\wl3dgs\3dgs_runs\two_scene_strict_screening_20260701")
BASELINE_COMMIT = "ddc6d8702b2e838dc989d612ca23fb311b79f280"
BASELINE_TAG = "gsw-strict-baseline-v2"
FREEZE_TAG = "gsw-two-scene-screening-v1"
OFFICIAL_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"
SCENES = {
    "H": "self_Steam_Locomotive",
    "M": "web_Terrestrial",
}


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def manifest_path(scene: str) -> Path:
    return SELECTION_REPORT_DIR / "generated_manifests" / f"{scene}_SPLIT.json"


def manifest_hash(scene: str) -> str:
    return sha256_file(manifest_path(scene))


def scene_path(scene: str) -> Path:
    return DATA_ROOT / scene


def gsw_checkpoint_dir(model_path: Path, iteration: int) -> Path:
    return model_path / "ckpts_point_cloud" / f"iteration_{iteration}"


def official_checkpoint_dir(model_path: Path, iteration: int) -> Path:
    return model_path / "point_cloud" / f"iteration_{iteration}"


def checkpoint_complete(model_path: Path, method: str, iteration: int) -> bool:
    base = gsw_checkpoint_dir(model_path, iteration) if method == "gsw" else official_checkpoint_dir(model_path, iteration)
    if method == "gsw":
        required = ["point_cloud.ply", "map_generator.pth", "color_net.pth", "other_atrributes_dict.pth"]
    else:
        required = ["point_cloud.ply"]
    return all((base / name).exists() and (base / name).is_file() for name in required)


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def gpu_info() -> dict[str, object]:
    result = run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        timeout=30,
    )
    if result.returncode != 0:
        return {"gpu_name": "", "gpu_total_memory_mb": ""}
    first = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not first:
        return {"gpu_name": "", "gpu_total_memory_mb": ""}
    parts = [part.strip() for part in first.split(",")]
    return {
        "gpu_name": parts[0] if parts else "",
        "gpu_total_memory_mb": int(float(parts[1])) if len(parts) > 1 and parts[1] else "",
    }


def nvidia_smi_memory_mb() -> int | None:
    result = run_cmd(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        timeout=30,
    )
    if result.returncode != 0:
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line)))
        except ValueError:
            pass
    return max(values) if values else None


@dataclass
class RunRecord:
    label: str
    command: list[str]
    cwd: Path
    stdout_log: Path
    stderr_log: Path
    start_time: float
    end_time: float
    returncode: int
    peak_nvidia_smi_memory_mb: int | None

    @property
    def duration_sec(self) -> float:
        return self.end_time - self.start_time


def run_logged(label: str, cmd: list[str], cwd: Path, stdout_log: Path, stderr_log: Path, poll_sec: float = 10.0) -> RunRecord:
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
            time.sleep(poll_sec)
        returncode = int(proc.returncode)
    end = time.time()
    return RunRecord(label, cmd, cwd, stdout_log, stderr_log, start, end, returncode, peak)


def load_split(scene: str) -> tuple[list[str], list[str]]:
    manifest = read_json(manifest_path(scene))
    return list(manifest["train_images"]), list(manifest["test_images"])


def ply_vertex_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def iso_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
