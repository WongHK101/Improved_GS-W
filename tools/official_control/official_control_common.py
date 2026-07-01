from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMPROVED_ROOT = Path(r"G:\wl3dgs\Improved_GS-W")
OFFICIAL_ROOT = Path(r"G:\wl3dgs\Official_3DGS_Strict_Control")
REPORT_DIR = IMPROVED_ROOT / "reports" / "official_control"
RUN_ROOT = Path(r"G:\wl3dgs\3dgs_runs\official_3dgs_strict_repeated_30k_20260701")
REVIEW_ROOT = Path(r"G:\WL3DGS\gpt_review_packages")
SCENE_NAME = "self_Trackmobile_4650TM_Mobile_Railcar_Mover"
SCENE_PATH = Path(r"G:\wl3dgs\3dgs_undistorted\max1600") / SCENE_NAME
SPLIT_MANIFEST = Path(r"G:\WL3DGS\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json")
GSW_RUN_ROOT = Path(r"G:\wl3dgs\3dgs_runs\gsw_strict_baseline_v2_repeated_30k_20260630")
EXPECTED_MANIFEST_SHA256 = "c65cc61677ffb46cd329d57fc72d30ed40d42e0f9a47d75c5a0931950c9f2cd7"
EXPECTED_TRAIN_IMAGES = [
    "0002.jpg",
    "0003.jpg",
    "0004.jpg",
    "0005.jpg",
    "0006.jpg",
    "0007.jpg",
    "0008.jpg",
    "0010.jpg",
    "0011.jpg",
    "0012.jpg",
    "0013.jpg",
    "0014.jpg",
    "0015.jpg",
]
EXPECTED_TEST_IMAGES = ["0001.jpg", "0009.jpg"]


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    return sha256_bytes("".join(f"{line}\n" for line in lines).encode("utf-8"))


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
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
        result = subprocess.CompletedProcess(cmd, 127, "", str(exc))
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def git_output(repo: Path, args: list[str]) -> str:
    result = run_cmd(["git", "-C", str(repo), *args], check=False)
    return (result.stdout + result.stderr).strip()


def conda_python_args(script_or_dash: str | Path, extra: list[str] | None = None) -> list[str]:
    target = str(script_or_dash)
    if os.environ.get("CONDA_DEFAULT_ENV") == "3dgs":
        return [sys.executable, target, *(extra or [])]
    return ["conda", "run", "-n", "3dgs", "--no-capture-output", "python", target, *(extra or [])]


def official_train_command(model_path: Path, iterations: int) -> list[str]:
    return conda_python_args(
        OFFICIAL_ROOT / "train.py",
        [
            "--source_path",
            str(SCENE_PATH),
            "--model_path",
            str(model_path),
            "--eval",
            "--resolution",
            "1",
            "--iterations",
            str(iterations),
            "--optimizer_type",
            "default",
            "--test_iterations",
            "1000000",
            "--save_iterations",
            str(iterations),
            "--disable_viewer",
            "--quiet",
        ],
    )


def official_render_command(model_path: Path, iteration: int, skip_train: bool = True) -> list[str]:
    args = [
        "--source_path",
        str(SCENE_PATH),
        "--model_path",
        str(model_path),
        "--eval",
        "--resolution",
        "1",
        "--iteration",
        str(iteration),
        "--quiet",
    ]
    if skip_train:
        args.append("--skip_train")
    return conda_python_args(OFFICIAL_ROOT / "render.py", args)


def ps_join(cmd: list[str]) -> str:
    quoted = []
    for arg in cmd:
        if re.search(r"\s", arg) or "\\" in arg or ":" in arg:
            quoted.append('"' + arg.replace('"', '\\"') + '"')
        else:
            quoted.append(arg)
    return " ".join(quoted)


def nvidia_smi_memory_mb() -> int | None:
    result = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
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


def run_logged(
    label: str,
    cmd: list[str],
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    poll_gpu: bool = True,
    gpu_poll_sec: float = 10.0,
) -> CommandRecord:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    peak = nvidia_smi_memory_mb() if poll_gpu else None
    with stdout_log.open("w", encoding="utf-8", errors="replace") as out, stderr_log.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        out.write(f"# label: {label}\n# command: {ps_join(cmd)}\n# cwd: {cwd}\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
        while proc.poll() is None:
            if poll_gpu:
                mem = nvidia_smi_memory_mb()
                if mem is not None:
                    peak = mem if peak is None else max(peak, mem)
            time.sleep(gpu_poll_sec)
        returncode = proc.returncode
    end = time.time()
    return CommandRecord(label, cmd, cwd, stdout_log, stderr_log, start, end, returncode, peak)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_small_file(src: Path, dst: Path, max_bytes: int = 2_000_000) -> bool:
    if not src.exists() or not src.is_file() or src.stat().st_size > max_bytes:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True
