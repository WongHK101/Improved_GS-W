from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
WL3DGS = REPO.parent
DATA_ROOT = WL3DGS / "3dgs_undistorted" / "max1600"
REPORT = REPO / "reports" / "baseline_completion"
REGISTRY = REPORT / "GSW_12SCENE_RUN_REGISTRY.csv"
RUN_ROOT = WL3DGS / "3dgs_runs" / "gsw_strict_12scene_single_run_20260702"
PILOT_ROOT = WL3DGS / "3dgs_runs" / "gsw_strict_12scene_pilots_20260702"
LOG_DIR = REPORT / "logs" / "strict_12scene_runner"
APPROVAL_TOKEN = "GPT_APPROVED_12SCENE_GSW_30K"
TOTAL_GPU_MEMORY_MB = 24564

PILOT_ITERATIONS = {
    "web_doss_images": 5000,
    "web_Trento_Duomo_images": 5000,
    "self_double-action_press": 3000,
    "web_cyprus_images": 3000,
    "self_CLG899III_Wheel_Loader": 3000,
    "web_metopa_images": 3000,
    "web_statue_images": 3000,
    "web_Baalshamin_images": 1000,
    "self_3000t_Press": 1000,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
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


def python_cmd(script: Path, args: list[str]) -> list[str]:
    if os.environ.get("CONDA_DEFAULT_ENV") == "3dgs":
        return [sys.executable, str(script), *args]
    return ["conda", "run", "-n", "3dgs", "--no-capture-output", "python", str(script), *args]


def ps_join(cmd: list[str]) -> str:
    out = []
    for arg in cmd:
        if any(ch.isspace() for ch in arg) or "\\" in arg or ":" in arg:
            out.append('"' + arg.replace('"', '\\"') + '"')
        else:
            out.append(arg)
    return " ".join(out)


def pending_rows() -> list[dict[str, str]]:
    rows = read_csv(REGISTRY)
    return [row for row in rows if row.get("needs_new_training") == "True"]


def model_path_for(row: dict[str, str], run_kind: str = "formal") -> Path:
    if run_kind == "pilot":
        return PILOT_ROOT / row["scene_name"] / f"iter_{PILOT_ITERATIONS[row['scene_name']]}"
    pending_path = Path(row["output_path"])
    if pending_path.name == row["scene_name"] and pending_path.parent.name.endswith("_pending_gpt"):
        return RUN_ROOT / row["scene_name"]
    return pending_path


def train_command(row: dict[str, str], iterations: int = 30000, run_kind: str = "formal") -> list[str]:
    scene = row["scene_name"]
    return python_cmd(
        REPO / "train.py",
        [
            "--source_path",
            str(DATA_ROOT / scene),
            "--scene_name",
            scene,
            "--model_path",
            str(model_path_for(row, run_kind)),
            "--resolution",
            "1",
            "--iterations",
            str(iterations),
            "--split_mode",
            "frozen_manifest",
            "--split_file",
            row["manifest_path"],
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


def render_command(row: dict[str, str], iteration: int = 30000) -> list[str]:
    scene = row["scene_name"]
    return python_cmd(
        REPO / "render.py",
        [
            "--source_path",
            str(DATA_ROOT / scene),
            "--scene_name",
            scene,
            "--model_path",
            str(model_path_for(row, "formal")),
            "--resolution",
            "1",
            "--iteration",
            str(iteration),
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


def eval_command(row: dict[str, str], iteration: int = 30000) -> list[str]:
    scene = row["scene_name"]
    method_dir = model_path_for(row, "formal") / "test" / f"ours_{iteration}_strict_intrinsic"
    return python_cmd(
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


def command_plan(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for row in rows:
        scene = row["scene_name"]
        pilot_iter = PILOT_ITERATIONS.get(scene, 1000)
        plan.extend(
            [
                {"scene": scene, "stage": f"pilot_{pilot_iter}", "command": ps_join(train_command(row, pilot_iter, "pilot"))},
                {"scene": scene, "stage": "train_30k", "command": ps_join(train_command(row, 30000, "formal"))},
                {"scene": scene, "stage": "render_strict_intrinsic", "command": ps_join(render_command(row, 30000))},
                {"scene": scene, "stage": "unified_eval", "command": ps_join(eval_command(row, 30000))},
            ]
        )
    return plan


def write_plan(rows: list[dict[str, str]]) -> None:
    plan = command_plan(rows)
    write_csv(REPORT / "GSW_12SCENE_EXECUTION_PLAN.csv", plan)
    md = [
        "# GSW_12SCENE_EXECUTION_PLAN",
        "",
        "This is a dry-run execution plan. It does not prove that the remaining runs were executed.",
        "",
        f"- Pending scenes: `{len(rows)}`",
        f"- Execution root: `{RUN_ROOT}`",
        f"- Pilot root: `{PILOT_ROOT}`",
        f"- Approval token required for execution: `{APPROVAL_TOKEN}`",
        "- Commands are serial; no parallel `conda run` jobs are launched by this runner.",
        "- Training uses `--test_iterations 1000000`, `--test_appearance_mode strict_intrinsic`, full-image rendering and the frozen manifest path from the registry.",
        "",
    ]
    for row in rows:
        scene_plan = [item for item in plan if item["scene"] == row["scene_name"]]
        md.extend([f"## {row['scene_name']}", ""])
        for item in scene_plan:
            md.extend([f"### {item['stage']}", "", "```powershell", str(item["command"]), "```", ""])
    write_text(REPORT / "GSW_12SCENE_EXECUTION_PLAN.md", "\n".join(md))


def gpu_memory_mb() -> tuple[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
        first = out.splitlines()[0]
        used, total = [int(part.strip()) for part in first.split(",")[:2]]
        return used, total
    except Exception:
        return -1, TOTAL_GPU_MEMORY_MB


def run_logged(label: str, cmd: list[str], cwd: Path) -> dict[str, object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_log = LOG_DIR / f"{label}.stdout.log"
    stderr_log = LOG_DIR / f"{label}.stderr.log"
    start = datetime.now()
    peak_used = -1
    total = TOTAL_GPU_MEMORY_MB
    with stdout_log.open("w", encoding="utf-8", errors="replace") as out, stderr_log.open("w", encoding="utf-8", errors="replace") as err:
        out.write(f"# start: {start.isoformat(timespec='seconds')}\n# cwd: {cwd}\n# command: {ps_join(cmd)}\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
        while proc.poll() is None:
            used, total = gpu_memory_mb()
            peak_used = max(peak_used, used)
            time.sleep(10)
        used, total = gpu_memory_mb()
        peak_used = max(peak_used, used)
        end = datetime.now()
        out.write(
            f"\n# end: {end.isoformat(timespec='seconds')}\n"
            f"# returncode: {proc.returncode}\n"
            f"# elapsed_seconds: {(end - start).total_seconds():.1f}\n"
            f"# peak_gpu_memory_used_mb: {peak_used}\n"
            f"# gpu_memory_total_mb: {total}\n"
        )
        return {
            "returncode": int(proc.returncode),
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "elapsed_seconds": f"{(end - start).total_seconds():.1f}",
            "peak_gpu_memory_used_mb": peak_used,
            "gpu_memory_total_mb": total,
        }


def checkpoint_dir(row: dict[str, str], iteration: int, run_kind: str) -> Path:
    return model_path_for(row, run_kind) / "ckpts_point_cloud" / f"iteration_{iteration}"


def gaussian_count_from_ply(path: Path) -> int:
    if not path.exists():
        return 0
    header = path.read_bytes()[:4096].decode("ascii", errors="ignore")
    match = re.search(r"element\s+vertex\s+(\d+)", header)
    return int(match.group(1)) if match else 0


def log_has_failure(stdout_log: str, stderr_log: str) -> tuple[bool, str]:
    text = ""
    for path in [Path(stdout_log), Path(stderr_log)]:
        if path.exists():
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        ("oom", r"CUDA out of memory|out of memory"),
        ("traceback", r"Traceback \(most recent call last\)|RuntimeError:|Exception:"),
        ("nan", r"(?i)(loss=nan|\bnan\b)"),
    ]
    hits = [name for name, pattern in patterns if re.search(pattern, text)]
    return bool(hits), ";".join(hits)


def inspect_training_result(row: dict[str, str], iteration: int, run_kind: str, run_result: dict[str, object]) -> dict[str, object]:
    ckpt = checkpoint_dir(row, iteration, run_kind)
    point_cloud = ckpt / "point_cloud.ply"
    required = ["point_cloud.ply", "map_generator.pth", "color_net.pth", "other_atrributes_dict.pth"]
    missing = [name for name in required if not (ckpt / name).exists()]
    has_failure, failure_hits = log_has_failure(str(run_result["stdout_log"]), str(run_result["stderr_log"]))
    peak = int(run_result.get("peak_gpu_memory_used_mb") or -1)
    total = int(run_result.get("gpu_memory_total_mb") or TOTAL_GPU_MEMORY_MB)
    mem_ratio = peak / total if peak >= 0 and total > 0 else -1
    gaussian_count = gaussian_count_from_ply(point_cloud)
    pass_status = (
        int(run_result["returncode"]) == 0
        and not missing
        and not has_failure
        and gaussian_count > 0
        and (mem_ratio < 0.90 if mem_ratio >= 0 else True)
    )
    return {
        "scene": row["scene_name"],
        "run_kind": run_kind,
        "iteration": iteration,
        "status": "PASS" if pass_status else "FAIL",
        "returncode": run_result["returncode"],
        "elapsed_seconds": run_result["elapsed_seconds"],
        "peak_gpu_memory_used_mb": peak,
        "gpu_memory_total_mb": total,
        "peak_gpu_memory_ratio": f"{mem_ratio:.4f}" if mem_ratio >= 0 else "",
        "checkpoint_dir": str(ckpt),
        "missing_checkpoint_files": ";".join(missing),
        "gaussian_count": gaussian_count,
        "failure_hits": failure_hits,
        "stdout_log": run_result["stdout_log"],
        "stderr_log": run_result["stderr_log"],
    }


def canonical_gate() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.baseline_completion import canonical_split_hash_audit

    canonical_split_hash_audit.generate(REPORT, REGISTRY)
    rows = read_csv(REPORT / "CANONICAL_SPLIT_HASHES.csv")
    track = [
        row
        for row in rows
        if row.get("scene_name") == "self_Trackmobile_4650TM_Mobile_Railcar_Mover" and row.get("status") == "ok"
    ]
    unique = {row.get("canonical_split_sha256") for row in track}
    if len(unique) != 1:
        raise RuntimeError("Trackmobile canonical split mismatch; refusing to run GS-W pilots/formal jobs.")


def existing_pilot_status() -> dict[str, str]:
    rows = read_csv(REPORT / "GSW_PILOT_RESULTS.csv") if (REPORT / "GSW_PILOT_RESULTS.csv").exists() else []
    return {row["scene"]: row["status"] for row in rows}


def execute_pilots(rows: list[dict[str, str]], scenes: set[str]) -> int:
    if os.environ.get(APPROVAL_TOKEN) != "1":
        raise RuntimeError(f"Refusing to execute long runs. Set environment variable {APPROVAL_TOKEN}=1 after GPT approval.")
    canonical_gate()
    selected = [row for row in rows if not scenes or row["scene_name"] in scenes]
    existing = existing_pilot_status()
    results = read_csv(REPORT / "GSW_PILOT_RESULTS.csv") if (REPORT / "GSW_PILOT_RESULTS.csv").exists() else []
    for row in selected:
        scene = row["scene_name"]
        if scene in existing:
            continue
        iteration = PILOT_ITERATIONS.get(scene, 1000)
        result = run_logged(f"{scene}_pilot_{iteration}", train_command(row, iteration, "pilot"), REPO)
        inspected = inspect_training_result(row, iteration, "pilot", result)
        results = [old for old in results if old.get("scene") != scene or old.get("run_kind") != "pilot"]
        results.append(inspected)
        write_csv(REPORT / "GSW_PILOT_RESULTS.csv", results)
        if inspected["status"] != "PASS":
            print(f"Pilot failed for {scene}; formal run will be skipped for this scene.")
    return 0


def execute_formal(rows: list[dict[str, str]], scenes: set[str]) -> int:
    if os.environ.get(APPROVAL_TOKEN) != "1":
        raise RuntimeError(f"Refusing to execute long runs. Set environment variable {APPROVAL_TOKEN}=1 after GPT approval.")
    canonical_gate()
    pilots = existing_pilot_status()
    selected = [row for row in rows if not scenes or row["scene_name"] in scenes]
    formal_results = read_csv(REPORT / "GSW_FORMAL_RUN_RESULTS.csv") if (REPORT / "GSW_FORMAL_RUN_RESULTS.csv").exists() else []
    for row in selected:
        scene = row["scene_name"]
        if pilots.get(scene) != "PASS":
            formal_results = [old for old in formal_results if old.get("scene") != scene]
            formal_results.append(
                {
                    "scene": scene,
                    "status": "SKIPPED_PILOT_NOT_PASS",
                    "notes": f"pilot_status={pilots.get(scene, 'missing')}",
                }
            )
            write_csv(REPORT / "GSW_FORMAL_RUN_RESULTS.csv", formal_results)
            continue
        if any(old.get("scene") == scene and old.get("stage") == "unified_eval" and old.get("status") == "PASS" for old in formal_results):
            continue
        existing_train = next((old for old in formal_results if old.get("scene") == scene and old.get("stage") == "train_30k"), None)
        if existing_train is None:
            train_result = run_logged(f"{scene}_train_30k", train_command(row, 30000, "formal"), REPO)
            inspected = inspect_training_result(row, 30000, "formal", train_result)
            inspected["stage"] = "train_30k"
            formal_results = [old for old in formal_results if not (old.get("scene") == scene and old.get("stage") == "train_30k")]
            formal_results.append(inspected)
            write_csv(REPORT / "GSW_FORMAL_RUN_RESULTS.csv", formal_results)
            existing_train = inspected
        if existing_train.get("status") != "PASS":
            continue
        for stage, command in [
            ("render_strict_intrinsic", render_command(row, 30000)),
            ("unified_eval", eval_command(row, 30000)),
        ]:
            run_result = run_logged(f"{scene}_{stage}", command, REPO)
            has_failure, failure_hits = log_has_failure(str(run_result["stdout_log"]), str(run_result["stderr_log"]))
            stage_status = "PASS" if int(run_result["returncode"]) == 0 and not has_failure else "FAIL"
            formal_results = [old for old in formal_results if not (old.get("scene") == scene and old.get("stage") == stage)]
            formal_results.append(
                {
                    "scene": scene,
                    "stage": stage,
                    "status": stage_status,
                    "returncode": run_result["returncode"],
                    "elapsed_seconds": run_result["elapsed_seconds"],
                    "peak_gpu_memory_used_mb": run_result["peak_gpu_memory_used_mb"],
                    "failure_hits": failure_hits,
                    "stdout_log": run_result["stdout_log"],
                    "stderr_log": run_result["stderr_log"],
                }
            )
            write_csv(REPORT / "GSW_FORMAL_RUN_RESULTS.csv", formal_results)
            if stage_status != "PASS":
                break
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial corrected GS-W strict 12-scene runner.")
    parser.add_argument("--dry-run", action="store_true", help="Write command plan only.")
    parser.add_argument("--execute", action="store_true", help="Run pending scenes serially: pilots first, then formal jobs that passed pilots.")
    parser.add_argument("--execute-pilots", action="store_true", help="Run only GPT-approved pilot jobs serially.")
    parser.add_argument("--execute-formal", action="store_true", help="Run only formal 30k jobs for scenes with PASS pilots.")
    parser.add_argument("--scene", action="append", default=[], help="Restrict execution to one scene. May be repeated.")
    args = parser.parse_args()

    rows = pending_rows()
    write_plan(rows)
    if args.execute or args.execute_pilots:
        rc = execute_pilots(rows, set(args.scene))
        if rc != 0 or args.execute_pilots:
            return rc
    if args.execute or args.execute_formal:
        return execute_formal(rows, set(args.scene))
    print(REPORT / "GSW_12SCENE_EXECUTION_PLAN.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
