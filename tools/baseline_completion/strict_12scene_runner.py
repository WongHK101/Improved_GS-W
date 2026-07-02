from __future__ import annotations

import argparse
import csv
import os
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
LOG_DIR = REPORT / "logs" / "strict_12scene_runner"
APPROVAL_TOKEN = "GPT_APPROVED_12SCENE_GSW_30K"


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


def model_path_for(row: dict[str, str]) -> Path:
    pending_path = Path(row["output_path"])
    if pending_path.name == row["scene_name"] and pending_path.parent.name.endswith("_pending_gpt"):
        return RUN_ROOT / row["scene_name"]
    return pending_path


def train_command(row: dict[str, str], iterations: int = 30000) -> list[str]:
    scene = row["scene_name"]
    return python_cmd(
        REPO / "train.py",
        [
            "--source_path",
            str(DATA_ROOT / scene),
            "--scene_name",
            scene,
            "--model_path",
            str(model_path_for(row)),
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
            str(model_path_for(row)),
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
    method_dir = model_path_for(row) / "test" / f"ours_{iteration}_strict_intrinsic"
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
        plan.extend(
            [
                {"scene": scene, "stage": "train_30k", "command": ps_join(train_command(row, 30000))},
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


def run_logged(label: str, cmd: list[str], cwd: Path) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_log = LOG_DIR / f"{label}.stdout.log"
    stderr_log = LOG_DIR / f"{label}.stderr.log"
    with stdout_log.open("w", encoding="utf-8", errors="replace") as out, stderr_log.open("w", encoding="utf-8", errors="replace") as err:
        out.write(f"# start: {datetime.now().isoformat(timespec='seconds')}\n# cwd: {cwd}\n# command: {ps_join(cmd)}\n")
        out.flush()
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=out, stderr=err, text=True)
        while proc.poll() is None:
            time.sleep(10)
        out.write(f"\n# end: {datetime.now().isoformat(timespec='seconds')}\n# returncode: {proc.returncode}\n")
        return int(proc.returncode)


def execute(rows: list[dict[str, str]], scenes: set[str]) -> int:
    if os.environ.get(APPROVAL_TOKEN) != "1":
        raise RuntimeError(f"Refusing to execute long runs. Set environment variable {APPROVAL_TOKEN}=1 after GPT approval.")
    selected = [row for row in rows if not scenes or row["scene_name"] in scenes]
    for row in selected:
        scene = row["scene_name"]
        for stage, command in [
            ("train_30k", train_command(row, 30000)),
            ("render_strict_intrinsic", render_command(row, 30000)),
            ("unified_eval", eval_command(row, 30000)),
        ]:
            rc = run_logged(f"{scene}_{stage}", command, REPO)
            if rc != 0:
                return rc
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial corrected GS-W strict 12-scene runner.")
    parser.add_argument("--dry-run", action="store_true", help="Write command plan only.")
    parser.add_argument("--execute", action="store_true", help="Run pending scenes serially. Requires GPT approval token in the environment.")
    parser.add_argument("--scene", action="append", default=[], help="Restrict execution to one scene. May be repeated.")
    args = parser.parse_args()

    rows = pending_rows()
    write_plan(rows)
    if args.execute:
        return execute(rows, set(args.scene))
    print(REPORT / "GSW_12SCENE_EXECUTION_PLAN.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
