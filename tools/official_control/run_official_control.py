from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from official_control_common import (
    OFFICIAL_ROOT,
    REPORT_DIR,
    RUN_ROOT,
    SCENE_NAME,
    ensure_dirs,
    official_render_command,
    official_train_command,
    ps_join,
    run_logged,
    write_csv,
    write_text,
)


def parse_gaussian_count(model_path: Path, iteration: int) -> int | None:
    ply = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        return None
    try:
        with ply.open("rb") as handle:
            for raw in handle:
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        return None
    return None


def checkpoint_size(model_path: Path, iteration: int) -> int:
    total = 0
    for path in (model_path / "point_cloud" / f"iteration_{iteration}").rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def run_train(label: str, iterations: int) -> dict[str, object]:
    model_path = RUN_ROOT / label / SCENE_NAME
    log_dir = REPORT_DIR / "logs"
    cmd = official_train_command(model_path, iterations)
    record = run_logged(
        f"{label}_train_{iterations}",
        cmd,
        OFFICIAL_ROOT,
        log_dir / f"{label}_train_{iterations}.stdout.log",
        log_dir / f"{label}_train_{iterations}.stderr.log",
        poll_gpu=True,
        gpu_poll_sec=10.0,
    )
    row = {
        "label": label,
        "stage": "train",
        "iterations": iterations,
        "model_path": str(model_path),
        "command": ps_join(cmd),
        "returncode": record.returncode,
        "duration_sec": round(record.duration_sec, 3),
        "duration_min": round(record.duration_sec / 60.0, 3),
        "peak_gpu_mem_mb": record.peak_gpu_mem_mb,
        "gaussian_count": parse_gaussian_count(model_path, iterations),
        "checkpoint_size_bytes": checkpoint_size(model_path, iterations),
        "stdout_log": str(record.stdout_log),
        "stderr_log": str(record.stderr_log),
        "start_time": datetime.fromtimestamp(record.start_time).isoformat(),
        "end_time": datetime.fromtimestamp(record.end_time).isoformat(),
    }
    config_md = [
        f"# {label}_CONFIG",
        "",
        f"- Model path: `{model_path}`",
        f"- Iterations: `{iterations}`",
        f"- Return code: `{record.returncode}`",
        f"- Duration seconds: `{row['duration_sec']}`",
        f"- Peak GPU memory MB: `{record.peak_gpu_mem_mb}`",
        f"- Gaussian count: `{row['gaussian_count']}`",
        f"- Checkpoint size bytes: `{row['checkpoint_size_bytes']}`",
        "",
        "## Train Command",
        "",
        "```text",
        ps_join(cmd),
        "```",
        "",
        "## Protocol Flags",
        "",
        "- `--eval` is enabled.",
        "- `--train_test_exp` is not present.",
        "- `--antialiasing` is not present.",
        "- `--depths` is not present.",
        "- `--optimizer_type default` is used.",
        "- `--test_iterations 1000000` prevents train-time test evaluation.",
    ]
    write_text(REPORT_DIR / f"{label}_CONFIG.md", "\n".join(config_md) + "\n")
    if record.returncode != 0:
        raise RuntimeError(f"{label} train failed with return code {record.returncode}; see {record.stderr_log}")
    return row


def run_render(label: str, iteration: int) -> dict[str, object]:
    model_path = RUN_ROOT / label / SCENE_NAME
    log_dir = REPORT_DIR / "logs"
    cmd = official_render_command(model_path, iteration, skip_train=True)
    record = run_logged(
        f"{label}_render_{iteration}",
        cmd,
        OFFICIAL_ROOT,
        log_dir / f"{label}_render_{iteration}.stdout.log",
        log_dir / f"{label}_render_{iteration}.stderr.log",
        poll_gpu=True,
        gpu_poll_sec=5.0,
    )
    method_dir = model_path / "test" / f"ours_{iteration}"
    render_count = len(list((method_dir / "renders").glob("*.png"))) if method_dir.exists() else 0
    gt_count = len(list((method_dir / "gt").glob("*.png"))) if method_dir.exists() else 0
    row = {
        "label": label,
        "stage": "render",
        "iterations": iteration,
        "model_path": str(model_path),
        "method_dir": str(method_dir),
        "command": ps_join(cmd),
        "returncode": record.returncode,
        "duration_sec": round(record.duration_sec, 3),
        "duration_min": round(record.duration_sec / 60.0, 3),
        "peak_gpu_mem_mb": record.peak_gpu_mem_mb,
        "render_count": render_count,
        "gt_count": gt_count,
        "stdout_log": str(record.stdout_log),
        "stderr_log": str(record.stderr_log),
    }
    if record.returncode != 0:
        raise RuntimeError(f"{label} render failed with return code {record.returncode}; see {record.stderr_log}")
    return row


def smoke_report(rows: list[dict[str, object]]) -> None:
    write_csv(REPORT_DIR / "OFFICIAL_SMOKE_RESULTS.csv", rows)
    md = [
        "# OFFICIAL_SMOKE_RESULTS",
        "",
        "| label | stage | iter | return | renders | gt | duration min | peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['label']} | {row['stage']} | {row['iterations']} | {row['returncode']} | "
            f"{row.get('render_count', '')} | {row.get('gt_count', '')} | {row['duration_min']} | {row.get('peak_gpu_mem_mb', '')} |"
        )
    md.extend(
        [
            "",
            "## Commands",
            "",
            *[f"- `{row['label']} {row['stage']}`: `{row['command']}`" for row in rows],
            "",
            "No official source files were patched for these smoke tests.",
        ]
    )
    write_text(REPORT_DIR / "OFFICIAL_SMOKE_RESULTS.md", "\n".join(md) + "\n")


def repeated_commands_report() -> None:
    lines = [
        "# OFFICIAL_REPEATED_COMMANDS",
        "",
        "All commands use the clean official 3DGS worktree and the same Trackmobile LLFF hold-8 split.",
        "",
    ]
    for label in ["O1", "O2", "O3"]:
        model_path = RUN_ROOT / label / SCENE_NAME
        lines.extend(
            [
                f"## {label} Train",
                "",
                "```text",
                ps_join(official_train_command(model_path, 30000)),
                "```",
                "",
                f"## {label} Render",
                "",
                "```text",
                ps_join(official_render_command(model_path, 30000, skip_train=True)),
                "```",
                "",
            ]
        )
    write_text(REPORT_DIR / "OFFICIAL_REPEATED_COMMANDS.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official 3DGS strict smoke/full controls")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--render-full", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    rows = []
    if args.smoke:
        for label, iterations in [("smoke_10", 10), ("smoke_300", 300)]:
            rows.append(run_train(label, iterations))
            rows.append(run_render(label, iterations))
        smoke_report(rows)
    if args.full:
        for label in ["O1", "O2", "O3"]:
            rows.append(run_train(label, 30000))
        write_csv(REPORT_DIR / "OFFICIAL_3DGS_REPEATED_SUMMARY_RAW.csv", rows)
    if args.render_full:
        for label in ["O1", "O2", "O3"]:
            rows.append(run_render(label, 30000))
        write_csv(REPORT_DIR / "OFFICIAL_3DGS_RENDER_SUMMARY_RAW.csv", rows)
    repeated_commands_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

