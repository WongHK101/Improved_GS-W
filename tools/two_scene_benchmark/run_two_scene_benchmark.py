from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from benchmark_common import (
    BASELINE_COMMIT,
    BASELINE_TAG,
    FREEZE_TAG,
    IMPROVED_ROOT,
    OFFICIAL_COMMIT,
    OFFICIAL_ROOT,
    REPORT_DIR,
    REVIEW_ROOT,
    RUN_ROOT,
    SCENES,
    checkpoint_complete,
    dir_size_bytes,
    ensure_dirs,
    git_output,
    gpu_info,
    gsw_checkpoint_dir,
    iso_time,
    load_split,
    manifest_hash,
    manifest_path,
    official_checkpoint_dir,
    ps_join,
    python_cmd,
    run_cmd,
    run_logged,
    scene_path,
    sha256_file,
    write_csv,
    write_text,
    ply_vertex_count,
)


ITERATION = 30000
PILOT_ITERATION = 5000
LOG_DIR = REPORT_DIR / "logs"
TRACKMOBILE_RESULTS = IMPROVED_ROOT / "reports" / "official_control" / "UNIFIED_OFFICIAL_GSW_RESULTS.csv"
TRACKMOBILE_PER_VIEW = IMPROVED_ROOT / "reports" / "official_control" / "UNIFIED_OFFICIAL_GSW_PER_VIEW.csv"


RUN_METADATA: list[dict[str, object]] = []


def gsw_model_path(phase: str, role: str, run_label: str, scene: str) -> Path:
    return RUN_ROOT / phase / role / run_label / scene


def official_model_path(phase: str, role: str, run_label: str, scene: str) -> Path:
    return RUN_ROOT / phase / role / run_label / scene


def method_name(method: str) -> str:
    return "official_3dgs" if method == "official" else "gsw_strict_intrinsic"


def method_dir(method: str, model_path: Path, iteration: int = ITERATION) -> Path:
    suffix = "_strict_intrinsic" if method == "gsw" else ""
    return model_path / "test" / f"ours_{iteration}{suffix}"


def load_existing_metadata() -> list[dict[str, object]]:
    path = REPORT_DIR / "TWO_SCENE_SCREENING_RUN_METADATA.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_has_partial_output(model_path: Path, method: str, final_iter: int) -> bool:
    if not model_path.exists():
        return False
    return not checkpoint_complete(model_path, method, final_iter)


def environment_summary() -> dict[str, object]:
    py = run_cmd([sys.executable, "--version"], timeout=30)
    conda = run_cmd(
        [
            "conda",
            "run",
            "-n",
            "3dgs",
            "--no-capture-output",
            "python",
            "-c",
            "import sys,torch; print(sys.version.split()[0]); print(torch.__version__); print(torch.version.cuda)",
        ],
        timeout=120,
    )
    info = gpu_info()
    info.update(
        {
            "sys_executable": sys.executable,
            "python_version": py.stdout.strip() or py.stderr.strip(),
            "conda_3dgs_probe": conda.stdout.strip().replace("\n", "; "),
        }
    )
    return info


def write_runner_audit() -> None:
    env = environment_summary()
    md = [
        "# BENCHMARK_RUNNER_AUDIT",
        "",
        "## Runner Scope",
        "",
        "- New orchestration files live under `tools/two_scene_benchmark/`.",
        "- Reports live under `reports/two_scene_benchmark/`.",
        "- The runner invokes existing official/GS-W training entry points serially and records metadata.",
        "- It does not modify training source files, loss definitions, renderer behavior, split protocol, densification parameters, or appearance mode.",
        "",
        "## Resumability Rules",
        "",
        "- Each run has an independent output directory.",
        "- If the final checkpoint is complete, the runner records `skipped_existing_complete` and does not rerun.",
        "- If a target directory exists without the complete final checkpoint, it records `interrupted_existing_incomplete` and stops before treating that case as successful.",
        "- Process return codes come directly from `subprocess.Popen(...).returncode`.",
        "- All commands are launched serially by this Python process; no concurrent `conda run` jobs are started.",
        "",
        "## Pilot Diagnostics",
        "",
        "The GS-W 5000-iteration pilot is launched via `tools/two_scene_benchmark/gsw_pilot_train.py`, a diagnostics-only wrapper. It imports the unchanged `train.py`, expands trace sample iterations, and monkeypatches runtime methods only to count densification events and record CUDA memory. The wrapped functions call the original implementations and do not change parameters, tensors, losses, renderer outputs, optimizer steps, or return values.",
        "",
        "## Protocol Freeze",
        "",
        f"- Required GS-W baseline commit: `{BASELINE_COMMIT}`",
        f"- Required GS-W baseline tag: `{BASELINE_TAG}`",
        f"- Screening freeze tag: `{FREEZE_TAG}`",
        f"- Official worktree: `{OFFICIAL_ROOT}`",
        f"- Official commit: `{OFFICIAL_COMMIT}`",
        "- Official flags disable depth regularization with `--depth_l1_weight_init 0 --depth_l1_weight_final 0`.",
        "- Official does not pass `--train_test_exp`; antialiasing remains default false.",
        "- GS-W always uses `--split_mode frozen_manifest` and `--test_appearance_mode strict_intrinsic`.",
        "- All training commands use `--test_iterations 1000000`, so no train-time test evaluation is triggered for 5000 or 30000 iteration runs.",
        "",
        "## Environment Probe",
        "",
        "```json",
        json.dumps(env, indent=2),
        "```",
    ]
    write_text(REPORT_DIR / "BENCHMARK_RUNNER_AUDIT.md", "\n".join(md) + "\n")


def gsw_train_command(
    scene: str,
    model_path: Path,
    iterations: int,
    save_iterations: list[int],
    pilot: bool,
    trace_path: Path | None = None,
    checkpoint_audit_path: Path | None = None,
) -> list[str]:
    script = IMPROVED_ROOT / "tools" / "two_scene_benchmark" / "gsw_pilot_train.py" if pilot else IMPROVED_ROOT / "train.py"
    args = [
        "--source_path",
        str(scene_path(scene)),
        "--scene_name",
        scene,
        "--model_path",
        str(model_path),
        "--resolution",
        "1",
        "--iterations",
        str(iterations),
        "--split_mode",
        "frozen_manifest",
        "--split_file",
        str(manifest_path(scene)),
        "--test_appearance_mode",
        "strict_intrinsic",
        "--test_iterations",
        "1000000",
        "--save_iterations",
        *[str(item) for item in save_iterations],
        "--disable_render_after_train",
        "--disable_metrics_after_train",
        "--disable_train_temp_images",
        "--quiet",
    ]
    if pilot:
        args.extend(["--trace_output", str(trace_path), "--checkpoint_audit_output", str(checkpoint_audit_path)])
    return python_cmd(script, args)


def official_train_command(scene: str, model_path: Path, iterations: int) -> list[str]:
    return python_cmd(
        OFFICIAL_ROOT / "train.py",
        [
            "--source_path",
            str(scene_path(scene)),
            "--model_path",
            str(model_path),
            "--eval",
            "--resolution",
            "1",
            "--iterations",
            str(iterations),
            "--optimizer_type",
            "default",
            "--depth_l1_weight_init",
            "0",
            "--depth_l1_weight_final",
            "0",
            "--test_iterations",
            "1000000",
            "--save_iterations",
            str(iterations),
            "--disable_viewer",
            "--quiet",
        ],
    )


def official_render_command(scene: str, model_path: Path, iteration: int) -> list[str]:
    return python_cmd(
        OFFICIAL_ROOT / "render.py",
        [
            "--source_path",
            str(scene_path(scene)),
            "--model_path",
            str(model_path),
            "--eval",
            "--resolution",
            "1",
            "--iteration",
            str(iteration),
            "--skip_train",
            "--quiet",
        ],
    )


def gsw_render_command(scene: str, model_path: Path, iteration: int) -> list[str]:
    return python_cmd(
        IMPROVED_ROOT / "render.py",
        [
            "--source_path",
            str(scene_path(scene)),
            "--scene_name",
            scene,
            "--model_path",
            str(model_path),
            "--resolution",
            "1",
            "--iteration",
            str(iteration),
            "--split_mode",
            "frozen_manifest",
            "--split_file",
            str(manifest_path(scene)),
            "--test_appearance_mode",
            "strict_intrinsic",
            "--render_output_tag",
            "strict_intrinsic",
            "--skip_train",
            "--quiet",
        ],
    )


def record_run_metadata(row: dict[str, object]) -> None:
    global RUN_METADATA
    if not RUN_METADATA:
        RUN_METADATA = load_existing_metadata()
    RUN_METADATA = [old for old in RUN_METADATA if old.get("label") != row.get("label") or old.get("stage") != row.get("stage")]
    RUN_METADATA.append(row)
    write_csv(REPORT_DIR / "TWO_SCENE_SCREENING_RUN_METADATA.csv", RUN_METADATA)


def run_training(
    label: str,
    method: str,
    scene: str,
    model_path: Path,
    iterations: int,
    command: list[str],
    cwd: Path,
    trace_path: Path | None = None,
    checkpoint_audit_path: Path | None = None,
) -> dict[str, object]:
    final_complete = checkpoint_complete(model_path, method, iterations)
    status = "pending"
    record = None
    if final_complete:
        status = "skipped_existing_complete"
        returncode = 0
        start = end = 0.0
        peak = None
    elif run_has_partial_output(model_path, method, iterations):
        status = "interrupted_existing_incomplete"
        returncode = -100
        start = end = 0.0
        peak = None
    else:
        status = "ran"
        record = run_logged(
            label,
            command,
            cwd,
            LOG_DIR / f"{label}.stdout.log",
            LOG_DIR / f"{label}.stderr.log",
            poll_sec=10.0,
        )
        returncode = record.returncode
        start = record.start_time
        end = record.end_time
        peak = record.peak_nvidia_smi_memory_mb
        final_complete = checkpoint_complete(model_path, method, iterations)
    ckpt_dir = gsw_checkpoint_dir(model_path, iterations) if method == "gsw" else official_checkpoint_dir(model_path, iterations)
    ckpt_ply = ckpt_dir / "point_cloud.ply"
    gpu = gpu_info()
    row = {
        "stage": "train",
        "label": label,
        "method": method,
        "scene": scene,
        "model_path": str(model_path),
        "iterations": iterations,
        "status": status,
        "returncode": returncode,
        "start_time": iso_time(start) if start else "",
        "end_time": iso_time(end) if end else "",
        "duration_sec": round(end - start, 3) if start and end else "",
        "duration_min": round((end - start) / 60.0, 3) if start and end else "",
        "stdout_log": str(record.stdout_log) if record else "",
        "stderr_log": str(record.stderr_log) if record else "",
        "command": ps_join(command),
        "git_commit": git_output(IMPROVED_ROOT, ["rev-parse", "HEAD"]) if method == "gsw" else git_output(OFFICIAL_ROOT, ["rev-parse", "HEAD"]),
        "environment": json.dumps(environment_summary(), sort_keys=True),
        "manifest_hash": manifest_hash(scene),
        "gpu_name": gpu.get("gpu_name", ""),
        "gpu_total_memory_mb": gpu.get("gpu_total_memory_mb", ""),
        "peak_nvidia_smi_memory_mb": peak if peak is not None else "",
        "checkpoint_complete": final_complete,
        "checkpoint_size_bytes": dir_size_bytes(ckpt_dir),
        "gaussian_count": ply_vertex_count(ckpt_ply) or "",
        "trace_path": str(trace_path) if trace_path else "",
        "checkpoint_audit_path": str(checkpoint_audit_path) if checkpoint_audit_path else "",
    }
    record_run_metadata(row)
    return row


def expected_render_count(scene: str) -> int:
    return len(load_split(scene)[1])


def render_complete(method: str, scene: str, model_path: Path, iteration: int) -> bool:
    render_dir = method_dir(method, model_path, iteration) / "renders"
    gt_dir = method_dir(method, model_path, iteration) / "gt"
    return render_dir.exists() and gt_dir.exists() and len(list(render_dir.glob("*.png"))) == expected_render_count(scene)


def run_render(label: str, method: str, scene: str, model_path: Path, iteration: int) -> dict[str, object]:
    command = gsw_render_command(scene, model_path, iteration) if method == "gsw" else official_render_command(scene, model_path, iteration)
    cwd = IMPROVED_ROOT if method == "gsw" else OFFICIAL_ROOT
    if render_complete(method, scene, model_path, iteration):
        status = "skipped_existing_complete"
        returncode = 0
        record = None
    else:
        status = "ran"
        record = run_logged(
            f"{label}_render",
            command,
            cwd,
            LOG_DIR / f"{label}_render.stdout.log",
            LOG_DIR / f"{label}_render.stderr.log",
            poll_sec=5.0,
        )
        returncode = record.returncode
    out_dir = method_dir(method, model_path, iteration)
    row = {
        "stage": "render",
        "label": label,
        "method": method,
        "scene": scene,
        "model_path": str(model_path),
        "method_dir": str(out_dir),
        "iteration": iteration,
        "status": status,
        "returncode": returncode,
        "duration_sec": round(record.duration_sec, 3) if record else "",
        "duration_min": round(record.duration_sec / 60.0, 3) if record else "",
        "stdout_log": str(record.stdout_log) if record else "",
        "stderr_log": str(record.stderr_log) if record else "",
        "command": ps_join(command),
        "render_count": len(list((out_dir / "renders").glob("*.png"))) if (out_dir / "renders").exists() else 0,
        "gt_count": len(list((out_dir / "gt").glob("*.png"))) if (out_dir / "gt").exists() else 0,
    }
    write_render_mapping(scene, out_dir)
    record_run_metadata(row)
    return row


def write_run_commands() -> None:
    lines = ["# RUN_COMMANDS", "", "## Pilot Commands", ""]
    for role, scene in SCENES.items():
        model = gsw_model_path("pilot", role, f"P-{role}", scene)
        trace = REPORT_DIR / "pilot_traces" / f"P-{role}.csv"
        audit = REPORT_DIR / "pilot_traces" / f"P-{role}_checkpoint.csv"
        lines.extend([f"### P-{role}", "", "```text", ps_join(gsw_train_command(scene, model, PILOT_ITERATION, [1000, 3000, 5000], True, trace, audit)), "```", ""])
    lines.extend(["## Screening Commands", ""])
    for case in screening_cases(include_third=True):
        lines.extend([f"### {case['label']}", "", "```text", ps_join(case["train_command"]), "```", ""])
    write_text(REPORT_DIR / "RUN_COMMANDS.md", "\n".join(lines))


def dry_run() -> None:
    ensure_dirs()
    write_runner_audit()
    write_run_commands()
    row = {
        "dry_run": True,
        "scenes": json.dumps(SCENES, sort_keys=True),
        "run_root": str(RUN_ROOT),
        "report_dir": str(REPORT_DIR),
        "gpu": json.dumps(gpu_info(), sort_keys=True),
    }
    write_csv(REPORT_DIR / "BENCHMARK_RUNNER_DRY_RUN.csv", [row])
    md = [
        "# BENCHMARK_RUNNER_DRY_RUN",
        "",
        "- Dry run completed without launching training.",
        f"- Run root: `{RUN_ROOT}`",
        f"- Report dir: `{REPORT_DIR}`",
        f"- Scenes: `{json.dumps(SCENES, sort_keys=True)}`",
    ]
    write_text(REPORT_DIR / "BENCHMARK_RUNNER_DRY_RUN.md", "\n".join(md) + "\n")


def read_trace_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stderr_contains_oom(stderr_log: object) -> bool:
    if not stderr_log:
        return False
    path = Path(str(stderr_log))
    if not path.exists():
        return False
    return "out of memory" in path.read_text(encoding="utf-8", errors="ignore").lower()


def pilot_summary_from_trace(role: str, scene: str, trace_path: Path, checkpoint_audit_path: Path, run_row: dict[str, object]) -> dict[str, object]:
    rows = read_trace_rows(trace_path)
    iter_rows = [row for row in rows if row.get("event") == "iteration"]
    by_iter = {int(float(row["iteration"])): row for row in iter_rows if row.get("iteration")}
    final = by_iter.get(PILOT_ITERATION, {})
    checkpoint_ok = False
    checkpoint_error = ""
    if checkpoint_audit_path.exists():
        with checkpoint_audit_path.open("r", encoding="utf-8", newline="") as handle:
            audit_rows = list(csv.DictReader(handle))
        if audit_rows:
            checkpoint_ok = str(audit_rows[0].get("checkpoint_loadable", "")).lower() == "true"
            checkpoint_error = audit_rows[0].get("error", "")
    peak_alloc = max([float(row.get("peak_allocated_mb") or 0) for row in iter_rows] or [0.0])
    peak_reserved = max([float(row.get("peak_reserved_mb") or 0) for row in iter_rows] or [0.0])
    total_gpu = float(final.get("gpu_total_memory_mb") or gpu_info().get("gpu_total_memory_mb") or 0)
    gaussian_counts = [(int(float(row["iteration"])), int(float(row.get("gaussian_count") or 0))) for row in iter_rows if row.get("iteration")]
    recent_memory = [
        (iteration, float(by_iter[iteration].get("peak_allocated_mb") or 0))
        for iteration in sorted(by_iter)
        if 4000 <= iteration <= 5000
    ]
    recent_times = [
        float(by_iter[iteration].get("iter_time_rolling_median_ms") or 0)
        for iteration in sorted(by_iter)
        if 3000 <= iteration <= 5000 and by_iter[iteration].get("iter_time_rolling_median_ms")
    ]
    recent_time_ms = statistics.median(recent_times) if recent_times else float(final.get("iter_time_rolling_median_ms") or 0)
    finite_loss = all(str(row.get("loss_finite", "True")).lower() == "true" for row in iter_rows)
    completed = bool(final) and bool(run_row.get("checkpoint_complete"))
    peak_ratio = peak_alloc / total_gpu if total_gpu else 1.0
    deltas = [recent_memory[idx + 1][1] - recent_memory[idx][1] for idx in range(len(recent_memory) - 1)]
    fast_growth = bool(deltas) and all(delta >= -1 for delta in deltas) and (recent_memory[-1][1] - recent_memory[0][1]) > total_gpu * 0.05
    gaussian_exploded = bool(gaussian_counts) and gaussian_counts[-1][1] > max(5_000_000, gaussian_counts[0][1] * 20)
    oom = int(run_row.get("returncode") or 0) != 0 and stderr_contains_oom(run_row.get("stderr_log", ""))
    stop = oom or peak_ratio >= 0.90 or fast_growth or gaussian_exploded or not checkpoint_ok or not finite_loss
    pass_gate = completed and not stop
    return {
        "role": role,
        "scene": scene,
        "completed_5000": completed,
        "returncode": run_row.get("returncode", ""),
        "oom": oom,
        "nan_or_inf": not finite_loss,
        "checkpoint_loadable": checkpoint_ok,
        "checkpoint_error": checkpoint_error,
        "peak_allocated_mb": round(peak_alloc, 3),
        "peak_reserved_mb": round(peak_reserved, 3),
        "gpu_total_memory_mb": total_gpu,
        "peak_allocated_ratio": round(peak_ratio, 4),
        "memory_fast_growth_4000_5000": fast_growth,
        "memory_samples_4000_5000": json.dumps(recent_memory),
        "gaussian_count_final": int(float(final.get("gaussian_count") or 0)) if final else "",
        "gaussian_exploded": gaussian_exploded,
        "recent_iter_time_median_ms": round(float(recent_time_ms), 3) if recent_time_ms else "",
        "estimated_remaining_25000_min": round((recent_time_ms / 1000.0) * 25000 / 60.0, 2) if recent_time_ms else "",
        "pass_gate": pass_gate,
        "stop_gate": stop,
        "trace_path": str(trace_path),
    }


def run_pilots() -> list[dict[str, object]]:
    ensure_dirs()
    summaries = []
    for role, scene in SCENES.items():
        label = f"P-{role}"
        model = gsw_model_path("pilot", role, label, scene)
        trace = REPORT_DIR / "pilot_traces" / f"{label}.csv"
        audit = REPORT_DIR / "pilot_traces" / f"{label}_checkpoint.csv"
        command = gsw_train_command(scene, model, PILOT_ITERATION, [1000, 3000, 5000], True, trace, audit)
        row = run_training(label, "gsw", scene, model, PILOT_ITERATION, command, IMPROVED_ROOT, trace, audit)
        summary = pilot_summary_from_trace(role, scene, trace, audit, row)
        summaries.append(summary)
        write_csv(REPORT_DIR / "POST_DENSIFICATION_PILOT.csv", summaries)
        write_pilot_md(summaries)
    update_budget_from_pilot(summaries)
    return summaries


def write_pilot_md(summaries: list[dict[str, object]]) -> None:
    md = [
        "# POST_DENSIFICATION_PILOT",
        "",
        "| role | scene | completed | pass | stop | peak allocated / total MB | ratio | final gaussians | recent iter ms | remaining 25k min |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        md.append(
            f"| {row['role']} | {row['scene']} | {row['completed_5000']} | {row['pass_gate']} | {row['stop_gate']} | "
            f"{row['peak_allocated_mb']}/{row['gpu_total_memory_mb']} | {row['peak_allocated_ratio']} | "
            f"{row['gaussian_count_final']} | {row['recent_iter_time_median_ms']} | {row['estimated_remaining_25000_min']} |"
        )
    md.extend(
        [
            "",
            "PASS requires completion, no OOM, finite loss, loadable checkpoint, peak allocated < 90% total GPU memory, no rapid monotonic 4000-5000 memory growth, and manageable Gaussian count.",
        ]
    )
    write_text(REPORT_DIR / "POST_DENSIFICATION_PILOT.md", "\n".join(md) + "\n")


def interval_speed(trace_path: Path, start_iter: int, end_iter: int) -> float:
    rows = [row for row in read_trace_rows(trace_path) if row.get("event") == "iteration"]
    by_iter = {int(float(row["iteration"])): row for row in rows if row.get("iteration")}
    left = by_iter.get(start_iter)
    right = by_iter.get(end_iter)
    if not left or not right:
        return 0.0
    wall_delta = float(right.get("elapsed_wall_sec") or 0) - float(left.get("elapsed_wall_sec") or 0)
    return (wall_delta / max(end_iter - start_iter, 1)) * 1000.0 if wall_delta > 0 else 0.0


def update_budget_from_pilot(summaries: list[dict[str, object]]) -> None:
    rows = {row["role"]: row for row in summaries}
    md = [
        "# UPDATED_LONG_RUN_BUDGET",
        "",
        "Budget is based on post-densification pilot measured iteration time, not sqrt(train-image-count). Official 3DGS estimates use the same per-scene pilot speed as a conservative scheduling placeholder until measured official 30k timings are available.",
        "",
        "| role | scene | 1000-3000 ms | 3000-5000 ms | central full 30k min | peak allocated ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    total_min = 0.0
    for role in ["H", "M"]:
        row = rows.get(role)
        if not row:
            continue
        trace = Path(str(row["trace_path"]))
        ms_1 = interval_speed(trace, 1000, 3000)
        ms_2 = interval_speed(trace, 3000, 5000)
        fallback = float(row.get("recent_iter_time_median_ms") or 0)
        central_ms = statistics.median([value for value in [ms_1, ms_2, fallback] if value > 0]) if any(v > 0 for v in [ms_1, ms_2, fallback]) else 0
        full = (central_ms / 1000.0) * ITERATION / 60.0 if central_ms else 0
        total_min += 4 * full
        md.append(f"| {role} | {row['scene']} | {ms_1:.3f} | {ms_2:.3f} | {full:.2f} | {row['peak_allocated_ratio']} |")
    md.extend(
        [
            "",
            f"- Optimistic estimate: `{total_min * 0.75 / 60.0:.2f}` GPU hours for 8 screening runs.",
            f"- Central estimate: `{total_min / 60.0:.2f}` GPU hours for 8 screening runs.",
            f"- Conservative estimate: `{total_min * 1.5 / 60.0:.2f}` GPU hours for 8 screening runs.",
            f"- Worst case with symmetric third runs for both scenes: `{total_min * 1.5 / 60.0:.2f}` additional central-equivalent GPU hours including O3/G3.",
            "",
            "This budget is for scheduling only and does not change the frozen experiment matrix.",
        ]
    )
    write_text(REPORT_DIR / "UPDATED_LONG_RUN_BUDGET.md", "\n".join(md) + "\n")


def pilot_passed() -> bool:
    path = REPORT_DIR / "POST_DENSIFICATION_PILOT.csv"
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows) == 2 and all(str(row.get("pass_gate", "")).lower() == "true" for row in rows)


def screening_cases(include_third: bool = False) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for role, scene in SCENES.items():
        reps = [1, 2, 3] if include_third else [1, 2]
        for idx in reps:
            label = f"{role}-O{idx}"
            model = official_model_path("screening", role, label, scene)
            cases.append(
                {
                    "role": role,
                    "scene": scene,
                    "label": label,
                    "method": "official",
                    "replicate": idx,
                    "model_path": model,
                    "train_command": official_train_command(scene, model, ITERATION),
                    "cwd": OFFICIAL_ROOT,
                }
            )
            label = f"{role}-G{idx}"
            model = gsw_model_path("screening", role, label, scene)
            cases.append(
                {
                    "role": role,
                    "scene": scene,
                    "label": label,
                    "method": "gsw",
                    "replicate": idx,
                    "model_path": model,
                    "train_command": gsw_train_command(scene, model, ITERATION, [ITERATION], False),
                    "cwd": IMPROVED_ROOT,
                }
            )
    return cases


def run_case(case: dict[str, object]) -> dict[str, object]:
    label = str(case["label"])
    method = str(case["method"])
    scene = str(case["scene"])
    model_path = Path(str(case["model_path"]))
    train_row = run_training(label, method, scene, model_path, ITERATION, list(case["train_command"]), Path(str(case["cwd"])))
    if int(train_row.get("returncode") or 0) != 0 or str(train_row.get("checkpoint_complete", "")).lower() != "true":
        raise RuntimeError(f"{label} training did not complete: {train_row}")
    render_row = run_render(label, method, scene, model_path, ITERATION)
    if int(render_row.get("returncode") or 0) != 0 or int(render_row.get("render_count") or 0) != expected_render_count(scene):
        raise RuntimeError(f"{label} rendering did not complete: {render_row}")
    return render_row


def run_screening(first_two_only: bool = True) -> None:
    if not pilot_passed():
        raise RuntimeError("Pilot PASS evidence is missing; refusing to start 30k screening.")
    cases = [case for case in screening_cases(include_third=False) if int(case["replicate"]) <= 2]
    for case in cases:
        run_case(case)
        evaluate_all()
        write_completion_eta()
    evaluate_all()
    write_third_run_decision()


def run_triggered_third() -> None:
    decision = read_third_decision()
    triggered_roles = [role for role, row in decision.items() if row.get("trigger_third_run") is True]
    for role in triggered_roles:
        for case in screening_cases(include_third=True):
            if case["role"] == role and int(case["replicate"]) == 3:
                run_case(case)
                evaluate_all()
                write_completion_eta()
    evaluate_all()
    write_third_run_decision()


def write_render_mapping(scene: str, out_dir: Path) -> None:
    render_dir = out_dir / "renders"
    if not render_dir.exists():
        return
    _, test_names = load_split(scene)
    render_files = sorted(path.name for path in render_dir.glob("*.png"))
    mapping_path = out_dir / "render_view_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["render_file", "image_name", "split_role"], lineterminator="\n")
        writer.writeheader()
        for idx, name in enumerate(render_files):
            writer.writerow({"render_file": name, "image_name": test_names[idx] if idx < len(test_names) else "", "split_role": "test"})


def decoded_rgb_sha256(path: Path) -> str:
    with Image.open(path) as img:
        return hashlib.sha256(np.asarray(img.convert("RGB")).tobytes()).hexdigest()


def tensor_from_image(path: Path, device: str):
    import torch

    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def psnr_value(pred, gt) -> float:
    import torch

    mse = ((pred - gt) ** 2).reshape(pred.shape[0], -1).mean(1)
    return float((20 * torch.log10(1.0 / torch.sqrt(mse))).mean().item())


def completed_render_cases() -> list[dict[str, object]]:
    rows = []
    for case in screening_cases(include_third=True):
        method = str(case["method"])
        scene = str(case["scene"])
        model = Path(str(case["model_path"]))
        if render_complete(method, scene, model, ITERATION):
            out_dir = method_dir(method, model, ITERATION)
            write_render_mapping(scene, out_dir)
            rows.append({**case, "method_dir": out_dir, "method_name": method_name(method)})
    return rows


def evaluate_all() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import torch
    from kornia.metrics import ssim as kornia_ssim
    import lpips

    cases = completed_render_cases()
    if not cases:
        return [], []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    summary_rows: list[dict[str, object]] = []
    per_view_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for case in cases:
            out_dir = Path(str(case["method_dir"]))
            scene = str(case["scene"])
            render_dir = out_dir / "renders"
            gt_dir = out_dir / "gt"
            mapping = {}
            with (out_dir / "render_view_mapping.csv").open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    mapping[row["render_file"]] = row["image_name"]
            render_files = sorted(path.name for path in render_dir.glob("*.png"))
            gt_files = sorted(path.name for path in gt_dir.glob("*.png"))
            if render_files != gt_files:
                raise ValueError(f"Render/GT mismatch for {out_dir}: {render_files} vs {gt_files}")
            psnrs: list[float] = []
            ssims: list[float] = []
            lpipss: list[float] = []
            black_count = 0
            finite_count = 0
            for name in render_files:
                render_path = render_dir / name
                gt_path = gt_dir / name
                pred = tensor_from_image(render_path, device)
                gt = tensor_from_image(gt_path, device)
                psnr = psnr_value(pred, gt)
                ssim = float(kornia_ssim(pred, gt, 3).mean().item())
                lp = float(lpips_model(pred, gt, normalize=True).item())
                arr = pred.detach().cpu().numpy()
                is_black = float(arr.max()) <= 1e-6 or float(arr.std()) <= 1e-8
                finite = all(math.isfinite(value) for value in [psnr, ssim, lp])
                black_count += int(is_black)
                finite_count += int(finite)
                psnrs.append(psnr)
                ssims.append(ssim)
                lpipss.append(lp)
                per_view_rows.append(
                    {
                        "role": case["role"],
                        "scene": scene,
                        "label": case["label"],
                        "method": case["method_name"],
                        "replicate": case["replicate"],
                        "method_dir": str(out_dir),
                        "render_file": name,
                        "image_name": mapping.get(name, ""),
                        "width": int(pred.shape[-1]),
                        "height": int(pred.shape[-2]),
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lp,
                        "render_min": float(arr.min()),
                        "render_max": float(arr.max()),
                        "render_mean": float(arr.mean()),
                        "render_std": float(arr.std()),
                        "gt_rgb_sha256": decoded_rgb_sha256(gt_path),
                        "render_rgb_sha256": decoded_rgb_sha256(render_path),
                        "is_black_render": is_black,
                        "finite_metrics": finite,
                        "full_image": True,
                        "lpips_net": "alex",
                        "lpips_normalize": True,
                    }
                )
            ckpt_dir = gsw_checkpoint_dir(Path(str(case["model_path"])), ITERATION) if case["method"] == "gsw" else official_checkpoint_dir(Path(str(case["model_path"])), ITERATION)
            summary_rows.append(
                {
                    "role": case["role"],
                    "scene": scene,
                    "label": case["label"],
                    "method": case["method_name"],
                    "replicate": case["replicate"],
                    "model_path": str(case["model_path"]),
                    "method_dir": str(out_dir),
                    "render_count": len(render_files),
                    "gt_count": len(gt_files),
                    "test_image_names": ";".join(mapping.get(name, "") for name in render_files),
                    "psnr": float(np.mean(psnrs)),
                    "ssim": float(np.mean(ssims)),
                    "lpips": float(np.mean(lpipss)),
                    "finite_metric_views": finite_count,
                    "black_render_views": black_count,
                    "all_metrics_finite": finite_count == len(render_files),
                    "no_black_images": black_count == 0,
                    "gaussian_count": ply_vertex_count(ckpt_dir / "point_cloud.ply") or "",
                    "checkpoint_size_bytes": dir_size_bytes(ckpt_dir),
                    "checkpoint_complete": checkpoint_complete(Path(str(case["model_path"])), str(case["method"]), ITERATION),
                    "full_image": True,
                    "lpips_net": "alex",
                    "lpips_normalize": True,
                }
            )
    write_csv(REPORT_DIR / "TWO_SCENE_SCREENING_RESULTS.csv", summary_rows)
    write_csv(REPORT_DIR / "TWO_SCENE_SCREENING_PER_VIEW.csv", per_view_rows)
    write_results_md(summary_rows)
    return summary_rows, per_view_rows


def write_results_md(rows: list[dict[str, object]]) -> None:
    md = [
        "# TWO_SCENE_SCREENING_RESULTS",
        "",
        "| role | label | method | PSNR | SSIM | LPIPS | views | black | gaussians |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: str(item["label"])):
        md.append(
            f"| {row['role']} | {row['label']} | {row['method']} | {float(row['psnr']):.6f} | "
            f"{float(row['ssim']):.6f} | {float(row['lpips']):.6f} | {row['render_count']} | "
            f"{row['black_render_views']} | {row['gaussian_count']} |"
        )
    write_text(REPORT_DIR / "TWO_SCENE_SCREENING_RESULTS.md", "\n".join(md) + "\n")


def intervals_overlap(a: list[float], b: list[float]) -> bool:
    return max(min(a), min(b)) <= min(max(a), max(b))


def read_result_rows() -> list[dict[str, str]]:
    path = REPORT_DIR / "TWO_SCENE_SCREENING_RESULTS.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_third_run_decision() -> None:
    rows = read_result_rows()
    decision_rows = []
    for role, scene in SCENES.items():
        official = [row for row in rows if row["role"] == role and row["method"] == "official_3dgs" and int(row["replicate"]) <= 2]
        gsw = [row for row in rows if row["role"] == role and row["method"] == "gsw_strict_intrinsic" and int(row["replicate"]) <= 2]
        values = {}
        trigger_reasons = []
        if len(official) < 2 or len(gsw) < 2:
            trigger = False
            trigger_reasons.append("insufficient_completed_first_two_runs")
        else:
            trigger = False
            for metric in ["psnr", "ssim", "lpips"]:
                off_vals = [float(row[metric]) for row in official]
                gsw_vals = [float(row[metric]) for row in gsw]
                overlap = intervals_overlap(off_vals, gsw_vals)
                values[f"{metric}_official_interval"] = f"[{min(off_vals):.6f},{max(off_vals):.6f}]"
                values[f"{metric}_gsw_interval"] = f"[{min(gsw_vals):.6f},{max(gsw_vals):.6f}]"
                values[f"{metric}_interval_overlap"] = overlap
                if overlap:
                    trigger = True
                    trigger_reasons.append(f"{metric}_interval_overlap")
            for method_label, method_rows in [("official", official), ("gsw", gsw)]:
                spread = abs(float(method_rows[0]["psnr"]) - float(method_rows[1]["psnr"]))
                values[f"{method_label}_psnr_spread"] = spread
                if spread > 0.3:
                    trigger = True
                    trigger_reasons.append(f"{method_label}_psnr_spread_gt_0.3")
        decision_rows.append(
            {
                "role": role,
                "scene": scene,
                "trigger_third_run": trigger,
                "trigger_reasons": ";".join(trigger_reasons),
                **values,
            }
        )
    write_csv(REPORT_DIR / "THIRD_RUN_TRIGGER_DECISION.csv", decision_rows)
    md = [
        "# THIRD_RUN_TRIGGER_DECISION",
        "",
        "Frozen trigger rules: interval overlap for PSNR/SSIM/LPIPS, either method PSNR spread > 0.3 dB, or run failure/checkpoint incompleteness.",
        "",
        "| role | scene | trigger | reasons |",
        "|---|---|---:|---|",
    ]
    for row in decision_rows:
        md.append(f"| {row['role']} | {row['scene']} | {row['trigger_third_run']} | {row['trigger_reasons']} |")
    md.append("")
    md.append("## Raw Values")
    md.append("")
    md.append("```json")
    md.append(json.dumps(decision_rows, indent=2))
    md.append("```")
    write_text(REPORT_DIR / "THIRD_RUN_TRIGGER_DECISION.md", "\n".join(md) + "\n")


def read_third_decision() -> dict[str, dict[str, object]]:
    path = REPORT_DIR / "THIRD_RUN_TRIGGER_DECISION.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        row["trigger_third_run"] = str(row.get("trigger_third_run", "")).lower() == "true"
        out[str(row["role"])] = row
    return out


def stats_for(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    values = [float(row[metric]) for row in rows]
    return {
        f"{metric}_mean": float(np.mean(values)),
        f"{metric}_sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{metric}_min": float(np.min(values)),
        f"{metric}_max": float(np.max(values)),
        f"{metric}_range": float(np.max(values) - np.min(values)),
        "n": len(values),
    }


def three_scene_comparison() -> None:
    two_rows = read_result_rows()
    combined: list[dict[str, object]] = []
    if TRACKMOBILE_RESULTS.exists():
        with TRACKMOBILE_RESULTS.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                combined.append(
                    {
                        "scene_group": "Trackmobile",
                        "scene": "self_Trackmobile_4650TM_Mobile_Railcar_Mover",
                        "label": row.get("label", ""),
                        "method": row.get("group", ""),
                        "psnr": row.get("psnr", ""),
                        "ssim": row.get("ssim", ""),
                        "lpips": row.get("lpips", ""),
                        "source": str(TRACKMOBILE_RESULTS),
                    }
                )
    for row in two_rows:
        combined.append({**row, "scene_group": row["role"], "source": str(REPORT_DIR / "TWO_SCENE_SCREENING_RESULTS.csv")})
    summary: list[dict[str, object]] = []
    for scene_group in sorted({str(row["scene_group"]) for row in combined}):
        for method in ["official_3dgs", "gsw_strict_intrinsic"]:
            group_rows = [row for row in combined if str(row["scene_group"]) == scene_group and str(row["method"]) == method]
            if not group_rows:
                continue
            out = {"scene_group": scene_group, "method": method}
            for metric in ["psnr", "ssim", "lpips"]:
                out.update(stats_for(group_rows, metric))
            out["n"] = len(group_rows)
            summary.append(out)
    write_csv(REPORT_DIR / "THREE_SCENE_BASELINE_COMPARISON.csv", summary)
    write_base_selection_md(summary)


def write_base_selection_md(summary: list[dict[str, object]]) -> None:
    by = {(row["scene_group"], row["method"]): row for row in summary}
    scene_decisions = []
    for scene_group in sorted({row["scene_group"] for row in summary}):
        off = by.get((scene_group, "official_3dgs"))
        gsw = by.get((scene_group, "gsw_strict_intrinsic"))
        if not off or not gsw:
            continue
        gsw_psnr_delta = float(gsw["psnr_mean"]) - float(off["psnr_mean"])
        gsw_ssim_delta = float(gsw["ssim_mean"]) - float(off["ssim_mean"])
        gsw_lpips_delta = float(gsw["lpips_mean"]) - float(off["lpips_mean"])
        scene_decisions.append(
            {
                "scene_group": scene_group,
                "gsw_psnr_delta": gsw_psnr_delta,
                "gsw_ssim_delta": gsw_ssim_delta,
                "gsw_lpips_delta": gsw_lpips_delta,
            }
        )
    new_scenes = [row for row in scene_decisions if row["scene_group"] in {"H", "M"}]
    official_wins_new = all(row["gsw_psnr_delta"] <= 0 and row["gsw_ssim_delta"] <= 0 for row in new_scenes) if new_scenes else False
    gsw_has_cross_scene_gain = any(row["gsw_psnr_delta"] > 0 and row["gsw_ssim_delta"] > 0 for row in new_scenes)
    if official_wins_new:
        selection = "B. official 3DGS"
        reason = "Official matches or exceeds GS-W on PSNR/SSIM in both new scenes; GS-W does not show reproducible cross-scene strict benefit."
    elif gsw_has_cross_scene_gain:
        selection = "A. GS-W"
        reason = "GS-W reproduces a PSNR/SSIM advantage in at least one new scene under strict_intrinsic protocol."
    else:
        selection = "C. still uncertain"
        reason = "The new-scene intervals or metric tradeoffs do not support a stable base choice."
    md = [
        "# THREE_SCENE_BASE_SELECTION",
        "",
        f"- Final choice: `{selection}`",
        f"- Reason: {reason}",
        "",
        "Delta is `GS-W mean - official mean`; positive PSNR/SSIM is better for GS-W, positive LPIPS is worse for GS-W.",
        "",
        "| scene | dPSNR | dSSIM | dLPIPS |",
        "|---|---:|---:|---:|",
    ]
    for row in scene_decisions:
        md.append(
            f"| {row['scene_group']} | {row['gsw_psnr_delta']:.6f} | {row['gsw_ssim_delta']:.6f} | {row['gsw_lpips_delta']:.6f} |"
        )
    md.extend(
        [
            "",
            "## Required discussion points",
            "",
            "- PSNR/SSIM/LPIPS are compared by repeated-run means, sample std, min, max and range in `THREE_SCENE_BASELINE_COMPARISON.csv`.",
            "- Per-view consistency is available in `TWO_SCENE_SCREENING_PER_VIEW.csv` and the accepted Trackmobile per-view CSV.",
            "- Training cost, peak memory, checkpoint size and Gaussian count are in `TWO_SCENE_SCREENING_RUN_METADATA.csv` and `TWO_SCENE_SCREENING_RESULTS.csv`.",
            "- Protocol: all new-scene GS-W renders use strict_intrinsic; no test RGB conditioning, no half-image metrics, no crop/resize/mask.",
        ]
    )
    write_text(REPORT_DIR / "THREE_SCENE_BASE_SELECTION.md", "\n".join(md) + "\n")


def find_render_for_image(out_dir: Path, image_name: str) -> Path | None:
    mapping_path = out_dir / "render_view_mapping.csv"
    if not mapping_path.exists():
        return None
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_name"] == image_name:
                return out_dir / "renders" / row["render_file"]
    return None


def find_gt_for_image(out_dir: Path, image_name: str) -> Path | None:
    mapping_path = out_dir / "render_view_mapping.csv"
    if not mapping_path.exists():
        return None
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_name"] == image_name:
                return out_dir / "gt" / row["render_file"]
    return None


def abs_error_image(render_path: Path, gt_path: Path) -> Image.Image:
    render = np.asarray(Image.open(render_path).convert("RGB"), dtype=np.int16)
    gt = np.asarray(Image.open(gt_path).convert("RGB"), dtype=np.int16)
    err = np.clip(np.abs(render - gt) * 4, 0, 255).astype(np.uint8)
    return Image.fromarray(err)


def thumbnail(path_or_image: Path | Image.Image, width: int = 240) -> Image.Image:
    image = Image.open(path_or_image).convert("RGB") if isinstance(path_or_image, Path) else path_or_image.convert("RGB")
    ratio = width / image.width
    out = image.resize((width, max(1, int(image.height * ratio))))
    return out


def create_contact_sheets() -> None:
    rows = read_result_rows()
    out_root = REPORT_DIR / "contact_sheets"
    out_root.mkdir(parents=True, exist_ok=True)
    for role, scene in SCENES.items():
        _, test_names = load_split(scene)
        if not test_names:
            continue
        selected = [test_names[0], test_names[len(test_names) // 2], test_names[-1]]
        selected = list(dict.fromkeys(selected))
        case_rows = [row for row in rows if row["role"] == role]
        row_items: list[tuple[str, list[Image.Image]]] = []
        first_out = Path(case_rows[0]["method_dir"]) if case_rows else None
        if not first_out:
            continue
        gt_images = []
        for name in selected:
            gt_path = find_gt_for_image(first_out, name)
            if gt_path and gt_path.exists():
                gt_images.append(thumbnail(gt_path))
        if gt_images:
            row_items.append(("GT", gt_images))
        for row in sorted(case_rows, key=lambda item: item["label"]):
            out_dir = Path(row["method_dir"])
            renders = []
            errors = []
            for name in selected:
                render_path = find_render_for_image(out_dir, name)
                gt_path = find_gt_for_image(out_dir, name)
                if render_path and gt_path and render_path.exists() and gt_path.exists():
                    renders.append(thumbnail(render_path))
                    errors.append(thumbnail(abs_error_image(render_path, gt_path)))
            if renders:
                row_items.append((f"{row['label']} render", renders))
                row_items.append((f"{row['label']} abs error x4", errors))
        if not row_items:
            continue
        tile_w = 240
        label_w = 180
        tile_h = max(img.height for _, imgs in row_items for img in imgs)
        sheet = Image.new("RGB", (label_w + tile_w * len(selected), tile_h * len(row_items)), "white")
        draw = ImageDraw.Draw(sheet)
        for r_idx, (label, imgs) in enumerate(row_items):
            y = r_idx * tile_h
            draw.text((8, y + 8), label, fill=(0, 0, 0))
            for c_idx, img in enumerate(imgs):
                sheet.paste(img, (label_w + c_idx * tile_w, y))
        path = out_root / f"{role}_{scene}_contact_sheet.jpg"
        sheet.save(path, quality=85)


def write_completion_eta() -> None:
    rows = load_existing_metadata()
    completed_train = [row for row in rows if row.get("stage") == "train" and row.get("iterations") == str(ITERATION) and row.get("duration_min")]
    durations = [float(row["duration_min"]) for row in completed_train if str(row.get("status")) in {"ran", "skipped_existing_complete"} and row.get("duration_min")]
    completed_labels = {row.get("label") for row in completed_train if str(row.get("checkpoint_complete", "")).lower() == "true"}
    planned = {case["label"] for case in screening_cases(include_third=False)}
    pending = sorted(planned - completed_labels)
    median = statistics.median(durations) if durations else 0
    eta_min = median * len(pending)
    md = [
        "# EXPERIMENT_PROGRESS",
        "",
        f"- Completed first-two 30k trainings: `{len(completed_labels & planned)}/{len(planned)}`",
        f"- Pending first-two labels: `{'; '.join(pending)}`",
        f"- Median measured 30k training duration: `{median:.2f}` min",
        f"- Estimated remaining first-two training time: `{eta_min:.2f}` min ({eta_min / 60.0:.2f} h)",
        f"- Updated: `{datetime.now().isoformat(timespec='seconds')}`",
    ]
    write_text(REPORT_DIR / "EXPERIMENT_PROGRESS.md", "\n".join(md) + "\n")


def digest_tensor(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def module_state_digest(module) -> str:
    pieces = [f"{key}:{digest_tensor(value)}" for key, value in module.state_dict().items()]
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()


def strict_gsw_audit_one(scene_name: str, model_path: Path, label: str) -> dict[str, object]:
    import argparse as _argparse
    import torch

    sys.path.insert(0, str(IMPROVED_ROOT))
    from arguments import ModelParams, PipelineParams, args_init
    from scene import GaussianModel, Scene
    from gaussian_renderer import render

    cfg = eval((model_path / "cfg_args").read_text(encoding="utf-8"), {"__builtins__": {}}, {"Namespace": _argparse.Namespace})
    cfg.model_path = str(model_path)
    cfg.source_path = str(scene_path(scene_name))
    cfg.split_mode = "frozen_manifest"
    cfg.split_file = str(manifest_path(scene_name))
    cfg.test_appearance_mode = "strict_intrinsic"
    cfg.data_perturb = getattr(cfg, "data_perturb", [])
    cfg = args_init.argument_init(cfg)
    model_parser = _argparse.ArgumentParser()
    pipeline_parser = _argparse.ArgumentParser()
    dataset = ModelParams(model_parser).extract(cfg)
    pipeline = PipelineParams(pipeline_parser).extract(cfg)
    gaussians = GaussianModel(dataset.sh_degree, cfg)
    scene_obj = Scene(dataset, gaussians, load_iteration=ITERATION, shuffle=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    original_forward = GaussianModel.forward
    calls = {"test_forward_calls": 0}

    def wrapped_forward(self, viewpoint_camera, *args, **kwargs):
        if getattr(viewpoint_camera, "split_role", "") == "test":
            calls["test_forward_calls"] += 1
        return original_forward(self, viewpoint_camera, *args, **kwargs)

    GaussianModel.forward = wrapped_forward
    before = module_state_digest(gaussians.map_generator)
    bn_before = module_state_digest(gaussians.map_generator)
    outputs = []
    try:
        gaussians.set_eval(True)
        with torch.no_grad():
            for view in scene_obj.getTestCameras():
                view.forbid_appearance_input = True
                image = render(view, gaussians, pipeline, background, appearance_mode="strict_intrinsic")["render"].detach().cpu()
                outputs.append(digest_tensor(image))
        gaussians.set_eval(False)
    finally:
        GaussianModel.forward = original_forward
    after = module_state_digest(gaussians.map_generator)
    return {
        "scene": scene_name,
        "label": label,
        "model_path": str(model_path),
        "test_view_count": len(outputs),
        "test_map_generator_forward_calls": calls["test_forward_calls"],
        "strict_intrinsic_no_test_rgb_forward": calls["test_forward_calls"] == 0,
        "map_generator_state_before": before,
        "map_generator_state_after": after,
        "map_generator_bn_immutable": bn_before == after,
        "render_output_digest": hashlib.sha256("\n".join(outputs).encode("utf-8")).hexdigest(),
    }


def strict_gsw_audits() -> None:
    rows = []
    for role, scene in SCENES.items():
        candidates = [row for row in read_result_rows() if row["role"] == role and row["method"] == "gsw_strict_intrinsic"]
        if not candidates:
            continue
        first = sorted(candidates, key=lambda row: row["label"])[0]
        rows.append(strict_gsw_audit_one(scene, Path(first["model_path"]), first["label"]))
    write_csv(REPORT_DIR / "GSW_STRICT_LEAKAGE_BN_AUDIT.csv", rows)
    md = [
        "# GSW Strict Leakage And BN Audit",
        "",
        "| scene | label | test views | test forward calls | no test RGB | BN immutable |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['scene']} | {row['label']} | {row['test_view_count']} | {row['test_map_generator_forward_calls']} | "
            f"{row['strict_intrinsic_no_test_rgb_forward']} | {row['map_generator_bn_immutable']} |"
        )
    write_text(REPORT_DIR / "GSW_STRICT_LEAKAGE_BN_AUDIT.md", "\n".join(md) + "\n")


def update_repo_status_files() -> None:
    status = git_output(IMPROVED_ROOT, ["status", "--short", "--branch"])
    head = git_output(IMPROVED_ROOT, ["rev-parse", "HEAD"])
    count = git_output(IMPROVED_ROOT, ["rev-list", "--left-right", "--count", "origin/main...main"])
    submodules = git_output(IMPROVED_ROOT, ["submodule", "status"])
    write_text(
        REPORT_DIR / "REPO_STATUS.txt",
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
                "## submodule status",
                submodules,
                "",
            ]
        ),
    )
    write_text(
        REPORT_DIR / "COMMIT_HISTORY.md",
        "\n".join(["# COMMIT_HISTORY", "", "```text", git_output(IMPROVED_ROOT, ["log", "--oneline", "--decorate", "-12"]), "```", ""]),
    )
    write_text(
        REPORT_DIR / "PUSH_STATUS.md",
        "\n".join(["# PUSH_STATUS", "", f"- HEAD: `{head}`", f"- origin/main...main: `{count}`", "", "```text", git_output(IMPROVED_ROOT, ["remote", "-v"]), "```", ""]),
    )


def package_results() -> Path:
    update_repo_status_files()
    create_contact_sheets()
    zip_path = REVIEW_ROOT / "codex_two_scene_strict_screening.zip"
    stage = REVIEW_ROOT / "codex_two_scene_strict_screening"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPORT_DIR, stage / "reports", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(IMPROVED_ROOT / "tools" / "two_scene_benchmark", stage / "tools" / "two_scene_benchmark", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches = stage / "patches"
    patches.mkdir(parents=True, exist_ok=True)
    (patches / "git_diff.patch").write_text(run_cmd(["git", "-C", str(IMPROVED_ROOT), "diff", "--", "tools/two_scene_benchmark", "reports/two_scene_benchmark"], timeout=120).stdout, encoding="utf-8")
    manifest = [
        "# Package Manifest",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Package: `{zip_path}`",
        f"- Run root: `{RUN_ROOT}`",
        "- Excludes datasets, checkpoints, complete renders, .git, build outputs, pycache and credentials.",
    ]
    write_text(stage / "PACKAGE_MANIFEST.md", "\n".join(manifest) + "\n")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=REVIEW_ROOT, base_dir=stage.name)
    hash_path = zip_path.with_suffix(".sha256.txt")
    hash_path.write_text(f"{sha256_file(zip_path)}  {zip_path.name}\n", encoding="ascii")
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable two-scene strict screening benchmark runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--screening", action="store_true")
    parser.add_argument("--third", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--package", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        dry_run()
        return 0
    if args.pilot:
        summaries = run_pilots()
        return 0 if all(row["pass_gate"] for row in summaries) and len(summaries) == 2 else 2
    if args.screening:
        run_screening()
        return 0
    if args.third:
        run_triggered_third()
        return 0
    if args.evaluate:
        evaluate_all()
        write_third_run_decision()
        return 0
    if args.finalize:
        evaluate_all()
        write_third_run_decision()
        strict_gsw_audits()
        three_scene_comparison()
        create_contact_sheets()
        update_repo_status_files()
        return 0
    if args.package:
        package_results()
        return 0
    parser.error("No action selected.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
