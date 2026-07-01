from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "scene_selection"))

from scene_selection_common import IMPROVED_ROOT, OFFICIAL_ROOT, REPORT_DIR, REVIEW_ROOT, git_output, run_cmd, write_text  # noqa: E402


PACKAGE_NAME = "codex_two_scene_selection_preflight"
PACKAGE_DIR = REVIEW_ROOT / PACKAGE_NAME
ZIP_PATH = REVIEW_ROOT / f"{PACKAGE_NAME}.zip"


REPORT_FILES = [
    "REPORT_PACKAGE_STATUS_CORRECTION.md",
    "RASTERIZER_BINARY_EQUIVALENCE.md",
    "RASTERIZER_BINARY_CHECKSUMS.csv",
    "SCENE_CANDIDATE_INVENTORY.csv",
    "TRAIN_ONLY_LIGHTING_DIFFICULTY.csv",
    "TRAIN_ONLY_LIGHTING_DIFFICULTY.md",
    "SELECTED_SCENES.md",
    "SELECTED_SCENES.json",
    "TWO_SCENE_SPLIT_PREFLIGHT_SUMMARY.csv",
    "H_SPLIT_PREFLIGHT.md",
    "H_SPLIT_PREFLIGHT.csv",
    "M_SPLIT_PREFLIGHT.md",
    "M_SPLIT_PREFLIGHT.csv",
    "TWO_SCENE_SMOKE_RESULTS.csv",
    "TWO_SCENE_SMOKE_PER_VIEW.csv",
    "TWO_SCENE_SMOKE_AUDIT.md",
    "TWO_SCENE_GT_CHECKSUMS.csv",
    "LONG_RUN_BUDGET.md",
    "LONG_RUN_BUDGET.csv",
    "NEXT_EXPERIMENT_DECISION.md",
]


SCRIPT_FILES = [
    "tools/scene_selection/scene_selection_common.py",
    "tools/scene_selection/audit_status_and_binaries.py",
    "tools/scene_selection/inventory_and_difficulty.py",
    "tools/two_scene_preflight/run_two_scene_preflight.py",
    "tools/two_scene_preflight/package_two_scene_preflight.py",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_logs() -> None:
    logs_src = REPORT_DIR / "logs"
    logs_dst = PACKAGE_DIR / "logs"
    logs_dst.mkdir(parents=True, exist_ok=True)
    if not logs_src.exists():
        return
    for path in sorted(logs_src.glob("*.log")):
        if path.stat().st_size <= 2_000_000:
            copy_file(path, logs_dst / path.name)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            excerpt = text[:200_000] + "\n\n[... truncated for review package ...]\n\n" + text[-200_000:]
            write_text(logs_dst / f"{path.name}.excerpt.txt", excerpt)


def status_files(push_status: str = "") -> dict[str, str]:
    head = git_output(IMPROVED_ROOT, ["rev-parse", "HEAD"])
    branch = git_output(IMPROVED_ROOT, ["branch", "--show-current"])
    remote = git_output(IMPROVED_ROOT, ["remote", "get-url", "origin"])
    ahead_behind = git_output(IMPROVED_ROOT, ["rev-list", "--left-right", "--count", "origin/main...main"])
    status = [
        "# Improved_GS-W",
        "",
        "```text",
        git_output(IMPROVED_ROOT, ["status", "--short", "--branch"]),
        "```",
        "",
        "```text",
        "origin/main...main:",
        git_output(IMPROVED_ROOT, ["rev-list", "--left-right", "--count", "origin/main...main"]),
        "```",
        "",
        "# Official_3DGS_Strict_Control",
        "",
        "```text",
        git_output(OFFICIAL_ROOT, ["status", "--short", "--branch"]),
        git_output(OFFICIAL_ROOT, ["rev-parse", "HEAD"]),
        "```",
    ]
    history = git_output(IMPROVED_ROOT, ["log", "--oneline", "-12"])
    push_text = push_status or "\n".join(
        [
            "# PUSH_STATUS",
            "",
            f"- Remote: `{remote}`",
            f"- Branch: `{branch}`",
            f"- HEAD: `{head}`",
            f"- origin/main...main: `{ahead_behind}`",
            "",
            "Conclusion: pushed and synchronized if `origin/main...main` is `0\t0` and `git status --short --branch` reports `main...origin/main` without ahead/behind markers.",
            "",
        ]
    )
    diff = run_cmd(["git", "-C", str(IMPROVED_ROOT), "diff", "HEAD", "--"], timeout=120)
    diff_cached = run_cmd(["git", "-C", str(IMPROVED_ROOT), "diff", "--cached", "--"], timeout=120)
    patch_text = ""
    if diff.stdout:
        patch_text += diff.stdout
    if diff_cached.stdout:
        patch_text += "\n\n# Cached diff\n\n" + diff_cached.stdout
    if not patch_text:
        patch_text = "# No uncommitted diff relative to HEAD.\n"
    return {
        "REPO_STATUS.txt": "\n".join(status) + "\n",
        "COMMIT_HISTORY.md": "# COMMIT_HISTORY\n\n```text\n" + history + "\n```\n",
        "PUSH_STATUS.md": push_text,
        "git_diff.patch": patch_text,
    }


def stage_package(meta_files: dict[str, str]) -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    for name in REPORT_FILES:
        copy_file(REPORT_DIR / name, PACKAGE_DIR / name)
    for name, text in meta_files.items():
        write_text(PACKAGE_DIR / name, text)
    manifests_src = REPORT_DIR / "generated_manifests"
    if manifests_src.exists():
        for path in sorted(manifests_src.glob("*.json")):
            copy_file(path, PACKAGE_DIR / "generated_manifests" / path.name)
    for rel in SCRIPT_FILES:
        copy_file(IMPROVED_ROOT / rel, PACKAGE_DIR / "scripts" / rel)
    copy_logs()


def zip_package() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if not path.is_file():
                continue
            zf.write(path, path.relative_to(PACKAGE_DIR.parent))


def main() -> int:
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    meta_files = status_files()
    stage_package(meta_files)
    zip_package()
    print(str(ZIP_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
