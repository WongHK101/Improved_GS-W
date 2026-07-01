from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from official_control_common import (
    IMPROVED_ROOT,
    OFFICIAL_ROOT,
    REPORT_DIR,
    REVIEW_ROOT,
    copy_small_file,
    ensure_dirs,
    git_output,
    write_text,
)


PACKAGE_NAME = "codex_official_vs_gsw_matched_control"


def add_if_exists(zipf: zipfile.ZipFile, path: Path, arcname: str | None = None, max_bytes: int = 3_000_000) -> None:
    if not path.exists() or not path.is_file():
        return
    if path.stat().st_size > max_bytes:
        return
    zipf.write(path, arcname or str(path.relative_to(REPORT_DIR)))


def main() -> int:
    ensure_dirs()
    package_dir = REVIEW_ROOT / PACKAGE_NAME
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    # Refresh status files immediately before packaging.
    write_text(
        REPORT_DIR / "COMMIT_HISTORY.md",
        "# COMMIT_HISTORY\n\n```text\n"
        + git_output(IMPROVED_ROOT, ["log", "--oneline", "-12"])
        + "\n```\n",
    )
    write_text(
        REPORT_DIR / "PUSH_STATUS.md",
        "# PUSH_STATUS\n\n```text\n"
        + git_output(IMPROVED_ROOT, ["rev-list", "--left-right", "--count", "origin/main...main"])
        + "\n```\n",
    )
    write_text(
        REPORT_DIR / "git_diff.patch",
        git_output(IMPROVED_ROOT, ["diff", "--", "tools/official_control", "reports/official_control"]) + "\n",
    )

    for path in REPORT_DIR.glob("*"):
        if path.is_file():
            copy_small_file(path, package_dir / path.name, max_bytes=5_000_000)

    scripts_dir = package_dir / "scripts" / "tools" / "official_control"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for path in (IMPROVED_ROOT / "tools" / "official_control").glob("*.py"):
        copy_small_file(path, scripts_dir / path.name, max_bytes=1_000_000)

    logs_dir = package_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for path in (REPORT_DIR / "logs").glob("*.log"):
        copy_small_file(path, logs_dir / path.name, max_bytes=800_000)

    zip_path = REVIEW_ROOT / f"{PACKAGE_NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for path in package_dir.rglob("*"):
            if path.is_file():
                zipf.write(path, str(path.relative_to(REVIEW_ROOT)))
    print(zip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

