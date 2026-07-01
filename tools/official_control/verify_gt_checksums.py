from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from official_control_common import GSW_RUN_ROOT, REPORT_DIR, SCENE_NAME, write_csv, write_text


def rgb_sha256(path: Path) -> str:
    return hashlib.sha256(np.asarray(Image.open(path).convert("RGB")).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare official render GT pixels against GS-W evaluation GT")
    parser.add_argument("--per-view-csv", type=Path, default=REPORT_DIR / "OFFICIAL_SMOKE_UNIFIED_PER_VIEW.csv")
    parser.add_argument("--output-csv", type=Path, default=REPORT_DIR / "OFFICIAL_SMOKE_GT_CHECKSUM.csv")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.per_view_csv.open("r", encoding="utf-8", newline="")))
    gsw_gt_dir = GSW_RUN_ROOT / "R1" / SCENE_NAME / "test" / "ours_30000_strict_intrinsic" / "gt"
    out = []
    for row in rows:
        gsw_gt = gsw_gt_dir / row["render_file"]
        gsw_hash = rgb_sha256(gsw_gt)
        out.append(
            {
                "label": row["label"],
                "render_file": row["render_file"],
                "image_name": row["image_name"],
                "official_gt_sha256": row["gt_rgb_sha256"],
                "gsw_gt_sha256": gsw_hash,
                "match": row["gt_rgb_sha256"] == gsw_hash,
            }
        )
    write_csv(
        args.output_csv,
        out,
        ["label", "render_file", "image_name", "official_gt_sha256", "gsw_gt_sha256", "match"],
    )
    all_match = all(bool(row["match"]) for row in out)
    lines = [
        "# OFFICIAL_SMOKE_GT_CHECKSUM",
        "",
        f"- Compared against: `{gsw_gt_dir}`",
        f"- All GT decoded RGB checksums match: `{all_match}`",
        "",
        "| label | render file | image name | match |",
        "|---|---|---|---:|",
    ]
    for row in out:
        lines.append(f"| {row['label']} | {row['render_file']} | {row['image_name']} | {row['match']} |")
    write_text(args.output_csv.with_suffix(".md"), "\n".join(lines) + "\n")
    print(f"all_match={all_match}")
    return 0 if all_match else 2


if __name__ == "__main__":
    raise SystemExit(main())

