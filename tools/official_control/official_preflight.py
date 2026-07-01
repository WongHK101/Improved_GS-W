from __future__ import annotations

import csv
import importlib
import json
import math
import os
import sys
from pathlib import Path

from official_control_common import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TRAIN_IMAGES,
    OFFICIAL_ROOT,
    REPORT_DIR,
    SCENE_PATH,
    SPLIT_MANIFEST,
    conda_python_args,
    ensure_dirs,
    git_output,
    run_cmd,
    sha256_file,
    sha256_lines,
    write_csv,
    write_text,
)


def import_official_reader():
    sys.path.insert(0, str(OFFICIAL_ROOT))
    return importlib.import_module("scene.dataset_readers")


def camera_rows(scene_info) -> list[dict[str, object]]:
    rows = []
    for role, cameras in (("train", scene_info.train_cameras), ("test", scene_info.test_cameras)):
        for idx, cam in enumerate(cameras):
            rows.append(
                {
                    "role": role,
                    "order": idx,
                    "image_name": cam.image_name,
                    "width": cam.width,
                    "height": cam.height,
                    "uid": cam.uid,
                    "fov_x": f"{float(cam.FovX):.12g}",
                    "fov_y": f"{float(cam.FovY):.12g}",
                    "R": json.dumps([[float(v) for v in row] for row in cam.R], separators=(",", ":")),
                    "T": json.dumps([float(v) for v in cam.T], separators=(",", ":")),
                    "image_path": cam.image_path,
                    "is_test": bool(cam.is_test),
                }
            )
    return rows


def stable_camera_checksum(rows: list[dict[str, object]]) -> str:
    payload_rows = []
    for row in rows:
        payload_rows.append(
            {
                key: row[key]
                for key in ["role", "order", "image_name", "width", "height", "uid", "fov_x", "fov_y", "R", "T"]
            }
        )
    payload = json.dumps(payload_rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def split_preflight() -> dict[str, object]:
    reader = import_official_reader()
    scene_info = reader.readColmapSceneInfo(str(SCENE_PATH), "images", "", True, False)
    train = [cam.image_name for cam in scene_info.train_cameras]
    test = [cam.image_name for cam in scene_info.test_cameras]
    registered = sorted(train + test)
    rows = camera_rows(scene_info)
    csv_path = REPORT_DIR / "OFFICIAL_SPLIT_PREFLIGHT.csv"
    write_csv(
        csv_path,
        rows,
        [
            "role",
            "order",
            "image_name",
            "width",
            "height",
            "uid",
            "fov_x",
            "fov_y",
            "R",
            "T",
            "image_path",
            "is_test",
        ],
    )
    manifest_sha = sha256_file(SPLIT_MANIFEST)
    summary = {
        "source_path": str(SCENE_PATH),
        "official_root": str(OFFICIAL_ROOT),
        "manifest_path": str(SPLIT_MANIFEST),
        "manifest_sha256": manifest_sha,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_sha256_matches_expected": manifest_sha == EXPECTED_MANIFEST_SHA256,
        "registered_images": registered,
        "train_images": train,
        "test_images": test,
        "registered_count": len(registered),
        "train_count": len(train),
        "test_count": len(test),
        "registered_sha256_sorted": sha256_lines(registered),
        "train_sha256_ordered": sha256_lines(train),
        "test_sha256_ordered": sha256_lines(test),
        "expected_train_images": EXPECTED_TRAIN_IMAGES,
        "expected_test_images": EXPECTED_TEST_IMAGES,
        "train_matches_expected": train == EXPECTED_TRAIN_IMAGES,
        "test_matches_expected": test == EXPECTED_TEST_IMAGES,
        "camera_checksum": stable_camera_checksum(rows),
        "csv_path": str(csv_path),
    }
    ok = (
        summary["manifest_sha256_matches_expected"]
        and summary["train_matches_expected"]
        and summary["test_matches_expected"]
        and len(set(train).intersection(test)) == 0
    )
    summary["ok_to_train"] = ok

    md = [
        "# OFFICIAL_SPLIT_PREFLIGHT",
        "",
        f"- Source path: `{SCENE_PATH}`",
        f"- Official reader: `{OFFICIAL_ROOT / 'scene' / 'dataset_readers.py'}`",
        f"- Split manifest: `{SPLIT_MANIFEST}`",
        f"- Manifest SHA256: `{manifest_sha}`",
        f"- Expected manifest SHA256: `{EXPECTED_MANIFEST_SHA256}`",
        f"- Manifest SHA matches expected: `{summary['manifest_sha256_matches_expected']}`",
        f"- Train count: `{len(train)}`",
        f"- Test count: `{len(test)}`",
        f"- Train matches frozen manifest: `{summary['train_matches_expected']}`",
        f"- Test matches frozen manifest: `{summary['test_matches_expected']}`",
        f"- Camera/intrinsics/extrinsics checksum: `{summary['camera_checksum']}`",
        f"- OK to train: `{ok}`",
        "",
        "## Train Images",
        "",
        *[f"- `{name}`" for name in train],
        "",
        "## Test Images",
        "",
        *[f"- `{name}`" for name in test],
        "",
        "## Notes",
        "",
        "- The official reader was imported from the clean official 3DGS worktree.",
        "- `train_test_exp=False` was passed into `readColmapSceneInfo`, so test cameras are excluded from the train camera list.",
        "- The manifest SHA is the frozen split JSON file hash; list hashes are also recorded in the JSON summary.",
    ]
    write_text(REPORT_DIR / "OFFICIAL_SPLIT_PREFLIGHT.md", "\n".join(md) + "\n")
    write_text(REPORT_DIR / "OFFICIAL_SPLIT_PREFLIGHT.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def provenance_report() -> None:
    lines = [
        "# OFFICIAL_UPSTREAM_PROVENANCE",
        "",
        f"- Official worktree path: `{OFFICIAL_ROOT}`",
        "- Upstream URL: `https://github.com/graphdeco-inria/gaussian-splatting.git`",
        "- Local source note: network clone attempts failed earlier; this control worktree was created from the existing clean local clone `G:\\wl3dgs\\3dgs_original` and pinned to the commit below.",
        f"- HEAD: `{git_output(OFFICIAL_ROOT, ['rev-parse', 'HEAD'])}`",
        f"- Branch/status: `{git_output(OFFICIAL_ROOT, ['status', '--short', '--branch']).replace(chr(10), '; ')}`",
        "",
        "## Latest Commit",
        "",
        "```text",
        git_output(OFFICIAL_ROOT, ["log", "-1", "--pretty=fuller"]),
        "```",
        "",
        "## Remotes",
        "",
        "```text",
        git_output(OFFICIAL_ROOT, ["remote", "-v"]),
        "```",
        "",
        "## Submodules",
        "",
        "```text",
        git_output(OFFICIAL_ROOT, ["submodule", "status", "--recursive"]),
        "```",
        "",
        "## Submodule URLs",
        "",
        "```text",
        git_output(OFFICIAL_ROOT, ["config", "--get-regexp", r"^submodule\..*\.url$"]),
        "```",
        "",
        "## License Summary",
        "",
        "- Official 3DGS is distributed for non-commercial research/evaluation use under `LICENSE.md`.",
        "- This round does not modify official source files.",
        "- `SIBR_viewers` is uninitialized and treated as viewer-only; training-relevant Python/submodule code is recorded above.",
    ]
    write_text(REPORT_DIR / "OFFICIAL_UPSTREAM_PROVENANCE.md", "\n".join(lines) + "\n")


def environment_report() -> None:
    env_code = r"""
import importlib.util
import json
import sys
info = {}
try:
    import torch
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
except Exception as exc:
    info["torch_error"] = repr(exc)
try:
    import torchvision
    info["torchvision"] = torchvision.__version__
except Exception as exc:
    info["torchvision_error"] = repr(exc)
try:
    import kornia
    info["kornia"] = kornia.__version__
except Exception as exc:
    info["kornia_error"] = repr(exc)
try:
    import lpips
    info["lpips_package"] = getattr(lpips, "__file__", "")
except Exception as exc:
    info["lpips_error"] = repr(exc)
try:
    import diff_gaussian_rasterization as dgr
    info["diff_gaussian_rasterization"] = getattr(dgr, "__file__", "")
    info["SparseGaussianAdam"] = hasattr(dgr, "SparseGaussianAdam")
except Exception as exc:
    info["diff_gaussian_rasterization_error"] = repr(exc)
try:
    import simple_knn
    info["simple_knn"] = getattr(simple_knn, "__file__", "")
except Exception as exc:
    info["simple_knn_error"] = repr(exc)
spec = importlib.util.find_spec("fused_ssim")
info["fused_ssim_spec"] = str(spec)
info["python"] = sys.version.replace("\n", " ")
print(json.dumps(info, indent=2))
"""
    result = run_cmd(conda_python_args("-c", [env_code]), timeout=120)
    env_json = result.stdout.strip()
    nvidia = run_cmd(["nvidia-smi"], timeout=60)
    nvcc = run_cmd(["nvcc", "--version"], timeout=60)
    cl = run_cmd(["cl"], timeout=60)
    lines = [
        "# OFFICIAL_ENVIRONMENT",
        "",
        "## Conda/Python Imports",
        "",
        "```json",
        env_json,
        "```",
        "",
        "## nvidia-smi",
        "",
        "```text",
        nvidia.stdout.strip() if nvidia.returncode == 0 else nvidia.stderr.strip(),
        "```",
        "",
        "## nvcc",
        "",
        "```text",
        nvcc.stdout.strip() if nvcc.returncode == 0 else nvcc.stderr.strip(),
        "```",
        "",
        "## MSVC cl",
        "",
        "```text",
        (cl.stdout + cl.stderr).strip(),
        "```",
        "",
        "## Fixed Protocol",
        "",
        "- `optimizer_type=default` because installed `diff_gaussian_rasterization` does not expose `SparseGaussianAdam`.",
        "- `antialiasing=False`, `depths=''`, `white_background=False`, `random_background=False`, `resolution=1`, `sh_degree=3`.",
        "- `train_test_exp` is not enabled. Current official source still creates an exposure optimizer, but `render.py` only applies trained exposure when `dataset.train_test_exp=True`.",
        "- Unified evaluation uses LPIPS AlexNet from the installed `lpips` package with `normalize=True` to match prior GS-W metrics.",
    ]
    write_text(REPORT_DIR / "OFFICIAL_ENVIRONMENT.md", "\n".join(lines) + "\n")
    write_text(REPORT_DIR / "OFFICIAL_ENVIRONMENT_RAW.json", env_json + "\n")


def repo_status_report() -> None:
    lines = [
        "# REPO_STATUS",
        "",
        "## Improved_GS-W",
        "",
        "```text",
        git_output(Path(r"G:\wl3dgs\Improved_GS-W"), ["status", "--short", "--branch"]),
        "```",
        "",
        "## Official 3DGS",
        "",
        "```text",
        git_output(OFFICIAL_ROOT, ["status", "--short", "--branch"]),
        "```",
        "",
        "## Official Submodules",
        "",
        "```text",
        git_output(OFFICIAL_ROOT, ["submodule", "status", "--recursive"]),
        "```",
    ]
    write_text(REPORT_DIR / "REPO_STATUS.txt", "\n".join(lines) + "\n")


def main() -> int:
    ensure_dirs()
    provenance_report()
    environment_report()
    summary = split_preflight()
    repo_status_report()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok_to_train"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
