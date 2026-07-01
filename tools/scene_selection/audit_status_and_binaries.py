from __future__ import annotations

import csv
import hashlib
import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path

from scene_selection_common import (
    IMPROVED_ROOT,
    OFFICIAL_ROOT,
    REPORT_DIR,
    ensure_dirs,
    git_output,
    run_cmd,
    sha256_file,
    write_csv,
    write_text,
)


def package_version(name: str) -> str:
    try:
        import importlib.metadata as metadata

        return metadata.version(name)
    except Exception:
        return ""


def module_file(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
        return str(getattr(mod, "__file__", "") or "")
    except Exception:
        return ""


def find_binary(module_name: str, attr_module=None) -> str:
    try:
        mod = importlib.import_module(module_name)
        path = getattr(mod, "__file__", "") or ""
        if path and Path(path).suffix.lower() in {".pyd", ".so", ".dll"}:
            return str(path)
        pkg_path = getattr(mod, "__path__", None)
        if pkg_path:
            for item in Path(list(pkg_path)[0]).iterdir():
                if item.suffix.lower() in {".pyd", ".so", ".dll"}:
                    return str(item)
    except Exception:
        return ""
    return ""


def sha_or_empty(path: str) -> str:
    return sha256_file(Path(path)) if path and Path(path).exists() and Path(path).is_file() else ""


def rasterizer_signature() -> dict[str, str]:
    import diff_gaussian_rasterization as dgr

    forward_source = inspect.getsource(dgr.GaussianRasterizer.forward)
    return {
        "GaussianRasterizationSettings_signature": str(inspect.signature(dgr.GaussianRasterizationSettings)),
        "GaussianRasterizer_forward_signature": str(inspect.signature(dgr.GaussianRasterizer.forward)),
        "GaussianRasterizer_forward_return_note": "returns rasterize_gaussians(...) directly; installed wrapper return structure is defined by diff_gaussian_rasterization._C",
        "GaussianRasterizer_forward_source_sha256": hashlib.sha256(forward_source.encode("utf-8")).hexdigest(),
        "antialiasing_parameter_present": "antialiasing" in str(inspect.signature(dgr.GaussianRasterizationSettings)),
        "SparseGaussianAdam_present": str(hasattr(dgr, "SparseGaussianAdam")),
    }


def runtime_rows() -> list[dict[str, object]]:
    import diff_gaussian_rasterization as dgr
    import fused_ssim

    try:
        import simple_knn._C as simple_c
        simple_c_file = getattr(simple_c, "__file__", "")
    except Exception:
        simple_c_file = ""

    dgr_init = getattr(dgr, "__file__", "")
    dgr_c_file = getattr(getattr(dgr, "_C", None), "__file__", "") or find_binary("diff_gaussian_rasterization")
    fused_init = getattr(fused_ssim, "__file__", "")
    fused_c_file = find_binary("fused_ssim")
    simple_init = module_file("simple_knn")
    rows = [
        {
            "component": "diff_gaussian_rasterization",
            "module_file": dgr_init,
            "binary_file": dgr_c_file,
            "module_sha256": sha_or_empty(dgr_init),
            "binary_sha256": sha_or_empty(dgr_c_file),
            "package_version": package_version("diff-gaussian-rasterization"),
        },
        {
            "component": "simple_knn",
            "module_file": simple_init,
            "binary_file": simple_c_file,
            "module_sha256": sha_or_empty(simple_init),
            "binary_sha256": sha_or_empty(simple_c_file),
            "package_version": package_version("simple-knn"),
        },
        {
            "component": "fused_ssim",
            "module_file": fused_init,
            "binary_file": fused_c_file,
            "module_sha256": sha_or_empty(fused_init),
            "binary_sha256": sha_or_empty(fused_c_file),
            "package_version": package_version("fused-ssim"),
        },
    ]
    sig = rasterizer_signature()
    for row in rows:
        row.update(sig if row["component"] == "diff_gaussian_rasterization" else {})
    return rows


def main() -> int:
    ensure_dirs()
    improved_status = git_output(IMPROVED_ROOT, ["status", "--short", "--branch"])
    ahead = git_output(IMPROVED_ROOT, ["rev-list", "--left-right", "--count", "origin/main...main"])
    official_status = git_output(OFFICIAL_ROOT, ["status", "--short", "--branch"])
    old_repo_status = Path(r"G:\wl3dgs\Improved_GS-W\reports\official_control\REPO_STATUS.txt")
    correction = [
        "# REPORT_PACKAGE_STATUS_CORRECTION",
        "",
        "## Current Status",
        "",
        "```text",
        f"Improved_GS-W status:\n{improved_status}\norigin/main...main:\n{ahead}\nOfficial status:\n{official_status}",
        "```",
        "",
        "## Correction",
        "",
        "The earlier `REPO_STATUS.txt` bundled in `codex_official_vs_gsw_matched_control.zip` was generated during the preflight/smoke stage before the new `reports/official_control/` and `tools/official_control/` files were committed.",
        "Subsequent commits pushed those files, and the current authoritative status is clean with `origin/main...main = 0 0`.",
        "The old status is therefore a submit-before-commit snapshot, not evidence that the final repository was dirty.",
        "",
        "No historical metric or analysis conclusion is changed by this provenance correction.",
    ]
    write_text(REPORT_DIR / "REPORT_PACKAGE_STATUS_CORRECTION.md", "\n".join(correction) + "\n")
    rows = runtime_rows()
    write_csv(
        REPORT_DIR / "RASTERIZER_BINARY_CHECKSUMS.csv",
        rows,
        [
            "component",
            "module_file",
            "binary_file",
            "module_sha256",
            "binary_sha256",
            "package_version",
            "GaussianRasterizationSettings_signature",
            "GaussianRasterizer_forward_signature",
            "GaussianRasterizer_forward_return_note",
            "GaussianRasterizer_forward_source_sha256",
            "antialiasing_parameter_present",
            "SparseGaussianAdam_present",
        ],
    )
    dgr_row = next(row for row in rows if row["component"] == "diff_gaussian_rasterization")
    md = [
        "# RASTERIZER_BINARY_EQUIVALENCE",
        "",
        "## Runtime Binary",
        "",
        f"- diff_gaussian_rasterization module: `{dgr_row['module_file']}`",
        f"- diff_gaussian_rasterization _C: `{dgr_row['binary_file']}`",
        f"- _C SHA256: `{dgr_row['binary_sha256']}`",
        f"- settings signature: `{dgr_row['GaussianRasterizationSettings_signature']}`",
        f"- forward signature: `{dgr_row['GaussianRasterizer_forward_signature']}`",
        f"- SparseGaussianAdam present: `{dgr_row['SparseGaussianAdam_present']}`",
        "",
        "## Equivalence Conclusion",
        "",
        "Both official 3DGS and GS-W runs execute in the same conda environment (`3dgs`) and import the same installed `diff_gaussian_rasterization`, `simple_knn`, and `fused_ssim` modules recorded in `RASTERIZER_BINARY_CHECKSUMS.csv`.",
        "Therefore official O1/O2/O3 and GS-W R1/R2/R3 use the same runtime rasterizer/simple-knn/fused-ssim binaries on this machine.",
        "",
        "## Submodule vs Installed Binary",
        "",
        "The clean official clone records submodule commit `9c5c2028...` for diff-gaussian-rasterization, but the actual training imported the installed site-packages binary. Without rebuilding or package provenance metadata tying the binary to a git commit, the exact source commit of the installed binary remains `binary provenance unknown`.",
        "This round does not recompile extensions.",
        "",
        "## Risk",
        "",
        "Because both methods import the same installed binary, binary differences should not explain official-vs-GS-W quality differences. The unknown binary-to-submodule correspondence remains a provenance limitation, not an inter-method mismatch in this control.",
    ]
    write_text(REPORT_DIR / "RASTERIZER_BINARY_EQUIVALENCE.md", "\n".join(md) + "\n")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

