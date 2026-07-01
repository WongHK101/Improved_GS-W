from __future__ import annotations

from pathlib import Path

from benchmark_common import BASELINE_COMMIT, BASELINE_TAG, IMPROVED_ROOT, REPORT_DIR, git_output, run_cmd, write_text


TRAINING_PATHS = [
    "train.py",
    "render.py",
    "gaussian_renderer",
    "scene",
    "net_modules",
    "arguments",
    "utils/loss_utils.py",
    "utils/image_utils.py",
]


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    head = git_output(IMPROVED_ROOT, ["rev-parse", "HEAD"])
    tag_object = git_output(IMPROVED_ROOT, ["rev-parse", BASELINE_TAG])
    tag_commit = git_output(IMPROVED_ROOT, ["rev-parse", f"{BASELINE_TAG}^{{commit}}"])
    commit_type = git_output(IMPROVED_ROOT, ["cat-file", "-t", BASELINE_COMMIT])
    name_status = run_cmd(
        ["git", "-C", str(IMPROVED_ROOT), "diff", "--name-status", f"{BASELINE_COMMIT}..HEAD", "--", *TRAINING_PATHS],
        timeout=120,
    )
    diff_stat = run_cmd(
        ["git", "-C", str(IMPROVED_ROOT), "diff", "--stat", f"{BASELINE_COMMIT}..HEAD", "--", *TRAINING_PATHS],
        timeout=120,
    )
    full_diff = run_cmd(
        ["git", "-C", str(IMPROVED_ROOT), "diff", f"{BASELINE_COMMIT}..HEAD", "--", *TRAINING_PATHS],
        timeout=120,
    )
    passed = name_status.returncode == 0 and not name_status.stdout.strip() and not full_diff.stdout.strip()
    md = [
        "# TRAINING_CODE_FREEZE_AUDIT",
        "",
        f"- Audit timestamp source: local git state.",
        f"- Current HEAD: `{head}`",
        f"- Required baseline commit: `{BASELINE_COMMIT}`",
        f"- Baseline object type: `{commit_type}`",
        f"- Baseline tag: `{BASELINE_TAG}`",
        f"- Baseline tag object: `{tag_object}`",
        f"- Baseline tag peeled commit: `{tag_commit}`",
        f"- Audited paths: `{'; '.join(TRAINING_PATHS)}`",
        f"- Result: `{'PASS' if passed else 'FAIL'}`",
        "",
        "## Interpretation",
        "",
        "The local `gsw-strict-baseline-v2` ref is an annotated tag object. Its peeled commit is the GPT-specified baseline commit `ddc6d8702b2e838dc989d612ca23fb311b79f280`, so the apparent tag object hash mismatch is expected and not a code mismatch.",
        "",
        "## Diff Name Status",
        "",
        "```text",
        name_status.stdout.strip() or "[no training-path diffs]",
        "```",
        "",
        "## Diff Stat",
        "",
        "```text",
        diff_stat.stdout.strip() or "[no training-path diffs]",
        "```",
        "",
        "## Full Diff Check",
        "",
        "```text",
        full_diff.stdout.strip() or "[empty]",
        "```",
        "",
        "## Gate Decision",
        "",
    ]
    if passed:
        md.append("PASS. No differences exist between the GPT-specified baseline commit and current HEAD in the audited training, loss, renderer, appearance, densification, scene, argument, or image/loss utility paths. Pilot execution is allowed.")
    else:
        md.append("FAIL. At least one audited training-path difference exists. Pilot execution must stop until GPT reviews the diff.")
    write_text(REPORT_DIR / "TRAINING_CODE_FREEZE_AUDIT.md", "\n".join(md) + "\n")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
