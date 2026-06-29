# Upstream Tree Audit

Generated: 2026-06-29 Asia/Shanghai

## Scope

This audit verifies that `G:\wl3dgs\Improved_GS-W` was initialized from the pinned GS-W upstream snapshot before any protocol or algorithm changes are added.

## Upstream Snapshot

- Upstream repository: https://github.com/EastbeanZhang/Gaussian-Wild.git
- Upstream branch checked: `main`
- Pinned upstream commit: `fbe12be37cc0054296e2ef8631a6579b2a136ea7`
- Fresh clone path used for audit: `G:\wl3dgs\_tmp_gsw_upstream_audit_20260629_1450`
- Current repository path: `G:\wl3dgs\Improved_GS-W`

## Submodule Metadata

The pinned upstream commit does not contain a root `.gitmodules` file. Running `git submodule status --recursive` in the fresh clone returns no entries.

Although the repository has a `submodules/` directory, at this commit these directories are normal tracked source trees, not Git submodule pointers:

```text
040000 tree bda9fef475163521c67891fef76c4040a4194896 submodules/diff-gaussian-rasterization
040000 tree d68a428ee685a6b6769fb4b4230ac30a143c7f7e submodules/simple-knn
```

Therefore there are no upstream submodule URLs or submodule commits to report for this pinned snapshot. The provenance anchor is the top-level upstream commit above plus the tracked tree hashes.

## File Content Comparison

I generated SHA256 manifests for the fresh upstream clone and the current `Improved_GS-W` working tree, excluding `.git`, `__pycache__`, and the intentional repository-local files listed below.

Excluded intentional local files:

- `.gitignore`
- `UPSTREAM.md`
- `UPSTREAM_TREE_AUDIT.md`

Result:

```text
upstream_files=1594
current_files=1594
content_diffs=0
```

This means the current file contents match the pinned upstream snapshot for all audited upstream files.

## Git-Tracked Tree Difference

The current Git history intentionally does not track upstream's historical build artifacts under:

- `submodules/simple-knn/dist/simple_knn-0.0.0-py3.7-linux-x86_64.egg`
- `submodules/simple-knn/simple_knn.egg-info/*`

These files still exist in the local working directory copied from upstream, but are ignored by the new repository `.gitignore` because GPT requested that build outputs and generated artifacts not be committed. This is a Git-tracked-tree difference, not a source-content difference in the audited working tree.

The source files under `submodules/diff-gaussian-rasterization` and `submodules/simple-knn` match the fresh upstream snapshot.

## License Notes

The pinned upstream snapshot does not include a top-level `LICENSE` file. The rasterizer directory contains the GraphDECO Gaussian-Splatting license, which permits non-commercial research and evaluation use. The top-level GS-W license status remains to be confirmed with upstream before public redistribution or artifact release.

## Conclusion

The source import is suitable as a clean GS-W baseline for subsequent protocol work, provided that we continue to document the omitted build artifacts and the unresolved top-level license status. No source mismatch was found that would require rebuilding the repository before data/split/strict-appearance adaptation.
