# Historical vs Clean Code Diff

- Historical code path: `G:\wl3dgs\external_baselines\Gaussian-Wild`
- Historical commit: `fbe12be37cc0054296e2ef8631a6579b2a136ea7`
- Historical status:

```text
## main...origin/main
 M gaussian_renderer/__init__.py
 M render.py
 M scene/dataset_readers.py
?? __pycache__/
?? arguments/__pycache__/
?? gaussian_renderer/__pycache__/
?? net_modules/__pycache__/
?? scene/__pycache__/
?? utils/__pycache__/
```

- Clean code path: `G:\WL3DGS\Improved_GS-W`
- Clean branch/head: `main` / `643109114b4d3aa408f3df3e0185204958ef94b8`

## Historical dirty worktree patch

```text
warning: in the working copy of 'gaussian_renderer/__init__.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'render.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'scene/dataset_readers.py', LF will be replaced by CRLF the next time Git touches it
 gaussian_renderer/__init__.py |  5 +++--
 render.py                     | 16 +++++++++++-----
 scene/dataset_readers.py      |  6 +++---
 3 files changed, 17 insertions(+), 10 deletions(-)
```

The dirty historical patch is focused on:

- `scene/dataset_readers.py`: uses `cam_intrinsics[extr.camera_id]` and `uid = extr.id`. This is critical for legacy TSV split membership when a scene has a shared camera intrinsic.
- `gaussian_renderer/__init__.py`: passes `antialiasing=False` and accepts a 3-value rasterizer return. This is rasterizer API compatibility.
- `render.py`: protects the rendering-speed probe on small scenes. This affects post-training render utilities, not the training loss.

Full focused historical patch is saved as `HISTORICAL_WORKTREE.patch`.

## Whether historical 30k used these dirty files

Status: **unknown**. The historical run preserves `cfg_args`, logs and outputs, but no immutable source snapshot or dirty-worktree checksum embedded in the run output was found. The current historical repo dirty files are frozen in this audit as the best available local evidence, but the audit does not assume they are exactly the files used on 2026-06-21.

## Clean-vs-historical relevant file checksums

- Relevant source files compared: `18`.
- Files with different worktree content: `8`.

| File | Equal | Interpretation |
|---|---:|---|
| `train.py` | `False` | expected clean strict/data-layout adaptation or historical dirty compatibility change |
| `render.py` | `False` | expected clean strict/data-layout adaptation or historical dirty compatibility change |
| `metrics.py` | `True` | same as historical worktree |
| `metrics_half.py` | `True` | same as historical worktree |
| `gaussian_renderer/__init__.py` | `False` | expected clean strict/data-layout adaptation or historical dirty compatibility change |
| `scene/__init__.py` | `False` | changed; inspect checksum CSV |
| `scene/dataset_readers.py` | `False` | expected clean strict/data-layout adaptation or historical dirty compatibility change |
| `scene/gaussian_model.py` | `False` | changed; inspect checksum CSV |
| `scene/cameras.py` | `False` | changed; inspect checksum CSV |
| `utils/camera_utils.py` | `True` | same as historical worktree |
| `utils/image_utils.py` | `True` | same as historical worktree |
| `arguments/__init__.py` | `False` | expected clean strict/data-layout adaptation or historical dirty compatibility change |
| `arguments/args_init.py` | `True` | same as historical worktree |
| `net_modules/feature_maps_generators.py` | `True` | same as historical worktree |
| `net_modules/feature_maps_projection.py` | `True` | same as historical worktree |
| `net_modules/feature_maps_sample.py` | `True` | same as historical worktree |
| `net_modules/color_features_net.py` | `True` | same as historical worktree |
| `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` | `True` | same as historical worktree |

## Result-difference relevance

The most result-relevant code differences are split/data-loading and evaluation appearance handling. Training loss, optimizer groups, GS-W feature-map path, color-net path and densification logic remain close to upstream GS-W in the clean repo, but the clean repo contains strict appearance modes and frozen split plumbing that were not in the historical repo.
