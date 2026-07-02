# EXTERNAL_BASELINE_BOUNDED_DIAGNOSTICS

No full 30k reruns were launched by this diagnostic. It audits existing adapters, logs, checkpoints and render samples.

## Luminance-GS

- Commit: `f9b4b86692a3fa676e90f42236c6237f443e7e23`.
- Local status: `M Luminance-GS/examples/datasets/colmap_mip360.py;  M Luminance-GS/examples/simple_trainer_ours.py;  M Luminance-GS/gsplat/cuda/_backend.py;  M Luminance-GS/gsplat/cuda/csrc/bindings.h;  M Luminance-GS/gsplat/cuda/csrc/projection.cu;  M Luminance-GS/gsplat/cuda/csrc/rasterization.cu; ?? Luminance-GS/examples/__pycache__/; ?? Luminance-GS/examples/datasets/__pycache__/; ?? Luminance-GS/gsplat/__pycache__/; ?? Luminance-GS/gsplat/cuda/__pycache__/; ?? Luminance-GS/gsplat/cuda/csrc/third_party/`.
- Scenes inspected: `12`.
- Rows with historical failures: `5`.
- Import smoke contains error: `True`.
- PyCOLMAP probe: `{"pycolmap_file": "D:\\anaconda\\envs\\3dgs\\lib\\site-packages\\pycolmap\\__init__.py", "has_SceneManager": false, "version": "4.0.4"}`.
- Adapter evidence: `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_ADAPTER.csv`.
- Run/log evidence: `LUMINANCE_GS_BOUNDED_DIAGNOSTIC_RUNS.csv`.

Interpretation: Luminance-GS remains unresolved for strict main comparison. Historical logs and adapter state should be treated as wrapper/config evidence, not as a fair complete method result.

## WildGaussians

- Commit: `66fa22ac74a6ffba024842dff29ded114d41f4d0`.
- Local status: `clean`.
- Scenes inspected: `12`.
- PSNR<10 rows: `12` / `13`.
- Import smoke passed: `True`.
- Reader smoke evidence: `    ], |     "color_space": "srgb" |   }, |   { |     "scene": "web_doss_images", |     "split": "test", |     "num_image_paths": 95, |     "num_cameras": 95, |     "first_image": "0001.jpg", |     "image_size_first": [ |       534, |       951 |     ], |     "near_far_first": [ |       0.009999999776482582, |       4.135848522186279 |     ], |     "color_space": "srgb" |   } | ]`.
- Adapter evidence: `WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_ADAPTER.csv`.
- Run/log/image-stat evidence: `WILDGAUSSIANS_BOUNDED_DIAGNOSTIC_RUNS.csv`.

Interpretation: current WildGaussians numeric rows should remain invalid/removed from the strict main table. The evidence supports an integration/protocol problem requiring wrapper-level diagnosis before any full rerun.

## Code Evidence

- `EXTERNAL_BASELINE_CODE_EVIDENCE.csv` records source locations for curve/test split handling, phototourism half-image evaluation and color/background conversion.
