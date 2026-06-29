# RASTERIZER_API_AUDIT

Date: 2026-06-30

## Scope

This audit covers the compatibility change in commit `bf6ad21 Support current rasterizer API during rendering`. It verifies that the active Python environment exposes the newer rasterizer API and that the compatibility layer keeps the original GS-W renderer semantics by consuming only rendered color and radii.

## Active Rasterizer

Regression command:

```powershell
conda run -n 3dgs python tests/test_rasterizer_api_compat.py --source_path "G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover" --scene_name self_Trackmobile_4650TM_Mobile_Railcar_Mover --model_path "G:\wl3dgs\3dgs_runs\improved_gsw_tests\rasterizer_api" --resolution 4 --output-json "G:\WL3DGS\gpt_review_packages\rasterizer_api_trackmobile.json"
```

Observed module:

```text
D:\anaconda\envs\3dgs\lib\site-packages\diff_gaussian_rasterization\__init__.py
```

Observed `GaussianRasterizer.forward` signature:

```text
(self, means3D, means2D, opacities, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None)
```

Observed `GaussianRasterizationSettings` fields:

```text
image_height, image_width, tanfovx, tanfovy, bg, scale_modifier, viewmatrix, projmatrix, sh_degree, campos, prefiltered, debug, antialiasing
```

The installed rasterizer therefore requires the extra `antialiasing` setting compared with the pinned GS-W source tree.

## Return Semantics

The installed rasterizer returns a tuple of length 3:

1. `color`
2. `radii`
3. `invdepths`

Trackmobile smoke output:

- `color`: shape `[3, 225, 400]`, dtype `torch.float32`, device `cuda:0`, finite.
- `radii`: shape `[1920]`, dtype `torch.int32`, device `cuda:0`, finite.
- `invdepths`: shape `[1, 225, 400]`, dtype `torch.float32`, device `cuda:0`, finite.

`invdepths` was not `None` in this environment. Older rasterizer builds may return only `(color, radii)`.

## Compatibility Boundary

`gaussian_renderer.raster_settings_kwargs()` injects `antialiasing=False` only when the installed `GaussianRasterizationSettings` exposes that field. This preserves compatibility with both old and new rasterizer signatures.

`gaussian_renderer.rasterizer_color_and_radii()` consumes only the first two outputs from the rasterizer. This preserves original GS-W renderer behavior because the existing training and rendering code only expects:

- rendered RGB image;
- per-Gaussian radii for visibility and densification bookkeeping.

The new `invdepths` output is intentionally ignored in the current foundation stage. No depth loss, depth regularization, or depth-dependent algorithm behavior was added.

## Backward Check

The regression test computes `loss = color.mean()` and calls `loss.backward()`. It verifies finite gradients for:

- `xyz`
- `opacity`
- `scaling`
- `rotation`
- `features_intrinsic`
- screen-space points

All checked gradients were present and finite on Trackmobile. Full JSON evidence:

```text
G:\WL3DGS\gpt_review_packages\rasterizer_api_trackmobile.json
```

## Conclusion

The compatibility change does not change the algorithmic renderer output consumed by GS-W. It only adapts to a newer installed rasterizer API by:

1. passing an explicit default `antialiasing=False`;
2. ignoring the additional `invdepths` return value.

The regression test passed and backward propagation reached the expected Gaussian parameters with finite gradients.
