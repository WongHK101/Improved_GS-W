# Checkpoint Comparison

- Historical Gaussian count: `133906`.
- Current clean Gaussian count: `140945`.
- Difference: `7039` Gaussians.
- Historical point cloud size: `33210228` bytes.
- Current point cloud size: `34955900` bytes.
- Map-generator checkpoint sizes equal: `True`.
- Color-net checkpoint sizes equal: `True`.
- Other-attributes sizes equal: `False`.
- Historical map-generator keys/params: `215` / `15797456`.
- Current map-generator keys/params: `215` / `15797456`.
- Historical color-net keys/params: `18` / `54195`.
- Current color-net keys/params: `18` / `54195`.
- Historical other-attributes keys: `box_coord,non`.
- Current other-attributes keys: `box_coord,non`.
- Feature dimensions: `48`; map_num: `3`; map generator: `unet/resnet18`; color net: `naive`.

The network architecture file sizes match, but the learned weights differ and the final Gaussian count differs. That means the two completed 30k optimizations are not byte-equivalent even though the high-level GS-W configuration is largely matched.
