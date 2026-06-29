# Checkpoint Comparison

- Historical Gaussian count: `133906`.
- Current clean Gaussian count: `140945`.
- Difference: `7039` Gaussians.
- Historical point cloud size: `33210228` bytes.
- Current point cloud size: `34955900` bytes.
- Map-generator checkpoint sizes equal: `True`.
- Color-net checkpoint sizes equal: `True`.
- Other-attributes sizes equal: `False`.

The network architecture file sizes match, but the learned weights differ and the final Gaussian count differs. That means the two completed 30k optimizations are not byte-equivalent even though the high-level GS-W configuration is largely matched.
