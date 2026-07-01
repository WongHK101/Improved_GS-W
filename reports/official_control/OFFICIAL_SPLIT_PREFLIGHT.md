# OFFICIAL_SPLIT_PREFLIGHT

- Source path: `G:\wl3dgs\3dgs_undistorted\max1600\self_Trackmobile_4650TM_Mobile_Railcar_Mover`
- Official reader: `G:\wl3dgs\Official_3DGS_Strict_Control\scene\dataset_readers.py`
- Split manifest: `G:\WL3DGS\Improved_GS-W\splits\TRACKMOBILE_SPLIT.json`
- Manifest SHA256: `c65cc61677ffb46cd329d57fc72d30ed40d42e0f9a47d75c5a0931950c9f2cd7`
- Expected manifest SHA256: `c65cc61677ffb46cd329d57fc72d30ed40d42e0f9a47d75c5a0931950c9f2cd7`
- Manifest SHA matches expected: `True`
- Train count: `13`
- Test count: `2`
- Train matches frozen manifest: `True`
- Test matches frozen manifest: `True`
- Camera/intrinsics/extrinsics checksum: `9b35948394546a359d66ae4ed0c6607557c7bbc39350e2e502b6895c2e531ddb`
- OK to train: `True`

## Train Images

- `0002.jpg`
- `0003.jpg`
- `0004.jpg`
- `0005.jpg`
- `0006.jpg`
- `0007.jpg`
- `0008.jpg`
- `0010.jpg`
- `0011.jpg`
- `0012.jpg`
- `0013.jpg`
- `0014.jpg`
- `0015.jpg`

## Test Images

- `0001.jpg`
- `0009.jpg`

## Notes

- The official reader was imported from the clean official 3DGS worktree.
- `train_test_exp=False` was passed into `readColmapSceneInfo`, so test cameras are excluded from the train camera list.
- The manifest SHA is the frozen split JSON file hash; list hashes are also recorded in the JSON summary.
