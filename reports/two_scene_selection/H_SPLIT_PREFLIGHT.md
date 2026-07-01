# H_SPLIT_PREFLIGHT

- Scene: `self_Steam_Locomotive`
- Scene path: `G:\wl3dgs\3dgs_undistorted\max1600\self_Steam_Locomotive`
- Manifest: `G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\self_Steam_Locomotive_SPLIT.json`
- Manifest SHA256: `42b57444323f45313a66a1f328de68062f7d4dfa61efb52092e467c1a9ad8378`
- Train/test: `70/10`
- Status: `PASS`

## Checks

| check | value |
|---|---:|
| manifest_sha256 | `42b57444323f45313a66a1f328de68062f7d4dfa61efb52092e467c1a9ad8378` |
| train_test_disjoint | `True` |
| union_covers_registered | `True` |
| hold8_train_match | `True` |
| hold8_test_match | `True` |
| official_train_match | `True` |
| official_test_match | `True` |
| gsw_train_match | `True` |
| gsw_test_match | `True` |

## Protocol

- Split rule: sort registered COLMAP image names lexicographically; `index % 8 == 0` is test.
- Official reader path uses clean official 3DGS `--eval` LLFF hold-8 behavior.
- GS-W reader path uses `split_mode=frozen_manifest` with the exact manifest above.
