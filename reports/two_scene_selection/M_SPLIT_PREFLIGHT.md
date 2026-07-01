# M_SPLIT_PREFLIGHT

- Scene: `web_Terrestrial`
- Scene path: `G:\wl3dgs\3dgs_undistorted\max1600\web_Terrestrial`
- Manifest: `G:\wl3dgs\Improved_GS-W\reports\two_scene_selection\generated_manifests\web_Terrestrial_SPLIT.json`
- Manifest SHA256: `567742eebac1d1ede9e726ea254172668b5d63d7bc137f61f7792658d572b0c0`
- Train/test: `166/24`
- Status: `PASS`

## Checks

| check | value |
|---|---:|
| manifest_sha256 | `567742eebac1d1ede9e726ea254172668b5d63d7bc137f61f7792658d572b0c0` |
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
