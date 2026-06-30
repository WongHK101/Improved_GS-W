# Eval State Test Results

| Mode | Legacy BN Changed | Fixed BN Changed | Fixed Order Max Diff | Fixed Repeat Max Diff | Pre/Post Max Diff |
|---|---:|---:|---:|---:|---:|
| legacy_target_rgb | True | False | 0.0 | 0.0 | 0.8011182546615601 |
| strict_intrinsic | False | False | 0.0 | 0.0 | 0.0 |
| strict_nearest_train | True | False | 0.0 | 0.0 | 0.6120296716690063 |

A passing fixed path requires `fixed_bn_changed=False`, order/repeat max pixel difference within tolerance, and `map_generator.training=False` inside eval.
