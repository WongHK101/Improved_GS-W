# LONG_RUN_BUDGET

Budget uses two references: measured Trackmobile 30k time scaled by sqrt(train image count), and a conservative linear extrapolation from the 300-iteration smoke. The recommended budget per case is the larger of the two.

| scene | method | train | Trackmobile-scaled 30k min | smoke300-linear 30k min | recommended 30k min |
|---|---|---:|---:|---:|---:|
| self_Steam_Locomotive | official_3dgs | 70 | 40.41 | 33.5 | 40.41 |
| self_Steam_Locomotive | gsw_strict_intrinsic | 70 | 211.36 | 117.2 | 211.36 |
| web_Terrestrial | official_3dgs | 166 | 62.23 | 33.5 | 62.23 |
| web_Terrestrial | gsw_strict_intrinsic | 166 | 325.48 | 134.0 | 325.48 |

## Scheme Cost

- Scheme A, complete 3-run x 2 methods x 2 scenes: `31.97` GPU hours.
- Scheme B, preregistered 2-run screening: `21.32` GPU hours before triggered third runs.
- Worst case for Scheme B, if every scene/method comparison triggers third runs: `31.97` GPU hours.

## Recommendation

Recommend Scheme B. It preserves a preregistered trigger rule while reducing expected cost, and any third-run trigger applies symmetrically to both methods for the affected scene.
