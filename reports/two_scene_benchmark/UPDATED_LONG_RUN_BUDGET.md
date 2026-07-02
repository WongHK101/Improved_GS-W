# UPDATED_LONG_RUN_BUDGET

Budget is based on post-densification pilot measured iteration time, not sqrt(train-image-count). Official 3DGS estimates use the same per-scene pilot speed as a conservative scheduling placeholder until measured official 30k timings are available.

| role | scene | 1000-3000 ms | 3000-5000 ms | central full 30k min | peak allocated ratio |
|---|---|---:|---:|---:|---:|
| H | self_Steam_Locomotive | 195.885 | 193.736 | 96.87 | 0.3903 |
| M | web_Terrestrial | 230.548 | 237.294 | 115.27 | 0.5799 |

- Optimistic estimate: `10.61` GPU hours for 8 screening runs.
- Central estimate: `14.14` GPU hours for 8 screening runs.
- Conservative estimate: `21.21` GPU hours for 8 screening runs.
- Worst case with symmetric third runs for both scenes: `21.21` additional central-equivalent GPU hours including O3/G3.

This budget is for scheduling only and does not change the frozen experiment matrix.
