# All-View Leakage Test

The leakage test was run on both frozen held-out views, `0001.jpg` and `0009.jpg`, using three perturbations of the held-out RGB:

- zero image
- fixed random noise
- channel swap

Results:

- `strict_intrinsic`: all max and mean absolute differences were `0.0`
- `strict_nearest_train`: all max and mean absolute differences were `0.0`
- `legacy_target_rgb`: both held-out views changed under at least one perturbation

30k checkpoint CSV and JSON evidence are stored in:

- `G:\WL3DGS\gpt_review_packages\ALL_VIEW_LEAKAGE_RESULTS_30K.csv`
- `G:\WL3DGS\gpt_review_packages\all_view_leakage_test_trackmobile_30k.json`

The strict modes therefore pass the held-out RGB leakage guard for all frozen test views.
