# CANONICAL_SPLIT_HASH_AUDIT

Canonical split hash is computed from only `scene_name`, lexicographically sorted `train_images`, and lexicographically sorted `test_images`.
Serialization is UTF-8 JSON with `sort_keys=True` and `separators=(',', ':')`; no absolute paths, timestamps, metadata, or formatting-sensitive fields are included.

- Trackmobile canonical equivalence gate: `PASS`
- Trackmobile unique canonical hashes: `1`

## Trackmobile Raw vs Canonical

- `frozen_split_text` raw=`n/a` canonical=`878faaddbf02176a84d9be0e4a3f26923d33df132b40e97b404f7c91bcce72ce`
- `registry_manifest` raw=`f56a74221510d7ea9ac3c8f9a22a22b310e1801a65c4872eb7f4f2b253a76b62` canonical=`878faaddbf02176a84d9be0e4a3f26923d33df132b40e97b404f7c91bcce72ce`
- `legacy_repo_trackmobile_split` raw=`c65cc61677ffb46cd329d57fc72d30ed40d42e0f9a47d75c5a0931950c9f2cd7` canonical=`878faaddbf02176a84d9be0e4a3f26923d33df132b40e97b404f7c91bcce72ce`
- `legacy_review_trackmobile_split_test` raw=`9a82b9eb690ac890d04f5bebabe17500bdbf8a02ddf7b850f1cbe85d0bdf084f` canonical=`878faaddbf02176a84d9be0e4a3f26923d33df132b40e97b404f7c91bcce72ce`

## Scene Summary

| scene | sources | unique canonical hashes | status | canonical hash |
|---|---:|---:|---|---|
| web_doss_images | 2 | 1 | PASS | 8a02ef80c6e90c5e00d118fbdf47917ce34622f27b02f8fb82568ab9629c69a5 |
| web_Trento_Duomo_images | 2 | 1 | PASS | 5ea4f37649c404df0b3b3acbc2d30332558c9fbe651d5c0e3c6483d72c2ecd4c |
| web_Terrestrial | 2 | 1 | PASS | 32e2adb276902bfa454ae2372580a449ac48c7c085706b5f9968990dd2d6338b |
| self_double-action_press | 2 | 1 | PASS | 8278e645eea52ffa4745a64a94cc36a0f94e92c48e418f3b28374d7a0dc9699b |
| web_cyprus_images | 2 | 1 | PASS | caa25b2f57956c829fd7cae9aee372a8d8d1e67a7f056b9760963d821778cfbf |
| self_CLG899III_Wheel_Loader | 2 | 1 | PASS | d6205d94e63042d1654145390caa456f2a89b440702e3707b6a13a44284d11e9 |
| web_metopa_images | 2 | 1 | PASS | ace7df2370f4e769916abf8d203fae03ede0447e1cfc4140efb84d36e9b399f3 |
| web_statue_images | 2 | 1 | PASS | 9e67fbafce9e72695cc7c52e35560cbf91805cd604e95611b64aa3320acbff15 |
| self_Steam_Locomotive | 2 | 1 | PASS | 8dfeb16d0d7c799737717e6c90b770b4abca985117ab91f3a0410a49aa5cf509 |
| web_Baalshamin_images | 2 | 1 | PASS | 88aa6f5091753b0119c7a4b55adc22695e28240fd4a5d353a66d7090219b1195 |
| self_3000t_Press | 2 | 1 | PASS | bd92b1cdc3b5aa9ffe1d4a5c398869abddd0bf39265ee329375ed9559fc7d4e9 |
| self_Trackmobile_4650TM_Mobile_Railcar_Mover | 4 | 1 | PASS | 878faaddbf02176a84d9be0e4a3f26923d33df132b40e97b404f7c91bcce72ce |
