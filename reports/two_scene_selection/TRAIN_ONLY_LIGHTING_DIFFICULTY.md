# TRAIN_ONLY_LIGHTING_DIFFICULTY

All scores use train images only. No test RGB, test metrics, historical PSNR/SSIM/LPIPS, method deltas, or qualitative test outputs are used.

Fixed thresholds: saturation if any RGB channel >= 0.98; dark if luminance <= 0.02. Luminance is sRGB relative luminance `Y = 0.2126 R + 0.7152 G + 0.0722 B`.

| rank | scene | source | score | log-lum range | sat spread | dark spread | chroma spread | lowfreq spread |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | web_Trento_Duomo_images | network/public scenic | 0.920000 | 0.616379 | 0.217337 | 0.037440 | 0.047260 | 0.392980 |
| 2 | web_Baalshamin_images | network/public scenic | 0.720000 | 0.354727 | 0.135474 | 0.001206 | 0.071329 | 0.375686 |
| 3 | web_doss_images | network/public scenic | 0.700000 | 0.521392 | 0.094126 | 0.013866 | 0.017987 | 0.411412 |
| 4 | web_Terrestrial | network/public scenic | 0.620000 | 0.617069 | 0.013792 | 0.008108 | 0.033078 | 0.258824 |
| 5 | self_Steam_Locomotive | self-captured | 0.580000 | 0.622226 | 0.034036 | 0.000027 | 0.017669 | 0.301961 |
| 6 | self_double-action_press | self-captured | 0.480000 | 0.553246 | 0.001714 | 0.014209 | 0.022648 | 0.144784 |
| 7 | web_statue_images | network/public scenic | 0.480000 | 0.465089 | 0.000000 | 0.012939 | 0.031945 | 0.200784 |
| 8 | self_CLG899III_Wheel_Loader | self-captured | 0.380000 | 0.215971 | 0.018359 | 0.000000 | 0.031901 | 0.296314 |
| 9 | web_metopa_images | network/public scenic | 0.320000 | 0.210047 | 0.049549 | 0.000000 | 0.020599 | 0.159608 |
| 10 | self_3000t_Press | self-captured | 0.280000 | 0.220237 | 0.014670 | 0.000003 | 0.015203 | 0.182902 |
| 11 | web_cyprus_images | network/public scenic | 0.020000 | 0.121089 | 0.000000 | 0.000000 | 0.011500 | 0.076157 |
