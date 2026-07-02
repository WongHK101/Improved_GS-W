# THIRD_RUN_TRIGGER_DECISION

Frozen trigger rules: interval overlap for PSNR/SSIM/LPIPS, either method PSNR spread > 0.3 dB, or run failure/checkpoint incompleteness.

| role | scene | trigger | reasons |
|---|---|---:|---|
| H | self_Steam_Locomotive | False |  |
| M | web_Terrestrial | True | gsw_psnr_spread_gt_0.3 |

## Raw Values

```json
[
  {
    "role": "H",
    "scene": "self_Steam_Locomotive",
    "trigger_third_run": false,
    "trigger_reasons": "",
    "psnr_official_interval": "[20.651723,20.708455]",
    "psnr_gsw_interval": "[19.859955,19.947747]",
    "psnr_interval_overlap": false,
    "ssim_official_interval": "[0.670001,0.670945]",
    "ssim_gsw_interval": "[0.650710,0.651654]",
    "ssim_interval_overlap": false,
    "lpips_official_interval": "[0.300881,0.302607]",
    "lpips_gsw_interval": "[0.500238,0.501250]",
    "lpips_interval_overlap": false,
    "official_psnr_spread": 0.05673198699951243,
    "gsw_psnr_spread": 0.08779239654541016
  },
  {
    "role": "M",
    "scene": "web_Terrestrial",
    "trigger_third_run": true,
    "trigger_reasons": "gsw_psnr_spread_gt_0.3",
    "psnr_official_interval": "[20.599540,20.829492]",
    "psnr_gsw_interval": "[18.879516,19.216016]",
    "psnr_interval_overlap": false,
    "ssim_official_interval": "[0.766537,0.774376]",
    "ssim_gsw_interval": "[0.726293,0.740393]",
    "ssim_interval_overlap": false,
    "lpips_official_interval": "[0.399848,0.413196]",
    "lpips_gsw_interval": "[0.517876,0.534607]",
    "lpips_interval_overlap": false,
    "official_psnr_spread": 0.22995241483052453,
    "gsw_psnr_spread": 0.33649937311808387
  }
]
```
