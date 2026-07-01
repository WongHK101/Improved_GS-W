# OFFICIAL_ENVIRONMENT

## Conda/Python Imports

```json
{
  "torch": "2.0.1",
  "torch_cuda": "11.8",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 4090",
  "torchvision": "0.15.2",
  "kornia": "0.8.2",
  "lpips_package": "D:\\anaconda\\envs\\3dgs\\lib\\site-packages\\lpips\\__init__.py",
  "diff_gaussian_rasterization": "D:\\anaconda\\envs\\3dgs\\lib\\site-packages\\diff_gaussian_rasterization\\__init__.py",
  "SparseGaussianAdam": false,
  "simple_knn": null,
  "fused_ssim_spec": "ModuleSpec(name='fused_ssim', loader=<_frozen_importlib_external.SourceFileLoader object at 0x000000007305F4F0>, origin='D:\\\\anaconda\\\\envs\\\\3dgs\\\\lib\\\\site-packages\\\\fused_ssim\\\\__init__.py', submodule_search_locations=['D:\\\\anaconda\\\\envs\\\\3dgs\\\\lib\\\\site-packages\\\\fused_ssim'])",
  "python": "3.10.18 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:08:55) [MSC v.1929 64 bit (AMD64)]"
}
```

## nvidia-smi

```text
Wed Jul  1 16:48:01 2026       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 526.98       Driver Version: 526.98       CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  Off |
|  0%   40C    P8    23W / 450W |    874MiB / 24564MiB |      5%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       736    C+G   ...cd70vzyy\ArmouryCrate.exe    N/A      |
|    0   N/A  N/A      5336    C+G   ...bbwe\Microsoft.Photos.exe    N/A      |
|    0   N/A  N/A      7688    C+G   ...03.112\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     10432    C+G   ...wekyb3d8bbwe\Video.UI.exe    N/A      |
|    0   N/A  N/A     14224    C+G   ...me\Application\chrome.exe    N/A      |
|    0   N/A  N/A     14896    C+G   D:\todesk\ToDesk.exe            N/A      |
|    0   N/A  N/A     15244    C+G   ...e6\promecefpluginhost.exe    N/A      |
|    0   N/A  N/A     20468    C+G   C:\Windows\System32\dwm.exe     N/A      |
|    0   N/A  N/A     22688    C+G   ...03.112\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     24748    C+G   ...2nqsd0c76g0\app\Codex.exe    N/A      |
|    0   N/A  N/A     29256    C+G   ...��\PCAnomalyDetection.exe    N/A      |
|    0   N/A  N/A     30220    C+G   ...03.112\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     30540    C+G   ...AweSun\flutter\AweSun.exe    N/A      |
|    0   N/A  N/A     30592    C+G   ...\current_new\DingTalk.exe    N/A      |
|    0   N/A  N/A     31932    C+G   C:\Windows\explorer.exe         N/A      |
|    0   N/A  N/A     36440    C+G   ...5n1h2txyewy\SearchApp.exe    N/A      |
|    0   N/A  N/A     38972    C+G   ...artMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A     41432    C+G   ...nputApp\TextInputHost.exe    N/A      |
|    0   N/A  N/A     42300    C+G   ...y\ShellExperienceHost.exe    N/A      |
+-----------------------------------------------------------------------------+
```

## nvcc

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

## MSVC cl

```text
[WinError 2] 系统找不到指定的文件。
```

## Fixed Protocol

- `optimizer_type=default` because installed `diff_gaussian_rasterization` does not expose `SparseGaussianAdam`.
- `antialiasing=False`, `depths=''`, `white_background=False`, `random_background=False`, `resolution=1`, `sh_degree=3`.
- `train_test_exp` is not enabled. Current official source still creates an exposure optimizer, but `render.py` only applies trained exposure when `dataset.train_test_exp=True`.
- Unified evaluation uses LPIPS AlexNet from the installed `lpips` package with `normalize=True` to match prior GS-W metrics.
