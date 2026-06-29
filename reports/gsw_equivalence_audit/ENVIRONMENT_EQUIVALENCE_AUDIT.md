# Environment Equivalence Audit

Generated: 2026-06-30T05:47:22

## Current audit/runtime environment

- Python executable: `D:\anaconda\envs\3dgs\python.exe`
- Python version: `3.10.18 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:08:55) [MSC v.1929 64 bit (AMD64)]`
- Working directory: `G:\WL3DGS\Improved_GS-W`
- Torch metric note: `device=cuda:0; lpips=alex; normalize=True; ssim=kornia.metrics.ssim(window=3)`

## Package versions

- torch: `2.0.1`
- torchvision: `0.15.2`
- lpips: `installed`
- kornia: `0.8.2`
- PIL: `10.2.0`
- numpy: `1.26.4`
- pandas: `2.3.3`
- plyfile: `installed`

## GPU / compiler probes

```text
$ nvidia-smi
Tue Jun 30 05:47:22 2026
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 526.98       Driver Version: 526.98       CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0  On |                  Off |
|  0%   43C    P2    70W / 450W |   2801MiB / 24564MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       736    C+G   ...cd70vzyy\ArmouryCrate.exe    N/A      |
|    0   N/A  N/A      5336    C+G   ...bbwe\Microsoft.Photos.exe    N/A      |
|    0   N/A  N/A      6648    C+G   ...e6\promecefpluginhost.exe    N/A      |
|    0   N/A  N/A      7688    C+G   ...03.112\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     10432    C+G   ...wekyb3d8bbwe\Video.UI.exe    N/A      |
|    0   N/A  N/A     11828    C+G   ...AweSun\flutter\AweSun.exe    N/A      |
|    0   N/A  N/A     14224    C+G   ...me\Application\chrome.exe    N/A      |
|    0   N/A  N/A     14896    C+G   D:\todesk\ToDesk.exe            N/A      |
|    0   N/A  N/A     20468    C+G   C:\Windows\System32\dwm.exe     N/A      |
|    0   N/A  N/A     22688    C+G   ...03.112\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     23616    C+G   ...03.112\msedgewebview2.exe    N/A      |
|    0   N/A  N/A     24748    C+G   ...2nqsd0c76g0\app\Codex.exe    N/A      |
|    0   N/A  N/A     29256    C+G   ...��\PCAnomalyDetection.exe    N/A      |
|    0   N/A  N/A     29936      C   ...onda\envs\3dgs\python.exe    N/A      |
|    0   N/A  N/A     30592    C+G   ...\current_new\DingTalk.exe    N/A      |
|    0   N/A  N/A     31932    C+G   C:\Windows\explorer.exe         N/A      |
|    0   N/A  N/A     36440    C+G   ...5n1h2txyewy\SearchApp.exe    N/A      |
|    0   N/A  N/A     38972    C+G   ...artMenuExperienceHost.exe    N/A      |
|    0   N/A  N/A     41432    C+G   ...nputApp\TextInputHost.exe    N/A      |
|    0   N/A  N/A     42300    C+G   ...y\ShellExperienceHost.exe    N/A      |
+-----------------------------------------------------------------------------+

$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:41:10_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

## Rasterizer signature probe

```text
GaussianRasterizationSettings._fields= ('image_height', 'image_width', 'tanfovx', 'tanfovy', 'bg', 'scale_modifier', 'viewmatrix', 'projmatrix', 'sh_degree', 'campos', 'prefiltered', 'debug', 'antialiasing')
```

## Historical environment limits

The historical run directory preserves `cfg_args`, logs, checkpoints, renders and metrics, but not a full exported conda environment. Therefore Python/PyTorch/CUDA identity for the historical 2026-06-21 run is not fully provable from files currently found. The historical working tree has a rasterizer compatibility patch adding `antialiasing=False`, which shows the active compiled rasterizer required the newer argument.

The torchvision pretrained/weights and grid_sample align_corners warnings can affect feature projection only indirectly through library behavior; they do not explain a deterministic split or metric leak by themselves.
