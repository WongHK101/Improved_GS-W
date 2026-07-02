# POST_DENSIFICATION_PILOT

| role | scene | completed | pass | stop | peak allocated / total MB | ratio | final gaussians | recent iter ms | remaining 25k min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H | self_Steam_Locomotive | True | True | False | 9586.09/24563.0 | 0.3903 | 109627 | 186.326 | 77.64 |
| M | web_Terrestrial | True | True | False | 14244.086/24563.0 | 0.5799 | 317365 | 227.092 | 94.62 |

PASS requires completion, no OOM, finite loss, loadable checkpoint, peak allocated < 90% total GPU memory, no rapid monotonic 4000-5000 memory growth, and manageable Gaussian count.
