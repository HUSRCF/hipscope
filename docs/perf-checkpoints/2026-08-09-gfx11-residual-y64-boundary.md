# gfx11 residual Y64/W4 boundary

The barrier-safe `MMQ_Y=64`, four-wave residual prototype was tested because a standalone `M=5120, K=17408, N=2048` case improved from 7.8793 ms to 7.6911 ms (`+2.45%`). The same prototype regressed at `N=4096` (`15.0789 ms` to `15.3483 ms`).

Production PP8192 testing used Qwen3.6-27B MQ4, asym3 KV, chunk size 2048, and the quantized CK attention sidecar. Three fresh-process pairs alternated baseline/candidate order:

| Pair | Baseline prefill | Y64/W4 prefill | Paired delta |
|---:|---:|---:|---:|
| 1 | 873.4 tok/s | 850.5 tok/s | -2.62% |
| 2 | 845.1 tok/s | 841.1 tok/s | -0.47% |
| 3 | 848.2 tok/s | 844.6 tok/s | -0.42% |
| Mean | 855.57 tok/s | 845.40 tok/s | -1.19% |

A profiled candidate run confirmed that the production dispatcher entered `gemm_hfq4g256_residual_mmq_y64_w4`, but its serialized residual time was 2340.2 ms versus 2333.3 ms in the current baseline profile. Decode remained effectively unchanged at 33.0-33.2 tok/s.

Conclusion: the standalone N=2048 gain does not transfer to the full model's projection mix. The feature flag and production route were removed; Y64/W4 is not a shipping candidate.
