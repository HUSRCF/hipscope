# Exact Group128 Direct-Activation Production A/B

Qwen3.6-27B MQ4, PP8192 with 2048-token chunks, Asym3 KV, optional CK
attention sidecar, Radeon Pro W7900 (`gfx1100`), GPU1. The candidate preserves
the group128 quantization contract and is controlled by the opt-in
`HIPFIRE_RDNA3_Q8_GROUP128_DIRECT=1` flag.

Five alternating fresh-process pairs were collected after both paths were
prewarmed. Each process used the median of three prefill runs.

| Path | Prefill samples (tok/s) | Median | Decode median |
| --- | --- | ---: | ---: |
| Production LDS baseline | 1065.0, 1059.8, 1060.3, 1060.7, 1056.7 | 1060.3 | 33.0 |
| Direct activation | 1048.1, 1046.5, 1045.2, 1044.5, 1045.7 | 1045.7 | 33.0 |

The candidate is **0.9862x (-1.38%)** and all short-output token IDs match.
Although isolated kernels improved by 1.36-3.88%, direct global activation
loads do not improve the full production workload. Keep the path disabled and
do not promote it to the default dispatch.
