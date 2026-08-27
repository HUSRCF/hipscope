# gfx11 GDN token-parallel causal conv

The retained Qwen3.6 prefill kernel assigned one thread to each channel and
looped serially over all tokens. A four-tap causal depthwise convolution only
depends on the current and three prior inputs, so the candidate computes every
`(token, channel)` independently and commits the final three-value state in a
second ordered kernel. Tree and independent-batch paths are unchanged.

## Correctness and local timing

`bench_conv1d_token_parallel` compares the old and new kernels at the real
Qwen3.6 GDN shape (`K_DIM=V_DIM=2048`, `N=2048`) and checks the state boundary
at `N=1/2/3`.

```text
short_correctness n=1 exact=true
short_correctness n=2 exact=true
short_correctness n=3 exact=true
Q/K/V/state max_abs at N=2048: 0
sequential median: 0.6279 ms
parallel median:   0.1392 ms
speedup:           4.5113x
```

## PP8192 product A/B

The model test used Qwen3.6-27B MQ4 on a W7900/gfx1100, chunk size 2048,
Asym3 KV, staged quantized CK attention, and the retained packed-MQ4/F16-FFN
stack. Each fresh process reported the median of three prefill runs.

| Pair | Baseline tok/s | Parallel tok/s | Ratio |
|---:|---:|---:|---:|
| 1 | 1240.5 | 1236.0 | 0.9964x |
| 2 | 1200.8 | 1220.4 | 1.0163x |
| 3 | 1204.5 | 1208.1 | 1.0030x |

Paired median is `1.0030x`, with two of three pairs positive. Recorded token
IDs matched in all arms. The route remains opt-in because the product-level
gain is small relative to run-to-run thermal variation.

`rocprofv3 --kernel-trace` confirms that the intended hotspot changed:

| Path | Calls | Total GPU time |
|---|---:|---:|
| Serial conv | 192 | 152.551 ms |
| Parallel conv | 192 | 46.784 ms |
| Parallel state commit | 192 | 0.456 ms |

The full-model conv block therefore falls from `152.551 ms` to `47.240 ms`
(`3.23x`, 69.0% less GPU time). The remaining PP8192 wall is dominated by
packed-MQ4 projections, so the isolated kernel win translates to only a small
end-to-end increase.
