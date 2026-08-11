# PP16384 staged CK ABBA validation

This run validates the optional quantized-KV CK sidecar on a Qwen3.6-27B MQ4 prefill workload. The benchmark binary was built with the `flash-attn-ck` Cargo feature. Samples were run on GPU 1 in `off/on/on/off` order with a 30-second idle interval and two prefill measurements per process.

## Performance

| Sample | CK sidecar | Prefill tok/s | Decode tok/s |
| --- | --- | ---: | ---: |
| 01 | off | 511.0 | 31.8 |
| 02 | on | 1157.0 | 31.8 |
| 03 | on | 1138.0 | 31.7 |
| 04 | off | 510.3 | 31.9 |

The off mean is `510.65 tok/s`; the on mean is `1147.50 tok/s`. The optional CK path is `2.247x` faster (`+124.7%`) at PP16384. Decode throughput is unchanged within run-to-run noise.

## Numerical boundary

The first post-prefill top-1 token is identical (`82`) and has a large margin in both paths. A full-vocabulary F32 logit comparison reports:

| Metric | Value |
| --- | ---: |
| cosine similarity | 0.9999373 |
| mean absolute difference | 0.0266794 |
| RMSE | 0.0343131 |
| maximum absolute difference | 0.2411977 |
| off top-1 margin | 0.918776 |
| on top-1 margin | 1.035300 |

The eight-token greedy continuation is not bit-identical after the shared initial tokens. This is consistent with accumulated numerical differences, but it means this result is performance and plumbing evidence rather than a complete quality-equivalence claim. The CK sidecar remains gated by both the `flash-attn-ck` Cargo feature and `HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB`; it is not made the default backend by this experiment.

## Reproduction

Run from the repository root after building the benchmark with `--features deltanet,flash-attn-ck` and providing the staged sidecar:

```bash
HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB=/tmp/libhipfire_flash_attn_ck_quantized_staged.so \
GPU_ID=1 \
experiments/gfx11-gate-up-x256y64/run_pp16384_staged_ck_abba.sh
```

`artifacts.sha256` records the exact model, sidecar, and benchmark binary used. `manifest.txt`, `results.tsv`, and the four raw logs preserve the run configuration and measurements.
