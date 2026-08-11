# PP16384 production-path steady check

This run validates the retained Qwen3.6-27B MQ4 production configuration at
PP16384 on AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14. All three
prefill passes ran in one process. The first pass includes cold module/JIT
cost and is reported separately rather than treated as steady-state data.

## Configuration

- prefill: 16384 tokens, `HIPFIRE_PREFILL_MAX_BATCH=2048`
- runs: 3 in one process
- KV: asym3
- attention: staged quantized CK sidecar
- graph: disabled
- packed MQ4: X256/Y64, permuted nibble, group128 row2, quad-row weight
- FFN: fused SwiGLU, F16 intermediate
- auxiliary projections: group256 serial-row

## Result

| Pass | Wall (ms) | Throughput (tok/s) | Interpretation |
| ---: | ---: | ---: | --- |
| 1 | 17053.3 | 960.8 | cold/JIT-contaminated |
| 2 | 13863.6 | 1181.8 | steady |
| 3 | 14010.2 | 1169.4 | steady |

The two steady passes span 1169.4-1181.8 tok/s. The benchmark-reported
three-pass median is 1169.4 tok/s because the cold pass is retained in the
input set. Decode after the long prefill was 31.9 tok/s. The staged CK route
was explicitly observed in the log.

This result is a long-prefill functionality and stability check. It does not
replace a multi-process A/B comparison, and the cold pass must not be used as
a production throughput estimate.
