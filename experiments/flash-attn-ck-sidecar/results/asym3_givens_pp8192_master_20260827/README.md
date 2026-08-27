# Asym3-Givens D256 current-master validation

Hardware and software:

- AMD Radeon Pro W7900, exact `gfx1100`
- ROCm 7.14
- Qwen3.6-27B MQ4
- Asym3-Givens K and Q8 V cache
- PP8192, batch 1, no speculative decoding
- branch base: upstream master `aaf5e3211`

Command:

```bash
BUILD=0 GPU_ID=0 KV_MODE=asym3 PREFILL=8192 RUNS=5 SLEEP_SECS=10 \
  ./scripts/bench_ck_q8_prefill_ab.sh
```

Results after two warmups per arm:

| Arm | Samples (tok/s) | Median |
| --- | --- | ---: |
| Native | `271.1, 582.2, 576.7, 572.5, 571.7` | `572.5` |
| CK | `806.4, 799.5, 797.4, 795.0, 794.3` | `797.4` |

The first native sample contains one-time JIT contamination; retaining it does
not change the five-run median. CK improves the median by `39.28%`. Both arms
produce next token `248046`.

The rocprof run records both an untimed JIT warmup pass and one profiled pass.
Dividing CK-specific aggregate dispatch times by two gives:

| Component | Time per PP8192 pass |
| --- | ---: |
| CK D256 FMHA | `283.6 ms` |
| Asym3 K decode | `28.5 ms` |
| Q8 V decode | `38.2 ms` |
| F16 output to F32 | `5.2 ms` |
| F32 Q Givens transform to F16 | `2.6 ms` |
| Total CK chain | `358.1 ms` |

This is about `3.3%` of the `10.8 s` profiled application wall after CK is
enabled. It bounds the benefit available from additional staging or CK-tile
work and keeps packed-MQ4 performance claims outside this PR.
