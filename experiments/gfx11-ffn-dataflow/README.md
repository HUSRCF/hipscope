# gfx11 FFN dataflow probes

These probes test FFN-level changes after the production packed-MQ4 kernel
reached 1101.2 prefill tok/s on Qwen3.6-27B PP8192. Most cases are standalone;
the FP16-intermediate case also has an opt-in, default-off serving path.

Production shape:

```text
gate/up: M=17408, K=5120, N=2048
GPU: Radeon Pro W7900, gfx1100
ROCm runtime: 7.14
```

Run:

```bash
GPU_ID=1 PAIRS=10 ./experiments/gfx11-ffn-dataflow/run_gate_up_dataflow_ab.sh
```

Observed on 2026-08-10:

| Comparison | Baseline | Candidate | Speedup | Correctness |
|---|---:|---:|---:|---:|
| serial vs two HIP streams | 9.8604 ms | 9.7443 ms | 1.0119x | max abs 0 |
| FP32 vs FP16 gate/up output | 9.0596 ms | 8.9386 ms | 1.0135x | max abs 9.77e-4, mean abs 1.02e-4 |
| split vs concatenated gate/up projection | 9.1830 ms | 9.1598 ms | 1.0025x | max abs 0 |
| full FP32 vs FP16 FFN intermediate path | 14.7030 ms | 14.1860 ms | 1.0364x | final output max abs 2.89e-2, mean abs 4.81e-3 |

The two projections do not expose useful stream-level overlap. Concatenating
their weights and output into one projection is also neutral. Halving only the
gate/up output traffic changes projection time by about 1-2%, so the production
bottleneck is not the FP32 intermediate store.

## Production-style FP16 intermediate A/B

The full FP16 intermediate path was integrated behind
`HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE=1`. It is restricted to the verified
gfx1100 MQ4 gate/up/down shapes; the default remains off and partial batches
fall back to the original FP32 path.

Run:

```bash
GPU_ID=1 PAIRS=3 COOL_SECS=20 \
  ./experiments/gfx11-ffn-dataflow/run_pp8192_f16_intermediate_ab.sh
```

Results:

| Pair | Off (tok/s) | On (tok/s) | Paired ratio |
|---:|---:|---:|---:|
| 1 | 1144.0 | 1143.7 | 0.9997x |
| 2 | 1107.8 | 1134.2 | 1.0238x |
| 3 | 1102.6 | 1115.1 | 1.0113x |

The paired-ratio median is **1.0113x**. A separate PP8192 greedy token-ID
check produced identical token sequences with the option off and on; its
prefill rates were 1133.4 and 1142.7 tok/s respectively (1.0082x). Logs are
under:

```text
experiments/gfx11-ffn-dataflow/results/pp8192_f16_intermediate_20260811_000438/
experiments/gfx11-ffn-dataflow/results/pp8192_f16_intermediate_quality_20260810/
```

## Decision

Keep tile tuning frozen. The tested FFN dataflow changes are correct or
numerically bounded, but none gives a material production gain: the strongest
serving result is only about 1% and remains sensitive to thermal/frequency
drift. The FP16-intermediate route stays experimental and default-off; it is
not evidence for progress toward the 1500 tok/s target.
