# Warmed PP8192 production-path runtime profile

This is the authoritative warmed runtime attribution for the retained
Qwen3.6-27B MQ4 prefill configuration on AMD Radeon Pro W7900 (`gfx1100`),
GPU1, ROCm 7.14. The first PP8192 pass warmed module JIT and GPU state. The
trace window contains only the second PP8192 pass, from its first
`embedding_q8_batched` dispatch through the final prefill LM-head dispatch.
The profiler launched one isolated benchmark child; the analyzer additionally
restricts attribution to the GPU agent that emitted the selected prefill
marker. Do not reuse this attribution method for a shared-process workload
with unrelated work on the same GPU agent.

## Configuration

- prefill: 8192 tokens, `HIPFIRE_PREFILL_MAX_BATCH=2048`
- KV: asym3
- attention: staged quantized CK sidecar
- graph: disabled
- packed MQ4: X256/Y64, permuted nibble, group128 row2, quad-row weight
- FFN: fused SwiGLU, F16 intermediate
- auxiliary projections: group256 serial-row

The two application runs were 1245.7 and 1237.9 tok/s. The reported median
was 1237.9 tok/s. This profile is used for attribution, not as a replacement
for the multi-process performance baseline.

## Runtime attribution

```text
window_ms          6616.884
kernel_busy_ms     6560.413
no_kernel_gap_ms     56.471  (0.85%)
dispatches             5467
```

| Category | Calls | Time (ms) | Wall |
| --- | ---: | ---: | ---: |
| packed MQ4 set | 1088 | 3514.114 | 53.11% |
| packed MQ4 add | 512 | 1698.060 | 25.66% |
| GDN core | 192 | 471.926 | 7.13% |
| CK attention and bridges | 256 | 312.522 | 4.72% |
| other | 2007 | 257.280 | 3.89% |
| Conv1D SiLU | 192 | 153.243 | 2.32% |
| fused SwiGLU rotate | 256 | 84.546 | 1.28% |
| MQ4 tails and LM head | 196 | 36.147 | 0.55% |
| Q8 activation quantization | 768 | 32.584 | 0.49% |

The packed-MQ4 family occupies 5212.174 ms, or 78.77% of the measured
prefill wall. Host-side or launch-gap work is not a useful optimization target
in this configuration: the union of GPU kernel intervals covers 99.15% of the
window.

## Dominant kernels

| Kernel family | Calls | Time (ms) | Wall |
| --- | ---: | ---: | ---: |
| group128 quad-row F16 full-set | 512 | 2484.502 | 37.55% |
| group128 quad-row full-add | 256 | 1276.312 | 19.29% |
| group256 serial-row full-set | 576 | 1029.612 | 15.56% |
| GDN Q8 core | 192 | 471.926 | 7.13% |
| group256 serial-row full-add | 256 | 421.748 | 6.37% |
| CK FMHA | 64 | 284.086 | 4.29% |

The raw `.pftrace` and full per-dispatch CSVs are intentionally not committed;
they are large generated artifacts. Re-run
`run_pp8192_best_runtime_profile.sh` to regenerate them. The analyzer and this
summary preserve the attribution contract needed for review.
