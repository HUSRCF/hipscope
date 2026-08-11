# Warmed PP16384 production-path runtime profile

This trace attributes the retained Qwen3.6-27B MQ4 production configuration
at PP16384 on AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14. The output
directory retains the old `pp8192` prefix because the profiling script had a
hard-coded directory label; `manifest.txt` is authoritative and records
`prefill_tokens=16384`. The script now derives this label from
`PREFILL_TOKENS`.

The first PP16384 pass warmed module JIT and GPU state. The analyzer selected
the second pass, from its first `embedding_q8_batched` dispatch through its
final prefill LM-head dispatch, and restricted attribution to the matching GPU
agent.

## Configuration

- prefill: 16384 tokens, `HIPFIRE_PREFILL_MAX_BATCH=2048` (8 chunks)
- KV: asym3
- attention: staged quantized CK sidecar
- graph: disabled
- packed MQ4: X256/Y64, permuted nibble, group128 row2, quad-row weight
- FFN: fused SwiGLU, F16 intermediate
- auxiliary projections: group256 serial-row

The two application passes were 1181.1 and 1164.9 tok/s. This profiler run is
used for attribution, not as a replacement for the unprofiled steady check.

## Runtime attribution

```text
window_ms         14063.942
kernel_busy_ms    13943.278
no_kernel_gap_ms    120.665  (0.86%)
dispatches            10935
```

| Category | Calls | Time (ms) | Wall |
| --- | ---: | ---: | ---: |
| packed MQ4 set | 2176 | 7149.458 | 50.84% |
| packed MQ4 add | 1024 | 3455.275 | 24.57% |
| CK attention and bridges | 512 | 1244.610 | 8.85% |
| GDN core | 384 | 961.077 | 6.83% |
| other | 4015 | 515.042 | 3.66% |
| Conv1D SiLU | 384 | 309.056 | 2.20% |
| fused SwiGLU rotate | 512 | 169.435 | 1.20% |
| MQ4 tails and LM head | 392 | 73.428 | 0.52% |
| Q8 activation quantization | 1536 | 65.896 | 0.47% |

Packed MQ4 occupies 10604.733 ms, or 75.41% of the selected PP16384 wall.
Attention grows from 4.72% at PP8192 to 8.85% here, but packed MQ4 remains the
dominant optimization target. The union of GPU kernel intervals covers
99.14% of the window, so host submission gaps are not a useful target.

## Dominant kernels

| Kernel family | Calls | Time (ms) | Wall |
| --- | ---: | ---: | ---: |
| group128 quad-row F16 full-set | 1024 | 5056.623 | 35.95% |
| group128 quad-row full-add | 512 | 2596.052 | 18.46% |
| group256 serial-row full-set | 1152 | 2092.835 | 14.88% |
| CK FMHA | 128 | 1145.975 | 8.15% |
| GDN Q8 core | 384 | 961.077 | 6.83% |
| group256 serial-row full-add | 512 | 859.224 | 6.11% |

The raw `.pftrace` and per-dispatch CSVs are intentionally not committed.
Re-run `run_pp8192_best_runtime_profile.sh` with `PREFILL_TOKENS=16384` to
regenerate them. The analyzer, manifest, hashes, and this summary preserve the
reviewable attribution contract.
