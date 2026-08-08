# CK-on PP8192 rocprof breakdown

This attribution run used the same W7900, Qwen3.6-27B MQ4, Asym3 KV, PP8192,
and 2048-token prefill chunks as the production-route A/B. `rocprofv3`
recorded the natural asynchronous path without `HIPFIRE_PROFILE` serialization.
The prefill interval is the first `embedding_q8_batched` dispatch through the
last prefill `gemv_hfq4g256`, ending immediately before the first scalar
`embedding_q8` decode dispatch.

The isolated GPU span was 9813.587 ms versus the application's 9815.56 ms
prefill wall time. The 1.97 ms difference confirms that the trace window
captures the production prefill interval closely.

| component | GPU ms | prefill wall share |
| --- | ---: | ---: |
| MQ4 projection/set family | 5291.164 | 53.917% |
| MQ4 residual-add family | 2491.980 | 25.393% |
| **MQ4 GEMM total** | **7783.144** | **79.310%** |
| Q8_1 MMQ input quantization | 108.447 | 1.105% |
| **MQ4 GEMM plus input quantization** | **7891.591** | **80.415%** |
| quantized CK attention | 822.577 | 8.382% |
| GDN fast core | 458.371 | 4.671% |
| remaining kernels and inter-dispatch gaps | 641.048 | 6.532% |

Within the set family, the 512 calls at grid `4352x128` account for 3534.069
ms, or 36.01% of the entire prefill. This is the 27B FFN gate/up-scale class.
The residual-add family contributes another 25.39%, covering the hidden-size
output projections such as attention output and FFN down.

The dispatch counts also close exactly against the model topology. PP8192 at
2048-token chunks executes four prefill chunks. Per chunk, 48 GDN layers issue
four QKVZA projections, 16 full-attention layers issue three QKV projections,
and all 64 layers issue two FFN gate/up projections:

```text
48 * 4 + 16 * 3 + 64 * 2 = 368 set calls/chunk
368 * 4 chunks = 1472 set calls
64 layers * 2 residual projections * 4 chunks = 512 residual-add calls
```

The measured `1472` set and `512` residual-add calls therefore match the model
graph, rather than relying only on kernel-name attribution.

An independent `HIPFIRE_PROFILE=1` run reported 5296.1 ms for
`gemm_hfq4g256_mmq_set` and 2491.5 ms for
`gemm_hfq4g256_residual_mmq`, or 78.8% of its 9886.6 ms wall time. The close
agreement with rocprof supports the attribution; the headline percentages
above use rocprof.

## Amdahl budget

Applying the measured 79.31% GEMM fraction to the unprofiled 808.1 tok/s
median gives:

| MQ4 GEMM speedup | projected total speedup | projected prefill tok/s |
| ---: | ---: | ---: |
| 1.10x | 1.078x | 871 |
| 1.20x | 1.152x | 931 |
| 1.25x | 1.189x | 960 |
| 1.50x | 1.359x | 1099 |
| 2.00x | 1.657x | 1339 |

The mathematical all-GEMM upper bound is 4.83x, but it is not a practical
claim. A more actionable bound is the 53.92% set-family bucket: improving that
bucket by 20% projects to 1.099x overall (about 888 tok/s), while a 50%
improvement projects to 1.219x (about 985 tok/s). Optimizing only Q8_1 input
quantization can recover at most 1.1% overall; eliminating CK attention
entirely can recover at most 8.4%.

The next structural target is therefore a real multi-output gfx11 MMQ path for
QKV/QKVZA/gate-up, followed by residual-add MMQ. Further CK-attention or
quantizer tuning has substantially less Amdahl leverage.

The absolute throughput in this table deliberately uses the three-run 808.1
tok/s benchmark median. The profiled process itself covered 8192 tokens in
9.81556 s, or about 834.6 tok/s. The component fractions and Amdahl multipliers
come from that profile run; applying them to the independent median is a
cross-run projection, not a claim that 8192 / 9.81556 equals 808.1.

## Gate/up multi-output probes

Two resource-audited gfx11 gate/up prototypes tested whether sharing the Q8_1
activation tile could reduce the dominant set-family cost. Both compared two
MQ4 projections at the canonical Qwen3.6 shape `M_gate=M_up=17408`, `K=5120`,
`N=2048` against the production pair of 128x128 full-set kernels.

| implementation | median us/call | relative to production | max abs | VGPR | scratch |
| --- | ---: | ---: | ---: | ---: | ---: |
| production two full-set launches | 13,554-13,607 | 1.000x | 0 | 250 | 0 B |
| paired 64-row tiles, 8 waves | 17,925-17,997 | 0.756x | 0 | 158 | 0 B |
| paired 128-row half-K tiles, 16 waves | 15,888 | 0.853x | 0 | 160 | 0 B |
| half-K with dword weight loader | 17,275 | 0.788x | 0 | not retained | 0 B |

The 64-row design preserves the aggregate workgroup count and therefore does
not actually reduce activation-tile loads. The half-K design does load each
activation half once for both projections without increasing weight bytes, but
its 16-wave workgroup still loses 17.2% at the canonical shape. Neither failure
is caused by scratch or numerical drift. These probes reject direct gate/up
multi-output fusion for the current LDS/WMMA organization; no production route
was enabled. The next experiment should optimize the existing single-output
full-set/full-add kernel instead of widening the fusion degree.

## Full-set specialization check

The canonical single-output shape (`M=17408, K=5120, N=2048`) was also run
through the lower-register generic set kernel to test whether the full-set
kernel's high VGPR count was itself the bottleneck. Both kernels consumed the
same pre-quantized Q8_1 activation and were alternated for nine trials of 20
launches after warmup.

| kernel | median ms | relative throughput | max abs |
| --- | ---: | ---: | ---: |
| production `full_set` | 6.629 | 1.000x | reference |
| generic set | 8.969 | 0.739x | 0.0 |

The generic kernel is 35.3% slower despite its lower static VGPR count. The
full specialization's removal of bounds checks and the runtime add branch is
therefore performance-critical at this shape; replacing it with the generic
path is not a viable occupancy optimization.

## HFQ4-G256 combined-zero correction

HFQ4-G256 uses one affine zero point across each 256-element weight group. The
previous full kernels applied the same zero point separately to eight Q8_1
activation subblock sums. The optimized path keeps every scale-dependent i8
WMMA term unchanged, accumulates the eight activation sums in 512 B of LDS,
and applies one zero-point correction per group.

The pre-barrier aligned `full_add` diagnostic at `M=5120, K=5120, N=2048`
measured:

| implementation | median ms | speedup | max abs | VGPR | scratch |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy zero correction | 2.2114 | 1.000x | reference | 250 | 0 B |
| combined zero correction | 2.0963 | **1.0549x** | 2.62e-6 | 240 | 0 B |

The same transformation improved aligned `full_set` by 5.3%-5.5% across the
Qwen3.6 projection M values 5120, 10240, 12288, and 17408. Independent review
then identified the need for a barrier after the correction and before the
next K group reuses LDS. With that barrier present in both full-set and
full-add, the three fresh-process PP8192 CK results were 863.4, 850.5, and
855.0 tok/s, for a **855.0 tok/s median**. This is 5.80% above the previous
808.1 tok/s CK median. Final metadata reports 240 VGPR and zero scratch/spills
for both aligned kernels. A separate 3369-token real-prompt run preserved the
exact 16-token greedy output sequence used by the earlier native/CK correctness
check.
