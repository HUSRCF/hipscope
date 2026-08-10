# PP8192 packed-MQ4 runtime timeline (gfx1100)

This run resolves the gap between application wall time and Hipfire's internal
kernel timers. Hardware was a Radeon Pro W7900 (`gfx1100`), using GPU1, asym3
KV, the quantized CK prefill sidecar, and the production quad-row packed-MQ4
route. The measured prefill rate under `rocprofv3 --runtime-trace` was
1136.3 tok/s (`7209.57 ms`).

## Prefill timeline slice

The prefill interval is bounded by the first `embedding_q8_batched` dispatch
and the last `gemv_hfq4g256` dispatch. This gives a `7207.481 ms` device
window, within 2.1 ms of the application measurement.

| Component | GPU time (ms) | Prefill wall |
|---|---:|---:|
| All kernel intervals (union) | 7143.620 | 99.11% |
| No-kernel gaps | 63.861 | 0.89% |
| packed-MQ4 `full_set` | 3418.834 | 47.43% |
| packed-MQ4 `full_add` | 1657.879 | 23.00% |
| CK quantized FMHA | 811.990 | 11.27% |
| Gated DeltaNet core | 453.826 | 6.30% |
| fused SwiGLU/rotate | 185.330 | 2.57% |
| Conv1D SiLU | 149.776 | 2.08% |
| Q8 activation quantization | 111.487 | 1.55% |

Including the small residual fallback, gate/up tail, and LM-head MQ4 kernels,
the packed-MQ4 family occupies about 71.7% of the measured prefill wall.

The prior `1027.3 ms` internal-timer gap is therefore not a 13.8% host/idle
opportunity. The internal timers omit the external CK kernel (about 812 ms in
this trace), while the actual no-kernel gap is only 63.9 ms. Internal timing
for non-CK kernels is otherwise close to the external trace, subject to normal
run-to-run frequency variation.

## Packed-MQ4 resource audit

The current `full_set` and `full_add` code-object metadata is identical:

```text
wavefront_size:                 32
vgpr_count:                    256
vgpr_spill_count:                4
private_segment_fixed_size:     20 B/work-item
sgpr_count:                     27
sgpr_spill_count:                0
```

The dispatch allocates dynamic LDS as:

```text
(256 * 36 + 64 * 76) * sizeof(i32) + 256 * sizeof(f32)
= 57,344 B
```

The device reports 64 KiB LDS, so one workgroup consumes 87.5% of that
capacity. A 256-thread workgroup contains eight wave32 waves; LDS alone limits
the kernel to one such workgroup per 64 KiB allocation domain. The maxed VGPR
count and four VGPR spills independently show that the decode/staging live
range is also under pressure.

Static ISA counts for `full_set` are:

| Instruction family | Static count |
|---|---:|
| `v_wmma_i32_16x16x16_iu8` | 128 |
| `v_fma_mix_f32` | 200 |
| `v_cvt_f32_i32` | 128 |
| `v_dual_fmac_f32` | 59 |
| global loads | 75 |
| LDS loads | 131 |
| LDS stores | 48 |
| `s_waitcnt*` | 117 |
| barriers | 5 |
| `v_perm*` | 16 |

This is a WMMA path, not an MFMA path. The evidence supports a resource-heavy
MQ4 decode/scale/staging pipeline around WMMA; it does not prove a single
memory-bandwidth or compute-utilization cause.

## Counter boundary

ROCm 7.14 `rocprofv3` counter collection was tested on the production kernel:

- `FETCH_SIZE + WRITE_SIZE` aborted because the request exceeded hardware
  collection capability.
- `FETCH_SIZE`: 368/368 samples were zero.
- raw `GL2C_EA_RDREQ_128B_sum`: 368/368 samples were zero.
- raw `SQ_INSTS_VALU`: 368/368 samples were zero.

These PMC values are unusable on this host and must not be cited as evidence.
The runtime timeline, code-object metadata, and static ISA remain valid.

## Backend implications

Further tile tweaking and FFN intermediate plumbing are frozen. A replacement
backend should target the common large packed-MQ4 primitive used by gate, up,
down, and the other projections. The first design should reduce execution
format decode work, dynamic LDS below the one-workgroup threshold, VGPR live
ranges, and spills. Multi-output gate/up fusion should follow only after the
single-projection primitive has enough resource headroom; fusing outputs on
top of the current 256-VGPR/57-KiB design is likely to increase pressure.

Raw artifacts:

```text
trace_results.pftrace
trace_kernel_trace.csv
trace_hip_api_trace.csv
trace_memory_copy_trace.csv
trace_memory_allocation_trace.csv
bench.log
```
