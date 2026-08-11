# Late-lane epilogue reconstruction probe on gfx1100

This exact-semantics probe tested whether reconstructing the Wave32 lane ID in
the output epilogue could remove work-item-derived VGPR spills from the retained
group128 quad-row packed-MQ4 kernel. The temporary path used
`__builtin_amdgcn_mbcnt_lo(~0u, 0u)` for the lane ID and preserved the logical
workgroup wave index. It did not change quantization, accumulation, output
indexing, or model routing.

Device: AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14. Each process ran 21
alternating pairs after warmup; four fresh processes were measured with five
seconds between processes.

## Resource result

| Resource | Production | Late-lane candidate |
| --- | ---: | ---: |
| VGPR allocation | 256 | 256 |
| VGPR spills | 4 | 2 |
| private bytes/thread | 20 | 12 |
| wave size | 32 | 32 |
| static instructions | 1513 | 1516 |

The remaining two spills hold the logical workgroup wave index. RDNA3 exposes
the physical SIMD wave slot through `HW_ID1.WAVE_ID`; that value is not an
exact replacement for the logical wave index within a workgroup.

## Paired timing

| Path | Production process medians (ms) | Candidate process medians (ms) | Aggregate |
| --- | --- | --- | ---: |
| gate/up set | 4.4281, 4.3956, 4.4194, 4.4516 | 4.4252, 4.4616, 4.4183, 4.4298 | -0.08% |
| down add | 4.5088, 4.5145, 4.4828, 4.5178 | 4.4987, 4.4998, 4.4981, 4.4911 | +0.29% |

Using the production call mix of two set projections per add projection, the
weighted local result is approximately +0.04%. Candidate and reference outputs
were identical (`max_abs=0`, `mean_abs=0`, relative L2=0, cosine=1).

Reducing two spills did not change the 256-VGPR allocation tier and produced
no stable performance gain. The source probe was removed and is not a
production route.
