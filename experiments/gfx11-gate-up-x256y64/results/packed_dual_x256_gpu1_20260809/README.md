# gfx11 Packed-Dual Gate/Up Negative Result

This standalone experiment tested whether one X256 workgroup could compute
both Qwen3.6 gate and up projections while sharing a compact Q8-group128
activation tile. It did not change the serving dispatch.

Hardware: Radeon Pro W7900 (`gfx1100`), selected with
`HIP_VISIBLE_DEVICES=1`. Shape: `M=17408, K=5120, N=2048`; results below are
15-pair medians after warmup.

| Path | Two-launch reference | Candidate | Relative | Gate/up max abs |
| --- | ---: | ---: | ---: | ---: |
| All rows packed | 9.0289 ms | 21.6633 ms | **0.4168x** | 4.396e-2 |
| Hybrid 15 packed + 49 expanded rows/plane | 9.2287 ms | 37.2976 ms | **0.2474x** | 4.396e-2 |

The hybrid layout exactly fits the 65,280-byte LDS budget, but the code object
resource audit identifies the actual failure:

| Kernel | VGPR | SGPR | VGPR spills | Private bytes/thread |
| --- | ---: | ---: | ---: | ---: |
| Production X256/Y64 row2 set | 252 | 31 | 0 | 0 |
| Packed-dual X256 gate/up | 256 | 45 | 208 | 600 |

The packed-dual kernel therefore loses to two production launches because the
combined unpack/decode and dual-output live ranges force extensive scratch
traffic. This is not a warmup or launch-count result. Stop this fusion line;
future work must first reduce the single-output accumulator/live-range cost.

The candidate remains standalone-only and must not be routed into serving.
