# gfx11 Q8 group128 PP8192 A/B

Qwen3.6-27B MQ4 on a Radeon Pro W7900 (`gfx1100`), Asym3 KV, quantized CK attention sidecar, 2048-token prefill chunks, and the existing X256/Y64 permutation routes. `HIPFIRE_RDNA3_Q8_GROUP128=1` shares one activation scale across each 128-value group while retaining four independent 32-value sums. The route is opt-in and gfx11-only.

| mode | raw prefill tok/s | median prefill tok/s | median decode tok/s |
| --- | --- | ---: | ---: |
| Q8 per-32 baseline | 920.6, 887.4, 885.1, 880.5, 880.7 | 885.1 | 33.0 |
| Q8 group128 | 1064.0, 1050.7, 1041.2, 1039.2, 1030.9 | **1041.2** | 33.0 |

The median prefill speedup is **1.1764x (+17.64%)**. All five candidate samples exceed all five baseline samples.

A final rebuild/cache check is archived in `pp8192_q8_group128_final_steady_20260809`: baseline `904.1 tok/s`, group128 `1043.2 tok/s`, or **1.1539x (+15.39%)**. An immediately preceding candidate run measured only `878.4 tok/s` because its timed prefill contained the one-time HIP source recompilation (`pre-compiled blob has no hash file, recompiling`). The archived steady-state logs contain no recompilation event; JIT-contaminated samples are excluded from the performance claim.

Standalone 10-pair medians:

| shape/path | baseline | group128 | speedup | max abs vs per-32 Q8 | mean abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| M17408 K5120 N2048 set | 6.5940 ms | 5.2419 ms | 1.2579x | 0.02441565 | 0.00600024 |
| M5120 K17408 N2048 residual-add | 7.0143 ms | 5.4017 ms | 1.2985x | 0.04749441 | 0.01434456 |

ISA resource audit for the group128 full set/add kernels: wave32, 252 VGPR, 31 SGPR, zero scratch, and zero VGPR/SGPR spills. The corresponding per-32 X256/Y64 kernels use 215 VGPR and 33 SGPR with zero scratch/spills.

Correctness checks are in the sibling `pp8192_q8_group128_correctness_20260809` directory. The 35-token PP8192 synthetic sequence and the 32-token real `docs/testINPUT.md` greedy sequence both match the per-32 baseline exactly. This is an initial routing/correctness gate, not a broad quality evaluation; the lower-precision activation format remains opt-in.

## Post-win boundary probes

The following narrow combinations were measured after the production A/B and rejected. None was promoted to the runtime route.

| candidate | group128 reference | candidate | result |
| --- | ---: | ---: | ---: |
| metadata single-loader + group128 | 5.2900 ms | 5.3128 ms | -0.43% |
| X128/Y64 group128 | 5.2642 ms | 5.4921 ms | -4.15% |
| X128/Y64 dual-128 LDS staging | 5.2733 ms | 5.5496 ms | -5.24% |
| equal-area X128/Y128, 16 waves | 5.2585 ms | 6.6630 ms | -21.08% |

All topology candidates matched the X256/Y64 group128 result (exactly or within `1.43e-6` max absolute difference). Explicit lane-local weight-scale caching also retained the same `252 VGPR / 0 spill` resource envelope and did not improve timing. These results keep X256/Y64 with eight waves as the measured default.

The post-group128 PP8192 profile reports 6925.7 ms serialized GPU time. Group128 HFQ4 GEMMs still account for about 81.7%: M17408/K5120 set 38.0%, M5120/K17408 residual-add 19.7%, M10240/K5120 set 8.5%, M5120/K6144 residual-add 7.0%, M6144/K5120 set 5.1%, and M12288/K5120 set 3.4%. The next optimization therefore requires a larger MMQ dataflow change; local loader/topology tweaks above do not provide another production win.
