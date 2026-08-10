# Q8 X256 Small-LDS Probe

This standalone gfx1100 probe tests whether a 256-token by 64-output, 16-wave
workgroup can amortize the small-LDS Q8 kernel's staged HFQ4 weights across
twice as many tokens. It is not connected to production dispatch.

## Configuration

- GPU: AMD Radeon Pro W7900, gfx1100, GPU1
- Shape: M=17408, K=5120, N=2048
- Mode: set, permuted-nibble reference
- Samples: 21 per variant
- Candidate launch: 512 threads, token tile 256, output tile 64, K tile 32

## Overall Results

| Variant | Median (ms) | Relative to current baseline |
|---|---:|---:|
| Current X256Y64 group256 baseline | 6.6009 | 1.0000x |
| Small-LDS Q8 X128Y64 K32 | 8.1134 | 0.8136x |
| Small-LDS Q8 X256Y64 K32 | 7.9044 | 0.8351x |
| Small-LDS Q8 X128Y64 K64 | 7.9046 | 0.8351x |

Increasing the token tile from 128 to 256 improves the K32 probe by 2.64%,
consistent with reducing duplicated weight staging. It remains 16.49% slower
than the current X256Y64 baseline, so weight-stage duplication is not the main
limit of this route.

Correctness against the current Q8 reference:

| Mode | max_abs | mean_abs |
|---|---:|---:|
| set, 21 samples | 2.86102295e-6 | 1.97867147e-7 |
| residual-add smoke, 3 samples | 2.86102295e-6 | 1.97863714e-7 |

Resource audit for the candidate set kernel:

| Resource | Value |
|---|---:|
| VGPR | 80 |
| SGPR | 24 |
| VGPR spills | 0 |
| SGPR spills | 0 |
| Fixed LDS | 5632 B |

Decision: keep this as a diagnostic probe only. Do not route production work
to the small-LDS Q8 family.
