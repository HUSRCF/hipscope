# Group128 X512 Direct-Activation Standalone Probe

Radeon Pro W7900 (`gfx1100`), GPU1, exact Qwen3.6-27B gate/up projection
shape (`M=17408`, `K=5120`, `N=2048`). Fifteen alternating pairs were run
after three warmup pairs.

| Path | Median |
| --- | ---: |
| Production group128 LDS, X256/Y64 | 5.0547 ms |
| Group128 direct activation, X512/Y64 | 5.0364 ms |

The standalone candidate is **1.0036x (+0.36%)**. Correctness remains exact
within the established floating-point tolerance (`max_abs=1.43e-6`,
`mean_abs=1.10e-7`).

Resource audit for the set kernels:

| Kernel | Workgroup | VGPR | SGPR | Scratch |
| --- | ---: | ---: | ---: | ---: |
| X256/Y64 group128 direct full-set | 256 | 140 | 34 | 0 B |
| X512/Y64 group128 direct full-set | 512 | 166 | 34 | 0 B |

The wider workgroup halves repeated weight staging but increases the thread
count and VGPR allocation. The standalone gain is too small to justify wider
production rollout without an end-to-end positive result.
