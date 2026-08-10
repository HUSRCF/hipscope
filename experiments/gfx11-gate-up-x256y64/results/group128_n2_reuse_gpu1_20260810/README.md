# Group128 N2 Fragment-Reuse Probe

GPU: Radeon Pro W7900 (`gfx1100`), GPU1. Baseline and candidate were alternated for 15 pairs after warmup.

The candidate preserves the production X256/Y64 tile and exact K128 scaling, but loads each K32 HFQ4 weight fragment once for two adjacent activation subtiles.

| Shape | Operation | Baseline | N2 reuse | Speedup | Correctness |
|---|---|---:|---:|---:|---:|
| M17408 K5120 N2048 | set (gate/up) | 4.7143 ms | 4.6917 ms | 1.0048x | max_abs 0 |
| M5120 K17408 N2048 | add (down/residual) | 4.8460 ms | 4.7496 ms | 1.0203x | max_abs 0 |

Five independent processes produced median speedups of 1.0168x for gate/up and 1.0222x for down/residual; all outputs remained bit-exact. Resource audit for the full set/add kernels: baseline and N2 both use 252 VGPR, zero scratch, wave32. The optimization therefore removes real LDS A-fragment reloads, but those reloads are not a major gate/up cost.

N4 reuse was also tested as an upper bound. It slowed gate/up to 0.9536x and compiled with 256 VGPR plus 12 VGPR spills, so it is rejected.

The production PP8192 A/B was neutral: baseline 1061.6 tok/s versus N2 1062.8 tok/s (`1.0011x`, +0.11%), with identical token IDs. N2 therefore remains a standalone characterization and is not routed into serving.
