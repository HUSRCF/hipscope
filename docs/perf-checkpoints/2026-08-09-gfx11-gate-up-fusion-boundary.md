# gfx11 Qwen3.6-27B gate/up fusion boundary

This checkpoint uses the complete production projection shape `gate_m=up_m=17408`, `K=5120`, `N=2048` on a W7900/gfx1100. It does not use the older `M=16384` proxy and does not compare effective quantized GEMM FLOP/s against dense FP16 peak.

The shipping production path quantizes the activation once and launches two aligned `gemm_hfq4g256_residual_mmq_full_set` kernels. A one-launch wrapper that virtually concatenated the separate gate/up allocations was bit-exact but slower:

| Variant | Time per gate/up pair | Relative to production |
|---|---:|---:|
| Two production `full_set` launches | 12.953 ms | 1.000x |
| One launch, no tile sharing | 13.191 ms | 0.982x |

A `128M x 64N` paired prototype retained two 64-column activation halves in LDS and computed gate/up in turn. It was bit-exact but compiled to `256 VGPR`, `720 VGPR spills`, and `2100 bytes` of private storage per thread, versus `240 VGPR`, zero spills, and zero private bytes for shipping `full_set`. Runtime regressed to 99-106 ms per pair. Combining the two accumulator declarations into one compile-time array did not remove the spills.

Conclusion: launch collapse alone is not useful, and paired gate/up fusion exceeds the gfx11 register budget in this formulation. Neither prototype is routed into serving. Further work should start from a fresh production profile after the combined-zero optimization rather than extending these kernels.
