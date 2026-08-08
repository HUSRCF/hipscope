# gfx11 Gate/Up Structural Probes

Hardware: Radeon Pro W7900 (`gfx1100`). Hot shape: `M=17408`, `K=5120`, `N=2048`.

## PP512 CK Stability

Ten-run measurements with the first and last two samples trimmed show that the CK prefill path is stable, but only modestly faster at PP512.

| Setup | Native median (tok/s) | CK median (tok/s) | Delta |
|---|---:|---:|---:|
| Warmed | 822.15 | 839.50 | +2.11% |
| Forced workspace | 822.55 | 838.00 | +1.88% |

The earlier isolated PP512 burst was a warmup/outlier effect, not a reproducible CK gain.

## FP16 Hot-Cache Control

Weights were dequantized to FP16 once, activations were cast once, and both conversions were excluded from the timed region. Two HFQ4/Q8 MMQ projections took `12.9406 ms`; two rocBLAS FP16 projections took `64.0318 ms` (`0.2021x`). Correctness against the quantized path was `max_abs=0.02618`, `mean_abs=0.00583`.

Conclusion: a pre-dequantized FP16 hot cache is not a viable replacement for this production shape on gfx11.

## Split-Wave Gate/Up Fusion

Two kernels attempted to share one activation tile while routing independent wave subsets to gate and up weights.

| Topology | Separate launches (ms) | Fused (ms) | Speedup | Correctness |
|---|---:|---:|---:|---|
| X256, 32 gate + 32 up, 8 waves | 12.0154 | 12.5482 | 0.9575x | exact |
| X128, 64 gate + 64 up, 16 waves | 12.1137 | 12.7238 | 0.9520x | exact |

The second topology provides real activation-tile reuse, yet remains slower. Larger workgroups and reduced per-output wave organization cost more than the saved activation staging. The split-wave route is therefore removed rather than carried into production dispatch.

## Next Probe

Precomputing Q8 activation sums was also rejected. Against the current X256/Y64 `v_perm_b32` path, the hot-shape median changed from `6.1179 ms` to `6.1093 ms` (`1.0014x`) with exact output, while requiring a 320 KiB sidecar. Combined-zero has already reduced this work below a useful optimization threshold.

Changing the X256/Y64 launch-bound contract from two resident blocks to one was neutral/slightly negative: `5.9209 ms` versus `5.9317 ms` (`0.9982x`), with exact output over 20 alternating pairs. The compiler constraint is therefore retained.

An LDS-transposed, coalesced output epilogue was also rejected. Set mode measured `6.1158 ms` baseline versus `6.1724 ms` (`0.9908x`); the residual hot shape measured `6.5699 ms` versus `6.6012 ms` (`0.9953x`). Both were exact. The residual projection's profile share is therefore dominated by its MQ4 matmul, not its final read-modify-write pattern.

## A16 K32 Production Probe

A gfx11 `128M x 64N`, K32 W4A16 WMMA kernel was tested as a structural control against the Q8 MMQ path. The kernel uses 94 VGPR, 18 SGPR, 8704 bytes LDS, wave32, and no scratch. Standalone speedups over the current Q8 X256/Y64 path were `8.9%` for gate/up (`M=17408, K=5120, N=2048`), `13.8%` for FFN-down residual (`M=5120, K=17408`), and `20.0%` for the auxiliary residual (`M=5120, K=6144`).

The first model A/B used pointer-keyed FP16 activation caching and is invalid: the prefill runtime reuses activation buffer addresses across layers. The corrected route converts the current activation once per grouped projection and shares only that fresh conversion.

Five-pair fresh-process PP8192 A/B results with quantized CK attention active were:

| Candidate | Baseline median (tok/s) | Candidate median (tok/s) | Delta |
|---|---:|---:|---:|
| Residual only | 852.5 | 862.7 | +1.20% |
| Set family + residual | 851.1 | 876.5 | +2.98% |

The combined candidate also passed a real-prompt greedy check: for a 3361-token prompt, baseline and candidate emitted the same 16 token IDs. The candidate changed prefill from `665.7` to `683.3 tok/s` in that diagnostic run, while decode remained effectively unchanged.

Conclusion: A16 K32 is a valid opt-in production probe, but not the route to a 1k tok/s prefill target. Most standalone gain is lost to activation conversion and unaffected model work. The next structural kernel should preserve the existing Q8 activation representation while adopting cooperative `128M x 64N`, K32 decode/staging.
