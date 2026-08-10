# Planar Quad-Row MQ4 Weight Probe

This standalone gfx1100 experiment separates each HFQ4G256 weight group into a 128-byte aligned payload plane and an 8-byte FP32 metadata plane. The quad-row loader then replaces two sequential `uint2` payload reads per lane with one aligned `uint4` read. Arithmetic, Q8 activation data, LDS layout, WMMA, and output handling are unchanged.

## Gate/Up Result

Same-window runs on GPU1, `M=17408`, `K=5120`, `N=2048`, 21 alternating pairs after warmup:

| Variant | Reference ms | Candidate ms | Relative to reference | max_abs |
|---|---:|---:|---:|---:|
| Interleaved quad-row | 4.6858 | 4.4258 | 1.0587x | 0 |
| Planar quad-row | 4.7419 | 4.4434 | 1.0672x | 0 |

The planar candidate is 0.40% slower than the interleaved quad-row candidate by absolute median (`4.4434 / 4.4258`). Reference drift makes the relative speedup less suitable for comparing the two separate processes, but neither measure shows a planar-layout gain. The complete logs and generated table are reproducible with:

```bash
OUT=$PWD/experiments/gfx11-gate-up-x256y64/results/planar_quad_row_weight_gpu1_20260810_132656 \
GPU_ID=1 PAIRS=21 IDLE_SECS=5 \
./experiments/gfx11-gate-up-x256y64/run_group128_planar_quad_row_gate_ab.sh
```

## Resource Audit

Both full set/add kernels report:

- 256 VGPR
- 25 SGPR
- 4 VGPR spills
- 0 SGPR spills
- 20-byte private segment
- wave32

## Decision

Do not add a production sidecar or routing flag. Aligned planar payloads preserve exact output but do not improve the current quad-row kernel, so the 136-byte interleaved group stride is not the next dominant bottleneck.
