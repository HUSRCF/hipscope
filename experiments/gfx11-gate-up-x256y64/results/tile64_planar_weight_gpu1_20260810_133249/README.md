# M64 Tile-Major Planar MQ4 Weight Probe

This standalone gfx1100 experiment changes only the offline weight sidecar layout. Within each 64-row production output tile, payload and FP32 metadata are ordered by K group and then row. Consequently, the four 128-byte rows loaded by one quad-row wave form one contiguous 512-byte region. Q8 activation data, unpack arithmetic, LDS layout, WMMA, and output handling are unchanged.

## Gate/Up Result

GPU1, `M=17408`, `K=5120`, `N=2048`, 21 alternating pairs after warmup:

| Variant | Reference ms | Candidate ms | Relative to reference | max_abs |
|---|---:|---:|---:|---:|
| Interleaved quad-row | 4.6551 | 4.4185 | 1.0536x | 0 |
| M64 tile-major planar quad | 4.6993 | 4.4205 | 1.0631x | 0 |

The tile-major candidate is 0.05% slower by absolute median (`4.4205 / 4.4185`). The result is performance-neutral within noise and preserves exact output.

## Resource Audit

Both full set/add kernels report 256 VGPR, 26 SGPR, 4 VGPR spills, a 20-byte private segment, and wave32.

## Decision

Do not add a production sidecar. Making the four rows consumed by a wave contiguous does not improve gate/up time, so global MQ4 payload spatial layout is not the next limiting stage in this kernel.
