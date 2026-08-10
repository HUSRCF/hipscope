# gfx11 X64/Y64 packed-MQ4 occupancy probe

This standalone experiment tests whether reducing the retained X256/Y64
kernel's accumulator and LDS footprint can recover enough occupancy to improve
the two production FFN shapes. It does not participate in serving dispatch.

The candidate uses an X64/Y64 tile, eight Wave32 warps, group128 Q8
activations, and the retained affine MQ4 quad-row weight contract. Relative to
the retained X256/Y64 reference, it reduces the per-thread accumulator from 64
to 16 floats and dynamic LDS to 28,928 bytes, while causing four times as many
workgroups to load each weight tile.

Controlled GPU1 result:

```text
GPU: AMD Radeon Pro W7900 / gfx1100, HIP 7.14
N: 2048
pairs: 9, alternating launch order after kernel and DPM warmup

shape                         reference    X64/Y64    speedup    max_abs
gate/up M=17408 K=5120 set     4.8368 ms   5.9503 ms   0.8129x          0
down M=5120 K=17408 add        4.8166 ms   5.8293 ms   0.8263x          0
```

Code-object audit for both full-set and full-add candidates:

```text
wavefront_size:       32
vgpr_count:           184
vgpr_spill_count:     0
private_segment:      0 bytes/work-item
sgpr_count:           27
dynamic_lds:          28,928 bytes/workgroup
```

The resource objective was met, but both production shapes are 17-19% slower.
The reduced accumulator and two-workgroup residency do not repay the 4x
workgroup-level weight-load duplication. This rejects this X64/Y64 execution
geometry, not Wave32 WMMA or a future execution format with cross-workgroup
weight reuse.

Reproduce:

```bash
cargo build --release -p rdna-compute --example bench_hfq4_x64y64
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_x64y64 --n 2048 --pairs 9
```

Raw output is retained under
`results/x64y64-occupancy-gpu1-20260811/bench.log`.
