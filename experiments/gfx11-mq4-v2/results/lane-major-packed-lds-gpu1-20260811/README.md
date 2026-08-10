# Lane-major packed-LDS MQ4-v2 admission result

Date: 2026-08-11. Device: AMD Radeon Pro W7900 (`gfx1100`), selected with
`HIP_VISIBLE_DEVICES=1`; ROCm runtime 7.14. Both GPUs were idle and no KFD
processes were present before the run.

## Candidate

This standalone probe preserves the exact affine HFQ4-G256 numerical contract
and resident byte count. It only replaces the execution layout:

```text
source group/row:  scale f32, zero f32, 128 packed payload bytes
execution tile16: scale[16], zero[16], payload[subK32=8][word=4][row=16]
```

The execution copy remains 136 bytes per row/group, so a replacement resident
layout has zero capacity overhead. The kernel stages the packed tile once in
LDS for cross-wave reuse, then expands each K32 fragment in registers before
Wave32 IU8 WMMA. It is not connected to serving dispatch.

## Command

```bash
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_mq4v2_lane_major \
  --n 2048 --pairs 7
```

The benchmark uses alternating reference/candidate order after three warmups
and a five-second DPM warmup. The retained reference is the exact group128
quad-row X256/Y64 kernel.

## Results

| Shape | Reference median | Candidate median | Relative | max_abs | Admission |
|---|---:|---:|---:|---:|---|
| gate/set, M17408 K5120 N2048 | 4.2752 ms | 9.5363 ms | 0.4483x | 0 | Reject |
| down/add, M5120 K17408 N2048 | 4.3916 ms | 10.7522 ms | 0.4084x | 0 | Reject |

Raw milliseconds:

```text
gate reference: [4.275217, 4.173072, 4.366551, 4.250831, 4.382622, 4.246713, 4.383283]
gate candidate: [9.536286, 9.117747, 9.760414, 9.394776, 9.824937, 9.528210, 9.832030]
down reference: [4.391590, 4.273705, 4.454851, 4.288733, 4.556034, 4.246002, 4.564230]
down candidate: [10.391258, 10.667115, 11.079101, 10.752237, 11.022744, 10.594847, 11.091114]
```

## Resource audit

Both set/add code objects report:

```text
wavefront_size:                  32
max_flat_workgroup_size:        256
vgpr_count:                     217
sgpr_count:                      26
vgpr_spill_count:                 0
sgpr_spill_count:                 0
private_segment_fixed_size:       0 B/work-item
dynamic LDS:                 46,336 B/workgroup
```

## Decision

Close this packed-LDS/register-decode architecture. The lane-major layout
removes the old row-major staging/access pattern and the code object has no
spill failure, yet both production FFN shapes remain more than 2x slower than
the retained expanded-IU8 LDS feed. A future MQ4-v2 backend must change the
decode/scale-to-WMMA contract rather than only reorder the same packed nibbles.
