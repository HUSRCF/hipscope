# gfx1100 coarse-scale / int32-accum probe

Standalone-only MQ4-v2 feasibility probe for `gfx1100`. This directory does not
touch serving or production code.

## Method

- Reference kernel: exact affine group128 WMMA path.
- Candidate kernel: synthetic coarse-scale upper bound that shares one weight
  scale/zero and one activation scale across `4 x 128 = 512` adjacent K values,
  accumulates all WMMA `i32` partials for that coarse block, then performs one
  FP32 dequant / zero-correction step.
- Controlled validation case: `M=16 K=512 N=16`, exact/coarse outputs matched
  exactly (`max_abs=0`, `relative_l2=0`, `cosine=1`).

## Command

Build:

```bash
bash run_probe.sh build
```

Measured GPU1 run:

```bash
env LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib \
  HIP_VISIBLE_DEVICES=1 \
  ./bench_mq4v2_coarse_scale_int32_accum \
  --warmup 3 --pairs 7 --dpm-warmup-ms 5000
```

## Results

Two independent GPU1 runs, each with 7 timing pairs, 3 warmups, and a 5 s DPM
warmup:

| Run | Shape | Exact median | Coarse median | Speedup | Admission |
|---|---|---:|---:|---:|---|
| A | gate/up `M17408 K5120 N2048` | 23.610540 ms | 23.242508 ms | `1.0158x` | Reject |
| A | down `M5120 K17408 N2048` | 29.328993 ms | 27.199036 ms | `1.0783x` | Reject |
| B | gate/up `M17408 K5120 N2048` | 23.693686 ms | 22.907539 ms | `1.0343x` | Reject |
| B | down `M5120 K17408 N2048` | 29.421831 ms | 27.592184 ms | `1.0663x` | Reject |

`results.txt` records run B.

Decision: reject. Both repetitions and both full-shape results are below the
strict `>= 1.30x` promotion line.

## Resource metadata

All four kernels (`exact/coarse x set/add`) match:

```text
wave32
vgpr_count: 73
vgpr_spill_count: 0
dynamic LDS: 0 B
private_segment_fixed_size: 0 B/work-item
```

`bash run_probe.sh build` regenerates the unbundled gfx1100 code-object metadata
as the ignored local artifact `kernel_notes.txt`.

## Limitations

- Candidate uses synthetic shared coarse scales; it is not a production weight
  or activation contract.
- This is not a quality validation run.
- The baseline is the matched standalone exact kernel in this directory, not the
  shipped production kernel or serving route.
