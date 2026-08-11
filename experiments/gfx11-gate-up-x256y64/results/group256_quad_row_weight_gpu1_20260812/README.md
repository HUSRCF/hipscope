# Group256 quad-row weight-loader probe

This standalone probe asks whether the quad-row weight staging retained by
the group128 production path also helps the group256 serial-row path. It does
not change production dispatch or model arithmetic.

## Environment

- GPU: AMD Radeon Pro W7900, gfx1100, `HIP_VISIBLE_DEVICES=1`
- ROCm: 7.14 runtime/toolchain
- Workgroup: 256 threads, Wave32
- Dynamic LDS: 19,456 bytes for both paths
- Timing: 15 alternating same-process pairs after warmup

## Results

| Shape | Operation | Baseline ms | Quad-row ms | Speedup (baseline / candidate) | Correctness |
|---|---|---:|---:|---:|---|
| M17408 K5120 N2048 | gate/set | 4.1564 | 4.3074 | 0.9649x | bit-exact |
| M5120 K17408 N2048 | down/add | 4.2456 | 4.3618 | 0.9734x | bit-exact |

For both shapes, `max_abs=0`, `mean_abs=0`, `rel_l2=0`, and `cosine=1`.

## Static resource audit

| Kernel | VGPR | SGPR | Spill/private | Static instructions | Global loads | LDS stores | Permute-like |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline full-set | 228 | 32 | 0 | 1043 | 75 | 4 | 4 |
| quad-row full-set | 244 | 27 | 0 | 1188 | 75 | 10 | 16 |
| baseline full-add | 228 | 32 | 0 | 1180 | 139 | 4 | 4 |
| quad-row full-add | 244 | 27 | 0 | 1330 | 139 | 10 | 16 |

The loader changes neither global-load count nor dynamic LDS. It increases
VGPR use and adds LDS-store and permutation work. The measured regression is
therefore consistent with extra loader/reordering work rather than a spill or
workspace-capacity failure.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 15 \
  --serial-row-quad-weight

HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 5120 --k 17408 --n 2048 --pairs 15 \
  --serial-row-quad-weight --add
```

## Decision

Closed. Keep the quad-row loader in the retained group128 path, but do not
promote it into group256 serial-row serving dispatch.
