# gfx11 A16 K32 compact perm-decode probe

Standalone-only comparison on Radeon Pro W7900 (`gfx1100`, GPU1) at the
Qwen3.6-27B gate/up shape `M=17408, K=5120, N=2048`. Both paths use the same
K32 staging, four-wave compact decode mapping, FP32 activation input, FP16
WMMA operands, FP32 accumulation, launch geometry, and output path. The only
changed variable is scalar nibble extraction versus gfx11 byte permutes plus
vector integer-to-FP16 conversion.

| Run | Pairs | Scalar compact | Perm compact | Relative | max_abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 25 | 5.6417 ms | 5.4754 ms | 1.0304x | 0 |
| Longer validation | 75 | 6.3950 ms | 6.1748 ms | 1.0357x | 0 |

The absolute times drifted together in the longer run, while the interleaved
relative result stayed near 3.5%. This is an exact, small positive primitive,
not a production route: the A16 kernel remains slower than the retained Q8
group128 MMQ path at this shape.

ISA/resource audit:

| Kernel | VGPR | spills | private bytes/thread | `v_perm_b32` | `v_bfe_u32` | `v_cvt_f16*` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar compact | 94 | 0 | 0 | 0 | 25 | 68 |
| Perm compact | 95 | 0 | 0 | 8 | 1 | 68 |

The compiler therefore preserves the intended byte-permute reduction without
introducing scratch traffic. Keep this helper available for later packed-MQ4
experiments, but do not route the current A16 kernel into serving.

Reproduction:

```bash
cargo build --release -p rdna-compute --example bench_hfq4_group256_direct
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 75 \
  --f32a-k32-compact-perm-decode
```
