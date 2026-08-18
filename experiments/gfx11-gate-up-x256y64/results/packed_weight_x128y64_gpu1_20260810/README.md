# X128/Y64 packed-weight residency probe on gfx1100

Standalone-only Radeon Pro W7900 (`gfx1100`, GPU1) experiment at the full
Qwen3.6-27B gate/up shape `M=17408, K=5120, N=2048`, set output, with 25
alternating timing pairs.

The candidate keeps HFQ4 payloads packed in LDS, narrows the activation tile
from X256 to X128, and retains the Y64 row2/col4 output mapping. Its dynamic LDS
footprint is 26,880 bytes, allowing a two-workgroup-per-CU launch bound. This
isolates whether higher residency can offset packed-nibble expansion and the
duplicated weight work caused by halving X.

| Path | Median | Relative | Max abs | Mean abs |
| --- | ---: | ---: | ---: | ---: |
| Production X256/Y64 expanded-i8 LDS | 4.5461 ms | 1.0000x | - | - |
| X128/Y64 packed-weight LDS | 8.1144 ms | 0.5603x | 0 | 0 |

The candidate compiles to wave32 with 197 VGPR, 28 SGPR, zero spills, and zero
fixed private storage (see `resources.txt`). The regression is therefore not a scratch/resource
failure. Halving X duplicates weight staging and grid-level work, while each
WMMA fragment still pays register-side nibble expansion. Those costs dominate
the additional residency. Together with the earlier X256/Y64 packed-weight
result, this closes the compressed-weight-LDS occupancy line for the current
HFQ4/Q8 dataflow; do not route this path into serving.

Reproduction:

```bash
cargo build --release -p rdna-compute --example bench_hfq4_group256_direct
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 25 --packed-weight-x128y64
```
