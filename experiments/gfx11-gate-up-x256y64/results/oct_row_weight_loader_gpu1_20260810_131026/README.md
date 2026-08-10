# Eight-rows-per-wave packed-weight staging probe

Standalone W7900 (`gfx1100`, GPU1) exact-Q8 group128 experiment. The X256/Y64 row2/col4 compute topology is unchanged. Each wave is divided into eight four-lane subgroups; each subgroup stages one complete 256-value HFQ4 row using four sequential aligned `u32x2` transactions per lane. This reduces the Y64 loader loop to one iteration.

| Projection | M | K | N | Operation | Baseline ms | Candidate ms | Speedup | max_abs |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| FFN gate/up | 17408 | 5120 | 2048 | set | 4.9415 | 4.6532 | 1.0620x | 0 |
| GDN QKVZA | 10240 | 5120 | 2048 | set | 2.9001 | 2.7530 | 1.0534x | 0 |
| GDN output | 6144 | 5120 | 2048 | set | 1.8037 | 1.7033 | 1.0589x | 0 |
| attention QKV | 12288 | 5120 | 2048 | set | 3.5317 | 3.3202 | 1.0637x | 0 |
| FFN down | 5120 | 17408 | 2048 | add | 5.0477 | 4.7798 | 1.0560x | 0 |
| auxiliary projection | 5120 | 6144 | 2048 | add | 1.8214 | 1.7352 | 1.0497x | 0 |

The full set/add kernels compile to wave32, `256 VGPR`, `27 SGPR`, 2 VGPR spills, and a 12-byte private segment. Despite reducing spills relative to the quad-row candidate, every oct-row candidate time is slower than the corresponding quad-row result. Four rows per wave remains the best measured staging granularity; this oct-row path stays standalone and is not routed into production.

Reproduction:

```bash
cargo build --release -p rdna-compute --example bench_hfq4_group256_direct
GPU_ID=1 PAIRS=15 \
  experiments/gfx11-gate-up-x256y64/run_group128_oct_row_hot_shapes.sh
```
