# Four-rows-per-wave packed-weight staging probe

Standalone W7900 (`gfx1100`, GPU1) exact-Q8 group128 experiment. The X256/Y64 row2/col4 compute topology is unchanged. Each wave is divided into four 8-lane subgroups; each subgroup stages one complete 256-value HFQ4 row using two sequential aligned `u32x2` transactions per lane. This reduces the Y64 loader loop from four iterations in the retained one-row-per-wave path to two.

| Projection | M | K | N | Operation | Baseline ms | Candidate ms | Speedup | max_abs |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| FFN gate/up | 17408 | 5120 | 2048 | set | 4.8083 | 4.4985 | 1.0689x | 0 |
| FFN down | 5120 | 17408 | 2048 | add | 4.9772 | 4.6697 | 1.0658x | 0 |
| GDN QKVZA | 10240 | 5120 | 2048 | set | 2.9102 | 2.7388 | 1.0626x | 0 |
| attention QKV | 12288 | 5120 | 2048 | set | 3.4448 | 3.2474 | 1.0608x | 0 |
| GDN output | 6144 | 5120 | 2048 | set | 1.7872 | 1.6852 | 1.0605x | 0 |
| auxiliary projection | 5120 | 6144 | 2048 | set | 1.8075 | 1.7180 | 1.0521x | 0 |

The full set/add kernels compile to wave32, `256 VGPR`, `27 SGPR`, 4 VGPR spills, and a 20-byte private segment. This is lower spill/private usage than the two-rows-per-wave candidate (`9` spills, `40` bytes), while improving every measured shape.

This remains standalone evidence. It requires a separate opt-in production route and PP8192 alternating-order A/B before promotion.
