# Sequence-wide packed-MQ4 projection probe

This exact-semantics standalone probe compares eight consecutive `N=2048` launches with one `N=16384` launch of the retained gfx1100 X256/Y64 group128 quad-row Wave32-WMMA primitive. Each small launch has independent Q8 input and output storage. The large input repeats the same 2048-row activation block eight times, and fresh output buffers verify every repeated block against the small launch.

## Results

AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14, nine alternating timing pairs:

| Shape | Operation | 8 x N2048 | 1 x N16384 | Speedup | max_abs |
| --- | --- | ---: | ---: | ---: | ---: |
| M17408 K5120 | set | 37.9871 ms | 37.4359 ms | 1.0147x | 0 |
| M5120 K17408 | residual-add | 38.1911 ms | 36.6819 ms | 1.0411x | 0 |

The sequence-wide launch is only 1-4% faster locally. Using the measured PP16384 wall shares, the estimated removable wall fraction is `36.03% * (1 - 1/1.014726) + 18.47% * (1 - 1/1.041142) = 1.25%`, or about `1.013x` projected end-to-end. This is below the backend admission line, so the result does not justify a layer-major serving rewrite. It also confirms that kernel-launch amortization and cross-chunk weight residency are not the missing path from roughly 1.15k to 1.5k tok/s.

## Reproduction

```bash
cargo build --release --locked -p rdna-compute --example bench_hfq4_sequence_wide

HIP_VISIBLE_DEVICES=1 target/release/examples/bench_hfq4_sequence_wide \
  --m 17408 --k 5120 --small-n 2048 --chunks 8 --pairs 9

HIP_VISIBLE_DEVICES=1 target/release/examples/bench_hfq4_sequence_wide \
  --m 5120 --k 17408 --small-n 2048 --chunks 8 --pairs 9 --residual
```

This is a standalone execution-structure experiment. It does not alter production dispatch.
