# Exact FFN target-width boundary on GPU1

```text
GPU: AMD Radeon Pro W7900 / gfx1100, HIP 7.14
N: 2048
pairs: 11
matching full width: gate 4.5577 ms, down 4.6500 ms
```

Commands used the retained `x256_y64_group128` result printed by
`bench_hfq4_group128_packed_weight_y128`:

```bash
HIP_VISIBLE_DEVICES=1 \
  ./target/release/examples/bench_hfq4_group128_packed_weight_y128 \
  --m WIDTH --k 5120 --n 2048 --pairs 11

HIP_VISIBLE_DEVICES=1 \
  ./target/release/examples/bench_hfq4_group128_packed_weight_y128 \
  --m 5120 --k WIDTH --n 2048 --pairs 11
```

Raw retained-primitive medians:

```text
width=10496 gate=2.7452 ms down=2.7975 ms
width= 9984 gate=2.5785 ms down=2.6680 ms
width= 9728 gate=2.5665 ms down=2.6035 ms
```

Derived values:

```text
weighted_speedup = (2 * 4.5577 + 4.6500) / (2 * gate + down)
overall_speedup  = 1 / (0.51 + 0.49 / weighted_speedup)

width  weighted  overall   from 1189   from 1115.4
10496  1.660903  1.242205  1476.98     1385.56 tok/s
 9984  1.759157  1.268162  1507.85     1414.51 tok/s
 9728  1.779280  1.273249  1513.89     1420.18 tok/s
```

This is a kernel-work boundary, not evidence that a narrowed checkpoint is
accurate or that an end-to-end run reaches the projection. Existing oracle and
static-pruning quality results reject transparent width reduction.
