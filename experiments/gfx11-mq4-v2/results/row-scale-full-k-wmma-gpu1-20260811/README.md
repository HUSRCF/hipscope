# Row-scale full-K Wave32-WMMA execution-format screen

This standalone screen changes both the resident weight representation and
the accumulation contract. It is not routed into serving.

- `row-I8`: one signed-I8 weight per element and one FP32 scale per output
  row.
- `row-Q4`: two signed-Q4 weights per byte and one FP32 scale per output row.
- Both paths quantize each activation row to signed I8 with one FP32 scale,
  keep the integer Wave32-WMMA accumulators live across the complete K range,
  and apply the row scales only in the epilogue.

The retained reference is the production group128 quad-row packed-MQ4 path.
The test runs both production FFN shapes in one process with alternating
launch order after kernel warmup and a five-second DPM warmup.
Both paths start from the same synthetic FP32 activation, but each uses the
activation quantizer defined by its own execution contract. The reported
steady-state GEMM timing excludes both activation quantizers and the load-time
weight repack; quality metrics compare the resulting end-to-end projections.

```text
GPU: AMD Radeon Pro W7900 / gfx1100
HIP: 7.14
N: 2048
pairs: 3

format  shape                 retained   candidate   speedup  exec/MQ4 bytes  relative L2  cosine
row-I8  gate M17408 K5120     4.1796 ms  10.3588 ms   0.4035x       1.8838x      0.06162    0.998917
row-I8  down M5120 K17408     4.3453 ms  10.4900 ms   0.4142x       1.8828x      0.06152    0.998777
row-Q4  gate M17408 K5120     4.2418 ms   8.8346 ms   0.4801x       0.9426x      0.13559    0.991018
row-Q4  down M5120 K17408     4.2675 ms   9.8228 ms   0.4344x       0.9416x      0.13752    0.990611
```

Code-object inspection found wave32 throughout. Row-I8 set/add used 93/91
VGPRs, 24 SGPRs, zero spills, and no private segment. Row-Q4 set/add used 93
VGPRs, 22 SGPRs, zero spills, and no private segment. Both launch with 40 KiB
of dynamic LDS per workgroup.

An earlier independent process put the four speedups in the same grossly
negative `0.391x-0.487x` band. The row-Q4 sibling removes the row-I8 path's
nearly 2x resident-weight cost, but remains less than half as fast as retained
MQ4 and has materially worse
synthetic output error. The failure is therefore not explained by spilling or
the expanded-I8 resident bytes alone. The full-K accumulator topology, K128
LDS staging, and its operand-feeding schedule are jointly noncompetitive in
this implementation.

Decision: reject both formats before serving integration or checkpoint
conversion. This closes this specific row-scale/full-K architecture; it does
not reject Wave32 WMMA or every possible gfx11-native execution format.

Reproduce after a clean kernel warmup:

```bash
HIP_VISIBLE_DEVICES=1 \
  ./target/release/examples/bench_mq4v2_row_i8_q8 --n 2048 --pairs 3 --row-i8

HIP_VISIBLE_DEVICES=1 \
  ./target/release/examples/bench_mq4v2_row_i8_q8 --n 2048 --pairs 3
```

The exact console outputs used by this table are in `row_i8.txt` and
`row_q4.txt`.
