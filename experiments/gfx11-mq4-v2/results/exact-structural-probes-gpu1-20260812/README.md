# gfx11 exact packed-MQ4 structural probes

Two exact-semantics standalone probes were run on GPU1, a Radeon Pro W7900
(`gfx1100`), after DPM warmup. Neither changes serving dispatch.

## Gate/up stream overlap

The existing `bench_hfq4_gate_up_streams` benchmark compared two production
group128 projections on one HIP stream with one projection on each of two HIP
streams. The full Qwen3.6-27B gate/up shape was `M=17408, K=5120, N=2048`.

| Path | Median | Relative | max_abs |
| --- | ---: | ---: | ---: |
| Serial production launches | 10.3952 ms | 1.0000x | - |
| Two HIP streams | 10.5532 ms | 0.9850x | 0 |

The same run also reconfirmed that a virtually concatenated gate/up weight
allocation was slower than two launches (`9.9946` versus `9.8306` ms). The
independent projections already occupy the device well enough that stream
concurrency does not reduce elapsed time.

Reproduction:

```bash
cargo build --release -p rdna-compute --example bench_hfq4_gate_up_streams
HIP_VISIBLE_DEVICES=1 target/release/examples/bench_hfq4_gate_up_streams \
  --m 17408 --k 5120 --n 2048 --pairs 10
```

## N2 fragment reuse plus quad-row weight staging

A temporary source composition combined the exact K-fragment N2-reuse loop
with the retained quad-row packed-weight loader. Its reference was the current
quad-row production kernel, not the older scalar-row loader.

| Shape | Operation | Reference | Candidate | Relative | max_abs |
| --- | --- | ---: | ---: | ---: | ---: |
| M17408 K5120 N2048 | gate/set | 4.5198 ms | 4.4680 ms | 1.0116x | 0 |
| M5120 K17408 N2048 | down/add | 4.5375 ms | 4.5497 ms | 0.9973x | 0 |

Weighting two gate/up projections and one down projection gives only about
`1.0068x` across the three large FFN projections. That is below the standalone
admission line and cannot produce a meaningful PP16384 gain, so the temporary
API and benchmark flag were removed after measurement.

## Decision

Close both paths. Stream overlap does not expose unused device capacity, and
combining two already-positive loader/fragment optimizations remains too small
to justify production routing or a PP16384 run.
