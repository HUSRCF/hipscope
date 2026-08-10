# gfx11 X256 dual payload-only gate/up negative result

Standalone-only Radeon Pro W7900 (`gfx1100`, GPU1) experiment at the full
Qwen3.6-27B gate/up shape `M=17408, K=5120, N=2048`.

The candidate uses one 512-thread workgroup for both output planes. Each plane
retains the production `X256/Y64`, eight-wave, row2/col4 compute topology. The
two planes share a 32-KiB Q8 activation payload; two expanded HFQ4 payloads use
the other 32 KiB. Both payloads use an XOR-swizzled eight-int fragment layout,
while scale/sum/zero metadata is read from the original buffers. This tests
whether activation reuse can justify a dual-output kernel without the packed
weight decode that caused the earlier 208-spill failure.

| Path | Median | Relative | Gate max_abs | Up max_abs |
| --- | ---: | ---: | ---: | ---: |
| Two production group128 launches | 9.1786 ms | 1.0000x | - | - |
| Dual payload-only | 16.9007 ms | 0.5431x | 0 | 0 |

The first diagnostic build used FP32 weight metadata and showed approximately
`7.8e-4` max absolute difference. Matching the production loader's FP16
metadata narrowing made both outputs bit-exact and did not change the
regression.

Resource audit for the candidate entrypoint: wave32, 256 VGPR, 40 SGPR, 18
VGPR spills, 76 private bytes/thread, plus exactly 65,536 bytes dynamic LDS.
The compact LDS layout therefore avoids the earlier packed-unpack explosion,
but the combined 512-thread/64-KiB residency and remaining metadata/live-range
pressure still make it much slower than two independent production kernels.
Do not route this path into serving, and stop the dual-output activation-share
line unless the single-output accumulator footprint is first reduced.

Reproduction:

```bash
cargo build --release -p rdna-compute --example bench_hfq4_dual_payload
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_dual_payload \
  --m 17408 --k 5120 --n 2048 --pairs 3
```
