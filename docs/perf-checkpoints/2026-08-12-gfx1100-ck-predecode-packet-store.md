# gfx1100 CK predecode packet-store checkpoint

Date: 2026-08-12
Hardware: AMD Radeon Pro W7900, gfx1100
Model: Qwen3.6-27B MQ4, asym3 KV

## Verdict

**Keep the packet-store implementation as an experimental sidecar probe; do
not enable it in the production route.** It is bit-exact and materially faster
in the isolated predecode stage, but the required PP16384 process-paired gate
is neutral to slightly negative.

## Isolated predecode

The packet kernel replaces ten scattered stores per wave with two aligned
128-bit global stores. K and V packet outputs are bit-identical to the scalar
implementation for every tested length.

| sequence length | scalar ms | packet ms | packet/scalar |
|---:|---:|---:|---:|
| 2,048 | 0.075281 | 0.013104 | 0.1741x |
| 4,096 | 0.176836 | 0.030636 | 0.1732x |
| 6,144 | 0.299002 | 0.047668 | 0.1594x |
| 8,192 | 0.395967 | 0.063663 | 0.1608x |

The emitted gfx1100 kernel uses 25 VGPRs, zero scratch, and two
`global_store_b128` instructions. The scalar kernel uses 16 VGPRs and zero
scratch.

## Full staged CK API

Five fresh-process measurements at `Q=2048` retained a smaller component-level
gain after including the CK attention stage.

| K length | scalar ms | packet ms | packet/scalar |
|---:|---:|---:|---:|
| 2,048 | 1.393917 | 1.298316 | 0.9318x |
| 4,096 | 3.347028 | 3.162306 | 0.9451x |
| 6,144 | 5.427256 | 5.116862 | 0.9394x |
| 8,192 | 7.788605 | 7.440608 | 0.9425x |

The original staged sweep always ran scalar before packet. The harness now
alternates arm order; a two-run harness check retained the same directional
signal. This checkpoint does not use the original fixed-order sweep as a
production claim.

## PP16384 production gate

Three independent process pairs used alternating AB/BA order, three prefill
runs per process, 20-second idle gaps, the same model and binary, and exact
greedy token comparison.

| pair | scalar tok/s | packet tok/s | packet/scalar |
|---:|---:|---:|---:|
| 1 | 1,155.6 | 1,133.1 | 0.9805x |
| 2 | 1,131.8 | 1,127.1 | 0.9958x |
| 3 | 1,126.4 | 1,133.2 | 1.0060x |

- Scalar process median: `1131.8 tok/s`
- Packet process median: `1133.1 tok/s`
- Median of paired ratios: `0.9958x` (`-0.42%`)
- Positive pairs: `1/3`
- Generated token IDs: exact match in all six processes

The process-median result supersedes an earlier summary that accidentally read
the final prefill run's `SUMMARY` field. The benchmark source assigns
`prefill_ms` from `prefill_samples_ms.last()`, so that field is not a multi-run
aggregate. The harness now records both metrics under explicit column names,
uses the printed per-process median for the gate, and records sidecar build
provenance.

## Decision boundary

The predecode store path is too small a fraction of PP16384 wall time for its
isolated 5.7-6.2x gain to move application throughput reliably. Further CK
bridge micro-optimization is below the current admission threshold. Production
work remains focused on changes that affect the packed-MQ4 family, which owns
most of the long-prefill wall time.

## Artifacts

- `experiments/flash-attn-ck-sidecar/quantized/results/predecode_packet_store_gpu1_20260812_100000/`
- `experiments/flash-attn-ck-sidecar/quantized/results/staged_packet_store_gpu1_20260812_103000/`
- `experiments/gfx11-gate-up-x256y64/results/pp16384_ck_packet_store_gpu1_20260812_110000/`
