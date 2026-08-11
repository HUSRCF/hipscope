# PP16384 retained-path verification

This run verifies that the retained production path remains active after the
standalone group256 F16 probe was added and rejected for serving. The binary
was built with `deltanet,flash-attn-ck`; the log confirms that the staged
quantized CK sidecar loaded and executed.

## Configuration

- GPU: AMD Radeon Pro W7900 (`gfx1100`), GPU1
- ROCm: 7.14
- model: Qwen3.6-27B MQ4
- prefill: 16384 tokens, three passes, max batch 2048
- KV: asym3
- attention: staged quantized CK sidecar
- FFN: retained group128 quad-row F16 intermediate
- rejected group256 F16 gate/up serving route: not connected

## Result

| Pass | Wall (ms) | Throughput (tok/s) |
| ---: | ---: | ---: |
| 1 | 13899.0 | 1178.8 |
| 2 | 14143.4 | 1158.4 |
| 3 | 14326.4 | 1143.6 |

The three-pass median is `1158.4 tok/s`; decode was `31.8 tok/s`. The benchmark
currently writes the final pass rather than the median into `PREFILL_SUMMARY`,
so the raw per-pass lines are authoritative for this report.
