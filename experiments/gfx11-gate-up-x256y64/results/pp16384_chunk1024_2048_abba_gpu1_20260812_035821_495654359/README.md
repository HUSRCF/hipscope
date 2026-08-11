# PP16384 prefill-chunk ABBA

This run compares 1024-token and 2048-token prefill chunks on the retained
Qwen3.6-27B MQ4 W7900 production path. Each point contains two prefill runs.
The outer order is 2048, 1024, 1024, 2048, with 60 seconds of idle time between
points to limit order and thermal bias.

## Result

| Order | Chunk | Median prefill | Median wall |
|---:|---:|---:|---:|
| 1 | 2048 | 1183.4 tok/s | 13845.1 ms |
| 2 | 1024 | 1057.9 tok/s | 15487.7 ms |
| 3 | 1024 | 1048.1 tok/s | 15632.5 ms |
| 4 | 2048 | 1146.2 tok/s | 14294.6 ms |

| Chunk | Mean of pair medians | Relative throughput |
|---:|---:|---:|
| 1024 | 1053.0 tok/s | `0.9040x` |
| 2048 | 1164.8 tok/s | `1.0000x` |

Chunk 1024 is 9.60% slower. Both bookending 2048 samples exceed both middle
1024 samples despite the run-to-run thermal decline, so the path retains chunk
2048. Decode remains effectively unchanged at 31.8-32.1 tok/s.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=1 GPU_ID=1 PAUSE_SECS=60 \
  ./experiments/gfx11-gate-up-x256y64/run_pp16384_chunk1024_2048_abba.sh
```

The exact child manifests, artifact hashes, logs, and parsed tables are kept in
this directory. The failed non-device sandbox attempt is not part of this
record.
