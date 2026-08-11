# Group256 staged-activation full-shape probe on gfx1100

This standalone probe completes the previously short-shape-only evaluation of
the group256 activation staging path. It compares the staged candidate with
both the group128 reference and the retained group256 serial-row backend at the
two dominant PP2048 projection shapes. No serving dispatch was changed.

Device: AMD Radeon Pro W7900 (`gfx1100`), GPU1, ROCm 7.14. Each value is the
median of 21 in-process alternating pairs after three kernel warmups and a
five-second DPM warmup.

| Shape | Mode | Group128 (ms) | Candidate (ms) | Relative to group128 | Correctness max abs |
| --- | --- | ---: | ---: | ---: | ---: |
| gate/set, `M17408 K5120 N2048` | staged | 4.6090 | 5.4335 | 0.8483x | 1.431e-6 |
| gate/set, `M17408 K5120 N2048` | serial-row | 4.7546 | 4.1019 | 1.1591x | 1.431e-6 |
| down/add, `M5120 K17408 N2048` | staged | 4.7154 | 5.9351 | 0.7945x | 5.722e-6 |
| down/add, `M5120 K17408 N2048` | serial-row | 4.8587 | 4.1755 | 1.1636x | 6.199e-6 |

Dividing the separately warmed process medians, staged activation is
approximately 0.755x the serial-row gate/set throughput and 0.704x the
serial-row down/add throughput.
The earlier short-shape regression therefore persists and grows at the real
hot shapes. Cooperative global-to-LDS activation staging adds synchronization
and LDS traffic without beating the cache-served serial-row global loads. The
staged route is closed; retain group256 serial-row.

## Reproduction

Run each command in a fresh process, with a five-second idle interval between
commands:

```bash
HIP_VISIBLE_DEVICES=1 HIPFIRE_DPM_WARMUP_SECS=5 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 21 --staged
HIP_VISIBLE_DEVICES=1 HIPFIRE_DPM_WARMUP_SECS=5 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 21 --serial-row
HIP_VISIBLE_DEVICES=1 HIPFIRE_DPM_WARMUP_SECS=5 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 5120 --k 17408 --n 2048 --pairs 21 --staged --add
HIP_VISIBLE_DEVICES=1 HIPFIRE_DPM_WARMUP_SECS=5 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 5120 --k 17408 --n 2048 --pairs 21 --serial-row --add
```
