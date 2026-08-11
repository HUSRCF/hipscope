# Group128 direct activation plus quad-row weight loader

This standalone probe removes activation LDS staging while retaining the
quad-row weight loader used by the current production path. It isolates the
loader confounder in the earlier direct-activation production A/B. No serving
dispatch or numerical contract is changed.

## Environment

- GPU: AMD Radeon Pro W7900, gfx1100, `HIP_VISIBLE_DEVICES=1`
- ROCm: 7.14 runtime/toolchain
- Workgroup: 256 threads, Wave32
- Timing: 15 alternating same-process pairs after warmup

## Results

| Shape | Operation | Quad-row LDS ms | Direct + quad ms | Speedup (baseline / candidate) | Max abs | Relative L2 |
|---|---|---:|---:|---:|---:|---:|
| M17408 K5120 N2048 | gate/set | 4.5676 | 4.6138 | 0.9900x | 1.43e-6 | 1.57e-7 |
| M5120 K17408 N2048 | down/add | 4.6527 | 4.7807 | 0.9732x | 5.72e-6 | 3.20e-7 |

Both comparisons have `cosine=1.0`. The differences are consistent with the
known FP32 accumulation-order difference of the direct path, not an address
or quantization-contract error.

## Resource audit

| Resource | Quad-row LDS | Direct + quad |
|---|---:|---:|
| VGPR | 256 | 163 |
| SGPR | 27 | 34 |
| VGPR spills | 4 | 0 |
| Private bytes/thread | 20 | 0 |
| Dynamic LDS/workgroup | 57,344 B | 19,456 B |
| Wave size | 32 | 32 |

The candidate materially reduces LDS and register pressure but remains slower
on both model shapes. The retained activation LDS tile therefore provides
useful reuse that is not recovered by the lower-resource direct path.

## Reproduce

```bash
HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 17408 --k 5120 --n 2048 --pairs 15 \
  --group128-direct-quad-weight

HIP_VISIBLE_DEVICES=1 \
  target/release/examples/bench_hfq4_group256_direct \
  --m 5120 --k 17408 --n 2048 --pairs 15 \
  --group128-direct-quad-weight --add
```

## Decision

Closed. The candidate is far below the 1.30x primitive admission line, so it
must not enter serving or trigger a PP16384 production run.
