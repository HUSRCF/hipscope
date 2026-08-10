# PP8192 quad-row production profile

W7900 (`gfx1100`), GPU1, Qwen3.6-27B MQ4, Asym3 KV, 2048-token prefill chunks, quantized CK attention sidecar active, and the opt-in quad-row group128 route. The profile was collected after JIT warmup with `HIPFIRE_PROFILE=1`.

| Stage | Calls | GPU ms | Serialized share |
|---|---:|---:|---:|
| FFN gate/up MQ4 set, M17408 K5120 | 512 | 2412.8 | 37.5% |
| FFN down MQ4 residual-add, M5120 K17408 | 256 | 1248.2 | 19.4% |
| GDN QKVZA MQ4 set, M10240 K5120 | 192 | 534.4 | 8.3% |
| auxiliary MQ4 residual-add, M5120 K6144 | 256 | 443.0 | 6.9% |
| GDN output MQ4 set, M6144 K5120 | 192 | 320.8 | 5.0% |
| attention QKV MQ4 set, M12288 K5120 | 64 | 214.5 | 3.3% |
| all profiled MQ4 GEMM | 1472 | 5173.7 | 80.39% |
| all serialized kernels | 4300 | 6435.6 | 100% |

Application prefill was `1097.8 tok/s` (`7462.2 ms` wall); decode was `33.1 tok/s`. GPU/application wall differs from serialized kernel time by `1026.6 ms`, so percentages above use the profiler's serialized denominator and should not be mixed with a different run's absolute throughput.

The current path remains strongly MQ4-GEMM bound. Moving from roughly `1101` to `1500 tok/s` would require about `1.49x` acceleration of the measured MQ4 component under a simple Amdahl model. Loader-only changes are therefore insufficient; the next experiment must change packed-weight/activation reuse or the residual/output dataflow.

Reproduction:

```bash
cargo build --release -p hipfire-runtime --example bench_qwen35_mq4 --features flash-attn-ck
GPU_ID=1 experiments/gfx11-gate-up-x256y64/run_pp8192_group128_quad_row_profile.sh
```
