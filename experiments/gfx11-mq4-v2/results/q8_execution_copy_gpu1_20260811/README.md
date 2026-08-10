# Q8_0 high-memory execution-copy probe

This standalone probe asks whether hipfire's existing gfx1100 Q8_0 Wave32
WMMA primitives are fast enough to justify a 2x resident-weight execution copy.
It does not add a model conversion or serving route.

Hardware: AMD Radeon Pro W7900 Dual Slot, gfx1100, HIP 7.14. GPU1 was idle and
had zero allocated VRAM before the serial runs. Each result is a ten-pair
alternating median; the two shapes were separated by 20 seconds. Activation
preprocessing and weight upload are outside the timed region for both paths.

| Shape | Retained MQ4 | Q8_0 WMMA | Q8 speedup | Weight bytes |
|---|---:|---:|---:|---:|
| gate/up, M17408 K5120 N2048 | 14.0800 ms | 50.4708 ms | 0.2790x | 2.0000x |
| down, M5120 K17408 N2048 | 6.2037 ms | 20.0466 ms | 0.3095x | 2.0000x |

The generated carriers encode the same signed 4-bit values, but the two
existing kernels use different activation quantization and accumulation paths.
The resulting relative-L2 differences were `8.09e-3` for gate/up and `7.65e-3`
for down; these are plumbing diagnostics, not model-quality evidence.

Decision: reject. The existing Q8_0 primitive consumes twice the resident
weight memory and is 3.2-3.6x slower on the target full shapes. Do not add a
high-memory model route for this candidate.

Reproduction:

```bash
HIP_VISIBLE_DEVICES=1 target/release/examples/bench_q8_vs_mq4_ffn \
  --mode gate-up --pairs 10

sleep 20

HIP_VISIBLE_DEVICES=1 target/release/examples/bench_q8_vs_mq4_ffn \
  --mode down --pairs 10
```
