# Q8 GDN compact-Q/K prefill routing probe

This experiment reused the existing production `compact2` Q8 GDN kernel in
Qwen3.6 prefill. The candidate kept normalized Q/K at the model's native eight
heads and mapped each pair of 16 value/state heads to one Q/K head, instead of
materializing the repeated 16-head tensors before the ordinary GDN kernel.
The route was opt-in during measurement and was removed after rejection.

## Configuration

- GPU: W7900 / gfx1100, GPU1, ROCm runtime 7.14
- model: `qwen3.6-27b.mq4`
- PP8192, three in-process prefill runs, 16-token decode check
- asym3 KV and quantized CK attention sidecar
- retained X256/Y64 group128 quad-row packed-MQ4 path

The baseline first run included a packed-MQ4 module recompile. The comparison
uses the reported medians and also records the two later hot runs.

| Mode | Hot prefill runs (tok/s) | Reported median (tok/s) | Decode (tok/s) |
|---|---:|---:|---:|
| Repeated Q/K + ordinary GDN | 1147.7, 1138.3 | 1138.3 | 33.2 |
| Compact Q/K + compact2 GDN | 1125.8, 1115.8 | 1125.8 | 33.0 |

The compact route is `0.9890x` the baseline (`-1.10%`) by reported median.
Both modes emitted the same recorded 19-token sequence. Eliminating the Q/K
interleave materialization does not offset the compact kernel's execution cost
for long prefill; no production flag or routing change is retained.
