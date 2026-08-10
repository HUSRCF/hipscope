# Asym3 constant-codebook directional A/B

This is a mechanism check, not clean production evidence. Two unrelated
hipfire daemons were resident while the test ran, with both W7900 cards near
65% VRAM allocation. Alternating order controls first-order drift, but the
absolute timings remain contaminated.

Both binaries use the same gfx1100 M64/N32 CK pipeline. The only difference is
Asym3 centroid selection: the baseline inlines an eight-way switch, while the
candidate performs an indexed load from a device constant codebook.

| Mode | Q | K | Five-run median | Raw total times (ms) |
| --- | ---: | ---: | ---: | --- |
| inline switch | 128 | 8192 | `4.5180 ms` | 4.4842, 4.5180, 4.4889, 4.5956, 6.1954 |
| constant codebook | 128 | 8192 | `2.6655 ms` | 2.6275, 2.7257, 2.6864, 2.5889, 2.6655 |

The directional ratio is `1.695x` in favor of the constant codebook. The
maximum CK/native output difference is unchanged at `2.14e-5`.

Static ISA evidence is consistent with the timing signal:

| Metric | inline switch | constant codebook |
| --- | ---: | ---: |
| FMHA disassembly lines | 13,560 | 7,548 |
| shared memory | 32 KiB | 32 KiB |
| VGPR | 256 | 256 |
| private bytes/work-item | 400 | 396 |
| scratch loads/stores | 31 / 30 | 31 / 29 |
| global loads | 300 | 419 |

The candidate trades additional constant-cache loads and waits for much lower
instruction footprint. It remains build-time opt-in until a clean-card Qwen
PP8192 multi-process A/B confirms a model-level gain.
