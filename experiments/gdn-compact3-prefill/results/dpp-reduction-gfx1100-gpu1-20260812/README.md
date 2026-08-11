# GDN DPP reduction probe on gfx1100

This PP8192 experiment reused the existing DPP/permlane reduction helper from
the gfx1151 compact GDN experiment in the ordinary gfx1100 Q8 prefill kernel.
The recurrence, state format, launch geometry, and model routing were otherwise
unchanged.

The first run exposed a correctness bug in the existing helper. Passing zero
selectors to `__builtin_amdgcn_permlanex16` does not map lane `i` to lane
`i+16`; the candidate produced all-zero token IDs. The identity cross-row
selectors are `0x76543210` and `0xfedcba98`. Correcting them restored the exact
recorded token sequence.

## Corrected A/B

- GPU: W7900 / gfx1100, GPU1, ROCm runtime 7.14
- model: `qwen3.6-27b.mq4`
- PP8192, asym3 KV, quantized CK attention, retained packed-MQ4 route

| Mode | Hot prefill runs (tok/s) | Reported median (tok/s) | Token sequence |
|---|---:|---:|---|
| Shuffle reduction | 1147.7, 1138.3 | 1138.3 | reference |
| Corrected DPP reduction | 1141.1, 1131.2 | 1131.2 | identical |

The corrected DPP route is `0.9938x` the baseline (`-0.62%`) and is not
retained for gfx1100. The selector correction remains because it fixes the
semantics of the already-present gfx1151 experimental helper.
