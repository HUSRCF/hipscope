# Group256 serial-row F16 gate/up rejection

This experiment tested whether the existing opt-in group256 activation contract
could be combined with the retained FP16 FFN intermediate path on gfx1100. It
was evaluated on GPU1 of an AMD Radeon Pro W7900 with ROCm 7.14.

## Standalone result

Shape: `M=17408, K=5120, N=2048`, seven alternating pairs after warmup. Both
times include activation quantization plus the two gate/up projections.

| Path | Median (ms) |
| --- | ---: |
| group128 quad-row, F16 output | 9.3463 |
| group256 serial-row, F16 output | 8.4169 |

The candidate was `1.1104x` faster locally. The cross-path intermediate drift
was `max_abs=2.00927734e-1` and `mean_abs=3.04594971e-2`; these are not errors
against a golden reference.

## Real-prompt quality result

The real-prompt comparison used `docs/testINPUT.md` (3371 prompt tokens),
thinking mode, greedy decoding, asym3 KV, and 257 output token IDs. The current
group128 route and the group256 candidate matched for 96 output tokens and then
diverged at index 96.

Artifacts:

```text
../real_prompt_group256_serial_gpu1_20260812_030658/
```

## Decision

Rejected for serving. The local speed signal does not justify changing the
generation trajectory. The new group256 F16 kernel remains a standalone probe;
the production Qwen path continues to use the exact retained group128 quad-row
F16 route. No PP16384 performance run was performed for this rejected
candidate.
