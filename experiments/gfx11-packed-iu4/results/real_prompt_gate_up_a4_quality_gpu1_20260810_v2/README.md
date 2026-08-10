# Real-prompt IU4-A4 quality gate

This result rejects the current signed-A4 activation route for production use.

- GPU: Radeon Pro W7900 / gfx1100, HIP device 1
- model: Qwen3.6-27B MQ4
- prompt: `docs/testINPUT.md`, 3371 input tokens
- decode: target-only greedy, Asym3 KV, 64 requested output tokens
- control: retained Q8-group128 gate/up path
- candidate: default-off IU4-A4 group128 gate/up path

Both arms produced 65 recorded token IDs including the initial token. The first
32 output token IDs match. The first divergence is at index 32, after which the
A4 stream becomes visibly degraded. The A4 route marker was present and the CK
attention sidecar remained active, so this is not a routing ambiguity.

The earlier PP8192 benchmark measured a repeatable `+4.89%` prefill gain, but
that performance result does not pass this longer quality gate. The route must
remain default-off and must not be promoted or extended to other projections.
