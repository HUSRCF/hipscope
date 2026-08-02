# Gemma 4

This crate is the staged home for Gemma 4 text inference (`arch_id = 13`).

Implemented scope:

- strict E2B/E4B text-topology recognition from HFQ metadata;
- text-tower quantization for flat and unified checkpoints, excluding non-text
  towers from unified exports;
- HFQ loader/Carrier registration and unload cleanup for `arch_id = 13`;
- single-GPU Q8-KV autoregressive prefill/decode with bounded batched prefill;
- attempt/commit/abort serving protocol integration;
- strict Gemma-native tool-call routing with structured terminal events;
- process-local assistant-turn prefix reuse for exact forward extensions;
- embedded chat templates when present, otherwise a built-in best-effort
  fallback for checkpoints missing `chat_template`.

Validation status:

- E2B has passed real HFQ load, short prefill/decode, multi-chunk prefill,
  terminal commit, unload, and gfx1100 GPU smoke tests;
- E4B has passed the same real-checkpoint GPU path on gfx1100 using the
  `google/gemma-4-E4B-it` BF16 checkpoint converted to text-only Q8 HFQ;
- E4B has passed an end-to-end tools and prefix-cache smoke, including reuse
  across an assistant tool call and its tool result;
- gfx1100 flash-attention partial workspace sizing follows the runtime-selected
  tile geometry rather than assuming a fixed 128-token tile.

Current non-scope:

- no vision or audio tower execution;
- no cross-process or persistent prefix cache;
- no DFlash or PFlash route;
- no speculative assistant for E2B or E4B;
- no pipeline parallelism or raw safetensors-directory serving;
- no Dense12B or MoE26B execution contract.

Use `scripts/gemma4-e-series-smoke.sh` for the base GPU smoke and
`scripts/gemma4-tools-prefix-smoke.sh` for the tools/prefix-cache path. Prefix
reuse is enabled by default and can be disabled with
`HIPFIRE_GEMMA4_PROMPT_CACHE=0`. It requires an exact cached assistant-turn
prefix and only reuses state when the cached token count matches both the
conversation token history and the live model state. Tool-enabled requests are
buffered until terminal routing so malformed or incomplete native tool syntax
can fail closed without exposing protocol fragments.

The fallback template is intentionally not presented as an instruction-tuning
contract: checkpoints without an embedded template may echo the prompt or stop
early even when tokenization and model execution are correct.

The E4B validation used source SHA-256
`cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`:

```bash
target/release/hipfire-quantize \
  --input /path/to/gemma-4-E4B-it \
  --output artifacts/gemma4-e4b-q8f16.hfq \
  --format q8

MODEL=artifacts/gemma4-e4b-q8f16.hfq \
GPU_ID=1 \
OUT=/tmp/hipfire-gemma4-e4b-smoke.log \
scripts/gemma4-e-series-smoke.sh
```

On a Radeon Pro W7900, the smoke completed both attempt/commit requests: the
106-token multi-chunk case measured `878.9660 prefill tok/s` and `68.2805
decode tok/s`. These numbers are smoke evidence for the E4B execution contract,
not a formal performance baseline.

On the same GPU, the tools/prefix smoke reused 18 tokens on a normal follow-up,
returned `get_weather({"city":"Taipei"})` as a structured tool call, and reused
81 tokens after the tool result. These are functional cache-hit observations,
not throughput claims.
