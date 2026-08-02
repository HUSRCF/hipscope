# Gemma 4

This crate is the staged home for Gemma 4 text inference (`arch_id = 13`).

Implemented scope:

- strict E2B/E4B text-topology recognition from HFQ metadata;
- text-tower quantization for flat and unified checkpoints, excluding non-text
  towers from unified exports;
- HFQ loader/Carrier registration and unload cleanup for `arch_id = 13`;
- single-GPU Q8-KV autoregressive prefill/decode with bounded batched prefill;
- attempt/commit/abort serving protocol integration;
- embedded chat templates when present, otherwise a built-in best-effort
  fallback for checkpoints missing `chat_template`.

Validation status:

- E2B has passed real HFQ load, short prefill/decode, multi-chunk prefill,
  terminal commit, unload, and gfx1100 GPU smoke tests;
- E4B has config and loader contract coverage but has not yet received a real
  checkpoint GPU validation run;
- gfx1100 flash-attention partial workspace sizing follows the runtime-selected
  tile geometry rather than assuming a fixed 128-token tile.

Current non-scope:

- no vision or audio tower execution;
- no tool calls, prefix cache, DFlash, or PFlash route;
- no speculative assistant for E2B or E4B;
- no pipeline parallelism or raw safetensors-directory serving;
- no Dense12B or MoE26B execution contract.

Use `scripts/gemma4-e-series-smoke.sh` for the reproducible GPU smoke. The
fallback template is intentionally not presented as an instruction-tuning
contract: checkpoints without an embedded template may echo the prompt or stop
early even when tokenization and model execution are correct.
