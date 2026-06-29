# Qwen3-8B DSpark drafter — ingest topology & attention mode (Task 0)

Settles the unknowns the config alone doesn't reveal, by reading the drafter
safetensors header + the DeepSpec reference (`deepspec/modeling/dspark/qwen3/
modeling.py`, and the on-disk deepseek4 reference `~/dspark-work/ref/model.py`).

**Headline finding (revises the plan):** the qwen3 DSpark drafter is **not** a
plain dense self-attention transformer with a modified layer-0 input. It is
**deepseek4's DSpark body algorithm with dense Qwen3 layers** — each layer's
attention is **bidirectional** and prepends a **projected target-hidden context**
to its KV (`KV = cat([k_proj(target_ctx), k_proj(block)])`). hipfire's deepseek4
`dspark_forward` (`crates/hipfire-arch-deepseek4/src/forward.rs:8911+`) already
implements this exact structure (MoE/MLA variant); the qwen3 body is the dense
variant of the same thing.

## Tensor name → role table (`dspark_qwen3_8b_block7`)

| tensor (safetensors) | shape | dtype | role |
|---|---|---|---|
| `model.embed_tokens.weight` | [151936, 4096] | bf16 | token embedding (noise block) |
| `model.layers.{0..4}.self_attn.{q,k,v,o}_proj.weight` | GQA (32h/8kv/hd128) | bf16 | dense Qwen3 attention |
| `model.layers.{0..4}.self_attn.{q,k}_norm.weight` | [128] | bf16 | QK-norm (Qwen3) |
| `model.layers.{0..4}.{input_layernorm,post_attention_layernorm}.weight` | [4096] | bf16 | block norms |
| `model.layers.{0..4}.mlp.{gate,up,down}_proj.weight` | inter 12288 | bf16 | SwiGLU MLP |
| `fc.weight` (`main_proj`) | **[4096, 20480]** | bf16 | **single concat** `[hidden, 5*hidden]` ingest |
| `hidden_norm.weight` | [4096] | bf16 | RMSNorm after `fc`, before layers |
| `norm.weight` | [4096] | bf16 | final norm → `x_head` |
| `markov_w1.weight` | [151936, 256] | bf16 | vanilla markov embed |
| `markov_w2.weight` | [151936, 256] | bf16 | vanilla markov bias proj |
| `confidence_head.proj.weight` | [1, 4352] | bf16 | confidence Linear (dim+rank) |
| `confidence_head.proj.bias` | [1] | bf16 | **confidence HAS BIAS** (deepseek4 has none) |
| `lm_head.weight` | [151936, 4096] | bf16 | separate lm_head (untied) |

## The 7 questions

1. **`main_proj` shape:** single `[4096, 20480]` = `[hidden, 5*hidden]` **concat**
   (named `fc`). Generalizes deepseek4's `[hidden, 3*hidden]` to `n_targets=5`.
2. **Layer-0 / ingest rule:** NOT `embed(noise)+main_proj` sum. `main_x =
   hidden_norm(fc(main_hidden))` is computed once and fed as a **separate context
   side-channel** to every layer. Per layer: query = block hidden (layer 0 =
   `embed(noise_block)`); `KV = cat([k_proj(main_x_context), k_proj(block)])`,
   `V` likewise. `hidden_norm` applies after `fc`, before the layer loop. This is
   structurally identical to deepseek4 `dspark_forward` step A (`main_x =
   main_norm(main_proj(main_hidden))`, forward.rs:9014-9058) + its custom
   bidirectional stager (forward.rs:9100 `n_valid[b]=n_committed+block`).
3. **Block attention:** **bidirectional.** Reference `DSparkAttention.is_causal =
   False`; at inference `attention_mask=None` ⇒ every block slot attends to all
   context + all block slots. Matches deepseek4's bidirectional mode.
4. **Persistent KV across windows:** NO. `deepspec_draft_ops.py` crops the draft
   KV per block (`past_key_values_draft.crop(start)`); committed context enters
   only through `main_x`, not a carried KV. (deepseek4 keeps a `win`-sized main_kv
   ring but writes only the seed per window — functionally the same "context =
   projected target hidden" scheme.)
5. **Markov head:** vanilla rank-256, identical to deepseek4 (`markov_w1`
   Emb[vocab,256], `markov_w2` Linear[256→vocab] used as bias). matches
   `forward.rs:9663-9715`.
6. **Confidence head:** `Linear[dim+rank=4352, 1]` over `[x_head ++ markov_emb]` —
   **with bias** (deepseek4's is bias-free fp32). Loader + forward must include the
   `confidence_head.proj.bias` term.
7. **`x_head` / `lm_head`:** `x_head` = post-`norm` hidden; `lm_head` is a separate
   `[151936,4096]` tensor (`tie_word_embeddings:false`).

## Branch points for later tasks

| affects | decision |
|---|---|
| ingest (`fc`) | single concat `[4096, 20480]`; generalize core `n_targets*dim` width |
| body attention | **bidirectional block over `[main_x context ++ block]` KV** — NOT llama's plain causal self-attn forward |
| body layers | dense Qwen3 (rmsnorm → GQA self-attn w/ qk-norm → swiglu) — kernels exist in llama/rdna-compute |
| closest existing impl | **deepseek4 `dspark_forward`** (forward.rs:8911-9739), dense variant — NOT llama `forward_scratch_layers` |
| confidence | include bias term |
| heads | separate `lm_head`; vanilla markov == deepseek4 |

## Plan impact (escalated to human)

The Stage-1 plan's premise — "the qwen3 body fits llama's existing forward
unchanged; only gemma4 forces generalization" — is **false**. The qwen3 body
needs the DSpark bidirectional-block-attention-over-projected-context machinery,
which deepseek4 has and llama does not. The body should be built by adapting
deepseek4's `dspark_forward` (swap MoE/MLA/HC → dense Qwen3 layers), or by a new
masked-batched-GQA path over `[context ++ block]` keys (the #483
`attention_q8_0_kv_batched_masked` can express the all-visible bias). This also
suggests the bidirectional-block-staging belongs in `dspark-core` (shared with
deepseek4), with only the per-layer compute as the arch seam. `hipfire-dense`
(gemma knobs) remains a separate Stage-2 concern.
