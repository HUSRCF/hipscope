# DFlash Architecture — hipfire port

Phase 1 scope check. Written after reading:

- `.dflash-reference/dflash/model.py` (338 LOC), `.dflash-reference/dflash/benchmark.py`
- z-lab/Qwen3.5-9B-DFlash config.json + the bundled `dflash.py` (HF repo)
- `.dflash-ref-hf/model.safetensors` tensor inventory (58 tensors, 2.1 GB BF16)
- arXiv:2602.06036 abstract + the z-lab blog post (full PDF not scraped)
- hipfire's existing `crates/engine/src/speculative.rs` (already has scaffolding)
- hipfire's `crates/engine/src/qwen35.rs` forward variants

**Go/no-go recommendation: GO.** MVP is reachable tonight. Sections below
explain why the scope is closer to 2 weeks of careful work than the
6-week "continuous noise schedule" nightmare the plan feared.

## 1. The critical finding

DFlash's "block diffusion" is **masked-token parallel infilling, not
continuous denoising.** The draft runs **one forward pass** per
speculative block, fills B=16 mask slots with predicted token IDs,
and hands them to the target for verification.

There is **no noise schedule, no timestep embedding, no iterative
denoising loop**. The paper's choice of the word "diffusion" is a
nod to the family of models; the runtime implementation is a single
bidirectional forward over mask-embedded inputs.

Confirmed by the blog post: *"Diffusion drafters generate all tokens
in a single parallel forward pass, so drafting cost is essentially
flat regardless of how many tokens you produce."*

Confirmed by the reference `spec_generate` in `dflash.py` — the draft
is called exactly once per speculative block, no inner loop.

## 2. Draft architecture (pinned to Qwen3.5-9B-DFlash)

- **Model type:** Qwen3 decoder (no DeltaNet, no sliding window, all full
  attention, `is_causal=False`)
- **Layers:** 5
- **Hidden / intermediate:** 4096 / 12288
- **Heads:** 32 Q, 8 KV, head_dim 128 (GQA 4:1)
- **Block size B:** 16
- **Vocab:** 248320 (Qwen3.5 vocab, shared with target)
- **mask_token_id:** 248070
- **Positional:** RoPE θ=10,000,000 (same as Qwen3.5 target — this is important)
- **RMS norm ε:** 1e-6
- **Extra head:**
  - `fc: Linear(5 × 4096, 4096, bias=False)` — projects concatenated
    target hidden states (from layers [1, 8, 15, 22, 29] of the 32-layer
    Qwen3.5 target) into a single hidden.
  - `hidden_norm: RMSNorm(4096)` — applied after fc.
  - `norm: RMSNorm(4096)` — final output norm.
- **NO embed_tokens, NO lm_head.** The draft shares the target's embedding
  table and LM head. Input is `target.embed_tokens(block_output_ids)`,
  output is mapped back to vocab via `target.lm_head(draft_hidden)`.
- **Tensor inventory (58 total, all BF16, 2.1 GB):** verified by
  `safetensors safe_open` against downloaded model.safetensors.
  - `fc.weight [4096, 20480]`, `hidden_norm.weight [4096]`, `norm.weight [4096]`
  - Per layer (×5): q_proj, k_proj, v_proj, o_proj, q_norm[128],
    k_norm[128], input_layernorm, post_attention_layernorm,
    mlp.{gate,up,down}_proj

## 3. What makes the draft attention different

The draft's `Qwen3DFlashAttention.forward(hidden_states, target_hidden)`
computes:

```
q = q_proj(hidden_states)                       # [B, heads, head_dim]
k_ctx   = k_proj(target_hidden)                 # [ctx_len, kv_heads, head_dim]
k_noise = k_proj(hidden_states)                 # [B, kv_heads, head_dim]
v_ctx   = v_proj(target_hidden)
v_noise = v_proj(hidden_states)
k = concat([k_ctx, k_noise], dim=seq)           # [ctx_len + B, kv_heads, head_dim]
v = concat([v_ctx, v_noise], dim=seq)
q = q_norm(q)                                   # per-head RMSNorm
k = k_norm(k)
(q, k) = apply_rotary(q, k, positions)
attn = softmax(q @ k.T / sqrt(head_dim)) @ v    # is_causal=False
out = o_proj(attn.flatten_heads())
```

Two non-standard things for hipfire:

1. **K/V is built from two sources.** The first `ctx_len` entries come from
   the (projected) target hidden states; the next `B` entries come from
   the draft's own rolled hidden states. Existing hipfire attention
   kernels already take a flat K/V tensor, so this is a CPU/host-side
   concat, not a kernel change.
2. **Non-causal.** Existing prefill-batch attention can be non-causal
   just by not zeroing out the upper triangle; hipfire already has
   full-attention flash paths (`attention_flash_asym2_tile.hip`,
   `attention.hip`) — we pass a permissive mask or a flag.

**What `target_hidden` looks like inside the draft:** the spec_generate
loop calls `self.hidden_norm(self.fc(target_hidden))` *once* at the top
of `DFlashDraftModel.forward`. Output shape is `[ctx_len, 4096]`. Every
one of the 5 draft layers consumes this same tensor. So the 5×hidden →
hidden projection is a single GEMV per cycle, not per layer.

## 4. Speculative decoding loop (pseudocode)

Stripped from `DFlashDraftModel.spec_generate` in the reference. One
important detail about the draft's KV cache was clarified only by
reading the actual code.

```text
# --- prefill ---
target.prefill(prompt)                           # stock Qwen3.5 prefill
# target output: {last_logits, hidden[5 chosen layers at all prompt positions]}
first_token = argmax(target.last_logits)         # bonus token from prefill
target_hidden = concat_on_hidden(hidden[1, 8, 15, 22, 29])  # [prompt_len, 5*4096]

output_ids = [prompt..., first_token, MASK, MASK, ..., MASK]  # padded with MASKs
start = prompt_len + 1                           # position of next-to-decide
pkv_target.end = start                           # target KV already prefilled to here
pkv_draft.end  = 0                               # draft KV starts empty

# --- decode loop ---
while start < max_length:
    # DRAFT: one forward over B mask positions, producing B-1 new tokens
    # (position 0 of the block is the last-committed token, not a mask)
    block = output_ids[start-1 : start-1+B]                  # [B]; block[0] known, block[1..B-1] = MASK
    noise_emb = target.embed_tokens(block)                   # [B, 4096]

    # Draft runs its 5 layers with cross-attention over target_hidden
    draft_hidden = draft_forward(
        noise_embedding = noise_emb,
        target_hidden   = target_hidden,        # [ctx_len, 5*4096]
        position_ids    = pkv_draft.end .. start-1+B,
        past_kv         = pkv_draft,            # persists cross-layer K/V of ACCEPTED positions
    )
    draft_logits = target.lm_head(draft_hidden[-(B-1):])     # last B-1 positions
    pkv_draft.crop(start - 1)                                # discard the mask-position K/V that this call appended

    block[1 : B] = argmax(draft_logits)                      # fills the mask slots

    # TARGET: batched verify — forward the full B-sized block, get B sets of logits + B hidden states
    verify_out = target.prefill_batch(block, start_pos = start - 1)
    # verify_out.logits: [B, vocab]
    # verify_out.hidden: extracted at layers [1, 8, 15, 22, 29], [B, 5*4096]
    posterior = argmax(verify_out.logits)                    # [B]

    # Accept: leading run where draft matches posterior (shifted by 1)
    #   draft proposal is block[1..B]       (B-1 tokens)
    #   target posterior is posterior[0..B-1] (B-1 tokens, aligned to same positions)
    accept_len = longest_prefix_match(block[1:], posterior[:-1])   # 0..B-1
    # Always commit: accepted draft prefix + 1 bonus posterior token
    committed = block[1 : 1 + accept_len] + [posterior[accept_len]]

    output_ids[start : start + len(committed)] = committed
    start += len(committed)
    pkv_target.crop(start)                                   # truncate target KV to accepted count
    target_hidden = verify_out.hidden[:accept_len + 1]       # [accept_len+1, 5*4096] for next iter

    if any(eos in committed) or start >= max_length: break
```

Greedy parity invariant: every committed token is either `block[i] =
argmax(draft_logits[i-1])` that matches target's `posterior[i-1] =
argmax(target_logits[i-1])`, or `posterior[accept_len]` itself. So the
final output equals `argmax(target_logits_at_each_position)` — which is
exactly what non-speculative greedy would produce. This is a
distribution-preserving optimization for temp=0; for temp>0 it needs
rejection sampling (deferred to 0.1.7).

## 5. What the existing hipfire code already has

Last three commits on master (`git log --oneline -- crates/engine/src/speculative.rs`):

- `8e2ea42 feat(speculative): Phase 1 dual model slot infrastructure`
- `bf6b653 feat(speculative): Phase 2 autoregressive verify-and-accept loop`
- `a015db9 feat(speculative): Phase 3 hidden state extraction for DFlash`

What we inherit:

| File / symbol | Status | What it gives us |
|---|---|---|
| `ModelSlot::load` | done | Loads Qwen3.5 target or draft into a GPU slot (KV, DeltaNet state, scratch) |
| `SpecPair::load` | done | Pairs target + draft on one GPU + tokenizer compat |
| `SpecPair::smoke_test` | done | Tests both slots run forward and produce finite logits |
| `spec_step_greedy` | done (classic) | Leviathan sequential verify — *not* DFlash-shaped but provides all the plumbing |
| `DeltaNetSnapshot::{new_for,save_from,restore_to}` | done | Save/restore DeltaNet state around speculation (useful but not strictly needed for dflash) |
| `HiddenStateRingBuffer::{new,write_at_head,advance_head}` | done | F32 ring buffer for target hidden states at layers [1,8,15,22,29]; ready for dflash |
| `dflash_extract_layer_ids(32, 5)` | done, returns `[1,8,15,22,29]` | Matches HF config exactly |
| `forward_scratch_with_hidden(...)` in qwen35 | done | Single-token target forward that writes extracted hiddens to the ring buffer |
| `forward_prefill_batch(...)` in qwen35 | done, no hidden | Batched target prefill; doesn't yet write to hidden ring buffer |
| `forward_scratch_embed(...)` in qwen35 | done | Lets us feed a pre-computed embedding into target (useful for the MASK-embed path) |
| `SpecStats::{tau, mean_committed}` | done | Metrics plumbing for accept rate reporting |

So: we **do not** need a new Qwen3.5 target forward. We do **not** need
to wire up hidden-state extraction at the per-token level. We **do**
need to:

1. Extend `forward_prefill_batch` with a hidden-ring-buffer variant (or
   fall back to per-token `forward_scratch_with_hidden` inside it — OK
   for MVP).
2. Build a dflash-specific draft forward (new `Qwen3Draft` type — the
   draft is *not* Qwen3.5 with DeltaNet).
3. Build a dflash spec step (`spec_step_dflash`) that calls the draft
   once, the target prefill-batch once, and slices the hidden buffer for
   next iteration.

## 6. Component inventory — reuse vs new

### Reuse as-is (no changes)

- `hfq.rs` HFQ file reader (draft weights land in a `.hfq`-flavored file
  with a `dflash` section — see Phase 2 below)
- Tokenizer (shared with target; verified by `SpecPair::load`)
- `qwen35::forward_scratch` / `forward_prefill_batch` — target path, untouched
- `qwen35::forward_scratch_with_hidden` — target decode with hidden extract
- All existing kernels in `kernels/src/*.hip`

### Reuse with small extensions

- `forward_prefill_batch` — add an optional `hidden_rb: Option<&mut HiddenStateRingBuffer>`
  parameter that, when `Some`, writes all extracted hidden states at
  batched positions. MVP fallback: call per-token hidden variant in a
  loop when `hidden_rb` is requested.
- `SpecPair` — add a `SpecPair::new_dflash(...)` constructor that
  distinguishes the draft type at load time (Qwen3Draft, not Qwen35Weights).
  Maybe cleaner: a `DraftModel` enum { Qwen35(ModelSlot), Dflash(DflashSlot) }.

### New code (additive only)

- `crates/engine/src/dflash.rs` — new module with:
  - `DflashConfig` (block_size, mask_token_id, target_layer_ids,
    hidden, heads, kv_heads, head_dim, n_layers, rope_theta, eps)
  - `DflashWeights` (5× decoder layers, fc, hidden_norm, norm)
  - `DflashScratch` (block-sized activations: x_block, qkv_block,
    logits_block, plus persistent draft KV cache of length ≥ max_seq)
  - `load_dflash_weights(hfq: &HfqFile, cfg: &DflashConfig, gpu: &mut Gpu) -> DflashWeights`
  - `draft_forward(gpu, weights, cfg, noise_embedding, target_hidden, positions, kv_cache, scratch) -> &[f32]`
    Returns logits for the B-1 mask positions. Embedding and lm_head stay on the target.
- `crates/engine/src/speculative.rs` additions (append, don't rewrite):
  - `spec_step_dflash(gpu, target_slot, dflash_slot, target_hidden_buf, pos, block_size, ...) -> SpecStepResult`
  - `DflashSpecPair` or extend `SpecPair` to hold a `DflashSlot` draft
- `crates/hipfire-quantize/src/bin/dflash_convert.rs` — new binary:
  - Reads HF safetensors + config.json
  - Writes `.hfq` with:
    - a `Dflash` header section containing `block_size`, `mask_token_id`,
      `target_layer_ids`, `num_target_layers`, `num_hidden_layers`,
      `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`,
      `intermediate_size`, `rms_norm_eps`, `rope_theta`, `vocab_size`
    - 58 tensors renamed to hipfire's convention (`blk.N.attn_q`, etc.)
    - Target model reference (name/hash) so the daemon can verify vocab compat
- `kernels/src/` — potentially *no new kernels* for MVP. Existing attention
  kernels support non-causal full attention; existing GEMV/GEMM is ok
  for BF16 or fp16 weights. New kernel only if a perf hot spot justifies it.

### Deferred to 0.1.7

- MQ4 draft quantization (Phase 7 of PORT_PLAN)
- Batched prefill hidden extract (the per-token fallback is correct;
  fast path is a perf polish)
- Rejection sampling at temp > 0
- Temp-variant accept-rate telemetry

## 7. Scope estimate

Rough hours (vs 6-hour session budget, in priority order):

- Phase 1 (this doc): 1 hr — **DONE at commit of this file**
- Phase 2 (dflash_convert.rs): 1.5 hr — HFQ format extension + 58-tensor
  rename table. Mechanical.
- Phase 3 (dflash.rs draft_forward): 2.5 hr — new module, but 90%
  re-uses existing RMSNorm/rotary/MLP/GEMV/attention kernels. Biggest
  risk is the cross-K/V concat + non-causal attention wiring.
- Phase 4 (batched verify + hidden extract): 1 hr — thin wrapper over
  the existing per-token hidden path.
- Phase 5 (spec_step_dflash): 1 hr — orchestration. Well-defined loop.
- Phase 6 (CLI `--dflash`, serve accept-rate): 1 hr — trivial plumbing.
- Phase 7 (ship BF16, quality gate pass): 0.5 hr — no quantize work
  tonight; defer MQ4 draft to 0.1.7.
- Phase 8 (BENCHMARKS.md + SPECULATIVE_DECODING.md): 0.5 hr.

Total: ~9 hours. **Tighter than 6.** Mitigation: cut Phase 6 and 7/8 to
minimum (flag-gated runtime, terse doc). Ship a CLI-runnable
`hipfire run qwen3.5:9b --dflash` path. Bench numbers and the v0.1.6
release ride the rails as far as they get.

Fallback MVP if Phase 3 hits a kernel dead end: Rust-host implementation
of the draft forward (BF16 GEMV in a simple hand-rolled kernel that
doesn't need WMMA/dot2). 1B at BF16 on 7900 XTX is BW-bound, draft cost
is amortized over the block, a naïve kernel is fine for proving the loop.

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Target needs batched hidden extract that doesn't exist | certain | Fallback to per-token `forward_scratch_with_hidden` inside verify. Correct; perf later. |
| Draft cross-attention kernel is a bug magnet | medium | Use `forward_prefill_batch`-style (B+ctx_len) attention with a custom K/V concat host-side. Verify against PyTorch reference logits via a scratch dump on a small prompt. |
| The `pkv_draft.crop(start-1)` semantics are subtly wrong | medium | Reference reads: after draft forward appends `ctx_len + B` K/V entries at positions `start-ctx_len-1..start-1+B`, crop to `start-1`. Net: cache retains only the ctx positions this iteration added. We can simplify: recompute `k_ctx`/`v_ctx` fresh every iteration, never persist draft KV. Removes the crop entirely. **Adopt this simplification for MVP.** |
| Target hidden layers differ on non-9B targets | likely | Read `target_layer_ids` from the HFQ dflash section, don't hard-code. |
| BF16 draft + quantized target mixing breaks quality gate | medium | Baseline capture for dflash path is separate; same greedy output as non-dflash is the invariant. |
| VRAM blowup on < 16 GB cards | low-medium for 9B | 9B MQ4 target ≈ 5.1 GB + 1B BF16 draft ≈ 2.1 GB ≈ ~7.5 GB model; KV + scratch pushes to ~10 GB. 7900 XTX (24 GB) is fine; V620 16 GB borderline; BC-250 8 GB out. Default off on <16 GB. |
| Paper algorithm differs from code | low | The paper abstract + blog + reference all agree: single-shot mask infill. We follow the code (authoritative). |

## 9. The simplification we should adopt

The reference keeps a draft KV cache that's cropped after every forward
so that, across iterations, it grows by `accept_len + 1` per step — which
is just enough to hold the projected target hidden of newly-accepted
positions. **In hipfire, skip this entirely.** Instead:

- Compute `k_ctx`, `v_ctx` fresh on every speculative step from the
  sliced `target_hidden` buffer.
- Compute `k_noise`, `v_noise` every step from the draft's rolling
  hidden states (as normal).
- Concat on the host CPU (or a tiny HIP copy kernel).
- Never persist draft KV across steps.

This is equivalent to the reference in outputs and strictly simpler.
The reference's cache is an opt for avoiding recomputing projections of
stable target context; we eat that recompute since `ctx_len ≤ B+1` by
construction after the first iteration, so the cost is bounded to one
small GEMV per layer per step.

First-iteration cost is the exception: `ctx_len = prompt_len` for the
first step. On a 2000-token prompt that's 2000×4096 projection per layer
per step, which is a one-shot cost. We'll accept it for MVP and optimize
later if it shows up in profiles.

## 10. Go / no-go

**GO.** Immediate next step: Phase 2 — write `dflash_convert.rs` and
convert the downloaded 2.1 GB BF16 safetensors into an `.hfq`-flavored
file that hipfire can memory-map. Commit to `origin/dflash`. Then
Phase 3 (draft forward in `dflash.rs`). Then Phase 4+5+6 end-to-end.

The MVP bar is: a `hipfire run qwen3.5:9b --dflash` invocation that
prints the same greedy output as `hipfire run qwen3.5:9b` and reports
an accept rate. Speed can be 1.0× for MVP. If we reach any speedup at
all, we've proven the loop. If Phase 3 runs long, pivot to a naïve
CPU-host reference implementation that calls existing hipfire GEMV
kernels per weight — slower, but proves the loop.
