# Maple-Preview batched prefill (arch 15)

**Status:** design, approved 2026-08-22. Implements fast prefill for
`hipfire-arch-maple`; the decode path is already serving (see
`2026-08-22-maple-preview-20b-a1b.md`).

**Spec location note:** the superpowers default is `docs/superpowers/specs/`,
which is gitignored in this repo (`.gitignore:47`). This lives in `docs/design/`
so it is tracked, alongside `lfm2moe-gfx1201-prefill-architecture.md`.

## Problem

Maple decodes at ~120-134 tok/s but "prefills" at the same rate, because the
harness runs one `decode_step` per prompt token. A 3,059-token prompt costs
**24.4 s**. There is no batched path in this arch.

Measured 2026-08-22, gfx1151, release:

| Context | Prefill | Decode |
|---|---|---|
| 20 tok prompt, 45 gen | — | 134.1 tok/s |
| 20 tok prompt, 256 gen | 130.6 tok/s | 120.3 tok/s |
| 3,059 tok prompt, 64 gen | 125.5 tok/s (24.4 s) | 120.8 tok/s |

Decode is flat with context (18 of 24 layers are sliding-window-512), so prefill
is the only thing that scales badly.

## Key finding: no new HIP kernels are required

Every operation already exists. The one apparent gap — there is no *dense*
batched GEMM for `MQ2G256LloydU`, since every `gemm_mq2g256_lloyd_*` is
MoE-grouped — is closed by specializing the grouped kernel to a single expert
(below). A purpose-built dense kernel is **deferred to a measurement**, not
written speculatively.

## Design

### 1. `forward_batch`

```rust
pub fn forward_batch(
    cfg: &MapleConfig, weights: &MapleWeights, state: &mut MapleState,
    gpu: &mut Gpu, tokens: &[u32], start_pos: usize,
) -> Result<Vec<f32>, String>
```

Runs each weight matrix once for `B` tokens, fills KV for
`[start_pos, start_pos+B)`, returns the **last** token's logits. Per layer:

```
rmsnorm_batched(h[B], input_norm)             -> normed[B]
dense_qt51_gemm(wq/wk/wv)                     -> q[B], k[B], v[B]
rmsnorm_batched(q, q_norm, n_heads*B, head_dim)   QK-norm, BEFORE rope
rmsnorm_batched(k, k_norm, n_kv*B,    head_dim)
rope_partial_interleaved_f32_batched(...)     -> sliding layers ONLY
Step::Attend { AttnQ8_0KvBatchedMaskedWindowed, positions[B] }
dense_qt51_gemm(wo) + residual add
rmsnorm_batched(h, post_attn_norm)            -> normed[B]
moe_batch(...)                                -> accumulates into h[B]
```

Head: final RMSNorm on the last row only, then the existing `lm_head` GEMV.
Batching a 151,936-wide projection to discard all but one row is waste.

The per-layer ordering constraints are the same ones the decode path documents:
**QK-norm before RoPE**, and **RoPE on sliding layers only**.

### 2. `dense_qt51_gemm` — single-expert specialization

`gemm_mq2g256_lloyd_moe_grouped_wmma` computes, for one expert,
`Y = X @ W^T`. Driven with:

- `expert_weight_ptrs`: 1-entry table pointing at the projection
- `expert_tile_ids`: all zero (every tile uses expert 0)
- `sorted_slot_index`: identity `0..B`, padded
- `x_row_div = 1` (slot index *is* the row index; the MoE case divides by k_top)
- `m_total = ceil(B/16)*16` (BLOCK_M padding, ≤15 wasted rows)

The one-entry pointer tables are built **once at load**, per projection per
layer — not per call.

**`ensure_fp16_x` hazard.** That kernel converts `x` to FP16 internally and
caches the conversion **keyed on the source pointer**. Maple reuses a single
`normed` scratch buffer every layer with new contents, so the cache would hand
layers 1..23 layer 0's activations — silently, with no error. `cohere2moe` hit
exactly this (`q8_proj_raw`) and works around it by converting into its own F16
buffer per call. Do the same, and comment it so it is not "optimized" back.

### 3. MoE batching

```
router GEMV(batched) -> softmax -> moe_topk_renorm_k8(norm_topk = true)
moe_scatter_fused_k8
gemm_mq2g256_lloyd_moe_grouped_wmma      (already accepts MQ2G256LloydU)
moe_unscatter_silu_clamp_k8(swiglu_limit = 7.0)
grouped down GEMM
combine into h
```

`moe_unscatter_silu_clamp_k8` takes `swiglu_limit` and fuses the unscatter with
the asymmetric clamp, so Maple's clamped SwiGLU is free in the prefill layout —
the same way `deepseek4_silu_mul_clamp_f32_batched` covers it in decode.

**No FWHT anywhere.** `run_moe_prefill` takes `x_norm_batch` and `x_rot_batch`
from the caller; both receive the natural-basis activation, because qt51 weights
are unrotated. This is the same invariant the decode path enforces and the same
one `gate_down_skips_rotation` pins in the dispatch layer.

### 4. Chunking and integration

- Default chunk size **256**. `forward_batch` **errors** on `B > 512` (scratch
  ceiling) rather than silently splitting — splitting is the caller's job.
  Larger `B` raises tokens-per-expert (`B*k_top/n_exp`) and shrinks BLOCK_M
  padding waste, which is why 256 rather than 64.
- `forward_batch_supported(weights)`: all experts `MQ2G256LloydU` and Q8 KV.
  Callers fall back to per-token when false.
- `maple_coherence` prefills through it.
- `MapleCarrier::bench_prefill` overrides to chunk through it.

Out of scope: the daemon generate route (that is "make Maple servable", a
separate and much larger job), and the arm-A differential (arm A does not exist).

### 5. Testing

The per-token path is already verified end to end, so it is the oracle.

1. **Logit parity** — same prompt through `forward_batch` and the per-token
   path (`crates/hipfire-arch-maple/examples/maple_prefill_parity.rs`).
   Run at **B=1**, **B=17** (deliberately not a multiple of BLOCK_M, to catch
   padding bugs) and **B=256**.

   **Acceptance criterion revised 2026-08-22, after measurement.** The
   original bar here was "cosine >= 0.9999 AND identical greedy argmax
   [at the final position]". That was wrong on two counts, established by
   running the oracle rather than by inspection:

   - It compared only the LAST position of the whole run, i.e. **one**
     comparison point regardless of how many chunks were involved — a run
     that split into 12 chunks was reported as if 12 things had been
     checked, when only 1 had.
   - Its default input was synthetic ids scattered across the vocab
     (`1000 + i*7919 % 100000`), which is out-of-distribution: the model's
     logits go flat and hypersensitive to perturbation there, which
     *exaggerates* the arithmetic gap below. Feeding the same code real
     tokenized prose instead measured materially higher cosine.
   - 0.9999 cosine is unachievable by construction: the batched path is a
     WMMA GEMM over F16 activations (codebook deltas computed in
     `_Float16`, F32 accumulate —
     `kernels/src/gemm_mq2g256_lloyd_moe_grouped_wmma_k2.hip` ~L89-136),
     while the per-token oracle is an F32 scalar GEMV. A ~1e-3 relative
     per-GEMM difference is expected and compounds over 24 layers — the
     same shape as the shipped cohere2moe Q8 prefill path, which also
     feeds F16 activations to a WMMA GEMM.

   The revised bar:

   - **Hard bar: identical greedy argmax at every comparison point**, where
     a comparison point is one `forward_batch` CALL (it returns only the
     last token's logits per call, so a run chunked into N calls yields N
     points). The harness prints how many points were compared so a
     1-point run cannot be misread as an N-point pass.
   - **Reported metric, not the hard bar: cosine, floor 0.85**, set with
     margin below the worst minimum actually measured on real text (see
     below), not from the old 0.9999.

   **Measured 2026-08-22, real tokenized prose, 200 tokens, gfx1151,
   release** (`--tokens 200`, run-to-run range across repeats where the
   GPU's reduction order is not fully deterministic):

   | Config | Points | Argmax | Cosine min | Cosine mean |
   |---|---|---|---|---|
   | B=1 (200 chained 1-token chunks) | 200 | **FAILS every run** (~14-17/200 mismatch) | 0.897-0.906 | ~0.994 |
   | B=17 (12 chunks) | 12 | **flaky**: 9/12-12/12 across repeats | 0.951-0.984 | ~0.99 |
   | B=256 (1 chunk, whole prompt) | 1 | passes every run | 0.980-0.998 | — |
   | chunk-split (2 chunks of 100) | 2 | passes every run | 0.998-0.999 | ~0.999 |

   Synthetic OOD input, for contrast (NOT a representative measurement,
   see `--synthetic` in the harness): cosine min as low as 0.52, argmax
   fails at all three B values including B=256.

   **This is an open finding, not swept under the bar.** At B=256 (the
   production chunk size) and at a 2-chunk split, argmax parity holds on
   every real-text run so far. At B=1, and intermittently at B=17, it does
   not: chunk N's KV cache is itself the *output* of the WMMA-batched path
   for chunk N-1, so with many sequential chunks the F16-vs-F32 gap
   compounds again at every chunk boundary through the cached K/V, not
   just once over 24 layers. Whether that residual divergence is
   acceptable at small chunk sizes, or needs a mitigation (e.g. F32 KV
   write-back, or a minimum chunk size), is unresolved — tracked as a
   follow-up, not fixed by loosening this bar.
2. **Chunk-boundary** — a prompt prefilled as 100+100 must match one 200-token
   chunk. Catches `start_pos` / KV-offset errors.
3. **Sliding-window liveness** — parity alone cannot prove the window is
   applied: if BOTH paths dropped it they would still agree. So run the same
   >512-token prompt twice through `forward_batch`, once normally and once with
   a test-only override forcing `window = 0` on every layer
   (`HIPFIRE_MAPLE_FORCE_FULL_CAUSAL=1`), and require the logits to **DIFFER**.
   That makes the answers differ by construction, so the assertion cannot pass
   vacuously.
4. **No decode regression** — re-run the coherence gate; decode tok/s and output
   unchanged.
5. **Measurement** — report prefill tok/s at 3,059 tokens against the 125.5
   baseline. Only if the profile shows the tile-id indirection or BLOCK_M
   padding costs meaningfully do we write the dedicated dense kernel.

## Risks

- **Stale FP16 cache** (§2) — silent wrong activations. Mitigated by the
  per-call conversion buffer; test 1 at B>1 would catch it.
- **Within-block causality** — batched prefill needs query `p` to not see keys
  `>p`. `attention_flash_q8_0_batched_masked_windowed` masks per query by
  absolute position (`[p-window+1, p]`), so this holds; test 2 guards it.
- **Padding rows** — the `m_total` tail beyond `B` computes garbage that must
  never be read back. Test 1 at B=17 is the guard.
