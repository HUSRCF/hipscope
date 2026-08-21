# Maple-Preview 20B-A1B — hipfire port (native ternary, arch 15) — design

**Status of every claim below is marked.** `measured` = read off the published
artifact by this branch's author, with the method stated. `published` = asserted
by the model card or `config.json`, not independently checked. `planned` =
intent, nothing implemented. Nothing here is `branch-implemented` yet.

**Branch:** `quant/maple-preview`, stacked on `quant/quality` (PR #599). Base
commit `673ea3ae0`. Push to `warpfront`, never the fork.

**Goal:** serve `deepgrove/maple-preview` — a 20B-A1B natively-ternary MoE — on
gfx1151, as a new arch crate (`hipfire-arch-maple`, **arch_id 15**), with the
ternary weights carried **losslessly**.

## Why this stacks on #599

Three things this port needs live only on `quant/quality`:

1. **The ternary formats themselves.** `TQ2G128` is qt40 there (renumbered from
   qt38 in #597). Master has no ternary at all.
2. **`quantize_mq2g256_lloyd_k3`** (`crates/hipfire-quantize/src/quant_mq.rs`) —
   the "MQ1.58 probe": a K=3 Lloyd codebook packed into the MQ2-Lloyd container
   with slot 3 duplicating slot 2, explicitly so it "runs on the existing
   MQ2G256Lloyd kernel with NO new kernel". This is arm B's container.
3. **The MoE kernel suite**, including the exact router and activation Maple
   needs (see "What already exists").

#599 also refactored the code this port must touch: `hipfire-quantize/src/main.rs`
is now split into `hfq.rs` / `pipeline.rs` / `pipeline_gguf.rs` / `quant_mq.rs`.
Work written against #597's layout would conflict structurally, not textually.
`quant/quality` contains `feat/ternary-bonsai-27b` outright (verified with
`git merge-base --is-ancestor`), so stacking on #597 would buy nothing and defer
the same merge.

This branch must not modify #599's qt40/qt41/qt44/qt45 internals, nor the
MQ2-Lloyd container layout. Ownership audit, same form #610 used:

```
git diff -G'TQ2G128|BQ1G128|MQ2G256Lloyd|qt.?40|qt.?41|72 B/group' 673ea3ae0 -- crates
```

Expected to show only *additive* dtype arms at review time, no edits to existing
packers or the 72 B/group layout.

## The model

`published` (`config.json`, `architectures: ["MapleForCausalLM"]`,
`model_type: maple`):

- 24 layers, hidden 2048, `intermediate_size 4096` (**unused** — `MapleMLP` is
  only ever constructed with `moe_intermediate_size`).
- GQA 16 heads / 4 KV, `head_dim 128`, `use_qk_norm: true`,
  `partial_rotary_factor 0.5`, θ=10000, `rms_norm_eps 1e-6`.
- **3:1 attention pattern**, `layer_types` = `[S,S,S,G] × 6`, `sliding_window 512`,
  and `nope_on_global_attention: true`.
- MoE on **every** layer: 256 experts, top-8, `moe_intermediate_size 512`,
  `num_shared_experts: 0`, `norm_topk_prob: true`, `router_dtype: fp32`,
  `moe_router_enable_expert_bias: false`.
- vocab 151936 (Qwen tokenizer), bos 151643, eos 151645,
  `tie_word_embeddings: false`, `max_position_embeddings 131072`.
- `quantize: true` and `preaffine: false` — **dead keys**, `measured`: neither
  string appears in `modeling_maple.py`, `configuration_maple.py` or `fa3.py`.
  There is no quantization path in the shipped modeling code to replicate; it is
  plain `nn.Linear` throughout. The ternary structure lives in the *values*, not
  in any code.

`measured` (HTTP range reads of the safetensors headers, no full download):
9 shards, **40.4 GB of BF16**, 1845 tensors in shard 1 alone. The card's
"5.31 GB checkpoint" is not what is published — **the repo is the dequantized
master.** Same pattern as the PrismML unpacked masters; see
`prismml_unpacked_masters_are_dequantized`.

Two details from `modeling_maple.py` that are load-bearing and easy to miss:

- **Clamped SwiGLU**: `silu(clamp(gate, max=7.0)) * clamp(up, min=-7.0, max=7.0)`.
- **RoPE only on sliding layers.** `apply_rotary_pos_emb` is called under
  `if self.sliding_window is not None`. Global layers get no positional signal
  at all. QK-norm is applied *before* RoPE, on both branches.
- Embedding tensor is `model.word_embeddings`, not `embed_tokens`.

## The key property: per-row ternary — `measured`

Every linear weight is exactly `{-s_r, 0, +s_r}` with **one bf16 scale per output
row**. Method: range-fetch the tensor bytes, reinterpret BF16→F32, and for each
row assert `unique(|w|)\{0}` has cardinality 1.

| tensor | rows checked | rows not ternary | nonzero frac | row-scale range |
|---|---:|---:|---:|---|
| `layers.0.mlp.experts.0.gate_proj` | 512 | **0** | 0.613 | 0.0193–0.0366 |
| `layers.0.self_attn.q_proj` | 1953 | **0** | 0.612 | 0.00873–0.137 |
| `layers.0.self_attn.o_proj` | 1953 | **0** | 0.607 | 0.0168–0.248 |

Not ternary, and to be carried at higher precision: `mlp.gate.weight` (router),
`word_embeddings`, `lm_head`, and all norms — each ~1000+ distinct values per
row and 100% nonzero.

**Consequence.** A container with a per-block scale or a per-block codebook can
reproduce these weights *exactly*, because every block within a row sees only
three distinct values. This is not a quantization problem. It is a packing
problem, and the acceptance bar is bit-exactness, not KLD.

All relevant K dimensions (2048 for gate/up/q/k/v/o, 512 for down) are divisible
by both 128 and 256, so neither container needs a tail case.

Parameter accounting (`measured` shapes, arithmetic ours): 815,792,128 ternary
params/layer × 24 = **19.58 B ternary**, plus 622 M for embeddings + lm_head
⇒ ~20.2 B total, ~0.87 B active/token excluding lm_head. Consistent with the
published "20B-A1B".

## Two container arms — B first

Both arms are **bit-exact reconstructions of the same weights**. This is
therefore not a quality experiment; it is a performance and engineering one,
with a free correctness oracle (below).

### Arm B (first) — MQ2-Lloyd container, K=3, no FWHT

Take `quantize_mq2g256_lloyd_k3` and drop the rotation. With only three distinct
values per 256-block, Lloyd converges on them at zero distortion; centroids round
to fp16, and Maple's bf16 `s_r` (0.0087–0.25 measured) is exactly representable
there. Slot 3 duplicates slot 2 and is never indexed.

- 72 B / 256 = 2.25 bpw ⇒ **5.51 GB** for the ternary part.
- **Zero new kernels.** Output DType stays `MQ2G256Lloyd`, which already has
  indexed gate_up, indexed down, batched k4/k8 and grouped GEMM across
  gfx1151 / gfx12 / gfx942 / gfx1030.

The blocker, and the reason this is spiked before anything else is built:
`quantize_mq2g256_lloyd_k3` calls `cpu_fwht_256` **unconditionally**. FWHT is
orthogonal so the *math* is transparent, but it destroys the three-value
structure, which is precisely what makes the K=3 codebook exact. Unrotated
weights then require the runtime to skip `rotate_x_mq` / `rotate_x_mq_batched`
for these tensors.

A per-model seam exists — `MoeBiasAwareMq2Backend::rotate_x_batched`
(`crates/hipfire-dispatch/src/families/moe.rs:433`) is model-owned, so an
identity implementation is expressible. But the generic MoE decode path calls
`gpu.rotate_x_mq_batched` directly (`crates/hipfire-dispatch/src/pipeline/mod.rs:1508`).
So arm B needs either that trait impl or a dtype-gated skip in the generic path.
`quantize_mq2g256_lloyd_no_fwht` already exists in `diagnostics.rs` as the
unrotated packer precedent.

**Spike B0 gates everything:** can an unrotated MQ2-Lloyd tensor be decoded
correctly by the existing kernels? If no, arm B is dead and arm A becomes the
critical path rather than the comparison.

### Arm A — TQ2G128 (qt40)

The format #597/#599 already ship, and the one whose GEMV/GEMM kernels were just
optimised (1.77× decode, 4.1× prefill).

- 34 B / 128 = 2.125 bpw ⇒ **5.20 GB** for the ternary part. Slightly smaller
  than B.
- **But TQ2G128 has no MoE kernels.** Zero hits in `tables/moe_table.rs` and
  `families/moe.rs`, and `MIXED_SUPPORTED_TIERS = [MQ4G256, MQ6G256, ParoQ4G128]`.
  #597's ternary kernels are dense-only. Arm A must add indexed gate_up, indexed
  down, k8-batched variants and a grouped GEMM — mechanical (templates exist for
  six other dtypes) but not small.

Note the storage difference is ~0.3 GB and, per
`hipfire_lowbit_gemv_not_bandwidth_bound`, low-bit decode GEMVs are x-load/ALU
bound rather than weight-bandwidth bound — so bpw is **not** the figure of merit
here. Compare ms/launch.

### Why both — the differential oracle

If both arms are exact, they must emit **identical logits**. Any divergence is a
kernel bug, not a quality difference. That is a far sharper acceptance test than
KLD, and it validates arm A's new kernels against arm B's mature ones. Per
`feedback_assert_the_event_not_the_proxy`, every arm still gets a generation
smoke — a KLD number alone has lied on this exact class of work before.

Embeddings / lm_head / router / norms use the same precision policy in both arms
so they cancel out of the comparison.

## Shared work — arm-independent, and most of the project

**`hipfire-arch-maple`, arch_id 15.** Confirmed free: `MODEL_TYPE_TO_ARCH_ID`
(`crates/hipfire-runtime/src/arch_mapping.rs:28`) tops out at 14 for primaries;
22/23 are drafter sidecars. Add `("maple", 15)`.

**Attention template = `hipfire-arch-muse-glimmer` (14), not cohere2moe.**
Glimmer is already 3:1 sliding/full with **NoPE on the full layers**, head_dim
128, QK-norm and untied lm_head. Maple's deltas: partial rotary 0.5 (Glimmer has
none), plain pre-norm instead of sandwich norm, no logit softcap, no attention
gate, and MoE instead of dense. Cohere2MoE (12) is the fallback reference for
the sliding/global *plumbing* but differs more (parallel block, sigmoid router,
dense layer-0, tied embeddings).

**Convert path.** Stream safetensors shard-by-shard: download, verify per-row
ternary, pack, delete. Peak disk ~11 GB rather than 46 GB — worth it, `/` is at
100% and `/data` has ~101 GB. Must handle 18,432 expert tensors
(256 × 3 × 24). The per-row ternary check should be a **hard gate** that refuses
to write a non-ternary row rather than silently falling back to lossy
quantization — the failure mode we want is a loud refusal, following
`check_ternary_pack_health`'s precedent.

## What already exists (and must not be rewritten)

Maple's MoE decode pipeline is, kernel-for-kernel, already in `quant/quality`:

| Maple needs | Kernel present |
|---|---|
| softmax → top-8 router | `moe_router_softmax_topk_k8_wave64{,_exact}.hip` |
| `norm_topk_prob: true` | `moe_topk_renorm_k8.hip` |
| **clamped** SwiGLU | `moe_unscatter_silu_clamp_k8.hip` |
| scatter / permute / combine | `moe_scatter_*`, `moe_down_combine_*` |

k=8 throughout, which is Maple's `num_experts_per_tok` exactly. The gap is the
weight dtype, nothing else.

## Verification

Following #610's template: `scripts/coherence-gate-maple.sh` +
`_coherence_runner.py`, a `registry_gen.py` entry with a matching test.

1. **Bit-exactness** — packed→dequantized weights compared against the source
   BF16 tensors. Must be exact, not "close". This is the arm gate.
2. **Differential** — arm A vs arm B logits on a fixed prompt set. Must be
   identical.
3. **Reference** — per-layer hidden-state cosine against `modeling_maple.py` on
   CPU, the method that localised the Bonsai double-norm bug to layer 0.
4. **Coherence smoke** — real generation, both arms. Non-negotiable regardless of
   what the numbers say.

## Open questions

- **B0 (blocking):** is unrotated MQ2-Lloyd decodable by the existing kernels?
  Decides whether B is a day of work or a dead end.
- Precision policy for `word_embeddings` / `lm_head` (622 M params). BF16 costs
  1.24 GB; Q8 halves it. Deepgrove's 5.31 GB implies they quantize these; we
  need not match that to ship.
- Does the 131072 context interact badly with SWA-512 + NoPE-global on
  hipfire's KV paths? Glimmer runs SWA 2048; Maple's 512 is tighter.
- 256 experts × 24 layers of *tiny* (512×2048) expert matrices is an unusual
  shape for these kernels; per-launch overhead may dominate regardless of arm.

## Non-goals

- Reproducing Deepgrove's training or their 5.31 GB packing.
- True 1.58-bpw storage. Both arms are ~2.1–2.25 bpw containers; the `_k3`
  comment already flags dense ternary packing as a mechanical follow-up.
- Matching the published "200+ tok/s on M4" claim, which is `published` and
  unverified by us on unrelated hardware.
- Vision, drafter/MTP sidecars, EP/multi-GPU.
