# Speculative decoding in hipfire — the DFlash path

hipfire 0.1.6 introduces speculative decoding via the **DFlash** block-
diffusion drafter (arXiv:2602.06036, z-lab/dflash). The draft predicts
the next `B` tokens in a SINGLE bidirectional forward pass instead of
`B` sequential autoregressive calls, then the target verifies them in
one batched sweep and accepts the leading prefix that matches its own
greedy argmax.

This document covers:

- [What it does and why](#what-it-does)
- [When it helps (and when it doesn't)](#when-it-helps)
- [How to enable it](#how-to-enable)
- [The algorithm](#the-algorithm)
- [Known limitations in 0.1.6](#known-limitations)

## What it does

Regular greedy decode samples one token per target forward pass:

```
target.forward(t₀) → argmax → t₁
target.forward(t₁) → argmax → t₂
...
```

Decode throughput is bounded by one forward per token — on a 9B model
reading MQ4 weights, that's ~132 tok/s on a 7900 XTX (~650 GiB/s of
effective weight bandwidth).

DFlash speculative decoding runs a small, bidirectional *draft model*
over a block of `B` positions. The draft takes the last committed
token plus `B-1` `<|mask|>` tokens and refines them to concrete
predictions in one shot:

```
block = [t₀, MASK, MASK, ..., MASK]            (length B)
draft.forward(block, target_hidden) → B-1 predictions for the masks
block = [t₀, d₁, d₂, ..., d_{B-1}]
target.forward_batch(block) → B logits + B hidden states
posterior = argmax per position
accept_len = longest prefix where block[i+1] == posterior[i]
commit = block[0..accept_len+1] + [posterior[accept_len]]
advance position by accept_len + 1
```

The greedy invariant is preserved: every committed token is either a
draft prediction that exactly matched the target's argmax at that
position, or the target's own argmax itself. Output is byte-equivalent
to greedy decode on the target alone; the only difference is that
multiple tokens may commit per target forward.

## When it helps

Speculative decoding amortizes one target forward across multiple
committed tokens. The speedup depends on the **acceptance rate τ** —
the mean number of draft tokens accepted per cycle:

```
speedup ≈ (1 + τ) × (draft_forward_cost + target_forward_cost) / target_forward_cost
```

For a 4B target with a 0.5B draft, if τ = 3 and draft is 10× cheaper
than target, the speedup is roughly:

- Classical decode: 1 target forward / 1 token = 1.0
- Spec with τ=3: 1 target + 1 draft / 4 tokens = (1 + 0.1) / 4 = 0.275
- Speedup = 1.0 / 0.275 ≈ 3.6×

Real speedups depend on domain. From the DFlash paper:

- **Code generation**: high τ (5-8), 3-4× speedup common.
- **Math reasoning**: moderate τ (3-4), 2-3× speedup.
- **Creative / conversational**: lower τ (1-2), 1.3-1.7× speedup.
- **Short responses**: no win — fixed draft/verify overhead amortizes
  over too few tokens.

## When it doesn't help

- **High-temperature sampling**: temp > 0 needs rejection sampling
  which is deferred to 0.1.7.
- **Very short outputs**: overhead dominates.
- **Unfamiliar domains**: if the draft is out-of-distribution, τ
  drops toward 0.
- **Memory-bound small cards** (<16 GB VRAM): draft + target weights
  compete for limited memory. 0.1.6 disables dflash on these cards
  by default (policy TBD in 0.1.7 — for now, don't enable manually).

## How to enable

**0.1.6 status:** runnable as an example binary, not yet wired into
`hipfire run` / `hipfire serve`. To try it:

```bash
# 1. Download + convert a DFlash draft (1-shot, outputs .hfq):
hf download z-lab/Qwen3.5-4B-DFlash --local-dir /tmp/qwen35-4b-dflash-src
cargo build --release -p hipfire-quantize --bin dflash_convert
./target/release/dflash_convert --input /tmp/qwen35-4b-dflash-src \
                                --output ~/.hipfire/models/qwen3.5-4b.dflash.hfq

# 2. Run the spec-decode demo against an existing target:
cargo build --release --example dflash_spec_demo -p engine --features deltanet
./target/release/examples/dflash_spec_demo \
    --target ~/.hipfire/models/qwen3.5-4b.mq4 \
    --draft  ~/.hipfire/models/qwen3.5-4b.dflash.hfq \
    --prompt "The quick brown fox" --max 32 --ctx 128
```

Output includes accept rate, τ, and tok/s.

**0.1.7 plan:** `hipfire pull qwen3.5:9b --dflash` downloads target +
draft as a bundled tag. `hipfire run qwen3.5:9b --dflash` turns on
speculative decode automatically. `hipfire config set dflash auto`
enables spec decode when a draft is available, transparently falling
back to standard decode when not.

## The algorithm

hipfire's dflash implementation in three files:

- `crates/engine/src/dflash.rs` — draft forward pass (bidirectional
  cross-attention Qwen3, 5 layers, B-way parallel over masks)
- `crates/engine/src/speculative.rs` — orchestrator
  (`spec_step_dflash`), target-side verify (`verify_dflash_block`),
  DeltaNet snapshot/restore for state rewind
- `kernels/src/attention_dflash.hip` — new non-causal GQA attention
  kernel (Q length B, K/V length `ctx + B`)

Per iteration:

1. **Build block.** `block[0]` = previously-committed token; `block[1..B]`
   = `<|mask|>` token ID (248070 for Qwen3.5).
2. **Embed block on target.** `target.embed_tokens(block)` →
   `noise_embedding: [B, hidden]`.
3. **Draft forward.** The draft runs its 5 layers with
   per-layer cross-attention over projected `target_hidden`. Attention
   is non-causal: each mask position sees all other positions +
   all context positions. Output: `[B, hidden]`.
4. **Sample masks.** Apply `target.lm_head` to the last B-1 rows of
   draft hidden. Argmax → `block[1..B]`.
5. **Target verify.** Forward the filled block one position at a time
   through `forward_scratch_with_hidden`, writing B new hiddens to the
   ring buffer. Collect per-position logits, argmax → `posterior[0..B-1]`.
6. **Acceptance.** `accept_len = len of prefix where block[i+1] ==
   posterior[i]`. `bonus = posterior[accept_len]`.
7. **Commit.** `committed = [seed_token, block[1..accept_len+1],
   bonus]`. Caller emits `committed[1..]`.
8. **Rewind.** Restore DeltaNet state from pre-verify snapshot; replay
   `accept_len + 1` target forwards to land state at
   `position + accept_len + 1`.
9. **Append target_hidden.** The first `accept_len + 1` rows from the
   ring buffer go into the caller's cumulative `target_hidden_host`
   for the next iteration's draft input.
10. `position += accept_len + 1`; `seed = bonus`; loop.

## Known limitations

### Correctness

- Greedy parity (temp=0): ✓ — every committed token is argmax on the
  target, same as non-dflash.
- Temp > 0: ✗ — rejection sampling is 0.1.7.

### Performance (0.1.6 is MVP, not optimized)

- Draft weights stored F16 on disk, dequantized to F32 on load → ~4 GB
  of VRAM for the 1B 9B-draft. MQ4 draft quant brings this to ~0.6 GB
  but wasn't validated for 0.1.6.
- Draft attention + projections run on F32 GEMM kernels. WMMA / MQ4
  fast paths exist for the target but not for the draft.
- Target verify is a per-token loop — `forward_prefill_batch` exists
  but doesn't extract hidden states; Phase 4 MVP falls back to per-
  token `forward_scratch_with_hidden`. A batched-with-hidden variant
  would collapse B target forwards into one.
- Draft's `lm_head` is B-1 sequential GEMVs against the target's
  output matrix. A single B-wide GEMM would be 10-20× faster.

### Expected 0.1.6 numbers

On 7900 XTX, 4B target + 4B-DFlash draft, debug/preview state:

- Warm-cache pangram prompt: 30 tok/s, τ=3.5, accept_rate 23%.
- Warm-cache fiction prompt: 15 tok/s, τ=1.1, accept_rate 7%.
- Non-spec baseline (decode-only 4B MQ4 on same card): 180 tok/s.

The baseline is faster than spec decode in 0.1.6 because the
per-iteration overhead (F32 draft + per-token verify + per-row draft
lm_head) is larger than the target forward it replaces at these accept
rates. **0.1.6 ships the loop, not the speedup.** The speedup numbers
in the DFlash paper and our target bench table are 0.1.7+ work.

### API stability

The dflash-specific APIs (`dflash::draft_forward`, `speculative::
spec_step_dflash`, `DflashConfig::from_hfq`, HFQ arch_id=20) are
**experimental** in 0.1.6. Expect signature changes in 0.1.7 as we
add batched paths and MQ4 draft quant.
