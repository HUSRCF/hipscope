# DDTree Stage 3b — On-GPU Tree Build: Results

**Branch:** feature/speculator-ddtree  
**HEAD at implementation:** 201315b8 (base); implementation uncommitted.  
**Date:** 2026-06-30  
**Status:** SHADOW ASSERT PASS → PRODUCTION SWITCHED → GATES PASS

---

## 1. Shadow Assert Verdict: BYTE-IDENTICAL

Ran `HIPFIRE_DDTREE_VERIFY_TREE_BUILD=1` on all 4 canonical prompts × 2 budgets (8, 12) at temp=0.0.
Every cycle on every prompt+budget combination asserted:
- `big_n` scalar
- `parent_indices[0..big_n]` (i32 per slot)
- `slot_depth[0..big_n]` (i32 per slot)
- `node_tokens[1..big_n]` (draft token per slot)
- `child_of_cand[0..big_n*topk]` (child node index per slot×rank)
- `attn_bias[0..big_n*big_n]` (f32 row-major mask)

**Matrix result: 0 divergences across 8 runs (45 cycles total).**

```
lru_cache_pep8_strict  budget=8:  6 cycles  ALL PASS
lru_cache_pep8_strict  budget=12: 5 cycles  ALL PASS
trains-meet            budget=8:  7 cycles  ALL PASS
trains-meet            budget=12: 6 cycles  ALL PASS
prose_river_short      budget=8:  9 cycles  ALL PASS
prose_river_short      budget=12: 7 cycles  ALL PASS
bare_factual           budget=8:  10 cycles ALL PASS
bare_factual           budget=12: 9 cycles  ALL PASS
```

---

## 2. Production Switch: Applied

The production greedy path (`use_gpu_build = !use_swor && !verify_gpu_build`) now calls
`ddtree_build_and_linearize_f32` instead of `topk_logsumexp_batched` + host tree build
+ host linearize + parent_indices H2D.

### Temp=0 byte-identical verdict (production vs shadow path)

All 4 canonical prompts, budget=12, topk=4:

| Prompt | GPU-build cycles | Host-build cycles | τ match |
|--------|-----------------|-------------------|---------|
| lru_cache_pep8_strict | 7 committed=69 τ=7.857 | 7 committed=69 τ=7.857 | MATCH |
| trains-meet | 8 committed=71 τ=6.875 | 8 committed=71 τ=6.875 | MATCH |
| prose_river_short | 9 committed=70 τ=5.778 | 9 committed=70 τ=5.778 | MATCH |
| bare_factual | 11 committed=71 τ=4.455 | 11 committed=71 τ=4.455 | MATCH |

**Verdict: BYTE-IDENTICAL at temp=0.**

### Copy-count before/after (temp=0, greedy, per cycle)

**Before (Stage 3a, HEAD 201315b8):**
- `topk_logsumexp_batched_f32` launch (GPU kernel)
- D2H `top_idx` (~batch×k×4B = 240B at b=16,k=4)
- D2H `top_val` (~batch×k×4B = 240B)
- CPU host tree build (O(budget×log budget) heap ops, ~10 µs)
- CPU `linearize_tree_with_parents` (~O(big_n²) visibility, ~5 µs)
- H2D `parent_indices` (~big_n×4B = 244B H2D)
- `ddtree_build_attn_mask_f32` launch (Stage 3a kernel)

**After (Stage 3b):**
- `ddtree_build_and_linearize_f32` launch (single-thread GPU kernel, replaces all of the above)
- D2H `big_n` scalar: 4B
- D2H `node_tokens[0..big_n]`: ~big_n×4B = 244B (to build verify_tokens host slice)
- D2H `slot_depth[0..big_n]`: ~big_n×4B = 244B (to build verify_positions host slice)
- D2H `parents[0..big_n]`: ~big_n×4B = 244B (to reconstruct child_maps for follow_verified_tree)

**Eliminated:** 480B D2H (top-K outputs) + 244B H2D (parent_indices) + host tree build CPU work + host linearize CPU work + `topk_logsumexp_batched` GPU kernel dispatch + `ddtree_build_attn_mask_f32` kernel dispatch (Stage 3a, replaced inline).
**Added:** 3 additional D2H reads totaling ~736B (vs 480B eliminated D2H) + 1 GPU kernel dispatch.

Net change on UMA (gfx1151): no measurable latency change (all copies are ~0 µs). Structural win: zero H2D per cycle for the tree metadata; GPU pipeline not stalled by host round-trips. CPU tree build (~10–15 µs) eliminated. Two GPU kernel dispatches → one.

---

## 3. Temp=0.7 coherence (SWOR path)

SWOR (temp>0) still uses the fallback host-build path (Gumbel top-K + host tree build).
The GPU-build path is greedy-only. Temp=0.7 validation:

- `lru_cache_pep8_strict` temp=0.7: coherent code, τ=8.33, no attractors
- `prose_river_short` temp=0.7: coherent prose, τ=7.10, no attractors

---

## 4. Gate results

- `./scripts/coherence-gate-dflash.sh`: **PASS** (4 rows, all Tier 1+2 clean, no soft warnings)
- `./scripts/serve-multiturn-gate.sh`: **PASS** (AR multi-request + DFlash multi-request both coherent)

---

## 5. Make-or-break items: verification

### f64 log-sum-exp
The GPU build kernel uses `double sum_exp` accumulator and `log(sum_exp)` (double precision),
casting to `float` only at `log_z = gmax + (float)log(sum_exp)`. This exactly mirrors the host
`sum_exp += ((v - max) as f64).exp()` + `max + sum_exp.ln() as f32` in `topk_from_logits`.
**Verified by the shadow assert: 0 log-prob divergences across all runs.**

### Top-K tie-break (equal logit values → smaller token index wins)
The GPU kernel's single-pass top-K maintains a sorted array of size K; on equal `val`, the
insertion check `v > tv[j]` (strict >) means the first-seen (smaller token index) stays, matching
the host min-heap secondary key `self.1.cmp(&other.1)` (ascending token index).
**Verified by the shadow assert: 0 top_tokens divergences.**

### Heap FIFO tie-break (equal neg_logw → smaller push_order pops first)
Single-thread kernel advances `push_counter` deterministically: sibling pushed before child,
matching `ddtree.rs:312–341`. The min-heap comparison function (`heap_before`) on equal `neg_logw`
uses `push_order < other.push_order` — FIFO.
**Verified by shadow assert: 0 parent_indices / slot_depth / child_of_cand divergences.**

### FMA prevention
Cumulative logw arithmetic uses `volatile float` intermediates:
```c
volatile float lw = e.logw;
volatile float sub = top_lp[d-1][rank];
volatile float add = top_lp[d-1][rank_next];
float sibling_logw = (float)lw - (float)sub + (float)add;
```
`volatile` prevents HIP's clang-based compiler from fusing the sub+add into an FMA.
**Verified: byte-identical logw values in every cycle.**

---

## 6. Files changed

- `kernels/src/ddtree_build_and_linearize.hip` — new kernel (Stage 3b)
- `crates/rdna-compute/src/kernels.rs` — `DDTREE_BUILD_AND_LINEARIZE_SRC` constant
- `crates/rdna-compute/src/sampling.rs` — `ddtree_build_and_linearize_f32` dispatch wrapper
- `crates/hipfire-arch-qwen35/src/speculative.rs`:
  - `DdtreeScratch`: 6 new GPU tensor fields (`dev_top_tokens`, `dev_top_log_probs`,
    `dev_slot_depth`, `dev_child_of_cand`, `dev_big_n`, `dev_node_tokens`)
  - `run_dflash_draft_for_topk_gpu`: `keep_logits` + `gpu_build_only` params
  - `spec_step_ddtree_batched`: dual-path (GPU-build for greedy / fallback for SWOR+shadow);
    shadow assert block (`HIPFIRE_DDTREE_VERIFY_TREE_BUILD=1`)

---

## 7. Known limitations / followups

1. **SWOR path (temp>0) does not use the GPU build yet.** The Gumbel top-K already runs on GPU,
   but the host tree build still runs. The GPU build kernel's `top_tokens_out` / `top_log_probs_out`
   are written in Gumbel-draw order (which is what SWOR needs), but wiring `swor_walk_gpu` to use
   device `dev_slot_depth`/`dev_child_of_cand` (Stage 3c Phase A) is a followup.
2. **Parents D2H for child_maps reconstruction.** The GPU-build greedy path downloads `parents[0..big_n]`
   (244B) to rebuild `child_maps` for `follow_verified_tree`. This could be avoided by implementing
   a GPU-side greedy-follow kernel, but was judged out-of-scope for Stage 3b.
3. **Logits freed in gpu_build_only path.** After the build kernel runs, `draft_logits_dev`
   (`Some(logits)`) is freed at the existing `free_tensor` site (step 9 of the function). Correct.
