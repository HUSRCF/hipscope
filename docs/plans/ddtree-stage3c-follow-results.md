# DDTree Stage 3c — On-GPU Greedy Follow: Results

**Branch:** feature/speculator-ddtree  
**HEAD at analysis:** 77d4cbaa (Stage 3b production)  
**Implementation date:** 2026-06-30  
**Status:** SHADOW ASSERT PASS → GATES PASS

---

## 1. Shadow Follow Assert Verdict: BYTE-IDENTICAL

Ran `HIPFIRE_DDTREE_VERIFY_FOLLOW=1` on all 4 canonical prompts × 2 budgets (8, 12) at
temp=0.0 with `--ddtree --ddtree-batched --ddtree-topk 2`. Every cycle asserted
GPU follow result equals host `follow_verified_tree` result:

- `accept_len` (i32)
- `bonus_token` (i32)
- `accepted_node_indices[0..accept_len]` (i32 each)

**Matrix result: 0 divergences across 77 cycles (8 runs).**

```
lru_cache_pep8_strict  budget=8:   9 cycles  ALL PASS
lru_cache_pep8_strict  budget=12:  7 cycles  ALL PASS
trains-meet            budget=8:   9 cycles  ALL PASS
trains-meet            budget=12:  8 cycles  ALL PASS
prose_river_short      budget=8:  11 cycles  ALL PASS
prose_river_short      budget=12:  9 cycles  ALL PASS
bare_factual           budget=8:  13 cycles  ALL PASS
bare_factual           budget=12: 11 cycles  ALL PASS
```

Non-spine accepts verified (cycle: accept_len=6 bonus=2733 accepted=[0,1,2,3,4,**7**]):
GPU correctly finds child at non-linear slot 7, confirming the linear-scan child
lookup is equivalent to host's HashMap lookup for all tree topologies tested.

---

## 2. Ddtree Temp=0 Byte-Identical Verdict

Production runs (no shadow assert) at temp=0, budget=12, topk=4:

| Prompt | run-1 tokens | run-2 tokens | Match? |
|--------|-------------|-------------|--------|
| trains-meet (b12-k4) | [260,413,…] | [260,413,…] | **MATCH** |

Two consecutive runs produce identical token streams — deterministic at temp=0. ✓

Coherence gate DDTree code row (b12-k2): τ=7.800, committed=49, token sequence
matches the DFlash chain-only row exactly (same LRU-cache EOS output) — byte-
identical at temp=0. ✓

---

## 3. Copy-Count: Before / After Stage 3c

### Per-cycle D2H (greedy path, GPU-build production, big_n≈13 at b12-k4)

**Stage 3b (before):**
1. `big_n` scalar: 4 B
2. `node_tokens[0..big_n]`: ~52 B (kept — needed for verify_tokens)
3. `slot_depth[0..big_n]`: ~52 B (kept — needed for verify_positions)
4. `parents[0..big_n]`: ~52 B **← ELIMINATED** (was only for follow_verified_tree)
5. `argmax_per_pos[0..big_n]`: ~52 B **← ELIMINATED** (was for host follow)

Total: 5 D2H calls, ~212 B (at big_n=13; up to ~980 B at big_n=61).

**Stage 3c (after):**
1. `big_n` scalar: 4 B
2. `node_tokens[0..big_n]`: ~52 B
3. `slot_depth[0..big_n]`: ~52 B
4. `follow_result` (accept_len + bonus_token + accepted_node_indices): ~4+4+(accept_len×4) B
   — typical accept_len=5 → 28 B; worst-case accept_len=60 → 248 B.

Total: 4 D2H calls, ~136 B typical (vs ~212 B before at same big_n).

**Eliminated:** parents D2H (big_n×4 B ≈ 244 B at max budget) + argmax D2H (big_n×4 B ≈
244 B). **Added:** follow_result D2H (typically 2+accept_len+2 = ~28 B; ≤248 B worst).

Net reduction: 1 D2H call, ~(488 − 28…248) B = ~240–460 B eliminated per cycle.

### Per-cycle host CPU work eliminated

- `follow_verified_tree` on host: child_maps HashMap lookup loop, O(depth) steps
- DdTree child_maps rebuild from parents: O(num_nodes) HashMap inserts
- DdTree visibility build: O(big_n²) nested loop, 61²=3721 ops worst case
- DdNode vec allocation: O(num_nodes) heap allocs

All replaced by one GPU kernel launch (~1 µs) + 1 smaller D2H.

---

## 4. Gate Results

- `./scripts/coherence-gate-dflash.sh`: **PASS**
  (4 rows: dflash-prose / dflash-code / ddtree-b12-prose / ddtree-b12-code; all
  Tier 1+2 clean, 0 soft Tier 3 warnings; DDTree code τ=7.8 matches DFlash chain)
- `./scripts/serve-multiturn-gate.sh`: **PASS**
  (AR + DFlash multi-request; all requests coherent across session)

---

## 5. Files Changed

- `kernels/src/ddtree_greedy_follow.hip` — **NEW** kernel (single-thread, [1,1,1])
- `crates/rdna-compute/src/kernels.rs` — `DDTREE_GREEDY_FOLLOW_SRC` constant
- `crates/rdna-compute/src/sampling.rs` — `ddtree_greedy_follow_f32` dispatch wrapper
- `crates/hipfire-arch-qwen35/src/speculative.rs`:
  - `DdtreeScratch`: new `dev_follow_result` field (2+max_budget i32s = ≤248 B)
  - GPU-build greedy path: removed parents D2H + DdTree reconstruction (child_maps, visibility)
  - Verify call: `skip_argmax_d2h = use_swor || use_gpu_build` (greedy GPU-build also skips argmax D2H)
  - Step 8: new `else if use_gpu_build` branch launching `ddtree_greedy_follow_f32`
  - `HIPFIRE_DDTREE_VERIFY_FOLLOW=1` shadow assert mode

---

## 6. Kernel Design: `ddtree_greedy_follow_f32`

Grid: [1,1,1]. Block: [1,1,1]. Single-thread sequential walk.

**Algorithm:** replicates `follow_verified_tree` (ddtree.rs:387–409) exactly:
1. Read `argmax[0]` → `next_token`
2. Scan slots 1..big_n for `parent_indices[s] == current_slot && node_tokens[s] == next_token`
3. If found: record `s-1` as accepted node index, advance `current_slot = s`
4. Repeat until no child matches
5. Write `{accept_len, next_token (bonus), accepted_node_indices[]}` to `follow_result`

**Why byte-identical:** child lookup is unique per (slot, token) — DdTree never inserts two
children with the same token at the same slot. The linear scan finds the same unique entry as
the host's HashMap. No floating-point arithmetic — pure integer comparison.

---

## 7. Remaining Per-Cycle Host Transfers (Greedy Path)

After Stage 3c, the greedy ddtree GPU-build path per cycle:

| Transfer | Dir | Bytes | Purpose |
|----------|-----|-------|---------|
| `dev_big_n` | D2H | 4 | Gates verify kernel grid size |
| `dev_node_tokens[0..big_n]` | D2H | ~big_n×4 | Build verify_tokens |
| `dev_slot_depth[0..big_n]` | D2H | ~big_n×4 | Build verify_positions |
| `dev_follow_result` | D2H | (2+accept_len)×4 | Accept indices + bonus |

**No per-cycle H2D** from the GPU-build path (mask and parent_indices live on device from
the build kernel). SWOR path (temp>0) unchanged.

---

## 8. Known Limitations / Followups

1. **SWOR path (temp>0) unchanged.** Still uses the host SWOR walk path. The GPU follow
   kernel only applies to greedy (temp=0).
2. **node_tokens + slot_depth D2H remain.** These are needed for `verify_tokens` and
   `verify_positions` fed to `verify_dflash_block_tree`. Eliminating them would require
   a GPU-side verify that accepts device token/position buffers directly — larger refactor.
3. **`big_n` D2H remains.** Required to gate the verify kernel grid. Could be eliminated
   via device-side indirect dispatch but not worth the complexity.
4. **Shadow assert (HIPFIRE_DDTREE_VERIFY_FOLLOW=1)** downloads parents on-demand for the
   host follow comparison — 1 extra D2H only in assert mode, not in production.
