# Plan — shared greedy-accept core + qwen-DFlash relocation + ChainSpeculator + generic guard

**Branch:** `feature/speculator-abstraction` (on top of `c0986b83`)
**Goal:** make the speculative-decode *acceptance rule* a single tested source of
truth across all drafters, move all qwen-specific drafter code into the qwen
crate, and give a generic per-arch slot-guard — so a future llama-DFlash / MTP
builds on the seam with no daemon/loader edits.

## Design (settled with the user after a 4-site accept audit)

The four greedy-accept sites share only a 3-line inner comparison; each wraps it
with genuinely-different concerns (DFlash: sampling/n-gram/repeat; MTP:
EOS-early-stop; deepseek4: per-position grammar mask + matcher advance). The
**precompute-then-match** core avoids a kitchen-sink: each arch computes its own
`target_pick[i]` *before* calling the core; the core does only prefix-match +
EOS-stop + bonus.

```rust
// hipfire-runtime/src/spec.rs
pub struct GreedyAccept { pub committed: Vec<u32>, pub accepted: usize, pub hit_eos: bool }
/// longest i where target_pick[i] == drafts[i]; if eos=Some and an accepted
/// token == eos, stop and skip bonus; else bonus = target_pick[accepted].
pub fn accept_greedy_prefix(drafts: &[u32], target_pick: &[u32], eos: Option<u32>) -> GreedyAccept
```

Captures: DFlash-greedy (`eos=None`), MTP `greedy_trunk_spine_accept` (`eos=Some`),
n-gram (`eos=None`), deepseek4 **non-grammar** path. Stays bespoke: DFlash
`temp>0` rejection-sampling (separate non-greedy fn), deepseek4 **grammar** path
(stateful per-position masking — `target_pick[i+1]` depends on accepted `i`).

## Steps (each → verify)

1. **Accept core** — add `GreedyAccept` + `accept_greedy_prefix` to `spec.rs` with
   unit tests (eos=None full/partial; eos=Some stop-mid-prefix; bonus==eos). →
   verify: `cargo test -p hipfire-runtime spec::`.
2. **n-gram onto core** — `NgramSpeculator::step` uses `accept_greedy_prefix`
   (eos=None). → verify: qwen35-9b + qwen3-0.6b-llama committed-ids byte-identical
   to current (fresh daemon).
3. **DFlash greedy onto core** — in `spec_step_dflash`, the greedy branch builds
   `target_pick` (its argmax + n-gram override) then calls the core; sampling
   branch untouched. → verify: coherence-gate-dflash + 27B byte-identical greedy.
4. **MTP onto core** — replace `greedy_trunk_spine_accept` body with the core
   (eos=Some). → verify: MTP coherence (mtp gate / deepseek4-or-qwen35-MTP run).
5. **deepseek4 non-grammar onto core**; grammar path stays sequential. → verify:
   deepseek4 coherence (no-tools + tool-call).
6. **Unify `SpecStepResult`** — single struct in runtime (`{drafted, accepted,
   bonus, committed}`); qwen35 + deepseek4 lower from it. → verify: build all.
7. **`ChainSpeculator<BlockDrafter>`** — `BlockDrafter::propose(emitted, seed, k)
   -> Vec<u32>`; `NgramSpeculator` becomes a `BlockDrafter`; `ChainSpeculator`
   does prefill/verify_block/accept-core/commit_prefix. → verify: byte-identical
   to step 2.
8. **Move qwen35 DFlash → qwen crate** — `DflashState`, `load_dflash_state`,
   `DflashSpeculator`, `build_dflash_speculator`, `Qwen35SlotGuard` substance →
   `hipfire-arch-qwen35` (types are runtime::dflash + qwen35::speculative, no
   loader types). Loader keeps a thin `ModelState` dispatch. → verify: build +
   serve-multiturn DFlash arm.
9. **Generic `SpecTargetGuard`** — trait in runtime; loader `spec_target_guard()`
   dispatch; daemon drops the `SpecSlotGuard` enum. → verify: serve-multiturn
   (AR+DFlash) + llama n-gram still route.

## Validation (mandatory)
- Greedy byte-identical checks use **fresh daemon** (rebuild daemon + probe — see
  [[coherence-probe-stale-daemon]]).
- `scripts/coherence-gate.sh`, `scripts/coherence-gate-dflash.sh`,
  `scripts/serve-multiturn-gate.sh`. deepseek4 coherence for step 5.
- **NEVER `cargo fmt`** / `fmt-changed.sh` on this long branch — per-file
  `rustfmt --edition 2021 --config skip_children=true` on ONLY edited files;
  do not touch llama.rs (legacy debt). See [[rustfmt-changed-files-only]].

## Risk
Steps 3–5 touch coherence-sensitive spec paths. The precompute-core preserves
each site's exact semantics (the arch still computes target_pick its own way), so
greedy output must stay byte-identical — that's the regression guard. Commit per
step so a regression bisects cleanly.
