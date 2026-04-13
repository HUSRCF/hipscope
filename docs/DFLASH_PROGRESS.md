# DFlash Overnight Progress Log

Session start: 2026-04-13, branch `dflash` at commit `6a8859c`.

User review doc: read this top-to-bottom before the code. Phase sections are
added as each phase starts. Commits push to `origin/dflash` only.

## Session plan

- Phase 1 — architecture scope check (HARD GATE → go/no-go)
- Phase 2 — draft weight converter (if go)
- Phase 3 — draft forward pass (native Rust+HIP)
- Phase 4 — batched verification (target side, most exists)
- Phase 5 — speculative daemon loop
- Phase 6 — CLI / serve integration
- Phase 7 — quantization + HF shipping
- Phase 8 — benchmarks + docs

Contract: `docs/DFLASH_OVERNIGHT_AUTONOMY.md`. Master plan: `docs/DFLASH_PORT_PLAN.md`.

## Phase 1 — architecture scope check

- Started: 2026-04-13 (session open)
- Actions completed:
  - Cloned `z-lab/dflash` → `.dflash-reference/`, added to `.gitignore`.
  - Downloaded full `z-lab/Qwen3.5-9B-DFlash` (config.json, dflash.py,
    model.safetensors 2.1 GB) to `.dflash-ref-hf/`.
  - Read reference `dflash/model.py` (338 LOC) and `dflash/benchmark.py`.
  - Read z-lab blog post + arXiv:2602.06036 abstract (full PDF not
    scraped; abstract + blog + reference agree, scope is clear).
  - Enumerated all 58 BF16 tensors in `model.safetensors` via
    `safetensors.safe_open`.
  - Mapped hipfire's existing `speculative.rs` (610 LOC) +
    `qwen35.rs` (2538 LOC) forward variants.
- **Critical finding:** "block diffusion" = single-pass masked token
  infilling, NOT iterative denoising. No noise schedule. Draft runs ONE
  forward per block and fills `B=16` mask slots at once. The word
  "diffusion" is nomenclature, not algorithm.
- **Prior scaffolding discovered:** `crates/engine/src/speculative.rs`
  already contains `ModelSlot`, `SpecPair`, `HiddenStateRingBuffer`,
  `dflash_extract_layer_ids`, and `spec_step_greedy` (classic
  Leviathan). `qwen35.rs` has `forward_scratch_with_hidden` (single-
  token target forward with hidden extract into the ring buffer). All
  committed on master as Phase 1-3 of a prior-session spec-decode
  series.
- **Scope estimate:** ~9 hours of work to MVP. Aggressive cuts to
  Phase 6-8 (CLI + docs + bench minimum) keeps it in budget.
- **Decision: GO.** Proceeding to Phase 2 (weight converter).
- Deliverable: `docs/DFLASH_ARCHITECTURE.md` committed.
- Completed: 2026-04-13 Phase 1 end.

## Phase 2 — draft weight converter

- Goal: write `crates/hipfire-quantize/src/bin/dflash_convert.rs` that
  reads `model.safetensors` + `config.json` and produces a `.hfq` file
  with a `dflash` header section and 58 renamed tensors (all BF16).
- Status: queued next.
