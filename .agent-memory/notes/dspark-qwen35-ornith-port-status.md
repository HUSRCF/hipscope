---
title: DSpark→qwen35 (ORNITH-35B) — QUANTIZATION DONE (A0 expert-fusion + drafter sidecar committed; target runs coherently); engine port B/C/D/E/F remains
date: 2026-07-03
tags: [dspark,spec-decode,qwen35,arch6,moe,quantize,gate_up_proj,eagle3,ornith,blocker,feature-dspark-qwen35]
---

## STATUS 2026-07-03 — quantization COMPLETE, engine port NOT started
- **A0 (expert fusion) DONE + committed `54e99d9d`**: arch-6 quantizer now fuses pre-split
  `experts.{N}.gate_proj`+`up_proj`→`gate_up_proj` (gate||up, [1024,2048]). `ornith-35b-aeon.mq6`
  (27.7GB, text-only, 10240 experts fused) LOADS + coherence probe 0-hard/0-soft 8/8.
- **A (drafter sidecar) DONE + committed `eb66c4f1`**: `qwen3-dspark-q8` generalized to
  DSparkDraftModel/speculators-v0.6.0 (nested `transformer_layer_config`, `aux_hidden_state_layer_ids`,
  d2t/t2d→F32, full `dspark_*` metadata). `ornith-35b-aeon-dspark.mq6` (1.5GB, 44 tensors).
  Alias `--format qwen35-dspark-q8`. Both models on disk under `~/.hipfire/models/`.
- **REMAINS (engine, none started, coupled — none independently testable):**
  D (qwen35 SpecTarget capture hooks — the crux, DeltaNet state + EAGLE-3 hidden capture, #462 hazard),
  B (dspark_core reduced-vocab d2t remap), C (drafter forward config-driven dims hd256/pr0.25/2048/3L/3-target),
  E (qwen35 carrier DSpark arm, precedence DSpark>DFlash>MTP>ngram), F (parity + coherence + serve-multiturn gates).
  Sidecar discovery name = `<target-stem>-dspark.<ext>` (⇒ `ornith-35b-aeon-dspark.mq6`).

## Scope (branch feature/dspark-qwen35, off feature/dspark-qwen3/PR#492)
Port DSpark spec-decode to **qwen35 MoE arch_id 6** (the DeltaNet-hybrid crate), target =
`pablogrant/ORNITH-1.0_35B_AEON_*` (a Qwen3_5MoeForConditionalGeneration **VL** finetune:
`model.language_model.*` text + `model.visual.*`; 40L linear×3→full every 4th, 256 exp/top-8,
hd256, partial-rope 0.25). Draft = `*_DSPARK-DRAFT_BF16` = EAGLE-3 DSpark head: **3-layer dense
qwen3**, fuses target hidden [9,19,29] (`fc`[2048,6144]), block_size 8, **reduced 32k draft vocab
(d2t/t2d)**, vanilla markov rank256 + confidence head. val: accept≈0.275, accept_len≈1.67 (MODEST →
small adaptive block, modest τ). Full sketch: `docs/design/2026-07-03-qwen35-dspark-port.md`.

## KEY FINDING: the hard parts are already done by PR#492 + qwen35
PR#492 generalized DSpark into arch-agnostic `dspark_core.rs` (DsparkBody trait, τ-block controller,
kernels, MtpDrafter/generate_spec) + ported to qwen3-**dense**/llama. qwen35 already has DeltaNet
snapshot/rewind + verify_block/commit (the hard recurrent-state part). Port = target-side: 3 capture
hooks on `impl SpecTarget for ModelSlot` (qwen35/spec_impl.rs, missing→Err defaults), carrier arm,
quantizer arm. One genuinely-NEW core piece: **reduced-vocab d2t remap (NOT plumbed anywhere today** —
#492's drafter used full vocab). Drafter forward = reuse `Qwen3DsparkBody`/`dspark_qwen3_block_forward`
but make dims config-driven (this head hd256/2048/3L/3-target vs #492 hd128/4096/5L/5-target).

## BLOCKER (Task A0, found by validating the mq6 — validation earned its keep)
`--format mq6` on ORNITH produced a clean 27.7GB mq6 that **panics on load**:
`tensor not found: layers.0.mlp.experts.0.gate_up_proj.weight`. ORNITH stores experts **un-stacked**
(separate `experts.{N}.{gate,up,down}_proj`, DeepSeek-V4-style). The arch-6 quantizer
(main.rs:7333/7360 "split 3D expert tensors per-expert") assumes Qwen3.5 **canonical stacked-3D**
`experts.gate_up_proj` and has NO branch to fuse separate gate+up → the loader's per-expert
`experts.{X}.gate_up_proj.weight` (qwen35.rs:508-510). Experts fell to generic 2D path → unloadable.
**Fix:** arch-6 ingest, detect pre-split, **vstack gate-then-up** ([2·inter,hidden], order load-bearing:
`silu(gu[:inter])*gu[inter:]`; swap = silent lobotomy, coherence gate catches), emit `gate_up_proj`.
Blocks even plain-AR ORNITH. Broken mq6 still at `~/.hipfire/models/ornith-35b-aeon.mq6` (delete+re-quant
after A0). Vision skipped (text-only; `--include-vision` if VL wanted).

## TRAPS
- `tail -N` on a quant/build cmd DROPS the early arch-detection banner from the captured log.
- Drafter (Q8/F16, user decision) sidecar deferred to task A — nothing consumes it until loader exists.
- Disk tight (1.8T @94%); freed 52G by pruning re-downloadable Qwen3.5-27B hf-cache. Related [[dspark-tau-adaptive-block-modulation-resume]].
