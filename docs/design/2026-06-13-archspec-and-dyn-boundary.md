# ArchSpec + the `ArchInstance` dyn boundary

**Date:** 2026-06-13
**Branch:** `feature/paro-transparent-loading`
**Status:** Design / proposal (no code yet, except the CarrierKit prototype tracked separately)
**Scope:** How to minimize the effort of adding a new model architecture to hipfire.

This doc is the written form of a 4-agent review of the unified-loading work on this
branch. It is the source-of-truth reasoning behind todo items **N1–N6 + C1–C3** in the
project memory (`unified-loading-review-todos`).

---

## 1. Context and the question

The branch landed two stacked abstractions that were supposed to make adding an
architecture cheap:

- **`WeightBackend`** (`crates/hipfire-runtime/src/weight_backend.rs`) — hides the quant
  matrix (`HfqBackend`/`ParoBackend`, `proj`/`norm`/`raw_f32`) from per-arch callers.
- **Carrier registry** (`crates/hipfire-loader/src/carriers.rs` + `lib.rs`) — a
  machine-checked, fail-loud dispatch table that replaced the old `match arch_id` ladder.

Both are genuinely good. The disjointness test (`carriers_are_disjoint`), the fail-loud
ambiguity error, and the `WeightSource` orchestrator in `model_load.rs` are the kind of
design we want more of.

**The problem:** these abstractions hide cost at the *caller* but concentrate it into a
few hand-maintained chokepoints. Adding an architecture is still a cross-cutting change
that edits files shared by every other arch. The review converged — independently, across
3 of 4 reviewers — on a single root cause.

---

## 2. The root cause (todo N1)

A causal chain, each link forced by the previous one:

```
forward() is deliberately kept OFF the Architecture trait   (runtime/src/arch.rs:30-46)
        │  (rationale: avoid dyn-dispatch cost in the hot loop)
        ▼
the runtime cannot hold a model opaquely
        ▼
LoadedModel becomes a ~45-field god-struct with ~20 per-arch Option<…> fields
        │   (hipfire-loader/src/lib.rs — deepseek4_weights, minimax_state, lfm2moe_config, …)
        ▼
hipfire-loader now structurally depends on every arch crate
        ▼
daemon.rs (~11k lines) hand-writes ~70 forward-dispatch ladders:
        if let Some(ref mut s) = m.deepseek4_weights { deepseek4::decode_step(s, …) }
```

So the *actual* cost of adding an arch — the part the `hipfire-arch-toy` README hides — is:

1. Add 3–5 `Option<…>` fields to `LoadedModel`.
2. Add a `None` initializer to `skeleton` / `skeleton_pp`.
3. Add a free branch to `unload_model` — **not compiler-enforced**, so forgetting it is a
   silent VRAM leak (this is exactly how **C2** below was found: `dots_ocr_weights` /
   `lfm2moe_weights` / `minimax_weights` appear to free only their `_state`).
4. Add a forward arm at every one of the ~70 daemon dispatch sites the arch participates in
   (decode, prefill, bench, spec-decode, EP).

Items 1–4 are edits to central structures that good design would leave *closed for
modification, open for extension*.

### Why the performance objection is weak

The trait keeps `forward` off itself to avoid dyn-dispatch cost. But dyn cost is
**per-call**, and `forward_step` / `forward_prefill` is called **once per token**. A single
vtable lookup amortized over a full transformer forward — thousands of kernel launches
across the layer stack — is unmeasurable. Nobody is proposing to dyn-dispatch individual
GEMVs; only the top-level entry point. The trait's own evidence for "measurable tok/s loss"
is *inner-loop* (per-op) dispatch, which is a different thing.

### The fix

Introduce an object-safe trait:

```rust
/// Object-safe, hot-path entry points. One vtable lookup per token step.
pub trait ArchInstance: Send {
    fn decode_step(&mut self, ctx: &mut DecodeCtx) -> Result<StepOut, String>;
    fn prefill(&mut self, ctx: &mut PrefillCtx) -> Result<PrefillOut, String>;

    /// Exhaustive, compiler-enforced teardown. Closes the C2 silent-leak class.
    fn free(&mut self, gpu: &mut Gpu);

    // Optional capabilities; default None so non-participating arches opt out cleanly.
    fn as_spec_decode(&mut self) -> Option<&mut dyn SpecDecode> { None }
    fn as_ep(&mut self) -> Option<&mut dyn EpServe> { None }
}
```

Then:

```rust
// before:  a 45-field god-struct with ~20 per-arch Option<…> fields
// after:
pub struct LoadedModel {
    pub model: Box<dyn ArchInstance>,
    pub tokenizer: Tokenizer,
    pub chat_template: Option<&'static str>,
    pub eos: EosSet,
    // … only genuinely shared fields remain
}
```

The ~70 daemon ladders collapse to single calls:

```rust
// before:
if let Some(ref mut s) = m.deepseek4_weights { deepseek4::decode_step(s, &mut ctx)?; }
else if let Some(ref mut s) = m.minimax_weights { minimax::decode_step(s, &mut ctx)?; }
else if /* … 68 more … */

// after:
m.model.decode_step(&mut ctx)?;
```

**Net effect:** adding an arch touches **zero shared files except one `REGISTRY` line.**
`Carrier::load` returns `Box<dyn ArchInstance>`; the loader and daemon never name the arch.

**Effort: L.** This reverses an explicit, commented design decision — the "beyond surgical"
change. Everything else in this doc is smaller and partly enabled by it.

---

## 3. Could a DSL help? — data vs. code (todo N4/N5)

### What is already declarative (and good)

hipfire is ~70% of the way to a data-driven arch model and doesn't advertise it:

| Seam | Where | What it already does |
|------|-------|----------------------|
| `WeightBackend` | `weight_backend.rs` | quant matrix (~25 formats) in one arch-agnostic place |
| `load_layer<B>` | qwen35 `layer_driver.rs` | per-layer weight **table** — `b.proj("self_attn.q_proj", …)` struct literals |
| `Step` op-list | `hipfire-dispatch/.../steps.rs` | the forward pass is **already** an interpreted op-list (`Step::{Gemv, RmsnormAutomatic, Attend}`) with a fusion engine |
| `WeightAugmentor` | `augmentor.rs` | transparent ParoQuant plugin keyed on `QuantConfig` |
| `Architecture` | `arch.rs` | bring-up contract (`config_from_hfq`/`load_weights`/`new_state` + override structs) |

### What is needlessly code

For a transformer *family*, the following are **data** but currently hand-written:

| Concern | Today | Lines/arch | Reducible |
|---------|-------|-----------:|----------:|
| config field → metadata-key map (HFQ **and** safetensors, ×2) | hand-walked `serde_json` | ~250 | ~96% |
| per-layer weight schema | `load_layer` struct literals (already near-data) | ~110 | unify |
| tokenizer / chat-template / skeleton / `pp>1` wiring | `carriers.rs` boilerplate | ~80 | ~90% |
| KV-mode selection ladder | copy-pasted `match` (×4–7) | ~50 | one helper |
| RoPE style / norm convention / `norm_bias` / qk-norm | scattered constants | ~20 | data row |
| dense forward graph | `Step` list — already interpreted | templated | `Forward::DenseTransformer` |

A **dense** transformer arch is ~95% data. A **hybrid/MoE/MLA** arch is ~60% data plus a
bounded set of named, hand-written *blocks*.

### The irreducible core — what stays hand-written

- Novel kernels: DeltaNet recurrence `S_t = decay·S_{t-1} + β·v·kᵀ`
  (`gated_delta_net_*.hip`), MLA, conv1d-in-attention, new WMMA/dot paths. This is the
  FWHT / INT4-native moat — it is HIP, not data.
- Custom forward control flow for hybrids: LA-vs-FA layer scheduling, MoE expert dispatch,
  VL conditioning, spec-decode/DFlash tree logic.
- `new_state` scratch allocation for recurrent/hybrid models.
- Genuinely derived config semantics (per-layer type arrays, derived rotary factors) — a
  small `derive: fn(&mut Config)` hook, not a config row.

**Rule:** *data describes structure and wiring; code implements novel math.*

### DSL spectrum — decision

| Option | Verdict | Why |
|--------|---------|-----|
| (a) Better Rust factoring (CarrierKit, ConfigSchema rows, generalized layer table) | ✅ first step | zero new machinery, kills duplication now, net-negative cost on current arch count |
| (b) In-Rust `ArchSpec` aggregate + generic drivers | ✅ **recommended** | single source of truth per arch; type-checked & inlinable (respects hot-path + no-Python); novel attention as *named blocks* so hybrids fit |
| (c) Macro-DSL (`declare_arch!{…}`) | ❌ | sugar with zero capability gain over a struct; hostile errors; opaque expansion in a repo that bisects perf to a single newline |
| (d) External manifest (TOML/JSON), llama.cpp-style | ❌ | **structurally cannot describe the moat** (FWHT, DeltaNet, MLA, INT4 WMMA); adds runtime string-dispatch in the load path; buys no-recompile flexibility hipfire explicitly does not want (arches ship *with* the engine, perf-validated against specific kernels) |

llama.cpp gets away with a flat manifest because GGUF arches are near-homogeneous dense/MoE
transformers over a fixed kernel library. hipfire's differentiators are exactly the parts a
manifest can't express. Choose (b), reached via the (a) refactors.

---

## 4. The `ArchSpec` sketch (todo N5)

Generic drivers, built **once** in `hipfire-runtime`:

- `interpret_config(schema, source) -> Config` — one parser over the existing `ModelSource`
  abstraction (replaces both `config_from_hfq` and `config_from_safetensors`).
- `interpret_layers(layer_schema, &mut dyn WeightBackend, cfg)` — generalizes today's
  `load_layer`.
- `run_forward(forward_template, ctx)` — feeds the existing `Step` interpreter.
- `CarrierKit` — absorbs tokenizer / template / skeleton / `pp>1` / KV-mode (see §6, the
  prototype).

A **new dense arch** becomes one file:

```rust
pub static SMOLLM: ArchSpec = ArchSpec {
    name: "smollm",
    arch_ids: &[12],
    norm: Norm::Rms { bias: 0.0, qk_norm: false },
    rope: Rope::Llama { theta_key: "rope_theta" },

    // CONFIG: field ← metadata key (+ default). Replaces 2× hand-walked parsers.
    config: &[
        cfg!(dim,        "hidden_size"),
        cfg!(n_layers,   "num_hidden_layers"),
        cfg!(n_heads,    "num_attention_heads"),
        cfg!(n_kv_heads, "num_key_value_heads"),
        cfg!(hidden_dim, "intermediate_size"),
        cfg!(norm_eps,   "rms_norm_eps", default = 1e-5),
        cfg!(vocab_size, "vocab_size"),
    ],

    // LAYER: weight table. Generalizes today's load_layer struct literals.
    layer: LayerSchema::Dense(&[
        slot!(attn_norm, Norm, "input_layernorm.weight",         [dim]),
        slot!(wq,        Proj, "self_attn.q_proj", q_out_dim,     dim),
        slot!(wk,        Proj, "self_attn.k_proj", kv_dim,        dim),
        slot!(wv,        Proj, "self_attn.v_proj", kv_dim,        dim),
        slot!(wo,        Proj, "self_attn.o_proj", dim,           o_in),
        slot!(ffn_norm,  Norm, "post_attention_layernorm.weight", [dim]),
        slot!(w_gate,    Proj, "mlp.gate_proj",    hidden_dim,    dim),
        slot!(w_up,      Proj, "mlp.up_proj",      hidden_dim,    dim),
        slot!(w_down,    Proj, "mlp.down_proj",    dim,           hidden_dim),
    ]),

    // FORWARD: standard dense block — emits the existing Step list. No bespoke code.
    forward: Forward::DenseTransformer,

    kv: KvPolicy::Standard,        // CarrierKit's shared asym3/q8/fwht ladder
    overrides: Overrides { prompt: Raw, ..DEFAULT },
};

// Registration is one line — no Carrier struct, no claims_arch_id/load impl:
register_arch(&SMOLLM);
```

A **hybrid / novel arch** (qwen35-class) uses the *same* spec, swapping the forward and
layer schema to reference hand-written blocks:

```rust
layer: LayerSchema::PerType(&[              // LA vs FA chosen by config.layer_types
    (LayerType::LinearAttention, &DELTANET_SLOTS),
    (LayerType::FullAttention,   &FULL_ATTN_SLOTS),
]),
forward: Forward::Custom(qwen35_forward),   // ← escape hatch: hand-written hybrid graph
blocks:  &[Block::DeltaNet, Block::Moe { experts: 256 }],  // named kernels stay Rust
```

The DeltaNet recurrence, MoE routing, and VL tower stay exactly as hand-written kernels /
closures — the spec only *names and wires* them. Nothing about the moat moves to data;
only the boilerplate around it does. `Forward::Custom` is a first-class citizen, not a
grudging exception — hybrids are hipfire's whole point.

### Payoff

| Component (per dense arch) | Today | Under ArchSpec | Saved |
|---|---:|---:|---:|
| `config_from_hfq` + `config_from_safetensors` | ~250 | ~10 | ~96% |
| Carrier (`claims_arch_id`/`load`/tokenizer/template/skeleton/KV) | ~90 | ~1 + spec fields | ~90% |
| Layer schema | ~110 | ~10 | unified |
| Forward (dense) | ~170 | 0 | ~100% |
| **Dense arch total** | **~620** | **~60** | **~90%** |

Build cost ~1–1.5 weeks. The first two pieces (CarrierKit, ConfigSchema) are
**net-negative cost** on the current 7-carrier set. Full break-even at the 2nd–3rd new
dense arch — a threshold the project roadmap (Qwen2/3/3.5/3.6, Llama, DeepSeek, MiniMax,
LFM2, dots-ocr, "any model") crosses immediately.

---

## 5. The qwen35 33k lines, demystified (todo N6)

Don't read 33k as complexity. Rough budget:

- **~18%** genuinely novel arch logic (DeltaNet, MoE, MLA, hybrid scheduling) — *stays*.
- **~49%** co-located spec-decode / MTP / PFlash / grammar feature stack — **not "the
  architecture"** at all; a new arch needs none of it for a forward pass.
- **~32%** plumbing the trait split was supposed to factor out and didn't.

llama looks "cheap" (396 lines) only because its 8.3k-line body is shelved in
`runtime/llama.rs` — an accounting artifact, not a design win. **N6**: qwen35 hand-rolls a
~2–3k-line `SuperOp` / `lower_variant` / `run_fused_*_key` kernel-lowering layer that llama
**already deleted** by adopting `hipfire-dispatch::execute_steps`. Porting qwen35 to it
removes ~2–3k lines with no behavior change, attacking the 32%.

---

## 6. Sequencing

Ordered by *cost-adjusted value* (do net-negative-cost items first):

| # | Change | Effort | Note |
|---|--------|:------:|------|
| **N2** | **CarrierKit** — collapse the 5 byte-identical non-core carriers into a generic `HfqCarrier{id,name,load_fn}`; extract one `build_kv_cache()` for the ×4–7 KV-mode ladder (which has 3 *disagreeing* defaults) | S–M | **net-negative cost. Prototype first — validates the direction.** |
| **N1** | **`ArchInstance` dyn boundary** — `Box<dyn>` replaces the god-struct, collapse ~70 daemon ladders, exhaustive `free()` closes C2 | L | root cause; highest payoff |
| **N3** | **`QuantCodec` registry** — one data table replaces the 3–4 lockstep `match quant_type` tables; extract `fwht256_inplace` (inlined **6×** in attractor-critical math, `weight_backend.rs:608,689,864,911,956,1032`) | L (+S for fwht) | do the fwht extraction first, independently |
| **N4** | **`ConfigSchema`** rows replace the ×2 hand-walked config parsers | M | |
| **N5** | **`ArchSpec`** aggregate + `Forward::DenseTransformer` over the Step interpreter | M | completes the declarative skin |
| **N6** | port qwen35 to `hipfire-dispatch::execute_steps` | M–L | deletes ~2–3k lines, no behavior change |

**N2 is the validation probe** for this whole direction and is implemented as a prototype
alongside this doc.

---

## 7. Correctness flags found in passing

File these regardless of whether the redesign proceeds:

- **C1** — `derive_arch_id` silently defaults an unknown `model_type` → `arch_id = 5`
  (Qwen35) at `safetensors_source.rs:244-249`. An unrecognized safetensors dir mis-routes
  to `Qwen35Carrier` and dies deep in weight loading with a confusing error instead of a
  clean "no carrier." This punches a hole through the otherwise-robust namespace guard.
  **Fix:** return an explicit unclaimed sentinel.
- **C2** — `unload_model` (`lib.rs:1170-1184`) appears to free only the `_state` for
  `dots_ocr` / `lfm2moe` / `minimax`, not their `_weights` — a possible VRAM leak. N1's
  exhaustive `match`/`free()` closes this whole class.
- **C3** — `bf16_loader.rs` is dead scaffold (`load_bf16_model` = `unimplemented!()`; only
  `is_gptq_target` is live). Inflates the surface under review. Pre-existing — flag, don't
  delete unasked.

---

## 8. Guardrails (project idiom)

- The spec layer is **load-time and config-time only**. It must not touch the forward hot
  path beyond feeding the existing `Step` interpreter, which already runs.
- Keep `Forward::Custom` first-class. A spec that can only express dense transformers would
  be the manifest trap (option d) in a nicer hat.
- Behavior-preserving refactors (N2, N6, the fwht extraction) must produce **byte-identical
  token-id streams** on the coherence-gate models before landing — same bar R2/R3 used.
- N1 and N3 touch dispatch/teardown and the quant cores, so they require the full
  coherence-gate (and cross-arch gates per #397: gfx1201 non-optional) before merge.
