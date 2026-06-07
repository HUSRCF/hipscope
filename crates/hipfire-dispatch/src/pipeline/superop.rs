// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Forward-as-pipeline (#397 Ship 6) — the **C-design lowered super-op
//! substrate**.
//!
//! A model's per-layer forward is lowered ONCE at model load into a
//! [`LoweredForward`] (a `Vec<LayerProgram>`); each [`LayerProgram`] is a short
//! list of COARSE [`SuperOp`]s that map 1:1 onto the existing fused kernels.
//! Per-token execution is then a tight loop over pre-resolved super-ops — no
//! `resolve()`, no `FUSED_TABLE`/`match_prefix` walk, no per-call `WeightRef`
//! construction. The fusion decision (which `FUSED_TABLE` entry fires) and the
//! kernel `KernelKey` are resolved at LOWER time and frozen in [`OpBinding`].
//!
//! ## Why super-ops are POD (no lifetimes / no raw-ptr capture)
//! Design B (a `Box<dyn Fn(&mut Gpu, &Frame)>` per op) was rejected: collapsing
//! the today-disjoint `&mut Gpu / &mut KvCache / &mut DeltaNetState / &Scratch`
//! args into one `Frame` forces `RefCell` (hot-path borrow checks = perf loss)
//! or `UnsafeCell` (the a9e8dfda aliasing-bug class, minus the compiler). So a
//! `SuperOp` carries only **indices** ([`WeightSlot`]/[`ScratchSlot`]) + a
//! resolved [`KernelKey`] + flavor data — pure POD. The per-token executor
//! (built in a later step) re-borrows the live `GpuTensor`s from the model's
//! weight/scratch/state tables BY INDEX and calls the resolved family method,
//! so the compiler still proves disjointness at each call site.
//!
//! ## Coverage (the whole served fleet, one substrate)
//! - `Proj` / `ResidualGemv` / `Moe`     → qwen35, MiniMax(reuse), cohere2moe(reuse)
//! - `Attend` (flavor-carrying)          → all; Gemma SWA/qk-norm/softcap/k_eq_v live here as flavors
//! - `Recurrent`                         → qwen35 DeltaNet linear-attention state
//! - `Conv`                              → LFM2 depthwise causal short-conv mixer (+ conv state)
//! - `Escape(EscapeKind)`                → irregular/stateful: deepseek4 compressor/indexer/SWA, etc.
//!
//! NOTE: nothing here is on a live path yet. The live forward remains
//! `execute_steps`; this substrate is wired behind `HIPFIRE_FORWARD_LOWERED`
//! (default off) in later steps, validated byte-identical via the
//! `HIPFIRE_FORWARD_ORACLE` dual-run.

use crate::types::KernelKey;

/// Index into the model's per-layer weight table (resolved at lower time, stable
/// for the model's lifetime). The executor maps this to the live `&GpuTensor`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightSlot(pub u32);

/// Typed handle into the live, per-token scratch/state/cache buffers. Kept TYPED
/// (not a bare index) so an activation buffer can never be confused with an
/// advancing-state or KV-cache buffer — the spike's #30/a9e8dfda-class
/// (stateful-rebind silent-wrong-output) mitigation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScratchSlot {
    /// Transient per-token activation (hidden state, rotated x, gate/up bufs…).
    Activation(u32),
    /// Per-token-ADVANCING recurrent state (DeltaNet double-buffer, LFM2 conv
    /// state). Rebind MUST recompute exactly where the hand path does.
    State(u32),
    /// KV-cache buffer + write-offset (advancing). Same rebind-fragility class.
    Cache(u32),
}

/// FFN/gate-up activation flavor. SiLU for qwen-family; GeLU-tanh (GeGLU) for
/// Gemma (`gelu_tanh(gate)·up`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActFlavor {
    SiluMul,
    GeluTanhMul,
}

/// RoPE flavor carried by an `Attend` super-op (resolved from config at load).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeFlavor {
    None,
    /// Standard rotate-half (most archs). `theta` = rope base.
    HalfRotate { theta: f32 },
    /// Interleaved full-dim RoPE (e.g. cohere2moe).
    Interleaved { theta: f32 },
}

/// Attention-block flavor — everything that distinguishes one arch's attention
/// from another, resolved at load so the per-token `Attend` is branch-free.
/// Gemma exercises the full surface (SWA window, per-head qk-norm, q·√hd scaling,
/// the k_eq_v weightless-V-RMSNorm prelude, logit softcap).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnFlavor {
    /// Sliding-window size; 0 = full (global) attention. Gemma alternates
    /// Sliding(1024)/Full per layer.
    pub window: u32,
    /// Per-head q_norm/k_norm over head_dim (Gemma, qwen35).
    pub qk_norm: bool,
    /// q *= sqrt(head_dim) (Gemma query scaling).
    pub q_scale_sqrt_hd: bool,
    /// V = copy of K before k_norm + weightless RMSNorm on V (Gemma full layers).
    pub k_eq_v: bool,
    /// Attention-logit softcap value; 0.0 = none (Gemma-2 style).
    pub logit_softcap: f32,
    pub rope: RopeFlavor,
}

/// Per-super-op flavor payload (None for ops with no flavor axis, e.g. Proj).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpFlavor {
    None,
    Attn(AttnFlavor),
    Act(ActFlavor),
}

/// Irregular/stateful ops that don't map onto a single fused kernel. Each is a
/// typed tag the executor matches to a bespoke `gpu.*` sequence (NOT dyn-trait).
/// Extensible: a new irregular arch adds a variant + an executor arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EscapeKind {
    /// deepseek4 MLA compressor (called twice/layer: main + indexer sub-compressor).
    Deepseek4Compressor,
    /// deepseek4 indexer top-K selection.
    Deepseek4IndexerTopK,
    /// deepseek4 sparse SWA over the gathered top-K KV.
    Deepseek4SwaTopK,
    /// Gemma final logit softcap (output stage).
    GemmaLogitSoftcap,
}

/// One coarse super-op. For `Proj`/`ResidualGemv`/`Moe` the `key` is the
/// FUSED_TABLE/`resolve()` result frozen at lower time; for `Attend`/`Recurrent`/
/// `Conv`/`Escape` it may be `None` (those route by kind + flavor + escape tag).
#[derive(Clone, Debug)]
pub struct SuperOp {
    pub kind: SuperOpKind,
    pub binding: OpBinding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuperOpKind {
    /// Fused projection cluster (QKV / QKVZA / gate+up / plain GEMM).
    Proj,
    /// Output/down projection with fused residual add (o_proj, down_proj).
    ResidualGemv,
    /// Attention block (flavor in `OpFlavor::Attn`).
    Attend,
    /// MoE FFN block (routes to MoeFamily / run_moe_decode).
    Moe,
    /// Recurrent linear-attention state advance (DeltaNet).
    Recurrent,
    /// Depthwise causal short-conv mixer with advancing conv state (LFM2).
    Conv,
    /// Bespoke irregular/stateful op.
    Escape(EscapeKind),
}

/// Pre-resolved binding for one super-op. Pure POD (indices + key + flavor) — no
/// borrows, no raw pointers; the executor binds against live state by index.
#[derive(Clone, Debug)]
pub struct OpBinding {
    /// Kernel resolved at lower time (FUSED_TABLE/resolve result). `None` for
    /// ops dispatched by kind+flavor+escape rather than a single GEMM key.
    pub key: Option<KernelKey>,
    /// Weight operands, in the order the kernel expects (e.g. [wq,wk,wv] for a
    /// QKV Proj). Indices into the model's weight table.
    pub weights: Vec<WeightSlot>,
    /// Input/output/scratch/state operands the executor binds per token.
    pub scratch: Vec<ScratchSlot>,
    /// Attention/activation/rope flavor (or `None`).
    pub flavor: OpFlavor,
}

/// A lowered per-layer program: the ordered super-ops for one transformer layer.
pub type LayerProgram = Vec<SuperOp>;

/// The whole lowered forward for a model: one `LayerProgram` per layer, plus a
/// generation counter guarding the load-time weight binding against any future
/// on-the-fly requant / adaptive-KV weight-floor realloc (40d98d4d). The
/// executor asserts `weight_gen == live weight-set gen` (debug) and re-lowers on
/// mismatch — the spike's stale-alias (risk #2) mitigation.
#[derive(Clone, Debug)]
pub struct LoweredForward {
    pub layers: Vec<LayerProgram>,
    pub weight_gen: u64,
}

impl LoweredForward {
    pub fn new(weight_gen: u64) -> Self {
        Self { layers: Vec::new(), weight_gen }
    }
}
