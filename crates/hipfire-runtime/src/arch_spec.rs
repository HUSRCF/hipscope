//! Shared dense-transformer decode forward (N5 Phase B).
//!
//! A plain dense transformer layer is the same op sequence across arches —
//! rmsnorm-rotate + QKV, optional attention bias, optional qk-norm, RoPE,
//! attention, o_proj+residual, ffn rmsnorm-rotate + gate/up, SwiGLU,
//! down+residual. llama and qwen2 hand-rolled byte-identical copies of it
//! (qwen2 wrapped in the `SuperOp` interpreter, llama inline). This module
//! factors that body into one [`dense_forward`] driver parameterized by a few
//! config-derived [`DenseKnobs`], with the one genuinely non-shared piece — the
//! KV-cache write + attention kernel family (llama's 7-tier KV ladder vs
//! qwen2's flash/gqa selector) — left to each arch via [`DenseArch::attend`].
//!
//! This is the "ArchSpec" authoring surface (greenfield BET 2), scoped to its
//! load/forward-time-durable core. Per the N4 review the design's `config:`
//! rows are superseded by serde `RawConfig + finalize`, so there is no config
//! schema here — each arch builds its own `Config` and derives `DenseKnobs`.
//!
//! Static dispatch only: [`dense_forward`] is generic over the concrete
//! `A: DenseArch`, so the per-token call graph stays fully inlinable (no
//! per-token `dyn`, per the forward-static rule in `arch.rs`). The driver feeds
//! the already-static `hipfire_dispatch::execute_steps`; it is not a runtime
//! op-interpreter.

use hip_bridge::{DeviceBuffer, HipResult};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_dispatch::types::RotationPlan;
use rdna_compute::{Gpu, GpuTensor};

/// Config-derived scalars that parameterize the shared dense forward. Built once
/// per forward from each arch's finalized `Config`.
pub struct DenseKnobs {
    /// Add q/k/v projection bias (qwen2: true; llama: false).
    pub attn_bias: bool,
    /// Apply per-head Q/K RMSNorm before RoPE (Qwen3-style; llama/qwen2: false).
    pub qk_norm: bool,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
}

/// Per-forward borrow of the shared decode scratch buffers. `pos_buf` is a raw
/// `DeviceBuffer` (matches both arches' scratch layout).
pub struct DenseScratch<'a> {
    pub x: &'a GpuTensor,
    pub tmp: &'a GpuTensor,
    pub x_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub o: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub ffn_hidden: &'a GpuTensor,
    pub ffn_out: &'a GpuTensor,
    pub pos_buf: &'a DeviceBuffer,
}

/// Per-layer borrow of one decoder layer's weights + its derived rotation plans.
/// Bias/qk-norm tensors are `Option` (present only when the matching knob is on).
pub struct DenseLayer<'a> {
    pub attn_norm: &'a GpuTensor,
    pub ffn_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub wk: WeightRef<'a>,
    pub wv: WeightRef<'a>,
    pub wo: WeightRef<'a>,
    pub w_gate: WeightRef<'a>,
    pub w_up: WeightRef<'a>,
    pub w_down: WeightRef<'a>,
    pub wq_bias: Option<&'a GpuTensor>,
    pub wk_bias: Option<&'a GpuTensor>,
    pub wv_bias: Option<&'a GpuTensor>,
    pub q_norm: Option<&'a GpuTensor>,
    pub k_norm: Option<&'a GpuTensor>,
    pub qkv_rot: RotationPlan,
    pub ffn_rot: RotationPlan,
    pub qkv_awq: Option<&'a GpuTensor>,
    pub ffn_awq: Option<&'a GpuTensor>,
    /// Activation dim fed to `RmsnormAutomatic` (the projection's `k`).
    pub qkv_k: usize,
    pub ffn_k: usize,
}

/// A dense transformer arch expressed for the shared [`dense_forward`] driver.
/// Implementors are thin per-forward borrow wrappers over the arch's weights +
/// scratch + KV cache + config.
pub trait DenseArch {
    fn n_layers(&self) -> usize;
    fn knobs(&self) -> &DenseKnobs;
    fn scratch(&self) -> DenseScratch<'_>;
    fn layer(&self, l: usize) -> DenseLayer<'_>;
    /// KV-cache write + single-token attention for layer `l`. By this point q/k/v
    /// are projected, biased, qk-normed and RoPE'd into the shared scratch; write
    /// the attention result into `attn_out`. This is the one op that does NOT
    /// unify across arches (different KV layouts + attention kernel families).
    fn attend(&self, gpu: &mut Gpu, l: usize) -> HipResult<()>;
}

#[inline]
fn herr(e: impl std::fmt::Display) -> hip_bridge::HipError {
    hip_bridge::HipError::new(0, &e.to_string())
}

/// Shared dense-transformer decode forward for one token. Runs the per-layer op
/// sequence; the caller does embedding (before) and final norm + lm_head +
/// sampling (after), since those buffers/dtypes differ per arch.
pub fn dense_forward<A: DenseArch>(gpu: &mut Gpu, ctx: &DispatchCtx, arch: &A) -> HipResult<()> {
    let k = arch.knobs();
    let s = arch.scratch();

    for l in 0..arch.n_layers() {
        let layer = arch.layer(l);

        // Attention QKV: rmsnorm-rotate + 3× Gemv (fused/per-op chosen by dispatch).
        execute_steps(
            gpu,
            ctx,
            &[
                Step::RmsnormAutomatic {
                    x: s.x,
                    norm_weight: layer.attn_norm,
                    x_plain: s.tmp,
                    out: s.x_rot,
                    awq_scale: layer.qkv_awq,
                    k: layer.qkv_k,
                    eps: k.norm_eps,
                    rotation: layer.qkv_rot,
                },
                Step::Gemv {
                    w: &layer.wq,
                    input: GemvInput::Prerotated(s.x_rot),
                    out: s.q,
                },
                Step::Gemv {
                    w: &layer.wk,
                    input: GemvInput::Prerotated(s.x_rot),
                    out: s.k,
                },
                Step::Gemv {
                    w: &layer.wv,
                    input: GemvInput::Prerotated(s.x_rot),
                    out: s.v,
                },
            ],
        )
        .map_err(herr)?;

        // QKV bias (qwen2).
        if k.attn_bias {
            gpu.bias_add_f32(s.q, layer.wq_bias.expect("attn_bias: wq_bias"), 1, k.q_dim)?;
            gpu.bias_add_f32(s.k, layer.wk_bias.expect("attn_bias: wk_bias"), 1, k.kv_dim)?;
            gpu.bias_add_f32(s.v, layer.wv_bias.expect("attn_bias: wv_bias"), 1, k.kv_dim)?;
        }

        // Per-head Q/K norm (Qwen3-style).
        if k.qk_norm {
            if let Some(qn) = layer.q_norm {
                gpu.rmsnorm_batched(s.q, qn, s.q, k.n_heads, k.head_dim, k.norm_eps)?;
            }
            if let Some(kn) = layer.k_norm {
                gpu.rmsnorm_batched(s.k, kn, s.k, k.n_kv_heads, k.head_dim, k.norm_eps)?;
            }
        }

        // RoPE.
        gpu.rope_f32(
            s.q,
            s.k,
            s.pos_buf,
            k.n_heads,
            k.n_kv_heads,
            k.head_dim,
            k.rope_theta,
        )?;

        // KV write + attention (arch-specific).
        arch.attend(gpu, l)?;

        // Attention output projection + residual.
        execute_steps(
            gpu,
            ctx,
            &[Step::GemvResidual {
                w: &layer.wo,
                input: GemvInput::Raw(s.attn_out),
                residual: s.x,
                out: s.o,
            }],
        )
        .map_err(herr)?;

        // FFN: rmsnorm-rotate + gate/up.
        execute_steps(
            gpu,
            ctx,
            &[
                Step::RmsnormAutomatic {
                    x: s.x,
                    norm_weight: layer.ffn_norm,
                    x_plain: s.tmp,
                    out: s.x_rot,
                    awq_scale: layer.ffn_awq,
                    k: layer.ffn_k,
                    eps: k.norm_eps,
                    rotation: layer.ffn_rot,
                },
                Step::Gemv {
                    w: &layer.w_gate,
                    input: GemvInput::Prerotated(s.x_rot),
                    out: s.gate,
                },
                Step::Gemv {
                    w: &layer.w_up,
                    input: GemvInput::Prerotated(s.x_rot),
                    out: s.up,
                },
            ],
        )
        .map_err(herr)?;

        // SwiGLU + down projection + residual.
        gpu.silu_mul_f32(s.gate, s.up, s.ffn_hidden)?;
        execute_steps(
            gpu,
            ctx,
            &[Step::GemvResidual {
                w: &layer.w_down,
                input: GemvInput::Raw(s.ffn_hidden),
                residual: s.x,
                out: s.ffn_out,
            }],
        )
        .map_err(herr)?;
    }

    Ok(())
}
