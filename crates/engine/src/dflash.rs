//! DFlash draft forward pass — native Rust+HIP.
//!
//! Minimal dependency surface: only reads HFQ draft files (arch_id = 20),
//! writes F32 GpuTensor weights, and runs a bidirectional cross-attention
//! Qwen3-flavored decoder over a block of masked positions.
//!
//! The draft model does not own a vocab head. Its output is the final
//! hidden state per block position; the caller applies the target's
//! `lm_head` to map to logits. This matches the upstream z-lab/dflash
//! reference and lets a single tokenizer / embedding table be shared.
//!
//! Architectural notes (also see `docs/DFLASH_ARCHITECTURE.md`):
//! - 5-layer Qwen3 decoder, all full attention, non-causal.
//! - Per-layer cross-attention over `target_hidden` (the projected
//!   concatenation of hidden states from a configured set of target
//!   layers, default `[1, 8, 15, 22, 29]` for a 32-layer target).
//! - Q length = `block_size`, K/V length = `ctx_len + block_size`
//!   (K/V = concat of projected target_hidden and current hidden_states).
//! - MVP simplification: draft has NO persistent KV cache; `k_ctx` /
//!   `v_ctx` are recomputed from the (caller-managed) cumulative
//!   `target_hidden` buffer on every step. This is functionally
//!   equivalent to the reference's cropped draft-KV cache and avoids
//!   one whole layer of persistence bookkeeping.

use crate::hfq::HfqFile;
use crate::llama::WeightTensor;
use hip_bridge::HipResult;
use rdna_compute::{DType, Gpu, GpuTensor};

// ─── Config ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DflashConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
}

impl DflashConfig {
    /// Returns the number of target hidden layers concatenated into fc input.
    pub fn num_extract(&self) -> usize {
        self.target_layer_ids.len()
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Parse from an HFQ file's metadata JSON. Expects the top-level
    /// `dflash` object written by `dflash_convert`.
    pub fn from_hfq(hfq: &HfqFile) -> Option<Self> {
        let meta: serde_json::Value = serde_json::from_str(&hfq.metadata_json).ok()?;
        let df = meta.get("dflash")?;

        let n_layers = df.get("num_hidden_layers").and_then(|v| v.as_u64())? as usize;
        let hidden = df.get("hidden_size").and_then(|v| v.as_u64())? as usize;
        let intermediate = df.get("intermediate_size").and_then(|v| v.as_u64())? as usize;
        let n_heads = df.get("num_attention_heads").and_then(|v| v.as_u64())? as usize;
        let n_kv_heads = df.get("num_key_value_heads").and_then(|v| v.as_u64())? as usize;
        let head_dim = df.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(
            (hidden / n_heads) as u64,
        ) as usize;
        let vocab_size = df.get("vocab_size").and_then(|v| v.as_u64())? as usize;
        let norm_eps = df
            .get("rms_norm_eps")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32;
        let rope_theta = df
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000_000.0) as f32;
        let block_size = df.get("block_size").and_then(|v| v.as_u64())? as usize;
        let mask_token_id = df.get("mask_token_id").and_then(|v| v.as_u64())? as u32;
        let target_layer_ids: Vec<usize> = df
            .get("target_layer_ids")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let num_target_layers = df
            .get("num_target_layers")
            .and_then(|v| v.as_u64())? as usize;

        Some(DflashConfig {
            n_layers,
            hidden,
            intermediate,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            norm_eps,
            rope_theta,
            block_size,
            mask_token_id,
            target_layer_ids,
            num_target_layers,
        })
    }
}

// ─── Weights ───────────────────────────────────────────────────────────────

pub struct DflashLayerWeights {
    pub attn_norm: GpuTensor,        // [hidden] — F32, RMSNorm weight
    pub wq: WeightTensor,            // [q_dim, hidden]
    pub wk: WeightTensor,            // [kv_dim, hidden]
    pub wv: WeightTensor,            // [kv_dim, hidden]
    pub wo: WeightTensor,            // [hidden, q_dim]
    pub q_norm: GpuTensor,           // [head_dim] — F32
    pub k_norm: GpuTensor,           // [head_dim] — F32
    pub ffn_norm: GpuTensor,         // [hidden] — F32
    pub w_gate: WeightTensor,        // [intermediate, hidden]
    pub w_up: WeightTensor,          // [intermediate, hidden]
    pub w_down: WeightTensor,        // [hidden, intermediate]
}

pub struct DflashWeights {
    /// `fc`: Linear(num_extract × hidden → hidden). Shape: [hidden, num_extract × hidden].
    pub fc: WeightTensor,
    pub hidden_norm: GpuTensor,    // [hidden] — F32
    pub norm: GpuTensor,           // [hidden] — F32, final output norm
    pub layers: Vec<DflashLayerWeights>,
    /// True when at least one matrix weight is MQ4G256 — drives whether
    /// the draft_forward path needs to allocate FWHT rotation scratches.
    pub has_mq: bool,
}

/// Load a F32-only tensor (norms, embedding-shaped scalars). Always F32 on GPU.
fn hfq_tensor_f32(hfq: &HfqFile, gpu: &mut Gpu, name: &str, shape: Vec<usize>) -> HipResult<GpuTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("dflash tensor missing: {name}"));
    let f32_data: Vec<f32> = match info.quant_type {
        1 => data
            .chunks_exact(2)
            .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        q => panic!("dflash: unsupported quant_type {q} for {name}"),
    };
    let expected: usize = shape.iter().product();
    assert_eq!(
        f32_data.len(),
        expected,
        "dflash: shape mismatch for {name}: have {}, expected {}",
        f32_data.len(),
        expected,
    );
    gpu.upload_f32(&f32_data, &shape)
}

/// Load a matrix tensor as a `WeightTensor` carrying its native dtype.
/// Supported quant_types:
///   1  (F16)      → lifted to F32 on GPU (legacy path).
///   2  (F32)      → uploaded as F32.
///   13 (MQ4-G256) → uploaded raw, kernel dispatch will FWHT-rotate x at use.
///
/// `shape = [m, k]` so m=output_dim and k=input_dim. The HFQ index stores
/// the unaligned byte length; for MQ4 we skip shape verification (the
/// quantized bytes are not a function of m*k alone — group padding can add
/// up to 255 trailing bytes per row group).
fn hfq_weight(hfq: &HfqFile, gpu: &mut Gpu, name: &str, m: usize, k: usize) -> HipResult<WeightTensor> {
    let (info, data) = hfq
        .tensor_data(name)
        .unwrap_or_else(|| panic!("dflash tensor missing: {name}"));
    match info.quant_type {
        1 => {
            // F16 on disk → F32 on GPU (legacy upload path).
            let f32_data: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| crate::llama::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            assert_eq!(f32_data.len(), m * k, "dflash {name} F16 size mismatch");
            let buf = gpu.upload_f32(&f32_data, &[m * k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        2 => {
            let f32_data: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(f32_data.len(), m * k, "dflash {name} F32 size mismatch");
            let buf = gpu.upload_f32(&f32_data, &[m * k])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::F32, m, k, row_stride: 0 })
        }
        13 => {
            // MQ4-G256: 136 bytes per 256 weights. The buffer is opaque to
            // the engine; the gemm_hfq4g256 kernel reads it directly.
            let buf = gpu.upload_raw(data, &[data.len()])?;
            Ok(WeightTensor { buf, gpu_dtype: DType::MQ4G256, m, k, row_stride: 0 })
        }
        q => panic!("dflash: unsupported matrix quant_type {q} for {name}"),
    }
}

impl DflashWeights {
    pub fn load(gpu: &mut Gpu, hfq: &HfqFile, cfg: &DflashConfig) -> HipResult<Self> {
        let fc = hfq_weight(hfq, gpu, "fc.weight", cfg.hidden, cfg.num_extract() * cfg.hidden)?;
        let hidden_norm = hfq_tensor_f32(hfq, gpu, "hidden_norm.weight", vec![cfg.hidden])?;
        let norm = hfq_tensor_f32(hfq, gpu, "norm.weight", vec![cfg.hidden])?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}");
            let layer = DflashLayerWeights {
                attn_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.input_layernorm.weight"), vec![cfg.hidden])?,
                wq: hfq_weight(hfq, gpu, &format!("{p}.self_attn.q_proj.weight"), cfg.q_dim(), cfg.hidden)?,
                wk: hfq_weight(hfq, gpu, &format!("{p}.self_attn.k_proj.weight"), cfg.kv_dim(), cfg.hidden)?,
                wv: hfq_weight(hfq, gpu, &format!("{p}.self_attn.v_proj.weight"), cfg.kv_dim(), cfg.hidden)?,
                wo: hfq_weight(hfq, gpu, &format!("{p}.self_attn.o_proj.weight"), cfg.hidden, cfg.q_dim())?,
                q_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.self_attn.q_norm.weight"), vec![cfg.head_dim])?,
                k_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.self_attn.k_norm.weight"), vec![cfg.head_dim])?,
                ffn_norm: hfq_tensor_f32(hfq, gpu, &format!("{p}.post_attention_layernorm.weight"), vec![cfg.hidden])?,
                w_gate: hfq_weight(hfq, gpu, &format!("{p}.mlp.gate_proj.weight"), cfg.intermediate, cfg.hidden)?,
                w_up: hfq_weight(hfq, gpu, &format!("{p}.mlp.up_proj.weight"), cfg.intermediate, cfg.hidden)?,
                w_down: hfq_weight(hfq, gpu, &format!("{p}.mlp.down_proj.weight"), cfg.hidden, cfg.intermediate)?,
            };
            layers.push(layer);
        }

        let has_mq = std::iter::once(&fc)
            .chain(layers.iter().flat_map(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down].into_iter()
            }))
            .any(|w| matches!(w.gpu_dtype, DType::MQ4G256));
        if has_mq {
            // The MQ4 dispatch needs the engine's FWHT sign tables uploaded
            // (matches `gemv_mq4g256_with_rotate`'s setup).
            gpu.ensure_mq_signs()?;
        }

        Ok(DflashWeights {
            fc,
            hidden_norm,
            norm,
            layers,
            has_mq,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.fc.buf);
        let _ = gpu.free_tensor(self.hidden_norm);
        let _ = gpu.free_tensor(self.norm);
        for l in self.layers {
            let _ = gpu.free_tensor(l.attn_norm);
            let _ = gpu.free_tensor(l.wq.buf);
            let _ = gpu.free_tensor(l.wk.buf);
            let _ = gpu.free_tensor(l.wv.buf);
            let _ = gpu.free_tensor(l.wo.buf);
            let _ = gpu.free_tensor(l.q_norm);
            let _ = gpu.free_tensor(l.k_norm);
            let _ = gpu.free_tensor(l.ffn_norm);
            let _ = gpu.free_tensor(l.w_gate.buf);
            let _ = gpu.free_tensor(l.w_up.buf);
            let _ = gpu.free_tensor(l.w_down.buf);
        }
    }
}

// ─── Scratch ───────────────────────────────────────────────────────────────

/// Activation buffers for one forward pass. Sized for up to
/// `max_block_size` query positions and up to `max_ctx_len` context
/// positions. A single scratch is reused across all speculative steps.
pub struct DflashScratch {
    pub max_block_size: usize,
    pub max_ctx_len: usize,

    // Block-sized activations (B rows).
    pub x: GpuTensor,              // [B, hidden] — hidden state rolled across layers
    pub x_norm: GpuTensor,         // [B, hidden]
    pub q: GpuTensor,              // [B, q_dim]
    pub k_noise: GpuTensor,        // [B, kv_dim]
    pub v_noise: GpuTensor,        // [B, kv_dim]
    pub gate: GpuTensor,           // [B, intermediate]
    pub up: GpuTensor,             // [B, intermediate]
    pub gate_up: GpuTensor,        // [B, intermediate]
    pub attn_out: GpuTensor,       // [B, q_dim]
    pub attn_proj: GpuTensor,      // [B, hidden]
    pub residual_attn: GpuTensor,  // [B, hidden]
    pub residual_ffn: GpuTensor,   // [B, hidden]

    // Context activations (L rows), where L ≤ max_ctx_len.
    pub target_hidden: GpuTensor,        // [L, num_extract × hidden]
    pub target_hidden_proj: GpuTensor,   // [L, hidden]
    pub k_ctx: GpuTensor,                // [L, kv_dim]
    pub v_ctx: GpuTensor,                // [L, kv_dim]

    // Concatenated K/V (L + B rows).
    pub k_cat: GpuTensor,                // [L + B, kv_dim]
    pub v_cat: GpuTensor,                // [L + B, kv_dim]

    // Positions (i32).
    pub positions_q: GpuTensor,          // [B]       i32
    pub positions_k: GpuTensor,          // [L + B]   i32

    // FWHT rotation scratch for MQ4 weight paths. Sized to the largest
    // single-call requirement: max(max_ctx × num_extract*hidden,
    // max_block × max_layer_K). Allocated only when DflashWeights.has_mq.
    pub mq_x_rot: Option<GpuTensor>,
}

impl DflashScratch {
    pub fn new(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
    ) -> HipResult<Self> {
        Self::new_with_mq(gpu, cfg, max_block_size, max_ctx_len, false)
    }

    /// `with_mq` allocates the FWHT rotation scratch needed when at least
    /// one matrix weight is MQ4-G256. Sized to handle every per-call
    /// rotation in the draft forward.
    pub fn new_with_mq(
        gpu: &mut Gpu,
        cfg: &DflashConfig,
        max_block_size: usize,
        max_ctx_len: usize,
        with_mq: bool,
    ) -> HipResult<Self> {
        let b = max_block_size;
        let l = max_ctx_len;
        let tot = l + b;
        let ne = cfg.num_extract();
        let h = cfg.hidden;
        let inter = cfg.intermediate;
        let qd = cfg.q_dim();
        let kvd = cfg.kv_dim();

        let mq_x_rot = if with_mq {
            // The widest single rotation: max(max_ctx × ne*h, max_block × max(intermediate, q_dim)).
            // ne*h on ctx is the `fc` rotation (target_hidden). intermediate is the `w_down`
            // rotation. q_dim is the `wo` rotation. Take the max so a single
            // buffer covers them all.
            let widest = std::cmp::max(l * ne * h, b * std::cmp::max(inter, qd));
            Some(gpu.alloc_tensor(&[widest], DType::F32)?)
        } else {
            None
        };

        Ok(DflashScratch {
            max_block_size: b,
            max_ctx_len: l,

            x:             gpu.alloc_tensor(&[b * h], DType::F32)?,
            x_norm:        gpu.alloc_tensor(&[b * h], DType::F32)?,
            q:             gpu.alloc_tensor(&[b * qd], DType::F32)?,
            k_noise:       gpu.alloc_tensor(&[b * kvd], DType::F32)?,
            v_noise:       gpu.alloc_tensor(&[b * kvd], DType::F32)?,
            gate:          gpu.alloc_tensor(&[b * inter], DType::F32)?,
            up:            gpu.alloc_tensor(&[b * inter], DType::F32)?,
            gate_up:       gpu.alloc_tensor(&[b * inter], DType::F32)?,
            attn_out:      gpu.alloc_tensor(&[b * qd], DType::F32)?,
            attn_proj:     gpu.alloc_tensor(&[b * h], DType::F32)?,
            residual_attn: gpu.alloc_tensor(&[b * h], DType::F32)?,
            residual_ffn:  gpu.alloc_tensor(&[b * h], DType::F32)?,

            target_hidden:      gpu.alloc_tensor(&[l * ne * h], DType::F32)?,
            target_hidden_proj: gpu.alloc_tensor(&[l * h], DType::F32)?,
            k_ctx:              gpu.alloc_tensor(&[l * kvd], DType::F32)?,
            v_ctx:              gpu.alloc_tensor(&[l * kvd], DType::F32)?,

            k_cat: gpu.alloc_tensor(&[tot * kvd], DType::F32)?,
            v_cat: gpu.alloc_tensor(&[tot * kvd], DType::F32)?,

            positions_q: gpu.alloc_tensor(&[b],   DType::F32)?,
            positions_k: gpu.alloc_tensor(&[tot], DType::F32)?,

            mq_x_rot,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.free_tensor(self.x);
        let _ = gpu.free_tensor(self.x_norm);
        let _ = gpu.free_tensor(self.q);
        let _ = gpu.free_tensor(self.k_noise);
        let _ = gpu.free_tensor(self.v_noise);
        let _ = gpu.free_tensor(self.gate);
        let _ = gpu.free_tensor(self.up);
        let _ = gpu.free_tensor(self.gate_up);
        let _ = gpu.free_tensor(self.attn_out);
        let _ = gpu.free_tensor(self.attn_proj);
        let _ = gpu.free_tensor(self.residual_attn);
        let _ = gpu.free_tensor(self.residual_ffn);
        let _ = gpu.free_tensor(self.target_hidden);
        let _ = gpu.free_tensor(self.target_hidden_proj);
        let _ = gpu.free_tensor(self.k_ctx);
        let _ = gpu.free_tensor(self.v_ctx);
        let _ = gpu.free_tensor(self.k_cat);
        let _ = gpu.free_tensor(self.v_cat);
        let _ = gpu.free_tensor(self.positions_q);
        let _ = gpu.free_tensor(self.positions_k);
        if let Some(t) = self.mq_x_rot {
            let _ = gpu.free_tensor(t);
        }
    }
}

// ─── Forward ───────────────────────────────────────────────────────────────

/// Dispatch a batched GEMM by weight dtype.
///
/// Layout (row-major):
///   x [batch × k]  F32 input activations
///   w.buf [m × k]  weight, format depends on w.gpu_dtype
///   y [batch × m]  F32 output
///
/// For MQ4-G256, the kernel needs the input FWHT-rotated. We do that into
/// `mq_x_rot` (sized to the per-call max in `DflashScratch`), then call the
/// HFQ4-G256 GEMM kernel against the pre-rotated weights.
fn gemm_dispatch(
    gpu: &mut Gpu,
    x: &GpuTensor,
    w: &WeightTensor,
    y: &GpuTensor,
    batch: usize,
    mq_x_rot: Option<&GpuTensor>,
) -> HipResult<()> {
    match w.gpu_dtype {
        DType::F32 => gpu.gemm_f32_batched(x, &w.buf, y, batch, w.k, w.m),
        DType::HFQ4G256 => gpu.gemm_hfq4g256(&w.buf, x, y, w.m, w.k, batch),
        DType::MQ4G256 => {
            let scratch = mq_x_rot.expect("MQ4 dispatch requires mq_x_rot scratch");
            // Use the prefix [0, batch * k) of the rotation scratch.
            let rot_view = scratch.sub_offset(0, batch * w.k);
            gpu.rotate_x_mq_batched(x, &rot_view, w.k, batch)?;
            gpu.gemm_hfq4g256(&w.buf, &rot_view, y, w.m, w.k, batch)
        }
        other => panic!("dflash gemm_dispatch: unsupported weight dtype {:?}", other),
    }
}

/// Upload f32 slice into a GPU tensor (bytes via memcpy_htod).
fn upload_slice_f32(gpu: &Gpu, dst: &GpuTensor, data: &[f32]) -> HipResult<()> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    gpu.hip.memcpy_htod(&dst.buf, bytes)
}

/// Upload i32 slice into a GPU tensor (interpreted as i32 by kernels).
fn upload_slice_i32(gpu: &Gpu, dst: &GpuTensor, data: &[i32]) -> HipResult<()> {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    gpu.hip.memcpy_htod(&dst.buf, bytes)
}

/// Run one draft forward. Inputs:
/// - `noise_embedding`: `[block_size × hidden]` f32, row-major. Comes from
///   `target.embed_tokens(block_output_ids)` on the caller side.
/// - `target_hidden`:   `[ctx_len × num_extract × hidden]` f32, row-major
///   (5-way concat of target's chosen-layer hidden states at `ctx_len`
///   accepted positions).
/// - `positions_q`:     `[block_size]` i32 — absolute position index of
///   each block position in the full sequence (used for RoPE on Q).
/// - `positions_k`:     `[ctx_len + block_size]` i32 — absolute position
///   index for every ctx position followed by every block position
///   (used for RoPE on K = concat(ctx, noise)).
///
/// Output: writes final hidden states `[block_size × hidden]` into
/// `scratch.x`. Caller applies target's `lm_head` over the last
/// `block_size - 1` rows to produce logits for the mask slots.
///
/// Precondition: `block_size ≤ scratch.max_block_size`,
/// `ctx_len ≤ scratch.max_ctx_len`.
/// Run one draft forward over `block_size` positions with `ctx_len` cached
/// context rows.
///
/// `noise_embedding`: if `Some`, uploaded into `scratch.x` before the forward.
///     If `None`, the caller must have already filled `scratch.x` with B × hidden
///     F32 embeddings — this avoids the target→host→draft round-trip in the
///     spec-decode hot loop (both target and draft share the same GPU, so
///     D2D copies into `scratch.x` suffice).
/// `target_hidden`: if `Some`, uploaded into `scratch.target_hidden`.
///     If `None`, the caller must have already filled `scratch.target_hidden`
///     with `ctx_len × num_extract × hidden` F32 rows.
#[allow(clippy::too_many_arguments)]
pub fn draft_forward(
    gpu: &mut Gpu,
    weights: &DflashWeights,
    cfg: &DflashConfig,
    noise_embedding: Option<&[f32]>,
    target_hidden: Option<&[f32]>,
    positions_q: &[i32],
    positions_k: &[i32],
    block_size: usize,
    ctx_len: usize,
    scratch: &mut DflashScratch,
) -> HipResult<()> {
    let b = block_size;
    let l = ctx_len;
    let tot = l + b;
    let h = cfg.hidden;
    let ne = cfg.num_extract();
    let qd = cfg.q_dim();
    let kvd = cfg.kv_dim();
    let hd = cfg.head_dim;
    let eps = cfg.norm_eps;
    let theta = cfg.rope_theta;

    assert!(b <= scratch.max_block_size, "block_size > scratch max");
    assert!(l <= scratch.max_ctx_len, "ctx_len > scratch max");
    if let Some(ne_slice) = noise_embedding {
        assert_eq!(ne_slice.len(), b * h, "noise_embedding size");
    }
    if let Some(th_slice) = target_hidden {
        assert_eq!(th_slice.len(), l * ne * h, "target_hidden size");
    }
    assert_eq!(positions_q.len(), b, "positions_q size");
    assert_eq!(positions_k.len(), tot, "positions_k size");

    // ── 0. Uploads ────────────────────────────────────────────────────────
    if let Some(ne_slice) = noise_embedding {
        upload_slice_f32(gpu, &scratch.x, ne_slice)?;
    }
    if let Some(th_slice) = target_hidden {
        upload_slice_f32(gpu, &scratch.target_hidden, th_slice)?;
    }
    upload_slice_i32(gpu, &scratch.positions_q, positions_q)?;
    upload_slice_i32(gpu, &scratch.positions_k, positions_k)?;

    // ── 1. target_hidden_proj = hidden_norm(fc @ target_hidden) ──────────
    // Dispatch on fc weight dtype: F32 → gemm_f32_batched (legacy),
    // MQ4 → FWHT-rotate target_hidden then gemm_hfq4g256.
    gemm_dispatch(
        gpu,
        &scratch.target_hidden,         // x [L, ne*h]
        &weights.fc,                    // w [hidden, ne*h]
        &scratch.target_hidden_proj,    // y [L, hidden]
        l,
        scratch.mq_x_rot.as_ref(),
    )?;
    // RMSNorm across each L-row of size hidden with hidden_norm weight.
    gpu.rmsnorm_batched(
        &scratch.target_hidden_proj,
        &weights.hidden_norm,
        &scratch.target_hidden_proj,
        l,
        h,
        eps,
    )?;

    // ── 2. Per-layer decoder ─────────────────────────────────────────────
    for li in 0..cfg.n_layers {
        let layer = &weights.layers[li];

        // Residual.
        gpu.hip.memcpy_dtod(&scratch.residual_attn.buf, &scratch.x.buf, (b * h) * 4)?;

        // attn_norm.
        gpu.rmsnorm_batched(
            &scratch.x,
            &layer.attn_norm,
            &scratch.x_norm,
            b,
            h,
            eps,
        )?;

        // Q/K/V projections — dispatched on each weight's dtype.
        gemm_dispatch(gpu, &scratch.x_norm, &layer.wq, &scratch.q,       b, scratch.mq_x_rot.as_ref())?;
        gemm_dispatch(gpu, &scratch.x_norm, &layer.wk, &scratch.k_noise, b, scratch.mq_x_rot.as_ref())?;
        gemm_dispatch(gpu, &scratch.x_norm, &layer.wv, &scratch.v_noise, b, scratch.mq_x_rot.as_ref())?;

        // K_ctx / V_ctx — same wk/wv weights but projected over the L
        // accepted-context rows of target_hidden_proj.
        gemm_dispatch(gpu, &scratch.target_hidden_proj, &layer.wk, &scratch.k_ctx, l, scratch.mq_x_rot.as_ref())?;
        gemm_dispatch(gpu, &scratch.target_hidden_proj, &layer.wv, &scratch.v_ctx, l, scratch.mq_x_rot.as_ref())?;

        // Concat K = [K_ctx | K_noise] → [L + B, kv_dim]
        //         V = [V_ctx | V_noise] → [L + B, kv_dim]
        let ctx_bytes   = (l * kvd) * 4;
        let noise_bytes = (b * kvd) * 4;
        gpu.hip.memcpy_dtod_at(&scratch.k_cat.buf, 0,          &scratch.k_ctx.buf,   0, ctx_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.k_cat.buf, ctx_bytes,  &scratch.k_noise.buf, 0, noise_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.v_cat.buf, 0,          &scratch.v_ctx.buf,   0, ctx_bytes)?;
        gpu.hip.memcpy_dtod_at(&scratch.v_cat.buf, ctx_bytes,  &scratch.v_noise.buf, 0, noise_bytes)?;

        // Per-head RMSNorm on Q: each of B*n_heads rows, size head_dim,
        // weight [head_dim].
        gpu.rmsnorm_batched(&scratch.q,    &layer.q_norm, &scratch.q,    b * cfg.n_heads,   hd, eps)?;
        // Per-head RMSNorm on K_cat: each of (L+B)*n_kv_heads rows.
        gpu.rmsnorm_batched(&scratch.k_cat, &layer.k_norm, &scratch.k_cat, tot * cfg.n_kv_heads, hd, eps)?;

        // RoPE. rope_batched_f32 expects q and k at the SAME batch size,
        // rotating at per-row positions. We call it twice with a zero
        // "head count" on the inactive tensor so its loop doesn't execute.
        // Call 1: rotate Q with positions_q. Pass k as a valid buffer
        // (scratch.k_noise is shape-compatible; n_heads_k=0 skips its loop).
        gpu.rope_batched_f32(
            &scratch.q,
            &scratch.k_noise,      // ignored because n_heads_k = 0
            &scratch.positions_q,  // [B]
            cfg.n_heads,
            0,
            hd,
            theta,
            b,
        )?;
        // Call 2: rotate K_cat with positions_k. n_heads_q = 0 skips Q.
        gpu.rope_batched_f32(
            &scratch.q,            // ignored because n_heads_q = 0
            &scratch.k_cat,
            &scratch.positions_k,  // [L + B]
            0,
            cfg.n_kv_heads,
            hd,
            theta,
            tot,
        )?;

        // Attention: Q [B, n_heads, hd] × K [tot, n_kv_heads, hd]^T → scores
        // (with GQA expansion) → softmax → @V.
        gpu.attention_dflash_f32(
            &scratch.q,
            &scratch.k_cat,
            &scratch.v_cat,
            &scratch.attn_out,
            b,
            tot,
            cfg.n_heads,
            cfg.n_kv_heads,
            hd,
        )?;

        // attn_proj = attn_out @ wo^T → [B, hidden]
        gemm_dispatch(gpu, &scratch.attn_out, &layer.wo, &scratch.attn_proj, b, scratch.mq_x_rot.as_ref())?;

        // x = residual_attn + attn_proj
        gpu.add_f32(&scratch.residual_attn, &scratch.attn_proj, &scratch.x)?;

        // Residual for FFN.
        gpu.hip.memcpy_dtod(&scratch.residual_ffn.buf, &scratch.x.buf, (b * h) * 4)?;

        // ffn_norm.
        gpu.rmsnorm_batched(&scratch.x, &layer.ffn_norm, &scratch.x_norm, b, h, eps)?;

        // gate = x_norm @ w_gate^T; up = x_norm @ w_up^T
        gemm_dispatch(gpu, &scratch.x_norm, &layer.w_gate, &scratch.gate, b, scratch.mq_x_rot.as_ref())?;
        gemm_dispatch(gpu, &scratch.x_norm, &layer.w_up,   &scratch.up,   b, scratch.mq_x_rot.as_ref())?;

        // SiLU(gate) * up → gate_up
        gpu.silu_mul_f32(&scratch.gate, &scratch.up, &scratch.gate_up)?;

        // x = w_down @ gate_up^T  (output [B, hidden])
        gemm_dispatch(gpu, &scratch.gate_up, &layer.w_down, &scratch.x, b, scratch.mq_x_rot.as_ref())?;

        // x = residual_ffn + x
        gpu.add_f32(&scratch.residual_ffn, &scratch.x, &scratch.x)?;
    }

    // ── 3. Final norm ────────────────────────────────────────────────────
    gpu.rmsnorm_batched(&scratch.x, &weights.norm, &scratch.x, b, h, eps)?;

    Ok(())
}
