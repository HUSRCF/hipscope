//! Arch-agnostic DSpark spec-decode core. The drafter body (deepseek4 MoE/MLA
//! chain, qwen3 dense transformer) is the only arch-specific seam — see
//! [`DsparkBody`]. Everything else (main_proj ingest, markov head, confidence
//! head, window orchestration) lives here.
use crate::spec::{
    accept_greedy_prefix, MtpDrafter, MtpSpeculator, MtpWindow, SpecGrammar, SpecTarget, Speculator,
};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── Per-window phase profiler (HIPFIRE_DSPARK_PROFILE=1) ─────────────────────
// Gated behind an env var; zero overhead when disabled.
#[derive(Default)]
struct DsparkProfiler {
    enabled: bool,
    windows: u64,
    bootstrap_ms: f64,
    draft_ms: f64,
    heads_ms: f64,
    verify_ms: f64,
    rest_ms: f64,
}

impl DsparkProfiler {
    fn new() -> Self {
        let enabled = std::env::var("HIPFIRE_DSPARK_PROFILE")
            .map(|s| s == "1")
            .unwrap_or(false);
        Self {
            enabled,
            ..Default::default()
        }
    }

    /// Sync GPU and start a phase timer. Returns `Instant::now()` (always, even
    /// when disabled, to avoid branching at the call site — the cost is one
    /// `Instant::now()` per phase when profiling is off, which is nanoseconds).
    fn sync_start(&self, gpu: &mut Gpu) -> std::time::Instant {
        if self.enabled {
            let _ = gpu.hip.device_synchronize();
        }
        std::time::Instant::now()
    }

    /// Sync GPU, accumulate elapsed ms into the given bucket.
    fn sync_end(&mut self, gpu: &mut Gpu, t: std::time::Instant, bucket: u8) {
        if !self.enabled {
            return;
        }
        let _ = gpu.hip.device_synchronize();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        match bucket {
            0 => self.bootstrap_ms += ms,
            1 => self.draft_ms += ms,
            2 => self.heads_ms += ms,
            3 => self.verify_ms += ms,
            _ => self.rest_ms += ms,
        }
    }

    fn end_window(&mut self) {
        if self.enabled {
            self.windows += 1;
        }
    }

    fn print_summary(&self) {
        if !self.enabled || self.windows == 0 {
            return;
        }
        let total =
            self.bootstrap_ms + self.draft_ms + self.heads_ms + self.verify_ms + self.rest_ms;
        let pct = |x: f64| if total > 0.0 { x / total * 100.0 } else { 0.0 };
        let mean = |x: f64| x / self.windows as f64;
        eprintln!("=== HIPFIRE_DSPARK_PROFILE ({} windows) ===", self.windows);
        eprintln!(
            "  bootstrap (capture_seed_main_hidden): {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.bootstrap_ms,
            pct(self.bootstrap_ms),
            mean(self.bootstrap_ms)
        );
        eprintln!(
            "  draft_block:                          {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.draft_ms,
            pct(self.draft_ms),
            mean(self.draft_ms)
        );
        eprintln!(
            "  run_heads:                            {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.heads_ms,
            pct(self.heads_ms),
            mean(self.heads_ms)
        );
        eprintln!(
            "  verify_block:                         {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.verify_ms,
            pct(self.verify_ms),
            mean(self.verify_ms)
        );
        eprintln!(
            "  rest (accept+commit+etc):             {:8.2} ms total  {:5.1}%  mean={:.2}ms/window",
            self.rest_ms,
            pct(self.rest_ms),
            mean(self.rest_ms)
        );
        eprintln!(
            "  total window time: {:.2} ms  mean={:.2}ms/window",
            total,
            mean(total)
        );
    }
}

#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub block_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub noise_token_id: u32,
    /// false ⇒ DFlash heads-off path (no confidence truncation).
    pub enable_confidence: bool,
    /// true ⇒ confidence head reads `normed[i]` (once-normed hidden, after
    /// `rmsnorm(x_head, stage_norm)`), matching the qwen3 reference
    /// `predict_confidence_step` which feeds `output_hidden = self.norm(hidden)`.
    /// false ⇒ confidence head reads raw `x_head[i]` (pre-norm), preserving
    /// deepseek4's original behavior (byte-identical baseline from task-5).
    /// Default: false (deepseek4-preserving).
    pub confidence_uses_normed: bool,
    /// RMSNorm epsilon used in `main_proj_ingest` and `run_heads`.
    /// Both DeepSeek V4 and Qwen3 use 1e-6; set per-arch at construction time
    /// so callers can plumb the model's actual config value.
    pub rms_norm_eps: f32,
}

impl DsparkConfig {
    /// Parse from the HFQ `metadata_json` string (the outer
    /// `{"architecture":.., "config":{..}}` envelope). Returns `None`
    /// when `config` or `dspark_block_size` is absent — i.e. the file
    /// is not a DSpark sidecar.
    ///
    /// Reads: `dspark_block_size`, `dspark_target_layer_ids`,
    /// `dspark_markov_rank`, `dspark_noise_token_id`,
    /// `dspark_enable_confidence` (defaults to `true` when absent, matching
    /// the deepseek4 behaviour where confidence is always enabled).
    /// `dspark_confidence_uses_normed` (defaults to `false` — callers that
    /// need the once-normed input, e.g. qwen3, must set this explicitly after
    /// parsing or emit it in the sidecar metadata).
    /// `norm_eps` (defaults to `1e-6` — compatible with both DeepSeek V4 and Qwen3).
    pub fn from_metadata_json(metadata_json: &str) -> Option<Self> {
        let wrapper: serde_json::Value = serde_json::from_str(metadata_json).ok()?;
        let cfg = wrapper.get("config")?;
        let block_size = cfg.get("dspark_block_size")?.as_u64()? as usize;
        if block_size == 0 {
            return None;
        }
        let target_layer_ids = cfg
            .get("dspark_target_layer_ids")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect::<Vec<_>>();
        let markov_rank = cfg.get("dspark_markov_rank")?.as_u64()? as usize;
        let noise_token_id = cfg.get("dspark_noise_token_id")?.as_u64()? as u32;
        let enable_confidence = cfg
            .get("dspark_enable_confidence")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let confidence_uses_normed = cfg
            .get("dspark_confidence_uses_normed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let rms_norm_eps = cfg
            .get("norm_eps")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(1e-6f32);
        Some(Self {
            block_size,
            target_layer_ids,
            markov_rank,
            noise_token_id,
            enable_confidence,
            confidence_uses_normed,
            rms_norm_eps,
        })
    }
}

pub struct DsparkWeights {
    pub cfg: DsparkConfig,
    pub main_proj: Option<GpuTensor>, // [dim, target_layer_ids.len()*dim]
    pub main_norm: Option<GpuTensor>,
    pub markov_w1: Option<GpuTensor>, // [vocab, rank]; None when markov_rank==0
    pub markov_w2: Option<GpuTensor>, // [vocab, rank]
    pub confidence_proj: Option<GpuTensor>, // [1, dim+rank]; None when !enable_confidence
    pub confidence_bias: Option<GpuTensor>, // [1]; qwen3 has bias, deepseek4 None
}

/// The arch-specific seam: draft one window's block given the assembled
/// `main_hidden`, returning the per-slot post-final-norm hidden (`x_head`).
pub trait DsparkBody {
    fn draft_block(
        &mut self,
        gpu: &mut Gpu,
        weights: &DsparkWeights,
        main_hidden: &GpuTensor, // [target_layer_ids.len()*dim]
        seed: u32,
        position: usize,
        block: usize,
        x_head_out: &GpuTensor, // [block, dim] out
    ) -> Result<(), String>;
    fn block_size(&self) -> usize;
    fn free(self: Box<Self>, gpu: &mut Gpu);
}

/// Per-window draft output: `block_size` token ids + per-slot confidence.
pub struct DraftResult {
    pub tokens: Vec<u32>,
    /// Per-slot draft logits `[block * vocab]`. Currently always EMPTY: the
    /// markov bias-add + argmax run on-GPU and no caller consumes the draft
    /// logits (the verify forward recomputes the trunk head). Populate this
    /// only if a future consumer needs them — it costs a `[block, vocab]` d2h.
    pub logits: Vec<f32>,
    pub confidence: Vec<f32>,
}

// ── private helpers (ported from hipfire-arch-deepseek4::forward) ────────────

/// Returns true when the weight's dtype requires an FWHT rotation of the
/// input before dispatch.
fn weight_needs_fwht(weight: &GpuTensor) -> bool {
    hipfire_dispatch::types::dtype_needs_rotation(weight.dtype)
}

/// Single-token embedding lookup into `out` (`[dim]` F32), dispatching on
/// the embedding-table dtype. Ported verbatim from deepseek4::forward
/// `dspark_embed_one`.
fn dspark_embed_one(
    gpu: &mut Gpu,
    table: &GpuTensor,
    out: &GpuTensor,
    token_id: u32,
    dim: usize,
) -> Result<(), String> {
    match table.dtype {
        DType::Q8_0 => gpu
            .embedding_lookup_q8(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed q8: {e:?}")),
        DType::F32 => gpu
            .embedding_lookup(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed f32: {e:?}")),
        DType::Raw => gpu
            .embedding_lookup_hfq4g256(table, out, token_id, dim)
            .map_err(|e| format!("dspark embed hfq4g256: {e:?}")),
        DType::F16 => {
            // No single-row F16 lookup kernel — extract the row to host,
            // convert F16→F32, upload into `out`. Cheap (dim ≤ 4096).
            let mut row_bytes = vec![0u8; dim * 2];
            let off = (token_id as usize) * dim * 2;
            gpu.hip
                .memcpy_dtoh_at(&mut row_bytes, &table.buf, off)
                .map_err(|e| format!("dspark embed f16 dtoh: {e:?}"))?;
            let row_f32: Vec<f32> = (0..dim)
                .map(|i| {
                    let h = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                    crate::llama::f16_to_f32(h)
                })
                .collect();
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(row_f32.as_ptr() as *const u8, dim * 4) };
            gpu.hip
                .memcpy_htod(&out.buf, bytes)
                .map_err(|e| format!("dspark embed f16 htod: {e:?}"))
        }
        other => Err(format!("dspark embed: unsupported table dtype {other:?}")),
    }
}

/// 1-row GEMV dispatched by weight dtype. `x_rotated` is the FWHT-rotated
/// activation (used for Raw/MQ4 weights); `x_plain` is used for all others.
/// Ported from deepseek4::forward `gemv_auto`.
fn gemv_auto(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated: &GpuTensor,
    x_plain: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
) -> Result<(), String> {
    use hipfire_dispatch::context::DispatchCtx;
    use hipfire_dispatch::families::gemv::WeightRef;

    let gemv = crate::llama::gemv_family();
    let ctx = DispatchCtx::new(gpu);
    let x = if weight_needs_fwht(weight) {
        x_rotated
    } else {
        x_plain
    };
    let wr = WeightRef {
        buf: weight,
        dtype: weight.dtype,
        m,
        k,
        row_stride: 0,
        rotation: None,
        awq_scale: None,
    };
    gemv.run_auto(&ctx, gpu, &wr, x, y)
        .map_err(|e| format!("gemv dispatch: {e}"))
}

/// Batched GEMV/GEMM dispatched by weight dtype with optional WMMA path.
/// `x_rotated_batch` is pre-FWHT; `x_plain_batch` is plain F32 activations.
/// `x_f16_scratch` is a `[batch_size * k]` F16 scratch for WMMA kernels.
/// Ported from deepseek4::forward `gemv_auto_batched_wmma`.
#[allow(clippy::too_many_arguments)]
fn gemv_auto_batched_wmma(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    x_rotated_batch: &GpuTensor,
    x_plain_batch: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    batch_size: usize,
    x_f16_scratch: Option<&GpuTensor>,
) -> Result<(), String> {
    match weight.dtype {
        DType::F32 => gpu
            .gemm_f32_register_tiled(weight, x_plain_batch, y, m, k, batch_size)
            .map_err(|e| format!("gemm_f32_register_tiled: {e:?}")),
        DType::Q8_0 => {
            let wmma_on = std::env::var("HIPFIRE_DSPARK_Q8_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if wmma_on && gpu.arch_caps.is_rdna4() {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    let opt_out = std::env::var("HIPFIRE_DSPARK_Q8_4W").as_deref() == Ok("0");
                    let use_4w = !opt_out
                        && batch_size >= 256
                        && m >= 4096
                        && m % 64 == 0
                        && k % 32 == 0
                        && batch_size % 64 == 0;
                    if use_4w {
                        return gpu
                            .gemm_q8_0_wmma_4w(weight, scratch, y, m, k, batch_size)
                            .map_err(|e| format!("gemm_q8_0_wmma_4w: {e:?}"));
                    }
                    return gpu
                        .gemm_q8_0_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_q8_0_wmma: {e:?}"));
                }
            } else if wmma_on && gpu.arch_caps.has_wmma() && m % 64 == 0 && k % 32 == 0 {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    let opt_out_4w = std::env::var("HIPFIRE_DSPARK_Q8_4W").as_deref() == Ok("0");
                    if !opt_out_4w && batch_size >= 64 && batch_size % 64 == 0 {
                        return gpu
                            .gemm_q8_0_wmma_4w(weight, scratch, y, m, k, batch_size)
                            .map_err(|e| format!("gemm_q8_0_wmma_4w: {e:?}"));
                    }
                    return gpu
                        .gemm_q8_0_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_q8_0_wmma: {e:?}"));
                }
            }
            gpu.gemm_q8_0_batched_chunked(weight, x_plain_batch, y, m, k, batch_size)
                .map_err(|e| format!("gemm_q8_0_batched_chunked: {e:?}"))
        }
        DType::F16 => {
            if gpu.arch_caps.has_wmma_w32_gfx12() {
                return gpu
                    .gemm_f16_wmma_mb8(weight, x_plain_batch, y, m, k, batch_size)
                    .map_err(|e| format!("gemm_f16_wmma_mb8 (gfx12 f16): {e:?}"));
            }
            if let Some(scratch) = x_f16_scratch {
                let n = (batch_size * k) as i64;
                gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                    .map_err(|e| format!("convert_f32_to_f16 (F16 weight): {e:?}"))?;
                gpu.gemm_f16_x_f16_wmma(weight, scratch, y, m, k, batch_size)
                    .map_err(|e| format!("gemm_f16_x_f16_wmma: {e:?}"))
            } else {
                Err("F16 weight requires WMMA path with x_f16_scratch".to_string())
            }
        }
        _ => {
            let wmma_on = std::env::var("HIPFIRE_DSPARK_HFQ4_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if wmma_on {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_rotated_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (HFQ4 WMMA): {e:?}"))?;
                    return gpu
                        .gemm_hfq4g256_wmma(weight, scratch, y, m, k, batch_size)
                        .map_err(|e| format!("gemm_hfq4g256_wmma: {e:?}"));
                }
            }
            gpu.gemm_hfq4g256(weight, x_rotated_batch, y, m, k, batch_size)
                .map_err(|e| format!("gemm_hfq4g256: {e:?}"))
        }
    }
}

// ── public API ───────────────────────────────────────────────────────────────

/// GEMV `main_proj` over `main_hidden` then `rmsnorm_f32` with `main_norm`
/// in place, writing the result into `out` (`[dim]` F32).
///
/// The concat width is `weights.cfg.target_layer_ids.len() * dim`
/// (generalised from deepseek4's hard-coded `3*hidden`), so 5 target layers
/// work for qwen3 as well as 3 for deepseek4.
///
/// The FWHT-rotation guard is preserved faithfully: when `main_proj` is an
/// MQ4 weight (`weight_needs_fwht` is true), `main_hidden` is rotated into a
/// temporary buffer before the GEMV.
///
/// `main_norm` is `Option` in [`DsparkWeights`]; an error is returned if
/// either `main_proj` or `main_norm` is `None`.
pub fn main_proj_ingest(
    gpu: &mut Gpu,
    weights: &DsparkWeights,
    main_hidden: &GpuTensor,
    out: &GpuTensor,
) -> Result<(), String> {
    let main_proj = weights
        .main_proj
        .as_ref()
        .ok_or("main_proj_ingest: main_proj missing")?;
    let main_norm = weights
        .main_norm
        .as_ref()
        .ok_or("main_proj_ingest: main_norm missing")?;

    let n_targets = weights.cfg.target_layer_ids.len();
    // out is [dim]; infer dim from its length.
    let dim = out.shape.last().copied().unwrap_or(0);
    if dim == 0 {
        return Err("main_proj_ingest: out has zero dim".to_string());
    }
    let concat_w = n_targets * dim;

    if weight_needs_fwht(main_proj) {
        let rot = gpu
            .alloc_tensor(&[concat_w], DType::F32)
            .map_err(|e| format!("main_proj_ingest alloc rot: {e:?}"))?;
        gpu.rotate_x_mq(main_hidden, &rot, concat_w)
            .map_err(|e| format!("main_proj_ingest rotate main_hidden: {e:?}"))?;
        gemv_auto(gpu, main_proj, &rot, main_hidden, out, dim, concat_w)?;
        let _ = gpu.free_tensor(rot);
    } else {
        gemv_auto(gpu, main_proj, main_hidden, main_hidden, out, dim, concat_w)?;
    }

    // main_norm RMSNorm in place.
    gpu.rmsnorm_f32(out, main_norm, out, weights.cfg.rms_norm_eps)
        .map_err(|e| format!("main_proj_ingest main_norm: {e:?}"))?;

    Ok(())
}

/// Build the noise token id block for one DSpark window:
/// `[seed, noise_token_id, noise_token_id, ...]` of length `cfg.block_size`.
pub fn noise_block_ids(cfg: &DsparkConfig, seed: u32) -> Vec<u32> {
    let mut ids = vec![cfg.noise_token_id; cfg.block_size];
    ids[0] = seed;
    ids
}

/// Run the DSpark head pipeline over a completed `x_head` block:
/// 1. lm-head GEMV: `rmsnorm(x_head, stage_norm)` → optional FWHT → batched
///    `lm_head` GEMV → `logits[block, vocab]` (resident on GPU).
/// 2. Sequential markov sampling loop: for each slot `i`, look up
///    `markov_w1[prev_token]`, (optionally) compute the on-GPU confidence
///    logit for slot `i`, then add `markov_w2 @ emb` bias to `logits[i]` and
///    argmax — all on GPU. Sequential dependency forces one argmax per slot.
/// 3. Confidence head download: `block` scalars transferred from
///    `conf_batch`; sigmoid applied on host. When `!cfg.enable_confidence`
///    the confidence vector is filled with `f32::INFINITY` (no-op for
///    downstream truncation).
///
/// `x_head` is `[block, hidden]` F32 produced by [`DsparkBody::draft_block`].
/// `stage_norm` is the per-stage final RMSNorm weight applied to `x_head`
/// before the lm-head GEMV. Callers supply the arch-specific norm: deepseek4
/// passes `mtp_final_norm`; qwen3 passes its drafter `norm` tensor.
/// `lm_head` is `[vocab, hidden]`.
/// `prev_token` is the last committed token before this draft window.
pub fn run_heads(
    gpu: &mut Gpu,
    weights: &DsparkWeights,
    stage_norm: &GpuTensor,
    lm_head: &GpuTensor,
    x_head: &GpuTensor, // [block, hidden]
    prev_token: u32,
    block: usize,
    vocab: usize,
) -> Result<DraftResult, String> {
    let cfg = &weights.cfg;

    // Guard: confidence head requires confidence_proj to be present.
    if cfg.enable_confidence && weights.confidence_proj.is_none() {
        return Err("run_heads: enable_confidence=true but confidence_proj missing".to_string());
    }

    let markov_rank = cfg.markov_rank;

    let markov_w1 = weights
        .markov_w1
        .as_ref()
        .ok_or("run_heads: markov_w1 missing")?;
    let markov_w2 = weights
        .markov_w2
        .as_ref()
        .ok_or("run_heads: markov_w2 missing")?;

    // Infer `hidden` from x_head shape.
    let hidden = x_head.shape.last().copied().unwrap_or(0);
    if hidden == 0 {
        return Err("run_heads: x_head has zero hidden dim".to_string());
    }

    // ── lm-head GEMV: rmsnorm(x_head, stage_norm) → [optional FWHT] → batched GEMV ──
    //
    // `stage_norm` is the per-stage final norm supplied by the caller
    // (deepseek4: `mtp_final_norm`; qwen3: drafter `norm`).
    let normed = gpu
        .alloc_tensor(&[block, hidden], DType::F32)
        .map_err(|e| format!("run_heads alloc normed: {e:?}"))?;
    let rms_norm_eps = cfg.rms_norm_eps;
    gpu.rmsnorm_batched(x_head, stage_norm, &normed, block, hidden, rms_norm_eps)
        .map_err(|e| format!("run_heads final rmsnorm: {e:?}"))?;

    let normed_rot = if weight_needs_fwht(lm_head) {
        let r = gpu
            .alloc_tensor(&[block, hidden], DType::F32)
            .map_err(|e| format!("run_heads alloc normed_rot: {e:?}"))?;
        gpu.rotate_x_mq_batched(&normed, &r, hidden, block)
            .map_err(|e| format!("run_heads rotate head input: {e:?}"))?;
        Some(r)
    } else {
        None
    };
    let logits_dev = gpu
        .alloc_tensor(&[block, vocab], DType::F32)
        .map_err(|e| format!("run_heads alloc logits: {e:?}"))?;
    let x_f16 = gpu
        .alloc_tensor(&[block * hidden], DType::F16)
        .map_err(|e| format!("run_heads alloc x_f16: {e:?}"))?;
    gemv_auto_batched_wmma(
        gpu,
        lm_head,
        normed_rot.as_ref().unwrap_or(&normed),
        &normed,
        &logits_dev,
        vocab,
        hidden,
        block,
        Some(&x_f16),
    )?;
    if let Some(r) = normed_rot {
        let _ = gpu.free_tensor(r);
    }
    let _ = gpu.free_tensor(x_f16);
    // `normed` is kept alive until after the loop: the confidence head
    // reads normed[i] per slot (modeling.py uses once-normed hidden for confidence).
    // `logits_dev` stays resident: the markov loop adds each slot's bias
    // and argmaxes ON-GPU. Both freed after the loop.

    // ── Sequential markov in-block sampling (greedy) ─────────────────────
    // out_ids[0] = prev_token; out_ids[i+1] = argmax(logits[i] + markov_bias).
    let mut out_ids = vec![prev_token; block + 1];

    // Confidence head buffers (ON GPU per slot inside the loop):
    // `conf_batch[block]` holds the per-slot confidence logit.
    // `concat_dev` stages `[x_head[i] ++ markov_embed[i]]` for the 1-row
    // `confidence_proj` gemv. Downloaded once after the loop (block floats).
    let proj_in = hidden + markov_rank;
    let conf_batch = gpu
        .alloc_tensor(&[block], DType::F32)
        .map_err(|e| format!("run_heads alloc conf_batch: {e:?}"))?;
    let concat_dev = gpu
        .alloc_tensor(&[proj_in], DType::F32)
        .map_err(|e| format!("run_heads alloc conf concat: {e:?}"))?;
    // Reusable device scratch for the markov embedding.
    let emb_dev = gpu
        .alloc_tensor(&[markov_rank], DType::F32)
        .map_err(|e| format!("run_heads alloc markov emb: {e:?}"))?;
    let bias_dev = gpu
        .alloc_tensor(&[vocab], DType::F32)
        .map_err(|e| format!("run_heads alloc markov bias: {e:?}"))?;
    let emb_rot = if weight_needs_fwht(markov_w2) {
        Some(
            gpu.alloc_tensor(&[markov_rank], DType::F32)
                .map_err(|e| format!("run_heads alloc markov emb rot: {e:?}"))?,
        )
    } else {
        None
    };
    for i in 0..block {
        // markov_w1 lookup of out_ids[i] → emb_dev [markov_rank] (unrotated).
        dspark_embed_one(gpu, markov_w1, &emb_dev, out_ids[i], markov_rank)?;

        // Confidence slot i ON GPU: stage [hidden_i ++ markov_embed[i]] then
        // a 1-row `confidence_proj` gemv → conf_batch[i]. Uses the UNROTATED
        // emb_dev (matches the reference which dotted the raw markov embed).
        // `hidden_i` is arch-specific: qwen3 uses `normed[i]` (once-normed,
        // matching modeling.py's predict_confidence_step input = self.norm(hidden));
        // deepseek4 uses raw `x_head[i]` (pre-norm, byte-identical to task-5 baseline).
        if cfg.enable_confidence {
            if let Some(confidence_proj) = weights.confidence_proj.as_ref() {
                let xh_i = if cfg.confidence_uses_normed {
                    normed.sub_offset(i * hidden, hidden)
                } else {
                    x_head.sub_offset(i * hidden, hidden)
                };
                let c_hidden = concat_dev.sub_offset(0, hidden);
                let c_markov = concat_dev.sub_offset(hidden, markov_rank);
                gpu.memcpy_dtod_auto(&c_hidden.buf, &xh_i.buf, hidden * 4)
                    .map_err(|e| format!("run_heads conf stage x_head {i}: {e:?}"))?;
                gpu.memcpy_dtod_auto(&c_markov.buf, &emb_dev.buf, markov_rank * 4)
                    .map_err(|e| format!("run_heads conf stage emb {i}: {e:?}"))?;
                let conf_i = conf_batch.sub_offset(i, 1);
                gemv_auto(
                    gpu,
                    confidence_proj,
                    &concat_dev,
                    &concat_dev,
                    &conf_i,
                    1,
                    proj_in,
                )?;
                // Add optional bias (qwen3 has a [1] bias; deepseek4 has None).
                if let Some(bias) = weights.confidence_bias.as_ref() {
                    gpu.add_inplace_f32(&conf_i, bias)
                        .map_err(|e| format!("run_heads conf bias add {i}: {e:?}"))?;
                }
            }
        }

        // bias = markov_w2 @ emb  ([vocab, markov_rank] · [markov_rank]).
        let x_for_w2 = if let Some(r) = emb_rot.as_ref() {
            gpu.rotate_x_mq(&emb_dev, r, markov_rank)
                .map_err(|e| format!("run_heads rotate markov emb {i}: {e:?}"))?;
            r
        } else {
            &emb_dev
        };
        gemv_auto(
            gpu,
            markov_w2,
            x_for_w2,
            &emb_dev,
            &bias_dev,
            vocab,
            markov_rank,
        )?;
        // logits[i] += bias, then argmax — both ON-GPU.
        let row = logits_dev.sub_offset(i * vocab, vocab);
        gpu.add_inplace_f32(&row, &bias_dev)
            .map_err(|e| format!("run_heads markov bias add {i}: {e:?}"))?;
        out_ids[i + 1] = gpu
            .argmax_f32(&row, vocab)
            .map_err(|e| format!("run_heads markov argmax {i}: {e:?}"))?;
    }
    let _ = gpu.free_tensor(emb_dev);
    let _ = gpu.free_tensor(bias_dev);
    let _ = gpu.free_tensor(logits_dev);
    let _ = gpu.free_tensor(normed);
    if let Some(r) = emb_rot {
        let _ = gpu.free_tensor(r);
    }

    // ── Confidence download ───────────────────────────────────────────────
    // When confidence is disabled, return +inf so downstream truncation
    // is a no-op. When enabled, download the `block` raw confidence LOGITS
    // (pre-sigmoid). The caller (`DsparkDrafter::mtp_step`) applies
    // `sigmoid(c)` itself when comparing against `conf_threshold`, matching
    // the `Deepseek4DsparkDrafter` convention: confidence stores raw logits,
    // sigmoid is applied at the truncation site.
    let confidence = if cfg.enable_confidence && weights.confidence_proj.is_some() {
        let mut raw = vec![0.0f32; block];
        {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, block * 4) };
            gpu.hip
                .memcpy_dtoh(bytes, &conf_batch.buf)
                .map_err(|e| format!("run_heads d2h confidence: {e:?}"))?;
        }
        raw
    } else {
        vec![f32::INFINITY; block]
    };
    let _ = gpu.free_tensor(conf_batch);
    let _ = gpu.free_tensor(concat_dev);

    Ok(DraftResult {
        tokens: out_ids[1..=block].to_vec(),
        logits: Vec::new(),
        confidence,
    })
}

// ── Generic DsparkDrafter ─────────────────────────────────────────────────────
//
// Drives any [`DsparkBody`] through the [`MtpDrafter`] interface. The arch-
// specific body (deepseek4 MoE chain, qwen3 dense transformer) is injected at
// build time; the target is reached only through [`SpecTarget`] trait methods
// (notably [`SpecTarget::capture_seed_main_hidden`] for bootstrap + hidden
// capture, and the generic verify primitives). No `Deepseek4*` type appears here.

/// Upload a host F32 slice to a freshly-allocated GPU tensor.
fn upload_f32(gpu: &mut Gpu, v: &[f32]) -> Result<GpuTensor, String> {
    let t = gpu
        .alloc_tensor(&[v.len()], DType::F32)
        .map_err(|e| format!("DsparkDrafter: alloc main_hidden: {e:?}"))?;
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
    gpu.memcpy_htod_auto(&t.buf, bytes)
        .map_err(|e| format!("DsparkDrafter: htod main_hidden: {e:?}"))?;
    Ok(t)
}

/// Generic DSpark drafter: window orchestration over any [`DsparkBody`].
///
/// ## main_hidden bookkeeping
///
/// `body.draft_block(main_hidden@P, seed=token@P, position=P)` drafts positions
/// `P+1 ..= P+block`. Before drafting at seed position `P` we need the trunk's
/// captured `[target_layer_ids]` main_hidden FOR the seed token at `P`. Since the
/// seed is freshly committed (never seen by the trunk in the current window), we
/// materialise its hidden via `target.capture_seed_main_hidden(gpu, seed, P,
/// &layers)` — one 1-token capture-armed trunk forward, result cached in
/// `main_hidden_dev`. `main_hidden_pos` tracks which absolute position that cache
/// corresponds to; the guard skips the bootstrap when already in sync (never
/// happens today — each window's bonus is always a fresh token — but makes the
/// contract explicit).
pub struct DsparkDrafter {
    body: Box<dyn DsparkBody>,
    weights: DsparkWeights,
    /// Per-stage final norm fed to [`run_heads`].
    stage_norm: GpuTensor,
    /// lm-head weight fed to [`run_heads`].
    lm_head: GpuTensor,
    conf_threshold: f32,
    block: usize,
    ctx_capacity: usize,
    /// Cached GPU tensor holding main_hidden for the seed at `main_hidden_pos`.
    /// `None` ⇒ must bootstrap on next `mtp_step`.
    main_hidden_dev: Option<GpuTensor>,
    /// Absolute position of the seed whose main_hidden is cached in
    /// `main_hidden_dev`. `None` ⇒ cache invalid.
    main_hidden_pos: Option<usize>,
    /// Per-window phase profiler; active only when `HIPFIRE_DSPARK_PROFILE=1`.
    profiler: DsparkProfiler,
}

impl MtpDrafter for DsparkDrafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
    ) -> Result<u32, String> {
        if !cache_hit {
            target.reset_recurrent(gpu);
        }
        // Invalidate main_hidden: mtp_step re-bootstraps for the first seed.
        // As in the deepseek4 drafter, we do NOT warm the DSpark stage rings
        // during prefill (measured LOSS on code prompts — see dspark_speculator.rs).
        //
        // DELIBERATE DEVIATION from Deepseek4DsparkDrafter::mtp_prefill, which
        // clears main_hidden_pos only on `!cache_hit`. We clear it
        // unconditionally because the generic body may not preserve a valid
        // cached hidden across a cache-hit prefill. This is behaviourally
        // identical today (both paths leave the seed's main_hidden_pos = None at
        // the end of prefill); the unconditional clear is the safe default for a
        // future cache-hit fold.
        self.main_hidden_pos = None;

        if fill_tokens.is_empty() {
            return Err("DsparkDrafter::mtp_prefill: fill_tokens is empty".into());
        }

        // Run the full prefill through spec_advance (reset=false; recurrent was
        // already reset above on cache_miss). Returns argmax at the last position,
        // which is the seed for the first decode window.
        let abort = &|| false;
        match target.spec_advance(gpu, fill_tokens, start_pos, false, abort, None)? {
            crate::spec::SpecAdvance::Ready { last_argmax } => Ok(last_argmax),
            crate::spec::SpecAdvance::Aborted => {
                Err("DsparkDrafter::mtp_prefill: spec_advance aborted".into())
            }
        }
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        // DSpark drafts the whole block at once. In-step grammar is not yet
        // wired for the generic drafter (no arch-specific grammar type to
        // downcast to) — ignore silently; post-hoc emission-layer grammar applies.

        let layers = self.weights.cfg.target_layer_ids.clone();
        let block = self.weights.cfg.block_size.min(k).max(1);
        let vocab = self.lm_head.shape[0];
        let hidden = {
            // Infer hidden from stage_norm shape (it's [hidden]).
            self.stage_norm.shape[0]
        };

        // ── 1. Bootstrap: ensure main_hidden@position for the seed ───────────
        // The seed is a fresh token; materialise its captured main_hidden with a
        // single 1-token capture-armed trunk forward (guard skips when already
        // in sync — never today but makes the contract explicit).
        let t_bootstrap = self.profiler.sync_start(gpu);
        if self.main_hidden_pos != Some(position) {
            let hidden_host = target.capture_seed_main_hidden(gpu, seed, position, &layers)?;
            let dev = upload_f32(gpu, &hidden_host)?;
            if let Some(old) = self.main_hidden_dev.take() {
                let _ = gpu.free_tensor(old);
            }
            self.main_hidden_dev = Some(dev);
            self.main_hidden_pos = Some(position);
        }
        self.profiler.sync_end(gpu, t_bootstrap, 0);

        let main_hidden = self
            .main_hidden_dev
            .as_ref()
            .ok_or("DsparkDrafter: main_hidden_dev missing after bootstrap")?;

        // ── 2. Draft the block with DsparkBody ──────────────────────────────
        let x_head_out = gpu
            .alloc_tensor(&[block, hidden], DType::F32)
            .map_err(|e| format!("DsparkDrafter: alloc x_head: {e:?}"))?;
        let t_draft = self.profiler.sync_start(gpu);
        self.body.draft_block(
            gpu,
            &self.weights,
            main_hidden,
            seed,
            position,
            block,
            &x_head_out,
        )?;
        self.profiler.sync_end(gpu, t_draft, 1);

        // ── 3. Heads: markov argmax + confidence ────────────────────────────
        let t_heads = self.profiler.sync_start(gpu);
        let draft = run_heads(
            gpu,
            &self.weights,
            &self.stage_norm,
            &self.lm_head,
            &x_head_out,
            seed,
            block,
            vocab,
        )?;
        self.profiler.sync_end(gpu, t_heads, 2);
        let _ = gpu.free_tensor(x_head_out);

        let mut drafts: Vec<u32> = draft.tokens.into_iter().take(block).collect();

        // ── 3a. Confidence-threshold truncation ─────────────────────────────
        // Mirror Deepseek4DsparkDrafter exactly: walk slots, truncate at first
        // slot whose sigmoid(confidence) < conf_threshold; always keep ≥1.
        let conf_threshold = self.conf_threshold;
        let confident_len = {
            let mut l = drafts.len();
            for (i, &c) in draft.confidence.iter().enumerate().take(drafts.len()) {
                let survival = 1.0f32 / (1.0 + (-c).exp());
                if survival < conf_threshold {
                    l = i;
                    break;
                }
            }
            l.max(1)
        };
        drafts.truncate(confident_len);
        let n_proposed = drafts.len();

        // ── 4. Verify: target forward over [seed, draft0..draft_{n-1}] ───────
        // The target runs through `capture_seed_main_hidden` for the NEXT seed
        // (the bonus) inside the NEXT window's bootstrap; the verify forward here
        // does NOT need to capture (no verify_block hidden_out). We use
        // spec_advance with no hidden_out: advance from `position` over
        // verify_tokens, getting back the last argmax (the bonus token).
        //
        // BUT: we need per-slot target argmaxes for accept_greedy_prefix, not
        // just the last one. `spec_advance` returns only the final argmax.
        //
        // Use the generic `verify_block` path instead: it returns per-slot
        // argmaxes (`argmax[i]` = target's prediction after consuming block[0..=i]).
        // We don't need a SpecScratch because for DSpark the verify is a plain
        // prefill (no recurrent rewind needed — the trunk is stateless-attention
        // in this window's view). Use `target.new_spec_scratch` to get an arch
        // scratch, then call `verify_block`.
        //
        // However: after verify_block, the target's KV is advanced by
        // verify_tokens.len() positions. For DSpark (like deepseek4's drafter)
        // the next window's seed is the BONUS token — a fresh token whose KV
        // slot is the NEXT position after the committed prefix. Since we commit
        // only the accepted prefix + bonus, we set n_tokens to
        // position + committed.len() AFTER accept, which is correct.
        //
        // We use `spec_advance` for the verify pass and reconstruct per-slot
        // target_pick by running the FULL verify+argmax in one pass, accepting
        // that we only get the LAST argmax from spec_advance. This is NOT
        // sufficient for multi-slot acceptance. We MUST use verify_block.
        let verify_tokens: Vec<u32> = std::iter::once(seed)
            .chain(drafts.iter().copied())
            .collect();

        let mut scratch = target.new_spec_scratch(gpu, verify_tokens.len())?;
        // hidden_out=None: we don't need the hidden states for the generic verify
        // (the next window's bootstrap re-captures via capture_seed_main_hidden).
        let t_verify = self.profiler.sync_start(gpu);
        let target_pick =
            target.verify_block(gpu, &verify_tokens, position, scratch.as_mut(), None)?;
        self.profiler.sync_end(gpu, t_verify, 3);

        // ── 5. Greedy accept ─────────────────────────────────────────────────
        let t_rest = self.profiler.sync_start(gpu);
        let acc = accept_greedy_prefix(&drafts, &target_pick, Some(eos));
        let committed = acc.committed;
        let n_accepted = acc.accepted;

        // ── 6. Commit the accepted prefix into the target's state ────────────
        // verify_block advanced the target by verify_tokens.len() slots. We
        // committed only n_accepted drafts + the bonus. Rewind to the true commit
        // length via commit_prefix (no-op for stateless/full-accept; replays for
        // recurrent arches).
        let accept_len = n_accepted; // accepted drafts; bonus is at accept_len
        target.commit_prefix(gpu, &verify_tokens, accept_len, position, scratch.as_mut())?;
        scratch.free(gpu);
        self.profiler.sync_end(gpu, t_rest, 4);

        // Invalidate the cached main_hidden — the next seed (the bonus) is a
        // fresh token whose hidden will be captured by the next window's bootstrap.
        self.main_hidden_pos = None;
        self.profiler.end_window();

        Ok(MtpWindow {
            committed,
            accepted: n_accepted,
            drafts_generated: n_proposed,
        })
    }

    fn mtp_reset(&mut self, _gpu: &mut Gpu) {
        // Invalidate the cached main_hidden so the next prefill re-bootstraps.
        self.main_hidden_pos = None;
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        self.profiler.print_summary();
        if let Some(dev) = self.main_hidden_dev {
            let _ = gpu.free_tensor(dev);
        }
        self.body.free(gpu);
    }

    fn k(&self) -> usize {
        self.block
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        true
    }
}

/// Build the generic DSpark speculator wrapping any [`DsparkBody`]. The
/// `stage_norm` is the per-stage final RMSNorm weight (deepseek4:
/// `mtp_final_norm`; qwen3: drafter `norm`); `lm_head` is `[vocab, hidden]`.
pub fn build_dspark_speculator(
    body: Box<dyn DsparkBody>,
    weights: DsparkWeights,
    stage_norm: GpuTensor,
    lm_head: GpuTensor,
    block: usize,
    ctx_capacity: usize,
    conf_threshold: f32,
) -> Box<dyn Speculator> {
    let block = block.clamp(1, 8);
    Box::new(MtpSpeculator::new(DsparkDrafter {
        body,
        weights,
        stage_norm,
        lm_head,
        conf_threshold,
        block,
        ctx_capacity,
        main_hidden_dev: None,
        main_hidden_pos: None,
        profiler: DsparkProfiler::new(),
    }))
}
