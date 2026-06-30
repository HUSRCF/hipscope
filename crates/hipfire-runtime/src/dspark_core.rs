//! Arch-agnostic DSpark spec-decode core. The drafter body (deepseek4 MoE/MLA
//! chain, qwen3 dense transformer) is the only arch-specific seam — see
//! [`DsparkBody`]. Everything else (main_proj ingest, markov head, confidence
//! head, window orchestration) lives here.
use rdna_compute::{DType, Gpu, GpuTensor};

#[derive(Clone, Debug)]
pub struct DsparkConfig {
    pub block_size: usize,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub noise_token_id: u32,
    /// false ⇒ DFlash heads-off path (no confidence truncation).
    pub enable_confidence: bool,
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
            let wmma_on = std::env::var("HIPFIRE_DEEPSEEK4_Q8_WMMA")
                .map(|s| s != "0")
                .unwrap_or(true);
            if wmma_on && gpu.arch_caps.is_rdna4() {
                if let Some(scratch) = x_f16_scratch {
                    let n = (batch_size * k) as i64;
                    gpu.deepseek4_convert_f32_to_f16(x_plain_batch, scratch, n)
                        .map_err(|e| format!("convert_f32_to_f16 (Q8 WMMA): {e:?}"))?;
                    let opt_out = std::env::var("HIPFIRE_DEEPSEEK4_Q8_4W").as_deref() == Ok("0");
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
                    let opt_out_4w = std::env::var("HIPFIRE_DEEPSEEK4_Q8_4W").as_deref() == Ok("0");
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
            let wmma_on = std::env::var("HIPFIRE_DEEPSEEK4_HFQ4_WMMA")
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
    // rms_norm_eps: use a sensible default; callers that need per-arch eps
    // should extend DsparkConfig. For now 1e-6 is compatible with both
    // DeepSeek V4 and Qwen3.
    let rms_norm_eps = 1e-6f32;
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
    let _ = gpu.free_tensor(normed);
    // `logits_dev` stays resident: the markov loop adds each slot's bias
    // and argmaxes ON-GPU. Freed after the loop.

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

        // Confidence slot i ON GPU: stage [x_head[i] ++ markov_embed[i]] then
        // a 1-row `confidence_proj` gemv → conf_batch[i]. Uses the UNROTATED
        // emb_dev (matches the reference which dotted the raw markov embed).
        if cfg.enable_confidence {
            if let Some(confidence_proj) = weights.confidence_proj.as_ref() {
                let xh_i = x_head.sub_offset(i * hidden, hidden);
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
    if let Some(r) = emb_rot {
        let _ = gpu.free_tensor(r);
    }

    // ── Confidence download ───────────────────────────────────────────────
    // When confidence is disabled, return +inf so downstream truncation
    // is a no-op. When enabled, download the `block` logits and sigmoid.
    let confidence = if cfg.enable_confidence && weights.confidence_proj.is_some() {
        let mut raw = vec![0.0f32; block];
        {
            let bytes: &mut [u8] =
                unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u8, block * 4) };
            gpu.hip
                .memcpy_dtoh(bytes, &conf_batch.buf)
                .map_err(|e| format!("run_heads d2h confidence: {e:?}"))?;
        }
        // sigmoid: 1 / (1 + exp(-x))
        raw.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
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
