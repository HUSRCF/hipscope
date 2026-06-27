// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! LLaMA-family per-position verify for model-free speculative decode.
//!
//! Kept in its own module (not in `llama.rs`) so it can be added without
//! touching that legacy file. The arch-generic `NgramSpeculator` reaches this
//! through `impl SpecTarget for LlamaBundle` (in `hipfire-arch-llama`).

use crate::llama::{
    argmax, forward_prefill_batch_capture, forward_prefill_batch_tree, forward_scratch_compute,
    forward_scratch_embed, is_batchable_la, weight_gemv, ForwardScratch, HiddenCaptureSink,
    KvCache, LlamaConfig, LlamaWeights, PrefillBatchScratch,
};
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// Per-position greedy verify: run the target over `block` (length `n`) at
/// positions `[start_pos, start_pos + n)`, advancing `kv_cache` by `n`, and
/// return the target's greedy argmax at each position — `argmax[i]` is the token
/// predicted after consuming `block[0..=i]`.
///
/// Pure attention ⇒ no recurrent state to snapshot and the accepted-prefix KV is
/// already correct, so the speculator's `commit_prefix` is a no-op.
///
/// Fast path (the block-parallel win): when the block is batchable (`n >= 4`,
/// batchable weight dtypes, quantized KV, single chunk) one batched
/// [`forward_prefill_batch`] over the whole block leaves every row's hidden in
/// `pbs.x_batch`; we then do `n` cheap per-row `rmsnorm + lm_head + argmax`.
/// Shorter / ineligible blocks fall back to a per-token decode loop
/// (`forward_scratch_compute` already produces per-token logits).
///
/// The eligibility test mirrors `forward_prefill_batch`'s own (so the batched
/// call actually populates `pbs.x_batch` rather than silently taking its
/// per-token fallback); keep the two in sync.
#[allow(clippy::too_many_arguments)]
pub fn verify_block_argmax(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<Vec<u32>> {
    verify_block_logits_or_argmax(
        gpu, weights, config, block, start_pos, kv_cache, scratch, pbs, capture, false,
    )
    .map(|VerifyOut { argmax, .. }| argmax)
}

/// Like [`verify_block_argmax`] but returns the FULL per-position target logits
/// (`block.len() × vocab_size`, row-major) instead of just the argmax. Used by
/// the temp>0 chain DFlash path (SpecInfer naive sampling draws from the per-
/// position target distribution rather than taking the argmax). The logits are
/// bit-identical to those `verify_block_argmax` argmaxes internally — both go
/// through the same single batched forward + per-row `rmsnorm + lm_head`.
pub fn verify_block_logits(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<Vec<f32>> {
    verify_block_logits_or_argmax(
        gpu, weights, config, block, start_pos, kv_cache, scratch, pbs, capture, true,
    )
    .map(|VerifyOut { logits, .. }| logits)
}

/// One single-pass TREE-masked verify, returning the FULL per-node target
/// logits (`tokens.len() × vocab_size`, row-major).
///
/// `tokens` is the linearized DDTree (slot 0 = seed token, slots `1..` =
/// `tree.nodes`), `mask_host` is the `[n × n]` row-major additive
/// (`0.0`/`-inf`) tree-attention bias, and `depth_positions` the per-slot DEPTH
/// RoPE positions (`position + node.depth`) — all from
/// [`crate::ddtree::linearize_tree_with_parents`]. The whole tree is verified in
/// ONE batched forward: Q/K RoPE rotates at the DEPTH positions (parent→child
/// distance 1) while the KV write + mask stay on contiguous slots, so a node's
/// logits equal a causal verify of that node's root-to-node chain (greedy-
/// LOSSLESS). `capture` collects the per-extract-layer residual rows for DFlash
/// hidden conditioning. The mask is uploaded into a scratch GPU tensor
/// allocated + freed within the call.
#[allow(clippy::too_many_arguments)]
pub fn verify_tree_logits(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    tokens: &[u32],
    mask_host: &[f32],
    depth_positions: &[i32],
    position: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
) -> HipResult<Vec<f32>> {
    let n = tokens.len();
    let dim = config.dim;
    let vocab = config.vocab_size;
    assert_eq!(
        mask_host.len(),
        n * n,
        "verify_tree_logits: mask_host len {} != n*n ({}*{})",
        mask_host.len(),
        n,
        n
    );

    // Upload the [n × n] additive mask into a scratch GPU tensor.
    let bias = gpu.alloc_tensor(&[n * n], rdna_compute::DType::F32)?;
    let mask_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(mask_host.as_ptr() as *const u8, mask_host.len() * 4) };
    gpu.hip.memcpy_htod(&bias.buf, mask_bytes)?;

    // ONE tree-masked batched forward → every node's hidden lands in pbs.x_batch.
    let fwd = forward_prefill_batch_tree(
        gpu,
        weights,
        config,
        tokens,
        position,
        &bias,
        depth_positions,
        kv_cache,
        scratch,
        pbs,
        capture,
    );
    let _ = gpu.free_tensor(bias);
    fwd?;

    // Per-node rmsnorm + lm_head over the n hidden rows in pbs.x_batch.
    let mut logits_out: Vec<f32> = Vec::with_capacity(n * vocab);
    for i in 0..n {
        let off_bytes = i * dim * 4;
        gpu.hip
            .memcpy_dtod_at(&scratch.x.buf, 0, &pbs.x_batch.buf, off_bytes, dim * 4)?;
        gpu.rmsnorm_f32(
            &scratch.x,
            &weights.output_norm,
            &scratch.tmp,
            config.norm_eps,
        )?;
        weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
        logits_out.extend_from_slice(&gpu.download_f32(&scratch.logits)?);
    }
    Ok(logits_out)
}

/// Output of the shared verify forward: per-position argmax always; full logits
/// only when `want_logits` was set (else empty).
struct VerifyOut {
    argmax: Vec<u32>,
    logits: Vec<f32>,
}

/// Shared body for [`verify_block_argmax`] / [`verify_block_logits`]: one batched
/// forward over `block`, then per-row `rmsnorm + lm_head`. When `want_logits` the
/// full per-row logits are collected; the argmax is always returned (cheap CPU
/// scan over the already-downloaded row).
#[allow(clippy::too_many_arguments)]
fn verify_block_logits_or_argmax(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    block: &[u32],
    start_pos: usize,
    kv_cache: &mut KvCache,
    scratch: &ForwardScratch,
    pbs: &PrefillBatchScratch,
    capture: Option<&mut HiddenCaptureSink>,
    want_logits: bool,
) -> HipResult<VerifyOut> {
    let n = block.len();
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut out = Vec::with_capacity(n);
    let mut logits_out: Vec<f32> = if want_logits {
        Vec::with_capacity(n * vocab)
    } else {
        Vec::new()
    };

    const MIN_BATCH: usize = 4;
    let arch = gpu.arch.as_str();
    let kv_ok =
        kv_cache.quant_q8 || kv_cache.quant_asym2 || kv_cache.quant_asym3 || kv_cache.quant_asym4;
    let weights_ok = weights.layers.iter().all(|l| {
        is_batchable_la(l.wq.gpu_dtype, arch)
            && is_batchable_la(l.wk.gpu_dtype, arch)
            && is_batchable_la(l.wv.gpu_dtype, arch)
            && is_batchable_la(l.wo.gpu_dtype, arch)
            && is_batchable_la(l.w_gate.gpu_dtype, arch)
            && is_batchable_la(l.w_up.gpu_dtype, arch)
            && is_batchable_la(l.w_down.gpu_dtype, arch)
    });
    let eligible = crate::config::get().prefill_batched
        && n >= MIN_BATCH
        && n <= pbs.max_batch
        && kv_ok
        && weights_ok;

    // DFlash hidden capture only flows through the batched path; the per-token
    // fallback below does not run the capturing per-layer loop.
    assert!(
        eligible || capture.is_none(),
        "verify_block: hidden capture requested but block is ineligible \
         for the batched path (n={n}, kv_ok={kv_ok}, weights_ok={weights_ok})"
    );

    if eligible {
        // Single batched forward (n <= pbs.max_batch ⇒ one chunk) populates
        // pbs.x_batch with all n rows of post-final-layer hidden. Its own
        // last-row lm_head is redundant here but cheap. `capture` (if Some)
        // collects the per-extract-layer residual rows for DFlash conditioning.
        forward_prefill_batch_capture(
            gpu,
            weights,
            config,
            block,
            start_pos,
            kv_cache,
            scratch,
            Some(pbs),
            capture,
        )?;
        for i in 0..n {
            let off_bytes = i * dim * 4;
            gpu.hip
                .memcpy_dtod_at(&scratch.x.buf, 0, &pbs.x_batch.buf, off_bytes, dim * 4)?;
            gpu.rmsnorm_f32(
                &scratch.x,
                &weights.output_norm,
                &scratch.tmp,
                config.norm_eps,
            )?;
            weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
            let row = gpu.download_f32(&scratch.logits)?;
            out.push(argmax(&row));
            if want_logits {
                logits_out.extend_from_slice(&row);
            }
        }
    } else {
        for (i, &tok) in block.iter().enumerate() {
            forward_scratch_embed(gpu, weights, config, tok, start_pos + i, scratch)?;
            forward_scratch_compute(gpu, weights, config, start_pos + i, kv_cache, scratch)?;
            let row = gpu.download_f32(&scratch.logits)?;
            out.push(argmax(&row));
            if want_logits {
                logits_out.extend_from_slice(&row);
            }
        }
    }
    Ok(VerifyOut {
        argmax: out,
        logits: logits_out,
    })
}

/// Apply the target lm_head (final-norm + output projection) to `n` rows of
/// pre-norm residual hidden states, returning `n × vocab_size` host-side f32
/// logits in row-major order.
///
/// `hidden_rows` must be an `F32` `GpuTensor` of length `n × dim` laid out
/// row-major (row `i` starts at byte offset `i * dim * 4`). `scratch` is used
/// as a single-row staging buffer — `scratch.x`, `scratch.tmp`, and
/// `scratch.logits` are overwritten on every iteration. Callers that need the
/// raw logits for SWOR sampling should call this instead of running argmax
/// inside the loop.
///
/// Concretely for each row `i`:
///   1. DtoD-copy row `i` of `hidden_rows` into `scratch.x` (single F32 vector).
///   2. `rmsnorm_f32(scratch.x, weights.output_norm, scratch.tmp, eps)`.
///   3. `weight_gemv(weights.output, scratch.tmp, scratch.logits)`.
///   4. Download `scratch.logits` and append to the output buffer.
///
/// This mirrors the per-row lm_head loop in `verify_block_argmax` exactly —
/// reusing the same scratch buffers and the same kernel dispatch path — so the
/// returned logits are bit-identical to what `verify_block_argmax` would compute
/// before taking `argmax`.
pub fn lm_head_logits_n_rows(
    gpu: &mut Gpu,
    weights: &LlamaWeights,
    config: &LlamaConfig,
    hidden_rows: &GpuTensor,
    n: usize,
    scratch: &ForwardScratch,
) -> HipResult<Vec<f32>> {
    let dim = config.dim;
    let vocab = config.vocab_size;
    let mut out = Vec::with_capacity(n * vocab);
    for i in 0..n {
        let off_bytes = i * dim * 4;
        gpu.hip
            .memcpy_dtod_at(&scratch.x.buf, 0, &hidden_rows.buf, off_bytes, dim * 4)?;
        gpu.rmsnorm_f32(
            &scratch.x,
            &weights.output_norm,
            &scratch.tmp,
            config.norm_eps,
        )?;
        weight_gemv(gpu, &weights.output, &scratch.tmp, &scratch.logits)?;
        out.extend_from_slice(&gpu.download_f32(&scratch.logits)?);
    }
    Ok(out)
}
