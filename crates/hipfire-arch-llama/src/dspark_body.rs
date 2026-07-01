// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3-8B DSpark drafter sidecar loader.
//!
//! Loads a `<stem>-dspark.hfq` sidecar (arch_id=1, 64 tensors produced by the
//! Task-6 quantiser) into:
//! - [`hipfire_runtime::dspark_core::DsparkWeights`] (globals: main_proj,
//!   main_norm, markov heads, confidence head + bias).
//! - [`Qwen3DrafterAssets`] (5-layer dense-GQA drafter body: LlamaWeights /
//!   LlamaConfig + block-sized KvCache + ForwardScratch + PrefillBatchScratch).
//!
//! ## Sidecar tensor layout (flat — no `model.` prefix)
//!
//! ```text
//! layers.{0..4}.self_attn.{q,k,v,o}_proj.weight   (qt=3, MQ4G256)
//! layers.{0..4}.self_attn.{q,k}_norm.weight        (qt=1, F16 → F32)
//! layers.{0..4}.{input_layernorm,post_attention_layernorm}.weight  (qt=1)
//! layers.{0..4}.mlp.{gate,up,down}_proj.weight     (qt=3)
//! embed_tokens.weight                              (qt=1, F16 → F32)
//! main_proj.weight                                 (qt=1, F16)
//! main_norm.weight                                 (qt=1, F16 → F32)
//! markov_head.markov_w1.weight                     (qt=1, F16)
//! markov_head.markov_w2.weight                     (qt=1, F16)
//! confidence_head.proj.weight                      (qt=1, F16)
//! confidence_head.proj.bias                        (qt=1, F16 → F32 scalar)
//! norm.weight                                      (qt=1, F16 → F32)
//! lm_head.weight                                   (qt=1, F16)
//! ```
//!
//! ## Hard requirements (Task-6 review)
//! 1. `confidence_bias` loaded from `confidence_head.proj.bias` — qwen3 HAS a
//!    bias; deepseek4 sets `confidence_bias: None`.
//! 2. `dspark_enable_confidence` parsed from the sidecar metadata —
//!    `DsparkConfig::from_metadata_json` (in dspark_core) reads it; deepseek4's
//!    local `DsparkConfig` hardcodes `enable_confidence: true`.

use hipfire_runtime::dspark_core::{DsparkConfig, DsparkWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::{
    ForwardScratch, KvCache, LayerWeights, LlamaConfig, LlamaWeights, ModelArch,
    PrefillBatchScratch, WeightTensor,
};
use hipfire_runtime::weight_backend::{
    dequant_f32, dequant_norm, dequant_weight_raw, load_awq_scale_for, load_embedding, read_first,
};
use rdna_compute::{DType, Gpu, GpuTensor};

// ── name resolver ─────────────────────────────────────────────────────────────
// The sidecar uses flat names (no `model.` prefix).  read_first's candidate fn
// must return just the bare name — not the `model.{name}` variant that
// flat_name_candidates would try first.
fn bare_name_candidates(name: &str) -> Vec<String> {
    vec![name.to_string()]
}

// ── Assets bundle ─────────────────────────────────────────────────────────────

/// GPU-resident assets for the 5-layer Qwen3-8B DSpark drafter body.
///
/// Produced by [`load_qwen3_dspark`] and consumed by Tasks 8–10 (body-forward,
/// window orchestration, speculator wiring).
///
/// YAGNI: only the fields definitely needed by forward + speculator are present.
pub struct Qwen3DrafterAssets {
    /// Drafter model config (n_layers=5, dim=4096, hidden=12288, n_heads=32,
    /// n_kv_heads=8, head_dim=128, has_qk_norm=true, rope_theta=1e6).
    pub config: LlamaConfig,
    /// Per-layer attention + FFN weights. Owned GPU tensors.
    pub weights: LlamaWeights,
    /// Block-only KvCache: F32, 5 layers, cap = block_size.  Reset per window.
    pub kv: KvCache,
    /// Single-token decode scratch.
    pub scratch: ForwardScratch,
    /// Block-parallel prefill scratch (block_size tokens × dim).
    pub pbs: PrefillBatchScratch,
}

// ── Public loader ─────────────────────────────────────────────────────────────

/// Load the Qwen3-8B DSpark sidecar into `(DsparkWeights, Qwen3DrafterAssets)`.
///
/// `source` is the already-opened sidecar HFQ.  The caller must call
/// `drop_mmap()` before calling this function (pread is used throughout to
/// avoid page-cache pressure on UMA).
///
/// Returns `None` when `dspark_block_size` is absent from the sidecar metadata
/// (i.e. the file is not a DSpark sidecar).  Returns `Err` on tensor load
/// failures.
pub fn load_qwen3_dspark(
    source: &HfqFile,
    gpu: &mut Gpu,
) -> Result<Option<(DsparkWeights, Qwen3DrafterAssets)>, String> {
    // 1. Parse DSpark config — includes dspark_enable_confidence (hard req #2)
    let dspark_cfg = match DsparkConfig::from_metadata_json(&source.metadata_json) {
        Some(c) => c,
        None => return Ok(None),
    };

    // 2. Derive drafter LlamaConfig from tensor shapes.
    //    The sidecar metadata only carries dspark_* keys (no model_type /
    //    hidden_size etc.), so config_from_hfq would fail on a missing
    //    `model_type` field.  Derive the config from tensor shapes instead.
    let cfg = config_from_sidecar_tensors(source)
        .map_err(|e| format!("qwen3_dspark: derive config: {e}"))?;

    let q_out_dim = cfg.n_heads * cfg.head_dim;
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;

    // 3. Load 5-layer drafter body
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        eprintln!("  qwen3_dspark: loading layer {i}/{} ...", cfg.n_layers);
        layers.push(load_drafter_layer(source, gpu, &cfg, i, q_out_dim, kv_dim)?);
    }

    // 4. Embedding table (embed_tokens.weight, qt=1 F16 → F32 EmbeddingFormat::F32)
    eprintln!("  qwen3_dspark: loading embed_tokens...");
    let (token_embd, embd_format) = {
        let (ei, ed) = source
            .tensor_data_pread("embed_tokens.weight")
            .ok_or_else(|| "qwen3_dspark: embed_tokens.weight missing".to_string())?;
        let qt = ei.quant_type;
        load_embedding(gpu, qt, &ed, cfg.vocab_size, cfg.dim)
            .map_err(|e| format!("qwen3_dspark: embed_tokens: {e:?}"))?
    };

    // 5. Final norm (norm.weight → F32)
    eprintln!("  qwen3_dspark: loading norm.weight...");
    let output_norm = {
        let (ni, nd) = source
            .tensor_data_pread("norm.weight")
            .ok_or_else(|| "qwen3_dspark: norm.weight missing".to_string())?;
        let qt = ni.quant_type;
        dequant_norm(gpu, qt, &nd, &[cfg.dim], 0.0)
            .map_err(|e| format!("qwen3_dspark: norm.weight: {e:?}"))?
    };

    // 6. lm_head.weight (qt=1 F16, used as WeightTensor for logit projection)
    eprintln!("  qwen3_dspark: loading lm_head.weight...");
    let lm_head = load_global_proj(source, gpu, "lm_head.weight", cfg.vocab_size, cfg.dim)?;

    let weights = LlamaWeights {
        token_embd,
        embd_format,
        output_norm,
        output: lm_head,
        layers,
        lm_head_aliases_embd: false,
    };

    // 7. DSpark globals
    //    main_proj: [dim, n_targets * dim] F16 on GPU
    eprintln!("  qwen3_dspark: loading main_proj.weight...");
    let main_proj = Some(load_global_tensor(source, gpu, "main_proj.weight")?);

    //    main_norm: [dim] F32
    eprintln!("  qwen3_dspark: loading main_norm.weight...");
    let main_norm = {
        let (mi, md) = source
            .tensor_data_pread("main_norm.weight")
            .ok_or_else(|| "qwen3_dspark: main_norm.weight missing".to_string())?;
        let qt = mi.quant_type;
        dequant_norm(gpu, qt, &md, &[cfg.dim], 0.0)
            .map_err(|e| format!("qwen3_dspark: main_norm.weight: {e:?}"))?
    };

    //    markov_w1/w2: [vocab, rank] F16
    eprintln!("  qwen3_dspark: loading markov_head.markov_w1.weight...");
    let markov_w1 = Some(load_global_tensor(
        source,
        gpu,
        "markov_head.markov_w1.weight",
    )?);
    eprintln!("  qwen3_dspark: loading markov_head.markov_w2.weight...");
    let markov_w2 = Some(load_global_tensor(
        source,
        gpu,
        "markov_head.markov_w2.weight",
    )?);

    //    confidence_head.proj.weight: [1, dim+rank] F16
    let confidence_proj = if dspark_cfg.enable_confidence {
        eprintln!("  qwen3_dspark: loading confidence_head.proj.weight...");
        Some(load_global_tensor(
            source,
            gpu,
            "confidence_head.proj.weight",
        )?)
    } else {
        None
    };

    //    confidence_head.proj.bias: [1] F16 → F32 — hard req #1 (qwen3 has bias)
    let confidence_bias = if dspark_cfg.enable_confidence {
        eprintln!("  qwen3_dspark: loading confidence_head.proj.bias...");
        let bias_gpu = {
            let (bi, bd) = source
                .tensor_data_pread("confidence_head.proj.bias")
                .ok_or_else(|| "qwen3_dspark: confidence_head.proj.bias missing".to_string())?;
            let qt = bi.quant_type;
            dequant_f32(gpu, qt, &bd, 1)
                .map_err(|e| format!("qwen3_dspark: confidence_head.proj.bias: {e:?}"))?
        };
        Some(bias_gpu)
    } else {
        None
    };

    let dspark_weights = DsparkWeights {
        cfg: dspark_cfg.clone(),
        main_proj,
        main_norm: Some(main_norm),
        markov_w1,
        markov_w2,
        confidence_proj,
        confidence_bias,
    };

    // 8. Allocate drafter KvCache (block-only: cap = block_size tokens)
    let block_cap = dspark_cfg.block_size;
    eprintln!(
        "  qwen3_dspark: KvCache (layers={}, kv_heads={}, head_dim={}, cap={}) ...",
        cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, block_cap,
    );
    let kv = KvCache::new_gpu(gpu, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, block_cap)
        .map_err(|e| format!("qwen3_dspark: KvCache::new_gpu: {e:?}"))?;

    // 9. ForwardScratch (single-token decode)
    let scratch = ForwardScratch::new(gpu, &cfg)
        .map_err(|e| format!("qwen3_dspark: ForwardScratch::new: {e:?}"))?;

    // 10. PrefillBatchScratch (block-parallel forward, max_batch = block_size)
    let pbs = PrefillBatchScratch::new(gpu, &cfg, block_cap, block_cap)
        .map_err(|e| format!("qwen3_dspark: PrefillBatchScratch::new: {e:?}"))?;

    let assets = Qwen3DrafterAssets {
        config: cfg,
        weights,
        kv,
        scratch,
        pbs,
    };

    Ok(Some((dspark_weights, assets)))
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Load one 5-layer drafter body layer from the flat-name sidecar.
/// Mirrors `hipfire_runtime::hfq::load_layer` but uses bare names (no
/// `model.` prefix) directly.
fn load_drafter_layer(
    source: &HfqFile,
    gpu: &mut Gpu,
    cfg: &LlamaConfig,
    i: usize,
    q_out_dim: usize,
    kv_dim: usize,
) -> Result<LayerWeights, String> {
    // ── Norms ──────────────────────────────────────────────────────────────
    let attn_norm = load_norm_by_name(
        source,
        gpu,
        &format!("layers.{i}.input_layernorm.weight"),
        &[cfg.dim],
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: attn_norm: {e}"))?;

    let ffn_norm = load_norm_by_name(
        source,
        gpu,
        &format!("layers.{i}.post_attention_layernorm.weight"),
        &[cfg.dim],
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: ffn_norm: {e}"))?;

    let q_norm = if cfg.has_qk_norm {
        Some(
            load_norm_by_name(
                source,
                gpu,
                &format!("layers.{i}.self_attn.q_norm.weight"),
                &[cfg.head_dim],
            )
            .map_err(|e| format!("qwen3_dspark layer {i}: q_norm: {e}"))?,
        )
    } else {
        None
    };
    let k_norm = if cfg.has_qk_norm {
        Some(
            load_norm_by_name(
                source,
                gpu,
                &format!("layers.{i}.self_attn.k_norm.weight"),
                &[cfg.head_dim],
            )
            .map_err(|e| format!("qwen3_dspark layer {i}: k_norm: {e}"))?,
        )
    } else {
        None
    };

    // ── Projections ────────────────────────────────────────────────────────
    let wq = load_proj_by_name(
        source,
        gpu,
        &format!("layers.{i}.self_attn.q_proj.weight"),
        q_out_dim,
        cfg.dim,
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: wq: {e}"))?;

    let wk = load_proj_by_name(
        source,
        gpu,
        &format!("layers.{i}.self_attn.k_proj.weight"),
        kv_dim,
        cfg.dim,
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: wk: {e}"))?;

    let wv = load_proj_by_name(
        source,
        gpu,
        &format!("layers.{i}.self_attn.v_proj.weight"),
        kv_dim,
        cfg.dim,
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: wv: {e}"))?;

    let wo = load_proj_by_name(
        source,
        gpu,
        &format!("layers.{i}.self_attn.o_proj.weight"),
        cfg.dim,
        q_out_dim,
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: wo: {e}"))?;

    let w_gate = load_proj_by_name(
        source,
        gpu,
        &format!("layers.{i}.mlp.gate_proj.weight"),
        cfg.hidden_dim,
        cfg.dim,
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: w_gate: {e}"))?;

    let w_up = load_proj_by_name(
        source,
        gpu,
        &format!("layers.{i}.mlp.up_proj.weight"),
        cfg.hidden_dim,
        cfg.dim,
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: w_up: {e}"))?;

    let w_down = load_proj_by_name(
        source,
        gpu,
        &format!("layers.{i}.mlp.down_proj.weight"),
        cfg.dim,
        cfg.hidden_dim,
    )
    .map_err(|e| format!("qwen3_dspark layer {i}: w_down: {e}"))?;

    Ok(LayerWeights {
        attn_norm,
        wq,
        wk,
        wv,
        wo,
        q_norm,
        k_norm,
        ffn_norm,
        w_gate,
        w_up,
        w_down,
    })
}

/// Load a norm tensor (F16 → F32, shape `shape`) from the flat-name sidecar.
fn load_norm_by_name(
    source: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    shape: &[usize],
) -> Result<GpuTensor, String> {
    let (info, data) = source
        .tensor_data_pread(name)
        .ok_or_else(|| format!("{name} missing"))?;
    let qt = info.quant_type;
    dequant_norm(gpu, qt, &data, shape, 0.0).map_err(|e| format!("{name}: {e:?}"))
}

/// Load a projection weight tensor (quantized or F16) from the flat-name sidecar.
/// Attaches AWQ sidecar when present.
fn load_proj_by_name(
    source: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) =
        read_first(source, name, bare_name_candidates).ok_or_else(|| format!("{name} missing"))?;
    let mut wt = dequant_weight_raw(gpu, info.quant_type, &data, m, k)
        .map_err(|e| format!("{name}: {e:?}"))?;
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale_for(source, gpu, name, k);
    }
    Ok(wt)
}

/// Upload a global weight tensor as `GpuTensor` (F16 kept as F16, MQ4 as
/// Raw/Q8_0 etc.).  Used for DSpark globals consumed by dspark_core.
fn load_global_tensor(source: &HfqFile, gpu: &mut Gpu, name: &str) -> Result<GpuTensor, String> {
    let (shape, qt, bytes) = {
        let (info, bytes) = source
            .tensor_data_pread(name)
            .ok_or_else(|| format!("qwen3_dspark: {name} missing"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        let qt = info.quant_type;
        // Copy shape/qt before Ref<Vec<u8>> is consumed; bytes moves here.
        (shape, qt, bytes)
    };
    let mut t = gpu
        .upload_raw(&bytes, &shape)
        .map_err(|e| format!("qwen3_dspark: upload {name}: {e:?}"))?;
    if qt == 1 {
        t.dtype = DType::F16;
    }
    Ok(t)
}

/// Load a global projection as `WeightTensor` (for lm_head.weight).
fn load_global_proj(
    source: &HfqFile,
    gpu: &mut Gpu,
    name: &str,
    m: usize,
    k: usize,
) -> Result<WeightTensor, String> {
    let (info, data) = read_first(source, name, bare_name_candidates)
        .ok_or_else(|| format!("qwen3_dspark: {name} missing"))?;
    let mut wt = dequant_weight_raw(gpu, info.quant_type, &data, m, k)
        .map_err(|e| format!("qwen3_dspark: {name}: {e:?}"))?;
    if wt.gpu_dtype.supports_awq_sidecar() {
        wt.awq_scale = load_awq_scale_for(source, gpu, name, k);
    }
    Ok(wt)
}

/// Derive a `LlamaConfig` from the sidecar tensor index.
///
/// The DSpark qwen3 sidecar metadata only carries `dspark_*` keys — it has no
/// `model_type`/`hidden_size`/etc., so `config_from_hfq` fails.  We derive
/// the config from tensor shapes instead.  The qwen3-8b drafter is always a
/// dense-GQA transformer, so the derivation is exact.
fn config_from_sidecar_tensors(source: &HfqFile) -> Result<LlamaConfig, String> {
    // ── dim from embed_tokens.weight ─────────────────────────────────────────
    let embed = source
        .find_tensor_info("embed_tokens.weight")
        .ok_or_else(|| "embed_tokens.weight missing".to_string())?;
    if embed.shape.len() < 2 {
        return Err(format!(
            "embed_tokens.weight unexpected shape {:?}",
            embed.shape
        ));
    }
    let vocab_size = embed.shape[0] as usize;
    let dim = embed.shape[1] as usize;

    // ── head_dim from q_norm.weight ───────────────────────────────────────────
    let q_norm = source
        .find_tensor_info("layers.0.self_attn.q_norm.weight")
        .ok_or_else(|| "layers.0.self_attn.q_norm.weight missing".to_string())?;
    let head_dim = q_norm.shape.first().copied().unwrap_or(128) as usize;
    let has_qk_norm = true; // presence of q_norm.weight confirms it

    // ── n_heads from q_proj.weight [q_out_dim, dim] ──────────────────────────
    let wq = source
        .find_tensor_info("layers.0.self_attn.q_proj.weight")
        .ok_or_else(|| "layers.0.self_attn.q_proj.weight missing".to_string())?;
    let q_out_dim = wq.shape[0] as usize;
    let n_heads = q_out_dim / head_dim;

    // ── n_kv_heads from k_proj.weight [kv_out_dim, dim] ──────────────────────
    let wk = source
        .find_tensor_info("layers.0.self_attn.k_proj.weight")
        .ok_or_else(|| "layers.0.self_attn.k_proj.weight missing".to_string())?;
    let kv_out_dim = wk.shape[0] as usize;
    let n_kv_heads = kv_out_dim / head_dim;

    // ── hidden_dim from gate_proj.weight [hidden_dim, dim] ───────────────────
    let wg = source
        .find_tensor_info("layers.0.mlp.gate_proj.weight")
        .ok_or_else(|| "layers.0.mlp.gate_proj.weight missing".to_string())?;
    let hidden_dim = wg.shape[0] as usize;

    // ── n_layers: probe layers.{N}.input_layernorm.weight until absent ────────
    let mut n_layers = 0usize;
    while source
        .find_tensor_info(&format!("layers.{n_layers}.input_layernorm.weight"))
        .is_some()
    {
        n_layers += 1;
    }
    if n_layers == 0 {
        return Err("qwen3_dspark: no body layers found (layers.0.* absent)".into());
    }

    Ok(LlamaConfig {
        arch: ModelArch::Qwen3,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        head_dim,
        norm_eps: 1e-6,              // qwen3 standard
        max_seq_len: 1024,           // drafter; actual cap = block_size (set by KvCache)
        rope_freq_base: 1_000_000.0, // qwen3 rope θ = 1e6
        bos_token: 1,
        eos_token: 2,
        has_qk_norm,
    })
}
