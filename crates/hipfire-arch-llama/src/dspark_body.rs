// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Bjoern Boesel
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3-8B DSpark drafter sidecar loader + block-attention body forward.
//!
//! ## Sidecar loader
//!
//! Loads a `<stem>-dspark.hfq` sidecar (arch_id=1, 64 tensors produced by the
//! Task-6 quantiser) into:
//! - [`hipfire_runtime::dspark_core::DsparkWeights`] (globals: main_proj,
//!   main_norm, markov heads, confidence head + bias).
//! - [`Qwen3DrafterAssets`] (5-layer dense-GQA drafter body: LlamaWeights /
//!   LlamaConfig + block-sized KvCache + ForwardScratch + PrefillBatchScratch).
//!
//! ## Block-attention body forward
//!
//! [`dspark_qwen3_block_forward`] implements the 5-layer dense Qwen3 forward
//! where each layer's block queries attend **bidirectionally** over
//! `[main_x context KV ++ block KV]`.  This matches
//! `Qwen3DSparkModel._forward_backbone` in the reference:
//!   - modeling.py:373  `target_hidden_states = self.hidden_norm(self.fc(...))`
//!                      → `main_x` is computed by the caller (Task 7) before entering
//!                      this function.
//!   - modeling.py:99–116 per-layer attention: q/k/v projections, q_norm/k_norm
//!     (on concatenated K), RoPE, bidirectional GQA over [ctx++block] KV.
//!   - modeling.py:375  single `position_embeddings` call before the layer loop →
//!     all layers share the same RoPE positions (not recomputed per layer).
//!   - modeling.py:386  `self.norm(hidden_states)` → final norm applied here.
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
use hipfire_runtime::hfq::{load_layer, load_weight_tensor_pread, HfqFile};
use hipfire_runtime::llama::{
    weight_gemv, ForwardScratch, KvCache, LayerWeights, LlamaConfig, LlamaWeights, ModelArch,
    PrefillBatchScratch, WeightTensor,
};
use hipfire_runtime::weight_backend::{
    dequant_f32, dequant_norm, dequant_weight_raw, load_awq_scale_for, load_embedding, read_first,
    HfqBackend,
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
        layers.push(load_drafter_layer(source, gpu, &cfg, i, q_out_dim, kv_dim)?);
    }

    // 4. Embedding table (embed_tokens.weight, qt=1 F16 → F32 EmbeddingFormat::F32)
    let (token_embd, embd_format) = {
        let (ei, ed) = source
            .tensor_data_pread("embed_tokens.weight")
            .ok_or_else(|| "qwen3_dspark: embed_tokens.weight missing".to_string())?;
        let qt = ei.quant_type;
        load_embedding(gpu, qt, &ed, cfg.vocab_size, cfg.dim)
            .map_err(|e| format!("qwen3_dspark: embed_tokens: {e:?}"))?
    };

    // 5. Final norm (norm.weight → F32)
    let output_norm = {
        let (ni, nd) = source
            .tensor_data_pread("norm.weight")
            .ok_or_else(|| "qwen3_dspark: norm.weight missing".to_string())?;
        let qt = ni.quant_type;
        dequant_norm(gpu, qt, &nd, &[cfg.dim], 0.0)
            .map_err(|e| format!("qwen3_dspark: norm.weight: {e:?}"))?
    };

    // 6. lm_head.weight (qt=1 F16, used as WeightTensor for logit projection)
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
    let main_proj = Some(load_global_tensor(source, gpu, "main_proj.weight")?);

    //    main_norm: [dim] F32
    let main_norm = {
        let (mi, md) = source
            .tensor_data_pread("main_norm.weight")
            .ok_or_else(|| "qwen3_dspark: main_norm.weight missing".to_string())?;
        let qt = mi.quant_type;
        dequant_norm(gpu, qt, &md, &[cfg.dim], 0.0)
            .map_err(|e| format!("qwen3_dspark: main_norm.weight: {e:?}"))?
    };

    //    markov_w1/w2: [vocab, rank] F16
    let markov_w1 = Some(load_global_tensor(
        source,
        gpu,
        "markov_head.markov_w1.weight",
    )?);
    let markov_w2 = Some(load_global_tensor(
        source,
        gpu,
        "markov_head.markov_w2.weight",
    )?);

    //    confidence_head.proj.weight: [1, dim+rank] F16
    let confidence_proj = if dspark_cfg.enable_confidence {
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

/// Load one drafter body layer from the flat-name sidecar.
///
/// Delegates to `hipfire_runtime::hfq::load_layer` via an `HfqBackend`
/// configured with `bare_name_candidates` so it resolves `layers.N.*`
/// without the `model.` prefix that `flat_name_candidates` would prepend.
fn load_drafter_layer(
    source: &HfqFile,
    gpu: &mut Gpu,
    cfg: &LlamaConfig,
    i: usize,
    q_out_dim: usize,
    kv_dim: usize,
) -> Result<LayerWeights, String> {
    let mut b = HfqBackend {
        hfq: source,
        gpu,
        norm_bias: 0.0,
        candidates: bare_name_candidates,
        read_proj: load_weight_tensor_pread,
        layer: i,
    };
    load_layer(&mut b, cfg, q_out_dim, kv_dim, i)
        .map_err(|e| format!("qwen3_dspark layer {i}: {e:?}"))
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

// ── Block-attention body forward ──────────────────────────────────────────────

/// GPU scratch buffers for [`dspark_qwen3_block_forward`].
///
/// Allocated once per model load (sized to `block_size`).  Reset is implicit:
/// every call re-embeds `block_ids` from scratch, so no state carries over.
///
/// Buffer sizing (qwen3-8b defaults: dim=4096, n_heads=32, n_kv_heads=8,
/// head_dim=128, hidden_dim=14336):
///   `q_dim = n_heads * head_dim = 4096`
///   `kv_dim = n_kv_heads * head_dim = 1024`
///   KV cache capacity = `1 + block_size` (context slot 0 + block slots 1..=block)
pub struct Qwen3DsparkScratch {
    /// Q8_0 KV cache (5 drafter layers, capacity = 1 + block_size).
    /// Layout: context K/V at compact slot 0; block K/V at slots 1..=block.
    /// Compact slots decouple absolute RoPE positions from KV write positions,
    /// matching the deepseek4 DSpark staging approach.
    pub kv: KvCache,

    /// Block-parallel scratch: x_batch[block×dim], fa_q/k/v[block×*], etc.
    /// Reuses PrefillBatchScratch so layer-loop kernels use the same buffers as
    /// `forward_prefill_chunk` (fa_q_batch, x_rot_batch, …).
    pub pbs: PrefillBatchScratch,

    /// Concatenated [ctx(1) ++ block(block)] K buffer [(1+block)×kv_dim] F32.
    /// Used to apply k_norm to the full combined K sequence before KV write
    /// (modeling.py:107–113 cats k_ctx+k_noise before applying k_norm).
    pub all_k: GpuTensor,

    /// Concatenated [ctx(1) ++ block(block)] V buffer [(1+block)×kv_dim] F32.
    /// V has no norm (modeling.py:114 just transposes), but is staged here for
    /// the batched Q8_0 KV-cache write.
    pub all_v: GpuTensor,

    /// KV positions for the combined [ctx ++ block] sequence,
    /// shape [1+block_size], as i32-in-F32.  Set to [seed_pos, seed_pos+1, ...,
    /// seed_pos+block] on each call.  Used for:
    ///   1. RoPE on the concatenated K (modeling.py:116 applies RoPE to all k).
    ///   2. Q8_0 KV-cache write (kv_cache_write_q8_0_batched positions arg).
    pub positions_kv_all: GpuTensor,

    /// Block query RoPE positions [block_size] i32-in-F32.
    /// = [seed_pos+1, seed_pos+2, ..., seed_pos+block].
    /// Separate from positions_kv_all because rope_batched_f32 takes one
    /// positions buffer for [Q++K] jointly, and we need Q-only positions for
    /// the block queries.
    pub positions_q_block: GpuTensor,

    /// Compact attention positions [block_size] i32-in-F32 = [1, 2, ..., block].
    /// Passed as `positions` to `attention_q8_0_kv_batched_masked`: each block
    /// query row i uses compact slot i+1 (KV was written at compact 1..=block),
    /// while the context slot 0 is always visible (it precedes block_start=1).
    pub positions_compact: GpuTensor,

    /// Additive bias [block × block] F32 = 0.0 (bidirectional in-block mask).
    /// Combined with `block_start=1`, `block_cols=block` in the masked-attention
    /// kernel: all block queries attend to all block keys.
    /// (modeling.py:58 `self.is_causal = False`; `create_dspark_attention_mask`
    /// makes every block query see all block keys.)
    pub bias: GpuTensor,
}

impl Qwen3DsparkScratch {
    /// Allocate scratch for a drafter with the given config and `block_size`.
    pub fn new(gpu: &mut Gpu, config: &LlamaConfig, block_size: usize) -> Result<Self, String> {
        let kv_cap = 1 + block_size;
        let kv = KvCache::new_gpu_q8(
            gpu,
            config.n_layers,
            config.n_kv_heads,
            config.head_dim,
            kv_cap,
        )
        .map_err(|e| format!("Qwen3DsparkScratch: kv: {e:?}"))?;

        let pbs = PrefillBatchScratch::new(gpu, config, block_size, kv_cap)
            .map_err(|e| format!("Qwen3DsparkScratch: pbs: {e:?}"))?;

        let kv_dim = config.n_kv_heads * config.head_dim;

        let all_k = gpu
            .alloc_tensor(&[kv_cap * kv_dim], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: all_k: {e:?}"))?;
        let all_v = gpu
            .alloc_tensor(&[kv_cap * kv_dim], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: all_v: {e:?}"))?;
        let positions_kv_all = gpu
            .alloc_tensor(&[kv_cap], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: positions_kv_all: {e:?}"))?;
        let positions_q_block = gpu
            .alloc_tensor(&[block_size], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: positions_q_block: {e:?}"))?;
        let positions_compact = gpu
            .alloc_tensor(&[block_size], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: positions_compact: {e:?}"))?;
        let bias = gpu
            .zeros(&[block_size * block_size], DType::F32)
            .map_err(|e| format!("Qwen3DsparkScratch: bias: {e:?}"))?;

        Ok(Self {
            kv,
            pbs,
            all_k,
            all_v,
            positions_kv_all,
            positions_q_block,
            positions_compact,
            bias,
        })
    }

    /// Release all GPU allocations.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        self.kv.free_gpu(gpu);
        self.pbs.free_gpu(gpu);
        for t in [
            self.all_k,
            self.all_v,
            self.positions_kv_all,
            self.positions_q_block,
            self.positions_compact,
            self.bias,
        ] {
            let _ = gpu.free_tensor(t);
        }
    }
}

// ── dspark_qwen3_block_forward ─────────────────────────────────────────────────

/// Qwen3-8B DSpark block-attention forward: 5-layer dense GQA over the
/// bidirectional `[context(1) ++ block(N)]` KV set.
///
/// # Numeric contract (verified against modeling.py)
///
/// ## `main_x` context: computed once, shared across all 5 layers
///
/// Caller computes `main_x = hidden_norm(fc(main_hidden))` (modeling.py:373).
/// Each layer re-uses the SAME `main_x` to form its context K/V via
/// this layer's `k_proj`/`v_proj` (modeling.py:103–106).
///
/// ## Per-layer op sequence (modeling.py:181–198, 99–151)
///
/// ```text
/// 1. input_layernorm(x_block)   [modeling.py:181]
/// 2. q_proj(normed_block)       [modeling.py:99]
/// 3. q_norm(q, per-head)        [modeling.py:102  — BEFORE RoPE]
/// 4. k_proj(main_x)  → ctx_k   [modeling.py:103]
/// 5. k_proj(normed_block) → blk_k [modeling.py:104]
/// 6. cat([ctx_k, blk_k]) → all_k  [modeling.py:107]
/// 7. k_norm(all_k, per-head)    [modeling.py:113  — on concatenated K, BEFORE RoPE]
/// 8. v_proj(main_x)  → ctx_v   [modeling.py:105]
/// 9. v_proj(normed_block) → blk_v [modeling.py:106]
/// 10. cat([ctx_v, blk_v]) → all_v  [modeling.py:110]
/// 11. RoPE(q, ctx_k=0_row_of_all_k, blk_k=1..=block_rows_of_all_k)
///         positions: ctx→seed_pos, blk→seed_pos+1..=+block
///         [modeling.py:116; apply_rotary_pos_emb:34–40 — q uses last q_len
///          entries of cos/sin, k uses full cos/sin]
/// 12. Write all_k, all_v to Q8 KV cache at compact slots 0..=block
/// 13. attention_q8_0_kv_batched_masked:
///         positions_compact=[1..=block], block_start=1, block_cols=block,
///         bias=zeros → bidirectional (slot 0 always visible, all block-slots open)
///         [modeling.py:58 `is_causal=False`]
/// 14. o_proj(attn_out) + residual  [modeling.py:193–194]
/// 15. post_attention_layernorm(x_block)  [modeling.py:196]
/// 16. MLP(gate/up SwiGLU) + residual    [modeling.py:197–198]
/// ```
///
/// ## RoPE position assignment
///
/// `apply_rotary_pos_emb` (modeling.py:34–40) takes `cos/sin` shaped
/// `[1+block, head_dim]` (computed from full position_ids = [seed_pos,
/// seed_pos+1, ..., seed_pos+block]).  For Q it uses the LAST `q_len` entries
/// (`cos[..., -q_len:, :]`), i.e. the block positions `seed_pos+1..=+block`.
/// For K it uses the full sequence.
///
/// Implemented here via two `rope_batched_f32` calls:
///   - Q: positions_q_block = [seed_pos+1, ..., seed_pos+block]
///   - K: positions_kv_all  = [seed_pos, seed_pos+1, ..., seed_pos+block]
///     (1 ctx row at seed_pos + block rows at seed_pos+1..=+block)
///
/// ## Bidirectional mask
///
/// `attention_q8_0_kv_batched_masked` with `block_start=1`, `block_cols=block`,
/// `bias=zeros[block×block]` gives every block query full visibility of all
/// in-block keys.  Slot 0 (context) is before `block_start`, so it is always
/// visible (the kernel never masks prompt keys).
///
/// # Arguments
///
/// * `drafter` — 5-layer Qwen3-8B body weights (LlamaWeights).
/// * `config`  — `n_layers=5`, `has_qk_norm=true`, `rope_freq_base=1e6`.
/// * `main_x`  — `[dim]` F32 context vector (= `hidden_norm(fc(main_hidden))`).
/// * `block_ids` — `[block]` token ids: `[seed_token, noise, noise, ...]`.
/// * `seed_position` — absolute KV position of the seed token.
/// * `block`   — number of block slots (= block_size in practice).
/// * `scratch` — pre-allocated [`Qwen3DsparkScratch`].
/// * `x_head_out` — `[block × dim]` F32 output (post-final-norm hidden states).
pub(crate) fn dspark_qwen3_block_forward(
    gpu: &mut Gpu,
    drafter: &LlamaWeights,
    config: &LlamaConfig,
    main_x: &GpuTensor,
    block_ids: &[u32],
    seed_position: usize,
    block: usize,
    scratch: &Qwen3DsparkScratch,
    x_head_out: &GpuTensor,
) -> Result<(), String> {
    debug_assert_eq!(block_ids.len(), block);
    debug_assert!(
        block <= scratch.pbs.max_batch,
        "block {block} > pbs.max_batch"
    );

    let dim = config.dim;
    let q_dim = config.n_heads * config.head_dim;
    let kv_dim = config.n_kv_heads * config.head_dim;
    let kv_cap = 1 + block; // compact slots: 0=ctx, 1..=block=block rows

    // ── 0. Upload positions ────────────────────────────────────────────────────

    // positions_kv_all = [seed_pos, seed_pos+1, ..., seed_pos+block]
    {
        let pos: Vec<i32> = (0..=block as i32)
            .map(|i| seed_position as i32 + i)
            .collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, kv_cap * 4) };
        gpu.hip
            .memcpy_htod(&scratch.positions_kv_all.buf, pos_bytes)
            .map_err(|e| format!("dspark_qwen3: htod positions_kv_all: {e:?}"))?;
    }

    // positions_q_block = [seed_pos+1, seed_pos+2, ..., seed_pos+block]
    {
        let pos: Vec<i32> = (1..=block as i32)
            .map(|i| seed_position as i32 + i)
            .collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, block * 4) };
        gpu.hip
            .memcpy_htod(&scratch.positions_q_block.buf, pos_bytes)
            .map_err(|e| format!("dspark_qwen3: htod positions_q_block: {e:?}"))?;
    }

    // positions_compact = [1, 2, ..., block]  (compact KV-cache slots for attention)
    {
        let pos: Vec<i32> = (1..=block as i32).collect();
        let pos_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, block * 4) };
        gpu.hip
            .memcpy_htod(&scratch.positions_compact.buf, pos_bytes)
            .map_err(|e| format!("dspark_qwen3: htod positions_compact: {e:?}"))?;
    }

    // ── 1. Embed block_ids → pbs.x_batch  ─────────────────────────────────────
    //
    // Embed each token into pbs.x_batch row i.
    // drafter.embd_format is F32 (qt=1 F16 was dequantized in the loader).
    // sub_offset takes offset in ELEMENTS (not bytes); pbs.x_batch is F32.
    for (i, &tok) in block_ids.iter().enumerate() {
        let x_row = scratch.pbs.x_batch.sub_offset(i * dim, dim);
        gpu.embedding_lookup(&drafter.token_embd, &x_row, tok, dim)
            .map_err(|e| format!("dspark_qwen3: embed[{i}]: {e:?}"))?;
    }

    // ── 2. Per-layer loop ×5 ───────────────────────────────────────────────────

    for layer_idx in 0..config.n_layers {
        let layer = &drafter.layers[layer_idx];

        // ── 2a. input_layernorm(x_batch) → x_rot_batch  ───────────────────────
        // modeling.py:181  `residual = hidden_states; hidden_states = input_layernorm(hidden_states)`
        // No MQ rotation here (drafter uses MQ4G256 for projections, but x is F32 and the
        // rmsnorm_batched path produces the normed output into x_rot_batch for the projections).
        gpu.rmsnorm_batched(
            &scratch.pbs.x_batch,
            &layer.attn_norm,
            &scratch.pbs.x_rot_batch,
            block,
            dim,
            config.norm_eps,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: attn_norm: {e:?}"))?;

        // ── 2b. Q projection: wq(normed_block) → fa_q_batch  ──────────────────
        // modeling.py:99   `q = self.q_proj(hidden_states).view(...)`
        // fa_q_batch is [block × q_dim] F32.
        // weight_gemv handles MQ4 auto-rotate internally.
        // sub_offset offset is in ELEMENTS (F32 tensors: 1 element = 4 bytes).
        for i in 0..block {
            let x_row = scratch.pbs.x_rot_batch.sub_offset(i * dim, dim);
            let q_row = scratch.pbs.fa_q_batch.sub_offset(i * q_dim, q_dim);
            weight_gemv(gpu, &layer.wq, &x_row, &q_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: q_proj[{i}]: {e:?}"))?;
        }

        // ── 2c. q_norm(q, per-head) — BEFORE RoPE  ────────────────────────────
        // modeling.py:102  `q = self.q_norm(q).transpose(1, 2)` — before apply_rotary_pos_emb
        if let Some(ref qn) = layer.q_norm {
            gpu.rmsnorm_batched(
                &scratch.pbs.fa_q_batch,
                qn,
                &scratch.pbs.fa_q_batch,
                block * config.n_heads,
                config.head_dim,
                config.norm_eps,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: q_norm: {e:?}"))?;
        }

        // ── 2d. Context K/V + block K/V → all_k, all_v  ───────────────────────
        // modeling.py:103  `k_ctx  = self.k_proj(target_hidden_states)` (main_x, 1 row)
        // modeling.py:104  `k_noise = self.k_proj(hidden_states)`        (block rows)
        // modeling.py:105  `v_ctx  = self.v_proj(target_hidden_states)`
        // modeling.py:106  `v_noise = self.v_proj(hidden_states)`
        // modeling.py:107  `k = cat([k_ctx, k_noise], dim=1)` → all_k[0..=block]
        // modeling.py:110  `v = cat([v_ctx, v_noise], dim=1)` → all_v[0..=block]

        // Context K at slot 0 of all_k.
        let ctx_k = scratch.all_k.sub_offset(0, kv_dim);
        weight_gemv(gpu, &layer.wk, main_x, &ctx_k)
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: k_proj(ctx): {e:?}"))?;

        // Block K at slots 1..=block of all_k (offset in ELEMENTS for F32).
        for i in 0..block {
            let x_row = scratch.pbs.x_rot_batch.sub_offset(i * dim, dim);
            let k_row = scratch.all_k.sub_offset((1 + i) * kv_dim, kv_dim);
            weight_gemv(gpu, &layer.wk, &x_row, &k_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: k_proj[{i}]: {e:?}"))?;
        }

        // Context V at slot 0 of all_v.
        let ctx_v = scratch.all_v.sub_offset(0, kv_dim);
        weight_gemv(gpu, &layer.wv, main_x, &ctx_v)
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: v_proj(ctx): {e:?}"))?;

        // Block V at slots 1..=block of all_v (offset in ELEMENTS for F32).
        for i in 0..block {
            let x_row = scratch.pbs.x_rot_batch.sub_offset(i * dim, dim);
            let v_row = scratch.all_v.sub_offset((1 + i) * kv_dim, kv_dim);
            weight_gemv(gpu, &layer.wv, &x_row, &v_row)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: v_proj[{i}]: {e:?}"))?;
        }

        // ── 2e. k_norm(all_k) — on concatenated [ctx ++ block] K, BEFORE RoPE ─
        // modeling.py:113  `k = self.k_norm(k).transpose(1, 2)`
        // all_k is [kv_cap × kv_dim] = [(1+block) × kv_dim] laid out as
        // [(1+block)*n_kv_heads] rows of [head_dim] each → rmsnorm_batched treats
        // it as (kv_cap * n_kv_heads) rows, each of head_dim floats.
        if let Some(ref kn) = layer.k_norm {
            gpu.rmsnorm_batched(
                &scratch.all_k,
                kn,
                &scratch.all_k,
                kv_cap * config.n_kv_heads,
                config.head_dim,
                config.norm_eps,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: k_norm: {e:?}"))?;
        }

        // ── 2f. RoPE on Q (block positions) and K (all kv_cap positions)  ──────
        // modeling.py:116  `q, k = apply_rotary_pos_emb(q, k, cos, sin)`
        // apply_rotary_pos_emb (modeling.py:34–40):
        //   q uses cos[..., -q_len:, :]  → block positions seed_pos+1..=+block
        //   k uses full cos              → ctx(seed_pos) + block(seed_pos+1..=+block)
        //
        // Implementation: two rope_batched_f32 calls sharing the same K buffer,
        // one for Q (positions_q_block, n_heads_k=0 trick not available → pass
        // a zero-length sub-tensor for k), one for K (positions_kv_all, n_heads_q=0).
        //
        // rope_batched_f32 signature: (q, k, positions, n_heads_q, n_heads_k, head_dim, freq, batch)
        // Setting n_heads_q=0 rotates only K; n_heads_k=0 rotates only Q.
        // This is the same trick used by the deepseek4 dspark (`n_heads=0` for kv-only RoPE).

        // RoPE on Q (only): pass all_k as a dummy k with n_heads_k=0.
        gpu.rope_batched_f32(
            &scratch.pbs.fa_q_batch,
            &scratch.all_k, // dummy k (n_heads_k=0 means it is not modified)
            &scratch.positions_q_block,
            config.n_heads,
            0, // n_heads_k=0 → skip K rotation (modeling.py:38 q uses last q_len entries)
            config.head_dim,
            config.rope_freq_base,
            block,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: rope Q: {e:?}"))?;

        // RoPE on K (only): pass fa_q_batch as dummy q with n_heads_q=0.
        gpu.rope_batched_f32(
            &scratch.pbs.fa_q_batch, // dummy q (n_heads_q=0 → skip Q rotation)
            &scratch.all_k,
            &scratch.positions_kv_all,
            0, // n_heads_q=0 → skip Q rotation
            config.n_kv_heads,
            config.head_dim,
            config.rope_freq_base,
            kv_cap, // batch = 1+block (ctx + block rows)
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: rope K: {e:?}"))?;

        // ── 2g. Write K and V to Q8 KV cache at compact slots 0..=block  ───────
        // Compact positions 0..=block (from positions_kv_all used as 0-based compact
        // slots): BUT positions_kv_all holds ABSOLUTE positions (seed_pos+0..+block).
        // We need a separate compact positions [0, 1, ..., block] for the kv write.
        //
        // We use the approach of writing context K/V (slot 0) then block K/V
        // (slots 1..=block) with the compact positions_compact + a context position=0
        // uploaded per-layer. To avoid per-layer host uploads, we use:
        //   - positions_compact for the block rows (slots 1..=block)
        //   - a single kv_cache_write call for the context row at slot 0
        //
        // Context K at compact slot 0 via single-token kv_cache_write_q8_0.
        // We upload a single i32(0) into scratch.pbs.positions slot 0 for the
        // context write (positions buffer reuse is safe here: we re-upload below).
        {
            let ctx_pos_bytes = 0i32.to_le_bytes();
            gpu.hip
                .memcpy_htod_offset(&scratch.pbs.positions.buf, 0, &ctx_pos_bytes)
                .map_err(|e| format!("dspark_qwen3 l{layer_idx}: htod ctx pos0: {e:?}"))?;
            // sub_offset(0, kv_dim): slot 0, length kv_dim elements (F32).
            let ctx_k = scratch.all_k.sub_offset(0, kv_dim);
            let ctx_v = scratch.all_v.sub_offset(0, kv_dim);
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.k_gpu[layer_idx],
                &ctx_k,
                &scratch.pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                1,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_k_ctx: {e:?}"))?;
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.v_gpu[layer_idx],
                &ctx_v,
                &scratch.pbs.positions,
                config.n_kv_heads,
                config.head_dim,
                1,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_v_ctx: {e:?}"))?;
        }

        // Block K at compact slots 1..=block.
        {
            // all_k[kv_dim..] holds the block K rows at byte offset kv_dim*4.
            // sub_offset(kv_dim, block*kv_dim): skip ctx slot (kv_dim elems), take block slots.
            let blk_k = scratch.all_k.sub_offset(kv_dim, block * kv_dim);
            let blk_v = scratch.all_v.sub_offset(kv_dim, block * kv_dim);
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.k_gpu[layer_idx],
                &blk_k,
                &scratch.positions_compact,
                config.n_kv_heads,
                config.head_dim,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_k_blk: {e:?}"))?;
            gpu.kv_cache_write_q8_0_batched(
                &scratch.kv.v_gpu[layer_idx],
                &blk_v,
                &scratch.positions_compact,
                config.n_kv_heads,
                config.head_dim,
                block,
            )
            .map_err(|e| format!("dspark_qwen3 l{layer_idx}: kv_write_v_blk: {e:?}"))?;
        }

        // ── 2h. Bidirectional masked GQA attention  ────────────────────────────
        // positions_compact = [1, 2, ..., block] (each block query is at its compact slot).
        // block_start=1, block_cols=block, bias=zeros → all block queries see all block keys.
        // Slot 0 (context) is before block_start → always visible (never masked by the kernel).
        // modeling.py:58 `self.is_causal = False`; modeling.py:137 `attn_is_causal = False`.
        gpu.attention_q8_0_kv_batched_masked(
            &scratch.pbs.fa_q_batch,
            &scratch.kv.k_gpu[layer_idx],
            &scratch.kv.v_gpu[layer_idx],
            &scratch.pbs.fa_attn_out_batch,
            &scratch.positions_compact,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            scratch.kv.physical_cap, // max_seq = kv_cap = 1+block
            kv_cap,                  // max_ctx_len = 1+block (all keys visible)
            block,                   // batch_size = block query rows
            Some(&scratch.bias),     // zero bias → bidirectional in-block
            1,                       // block_start = 1 (ctx slot 0 always visible)
            block,                   // block_cols = block
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: attn: {e:?}"))?;

        // ── 2i. o_proj(attn_out) + residual  ──────────────────────────────────
        // modeling.py:148–150  `attn_output = attn_output.reshape(...)` then `o_proj`
        // modeling.py:194      `hidden_states = residual + hidden_states`
        // gemm_hfq4g256_residual handles the MQ4 o_proj + residual accumulation.
        gpu.gemm_hfq4g256_residual(
            &layer.wo.buf,
            &scratch.pbs.fa_attn_out_batch,
            &scratch.pbs.x_batch,
            layer.wo.m,
            layer.wo.k,
            block,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: o_proj: {e:?}"))?;

        // ── 2j. post_attention_layernorm(x_batch) → x_rot_batch  ──────────────
        // modeling.py:196  `hidden_states = self.post_attention_layernorm(hidden_states)`
        gpu.rmsnorm_batched(
            &scratch.pbs.x_batch,
            &layer.ffn_norm,
            &scratch.pbs.x_rot_batch,
            block,
            dim,
            config.norm_eps,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: ffn_norm: {e:?}"))?;

        // ── 2k. MLP SwiGLU: gate/up → silu_mul → down + residual  ─────────────
        // modeling.py:197  `hidden_states = self.mlp(hidden_states)` (Qwen3MLP = SwiGLU)
        // modeling.py:198  `return residual + hidden_states`
        gpu.gemm_gate_up_hfq4g256(
            &layer.w_gate.buf,
            &layer.w_up.buf,
            &scratch.pbs.x_rot_batch,
            &scratch.pbs.gate_ffn_batch,
            &scratch.pbs.up_batch,
            layer.w_gate.m,
            layer.w_up.m,
            layer.w_gate.k,
            block,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: gate_up: {e:?}"))?;

        gpu.silu_mul_f32(
            &scratch.pbs.gate_ffn_batch,
            &scratch.pbs.up_batch,
            &scratch.pbs.ffn_hidden_batch,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: silu_mul: {e:?}"))?;

        gpu.gemm_hfq4g256_residual(
            &layer.w_down.buf,
            &scratch.pbs.ffn_hidden_batch,
            &scratch.pbs.x_batch,
            layer.w_down.m,
            layer.w_down.k,
            block,
        )
        .map_err(|e| format!("dspark_qwen3 l{layer_idx}: w_down: {e:?}"))?;
    }

    // ── 3. Final norm: output_norm(x_batch) → x_head_out  ─────────────────────
    // modeling.py:386  `return self.norm(hidden_states)`
    // Apply drafter.output_norm (= sidecar `norm.weight`) to each of the `block`
    // rows in pbs.x_batch, writing into x_head_out[block × dim].
    for i in 0..block {
        // sub_offset(i*dim, dim): row i of x_batch / x_head_out (F32, offset in elements).
        let x_row = scratch.pbs.x_batch.sub_offset(i * dim, dim);
        let out_row = x_head_out.sub_offset(i * dim, dim);
        gpu.rmsnorm_f32(&x_row, &drafter.output_norm, &out_row, config.norm_eps)
            .map_err(|e| format!("dspark_qwen3: output_norm[{i}]: {e:?}"))?;
    }

    Ok(())
}
