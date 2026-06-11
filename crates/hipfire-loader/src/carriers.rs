// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch carrier structs with object-safe [`Carrier`] impls.
//! Each carrier owns its full load path (HFQ + safetensors-dir).

use hipfire_runtime::loader_api::{ModelSource, LoadCtx};
use hipfire_runtime::model_source::ModelSource as _;
use crate::Carrier;
use crate::{LoadedModel, ModelState, finish_qwen35_load, resolve_chat_template};

// ─── Qwen2Carrier ────────────────────────────────────────────────────

pub struct Qwen2Carrier;
impl Carrier for Qwen2Carrier {
    fn name(&self) -> &'static str { "qwen2" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src, ModelSource::Hfq(h) if h.arch_id == 7)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("qwen2: pipeline-parallel (pp>1) unsupported".into()); }
        let ModelSource::Hfq(hfq) = &src else { return Err("qwen2: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        let chat_template = resolve_chat_template(hfq, ctx.path);
        let arch_id = hfq.arch_id;
        let bundle = hipfire_arch_qwen2::load_qwen2_bundle(src, ctx)?;
        Ok(LoadedModel {
            state: Some(ModelState::Qwen2(bundle)),
            ..LoadedModel::skeleton(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.to_string(), chat_template)
        })
    }
}

// ─── Qwen35Carrier ───────────────────────────────────────────────────

fn qwen35_kv_mode(ctx: &LoadCtx) -> String {
    ctx.kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default())
}

pub struct Qwen35Carrier;
impl Carrier for Qwen35Carrier {
    fn name(&self) -> &'static str { "qwen35" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src, ModelSource::Hfq(h) if matches!(h.arch_id, 5 | 6))
            || matches!(src, ModelSource::Dir(s) if matches!(s.arch_id(), 5 | 6))
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        match src {
            ModelSource::Hfq(mut hfq_file) => {
                // ── pp>1 path (pipeline-parallel) ──────────────
                if ctx.pp > 1 {
                    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_file.metadata_json)
                        .map_err(|e| format!("tokenizer not found: {e}"))?;
                    let chat_template = resolve_chat_template(&hfq_file, ctx.path);
                    let arch_id = hfq_file.arch_id;
                    let pp = ctx.pp;
                    let config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq_file)
                        .ok_or("failed to read Qwen3.5 config")?;
                    let kv_mode = ctx.kv_mode_override
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
                    let mut gpus = match std::env::var("HIPFIRE_PP_LAYERS").ok().filter(|s| !s.is_empty()) {
                        Some(spec) => {
                            let counts: Result<Vec<usize>, _> =
                                spec.split(',').map(|s| s.trim().parse::<usize>()).collect();
                            let counts = counts.map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
                            if counts.len() != pp {
                                return Err(format!("HIPFIRE_PP_LAYERS has {} entries, expected pp={}", counts.len(), pp));
                            }
                            let sum: usize = counts.iter().sum();
                            if sum != config.n_layers {
                                return Err(format!("HIPFIRE_PP_LAYERS sum={} != n_layers={}", sum, config.n_layers));
                            }
                            hipfire_runtime::multi_gpu::Gpus::init_layers(&counts).map_err(|e| format!("{e}"))?
                        }
                        None => hipfire_runtime::multi_gpu::Gpus::init_uniform(pp, config.n_layers).map_err(|e| format!("{e}"))?,
                    };
                    let weights = hipfire_arch_qwen35::qwen35::load_weights_multi(&hfq_file, &config, &mut gpus)
                        .map_err(|e| format!("{e}"))?;
                    let is_kv_layer: Vec<bool> = config.layer_types.iter().map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention).collect();
                    let kv = match kv_mode.as_str() {
                        "q8" => hipfire_runtime::llama::KvCache::new_gpu_q8_capped_multi_filtered(
                            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq),
                        "asym3" | "turbo3" | "turbo" | "auto" | "" => hipfire_runtime::llama::KvCache::new_gpu_asym3_capped_multi_filtered(
                            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq),
                        "fwht3" => hipfire_runtime::llama::KvCache::new_gpu_fwht3_capped_multi_filtered(
                            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq),
                        "fwht2" => hipfire_runtime::llama::KvCache::new_gpu_fwht2_capped_multi_filtered(
                            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq),
                        _ => {
                            eprintln!("  KV cache: unrecognized '{}', defaulting to asym3 for pp>1", kv_mode);
                            hipfire_runtime::llama::KvCache::new_gpu_asym3_capped_multi_filtered(
                                &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq)
                        }
                    }.map_err(|e| format!("{e}"))?;
                    let dn_quant = crate::parse_state_quant(ctx.state_quant_override)
                        .map_err(|e| format!("{e}"))?;
                    let (dn, la_to_device) = hipfire_arch_qwen35::qwen35::DeltaNetState::new_with_quant_multi(&mut gpus, &config, dn_quant)
                        .map_err(|e| format!("{e}"))?;
                    let scratch_set = hipfire_arch_qwen35::qwen35::Qwen35ScratchSet::new_with_kv_max_multi(
                        &mut gpus, &config, 2048, ctx.max_seq).map_err(|e| format!("{e}"))?;
                    let gpu0 = &mut gpus.devices[0];
                    let single_scratch = hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(gpu0, &config, 2048, ctx.max_seq)
                        .map_err(|e| format!("{e}"))?;
                    let bundle = hipfire_arch_qwen35::Qwen35Bundle { config, weights, scratch: single_scratch, kv_cache: kv, dn_state: dn };
                    return Ok(LoadedModel {
                        state: Some(ModelState::Qwen35(bundle)),
                        ..LoadedModel::skeleton_pp(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.to_string(), chat_template, pp, gpus, scratch_set, la_to_device)
                    });
                }

                // ── pp=1 path (single-GPU) ────────────────────
                let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_file.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
                let chat_template = resolve_chat_template(&hfq_file, ctx.path);
                let arch_id = hfq_file.arch_id;
                let physical_cap = if ctx.cask.sidecar.is_some() {
                    let env_override = std::env::var("HIPFIRE_KV_PHYSICAL_CAP")
                        .ok().and_then(|s| s.parse::<usize>().ok());
                    let safety = 256usize;
                    let floor = ctx.cask.budget + ctx.cask.beta + 4;
                    let derived = ctx.cask.budget + ctx.cask.beta + safety;
                    env_override.unwrap_or(derived).clamp(floor, ctx.max_seq)
                } else {
                    ctx.max_seq
                };

                // VL detection — loads weights from hfq_file in-place
                let (vision_config, vision_weights) = {
                    use hipfire_arch_qwen35_vl::Qwen35Vl;
                    use hipfire_runtime::arch::Architecture;
                    let has_vision = hfq_file.tensor_data("model.visual.patch_embed.proj.weight").is_some();
                    let vc = Qwen35Vl::config_from_hfq(&hfq_file).ok();
                    match vc {
                        Some(vc) if has_vision => {
                            let vw = Qwen35Vl::load_weights(&mut hfq_file, &vc, ctx.gpu)
                                .map_err(|e| eprintln!("  VL weight load failed: {e}")).ok();
                            eprintln!("  VL model: vision encoder (hidden={}, layers={})",
                                vc.hidden_size, vc.num_layers);
                            (Some(vc), vw)
                        }
                        _ => (None, None),
                    }
                };

                let bundle = hipfire_arch_qwen35::load_qwen35_bundle(ModelSource::Hfq(hfq_file), ctx)?;
                finish_qwen35_load(bundle, tokenizer, physical_cap, arch_id, chat_template, ctx, vision_config, vision_weights)
            }
            ModelSource::Dir(source) => {
                if ctx.pp > 1 { return Err("qwen35: safetensors + pp>1 unsupported".into()); }
                let arch_id = source.arch_id();
                let qm = source.quant_config().map(|q| q.method.as_str()).unwrap_or("none");
                eprintln!("  safetensors arch_id={arch_id}, quant_method={qm}");

                let tokenizer = if let Some(tok_path) = source.tokenizer_json_path() {
                    hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
                        .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
                        .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))?
                } else {
                    return Err("no tokenizer.json found in model directory".into());
                };
                let chat_template = source.chat_template();

                let config = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source)
                    .ok_or("failed to parse Qwen3.5 config from config.json")?;
                let weights = hipfire_arch_qwen35::qwen35::load_weights_paroquant(&source, &config, ctx.gpu)
                    .map_err(|e| format!("load_weights_paroquant: {e:?}"))?;
                let is_kv_layer: Vec<bool> = config
                    .layer_types.iter().map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention).collect();
                let kv_mode = qwen35_kv_mode(ctx);
                let kv_cache = match kv_mode.as_str() {
                    "q8" => hipfire_runtime::llama::KvCache::new_gpu_q8_capped_filtered(
                        ctx.gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                    "asym4" | "turbo4" => hipfire_runtime::llama::KvCache::new_gpu_asym4_filtered(
                        ctx.gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq,
                    ),
                    "asym2" | "turbo2" => hipfire_runtime::llama::KvCache::new_gpu_asym2_filtered(
                        ctx.gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq,
                    ),
                    "fwht4" => hipfire_runtime::llama::KvCache::new_gpu_fwht4_filtered(
                        ctx.gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq,
                    ),
                    "fwht3" => hipfire_runtime::llama::KvCache::new_gpu_fwht3_capped_filtered(
                        ctx.gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                    "fwht2" => hipfire_runtime::llama::KvCache::new_gpu_fwht2_capped_filtered(
                        ctx.gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                    _ => hipfire_runtime::llama::KvCache::new_gpu_asym3_capped_filtered(
                        ctx.gpu, &is_kv_layer, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                }.map_err(|e| format!("KvCache: {e}"))?;

                let dn_state = hipfire_arch_qwen35::qwen35::DeltaNetState::new(ctx.gpu, &config)
                    .map_err(|e| format!("DeltaNetState::new: {e:?}"))?;
                let scratch = hipfire_arch_qwen35::qwen35::Qwen35Scratch::new(ctx.gpu, &config, 256)
                    .map_err(|e| format!("Qwen35Scratch::new: {e:?}"))?;

                let bundle = hipfire_arch_qwen35::Qwen35Bundle { config, weights, scratch, kv_cache, dn_state };
                Ok(LoadedModel {
                    state: Some(ModelState::Qwen35(bundle)),
                    ..LoadedModel::skeleton(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.to_string(), chat_template)
                })
            }
        }
    }
}

// ─── LlamaCarrier ────────────────────────────────────────────────────

pub struct LlamaCarrier;
impl Carrier for LlamaCarrier {
    fn name(&self) -> &'static str { "llama" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src, ModelSource::Hfq(h) if h.arch_id < 5)
            || matches!(src, ModelSource::Dir(s) if matches!(s.arch_id(), 0 | 1))
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        match src {
            ModelSource::Hfq(hfq) => {
                if ctx.pp > 1 { return Err("llama: pipeline-parallel (pp>1) unsupported".into()); }
                let hfq_file = &hfq;
                let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq_file.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
                let chat_template = resolve_chat_template(hfq_file, ctx.path);
                let arch_id = hfq_file.arch_id;
                let bundle = hipfire_arch_llama::load_llama_bundle(ModelSource::Hfq(hfq), ctx)?;
                Ok(LoadedModel {
                    state: Some(ModelState::Llama(bundle)),
                    ..LoadedModel::skeleton(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.to_string(), chat_template)
                })
            }
            ModelSource::Dir(source) => {
                if ctx.pp > 1 { return Err("llama: safetensors + pp>1 unsupported".into()); }
                let arch_id = source.arch_id();
                eprintln!("  safetensors arch_id={arch_id}");

                let tokenizer = if let Some(tok_path) = source.tokenizer_json_path() {
                    hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
                        .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
                        .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))?
                } else {
                    return Err("no tokenizer.json found in model directory".into());
                };
                let chat_template = source.chat_template();

                let config = hipfire_runtime::hfq::config_from_safetensors_llama(&source)
                    .ok_or("failed to parse LLaMA/Qwen3 config from config.json")?;
                let weights = hipfire_runtime::hfq::load_weights_paroquant_llama(&source, &config, ctx.gpu)
                    .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;
                let kv_mode = {
                    ctx.kv_mode_override
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default())
                };
                let asym3_auto = matches!(kv_mode.as_str(), "turbo3" | "turbo" | "auto" | "");
                let kv = match kv_mode.as_str() {
                    "q8" => hipfire_runtime::llama::KvCache::new_gpu_q8_capped(
                        ctx.gpu, config.n_layers, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                    "asym4" | "turbo4" => hipfire_runtime::llama::KvCache::new_gpu_asym4_capped(
                        ctx.gpu, config.n_layers, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                    "asym3" => hipfire_runtime::llama::KvCache::new_gpu_asym3_capped(
                        ctx.gpu, config.n_layers, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                    _ if asym3_auto && config.head_dim == 256 => hipfire_runtime::llama::KvCache::new_gpu_asym3_capped(
                        ctx.gpu, config.n_layers, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                    _ => hipfire_runtime::llama::KvCache::new_gpu_q8_capped(
                        ctx.gpu, config.n_layers, config.n_kv_heads, config.head_dim, ctx.max_seq, ctx.max_seq,
                    ),
                }.map_err(|e| format!("KvCache: {e}"))?;
                let scratch = hipfire_runtime::llama::ForwardScratch::new(ctx.gpu, &config)
                    .map_err(|e| format!("ForwardScratch::new: {e:?}"))?;

                let bundle = hipfire_arch_llama::LlamaBundle { config, weights, scratch, kv };
                Ok(LoadedModel {
                    state: Some(ModelState::Llama(bundle)),
                    ..LoadedModel::skeleton(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.to_string(), chat_template)
                })
            }
        }
    }
}

// ─── Non-core carriers ───────────────────────────────────────────────

pub struct Deepseek4Carrier;
impl Carrier for Deepseek4Carrier {
    fn name(&self) -> &'static str { "deepseek4" }
    fn probe(&self, src: &ModelSource) -> bool { matches!(src, ModelSource::Hfq(h) if h.arch_id == 9) }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("deepseek4: pp>1 unsupported via registry".into()); }
        let ModelSource::Hfq(hfq) = src else { return Err("deepseek4: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        crate::load_deepseek4(hfq, tokenizer, ctx.gpu, ctx.max_seq, ctx.path)
    }
}

pub struct DotsOcrCarrier;
impl Carrier for DotsOcrCarrier {
    fn name(&self) -> &'static str { "dots_ocr" }
    fn probe(&self, src: &ModelSource) -> bool { matches!(src, ModelSource::Hfq(h) if h.arch_id == 8) }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("dots_ocr: pp>1 unsupported via registry".into()); }
        let ModelSource::Hfq(hfq) = src else { return Err("dots_ocr: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        crate::load_dots_ocr(hfq, tokenizer, ctx.gpu, ctx.max_seq, ctx.path)
    }
}

pub struct Lfm2MoeCarrier;
impl Carrier for Lfm2MoeCarrier {
    fn name(&self) -> &'static str { "lfm2moe" }
    fn probe(&self, src: &ModelSource) -> bool { matches!(src, ModelSource::Hfq(h) if h.arch_id == 11) }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("lfm2moe: pp>1 unsupported via registry".into()); }
        let ModelSource::Hfq(hfq) = src else { return Err("lfm2moe: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        crate::load_lfm2moe(hfq, tokenizer, ctx.gpu, ctx.max_seq, ctx.path)
    }
}

pub struct MinimaxCarrier;
impl Carrier for MinimaxCarrier {
    fn name(&self) -> &'static str { "minimax" }
    fn probe(&self, src: &ModelSource) -> bool { matches!(src, ModelSource::Hfq(h) if h.arch_id == 10) }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("minimax: pp>1 unsupported via registry".into()); }
        let ModelSource::Hfq(hfq) = src else { return Err("minimax: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        crate::load_minimax(hfq, tokenizer, ctx.gpu, ctx.max_seq, ctx.path)
    }
}
