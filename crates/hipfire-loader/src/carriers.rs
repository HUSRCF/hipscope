// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch carrier structs with object-safe [`Carrier`] impls.
//! Each carrier owns its full load path (HFQ arm in this file;
//! safetensors-dir and pp>1 arms added in Tasks 5–6).

use hipfire_runtime::loader_api::{ModelSource, LoadCtx};
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

pub struct Qwen35Carrier;
impl Carrier for Qwen35Carrier {
    fn name(&self) -> &'static str { "qwen35" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src, ModelSource::Hfq(h) if matches!(h.arch_id, 5 | 6))
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("qwen35: pp>1 not yet supported via HFQ carrier (use top-level pp path)".into()); }
        // Destructure src to own the HfqFile for VL detection, then repackage
        let ModelSource::Hfq(mut hfq_file) = src else {
            return Err("qwen35: directory source unsupported".into());
        };
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
}

// ─── LlamaCarrier ────────────────────────────────────────────────────

pub struct LlamaCarrier;
impl Carrier for LlamaCarrier {
    fn name(&self) -> &'static str { "llama" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src, ModelSource::Hfq(h) if h.arch_id < 5)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 { return Err("llama: pipeline-parallel (pp>1) unsupported".into()); }
        let ModelSource::Hfq(hfq) = &src else { return Err("llama: directory source unsupported".into()); };
        let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .map_err(|e| format!("tokenizer not found: {e}"))?;
        let chat_template = resolve_chat_template(hfq, ctx.path);
        let arch_id = hfq.arch_id;
        let bundle = hipfire_arch_llama::load_llama_bundle(src, ctx)?;
        Ok(LoadedModel {
            state: Some(ModelState::Llama(bundle)),
            ..LoadedModel::skeleton(arch_id, tokenizer, ctx.max_seq, ctx.max_seq, ctx.path.to_string(), chat_template)
        })
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
