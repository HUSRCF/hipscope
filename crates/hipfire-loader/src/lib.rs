// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Top-of-DAG model loader. Owns `LoadedModel`, the carrier registry,
//! and `load_model` — the single arch-dispatch point for the daemon.

mod carriers;
pub use carriers::*;

use std::path::Path;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, LayerType, Qwen35ScratchSet};
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_arch_llama::LlamaBundle;
use hipfire_arch_qwen35::speculative::{
    DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::loader_api::{CaskConfig, ModelSource, LoadCtx};
use hipfire_runtime::llama;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use rdna_compute::Gpu;

// ─── Object-safe Carrier trait ──────────────────────────────────────

/// One arch's complete load contract. Object-safe → usable as `&dyn Carrier`.
pub trait Carrier: Send + Sync {
    fn name(&self) -> &'static str;
    fn probe(&self, src: &ModelSource) -> bool;
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String>;
}

// ─── Registry ─────────────────────────────────────────────────────────

use crate::carriers::*;
const REGISTRY: &[&dyn Carrier] = &[
    &Qwen2Carrier, &Qwen35Carrier, &LlamaCarrier,
    &DotsOcrCarrier, &Deepseek4Carrier, &MinimaxCarrier, &Lfm2MoeCarrier,
];

// ─── Constants ────────────────────────────────────────────────────────

/// Built-in Qwen3.5/3.6 chat template (froggeric/Qwen at HF).
/// Used when no per-model or env-override template is available.
const FROGGERIC_QWEN35_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/qwen35-froggeric-v20.jinja");

/// Built-in LFM2.5 chat template.
const LFM2_TEMPLATE: &str =
    include_str!("../../hipfire-runtime/templates/eval/lfm2-liquidai.jinja");

// ─── Eviction policy wrapper ──────────────────────────────────────────

/// Eviction policy wrapper — dispatches to plain TriAttention or CASK m-folding.
pub enum Eviction {
    Plain(EvictionCtx),
    Cask(CaskCtx),
}

impl Eviction {
    pub fn maybe_evict(
        &self,
        gpu: &mut rdna_compute::Gpu,
        kv: &mut llama::KvCache,
        physical: usize,
    ) -> hip_bridge::HipResult<Option<hipfire_runtime::triattn::EvictionResult>> {
        match self {
            Eviction::Plain(c) => c.maybe_evict(gpu, kv, physical),
            Eviction::Cask(c) => c.maybe_evict(gpu, kv, physical),
        }
    }
    pub fn budget(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.budget,
            Eviction::Cask(c) => c.base.budget,
        }
    }
    pub fn beta(&self) -> usize {
        match self {
            Eviction::Plain(c) => c.beta,
            Eviction::Cask(c) => c.base.beta,
        }
    }
    pub fn free_gpu(self, gpu: &mut rdna_compute::Gpu) {
        match self {
            Eviction::Plain(c) => c.free_gpu(gpu),
            Eviction::Cask(c) => c.free_gpu(gpu),
        }
    }
}

// ─── DDTree side state ────────────────────────────────────────────────

/// Side state for DDTree-mode speculative decoding.
pub struct DdtreeState {
    pub post_seed_snap: DeltaNetSnapshot,
    pub scratch: DdtreeScratch,
    pub budget: usize,
    pub topk: usize,
    pub path_c_parent_pre_snap: DeltaNetSnapshot,
    pub path_c_main_end_snap: DeltaNetSnapshot,
}

// ─── DFlash state ─────────────────────────────────────────────────────

/// Optional DFlash speculative-decoding state.
pub struct DflashState {
    pub draft_config: DflashConfig,
    pub draft_weights: DflashWeights,
    pub draft_scratch: DflashScratch,
    pub hidden_rb: HiddenStateRingBuffer,
    pub verify_scratch: VerifyScratch,
    pub target_snap: DeltaNetSnapshot,
    pub gdn_tape: GdnTape,
    pub target_hidden_host: Vec<f32>,
    pub ctx_capacity: usize,
    pub block_size: usize,
    pub ddtree: Option<DdtreeState>,
}

// ─── AsstTurnCache ────────────────────────────────────────────────────

/// Per-turn token cache for V4F prefix-cache stability.
pub struct AsstTurnCache {
    cap: Option<usize>,
    map: std::collections::HashMap<u64, Vec<u32>>,
    order: std::collections::VecDeque<u64>,
}

impl AsstTurnCache {
    pub fn new_from_env() -> Self {
        let unbounded = std::env::var("HIPFIRE_PROMPT_CACHE_UNBOUNDED")
            .ok()
            .as_deref()
            == Some("1");
        let cap = if unbounded {
            None
        } else {
            Some(
                std::env::var("HIPFIRE_PROMPT_CACHE_CAP")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(32),
            )
        };
        Self {
            cap,
            map: std::collections::HashMap::new(),
            order: std::collections::VecDeque::new(),
        }
    }

    pub fn touch_mru(&mut self, fp: u64) {
        if let Some(pos) = self.order.iter().position(|k| *k == fp) {
            self.order.remove(pos);
        }
        self.order.push_back(fp);
    }

    pub fn contains_key(&self, fp: &u64) -> bool {
        self.map.contains_key(fp)
    }

    pub fn get(&mut self, fp: &u64) -> Option<&Vec<u32>> {
        if self.map.contains_key(fp) {
            self.touch_mru(*fp);
            self.map.get(fp)
        } else {
            None
        }
    }

    pub fn insert(&mut self, fp: u64, tokens: Vec<u32>) {
        if self.map.contains_key(&fp) {
            self.map.insert(fp, tokens);
            self.touch_mru(fp);
            return;
        }
        if let Some(c) = self.cap {
            while self.order.len() >= c {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                } else {
                    break;
                }
            }
        }
        self.map.insert(fp, tokens);
        self.order.push_back(fp);
    }
}

// ─── ModelState ────────────────────────────────────────────────────────

/// Arch-specific core state, dispatched in `LoadedModel.state`.
/// Shared fields (kv_cache, dn_state) stay on `LoadedModel` directly.
pub enum ModelState {
    Qwen2(hipfire_arch_qwen2::Qwen2Bundle),
    Qwen35(hipfire_arch_qwen35::Qwen35Bundle),
    Llama(hipfire_arch_llama::LlamaBundle),
}

// ─── LoadedModel ──────────────────────────────────────────────────────

pub struct LoadedModel {
    pub arch_id: u32,
    pub pp: usize,
    pub pp_gpus: Option<Gpus>,
    pub pp_scratch_set: Option<Qwen35ScratchSet>,
    pub pp_dn_la_to_device: Option<Vec<u8>>,
    pub ep: Option<EpState>,
    // Shared arch state
    pub state: Option<ModelState>,
    pub kv_cache: Option<llama::KvCache>,
    pub dn_state: Option<DeltaNetState>,
    // Reusable Qwen2 recurrent state (used by dots_ocr and Qwen2 non-core falcon)
    pub qwen2_state: Option<qwen2::Qwen2State>,
    // DeepSeek V4 Flash state
    pub deepseek4_config: Option<hipfire_arch_deepseek4::DeepseekV4Config>,
    pub deepseek4_weights: Option<hipfire_arch_deepseek4::DeepseekV4Weights>,
    pub deepseek4_state: Option<hipfire_arch_deepseek4::DeepseekV4State>,
    pub deepseek4_pbs: Option<hipfire_arch_deepseek4::forward::PrefillBatchScratch>,
    pub deepseek4_eos_tok: u32,
    // LFM2.5-8B-A1B state
    pub lfm2moe_config: Option<lfm2moe::config::Lfm2MoeConfig>,
    pub lfm2moe_weights: Option<lfm2moe::lfm2moe::Lfm2MoeWeights>,
    pub lfm2moe_state: Option<lfm2moe::lfm2moe::Lfm2MoeState>,
    pub lfm2moe_eos_tok: u32,
    // MiniMax-M2 state
    pub minimax_config: Option<minimax::MiniMaxConfig>,
    pub minimax_weights: Option<minimax::MiniMaxWeights>,
    pub minimax_state: Option<minimax::MiniMaxState>,
    pub minimax_eos_tok: u32,
    // MTP config
    pub mtp_mode: String,
    pub mtp_k: usize,
    pub mtp_weights_present: bool,
    // dots.ocr state
    pub dots_ocr_config: Option<dots_ocr::DotsOcrConfig>,
    pub dots_ocr_weights: Option<dots_ocr::DotsOcrWeights>,
    // Vision state
    pub vision_config: Option<qwen35_vl::VisionConfig>,
    pub vision_weights: Option<qwen35_vl::VisionWeights>,
    // Shared
    pub tokenizer: Option<hipfire_runtime::tokenizer::Tokenizer>,
    pub seq_pos: usize,
    pub max_seq: usize,
    pub physical_cap: usize,
    pub eviction: Option<Eviction>,
    pub kv_adaptive: Option<hipfire_runtime::kv_adaptive::KvAdaptive>,
    pub conversation_tokens: Vec<u32>,
    pub prefill_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub dflash_checkpoints: Vec<(usize, DeltaNetSnapshot)>,
    pub asst_turn_cache: AsstTurnCache,
    pub decoded_vocab: Option<std::sync::Arc<Vec<String>>>,
    pub model_path: String,
    pub dflash: Option<DflashState>,
    pub chat_template: Option<String>,
}

impl LoadedModel {
    /// Shared-field skeleton: arch state None, pp = 1, all non-core arch slots
    /// None, collections empty, mtp defaults, asst cache from env. Callers set
    /// only the fields they own via struct-update (`..LoadedModel::skeleton(..)`).
    pub fn skeleton(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
    ) -> Self {
        LoadedModel {
            arch_id, pp: 1, ep: None,
            pp_gpus: None, pp_scratch_set: None, pp_dn_la_to_device: None,
            state: None, kv_cache: None, dn_state: None, qwen2_state: None,
            deepseek4_config: None, deepseek4_weights: None, deepseek4_state: None,
            deepseek4_pbs: None, deepseek4_eos_tok: 0,
            lfm2moe_config: None, lfm2moe_weights: None, lfm2moe_state: None, lfm2moe_eos_tok: 0,
            minimax_config: None, minimax_weights: None, minimax_state: None, minimax_eos_tok: 0,
            mtp_mode: "auto".to_string(), mtp_k: 3, mtp_weights_present: false,
            dots_ocr_config: None, dots_ocr_weights: None,
            vision_config: None, vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0, max_seq, physical_cap,
            eviction: None, kv_adaptive: None,
            conversation_tokens: Vec::new(),
            asst_turn_cache: AsstTurnCache::new_from_env(),
            prefill_checkpoints: Vec::new(), dflash_checkpoints: Vec::new(),
            decoded_vocab: None,
            model_path,
            dflash: None,
            chat_template,
        }
    }

    /// pp>1 skeleton — sets all four load-bearing multi-GPU fields together so
    /// they cannot be set piecemeal (a dropped `pp_scratch_set` is a silent
    /// VRAM leak; `pp_gpus`/`pp_dn_la_to_device` are `.expect()`ed in unload).
    pub fn skeleton_pp(
        arch_id: u32,
        tokenizer: hipfire_runtime::tokenizer::Tokenizer,
        max_seq: usize,
        physical_cap: usize,
        model_path: String,
        chat_template: Option<String>,
        pp: usize,
        pp_gpus: Gpus,
        pp_scratch_set: Qwen35ScratchSet,
        pp_dn_la_to_device: Vec<u8>,
    ) -> Self {
        LoadedModel {
            pp,
            pp_gpus: Some(pp_gpus),
            pp_scratch_set: Some(pp_scratch_set),
            pp_dn_la_to_device: Some(pp_dn_la_to_device),
            ..LoadedModel::skeleton(arch_id, tokenizer, max_seq, physical_cap, model_path, chat_template)
        }
    }
}

/// Expert-parallel serving state.
pub struct EpState {
    pub gpus: Gpus,
    pub inner: EpArch,
}

pub enum EpArch {
    Ds4 {
        config: hipfire_arch_deepseek4::DeepseekV4Config,
        weights: Vec<hipfire_arch_deepseek4::DeepseekV4Weights>,
        state: Vec<hipfire_arch_deepseek4::DeepseekV4State>,
        partials: Vec<rdna_compute::GpuTensor>,
    },
    Minimax {
        config: minimax::MiniMaxConfig,
        weights: Vec<minimax::MiniMaxWeights>,
        state: Vec<minimax::MiniMaxState>,
        partials: Vec<rdna_compute::GpuTensor>,
    },
}

// ─── Helper functions ─────────────────────────────────────────────────

fn resolve_chat_template(hfq: &HfqFile, model_path: &str) -> Option<String> {
    if let Ok(env_path) = std::env::var("HIPFIRE_CHAT_TEMPLATE_FILE") {
        if !env_path.is_empty() {
            match std::fs::read_to_string(&env_path) {
                Ok(s) => {
                    eprintln!("[chat_template] using HIPFIRE_CHAT_TEMPLATE_FILE={}", env_path);
                    return Some(s);
                }
                Err(e) => eprintln!(
                    "[chat_template] HIPFIRE_CHAT_TEMPLATE_FILE={env_path} failed to read ({e}); falling through"
                ),
            }
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let basename = std::path::Path::new(model_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !basename.is_empty() {
            let per_model = std::path::Path::new(&home)
                .join(".hipfire")
                .join("templates")
                .join(format!("{basename}.j2"));
            if per_model.is_file() {
                match std::fs::read_to_string(&per_model) {
                    Ok(s) => {
                        eprintln!(
                            "[chat_template] using per-model override {}",
                            per_model.display()
                        );
                        return Some(s);
                    }
                    Err(e) => eprintln!(
                        "[chat_template] per-model file {} failed to read ({e}); falling through",
                        per_model.display()
                    ),
                }
            }
        }
    }
    match hfq.arch_id {
        5 | 6 => return Some(FROGGERIC_QWEN35_TEMPLATE.to_string()),
        11 => {
            if let Some(t) = hfq.chat_template() {
                return Some(t);
            }
            return Some(LFM2_TEMPLATE.to_string());
        }
        _ => {}
    }
    hfq.chat_template()
}

fn parse_state_quant(
    mode: Option<&str>,
) -> Result<hipfire_arch_qwen35::qwen35::StateQuant, String> {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match mode.unwrap_or("q8").to_ascii_lowercase().as_str() {
        "" | "auto" | "q8" | "int8" => Ok(StateQuant::Q8),
        "fp32" | "f32" => Ok(StateQuant::FP32),
        "q4" | "int4" => Ok(StateQuant::Q4),
        other => Err(format!(
            "unsupported DeltaNet state_quant '{other}' (expected q8|fp32|q4)"
        )),
    }
}

fn state_quant_label(q: hipfire_arch_qwen35::qwen35::StateQuant) -> &'static str {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    match q {
        StateQuant::FP32 => "FP32",
        StateQuant::Q8 => "Q8",
        StateQuant::Q4 => "Q4",
    }
}

fn hfq_parameter_count(hfq: &HfqFile) -> u128 {
    hfq.tensors()
        .iter()
        .map(|t| {
            t.shape
                .iter()
                .fold(1u128, |acc, &dim| acc.saturating_mul(dim as u128))
        })
        .sum()
}

fn warn_tiny_model_state(hfq: &HfqFile, q: hipfire_arch_qwen35::qwen35::StateQuant) {
    use hipfire_arch_qwen35::qwen35::StateQuant;
    const TINY_MODEL_PARAMS: u128 = 2_000_000_000;
    let params = hfq_parameter_count(hfq);
    if params < TINY_MODEL_PARAMS && q != StateQuant::FP32 {
        eprintln!(
            "  warning: model has ~{:.2}B params; FP32 DeltaNet state is recommended below 2B for long-generation coherence (current: {})",
            params as f64 / 1.0e9,
            state_quant_label(q)
        );
    }
}

fn parse_kv_adaptive(
    s: &str,
) -> Option<(
    Option<hipfire_runtime::kv_adaptive::Preset>,
    hipfire_runtime::kv_adaptive::KMode,
    llama::VMode,
)> {
    use hipfire_runtime::kv_adaptive::{KMode, Preset};
    use llama::VMode;
    match s {
        "" | "off" => None,
        "conservative" => Some((Some(Preset::Conservative), KMode::Fwht4, VMode::Lloyd4)),
        "balanced" => Some((Some(Preset::Balanced), KMode::Fwht2, VMode::Lloyd2)),
        "aggressive" => Some((Some(Preset::Aggressive), KMode::Fwht2, VMode::Lloyd2)),
        other if other.starts_with("advanced:") => {
            let spec = &other["advanced:".len()..];
            let mut k = None;
            let mut v = None;
            for kvp in spec.split(',') {
                let mut it = kvp.splitn(2, '=');
                match (it.next(), it.next()) {
                    (Some("k"), Some("fwht4")) => k = Some(KMode::Fwht4),
                    (Some("k"), Some("fwht3")) => k = Some(KMode::Fwht3),
                    (Some("k"), Some("fwht2")) => k = Some(KMode::Fwht2),
                    (Some("v"), Some("lloyd4")) => v = Some(VMode::Lloyd4),
                    (Some("v"), Some("lloyd3")) => v = Some(VMode::Lloyd3),
                    (Some("v"), Some("lloyd2")) => v = Some(VMode::Lloyd2),
                    _ => {}
                }
            }
            match (k, v) {
                (Some(k), Some(v)) => Some((None, k, v)),
                _ => {
                    eprintln!("[daemon] kv_adaptive='{other}' malformed — expected advanced:k=<fwht4|fwht3|fwht2>,v=<lloyd4|lloyd3|lloyd2>; ignoring");
                    None
                }
            }
        }
        other => {
            eprintln!("[daemon] kv_adaptive='{other}' unknown — expected off|conservative|balanced|aggressive|advanced:k=..,v=..; ignoring");
            None
        }
    }
}

// ─── Load functions ───────────────────────────────────────────────────

// ─── Core arch carrier load ─────────────────────────────────────────────

/// Build a `LoadedModel` from a carrier `Bundle`, shared fields, and
/// eviction/DFlash state. This is the common body for qwen35 dispatch
/// where eviction and DFlash need per-arch type info.
fn finish_qwen35_load(
    bundle: Qwen35Bundle,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    physical_cap: usize,
    arch_id: u32,
    chat_template: Option<String>,
    ctx: &mut LoadCtx,
    vision_config: Option<qwen35_vl::VisionConfig>,
    vision_weights: Option<qwen35_vl::VisionWeights>,
) -> Result<LoadedModel, String> {
    use hipfire_arch_qwen35::qwen35::LayerType;
    // Extract references for eviction/DFlash setup (borrow, don't move)
    let config = &bundle.config;
    let dn_state = &bundle.dn_state;
    // ── Eviction ───────────────────────────────────────────────────
    let eviction = if let Some(ref sidecar_path) = ctx.cask.sidecar {
        let centers = TriAttnCenters::load(Path::new(sidecar_path)).map_err(|e| {
            use std::io::ErrorKind;
            let p = Path::new(sidecar_path);
            let why = match e.kind() {
                ErrorKind::NotFound if p.symlink_metadata().is_ok() =>
                    format!("dangling symlink (target absent): {sidecar_path}"),
                ErrorKind::NotFound => format!("file not found: {sidecar_path}"),
                ErrorKind::InvalidData => format!("bad format ({e}): {sidecar_path}"),
                ErrorKind::UnexpectedEof => format!("truncated/corrupt sidecar: {sidecar_path}"),
                _ => format!("read error ({e}): {sidecar_path}"),
            };
            format!("cask sidecar load failed — {why} (regen: hipfire sidecar-gen, or HIPFIRE_CASK_OFF=1)")
        })?;
        let fa_layer_ids: Vec<usize> = config
            .layer_types.iter().enumerate()
            .filter_map(|(i, t)| if *t == LayerType::FullAttention { Some(i) } else { None })
            .collect();
        if fa_layer_ids.is_empty() {
            eprintln!("  cask_sidecar set but model has no FullAttention layers — ignoring");
            None
        } else {
            let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
            let base = EvictionCtx::new(
                ctx.gpu, &centers, fa_layer_ids, ctx.cask.budget, ctx.cask.beta,
                config.n_heads, config.n_kv_heads, config.head_dim, n_rot,
                config.rope_theta, physical_cap,
            ).map_err(|e| format!("build EvictionCtx: {e}"))?;
            if ctx.cask.cask_m_folding {
                eprintln!("  eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                    ctx.cask.core_frac, ctx.cask.fold_m, ctx.cask.budget, ctx.cask.beta, physical_cap);
                Some(Eviction::Cask(CaskCtx::new(base, ctx.cask.core_frac, ctx.cask.fold_m)))
            } else {
                eprintln!("  eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                    ctx.cask.budget, ctx.cask.beta, physical_cap);
                Some(Eviction::Plain(base))
            }
        }
    } else {
        None
    };

    // ── DFlash ─────────────────────────────────────────────────────
    let dflash = if let Some(dp) = ctx.draft_path {
        match load_dflash_state(dp, physical_cap, config, dn_state, ctx.gpu) {
            Ok(s) => {
                eprintln!("  DFlash draft loaded: {} (layers={}, hidden={}, block={})",
                    dp, s.draft_config.n_layers, s.draft_config.hidden, s.draft_config.block_size);
                Some(s)
            }
            Err(e) => {
                eprintln!("  DFlash draft load failed ({}): {} — falling back to AR only", dp, e);
                None
            }
        }
    } else {
        None
    };

    let state = Some(ModelState::Qwen35(bundle));
    Ok(LoadedModel {
        state, eviction, dflash,
        vision_config, vision_weights,
        max_seq: ctx.max_seq,
        ..LoadedModel::skeleton(arch_id, tokenizer, ctx.max_seq, physical_cap, ctx.path.to_string(), chat_template)
    })
}

// ─── Main public API ──────────────────────────────────────────────────

/// Load a model from an HFQ file (or safetensors directory). This is the
/// single arch-dispatch point via the carrier registry.
pub fn load_model(
    path: &str,
    max_seq: usize,
    draft_path: Option<&str>,
    kv_mode_override: Option<&str>,
    kv_adaptive_override: Option<&str>,
    state_quant_override: Option<&str>,
    cask: &CaskConfig,
    pp: usize,
    gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    if pp > 1 {
        let _ = (draft_path, cask, kv_adaptive_override);
        return load_model_pp(path, max_seq, kv_mode_override, state_quant_override, pp, gpu);
    }

    let src = ModelSource::from_path(path)?;

    // DFlash lm_head quant check — only for HFQ sources
    if draft_path.is_some() {
        if let ModelSource::Hfq(ref hfq) = src {
            let lm_qt = hfq
                .tensor_data("lm_head.weight")
                .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
                .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
                .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
                .map(|(info, _)| info.quant_type);
            let arch_is_gfx11 = matches!(
                gpu.arch.as_str(),
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1150" | "gfx1151" | "gfx1200" | "gfx1201"
            );
            let supported = match lm_qt {
                Some(3 | 6 | 13) => true,
                Some(17) => arch_is_gfx11,
                _ => false,
            };
            if !supported {
                let qt_desc = match lm_qt {
                    Some(qt) => format!("quant_type={qt}"),
                    None => "no lm_head/embed_tokens tensor found at any known name".to_string(),
                };
                return Err(format!(
                    "DFlash draft requested but target lm_head {} is not \
                     supported by speculative.rs's batched GEMM paths on this arch \
                     ({}). Supported: Q8_0 (qt=3), HFQ4G256 (qt=6), MQ4G256 (qt=13) \
                     always; MQ3G256 (qt=17) on gfx11 only. Other dtypes \
                     (MQ2 qt=18, MQ6/MQ8, HFQ3/HFQ2, HFQ4G128, HFQ6, F16, …) fall \
                     through to a per-row GEMV that hangs verify. Reload without a \
                     draft, or use an MQ4 / HFQ4 / Q8 target.",
                    qt_desc, gpu.arch
                ));
            }
            let arch_is_dense_qwen35 = hfq.arch_id == 5;
            let mq3_supported = arch_is_gfx11 && arch_is_dense_qwen35;
            let mq_unsupported = hfq
                .first_tensor_with_quant_type(18)
                .map(|n| ("MQ2 (qt=18)", n));
            let mq_unsupported = mq_unsupported.or_else(|| {
                if !mq3_supported {
                    hfq.first_tensor_with_quant_type(17)
                        .map(|n| ("MQ3 (qt=17)", n))
                } else {
                    None
                }
            });
            if let Some((qt_label, name)) = mq_unsupported {
                let arch_reason = if !arch_is_dense_qwen35 && qt_label.starts_with("MQ3") {
                    format!("arch_id={} (MoE/A3B-class) has no MQ3 MoE kernels", hfq.arch_id)
                } else {
                    format!("arch={} lacks the corresponding batched WMMA prefill family", gpu.arch)
                };
                return Err(format!(
                    "DFlash draft requested but model contains {qt_label} weight \
                     `{name}` and {arch_reason}. The prefill fast-path falls back \
                     to per-token `forward_scratch` for every spec verify cycle \
                     (or worse, a kernel-stride mismatch on MoE) — defeating \
                     DFlash's speedup. Reload without a draft, or use an MQ4 / \
                     HFQ4 / Q8 target.",
                ));
            }
        }
    }

    let mut ctx = LoadCtx {
        path, max_seq, draft_path,
        kv_mode_override, kv_adaptive_override, state_quant_override,
        cask, pp, gpu,
    };

    // Carrier registry dispatch
    let carrier = REGISTRY.iter().find(|c| c.probe(&src))
        .ok_or_else(|| format!("no carrier for arch_id={:?}", src.arch_id()))?;
    carrier.load(src, &mut ctx)
}

fn load_dots_ocr(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
) -> Result<LoadedModel, String> {
    use hipfire_arch_dots_ocr::DotsOcr;
    use hipfire_runtime::arch::Architecture;
    let config = <DotsOcr as Architecture>::config_from_hfq(&hfq)?;
    let weights = <DotsOcr as Architecture>::load_weights(&mut hfq, &config, gpu)?;
    let state = qwen2::Qwen2State::new_with_max_seq(gpu, &config.text, max_seq)
        .map_err(|e| format!("dots-ocr: Qwen2State::new_with_max_seq failed: {e:?}"))?;
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        qwen2_state: Some(state),
        dots_ocr_config: Some(config), dots_ocr_weights: Some(weights),
        ..LoadedModel::skeleton(hfq.arch_id, tokenizer, max_seq, max_seq, path.to_string(), chat_template)
    })
}

fn load_deepseek4(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
    let weights = <deepseek4::DeepseekV4 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
    let state = deepseek4::DeepseekV4State::new(&config)?;
    let pbs_max_batch: usize = std::env::var("HIPFIRE_DEEPSEEK4_PP_BATCH")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let pbs = deepseek4::forward::PrefillBatchScratch::new(gpu, &config, pbs_max_batch)?;
    let eos_tok: u32 = {
        let ids = tokenizer.encode("<｜end▁of▁sentence｜>");
        if ids.len() == 1 { ids[0] } else { 1 }
    };
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        deepseek4_config: Some(config), deepseek4_weights: Some(weights),
        deepseek4_state: Some(state), deepseek4_pbs: Some(pbs),
        deepseek4_eos_tok: eos_tok,
        ..LoadedModel::skeleton(hfq.arch_id, tokenizer, max_seq, max_seq, path.to_string(), chat_template)
    })
}

fn load_lfm2moe(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
) -> Result<LoadedModel, String> {
    let config = lfm2moe::config::Lfm2MoeConfig::from_hfq(&hfq)?;
    let weights = lfm2moe::lfm2moe::Lfm2MoeWeights::load(&mut hfq, &config, gpu)?;
    let state = lfm2moe::lfm2moe::Lfm2MoeState::new_with_max_seq(gpu, &config, max_seq)
        .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
    let eos_tok: u32 = {
        let try_one = |s: &str| -> Option<u32> {
            let ids = tokenizer.encode(s);
            if ids.len() == 1 { Some(ids[0]) } else { None }
        };
        try_one("<|im_end|>").or_else(|| try_one("</s>")).or_else(|| try_one("<|endoftext|>")).unwrap_or(1)
    };
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        lfm2moe_config: Some(config), lfm2moe_weights: Some(weights),
        lfm2moe_state: Some(state), lfm2moe_eos_tok: eos_tok,
        ..LoadedModel::skeleton(hfq.arch_id, tokenizer, max_seq, max_seq, path.to_string(), chat_template)
    })
}

fn load_minimax(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
    let weights = <minimax::MiniMaxM2 as Architecture>::load_weights(&mut hfq, &config, gpu)?;
    let state = minimax::MiniMaxState::new_with_max_seq(gpu, &config, max_seq)
        .map_err(|e| format!("minimax: MiniMaxState::new_with_max_seq failed: {e}"))?;
    let eos_tok: u32 = {
        let try_one = |s: &str| -> Option<u32> {
            let ids = tokenizer.encode(s);
            if ids.len() == 1 { Some(ids[0]) } else { None }
        };
        try_one("[e~[").or_else(|| try_one("<|im_end|>")).or_else(|| try_one("</s>")).or_else(|| try_one("<|endoftext|>")).unwrap_or(1)
    };
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        minimax_config: Some(config), minimax_weights: Some(weights),
        minimax_state: Some(state), minimax_eos_tok: eos_tok,
        ..LoadedModel::skeleton(hfq.arch_id, tokenizer, max_seq, max_seq, path.to_string(), chat_template)
    })
}

// ─── Pipeline-parallel load ───────────────────────────────────────────

fn load_model_pp(
    path: &str,
    max_seq: usize,
    kv_mode_override: Option<&str>,
    state_quant_override: Option<&str>,
    pp: usize,
    _gpu: &mut rdna_compute::Gpu,
) -> Result<LoadedModel, String> {
    let kv_mode = kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;

    if hfq.arch_id != 5 && hfq.arch_id != 6 {
        return Err(format!(
            "pp>1 supports Qwen3.5 dense (arch_id=5) and Qwen3.5-MoE / \
             Qwen3.6-A3B (arch_id=6) only; got arch_id={}. LLaMA / Qwen3 \
             dense (arch_id<5) is pp=1 only.",
            hfq.arch_id
        ));
    }
    // PP continues with the full load_model_pp body from daemon.rs...
    // For the initial scaffold, we rely on the daemon's copy. Full move coming.
    let config = qwen35::config_from_hfq(&hfq).ok_or("failed to read Qwen3.5 config")?;
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
            Gpus::init_layers(&counts).map_err(|e| format!("{e}"))?
        }
        None => Gpus::init_uniform(pp, config.n_layers).map_err(|e| format!("{e}"))?,
    };
    let weights = qwen35::load_weights_multi(&hfq, &config, &mut gpus)
        .map_err(|e| format!("{e}"))?;
    let is_kv_layer: Vec<bool> = config.layer_types.iter().map(|t| *t == LayerType::FullAttention).collect();
    let kv = match kv_mode.as_str() {
        "q8" => llama::KvCache::new_gpu_q8_capped_multi_filtered(
            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq),
        "asym3" | "turbo3" | "turbo" | "auto" | "" => llama::KvCache::new_gpu_asym3_capped_multi_filtered(
            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq),
        "fwht3" => llama::KvCache::new_gpu_fwht3_capped_multi_filtered(
            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq),
        "fwht2" => llama::KvCache::new_gpu_fwht2_capped_multi_filtered(
            &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq),
        _ => {
            eprintln!("  KV cache: unrecognized '{}', defaulting to asym3 for pp>1", kv_mode);
            llama::KvCache::new_gpu_asym3_capped_multi_filtered(
                &mut gpus, &is_kv_layer, config.n_kv_heads, config.head_dim, max_seq, max_seq)
        }
    }.map_err(|e| format!("{e}"))?;
    let dn_quant = parse_state_quant(state_quant_override)?;
    let (dn, la_to_device) = DeltaNetState::new_with_quant_multi(&mut gpus, &config, dn_quant)
        .map_err(|e| format!("{e}"))?;
    let scratch_set = Qwen35ScratchSet::new_with_kv_max_multi(
        &mut gpus, &config, 2048, max_seq).map_err(|e| format!("{e}"))?;
    // PP needs a single-GPU scratch for the bundle (the multi-GPU scratch_set is kept at top level)
    let gpu0 = &mut gpus.devices[0];
    let single_scratch = qwen35::Qwen35Scratch::new_with_kv_max(gpu0, &config, 2048, max_seq)
        .map_err(|e| format!("{e}"))?;
    let state = Some(ModelState::Qwen35(Qwen35Bundle { config, weights, scratch: single_scratch, kv_cache: kv, dn_state: dn }));
    let chat_template = resolve_chat_template(&hfq, path);
    let arch_id = hfq.arch_id;
    Ok(LoadedModel {
        state,
        ..LoadedModel::skeleton_pp(arch_id, tokenizer, max_seq, max_seq, path.to_string(), chat_template, pp, gpus, scratch_set, la_to_device)
    })
}

// ─── MMQ screening ────────────────────────────────────────────────────

fn screen_weights_qwen35(
    weights: &qwen35::Qwen35Weights,
    gpu: &mut rdna_compute::Gpu,
) -> (usize, usize) {
    use hipfire_arch_qwen35::qwen35::LayerWeights;
    let mut n_safe = 0usize;
    let mut n_unsafe = 0usize;

    for layer in &weights.layers {
        let wts: Vec<&hipfire_runtime::llama::WeightTensor> = match layer {
            LayerWeights::DeltaNet(l) => vec![&l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.w_gate, &l.w_up, &l.wo],
            LayerWeights::FullAttn(l) => vec![&l.wq, &l.wk, &l.wv, &l.w_gate, &l.w_up, &l.wo],
            LayerWeights::DeltaNetMoe(l) => vec![&l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.wo],
            LayerWeights::FullAttnMoe(l) => vec![&l.wq, &l.wk, &l.wv, &l.wo],
        };
        for wt in wts {
            if !matches!(
                wt.gpu_dtype,
                rdna_compute::DType::HFQ4G256 | rdna_compute::DType::MQ4G256
            ) {
                continue;
            }
            if gpu.mmq_screen_weight(&wt.buf, wt.m, wt.k) {
                n_safe += 1;
            } else {
                n_unsafe += 1;
            }
        }
    }
    (n_safe, n_unsafe)
}

// ─── DFlash state load ────────────────────────────────────────────────

fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &qwen35::Qwen35Config,
    target_dn: &DeltaNetState,
    gpu: &mut Gpu,
) -> Result<DflashState, String> {
    use hipfire_arch_qwen35::qwen35::LayerType;
    use hipfire_arch_qwen35::speculative::{DeltaNetSnapshot, DdtreeScratch, GdnTape, VerifyScratch, HiddenStateRingBuffer};
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("{e}"))?;
    let draft_config = hipfire_runtime::dflash::DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    let draft_weights = DflashWeights::load(gpu, &draft_hfq, &draft_config)
        .map_err(|e| format!("{e}"))?;
    let block_size = draft_config.block_size;
    let max_n = block_size + 1;
    let draft_scratch = DflashScratch::new(gpu, &draft_config, block_size, ctx_capacity)
        .map_err(|e| format!("{e}"))?;
    let _ = draft_hfq;
    let hidden_rb = HiddenStateRingBuffer::new(
        gpu,
        target_config.n_layers,
        draft_config.num_extract(),
        target_config.dim,
        ctx_capacity,
        max_n,
    ).map_err(|e| format!("HiddenStateRingBuffer::new: {e}"))?;
    let hidden_k = target_config.dim.next_power_of_two();
    let verify_scratch = VerifyScratch::with_prefill(
        gpu, max_n, target_config.dim, target_config.vocab_size, hidden_k, target_config,
    ).map_err(|e| format!("VerifyScratch::with_prefill: {e}"))?;
    let target_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
        .map_err(|e| format!("DeltaNetSnapshot::new_for: {e}"))?;
    let gdn_tape = GdnTape::new_for_config(gpu, target_config, max_n)
        .map_err(|e| format!("GdnTape::new_for_config: {e}"))?;
    let target_hidden_host = vec![0.0f32; ctx_capacity * target_config.dim];
    // DDTree
    let ddtree_budget: usize = std::env::var("HIPFIRE_DDTREE_BUDGET")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let ddtree = if ddtree_budget > 0 {
        let topk: usize = std::env::var("HIPFIRE_DDTREE_TOPK")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let qkv_dim = target_config.linear_num_key_heads * target_config.linear_key_head_dim * 2
            + target_config.linear_num_value_heads * target_config.linear_value_head_dim;
        let n_fa_layers = target_config.layer_types.iter()
            .filter(|t| **t == LayerType::FullAttention).count();
        let post_seed_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
            .map_err(|e| format!("{e}"))?;
        let scratch = DdtreeScratch::new(
            gpu,
            ddtree_budget,
            target_config.n_kv_heads,
            target_config.head_dim,
            qkv_dim,
            n_fa_layers,
        ).map_err(|e| format!("DdtreeScratch::new: {e}"))?;
        let path_c_parent_pre_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
            .map_err(|e| format!("{e}"))?;
        let path_c_main_end_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
            .map_err(|e| format!("{e}"))?;
        Some(DdtreeState {
            post_seed_snap,
            scratch,
            budget: ddtree_budget,
            topk,
            path_c_parent_pre_snap,
            path_c_main_end_snap,
        })
    } else {
        None
    };
    Ok(DflashState {
        draft_config,
        draft_weights,
        draft_scratch,
        hidden_rb,
        verify_scratch,
        target_snap,
        gdn_tape,
        target_hidden_host,
        ctx_capacity,
        block_size,
        ddtree,
    })
}

// ─── EP load functions ────────────────────────────────────────────────

pub fn load_model_ep(
    path: &str, max_seq: usize, tp: usize,
) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    match hfq.arch_id {
        9 => load_model_ep_ds4(path, max_seq, tp),
        10 => load_model_ep_minimax(path, max_seq, tp),
        id => Err(format!("EP not supported for arch_id={id} (expected 9 for DeepSeek V4 or 10 for MiniMax)")),
    }
}

fn load_model_ep_ds4(
    path: &str, max_seq: usize, tp: usize,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
    let arch_id = hfq.arch_id;
    let mut gpus = Gpus::init_uniform(tp, config.num_hidden_layers).map_err(|e| format!("Gpus: {e}"))?;
    let weights: Vec<deepseek4::DeepseekV4Weights> = (0..tp)
        .map(|rank| {
            let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
            <deepseek4::DeepseekV4 as Architecture>::load_weights(
                &mut hfq, &config, &mut gpus.devices[rank],
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let state: Vec<deepseek4::DeepseekV4State> = (0..tp)
        .map(|rank| {
            let _ = rank;
            deepseek4::DeepseekV4State::new(&config).unwrap()
        })
        .collect();
    let partials: Vec<rdna_compute::GpuTensor> = (0..tp)
        .map(|rank| {
            let g = &mut gpus.devices[rank];
            g.alloc_tensor(&[config.hidden_size], rdna_compute::DType::F32).unwrap()
        })
        .collect();
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        ep: Some(EpState { gpus, inner: EpArch::Ds4 { config, weights, state, partials } }),
        ..LoadedModel::skeleton(arch_id, tokenizer, max_seq, max_seq, path.to_string(), chat_template)
    })
}

fn load_model_ep_minimax(
    path: &str, max_seq: usize, tp: usize,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
    let arch_id = hfq.arch_id;
    let mut gpus = Gpus::init_uniform(tp, config.num_hidden_layers).map_err(|e| format!("Gpus: {e}"))?;
    let weights: Vec<minimax::MiniMaxWeights> = (0..tp)
        .map(|rank| {
            let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
            <minimax::MiniMaxM2 as Architecture>::load_weights(
                &mut hfq, &config, &mut gpus.devices[rank],
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let state: Vec<minimax::MiniMaxState> = (0..tp)
        .map(|rank| {
            minimax::MiniMaxState::new_with_max_seq(
                &mut gpus.devices[rank], &config, max_seq,
            ).unwrap()
        })
        .collect();
    let partials: Vec<rdna_compute::GpuTensor> = (0..tp)
        .map(|rank| {
            let g = &mut gpus.devices[rank];
            g.alloc_tensor(&[config.hidden_size], rdna_compute::DType::F32).unwrap()
        })
        .collect();
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        ep: Some(EpState { gpus, inner: EpArch::Minimax { config, weights, state, partials } }),
        ..LoadedModel::skeleton(arch_id, tokenizer, max_seq, max_seq, path.to_string(), chat_template)
    })
}

// ─── Unload ───────────────────────────────────────────────────────────

pub fn unload_model(mut m: LoadedModel, gpu: &mut rdna_compute::Gpu) {
    if m.pp > 1 {
        let mut gpus = m.pp_gpus.expect("pp>1 must carry pp_gpus");
        if let Some(scratch_set) = m.pp_scratch_set {
            scratch_set.free_gpu_multi(&mut gpus);
        }
        if let Some(ModelState::Qwen35(b)) = m.state.take() {
            b.kv_cache.free_gpu_multi(&mut gpus);
            let la_to_device = m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
            b.dn_state.free_gpu_multi(&mut gpus, &la_to_device);
            b.weights.free_gpu_multi(&mut gpus);
        }
        for g in gpus.devices.iter_mut() {
            g.invalidate_weight_caches();
            g.invalidate_graph_state();
            g.drain_pool();
        }
        let _ = gpu;
        return;
    }
    if let Some(df) = m.dflash {
        df.draft_weights.free_gpu(gpu);
        df.draft_scratch.free_gpu(gpu);
    }
    if let Some(ev) = m.eviction {
        ev.free_gpu(gpu);
    }
    if let Some(kv) = m.kv_cache {
        kv.free_gpu(gpu);
    }
    if let Some(dn) = m.dn_state {
        dn.free_gpu(gpu);
    }
    for (_, snap) in m.prefill_checkpoints {
        snap.free_gpu(gpu);
    }
    for (_, snap) in m.dflash_checkpoints {
        snap.free_gpu(gpu);
    }
    // Free arch-specific GPU state from the carrier bundle
    if let Some(state) = m.state {
        match state {
            ModelState::Qwen2(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::Qwen35(b) => {
                b.kv_cache.free_gpu(gpu);
                b.scratch.free_gpu(gpu);
                b.weights.free_gpu(gpu);
                b.dn_state.free_gpu(gpu);
            }
            ModelState::Llama(b) => {
                b.scratch.free_gpu(gpu);
                b.weights.free_gpu(gpu);
                b.kv.free_gpu(gpu);
            }
        }
    }
    // Non-core arch weights
    if let Some(s) = m.qwen2_state {
        s.free_gpu(gpu);
    }
    if let Some(s) = m.deepseek4_state {
        s.free_gpu(gpu);
    }
    if let Some(pbs) = m.deepseek4_pbs {
        pbs.free_gpu(gpu);
    }
    if let Some(w) = m.vision_weights {
        w.free_gpu(gpu);
    }
    if let Some(w) = m.deepseek4_weights {
        w.free_gpu(gpu);
    }
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
}
