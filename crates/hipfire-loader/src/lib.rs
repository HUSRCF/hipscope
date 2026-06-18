// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Top-of-DAG model loader. Owns `LoadedModel`, the carrier registry,
//! and `load_model` — the single arch-dispatch point for the daemon.

mod carriers;
pub use carriers::*;

use hipfire_arch_cohere2moe as cohere2moe;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_dots_ocr::dots_ocr;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_minimax as minimax;
use hipfire_arch_qwen2::qwen2;
use hipfire_arch_qwen35::qwen35;
use hipfire_arch_qwen35::qwen35::{DeltaNetState, Qwen35ScratchSet};
use hipfire_arch_qwen35::speculative::{
    DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
};
use hipfire_arch_qwen35::Qwen35Bundle;
use hipfire_arch_qwen35_vl::qwen35_vl;
use hipfire_runtime::cask::CaskCtx;
use hipfire_runtime::dflash::{DflashConfig, DflashScratch, DflashWeights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource};
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::triattn::{EvictionCtx, TriAttnCenters};
use rdna_compute::Gpu;
use std::path::Path;

// ─── Object-safe Carrier trait ──────────────────────────────────────

/// One arch's complete load contract. Object-safe → usable as `&dyn Carrier`.
pub trait Carrier: Send + Sync {
    fn name(&self) -> &'static str;
    /// Whether this carrier claims a given `arch_id`. `is_dir` distinguishes
    /// the two namespaces: HFQ-header ids (`HfqFile::arch_id`) vs the
    /// `derive_arch_id` ids emitted for safetensors directories. Kept as a
    /// pure `(u32, bool) -> bool` fn so the registry's disjointness can be
    /// unit-tested without constructing a real `ModelSource`.
    fn claims_arch_id(&self, arch_id: u32, is_dir: bool) -> bool;
    /// Default probe delegates to [`Carrier::claims_arch_id`]; carriers only
    /// implement the pure id predicate.
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src.arch_id(), Some(id) if self.claims_arch_id(id, src.is_dir()))
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String>;
}

// ─── Registry ─────────────────────────────────────────────────────────

const REGISTRY: &[&dyn Carrier] = &[
    &Qwen2Carrier,
    &Qwen35Carrier,
    &LlamaCarrier,
    &HfqCarrier {
        arch_id: 8,
        name: "dots_ocr",
        load: load_dots_ocr,
    },
    &HfqCarrier {
        arch_id: 9,
        name: "deepseek4",
        load: load_deepseek4,
    },
    &MinimaxCarrier,
    &HfqCarrier {
        arch_id: 11,
        name: "lfm2moe",
        load: load_lfm2moe,
    },
    &HfqCarrier {
        arch_id: 12,
        name: "cohere2moe",
        load: load_cohere2moe,
    },
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
///
/// `unload_model` matches this exhaustively with NO wildcard: adding a variant
/// without a teardown arm is a compile error, which is the whole point of
/// folding self-contained arch state in here rather than leaving it as loose
/// `Option<…>` fields that a reload can silently leak.
pub enum ModelState {
    Qwen2(hipfire_arch_qwen2::Qwen2Bundle),
    Qwen35(hipfire_arch_qwen35::Qwen35Bundle),
    Llama(hipfire_arch_llama::LlamaBundle),
    Lfm2Moe(Lfm2MoeBundle),
    Minimax(MiniMaxBundle),
    Cohere2Moe(Cohere2MoeBundle),
}

/// LFM2.5-MoE (arch_id=11) GPU bundle. `eos_tok` is resolved at load time and
/// rides along so the generate path doesn't re-tokenize.
pub struct Lfm2MoeBundle {
    pub config: lfm2moe::config::Lfm2MoeConfig,
    pub weights: lfm2moe::lfm2moe::Lfm2MoeWeights,
    pub state: lfm2moe::lfm2moe::Lfm2MoeState,
    pub eos_tok: u32,
}

/// MiniMax-M2 (arch_id=10) GPU bundle.
pub struct MiniMaxBundle {
    pub config: minimax::MiniMaxConfig,
    pub weights: minimax::MiniMaxWeights,
    pub state: minimax::MiniMaxState,
    pub eos_tok: u32,
}

/// Cohere2-MoE / North-Mini-Code (arch_id=12) GPU bundle.
pub struct Cohere2MoeBundle {
    pub config: cohere2moe::Cohere2MoeConfig,
    pub weights: cohere2moe::Cohere2MoeWeights,
    pub state: cohere2moe::Cohere2MoeState,
    pub eos_tok: u32,
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
    // LFM2.5-8B-A1B (arch_id=11) and MiniMax-M2 (arch_id=10) live in
    // `state` as ModelState::{Lfm2Moe,Minimax} so unload teardown is
    // compiler-enforced (see ModelState).
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
            arch_id,
            pp: 1,
            ep: None,
            pp_gpus: None,
            pp_scratch_set: None,
            pp_dn_la_to_device: None,
            state: None,
            kv_cache: None,
            dn_state: None,
            qwen2_state: None,
            deepseek4_config: None,
            deepseek4_weights: None,
            deepseek4_state: None,
            deepseek4_pbs: None,
            deepseek4_eos_tok: 0,
            mtp_mode: "auto".to_string(),
            mtp_k: 3,
            mtp_weights_present: false,
            dots_ocr_config: None,
            dots_ocr_weights: None,
            vision_config: None,
            vision_weights: None,
            tokenizer: Some(tokenizer),
            seq_pos: 0,
            max_seq,
            physical_cap,
            eviction: None,
            kv_adaptive: None,
            conversation_tokens: Vec::new(),
            asst_turn_cache: AsstTurnCache::new_from_env(),
            prefill_checkpoints: Vec::new(),
            dflash_checkpoints: Vec::new(),
            decoded_vocab: None,
            model_path,
            dflash: None,
            chat_template,
        }
    }

    /// LFM2.5-MoE bundle if this model is arch_id=11, else None.
    pub fn lfm2moe(&self) -> Option<&Lfm2MoeBundle> {
        match &self.state {
            Some(ModelState::Lfm2Moe(b)) => Some(b),
            _ => None,
        }
    }

    pub fn lfm2moe_mut(&mut self) -> Option<&mut Lfm2MoeBundle> {
        match &mut self.state {
            Some(ModelState::Lfm2Moe(b)) => Some(b),
            _ => None,
        }
    }

    /// MiniMax-M2 bundle if this model is arch_id=10, else None.
    pub fn minimax(&self) -> Option<&MiniMaxBundle> {
        match &self.state {
            Some(ModelState::Minimax(b)) => Some(b),
            _ => None,
        }
    }

    pub fn minimax_mut(&mut self) -> Option<&mut MiniMaxBundle> {
        match &mut self.state {
            Some(ModelState::Minimax(b)) => Some(b),
            _ => None,
        }
    }

    /// Cohere2-MoE bundle if this model is arch_id=12, else None.
    pub fn cohere2moe(&self) -> Option<&Cohere2MoeBundle> {
        match &self.state {
            Some(ModelState::Cohere2Moe(b)) => Some(b),
            _ => None,
        }
    }

    pub fn cohere2moe_mut(&mut self) -> Option<&mut Cohere2MoeBundle> {
        match &mut self.state {
            Some(ModelState::Cohere2Moe(b)) => Some(b),
            _ => None,
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
            ..LoadedModel::skeleton(
                arch_id,
                tokenizer,
                max_seq,
                physical_cap,
                model_path,
                chat_template,
            )
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
        12 => {
            if let Some(t) = hfq.chat_template_named("tool_use") {
                return Some(
                    t.replace("<|START_RESPONSE|>", "<|START_TEXT|>")
                        .replace("<|END_RESPONSE|>", "<|END_TEXT|>")
                        .replace("{{message.tool_plan}}", "{{ message.tool_plan or '' }}")
                        .replace("{{ tc['function']['name'] }}", "{{ tc.name }}")
                        .replace(
                            "{{ tc['function']['arguments']|tojson }}",
                            "{{ tc.arguments|tojson }}",
                        ),
                );
            }
        }
        _ => {}
    }
    hfq.chat_template()
}

pub(crate) fn parse_state_quant(
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
            .layer_types
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if *t == LayerType::FullAttention {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if fa_layer_ids.is_empty() {
            eprintln!("  cask_sidecar set but model has no FullAttention layers — ignoring");
            None
        } else {
            let n_rot = (config.head_dim as f32 * config.partial_rotary_factor) as usize;
            let base = EvictionCtx::new(
                ctx.gpu,
                &centers,
                fa_layer_ids,
                ctx.cask.budget,
                ctx.cask.beta,
                config.n_heads,
                config.n_kv_heads,
                config.head_dim,
                n_rot,
                config.rope_theta,
                physical_cap,
            )
            .map_err(|e| format!("build EvictionCtx: {e}"))?;
            if ctx.cask.cask_m_folding {
                eprintln!(
                    "  eviction: CASK α={:.2} m={} budget={} β={} physical_cap={}",
                    ctx.cask.core_frac,
                    ctx.cask.fold_m,
                    ctx.cask.budget,
                    ctx.cask.beta,
                    physical_cap
                );
                Some(Eviction::Cask(CaskCtx::new(
                    base,
                    ctx.cask.core_frac,
                    ctx.cask.fold_m,
                )))
            } else {
                eprintln!(
                    "  eviction: TriAttention (plain drop) budget={} β={} physical_cap={}",
                    ctx.cask.budget, ctx.cask.beta, physical_cap
                );
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
                eprintln!(
                    "  DFlash draft loaded: {} (layers={}, hidden={}, block={})",
                    dp, s.draft_config.n_layers, s.draft_config.hidden, s.draft_config.block_size
                );
                Some(s)
            }
            Err(e) => {
                eprintln!(
                    "  DFlash draft load failed ({}): {} — falling back to AR only",
                    dp, e
                );
                None
            }
        }
    } else {
        None
    };

    let state = Some(ModelState::Qwen35(bundle));
    Ok(LoadedModel {
        state,
        eviction,
        dflash,
        vision_config,
        vision_weights,
        max_seq: ctx.max_seq,
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            ctx.max_seq,
            physical_cap,
            ctx.path.to_string(),
            chat_template,
        )
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
                    format!(
                        "arch_id={} (MoE/A3B-class) has no MQ3 MoE kernels",
                        hfq.arch_id
                    )
                } else {
                    format!(
                        "arch={} lacks the corresponding batched WMMA prefill family",
                        gpu.arch
                    )
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
        path,
        max_seq,
        draft_path,
        kv_mode_override,
        kv_adaptive_override,
        state_quant_override,
        cask,
        pp,
        gpu,
    };

    // Carrier registry dispatch. Collect all matches so an overlap between
    // two carriers' `claims_arch_id` fails loudly here instead of silently
    // resolving to whichever was registered first.
    let mut matches = REGISTRY.iter().filter(|c| c.probe(&src));
    let carrier = matches
        .next()
        .ok_or_else(|| format!("no carrier for {}", src.describe()))?;
    if let Some(other) = matches.next() {
        return Err(format!(
            "ambiguous carrier dispatch for {}: '{}' and '{}' both claim it",
            src.describe(),
            carrier.name(),
            other.name()
        ));
    }
    let result = carrier.load(src, &mut ctx)?;
    debug_assert!(
        !(result.pp > 1) || result.pp_gpus.is_some(),
        "pp>1 LoadedModel missing pp_gpus"
    );
    Ok(result)
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
        dots_ocr_config: Some(config),
        dots_ocr_weights: Some(weights),
        ..LoadedModel::skeleton(
            hfq.arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
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
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let pbs = deepseek4::forward::PrefillBatchScratch::new(gpu, &config, pbs_max_batch)?;
    let eos_tok: u32 = {
        let ids = tokenizer.encode("<｜end▁of▁sentence｜>");
        if ids.len() == 1 {
            ids[0]
        } else {
            1
        }
    };
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        deepseek4_config: Some(config),
        deepseek4_weights: Some(weights),
        deepseek4_state: Some(state),
        deepseek4_pbs: Some(pbs),
        deepseek4_eos_tok: eos_tok,
        ..LoadedModel::skeleton(
            hfq.arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
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
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        };
        try_one("<|im_end|>")
            .or_else(|| try_one("</s>"))
            .or_else(|| try_one("<|endoftext|>"))
            .unwrap_or(1)
    };
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        state: Some(ModelState::Lfm2Moe(Lfm2MoeBundle {
            config,
            weights,
            state,
            eos_tok,
        })),
        ..LoadedModel::skeleton(
            hfq.arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
    })
}

fn load_cohere2moe(
    mut hfq: HfqFile,
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    gpu: &mut Gpu,
    max_seq: usize,
    path: &str,
) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let config = <cohere2moe::Cohere2Moe as Architecture>::config_from_hfq(&hfq)?;
    let weights = <cohere2moe::Cohere2Moe as Architecture>::load_weights(&mut hfq, &config, gpu)?;
    let state = cohere2moe::Cohere2MoeState::new_with_max_seq(gpu, &config, max_seq)
        .map_err(|e| format!("cohere2moe: new_with_max_seq failed: {e}"))?;
    let eos_tok: u32 = {
        let try_one = |s: &str| -> Option<u32> {
            let ids = tokenizer.encode(s);
            if ids.len() == 1 {
                Some(ids[0])
            } else {
                None
            }
        };
        try_one("<|END_OF_TURN_TOKEN|>")
            .or_else(|| try_one("</s>"))
            .or_else(|| try_one("<|endoftext|>"))
            .unwrap_or(255001)
    };
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        state: Some(ModelState::Cohere2Moe(Cohere2MoeBundle {
            config,
            weights,
            state,
            eos_tok,
        })),
        ..LoadedModel::skeleton(
            hfq.arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
    })
}

// ─── MMQ screening ────────────────────────────────────────────────────

// ─── DFlash state load ────────────────────────────────────────────────

fn load_dflash_state(
    draft_path: &str,
    ctx_capacity: usize,
    target_config: &qwen35::Qwen35Config,
    target_dn: &DeltaNetState,
    gpu: &mut Gpu,
) -> Result<DflashState, String> {
    use hipfire_arch_qwen35::qwen35::LayerType;
    use hipfire_arch_qwen35::speculative::{
        DdtreeScratch, DeltaNetSnapshot, GdnTape, HiddenStateRingBuffer, VerifyScratch,
    };
    let draft_hfq = HfqFile::open(Path::new(draft_path)).map_err(|e| format!("{e}"))?;
    let draft_config = hipfire_runtime::dflash::DflashConfig::from_hfq(&draft_hfq)
        .ok_or_else(|| "draft: failed to parse DflashConfig from HFQ metadata".to_string())?;
    let draft_weights =
        DflashWeights::load(gpu, &draft_hfq, &draft_config).map_err(|e| format!("{e}"))?;
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
    )
    .map_err(|e| format!("HiddenStateRingBuffer::new: {e}"))?;
    let hidden_k = target_config.dim.next_power_of_two();
    let verify_scratch = VerifyScratch::with_prefill(
        gpu,
        max_n,
        target_config.dim,
        target_config.vocab_size,
        hidden_k,
        target_config,
    )
    .map_err(|e| format!("VerifyScratch::with_prefill: {e}"))?;
    let target_snap = DeltaNetSnapshot::new_for(gpu, target_dn)
        .map_err(|e| format!("DeltaNetSnapshot::new_for: {e}"))?;
    let gdn_tape = GdnTape::new_for_config(gpu, target_config, max_n)
        .map_err(|e| format!("GdnTape::new_for_config: {e}"))?;
    let target_hidden_host = vec![0.0f32; ctx_capacity * target_config.dim];
    // DDTree
    let ddtree_budget: usize = std::env::var("HIPFIRE_DDTREE_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let ddtree = if ddtree_budget > 0 {
        let topk: usize = std::env::var("HIPFIRE_DDTREE_TOPK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let qkv_dim = target_config.linear_num_key_heads * target_config.linear_key_head_dim * 2
            + target_config.linear_num_value_heads * target_config.linear_value_head_dim;
        let n_fa_layers = target_config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        let post_seed_snap =
            DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("{e}"))?;
        let scratch = DdtreeScratch::new(
            gpu,
            ddtree_budget,
            target_config.n_kv_heads,
            target_config.head_dim,
            qkv_dim,
            n_fa_layers,
        )
        .map_err(|e| format!("DdtreeScratch::new: {e}"))?;
        let path_c_parent_pre_snap =
            DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("{e}"))?;
        let path_c_main_end_snap =
            DeltaNetSnapshot::new_for(gpu, target_dn).map_err(|e| format!("{e}"))?;
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

pub fn load_model_ep(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    match hfq.arch_id {
        9 => load_model_ep_ds4(path, max_seq, tp),
        10 => load_model_ep_minimax(path, max_seq, tp),
        id => Err(format!(
            "EP not supported for arch_id={id} (expected 9 for DeepSeek V4 or 10 for MiniMax)"
        )),
    }
}

fn load_model_ep_ds4(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <deepseek4::DeepseekV4 as Architecture>::config_from_hfq(&hfq)?;
    let arch_id = hfq.arch_id;
    let mut gpus =
        Gpus::init_uniform(tp, config.num_hidden_layers).map_err(|e| format!("Gpus: {e}"))?;
    let weights: Vec<deepseek4::DeepseekV4Weights> = (0..tp)
        .map(|rank| {
            let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
            <deepseek4::DeepseekV4 as Architecture>::load_weights(
                &mut hfq,
                &config,
                &mut gpus.devices[rank],
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
            g.alloc_tensor(&[config.hidden_size], rdna_compute::DType::F32)
                .unwrap()
        })
        .collect();
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        ep: Some(EpState {
            gpus,
            inner: EpArch::Ds4 {
                config,
                weights,
                state,
                partials,
            },
        }),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
    })
}

fn load_model_ep_minimax(path: &str, max_seq: usize, tp: usize) -> Result<LoadedModel, String> {
    use hipfire_runtime::arch::Architecture;
    let hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|e| format!("tokenizer not found: {e}"))?;
    let config = <minimax::MiniMaxM2 as Architecture>::config_from_hfq(&hfq)?;
    let arch_id = hfq.arch_id;
    let mut gpus =
        Gpus::init_uniform(tp, config.num_hidden_layers).map_err(|e| format!("Gpus: {e}"))?;
    let weights: Vec<minimax::MiniMaxWeights> = (0..tp)
        .map(|rank| {
            let mut hfq = HfqFile::open(Path::new(path)).map_err(|e| format!("{e}"))?;
            <minimax::MiniMaxM2 as Architecture>::load_weights(
                &mut hfq,
                &config,
                &mut gpus.devices[rank],
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let state: Vec<minimax::MiniMaxState> = (0..tp)
        .map(|rank| {
            minimax::MiniMaxState::new_with_max_seq(&mut gpus.devices[rank], &config, max_seq)
                .unwrap()
        })
        .collect();
    let partials: Vec<rdna_compute::GpuTensor> = (0..tp)
        .map(|rank| {
            let g = &mut gpus.devices[rank];
            g.alloc_tensor(&[config.hidden_size], rdna_compute::DType::F32)
                .unwrap()
        })
        .collect();
    let chat_template = resolve_chat_template(&hfq, path);
    Ok(LoadedModel {
        ep: Some(EpState {
            gpus,
            inner: EpArch::Minimax {
                config,
                weights,
                state,
                partials,
            },
        }),
        ..LoadedModel::skeleton(
            arch_id,
            tokenizer,
            max_seq,
            max_seq,
            path.to_string(),
            chat_template,
        )
    })
}

// ─── Unload ───────────────────────────────────────────────────────────

pub fn unload_model(mut m: LoadedModel, gpu: &mut rdna_compute::Gpu) {
    if m.pp > 1 {
        let mut gpus = m.pp_gpus.expect("pp>1 must carry pp_gpus");
        if let Some(scratch_set) = m.pp_scratch_set {
            scratch_set.free_gpu_multi(&mut gpus);
        }
        match m.state.take() {
            Some(ModelState::Qwen35(b)) => {
                b.kv_cache.free_gpu_multi(&mut gpus);
                let la_to_device = m.pp_dn_la_to_device.expect("pp>1 must carry la_to_device");
                b.dn_state.free_gpu_multi(&mut gpus, &la_to_device);
                b.weights.free_gpu_multi(&mut gpus);
            }
            // Only Qwen35 supports pp>1 today, so the other carriers can never
            // reach this arm with multi-GPU state to free — dropping is correct.
            // Listing them explicitly (rather than `_`) makes that a
            // compiler-enforced invariant: adding a pp>1-capable carrier without
            // a teardown arm here is a build error, not a silent VRAM leak.
            Some(ModelState::Qwen2(_))
            | Some(ModelState::Llama(_))
            | Some(ModelState::Lfm2Moe(_))
            | Some(ModelState::Minimax(_))
            | Some(ModelState::Cohere2Moe(_))
            | None => {}
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
            ModelState::Lfm2Moe(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::Minimax(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
            }
            ModelState::Cohere2Moe(b) => {
                b.state.free_gpu(gpu);
                b.weights.free_gpu(gpu);
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
    // lfm2moe / minimax teardown is now compiler-enforced via the exhaustive
    // ModelState match above. dots_ocr already had a free_gpu, it just wasn't
    // called here (still a loose Option — fold in a future pass).
    if let Some(w) = m.dots_ocr_weights {
        w.free_gpu(gpu);
    }
    gpu.invalidate_weight_caches();
    gpu.invalidate_graph_state();
    gpu.drain_pool();
}

#[cfg(test)]
mod registry_tests {
    use super::REGISTRY;

    /// Every known arch_id must be claimed by AT MOST one carrier, for both
    /// source namespaces (HFQ header ids and `derive_arch_id` dir ids). This
    /// guards the otherwise-silent first-match overlap in `load_model`: add a
    /// carrier whose `claims_arch_id` collides with an existing one and this
    /// fails in CI instead of mis-routing weights at runtime.
    #[test]
    fn carriers_are_disjoint() {
        // Sweep well past the assigned range, plus the reserved sentinels
        // (20 = DFlash draft, 0xFF = toy/template — neither should dispatch).
        let ids = (0u32..=64).chain([20, 0xFF]);
        for id in ids {
            for is_dir in [false, true] {
                let claimers: Vec<&str> = REGISTRY
                    .iter()
                    .filter(|c| c.claims_arch_id(id, is_dir))
                    .map(|c| c.name())
                    .collect();
                assert!(
                    claimers.len() <= 1,
                    "arch_id={id} is_dir={is_dir} claimed by multiple carriers: {claimers:?}"
                );
            }
        }
    }

    /// Pin the intended routing so a future probe edit can't silently move an
    /// existing model to the wrong carrier. `is_dir` matters: Qwen2 is HFQ id
    /// 7 but its dir form derives to id 1 (→ llama path).
    #[test]
    fn known_ids_route_as_expected() {
        let cases: &[(u32, bool, &str)] = &[
            (7, false, "qwen2"),
            (5, false, "qwen35"),
            (6, false, "qwen35"),
            (5, true, "qwen35"),
            (6, true, "qwen35"),
            (0, false, "llama"),
            (1, false, "llama"),
            (0, true, "llama"),
            (1, true, "llama"),
            (8, false, "dots_ocr"),
            (9, false, "deepseek4"),
            (10, false, "minimax"),
            (11, false, "lfm2moe"),
            (12, false, "cohere2moe"),
        ];
        for &(id, is_dir, want) in cases {
            let got: Vec<&str> = REGISTRY
                .iter()
                .filter(|c| c.claims_arch_id(id, is_dir))
                .map(|c| c.name())
                .collect();
            assert_eq!(
                got,
                vec![want],
                "arch_id={id} is_dir={is_dir} should route to exactly [{want}]"
            );
        }
    }

    /// The unassigned HFQ ids 2..=4 must reach NO carrier — this is the
    /// regression guard for fix B (the old `arch_id < 5` open range silently
    /// loaded them as llama).
    #[test]
    fn unassigned_low_ids_match_nothing() {
        for id in [2u32, 3, 4] {
            let n = REGISTRY
                .iter()
                .filter(|c| c.claims_arch_id(id, false))
                .count();
            assert_eq!(n, 0, "arch_id={id} (unassigned) should match no carrier");
        }
    }
}
