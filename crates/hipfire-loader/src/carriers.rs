// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Per-arch carrier structs with object-safe [`Carrier`] impls.
//! Each carrier owns its full load path (HFQ + safetensors-dir).

use crate::Carrier;
use crate::{finish_qwen35_load, resolve_chat_template, LoadedModel, ModelState};
use hipfire_arch_minimax::{config_from_safetensors, load_weights_from_safetensors, MiniMaxState};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as _;

// ─── Source-only metadata (tokenizer / chat_template / arch_id) ───────
//
// The single seam for the source-varying-but-arch-invariant axis. Adding a
// future source kind (e.g. GGUF) is one new `match` arm here plus the
// irreducible per-arch `(config, weights)` block in each carrier. Lives in
// `hipfire-loader` (not `loader_api`) because it calls `resolve_chat_template`,
// which reads the loader's built-in arch templates.
//
// NOTE: `arch_id` extraction is purely source-varying (`hfq.arch_id` vs
// `source.arch_id()`), so it belongs here — but the *values* live in two
// distinct namespaces (HFQ header ids vs `derive_arch_id` dir ids). A GGUF
// plug-in author must pick the correct namespace, not assume a single one.
struct SourceMeta {
    tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    chat_template: Option<String>,
    arch_id: u32,
}

fn resolve_source_meta(src: &ModelSource, path: &str) -> Result<SourceMeta, String> {
    match src {
        ModelSource::Hfq(hfq) => Ok(SourceMeta {
            tokenizer: hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(
                &hfq.metadata_json,
            )
            .map_err(|e| format!("tokenizer not found: {e}"))?,
            chat_template: resolve_chat_template(hfq, path),
            arch_id: hfq.arch_id,
        }),
        ModelSource::Dir(source) => Ok(SourceMeta {
            tokenizer: tokenizer_from_dir(source)?,
            chat_template: source.chat_template(),
            arch_id: source.arch_id(),
        }),
    }
}

/// Folds the "no tokenizer.json / failed to parse" block duplicated verbatim
/// in every Dir arm today.
fn tokenizer_from_dir(
    source: &hipfire_runtime::safetensors_source::SafetensorsSource,
) -> Result<hipfire_runtime::tokenizer::Tokenizer, String> {
    if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
            .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))
    } else {
        Err("no tokenizer.json found in model directory".into())
    }
}

// ─── Qwen2Carrier ────────────────────────────────────────────────────

pub struct Qwen2Carrier;
impl Carrier for Qwen2Carrier {
    fn name(&self) -> &'static str {
        "qwen2"
    }
    fn claims_arch_id(&self, arch_id: u32, is_dir: bool) -> bool {
        // HFQ id 7 only; qwen2 dirs derive to id 1 → handled by LlamaCarrier.
        !is_dir && arch_id == 7
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err("qwen2: pipeline-parallel (pp>1) unsupported".into());
        }
        let ModelSource::Hfq(_) = &src else {
            return Err("qwen2: directory source unsupported".into());
        };
        let meta = resolve_source_meta(&src, ctx.path)?;
        let bundle = hipfire_arch_qwen2::load_qwen2_bundle(src, ctx)?;
        Ok(LoadedModel {
            state: Some(ModelState::Qwen2(bundle)),
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
            )
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

/// Qwen3.5 pipeline-parallel (pp>1) load. Extracted from the carrier body so
/// the pp>1 multi-GPU tail (`skeleton_pp`) lives in one place; qwen35 is the
/// only carrier with a pp>1 path. KV policy (`QWEN35_PP_POLICY`), DeltaNet
/// quant, and scratch sizing are byte-identical to the previous inline block.
fn load_qwen35_pp(
    mut hfq_file: hipfire_runtime::hfq::HfqFile,
    meta: SourceMeta,
    ctx: &mut LoadCtx,
) -> Result<LoadedModel, String> {
    let pp = ctx.pp;
    let config = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq_file)
        .map_err(|e| format!("failed to read Qwen3.5 config: {e}"))?;
    let kv_mode = ctx
        .kv_mode_override
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
    let mut gpus = match std::env::var("HIPFIRE_PP_LAYERS")
        .ok()
        .filter(|s| !s.is_empty())
    {
        Some(spec) => {
            let counts: Result<Vec<usize>, _> =
                spec.split(',').map(|s| s.trim().parse::<usize>()).collect();
            let counts = counts.map_err(|e| format!("HIPFIRE_PP_LAYERS parse: {e}"))?;
            if counts.len() != pp {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS has {} entries, expected pp={}",
                    counts.len(),
                    pp
                ));
            }
            let sum: usize = counts.iter().sum();
            if sum != config.n_layers {
                return Err(format!(
                    "HIPFIRE_PP_LAYERS sum={} != n_layers={}",
                    sum, config.n_layers
                ));
            }
            hipfire_runtime::multi_gpu::Gpus::init_layers(&counts).map_err(|e| format!("{e}"))?
        }
        None => hipfire_runtime::multi_gpu::Gpus::init_uniform(pp, config.n_layers)
            .map_err(|e| format!("{e}"))?,
    };
    let layout = hipfire_arch_qwen35::qwen35::Layout::from_gpus(&gpus, config.n_layers);
    let mut hfq_source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq_file, &config);
    let weights = hipfire_arch_qwen35::qwen35::load_weights(
        &mut hfq_source,
        &mut gpus.devices,
        &layout,
    )
    .map_err(|e| format!("{e}"))?;
    let is_kv_layer: Vec<bool> = config
        .layer_types
        .iter()
        .map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention)
        .collect();
    let hipfire_runtime::kv_mode::ResolveResult { mode, warning } =
        hipfire_runtime::kv_mode::resolve(
            &kv_mode,
            &hipfire_runtime::kv_mode::QWEN35_PP_POLICY,
            config.head_dim,
        );
    if let Some(w) = warning {
        eprintln!(
            "  KV cache: {w} (site {})",
            hipfire_runtime::kv_mode::QWEN35_PP_POLICY.site
        );
    }
    let dims = hipfire_runtime::llama::KvDims {
        layers: hipfire_runtime::llama::KvLayers::Mask(is_kv_layer),
        n_kv_heads: config.n_kv_heads,
        head_dim: config.head_dim,
        max_seq: ctx.max_seq,
        physical_cap: Some(ctx.max_seq),
    };
    let kv = hipfire_runtime::llama::KvCache::from_mode(
        mode,
        hipfire_runtime::llama::KvTarget::Multi(&mut gpus),
        &dims,
    )
    .map_err(|e| format!("{e}"))?;
    let dn_quant = crate::parse_state_quant(ctx.state_quant_override).map_err(|e| format!("{e}"))?;
    let (dn, la_to_device) = hipfire_arch_qwen35::qwen35::DeltaNetState::new_with_quant_multi(
        &mut gpus, &config, dn_quant,
    )
    .map_err(|e| format!("{e}"))?;
    let scratch_set = hipfire_arch_qwen35::qwen35::Qwen35ScratchSet::new_with_kv_max_multi(
        &mut gpus,
        &config,
        2048,
        ctx.max_seq,
    )
    .map_err(|e| format!("{e}"))?;
    let gpu0 = &mut gpus.devices[0];
    let single_scratch = hipfire_arch_qwen35::qwen35::Qwen35Scratch::new_with_kv_max(
        gpu0,
        &config,
        2048,
        ctx.max_seq,
    )
    .map_err(|e| format!("{e}"))?;
    let bundle = hipfire_arch_qwen35::Qwen35Bundle {
        config,
        weights,
        scratch: single_scratch,
        kv_cache: kv,
        dn_state: dn,
    };
    Ok(LoadedModel {
        state: Some(ModelState::Qwen35(bundle)),
        ..LoadedModel::skeleton_pp(
            meta.arch_id,
            meta.tokenizer,
            ctx.max_seq,
            ctx.max_seq,
            ctx.path.to_string(),
            meta.chat_template,
            pp,
            gpus,
            scratch_set,
            la_to_device,
        )
    })
}

pub struct Qwen35Carrier;
impl Carrier for Qwen35Carrier {
    fn name(&self) -> &'static str {
        "qwen35"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 5 = dense (+VL), 6 = MoE — same ids in both namespaces.
        matches!(arch_id, 5 | 6)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        // Dir + pp>1: early return before any diagnostics/meta resolution,
        // preserving the original error string and preventing tokenizer work.
        if ctx.pp > 1 {
            if let ModelSource::Dir(..) = &src {
                return Err("qwen35: safetensors + pp>1 unsupported".into());
            }
        }
        // Per-source diagnostics stay at the call site, before resolve_source_meta.
        if let ModelSource::Dir(s) = &src {
            let qm = s
                .quant_config()
                .map(|q| q.method.as_str())
                .unwrap_or("none");
            eprintln!("  safetensors arch_id={}, quant_method={qm}", s.arch_id());
        }
        let meta = resolve_source_meta(&src, ctx.path)?;

        match src {
            ModelSource::Hfq(mut hfq_file) => {
                // ── pp>1 path (pipeline-parallel) — extracted helper ──
                if ctx.pp > 1 {
                    return load_qwen35_pp(hfq_file, meta, ctx);
                }

                // ── pp=1 path (single-GPU) ────────────────────
                let physical_cap = if ctx.cask.sidecar.is_some() {
                    let env_override = std::env::var("HIPFIRE_KV_PHYSICAL_CAP")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok());
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
                    let has_vision = hfq_file
                        .tensor_data("model.visual.patch_embed.proj.weight")
                        .is_some();
                    let vc = Qwen35Vl::config_from_hfq(&hfq_file).ok();
                    match vc {
                        Some(vc) if has_vision => {
                            let vw = Qwen35Vl::load_weights(&mut hfq_file, &vc, ctx.gpu)
                                .map_err(|e| eprintln!("  VL weight load failed: {e}"))
                                .ok();
                            eprintln!(
                                "  VL model: vision encoder (hidden={}, layers={})",
                                vc.hidden_size, vc.num_layers
                            );
                            (Some(vc), vw)
                        }
                        _ => (None, None),
                    }
                };

                let bundle =
                    hipfire_arch_qwen35::load_qwen35_bundle(ModelSource::Hfq(hfq_file), ctx)?;
                finish_qwen35_load(
                    bundle,
                    meta.tokenizer,
                    physical_cap,
                    meta.arch_id,
                    meta.chat_template,
                    ctx,
                    vision_config,
                    vision_weights,
                )
            }
            ModelSource::Dir(source) => {
                if ctx.pp > 1 {
                    return Err("qwen35: safetensors + pp>1 unsupported".into());
                }
                let config = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source)
                    .map_err(|e| {
                        format!("failed to parse Qwen3.5 config from config.json: {e}")
                    })?;
                let mut paro_source =
                    hipfire_arch_qwen35::qwen35::ParoSource::new(&source, &config)
                        .map_err(|e| format!("ParoSource::new: {e:?}"))?;
                let paro_layout = hipfire_arch_qwen35::qwen35::Layout::single(config.n_layers);
                let weights = hipfire_arch_qwen35::qwen35::load_weights(
                    &mut paro_source,
                    std::slice::from_mut(ctx.gpu),
                    &paro_layout,
                )
                .map_err(|e| format!("load_weights: {e:?}"))?;
                let is_kv_layer: Vec<bool> = config
                    .layer_types
                    .iter()
                    .map(|t| *t == hipfire_arch_qwen35::qwen35::LayerType::FullAttention)
                    .collect();
                let kv_mode = qwen35_kv_mode(ctx);
                let hipfire_runtime::kv_mode::ResolveResult { mode, warning } =
                    hipfire_runtime::kv_mode::resolve(
                        &kv_mode,
                        &hipfire_runtime::kv_mode::QWEN35_PARO_POLICY,
                        config.head_dim,
                    );
                if let Some(w) = warning {
                    eprintln!(
                        "  KV cache: {w} (site {})",
                        hipfire_runtime::kv_mode::QWEN35_PARO_POLICY.site
                    );
                }
                let dims = hipfire_runtime::llama::KvDims {
                    layers: hipfire_runtime::llama::KvLayers::Mask(is_kv_layer),
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    max_seq: ctx.max_seq,
                    physical_cap: Some(ctx.max_seq),
                };
                let kv_cache = hipfire_runtime::llama::KvCache::from_mode(
                    mode,
                    hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                    &dims,
                )
                .map_err(|e| format!("KvCache: {e}"))?;

                let dn_state =
                    hipfire_arch_qwen35::qwen35::DeltaNetState::new(ctx.gpu, &config)
                        .map_err(|e| format!("DeltaNetState::new: {e:?}"))?;
                let scratch =
                    hipfire_arch_qwen35::qwen35::Qwen35Scratch::new(ctx.gpu, &config, 256)
                        .map_err(|e| format!("Qwen35Scratch::new: {e:?}"))?;

                let bundle = hipfire_arch_qwen35::Qwen35Bundle {
                    config,
                    weights,
                    scratch,
                    kv_cache,
                    dn_state,
                };
                Ok(LoadedModel {
                    state: Some(ModelState::Qwen35(bundle)),
                    ..LoadedModel::skeleton(
                        meta.arch_id,
                        meta.tokenizer,
                        ctx.max_seq,
                        ctx.max_seq,
                        ctx.path.to_string(),
                        meta.chat_template,
                    )
                })
            }
        }
    }
}

// ─── LlamaCarrier ────────────────────────────────────────────────────

pub struct LlamaCarrier;
impl Carrier for LlamaCarrier {
    fn name(&self) -> &'static str {
        "llama"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        // 0 = LLaMA/Mistral, 1 = plain Qwen3/Qwen2 (both namespaces).
        // Explicit allowlist (was an open `< 5` range that would silently
        // swallow any future HFQ id in 2..=4 into the llama path).
        matches!(arch_id, 0 | 1)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(match &src {
                ModelSource::Hfq(_) => "llama: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "llama: safetensors + pp>1 unsupported",
            }
            .into());
        }
        if let ModelSource::Dir(s) = &src {
            eprintln!("  safetensors arch_id={}", s.arch_id());
        }
        let meta = resolve_source_meta(&src, ctx.path)?;

        // ── source-varying seam: yields a LlamaBundle ──
        let bundle = match src {
            ModelSource::Hfq(hfq) => {
                hipfire_arch_llama::load_llama_bundle(ModelSource::Hfq(hfq), ctx)?
            }
            ModelSource::Dir(source) => {
                let config = hipfire_runtime::hfq::config_from_safetensors_llama(&source)
                    .map_err(|e| {
                        format!("failed to parse LLaMA/Qwen3 config from config.json: {e}")
                    })?;
                let weights = hipfire_runtime::hfq::load_weights_paroquant_llama(
                    &source, &config, ctx.gpu,
                )
                .map_err(|e| format!("load_weights_paroquant_llama: {e:?}"))?;
                let kv_mode = ctx
                    .kv_mode_override
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| std::env::var("HIPFIRE_KV_MODE").unwrap_or_default());
                let hipfire_runtime::kv_mode::ResolveResult { mode, warning } =
                    hipfire_runtime::kv_mode::resolve(
                        &kv_mode,
                        &hipfire_runtime::kv_mode::DIR_SAFETENSORS_POLICY,
                        config.head_dim,
                    );
                if let Some(w) = warning {
                    eprintln!(
                        "  KV cache: {w} (site {})",
                        hipfire_runtime::kv_mode::DIR_SAFETENSORS_POLICY.site
                    );
                }
                let dims = hipfire_runtime::llama::KvDims {
                    layers: hipfire_runtime::llama::KvLayers::Flat(config.n_layers),
                    n_kv_heads: config.n_kv_heads,
                    head_dim: config.head_dim,
                    max_seq: ctx.max_seq,
                    physical_cap: Some(ctx.max_seq),
                };
                let kv = hipfire_runtime::llama::KvCache::from_mode(
                    mode,
                    hipfire_runtime::llama::KvTarget::Single(ctx.gpu),
                    &dims,
                )
                .map_err(|e| format!("KvCache: {e}"))?;
                let scratch = hipfire_runtime::llama::ForwardScratch::new(ctx.gpu, &config)
                    .map_err(|e| format!("ForwardScratch::new: {e:?}"))?;
                hipfire_arch_llama::LlamaBundle {
                    config,
                    weights,
                    scratch,
                    kv,
                }
            }
        };

        // ── single shared tail ──
        Ok(LoadedModel {
            state: Some(ModelState::Llama(bundle)),
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
            )
        })
    }
}

// ─── Non-core carriers ───────────────────────────────────────────────

/// Generic HFQ-only carrier: every non-core arch has the same load shape
/// (pp>1 guard → `Hfq` destructure → tokenizer-from-metadata → delegate to
/// a `crate::load_*` fn). Each concrete carrier differs only in its
/// `arch_id`, `name`, and the load fn it calls — so they collapse into one
/// struct parameterized by those three values.
pub struct HfqCarrier {
    pub arch_id: u32,
    pub name: &'static str,
    pub load: fn(
        hipfire_runtime::hfq::HfqFile,
        hipfire_runtime::tokenizer::Tokenizer,
        &mut rdna_compute::Gpu,
        usize,
        &str,
    ) -> Result<LoadedModel, String>,
}

impl Carrier for HfqCarrier {
    fn name(&self) -> &'static str {
        self.name
    }
    fn claims_arch_id(&self, arch_id: u32, is_dir: bool) -> bool {
        !is_dir && arch_id == self.arch_id
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err(format!("{}: pp>1 unsupported via registry", self.name));
        }
        let ModelSource::Hfq(hfq) = src else {
            return Err(format!("{}: directory source unsupported", self.name));
        };
        let tokenizer =
            hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                .map_err(|e| format!("tokenizer not found: {e}"))?;
        (self.load)(hfq, tokenizer, ctx.gpu, ctx.max_seq, ctx.path)
    }
}

// ─── DotsOcrCarrier ──────────────────────────────────────────────────

pub struct DotsOcrCarrier;
impl Carrier for DotsOcrCarrier {
    fn name(&self) -> &'static str {
        "dots_ocr"
    }
    fn claims_arch_id(&self, arch_id: u32, is_dir: bool) -> bool {
        !is_dir && arch_id == 8
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            return Err("dots_ocr: pipeline-parallel (pp>1) unsupported".into());
        }
        let ModelSource::Hfq(_) = &src else {
            return Err("dots_ocr: directory source unsupported".into());
        };
        let meta = resolve_source_meta(&src, ctx.path)?;
        let ModelSource::Hfq(mut hfq) = src else {
            unreachable!()
        };

        use hipfire_arch_dots_ocr::DotsOcr;
        use hipfire_runtime::arch::Architecture;
        let config = <DotsOcr as Architecture>::config_from_hfq(&hfq)?;
        let weights = <DotsOcr as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
        let state =
            hipfire_arch_qwen2::qwen2::Qwen2State::new_with_max_seq(ctx.gpu, &config.text, ctx.max_seq)
                .map_err(|e| format!("dots-ocr: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        Ok(LoadedModel {
            qwen2_state: Some(state),
            dots_ocr_config: Some(config),
            dots_ocr_weights: Some(weights),
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
            )
        })
    }
}

// ─── MinimaxCarrier ──────────────────────────────────────────────────

pub struct MinimaxCarrier;
impl Carrier for MinimaxCarrier {
    fn name(&self) -> &'static str {
        "minimax"
    }
    fn claims_arch_id(&self, arch_id: u32, _is_dir: bool) -> bool {
        arch_id == 10
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LoadedModel, String> {
        if ctx.pp > 1 {
            // Preserve the two per-source error strings byte-for-byte.
            return Err(match &src {
                ModelSource::Hfq(_) => "minimax: pipeline-parallel (pp>1) unsupported",
                ModelSource::Dir(_) => "minimax: safetensors + pp>1 unsupported",
            }
            .into());
        }
        // Per-source diagnostic stays at the call site, before resolve_source_meta.
        if let ModelSource::Dir(s) = &src {
            eprintln!("  safetensors arch_id={}", s.arch_id());
        }
        let meta = resolve_source_meta(&src, ctx.path)?;

        // ── source-varying seam: (config, weights) only ──
        use hipfire_runtime::arch::Architecture;
        let (config, weights) = match src {
            ModelSource::Hfq(mut hfq_file) => {
                let config =
                    <hipfire_arch_minimax::arch::MiniMaxM2 as Architecture>::config_from_hfq(
                        &hfq_file,
                    )?;
                let weights =
                    <hipfire_arch_minimax::arch::MiniMaxM2 as Architecture>::load_weights(
                        &mut hfq_file,
                        &config,
                        ctx.gpu,
                    )?;
                (config, weights)
            }
            ModelSource::Dir(source) => {
                let config = config_from_safetensors(&source)
                    .ok_or_else(|| "failed to parse MiniMax config".to_string())?;
                let weights = load_weights_from_safetensors(&source, &config, ctx.gpu)?;
                (config, weights)
            }
        };

        // ── single shared tail (byte-identical to the previous per-arm tails) ──
        let state = MiniMaxState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
            .map_err(|e| format!("minimax: MiniMaxState::new_with_max_seq failed: {e}"))?;
        let eos_tok: u32 = {
            let try_one = |s: &str| -> Option<u32> {
                let ids = meta.tokenizer.encode(s);
                if ids.len() == 1 {
                    Some(ids[0])
                } else {
                    None
                }
            };
            try_one("[e~[")
                .or_else(|| try_one("<|im_end|>"))
                .or_else(|| try_one("</s>"))
                .or_else(|| try_one("<|endoftext|>"))
                .unwrap_or(1)
        };
        Ok(LoadedModel {
            state: Some(ModelState::Minimax(crate::MiniMaxBundle {
                config,
                weights,
                state,
                eos_tok,
            })),
            ..LoadedModel::skeleton(
                meta.arch_id,
                meta.tokenizer,
                ctx.max_seq,
                ctx.max_seq,
                ctx.path.to_string(),
                meta.chat_template,
            )
        })
    }
}
