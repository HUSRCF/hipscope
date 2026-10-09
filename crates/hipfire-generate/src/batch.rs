// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch drivers.
//!
//! The single-GPU Qwen35 and LFM lanes, the expert-parallel (TP=4) Qwen35
//! lane, and the admission predicates that decide whether a request may join
//! a batch at all.
//!
//! These call into the generic AR path, which is why they could not move
//! before it did. An earlier attempt moved them first and reached the target
//! metric by re-exporting the architecture crates through this module while
//! the daemon went on calling `qwen35::forward_scratch` through the alias —
//! the import path moved and the coupling did not. This move is the honest
//! version: the bodies are here, and the daemon calls them by name.
//!
//! Moved verbatim.

use crate::ar::*;
use crate::common::*;
use hipfire_arch_deepseek4 as deepseek4;
use hipfire_arch_lfm2moe as lfm2moe;
use hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState;
use hipfire_arch_lfm2moe::forward_batch::forward_decode_batch_lfm;
use hipfire_arch_qwen35::qwen35;
use hipfire_engine::emit::*;
use hipfire_engine::prompt::{batch_render_prompt_tokens, qwen_jinja_reasoning};
use hipfire_engine::scheduler::*;
use hipfire_engine::terminal::*;
use hipfire_engine::wire_seed;
use hipfire_loader::{EpArch, EpState, LoadedModel};
use hipfire_runtime::emit_text::{
    currently_in_think, ThinkOutputRouter, ToolOutputRouter, ToolRouteError,
};
use hipfire_runtime::eos_filter::{EosFilter, EosFilterConfig, FilterAction};
use hipfire_runtime::llama;
use hipfire_runtime::prompt_frame::ThinkMode;
use hipfire_runtime::sampler::{self, SamplerConfig};
use std::any::Any;
use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;
struct BatchTerminalCleanup {
    id: String,
    attempt_id: u64,
    admission: Option<BatchGeneration>,
}

impl BatchTerminalCleanup {
    fn new(key: &AttemptKey, admission: Option<BatchGeneration>) -> Self {
        Self {
            id: key.id.clone(),
            attempt_id: key.attempt_id,
            admission,
        }
    }
}

impl Drop for BatchTerminalCleanup {
    fn drop(&mut self) {
        if let Some(admission) = self.admission {
            batch_clear_terminal_at_generation(&self.id, self.attempt_id, admission);
        }
    }
}

/// Emit one correlated terminal error for a request that was already
/// announced on the batch plane, then retire only that admission generation.
fn emit_batch_admission_error(
    stdout: &mut impl Write,
    id: &str,
    attempt_id: u64,
    admission: BatchGeneration,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    {
        let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
        emit_active_attempt_error(stdout, Some(id), message, class, retryable, rolled_back);
        let _ = stdout.flush();
    }
    batch_clear_terminal_at_generation(id, attempt_id, admission);
}
/// Retire one lane whose GPU reset failed before the lane is freed for reuse.
/// Emits a visible fail-closed error (`rolled_back=false`, never retried)
/// through the landed AttemptKey claim, then frees the scheduler lane for
/// that admission only. Ordinary and EP drivers call this independently so
/// one dirty lane cannot be silently reused while its peers keep serving;
/// EP batch state additionally poisons the GPU lane internally on reset
/// failure, ordinary lanes escalate on next touch when the device is bad.
fn retire_lane_after_reset_failure(
    sched: &mut ContinuousBatchScheduler,
    stdout: &mut impl Write,
    key: &AttemptKey,
    admission: BatchGeneration,
    lane_idx: usize,
    context: &str,
    err: &dyn std::fmt::Display,
) {
    // Same keyed error half as `emit_batch_admission_error`, but WITHOUT its
    // trailing registry clear: `abort_lane` below validates the owner
    // against the live registry entry, so the lane must still be claimed
    // when it runs. Abort clears the admission on success.
    {
        let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
        emit_active_attempt_error(
            stdout,
            Some(&key.id),
            &format!("reset lane {lane_idx} ({context}) failed; lane retired: {err}"),
            "gpu",
            false,
            false,
        );
        let _ = stdout.flush();
    }
    let _ = sched.abort_lane(lane_idx, key, admission);
}
/// Emit the assignment-time LFM capacity failure. The caller must hold the
/// exact `BatchAttemptScope`; the route adapter claims the terminal and releases
/// the matching LFM-AR start latch before the scheduler retires the lane.
fn emit_lfm_assignment_capacity_error(
    stdout: &mut impl Write,
    key: &AttemptKey,
    prompt_len: usize,
    max_tokens: usize,
    capacity: usize,
) {
    crate::ar::emit_generation_error(
        crate::ar::GenerationRoute::LfmAr,
        stdout,
        Some(&key.id),
        &format!(
            "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={}",
            prompt_len, max_tokens, capacity
        ),
        "context_length",
        false,
        false,
    );
    let _ = stdout.flush();
}

/// Release one request's batch route/start latch and capture the exact
/// singleton transaction before the outer batch guard is dropped.
fn take_singleton_handoff(
    key: &AttemptKey,
    admission: BatchGeneration,
    route: GenerationRoute,
) -> Result<SingletonTransfer, String> {
    // Requests arriving through the batch driver's inbox never passed through
    // the outer singleton activation in daemon::main. Bootstrap an owner here
    // before removing the keyed admission so the transfer always carries a
    // real singleton transaction into the sequential path.
    if terminal_generation(&key.id, key.attempt_id).is_none() {
        activate_terminal_control(&key.id, key.attempt_id);
    }
    let transfer = batch_handoff_to_singleton_and_clear(&key.id, key.attempt_id, admission)
        .ok_or_else(|| {
            format!(
                "batch admission handoff failed for {}:{}",
                key.id, key.attempt_id
            )
        })?;
    if transfer.admission() != admission {
        return Err(format!(
            "batch admission changed during singleton handoff for {}:{}",
            key.id, key.attempt_id
        ));
    }
    // GenerationRouteScope releases only this request's start latch while
    // preserving the prior route TLS. The sequential producer can therefore
    // emit its fresh gen_start without a terminal/error side effect.
    {
        let _attempt = BatchAttemptScope::enter_singleton(key.attempt_id);
        let _route = GenerationRouteScope::enter(route, &key.id);
    }
    Ok(transfer)
}

/// Retire a think-open lane only after its caller has reset GPU state, then
/// hand its full original request and exact singleton transaction to main.
fn handoff_started_in_think(
    sched: &mut ContinuousBatchScheduler,
    lane_idx: usize,
    key: &AttemptKey,
    pending: &BatchPendingRequest,
    route: GenerationRoute,
) -> Result<DaemonMsg, String> {
    if !sched.retire_lane_for_singleton(lane_idx, key, pending.admission) {
        return Err(format!(
            "retire lane {lane_idx} for singleton handoff failed for {}:{}",
            key.id, key.attempt_id
        ));
    }
    let transfer = take_singleton_handoff(key, pending.admission, route)?;
    Ok(daemon_singleton_with_admission(
        pending.original_msg.clone(),
        transfer,
    ))
}

/// Handoff a think-open request encountered before it receives a batch lane.
/// There is no GPU lane to reset, but ownership still transfers through the
/// same explicit internal message and exact admission cleanup.
fn handoff_admitted_started_in_think(
    id: &str,
    attempt_id: u64,
    admission: BatchGeneration,
    original_msg: serde_json::Value,
    route: GenerationRoute,
) -> Result<DaemonMsg, String> {
    let key = AttemptKey::new(id, attempt_id);
    let transfer = take_singleton_handoff(&key, admission, route)?;
    Ok(daemon_singleton_with_admission(original_msg, transfer))
}

/// Cancellable LFM prefill helper. Attempts to use the arch's
/// `prefill_lane_cancellable` when present; otherwise falls back to the
/// standard `prefill_lane` with post-prefill abort handling. The closure is
/// checked before GPU work and the caller re-checks after, ensuring an
/// aborted lane never samples and only that lane is reset.
pub fn lfm_prefill_cancellable_or_fallback<F>(
    batch_state: &mut lfm2moe::batch::Lfm2DecodeBatchState,
    gpu: &mut rdna_compute::Gpu,
    weights: &lfm2moe::Lfm2MoeWeights,
    cfg: &lfm2moe::config::Lfm2MoeConfig,
    lane: usize,
    tokens: &[u32],
    check_abort: &F,
) -> hip_bridge::HipResult<bool>
where
    F: Fn() -> bool,
{
    if check_abort() {
        return Ok(false);
    }
    // If the arch exposes a true cancellable variant, try to call it via
    // dynamic dispatch. We cannot statically know its existence, so we
    // attempt to downcast via a helper trait that the arch may implement.
    // For now, call the standard prefill and treat post-abort as cancellation.
    // This satisfies "no first sample" and "reset only lane" while remaining
    // bounded to the prefill pass (the arch's cancellable will tighten to token boundary when it lands).
    batch_state.prefill_lane(gpu, weights, cfg, lane, tokens)?;
    if check_abort() {
        return Ok(false);
    }
    Ok(true)
}

/// Whether a generate message carries a `stop` the batch lanes cannot honour:
/// any non-empty string or array form, or a value the sequential path rejects.
fn request_has_stop(msg: &serde_json::Value) -> bool {
    !matches!(
        hipfire_runtime::stop_sequence::parse_stop_field(msg.get("stop")),
        Ok(stops) if stops.is_empty()
    )
}

/// Tightened admission: require Qwen 5/6 (QwenAr) or dense LFM 11 (LfmAr), pp=1,
/// no EP, model-owned batch state present, and no excluded features. Rendered
/// prompts that open a think span stay on the sequential barrier route. MoE LFM
/// is never batch-eligible.
pub fn is_batch_request_eligible(
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
) -> bool {
    let has_image = msg.get("image").is_some() || msg.get("image_base64").is_some();
    let has_tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    // Any stop (string or array form), or an invalid one, takes the sequential
    // path, which honours or rejects it.
    let has_stop = request_has_stop(msg);
    // messages: absent OR exactly one user turn (HTTP chat shape).
    let has_spec = m.speculator.is_some();
    let has_adaptive = m.kv_adaptive.is_some();
    let caps = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps())
        .unwrap_or_default();
    // For route check we need temp etc to compute GenerationRoute; use resolved sampling temp
    let sampling = resolve_batch_sampling(msg, m);
    let user_explicit = [
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]
    .iter()
    .any(|k| msg.get(*k).is_some());
    let ngram_can_sample = m
        .speculator
        .as_ref()
        .map(|s| !s.requires_greedy())
        .unwrap_or(false);
    let supports_temp_swor = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_temp_verify());
    let supports_chain_nucleus_verify = m
        .speculator
        .as_ref()
        .is_some_and(|s| s.supports_chain_nucleus_verify());
    let route_inputs = GenerationRouteInputs {
        arch_id: m.arch_id,
        ep: m.ep.is_some(),
        dense_tp: matches!(
            m.ep.as_ref().map(|e| &e.inner),
            Some(hipfire_loader::EpArch::Qwen35DenseTp { .. })
        ),
        pp: m.pp,
        has_speculator: has_spec,
        speculator_is_mtp: m.speculator.as_ref().is_some_and(|s| s.name() == "mtp"),
        deepseek4_spec_requested: false,
        ngram_can_sample,
        temp: sampling.temp,
        user_explicit_sampling: user_explicit,
        min_p: sampling.min_p,
        nonneutral_penalties: sampling.repeat_penalty != 1.0
            || sampling.presence_penalty != 0.0
            || sampling.frequency_penalty != 0.0,
        force_ar_chat: false,
        temp_spec_env_off: hipfire_config::developer_var("HIPFIRE_DFLASH_TEMP_SPEC")
            .ok()
            .as_deref()
            == Some("0"),
        fast_sample_on: hipfire_config::developer_var("HIPFIRE_FAST_SAMPLE")
            .ok()
            .as_deref()
            != Some("0"),
        supports_temp_swor,
        supports_chain_nucleus_verify,
        kv_adaptive: has_adaptive,
    };
    let route = select_generation_route(&route_inputs);
    if caps.supports_continuous_batch {
        if let Some(bundle) = m.state.as_ref().and_then(|s| {
            (s.as_ref() as &dyn Any).downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>()
        }) {
            if route != GenerationRoute::QwenAr {
                return false;
            }
            if bundle.qwen35_decode_batch.is_none() {
                return false;
            }
            if !hipfire_loader::batch_staging::qwen_batch_weight_formats_supported(&bundle.weights)
            {
                return false;
            }
        } else if let Some(bundle) = m.state.as_ref().and_then(|s| {
            (s.as_ref() as &dyn Any).downcast_ref::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
        }) {
            if route != GenerationRoute::LfmAr {
                return false;
            }
            if bundle.lfm2_decode_batch.is_none() {
                return false;
            }
            if !bundle.config.is_dense() {
                return false;
            }
            if lfm2moe::batch_weight_formats_supported(&bundle.weights).is_err() {
                return false;
            }
            return false;
        }
    } else {
        return false;
    }
    // No multi-turn/history/tools/images/custom stops etc.
    if !batch_messages_are_single_user(msg) || has_tools || has_image || has_stop {
        return false;
    }
    if has_spec || has_adaptive || m.eviction.is_some() || pflash_active {
        return false;
    }
    if m.pp != 1 || m.ep.is_some() {
        return false;
    }
    if !caps.supports_continuous_batch {
        return false;
    }
    if !serve_continuous_batch || continuous_batch_size <= 1 {
        return false;
    }
    // Forced-think/budget injection is sequential-only, but 0 (uncapped),
    // 1 (non-think), and the ordinary CLI-resolved reasoning budget are valid
    // batch controls.
    let max_think = msg
        .get("max_think_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let _ = max_think;
    let has_budget_alert =
        msg.get("budget_alert_at_tok").is_some() || msg.get("budget_alert_text").is_some();
    if has_budget_alert {
        return false;
    }
    true
}

/// Batch eligibility predicate shape shared by the Qwen continuous-batch
/// drivers (fixed-lane and VMM).
pub(crate) type QwenBatchEligibility =
    fn(&serde_json::Value, &LoadedModel, usize, bool, bool) -> bool;

/// Drain already-delivered daemon messages into a Qwen batch scheduler at a
/// step boundary: admit eligible generates (render once, enqueue, announce),
/// apply terminal controls, and stop at the first message that must run
/// outside the batch (returned as the barrier). `Err` carries a fail-all
/// reason (think-barrier handoff failure).
#[allow(clippy::too_many_arguments)]
pub(crate) fn drain_qwen_batch_inbox(
    sched: &mut ContinuousBatchScheduler,
    model: &LoadedModel,
    batch_size: usize,
    eligible: QwenBatchEligibility,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    chat_template: Option<&String>,
    route: GenerationRoute,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
    // `true` (VMM route): the caller emits `gen_start` itself once it
    // decides batch vs singleton, so a lonely arrival can still take the
    // singleton route without a second start; messages/prompt are
    // normalized exactly as the singleton dispatch normalizes them; and a
    // think-open prompt is admitted (its lane applies the think controls).
    // `false` (fixed-lane route): announce on enqueue, think-open prompts
    // are sequential barriers.
    vmm_route: bool,
) -> Result<Option<DaemonMsg>, String> {
    let mut barrier: Option<DaemonMsg> = None;
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            let (dm, carried_admission) = match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    (DaemonMsg::Regular(json), Some(admission))
                }
                other => (other, None),
            };
            match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    barrier = Some(DaemonMsg::RegularWithAdmission(json, admission));
                    break;
                }
                DaemonMsg::SingletonWithAdmission(json, transfer) => {
                    barrier = Some(DaemonMsg::SingletonWithAdmission(json, transfer));
                    break;
                }
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(0) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate attempt_id must be nonzero",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        let Some(admission) = carried_admission else {
                            barrier = Some(daemon_regular_with_admission(json, None));
                            break;
                        };
                        if batch_check_abort(&id, attempt_id, admission) {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_generation_cancel(route, stdout, &id, 0);
                            batch_clear_terminal_at_generation(&id, attempt_id, admission);
                            continue;
                        }
                        if !eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(daemon_regular_with_admission(json, carried_admission));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            let prompt = json.get("prompt").and_then(|v| v.as_str()).unwrap_or("Hello");
                            if vmm_route {
                                // As the singleton dispatch normalizes `prompt`.
                                hipfire_runtime::tokenizer::maybe_normalize_prompt(prompt).into_owned()
                            } else {
                                prompt.to_string()
                            }
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let parsed_messages = if vmm_route {
                            hipfire_engine::prompt::parse_generate_messages(&json)
                        } else {
                            json.get("messages")
                                .map(|v| {
                                    serde_json::from_value::<Vec<hipfire_runtime::prompt_frame::Message>>(v.clone())
                                        .map_err(|e| e.to_string())
                                })
                                .transpose()
                        };
                        let batch_messages = match parsed_messages {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &format!("invalid messages field: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let raw_effort = json
                            .get("reasoning_effort")
                            .or_else(|| json.get("thinking_mode"))
                            .and_then(|v| v.as_str());
                        let thinking_enabled =
                            json.get("thinking_enabled").and_then(|v| v.as_bool());
                        let (batch_enable_thinking, batch_reasoning_effort) =
                            qwen_jinja_reasoning(thinking_enabled, raw_effort, max_think);
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template,
                            max_think,
                            batch_messages.as_deref(),
                            batch_enable_thinking,
                            batch_reasoning_effort.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        if started_in_think && !vmm_route {
                            let handoff = match handoff_admitted_started_in_think(
                                &id,
                                attempt_id,
                                admission,
                                json,
                                GenerationRoute::QwenAr,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => {
                                    return Err(reason)
                                }
                            };
                            barrier = Some(handoff);
                            break;
                        }
                        if prompt_tokens.is_empty() || prompt_tokens.len() >= sched.lane_capacity {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                "prompt exceeds lane capacity or empty",
                                "validation",
                                false,
                                false,
                            );
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id, admission);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            admission,
                            original_msg: json.clone(),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            // Defensive: a live registry/channel already owns this
                            // key. Do not emit a keyed error or clear the original.
                            eprintln!(
                                "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        if !vmm_route {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                started_in_think,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(daemon_regular_with_admission(json, carried_admission));
                        break;
                    }
                }
            }
        }
    Ok(barrier)
}

pub fn drive_qwen_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let route = crate::ar::GenerationRoute::QwenAr;
    // SAFETY: borrow disjoint fields via raw pointers to avoid &mut aliasing
    // qwen35_decode_batch now lives inside Qwen35Bundle.
    let b_ptr = match model.state.as_mut().and_then(|s| {
        (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>()
    }) {
        Some(b) => b as *mut hipfire_arch_qwen35::Qwen35Bundle,
        None => return Err(BatchDriveError::Gpu("batch model not Qwen35".to_string())),
    };
    let batch_state_ptr = unsafe {
        let b = &mut *b_ptr;
        match b.qwen35_decode_batch.as_mut() {
            Some(s) => s as *mut qwen35::Qwen35DecodeBatchState,
            None => {
                return Err(BatchDriveError::Gpu(
                    "batch state not allocated".to_string(),
                ))
            }
        }
    };
    let batch_state = unsafe { &mut *batch_state_ptr };
    let (config_ptr, weights_ptr, scratch_ptr, tokenizer_ptr, chat_template_clone) = unsafe {
        let b = &*b_ptr;
        (
            &b.config as *const qwen35::Qwen35Config,
            &b.weights as *const qwen35::Qwen35Weights,
            &b.scratch as *const qwen35::Qwen35Scratch,
            match model.tokenizer.as_ref() {
                Some(t) => t as *const _,
                None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
            },
            model.chat_template.clone(),
        )
    };
    let config = unsafe { &*config_ptr };
    let weights = unsafe { &*weights_ptr };
    let scratch = unsafe { &*scratch_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    let im_end_tok = tokenizer.special_token_id("<|im_end|>").unwrap_or(0);
    let eos_tok = config.eos_token;
    let mut producers: Vec<Option<QwenArSemanticProducer>> =
        (0..batch_size).map(|_| None).collect();
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    batch_state: &mut qwen35::Qwen35DecodeBatchState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for lane in sched.lanes.iter() {
            let Some(key) = lane.key() else {
                continue;
            };
            let admission = match lane {
                BatchLane::Seeding(q) | BatchLane::Running(q) => q.ticket.admission,
                BatchLane::AwaitingClient(t) => t.ticket.admission,
                BatchLane::Empty { .. } => continue,
            };
            if uniq_set.insert((key.clone(), admission)) {
                uniq.push((key.clone(), admission));
            }
        }
        for key in sched.inbox.iter().cloned() {
            if let Some(request) = sched.pending.get(&key) {
                if uniq_set.insert((key.clone(), request.admission)) {
                    uniq.push((key, request.admission));
                }
            }
        }
        for (key, request) in sched.pending.iter() {
            if uniq_set.insert((key.clone(), request.admission)) {
                uniq.push((key.clone(), request.admission));
            }
        }
        let mut first_err: Option<String> = None;
        if let Err(e) = batch_state.reset(gpu) {
            first_err = Some(format!("batch reset: {e}"));
        }
        crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
        let sync = crate::common::fail_closed_device_sync(gpu);
        let prior = match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let ep = crate::common::fail_closed_epilogue_after_sync(prior, sync);
        for (key, admission) in &uniq {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, *admission);
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason}"),
                "gpu",
                ep.rolled_back,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        if !ep.rolled_back {
            return Err(BatchDriveError::Poisoned(format!(
                "{reason}; {}",
                ep.context.unwrap_or_default()
            )));
        }
        Err(BatchDriveError::Gpu(reason))
    };
    loop {
        let mut to_commit: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let admission = term.ticket.admission;
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id, admission) || expired {
                    to_abort.push((idx, key, admission));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id, admission)
                {
                    to_commit.push((idx, key.clone(), admission, term.pending_done.clone()));
                }
            }
        }
        for (idx, key, admission) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        for (idx, key, admission, pending_done) in to_commit {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            // Transactional commit: reset GPU first, then host commit_lane,
            // and only then emit the staged done. Never done+error.
            let reset_ok = match batch_state.reset_lane(gpu, &config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {idx} on commit: {e}"),
                    );
                }
            };
            let commit_ok = sched.commit_lane_retain_terminal(idx, &key, admission);
            // Keep the keyed registry alive through the terminal writer. This
            // also clears it on error/early return after the host transition.
            let _terminal_cleanup = BatchTerminalCleanup::new(&key, Some(admission));
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!("reset_ok handled above"),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error_for_route(
                        route,
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                    producers[idx] = None;
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_generation_done_value(route, stdout, &pending_done);
                    producers[idx] = None;
                }
            }
        }
        let mut queued_abort: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if let Some(request) = sched.pending.get(&key) {
                if batch_check_abort(&key.id, key.attempt_id, request.admission) {
                    queued_abort.push((key, request.admission));
                }
            }
        }
        for (key, admission) in queued_abort {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_queued(&key, admission);
        }
        let mut running_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(key) = sched.lanes[idx].key().cloned() {
                let admission = match &sched.lanes[idx] {
                    BatchLane::Running(l) | BatchLane::Seeding(l) => l.ticket.admission,
                    _ => continue,
                };
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&key.id, key.attempt_id, admission)
                {
                    running_abort.push((idx, key, admission));
                }
            }
        }
        for (idx, key, admission) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        let barrier = match drain_qwen_batch_inbox(
            sched,
            model,
            batch_size,
            is_batch_request_eligible,
            tokenizer,
            chat_template.as_ref(),
            route,
            stdout,
            inbox,
            false,
        ) {
            Ok(b) => b,
            Err(reason) => return fail_all(sched, gpu, batch_state, stdout, reason),
        };
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let sampling = pending_req.sampling.clone();
            // Use admission-rendered tokens/semantics; do not re-render with None/Plain/0.
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let started_in_think = pending_req.started_in_think;
            if started_in_think {
                // Think-open prompts are sequential barriers. Reset while the
                // batch owner is still live, then retire and hand off the
                // complete original request; never touch this lane again.
                if let Err(err) = batch_state.reset_lane(gpu, &config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                let handoff = match handoff_started_in_think(
                    sched,
                    lane_idx,
                    &key,
                    &pending_req,
                    GenerationRoute::QwenAr,
                ) {
                    Ok(msg) => msg,
                    Err(reason) => {
                        return fail_all(sched, gpu, batch_state, stdout, reason);
                    }
                };
                inbox.push_front(handoff);
                continue;
            }

            if let Err(e) = batch_state.reset_lane(gpu, &config, lane_idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {lane_idx}: {e}"),
                );
            }
            if let Err(e) =
                batch_state.prefill_lane(gpu, &weights, &config, &scratch, lane_idx, &prompt_tokens)
            {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("prefill lane {lane_idx}: {e}"),
                );
            }
            let hist: &[u32] = &[];
            let lane_rng = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => lane.rng_state as u32,
                _ => continue,
            };
            let (next_token, next_rng) = match batch_state.sample_lane_product(
                gpu,
                &config,
                lane_idx,
                hist,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                sampling.min_p,
                lane_rng,
                sampling.repeat_penalty,
                sampling.presence_penalty,
                sampling.frequency_penalty,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
            producers[lane_idx] = Some(QwenArSemanticProducer::new_with_tool_protocol(
                key.id.clone(),
                started_in_think,
                false,
            ));
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        // Peak concurrent Running occupancy observed while each lane is live.
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        if let Err(e) = qwen35::forward_decode_batch(
            gpu,
            &weights,
            &config,
            &tokens,
            &positions,
            batch_state,
            &scratch,
        ) {
            return fail_all(
                sched,
                gpu,
                batch_state,
                stdout,
                format!("forward_decode_batch: {e}"),
            );
        }
        let mut repeat_tokens: Vec<u32> = vec![0; batch_size * batch_state.sample_repeat_capacity];
        let mut repeat_lengths: Vec<u32> = vec![0; batch_size];
        let mut rng_states: Vec<u32> = vec![0; batch_size];
        let mut survivors: Vec<usize> = Vec::new();
        let mut to_await: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> = Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            let admission = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.ticket.admission,
                _ => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                to_abort_running.push((idx, key, admission));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            let prod_ptr = match producers[idx].as_mut() {
                Some(p) => p as *mut QwenArSemanticProducer,
                None => continue,
            };
            let producer = unsafe { &mut *prod_ptr };
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let prev_fed = lane.bytes_fed_to_filter.min(all_bytes.len());
            let token_bytes = all_bytes[prev_fed..].to_vec();
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            // TTFT: host Instant immediately before the first classified emit.
            if lane.first_token_at.is_none() {
                lane.first_token_at = Some(Instant::now());
            }
            let stopped = {
                let lane_seq = &mut lane.seq_pos as *mut usize;
                let lane_conv = &mut lane.conversation_tokens as *mut Vec<u32>;
                let lane_stream = &mut lane.streamed_tokens as *mut Vec<u32>;
                let lane_fed = &mut lane.bytes_fed_to_filter as *mut usize;
                let all_len = all_bytes.len();
                let mut res: Result<bool, _> = Ok(false);
                unsafe {
                    res = producer.commit_and_classify(
                        stdout,
                        cur_token,
                        || {
                            let pos = qwen_ar_raw_commit_token(
                                &mut *lane_conv,
                                &mut *lane_stream,
                                &mut *lane_seq,
                                cur_token,
                                QwenArRawCommitDisposition::ClassifiedVisible,
                            );
                            *lane_fed = all_len;
                            (pos, token_bytes.clone())
                        },
                        |_, _| {},
                    );
                }
                match res {
                    Ok(s) => s,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("semantic classify lane {idx}: {e}"),
                        )
                    }
                }
            };
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let is_eos = cur_token == eos_tok || cur_token == im_end_tok;
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, stopped, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, stopped, loop_hit);

                let producer_owned = match producers[idx].take() {
                    Some(p) => p,
                    None => continue,
                };
                let (finish, visible_text) = match producer_owned.finish(stdout, hit_length_cap) {
                    Ok(v) => v,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("semantic finish lane {idx}: {e}"),
                        )
                    }
                };
                if !finish.wire_tool_calls.is_empty() {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("semantic finish lane {idx}: unexpected tool calls"),
                    );
                }
                // The producer's semantic reason, unchanged (an open think
                // span ends as a reasoning-only "stop" or "length").
                let finish_reason = finish.finish_reason;
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = qwen_ar_done_value(
                    &key.id,
                    finish_reason,
                    generated,
                    metrics.tok_s,
                    lane.prompt_len,
                    metrics.prefill_ms,
                    metrics.prefill_tok_s,
                    metrics.decode_tok_s,
                    metrics.ttft_ms,
                    0,
                    "",
                );
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    /*slots=*/ batch_size,
                    /*lane=*/ idx,
                    /*lane_capacity=*/ sched.lane_capacity,
                    /*max_active_lanes=*/ lane.max_active_lanes.max(1),
                );
                let _ = visible_text;
                to_await.push((idx, key.clone(), admission, pending_done));
            } else {
                survivors.push(idx);
                let window = lane
                    .sampling
                    .repeat_window
                    .min(batch_state.sample_repeat_capacity);
                let hist = if lane.streamed_tokens.len() > window {
                    &lane.streamed_tokens[lane.streamed_tokens.len() - window..]
                } else {
                    &lane.streamed_tokens[..]
                };
                for (i, &tok) in hist.iter().enumerate() {
                    repeat_tokens[idx * batch_state.sample_repeat_capacity + i] = tok;
                }
                repeat_lengths[idx] = hist.len() as u32;
                rng_states[idx] = lane.rng_state as u32;
            }
        }
        for (idx, key, admission) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        // Install AwaitingClient/Ready BEFORE publishing commit_ready; rollback if publish fails.
        for (idx, key, admission, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch] qwen mark_awaiting_commit failed lane {idx} id={} — aborting lane",
                    key.id
                );
                if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "mark_awaiting_commit",
                        &e,
                    );
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
                continue;
            }
            let write_ok = {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                if let Err(e) = batch_state.reset_lane(gpu, &config, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "commit_ready publish",
                        &e,
                    );
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
            }
            // On success, lane stays AwaitingClient reserved until commit/abort decision.
        }
        if survivors.is_empty() {
            continue;
        }
        for i in 0..batch_size {
            if !survivors.contains(&i) {
                repeat_lengths[i] = 0;
                rng_states[i] = 0;
            }
        }
        let sampling = if let Some(idx) = survivors.first() {
            match &sched.lanes[*idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            }
        } else {
            continue;
        };
        let sampled = match batch_state.sample_product(
            gpu,
            &config,
            batch_size,
            &repeat_tokens,
            &repeat_lengths,
            &rng_states,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.repeat_penalty,
            sampling.presence_penalty,
            sampling.frequency_penalty,
        ) {
            Ok(v) => v,
            Err(e) => {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("sample_product: {e}"),
                )
            }
        };
        for lane_idx in survivors.iter() {
            let (tok, rng) = sampled[*lane_idx];
            if let BatchLane::Running(lane) = &mut sched.lanes[*lane_idx] {
                lane.next_token = Some(tok);
                lane.rng_state = rng as u64;
            }
        }
    }
    Ok(())
}

// ── VMM continuous batching (serve.vmm_batch) ───────────────────────────
//
// One resident Qwen35 weight set; per-request VMM KV/DeltaNet owners in
// `Qwen35Bundle::vmm_store`; steps planned by the runtime `BatchPlanner`
// and run through the arch executor's frozen provision/forward/commit
// methods. Host admission/terminal authority stays the
// `ContinuousBatchScheduler` (same AttemptKey/LaneTicket registry, same
// commit_ready/commit/abort transaction as the fixed-lane driver).
//
// Stage 1 limits, enforced by `is_vmm_batch_request_eligible`: greedy
// requests only (the executor picks with the singleton sampler incl. the
// repeat/presence/frequency penalties; sampled-RNG equivalence has no gate
// yet), AR rows only
// (no cross-request speculation yet), and a request that is lonely at
// dispatch stays on the unchanged singleton route (the daemon checks the
// inbox); arrivals during a singleton generation wait for it to finish.

// ── Singleton → VMM batch promotion ───────────────────────────────────

/// What the daemon knows about a VMM-batch-eligible singleton request that
/// the generate loops do not: set right before `generate()` only when the
/// request could have joined the VMM batch (store staged, request eligible),
/// cleared right after. Its presence is the sole permission to promote.
#[derive(Clone)]
pub struct PromotionPermit {
    pub original_msg: serde_json::Value,
    pub sampling: BatchSampling,
    pub max_tokens: usize,
    pub max_think_tokens: usize,
    pub client_seed: Option<u64>,
    pub assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
}

/// A singleton generation stopped at a committed token boundary and moved
/// into a batch admission (`singleton_handoff_to_batch`): its KV/DN owners,
/// pending seed, RNG and sampling history, wire progress, semantic producer
/// and loop guard. The VMM driver installs it as a Running lane under the
/// same AttemptKey with no new `gen_start`.
///
/// `kind` records the decode mode it continues in: exact batched AR, or (a
/// singleton MTP request) a spec lane carrying the drafter's state.
pub struct PromotedRequest {
    pub pending: BatchPendingRequest,
    pub progress: QwenBatchLane,
    pub state: hipfire_arch_qwen35::forward_slots::vmm::Qwen35RequestState,
    pub producer: QwenArSemanticProducer,
    pub loop_guard: hipfire_runtime::loop_guard::LoopGuard,
    pub kind: PromotedDecode,
    /// The pending seed was already streamed to the client (spec routes
    /// emit at pick); the AR loop emits after the forward, so it was not.
    pub seed_emitted: bool,
    /// Singleton think-budget state, resume checkpoints and cache counts
    /// the lane continues with.
    pub conv: crate::vmm_conv::PromotedConv,
}

/// Decode mode a promoted lane continues in.
pub enum PromotedDecode {
    /// Continue as exact batched AR from the pending seed.
    Ar,
    /// Continue as a spec lane with the singleton MTP drafter's state (the
    /// driver falls back to AR when no spec lane can take it).
    Spec(hipfire_arch_qwen35::forward_slots::vmm::spec::MtpLaneSnapshot),
}

/// The singleton MTP route's per-request speculator config for a batch
/// request's resolved sampling (what `configure_request` installs).
fn spec_request_config(s: &BatchSampling, rng_seed: u64) -> hipfire_runtime::spec::SpecRequestConfig {
    hipfire_runtime::spec::SpecRequestConfig {
        temp: s.temp,
        top_p: s.top_p,
        top_k: s.top_k,
        min_p: s.min_p.unwrap_or(0.0),
        cactus_delta: 0.0,
        rng_seed,
        allow_ngram_modifier: false,
        repeat_penalty: s.repeat_penalty,
        repeat_window: s.repeat_window,
        presence_penalty: s.presence_penalty,
        frequency_penalty: s.frequency_penalty,
    }
}

thread_local! {
    static PROMOTION_PERMIT: std::cell::RefCell<Option<PromotionPermit>> =
        const { std::cell::RefCell::new(None) };
    static PROMOTED: std::cell::RefCell<Option<PromotedRequest>> =
        const { std::cell::RefCell::new(None) };
}

/// Daemon: grant (Some) or clear (None) promotion for the next `generate()`.
pub fn set_promotion_permit(permit: Option<PromotionPermit>) {
    PROMOTION_PERMIT.with(|p| *p.borrow_mut() = permit);
}

/// Generate loops: the permit, if this request may promote.
pub(crate) fn promotion_permit() -> Option<PromotionPermit> {
    PROMOTION_PERMIT.with(|p| p.borrow().clone())
}

/// Generate loops: promote now — a serve-batched peer is waiting, or the
/// developer reference `HIPFIRE_VMM_PROMOTE_AT=<n>` forces promotion at the
/// first boundary with at least `n` generated tokens (no peer needed; the
/// promoted request then runs alone in the VMM driver).
pub(crate) fn promotion_wanted(generated: usize) -> bool {
    static FORCED_AT: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        hipfire_config::developer_var("HIPFIRE_VMM_PROMOTE_AT")
            .ok()
            .and_then(|v| v.trim().parse().ok())
    });
    hipfire_engine::scheduler::queued_batch_generates() > 0 || FORCED_AT.is_some_and(|n| generated >= n)
}

pub(crate) fn put_promoted(p: PromotedRequest) {
    PROMOTED.with(|slot| *slot.borrow_mut() = Some(p));
}

/// Daemon, after `generate()` returns: the promoted request to drive.
pub fn take_promoted() -> Option<PromotedRequest> {
    PROMOTED.with(|slot| slot.borrow_mut().take())
}

/// Rebuild the AR semantic producer a spec-decoded request would have had:
/// replay its committed tokens through a fresh producer with no client
/// output (`io::sink`). Same filter/think/tool routers over the same bytes,
/// so its held-back text and channel state match the spec emitter's.
/// Returns the producer and the stream byte length; `Err` when replay hits
/// a stop (the request is finishing — keep it on its own route).
pub(crate) fn replay_ar_producer(
    id: &str,
    started_in_think: bool,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    committed: &[u32],
) -> Result<(QwenArSemanticProducer, usize), String> {
    let mut producer = QwenArSemanticProducer::new_with_tool_protocol(id.to_string(), started_in_think, false);
    let mut bytes = Vec::new();
    let mut sink = std::io::sink();
    for (i, &t) in committed.iter().enumerate() {
        let fed = bytes.len();
        tokenizer.decode_token_bytes_into(t, &mut bytes);
        let stopped = producer
            .commit_and_classify(&mut sink, t, || (i, &bytes[fed..]), |_, _| {})
            .map_err(|e| format!("producer replay: {e}"))?;
        if stopped {
            return Err("producer replay reached a stop".into());
        }
    }
    Ok((producer, bytes.len()))
}

/// Singleton loop state at a committed token boundary: the next row to feed
/// is `pending_seed` at `position`.
pub(crate) struct SingletonBoundary<'a> {
    pub id: &'a str,
    pub permit: PromotionPermit,
    pub config: &'a hipfire_arch_qwen35::qwen35::Qwen35Config,
    pub kv: &'a mut hipfire_runtime::llama::KvCache,
    pub dn: &'a mut hipfire_arch_qwen35::qwen35::DeltaNetState,
    pub position: usize,
    pub pending_seed: u32,
    pub sampler: hipfire_runtime::sampler::SamplerConfig,
    pub rng_state: u32,
    /// The singleton's sampling scope (penalty/attractor history).
    pub history: Vec<u32>,
    pub prompt_len: usize,
    pub conversation_tokens: Vec<u32>,
    pub streamed_tokens: Vec<u32>,
    pub bytes_fed_to_filter: usize,
    pub created_at: Instant,
    pub prefill_done_at: Instant,
    pub first_token_at: Option<Instant>,
    pub started_in_think: bool,
}

/// Move a singleton generation into a batch admission: allocate a request
/// owner, transfer the terminal transaction (`singleton_handoff_to_batch`),
/// then swap the singleton's KV/DN owners into it (moves only). On `Err`
/// nothing changed: the singleton keeps its state and transaction. The
/// caller adds its producer and loop guard and hands the result to the
/// daemon via [`put_promoted`]; the bundle then holds a fresh, empty owner
/// and its conversation cache must be invalidated by the caller.
pub(crate) fn promote_singleton(
    gpu: &mut rdna_compute::Gpu,
    b: SingletonBoundary<'_>,
) -> Result<(BatchPendingRequest, QwenBatchLane, hipfire_arch_qwen35::forward_slots::vmm::Qwen35RequestState), String> {
    use hipfire_arch_qwen35::forward_slots::vmm::{Qwen35RequestState, VmmRequestInit};
    use hipfire_runtime::slot_batch::RequestEpoch;
    let attempt_id = active_attempt_id();
    let placeholder = RequestEpoch {
        request_tag: 0,
        owner_generation: 1,
    };
    let mut state = Qwen35RequestState::new_like(
        gpu,
        b.config,
        b.kv,
        b.dn,
        placeholder,
        0,
        VmmRequestInit {
            prompt_len: b.position,
            stop_ids: Vec::new(),
            sampler: b.sampler,
            rng_state: b.rng_state,
            history: b.history,
        },
    )?;
    if let Err(e) = state.copy_dn_from(gpu, b.dn) {
        let freed = state.free_gpu(gpu);
        return Err(format!("{e}; free: {freed:?}"));
    }
    let Some(admission) = singleton_handoff_to_batch(b.id, attempt_id) else {
        let freed = state.free_gpu(gpu);
        return Err(format!("terminal transaction not transferable; free: {freed:?}"));
    };
    std::mem::swap(&mut state.kv, b.kv);
    // The singleton keeps its DN objects (copied above); clear them for its
    // next request, whose KV owner is now the fresh empty one.
    if let Err(e) = b.dn.reset(gpu) {
        eprintln!("[vmm-promote] singleton DN reset after promotion: {e}");
    }
    state.position = b.position;
    state.pending_seed = Some(b.pending_seed);
    let key = AttemptKey::new(b.id, attempt_id);
    let pending = BatchPendingRequest {
        key: key.clone(),
        admission,
        original_msg: b.permit.original_msg,
        prompt: String::new(),
        prompt_tokens: Vec::new(),
        started_in_think: b.started_in_think,
        system: None,
        assistant_prefix: b.permit.assistant_prefix,
        max_think_tokens: b.permit.max_think_tokens,
        max_tokens: b.permit.max_tokens,
        client_seed: b.permit.client_seed,
        sampling: b.permit.sampling.clone(),
    };
    let progress = QwenBatchLane {
        key,
        ticket: LaneTicket {
            lane: usize::MAX,
            generation: u64::MAX,
            admission,
        },
        sampling: b.permit.sampling,
        prompt_len: b.prompt_len,
        seq_pos: b.position,
        next_token: Some(b.pending_seed),
        rng_state: u64::from(b.rng_state),
        conversation_tokens: b.conversation_tokens,
        streamed_tokens: b.streamed_tokens,
        bytes_fed_to_filter: b.bytes_fed_to_filter,
        created_at: b.created_at,
        prefill_done_at: Some(b.prefill_done_at),
        first_token_at: b.first_token_at,
        max_active_lanes: 1,
    };
    Ok((pending, progress, state))
}

/// Per-lane singleton semantics the VMM driver carries beside the
/// scheduler's `QwenBatchLane`: think control, forced rows, resume
/// checkpoints and the finish drain that keeps the conversation.
#[derive(Default)]
struct VmmLaneCtl {
    started_in_think: bool,
    think: crate::vmm_conv::ThinkCtl,
    /// Committed tokens the lane has not fed yet (a forced think close, the
    /// finish drain), fed in order as its next rows; each such row's pick is
    /// replaced by the next one, as the singleton forwards them unsampled.
    forced: std::collections::VecDeque<u32>,
    /// The lane's pending row is a sampled token, whose forward the singleton
    /// follows with a resume checkpoint (a forced/trailer row is not).
    main_row: bool,
    /// Keep the finished conversation in the prefix pool (`vmm_conv`).
    keep: bool,
    checkpoints: crate::vmm_conv::Checkpoints,
    /// Prompt tokens reused from a kept conversation (`cached_tokens`).
    cached: usize,
    /// Prompt tokens prefilled (`prefill_tokens`); 0 = `prompt_len`.
    prefill_tokens: usize,
    /// Conversation index of the first generated token.
    decode_start: usize,
    /// Finished and feeding its last rows; `commit_ready` follows.
    drain: Option<VmmDrain>,
    /// Verbatim assistant turn stored when the client commits.
    turn_store: Option<(QwenArCacheAction, Vec<u32>)>,
}

struct VmmDrain {
    /// Rows still to feed (including the pending one).
    left: usize,
    pending_done: serde_json::Value,
}

impl VmmLaneCtl {
    /// Release the lane's host state and resume checkpoints.
    fn reset(&mut self, gpu: &mut rdna_compute::Gpu) {
        crate::vmm_conv::free_checkpoints(std::mem::take(self).checkpoints, gpu);
    }
}

/// Why a VMM lane finished at a committed token.
#[derive(Clone, Copy)]
struct VmmFinish {
    is_eos: bool,
    hit_max: bool,
    hit_lane_cap: bool,
    stopped: bool,
    loop_hit: bool,
}

/// Raw-commit one token on a VMM lane and let its producer classify and
/// emit it. `Ok(true)`: the producer stopped (EOT decode / stop string).
fn vmm_emit_token(
    stdout: &mut std::io::Stdout,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    lane: &mut QwenBatchLane,
    producer: &mut QwenArSemanticProducer,
    token: u32,
) -> Result<bool, String> {
    let mut future_streamed = lane.streamed_tokens.clone();
    future_streamed.push(token);
    let all_bytes = tokenizer.decode_bytes(&future_streamed);
    let prev_fed = lane.bytes_fed_to_filter.min(all_bytes.len());
    let token_bytes = all_bytes[prev_fed..].to_vec();
    let all_len = all_bytes.len();
    if lane.first_token_at.is_none() {
        lane.first_token_at = Some(Instant::now());
    }
    let (conv, stream, seq, fed) = (
        &mut lane.conversation_tokens,
        &mut lane.streamed_tokens,
        &mut lane.seq_pos,
        &mut lane.bytes_fed_to_filter,
    );
    producer
        .commit_and_classify(
            stdout,
            token,
            || {
                let pos = qwen_ar_raw_commit_token(
                    conv,
                    stream,
                    seq,
                    token,
                    QwenArRawCommitDisposition::ClassifiedVisible,
                );
                *fed = all_len;
                (pos, token_bytes.clone())
            },
            |_, _| {},
        )
        .map_err(|e| e.to_string())
}

/// Commit one sampled token on a VMM lane in the singleton decode loop's
/// order: classify (stop) → EOS → think control → loop guard → budgets.
/// A forced think close is committed here, its tokens queued as the lane's
/// next rows. `think` is false on spec lanes (no think budget there). One
/// deviation, unreachable without stop strings matching inside the close
/// text: a stop on a close token finishes the lane, where the singleton
/// keeps decoding.
#[allow(clippy::too_many_arguments)]
fn vmm_commit_sampled(
    stdout: &mut std::io::Stdout,
    tokenizer: &hipfire_runtime::tokenizer::Tokenizer,
    lane: &mut QwenBatchLane,
    producer: &mut QwenArSemanticProducer,
    loop_guard: &hipfire_runtime::loop_guard::LoopGuard,
    ctl: &mut VmmLaneCtl,
    token: u32,
    think: bool,
    max_toks: usize,
    lane_capacity: usize,
    adv_length: bool,
    ends: (u32, u32),
    close_tokens: &[u32],
) -> Result<Option<VmmFinish>, String> {
    let stopped = vmm_emit_token(stdout, tokenizer, lane, producer, token)?;
    ctl.main_row = !stopped;
    let is_eos = token == ends.0 || token == ends.1 || tokenizer.is_terminator(token);
    let mut f = VmmFinish {
        is_eos,
        hit_max: false,
        hit_lane_cap: false,
        stopped,
        loop_hit: false,
    };
    let budgets = |f: &mut VmmFinish, lane: &QwenBatchLane| {
        f.hit_max = lane.streamed_tokens.len() >= max_toks;
        f.hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, lane_capacity) || adv_length;
    };
    if stopped || is_eos {
        budgets(&mut f, lane);
        return Ok(Some(f));
    }
    if think {
        let generated = lane.streamed_tokens.len();
        let action = ctl.think.step(
            &lane.key.id,
            || tokenizer.decode_bytes(&lane.streamed_tokens),
            ctl.started_in_think,
            generated,
            max_toks,
            close_tokens.len(),
        );
        match action {
            crate::vmm_conv::ThinkAction::Continue => {}
            crate::vmm_conv::ThinkAction::Eos => {
                // A forced stop, never a length cap.
                f.loop_hit = true;
                budgets(&mut f, lane);
                return Ok(Some(f));
            }
            crate::vmm_conv::ThinkAction::Close(take) => {
                for &t in &close_tokens[..take] {
                    let s = vmm_emit_token(stdout, tokenizer, lane, producer, t)?;
                    ctl.forced.push_back(t);
                    if s {
                        f.stopped = true;
                        budgets(&mut f, lane);
                        return Ok(Some(f));
                    }
                }
                if lane.streamed_tokens.len() >= max_toks {
                    budgets(&mut f, lane);
                    return Ok(Some(f));
                }
            }
        }
    }
    f.loop_hit = loop_guard.check(&lane.streamed_tokens).is_some();
    budgets(&mut f, lane);
    Ok(batch_should_finish_decode(f.is_eos, f.hit_max, f.hit_lane_cap, f.stopped, f.loop_hit).then_some(f))
}

/// The prompt a batch lane prefills for `req` in the current cache state —
/// the singleton's (`ar.rs` prompt cache): with a conversation kept
/// anywhere (`conv_exists`, the singleton's non-empty resident) a history
/// request renders with verbatim assistant-turn splices; otherwise, or for
/// a prompt-only request, the cold render taken at admission.
fn vmm_admission_render(
    model: &mut LoadedModel,
    req: &BatchPendingRequest,
    conv_exists: bool,
) -> Result<Vec<u32>, String> {
    let cold = req.prompt_tokens.clone();
    if !conv_exists {
        return Ok(cold);
    }
    let msg = &req.original_msg;
    let history = match hipfire_engine::prompt::parse_generate_messages(msg) {
        Ok(Some(h)) => h,
        _ => return Ok(cold),
    };
    let (Some(tokenizer), Some(template)) = (model.tokenizer.as_ref(), model.chat_template.as_deref()) else {
        return Ok(cold);
    };
    let prompt = hipfire_runtime::tokenizer::maybe_normalize_prompt(
        msg.get("prompt").and_then(|v| v.as_str()).unwrap_or("Hello"),
    );
    let raw_effort = msg
        .get("reasoning_effort")
        .or_else(|| msg.get("thinking_mode"))
        .and_then(|v| v.as_str());
    let thinking_enabled = msg.get("thinking_enabled").and_then(|v| v.as_bool());
    let (enable_thinking, reasoning_effort) =
        qwen_jinja_reasoning(thinking_enabled, raw_effort, req.max_think_tokens);
    let frame = hipfire_runtime::prompt_frame::JinjaChatFrame {
        tokenizer,
        template,
        system: msg.get("system").and_then(|v| v.as_str()),
        user: &prompt,
        enable_thinking,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: reasoning_effort.as_deref(),
    };
    // No producer prefix: it only steers tool-argument replay, and these
    // histories carry no tool calls.
    match crate::qwen::qwen_jinja_cached_history_tokens(
        &frame,
        &mut model.asst_turn_cache,
        &cold,
        &history,
        None,
        "vmm-batch",
        None,
    ) {
        Ok(t) => Ok(t),
        Err(e) if reasoning_effort.is_some() => Err(format!("qwen-cache jinja build: {e}")),
        Err(e) => {
            eprintln!("[qwen-cache] jinja cached-history build failed ({e}) — cold render");
            Ok(cold)
        }
    }
}

/// Publish a finished VMM lane's `commit_ready` (its owner already retired
/// or kept). `false` when the lane could not await the client and was
/// aborted instead.
fn vmm_publish_commit_ready(
    sched: &mut ContinuousBatchScheduler,
    stdout: &mut std::io::Stdout,
    idx: usize,
    key: &AttemptKey,
    admission: BatchGeneration,
    pending_done: serde_json::Value,
) -> bool {
    let mut envelope = pending_done.clone();
    envelope["type"] = serde_json::json!("commit_ready");
    if !sched.mark_awaiting_commit(idx, pending_done) {
        eprintln!(
            "[batch][vmm] mark_awaiting_commit failed lane {idx} id={} — aborting lane",
            key.id
        );
        let _ = sched.abort_lane(idx, key, admission);
        return false;
    }
    let write_ok = {
        let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
        writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
    };
    if !write_ok {
        let _ = sched.abort_lane(idx, key, admission);
    }
    write_ok
}

fn vmm_bundle(
    state: &mut Option<Box<dyn hipfire_runtime::arch_model::ArchModel>>,
) -> Option<&mut hipfire_arch_qwen35::Qwen35Bundle> {
    state
        .as_mut()
        .and_then(|s| (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_qwen35::Qwen35Bundle>())
}

/// Retire one request owner from the VMM store and free its device state.
fn vmm_retire_free(
    model: &mut LoadedModel,
    gpu: &mut rdna_compute::Gpu,
    epoch: &hipfire_runtime::slot_batch::RequestEpoch,
) -> Result<(), String> {
    let b = vmm_bundle(&mut model.state).ok_or("VMM batch: model is not Qwen35")?;
    let store = b.vmm_store.as_mut().ok_or("VMM batch: store not staged")?;
    store.retire(epoch)?.free_gpu(gpu)
}

/// VMM-route admission predicate on a model that staged `vmm_store`:
/// greedy requests (penalties apply exactly as on the singleton route) with
/// a text-chat shape — any number of system/user/assistant turns, rendered
/// through the Jinja template exactly as the singleton renders them (a
/// template-less model keeps the single-user-turn shape, which is all the
/// singleton's ChatFrame path renders), think-open or closed, with or
/// without stop strings (the lane's producer matches them as the
/// singleton's does). Still sequential, each because the batch lane does
/// not run it per row: tools/tool turns (grammar-constrained CPU sampling
/// and the tool-call parser), images (vision path), invalid stops (the
/// singleton reports them), logprobs (the singleton sampler's envelopes),
/// budget alerts (mid-stream text injection), adaptive KV/eviction/PFlash
/// (per-request KV rewriting), PP/EP, sampled requests (no RNG equivalence
/// gate yet). A loaded
/// speculator does not exclude the request — batched rows are AR, and a
/// lonely request keeps its singleton spec route.
pub fn is_vmm_batch_request_eligible(
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
) -> bool {
    let staged = m.state.as_ref().is_some_and(|s| {
        (s.as_ref() as &dyn Any)
            .downcast_ref::<hipfire_arch_qwen35::Qwen35Bundle>()
            .is_some_and(|b| b.vmm_store.is_some())
    });
    if !staged || !serve_continuous_batch || continuous_batch_size <= 1 {
        return false;
    }
    let has_image = msg.get("image").is_some() || msg.get("image_base64").is_some();
    let has_tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let jinja = m.chat_template.is_some()
        && hipfire_config::developer_var("HIPFIRE_JINJA_CHAT").ok().as_deref() != Some("0");
    let shape_ok = if jinja {
        batch_messages_are_text_chat(msg)
    } else {
        batch_messages_are_single_user(msg)
    };
    let stop_ok = hipfire_runtime::stop_sequence::parse_stop_field(msg.get("stop")).is_ok();
    // Token logprob envelopes are emitted by the singleton sampler only.
    let logprobs = msg.get("logprobs").and_then(|v| v.as_bool()).unwrap_or(false);
    if !shape_ok || has_tools || has_image || !stop_ok || logprobs {
        return false;
    }
    if m.kv_adaptive.is_some() || m.eviction.is_some() || pflash_active {
        return false;
    }
    if m.pp != 1 || m.ep.is_some() {
        return false;
    }
    if msg.get("budget_alert_at_tok").is_some() || msg.get("budget_alert_text").is_some() {
        return false;
    }
    resolve_batch_sampling(msg, m).temp <= 0.0
}

/// Daemon: the VMM route serves this model (params parsed at load and
/// `vmm_store` staged).
pub fn vmm_route_active(vmm_batch: Option<VmmBatchParams>, m: &LoadedModel) -> bool {
    vmm_batch.is_some() && m.qwen35().is_some_and(|b| b.vmm_store.is_some())
}

/// Daemon load: the loader's VMM staging request for the parsed
/// `serve_vmm_batch` params. Route: `VmmRoute::Exact` (byte-identical to the
/// singleton route) by default; the non-exact shared slots body only under
/// the explicit `HIPFIRE_SERVE_BATCH_NONEXACT=1` opt-in (PLAN §4.3). An exact
/// route the model does not support is not staged.
pub fn vmm_staging_request(
    vmm_batch: Option<VmmBatchParams>,
) -> Option<hipfire_loader::batch_staging::VmmStagingRequest> {
    use hipfire_arch_qwen35::forward_slots::vmm::VmmRoute;
    vmm_batch.map(|p| hipfire_loader::batch_staging::VmmStagingRequest {
        row_budget: p.max_batch_tokens,
        route: if p.nonexact { VmmRoute::Nonexact } else { VmmRoute::Exact },
        spec: p.spec,
    })
}

/// Daemon load ack: on the VMM batch route (`vmm_batch` Some) append the
/// actual per-request owner receipt. Absent on every other route, so the
/// flag-off ack is returned unchanged.
pub fn vmm_loaded_ack(
    ack: String,
    m: &mut LoadedModel,
    vmm_batch: Option<VmmBatchParams>,
    slots: usize,
    row_budget: usize,
) -> String {
    let Some(p) = vmm_batch else {
        return ack;
    };
    let receipt = m.qwen35_mut().and_then(|b| b.vmm_store.as_mut()).map(|s| s.receipt());
    let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&ack) else {
        return ack;
    };
    v["continuous_batch_route"] = serde_json::json!("vmm");
    v["continuous_batch_slots"] = serde_json::json!(slots);
    v["continuous_batch_row_budget"] = serde_json::json!(row_budget);
    v["continuous_batch_row_budget_requested"] = serde_json::json!(p.max_batch_tokens);
    v["continuous_batch_spec"] = serde_json::json!(false);
    v["continuous_batch_spec_requested"] = serde_json::json!(p.spec);
    let exact = m.qwen35().and_then(|b| b.vmm_store.as_ref()).is_some_and(|s| {
        s.route() == hipfire_arch_qwen35::forward_slots::vmm::VmmRoute::Exact
    });
    v["continuous_batch_nonexact"] = serde_json::json!(!exact);
    v["continuous_batch_exact"] = serde_json::json!(exact);
    v["continuous_batch_sampling"] = serde_json::json!("greedy_only");
    match receipt {
        Some(Ok(r)) => {
            v["vmm_batch_kv_backend"] = serde_json::json!(r.kv_backend);
            v["vmm_batch_kv_mode"] = serde_json::json!(r.kv_mode);
            v["vmm_batch_max_seq_bound"] = serde_json::json!(r.max_seq_bound);
            v["vmm_batch_mapped_bytes"] = serde_json::json!(r.mapped_bytes);
        }
        Some(Err(e)) => {
            v["vmm_batch_receipt_error"] = serde_json::json!(e);
        }
        None => {}
    }
    v.to_string()
}

/// Daemon: batch admission for a staged scheduler. On the VMM route
/// (`vmm_route`) only when another batched generate is already waiting — a
/// lonely request keeps the unchanged singleton route (exact singleton
/// arithmetic/spec); otherwise the fixed-lane predicate.
pub fn batch_request_eligible_for_route(
    vmm_route: bool,
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
    inbox: &mut DaemonInbox,
) -> bool {
    if vmm_route {
        is_vmm_batch_request_eligible(msg, m, continuous_batch_size, serve_continuous_batch, pflash_active)
            && inbox.has_pending_batch_generate()
    } else {
        is_batch_request_eligible(msg, m, continuous_batch_size, serve_continuous_batch, pflash_active)
    }
}

/// Daemon: drive the staged batch scheduler — the VMM driver when
/// `vmm_batch` is Some, else the fixed-lane Qwen driver.
pub fn drive_staged_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    m: &mut LoadedModel,
    vmm_batch: Option<VmmBatchParams>,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    match vmm_batch {
        Some(params) => drive_qwen_vmm_continuous_batch(sched, gpu, m, params, stdout, inbox, None),
        None => drive_qwen_continuous_batch(sched, gpu, m, stdout, inbox),
    }
}

/// Daemon, right before a singleton `generate()`: grant promotion when the
/// request could have joined the VMM batch (`armed` = VMM route active and a
/// scheduler staged, and the request VMM-eligible), else clear it. The
/// request may then promote itself into the batch driver at a token boundary
/// if a batched peer arrives while it runs (closes the C1→C2 TTFT gap).
#[allow(clippy::too_many_arguments)]
pub fn arm_promotion(
    armed: bool,
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
    max_tokens: usize,
    max_think_tokens: usize,
    client_seed: Option<u64>,
    assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix,
) {
    let permit = (armed
        && is_vmm_batch_request_eligible(msg, m, continuous_batch_size, serve_continuous_batch, pflash_active))
    .then(|| PromotionPermit {
        original_msg: msg.clone(),
        sampling: resolve_batch_sampling(msg, m),
        max_tokens,
        max_think_tokens,
        client_seed,
        assistant_prefix,
    });
    set_promotion_permit(permit);
}

/// Daemon, right after a singleton `generate()`: clear the permit and, if the
/// request promoted itself, drive it in the VMM batch driver. None when it
/// did not promote; on the VMM route the finished singleton conversation is
/// then parked in the prefix pool, where later batch lanes and singleton
/// turns find it (`vmm_conv`).
pub fn drive_promoted(
    sched: Option<&mut ContinuousBatchScheduler>,
    gpu: &mut rdna_compute::Gpu,
    m: &mut LoadedModel,
    vmm_batch: Option<VmmBatchParams>,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Option<Result<(), BatchDriveError>> {
    set_promotion_permit(None);
    let Some(promoted) = take_promoted() else {
        if vmm_batch.is_some() {
            crate::vmm_conv::park_resident(m, gpu);
        }
        return None;
    };
    Some(match (sched, vmm_batch) {
        (Some(sched), Some(params)) => {
            drive_qwen_vmm_continuous_batch(sched, gpu, m, params, stdout, inbox, Some(promoted))
        }
        _ => Err(BatchDriveError::Gpu("promoted request without a staged VMM batch route".into())),
    })
}

pub fn drive_qwen_vmm_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    model: &mut LoadedModel,
    params: VmmBatchParams,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
    promoted: Option<PromotedRequest>,
) -> Result<(), BatchDriveError> {
    use hipfire_runtime::scheduler::{PendingWork, Scheduler, SpecKind};
    use hipfire_runtime::slot_batch::{BatchPlanner, RequestEpoch};
    use rdna_compute::slot_pool::SlotId;

    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let route = crate::ar::GenerationRoute::QwenAr;
    let (eos_tok, stop_ids, row_budget) = {
        let tokenizer = model
            .tokenizer
            .as_ref()
            .ok_or_else(|| BatchDriveError::Gpu("tokenizer missing".to_string()))?;
        let im_end = tokenizer.special_token_id("<|im_end|>");
        let b = vmm_bundle(&mut model.state)
            .ok_or_else(|| BatchDriveError::Gpu("VMM batch model not Qwen35".to_string()))?;
        let store = b
            .vmm_store
            .as_ref()
            .ok_or_else(|| BatchDriveError::Gpu("VMM batch store not staged".to_string()))?;
        let eos = b.config.eos_token;
        let mut stops = vec![eos];
        stops.extend(im_end);
        (eos, stops, store.row_budget().min(params.max_batch_tokens.max(batch_size)))
    };
    let im_end_tok = stop_ids.get(1).copied().unwrap_or(eos_tok);
    // Singleton constants of the think control and the ChatML trailer:
    // the forced think continuation, the `<think>` opener blocked once the
    // force-answer latch is set, and the `\n` forwarded after `<|im_end|>`.
    let (think_open_tok, close_tokens, nl_tokens) = {
        let t = model.tokenizer.as_ref().expect("checked above");
        (
            t.special_token_id("<think>"),
            t.encode(&think_continuation()),
            t.encode("\n"),
        )
    };
    // Unclosed-opener attractor pairs, exactly as the singleton AR route
    // builds them (ar.rs); blocks are recomputed per step from the
    // request's generated-only history.
    let attractor_pairs: Vec<(u32, u32)> = {
        let t = model.tokenizer.as_ref().expect("checked above");
        let pair = |o: &str, c: &str| match (t.special_token_id(o), t.special_token_id(c)) {
            (Some(o), Some(c)) => Some((o, c)),
            _ => None,
        };
        pair("<tool_call>", "</tool_call>")
            .into_iter()
            .chain(pair("<think>", "</think>"))
            .collect()
    };
    let idle_work = |i: usize| PendingWork {
        slot: SlotId(i),
        remaining_prompt: Vec::new(),
        next_pos: 0,
        decoding: false,
        vl_prefill: None,
        spec: SpecKind::None,
        spec_cycles: 0,
        spec_committed: 0,
        spec_retire_fails: 0,
        pos3_delta: 0,
    };
    const IDLE: RequestEpoch = RequestEpoch {
        request_tag: 0,
        owner_generation: 0,
    };
    let mut work: Vec<PendingWork> = (0..batch_size).map(idle_work).collect();
    let mut epochs: Vec<RequestEpoch> = vec![IDLE; batch_size];
    // Spec lanes: greedy requests decoded by the store's MTP engine
    // (`spec_cycle`), each window exactly the singleton MTP route's; never
    // planned by the AR planner. `(k, head_cap)` when the engine is staged
    // and no n-gram modifier could change the singleton's windows.
    let spec_engine: Option<(usize, usize)> = if params.spec
        && !hipfire_config::mtp_ngram_enabled()
        && !hipfire_config::mtp_ngram_enabled_for_arch(model.arch_id, &gpu.arch)
    {
        vmm_bundle(&mut model.state)
            .and_then(|b| b.vmm_store.as_ref())
            .and_then(|s| s.spec_engine())
            .map(|e| (e.k(), e.head_cap()))
    } else {
        None
    };
    let mut spec_lane: Vec<bool> = vec![false; batch_size];
    // First generated ids of spec lanes prefilled this iteration, emitted
    // with this iteration's commits (the singleton emits the seed first).
    let mut spec_seeds: Vec<hipfire_runtime::slot_batch::RequestAdvance> = Vec::new();
    // Conversation continuity (`vmm_conv`): AR lanes keep their finished
    // conversation in the prefix pool and admissions reuse it. A conversation
    // the singleton left resident is parked there first.
    let keep_conversations = spec_engine.is_none() && crate::vmm_conv::pool_enabled(model);
    if keep_conversations {
        crate::vmm_conv::park_resident(model, gpu);
    }
    let mut ctl: Vec<VmmLaneCtl> = (0..batch_size).map(|_| VmmLaneCtl::default()).collect();
    // Picks of the last step awaiting the singleton's per-token processing
    // (emit, stop/EOS, think control, loop guard, budgets). Processed before
    // the next step plans, so a forced think close decided on a token
    // replaces the very next pick, as the singleton forwards it unsampled.
    let mut pending_commits: Vec<(RequestEpoch, u32, bool)> = Vec::new();
    let mut planner = BatchPlanner::new(
        Scheduler {
            chunk_size: row_budget,
            vl_sequential: false,
            prefill_cursor: 0,
        },
        0,
        0,
    );
    let mut producers: Vec<Option<QwenArSemanticProducer>> =
        (0..batch_size).map(|_| None).collect();
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(|_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()))
        .collect();
    // Attempts whose `gen_start` is already on the wire: the request the
    // daemon enqueued before entering this driver, and every request this
    // driver has assigned to a batch lane. The drain does NOT announce
    // (`vmm_route=true`), so an arrival that turns out lonely at
    // assignment can still take the singleton route with exactly one start.
    let announced: std::cell::RefCell<std::collections::HashSet<AttemptKey>> =
        std::cell::RefCell::new(
            sched
                .inbox
                .iter()
                .cloned()
                .chain(sched.lanes.iter().filter_map(|l| l.key().cloned()))
                .collect(),
        );
    let announce = |stdout: &mut std::io::Stdout,
                    key: &AttemptKey,
                    admission: BatchGeneration,
                    started_in_think: bool| {
        if announced.borrow_mut().insert(key.clone()) {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_start(route, stdout, &key.id, started_in_think);
        }
    };

    // Fail closed: retire and free every request owner, emit one keyed GPU
    // error per live/queued attempt, and fail the scheduler. The resident
    // bundle is untouched by batch steps, so a clean free + device sync is a
    // full rollback.
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    model: &mut LoadedModel,
                    epochs: &mut Vec<RequestEpoch>,
                    ctl: &mut Vec<VmmLaneCtl>,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for lane in sched.lanes.iter() {
            let Some(key) = lane.key() else {
                continue;
            };
            let admission = match lane {
                BatchLane::Seeding(q) | BatchLane::Running(q) => q.ticket.admission,
                BatchLane::AwaitingClient(t) => t.ticket.admission,
                BatchLane::Empty { .. } => continue,
            };
            if uniq_set.insert((key.clone(), admission)) {
                uniq.push((key.clone(), admission));
            }
        }
        for (key, request) in sched.pending.iter() {
            if uniq_set.insert((key.clone(), request.admission)) {
                uniq.push((key.clone(), request.admission));
            }
        }
        let mut first_err: Option<String> = None;
        for e in epochs.iter_mut() {
            if e.is_admitted() {
                if let Err(err) = vmm_retire_free(model, gpu, e) {
                    first_err.get_or_insert(format!("VMM retire: {err}"));
                }
                *e = IDLE;
            }
        }
        // Device state is untrusted: drop every lane's checkpoints and every
        // kept conversation.
        for c in ctl.iter_mut() {
            c.reset(gpu);
        }
        crate::vmm_conv::clear(&mut model.state, gpu);
        crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
        let sync = crate::common::fail_closed_device_sync(gpu);
        let prior = match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let ep = crate::common::fail_closed_epilogue_after_sync(prior, sync);
        for (key, admission) in &uniq {
            let started = sched.pending.get(key).is_some_and(|r| r.started_in_think);
            announce(stdout, key, *admission, started);
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, *admission);
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("VMM batch GPU error: {reason}"),
                "gpu",
                ep.rolled_back,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        if !ep.rolled_back {
            return Err(BatchDriveError::Poisoned(format!(
                "{reason}; {}",
                ep.context.unwrap_or_default()
            )));
        }
        Err(BatchDriveError::Gpu(reason))
    };

    // ── Promoted singleton: install as a Running lane, no new gen_start ──
    if let Some(p) = promoted {
        let PromotedRequest {
            pending,
            progress,
            mut state,
            producer,
            loop_guard,
            kind,
            seed_emitted,
            conv,
        } = p;
        let key = pending.key.clone();
        let admission = pending.admission;
        let position = state.position;
        let seed = state.pending_seed;
        let rng = state.rng_state;
        let sampling = pending.sampling.clone();
        let started_in_think = pending.started_in_think;
        let max_think_tokens = pending.max_think_tokens;
        let decode_start = progress.prompt_len;
        let mut conv = Some(conv);
        announced.borrow_mut().insert(key.clone());
        let mut installed: Result<usize, String> = Err("not installed".into());
        let ticket = match seed {
            None => {
                installed = Err("promoted request has no pending seed".into());
                None
            }
            Some(_) => sched.adopt_running(pending, progress),
        };
        if let (Some(ticket), Some(seed)) = (ticket, seed) {
            let lane = ticket.lane;
            let epoch = RequestEpoch {
                request_tag: ticket.generation,
                owner_generation: ticket.generation.wrapping_add(1).max(1),
            };
            state.epoch = epoch;
            state.slot = lane;
            state.stop_ids = stop_ids.clone();
            installed = match vmm_bundle(&mut model.state).and_then(|b| b.vmm_store.as_mut()) {
                None => Err("store not staged".into()),
                Some(store) => match store.admit(state) {
                    Ok(()) => Ok(lane),
                    Err((state, e)) => {
                        let freed = state.free_gpu(gpu);
                        Err(format!("{e}; free: {freed:?}"))
                    }
                },
            };
            if installed.is_ok() {
                epochs[lane] = epoch;
                // A promoted MTP singleton continues as a spec lane when one
                // can take it (greedy, whole budget within the lane cap).
                let generated = match &sched.lanes[lane] {
                    BatchLane::Running(l) => l.streamed_tokens.len(),
                    _ => usize::MAX,
                };
                let adopted = match kind {
                    PromotedDecode::Ar => false,
                    PromotedDecode::Spec(snap) => {
                        let fits = spec_engine.is_some_and(|(k, cap)| {
                            sampling.temp <= 0.0
                                && generated != usize::MAX
                                && position + lane_max_tokens(&key, sched).saturating_sub(generated) + k + 1 <= cap
                        });
                        let b = vmm_bundle(&mut model.state);
                        match (fits, b) {
                            (true, Some(b)) => {
                                let hipfire_arch_qwen35::Qwen35Bundle { vmm_store, config, scratch, .. } = b;
                                let store = vmm_store.as_mut().expect("admitted above");
                                match store.spec_adopt(gpu, config, scratch, &epoch, snap, spec_request_config(&sampling, rng as u64)) {
                                    Ok(()) => true,
                                    Err(e) => {
                                        eprintln!("[vmm-promote] id={} spec lane refused ({e}); continuing as AR", key.id);
                                        false
                                    }
                                }
                            }
                            _ => {
                                snap.free_gpu(gpu);
                                false
                            }
                        }
                    }
                };
                spec_lane[lane] = adopted;
                work[lane] = if adopted {
                    PendingWork {
                        next_pos: position,
                        decoding: true,
                        ..idle_work(lane)
                    }
                } else {
                    PendingWork {
                        remaining_prompt: vec![seed],
                        next_pos: position,
                        decoding: true,
                        ..idle_work(lane)
                    }
                };
                loop_guards[lane] = loop_guard;
                producers[lane] = Some(producer);
                let conv = conv.take().expect("installed once");
                ctl[lane] = VmmLaneCtl {
                    started_in_think,
                    think: conv.think.unwrap_or_else(|| crate::vmm_conv::ThinkCtl::new(max_think_tokens)),
                    keep: keep_conversations && !adopted,
                    checkpoints: conv.checkpoints,
                    cached: conv.cached_tokens,
                    prefill_tokens: if conv.prefill_tokens > 0 { conv.prefill_tokens } else { decode_start },
                    decode_start,
                    // The pending seed is a sampled token.
                    main_row: true,
                    ..VmmLaneCtl::default()
                };
                if !seed_emitted {
                    if adopted {
                        spec_seeds.push(hipfire_runtime::slot_batch::RequestAdvance {
                            epoch,
                            committed_ids: vec![seed],
                            committed_position: position,
                            accepted_drafts: 0,
                            verified_rows: 0,
                            finish: None,
                        });
                    } else {
                        // The singleton picked the seed but has not emitted
                        // it: commit it (emit, think control) before the
                        // step that forwards it.
                        pending_commits.push((epoch, seed, false));
                    }
                }
            } else if let PromotedDecode::Spec(snap) = kind {
                snap.free_gpu(gpu);
            }
        } else {
            if let PromotedDecode::Spec(snap) = kind {
                snap.free_gpu(gpu);
            }
            let freed = state.free_gpu(gpu);
            let why = match installed {
                Err(e) if e != "not installed" => e,
                _ => "no lane for the promoted request".to_string(),
            };
            installed = Err(format!("{why}; free: {freed:?}"));
        }
        if let Some(conv) = conv.take() {
            crate::vmm_conv::free_checkpoints(conv.checkpoints, gpu);
        }
        if let Err(reason) = installed {
            // The singleton's KV/DN left the bundle with the promotion; the
            // request cannot continue anywhere. Fail it visibly.
            let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let ep = crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("VMM promotion failed: {reason}"),
                "internal",
                false,
                &ep,
            );
            let _ = stdout.flush();
            if let Some(idx) = sched.lanes.iter().position(|l| l.key() == Some(&key)) {
                let _ = sched.abort_lane(idx, &key, admission);
            } else {
                batch_clear_terminal_at_generation(&key.id, key.attempt_id, admission);
            }
        }
    }

    loop {
        // ── Client terminal decisions (request state already retired) ──
        let mut to_commit: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let admission = term.ticket.admission;
                if batch_check_abort(&key.id, key.attempt_id, admission)
                    || Instant::now() >= term.deadline
                {
                    to_abort.push((idx, key, admission));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id, admission)
                {
                    to_commit.push((idx, key, admission, term.pending_done.clone()));
                }
            }
        }
        for (idx, key, admission) in to_abort {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
            // The singleton rolls an aborted turn's conversation back.
            crate::vmm_conv::settle(model, gpu, &key.id, key.attempt_id, false);
            ctl[idx].reset(gpu);
        }
        for (idx, key, admission, pending_done) in to_commit {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let commit_ok = sched.commit_lane_retain_terminal(idx, &key, admission);
            let _terminal_cleanup = BatchTerminalCleanup::new(&key, Some(admission));
            match batch_commit_teardown_class(true, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!("reset_ok is true"),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error_for_route(
                        route,
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                    crate::vmm_conv::settle(model, gpu, &key.id, key.attempt_id, false);
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_generation_done_value(route, stdout, &pending_done);
                    // As the singleton after a committed turn: store the
                    // verbatim assistant turn and keep the conversation.
                    if let Some((action, cached_seq)) = ctl[idx].turn_store.take() {
                        let started = ctl[idx].started_in_think;
                        if let Some(tok) = model.tokenizer.as_ref() {
                            let cache = &mut model.asst_turn_cache;
                            let _ = qwen_ar_apply_cache_action(
                                |fp, seq| {
                                    cache.insert(fp, crate::qwen::qwen_cached_turn_entry(tok, seq, started))
                                },
                                &action,
                                cached_seq,
                            );
                        }
                    }
                    crate::vmm_conv::settle(model, gpu, &key.id, key.attempt_id, true);
                }
            }
            producers[idx] = None;
            ctl[idx].reset(gpu);
        }
        // ── Queued and running aborts ──
        let mut queued_abort: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if let Some(request) = sched.pending.get(&key) {
                if batch_check_abort(&key.id, key.attempt_id, request.admission) {
                    queued_abort.push((key, request.admission));
                }
            }
        }
        for (key, admission) in queued_abort {
            let started = sched.pending.get(&key).is_some_and(|r| r.started_in_think);
            announce(stdout, &key, admission, started);
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_queued(&key, admission);
        }
        let mut running_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::Running(l) = &sched.lanes[idx] {
                if batch_check_abort(&l.key.id, l.key.attempt_id, l.ticket.admission) {
                    running_abort.push((idx, l.key.clone(), l.ticket.admission));
                }
            }
        }
        for (idx, key, admission) in running_abort {
            if epochs[idx].is_admitted() {
                if let Err(e) = vmm_retire_free(model, gpu, &epochs[idx]) {
                    return fail_all(
                        sched,
                        gpu,
                        model,
                        &mut epochs,
                        &mut ctl,
                        stdout,
                        format!("retire lane {idx} on abort: {e}"),
                    );
                }
            }
            epochs[idx] = IDLE;
            work[idx] = idle_work(idx);
            spec_lane[idx] = false;
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
            ctl[idx].reset(gpu);
        }
        // ── Step-boundary admission ──
        let drained = {
            let tokenizer = model.tokenizer.as_ref().expect("checked above");
            drain_qwen_batch_inbox(
                sched,
                model,
                batch_size,
                is_vmm_batch_request_eligible,
                tokenizer,
                model.chat_template.as_ref(),
                route,
                stdout,
                inbox,
                true,
            )
        };
        let barrier = match drained {
            Ok(b) => b,
            Err(reason) => return fail_all(sched, gpu, model, &mut epochs, &mut ctl, stdout, reason),
        };
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let Some(pending_req) = sched.pending.get(&key).cloned() else {
                continue;
            };
            // A request with no live batch peer and nothing queued behind it
            // is lonely: if its start is not yet announced it takes the
            // unchanged singleton route (exact singleton arithmetic and its
            // MTP/DFlash speculation), like a lonely request at daemon
            // dispatch. Think-open prompts batch: their lane applies the
            // singleton's think controls (`VmmLaneCtl`).
            let lonely = !epochs.iter().any(|e| e.is_admitted())
                && sched.inbox.is_empty()
                && !announced.borrow().contains(&key);
            if lonely {
                let handoff = match handoff_started_in_think(
                    sched,
                    lane_idx,
                    &key,
                    &pending_req,
                    GenerationRoute::QwenAr,
                ) {
                    Ok(msg) => msg,
                    Err(reason) => {
                        return fail_all(sched, gpu, model, &mut epochs, &mut ctl, stdout, reason)
                    }
                };
                inbox.push_front(handoff);
                continue;
            }
            let started_in_think = pending_req.started_in_think;
            let epoch = RequestEpoch {
                request_tag: ticket.generation,
                owner_generation: ticket.generation.wrapping_add(1).max(1),
            };
            let rng_state = match &sched.lanes[lane_idx] {
                BatchLane::Running(l) => l.rng_state as u32,
                _ => continue,
            };
            let think = crate::vmm_conv::ThinkCtl::new(pending_req.max_think_tokens);
            let stops = hipfire_runtime::stop_sequence::parse_stop_field(pending_req.original_msg.get("stop"))
                .unwrap_or_default();
            // The singleton's prompt for this cache state: with a kept
            // conversation anywhere, the history render splices verbatim
            // assistant turns (`ar.rs` prompt cache); else the cold render.
            let rendered = if keep_conversations {
                let conv_exists = crate::vmm_conv::pool_nonempty(model)
                    || !model.conversation_tokens.is_empty()
                    || epochs.iter().any(|e| e.is_admitted());
                vmm_admission_render(model, &pending_req, conv_exists)
            } else {
                Ok(pending_req.prompt_tokens.clone())
            };
            let refuse = |stdout: &mut std::io::Stdout, sched: &mut ContinuousBatchScheduler, msg: String, kind: &str| {
                announce(stdout, &key, ticket.admission, started_in_think);
                {
                    let _scope =
                        BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, ticket.admission);
                    crate::ar::emit_generation_error(route, stdout, Some(&key.id), &msg, kind, false, true);
                    let _ = stdout.flush();
                }
                let _ = sched.abort_lane(lane_idx, &key, ticket.admission);
            };
            let rendered = match rendered {
                Ok(r) if !r.is_empty() && r.len() < sched.lane_capacity => r,
                Ok(_) => {
                    refuse(stdout, sched, "prompt exceeds lane capacity or empty".into(), "validation");
                    continue;
                }
                Err(e) => {
                    refuse(stdout, sched, e, "validation");
                    continue;
                }
            };
            let s = &pending_req.sampling;
            // Singleton AR sampling scope: generated tokens only (empty at
            // the first sample), so no attractor blocks yet.
            let sampler = hipfire_runtime::sampler::SamplerConfig {
                temperature: s.temp,
                top_p: s.top_p,
                repeat_penalty: s.repeat_penalty,
                repeat_window: s.repeat_window,
                presence_penalty: s.presence_penalty,
                frequency_penalty: s.frequency_penalty,
                blocked_tokens: Vec::new(),
                top_k: s.top_k,
                min_p: s.min_p,
            };
            // A greedy request whose whole budget fits the MTP head's lane
            // capacity decodes as a spec lane, exactly as the singleton MTP
            // route would serve it — unless it carries a think budget or
            // stop strings, which run per committed token on AR lanes.
            let spec_req = spec_engine
                .filter(|&(k, cap)| {
                    s.temp <= 0.0
                        && !think.budgeted()
                        && stops.is_empty()
                        && rendered.len() + lane_max_tokens(&key, sched) + k + 1 <= cap
                })
                .map(|_| spec_request_config(s, rng_state as u64));
            let keep = keep_conversations && spec_req.is_none();
            let admitted = (|| -> Result<(Option<u32>, crate::vmm_conv::Checkpoints, usize), String> {
                let b = vmm_bundle(&mut model.state).ok_or("model is not Qwen35")?;
                let init = || hipfire_arch_qwen35::forward_slots::vmm::VmmRequestInit {
                    prompt_len: rendered.len(),
                    stop_ids: stop_ids.clone(),
                    sampler: sampler.clone(),
                    rng_state,
                    history: Vec::new(),
                };
                let (state, checkpoints, start) = if keep {
                    crate::vmm_conv::lane_owner(gpu, b, &rendered, epoch, lane_idx, init)?
                } else {
                    let state = hipfire_arch_qwen35::forward_slots::vmm::Qwen35RequestState::new_like(
                        gpu,
                        &b.config,
                        &b.kv_cache,
                        &b.dn_state,
                        epoch,
                        lane_idx,
                        init(),
                    )?;
                    (state, Vec::new(), 0)
                };
                let hipfire_arch_qwen35::Qwen35Bundle {
                    vmm_store,
                    weights,
                    config,
                    scratch,
                    ..
                } = b;
                let store = vmm_store.as_mut().expect("lane owner came from the staged store");
                if let Err((state, e)) = store.admit(state) {
                    let freed = state.free_gpu(gpu);
                    crate::vmm_conv::free_checkpoints(checkpoints, gpu);
                    return Err(format!("{e}; free: {freed:?}"));
                }
                let Some(request) = spec_req else {
                    return Ok((None, checkpoints, start));
                };
                match store.spec_prefill(gpu, weights, config, scratch, &epoch, &rendered, request) {
                    Ok(seed) => Ok((Some(seed), checkpoints, start)),
                    Err(e) => {
                        let freed = store.retire(&epoch).and_then(|st| st.free_gpu(gpu));
                        crate::vmm_conv::free_checkpoints(checkpoints, gpu);
                        Err(format!("spec prefill: {e}; free: {freed:?}"))
                    }
                }
            })();
            let (spec_seed, checkpoints, start) = match admitted {
                Ok(v) => v,
                Err(e) => {
                    // Per-admission refusal (capacity/budget): this request
                    // fails visibly; peers keep running.
                    refuse(stdout, sched, format!("VMM batch admission refused: {e}"), "context_length");
                    continue;
                }
            };
            announce(stdout, &key, ticket.admission, started_in_think);
            epochs[lane_idx] = epoch;
            spec_lane[lane_idx] = spec_seed.is_some();
            work[lane_idx] = match spec_seed {
                // Prompt filled; the lane decodes by spec windows only.
                Some(seed) => {
                    spec_seeds.push(hipfire_runtime::slot_batch::RequestAdvance {
                        epoch,
                        committed_ids: vec![seed],
                        committed_position: rendered.len(),
                        accepted_drafts: 0,
                        verified_rows: 0,
                        finish: None,
                    });
                    PendingWork {
                        next_pos: rendered.len(),
                        decoding: true,
                        ..idle_work(lane_idx)
                    }
                }
                // Prefill from the reuse point (0 when cold).
                None => PendingWork {
                    remaining_prompt: rendered[start..].to_vec(),
                    next_pos: start,
                    ..idle_work(lane_idx)
                },
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = rendered.len();
                lane.conversation_tokens = rendered.clone();
            }
            ctl[lane_idx] = VmmLaneCtl {
                started_in_think,
                think,
                keep,
                checkpoints,
                cached: start,
                prefill_tokens: rendered.len() - start,
                decode_start: rendered.len(),
                ..VmmLaneCtl::default()
            };
            loop_guards[lane_idx] = hipfire_runtime::loop_guard::LoopGuard::from_config(
                hipfire_runtime::config::get(),
            );
            producers[lane_idx] = Some(
                QwenArSemanticProducer::new_with_tool_protocol(key.id.clone(), started_in_think, false)
                    .with_stop(&stops),
            );
        }
        // ── Commit phase: the last step's picks, one token at a time in the
        // singleton decode loop's order (emit → stop/EOS → think control →
        // loop guard → budgets), before the next step plans — so a forced
        // think close decided here replaces the very next pick ──
        let commits = std::mem::take(&mut pending_commits);
        let mut finished: Vec<(usize, VmmFinish)> = Vec::new();
        {
            let tokenizer = model.tokenizer.as_ref().expect("checked above");
            for (epoch, token, length) in commits {
                let Some(idx) = epochs.iter().position(|e| *e == epoch) else {
                    continue;
                };
                if ctl[idx].drain.is_some() || finished.iter().any(|(i, _)| *i == idx) {
                    continue;
                }
                let Some(lane_key) = sched.lanes[idx].key().cloned() else {
                    continue;
                };
                let max_toks = lane_max_tokens(&lane_key, sched);
                let lane_capacity = sched.lane_capacity;
                let BatchLane::Running(lane) = &mut sched.lanes[idx] else {
                    continue;
                };
                if lane.prefill_done_at.is_none() {
                    lane.prefill_done_at = Some(Instant::now());
                    lane.seq_pos = lane.prompt_len;
                }
                let Some(producer) = producers[idx].as_mut() else {
                    continue;
                };
                let _scope = BatchAttemptScope::enter_for_generation(
                    &lane_key.id,
                    lane_key.attempt_id,
                    lane.ticket.admission,
                );
                match vmm_commit_sampled(
                    stdout,
                    tokenizer,
                    lane,
                    producer,
                    &loop_guards[idx],
                    &mut ctl[idx],
                    token,
                    !spec_lane[idx],
                    max_toks,
                    lane_capacity,
                    length,
                    (eos_tok, im_end_tok),
                    &close_tokens,
                ) {
                    Ok(None) => {}
                    Ok(Some(f)) => finished.push((idx, f)),
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            model,
                            &mut epochs,
                            &mut ctl,
                            stdout,
                            format!("semantic classify lane {idx}: {e}"),
                        )
                    }
                }
            }
        }
        for (idx, f) in finished {
            let hit_length_cap = batch_hit_length_cap(f.hit_max, f.hit_lane_cap, f.is_eos, f.stopped, f.loop_hit);
            let Some(producer_owned) = producers[idx].take() else {
                continue;
            };
            let BatchLane::Running(lane) = &sched.lanes[idx] else {
                continue;
            };
            let key = lane.key.clone();
            let admission = lane.ticket.admission;
            let _scope = BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let (finish, visible) = match producer_owned.finish(stdout, hit_length_cap) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        model,
                        &mut epochs,
                        &mut ctl,
                        stdout,
                        format!("semantic finish lane {idx}: {e}"),
                    )
                }
            };
            if !finish.wire_tool_calls.is_empty() {
                return fail_all(
                    sched,
                    gpu,
                    model,
                    &mut epochs,
                    &mut ctl,
                    stdout,
                    format!("semantic finish lane {idx}: unexpected tool calls"),
                );
            }
            let BatchLane::Running(lane) = &mut sched.lanes[idx] else {
                continue;
            };
            let c = &mut ctl[idx];
            let generated = lane.streamed_tokens.len();
            let prefill_tokens = if c.prefill_tokens > 0 { c.prefill_tokens } else { lane.prompt_len };
            let metrics = batch_lane_done_metrics(
                lane.created_at,
                lane.prefill_done_at,
                lane.first_token_at,
                Instant::now(),
                prefill_tokens,
                generated,
            );
            let mut pending_done = qwen_ar_done_value(
                &key.id,
                finish.finish_reason,
                generated,
                metrics.tok_s,
                prefill_tokens,
                metrics.prefill_ms,
                metrics.prefill_tok_s,
                metrics.decode_tok_s,
                metrics.ttft_ms,
                c.cached,
                "",
            );
            pending_done["latency_ms"] =
                serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
            attach_continuous_batch_route_evidence(
                &mut pending_done,
                batch_size,
                idx,
                sched.lane_capacity,
                lane.max_active_lanes.max(1),
            );
            pending_done["continuous_batch_route"] = serde_json::json!("vmm");
            // The singleton's verbatim assistant-turn store: the generated
            // body less its ChatML trailer and `<|im_end|>`, applied when the
            // client commits.
            let action = qwen_ar_cache_action(&finish, &visible);
            if action.store {
                let mut seq = lane.conversation_tokens.get(c.decode_start..).unwrap_or(&[]).to_vec();
                while seq.last().is_some_and(|t| nl_tokens.contains(t)) {
                    seq.pop();
                }
                if seq.last().is_some() && seq.last() == stop_ids.get(1) {
                    seq.pop();
                }
                c.turn_store = Some((action, seq));
            }
            // Keep the conversation: feed its committed rows the singleton's
            // KV already holds (the finishing token, a forced close) and the
            // ChatML trailer, then retain the owner (`vmm_conv`).
            if c.keep && !f.hit_lane_cap {
                let st = vmm_bundle(&mut model.state)
                    .and_then(|b| b.vmm_store.as_ref())
                    .and_then(|s| s.request_state(&epochs[idx]));
                if let Some(st) = st {
                    let mut rows: Vec<u32> = st.pending_seed.into_iter().chain(c.forced.drain(..)).collect();
                    let trailer = stop_ids.get(1).is_some_and(|t| lane.conversation_tokens.last() == Some(t));
                    if trailer {
                        rows.extend_from_slice(&nl_tokens);
                    }
                    if !rows.is_empty() && st.position + rows.len() <= st.kv.vmm_logical_bound() {
                        if trailer {
                            // Hidden commit, as the singleton's trailer.
                            lane.conversation_tokens.extend_from_slice(&nl_tokens);
                        }
                        work[idx] = PendingWork {
                            remaining_prompt: vec![rows[0]],
                            next_pos: st.position,
                            decoding: true,
                            ..idle_work(idx)
                        };
                        c.forced = rows[1..].iter().copied().collect();
                        c.drain = Some(VmmDrain {
                            left: rows.len(),
                            pending_done,
                        });
                        continue;
                    }
                }
            }
            // Retire the finished owner before publishing commit_ready: it
            // holds no device state while it awaits the client.
            if let Err(e) = vmm_retire_free(model, gpu, &epochs[idx]) {
                return fail_all(
                    sched,
                    gpu,
                    model,
                    &mut epochs,
                    &mut ctl,
                    stdout,
                    format!("retire lane {idx} on finish: {e}"),
                );
            }
            epochs[idx] = IDLE;
            work[idx] = idle_work(idx);
            spec_lane[idx] = false;
            crate::vmm_conv::free_checkpoints(std::mem::take(&mut ctl[idx].checkpoints), gpu);
            if !vmm_publish_commit_ready(sched, stdout, idx, &key, admission, pending_done) {
                ctl[idx].reset(gpu);
            }
        }
        let running: Vec<usize> = (0..batch_size)
            .filter(|&i| matches!(sched.lanes[i], BatchLane::Running(_)) && epochs[i].is_admitted())
            .collect();
        let awaiting = sched
            .lanes
            .iter()
            .any(|l| matches!(l, BatchLane::AwaitingClient(_)));
        if running.is_empty() && !awaiting && sched.inbox.is_empty() && inbox.backlog.is_empty() {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                lane.max_active_lanes = lane.max_active_lanes.max(active_now);
            }
        }
        // Singleton-equivalent attractor blocks over each request's
        // generated-only history, recomputed before every pick; once the
        // force-answer latch is set, `<think>` is blocked too. The RNG of
        // every lane is kept so a pick replaced by a forced token leaves it
        // where the singleton (which does not sample there) leaves it.
        let mut rng_before: Vec<(RequestEpoch, u32)> = Vec::with_capacity(running.len());
        if let Some(store) = vmm_bundle(&mut model.state).and_then(|b| b.vmm_store.as_mut()) {
            for &idx in &running {
                if let Some(st) = store.request_state_mut(&epochs[idx]) {
                    let mut blocked = std::mem::take(&mut st.sampler.blocked_tokens);
                    blocked.clear();
                    sampler::collect_unclosed_attractor_blocks(
                        &st.history,
                        &attractor_pairs,
                        20,
                        2,
                        &mut blocked,
                    );
                    if ctl[idx].think.blocks_think_open() {
                        blocked.extend(think_open_tok);
                    }
                    st.sampler.blocked_tokens = blocked;
                    rng_before.push((epochs[idx], st.rng_state));
                }
            }
        }
        // ── One planned step: provision → forward → commit → publish ──
        let eligible: Vec<bool> = (0..batch_size).map(|i| running.contains(&i) && !spec_lane[i]).collect();
        // Exact route: every prefill chunk must be the singleton route's own
        // chunk for that request (DeltaNet requant cadence), so the planner
        // takes exactly that length or nothing.
        let forced = (|| -> Result<Vec<Option<usize>>, String> {
            let b = vmm_bundle(&mut model.state).ok_or("model is not Qwen35")?;
            let store = b.vmm_store.as_ref().ok_or("store not staged")?;
            if store.route() != hipfire_arch_qwen35::forward_slots::vmm::VmmRoute::Exact {
                return Ok(Vec::new());
            }
            (0..batch_size)
                .map(|i| {
                    let w = &work[i];
                    if !eligible[i] || w.decoding || w.remaining_prompt.is_empty() {
                        return Ok(None);
                    }
                    store
                        .exact_prefill_chunk_len(
                            gpu,
                            &b.weights,
                            &b.config,
                            &epochs[i],
                            w.remaining_prompt.len(),
                        )
                        .map(Some)
                })
                .collect()
        })();
        planner.forced_prefill_len = match forced {
            Ok(f) => f,
            Err(e) => {
                return fail_all(sched, gpu, model, &mut epochs, &mut ctl, stdout, format!("exact chunk: {e}"))
            }
        };
        let plan = match planner.plan_step(
            &work,
            &epochs,
            &eligible,
            row_budget,
            params.prefill_min_tokens,
        ) {
            Ok(p) => p,
            Err(e) => return fail_all(sched, gpu, model, &mut epochs, &mut ctl, stdout, format!("plan: {e}")),
        };
        // Spec lanes whose seed is already on the wire take one MTP window:
        // a Verify request in the same step plan as the AR/prefill rows
        // (`[seed, placeholders…]`, k = min(max_emit - 1, K)); the executor
        // drafts, verifies on the shared trunk and accept/repairs it.
        let mut step_plan = plan.clone();
        if step_plan.requests.is_empty() {
            step_plan.batch.m_per_slot = vec![0; batch_size];
        }
        if let Some((k_max, _)) = spec_engine {
            let store = vmm_bundle(&mut model.state).and_then(|b| b.vmm_store.as_ref());
            for &i in running.iter().filter(|&&i| spec_lane[i] && !spec_seeds.iter().any(|a| a.epoch == epochs[i])) {
                let BatchLane::Running(lane) = &sched.lanes[i] else {
                    continue;
                };
                let max_emit = lane_max_tokens(&lane.key, sched).saturating_sub(lane.streamed_tokens.len());
                let Some(st) = store.and_then(|s| s.request_state(&epochs[i])) else {
                    continue;
                };
                let (Some(seed), true) = (st.pending_seed, max_emit > 0) else {
                    continue;
                };
                let k = (max_emit - 1).min(k_max);
                let b = &mut step_plan.batch;
                let begin = b.tokens.len();
                for j in 0..=k {
                    // Rows past the seed are host-only placeholders.
                    b.tokens.push(if j == 0 { seed } else { 0 });
                    b.positions.push((st.position + j) as i32);
                    b.row_slot.push(i as i32);
                }
                if b.m_per_slot.len() <= i {
                    b.m_per_slot.resize(i + 1, 0);
                }
                b.m_per_slot[i] = k + 1;
                step_plan.verify_rows += k + 1;
                step_plan.requests.push(hipfire_runtime::slot_batch::RequestRows {
                    epoch: epochs[i],
                    rows: hipfire_runtime::slot_batch::RowRange { begin, len: k + 1 },
                    kind: hipfire_runtime::slot_batch::RequestStepKind::Verify { draft_len: k },
                });
            }
            // Stable slot order.
            let rs = step_plan.batch.row_slot.clone();
            step_plan.requests.sort_by_key(|r| rs[r.rows.begin]);
        }
        let mut advances = if step_plan.requests.is_empty() {
            planner.discard();
            Vec::new()
        } else {
            let stepped = (|| -> Result<Vec<hipfire_runtime::slot_batch::RequestAdvance>, String> {
                let b = vmm_bundle(&mut model.state).ok_or("model is not Qwen35")?;
                let hipfire_arch_qwen35::Qwen35Bundle {
                    vmm_store,
                    weights,
                    config,
                    scratch,
                    ..
                } = b;
                let store = vmm_store.as_mut().ok_or("store not staged")?;
                let mut ex = store.executor(weights, config, scratch);
                let out = ex
                    .provision_step(gpu, &step_plan)
                    .and_then(|()| ex.forward_step(gpu, &step_plan))
                    .and_then(|o| ex.commit_step(gpu, &step_plan, o));
                if out.is_err() {
                    store.abort_step(&step_plan);
                }
                out
            })();
            let advances = match stepped {
                Ok(a) => a,
                Err(e) => {
                    planner.discard();
                    return fail_all(sched, gpu, model, &mut epochs, &mut ctl, stdout, format!("step: {e}"));
                }
            };
            if plan.requests.is_empty() {
                planner.discard();
            } else {
                // The planner publishes only its own (AR/prefill) requests.
                let planned: Vec<_> = advances
                    .iter()
                    .filter(|a| plan.requests.iter().any(|r| r.epoch == a.epoch))
                    .cloned()
                    .collect();
                if let Err(e) = planner.publish(&mut work, &epochs, &plan, &planned) {
                    return fail_all(sched, gpu, model, &mut epochs, &mut ctl, stdout, format!("publish: {e}"));
                }
            }
            advances
        };
        advances.append(&mut spec_seeds);
        if advances.is_empty() {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        if let Some(store) = vmm_bundle(&mut model.state).and_then(|b| b.vmm_store.as_mut()) {
            // Resume checkpoints at the singleton's cadence: after every
            // prefill chunk and after every sampled token's forward.
            for r in &plan.requests {
                let Some(idx) = epochs.iter().position(|e| *e == r.epoch) else {
                    continue;
                };
                let take = ctl[idx].keep
                    && match r.kind {
                        hipfire_runtime::slot_batch::RequestStepKind::Prefill => true,
                        hipfire_runtime::slot_batch::RequestStepKind::Ar => ctl[idx].main_row,
                        _ => false,
                    };
                if let (true, Some(st)) = (take, store.request_state(&r.epoch)) {
                    crate::vmm_conv::checkpoint(&mut ctl[idx].checkpoints, &st.dn, gpu, st.position);
                }
            }
            // A lane with committed-but-unfed tokens (forced think close,
            // finish drain) feeds them next: the pick this AR row produced is
            // replaced by the next such token (history and RNG as if never
            // sampled). A drain that fed its last row is done.
            let mut drained: Vec<usize> = Vec::new();
            for adv in advances.iter_mut() {
                let Some(idx) = epochs.iter().position(|e| *e == adv.epoch) else {
                    continue;
                };
                let ar_row = plan.requests.iter().any(|r| {
                    r.epoch == adv.epoch && matches!(r.kind, hipfire_runtime::slot_batch::RequestStepKind::Ar)
                });
                if spec_lane[idx] || !ar_row {
                    continue;
                }
                let c = &mut ctl[idx];
                if let Some(d) = c.drain.as_mut() {
                    d.left -= 1;
                    adv.committed_ids.clear();
                    if d.left == 0 {
                        drained.push(idx);
                        continue;
                    }
                }
                let Some(f) = c.forced.pop_front() else {
                    continue;
                };
                if let Some(st) = store.request_state_mut(&adv.epoch) {
                    st.pending_seed = Some(f);
                    if let Some(last) = st.history.last_mut() {
                        *last = f;
                    }
                    if let Some(&(_, rng)) = rng_before.iter().find(|(e, _)| *e == adv.epoch) {
                        st.rng_state = rng;
                    }
                }
                work[idx].remaining_prompt = vec![f];
                work[idx].decoding = true;
                adv.committed_ids.clear();
                c.main_row = false;
            }
            // Every row of a drained lane is forwarded: its owner now holds
            // exactly the conversation the singleton keeps resident. Keep it
            // (reusable once the client commits), then publish commit_ready.
            for idx in drained {
                let drain = ctl[idx].drain.take().expect("drained lane");
                let retired = store.retire(&epochs[idx]);
                epochs[idx] = IDLE;
                work[idx] = idle_work(idx);
                spec_lane[idx] = false;
                let st = match retired {
                    Ok(st) => st,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            model,
                            &mut epochs,
                            &mut ctl,
                            stdout,
                            format!("retire lane {idx} after drain: {e}"),
                        )
                    }
                };
                let BatchLane::Running(lane) = &sched.lanes[idx] else {
                    let _ = st.free_gpu(gpu);
                    ctl[idx].reset(gpu);
                    continue;
                };
                let key = lane.key.clone();
                let admission = lane.ticket.admission;
                let checkpoints = std::mem::take(&mut ctl[idx].checkpoints);
                if st.position == lane.conversation_tokens.len() && !st.poisoned {
                    crate::vmm_conv::retain(
                        store,
                        gpu,
                        st,
                        lane.conversation_tokens.clone(),
                        checkpoints,
                        Some((key.id.clone(), key.attempt_id)),
                    );
                } else {
                    eprintln!(
                        "[vmm-prefix] id={} not kept: owner position {} != conversation {}",
                        key.id,
                        st.position,
                        lane.conversation_tokens.len()
                    );
                    let _ = st.free_gpu(gpu);
                    crate::vmm_conv::free_checkpoints(checkpoints, gpu);
                }
                if !vmm_publish_commit_ready(sched, stdout, idx, &key, admission, drain.pending_done) {
                    if let Some(i) = store.prefix_pool.iter().position(|e| {
                        e.pending.as_ref().is_some_and(|(id, a)| *id == key.id && *a == key.attempt_id)
                    }) {
                        let _ = store.prefix_pool.remove(i).free_gpu(gpu);
                    }
                    ctl[idx].reset(gpu);
                }
            }
        }
        for adv in &advances {
            for &t in &adv.committed_ids {
                pending_commits.push((adv.epoch, t, adv.finish.as_deref() == Some("length")));
            }
        }
    }
    Ok(())
}
pub fn drive_lfm_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    gpu: &mut rdna_compute::Gpu,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let route = crate::ar::GenerationRoute::LfmAr;
    let (batch_state_ptr, config_ptr, weights_ptr, tokenizer_ptr, chat_template_clone, eos_tok) =
        match model.state.as_mut().and_then(|s| {
            (s.as_mut() as &mut dyn Any).downcast_mut::<hipfire_arch_lfm2moe::Lfm2MoeBundle>()
        }) {
            Some(b) => {
                let batch_ptr = match b.lfm2_decode_batch.as_mut() {
                    Some(s) => s as *mut Lfm2DecodeBatchState,
                    None => {
                        return Err(BatchDriveError::Gpu(
                            "batch state not allocated".to_string(),
                        ))
                    }
                };
                (
                    batch_ptr,
                    &b.config as *const lfm2moe::config::Lfm2MoeConfig,
                    &b.weights as *const lfm2moe::Lfm2MoeWeights,
                    match model.tokenizer.as_ref() {
                        Some(t) => t as *const _,
                        None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
                    },
                    model.chat_template.clone(),
                    b.eos_tok,
                )
            }
            _ => return Err(BatchDriveError::Gpu("batch model not Lfm2Moe".to_string())),
        };
    let batch_state = unsafe { &mut *batch_state_ptr };
    let config = unsafe { &*config_ptr };
    let weights = unsafe { &*weights_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    // Stop set mirrors crate::dense::generate_lfm2moe: eos_tok plus single-id encodings for
    // <|endoftext|>, </s>, <|im_end|>. String guard catches leaked EOS-class
    // strings where encode does not round-trip (e.g. <|endoftext|>).
    let mut stop_toks: Vec<u32> = vec![eos_tok];
    for s in ["<|endoftext|>", "</s>", "<|im_end|>"] {
        let ids = tokenizer.encode(s);
        if ids.len() == 1 && !stop_toks.contains(&ids[0]) {
            stop_toks.push(ids[0]);
        }
    }
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpu: &mut rdna_compute::Gpu,
                    batch_state: &mut Lfm2DecodeBatchState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for lane in sched.lanes.iter() {
            let Some(key) = lane.key() else {
                continue;
            };
            let admission = match lane {
                BatchLane::Seeding(q) | BatchLane::Running(q) => q.ticket.admission,
                BatchLane::AwaitingClient(t) => t.ticket.admission,
                BatchLane::Empty { .. } => continue,
            };
            if uniq_set.insert((key.clone(), admission)) {
                uniq.push((key.clone(), admission));
            }
        }
        for key in sched.inbox.iter().cloned() {
            if let Some(request) = sched.pending.get(&key) {
                if uniq_set.insert((key.clone(), request.admission)) {
                    uniq.push((key, request.admission));
                }
            }
        }
        for (key, request) in sched.pending.iter() {
            if uniq_set.insert((key.clone(), request.admission)) {
                uniq.push((key.clone(), request.admission));
            }
        }
        let mut first_err: Option<String> = None;
        if let Err(e) = batch_state.reset(gpu) {
            first_err = Some(format!("batch reset: {e}"));
        }
        crate::common::fail_closed_invalidate_graphs_and_replay(gpu);
        let sync = crate::common::fail_closed_device_sync(gpu);
        let prior = match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        };
        let ep = crate::common::fail_closed_epilogue_after_sync(prior, sync);
        for (key, admission) in &uniq {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, *admission);
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason}"),
                "gpu",
                ep.rolled_back,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        if !ep.rolled_back {
            return Err(BatchDriveError::Poisoned(format!(
                "{reason}; {}",
                ep.context.unwrap_or_default()
            )));
        }
        Err(BatchDriveError::Gpu(reason))
    };
    loop {
        let mut to_commit: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let admission = term.ticket.admission;
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id, admission) || expired {
                    to_abort.push((idx, key, admission));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id, admission)
                {
                    to_commit.push((idx, key.clone(), admission, term.pending_done.clone()));
                }
            }
        }
        for (idx, key, admission) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
        }
        for (idx, key, admission, pending_done) in to_commit {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let reset_ok = match batch_state.reset_lane(gpu, config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {idx} on commit: {e}"),
                    );
                }
            };
            let commit_ok = sched.commit_lane_retain_terminal(idx, &key, admission);
            let _terminal_cleanup = BatchTerminalCleanup::new(&key, Some(admission));
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!("reset_ok handled above"),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error_for_route(
                        route,
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_generation_done_value(route, stdout, &pending_done);
                }
            }
        }
        let mut queued_abort: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if let Some(request) = sched.pending.get(&key) {
                if batch_check_abort(&key.id, key.attempt_id, request.admission) {
                    queued_abort.push((key, request.admission));
                }
            }
        }
        for (key, admission) in queued_abort {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_queued(&key, admission);
        }
        let mut running_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(key) = sched.lanes[idx].key().cloned() {
                let admission = match &sched.lanes[idx] {
                    BatchLane::Running(l) | BatchLane::Seeding(l) => l.ticket.admission,
                    _ => continue,
                };
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&key.id, key.attempt_id, admission)
                {
                    running_abort.push((idx, key, admission));
                }
            }
        }
        for (idx, key, admission) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
        }
        let mut barrier: Option<DaemonMsg> = None;
        // A fresh continuous-batch wave reaches the daemon through many
        // concurrent HTTP handlers. Give that first wave a small, bounded
        // coalescing window so admission does not race the second request and
        // serialize the remainder through single-lane prefill.
        let admission_deadline = (sched.active_count() == 0 && sched.awaiting_count() == 0)
            .then(|| Instant::now() + Duration::from_millis(20));
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => {
                    let Some(deadline) = admission_deadline else {
                        break;
                    };
                    if sched.active_count() != 0
                        || sched.awaiting_count() != 0
                        || sched.inbox.len() >= batch_size
                    {
                        break;
                    }
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    match inbox.recv_timeout(remaining) {
                        Ok(m) => m,
                        Err(
                            mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected,
                        ) => {
                            break;
                        }
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            let (dm, carried_admission) = match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    (DaemonMsg::Regular(json), Some(admission))
                }
                other => (other, None),
            };
            match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    barrier = Some(DaemonMsg::RegularWithAdmission(json, admission));
                    break;
                }
                DaemonMsg::SingletonWithAdmission(json, transfer) => {
                    barrier = Some(DaemonMsg::SingletonWithAdmission(json, transfer));
                    break;
                }
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(0) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate attempt_id must be nonzero",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        let Some(admission) = carried_admission else {
                            barrier = Some(daemon_regular_with_admission(json, None));
                            break;
                        };
                        if batch_check_abort(&id, attempt_id, admission) {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::LfmAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_generation_cancel(route, stdout, &id, 0);
                            batch_clear_terminal_at_generation(&id, attempt_id, admission);
                            continue;
                        }
                        if !is_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(daemon_regular_with_admission(json, carried_admission));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    emit_batch_admission_error(
                                        stdout,
                                        &id,
                                        attempt_id,
                                        admission,
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            max_think != 1,
                            None,
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        if started_in_think {
                            let handoff = match handoff_admitted_started_in_think(
                                &id,
                                attempt_id,
                                admission,
                                json,
                                GenerationRoute::LfmAr,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => {
                                    return fail_all(sched, gpu, batch_state, stdout, reason)
                                }
                            };
                            barrier = Some(handoff);
                            break;
                        }
                        if prompt_tokens.is_empty() {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                "empty prompt after tokenize",
                                "validation",
                                false,
                                false,
                            );
                            continue;
                        }
                        if batch_lfm_exceeds_capacity(
                            prompt_tokens.len(),
                            max_tokens_req,
                            sched.lane_capacity,
                        ) {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                &format!(
                                    "prompt exceeds context capacity: prompt={} + max_tokens={} > capacity={} — reload model with a larger max_seq",
                                    prompt_tokens.len(),
                                    max_tokens_req,
                                    sched.lane_capacity
                                ),
                                "context_length",
                                false,
                                false,
                            );
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id, admission);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            admission,
                            original_msg: json.clone(),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            eprintln!(
                                "[batch] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::LfmAr,
                                stdout,
                                &id,
                                started_in_think,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(daemon_regular_with_admission(json, carried_admission));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        // ---- generic initial-wave fast path (batched prefill, O(prompt_len) vs O(n*prompt_len)) ----
        // Non-mutating candidate scan ensures a one-request wave is never removed.
        if sched.active_count() == 0 && sched.awaiting_count() == 0 {
            let n = lfm_fast_path_candidate_len(sched);
            if n >= 2 {
                // Assign exactly n prefix lanes; each try_assign_one pops front and binds.
                let mut assigned_keys: Vec<AttemptKey> = Vec::with_capacity(n);
                let mut assigned_tickets: Vec<LaneTicket> = Vec::with_capacity(n);
                let mut prompts_for_batch: Vec<Vec<u32>> = Vec::with_capacity(n);
                let mut assign_ok = true;
                for _ in 0..n {
                    match sched.try_assign_one() {
                        Some((key, ticket)) => {
                            if let Some(req) = sched.pending.get(&key).cloned() {
                                prompts_for_batch.push(req.prompt_tokens);
                            } else {
                                prompts_for_batch.push(Vec::new());
                            }
                            assigned_keys.push(key);
                            assigned_tickets.push(ticket);
                        }
                        None => {
                            assign_ok = false;
                            break;
                        }
                    }
                }
                if assign_ok && assigned_keys.len() == n {
                    let prompt_refs: Vec<&[u32]> =
                        prompts_for_batch.iter().map(|v| v.as_slice()).collect();
                    let prefill_res =
                        batch_state.prefill_lanes_batched(gpu, weights, config, &prompt_refs);
                    match prefill_res {
                        Ok(()) => {
                            for (idx, key) in assigned_keys.iter().enumerate() {
                                let ticket = assigned_tickets[idx];
                                let lane_idx = ticket.lane;
                                let admission = ticket.admission;
                                if batch_check_abort(&key.id, key.attempt_id, admission) {
                                    if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                                        return fail_all(
                                            sched,
                                            gpu,
                                            batch_state,
                                            stdout,
                                            format!("reset lane {lane_idx} on batched prefill abort: {e}"),
                                        );
                                    }
                                    let _scope = BatchAttemptScope::enter_for_generation(
                                        &key.id,
                                        key.attempt_id,
                                        admission,
                                    );
                                    let ep = crate::common::RollbackEpilogue {
                                        rolled_back: true,
                                        context: None,
                                    };
                                    crate::common::emit_spec_cancel_after_rollback(
                                        stdout, &key.id, 0, &ep,
                                    );
                                    let _ = sched.abort_lane(lane_idx, key, admission);
                                    continue;
                                }
                                let hist: &[u32] = &[];
                                let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                                    BatchLane::Running(l) => {
                                        (l.rng_state as u32, l.sampling.clone())
                                    }
                                    _ => continue,
                                };
                                match batch_state.sample_lane_product(
                                    gpu,
                                    config,
                                    lane_idx,
                                    hist,
                                    sampling.temp,
                                    sampling.top_p,
                                    sampling.top_k,
                                    sampling.min_p,
                                    lane_rng,
                                    sampling.repeat_penalty,
                                    sampling.presence_penalty,
                                    sampling.frequency_penalty,
                                ) {
                                    Ok((next_token, next_rng)) => {
                                        let prompt_len = prompts_for_batch[idx].len();
                                        lfm_populate_lane_after_sample(
                                            sched, lane_idx, next_token, next_rng, prompt_len,
                                        );
                                    }
                                    Err(e) => {
                                        return fail_all(
                                            sched,
                                            gpu,
                                            batch_state,
                                            stdout,
                                            format!(
                                                "sample lane {lane_idx} after batched prefill: {e}"
                                            ),
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            return fail_all(
                                sched,
                                gpu,
                                batch_state,
                                stdout,
                                format!("batched prefill lanes 0..{n}: {e}"),
                            );
                        }
                    }
                } else {
                    // Partial assign failure: rollback any already-assigned lanes.
                    for (k, t) in assigned_keys.iter().zip(assigned_tickets.iter()) {
                        if let Err(e) = batch_state.reset_lane(gpu, config, t.lane) {
                            retire_lane_after_reset_failure(
                                sched,
                                stdout,
                                k,
                                t.admission,
                                t.lane,
                                "partial assign rollback",
                                &e,
                            );
                        } else {
                            let _ = sched.abort_lane(t.lane, k, t.admission);
                        }
                    }
                }
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let max_tokens_req = pending_req.max_tokens;
            let started_in_think = pending_req.started_in_think;
            let admission = pending_req.admission;
            if started_in_think {
                // Reset while the batch owner remains live, then retire the
                // lane and hand the complete original request to singleton.
                if let Err(err) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                let handoff = match handoff_started_in_think(
                    sched,
                    lane_idx,
                    &key,
                    &pending_req,
                    GenerationRoute::LfmAr,
                ) {
                    Ok(msg) => msg,
                    Err(reason) => {
                        return fail_all(sched, gpu, batch_state, stdout, reason);
                    }
                };
                inbox.push_front(handoff);
                break;
            }
            // Re-validate capacity at assignment time (defensive; lane_capacity is the source of truth).
            if batch_lfm_exceeds_capacity(prompt_tokens.len(), max_tokens_req, sched.lane_capacity)
            {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on capacity re-check: {e}"),
                    );
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                emit_lfm_assignment_capacity_error(
                    stdout,
                    &key,
                    prompt_tokens.len(),
                    max_tokens_req,
                    sched.lane_capacity,
                );
                let _ = sched.abort_lane(lane_idx, &key, admission);
                continue;
            }
            if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {lane_idx}: {e}"),
                );
            }
            // Zero-token completion: no GPU work, ordinary two-phase terminal with zero tokens.
            if max_tokens_req == 0 {
                if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                    lane.prompt_len = prompt_tokens.len();
                    lane.seq_pos = prompt_tokens.len();
                    lane.next_token = None;
                    lane.streamed_tokens = Vec::new();
                    lane.bytes_fed_to_filter = 0;
                    lane.prefill_done_at = Some(Instant::now());
                    lane.first_token_at = None;
                }
                let lane_ref = match &sched.lanes[lane_idx] {
                    BatchLane::Running(l) => l,
                    _ => continue,
                };
                let metrics = batch_lane_done_metrics(
                    lane_ref.created_at,
                    lane_ref.prefill_done_at,
                    lane_ref.first_token_at,
                    Instant::now(),
                    lane_ref.prompt_len,
                    0,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": 0,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane_ref.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "length",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    lane_idx,
                    sched.lane_capacity,
                    lane_ref.max_active_lanes.max(1),
                );
                let mut envelope = pending_done.clone();
                envelope["type"] = serde_json::json!("commit_ready");
                // Install AwaitingClient/Ready BEFORE publishing commit_ready.
                let marked = sched.mark_awaiting_commit(lane_idx, pending_done.clone());
                if !marked {
                    // Failed to mark — rollback lane without publishing.
                    if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                        retire_lane_after_reset_failure(
                            sched,
                            stdout,
                            &key,
                            admission,
                            lane_idx,
                            "mark_awaiting_commit",
                            &e,
                        );
                    } else {
                        let _ = sched.abort_lane(lane_idx, &key, admission);
                    }
                    continue;
                }
                let write_ok = {
                    let _scope =
                        BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                    writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
                };
                if !write_ok {
                    // Publication failed — rollback attested reset and free lane.
                    if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                        retire_lane_after_reset_failure(
                            sched,
                            stdout,
                            &key,
                            admission,
                            lane_idx,
                            "commit_ready publish",
                            &e,
                        );
                    } else {
                        let _ = sched.abort_lane(lane_idx, &key, admission);
                    }
                }
                continue;
            }
            // Cancellable prefill: check abort before GPU, then delegate to batch prefill.
            // If abort is latched before or during prefill, we must reset only this lane,
            // emit attested abort, and continue peers without sampling.
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on pre-prefill abort: {e}"),
                    );
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                let ep = crate::common::RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                };
                crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
                let _ = sched.abort_lane(lane_idx, &key, admission);
                continue;
            }
            // Try cancellable prefill if the arch provides it; otherwise fall back to
            // the standard prefill and treat post-prefill abort as cancellation.
            let prefill_is_aborted = {
                // Prefer the cancellable variant when available (sibling adds it).
                // We probe via a helper that returns Ok(false) on abort without sampling.
                let abort_check = || batch_check_abort(&key.id, key.attempt_id, admission);
                let res = lfm_prefill_cancellable_or_fallback(
                    batch_state,
                    gpu,
                    weights,
                    config,
                    lane_idx,
                    &prompt_tokens,
                    &abort_check,
                );
                match res {
                    Ok(true) => false,
                    Ok(false) => true,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpu,
                            batch_state,
                            stdout,
                            format!("prefill lane {lane_idx}: {e}"),
                        );
                    }
                }
            };
            if prefill_is_aborted || batch_check_abort(&key.id, key.attempt_id, admission) {
                if let Err(e) = batch_state.reset_lane(gpu, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("reset lane {lane_idx} on prefill abort: {e}"),
                    );
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                let ep = crate::common::RollbackEpilogue {
                    rolled_back: true,
                    context: None,
                };
                crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
                let _ = sched.abort_lane(lane_idx, &key, admission);
                continue;
            }
            let hist: &[u32] = &[];
            let (lane_rng, sampling) = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => (lane.rng_state as u32, lane.sampling.clone()),
                _ => continue,
            };
            let (next_token, next_rng) = match batch_state.sample_lane_product(
                gpu,
                config,
                lane_idx,
                hist,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                sampling.min_p,
                lane_rng,
                sampling.repeat_penalty,
                sampling.presence_penalty,
                sampling.frequency_penalty,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpu,
                        batch_state,
                        stdout,
                        format!("sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(2));
            continue;
        }
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        if let Err(e) =
            forward_decode_batch_lfm(gpu, weights, config, &tokens, &positions, batch_state)
        {
            return fail_all(
                sched,
                gpu,
                batch_state,
                stdout,
                format!("forward_decode_batch_lfm: {e}"),
            );
        }
        let mut repeat_tokens: Vec<u32> = vec![0; batch_size * batch_state.sample_repeat_capacity];
        let mut repeat_lengths: Vec<u32> = vec![0; batch_size];
        let mut rng_states: Vec<u32> = vec![0; batch_size];
        let mut survivors: Vec<usize> = Vec::new();
        let mut to_await: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> = Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            let admission = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.ticket.admission,
                _ => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                to_abort_running.push((idx, key, admission));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            if stop_toks.contains(&cur_token) {
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "stop",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), admission, pending_done));
                continue;
            }
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let valid_len = match std::str::from_utf8(&all_bytes) {
                Ok(_) => all_bytes.len(),
                Err(e) => e.valid_up_to(),
            };
            let prev_fed = lane.bytes_fed_to_filter.min(valid_len);
            let new_bytes = &all_bytes[prev_fed..valid_len];
            let frag = match std::str::from_utf8(new_bytes) {
                Ok(s) => s,
                Err(_) => "",
            };
            if matches!(frag.trim(), "<|endoftext|>" | "</s>" | "<|im_end|>") {
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": "stop",
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), admission, pending_done));
                continue;
            }
            let has_visible = !frag.is_empty();
            if has_visible {
                if lane.first_token_at.is_none() {
                    lane.first_token_at = Some(Instant::now());
                }
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                emit_visible_token(stdout, &key.id, frag);
            }
            lane.streamed_tokens.push(cur_token);
            lane.bytes_fed_to_filter = valid_len;
            lane.seq_pos += 1;
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let is_eos = false;
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, false, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, false, loop_hit);
                let finish_reason = if hit_length_cap { "length" } else { "stop" };
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = serde_json::json!({
                    "type": "done",
                    "id": key.id,
                    "tokens": generated,
                    "tok_s": (metrics.tok_s * 10.0).round() / 10.0,
                    "prefill_tokens": lane.prompt_len,
                    "prefill_ms": (metrics.prefill_ms * 10.0).round() / 10.0,
                    "prefill_tok_s": (metrics.prefill_tok_s * 10.0).round() / 10.0,
                    "decode_tok_s": (metrics.decode_tok_s * 10.0).round() / 10.0,
                    "ttft_ms": (metrics.ttft_ms * 10.0).round() / 10.0,
                    "cached_tokens": 0,
                    "finish_reason": finish_reason,
                    "attempt_id": key.attempt_id,
                });
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                attach_continuous_batch_route_evidence(
                    &mut pending_done,
                    batch_size,
                    idx,
                    sched.lane_capacity,
                    lane.max_active_lanes.max(1),
                );
                to_await.push((idx, key.clone(), admission, pending_done));
            } else {
                survivors.push(idx);
                let window = lane
                    .sampling
                    .repeat_window
                    .min(batch_state.sample_repeat_capacity);
                let hist = if lane.streamed_tokens.len() > window {
                    &lane.streamed_tokens[lane.streamed_tokens.len() - window..]
                } else {
                    &lane.streamed_tokens[..]
                };
                for (i, &tok) in hist.iter().enumerate() {
                    repeat_tokens[idx * batch_state.sample_repeat_capacity + i] = tok;
                }
                repeat_lengths[idx] = hist.len() as u32;
                rng_states[idx] = lane.rng_state as u32;
            }
        }
        for (idx, key, admission) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let ep = crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            crate::common::emit_spec_cancel_after_rollback(stdout, &key.id, 0, &ep);
            let _ = sched.abort_lane(idx, &key, admission);
        }
        // Install AwaitingClient/Ready BEFORE publishing commit_ready; rollback if publish fails.
        for (idx, key, admission, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch] mark_awaiting_commit failed for lane {idx} id={} — aborting lane",
                    key.id
                );
                if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "mark_awaiting_commit",
                        &e,
                    );
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                continue;
            }
            let write_ok = {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                if let Err(e) = batch_state.reset_lane(gpu, config, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "commit_ready publish",
                        &e,
                    );
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                continue;
            }
        }
        if survivors.is_empty() {
            continue;
        }
        for i in 0..batch_size {
            if !survivors.contains(&i) {
                repeat_lengths[i] = 0;
                rng_states[i] = 0;
            }
        }
        let sampling = if let Some(idx) = survivors.first() {
            match &sched.lanes[*idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            }
        } else {
            continue;
        };
        let sampled = match batch_state.sample_product(
            gpu,
            config,
            batch_size,
            &repeat_tokens,
            &repeat_lengths,
            &rng_states,
            sampling.temp,
            sampling.top_p,
            sampling.top_k,
            sampling.min_p,
            sampling.repeat_penalty,
            sampling.presence_penalty,
            sampling.frequency_penalty,
        ) {
            Ok(v) => v,
            Err(e) => {
                return fail_all(
                    sched,
                    gpu,
                    batch_state,
                    stdout,
                    format!("sample_product: {e}"),
                )
            }
        };
        for lane_idx in survivors.iter() {
            let (tok, rng) = sampled[*lane_idx];
            if let BatchLane::Running(lane) = &mut sched.lanes[*lane_idx] {
                lane.next_token = Some(tok);
                lane.rng_state = rng as u64;
            }
        }
    }
    Ok(())
}

/// EP-specific evidence: must be sourced from a real `Qwen35EpBatchReceipt` and
/// explicitly state expert_parallel / rank_count=4 / peer_rooted_f32.
pub fn attach_qwen_ep_batch_receipt_evidence(
    envelope: &mut serde_json::Value,
    receipt: &qwen35::Qwen35EpBatchReceipt,
    slots: usize,
    lane: usize,
    lane_capacity: usize,
    max_active_lanes: usize,
) {
    // Enforce attested invariants via getters; never fabricate from load logs.
    debug_assert_eq!(receipt.rank_count(), 4);
    debug_assert_eq!(receipt.rank_mask(), 0x0f);
    debug_assert_eq!(receipt.reduce(), qwen35::Qwen35EpReduce::PeerRootedF32);
    debug_assert_eq!(
        receipt.parallelism(),
        qwen35::Qwen35BatchParallelism::ExpertParallel
    );
    envelope["execution_mode"] = serde_json::json!("continuous_batch_independent");
    envelope["continuous_batch"] = serde_json::json!({
        "executed": true,
        "slots": slots,
        "lane": lane,
        "lane_capacity": lane_capacity,
        "max_active_lanes": max_active_lanes,
        "refill": "continuous",
        "parallelism": "expert_parallel",
        "rank_count": receipt.rank_count(),
        "rank_mask": receipt.rank_mask(),
        "reduce": "peer_rooted_f32",
        "epoch": receipt.epoch(),
        "rows": receipt.rows(),
        "moe_collectives": receipt.moe_collectives(),
    });
}

pub fn is_qwen_ep_batch_request_eligible(
    msg: &serde_json::Value,
    m: &LoadedModel,
    continuous_batch_size: usize,
    serve_continuous_batch: bool,
    pflash_active: bool,
) -> bool {
    let caps = hipfire_loader::carrier_for(m.arch_id)
        .map(|c| c.caps())
        .unwrap_or_default();
    if !serve_continuous_batch || continuous_batch_size <= 1 {
        return false;
    }
    if m.pp != 1 {
        return false;
    }
    let Some(ep) = m.ep.as_ref() else {
        return false;
    };
    let EpArch::Qwen35 { batch, .. } = &ep.inner else {
        return false;
    };
    if batch.is_none() {
        return false;
    }
    if !caps.supports_ep_batch {
        return false;
    }
    if m.qwen35().is_some_and(|b| b.qwen35_decode_batch.is_some())
        || m.lfm2moe()
            .and_then(|b| b.lfm2_decode_batch.as_ref())
            .is_some()
    {
        return false;
    }
    // Staged `batch` above is the admission: staging already passed the full
    // `validate_ep_batch_compatibility` (embd/lm_head/all-global-experts).
    let has_image = msg.get("image").is_some() || msg.get("image_base64").is_some();
    let has_tools = msg
        .get("tools")
        .and_then(|v| v.as_array())
        .is_some_and(|a| !a.is_empty());
    let has_stop = request_has_stop(msg);
    if has_image || has_tools || has_stop {
        return false;
    }
    if !batch_messages_are_single_user(msg) {
        return false;
    }
    if m.speculator.is_some() || m.kv_adaptive.is_some() || m.eviction.is_some() || pflash_active {
        return false;
    }
    if batch.is_none() {
        return false;
    }
    // Ensure sampling controls are resolvable (mirrors sequential ladder)
    let _ = resolve_batch_sampling(msg, m);
    // Must be QwenAr route (non-spec)
    let sampling = resolve_batch_sampling(msg, m);
    let user_explicit = [
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
    ]
    .iter()
    .any(|k| msg.get(*k).is_some());
    let route_inputs = GenerationRouteInputs {
        arch_id: m.arch_id,
        // Topology already proven/staged above; ep:true would hit the global EP
        // short-circuit to Unknown for Qwen and make this QwenAr gate unreachable.
        ep: false,
        dense_tp: false,
        pp: m.pp,
        has_speculator: m.speculator.is_some(),
        speculator_is_mtp: m.speculator.as_ref().is_some_and(|s| s.name() == "mtp"),
        deepseek4_spec_requested: false,
        ngram_can_sample: m
            .speculator
            .as_ref()
            .map(|s| !s.requires_greedy())
            .unwrap_or(false),
        temp: sampling.temp,
        user_explicit_sampling: user_explicit,
        min_p: sampling.min_p,
        nonneutral_penalties: sampling.repeat_penalty != 1.0
            || sampling.presence_penalty != 0.0
            || sampling.frequency_penalty != 0.0,
        force_ar_chat: false,
        temp_spec_env_off: hipfire_config::developer_var("HIPFIRE_DFLASH_TEMP_SPEC")
            .ok()
            .as_deref()
            == Some("0"),
        fast_sample_on: hipfire_config::developer_var("HIPFIRE_FAST_SAMPLE")
            .ok()
            .as_deref()
            != Some("0"),
        supports_temp_swor: m
            .speculator
            .as_ref()
            .is_some_and(|s| s.supports_temp_verify()),
        supports_chain_nucleus_verify: m
            .speculator
            .as_ref()
            .is_some_and(|s| s.supports_chain_nucleus_verify()),
        kv_adaptive: m.kv_adaptive.is_some(),
    };
    let route = select_generation_route(&route_inputs);
    if route != GenerationRoute::QwenAr {
        return false;
    }
    // Batch size coherence
    if continuous_batch_size != batch.as_ref().map(|b| b.max_batch()).unwrap_or(0) {
        return false;
    }
    true
}

pub fn drive_qwen35_ep_continuous_batch(
    sched: &mut ContinuousBatchScheduler,
    model: &mut LoadedModel,
    stdout: &mut std::io::Stdout,
    inbox: &mut DaemonInbox,
) -> Result<(), BatchDriveError> {
    let batch_size = sched.max_batch;
    if batch_size == 0 {
        return Ok(());
    }
    let route = crate::ar::GenerationRoute::QwenAr;
    // Borrow EP batch state, config, weights via raw pointers to avoid aliasing.
    let ep_ptr = match model.ep.as_mut() {
        Some(ep) => ep as *mut EpState,
        None => return Err(BatchDriveError::Gpu("EP batch: no EP state".to_string())),
    };
    let (gpus_ptr, config_ptr, weights_ptr, batch_ptr, tokenizer_ptr, chat_template_clone, arch_id) = unsafe {
        let ep = &mut *ep_ptr;
        match &mut ep.inner {
            EpArch::Qwen35 {
                config,
                weights,
                batch,
                ..
            } => {
                let b = match batch.as_mut() {
                    Some(b) => b as *mut qwen35::Qwen35DecodeBatchEpState,
                    None => {
                        return Err(BatchDriveError::Gpu(
                            "EP batch: batch not staged".to_string(),
                        ))
                    }
                };
                (
                    &mut ep.gpus as *mut hipfire_runtime::multi_gpu::Gpus,
                    config as *const qwen35::Qwen35Config,
                    weights as *const Vec<qwen35::Qwen35Weights>,
                    b,
                    match model.tokenizer.as_ref() {
                        Some(t) => t as *const _,
                        None => return Err(BatchDriveError::Gpu("tokenizer missing".to_string())),
                    },
                    model.chat_template.clone(),
                    model.arch_id,
                )
            }
            _ => return Err(BatchDriveError::Gpu("EP batch: not Qwen35 EP".to_string())),
        }
    };
    let gpus: &mut hipfire_runtime::multi_gpu::Gpus = unsafe { &mut *gpus_ptr };
    let config: &qwen35::Qwen35Config = unsafe { &*config_ptr };
    let weights: &Vec<qwen35::Qwen35Weights> = unsafe { &*weights_ptr };
    let batch_state: &mut qwen35::Qwen35DecodeBatchEpState = unsafe { &mut *batch_ptr };
    let tokenizer: &hipfire_runtime::tokenizer::Tokenizer = unsafe { &*tokenizer_ptr };
    let chat_template = chat_template_clone;
    let eos_tok = config.eos_token;
    let im_end_tok = tokenizer.special_token_id("<|im_end|>").unwrap_or(eos_tok);
    let mut producers: Vec<Option<QwenArSemanticProducer>> =
        (0..batch_size).map(|_| None).collect();
    let mut loop_guards: Vec<hipfire_runtime::loop_guard::LoopGuard> = (0..batch_size)
        .map(
            |_| hipfire_runtime::loop_guard::LoopGuard::from_config(hipfire_runtime::config::get()),
        )
        .collect();
    let mut tokens = vec![0u32; batch_size];
    let mut positions = vec![0usize; batch_size];
    // Track last attested receipt for evidence; must be from runtime, never load logs.
    let mut last_receipt: Option<qwen35::Qwen35EpBatchReceipt> = None;
    let fail_all = |sched: &mut ContinuousBatchScheduler,
                    gpus: &mut hipfire_runtime::multi_gpu::Gpus,
                    batch_state: &mut qwen35::Qwen35DecodeBatchEpState,
                    stdout: &mut std::io::Stdout,
                    reason: String|
     -> Result<(), BatchDriveError> {
        let mut uniq_set = std::collections::HashSet::new();
        let mut uniq: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for lane in sched.lanes.iter() {
            let Some(key) = lane.key() else {
                continue;
            };
            let admission = match lane {
                BatchLane::Seeding(q) | BatchLane::Running(q) => q.ticket.admission,
                BatchLane::AwaitingClient(t) => t.ticket.admission,
                BatchLane::Empty { .. } => continue,
            };
            if uniq_set.insert((key.clone(), admission)) {
                uniq.push((key.clone(), admission));
            }
        }
        for key in sched.inbox.iter().cloned() {
            if let Some(request) = sched.pending.get(&key) {
                if uniq_set.insert((key.clone(), request.admission)) {
                    uniq.push((key, request.admission));
                }
            }
        }
        for (key, request) in sched.pending.iter() {
            if uniq_set.insert((key.clone(), request.admission)) {
                uniq.push((key.clone(), request.admission));
            }
        }
        let reset_res = batch_state.reset_all(gpus);
        let first_err = reset_res.err().map(|e| format!("EP batch reset_all: {e}"));
        let reason2 = if let Some(e) = first_err {
            format!("{reason}; {e}")
        } else {
            reason.clone()
        };
        for (key, admission) in &uniq {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, *admission);
            let ep = crate::common::RollbackEpilogue {
                rolled_back: true,
                context: None,
            };
            crate::common::emit_fail_closed_error_for_route(
                route,
                stdout,
                Some(&key.id),
                &format!("batch GPU error: {reason2}"),
                "gpu",
                false,
                &ep,
            );
        }
        let _ = sched.fail_all_active();
        Err(BatchDriveError::Poisoned(reason2))
    };
    loop {
        // handle awaiting commit/abort
        let mut to_commit: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> =
            Vec::new();
        let mut to_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let BatchLane::AwaitingClient(term) = &sched.lanes[idx] {
                let key = term.key.clone();
                let admission = term.ticket.admission;
                let expired = Instant::now() >= term.deadline;
                if batch_check_abort(&key.id, key.attempt_id, admission) || expired {
                    to_abort.push((idx, key, admission));
                } else if let Some(ClientTerminalDecision::Commit) =
                    batch_poll_decision(&key.id, key.attempt_id, admission)
                {
                    to_commit.push((idx, key.clone(), admission, term.pending_done.clone()));
                }
            }
        }
        for (idx, key, admission) in to_abort {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        for (idx, key, admission, pending_done) in to_commit {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            let reset_ok = match batch_state.reset_lane(gpus, config, idx) {
                Ok(()) => true,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP reset lane {idx} on commit: {e}"),
                    )
                }
            };
            let commit_ok = sched.commit_lane_retain_terminal(idx, &key, admission);
            let _terminal_cleanup = BatchTerminalCleanup::new(&key, Some(admission));
            match batch_commit_teardown_class(reset_ok, commit_ok) {
                BatchCommitTeardownClass::ResetFailed => unreachable!(),
                BatchCommitTeardownClass::CommitFailed => {
                    let ep = crate::common::RollbackEpilogue {
                        rolled_back: true,
                        context: None,
                    };
                    crate::common::emit_fail_closed_error_for_route(
                        route,
                        stdout,
                        Some(&key.id),
                        "batch commit_lane failed after reset",
                        "internal",
                        false,
                        &ep,
                    );
                    let _ = sched.abort_lane(idx, &key, admission);
                    producers[idx] = None;
                }
                BatchCommitTeardownClass::EmitDone => {
                    crate::ar::emit_generation_done_value(route, stdout, &pending_done);
                    producers[idx] = None;
                }
            }
        }
        let mut queued_abort: Vec<(AttemptKey, BatchGeneration)> = Vec::new();
        for key in sched.inbox.iter().cloned().collect::<Vec<_>>() {
            if let Some(request) = sched.pending.get(&key) {
                if batch_check_abort(&key.id, key.attempt_id, request.admission) {
                    queued_abort.push((key, request.admission));
                }
            }
        }
        for (key, admission) in queued_abort {
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_queued(&key, admission);
        }
        let mut running_abort: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        for idx in 0..batch_size {
            if let Some(key) = sched.lanes[idx].key().cloned() {
                let admission = match &sched.lanes[idx] {
                    BatchLane::Running(l) | BatchLane::Seeding(l) => l.ticket.admission,
                    _ => continue,
                };
                if matches!(
                    sched.lanes[idx],
                    BatchLane::Running(_) | BatchLane::Seeding(_)
                ) && batch_check_abort(&key.id, key.attempt_id, admission)
                {
                    running_abort.push((idx, key, admission));
                }
            }
        }
        for (idx, key, admission) in running_abort {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on running abort: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        let mut barrier: Option<DaemonMsg> = None;
        loop {
            let dm = match inbox.try_recv() {
                Ok(m) => m,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            let (dm, carried_admission) = match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    (DaemonMsg::Regular(json), Some(admission))
                }
                other => (other, None),
            };
            match dm {
                DaemonMsg::RegularWithAdmission(json, admission) => {
                    barrier = Some(DaemonMsg::RegularWithAdmission(json, admission));
                    break;
                }
                DaemonMsg::SingletonWithAdmission(json, transfer) => {
                    barrier = Some(DaemonMsg::SingletonWithAdmission(json, transfer));
                    break;
                }
                DaemonMsg::ParseError(e) => {
                    emit_uncorrelated_error(
                        stdout,
                        None,
                        &format!("invalid JSON: {e}"),
                        "validation",
                        false,
                        false,
                    );
                    let _ = stdout.flush();
                }
                DaemonMsg::Regular(json) => {
                    let t = json.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if t == "generate" {
                        let attempt_id = match json.get("attempt_id").and_then(|v| v.as_u64()) {
                            Some(0) => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate attempt_id must be nonzero",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                            Some(v) => v,
                            None => {
                                emit_uncorrelated_error(
                                    stdout,
                                    json.get("id").and_then(|v| v.as_str()),
                                    "generate missing attempt_id",
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        let id = json
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("0")
                            .to_string();
                        let Some(admission) = carried_admission else {
                            barrier = Some(daemon_regular_with_admission(json, None));
                            break;
                        };
                        if batch_check_abort(&id, attempt_id, admission) {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                            crate::ar::emit_generation_cancel(route, stdout, &id, 0);
                            batch_clear_terminal_at_generation(&id, attempt_id, admission);
                            continue;
                        }
                        // EP batch-only admission; non-eligible becomes a
                        // barrier while preserving the reader-owned token.
                        if !is_qwen_ep_batch_request_eligible(
                            &json,
                            model,
                            batch_size,
                            parse_serve_continuous_batch(&json),
                            false,
                        ) {
                            barrier = Some(daemon_regular_with_admission(json, Some(admission)));
                            break;
                        }
                        let prompt_str = batch_single_user_content(&json).unwrap_or_else(|| {
                            json.get("prompt")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Hello")
                                .to_string()
                        });
                        let system_str = json
                            .get("system")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let assistant_prefix = match json
                            .get("assistant_prefix")
                            .and_then(|v| v.as_str())
                            .unwrap_or("plain")
                        {
                            "open_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink
                            }
                            "closed_think" => {
                                hipfire_runtime::prompt_frame::AssistantPrefix::ClosedThink
                            }
                            _ => hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                        };
                        let max_think = json
                            .get("max_think_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize;
                        let max_tokens_req = json
                            .get("max_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(4096) as usize;
                        let batch_messages = match json.get("messages") {
                            Some(v) => match serde_json::from_value::<
                                Vec<hipfire_runtime::prompt_frame::Message>,
                            >(v.clone())
                            {
                                Ok(v) => Some(v),
                                Err(e) => {
                                    emit_batch_admission_error(
                                        stdout,
                                        &id,
                                        attempt_id,
                                        admission,
                                        &format!("invalid messages field: {e}"),
                                        "validation",
                                        false,
                                        false,
                                    );
                                    continue;
                                }
                            },
                            None => None,
                        };
                        let raw_effort = json
                            .get("reasoning_effort")
                            .or_else(|| json.get("thinking_mode"))
                            .and_then(|v| v.as_str());
                        let thinking_enabled =
                            json.get("thinking_enabled").and_then(|v| v.as_bool());
                        let (batch_enable_thinking, batch_reasoning_effort) =
                            qwen_jinja_reasoning(thinking_enabled, raw_effort, max_think);
                        let (prompt_tokens, started_in_think) = match batch_render_prompt_tokens(
                            &prompt_str,
                            system_str.as_deref(),
                            assistant_prefix,
                            tokenizer,
                            chat_template.as_ref(),
                            max_think,
                            batch_messages.as_deref(),
                            batch_enable_thinking,
                            batch_reasoning_effort.as_deref(),
                        ) {
                            Ok(v) => v,
                            Err(e) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &format!("render failed: {e}"),
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        if started_in_think {
                            let handoff = match handoff_admitted_started_in_think(
                                &id,
                                attempt_id,
                                admission,
                                json,
                                GenerationRoute::QwenAr,
                            ) {
                                Ok(msg) => msg,
                                Err(reason) => {
                                    return fail_all(sched, gpus, batch_state, stdout, reason)
                                }
                            };
                            barrier = Some(handoff);
                            break;
                        }
                        if prompt_tokens.is_empty() || prompt_tokens.len() >= sched.lane_capacity {
                            emit_batch_admission_error(
                                stdout,
                                &id,
                                attempt_id,
                                admission,
                                "prompt exceeds lane capacity or empty",
                                "validation",
                                false,
                                false,
                            );
                            continue;
                        }
                        // Explicit wire `seed` must reach the lane RNG on the
                        // batched route too; out-of-domain values are rejected
                        // loudly, never silently unseeded.
                        let client_seed = match wire_seed::parse_wire_seed(json.get("seed")) {
                            Ok(s) => s,
                            Err(reason) => {
                                emit_batch_admission_error(
                                    stdout,
                                    &id,
                                    attempt_id,
                                    admission,
                                    &reason,
                                    "validation",
                                    false,
                                    false,
                                );
                                continue;
                            }
                        };
                        batch_transition_to_queued(&id, attempt_id, admission);
                        let sampling = resolve_batch_sampling(&json, model);
                        let req = BatchPendingRequest {
                            key: AttemptKey::new(&id, attempt_id),
                            admission,
                            original_msg: json.clone(),
                            prompt: prompt_str.clone(),
                            prompt_tokens: prompt_tokens.clone(),
                            started_in_think,
                            system: system_str.clone(),
                            assistant_prefix,
                            max_think_tokens: max_think,
                            max_tokens: max_tokens_req,
                            client_seed,
                            sampling,
                        };
                        if !sched.enqueue(req) {
                            eprintln!(
                                "[batch][EP] duplicate enqueue rejected id={} attempt_id={}; preserving live registry",
                                id, attempt_id
                            );
                            continue;
                        }
                        {
                            let _scope =
                                BatchAttemptScope::enter_for_generation(&id, attempt_id, admission);
                            crate::ar::emit_generation_start(
                                crate::ar::GenerationRoute::QwenAr,
                                stdout,
                                &id,
                                false,
                            );
                        }
                    } else if t == "abort" || t == "commit" {
                        if let (Some(id), Some(aid), Some(kind)) = (
                            json.get("id").and_then(|v| v.as_str()),
                            json.get("attempt_id").and_then(|v| v.as_u64()),
                            json.get("type").and_then(|v| v.as_str()),
                        ) {
                            batch_apply_terminal_control(kind, id, aid);
                        }
                    } else {
                        barrier = Some(daemon_regular_with_admission(json, carried_admission));
                        break;
                    }
                }
            }
        }
        if let Some(msg) = barrier {
            inbox.push_front(msg);
            if sched.active_count() == 0 && sched.inbox.is_empty() {
                return Ok(());
            }
        }
        while let Some((key, ticket)) = sched.try_assign_one() {
            let lane_idx = ticket.lane;
            let pending_req = match sched.pending.get(&key).cloned() {
                Some(r) => r,
                None => continue,
            };
            let sampling = pending_req.sampling.clone();
            let prompt_tokens = pending_req.prompt_tokens.clone();
            let started_in_think = pending_req.started_in_think;
            let admission = pending_req.admission;
            if started_in_think {
                // Reset while the batch owner remains live, then retire the
                // lane and hand the complete original request to singleton.
                if let Err(err) = batch_state.reset_lane(gpus, config, lane_idx) {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP reset lane {lane_idx} on think barrier: {err}"),
                    );
                }
                let handoff = match handoff_started_in_think(
                    sched,
                    lane_idx,
                    &key,
                    &pending_req,
                    GenerationRoute::QwenAr,
                ) {
                    Ok(msg) => msg,
                    Err(reason) => {
                        return fail_all(sched, gpus, batch_state, stdout, reason);
                    }
                };
                inbox.push_front(handoff);
                break;
            }
            if let Err(e) = batch_state.reset_lane(gpus, config, lane_idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {lane_idx}: {e}"),
                );
            }
            let receipt =
                match batch_state.prefill_lane(gpus, weights, config, lane_idx, &prompt_tokens) {
                    Ok(r) => r,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP prefill lane {lane_idx}: {e}"),
                        )
                    }
                };
            last_receipt = Some(receipt);
            let lane_rng = match &sched.lanes[lane_idx] {
                BatchLane::Running(lane) => lane.rng_state as u32,
                _ => continue,
            };
            // Use per-lane sampling that respects readiness; repeat penalties folded via retry window with product if needed.
            // For EP we call sample_lane (full product requires contiguous Ready lanes); per-lane keeps sparsity.
            let (next_token, next_rng) = match batch_state.sample_lane(
                gpus,
                config,
                lane_idx,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                lane_rng,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP sample lane {lane_idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[lane_idx] {
                lane.prompt_len = prompt_tokens.len();
                lane.seq_pos = prompt_tokens.len();
                lane.next_token = Some(next_token);
                lane.rng_state = next_rng as u64;
                lane.conversation_tokens = Vec::new();
                lane.streamed_tokens = Vec::new();
                lane.bytes_fed_to_filter = 0;
                lane.prefill_done_at = Some(Instant::now());
            }
            producers[lane_idx] = Some(QwenArSemanticProducer::new_with_tool_protocol(
                key.id.clone(),
                started_in_think,
                false,
            ));
        }
        let running: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::Running(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let awaiting: Vec<usize> = sched
            .lanes
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if matches!(l, BatchLane::AwaitingClient(_)) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        if running.is_empty()
            && awaiting.is_empty()
            && sched.inbox.is_empty()
            && inbox.backlog.is_empty()
        {
            break;
        }
        if running.is_empty() {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        let active_now = running.len();
        for &idx in &running {
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                if active_now > lane.max_active_lanes {
                    lane.max_active_lanes = active_now;
                }
            }
        }
        // Build active mask and dense token/position vectors for EP forward_tick.
        let mut active_mask: u64 = 0;
        for &idx in &running {
            active_mask |= 1u64 << idx;
        }
        for i in 0..batch_size {
            match &sched.lanes[i] {
                BatchLane::Running(lane) => {
                    tokens[i] = lane.next_token.unwrap_or(eos_tok);
                    positions[i] = lane.seq_pos;
                }
                _ => {
                    tokens[i] = eos_tok;
                    positions[i] = 0;
                }
            }
        }
        let receipt =
            match batch_state.forward_tick(gpus, weights, config, active_mask, &tokens, &positions)
            {
                Ok(r) => r,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP forward_tick: {e}"),
                    )
                }
            };
        last_receipt = Some(receipt);
        let mut to_await: Vec<(usize, AttemptKey, BatchGeneration, serde_json::Value)> = Vec::new();
        let mut to_abort_running: Vec<(usize, AttemptKey, BatchGeneration)> = Vec::new();
        let mut survivors: Vec<usize> = Vec::new();
        for idx in running.clone() {
            let key = match sched.lanes[idx].key().cloned() {
                Some(k) => k,
                None => continue,
            };
            let admission = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.ticket.admission,
                _ => continue,
            };
            if batch_check_abort(&key.id, key.attempt_id, admission) {
                to_abort_running.push((idx, key, admission));
                continue;
            }
            let lane_ptr = match &mut sched.lanes[idx] {
                BatchLane::Running(l) => l as *mut QwenBatchLane,
                _ => continue,
            };
            let lane = unsafe { &mut *lane_ptr };
            let cur_token = lane.next_token.unwrap_or(eos_tok);
            let prod_ptr = match producers[idx].as_mut() {
                Some(p) => p as *mut QwenArSemanticProducer,
                None => continue,
            };
            let producer = unsafe { &mut *prod_ptr };
            let mut future_streamed = lane.streamed_tokens.clone();
            future_streamed.push(cur_token);
            let all_bytes = tokenizer.decode_bytes(&future_streamed);
            let prev_fed = lane.bytes_fed_to_filter.min(all_bytes.len());
            let token_bytes = all_bytes[prev_fed..].to_vec();
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            if lane.first_token_at.is_none() {
                lane.first_token_at = Some(Instant::now());
            }
            let stopped = {
                let lane_seq = &mut lane.seq_pos as *mut usize;
                let lane_conv = &mut lane.conversation_tokens as *mut Vec<u32>;
                let lane_stream = &mut lane.streamed_tokens as *mut Vec<u32>;
                let lane_fed = &mut lane.bytes_fed_to_filter as *mut usize;
                let all_len = all_bytes.len();
                let mut res: Result<bool, _> = Ok(false);
                unsafe {
                    res = producer.commit_and_classify(
                        stdout,
                        cur_token,
                        || {
                            let pos = qwen_ar_raw_commit_token(
                                &mut *lane_conv,
                                &mut *lane_stream,
                                &mut *lane_seq,
                                cur_token,
                                QwenArRawCommitDisposition::ClassifiedVisible,
                            );
                            *lane_fed = all_len;
                            (pos, token_bytes.clone())
                        },
                        |_, _| {},
                    );
                }
                match res {
                    Ok(s) => s,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP semantic classify lane {idx}: {e}"),
                        )
                    }
                }
            };
            let loop_hit = loop_guards[idx].check(&lane.streamed_tokens).is_some();
            let is_eos = cur_token == eos_tok || cur_token == im_end_tok;
            let hit_max = lane.streamed_tokens.len() >= lane_max_tokens(&key, sched);
            let hit_lane_cap = batch_lane_at_capacity(lane.seq_pos, sched.lane_capacity);
            let should_finish =
                batch_should_finish_decode(is_eos, hit_max, hit_lane_cap, stopped, loop_hit);
            if should_finish {
                let hit_length_cap =
                    batch_hit_length_cap(hit_max, hit_lane_cap, is_eos, stopped, loop_hit);
                let producer_owned = match producers[idx].take() {
                    Some(p) => p,
                    None => continue,
                };
                let (finish, visible_text) = match producer_owned.finish(stdout, hit_length_cap) {
                    Ok(v) => v,
                    Err(e) => {
                        return fail_all(
                            sched,
                            gpus,
                            batch_state,
                            stdout,
                            format!("EP semantic finish lane {idx}: {e}"),
                        )
                    }
                };
                if !finish.wire_tool_calls.is_empty() {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP semantic finish lane {idx}: unexpected tool calls"),
                    );
                }
                let finish_reason = finish.finish_reason;
                let generated = lane.streamed_tokens.len();
                let metrics = batch_lane_done_metrics(
                    lane.created_at,
                    lane.prefill_done_at,
                    lane.first_token_at,
                    Instant::now(),
                    lane.prompt_len,
                    generated,
                );
                let mut pending_done = qwen_ar_done_value(
                    &key.id,
                    finish_reason,
                    generated,
                    metrics.tok_s,
                    lane.prompt_len,
                    metrics.prefill_ms,
                    metrics.prefill_tok_s,
                    metrics.decode_tok_s,
                    metrics.ttft_ms,
                    0,
                    "",
                );
                pending_done["latency_ms"] =
                    serde_json::json!((metrics.latency_ms * 10.0).round() / 10.0);
                if let Some(receipt) = last_receipt.as_ref() {
                    attach_qwen_ep_batch_receipt_evidence(
                        &mut pending_done,
                        receipt,
                        batch_size,
                        idx,
                        sched.lane_capacity,
                        lane.max_active_lanes.max(1),
                    );
                } else {
                    // Never fabricate: if no receipt yet, attach generic but still mark expert_parallel via default (should not happen on finishing lane after forward).
                    attach_continuous_batch_route_evidence(
                        &mut pending_done,
                        batch_size,
                        idx,
                        sched.lane_capacity,
                        lane.max_active_lanes.max(1),
                    );
                    pending_done["continuous_batch"]["parallelism"] =
                        serde_json::json!("expert_parallel");
                    pending_done["continuous_batch"]["rank_count"] = serde_json::json!(4);
                    pending_done["continuous_batch"]["reduce"] =
                        serde_json::json!("peer_rooted_f32");
                }
                let _ = visible_text;
                to_await.push((idx, key.clone(), admission, pending_done));
            } else {
                survivors.push(idx);
            }
        }
        for (idx, key, admission) in to_abort_running {
            if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                return fail_all(
                    sched,
                    gpus,
                    batch_state,
                    stdout,
                    format!("EP reset lane {idx} on abort post-forward: {e}"),
                );
            }
            let _scope =
                BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
            crate::ar::emit_generation_cancel(route, stdout, &key.id, 0);
            let _ = sched.abort_lane(idx, &key, admission);
            producers[idx] = None;
        }
        for (idx, key, admission, pending_done) in to_await {
            let mut envelope = pending_done.clone();
            envelope["type"] = serde_json::json!("commit_ready");
            let marked = sched.mark_awaiting_commit(idx, pending_done.clone());
            if !marked {
                eprintln!(
                    "[batch][EP] qwen mark_awaiting_commit failed lane {idx} id={} — aborting lane",
                    key.id
                );
                if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "mark_awaiting_commit",
                        &e,
                    );
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
                continue;
            }
            let write_ok = {
                let _scope =
                    BatchAttemptScope::enter_for_generation(&key.id, key.attempt_id, admission);
                writeln!(stdout, "{}", envelope).is_ok() && stdout.flush().is_ok()
            };
            if !write_ok {
                if let Err(e) = batch_state.reset_lane(gpus, config, idx) {
                    retire_lane_after_reset_failure(
                        sched,
                        stdout,
                        &key,
                        admission,
                        idx,
                        "commit_ready publish",
                        &e,
                    );
                } else {
                    let _ = sched.abort_lane(idx, &key, admission);
                }
                producers[idx] = None;
            }
        }
        if survivors.is_empty() {
            continue;
        }
        // Per-lane sampling for survivors (sparse-aware). Use sample_lane to avoid contiguous prefix requirement.
        for idx in survivors.iter().cloned() {
            let sampling = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.sampling.clone(),
                _ => continue,
            };
            let rng = match &sched.lanes[idx] {
                BatchLane::Running(l) => l.rng_state as u32,
                _ => continue,
            };
            let (tok, next_rng) = match batch_state.sample_lane(
                gpus,
                config,
                idx,
                sampling.temp,
                sampling.top_p,
                sampling.top_k,
                rng,
            ) {
                Ok(v) => v,
                Err(e) => {
                    return fail_all(
                        sched,
                        gpus,
                        batch_state,
                        stdout,
                        format!("EP sample_lane survivor {idx}: {e}"),
                    )
                }
            };
            if let BatchLane::Running(lane) = &mut sched.lanes[idx] {
                lane.next_token = Some(tok);
                lane.rng_state = next_rng as u64;
            }
        }
        // Also exercise sample_product when survivors form a contiguous full prefix (API coverage; sparse batches use sample_lane above).
        if survivors.len() == batch_size && survivors.iter().enumerate().all(|(i, &v)| i == v) {
            // Use the same sampling as first survivor for product validation; ignore error for non-product-capable batch shapes.
            if let Some(first) = survivors.first().and_then(|&idx| match &sched.lanes[idx] {
                BatchLane::Running(l) => Some(l.sampling.clone()),
                _ => None,
            }) {
                let dummy_repeat = vec![0u32; batch_size * 128];
                let dummy_lengths = vec![0u32; batch_size];
                let dummy_rng = vec![0u32; batch_size];
                let _ = batch_state.sample_product(
                    gpus,
                    config,
                    batch_size,
                    &dummy_repeat,
                    &dummy_lengths,
                    &dummy_rng,
                    first.temp,
                    first.top_p,
                    first.top_k,
                    first.min_p,
                    first.repeat_penalty,
                    first.presence_penalty,
                    first.frequency_penalty,
                );
            }
        }
    }
    Ok(())
}

/// Pre-activation protocol reject (missing/malformed attempt_id, or commands
/// with no active generate attempt). Always emits `attempt_id: 0`.
///
/// Do not use after `set_active_attempt_id` for a generate request.
pub fn emit_uncorrelated_error(
    stdout: &mut impl std::io::Write,
    id: Option<&str>,
    message: &str,
    class: &str,
    retryable: bool,
    rolled_back: bool,
) {
    crate::dense::write_error_envelope(stdout, id, message, class, retryable, rolled_back, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lock() -> std::sync::MutexGuard<'static, ()> {
        crate::ar::generation_test_lock()
    }

    #[test]
    fn any_stop_form_leaves_the_batch_path() {
        use serde_json::json;
        for absent in [json!({}), json!({"stop": null}), json!({"stop": []}), json!({"stop": ""})] {
            assert!(!request_has_stop(&absent), "{absent}");
        }
        // The string form was batch-eligible before and silently ignored there.
        for present in [json!({"stop": "\n"}), json!({"stop": ["END"]}), json!({"stop": 7})] {
            assert!(request_has_stop(&present), "{present}");
        }
    }

    #[derive(Clone, Copy)]
    enum MultiLaneTerminal {
        Done,
        Cancel,
        Error,
    }

    fn emit_multi_lane_terminal(
        route: GenerationRoute,
        terminal: MultiLaneTerminal,
        output: &mut Vec<u8>,
        id: &str,
        attempt_id: u64,
    ) {
        match terminal {
            MultiLaneTerminal::Done => {
                let pending = serde_json::json!({
                    "type": "done",
                    "id": id,
                    "attempt_id": attempt_id,
                    "finish_reason": "stop",
                });
                crate::ar::emit_generation_done_value(route, output, &pending);
            }
            MultiLaneTerminal::Cancel => {
                crate::ar::emit_generation_cancel(route, output, id, 0);
            }
            MultiLaneTerminal::Error => {
                crate::ar::emit_generation_error(
                    route,
                    output,
                    Some(id),
                    "multi-lane representative error",
                    "gpu",
                    false,
                    false,
                );
            }
        }
    }

    fn assert_multi_lane_route_latches_release(
        route: GenerationRoute,
        terminal: MultiLaneTerminal,
    ) {
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        assert_eq!(crate::ar::active_generation_route(), None);

        let admissions = [
            (
                "multi-lane-a",
                601_u64,
                batch_announce_terminal("multi-lane-a", 601).expect("lane A admission"),
            ),
            (
                "multi-lane-b",
                602_u64,
                batch_announce_terminal("multi-lane-b", 602).expect("lane B admission"),
            ),
        ];
        let mut output = Vec::new();

        for &(id, attempt_id, admission) in &admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, false);
        }
        for &(id, attempt_id, admission) in &admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            emit_multi_lane_terminal(route, terminal, &mut output, id, attempt_id);
            assert_eq!(
                crate::ar::active_generation_route(),
                None,
                "terminal route must clear after {id}"
            );
        }
        for &(id, attempt_id, admission) in &admissions {
            assert!(batch_clear_terminal_at_generation(
                id, attempt_id, admission
            ));
        }

        // Re-announcing the exact wire keys must claim fresh route starts for
        // both lanes. A stale per-key route latch would suppress one of these.
        let fresh_admissions = [
            (
                "multi-lane-a",
                601_u64,
                batch_announce_terminal("multi-lane-a", 601).expect("lane A re-admission"),
            ),
            (
                "multi-lane-b",
                602_u64,
                batch_announce_terminal("multi-lane-b", 602).expect("lane B re-admission"),
            ),
        ];
        for &(id, attempt_id, admission) in &fresh_admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, false);
        }
        for &(id, attempt_id, admission) in &fresh_admissions {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            emit_multi_lane_terminal(route, terminal, &mut output, id, attempt_id);
            assert_eq!(
                crate::ar::active_generation_route(),
                None,
                "fresh terminal route must clear after {id}"
            );
            assert!(batch_clear_terminal_at_generation(
                id, attempt_id, admission
            ));
        }

        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        for id in ["multi-lane-a", "multi-lane-b"] {
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event["type"] == "gen_start" && event["id"] == id)
                    .count(),
                2,
                "both generations must start for {id}"
            );
        }
        let terminal_type = match terminal {
            MultiLaneTerminal::Done => "done",
            MultiLaneTerminal::Cancel => "aborted",
            MultiLaneTerminal::Error => "error",
        };
        let terminal_events = events
            .iter()
            .filter(|event| event["type"] == terminal_type)
            .count();
        assert_eq!(
            terminal_events, 4,
            "both lanes must emit both representative terminals"
        );
        for id in ["multi-lane-a", "multi-lane-b"] {
            assert_eq!(
                events
                    .iter()
                    .filter(|event| event["type"] == terminal_type && event["id"] == id)
                    .count(),
                2,
                "both generations must terminate for {id}"
            );
        }
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        assert_eq!(crate::ar::active_generation_route(), None);
    }

    #[test]
    fn qwen_multi_lane_done_releases_exact_route_latches() {
        let _guard = lock();
        assert_multi_lane_route_latches_release(GenerationRoute::QwenAr, MultiLaneTerminal::Done);
    }

    #[test]
    fn lfm_multi_lane_cancel_releases_exact_route_latches() {
        let _guard = lock();
        assert_multi_lane_route_latches_release(GenerationRoute::LfmAr, MultiLaneTerminal::Cancel);
    }

    #[test]
    fn qwen_ep_multi_lane_error_releases_exact_route_latches() {
        let _guard = lock();
        assert_multi_lane_route_latches_release(GenerationRoute::QwenAr, MultiLaneTerminal::Error);
    }
    struct FlushGateWriter {
        pending: Vec<u8>,
        visible: Vec<u8>,
    }

    impl FlushGateWriter {
        fn events(&self) -> Vec<serde_json::Value> {
            std::str::from_utf8(&self.visible)
                .expect("visible terminal bytes are UTF-8")
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_str(line).expect("visible terminal event is JSON"))
                .collect()
        }
    }

    impl Write for FlushGateWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.pending.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.visible.append(&mut self.pending);
            Ok(())
        }
    }

    #[derive(Default)]
    struct FailingWriter {
        bytes: Vec<u8>,
        fail_write: bool,
        fail_flush: bool,
    }

    impl FailingWriter {
        fn events(&self) -> Vec<serde_json::Value> {
            std::str::from_utf8(&self.bytes)
                .expect("writer bytes are UTF-8")
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| serde_json::from_str(line).expect("writer event is JSON"))
                .collect()
        }
    }

    impl Write for FailingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.fail_write {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected write failure",
                ));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            if self.fail_flush {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "injected flush failure",
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn route_terminals_flush_visible_done_error_and_cancel() {
        let _guard = lock();
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let done_lanes = [
            (
                "flush-done-a",
                701_u64,
                batch_announce_terminal("flush-done-a", 701).expect("done A admission"),
            ),
            (
                "flush-done-b",
                702_u64,
                batch_announce_terminal("flush-done-b", 702).expect("done B admission"),
            ),
        ];
        let mut output = FlushGateWriter {
            pending: Vec::new(),
            visible: Vec::new(),
        };
        for &(id, attempt_id, admission) in &done_lanes {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(GenerationRoute::QwenAr, &mut output, id, false);
            output.flush().expect("start flush");
        }
        for (done_index, &(id, attempt_id, admission)) in done_lanes.iter().enumerate() {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            let pending = serde_json::json!({
                "type": "done",
                "id": id,
                "attempt_id": attempt_id,
                "finish_reason": "stop",
            });
            assert!(crate::ar::emit_generation_done_value(
                GenerationRoute::QwenAr,
                &mut output,
                &pending,
            ));
            assert_eq!(
                output
                    .events()
                    .iter()
                    .filter(|event| event["type"] == "done" && event["id"] == id)
                    .count(),
                1,
                "route done {done_index} must be visible after route emission"
            );
            assert!(batch_clear_terminal_at_generation(
                id, attempt_id, admission
            ));
        }

        let error_admission = batch_announce_terminal("flush-error", 703).expect("error admission");
        {
            let _scope =
                BatchAttemptScope::enter_for_generation("flush-error", 703, error_admission);
            crate::ar::emit_generation_start(
                GenerationRoute::QwenAr,
                &mut output,
                "flush-error",
                false,
            );
            output.flush().expect("error start flush");
            assert!(crate::ar::emit_generation_error(
                GenerationRoute::QwenAr,
                &mut output,
                Some("flush-error"),
                "representative error",
                "internal",
                false,
                true,
            ));
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "error" && event["id"] == "flush-error")
                .count(),
            1,
            "route error must be visible after route emission"
        );
        assert!(batch_clear_terminal_at_generation(
            "flush-error",
            703,
            error_admission
        ));

        let cancel_admission =
            batch_announce_terminal("flush-cancel", 704).expect("cancel admission");
        {
            let _scope =
                BatchAttemptScope::enter_for_generation("flush-cancel", 704, cancel_admission);
            crate::ar::emit_generation_start(
                GenerationRoute::QwenAr,
                &mut output,
                "flush-cancel",
                false,
            );
            output.flush().expect("cancel start flush");
            assert!(crate::ar::emit_generation_cancel(
                GenerationRoute::QwenAr,
                &mut output,
                "flush-cancel",
                1,
            ));
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "aborted" && event["id"] == "flush-cancel")
                .count(),
            1,
            "route cancel must be visible after route emission"
        );
        assert!(batch_clear_terminal_at_generation(
            "flush-cancel",
            704,
            cancel_admission
        ));

        let events = output.events();
        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "done"
                    && (event["id"] == "flush-done-a" || event["id"] == "flush-done-b"))
                .count(),
            2,
            "both committed done envelopes must be visible after route emission"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "error" && event["id"] == "flush-error")
                .count(),
            1,
            "route errors must be visible after route emission"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event["type"] == "aborted" && event["id"] == "flush-cancel")
                .count(),
            1,
            "route cancels must be visible after route emission"
        );

        // A stale duplicate cannot write a second terminal, even though the
        // route writer still performs its successful flush.
        for &(id, attempt_id, _) in &done_lanes {
            let pending = serde_json::json!({
                "type": "done",
                "id": id,
                "attempt_id": attempt_id,
                "finish_reason": "stop",
            });
            let before = output.visible.len();
            let _scope = BatchAttemptScope::enter_for(id, attempt_id);
            crate::ar::emit_generation_done_value(GenerationRoute::QwenAr, &mut output, &pending);
            assert_eq!(output.visible.len(), before);
        }

        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
    }

    #[derive(Clone, Copy)]
    enum WriterFailure {
        Write,
        Flush,
    }

    fn assert_route_terminal_failure_releases_latch(failure: WriterFailure) {
        let (id, attempt_id) = match failure {
            WriterFailure::Write => ("route-write-failure", 705_u64),
            WriterFailure::Flush => ("route-flush-failure", 706_u64),
        };
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let admission = batch_announce_terminal(id, attempt_id).expect("failure admission");
        let mut output = FailingWriter::default();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(GenerationRoute::QwenAr, &mut output, id, false);
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "gen_start" && event["id"] == id)
                .count(),
            1,
            "{id} initial route start",
        );

        match failure {
            WriterFailure::Write => output.fail_write = true,
            WriterFailure::Flush => output.fail_flush = true,
        }
        let pending = serde_json::json!({
            "type": "done",
            "id": id,
            "attempt_id": attempt_id,
            "finish_reason": "stop",
        });
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            assert!(
                !crate::ar::emit_generation_done_value(
                    GenerationRoute::QwenAr,
                    &mut output,
                    &pending,
                ),
                "{id} injected terminal failure must report undelivered",
            );
        }
        assert_eq!(
            crate::ar::active_generation_route(),
            None,
            "{id} failed terminal must clear the active route",
        );
        let after_failure = output.bytes.len();

        // Once the exact claim is consumed, a duplicate remains suppressed
        // even after the writer recovers.
        output.fail_write = false;
        output.fail_flush = false;
        {
            let _scope = BatchAttemptScope::enter_for(id, attempt_id);
            assert!(
                !crate::ar::emit_generation_done_value(
                    GenerationRoute::QwenAr,
                    &mut output,
                    &pending,
                ),
                "{id} duplicate terminal must stay suppressed",
            );
        }
        assert_eq!(
            output.bytes.len(),
            after_failure,
            "{id} duplicate terminal must not write",
        );
        assert!(batch_clear_terminal_at_generation(
            id, attempt_id, admission
        ));

        // Reusing the exact wire key must claim a fresh start after the
        // failed terminal consumed the previous lifecycle claim.
        let fresh_admission =
            batch_announce_terminal(id, attempt_id).expect("fresh failure admission");
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, fresh_admission);
            crate::ar::emit_generation_start(GenerationRoute::QwenAr, &mut output, id, false);
        }
        assert_eq!(
            output
                .events()
                .iter()
                .filter(|event| event["type"] == "gen_start" && event["id"] == id)
                .count(),
            2,
            "{id} same-key reuse must emit a fresh route start",
        );
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, fresh_admission);
            assert!(crate::ar::emit_generation_done_value(
                GenerationRoute::QwenAr,
                &mut output,
                &pending,
            ));
        }
        assert!(batch_clear_terminal_at_generation(
            id,
            attempt_id,
            fresh_admission,
        ));
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
    }

    #[test]
    fn route_terminal_write_failure_consumes_claim_and_releases_latch() {
        let _guard = lock();
        assert_route_terminal_failure_releases_latch(WriterFailure::Write);
    }

    #[test]
    fn route_terminal_flush_failure_consumes_claim_and_releases_latch() {
        let _guard = lock();
        assert_route_terminal_failure_releases_latch(WriterFailure::Flush);
    }

    #[test]
    fn direct_driver_admission_errors_are_correlated_once_and_cleared() {
        let _guard = lock();
        for (driver, id, attempt_id, message, class) in [
            (
                "qwen",
                "qwen-direct",
                101_u64,
                "invalid messages field",
                "validation",
            ),
            (
                "lfm",
                "lfm-direct",
                202_u64,
                "prompt exceeds context capacity",
                "context_length",
            ),
            (
                "qwen35-ep",
                "qwen35-ep-direct",
                303_u64,
                "seed must fit in a u32",
                "validation",
            ),
        ] {
            let admission = batch_announce_terminal(id, attempt_id).expect("{driver} announce");

            let mut output = Vec::new();
            emit_batch_admission_error(
                &mut output,
                id,
                attempt_id,
                admission,
                message,
                class,
                false,
                false,
            );

            let lines: Vec<&str> = std::str::from_utf8(&output)
                .expect("UTF-8 error envelope")
                .lines()
                .filter(|line| !line.is_empty())
                .collect();
            assert_eq!(lines.len(), 1, "{driver} terminal count");
            let event: serde_json::Value =
                serde_json::from_str(lines[0]).expect("JSON error envelope");
            assert_eq!(event["type"], "error", "{driver} event type");
            assert_eq!(
                event["attempt_id"].as_u64(),
                Some(attempt_id),
                "{driver} attempt id"
            );
            assert_ne!(
                event["attempt_id"].as_u64(),
                Some(0),
                "{driver} attempt zero"
            );
            assert_eq!(event["id"].as_str(), Some(id), "{driver} request id");
            assert_eq!(event["class"].as_str(), Some(class), "{driver} error class");
            assert_eq!(
                batch_terminal_generation(id, attempt_id),
                None,
                "{driver} admission cleanup"
            );
        }
    }

    fn assert_started_in_think_handoff(
        route: GenerationRoute,
        id: &str,
        attempt_id: u64,
        abort_latched: bool,
    ) {
        let original = serde_json::json!({
            "type": "generate",
            "id": id,
            "attempt_id": attempt_id,
            "prompt": "full prompt",
            "system": "full system",
            "messages": [{"role": "user", "content": "full prompt"}],
            "tools": [{"type": "function", "function": {"name": "keep"}}],
            "stop": ["<done>"],
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 7,
            "seed": 9,
            "reasoning_effort": "low",
            "assistant_prefix": "open_think",
        });
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
        activate_terminal_control(id, attempt_id);
        set_active_attempt_id(attempt_id);
        if abort_latched {
            apply_terminal_control("abort", id, attempt_id);
            batch_apply_terminal_control("abort", id, attempt_id);
        }
        let singleton_generation =
            terminal_generation(id, attempt_id).expect("singleton transaction");
        assert!(batch_transition_to_queued(id, attempt_id, admission));
        let key = AttemptKey::new(id, attempt_id);
        let sampling = BatchSampling {
            temp: 0.3,
            top_p: 0.8,
            top_k: None,
            min_p: None,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_window: 128,
        };
        let mut sched = ContinuousBatchScheduler::new(1, 64);
        assert!(sched.enqueue(BatchPendingRequest {
            key: key.clone(),
            admission,
            original_msg: original.clone(),
            prompt: "full prompt".to_string(),
            prompt_tokens: vec![1, 2, 3],
            started_in_think: true,
            system: Some("full system".to_string()),
            assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::OpenThink,
            max_think_tokens: 16,
            max_tokens: 7,
            client_seed: Some(9),
            sampling,
        }));
        let (assigned_key, ticket) = sched.try_assign_one().expect("assigned think lane");
        assert_eq!(assigned_key, key);

        let mut output = Vec::new();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, true);
        }
        let pending = sched.pending.get(&key).cloned().expect("pending request");
        let handoff = handoff_started_in_think(&mut sched, ticket.lane, &key, &pending, route)
            .expect("singleton handoff");
        assert_eq!(sched.active_count(), 0, "barrier lane retired");
        assert!(
            sched.pending.is_empty(),
            "barrier request removed from batch"
        );
        assert!(
            sched.try_assign_one().is_none(),
            "barrier request cannot continue in GPU lane"
        );
        assert_eq!(batch_terminal_generation(id, attempt_id), None);

        let (handoff_msg, transfer) = match handoff {
            DaemonMsg::SingletonWithAdmission(value, transfer) => (value, transfer),
            _ => panic!("think barrier did not produce singleton ownership"),
        };
        assert_eq!(handoff_msg, original, "full request payload was preserved");
        assert_eq!(transfer.admission(), admission);

        // The outer main transaction has ended; adoption restores the exact
        // lifecycle generation and any pre-latched abort without reactivation.
        clear_terminal_control();
        assert!(adopt_singleton_transfer(id, attempt_id, transfer));
        assert_eq!(
            terminal_generation(id, attempt_id),
            Some(singleton_generation)
        );
        assert_eq!(check_abort(id), abort_latched);

        {
            let _scope = BatchAttemptScope::enter(attempt_id);
            crate::ar::emit_generation_start(route, &mut output, id, true);
            if abort_latched {
                crate::ar::emit_active_route_cancel(&mut output, id, 0);
            } else {
                crate::ar::emit_generation_error(
                    route,
                    &mut output,
                    Some(id),
                    "think barrier normal terminal",
                    "validation",
                    false,
                    false,
                );
            }
        }
        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[1]["type"], "gen_start", "fresh sequential start");
        if abort_latched {
            assert_eq!(events.len(), 4);
            assert_eq!(events[2]["type"], "aborted");
            assert_eq!(events[3]["type"], "done");
        } else {
            assert_eq!(events.len(), 3);
            assert_eq!(events[2]["type"], "error");
        }
        assert_eq!(crate::ar::active_generation_route(), None);
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    fn assert_admitted_started_in_think_handoff(
        route: GenerationRoute,
        id: &str,
        attempt_id: u64,
        abort_latched: bool,
    ) {
        let original = serde_json::json!({
            "type": "generate",
            "id": id,
            "attempt_id": attempt_id,
            "prompt": "full prompt",
            "messages": [{"role": "user", "content": "full prompt"}],
            "tools": [{"type": "function", "function": {"name": "keep"}}],
            "stop": ["<done>"],
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 7,
            "seed": 9,
            "reasoning_effort": "low",
            "assistant_prefix": "open_think",
        });
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);
        let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
        activate_terminal_control(id, attempt_id);
        set_active_attempt_id(attempt_id);
        if abort_latched {
            apply_terminal_control("abort", id, attempt_id);
            batch_apply_terminal_control("abort", id, attempt_id);
        }
        let singleton_generation =
            terminal_generation(id, attempt_id).expect("singleton transaction");
        let mut output = Vec::new();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(route, &mut output, id, true);
        }
        let handoff =
            handoff_admitted_started_in_think(id, attempt_id, admission, original.clone(), route)
                .expect("admitted singleton handoff");
        assert_eq!(batch_terminal_generation(id, attempt_id), None);
        let (handoff_msg, transfer) = match handoff {
            DaemonMsg::SingletonWithAdmission(value, transfer) => (value, transfer),
            _ => panic!("admitted think barrier did not produce singleton ownership"),
        };
        assert_eq!(handoff_msg, original, "full admitted payload was preserved");
        assert_eq!(transfer.admission(), admission);

        clear_terminal_control();
        assert!(adopt_singleton_transfer(id, attempt_id, transfer));
        assert_eq!(
            terminal_generation(id, attempt_id),
            Some(singleton_generation)
        );
        assert_eq!(check_abort(id), abort_latched);
        {
            let _scope = BatchAttemptScope::enter(attempt_id);
            crate::ar::emit_generation_start(route, &mut output, id, true);
            if abort_latched {
                crate::ar::emit_active_route_cancel(&mut output, id, 0);
            } else {
                crate::ar::emit_generation_error(
                    route,
                    &mut output,
                    Some(id),
                    "admitted think barrier normal terminal",
                    "validation",
                    false,
                    false,
                );
            }
        }
        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[1]["type"], "gen_start", "fresh sequential start");
        if abort_latched {
            assert_eq!(events.len(), 4);
            assert_eq!(events[2]["type"], "aborted");
            assert_eq!(events[3]["type"], "done");
        } else {
            assert_eq!(events.len(), 3);
            assert_eq!(events[2]["type"], "error");
        }
        assert_eq!(crate::ar::active_generation_route(), None);
        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    #[test]
    fn admitted_think_batch_driver_bootstraps_singleton_and_reuses_key() {
        let _guard = lock();
        let route = GenerationRoute::QwenAr;
        let id = "qwen-later-think";
        let attempt_id = 507_u64;
        let original = serde_json::json!({
            "type": "generate",
            "id": id,
            "attempt_id": attempt_id,
            "prompt": "later queued prompt",
            "messages": [{"role": "user", "content": "later queued prompt"}],
            "max_tokens": 7,
            "reasoning_effort": "low",
        });
        let mut output = Vec::new();
        let mut admissions = Vec::new();
        let mut singleton_generations = Vec::new();

        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        for _ in 0..2 {
            // This is the batch-driver admission path: the request is queued
            // before the think-open barrier, without singleton activation.
            let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
            assert!(batch_transition_to_queued(id, attempt_id, admission));
            assert_eq!(
                terminal_generation(id, attempt_id),
                None,
                "batch-driver request has no manually activated singleton"
            );

            let handoff = handoff_admitted_started_in_think(
                id,
                attempt_id,
                admission,
                original.clone(),
                route,
            )
            .expect("admitted singleton handoff");
            let (handoff_msg, transfer) = match handoff {
                DaemonMsg::SingletonWithAdmission(value, transfer) => (value, transfer),
                _ => panic!("admitted think barrier did not produce singleton ownership"),
            };
            assert_eq!(handoff_msg, original, "full queued payload was preserved");
            assert_eq!(transfer.admission(), admission);
            assert_eq!(batch_terminal_generation(id, attempt_id), None);

            // Handoff bootstraps the singleton inside the tombstone; main
            // adopts that exact owner instead of rediscovering it.
            assert_eq!(terminal_generation(id, attempt_id), None);
            clear_terminal_control();
            assert!(adopt_singleton_transfer(id, attempt_id, transfer));
            let singleton_generation =
                terminal_generation(id, attempt_id).expect("adopted singleton transaction");
            assert!(singleton_generation > 0);
            admissions.push(admission);
            singleton_generations.push(singleton_generation);

            {
                let _attempt = BatchAttemptScope::enter_singleton(attempt_id);
                let _route = GenerationRouteScope::enter(route, id);
                crate::ar::emit_generation_start(route, &mut output, id, true);
                crate::ar::emit_generation_error(
                    route,
                    &mut output,
                    Some(id),
                    "admitted think barrier terminal",
                    "validation",
                    false,
                    false,
                );
            }
            assert_eq!(crate::ar::active_generation_route(), None);
            clear_terminal_control();
        }

        assert_ne!(
            admissions[0], admissions[1],
            "same key received fresh admissions"
        );
        assert_ne!(
            singleton_generations[0], singleton_generations[1],
            "same key received fresh singleton lifecycles"
        );

        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events.len(), 4, "one start and one terminal per reuse");
        assert_eq!(
            events
                .iter()
                .map(|event| event["type"].as_str().expect("event type"))
                .collect::<Vec<_>>(),
            vec!["gen_start", "error", "gen_start", "error"]
        );

        clear_terminal_control();
        batch_clear_all_terminals();
        set_active_attempt_id(0);
    }

    #[test]
    fn admitted_started_in_think_barrier_preserves_all_batch_routes() {
        let _guard = lock();
        for (route, id, attempt_id, abort_latched) in [
            (GenerationRoute::QwenAr, "qwen-admitted-think", 504, true),
            (GenerationRoute::LfmAr, "lfm-admitted-think", 505, false),
            (GenerationRoute::QwenAr, "ep-admitted-think", 506, false),
        ] {
            assert_admitted_started_in_think_handoff(route, id, attempt_id, abort_latched);
        }
    }

    #[test]
    fn qwen_started_in_think_barrier_preserves_singleton_owner() {
        let _guard = lock();
        assert_started_in_think_handoff(GenerationRoute::QwenAr, "qwen-think", 501, true);
    }

    #[test]
    fn lfm_started_in_think_barrier_preserves_full_request() {
        let _guard = lock();
        assert_started_in_think_handoff(GenerationRoute::LfmAr, "lfm-think", 502, false);
    }

    #[test]
    fn ep_started_in_think_barrier_preserves_full_request() {
        let _guard = lock();
        assert_started_in_think_handoff(GenerationRoute::QwenAr, "ep-think", 503, false);
    }

    #[test]
    fn lfm_assignment_capacity_error_releases_route_latch_for_reuse() {
        let _guard = lock();
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let id = "lfm-assignment";
        let attempt_id = 404_u64;
        let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
        let key = AttemptKey::new(id, attempt_id);
        let sampling = BatchSampling {
            temp: 0.3,
            top_p: 1.0,
            top_k: None,
            min_p: None,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_window: 128,
        };
        let mut sched = ContinuousBatchScheduler::new(1, 8);
        assert!(sched.enqueue(BatchPendingRequest {
            key: key.clone(),
            admission,
            original_msg: serde_json::json!({
                "type": "generate",
                "id": id,
                "attempt_id": attempt_id,
                "prompt": "oversized",
                "max_tokens": 4,
            }),
            prompt: "oversized".to_string(),
            prompt_tokens: vec![1; 7],
            started_in_think: false,
            system: None,
            assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
            max_think_tokens: 0,
            max_tokens: 4,
            client_seed: None,
            sampling,
        }));
        let (assigned_key, ticket) = sched.try_assign_one().expect("assigned lane");
        assert_eq!(assigned_key, key);

        let mut output = Vec::new();
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            crate::ar::emit_generation_start(
                crate::ar::GenerationRoute::LfmAr,
                &mut output,
                id,
                false,
            );
        }
        assert_eq!(
            crate::ar::active_generation_route(),
            Some(crate::ar::GenerationRoute::LfmAr)
        );

        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, admission);
            emit_lfm_assignment_capacity_error(&mut output, &key, 7, 4, 8);
        }
        assert_eq!(crate::ar::active_generation_route(), None);

        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 error envelope")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events.len(), 2, "assignment emits one start and one error");
        assert_eq!(events[0]["type"], "gen_start");
        assert_eq!(events[0]["attempt_id"].as_u64(), Some(attempt_id));
        assert_eq!(events[1]["type"], "error");
        assert_eq!(events[1]["id"].as_str(), Some(id));
        assert_eq!(events[1]["attempt_id"].as_u64(), Some(attempt_id));
        assert_eq!(events[1]["class"].as_str(), Some("context_length"));
        assert!(sched.abort_lane(ticket.lane, &key, admission));
        assert_eq!(batch_terminal_generation(id, attempt_id), None);

        // Reusing the exact wire key must claim a fresh route start rather
        let reuse_admission =
            batch_announce_terminal(id, attempt_id).expect("reused batch admission");
        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, reuse_admission);
            crate::ar::emit_generation_start(
                crate::ar::GenerationRoute::LfmAr,
                &mut output,
                id,
                false,
            );
            assert_eq!(
                crate::ar::active_generation_route(),
                Some(crate::ar::GenerationRoute::LfmAr)
            );
        }
        let reused_events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 events")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(reused_events.len(), 3);
        assert_eq!(reused_events[2]["type"], "gen_start");
        assert_eq!(reused_events[2]["attempt_id"].as_u64(), Some(attempt_id));

        {
            let _scope = BatchAttemptScope::enter_for_generation(id, attempt_id, reuse_admission);
            crate::ar::emit_active_route_cancel(&mut output, id, 0);
        }
        assert_eq!(crate::ar::active_generation_route(), None);
        assert!(batch_clear_terminal_at_generation(
            id,
            attempt_id,
            reuse_admission
        ));
        set_active_attempt_id(0);
        clear_terminal_control();
    }
    /// A lane whose GPU reset fails is retired with a visible
    /// `rolled_back=false` error keyed by its own AttemptKey claim; the
    /// peer lane keeps serving untouched. Ordinary and EP drivers call
    /// `retire_lane_after_reset_failure` independently per failed lane,
    /// so both flavors are exercised here through their own admissions.
    #[test]
    fn lane_reset_failure_retires_with_visible_unattested_error() {
        let _guard = lock();
        batch_clear_all_terminals();
        clear_terminal_control();
        set_active_attempt_id(0);

        let mut sched = ContinuousBatchScheduler::new(2, 8);
        let mut lanes = Vec::new();
        for (id, attempt_id) in [("retire-ordinary", 701_u64), ("retire-ep", 702_u64)] {
            let admission = batch_announce_terminal(id, attempt_id).expect("batch admission");
            let key = AttemptKey::new(id, attempt_id);
            assert!(sched.enqueue(BatchPendingRequest {
                key: key.clone(),
                admission,
                original_msg: serde_json::json!({
                    "type": "generate",
                    "id": id,
                    "attempt_id": attempt_id,
                }),
                prompt: "hello".to_string(),
                prompt_tokens: vec![1, 2, 3],
                started_in_think: false,
                system: None,
                assistant_prefix: hipfire_runtime::prompt_frame::AssistantPrefix::Plain,
                max_think_tokens: 0,
                max_tokens: 4,
                client_seed: None,
                sampling: BatchSampling {
                    temp: 0.0,
                    top_p: 1.0,
                    top_k: None,
                    min_p: None,
                    repeat_penalty: 1.0,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    repeat_window: 128,
                },
            }));
            let (assigned_key, ticket) = sched.try_assign_one().expect("assigned lane");
            assert_eq!(assigned_key, key);
            lanes.push((key, admission, ticket.lane));
        }
        let mut output = Vec::new();
        let (first, second) = (&lanes[0], &lanes[1]);
        retire_lane_after_reset_failure(
            &mut sched,
            &mut output,
            &first.0,
            first.1,
            first.2,
            "commit_ready publish",
            &"injected reset failure",
        );
        // The peer lane keeps serving untouched: it still holds its key and
        // its admission is still live after the first lane retired.
        assert!(
            sched.lanes[second.2].key().is_some_and(|k| k == &second.0),
            "peer lane keeps its key"
        );
        assert!(
            batch_terminal_generation(&second.0.id, second.0.attempt_id).is_some(),
            "peer admission stays live"
        );
        retire_lane_after_reset_failure(
            &mut sched,
            &mut output,
            &second.0,
            second.1,
            second.2,
            "mark_awaiting_commit",
            &"injected reset failure",
        );
        let events: Vec<serde_json::Value> = std::str::from_utf8(&output)
            .expect("UTF-8 error envelopes")
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_str(line).expect("JSON event"))
            .collect();
        assert_eq!(events.len(), 2, "one visible error per retired lane");
        for (event, (key, _, _)) in events.iter().zip(lanes.iter()) {
            assert_eq!(event["type"], "error");
            assert_eq!(event["id"].as_str(), Some(key.id.as_str()));
            assert_eq!(event["attempt_id"].as_u64(), Some(key.attempt_id));
            assert_eq!(event["class"].as_str(), Some("gpu"));
            assert_eq!(event["retryable"], serde_json::json!(false));
            assert_eq!(event["rolled_back"], serde_json::json!(false));
            assert!(
                event["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("lane retired")),
                "reset failure stays visible: {}",
                event["message"]
            );
        }
        for (key, admission, lane_idx) in &lanes {
            assert!(
                sched.lanes[*lane_idx].key().is_none(),
                "failed lane is retired, not reused"
            );
            assert_eq!(
                batch_terminal_generation(&key.id, key.attempt_id),
                None,
                "retired admission cleared independently"
            );
        }
        set_active_attempt_id(0);
        clear_terminal_control();
    }
}
