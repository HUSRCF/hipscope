// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Multi-slot concurrent engine.
//!
//! Concern: concurrent in-process `SlotEngine` backend, its tokenizer, and
//! the `complete_request_slots` fast path. Behind the `multi-slot` Cargo feature;
//! isolates the only arch-specific crate (`hipfire-arch-qwen35`) so the CLI
//! can be built arch-free with `--no-default-features`.

use crate::serve::complete::Completion;
#[cfg(feature = "multi-slot")]
use anyhow::{anyhow, bail, Result};
#[cfg(not(feature = "multi-slot"))]
use anyhow::{bail, Result};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame, Role};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::serve::{Event, SubmitRequest};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::tokenizer::Tokenizer;
use serde_json;
#[cfg(feature = "multi-slot")]
use std::sync::Mutex;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{self, RecvTimeoutError},
    Arc,
};
use std::time::Duration;

#[cfg(feature = "multi-slot")]
pub(crate) struct SlotBackend {
    pub(crate) engine: hipfire_arch_qwen35::serve_engine::SlotEngine,
    pub(crate) tokenizer: hipfire_runtime::tokenizer::Tokenizer,
    /// Model's Jinja chat template from HFQ metadata. `None` falls back to
    /// the hand-rolled ChatML frame (no tools rendering).
    pub(crate) chat_template: Option<String>,
    /// Whether the qwen35 tool-call grammar applies to this model
    /// (`qwen35_grammar_on`, resolved once at engine start). Mirrors the
    /// daemon: when off, `tools` are withheld from the emitter so the grammar
    /// stays inactive while the tool-protocol router still parses.
    pub(crate) tool_grammar: bool,
    /// Per-model configuration, for the reasoning projection this path shares
    /// with the daemon path (`reasoning.effort` / `reasoning.budget`).
    pub(crate) resolved: hipfire_config::ResolvedConfig,
    pub(crate) pending_tools: Mutex<Vec<PendingToolTurn>>,
}

/// A session's unanswered tool calls, as the client saw them.
///
/// The engine cannot resolve a tool-result turn by itself: the ids it would
/// have to match on (`call_0`, `call_1`, …) are per-response indices, so two
/// sessions that each emitted one call carry identical ids. This path minted
/// those ids, so it is the only place that can say WHICH session a set of
/// results answers — and refuse when more than one candidate fits.
#[cfg(feature = "multi-slot")]
pub(crate) struct PendingToolTurn {
    pub(crate) session: u64,
    pub(crate) convo: Vec<u64>,
    pub(crate) calls: Vec<ToolCallKey>,
}

#[cfg(feature = "multi-slot")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ToolCallKey {
    pub(crate) id: String,
    pub(crate) name: String,
    /// Canonical JSON of the call arguments, so a client that re-serializes
    /// them still compares equal.
    pub(crate) arguments: String,
}

/// Sessions whose last turn ended in unanswered tool calls. One entry per
/// session; the bound is a backstop against a client that never answers.
#[cfg(feature = "multi-slot")]
const MAX_PENDING_TOOL_TURNS: usize = 64;

#[cfg(not(feature = "multi-slot"))]
pub(crate) struct SlotBackend;

#[cfg(feature = "multi-slot")]
pub(crate) fn complete_request_slots(
    backend: &SlotBackend,
    body: &serde_json::Value,
    contract: &crate::serve::complete::RequestContract,
    identity: &(String, u64),
    cancelled: Option<&AtomicBool>,
    event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    use hipfire_arch_qwen35::spec_emit::Qwen35Emit;
    use hipfire_runtime::prompt_frame::{
        AssistantPrefix, ChatFrame, JinjaChatFrame, Message, Role, ToolCall,
    };
    use hipfire_runtime::serve::{Continuation, Event, SubmitRequest};
    use hipfire_runtime::spec::{ClientEvent, SpecEmit, SpecEmitCtx};

    let model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let max_tokens = contract.max_tokens as usize;

    let Some(msgs) = contract.messages.as_array() else {
        bail!("messages is required");
    };
    let mut messages: Vec<Message> = Vec::with_capacity(msgs.len() + 1);
    for m in msgs {
        let role = match m.get("role").and_then(serde_json::Value::as_str) {
            Some("system") => Role::System,
            Some("assistant") => Role::Assistant,
            Some("tool") => Role::Tool,
            _ => Role::User,
        };
        let content = m
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_owned();
        let tool_calls: Vec<ToolCall> = m
            .get("tool_calls")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|tc| {
                        let f = tc.get("function")?;
                        let name = f.get("name")?.as_str()?.to_owned();
                        // OpenAI carries arguments as a JSON-encoded STRING;
                        // the template renders the decoded object.
                        let arguments = match f.get("arguments") {
                            Some(serde_json::Value::String(s)) => serde_json::from_str(s)
                                .unwrap_or(serde_json::Value::String(s.clone())),
                            Some(v) => v.clone(),
                            None => serde_json::Value::Object(serde_json::Map::new()),
                        };
                        Some(ToolCall {
                            id: tc
                                .get("id")
                                .and_then(serde_json::Value::as_str)
                                .map(str::to_owned),
                            name,
                            arguments,
                            rendered_body: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        messages.push(Message {
            role,
            content,
            reasoning_content: None,
            name: m
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            rendered_name: None,
            tool_calls,
            tool_call_id: m
                .get("tool_call_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            tool_plan: String::new(),
        });
    }
    if !messages.iter().any(|m| m.role == Role::User) {
        bail!("messages must contain at least one user message");
    }

    // Same projection the daemon path runs: the request's effort (or, absent
    // one, the configured `reasoning.effort` / `reasoning.budget`) lowered into
    // the daemon's own fields. Reading those back is what keeps a `high` or
    // `max` request from arriving here as the hardcoded default level.
    let mut lowered = serde_json::json!({});
    crate::apply_http_reasoning_request(body, &backend.resolved, &mut lowered, false)?;
    let effort = lowered
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str);
    // `max_think_tokens == 1` is the engine's no-thinking sentinel.
    let reasoning_off = lowered
        .get("max_think_tokens")
        .and_then(serde_json::Value::as_u64)
        == Some(1);
    let has_think = backend.tokenizer.special_token_id("<think>").is_some();
    let enable_thinking = has_think && !reasoning_off;
    let think_mode = think_mode_for(effort, enable_thinking);

    let tools: Option<Vec<serde_json::Value>> = contract
        .forwarded_tools
        .as_ref()
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
        .cloned();

    let (prompt_tokens, started_in_think) = match backend.chat_template.as_deref() {
        Some(template) => {
            let frame = JinjaChatFrame {
                tokenizer: &backend.tokenizer,
                template,
                system: None,
                user: "",
                enable_thinking,
                bos_token: None,
                reasoning_strength: None,
                reasoning_effort: effort,
            };
            let rendered = frame
                .render_messages(&messages, tools.as_deref(), None)
                .map_err(|e| anyhow!("multi_slot jinja render: {e}"))?;
            let sit = rendered.trim_end().ends_with("<think>");
            (backend.tokenizer.encode(&rendered), sit)
        }
        None => {
            // No template in the HFQ: hand-rolled ChatML, no tools rendering.
            if tools.is_some() {
                bail!("multi_slot: model has no chat template; tools are not supported");
            }
            let mut system: Option<String> = None;
            let mut turns: Vec<(Role, String)> = Vec::new();
            for m in &messages {
                match m.role {
                    Role::System => match system.as_mut() {
                        Some(existing) => {
                            existing.push('\n');
                            existing.push_str(&m.content);
                        }
                        None => system = Some(m.content.clone()),
                    },
                    Role::Assistant => turns.push((Role::Assistant, m.content.clone())),
                    // A tool result reads as user-side context to the model.
                    _ => turns.push((Role::User, m.content.clone())),
                }
            }
            let last_user = match turns.iter().rposition(|(r, _)| *r == Role::User) {
                Some(i) => turns.remove(i).1,
                None => bail!("messages must contain at least one user message"),
            };
            let history: Vec<(Role, &str)> = turns.iter().map(|(r, t)| (*r, t.as_str())).collect();
            let prefix = if enable_thinking {
                AssistantPrefix::OpenThink
            } else if has_think {
                AssistantPrefix::ClosedThink
            } else {
                AssistantPrefix::Plain
            };
            let frame = ChatFrame {
                tokenizer: &backend.tokenizer,
                system: system.as_deref(),
                user: &last_user,
                assistant_prefix: prefix,
                raw: false,
            };
            (
                frame.build_multi_turn(&history),
                matches!(prefix, AssistantPrefix::OpenThink),
            )
        }
    };
    // The continuation suffix must open the assistant turn exactly the way the
    // full render does, or turn 2 diverges from what a cold render of the same
    // history would produce.
    let prefix = if started_in_think {
        AssistantPrefix::OpenThink
    } else if has_think {
        AssistantPrefix::ClosedThink
    } else {
        AssistantPrefix::Plain
    };

    // Conversation identity is the USER turns. The assistant side is whatever
    // we generated, and the client's echo of it may differ (reasoning split to
    // its own channel, whitespace, an edited message), so it cannot be part of
    // the key.
    let convo: Vec<u64> = messages
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| turn_hash(&m.content))
        .collect();

    // Tokens that continue the previous assistant turn into this one. The
    // engine appends these to the session's exact stored tokens instead of
    // re-rendering the history, which is what makes the result a strict
    // extension of the KV. Two shapes, and the engine is told WHICH: a new user
    // turn, or a run of tool results answering one specific session's calls.
    let continuation = match messages.last().map(|m| m.role) {
        Some(Role::User) if convo.len() >= 2 => {
            Continuation::UserTurn(hipfire_runtime::prompt_frame::continuation_suffix(
                &backend.tokenizer,
                messages.last().map(|m| m.content.as_str()).unwrap_or(""),
                prefix,
            ))
        }
        Some(Role::Tool) => {
            let resolved = tool_result_tail(&messages).and_then(|tail| {
                let pending = backend
                    .pending_tools
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                resolve_pending_session(&pending, &convo, &tail.answered)
                    .map(|session| (session, tail))
            });
            match resolved {
                Some((session, tail)) => Continuation::ToolResults {
                    tokens: hipfire_runtime::prompt_frame::continuation_suffix_tool_results(
                        &backend.tokenizer,
                        &tail.results,
                        prefix,
                    ),
                    session,
                },
                None => Continuation::Cold,
            }
        }
        _ => Continuation::Cold,
    };

    let (tx, rx) = mpsc::channel::<Event>();
    backend
        .engine
        .submit(SubmitRequest {
            prompt_tokens,
            convo: convo.clone(),
            continuation,
            max_tokens,
            // Absent sampling fields mean greedy, NOT an implicit creative
            // default: clients that care (hag sends per-model values) say so.
            temperature: body
                .get("temperature")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0) as f32,
            top_p: body
                .get("top_p")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0) as f32,
            top_k: body
                .get("top_k")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0) as i32,
            seed: {
                let client_seed = hipfire_engine::wire_seed::parse_wire_seed(body.get("seed"))
                    .map_err(|e| anyhow!("seed: {e}"))?;
                let key = hipfire_engine::terminal::AttemptKey::new(&identity.0, identity.1);
                hipfire_engine::scheduler::request_seed_for(&key, client_seed)
            },
            reply: tx,
        })
        .map_err(|e| anyhow!("multi_slot submit: {e}"))?;

    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut finish = "stop";
    let mut cached_tokens = 0usize;
    let mut prefill_tokens = 0usize;
    let mut generated_tokens = 0usize;
    let mut session_id: Option<u64> = None;

    // The daemon's own emission aggregate (EosFilter UTF-8/EOT authority +
    // think routing + tool-protocol routing + post-hoc tool grammar) drives
    // the slots path too, so both backends classify output identically.
    // Slots-specific configuration: no stop sequences yet, and max_think=0 --
    // a think force-close would need to inject `</think>` tokens into the
    // slot's KV mid-decode (`take_forced`), which the engine does not support.
    let mut emit: Option<Box<dyn SpecEmit + '_>> = Some(Qwen35Emit::from_ctx(SpecEmitCtx {
        tokenizer: &backend.tokenizer,
        eos: backend.tokenizer.eos_id,
        im_end: backend.tokenizer.special_token_id("<|im_end|>"),
        tools: if backend.tool_grammar {
            tools.as_deref()
        } else {
            None
        },
        stop: Vec::new(),
        max_think: 0,
        max_tokens,
        assistant_prefix: prefix,
        think_mode,
        decoded_vocab: None,
    }));
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    let render = |events: Vec<ClientEvent>,
                  content: &mut String,
                  reasoning_content: &mut String,
                  tool_calls: &mut Vec<ToolCall>,
                  cb: &mut dyn FnMut(
        &serde_json::Value,
    ) -> Result<(), hipfire_client::ClientError>| {
        for ev in events {
            match ev {
                ClientEvent::Token(text) => {
                    content.push_str(&text);
                    let _ = cb(&serde_json::json!({ "type": "token", "text": text }));
                }
                ClientEvent::Reasoning(text) => {
                    reasoning_content.push_str(&text);
                    let _ = cb(&serde_json::json!({ "type": "reasoning", "text": text }));
                }
                ClientEvent::ToolCalls(calls) => tool_calls.extend(calls),
                ClientEvent::Committed { .. } => {}
            }
        }
    };

    let mut first_token = true;
    loop {
        if cancelled.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            return Err(anyhow::Error::new(hipfire_client::ClientError::Cancelled));
        }
        let ev = match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        match ev {
            Event::Accepted {
                session,
                reused,
                prefill,
            } => {
                session_id = Some(session);
                cached_tokens = reused;
                prefill_tokens = prefill;
            }
            Event::Token { id } => {
                let outcome = {
                    let e = emit.as_mut().expect("emitter live until finish");
                    if first_token {
                        first_token = false;
                        e.begin(id)
                    } else {
                        e.observe(id)
                    }
                };
                render(
                    outcome.events,
                    &mut content,
                    &mut reasoning_content,
                    &mut tool_calls,
                    event_callback,
                );
                if outcome.stop.is_some() {
                    // Grammar violation or an in-stream stop marker ends the
                    // turn for the client. Dropping `rx` surfaces to the engine
                    // as ClientGone, which closes the session -- the slots
                    // analog of the daemon's forced reset after a violation.
                    generated_tokens = emit
                        .as_ref()
                        .expect("emitter live until finish")
                        .streamed_tokens()
                        .len();
                    break;
                }
            }
            Event::Done { reason, generated } => {
                finish = match reason {
                    hipfire_runtime::serve::DoneReason::MaxTokens => "length",
                    _ => "stop",
                };
                generated_tokens = generated;
                break;
            }
            Event::Rejected { reason } => {
                // Saturation is a real, expected answer here, not a crash: all
                // slots busy means the caller should retry, not that anything
                // failed.
                bail!("multi_slot rejected: {reason}");
            }
        }
    }

    let summary = emit.take().expect("emitter live until finish").finish();
    render(
        summary.events,
        &mut content,
        &mut reasoning_content,
        &mut tool_calls,
        event_callback,
    );
    // The emitter's verdict wins except on a pure token-cap exit, where the
    // OpenAI contract says `length` (unless complete tool calls were parsed,
    // which the daemon also reports as `tool_calls`).
    if summary.tool_calls > 0 && !tool_calls.is_empty() {
        finish = "tool_calls";
    } else if finish != "length" {
        finish = summary.finish_reason;
    }

    if let Some(session) = session_id {
        record_pending_tools(backend, session, &convo, &tool_calls);
    }

    // Apply tool_choice terminal postconditions before the Completion is built.
    // Now that the multi-slot path emits real structured tool_calls via the
    // shared emitter, this is load-bearing (none/required/specific fail-closed
    // like the daemon path). `contract.forwarded_tools` is the projected source
    // of truth for schemas (withheld under tool_choice none), never body["tools"].
    let mut done = serde_json::json!({
        "finish_reason": finish,
        "cached_tokens": cached_tokens,
        "prefill_tokens": prefill_tokens,
        "tokens": generated_tokens,
    });
    crate::serve::complete::finalize_tool_calls_for_choice(
        &contract.tool_choice_policy,
        &mut tool_calls,
        &mut done,
        body,
    )?;
    let completion = Completion {
        id: identity.0.clone(),
        created: identity.1,
        model,
        content,
        reasoning_content,
        preserve_thinking: false,
        tool_calls,
        done,
        logprobs: None,
    };
    // The terminal callback is what stages the response body and signals the
    // HTTP handler that the request succeeded. Skipping it leaves the handler
    // waiting on a status that never arrives, which surfaces to the client as
    // "generation worker disconnected".
    terminal_callback(&completion).map_err(|e| anyhow!("terminal callback: {e}"))?;
    Ok(completion)
}

/// FNV-1a over a user turn's text. Conversation identity is the user turns
/// alone — see `Session::convo`.
#[cfg(feature = "multi-slot")]
pub(crate) fn turn_hash(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for b in s.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The tool results this request feeds back, paired with the calls they answer.
#[cfg(feature = "multi-slot")]
pub(crate) struct ToolResultTail {
    /// The assistant's calls, in the order the client answered them.
    pub(crate) answered: Vec<ToolCallKey>,
    pub(crate) results: Vec<String>,
}

/// Read a trailing run of tool results, or refuse.
///
/// Refuses (→ cold prefill) unless every result carries a `tool_call_id` and
/// those ids are exactly the preceding assistant turn's calls, in order. A
/// partial or reordered set cannot be pinned onto a session's KV: the
/// `<tool_response>` blocks are appended positionally, so answering call 2
/// before call 1 — or answering only one of two — puts the model's own calls
/// and their results out of correspondence with no way to notice downstream.
#[cfg(feature = "multi-slot")]
pub(crate) fn tool_result_tail(
    messages: &[hipfire_runtime::prompt_frame::Message],
) -> Option<ToolResultTail> {
    use hipfire_runtime::prompt_frame::Role;

    let head = messages.iter().rposition(|m| m.role != Role::Tool)?;
    let tail = &messages[head + 1..];
    if tail.is_empty() {
        return None;
    }
    let assistant = &messages[head];
    if assistant.role != Role::Assistant || assistant.tool_calls.is_empty() {
        return None;
    }
    let mut answered = Vec::with_capacity(tail.len());
    let mut results = Vec::with_capacity(tail.len());
    for (result, call) in tail.iter().zip(assistant.tool_calls.iter()) {
        let (id, call_id) = (result.tool_call_id.as_deref()?, call.id.as_deref()?);
        if id != call_id {
            return None;
        }
        answered.push(ToolCallKey {
            id: id.to_owned(),
            name: call.name.clone(),
            arguments: canonical_arguments(&call.arguments),
        });
        results.push(result.content.clone());
    }
    (answered.len() == assistant.tool_calls.len()).then_some(ToolResultTail { answered, results })
}

#[cfg(feature = "multi-slot")]
fn canonical_arguments(arguments: &serde_json::Value) -> String {
    serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_owned())
}

/// The one session whose unanswered calls these results answer, or `None`.
///
/// `None` on no match AND on several: the ids are per-response indices, so two
/// sessions on the same conversation can carry the same ones, and picking
/// either would splice tool results into a stranger's KV. A cold prefill costs
/// prompt-processing time; the wrong session costs a wrong answer.
#[cfg(feature = "multi-slot")]
pub(crate) fn resolve_pending_session(
    pending: &[PendingToolTurn],
    convo: &[u64],
    answered: &[ToolCallKey],
) -> Option<u64> {
    let mut hit = None;
    for entry in pending
        .iter()
        .filter(|entry| entry.convo == convo && entry.calls == answered)
    {
        if hit.is_some() {
            return None;
        }
        hit = Some(entry.session);
    }
    hit
}

/// Effort level for the emitter. `None` is an undefined `reasoning_effort`,
/// which is the model template's own default rather than "no reasoning" — the
/// turn is still a thinking turn, so it must not collapse to `NonThink`.
#[cfg(feature = "multi-slot")]
pub(crate) fn think_mode_for(
    effort: Option<&str>,
    enable_thinking: bool,
) -> hipfire_runtime::prompt_frame::ThinkMode {
    use hipfire_runtime::prompt_frame::ThinkMode;
    if !enable_thinking {
        return ThinkMode::NonThink;
    }
    effort.map(ThinkMode::from_str).unwrap_or(ThinkMode::Low)
}

/// Remember (or forget) what this session left unanswered.
///
/// The ids recorded here are the ones the OpenAI adapter mints for the
/// response, so the next turn's `tool_call_id`s compare against what the
/// client actually received.
#[cfg(feature = "multi-slot")]
fn record_pending_tools(
    backend: &SlotBackend,
    session: u64,
    convo: &[u64],
    tool_calls: &[hipfire_runtime::prompt_frame::ToolCall],
) {
    let mut pending = backend
        .pending_tools
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    pending.retain(|entry| entry.session != session);
    if tool_calls.is_empty() {
        return;
    }
    if pending.len() >= MAX_PENDING_TOOL_TURNS {
        pending.remove(0);
    }
    pending.push(pending_turn(session, convo, tool_calls));
}

/// The calls as the CLIENT will see them: ids come from the same OpenAI
/// adapter that serializes the response, so the next turn's `tool_call_id`s
/// compare against what was actually sent.
#[cfg(feature = "multi-slot")]
fn pending_turn(
    session: u64,
    convo: &[u64],
    tool_calls: &[hipfire_runtime::prompt_frame::ToolCall],
) -> PendingToolTurn {
    PendingToolTurn {
        session,
        convo: convo.to_vec(),
        calls: crate::serve::complete::openai_tool_call_adapter_results(tool_calls)
            .into_iter()
            .zip(tool_calls)
            .map(|(adapted, call)| ToolCallKey {
                id: adapted.id,
                name: call.name.clone(),
                arguments: canonical_arguments(&call.arguments),
            })
            .collect(),
    }
}

#[cfg(all(test, feature = "multi-slot"))]
mod tests {
    use super::*;
    use hipfire_runtime::prompt_frame::{Message, Role, ThinkMode, ToolCall};

    fn message(role: Role, content: &str) -> Message {
        Message {
            role,
            content: content.to_owned(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: Some(id.to_owned()),
            name: name.to_owned(),
            arguments: serde_json::json!({ "q": name }),
            rendered_body: None,
        }
    }

    fn result(id: &str, content: &str) -> Message {
        let mut m = message(Role::Tool, content);
        m.tool_call_id = Some(id.to_owned());
        m
    }

    fn conversation(calls: Vec<ToolCall>, results: Vec<Message>) -> Vec<Message> {
        let mut msgs = vec![message(Role::User, "weather?")];
        let mut assistant = message(Role::Assistant, "");
        assistant.tool_calls = calls;
        msgs.push(assistant);
        msgs.extend(results);
        msgs
    }

    fn key(id: &str, name: &str) -> ToolCallKey {
        ToolCallKey {
            id: id.to_owned(),
            name: name.to_owned(),
            arguments: canonical_arguments(&serde_json::json!({ "q": name })),
        }
    }

    #[test]
    fn a_matching_tool_result_run_names_the_calls_it_answers() {
        let msgs = conversation(
            vec![call("call_0", "weather"), call("call_1", "time")],
            vec![result("call_0", "sunny"), result("call_1", "noon")],
        );
        let tail = tool_result_tail(&msgs).expect("a complete, in-order tail");
        assert_eq!(
            tail.answered,
            vec![key("call_0", "weather"), key("call_1", "time")]
        );
        assert_eq!(tail.results, vec!["sunny".to_owned(), "noon".to_owned()]);
    }

    #[test]
    fn out_of_order_or_partial_tool_results_refuse_reuse() {
        let swapped = conversation(
            vec![call("call_0", "weather"), call("call_1", "time")],
            vec![result("call_1", "noon"), result("call_0", "sunny")],
        );
        assert!(tool_result_tail(&swapped).is_none(), "reordered");

        let partial = conversation(
            vec![call("call_0", "weather"), call("call_1", "time")],
            vec![result("call_0", "sunny")],
        );
        assert!(tool_result_tail(&partial).is_none(), "one call unanswered");

        let unidentified = conversation(
            vec![call("call_0", "weather")],
            vec![message(Role::Tool, "sunny")],
        );
        assert!(tool_result_tail(&unidentified).is_none(), "no tool_call_id");

        let no_calls = conversation(Vec::new(), vec![result("call_0", "sunny")]);
        assert!(
            tool_result_tail(&no_calls).is_none(),
            "results for a turn that made no calls"
        );
    }

    #[test]
    fn a_duplicate_conversation_makes_the_reentry_ambiguous_not_lru() {
        let answered = vec![key("call_0", "weather")];
        let convo = vec![7u64];
        let entry = |session| PendingToolTurn {
            session,
            convo: convo.clone(),
            calls: answered.clone(),
        };
        assert_eq!(
            resolve_pending_session(&[entry(1)], &convo, &answered),
            Some(1)
        );
        assert_eq!(
            resolve_pending_session(&[entry(1), entry(2)], &convo, &answered),
            None,
            "two sessions fit these ids equally well"
        );
        assert_eq!(
            resolve_pending_session(&[entry(1)], &[9], &answered),
            None,
            "another conversation"
        );
        assert_eq!(
            resolve_pending_session(&[entry(1)], &convo, &[key("call_0", "time")]),
            None,
            "same id, different call"
        );
    }

    #[test]
    fn the_ids_handed_to_the_client_are_the_ids_that_resolve_the_session() {
        // The round trip the reentry depends on: this path records what the
        // OpenAI adapter minted, the client echoes it back in `tool_call_id`,
        // and the two must meet. A change to either side alone breaks reentry
        // silently — the turn still answers, just from a cold prefill.
        let convo = vec![turn_hash("weather?")];
        let emitted = vec![
            ToolCall {
                id: None,
                name: "get_weather".to_owned(),
                arguments: serde_json::json!({ "city": "Paris" }),
                rendered_body: None,
            },
            ToolCall {
                id: None,
                name: "get_time".to_owned(),
                arguments: serde_json::json!({ "tz": "CET" }),
                rendered_body: None,
            },
        ];
        let pending = vec![pending_turn(7, &convo, &emitted)];

        let echoed = crate::serve::complete::openai_tool_calls(&emitted);
        let mut assistant = message(Role::Assistant, "");
        assistant.tool_calls = echoed
            .iter()
            .map(|call| ToolCall {
                id: call["id"].as_str().map(str::to_owned),
                name: call["function"]["name"].as_str().unwrap().to_owned(),
                arguments: serde_json::from_str(call["function"]["arguments"].as_str().unwrap())
                    .unwrap(),
                rendered_body: None,
            })
            .collect();
        let msgs = vec![
            message(Role::User, "weather?"),
            assistant,
            result(echoed[0]["id"].as_str().unwrap(), "sunny"),
            result(echoed[1]["id"].as_str().unwrap(), "noon"),
        ];

        let tail = tool_result_tail(&msgs).expect("the client's echo must parse");
        assert_eq!(
            resolve_pending_session(&pending, &convo, &tail.answered),
            Some(7)
        );
    }

    #[test]
    fn configured_effort_reaches_the_emitter_instead_of_a_fixed_level() {
        assert_eq!(think_mode_for(Some("high"), true), ThinkMode::High);
        assert_eq!(think_mode_for(Some("max"), true), ThinkMode::Max);
        assert_eq!(think_mode_for(Some("low"), true), ThinkMode::Low);
        // Undefined effort is the template's own default, still a thinking turn.
        assert_eq!(think_mode_for(None, true), ThinkMode::Low);
        assert_eq!(think_mode_for(Some("high"), false), ThinkMode::NonThink);
    }
}

#[cfg(not(feature = "multi-slot"))]
pub(crate) fn complete_request_slots(
    _backend: &SlotBackend,
    _body: &serde_json::Value,
    _contract: &crate::serve::complete::RequestContract,
    _identity: &(String, u64),
    _cancelled: Option<&std::sync::atomic::AtomicBool>,
    _event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    _terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    bail!("multi-slot feature disabled")
}
