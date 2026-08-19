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
use anyhow::{anyhow, bail, Context, Result};
#[cfg(not(feature = "multi-slot"))]
use anyhow::{bail, Result};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::prompt_frame::{AssistantPrefix, ChatFrame, Role};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::serve::{Event, SubmitRequest};
#[cfg(feature = "multi-slot")]
use hipfire_runtime::tokenizer::Tokenizer;
use serde_json;
use std::sync::{mpsc, Arc};

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
}

#[cfg(not(feature = "multi-slot"))]
pub(crate) struct SlotBackend;

#[cfg(feature = "multi-slot")]
pub(crate) fn complete_request_slots(
    backend: &SlotBackend,
    body: &serde_json::Value,
    identity: &(String, u64),
    event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    use hipfire_arch_qwen35::spec_emit::Qwen35Emit;
    use hipfire_runtime::prompt_frame::{
        AssistantPrefix, ChatFrame, JinjaChatFrame, Message, Role, ThinkMode, ToolCall,
    };
    use hipfire_runtime::serve::{Event, SubmitRequest};
    use hipfire_runtime::spec::{ClientEvent, SpecEmit, SpecEmitCtx};

    let model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown")
        .to_owned();
    let max_tokens = body
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(512) as usize;

    let Some(msgs) = body.get("messages").and_then(serde_json::Value::as_array) else {
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

    // hag/OpenAI clients disable reasoning per-request; without honoring it a
    // small max_tokens budget burns entirely inside <think> and content
    // arrives empty (observed on ChatJSON title generation).
    let reasoning_off = body
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        == Some("none")
        || body
            .get("reasoning_budget_tokens")
            .and_then(serde_json::Value::as_u64)
            == Some(0)
        || body
            .get("max_think_tokens")
            .and_then(serde_json::Value::as_u64)
            == Some(1);
    let has_think = backend.tokenizer.special_token_id("<think>").is_some();
    let enable_thinking = has_think && !reasoning_off;

    let tools: Option<Vec<serde_json::Value>> = body
        .get("tools")
        .and_then(serde_json::Value::as_array)
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
                reasoning_effort: None,
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
    pub(crate) fn turn_hash(s: &str) -> u64 {
        let mut h = 0xcbf29ce484222325_u64;
        for b in s.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
    let convo: Vec<u64> = messages
        .iter()
        .filter(|m| m.role == Role::User)
        .map(|m| turn_hash(&m.content))
        .collect();

    // Tokens that continue the previous assistant turn into this one. The
    // engine appends these to the session's exact stored tokens instead of
    // re-rendering the history, which is what makes the result a strict
    // extension of the KV. Two shapes: a new user turn (continuation match,
    // turn 2+), or a trailing run of tool results (reentry match — same user
    // turns, the model's tool calls being answered).
    let continuation = match messages.last().map(|m| m.role) {
        Some(Role::User) if convo.len() >= 2 => hipfire_runtime::prompt_frame::continuation_suffix(
            &backend.tokenizer,
            &messages.last().map(|m| m.content.as_str()).unwrap_or(""),
            prefix,
        ),
        Some(Role::Tool) => {
            let mut tail: Vec<String> = messages
                .iter()
                .rev()
                .take_while(|m| m.role == Role::Tool)
                .map(|m| m.content.clone())
                .collect();
            tail.reverse();
            hipfire_runtime::prompt_frame::continuation_suffix_tool_results(
                &backend.tokenizer,
                &tail,
                prefix,
            )
        }
        _ => Vec::new(),
    };

    let (tx, rx) = mpsc::channel::<Event>();
    backend
        .engine
        .submit(SubmitRequest {
            session: None,
            prompt_tokens,
            convo,
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
            seed: body
                .get("seed")
                .and_then(serde_json::Value::as_u64)
                .map(|s| s as u32)
                .unwrap_or_else(|| (turn_hash(&identity.0) ^ identity.1) as u32),
            reply: tx,
        })
        .map_err(|e| anyhow!("multi_slot submit: {e}"))?;

    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut finish = "stop";
    let mut cached_tokens = 0usize;
    let mut prefill_tokens = 0usize;
    let mut generated_tokens = 0usize;

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
        think_mode: if enable_thinking {
            ThinkMode::Low
        } else {
            ThinkMode::NonThink
        },
        decoded_vocab: None,
    }));
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    let mut render = |events: Vec<ClientEvent>,
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
    while let Ok(ev) = rx.recv() {
        match ev {
            Event::Accepted {
                reused, prefill, ..
            } => {
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

    let completion = Completion {
        id: identity.0.clone(),
        created: identity.1,
        model,
        content,
        reasoning_content,
        preserve_thinking: false,
        tool_calls,
        done: serde_json::json!({
            "finish_reason": finish,
            "cached_tokens": cached_tokens,
            "prefill_tokens": prefill_tokens,
            "tokens": generated_tokens,
        }),
        logprobs: None,
    };
    // The terminal callback is what stages the response body and signals the
    // HTTP handler that the request succeeded. Skipping it leaves the handler
    // waiting on a status that never arrives, which surfaces to the client as
    // "generation worker disconnected".
    terminal_callback(&completion).map_err(|e| anyhow!("terminal callback: {e}"))?;
    Ok(completion)
}

#[cfg(not(feature = "multi-slot"))]
pub(crate) fn complete_request_slots(
    _backend: &SlotBackend,
    _body: &serde_json::Value,
    _identity: &(String, u64),
    _event_callback: &mut dyn FnMut(&serde_json::Value) -> Result<(), hipfire_client::ClientError>,
    _terminal_callback: &mut dyn FnMut(&Completion) -> Result<(), hipfire_client::ClientError>,
) -> Result<Completion> {
    bail!("multi-slot feature disabled")
}
