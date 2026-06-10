// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Repro for the Pi multi-turn agentic derail: render a conversation that has
//! an assistant turn WITH tool_calls (+ a tool result) through the Cohere
//! tool_use template, exactly like the daemon's generate_cohere2moe does. If
//! render_messages ERRORS, generate_cohere2moe falls back to the hand-rolled
//! ChatML frame → the model sees <|im_start|>/<|im_end|> → derails.
//!   usage: render_cohere2moe <model.hfq>

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role, ToolCall};
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;

fn main() {
    let model = std::env::args().nth(1).expect("usage: render_cohere2moe <model.hfq>");
    let hfq = HfqFile::open(Path::new(&model)).expect("open model");
    // Mirror the daemon's arch-12 selection + START_RESPONSE→START_TEXT rewrite.
    let template = hfq
        .chat_template_named("tool_use")
        .expect("no tool_use template")
        .replace("<|START_RESPONSE|>", "<|START_TEXT|>")
        .replace("<|END_RESPONSE|>", "<|END_TEXT|>")
        // The daemon's Message/ToolCall are flat {name, arguments} with no
        // tool_plan; the upstream template reads message.tool_plan + the
        // OpenAI-nested tc['function'][...]. Bridge the shape:
        .replace("{{message.tool_plan}}", "{{ message.tool_plan or '' }}")
        .replace("{{ tc['function']['name'] }}", "{{ tc.name }}")
        .replace("{{ tc['function']['arguments']|tojson }}", "{{ tc.arguments|tojson }}");
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let frame = JinjaChatFrame {
        tokenizer: &tok, template: &template, system: None, user: "",
        enable_thinking: true, bos_token: None,
    };

    let tools = serde_json::json!([{
        "type": "function",
        "function": { "name": "bash", "description": "Run a bash command.",
            "parameters": {"type":"object","properties":{"command":{"type":"string"}},"required":["command"]} }
    }]);
    let tools_arr = tools.as_array().unwrap();

    // (A) plain multi-turn — known good.
    let plain = vec![
        Message { role: Role::User, content: "Hi".into(), tool_calls: vec![], tool_call_id: None },
        Message { role: Role::Assistant, content: "Hello!".into(), tool_calls: vec![], tool_call_id: None },
        Message { role: Role::User, content: "List files.".into(), tool_calls: vec![], tool_call_id: None },
    ];
    match frame.render_messages(&plain, Some(tools_arr), None) {
        Ok(_) => println!("(A) plain multi-turn + tools: OK"),
        Err(e) => println!("(A) plain multi-turn + tools: ERR -> {e}"),
    }

    // (B) THE PI CASE: assistant turn WITH tool_calls, then a tool result.
    let agentic = vec![
        Message { role: Role::User, content: "Implement a Blink-hash tree.".into(), tool_calls: vec![], tool_call_id: None },
        Message {
            role: Role::Assistant,
            content: "".into(),
            tool_calls: vec![ToolCall { name: "bash".into(), arguments: serde_json::json!({"command":"ls -la"}) }],
            tool_call_id: None,
        },
        Message { role: Role::Tool, content: "total 7896\ndrwx... blink_hash.pdf".into(), tool_calls: vec![], tool_call_id: Some("0".into()) },
    ];
    match frame.render_messages(&agentic, Some(tools_arr), None) {
        Ok(r) => {
            println!("(B) agentic (tool_calls in history): OK — {} chars", r.len());
            println!("--- tail ---\n{}", &r[r.len().saturating_sub(400)..]);
        }
        Err(e) => println!("(B) agentic (tool_calls in history): ERR -> {e}   <<< this is the bug (falls back to ChatML)"),
    }
}
