// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Daemon-free validation of the Cohere2-MoE serving chat path: select the
//! `tool_use` named template from the .hfq (list-form chat_template) and render
//! a multi-turn conversation through the SAME `JinjaChatFrame::render_messages`
//! the daemon uses. Validates (1) hfq.chat_template_named selection, (2)
//! minijinja renders the Cohere template with no undefined-var error, (3)
//! conversation-history interleaving, (4) tool-definition injection.
//!   usage: render_cohere2moe <model.hfq>

use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::tokenizer::Tokenizer;
use std::path::Path;

fn msg(role: Role, content: &str) -> Message {
    Message { role, content: content.to_string(), tool_calls: vec![], tool_call_id: None }
}

fn main() {
    let model = std::env::args().nth(1).expect("usage: render_cohere2moe <model.hfq>");
    let hfq = HfqFile::open(Path::new(&model)).expect("open model");
    let template = hfq
        .chat_template_named("tool_use")
        .expect("model has no tool_use chat_template");
    eprintln!("selected tool_use template ({} chars)", template.len());
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");

    let frame = JinjaChatFrame {
        tokenizer: &tok,
        template: &template,
        system: None,
        user: "",
        enable_thinking: true,
        bos_token: None,
    };

    // (1) Plain multi-turn conversation — history interleaving.
    let convo = vec![
        msg(Role::User, "Hi, who are you?"),
        msg(Role::Assistant, "I'm Command, a model built by Cohere. How can I help?"),
        msg(Role::User, "Write a Python function to reverse a string."),
    ];
    match frame.render_messages(&convo, None, None) {
        Ok(r) => {
            let t = tok.encode(&r);
            println!("=== PLAIN MULTI-TURN OK — {} chars, {} tokens ===", r.len(), t.len());
            println!("{r}");
            // structural assertions
            for need in [
                "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
                "<|CHATBOT_TOKEN|>",
                "Write a Python function to reverse a string.",
            ] {
                println!("  contains {:?}: {}", need, r.contains(need));
            }
            println!("  ends with CHATBOT gen-prompt: {}", r.trim_end().ends_with("<|CHATBOT_TOKEN|>"));
        }
        Err(e) => {
            eprintln!("PLAIN render FAILED: {e}");
            std::process::exit(1);
        }
    }

    // (2) With a tool definition — exercises the `{% if tools %}` injection
    // (and the `enable_citations` reference inside that block).
    let tools = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        }
    }]);
    let tools_arr = tools.as_array().unwrap();
    match frame.render_messages(&convo, Some(tools_arr), None) {
        Ok(r) => println!(
            "\n=== WITH-TOOLS OK — {} chars; injects get_weather: {}; has ## Tool Use: {} ===",
            r.len(), r.contains("get_weather"), r.contains("## Tool Use")
        ),
        Err(e) => eprintln!("\n=== WITH-TOOLS render FAILED: {e}\n(fix: supply enable_citations in JinjaChatFrame context)"),
    }
}
