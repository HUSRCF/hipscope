#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""Fake omp for tests: emits realistic JSONL per /tmp/omp-probe.out."""

import json
import os
import sys
from pathlib import Path

# Env controls:
#   FAKE_OMP_LOG: path to append JSON lines of invocation (args)
#   FAKE_OMP_RESPONSES: path to JSON file containing list of responses to return sequentially
#       Each entry: {"phase": "prelim"|"verdict"|"probe", "json": {...}} or {"text": "raw text not JSON"}
#       If not set, returns default canned prelim/verdict based on call count.
#   FAKE_OMP_GARBAGE: if "1", always return garbage text that fails extract_json
#   FAKE_OMP_CALL_COUNT: file to track call count across invocations

def _emit(assistant_text: str):
    # Emit minimal realistic JSONL matching /tmp/omp-probe.out shape
    events = [
        {"type": "session", "version": 3, "id": "test-session", "timestamp": "2026-09-02T00:00:00Z", "cwd": "/tmp"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": {"role": "user", "content": [{"type": "text", "text": "prompt"}], "attribution": "user", "timestamp": 0}},
        {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "prompt"}], "attribution": "user", "timestamp": 0}},
        {"type": "message_start", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking...", "thinkingSignature": "{}"}, {"type": "text", "text": assistant_text, "textSignature": "{}"}], "api": "openai-responses", "provider": "fake", "model": "fake-model", "usage": {"input": 0, "output": 0}}},
        {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking...", "thinkingSignature": "{}"}, {"type": "text", "text": assistant_text, "textSignature": "{}"}], "api": "openai-responses", "provider": "fake", "model": "fake-model", "usage": {"input": 0, "output": 0}}},
        {"type": "turn_end", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "thinking..."}, {"type": "text", "text": assistant_text}]}},
        {"type": "agent_end", "messages": []},
    ]
    for ev in events:
        sys.stdout.write(json.dumps(ev) + "\n")

def main():
    args = sys.argv[1:]
    # Log invocation
    log_path = os.environ.get("FAKE_OMP_LOG")
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                json.dump({"args": args}, f)
                f.write("\n")
        except Exception:
            pass

    count_path = os.environ.get("FAKE_OMP_CALL_COUNT")
    call_idx = 0
    if count_path and Path(count_path).is_file():
        try:
            call_idx = int(Path(count_path).read_text().strip() or "0")
        except Exception:
            call_idx = 0
    if count_path:
        try:
            Path(count_path).parent.mkdir(parents=True, exist_ok=True)
            Path(count_path).write_text(str(call_idx + 1))
        except Exception:
            pass

    if os.environ.get("FAKE_OMP_GARBAGE") == "1":
        # Return garbage that is not JSON, both attempts
        _emit("this is not json at all — garbage !@#")
        sys.exit(0)

    responses_path = os.environ.get("FAKE_OMP_RESPONSES")
    if responses_path and Path(responses_path).is_file():
        try:
            responses = json.loads(Path(responses_path).read_text())
            if call_idx < len(responses):
                entry = responses[call_idx]
            else:
                entry = responses[-1] if responses else {}
            if "json" in entry:
                text = json.dumps(entry["json"])
                # Optionally wrap in fences if entry says
                if entry.get("fenced"):
                    text = "Here is the result:\n```json\n" + text + "\n```\n"
                elif entry.get("prose"):
                    text = "Sure, here it is: " + text + " hope that helps."
                _emit(text)
                sys.exit(0)
            elif "text" in entry:
                _emit(entry["text"])
                sys.exit(0)
        except Exception as e:
            sys.stderr.write(f"fake_omp response load failed: {e}\n")

    # Default canned responses: first call prelim, second verdict
    if call_idx == 0:
        # prelim
        text = json.dumps({
            "phase": "prelim",
            "summary": "test change",
            "surfaces": ["load"],
            "suspected_regressions": [],
            "extra_routes": [],
            "questions_for_author": []
        })
    else:
        text = json.dumps({
            "phase": "verdict",
            "decision": "greenlight",
            "confidence": 0.9,
            "regressions": [],
            "coverage": {"surfaces_touched": ["load"], "surfaces_evidenced": ["load"], "gaps": []},
            "eyeball": [],
            "rationale": "all evidenced"
        })
        # Check if prompt indicates verdict should be block/needs-human via env override
        override = os.environ.get("FAKE_OMP_VERDICT_DECISION")
        if override:
            obj = json.loads(text)
            obj["decision"] = override
            text = json.dumps(obj)
    _emit(text)
    sys.exit(0)

if __name__ == "__main__":
    main()
