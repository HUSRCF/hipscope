#!/usr/bin/env bash
# Gemma4 E-series agentic smoke: strict two-turn prefix reuse plus native tools.
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-artifacts/gemma4-e4b-q8f16.hfq}"
MAX_SEQ="${MAX_SEQ:-1024}"
EXE="${EXE:-target/release/examples/daemon}"
OUT="${OUT:-/tmp/hipfire-gemma4-tools-prefix-smoke.log}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-600}"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 2; }
[[ -x "$EXE" ]] || { echo "daemon not found: $EXE" >&2; exit 2; }

home_dir="$(mktemp -d /tmp/hipfire-gemma4-agentic-home-XXXXXX)"
daemon_pid=""
cleanup() {
    local status=$?
    exec 3>&- 2>/dev/null || true
    exec 4<&- 2>/dev/null || true
    if [[ -n "$daemon_pid" ]] && kill -0 "$daemon_pid" 2>/dev/null; then
        kill "$daemon_pid" 2>/dev/null || true
        wait "$daemon_pid" 2>/dev/null || true
    fi
    rm -rf "$home_dir"
    return "$status"
}
trap cleanup EXIT
: >"$OUT"

coproc GEMMA_DAEMON {
    HOME="$home_dir" HIP_VISIBLE_DEVICES="$GPU_ID" "$EXE" 2>&1
}
daemon_pid=$GEMMA_DAEMON_PID
daemon_in_fd="${GEMMA_DAEMON[1]}"
daemon_out_fd="${GEMMA_DAEMON[0]}"
exec 3>&"$daemon_in_fd"
exec 4<&"$daemon_out_fd"
eval "exec ${daemon_in_fd}>&-"
eval "exec ${daemon_out_fd}<&-"

LAST_TEXT=""
LAST_DONE=""
send_request() {
    local id="$1"
    local attempt_id="$2"
    local payload="$3"
    LAST_TEXT=""
    LAST_DONE=""
    printf '%s\n' "$payload" >&3
    while IFS= read -r -t "$REQUEST_TIMEOUT_S" line <&4; do
        printf '%s\n' "$line" >>"$OUT"
        if [[ "$line" == *'"type":"error"'* && "$line" == *"\"id\":\"$id\""* ]]; then
            echo "request $id failed: $line" >&2
            return 1
        fi
        if [[ "$line" == *'"type":"token"'* && "$line" == *"\"id\":\"$id\""* ]]; then
            fragment="$(jq -er --arg id "$id" 'select(.type == "token" and .id == $id) | .text' <<<"$line")"
            LAST_TEXT+="$fragment"
        fi
        if [[ "$line" == *'"type":"commit_ready"'* && "$line" == *"\"id\":\"$id\""* ]]; then
            printf '{"type":"commit","id":"%s","attempt_id":%s}\n' "$id" "$attempt_id" >&3
        fi
        if [[ "$line" == *'"type":"done"'* && "$line" == *"\"id\":\"$id\""* ]]; then
            LAST_DONE="$line"
            return 0
        fi
    done
    echo "request $id timed out or daemon exited" >&2
    return 1
}

printf '{"type":"load","model":"%s","params":{"max_seq":%s,"kv_mode":"q8"}}\n' \
    "$MODEL" "$MAX_SEQ" >&3

first_payload="$(jq -nc '{type:"generate", id:"gemma-cache-1", attempt_id:1, prompt:"Reply with one short word meaning affirmative.", messages:[{role:"user",content:"Reply with one short word meaning affirmative."}], temperature:0.0, max_tokens:12}')"
send_request "gemma-cache-1" 1 "$first_payload"
first_text="$LAST_TEXT"
[[ -n "$first_text" ]] || { echo "first turn produced no visible text" >&2; exit 1; }

second_payload="$(jq -nc --arg answer "$first_text" '{type:"generate", id:"gemma-cache-2", attempt_id:2, prompt:"Now give the opposite word.", messages:[{role:"user",content:"Reply with one short word meaning affirmative."},{role:"assistant",content:$answer},{role:"user",content:"Now give the opposite word."}], temperature:0.0, max_tokens:12}')"
send_request "gemma-cache-2" 2 "$second_payload"
cached_tokens="$(jq -er '.cached_tokens' <<<"$LAST_DONE")"
if (( cached_tokens <= 0 )); then
    echo "expected Gemma prefix-cache hit, got cached_tokens=$cached_tokens" >&2
    exit 1
fi

tool_payload="$(jq -nc '{type:"generate", id:"gemma-tool", attempt_id:3, prompt:"Call get_weather for Taipei. Do not answer directly.", messages:[{role:"user",content:"Call get_weather for Taipei. Do not answer directly."}], tools:[{type:"function",function:{name:"get_weather",description:"Get current weather",parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}}}], temperature:0.0, max_tokens:64}')"
send_request "gemma-tool" 3 "$tool_payload"
tool_finish="$(jq -er '.finish_reason' <<<"$LAST_DONE")"
if [[ "$tool_finish" != "tool_calls" ]]; then
    echo "expected tool_calls terminal, got $tool_finish" >&2
    exit 1
fi
jq -e '.calls | length > 0 and .[0].name == "get_weather"' <<<"$LAST_DONE" >/dev/null
tool_calls="$(jq -c '.calls' <<<"$LAST_DONE")"

tool_followup_payload="$(jq -nc --argjson calls "$tool_calls" '{type:"generate", id:"gemma-tool-followup", attempt_id:4, prompt:"State the weather result in one short sentence.", messages:[{role:"user",content:"Call get_weather for Taipei. Do not answer directly."},{role:"assistant",content:"",tool_calls:$calls},{role:"tool",content:"Sunny, 30 C",tool_call_id:"call_weather_1"},{role:"user",content:"State the weather result in one short sentence."}], tools:[{type:"function",function:{name:"get_weather",description:"Get current weather",parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}}}], temperature:0.0, max_tokens:24}')"
send_request "gemma-tool-followup" 4 "$tool_followup_payload"
tool_cached_tokens="$(jq -er '.cached_tokens' <<<"$LAST_DONE")"
if (( tool_cached_tokens <= 0 )); then
    echo "expected tool-turn prefix-cache hit, got cached_tokens=$tool_cached_tokens" >&2
    exit 1
fi
[[ -n "$LAST_TEXT" ]] || { echo "tool follow-up produced no visible text" >&2; exit 1; }

printf '{"type":"unload"}\n' >&3
exec 3>&-
wait "$daemon_pid"

grep -aE '"type":"(tool_calls|done)".*"id":"gemma-(cache-[12]|tool|tool-followup)"' "$OUT" || true
echo "Gemma4 tools + prefix-cache smoke passed on HIP_VISIBLE_DEVICES=$GPU_ID; plain_cached_tokens=$cached_tokens; tool_cached_tokens=$tool_cached_tokens; log: $OUT"
