#!/usr/bin/env bash
# Gemma4 E-series HFQ serving smoke. Exercises short and multi-chunk prefill.
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${GPU_ID:-1}"
MODEL="${MODEL:-artifacts/gemma4-e2b-q8f16.hfq}"
MAX_SEQ="${MAX_SEQ:-512}"
EXE="${EXE:-target/release/examples/daemon}"
OUT="${OUT:-/tmp/hipfire-gemma4-e-series-smoke.log}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-600}"

if [[ ! -f "$MODEL" ]]; then
    echo "Gemma4 smoke model not found: $MODEL" >&2
    exit 2
fi
if [[ ! -x "$EXE" ]]; then
    echo "Gemma4 smoke daemon not found: $EXE" >&2
    exit 2
fi

home_dir="$(mktemp -d /tmp/hipfire-gemma4-home-XXXXXX)"
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
# Keep only stable fds 3/4; otherwise Bash retains the coproc's original write
# fd and EOF never reaches the daemon after the final unload.
eval "exec ${daemon_in_fd}>&-"
eval "exec ${daemon_out_fd}<&-"

send_generate() {
    local id="$1"
    local attempt_id="$2"
    local prompt="$3"
    local max_tokens="$4"
    printf '{"type":"generate","id":"%s","attempt_id":%s,"prompt":"%s","temperature":0.0,"max_tokens":%s}\n' \
        "$id" "$attempt_id" "$prompt" "$max_tokens" >&3
    while IFS= read -r -t "$REQUEST_TIMEOUT_S" line <&4; do
        printf '%s\n' "$line" >>"$OUT"
        if [[ "$line" == *'"type":"error"'* && "$line" == *"\"id\":\"$id\""* ]]; then
            return 1
        fi
        if [[ "$line" == *'"type":"commit_ready"'* && "$line" == *"\"id\":\"$id\""* ]]; then
            printf '{"type":"commit","id":"%s","attempt_id":%s}\n' "$id" "$attempt_id" >&3
        fi
        if [[ "$line" == *'"type":"done"'* && "$line" == *"\"id\":\"$id\""* ]]; then
            return 0
        fi
    done
    echo "Timed out or lost daemon output while waiting for request $id" >&2
    return 1
}

printf '{"type":"load","model":"%s","params":{"max_seq":%s,"kv_mode":"q8"}}\n' \
    "$MODEL" "$MAX_SEQ" >&3

send_generate "gemma-short" 1 "Reply with exactly: hello" 8
send_generate "gemma-multichunk" 2 \
    "Summarize this text in one short sentence: Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta. Alpha beta gamma delta epsilon zeta eta theta." \
    16

printf '{"type":"unload"}\n' >&3
exec 3>&-
wait "$daemon_pid"

if grep -aEq 'panicked|FATAL|"type":"error"|HIP error' "$OUT"; then
    echo "Gemma4 E-series smoke failed; see $OUT" >&2
    grep -aE 'panicked|FATAL|"type":"error"|HIP error' "$OUT" >&2 || true
    exit 1
fi

done_count="$(grep -ac '"type":"done"' "$OUT" || true)"
if [[ "$done_count" -ne 2 ]]; then
    echo "Gemma4 E-series smoke expected 2 completed requests, got $done_count; see $OUT" >&2
    exit 1
fi

grep -aE '"type":"done"|\[gemma4\]' "$OUT" || true
echo "Gemma4 E-series smoke passed on HIP_VISIBLE_DEVICES=$GPU_ID; log: $OUT"
