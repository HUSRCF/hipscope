#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fresh-daemon growing-conversation A/B with the prefix cache disabled.
#
# Unlike serve_harness.py --mode chain's normal cached path, every turn resets
# recurrent/KV state and prefills the complete rendered conversation. This
# exercises the user-facing multi-turn shape while keeping each prefill cold.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-35b-a3b.mq4r}"
TAG="${TAG:-qwen3.6:35b-a3b-mq4r}"
GPU_ID="${GPU_ID:-0}"
BUN_BIN="${BUN_BIN:-$(command -v bun || true)}"
DAEMON_BIN="${DAEMON_BIN:-$PWD/target/release/examples/daemon}"
PROCESS_REPEATS="${PROCESS_REPEATS:-3}"
MIN_PREFILL_TOKENS="${MIN_PREFILL_TOKENS:-4096}"
MAX_SEQ="${MAX_SEQ:-32768}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
THINKING="${THINKING:-med}"
SAMPLING="${SAMPLING:-registry}"
CACHE_CKPT_RESUME="${CACHE_CKPT_RESUME:-0}"
DPM_WARMUP_SECS="${DPM_WARMUP_SECS:-5}"
EXCLUDE_FIRST_TURNS="${EXCLUDE_FIRST_TURNS:-1}"
LONG_PROMPT_SOURCE="${LONG_PROMPT_SOURCE:-}"
PORT="${PORT:-11530}"
SLEEP_BETWEEN_MODES="${SLEEP_BETWEEN_MODES:-20}"
SLEEP_BETWEEN_PAIRS="${SLEEP_BETWEEN_PAIRS:-30}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results/qkvza_uncached_multiturn_ab_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$MODEL" ]]; then
    echo "model not found: $MODEL" >&2
    exit 2
fi
if [[ ! -x "$DAEMON_BIN" ]]; then
    echo "daemon not executable: $DAEMON_BIN" >&2
    exit 2
fi
if [[ -z "$BUN_BIN" || ! -x "$BUN_BIN" ]]; then
    echo "bun not found; set BUN_BIN" >&2
    exit 2
fi
if (( EXCLUDE_FIRST_TURNS < 0 || EXCLUDE_FIRST_TURNS >= 5 )); then
    echo "EXCLUDE_FIRST_TURNS must be between 0 and 4" >&2
    exit 2
fi

mkdir -p "$RESULT_DIR"
PROMPT_ARGS=()
if [[ -n "$LONG_PROMPT_SOURCE" ]]; then
    if [[ ! -f "$LONG_PROMPT_SOURCE" ]]; then
        echo "long prompt source not found: $LONG_PROMPT_SOURCE" >&2
        exit 2
    fi
    prompt_json="$RESULT_DIR/long_chain_prompts.json"
    python3 - "$LONG_PROMPT_SOURCE" "$prompt_json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
rows = [
    {
        "genre": f"long_turn_{turn}",
        "prompt": source + f"\n\nTurn {turn}: acknowledge this document in one short sentence.",
    }
    for turn in range(1, 6)
]
pathlib.Path(sys.argv[2]).write_text(json.dumps(rows))
PY
    PROMPT_ARGS=(--prompts-file "$prompt_json")
fi
printf 'pair\torder\tmode\tturn\tctx\tcached\tuncached\tprefill_ms\tprefill_tok_s\tdecode_tok_s\n' >"$RESULT_DIR/raw.tsv"
printf 'pair\torder\tmode\teligible_events\troute_hit_events\n' >"$RESULT_DIR/routes.tsv"

{
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || true)"
    echo "date=$(date -Is)"
    echo "model=$MODEL"
    echo "tag=$TAG"
    echo "gpu_id=$GPU_ID"
    echo "daemon_bin=$DAEMON_BIN"
    echo "bun_bin=$BUN_BIN"
    echo "process_repeats=$PROCESS_REPEATS"
    echo "min_prefill_tokens=$MIN_PREFILL_TOKENS"
    echo "max_seq=$MAX_SEQ"
    echo "max_tokens=$MAX_TOKENS"
    echo "thinking=$THINKING"
    echo "sampling=$SAMPLING"
    echo "cache_ckpt_resume=$CACHE_CKPT_RESUME"
    echo "dpm_warmup_secs=$DPM_WARMUP_SECS"
    echo "exclude_first_turns=$EXCLUDE_FIRST_TURNS"
    echo "long_prompt_source=${LONG_PROMPT_SOURCE:-built-in-chain}"
    if [[ -n "$LONG_PROMPT_SOURCE" ]]; then
        echo "long_prompt_sha256=$(sha256sum "$LONG_PROMPT_SOURCE" | awk '{print $1}')"
    fi
    echo "prompt_cache=disabled"
    echo "kv=q8"
    echo "mtp=off"
    echo "thinking=$THINKING"
    echo
    echo "rocm_smi_before:"
    rocm-smi --showproductname --showuse --showmemuse --showpids 2>/dev/null || true
} >"$RESULT_DIR/meta.txt"

run_mode() {
    local pair="$1"
    local order="$2"
    local mode="$3"
    local cell="$RESULT_DIR/pair$(printf '%02d' "$pair")/$mode"
    mkdir -p "$cell"

    export HIP_VISIBLE_DEVICES="$GPU_ID"
    export HIPFIRE_DAEMON_BIN="$DAEMON_BIN"
    export HIPFIRE_QWEN_PROMPT_CACHE=0
    export HIPFIRE_CACHE_CKPT_RESUME="$CACHE_CKPT_RESUME"
    export HIPFIRE_DPM_WARMUP_SECS="$DPM_WARMUP_SECS"
    export HIPFIRE_QKVZA_SPLIT_TAIL_MIN_PREFILL_TOKENS="$MIN_PREFILL_TOKENS"
    export HIPFIRE_QKVZA_SPLIT_TAIL_DIAG=1
    if [[ "$mode" == "on" ]]; then
        export HIPFIRE_QKVZA_SPLIT_TAIL=1
    else
        unset HIPFIRE_QKVZA_SPLIT_TAIL
    fi

    # Persist the exact feature/cache environment before the daemon is spawned.
    env | LC_ALL=C sort | grep -E '^(HIP_VISIBLE_DEVICES|HIPFIRE_(DAEMON_BIN|QWEN_PROMPT_CACHE|CACHE_CKPT_RESUME|DPM_WARMUP_SECS|QKVZA_SPLIT_TAIL(_.*)?))=' \
        >"$cell/launch_env.txt"

    python3 scripts/serve_harness.py \
        --bun "$BUN_BIN" \
        --model "$MODEL" \
        --tag "$TAG" \
        --kv q8 \
        --mtp off \
        --thinking "$THINKING" \
        --max-tokens "$MAX_TOKENS" \
        --max-seq "$MAX_SEQ" \
        --sampling "$SAMPLING" \
        --mode chain \
        "${PROMPT_ARGS[@]}" \
        --port "$PORT" \
        --home "$cell/home" \
        --serve-log "$cell/serve.log" \
        --out "$cell/rows.json" \
        --seed 305419896 \
        2>&1 | tee "$cell/harness.log"

    python3 - "$pair" "$order" "$mode" "$cell" "$RESULT_DIR/raw.tsv" "$RESULT_DIR/routes.tsv" <<'PY'
import json
import pathlib
import re
import sys

pair, order, mode, cell, raw_path, routes_path = sys.argv[1:]
cell = pathlib.Path(cell)
rows = json.loads((cell / "rows.json").read_text())
if len(rows) != 5:
    raise SystemExit(f"expected five chain turns, got {len(rows)}")
log = (cell / "serve.log").read_text(errors="replace")
eligible = len(re.findall(r"qkvza_split_tail request eligible=true", log))
route_hits = len(re.findall(r"qkvza_split_tail route=hit", log))
if any(row.get("cached") != 0 for row in rows):
    raise SystemExit(f"prefix cache was not disabled: {[row.get('cached') for row in rows]}")
if mode == "on" and (eligible == 0 or route_hits == 0):
    raise SystemExit("active serve did not report an eligible request and route hit")
if mode == "off" and (eligible != 0 or route_hits != 0):
    raise SystemExit("off serve unexpectedly reported an eligible request or route hit")
with pathlib.Path(raw_path).open("a") as handle:
    for turn, row in enumerate(rows, 1):
        handle.write(
            f"{pair}\t{order}\t{mode}\t{turn}\t{row['ctx']}\t{row['cached']}\t{row['ctx']}\t"
            f"{row['prefill_ms']}\t{row['prefill_tok_s']}\t{row['decode_tok_s']}\n"
        )
with pathlib.Path(routes_path).open("a") as handle:
    handle.write(f"{pair}\t{order}\t{mode}\t{eligible}\t{route_hits}\n")
PY
}

for ((pair = 1; pair <= PROCESS_REPEATS; pair++)); do
    if (( pair % 2 == 1 )); then
        order="off-on"
        modes="off on"
    else
        order="on-off"
        modes="on off"
    fi
    echo "===== pair=$pair/$PROCESS_REPEATS order=$order ====="
    index=0
    for mode in $modes; do
        if (( index > 0 )); then
            sleep "$SLEEP_BETWEEN_MODES"
        fi
        run_mode "$pair" "$order" "$mode"
        index=$((index + 1))
    done
    if (( pair < PROCESS_REPEATS )); then
        sleep "$SLEEP_BETWEEN_PAIRS"
    fi
done

python3 - "$RESULT_DIR" "$MIN_PREFILL_TOKENS" "$CACHE_CKPT_RESUME" "$DPM_WARMUP_SECS" "$EXCLUDE_FIRST_TURNS" <<'PY'
import csv
import pathlib
import statistics
import sys

result_dir = pathlib.Path(sys.argv[1])
min_prefill_tokens = int(sys.argv[2])
cache_ckpt_resume = sys.argv[3]
dpm_warmup_secs = sys.argv[4]
exclude_first_turns = int(sys.argv[5])
with (result_dir / "raw.tsv").open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
with (result_dir / "routes.tsv").open(newline="") as handle:
    routes = list(csv.DictReader(handle, delimiter="\t"))

arms = {}
for row in rows:
    if int(row["turn"]) <= exclude_first_turns:
        continue
    key = (int(row["pair"]), row["mode"])
    arm = arms.setdefault(key, {"tokens": 0, "prefill_ms": 0.0, "decode": []})
    arm["tokens"] += int(row["uncached"])
    arm["prefill_ms"] += float(row["prefill_ms"])
    if row["decode_tok_s"] not in {"", "None"}:
        arm["decode"].append(float(row["decode_tok_s"]))

for arm in arms.values():
    arm["prefill_tok_s"] = arm["tokens"] * 1000.0 / arm["prefill_ms"]
    arm["decode_tok_s"] = statistics.mean(arm["decode"]) if arm["decode"] else float("nan")

pairs = sorted({pair for pair, _ in arms})
prefill_deltas = []
decode_deltas = []
for pair in pairs:
    off = arms[pair, "off"]
    on = arms[pair, "on"]
    prefill_deltas.append((on["prefill_tok_s"] / off["prefill_tok_s"] - 1.0) * 100.0)
    decode_deltas.append((on["decode_tok_s"] / off["decode_tok_s"] - 1.0) * 100.0)

off_rates = [arms[pair, "off"]["prefill_tok_s"] for pair in pairs]
on_rates = [arms[pair, "on"]["prefill_tok_s"] for pair in pairs]
active_routes = [row for row in routes if row["mode"] == "on"]
eligible_events = sum(int(row["eligible_events"]) for row in active_routes)
route_hit_daemons = sum(int(row["route_hit_events"]) > 0 for row in active_routes)
expected_eligible_events = len(active_routes) * 5
context_tokens = sorted({int(row["ctx"]) for row in rows})
aggregate_context_tokens = sorted(
    {int(row["ctx"]) for row in rows if int(row["turn"]) > exclude_first_turns}
)
lines = [
    "# QKVZA uncached multi-turn serving A/B",
    "",
    f"- fresh-daemon pairs: {len(pairs)}",
    "- conversation turns per arm: 5",
    "- prefix cache: disabled (`HIPFIRE_QWEN_PROMPT_CACHE=0`)",
    f"- DeltaNet checkpoint configuration: {'enabled' if cache_ckpt_resume == '1' else 'disabled'} (`HIPFIRE_CACHE_CKPT_RESUME={cache_ckpt_resume}`; runtime creation was not separately instrumented)",
    f"- daemon-load DPM warmup: {dpm_warmup_secs} seconds",
    f"- discarded long-prefill priming turns per daemon: {exclude_first_turns}",
    f"- split-tail admission threshold: {min_prefill_tokens} uncached tokens",
    f"- all full-context tokens: {', '.join(str(value) for value in context_tokens)}",
    f"- aggregate full-context tokens: {', '.join(str(value) for value in aggregate_context_tokens)}",
    f"- active eligible request events: {eligible_events}/{expected_eligible_events}",
    f"- active daemons reporting a route hit: {route_hit_daemons}/{len(active_routes)} (the hit diagnostic logs once per process)",
    f"- off aggregate-prefill median: {statistics.median(off_rates):.1f} tok/s",
    f"- active aggregate-prefill median: {statistics.median(on_rates):.1f} tok/s",
    f"- cross-sample prefill delta: {(statistics.median(on_rates) / statistics.median(off_rates) - 1.0) * 100.0:+.2f}%",
    f"- paired prefill median delta: {statistics.median(prefill_deltas):+.2f}%",
    f"- positive prefill pairs: {sum(x > 0 for x in prefill_deltas)}/{len(prefill_deltas)}",
    f"- paired prefill deltas: {', '.join(f'{x:+.2f}%' for x in prefill_deltas)}",
    f"- paired decode median delta: {statistics.median(decode_deltas):+.2f}%",
    f"- paired decode deltas: {', '.join(f'{x:+.2f}%' for x in decode_deltas)}",
    "",
    "| Pair | Off prefill tok/s | Active prefill tok/s | Prefill delta | Off decode tok/s | Active decode tok/s |",
    "|---:|---:|---:|---:|---:|---:|",
]
for pair, pdelta, ddelta in zip(pairs, prefill_deltas, decode_deltas):
    off = arms[pair, "off"]
    on = arms[pair, "on"]
    lines.append(
        f"| {pair} | {off['prefill_tok_s']:.1f} | {on['prefill_tok_s']:.1f} | {pdelta:+.2f}% | "
        f"{off['decode_tok_s']:.2f} | {on['decode_tok_s']:.2f} |"
    )

per_turn = {}
for row in rows:
    key = (int(row["pair"]), int(row["turn"]))
    per_turn.setdefault(key, {})[row["mode"]] = row
lines.extend([
    "",
    "## Per-turn prefill",
    "",
    "| Turn | Context tokens | Off median ms | Active median ms | Paired rate delta median | Pair deltas |",
    "|---:|---:|---:|---:|---:|:---|",
])
for turn in sorted({turn for _, turn in per_turn}):
    turn_deltas = []
    off_ms = []
    on_ms = []
    contexts = []
    for pair in pairs:
        modes = per_turn[pair, turn]
        off_value = float(modes["off"]["prefill_ms"])
        on_value = float(modes["on"]["prefill_ms"])
        contexts.append(int(modes["off"]["ctx"]))
        off_ms.append(off_value)
        on_ms.append(on_value)
        turn_deltas.append((off_value / on_value - 1.0) * 100.0)
    lines.append(
        f"| {turn} | {statistics.median(contexts):.0f} | {statistics.median(off_ms):.1f} | "
        f"{statistics.median(on_ms):.1f} | {statistics.median(turn_deltas):+.2f}% | "
        f"{', '.join(f'{value:+.2f}%' for value in turn_deltas)} |"
    )
(result_dir / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

rocm-smi --showuse --showmemuse --showpids >"$RESULT_DIR/rocm_smi_after.txt" 2>&1 || true
echo "results: $RESULT_DIR"
