#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fresh-server, uncached long-prefill A/B through the official serve harness.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-$HOME/.hipfire/models/qwen3.6-27b.mq4}"
GPU_ID="${GPU_ID:-0}"
BUN_BIN="${BUN_BIN:-}"
if [[ -z "$BUN_BIN" ]]; then
    BUN_BIN="$(command -v bun || true)"
fi
PROCESS_REPEATS="${PROCESS_REPEATS:-3}"
MIN_PREFILL_TOKENS="${MIN_PREFILL_TOKENS:-4096}"
MAX_SEQ="${MAX_SEQ:-32768}"
PORT="${PORT:-11520}"
SLEEP_BETWEEN_MODES="${SLEEP_BETWEEN_MODES:-20}"
SLEEP_BETWEEN_PAIRS="${SLEEP_BETWEEN_PAIRS:-30}"
PROMPT_SOURCE="${PROMPT_SOURCE:-benchmarks/prompts/longprose_multidoc.jsonl}"
RESULT_DIR="${RESULT_DIR:-benchmarks/results/qkvza_cold_serve_ab_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RESULT_DIR"

prompt_json="$RESULT_DIR/prompt.json"
python3 - "$PROMPT_SOURCE" "$prompt_json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
with source.open() as handle:
    row = json.loads(handle.readline())
prompt = row["filler_text"] + "\n\n" + row["question"]
destination.write_text(json.dumps([{"genre": "long_uncached", "prompt": prompt}]))
PY

printf 'pair\torder\tmode\tctx\tcached\tprefill_ms\tprefill_tok_s\teligible_events\troute_hit_events\n' >"$RESULT_DIR/raw.tsv"
{
    echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
    echo "git_branch=$(git branch --show-current 2>/dev/null || true)"
    echo "date=$(date -Is)"
    echo "model=$MODEL"
    echo "gpu_id=$GPU_ID"
    echo "bun_bin=${BUN_BIN:-auto}"
    echo "process_repeats=$PROCESS_REPEATS"
    echo "min_prefill_tokens=$MIN_PREFILL_TOKENS"
    echo "max_seq=$MAX_SEQ"
    echo "prompt_source=$PROMPT_SOURCE"
    echo "prompt_sha256=$(sha256sum "$PROMPT_SOURCE" | awk '{print $1}')"
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
    export HIPFIRE_QKVZA_SPLIT_TAIL_MIN_PREFILL_TOKENS="$MIN_PREFILL_TOKENS"
    export HIPFIRE_QKVZA_SPLIT_TAIL_DIAG=1
    if [[ "$mode" == "on" ]]; then
        export HIPFIRE_QKVZA_SPLIT_TAIL=1
    else
        unset HIPFIRE_QKVZA_SPLIT_TAIL
    fi

    bun_args=()
    if [[ -n "$BUN_BIN" ]]; then
        bun_args=(--bun "$BUN_BIN")
    fi
    python3 scripts/serve_harness.py \
        "${bun_args[@]}" \
        --model "$MODEL" \
        --tag qwen3.6:27b \
        --kv q8 \
        --mtp off \
        --thinking low \
        --max-tokens 4 \
        --max-seq "$MAX_SEQ" \
        --sampling recipe:nothink \
        --mode battery \
        --prompts-file "$prompt_json" \
        --port "$PORT" \
        --home "$cell/home" \
        --serve-log "$cell/serve.log" \
        --out "$cell/rows.json" \
        --seed 1 \
        2>&1 | tee "$cell/harness.log"

    python3 - "$pair" "$order" "$mode" "$cell" "$RESULT_DIR/raw.tsv" <<'PY'
import json
import pathlib
import re
import sys

pair, order, mode, cell, raw_path = sys.argv[1:]
cell = pathlib.Path(cell)
rows = json.loads((cell / "rows.json").read_text())
if len(rows) != 1:
    raise SystemExit(f"expected one serve row, got {len(rows)}")
row = rows[0]
log = (cell / "serve.log").read_text(errors="replace")
eligible = len(re.findall(r"qkvza_split_tail request eligible=true", log))
route_hits = len(re.findall(r"qkvza_split_tail route=hit", log))
if row.get("cached") != 0:
    raise SystemExit(f"expected cold prompt cached=0, got {row.get('cached')}")
if not isinstance(row.get("prefill_tok_s"), (int, float)):
    raise SystemExit(f"missing prefill timing: {row}")
if mode == "on" and (eligible == 0 or route_hits == 0):
    raise SystemExit("active serve did not report an eligible request and route hit")
if mode == "off" and (eligible != 0 or route_hits != 0):
    raise SystemExit("off serve unexpectedly reported an eligible request or route hit")
with pathlib.Path(raw_path).open("a") as handle:
    handle.write(
        f"{pair}\t{order}\t{mode}\t{row['ctx']}\t{row['cached']}\t"
        f"{row['prefill_ms']}\t{row['prefill_tok_s']}\t{eligible}\t{route_hits}\n"
    )
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

python3 - "$RESULT_DIR" <<'PY'
import csv
import pathlib
import statistics
import sys

result_dir = pathlib.Path(sys.argv[1])
with (result_dir / "raw.tsv").open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
by_pair = {}
for row in rows:
    by_pair.setdefault(int(row["pair"]), {})[row["mode"]] = float(row["prefill_tok_s"])
deltas = []
for pair, modes in sorted(by_pair.items()):
    if set(modes) != {"off", "on"}:
        raise SystemExit(f"incomplete pair {pair}: {modes}")
    deltas.append((modes["on"] / modes["off"] - 1.0) * 100.0)
off = [float(row["prefill_tok_s"]) for row in rows if row["mode"] == "off"]
on = [float(row["prefill_tok_s"]) for row in rows if row["mode"] == "on"]
ordered_deltas = sorted(deltas)

def percentile(values, q):
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction

report = (
    "# QKVZA cold user-facing serve A/B\n\n"
    f"- fresh-process pairs: {len(deltas)}\n"
    f"- off median: {statistics.median(off):.1f} tok/s\n"
    f"- active median: {statistics.median(on):.1f} tok/s\n"
    f"- cross-sample delta: {(statistics.median(on) / statistics.median(off) - 1.0) * 100.0:+.2f}%\n"
    f"- paired median delta: {statistics.median(deltas):+.2f}%\n"
    f"- paired delta IQR: {percentile(ordered_deltas, 0.25):+.2f}% to {percentile(ordered_deltas, 0.75):+.2f}%\n"
    f"- paired delta range: {min(deltas):+.2f}% to {max(deltas):+.2f}%\n"
    f"- positive pairs: {sum(delta > 0.0 for delta in deltas)}/{len(deltas)}\n"
    f"- pair deltas: {', '.join(f'{delta:+.2f}%' for delta in deltas)}\n"
)
(result_dir / "report.md").write_text(report)
print(report)
PY

echo "results: $RESULT_DIR"
