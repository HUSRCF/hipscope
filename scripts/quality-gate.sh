#!/usr/bin/env bash
# MQ4 output quality gate.
#
# Re-runs greedy (temp=0, no sampling, no penalty) generation on a fixed
# matrix of (model × prompt) and compares token-ID outputs byte-exact
# against the committed baselines in tests/quality-baselines/.
#
# ANY divergence is a BUG. The phrase "model quality" or "sampling variance"
# is not an acceptable explanation for a failure here — these tests are
# fully deterministic.
#
# Modes:
#   ./scripts/quality-gate.sh              # full suite (9 tests, ~6 minutes)
#   ./scripts/quality-gate.sh --fast       # 4B Federalist only (~30 seconds)
#   ./scripts/quality-gate.sh --verbose    # show first divergence on fail
#   ./scripts/quality-gate.sh --update-baselines  # regenerate baselines
#
# Exit codes:
#   0   all tests passed
#   1   one or more tests failed
#   2   build or environment error
#
# The baselines live in tests/quality-baselines/ with format:
#   {model}_mq4_{prompt}.tokens       — raw token IDs, one per line
#   {model}_mq4_{prompt}.md5          — md5 of the .tokens file
#   {model}_mq4_{prompt}.head512.md5  — md5 of first 512 lines (quick-check)
#   BASELINE_COMMIT.txt               — reference commit hash

set -u
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
BASELINE_DIR="tests/quality-baselines"
MODELS_DIR="/home/kaden/ClaudeCode/autorocm/hipfire/models"
EXE="./target/release/examples/greedy_dump"
LOCK_SCRIPT="./scripts/gpu-lock.sh"

# Prompts, hardcoded here and also in the baselines.
PROMPT_compiler="Explain the difference between a compiler and an interpreter. Give a concrete example of each."
PROMPT_math="What is the square root of 144 multiplied by 3, minus 7?"
PROMPT_federalist="Write a 500-word essay about Federalist No. 10 by James Madison."

# Test matrix in execution order (fast → slow for early exit).
# Format: "model prompt"
FAST_TESTS=("4b federalist")
FULL_TESTS=(
    "0.8b compiler"   "0.8b math"   "0.8b federalist"
    "4b compiler"     "4b math"     "4b federalist"
    "9b compiler"     "9b math"     "9b federalist"
)

FAST=0
VERBOSE=0
UPDATE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --fast) FAST=1 ;;
        --verbose|-v) VERBOSE=1 ;;
        --update-baselines) UPDATE=1 ;;
        -h|--help)
            sed -n '3,25p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

color() {
    if [ -t 1 ]; then
        case "$1" in
            green)  printf '\033[32m%s\033[0m' "$2" ;;
            red)    printf '\033[31m%s\033[0m' "$2" ;;
            yellow) printf '\033[33m%s\033[0m' "$2" ;;
            bold)   printf '\033[1m%s\033[0m'  "$2" ;;
            *) printf '%s' "$2" ;;
        esac
    else
        printf '%s' "$2"
    fi
}

# Ensure the build exists and is up to date.
ensure_build() {
    if [ ! -x "$EXE" ] || [ "crates/engine/examples/greedy_dump.rs" -nt "$EXE" ] \
       || ! find crates/rdna-compute/src crates/engine/src kernels/src \
              -newer "$EXE" -print -quit 2>/dev/null | grep -q .; then
        echo "Building greedy_dump (release)..."
        cargo build --release -p engine --example greedy_dump --features deltanet 2>&1 \
            | grep -E '^(error|warning: unused|   Compiling)' | tail -10
        if [ ! -x "$EXE" ]; then
            color red "BUILD FAILED"; echo
            exit 2
        fi
    fi
}

# Run one test case. Returns token ID sequence file path.
# Args: model prompt_name
run_one() {
    local model="$1" prompt_name="$2"
    local out="/tmp/qg_${model}_${prompt_name}.tokens"
    local prompt_var="PROMPT_${prompt_name}"
    local prompt="${!prompt_var}"
    "$EXE" "$MODELS_DIR/qwen3.5-${model}.mq4" "$out" "$prompt" 2>/dev/null
    echo "$out"
}

# Compare output tokens to baseline, print PASS/FAIL.
# Args: model prompt_name
verify_one() {
    local model="$1" prompt_name="$2"
    local label
    label=$(printf "%s MQ4 %s" "$model" "$prompt_name")
    printf "  %-30s " "$label"

    local baseline_tokens="$BASELINE_DIR/${model}_mq4_${prompt_name}.tokens"
    local baseline_md5_file="$BASELINE_DIR/${model}_mq4_${prompt_name}.md5"
    if [ ! -f "$baseline_md5_file" ]; then
        color red "NO BASELINE"; echo
        return 1
    fi
    local baseline_md5
    baseline_md5=$(cat "$baseline_md5_file")

    local out
    out=$(run_one "$model" "$prompt_name")
    local got_md5
    got_md5=$(md5sum "$out" | cut -d' ' -f1)

    if [ "$got_md5" = "$baseline_md5" ]; then
        color green "PASS"
        printf " (%s)\n" "$(wc -l < "$out") tokens"
        rm -f "$out"
        return 0
    fi

    color red "FAIL"; echo
    echo "    expected md5: $baseline_md5"
    echo "    actual md5:   $got_md5"

    if [ "$VERBOSE" -eq 1 ]; then
        # First divergent line number (1-indexed), from the first diff hunk header.
        # diff header format: "NNNcMMM" or "NNNdMMM" or "NNNaMMM" → take the leading integer.
        local div_line
        div_line=$(diff "$baseline_tokens" "$out" | head -1 | sed -nE 's/^([0-9]+).*/\1/p')
        if [ -n "$div_line" ]; then
            local ctx_lo=$((div_line > 2 ? div_line - 2 : 1))
            local ctx_hi=$((div_line + 2))
            echo "    first divergence at line $div_line:"
            echo "    baseline context (lines $ctx_lo-$ctx_hi):"
            sed -n "${ctx_lo},${ctx_hi}p" "$baseline_tokens" | nl -ba -v "$ctx_lo" | sed 's/^/      /'
            echo "    actual output (lines $ctx_lo-$ctx_hi):"
            sed -n "${ctx_lo},${ctx_hi}p" "$out" | nl -ba -v "$ctx_lo" | sed 's/^/      /'
        fi
    fi
    # Keep failing output for post-mortem
    mv "$out" "/tmp/qg_FAIL_${model}_${prompt_name}.tokens"
    echo "    failed output preserved: /tmp/qg_FAIL_${model}_${prompt_name}.tokens"
    return 1
}

update_one() {
    local model="$1" prompt_name="$2"
    local label
    label=$(printf "%s MQ4 %s" "$model" "$prompt_name")
    printf "  %-30s " "$label"

    local out
    out=$(run_one "$model" "$prompt_name")

    local baseline="$BASELINE_DIR/${model}_mq4_${prompt_name}.tokens"
    local baseline_md5_file="$BASELINE_DIR/${model}_mq4_${prompt_name}.md5"
    local new_md5
    new_md5=$(md5sum "$out" | cut -d' ' -f1)

    if [ -f "$baseline_md5_file" ] && [ "$(cat "$baseline_md5_file")" = "$new_md5" ]; then
        color yellow "unchanged"; echo " ($(wc -l < "$out") tokens)"
    else
        cp "$out" "$baseline"
        echo "$new_md5" > "$baseline_md5_file"
        head -512 "$out" | md5sum | cut -d' ' -f1 > "$BASELINE_DIR/${model}_mq4_${prompt_name}.head512.md5"
        color green "UPDATED"; echo " ($(wc -l < "$out") tokens, md5 $new_md5)"
    fi
    rm -f "$out"
}

# -------- main --------

# GPU lock (mandatory — the bench/tests share the GPU)
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "quality-gate" || { color red "could not acquire GPU lock"; echo; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

ensure_build

if [ "$UPDATE" -eq 1 ]; then
    color bold "=== Updating MQ4 quality baselines ==="; echo
    color yellow "WARNING: This replaces all reference outputs. Only do this if the"; echo
    color yellow "         new outputs are CORRECT (verified), not just different."; echo
    echo
    for test_case in "${FULL_TESTS[@]}"; do
        read -r model prompt_name <<< "$test_case"
        update_one "$model" "$prompt_name"
    done
    cur_commit=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "$cur_commit" > "$BASELINE_DIR/BASELINE_COMMIT.txt"
    echo
    color bold "Baselines updated against commit $cur_commit"; echo
    echo "Review the changes with:  git diff tests/quality-baselines/"
    echo "Then commit them alongside your code change."
    exit 0
fi

if [ "$FAST" -eq 1 ]; then
    color bold "=== MQ4 Quality Gate (fast mode) ==="; echo
    tests=("${FAST_TESTS[@]}")
else
    color bold "=== MQ4 Quality Gate (full suite) ==="; echo
    tests=("${FULL_TESTS[@]}")
fi
echo "baseline commit: $(cat "$BASELINE_DIR/BASELINE_COMMIT.txt" 2>/dev/null || echo unknown)"
echo

pass=0
fail=0
for test_case in "${tests[@]}"; do
    read -r model prompt_name <<< "$test_case"
    if verify_one "$model" "$prompt_name"; then
        pass=$((pass+1))
    else
        fail=$((fail+1))
    fi
done
echo
if [ "$fail" -eq 0 ]; then
    color green "=== ALL ${pass} TESTS PASSED ==="; echo
    exit 0
fi

color red "=== ${fail} FAILED, ${pass} PASSED ==="; echo
echo
echo "This is a NUMERICAL CORRECTNESS BUG. Do not dismiss as sampling variance or"
echo "model quality — the tests are fully deterministic (greedy argmax, no sampling,"
echo "no repeat penalty)."
echo
echo "Investigation steps:"
echo "  1. Run with --verbose to see first divergence point"
echo "  2. Check if the change touched any GEMV, dispatch, rotation, or fusion code"
echo "  3. Compare against the commit in BASELINE_COMMIT.txt ($(cat $BASELINE_DIR/BASELINE_COMMIT.txt 2>/dev/null))"
echo "  4. If the change intentionally modifies output, run --update-baselines AND"
echo "     get the new outputs reviewed before committing them."
echo
exit 1
