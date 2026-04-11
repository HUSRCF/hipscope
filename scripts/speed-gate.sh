#!/usr/bin/env bash
# speed-gate.sh — lightweight prefill / decode regression gate
#
# Runs bench_qwen35_mq4 on a small fixed config and compares
# prefill_tok_s + gen_tok_s against baselines committed to
# tests/speed-baselines/<arch>.txt. Fails if either regresses below
# (baseline * (1 - TOLERANCE)).
#
# Designed to run in ~10-15 seconds so it can be wired into the
# pre-commit hook alongside quality-gate.sh.
#
# Usage:
#   ./scripts/speed-gate.sh              — run and compare against baseline
#   ./scripts/speed-gate.sh --update     — regenerate baseline files from the current run
#   ./scripts/speed-gate.sh --verbose    — print full bench output on fail
#
# Baseline format: one `key=value` per line, e.g.
#     0.8b_mq4_prefill_tok_s=434
#     0.8b_mq4_gen_tok_s=360
#
# Baselines are per-architecture (gfx1100 / gfx1010 / gfx1030 / gfx1013).
# Detects the active GPU the same way quality-gate.sh does.

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

UPDATE=0
VERBOSE=0
for arg in "$@"; do
    case "$arg" in
        --update|--update-baselines) UPDATE=1 ;;
        --verbose) VERBOSE=1 ;;
        --help|-h)
            sed -n '2,24p' "$0"
            exit 0
            ;;
    esac
done

# ── Arch detection (same logic as quality-gate.sh) ────────────────────────
detect_arch() {
    if [ -n "${HIPFIRE_BASELINE_ARCH:-}" ]; then
        echo "$HIPFIRE_BASELINE_ARCH"
        return
    fi
    if [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
        case "$HSA_OVERRIDE_GFX_VERSION" in
            10.1.0) echo "gfx1010"; return ;;
            10.3.0) echo "gfx1030"; return ;;
            11.0.0) echo "gfx1100"; return ;;
        esac
    fi
    if command -v amdgpu-arch >/dev/null 2>&1; then
        amdgpu-arch 2>/dev/null | head -1
        return
    fi
    if command -v /opt/rocm/llvm/bin/offload-arch >/dev/null 2>&1; then
        /opt/rocm/llvm/bin/offload-arch 2>/dev/null | head -1
        return
    fi
    echo "gfx1100"  # fallback
}
ARCH="$(detect_arch)"
BASELINE_DIR="$ROOT_DIR/tests/speed-baselines"
BASELINE_FILE="$BASELINE_DIR/${ARCH}.txt"
TOLERANCE="${SPEED_GATE_TOLERANCE:-0.05}"  # 5% default

# ── Model selection ──────────────────────────────────────────────────────
MODEL="${SPEED_GATE_MODEL:-$HOME/.hipfire/models/qwen3.5-0.8b.mq4}"
if [ ! -f "$MODEL" ]; then
    echo "[speed-gate] model not found: $MODEL"
    echo "             set SPEED_GATE_MODEL to override (expects a .mq4 file)"
    exit 2
fi

# ── Build bench binary ───────────────────────────────────────────────────
echo "Building bench_qwen35_mq4 (release)..."
cargo build --release --features deltanet --example bench_qwen35_mq4 -p engine 2>&1 \
    | grep -E "error|warning:.*unused" || true

BENCH_BIN="$ROOT_DIR/target/release/examples/bench_qwen35_mq4"
if [ ! -x "$BENCH_BIN" ]; then
    echo "[speed-gate] bench_qwen35_mq4 binary not found after build"
    exit 2
fi

# ── Run bench with fixed config ──────────────────────────────────────────
# Small/fast config: 128-token prefill, 20-token warmup, 100-token gen.
# Captures both prefill_tok_s and gen_tok_s in the SUMMARY line.
echo "=== Speed Gate (arch=$ARCH) ==="
RAW_OUTPUT="$(mktemp)"
trap 'rm -f "$RAW_OUTPUT"' EXIT

"$BENCH_BIN" "$MODEL" --prefill 128 --warmup 20 --gen 100 > "$RAW_OUTPUT" 2>&1 || {
    echo "[speed-gate] bench_qwen35_mq4 failed — see output:"
    cat "$RAW_OUTPUT"
    exit 1
}

SUMMARY_LINE="$(grep -E '^SUMMARY' "$RAW_OUTPUT" || true)"
if [ -z "$SUMMARY_LINE" ]; then
    echo "[speed-gate] could not find SUMMARY line in bench output"
    cat "$RAW_OUTPUT"
    exit 1
fi

# Parse SUMMARY line:
#  SUMMARY  gen_tok_s=359.7  bw_gib_s=184.0  prefill_tok_s=433.8  avg_ms=2.55  p50_ms=2.55
PREFILL_TOK_S="$(echo "$SUMMARY_LINE" | grep -oE 'prefill_tok_s=[0-9.]+' | cut -d= -f2)"
GEN_TOK_S="$(echo "$SUMMARY_LINE" | grep -oE 'gen_tok_s=[0-9.]+' | cut -d= -f2)"

printf '  0.8b MQ4 prefill: %7.1f tok/s\n' "$PREFILL_TOK_S"
printf '  0.8b MQ4 gen    : %7.1f tok/s\n' "$GEN_TOK_S"

# ── Update mode: write out and exit ──────────────────────────────────────
if [ "$UPDATE" -eq 1 ]; then
    mkdir -p "$BASELINE_DIR"
    cat > "$BASELINE_FILE" <<EOF
# hipfire speed-gate baseline — $ARCH
# Captured $(date -u +%Y-%m-%dT%H:%M:%SZ) from HEAD $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
# Tolerance: ${TOLERANCE} (fail if any metric drops below baseline * (1 - tolerance))
0.8b_mq4_prefill_tok_s=$PREFILL_TOK_S
0.8b_mq4_gen_tok_s=$GEN_TOK_S
EOF
    echo
    echo "Updated baseline: $BASELINE_FILE"
    exit 0
fi

# ── Compare against baseline ────────────────────────────────────────────
if [ ! -f "$BASELINE_FILE" ]; then
    echo
    echo "[speed-gate] no baseline file at $BASELINE_FILE"
    echo "             run with --update to create one from current run"
    exit 2
fi

read_baseline() {
    local key="$1"
    grep -E "^${key}=" "$BASELINE_FILE" 2>/dev/null | cut -d= -f2 | head -1
}

PREFILL_BASELINE="$(read_baseline '0.8b_mq4_prefill_tok_s')"
GEN_BASELINE="$(read_baseline '0.8b_mq4_gen_tok_s')"

FAILED=0

compare() {
    local label="$1" actual="$2" baseline="$3"
    if [ -z "$baseline" ]; then
        echo "  $label: NO BASELINE (skipping)"
        return
    fi
    local floor
    floor="$(awk -v b="$baseline" -v t="$TOLERANCE" 'BEGIN { printf "%.2f", b * (1 - t) }')"
    local pct
    pct="$(awk -v a="$actual" -v b="$baseline" 'BEGIN { printf "%+.1f", (a - b) / b * 100 }')"
    if awk -v a="$actual" -v f="$floor" 'BEGIN { exit !(a < f) }'; then
        printf '  %-24s %7.1f tok/s  baseline=%s floor=%s  %s%%  FAIL\n' \
            "$label" "$actual" "$baseline" "$floor" "$pct"
        FAILED=1
    else
        printf '  %-24s %7.1f tok/s  baseline=%s  %s%%  OK\n' \
            "$label" "$actual" "$baseline" "$pct"
    fi
}

echo
echo "=== Speed Gate compare (tolerance=$TOLERANCE) ==="
compare "0.8b MQ4 prefill" "$PREFILL_TOK_S" "$PREFILL_BASELINE"
compare "0.8b MQ4 gen"     "$GEN_TOK_S"     "$GEN_BASELINE"

if [ "$FAILED" -eq 1 ]; then
    echo
    echo "========================================================================"
    echo "COMMIT BLOCKED: speed regression detected."
    echo "========================================================================"
    echo
    echo "One or more metrics dropped below baseline * (1 - $TOLERANCE)."
    if [ "$VERBOSE" -eq 1 ]; then
        echo
        echo "Full bench output:"
        cat "$RAW_OUTPUT"
    fi
    echo
    echo "If the regression is INTENTIONAL (new kernel path, lower-precision"
    echo "tradeoff, etc.), re-baseline with:"
    echo "  ./scripts/speed-gate.sh --update"
    echo "and commit the updated tests/speed-baselines/${ARCH}.txt."
    exit 1
fi

echo
echo "=== speed gate passed ==="
