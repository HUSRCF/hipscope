# Size-limit CI gates — preventing oversized files + impl blocks

**Status:** plan rev-2 (2026-05-09). Combined-review findings folded (`size-limit-ci-gates-rev-plan-combined.md`); empirical recon added; Layer 4 promoted from optional to mandatory (baseline-tracked exemptions, per Gemini's ratchet finding); Layers 2 + 3 marked deferred until CI exists.
**Tracking:** TBD (open issue once this plan lands).
**Companion plan:** `oversize-files-modularization.md` rev-2 (cleans up the existing oversized files; this plan prevents regression).

## Why

The companion plan describes splitting the existing god-files (`dispatch.rs` 15K, `qwen35.rs` 7K, `speculative.rs` 4.9K, `llama.rs` 4K, `hipfire-quantize/main.rs` 3.2K). Without enforcement, the codebase will re-grow into the same shape within a quarter — particularly because much of the new code lands via AI agents, which tend to append to existing files rather than create new modules unless explicitly told otherwise.

Goal: a small, layered set of gates that make oversized growth louder than splitting, without producing false-positive friction on legitimate work.

## Critical context: no CI exists today

Verified empirically (`find . -name "*.yml" -o -name "*.yaml"`): no `.github/workflows/`, no other CI config. **All quality gates run via the local pre-commit hook (`.githooks/pre-commit`).**

This invalidates the rev-1 framing of Layers 2 and 3 (clippy warning surface + PR-comment size delta) as "ready to wire up." Two paths in rev-2:

- **Path A (current default):** keep Layers 2 + 3 as **deferred** — they're documented but the work waits until CI exists. Layer 0, Layer 1, Layer 1.5, and Layer 4 all run locally and are landable today.
- **Path B (alternative):** fold soft-warning machinery into the local pre-commit hook as non-blocking stderr output. Trades CI-portability for immediate value.

This plan adopts Path A; Layer 2/3 sections are tagged `(deferred until CI)`.

## Empirical baseline (rev-2 recon)

Numbers `find crates -name "*.rs" -not -path "*/target/*" -not -path "*/examples/*" | xargs wc -l` reveals:

```
Files >= 2500 lines (current):  5 (the known offenders)
Files in 2000-2499 (close):     0
Largest non-grandfathered:      ~1700 (pflash.rs, triattn.rs)
```

**Why 2500 is the right cap:** no legitimate library file is between 2000 and 2499 today; only the 5 known oversized files are above. Cap is set to grandfather the known set without flagging anything legitimate. 1500 is the *target* (per the modularization plan); 2500 is the *hard floor* (this plan's hook).

**Generated `.rs` files in the project:** none today (`find . -name "build.rs"` returns 0). Forward-looking exclusion rule still added (M8) for safety.

**`#[cfg(test)] mod tests` blocks:** only in `llama.rs`. Most files have no inline tests.

## Non-goals

1. **Don't enforce style preferences as hard gates.** No clippy-deny on `cognitive_complexity` or `too_many_lines` crate-wide. False-positive cost crushes the signal.
2. **Don't write a custom AST checker before the cheap rules prove insufficient.** A bash hook on file size catches 80% of the failure mode; a coarse `grep -c` for fn-count adds another 10%; full AST tooling is the last 10% and not worth the maintenance cost yet.
3. **Don't grandfather forever.** Existing offenders get exemptions tracked in `.size-limit-exemptions` with **baselines, not just presence**. Files cannot grow past their recorded baseline (the ratchet). Splits land; baselines remove or shrink.

## Layered gates

### Layer 0 — CLAUDE.md rule (the load-bearing one for agents)

`CLAUDE.md` is the only documentation an AI agent reads before each session. Adding the rule there is what changes agent behavior at the moment-of-writing, before any hook fires.

**Position:** add this section after the "Rules" block (around line ~350 in the current 398-line CLAUDE.md), as a peer to "Coherence Gate" and "GPU Lock Protocol" — those are also project-rules-with-enforcement.

**Section content:**

```markdown
## File and impl-block size

**Target:** ≤1500 lines per `.rs` file in `crates/**/src/`. This matches
the modularization plan's split target. Files between 1500–2500 should
be flagged for near-term splitting in PR review.

**Hard cap:** 2500 lines per file, enforced by pre-commit hook. The
hook also enforces a per-file growth ratchet against
`.size-limit-exemptions` for grandfathered files (existing files
above the cap can shrink but cannot grow past their recorded baseline).

New code that would push a file over the cap must go in a sibling
module (`foo_bar.rs` next to `foo.rs`, or `foo/bar.rs` if `foo` is a
directory module).

**Splits must be semantic, not numerical.** Files like `foo_part2.rs`,
`helpers.rs`, or `overflow.rs` containing arbitrary surplus from a
parent file are forbidden — they pass the line cap but make the
codebase worse. Split by responsibility (FFI / state / dispatch /
serialization / kernel-family / etc.). If a file approaches the cap
without an obvious semantic boundary, that's a signal to refactor
the underlying API, not to invent an arbitrary split. The companion
plan (`oversize-files-modularization.md`) has examples.

Single `impl Foo` blocks should not exceed **~50 methods** as a
guideline. The pre-commit hook surfaces a soft warning when a single
file's `^    (pub )?fn ` count exceeds 60 (god-impl signal). Splitting
an `impl` across multiple files is supported via `impl super::Foo {}`
in sibling modules:

    // src/foo/dispatch.rs
    impl super::Foo {
        pub fn dispatch_a() {}
        pub fn dispatch_b() {}
    }
    // src/foo/state.rs
    impl super::Foo {
        pub fn save() {}
        pub fn load() {}
    }

When extending an existing oversized file (see `.size-limit-exemptions`
for the grandfathered list), prefer creating a new sibling module
rather than appending. The pre-commit hook will reject growth past
the recorded baseline.
```

This is the highest-leverage line item in the entire plan. It costs nothing to add and changes behavior on the next agent session.

### Layer 1 — pre-commit file-size cap (the hard gate)

`.githooks/pre-commit` already exists for the coherence gate. The current hook (verified by reading it) has this structure:

```
line 38:  set -e
line 41:  CHANGED=$(git diff --cached --name-only --diff-filter=ACM)
line 67:  if ! echo "$CHANGED" | grep -qE "$HOTSPOT"; then
line 68:      exit 0    ← non-hotspot commits early-exit here
line 69:  fi
```

**Insertion point** (CRITICAL): the size check goes between `set -e` and `CHANGED=...` — i.e., AHEAD of the HOTSPOT early-exit, so it runs on every commit (doc-only, config-only, anything). If placed after the early-exit, the gate misses non-hotspot commits, allowing a 5000-line `utils.rs` to land if it doesn't match any kernel keyword.

**Hook fragment to add (rev-2):**

```bash
# .githooks/pre-commit (insert between `set -e` at line 38 and `CHANGED=...` at line 41)

# ── File-size gate (universal — runs on every commit, ahead of HOTSPOT early-exit) ──
MAX_RS_LINES=2500
SIZE_VIOLATIONS=""
SIZE_GROWTH_VIOLATIONS=""
EXEMPTIONS_FILE=".size-limit-exemptions"

# Iterate staged .rs files (Added, Copied, Modified, Renamed).
while IFS= read -r f; do
    case "$f" in *.rs) ;; *) continue ;; esac
    [ -f "$f" ] || continue

    # Skip generated files (forward-looking; no build.rs in tree today).
    if head -n 5 "$f" 2>/dev/null | grep -qE "^// (@generated|Code generated by)"; then
        continue
    fi

    # Read both staged and working-tree line counts; check the larger.
    # Catches `git add -p` cases where staged != working-tree.
    n_staged=$(git show ":$f" 2>/dev/null | wc -l)
    n_work=$(wc -l < "$f")
    n=$(( n_staged > n_work ? n_staged : n_work ))

    # Look up baseline in .size-limit-exemptions (if present).
    baseline=""
    if [ -f "$EXEMPTIONS_FILE" ]; then
        baseline=$(awk -v p="$f" '$1==p {print $2}' "$EXEMPTIONS_FILE")
    fi

    if [ -n "$baseline" ]; then
        # Exempted file — apply growth ratchet (allow +50 lines tolerance).
        if [ "$n" -gt "$((baseline + 50))" ]; then
            SIZE_GROWTH_VIOLATIONS="$SIZE_GROWTH_VIOLATIONS  $f: now $n lines (baseline $baseline; cannot grow past +50)"$'\n'
        fi
        continue
    fi

    if [ "$n" -gt "$MAX_RS_LINES" ]; then
        # Honor in-file NOLINT exemption — must include issue #N tracking link.
        # Permissive regex: tolerates `//NOLINT`, `// NOLINT`, lowercase, etc.
        if head -n 20 "$f" | grep -qiE "^[[:space:]]*//[[:space:]]*nolint:?[[:space:]]*file-?size.*issue[[:space:]]*#[0-9]+"; then
            continue
        fi
        SIZE_VIOLATIONS="$SIZE_VIOLATIONS  $f: $n lines (cap $MAX_RS_LINES)"$'\n'
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR)

if [ -n "$SIZE_VIOLATIONS" ] || [ -n "$SIZE_GROWTH_VIOLATIONS" ]; then
    echo "ERROR: file-size cap violation:" >&2
    [ -n "$SIZE_VIOLATIONS" ] && printf '%s' "$SIZE_VIOLATIONS" >&2
    [ -n "$SIZE_GROWTH_VIOLATIONS" ] && {
        echo "  ── Exempted files growing past baseline: ──" >&2
        printf '%s' "$SIZE_GROWTH_VIOLATIONS" >&2
    }
    cat <<'EOF' >&2

Either:
  1. Split the file into a module directory:
       mv src/foo.rs src/foo/mod.rs
       (then mkdir + move sub-content into src/foo/<topic>.rs)
       Splits must be semantic, not numerical (foo_part2.rs is forbidden).
  2. Add a justified exemption AT THE TOP of the file (within first 20 lines):
       // NOLINT: file-size <reason> issue #<N>
     Exemptions MUST cite a tracked issue number. The hook validates.
  3. For grandfathered files, update .size-limit-exemptions with a new
     baseline ONLY if the growth represents legitimate work-in-progress
     during a split PR. Otherwise: split.

See docs/plans/size-limit-ci-gates.md for the policy.
EOF
    exit 1
fi
```

**Key changes vs rev-1:**
- `--diff-filter=ACMR` (was AM) — matches existing hook's ACM convention plus adds R for renames.
- Read both staged + working-tree, check the larger — catches `git add -p` partial stages.
- Permissive NOLINT regex (case-insensitive, tolerates spacing variations).
- **NOLINT exemption requires `issue #N` reference** — raises the cost of writing a fake exemption.
- `head -n 20` (was 5) — accommodates module doc-comments at the top of the file.
- Skips files with `// @generated` or `// Code generated by` markers.
- `.size-limit-exemptions` consulted; exempted files allowed but cannot grow past baseline + 50 lines (the ratchet).

**Atomicity rule (CRITICAL): Layer 1 hook + grandfathering exemption file land in the same PR.** If the hook lands without the exemption file populated, the next commit touching any of the 5 known oversized files breaks the working tree. Both go in together.

### Layer 1.5 — fn-count soft warning (NEW in rev-2)

The CLAUDE.md "~50 methods per impl" guideline is otherwise unenforced. Add a coarse soft warning to the same hook fragment:

```bash
# Append to the hook fragment above, before the violation summary:

FN_VIOLATIONS=""
while IFS= read -r f; do
    case "$f" in *.rs) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    n_fns=$(grep -cE "^    (pub )?fn " "$f" 2>/dev/null || echo 0)
    if [ "$n_fns" -gt 60 ]; then
        FN_VIOLATIONS="$FN_VIOLATIONS  $f: $n_fns methods (soft warn at >60)"$'\n'
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR)

if [ -n "$FN_VIOLATIONS" ]; then
    echo "warning: file(s) have many methods (god-impl signal):" >&2
    printf '%s' "$FN_VIOLATIONS" >&2
    echo "  (this is a non-blocking soft signal — split if natural)" >&2
fi
# ... (continues to the hard-fail size-violation summary above)
```

The `^    (pub )?fn ` regex is fragile (won't catch nested fns, lambdas, fns with non-standard indentation), but works as a coarse signal for god-impl detection — empirically returns 324 for dispatch.rs (matches actual count) and 15 for qwen35.rs (most fns are free, not in impl blocks). Soft warning, never blocks.

### Layer 2 — clippy.toml + CI warning surface (deferred until CI exists)

A workspace-level `clippy.toml`:

```toml
# clippy.toml — workspace root
cognitive-complexity-threshold = 30
too-many-lines-threshold = 200
too-many-arguments-threshold = 8
```

**Empirical recon required before turning on (rev-2 hardening):** before wiring this up to fail-as-warnings in CI, run `cargo clippy --workspace -- -W clippy::too_many_lines` against the current codebase and count warnings. If >50, the threshold is too tight for the existing codebase — either:

- Bump thresholds higher (e.g., `too-many-lines-threshold = 500`) until the wall-of-warnings shrinks to ~20.
- Add `#[expect(clippy::too_many_lines, reason = "grandfathered before split")]` annotations to grandfathered functions.
- Accept the wall as a one-time event during the modularization period; thresholds tighten after splits land.

**CI step (when CI exists):**

```yaml
# .github/workflows/lint.yml — assumes CI infrastructure is in place
- name: clippy size signals
  run: cargo clippy --workspace --all-targets -- \
         -W clippy::cognitive_complexity \
         -W clippy::too_many_lines \
         -W clippy::too_many_arguments
```

**Surface only NEW warnings vs base** (Gemini's recommendation; reduces noise from grandfathered code). Use `clippy-sarif` or equivalent that diffs warnings against the base branch and emits only deltas to the PR review.

**Don't use `actions-rs/clippy-check`** — unmaintained since 2023. Use a maintained alternative (e.g., `clechasseur/rs-clippy-check`) or run clippy directly with `--message-format=json` and parse in-step.

**Why warnings instead of denies:** the clippy thresholds catch genuine smells but also produce false positives on legitimate state-machine code, parser tables, and one-off CLI dispatch. Hard-failing on those creates churn without proportional safety. Layer 1 (file-size cap) is the actual gate; clippy is the soft signal.

For genuinely strict gates on critical hot paths, opt in per-function (post-split, when the function exists at the right size):

```rust
// (apply this AFTER a function has been split to a small, intentionally-tight size)
#[deny(clippy::too_many_lines)]
pub fn hot_decode_inner_loop(...) -> HipResult<()> {
    // ...
}
```

(Don't apply to current functions like `forward_scratch` that are already over 200 lines — would fail the build.)

### Layer 3 — informational PR-comment job (deferred until CI exists)

When CI exists, surface absolute file sizes for files modified in the PR (not just diffs):

```yaml
# CI step: surface absolute file sizes for any oversized .rs files in the PR
- name: oversized-file watch
  run: |
    git fetch origin ${{ github.base_ref }}
    base=$(git merge-base HEAD origin/${{ github.base_ref }})
    echo "## Absolute size of modified .rs files (warn at > 1500)" >> $GITHUB_STEP_SUMMARY
    for f in $(git diff --name-only $base...HEAD -- '*.rs'); do
        if [ -f "$f" ]; then
            n=$(wc -l < "$f")
            if [ "$n" -gt 1500 ]; then
                echo "  $f: now $n lines (warning)" >> $GITHUB_STEP_SUMMARY
            fi
        fi
    done
```

(rev-1's bash used `git diff --stat` which only shows added/removed lines, not absolute sizes — fixed in rev-2.)

CI-host-agnostic phrasing: on GitHub Actions this writes to `$GITHUB_STEP_SUMMARY`; adapt to whichever CI runs the project.

### Layer 4 — `.size-limit-exemptions` baseline database (MANDATORY in rev-2)

**Promoted from optional to mandatory** because the hook's growth ratchet (Layer 1) reads from this file. Without it, exempted files can grow indefinitely.

**Format:**

```
# .size-limit-exemptions — baseline-tracked exemptions for files
# above the 2500-line hard cap. The hook checks current_lines <=
# baseline + 50 (tolerance for legitimate small additions). Exempted
# files cannot grow past their baseline, only shrink.
#
# Each line: <path> <baseline_lines> <issue>
# Lines starting with # are comments.

crates/rdna-compute/src/dispatch.rs                            15137  issue #N
crates/hipfire-arch-qwen35/src/qwen35.rs                        7202  issue #N
crates/hipfire-arch-qwen35/src/speculative.rs                   4889  issue #N
crates/hipfire-runtime/src/llama.rs                             4029  issue #N
crates/hipfire-quantize/src/main.rs                             3253  issue #N
```

**Location:** `.size-limit-exemptions` at repo root (alternative: `.github/size-limit-exemptions` if/when `.github/` is created for CI).

**Lifecycle:**
- Each modularization PR (per companion plan) removes its file's entry from `.size-limit-exemptions`.
- A file's entry can be added (at a new baseline) only when a tracked issue justifies legitimate work-in-progress — e.g., temporary growth during a multi-PR split sequence.
- After all 5 files split, `.size-limit-exemptions` is empty (or ideally deleted).

### Layer 4.5 — stale-exemption catcher (NEW in rev-2; soft warning)

When a modularization PR splits a file under the cap, the author may forget to remove the in-file `// NOLINT:` header. Stale exemptions accumulate invisibly.

**Hook fragment (or CI step when CI exists):**

```bash
# Soft-warn when an in-file NOLINT exemption header is present on a file
# that's actually under the cap (i.e., the exemption is stale).
while IFS= read -r f; do
    case "$f" in *.rs) ;; *) continue ;; esac
    [ -f "$f" ] || continue
    n=$(wc -l < "$f")
    if [ "$n" -le "$MAX_RS_LINES" ] && head -n 20 "$f" | grep -qiE "nolint:?[[:space:]]*file-?size"; then
        echo "warning: $f has NOLINT: file-size header but is only $n lines (stale exemption — remove the header)" >&2
    fi
done < <(git diff --cached --name-only --diff-filter=ACMR)
```

Soft warning, never blocks.

### Layer 5 — explicitly NOT done

- **No custom dylint for impl-block size.** The Layer 1.5 grep-based fn-count check covers the same failure mode at coarser granularity. Custom dylint is ~150 LOC of new infrastructure with its own maintenance cost; payoff doesn't justify it unless we hit a god-impl in a small file (so far we haven't).
- **No clippy-deny on `cognitive_complexity` crate-wide.** Causes more churn than it prevents.
- **No "method count per impl" precise enforcement.** The Layer 1.5 grep is the coarse approximation; full AST-aware counting is overkill.
- **No DRY-style lints.** False-positive disasters in language-model code.

## Sequencing

| # | Step | Cost | Owner | Depends on |
|---|---|---|---|---|
| 1 | Add `## File and impl-block size` section to CLAUDE.md (after `## Rules`) | 5 min | anyone | — |
| 2 | **Atomic PR**: add file-size + ratchet fragment to `.githooks/pre-commit` AND populate `.size-limit-exemptions` with the 5 baselines AND add NOLINT headers to those files | 1 hour | anyone | (1) |
| 3 | Add Layer 1.5 fn-count soft-warning fragment | 15 min | anyone | (2) |
| 4 | Add Layer 4.5 stale-exemption catcher | 15 min | anyone | (2) |
| 5 | Write `tests/test-pre-commit-size-cap.sh` fixture script (validation) | 30 min | anyone | (2) |
| 6 | (Deferred until CI exists) clippy.toml + Layer 2 + Layer 3 | 1 hour | CI owner | CI infrastructure |

**Critical path:** (1) + (2) gives the bulk of the prevention value. ~1h5min of work. Steps (3)+(4)+(5) are incremental hardening; step (6) is deferred.

**Atomicity for step 2 (NOTE):** the hook + exemptions file + NOLINT headers MUST land in one PR. Otherwise the gate breaks the working tree on the first subsequent commit touching any of the 5 oversized files.

## Validation

**`tests/test-pre-commit-size-cap.sh`** (new in rev-2): scripted test fixture covering the hook's behavior. ~30 lines of bash creating temp files and invoking the hook against them:

1. Fixture: 2501-line file without exemption → hook should reject.
2. Fixture: 2501-line file with valid `// NOLINT: file-size <reason> issue #1` header → hook should pass.
3. Fixture: 2501-line file with malformed header (`// NOLINT: file-size` no issue link) → hook should reject.
4. Fixture: 1500-line file → hook should pass silently.
5. Fixture: file in `.size-limit-exemptions` at baseline → hook should pass.
6. Fixture: file in `.size-limit-exemptions` at baseline+51 → hook should reject (ratchet).
7. Fixture: 100-line file with stale `// NOLINT:` header → hook should warn (Layer 4.5).
8. Fixture: file with `// @generated` header at 5000 lines → hook should pass (skip generated).

After Layer 0 + Layer 1 land:

1. Run `tests/test-pre-commit-size-cap.sh`. All 8 cases pass.
2. Run the hook against the existing tree on a no-op commit: should be silent (the 5 grandfathered files are in `.size-limit-exemptions` and at baseline; nothing else exceeds the cap).
3. (Deferred until CI exists) Validate Layer 2: confirm clippy warnings appear in CI for the existing oversized functions; CI does NOT fail (warnings only).

(rev-1's untestable validation step on `--no-verify` usage dropped — git history doesn't record `--no-verify` usage. The existing hook already says "Do NOT bypass with --no-verify" at line 35; that doc is the policy.)

## Honest expectation

Hooks and CI gates do not stop a determined human (`--no-verify`, `git push --force`, etc.). Their value is:

- **For agents:** the friction at commit-time forces an explicit alternative path (split the file). The CLAUDE.md rule + the loud failure message together make "split the file" the path of least resistance. The NOLINT-with-issue-link requirement raises the cost of writing a fake exemption (the agent must invent a plausible issue number, which is immediately suspicious in review).
- **For humans:** a visible nudge during code review that "this file is already oversized; can this go in a sibling module instead?". The Layer 1.5 fn-count soft warning surfaces god-impl growth.

The combined system is friction calibrated to encourage modularization, not a wall preventing all growth. The actual cleanup happens via the companion modularization plan; these gates are how we don't undo that work over the next 12 months.

## Definition of done

**Final end-state:**

- Layer 0 (CLAUDE.md), Layer 1 (pre-commit hook + Layer 1.5 fn-count + Layer 4.5 stale-exemption catcher), and Layer 4 (`.size-limit-exemptions` populated) all landed.
- After modularization completes (companion plan), `.size-limit-exemptions` is empty (or deleted), no in-file `// NOLINT: file-size` headers remain anywhere in the tree, and every `.rs` file in `crates/**/src/` is ≤1500 lines (per the modularization target). Hook gate is meaningful — every cap violation is a real signal.

**Interim milestones (not yet definition-of-done):**

- *Hook is live, 5 files exempted via baselines:* atomic PR landed (CLAUDE.md + hook + exemptions file + per-file NOLINT headers).
- *Splits in progress:* exemption count decreases as each modularization PR lands. Each split PR removes its file's entry from `.size-limit-exemptions` AND removes the in-file `// NOLINT:` header.
- *(Deferred)* When CI infrastructure is set up, Layers 2 + 3 ship as additional soft signals; a hard-blocking file-size CI job is added as backup for fork contributors who don't have `core.hooksPath` configured.

## References

- `oversize-files-modularization.md` rev-2 — the cleanup plan this gate prevents regression of.
- `.githooks/pre-commit` lines 38–69 — existing hook structure that determines insertion point.
- `AGENTS.md:30` — coherence-gate-dflash is canonical (not directly relevant to this plan, but cross-referenced for AGENTS.md alignment).
