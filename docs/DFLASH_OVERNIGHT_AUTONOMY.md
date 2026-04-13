# DFlash Overnight Autonomy Contract

**Audience:** the agent that picks up the 0.1.6 dflash build after a
context compaction or fresh session.

**Your job:** reach a working MVP of DFlash speculative decoding — a
loop that generates tokens with draft + target, verifies them, and
emits accepted tokens — by the morning of 2026-04-14. Ideally more.
Do not stop until you ship something reviewable.

**Branch you work on:** `dflash` (never `master`). The branch was cut
from master at commit `6383440` so master is your safe rollback point.
All commits push to `origin/dflash`. Final merge to master is a human
review step the user does in the morning. If you accidentally commit
to master, `git reset --hard origin/master` + cherry-pick onto dflash.

## Stop conditions (exhaustive)

You may stop only if ONE of these is true:

1. **Phase 8 complete.** v0.1.6 tagged, pushed, GitHub release live,
   benchmarks in BENCHMARKS.md, docs/SPECULATIVE_DECODING.md exists.
2. **Fundamental technical block** no amount of additional work can
   unblock in this session. Before declaring this, you must:
   - Have committed at least one intermediate artifact (a partial
     converter, a stub forward, a doc of what you learned)
   - Have written `docs/DFLASH_BLOCKED.md` describing exactly where
     you stalled and what the user should do to unblock
   - Have pushed to `master` so the user wakes up to the state
3. **MVP floor reached** (Phase 1-5 end-to-end, greedy-only, single
   prompt works) AND you have 2+ hours of additional work already
   committed toward later phases. Phase 1-5 without continuation is
   NOT a valid stop — push further.

## What "MVP" means here

A working implementation capable of:
- Loading target (Qwen 3.5 9B MQ4) + draft (BF16 or MQ4)
- Running a multi-turn conversation via `hipfire run` or a test binary
- Producing byte-exact the same greedy output as non-dflash path
- Measuring end-to-end tok/s and accept-rate

Speed gain is not required for MVP. Correctness + the loop running is
the floor. Speed comes in Phase 7-8 tuning.

## Autonomy rules

### Do not ask clarifying questions

Make judgment calls. Document them in commit messages. Examples:

- "Phase 1 doc revealed the draft uses 4 denoising steps. Chose to
  implement 1-step greedy approximation first for MVP; 4-step correct
  path queued as follow-up."
- "Draft forward matches Qwen 3.5 9B architecture up to layer 12 then
  diverges. Implementing Qwen-only path; the divergent suffix is noted
  in dflash.rs TODO at line 234."
- "MQ4 draft quantize breaks accept-rate (0.78 → 0.31). Shipping with
  BF16 draft for 0.1.6; MQ4 quantize deferred to 0.1.7."

### Commit + push after every phase — to the `dflash` branch

Per-phase commits even if phase isn't finished. The user wakes up to
incremental progress, not a 6-hour uncommitted diff.

```bash
git branch --show-current   # must say "dflash" before every commit
git push origin dflash      # not master
```

Never push to master. If you discover you're on master:

```bash
git branch                  # see which branches exist
git stash                   # save any in-flight changes
git checkout dflash
git stash pop
```

Commit format:
```
feat(dflash): Phase N — <one line>

<what shipped, 1-3 bullets>
<what's deferred, 1-3 bullets>
<any surprises found>

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Push after every commit. Build will gate commits through the quality +
speed gates — if a gate fails, investigate, don't bypass with
`--no-verify` unless you're certain it's a false positive (and document
in the commit).

### Never break existing code

All dflash work is **additive**. The non-dflash code path
(`forward_scratch`, `forward_prefill_batch`, MQ4/MQ6 weights, asym3 KV)
stays byte-exact. Greedy parity is the quality-gate invariant.

If you're modifying existing code (not just adding new), you're doing
something wrong. Revisit the approach.

### Scope triage on each phase boundary

At each phase transition, ask yourself in writing (as a comment in
the commit or in DFLASH_PROGRESS.md):

1. How much time did the prior phase take?
2. Is MVP still reachable by morning?
3. Should I cut scope on the next phase? If so, what's the minimum to
   keep the loop coherent?

Scope cuts are encouraged. Ship a tight MVP, not a sprawling mess.

### Never commit secrets

No HF tokens, no API keys. If you create a `.env` or similar, add it
to `.gitignore` first.

### Log as you go

Keep `docs/DFLASH_PROGRESS.md` updated with a running log:

```markdown
## Phase 1 — architecture scope check
- Started: <time>
- Clone, safetensors inspection, paper read.
- Found: draft is Qwen-arch 1B + diffusion head (N denoising steps).
- Decision: simple head, continuing to Phase 2.
- Completed: <time>

## Phase 2 — weight converter
- Started: <time>
- ...
```

User should be able to read this doc top-to-bottom in 3 minutes and
understand the night.

## Handling specific risks

### Risk: Phase 1 says "6 weeks, complex noise schedule"

Do not stop. Instead:

1. Write the full finding in DFLASH_ARCHITECTURE.md
2. Identify the simplest subset that could work as a proof of concept
   (e.g., "1-step denoising ignoring the schedule, accept-rate will
   be low but loop will demonstrate")
3. Ship that MVP as 0.1.6-preview. Real dflash ships 0.1.7.
4. The user would much rather see a working loop with 1.2× speedup
   than a "we looked at it and gave up" report.

### Risk: kernel compilation breaks on some HIP feature

Try:
- Simpler kernel (e.g., use fp32 instead of fp16)
- Reference existing similar kernel in `kernels/src/*.hip`
- Ask `hipcc` with `--verbose` for what it's rejecting
- If stuck > 30 min, skip the optimization, ship a slower but correct
  path, note the TODO

### Risk: draft converter fails on safetensors

Try:
- Inspect the file with `safetensors-cli show` or Python
  `from safetensors import safe_open`
- Cross-reference against an existing `hipfire-quantize` loader for
  stock Qwen 3.5 — the draft is Qwen-arch so most tensor names match
- If dflash-specific tensors are the blocker, skip those and load the
  Qwen core. Diffusion head can be a separate PR.

### Risk: verification reveals a numerical mismatch

This is the one that needs care. Byte-exact greedy parity is the
invariant. If the target's argmax at position i differs from the
non-dflash path's argmax at position i for the same context, something
is corrupted. Options:

1. Likely cause: your new code accidentally touches `forward_prefill_batch`.
   Confirm the diff is strictly additive.
2. Likely cause: KV cache state is polluted by the draft between
   verify calls. Add explicit KV snapshot/restore around each verify.
3. Likely cause: the pre-compiled kernel blob for one of the existing
   kernels got invalidated. `rm -rf /tmp/hipfire_kernels` and rebuild.

## Exact execution order

After the user runs the overnight prompt, your first five actions are:

1. `git checkout dflash` — switch to the overnight branch (never master)
2. `git log --oneline -5` — confirm you're on commit `6383440` or later
3. `cat docs/DFLASH_PORT_PLAN.md` — the master plan
4. `cat .claude/projects/-home-kaden-ClaudeCode-autorocm-hipfire/memory/project_016_dflash_port.md` if accessible — the memory
5. Start Phase 1: clone z-lab/dflash into `.dflash-reference/`, open
   model.py, begin the architecture dig.

Then work the phases. Commit + push each to `origin/dflash`.

## Wake-up report

At the end of your session (morning or stop-condition hit), write
`docs/DFLASH_MORNING_REPORT.md` summarizing:

- Phases completed / deferred
- Key architectural finding from Phase 1
- Current state of the code (what works, what doesn't)
- Benchmarks if Phase 7-8 reached
- What the user should try first (`hipfire run qwen3.5:9b --dflash` ?)
- Follow-ups that are blocking vs nice-to-have

Commit this last. User reviews it first thing.

## On running shell commands

You have full shell access. Use it liberally for:
- Cloning repos (to `.dflash-reference/`)
- Downloading HF models via `hf download`
- Running bench + quality-gate after each phase
- Remote ssh to v620 and bc250 for cross-arch testing if something
  specific to those archs surfaces

Do NOT:
- Push to master — only to `dflash`. Never force-push either branch.
- Skip pre-commit hooks with `--no-verify` unless you've verified the
  hook is wrong
- Upload draft model weights to HF until Phase 7 gate passes (quality
  + speed gate must clear first)
- Blow away `~/.hipfire/models/` or the repo `models/` directory
- Merge dflash into master — that's the user's review step in the morning

## Monitor injections — check at every phase boundary

There is a live monitor watching your progress on `origin/dflash`. It
communicates via `docs/DFLASH_INJECTIONS.md`. Treat this file as a
real-time command channel from the user.

**Required checks:**

1. `git pull origin dflash` at the start of every phase.
2. `cat docs/DFLASH_INJECTIONS.md` — read the "Active injections" section.
3. If there's an entry dated after your last applied injection:
   - Read it in full.
   - Apply it according to its priority level:
     - `INFO` — add to your mental context, no action
     - `GUIDANCE` — suggested adjustment; apply if it helps your MVP
     - `OVERRIDE` — supersedes the autonomy contract for the stated
       scope (e.g., "quality-gate parity relaxed for Phase 3")
     - `ABORT` — stop current phase, follow the injection's next-step
   - Acknowledge by appending to `docs/DFLASH_PROGRESS.md`:
     `[injection applied <timestamp>] <one-line summary>`
4. Commit any file changes the injection produced. Push.

**Why the mailbox exists:** the monitor sees things you can't (rate of
progress, stall patterns, whether a quality-gate failure is a real
regression vs. an ambient cost of new code). Trust its overrides unless
they'd clearly break the spec-decoding correctness contract.

**If the mailbox file doesn't exist or is empty:** proceed with the
autonomy contract as originally written. No injections is the default.

## Emergency escalation — worker → monitor

If you hit a fundamental block and need human context before morning,
append a priority block to the END of `docs/DFLASH_INJECTIONS.md`
yourself, titled:

```
## WORKER → MONITOR <timestamp>
priority: HELP
context: <1-2 sentences>
blocked on: <specific technical question>
```

Commit + push. The monitor checks the file on its next 30-min cycle and
will either inject an OVERRIDE/GUIDANCE or (via codex-rescue fallback)
kick a debugging agent at the problem. While waiting, continue on
anything parallelizable — never sit idle.

## Final rule

If in doubt at a decision point, prefer **correct + committed + documented**
over **fast + incomplete + hidden**. The user trades speed for clarity
every time.
