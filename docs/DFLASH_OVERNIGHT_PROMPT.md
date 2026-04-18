# DFlash Overnight Prompt

Copy-paste the prompt below into a fresh Claude Code session after
compaction. Do not modify it — the autonomy contract references exact
file paths.

---

## THE PROMPT

```
You are picking up the hipfire 0.1.6 "dflash" build. This is an
autonomous overnight session — I will not be available to answer
questions. Work until you either complete Phase 8 or hit the explicit
stop conditions documented in docs/DFLASH_OVERNIGHT_AUTONOMY.md.

IMPORTANT: all work happens on the `dflash` branch. Never touch
master. Master is the safe rollback point. First action below handles
this.

Your first six actions are prescribed:

1. `git checkout dflash` — switch to the overnight branch. `git branch
   --show-current` must print `dflash` before you proceed. If the
   branch doesn't exist locally, `git fetch origin && git checkout -b
   dflash origin/dflash`.

2. Read docs/DFLASH_OVERNIGHT_AUTONOMY.md in full — this is the
   contract. Non-negotiable rules include: never ask clarifying
   questions, commit + push after every phase (to origin/dflash,
   NEVER master), never break existing code (all dflash work is
   additive), never skip the quality gate, never merge dflash into
   master (that's the user's review step).

3. Read docs/DFLASH_PORT_PLAN.md — the 8-phase master plan.

4. Check your git state with `git log --oneline -5`. You should be on
   commit 6383440 or later. If there's uncommitted work from a prior
   session (`git status --short`), inspect it first — it may be work
   from an earlier overnight run that you should continue rather than
   restart.

5. Create docs/DFLASH_PROGRESS.md if it does not exist. Log every
   phase start, finding, decision, and completion there. This is how I
   review the night's work in the morning.

6. Start Phase 1 (architecture scope check). Clone z-lab/dflash into
   .dflash-reference/ (add to .gitignore before cloning), read
   dflash/model.py and dflash/generate.py, download
   z-lab/Qwen3.5-9B-DFlash config + one safetensors shard via
   `hf download`, read the relevant sections of arXiv:2602.06036,
   then write docs/DFLASH_ARCHITECTURE.md with: pseudocode of draft
   forward, pseudocode of speculative loop, component inventory
   (what reuses hipfire's existing code vs what's new), scope
   estimate, and go/no-go recommendation.

After Phase 1, do not pause for approval. Use your judgment. If Phase 1
reveals the full port is more than a 6-hour session, pivot to the
simplest working MVP (even a 1-step denoising approximation with low
accept-rate is valuable — a working loop beats a "we looked at it"
report).

Commit + push after every phase, even partial phases, to the `dflash`
branch (`git push origin dflash` — never `master`). Name commits
`feat(dflash): Phase N — <one-line>`. If a commit fails the quality
gate or speed gate, investigate — do NOT bypass with --no-verify
unless you're certain it's a false positive (and explain in the
commit body).

Target hardware for testing: the local 7900 XTX (gfx1100, 24 GB). If
a dflash-specific bug surfaces on other archs, ssh bc250 or ssh v620
and reproduce — do not defer cross-arch verification.

Update tasks #135-142 as you complete each phase with TaskUpdate. Add
follow-ups via TaskCreate when you discover work that should ship in
0.1.7 rather than 0.1.6.

At the end of your session, write docs/DFLASH_MORNING_REPORT.md
summarizing phases completed, architectural findings, what works,
what doesn't, benchmarks if reached, and what I should try first
when I wake up. Commit this last to `origin/dflash`.

Stop conditions are exhaustively listed in
docs/DFLASH_OVERNIGHT_AUTONOMY.md §"Stop conditions". None of them
are "agent feels unsure" or "agent wants to confirm". Make the call,
document it, ship.

Begin.
```

---

## What to do if the agent stops early

If you wake up and the agent has stopped before Phase 8 (or MVP
floor + 2h), check in this order:

1. `docs/DFLASH_MORNING_REPORT.md` — should exist. If yes, agent's
   story of the night lives there.
2. `docs/DFLASH_PROGRESS.md` — running log. Shows exactly where it
   stopped.
3. `docs/DFLASH_BLOCKED.md` — only present if agent hit a stop
   condition. Describes the block.
4. `git log --oneline -20` — what commits landed.
5. Task list — which of #135-#142 are marked complete.

If none of those files exist and git shows no recent commits, the
agent either crashed or ignored the autonomy contract. Kick off a
fresh session with the same prompt.

## What to do if the agent breaks existing code

Master is isolated — the agent only pushes to the `dflash` branch. If
dflash is broken, master is untouched. To reject the overnight work:

```bash
git checkout master                      # known-good
git branch -D dflash                      # drop local dflash
git push origin --delete dflash           # drop remote dflash
git push origin dflash                    # re-create from master if you want a fresh try
```

Or to cherry-pick the good parts from dflash into master:

```bash
git checkout master
git log --oneline master..dflash           # review what shipped
git cherry-pick <good-commit-sha>          # pick per-commit
```

Master baseline commits to know:
- `6383440` — pre-implementation state (docs + plan committed, zero
  implementation-side code changed).
- `ccb1064` — 0.1.5 "redline" tag baseline (documented, benchmarked,
  released).

Either is a safe reset target if you ever need it.

## What to send me in the morning

If you're happy with what shipped overnight, merge `dflash` → `master`
yourself, then tag + release:

```bash
git checkout master
git merge --ff-only dflash            # fast-forward if the branch linear
# OR
git merge --no-ff dflash -m "merge: dflash into master for 0.1.6"
git push origin master
git tag -a v0.1.6 -m "hipfire 0.1.6 dflash"
git push origin v0.1.6
gh release create v0.1.6 --title "..." --notes-file <notes>
```

If something looks off, inspect via `git log --oneline master..dflash`
and `git diff master..dflash` before merging. The `dflash` branch
stays around until you decide.

Never merge the overnight work into `master` if the quality gate fails
greedy parity. That's the one hill to die on.
