# DFlash Worker Injections

Real-time messages from the monitor to the worker. Check this file at
the **start of every phase** and at **every commit boundary**. Apply
any instructions below that are dated AFTER your last consulted entry.

## How to read this file

Each entry has a timestamp and optional priority. When you see a new
entry since your last check:

1. Read the instruction in full.
2. Apply it if it overrides any prior contract rule.
3. Acknowledge the injection by appending a brief note to
   `docs/DFLASH_PROGRESS.md` like:
   `[injection applied 2026-04-13T04:17:00Z] favoring human-readability
   over byte-exact greedy parity for MVP per monitor instruction`.
4. Continue.

Injection priority levels:

- `INFO` — context, no action required
- `GUIDANCE` — suggested adjustment, apply if it helps
- `OVERRIDE` — supersedes the autonomy contract for the noted scope
- `ABORT` — stop current phase immediately, see message for next step

## Active injections

_(empty — monitor has not yet sent any instructions)_

## History (append-only, newest on top)

_(empty)_
