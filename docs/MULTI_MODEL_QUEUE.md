# Multi-Model Daemon with VRAM Request Queue

Design spec for extending the hipfire daemon from one-model-at-a-time
to N-loaded-models with a request queue for VRAM collisions. The
motivation, shipped pieces, and remaining work are all captured here so
the next session can pick up without re-discovery.

## Status

| Piece | Status | Commit |
|---|---|---|
| Daemon flock mutex (`~/.hipfire/daemon.pid`) | **shipped** | `058cdb3` |
| Multi-model daemon state | pending | — |
| VRAM projection | pending | — |
| Request queue + scheduler | pending | — |
| CLI serve multi-model routing | pending | — |
| Tests | pending | — |

## Context

Before this work the daemon held exactly one loaded model in an
`Option<LoadedModel>`. Every `{"type":"load"}` unconditionally unloaded
the previous one. Two user-visible consequences motivated the redesign:

1. **Serve reloads trash wall-clock time.** Idle-evicting or swapping
   between two models on a serve means a 2-5 s model-load cost on every
   cross-model request, even when both would fit in VRAM simultaneously.
2. **No protection against orphan daemons.** A crashed bun CLI could
   leave a daemon running, a second bun spawn would fire up another,
   and two daemons would double-allocate VRAM. Fixed in `058cdb3` by
   an `flock(2)` mutex on `~/.hipfire/daemon.pid` that fails the second
   invocation with `FATAL: hipfire daemon already running (PID N)`.
   Runs BEFORE `Gpu::init()` so the rejected process never touches the
   GPU.

With the mutex landed, the remaining work is making the one allowed
daemon hold N models at once, safely, within a projected VRAM budget.

## Design: request queue for collision handling

The first branch we explored was B+A: try the load, evict idle models
(last_used > `idle_timeout`) LRU-first until it fits, hard-reject with
a candidate list if still over budget. The user argued for a queue
instead, and on reflection it subsumes B+A as limit cases:

- **Fits immediately** → load, respond `{"type":"loaded"}` (same as B)
- **Fits after idle eviction** → sweep idle, load, respond `{"type":"loaded","evicted":[...]}` (same as B)
- **Still doesn't fit** → enqueue, respond `{"type":"load_queued"}`. Client
  keeps reading the stream. Each time the daemon finishes a generate /
  processes an unload / runs the idle sweep, it re-checks the queue head.
  If it now fits, load and respond `{"type":"loaded"}` to the waiting
  client unsolicited on the same stdout stream. On timeout, respond
  `{"type":"error"}` (same as A, but bounded rather than immediate).

Net upshot: transient contention resolves itself, no surprise
evictions of active models, no manual user intervention for the common
"just finished with model A, now loading B" case.

## Protocol changes

```
→ {"type":"load","model":"path.hfq","params":{"max_seq":4096,"wait_timeout_sec":60}}

  Daemon immediately responds with ONE of:
← {"type":"loaded","model_id":"path.hfq","arch":"qwen3_5","dim":4096,"layers":32,"vocab":248320,"vl":false,"bytes":5400000000,"evicted":["other.hfq"],"queued_ms":0}
← {"type":"load_queued","model_id":"path.hfq","position":2,"projected_bytes":5400000000,"free_bytes":2100000000,"reason":"need 5.4 GB, 2.1 GB free; waiting for VRAM"}

  If queued, the daemon will LATER send unsolicited on the same stdout:
← {"type":"loaded","model_id":"path.hfq",...,"queued_ms":23400}
  OR
← {"type":"error","model_id":"path.hfq","message":"load timed out after 60s in queue; 5.4 GB needed, max free during wait was 3.1 GB"}

→ {"type":"generate","id":"r1","model_id":"path.hfq","prompt":"...","max_tokens":16,...}
← {"type":"token","id":"r1","text":"..."}...
← {"type":"done","id":"r1","model_id":"path.hfq","tokens":42,"tok_s":44.5}
     (model_id echoed so the client can mux responses across models)

→ {"type":"unload","model_id":"path.hfq"}
← {"type":"unloaded","model_id":"path.hfq"}

→ {"type":"unload"}      (no model_id = unload all, legacy compat)
← {"type":"unloaded","count":N}

→ {"type":"reset","model_id":"path.hfq"}
← {"type":"reset","model_id":"path.hfq","seq_pos":0}

→ {"type":"list_models"}
← {"type":"models","loaded":[{"model_id":"...","arch":"...","bytes":N,"max_seq":K,"last_used_ms":M,"idle_ms":T},...],"pending":[{"model_id":"...","position":N,"queued_ms":T}]}
```

### Backward-compatibility rules

- Requests **without** `model_id` on `generate`/`reset` target the MOST
  RECENTLY LOADED model (emulates single-model semantics). Responses
  always include the resolved `model_id`.
- `unload` without `model_id` unloads ALL models (matches today's "drop
  the one" semantic, extended to the multi-model case).
- Old clients that don't know about `load_queued` will see the response
  type, fail to parse, and the daemon's later unsolicited `loaded` will
  arrive on a now-unused stream. CLI must be updated in lockstep — no
  mixed-version operation.

## VRAM projection

Computed from the HFQ header alone (no allocation required):

```
projected = weights_bytes              // from header: sum of tensor byte sizes
          + kv_cache_bytes             // max_seq * n_layers * n_kv_heads * bytes_per_head_slot
          + scratch_bytes              // ~ 2 * dim * 4 (f32 activations, 2 buffers)
          + dn_state_bytes             // DeltaNet recurrent state for Qwen3.5 (non-zero for layer_type==LinearAttention)
          + safety_buffer_bytes        // default 512 MiB, env override HIPFIRE_VRAM_BUFFER_MB
```

`bytes_per_head_slot` is the KV-mode-dependent constant already used
to allocate the KV cache in `llama::KvCache::new` / equivalent:

| KV mode | bytes/head/token |
|---|---|
| `q8` | `head_dim + 4` (i8 quantized + f32 scale) |
| `asym3` | K=rotated-3b packed + V=Q8 scale → `ceil(head_dim * 3 / 8) + 4 + head_dim + 4` |
| `asym4` | `ceil(head_dim * 4 / 8) + 4 + head_dim + 4` |
| `asym2` | `ceil(head_dim * 2 / 8) + 4 + head_dim + 4` |

Check against `gpu.hip.get_vram_info().0` (free bytes) minus the safety
buffer. If `projected > available`, eviction/queueing kicks in.

### Measured vs projected

After load completes, the daemon captures the ACTUAL allocated bytes
by diffing `get_vram_info().0` before vs after. Stores on
`LoadedModel.allocated_bytes`. Subsequent eviction decisions use
ACTUAL bytes for already-loaded models and PROJECTED bytes for the
queue head. This handles the case where projection under-counts (e.g.,
fragmentation overhead, kernel JIT allocations) — we won't repeatedly
misproject.

## Scheduler tick

The main daemon loop becomes an event loop:

```rust
let (stdin_tx, stdin_rx) = mpsc::channel();
thread::spawn(move || {
    for line in stdin.lock().lines() { stdin_tx.send(line).unwrap(); }
});

loop {
    match stdin_rx.recv_timeout(Duration::from_millis(500)) {
        Ok(Ok(line)) => handle_message(line),
        Ok(Err(_)) => break,                    // stdin closed
        Err(RecvTimeoutError::Timeout) => {}    // fall through to tick
        Err(RecvTimeoutError::Disconnected) => break,
    }
    scheduler_tick();
}
```

A tick:

1. **Idle sweep.** For each loaded model, if `now - last_used > idle_timeout`
   AND no pending queue entries, OR the queue head would fit if this
   model were evicted → unload. Logs `{"type":"evicted","model_id":"...","reason":"idle_300s"}` on stdout for observability.
2. **Timeout sweep.** For each `pending_loads` entry, if
   `now - enqueued_at > timeout` → drop, emit
   `{"type":"error","model_id":"...","message":"..."}`.
3. **Advance queue head.** If `pending_loads` non-empty, compute
   projection for head. If fits (after applying idle evictions from
   step 1), load it, emit `{"type":"loaded","model_id":"...","queued_ms":N}`.

Ticks fire from TWO sources:
- **Timer** (every 500 ms via the `recv_timeout` expiry above)
- **Completion** (at the end of every `handle_message` for load /
  unload / generate / reset)

### FIFO discipline

Pending queue is strict FIFO. If the head won't fit, the scheduler
does NOT skip to other entries — this prevents small-model starvation
(a 1 GB load always jumping ahead of a 9 GB load that keeps waiting
for just-enough eviction). Trade-off: a giant load behind several
small ones blocks them all. Rationale: predictable ordering > optimal
packing for the single-user case. Revisit if it proves annoying.

## Daemon state shape

```rust
struct LoadedModel {
    model_id: String,               // absolute model path, used as lookup key
    arch_id: u32,
    last_used: Instant,             // bumped on every generate completion
    allocated_bytes: u64,           // actual VRAM delta at load time
    max_seq: usize,
    // ... existing per-model fields (weights, kv_cache, dn_state, scratch, tokenizer, conversation_tokens, seq_pos) ...
}

struct PendingLoad {
    model_id: String,
    params: LoadParams,             // max_seq, etc
    projected_bytes: u64,
    enqueued_at: Instant,
    timeout: Duration,              // from wait_timeout_sec, default 60s
}

struct DaemonState {
    models: HashMap<String, LoadedModel>,
    pending_loads: VecDeque<PendingLoad>,
    safety_buffer_bytes: u64,       // 512 MiB default
    idle_timeout: Duration,         // from HIPFIRE_IDLE_TIMEOUT env or config (currently CLI-side only)
}
```

`Gpu` lives alongside in main; no locking needed because the daemon
runs single-threaded (the stdin reader is a pure producer that only
forwards lines into the channel, doesn't touch GPU state).

## Client (CLI) implications

`cli/index.ts` changes:

- Serve's `current: string | null` + `currentMaxSeq: number | null`
  → `loaded: Map<string, { maxSeq: number, lastUsed: number }>`.
- Per-model `lastRequestTime` tracked; the existing 60s-interval
  eviction loop iterates the map.
- `acquireLock` becomes per-model (a per-`model_id` busy flag) so a
  `generate` on model A and a pending load on model B can coexist
  without the load waiting for an in-flight generate on A. Actually,
  simpler: keep the single `busy` lock but the daemon itself handles
  multi-request queueing — CLI serializes at the HTTP layer, daemon at
  the model layer.

  On second thought: keep the single CLI-side lock; the daemon is
  still single-threaded for generate anyway. The queue only interacts
  with LOAD, not generate. So CLI-side serialization is fine.

- `Engine.generate()` needs to handle the new `load_queued` response
  type: when seen, continue reading from daemon stdout until a
  matching `{"type":"loaded","model_id":"..."}` or
  `{"type":"error","model_id":"..."}` arrives. Add a timeout guard
  against a daemon that promises queuing and never fulfills.

- `runViaHttp` / direct `run` paths surface `load_queued` to stderr
  (`[hipfire] loading qwen3.5:9b (waiting for VRAM, queued at position 2)...`)
  and then proceed once `loaded` arrives.

## Failure modes and edge cases

- **Daemon crash with pending loads.** `load_queued` responses were
  sent but no follow-up `loaded` / `error` is coming. Client hangs
  until stream closes (EOF). CLI should interpret EOF on a pending
  load as a daemon crash — log it, exit nonzero.
- **Timeout exactly at load moment.** Scheduler tick order: check
  timeouts first, THEN advance head. A request that just timed out
  doesn't get a spurious `loaded`.
- **Client disconnects before queued load completes.** Daemon doesn't
  know the client is gone (stdio is one-directional from daemon's
  perspective). It'll still run the load when space frees, waste the
  allocation, then discover the client is gone on next write error.
  Accept this — it's rare, and the alternative (heartbeat / cancel
  messages) is complexity bloat.
- **Duplicate `load` for the same `model_id`.** If already loaded,
  return `loaded` immediately with `queued_ms: 0`. If already pending,
  deduplicate — the client gets the response for the earlier enqueue.
- **`generate` on a model that's still in the pending queue.** Return
  `{"type":"error","message":"model X is queued for loading; wait for loaded response"}`.
  Or auto-block until it loads. Simpler: error. Client retries after
  seeing `loaded`.

## Implementation roadmap (file by file)

### Daemon (Rust)

- [ ] `crates/engine/examples/daemon.rs`: refactor `model: Option<LoadedModel>`
      → `models: HashMap<String, LoadedModel>`. Plumb `model_id` through every
      existing handler. Estimate ~150 LOC touched.
- [ ] Add `DaemonState` + pending queue + scheduler tick. ~100 LOC.
- [ ] Add stdin-reader thread + event loop with `recv_timeout`. ~30 LOC.
- [ ] VRAM projection function. Use the HFQ header the loader already
      parses to estimate `weights_bytes`. KV-mode constants pulled from
      the same env/config the loader uses. ~60 LOC.
- [ ] Measure `allocated_bytes` post-load via `get_vram_info` diff.
      Store on `LoadedModel`. ~10 LOC.

### CLI (TypeScript)

- [ ] `cli/index.ts`: extend `serve()` to track a `Map<string, LoadedEntry>`.
      Per-model idle eviction. ~80 LOC.
- [ ] `Engine.generate()` / relevant streams: handle `load_queued` by
      continuing to read until matching `loaded`/`error`. ~40 LOC.
- [ ] `runViaHttp` / `run`: surface queued status to stderr. ~20 LOC.
- [ ] Optional: `hipfire ps` enhancement to show loaded + pending models
      (reads `list_models` from daemon).

### Tests

- [ ] `tests/daemon_queue.sh`:
      - Load two small models, verify both are listed and generate.
      - Force a collision: load model C that won't fit; verify
        `load_queued` response and that `loaded` arrives after an
        idle-eviction or explicit unload.
      - Timeout case: load with `wait_timeout_sec: 2` when no eviction
        is possible; verify `error` after 2 s.
- [ ] `tests/vram_projection.sh` (unit-ish): spawn daemon, load models
      of known sizes, compare projected vs measured allocated_bytes;
      alert if projection off by > 15%.
- [ ] Keep existing `tests/daemon_mutex.sh` (P1 coverage).

## Open questions

1. **Scratch-buffer reuse.** Each loaded model currently has its own
   `ForwardScratch` / `Qwen35Scratch` (activation buffers sized for
   max_seq). Can we share ONE scratch buffer across loaded models,
   sized to the max of all loaded max_seqs? Would save meaningful VRAM
   at the cost of some per-request setup. Unclear if the saving is
   worth the refactor — the scratch is usually <5% of model weight
   size. **Punt for now; measure post-MVP.**

2. **Concurrent generates on different models.** The daemon
   serializes all generates through one main thread. If you had two
   HTTP clients hitting different loaded models on the serve, they'd
   still queue behind each other at the daemon. Per-model GPU streams
   in HIP would unblock this but is a much bigger refactor. **Out of
   scope for this design; tracked as future work.**

3. **`wait_timeout_sec` semantics for explicit unload.** If a user
   runs `hipfire unload X` while a queued load Y is waiting for X's
   VRAM, should Y's queued_ms reset to 0 (now that it's loading)? Or
   should its timeout countdown continue from enqueue? **Let timeout
   continue from enqueue** — Y has been waiting, don't punish the
   explicit unload.

4. **Per-model config overrides at load time.** Already supported via
   `params.max_seq`; extending to `params.kv_mode` etc. is a separate
   concern. **Keep out of scope.**

## Glossary

- **Projected bytes**: pre-allocation estimate from HFQ header.
- **Allocated bytes**: measured VRAM delta post-load, stored per model.
- **Queued ms**: wall-clock milliseconds between enqueue and load.
- **Safety buffer**: 512 MiB headroom reserved outside projection to
  absorb fragmentation and kernel JIT allocations.
- **Tick**: one pass of the scheduler through timeout sweep + idle
  sweep + queue-head advancement.
