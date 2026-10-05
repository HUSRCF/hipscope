# railgun for the NPU (XDNA2 / AIE2P, Strix Halo NPU5)

Retained, certified, low-overhead dispatch for our own amdxdna runtime (`railgun::npu`, DRM ioctls only), plus the
CPU-free GPU→NPU work ring that GPU+NPU co-op needs. Everything here is Rust; no driver or firmware change.

Measured results and logs: `report.md`, section "railgun for the NPU". This note is the design: the four
mechanisms, the patch inventory, the hash-keyed program cache / step tape, and the exact interface the GPU side
(hipfire railgun PM4) uses.

## 0. Driver facts this design relies on (linux 7.0 `drivers/accel/amdxdna`)
| fact | source |
|---|---|
| `ERT_CMD_CHAIN = 19`; payload `{u32 command_count, submit_index, error_index, reserved[3]; u64 data[count]}` (CMD BO handles) | `amdxdna_ctx.h:15-58` |
| chain = one 4 KiB firmware cmdbuf, one mailbox message, one response; 112 B per START_CU slot ⇒ ≤ 36 commands | `aie2_message.c:960-1030`, `aie2_msg_priv.h:364,384-396` |
| failure reports `error_index`; sub-command headers are not updated (only the chain BO); on error the driver memsets the (first) cmd payload to 0xff | `aie2_ctx.c:229-282`, `amdxdna_ctx.c:140-160` |
| `force_cmdlist = true`: every single START_CU already travels as a chain of 1 | `aie2_ctx.c:26` |
| the driver sets the cmd state to NEW itself at job run, so a retained cmd BO needs no header rewrite | `aie2_ctx.c:313` |
| EXEC_CMD takes exactly one cmd BO; arg BOs (every BO any sub-command touches) ≤ 4095, pinned per job | `amdxdna_ctx.c:592-617` |
| job timeout 60 s (`HWCTX_MAX_TIMEOUT`) | `aie2_ctx.c:30,538` |
| UMQ / doorbell fields exist in the uapi `CreateHwctx` but the aie2 driver never reads them: no user-mode queue | uapi `amdxdna_accel.h`; no `umq`/`doorbell` use under `drivers/accel/amdxdna` |

## 1. Prepared programs and in-place patching (`railgun::npu`)
`HwCtx` + `config_cu(PDI)` + the instruction (TXN) DEV BO are built once per design/shape. A `Prepared` is one
retained ERT_START_CU command BO (17 words) plus its retained EXEC_CMD handle array:

| word | content | hot path |
|---|---|---|
| 0 | header `NEW | 16<<12 | START_CU<<23 | ERT_CU<<28` | never (driver resets state) |
| 1 | cu_mask = 1 | never |
| 2, 3 | TXN opcode 3, 0 | never |
| 4, 5, 6 | insts device address lo/hi, insts bytes | never (a different program = a different `Prepared`, see §5) |
| 7+2i, 8+2i | arg i device address lo/hi (i < 5) | **patched** by `patch_arg(i, &Bo)` only when the BO changes; only that cache line is flushed |

`Chain` = one retained ERT_CMD_CHAIN BO over N ≤ 36 `Prepared` (payload never rewritten; the deduplicated arg
handle array is rebuilt only when a sub-command's handle changed). `HwCtx::submit_prepared` / `submit_chain`
issue one EXEC_CMD with no encoding; one syncobj wait per chain.

## 2. Lean TXN (`crates/pm-npu/src/kernels/gemm_array_lean.rs`, `ArrayDesign::lean_insts`)
The vendor-style dynamic TXN of the V8/V9/V10 pair designs re-initialises the whole array every submit (core
reset, 176 channel resets, all memtile descriptors, 512 core + 512 memtile lock writes, ring tasks; ≈ 51 KB,
1746 ops). Measured on silicon that costs ≈ 150 µs per GEMM of firmware time. The lean TXN keeps the array
configured across submits and repairs only what a completed submit leaves in a non-initial state, derived
analytically per shape from completed BD loads per channel (`J = kc·waves`, `W = waves`, `NW`, `MW` parities;
table in the module doc) and proven in the simulator by `state_snapshot()` equality with a full submit
(locks, every BD register including iteration `current`, channel engine positions, stream FIFOs).

Contract: valid only directly after a *completed* full or lean submit of the same PDI on the same hwctx (no other
hwctx may run on the partition in between). The first submit after `config_cu` is always the full TXN.

| design / shape | full TXN | lean TXN | lean ops per column |
|---|---:|---:|---|
| V9 gate_up 4096×1280×2560, down 4096×2560×640 | 51,056 B | 11,056 B | 4 cores × (halt/reset, release, PC, enable); shim BDs+patch+tasks; reset+requeue memtile S2MM4 / MM2S1; B_EMPTY lock; clear `current` of 8 B-replay BDs |
| V9 periodic (NW even) | 48,368 B | 7,792 B | cores + shim only |
| V10 down | 48,560 B | 7,984 B | cores + shim + requeue the finite B fill |
| V10 `--b-prefix` down / 160×2560×640 | 49,136 / 48,944 B | 8,816 / 15,824 B | cores + shim + requeue the finite fill / guarded-replay groups in order (+ BD 36 when MW > 1), clear their iteration `current` |

## 3. Persistent work ring P1 (`crates/pm-npu/src/kernels/ring.rs`)
One START_CU whose TXN runs S slot-runs back to back with no CPU in between. The NPU-side "controller" is the
firmware's TXN interpreter itself: the shim DMA cannot branch on data, so a self-looping shim BD mirrors the slot's
`seq` word from DDR into memtile memory and the TXN `MASKPOLL`s it.

### 3.1 Shared memory (frozen, version 1)
One buffer (CPU `shmem_bo` in the CPU-producer harnesses; in co-op, anonymous host pages pinned by amdxdna as a userptr BO
(`Device::userptr_bo`) and `hipHostRegister`ed for the GPU — NOT a HIP VMM dma-buf export, which stops aliasing after the
import: report.md amendment 2026-10-03), 64-byte lines, little endian. `nslots` is a power of two.

| line | offset | writer | content |
|---|---|---|---|
| header | 0 | host, once | `u32 magic 0x504E4752 ('RGNP')`, `u32 version 1`, `u32 nslots`, `u32 64`, `u32 sentinel 0xFFFFFFFF` (byte 16) |
| slot s | `64·(1+s)` | **producer (GPU)** | `u32 seq` (byte 0, written LAST), `u32 prog`, `u32 m`, `u32 flags`, `u64 a_off`, `u64 b_off`, `u64 c_off` (P1 reads only `seq`; the rest is informational) |
| done s | `64·(1+nslots+s)` | **NPU only** | `u32 done_seq` (byte 0), `u32 slot`, `u32 0x454E4F44 ('DONE')`, zeros — one 64-byte DMA write |

Run k (global, monotonic) uses slot `s = k mod nslots` and `seq = k + 1` (never 0, never the sentinel).
Within a retained persistent command of S runs started at `seq0`, run j has `seq = seq0 + j`.

### 3.2 NPU side, per run j (slot s)
1. POLL: reset+release memtile(0) S2MM4; `WRITE32 X := 0`; queue shim(0) MM2S0 BD P (1 word ← slot line s, `next = P`)
   and memtile(0) S2MM4 BD Q (1 word → X, `next = Q`); `MASKPOLL X == seq`.
2. DRAIN: one `WRITE32` flips P's `next` to the finite sentinel BD SENT; `MASKPOLL X == 0xFFFFFFFF` (in-order
   stream ⇒ every poll word has landed, nothing stray reaches the GEMM's B channel).
3. BODY: the V9 GEMM TXN with shim DDR patches advanced by the slot's fixed arena offsets; ends with both C SYNCs
   (all 16 C channels joined by the firmware, Astra gate 2). `persistent_v9_lean`: run 0 of a command takes the full
   body, runs 1.. the lean body (§2) plus a forced reset/requeue of memtile(0) S2MM4 and MM2S0, the two channels POLL
   and DONE borrow (136 KB vs 416 KB of TXN per 8 slots).
4. DONE: reset+release memtile(0) MM2S0 (strictly after the C SYNC); 16 `WRITE32` stage {seq, s, 'DONE'} at memtile
   Y; memtile MM2S0 BD DQ → shim S2MM0 BD D (token) writes the 64-byte done line; `SYNC(shim 0 S2MM0)`.

Spare resources (asserted against V9's own at the maximal topology): shim BDs P=4, SENT=5, D=6; memtile BDs Q=10,
DQ=11; memtile scratch X=0x1000, Y=0x3000 (never touched by any V9 DMA). Latch argument for the single-word flip:
`ring.rs` module doc.

Limits of P1 (stated, not hidden): one shape/PDI per persistent command; each slot's operand offsets are fixed when
the command is built (the GPU writes operands *into* the slot's arena, it does not pass arbitrary addresses);
multi-round retained resubmission needs `S mod nslots == 0`; a whole command is bounded by the 60 s job timeout.
Dynamic per-slot addresses would need the NPU to rewrite shim BD registers from DDR (control packets from a
controller core or from the shim DMA to its own TileCtrl port) — a P2 option, not built.

### 3.3 CPU role
The CPU never sits between a producer write and the NPU start, or between the NPU done and the consumer. It
re-arms the queue: per round it `patch_seq`s the retained insts BO (the declared words only) and submits the same
cmd BO; rounds can be queued ahead (the driver keeps them in order), so re-arming never synchronises with the GPU.

### 3.4 Grouped expert launch (`crates/pm-npu/src/kernels/experts.rs`, `grouped_v9`)
One START_CU (optionally one ring slot) runs E experts of one projection back to back: one PDI / ConfigKey for every
expert with M ≤ 512 (one 512-row wave), expert 0 the full body, experts 1..E−1 the lean body, each with its own
A rows / B (expert weights) / C arena patches. No per-expert full TXN, no per-expert submit. Experts are serialized
by completion (each body ends with its C SYNCs). Size: 147 KB (E=8), 485 KB (E=32), 935 KB (E=64), all run on silicon.
A design that wants to join (e.g. G80) exposes `append_run_body(txn, arena_base)`, `append_lean_run_body(txn,
arena_base, extra_requeue)` with its case in the lean derivation, `pack_in`/`unpack_out`/`reference`, and its wave M.

## 4. GPU-side interface (hipfire railgun PM4)
Addresses are GPU VAs of the shared host pages (`HipRuntime::register_host`): `ring` (ring base), `a_arena`, `b_arena`,
`c_arena`, and the per-slot offsets fixed when the persistent command was built (`slot_a_off[s]`, `slot_c_off[s]`).
Measured implementation: `npu_tools::coop_gpu` (PM-native kernels, a polling kernel instead of PM4 packets) and
`npu-coop`; report.md "coop-w2" (publish → NPU done → GPU observe p50 3.13 µs, exact GEMM slots).

Producer, run k (slot `s = k mod nslots`, `seq = k + 1`):
1. Slot free: if `k ≥ nslots`, `WAIT_REG_MEM` (memory space, function `==`, mask 0xFFFFFFFF, reference `seq − nslots`)
   on `ring + 64·(1+nslots+s)`. Equality, never ≥: wrap-safe and cannot pass on a skipped slot.
2. The producing kernel writes A (and any per-slot operand) to `a_arena + slot_a_off[s]`.
3. Make those writes visible in memory before the publish. Measured: on the userptr / `hipHostRegister` pages, stores
   issued `glc slc dlc` and completed (`s_waitcnt_vscnt 0`) are seen by the NPU (and the CPU); the A copy kernel's end
   of kernel precedes the publish kernel in the same stream.
4. Optional slot body: `WRITE_DATA` of words 1..15 of `ring + 64·(1+s)` (prog/m/flags/offsets; P1 ignores them).
5. Publish: `WRITE_DATA` (dst = memory, `WR_CONFIRM` = 1) of `seq` to `ring + 64·(1+s)`.

Consumer, run k: `WAIT_REG_MEM` (memory, `==`, mask 0xFFFFFFFF, reference `seq`) on `ring + 64·(1+nslots+s)` — measured
form: a poll loop of `glc dlc` loads — before reading C at `c_arena + slot_c_off[s]` (loads `glc dlc`).

Ordering guarantees from the NPU side: the done line is written only after the firmware's SYNC on every C channel's
task-complete token, so every C byte was handed to DDR before the done write is issued; the done line itself is a
single 64-byte DMA write.

Schedule (`npu-coop`, measured): producer and consumer are two in-order GPU queues, not one. The consumer, after
reading C of run k, stores `seq` to a GPU-only `consumed[s]` line; the producer's slot-free wait is
`consumed[s] == seq − nslots` (stricter than step 1: it also covers the C arena) and then runs up to `nslots` runs
ahead, so seq(k+1) is published while the NPU computes run k. On one queue the copy / publish of run k+1 waited for
the C read of run k and the NPU idled between runs (gemm 512×1280×2560: 271 → 230 µs per run).

Rounds are queued one ahead in `npu-coop`: round r+1's re-armed command (`Bodies::LeanFirst`, two retained command /
insts copies) and its GPU work are issued before the CPU waits for round r, so each command's ~45–50 µs start-up
(measured on the empty ring, independent of TXN and BO size) overlaps the previous round. With one shared B on G80
(`persistent_with(.., b_shared = true)`), lean runs keep the whole-K B resident in the memtile: no shim B stream, no
memtile fill, the B ready locks are written with the `kc` credits the fill would have released.

## 5. Hash-keyed program cache and step tape (`crates/railgun/src/npu/tape.rs`)
Mirrors the GPU railgun replay tape (`replay_sequence_hash`, FNV-1a 64).
- `ProgramKey` = FNV-1a64 over `'P' ‖ len ‖ PDI ‖ 'T' ‖ len ‖ TXN template ‖ 'D' ‖ count ‖ {word index u64 ‖ kind tag}`.
  The TXN template has every declared dynamic TXN word (ring `Seq`) zeroed, so current values never change the key,
  but moving one word between static and dynamic does. Kind tags: ArgAddress 0x01‖arg, InstsAddress 0x02,
  Seq 0x03‖run u32, Custom 0x04‖id u16.
- `ConfigKey` = FNV-1a64 over `'C' ‖ PDI`: the array-configuration identity.
- `ProgramCache`: key → retained insts BO; every hit byte-compares the stored canonical bytes, a mismatch is a
  "hash collision" error, never a reuse.
- Step `Tape`: ordered entries (program key, config key, binding kinds); `sequence_hash` covers keys and binding
  identities but not values, so per-step address/seq changes replay the same tape. `check_replay` detects a stale
  key (program bytes changed since recording).
- Lean TXN is keyed here: an entry whose ConfigKey equals the previous entry's (same hwctx, array still
  configured) takes the lean program; otherwise the full one. Gates: collision + stale-key negatives, replay
  byte-equal to a fresh encode for every entry (`verify_fresh`), every changed word declared (`verify_patch`),
  outputs exact vs eager.

## 6. Certification
| gate | how | negative control |
|---|---|---|
| exact vs CPU | first + last (and `--verify-every`: every) submit/slot vs the CPU reference | — |
| byte-exact replay | prepared/chain/lean/ring C buffers byte-equal to an eager full-TXN submit of the same operands | — |
| patch inventory | snapshot every retained BO, patch, diff: changed words ⊆ declared sites, result == fresh encode | stale patch (skip one arg patch): caught by the fresh-encode diff and by the output check |
| ring done protocol | per-slot done lines, equality waits | skipped done word (`Neg::SkipDone`): the missing done line is reported; stale seq patch: 0 of 16 words patched and no new done lines |
| ring overrun | producer guard `producer_may_publish` (done[s] must equal seq − nslots) | guard refuses; forced in the simulator ⇒ MASKPOLL deadlock reported |
| AIE2P control rules | `crates/pm-npu/src/isa/rules.rs` in `Program::finish` (unchanged core programs) | its own unit tests |
