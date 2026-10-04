// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//! G80 whole-array deployment ([`Variant::G80`](super::gemm_array::Variant::G80), [`design_g80`]): the vertical-pair A-sharing
//! V8 structure on the 128x80 core of [`gemm_core_g80`]. Nothing here is a hardware claim: every statement below is
//! what the encoders in this file emit; behaviour on silicon is not established.
//!
//! # Problem shape and wave
//! `N = 1280` exactly, `K = 64 * kc` with `kc` in `1..=40` (`K <= 2560`), `M >= 1` padded with zero rows to `256 * mw`;
//! `waves = mw` (there is one N wave) and `waves <= 256`. Core `(col c, core row r)` of wave `w` computes the 128x80
//! output tile with M block `r / 2` (rows `w*256 + (r/2)*128 ..+128`) and N tile `2c + r % 2` (columns
//! `(2c + r%2)*80 ..+80`). The two cores of a vertical pair (core rows `2q`, `2q+1`) share one 128-row A block through
//! neighbour data memory (`PairRole::Lower` row `2q` receives the even-`mb` half, `Upper` row `2q+1` the odd half; the
//! peer lock protocol and the reverse int8 epilogue are [`gemm_core_g80`]'s). `Epilogue::Int8` only; `Control` may be
//! `Fast`, `Slow` or a `Probe` with `layout == 0` (`nocompute` and `repeat=N` are timing-only: the host packing, the
//! DMA protocol and every byte count below are unchanged, only the C content is not a GEMM).
//!
//! # Host arguments (byte counts, `waves = W`, `kc`)
//! * arg0 = A, **4 segments** of `W*kc*4096` B, segment `c = 0..4` is injected by column `c`. Inside a segment: wave,
//!   K chunk, one 4 KiB E/O half chunk (8 `mb_local` x 8 `kb` 8x8 blocks, [`Geometry::pack_a_half_chunk`]) of rows
//!   `w*256 + q*128 ..` with `q = c/2`, half `h = c%2`. A total is `4*W*kc*4096`. Columns 4..7 read no A.
//! * arg1 = B, **8 column segments** of `kc*10240` B (total `8*kc*10240`), chunk-major, per chunk `[even N tile 5120 B,
//!   odd N tile 5120 B]` = N tiles `2c`, `2c+1`; each 5120 B tile is `(kb 0..8, nb 0..10)` 8x8 blocks. B is read from DDR
//!   exactly once per submit, whatever `mw` is.
//! * arg2 = C, **8 column segments** of `W*4*10240` B (total `8*W*4*10240`): the S2MM0 stream (core rows 0, 1) followed
//!   by the S2MM1 stream (core rows 2, 3), each `(wave, rho % 2)` int8 tiles of 10240 B in `(mb 0..16, nb 0..10)` 8x8
//!   blocks ([`Geometry::pair_c_offset`]). For `K = 2560, M = 4096`: A 10 MiB, B 3.125 MiB, C 5 MiB.
//!
//! # Memtile map (own-view `MEM_BASE + off`, lock ids `MEM_LOCK + id`, per column)
//! | region | half-open offsets |
//! |---|---|
//! | **reserved for the lean ring scratch (X `0x1000`, Y `0x3000`) and spare BD queues, never accessed by G80** | `[0, 0x4000)` |
//! | C slot `s` row `r` (10240 B) | `0x4000 + s*0xa000 + r*0x2800`, both slots end `0x18000` |
//! | A ping / pong (4096 B live of an 8 KiB reservation each; columns 0..3 only) | `[0x18000,0x1a000)` / `[0x1a000,0x1c000)` |
//! | B parity 0 (`kc*5120` B) | `[0x1c000, 0x1c000 + kc*5120)` |
//! | B parity 1 (`kc*5120` B) | `[0x1c000 + kc*5120, 0x1c000 + 2*kc*5120)`, ends `<= 0x80000` (`= 0x80000` at `kc = 40`) |
//!
//! Parity `p` of B is the N-tile parity, i.e. the core-row parity that consumes it (rows 0, 2 take parity 0, rows 1, 3
//! parity 1). Every B chunk of the whole K stays resident for the whole submit; the B regions are single buffered.
//!
//! # Memtile BDs (26 BDs, ids < 48; even channel => id < 24, odd channel => id >= 24) and locks (22 active)
//! Channels: S2MM0..5 and MM2S0, 1, 2, 3, 5. BD ids 10 and 11 and shim BDs 4, 5, 6 are never touched (the lean ring's
//! spare Q / DQ / P / SENT / D descriptors).
//! | function | channel | BD ids | length | locks (memtile-local ids) |
//! |---|---|---|---|---|
//! | C producer row 0 / 1 / 2 / 3 | S2MM0 / 1 / 2 / 3 | `0<->1` / `24<->25` / `2<->3` / `26<->27` (slot 0 / 1, cyclic) | 2560 words | acquire `c_lock(slot,row)` (empty) -1, release `+1` (full); `c_lock = 8 + slot*8 + row*2` |
//! | B fill | S2MM4 | `4 -> 5 -> end` (parity 0 / 1), **finite, task `repeat = kc`** | 1280 words each | BD 4 releases ready0 (`4`) +1, BD 5 releases ready1 (`6`) +1, no acquire |
//! | A fill (cols 0..3) | S2MM5 | `28<->29` (ping / pong, cyclic) | 1024 words | acquire `A_EMPTY[s]` -1, release `A_FULL[s]` +1 |
//! | C drain rows 0, 1 | MM2S0 | `6->7->8->9->6`: slot 0 row 0, slot 0 row 1, slot 1 row 0, slot 1 row 1 (cyclic) | 2560 words | acquire `c_lock+1` (full) -1, release `c_lock` +1 |
//! | C drain rows 2, 3 | MM2S3 | `32->33->34->35->32` (same pattern, rows 2, 3) | 2560 words | as above |
//! | B parity 0 guard / whole | MM2S1 | `30` finite (`repeat = kc`), then `31` finite (`repeat = mw - 1`, omitted when `mw = 1`) | 1280 / `kc*1280` words | BD 30 acquires ready0 (`4`) -1; BD 31 no lock |
//! | B parity 1 guard / whole | MM2S5 | `42` finite (`repeat = kc`), then `43` finite (`repeat = mw - 1`, omitted when `mw = 1`) | 1280 / `kc*1280` words | BD 42 acquires ready1 (`6`) -1; BD 43 no lock |
//! | A replay (cols 0..3) | MM2S2 | `14<->15` (cyclic) | 1024 words | acquire `A_FULL[s]` -1, release `A_EMPTY[s]` +1 |
//!
//! Columns 4..7 have no A rows: 22 BDs and 18 locks there; columns 0..3 have the 26 BDs and 22 locks. Lock ids used:
//! `0..3` (A), `4`, `6` (B ready), `8..23` (C). Initial values: every `*_EMPTY` 1, every `*_FULL` 0, ready0 = ready1 = 0.
//! A ready lock never exceeds `kc <= 40 < 63`.
//!
//! ## B fill, guarded read, whole read (iteration contract)
//! Fill BD 4 / 5: base `0x1c000` / `0x1c000 + kc*5120`, 1280 words, iteration `step = 1280` words, `wrap = kc`,
//! `current = 0`; BD 4 links to BD 5, BD 5 has no `next`; the queued task repeats the chain `kc` times, so each BD is
//! loaded exactly `kc` times and its iteration `current` returns to 0 with the last load. Guard BD 30 / 42: same base,
//! length, iteration (`step = 1280, wrap = kc, current = 0`) and no `next`, task `repeat = kc`. Whole BD 31 / 43: same
//! base, `kc*1280` words, **default** iteration (`step 1, wrap 1`), no lock, no `next`, task `repeat = mw - 1`, queued
//! on the same channel strictly after the guard task (ordered queue, depth 2 of the memtile limit 4; `repeat <= 256`).
//! Reader chunk `j` therefore starts only after fill chunk `j` of its own parity landed
//! (`ready_p = fills_p - guarded reads_p`), and the unguarded whole pass starts only after the guard task consumed
//! all `kc` credits, so B is immutable afterwards and needs no lock. The two parity readers are independent queues.
//! All plain 1-D transfers, no dimension wrap (so a chunk of parity 1 can be read before all parity 0 chunks landed).
//!
//! # Completed-submit end state and lean repair recipe
//! Contract for a lean/persistent Railgun integration. "Start" is the canonical start state `S0` written by the full
//! dynamic TXN (`ArrayDesign::insts`): cores reset, every listed channel reset and queued at its first BD, **all G80
//! memtile BDs rewritten after the channel reset and before any queue write**, every lock at its initial value, shim
//! tokens armed. With `J = kc * waves` chunk transfers and `W = waves` tiles per column the end state of a submit whose C
//! token SYNCs (both S2MM channels, all 8 columns) returned is:
//! * **B fill S2MM4** (BD 4, 5, task `repeat = kc`): finite, idle, queue empty, **never parked**. `current` of BD 4 and BD 5
//!   returned to 0 (`kc` loads, `wrap = kc`). ready0 = ready1 = `kc - kc = 0`. There is no lookahead credit (finite tasks load
//!   no BD after the last repeat). Requeue is mandatory for every submit: `Task{S2mm, 4, BD 4, repeat = kc}`.
//! * **B readers MM2S1 / MM2S5**: guard task (`repeat = kc`) and whole task (`repeat = mw-1`) both finished, idle, queue
//!   empty; `current` of BD 30 / 42 is 0 again; BD 31 / 43 carry no iteration or lock. Requeue is mandatory for every submit,
//!   in this order on each channel: guard `repeat = kc`, then whole `repeat = mw - 1` only if `mw > 1`.
//! * **A fill S2MM5 {28,29} / A replay MM2S2 {14,15}** (cols 0..3): `J` loads each, parked at BD `28 + J%2` /
//!   `14 + J%2`; `A_FULL = 0`; the S2MM parked BD holds the lookahead acquire (`A_EMPTY[J%2] = 0`, the other slot's empty 1).
//!   Equivalent to S0 iff `J` is even. Minimal lean repair for odd `J` (slot 1): reset (CTRL bit1 pulse) both channels, restore only the
//!   dropped lookahead credit `A_EMPTY[1] = 1` (the canonical full reset writes every A lock to its initial value), requeue BD 28 / 14.
//! * **C producers S2MM0..3** (`{0,1}`, `{24,25}`, `{2,3}`, `{26,27}`) : `W` loads, parked at slot `W%2` with its empty-lock
//!   lookahead taken; **C drains MM2S0 / MM2S3**: `2W` loads, parked at BD `6 + 2*(W%2)` / `32 + 2*(W%2)`, full locks unacquired.
//!   Equivalent to S0 iff `W` is even. Minimal lean repair for odd `W` (parked at slot 1): reset all six channels (S2MM0..3, MM2S0,
//!   MM2S3), restore only the dropped slot-1 `EMPTY` lock of each row (`c_lock(1, r) = 1`), requeue BD `C_S2MM_BD[r][0]`, 6, 32.
//!   C rings use no iteration, so there is no `current` to repair.
//! * **Cores**: `DONE`; always halt+reset, `CORE_PC = 0`, enable last, as the full TXN. Canonical full reset reloads every core lock
//!   (`A/B_EMPTY = 1`, `C_EMPTY = 1`, rest 0); the minimal lean repair reloads only what the end state lost: the dropped pong
//!   `A_EMPTY[1]` / `B_EMPTY[1]` lookahead credit when `J` is odd (`C_EMPTY` stays 1, FULL and PEER locks are 0). Core S2MM0 (A)
//!   {0,1} / S2MM1 (B) {2,3} are parked at BD `J%2` (reset and requeue BD 0 / 2 when `J` is odd); core MM2S0 C (BD 4 self-cyclic)
//!   parks at BD 4 with `C_FULL` unacquired for any `W`.
//! * **Shim**: finite BDs/tasks consumed (shim channels are never reset); BDs are rewritten with new DDR patches each submit.
//! * **Streams**: no residue after the C tokens: the cores consume exactly `kc*W` A and B chunks, A/B shim BDs have drained,
//!   the C streams carry exactly `W*2*10240` B per channel; every switch lane is empty.
//!
//! A lean TXN is valid **only** directly after a *completed* submit (both aggregate C SYNCs returned). After an interrupted or
//! aborted run the state above does not hold and only the canonical full reset is valid: full channel resets, a rewrite of
//! every G80 memtile BD (clearing the `current` of BD 4, 5, 30, 42, which a channel reset does not clear), and initialisation of
//! every lock. After a completed submit a lean TXN, for every column, requeues the finite B fill / guard / whole tasks (above,
//! `repeat = kc` for fill and guard, `mw - 1` for whole) and applies only the parity repairs listed (`J` odd: A / core rings; `W`
//! odd: C rings / drains). The ready locks 4 and 6 end at 0 and the completed finite fill holds **no** lookahead credit, so no lock
//! write is needed for them: a forced reset of S2MM4 (POLL) or MM2S0 (DONE) by the lean ring does not touch locks 4 / 6 and needs no
//! lock repair, only the requeue of BD 4 with `repeat = kc` (retain `kc`) and of all B tasks, and BD 6 for MM2S0 (valid only while
//! the drain chain is parked at BD 6, i.e. `W` even; for odd `W` the parity plan above already resets and requeues it). Ring
//! scratch X / Y are never touched (`[0, 0x4000)`).
//! A submit must start only after the previous submit's two aggregate C SYNCs returned (quiescent array) and the next
//! expert's B is not overlapped with the previous submit: the B regions are single buffered per submit.
//!
//! # Shim (DDR patch points, built by `gemm_array`)
//! Per column: BD 0 B on MM2S0 (`kc*2560` words, arg1 + `col*kc*10240`), BD 1 C stream 0 on S2MM0 (`W*5120` words, arg2 +
//! `col*W*40960`, token), BD 2 A on MM2S1 (columns 0..3 only, `W*kc*1024` words, arg0 + `col*W*kc*4096`), BD 3 C stream 1 on
//! S2MM1 (`W*5120` words, arg2 + `col*W*40960 + W*2*10240`, token). Each BD's address word (`Bd::address(shim, id) + 4`)
//! is DDR-patched with `column_offset (+ arena_base[arg])` in the grouped launch.
//!
//! # Switch circuits (column `c`, lanes preserved, no master has two slaves)
//! * shim: MM2S0 -> North0 (B), MM2S1 -> North1 (A, columns 0..3), North0 -> S2MM0 and North1 -> S2MM1 (C).
//! * memtile: South0 -> Dma4 (B fill); South1 -> Dma5 and Dma2 -> North5 (A, columns 0..3); Dma1 -> North4 (parity 0 B);
//!   Dma5 -> North3 (parity 1 B); North(r) -> Dma(r) (C of core row r, r = 0..3); Dma0 -> South0 (rows 0, 1 drain); Dma3 -> South1
//!   (rows 2, 3 drain).
//! * core row `r`: C `Dma0 -> South(r)` and `North(s) -> South(s)` for `s > r`. B parity 0 on lane 4: `South4 -> Dma1` on rows 0, 2,
//!   `South4 -> North4` on rows 0, 1. B parity 1 on lane 3: `South3 -> Dma1` on rows 1, 3, `South3 -> North3` on rows 0, 1, 2.
//!   A of row `r` is injected by column `r` (columns 0..3) on North5, passed `South5 -> North5` by rows below `r`, selected at row `r`
//!   as `South5` (column `r`), `East0` (columns `< r`) or `West0` (columns `> r`), driving `Dma0`, `West0` (`0 < col <= r`) and
//!   `East0` (`col < 7`, `col >= r`), so every column of the row receives the same half.
//!
//! # Estimate
//! [`estimate`] is a model, never a measurement; see its documentation. It uses the explicit 128x80 geometry and the
//! G80 core's issue cycles, not any TN64 constant.
use super::gemm_array::{self as array, ArrayDesign, CycleEstimate, Geometry, Kind, Topo, Variant, MAX_WAVES};
use super::gemm_core::{self, Control, CoreVariant, Epilogue, PairRole};
use super::gemm_core_g80 as g80_core;
use super::gemm_i8::{peak_ops_per_second, ArgKind, ArgSpec};
use crate::{
    dma::{Bd, BdLocks, Direction::{Mm2s, S2mm}, Iteration, Location, Task},
    route::{Circuit, Port, ShimDma},
};

const TM: usize = g80_core::TM;
const TN: usize = g80_core::TN;
const CHUNK_K: usize = g80_core::CHUNK_K;
const A_HALF_BYTES: usize = g80_core::A_HALF_BYTES;
const B_BYTES: usize = g80_core::B_BYTES;
const COUT_BYTES: usize = g80_core::COUT_BYTES;
const MB: usize = TM / 8;
const NB: usize = TN / 8;
const KB: usize = CHUNK_K / 8;
const ROWS: usize = array::ROWS;
const COLS: usize = array::COLS;
/// Columns that inject A (core row `c`'s half is injected by column `c`).
const A_COLS: usize = array::G80_A_COLS;
/// Rows / columns of one wave: two 128-row M tiles x sixteen 80-column N tiles.
pub const WAVE_M: usize = array::G80_WAVE_M;
pub const WAVE_N: usize = array::G80_WAVE_N;
const _: () = assert!(WAVE_M == 256 && WAVE_N == 1280 && A_COLS == 4 && A_HALF_BYTES == 4096 && B_BYTES == 5120);
const _: () = assert!(COUT_BYTES == 10240 && MB == 16 && NB == 10 && KB == 8 && ROWS == 4 && COLS == 8);

// Memtile map (byte offsets, see the module header).
const C_OFF: u32 = 0x4000;
const C_SLOT_STRIDE: u32 = 0xa000;
const C_ROW_STRIDE: u32 = 0x2800;
const A_OFF: [u32; 2] = [0x18000, 0x1a000];
const B_OFF: u32 = 0x1c000;
const MEM_END: u32 = 0x80000;
const _: () = assert!(C_ROW_STRIDE as usize == COUT_BYTES && C_SLOT_STRIDE == 4 * C_ROW_STRIDE);
const _: () = assert!(C_OFF + 2 * C_SLOT_STRIDE == A_OFF[0] && A_OFF[0] + 0x2000 == A_OFF[1] && A_OFF[1] + 0x2000 == B_OFF);
const _: () = assert!(B_OFF as usize + 2 * g80_core::MAX_KC * B_BYTES == MEM_END as usize);
const _: () = assert!(A_HALF_BYTES <= 0x2000);

// BD ids and channels (banks: even channel < 24, odd >= 24).
const C_S2MM_BD: [[u32; 2]; ROWS] = [[0, 1], [24, 25], [2, 3], [26, 27]];
const C_DRAIN_CH: [u32; 2] = [0, 3];
const C_DRAIN_BD0: [u32; 2] = [6, 32];
const B_FILL_CH: u32 = 4;
const B_FILL_BD: [u32; 2] = [4, 5];
/// B readers per parity: MM2S1 (parity 0) and MM2S5 (parity 1).
pub(crate) const G80_B_MM2S_CH: [u32; 2] = [1, 5];
/// Finite guarded B readers (`repeat = kc`, iteration `step 1280, wrap kc`) and whole-stream readers (`repeat = mw - 1`).
pub(crate) const G80_B_GUARD_BD: [u32; 2] = [30, 42];
pub(crate) const G80_B_WHOLE_BD: [u32; 2] = [31, 43];
const B_READ_CH: [u32; 2] = G80_B_MM2S_CH;
const B_GUARD_BD: [u32; 2] = G80_B_GUARD_BD;
const B_WHOLE_BD: [u32; 2] = G80_B_WHOLE_BD;
/// B ready locks per parity (`gemm_core::B_EMPTY` ids 4 and 6; the `B_FULL` ids 5 and 7 are unused).
const B_READY: [u32; 2] = [gemm_core::B_EMPTY[0], gemm_core::B_EMPTY[1]];
const A_S2MM_CH: u32 = 5;
const A_S2MM_BD: [u32; 2] = [28, 29];
const A_MM2S_CH: u32 = 2;
const A_MM2S_BD: [u32; 2] = [14, 15];
const C_LOCK0: u32 = 8;
const _: () = assert!(B_READY[0] == 4 && B_READY[1] == 6 && g80_core::MAX_KC < 63);

fn c_lock(slot: usize, row: usize) -> u32 { C_LOCK0 + (slot * 8 + row * 2) as u32 }
fn c_off(slot: usize, row: usize) -> u32 { C_OFF + slot as u32 * C_SLOT_STRIDE + row as u32 * C_ROW_STRIDE }
fn b_off(kc: usize, parity: usize) -> u32 { B_OFF + (parity * kc * B_BYTES) as u32 }
fn a_locks() -> ([u32; 2], [u32; 2]) { (gemm_core::A_EMPTY, gemm_core::A_FULL) }

/// `(kc, mw)` of a G80 topology; panics for any other kind.
fn shape(topo: Topo) -> (usize, usize) {
    match topo.kind {
        Kind::G80 { kc, mw, .. } => (kc, mw),
        other => panic!("gemm_g80 needs a G80 topology, got {other:?}"),
    }
}

fn check_geometry(geo: &Geometry) -> (usize, usize) {
    let (kc, mw) = shape(geo.topo);
    assert!(geo.kc == kc && geo.mw == mw && geo.nw == 1 && geo.waves == mw && geo.n == WAVE_N && geo.k == kc * CHUNK_K,
        "inconsistent G80 geometry {geo:?}");
    (kc, mw)
}

/// One memtile BD in a cyclic chain: `idx` of `len`, ids from `first`, acquiring `acq` (-1) and releasing `rel` (+1).
fn chain_bd(first: u32, idx: usize, len: usize, addr: u32, words: u32, acq: u32, rel: u32) -> (u32, Bd) {
    let mut bd = Bd::new((array::MEM_BASE + addr) as u64, words);
    bd.locks = BdLocks { acq: Some((array::MEM_LOCK + acq, -1)), rel: Some((array::MEM_LOCK + rel, 1)) };
    bd.next = Some(first + ((idx + 1) % len) as u32);
    (first + idx as u32, bd)
}

/// Static stream-switch circuits of one column (see the module header, "Switch circuits").
pub(crate) fn column_circuits(topo: Topo, col: u32) -> Vec<Circuit> {
    shape(topo);
    let has_a = topo.has_a(col);
    let mut v = Vec::new();
    let (shim, mem) = (array::shim_loc(col), array::mem_loc(col));
    let sh = |direction, channel| ShimDma { tile: shim, direction, channel };
    v.push(Circuit { tile: shim, slave: sh(Mm2s, 0).port(), master: Port::North(0) });
    v.push(Circuit { tile: shim, slave: Port::North(0), master: sh(S2mm, 0).port() });
    v.push(Circuit { tile: shim, slave: Port::North(1), master: sh(S2mm, 1).port() });
    v.push(Circuit { tile: mem, slave: Port::South(0), master: Port::Dma(B_FILL_CH as u8) });
    v.push(Circuit { tile: mem, slave: Port::Dma(B_READ_CH[0] as u8), master: Port::North(4) });
    v.push(Circuit { tile: mem, slave: Port::Dma(B_READ_CH[1] as u8), master: Port::North(3) });
    v.push(Circuit { tile: mem, slave: Port::Dma(C_DRAIN_CH[0] as u8), master: Port::South(0) });
    v.push(Circuit { tile: mem, slave: Port::Dma(C_DRAIN_CH[1] as u8), master: Port::South(1) });
    for r in 0..ROWS as u8 { v.push(Circuit { tile: mem, slave: Port::North(r), master: Port::Dma(r) }); }
    if has_a {
        v.push(Circuit { tile: shim, slave: sh(Mm2s, 1).port(), master: Port::North(1) });
        v.push(Circuit { tile: mem, slave: Port::South(1), master: Port::Dma(A_S2MM_CH as u8) });
        v.push(Circuit { tile: mem, slave: Port::Dma(A_MM2S_CH as u8), master: Port::North(5) });
    }
    let c = col as usize;
    for r in 0..ROWS {
        let tile = array::core_loc(col, r);
        // C down: own tile, then the rows above forward on their own lanes.
        v.push(Circuit { tile, slave: Port::Dma(0), master: Port::South(r as u8) });
        for s in r + 1..ROWS { v.push(Circuit { tile, slave: Port::North(s as u8), master: Port::South(s as u8) }); }
        // B: parity 0 on lane 4 (rows 0, 2), parity 1 on lane 3 (rows 1, 3); lanes are passed up through the rows between.
        if r % 2 == 0 { v.push(Circuit { tile, slave: Port::South(4), master: Port::Dma(1) }); }
        if r < 2 { v.push(Circuit { tile, slave: Port::South(4), master: Port::North(4) }); }
        if r % 2 == 1 { v.push(Circuit { tile, slave: Port::South(3), master: Port::Dma(1) }); }
        if r < 3 { v.push(Circuit { tile, slave: Port::South(3), master: Port::North(3) }); }
        // A of row r is injected by column r and broadcast along the row on horizontal lane 0.
        let slave = match c.cmp(&r) {
            std::cmp::Ordering::Equal => Port::South(5),
            std::cmp::Ordering::Less => Port::East(0),
            std::cmp::Ordering::Greater => Port::West(0),
        };
        v.push(Circuit { tile, slave, master: Port::Dma(0) });
        if c > 0 && c <= r { v.push(Circuit { tile, slave, master: Port::West(0) }); }
        if c < COLS - 1 && c >= r { v.push(Circuit { tile, slave, master: Port::East(0) }); }
        if c < A_COLS && r < c { v.push(Circuit { tile, slave: Port::South(5), master: Port::North(5) }); }
    }
    v
}

/// All memtile BDs of a column `(id, bd)`: B fill / guarded / whole readers, C rings and drains, A rings (columns 0..3).
pub(crate) fn memtile_descriptors(topo: Topo, col: u32) -> Vec<(u32, Bd)> {
    let (kc, _) = shape(topo);
    let mut v = Vec::new();
    let chunk_words = (B_BYTES / 4) as u32;
    let iteration = Iteration { step: chunk_words, wrap: kc as u32, current: 0 };
    for p in 0..2 {
        let addr = (array::MEM_BASE + b_off(kc, p)) as u64;
        let mut fill = Bd::new(addr, chunk_words);
        fill.iteration = iteration;
        fill.locks = BdLocks { acq: None, rel: Some((array::MEM_LOCK + B_READY[p], 1)) };
        fill.next = (p == 0).then_some(B_FILL_BD[1]);
        v.push((B_FILL_BD[p], fill));
        let mut guard = Bd::new(addr, chunk_words);
        guard.iteration = iteration;
        guard.locks = BdLocks { acq: Some((array::MEM_LOCK + B_READY[p], -1)), rel: None };
        v.push((B_GUARD_BD[p], guard));
        v.push((B_WHOLE_BD[p], Bd::new(addr, kc as u32 * chunk_words)));
    }
    if topo.has_a(col) {
        let (empty, full) = a_locks();
        let words = (A_HALF_BYTES / 4) as u32;
        v.extend(array::ring(A_S2MM_BD, A_OFF, words, empty, full, true));
        v.extend(array::ring(A_MM2S_BD, A_OFF, words, empty, full, false));
    }
    let words = (COUT_BYTES / 4) as u32;
    for r in 0..ROWS {
        let empty = [c_lock(0, r), c_lock(1, r)];
        v.extend(array::ring(C_S2MM_BD[r], [c_off(0, r), c_off(1, r)], words, empty, [empty[0] + 1, empty[1] + 1], true));
    }
    for g in 0..2 {
        for slot in 0..2 { for rr in 0..2 {
            let row = 2 * g + rr;
            let lock = c_lock(slot, row);
            v.push(chain_bd(C_DRAIN_BD0[g], slot * 2 + rr, 4, c_off(slot, row), words, lock + 1, lock));
        } }
    }
    v
}

/// Initial memtile lock values: `*_EMPTY` 1, `*_FULL` 0, B ready 0.
pub(crate) fn memtile_locks(topo: Topo, col: u32) -> Vec<(u32, u32)> {
    shape(topo);
    let mut v = vec![(B_READY[0], 0), (B_READY[1], 0)];
    if topo.has_a(col) {
        let (empty, full) = a_locks();
        for i in 0..2 { v.extend([(empty[i], 1), (full[i], 0)]); }
    }
    for slot in 0..2 { for r in 0..ROWS {
        v.extend([(c_lock(slot, r), 1), (c_lock(slot, r) + 1, 0)]);
    } }
    v
}

/// Core ring tasks plus the memtile tasks of a column (the shim tasks depend on the arguments). Finite B tasks come in queue
/// order: per reader channel the guarded task, then the whole-stream task (omitted for `mw = 1`).
pub(crate) fn ring_tasks(topo: Topo, col: u32) -> Vec<(Location, Task)> {
    let (kc, mw) = shape(topo);
    let rep = |direction, channel, bd, repeat| Task { direction, channel, bd, repeat, issue_token: false };
    let task = |direction, channel, bd| rep(direction, channel, bd, 1);
    let mut v = Vec::new();
    for r in 0..ROWS {
        let tile = array::core_loc(col, r);
        v.push((tile, task(S2mm, gemm_core::A_CHANNEL, gemm_core::A_BD[0])));
        v.push((tile, task(S2mm, gemm_core::B_CHANNEL, gemm_core::B_BD[0])));
        v.push((tile, task(Mm2s, gemm_core::C_CHANNEL, gemm_core::C_BD)));
    }
    let mem = array::mem_loc(col);
    for r in 0..ROWS { v.push((mem, task(S2mm, r as u32, C_S2MM_BD[r][0]))); }
    v.push((mem, rep(S2mm, B_FILL_CH, B_FILL_BD[0], kc as u32)));
    if topo.has_a(col) { v.push((mem, task(S2mm, A_S2MM_CH, A_S2MM_BD[0]))); }
    for g in 0..2 { v.push((mem, task(Mm2s, C_DRAIN_CH[g], C_DRAIN_BD0[g]))); }
    for p in 0..2 {
        v.push((mem, rep(Mm2s, B_READ_CH[p], B_GUARD_BD[p], kc as u32)));
        if mw > 1 { v.push((mem, rep(Mm2s, B_READ_CH[p], B_WHOLE_BD[p], (mw - 1) as u32))); }
    }
    if topo.has_a(col) { v.push((mem, task(Mm2s, A_MM2S_CH, A_MM2S_BD[0]))); }
    v
}

/// A argument: 4 injecting-column segments, `(wave, chunk, E/O half 4 KiB)`.
pub(crate) fn pack_a(geo: &Geometry, a: &[i8]) -> Vec<u8> {
    let (kc, _) = check_geometry(geo);
    assert_eq!(a.len(), geo.m * geo.k, "A must be m*k");
    let segment = geo.waves * kc * A_HALF_BYTES;
    let mut out = vec![0u8; A_COLS * segment];
    assert_eq!(out.len(), geo.a_bytes(), "A stream size");
    for c in 0..A_COLS {
        let (q, h) = (c / 2, c % 2);
        for w in 0..geo.waves { for ch in 0..kc {
            let at = c * segment + (w * kc + ch) * A_HALF_BYTES;
            geo.pack_a_half_chunk(a, w * WAVE_M + q * TM, h, ch, &mut out[at..at + A_HALF_BYTES]);
        } }
    }
    out
}

/// B argument: 8 column segments, chunk-major `[even N tile, odd N tile]`, each tile `(kb 0..8, nb 0..10)` 8x8 blocks.
pub(crate) fn pack_b(geo: &Geometry, b: &[i8]) -> Vec<u8> {
    let (kc, _) = check_geometry(geo);
    assert_eq!(b.len(), geo.k * geo.n, "B must be k*n");
    let mut out = vec![0u8; COLS * kc * 2 * B_BYTES];
    assert_eq!(out.len(), geo.b_bytes(), "B stream size");
    for col in 0..COLS { for ch in 0..kc { for par in 0..2 {
        let n0 = (2 * col + par) * TN;
        let base = (col * kc + ch) * 2 * B_BYTES + par * B_BYTES;
        for kb in 0..KB { for nb in 0..NB { for kk in 0..8 {
            let c0 = n0 + nb * 8;
            if c0 >= geo.n { continue; }
            let width = (geo.n - c0).min(8);
            let row = ch * CHUNK_K + kb * 8 + kk;
            let dst = base + (kb * NB + nb) * 64 + kk * 8;
            let s = row * geo.n + c0;
            for (x, y) in out[dst..dst + width].iter_mut().zip(&b[s..s + width]) { *x = *y as u8; }
        } } }
    } } }
    out
}

/// Row-major `m*n` C from the packed C stream (int8 tiles sign extended, M padding discarded).
pub(crate) fn unpack_c(geo: &Geometry, output: &[u8]) -> Vec<i32> {
    check_geometry(geo);
    assert_eq!(output.len(), geo.c_bytes(), "C stream size");
    let mut c = vec![0i32; geo.m * geo.n];
    for col in 0..COLS { for w in 0..geo.waves { for rho in 0..ROWS {
        let m0 = w * WAVE_M + (rho / 2) * TM;
        let n0 = (2 * col + rho % 2) * TN;
        let base = geo.pair_c_offset(col, w, rho);
        if m0 >= geo.m || n0 >= geo.n { continue; }
        for mb in 0..MB { for nb in 0..NB { for i in 0..8 {
            let row = m0 + mb * 8 + i;
            if row >= geo.m { break; }
            for j in 0..8 {
                let cc = n0 + nb * 8 + j;
                if cc >= geo.n { break; }
                c[row * geo.n + cc] = output[base + (mb * NB + nb) * 64 + i * 8 + j] as i8 as i32;
            }
        } } }
    } } }
    c
}

/// Issue-cycle growth of a K64 chunk against the 128x64 pair core: `1280 - 1024` VMACs per chunk (the wavefront is
/// `1315` against `1059` bundles), control unchanged (`docs/g80-feasibility.md` Gate 5c: later chunk ping / pong 1387 / 1379).
const CHUNK_GROWTH: u64 = (TM * TN * CHUNK_K / 512 - gemm_core::TM * gemm_core::TN * gemm_core::CHUNK_K / 512) as u64;
/// Issue cycles of the G80 reverse int8 epilogue `[Fast, Slow]`: 663 emitted for `Fast` (Gate 5c); `Slow` adds the baseline's
/// `550 - 530` serial-control delta ([INFERENCE]: not separately executed).
const EPILOGUE_CYCLES: [u64; 2] = [663, 663 + gemm_core::PAIR_EPILOGUE_CYCLES[1] - gemm_core::PAIR_EPILOGUE_CYCLES[0]];
/// One full wavefront of the compute region, the cost of each extra `Probe::repeat` pass.
const WAVEFRONT_BUNDLES: u64 = (32 * 5 * 8 + 35) as u64;

/// ESTIMATE only (never a measurement). Per K64 chunk a core needs `max(issue, 1280)`: the G80 issue cycles
/// ([`gemm_core::PAIR_CHUNK_CYCLES`] + [`CHUNK_GROWTH`], ping and pong averaged, per control discipline) against the
/// 1280-cycle B stream of the core (5120 B at 4 B/cycle; the 1024-cycle A half is hidden). Per wave the core adds the
/// reverse-epilogue issue cycles and the exposed `Cout` drain (`10240 / 4 = 2560` cycles: `Cout` shares the top of the i32 `C`,
/// so chunk 0 of the next wave waits for the previous wave's drain). The memtile -> shim C transfer (2 channels x 4 B/cycle,
/// two 10240 B tiles each = 5120 cycles) overlaps the next wave through the two memtile slots, so the wave period is
/// `max(core wave cycles, shim C cycles)` and the last shim transfer is exposed; start-up is one B chunk (1280 cycles).
/// `Probe::no_compute` removes the compute and the epilogue body, `Probe::repeat = N` adds `N - 1` wavefronts to every
/// later chunk. Ignores lock waits beyond this, bank conflicts, DDR contention and start-up beyond the first chunk.
/// `output_dma_cycles` reports the per-wave epilogue plus exposed `Cout` drain cycles summed over the waves.
pub(crate) fn estimate(geo: &Geometry, ctl: Control, clock_hz: u64) -> CycleEstimate {
    assert!(clock_hz > 0);
    let (kc, _) = check_geometry(geo);
    let (kc, waves) = (kc as u64, geo.waves as u64);
    let probe = ctl.probe();
    let slow = usize::from(probe.serial);
    let input_chunk = (B_BYTES / 4) as u64;
    let chunk = |cycles: u64| {
        if probe.no_compute { input_chunk } else {
            (cycles + CHUNK_GROWTH + (u64::from(probe.repeat.max(1)) - 1) * WAVEFRONT_BUNDLES).max(input_chunk)
        }
    };
    let [ping, pong] = gemm_core::PAIR_CHUNK_CYCLES[slow].map(chunk);
    let epilogue = if probe.no_compute { 0 } else { EPILOGUE_CYCLES[slow] };
    let cout_drain = (COUT_BYTES / 4) as u64;
    let shim_c = (2 * COUT_BYTES / 4) as u64;
    let wave_core = (kc * (ping + pong)).div_ceil(2) + epilogue + cout_drain;
    let estimated_cycles = input_chunk + (waves - 1) * wave_core.max(shim_c) + wave_core + shim_c;
    let estimated_seconds = estimated_cycles as f64 / clock_hz as f64;
    let useful_ops = 2.0 * geo.m as f64 * geo.n as f64 * geo.k as f64;
    CycleEstimate {
        ideal_mac_cycles: (TM * TN * CHUNK_K / 512) as u64 * kc * waves,
        input_dma_cycles: input_chunk * kc * waves,
        output_dma_cycles: (epilogue + cout_drain) * waves,
        estimated_cycles,
        estimated_seconds,
        useful_tops: useful_ops / estimated_seconds / 1e12,
        peak_ops_per_second: peak_ops_per_second(clock_hz),
    }
}

/// Build the G80 whole-array design (module header): `n = 1280`, `k` a multiple of 64 in `64..=2560`, `m >= 1` padded to 256
/// rows per wave with at most 256 waves, `Epilogue::Int8 { shift <= 31 }` and a `Control` with `Probe::layout == 0`.
/// Panics with an explanatory message on any other shape before emitting a byte.
pub fn design_g80(m: usize, n: usize, k: usize, epi: Epilogue, ctl: Control) -> ArrayDesign {
    match epi {
        Epilogue::Int8 { shift } => assert!(shift <= 31, "G80 SRS shift {shift} exceeds 31"),
        Epilogue::I32 => panic!("G80 supports only Epilogue::Int8: the int8 Cout buffer shares the top of the i32 C accumulator"),
    }
    assert_eq!(ctl.probe().layout, 0, "G80 has one fixed data-memory map; Probe::layout={} is not defined", ctl.probe().layout);
    assert!(m > 0, "G80 needs M >= 1");
    assert_eq!(n, WAVE_N, "G80 needs N = {WAVE_N} exactly, got {n}");
    assert!(k % CHUNK_K == 0 && (CHUNK_K..=g80_core::MAX_KC * CHUNK_K).contains(&k),
        "G80 K must be a multiple of {CHUNK_K} in {CHUNK_K}..={}, got {k}", g80_core::MAX_KC * CHUNK_K);
    let waves = m.div_ceil(WAVE_M);
    assert!(waves <= MAX_WAVES, "G80 M={m} needs {waves} waves, exceeding {MAX_WAVES}");
    let geo = Geometry::g80(m, n, k, epi, ctl);
    let topo = geo.topo;
    let program = |role| g80_core::program_pair(geo.kc, geo.waves, role, epi, ctl).finish();
    let (lower, upper) = (program(PairRole::Lower), program(PairRole::Upper));
    for p in [&lower, &upper] { assert!(p.len() <= 16 * 1024, "G80 core program {} B exceeds 16 KiB", p.len()); }
    let insts = array::build_txn_dynamic(&geo, false);
    ArrayDesign {
        pdi: crate::pdi::build(&array::build_cdo(topo, [&lower, &upper], false).to_words()),
        insts: insts.to_bytes(),
        args: vec![
            ArgSpec { bytes: geo.a_bytes(), kind: ArgKind::In },
            ArgSpec { bytes: geo.b_bytes(), kind: ArgKind::In },
            ArgSpec { bytes: geo.c_bytes(), kind: ArgKind::Out },
        ],
        geo,
        variant: Variant::G80,
        program: lower,
        upper_program: upper,
        core: if ctl.is_slow() { CoreVariant::FastSlowCtl } else { CoreVariant::Fast },
        v8: Some((epi, ctl)),
        shim_axi: crate::dma::ShimAxi::default(),
    }
}
