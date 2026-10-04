// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// AIE2P instruction fields and scheduling are derived from Xilinx/llvm-aie
// AIE2PGenInstrInfo.td and AIE2PGenSchedule.td (Apache-2.0 WITH LLVM-exception).
//! G80 pair core: the V8 pair wavefront of [`gemm_core`] widened from a 128x64 to a 128x80 output tile (five N
//! pairs per M pair, 40 groups of 32 cycles, 1,315-cycle wavefront), Int8 epilogue only.
//!
//! Everything that does not depend on the tile width is reused from `gemm_core`: the [`Assembler`], the
//! 9-step `PEER_FULL`/`PEER_EMPTY` chunk lock protocol ([`pair_acquire`]/[`pair_release`]), the
//! `Probe::repeat` pointer reload ([`pair_pointers`]), instruction encoders, lock ids, DMA channel/BD ids and
//! the `Control`/`Probe` knobs. This file owns the N=80 wavefront, its loop plan, the reverse int8 epilogue and
//! the relocated data-memory map.
//!
//! ## Data-memory map (own-tile offsets, a single fixed map; `Probe::layout` must be 0)
//! `C` i32 accumulator `0x0000..0xa000` (128 x 80 x 4 = 40,960 B); `Cout` int8 `0x7800..0xa000` (10,240 B, inside
//! the top of `C`); A half slots `0xa000`/`0xb000` (4 KiB); B slots `0xc000`/`0xd400` (5 KiB).
//!
//! ## Wavefront changes against the 128x64 core
//! * B column stride is `128 * NP = 640` B (`NP = 5`), a C row is `512 * NP = 2560` B. The B pointer steps need the
//!   eight modifiers [`MODIFIERS`] (`-512` resets A to repeat it for the next N pair, `3072` plus a `-512`
//!   immediate is the C row skip after five contiguous blocks, `640` is the B stride).
//! * The 1,315-bundle timeline is emitted as a 37-bundle peel, seven trips of a 160-bundle `jnz` body, and a
//!   158-bundle tail (the 128x64 scheme's 170/160x6/185 split overflows the 16 KiB program memory). The last
//!   group's row-1 C-skip pointer op falls past the end of the timeline and is the only dropped op; every other
//!   pointer update is needed for the periodicity of the repeated body. Program size is 12,848 B.
//! * The int8 epilogue converts the 160 blocks in descending order (`VLDA bm` quarters 3..0 with a `-64 B`
//!   post-modify, `VST.SRS.4x` with `-64 B` steps) so `Cout`, which overlaps the upper part of `C`, only
//!   overwrites i32 blocks that were already loaded; the closest (store, load) pair is ten cycles apart.
//! * `C_EMPTY` is acquired before chunk 0 of every wave (also with `Probe::no_compute`): `Cout` lives inside
//!   `C`, so the previous wave's `Cout` DMA drain must be finished before the first chunk writes `C`. `C_FULL` is
//!   released after the last `VST.SRS` has committed.
//!
//! Lane order of the dm accumulator vs the four bm quarters is the same simulator assumption as in `gemm_core`.
use super::gemm_core::{self, descriptor, instruction, pair_acquire, pair_pointers, pair_release, Assembler, Control,
    Cycle, Epilogue, PairLayout, PairRole, A_LOAD_TIMES, EPI_BODY, EPI_DRAIN_PAD, EPI_LOOP_DEC, EPI_LOOP_JNZ,
    EPI_STORE_DELAY, MAC_TIMES};
use super::gemm_core::{emit_cycle, load_a, matrix, pointer, pointer_immediate, put, store};
use super::gemm_i8::SIGNED_8X8;
use crate::{dma::{self, BdLocks}, isa::{self, bundle::encode_slot, gen::{Encoding, Slot}, Program}};
use std::sync::LazyLock;

pub const TM: usize = 128;
pub const TN: usize = 80;
pub const CHUNK_K: usize = 64;
/// One E/O half of a 128-row A block per K64 chunk: 8 mb x 8 kb 8x8 int8 blocks.
pub const A_HALF_BYTES: usize = TM * CHUNK_K / 2;
pub const B_BYTES: usize = CHUNK_K * TN;
pub const C_BYTES: usize = TM * TN * 4;
pub const COUT_BYTES: usize = TM * TN;
/// Largest K64 chunk count per output tile (`K = 64 * kc`).
pub const MAX_KC: usize = 40;
pub const A_ADDR: [u32; 2] = [0xa000, 0xb000];
pub const B_ADDR: [u32; 2] = [0xc000, 0xd400];
pub const C_ADDR: u32 = 0x0000;
pub const COUT_ADDR: u32 = 0x7800;
const LAYOUT: PairLayout = PairLayout { a: A_ADDR, b: B_ADDR, c: C_ADDR, cout: COUT_ADDR };
const _: () = assert!(COUT_ADDR as usize + COUT_BYTES == C_ADDR as usize + C_BYTES && A_ADDR[0] as usize == C_BYTES);

/// N pairs per M pair (each pair is 16 columns).
const NP: i32 = 5;
const GROUPS: i32 = 8 * NP;
const B_STRIDE: i32 = 128 * NP;
/// One C row of blocks (`mb` pair step) in bytes.
const ROW_BYTES: u32 = 512 * NP as u32;
/// Pointer modifier table (`m0..m7`): sorted B steps `{-3072, -2560, -2432, 640, 3072, 3200, 3712}` plus the A reset
/// (`-512`), the C row skip (`512 * NP + 512`) and the B stride (`640`), which are already in the set or merge into it.
const MODIFIERS: [i32; 8] = [-3072, -2560, -2432, -512, 640, 3072, 3200, 3712];
const A_RESET: u8 = 3;
const C_SKIP: u8 = 5;
const _: () = assert!(MODIFIERS[A_RESET as usize] == -512 && MODIFIERS[C_SKIP as usize] == 512 * NP + 512 && MODIFIERS[4] == B_STRIDE);

/// Compressed-timeline plan, identical for the overwrite and accumulate wavefronts: `PEEL` straight bundles,
/// `TRIPS` trips of a `PERIOD`-bundle `jnz` body (its counter decrement rides in the free ALU slot of body bundle
/// `DEC`, the branch in the LNG slot of bundle `PERIOD - 1 - JUMP_DELAY_SLOTS`), then the tail.
const PEEL: usize = 37;
const PERIOD: usize = 32 * NP as usize;
const TRIPS: usize = 7;
const DEC: usize = 146;
const WAVE_BUNDLES: usize = GROUPS as usize * 32 + 35;
const _: () = assert!(PEEL + PERIOD * TRIPS < WAVE_BUNDLES);

/// Like `gemm_core`'s `free_store`, but a store-slot op that falls past the end of the timeline is dropped and
/// reported as `false`.
fn free_store(cycles: &mut [Cycle], mut time: i32, instructions: &[(Slot, u64)]) -> bool {
    let mut all = true;
    for &instruction in instructions {
        while ((time + 10) as usize) < cycles.len() && cycles[(time + 10) as usize][Slot::St as usize].is_some() { time += 1; }
        if ((time + 10) as usize) >= cycles.len() { all = false; time += 1; continue; }
        put(cycles, time, instruction); time += 1;
    }
    all
}

/// The 128x80 pair wavefront (timeline offset 10: bundle index = cycle + 10).
fn wavefront(overwrite: bool) -> Vec<Cycle> {
    let mut cycles = vec![[None; 8]; WAVE_BUNDLES];
    let mut b_loads = [Vec::new(), Vec::new()];
    for group in 0..GROUPS { for dm in 0..4u8 {
        let start = group * 32 + dm as i32 * 8;
        for k in 0..8u8 {
            let time = start + MAC_TIMES[k as usize];
            put(&mut cycles, time, matrix(dm, k, overwrite && k == 0));
            b_loads[(dm % 2) as usize].push((time - 7, k as i32 * B_STRIDE + group % NP * 128 + (dm % 2) as i32 * 64));
            if dm % 2 == 0 {
                let row = dm / 2;
                put(&mut cycles, start + A_LOAD_TIMES[row as usize][k as usize], load_a(row * 4 + k % 4, row));
            }
        }
        let last = start + MAC_TIMES[7];
        for quarter in 0..4u8 {
            put(&mut cycles, last + 6 + quarter as i32, store(dm, quarter));
            if !overwrite {
                put(&mut cycles, start - 9 + quarter as i32,
                    instruction!("VLDA_dmx_lda_bm_pstm_nrm_imm", &[("dst", (dm * 4 + quarter) as u64),
                        ("ptr", (6 + dm / 2) as u64), ("imm", 1)]));
            }
        }
    } }
    for column in 0..2 {
        b_loads[column].sort_unstable();
        for (index, &(time, offset)) in b_loads[column].iter().enumerate() {
            let step = b_loads[column].get(index + 1).map_or(B_STRIDE, |next| next.1 - offset);
            let modifier = MODIFIERS.iter().position(|&m| m == step).expect("G80 B wavefront modifier");
            put(&mut cycles, time, instruction!("VLDB_dmx_ldb_x_pstm_nrm",
                &[("dst", 11), ("ptr", (2 + column) as u64), ("mod", modifier as u64)]));
        }
    }
    // A cursors repeat the five N pairs before advancing two M blocks (the pair E/O walk already sits on the next
    // mb_local after the fifth). C cursors independently skip the intervening row after five contiguous C blocks.
    for group in 0..GROUPS {
        for row in 0..2u8 {
            let start = group * 32 + row as i32 * 16;
            let last_load = start + A_LOAD_TIMES[row as usize][7];
            if group % NP != NP - 1 {
                assert!(free_store(&mut cycles, last_load + 1, &[pointer(Slot::St, row, A_RESET)]));
            }
        }
        if group % NP == NP - 1 {
            for row in 0..2u8 {
                let store_end = group * 32 + row as i32 * 16 + 40;
                let ok = free_store(&mut cycles, store_end + 1,
                    &[pointer(Slot::St, 4 + row, C_SKIP), pointer_immediate(4 + row, -512)]);
                assert!(ok || group == GROUPS - 1, "dropped C-skip op for non-final group {group}");
                if !overwrite {
                    let load_end = group * 32 + row as i32 * 16 + 2;
                    assert!(free_store(&mut cycles, load_end + 1,
                        &[pointer(Slot::St, 6 + row, C_SKIP), pointer_immediate(6 + row, -512)]));
                }
            }
        }
    }
    cycles
}

/// One K64 chunk compute: pointer setup, then peel / `jnz` body / tail. Always the VLIW wavefront, whatever the
/// control discipline (serial gaps are suspended).
fn chunk(a: &mut Assembler, overwrite: bool) {
    static WAVES: LazyLock<[Vec<Cycle>; 2]> = LazyLock::new(|| [wavefront(false), wavefront(true)]);
    let cycles = &WAVES[overwrite as usize];
    let serial = a.serial; a.serial = false;
    a.mov(1, 4, gemm_core::CORE_BASE + C_ADDR); a.mov(1, 5, gemm_core::CORE_BASE + C_ADDR + ROW_BYTES);
    if !overwrite {
        a.mov(1, 6, gemm_core::CORE_BASE + C_ADDR); a.mov(1, 7, gemm_core::CORE_BASE + C_ADDR + ROW_BYTES);
    }
    a.mov(0, 12, TRIPS as u32);
    while a.program.pc() % 16 != 0 { a.nop(1); }
    for cycle in &cycles[..PEEL] { emit_cycle(a, cycle); }
    let begin = a.program.pc();
    let jnz = PERIOD - 1 - isa::sched::JUMP_DELAY_SLOTS;
    let mut body: Vec<Cycle> = cycles[PEEL..PEEL + PERIOD].to_vec();
    assert!(body[DEC][Slot::Alu as usize].is_none(), "loop counter needs the ALU slot");
    body[DEC][Slot::Alu as usize] = Some(instruction!("ADD_add_r_ri", &[("d0", 12), ("s0", 12), ("imm", 0x7f)]).1);
    assert!(DEC + 8 <= jnz, "loop counter decrement must retire before the branch");
    assert!(body[jnz][Slot::Alu as usize].is_none() && body[jnz][Slot::Lng as usize].is_none() && body[jnz][Slot::Mv as usize].is_none(),
        "loop branch needs the LNG slot");
    body[jnz][Slot::Lng as usize] = Some(instruction!("JNZ", &[("i", begin as u64), ("s0", 12)]).1);
    for repeat in 1..TRIPS {
        assert_eq!(&cycles[PEEL..PEEL + PERIOD], &cycles[PEEL + repeat * PERIOD..PEEL + (repeat + 1) * PERIOD]);
    }
    for cycle in &body { emit_cycle(a, cycle); }
    for cycle in &cycles[PEEL + PERIOD * TRIPS..] { emit_cycle(a, cycle); }
    a.serial = serial;
}

const EPI_BLOCKS: usize = TM / 8 * TN / 8;
/// `Imm` field of a `-64 B` post-modify (4-bit two's complement of `-1` in 64 B units).
const EPI_BACK: u64 = 15;
/// Straight-line reverse epilogue timeline: processing index `j` handles block `159 - j`; it loads the four 16 x i32
/// quarters `3, 2, 1, 0` into `bm[4(j%4)..]` of `dm(j%4)` on cycles `4j..4j+3` and stores them as 64 int8 lanes with
/// one `VST.SRS.4x` `EPI_STORE_DELAY` cycles after the last quarter (pointers walk backwards, `-64 B` per op).
fn epilogue_wavefront() -> Vec<Cycle> {
    let mut cycles = vec![[None; 8]; (EPI_BLOCKS - 1) * 4 + 3 + EPI_STORE_DELAY + 1];
    for block in 0..EPI_BLOCKS {
        let dm = (block % 4) as u8;
        for i in 0..4u8 {
            let quarter = 3 - i;
            let (slot, bits) = instruction!("VLDA_dmx_lda_bm_pstm_nrm_imm",
                &[("dst", (dm * 4 + quarter) as u64), ("ptr", 6), ("imm", EPI_BACK)]);
            assert!(cycles[block * 4 + i as usize][slot as usize].replace(bits).is_none());
        }
        let (slot, bits) = instruction!("VST_SRS_4x_dmx_sts_srs_dm_pstm_nrm_imm_srsSign1",
            &[("src", dm as u64), ("su", 0), ("ptr", 4), ("imm", EPI_BACK)]);
        assert!(cycles[block * 4 + 3 + EPI_STORE_DELAY][slot as usize].replace(bits).is_none());
    }
    cycles
}
/// Int8 epilogue: peeled first 16 cycles, a `jnz` loop of 39 identical 16-cycle bodies (`add r12,-1` / `jnz r12`
/// in free slots, delay slots are body bundles), then the store drain and `EPI_DRAIN_PAD` NOPs so `rel C_FULL`
/// follows the last `VST.SRS` commit. `p6` starts at the top quarter of the last C block, `p4` at the last Cout block.
fn epilogue(a: &mut Assembler) {
    static CYCLES: LazyLock<Vec<Cycle>> = LazyLock::new(epilogue_wavefront);
    let cycles = &*CYCLES;
    let trips = EPI_BLOCKS * 4 / EPI_BODY - 1;
    for repeat in 2..=trips { assert_eq!(&cycles[EPI_BODY..2 * EPI_BODY], &cycles[repeat * EPI_BODY..(repeat + 1) * EPI_BODY]); }
    let last = EPI_BLOCKS as u32 - 1;
    a.mov(1, 6, gemm_core::CORE_BASE + C_ADDR + 256 * last + 192);
    a.mov(1, 4, gemm_core::CORE_BASE + COUT_ADDR + 64 * last);
    a.mov(0, 12, trips as u32);
    let serial = a.serial; a.serial = false;
    while a.program.pc() % 16 != 0 { a.nop(1); }
    for cycle in &cycles[..EPI_BODY] { emit_cycle(a, cycle); }
    let begin = a.program.pc();
    for (i, cycle) in cycles[EPI_BODY..2 * EPI_BODY].iter().enumerate() {
        let mut cycle = *cycle;
        if i == EPI_LOOP_DEC {
            assert!(cycle[Slot::Alu as usize].is_none());
            cycle[Slot::Alu as usize] = Some(instruction!("ADD_add_r_ri", &[("d0", 12), ("s0", 12), ("imm", 0x7f)]).1);
        }
        if i == EPI_LOOP_JNZ {
            assert!(cycle[Slot::Alu as usize].is_none() && cycle[Slot::Lng as usize].is_none() && cycle[Slot::Mv as usize].is_none());
            cycle[Slot::Lng as usize] = Some(instruction!("JNZ", &[("i", begin as u64), ("s0", 12)]).1);
        }
        emit_cycle(a, &cycle);
    }
    for cycle in &cycles[(trips + 1) * EPI_BODY..] { emit_cycle(a, cycle); }
    a.nop(EPI_DRAIN_PAD);
    a.serial = serial;
}

/// Validate the epilogue/control knobs shared by [`program_pair`] and [`tile_bds_pair`]; returns the SRS shift.
fn check(epi: Epilogue, ctl: Control) -> u8 {
    assert_eq!(ctl.probe().layout, 0,
        "G80 128x80 has one fixed data-memory map (C 0x0000, Cout 0x7800, A 0xa000/0xb000, B 0xc000/0xd400); layout={} is not defined",
        ctl.probe().layout);
    match epi {
        Epilogue::Int8 { shift } => { assert!(shift <= 31, "SRS shift {shift}"); shift }
        Epilogue::I32 => panic!("G80 128x80 supports only Epilogue::Int8: the int8 Cout buffer (0x7800, {COUT_BYTES} B) \
            shares the top of the {C_BYTES} B i32 C accumulator, and no separate i32 drain path exists"),
    }
}

/// G80 pair core (128x80 tile, Int8 epilogue). Same chunk lock protocol, `Control` disciplines and `Probe` knobs
/// (`serial`, `nocompute`, `repeat=N`) as [`gemm_core::program_pair`]; `kc` in `1..=MAX_KC`, `tiles` in `1..=256`,
/// `Int8 { shift <= 31 }` with `clamp(c >> shift, -128, 127)` semantics, `Probe::layout == 0`.
/// `C_EMPTY` is acquired before chunk 0 of every wave (also with `no_compute`, which skips only the chunk compute
/// and the epilogue body); `C_FULL` is released after the epilogue stores retire.
pub fn program_pair(kc: usize, tiles: usize, role: PairRole, epi: Epilogue, ctl: Control) -> Program {
    assert!((1..=MAX_KC).contains(&kc) && (1..=256).contains(&tiles), "G80 kc {kc} (1..={MAX_KC}) tiles {tiles} (1..=256)");
    let shift = check(epi, ctl);
    let probe = ctl.probe();
    let mut a = Assembler::default();
    a.serial = probe.serial;
    for (r, v) in [(0, -1i32 as u32), (1, 1), (2, SIGNED_8X8), (3, tiles as u32), (4, 0)] { a.mov(0, r, v); }
    for (m, v) in MODIFIERS.into_iter().enumerate() { a.mov(2, m, v as u32); }
    a.set_cr("crSat", 1); a.set_cr("crRnd", 0); a.set_cr("crSRSMode", 0);
    a.mov_named("s0", u32::from(shift));
    let tiles_loop = a.label(); let accumulated = a.label(); let drain = a.label();
    a.bind(tiles_loop);
    a.lock(true, gemm_core::C_EMPTY);
    a.mov(0, 5, kc as u32);
    pair_acquire(&mut a, role, LAYOUT);
    if !probe.no_compute { chunk(&mut a, true); }
    pair_release(&mut a, role);
    a.branch(drain, Some((5, false))); a.delay();
    a.bind(accumulated);
    pair_acquire(&mut a, role, LAYOUT);
    if probe.repeat > 1 {
        a.mov(0, 11, u32::from(probe.repeat));
        let again = a.label(); a.bind(again);
        pair_pointers(&mut a, role, LAYOUT);
        chunk(&mut a, false);
        a.add(11, -1); if !a.serial { a.nop(7); }
        a.branch(again, Some((11, true))); a.delay();
    } else if !probe.no_compute { chunk(&mut a, false); }
    pair_release(&mut a, role);
    a.branch(accumulated, Some((5, true))); a.delay();
    a.bind(drain);
    if !probe.no_compute { epilogue(&mut a); }
    a.lock(false, gemm_core::C_FULL);
    a.add(3, -1); a.branch(tiles_loop, Some((3, true))); a.delay();
    a.emit(&[instruction!("DONE", &[])]);
    a.finish()
}

/// DMA-view descriptors of the G80 pair core: A halves (4 KiB) on S2MM0, B (5 KiB) on S2MM1, and the int8 `Cout`
/// (10 KiB) on MM2S0, with the BD ids, channels and lock numbering of `gemm_core`.
pub fn tile_bds_pair(epi: Epilogue, ctl: Control) -> [[u32; 6]; 5] {
    check(epi, ctl);
    let mut bds = [[0; 6]; 5];
    for slot in 0..2 {
        bds[gemm_core::A_BD[slot] as usize] = dma::tile_bd(A_ADDR[slot], (A_HALF_BYTES / 4) as u32,
            BdLocks {acq: Some((gemm_core::A_EMPTY[slot], -1)), rel: Some((gemm_core::A_FULL[slot], 1))},
            Some(gemm_core::A_BD[1 - slot]));
        bds[gemm_core::B_BD[slot] as usize] = dma::tile_bd(B_ADDR[slot], (B_BYTES / 4) as u32,
            BdLocks {acq: Some((gemm_core::B_EMPTY[slot], -1)), rel: Some((gemm_core::B_FULL[slot], 1))},
            Some(gemm_core::B_BD[1 - slot]));
    }
    bds[gemm_core::C_BD as usize] = dma::tile_bd(COUT_ADDR, (COUT_BYTES / 4) as u32,
        BdLocks {acq: Some((gemm_core::C_FULL, -1)), rel: Some((gemm_core::C_EMPTY, 1))}, Some(gemm_core::C_BD));
    bds
}

/// Same lock numbering and initial values as `gemm_core` (`C_EMPTY` starts at 1, `PEER_*` at 0).
pub fn initial_locks() -> [i32; 16] { gemm_core::initial_locks_pair() }
