//! Builder-emitted Qwen4 grouped symmetric IU4 MoE GEMMs (fn-moe-sym route,
//! `HIPFIRE_QWEN4_MOE_SYM_IU4=1`): the SwiGLU gate/up and the down projection
//! over packed `block_i4_128` activations, for gfx1151
//! (`v_wmma_i32_16x16x16_iu4`, eight K16 steps per K128 epoch) and gfx1201
//! (`v_wmma_i32_16x16x32_iu4`, four K32 steps).
//!
//! Same 60-byte ABI, grid, block and output bytes as the hand-written
//! `kernels/src/qwen4_moe_iu4_sym.gfx1151.hip`:
//! `(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, Xq, Y_grouped,
//! M, K, x_row_div, m_total, x_src_rows)`.
//! - gate/up: block 64 (two waves), grid `(M/64, m_total/16)`; wave `w` of
//!   CTA `x` owns gate rows `32x + 16w ..+16` and the up rows `M/2` further,
//!   and stores `rt(silu(rt(g)) * rt(u))` into `Y[P][M/2]` (BF16 bits).
//! - down: block 128 (four waves), grid `(M/64, m_total/16)`; wave `w` owns
//!   rows `64x + 16w ..+16` and stores `rt(sum)` into `Y[P][M]`.
//!
//! Every wave is independent: operands are loaded straight into registers
//! (the next K128 epoch prefetched while the current one is consumed), no
//! LDS and no barrier. The tile of 16 grouped slots is `blockIdx.y`; a tile
//! past `m_total` or with expert id -1 returns before any load or store.
//! Each lane loads its slot `s = sorted[16*tile + lane%16]`; a padding slot
//! (`s < 0`) gets an out-of-range buffer offset, so its activation loads
//! return 0 without touching memory, and its outputs store +0.
//!
//! Numerics, per output and ascending epoch `h`: `C_h` is the exact int32
//! WMMA chain (A nibbles rebiased with XOR 0x88888888, signed A and X)
//! seeded with the magic 0x4b400000; `sum = fma(RN(d_h * sc_h), C_h -
//! 12582912.0, sum)` from `+0`, the HIP kernel's `fma(RN(sc*d), float(C),
//! sum)` exactly (`|C_h| <= 8192`). The row scale of each accumulator comes
//! from a `ds_swizzle_b32` broadcast: lane `(hi, k)` loads the header of the
//! row its broadcast partner needs (gfx11 C rows `2j + hi`, gfx12 `8hi + j`).
//! BF16 rounding is RNE with non-finite values passed through; the SiLU is
//! the hipcc `g / (1 + expf(-g)) * u` DAG ([`super::iu4_v2b::silu_mul`]).
use super::bf16::Bf16;
use super::common::{self, GatherTemps, add64, add64_imm, bload, bstore_b128, lit, op, s, s_add_i32, smem, sop, srd_tail, v, SRD_WORD3};
use super::iu4_fold::{self, MAGIC, REBIAS};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V, insn::MemoryClass, reg::Live};

pub const KERNARG_BYTES: u32 = 60;
pub const VGPR_CEILING: u16 = 256;
/// Slot indices (`x_row_div * x_src_rows`) the activation gather resolves exactly.
pub const MAX_SLOTS: u32 = common::GATHER_MAX_INDEX;
/// SiLU elements per interleaved group (two groups per lane).
const SILU_N: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind { GateUp, Down }
impl Kind {
    pub const ALL: [Kind; 2] = [Kind::GateUp, Kind::Down];
    pub fn tag(self) -> &'static str { match self { Self::GateUp => "gate_up", Self::Down => "down" } }
    /// Weight tensors folded per wave (gate and up, or down).
    fn tensors(self) -> u8 { match self { Self::GateUp => 2, Self::Down => 1 } }
    pub fn waves(self) -> u32 { match self { Self::GateUp => 2, Self::Down => 4 } }
    pub fn threads(self) -> u32 { self.waves() * 32 }
    /// Output columns of one CTA (`blockIdx.x` stride of the row base).
    pub fn cta_rows(self) -> u32 { match self { Self::GateUp => 32, Self::Down => 64 } }
    pub fn group_bytes(self) -> u32 { match self { Self::GateUp => 136, Self::Down => 68 } }
}
impl std::str::FromStr for Kind {
    type Err = String;
    fn from_str(t: &str) -> Result<Self, String> {
        match t { "gate_up" => Ok(Self::GateUp), "down" => Ok(Self::Down), _ => Err(format!("qwen4_moe_sym kind {t} (gate_up|down)")) }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec { pub arch: Arch, pub kind: Kind }

impl Spec {
    pub fn symbol(self) -> String {
        match self.kind {
            Kind::GateUp => format!("qwen4_moe_gate_up_silu_iu4_sym_pm_{}", self.arch.name()),
            Kind::Down => format!("qwen4_moe_down_iu4_sym_pm_{}", self.arch.name()),
        }
    }
    pub fn module(arch: Arch) -> String { format!("qwen4_moe_iu4_sym_pm_{}", arch.name()) }
    pub fn validate(self) -> Result<(), String> {
        if !matches!(self.arch, Arch::Gfx1151 | Arch::Gfx1201) { return Err("qwen4_moe_sym: built for gfx1151 and gfx1201 only".into()) }
        Ok(())
    }
    pub fn kernargs(self) -> KernargLayout {
        KernargLayout::new(KERNARG_BYTES).pointer("expert_weight_ptrs", 0).pointer("expert_tile_ids", 8)
            .pointer("sorted_slot_index", 16).pointer("Xq", 24).pointer("Y_grouped", 32)
            .hidden("M", 40, 4, "by_value").hidden("K", 44, 4, "by_value").hidden("x_row_div", 48, 4, "by_value")
            .hidden("m_total", 52, 4, "by_value").hidden("x_src_rows", 56, 4, "by_value")
    }
    /// Operand dwords per lane per K128 epoch of one 16-row/16-token fragment:
    /// gfx11 lanes `r` and `r+16` both carry all 16; gfx12 lanes split K32.
    fn aw(self) -> u8 { if self.arch.gfx12() { 8 } else { 16 } }
}

// VGPRs.
const TID: u8 = 0;
const XOFF: u8 = 16;   // d offset of this lane's slot row (or X_OOB)
const XQOFF: u8 = 17;  // nibble base offset (gfx12: + 8*hi)
const WOFF: u8 = 18;   // lr*row_bytes (gfx12: + 8*hi)
const HOFF: u8 = 19;   // header row * row_bytes
const YOFF: u8 = 20;
const MAGIC8: u8 = 24;
const CACC: u8 = 32;   // int32 chains, 8 per tensor
const SUM: u8 = 48;    // f32 sums, 8 per tensor
const SCF: u8 = 64;    // broadcast row scales, 8 per tensor
const TPROD: u8 = 80;  // fold products d*sc
const HF: u8 = 88;     // f32 scale of this lane's header row, per tensor
const SET0: u8 = 92;
// Epilogue (aliases dead K-loop registers).
const RT_TMP: u8 = 32;
const OWN: u8 = 64; const SEND: u8 = 68; const RECV: u8 = 70; const SEL: u8 = 72; const OUT: u8 = 76; const PACK: u8 = 80;

// SGPRs.
const KARG: u8 = 0;
const T0: u8 = 4; const T1: u8 = 5; const T64: u8 = 6;
const ARGS0: u8 = 8;   // ptrs 8:9, tiles 10:11, sorted 12:13, xq 14:15
const ARGS1: u8 = 16;  // Y 16:17, M 18, K 19
const ARGS2: u8 = 20;  // x_row_div 20, m_total 21
const ARGS3: u8 = 22;  // x_src_rows
const EID: u8 = 23;
const SRD_W: u8 = 24; const SRD_U: u8 = 28; const SRD_X: [u8; 2] = [32, 36]; const SRD_S: u8 = 40;
const TRIPS: u8 = 44; const XS2: u8 = 45; const ROWB: u8 = 46; const TILE16: u8 = 47; const WAVE: u8 = 48; const RBASE: u8 = 49;
const WGX: u8 = 50; const WGY: u8 = 51;
const LIVE: u8 = 52; const HIMASK: u8 = 54; const PSEL: u8 = 56; const WPTR: u8 = 58;
const MASKT: [u8; 2] = [60, 62];
const SILU_MASK: u8 = 24;
const SRD_Y: u8 = 64;

struct Gen { spec: Spec }

impl Gen {
    fn arch(&self) -> Arch { self.spec.arch }
    fn label(&self, name: &str) -> String { format!(".Lq4s_{}_{name}", self.spec.kind.tag()) }
    fn t(&self) -> u8 { self.spec.kind.tensors() }
    fn aw(&self) -> u8 { self.spec.aw() }
    /// Set `p`'s registers: A of tensor t, X, header word of tensor t, d.
    fn set_base(&self, p: usize) -> u8 { SET0 + p as u8 * self.set_len() }
    fn set_len(&self) -> u8 { (self.aw() * (self.t() + 1) + self.t() + 1).div_ceil(4) * 4 }
    fn a(&self, p: usize, t: u8) -> u8 { self.set_base(p) + self.aw() * t }
    fn x(&self, p: usize) -> u8 { self.set_base(p) + self.aw() * self.t() }
    fn h(&self, p: usize, t: u8) -> u8 { self.x(p) + self.aw() + t }
    fn d(&self, p: usize) -> u8 { self.h(p, self.t()) }
    fn srd(&self, t: u8) -> u8 { if t == 0 { SRD_W } else { SRD_U } }
    /// Header and nibble byte offsets of set `p` inside one two-epoch trip.
    fn header_off(&self, p: usize, t: u8) -> u32 { let _ = t; match self.spec.kind { Kind::GateUp => 4 * p as u32, Kind::Down => 68 * p as u32 } }
    fn nibble_off(&self, p: usize) -> u32 { match self.spec.kind { Kind::GateUp => 8 + 64 * p as u32, Kind::Down => 4 + 68 * p as u32 } }

    fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(VGPR_CEILING, 104)?;
        let l = |n: &str| self.label(n);
        let kernel = || Live::Between("entry".into(), l("epilogue"));
        let pro = || Live::Between("entry".into(), l("k_begin"));
        let epi = || Live::Between(l("epilogue"), l("end"));
        p.v::<1>("tid", TID, pro())?;
        for i in 1..16u8 { p.v::<1>("prologue_tmp", i, pro())?; }
        for (name, r) in [("x_off", XOFF), ("w_off", WOFF), ("h_off", HOFF)] { p.v::<1>(name, r, kernel())?; }
        if self.arch().gfx12() { p.v::<1>("xq_off", XQOFF, kernel())?; }
        p.v::<1>("y_off", YOFF, Live::Whole)?;
        p.v::<8>("magic8", MAGIC8, kernel())?;
        for t in 0..self.t() {
            p.v::<8>("cacc", CACC + 8 * t, kernel())?;
            p.v::<8>("sum", SUM + 8 * t, Live::Whole)?;
            p.v::<8>("scale_rows", SCF + 8 * t, kernel())?;
            p.v::<1>("scale_f32", HF + t, kernel())?;
        }
        p.v::<8>("fold_t", TPROD, kernel())?;
        // Operand sets at their load granularity (gfx11 b128, gfx12 b64):
        // A per tensor, X, then the header words and d.
        for set in 0..2 {
            for chunk in 0..(self.t() + 1) * 4 {
                let base = self.a(set, 0) + chunk * self.aw() / 4;
                if self.arch().gfx12() { p.v::<2>("operand", base, kernel())?; } else { p.v::<4>("operand", base, kernel())?; }
            }
            for t in 0..self.t() { p.v::<1>("header", self.h(set, t), kernel())?; }
            p.v::<1>("d", self.d(set), kernel())?;
        }
        // Epilogue registers over the dead K-loop ranges.
        for i in 0..8u8 { p.v::<1>("rt_tmp", RT_TMP + i, epi())?; }
        if self.spec.kind == Kind::GateUp {
            for i in 0..(6 * SILU_N) { p.v::<1>("silu_tmp", SET0 + i, epi())?; }
            for i in 0..(3 * SILU_N) { p.s::<2>("silu_mask", SILU_MASK + 2 * i, epi())?; }
        }
        p.v::<4>("out", OUT, epi())?;
        if self.arch().gfx12() {
            // Packed rows come straight from the sums.
        } else {
            for i in 0..4u8 { p.v::<1>("own", OWN + i, epi())?; }
            for i in 0..2u8 { p.v::<1>("send", SEND + i, epi())?; p.v::<1>("recv", RECV + i, epi())?; p.v::<1>("sel", SEL + i, epi())?; }
            for i in 0..4u8 { p.v::<1>("pack", PACK + i, epi())?; }
            p.s::<2>("hi_mask", HIMASK, epi())?;
            p.s::<1>("permlane_sel", PSEL, epi())?;
        }
        p.s::<2>("kernarg_ptr", KARG, Live::Whole)?;
        p.s::<4>("srd_y", SRD_Y, epi())?;
        if !self.arch().gfx12() { p.s::<1>("wg_x_in", 2, Live::Whole)?; p.s::<1>("wg_y_in", 3, Live::Whole)?; }
        p.s::<1>("s_tmp0", T0, kernel())?;
        p.s::<1>("s_tmp1", T1, kernel())?;
        p.s::<2>("s_tmp64", T64, kernel())?;
        p.s::<8>("kernargs_0x00", ARGS0, kernel())?;
        p.s::<2>("y_base", ARGS1, Live::Whole)?;
        p.s::<2>("kernargs_m_k", ARGS1 + 2, kernel())?;
        p.s::<2>("kernargs_0x30", ARGS2, kernel())?;
        p.s::<1>("x_src_rows", ARGS3, kernel())?;
        p.s::<1>("expert", EID, kernel())?;
        for t in 0..self.t() { p.s::<4>("srd_w", self.srd(t), kernel())?; }
        for r in SRD_X { p.s::<4>("srd_x", r, kernel())?; }
        p.s::<4>("srd_sorted", SRD_S, kernel())?;
        for (name, r) in [("trips", TRIPS), ("x_stride2", XS2), ("row_bytes", ROWB), ("tile16", TILE16), ("wave", WAVE), ("row_base", RBASE), ("wg_x", WGX), ("wg_y", WGY)] {
            p.s::<1>(name, r, kernel())?;
        }
        p.s::<2>("w_ptr", WPTR, kernel())?;
        p.s::<2>("live", LIVE, Live::Whole)?;
        for m in MASKT { p.s::<2>("cmp_mask", m, Live::Whole)?; }
        Ok(p)
    }
}

fn prologue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    let a = g.arch();
    let kind = g.spec.kind;
    smem(b, ARGS0, 8, KARG, 0)?;
    smem(b, ARGS1, 2, KARG, 0x20)?;
    smem(b, ARGS1 + 2, 2, KARG, 0x28)?;
    smem(b, ARGS2, 2, KARG, 0x30)?;
    smem(b, ARGS3, 1, KARG, 0x38)?;
    if a.gfx12() {
        // Workgroup ids: ttmp9 (x), ttmp7[15:0] (y).
        sop(b, format!("s_mov_b32 s{WGX}, ttmp9"), &[WGX], &[])?;
        sop(b, format!("s_and_b32 s{WGY}, ttmp7, 0xffff"), &[WGY], &[])?;
    } else {
        sop(b, format!("s_mov_b32 s{WGX}, s2"), &[WGX], &[2])?;
        sop(b, format!("s_mov_b32 s{WGY}, s3"), &[WGY], &[3])?;
    }
    // v1 = wave, v2 = lane, v3 = lr, v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(TID)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(TID)])?;
    op(b, format!("v_readfirstlane_b32 s{WAVE}, v1"), &[s(WAVE)], &[v(1)])?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    // A tile at or past m_total, or a sentinel tile, returns before any store.
    sop(b, format!("s_lshl_b32 s{TILE16}, s{WGY}, 4"), &[TILE16], &[WGY])?;
    sop(b, format!("s_cmp_ge_i32 s{TILE16}, s{}", ARGS2 + 1), &[], &[TILE16, ARGS2 + 1])?;
    op(b, format!("s_cbranch_scc1 {}", g.label("end")), &[], &[])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 2"), &[T0], &[WGY])?;
    add64(b, T64, ARGS0 + 2, T0)?;
    smem(b, EID, 1, T64, 0)?;
    sop(b, format!("s_cmp_lt_i32 s{EID}, 0"), &[], &[EID])?;
    op(b, format!("s_cbranch_scc1 {}", g.label("end")), &[], &[])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{EID}, 3"), &[T0], &[EID])?;
    add64(b, T64, ARGS0, T0)?;
    smem(b, WPTR, 2, T64, 0)?;
    // row_bytes = (K/256)*136 (QT44) or (K/128)*68 (QT53).
    let k = ARGS1 + 3;
    match kind {
        Kind::GateUp => { sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 8"), &[ROWB], &[k])?; sop(b, format!("s_mulk_i32 s{ROWB}, 0x88"), &[ROWB], &[ROWB])?; }
        Kind::Down => { sop(b, format!("s_lshr_b32 s{ROWB}, s{k}, 7"), &[ROWB], &[k])?; sop(b, format!("s_mulk_i32 s{ROWB}, 0x44"), &[ROWB], &[ROWB])?; }
    }
    // Two-epoch trips before the tail: (K/128 - 1) / 2.
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{k}, 7"), &[TRIPS], &[k])?;
    sop(b, format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(a)), &[TRIPS], &[TRIPS])?;
    sop(b, format!("s_lshr_b32 s{TRIPS}, s{TRIPS}, 1"), &[TRIPS], &[TRIPS])?;
    // Row base of this wave: x*cta_rows + 16*wave.
    let shift = if kind == Kind::GateUp { 5 } else { 6 };
    sop(b, format!("s_lshl_b32 s{RBASE}, s{WGX}, {shift}"), &[RBASE], &[WGX])?;
    sop(b, format!("s_lshl_b32 s{T0}, s{WAVE}, 4"), &[T0], &[WAVE])?;
    sop(b, format!("{} s{RBASE}, s{RBASE}, s{T0}", s_add_i32(a)), &[RBASE], &[RBASE, T0])?;
    // Sorted slots of this tile.
    sop(b, format!("s_lshl_b32 s{T0}, s{WGY}, 6"), &[T0], &[WGY])?;
    add64(b, SRD_S, ARGS0 + 4, T0)?;
    sop(b, format!("s_mov_b32 s{}, 64", SRD_S + 2), &[SRD_S + 2], &[])?;
    sop(b, format!("s_mov_b32 s{}, {}", SRD_S + 3, lit(SRD_WORD3)), &[SRD_S + 3], &[])?;
    op(b, "v_lshlrev_b32_e32 v5, 2, v3", &[v(5)], &[v(3)])?;
    bload(b, 1, 6, 5, SRD_S, 0)?;
    // Activation descriptors: epoch 0 (X0) and 1 (X1), x_src_rows*72 records.
    sop(b, format!("s_mul_i32 s{T0}, s{ARGS3}, 0x48"), &[T0], &[ARGS3])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD_X[0], ARGS0 + 6), &[SRD_X[0]], &[ARGS0 + 6])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD_X[0] + 1, ARGS0 + 7), &[SRD_X[0] + 1], &[ARGS0 + 7])?;
    srd_tail(b, SRD_X[0], Some(T0))?;
    add64(b, SRD_X[1], ARGS0 + 6, T0)?;
    srd_tail(b, SRD_X[1], Some(T0))?;
    sop(b, format!("s_lshl_b32 s{XS2}, s{T0}, 1"), &[XS2], &[T0])?;
    // Weight descriptors: rows rbase.. of the expert (up rows M/2 further).
    sop(b, format!("s_mul_i32 s{T0}, s{RBASE}, s{ROWB}"), &[T0], &[RBASE, ROWB])?;
    add64(b, SRD_W, WPTR, T0)?;
    srd_tail(b, SRD_W, None)?;
    if kind == Kind::GateUp {
        sop(b, format!("s_lshr_b32 s{T1}, s{}, 1", ARGS1 + 2), &[T1], &[ARGS1 + 2])?;
        sop(b, format!("s_mul_i32 s{T1}, s{T1}, s{ROWB}"), &[T1], &[T1, ROWB])?;
        add64(b, SRD_U, SRD_W, T1)?;
        srd_tail(b, SRD_U, None)?;
    }
    // Lane offsets. Weights: lr*row_bytes (gfx12 + 8*hi, its K32 half).
    op(b, format!("v_mul_u32_u24_e32 v{WOFF}, s{ROWB}, v3"), &[v(WOFF)], &[s(ROWB), v(3)])?;
    if a.gfx12() { op(b, format!("v_lshl_add_u32 v{WOFF}, v4, 3, v{WOFF}"), &[v(WOFF)], &[v(4), v(WOFF)])?; }
    // Header row: the row whose scale this lane's broadcast slot serves.
    if a.gfx12() {
        op(b, "v_and_b32_e32 v7, 7, v3", &[v(7)], &[v(3)])?;
        op(b, "v_lshl_or_b32 v7, v4, 3, v7", &[v(7)], &[v(4), v(7)])?;
    } else {
        op(b, "v_and_b32_e32 v7, 14, v3", &[v(7)], &[v(3)])?;
        op(b, "v_or_b32_e32 v7, v7, v4", &[v(7)], &[v(7), v(4)])?;
    }
    op(b, format!("v_mul_u32_u24_e32 v{HOFF}, s{ROWB}, v7"), &[v(HOFF)], &[s(ROWB), v(7)])?;
    // Y offset: ((16*tile + lr)*row_len + rbase + 8*hi)*2.
    match kind {
        Kind::GateUp => sop(b, format!("s_lshr_b32 s{T1}, s{}, 1", ARGS1 + 2), &[T1], &[ARGS1 + 2])?,
        Kind::Down => sop(b, format!("s_mov_b32 s{T1}, s{}", ARGS1 + 2), &[T1], &[ARGS1 + 2])?,
    }
    op(b, format!("v_add_nc_u32_e32 v8, s{TILE16}, v3"), &[v(8)], &[s(TILE16), v(3)])?;
    op(b, format!("v_mul_lo_u32 v8, v8, s{T1}"), &[v(8)], &[v(8), s(T1)])?;
    op(b, format!("v_lshl_add_u32 v9, v4, 3, s{RBASE}"), &[v(9)], &[v(4), s(RBASE)])?;
    op(b, "v_add_nc_u32_e32 v8, v8, v9", &[v(8)], &[v(8), v(9)])?;
    op(b, format!("v_lshlrev_b32_e32 v{YOFF}, 1, v8"), &[v(YOFF)], &[v(8)])?;
    // Activation row offset: (slot / x_row_div) * 72, or the gather's
    // out-of-range offset for a padding slot so its loads read 0.
    common::gather_offset(b, XOFF, 6, Some(ARGS2), iu4_fold::XBLK_BYTES, LIVE, GatherTemps { v: [11, 12, 13, 10], mask: MASKT[0] })?;
    if a.gfx12() { op(b, format!("v_lshl_add_u32 v{XQOFF}, v4, 3, v{XOFF}"), &[v(XQOFF)], &[v(4), v(XOFF)])?; }
    load_set(b, g, 0)?;
    for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, {}", MAGIC8 + j, lit(MAGIC)), &[v(MAGIC8 + j)], &[])?; }
    for t in 0..g.t() { for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, 0", SUM + 8 * t + j), &[v(SUM + 8 * t + j)], &[])?; } }
    Ok(())
}

/// Issue set `p`'s loads (epoch parity p of the current trip): header words,
/// d, the activation nibbles, then the weight nibbles, as one clause.
fn load_set(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let gfx12 = g.arch().gfx12();
    let xq = if gfx12 { XQOFF } else { XOFF };
    let (width, step) = if gfx12 { (2u8, 2u8) } else { (4u8, 4u8) };
    b.clause(|b| {
        for t in 0..g.t() { bload(b, 1, g.h(p, t), HOFF, g.srd(t), g.header_off(p, t))?; }
        bload(b, 1, g.d(p), XOFF, SRD_X[p], 0)?;
        for i in 0..4u8 { bload(b, width, g.x(p) + step * i, xq, SRD_X[p], 8 + 16 * u32::from(i))?; }
        for t in 0..g.t() {
            for i in 0..4u8 { bload(b, width, g.a(p, t) + step * i, WOFF, g.srd(t), g.nibble_off(p) + 16 * u32::from(i))?; }
        }
        Ok(())
    })
}

/// Fold set `p` (one K128 epoch) into the sums.
fn compute(b: &mut Builder, g: &Gen, p: usize) -> Result<(), String> {
    let a = g.arch();
    // Row scales: this lane's header word -> f32 -> the eight rows of its
    // accumulators by ds_swizzle broadcast within each 16-lane half.
    for t in 0..g.t() {
        op(b, format!("v_cvt_f32_f16_e64 v{}, v{}.l", HF + t, g.h(p, t)), &[v(HF + t)], &[v(g.h(p, t))])?;
        for j in 0..8u8 {
            let k = if a.gfx12() { j } else { 2 * j };
            let text = format!("ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{k})", SCF + 8 * t + j, HF + t);
            b.ds_crosslane(crate::insn::Instruction::new(text, vec![v(SCF + 8 * t + j)], vec![v(HF + t)]).memory(MemoryClass::DsLoad))?;
        }
    }
    for t in 0..g.t() {
        for i in 0..g.aw() {
            let r = g.a(p, t) + i;
            op(b, format!("v_xor_b32_e32 v{r}, {}, v{r}", lit(REBIAS)), &[v(r)], &[v(r)])?;
        }
    }
    for i in 0..g.aw() / 2 {
        for t in 0..g.t() {
            iu4_fold::wmma_step(b, a, V::<8>(CACC + 8 * t), V::<2>(g.a(p, t) + 2 * i), V::<2>(g.x(p) + 2 * i), i == 0, V::<8>(MAGIC8))?;
        }
    }
    for t in 0..g.t() { iu4_fold::fold_pass(b, CACC + 8 * t, SUM + 8 * t, SCF + 8 * t, g.d(p), TPROD)?; }
    Ok(())
}

fn advance(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for t in 0..g.t() { add64_imm(b, g.srd(t), 136)?; }
    for r in SRD_X { add64(b, r, r, XS2)?; }
    Ok(())
}

fn kloop(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(&g.label("k_begin"))?;
    let (head, done) = (g.label("k_loop"), g.label("k_loop_end"));
    let entry = b.ledger.shape();
    op(b, format!("s_cmp_eq_u32 s{TRIPS}, 0"), &[], &[s(TRIPS)])?;
    op(b, format!("s_cbranch_scc1 {done}"), &[], &[])?;
    b.loop_(&head, |b| {
        load_set(b, g, 1)?;
        compute(b, g, 0)?;
        advance(b, g)?;
        load_set(b, g, 0)?;
        compute(b, g, 1)?;
        op(b, format!("{} s{TRIPS}, s{TRIPS}, -1", s_add_i32(b.spec.arch)), &[s(TRIPS)], &[s(TRIPS)])?;
        op(b, format!("s_cmp_lg_u32 s{TRIPS}, 0"), &[], &[s(TRIPS)])?;
        op(b, format!("s_cbranch_scc1 {head}"), &[], &[])
    })?;
    if b.ledger.shape() != entry { return Err("k loop exit ledger differs from its entry".into()) }
    b.label(&done)?;
    match g.spec.kind {
        // K % 256 == 0: an even epoch count, two epochs left.
        Kind::GateUp => {
            load_set(b, g, 1)?;
            compute(b, g, 0)?;
            compute(b, g, 1)
        }
        // K % 128 == 0: one epoch left when K/128 is odd, else two.
        Kind::Down => {
            let odd = g.label("tail_odd");
            let at_branch = b.ledger.clone();
            op(b, format!("s_bitcmp1_b32 s{}, 7", ARGS1 + 3), &[], &[s(ARGS1 + 3)])?;
            op(b, format!("s_cbranch_scc1 {odd}"), &[], &[])?;
            load_set(b, g, 1)?;
            compute(b, g, 0)?;
            compute(b, g, 1)?;
            op(b, format!("s_branch {}", g.label("epilogue")), &[], &[])?;
            b.ledger = at_branch;
            b.label(&odd)?;
            compute(b, g, 0)
        }
    }
}

fn epilogue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(&g.label("epilogue"))?;
    let vals = SUM;
    if g.spec.kind == Kind::GateUp {
        // g and u rounded to BF16 before the shipped SwiGLU expression.
        for t in 0..2u8 { for j in 0..8u8 { Bf16::rne_finite_passthrough(b, SUM + 8 * t + j, RT_TMP + j, MASKT[usize::from(j % 2)], true)?; } }
        for grp in 0..2u8 { super::iu4_v2b::silu_mul(b, SUM + SILU_N * grp, SUM + 8 + SILU_N * grp, SET0, SILU_MASK, SILU_N)?; }
    }
    for j in 0..8u8 { Bf16::rne_finite_passthrough(b, vals + j, RT_TMP + j, MASKT[usize::from(j % 2)], false)?; }
    // Padding slots store +0.
    for j in 0..8u8 { op(b, format!("v_cndmask_b32_e64 v{0}, 0, v{0}, s{LIVE}", vals + j), &[v(vals + j)], &[v(vals + j), s(LIVE)])?; }
    let perm = |b: &mut Builder, dst: u8, hi_src: u8, lo_src: u8, sel: String| -> Result<(), String> {
        let mut uses = vec![v(hi_src), v(lo_src)];
        if sel.starts_with('v') { uses.push(v(sel[1..].parse::<u8>().map_err(|e| e.to_string())?)); }
        op(b, format!("v_perm_b32 v{dst}, v{hi_src}, v{lo_src}, {sel}"), &[v(dst)], &uses)
    };
    if g.arch().gfx12() {
        // Lane (hi, lr) holds rows 8hi..8hi+7 of token lr: pack and store.
        for k in 0..4u8 { perm(b, OUT + k, vals + 2 * k + 1, vals + 2 * k, lit(0x0706_0302))?; }
    } else {
        // Lane (hi, lr) holds rows 2j+hi of token lr. Exchange with the
        // other half so lane hi stores rows 8hi..8hi+7: lane 0 sends rows
        // 8..14 (even), lane 1 sends rows 1..7 (odd), via v_permlanex16.
        // A VOP3 lane-mask operand is a pair for M7; the high word is unused in wave32.
        op(b, format!("s_mov_b32 s{HIMASK}, 0xffff0000"), &[s(HIMASK)], &[])?;
        op(b, format!("s_mov_b32 s{}, 0", HIMASK + 1), &[s(HIMASK + 1)], &[])?;
        op(b, format!("s_mov_b32 s{PSEL}, 0x76543210"), &[s(PSEL)], &[])?;
        for k in 0..4u8 {
            op(b, format!("v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}", OWN + k, vals + k, vals + 4 + k), &[v(OWN + k)], &[v(vals + k), v(vals + 4 + k), s(HIMASK)])?;
        }
        for m in 0..2u8 {
            // hi = 0 sends pack(H[4+2m], H[5+2m]); hi = 1 sends pack(H[2m], H[2m+1]).
            perm(b, PACK + 2 * m, vals + 5 + 2 * m, vals + 4 + 2 * m, lit(0x0706_0302))?;
            perm(b, PACK + 2 * m + 1, vals + 1 + 2 * m, vals + 2 * m, lit(0x0706_0302))?;
            op(b, format!("v_cndmask_b32_e64 v{}, v{}, v{}, s{HIMASK}", SEND + m, PACK + 2 * m, PACK + 2 * m + 1), &[v(SEND + m)], &[v(PACK + 2 * m), v(PACK + 2 * m + 1), s(HIMASK)])?;
            op(b, format!("v_mov_b32_e32 v{}, v{}", RECV + m, SEND + m), &[v(RECV + m)], &[v(SEND + m)])?;
            op(b, format!("v_permlanex16_b32 v{}, v{}, s{PSEL}, 0xfedcba98", RECV + m, SEND + m), &[v(RECV + m)], &[v(RECV + m), v(SEND + m), s(PSEL)])?;
        }
        // Byte selectors (src0 = received word, src1 = own row): hi = 0
        // puts its own row low, hi = 1 puts the received row low.
        for (i, (lo, hi)) in [(0x0504_0302u32, 0x0302_0504u32), (0x0706_0302, 0x0302_0706)].into_iter().enumerate() {
            let r = SEL + i as u8;
            op(b, format!("v_mov_b32_e32 v{r}, {}", lit(hi)), &[v(r)], &[])?;
            op(b, format!("v_cndmask_b32_e64 v{r}, {}, v{r}, s{HIMASK}", lit(lo)), &[v(r)], &[v(r), s(HIMASK)])?;
        }
        for k in 0..4u8 { perm(b, OUT + k, RECV + k / 2, OWN + k, format!("v{}", SEL + k % 2))?; }
    }
    // Raw buffer store over Y (the same VMEM family as every load).
    sop(b, format!("s_mov_b32 s{SRD_Y}, s{ARGS1}"), &[SRD_Y], &[ARGS1])?;
    sop(b, format!("s_mov_b32 s{}, s{}", SRD_Y + 1, ARGS1 + 1), &[SRD_Y + 1], &[ARGS1 + 1])?;
    srd_tail(b, SRD_Y, None)?;
    bstore_b128(b, OUT, YOFF, SRD_Y, 0)
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let g = Gen { spec };
    let kspec = KernelSpec {
        kernel_id: "qwen4_moe_sym".into(), variant: spec.kind.tag().into(), arch: spec.arch, symbol: spec.symbol(),
        kernargs: spec.kernargs(), user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
        workgroup_size: spec.kind.threads() as u16, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, g.plan()?);
    b.enable_delay_alu();
    prologue(&mut b, &g)?;
    kloop(&mut b, &g)?;
    epilogue(&mut b, &g)?;
    b.label(&g.label("end"))?;
    b.push(crate::insn::Sop::End.encode(spec.arch)?)?;
    b.finish()
}

/// Both entries of one architecture as one code object.
pub fn emit_module(arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Kind::ALL.into_iter().map(|kind| emit(Spec { arch, kind })).collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, &Spec::module(arch))?;
    Ok((emitted, text, proof))
}
