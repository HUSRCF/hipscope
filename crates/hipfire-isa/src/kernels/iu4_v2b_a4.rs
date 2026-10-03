//! Builder-emitted gfx1151 (RDNA3.5, wave32) MQ4V2 x block_i4_128 gate/up
//! SiLU GEMM on an M512 x N128 tile: the retile of the certified V2B SiLU
//! entry (`iu4_v2b.rs`) that puts one whole 256-feature `h` group of each
//! token in one CTA (stage 1 of the A4 epilogue fusion,
//! `a4-fusion/design.md` §3). Same `G, U, Xq, Y, M, K, N` ABI and the same
//! `h` bits as the V2B entry.
//!
//! Tile: 512 virtual rows (gate/up interleaved at 16 rows: 256 h features)
//! x 128 tokens, 16 waves, one CTA per WGP. Wave `w = 2*wr + wc` owns
//! virtual rows `64*wr..+64` (four 16-row fragments `a`, even = gate, odd =
//! up, h features `32*wr + 16*(a/2)..+16`) and tokens `64*wc..+64` (four
//! 16-token fragments `c`): V2B's 4a x 4c fragment structure, fold and
//! scale broadcast, and its epilogue, unchanged.
//!
//! K is staged in K64 halves (A 16 KiB + X 4 KiB per half): a K128 period
//! needs both halves resident (one set of chain registers walks the four
//! passes in turn), so the LDS holds three 20 KiB slots: the period's first
//! half in a two-slot ring and its second half in a fixed slot. Within a
//! period pass `a` (eight K16 WMMA steps) reads its first four slices from
//! the ring's current slot and the last four from the fixed slot, except
//! pass 3, which reads the fixed slot first. Barrier B1 after pass 3's
//! fourth step retires the fixed slot, which then takes the next period's
//! second half; the period-end barrier B2 rotates the ring (the next first
//! half, stored after pass `PUBLISH_AFTER_PASS`) and publishes the fixed
//! slot. Two barriers per K128 instead of one.
//!
//! Numerics, per output and K128 period `e`: `C_e` is the exact int32
//! `v_wmma_i32_16x16x16_iu4` chain over the eight K16 slices of the period,
//! seeded with the magic `0x4b400000`: a wrapping integer sum, the same
//! value in any slice order. `sum = fma(RN(d_e * sc_e), C_e + (-12582912.0),
//! sum)` from `sum = +0`, ascending `e`, exactly V2B's fold DAG; `h` is
//! `Epilogue::silu_dense` on the same register pairs. Hence `h` is
//! bit-identical to V2B's.
//!
//! Launch contract (block `[512,1,1]`, dynamic LDS 65536; `M % 256 == 0`
//! (M = h rows), `N % 128 == 0`, `K % 256 == 0`; Y = h token-major
//! `[N][M]` f32): `G, U, Xq, Y, M, K, N`, grid `[N/128, 2M/512]`.
use super::common::{add64, add64_imm, lit, mem, op, s, sop, sr, v, vr};
use super::iu4_fold::{GROUP_BYTES, MAGIC, REBIAS, XBLK_BYTES};
use super::iu4_gemm::ds_offsets;
use super::iu4_v2c::off;
use super::gemm_uk::{Chain, DENSE_SILU_TEMPS, Epilogue, Iu4};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V,
    insn::{Instruction, MemoryClass, Sop},
    reg::Live};
use peacemaker_author::{Free, Gfx1151, Gfx11Waits, LdsRegion, LdsWrite, MmaIu4, Pending, Published, Ring, Scc, Wave, WgUniform,
    Workgroup, Writing, prime, ready, retire, rotate};

pub const TILE_M: u32 = 512;
pub const TILE_N: u32 = 128;
pub const EPOCH_K: u32 = 64;
pub const WAVES: u32 = 16;
pub const THREADS: u16 = 512;
pub const LDS_BYTES: u32 = 65536;
pub const VGPR_CEILING: u16 = 256;
/// One K64 half: A (512 rows x 32 B) then X (128 tokens x 32 B).
pub const A_BYTES: u32 = 16384;
pub const X_BYTES: u32 = 4096;
pub const SLOT_BYTES: u32 = A_BYTES + X_BYTES;
/// Ring slots 0, 1 (first halves) and the fixed second-half slot.
const SLOT_BASE: [u32; 3] = [0, SLOT_BYTES, 2 * SLOT_BYTES];
pub const MODULE: &str = "gemm_mq4g256v2_gate_up_silu_a4_iu4_pm_v2b_gfx1151";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi { SiluM512 }
impl Epi {
    pub const ALL: [Epi; 1] = [Epi::SiluM512];
    pub fn tag(self) -> &'static str { match self { Self::SiluM512 => "m512" } }
}
impl std::str::FromStr for Epi {
    type Err = String;
    fn from_str(t: &str) -> Result<Self, String> {
        match t { "m512" => Ok(Self::SiluM512), _ => Err(format!("iu4_v2b_a4 epilogue {t} (m512)")) }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec { pub arch: Arch, pub epi: Epi }

impl Spec {
    pub fn symbol(self) -> String {
        match self.epi {
            Epi::SiluM512 => format!("gemm_mq4g256v2_gate_up_silu_iu4_pm_v2b_m512_{}", self.arch.name()),
        }
    }
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1151 { return Err("iu4_v2b_a4: the M512 V2B tile is built for gfx1151 only".into()) }
        Ok(())
    }
    pub fn kernargs(self) -> KernargLayout {
        match self.epi {
            Epi::SiluM512 => KernargLayout::new(44).pointer("G", 0).pointer("U", 8).pointer("Xq", 16).pointer("Y", 24)
                .hidden("M", 32, 4, "by_value").hidden("K", 36, 4, "by_value").hidden("N", 40, 4, "by_value"),
        }
    }
}

/// Physical registers. The K loop's set is V2B's (same chain, accumulator,
/// fold and scale registers, so the same fold banks), plus a second staging
/// packet and one LDS base per slot. VGPRs above v159 and v0..v31 are
/// reused by the epilogue once the K loop is done.
struct Regs;
impl Regs {
    const C: u8 = 0;          // int32 WMMA chains C[c] = v[8c..8c+7]
    const ACC: u8 = 32;       // f32 sums acc[a][c] = v[32 + 8*(4a+c) ..]
    const MAGIC: u8 = 160;    // 8 x 0x4b400000, the chain seed
    const AV: [u8; 2] = [168, 172];  // A fragment pairs (slice s, s+1)
    const XV: [u8; 2] = [176, 184];  // X fragments c=0..3 of one slice
    const SCF: u8 = 192;      // row scales of the fold pass (ds_swizzle broadcast)
    const T: u8 = 200;        // fold products t = d*sc
    const DC: [u8; 2] = [208, 212];  // d of the four token fragments, by period parity
    const WSF: [u8; 2] = [220, 255]; // f32 scale words (v255 keeps the reservation a whole granule)
    const WS: [u8; 2] = [221, 222];  // raw f16 scale words
    const YOFF: u8 = 223;     // Y byte offset of (token wc*64+lr, h row 32wr + 8hi)
    /// Staged A of K64 half h: G slices (hi, 2+hi) then U slices (hi, 2+hi), 4 x b64.
    const STA: [u8; 2] = [224, 246];
    /// Staged X of K64 half h: one b64.
    const STX: [u8; 2] = [232, 216];
    const A_OFF: u8 = 234; const X_OFF: u8 = 235; const LWS: u8 = 236; const LXS: u8 = 237;
    /// LDS store addresses of the staged A and X packets (slot 0).
    const ST: u8 = 238; const STXA: u8 = 239;
    /// A / X fragment read bases of slots 0, 1 (ring) and 2 (fixed).
    const AB: [u8; 3] = [240, 241, 242];
    const XB: [u8; 3] = [243, 244, 245];
    // SGPRs
    const KARG: u8 = 0; const WGX: u8 = 2; const WGY: u8 = 3;
    const T0: u8 = 4; const T1: u8 = 5; const T64: u8 = 6;
    const ARGS: u8 = 8; const ARGS2: u8 = 16; const ARGS3: u8 = 18;
    const ROWB: u8 = 20; const WAVE: u8 = 21; const TRIPS: u8 = 22; const WR: u8 = 23; const WC: u8 = 24;
    const N72: u8 = 25; const M64: u8 = 26;
    /// Staging bases: gate rows and up rows `t0 + 16*wave`.
    const SS: [u8; 2] = [28, 38];
    const SW0: u8 = 30; const SW1: u8 = 32; const SX: u8 = 34; const SY: u8 = 36;
    const MUL: u8 = 42;
    /// SiLU epilogue per-element masks (underflow, overflow, numerator scale).
    const MASK: u8 = 48;
}

/// Kernel-argument SGPRs: `G, U, Xq, Y, M, K, N`.
const ARG_G: u8 = 8; const ARG_U: u8 = 10; const ARG_XQ: u8 = 12; const ARG_Y: u8 = 14;
const ARG_M: u8 = 16; const ARG_K: u8 = 17; const ARG_N: u8 = 18;

/// SiLU elements folded per interleaved group.
const SILU_GROUP: u8 = 8;
/// The next period's first half is published after this pass's fold.
const PUBLISH_AFTER_PASS: usize = 2;
/// Barrier B1 follows this step (pass 3's last read of the fixed slot).
const B1_AFTER_STEP: usize = 27;
/// WMMA step after which pass 0's scale broadcast issues (V2B `Early`).
const SCALES0_STEP: usize = 2;

struct Gen { spec: Spec }

impl Gen {
    fn label(&self, name: &str) -> String { format!(".Lv2b_a4_{}_{name}", self.spec.epi.tag()) }

    fn plan(&self) -> Result<RegPlan, String> {
        let mut p = RegPlan::new(VGPR_CEILING, 104)?;
        let l = |n: &str| self.label(n);
        let kl = || Live::Between(l("k_begin"), l("epilogue"));
        let kernel = || Live::Between("entry".into(), l("epilogue"));
        let pro = || Live::Between("entry".into(), l("k_begin"));
        let epi = || Live::Between(l("epilogue"), l("end"));
        let body = || Live::Between(l("epi_body"), l("end"));
        for c in 0..4u8 { p.v::<8>("C", Regs::C + 8 * c, kl())?; }
        for i in 0..32u8 { p.v::<1>(if i == 0 { "tid" } else { "prologue_tmp" }, Regs::C + i, pro())?; }
        for i in 0..4u8 { p.v::<1>("y_off_c", Regs::C + i, epi())?; }
        for k in 0..16u8 { p.v::<8>("acc", Regs::ACC + 8 * k, Live::Whole)?; }
        p.v::<8>("magic8", Regs::MAGIC, kernel())?;
        for r in Regs::AV { p.v::<4>("a_frag_pair", r, kernel())?; }
        for r in Regs::XV { p.v::<8>("x_frags", r, kernel())?; }
        p.v::<8>("scale_rows", Regs::SCF, kl())?;
        p.v::<8>("fold_t", Regs::T, kl())?;
        for r in Regs::DC { p.v::<4>("d_x", r, kernel())?; }
        for r in Regs::WSF { p.v::<1>("scale_f32", r, kl())?; }
        for r in Regs::WS { p.v::<1>("scale_f16", r, kernel())?; }
        p.v::<1>("y_off", Regs::YOFF, Live::Between("entry".into(), l("epi_body")))?;
        for h in 0..2usize {
            for i in 0..4u8 { p.v::<2>("stage_a", Regs::STA[h] + 2 * i, kernel())?; }
            p.v::<2>("stage_x", Regs::STX[h], kernel())?;
        }
        for (name, r) in [("a_off", Regs::A_OFF), ("x_off", Regs::X_OFF), ("lws", Regs::LWS), ("lxs", Regs::LXS), ("st", Regs::ST), ("stx", Regs::STXA),
            ("ab0", Regs::AB[0]), ("ab1", Regs::AB[1]), ("ab2", Regs::AB[2]), ("xb0", Regs::XB[0]), ("xb1", Regs::XB[1]), ("xb2", Regs::XB[2])] {
            p.v::<1>(name, r, kernel())?;
        }
        for i in 0..(SILU_GROUP * DENSE_SILU_TEMPS / 8) { p.v::<8>("silu_tmp", 160 + 8 * i, body())?; }
        for i in 0..(3 * SILU_GROUP) { p.s::<2>("silu_mask", Regs::MASK + 2 * i, body())?; }
        p.s::<2>("kernarg_ptr", Regs::KARG, Live::Whole)?;
        p.s::<1>("wg_x", Regs::WGX, Live::Whole)?;
        p.s::<1>("wg_y", Regs::WGY, Live::Whole)?;
        p.s::<1>("s_tmp0", Regs::T0, Live::Whole)?;
        p.s::<1>("s_tmp1", Regs::T1, Live::Whole)?;
        p.s::<2>("s_tmp64", Regs::T64, Live::Whole)?;
        p.s::<8>("kernargs_0x00", Regs::ARGS, Live::Whole)?;
        p.s::<2>("kernargs_0x20", Regs::ARGS2, Live::Whole)?;
        p.s::<1>("kernargs_0x28", Regs::ARGS3, Live::Whole)?;
        for (name, r) in [("row_bytes", Regs::ROWB), ("wave", Regs::WAVE), ("trips", Regs::TRIPS), ("wr", Regs::WR), ("wc", Regs::WC),
            ("n72", Regs::N72), ("m64", Regs::M64)] {
            p.s::<1>(name, r, Live::Whole)?;
        }
        for (name, r) in [("stage_base_g", Regs::SS[0]), ("stage_base_u", Regs::SS[1]), ("scale_base0", Regs::SW0), ("scale_base1", Regs::SW1),
            ("x_base", Regs::SX), ("y_base", Regs::SY), ("mul64", Regs::MUL)] {
            p.s::<2>(name, r, Live::Whole)?;
        }
        Ok(p)
    }
}

/// LDS tags: the A (weight) and X (activation) parts of the first-half ring
/// (`A`, `X`) and of the fixed second-half slot (`AH`, `XH`).
pub enum A {}
pub enum X {}
pub enum AH {}
pub enum XH {}
type Wg<'b> = Workgroup<'b, Gfx1151, Builder>;
type Wv<'b> = Wave<'b, Gfx1151, Builder>;
/// The K loop's state at a period boundary: the period's first half
/// published in the ring's current slot (the other free) and its second
/// half published in the fixed slot.
type Rings = (Ring<A, Published, Free>, Ring<X, Published, Free>, LdsRegion<AH, Published>, LdsRegion<XH, Published>);
type Staged<R> = (Ring<R, Published, Writing>, Pending<Gfx11Waits, LdsWrite<R>>);
type StagedH<R> = (LdsRegion<R, Writing>, Pending<Gfx11Waits, LdsWrite<R>>);

/// Slots 0, 1 (ring) and 2 (fixed), A then X in each.
fn declare_lds(wg: &mut Wg) -> Result<(Ring<A, Free, Free>, Ring<X, Free, Free>, LdsRegion<AH, Free>, LdsRegion<XH, Free>), String> {
    let (a0, x0) = (wg.lds("A0", SLOT_BASE[0], A_BYTES)?, wg.lds("X0", SLOT_BASE[0] + A_BYTES, X_BYTES)?);
    let (a1, x1) = (wg.lds("A1", SLOT_BASE[1], A_BYTES)?, wg.lds("X1", SLOT_BASE[1] + A_BYTES, X_BYTES)?);
    let (ah, xh) = (wg.lds("AH", SLOT_BASE[2], A_BYTES)?, wg.lds("XH", SLOT_BASE[2] + A_BYTES, X_BYTES)?);
    Ok((Ring::new(a0, a1), Ring::new(x0, x1), ah, xh))
}

/// `dst = base + x*y` (64-bit), `x*y` as a full 64-bit product.
fn mul_add64(b: &mut Builder, dst: u8, base: u8, x: u8, y: u8) -> Result<(), String> {
    let (lo, hi) = (Regs::MUL, Regs::MUL + 1);
    sop(b, format!("s_mul_i32 s{lo}, s{x}, s{y}"), &[lo], &[x, y])?;
    sop(b, format!("s_mul_hi_u32 s{hi}, s{x}, s{y}"), &[hi], &[x, y])?;
    sop(b, format!("s_add_u32 s{dst}, s{base}, s{lo}"), &[dst], &[base, lo])?;
    sop(b, format!("s_addc_u32 s{}, s{}, s{hi}", dst + 1, base + 1), &[dst + 1], &[base + 1, hi])
}

/// Kernel arguments, tile bases and every lane-invariant offset; stages
/// period 0 (both halves) and issues period 0's head.
fn prologue(wg: &mut Wg, (ra, rx, ah, xh): (Ring<A, Free, Free>, Ring<X, Free, Free>, LdsRegion<AH, Free>, LdsRegion<XH, Free>)) -> Result<Rings, String> {
    let b = wg.isa();
    let (t0, t1) = (Regs::T0, Regs::T1);
    let (by, bx) = (Regs::WGY, Regs::WGX);
    // gfx11 llvm-objdump spells a zero SMEM offset `null`; parse-back compares canonical text.
    mem(b, format!("s_load_b256 s[{}:{}], s[0:1], null", Regs::ARGS, Regs::ARGS + 7), &[sr(Regs::ARGS, 8)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
    mem(b, format!("s_load_b64 s[{}:{}], s[0:1], 0x20", Regs::ARGS2, Regs::ARGS2 + 1), &[sr(Regs::ARGS2, 2)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
    mem(b, format!("s_load_b32 s{}, s[0:1], 0x28", Regs::ARGS3), &[s(Regs::ARGS3)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
    // v0 = tid; v1 = wave; v2 = lane; v3 = lr; v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(0)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(0)])?;
    op(b, format!("v_readfirstlane_b32 s{}, v1", Regs::WAVE), &[s(Regs::WAVE)], &[v(1)])?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    sop(b, format!("s_lshr_b32 s{}, s{}, 1", Regs::WR, Regs::WAVE), &[Regs::WR], &[Regs::WAVE])?;
    sop(b, format!("s_and_b32 s{}, s{}, 1", Regs::WC, Regs::WAVE), &[Regs::WC], &[Regs::WAVE])?;
    // row_bytes = (K/256)*136; trips = K/256 - 1 whole two-period trips before the tail pair.
    sop(b, format!("s_lshr_b32 s{}, s{}, 8", Regs::ROWB, ARG_K), &[Regs::ROWB], &[ARG_K])?;
    sop(b, format!("s_add_i32 s{}, s{}, -1", Regs::TRIPS, Regs::ROWB), &[Regs::TRIPS], &[Regs::ROWB])?;
    sop(b, format!("s_mulk_i32 s{}, {}", Regs::ROWB, lit(GROUP_BYTES)), &[Regs::ROWB], &[Regs::ROWB])?;
    // t0 = 256*by: the tile's first h row (and gate/up row).
    sop(b, format!("s_lshl_b32 s{t0}, s{by}, 8"), &[t0], &[by])?;
    // Staging wave w owns tile fragments 2w (G) and 2w+1 (U): rows t0 + 16w of both.
    sop(b, format!("s_lshl_b32 s{t1}, s{}, 4", Regs::WAVE), &[t1], &[Regs::WAVE])?;
    sop(b, format!("s_add_u32 s{t1}, s{t0}, s{t1}"), &[t1], &[t0, t1])?;
    mul_add64(b, Regs::SS[0], ARG_G, t1, Regs::ROWB)?;
    mul_add64(b, Regs::SS[1], ARG_U, t1, Regs::ROWB)?;
    // Scale words: gate (word 0) and up (word 1) rows t0 + 32*wr + lane part.
    sop(b, format!("s_lshl_b32 s{t1}, s{}, 5", Regs::WR), &[t1], &[Regs::WR])?;
    sop(b, format!("s_add_u32 s{t1}, s{t0}, s{t1}"), &[t1], &[t0, t1])?;
    mul_add64(b, Regs::SW0, ARG_G, t1, Regs::ROWB)?;
    mul_add64(b, Regs::SW1, ARG_U, t1, Regs::ROWB)?;
    // X base: Xq + (128*bx)*72.
    sop(b, format!("s_mul_i32 s{t1}, s{bx}, {}", lit(TILE_N * XBLK_BYTES)), &[t1], &[bx])?;
    sop(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::SX, ARG_XQ), &[Regs::SX], &[ARG_XQ, t1])?;
    sop(b, format!("s_addc_u32 s{}, s{}, 0", Regs::SX + 1, ARG_XQ + 1), &[Regs::SX + 1], &[ARG_XQ + 1])?;
    sop(b, format!("s_mul_i32 s{}, s{}, {}", Regs::N72, ARG_N, lit(XBLK_BYTES)), &[Regs::N72], &[ARG_N])?;
    sop(b, format!("s_lshl_b32 s{}, s{}, 6", Regs::M64, ARG_M), &[Regs::M64], &[ARG_M])?;
    // Y base: Y + 4*(128*bx*M + t0), 64-bit.
    let (lo, hi) = (Regs::MUL, Regs::MUL + 1);
    sop(b, format!("s_lshl_b32 s{t1}, s{bx}, 7"), &[t1], &[bx])?;
    sop(b, format!("s_mul_hi_u32 s{hi}, s{t1}, s{ARG_M}"), &[hi], &[t1, ARG_M])?;
    sop(b, format!("s_mul_i32 s{lo}, s{t1}, s{ARG_M}"), &[lo], &[t1, ARG_M])?;
    sop(b, format!("s_add_u32 s{lo}, s{lo}, s{t0}"), &[lo], &[lo, t0])?;
    sop(b, format!("s_addc_u32 s{hi}, s{hi}, 0"), &[hi], &[hi])?;
    op(b, format!("s_lshl_b64 s[{lo}:{hi}], s[{lo}:{hi}], 2"), &[sr(lo, 2)], &[sr(lo, 2)])?;
    sop(b, format!("s_add_u32 s{}, s{ARG_Y}, s{lo}", Regs::SY), &[Regs::SY], &[ARG_Y, lo])?;
    sop(b, format!("s_addc_u32 s{}, s{}, s{hi}", Regs::SY + 1, ARG_Y + 1), &[Regs::SY + 1], &[ARG_Y + 1, hi])?;

    // A staging: lane (hi, lr) loads row lr's K16 slices (hi, 2+hi) of the
    // K64 half at +8 (header) + 8*hi + 16*i.
    op(b, "v_lshl_add_u32 v6, v4, 3, 8", &[v(6)], &[v(4)])?;
    op(b, format!("v_mad_u32_u24 v{}, v3, s{}, v6", Regs::A_OFF, Regs::ROWB), &[v(Regs::A_OFF)], &[v(3), s(Regs::ROWB), v(6)])?;
    // X staging: token 16*wr + lr, K16 slice 2*wc + hi of the half:
    // (16*wr + lr)*72 + 8 + 16*wc + 8*hi.
    op(b, format!("v_lshl_add_u32 v5, s{}, 4, v3", Regs::WR), &[v(5)], &[s(Regs::WR), v(3)])?;
    op(b, format!("v_lshl_add_u32 v17, s{}, 4, v6", Regs::WC), &[v(17)], &[s(Regs::WC), v(6)])?;
    op(b, format!("v_mad_u32_u24 v{}, v5, {}, v17", Regs::X_OFF, lit(XBLK_BYTES)), &[v(Regs::X_OFF)], &[v(5), v(17)])?;
    // LDS store addresses: A wave*1024 + hi*128 + lr*8 (fragment block 2w,
    // slice hi; +512 for fragment 2w+1, +256 for slice 2+hi); X wave*256 +
    // hi*128 + lr*8 (+A_BYTES): token fragment w/2, slice 2*(w%2) + hi.
    op(b, "v_lshlrev_b32_e32 v7, 3, v3", &[v(7)], &[v(3)])?;
    op(b, "v_lshl_add_u32 v16, v4, 7, v7", &[v(16)], &[v(4), v(7)])?;
    op(b, format!("v_lshl_add_u32 v{}, s{}, 10, v16", Regs::ST, Regs::WAVE), &[v(Regs::ST)], &[s(Regs::WAVE), v(16)])?;
    op(b, format!("v_lshl_add_u32 v{}, s{}, 8, v16", Regs::STXA, Regs::WAVE), &[v(Regs::STXA)], &[s(Regs::WAVE), v(16)])?;
    // A fragment base: wr*2048 + prow*8 (+a*512 per fragment, +SLOT per
    // slot), prow = 8*(lr&1) + (lr>>1): A lane i supplies row
    // 8*(i&1)+(i>>1), so result VGPR j of lane (hi, lr) is row 8*hi + j.
    op(b, "v_and_b32_e32 v8, 1, v3", &[v(8)], &[v(3)])?;
    op(b, "v_lshrrev_b32_e32 v9, 1, v3", &[v(9)], &[v(3)])?;
    op(b, "v_lshlrev_b32_e32 v9, 3, v9", &[v(9)], &[v(9)])?;
    op(b, "v_lshl_add_u32 v8, v8, 6, v9", &[v(8)], &[v(8), v(9)])?;
    op(b, format!("v_lshl_add_u32 v{}, s{}, 11, v8", Regs::AB[0], Regs::WR), &[v(Regs::AB[0])], &[s(Regs::WR), v(8)])?;
    for k in 1..3usize {
        op(b, format!("v_add_nc_u32_e32 v{}, {}, v{}", Regs::AB[k], lit(SLOT_BASE[k]), Regs::AB[0]), &[v(Regs::AB[k])], &[v(Regs::AB[0])])?;
    }
    // X fragment bases: A_BYTES + wc*2048 + lr*8 (+c*512, +SLOT per slot).
    op(b, format!("v_lshl_add_u32 v10, s{}, 11, v7", Regs::WC), &[v(10)], &[s(Regs::WC), v(7)])?;
    for k in 0..3usize {
        op(b, format!("v_add_nc_u32_e32 v{}, {}, v10", Regs::XB[k], lit(SLOT_BASE[k] + A_BYTES)), &[v(Regs::XB[k])], &[v(10)])?;
    }
    // Scale row of lane (hi, lr), relative to the word bases:
    // (16*(lr>>3) + 8*hi + (lr&7))*row_bytes.
    op(b, "v_lshrrev_b32_e32 v12, 3, v3", &[v(12)], &[v(3)])?;
    op(b, "v_lshlrev_b32_e32 v12, 4, v12", &[v(12)], &[v(12)])?;
    op(b, "v_lshl_add_u32 v12, v4, 3, v12", &[v(12)], &[v(4), v(12)])?;
    op(b, "v_and_b32_e32 v13, 7, v3", &[v(13)], &[v(3)])?;
    op(b, "v_add_nc_u32_e32 v12, v12, v13", &[v(12)], &[v(12), v(13)])?;
    op(b, format!("v_mul_u32_u24_e32 v{}, s{}, v12", Regs::LWS, Regs::ROWB), &[v(Regs::LWS)], &[s(Regs::ROWB), v(12)])?;
    // d of token wc*64 + lr (+16c): (wc*64 + lr)*72.
    op(b, format!("v_lshl_add_u32 v14, s{}, 6, v3", Regs::WC), &[v(14)], &[s(Regs::WC), v(3)])?;
    op(b, format!("v_mul_u32_u24_e32 v{}, {}, v14", Regs::LXS, lit(XBLK_BYTES)), &[v(Regs::LXS)], &[v(14)])?;
    // Y offset of (token wc*64 + lr, h row wr*32 + 8*hi), bytes.
    op(b, "v_lshlrev_b32_e32 v15, 3, v4", &[v(15)], &[v(4)])?;
    op(b, format!("v_lshl_add_u32 v15, s{}, 5, v15", Regs::WR), &[v(15)], &[s(Regs::WR), v(15)])?;
    op(b, format!("v_mad_u32_u24 v15, v14, s{ARG_M}, v15"), &[v(15)], &[v(14), s(ARG_M), v(15)])?;
    op(b, format!("v_lshlrev_b32_e32 v{}, 2, v15", Regs::YOFF), &[v(Regs::YOFF)], &[v(15)])?;

    // Stage period 0 (both halves; ring slot 0 and the fixed slot), and seed
    // the chain constant and the sums while the loads are in flight.
    stage_loads(b, 0, 0)?;
    stage_loads(b, 1, 0)?;
    for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, {}", Regs::MAGIC + j, lit(MAGIC)), &[v(Regs::MAGIC + j)], &[])?; }
    for r in 0..128u8 { op(b, format!("v_mov_b32_e32 v{}, 0", Regs::ACC + r), &[v(Regs::ACC + r)], &[])?; }
    let (a0, x0) = (wg.begin_write(ra), wg.begin_write(rx));
    let ((ra, pa), (rx, px)) = stage_store_ring(wg, a0, x0)?;
    let ((ah, pah), (xh, pxh)) = stage_store_fixed(wg, ah, xh)?;
    let (da, dx, dah, dxh) = wg.wait_all((pa, px, pah, pxh))?;
    let (ra, rx, ah, xh) = wg.barrier((prime(ra, da), prime(rx, dx), ready(ah, dah), ready(xh, dxh)))?;
    step_loads(wg, &ra, &rx, Some((&ah, &xh)), 0)?;
    head(wg.isa(), 0, true)?;
    Ok((ra, rx, ah, xh))
}

fn global_load_b64(b: &mut Builder, dst: u8, voff: u8, base: u8, offset: u32) -> Result<(), String> {
    mem(b, format!("global_load_b64 {}, v{voff}, s[{base}:{}]{}", vr(dst, 2), base + 1, off(offset)?), &[vr(dst, 2)], &[v(voff), sr(base, 2)], MemoryClass::VmemLoad)
}
fn global_load_b32(b: &mut Builder, dst: u8, voff: u8, base: u8, offset: u32) -> Result<(), String> {
    mem(b, format!("global_load_b32 v{dst}, v{voff}, s[{base}:{}]{}", base + 1, off(offset)?), &[v(dst)], &[v(voff), sr(base, 2)], MemoryClass::VmemLoad)
}

/// Global loads of K64 half `h` of a period into staging packet `h`;
/// `a_imm` (0 or 64) selects the K128 half inside the current A group.
fn stage_loads(b: &mut Builder, h: usize, a_imm: u32) -> Result<(), String> {
    let half = 32 * h as u32;
    b.clause(|b| {
        for t in 0..2usize { for i in 0..2u8 {
            global_load_b64(b, Regs::STA[h] + 4 * t as u8 + 2 * i, Regs::A_OFF, Regs::SS[t], a_imm + half + 16 * u32::from(i))?;
        } }
        Ok(())
    })?;
    global_load_b64(b, Regs::STX[h], Regs::X_OFF, Regs::SX, half)
}

/// Rebias staged A packet `h` and build its LDS stores into slot `slot`:
/// four A stores (G slices hi, 2+hi; U slices hi, 2+hi) and one X store.
fn packet_stores(b: &mut Builder, h: usize, slot: usize) -> Result<(Vec<Instruction>, Instruction), String> {
    for r in 0..8u8 {
        let x = Regs::STA[h] + r;
        op(b, format!("v_xor_b32_e32 v{x}, {}, v{x}", lit(REBIAS)), &[v(x)], &[v(x)])?;
    }
    let store = |addr: u8, data: u8, o: u32| {
        let text = format!("ds_store_b64 v{addr}, {}{}", vr(data, 2), if o == 0 { String::new() } else { format!(" offset:{o}") });
        Instruction::new(text, vec![], vec![v(addr), vr(data, 2)]).memory(MemoryClass::DsStore)
    };
    let a = (0..2u32).flat_map(|t| (0..2u32).map(move |i| (t, i)))
        .map(|(t, i)| store(Regs::ST, Regs::STA[h] + (4 * t + 2 * i) as u8, SLOT_BASE[slot] + 512 * t + 256 * i)).collect();
    Ok((a, store(Regs::STXA, Regs::STX[h], SLOT_BASE[slot] + A_BYTES)))
}

/// Publish staged packet 0 (a period's first half) into the ring's next slot.
fn stage_store_ring<C: peacemaker_author::State>(w: &mut Wv, a: (Ring<A, C, Writing>, Pending<Gfx11Waits, LdsWrite<A>>),
    x: (Ring<X, C, Writing>, Pending<Gfx11Waits, LdsWrite<X>>))
    -> Result<((Ring<A, C, Writing>, Pending<Gfx11Waits, LdsWrite<A>>), (Ring<X, C, Writing>, Pending<Gfx11Waits, LdsWrite<X>>)), String> {
    let slot = a.0.next_index();
    let (sa, sx) = packet_stores(w.isa(), 0, slot)?;
    let mut a = a;
    for i in sa { a = w.ds_store(a, i)?; }
    let x = w.ds_store(x, sx)?;
    Ok((a, x))
}

/// Publish staged packet 1 (a period's second half) into the fixed slot.
fn stage_store_fixed(w: &mut Wv, ah: LdsRegion<AH, Free>, xh: LdsRegion<XH, Free>) -> Result<(StagedH<AH>, StagedH<XH>), String> {
    let (sa, sx) = packet_stores(w.isa(), 1, 2)?;
    let mut a = w.begin_write(ah);
    for i in sa { a = w.ds_store(a, i)?; }
    let x = w.ds_store(xh, sx)?;
    Ok((a, x))
}

/// K64 half and slice (0..3) that period step `i = 8a + pos` reads: passes
/// 0..2 read the first half then the second, pass 3 the second half first
/// so the fixed slot is free after step `B1_AFTER_STEP`.
fn source(i: usize) -> (usize, u32) {
    let (a, pos) = (i / 8, i % 8);
    let first = usize::from(a == 3);
    (if pos < 4 { first } else { 1 - first }, (pos % 4) as u32)
}

/// Fragment loads of step `i`: the X fragments of its slice (c = 0,1 and
/// c = 2,3) and, at even slices, the A pair (s, s+1) of pass `a`, from the
/// ring's current slot or the fixed slot (`None` once B1 retired it).
type FixedRef<'r> = Option<(&'r LdsRegion<AH, Published>, &'r LdsRegion<XH, Published>)>;
fn step_loads<NA: peacemaker_author::State, NX: peacemaker_author::State>(w: &mut Wv, ra: &Ring<A, Published, NA>, rx: &Ring<X, Published, NX>,
    fixed: FixedRef, i: usize) -> Result<(), String> {
    let (h, s_) = source(i);
    let a = (i / 8) as u32;
    let fixed = match (h, fixed) {
        (0, _) => None,
        (_, Some(f)) => Some(f),
        (_, None) => return Err(format!("step {i} reads the fixed slot after B1 retired it")),
    };
    let slot = if fixed.is_some() { 2 } else { ra.cur_index() };
    if s_ % 2 == 0 {
        let dst = Regs::AV[(i / 2) % 2];
        let o = 64 * a + 16 * s_;
        let insn = Instruction::new(format!("ds_load_2addr_b64 {}, v{}{}", vr(dst, 4), Regs::AB[slot], ds_offsets(o, o + 16)), vec![vr(dst, 4)], vec![v(Regs::AB[slot])]).memory(MemoryClass::DsLoad);
        match fixed { Some((ah, _)) => w.ds_load(ah, insn)?, None => w.ds_load_cur(ra, insn)? }
    }
    for half in 0..2u32 {
        let dst = Regs::XV[i % 2] + 4 * half as u8;
        let o = 128 * half + 16 * s_;
        let insn = Instruction::new(format!("ds_load_2addr_b64 {}, v{}{}", vr(dst, 4), Regs::XB[slot], ds_offsets(o, o + 64)), vec![vr(dst, 4)], vec![v(Regs::XB[slot])]).memory(MemoryClass::DsLoad);
        match fixed { Some((_, xh)) => w.ds_load(xh, insn)?, None => w.ds_load_cur(rx, insn)? }
    }
    Ok(())
}

fn step_wmma<T: MmaIu4>(w: &mut Wave<T, Builder>, i: usize) -> Result<(), String> {
    Chain::<4, Iu4>::new(std::array::from_fn(|c| V::<8>(Regs::C + 8 * c as u8)), V::<8>(Regs::MAGIC))?
        .step(w, V::<2>(Regs::AV[(i / 2) % 2] + 2 * (i % 2) as u8),
            std::array::from_fn(|c| V::<2>(Regs::XV[i % 2] + 2 * c as u8)), i % 8 == 0)
}

/// Scale broadcast of fold pass `a` on the LDS crossbar (V2B's `scales`):
/// row 16a + 8hi + j's scale is word `a&1` of lane 8*(a>>1) + j of this
/// lane's row of 16.
fn scales(b: &mut Builder, a: u8) -> Result<(), String> {
    let w = usize::from(a & 1);
    if a < 2 {
        // True16 VOP1 reaches only v0-v127 halves; v221/v222 need the VOP3 form.
        op(b, format!("v_cvt_f32_f16_e64 v{}, v{}.l", Regs::WSF[w], Regs::WS[w]), &[v(Regs::WSF[w])], &[v(Regs::WS[w])])?;
    }
    for j in 0..8u8 {
        let text = format!("ds_swizzle_b32 v{}, v{} offset:swizzle(BROADCAST,16,{})", Regs::SCF + j, Regs::WSF[w], 8 * (a >> 1) + j);
        b.ds_crosslane(Instruction::new(text, vec![v(Regs::SCF + j)], vec![v(Regs::WSF[w])]).memory(MemoryClass::DsLoad))?;
    }
    Ok(())
}

/// Fold pass `a`: sum[a][c][j] = fma(d_c * sc_j, C[c][j] - 1.5*2^23, sum).
fn fold(b: &mut Builder, a: u8, dc: u8) -> Result<(), String> {
    Chain::<4, Iu4>::new(std::array::from_fn(|c| V::<8>(Regs::C + 8 * c as u8)), V::<8>(Regs::MAGIC))?
        .fold(b, std::array::from_fn(|c| Regs::ACC + 8 * (4 * a + c as u8)),
            Regs::SCF, std::array::from_fn(|c| dc + c as u8), Regs::T)
}

/// Period `e`'s head (parity p): its metadata (scale words, token d) and,
/// with `stage`, the global loads of both K64 halves of period e+1.
fn head(b: &mut Builder, p: usize, stage: bool) -> Result<(), String> {
    // Metadata first: in-order VMcnt returns it before the packets.
    for (w, base) in [Regs::SW0, Regs::SW1].into_iter().enumerate() {
        mem(b, format!("global_load_u16 v{}, v{}, s[{base}:{}]{}", Regs::WS[w], Regs::LWS, base + 1, off(4 * p as u32)?),
            &[v(Regs::WS[w])], &[v(Regs::LWS), sr(base, 2)], MemoryClass::VmemLoad)?;
    }
    for c in 0..4u8 { global_load_b32(b, Regs::DC[p] + c, Regs::LXS, Regs::SX, 16 * XBLK_BYTES * u32::from(c))?; }
    if stage {
        if p == 1 {
            // Period e+1 starts the next 136-byte A group.
            for base in [Regs::SS[0], Regs::SS[1], Regs::SW0, Regs::SW1] { add64_imm(b, base, GROUP_BYTES)?; }
        }
        add64(b, Regs::SX, Regs::SX, Regs::N72)?;
        let a_imm = if p == 0 { 64 } else { 0 };
        stage_loads(b, 0, a_imm)?;
        stage_loads(b, 1, a_imm)?;
    }
    Ok(())
}

/// The ring side of a period: still reading only, or with the next first
/// half published into the next slot (stores pending).
enum Ring0 { Reading(Ring<A, Published, Free>, Ring<X, Published, Free>), Publishing(Staged<A>, Staged<X>) }
/// The fixed slot: read by this period, retired by B1, then written with
/// the next second half.
enum Fixed { Published(LdsRegion<AH, Published>, LdsRegion<XH, Published>), Writing(StagedH<AH>, StagedH<XH>) }

fn loads(w: &mut Wv, r: &Ring0, f: &Fixed, i: usize) -> Result<(), String> {
    let fixed = match f { Fixed::Published(ah, xh) => Some((ah, xh)), Fixed::Writing(..) => None };
    match r {
        Ring0::Reading(ra, rx) => step_loads(w, ra, rx, fixed, i),
        Ring0::Publishing((ra, _), (rx, _)) => step_loads(w, ra, rx, fixed, i),
    }
}

/// One K128 period reading ring slot `p` and the fixed slot. With `next`,
/// the staged first half is published into ring slot 1-p after pass
/// `PUBLISH_AFTER_PASS`, B1 retires the fixed slot after step
/// `B1_AFTER_STEP` and takes the staged second half, and B2 at the period
/// end rotates the ring and publishes the fixed slot; the last period has
/// neither. `succ_stages`: period e+1 has a successor too (its head stages
/// e+2). Pass 0's scale broadcast issues after WMMA step `SCALES0_STEP`,
/// pass a+1's right after fold pass a released the broadcast registers.
fn period(wg: &mut Wg, rings: Rings, next: bool, succ_stages: bool) -> Result<Rings, String> {
    let (ra, rx, ah, xh) = rings;
    let p = ra.cur_index();
    let mut r = Ring0::Reading(ra, rx);
    let mut f = Fixed::Published(ah, xh);
    for i in 0..32 {
        if i + 1 < 32 { loads(wg, &r, &f, i + 1)?; }
        step_wmma(wg, i)?;
        let b = wg.isa();
        if i == SCALES0_STEP { scales(b, 0)?; }
        if i % 8 == 7 && i < 31 {
            fold(b, (i / 8) as u8, Regs::DC[p])?;
            scales(b, (i / 8) as u8 + 1)?;
            if next && i / 8 == PUBLISH_AFTER_PASS {
                r = match r {
                    Ring0::Reading(ra, rx) => {
                        let (a, x) = (wg.begin_write(ra), wg.begin_write(rx));
                        let (a, x) = stage_store_ring(wg, a, x)?;
                        Ring0::Publishing(a, x)
                    }
                    Ring0::Publishing(..) => return Err("period publishes its first half twice".into()),
                };
            }
        }
        if next && i == B1_AFTER_STEP {
            f = match f {
                Fixed::Published(ah, xh) => {
                    let (ah, xh) = wg.barrier((retire(ah), retire(xh)))?;
                    let (a, x) = stage_store_fixed(wg, ah, xh)?;
                    Fixed::Writing(a, x)
                }
                Fixed::Writing(..) => return Err("period retires the fixed slot twice".into()),
            };
        }
    }
    let rings = match (r, f, next) {
        (Ring0::Publishing((ra, pa), (rx, px)), Fixed::Writing((ah, pah), (xh, pxh)), true) => {
            let (da, dx, dah, dxh) = wg.wait_all((pa, px, pah, pxh))?;
            let (ra, rx, ah, xh) = wg.barrier((rotate(ra, da), rotate(rx, dx), ready(ah, dah), ready(xh, dxh)))?;
            step_loads(wg, &ra, &rx, Some((&ah, &xh)), 0)?;
            head(wg.isa(), 1 - p, succ_stages)?;
            (ra, rx, ah, xh)
        }
        (Ring0::Reading(ra, rx), Fixed::Published(ah, xh), false) => (ra, rx, ah, xh),
        _ => return Err("period publication does not match its successor".into()),
    };
    fold(wg.isa(), 3, Regs::DC[p])?;
    Ok(rings)
}

/// The trip counter is `K`-derived: workgroup-uniform.
fn trips_cmp(wg: &mut Wg, cmp: &str) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(format!("{cmp} s{}, 0", Regs::TRIPS), vec![], vec![s(Regs::TRIPS)]))
}

/// The counted loop of two-period trips over `Regs::TRIPS` (skipped at zero
/// trips), then the tail pair.
fn kloop(wg: &mut Wg, g: &Gen, rings: Rings) -> Result<Rings, String> {
    wg.label(&g.label("k_begin"))?;
    let (head, end) = (g.label("k_loop"), g.label("k_loop_end"));
    // What a trip boundary leaves in flight (the next period's step-0 loads
    // and its head) is in flight on entry, on every back-edge (the
    // builder's loop fixpoint checks it) and on exit.
    let entry = wg.isa().ledger.shape();
    let no_trip = trips_cmp(wg, "s_cmp_eq_u32")?;
    let rings = wg.wg_skip_if(no_trip, &end, rings, |wg, rings| {
        let rings = wg.loop_carried(&head, rings, |wg, rings| {
            let rings = period(wg, rings, true, true)?;
            let rings = period(wg, rings, true, true)?;
            op(wg.isa(), format!("s_add_i32 s{0}, s{0}, -1", Regs::TRIPS), &[s(Regs::TRIPS)], &[s(Regs::TRIPS)])?;
            Ok((rings, trips_cmp(wg, "s_cmp_lg_u32")?))
        })?;
        if wg.isa().ledger.shape() != entry { return Err(format!("{head} exit ledger differs from its entry")) }
        Ok(rings)
    })?;
    wg.label(&g.label("tail"))?;
    let rings = period(wg, rings, true, false)?;
    period(wg, rings, false, false)
}

fn acc(a: u8, c: u8) -> u8 { Regs::ACC + 8 * (4 * a + c) }

/// V2B's SiLU epilogue: lane (hi, lr) owns h rows 8hi..8hi+7 of each h
/// fragment for one token; token fragment c is 16*M*4 bytes further
/// (v0..v3 = its Y offsets).
fn epilogue(b: &mut Builder, g: &Gen) -> Result<(), String> {
    b.label(&g.label("epilogue"))?;
    op(b, format!("v_mov_b32_e32 v0, v{}", Regs::YOFF), &[v(0)], &[v(Regs::YOFF)])?;
    for c in 1..4u8 {
        op(b, format!("v_add_nc_u32_e32 v{c}, s{}, v{}", Regs::M64, c - 1), &[v(c)], &[s(Regs::M64), v(c - 1)])?;
    }
    b.label(&g.label("epi_body"))?;
    // h = SILU_MUL(g, u) for gate fragment 2p and up fragment 2p+1,
    // written over the gate sums and stored from there.
    for p in 0..2u8 { for c in 0..4u8 {
        Epilogue::silu_dense(b, acc(2 * p, c), acc(2 * p + 1, c), 160, Regs::MASK, SILU_GROUP)?;
        for q in 0..2u8 {
            let data = vr(acc(2 * p, c) + 4 * q, 4);
            mem(b, format!("global_store_b128 v{c}, {data}, s[{}:{}]{}", Regs::SY, Regs::SY + 1, off(4 * (16 * u32::from(p) + 4 * u32::from(q)))?),
                &[], &[v(c), data, sr(Regs::SY, 2)], MemoryClass::VmemStore)?;
        }
    } }
    Ok(())
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let g = Gen { spec };
    let kspec = KernelSpec {
        kernel_id: "iu4_v2b_a4".into(), variant: spec.epi.tag().into(), arch: spec.arch, symbol: spec.symbol(),
        kernargs: spec.kernargs(), user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
        workgroup_size: THREADS, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, g.plan()?);
    b.enable_delay_alu();
    let mut wg = Wg::new(&mut b)?;
    let lds = declare_lds(&mut wg)?;
    let end = wg.exit(&g.label("end"))?;
    let rings = prologue(&mut wg, lds)?;
    // The last period's slots stay published: nothing writes LDS after it.
    let _published = kloop(&mut wg, &g, rings)?;
    epilogue(wg.isa(), &g)?;
    // One CTA per WGP: release the VGPRs before the epilogue's stores drain
    // so the next CTA's waves launch (V2B and hipcc emit the same message).
    // M7 models `s_sendmsg` as reading M0; the message carries no data.
    wg.end_with(end, |w| {
        op(w.isa(), "s_mov_b32 m0, 0", &[], &[])?;
        w.isa().push(Sop::Dealloc.encode(spec.arch)?)
    })?;
    b.finish()
}

/// The module's entries as one code object (stage 1: the M512 SiLU twin).
pub fn emit_module(arch: Arch) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = Epi::ALL.into_iter().map(|epi| emit(Spec { arch, epi })).collect::<Result<Vec<_>, _>>()?;
    let (text, proof) = super::iu4_gemm::module(&emitted, MODULE)?;
    Ok((emitted, text, proof))
}

/// Instruction census of one steady-state trip (two K128 periods, four K64
/// halves) of the K loop of `epi`.
pub fn hot_loop_census(s_text: &str, epi: Epi) -> std::collections::BTreeMap<String, u32> {
    let (head, end) = (format!(".Lv2b_a4_{}_k_loop:", epi.tag()), format!(".Lv2b_a4_{}_k_loop_end:", epi.tag()));
    let mut counts = std::collections::BTreeMap::new();
    let mut inside = false;
    for line in s_text.lines() {
        let t = line.trim();
        if t == head { inside = true; continue }
        if t == end { break }
        if !inside || t.ends_with(':') || t.is_empty() || t.starts_with('.') { continue }
        let name = t.split_whitespace().next().unwrap_or("").to_owned();
        *counts.entry(name.clone()).or_insert(0) += 1;
        if name.starts_with("v_dual_") { *counts.entry("vopd_packets".into()).or_insert(0) += 1 }
        if name.starts_with("v_") && !name.starts_with("v_wmma_") { *counts.entry("valu_slots".into()).or_insert(0) += 1 }
    }
    counts
}
