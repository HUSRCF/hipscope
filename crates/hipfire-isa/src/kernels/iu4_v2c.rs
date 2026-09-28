//! Builder-emitted gfx11 (RDNA3, wave32) MQ4V2 x block_i4_128 GEMM with the
//! hipcc V2C algorithm (`kernels/src/gemm_mq4g256v2_residual_iu4_v2c.gfx11.hip`),
//! bit for bit, for its three epilogue families:
//! SET (`gemm_mq4g256v2_residual_iu4_v2c_set_gfx11`), ADD (`_add_touch`) and
//! the F1-lite gate/up SiLU entry
//! (`gemm_mq4g256v2_gate_up_silu_iu4_v2c_gfx11`).
//!
//! Tile M128 x N128, 8 waves; wave `w` owns rows `32*(w/2)..+32` (two 16-row
//! fragments `a`) and tokens `64*(w%2)..+64` (four 16-token fragments `c`).
//! K is walked in K128 epochs with two 16 KiB LDS slots (A 8 KiB, X 8 KiB,
//! fragment-major: K16 slice `s` of row/token `r` at `((r/16)*8+s)*128 +
//! (r%16)*8`). Epoch `e` reads slot `e%2` while the staging registers carry
//! epoch `e+1` into slot `(e+1)%2`; one workgroup barrier per epoch.
//!
//! Numerics, per output and ascending epoch `e`: `C_e` is the exact int32
//! `v_wmma_i32_16x16x16_iu4` chain over the eight K16 slices seeded with the
//! magic `0x4b400000`; `sum = fma(RN(d_e * sc_e), C_e + (-12582912.0), sum)`
//! from `sum = +0`, as hipcc compiles V2C (`v_mul_f32`, `v_add_f32`
//! literal, `v_fmac_f32`). A is rebiased once at staging (XOR 0x88888888,
//! the symmetric `zp == -8*sc` contract); both operands are signed nibbles.
//! SET stores `sum`, ADD stores `RN(Y + sum)`, and gate/up stores
//! `h = SILU_MUL(g, u)` through the imported hipcc region
//! (`kernels/iu4_v2c.gfx1100.silu.region.s`).
//!
//! Gate/up (F1-lite): the CTA covers 128 virtual rows = 64 h rows; tile
//! fragment `2p` is gate rows and `2p+1` the same up rows (staging wave `w`
//! loads gate (`w` even) or up (`w` odd) rows `64*wgy + 16*(w/2) + 0..15`).
//! Each lane loads both the gate and the up FP16 scale of its row; fold
//! pass 0 shares gate scales and pass 1 up scales, so the fold (and its
//! full VOPD pairing) is SET's plus one conversion.
//!
//! Launch contract (as V2C): grid `[N/128, M/128]` (gate/up: `[N/128,
//! 2M/128]`), block `[256,1,1]`, dynamic LDS 32768 bytes; `M % 128 == N % 128
//! == 0`, `K % 256 == 0`, `M * 4 < 2^24`; kernargs `A, Xq, Y, M, K, N`
//! (gate/up: `G, U, Xq, H, M, K, N` with `M` the gate row count), Y/H
//! token-major `[N][M]` f32.
use super::iu4_gemm::{ds_offsets, lit, region::{self, Binding, Region}, s, sr, v, vr};
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan, V,
    insn::{Instruction, MemoryClass, Sop, Wmma}, lds::Transition,
    reg::{Live, RegRef}, vopd::{Operand, VopdF32, VopdOp}};

pub const THREADS: u16 = 256;
pub const LDS_BYTES: u32 = 32768;
pub const SLOT_BYTES: u32 = 16384;
pub const A_BYTES: u32 = 8192;
pub const VGPR_CEILING: u16 = 192;
pub const MAGIC: u32 = 0x4b40_0000;
/// `-12582912.0f`: `float(C) = bits(C + magic) - 1.5*2^23` exactly.
pub const MAGIC_NEG: u32 = 0xcb40_0000;
pub const REBIAS: u32 = 0x8888_8888;
pub const GROUP_BYTES: u32 = 136;
pub const XBLK_BYTES: u32 = 72;

pub const K_BEGIN: &str = ".Lv2c_k_begin";
pub const K_LOOP: &str = ".Lv2c_k_loop";
pub const K_LOOP_END: &str = ".Lv2c_k_loop_end";
pub const TAIL: &str = ".Lv2c_tail";
pub const EPI: &str = ".Lv2c_epilogue";
pub const END: &str = ".Lv2c_end";

/// Epilogue family. ADD touches the tile's residual over epochs E-16..E-13
/// (hipcc `_add_touch`'s window) so its epilogue loads hit cache; the
/// output is RN(Y + sum) either way.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Epi { Set, Add, Silu }
impl Epi {
    pub const ALL: [Epi; 3] = [Epi::Set, Epi::Add, Epi::Silu];
    pub fn name(self) -> &'static str { match self { Self::Set => "set", Self::Add => "add", Self::Silu => "silu" } }
    fn silu(self) -> bool { self == Self::Silu }
}
impl std::str::FromStr for Epi {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        Self::ALL.into_iter().find(|e| e.name() == s).ok_or_else(|| format!("iu4_v2c epilogue {s}: expected set|add|silu"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Spec { pub arch: Arch, pub epi: Epi }

impl Spec {
    pub fn symbol(self) -> String {
        let arch = self.arch.name();
        match self.epi {
            Epi::Silu => format!("gemm_mq4g256v2_gate_up_silu_iu4_pm_{arch}"),
            e => format!("gemm_mq4g256v2_residual_iu4_pm_{}_{arch}", e.name()),
        }
    }
    pub fn validate(self) -> Result<(), String> {
        if self.arch != Arch::Gfx1100 { return Err("iu4_v2c: the V2C tile is proven on gfx1100 only (gfx1151 ships V2B)".into()) }
        Ok(())
    }
    pub fn kernargs(self) -> KernargLayout {
        if self.epi.silu() {
            return KernargLayout::new(44).pointer("G", 0).pointer("U", 8).pointer("Xq", 16).pointer("Y", 24)
                .hidden("M", 32, 4, "by_value").hidden("K", 36, 4, "by_value").hidden("N", 40, 4, "by_value");
        }
        KernargLayout::new(36).pointer("A", 0).pointer("Xq", 8).pointer("Y", 16)
            .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("N", 32, 4, "by_value")
    }
}

/// Kernel-argument SGPRs: SET/ADD `s_load_b256 s[8:15]` (A, Xq, Y, M, K) +
/// `s_load_b32 s16` (N); gate/up `s_load_b256 s[8:15]` (G, U, Xq, Y) +
/// `s_load_b64 s[30:31]` (M, K) + `s_load_b32 s16` (N).
#[derive(Clone, Copy)]
struct Args { a: u8, xq: u8, y: u8, m: u8, k: u8, n: u8 }
fn args(epi: Epi) -> Args {
    if epi.silu() { Args { a: 8, xq: 12, y: 14, m: 30, k: 31, n: 16 } }
    else { Args { a: Regs::A, xq: Regs::XQ, y: Regs::Y, m: Regs::M, k: Regs::K, n: Regs::N } }
}

/// Physical registers (see `plan`).
struct Regs;
impl Regs {
    // VGPRs
    const C: u8 = 0;          // int32 WMMA chains C[c] = v[8c..8c+7]
    const ACC: u8 = 32;       // f32 sums acc[a][c] = v[32 + 8*(4a+c) ..]
    const MAGIC: u8 = 96;     // 8 x 0x4b400000, the chain seed
    const AV: [u8; 2] = [104, 108];  // A fragment pairs (slice s, s+1)
    const XV: [u8; 2] = [112, 120];  // X fragments c=0..3 of one slice
    const SCF: u8 = 128;      // row scales of the fold pass (DPP row_share)
    const T: [u8; 2] = [136, 184];   // fold products t = d*sc, two sets
    const DC: u8 = 144;       // d of the four token fragments
    const WSF: u8 = 148;      // this lane's f32 row scale
    const WS: u8 = 149;       // this lane's raw f16 row scale
    const STA: u8 = 150;      // staged A (4 x b64)
    const STX: u8 = 158;      // staged X (4 x b64)
    const A_OFF: u8 = 166; const X_OFF: u8 = 167; const LWS: u8 = 168; const LXS: u8 = 169;
    const ST_A: [u8; 2] = [170, 172]; const ST_X: [u8; 2] = [171, 173];
    const AB: [u8; 2] = [174, 175];
    const XB: [[u8; 2]; 2] = [[176, 177], [178, 179]];
    const YOFF: u8 = 180;
    // Gate/up: this lane's raw / f32 up-row scale.
    const WSU: u8 = 181; const WSFU: u8 = 182;
    // ADD touch: this thread's residual line offset and the touch sink
    // (v182 stays defined: M7 reads a `saddr` offset as a VGPR pair).
    const TOFF: u8 = 181; const TPAIR: u8 = 182; const TSINK: u8 = 183;
    // Epilogue (every K-loop register is dead): ADD residual quads
    // v104..v167; gate/up h octets v104..v135 and four slots of nine SiLU
    // temporaries, v136.. and v145.. (VOPD partner slots 0/1) and v4.. and
    // v13.. (slots 2/3), clear of every LDS address register (v170..v179,
    // which the all-lane LDS bound requires unchanged after the prologue).
    const RESID: u8 = 104; const H: u8 = 104; const SILU_T: [u8; 4] = [136, 145, 4, 13];
    // SGPRs
    const KARG: u8 = 0; const WGX: u8 = 2; const WGY: u8 = 3;
    const T0: u8 = 4; const T1: u8 = 5; const T64: u8 = 6;
    const ARGS: u8 = 8; // A s[8:9], Xq s[10:11], Y s[12:13], M s14, K s15
    const N: u8 = 16; const ROWB: u8 = 17; const WAVE: u8 = 18; const TRIPS: u8 = 19;
    const SA: u8 = 20; const SX: u8 = 22; const SY: u8 = 24;
    const N72: u8 = 26; const M64: u8 = 27; const WR: u8 = 28; const WC: u8 = 29;
    const A: u8 = 8; const XQ: u8 = 10; const Y: u8 = 12; const M: u8 = 14; const K: u8 = 15;
    // Gate/up: M, K s[30:31]; staging base (gate or up rows of this wave)
    // s[32:33]; the up-row scale base s[34:35] (SA holds the gate-row one);
    // SiLU lane masks s36, s38, .., s50 (one aligned pair each).
    const MK: u8 = 30; const SS: u8 = 32; const SU: u8 = 34; const MASKS: u8 = 36;
    // ADD touch: M*128 (32 tokens of residual) and the touched line's base.
    const M128: u8 = 30; const STB: u8 = 32;
}

/// SiLU region slots (instances live at once) and their temporaries.
const SILU_SLOTS: u8 = 4;
const SILU_TEMPS: u8 = 9;

/// Staging A base: the tile's rows (SET/ADD) or this wave's gate/up rows.
fn stage_base(epi: Epi) -> u8 { if epi.silu() { Regs::SS } else { Regs::SA } }

const SLOT_A: [usize; 2] = [0, 2];
const SLOT_X: [usize; 2] = [1, 3];

pub(super) fn op(b: &mut Builder, text: impl Into<String>, defs: &[RegRef], uses: &[RegRef]) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()))
}
pub(super) fn mem(b: &mut Builder, text: impl Into<String>, defs: &[RegRef], uses: &[RegRef], class: MemoryClass) -> Result<(), String> {
    b.push(Instruction::new(text, defs.to_vec(), uses.to_vec()).memory(class))
}
pub(super) fn off(o: u32) -> Result<String, String> {
    if o > 4095 { return Err(format!("global offset {o} exceeds the gfx11 13-bit signed field")) }
    Ok(if o == 0 { String::new() } else { format!(" offset:{o}") })
}
fn global_load(b: &mut Builder, width: u8, dst: u8, voff: u8, base: u8, offset: u32) -> Result<(), String> {
    let (name, reg) = match width { 1 => ("global_load_b32", v(dst)), 2 => ("global_load_b64", vr(dst, 2)), _ => return Err("load width".into()) };
    mem(b, format!("{name} {reg}, v{voff}, s[{base}:{}]{}", base + 1, off(offset)?), &[reg], &[v(voff), sr(base, 2)], MemoryClass::VmemLoad)
}

fn plan(epi: Epi) -> Result<RegPlan, String> {
    let mut p = RegPlan::new(VGPR_CEILING, 104)?;
    let kl = || Live::Between(K_BEGIN.into(), EPI.into());
    let kernel = || Live::Between("entry".into(), EPI.into());
    let pro = || Live::Between("entry".into(), K_BEGIN.into());
    let epl = || Live::Between(EPI.into(), END.into());
    for c in 0..4u8 { p.v::<8>("C", Regs::C + 8 * c, kl())?; }
    for i in 0..32u8 { p.v::<1>(if i == 0 { "tid" } else { "prologue_tmp" }, Regs::C + i, pro())?; }
    for i in 0..4u8 { p.v::<1>("y_off_c", Regs::C + i, epl())?; }
    for k in 0..8u8 { p.v::<8>("acc", Regs::ACC + 8 * k, Live::Whole)?; }
    p.v::<8>("magic8", Regs::MAGIC, kernel())?;
    for r in Regs::AV { p.v::<4>("a_frag_pair", r, kl())?; }
    for r in Regs::XV { p.v::<8>("x_frags", r, kl())?; }
    p.v::<8>("scale_rows", Regs::SCF, kl())?;
    for r in Regs::T { p.v::<8>("fold_t", r, kl())?; }
    p.v::<4>("d_x", Regs::DC, kl())?;
    p.v::<1>("scale_f32", Regs::WSF, kl())?;
    p.v::<1>("scale_f16", Regs::WS, kl())?;
    for i in 0..4u8 { p.v::<2>("stage_a", Regs::STA + 2 * i, kernel())?; p.v::<2>("stage_x", Regs::STX + 2 * i, kernel())?; }
    for (name, r) in [("a_off", Regs::A_OFF), ("x_off", Regs::X_OFF), ("lws", Regs::LWS), ("lxs", Regs::LXS),
        ("st_a0", Regs::ST_A[0]), ("st_a1", Regs::ST_A[1]), ("st_x0", Regs::ST_X[0]), ("st_x1", Regs::ST_X[1]),
        ("ab0", Regs::AB[0]), ("ab1", Regs::AB[1]), ("xb00", Regs::XB[0][0]), ("xb01", Regs::XB[0][1]),
        ("xb10", Regs::XB[1][0]), ("xb11", Regs::XB[1][1])] {
        p.v::<1>(name, r, kernel())?;
    }
    p.v::<1>("y_off", Regs::YOFF, Live::Whole)?;
    p.s::<2>("kernarg_ptr", Regs::KARG, Live::Whole)?;
    p.s::<1>("wg_x", Regs::WGX, Live::Whole)?;
    p.s::<1>("wg_y", Regs::WGY, Live::Whole)?;
    p.s::<1>("s_tmp0", Regs::T0, Live::Whole)?;
    p.s::<1>("s_tmp1", Regs::T1, Live::Whole)?;
    p.s::<2>("s_tmp64", Regs::T64, Live::Whole)?;
    p.s::<8>("kernargs_0x00", Regs::ARGS, Live::Whole)?;
    for (name, r) in [("n", Regs::N), ("row_bytes", Regs::ROWB), ("wave", Regs::WAVE), ("trips", Regs::TRIPS),
        ("n72", Regs::N72), ("m64", Regs::M64), ("wr", Regs::WR), ("wc", Regs::WC)] {
        p.s::<1>(name, r, Live::Whole)?;
    }
    p.s::<2>("a_base", Regs::SA, Live::Whole)?;
    p.s::<2>("x_base", Regs::SX, Live::Whole)?;
    p.s::<2>("y_base", Regs::SY, Live::Whole)?;
    match epi {
        Epi::Set => {}
        Epi::Add => {
            for q in 0..16u8 { p.v::<4>("residual", Regs::RESID + 4 * q, epl())?; }
            for (name, r) in [("touch_off", Regs::TOFF), ("touch_pair", Regs::TPAIR), ("touch_sink", Regs::TSINK)] { p.v::<1>(name, r, Live::Whole)?; }
            p.s::<1>("m128", Regs::M128, Live::Whole)?;
            p.s::<2>("touch_base", Regs::STB, Live::Whole)?;
        }
        Epi::Silu => {
            p.v::<1>("scale_f16_up", Regs::WSU, kl())?;
            p.v::<1>("scale_f32_up", Regs::WSFU, kl())?;
            for c in 0..4u8 { p.v::<8>("h", Regs::H + 8 * c, epl())?; }
            for base in Regs::SILU_T { for t in 0..SILU_TEMPS { p.v::<1>("silu_tmp", base + t, epl())?; } }
            p.s::<2>("m_k", Regs::MK, Live::Whole)?;
            p.s::<2>("stage_base", Regs::SS, Live::Whole)?;
            p.s::<2>("up_scale_base", Regs::SU, Live::Whole)?;
            // Each wave32 lane mask owns an aligned SGPR pair: M7's gfx1100
            // table models VOP3 mask operands with their wave64 width.
            for m in 0..2 * SILU_SLOTS { p.s::<2>("silu_mask", Regs::MASKS + 2 * m, epl())?; }
        }
    }
    Ok(p)
}

fn declare_lds(b: &mut Builder) -> Result<(), String> {
    for (i, (name, base, len)) in [("A0", 0, A_BYTES), ("X0", A_BYTES, A_BYTES), ("A1", SLOT_BYTES, A_BYTES), ("X1", SLOT_BYTES + A_BYTES, A_BYTES)].into_iter().enumerate() {
        if b.lds.add(name, base, len)? != i { return Err("LDS slot order".into()) }
    }
    Ok(())
}

/// Kernel arguments, workgroup bases and every lane-invariant offset.
fn prologue(b: &mut Builder, epi: Epi) -> Result<(), String> {
    let (t0, t1, t64) = (Regs::T0, Regs::T1, Regs::T64);
    let ar = args(epi);
    // gfx11 llvm-objdump spells a zero SMEM offset `null`; parse-back compares canonical text.
    mem(b, format!("s_load_b256 s[{}:{}], s[0:1], null", Regs::ARGS, Regs::ARGS + 7), &[sr(Regs::ARGS, 8)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
    if epi.silu() {
        mem(b, format!("s_load_b64 s[{}:{}], s[0:1], 0x20", Regs::MK, Regs::MK + 1), &[sr(Regs::MK, 2)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
        mem(b, format!("s_load_b32 s{}, s[0:1], 0x28", ar.n), &[s(ar.n)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
    } else {
        mem(b, format!("s_load_b32 s{}, s[0:1], 0x20", ar.n), &[s(ar.n)], &[sr(Regs::KARG, 2)], MemoryClass::SmemLoad)?;
    }
    // v0 = tid; v1 = wave; v2 = lane; v3 = lr; v4 = hi.
    op(b, "v_lshrrev_b32_e32 v1, 5, v0", &[v(1)], &[v(0)])?;
    op(b, "v_and_b32_e32 v2, 31, v0", &[v(2)], &[v(0)])?;
    op(b, format!("v_readfirstlane_b32 s{}, v1", Regs::WAVE), &[s(Regs::WAVE)], &[v(1)])?;
    op(b, "v_and_b32_e32 v3, 15, v2", &[v(3)], &[v(2)])?;
    op(b, "v_lshrrev_b32_e32 v4, 4, v2", &[v(4)], &[v(2)])?;
    op(b, format!("s_lshr_b32 s{}, s{}, 1", Regs::WR, Regs::WAVE), &[s(Regs::WR)], &[s(Regs::WAVE)])?;
    op(b, format!("s_and_b32 s{}, s{}, 1", Regs::WC, Regs::WAVE), &[s(Regs::WC)], &[s(Regs::WAVE)])?;
    // row_bytes = (K/256)*136; trips = K/256 - 1 (whole two-epoch trips before the tail pair).
    op(b, format!("s_lshr_b32 s{}, s{}, 8", Regs::ROWB, ar.k), &[s(Regs::ROWB)], &[s(ar.k)])?;
    op(b, format!("s_add_i32 s{}, s{}, -1", Regs::TRIPS, Regs::ROWB), &[s(Regs::TRIPS)], &[s(Regs::ROWB)])?;
    op(b, format!("s_mulk_i32 s{}, {}", Regs::ROWB, lit(GROUP_BYTES)), &[s(Regs::ROWB)], &[s(Regs::ROWB)])?;
    if epi.silu() {
        // h rows 64*wgy..+63: gate/up scale bases G, U + (64*wgy)*row_bytes;
        // staging base (wave odd ? U : G) + (64*wgy + 16*(wave/2))*row_bytes.
        let u = ar.a + 2;
        op(b, format!("s_lshl_b32 s{t0}, s{}, 6", Regs::WGY), &[s(t0)], &[s(Regs::WGY)])?;
        op(b, format!("s_mul_i32 s{t1}, s{t0}, s{}", Regs::ROWB), &[s(t1)], &[s(t0), s(Regs::ROWB)])?;
        op(b, format!("s_mul_hi_u32 s{t64}, s{t0}, s{}", Regs::ROWB), &[s(t64)], &[s(t0), s(Regs::ROWB)])?;
        op(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::SA, ar.a), &[s(Regs::SA)], &[s(ar.a), s(t1)])?;
        op(b, format!("s_addc_u32 s{}, s{}, s{t64}", Regs::SA + 1, ar.a + 1), &[s(Regs::SA + 1)], &[s(ar.a + 1), s(t64)])?;
        op(b, format!("s_add_u32 s{}, s{u}, s{t1}", Regs::SU), &[s(Regs::SU)], &[s(u), s(t1)])?;
        op(b, format!("s_addc_u32 s{}, s{}, s{t64}", Regs::SU + 1, u + 1), &[s(Regs::SU + 1)], &[s(u + 1), s(t64)])?;
        op(b, format!("s_bitcmp1_b32 s{}, 0", Regs::WAVE), &[], &[s(Regs::WAVE)])?;
        op(b, format!("s_cselect_b64 s[{}:{}], s[{}:{}], s[{}:{}]", Regs::SS, Regs::SS + 1, Regs::SU, Regs::SU + 1, Regs::SA, Regs::SA + 1),
            &[sr(Regs::SS, 2)], &[sr(Regs::SU, 2), sr(Regs::SA, 2)])?;
        op(b, format!("s_lshl_b32 s{t1}, s{}, 4", Regs::WR), &[s(t1)], &[s(Regs::WR)])?;
        op(b, format!("s_mul_i32 s{t1}, s{t1}, s{}", Regs::ROWB), &[s(t1)], &[s(t1), s(Regs::ROWB)])?;
        op(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::SS, Regs::SS), &[s(Regs::SS)], &[s(Regs::SS), s(t1)])?;
        op(b, format!("s_addc_u32 s{}, s{}, 0", Regs::SS + 1, Regs::SS + 1), &[s(Regs::SS + 1)], &[s(Regs::SS + 1)])?;
    } else {
        // A base: A + (128*wgy)*row_bytes.
        op(b, format!("s_lshl_b32 s{t0}, s{}, 7", Regs::WGY), &[s(t0)], &[s(Regs::WGY)])?;
        op(b, format!("s_mul_i32 s{t1}, s{t0}, s{}", Regs::ROWB), &[s(t1)], &[s(t0), s(Regs::ROWB)])?;
        op(b, format!("s_mul_hi_u32 s{t64}, s{t0}, s{}", Regs::ROWB), &[s(t64)], &[s(t0), s(Regs::ROWB)])?;
        op(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::SA, ar.a), &[s(Regs::SA)], &[s(ar.a), s(t1)])?;
        op(b, format!("s_addc_u32 s{}, s{}, s{t64}", Regs::SA + 1, ar.a + 1), &[s(Regs::SA + 1)], &[s(ar.a + 1), s(t64)])?;
    }
    // X base: Xq + (128*wgx)*72.
    op(b, format!("s_mul_i32 s{t1}, s{}, {}", Regs::WGX, lit(128 * XBLK_BYTES)), &[s(t1)], &[s(Regs::WGX)])?;
    op(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::SX, ar.xq), &[s(Regs::SX)], &[s(ar.xq), s(t1)])?;
    op(b, format!("s_addc_u32 s{}, s{}, 0", Regs::SX + 1, ar.xq + 1), &[s(Regs::SX + 1)], &[s(ar.xq + 1)])?;
    op(b, format!("s_mul_i32 s{}, s{}, {}", Regs::N72, ar.n, lit(XBLK_BYTES)), &[s(Regs::N72)], &[s(ar.n)])?;
    op(b, format!("s_lshl_b32 s{}, s{}, 6", Regs::M64, ar.m), &[s(Regs::M64)], &[s(ar.m)])?;
    // Y base: Y + 4*(128*wgx*M + first row), 64-bit; the first row
    // (128*wgy, gate/up 64*wgy) is still in t0.
    op(b, format!("s_lshl_b32 s{t1}, s{}, 7", Regs::WGX), &[s(t1)], &[s(Regs::WGX)])?;
    op(b, format!("s_mul_hi_u32 s{}, s{t1}, s{}", t64 + 1, ar.m), &[s(t64 + 1)], &[s(t1), s(ar.m)])?;
    op(b, format!("s_mul_i32 s{t64}, s{t1}, s{}", ar.m), &[s(t64)], &[s(t1), s(ar.m)])?;
    op(b, format!("s_add_u32 s{t64}, s{t64}, s{t0}"), &[s(t64)], &[s(t64), s(t0)])?;
    op(b, format!("s_addc_u32 s{}, s{}, 0", t64 + 1, t64 + 1), &[s(t64 + 1)], &[s(t64 + 1)])?;
    op(b, format!("s_lshl_b64 s[{t64}:{}], s[{t64}:{}], 2", t64 + 1, t64 + 1), &[sr(t64, 2)], &[sr(t64, 2)])?;
    op(b, format!("s_add_u32 s{}, s{}, s{t64}", Regs::SY, ar.y), &[s(Regs::SY)], &[s(ar.y), s(t64)])?;
    op(b, format!("s_addc_u32 s{}, s{}, s{}", Regs::SY + 1, ar.y + 1, t64 + 1), &[s(Regs::SY + 1)], &[s(ar.y + 1), s(t64 + 1)])?;

    // Staging offsets. sr = 16*wave + lr; the row/token's K16 slice pair i
    // is at +8 (header / d,s skip) + 8*hi + 16*i.
    op(b, format!("v_lshl_add_u32 v5, s{}, 4, v3", Regs::WAVE), &[v(5)], &[s(Regs::WAVE), v(3)])?;
    op(b, "v_lshl_add_u32 v6, v4, 3, 8", &[v(6)], &[v(4)])?;
    // Gate/up stages row lr of its wave's base (the wave's 16 rows are in SS).
    let a_row = if epi.silu() { 3 } else { 5 };
    op(b, format!("v_mad_u32_u24 v{}, v{a_row}, s{}, v6", Regs::A_OFF, Regs::ROWB), &[v(Regs::A_OFF)], &[v(a_row), s(Regs::ROWB), v(6)])?;
    op(b, format!("v_mad_u32_u24 v{}, v5, {}, v6", Regs::X_OFF, lit(XBLK_BYTES)), &[v(Regs::X_OFF)], &[v(5), v(6)])?;
    // LDS store address: wave*1024 + hi*128 + lr*8 (fragment block w, slice 2i+hi).
    op(b, "v_lshlrev_b32_e32 v7, 3, v3", &[v(7)], &[v(3)])?;
    op(b, "v_lshl_add_u32 v16, v4, 7, v7", &[v(16)], &[v(4), v(7)])?;
    op(b, format!("v_lshl_add_u32 v{}, s{}, 10, v16", Regs::ST_A[0], Regs::WAVE), &[v(Regs::ST_A[0])], &[s(Regs::WAVE), v(16)])?;
    for (dst, add) in [(Regs::ST_X[0], A_BYTES), (Regs::ST_A[1], SLOT_BYTES), (Regs::ST_X[1], SLOT_BYTES + A_BYTES)] {
        op(b, format!("v_add_nc_u32_e32 v{dst}, {}, v{}", lit(add), Regs::ST_A[0]), &[v(dst)], &[v(Regs::ST_A[0])])?;
    }
    // A fragment base: wr*2048 + prow*8, prow = 8*(lr&1) + (lr>>1): A lane i
    // supplies row 8*(i&1)+(i>>1), so result VGPR j of lane (hi, lr) is row
    // 8*hi + j of token lr.
    op(b, "v_and_b32_e32 v8, 1, v3", &[v(8)], &[v(3)])?;
    op(b, "v_lshrrev_b32_e32 v9, 1, v3", &[v(9)], &[v(3)])?;
    op(b, "v_lshlrev_b32_e32 v9, 3, v9", &[v(9)], &[v(9)])?;
    op(b, "v_lshl_add_u32 v8, v8, 6, v9", &[v(8)], &[v(8), v(9)])?;
    op(b, format!("v_lshl_add_u32 v{}, s{}, 11, v8", Regs::AB[0], Regs::WR), &[v(Regs::AB[0])], &[s(Regs::WR), v(8)])?;
    op(b, format!("v_add_nc_u32_e32 v{}, {}, v{}", Regs::AB[1], lit(SLOT_BYTES), Regs::AB[0]), &[v(Regs::AB[1])], &[v(Regs::AB[0])])?;
    // X fragment bases: A_BYTES + wc*4096 + lr*8 (+2048 for c = 2, 3).
    op(b, format!("v_lshl_add_u32 v10, s{}, 12, v7", Regs::WC), &[v(10)], &[s(Regs::WC), v(7)])?;
    for (dst, add) in [(Regs::XB[0][0], A_BYTES), (Regs::XB[0][1], A_BYTES + 2048), (Regs::XB[1][0], SLOT_BYTES + A_BYTES), (Regs::XB[1][1], SLOT_BYTES + A_BYTES + 2048)] {
        op(b, format!("v_add_nc_u32_e32 v{dst}, {}, v10", lit(add)), &[v(dst)], &[v(10)])?;
    }
    // Scale row of lane (hi, lr): wr*32 + 16*(lr>>3) + 8*hi + (lr&7);
    // gate/up: h row wr*16 + 8*hi + (lr&7) of both the G and U bases.
    if !epi.silu() {
        op(b, "v_lshrrev_b32_e32 v12, 3, v3", &[v(12)], &[v(3)])?;
        op(b, "v_lshlrev_b32_e32 v12, 4, v12", &[v(12)], &[v(12)])?;
        op(b, "v_lshl_add_u32 v12, v4, 3, v12", &[v(12)], &[v(4), v(12)])?;
    } else {
        op(b, "v_lshlrev_b32_e32 v12, 3, v4", &[v(12)], &[v(4)])?;
    }
    op(b, "v_and_b32_e32 v13, 7, v3", &[v(13)], &[v(3)])?;
    op(b, "v_add_nc_u32_e32 v12, v12, v13", &[v(12)], &[v(12), v(13)])?;
    let wave_rows_log2 = if epi.silu() { 4 } else { 5 };
    op(b, format!("v_lshl_add_u32 v12, s{}, {wave_rows_log2}, v12", Regs::WR), &[v(12)], &[s(Regs::WR), v(12)])?;
    op(b, format!("v_mul_u32_u24_e32 v{}, s{}, v12", Regs::LWS, Regs::ROWB), &[v(Regs::LWS)], &[s(Regs::ROWB), v(12)])?;
    // d of token wc*64 + lr (+16c): (wc*64 + lr)*72.
    op(b, format!("v_lshl_add_u32 v14, s{}, 6, v3", Regs::WC), &[v(14)], &[s(Regs::WC), v(3)])?;
    op(b, format!("v_mul_u32_u24_e32 v{}, {}, v14", Regs::LXS, lit(XBLK_BYTES)), &[v(Regs::LXS)], &[v(14)])?;
    // Y offset of (token wc*64 + lr, row wr*32 + 8*hi; gate/up h row
    // wr*16 + 8*hi), bytes.
    op(b, "v_lshlrev_b32_e32 v15, 3, v4", &[v(15)], &[v(4)])?;
    op(b, format!("v_lshl_add_u32 v15, s{}, {wave_rows_log2}, v15", Regs::WR), &[v(15)], &[s(Regs::WR), v(15)])?;
    op(b, format!("v_mad_u32_u24 v15, v14, s{}, v15", ar.m), &[v(15)], &[v(14), s(ar.m), v(15)])?;
    op(b, format!("v_lshlrev_b32_e32 v{}, 2, v15", Regs::YOFF), &[v(Regs::YOFF)], &[v(15)])?;
    if epi == Epi::Add {
        // Residual line l = tid + 256*i of the tile (8 64-byte lines per
        // token): token tid/8 + 32*i, byte (tid%8)*64; i*32 tokens are
        // added in the tail.
        op(b, format!("s_lshl_b32 s{t1}, s{}, 2", ar.m), &[s(t1)], &[s(ar.m)])?;
        op(b, "v_lshrrev_b32_e32 v17, 3, v0", &[v(17)], &[v(0)])?;
        op(b, "v_and_b32_e32 v18, 7, v0", &[v(18)], &[v(0)])?;
        op(b, "v_lshlrev_b32_e32 v18, 6, v18", &[v(18)], &[v(18)])?;
        op(b, format!("v_mad_u32_u24 v{}, v17, s{t1}, v18", Regs::TOFF), &[v(Regs::TOFF)], &[v(17), s(t1), v(18)])?;
        // M7's gfx1100 table reads a `saddr` global offset as a VGPR pair:
        // the touch reads v182 as the pair's high half.
        op(b, format!("v_mov_b32_e32 v{}, 0", Regs::TPAIR), &[v(Regs::TPAIR)], &[])?;
        op(b, format!("s_lshl_b32 s{}, s{}, 7", Regs::M128, ar.m), &[s(Regs::M128)], &[s(ar.m)])?;
    }

    // Stage epoch 0 into slot 0, and seed the chain constant and the sums
    // while the loads are in flight.
    stage_loads(b, epi, 0)?;
    for j in 0..8u8 { op(b, format!("v_mov_b32_e32 v{}, {}", Regs::MAGIC + j, lit(MAGIC)), &[v(Regs::MAGIC + j)], &[])?; }
    for r in 0..64u8 { op(b, format!("v_mov_b32_e32 v{}, 0", Regs::ACC + r), &[v(Regs::ACC + r)], &[])?; }
    stage_store(b, 0)?;
    b.barrier(&[Transition::Ready(SLOT_A[0]), Transition::Ready(SLOT_X[0])])
}

/// Global loads of the next epoch's staging packet. `a_imm` selects the K
/// half inside the current A group (64 for an odd epoch).
fn stage_loads(b: &mut Builder, epi: Epi, a_imm: u32) -> Result<(), String> {
    let base = stage_base(epi);
    b.clause(|b| { for i in 0..4u8 { global_load(b, 2, Regs::STA + 2 * i, Regs::A_OFF, base, a_imm + 16 * u32::from(i))?; } Ok(()) })?;
    b.clause(|b| { for i in 0..4u8 { global_load(b, 2, Regs::STX + 2 * i, Regs::X_OFF, Regs::SX, 16 * u32::from(i))?; } Ok(()) })
}

/// Rebias A and publish the staged packet into slot `slot`: each store fills
/// one 256-byte block (slices 2i and 2i+1 of fragment block `wave`).
fn stage_store(b: &mut Builder, slot: usize) -> Result<(), String> {
    for r in 0..8u8 {
        let x = Regs::STA + r;
        op(b, format!("v_xor_b32_e32 v{x}, {}, v{x}", lit(REBIAS)), &[v(x)], &[v(x)])?;
    }
    for (regs, addr, slot_id) in [(Regs::STA, Regs::ST_A[slot], SLOT_A[slot]), (Regs::STX, Regs::ST_X[slot], SLOT_X[slot])] {
        for pair in 0..2u8 {
            let (d0, d1) = (regs + 4 * pair, regs + 4 * pair + 2);
            let text = format!("ds_store_2addr_b64 v{addr}, v[{d0}:{}], v[{d1}:{}]{}", d0 + 1, d1 + 1, ds_offsets(64 * u32::from(pair), 64 * u32::from(pair) + 32));
            b.ds_store(slot_id, Instruction::new(text, vec![], vec![v(addr), vr(d0, 2), vr(d1, 2)]).memory(MemoryClass::DsStore))?;
        }
    }
    Ok(())
}

/// Fragment loads of step `i = 8a + s` from slot `slot`: the X fragments of
/// slice s (c = 0,1 and c = 2,3) and, at even s, the A pair (s, s+1).
fn step_loads(b: &mut Builder, slot: usize, i: usize) -> Result<(), String> {
    let (a, s_) = ((i / 8) as u32, (i % 8) as u32);
    if s_ % 2 == 0 {
        let dst = Regs::AV[(i / 2) % 2];
        let base = Regs::AB[slot];
        let o = a * 128 + 16 * s_;
        b.ds_load(SLOT_A[slot], Instruction::new(format!("ds_load_2addr_b64 {}, v{base}{}", vr(dst, 4), ds_offsets(o, o + 16)), vec![vr(dst, 4)], vec![v(base)]).memory(MemoryClass::DsLoad))?;
    }
    for half in 0..2usize {
        let dst = Regs::XV[i % 2] + 4 * half as u8;
        let base = Regs::XB[slot][half];
        b.ds_load(SLOT_X[slot], Instruction::new(format!("ds_load_2addr_b64 {}, v{base}{}", vr(dst, 4), ds_offsets(16 * s_, 16 * s_ + 128)), vec![vr(dst, 4)], vec![v(base)]).memory(MemoryClass::DsLoad))?;
    }
    Ok(())
}

fn step_wmma(b: &mut Builder, i: usize) -> Result<(), String> {
    let a = V::<2>(Regs::AV[(i / 2) % 2] + 2 * (i % 2) as u8);
    for c in 0..4u8 {
        let dst = V::<8>(Regs::C + 8 * c);
        let seed = if i % 8 == 0 { V::<8>(Regs::MAGIC) } else { dst };
        b.push(Wmma::iu4(b.spec.arch, dst, a, V::<2>(Regs::XV[i % 2] + 2 * c), Some(seed)))?;
    }
    Ok(())
}

/// Fold pass `a`: sum[a][c][j] = fma(d_c * sc_j, C[c][j] - 1.5*2^23, sum).
/// Row 16a + 8hi + j's scale comes from lane 8a + j of this lane's row of 16
/// (gate/up: pass 0 from the gate scales, pass 1 from the up scales).
fn fold(b: &mut Builder, epi: Epi, a: u8) -> Result<(), String> {
    let scale = if epi.silu() && a == 1 { Regs::WSFU } else { Regs::WSF };
    if a == 0 || epi.silu() {
        // True16 VOP1 reaches only v0-v127 halves; v149/v181 need the VOP3 form.
        let raw = if a == 0 { Regs::WS } else { Regs::WSU };
        op(b, format!("v_cvt_f32_f16_e64 v{scale}, v{raw}.l"), &[v(scale)], &[v(raw)])?;
    }
    for j in 0..8u8 {
        op(b, format!("v_mov_b32_dpp v{}, v{scale} row_share:{} row_mask:0xf bank_mask:0xf", Regs::SCF + j, 8 * a + j),
            &[v(Regs::SCF + j)], &[v(scale)])?;
    }
    for cp in [0u8, 2] {
        // t_j = d_c * sc_j paired with C[c][j^1] += -1.5*2^23 (opposite
        // destination parity, distinct src1 banks); then the fmac pairs.
        for c in [cp, cp + 1] {
            let t = Regs::T[usize::from(c % 2)];
            for j in 0..8u8 {
                let mul = VopdOp { op: VopdF32::Mul, dst: t + j, src0: Operand::V(Regs::DC + c), src1: Regs::SCF + j };
                let cf = Regs::C + 8 * c + (j ^ 1);
                let add = VopdOp { op: VopdF32::Add, dst: cf, src0: Operand::Lit(MAGIC_NEG), src1: cf };
                b.vopd(mul, add)?;
            }
        }
        for c in [cp, cp + 1] {
            let t = Regs::T[usize::from(c % 2)];
            let acc = Regs::ACC + 8 * (4 * a + c);
            for j in (0..8u8).step_by(2) {
                let x = VopdOp { op: VopdF32::Fmac, dst: acc + j, src0: Operand::V(t + j), src1: Regs::C + 8 * c + j };
                let y = VopdOp { op: VopdF32::Fmac, dst: acc + j + 1, src0: Operand::V(t + j + 1), src1: Regs::C + 8 * c + j + 1 };
                b.vopd(x, y)?;
            }
        }
    }
    Ok(())
}

/// One K128 epoch reading slot `p` (epoch parity p). With `next`, the
/// staging packet of epoch e+1 is loaded, published into slot 1-p and the
/// barrier hands the slots over; the last epoch has neither. `touch` (ADD
/// touch, last two epochs) loads residual lines `touch` and `touch + 1`.
fn epoch(b: &mut Builder, epi: Epi, p: usize, next: bool, touch: bool) -> Result<(), String> {
    // Current-epoch metadata first: in-order VMcnt returns it before the packet.
    let scales: &[(u8, u8)] = if epi.silu() { &[(Regs::WS, Regs::SA), (Regs::WSU, Regs::SU)] } else { &[(Regs::WS, Regs::SA)] };
    for &(dst, base) in scales {
        mem(b, format!("global_load_u16 v{dst}, v{}, s[{base}:{}]{}", Regs::LWS, base + 1, off(4 * p as u32)?),
            &[v(dst)], &[v(Regs::LWS), sr(base, 2)], MemoryClass::VmemLoad)?;
    }
    for c in 0..4u8 { global_load(b, 1, Regs::DC + c, Regs::LXS, Regs::SX, 16 * XBLK_BYTES * u32::from(c))?; }
    if touch { touch_window(b, p)?; }
    if next {
        if p == 1 {
            // Epoch e+1 starts the next 136-byte A group.
            let bases: &[u8] = if epi.silu() { &[Regs::SA, Regs::SU, Regs::SS] } else { &[Regs::SA] };
            for &base in bases {
                op(b, format!("s_add_u32 s{0}, s{0}, {1}", base, lit(GROUP_BYTES)), &[s(base)], &[s(base)])?;
                op(b, format!("s_addc_u32 s{0}, s{0}, 0", base + 1), &[s(base + 1)], &[s(base + 1)])?;
            }
        }
        op(b, format!("s_add_u32 s{0}, s{0}, s{1}", Regs::SX, Regs::N72), &[s(Regs::SX)], &[s(Regs::SX), s(Regs::N72)])?;
        op(b, format!("s_addc_u32 s{0}, s{0}, 0", Regs::SX + 1), &[s(Regs::SX + 1)], &[s(Regs::SX + 1)])?;
        stage_loads(b, epi, if p == 0 { 64 } else { 0 })?;
    }
    step_loads(b, p, 0)?;
    for i in 0..16 {
        if i + 1 < 16 { step_loads(b, p, i + 1)?; }
        step_wmma(b, i)?;
        if i == 7 { fold(b, epi, 0)?; }
    }
    // Publish the staged packet before fold pass 1 (slot 1-p was retired by
    // the previous barrier), so its LGKM drain hides under the fold: this
    // measured 0.3-0.6% faster on gate/up than publishing after the fold.
    if next { stage_store(b, 1 - p)?; }
    fold(b, epi, 1)?;
    if next {
        b.barrier(&[Transition::Retire(SLOT_A[p]), Transition::Retire(SLOT_X[p]), Transition::Ready(SLOT_A[1 - p]), Transition::Ready(SLOT_X[1 - p])])?;
    }
    Ok(())
}

/// ADD residual touch, every loop epoch: loop trip t (the trip-count SGPR, which
/// runs K/256-1 .. 1) covers epochs E-2-2t and E-1-2t, so epoch parity p of
/// trips 7 and 6 is window epoch i = 14 - 2t + p of E-16..E-13 and touches
/// line i (32*i tokens past `TOFF`). EXEC is empty outside the window, so
/// the body stays one straight-line trip; the load always counts on VMcnt
/// and rides ahead of the staging packet, whose wait retires it.
fn touch_window(b: &mut Builder, p: usize) -> Result<(), String> {
    let (t0, t1) = (Regs::T0, Regs::T1);
    op(b, format!("s_add_i32 s{t0}, s{}, -6", Regs::TRIPS), &[s(t0)], &[s(Regs::TRIPS)])?;
    op(b, format!("s_cmp_lt_u32 s{t0}, 2"), &[], &[s(t0)])?;
    op(b, "s_cselect_b32 exec_lo, -1, 0", &[], &[])?;
    op(b, format!("s_lshl_b32 s{t1}, s{t0}, 1"), &[s(t1)], &[s(t0)])?;
    op(b, format!("s_sub_i32 s{t1}, {}, s{t1}", 2 + p), &[s(t1)], &[s(t1)])?;
    op(b, format!("s_mul_i32 s{t1}, s{t1}, s{}", Regs::M128), &[s(t1)], &[s(t1), s(Regs::M128)])?;
    op(b, format!("s_add_u32 s{}, s{}, s{t1}", Regs::STB, Regs::SY), &[s(Regs::STB)], &[s(Regs::SY), s(t1)])?;
    op(b, format!("s_addc_u32 s{}, s{}, 0", Regs::STB + 1, Regs::SY + 1), &[s(Regs::STB + 1)], &[s(Regs::SY + 1)])?;
    let sink = Regs::TSINK;
    mem(b, format!("global_load_b32 v{sink}, v{}, s[{}:{}]", Regs::TOFF, Regs::STB, Regs::STB + 1),
        &[v(sink)], &[v(Regs::TOFF), sr(Regs::STB, 2)], MemoryClass::VmemLoad)?;
    op(b, "s_mov_b32 exec_lo, -1", &[], &[])
}

fn kloop(b: &mut Builder, epi: Epi) -> Result<(), String> {
    b.label(K_BEGIN)?;
    if !b.ledger.is_empty() { return Err(format!("prologue left memory operations pending at the K loop: {:?}", b.ledger.shape())) }
    op(b, format!("s_cmp_eq_u32 s{}, 0", Regs::TRIPS), &[], &[s(Regs::TRIPS)])?;
    op(b, format!("s_cbranch_scc1 {TAIL}"), &[], &[])?;
    b.loop_(K_LOOP, |b| {
        let touch = epi == Epi::Add;
        epoch(b, epi, 0, true, touch)?;
        epoch(b, epi, 1, true, touch)?;
        op(b, format!("s_add_i32 s{0}, s{0}, -1", Regs::TRIPS), &[s(Regs::TRIPS)], &[s(Regs::TRIPS)])?;
        op(b, format!("s_cmp_lg_u32 s{}, 0", Regs::TRIPS), &[], &[s(Regs::TRIPS)])?;
        op(b, format!("s_cbranch_scc1 {K_LOOP}"), &[], &[])
    })?;
    b.label(K_LOOP_END)?;
    // Both paths into the tail (no trip, or after the loop) arrive with an
    // empty ledger, so its waits hold on either.
    if !b.ledger.is_empty() { return Err("K loop exit leaves memory operations pending".into()) }
    b.label(TAIL)?;
    epoch(b, epi, 0, true, false)?;
    epoch(b, epi, 1, false, false)
}

/// Lane (hi, lr) owns rows 8hi..8hi+7 of each fragment for one token; two
/// b128 accesses per (a, c), token fragment c 16*M*4 bytes further.
/// SET stores the sums; ADD first loads every residual quad, then stores
/// RN(old + sum) per quad; gate/up stores h = SILU_MUL(g, u) of fragment
/// pair (0, 1), one h octet per token fragment.
fn epilogue(b: &mut Builder, epi: Epi) -> Result<(), String> {
    b.label(EPI)?;
    op(b, format!("v_mov_b32_e32 v0, v{}", Regs::YOFF), &[v(0)], &[v(Regs::YOFF)])?;
    for c in 1..4u8 {
        op(b, format!("v_add_nc_u32_e32 v{c}, s{}, v{}", Regs::M64, c - 1), &[v(c)], &[s(Regs::M64), v(c - 1)])?;
    }
    let store = |b: &mut Builder, c: u8, data: u8, offset: u32| {
        let data = vr(data, 4);
        mem(b, format!("global_store_b128 v{c}, {data}, s[{}:{}]{}", Regs::SY, Regs::SY + 1, off(offset)?),
            &[], &[v(c), data, sr(Regs::SY, 2)], MemoryClass::VmemStore)
    };
    let quads = || (0..4u8).flat_map(|c| (0..2u8).flat_map(move |a| (0..2u8).map(move |q| (c, a, q))));
    let offset = |a: u8, q: u8| 4 * (16 * u32::from(a) + 4 * u32::from(q));
    let acc = |c: u8, a: u8, q: u8| Regs::ACC + 8 * (4 * a + c) + 4 * q;
    match epi {
        Epi::Set => {
            for (c, a, q) in quads() { store(b, c, acc(c, a, q), offset(a, q))?; }
        }
        Epi::Add => {
            for (i, (c, a, q)) in quads().enumerate() {
                let r = Regs::RESID + 4 * i as u8;
                mem(b, format!("global_load_b128 {}, v{c}, s[{}:{}]{}", vr(r, 4), Regs::SY, Regs::SY + 1, off(offset(a, q))?),
                    &[vr(r, 4)], &[v(c), sr(Regs::SY, 2)], MemoryClass::VmemLoad)?;
            }
            for (i, (c, a, q)) in quads().enumerate() {
                let (r, x) = (Regs::RESID + 4 * i as u8, acc(c, a, q));
                for k in [0u8, 2] {
                    let lo = VopdOp { op: VopdF32::Add, dst: r + k, src0: Operand::V(r + k), src1: x + k };
                    let hi = VopdOp { op: VopdF32::Add, dst: r + k + 1, src0: Operand::V(r + k + 1), src1: x + k + 1 };
                    b.vopd(lo, hi)?;
                }
                store(b, c, r, offset(a, q))?;
            }
        }
        Epi::Silu => {
            let region = Region::silu_gfx1100()?;
            if region.temps > usize::from(SILU_TEMPS) || region.masks > 2 { return Err("gfx1100 SiLU region needs more temporaries than planned".into()) }
            // Element k of token fragment c uses slot k % 4 (temporaries
            // SILU_T[q] + 0..8, masks s[36 + 4q], s[38 + 4q]), so k + 4 starts once k
            // has issued its last op; VOPD partners (2i, 2i + 1) sit an odd
            // distance apart in every operand (g, u, h: 1; temporaries: 9),
            // which keeps each same-op packet parity- and bank-legal.
            for c in 0..4u8 {
                let binds: Vec<Binding> = (0..8u8).map(|k| {
                    let q = k % SILU_SLOTS;
                    Binding {
                        g: acc(c, 0, 0) + k, u: acc(c, 1, 0) + k, out: Regs::H + 8 * c + k,
                        temps: (0..SILU_TEMPS).map(|t| Regs::SILU_T[usize::from(q)] + t).collect(),
                        masks: vec![Regs::MASKS + 4 * q, Regs::MASKS + 4 * q + 2],
                    }
                }).collect();
                region::emit_interleaved(b, &region, &binds)?;
                for q in 0..2u8 { store(b, c, Regs::H + 8 * c + 4 * q, 16 * u32::from(q))?; }
            }
        }
    }
    Ok(())
}

pub fn emit(spec: Spec) -> Result<Emitted, String> {
    spec.validate()?;
    let kspec = KernelSpec {
        kernel_id: "iu4_v2c".into(), variant: spec.epi.name().into(), arch: spec.arch, symbol: spec.symbol(),
        kernargs: spec.kernargs(), user_sgpr_count: 2, system_sgpr_workgroup_id_y: true,
        workgroup_size: THREADS, group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    };
    let mut b = Builder::new(kspec, plan(spec.epi)?);
    b.enable_delay_alu();
    declare_lds(&mut b)?;
    prologue(&mut b, spec.epi)?;
    kloop(&mut b, spec.epi)?;
    epilogue(&mut b, spec.epi)?;
    b.label(END)?;
    b.push(Sop::End.encode(spec.arch)?)?;
    b.finish()
}

/// Name of the gfx1100 product code object.
pub const MODULE: &str = "gemm_mq4g256v2_residual_iu4_pm_gfx1100";

/// One code object with the given epilogue symbols: each kernel's local
/// `.Lv2c_` labels are qualified by its epilogue so they share one
/// assembler unit (the machine code of every symbol is unchanged).
pub fn module(arch: Arch, epis: &[Epi]) -> Result<(Vec<Emitted>, String, super::iu4_gemm::ModuleProof), String> {
    let emitted = epis.iter().map(|&epi| emit(Spec { arch, epi })).collect::<Result<Vec<_>, _>>()?;
    let renamed: Vec<Emitted> = emitted.iter().map(|e| {
        let mut e = e.clone();
        e.s_text = e.s_text.replace(".Lv2c_", &format!(".Lv2c_{}_", e.proof.variant));
        e
    }).collect();
    let (text, proof) = super::iu4_gemm::module(&renamed, MODULE)?;
    Ok((emitted, text, proof))
}

/// Instruction census of one steady-state K-loop trip (two epochs).
pub fn hot_loop_census(s_text: &str) -> std::collections::BTreeMap<String, u32> {
    let mut counts = std::collections::BTreeMap::new();
    let mut inside = false;
    for line in s_text.lines() {
        let t = line.trim();
        if t == format!("{K_LOOP}:") { inside = true; continue }
        if t == format!("{K_LOOP_END}:") { break }
        if !inside || t.ends_with(':') || t.is_empty() || t.starts_with('.') { continue }
        let name = t.split_whitespace().next().unwrap_or("").to_owned();
        *counts.entry(name.clone()).or_insert(0) += 1;
        if name.starts_with("v_dual_") { *counts.entry("vopd_packets".into()).or_insert(0) += 1 }
        if name.starts_with("v_") && !name.starts_with("v_wmma_") { *counts.entry("valu_slots".into()).or_insert(0) += 1 }
    }
    counts
}
