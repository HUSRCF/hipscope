//! Exact CPU semantics for scalar and vector ALU opcodes (wave32).
//!
//! Every opcode is keyed by its exact ISA name; an unknown name is an error, never a
//! best guess. Float arithmetic uses host IEEE-754 binary32 (`+ - * fma`, round to
//! nearest even), which is bit-exact for every non-NaN operand and every non-NaN result. Whatever the
//! ISA documents only loosely is a hard error rather than a guess:
//!
//! * NaN operands of any arithmetic (propagation order and payload are not measured), and NaN
//!   results of arithmetic not listed below (sign/payload of the default NaN is not specified
//!   for add/fma/ldexp/convert), except where the ISA gives an explicit rule
//!   (`max_num`/`min_num`, `div_fixup`, compares, class, f32->int).
//! * Invalid operations (NaN result from non-NaN operands) of `v_sub_f32`/`v_subrev_f32`/
//!   `v_mul_f32` (and their VOPD aliases) on every arch except gfx1151, where all 32 lanes of
//!   the device probe (`gfx1151-alu-edges.log`: `-inf - -inf`, `-inf * 0`) produced the default
//!   NaN `0xffc00000`. The rule is applied to any invalid operation of those opcodes, not to
//!   the probed operands only; gfx1201 is not measured.
//! * `v_div_scale_f32` lanes whose 1/S1 is possibly denormal (`exponent(S1) >= 253`),
//!   whose quotient lies on the denormal boundary, or that need `NAN.f32`.
//! * `v_div_fmas_f32` with a set VCC lane (the document says `2.0F ** 32` while
//!   `v_div_scale_f32` scales by `ldexp(.., 64)`; the gfx1151 probe measured a `2^-64` factor,
//!   but fused scaled rounding, subnormal and overflow behavior are not qualified).
//! * `v_div_fixup_f32` quotients that underflow (`UNDERFLOW_F32` is not defined).
//! * `v_rndne_f32` of a negative input that rounds to zero on any arch except gfx1151 (documented
//!   pseudocode gives `+0`, IEEE gives `-0`; gfx1151 measured `-0` = `0x80000000` for
//!   `-0.25, -0.5, -minsub, -0`, so host `round_ties_even` is exact there; gfx1201 is not measured).
//! * OMOD results that would flush a non-zero denormal (sign of the flushed zero).
//! * DPP with FI=1 and an out-of-range source (the BC/FI table disagrees with its prose).
//!
//! Descriptor float mode must be round to nearest even, allow input/output denormals,
//! use FP16_OVFL=0, and on gfx1151 use IEEE_MODE=1/DX10_CLAMP=1; `State::new` checks it.
//! F8_MODE has no descriptor field: FP8 conversion assumes the normal launch default
//! F8_MODE=0. No opcode in the semantic table changes MODE.
use super::{convert, Result, State};
use crate::{
    operand::{Dpp, InlineConst, Omod, Operand},
    Arch, FormFields, Inst,
};

const NAN_IN: &str = "NaN operand: the ISA document does not specify NaN propagation (selection/payload/sign) for this opcode";
const NAN_OUT: &str = "NaN result: the ISA document does not specify the default NaN encoding for this opcode";
const UNSUPPORTED: &str = "unsupported vector opcode";
const INF: u32 = 0x7f80_0000;

/// Where the first source operand of a lane comes from (DPP routing).
#[derive(Clone, Copy)]
enum Src0 {
    Plain,
    Lane(usize),
    Zero,
}

/// How a computed dword lands in the destination register.
#[derive(Clone, Copy)]
enum Val {
    Dword(u32),
    Low16(u32),
    High16(u32),
}

/// Modifier policy of an opcode's result.
#[derive(Clone, Copy)]
enum Class {
    /// Integer/bitwise: no source modifiers, clamp or omod.
    Int,
    /// Bit select: source neg/abs only.
    Select,
    /// Integer result from a float source: source neg/abs and (inert) clamp.
    FloatToInt,
    /// f32 result: source modifiers, omod, clamp.
    F32,
    /// Handled by the opcode itself.
    Own,
}

fn is_nan(bits: u32) -> bool {
    bits & 0x7fff_ffff > INF
}

fn num(bits: u32) -> Result<f32> {
    if is_nan(bits) {
        Err(NAN_IN.into())
    } else {
        Ok(f32::from_bits(bits))
    }
}

fn fin(x: f32) -> Result<u32> {
    if x.is_nan() {
        Err(NAN_OUT.into())
    } else {
        Ok(x.to_bits())
    }
}

/// gfx1151 default NaN of an invalid operation, measured on hardware for `v_sub_f32` and `v_mul_f32`.
const GFX1151_DEFAULT_NAN: u32 = 0xffc0_0000;

/// Result of `v_sub_f32`/`v_subrev_f32`/`v_mul_f32`, whose operands are already known to be
/// non-NaN (`num`): a NaN result is then an invalid operation. Only gfx1151 has a measured
/// default NaN; other arches keep the hard error.
fn fin_invalid(arch: Arch, x: f32) -> Result<u32> {
    if x.is_nan() && arch == Arch::Gfx1151 {
        Ok(GFX1151_DEFAULT_NAN)
    } else {
        fin(x)
    }
}

/// Source modifiers: absolute value first, negate after.
fn sgn(v: u32, abs: bool, neg: bool) -> u32 {
    let v = if abs { v & 0x7fff_ffff } else { v };
    if neg {
        v ^ 0x8000_0000
    } else {
        v
    }
}

fn active(s: &State) -> impl Iterator<Item = usize> {
    let exec = s.exec;
    (0..32).filter(move |lane| exec & (1 << lane) != 0)
}

pub(super) fn execute(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    if let FormFields::Vopd { y_op, x_operands } = i.fields {
        let y = y_op.name(arch).ok_or("unknown VOPD Y opcode")?;
        return vopd(arch, name, y, usize::from(x_operands), i, s);
    }
    if name.starts_with("s_") {
        return scalar(name, i, s);
    }
    match name {
        "v_readfirstlane_b32" => readfirstlane(i, s),
        _ if name.starts_with("v_wmma") => super::wmma::execute(arch,name,i,s),
        "v_permlane16_b32" | "v_permlanex16_b32" => permlane(name, i, s),
        "v_div_scale_f32" => div_scale(arch, i, s),
        "v_cvt_pk_f32_fp8_e32" | "v_cvt_pk_f32_fp8_e64" => cvt_pk_f32_fp8(i, s),
        _ if name.starts_with("v_cmp") => compare(name, i, s),
        _ => lanes(arch, name, i, s),
    }
}

fn store(s: &mut State, dst: &Operand, lane: usize, v: Val) -> Result<()> {
    match v {
        Val::Dword(x) => s.put(dst, lane, 0, x),
        Val::Low16(x) => {
            let old = s.read(dst, lane, 0)?;
            s.put(dst, lane, 0, (old & 0xffff_0000) | (x & 0xffff))
        }
        Val::High16(x) => {
            let old = s.read(dst, lane, 0)?;
            s.put(dst, lane, 0, (old & 0xffff) | (x << 16))
        }
    }
}

fn dword(v: Val) -> Result<u32> {
    match v {
        Val::Dword(x) => Ok(x),
        _ => Err("VOPD half-dword destination unsupported".into()),
    }
}

/// VOPD: both halves read the pre-instruction state; both writes land afterwards.
fn vopd(arch: Arch, x: &str, y: &str, split: usize, i: &Inst, s: &mut State) -> Result<()> {
    let (ox, oy) = i.operands.split_at(split);
    if s.exec == 0 {
        // Still reject unknown opcodes when no lane is active.
        for (name, ops) in [(x, ox), (y, oy)] {
            if let Err(e) = vector(arch, name, ops, i, s, 0, Src0::Plain) {
                if e.starts_with(UNSUPPORTED) {
                    return Err(e);
                }
            }
        }
        return Ok(());
    }
    let mut out = [[0u32; 2]; 32];
    for lane in active(s) {
        out[lane] = [
            dword(vector(arch, x, ox, i, s, lane, Src0::Plain)?)?,
            dword(vector(arch, y, oy, i, s, lane, Src0::Plain)?)?,
        ];
    }
    for lane in active(s) {
        s.put(&ox[0], lane, 0, out[lane][0])?;
        s.put(&oy[0], lane, 0, out[lane][1])?;
    }
    Ok(())
}

fn readfirstlane(i: &Inst, s: &mut State) -> Result<()> {
    // EXEC == 0 forces lane 0.
    let lane = if s.exec == 0 { 0 } else { s.exec.trailing_zeros() as usize };
    let v = s.read(&i.operands[1], lane, 0)?;
    s.put(&i.operands[0], 0, 0, v)
}

fn dpp_fi(i: &Inst) -> bool {
    match &i.fields {
        FormFields::Bits { honored, ignored } => honored.iter().chain(ignored.iter()).any(|f| f.name == "dpp_fi" && f.value != 0),
        _ => false,
    }
}

/// Source lane for a DPP16 control word (wave32 layout); `None` is out of range.
fn dpp_lane(ctrl: u16, lane: usize) -> Result<Option<usize>> {
    let row = lane & !15;
    let col = lane & 15;
    let n = usize::from(ctrl & 15);
    Ok(match ctrl {
        0x000..=0x0ff => Some((lane & !3) | usize::from((ctrl >> (2 * (lane & 3))) & 3)),
        0x101..=0x10f => (col + n < 16).then_some(lane + n),
        0x111..=0x11f => col.checked_sub(n).map(|c| row + c),
        0x121..=0x12f => Some(row + ((col + 16 - n) & 15)),
        0x140 => Some(row + 15 - col),
        0x141 => Some((lane & !7) + 7 - (lane & 7)),
        0x150..=0x15f => Some(row + n),
        0x160..=0x16f => Some(row + (col ^ n)),
        _ => return Err(format!("DPP control {ctrl:#x} is not a valid RDNA DPP16 control")),
    })
}

/// DPP routing for one lane: `None` disables the destination write.
///
/// BC/FI table (FI=0): out-of-range or disabled source lane -> write disabled when
/// BC=0, source reads zero when BC=1. FI=1 reads a disabled in-range lane normally.
fn dpp_plan(d: &Dpp, fi: bool, s: &State, lane: usize) -> Result<Option<Src0>> {
    if d.row_mask & (1 << (lane / 16)) == 0 || d.bank_mask & (1 << ((lane / 4) & 3)) == 0 {
        return Ok(None);
    }
    let invalid = if d.bound_ctrl { Some(Src0::Zero) } else { None };
    Ok(match dpp_lane(d.ctrl, lane)? {
        None if fi => return Err("DPP FI=1 with an out-of-range source lane: the BC/FI table and its prose disagree".into()),
        None => invalid,
        Some(src) if fi || s.exec & (1 << src) != 0 => Some(Src0::Lane(src)),
        Some(_) => invalid,
    })
}

fn read_src(s: &State, op: &Operand, lane: usize, d: Src0) -> Result<u32> {
    match d {
        Src0::Plain => s.read(op, lane, 0),
        Src0::Lane(src) => s.read(op, src, 0),
        Src0::Zero => Ok(0),
    }
}

fn stem(name: &str) -> &str {
    name.strip_suffix("_e32")
        .or_else(|| name.strip_suffix("_e64"))
        .or_else(|| name.strip_suffix("_dpp"))
        .unwrap_or(name)
}

fn lanes(arch: Arch, name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let dpp = i.mods.dpp;
    if dpp.is_some() != name.ends_with("_dpp") {
        return Err(format!("{name}: DPP control and opcode form disagree"));
    }
    if s.exec == 0 {
        if let Err(e) = vector(arch, name, o, i, s, 0, Src0::Plain) {
            if e.starts_with(UNSUPPORTED) {
                return Err(e);
            }
        }
        return Ok(());
    }
    let fi = dpp_fi(i);
    let mut vals: [Option<Val>; 32] = [None; 32];
    for lane in active(s) {
        let d = match dpp {
            Some(d) => match dpp_plan(&d, fi, s, lane)? {
                Some(d) => d,
                None => continue,
            },
            None => Src0::Plain,
        };
        vals[lane] = Some(vector(arch, name, o, i, s, lane, d)?);
    }
    for (lane, v) in vals.into_iter().enumerate() {
        if let Some(v) = v {
            store(s, &o[0], lane, v)?;
        }
    }
    Ok(())
}

fn plain(i: &Inst) -> Result<()> {
    let m = &i.mods;
    if m.neg | m.abs != 0 || m.omod != Omod::None || m.clamp {
        Err("neg/abs/omod/clamp on an integer-result opcode is not modeled".into())
    } else {
        Ok(())
    }
}

/// Output modifiers for an f32 result: OMOD, then CLAMP (NaN clamps to +0, -0 to +0).
fn out_f32(arch: Arch, bits: u32, i: &Inst) -> Result<u32> {
    let m = &i.mods;
    let mut x = f32::from_bits(bits);
    if m.omod != Omod::None {
        match arch {
            // RDNA3: OMOD is ignored when output denormals are enabled.
            Arch::Gfx1100 | Arch::Gfx1151 => {}
            Arch::Gfx1201 => {
                if x.is_nan() {
                    return Err("OMOD on a NaN result: payload not specified".into());
                }
                let y = match m.omod {
                    Omod::Mul2 => x * 2.0,
                    Omod::Mul4 => x * 4.0,
                    Omod::Div2 => x * 0.5,
                    Omod::None => x,
                };
                if y != 0.0 && y.abs() < f32::MIN_POSITIVE {
                    return Err("OMOD flushes a non-zero denormal: sign of the flushed zero is not specified".into());
                }
                // -0 * OMOD = +0.
                x = if y == 0.0 { 0.0 } else { y };
            }
            _ => return Err("OMOD semantics for this architecture are not modeled".into()),
        }
    }
    if m.clamp {
        x = if x.is_nan() || x <= 0.0 {
            0.0
        } else if x >= 1.0 {
            1.0
        } else {
            x
        };
    }
    Ok(x.to_bits())
}

fn max_gt(a: f32, b: f32) -> bool {
    a > b || (a == 0.0 && b == 0.0 && !a.is_sign_negative() && b.is_sign_negative())
}

fn min_lt(a: f32, b: f32) -> bool {
    a < b || (a == 0.0 && b == 0.0 && a.is_sign_negative() && !b.is_sign_negative())
}

/// maxNum/minNum rules of the ISA (`v_max_num_f32`): a NaN operand loses to a number.
fn num_select(a: u32, b: u32, max: bool) -> u32 {
    match (is_nan(a), is_nan(b)) {
        (true, true) => a | 0x0040_0000,
        (true, false) => b,
        (false, true) => a,
        (false, false) => {
            let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
            if if max { max_gt(fa, fb) } else { min_lt(fa, fb) } {
                a
            } else {
                b
            }
        }
    }
}

/// Legacy `v_max_f32`/`v_min_f32`: NaN behavior depends on MODE.IEEE, so NaN is an error.
fn legacy_select(a: u32, b: u32, max: bool) -> Result<u32> {
    let (fa, fb) = (num(a)?, num(b)?);
    Ok(if if max { max_gt(fa, fb) } else { min_lt(fa, fb) } { a } else { b })
}

fn f16_in(h: u16) -> Result<u32> {
    if h & 0x7c00 == 0x7c00 && h & 0x3ff != 0 {
        return Err("NaN half operand: payload handling not specified".into());
    }
    Ok(convert::f16_to_f32(h))
}

/// A 16-bit source: true16 operands already carry their half; SGPR/literal operands use
/// OP_SEL; float inline constants are f16 values.
fn half_src(s: &State, op: &Operand, i: &Inst, lane: usize, n: usize) -> Result<u16> {
    let v = match op {
        Operand::Half(..) => s.read(op, lane, 0)?,
        Operand::Inline(InlineConst::FloatBits(b)) => u32::from(convert::f32_to_f16(*b)),
        Operand::Inline(InlineConst::InvTwoPi) => 0x3118,
        _ => {
            let w = s.read(op, lane, 0)?;
            if i.mods.op_sel & (1 << n) != 0 {
                w >> 16
            } else {
                w
            }
        }
    };
    Ok(v as u16)
}

/// Fused multiply-add inputs for `v_fma_mix*`: `{OPSEL_HI,OPSEL}` choose f32/lo/hi half,
/// NEG_HI is an absolute value, NEG a negation.
fn mix_in(s: &State, o: &[Operand], i: &Inst, lane: usize) -> Result<[f32; 3]> {
    let m = &i.mods;
    let mut out = [0f32; 3];
    for n in 0..3 {
        let op = o.get(n + 1).ok_or("missing vector operand")?;
        let bits = if m.op_sel_hi & (1 << n) == 0 {
            s.read(op, lane, 0)?
        } else if let Operand::Inline(InlineConst::FloatBits(b)) = op {
            f16_in(convert::f32_to_f16(*b))?
        } else {
            let w = s.read(op, lane, 0)?;
            f16_in((if m.op_sel & (1 << n) != 0 { w >> 16 } else { w }) as u16)?
        };
        out[n] = num(sgn(bits, m.neg_hi & (1 << n) != 0, m.neg_lo & (1 << n) != 0))?;
    }
    Ok(out)
}

fn clamp01(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x
    }
}

fn vector(arch: Arch, name: &str, o: &[Operand], i: &Inst, s: &State, lane: usize, d: Src0) -> Result<Val> {
    let m = &i.mods;
    let raw = |n: usize| -> Result<u32> {
        let op = o.get(n + 1).ok_or("missing vector operand")?;
        read_src(s, op, lane, if n == 0 { d } else { Src0::Plain })
    };
    let f = |n: usize| -> Result<u32> {
        let v = raw(n)?;
        if n == 0 && matches!(d, Src0::Zero) && (m.abs | m.neg) & 1 != 0 {
            return Err("DPP bound-control zero with src0 modifiers: order not specified".into());
        }
        Ok(sgn(v, m.abs >> n & 1 != 0, m.neg >> n & 1 != 0))
    };
    let fnum = |n: usize| -> Result<f32> { num(f(n)?) };
    // Literal constant of fmamk/fmaak: VOP2 keeps it in `Inst::literal`, VOPD as an operand.
    let lit = |pos: usize| -> Result<u32> {
        if o.len() > 3 {
            raw(pos)
        } else {
            i.literal.ok_or_else(|| "missing literal constant".into())
        }
    };
    let (class, value) = match stem(name) {
        "v_mov_b32" | "v_dual_mov_b32" => (Class::Int, raw(0)?),
        "v_cndmask_b32" | "v_dual_cndmask_b32" => {
            let mask = if o.len() > 3 { s.read(&o[3], lane, 0)? } else { s.vcc };
            let pick = if mask >> lane & 1 != 0 { 1 } else { 0 };
            (Class::Select, f(pick)?)
        }
        "v_add_nc_u32" | "v_dual_add_nc_u32" => (Class::Int, raw(0)?.wrapping_add(raw(1)?)),
        "v_sub_nc_u32" => (Class::Int, raw(0)?.wrapping_sub(raw(1)?)),
        "v_subrev_nc_u32" => (Class::Int, raw(1)?.wrapping_sub(raw(0)?)),
        "v_add3_u32" => (Class::Int, raw(0)?.wrapping_add(raw(1)?).wrapping_add(raw(2)?)),
        "v_and_b32" | "v_dual_and_b32" => (Class::Int, raw(0)? & raw(1)?),
        "v_or_b32" => (Class::Int, raw(0)? | raw(1)?),
        "v_xor_b32" => (Class::Int, raw(0)? ^ raw(1)?),
        "v_lshlrev_b32" | "v_dual_lshlrev_b32" => (Class::Int, raw(1)? << (raw(0)? & 31)),
        "v_lshrrev_b32" => (Class::Int, raw(1)? >> (raw(0)? & 31)),
        "v_ashrrev_i32" => (Class::Int, ((raw(1)? as i32) >> (raw(0)? & 31)) as u32),
        "v_lshl_add_u32" => (Class::Int, (raw(0)? << (raw(1)? & 31)).wrapping_add(raw(2)?)),
        "v_lshl_or_b32" => (Class::Int, (raw(0)? << (raw(1)? & 31)) | raw(2)?),
        "v_mul_lo_u32" => (Class::Int, raw(0)?.wrapping_mul(raw(1)?)),
        "v_mul_hi_u32" => (Class::Int, (u64::from(raw(0)?) * u64::from(raw(1)?) >> 32) as u32),
        "v_mul_hi_i32" => (Class::Int, ((i64::from(raw(0)? as i32) * i64::from(raw(1)? as i32)) >> 32) as u32),
        "v_mul_u32_u24" => (Class::Int, (raw(0)? & 0xff_ffff).wrapping_mul(raw(1)? & 0xff_ffff)),
        "v_mad_u32_u24" => (Class::Int, (raw(0)? & 0xff_ffff).wrapping_mul(raw(1)? & 0xff_ffff).wrapping_add(raw(2)?)),
        "v_min_i32" => (Class::Int, (raw(0)? as i32).min(raw(1)? as i32) as u32),
        "v_max_i32" => (Class::Int, (raw(0)? as i32).max(raw(1)? as i32) as u32),
        "v_min_u32" => (Class::Int, raw(0)?.min(raw(1)?)),
        "v_max_u32" => (Class::Int, raw(0)?.max(raw(1)?)),
        "v_bfe_u32" => {
            let width = raw(2)? & 31;
            (Class::Int, if width == 0 { 0 } else { (raw(0)? >> (raw(1)? & 31)) & ((1u32 << width) - 1) })
        }
        "v_perm_b32" => {
            // BYTE_PERMUTE({S0,S1}, sel): 0..7 bytes, 8..11 sign of bytes 1/3/5/7, 12 zero, >=13 0xff.
            let (a, b, c) = (raw(0)?, raw(1)?, raw(2)?);
            let bytes = (u64::from(b) | (u64::from(a) << 32)).to_le_bytes();
            let sign = |byte: u8| if byte & 0x80 != 0 { 0xff } else { 0 };
            let mut out = 0u32;
            for n in 0..4 {
                let byte = match (c >> (8 * n)) & 255 {
                    sel @ 0..=7 => bytes[sel as usize],
                    8 => sign(bytes[1]),
                    9 => sign(bytes[3]),
                    10 => sign(bytes[5]),
                    11 => sign(bytes[7]),
                    12 => 0,
                    _ => 0xff,
                };
                out |= u32::from(byte) << (8 * n);
            }
            (Class::Int, out)
        }
        "v_cvt_f32_u32" => (Class::F32, (raw(0)? as f32).to_bits()),
        "v_cvt_f32_i32" => (Class::F32, (raw(0)? as i32 as f32).to_bits()),
        // Saturating, truncating, NaN -> 0 (ISA: "NAN is converted to 0").
        "v_cvt_i32_f32" => (Class::FloatToInt, convert::f32_to_i32(f(0)?)),
        "v_cvt_u32_f32" => (Class::FloatToInt, convert::f32_to_u32(f(0)?)),
        "v_rndne_f32" => {
            let bits = f(0)?;
            let x = num(bits)?;
            let r = x.round_ties_even();
            if bits >> 31 != 0 && r == 0.0 && arch != Arch::Gfx1151 {
                return Err("v_rndne_f32 of a negative input rounding to zero: pseudocode gives +0, IEEE gives -0; only gfx1151 is measured (-0)".into());
            }
            (Class::F32, r.to_bits())
        }
        "v_cvt_f32_f16" => {
            let h = half_src(s, o.get(1).ok_or("missing vector operand")?, i, lane, 0)?;
            let h = sgn(u32::from(h) << 16, m.abs & 1 != 0, m.neg & 1 != 0) >> 16;
            (Class::F32, f16_in(h as u16)?)
        }
        "v_cvt_f16_f32" => {
            plain(i)?;
            if !matches!(o[0], Operand::Half(..)) {
                return Err("16-bit destination without a true16 half selector: upper-half policy not specified".into());
            }
            (Class::Own, u32::from(convert::f32_to_f16(fnum(0)?.to_bits())))
        }
        "v_ldexp_f32" => (Class::F32, convert::ldexp_f32(fnum(0)?.to_bits(), raw(1)? as i32)),
        "v_add_f32" | "v_dual_add_f32" => (Class::F32, fin(fnum(0)? + fnum(1)?)?),
        "v_sub_f32" | "v_dual_sub_f32" => (Class::F32, fin_invalid(arch, fnum(0)? - fnum(1)?)?),
        "v_subrev_f32" | "v_dual_subrev_f32" => (Class::F32, fin_invalid(arch, fnum(1)? - fnum(0)?)?),
        "v_mul_f32" | "v_dual_mul_f32" => (Class::F32, fin_invalid(arch, fnum(0)? * fnum(1)?)?),
        "v_fma_f32" => (Class::F32, fin(fnum(0)?.mul_add(fnum(1)?, fnum(2)?))?),
        "v_fmac_f32" | "v_dual_fmac_f32" => {
            let acc = num(s.read(&o[0], lane, 0)?)?;
            (Class::F32, fin(fnum(0)?.mul_add(fnum(1)?, acc))?)
        }
        // fmamk: D = S0 * K + S1; fmaak: D = S0 * S1 + K.
        "v_fmamk_f32" | "v_dual_fmamk_f32" => (Class::F32, fin(fnum(0)?.mul_add(num(lit(1)?)?, fnum(if o.len() > 3 { 2 } else { 1 })?))?),
        "v_fmaak_f32" | "v_dual_fmaak_f32" => (Class::F32, fin(fnum(0)?.mul_add(fnum(1)?, num(lit(2)?)?))?),
        "v_max_f32" => (Class::F32, legacy_select(f(0)?, f(1)?, true)?),
        "v_min_f32" => (Class::F32, legacy_select(f(0)?, f(1)?, false)?),
        "v_max3_f32" => (Class::F32, legacy_select(legacy_select(f(0)?, f(1)?, true)?, f(2)?, true)?),
        "v_min3_f32" => (Class::F32, legacy_select(legacy_select(f(0)?, f(1)?, false)?, f(2)?, false)?),
        "v_max_num_f32" | "v_dual_max_num_f32" => (Class::F32, num_select(f(0)?, f(1)?, true)),
        "v_min_num_f32" | "v_dual_min_num_f32" => (Class::F32, num_select(f(0)?, f(1)?, false)),
        "v_max3_num_f32" => (Class::F32, num_select(num_select(f(0)?, f(1)?, true), f(2)?, true)),
        "v_min3_num_f32" => (Class::F32, num_select(num_select(f(0)?, f(1)?, false), f(2)?, false)),
        "v_div_fmas_f32" => {
            if s.vcc & (1 << lane) != 0 {
                return Err("v_div_fmas_f32 with VCC set: ISA says 2.0F**32 while v_div_scale_f32 scales by 2**64; hardware-verified factor required".into());
            }
            (Class::F32, fin(fnum(0)?.mul_add(fnum(1)?, fnum(2)?))?)
        }
        "v_div_fixup_f32" => {
            let (q, den, nu) = (f(0)?, f(1)?, f(2)?);
            let (da, na) = (den & 0x7fff_ffff, nu & 0x7fff_ffff);
            let sign = (den ^ nu) & 0x8000_0000;
            let r = if na > INF {
                nu | 0x0040_0000
            } else if da > INF {
                den | 0x0040_0000
            } else if (da == 0 && na == 0) || (da == INF && na == INF) {
                0xffc0_0000
            } else if da == 0 || na == INF {
                sign | INF
            } else if da == INF || na == 0 {
                sign
            } else if ((na >> 23) as i32) - ((da >> 23) as i32) < -150 {
                return Err("v_div_fixup_f32 underflow branch: UNDERFLOW_F32 is not defined by the ISA".into());
            } else {
                // exponent(S1) == 255 is unreachable here: NaN and infinity were handled above.
                sign | (q & 0x7fff_ffff)
            };
            (Class::F32, r)
        }
        "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
            let [a, b, c] = mix_in(s, o, i, lane)?;
            let mut r = a.mul_add(b, c);
            if r.is_nan() {
                return Err(NAN_OUT.into());
            }
            if m.clamp {
                r = clamp01(r);
            }
            let bits = r.to_bits();
            (Class::Own, if stem(name) == "v_fma_mix_f32" { bits } else { u32::from(convert::f32_to_f16(bits)) })
        }
        "v_exp_f32" | "v_rcp_f32" | "v_rcp_iflag_f32" | "v_log_f32" | "v_rsq_f32" | "v_sqrt_f32" => {
            if !matches!(arch,Arch::Gfx1151|Arch::Gfx1201) {return Err(format!("unqualified numerical architecture {arch:?}"));}
            let model=match stem(name) {
                "v_exp_f32"=>super::trans::exp_f32,"v_rcp_f32"=>super::trans::rcp_f32,
                "v_rcp_iflag_f32"=>super::trans::rcp_iflag_f32,"v_log_f32"=>super::trans::log_f32,
                "v_rsq_f32"=>super::trans::rsq_f32,_=>super::trans::sqrt_f32,
            };
            (Class::F32,model(arch,f(0)?))
        },
        _ => return Err(format!("{UNSUPPORTED} {name}")),
    };
    let value = match class {
        Class::Int => {
            plain(i)?;
            value
        }
        Class::Select => {
            if m.omod != Omod::None || m.clamp {
                return Err("omod/clamp on v_cndmask_b32 is not modeled".into());
            }
            value
        }
        Class::FloatToInt => {
            if m.omod != Omod::None {
                return Err("omod on an integer-result opcode is not modeled".into());
            }
            value
        }
        Class::F32 => out_f32(arch, value, i)?,
        Class::Own => value,
    };
    Ok(match stem(name) {
        "v_fma_mixlo_f16" => Val::Low16(value),
        "v_fma_mixhi_f16" => Val::High16(value),
        _ => Val::Dword(value),
    })
}

fn cvt_pk_f32_fp8(i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let mut out = [None; 32];
    for lane in active(s) {
        let h = half_src(s, &o[1], i, lane, 0)?;
        let mut pair = [0u32; 2];
        for (n, slot) in pair.iter_mut().enumerate() {
            let byte = (h >> (8 * n)) as u8;
            if byte & 0x7f == 0x7f {
                return Err("FP8 NaN to f32: NaN payload not specified".into());
            }
            *slot = convert::fp8_to_f32(byte);
        }
        out[lane] = Some(pair);
    }
    for (lane, pair) in out.into_iter().enumerate() {
        if let Some(pair) = pair {
            s.put(&o[0], lane, 0, pair[0])?;
            s.put(&o[0], lane, 1, pair[1])?;
        }
    }
    Ok(())
}

/// `v_permlane16_b32` / `v_permlanex16_b32`. OPSEL[0] is fetch-inactive, OPSEL[1] bound
/// control: an inactive source lane disables the write, reads zero (BC) or is read (FI).
fn permlane(name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let (o, m) = (&i.operands, &i.mods);
    if m.abs | m.neg != 0 || m.omod != Omod::None || m.clamp || m.op_sel & 0b1100 != 0 {
        return Err(format!("{name}: abs/neg/omod/clamp and OPSEL[2]/OPSEL[3] are not modeled (only OPSEL[0]=FI, OPSEL[1]=BC)"));
    }
    let sel = u64::from(s.read(&o[2], 0, 0)?) | (u64::from(s.read(&o[3], 0, 0)?) << 32);
    let cross = name == "v_permlanex16_b32";
    let mut vals = [None; 32];
    for lane in active(s) {
        let row = if cross { (lane ^ 16) & 16 } else { lane & 16 };
        let src = row | ((sel >> (4 * (lane & 15))) & 15) as usize;
        vals[lane] = if i.mods.op_sel & 1 != 0 || s.exec & (1 << src) != 0 {
            Some(s.read(&o[1], src, 0)?)
        } else if i.mods.op_sel & 2 != 0 {
            Some(0)
        } else {
            None
        };
    }
    for (lane, v) in vals.into_iter().enumerate() {
        if let Some(v) = v {
            s.put(&o[0], lane, 0, v)?;
        }
    }
    Ok(())
}

/// `v_div_scale_f32`: chain of the RDNA4 pseudocode, restricted to its proven domain.
fn div_scale(arch: Arch, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let m = &i.mods;
    let ldexp64 = |x: u32| -> Result<u32> {
        if is_nan(x) {
            return Err(NAN_IN.into());
        }
        Ok(convert::ldexp_f32(x, 64))
    };
    let mut out = [None; 32];
    let mut vcc = 0u32;
    for lane in active(s) {
        let src = |n: usize| -> Result<u32> { Ok(sgn(s.read(&o[2 + n], lane, 0)?, m.abs >> n & 1 != 0, m.neg >> n & 1 != 0)) };
        let (s0, s1, s2) = (src(0)?, src(1)?, src(2)?);
        let (e1, e2) = ((s1 >> 23 & 255) as i32, (s2 >> 23 & 255) as i32);
        if s1 & 0x7fff_ffff == 0 || s2 & 0x7fff_ffff == 0 {
            return Err("v_div_scale_f32 with a zero numerator/denominator yields NAN.f32: encoding not specified".into());
        }
        if is_nan(s1) || is_nan(s2) {
            return Err(NAN_IN.into());
        }
        if e1 >= 253 {
            return Err("v_div_scale_f32: `1.0/S1 == DENORM.f64` is ambiguous when |S1| >= 2^126".into());
        }
        let mut flag = false;
        let d = if e2 - e1 >= 96 {
            flag = true;
            if s0 == s1 { ldexp64(s0)? } else { s0 }
        } else if e1 == 0 {
            ldexp64(s0)?
        } else {
            let qf = f32::from_bits(s2) / f32::from_bits(s1);
            let qd = f64::from(f32::from_bits(s2)) / f64::from(f32::from_bits(s1));
            let denorm32 = qf != 0.0 && qf.abs() < f32::MIN_POSITIVE;
            let denorm64 = qd.abs() < f64::from(f32::MIN_POSITIVE);
            if denorm32 != denorm64 || qd.abs() == f64::from(f32::MIN_POSITIVE) {
                return Err("v_div_scale_f32: quotient lies on the denormal boundary; hardware rounding not specified".into());
            }
            if denorm32 {
                flag = true;
                if f32::from_bits(s0) == f32::from_bits(s2) { ldexp64(s0)? } else { s0 }
            } else if e2 <= 23 {
                ldexp64(s0)?
            } else {
                s0
            }
        };
        if flag {
            vcc |= 1 << lane;
        }
        out[lane] = Some(out_f32(arch, d, i)?);
    }
    for (lane, v) in out.into_iter().enumerate() {
        if let Some(v) = v {
            s.put(&o[0], lane, 0, v)?;
        }
    }
    s.put(&o[1], 0, 0, vcc)
}

#[derive(Clone, Copy)]
enum Pred {
    F,
    Tru,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Lg,
    O,
    U,
    Nge,
    Nlg,
    Ngt,
    Nle,
    Neq,
    Nlt,
    Class,
}

fn pred(p: &str) -> Option<Pred> {
    Some(match p {
        "f" => Pred::F,
        "tru" => Pred::Tru,
        "eq" => Pred::Eq,
        "ne" => Pred::Ne,
        "lt" => Pred::Lt,
        "le" => Pred::Le,
        "gt" => Pred::Gt,
        "ge" => Pred::Ge,
        "lg" => Pred::Lg,
        "o" => Pred::O,
        "u" => Pred::U,
        "nge" => Pred::Nge,
        "nlg" => Pred::Nlg,
        "ngt" => Pred::Ngt,
        "nle" => Pred::Nle,
        "neq" => Pred::Neq,
        "nlt" => Pred::Nlt,
        "class" => Pred::Class,
        _ => return None,
    })
}

fn ord_test<T: Ord>(p: Pred, a: T, b: T) -> Option<bool> {
    Some(match p {
        Pred::F => false,
        Pred::Tru => true,
        Pred::Eq => a == b,
        Pred::Ne => a != b,
        Pred::Lt => a < b,
        Pred::Le => a <= b,
        Pred::Gt => a > b,
        Pred::Ge => a >= b,
        _ => return None,
    })
}

fn float_test(p: Pred, a: f32, b: f32) -> Option<bool> {
    Some(match p {
        Pred::F => false,
        Pred::Tru => true,
        Pred::Eq => a == b,
        Pred::Lt => a < b,
        Pred::Le => a <= b,
        Pred::Gt => a > b,
        Pred::Ge => a >= b,
        Pred::Lg => a < b || a > b,
        Pred::O => !a.is_nan() && !b.is_nan(),
        Pred::U => a.is_nan() || b.is_nan(),
        Pred::Nge => !(a >= b),
        Pred::Nlg => !(a < b || a > b),
        Pred::Ngt => !(a > b),
        Pred::Nle => !(a <= b),
        Pred::Neq => !(a == b),
        Pred::Nlt => !(a < b),
        _ => return None,
    })
}

/// `v_cmp_*` / `v_cmpx_*` over u32/i32/f32 and `class_f32`. Lanes that are inactive or
/// disabled by DPP write a zero bit.
fn compare(name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let m = &i.mods;
    let cmpx = name.starts_with("v_cmpx_");
    let rest = name.strip_prefix("v_cmpx_").or_else(|| name.strip_prefix("v_cmp_")).ok_or("not a compare")?;
    let rest = rest.strip_suffix("_e64").or_else(|| rest.strip_suffix("_e32")).or_else(|| rest.strip_suffix("_dpp")).unwrap_or(rest);
    let (p, ty) = rest.rsplit_once('_').ok_or_else(|| format!("unsupported compare {name}"))?;
    let p = pred(p).ok_or_else(|| format!("unsupported compare {name}"))?;
    let float = ty == "f32";
    if !float && ty != "u32" && ty != "i32" {
        return Err(format!("unsupported compare {name}"));
    }
    if float != matches!(p, Pred::Lg | Pred::O | Pred::U | Pred::Nge | Pred::Nlg | Pred::Ngt | Pred::Nle | Pred::Neq | Pred::Nlt | Pred::Class)
        && !matches!(p, Pred::F | Pred::Tru | Pred::Eq | Pred::Lt | Pred::Le | Pred::Gt | Pred::Ge | Pred::Ne)
        || matches!(p, Pred::Ne) && float
        || matches!(p, Pred::Class) && !float
    {
        return Err(format!("unsupported compare {name}"));
    }
    if m.omod != Omod::None || (!float && m.neg | m.abs != 0) {
        return Err("omod, or neg/abs on an integer compare, is not modeled".into());
    }
    if o.len() < 2 {
        return Err("missing compare operand".into());
    }
    let (a_op, b_op) = (&o[o.len() - 2], &o[o.len() - 1]);
    let dpp = m.dpp;
    let fi = dpp_fi(i);
    let mut mask = 0u32;
    for lane in active(s) {
        let d = match dpp {
            Some(d) => match dpp_plan(&d, fi, s, lane)? {
                Some(d) => d,
                None => continue,
            },
            None => Src0::Plain,
        };
        let a = read_src(s, a_op, lane, d)?;
        let b = s.read(b_op, lane, 0)?;
        let yes = if let Pred::Class = p {
            // Modifiers apply to S0 only; S1 is the integer class mask.
            let class = f32_class(sgn(a, m.abs & 1 != 0, m.neg & 1 != 0));
            Some(b & (1 << class) != 0)
        } else if float {
            let x = f32::from_bits(sgn(a, m.abs & 1 != 0, m.neg & 1 != 0));
            let y = f32::from_bits(sgn(b, m.abs & 2 != 0, m.neg & 2 != 0));
            float_test(p, x, y)
        } else if ty == "i32" {
            ord_test(p, a as i32, b as i32)
        } else {
            ord_test(p, a, b)
        };
        if yes.ok_or_else(|| format!("unsupported compare {name}"))? {
            mask |= 1 << lane;
        }
    }
    if cmpx {
        s.exec = mask;
        Ok(())
    } else if o.len() >= 3 {
        s.put(&o[0], 0, 0, mask)
    } else {
        s.vcc = mask;
        Ok(())
    }
}

/// `v_cmp_class_f32` class index: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denorm, 5 -0,
/// 6 +0, 7 +denorm, 8 +normal, 9 +inf.
fn f32_class(a: u32) -> u32 {
    let abs = a & 0x7fff_ffff;
    let neg = a >> 31 != 0;
    if abs > INF {
        u32::from(a & 0x0040_0000 != 0)
    } else if abs == INF {
        if neg { 2 } else { 9 }
    } else if abs == 0 {
        if neg { 5 } else { 6 }
    } else if abs < 0x0080_0000 {
        if neg { 4 } else { 7 }
    } else if neg {
        3
    } else {
        8
    }
}

fn scalar(name: &str, i: &Inst, s: &mut State) -> Result<()> {
    let o = &i.operands;
    let rd = |s: &State, idx: usize| -> Result<u32> { s.read(o.get(idx).ok_or("missing scalar operand")?, 0, 0) };
    if let Some(rest) = name.strip_prefix("s_cmp_") {
        let (p, ty) = rest.split_once('_').ok_or_else(|| format!("unsupported scalar compare {name}"))?;
        let p = match p {
            "eq" => Pred::Eq,
            "lg" => Pred::Ne,
            "gt" => Pred::Gt,
            "ge" => Pred::Ge,
            "lt" => Pred::Lt,
            "le" => Pred::Le,
            _ => return Err(format!("unsupported scalar compare {name}")),
        };
        let (a, b) = (rd(s, 0)?, rd(s, 1)?);
        s.scc = match ty {
            "u32" => ord_test(p, a, b),
            "i32" => ord_test(p, a as i32, b as i32),
            _ => return Err(format!("unsupported scalar compare {name}")),
        }
        .ok_or_else(|| format!("unsupported scalar compare {name}"))?;
        return Ok(());
    }
    if name == "s_bitcmp1_b32" || name == "s_bitcmp0_b32" {
        let bit = rd(s, 0)? >> (rd(s, 1)? & 31) & 1;
        s.scc = (bit == 1) == (name == "s_bitcmp1_b32");
        return Ok(());
    }
    let a = rd(s, 1)?;
    let b = if o.len() > 2 { rd(s, 2)? } else { 0 };
    let nonzero = |v: u32| (v, Some(v != 0));
    let (value, flag) = match name {
        "s_mov_b32" => (a, None),
        "s_cselect_b32" => (if s.scc { a } else { b }, None),
        "s_add_u32" | "s_add_co_u32" | "s_addc_u32" | "s_add_co_ci_u32" => {
            let carry = u64::from((name == "s_addc_u32" || name == "s_add_co_ci_u32") && s.scc);
            let sum = u64::from(a) + u64::from(b) + carry;
            (sum as u32, Some(sum >> 32 != 0))
        }
        "s_sub_u32" | "s_sub_co_u32" | "s_subb_u32" | "s_sub_co_ci_u32" => {
            let borrow = u64::from((name == "s_subb_u32" || name == "s_sub_co_ci_u32") && s.scc);
            (a.wrapping_sub(b).wrapping_sub(borrow as u32), Some(u64::from(b) + borrow > u64::from(a)))
        }
        "s_add_i32" | "s_add_co_i32" => {
            let (v, f) = (a as i32).overflowing_add(b as i32);
            (v as u32, Some(f))
        }
        "s_sub_i32" | "s_sub_co_i32" => {
            let (v, f) = (a as i32).overflowing_sub(b as i32);
            (v as u32, Some(f))
        }
        "s_mul_i32" => (a.wrapping_mul(b), None),
        "s_mulk_i32" => (s.read(&o[0], 0, 0)?.wrapping_mul(a), None),
        "s_mul_hi_u32" => ((u64::from(a) * u64::from(b) >> 32) as u32, None),
        "s_and_b32" => nonzero(a & b),
        "s_or_b32" => nonzero(a | b),
        "s_xor_b32" => nonzero(a ^ b),
        "s_andn2_b32" => nonzero(a & !b),
        "s_orn2_b32" => nonzero(a | !b),
        "s_not_b32" => nonzero(!a),
        "s_lshl_b32" => nonzero(a << (b & 31)),
        "s_lshr_b32" => nonzero(a >> (b & 31)),
        "s_ashr_i32" => nonzero(((a as i32) >> (b & 31)) as u32),
        "s_min_i32" => ((a as i32).min(b as i32) as u32, Some((a as i32) < (b as i32))),
        "s_max_i32" => ((a as i32).max(b as i32) as u32, Some((a as i32) >= (b as i32))),
        "s_min_u32" => (a.min(b), Some(a < b)),
        "s_max_u32" => (a.max(b), Some(a >= b)),
        "s_ctz_i32_b32" => (if a == 0 { u32::MAX } else { a.trailing_zeros() }, None),
        _ => return Err(format!("unsupported scalar opcode {name}")),
    };
    s.put(&o[0], 0, 0, value)?;
    if let Some(f) = flag {
        s.scc = f;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        operand::{Half, Special},
        reg::{Kind, RegRef},
    };

    // Update together with `State` in mod.rs when fields change.
    fn state() -> State {
        State { s: [0; 106], t: [0; 16], v: Box::new([[0; 32]; 256]), exec: u32::MAX, vcc: 0, scc: false, m0: 0, pc: 0, ended: false, barrier: None, signals: 0 }
    }
    fn reg(kind: Kind, base: u16, len: u8) -> Operand {
        Operand::Reg(RegRef { kind, base, len })
    }
    fn v(base: u16) -> Operand {
        reg(Kind::V, base, 1)
    }
    fn half(base: u16, h: Half) -> Operand {
        Operand::Half(RegRef { kind: Kind::V, base, len: 1 }, h)
    }
    fn lit(bits: u32) -> Operand {
        Operand::Literal(bits)
    }
    fn p2(k: i32) -> u32 {
        ((127 + k) as u32) << 23
    }
    fn insn(name: &str, ops: Vec<Operand>) -> Inst {
        let r = crate::isa::gfx12().iter().find(|r| r.name == name).unwrap();
        Inst::from_parts(Arch::Gfx1201, r.op, r.form, Default::default(), ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap()
    }
    fn run(name: &str, i: &Inst, s: &mut State) -> Result<()> {
        execute(Arch::Gfx1201, name, i, s)
    }

    // DPP16: row_shl/quad_perm/row_share/xmask/mirror routing, FI=0 inactive sources, BC, row/bank masks.
    #[test]
    fn dpp_routes_lanes_and_applies_masks_bound_ctrl_and_inactive_sources() {
        let mut i = insn("v_mov_b32_dpp", vec![v(1), v(0)]);
        let dpp = |ctrl, row_mask, bank_mask, bound_ctrl| Some(Dpp { ctrl, row_mask, bank_mask, bound_ctrl });
        let fresh = || {
            let mut st = state();
            for l in 0..32 {
                st.v[0][l] = 100 + l as u32;
                st.v[1][l] = 7;
            }
            st
        };
        // row_shl:1 with lane 5 inactive: lanes 15/31 are out of range, lane 4 reads inactive lane 5.
        i.mods.dpp = dpp(0x101, 0xf, 0xf, false);
        let mut st = fresh();
        st.exec = !(1 << 5);
        run("v_mov_b32_dpp", &i, &mut st).unwrap();
        for l in 0..32 {
            let keep = l == 5 || l % 16 == 15 || l == 4;
            assert_eq!(st.v[1][l], if keep { 7 } else { 101 + l as u32 }, "bc=0 lane {l}");
        }
        i.mods.dpp = dpp(0x101, 0xf, 0xf, true);
        let mut st = fresh();
        st.exec = !(1 << 5);
        run("v_mov_b32_dpp", &i, &mut st).unwrap();
        for l in 0..32 {
            let want = if l == 5 { 7 } else if l % 16 == 15 || l == 4 { 0 } else { 101 + l as u32 };
            assert_eq!(st.v[1][l], want, "bc=1 lane {l}");
        }
        // row_mask keeps row 0, bank_mask 0b1101 keeps lanes 4..7 and 20..23.
        i.mods.dpp = dpp(0x101, 0b10, 0b1101, true);
        let mut st = fresh();
        run("v_mov_b32_dpp", &i, &mut st).unwrap();
        for l in 0..32 {
            let keep = l < 16 || (l / 4) % 4 == 1;
            let want = if keep { 7 } else if l % 16 == 15 { 0 } else { 101 + l as u32 };
            assert_eq!(st.v[1][l], want, "masked lane {l}");
        }
        // Controls that never leave the row: source lane of each destination lane.
        let cases: [(u16, fn(usize) -> usize); 5] = [
            (0x1b, |l| (l & !3) | (3 - (l & 3))),   // quad_perm [3,2,1,0]
            (0x153, |l| (l & !15) + 3),             // row_share:3
            (0x161, |l| (l & !15) + ((l & 15) ^ 1)), // row_xmask:1
            (0x140, |l| (l & !15) + 15 - (l & 15)),  // row_mirror
            (0x123, |l| (l & !15) + (((l & 15) + 13) & 15)), // row_ror:3
        ];
        for (ctrl, src) in cases {
            i.mods.dpp = dpp(ctrl, 0xf, 0xf, false);
            let mut st = fresh();
            run("v_mov_b32_dpp", &i, &mut st).unwrap();
            for l in 0..32 {
                assert_eq!(st.v[1][l], 100 + src(l) as u32, "ctrl {ctrl:#x} lane {l}");
            }
        }
        i.mods.dpp = dpp(0x130, 0xf, 0xf, false);
        assert!(run("v_mov_b32_dpp", &i, &mut fresh()).is_err(), "wave_shl is not an RDNA control");
    }

    // VOPD reads every input before either half writes.
    #[test]
    fn vopd_halves_read_old_registers() {
        let (x, y) = {
            let t = crate::isa::gfx12();
            (t.iter().find(|r| r.name == "v_dual_mul_f32").unwrap(), t.iter().find(|r| r.name == "v_dual_add_f32").unwrap())
        };
        let ops = vec![v(0), v(1), v(2), v(1), v(0), v(3)];
        let i = Inst::from_parts(Arch::Gfx1201, x.op, x.form, FormFields::Vopd { y_op: y.op, x_operands: 3 }, ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap();
        let mut st = state();
        st.exec = 0b01;
        for l in 0..2 {
            st.v[0][l] = 2f32.to_bits();
            st.v[1][l] = 3f32.to_bits();
            st.v[2][l] = 4f32.to_bits();
            st.v[3][l] = 0.5f32.to_bits();
        }
        run("v_dual_mul_f32", &i, &mut st).unwrap();
        assert_eq!((st.v[0][0], st.v[1][0]), (12f32.to_bits(), 2.5f32.to_bits()), "X=v1*v2 and Y=old v0+v3");
        assert_eq!((st.v[0][1], st.v[1][1]), (2f32.to_bits(), 3f32.to_bits()), "inactive lane untouched");
    }

    // Permlane supports only FI=OPSEL[0] and BC=OPSEL[1]; every other modifier errors, even with EXEC=0.
    #[test]
    fn permlane_rejects_unsupported_modifiers_and_reserved_opsel() {
        // gfx12 has no v_permlane16_b32: build it from the gfx1100 table, with SGPR selectors.
        for (name, arch) in [("v_permlane16_b32", Arch::Gfx1100), ("v_permlanex16_b32", Arch::Gfx1201)] {
            let (table, ops) = if arch == Arch::Gfx1100 {
                (crate::isa::gfx1100(), vec![v(0), v(1), reg(Kind::S, 0, 1), reg(Kind::S, 1, 1)])
            } else {
                (crate::isa::gfx12(), vec![v(0), v(1), lit(0x76543210), lit(0xfedcba98)])
            };
            let r = table.iter().find(|r| r.name == name).unwrap();
            let base = Inst::from_parts(arch, r.op, r.form, Default::default(), ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap();
            let fresh = || {
                let mut st = state();
                st.s[0] = 0x7654_3210;
                st.s[1] = 0xfedc_ba98;
                for l in 0..32 {
                    st.v[1][l] = 100 + l as u32;
                }
                st
            };
            let run = |i: &Inst, s: &mut State| execute(arch, name, i, s);
            let mut legal=base.clone();legal.mods.op_sel=3;
            let mut st=fresh();run(&legal,&mut st).unwrap();
            for lane in 0..32 {
                let source=if name=="v_permlanex16_b32"{lane^16}else{lane};
                assert_eq!(st.v[0][lane],100+source as u32,"{name} FI+BC lane {lane}");
            }
            let bad: [(&str, fn(&mut Inst)); 8] = [
                ("abs", |i| i.mods.abs = 1),
                ("neg", |i| i.mods.neg = 0b100),
                ("omod", |i| i.mods.omod = Omod::Mul2),
                ("clamp", |i| i.mods.clamp = true),
                ("op_sel[2]", |i| i.mods.op_sel = 0b0100),
                ("op_sel[3]", |i| i.mods.op_sel = 0b1000),
                ("op_sel[3]+FI", |i| i.mods.op_sel = 0b1001),
                ("op_sel[2]+BC", |i| i.mods.op_sel = 0b0110),
            ];
            for (what, apply) in bad {
                for exec in [u32::MAX, 0] {
                    let mut i = base.clone();
                    apply(&mut i);
                    let mut st = fresh();
                    st.exec = exec;
                    let before = st.v[0];
                    assert!(run(&i, &mut st).is_err(), "{name}: {what} exec {exec:#x} must error");
                    assert_eq!(st.v[0], before, "{name}: {what} must not write");
                }
            }
        }
    }

    // Input neg/abs apply abs first; OMOD then CLAMP; -0 clamps to +0 and -0*OMOD = +0.
    #[test]
    fn float_modifiers_order_and_zero_signs() {
        let mut i = insn("v_fma_f32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.v[1][0] = 2f32.to_bits();
        st.v[2][0] = (-3f32).to_bits();
        st.v[3][0] = 10f32.to_bits();
        i.mods.neg = 0b001;
        i.mods.abs = 0b010;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 4.0, "-2 * |-3| + 10");
        i.mods.omod = Omod::Mul2;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 8.0);
        i.mods.omod = Omod::Div2;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 2.0);
        i.mods.clamp = true;
        i.mods.omod = Omod::Mul2;
        run("v_fma_f32", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 1.0, "clamp after omod");
        // fma(1, -0, -0) = -0: clamp and omod both give +0.
        let mut j = insn("v_fma_f32", vec![v(0), v(1), v(2), v(3)]);
        st.v[1][0] = 1f32.to_bits();
        st.v[2][0] = (-0f32).to_bits();
        st.v[3][0] = (-0f32).to_bits();
        run("v_fma_f32", &j, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0x8000_0000);
        j.mods.clamp = true;
        run("v_fma_f32", &j, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0);
        j.mods.clamp = false;
        j.mods.omod = Omod::Mul2;
        run("v_fma_f32", &j, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0);
        // A scaled result that flushes a non-zero denormal has an unspecified zero sign.
        st.v[1][0] = p2(-126);
        st.v[2][0] = 1f32.to_bits();
        st.v[3][0] = 0;
        j.mods.omod = Omod::Div2;
        assert!(run("v_fma_f32", &j, &mut st).is_err());
    }

    // NaN: documented maxNum rules are modeled; undocumented arithmetic NaN is a hard error.
    #[test]
    fn nan_rules_and_signed_zero_in_max_num() {
        let i = insn("v_max_num_f32_e32", vec![v(0), v(1), v(2)]);
        let mut st = state();
        let cases = [
            (0x7fc0_0001, 3f32.to_bits(), 3f32.to_bits()),
            (3f32.to_bits(), 0x7fc0_0001, 3f32.to_bits()),
            (0x7f80_0001, 0x7fc0_0002, 0x7fc0_0001),
            ((-0f32).to_bits(), 0f32.to_bits(), 0),
            (0f32.to_bits(), (-0f32).to_bits(), 0),
            (1f32.to_bits(), 2f32.to_bits(), 2f32.to_bits()),
        ];
        for (l, (a, b, _)) in cases.iter().enumerate() {
            st.v[1][l] = *a;
            st.v[2][l] = *b;
        }
        run("v_max_num_f32_e32", &i, &mut st).unwrap();
        for (l, (_, _, want)) in cases.iter().enumerate() {
            assert_eq!(st.v[0][l], *want, "lane {l}");
        }
        let add = insn("v_add_f32_e32", vec![v(0), v(1), v(2)]);
        st.v[1][0] = 0x7fc0_0000;
        let e = run("v_add_f32_e32", &add, &mut st).unwrap_err();
        assert!(e.contains("NaN"), "{e}");
        st.v[1][0] = f32::INFINITY.to_bits();
        st.v[2][0] = f32::NEG_INFINITY.to_bits();
        assert!(run("v_add_f32_e32", &add, &mut st).is_err(), "inf - inf default NaN encoding is unspecified");
    }

    // v_div_scale_f32 flag/scale decisions follow the RDNA4 pseudocode chain in order.
    #[test]
    fn div_scale_chain_and_vcc_mask() {
        let i = insn("v_div_scale_f32", vec![v(0), Operand::Special(Special::VccLo), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 0b11111;
        // (S0, S1 = denominator, S2 = numerator) per lane.
        let lanes = [
            (3f32.to_bits(), 3f32.to_bits(), 1f32.to_bits()),            // ordinary: pass S0, no flag
            (p2(-110), 3f32.to_bits(), p2(-110)),                        // exponent(S2) <= 23: scale S0 by 2^64
            (p2(-120), p2(10), p2(-120)),                                // quotient 2^-130 denormal, S0 = numerator
            (p2(10), p2(10), p2(-120)),                                  // same, S0 = denominator: flag only
            (p2(-100), p2(-100), p2(30)),                                // exponent gap 130 >= 96, S0 = denominator
        ];
        for (l, (a, b, c)) in lanes.iter().enumerate() {
            st.v[1][l] = *a;
            st.v[2][l] = *b;
            st.v[3][l] = *c;
        }
        run("v_div_scale_f32", &i, &mut st).unwrap();
        let want = [3f32.to_bits(), p2(-46), p2(-56), p2(10), p2(-36)];
        for l in 0..5 {
            assert_eq!(st.v[0][l], want[l], "lane {l}");
        }
        assert_eq!(st.vcc, 0b11100);
        // Unspecified or ambiguous domains are errors, not guesses.
        for (a, b, c) in [(1f32.to_bits(), 0, 1f32.to_bits()), (1f32.to_bits(), p2(127), 1f32.to_bits()), (1f32.to_bits(), 1f32.to_bits(), 0x7fc0_0000)] {
            let mut st = state();
            st.exec = 1;
            st.v[1][0] = a;
            st.v[2][0] = b;
            st.v[3][0] = c;
            assert!(run("v_div_scale_f32", &i, &mut st).is_err(), "{a:#x} {b:#x} {c:#x}");
        }
    }

    #[test]
    fn div_fmas_and_fixup_exceptional_cases() {
        let fmas = insn("v_div_fmas_f32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 2f32.to_bits();
        st.v[2][0] = 3f32.to_bits();
        st.v[3][0] = 1f32.to_bits();
        run("v_div_fmas_f32", &fmas, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 7.0);
        st.vcc = 1;
        assert!(run("v_div_fmas_f32", &fmas, &mut st).unwrap_err().contains("VCC"));

        let fix = insn("v_div_fixup_f32", vec![v(0), v(1), v(2), v(3)]);
        let q = (1f32 / 3f32).to_bits();
        let inf = f32::INFINITY.to_bits();
        // (quotient, denominator, numerator) -> result
        let cases: [(u32, u32, u32, u32); 9] = [
            (q, 3f32.to_bits(), 1f32.to_bits(), q),
            (q, (-3f32).to_bits(), 1f32.to_bits(), q | 0x8000_0000),
            (q, 0, 0, 0xffc0_0000),                                  // 0/0
            (q, inf, inf, 0xffc0_0000),                              // inf/inf
            (q, inf, 1f32.to_bits(), 0),                             // x/inf
            (q, inf | 0x8000_0000, 1f32.to_bits(), 0x8000_0000),     // x/-inf
            (q, 0, 1f32.to_bits(), inf),                             // x/0
            (q, 3f32.to_bits(), inf, inf),                           // inf/y
            (q, 3f32.to_bits(), 0x7f80_0001, 0x7fc0_0001),           // NaN numerator is quieted
        ];
        let mut st = state();
        st.exec = (1 << cases.len()) - 1;
        for (l, (a, b, c, _)) in cases.iter().enumerate() {
            st.v[1][l] = *a;
            st.v[2][l] = *b;
            st.v[3][l] = *c;
        }
        run("v_div_fixup_f32", &fix, &mut st).unwrap();
        for (l, (_, _, _, want)) in cases.iter().enumerate() {
            assert_eq!(st.v[0][l], *want, "lane {l}");
        }
        // exponent(S2) - exponent(S1) < -150 selects the undefined UNDERFLOW_F32.
        st.exec = 1;
        st.v[2][0] = p2(100);
        st.v[3][0] = p2(-60);
        assert!(run("v_div_fixup_f32", &fix, &mut st).is_err());
    }

    // BYTE_PERMUTE selectors 8..11 replicate the sign of bytes 1/3/5/7, 12 is 0, >= 13 is 0xff.
    #[test]
    fn perm_b32_selectors() {
        let i = insn("v_perm_b32", vec![v(0), v(1), v(2), v(3)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 0xaabb_ccdd; // S0: bytes 4..7
        st.v[2][0] = 0x1122_3380; // S1: bytes 0..3
        for (sel, want) in [(0x0c0d_0a09u32, 0x00ff_ff00u32), (0x0b08_0700, 0xff00_aa80), (0x0d0c_0302, 0xff00_1122)] {
            st.v[3][0] = sel;
            run("v_perm_b32", &i, &mut st).unwrap();
            assert_eq!(st.v[0][0], want, "sel {sel:#x}");
        }
    }

    // True16 sources: the half selector rides in the operand and OP_SEL is not applied twice.
    #[test]
    fn cvt_f32_f16_half_selector_neg_and_nan() {
        let mut i = insn("v_cvt_f32_f16_e64", vec![v(0), half(1, Half::Hi)]);
        i.mods.op_sel = 1;
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 0x3c00_4000;
        run("v_cvt_f32_f16_e64", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 1.0);
        i.mods.neg = 1;
        run("v_cvt_f32_f16_e64", &i, &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), -1.0);
        st.v[1][0] = 0x7e00_0000;
        assert!(run("v_cvt_f32_f16_e64", &i, &mut st).is_err());
        // v_cvt_f16_f32 only rewrites the selected half.
        let o = insn("v_cvt_f16_f32_e32", vec![half(2, Half::Hi), v(1)]);
        st.v[1][0] = 0.5f32.to_bits();
        st.v[2][0] = 0xdead_beef;
        run("v_cvt_f16_f32_e32", &o, &mut st).unwrap();
        assert_eq!(st.v[2][0], 0x3800_beef);
    }

    #[test]
    fn f32_to_integer_conversions_saturate_and_zero_nan() {
        let vals = [f32::NAN.to_bits(), f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits(), 3e9f32.to_bits(), (-1.9f32).to_bits(), 2.5f32.to_bits()];
        let (si, su) = (insn("v_cvt_i32_f32_e32", vec![v(0), v(1)]), insn("v_cvt_u32_f32_e32", vec![v(2), v(1)]));
        let mut st = state();
        st.exec = 0b111111;
        for (l, x) in vals.iter().enumerate() {
            st.v[1][l] = *x;
        }
        run("v_cvt_i32_f32_e32", &si, &mut st).unwrap();
        run("v_cvt_u32_f32_e32", &su, &mut st).unwrap();
        let want_i = [0, i32::MAX as u32, i32::MIN as u32, i32::MAX as u32, (-1i32) as u32, 2];
        let want_u = [0, u32::MAX, 0, 3_000_000_000, 0, 2];
        for l in 0..6 {
            assert_eq!((st.v[0][l], st.v[2][l]), (want_i[l], want_u[l]), "lane {l}");
        }
    }

    // fma_mix: NEG_HI is |x|, NEG negates after it, OPSEL_HI selects f16, OPSEL the high half.
    #[test]
    fn fma_mixlo_modifiers_and_destination_merge() {
        let mut i = insn("v_fma_mixlo_f16", vec![v(0), v(1), v(2), Operand::Inline(InlineConst::Integer(0))]);
        let mut st = state();
        st.exec = 1;
        st.v[0][0] = 0xdead_beef;
        st.v[1][0] = (-3f32).to_bits();
        st.v[2][0] = 0x3c00_4000; // low half 2.0, high half 1.0
        i.mods.op_sel_hi = 0b010;
        i.mods.neg_hi = 0b001;
        i.mods.neg_lo = 0b001;
        run("v_fma_mixlo_f16", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0xdead_c600, "-|-3| * 2 = -6, high half preserved");
        i.mods.op_sel = 0b010;
        run("v_fma_mixlo_f16", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0xdead_c200, "-|-3| * 1 = -3");
        i.mods.clamp = true;
        run("v_fma_mixlo_f16", &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 0xdead_0000, "clamp of a negative result is +0");
    }

    #[test]
    fn cvt_pk_f32_fp8_selects_half_and_rejects_nan() {
        let i = insn("v_cvt_pk_f32_fp8_e32", vec![reg(Kind::V, 10, 2), half(4, Half::Hi)]);
        let mut st = state();
        st.exec = 1;
        st.v[4][0] = 0xb838_0000; // high half: 0x38 = 1.0, 0xb8 = -1.0
        run("v_cvt_pk_f32_fp8_e32", &i, &mut st).unwrap();
        assert_eq!((st.v[10][0], st.v[11][0]), (1f32.to_bits(), (-1f32).to_bits()));
        st.v[4][0] = 0x007f_0000;
        assert!(run("v_cvt_pk_f32_fp8_e32", &i, &mut st).is_err());
    }

    // Source modifiers reach the class test; NaN compares follow the ISA predicates; inactive lanes are 0.
    #[test]
    fn compares_apply_modifiers_and_zero_inactive_lanes() {
        let mut class = insn("v_cmp_class_f32_e64", vec![reg(Kind::S, 10, 2), v(1), lit(0x200)]);
        let mut st = state();
        st.exec = 0b0111;
        st.v[1][0] = f32::NEG_INFINITY.to_bits();
        st.v[1][1] = f32::INFINITY.to_bits();
        st.v[1][2] = 1f32.to_bits();
        st.v[1][3] = f32::INFINITY.to_bits(); // inactive
        run("v_cmp_class_f32_e64", &class, &mut st).unwrap();
        assert_eq!(st.s[10], 0b010);
        class.mods.abs = 1;
        run("v_cmp_class_f32_e64", &class, &mut st).unwrap();
        assert_eq!(st.s[10], 0b011, "|-inf| is +inf");
        let nan = 0x7fc0_0000;
        for (name, want) in [("v_cmp_neq_f32_e64", 0b01), ("v_cmp_nlt_f32_e64", 0b11), ("v_cmp_lt_f32_e64", 0b00)] {
            let c = insn(name, vec![reg(Kind::S, 12, 2), v(1), v(2)]);
            st.v[1][0] = nan;
            st.v[1][1] = 1f32.to_bits();
            st.v[2][0] = 1f32.to_bits();
            st.v[2][1] = 1f32.to_bits();
            st.exec = 0b11;
            run(name, &c, &mut st).unwrap();
            assert_eq!(st.s[12], want, "{name}");
        }
    }

    // VOP2 fmamk/fmaak keep K in the instruction's literal word: fmamk = S0*K + V1, fmaak = S0*V1 + K.
    #[test]
    fn fmamk_and_fmaak_use_the_literal_constant() {
        let build = |name: &str| {
            let r = crate::isa::gfx12().iter().find(|r| r.name == name).unwrap();
            Inst::from_parts(Arch::Gfx1201, r.op, r.form, Default::default(), vec![v(0), v(1), v(2)].into_iter().collect(), Default::default(), Some(3f32.to_bits()), Default::default()).unwrap()
        };
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 2f32.to_bits();
        st.v[2][0] = 1f32.to_bits();
        run("v_fmamk_f32", &build("v_fmamk_f32"), &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 7.0);
        run("v_fmaak_f32", &build("v_fmaak_f32"), &mut st).unwrap();
        assert_eq!(f32::from_bits(st.v[0][0]), 5.0);
    }

    #[test]
    fn readfirstlane_forces_lane_zero_when_exec_is_empty() {
        let i = insn("v_readfirstlane_b32", vec![reg(Kind::S, 5, 1), v(0)]);
        let mut st = state();
        st.v[0][0] = 42;
        st.v[0][2] = 9;
        st.exec = 0;
        run("v_readfirstlane_b32", &i, &mut st).unwrap();
        assert_eq!(st.s[5], 42);
        st.exec = 0b100;
        run("v_readfirstlane_b32", &i, &mut st).unwrap();
        assert_eq!(st.s[5], 9);
    }

    fn insn_1151(name: &str, ops: Vec<Operand>) -> Inst {
        let r = crate::isa::gfx1151().iter().find(|r| r.name == name).unwrap();
        Inst::from_parts(Arch::Gfx1151, r.op, r.form, Default::default(), ops.into_iter().collect(), Default::default(), None, Default::default()).unwrap()
    }

    // gfx1151 measured v_rndne_f32 (gfx1151-alu-edges.log): negative inputs rounding to zero give -0,
    // positive ones +0, normal ties are even. gfx1201 is not measured and stays a hard error.
    #[test]
    fn rndne_gfx1151_signed_zero_and_ties_even_while_gfx1201_stays_unqualified() {
        let name = "v_rndne_f32_e32";
        let i = insn_1151(name, vec![v(0), v(1)]);
        let mut st = state();
        st.exec = 1;
        let neg_zero = 0x8000_0000;
        for (x, want) in [
            ((-0.25f32).to_bits(), neg_zero),
            ((-0.5f32).to_bits(), neg_zero),
            (0x8000_0001, neg_zero),
            (0x8000_0000, neg_zero),
            (0.25f32.to_bits(), 0),
            (0.5f32.to_bits(), 0),
            (0x0000_0001, 0),
            (0, 0),
            (2.5f32.to_bits(), 2.0f32.to_bits()),
            (3.5f32.to_bits(), 4.0f32.to_bits()),
            ((-2.5f32).to_bits(), (-2.0f32).to_bits()),
            ((-1.5f32).to_bits(), (-2.0f32).to_bits()),
        ] {
            st.v[1][0] = x;
            execute(Arch::Gfx1151, name, &i, &mut st).unwrap();
            assert_eq!(st.v[0][0], want, "rndne({x:#010x})");
        }
        let i = insn(name, vec![v(0), v(1)]);
        let mut st = state();
        st.exec = 1;
        st.v[1][0] = 2.5f32.to_bits();
        run(name, &i, &mut st).unwrap();
        assert_eq!(st.v[0][0], 2.0f32.to_bits());
        st.v[1][0] = (-0.3f32).to_bits();
        assert!(run(name, &i, &mut st).is_err());
    }

    // gfx1151 measured default NaN 0xffc00000 for invalid sub/subrev/mul of non-NaN operands; valid
    // arithmetic is unchanged, NaN operands and every unmeasured arch (gfx1201) stay hard errors.
    #[test]
    fn invalid_sub_mul_default_nan_is_gfx1151_only_and_valid_arithmetic_is_exact() {
        let ninf = f32::NEG_INFINITY.to_bits();
        let pinf = f32::INFINITY.to_bits();
        let zero = 0f32.to_bits();
        let nan_default = 0xffc0_0000u32;
        // (opcode, src0, src1, expected) -- invalid: inf-inf (same sign), inf*0 either order.
        let cases = [
            ("v_sub_f32_e32", ninf, ninf, Some(nan_default)),
            ("v_sub_f32_e32", pinf, pinf, Some(nan_default)),
            ("v_subrev_f32_e32", ninf, ninf, Some(nan_default)),
            ("v_mul_f32_e32", ninf, zero, Some(nan_default)),
            ("v_mul_f32_e32", zero, pinf, Some(nan_default)),
            ("v_sub_f32_e32", 5f32.to_bits(), 2f32.to_bits(), Some(3f32.to_bits())),
            ("v_subrev_f32_e32", 5f32.to_bits(), 2f32.to_bits(), Some((-3f32).to_bits())),
            ("v_sub_f32_e32", pinf, ninf, Some(pinf)),
            ("v_mul_f32_e32", ninf, 2f32.to_bits(), Some(ninf)),
            ("v_mul_f32_e32", (-0f32).to_bits(), 3f32.to_bits(), Some((-0f32).to_bits())),
            ("v_sub_f32_e32", 0x7fc0_0000, 1f32.to_bits(), None),
            ("v_mul_f32_e32", 1f32.to_bits(), 0x7f80_0001, None),
        ];
        for (name, a, b, want) in cases {
            let mut st = state();
            st.exec = 1;
            st.v[1][0] = a;
            st.v[2][0] = b;
            let i = insn_1151(name, vec![v(0), v(1), v(2)]);
            match want {
                Some(w) => {
                    execute(Arch::Gfx1151, name, &i, &mut st).unwrap();
                    assert_eq!(st.v[0][0], w, "{name} {a:#010x} {b:#010x}");
                }
                None => assert!(execute(Arch::Gfx1151, name, &i, &mut st).is_err(), "{name} NaN operand"),
            }
        }
        for (name, a, b) in [("v_sub_f32_e32", ninf, ninf), ("v_mul_f32_e32", ninf, zero)] {
            let mut st = state();
            st.exec = 1;
            st.v[1][0] = a;
            st.v[2][0] = b;
            let i = insn(name, vec![v(0), v(1), v(2)]);
            assert!(run(name, &i, &mut st).is_err(), "{name} gfx1201 default NaN is not measured");
            assert_eq!(st.v[0][0], 0, "{name} must not write on error");
        }
    }

    #[test]
    fn unknown_vector_opcode_errors_even_without_active_lanes() {
        let i = insn("v_add_f32_e32", vec![v(0), v(1), v(2)]);
        let mut st = state();
        st.exec = 0;
        assert!(run("v_madeup_f32_e32", &i, &mut st).is_err());
    }
}
