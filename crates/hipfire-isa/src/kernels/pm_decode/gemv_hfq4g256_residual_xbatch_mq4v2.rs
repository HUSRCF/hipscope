//! Exact gfx1201 multi-column twin of `gemv_mq4g256v2_residual`.
//!
//! `gemv_mq4g256v2_residual_xbatch(A, x[B][K], y[B][M], M, K, B)`, B in 1..=8:
//! for every column `b`, `y[b]` is byte-identical to one singleton
//! `gemv_mq4g256v2_residual` launch on `x[b]`, `y[b]`.
//! ABI: A/x/y pointers at 0/8/16, M/K/B i32 at 24/28/32; grid M (row pair
//! `2*wg`, the singleton and x-batch launcher rule), block 32, no LDS or
//! scratch. x is `[B][K]` and y is `[B][M]`, both row-major f32.
//!
//! Each 136-byte weight group of the row pair is loaded and dequantized once
//! (`fma_mix(sc, nibble, zp)`), then applied to every column. Per column the
//! singleton's per-lane DAG is kept exactly: group `g` accumulates into stream
//! `g % 4` in increasing `g` (quads and scalar tails alike), each group dot is
//! one `mul` plus seven `fmac` (row0 seeds x0 then x1; row1 seeds x1 then x0),
//! the stream add is `dot + acc` for row0 and `acc + dot` for row1, the fold is
//! `(a0 + a1) + (a2 + a3)`, the reduction is the frozen shfl_down tree
//! (swizzle `1pppp`, then self-reading bpermutes 8/4/2/1), and the residual
//! epilogue adds `acc + y` for a row pair and `y + acc` for an odd final row.
//! Row0/row1 FMAs of one column are VOPD pairs (multiplication commutes; the
//! shared x operand is the gfx12 shared-src1 form). B is specialized: one code
//! path per column count, selected once per workgroup.
use crate::{Arch, Builder, Emitted, KernargLayout, KernelSpec, RegPlan};
use crate::insn::{Instruction, MemoryClass};
use crate::reg::{Kind, Live};
use crate::vopd::{Operand, VopdF32, VopdOp};

pub const SYMBOL: &str = "gemv_mq4g256v2_residual_xbatch";
pub const MAX_COLUMNS: u8 = 8;

// VGPRs.
const LANE: u8 = 0;
const LANE_X4: u8 = 1;
const PK_ADDR: [u8; 2] = [2, 4];
const HDR_ADDR: [u8; 2] = [3, 5];
const X_OFF: u8 = 6;
const SHUF_ADDR: u8 = 7;
/// Header pair of (group slot j, row r): `8 + 2*(2j + r)`.
fn hdr(j: u8, r: u8) -> u8 { 8 + 2 * (2 * j + r) }
/// Packed nibbles of (group slot j, row r).
fn pk(j: u8, r: u8) -> u8 { 24 + 2 * j + r }
/// Dequantized weights of the current group, row0 / row1. The bases differ
/// by 2 mod 4 so every row0/row1 VOPD pair has distinct src0 banks.
const W: [u8; 2] = [32, 42];
/// Column dots: (row0, row1) for even / odd interleaved columns.
const DOT: [[u8; 2]; 2] = [[50, 51], [52, 53]];
/// Stream accumulator (column b, stream s, row r); row0 even, row1 odd.
fn acc(b: u8, s: u8, r: u8) -> u8 { 54 + 8 * b + 2 * s + r }
/// Two x buffers (group j in buffer j % 2): column b's 8 floats at base + 8b.
const XBUF: [u8; 2] = [118, 182];
const VGPRS: u8 = 246;
fn xcol(buf: usize, b: u8) -> u8 { XBUF[buf] + 8 * b }
// Epilogue temporaries (x buffer 0 is dead after the last group).
fn red_tmp(b: u8, r: u8) -> u8 { 118 + 2 * b + r }
fn y_tmp(b: u8, r: u8) -> u8 { 134 + 2 * b + r }
fn y_addr(b: u8) -> u8 { 150 + b }

// SGPRs.
const S_QUADS: u8 = 2;
const S_B: u8 = 3;
const S_M: u8 = 10;
const S_K: u8 = 11;
const S_ROW0: u8 = 12;
const S_ROW1: u8 = 13;
const S_GROUPS: u8 = 14;
const S_G: u8 = 15;
const S_OFF0: u8 = 16;
const S_OFF1: u8 = 17;
const S_HAVE1: u8 = 18;
const S_STRIDE: u8 = 19;
fn s_xbase(b: u8) -> u8 { 28 + 2 * b }
const S_KX4: u8 = 44;
const S_MX4: u8 = 45;
const S_ROW0X4: u8 = 46;

pub fn build_gfx1201() -> Result<Vec<Emitted>, String> {
    let mut regs = RegPlan::new(u16::from(VGPRS), 48)?;
    for (name, base) in [("kernarg", 0), ("matrix", 4), ("activation", 6), ("residual", 8)] {
        regs.s::<2>(name, base, Live::Whole)?;
    }
    regs.s::<4>("weight_resource", 20, Live::Whole)?;
    for col in 0..MAX_COLUMNS { regs.add_range(&format!("column_x_base{col}"), Kind::S, s_xbase(col), 2, Live::Whole)?; }
    for base in [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27, 44, 45, 46, 47] {
        regs.add_range(&format!("scalar{base}"), Kind::S, base, 1, Live::Whole)?;
    }
    for (name, base, width) in [
        ("lane", LANE, 1), ("lane_x4", LANE_X4, 1), ("packed_addr0", PK_ADDR[0], 1), ("header_addr0", HDR_ADDR[0], 1),
        ("packed_addr1", PK_ADDR[1], 1), ("header_addr1", HDR_ADDR[1], 1), ("x_offset", X_OFF, 1),
        ("shuffle_addr", SHUF_ADDR, 1), ("headers", 8, 16), ("packed", 24, 8),
        ("weights_row0", W[0], 8), ("gap", 40, 2), ("weights_row1", W[1], 8), ("dots", 50, 4),
        ("accumulators", 54, 64), ("x_buffer0", XBUF[0], 64), ("x_buffer1", XBUF[1], 64),
    ] {
        // Plan ranges are 1/2/4/8 wide.
        let mut at = 0;
        while at < width {
            let w = (width - at).min(8);
            regs.add_range(&format!("{name}{at}"), Kind::V, base + at, w, Live::Whole)?;
            at += w;
        }
    }
    let mut b = Builder::new(KernelSpec {
        kernel_id: "gemv_hfq4g256_residual_xbatch_mq4v2".into(), variant: "exact_columns_b8".into(),
        arch: Arch::Gfx1201, symbol: SYMBOL.into(),
        kernargs: KernargLayout::new(36).pointer_access("A", 0, crate::plan::Access::ReadOnly)
            .pointer_access("x", 8, crate::plan::Access::ReadOnly).pointer("y", 16)
            .hidden("M", 24, 4, "by_value").hidden("K", 28, 4, "by_value").hidden("B", 32, 4, "by_value"),
        user_sgpr_count: 2, system_sgpr_workgroup_id_y: false, workgroup_size: 32,
        group_segment_fixed_size: 0, wave32: true, cu_mode: false,
    }, regs);
    smem(&mut b, "s_load_b64 s[10:11], s[0:1], 0x18", &[10, 11], &[0, 1])?;
    salu(&mut b, "s_lshl_b32 s12, ttmp9, 1", &[S_ROW0], &[])?;
    b.wait_all()?;
    salu(&mut b, "s_cmp_ge_i32 s12, s10", &[], &[S_ROW0, S_M])?;
    branch(&mut b, "s_cbranch_scc1 .Lrx_end")?;
    smem(&mut b, "s_load_b128 s[4:7], s[0:1], 0x0", &[4, 5, 6, 7], &[0, 1])?;
    smem(&mut b, "s_load_b64 s[8:9], s[0:1], 0x10", &[8, 9], &[0, 1])?;
    smem(&mut b, "s_load_b32 s3, s[0:1], 0x20", &[S_B], &[0, 1])?;
    b.wait_all()?;
    salu(&mut b, "s_add_co_i32 s13, s12, 1", &[S_ROW1], &[S_ROW0])?;
    salu(&mut b, "s_cmp_lt_i32 s13, s10", &[], &[S_ROW1, S_M])?;
    salu(&mut b, "s_cselect_b32 s18, 1, 0", &[S_HAVE1], &[])?;
    salu(&mut b, "s_cselect_b32 s13, s13, s12", &[S_ROW1], &[S_ROW1, S_ROW0])?;
    salu(&mut b, "s_lshr_b32 s14, s11, 8", &[S_GROUPS], &[S_K])?;
    salu(&mut b, "s_mul_i32 s19, s14, 0x88", &[S_STRIDE], &[S_GROUPS])?;
    salu(&mut b, "s_mul_i32 s16, s19, s12", &[S_OFF0], &[S_STRIDE, S_ROW0])?;
    salu(&mut b, "s_mul_i32 s17, s19, s13", &[S_OFF1], &[S_STRIDE, S_ROW1])?;
    salu(&mut b, "s_mov_b32 s20, s4", &[20], &[4])?;
    salu(&mut b, "s_and_b32 s21, s5, 0xffff", &[21], &[5])?;
    salu(&mut b, "s_mov_b32 s22, -1", &[22], &[])?;
    salu(&mut b, "s_mov_b32 s23, 0x31004000", &[23], &[])?;
    salu(&mut b, "s_lshl_b32 s44, s11, 2", &[S_KX4], &[S_K])?;
    salu(&mut b, "s_lshl_b32 s45, s10, 2", &[S_MX4], &[S_M])?;
    salu(&mut b, "s_lshl_b32 s46, s12, 2", &[S_ROW0X4], &[S_ROW0])?;
    for col in 0..MAX_COLUMNS {
        let (lo, hi) = (s_xbase(col), s_xbase(col) + 1);
        if col == 0 {
            salu(&mut b, &format!("s_mov_b32 s{lo}, s6"), &[lo], &[6])?;
            salu(&mut b, &format!("s_mov_b32 s{hi}, s7"), &[hi], &[7])?;
            continue;
        }
        salu(&mut b, &format!("s_mul_i32 s24, s44, {col}"), &[24], &[S_KX4])?;
        salu(&mut b, &format!("s_add_co_u32 s{lo}, s6, s24"), &[lo], &[6, 24])?;
        salu(&mut b, &format!("s_add_co_ci_u32 s{hi}, s7, 0"), &[hi], &[7])?;
    }
    valu(&mut b, "v_lshlrev_b32_e32 v1, 2, v0", &[LANE_X4], &[LANE], &[])?;
    for nb in 1..=MAX_COLUMNS {
        salu(&mut b, &format!("s_cmp_eq_u32 s3, {nb}"), &[], &[S_B])?;
        branch(&mut b, &format!("s_cbranch_scc1 .Lrx_b{nb}"))?;
    }
    branch(&mut b, "s_branch .Lrx_end")?;
    for nb in 1..=MAX_COLUMNS {
        section(&mut b, nb)?;
    }
    b.label(".Lrx_end")?;
    branch(&mut b, "s_endpgm")?;
    Ok(vec![b.finish()?])
}

/// The complete kernel body for `nb` columns.
fn section(b: &mut Builder, nb: u8) -> Result<(), String> {
    b.label(&format!(".Lrx_b{nb}"))?;
    for col in 0..nb {
        for s in 0..4 {
            for r in 0..2 {
                let a = acc(col, s, r);
                valu(b, &format!("v_mov_b32_e32 v{a}, 0"), &[a], &[], &[])?;
            }
        }
    }
    salu(b, "s_mov_b32 s15, 0", &[S_G], &[])?;
    salu(b, "s_lshr_b32 s2, s14, 2", &[S_QUADS], &[S_GROUPS])?;
    salu(b, "s_cmp_eq_u32 s2, 0", &[], &[S_QUADS])?;
    branch(b, &format!("s_cbranch_scc1 .Lrx_b{nb}_tail"))?;
    b.loop_(&format!(".Lrx_b{nb}_quad"), |b| {
        quad(b, nb)?;
        salu(b, "s_add_co_i32 s2, s2, -1", &[S_QUADS], &[S_QUADS])?;
        salu(b, "s_cmp_lg_u32 s2, 0", &[], &[S_QUADS])?;
        branch(b, &format!("s_cbranch_scc1 .Lrx_b{nb}_quad"))
    })?;
    b.label(&format!(".Lrx_b{nb}_tail"))?;
    for stream in 0..3 {
        salu(b, "s_cmp_lt_u32 s15, s14", &[], &[S_G, S_GROUPS])?;
        branch(b, &format!("s_cbranch_scc0 .Lrx_b{nb}_fold"))?;
        group_addresses(b)?;
        weight_loads(b, 0)?;
        x_loads(b, nb, 0, 0)?;
        dequant(b, 0)?;
        columns(b, nb, stream, 0)?;
        salu(b, "s_add_co_i32 s15, s15, 1", &[S_G], &[S_G])?;
        b.wait_all()?;
    }
    b.label(&format!(".Lrx_b{nb}_fold"))?;
    fold(b, nb)?;
    reduce(b, nb)?;
    epilogue(b, nb)
}

/// Row-pair weight and x addresses of group `s15` (slot offsets are immediates).
fn group_addresses(b: &mut Builder) -> Result<(), String> {
    salu(b, "s_mul_i32 s24, s15, 0x88", &[24], &[S_G])?;
    salu(b, "s_add_co_i32 s25, s16, s24", &[25], &[S_OFF0, 24])?;
    salu(b, "s_add_co_i32 s26, s17, s24", &[26], &[S_OFF1, 24])?;
    salu(b, "s_lshl_b32 s27, s15, 10", &[27], &[S_G])?;
    for (r, s) in [(0usize, 25u8), (1, 26)] {
        valu(b, &format!("v_mov_b32_e32 v{}, s{s}", HDR_ADDR[r]), &[HDR_ADDR[r]], &[], &[s])?;
        valu(b, &format!("v_add_nc_u32_e32 v{}, s{s}, v1", PK_ADDR[r]), &[PK_ADDR[r]], &[LANE_X4], &[s])?;
    }
    valu(b, "v_lshlrev_b32_e32 v6, 5, v0", &[X_OFF], &[LANE], &[])?;
    valu(b, "v_add_nc_u32_e32 v6, s27, v6", &[X_OFF], &[X_OFF], &[27])?;
    valu(b, "v_cmp_gt_u32_e32 vcc_lo, 16, v0", &[], &[LANE], &[])
}

fn quad(b: &mut Builder, nb: u8) -> Result<(), String> {
    group_addresses(b)?;
    weight_loads(b, 0)?;
    x_loads(b, nb, 0, 0)?;
    for j in 1..4 { weight_loads(b, j)?; }
    for j in 0..4u8 {
        if j < 3 { x_loads(b, nb, j + 1, usize::from((j + 1) % 2))?; }
        dequant(b, j)?;
        columns(b, nb, j, usize::from(j % 2))?;
    }
    salu(b, "s_add_co_i32 s15, s15, 4", &[S_G], &[S_G])?;
    b.wait_all()
}

fn weight_loads(b: &mut Builder, j: u8) -> Result<(), String> {
    let off = u32::from(j) * 136;
    for r in 0..2u8 {
        let h = hdr(j, r);
        let a = HDR_ADDR[usize::from(r)];
        let suffix = if off == 0 { String::new() } else { format!(" offset:{off}") };
        vmem(b, &format!("buffer_load_b64 v[{h}:{}], v{a}, s[20:23], null offen{suffix} scope:SCOPE_DEV", h + 1),
            &[h, h + 1], &[a], &[20, 21, 22, 23], false)?;
    }
    for r in 0..2u8 {
        let p = pk(j, r);
        let a = PK_ADDR[usize::from(r)];
        vmem(b, &format!("buffer_load_b32 v{p}, v{a}, s[20:23], null offen offset:{} scope:SCOPE_DEV", off + 8),
            &[p], &[a], &[20, 21, 22, 23], false)?;
    }
    Ok(())
}

fn x_loads(b: &mut Builder, nb: u8, j: u8, buf: usize) -> Result<(), String> {
    for col in 0..nb {
        let x = xcol(buf, col);
        let (lo, hi) = (s_xbase(col), s_xbase(col) + 1);
        for half in 0..2u8 {
            let off = u32::from(j) * 1024 + u32::from(half) * 16;
            let suffix = if off == 0 { String::new() } else { format!(" offset:{off}") };
            let d = x + 4 * half;
            vmem(b, &format!("global_load_b128 v[{d}:{}], v6, s[{lo}:{hi}]{suffix}", d + 3),
                &[d, d + 1, d + 2, d + 3], &[X_OFF], &[lo, hi], false)?;
        }
    }
    Ok(())
}

/// `w = fma_mix(sc, (pk >> 4n) & 15, zp)` for both rows of group slot `j`;
/// `vcc_lo` holds `lane < 16` (half-header select).
fn dequant(b: &mut Builder, j: u8) -> Result<(), String> {
    for r in 0..2u8 {
        let h = hdr(j, r);
        valu(b, &format!("v_cndmask_b32_e32 v{h}, v{}, v{h}, vcc_lo", h + 1), &[h], &[h, h + 1], &[])?;
    }
    for n in 0..8u8 {
        for r in 0..2u8 {
            let (w, p) = (W[usize::from(r)] + n, pk(j, r));
            valu(b, &format!("v_bfe_u32 v{w}, v{p}, {}, 4", n * 4), &[w], &[p], &[])?;
        }
    }
    for n in 0..8u8 {
        for r in 0..2u8 {
            let w = W[usize::from(r)] + n;
            valu(b, &format!("v_cvt_f32_ubyte0_e32 v{w}, v{w}"), &[w], &[w], &[])?;
        }
    }
    for n in 0..8u8 {
        for r in 0..2u8 {
            let (w, h) = (W[usize::from(r)] + n, hdr(j, r));
            valu(b, &format!("v_fma_mix_f32 v{w}, v{h}, v{w}, v{h} op_sel:[0,0,1] op_sel_hi:[1,0,1]"), &[w], &[h, w], &[])?;
        }
    }
    Ok(())
}

/// Dot both rows of the dequantized group against every column and add the
/// dots into stream `s`. Two columns interleave on separate dot registers.
fn columns(b: &mut Builder, nb: u8, s: u8, buf: usize) -> Result<(), String> {
    let mut col = 0;
    while col < nb {
        let group: Vec<u8> = (col..nb.min(col + 2)).collect();
        for step in 0..8u8 {
            for (slot, &c) in group.iter().enumerate() {
                let x = xcol(buf, c);
                let [d0, d1] = DOT[slot];
                // Row0 seeds x0 then x1; row1 seeds x1 then x0 (frozen DAG).
                let (n0, n1) = match step { 0 => (0, 1), 1 => (1, 0), k => (k, k) };
                let op = if step == 0 { VopdF32::Mul } else { VopdF32::Fmac };
                b.vopd(
                    VopdOp { op, dst: d0, src0: Operand::V(W[0] + n0), src1: x + n0 },
                    VopdOp { op, dst: d1, src0: Operand::V(W[1] + n1), src1: x + n1 },
                )?;
            }
        }
        for (slot, &c) in group.iter().enumerate() {
            let [d0, d1] = DOT[slot];
            let (a0, a1) = (acc(c, s, 0), acc(c, s, 1));
            b.vopd(
                VopdOp { op: VopdF32::Add, dst: a0, src0: Operand::V(d0), src1: a0 },
                VopdOp { op: VopdF32::Add, dst: a1, src0: Operand::V(a1), src1: d1 },
            )?;
        }
        col += 2;
    }
    Ok(())
}

/// `(a0 + a1) + (a2 + a3)` per column and row, into stream 0.
fn fold(b: &mut Builder, nb: u8) -> Result<(), String> {
    for col in 0..nb {
        for (dst, other) in [(0u8, 1u8), (2, 3), (0, 2)] {
            let x = acc(col, dst, 0);
            let y = acc(col, dst, 1);
            b.vopd(
                VopdOp { op: VopdF32::Add, dst: x, src0: Operand::V(x), src1: acc(col, other, 0) },
                VopdOp { op: VopdF32::Add, dst: y, src0: Operand::V(y), src1: acc(col, other, 1) },
            )?;
        }
    }
    Ok(())
}

/// Frozen shfl_down 16/8/4/2/1 tree, batched over every (column, row) value.
fn reduce(b: &mut Builder, nb: u8) -> Result<(), String> {
    let values: Vec<(u8, u8)> = (0..nb).flat_map(|c| [(c, 0u8), (c, 1u8)]).collect();
    for &(c, r) in &values {
        let (v, t) = (acc(c, 0, r), red_tmp(c, r));
        b.ds_crosslane(Instruction::new(format!("ds_swizzle_b32 v{t}, v{v} offset:swizzle(BITMASK_PERM,\"1pppp\")"),
            refs(Kind::V, &[t]), refs(Kind::V, &[v])).memory(MemoryClass::DsLoad))?;
    }
    for &(c, r) in &values {
        let (v, t) = (acc(c, 0, r), red_tmp(c, r));
        valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
    }
    for offset in [8u8, 4, 2, 1] {
        valu(b, &format!("v_cmp_gt_u32_e32 vcc_lo, {}, v0", 32 - offset), &[], &[LANE], &[])?;
        valu(b, &format!("v_cndmask_b32_e64 v7, 0, {offset}, vcc_lo"), &[SHUF_ADDR], &[], &[])?;
        valu(b, "v_add_lshl_u32 v7, v7, v0, 2", &[SHUF_ADDR], &[SHUF_ADDR, LANE], &[])?;
        for &(c, r) in &values {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            b.ds_crosslane(Instruction::new(format!("ds_bpermute_b32 v{t}, v7, v{v}"),
                refs(Kind::V, &[t]), refs(Kind::V, &[SHUF_ADDR, v])).memory(MemoryClass::DsLoad))?;
        }
        for &(c, r) in &values {
            let (v, t) = (acc(c, 0, r), red_tmp(c, r));
            valu(b, &format!("v_add_f32_e32 v{v}, v{v}, v{t}"), &[v], &[v, t], &[])?;
        }
    }
    Ok(())
}

/// Lane 0: `y[b][row0] = acc + y`, `y[b][row1] = bcc + y` for a row pair;
/// `y[b][row0] = y + acc` for an odd final row (the singleton's operand order).
fn epilogue(b: &mut Builder, nb: u8) -> Result<(), String> {
    valu(b, "v_cmpx_eq_u32_e32 0, v0", &[], &[LANE], &[])?;
    for col in 0..nb {
        let a = y_addr(col);
        if col == 0 {
            valu(b, &format!("v_mov_b32_e32 v{a}, s46"), &[a], &[], &[S_ROW0X4])?;
        } else {
            salu(b, &format!("s_mul_i32 s47, s45, {col}"), &[47], &[S_MX4])?;
            salu(b, "s_add_co_i32 s47, s47, s46", &[47], &[47, S_ROW0X4])?;
            valu(b, &format!("v_mov_b32_e32 v{a}, s47"), &[a], &[], &[47])?;
        }
    }
    salu(b, "s_cmp_eq_u32 s18, 0", &[], &[S_HAVE1])?;
    branch(b, &format!("s_cbranch_scc1 .Lrx_b{nb}_single"))?;
    for col in 0..nb {
        let a = y_addr(col);
        for r in 0..2u8 {
            let y = y_tmp(col, r);
            let suffix = if r == 0 { "" } else { " offset:4" };
            vmem(b, &format!("global_load_b32 v{y}, v{a}, s[8:9]{suffix}"), &[y], &[a], &[8, 9], false)?;
        }
    }
    for col in 0..nb {
        let a = y_addr(col);
        for r in 0..2u8 {
            let (y, v) = (y_tmp(col, r), acc(col, 0, r));
            valu(b, &format!("v_add_f32_e32 v{y}, v{v}, v{y}"), &[y], &[v, y], &[])?;
        }
        for r in 0..2u8 {
            let y = y_tmp(col, r);
            let suffix = if r == 0 { "" } else { " offset:4" };
            vmem(b, &format!("global_store_b32 v{a}, v{y}, s[8:9]{suffix}"), &[], &[a, y], &[8, 9], true)?;
        }
    }
    b.wait_all()?;
    branch(b, "s_branch .Lrx_end")?;
    b.label(&format!(".Lrx_b{nb}_single"))?;
    for col in 0..nb {
        let (a, y) = (y_addr(col), y_tmp(col, 0));
        vmem(b, &format!("global_load_b32 v{y}, v{a}, s[8:9]"), &[y], &[a], &[8, 9], false)?;
    }
    for col in 0..nb {
        let (a, y, v) = (y_addr(col), y_tmp(col, 0), acc(col, 0, 0));
        valu(b, &format!("v_add_f32_e32 v{y}, v{y}, v{v}"), &[y], &[y, v], &[])?;
        vmem(b, &format!("global_store_b32 v{a}, v{y}, s[8:9]"), &[], &[a, y], &[8, 9], true)?;
    }
    b.wait_all()?;
    branch(b, "s_branch .Lrx_end")
}

fn refs(kind: Kind, indices: &[u8]) -> Vec<crate::reg::RegRef> {
    indices.iter().map(|&base| crate::reg::RegRef { kind, base, len: 1 }).collect()
}
fn salu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(text, refs(Kind::S, defs), refs(Kind::S, uses)))
}
fn valu(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8]) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses))
}
fn smem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8]) -> Result<(), String> {
    b.push(Instruction::new(text, refs(Kind::S, defs), refs(Kind::S, uses)).memory(MemoryClass::SmemLoad))
}
fn vmem(b: &mut Builder, text: &str, defs: &[u8], uses: &[u8], scalar: &[u8], store: bool) -> Result<(), String> {
    let mut uses = refs(Kind::V, uses);
    uses.extend(refs(Kind::S, scalar));
    b.push(Instruction::new(text, refs(Kind::V, defs), uses)
        .memory(if store { MemoryClass::VmemStore } else { MemoryClass::VmemLoad }))
}
fn branch(b: &mut Builder, text: &str) -> Result<(), String> { b.control(Instruction::new(text, vec![], vec![])) }

#[cfg(all(test, feature = "toolchain"))]
mod tests {
    use super::*;
    #[test]
    fn residual_xbatch_m7() {
        let emitted = build_gfx1201().expect("residual xbatch").remove(0);
        // Row1 seeds x1 then x0; the two rows share x as the gfx12 src1.
        assert!(emitted.s_text.contains("v_dual_mul_f32 v50, v32, v118 :: v_dual_mul_f32 v51, v43, v119"));
        assert!(emitted.s_text.contains("v_dual_fmac_f32 v50, v33, v119 :: v_dual_fmac_f32 v51, v42, v118"));
        assert!(emitted.s_text.contains("v_dual_add_f32 v54, v50, v54 :: v_dual_add_f32 v55, v55, v51"));
        let elf = crate::native::assemble(&emitted.s_text, Arch::Gfx1201).expect("native assemble");
        let dir = std::path::PathBuf::from("/home/kaden/qcal/release-0.4.2/pm-decode-twins/residual-xbatch");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("gemv_hfq4g256_residual_xbatch_mq4v2.test.s"), &emitted.s_text).unwrap();
        let co = dir.join("gemv_hfq4g256_residual_xbatch_mq4v2.test.co");
        std::fs::write(&co, &elf).unwrap();
        let report = crate::pm_check::m7(&co, "gfx1201", SYMBOL).expect("M7");
        println!("{report}");
        assert_eq!(report["obligations"], serde_json::json!({}), "{report}");
    }
}
