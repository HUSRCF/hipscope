// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The rolled K-loop. One trip is a whole 256-K group (two 128-K blocks,
//! plane parity 0 then 1), so every LDS address is an immediate and the
//! trip ends in the state it started in. The last group is peeled: its
//! second block fetches and publishes nothing.
//!
//! Block (plane `h`), after hipcc K1-ILP's measured order:
//! - Phase A: slab-1 was fetched by the preceding block (or the prologue);
//!   fragment loads from slot 0; slab-0 WMMAs; publish slab 1 into slot 1;
//!   B1 retires slot 0 and publishes slot 1.
//! - Phase B: next-block slab-0 fetch and metadata; fragment loads from slot 1;
//!   slab-1 WMMAs and the fold's `subrev` stage of row group 0; publish the
//!   next block into slot 0; B2 signal retires its payload; fetch next block's
//!   slab 1 during the fold's `mul`/`fmac` stage (row group 1's `subrev`
//!   rides in row group 0's `fmac` packets); B2 wait.
use super::{
    blocks, fold,
    publish::{self, ds_load},
    Gen, Lds, MetaD, MetaS, Mma, PayA, PayW, Staged, WaitPlace, Wg, Wv, EPI, K_LOOP, K_LOOP_END,
};
use crate::kernels::common::{op, s, sr, v, vr};
use crate::{
    insn::{Instruction, Swmmac, SwmmacMods, Wmma},
    reg::Live,
    Builder, V,
};
use peacemaker_author::{rotate, Free, Published, Ring, Scc, State, WgUniform};

fn between(a: &str, b: &str) -> Live {
    Live::Between(a.into(), b.into())
}

/// The trip counter `K/256 - 1` derives from the `K` kernel argument only.
pub(crate) fn trips_cmp(wg: &mut Wg, g: &Gen, cmp: &str) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(
        format!("{cmp} s{}, 0", g.trips),
        vec![],
        vec![s(g.trips)],
    ))
}

/// Advance buffer descriptor `which` (0: the even block's `srd_a[0]`, 1: the
/// odd block's `srd_a[1]`) by the two-block group stride.
pub(crate) fn step_descriptor(b: &mut Builder, g: &Gen, which: usize) -> Result<(), String> {
    let srd = *g
        .srd_a
        .get(which)
        .ok_or_else(|| format!("step_descriptor: descriptor {which} is not 0 or 1"))?;
    op(
        b,
        format!(
            "s_add_nc_u64 s[{0}:{1}], s[{0}:{1}], s[{2}:{3}]",
            srd,
            srd + 1,
            g.step2,
            g.step2 + 1
        ),
        &[sr(srd, 2)],
        &[sr(srd, 2), sr(g.step2, 2)],
    )
}

/// Trip end: the odd-block descriptor and the group offset move on after
/// every read of them in this trip, and the trip counter drops by one.
pub(crate) fn trip_end(wg: &mut Wg, g: &Gen) -> Result<(), String> {
    let b = wg.isa();
    step_descriptor(b, g, 1)?;
    op(
        b,
        format!("s_add_co_i32 s{0}, s{0}, 0x88", g.goff),
        &[s(g.goff)],
        &[s(g.goff)],
    )?;
    op(
        b,
        format!("s_add_co_i32 s{0}, s{0}, -1", g.trips),
        &[s(g.trips)],
        &[s(g.trips)],
    )
}

/// The K-loop's block for the spec's matrix instruction: the dense `block` /
/// `block_stepping`, or their sparse twins. `step`: advance descriptor `step`
/// after the block's slab-1 fetch (`block_stepping`).
#[allow(clippy::too_many_arguments)]
fn mma_block(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end: &str,
    entry_label: bool,
    step: Option<usize>,
    lds: Lds,
) -> Result<Lds, String> {
    match (g.spec.mma.is_sparse(), step) {
        (false, None) => blocks::block(wg, g, tag, h, fetch_next, end, entry_label, lds),
        (false, Some(s)) => {
            blocks::block_stepping(wg, g, tag, h, fetch_next, end, entry_label, s, lds)
        }
        (true, None) => blocks::sparse_block(wg, g, tag, h, fetch_next, end, entry_label, lds),
        (true, Some(s)) => {
            blocks::sparse_block_stepping(wg, g, tag, h, fetch_next, end, entry_label, s, lds)
        }
    }
}

pub(crate) fn emit(wg: &mut Wg, g: &Gen, lds: Lds) -> Result<Lds, String> {
    wg.label(super::K_BEGIN)?;
    // K == 256: no full trip; go straight to the peeled group.
    let no_trip = blocks::trips_cmp(wg, g, "s_cmp_eq_u32")?;
    let lds = wg.wg_skip_if(no_trip, &blocks::tail_label(0), lds, |wg, lds| {
        let lds = wg.loop_carried(K_LOOP, lds, |wg, lds| {
            let m1 = blocks::label("m1");
            let lds = mma_block(wg, g, "m0", 0, true, &m1, true, None, lds)?;
            // The even-block descriptor was last read by block 0's slab-1 fetch;
            // move it to block 2g+2 before block 1's next-block fetch reads it.
            let lds = mma_block(wg, g, "m1", 1, true, K_LOOP_END, true, Some(0), lds)?;
            // Trip end: the odd-block descriptor and the group offset move on
            // after every read of them in this trip.
            blocks::trip_end(wg, g)?;
            Ok((lds, blocks::trips_cmp(wg, g, "s_cmp_lg_u32")?))
        })?;
        wg.label(K_LOOP_END)?;
        Ok(lds)
    })?;
    // The skip target is the peeled group's first block label.
    let t1 = blocks::label("t1");
    let lds = mma_block(wg, g, "t0", 0, true, &t1, false, None, lds)?;
    mma_block(wg, g, "t1", 1, false, EPI, true, None, lds)
}

pub(crate) fn label(tag: &str) -> String {
    format!(".Liu4_{tag}")
}
pub(crate) fn tail_label(h: usize) -> String {
    label(&format!("t{h}"))
}
/// The head of a block on plane `h`: the block's own label (`entry_label`;
/// the peeled group's first block is the K-loop skip target, which places
/// it) and the lifetimes of the registers its phases alias (fragments, then
/// the fold's scale rows and products). `ra_cur` / `rd_cur` are the physical
/// indices of the payload and token-scale rings' current buffers: block `h`
/// starts with the payload ring on slot 0 and the metadata ring on plane `h`.
/// The sub-block labels `x0` / `x1` the lifetimes end at are placed by
/// [`scale_loads`] / [`scale_tail`]; `end` closes the fold registers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_begin(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    end: &str,
    entry_label: bool,
    ra_cur: usize,
    rd_cur: usize,
) -> Result<(), String> {
    if g.spec.mma.is_sparse() {
        return Err(format!(
            "block {tag}: the spec is {}: use the sparse blocks (`sparse_block`)",
            g.spec.mma.name()
        ));
    }
    if rd_cur != h || ra_cur != 0 {
        return Err(format!(
            "block {tag}: LDS rings out of phase with plane {h}"
        ));
    }
    let (x, x0, x1) = sub_labels(tag);
    let b = wg.isa();
    if entry_label {
        b.label(&x)?;
    }
    let [f0, f1] = g.f;
    b.regs.v::<4>("frag0_w", f0, between(&x, &x0))?;
    b.regs.v::<8>("frag0_a", f0 + 4, between(&x, &x0))?;
    b.regs.v::<4>("frag1_w", f1, between(&x, &x1))?;
    b.regs.v::<8>("frag1_a", f1 + 4, between(&x, &x1))?;
    b.regs.v::<8>("sc_rg0", f0 + 4, between(&x0, end))?;
    b.regs.v::<4>("sc_rg1_lo", f0, between(&x0, end))?;
    b.regs.v::<4>("sc_rg1_hi", f1, between(&x1, end))?;
    b.regs.v::<8>("fold_t", f1 + 4, between(&x1, end))?;
    Ok(())
}

/// A block's own label and its two sub-block labels (`.Liu4_<tag>`,
/// `.Liu4_<tag>_s0`, `.Liu4_<tag>_s1`).
pub(crate) fn sub_labels(tag: &str) -> (String, String, String) {
    (
        label(tag),
        label(&format!("{tag}_s0")),
        label(&format!("{tag}_s1")),
    )
}

/// One K128 block on token/row plane `h`; `after_slab1_fetch` runs when its
/// preceding fetch has issued. `entry_label`: emit the block's label (the
/// peeled group's first block is the K-loop skip target, which emits it).
///
/// The block is a composition of the fine steps ([`block_begin`],
/// [`fragments`], [`wmma_group`], [`publish::publish_slab`] /
/// [`publish::publish_meta`], the `publish::fetch_*` loads, [`d_loads`],
/// [`scale_loads`], [`scale_tail`], `fold::scale`) which `blocks` exposes and
/// `.rip` scripts call one by one; `Tune::wait` picks where the publishes sit
/// between the WMMA groups.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_with(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end: &str,
    entry_label: bool,
    after_slab1_fetch: impl Fn(&mut Builder) -> Result<(), String>,
    (ra, rw, rd, rs): Lds,
) -> Result<Lds, String> {
    block_begin(
        wg,
        g,
        tag,
        h,
        end,
        entry_label,
        ra.cur_index(),
        rd.cur_index(),
    )?;
    let (_, x0, x1) = sub_labels(tag);

    let place = g.spec.tune.wait;
    // Phase A: slab 0 from slot 0, slab 1 staged into slot 1. The publish
    // (`WaitPlace`) writes the free next buffer, so it may precede the reads
    // of the current one.
    after_slab1_fetch(wg.isa())?;
    fragments(wg, g, &ra, &rw)?;
    let ((ra, pa), (rw, pw)) = match place {
        WaitPlace::AfterMma => {
            for sb in 0..2 {
                wmma_group(wg.isa(), g, sb, sb == 0, false)?;
            }
            publish::publish_slab(wg, g, ra, rw)?
        }
        WaitPlace::MidMma => {
            wmma_group(wg.isa(), g, 0, true, false)?;
            let staged = publish::publish_slab(wg, g, ra, rw)?;
            wmma_group(wg.isa(), g, 1, false, false)?;
            staged
        }
        WaitPlace::BeforeMma => {
            let staged = publish::publish_slab(wg, g, ra, rw)?;
            for sb in 0..2 {
                wmma_group(wg.isa(), g, sb, sb == 0, false)?;
            }
            staged
        }
    };
    // The next block's scale/header registers are independent of the
    // published slab; overlap their VMEM with the B1 wave rendezvous.
    if fetch_next {
        publish::fetch_next_meta(wg.isa(), g, h)?;
    }
    let (da, dw) = wg.wait_all((pa, pw))?;
    let (ra, rw) = wg.barrier((rotate(ra, da), rotate(rw, dw)))?;

    // Phase B: slab 1 from slot 1, the fold, next block staged into slot 0.
    if fetch_next {
        publish::fetch_next(wg.isa(), g, h)?;
    }
    if !fetch_next {
        fragments(wg, g, &ra, &rw)?;
        d_loads(wg, g, &rd)?;
        wmma_group(wg.isa(), g, 0, false, false)?;
        scale_loads(wg, g, &x0, &rs)?;
        wmma_group(wg.isa(), g, 1, false, true)?;
        scale_tail(wg, g, &x1, &rs)?;
        fold::scale(wg.isa(), g)?;
        return Ok((ra, rw, rd, rs));
    }
    // Next block's slab 0 and metadata publish (`WaitPlace`): the stores go to
    // the free next buffers; the B2 signal needs them drained after the last read.
    fragments(wg, g, &ra, &rw)?;
    d_loads(wg, g, &rd)?;
    let ((ra, pa), (rw, pw), (rd, pd), (rs, ps)) = match place {
        WaitPlace::AfterMma => {
            wmma_group(wg.isa(), g, 0, false, false)?;
            scale_loads(wg, g, &x0, &rs)?;
            wmma_group(wg.isa(), g, 1, false, true)?;
            scale_tail(wg, g, &x1, &rs)?;
            publish_next(wg, g, ra, rw, rd, rs)?
        }
        WaitPlace::MidMma => {
            wmma_group(wg.isa(), g, 0, false, false)?;
            scale_loads(wg, g, &x0, &rs)?;
            let (sa, sw, sd, ss) = publish_next(wg, g, ra, rw, rd, rs)?;
            wmma_group(wg.isa(), g, 1, false, true)?;
            scale_tail(wg, g, &x1, &ss.0)?;
            (sa, sw, sd, ss)
        }
        WaitPlace::BeforeMma => {
            let (sa, sw, sd, ss) = publish_next(wg, g, ra, rw, rd, rs)?;
            wmma_group(wg.isa(), g, 0, false, false)?;
            scale_loads(wg, g, &x0, &ss.0)?;
            wmma_group(wg.isa(), g, 1, false, true)?;
            scale_tail(wg, g, &x1, &ss.0)?;
            (sa, sw, sd, ss)
        }
    };
    let (da, dw, dd, ds) = wg.wait_all((pa, pw, pd, ps))?;
    let b2 = wg.signal((
        rotate(ra, da),
        rotate(rw, dw),
        rotate(rd, dd),
        rotate(rs, ds),
    ))?;
    // The B2 signal drained slab-0 stores; the retired payload ring can
    // now hold the next K128's slab 1 while this block still folds.
    // At the odd-to-even boundary goff has not yet advanced, so its W
    // addresses include the 136-byte group stride explicitly.
    if h == 0 {
        publish::fetch_slab1(wg.isa(), g, 1)?;
    } else {
        publish::fetch_slab1_next_group(wg.isa(), g)?;
    }
    fold::scale(wg.isa(), g)?;
    wg.wait_arrived(b2)
}

type StagedRings = (
    Staged<PayA, Published>,
    Staged<PayW, Published>,
    Staged<MetaD, Published>,
    Staged<MetaS, Published>,
);

/// Publish the next block's slab 0 into the payload rings' free next buffer
/// and its token/row metadata into the metadata rings'.
fn publish_next(
    wg: &mut Wg,
    g: &Gen,
    ra: Ring<PayA, Published, Free>,
    rw: Ring<PayW, Published, Free>,
    rd: Ring<MetaD, Published, Free>,
    rs: Ring<MetaS, Published, Free>,
) -> Result<StagedRings, String> {
    let (sa, sw) = publish::publish_slab(wg, g, ra, rw)?;
    let (sd, ss) = publish::publish_meta(wg, g, rd, rs)?;
    Ok((sa, sw, sd, ss))
}

/// Phase B: the token-scale (`d`) loads of the plane the `d` ring's current
/// buffer is (the block's plane `h`).
pub(crate) fn d_loads<ND: State>(
    w: &mut Wv,
    g: &Gen,
    rd: &Ring<MetaD, Published, ND>,
) -> Result<(), String> {
    let l = g.layout;
    let dp = rd.cur_index() as u32 * (l.ds[1] - l.ds[0]) / 4;
    for pair in 0..2u32 {
        w.ds_load_cur(
            rd,
            ds_load(
                format!(
                    "ds_load_2addr_b32 v[{}:{}], v{}{}",
                    g.d + 2 * pair as u8,
                    g.d + 2 * pair as u8 + 1,
                    g.d_addr,
                    super::ds_offsets(dp + 32 * pair, dp + 32 * pair + 16)
                ),
                vr(g.d + 2 * pair as u8, 2),
                g.d_addr,
            ),
        )?;
    }
    Ok(())
}

/// Phase B: the row-scale loads of row group 0 and the first half of row
/// group 1, issued at the sub-block-0 label `x0`.
pub(crate) fn scale_loads<NS: State>(
    w: &mut Wv,
    g: &Gen,
    x0: &str,
    rs: &Ring<MetaS, Published, NS>,
) -> Result<(), String> {
    let [f0, _] = g.f;
    w.label(x0)?;
    let sz = g.layout.sz[rs.cur_index()];
    for (dst, off) in [(f0 + 4, 0), (f0 + 8, 16), (f0, 64)] {
        w.ds_load_cur(
            rs,
            ds_load(
                format!(
                    "ds_load_b128 v[{}:{}], v{}{}",
                    dst,
                    dst + 3,
                    g.sc_addr,
                    super::ds_offset(sz + off)
                ),
                vr(dst, 4),
                g.sc_addr,
            ),
        )?;
    }
    Ok(())
}

/// Phase B: the last row-scale load, issued at the sub-block-1 label `x1`.
pub(crate) fn scale_tail<NS: State>(
    w: &mut Wv,
    g: &Gen,
    x1: &str,
    rs: &Ring<MetaS, Published, NS>,
) -> Result<(), String> {
    let [_, f1] = g.f;
    w.label(x1)?;
    w.ds_load_cur(
        rs,
        ds_load(
            format!(
                "ds_load_b128 v[{}:{}], v{}{}",
                f1,
                f1 + 3,
                g.sc_addr,
                super::ds_offset(g.layout.sz[rs.cur_index()] + 80)
            ),
            vr(f1, 4),
            g.sc_addr,
        ),
    )
}

/// Fragment loads of both 32-K sub-blocks from the payload rings' current
/// slot: per sub-block one W load (both row groups) and two A loads
/// (column blocks 0-1, 2-3).
pub(crate) fn fragments<N: State>(
    w: &mut Wv,
    g: &Gen,
    ra: &Ring<PayA, Published, N>,
    rw: &Ring<PayW, Published, N>,
) -> Result<(), String> {
    let slot = ra.cur_index();
    let l = g.layout;
    let (a, wo) = (l.a[slot] / 512, l.w[slot] / 512);
    for sb in 0..2 {
        let f = g.f[sb];
        w.ds_load_cur(
            rw,
            ds_load(
                format!(
                    "ds_load_2addr_stride64_b64 v[{}:{}], v{}{}",
                    f,
                    f + 3,
                    g.fr_w[sb],
                    super::ds_offsets(wo, wo + 1)
                ),
                vr(f, 4),
                g.fr_w[sb],
            ),
        )?;
        for half in 0..2u8 {
            let dst = f + 4 + 4 * half;
            let o = a + 2 * u32::from(half);
            w.ds_load_cur(
                ra,
                ds_load(
                    format!(
                        "ds_load_2addr_stride64_b64 v[{}:{}], v{}{}",
                        dst,
                        dst + 3,
                        g.fr_a[sb],
                        super::ds_offsets(o, o + 1)
                    ),
                    vr(dst, 4),
                    g.fr_a[sb],
                ),
            )?;
        }
    }
    Ok(())
}

/// Eight WMMAs of sub-block `sb` in `fold::BUNDLE_ORDER`. `first` seeds the
/// accumulators (`C = magic8`) instead of accumulating; `fold` interleaves
/// the fold's stage 1 (`fold::unbias_after_wmma`) between them. Registers
/// only: the fragments were loaded by [`fragments`].
pub(crate) fn wmma_group(
    b: &mut Builder,
    g: &Gen,
    sb: usize,
    first: bool,
    fold: bool,
) -> Result<(), String> {
    let f = g.f[sb];
    for (i, &k) in fold::BUNDLE_ORDER.iter().enumerate() {
        let (nb, rg) = (k / 2, k % 2);
        let dst = V::<8>(g.cacc + 8 * k as u8);
        let c = if first { V::<8>(g.magic) } else { dst };
        b.push(Wmma::iu4(
            g.spec.arch,
            dst,
            V::<2>(f + 2 * rg as u8),
            V::<2>(f + 4 + 2 * nb as u8),
            Some(c),
        ))?;
        if fold {
            fold::unbias_after_wmma(b, g, i)?;
        }
    }
    Ok(())
}

// ---- 2:4-sparse K-loop blocks (`Mma::S24X64`, `Mma::S24X32`) -----------------
//
// The weights are the sparse SWMMAC operand A, packed into the dense block's
// code bytes by `sparse24::pack_block`, so the W loads below are the dense
// loads: after them `f[0] + 2*rg` holds the compressed codes, `f[1] + 2*rg`
// the index word (and `f[1] + 2*rg + 1` padding), per lane `(row, h)`.
// SWMMAC has no C operand: each accumulator is seeded with `magic8` (`v_mov`)
// where dense passes `C = magic8` on its first WMMA. Fold, unbias and the
// epilogue are the dense ones.

/// Both operands signed, as dense: `neg_lo:[1,1,0]`.
const SWMMAC_SIGNED: SwmmacMods = SwmmacMods {
    index_key: 0,
    neg_lo: 3,
    neg_hi: 0,
    clamp: false,
};

/// First VGPR of column block `nb`'s x64 B operand: four contiguous VGPRs
/// `[act(sb0, nb), act(sb1, nb)]`; `nb` 0-1 in F0's activation registers, 2-3 in F1's.
fn b64_base(g: &Gen, nb: usize) -> u8 {
    if nb < 2 {
        g.f[0] + 4 + 4 * nb as u8
    } else {
        g.f[1] + 4 + 4 * (nb as u8 - 2)
    }
}

/// Seed accumulator `k` with `magic8` (what dense's `C = magic8` does).
fn seed_accumulator(b: &mut Builder, g: &Gen, k: usize) -> Result<(), String> {
    for j in 0..8u8 {
        let (dst, src) = (g.cacc + 8 * k as u8 + j, g.magic + j);
        op(
            b,
            format!("v_mov_b32_e32 v{dst}, v{src}"),
            &[v(dst)],
            &[v(src)],
        )?;
    }
    Ok(())
}

/// x64 fragment loads of a slab: the W loads as dense, then per column block
/// two `ds_load_b64` from the exact LDS addresses dense uses for `act(sb0, nb)`
/// and `act(sb1, nb)`, into one contiguous four-VGPR B operand.
fn fragments_s24x64<N: State>(
    w: &mut Wv,
    g: &Gen,
    ra: &Ring<PayA, Published, N>,
    rw: &Ring<PayW, Published, N>,
) -> Result<(), String> {
    let slot = ra.cur_index();
    let l = g.layout;
    let (a, wo) = (l.a[slot] / 512, l.w[slot] / 512);
    for sb in 0..2 {
        let f = g.f[sb];
        w.ds_load_cur(
            rw,
            ds_load(
                format!(
                    "ds_load_2addr_stride64_b64 v[{}:{}], v{}{}",
                    f,
                    f + 3,
                    g.fr_w[sb],
                    super::ds_offsets(wo, wo + 1)
                ),
                vr(f, 4),
                g.fr_w[sb],
            ),
        )?;
    }
    for nb in 0..4usize {
        let base = b64_base(g, nb);
        for sb in 0..2usize {
            let dst = base + 2 * sb as u8;
            w.ds_load_cur(
                ra,
                ds_load(
                    format!(
                        "ds_load_b64 v[{}:{}], v{}{}",
                        dst,
                        dst + 1,
                        g.fr_a[sb],
                        super::ds_offset(512 * (a + nb as u32))
                    ),
                    vr(dst, 2),
                    g.fr_a[sb],
                ),
            )?;
        }
    }
    Ok(())
}

/// Eight x64 SWMMACs (one K64 slab) in `fold::BUNDLE_ORDER`; `first` seeds each
/// accumulator with `magic8` before its first SWMMAC; `after(b, i)` runs after
/// the `i`-th.
fn bundle_s24x64(
    b: &mut Builder,
    g: &Gen,
    first: bool,
    after: impl Fn(&mut Builder, usize) -> Result<(), String>,
) -> Result<(), String> {
    let [f0, f1] = g.f;
    for (i, &k) in fold::BUNDLE_ORDER.iter().enumerate() {
        let (nb, rg) = (k / 2, k % 2);
        let dst = V::<8>(g.cacc + 8 * k as u8);
        if first {
            seed_accumulator(b, g, k)?;
        }
        b.push(Wmma::swmmac(
            g.spec.arch,
            Swmmac::I32Iu4X64,
            dst.reg(),
            vr(f0 + 2 * rg as u8, 2),
            vr(b64_base(g, nb), 4),
            V::<1>(f1 + 2 * rg as u8),
            SWMMAC_SIGNED,
        )?)?;
        after(b, i)?;
    }
    Ok(())
}

/// Eight x32 SWMMACs of sub-block `sb` in `fold::BUNDLE_ORDER`: A is the
/// compressed codes in W VGPR `sb` of row group `rg`, B the dense activation
/// fragment of `sb`, the index word `f[1] + 2*rg` with `index_key = sb`.
fn bundle_s24x32(
    b: &mut Builder,
    g: &Gen,
    sb: usize,
    first: bool,
    after: impl Fn(&mut Builder, usize) -> Result<(), String>,
) -> Result<(), String> {
    let [f0, f1] = g.f;
    for (i, &k) in fold::BUNDLE_ORDER.iter().enumerate() {
        let (nb, rg) = (k / 2, k % 2);
        let dst = V::<8>(g.cacc + 8 * k as u8);
        if first {
            seed_accumulator(b, g, k)?;
        }
        b.push(Wmma::swmmac(
            g.spec.arch,
            Swmmac::I32Iu4,
            dst.reg(),
            vr(f0 + 2 * rg as u8 + sb as u8, 1),
            V::<2>(g.f[sb] + 4 + 2 * nb as u8).reg(),
            V::<1>(f1 + 2 * rg as u8),
            SwmmacMods {
                index_key: sb as u8,
                ..SWMMAC_SIGNED
            },
        )?)?;
        after(b, i)?;
    }
    Ok(())
}

/// One 16-byte row-scale load of phase B.
fn sc_load<N: State>(
    w: &mut Wv,
    g: &Gen,
    rs: &Ring<MetaS, Published, N>,
    dst: u8,
    offset: u32,
) -> Result<(), String> {
    w.ds_load_cur(
        rs,
        ds_load(
            format!(
                "ds_load_b128 v[{}:{}], v{}{}",
                dst,
                dst + 3,
                g.sc_addr,
                super::ds_offset(offset)
            ),
            vr(dst, 4),
            g.sc_addr,
        ),
    )
}

/// One K128 block of the sparse K-loop on plane `h`; the sparse twin of
/// [`block_with`] (same arguments, same ring transitions, same publication,
/// next-block fetches and fold; only the fragment loads and the MMA differ).
///
/// - x64: per slab, W loads as dense, eight `ds_load_b64` for the activations,
///   eight SWMMACs. Phase B issues the scale loads after the last SWMMAC.
/// - x32: dense fragment loads, per sub-block eight SWMMACs with `index_key`;
///   the scale loads sit between the sub-blocks as in dense.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sparse_block_with(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end: &str,
    entry_label: bool,
    after_slab1_fetch: impl Fn(&mut Builder) -> Result<(), String>,
    (ra, rw, rd, rs): Lds,
) -> Result<Lds, String> {
    let x64 = match g.spec.mma {
        Mma::S24X64 => true,
        Mma::S24X32 => false,
        Mma::Dense => {
            return Err(format!(
                "block {tag}: the spec is dense: use the dense blocks (`block`)"
            ))
        }
    };
    if rd.cur_index() != h || ra.cur_index() != 0 {
        return Err(format!(
            "block {tag}: LDS rings out of phase with plane {h}"
        ));
    }
    let (x, x0, x1) = (
        label(tag),
        label(&format!("{tag}_s0")),
        label(&format!("{tag}_s1")),
    );
    let b = wg.isa();
    if entry_label {
        b.label(&x)?;
    }
    let [f0, f1] = g.f;
    if x64 {
        // Everything of F0/F1 is read by the slab's SWMMACs; the scale rows and
        // the fold's product registers reuse them only afterwards.
        b.regs.v::<4>("sfrag0_w", f0, between(&x, &x0))?;
        b.regs.v::<8>("sfrag0_a", f0 + 4, between(&x, &x0))?;
        b.regs.v::<4>("sfrag1_w", f1, between(&x, &x0))?;
        b.regs.v::<8>("sfrag1_a", f1 + 4, between(&x, &x0))?;
        b.regs.v::<8>("sc_rg0", f0 + 4, between(&x0, end))?;
        b.regs.v::<4>("sc_rg1_lo", f0, between(&x0, end))?;
        b.regs.v::<4>("sc_rg1_hi", f1, between(&x0, end))?;
        b.regs.v::<8>("fold_t", f1 + 4, between(&x0, end))?;
    } else {
        // sb1's SWMMACs still read F0's codes (VGPR 1 of each W pair) and F1's
        // indices, so those stay live to the second sub-block.
        b.regs.v::<4>("sfrag0_w", f0, between(&x, &x1))?;
        b.regs.v::<8>("sfrag0_a", f0 + 4, between(&x, &x0))?;
        b.regs.v::<4>("sfrag1_w", f1, between(&x, &x1))?;
        b.regs.v::<8>("sfrag1_a", f1 + 4, between(&x, &x1))?;
        b.regs.v::<8>("sc_rg0", f0 + 4, between(&x0, end))?;
        b.regs.v::<4>("sc_rg1_lo", f0, between(&x1, end))?;
        b.regs.v::<4>("sc_rg1_hi", f1, between(&x1, end))?;
        b.regs.v::<8>("fold_t", f1 + 4, between(&x1, end))?;
    }

    // Phase A: slab 0 from slot 0, slab 1 staged into slot 1.
    after_slab1_fetch(b)?;
    if x64 {
        fragments_s24x64(wg, g, &ra, &rw)?;
        bundle_s24x64(wg.isa(), g, true, |_, _| Ok(()))?;
    } else {
        fragments(wg, g, &ra, &rw)?;
        for sb in 0..2 {
            bundle_s24x32(wg.isa(), g, sb, sb == 0, |_, _| Ok(()))?;
        }
    }
    let ((ra, pa), (rw, pw)) = publish::publish_slab(wg, g, ra, rw)?;
    if fetch_next {
        publish::fetch_next_meta(wg.isa(), g, h)?;
    }
    let (da, dw) = wg.wait_all((pa, pw))?;
    let (ra, rw) = wg.barrier((rotate(ra, da), rotate(rw, dw)))?;

    // Phase B: slab 1 from slot 1, the fold, next block staged into slot 0.
    if fetch_next {
        publish::fetch_next(wg.isa(), g, h)?;
    }
    if x64 {
        fragments_s24x64(wg, g, &ra, &rw)?;
    } else {
        fragments(wg, g, &ra, &rw)?;
    }
    let l = g.layout;
    let dp = h as u32 * (l.ds[1] - l.ds[0]) / 4;
    for pair in 0..2u32 {
        wg.ds_load_cur(
            &rd,
            ds_load(
                format!(
                    "ds_load_2addr_b32 v[{}:{}], v{}{}",
                    g.d + 2 * pair as u8,
                    g.d + 2 * pair as u8 + 1,
                    g.d_addr,
                    super::ds_offsets(dp + 32 * pair, dp + 32 * pair + 16)
                ),
                vr(g.d + 2 * pair as u8, 2),
                g.d_addr,
            ),
        )?;
    }
    let sz = l.sz[h];
    if x64 {
        bundle_s24x64(wg.isa(), g, false, |b, i| fold::unbias_after_wmma(b, g, i))?;
        wg.label(&x0)?;
        for (dst, off) in [(f0 + 4, 0), (f0 + 8, 16), (f0, 64), (f1, 80)] {
            sc_load(wg, g, &rs, dst, sz + off)?;
        }
    } else {
        bundle_s24x32(wg.isa(), g, 0, false, |_, _| Ok(()))?;
        wg.label(&x0)?;
        for (dst, off) in [(f0 + 4, 0), (f0 + 8, 16)] {
            sc_load(wg, g, &rs, dst, sz + off)?;
        }
        bundle_s24x32(wg.isa(), g, 1, false, |b, i| {
            fold::unbias_after_wmma(b, g, i)
        })?;
        wg.label(&x1)?;
        for (dst, off) in [(f0, 64), (f1, 80)] {
            sc_load(wg, g, &rs, dst, sz + off)?;
        }
    }
    if !fetch_next {
        fold::scale(wg.isa(), g)?;
        return Ok((ra, rw, rd, rs));
    }
    let ((ra, pa), (rw, pw)) = publish::publish_slab(wg, g, ra, rw)?;
    let ((rd, pd), (rs, ps)) = publish::publish_meta(wg, g, rd, rs)?;
    let (da, dw, dd, ds) = wg.wait_all((pa, pw, pd, ps))?;
    let b2 = wg.signal((
        rotate(ra, da),
        rotate(rw, dw),
        rotate(rd, dd),
        rotate(rs, ds),
    ))?;
    // As dense: the B2 signal drained slab-0 stores, so the retired payload
    // ring can hold the next K128's slab 1 while this block folds.
    if h == 0 {
        publish::fetch_slab1(wg.isa(), g, 1)?;
    } else {
        publish::fetch_slab1_next_group(wg.isa(), g)?;
    }
    fold::scale(wg.isa(), g)?;
    wg.wait_arrived(b2)
}
