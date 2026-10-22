// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The rolled K-loop. One trip is a whole 256-K group (two 128-K blocks, plane parity 0 then 1),
//! so every LDS address is an immediate and the trip ends in the state it started in. The last
//! group is peeled: its second block fetches and publishes nothing.
//!
//! Block `b` (plane `h`, entry state: slab 0 published in slot 0, slab 1's raw A half 0 and
//! weight dwords in `a_pf` / `w_pf`, token/row planes `h` published):
//! - Phase A: four K16 steps from slot 0 on the two fragment sets (`F0`, `F1` ping-pong: the
//!   next-but-one step loads while the current one issues), interleaved with the decode and
//!   publication of slab 1 into slot 1; fetch of the next block's metadata; B1 retires slot 0
//!   and publishes slot 1.
//! - Phase B: next-block slab-0 fetch; four K16 steps from slot 1 interleaved with the decode and
//!   publication of the next block's slab 0 into slot 0 and of its metadata; the fold of this
//!   block (row group 0 first; row group 1's `sc` lands in the already folded accumulators'
//!   dead registers, its `z'` in the decode scratch); B2 is signalled after the last LDS load and
//!   waited at the block end, overlapping the fold's remaining packets. The block's last
//!   instructions refill the scratch with the next block's WMMA seed.
use super::{
    fold, publish, publish::ds_load, spec, Gen, Lds, MetaD, MetaS, PayA, PayW, Staged, Wg, EPI,
    K_LOOP, K_LOOP_END,
};
use crate::kernels::common::{op, s, sr, vr};
use crate::{insn::Instruction, reg::Live, Builder, V};
use peacemaker_author::{rotate, Free, Published, Ring, Scc, WgUniform};
use std::collections::VecDeque;

fn between(a: &str, b: &str) -> Live {
    Live::Between(a.into(), b.into())
}
fn label(tag: &str) -> String {
    format!(".Lmq6i8_{tag}")
}
fn tail_label(h: usize) -> String {
    label(&format!("t{h}"))
}

fn trips_cmp(wg: &mut Wg, g: &Gen, cmp: &str) -> Result<WgUniform<Scc>, String> {
    wg.scmp_wg_uniform(Instruction::new(
        format!("{cmp} s{}, 0", g.trips),
        vec![],
        vec![s(g.trips)],
    ))
}

pub(crate) fn emit(wg: &mut Wg, g: &Gen, lds: Lds) -> Result<Lds, String> {
    wg.label(super::K_BEGIN)?;
    // K == 256: no full trip; go straight to the peeled group.
    let no_trip = trips_cmp(wg, g, "s_cmp_eq_u32")?;
    let step = |b: &mut Builder, srd: u8| {
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
    };
    let lds = wg.wg_skip_if(no_trip, &tail_label(0), lds, |wg, lds| {
        let lds = wg.loop_carried(K_LOOP, lds, |wg, lds| {
            let lds = block_with(wg, g, "m0", 0, true, &label("m1"), true, |_| Ok(()), lds)?;
            // The even-block descriptor was last read by block 0's slab-1 fetch; move it to block
            // 2g+2 before block 1's next-block fetch reads it.
            let lds = block_with(
                wg,
                g,
                "m1",
                1,
                true,
                K_LOOP_END,
                true,
                |b| step(b, g.srd_a[0]),
                lds,
            )?;
            // Trip end: the odd-block descriptor and the group offsets move on after every read of
            // them in this trip.
            let b = wg.isa();
            step(b, g.srd_a[1])?;
            for r in [g.goff, g.goff1] {
                op(
                    b,
                    format!("s_add_co_i32 s{0}, s{0}, 0xc8", r),
                    &[s(r)],
                    &[s(r)],
                )?;
            }
            op(
                b,
                format!("s_add_co_i32 s{0}, s{0}, -1", g.trips),
                &[s(g.trips)],
                &[s(g.trips)],
            )?;
            Ok((lds, trips_cmp(wg, g, "s_cmp_lg_u32")?))
        })?;
        wg.label(K_LOOP_END)?;
        Ok(lds)
    })?;
    // The skip target is the peeled group's first block label.
    let lds = block_with(wg, g, "t0", 0, true, &label("t1"), false, |_| Ok(()), lds)?;
    block_with(wg, g, "t1", 1, false, EPI, true, |_| Ok(()), lds)
}

/// A ring as the interleaver holds it: not yet staging, or staging (next buffer `Writing`).
enum Pr<R: 'static> {
    Free(Ring<R, Published, Free>),
    Stg(Staged<R, Published>),
    Taken,
}
impl<R: 'static> Pr<R> {
    fn ds_load(&self, wg: &mut Wg, insn: Instruction) -> Result<(), String> {
        match self {
            Self::Free(r) => wg.ds_load_cur(r, insn),
            Self::Stg((r, _)) => wg.ds_load_cur(r, insn),
            Self::Taken => Err("ring is taken".into()),
        }
    }
    fn begin(&mut self, wg: &mut Wg) -> Result<usize, String> {
        match std::mem::replace(self, Self::Taken) {
            Self::Free(r) => {
                let slot = r.next_index();
                *self = Self::Stg(wg.begin_write(r));
                Ok(slot)
            }
            _ => Err("begin on a ring that is not Free".into()),
        }
    }
    fn staged(&mut self) -> Result<Staged<R, Published>, String> {
        match std::mem::replace(self, Self::Taken) {
            Self::Stg(s) => Ok(s),
            _ => Err("ring is not staging".into()),
        }
    }
    fn free(&mut self) -> Result<Ring<R, Published, Free>, String> {
        match std::mem::replace(self, Self::Taken) {
            Self::Free(r) => Ok(r),
            _ => Err("ring is not Free".into()),
        }
    }
}

/// One deferred unit of the interleaved slab publication.
enum Work {
    Insn(Instruction),
    StoreW(usize),
    StoreA(u32),
    FetchA { slab: u32, sb: u32, srd: u8 },
    Meta,
}

/// The four rings plus the queue of work to interleave between WMMAs.
struct Stage {
    slot: usize,
    work: VecDeque<Work>,
    a: Pr<PayA>,
    w: Pr<PayW>,
    d: Pr<MetaD>,
    s: Pr<MetaS>,
}
impl Stage {
    fn push_insns(&mut self, insns: Vec<Instruction>) {
        self.work.extend(insns.into_iter().map(Work::Insn));
    }
    fn pump(&mut self, wg: &mut Wg, g: &Gen, n: usize) -> Result<(), String> {
        for _ in 0..n {
            let Some(item) = self.work.pop_front() else {
                return Ok(());
            };
            match item {
                Work::Insn(i) => wg.isa().push(i)?,
                Work::StoreW(r) => {
                    let sw = self.w.staged()?;
                    self.w = Pr::Stg(publish::store_w_round(wg, g, sw, self.slot, r)?);
                }
                Work::StoreA(sb) => {
                    let sa = self.a.staged()?;
                    self.a = Pr::Stg(publish::store_a_half(wg, g, sa, self.slot, sb)?);
                }
                Work::FetchA { slab, sb, srd } => {
                    // The raw staging registers are still sources of the previous half's stores.
                    wg.isa().wait_all()?;
                    publish::fetch_a(wg.isa(), g, slab, sb, srd)?
                }
                Work::Meta => {
                    let (rd, rs) = (self.d.free()?, self.s.free()?);
                    let (sd, ss) = publish::publish_meta(wg, g, rd, rs)?;
                    self.d = Pr::Stg(sd);
                    self.s = Pr::Stg(ss);
                }
            }
        }
        Ok(())
    }
    fn flush(&mut self, wg: &mut Wg, g: &Gen) -> Result<(), String> {
        let n = self.work.len();
        self.pump(wg, g, n)
    }

    /// `ds_load_b128` of one K16 step (`step` = `sb * 2 + e`) into fragment set `set` from slot
    /// `slot`: the weight row-group pair, then the two token-group pairs.
    fn step_loads(
        &self,
        wg: &mut Wg,
        g: &Gen,
        slot: usize,
        step: u32,
        set: usize,
    ) -> Result<(), String> {
        let f = g.f[set];
        let l = spec::LAYOUT;
        let o = 512 * step;
        self.w.ds_load(
            wg,
            ds_load(
                format!(
                    "ds_load_b128 v[{}:{}], v{}{}",
                    f,
                    f + 3,
                    g.fr_w,
                    super::ds_offset(l.w[slot] + o)
                ),
                vr(f, 4),
                g.fr_w,
            ),
        )?;
        for half in 0..2u32 {
            let dst = f + 4 + 4 * half as u8;
            self.a.ds_load(
                wg,
                ds_load(
                    format!(
                        "ds_load_b128 v[{}:{}], v{}{}",
                        dst,
                        dst + 3,
                        g.fr_a,
                        super::ds_offset(l.a[slot] + o + 2048 * half)
                    ),
                    vr(dst, 4),
                    g.fr_a,
                ),
            )?;
        }
        Ok(())
    }

    /// Four K16 steps from `slot`; after the `n`-th WMMA (`n >= start`) `per` work units run.
    fn phase(
        &mut self,
        wg: &mut Wg,
        g: &Gen,
        slot: usize,
        first: bool,
        start: usize,
        per: usize,
    ) -> Result<(), String> {
        self.step_loads(wg, g, slot, 0, 0)?;
        self.step_loads(wg, g, slot, 1, 1)?;
        for step in 0..4u32 {
            let set = (step & 1) as usize;
            let base = 8 * step as usize;
            mma_step(wg, g, step, set, first && step == 0, &mut |wg, i| {
                if base + i >= start {
                    self.pump(wg, g, per)
                } else {
                    Ok(())
                }
            })?;
            if step + 2 < 4 {
                self.step_loads(wg, g, slot, step + 2, set)?;
            }
        }
        Ok(())
    }
}

/// The eight WMMAs of one K16 step on fragment set `set`; `after(wg, i)` runs after the `i`-th.
fn mma_step(
    wg: &mut Wg,
    g: &Gen,
    step: u32,
    set: usize,
    first: bool,
    after: &mut dyn FnMut(&mut Wg, usize) -> Result<(), String>,
) -> Result<(), String> {
    let f = g.f[set];
    let (sb, e) = (step >> 1, step & 1);
    for (i, &k) in fold::BUNDLE_ORDER.iter().enumerate() {
        let (nb, rg) = (k / 2, k % 2);
        let dst = V::<8>(g.cacc + 8 * k as u8);
        // Weight operand: data half `rg` sits in slot `rg ^ sb`; token operand: nb pair
        // `nb >> 1`, data half `nb & 1` in slot `(nb & 1) ^ e`.
        let wop = V::<2>(f + 2 * (rg as u8 ^ sb as u8));
        let xop = V::<2>(f + 4 + 4 * (nb as u8 >> 1) + 2 * ((nb as u8 & 1) ^ e as u8));
        let insn = if first {
            // Seed: the octet of `0x4B400000` words the previous block's tail (or the prologue)
            // left in the decode scratch.
            let seed = V::<8>(g.scratch);
            Instruction::new(
                format!(
                    "v_wmma_i32_16x16x16_iu8 {}, {}, {}, {} neg_lo:[0,1,0]",
                    dst.reg(),
                    wop.reg(),
                    xop.reg(),
                    seed.reg()
                ),
                vec![dst.reg()],
                vec![wop.reg(), xop.reg(), seed.reg()],
            )
        } else {
            Instruction::new(
                format!(
                    "v_wmma_i32_16x16x16_iu8 {}, {}, {}, {} neg_lo:[0,1,0]",
                    dst.reg(),
                    wop.reg(),
                    xop.reg(),
                    dst.reg()
                ),
                vec![dst.reg()],
                vec![wop.reg(), xop.reg(), dst.reg()],
            )
        };
        wg.isa().push(insn)?;
        after(wg, i)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn block_with(
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
    if rd.cur_index() != h || ra.cur_index() != 0 {
        return Err(format!(
            "block {tag}: LDS rings out of phase with plane {h}"
        ));
    }
    let (x, xf) = (label(tag), label(&format!("{tag}_fold")));
    let b = wg.isa();
    if entry_label {
        b.label(&x)?;
    }
    for (set, f) in g.f.into_iter().enumerate() {
        let names: [&'static str; 3] = if set == 0 {
            ["f0_w", "f0_a0", "f0_a1"]
        } else {
            ["f1_w", "f1_a0", "f1_a1"]
        };
        for (part, name) in names.into_iter().enumerate() {
            b.regs.v::<4>(name, f + 4 * part as u8, between(&x, &xf))?;
        }
    }
    // Fold registers: row group 0's sc / z' octets and the products, over the fragment sets.
    b.regs.v::<8>("fold_sc", fold::SC, between(&xf, end))?;
    b.regs.v::<8>("fold_zp", fold::ZP, between(&xf, end))?;
    b.regs.v::<8>("fold_t", fold::T, between(&xf, end))?;
    after_slab1_fetch(b)?;

    let l = spec::LAYOUT;
    let mut st = Stage {
        slot: 0,
        work: VecDeque::new(),
        a: Pr::Free(ra),
        w: Pr::Free(rw),
        d: Pr::Free(rd),
        s: Pr::Free(rs),
    };

    // ------------------------------------------------------------------ phase A
    st.slot = st.a.begin(wg)?;
    st.w.begin(wg)?;
    let mut insns = Vec::new();
    // Half 0 of this slab's activations was fetched a block ago: publish it first, so the
    // refill of the raw staging registers (which must wait for these stores to read them) finds
    // them long drained.
    st.work.push_back(Work::StoreA(0));
    publish::decode_round(&mut insns, g, 0)?;
    st.push_insns(std::mem::take(&mut insns));
    st.work.push_back(Work::FetchA {
        slab: 1,
        sb: 1,
        srd: g.srd_a[h],
    });
    st.work.push_back(Work::StoreW(0));
    publish::decode_round(&mut insns, g, 1)?;
    st.push_insns(std::mem::take(&mut insns));
    st.work.push_back(Work::StoreW(1));
    st.work.push_back(Work::StoreA(1));
    // The seed octet is the decode scratch: decode starts after the last seeded WMMA.
    st.phase(wg, g, 0, true, 8, 3)?;
    st.flush(wg, g)?;
    if fetch_next {
        // Next block's token word and packed row header (header word of the next half; across a
        // group boundary the group stride, since `goff` has not advanced).
        publish::fetch_meta(wg.isa(), g, h, if h == 0 { 4 } else { spec::GROUP_BYTES })?;
    }
    // B1 is split: the next block's raw slab-0 fetch (VMEM, its registers are free once the
    // stores drained) overlaps the rendezvous.
    let ((ra, pa), (rw, pw)) = (st.a.staged()?, st.w.staged()?);
    let (da, dw) = wg.wait_all((pa, pw))?;
    let b1 = wg.signal((rotate(ra, da), rotate(rw, dw)))?;
    if fetch_next {
        let b = wg.isa();
        publish::fetch_a(b, g, 0, 0, g.srd_a[h ^ 1])?;
        publish::fetch_w(
            b,
            g,
            if h == 0 {
                8 + 96
            } else {
                spec::GROUP_BYTES + 8
            },
        )?;
    }
    let (ra, rw) = wg.wait_arrived(b1)?;
    st.a = Pr::Free(ra);
    st.w = Pr::Free(rw);

    // ------------------------------------------------------------------ phase B
    if fetch_next {
        st.slot = st.a.begin(wg)?;
        st.w.begin(wg)?;
        publish::decode_round(&mut insns, g, 0)?;
        st.push_insns(std::mem::take(&mut insns));
        st.work.push_back(Work::StoreW(0));
        st.work.push_back(Work::StoreA(0));
        st.work.push_back(Work::FetchA {
            slab: 0,
            sb: 1,
            srd: g.srd_a[h ^ 1],
        });
        publish::decode_round(&mut insns, g, 1)?;
        st.push_insns(insns);
        st.work.push_back(Work::StoreW(1));
        st.work.push_back(Work::StoreA(1));
        st.work.push_back(Work::Meta);
    }
    st.phase(wg, g, 1, false, 6, 3)?;
    st.flush(wg, g)?;

    // ------------------------------------------------------------------ fold
    wg.label(&xf)?;
    let scratch = g.scratch;
    let b = wg.isa();
    // Plane `h` addresses: token (d, float(s)) pairs via `msn` as the address temporary, rows' sc / z'
    // from `sc_addr`.
    crate::kernels::common::op(
        b,
        format!(
            "v_add_nc_u32_e32 v{}, {}, v{}",
            g.msn,
            crate::kernels::common::lit(l.ds[h]),
            g.d_addr
        ),
        &[crate::kernels::common::v(g.msn)],
        &[crate::kernels::common::v(g.d_addr)],
    )?;
    let sz_load = |wg: &mut Wg, st: &Stage, dst: u8, off: u32| -> Result<(), String> {
        st.s.ds_load(
            wg,
            ds_load(
                format!(
                    "ds_load_b128 v[{}:{}], v{}{}",
                    dst,
                    dst + 3,
                    g.sc_addr,
                    super::ds_offset(off)
                ),
                vr(dst, 4),
                g.sc_addr,
            ),
        )
    };
    for nb in 0..4u32 {
        let dst = g.tok + 2 * nb as u8;
        st.d.ds_load(
            wg,
            ds_load(
                format!(
                    "ds_load_2addr_b32 v[{}:{}], v{}{}",
                    dst,
                    dst + 1,
                    g.msn,
                    super::ds_offsets(16 * nb, 128 + 16 * nb)
                ),
                vr(dst, 2),
                g.msn,
            ),
        )?;
    }
    let sz = l.sz[h];
    // Row group 0: sc octet then z' octet (planar, 8 rows of this lane's row half).
    sz_load(wg, &st, fold::SC, sz)?;
    sz_load(wg, &st, fold::SC + 4, sz + 16)?;
    sz_load(wg, &st, fold::ZP, sz + 512)?;
    sz_load(wg, &st, fold::ZP + 4, sz + 512 + 16)?;
    // Row group 1's z' into the decode scratch (free until the next block's decode).
    sz_load(wg, &st, scratch, sz + 64 + 512)?;
    sz_load(wg, &st, scratch + 4, sz + 64 + 512 + 16)?;

    let frag = |wg: &mut Wg, nb: usize, rg: usize, sc: u8, zp: u8| -> Result<(), String> {
        let b = wg.isa();
        fold::emit_msn(b, g, nb)?;
        fold::fragment(b, g, nb * 2 + rg, sc, zp)
    };
    frag(wg, 0, 0, fold::SC, fold::ZP)?;
    // Fragment k = 0 is folded: its int32 registers now hold row group 1's sc.
    sz_load(wg, &st, g.cacc, sz + 64)?;
    sz_load(wg, &st, g.cacc + 4, sz + 64 + 16)?;
    frag(wg, 1, 0, fold::SC, fold::ZP)?;

    let arrived = if fetch_next {
        let ((ra, pa), (rw, pw)) = (st.a.staged()?, st.w.staged()?);
        let ((rd, pd), (rs, ps)) = (st.d.staged()?, st.s.staged()?);
        let (da, dw, dd, ds) = wg.wait_all((pa, pw, pd, ps))?;
        let b2 = wg.signal((
            rotate(ra, da),
            rotate(rw, dw),
            rotate(rd, dd),
            rotate(rs, ds),
        ))?;
        // Next block's slab 1 (K32 half 0 and the weight codes): `goff` has not advanced at
        // the odd-to-even boundary, so its offsets carry the group stride.
        let b = wg.isa();
        publish::fetch_a(b, g, 1, 0, g.srd_a[h ^ 1])?;
        publish::fetch_w(
            b,
            g,
            if h == 0 {
                8 + 96 + 48
            } else {
                spec::GROUP_BYTES + 8 + 48
            },
        )?;
        Some(b2)
    } else {
        None
    };
    frag(wg, 2, 0, fold::SC, fold::ZP)?;
    frag(wg, 3, 0, fold::SC, fold::ZP)?;
    for nb in 0..4 {
        frag(wg, nb, 1, g.cacc, scratch)?;
    }
    if fetch_next {
        // Next block's WMMA seed octet (the scratch is dead after row group 1's fold).
        seed_fill(wg.isa(), g)?;
    }
    match arrived {
        Some(b2) => wg.wait_arrived(b2),
        None => Ok((st.a.free()?, st.w.free()?, st.d.free()?, st.s.free()?)),
    }
}

/// `scratch[0..8] = 0x4B400000`: the WMMA chain seed (`as_float` of `0x4B400000 + n` is exactly
/// `12582912 + n` for `|n| < 2^22`).
pub(crate) fn seed_fill(b: &mut Builder, g: &Gen) -> Result<(), String> {
    for i in 0..8u8 {
        crate::kernels::common::op(
            b,
            format!(
                "v_mov_b32_e32 v{}, {}",
                g.scratch + i,
                crate::kernels::common::lit(0x4b40_0000)
            ),
            &[crate::kernels::common::v(g.scratch + i)],
            &[],
        )?;
    }
    Ok(())
}
