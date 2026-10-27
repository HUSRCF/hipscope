// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `iu4_gemm` generator as public building blocks.
//!
//! [`super::emit`] and [`super::kloop::emit`] are written over exactly these
//! functions, so a composition that mirrors them (the `.rip` twin
//! `kernels/iu4_gemm.rip`) re-emits their bytes, and a composition that
//! differs (another trip structure, a moved fetch, a hand-written epilogue) is
//! checked by the same typed core and builder as the production kernel.
//!
//! The blocks work on the typed workgroup [`Wg`] and the typed LDS rings
//! ([`FreeRings`] as declared, [`Lds`] at a block boundary), so a block still
//! cannot read a `Free` buffer. A dense [`block`] is a composition of the fine
//! steps below ([`block_begin`], [`fragments`], [`wmma_group`],
//! [`publish_slab`], the `fetch_*` loads, ...): the generator and the `.rip`
//! natives (`iu4::fragments`, `iu4::wmma`, `iu4::publish_slab`, ..., cookbook
//! 17) call the same functions, so a script can place them itself. The sparse
//! blocks are not split.
//!
//! The kernel order, as `emit` composes it:
//!
//! ```text
//! lds = declare_lds(wg, g)        end = wg.exit(END);  s_setprio 1
//! lds = prologue(wg, g, end, lds)
//! lds = kloop(wg, g, lds)         // or the trip structure spelled out, see kloop.rs
//! epilogue(wg, g, end)
//! gdn_epilogue(wg, g, lds)        // Epi::QkvzaGdn only
//! wg.end(end)
//! ```
use super::{epilogue as epi, gdn_epilogue as gdn, kloop, prologue as pro, publish, Epi};
use crate::Builder;
use peacemaker_author::{End, Free, Published, Ring, Scc, WgUniform};

pub use super::{
    FreeRings, Gen, Lds, MetaD, MetaS, PayA, PayW, Staged, Wg, END, ENTRY, EPI, GDN, K_BEGIN,
    K_LOOP, K_LOOP_END,
};

/// Declare the eight LDS slots (A0 W0 A1 W1 DS0 DS1 SZ0 SZ1) as four two-buffer
/// rings, every buffer `Free`.
pub fn declare_lds(wg: &mut Wg, g: &Gen) -> Result<FreeRings, String> {
    g.declare_lds(wg)
}

/// Kernel arguments, banded raster, buffer descriptors, hoisted lane offsets,
/// block-0 staging and the first rendezvous. `end` is the kernel exit the
/// out-of-range tiles leave through.
pub fn prologue(wg: &mut Wg, g: &Gen, end: &End, lds: FreeRings) -> Result<Lds, String> {
    pro::emit(wg, g, end, lds)
}

/// The whole rolled K-loop with its peeled last group: [`trips_cmp`],
/// `wg_skip_if` over [`block`]s inside `loop_carried`, then the peeled pair.
pub fn kloop(wg: &mut Wg, g: &Gen, lds: Lds) -> Result<Lds, String> {
    kloop::emit(wg, g, lds)
}

/// One K128 block on token/row plane `h` (`0` or `1`); `tag` names its
/// labels (`.Liu4_<tag>`), `end_label` is the label that closes its register
/// lifetimes (the next block's, or the loop end), `fetch_next` fetches the
/// following block, and `entry_label` emits the block's own label (the peeled
/// group's first block is the K-loop skip target, which places it, so it passes
/// `false`).
#[allow(clippy::too_many_arguments)]
pub fn block(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end_label: &str,
    entry_label: bool,
    lds: Lds,
) -> Result<Lds, String> {
    kloop::block_with(
        wg,
        g,
        tag,
        h,
        fetch_next,
        end_label,
        entry_label,
        |_| Ok(()),
        lds,
    )
}

/// [`block`] that also advances buffer descriptor `step` ([`step_descriptor`])
/// as soon as the block's slab-1 fetch has issued: the production K-loop's
/// odd block moves the even descriptor there, between that fetch and its own
/// next-block fetch, which is inside one block and so cannot be expressed
/// between two [`block`] calls.
#[allow(clippy::too_many_arguments)]
pub fn block_stepping(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end_label: &str,
    entry_label: bool,
    step: usize,
    lds: Lds,
) -> Result<Lds, String> {
    if step > 1 {
        return Err(format!("block {tag}: descriptor {step} is not 0 or 1"));
    }
    kloop::block_with(
        wg,
        g,
        tag,
        h,
        fetch_next,
        end_label,
        entry_label,
        |b| kloop::step_descriptor(b, g, step),
        lds,
    )
}

/// [`block`] for a sparse spec (`Mma::S24X64` / `Mma::S24X32`): the same
/// arguments, ring transitions, publication, next-block fetches and fold, with
/// SWMMAC in place of WMMA. The weights must be 2:4-packed
/// (`sparse24::pack_block`). Refuses a dense spec.
#[allow(clippy::too_many_arguments)]
pub fn sparse_block(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end_label: &str,
    entry_label: bool,
    lds: Lds,
) -> Result<Lds, String> {
    kloop::sparse_block_with(
        wg,
        g,
        tag,
        h,
        fetch_next,
        end_label,
        entry_label,
        |_| Ok(()),
        lds,
    )
}

/// [`sparse_block`] that also advances descriptor `step` after the block's
/// slab-1 fetch, as [`block_stepping`] does.
#[allow(clippy::too_many_arguments)]
pub fn sparse_block_stepping(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    fetch_next: bool,
    end_label: &str,
    entry_label: bool,
    step: usize,
    lds: Lds,
) -> Result<Lds, String> {
    if step > 1 {
        return Err(format!("block {tag}: descriptor {step} is not 0 or 1"));
    }
    kloop::sparse_block_with(
        wg,
        g,
        tag,
        h,
        fetch_next,
        end_label,
        entry_label,
        |b| kloop::step_descriptor(b, g, step),
        lds,
    )
}

/// `s_add_nc_u64` of the K-group stride onto activation descriptor `which`
/// (`0` even block, `1` odd block).
pub fn step_descriptor(b: &mut Builder, g: &Gen, which: usize) -> Result<(), String> {
    kloop::step_descriptor(b, g, which)
}

// ---- the fine steps of a dense K128 block -------------------------------------
//
// `block` is `kloop::block_with`, a composition of exactly these (the generator
// and the `.rip` natives call the same functions), so a script that spells the
// block out with them, in the generator's order, re-emits `block`'s bytes, and
// a script that moves a step is checked by the same typed core. The rings keep
// their typing: the publishes want a `Free` next buffer and leave it `Writing`
// with a `Pending` token (`Staged`), the fragment and scale loads read a
// `Published` current buffer, and only a drained `Writing` buffer rotates.

/// The head of a dense block on plane `h`: its own label (`entry_label`) and
/// the lifetimes of the registers its phases alias; see [`block`] for `tag` and
/// `end_label`. `ra_cur` / `rd_cur` are the current-buffer indices of the
/// payload-A and token-scale rings: a block on plane `h` starts with payload
/// slot 0 and metadata plane `h`.
#[allow(clippy::too_many_arguments)]
pub fn block_begin(
    wg: &mut Wg,
    g: &Gen,
    tag: &str,
    h: usize,
    end_label: &str,
    entry_label: bool,
    ra_cur: usize,
    rd_cur: usize,
) -> Result<(), String> {
    kloop::block_begin(wg, g, tag, h, end_label, entry_label, ra_cur, rd_cur)
}

/// A block's own label and its two sub-block labels: `.Liu4_<tag>`,
/// `.Liu4_<tag>_s0` (placed by [`scale_loads`]) and `.Liu4_<tag>_s1` (placed by
/// [`scale_tail`]).
pub fn sub_labels(tag: &str) -> (String, String, String) {
    kloop::sub_labels(tag)
}

/// Fragment loads of both 32-K sub-blocks from the payload rings' current
/// buffer (per sub-block one W load and two A loads). The buffer must be
/// `Published`; the next buffer may be `Free` or `Writing`.
pub fn fragments<N: peacemaker_author::State>(
    wg: &mut Wg,
    g: &Gen,
    ra: &Ring<PayA, Published, N>,
    rw: &Ring<PayW, Published, N>,
) -> Result<(), String> {
    kloop::fragments(wg, g, ra, rw)
}

/// Phase B's token-scale (`d`) loads, from the plane the `d` ring is on.
pub fn d_loads<N: peacemaker_author::State>(
    wg: &mut Wg,
    g: &Gen,
    rd: &Ring<MetaD, Published, N>,
) -> Result<(), String> {
    kloop::d_loads(wg, g, rd)
}

/// Phase B's row-scale loads of row group 0 and half of row group 1; places
/// the block's `.Liu4_<tag>_s0` label `x0` first.
pub fn scale_loads<N: peacemaker_author::State>(
    wg: &mut Wg,
    g: &Gen,
    x0: &str,
    rs: &Ring<MetaS, Published, N>,
) -> Result<(), String> {
    kloop::scale_loads(wg, g, x0, rs)
}

/// Phase B's last row-scale load; places the block's `.Liu4_<tag>_s1` label
/// `x1` first.
pub fn scale_tail<N: peacemaker_author::State>(
    wg: &mut Wg,
    g: &Gen,
    x1: &str,
    rs: &Ring<MetaS, Published, N>,
) -> Result<(), String> {
    kloop::scale_tail(wg, g, x1, rs)
}

/// The eight WMMAs of 32-K sub-block `sb` (`0` or `1`) in bundle order.
/// `first` seeds the accumulators instead of accumulating onto them; `fold`
/// interleaves the fold's stage 1 between them (phase B's second sub-block).
/// Registers only: the fragments come from [`fragments`].
pub fn wmma_group(
    b: &mut Builder,
    g: &Gen,
    sb: usize,
    first: bool,
    fold: bool,
) -> Result<(), String> {
    if sb > 1 {
        return Err(format!("wmma_group: sub-block {sb} is not 0 or 1"));
    }
    kloop::wmma_group(b, g, sb, first, fold)
}

/// Publish the staged slab (the prefetch registers) into the payload rings'
/// `Free` next buffer: the weight sign flip, then the stores. Both rings come
/// back `Writing` with their `Pending` tokens.
pub fn publish_slab<C: peacemaker_author::State>(
    wg: &mut Wg,
    g: &Gen,
    ra: Ring<PayA, C, Free>,
    rw: Ring<PayW, C, Free>,
) -> Result<(Staged<PayA, C>, Staged<PayW, C>), String> {
    publish::publish_slab(wg, g, ra, rw)
}

/// Publish the next block's token scale (`d`) and converted row scale (`sc`)
/// into the metadata rings' `Free` next buffer.
pub fn publish_meta<C: peacemaker_author::State>(
    wg: &mut Wg,
    g: &Gen,
    rd: Ring<MetaD, C, Free>,
    rs: Ring<MetaS, C, Free>,
) -> Result<(Staged<MetaD, C>, Staged<MetaS, C>), String> {
    publish::publish_meta(wg, g, rd, rs)
}

/// Fetch slab 1 of block `h` (K bytes 32..63) into the prefetch registers.
pub fn fetch_slab1(b: &mut Builder, g: &Gen, h: usize) -> Result<(), String> {
    check_plane("fetch_slab1", h)?;
    publish::fetch_slab1(b, g, h)
}

/// Fetch the next group's even block's slab 1 before the group offset
/// advances at trip end (the odd block's form of [`fetch_slab1`]).
pub fn fetch_slab1_next_group(b: &mut Builder, g: &Gen) -> Result<(), String> {
    publish::fetch_slab1_next_group(b, g)
}

/// Next block's slab-0 fetch (from block `h`) into the prefetch registers.
pub fn fetch_next(b: &mut Builder, g: &Gen, h: usize) -> Result<(), String> {
    check_plane("fetch_next", h)?;
    publish::fetch_next(b, g, h)
}

/// Next block's token scale and packed row header (from block `h`).
pub fn fetch_next_meta(b: &mut Builder, g: &Gen, h: usize) -> Result<(), String> {
    check_plane("fetch_next_meta", h)?;
    publish::fetch_next_meta(b, g, h)
}

/// The fold's stage 2: scale and accumulate the block's products.
pub fn fold_scale(b: &mut Builder, g: &Gen) -> Result<(), String> {
    super::fold::scale(b, g)
}

fn check_plane(what: &str, h: usize) -> Result<(), String> {
    if h > 1 {
        return Err(format!("{what}: plane {h} is not 0 or 1"));
    }
    Ok(())
}

/// The K-loop's trip end: odd-block descriptor and group offset advance, the
/// trip counter drops by one (kloop.rs `Trip end`).
pub fn trip_end(wg: &mut Wg, g: &Gen) -> Result<(), String> {
    kloop::trip_end(wg, g)
}

/// Compare the trip counter (`K/256 - 1`, derived from the `K` kernel
/// argument only) with zero: `cmp` is `s_cmp_eq_u32` (no full trip) or
/// `s_cmp_lg_u32` (another trip), or another scalar compare mnemonic of the
/// 32-bit family (`s_cmp_{eq,lg,lt,le,gt,ge}_{u32,i32}`); anything else is
/// refused, so the text cannot smuggle in another instruction.
pub fn trips_cmp(wg: &mut Wg, g: &Gen, cmp: &str) -> Result<WgUniform<Scc>, String> {
    let ok = ["eq", "lg", "lt", "le", "gt", "ge"].iter().any(|c| {
        ["u32", "i32"]
            .iter()
            .any(|t| cmp == format!("s_cmp_{c}_{t}"))
    });
    if !ok {
        return Err(format!(
            "trips_cmp: `{cmp}` is not a 32-bit scalar compare (s_cmp_eq_u32, s_cmp_lg_u32, ...)"
        ));
    }
    kloop::trips_cmp(wg, g, cmp)
}

/// The store path of the spec's epilogue family. Places [`EPI`].
pub fn epilogue(wg: &mut Wg, g: &Gen, end: &End) -> Result<(), String> {
    epi::emit(wg, g, end)
}

/// The fused projection's GDN preparation (after the stores); it consumes the
/// LDS rings. `Epi::QkvzaGdn` only.
pub fn gdn_epilogue(wg: &mut Wg, g: &Gen, lds: Lds) -> Result<(), String> {
    if g.spec.epi != Epi::QkvzaGdn {
        return Err("gdn_epilogue: only the fused QKVZ+GDN projection has one".into());
    }
    gdn::emit(wg, g, lds)
}

/// `.Liu4_<tag>`: the label a block named `tag` places.
pub fn label(tag: &str) -> String {
    kloop::label(tag)
}

/// The label of the peeled group's block `h` (`.Liu4_t<h>`).
pub fn tail_label(h: usize) -> String {
    kloop::tail_label(h)
}
