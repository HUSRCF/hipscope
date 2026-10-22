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
//! cannot read a `Free` buffer. The fused per-WMMA fold stage
//! (`fold::unbias_after_wmma`) stays inside [`block`]: it is interleaved within
//! one instruction bundle.
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
use super::{epilogue as epi, gdn_epilogue as gdn, kloop, prologue as pro, Epi};
use crate::Builder;
use peacemaker_author::{End, Scc, WgUniform};

pub use super::{
    FreeRings, Gen, Lds, MetaD, MetaS, PayA, PayW, Wg, END, ENTRY, EPI, GDN, K_BEGIN, K_LOOP,
    K_LOOP_END,
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

/// `s_add_nc_u64` of the K-group stride onto activation descriptor `which`
/// (`0` even block, `1` odd block).
pub fn step_descriptor(b: &mut Builder, g: &Gen, which: usize) -> Result<(), String> {
    kloop::step_descriptor(b, g, which)
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
