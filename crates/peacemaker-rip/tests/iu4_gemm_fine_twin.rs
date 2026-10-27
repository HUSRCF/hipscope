// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The fine K-loop natives (`iu4::block_begin`, `fragments`, `wmma`,
//! `publish_slab`, `wait`, ..., cookbook 17). `kernels/iu4_gemm_fine.rip`
//! writes the dense K128 block with only them; with `WAIT = "after"` it
//! re-emits the shipped kernel's bytes, and the other two placements re-emit
//! the generator's `Tune { wait: Mid | Before }`. Every typing rule of the
//! split (publish into a non-`Free` buffer, rotate before the wait, read or
//! WMMA of an unpublished slot, a ring of the wrong kind) is a refusal at
//! build time.

use peacemaker_isa::kernels::iu4_gemm::{
    self, ALayout, Cacc, Epi, Fold, Mma, Spec, Tile, Tune, WaitPlace,
};
use peacemaker_isa::Arch;

const FINE: &str = include_str!("../../peacemaker-isa/src/kernels/iu4_gemm_fine.rip");
const SET: &str = "k128-128x128x8-cacc1-set";

fn dense_specs() -> Vec<Spec> {
    let mut out = Vec::new();
    for tile in [Tile::T128x128x8, Tile::T256x128x16] {
        for epi in [
            Epi::Set,
            Epi::Add,
            Epi::AddSigmoid,
            Epi::GateUpSilu,
            Epi::GateUpSiluBf16,
            Epi::QkvzaGdn,
        ] {
            for act in [ALayout::Token, ALayout::Slab] {
                let spec = Spec {
                    fold: Fold::K128,
                    tile,
                    cacc: Cacc::One,
                    epi,
                    act,
                    arch: Arch::Gfx1201,
                    mma: Mma::Dense,
                    tune: Tune::default(),
                };
                if spec.validate().is_ok() {
                    out.push(spec);
                }
            }
        }
    }
    out
}

fn build(src: &str, variant: &str) -> Result<peacemaker_isa::Emitted, String> {
    peacemaker_rip::compile_named(src, "fine.rip", Arch::Gfx1201, variant)
}

fn with_wait(place: &str) -> String {
    let from = "const WAIT: &str = \"after\";";
    assert!(FINE.contains(from), "the fine twin no longer has `{from}`");
    FINE.replacen(from, &format!("const WAIT: &str = \"{place}\";"), 1)
}

fn edit(src: &str, from: &str, to: &str) -> String {
    assert!(
        src.contains(from),
        "the fine twin no longer contains `{from}`"
    );
    src.replacen(from, to, 1)
}

fn assert_same(what: &str, rust: &peacemaker_isa::Emitted, rip: &peacemaker_isa::Emitted) {
    assert!(
        rust.s_text == rip.s_text,
        "{what}: {}",
        rust.s_text
            .lines()
            .zip(rip.s_text.lines())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| format!("line {}: rust `{a}` vs rip `{b}`", i + 1))
            .unwrap_or_else(|| "lengths differ".into())
    );
    let mut want = serde_json::to_value(&rust.proof).unwrap();
    let mut got = serde_json::to_value(&rip.proof).unwrap();
    // The variant name carries the tune suffix on the generator's side only.
    want["variant"] = got["variant"].clone();
    got["variant"] = want["variant"].clone();
    assert_eq!(want, got, "{what}: proof");
}

/// The K-loop block written with only the fine natives emits the shipped bytes
/// for every dense variant.
#[test]
fn the_fine_composition_emits_the_shipped_bytes() {
    let specs = dense_specs();
    assert_eq!(specs.len(), 14, "the dense variant count changed");
    for spec in specs {
        let name = spec.variant_name();
        let rust = iu4_gemm::emit(spec).unwrap_or_else(|e| panic!("{name}: {e}"));
        let rip = build(FINE, &name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_same(&name, &rust, &rip);
    }
}

/// A script that moves the wait (here: where the publish sits between the WMMA
/// groups) compiles and equals the generator's `Tune { wait }` bytes.
#[test]
fn a_wait_moved_script_equals_the_generators_tune() {
    for (place, tune) in [("mid", WaitPlace::MidMma), ("before", WaitPlace::BeforeMma)] {
        let src = with_wait(place);
        for spec in dense_specs() {
            let name = spec.variant_name();
            let shipped = iu4_gemm::emit(spec).unwrap();
            let want = iu4_gemm::emit(Spec {
                tune: Tune {
                    wait: tune,
                    ..Tune::default()
                },
                ..spec
            })
            .unwrap_or_else(|e| panic!("{name} {place}: {e}"));
            let got = build(&src, &name).unwrap_or_else(|e| panic!("{name} {place}: {e}"));
            assert_same(&format!("{name} wait={place}"), &want, &got);
            assert!(
                got.s_text != shipped.s_text,
                "{name}: wait={place} must change the schedule"
            );
        }
    }
}

/// The sparse variants keep the opaque sparse blocks and still emit the shipped bytes.
#[test]
fn the_sparse_variants_are_untouched() {
    for (epi, mma) in [
        (Epi::Set, Mma::S24X64),
        (Epi::Set, Mma::S24X32),
        (Epi::Add, Mma::S24X64),
        (Epi::Add, Mma::S24X32),
    ] {
        let spec = Spec {
            fold: Fold::K128,
            tile: Tile::T128x128x8,
            cacc: Cacc::One,
            epi,
            act: ALayout::Token,
            arch: Arch::Gfx1201,
            mma,
            tune: Tune::default(),
        };
        let variant = spec.variant_name();
        let rust = iu4_gemm::emit(spec).unwrap();
        let rip = build(FINE, &variant).unwrap();
        assert_same(&variant, &rust, &rip);
    }
}

// ---- the typing rules ---------------------------------------------------------

/// A script that runs `body` after the prologue of the `set` variant, inside the
/// K-loop's first block head, returning `lds`. `body` sees `wg`, `g`, `lds` and
/// the plane `h = 0`.
fn script(body: &str) -> String {
    let from = "    let lds = iu4::block_begin(wg, g, tag, h, end_label, entry_label, lds)?;\n";
    let to = format!("{from}    let lds = {{ {body} }};\n");
    let src = edit(FINE, from, &to);
    // Only the first dense block runs these lines in a failing build.
    src
}

fn refused(src: &str) -> String {
    build(src, SET).err().expect("the script must be refused")
}

/// The stable diagnostic code of the script's first error.
fn code(src: &str) -> &'static str {
    let c = peacemaker_rip::check(src, "fine.rip", Arch::Gfx1201, SET);
    c.diagnostics
        .first()
        .unwrap_or_else(|| panic!("no diagnostic"))
        .code
}

/// Publishing the next slab into a buffer that is already `Writing` (the first
/// publish's stores have not rotated) is refused: a publish wants a `Free` one.
#[test]
fn publish_into_a_non_free_buffer_is_refused() {
    let src = script("let lds = iu4::publish_slab(wg, g, lds)?; iu4::publish_slab(wg, g, lds)?");
    let e = refused(&src);
    assert!(
        e.contains("iu4::publish_slab") && e.contains("Writing") && e.contains("Free"),
        "{e}"
    );
    assert_eq!(code(&src), "R0399");
    // The metadata publish has the same rule.
    let src = script("let lds = iu4::publish_meta(wg, g, lds)?; iu4::publish_meta(wg, g, lds)?");
    let e = refused(&src);
    assert!(
        e.contains("iu4::publish_meta") && e.contains("Writing"),
        "{e}"
    );
    assert_eq!(code(&src), "R0399");
}

/// A barrier that rotates a published-but-not-waited buffer is refused: the
/// stores must be drained first (`s_barrier` does not wait for them).
#[test]
fn rotating_before_the_wait_is_refused() {
    let src = script(
        "let lds = iu4::publish_slab(wg, g, lds)?; wg.barrier((rotate(lds[0]), rotate(lds[1])))?; lds",
    );
    let e = refused(&src);
    assert!(e.contains("undrained LDS store"), "{e}");
    assert_eq!(code(&src), "R0301");
    // With the wait the same lines build the slab handoff (the block then
    // fails later, at a ring it has not finished: only the barrier is judged).
    let ok = script(
        "let lds = iu4::publish_slab(wg, g, lds)?; let (da, dw) = iu4::wait(wg, lds, \"slab\")?; wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?; lds",
    );
    let e = build(&ok, SET).err().unwrap_or_default();
    assert!(!e.contains("undrained LDS store"), "{e}");
}

/// Waiting on a ring that has no store in flight is refused.
#[test]
fn waiting_without_a_publish_is_refused() {
    let src = script("let _d = iu4::wait(wg, lds, \"slab\")?; lds");
    let e = refused(&src);
    assert!(e.contains("iu4::wait") && e.contains("no store"), "{e}");
    assert_eq!(code(&src), "R0399");
    let src = script("let _d = iu4::wait(wg, lds, \"rings\")?; lds");
    let e = refused(&src);
    assert!(e.contains("not a ring set"), "{e}");
}

/// A WMMA group and the fragment and scale loads read a published slot: on
/// rings that nothing has been published into (before the prologue), they are
/// refused.
#[test]
fn reading_or_wmma_of_an_unpublished_slot_is_refused() {
    let early = |call: &str| {
        edit(
            FINE,
            "    let lds = iu4::prologue(wg, g, end, lds)?;\n",
            &format!("    let _early = {call}?;\n    let lds = iu4::prologue(wg, g, end, lds)?;\n"),
        )
    };
    for (call, native) in [
        ("iu4::fragments(wg, g, lds)", "iu4::fragments"),
        ("iu4::d_loads(wg, g, lds)", "iu4::d_loads"),
        ("iu4::scale_loads(wg, g, \"m0\", lds)", "iu4::scale_loads"),
        ("iu4::scale_tail(wg, g, \"m0\", lds)", "iu4::scale_tail"),
        ("iu4::wmma(wg, g, 0, true, false, lds)", "iu4::wmma"),
    ] {
        let src = early(call);
        let e = refused(&src);
        assert!(
            e.contains(native) && e.contains("not published"),
            "{native}: {e}"
        );
        assert_eq!(code(&src), "R0303", "{native}");
    }
}

/// A ring of the wrong kind (the weight ring where the activation ring is
/// expected) fails the tag check of the typed lend.
#[test]
fn a_ring_of_the_wrong_kind_is_refused_by_the_fine_natives() {
    let src = script("let lds = [lds[1], lds[0], lds[2], lds[3]]; iu4::publish_slab(wg, g, lds)?");
    let e = refused(&src);
    assert!(e.contains("tag"), "{e}");
    assert_eq!(code(&src), "R0399");
}

/// The fine natives need the typed lend: a ring the script made cannot enter them.
#[test]
fn a_ring_the_script_made_cannot_enter_the_fine_natives() {
    let src = script(
        "let a = wg.lds(\"x\", 0, 64)?; let b = wg.lds(\"y\", 64, 64)?; let r = Ring::new(a, b); iu4::publish_slab(wg, g, [r, r, lds[2], lds[3]])?",
    );
    let e = refused(&src);
    assert!(
        e.contains("not given by a typed body") || e.contains("LDS"),
        "{e}"
    );
    assert_eq!(code(&src), "R0399");
}

/// A fine step under wave-uniform control is refused like a block.
#[test]
fn a_fine_step_under_wave_uniform_control_is_refused() {
    let src = script(
        "let c = wg.scmp(Instruction::new(\"s_cmp_eq_u32 s0, 0\", vec![], vec![s(0)]))?; wg.skip_if(c, \".Lskip\", lds, |wg, lds| { iu4::fragments(wg, g, lds) })?",
    );
    let e = refused(&src);
    assert!(
        e.contains("workgroup scope") || e.contains("cannot run under"),
        "{e}"
    );
    assert_eq!(code(&src), "R0304");
}

/// The loop rule still covers the fine natives: a K-loop trip that leaves a
/// ring `Writing` (a publish no barrier rotated) at the back edge is refused.
#[test]
fn a_trip_that_leaves_a_publish_in_flight_is_refused() {
    let src = edit(
        FINE,
        "            iu4::trip_end(wg, g)?;\n",
        "            let lds = iu4::publish_slab(wg, g, lds)?;\n            iu4::trip_end(wg, g)?;\n",
    );
    let e = refused(&src);
    assert!(e.contains("Writing") || e.contains("ownership"), "{e}");
    assert_eq!(code(&src), "R0399");
}
