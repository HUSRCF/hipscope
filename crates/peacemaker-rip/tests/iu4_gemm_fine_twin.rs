// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The fine K-loop natives (`iu4::block_begin`, `fragments`, `wmma`,
//! `publish_slab`, `wait`, ..., and the sparse `iu4::sparse_block_begin`,
//! `sparse_fragments`, `sparse_wmma`, `sparse_scale_loads`, `sparse_scale_tail`;
//! cookbook 17). `kernels/iu4_gemm_fine.rip` writes the dense and the 2:4-sparse
//! K128 blocks with only them; with `WAIT = "after"` it re-emits the shipped
//! kernel's bytes for every dense and sparse variant, and the other two
//! placements re-emit the generator's `Tune { wait: Mid | Before }` (dense), and `WAIT = "window"` its
//! `Tune { wait: Window }` (the 128x128x8 tile: the schedule the shipped 27B gate/up `_b1s` export runs);
//! `RING = 3` writes the depth-3 K-loop (third LDS slab, second prefetch set,
//! entry branch, three-group loop body, peeled tail) and re-emits the
//! generator's `Tune { ring: Three }` bytes for every wait placement.
//! Every typing rule of the split (publish into a non-`Free` buffer, rotate
//! before the wait, read or WMMA of an unpublished slot, a ring of the wrong
//! kind, a dense step on a sparse spec and the reverse) is a refusal at build
//! time.

use peacemaker_isa::kernels::iu4_gemm::{
    self, ALayout, Cacc, Epi, Fold, Mma, RingDepth, Spec, Tile, Tune, WaitPlace,
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
    let mut windowed = 0;
    for (place, tune) in [
        ("mid", WaitPlace::MidMma),
        ("before", WaitPlace::BeforeMma),
        ("window", WaitPlace::Window),
    ] {
        let src = with_wait(place);
        for spec in dense_specs() {
            let name = spec.variant_name();
            let shipped = iu4_gemm::emit(spec).unwrap();
            let tuned = Spec {
                tune: Tune {
                    wait: tune,
                    ..Tune::default()
                },
                ..spec
            };
            if tune == WaitPlace::Window {
                // The window placement is the 128x128x8 tile's (its raw sub-block-1 WMMA group names that
                // tile's registers): the generator refuses the 16-wave tile.
                if tuned.validate().is_err() {
                    assert_eq!(spec.tile, Tile::T256x128x16, "{name}");
                    continue;
                }
                windowed += 1;
            }
            let want = iu4_gemm::emit(tuned).unwrap_or_else(|e| panic!("{name} {place}: {e}"));
            let got = build(&src, &name).unwrap_or_else(|e| panic!("{name} {place}: {e}"));
            assert_same(&format!("{name} wait={place}"), &want, &got);
            assert!(
                got.s_text != shipped.s_text,
                "{name}: wait={place} must change the schedule"
            );
        }
    }
    assert_eq!(
        windowed, 11,
        "the 128x128x8 dense variants take the window placement"
    );
}

/// The shipped `_b1s` module's 27B gate/up export (`gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s`) is the `WAIT = "window"`
/// twin: the module emission's symbol equals the script's bytes, and every other symbol of that module equals the
/// `WAIT = "after"` script's.
#[test]
fn the_shipped_gate_up_export_is_the_window_twin() {
    let (emitted, _, _) = iu4_gemm::emit_module(
        Fold::K128,
        Tile::T128x128x8,
        Cacc::One,
        ALayout::Slab,
        Arch::Gfx1201,
    )
    .unwrap();
    let window = with_wait("window");
    let mut windowed = 0;
    for e in &emitted {
        let (src, name) = if e.proof.variant == "k128-128x128x8-cacc1-slab-silu-wwindow" {
            windowed += 1;
            (&window, "k128-128x128x8-cacc1-slab-silu")
        } else {
            (&FINE.to_owned(), e.proof.variant.as_str())
        };
        let rip = build(src, name).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_same(&e.proof.variant, e, &rip);
    }
    assert_eq!(windowed, 1, "exactly the gate/up slab export is windowed");
}

fn with_ring(src: &str) -> String {
    let from = "const RING: usize = 2;";
    assert!(src.contains(from), "the fine twin no longer has `{from}`");
    src.replacen(from, "const RING: usize = 3;", 1)
}

/// `RING = 3` (every `WAIT`) emits exactly the generator's depth-3 bytes for every dense variant that has a depth-3
/// generator, and the depth-3 program is not the shipped one. The fused GDN projection has none: the script refuses
/// it with the generator's own reason.
#[test]
fn a_ring_3_script_equals_the_generators_tune() {
    let mut checked = 0;
    for (place, wait) in [
        ("after", WaitPlace::AfterMma),
        ("mid", WaitPlace::MidMma),
        ("before", WaitPlace::BeforeMma),
    ] {
        let src = with_ring(&with_wait(place));
        for spec in dense_specs() {
            let name = spec.variant_name();
            let tuned = Spec {
                tune: Tune {
                    ring: RingDepth::Three,
                    wait,
                    ..Tune::default()
                },
                ..spec
            };
            if tuned.validate().is_err() {
                let e = build(&src, &name)
                    .err()
                    .expect("the script must be refused");
                assert!(e.contains("ring depth 3"), "{name}: {e}");
                continue;
            }
            let want = iu4_gemm::emit(tuned).unwrap_or_else(|e| panic!("{name} {place}: {e}"));
            let got = build(&src, &name).unwrap_or_else(|e| panic!("{name} {place} ring 3: {e}"));
            assert_same(&format!("{name} ring=3 wait={place}"), &want, &got);
            assert!(
                got.s_text != iu4_gemm::emit(spec).unwrap().s_text,
                "{name}: ring 3 must change the program"
            );
            checked += 1;
        }
    }
    // 12 dense variants with a depth-3 generator, three wait placements.
    assert_eq!(checked, 36);
}

/// Every sparse variant the generator publishes (`Spec::validate` accepts it):
/// the 2:4-sparse SWMMAC K-loops on their tile, epilogue and layout.
fn sparse_specs() -> Vec<Spec> {
    let mut out = Vec::new();
    for mma in [Mma::S24X64, Mma::S24X32] {
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
                        mma,
                        tune: Tune::default(),
                    };
                    if spec.validate().is_ok() {
                        out.push(spec);
                    }
                }
            }
        }
    }
    out
}

/// The sparse K-loop block written with only the fine natives
/// (`fine_sparse_block`) emits the shipped bytes, and the proof, for every
/// sparse variant: no opaque `iu4::sparse_block*` remains in the script.
#[test]
fn the_fine_sparse_composition_emits_the_shipped_bytes() {
    for opaque in ["iu4::sparse_block(", "iu4::sparse_block_stepping("] {
        assert!(
            !FINE.contains(opaque),
            "the fine twin still calls the opaque `{opaque}`"
        );
    }
    let specs = sparse_specs();
    let names: Vec<String> = specs.iter().map(|s| s.variant_name()).collect();
    assert_eq!(
        names,
        [
            "k128-128x128x8-cacc1-set-s24x64",
            "k128-128x128x8-cacc1-add-s24x64",
            "k128-128x128x8-cacc1-set-s24x32",
            "k128-128x128x8-cacc1-add-s24x32",
        ],
        "the sparse variant set changed"
    );
    for spec in specs {
        let name = spec.variant_name();
        let rust = iu4_gemm::emit(spec).unwrap_or_else(|e| panic!("{name}: {e}"));
        let rip = build(FINE, &name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_same(&name, &rust, &rip);
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

// ---- the sparse steps ---------------------------------------------------------

const X64: &str = "k128-128x128x8-cacc1-set-s24x64";
const X32: &str = "k128-128x128x8-cacc1-add-s24x32";

fn refused_for(src: &str, variant: &str) -> String {
    build(src, variant)
        .err()
        .expect("the script must be refused")
}

fn code_for(src: &str, variant: &str) -> &'static str {
    let c = peacemaker_rip::check(src, "fine.rip", Arch::Gfx1201, variant);
    c.diagnostics
        .first()
        .unwrap_or_else(|| panic!("no diagnostic"))
        .code
}

/// A sparse fine step on the dense spec, and a dense step on a sparse spec
/// (either of its two SWMMAC forms), is refused before anything is emitted.
#[test]
fn a_fine_step_of_the_other_mma_family_is_refused() {
    // (dense call in `fine_block`, its sparse twin in `fine_sparse_block`, the
    // sparse specs whose block runs that twin: x64 has no scale tail, and its
    // scale loads are the first, `x64` branch's)
    for (dense, sparse, variants) in [
        (
            "iu4::block_begin(wg, g, tag, h, end_label, entry_label, lds)",
            "iu4::sparse_block_begin(wg, g, tag, h, end_label, entry_label, lds)",
            &[X64, X32][..],
        ),
        (
            "iu4::fragments(wg, g, lds)",
            "iu4::sparse_fragments(wg, g, lds)",
            &[X64, X32][..],
        ),
        (
            "iu4::wmma(wg, g, 0, true, false, lds)",
            "iu4::sparse_wmma(wg, g, 0, true, false, lds)",
            &[X64, X32][..],
        ),
        (
            "iu4::scale_loads(wg, g, tag, lds)",
            "iu4::sparse_scale_loads(wg, g, tag, lds)",
            &[X64][..],
        ),
        (
            "iu4::scale_tail(wg, g, tag, lds)",
            "iu4::sparse_scale_tail(wg, g, tag, lds)",
            &[X32][..],
        ),
    ] {
        // The sparse step in the dense block: the dense spec refuses it.
        let src = edit(FINE, dense, sparse);
        let e = refused_for(&src, SET);
        let native = sparse.split('(').next().unwrap();
        assert!(
            e.contains(native) && e.contains("the spec is dense"),
            "{native} on dense: {e}"
        );
        // The dense step in the sparse block: either sparse spec refuses it.
        for &variant in variants {
            let src = edit(FINE, sparse, dense);
            let e = refused_for(&src, variant);
            let native = dense.split('(').next().unwrap();
            assert!(
                e.contains(native) && e.contains("cannot run in the sparse K-loop"),
                "{native} on {variant}: {e}"
            );
        }
    }
}

/// x64's one SWMMAC bundle covers the whole K64 slab: a second sub-block group
/// is refused, and so is the `_s1` scale tail (all four scale loads are the
/// `_s0` group's). x32 has both.
#[test]
fn an_x64_block_has_no_second_sub_block() {
    let wmma1 = edit(
        FINE,
        "iu4::sparse_wmma(wg, g, 0, true, false, lds)",
        "iu4::sparse_wmma(wg, g, 1, true, false, lds)",
    );
    let e = refused_for(&wmma1, X64);
    assert!(e.contains("covers the whole K64 slab"), "{e}");

    let from =
        "    let lds = iu4::sparse_block_begin(wg, g, tag, h, end_label, entry_label, lds)?;\n";
    let tail = edit(
        FINE,
        from,
        &format!("{from}    let _tail = iu4::sparse_scale_tail(wg, g, tag, lds)?;\n"),
    );
    let e = refused_for(&tail, X64);
    assert!(e.contains("no sub-block 1"), "{e}");
    // x32: the same call builds as a step (the block then fails later, on the
    // `_s1` label the real tail places a second time).
    let e = build(&tail, X32).err().unwrap_or_default();
    assert!(!e.contains("no sub-block 1"), "{e}");
}

/// The sparse steps keep the typing rules: they read a published slot, so on
/// rings nothing has been published into (before the prologue) they are refused.
#[test]
fn a_sparse_read_or_wmma_of_an_unpublished_slot_is_refused() {
    let early = |call: &str| {
        edit(
            FINE,
            "    let lds = iu4::prologue(wg, g, end, lds)?;\n",
            &format!("    let _early = {call}?;\n    let lds = iu4::prologue(wg, g, end, lds)?;\n"),
        )
    };
    for variant in [X64, X32] {
        for (call, native) in [
            ("iu4::sparse_fragments(wg, g, lds)", "iu4::sparse_fragments"),
            (
                "iu4::sparse_scale_loads(wg, g, \"m0\", lds)",
                "iu4::sparse_scale_loads",
            ),
            (
                "iu4::sparse_wmma(wg, g, 0, true, false, lds)",
                "iu4::sparse_wmma",
            ),
        ] {
            let src = early(call);
            let e = refused_for(&src, variant);
            assert!(
                e.contains(native) && e.contains("not published"),
                "{native} {variant}: {e}"
            );
            assert_eq!(code_for(&src, variant), "R0303", "{native} {variant}");
        }
    }
}

/// A sparse script that moves a publish before the SWMMACs or drops the wait
/// is judged by the same driver: a rotate before the wait is refused.
#[test]
fn a_sparse_script_that_rotates_before_the_wait_is_refused() {
    let src = edit(
        FINE,
        "    let (da, dw) = iu4::wait(wg, lds, \"slab\")?;\n    wg.barrier((rotate(lds[0], da), rotate(lds[1], dw)))?;\n    // Phase B: slab 1 from slot 1, the fold, the next block into slot 0.\n    if fetch_next {\n        iu4::fetch_next(wg, g, h)?;\n    }\n    let lds = iu4::sparse_fragments",
        "    wg.barrier((rotate(lds[0]), rotate(lds[1])))?;\n    if fetch_next {\n        iu4::fetch_next(wg, g, h)?;\n    }\n    let lds = iu4::sparse_fragments",
    );
    for variant in [X64, X32] {
        let e = refused_for(&src, variant);
        assert!(e.contains("undrained LDS store"), "{variant}: {e}");
        assert_eq!(code_for(&src, variant), "R0301", "{variant}");
    }
}
