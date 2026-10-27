// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The `iu4_gemm::Tune` generator knobs (`ring`, `wait`, `fold_op`): the default is today's codegen, every legal
//! non-default point emits and assembles for the 27B prefill gate/up spec (`gemm_mq4g256v2_gate_up_silu_mmq_iu4_b1s_bf16`),
//! the schedule-only knobs are bit-identical to the control on the CPU emulator, `fold_op = muladd` is the exact
//! non-fused fold, and `Spec::validate` rejects what has no generator.
use iu4_emu::*;
use peacemaker_isa::kernels::iu4_gemm::{
    emit, ALayout, Epi, FoldOp, Mma, RingDepth, Spec, Tile, Tune, WaitPlace,
};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;
#[path = "support/iu4_emu.rs"]
mod iu4_emu;

const WAITS: [WaitPlace; 3] = [WaitPlace::AfterMma, WaitPlace::MidMma, WaitPlace::BeforeMma];

/// The 27B prefill gate/up node: slab activations, packed bf16 h, the production tile.
fn node() -> Spec {
    Spec {
        act: ALayout::Slab,
        ..Spec::control(Epi::GateUpSiluBf16)
    }
}

fn with(spec: Spec, tune: Tune) -> Spec {
    Spec { tune, ..spec }
}

fn bitwise_eq(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

#[test]
fn default_tune_is_the_shipped_codegen() {
    let spec = node();
    assert!(spec.tune.is_default());
    assert_eq!(
        spec.tune,
        Tune {
            ring: RingDepth::Two,
            wait: WaitPlace::AfterMma,
            fold_op: FoldOp::Fma
        }
    );
    // The default spec keeps its variant name and symbol; a tuned one keeps the symbol (the product ABI) and
    // only extends the variant name.
    assert_eq!(spec.variant_name(), "k128-128x128x8-cacc1-slab-silu-bf16");
    let tuned = with(
        spec,
        Tune {
            wait: WaitPlace::MidMma,
            fold_op: FoldOp::MulAdd,
            ..Tune::default()
        },
    );
    assert_eq!(tuned.symbol(), spec.symbol());
    assert_eq!(
        tuned.variant_name(),
        "k128-128x128x8-cacc1-slab-silu-bf16-wmid-fmuladd"
    );
}

/// Every legal wait placement emits and assembles at the 27B spec; each differs from the control, from the
/// others, and re-emits to the same bytes.
#[test]
fn wait_placements_emit_distinct_schedules() {
    let spec = node();
    let mut texts = Vec::new();
    for wait in WAITS {
        let s = with(
            spec,
            Tune {
                wait,
                ..Tune::default()
            },
        );
        s.validate().unwrap();
        let e = emit(s).unwrap_or_else(|e| panic!("wait {}: {e}", wait.name()));
        peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
        assert_eq!(
            emit(s).unwrap().s_text,
            e.s_text,
            "wait {} is not deterministic",
            wait.name()
        );
        texts.push(e.s_text);
    }
    for i in 0..texts.len() {
        for j in i + 1..texts.len() {
            assert_ne!(
                texts[i], texts[j],
                "{:?} and {:?} emit the same text",
                WAITS[i], WAITS[j]
            );
        }
    }
    assert_eq!(texts[0], emit(spec).unwrap().s_text, "after is the default");
}

/// The other epilogues and the 16-wave tile take every wait placement too.
#[test]
fn wait_placements_emit_for_every_dense_variant() {
    for (epi, act, tile) in [
        (Epi::Set, ALayout::Token, Tile::T128x128x8),
        (Epi::Add, ALayout::Slab, Tile::T128x128x8),
        (Epi::GateUpSilu, ALayout::Token, Tile::T256x128x16),
        (Epi::QkvzaGdn, ALayout::Token, Tile::T128x128x8),
    ] {
        for wait in WAITS {
            let s = Spec {
                act,
                tile,
                tune: Tune {
                    wait,
                    ..Tune::default()
                },
                ..Spec::control(epi)
            };
            let e = emit(s)
                .unwrap_or_else(|e| panic!("{} wait {}: {e}", s.variant_name(), wait.name()));
            peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
        }
    }
}

#[test]
fn fold_op_muladd_emits_and_replaces_the_fused_accumulate() {
    let spec = node();
    let base = emit(spec).unwrap().s_text;
    let s = with(
        spec,
        Tune {
            fold_op: FoldOp::MulAdd,
            ..Tune::default()
        },
    );
    s.validate().unwrap();
    let e = emit(s).unwrap();
    peacemaker_isa::native::assemble(&e.s_text, s.arch).unwrap();
    let count = |t: &str, m: &str| t.lines().filter(|l| l.trim_start().starts_with(m)).count();
    assert!(base.contains("v_dual_fmac_f32"));
    // One non-fused product and one add per accumulated element, each trip's and the peeled group's blocks.
    let muls = count(&e.s_text, "v_mul_f32_e32") - count(&base, "v_mul_f32_e32");
    let adds = count(&e.s_text, "v_add_f32_e32") - count(&base, "v_add_f32_e32");
    assert!(muls > 0 && muls == adds, "muls {muls} adds {adds}");
}

#[test]
fn validate_rejects_what_has_no_generator() {
    let spec = node();
    let e = with(
        spec,
        Tune {
            ring: RingDepth::Three,
            ..Tune::default()
        },
    )
    .validate()
    .unwrap_err();
    assert!(e.contains("ring depth 3"), "{e}");
    // The reason survives `emit`.
    let e = emit(with(
        spec,
        Tune {
            ring: RingDepth::Three,
            ..Tune::default()
        },
    ))
    .err()
    .unwrap();
    assert!(e.contains("ring depth 3"), "{e}");
    // A non-default wait placement is for the dense K-loop block only.
    let sparse = Spec {
        mma: Mma::S24X64,
        tune: Tune {
            wait: WaitPlace::MidMma,
            ..Tune::default()
        },
        ..Spec::control(Epi::Set)
    };
    let e = sparse.validate().unwrap_err();
    assert!(e.contains("wait placement mid"), "{e}");
    assert!(Spec {
        tune: Tune::default(),
        ..sparse
    }
    .validate()
    .is_ok());
}

#[test]
fn knob_strings_round_trip() {
    for (s, r) in [("2", RingDepth::Two), ("3", RingDepth::Three)] {
        assert_eq!(s.parse::<RingDepth>().unwrap(), r);
        assert_eq!(r.name(), s);
    }
    for w in WAITS {
        assert_eq!(w.name().parse::<WaitPlace>().unwrap(), w);
    }
    for f in [FoldOp::Fma, FoldOp::MulAdd] {
        assert_eq!(f.name().parse::<FoldOp>().unwrap(), f);
    }
    assert!("4".parse::<RingDepth>().is_err());
    assert!("late".parse::<WaitPlace>().is_err());
    assert!("fused".parse::<FoldOp>().is_err());
}

/// The schedule-only knobs are bit-identical to the control on the emulator (SET and ADD, token and slab, three
/// K groups so the rolled loop and the peeled tail both run).
#[test]
fn wait_placements_are_bit_identical_on_the_emulator() {
    require_rocm_tool!("llvm-mc");
    let p = Problem::synth(136, 40, 768, 0x9e37_79b9_7f4a_7c15);
    for (epi, act) in [
        (Epi::Set, ALayout::Token),
        (Epi::Set, ALayout::Slab),
        (Epi::Add, ALayout::Token),
    ] {
        let want = run_gemm(
            Spec {
                act,
                ..Spec::control(epi)
            },
            &p,
        )
        .unwrap();
        for wait in [WaitPlace::MidMma, WaitPlace::BeforeMma] {
            let got = run_gemm(
                Spec {
                    act,
                    tune: Tune {
                        wait,
                        ..Tune::default()
                    },
                    ..Spec::control(epi)
                },
                &p,
            )
            .unwrap();
            assert_eq!(
                bitwise_eq(&got, &want),
                0,
                "{epi:?} {act:?} wait {}",
                wait.name()
            );
        }
    }
}

/// `fold_op = muladd` reproduces the non-fused CPU fold bit for bit, and (on this data) is not the fused result.
#[test]
fn muladd_matches_the_non_fused_cpu_fold_bitwise() {
    require_rocm_tool!("llvm-mc");
    let p = Problem::synth(136, 40, 768, 0x9e37_79b9_7f4a_7c15);
    for (epi, act) in [(Epi::Set, ALayout::Token), (Epi::Add, ALayout::Slab)] {
        let spec = Spec {
            act,
            tune: Tune {
                fold_op: FoldOp::MulAdd,
                ..Tune::default()
            },
            ..Spec::control(epi)
        };
        let got = run_gemm(spec, &p).unwrap();
        let add = epi == Epi::Add;
        assert_eq!(
            bitwise_eq(&got, &reference_with(&p, add, false)),
            0,
            "{epi:?} {act:?}: muladd must equal the non-fused fold"
        );
        assert!(
            bitwise_eq(&got, &reference_with(&p, add, true)) > 0,
            "{epi:?} {act:?}: this data does not separate fused from non-fused"
        );
    }
}
