// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The production dense `iu4_gemm` kernel on the PeaceMaker CPU emulator, through the reusable harness of
//! `support/iu4_emu.rs`: hand-checkable cases (`Y = K` exactly, signed extremes) and a random case against the exact
//! CPU fold, bitwise, for SET (token and slab activations) and ADD.
use iu4_emu::*;
use peacemaker_isa::kernels::iu4_gemm::{ALayout, Epi, Spec, Tune, WaitPlace};

#[macro_use]
#[path = "support/rocm.rs"]
mod rocm;
#[path = "support/iu4_emu.rs"]
mod iu4_emu;

fn control(epi: Epi) -> Spec {
    Spec::control(epi)
}

fn bitwise_eq(got: &[f32], want: &[f32]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

/// All weight codes 9 (signed +1 after `q - 8`), all activation q = +1, unit scales: `Y = K` exactly. M = 136 and
/// N = 40 cover a second, partial row tile and a partial token tile.
#[test]
fn dense_all_ones_gives_k_exactly() {
    require_rocm_tool!("llvm-mc");
    for (k, epi) in [(256, Epi::Set), (512, Epi::Set), (256, Epi::Add)] {
        let p = Problem::uniform(136, 40, k, 9, 1);
        let y = run_gemm(control(epi), &p).unwrap();
        assert!(
            y.iter().all(|&v| v == k as f32),
            "{epi:?} K={k}: y[0]={} y[last]={}",
            y[0],
            y[y.len() - 1]
        );
    }
}

/// Signed extremes: code 0 (-8) against q = -8 gives +64 per K, code 15 (+7) against q = -8 gives -56.
#[test]
fn dense_signed_extremes_are_exact() {
    require_rocm_tool!("llvm-mc");
    for (code, q) in [(0u8, -8), (15, -8), (0, 7)] {
        let p = Problem::uniform(128, 16, 256, code, q);
        let y = run_gemm(control(Epi::Set), &p).unwrap();
        let want = (256 * (i32::from(code) - 8) * q) as f32;
        assert!(y.iter().all(|&v| v == want), "code {code} q {q}: {}", y[0]);
    }
}

/// Random weights and activations against the CPU fold, bitwise: SET with token and slab activations, ADD.
#[test]
fn dense_random_matches_the_cpu_fold_bitwise() {
    require_rocm_tool!("llvm-mc");
    let p = Problem::synth(136, 40, 384 + 128, 0x9e37_79b9_7f4a_7c15);
    for (epi, act) in [
        (Epi::Set, ALayout::Token),
        (Epi::Set, ALayout::Slab),
        (Epi::Add, ALayout::Token),
    ] {
        let spec = Spec {
            act,
            ..control(epi)
        };
        let got = run_gemm(spec, &p).unwrap();
        let want = reference(&p, epi == Epi::Add);
        assert_eq!(
            bitwise_eq(&got, &want),
            0,
            "{epi:?} {act:?}: first got {} want {}",
            got[0],
            want[0]
        );
    }
}

/// `WaitPlace::Window` (the shipped schedule of the 27B gate/up `_b1s` export, `Tune::shipped_gate_up_slab`) is
/// scheduling only: the split B1 rendezvous with sub-block 1's WMMA group in its window and `d_loads` hoisted into phase
/// A leave every result bit equal to the default schedule's and to the CPU fold's, at K = 256 (peeled group only),
/// 512, 1024 and 5120 (the 27B hidden size: a rolled loop of full trips). SiLU is not driven by the emulator
/// (`run_gemm` takes set/add): the epilogue follows the K-loop and is untouched by the schedule, so SET and ADD on the
/// slab layout (the export's activation layout) and the token layout cover the window block.
#[test]
fn window_schedule_is_bit_identical_to_the_default() {
    require_rocm_tool!("llvm-mc");
    for k in [256, 512, 1024, 5120] {
        let p = Problem::synth(136, 40, k, 0x9e37_79b9_7f4a_7c15 ^ k as u64);
        for (epi, act) in [
            (Epi::Set, ALayout::Slab),
            (Epi::Add, ALayout::Slab),
            (Epi::Set, ALayout::Token),
        ] {
            let base = Spec {
                act,
                ..control(epi)
            };
            let want = run_gemm(base, &p).unwrap();
            let got = run_gemm(
                Spec {
                    tune: Tune::shipped_gate_up_slab(),
                    ..base
                },
                &p,
            )
            .unwrap();
            assert_eq!(
                bitwise_eq(&got, &want),
                0,
                "{epi:?} {act:?} K={k}: window vs default"
            );
            assert_eq!(
                bitwise_eq(&got, &reference(&p, epi == Epi::Add)),
                0,
                "{epi:?} {act:?} K={k}: window vs the CPU fold"
            );
        }
    }
    assert_eq!(Tune::shipped_gate_up_slab().wait, WaitPlace::Window);
}
