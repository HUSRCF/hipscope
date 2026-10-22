// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every valid `iu4_gemm::Spec` re-emits the bytes recorded before the
//! building blocks (`iu4_gemm::blocks`) were factored out of `emit`: the
//! sha256 of `emit(spec).s_text` per spec, computed from origin/beta's unrefactored
//! `iu4_gemm::emit` (a scratch worktree of origin/beta, this file copied in). A change to any digest is a change
//! to a production kernel and must be intentional.

use peacemaker_isa::kernels::iu4_gemm::{emit, ALayout, Cacc, Epi, Fold, Spec, Tile};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

/// `(variant name, sha256 of s_text)` for every valid spec, in iteration order.
const PINNED: &[(&str, &str)] = &[
    (
        "k128-128x128x8-cacc1-set",
        "079960944bdfc28bb64cf80c89de13438d64f9b51ece918ecbb08d8e86099fc0",
    ),
    (
        "k128-128x128x8-cacc1-slab-set",
        "e769dd8109c5b66783661b507a297add3b20286acd3e119be3201f911c9b183f",
    ),
    (
        "k128-128x128x8-cacc1-add",
        "3ae2eeab50318099623b726019a0d08b7a3b7af623ea89898f32198b9484df92",
    ),
    (
        "k128-128x128x8-cacc1-slab-add",
        "dff9618681be419f4df102f9e75e1ebf19779ff4408449f749520366413fb45e",
    ),
    (
        "k128-128x128x8-cacc1-addsigmoid",
        "4af41d09171edbc9065b613c54a0d0c35a085e19bc0333072f2bb018ed37bfd0",
    ),
    (
        "k128-128x128x8-cacc1-silu",
        "eb12a394f3a62f8454fbd977b7e6b257503f440fce456749ac9aaa933d482e19",
    ),
    (
        "k128-128x128x8-cacc1-slab-silu",
        "3f50cdeff5482a41a923d3150871963c9025c9b6968fb084a381819d64df4924",
    ),
    (
        "k128-128x128x8-cacc1-silu-bf16",
        "0142a4330302a5a9a32d222d259c84836a26e97380099d72514e6d1bfc7aa428",
    ),
    (
        "k128-128x128x8-cacc1-slab-silu-bf16",
        "3e388e47115501d1de4e32108b3b779814a03713729d62a837eddf0880c46066",
    ),
    (
        "k128-128x128x8-cacc1-qkvzagdn",
        "a26cedbe8c421d48fd31cb587a48fe863aae2b9afcc29397961ab4a15523367f",
    ),
    (
        "k128-128x128x8-cacc1-slab-qkvzagdn",
        "a26b3207951c032b948024294970af5900bdff204900d83fa7a28a1eaa66acf0",
    ),
    (
        "k128-256x128x16-cacc1-set",
        "28489b418ce74a6a8378c9ca0d5f09b79d27aed906164c38cc90febe229ddf0b",
    ),
    (
        "k128-256x128x16-cacc1-add",
        "d246a5009d97d897b3e0747d26e6f7947a0edf7a0eaf982be73158b15036d899",
    ),
    (
        "k128-256x128x16-cacc1-silu",
        "0177ce76b08e295200c647b2dc3a56b0e0d9522c3696c417f0e1803337e4a934",
    ),
];

fn digests() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for fold in [Fold::K128, Fold::K256Shared, Fold::K256Pow2] {
        for tile in [Tile::T128x128x8, Tile::T256x128x16] {
            for cacc in [Cacc::One, Cacc::Two] {
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
                            fold,
                            tile,
                            cacc,
                            epi,
                            act,
                            arch: Arch::Gfx1201,
                        };
                        if spec.validate().is_err() {
                            continue;
                        }
                        let e = emit(spec).unwrap_or_else(|e| {
                            panic!("{} is valid but does not emit: {e}", spec.variant_name())
                        });
                        out.push((
                            spec.variant_name(),
                            format!("{:x}", Sha256::digest(e.s_text.as_bytes())),
                        ));
                    }
                }
            }
        }
    }
    out
}

#[test]
fn iu4_gemm_emit_digests_pinned() {
    let got = digests();
    if std::env::var_os("IU4_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, PINNED, "iu4_gemm emitted bytes changed");
}
