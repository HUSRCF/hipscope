// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Every valid `mq6_i8_gemm::Spec` (and the product module) re-emits the bytes recorded
//! before the building blocks (`mq6_i8_gemm::blocks`) were factored out of `emit`: the sha256 of
//! `emit(spec).s_text` per spec and of the `emit_module` text, computed from beta's unrefactored
//! generator. A change to any digest is a change to a production kernel and must be intentional.

use peacemaker_isa::kernels::mq6_i8_gemm::{emit, emit_module, Epi, Raster, Spec, Tile};
use peacemaker_isa::Arch;
use sha2::{Digest, Sha256};

/// `(variant name, sha256 of s_text)` for every valid spec, in iteration order.
const PINNED: &[(&str, &str)] = &[
    (
        "128x128x8-set",
        "cde3c939da7fef1282a34a514ce187a93474e0b5e54b4c9874ca3e9ac178d773",
    ),
    (
        "128x128x8-set-linear",
        "65328b5ba08afd79237fd8577322e1fdd15b04c726523fc264594b8db7e7d87a",
    ),
    (
        "128x128x8-set_bf16",
        "d5278e048899516f522152eba9ec1f6cb349b3cc93e96f4ac2aeaff2a03b6e95",
    ),
    (
        "128x128x8-set_bf16-linear",
        "302b9caafd3351d24b077ef53268f73b676920d6dea8595d4c9844a3c11d6afa",
    ),
];

/// `(variant name, sha256 of s_text)` of the non-shipped tiles.
const PINNED_TILES: &[(&str, &str)] = &[
    (
        "256x128x16-set",
        "1c93780150e77331b2687ebcfe926fc63e165cc9a766b715221faf5d9a60b566",
    ),
    (
        "256x128x16-set-linear",
        "548778c7ff87624854e679ef5dc02b280fefd1e59a58d1d55df4f2c6f644f437",
    ),
    (
        "256x128x16-set_bf16",
        "47a5b9da3fd080aaacd1e77a2bab1b8971a1faf34c4858d3c9ebd2d8857b7380",
    ),
    (
        "256x128x16-set_bf16-linear",
        "24976b03a0cb7543664849043bb533121e582c8cb69608f9a60fe2b3528ab853",
    ),
    (
        "256x128x16-set_nr",
        "75c4f0a9a6efe60d749316dfa0176b65b83a756db769dc0b2bdfc870dfc05a09",
    ),
    (
        "256x128x16-set_nr-linear",
        "d8a6e33915bb02752c7c26d422c14db586e538749ff0d3cefe247985160a1eee",
    ),
    (
        "128x256x16-set",
        "432d787f344855239aec302e06f73d5d30d4a8ce9f9f1a75a53321a44e4c9acf",
    ),
    (
        "128x256x16-set-linear",
        "dd89af59d4cd69a2115f826a1d27b49e084556cbc4020b9cbcd9b1166aca6b1a",
    ),
    (
        "128x256x16-set_bf16",
        "af3239dc13f8fba0fd1052e56a4790afc5dc3013810478cff954f88308220371",
    ),
    (
        "128x256x16-set_bf16-linear",
        "78edab2a02e3fd650866d7310301c9a314d60db3b6736cd9d230767cb7861f31",
    ),
];

/// sha256 of the `emit_module` text (the five banded product symbols: both epilogues on the
/// shipped 128x128 tile, whose entries are `PINNED`, on 128x256, and the QSA norm+RoPE `set_nr`
/// on 256x128, `PINNED_TILES`).
const PINNED_MODULE: &str = "9872922f88767ef7c938aee929ce66d4f2b207e6bba4904e0ffe7397b3d86c07";

fn sha(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn digests(tiles: &[Tile]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for &tile in tiles {
        for epi in [Epi::Set, Epi::SetBf16, Epi::SetNr] {
            for raster in [Raster::Banded, Raster::Linear] {
                let spec = Spec {
                    epi,
                    raster,
                    tile,
                    arch: Arch::Gfx1201,
                    tune: Default::default(),
                };
                if spec.validate().is_err() {
                    continue;
                }
                let e = emit(spec).unwrap_or_else(|e| {
                    panic!("{} is valid but does not emit: {e}", spec.variant_name())
                });
                out.push((spec.variant_name(), sha(&e.s_text)));
            }
        }
    }
    out
}

fn check(tiles: &[Tile], pinned: &[(&str, &str)], what: &str) {
    let got = digests(tiles);
    if std::env::var_os("MQ6_PRINT_DIGESTS").is_some() {
        for (n, d) in &got {
            println!("    (\"{n}\", \"{d}\"),");
        }
    }
    let got: Vec<(&str, &str)> = got.iter().map(|(n, d)| (n.as_str(), d.as_str())).collect();
    assert_eq!(got, pinned, "{what} emitted bytes changed");
}

#[test]
fn mq6_i8_gemm_emit_digests_pinned() {
    check(&[Tile::T128x128], PINNED, "mq6_i8_gemm");
}

/// The tile axis: the 256x128 and 128x256 tiles, pinned once each was bit-equal to the exact
/// reference and to the shipped tile on the interpreter (`tests/mq6_i8_gemm.rs`). These digests
/// guard them; the shipped tile's are `PINNED`.
#[test]
fn mq6_i8_gemm_tile_digests_pinned() {
    check(
        &[Tile::T256x128, Tile::T128x256],
        PINNED_TILES,
        "mq6_i8_gemm tile",
    );
}

#[test]
fn mq6_i8_gemm_module_digest_pinned() {
    let (_, text, proof) = emit_module(Arch::Gfx1201).unwrap();
    if std::env::var_os("MQ6_PRINT_DIGESTS").is_some() {
        println!("module {}", sha(&text));
    }
    assert_eq!(sha(&text), PINNED_MODULE, "mq6_i8_gemm module text changed");
    assert_eq!(proof.s_text_sha256, sha(&text));
}
