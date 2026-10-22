// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `mq6_i8_gemm` on the peacemaker CPU interpreter: the linear-raster twin of both entries must equal, bit for bit,
//! a Rust transcription of the hipcc kernel's arithmetic (`gemm_mq6g256v2_mmq_i8.gfx12.hip`): per K128 block the
//! exact integer chain `C = sum (q - 32) * qx`, then `acc = fma(RN(s_h * d), fc, acc)`,
//! `acc = fma(RN(z'_h * d), (float)s, acc)` with `z' = fma(32, s_h, z_h)`; BF16 = the incumbent's `isfinite ? RNE :
//! truncate` of `0 + y`. Shapes cover M and N tails (ragged row / token tiles), a single peeled group (K = 256), one
//! loop trip plus the peeled group (K = 512) and an odd trip count; every word outside Y's `M * N` and every
//! out-of-tile word is checked against a sentinel.
use peacemaker_ir::emu::{Emulator, Launch, Memory};
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::mq6_i8_gemm::{self, spec, Epi, Spec};
use peacemaker_isa::Arch;

const KERNARG: u64 = 0x1_0000;
const A_BASE: u64 = 0x100_0000;
const X_BASE: u64 = 0x1000_0000;
const Y_BASE: u64 = 0x2000_0000;
const GUARD: usize = 256;
const SENTINEL: u8 = 0xa5;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = (
        u32::from(h >> 15),
        u32::from((h >> 10) & 31),
        u32::from(h & 0x3ff),
    );
    assert!(e != 0 && e != 31, "test data uses normal halves only");
    f32::from_bits((s << 31) | ((e + 112) << 23) | (m << 13))
}

/// Normal f16 with exponent field `lo..lo+span`.
fn half(rng: &mut Rng, lo: usize, span: usize) -> u16 {
    (((rng.next() & 1) as u16) << 15)
        | (((lo + rng.below(span)) as u16) << 10)
        | (rng.next() as u16 & 0x3ff)
}

/// Activation fill: full-range random, all `-1` (a negative chain sum: the seed must stay centred), or a
/// single `+1` at qs byte `j` of every block (isolates one weight code).
#[derive(Clone, Copy)]
enum Fill {
    Random,
    NegOnes,
    OneHot(usize),
}
struct Case {
    m: usize,
    k: usize,
    n: usize,
    fill: Fill,
}

struct Data {
    /// `[M][K/256][200]`.
    a: Vec<u8>,
    /// `[K/128][N][136]`.
    x: Vec<u8>,
}

fn code(group: &[u8], half: usize, c: usize) -> u32 {
    let bits = 6 * c;
    let payload = &group[8 + 96 * half..8 + 96 * half + 96];
    let (byte, sh) = (bits / 8, bits % 8);
    let v = u32::from(payload[byte]) | u32::from(*payload.get(byte + 1).unwrap_or(&0)) << 8;
    (v >> sh) & 63
}

fn make(rng: &mut Rng, c: &Case) -> Data {
    let g = c.k / 256;
    let mut a = vec![0u8; c.m * g * 200];
    for r in 0..c.m {
        for gi in 0..g {
            let base = (r * g + gi) * 200;
            for h in 0..2 {
                // Scales / zeros over a band that keeps every fold product a normal f32.
                let s = half(rng, 6, 12);
                let z = half(rng, 6, 12);
                a[base + 4 * h..base + 4 * h + 2].copy_from_slice(&s.to_le_bytes());
                a[base + 4 * h + 2..base + 4 * h + 4].copy_from_slice(&z.to_le_bytes());
            }
            for byte in &mut a[base + 8..base + 200] {
                *byte = rng.next() as u8;
            }
        }
    }
    let mut x = vec![0u8; (c.k / 128) * c.n * 136];
    for kb in 0..c.k / 128 {
        for t in 0..c.n {
            let base = (kb * c.n + t) * 136;
            // Mix of full-range and small |q| blocks; the scale `d` over a wide positive band.
            let wide = rng.below(3) != 0;
            let mut sum = 0i32;
            for j in 0..128 {
                let q = if wide {
                    rng.below(255) as i32 - 127
                } else {
                    rng.below(9) as i32 - 4
                };
                let q = match c.fill {
                    Fill::Random => q,
                    Fill::NegOnes => -1,
                    Fill::OneHot(at) => i32::from(j == at),
                };
                sum += q;
                x[base + 8 + j] = q as i8 as u8;
            }
            let d = f32::from_bits(
                (((110 + rng.below(18)) as u32) << 23) | (rng.next() as u32 & 0x7f_ffff),
            );
            x[base..base + 4].copy_from_slice(&d.to_le_bytes());
            x[base + 4..base + 8].copy_from_slice(&sum.to_le_bytes());
        }
    }
    Data { a, x }
}

/// Natural K index (within the K128 block) of qs byte `j` (nibble-pair order).
fn nat(j: usize) -> usize {
    let (g, r) = (j / 8, j % 8);
    8 * g + if r < 4 { 2 * r } else { 2 * (r - 4) + 1 }
}

fn reference(c: &Case, d: &Data, epi: Epi) -> Vec<u8> {
    let g = c.k / 256;
    let mut out = Vec::new();
    for t in 0..c.n {
        for r in 0..c.m {
            let mut acc = 0.0f32;
            for kb in 0..c.k / 128 {
                let (gi, h) = (kb / 2, kb % 2);
                let grp = &d.a[(r * g + gi) * 200..(r * g + gi) * 200 + 200];
                let xb = &d.x[(kb * c.n + t) * 136..(kb * c.n + t) * 136 + 136];
                let (dd, s) = (
                    f32::from_le_bytes(xb[0..4].try_into().unwrap()),
                    i32::from_le_bytes(xb[4..8].try_into().unwrap()),
                );
                let mut cq = 0i32;
                for j in 0..128 {
                    cq += code(grp, h, nat(j)) as i32 * i32::from(xb[8 + j] as i8);
                }
                let fc = (cq - 32 * s) as f32;
                let sh = f16_to_f32(u16::from_le_bytes(
                    grp[4 * h..4 * h + 2].try_into().unwrap(),
                ));
                let zh = f16_to_f32(u16::from_le_bytes(
                    grp[4 * h + 2..4 * h + 4].try_into().unwrap(),
                ));
                let zp = 32.0f32.mul_add(sh, zh);
                acc = (sh * dd).mul_add(fc, acc);
                acc = (zp * dd).mul_add(s as f32, acc);
            }
            match epi {
                Epi::Set => out.extend(acc.to_bits().to_le_bytes()),
                Epi::SetBf16 => {
                    let u = (0.0f32 + acc).to_bits();
                    let b = if (0.0f32 + acc).is_finite() {
                        (u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16
                    } else {
                        u >> 16
                    };
                    out.extend((b as u16).to_le_bytes());
                }
            }
        }
    }
    out
}

fn run(c: &Case, epi: Epi, seed: u64) {
    let rng = &mut Rng(seed);
    let data = make(rng, c);
    let want = reference(c, &data, epi);
    let emitted = mq6_i8_gemm::emit(Spec::emulation(epi)).unwrap();
    let elf = peacemaker_isa::native::assemble(&emitted.s_text, Arch::Gfx1201).unwrap();
    let program = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .unwrap()
    .program;
    let emu = Emulator::new(&program, &Spec::emulation(epi).symbol()).unwrap();

    let mut mem = Memory::default();
    let mut karg = vec![0u8; 296];
    karg[0..8].copy_from_slice(&A_BASE.to_le_bytes());
    karg[8..16].copy_from_slice(&X_BASE.to_le_bytes());
    karg[16..24].copy_from_slice(&(Y_BASE + GUARD as u64).to_le_bytes());
    karg[24..28].copy_from_slice(&(c.m as u32).to_le_bytes());
    karg[28..32].copy_from_slice(&(c.k as u32).to_le_bytes());
    karg[32..36].copy_from_slice(&(c.n as u32).to_le_bytes());
    let (gx, gy) = (c.m.div_ceil(128) as u32, c.n.div_ceil(128) as u32);
    karg[40..44].copy_from_slice(&gx.to_le_bytes());
    karg[44..48].copy_from_slice(&gy.to_le_bytes());
    karg[48..52].copy_from_slice(&1u32.to_le_bytes());
    mem.map("karg".into(), KERNARG, karg).unwrap();
    mem.map("a".into(), A_BASE, data.a.clone()).unwrap();
    mem.map("x".into(), X_BASE, data.x.clone()).unwrap();
    mem.map("y".into(), Y_BASE, vec![SENTINEL; want.len() + 2 * GUARD])
        .unwrap();
    for gy_i in 0..gy {
        for gx_i in 0..gx {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: KERNARG,
                    workgroup: [gx_i, gy_i, 0],
                    block: [spec::THREADS, 1, 1],
                    dynamic_lds: spec::LAUNCH_LDS_BYTES,
                },
                &mut mem,
            )
            .unwrap_or_else(|e| panic!("wg({gx_i},{gy_i}): {e}"));
        }
    }
    let y = &mem.regions.iter().find(|r| r.name == "y").unwrap().bytes;
    assert!(
        y[..GUARD].iter().all(|&b| b == SENTINEL),
        "guard before Y clobbered"
    );
    assert!(
        y[GUARD + want.len()..].iter().all(|&b| b == SENTINEL),
        "guard after Y clobbered"
    );
    let got = &y[GUARD..GUARD + want.len()];
    let w = if epi == Epi::Set { 4 } else { 2 };
    for (i, (g, e)) in got.chunks(w).zip(want.chunks(w)).enumerate() {
        assert_eq!(
            g,
            e,
            "M={} K={} N={} {epi:?}: token {} row {}",
            c.m,
            c.k,
            c.n,
            i / c.m,
            i % c.m
        );
    }
}

#[test]
fn single_group_tails_f32() {
    run(
        &Case {
            m: 48,
            k: 256,
            n: 40,
            fill: Fill::Random,
        },
        Epi::Set,
        1,
    );
}
#[test]
fn single_group_tails_bf16() {
    run(
        &Case {
            m: 52,
            k: 256,
            n: 24,
            fill: Fill::Random,
        },
        Epi::SetBf16,
        2,
    );
}
#[test]
fn loop_and_peel_f32() {
    run(
        &Case {
            m: 132,
            k: 512,
            n: 130,
            fill: Fill::Random,
        },
        Epi::Set,
        3,
    );
}
#[test]
fn loop_and_peel_bf16() {
    run(
        &Case {
            m: 128,
            k: 768,
            n: 128,
            fill: Fill::Random,
        },
        Epi::SetBf16,
        4,
    );
}

#[test]
fn negative_chain_sums_f32() {
    run(
        &Case {
            m: 8,
            k: 256,
            n: 16,
            fill: Fill::NegOnes,
        },
        Epi::Set,
        5,
    );
}
#[test]
fn every_code_position_is_paired_with_its_activation_byte() {
    for j in [0, 1, 3, 4, 7, 8, 31, 32, 63, 64, 95, 127] {
        run(
            &Case {
                m: 8,
                k: 256,
                n: 16,
                fill: Fill::OneHot(j),
            },
            Epi::Set,
            6,
        );
    }
}
#[test]
fn multi_tile_grid_with_three_loop_trips() {
    run(
        &Case {
            m: 260,
            k: 1024,
            n: 200,
            fill: Fill::Random,
        },
        Epi::Set,
        8,
    );
}
#[test]
fn multi_tile_grid_bf16() {
    run(
        &Case {
            m: 136,
            k: 512,
            n: 136,
            fill: Fill::Random,
        },
        Epi::SetBf16,
        9,
    );
}

// ---------------------------------------------------------------------------------------------
// Offline gates: hot-loop shape contract, occupancy ceiling, determinism, assembly of the product module.

fn contract() -> serde_json::Value {
    let path = format!(
        "{}/kernels/mq6_i8_gemm.gfx1201.hotloop.contract.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn hot_loop_meets_shape_contract_for_both_entries() {
    let c = contract();
    for epi in [Epi::Set, Epi::SetBf16] {
        let e = mq6_i8_gemm::emit(Spec::product(epi)).unwrap();
        let census = mq6_i8_gemm::hot_loop_census(&e.s_text);
        for (name, bound) in c["counts"].as_object().unwrap() {
            let actual = census.get(name).copied().unwrap_or(0);
            assert_eq!(
                u64::from(actual),
                bound.as_u64().unwrap(),
                "{epi:?}: {name}"
            );
        }
        for f in c["forbidden"].as_array().unwrap() {
            let f = f.as_str().unwrap();
            assert!(
                !census
                    .keys()
                    .any(|m| m == f || f.strip_suffix('*').is_some_and(|p| m.starts_with(p))),
                "{epi:?}: forbidden {f}"
            );
        }
        assert!(u64::from(e.shape.next_free_vgpr) <= c["vgpr_max"].as_u64().unwrap());
        assert!(u64::from(e.shape.next_free_sgpr) <= c["sgpr_max"].as_u64().unwrap());
        assert!(e.s_text.contains(&format!(
            ".amdhsa_group_segment_fixed_size {}",
            c["group_segment_fixed_bytes"]
        )));
    }
}

#[test]
fn register_plan_keeps_the_occupancy_ceiling_and_lds_fits_two_workgroups() {
    for epi in [Epi::Set, Epi::SetBf16] {
        let e = mq6_i8_gemm::emit(Spec::product(epi)).unwrap();
        assert!(e.shape.next_free_vgpr <= spec::VGPR_CEILING, "{epi:?}");
        assert_eq!(
            e.shape.wmma, 256,
            "{epi:?}: 64 WMMAs x 4 blocks (2 loop + 2 peeled)"
        );
    }
    assert_eq!(spec::LAUNCH_LDS_BYTES, spec::LAYOUT.launch);
    // Two workgroups per WGP (128 KiB of LDS).
    assert!(2 * spec::LAUNCH_LDS_BYTES <= 128 * 1024);
}

#[test]
fn emission_is_deterministic_and_the_module_assembles() {
    let (a, text_a, proof_a) = mq6_i8_gemm::emit_module(Arch::Gfx1201).unwrap();
    let (_, text_b, proof_b) = mq6_i8_gemm::emit_module(Arch::Gfx1201).unwrap();
    assert_eq!(text_a, text_b);
    assert_eq!(proof_a.s_text_sha256, proof_b.s_text_sha256);
    assert_eq!(a.len(), 2);
    let elf = peacemaker_isa::native::assemble(&text_a, Arch::Gfx1201).unwrap();
    for epi in [Epi::Set, Epi::SetBf16] {
        let sym = Spec::product(epi).symbol();
        assert!(
            elf.windows(sym.len()).any(|w| w == sym.as_bytes()),
            "{sym} missing from the code object"
        );
    }
}

#[test]
fn gfx1201_only() {
    for arch in [Arch::Gfx1100, Arch::Gfx1151] {
        assert!(mq6_i8_gemm::emit(Spec {
            epi: Epi::Set,
            raster: mq6_i8_gemm::Raster::Banded,
            arch
        })
        .is_err());
    }
}
