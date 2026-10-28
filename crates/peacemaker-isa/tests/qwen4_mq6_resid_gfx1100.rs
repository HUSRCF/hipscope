// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1100 MQ6 residual twin (`qwen4_mq6_x4_gfx11`, `Kind::Resid`): exact twin of hipcc's
//! `gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds` (`Y[col * M + row] += W @ X`, F32).
//!
//! CPU only. Emission and ABI tests run on gfx1100; the numeric differential runs the same builder
//! output on the PeaceMaker emulator, which has WMMA f16 fragment semantics for gfx1151 (not gfx1100;
//! `emu/wmma.rs` refuses to approximate it): the gfx1100 and gfx1151 instruction streams are asserted
//! identical, so the emulated program is the gfx1100 program. The reference chain is the emulator's
//! own gfx11 WMMA model (`emu::mma`), bit for bit.
use peacemaker_ir::emu::{mma, Emulator, Launch, Memory};
use peacemaker_ir::Arch as IrArch;
use peacemaker_ir::inst::Frontend;
use peacemaker_isa::kernels::qwen4_mq6_x4_gfx11::{self as mq6, Cfg, Fmt, Kind};
use peacemaker_isa::Arch;

const SYMBOL: &str = "gemm_mq6g256v2_residual_wmma_gfx11_mw8_lds";

fn cfg(arch: Arch, wr: u8) -> Cfg {
    Cfg {
        arch,
        wr,
        kind: Kind::Resid,
        fmt: Fmt::Mq6,
    }
}

#[test]
fn residual_entries_emit_on_gfx1100_deterministically() {
    for wr in [4u8, 8] {
        let (a, b) = (
            mq6::emit_cfg(cfg(Arch::Gfx1100, wr)).unwrap(),
            mq6::emit_cfg(cfg(Arch::Gfx1100, wr)).unwrap(),
        );
        assert_eq!(a.s_text, b.s_text, "w{wr}");
        assert_eq!(a.proof.s_text_sha256, b.proof.s_text_sha256, "w{wr}");
        assert_eq!(a.shape.wmma, 256, "w{wr}");
        assert!(a.s_text.contains(&format!(".globl {SYMBOL}\n")), "w{wr}");
        // hipcc's incumbent metadata: A X Y pointers, M K N by value, 36-byte block, 40 KiB static LDS.
        assert!(a.s_text.contains(".kernarg_segment_size: 36\n"), "w{wr}");
        let field = |key: &str| -> Vec<u32> {
            a.s_text
                .lines()
                .filter_map(|l| l.trim().strip_prefix(key))
                .map(|v| v.trim().parse().unwrap())
                .collect()
        };
        assert_eq!(field(".offset:"), [0, 8, 16, 24, 28, 32], "w{wr}");
        assert_eq!(field(".size:"), [8, 8, 8, 4, 4, 4], "w{wr}");
        assert!(a.s_text.contains(".amdhsa_group_segment_fixed_size 40960"), "w{wr}");
        assert!(a.s_text.contains("amdgcn-amd-amdhsa--gfx1100") || a.s_text.contains("gfx1100"));
        // The accumulator is added to the loaded Y as is (hipcc: `v_add_f32_e32 acc, acc, y`); the
        // overwrite epilogue's `0 + acc` canonicalisation is absent.
        assert!(a.s_text.contains("v_add_f32_e32 v"), "w{wr}");
        assert!(!a.s_text.contains("v_add_f32_e32 v1, v1,"), "w{wr}");
        assert!(!a.s_text.lines().any(|l| l.trim().starts_with("v_add_f32_e32") && l.contains(", 0, v")), "w{wr}");
    }
}

#[test]
fn residual_is_gfx1100_and_gfx1151_only_with_two_entries() {
    for arch in [Arch::Gfx1201, Arch::Gfx942] {
        for wr in [4u8, 8] {
            assert!(mq6::emit_cfg(cfg(arch, wr)).is_err(), "{arch:?} w{wr}");
        }
    }
    for wr in [0u8, 2, 6, 16] {
        assert!(mq6::emit_cfg(cfg(Arch::Gfx1100, wr)).is_err(), "w{wr}");
    }
    // MQ4 has no residual twin; the overwrite family stays exact-gfx1151.
    let mq4 = Cfg {
        fmt: Fmt::Mq4,
        ..cfg(Arch::Gfx1100, 8)
    };
    assert!(mq6::emit_cfg(mq4).is_err());
    for (wr, kind) in mq6::Spec::ALL {
        assert!(
            mq6::emit(mq6::Spec {
                arch: Arch::Gfx1100,
                wr,
                kind
            })
            .is_err(),
            "gfx1100 w{wr} {kind:?} stays refused"
        );
    }
}

/// The gfx1100 and gfx1151 residual programs are the same instruction stream: only the target header
/// differs, so the emulator differential below (gfx1151 semantics) covers the gfx1100 object.
#[test]
fn residual_gfx1100_and_gfx1151_instruction_streams_are_identical() {
    for wr in [4u8, 8] {
        let a = mq6::emit_cfg(cfg(Arch::Gfx1100, wr)).unwrap().s_text;
        let b = mq6::emit_cfg(cfg(Arch::Gfx1151, wr)).unwrap().s_text;
        let body = |s: &str| -> Vec<String> {
            s.lines()
                .filter(|l| !l.contains("gfx1100") && !l.contains("gfx1151"))
                .map(str::to_owned)
                .collect()
        };
        assert_eq!(body(&a), body(&b), "w{wr}");
    }
}

// ---------------------------------------------------------------------------------------- emulator

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// f64 -> f16 bits, round to nearest even, finite values up to the f16 range (subnormals included).
fn f16_round(v: f64) -> u16 {
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a == 0.0 {
        return sign;
    }
    let e = (((a.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(-14);
    let ulp = 2f64.powi(e - 10);
    let r = (a / ulp).round_ties_even() * ulp;
    assert!(r < 65520.0, "{v} outside the f16 range");
    if r < 2f64.powi(-14) {
        return sign | (r / 2f64.powi(-24)) as u16;
    }
    let e2 = ((r.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let m = ((r / 2f64.powi(e2) - 1.0) * 1024.0) as u16;
    sign | (((e2 + 15) as u16) << 10) | m
}
fn f16_bits(x: f32) -> u16 {
    f16_round(f64::from(x))
}
fn f16_val(h: u16) -> f32 {
    let (e, m) = (i32::from(h >> 10 & 31), f64::from(h & 0x3ff));
    assert!(e != 31, "{h:#x} is not finite");
    let mag = if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    };
    (if h >> 15 != 0 { -mag } else { mag }) as f32
}

struct Problem {
    m: usize,
    n: usize,
    k: usize,
    /// `[row][group] 200 B`: `sc0 zp0 sc1 zp1` f16, then 256 six-bit codes LSB first.
    w: Vec<u8>,
    codes: Vec<u8>,
    hdr: Vec<[u16; 4]>,
    /// `[token][K]` f16 bits.
    x: Vec<u16>,
    y0: Vec<f32>,
}

impl Problem {
    /// `exact`: scales in {0.5, 1, 2}, integer zero points and X in -3..=3, integer Y0: every partial
    /// sum is exact in f32, so the result does not depend on the accumulation order. Otherwise random
    /// f16 scales / zero points / X and a tolerance comparison.
    fn new(m: usize, n: usize, k: usize, seed: u64, exact: bool) -> Self {
        let mut r = Rng(seed);
        let gpr = k / 256;
        let mut w = vec![0u8; m * gpr * 200];
        let mut codes = vec![0u8; m * k];
        let mut hdr = Vec::new();
        for row in 0..m {
            for g in 0..gpr {
                let h: [u16; 4] = std::array::from_fn(|i| {
                    let v = if exact {
                        if i % 2 == 0 {
                            [0.5f32, 1.0, 2.0][(r.next() % 3) as usize]
                        } else {
                            (r.next() % 17) as f32 - 8.0
                        }
                    } else if i % 2 == 0 {
                        0.01 + (r.next() % 64) as f32 * 0.001
                    } else {
                        ((r.next() % 2001) as f32 - 1000.0) * 0.001
                    };
                    f16_bits(v)
                });
                let blk = &mut w[(row * gpr + g) * 200..][..200];
                for (i, v) in h.iter().enumerate() {
                    blk[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
                }
                for c in 0..256 {
                    let q = (r.next() % 64) as u8;
                    codes[row * k + g * 256 + c] = q;
                    for bit in 0..6 {
                        if q >> bit & 1 == 1 {
                            let p = c * 6 + bit;
                            blk[8 + p / 8] |= 1 << (p % 8);
                        }
                    }
                }
                hdr.push(h);
            }
        }
        let x = (0..n * k)
            .map(|_| {
                f16_bits(if exact {
                    ((r.next() % 7) as f32 - 3.0) + 0.0
                } else {
                    ((r.next() % 2001) as f32 - 1000.0) * 0.001
                })
            })
            .collect();
        let y0 = (0..n * m)
            .map(|_| {
                if exact {
                    (r.next() % 2001) as f32 - 1000.0
                } else {
                    ((r.next() % 2001) as f32 - 1000.0) * 0.01
                }
            })
            .collect();
        Self {
            m,
            n,
            k,
            w,
            codes,
            hdr,
            x,
            y0,
        }
    }

    /// Expected `Y`: per (row, col) the emulator's gfx11 WMMA chain (`peacemaker_ir::emu::mma`, ascending
    /// 16-K steps from +0.0, the model of the hardware's non-exact accumulation) over A = f16(sc * q + zp)
    /// (one rounding, the K half's header) and X, then `acc + Y0` in f32.
    fn reference(&self) -> Vec<f32> {
        let (m, n, k) = (self.m, self.n, self.k);
        let gpr = k / 256;
        let mut a = vec![0u16; m * k];
        for row in 0..m {
            for g in 0..gpr {
                let h = self.hdr[row * gpr + g];
                for c in 0..256 {
                    let (sc, zp) = if c < 128 { (h[0], h[1]) } else { (h[2], h[3]) };
                    let q = f64::from(self.codes[row * k + g * 256 + c]);
                    let v = f64::from(f16_val(sc)) * q + f64::from(f16_val(zp));
                    a[row * k + g * 256 + c] = f16_round(v);
                }
            }
        }
        let mut y = vec![0f32; n * m];
        for col in 0..n {
            for row in 0..m {
                let mut c = 0f32.to_bits();
                for kt in 0..k / 16 {
                    c = mma::wmma_f32_f16(
                        IrArch::Gfx1151,
                        &a[row * k + 16 * kt..][..16],
                        &self.x[col * k + 16 * kt..][..16],
                        c,
                    );
                }
                y[col * m + row] = f32::from_bits(c) + self.y0[col * m + row];
            }
        }
        y
    }
}

/// Runs the whole launch grid of the gfx1151-labelled residual program; returns `Y`.
fn run(p: &Problem, wr: u8) -> Result<Vec<f32>, String> {
    let e = mq6::emit_cfg(cfg(Arch::Gfx1151, wr))?;
    let elf = peacemaker_isa::native::assemble(&e.s_text, Arch::Gfx1151)?;
    let lifted = peacemaker_lift::lift_object(
        &elf,
        peacemaker_lift::Options {
            frontend: Frontend::Builder,
        },
    )
    .map_err(|e| format!("lift: {e:?}"))?;
    let emu = Emulator::new(&lifted.program, SYMBOL)?;
    let (kva, ava, xva, yva) = (0x1000_0000u64, 0x2000_0000u64, 0x3000_0000u64, 0x4000_0000u64);
    let mut kern = Vec::new();
    for ptr in [ava, xva, yva] {
        kern.extend_from_slice(&ptr.to_le_bytes());
    }
    for v in [p.m, p.k, p.n] {
        kern.extend_from_slice(&(v as i32).to_le_bytes());
    }
    kern.resize(36, 0);
    let xb: Vec<u8> = p.x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let yb: Vec<u8> = p.y0.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut mem = Memory::default();
    mem.map("kernarg".into(), kva, kern)?;
    for (name, va, bytes) in [("a", ava, p.w.clone()), ("x", xva, xb), ("y", yva, yb)] {
        let len = bytes.len() as u64;
        mem.map(name.into(), va, bytes)?;
        mem.map(format!("{name}_tail"), va + len, vec![0; 4096])?;
    }
    let rows = 16 * usize::from(wr);
    for y in 0..p.n.div_ceil(256) {
        for x in 0..p.m.div_ceil(rows) {
            emu.run_workgroup(
                &Launch {
                    kernarg_ptr: kva,
                    workgroup: [x as u32, y as u32, 0],
                    block: [32 * u32::from(wr), 1, 1],
                    dynamic_lds: 0,
                },
                &mut mem,
            )
            .map_err(|e| format!("workgroup ({x},{y}): {e}"))?;
        }
    }
    for name in ["a_tail", "x_tail", "y_tail"] {
        let r = mem.regions.iter().find(|r| r.name == name).unwrap();
        if r.bytes.iter().any(|&b| b != 0) {
            return Err(format!("kernel wrote past the end of the {name} buffer"));
        }
    }
    Ok(mem
        .read(yva, p.n * p.m * 4)?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn check(p: &Problem, wr: u8) {
    let (m, n, k) = (p.m, p.n, p.k);
    let got = run(p, wr).unwrap_or_else(|e| panic!("M{m} N{n} K{k} w{wr}: {e}"));
    let want = p.reference();
    let bad: Vec<_> = got
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (g, w))| g.to_bits() != w.to_bits())
        .map(|(i, (g, w))| (i / m, i % m, *g, *w))
        .collect();
    assert!(
        bad.is_empty(),
        "M{m} N{n} K{k} w{wr}: {} mismatches, first (col,row,got,want) {:?}",
        bad.len(),
        &bad[..bad.len().min(4)]
    );
}

fn check_exact(m: usize, n: usize, k: usize, wr: u8, seed: u64) {
    check(&Problem::new(m, n, k, seed, true), wr);
}

/// Small-integer values: layout, group walk, residual add, row / token tails, bit for bit.
#[test]
fn residual_matches_cpu_bitwise_on_aligned_and_ragged_shapes() {
    // M % 8 == 0 fast path: partial row tile (40 of 64 / 128), partial token tile.
    check_exact(40, 40, 512, 4, 1);
    check_exact(40, 40, 512, 8, 2);
    // Two row blocks at wr=4, a token tile boundary crossed (N > 256 needs a second grid y).
    check_exact(80, 300, 256, 4, 3);
    // Ragged M (M % 8 != 0): per-element path; ragged N.
    check_exact(21, 7, 512, 8, 4);
    check_exact(21, 7, 256, 4, 5);
}

/// Random f16 scales / zero points / X, bitwise (same chain model, A rounding exercised).
#[test]
fn residual_matches_cpu_bitwise_on_random_values() {
    for (m, n, k, wr, seed) in [(48, 33, 512, 8, 11u64), (48, 33, 512, 4, 12), (29, 19, 256, 8, 13)] {
        let p = Problem::new(m, n, k, seed, false);
        check(&p, wr);
    }
}
