//! Host-vs-GPU parity for the gfx11 MQ4-G256-V2 (qt=44) batched GEMM kernels.
//!
//! Sibling of `mq4v2_parity`, which covers the GEMV/decode path. This one covers
//! the batched-prefill GEMM kernels ported to gfx11 (RDNA3/3.5), which
//! previously did not exist — qt44's GEMM family was gated
//! `ArchPredicate::HasWmmaGfx12`, so a qt44 model could not prefill off gfx12.
//!
//! ## Why this construction and not random weights
//!
//! qt44 and qt13 differ ONLY in how 8 header bytes are read. Every cheap check
//! is blind to a mis-decode: identical stride, identical byte count, identical
//! dtype census, and a v1 kernel fed v2 bytes still runs at full speed. #599's
//! own notes record a first v2 artifact scoring WT2 KLD 12.137559 against a
//! 0.043776 baseline — pure noise — while passing all of them.
//!
//! So the weights here are built with DELIBERATELY DISJOINT halves: weights
//! 0..127 in `[-1, 1]`, weights 128..255 in `[+96, +160]`. Against that:
//!
//!   * a kernel that applies `(s0,z0)` to all 256 weights reconstructs half 1
//!     as ~0 instead of ~128;
//!   * a kernel that reads the header as qt13's `[f32 scale][f32 zero]`
//!     reinterprets four fp16 fields as two f32s and produces noise;
//!   * a kernel that swaps halves reconstructs 128 where 0 belongs.
//!
//! An "equal halves" fixture provably cannot detect the first or third, because
//! both headers hold the same value. That is the whole point of the asymmetry.
//!
//! The single-header reference is computed too and asserted to DISAGREE — a
//! passing test whose negative control also passes is measuring nothing.
//!
//! Run: `cargo run --release -p hipfire-runtime --example mq4v2_gemm_gfx11_parity`

use half::f16;
use rdna_compute::Gpu;

const GROUP: usize = 256;
const HALF: usize = 128;
const GROUP_BYTES: usize = 136;

fn prng(i: usize, salt: u32) -> f32 {
    let x = (i as u32)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(salt.wrapping_mul(0x85EB_CA6B));
    let x = x ^ (x >> 15);
    let x = x.wrapping_mul(0x2545_F491);
    let x = x ^ (x >> 13);
    (x >> 8) as f32 / (1u32 << 24) as f32
}

fn build_disjoint_halves(m: usize, k: usize, salt0: u32) -> Vec<f32> {
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32 ^ salt0;
            for i in 0..HALF {
                w[base + i] = prng(i, salt) * 2.0 - 1.0;
            }
            for i in HALF..GROUP {
                w[base + i] = 96.0 + prng(i, salt ^ 0xA5A5_A5A5) * 64.0;
            }
        }
    }
    w
}

fn pack_mq4g256v2(w: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % GROUP, 0);
    let gpr = k / GROUP;
    let mut blob = vec![0u8; m * gpr * GROUP_BYTES];
    for r in 0..m {
        for g in 0..gpr {
            let src = r * k + g * GROUP;
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut codes = [0u8; GROUP];
            for h in 0..2 {
                let off = h * HALF;
                let slice = &w[src + off..src + off + HALF];
                let lo = slice.iter().cloned().fold(f32::INFINITY, f32::min);
                let hi = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let step = if hi > lo { (hi - lo) / 15.0 } else { 0.0 };
                let s_bits = if hi == lo {
                    0u16
                } else {
                    f16::from_f32(step).to_bits()
                };
                let z_bits = f16::from_f32(lo).to_bits();
                blob[dst + h * 4..dst + h * 4 + 2].copy_from_slice(&s_bits.to_le_bytes());
                blob[dst + h * 4 + 2..dst + h * 4 + 4].copy_from_slice(&z_bits.to_le_bytes());
                let s_rt = f16::from_bits(s_bits).to_f32();
                let z_rt = f16::from_bits(z_bits).to_f32();
                if s_rt == 0.0 {
                    continue;
                }
                let inv = 1.0 / s_rt;
                for i in 0..HALF {
                    let q = ((slice[i] - z_rt) * inv + 0.5).floor().clamp(0.0, 15.0);
                    codes[off + i] = q as u8;
                }
            }
            for i in 0..HALF {
                blob[dst + 8 + i] = (codes[2 * i] & 0xF) | ((codes[2 * i + 1] & 0xF) << 4);
            }
        }
    }
    blob
}

/// Oracle: decode per the v2 spec (BOTH headers) and compute y[b] = W · x[b].
fn ref_gemm_f64(blob: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; n * m];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GROUP_BYTES;
            let mut hdr = [(0.0f32, 0.0f32); 2];
            for h in 0..2 {
                let s = u16::from_le_bytes([blob[dst + h * 4], blob[dst + h * 4 + 1]]);
                let z = u16::from_le_bytes([blob[dst + h * 4 + 2], blob[dst + h * 4 + 3]]);
                hdr[h] = (f16::from_bits(s).to_f32(), f16::from_bits(z).to_f32());
            }
            for i in 0..GROUP {
                let byte = blob[dst + 8 + i / 2];
                let q = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let (s, z) = hdr[i / HALF];
                let wv = (s * q as f32 + z) as f64;
                let kk = g * GROUP + i;
                for b in 0..n {
                    y[b * m + r] += wv * x[b * k + kk] as f64;
                }
            }
        }
    }
    y
}

/// NEGATIVE CONTROL: apply header 0 to all 256 weights, i.e. the half-select
/// bug. Must disagree with the oracle, or the fixture is not discriminating.
fn ref_gemm_single_header_f64(blob: &[u8], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let gpr = k / GROUP;
    let mut y = vec![0.0f64; n * m];
    for r in 0..m {
        for g in 0..gpr {
            let dst = (r * gpr + g) * GROUP_BYTES;
            let s = f16::from_bits(u16::from_le_bytes([blob[dst], blob[dst + 1]])).to_f32();
            let z = f16::from_bits(u16::from_le_bytes([blob[dst + 2], blob[dst + 3]])).to_f32();
            for i in 0..GROUP {
                let byte = blob[dst + 8 + i / 2];
                let q = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                let wv = (s * q as f32 + z) as f64;
                let kk = g * GROUP + i;
                for b in 0..n {
                    y[b * m + r] += wv * x[b * k + kk] as f64;
                }
            }
        }
    }
    y
}

fn rel_err(got: &[f32], want: &[f64]) -> (f64, usize) {
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1e-6);
        let e = ((g as f64) - w).abs() / denom;
        if e > worst {
            worst = e;
            at = i;
        }
    }
    (worst, at)
}

fn main() {
    let mut gpu = match Gpu::init_with_device(0) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            std::process::exit(2);
        }
    };
    println!("arch: {}", gpu.arch);

    // The kernel splits each row into `quads = gpr >> 2` four-group iterations
    // plus `tail = gpr & 3` leftovers, which are SEPARATE code paths. A single
    // shape cannot cover both:
    //   gpr = 3  → quads 0, tail 3 — tail only, main loop never runs
    //   gpr = 5  → quads 1, tail 1 — both
    //   gpr = 8  → quads 2, tail 0 — main loop only
    // Run all three. (The first version of this test used only gpr=3 and
    // therefore never executed the main quad loop at all.)
    for &(k, n, label) in &[
        (768usize, 5usize, "gpr=3 quads=0 tail=3 (tail only)"),
        (1280usize, 5usize, "gpr=5 quads=1 tail=1 (both)"),
        (2048usize, 9usize, "gpr=8 quads=2 tail=0 (main only, n>BATCH_TILE)"),
    ] {
        println!("\n--- shape: K={k} N={n}  {label} ---");
        run_shape(&mut gpu, k, n);
    }
    println!("\nPASS — gfx11 qt44 gate_up GEMM matches the two-grid oracle on all shapes");
}

fn run_shape(gpu: &mut Gpu, k: usize, n: usize) {
    let gate_m = 32usize;
    let up_m = 16usize;

    let w_gate = build_disjoint_halves(gate_m, k, 0x1111_1111);
    let w_up = build_disjoint_halves(up_m, k, 0x2222_2222);
    let blob_gate = pack_mq4g256v2(&w_gate, gate_m, k);
    let blob_up = pack_mq4g256v2(&w_up, up_m, k);

    let mut x = vec![0.0f32; n * k];
    for (i, v) in x.iter_mut().enumerate() {
        *v = prng(i, 0xDEAD_BEEF) * 2.0 - 1.0;
    }

    let want_gate = ref_gemm_f64(&blob_gate, &x, gate_m, k, n);
    let want_up = ref_gemm_f64(&blob_up, &x, up_m, k, n);
    let bad_gate = ref_gemm_single_header_f64(&blob_gate, &x, gate_m, k, n);

    // Fixture self-check: the negative control MUST differ from the oracle,
    // otherwise a half-select bug would pass this test silently.
    let (control_sep, _) = rel_err(
        &want_gate.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        &bad_gate,
    );
    println!("negative-control separation: {control_sep:.4} (want >> 0)");
    if control_sep < 0.10 {
        eprintln!(
            "FIXTURE NOT DISCRIMINATING: single-header reference is within {control_sep:.4} \
             of the oracle. A half-select bug would pass. Aborting."
        );
        std::process::exit(1);
    }

    let d_ag = gpu.upload_raw(&blob_gate, &[blob_gate.len()]).unwrap();
    let d_au = gpu.upload_raw(&blob_up, &[blob_up.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
    let d_yg = gpu.zeros(&[n * gate_m], rdna_compute::DType::F32).unwrap();
    let d_yu = gpu.zeros(&[n * up_m], rdna_compute::DType::F32).unwrap();

    gpu.gemm_gate_up_mq4g256v2_gfx11(
        &d_ag, &d_au, &d_x, &d_yg, &d_yu, gate_m, up_m, k, n,
    )
    .expect("gemm_gate_up_mq4g256v2_gfx11 launch");
    gpu.hip.device_synchronize().unwrap();

    let got_gate = gpu.download_f32(&d_yg).unwrap();
    let got_up = gpu.download_f32(&d_yu).unwrap();

    let (eg, ig) = rel_err(&got_gate, &want_gate);
    let (eu, iu) = rel_err(&got_up, &want_up);
    println!("gate worst rel err: {eg:.3e} at {ig}");
    println!("up   worst rel err: {eu:.3e} at {iu}");

    // Also confirm the GPU result is NOT the single-header answer.
    let (e_bad, _) = rel_err(&got_gate, &bad_gate);
    println!("gate vs single-header control: {e_bad:.4} (want >> 0)");

    const TOL: f64 = 2e-3;
    let mut fail = false;
    if eg > TOL {
        eprintln!("FAIL: gate rel err {eg:.3e} > {TOL:.0e}");
        fail = true;
    }
    if eu > TOL {
        eprintln!("FAIL: up rel err {eu:.3e} > {TOL:.0e}");
        fail = true;
    }
    if e_bad < 0.10 {
        eprintln!("FAIL: GPU output matches the single-header control — half-select bug");
        fail = true;
    }
    if fail {
        std::process::exit(1);
    }
}
