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
    // Production shared-expert dimensions. The earlier shapes used tiny
    // gate_m/up_m and N<16; Ornith 1.5's shared expert is 512/512 with K=2048
    // and N=512 (a full prefill chunk), i.e. grid.y=64 batch tiles. A kernel can
    // be correct at N=5 and wrong at N=512.
    run_gate_up_production_benign(&mut gpu);
    run_residual_production(&mut gpu);
    for &(k, n, label) in &[
        (768usize, 5usize, "gpr=3 quads=0 tail=3 (tail only)"),
        (1280usize, 5usize, "gpr=5 quads=1 tail=1 (both)"),
        (2048usize, 9usize, "gpr=8 quads=2 tail=0 (main only, n>BATCH_TILE)"),
    ] {
        println!("\n--- shape: K={k} N={n}  {label} ---");
        run_shape(&mut gpu, k, n);
    }
    for &(k, m, mt) in &[
        (2048usize, 32usize, 16usize),
        (512usize, 64usize, 32usize),
    ] {
        run_moe_shape(&mut gpu, k, m, mt);
    }
    run_moe_pipeline_shape(&mut gpu, 2048, 32);
    run_moe_pipeline_shape(&mut gpu, 512, 64);
    println!("\nPASS — gfx11 qt44 gate_up + MoE grouped GEMMs match the two-grid oracle");
}

/// Parity for the MoE grouped-expert GEMM — the kernel that decodes the routed
/// experts, i.e. ~99% of an A3B MoE's tensors.
///
/// This was NOT covered when the kernel was first written. It compiled, the
/// model produced fluent text, and that was treated as validation. A later KLD
/// measurement put qt44 at 0.993 against qt13's 0.089 with identical precision
/// mixes, which is exactly what a mis-decoding expert path looks like: coherent
/// greedy output, badly wrong distributions.
///
/// Single expert, all slots routed to it, x_row_div = 1 — the simplest
/// configuration that still exercises the real gather + tiling + WMMA path.
fn run_moe_shape(gpu: &mut Gpu, k: usize, m: usize, m_total: usize) {
    println!("\n--- MoE grouped: K={k} M={m} m_total={m_total} ---");
    // Benign-magnitude fixture, deliberately NOT the [-1,1] / [96,160] one used
    // for gate_up.
    //
    // This kernel's WMMA A operand is fp16, so dequantized weights are
    // fp16-rounded. With half-1 weights near 128 (fp16 resolution 0.0625) and
    // +/-1 activations, the 2048-term dot product cancels heavily and amplifies
    // that rounding into ~20% relative error — which looks exactly like a broken
    // kernel but is a property of the fixture. Halves of [1,2] and [4,8] with
    // strictly positive activations keep the two grids clearly distinguishable
    // (a single-header read reconstructs half 1 as ~1.5 instead of ~6) while
    // removing the cancellation.
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32 ^ 0x5151_5151;
            for i in 0..HALF {
                w[base + i] = 1.0 + prng(i, salt);
            }
            for i in HALF..GROUP {
                w[base + i] = 4.0 + prng(i, salt ^ 0x5A5A_5A5A) * 4.0;
            }
        }
    }
    let blob = pack_mq4g256v2(&w, m, k);

    // One expert; every tile maps to expert 0; slot i gathers source row i.
    let n_tiles = m_total / 16;
    let tile_ids: Vec<i32> = vec![0; n_tiles];
    let slot_index: Vec<i32> = (0..m_total as i32).collect();

    // The launcher converts X to fp16 (ensure_fp16_x), so the reference must be
    // computed on the SAME fp16-rounded activations. Otherwise the comparison
    // also measures the input conversion — and with half-1 weights near 128
    // against +/-1 activations the dot product cancels heavily, amplifying that
    // conversion error into a false "kernel is wrong" verdict.
    let mut x = vec![0.0f32; m_total * k];
    for (i, v) in x.iter_mut().enumerate() {
        // Strictly positive: no sign cancellation in the dot product.
        let raw = 0.5 + prng(i, 0x0BAD_F00D);
        *v = f16::from_f32(raw).to_f32();
    }

    let d_blob = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let ptr_val: u64 = d_blob.buf.as_ptr() as u64;
    let d_ptrs = gpu.upload_raw(&ptr_val.to_le_bytes(), &[8]).unwrap();
    let d_tiles = gpu
        .upload_raw(
            &tile_ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            &[n_tiles * 4],
        )
        .unwrap();
    let d_slots = gpu
        .upload_raw(
            &slot_index.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
            &[m_total * 4],
        )
        .unwrap();
    let d_x = gpu.upload_f32(&x, &[m_total * k]).unwrap();
    let d_y = gpu.zeros(&[m_total * m], rdna_compute::DType::F32).unwrap();

    gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
        &d_ptrs, &d_tiles, &d_slots, &d_x, &d_y, m, k, 1, m_total, m_total,
    )
    .expect("moe grouped launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&d_y).unwrap();

    // Oracle: y[slot * m + row] = sum_k W[row][k] * x[slot][k], both headers.
    let want = ref_gemm_f64(&blob, &x, m, k, m_total);
    let bad = ref_gemm_single_header_f64(&blob, &x, m, k, m_total);

    let (ctrl, _) = rel_err(&want.iter().map(|&v| v as f32).collect::<Vec<_>>(), &bad);
    println!("negative-control separation: {ctrl:.4} (want >> 0)");

    let (e, at) = rel_err(&got, &want);
    let (e_bad, _) = rel_err(&got, &bad);
    println!("moe worst rel err: {e:.3e} at {at}");
    println!("moe vs single-header control: {e_bad:.4} (want >> 0)");

    // X is converted to fp16 inside the launcher, so the tolerance is looser
    // here than for the f32-X gate_up kernel. 11x KLD damage would show as an
    // error orders of magnitude above this, not near it.
    const TOL: f64 = 5e-3;
    if e > TOL {
        eprintln!("FAIL: moe rel err {e:.3e} > {TOL:.0e} — expert decode is wrong");
        std::process::exit(1);
    }
    if e_bad < 0.10 {
        eprintln!("FAIL: moe output matches the single-header control — half-select bug");
        std::process::exit(1);
    }
    println!("moe grouped: OK");
}

/// MoE parity in the kernel's REAL pipeline configuration.
///
/// The simple single-expert test above passes, but the pipeline uses:
///   * many experts, selected per 16-slot tile via `expert_tile_ids`
///   * `x_row_div = 8` (K_TOP), so slot -> source row is a DIVISION, not identity
///   * `-1` padding slots, which must contribute zero
/// Any of those can be wrong while the simple case is right, so test them.
fn run_moe_pipeline_shape(gpu: &mut Gpu, k: usize, m: usize) {
    const E: usize = 4;
    const KTOP: i32 = 8;
    let n_tokens = 6usize;
    // 3 tiles of real slots + 1 all-padding tile.
    let m_total = 64usize;
    println!("\n--- MoE pipeline-like: K={k} M={m} experts={E} x_row_div={KTOP} m_total={m_total} ---");

    let mut blobs = Vec::new();
    for e in 0..E {
        let mut w = vec![0.0f32; m * k];
        for r in 0..m {
            for g in 0..(k / GROUP) {
                let base = r * k + g * GROUP;
                let salt = (r * 7919 + g * 104_729 + e * 31_337) as u32;
                for i in 0..HALF {
                    w[base + i] = 1.0 + prng(i, salt);
                }
                for i in HALF..GROUP {
                    w[base + i] = 4.0 + prng(i, salt ^ 0x5A5A_5A5A) * 4.0;
                }
            }
        }
        blobs.push(pack_mq4g256v2(&w, m, k));
    }

    // tile t -> expert t % E; last tile is all padding.
    let n_tiles = m_total / 16;
    let tile_ids: Vec<i32> = (0..n_tiles).map(|t| (t % E) as i32).collect();
    let mut slot_index = vec![-1i32; m_total];
    for slot in 0..(m_total - 16) {
        // flat = token * KTOP + rank; x_row = flat / KTOP must stay < n_tokens
        let flat = (slot as i32) % (n_tokens as i32 * KTOP);
        slot_index[slot] = flat;
    }

    let mut x = vec![0.0f32; n_tokens * k];
    for (i, v) in x.iter_mut().enumerate() {
        let raw = 0.5 + prng(i, 0x51DE_51DE);
        *v = f16::from_f32(raw).to_f32();
    }

    let d_blobs: Vec<_> = blobs.iter().map(|b| gpu.upload_raw(b, &[b.len()]).unwrap()).collect();
    let ptr_bytes: Vec<u8> = d_blobs.iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let d_ptrs = gpu.upload_raw(&ptr_bytes, &[E * 8]).unwrap();
    let d_tiles = gpu.upload_raw(&tile_ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), &[n_tiles * 4]).unwrap();
    let d_slots = gpu.upload_raw(&slot_index.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(), &[m_total * 4]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n_tokens * k]).unwrap();
    let d_y = gpu.zeros(&[m_total * m], rdna_compute::DType::F32).unwrap();

    gpu.gemm_mq4g256v2_moe_grouped_wmma_k2(
        &d_ptrs, &d_tiles, &d_slots, &d_x, &d_y, m, k, KTOP as usize, m_total, n_tokens,
    ).expect("moe pipeline launch");
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&d_y).unwrap();

    // Oracle mirroring the kernel contract exactly.
    let gpr = k / GROUP;
    let mut want = vec![0.0f64; m_total * m];
    for slot in 0..m_total {
        let flat = slot_index[slot];
        if flat < 0 { continue; }
        let x_row = (flat / KTOP) as usize;
        let e = tile_ids[slot / 16] as usize;
        let blob = &blobs[e];
        for r in 0..m {
            let mut acc = 0.0f64;
            for g in 0..gpr {
                let dst = (r * gpr + g) * GROUP_BYTES;
                let mut hdr = [(0.0f32, 0.0f32); 2];
                for h in 0..2 {
                    let sc = u16::from_le_bytes([blob[dst + h * 4], blob[dst + h * 4 + 1]]);
                    let zp = u16::from_le_bytes([blob[dst + h * 4 + 2], blob[dst + h * 4 + 3]]);
                    hdr[h] = (f16::from_bits(sc).to_f32(), f16::from_bits(zp).to_f32());
                }
                for i in 0..GROUP {
                    let byte = blob[dst + 8 + i / 2];
                    let q = if i % 2 == 0 { byte & 0xF } else { byte >> 4 };
                    let (sc, zp) = hdr[i / HALF];
                    acc += (sc * q as f32 + zp) as f64 * x[x_row * k + g * GROUP + i] as f64;
                }
            }
            want[slot * m + r] = acc;
        }
    }

    let (e, at) = rel_err(&got, &want);
    println!("pipeline moe worst rel err: {e:.3e} at {at}");
    // padded slots must be exactly zero
    let pad_bad = (m_total - 16..m_total)
        .flat_map(|slot| (0..m).map(move |r| slot * m + r))
        .filter(|&i| got[i] != 0.0)
        .count();
    println!("padded-slot outputs non-zero: {pad_bad} (want 0)");
    if e > 5e-3 || pad_bad > 0 {
        eprintln!("FAIL: pipeline-config moe decode is wrong (err {e:.3e}, pad_bad {pad_bad})");
        std::process::exit(1);
    }
    println!("moe pipeline-like: OK");
}


/// Production dims with a cancellation-free fixture AND an RMS-normalised error.
///
/// Two traps this avoids, both of which produced false verdicts earlier:
///   * per-element relative error divides by max(|want|,1e-6); with 512x1024
///     outputs and +/-1 activations some land near zero and blow up. At N=5
///     there were only 160 outputs so none did.
///   * fp16/f32 rounding amplified by catastrophic cancellation looks identical
///     to a decode bug.
/// Positive weights and positive activations remove both.
fn run_gate_up_production_benign(gpu: &mut Gpu) {
    let (k, n, gate_m, up_m) = (2048usize, 512usize, 512usize, 512usize);
    println!("\n--- gate_up PRODUCTION (benign fixture): K={k} N={n} gate_m={gate_m} up_m={up_m} ---");
    let mk = |m: usize, salt0: u32| {
        let mut w = vec![0.0f32; m * k];
        for r in 0..m {
            for g in 0..(k / GROUP) {
                let base = r * k + g * GROUP;
                let salt = (r * 7919 + g * 104_729) as u32 ^ salt0;
                for i in 0..HALF { w[base + i] = 1.0 + prng(i, salt); }
                for i in HALF..GROUP { w[base + i] = 4.0 + prng(i, salt ^ 0x5A5A_5A5A) * 4.0; }
            }
        }
        w
    };
    let wg = mk(gate_m, 0x1111_1111);
    let wu = mk(up_m, 0x2222_2222);
    let bg = pack_mq4g256v2(&wg, gate_m, k);
    let bu = pack_mq4g256v2(&wu, up_m, k);
    let mut x = vec![0.0f32; n * k];
    for (i, v) in x.iter_mut().enumerate() { *v = 0.5 + prng(i, 0xDEAD_BEEF); }

    let want_g = ref_gemm_f64(&bg, &x, gate_m, k, n);
    let want_u = ref_gemm_f64(&bu, &x, up_m, k, n);

    let d_ag = gpu.upload_raw(&bg, &[bg.len()]).unwrap();
    let d_au = gpu.upload_raw(&bu, &[bu.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
    let d_yg = gpu.zeros(&[n * gate_m], rdna_compute::DType::F32).unwrap();
    let d_yu = gpu.zeros(&[n * up_m], rdna_compute::DType::F32).unwrap();
    gpu.gemm_gate_up_mq4g256v2_gfx11(&d_ag, &d_au, &d_x, &d_yg, &d_yu, gate_m, up_m, k, n).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let gg = gpu.download_f32(&d_yg).unwrap();
    let gu = gpu.download_f32(&d_yu).unwrap();

    let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
    let worst_norm = |got: &[f32], want: &[f64]| {
        let r = rms(want);
        got.iter().zip(want).map(|(&g, &w)| ((g as f64) - w).abs() / r).fold(0.0f64, f64::max)
    };
    let (eg, _) = rel_err(&gg, &want_g);
    let (eu, _) = rel_err(&gu, &want_u);
    let ng = worst_norm(&gg, &want_g);
    let nu = worst_norm(&gu, &want_u);
    println!("gate per-elem rel {eg:.3e} | RMS-normalised {ng:.3e}");
    println!("up   per-elem rel {eu:.3e} | RMS-normalised {nu:.3e}");
    if ng > 1e-3 || nu > 1e-3 {
        eprintln!("FAIL: gate_up is WRONG at production dims (RMS-norm {ng:.3e}/{nu:.3e})");
        std::process::exit(1);
    }
    println!("gate_up production: OK");
}


/// Residual GEMM at the shared-expert down_proj's production shape.
/// This kernel was generated by script and wired without a parity test.
fn run_residual_production(gpu: &mut Gpu) {
    let (m, k, n) = (2048usize, 512usize, 512usize);
    println!("\n--- residual PRODUCTION (benign): M={m} K={k} N={n} ---");
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        for g in 0..(k / GROUP) {
            let base = r * k + g * GROUP;
            let salt = (r * 7919 + g * 104_729) as u32 ^ 0x7373_7373;
            for i in 0..HALF { w[base + i] = 1.0 + prng(i, salt); }
            for i in HALF..GROUP { w[base + i] = 4.0 + prng(i, salt ^ 0x5A5A_5A5A) * 4.0; }
        }
    }
    let blob = pack_mq4g256v2(&w, m, k);
    let mut x = vec![0.0f32; n * k];
    for (i, v) in x.iter_mut().enumerate() { *v = 0.5 + prng(i, 0xC0FF_EE00); }
    let want = ref_gemm_f64(&blob, &x, m, k, n);

    let d_a = gpu.upload_raw(&blob, &[blob.len()]).unwrap();
    let d_x = gpu.upload_f32(&x, &[n * k]).unwrap();
    let d_y = gpu.zeros(&[n * m], rdna_compute::DType::F32).unwrap();
    gpu.gemm_mq4g256v2_residual_gfx11(&d_a, &d_x, &d_y, m, k, n).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let got = gpu.download_f32(&d_y).unwrap();

    let rms = (want.iter().map(|x| x * x).sum::<f64>() / want.len() as f64).sqrt();
    let worst = got.iter().zip(&want)
        .map(|(&g, &w)| ((g as f64) - w).abs() / rms)
        .fold(0.0f64, f64::max);
    println!("residual RMS-normalised worst err: {worst:.3e}");
    if worst > 1e-3 {
        eprintln!("FAIL: residual kernel is WRONG ({worst:.3e})");
        std::process::exit(1);
    }
    println!("residual production: OK");
}

fn run_shape(gpu: &mut Gpu, k: usize, n: usize) {
    run_shape_dims(gpu, k, n, 32, 16)
}

fn run_shape_dims(gpu: &mut Gpu, k: usize, n: usize, gate_m: usize, up_m: usize) {

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
