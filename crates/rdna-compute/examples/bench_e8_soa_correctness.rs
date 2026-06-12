// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — mfp4-E8-SoA correctness + perf bench on gfx1151.
//
// Correctness gate: SoA kernel output MUST be bit-exact with AoS kernel output.
// Perf: SoA tok/s + GiB/s vs AoS to verify the alignment improvement hypothesis.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const PEAK_GBPS: f64 = 256.0;

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== mfp4-E8-SoA correctness + perf bench ===");
    eprintln!("  arch={arch}  peak_bw_gbps={PEAK_GBPS}");
    eprintln!();

    let shapes: Vec<(usize, usize, &str)> = vec![
        (2048,  2048,  "qkv-q      M=2048  K=2048 "),
        (512,   2048,  "qkv-kv     M=512   K=2048 "),
        (11008, 2048,  "gate_up    M=11008 K=2048 "),
        (2048,  11008, "w_down     M=2048  K=11008"),
        (4096,  2048,  "med        M=4096  K=2048 "),
    ];

    let warmup = 20usize;
    let trials = 200usize;

    // ---- CORRECTNESS GATE ----
    eprintln!("--- Correctness gate: SoA output == AoS output (bit-exact) ---");
    let (m, k) = (512, 2048);
    let aos_data = synth_e8_aos(m, k, 0xDEAD_BEEF);
    let soa_data = aos_to_soa_full(&aos_data, m, k);

    let aos_w = gpu.upload_raw(&aos_data, &[aos_data.len()]).expect("upload AoS");
    let soa_w = gpu.upload_raw(&soa_data, &[soa_data.len()]).expect("upload SoA");
    let x = gpu.alloc_tensor(&[k], DType::F32).expect("alloc x");
    let y_aos = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_aos");
    let y_soa = gpu.alloc_tensor(&[m], DType::F32).expect("alloc y_soa");

    let xh = make_x(k, 0x1234_5678);
    gpu.hip.memcpy_htod(&x.buf, bytes_of(&xh)).unwrap();

    gpu.gemv_mfp4g32_e8(&aos_w, &x, &y_aos, m, k).expect("AoS GEMV");
    gpu.gemv_mfp4g32_e8_soa(&soa_w, &x, &y_soa, m, k).expect("SoA GEMV");
    gpu.hip.device_synchronize().unwrap();

    let mut res_aos = vec![0f32; m];
    let mut res_soa = vec![0f32; m];
    gpu.hip.memcpy_dtoh(bytes_of_mut(&mut res_aos), &y_aos.buf).unwrap();
    gpu.hip.memcpy_dtoh(bytes_of_mut(&mut res_soa), &y_soa.buf).unwrap();

    let mut all_exact = true;
    let mut n_diff = 0usize;
    for i in 0..m {
        if res_aos[i].to_bits() != res_soa[i].to_bits() {
            if n_diff < 3 {
                eprintln!("  MISMATCH at i={}: aos={} soa={}", i, res_aos[i], res_soa[i]);
            }
            n_diff += 1;
            all_exact = false;
        }
    }
    if all_exact {
        eprintln!("  CORRECTNESS PASS: SoA output == AoS output (bit-exact, {} outputs)", m);
    } else {
        eprintln!("  CORRECTNESS FAIL: {} of {} outputs differ!", n_diff, m);
        std::process::exit(1);
    }
    eprintln!();

    // ---- PERF BENCH ----
    eprintln!("--- Perf bench: AoS E8 vs SoA E8 vs MQ4G256-Lloyd ---");
    eprintln!(
        "  {:<42}  {:>26}  {:>26}  {:>12}",
        "shape", "AoS-E8", "SoA-E8", "soa/aos ratio"
    );
    eprintln!(
        "  {:<42}  {:>26}  {:>26}  {:>12}",
        "-".repeat(42), "-".repeat(26), "-".repeat(26), "-".repeat(12)
    );

    for (m, k, label) in &shapes {
        let (m, k) = (*m, *k);

        let aos_data = synth_e8_aos(m, k, 0x1234 ^ m as u64 ^ k as u64);
        let soa_data = aos_to_soa_full(&aos_data, m, k);

        let aos_total = aos_data.len();
        let soa_total = soa_data.len();

        let aos_w = gpu.upload_raw(&aos_data, &[aos_total]).unwrap();
        let soa_w = gpu.upload_raw(&soa_data, &[soa_total]).unwrap();
        let x  = gpu.alloc_tensor(&[k], DType::F32).unwrap();
        let y  = gpu.alloc_tensor(&[m], DType::F32).unwrap();

        let xh = make_x(k, 0xABCD);
        gpu.hip.memcpy_htod(&x.buf, bytes_of(&xh)).unwrap();

        // Warmup AoS
        for _ in 0..warmup {
            gpu.gemv_mfp4g32_e8(&aos_w, &x, &y, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..trials {
            gpu.gemv_mfp4g32_e8(&aos_w, &x, &y, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let aos_us = t0.elapsed().as_secs_f64() * 1e6 / trials as f64;
        let aos_gbps = aos_total as f64 / (aos_us * 1e-6) / 1e9;
        let aos_pct = aos_gbps / PEAK_GBPS * 100.0;

        // Warmup SoA
        for _ in 0..warmup {
            gpu.gemv_mfp4g32_e8_soa(&soa_w, &x, &y, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let t1 = Instant::now();
        for _ in 0..trials {
            gpu.gemv_mfp4g32_e8_soa(&soa_w, &x, &y, m, k).unwrap();
        }
        gpu.hip.device_synchronize().unwrap();
        let soa_us = t1.elapsed().as_secs_f64() * 1e6 / trials as f64;
        let soa_gbps = soa_total as f64 / (soa_us * 1e-6) / 1e9;
        let soa_pct = soa_gbps / PEAK_GBPS * 100.0;

        let ratio = soa_us / aos_us;

        eprintln!(
            "  {:<42}  {:6.2} µs  {:5.1} GB/s ({:4.1}%)  {:6.2} µs  {:5.1} GB/s ({:4.1}%)  {:8.3}x",
            label,
            aos_us, aos_gbps, aos_pct,
            soa_us, soa_gbps, soa_pct,
            ratio
        );
    }

    eprintln!();
    eprintln!("  ratio = soa_time / aos_time  (< 1.0 = SoA faster, > 1.0 = SoA slower)");
}

/// Convert AoS mfp4-E8 buffer to SoA layout.
/// AoS per-row: [16B hdr] + n_blocks * [1B scale + 16B codewords]
/// SoA per-row: [16B hdr (flag changed to 0x06)] + [n_blocks scales, pad16] + [n_blocks*16B codewords]
fn aos_to_soa_row(aos_row: &[u8], n_blocks: usize) -> Vec<u8> {
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_len = 16 + scale_padded + n_blocks * 16;
    let mut out = vec![0u8; soa_len];
    out[..16].copy_from_slice(&aos_row[..16]);
    out[6] = 0x06; // SoA flag
    for b in 0..n_blocks {
        out[16 + b] = aos_row[16 + b * 17];
    }
    let cw_start = 16 + scale_padded;
    for b in 0..n_blocks {
        let src = 16 + b * 17 + 1;
        let dst = cw_start + b * 16;
        out[dst..dst + 16].copy_from_slice(&aos_row[src..src + 16]);
    }
    out
}

fn aos_to_soa_full(aos: &[u8], m: usize, k: usize) -> Vec<u8> {
    let n_blocks = k / 32;
    let aos_row_bytes = 16 + n_blocks * 17;
    let scale_padded = ((n_blocks + 15) >> 4) << 4;
    let soa_row_bytes = 16 + scale_padded + n_blocks * 16;
    let mut out = Vec::with_capacity(m * soa_row_bytes);
    for r in 0..m {
        let row = &aos[r * aos_row_bytes..(r + 1) * aos_row_bytes];
        out.extend_from_slice(&aos_to_soa_row(row, n_blocks));
    }
    out
}

fn synth_e8_aos(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let blocks_per_row = k / 32;
    let row_bytes = 16 + blocks_per_row * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut rng = || -> u32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let roff = row * row_bytes;
        let rs_f16: u16 = 0x2400;
        out[roff..roff + 2].copy_from_slice(&rs_f16.to_le_bytes());
        out[roff + 4..roff + 6].copy_from_slice(&(blocks_per_row as u16).to_le_bytes());
        out[roff + 6] = 0x05;
        for b in 0..blocks_per_row {
            let bp = roff + 16 + b * 17;
            out[bp] = 120u8.wrapping_add((rng() & 0x3F) as u8);
            for w in 0..4 {
                let cw = rng();
                out[bp + 1 + w * 4..bp + 1 + w * 4 + 4].copy_from_slice(&cw.to_le_bytes());
            }
        }
    }
    out
}

fn make_x(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32) * 2.3e-10 - 0.5
    }).collect()
}

fn bytes_of(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytes_of_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, v.len() * 4) }
}
