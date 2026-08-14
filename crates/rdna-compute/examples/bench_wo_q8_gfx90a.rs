// SPDX-License-Identifier: Apache-2.0

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const G: usize = 8;
const M: usize = 1024;
const K: usize = 4096;
const ITERS: usize = 100;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = (((bits >> 23) & 0xff) as i32) - 127 + 15;
    let mant = bits & 0x7fffff;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn quantize_q8() -> Vec<u8> {
    let mut out = Vec::with_capacity(G * M * (K / 32) * 34);
    for g in 0..G {
        for row in 0..M {
            for block in 0..K / 32 {
                out.extend_from_slice(&f32_to_f16_bits(1.0 / 64.0).to_le_bytes());
                for lane in 0..32 {
                    let q = ((g * 17 + row * 13 + block * 7 + lane) % 31) as i8 - 15;
                    out.push(q as u8);
                }
            }
        }
    }
    out
}

fn time_us(gpu: &mut Gpu, mut launch: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..8 {
        launch(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let start = Instant::now();
    for _ in 0..ITERS {
        launch(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    start.elapsed().as_secs_f64() * 1e6 / ITERS as f64
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    assert_eq!(gpu.arch, "gfx90a");
    let w_data = quantize_q8();
    let x_data: Vec<f32> = (0..G * K)
        .map(|i| ((i % 29) as f32 - 14.0) / 32.0)
        .collect();
    let w = gpu.upload_raw(&w_data, &[w_data.len()]).expect("w");
    let x = gpu.upload_f32(&x_data, &[1, G, K]).expect("x");
    let y_ref = gpu.zeros(&[1, G, M], DType::F32).expect("y_ref");
    let y_w64 = gpu.zeros(&[1, G, M], DType::F32).expect("y_w64");

    gpu.wo_per_group_batched_q8_0_1w(&w, &x, &y_ref, G as i32, M as i32, K as i32, 1)
        .expect("generic");
    gpu.wo_per_group_batched_q8_0_wave64_row2_gfx90a(
        &w, &x, &y_w64, G as i32, M as i32, K as i32, 1,
    )
    .expect("wave64");
    gpu.hip.device_synchronize().unwrap();

    let reference = gpu.download_f32(&y_ref).unwrap();
    let candidate = gpu.download_f32(&y_w64).unwrap();
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut bad = 0usize;
    for (&a, &b) in reference.iter().zip(candidate.iter()) {
        let abs = (a - b).abs();
        let rel = abs / a.abs().max(1e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > 2e-3 && rel > 2e-3 {
            bad += 1;
        }
    }

    let generic_us = time_us(&mut gpu, |gpu| {
        gpu.wo_per_group_batched_q8_0_1w(&w, &x, &y_ref, G as i32, M as i32, K as i32, 1)
            .unwrap();
    });
    let wave64_us = time_us(&mut gpu, |gpu| {
        gpu.wo_per_group_batched_q8_0_wave64_row2_gfx90a(
            &w, &x, &y_w64, G as i32, M as i32, K as i32, 1,
        )
        .unwrap();
    });
    println!("shape G={G} M={M} K={K} B=1");
    println!("parity max_abs={max_abs:.6e} max_rel={max_rel:.6e} bad={bad}");
    println!(
        "generic_grouped={generic_us:.3} us wave64_row2={wave64_us:.3} us speedup={:.3}x",
        generic_us / wave64_us
    );
}
