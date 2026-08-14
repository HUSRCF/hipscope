// SPDX-License-Identifier: Apache-2.0

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const ITERS: usize = 200;

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

fn quantize_q8(m: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(m * (k / 32) * 34);
    for row in 0..m {
        for block in 0..k / 32 {
            out.extend_from_slice(&f32_to_f16_bits(1.0 / 64.0).to_le_bytes());
            for lane in 0..32 {
                let q = ((row * 13 + block * 7 + lane) % 31) as i8 - 15;
                out.push(q as u8);
            }
        }
    }
    out
}

fn time_us(gpu: &mut Gpu, mut launch: impl FnMut(&mut Gpu)) -> f64 {
    for _ in 0..12 {
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

fn run_shape(gpu: &mut Gpu, label: &str, m: usize, k: usize) {
    let w_data = quantize_q8(m, k);
    let x_data: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) / 32.0).collect();
    let w = gpu.upload_raw(&w_data, &[w_data.len()]).expect("w");
    let x = gpu.upload_f32(&x_data, &[1, 1, k]).expect("x");
    let y_ref = gpu.zeros(&[1, 1, m], DType::F32).expect("y_ref");
    let y_w64 = gpu.zeros(&[1, 1, m], DType::F32).expect("y_w64");

    gpu.wo_per_group_batched_q8_0_1w(&w, &x, &y_ref, 1, m as i32, k as i32, 1)
        .expect("generic");
    gpu.wo_per_group_batched_q8_0_wave64_row2_gfx90a(&w, &x, &y_w64, 1, m as i32, k as i32, 1)
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

    let generic_us = time_us(gpu, |gpu| {
        gpu.wo_per_group_batched_q8_0_1w(&w, &x, &y_ref, 1, m as i32, k as i32, 1)
            .unwrap();
    });
    let wave64_us = time_us(gpu, |gpu| {
        gpu.wo_per_group_batched_q8_0_wave64_row2_gfx90a(&w, &x, &y_w64, 1, m as i32, k as i32, 1)
            .unwrap();
    });
    println!(
        "{label:24} M={m:5} K={k:5} generic={generic_us:8.3} us wave64={wave64_us:8.3} us speedup={:.3}x parity_abs={max_abs:.3e} rel={max_rel:.3e} bad={bad}",
        generic_us / wave64_us
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu");
    assert_eq!(gpu.arch, "gfx90a");
    for (label, m, k) in [
        ("q_lora_a", 1024, 4096),
        ("q_lora_b", 32768, 1024),
        ("kv_projection", 512, 4096),
        ("compressor_main", 1024, 4096),
        ("compressor_index", 256, 4096),
        ("indexer_q", 8192, 1024),
    ] {
        run_shape(&mut gpu, label, m, k);
    }
}
