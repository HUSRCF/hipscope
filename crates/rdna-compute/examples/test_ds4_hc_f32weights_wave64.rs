// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx90a CPU/scalar/batched parity oracle for the official DeepSeek V4
//! FP32 mHC control tensors.

use rdna_compute::{DType, Gpu};

const HC_MULT: usize = 4;
const N_CTRL: usize = 24;
const X_DIM: usize = 16_384;
const BATCH: usize = 3;

fn sample(index: usize, salt: u32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x & 0xffff) as f32 / 32_767.5 - 1.0
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn cpu_control(x: &[f32], w: &[f32], base: &[f32]) -> Vec<f32> {
    let sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
    let inv_rms = 1.0 / (sq / X_DIM as f64 + 1.0e-6).sqrt();
    (0..base.len())
        .map(|ctrl| {
            let dot = x
                .iter()
                .enumerate()
                .map(|(d, &v)| (v as f64) * (w[ctrl * X_DIM + d] as f64))
                .sum::<f64>();
            (dot * inv_rms) as f32 + base[ctrl]
        })
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this oracle targets MI250 wave64");

    let x = (0..BATCH * X_DIM)
        .map(|i| sample(i, 17) * 0.75)
        .collect::<Vec<_>>();
    let w = (0..N_CTRL * X_DIM)
        .map(|i| sample(i, 29) * 0.02)
        .collect::<Vec<_>>();
    let base = (0..N_CTRL)
        .map(|i| sample(i, 43) * 0.1)
        .collect::<Vec<_>>();
    let alpha = [0.187_531_25, 0.562_468_77, 0.874_937_5];

    let d_x = gpu.upload_f32(&x, &[BATCH, X_DIM]).unwrap();
    let d_x0 = gpu.upload_f32(&x[..X_DIM], &[X_DIM]).unwrap();
    let d_w = gpu.upload_f32(&w, &[N_CTRL, X_DIM]).unwrap();
    let d_base = gpu.upload_f32(&base, &[N_CTRL]).unwrap();
    let d_alpha = gpu.upload_f32(&alpha, &[3]).unwrap();
    let d_c = gpu.zeros(&[BATCH, N_CTRL], DType::F32).unwrap();
    let d_c0 = gpu.zeros(&[N_CTRL], DType::F32).unwrap();

    gpu.hc_compute_control_batched(
        &d_x,
        &d_w,
        &d_base,
        &d_c,
        N_CTRL as i32,
        X_DIM as i32,
        BATCH as i32,
    )
    .unwrap();
    gpu.hc_compute_control(&d_x0, &d_w, &d_base, &d_c0, N_CTRL as i32, X_DIM as i32)
        .unwrap();

    let c = gpu.download_f32(&d_c).unwrap();
    let c0 = gpu.download_f32(&d_c0).unwrap();
    assert_eq!(
        c0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c[..N_CTRL]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "scalar and batched row 0 differ"
    );
    let mut worst = 0.0f32;
    for b in 0..BATCH {
        let reference = cpu_control(&x[b * X_DIM..(b + 1) * X_DIM], &w, &base);
        worst = worst.max(max_abs(&c[b * N_CTRL..(b + 1) * N_CTRL], &reference));
    }
    println!("F32 hc_compute_control CPU worst_max_abs={worst:.3e}");
    assert!(worst <= 2.0e-3);

    gpu.hc_apply_alpha_batched(&d_c, &d_alpha, &d_base, BATCH as i32)
        .unwrap();
    gpu.hc_apply_alpha(&d_c0, &d_alpha, &d_base).unwrap();
    let c_alpha = gpu.download_f32(&d_c).unwrap();
    let c0_alpha = gpu.download_f32(&d_c0).unwrap();
    assert_eq!(
        c0_alpha.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c_alpha[..N_CTRL]
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "F32 alpha scalar and batched row 0 differ"
    );

    let d_head_w = gpu.upload_f32(&w[..HC_MULT * X_DIM], &[HC_MULT, X_DIM]).unwrap();
    let d_head_base = gpu.upload_f32(&base[..HC_MULT], &[HC_MULT]).unwrap();
    let d_pre = gpu.zeros(&[HC_MULT], DType::F32).unwrap();
    let scale = 0.687_531_23;
    gpu.hc_head_compute_pre(
        &d_x0,
        &d_head_w,
        &d_head_base,
        &d_pre,
        HC_MULT as i32,
        X_DIM as i32,
        scale,
        1.0e-6,
        1.0e-6,
    )
    .unwrap();
    let got = gpu.download_f32(&d_pre).unwrap();
    let raw = cpu_control(&x[..X_DIM], &w[..HC_MULT * X_DIM], &base[..HC_MULT]);
    let expected = raw
        .iter()
        .enumerate()
        .map(|(h, &with_base)| {
            let mix = with_base - base[h];
            let score = mix * scale + base[h];
            1.0 / (1.0 + (-score).exp()) + 1.0e-6
        })
        .collect::<Vec<_>>();
    let head_err = max_abs(&got, &expected);
    println!("F32 hc_head_compute_pre CPU max_abs={head_err:.3e}");
    assert!(head_err <= 3.0e-4);

    println!("PASS: DeepSeek V4 official-F32 mHC parity on gfx90a");
}
