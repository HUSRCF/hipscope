// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused gfx90a CPU/scalar/batched parity oracle for DeepSeek V4 mHC.

use half::f16;
use rdna_compute::{DType, Gpu};

const HC_MULT: usize = 4;
const N_CTRL: usize = 24;
const X_DIM: usize = 16_384;
const BATCH: usize = 4;
const HC_EPS: f32 = 1.0e-6;
const POST_SCALE: f32 = 2.0;

fn sample(index: usize, salt: u32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x & 0xffff) as f32 / 32_767.5 - 1.0
}

fn f16_bytes(values: &[f16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn assert_bits(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label} differs at {i}: {x} != {y}"
        );
    }
    println!("{label}: {} values bit-identical", a.len());
}

fn cpu_control(x: &[f32], w: &[f16], base: &[f16]) -> Vec<f32> {
    let sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
    let inv_rms = 1.0 / (sq / X_DIM as f64 + 1.0e-6).sqrt();
    (0..N_CTRL)
        .map(|ctrl| {
            let dot = x
                .iter()
                .enumerate()
                .map(|(d, &v)| (v as f64) * (w[ctrl * X_DIM + d].to_f32() as f64))
                .sum::<f64>();
            (dot * inv_rms) as f32 + base[ctrl].to_f32()
        })
        .collect()
}

fn cpu_sinkhorn(mut m: Vec<f32>, eps: f32, iters: usize) -> Vec<f32> {
    for r in 0..4 {
        let row = &mut m[r * 4..r * 4 + 4];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
        }
        let sum = (row[0] + row[1]) + (row[2] + row[3]);
        for v in row.iter_mut() {
            *v = *v / sum + eps;
        }
    }
    normalize_columns(&mut m, eps);
    for _ in 1..iters {
        for r in 0..4 {
            let row = &mut m[r * 4..r * 4 + 4];
            let sum = (row[0] + row[1]) + (row[2] + row[3]) + eps;
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
        normalize_columns(&mut m, eps);
    }
    m
}

fn normalize_columns(m: &mut [f32], eps: f32) {
    for c in 0..4 {
        let sum = (m[c] + m[4 + c]) + (m[8 + c] + m[12 + c]) + eps;
        for r in 0..4 {
            m[r * 4 + c] /= sum;
        }
    }
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this oracle targets MI250 wave64");

    let x = (0..BATCH * X_DIM)
        .map(|i| sample(i, 17) * 0.75)
        .collect::<Vec<_>>();
    let w = (0..N_CTRL * X_DIM)
        .map(|i| f16::from_f32(sample(i, 29) * 0.02))
        .collect::<Vec<_>>();
    let base = (0..N_CTRL)
        .map(|i| f16::from_f32(sample(i, 43) * 0.1))
        .collect::<Vec<_>>();
    let alpha = [
        f16::from_f32(0.1875),
        f16::from_f32(0.5625),
        f16::from_f32(0.875),
    ];

    let d_x = gpu.upload_f32(&x, &[BATCH, X_DIM]).unwrap();
    let d_x0 = gpu.upload_f32(&x[..X_DIM], &[X_DIM]).unwrap();
    let d_w = gpu
        .upload_raw(&f16_bytes(&w), &[N_CTRL * X_DIM * 2])
        .unwrap();
    let d_base = gpu.upload_raw(&f16_bytes(&base), &[N_CTRL * 2]).unwrap();
    let d_alpha = gpu
        .upload_raw(&f16_bytes(&alpha), &[alpha.len() * 2])
        .unwrap();
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
    assert_bits("hc_compute_control scalar/batched[0]", &c0, &c[..N_CTRL]);
    let mut worst_control = 0.0f32;
    for b in 0..BATCH {
        let cpu = cpu_control(&x[b * X_DIM..(b + 1) * X_DIM], &w, &base);
        worst_control = worst_control.max(max_abs(&c[b * N_CTRL..(b + 1) * N_CTRL], &cpu));
    }
    println!("hc_compute_control CPU worst_max_abs={worst_control:.3e}");
    assert!(worst_control <= 2.0e-3);

    gpu.hc_apply_alpha_batched(&d_c, &d_alpha, &d_base, BATCH as i32)
        .unwrap();
    gpu.hc_apply_alpha(&d_c0, &d_alpha, &d_base).unwrap();
    let c_alpha = gpu.download_f32(&d_c).unwrap();
    let c0_alpha = gpu.download_f32(&d_c0).unwrap();
    assert_bits(
        "hc_apply_alpha scalar/batched[0]",
        &c0_alpha,
        &c_alpha[..N_CTRL],
    );

    let d_pre = gpu.zeros(&[BATCH, HC_MULT], DType::F32).unwrap();
    let d_post = gpu.zeros(&[BATCH, HC_MULT], DType::F32).unwrap();
    let d_comb = gpu.zeros(&[BATCH, 16], DType::F32).unwrap();
    gpu.hc_split_finalize_batched(
        &d_c,
        &d_pre,
        &d_post,
        &d_comb,
        HC_EPS,
        POST_SCALE,
        BATCH as i32,
    )
    .unwrap();
    gpu.hc_sinkhorn_4x4_batched(&d_comb, HC_EPS, 20, BATCH as i32)
        .unwrap();
    gpu.hc_pre_post_sigmoid_scale_f32(&d_c0, HC_EPS, POST_SCALE)
        .unwrap();
    let comb0 = d_c0.sub_offset(8, 16);
    gpu.hc_sinkhorn_4x4(&comb0, HC_EPS, 20).unwrap();

    let pre = gpu.download_f32(&d_pre).unwrap();
    let post = gpu.download_f32(&d_post).unwrap();
    let comb = gpu.download_f32(&d_comb).unwrap();
    let scalar = gpu.download_f32(&d_c0).unwrap();
    let pre_delta = max_abs(&pre[..4], &scalar[..4]);
    let post_delta = max_abs(&post[..4], &scalar[4..8]);
    let comb_delta = max_abs(&comb[..16], &scalar[8..24]);
    println!(
        "mHC finalize scalar/batched[0]: pre={pre_delta:.3e} post={post_delta:.3e} comb={comb_delta:.3e}"
    );
    assert!(pre_delta <= 2.0e-7 && post_delta <= 2.0e-7 && comb_delta == 0.0);

    let mut worst_pre = 0.0f32;
    let mut worst_post = 0.0f32;
    let mut worst_comb = 0.0f32;
    for b in 0..BATCH {
        let row = &c_alpha[b * N_CTRL..(b + 1) * N_CTRL];
        let cpu_pre = row[..4]
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()) + HC_EPS)
            .collect::<Vec<_>>();
        let cpu_post = row[4..8]
            .iter()
            .map(|&v| POST_SCALE / (1.0 + (-v).exp()))
            .collect::<Vec<_>>();
        let cpu_comb = cpu_sinkhorn(row[8..24].to_vec(), HC_EPS, 20);
        worst_pre = worst_pre.max(max_abs(&pre[b * 4..b * 4 + 4], &cpu_pre));
        worst_post = worst_post.max(max_abs(&post[b * 4..b * 4 + 4], &cpu_post));
        worst_comb = worst_comb.max(max_abs(&comb[b * 16..b * 16 + 16], &cpu_comb));
    }
    println!("mHC finalize CPU: pre={worst_pre:.3e} post={worst_post:.3e} comb={worst_comb:.3e}");
    assert!(worst_pre <= 3.0e-7 && worst_post <= 3.0e-7 && worst_comb <= 2.0e-6);

    check_head(&mut gpu, &x[..X_DIM], &w[..HC_MULT * X_DIM], &base[..4]);
    println!("PASS: DeepSeek V4 mHC CPU/scalar/batched parity on gfx90a");
}

fn check_head(gpu: &mut Gpu, x: &[f32], w: &[f16], base: &[f16]) {
    let d_x = gpu.upload_f32(x, &[X_DIM]).unwrap();
    let d_w = gpu
        .upload_raw(&f16_bytes(w), &[HC_MULT * X_DIM * 2])
        .unwrap();
    let d_base = gpu.upload_raw(&f16_bytes(base), &[HC_MULT * 2]).unwrap();
    let d_out = gpu.zeros(&[HC_MULT], DType::F32).unwrap();
    let scale = 0.6875;
    gpu.hc_head_compute_pre(
        &d_x,
        &d_w,
        &d_base,
        &d_out,
        HC_MULT as i32,
        X_DIM as i32,
        scale,
        1.0e-6,
        HC_EPS,
    )
    .unwrap();
    let got = gpu.download_f32(&d_out).unwrap();
    let sq = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
    let inv_rms = 1.0 / (sq / X_DIM as f64 + 1.0e-6).sqrt();
    let expected = (0..HC_MULT)
        .map(|h| {
            let dot = x
                .iter()
                .enumerate()
                .map(|(d, &v)| (v as f64) * (w[h * X_DIM + d].to_f32() as f64))
                .sum::<f64>();
            let score = (dot * inv_rms) as f32 * scale + base[h].to_f32();
            1.0 / (1.0 + (-score).exp()) + HC_EPS
        })
        .collect::<Vec<_>>();
    let err = max_abs(&got, &expected);
    println!("hc_head_compute_pre CPU max_abs={err:.3e}");
    assert!(err <= 3.0e-4);
}
