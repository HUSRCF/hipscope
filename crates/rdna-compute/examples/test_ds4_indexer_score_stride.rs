// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx90a oracle for batched indexer score storage-stride vs active-length.

use rdna_compute::Gpu;

const B: usize = 3;
const H: usize = 64;
const D: usize = 128;
const N_STRIDE: usize = 37;
const N_ITER: usize = 13;
const SENTINEL: f32 = 1234.5;

fn sample(index: usize, salt: u32, scale: f32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ((x & 0xffff) as f32 / 32_767.5 - 1.0) * scale
}

fn reference(q: &[f32], k: &[f32], weights: &[f32], b: usize, n: usize) -> f32 {
    let mut values = [0.0f32; H];
    for h in 0..H {
        let mut dot = 0.0f32;
        for d in 0..D {
            dot += q[(b * H + h) * D + d] * k[n * D + d];
        }
        values[h] = dot.max(0.0) * weights[b * H + h];
    }
    for stride in [32usize, 16, 8, 4, 2, 1] {
        for h in 0..stride {
            values[h] += values[h + stride];
        }
    }
    values[0]
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this oracle targets MI250 wave64");

    let q = (0..B * H * D)
        .map(|i| sample(i, 17, 0.125))
        .collect::<Vec<_>>();
    let k = (0..N_STRIDE * D)
        .map(|i| sample(i, 29, 0.125))
        .collect::<Vec<_>>();
    let weights = (0..B * H).map(|i| sample(i, 43, 0.25)).collect::<Vec<_>>();
    let n_per_batch = [5i32, N_ITER as i32, 0i32];
    let n_bytes = unsafe { std::slice::from_raw_parts(n_per_batch.as_ptr().cast::<u8>(), B * 4) };

    let d_q = gpu.upload_f32(&q, &[B, H, D]).unwrap();
    let d_k = gpu.upload_f32(&k, &[N_STRIDE, D]).unwrap();
    let d_weights = gpu.upload_f32(&weights, &[B, H]).unwrap();
    let d_n = gpu.upload_raw(n_bytes, &[B * 4]).unwrap();
    let d_scores = gpu
        .full_f32(&[B, N_STRIDE], SENTINEL)
        .expect("score sentinel");

    gpu.indexer_relu_score_batched_f32(
        &d_q,
        &d_k,
        &d_weights,
        &d_n,
        &d_scores,
        H as i32,
        D as i32,
        N_STRIDE as i32,
        N_ITER as i32,
        B as i32,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    let scores = gpu.download_f32(&d_scores).unwrap();

    let mut max_abs = 0.0f32;
    for b in 0..B {
        for n in 0..N_ITER {
            let got = scores[b * N_STRIDE + n];
            if n >= n_per_batch[b] as usize {
                assert_eq!(got, -1.0e30, "masked score b={b} n={n}");
            } else {
                max_abs = max_abs.max((got - reference(&q, &k, &weights, b, n)).abs());
            }
        }
        for n in N_ITER..N_STRIDE {
            assert_eq!(
                scores[b * N_STRIDE + n],
                SENTINEL,
                "kernel wrote past N_iter at b={b} n={n}"
            );
        }
    }
    println!("active-score CPU max_abs={max_abs:.3e}");
    assert!(max_abs <= 2.0e-5, "active score mismatch");
    println!("PASS: indexer score preserves N_stride tail beyond N_iter");
}
