// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx90a parity oracle for eager active-prefix Indexer score + top-K.

use rdna_compute::{DType, Gpu};

const H: usize = 64;
const D: usize = 128;
const N_STRIDE: usize = 32_768;
const N_ACTIVE: usize = 2_705;
const K: usize = 512;

fn sample(index: usize, salt: u32, scale: f32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ((x & 0xffff) as f32 / 32_767.5 - 1.0) * scale
}

fn raw_i32(value: i32) -> [u8; 4] {
    value.to_ne_bytes()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this oracle targets MI250 wave64");

    let q = (0..H * D).map(|i| sample(i, 17, 0.125)).collect::<Vec<_>>();
    let k_cache = (0..N_STRIDE * D)
        .map(|i| sample(i, 29, 0.125))
        .collect::<Vec<_>>();
    let weights = (0..H).map(|i| sample(i, 43, 0.25)).collect::<Vec<_>>();

    let d_q = gpu.upload_f32(&q, &[H, D]).unwrap();
    let d_k = gpu.upload_f32(&k_cache, &[N_STRIDE, D]).unwrap();
    let d_weights = gpu.upload_f32(&weights, &[H]).unwrap();
    let d_n = gpu.upload_raw(&raw_i32(N_ACTIVE as i32), &[4]).unwrap();
    let d_k_active = gpu.upload_raw(&raw_i32(K as i32), &[4]).unwrap();
    let scores_buf = gpu.full_f32(&[N_STRIDE], 1234.5).unwrap();
    let scores_active = gpu.full_f32(&[N_STRIDE], 1234.5).unwrap();
    let topk_buf = gpu.zeros(&[K], DType::F32).unwrap();
    let topk_active = gpu.zeros(&[K], DType::F32).unwrap();

    gpu.indexer_relu_score_f32_buf(
        &d_q,
        &d_k,
        &d_weights,
        &scores_buf,
        &d_n,
        N_STRIDE as i32,
        H as i32,
        D as i32,
    )
    .unwrap();
    gpu.indexer_top_k_buf(
        &scores_buf,
        &topk_buf,
        &d_n,
        &d_k_active,
        1,
        N_STRIDE as i32,
        K as i32,
    )
    .unwrap();

    gpu.indexer_relu_score_batched_f32(
        &d_q,
        &d_k,
        &d_weights,
        &d_n,
        &scores_active,
        H as i32,
        D as i32,
        N_STRIDE as i32,
        N_ACTIVE as i32,
        1,
    )
    .unwrap();
    gpu.indexer_top_k_batched(
        &scores_active,
        &topk_active,
        1,
        N_STRIDE as i32,
        N_ACTIVE as i32,
        K as i32,
        K as i32,
        1,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let reference_scores = gpu.download_f32(&scores_buf).unwrap();
    let active_scores = gpu.download_f32(&scores_active).unwrap();
    let score_mismatches = reference_scores[..N_ACTIVE]
        .iter()
        .zip(&active_scores[..N_ACTIVE])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let reference_topk = gpu.download_f32(&topk_buf).unwrap();
    let active_topk = gpu.download_f32(&topk_active).unwrap();
    let topk_mismatches = reference_topk
        .iter()
        .zip(&active_topk)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();

    println!(
        "N_stride={N_STRIDE} N_active={N_ACTIVE} K={K}: score_bit_mismatch={score_mismatches} topk_bit_mismatch={topk_mismatches}"
    );
    assert_eq!(score_mismatches, 0, "active-prefix score must be bit-exact");
    assert_eq!(
        topk_mismatches, 0,
        "parallel stable-rank top-K must be bit-exact"
    );
    println!("PASS: eager active-prefix Indexer selection matches graph-safe reference");
}
