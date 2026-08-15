// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Bit-exact oracle for the gfx90a DeepSeek V4 eager Indexer radix top-512.

use rdna_compute::{DType, Gpu};

const N_STRIDE: usize = 32_768;
const N_ACTIVE: usize = 2_705;
const K: usize = 512;

fn sample(index: usize) -> f32 {
    let mut x = (index as u32).wrapping_add(17).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x as i32) as f32 * (1.0 / 2_147_483_648.0)
}

fn check_case(gpu: &mut Gpu, name: &str, active: &[f32]) {
    assert_eq!(active.len(), N_ACTIVE);
    let mut padded = vec![-1234.5; N_STRIDE];
    padded[..N_ACTIVE].copy_from_slice(active);
    let scores = gpu.upload_f32(&padded, &[N_STRIDE]).unwrap();
    let reference = gpu.zeros(&[K], DType::F32).unwrap();
    let candidate = gpu.zeros(&[K], DType::F32).unwrap();

    gpu.indexer_top_k_batched(
        &scores,
        &reference,
        1,
        N_STRIDE as i32,
        N_ACTIVE as i32,
        K as i32,
        K as i32,
        1,
    )
    .unwrap();
    gpu.indexer_top_k_radix512_gfx90a(&scores, &candidate, N_STRIDE as i32, N_ACTIVE as i32)
        .unwrap();
    gpu.hip.device_synchronize().unwrap();

    let reference = gpu.download_f32(&reference).unwrap();
    let candidate = gpu.download_f32(&candidate).unwrap();
    let mismatches = reference
        .iter()
        .zip(&candidate)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("{name}: topk_bit_mismatch={mismatches}");
    if mismatches != 0 {
        for (rank, (a, b)) in reference.iter().zip(&candidate).enumerate() {
            if a.to_bits() != b.to_bits() {
                eprintln!(
                    "first mismatch rank={rank}: reference={} candidate={}",
                    a.to_bits() as i32,
                    b.to_bits() as i32
                );
                break;
            }
        }
    }
    assert_eq!(mismatches, 0, "{name} must match stable-rank reference");
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this oracle targets MI250 wave64");

    let varied = (0..N_ACTIVE).map(sample).collect::<Vec<_>>();
    check_case(&mut gpu, "varied", &varied);

    let tied = (0..N_ACTIVE)
        .map(|i| match i % 11 {
            0 => 0.0,
            1 => -0.0,
            2 | 3 => 4.0,
            4 | 5 | 6 => -2.0,
            _ => ((i % 37) as f32) - 18.0,
        })
        .collect::<Vec<_>>();
    check_case(&mut gpu, "ties_and_signed_zero", &tied);

    println!("PASS: gfx90a radix top-512 is bit-exact");
}
