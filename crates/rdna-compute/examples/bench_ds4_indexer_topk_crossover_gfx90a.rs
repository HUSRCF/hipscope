// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! gfx90a DeepSeek V4 Indexer top-512 crossover benchmark.

use rdna_compute::{DType, Gpu};

const N_STRIDE: usize = 32_768;
const K: usize = 512;
const WARMUP: usize = 50;
const ITERS: usize = 500;
const SAMPLES: usize = 5;

fn sample(index: usize) -> f32 {
    let mut x = (index as u32).wrapping_add(17).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x as i32) as f32 * (1.0 / 2_147_483_648.0)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this benchmark targets MI250 wave64");
    eprintln!(
        "=== DS4 Indexer top-512 crossover ===\narch={} warmup={WARMUP} iterations={ITERS}",
        gpu.arch
    );

    let padded = (0..N_STRIDE).map(sample).collect::<Vec<_>>();
    let scores = gpu.upload_f32(&padded, &[N_STRIDE]).unwrap();
    let rank_out = gpu.zeros(&[K], DType::F32).unwrap();
    let radix_out = gpu.zeros(&[K], DType::F32).unwrap();

    for n in [513usize, 640, 768, 896, 1024, 1280, 1536, 2048, 2705] {
        gpu.indexer_top_k_batched(
            &scores,
            &rank_out,
            1,
            N_STRIDE as i32,
            n as i32,
            K as i32,
            K as i32,
            1,
        )
        .unwrap();
        gpu.indexer_top_k_radix512_gfx90a(&scores, &radix_out, N_STRIDE as i32, n as i32)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let rank = gpu.download_f32(&rank_out).unwrap();
        let radix = gpu.download_f32(&radix_out).unwrap();
        let mismatches = rank
            .iter()
            .zip(&radix)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(mismatches, 0, "N={n} parity");

        for _ in 0..WARMUP {
            gpu.indexer_top_k_batched(
                &scores,
                &rank_out,
                1,
                N_STRIDE as i32,
                n as i32,
                K as i32,
                K as i32,
                1,
            )
            .unwrap();
            gpu.indexer_top_k_radix512_gfx90a(&scores, &radix_out, N_STRIDE as i32, n as i32)
                .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let mut rank_samples = Vec::with_capacity(SAMPLES);
        let mut radix_samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.indexer_top_k_batched(
                    &scores,
                    &rank_out,
                    1,
                    N_STRIDE as i32,
                    n as i32,
                    K as i32,
                    K as i32,
                    1,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            rank_samples.push(started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64);

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.indexer_top_k_radix512_gfx90a(&scores, &radix_out, N_STRIDE as i32, n as i32)
                    .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            radix_samples.push(started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64);
        }

        let rank = median(rank_samples);
        let radix = median(radix_samples);
        eprintln!(
            "N={n} parity=PASS rank={rank:.3}us radix={radix:.3}us winner={} ratio={:.3}",
            if rank < radix { "rank" } else { "radix" },
            rank / radix,
        );
    }
}
