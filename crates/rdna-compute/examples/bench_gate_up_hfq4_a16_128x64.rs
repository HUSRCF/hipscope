// SPDX-License-Identifier: MIT OR Apache-2.0
//! Full gate+up comparison for the gfx11 HFQ4 large-M A16 probe.

use rdna_compute::Gpu;
use std::time::Instant;

fn build_hfq4g256(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13 + seed) % 97) as f32 * 0.0001;
            let zero = ((row * 7 + group * 11 + seed) % 31) as f32 * 0.001 - 0.015;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[off + 8 + byte] =
                    ((row * 29 + group * 19 + byte * 23 + seed) & 0xff) as u8;
            }
        }
    }
    out
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    const M: usize = 17_408;
    const K: usize = 5_120;
    const N: usize = 2_048;
    const PAIRS: usize = 10;

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={} gate+up M={M} K={K} N={N}", gpu.arch);
    let gate = gpu
        .upload_raw(&build_hfq4g256(M, K, 3), &[M, K])
        .expect("upload gate");
    let up = gpu
        .upload_raw(&build_hfq4g256(M, K, 71), &[M, K])
        .expect("upload up");
    let x_host: Vec<f32> = (0..N * K)
        .map(|i| ((i * 17 + i / K * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[N, K]).expect("upload X");
    let zeros = vec![0.0f32; N * M];
    let q8_gate = gpu.upload_f32(&zeros, &[N, M]).expect("q8 gate");
    let q8_up = gpu.upload_f32(&zeros, &[N, M]).expect("q8 up");
    let a16_gate = gpu.upload_f32(&zeros, &[N, M]).expect("A16 gate");
    let a16_up = gpu.upload_f32(&zeros, &[N, M]).expect("A16 up");
    let xq = gpu.ensure_q8_1_mmq_x(&x, N, K).expect("quantize X");

    let run_q8 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&gate, xq, &q8_gate, M, K, N)?;
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&up, xq, &q8_up, M, K, N)
    };
    let run_a16 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_a16_wmma_128x64_k32_set(&gate, &x, &a16_gate, M, K, N)?;
        gpu.gemm_hfq4g256_a16_wmma_128x64_k32_set(&up, &x, &a16_up, M, K, N)
    };

    run_q8(&mut gpu).expect("Q8 correctness");
    run_a16(&mut gpu).expect("A16 correctness");
    gpu.hip.device_synchronize().expect("correctness sync");
    let q8 = gpu.download_f32(&q8_gate).expect("download Q8");
    let a16 = gpu.download_f32(&a16_gate).expect("download A16");
    let max_abs = q8
        .iter()
        .zip(&a16)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs = q8
        .iter()
        .zip(&a16)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / q8.len() as f64;

    for _ in 0..3 {
        run_q8(&mut gpu).expect("Q8 warmup");
        run_a16(&mut gpu).expect("A16 warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let mut q8_ms = Vec::with_capacity(PAIRS);
    let mut a16_ms = Vec::with_capacity(PAIRS);
    for pair in 0..PAIRS {
        if pair & 1 == 0 {
            let start = Instant::now();
            run_q8(&mut gpu).expect("Q8 timed");
            gpu.hip.device_synchronize().expect("Q8 sync");
            q8_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            let start = Instant::now();
            run_a16(&mut gpu).expect("A16 timed");
            gpu.hip.device_synchronize().expect("A16 sync");
            a16_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        } else {
            let start = Instant::now();
            run_a16(&mut gpu).expect("A16 timed");
            gpu.hip.device_synchronize().expect("A16 sync");
            a16_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            let start = Instant::now();
            run_q8(&mut gpu).expect("Q8 timed");
            gpu.hip.device_synchronize().expect("Q8 sync");
            q8_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
    }

    let q8_median = median(&mut q8_ms);
    let a16_median = median(&mut a16_ms);
    println!("q8_gate_up_ms={q8_median:.4}");
    println!("a16_gate_up_ms={a16_median:.4}");
    println!("a16_speedup={:.4}x", q8_median / a16_median);
    println!("max_abs={max_abs:.8e}");
    println!("mean_abs={mean_abs:.8e}");
}
