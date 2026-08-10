// SPDX-License-Identifier: Apache-2.0
//! Admission benchmark for the exact lane-major packed-LDS MQ4-v2 probe.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|v| v[0] == name)
        .map(|v| v[1].parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn median(v: &[f64]) -> f64 {
    let mut x = v.to_vec();
    x.sort_by(f64::total_cmp);
    x[x.len() / 2]
}

fn source_mq4(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let base = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13 + seed) % 97) as f32 * 0.0001;
            let zero = ((row * 7 + group * 11 + 3 * seed) % 31) as f32 * 0.001 - 0.015;
            out[base..base + 4].copy_from_slice(&scale.to_le_bytes());
            out[base + 4..base + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[base + 8 + byte] =
                    ((row * 29 + group * 19 + byte * 23 + seed * 37) & 0xff) as u8;
            }
        }
    }
    out
}

fn repack_lane_major(source: &[u8], m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0; source.len()];
    for tile in 0..m / 16 {
        for group in 0..groups {
            let dst = (tile * groups + group) * 16 * 136;
            for row in 0..16 {
                let src = ((tile * 16 + row) * groups + group) * 136;
                out[dst + 4 * row..dst + 4 * row + 4].copy_from_slice(&source[src..src + 4]);
                out[dst + 64 + 4 * row..dst + 64 + 4 * row + 4]
                    .copy_from_slice(&source[src + 4..src + 8]);
                for subtile in 0..8 {
                    for word in 0..4 {
                        let src_word = src + 8 + subtile * 16 + word * 4;
                        let dst_word = dst + 128 + subtile * 256 + word * 64 + row * 4;
                        out[dst_word..dst_word + 4]
                            .copy_from_slice(&source[src_word..src_word + 4]);
                    }
                }
            }
        }
    }
    out
}

fn x_values(n: usize, k: usize, seed: usize) -> Vec<f32> {
    (0..n * k)
        .map(|i| ((i * 17 + (i / k) * 31 + seed) % 101) as f32 * 0.01 - 0.5)
        .collect()
}

fn max_abs(gpu: &mut Gpu, a: &GpuTensor, b: &GpuTensor) -> f32 {
    let a = gpu.download_f32(a).expect("download reference");
    let b = gpu.download_f32(b).expect("download candidate");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn bench_shape(
    gpu: &mut Gpu,
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    pairs: usize,
    add: bool,
    seed: usize,
) -> (f64, f64, f32) {
    let source = source_mq4(m, k, seed);
    let execution = repack_lane_major(&source, m, k);
    assert_eq!(source.len(), execution.len());
    let source = gpu
        .upload_raw(&source, &[m, k])
        .expect("upload source weight");
    let execution = gpu
        .upload_raw(&execution, &[execution.len()])
        .expect("upload execution weight");
    let x = gpu
        .upload_f32(&x_values(n, k, seed + 5), &[n, k])
        .expect("upload X");
    let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize X");
    let reference = gpu.zeros(&[n, m], DType::F32).expect("reference Y");
    let candidate = gpu.zeros(&[n, m], DType::F32).expect("candidate Y");

    let run_reference = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
            &source, xq, &reference, m, k, n, add,
        )
        .expect("reference");
    };
    let run_candidate = |gpu: &mut Gpu| {
        gpu.gemm_mq4v2_lane_major_packed_lds_prequant(&execution, xq, &candidate, m, k, n, add)
            .expect("candidate");
    };
    run_reference(gpu);
    run_candidate(gpu);
    gpu.hip.device_synchronize().expect("correctness sync");
    let error = max_abs(gpu, &reference, &candidate);
    for _ in 0..3 {
        run_reference(gpu);
        run_candidate(gpu);
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    gpu.dpm_warmup(5.0).expect("DPM warmup");

    let mut reference_ms = Vec::with_capacity(pairs);
    let mut candidate_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for candidate_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if candidate_first {
                run_candidate(gpu);
            } else {
                run_reference(gpu);
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            if candidate_first {
                candidate_ms.push(elapsed);
            } else {
                reference_ms.push(elapsed);
            }
        }
    }
    let reference = median(&reference_ms);
    let candidate = median(&candidate_ms);
    println!("{label}_reference_ms={reference:.4}");
    println!("{label}_candidate_ms={candidate:.4}");
    println!("{label}_speedup={:.4}x", reference / candidate);
    println!("{label}_max_abs={error:.8e}");
    println!("{label}_reference_raw_ms={reference_ms:?}");
    println!("{label}_candidate_raw_ms={candidate_ms:?}");
    (reference, candidate, error)
}

fn main() {
    let n = arg("--n", 2_048);
    let pairs = arg("--pairs", 7);
    assert_eq!(n % 256, 0);
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1100");
    println!("arch={} n={n} pairs={pairs}", gpu.arch);

    let (gate_ref, gate_candidate, gate_error) =
        bench_shape(&mut gpu, "gate", 17_408, 5_120, n, pairs, false, 11);
    let (down_ref, down_candidate, down_error) =
        bench_shape(&mut gpu, "down", 5_120, 17_408, n, pairs, true, 47);
    let admitted = gate_ref / gate_candidate >= 1.30
        && down_ref / down_candidate >= 1.30
        && gate_error <= 1.0e-3
        && down_error <= 1.0e-3;
    println!(
        "lane_major_admission={}",
        if admitted { "PASS" } else { "REJECT" }
    );
}
