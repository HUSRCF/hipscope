// SPDX-License-Identifier: Apache-2.0
//! Full-shape admission benchmark for the gfx1100 X64/Y64 MQ4 probe.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|v| v[0] == name)
        .map(|v| v[1].parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn weights(m: usize, k: usize, seed: usize) -> Vec<u8> {
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

fn activations(n: usize, k: usize, seed: usize) -> Vec<f32> {
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

fn run_shape(
    gpu: &mut Gpu,
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    pairs: usize,
    add: bool,
    seed: usize,
) -> (f64, f64, f32) {
    let a = gpu
        .upload_raw(&weights(m, k, seed), &[m, k])
        .expect("upload weights");
    let x = gpu
        .upload_f32(&activations(n, k, seed + 5), &[n, k])
        .expect("upload activations");
    let xq = gpu
        .ensure_q8_1_mmq_x(&x, n, k)
        .expect("quantize activations");
    let reference = gpu.zeros(&[n, m], DType::F32).expect("reference output");
    let candidate = gpu.zeros(&[n, m], DType::F32).expect("candidate output");

    let reference_run = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
            &a, xq, &reference, m, k, n, add,
        )
        .expect("reference");
    };
    let candidate_run = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x64y64_group128(&a, xq, &candidate, m, k, n, add)
            .expect("candidate");
    };

    reference_run(gpu);
    candidate_run(gpu);
    gpu.hip.device_synchronize().expect("correctness sync");
    let error = max_abs(gpu, &reference, &candidate);
    for _ in 0..3 {
        reference_run(gpu);
        candidate_run(gpu);
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    gpu.dpm_warmup(5.0).expect("DPM warmup");

    let mut reference_ms = Vec::with_capacity(pairs);
    let mut candidate_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for candidate_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if candidate_first {
                candidate_run(gpu);
            } else {
                reference_run(gpu);
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
    let pairs = arg("--pairs", 9);
    assert_eq!(n % 256, 0);
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1100");
    println!("arch={} n={n} pairs={pairs}", gpu.arch);

    let gate = run_shape(&mut gpu, "gate", 17_408, 5_120, n, pairs, false, 11);
    let down = run_shape(&mut gpu, "down", 5_120, 17_408, n, pairs, true, 47);
    let admitted =
        gate.0 / gate.1 >= 1.30 && down.0 / down.1 >= 1.30 && gate.2 <= 1.0e-3 && down.2 <= 1.0e-3;
    println!(
        "x64y64_admission={}",
        if admitted { "PASS" } else { "REJECT" }
    );
}
