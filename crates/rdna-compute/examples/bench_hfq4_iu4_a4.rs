// SPDX-License-Identifier: MIT OR Apache-2.0
//! Standalone packed MQ4 x signed A4 IU4-WMMA correctness/performance gate.

use rdna_compute::Gpu;
use std::time::Instant;

fn build_hfq4g256(m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13) % 97) as f32 * 0.0001;
            let zero = ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[off + 8 + byte] = ((row * 29 + group * 19 + byte * 23) & 0xff) as u8;
            }
        }
    }
    out
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let mut m = 512usize;
    let mut k = 512usize;
    let mut n = 256usize;
    let mut pairs = 15usize;
    let mut a4_group = 128usize;
    let mut exact_q8 = false;
    let mut skip_correctness = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--m" => {
                i += 1;
                m = args[i].parse().expect("valid --m");
            }
            "--k" => {
                i += 1;
                k = args[i].parse().expect("valid --k");
            }
            "--n" => {
                i += 1;
                n = args[i].parse().expect("valid --n");
            }
            "--pairs" => {
                i += 1;
                pairs = args[i].parse().expect("valid --pairs");
            }
            "--a4-group" => {
                i += 1;
                a4_group = args[i].parse().expect("valid --a4-group");
            }
            "--exact-q8" => exact_q8 = true,
            "--skip-correctness" => skip_correctness = true,
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    assert!(m % 64 == 0 && k % 256 == 0 && n % 256 == 0);
    assert!(
        a4_group == 32 || a4_group == 128,
        "--a4-group must be 32 or 128"
    );

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "arch={} M={m} K={k} N={n} pairs={pairs} a4_group={a4_group} exact_q8={exact_q8}",
        gpu.arch
    );
    let weights = build_hfq4g256(m, k);
    let a = gpu.upload_raw(&weights, &[m, k]).expect("upload A");
    let x_host: Vec<f32> = (0..n * k)
        .map(|idx| ((idx * 17 + idx / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let zeros = vec![0.0f32; n * m];
    let y_q8 = gpu.upload_f32(&zeros, &[n, m]).expect("Q8 output");
    let y_q4 = gpu.upload_f32(&zeros, &[n, m]).expect("Q4 output");
    let q8_storage = gpu.hip.malloc((k / 128) * n * 144).expect("Q8 storage");
    let q4_block_bytes = if exact_q8 {
        144
    } else if a4_group == 32 {
        76
    } else {
        72
    };
    let q4_storage = gpu
        .hip
        .malloc((k / 128) * n * q4_block_bytes)
        .expect("Q4 storage");
    let q8 = q8_storage.as_ptr();
    let q4 = q4_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, q8, n, k)
        .expect("quantize Q8");
    if exact_q8 {
        gpu.quantize_q8_1_group128_iu4_planes_into(&x, q4, n, k)
            .expect("quantize exact Q8 IU4 planes");
    } else if a4_group == 32 {
        gpu.quantize_q4_1_group32_into(&x, q4, n, k)
            .expect("quantize Q4 group32");
    } else {
        gpu.quantize_q4_1_group128_into(&x, q4, n, k)
            .expect("quantize Q4 group128");
    }

    let run_q8 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(&a, q8, &y_q8, m, k, n)
    };
    let run_q4 = |gpu: &mut Gpu| {
        if exact_q8 {
            gpu.gemm_hfq4g256_mmq_iu4_q8_exact(&a, q4, &y_q4, m, k, n, false)
        } else if a4_group == 32 {
            gpu.gemm_hfq4g256_mmq_iu4_a4_group32(&a, q4, &y_q4, m, k, n, false)
        } else {
            gpu.gemm_hfq4g256_mmq_iu4_a4(&a, q4, &y_q4, m, k, n, false)
        }
    };
    run_q8(&mut gpu).expect("Q8 correctness");
    run_q4(&mut gpu).expect("Q4 correctness");
    gpu.hip.device_synchronize().expect("correctness sync");

    let (max_abs, mean_abs, ref_rms, relative_l2, cosine) = if skip_correctness {
        (f32::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let ref_host = gpu.download_f32(&y_q8).expect("download Q8");
        let q4_host = gpu.download_f32(&y_q4).expect("download Q4");
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        let mut q4_sq = 0.0f64;
        let mut dot = 0.0f64;
        for (a, b) in ref_host.iter().zip(q4_host.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d as f64;
            diff_sq += (d as f64) * (d as f64);
            ref_sq += (*a as f64) * (*a as f64);
            q4_sq += (*b as f64) * (*b as f64);
            dot += (*a as f64) * (*b as f64);
        }
        let count = ref_host.len() as f64;
        (
            max_abs,
            sum_abs / count,
            (ref_sq / count).sqrt(),
            (diff_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt(),
            dot / (ref_sq * q4_sq).max(f64::MIN_POSITIVE).sqrt(),
        )
    };

    for _ in 0..4 {
        run_q8(&mut gpu).expect("Q8 warmup");
        run_q4(&mut gpu).expect("Q4 warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut q8_ms = Vec::with_capacity(pairs);
    let mut q4_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for q4_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if q4_first {
                run_q4(&mut gpu).expect("timed Q4");
            } else {
                run_q8(&mut gpu).expect("timed Q8");
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let ms = start.elapsed().as_secs_f64() * 1_000.0;
            if q4_first {
                q4_ms.push(ms);
            } else {
                q8_ms.push(ms);
            }
        }
    }
    let q8_med = median(q8_ms);
    let q4_med = median(q4_ms);
    println!("m={m} k={k} n={n} a4_group={a4_group} exact_q8={exact_q8}");
    println!("q8_group128_ms={q8_med:.4}");
    let candidate = if exact_q8 { "iu4_q8_exact" } else { "iu4_a4" };
    println!("{candidate}_ms={q4_med:.4}");
    println!("{candidate}_speedup={:.4}x", q8_med / q4_med);
    println!("max_abs_vs_q8={max_abs:.8e}");
    println!("mean_abs_vs_q8={mean_abs:.8e}");
    println!("q8_ref_rms={ref_rms:.8e}");
    println!("relative_l2_vs_q8={relative_l2:.8e}");
    println!("cosine_vs_q8={cosine:.10}");
}
