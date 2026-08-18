// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare the gfx11 packed-MQ4 gate/up path with a cached FP16 shadow plus rocBLAS.
//!
//! The one-time weight dequantization and activation casts are outside the timed
//! region. Each timed arm executes two projections to match gate + up.

use rdna_compute::{DType, Gpu};
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

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn parse_args() -> (usize, usize, usize, usize) {
    let mut m = 17_408;
    let mut k = 5_120;
    let mut n = 2_048;
    let mut pairs = 10;
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
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    (m, k, n, pairs)
}

fn main() {
    let (m, k, n, pairs) = parse_args();
    assert!(m % 64 == 0 && k % 256 == 0 && n % 256 == 0);

    let mut gpu = Gpu::init().expect("GPU init");
    assert!(gpu.rocblas.is_some(), "set HIPFIRE_ROCBLAS_ALL_ARCHS=1");
    eprintln!("arch={} M={m} K={k} N={n} pairs={pairs}", gpu.arch);

    let packed_host = build_hfq4g256(m, k);
    let packed = gpu.upload_raw(&packed_host, &[m, k]).expect("upload W MQ4");
    let w_fp16 = gpu.alloc_tensor(&[m, k], DType::F16).expect("alloc W FP16");
    gpu.dequantize_hfq4g256_to_f16(&packed.buf, &w_fp16.buf, m, k)
        .expect("dequant W");

    let x_host: Vec<f32> = (0..n * k)
        .map(|i| ((i * 17 + i / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let x_fp16 = gpu.alloc_tensor(&[n, k], DType::F16).expect("alloc X FP16");
    gpu.cast_f32_to_f16(&x, &x_fp16).expect("cast X");
    let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize X");

    let q8_gate = gpu.zeros(&[n, m], DType::F32).expect("alloc q8 gate");
    let q8_up = gpu.zeros(&[n, m], DType::F32).expect("alloc q8 up");
    let fp16_gate = gpu.zeros(&[n, m], DType::F32).expect("alloc fp16 gate");
    let fp16_up = gpu.zeros(&[n, m], DType::F32).expect("alloc fp16 up");

    let run_q8 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&packed, xq, &q8_gate, m, k, n)?;
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&packed, xq, &q8_up, m, k, n)
    };
    let run_fp16 = |gpu: &mut Gpu| {
        gpu.rocblas_gemm_hfq4_prefill(&w_fp16.buf, &x_fp16.buf, &fp16_gate.buf, m, n, k)?;
        gpu.rocblas_gemm_hfq4_prefill(&w_fp16.buf, &x_fp16.buf, &fp16_up.buf, m, n, k)
    };

    run_q8(&mut gpu).expect("q8 correctness");
    run_fp16(&mut gpu).expect("fp16 correctness");
    gpu.hip.device_synchronize().expect("sync correctness");
    let q8 = gpu.download_f32(&q8_gate).expect("download q8");
    let fp16 = gpu.download_f32(&fp16_gate).expect("download fp16");
    let (max_abs, mean_abs) = q8.iter().zip(&fp16).fold((0.0f32, 0.0f64), |acc, (a, b)| {
        let d = (a - b).abs();
        (acc.0.max(d), acc.1 + d as f64)
    });
    let mean_abs = mean_abs / q8.len() as f64;

    for _ in 0..3 {
        run_q8(&mut gpu).expect("q8 warmup");
        run_fp16(&mut gpu).expect("fp16 warmup");
    }
    gpu.hip.device_synchronize().expect("sync warmup");

    let mut q8_ms = Vec::with_capacity(pairs);
    let mut fp16_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let modes = if pair % 2 == 0 { [false, true] } else { [true, false] };
        for fp16_mode in modes {
            let start = Instant::now();
            if fp16_mode {
                run_fp16(&mut gpu).expect("timed fp16");
            } else {
                run_q8(&mut gpu).expect("timed q8");
            }
            gpu.hip.device_synchronize().expect("sync timed");
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            if fp16_mode {
                fp16_ms.push(elapsed);
            } else {
                q8_ms.push(elapsed);
            }
        }
    }

    let q8_median = median(&mut q8_ms);
    let fp16_median = median(&mut fp16_ms);
    println!("m={m} k={k} n={n}");
    println!("q8_mmq_pair_ms={q8_median:.4}");
    println!("fp16_rocblas_pair_ms={fp16_median:.4}");
    println!("rocblas_speedup={:.4}x", q8_median / fp16_median);
    println!("max_abs={max_abs:.8e}");
    println!("mean_abs={mean_abs:.8e}");
}
