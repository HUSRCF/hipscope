// SPDX-License-Identifier: MIT OR Apache-2.0
//! Full-shape gfx11 probe for the 256-column/64-row HFQ4 MMQ topology.

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

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn parse_args() -> (usize, usize, usize, usize, bool, bool, bool) {
    let mut m = 17_408;
    let mut k = 5_120;
    let mut n = 2_048;
    let mut pairs = 10;
    let mut residual = false;
    let mut perm_nibble = false;
    let mut base_x256y64 = false;
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
            "--residual" => residual = true,
            "--perm-nibble" => perm_nibble = true,
            "--base-x256y64" => base_x256y64 = true,
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    (m, k, n, pairs, residual, perm_nibble, base_x256y64)
}

fn main() {
    let (m, k, n, pairs, residual, perm_nibble, base_x256y64) = parse_args();
    assert!(!base_x256y64 || perm_nibble);
    assert!(m % 64 == 0 && k % 256 == 0 && n % 256 == 0);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "arch={} mode={} M={m} K={k} N={n} pairs={pairs}",
        gpu.arch,
        if residual {
            "residual"
        } else if perm_nibble {
            "set-perm-nibble"
        } else {
            "set"
        }
    );
    let a_host = build_hfq4g256(m, k);
    let a = gpu.upload_raw(&a_host, &[m, k]).expect("upload A");
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| ((i * 17 + i / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let initial: Vec<f32> = if residual {
        (0..n * m).map(|i| (i % 17) as f32 * 0.001).collect()
    } else {
        vec![0.0; n * m]
    };
    let y_base = gpu.upload_f32(&initial, &[n, m]).expect("upload base");
    let y_wide = gpu.upload_f32(&initial, &[n, m]).expect("upload wide");
    let y_a16 = gpu.upload_f32(&initial, &[n, m]).expect("upload A16");
    let y_a16_wide = gpu.upload_f32(&initial, &[n, m]).expect("upload A16 wide");
    let y_a16_k32 = gpu.upload_f32(&initial, &[n, m]).expect("upload A16 K32");
    let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize X");

    let run = |gpu: &mut Gpu, wide: bool| {
        if residual {
            if wide && perm_nibble {
                gpu.gemm_hfq4g256_residual_mmq_x256y64_perm(&a, &x, &y_wide, m, k, n)
            } else if wide {
                gpu.gemm_hfq4g256_residual_mmq_x256y64(&a, &x, &y_wide, m, k, n)
            } else if base_x256y64 && perm_nibble {
                gpu.gemm_hfq4g256_residual_mmq_x256y64_perm(&a, &x, &y_base, m, k, n)
            } else if base_x256y64 {
                gpu.gemm_hfq4g256_residual_mmq_x256y64(&a, &x, &y_base, m, k, n)
            } else {
                gpu.gemm_hfq4g256_residual_mmq(&a, &x, &y_base, m, k, n)
            }
        } else if wide && perm_nibble {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&a, xq, &y_wide, m, k, n)
        } else if wide {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64(&a, xq, &y_wide, m, k, n)
        } else if base_x256y64 && perm_nibble {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&a, xq, &y_base, m, k, n)
        } else if base_x256y64 {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64(&a, xq, &y_base, m, k, n)
        } else {
            gpu.gemm_hfq4g256_mmq_set_prequant(&a, xq, &y_base, m, k, n)
        }
    };
    let run_a16_k32 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_a16_wmma_128x64_k32_add(&a, &x, &y_a16_k32, m, k, n)
        } else {
            gpu.gemm_hfq4g256_a16_wmma_128x64_k32_set(&a, &x, &y_a16_k32, m, k, n)
        }
    };

    run(&mut gpu, false).expect("baseline correctness");
    run(&mut gpu, true).expect("x256y64 correctness");
    gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_a16, m, k, n)
        .expect("A16 correctness");
    gpu.gemm_hfq4g256_a16_wmma_128x64_set(&a, &x, &y_a16_wide, m, k, n)
        .expect("A16 128x64 correctness");
    run_a16_k32(&mut gpu).expect("A16 128x64 K32 correctness");
    gpu.hip.device_synchronize().expect("sync correctness");
    let base = gpu.download_f32(&y_base).expect("download base");
    let wide = gpu.download_f32(&y_wide).expect("download wide");
    let a16 = gpu.download_f32(&y_a16).expect("download A16");
    let a16_wide = gpu.download_f32(&y_a16_wide).expect("download A16 wide");
    let a16_k32 = gpu.download_f32(&y_a16_k32).expect("download A16 K32");
    let max_abs = base
        .iter()
        .zip(&wide)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("correctness max_abs={max_abs:.8e}");
    let a16_max_abs = base
        .iter()
        .zip(&a16)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let a16_mean_abs = base
        .iter()
        .zip(&a16)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    eprintln!("A16 vs Q8 max_abs={a16_max_abs:.8e} mean_abs={a16_mean_abs:.8e}");
    let a16_wide_max_abs = a16
        .iter()
        .zip(&a16_wide)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("A16 128x64 vs A16 16x16 max_abs={a16_wide_max_abs:.8e}");
    let a16_k32_max_abs = a16
        .iter()
        .zip(&a16_k32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("A16 128x64 K32 vs A16 16x16 max_abs={a16_k32_max_abs:.8e}");

    for _ in 0..3 {
        run(&mut gpu, false).expect("baseline warmup");
        run(&mut gpu, true).expect("x256y64 warmup");
        gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_a16, m, k, n)
            .expect("A16 warmup");
        gpu.gemm_hfq4g256_a16_wmma_128x64_set(&a, &x, &y_a16_wide, m, k, n)
            .expect("A16 128x64 warmup");
        run_a16_k32(&mut gpu).expect("A16 128x64 K32 warmup");
    }
    gpu.hip.device_synchronize().expect("sync warmup");

    let mut base_ms = Vec::with_capacity(pairs);
    let mut wide_ms = Vec::with_capacity(pairs);
    let mut a16_ms = Vec::with_capacity(pairs);
    let mut a16_wide_ms = Vec::with_capacity(pairs);
    let mut a16_k32_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let modes = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for wide in modes {
            let start = Instant::now();
            run(&mut gpu, wide).expect("timed kernel");
            gpu.hip.device_synchronize().expect("sync timed");
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            if wide {
                wide_ms.push(elapsed);
            } else {
                base_ms.push(elapsed);
            }
        }
        let start = Instant::now();
        gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_a16, m, k, n)
            .expect("timed A16 kernel");
        gpu.hip.device_synchronize().expect("sync timed A16");
        a16_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        gpu.gemm_hfq4g256_a16_wmma_128x64_set(&a, &x, &y_a16_wide, m, k, n)
            .expect("timed A16 128x64 kernel");
        gpu.hip.device_synchronize().expect("sync timed A16 128x64");
        a16_wide_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_a16_k32(&mut gpu).expect("timed A16 128x64 K32 kernel");
        gpu.hip.device_synchronize().expect("sync timed A16 128x64 K32");
        a16_k32_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let base_median = median(&mut base_ms);
    let wide_median = median(&mut wide_ms);
    let a16_median = median(&mut a16_ms);
    let a16_wide_median = median(&mut a16_wide_ms);
    let a16_k32_median = median(&mut a16_k32_ms);
    println!(
        "mode={}",
        if residual {
            "residual"
        } else if perm_nibble {
            "set-perm-nibble"
        } else {
            "set"
        }
    );
    println!("m={m} k={k} n={n}");
    println!("baseline_ms={base_median:.4}");
    println!("x256y64_ms={wide_median:.4}");
    println!("speedup={:.4}x", base_median / wide_median);
    println!("a16_wmma_ms={a16_median:.4}");
    println!("q8_over_a16={:.4}x", a16_median / wide_median);
    println!("a16_128x64_ms={a16_wide_median:.4}");
    println!("a16_topology_speedup={:.4}x", a16_median / a16_wide_median);
    println!("q8_over_a16_128x64={:.4}x", a16_wide_median / wide_median);
    println!("a16_128x64_k32_ms={a16_k32_median:.4}");
    println!("a16_k32_speedup={:.4}x", a16_wide_median / a16_k32_median);
    println!("q8_over_a16_k32={:.4}x", a16_k32_median / wide_median);
    println!("max_abs={max_abs:.8e}");
    println!("a16_max_abs={a16_max_abs:.8e}");
    println!("a16_mean_abs={a16_mean_abs:.8e}");
    println!("a16_128x64_max_abs={a16_wide_max_abs:.8e}");
    println!("a16_128x64_k32_max_abs={a16_k32_max_abs:.8e}");
}
