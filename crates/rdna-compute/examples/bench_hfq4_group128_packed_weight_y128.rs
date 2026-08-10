// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exact-Q8 packed-weight Y128 probe against the retained X256/Y64 path.

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
    let mut m = 17_408usize;
    let mut k = 5_120usize;
    let mut n = 2_048usize;
    let mut pairs = 9usize;
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
    assert!(m % 128 == 0 && k % 256 == 0 && n % 256 == 0);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={} M={m} K={k} N={n} pairs={pairs}", gpu.arch);
    let a_host = build_hfq4g256(m, k);
    let a = gpu.upload_raw(&a_host, &[m, k]).expect("upload A");
    let x_host: Vec<f32> = (0..n * k)
        .map(|idx| ((idx * 17 + idx / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let zeros = vec![0.0f32; n * m];
    let y_ref = gpu.upload_f32(&zeros, &[n, m]).expect("reference output");
    let y_candidate = gpu.upload_f32(&zeros, &[n, m]).expect("candidate output");
    let xq_storage = gpu
        .hip
        .malloc((k / 128) * n * 144)
        .expect("Q8 group128 storage");
    let xq = xq_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, xq, n, k)
        .expect("quantize Q8 group128");

    let run_ref = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(&a, xq, &y_ref, m, k, n)
    };
    let run_candidate = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_packed_weight_y128_group128(&a, xq, &y_candidate, m, k, n, false)
    };

    run_ref(&mut gpu).expect("reference correctness");
    run_candidate(&mut gpu).expect("candidate correctness");
    gpu.hip.device_synchronize().expect("correctness sync");
    let ref_host = gpu.download_f32(&y_ref).expect("download reference");
    let candidate_host = gpu.download_f32(&y_candidate).expect("download candidate");
    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f64;
    for (reference, candidate) in ref_host.iter().zip(candidate_host.iter()) {
        let diff = (reference - candidate).abs();
        max_abs = max_abs.max(diff);
        mean_abs += diff as f64;
    }
    mean_abs /= ref_host.len() as f64;

    for _ in 0..4 {
        run_ref(&mut gpu).expect("reference warmup");
        run_candidate(&mut gpu).expect("candidate warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut ref_ms = Vec::with_capacity(pairs);
    let mut candidate_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for candidate_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if candidate_first {
                run_candidate(&mut gpu).expect("timed candidate");
            } else {
                run_ref(&mut gpu).expect("timed reference");
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            if candidate_first {
                candidate_ms.push(elapsed);
            } else {
                ref_ms.push(elapsed);
            }
        }
    }
    let ref_median = median(ref_ms);
    let candidate_median = median(candidate_ms);
    println!("m={m} k={k} n={n}");
    println!("x256_y64_group128_ms={ref_median:.4}");
    println!("packed_weight_y128_ms={candidate_median:.4}");
    println!("candidate_speedup={:.4}x", ref_median / candidate_median);
    println!("max_abs={max_abs:.8e}");
    println!("mean_abs={mean_abs:.8e}");
}
