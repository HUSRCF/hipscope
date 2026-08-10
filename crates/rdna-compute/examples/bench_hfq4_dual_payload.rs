// SPDX-License-Identifier: MIT OR Apache-2.0
//! Focused standalone A/B for the gfx11 gate/up payload-only dual kernel.

use rdna_compute::Gpu;
use std::time::Instant;

fn build_hfq4g256(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13 + seed) % 97) as f32 * 0.0001;
            let zero = ((row * 7 + group * 11 + seed * 3) % 31) as f32 * 0.001 - 0.015;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[off + 8 + byte] =
                    ((row * 29 + group * 19 + byte * 23 + seed * 37) & 0xff) as u8;
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
    let mut pairs = 15usize;
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
    assert!(m % 64 == 0 && k % 256 == 0 && n % 256 == 0);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={} M={m} K={k} N={n} pairs={pairs}", gpu.arch);
    let gate = gpu
        .upload_raw(&build_hfq4g256(m, k, 0), &[m, k])
        .expect("upload gate");
    let up = gpu
        .upload_raw(&build_hfq4g256(m, k, 1), &[m, k])
        .expect("upload up");
    let x_host: Vec<f32> = (0..n * k)
        .map(|idx| ((idx * 17 + idx / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let xq_storage = gpu.hip.malloc((k / 128) * n * 144).expect("Xq");
    let xq = xq_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, xq, n, k)
        .expect("quantize group128");

    let zeros = vec![0.0f32; n * m];
    let ref_gate = gpu.upload_f32(&zeros, &[n, m]).expect("ref gate");
    let ref_up = gpu.upload_f32(&zeros, &[n, m]).expect("ref up");
    let dual_gate = gpu.upload_f32(&zeros, &[n, m]).expect("dual gate");
    let dual_up = gpu.upload_f32(&zeros, &[n, m]).expect("dual up");

    let run_ref = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &gate, xq, &ref_gate, m, k, n,
        )?;
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &up, xq, &ref_up, m, k, n,
        )
    };
    let run_dual = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfq4g256_mmq_dual_payload_only(
            &gate, &up, xq, &dual_gate, &dual_up, m, k, n,
        )
    };

    run_ref(&mut gpu).expect("reference correctness");
    run_dual(&mut gpu).expect("dual correctness");
    gpu.hip.device_synchronize().expect("correctness sync");
    let ref_gate_host = gpu.download_f32(&ref_gate).expect("download ref gate");
    let ref_up_host = gpu.download_f32(&ref_up).expect("download ref up");
    let dual_gate_host = gpu.download_f32(&dual_gate).expect("download dual gate");
    let dual_up_host = gpu.download_f32(&dual_up).expect("download dual up");
    let compare = |reference: &[f32], candidate: &[f32]| {
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        for (a, b) in reference.iter().zip(candidate) {
            let delta = (a - b).abs();
            max_abs = max_abs.max(delta);
            sum_abs += delta as f64;
        }
        (max_abs, sum_abs / reference.len() as f64)
    };
    let (gate_max_abs, gate_mean_abs) = compare(&ref_gate_host, &dual_gate_host);
    let (up_max_abs, up_mean_abs) = compare(&ref_up_host, &dual_up_host);

    for _ in 0..3 {
        run_ref(&mut gpu).expect("reference warmup");
        run_dual(&mut gpu).expect("dual warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut ref_ms = Vec::with_capacity(pairs);
    let mut dual_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for dual_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if dual_first {
                run_dual(&mut gpu).expect("timed dual");
            } else {
                run_ref(&mut gpu).expect("timed reference");
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            if dual_first {
                dual_ms.push(elapsed);
            } else {
                ref_ms.push(elapsed);
            }
        }
    }

    let ref_median = median(ref_ms);
    let dual_median = median(dual_ms);
    println!("m={m} k={k} n={n}");
    println!("reference_two_launch_ms={ref_median:.4}");
    println!("dual_payload_ms={dual_median:.4}");
    println!("dual_payload_speedup={:.4}x", ref_median / dual_median);
    println!("gate_max_abs={gate_max_abs:.8e}");
    println!("gate_mean_abs={gate_mean_abs:.8e}");
    println!("up_max_abs={up_max_abs:.8e}");
    println!("up_mean_abs={up_mean_abs:.8e}");
}
