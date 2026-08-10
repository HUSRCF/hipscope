// SPDX-License-Identifier: MIT OR Apache-2.0
//! Focused gfx11 packed-MQ4 dual-output probe.

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
    let mut one_packed_plane = false;
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
            "--one-packed-plane" => one_packed_plane = true,
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
        .upload_raw(&build_hfq4g256(m, k, 101), &[m, k])
        .expect("upload up");
    let x_host: Vec<f32> = (0..n * k)
        .map(|idx| ((idx * 17 + idx / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let zeros = vec![0.0f32; n * m];
    let ref_gate = gpu.upload_f32(&zeros, &[n, m]).expect("ref gate");
    let ref_up = gpu.upload_f32(&zeros, &[n, m]).expect("ref up");
    let packed_gate = gpu.upload_f32(&zeros, &[n, m]).expect("packed gate");
    let packed_up = gpu.upload_f32(&zeros, &[n, m]).expect("packed up");

    let normal_storage = gpu.hip.malloc((k / 128) * n * 144).expect("normal Xq");
    let normal_xq = normal_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, normal_xq, n, k)
        .expect("quantize normal Xq");
    let compact_storage = gpu.hip.malloc((k / 128) * n * 140).expect("compact Xq");
    let compact_xq = compact_storage.as_ptr();
    gpu.quantize_q8_1_group128_compact_into(&x, compact_xq, n, k)
        .expect("quantize compact Xq");

    let run_ref = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &gate, normal_xq, &ref_gate, m, k, n,
        )?;
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(&up, normal_xq, &ref_up, m, k, n)
    };
    let run_packed = |gpu: &mut Gpu| {
        if one_packed_plane {
            gpu.gemm_gate_up_hfq4g256_mmq_one_packed_plane_x256(
                &gate,
                &up,
                compact_xq,
                &packed_gate,
                &packed_up,
                m,
                k,
                n,
            )
        } else {
            gpu.gemm_gate_up_hfq4g256_mmq_packed_dual_x256(
                &gate,
                &up,
                compact_xq,
                &packed_gate,
                &packed_up,
                m,
                k,
                n,
            )
        }
    };

    run_ref(&mut gpu).expect("reference correctness");
    run_packed(&mut gpu).expect("packed correctness");
    gpu.hip.device_synchronize().expect("correctness sync");
    let ref_gate_host = gpu.download_f32(&ref_gate).expect("download ref gate");
    let ref_up_host = gpu.download_f32(&ref_up).expect("download ref up");
    let packed_gate_host = gpu
        .download_f32(&packed_gate)
        .expect("download packed gate");
    let packed_up_host = gpu.download_f32(&packed_up).expect("download packed up");
    let gate_max_abs = ref_gate_host
        .iter()
        .zip(&packed_gate_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let up_max_abs = ref_up_host
        .iter()
        .zip(&packed_up_host)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    for _ in 0..3 {
        run_ref(&mut gpu).expect("reference warmup");
        run_packed(&mut gpu).expect("packed warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut ref_ms = Vec::with_capacity(pairs);
    let mut packed_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for packed_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if packed_first {
                run_packed(&mut gpu).expect("timed packed");
            } else {
                run_ref(&mut gpu).expect("timed reference");
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let ms = start.elapsed().as_secs_f64() * 1_000.0;
            if packed_first {
                packed_ms.push(ms);
            } else {
                ref_ms.push(ms);
            }
        }
    }

    let ref_median = median(ref_ms);
    let packed_median = median(packed_ms);
    println!("m={m} k={k} n={n}");
    println!(
        "candidate={}",
        if one_packed_plane {
            "one_packed_plane"
        } else {
            "hybrid_packed_rows"
        }
    );
    println!("reference_two_launch_ms={ref_median:.4}");
    println!("packed_dual_ms={packed_median:.4}");
    println!("packed_dual_speedup={:.4}x", ref_median / packed_median);
    println!("gate_max_abs={gate_max_abs:.8e}");
    println!("up_max_abs={up_max_abs:.8e}");
}
