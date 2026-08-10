// SPDX-License-Identifier: MIT OR Apache-2.0
//! Focused gfx11 group128 X256/Y64 versus X256/Y32 probe.

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
    let mut pairs = 15usize;
    let mut skip_correctness = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--m" => { i += 1; m = args[i].parse().expect("valid --m"); }
            "--k" => { i += 1; k = args[i].parse().expect("valid --k"); }
            "--n" => { i += 1; n = args[i].parse().expect("valid --n"); }
            "--pairs" => { i += 1; pairs = args[i].parse().expect("valid --pairs"); }
            "--skip-correctness" => skip_correctness = true,
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    assert!(m % 64 == 0 && k % 256 == 0 && n % 256 == 0);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={} M={m} K={k} N={n} pairs={pairs}", gpu.arch);
    let a = gpu.upload_raw(&build_hfq4g256(m, k), &[m, k]).expect("upload A");
    let x_host: Vec<f32> = (0..n * k)
        .map(|idx| ((idx * 17 + idx / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let zeros = vec![0.0f32; n * m];
    let y64 = gpu.upload_f32(&zeros, &[n, m]).expect("Y64 output");
    let y32 = gpu.upload_f32(&zeros, &[n, m]).expect("Y32 output");
    let xq_storage = gpu.hip.malloc((k / 128) * n * 144).expect("Xq");
    let xq = xq_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, xq, n, k).expect("quantize Xq");

    let run64 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &a, xq, &y64, m, k, n,
        )
    };
    let run32 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x256y32_perm_group128(
            &a, xq, &y32, m, k, n, false,
        )
    };

    run64(&mut gpu).expect("Y64 correctness");
    run32(&mut gpu).expect("Y32 correctness");
    gpu.hip.device_synchronize().expect("correctness sync");
    let max_abs = if skip_correctness {
        f32::NAN
    } else {
        let ref_host = gpu.download_f32(&y64).expect("download Y64");
        let candidate_host = gpu.download_f32(&y32).expect("download Y32");
        ref_host.iter().zip(candidate_host.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
    };

    for _ in 0..3 {
        run64(&mut gpu).expect("Y64 warmup");
        run32(&mut gpu).expect("Y32 warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut ms64 = Vec::with_capacity(pairs);
    let mut ms32 = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for y32_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if y32_first {
                run32(&mut gpu).expect("timed Y32");
            } else {
                run64(&mut gpu).expect("timed Y64");
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let ms = start.elapsed().as_secs_f64() * 1_000.0;
            if y32_first { ms32.push(ms); } else { ms64.push(ms); }
        }
    }

    let med64 = median(ms64);
    let med32 = median(ms32);
    println!("m={m} k={k} n={n}");
    println!("x256y64_ms={med64:.4}");
    println!("x256y32_ms={med32:.4}");
    println!("x256y32_speedup={:.4}x", med64 / med32);
    println!("max_abs={max_abs:.8e}");
}
