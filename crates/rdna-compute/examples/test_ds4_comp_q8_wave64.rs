// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 compressor Q8 decode microbenchmark for gfx90a wave64.

#[path = "common/q8_test_utils.rs"]
mod q8_test_utils;

use q8_test_utils::synth_q8;
use rdna_compute::{DType, Gpu};

const WARMUP: usize = 500;
const ITERS: usize = 5_000;
const SAMPLES: usize = 5;

fn synth_x(i: usize) -> f32 {
    let x = (i as f32 * 0.013_579).sin() + (i as f32 * 0.007_331).cos();
    x * 0.25
}

fn compare_bits(a: &[f32], b: &[f32]) -> (usize, f32) {
    a.iter()
        .zip(b)
        .fold((0usize, 0.0f32), |(bad, max_abs), (&x, &y)| {
            (
                bad + usize::from(x.to_bits() != y.to_bits()),
                max_abs.max((x - y).abs()),
            )
        })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!("=== DeepSeek V4 compressor Q8 wave64 ===");
    eprintln!("arch={} warmup={WARMUP} iterations={ITERS}", gpu.arch);

    let mut failures = 0usize;
    for (m, k, ratio, label) in [
        (1024usize, 4096usize, 4usize, "main"),
        (512usize, 4096usize, 128usize, "main ratio128"),
        (256usize, 4096usize, 4usize, "index"),
    ] {
        eprintln!("\n--- {label}: M={m} K={k} ---");
        let w0_host = synth_q8(m, k, 0x3141_5926);
        let w1_host = synth_q8(m, k, 0x2718_2818);
        let x_host: Vec<f32> = (0..k).map(synth_x).collect();
        let bias_host: Vec<f32> = (0..ratio * m)
            .map(|i| ((i as f32 * 0.031_25).sin()) * 0.125)
            .collect();
        let pos = ratio as i32 + 1;
        let pos_bytes = pos.to_le_bytes();

        let w0 = gpu.upload_raw(&w0_host, &[w0_host.len()]).unwrap();
        let w1 = gpu.upload_raw(&w1_host, &[w1_host.len()]).unwrap();
        let x = gpu.upload_f32(&x_host, &[k]).unwrap();
        let bias = gpu.upload_f32(&bias_host, &[ratio, m]).unwrap();
        let bias_row = bias.sub_offset(m, m);
        let pos_buf = gpu.upload_raw(&pos_bytes, &[1]).unwrap();
        let ref0 = gpu.zeros(&[m], DType::F32).unwrap();
        let ref1 = gpu.zeros(&[m], DType::F32).unwrap();
        let old0 = gpu.zeros(&[m], DType::F32).unwrap();
        let old1 = gpu.zeros(&[m], DType::F32).unwrap();
        let wave0 = gpu.zeros(&[m], DType::F32).unwrap();
        let wave1 = gpu.zeros(&[m], DType::F32).unwrap();

        gpu.gemv_q8_0(&w0, &x, &ref0, m, k).unwrap();
        gpu.gemv_q8_0(&w1, &x, &ref1, m, k).unwrap();
        gpu.add_inplace_f32(&ref1, &bias_row).unwrap();
        gpu.fused_gate_up_q8_0(&w0, &w1, &x, &old0, &old1, m, m, k)
            .unwrap();
        gpu.add_inplace_f32(&old1, &bias_row).unwrap();
        gpu.fused_gate_up_q8_0_wave64_bias(
            &w0,
            &w1,
            &x,
            &wave0,
            &wave1,
            &bias,
            &pos_buf,
            ratio as i32,
            m,
            m,
            k,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let ref0_host = gpu.download_f32(&ref0).unwrap();
        let ref1_host = gpu.download_f32(&ref1).unwrap();
        let old0_diff = compare_bits(&gpu.download_f32(&old0).unwrap(), &ref0_host);
        let old1_diff = compare_bits(&gpu.download_f32(&old1).unwrap(), &ref1_host);
        let wave0_diff = compare_bits(&gpu.download_f32(&wave0).unwrap(), &ref0_host);
        let wave1_diff = compare_bits(&gpu.download_f32(&wave1).unwrap(), &ref1_host);
        eprintln!(
            "parity old=({},{}) wave64=({},{}) max_abs=({:.3e},{:.3e})",
            old0_diff.0, old1_diff.0, wave0_diff.0, wave1_diff.0, wave0_diff.1, wave1_diff.1,
        );
        if wave0_diff.0 != 0 || wave1_diff.0 != 0 {
            failures += 1;
        }

        for _ in 0..WARMUP {
            gpu.gemv_q8_0(&w0, &x, &ref0, m, k).unwrap();
            gpu.gemv_q8_0(&w1, &x, &ref1, m, k).unwrap();
            gpu.add_inplace_f32(&ref1, &bias_row).unwrap();
            gpu.fused_gate_up_q8_0(&w0, &w1, &x, &old0, &old1, m, m, k)
                .unwrap();
            gpu.add_inplace_f32(&old1, &bias_row).unwrap();
            gpu.fused_gate_up_q8_0_wave64_bias(
                &w0,
                &w1,
                &x,
                &wave0,
                &wave1,
                &bias,
                &pos_buf,
                ratio as i32,
                m,
                m,
                k,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let mut pair_samples = Vec::with_capacity(SAMPLES);
        let mut old_samples = Vec::with_capacity(SAMPLES);
        let mut wave_samples = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.gemv_q8_0(&w0, &x, &ref0, m, k).unwrap();
                gpu.gemv_q8_0(&w1, &x, &ref1, m, k).unwrap();
                gpu.add_inplace_f32(&ref1, &bias_row).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            pair_samples.push(started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64);

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.fused_gate_up_q8_0(&w0, &w1, &x, &old0, &old1, m, m, k)
                    .unwrap();
                gpu.add_inplace_f32(&old1, &bias_row).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            old_samples.push(started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64);

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.fused_gate_up_q8_0_wave64_bias(
                    &w0,
                    &w1,
                    &x,
                    &wave0,
                    &wave1,
                    &bias,
                    &pos_buf,
                    ratio as i32,
                    m,
                    m,
                    k,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            wave_samples.push(started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64);
            eprintln!(
                "sample={sample} pair={:.3}us old={:.3}us wave64={:.3}us",
                pair_samples[sample], old_samples[sample], wave_samples[sample],
            );
        }

        let pair = median(pair_samples);
        let old = median(old_samples);
        let wave = median(wave_samples);
        eprintln!(
            "median pair={pair:.3}us old={old:.3}us wave64={wave:.3}us speedup_pair={:.1}% speedup_old={:.1}%",
            (pair / wave - 1.0) * 100.0,
            (old / wave - 1.0) * 100.0,
        );
    }

    assert_eq!(failures, 0, "{failures} wave64 parity failures");
    eprintln!("\nALL PASS");
}
