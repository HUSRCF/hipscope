// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 F16 compressor decode microbenchmark for gfx90a wave64.

use rdna_compute::{DType, Gpu};

const WARMUP: usize = 500;
const ITERS: usize = 5_000;
const SAMPLES: usize = 5;

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp_f32 == 0 {
        return sign;
    }
    if exp_f32 == 0xff {
        return sign | 0x7c00 | u16::from(mant != 0);
    }
    let exp = exp_f32 - 127 + 15;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

fn to_f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&x| f32_to_f16_bits(x).to_le_bytes())
        .collect()
}

fn synth_value(i: usize, scale: f32) -> f32 {
    ((i as f32 * 0.013_579).sin() + (i as f32 * 0.007_331).cos()) * scale
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
    eprintln!("=== DeepSeek V4 compressor F16 wave64 ===");
    eprintln!("arch={} warmup={WARMUP} iterations={ITERS}", gpu.arch);

    let mut failures = 0usize;
    for (m, k, ratio, label) in [
        (1024usize, 4096usize, 4usize, "main ratio4"),
        (512usize, 4096usize, 128usize, "main ratio128"),
        (256usize, 4096usize, 4usize, "index"),
    ] {
        eprintln!("\n--- {label}: M={m} K={k} ---");
        let w0_host: Vec<f32> = (0..m * k).map(|i| synth_value(i, 0.03125)).collect();
        let w1_host: Vec<f32> = (0..m * k)
            .map(|i| synth_value(i ^ 0x5a5a, 0.03125))
            .collect();
        let x_host: Vec<f32> = (0..k).map(|i| synth_value(i, 0.25)).collect();
        let bias_host: Vec<f32> = (0..ratio * m).map(|i| synth_value(i, 0.125)).collect();
        let pos = ratio as i32 + 1;
        let pos_bytes = pos.to_le_bytes();
        let w0_bytes = to_f16_bytes(&w0_host);
        let w1_bytes = to_f16_bytes(&w1_host);

        let w0 = gpu.upload_raw(&w0_bytes, &[w0_bytes.len()]).unwrap();
        let w1 = gpu.upload_raw(&w1_bytes, &[w1_bytes.len()]).unwrap();
        let x = gpu.upload_f32(&x_host, &[k]).unwrap();
        let bias = gpu.upload_f32(&bias_host, &[ratio, m]).unwrap();
        let bias_row = bias.sub_offset(m, m);
        let pos_buf = gpu.upload_raw(&pos_bytes, &[1]).unwrap();
        let ref0 = gpu.zeros(&[m], DType::F32).unwrap();
        let ref1 = gpu.zeros(&[m], DType::F32).unwrap();
        let wave0 = gpu.zeros(&[m], DType::F32).unwrap();
        let wave1 = gpu.zeros(&[m], DType::F32).unwrap();

        gpu.gemm_f16_tiled(&w0, &x, &ref0, m, k, 1).unwrap();
        gpu.gemm_f16_tiled(&w1, &x, &ref1, m, k, 1).unwrap();
        gpu.add_inplace_f32(&ref1, &bias_row).unwrap();
        gpu.fused_twin_f16_xf32_wave64_bias_gfx90a(
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
        let wave0_diff = compare_bits(&gpu.download_f32(&wave0).unwrap(), &ref0_host);
        let wave1_diff = compare_bits(&gpu.download_f32(&wave1).unwrap(), &ref1_host);
        eprintln!(
            "parity wave64=({},{}) max_abs=({:.3e},{:.3e})",
            wave0_diff.0, wave1_diff.0, wave0_diff.1, wave1_diff.1,
        );
        failures += usize::from(wave0_diff.0 != 0 || wave1_diff.0 != 0);

        for _ in 0..WARMUP {
            gpu.gemm_f16_tiled(&w0, &x, &ref0, m, k, 1).unwrap();
            gpu.gemm_f16_tiled(&w1, &x, &ref1, m, k, 1).unwrap();
            gpu.add_inplace_f32(&ref1, &bias_row).unwrap();
            gpu.fused_twin_f16_xf32_wave64_bias_gfx90a(
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
        let mut wave_samples = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.gemm_f16_tiled(&w0, &x, &ref0, m, k, 1).unwrap();
                gpu.gemm_f16_tiled(&w1, &x, &ref1, m, k, 1).unwrap();
                gpu.add_inplace_f32(&ref1, &bias_row).unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            pair_samples.push(started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64);

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.fused_twin_f16_xf32_wave64_bias_gfx90a(
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
                "sample={sample} pair={:.3}us wave64={:.3}us",
                pair_samples[sample], wave_samples[sample],
            );
        }

        let pair = median(pair_samples);
        let wave = median(wave_samples);
        eprintln!(
            "median pair={pair:.3}us wave64={wave:.3}us speedup={:.1}%",
            (pair / wave - 1.0) * 100.0,
        );
    }

    assert_eq!(failures, 0, "{failures} wave64 parity failures");
    eprintln!("\nALL PASS");
}
