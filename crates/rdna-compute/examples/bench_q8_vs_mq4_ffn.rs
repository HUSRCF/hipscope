//! High-memory Q8_0 execution-format upper bound against retained MQ4 FFN.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        return (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp < 1 {
        return sign << 15;
    }
    if half_exp > 30 {
        return (sign << 15) | (0x1f << 10);
    }
    let mut half_mant = (mant >> 13) as u16;
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) != 0) {
        half_mant += 1;
    }
    let mut half_exp = half_exp as u16;
    if half_mant == 0x400 {
        half_mant = 0;
        half_exp += 1;
    }
    (sign << 15) | (half_exp << 10) | half_mant
}

fn signed_q4(row: usize, col: usize) -> i8 {
    (((row * 29 + col * 23) & 15) as i8) - 8
}

fn build_mq4(m: usize, k: usize, scale: f32) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&(-8.0 * scale).to_le_bytes());
            for byte in 0..128 {
                let col = group * 256 + 2 * byte;
                let lo = (signed_q4(row, col) + 8) as u8;
                let hi = (signed_q4(row, col + 1) + 8) as u8;
                out[off + 8 + byte] = lo | (hi << 4);
            }
        }
    }
    out
}

fn build_q8(m: usize, k: usize, scale: f32) -> Vec<u8> {
    let blocks = k / 32;
    let row_bytes = blocks * 34;
    let scale = f32_to_f16_bits(scale).to_le_bytes();
    let mut out = vec![0u8; m * row_bytes];
    for row in 0..m {
        for block in 0..blocks {
            let off = row * row_bytes + block * 34;
            out[off..off + 2].copy_from_slice(&scale);
            for lane in 0..32 {
                out[off + 2 + lane] = signed_q4(row, block * 32 + lane) as u8;
            }
        }
    }
    out
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn compare(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max_abs = 0.0f32;
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for (a, b) in a.iter().zip(b.iter()) {
        max_abs = max_abs.max((a - b).abs());
        let diff = (*a - *b) as f64;
        diff_sq += diff * diff;
        ref_sq += (*a as f64) * (*a as f64);
    }
    let relative_l2 = (diff_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt();
    (max_abs, relative_l2)
}

fn main() {
    let mut mode = "gate-up";
    let mut pairs = 10usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                mode = &args[i];
            }
            "--pairs" => {
                i += 1;
                pairs = args[i].parse().expect("valid --pairs");
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    assert!(mode == "gate-up" || mode == "down");

    let (m, k, add) = if mode == "gate-up" {
        (17408usize, 5120usize, false)
    } else {
        (5120usize, 17408usize, true)
    };
    let n = 2048usize;
    let scale = 0.01f32;
    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={} mode={mode} M={m} K={k} N={n} pairs={pairs}", gpu.arch);

    let mq4 = gpu.upload_raw(&build_mq4(m, k, scale), &[m, k]).expect("MQ4 weight");
    let q8 = gpu.upload_raw(&build_q8(m, k, scale), &[m, k]).expect("Q8 weight");
    let mq4_b = (mode == "gate-up")
        .then(|| gpu.upload_raw(&build_mq4(m, k, scale), &[m, k]).expect("MQ4 up weight"));
    let q8_b = (mode == "gate-up")
        .then(|| gpu.upload_raw(&build_q8(m, k, scale), &[m, k]).expect("Q8 up weight"));

    let x_host: Vec<f32> = (0..n * k)
        .map(|idx| ((idx * 17 + idx / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("activation");
    let y_mq4 = gpu.zeros(&[n, m], DType::F32).expect("MQ4 output");
    let y_q8 = gpu.zeros(&[n, m], DType::F32).expect("Q8 output");
    let y_mq4_b = (mode == "gate-up")
        .then(|| gpu.zeros(&[n, m], DType::F32).expect("MQ4 up output"));
    let y_q8_b = (mode == "gate-up")
        .then(|| gpu.zeros(&[n, m], DType::F32).expect("Q8 up output"));
    let xq_storage = gpu.hip.malloc((k / 128) * n * 144).expect("Q8_1 activation");
    let xq = xq_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, xq, n, k)
        .expect("quantize MQ4 activation");

    let run_mq4 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
            &mq4, xq, &y_mq4, m, k, n, add,
        )
        .expect("MQ4 projection");
        if let (Some(weight), Some(output)) = (mq4_b.as_ref(), y_mq4_b.as_ref()) {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                weight, xq, output, m, k, n, false,
            )
            .expect("MQ4 up projection");
        }
    };
    let run_q8 = |gpu: &mut Gpu| {
        if let (Some(weight), Some(output)) = (q8_b.as_ref(), y_q8_b.as_ref()) {
            gpu.gemm_gate_up_q8_0_wmma(&q8, weight, &x, &y_q8, output, m, m, k, n)
                .expect("Q8 gate/up");
        } else {
            gpu.gemm_q8_0_residual_wmma(&q8, &x, &y_q8, m, k, n)
                .expect("Q8 down");
        }
    };

    run_mq4(&mut gpu);
    run_q8(&mut gpu);
    gpu.hip.device_synchronize().expect("correctness sync");
    let first = compare(
        &gpu.download_f32(&y_mq4).expect("download MQ4"),
        &gpu.download_f32(&y_q8).expect("download Q8"),
    );
    let second = if let (Some(mq4_out), Some(q8_out)) = (y_mq4_b.as_ref(), y_q8_b.as_ref()) {
        Some(compare(
            &gpu.download_f32(mq4_out).expect("download MQ4 up"),
            &gpu.download_f32(q8_out).expect("download Q8 up"),
        ))
    } else {
        None
    };

    for _ in 0..3 {
        run_mq4(&mut gpu);
        run_q8(&mut gpu);
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut mq4_ms = Vec::with_capacity(pairs);
    let mut q8_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for q8_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if q8_first {
                run_q8(&mut gpu);
            } else {
                run_mq4(&mut gpu);
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let ms = start.elapsed().as_secs_f64() * 1_000.0;
            if q8_first {
                q8_ms.push(ms);
            } else {
                mq4_ms.push(ms);
            }
        }
    }

    let mq4_ms = median(mq4_ms);
    let q8_ms = median(q8_ms);
    let mq4_weight_bytes = m * (k / 256) * 136 * if mode == "gate-up" { 2 } else { 1 };
    let q8_weight_bytes = m * (k / 32) * 34 * if mode == "gate-up" { 2 } else { 1 };
    println!("mode={mode}");
    println!("m={m} k={k} n={n}");
    println!("mq4_ms={mq4_ms:.4}");
    println!("q8_ms={q8_ms:.4}");
    println!("q8_speedup={:.4}x", mq4_ms / q8_ms);
    println!("mq4_weight_bytes={mq4_weight_bytes}");
    println!("q8_weight_bytes={q8_weight_bytes}");
    println!("weight_ratio={:.4}x", q8_weight_bytes as f64 / mq4_weight_bytes as f64);
    println!("first_max_abs={:.8e}", first.0);
    println!("first_relative_l2={:.8e}", first.1);
    if let Some(second) = second {
        println!("second_max_abs={:.8e}", second.0);
        println!("second_relative_l2={:.8e}", second.1);
    }
}
