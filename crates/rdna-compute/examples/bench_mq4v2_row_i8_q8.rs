// SPDX-License-Identifier: MIT OR Apache-2.0
//! Row-scale I8/Q8 execution-format upper bound against retained packed MQ4.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn parse_usize(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn has_flag(name: &str) -> bool {
    std::env::args().any(|arg| arg == name)
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn synth_mq4(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let offset = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13 + seed) % 97) as f32 * 0.0001;
            let zero = ((row * 7 + group * 11 + seed * 3) % 31) as f32 * 0.001 - 0.015;
            out[offset..offset + 4].copy_from_slice(&scale.to_le_bytes());
            out[offset + 4..offset + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[offset + 8 + byte] =
                    ((row * 29 + group * 19 + byte * 23 + seed * 37) & 0xff) as u8;
            }
        }
    }
    out
}

fn mq4_value(src: &[u8], row: usize, k_index: usize, k: usize) -> f32 {
    let groups = k / 256;
    let group = k_index / 256;
    let in_group = k_index % 256;
    let offset = (row * groups + group) * 136;
    let scale = f32::from_le_bytes(src[offset..offset + 4].try_into().unwrap());
    let zero = f32::from_le_bytes(src[offset + 4..offset + 8].try_into().unwrap());
    let packed = src[offset + 8 + in_group / 2];
    let quant = if in_group & 1 == 0 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    scale * quant as f32 + zero
}

fn repack_row_i8(src: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut payload = vec![0u8; m * k];
    let mut scales = vec![0.0f32; m];
    for row in 0..m {
        let mut abs_max = 0.0f32;
        for k_index in 0..k {
            abs_max = abs_max.max(mq4_value(src, row, k_index, k).abs());
        }
        let scale = (abs_max / 127.0).max(f32::MIN_POSITIVE);
        scales[row] = scale;
        for k_index in 0..k {
            let value = mq4_value(src, row, k_index, k);
            let quant = (value / scale).round().clamp(-127.0, 127.0) as i8;
            payload[row * k + k_index] = quant as u8;
        }
    }
    (payload, scales)
}

fn repack_row_q4(src: &[u8], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut payload = vec![0u8; m * k / 2];
    let mut scales = vec![0.0f32; m];
    for row in 0..m {
        let mut abs_max = 0.0f32;
        for k_index in 0..k {
            abs_max = abs_max.max(mq4_value(src, row, k_index, k).abs());
        }
        let scale = (abs_max / 7.0).max(f32::MIN_POSITIVE);
        scales[row] = scale;
        for packed_index in 0..k / 2 {
            let lo = (mq4_value(src, row, 2 * packed_index, k) / scale)
                .round()
                .clamp(-8.0, 7.0) as i8;
            let hi = (mq4_value(src, row, 2 * packed_index + 1, k) / scale)
                .round()
                .clamp(-8.0, 7.0) as i8;
            let lo = (lo + 8) as u8;
            let hi = (hi + 8) as u8;
            payload[row * (k / 2) + packed_index] = lo | (hi << 4);
        }
    }
    (payload, scales)
}

fn make_x(n: usize, k: usize, seed: usize) -> Vec<f32> {
    (0..n * k)
        .map(|index| ((index * 17 + (index / k) * 31 + seed) % 101) as f32 * 0.01 - 0.5)
        .collect()
}

fn quantize_row_q8(x: &[f32], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut payload = vec![0u8; n * k];
    let mut scales = vec![0.0f32; n];
    for row in 0..n {
        let values = &x[row * k..(row + 1) * k];
        let abs_max = values
            .iter()
            .fold(0.0f32, |acc, value| acc.max(value.abs()));
        let scale = (abs_max / 127.0).max(f32::MIN_POSITIVE);
        scales[row] = scale;
        for (column, value) in values.iter().enumerate() {
            let quant = (*value / scale).round().clamp(-127.0, 127.0) as i8;
            payload[row * k + column] = quant as u8;
        }
    }
    (payload, scales)
}

fn output_metrics(gpu: &mut Gpu, reference: &GpuTensor, candidate: &GpuTensor) {
    let reference = gpu.download_f32(reference).expect("download reference");
    let candidate = gpu.download_f32(candidate).expect("download candidate");
    let mut max_abs = 0.0f32;
    let mut abs_sum = 0.0f64;
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut candidate_sq = 0.0f64;
    let mut dot = 0.0f64;
    for (reference, candidate) in reference.iter().zip(candidate.iter()) {
        let diff = (*reference - *candidate) as f64;
        max_abs = max_abs.max(diff.abs() as f32);
        abs_sum += diff.abs();
        diff_sq += diff * diff;
        ref_sq += (*reference as f64) * (*reference as f64);
        candidate_sq += (*candidate as f64) * (*candidate as f64);
        dot += (*reference as f64) * (*candidate as f64);
    }
    let count = reference.len() as f64;
    let relative_l2 = (diff_sq / ref_sq.max(f64::MIN_POSITIVE)).sqrt();
    let cosine = dot / (ref_sq.sqrt() * candidate_sq.sqrt()).max(f64::MIN_POSITIVE);
    println!("max_abs={max_abs:.8e}");
    println!("mean_abs={:.8e}", abs_sum / count);
    println!("relative_l2={relative_l2:.8e}");
    println!("cosine={cosine:.10}");
}

fn paired<F, G>(
    gpu: &mut Gpu,
    pairs: usize,
    mut reference: F,
    mut candidate: G,
) -> (Vec<f64>, Vec<f64>)
where
    F: FnMut(&mut Gpu),
    G: FnMut(&mut Gpu),
{
    for _ in 0..3 {
        reference(gpu);
        candidate(gpu);
    }
    gpu.dpm_warmup(5.0).expect("DPM warmup");
    let mut reference_ms = Vec::with_capacity(pairs);
    let mut candidate_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for reference_first in [pair % 2 == 0, pair % 2 != 0] {
            let start = Instant::now();
            if reference_first {
                reference(gpu);
                reference_ms.push(start.elapsed().as_secs_f64() * 1e3);
            } else {
                candidate(gpu);
                candidate_ms.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    (reference_ms, candidate_ms)
}

fn run_shape(
    gpu: &mut Gpu,
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    pairs: usize,
    add: bool,
    seed: usize,
    row_i8_mode: bool,
) {
    println!("shape={label} m={m} k={k} n={n} add={add}");
    let packed = synth_mq4(m, k, seed);
    let (execution_weights, weight_scales) = if row_i8_mode {
        repack_row_i8(&packed, m, k)
    } else {
        repack_row_q4(&packed, m, k)
    };
    let x_host = make_x(n, k, seed + 5);
    let (row_q8, activation_scales) = quantize_row_q8(&x_host, n, k);

    let packed_bytes = packed.len();
    let execution_bytes = execution_weights.len() + weight_scales.len() * 4;
    println!(
        "execution_format={}",
        if row_i8_mode { "row-i8" } else { "row-q4" }
    );
    println!("packed_weight_bytes={packed_bytes}");
    println!("execution_weight_bytes={execution_bytes}");
    println!(
        "execution_weight_ratio={:.4}x",
        execution_bytes as f64 / packed_bytes as f64
    );

    let packed = gpu
        .upload_raw(&packed, &[packed_bytes])
        .expect("upload MQ4");
    let execution_weights = gpu
        .upload_raw(
            &execution_weights,
            &[execution_bytes - weight_scales.len() * 4],
        )
        .expect("upload execution weights");
    let weight_scales = gpu
        .upload_f32(&weight_scales, &[m])
        .expect("upload weight scales");
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let xq = gpu
        .ensure_q8_1_mmq_x(&x, n, k)
        .expect("quantize retained X");
    let row_q8 = gpu
        .upload_raw(&row_q8, &[n, k])
        .expect("upload row-Q8 activations");
    let activation_scales = gpu
        .upload_f32(&activation_scales, &[n])
        .expect("upload activation scales");
    let reference = gpu.zeros(&[n, m], DType::F32).expect("reference output");
    let candidate = gpu.zeros(&[n, m], DType::F32).expect("candidate output");

    gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
        &packed, xq, &reference, m, k, n, add,
    )
    .expect("reference correctness");
    if row_i8_mode {
        gpu.gemm_mq4v2_row_i8_q8_wmma_256x64(
            &execution_weights,
            &weight_scales,
            &row_q8,
            &activation_scales,
            &candidate,
            m,
            k,
            n,
            add,
        )
        .expect("row-I8 correctness");
    } else {
        gpu.gemm_mq4v2_row_q4_q8_wmma_256x64(
            &execution_weights,
            &weight_scales,
            &row_q8,
            &activation_scales,
            &candidate,
            m,
            k,
            n,
            add,
        )
        .expect("row-Q4 correctness");
    }
    gpu.hip.device_synchronize().expect("correctness sync");
    output_metrics(gpu, &reference, &candidate);

    let (reference_raw, candidate_raw) = paired(
        gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &packed, xq, &reference, m, k, n, add,
            )
            .expect("reference run");
            gpu.hip.device_synchronize().expect("reference sync");
        },
        |gpu| {
            if row_i8_mode {
                gpu.gemm_mq4v2_row_i8_q8_wmma_256x64(
                    &execution_weights,
                    &weight_scales,
                    &row_q8,
                    &activation_scales,
                    &candidate,
                    m,
                    k,
                    n,
                    add,
                )
                .expect("row-I8 run");
            } else {
                gpu.gemm_mq4v2_row_q4_q8_wmma_256x64(
                    &execution_weights,
                    &weight_scales,
                    &row_q8,
                    &activation_scales,
                    &candidate,
                    m,
                    k,
                    n,
                    add,
                )
                .expect("row-Q4 run");
            }
            gpu.hip.device_synchronize().expect("candidate sync");
        },
    );
    let reference_ms = median(&reference_raw);
    let candidate_ms = median(&candidate_raw);
    println!("reference_ms={reference_ms:.4}");
    println!("candidate_ms={candidate_ms:.4}");
    println!("speedup={:.4}x", reference_ms / candidate_ms);
    println!("reference_raw_ms={reference_raw:?}");
    println!("candidate_raw_ms={candidate_raw:?}");
}

fn main() {
    let n = parse_usize("--n", 2_048);
    let pairs = parse_usize("--pairs", 7);
    let row_i8_mode = has_flag("--row-i8");
    assert_eq!(n % 256, 0);
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1100");
    println!("arch={} pairs={pairs}", gpu.arch);
    run_shape(
        &mut gpu,
        "gate",
        17_408,
        5_120,
        n,
        pairs,
        false,
        11,
        row_i8_mode,
    );
    run_shape(
        &mut gpu,
        "down",
        5_120,
        17_408,
        n,
        pairs,
        true,
        29,
        row_i8_mode,
    );
}
