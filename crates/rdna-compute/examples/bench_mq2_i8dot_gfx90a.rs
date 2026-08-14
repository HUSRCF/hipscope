//! gfx90a MQ2-I8DOT affine gate/up correctness and throughput probe.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const A8_BLOCK: usize = 136;

fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return (sign << 15) | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    if exponent > 142 {
        return (sign << 15) | 0x7c00;
    }
    if exponent >= 113 {
        let half_exp = (exponent - 112) as u16;
        let mut half_mantissa = (mantissa >> 13) as u16;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && half_mantissa & 1 != 0) {
            half_mantissa += 1;
            if half_mantissa == 0x0400 {
                return (sign << 15) | ((half_exp + 1) << 10);
            }
        }
        return (sign << 15) | (half_exp << 10) | half_mantissa;
    }
    if exponent >= 103 {
        let shift = (113 - exponent) as u32;
        let full = mantissa | 0x80_0000;
        let mut half_mantissa = (full >> (shift + 13)) as u16;
        let remainder_mask = (1_u32 << (shift + 13)) - 1;
        let remainder = full & remainder_mask;
        let halfway = 1_u32 << (shift + 12);
        if remainder > halfway || (remainder == halfway && half_mantissa & 1 != 0) {
            half_mantissa += 1;
        }
        return (sign << 15) | half_mantissa;
    }
    sign << 15
}

fn f16_value(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x03ff) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut mantissa = mantissa;
        let mut exponent = -14_i32;
        while mantissa & 0x400 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        return f32::from_bits(
            sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x3ff) << 13),
        );
    }
    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | (((exponent as i32 - 15 + 127) as u32) << 23) | (mantissa << 13))
}

fn make_expert(m: usize, k: usize, seed: u32) -> (Vec<u8>, Vec<u8>) {
    let groups = k / 256;
    let mut state = seed;
    let mut lloyd = Vec::with_capacity(m * groups * 72);
    let mut affine = Vec::with_capacity(m * groups * 72);
    for row in 0..m {
        for group in 0..groups {
            let q = [
                -127_i8,
                (-72 + ((row + 3 * group) % 48) as i32) as i8,
                (18 + ((5 * row + group) % 44) as i32) as i8,
                127_i8,
            ];
            let sw_bits = f16_bits(0.0009 + ((row + group) % 13) as f32 * 0.000_075);
            let bw_bits = f16_bits((((row * 7 + group * 11) % 17) as f32 - 8.0) * 0.000_45);
            let sw = f16_value(sw_bits);
            let bw = f16_value(bw_bits);
            for &code in &q {
                lloyd.extend_from_slice(&f16_bits(sw.mul_add(code as f32, bw)).to_le_bytes());
                affine.push(code as u8);
            }
            affine.extend_from_slice(&sw_bits.to_le_bytes());
            affine.extend_from_slice(&bw_bits.to_le_bytes());
            for _ in 0..64 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let packed = (state >> 24) as u8;
                lloyd.push(packed);
                affine.push(packed);
            }
        }
    }
    (lloyd, affine)
}

fn transcode_lloyd_affine(weights: &[u8]) -> Vec<u8> {
    assert_eq!(weights.len() % 72, 0);
    let mut out = Vec::with_capacity(weights.len());
    for group in weights.chunks_exact(72) {
        let c = [
            f16_value(u16::from_le_bytes([group[0], group[1]])),
            f16_value(u16::from_le_bytes([group[2], group[3]])),
            f16_value(u16::from_le_bytes([group[4], group[5]])),
            f16_value(u16::from_le_bytes([group[6], group[7]])),
        ];
        let mut counts = [0.0_f32; 4];
        for &packed in &group[8..] {
            for shift in [0, 2, 4, 6] {
                counts[((packed >> shift) & 3) as usize] += 1.0;
            }
        }
        let initial_scale = ((c[3] - c[0]) / 254.0).max(1.0e-12);
        let initial_bias = 0.5 * (c[3] + c[0]);
        let mut q = [-127_i32, 0, 0, 127];
        q[1] = (((c[1] - initial_bias) / initial_scale).round() as i32).clamp(-126, 125);
        q[2] = (((c[2] - initial_bias) / initial_scale).round() as i32).clamp(q[1] + 1, 126);
        let sw = counts.iter().sum::<f32>();
        let swq = (0..4).map(|i| counts[i] * q[i] as f32).sum::<f32>();
        let swqq = (0..4)
            .map(|i| counts[i] * (q[i] * q[i]) as f32)
            .sum::<f32>();
        let swc = (0..4).map(|i| counts[i] * c[i]).sum::<f32>();
        let swqc = (0..4).map(|i| counts[i] * q[i] as f32 * c[i]).sum::<f32>();
        let det = swqq * sw - swq * swq;
        let scale = if det.abs() > 1.0e-20 {
            (swqc * sw - swc * swq) / det
        } else {
            initial_scale
        };
        let bias = if sw > 0.0 {
            (swc - scale * swq) / sw
        } else {
            initial_bias
        };
        out.extend(q.map(|v| v as i8 as u8));
        out.extend_from_slice(&f16_bits(scale).to_le_bytes());
        out.extend_from_slice(&f16_bits(bias).to_le_bytes());
        out.extend_from_slice(&group[8..]);
    }
    out
}

#[derive(Clone)]
struct HostA8 {
    d: f32,
    sum: i32,
    qs: [i8; 128],
}

fn quantize_a8(x: &[f32]) -> Vec<HostA8> {
    x.chunks_exact(128)
        .map(|chunk| {
            let amax = chunk.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            let d = amax * (1.0 / 127.0);
            let inv_d = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            let mut qs = [0_i8; 128];
            let mut sum = 0_i32;
            for (dst, &value) in qs.iter_mut().zip(chunk) {
                let q = (value * inv_d).round().clamp(-127.0, 127.0) as i8;
                *dst = q;
                sum += q as i32;
            }
            HostA8 { d, sum, qs }
        })
        .collect()
}

fn verify_a8(gpu_bytes: &[u8], host: &[HostA8]) -> bool {
    let mut bad_q = 0_usize;
    let mut bad_sum = 0_usize;
    let mut max_scale = 0.0_f32;
    for (group, expected) in host.iter().enumerate() {
        let base = group * A8_BLOCK;
        let d = f32::from_le_bytes(gpu_bytes[base..base + 4].try_into().unwrap());
        let sum = i32::from_le_bytes(gpu_bytes[base + 4..base + 8].try_into().unwrap());
        max_scale = max_scale.max((d - expected.d).abs());
        bad_sum += usize::from(sum != expected.sum);
        for i in 0..128 {
            bad_q += usize::from(gpu_bytes[base + 8 + i] as i8 != expected.qs[i]);
        }
    }
    println!("A8 parity: bad_q={bad_q} bad_sum={bad_sum} max_scale_abs={max_scale:.3e}");
    bad_q == 0 && bad_sum == 0 && max_scale < 1.0e-7
}

fn i8dot_reference(weights: &[u8], row: usize, xq: &[HostA8], m: usize, k: usize) -> f32 {
    assert!(row < m);
    let groups = k / 256;
    let row_bytes = groups * 72;
    let mut output = 0.0_f32;
    for group in 0..groups {
        let gp = row * row_bytes + group * 72;
        let lut = [
            weights[gp] as i8,
            weights[gp + 1] as i8,
            weights[gp + 2] as i8,
            weights[gp + 3] as i8,
        ];
        let sw = f16_value(u16::from_le_bytes([weights[gp + 4], weights[gp + 5]]));
        let bw = f16_value(u16::from_le_bytes([weights[gp + 6], weights[gp + 7]]));
        for subgroup in 0..2 {
            let xb = &xq[group * 2 + subgroup];
            let mut dot = 0_i32;
            for i in 0..128 {
                let wi = subgroup * 128 + i;
                let packed = weights[gp + 8 + wi / 4];
                let code = ((packed >> (2 * (wi & 3))) & 3) as usize;
                dot += xb.qs[i] as i32 * lut[code] as i32;
            }
            output += xb.d * sw.mul_add(dot as f32, bw * xb.sum as f32);
        }
    }
    output
}

fn upload_ptrs(gpu: &mut Gpu, tensors: &[GpuTensor]) -> GpuTensor {
    let bytes: Vec<u8> = tensors
        .iter()
        .flat_map(|tensor| (tensor.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    gpu.upload_raw(&bytes, &[bytes.len()])
        .expect("upload pointers")
}

fn output_metrics(label: &str, gate: &[f32], up: &[f32], gate_ref: &[f32], up_ref: &[f32]) {
    let mut err2 = 0.0_f64;
    let mut ref2 = 0.0_f64;
    let mut max_abs = 0.0_f32;
    for (&actual, &expected) in gate.iter().zip(gate_ref).chain(up.iter().zip(up_ref)) {
        let error = actual - expected;
        err2 += (error as f64).powi(2);
        ref2 += (expected as f64).powi(2);
        max_abs = max_abs.max(error.abs());
    }
    println!(
        "{label}: rel_rms={:.4}% max_abs={max_abs:.5e}",
        100.0 * (err2 / ref2.max(1.0e-30)).sqrt()
    );
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let m = env_usize("HIPFIRE_I8DOT_M", 8192);
    let k = env_usize("HIPFIRE_I8DOT_K", 4096);
    let top_k = env_usize("HIPFIRE_I8DOT_TOP_K", 6);
    let warmup = env_usize("HIPFIRE_I8DOT_WARMUP", 50);
    let perf_only = std::env::var_os("HIPFIRE_I8DOT_PERF_ONLY").is_some();
    let iters = env_usize("HIPFIRE_I8DOT_ITERS", 200);
    assert!(m % 2 == 0 && k % 256 == 0 && top_k > 0);

    let mut gpu = Gpu::init().expect("Gpu::init");
    println!(
        "arch={} total_m={m} split_m={} K={k} top_k={top_k} warmup={warmup} iters={iters}",
        gpu.arch,
        m / 2
    );
    assert_eq!(gpu.arch, "gfx90a", "this probe requires gfx90a");

    let pairs: Vec<(Vec<u8>, Vec<u8>)> = match (
        std::env::var_os("HIPFIRE_I8DOT_REAL_W1"),
        std::env::var_os("HIPFIRE_I8DOT_REAL_W3"),
    ) {
        (Some(w1), Some(w3)) => {
            let w1 = std::fs::read(w1).expect("read real w1");
            let w3 = std::fs::read(w3).expect("read real w3");
            let mut lloyd = Vec::with_capacity(w1.len() + w3.len());
            lloyd.extend_from_slice(&w1);
            lloyd.extend_from_slice(&w3);
            assert_eq!(
                lloyd.len(),
                m * (k / 256) * 72,
                "real gate/up shape mismatch"
            );
            let affine =
                rdna_compute::mq2_i8dot::transcode_affine(&lloyd).expect("transcode real MQ2");
            println!("real MQ2 tensors active: {} bytes/expert", lloyd.len());
            (0..top_k)
                .map(|_| (lloyd.clone(), affine.clone()))
                .collect()
        }
        (None, None) => (0..top_k)
            .map(|expert| make_expert(m, k, 0x1234_5678 + expert as u32 * 977))
            .collect(),
        _ => panic!("set both HIPFIRE_I8DOT_REAL_W1 and HIPFIRE_I8DOT_REAL_W3"),
    };
    let lloyd_tensors: Vec<GpuTensor> = pairs
        .iter()
        .map(|(weights, _)| {
            gpu.upload_raw(weights, &[weights.len()])
                .expect("upload Lloyd")
        })
        .collect();
    let affine_tensors: Vec<GpuTensor> = pairs
        .iter()
        .map(|(_, weights)| {
            gpu.upload_raw(weights, &[weights.len()])
                .expect("upload affine")
        })
        .collect();
    let affine_tiled_tensors: Vec<GpuTensor> = pairs
        .iter()
        .map(|(_, weights)| {
            let tiled = rdna_compute::mq2_i8dot::tile_sg8(weights, m, k).expect("tile affine SG8");
            gpu.upload_raw(&tiled, &[tiled.len()])
                .expect("upload tiled affine")
        })
        .collect();
    let lloyd_ptrs = upload_ptrs(&mut gpu, &lloyd_tensors);
    let affine_ptrs = upload_ptrs(&mut gpu, &affine_tensors);
    let affine_tiled_ptrs = upload_ptrs(&mut gpu, &affine_tiled_tensors);
    let index_bytes: Vec<u8> = (0..top_k as i32).flat_map(i32::to_le_bytes).collect();
    let indices = gpu
        .upload_raw(&index_bytes, &[index_bytes.len()])
        .expect("upload indices");

    let x: Vec<f32> = (0..k)
        .map(|i| {
            let base = ((i as f32 * 0.173).sin() + (i as f32 * 0.037).cos()) * 0.52;
            if i % 127 == 0 {
                base * 1.8
            } else {
                base
            }
        })
        .collect();
    let x_gpu = gpu.upload_f32(&x, &[k]).expect("upload x");
    let host_xq = quantize_a8(&x);
    let xq_bytes = (k / 128) * A8_BLOCK;
    let xq = gpu
        .alloc_tensor(&[xq_bytes], DType::Raw)
        .expect("allocate A8");
    let xq_sg8 = gpu
        .alloc_tensor(&[xq_bytes], DType::Raw)
        .expect("allocate SG8 A8");
    gpu.deepseek4_quantize_f32_a8_g128_gfx90a(&x_gpu, &xq, k, false)
        .expect("quantize A8");
    gpu.hip.device_synchronize().expect("quant sync");
    let mut gpu_xq = vec![0_u8; xq_bytes];
    gpu.hip
        .memcpy_dtoh(&mut gpu_xq, &xq.buf)
        .expect("download A8");
    assert!(verify_a8(&gpu_xq, &host_xq), "A8 parity failed");
    gpu.deepseek4_quantize_f32_a8_g128_gfx90a(&x_gpu, &xq_sg8, k, true)
        .expect("quantize SG8 A8");

    let out_len = top_k * (m / 2);
    let gate_ref_gpu = gpu
        .alloc_tensor(&[out_len], DType::F32)
        .expect("allocate reference gate");
    let up_ref_gpu = gpu
        .alloc_tensor(&[out_len], DType::F32)
        .expect("allocate reference up");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &lloyd_ptrs,
        &indices,
        &x_gpu,
        &gate_ref_gpu,
        &up_ref_gpu,
        m,
        k,
        top_k,
    )
    .expect("Lloyd reference");
    gpu.hip.device_synchronize().expect("reference sync");
    let gate_ref = gpu.download_f32(&gate_ref_gpu).expect("download gate ref");
    let up_ref = gpu.download_f32(&up_ref_gpu).expect("download up ref");

    let gate = gpu
        .alloc_tensor(&[out_len], DType::F32)
        .expect("allocate I8 gate");
    let up = gpu
        .alloc_tensor(&[out_len], DType::F32)
        .expect("allocate I8 up");

    let variants: &[usize] = if perf_only {
        &[
            8, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
        ]
    } else {
        &[1, 2, 4, 8, 16]
    };
    for &row_tile in variants {
        let selected_ptrs = if row_tile == 8 || row_tile >= 80 {
            &affine_tiled_ptrs
        } else {
            &affine_ptrs
        };
        gpu.deepseek4_gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_gfx90a(
            selected_ptrs,
            &indices,
            if row_tile == 8 || row_tile >= 80 {
                &xq_sg8
            } else {
                &xq
            },
            &gate,
            &up,
            m,
            k,
            top_k,
            row_tile,
        )
        .expect("I8DOT parity");
        gpu.hip.device_synchronize().expect("I8DOT parity sync");
        let gate_host = gpu.download_f32(&gate).expect("download I8 gate");
        let up_host = gpu.download_f32(&up).expect("download I8 up");
        output_metrics(
            &format!("row{row_tile} vs Lloyd"),
            &gate_host,
            &up_host,
            &gate_ref,
            &up_ref,
        );

        let sample_rows = [0, 1, m / 4, m / 2 - 1, m / 2, m / 2 + 1, m - 2, m - 1];
        let mut max_abs = 0.0_f32;
        let mut bad = 0_usize;
        for expert in 0..top_k {
            for &row in &sample_rows {
                let expected = i8dot_reference(&pairs[expert].1, row, &host_xq, m, k);
                let actual = if row < m / 2 {
                    gate_host[expert * (m / 2) + row]
                } else {
                    up_host[expert * (m / 2) + row - m / 2]
                };
                let error = (actual - expected).abs();
                max_abs = max_abs.max(error);
                bad += usize::from(error > 3.0e-4);
            }
        }
        println!("row{row_tile} CPU-I8 probe: bad={bad} max_abs={max_abs:.5e}");
        if !perf_only {
            assert_eq!(bad, 0, "row{row_tile} CPU-I8 parity failed");
        }
    }

    let batch_test = env_usize("HIPFIRE_I8DOT_BATCH_TEST", 0);
    if batch_test > 0 {
        let mut x_batch = Vec::with_capacity(batch_test * k);
        for _ in 0..batch_test {
            x_batch.extend_from_slice(&x);
        }
        let x_batch_gpu = gpu
            .upload_f32(&x_batch, &[batch_test, k])
            .expect("upload batched x");
        let xq_batch = gpu
            .alloc_tensor(&[batch_test * xq_bytes], DType::Raw)
            .expect("allocate batched A8");
        let batch_index_bytes: Vec<u8> = (0..batch_test)
            .flat_map(|_| (0..top_k as i32).flat_map(i32::to_le_bytes))
            .collect();
        let batch_indices = gpu
            .upload_raw(&batch_index_bytes, &[batch_index_bytes.len()])
            .expect("upload batched indices");
        let gate_batch = gpu
            .alloc_tensor(&[batch_test * out_len], DType::F32)
            .expect("allocate batched gate");
        let up_batch = gpu
            .alloc_tensor(&[batch_test * out_len], DType::F32)
            .expect("allocate batched up");

        gpu.deepseek4_quantize_f32_a8_g128_sg8_batched_gfx90a(
            &x_batch_gpu,
            &xq_batch,
            k,
            batch_test,
        )
        .expect("batched A8 quantize");
        gpu.hip.device_synchronize().expect("batched A8 sync");
        let mut xq_single_bytes = vec![0_u8; xq_bytes];
        gpu.hip
            .memcpy_dtoh(&mut xq_single_bytes, &xq_sg8.buf)
            .expect("download single SG8 A8");
        let mut xq_batch_bytes = vec![0_u8; batch_test * xq_bytes];
        gpu.hip
            .memcpy_dtoh(&mut xq_batch_bytes, &xq_batch.buf)
            .expect("download batched SG8 A8");
        assert!(
            xq_batch_bytes
                .chunks_exact(xq_bytes)
                .all(|token| token == xq_single_bytes),
            "batched SG8 A8 differs from per-token reference"
        );

        gpu.deepseek4_gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_gfx90a(
            &affine_tiled_ptrs,
            &indices,
            &xq_sg8,
            &gate,
            &up,
            m,
            k,
            top_k,
            87,
        )
        .expect("single PIPE2 reference");
        gpu.hip.device_synchronize().expect("single PIPE2 sync");
        let gate_single = gpu.download_f32(&gate).expect("download single gate");
        let up_single = gpu.download_f32(&up).expect("download single up");

        gpu.deepseek4_gemv_mq2g256_i8dot_affine_moe_gate_up_batched_gfx90a(
            &affine_tiled_ptrs,
            &batch_indices,
            &xq_batch,
            &gate_batch,
            &up_batch,
            m,
            k,
            top_k,
            batch_test,
        )
        .expect("batched PIPE2");
        gpu.hip.device_synchronize().expect("batched PIPE2 sync");
        let gate_batch_host = gpu
            .download_f32(&gate_batch)
            .expect("download batched gate");
        let up_batch_host = gpu.download_f32(&up_batch).expect("download batched up");
        let mut bad = 0_usize;
        let mut max_abs = 0.0_f32;
        for token in 0..batch_test {
            for (&actual, &expected) in gate_batch_host[token * out_len..(token + 1) * out_len]
                .iter()
                .zip(&gate_single)
                .chain(
                    up_batch_host[token * out_len..(token + 1) * out_len]
                        .iter()
                        .zip(&up_single),
                )
            {
                let error = (actual - expected).abs();
                max_abs = max_abs.max(error);
                bad += usize::from(error > 1.0e-6);
            }
        }
        println!("batched prefill parity: batch={batch_test} bad={bad} max_abs={max_abs:.3e}");
        assert_eq!(bad, 0, "batched PIPE2 differs from per-token reference");
    }

    for _ in 0..warmup {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            &lloyd_ptrs,
            &indices,
            &x_gpu,
            &gate_ref_gpu,
            &up_ref_gpu,
            m,
            k,
            top_k,
        )
        .expect("reference warmup");
    }
    gpu.hip.device_synchronize().expect("reference warmup sync");
    let start = Instant::now();
    for _ in 0..iters {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            &lloyd_ptrs,
            &indices,
            &x_gpu,
            &gate_ref_gpu,
            &up_ref_gpu,
            m,
            k,
            top_k,
        )
        .expect("reference timed");
    }
    gpu.hip.device_synchronize().expect("reference timed sync");
    let baseline_us = start.elapsed().as_secs_f64() * 1.0e6 / iters as f64;
    println!("Lloyd row2 kernel-only: {baseline_us:.3} us");

    let quant_iters = iters.max(1000);
    for _ in 0..warmup {
        gpu.deepseek4_quantize_f32_a8_g128_gfx90a(&x_gpu, &xq, k, false)
            .expect("quant warmup");
    }
    gpu.hip.device_synchronize().expect("quant warmup sync");
    let start = Instant::now();
    for _ in 0..quant_iters {
        gpu.deepseek4_quantize_f32_a8_g128_gfx90a(&x_gpu, &xq, k, false)
            .expect("quant timed");
    }
    gpu.hip.device_synchronize().expect("quant timed sync");
    let quant_us = start.elapsed().as_secs_f64() * 1.0e6 / quant_iters as f64;
    println!("A8 quant per-128: {quant_us:.3} us");

    let mut best = (0_usize, f64::INFINITY);
    let variants: &[usize] = if perf_only {
        &[
            8, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
        ]
    } else {
        &[1, 2, 4, 8, 16]
    };
    for &row_tile in variants {
        let selected_ptrs = if row_tile == 8 || row_tile >= 80 {
            &affine_tiled_ptrs
        } else {
            &affine_ptrs
        };
        for _ in 0..warmup {
            gpu.deepseek4_gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_gfx90a(
                selected_ptrs,
                &indices,
                if row_tile == 8 || row_tile >= 80 {
                    &xq_sg8
                } else {
                    &xq
                },
                &gate,
                &up,
                m,
                k,
                top_k,
                row_tile,
            )
            .expect("I8DOT warmup");
        }
        gpu.hip.device_synchronize().expect("I8DOT warmup sync");
        let start = Instant::now();
        for _ in 0..iters {
            gpu.deepseek4_gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_gfx90a(
                selected_ptrs,
                &indices,
                if row_tile == 8 || row_tile >= 80 {
                    &xq_sg8
                } else {
                    &xq
                },
                &gate,
                &up,
                m,
                k,
                top_k,
                row_tile,
            )
            .expect("I8DOT timed");
        }
        gpu.hip.device_synchronize().expect("I8DOT timed sync");
        let kernel_us = start.elapsed().as_secs_f64() * 1.0e6 / iters as f64;
        let gain = (baseline_us / kernel_us - 1.0) * 100.0;
        println!(
            "I8DOT row{row_tile} kernel-only: {kernel_us:.3} us ({gain:+.1}%), amortized={:.3} us",
            kernel_us + quant_us
        );
        // 81-84, 88-91, 93, and 94 are diagnostic ablations or use non-refitted metadata.
        // They do not preserve the affine reference and must not be selected.
        let numerically_valid = matches!(row_tile, 1 | 2 | 4 | 8 | 16 | 85 | 86 | 87 | 92);
        if numerically_valid && kernel_us < best.1 {
            best = (row_tile, kernel_us);
        }
    }

    let best_ptrs = if best.0 == 8 || best.0 >= 80 {
        &affine_tiled_ptrs
    } else {
        &affine_ptrs
    };
    let best_xq = if best.0 == 8 || best.0 >= 80 {
        &xq_sg8
    } else {
        &xq
    };
    for _ in 0..warmup {
        gpu.deepseek4_quantize_f32_a8_g128_gfx90a(&x_gpu, best_xq, k, best.0 == 8 || best.0 >= 80)
            .expect("combined quant warmup");
        gpu.deepseek4_gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_gfx90a(
            best_ptrs, &indices, best_xq, &gate, &up, m, k, top_k, best.0,
        )
        .expect("combined I8DOT warmup");
    }
    gpu.hip.device_synchronize().expect("combined warmup sync");
    let start = Instant::now();
    for _ in 0..iters {
        gpu.deepseek4_quantize_f32_a8_g128_gfx90a(&x_gpu, best_xq, k, best.0 == 8 || best.0 >= 80)
            .expect("combined quant timed");
        gpu.deepseek4_gemv_mq2g256_i8dot_affine_moe_gate_up_indexed_gfx90a(
            best_ptrs, &indices, best_xq, &gate, &up, m, k, top_k, best.0,
        )
        .expect("combined I8DOT timed");
    }
    gpu.hip.device_synchronize().expect("combined timed sync");
    let combined_us = start.elapsed().as_secs_f64() * 1.0e6 / iters as f64;
    println!(
        "BEST valid row{}: kernel={:.3} us direct_quant+kernel={combined_us:.3} us total_gain={:+.1}%",
        best.0,
        best.1,
        (baseline_us / combined_us - 1.0) * 100.0
    );
}
fn tile_affine_sg8(weights: &[u8], m: usize, k: usize) -> Vec<u8> {
    assert!(m % 8 == 0 && k % 256 == 0);
    let groups = k / 256;
    let row_bytes = groups * 72;
    let mut tiled = Vec::with_capacity(weights.len());
    for tile in 0..m / 8 {
        for group in 0..groups {
            for row_local in 0..8 {
                let base = (tile * 8 + row_local) * row_bytes + group * 72;
                tiled.extend_from_slice(&weights[base..base + 8]);
            }
            for batch in 0..2 {
                for subgroup in 0..8 {
                    let row_local = batch * 4 + subgroup / 2;
                    let half = subgroup & 1;
                    let base = (tile * 8 + row_local) * row_bytes + group * 72;
                    for lane8 in 0..8 {
                        for chunk in 0..4 {
                            tiled.push(weights[base + 8 + half * 32 + lane8 + chunk * 8]);
                        }
                    }
                }
            }
        }
    }
    assert_eq!(tiled.len(), weights.len());
    tiled
}
