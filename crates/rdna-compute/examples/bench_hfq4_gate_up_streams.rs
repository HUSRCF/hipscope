// SPDX-License-Identifier: Apache-2.0
//! Test whether the two independent FFN gate/up projections overlap on gfx11.
//!
//! This benchmark keeps the production packed-MQ4/Q8 path unchanged. It
//! quantizes the shared activation once, then compares two projection launches
//! on one HIP stream with one launch on each of two HIP streams.

use rdna_compute::{DType, Gpu};
use std::time::Instant;

const GROUP: usize = 256;
const GROUP_BYTES: usize = 136;

fn synth_hfq4_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / GROUP;
    let mut out = vec![0u8; m * groups * GROUP_BYTES];
    let mut state = seed;
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * GROUP_BYTES;
            let scale = 0.005 + ((row * 17 + group * 13) % 97) as f32 * 0.00005;
            let zero = ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                out[off + 8 + byte] = (state >> 32) as u8;
            }
        }
    }
    out
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    match exp {
        0 if mant == 0 => f32::from_bits(sign),
        0 => {
            let mut normalized = mant;
            let mut exponent = -14i32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                exponent -= 1;
            }
            f32::from_bits(sign | (((exponent + 127) as u32) << 23) | ((normalized & 0x03ff) << 13))
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mant << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13)),
    }
}

fn parse_arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
}

fn launch_projection(
    gpu: &mut Gpu,
    weight: &rdna_compute::GpuTensor,
    xq: *mut std::ffi::c_void,
    output: &rdna_compute::GpuTensor,
    m: usize,
    k: usize,
    n: usize,
) {
    gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(weight, xq, output, m, k, n)
        .expect("projection launch");
}

fn main() {
    let m = parse_arg("--m", 17_408);
    let k = parse_arg("--k", 5_120);
    let n = parse_arg("--n", 2_048);
    let pairs = parse_arg("--pairs", 10);
    assert!(m % 64 == 0 && k % GROUP == 0 && n % 256 == 0);
    let combined_m = m.checked_mul(2).expect("combined gate/up M overflow");
    n.checked_mul(combined_m)
        .expect("combined gate/up output size overflow");

    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1100", "this probe is scoped to RDNA3 gfx1100");
    eprintln!("arch={} M={m} K={k} N={n} pairs={pairs}", gpu.arch);

    let gate_host = synth_hfq4_weights(m, k, 0x1234_5678);
    let up_host = synth_hfq4_weights(m, k, 0x9abc_def0);
    let mut combined_host = Vec::with_capacity(gate_host.len() + up_host.len());
    combined_host.extend_from_slice(&gate_host);
    combined_host.extend_from_slice(&up_host);
    let gate_weight = gpu.upload_raw(&gate_host, &[m, k]).expect("upload gate");
    let up_weight = gpu.upload_raw(&up_host, &[m, k]).expect("upload up");
    let combined_weight = gpu
        .upload_raw(&combined_host, &[combined_m, k])
        .expect("upload combined gate/up");
    let down_host = synth_hfq4_weights(k, m, 0x0ddc_0ffe);
    let down_weight = gpu
        .upload_raw(&down_host, &[k, m])
        .expect("upload down");
    drop(gate_host);
    drop(up_host);
    drop(combined_host);
    drop(down_host);

    let x_host: Vec<f32> = (0..n * k)
        .map(|i| ((i * 17 + i / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    drop(x_host);
    let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize X");
    gpu.hip.device_synchronize().expect("sync prequant");

    let serial_gate = gpu.zeros(&[n, m], DType::F32).expect("serial gate");
    let serial_up = gpu.zeros(&[n, m], DType::F32).expect("serial up");
    let parallel_gate = gpu.zeros(&[n, m], DType::F32).expect("parallel gate");
    let parallel_up = gpu.zeros(&[n, m], DType::F32).expect("parallel up");
    let f16_gate = gpu.zeros(&[n, m], DType::F16).expect("F16 gate");
    let f16_up = gpu.zeros(&[n, m], DType::F16).expect("F16 up");
    let combined_output = gpu
        .zeros(&[n, combined_m], DType::F32)
        .expect("combined gate/up output");
    let f32_ffn_output = gpu.zeros(&[n, k], DType::F32).expect("F32 FFN output");
    let f16_ffn_output = gpu.zeros(&[n, k], DType::F32).expect("F16 FFN output");

    let mut serial_stream = Some(gpu.hip.stream_create().expect("serial stream"));
    let mut gate_stream = Some(gpu.hip.stream_create().expect("gate stream"));
    let mut up_stream = Some(gpu.hip.stream_create().expect("up stream"));

    let run_serial = |gpu: &mut Gpu, stream: &mut Option<hip_bridge::Stream>| {
        gpu.active_stream = stream.take();
        launch_projection(gpu, &gate_weight, xq, &serial_gate, m, k, n);
        launch_projection(gpu, &up_weight, xq, &serial_up, m, k, n);
        *stream = gpu.active_stream.take();
        gpu.hip
            .stream_synchronize(stream.as_ref().unwrap())
            .expect("sync serial");
    };
    let run_parallel = |
        gpu: &mut Gpu,
        gate_stream: &mut Option<hip_bridge::Stream>,
        up_stream: &mut Option<hip_bridge::Stream>,
    | {
        gpu.active_stream = gate_stream.take();
        launch_projection(gpu, &gate_weight, xq, &parallel_gate, m, k, n);
        *gate_stream = gpu.active_stream.take();

        gpu.active_stream = up_stream.take();
        launch_projection(gpu, &up_weight, xq, &parallel_up, m, k, n);
        *up_stream = gpu.active_stream.take();

        gpu.hip
            .stream_synchronize(gate_stream.as_ref().unwrap())
            .expect("sync gate");
        gpu.hip
            .stream_synchronize(up_stream.as_ref().unwrap())
            .expect("sync up");
    };

    for _ in 0..3 {
        run_serial(&mut gpu, &mut serial_stream);
        run_parallel(&mut gpu, &mut gate_stream, &mut up_stream);
    }
    gpu.dpm_warmup(5.0).expect("DPM warmup");

    let mut serial_ms = Vec::with_capacity(pairs);
    let mut parallel_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let serial_first = pair % 2 == 0;
        for serial in [serial_first, !serial_first] {
            let start = Instant::now();
            if serial {
                run_serial(&mut gpu, &mut serial_stream);
                serial_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            } else {
                run_parallel(&mut gpu, &mut gate_stream, &mut up_stream);
                parallel_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
        }
    }

    let serial_gate_host = gpu.download_f32(&serial_gate).expect("download serial gate");
    let parallel_gate_host = gpu
        .download_f32(&parallel_gate)
        .expect("download parallel gate");
    let serial_up_host = gpu.download_f32(&serial_up).expect("download serial up");
    let parallel_up_host = gpu.download_f32(&parallel_up).expect("download parallel up");
    let max_abs = serial_gate_host
        .iter()
        .zip(&parallel_gate_host)
        .chain(serial_up_host.iter().zip(&parallel_up_host))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let serial_median = median(&mut serial_ms);
    let parallel_median = median(&mut parallel_ms);
    println!("serial_ms={serial_median:.4}");
    println!("parallel_ms={parallel_median:.4}");
    println!("stream_overlap_speedup={:.4}x", serial_median / parallel_median);
    println!("max_abs={max_abs:.8e}");
    println!("serial_raw_ms={serial_ms:?}");
    println!("parallel_raw_ms={parallel_ms:?}");

    let run_combined = |gpu: &mut Gpu, stream: &mut Option<hip_bridge::Stream>| {
        gpu.active_stream = stream.take();
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &combined_weight,
            xq,
            &combined_output,
            combined_m,
            k,
            n,
        )
        .expect("combined gate/up launch");
        *stream = gpu.active_stream.take();
        gpu.hip
            .stream_synchronize(stream.as_ref().unwrap())
            .expect("sync combined gate/up");
    };
    for _ in 0..3 {
        run_serial(&mut gpu, &mut serial_stream);
        run_combined(&mut gpu, &mut serial_stream);
    }
    let mut split_pair_ms = Vec::with_capacity(pairs);
    let mut combined_pair_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let split_first = pair % 2 == 0;
        for split_mode in [split_first, !split_first] {
            let start = Instant::now();
            if split_mode {
                run_serial(&mut gpu, &mut serial_stream);
                split_pair_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            } else {
                run_combined(&mut gpu, &mut serial_stream);
                combined_pair_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
        }
    }
    let combined_host = gpu
        .download_f32(&combined_output)
        .expect("download combined gate/up");
    let mut combined_max_abs = 0.0f32;
    for row in 0..n {
        let combined_row = &combined_host[row * combined_m..(row + 1) * combined_m];
        let gate_row = &serial_gate_host[row * m..(row + 1) * m];
        let up_row = &serial_up_host[row * m..(row + 1) * m];
        for (reference, candidate) in gate_row.iter().zip(&combined_row[..m]) {
            combined_max_abs = combined_max_abs.max((reference - candidate).abs());
        }
        for (reference, candidate) in up_row.iter().zip(&combined_row[m..]) {
            combined_max_abs = combined_max_abs.max((reference - candidate).abs());
        }
    }
    let split_pair_median = median(&mut split_pair_ms);
    let combined_pair_median = median(&mut combined_pair_ms);
    println!("split_weight_pair_ms={split_pair_median:.4}");
    println!("combined_weight_pair_ms={combined_pair_median:.4}");
    println!(
        "combined_weight_speedup={:.4}x",
        split_pair_median / combined_pair_median
    );
    println!("combined_weight_max_abs={combined_max_abs:.8e}");

    let run_f16 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_f16_output(
            &gate_weight,
            xq,
            &f16_gate,
            m,
            k,
            n,
        )
        .expect("F16 gate launch");
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_f16_output(
            &up_weight,
            xq,
            &f16_up,
            m,
            k,
            n,
        )
        .expect("F16 up launch");
        gpu.hip.device_synchronize().expect("sync F16 pair");
    };
    for _ in 0..3 {
        run_serial(&mut gpu, &mut serial_stream);
        run_f16(&mut gpu);
    }
    let mut f32_pair_ms = Vec::with_capacity(pairs);
    let mut f16_pair_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let f32_first = pair % 2 == 0;
        for f32_mode in [f32_first, !f32_first] {
            let start = Instant::now();
            if f32_mode {
                run_serial(&mut gpu, &mut serial_stream);
                f32_pair_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            } else {
                run_f16(&mut gpu);
                f16_pair_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
        }
    }

    let mut f16_gate_raw = vec![0u8; f16_gate.byte_size()];
    let mut f16_up_raw = vec![0u8; f16_up.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut f16_gate_raw, &f16_gate.buf)
        .expect("download F16 gate");
    gpu.hip
        .memcpy_dtoh(&mut f16_up_raw, &f16_up.buf)
        .expect("download F16 up");
    let f16_values = f16_gate_raw
        .chunks_exact(2)
        .chain(f16_up_raw.chunks_exact(2))
        .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])));
    let f32_values = serial_gate_host.iter().chain(&serial_up_host).copied();
    let (f16_max_abs, f16_mean_abs, count) = f32_values.zip(f16_values).fold(
        (0.0f32, 0.0f64, 0usize),
        |(max_abs, sum_abs, count), (reference, candidate)| {
            let diff = (reference - candidate).abs();
            (max_abs.max(diff), sum_abs + diff as f64, count + 1)
        },
    );
    let f32_pair_median = median(&mut f32_pair_ms);
    let f16_pair_median = median(&mut f16_pair_ms);
    println!("f32_pair_ms={f32_pair_median:.4}");
    println!("f16_pair_ms={f16_pair_median:.4}");
    println!("f16_output_speedup={:.4}x", f32_pair_median / f16_pair_median);
    println!("f16_max_abs={f16_max_abs:.8e}");
    println!("f16_mean_abs={:.8e}", f16_mean_abs / count as f64);

    let run_group128_f16_fresh = |gpu: &mut Gpu| {
        let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("group128 quantize X");
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_f16_output(
            &gate_weight,
            xq,
            &f16_gate,
            m,
            k,
            n,
        )
        .expect("group128 F16 gate");
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_f16_output(
            &up_weight,
            xq,
            &f16_up,
            m,
            k,
            n,
        )
        .expect("group128 F16 up");
        gpu.hip.device_synchronize().expect("sync group128 F16 pair");
    };
    let run_group256_f16_fresh = |gpu: &mut Gpu| {
        let xq = gpu
            .ensure_q8_1_mmq_group256_x(&x, n, k)
            .expect("group256 quantize X");
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group256_serial_row_f16_output(
            &gate_weight,
            xq,
            &f16_gate,
            m,
            k,
            n,
        )
        .expect("group256 F16 gate");
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group256_serial_row_f16_output(
            &up_weight,
            xq,
            &f16_up,
            m,
            k,
            n,
        )
        .expect("group256 F16 up");
        gpu.hip.device_synchronize().expect("sync group256 F16 pair");
    };
    for _ in 0..3 {
        run_group128_f16_fresh(&mut gpu);
        run_group256_f16_fresh(&mut gpu);
    }
    let mut group128_f16_ms = Vec::with_capacity(pairs);
    let mut group256_f16_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let group128_first = pair % 2 == 0;
        for group128_mode in [group128_first, !group128_first] {
            let start = Instant::now();
            if group128_mode {
                run_group128_f16_fresh(&mut gpu);
                group128_f16_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            } else {
                run_group256_f16_fresh(&mut gpu);
                group256_f16_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
        }
    }
    run_group256_f16_fresh(&mut gpu);
    let mut group256_gate_raw = vec![0u8; f16_gate.byte_size()];
    let mut group256_up_raw = vec![0u8; f16_up.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut group256_gate_raw, &f16_gate.buf)
        .expect("download group256 F16 gate");
    gpu.hip
        .memcpy_dtoh(&mut group256_up_raw, &f16_up.buf)
        .expect("download group256 F16 up");
    let group128_values = f16_gate_raw
        .chunks_exact(2)
        .chain(f16_up_raw.chunks_exact(2))
        .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])));
    let group256_values = group256_gate_raw
        .chunks_exact(2)
        .chain(group256_up_raw.chunks_exact(2))
        .map(|bytes| f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])));
    let (group256_max_abs, group256_mean_abs, group256_count) = group128_values
        .zip(group256_values)
        .fold((0.0f32, 0.0f64, 0usize), |(max_abs, sum, count), (a, b)| {
            let diff = (a - b).abs();
            (max_abs.max(diff), sum + diff as f64, count + 1)
        });
    let group128_f16_median = median(&mut group128_f16_ms);
    let group256_f16_median = median(&mut group256_f16_ms);
    println!("group128_f16_quant_plus_pair_ms={group128_f16_median:.4}");
    println!("group256_f16_quant_plus_pair_ms={group256_f16_median:.4}");
    println!(
        "group256_f16_quant_plus_pair_speedup={:.4}x",
        group128_f16_median / group256_f16_median
    );
    println!("group256_f16_cross_path_max_abs={group256_max_abs:.8e}");
    println!(
        "group256_f16_cross_path_mean_abs={:.8e}",
        group256_mean_abs / group256_count as f64
    );

    let run_f32_ffn = |gpu: &mut Gpu, stream: &mut Option<hip_bridge::Stream>| {
        gpu.active_stream = stream.take();
        launch_projection(gpu, &gate_weight, xq, &serial_gate, m, k, n);
        launch_projection(gpu, &up_weight, xq, &serial_up, m, k, n);
        let hidden_q8 = gpu
            .fused_silu_mul_rotate_mq_q8_group128_batched(&serial_gate, &serial_up, m, n)
            .expect("F32 SwiGLU pack");
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &down_weight,
            hidden_q8,
            &f32_ffn_output,
            k,
            m,
            n,
        )
        .expect("F32 down projection");
        *stream = gpu.active_stream.take();
        gpu.hip
            .stream_synchronize(stream.as_ref().unwrap())
            .expect("sync F32 FFN");
    };
    let run_f16_ffn = |gpu: &mut Gpu, stream: &mut Option<hip_bridge::Stream>| {
        gpu.active_stream = stream.take();
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_f16_output(
            &gate_weight,
            xq,
            &f16_gate,
            m,
            k,
            n,
        )
        .expect("F16 FFN gate");
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_f16_output(
            &up_weight,
            xq,
            &f16_up,
            m,
            k,
            n,
        )
        .expect("F16 FFN up");
        let hidden_q8 = gpu
            .fused_silu_mul_rotate_mq_q8_group128_f16_batched(&f16_gate, &f16_up, m, n)
            .expect("F16 SwiGLU pack");
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &down_weight,
            hidden_q8,
            &f16_ffn_output,
            k,
            m,
            n,
        )
        .expect("F16 down projection");
        *stream = gpu.active_stream.take();
        gpu.hip
            .stream_synchronize(stream.as_ref().unwrap())
            .expect("sync F16 FFN");
    };
    for _ in 0..3 {
        run_f32_ffn(&mut gpu, &mut serial_stream);
        run_f16_ffn(&mut gpu, &mut serial_stream);
    }
    let mut f32_ffn_ms = Vec::with_capacity(pairs);
    let mut f16_ffn_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let f32_first = pair % 2 == 0;
        for f32_mode in [f32_first, !f32_first] {
            let start = Instant::now();
            if f32_mode {
                run_f32_ffn(&mut gpu, &mut serial_stream);
                f32_ffn_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            } else {
                run_f16_ffn(&mut gpu, &mut serial_stream);
                f16_ffn_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
        }
    }
    let f32_ffn_host = gpu
        .download_f32(&f32_ffn_output)
        .expect("download F32 FFN output");
    let f16_ffn_host = gpu
        .download_f32(&f16_ffn_output)
        .expect("download F16 FFN output");
    let (ffn_max_abs, ffn_mean_abs, ffn_count) = f32_ffn_host
        .iter()
        .zip(&f16_ffn_host)
        .fold((0.0f32, 0.0f64, 0usize), |(max_abs, sum, count), (a, b)| {
            let diff = (a - b).abs();
            (max_abs.max(diff), sum + diff as f64, count + 1)
        });
    let f32_ffn_median = median(&mut f32_ffn_ms);
    let f16_ffn_median = median(&mut f16_ffn_ms);
    println!("f32_ffn_ms={f32_ffn_median:.4}");
    println!("f16_ffn_ms={f16_ffn_median:.4}");
    println!("f16_ffn_speedup={:.4}x", f32_ffn_median / f16_ffn_median);
    println!("f16_ffn_max_abs={ffn_max_abs:.8e}");
    println!("f16_ffn_mean_abs={:.8e}", ffn_mean_abs / ffn_count as f64);
}
