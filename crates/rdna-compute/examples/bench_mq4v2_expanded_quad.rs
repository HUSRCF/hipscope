// SPDX-License-Identifier: Apache-2.0
//! Full-shape upper bound for offline-expanded IU8 MQ4 weights on gfx1100.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn parse_arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].parse().unwrap_or_else(|_| panic!("invalid {name}")))
        .unwrap_or(default)
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

fn expand_i8(src: &[u8], m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 272];
    for row in 0..m {
        for group in 0..groups {
            let src_off = (row * groups + group) * 136;
            let dst_off = (row * groups + group) * 272;
            out[dst_off..dst_off + 8].copy_from_slice(&src[src_off..src_off + 8]);
            for byte in 0..128 {
                let packed = src[src_off + 8 + byte];
                out[dst_off + 16 + 2 * byte] = packed & 0x0f;
                out[dst_off + 16 + 2 * byte + 1] = packed >> 4;
            }
        }
    }
    out
}

fn upload_pair(gpu: &mut Gpu, m: usize, k: usize, seed: usize) -> (GpuTensor, GpuTensor) {
    let packed = synth_mq4(m, k, seed);
    let expanded = expand_i8(&packed, m, k);
    let packed_gpu = gpu
        .upload_raw(&packed, &[packed.len()])
        .expect("upload packed");
    let expanded_gpu = gpu
        .upload_raw(&expanded, &[expanded.len()])
        .expect("upload expanded");
    (packed_gpu, expanded_gpu)
}

fn make_x(elements: usize, width: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|i| ((i * 17 + (i / width) * 31 + seed) % 101) as f32 * 0.01 - 0.5)
        .collect()
}

fn max_abs(gpu: &mut Gpu, reference: &GpuTensor, candidate: &GpuTensor) -> f32 {
    let reference = gpu.download_f32(reference).expect("download reference");
    let candidate = gpu.download_f32(candidate).expect("download candidate");
    reference
        .iter()
        .zip(candidate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
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

fn main() {
    let ffn = parse_arg("--ffn", 17_408);
    let hidden = parse_arg("--hidden", 5_120);
    let n = parse_arg("--n", 2_048);
    let pairs = parse_arg("--pairs", 7);
    assert_eq!(ffn % 256, 0);
    assert_eq!(hidden % 256, 0);
    assert_eq!(n % 256, 0);

    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1100");
    println!(
        "arch={} ffn={ffn} hidden={hidden} n={n} pairs={pairs}",
        gpu.arch
    );

    let (gate, gate_i8) = upload_pair(&mut gpu, ffn, hidden, 11);
    let (up, up_i8) = upload_pair(&mut gpu, ffn, hidden, 29);
    let x_host = make_x(n * hidden, hidden, 7);
    let x = gpu
        .upload_f32(&x_host, &[n, hidden])
        .expect("upload gate/up X");
    drop(x_host);
    let xq = gpu
        .ensure_q8_1_mmq_x(&x, n, hidden)
        .expect("quantize gate/up X");
    let gate_ref = gpu.zeros(&[n, ffn], DType::F32).expect("gate ref");
    let up_ref = gpu.zeros(&[n, ffn], DType::F32).expect("up ref");
    let gate_i8_out = gpu.zeros(&[n, ffn], DType::F32).expect("gate i8");
    let up_i8_out = gpu.zeros(&[n, ffn], DType::F32).expect("up i8");
    let gate_skip_scale = gpu.zeros(&[n, ffn], DType::F32).expect("gate skip scale");
    let up_skip_scale = gpu.zeros(&[n, ffn], DType::F32).expect("up skip scale");

    let (gate_up_ref_raw, gate_up_i8_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &gate, xq, &gate_ref, ffn, hidden, n, false,
            )
            .expect("gate ref");
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &up, xq, &up_ref, ffn, hidden, n, false,
            )
            .expect("up ref");
            gpu.hip.device_synchronize().expect("sync gate/up ref");
        },
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_expanded_i8_quad_row(
                &gate_i8,
                xq,
                &gate_i8_out,
                ffn,
                hidden,
                n,
                false,
            )
            .expect("gate expanded");
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_expanded_i8_quad_row(
                &up_i8, xq, &up_i8_out, ffn, hidden, n, false,
            )
            .expect("up expanded");
            gpu.hip.device_synchronize().expect("sync gate/up expanded");
        },
    );
    let gate_up_ref = median(&gate_up_ref_raw);
    let gate_up_i8 = median(&gate_up_i8_raw);
    let gate_diff = max_abs(&mut gpu, &gate_ref, &gate_i8_out);
    let up_diff = max_abs(&mut gpu, &up_ref, &up_i8_out);
    println!("gate_up_reference_ms={gate_up_ref:.4}");
    println!("gate_up_expanded_i8_quad_ms={gate_up_i8:.4}");
    println!("gate_up_speedup={:.4}x", gate_up_ref / gate_up_i8);
    println!("gate_up_max_abs={:.8e}", gate_diff.max(up_diff));
    println!("gate_up_reference_raw_ms={gate_up_ref_raw:?}");
    println!("gate_up_expanded_raw_ms={gate_up_i8_raw:?}");

    let (gate_up_scale_ref_raw, gate_up_skip_scale_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &gate, xq, &gate_ref, ffn, hidden, n, false,
            )
            .expect("gate scale ref");
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &up, xq, &up_ref, ffn, hidden, n, false,
            )
            .expect("up scale ref");
            gpu.hip
                .device_synchronize()
                .expect("sync gate/up scale ref");
        },
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_quad_row_skip_scale(
                &gate,
                xq,
                &gate_skip_scale,
                ffn,
                hidden,
                n,
                false,
            )
            .expect("gate skip scale");
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_quad_row_skip_scale(
                &up,
                xq,
                &up_skip_scale,
                ffn,
                hidden,
                n,
                false,
            )
            .expect("up skip scale");
            gpu.hip
                .device_synchronize()
                .expect("sync gate/up skip scale");
        },
    );
    let gate_up_scale_ref = median(&gate_up_scale_ref_raw);
    let gate_up_skip_scale = median(&gate_up_skip_scale_raw);
    println!("gate_up_scale_reference_ms={gate_up_scale_ref:.4}");
    println!("gate_up_skip_scale_ms={gate_up_skip_scale:.4}");
    println!(
        "gate_up_skip_scale_upper_bound={:.4}x",
        gate_up_scale_ref / gate_up_skip_scale
    );

    let (down, down_i8) = upload_pair(&mut gpu, hidden, ffn, 47);
    let down_x_host = make_x(n * ffn, ffn, 13);
    let down_x = gpu
        .upload_f32(&down_x_host, &[n, ffn])
        .expect("upload down X");
    drop(down_x_host);
    let down_xq = gpu
        .ensure_q8_1_mmq_x(&down_x, n, ffn)
        .expect("quantize down X");
    let down_ref = gpu.zeros(&[n, hidden], DType::F32).expect("down ref");
    let down_i8_out = gpu.zeros(&[n, hidden], DType::F32).expect("down i8");
    let down_skip_scale = gpu
        .zeros(&[n, hidden], DType::F32)
        .expect("down skip scale");
    let (down_ref_raw, down_i8_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &down, down_xq, &down_ref, hidden, ffn, n, true,
            )
            .expect("down ref");
            gpu.hip.device_synchronize().expect("sync down ref");
        },
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_expanded_i8_quad_row(
                &down_i8,
                down_xq,
                &down_i8_out,
                hidden,
                ffn,
                n,
                true,
            )
            .expect("down expanded");
            gpu.hip.device_synchronize().expect("sync down expanded");
        },
    );
    let down_ref_ms = median(&down_ref_raw);
    let down_i8_ms = median(&down_i8_raw);
    let down_diff = max_abs(&mut gpu, &down_ref, &down_i8_out);
    println!("down_reference_ms={down_ref_ms:.4}");
    println!("down_expanded_i8_quad_ms={down_i8_ms:.4}");
    println!("down_speedup={:.4}x", down_ref_ms / down_i8_ms);
    println!("down_max_abs={down_diff:.8e}");
    println!("down_reference_raw_ms={down_ref_raw:?}");
    println!("down_expanded_raw_ms={down_i8_raw:?}");

    let (down_scale_ref_raw, down_skip_scale_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &down, down_xq, &down_ref, hidden, ffn, n, true,
            )
            .expect("down scale ref");
            gpu.hip.device_synchronize().expect("sync down scale ref");
        },
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_quad_row_skip_scale(
                &down,
                down_xq,
                &down_skip_scale,
                hidden,
                ffn,
                n,
                true,
            )
            .expect("down skip scale");
            gpu.hip.device_synchronize().expect("sync down skip scale");
        },
    );
    let down_scale_ref = median(&down_scale_ref_raw);
    let down_skip_scale = median(&down_skip_scale_raw);
    println!("down_scale_reference_ms={down_scale_ref:.4}");
    println!("down_skip_scale_ms={down_skip_scale:.4}");
    println!(
        "down_skip_scale_upper_bound={:.4}x",
        down_scale_ref / down_skip_scale
    );

    let admitted = gate_up_ref / gate_up_i8 >= 1.30 && down_ref_ms / down_i8_ms >= 1.30;
    println!(
        "expanded_i8_quad_admission={}",
        if admitted { "PASS" } else { "REJECT" }
    );
}
