// SPDX-License-Identifier: Apache-2.0
//! Admission benchmark for existing 3-bit gfx11 MQ4-v2 execution formats.
//!
//! This is a performance screen, not a quality comparison: the synthetic
//! HFQ3 and MQ4 matrices are independent. A candidate must beat the retained
//! packed-MQ4 primitive by at least 1.30x on both Qwen3.6-27B FFN shapes.

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

fn synth_mq4(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    let mut state = seed;
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.005 + ((row * 17 + group * 13) % 97) as f32 * 0.00005;
            let zero = ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for packed in &mut out[off + 8..off + 136] {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *packed = (state >> 32) as u8;
            }
        }
    }
    out
}

fn pack_3bit_group(qs: &[u8; 256]) -> [u8; 96] {
    let mut out = [0u8; 96];
    for lane in 0..32 {
        let mut packed = 0u32;
        for i in 0..8 {
            packed |= (u32::from(qs[lane * 8 + i]) & 7) << (3 * i);
        }
        out[lane * 3] = packed as u8;
        out[lane * 3 + 1] = (packed >> 8) as u8;
        out[lane * 3 + 2] = (packed >> 16) as u8;
    }
    out
}

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0 {
        return sign;
    }
    if exp >= 143 {
        return sign | 0x7c00;
    }
    if exp <= 112 {
        return sign;
    }
    sign | (((exp - 112) as u16) << 10) | ((mant >> 13) as u16)
}

fn synth_hfq3(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(m * groups * 104);
    for row in 0..m {
        for group in 0..groups {
            let scale = 0.005 + ((row * 17 + group * 13 + seed) % 97) as f32 * 0.00005;
            let zero = ((row * 7 + group * 11 + seed * 3) % 31) as f32 * 0.001 - 0.015;
            out.extend_from_slice(&scale.to_le_bytes());
            out.extend_from_slice(&zero.to_le_bytes());
            let mut q = [0u8; 256];
            for (i, value) in q.iter_mut().enumerate() {
                *value = ((row.wrapping_mul(31)
                    ^ group.wrapping_mul(53)
                    ^ i.wrapping_mul(7)
                    ^ seed.wrapping_mul(101))
                    & 7) as u8;
            }
            out.extend_from_slice(&pack_3bit_group(&q));
        }
    }
    assert_eq!(out.len(), m * groups * 104);
    out
}

fn synth_mq3_lloyd(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = Vec::with_capacity(m * groups * 112);
    for row in 0..m {
        for group in 0..groups {
            let base = ((row * 7 + group * 11 + seed * 31) % 19) as f32 * 0.013 - 0.1;
            for i in 0..8 {
                let value = base + (i as f32 - 3.5) * 0.025;
                out.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
            }
            let mut q = [0u8; 256];
            for (i, value) in q.iter_mut().enumerate() {
                *value = ((row.wrapping_mul(31)
                    ^ group.wrapping_mul(53)
                    ^ i.wrapping_mul(7)
                    ^ seed.wrapping_mul(101))
                    & 7) as u8;
            }
            out.extend_from_slice(&pack_3bit_group(&q));
        }
    }
    assert_eq!(out.len(), m * groups * 112);
    out
}

fn upload_weights(gpu: &mut Gpu, m: usize, k: usize, seed: usize) -> (GpuTensor, GpuTensor) {
    let mq4 = synth_mq4(m, k, seed as u64);
    let hfq3 = synth_hfq3(m, k, seed);
    let mq4_gpu = gpu.upload_raw(&mq4, &[mq4.len()]).expect("upload MQ4");
    let hfq3_gpu = gpu.upload_raw(&hfq3, &[hfq3.len()]).expect("upload HFQ3");
    (mq4_gpu, hfq3_gpu)
}

fn make_x(elements: usize, width: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|i| ((i * 17 + (i / width) * 31 + seed) % 101) as f32 * 0.01 - 0.5)
        .collect()
}

fn pre_rotate_mq3(gpu: &mut Gpu, x: &GpuTensor, rows: usize, width: usize) -> GpuTensor {
    let rotated = gpu
        .zeros(&[rows, width], DType::F32)
        .expect("allocate MQ3 rotated X");
    for row in 0..rows {
        let source = x.sub_offset(row * width, width);
        let destination = rotated.sub_offset(row * width, width);
        gpu.rotate_x_mq(&source, &destination, width)
            .expect("rotate MQ3 X");
    }
    gpu.hip.device_synchronize().expect("sync MQ3 rotation");
    rotated
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
    assert_eq!(gpu.arch, "gfx1100", "this benchmark is scoped to gfx1100");
    println!(
        "arch={} ffn={ffn} hidden={hidden} n={n} pairs={pairs}",
        gpu.arch
    );

    let (mq4_gate, hfq3_gate) = upload_weights(&mut gpu, ffn, hidden, 11);
    let (mq4_up, hfq3_up) = upload_weights(&mut gpu, ffn, hidden, 29);
    let x_host = make_x(n * hidden, hidden, 7);
    let x = gpu
        .upload_f32(&x_host, &[n, hidden])
        .expect("upload gate/up X");
    drop(x_host);
    let xq = gpu
        .ensure_q8_1_mmq_x(&x, n, hidden)
        .expect("quantize gate/up X");
    let mq4_gate_y = gpu.zeros(&[n, ffn], DType::F32).expect("MQ4 gate Y");
    let mq4_up_y = gpu.zeros(&[n, ffn], DType::F32).expect("MQ4 up Y");
    let hfq3_gate_y = gpu.zeros(&[n, ffn], DType::F32).expect("HFQ3 gate Y");
    let hfq3_up_y = gpu.zeros(&[n, ffn], DType::F32).expect("HFQ3 up Y");

    let (gate_mq4_raw, gate_hfq3_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &mq4_gate,
                xq,
                &mq4_gate_y,
                ffn,
                hidden,
                n,
                false,
            )
            .expect("MQ4 gate");
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &mq4_up, xq, &mq4_up_y, ffn, hidden, n, false,
            )
            .expect("MQ4 up");
            gpu.hip.device_synchronize().expect("sync MQ4 gate/up");
        },
        |gpu| {
            gpu.gemm_gate_up_hfq3g256_wmma(
                &hfq3_gate,
                &hfq3_up,
                &x,
                &hfq3_gate_y,
                &hfq3_up_y,
                ffn,
                ffn,
                hidden,
                n,
            )
            .expect("HFQ3 gate/up");
            gpu.hip.device_synchronize().expect("sync HFQ3 gate/up");
        },
    );
    let gate_mq4 = median(&gate_mq4_raw);
    let gate_hfq3 = median(&gate_hfq3_raw);
    let gate_speedup = gate_mq4 / gate_hfq3;
    println!("gate_up_mq4_ms={gate_mq4:.4}");
    println!("gate_up_hfq3_ms={gate_hfq3:.4}");
    println!("gate_up_hfq3_speedup={gate_speedup:.4}x");
    println!("gate_up_mq4_raw_ms={gate_mq4_raw:?}");
    println!("gate_up_hfq3_raw_ms={gate_hfq3_raw:?}");

    let (mq4_down, hfq3_down) = upload_weights(&mut gpu, hidden, ffn, 47);
    let down_x_host = make_x(n * ffn, ffn, 13);
    let down_x = gpu
        .upload_f32(&down_x_host, &[n, ffn])
        .expect("upload down X");
    drop(down_x_host);
    let down_xq = gpu
        .ensure_q8_1_mmq_x(&down_x, n, ffn)
        .expect("quantize down X");
    let mq4_down_y = gpu.zeros(&[n, hidden], DType::F32).expect("MQ4 down Y");
    let hfq3_down_y = gpu.zeros(&[n, hidden], DType::F32).expect("HFQ3 down Y");

    let (down_mq4_raw, down_hfq3_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &mq4_down,
                down_xq,
                &mq4_down_y,
                hidden,
                ffn,
                n,
                true,
            )
            .expect("MQ4 down");
            gpu.hip.device_synchronize().expect("sync MQ4 down");
        },
        |gpu| {
            gpu.gemm_hfq3g256_residual_wmma(&hfq3_down, &down_x, &hfq3_down_y, hidden, ffn, n)
                .expect("HFQ3 down");
            gpu.hip.device_synchronize().expect("sync HFQ3 down");
        },
    );
    let down_mq4 = median(&down_mq4_raw);
    let down_hfq3 = median(&down_hfq3_raw);
    let down_speedup = down_mq4 / down_hfq3;
    println!("down_mq4_ms={down_mq4:.4}");
    println!("down_hfq3_ms={down_hfq3:.4}");
    println!("down_hfq3_speedup={down_speedup:.4}x");
    println!("down_mq4_raw_ms={down_mq4_raw:?}");
    println!("down_hfq3_raw_ms={down_hfq3_raw:?}");

    let lloyd_gate_host = synth_mq3_lloyd(ffn, hidden, 61);
    let lloyd_up_host = synth_mq3_lloyd(ffn, hidden, 79);
    let lloyd_gate = gpu
        .upload_raw(&lloyd_gate_host, &[lloyd_gate_host.len()])
        .expect("upload Lloyd gate");
    let lloyd_up = gpu
        .upload_raw(&lloyd_up_host, &[lloyd_up_host.len()])
        .expect("upload Lloyd up");
    drop((lloyd_gate_host, lloyd_up_host));
    let lloyd_gate_y = gpu.zeros(&[n, ffn], DType::F32).expect("Lloyd gate Y");
    let lloyd_up_y = gpu.zeros(&[n, ffn], DType::F32).expect("Lloyd up Y");
    // MQ3-Lloyd checkpoints require FWHT-rotated activations. Materializing
    // rotation before timing makes this an optimistic GEMM-core screen.
    let lloyd_x = pre_rotate_mq3(&mut gpu, &x, n, hidden);
    let (gate_mq4_lloyd_raw, gate_lloyd_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &mq4_gate,
                xq,
                &mq4_gate_y,
                ffn,
                hidden,
                n,
                false,
            )
            .expect("MQ4 gate");
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &mq4_up, xq, &mq4_up_y, ffn, hidden, n, false,
            )
            .expect("MQ4 up");
            gpu.hip.device_synchronize().expect("sync MQ4 gate/up");
        },
        |gpu| {
            gpu.gemm_gate_up_mq3g256_lloyd_wmma(
                &lloyd_gate,
                &lloyd_up,
                &lloyd_x,
                &lloyd_gate_y,
                &lloyd_up_y,
                ffn,
                ffn,
                hidden,
                n,
            )
            .expect("Lloyd gate/up");
            gpu.hip.device_synchronize().expect("sync Lloyd gate/up");
        },
    );
    let gate_mq4_lloyd = median(&gate_mq4_lloyd_raw);
    let gate_lloyd = median(&gate_lloyd_raw);
    let gate_lloyd_speedup = gate_mq4_lloyd / gate_lloyd;
    println!("gate_up_mq4_lloyd_control_ms={gate_mq4_lloyd:.4}");
    println!("gate_up_mq3_lloyd_ms={gate_lloyd:.4}");
    println!("gate_up_mq3_lloyd_speedup={gate_lloyd_speedup:.4}x");
    println!("gate_up_mq4_lloyd_control_raw_ms={gate_mq4_lloyd_raw:?}");
    println!("gate_up_mq3_lloyd_raw_ms={gate_lloyd_raw:?}");

    let lloyd_down_host = synth_mq3_lloyd(hidden, ffn, 97);
    let lloyd_down = gpu
        .upload_raw(&lloyd_down_host, &[lloyd_down_host.len()])
        .expect("upload Lloyd down");
    drop(lloyd_down_host);
    let lloyd_down_y = gpu.zeros(&[n, hidden], DType::F32).expect("Lloyd down Y");
    let lloyd_down_x = pre_rotate_mq3(&mut gpu, &down_x, n, ffn);
    let (down_mq4_lloyd_raw, down_lloyd_raw) = paired(
        &mut gpu,
        pairs,
        |gpu| {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &mq4_down,
                down_xq,
                &mq4_down_y,
                hidden,
                ffn,
                n,
                true,
            )
            .expect("MQ4 down");
            gpu.hip.device_synchronize().expect("sync MQ4 down");
        },
        |gpu| {
            gpu.gemm_mq3g256_lloyd_residual_wmma(
                &lloyd_down,
                &lloyd_down_x,
                &lloyd_down_y,
                hidden,
                ffn,
                n,
            )
            .expect("Lloyd down");
            gpu.hip.device_synchronize().expect("sync Lloyd down");
        },
    );
    let down_mq4_lloyd = median(&down_mq4_lloyd_raw);
    let down_lloyd = median(&down_lloyd_raw);
    let down_lloyd_speedup = down_mq4_lloyd / down_lloyd;
    println!("down_mq4_lloyd_control_ms={down_mq4_lloyd:.4}");
    println!("down_mq3_lloyd_ms={down_lloyd:.4}");
    println!("down_mq3_lloyd_speedup={down_lloyd_speedup:.4}x");
    println!("down_mq4_lloyd_control_raw_ms={down_mq4_lloyd_raw:?}");
    println!("down_mq3_lloyd_raw_ms={down_lloyd_raw:?}");

    println!("mq4_resident_bytes={}", 3 * ffn * hidden * 136 / 256);
    println!("hfq3_resident_bytes={}", 3 * ffn * hidden * 104 / 256);
    println!("mq3_lloyd_resident_bytes={}", 3 * ffn * hidden * 112 / 256);
    let admitted = gate_speedup >= 1.30 && down_speedup >= 1.30;
    let lloyd_admitted = gate_lloyd_speedup >= 1.30 && down_lloyd_speedup >= 1.30;
    println!(
        "hfq3_admission={}",
        if admitted { "PASS" } else { "REJECT" }
    );
    println!(
        "mq3_lloyd_admission={}",
        if lloyd_admitted { "PASS" } else { "REJECT" }
    );
}
