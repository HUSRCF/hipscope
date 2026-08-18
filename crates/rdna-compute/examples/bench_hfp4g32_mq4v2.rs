// SPDX-License-Identifier: Apache-2.0
//! Admission benchmark for a gfx11-native FP4 execution contract.
//!
//! Compares the retained packed-MQ4/group128 production path with the existing
//! HFP4G32 wave32-WMMA path at Qwen3.6-27B's large FFN projection shapes. This
//! is deliberately standalone: it does not alter runtime routing or serving.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const MQ4_GROUP: usize = 256;
const MQ4_GROUP_BYTES: usize = 136;

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

fn synth_mq4(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % MQ4_GROUP, 0);
    let groups = k / MQ4_GROUP;
    let mut out = vec![0u8; m * groups * MQ4_GROUP_BYTES];
    let mut state = seed;
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * MQ4_GROUP_BYTES;
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

fn synth_hfp4(m: usize, k: usize, seed: u64) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let blocks = k / 32;
    let row_bytes = 16 + blocks * 17;
    let mut out = vec![0u8; m * row_bytes];
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for row in 0..m {
        let row_off = row * row_bytes;
        let row_scale = 0.02 + (next() & 0xff) as f32 * 1e-4;
        out[row_off..row_off + 2].copy_from_slice(&f32_to_f16_bits(row_scale).to_le_bytes());
        out[row_off + 4..row_off + 6].copy_from_slice(&(blocks as u16).to_le_bytes());
        for block in 0..blocks {
            let off = row_off + 16 + block * 17;
            out[off] = 120 + (next() & 7) as u8;
            for packed in &mut out[off + 1..off + 17] {
                *packed = (next() & 0xff) as u8;
            }
        }
    }
    out
}

fn make_x(elements: usize, row_width: usize, seed: usize) -> Vec<f32> {
    (0..elements)
        .map(|i| ((i * 17 + (i / row_width) * 31 + seed) % 101) as f32 * 0.01 - 0.5)
        .collect()
}

fn upload_weights(gpu: &mut Gpu, m: usize, k: usize, seed: u64) -> (GpuTensor, GpuTensor) {
    let mq4 = synth_mq4(m, k, seed);
    let hfp4 = synth_hfp4(m, k, seed ^ 0x9e37_79b9_7f4a_7c15);
    let mq4_gpu = gpu.upload_raw(&mq4, &[mq4.len()]).expect("upload MQ4");
    let hfp4_gpu = gpu.upload_raw(&hfp4, &[hfp4.len()]).expect("upload HFP4");
    (mq4_gpu, hfp4_gpu)
}

fn run_gate_up(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    pairs: usize,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    let (mq4_gate, hfp4_gate) = upload_weights(gpu, m, k, 0x1234_5678);
    let (mq4_up, hfp4_up) = upload_weights(gpu, m, k, 0x9abc_def0);
    let x_host = make_x(n * k, k, 11);
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload gate/up X");
    drop(x_host);
    let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize gate/up X");
    let mq4_gate_y = gpu.zeros(&[n, m], DType::F32).expect("MQ4 gate Y");
    let mq4_up_y = gpu.zeros(&[n, m], DType::F32).expect("MQ4 up Y");
    let hfp4_gate_y = gpu.zeros(&[n, m], DType::F32).expect("HFP4 gate Y");
    let hfp4_up_y = gpu.zeros(&[n, m], DType::F32).expect("HFP4 up Y");

    let run_mq4 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &mq4_gate,
            xq,
            &mq4_gate_y,
            m,
            k,
            n,
        )
        .expect("MQ4 gate");
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(&mq4_up, xq, &mq4_up_y, m, k, n)
            .expect("MQ4 up");
        gpu.hip.device_synchronize().expect("sync MQ4 gate/up");
    };
    let run_hfp4 = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfp4g32(
            &hfp4_gate,
            &hfp4_up,
            &x,
            &hfp4_gate_y,
            &hfp4_up_y,
            m,
            m,
            k,
            n,
        )
        .expect("HFP4 gate/up");
        gpu.hip.device_synchronize().expect("sync HFP4 gate/up");
    };

    for _ in 0..3 {
        run_mq4(gpu);
        run_hfp4(gpu);
    }
    gpu.dpm_warmup(5.0).expect("DPM warmup gate/up");

    let mut mq4_ms = Vec::with_capacity(pairs);
    let mut hfp4_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for mq4_first in [pair % 2 == 0, pair % 2 != 0] {
            let start = Instant::now();
            if mq4_first {
                run_mq4(gpu);
                mq4_ms.push(start.elapsed().as_secs_f64() * 1e3);
            } else {
                run_hfp4(gpu);
                hfp4_ms.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    (median(&mq4_ms), median(&hfp4_ms), mq4_ms, hfp4_ms)
}

fn run_down(
    gpu: &mut Gpu,
    m: usize,
    k: usize,
    n: usize,
    pairs: usize,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    let (mq4, hfp4) = upload_weights(gpu, m, k, 0x0ddc_0ffe);
    let x_host = make_x(n * k, k, 29);
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload down X");
    drop(x_host);
    let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize down X");
    let mq4_y = gpu.zeros(&[n, m], DType::F32).expect("MQ4 down Y");
    let hfp4_y = gpu.zeros(&[n, m], DType::F32).expect("HFP4 down Y");

    let run_mq4 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_add_prequant_x256y64_perm_group128(&mq4, xq, &mq4_y, m, k, n)
            .expect("MQ4 down");
        gpu.hip.device_synchronize().expect("sync MQ4 down");
    };
    let run_hfp4 = |gpu: &mut Gpu| {
        gpu.gemm_hfp4g32_residual(&hfp4, &x, &hfp4_y, m, k, n)
            .expect("HFP4 down");
        gpu.hip.device_synchronize().expect("sync HFP4 down");
    };

    for _ in 0..3 {
        run_mq4(gpu);
        run_hfp4(gpu);
    }
    gpu.dpm_warmup(5.0).expect("DPM warmup down");

    let mut mq4_ms = Vec::with_capacity(pairs);
    let mut hfp4_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for mq4_first in [pair % 2 == 0, pair % 2 != 0] {
            let start = Instant::now();
            if mq4_first {
                run_mq4(gpu);
                mq4_ms.push(start.elapsed().as_secs_f64() * 1e3);
            } else {
                run_hfp4(gpu);
                hfp4_ms.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    (median(&mq4_ms), median(&hfp4_ms), mq4_ms, hfp4_ms)
}

fn main() {
    let ffn_m = parse_arg("--ffn-m", 17_408);
    let hidden = parse_arg("--hidden", 5_120);
    let n = parse_arg("--n", 2_048);
    let pairs = parse_arg("--pairs", 7);
    assert_eq!(ffn_m % 256, 0);
    assert_eq!(hidden % 256, 0);
    assert_eq!(n % 256, 0);

    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx1100", "this benchmark is scoped to gfx1100");
    println!(
        "arch={} ffn_m={ffn_m} hidden={hidden} n={n} pairs={pairs}",
        gpu.arch
    );

    let (mq4_gate_up, hfp4_gate_up, mq4_gate_up_raw, hfp4_gate_up_raw) =
        run_gate_up(&mut gpu, ffn_m, hidden, n, pairs);
    println!("gate_up_mq4_ms={mq4_gate_up:.4}");
    println!("gate_up_hfp4_ms={hfp4_gate_up:.4}");
    println!("gate_up_hfp4_speedup={:.4}x", mq4_gate_up / hfp4_gate_up);
    println!("gate_up_mq4_raw_ms={mq4_gate_up_raw:?}");
    println!("gate_up_hfp4_raw_ms={hfp4_gate_up_raw:?}");

    let (mq4_down, hfp4_down, mq4_down_raw, hfp4_down_raw) =
        run_down(&mut gpu, hidden, ffn_m, n, pairs);
    println!("down_mq4_ms={mq4_down:.4}");
    println!("down_hfp4_ms={hfp4_down:.4}");
    println!("down_hfp4_speedup={:.4}x", mq4_down / hfp4_down);
    println!("down_mq4_raw_ms={mq4_down_raw:?}");
    println!("down_hfp4_raw_ms={hfp4_down_raw:?}");

    let admitted = mq4_gate_up / hfp4_gate_up >= 1.30 && mq4_down / hfp4_down >= 1.30;
    println!(
        "mq4_v2_admission={}",
        if admitted { "PASS" } else { "REJECT" }
    );
}
