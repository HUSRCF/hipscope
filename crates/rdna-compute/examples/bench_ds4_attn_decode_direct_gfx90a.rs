// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compare DeepSeek V4 decode gathered top-K attention with the existing
//! direct-cache B=1 kernel on the production gfx90a shape.

use rdna_compute::{DType, Gpu};

const H: usize = 64;
const D: usize = 512;
const SWA: usize = 128;
const TOPK: usize = 512;
const N_COMPRESSED: usize = 2_705;

fn sample(index: usize, salt: u32, scale: f32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ((x & 0xffff) as f32 / 32_767.5 - 1.0) * scale
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this benchmark targets MI250");
    let active_topk = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>().expect("active top-K"))
        .unwrap_or(TOPK);
    assert!(active_topk <= TOPK, "active top-K exceeds storage stride");

    let q = (0..H * D).map(|i| sample(i, 11, 0.125)).collect::<Vec<_>>();
    let swa_k = (0..D * SWA)
        .map(|i| sample(i, 23, 0.125))
        .collect::<Vec<_>>();
    let swa_v = (0..D * SWA)
        .map(|i| sample(i, 37, 0.125))
        .collect::<Vec<_>>();
    let kv = (0..N_COMPRESSED * D)
        .map(|i| sample(i, 41, 0.125))
        .collect::<Vec<_>>();
    let sink = (0..H).map(|i| sample(i, 53, 0.5)).collect::<Vec<_>>();
    let indices = (0..TOPK)
        .map(|i| ((i * 104_729 + 17) % N_COMPRESSED) as i32)
        .collect::<Vec<_>>();

    let d_q = gpu.upload_f32(&q, &[H, D]).unwrap();
    let d_swa_k = gpu.upload_f32(&swa_k, &[D, SWA]).unwrap();
    let d_swa_v = gpu.upload_f32(&swa_v, &[D, SWA]).unwrap();
    let d_kv = gpu.upload_f32(&kv, &[N_COMPRESSED, D]).unwrap();
    let d_sink = gpu.upload_f32(&sink, &[H]).unwrap();
    let d_indices = gpu.upload_raw(&i32_bytes(&indices), &[TOPK * 4]).unwrap();
    let d_n_swa = gpu.upload_raw(&i32_bytes(&[SWA as i32]), &[4]).unwrap();
    let d_n_topk = gpu
        .upload_raw(&i32_bytes(&[active_topk as i32]), &[4])
        .unwrap();
    let d_n_compressed = gpu
        .upload_raw(&i32_bytes(&[N_COMPRESSED as i32]), &[4])
        .unwrap();
    let d_gathered = gpu.zeros(&[D, TOPK], DType::F32).unwrap();
    let d_gathered_out = gpu.zeros(&[H, D], DType::F32).unwrap();
    let d_direct_out = gpu.zeros(&[H, D], DType::F32).unwrap();

    let run_gathered = |gpu: &mut Gpu| {
        gpu.deepseek4_topk_kv_gather_f32_buf(
            &d_kv,
            &d_indices,
            &d_gathered,
            &d_n_topk,
            &d_n_compressed,
            TOPK as i32,
            D as i32,
            TOPK as i32,
            0,
            1.0,
        )
        .unwrap();
        gpu.deepseek4_attn_swa_topk_f32_buf(
            &d_q,
            &d_swa_k,
            &d_swa_v,
            &d_gathered,
            &d_gathered,
            &d_sink,
            &d_gathered_out,
            &d_n_swa,
            &d_n_topk,
            H as i32,
            D as i32,
            SWA as i32,
            TOPK as i32,
        )
        .unwrap();
    };
    let run_direct = |gpu: &mut Gpu| {
        gpu.deepseek4_attn_swa_topk_direct_batched_f32(
            &d_q,
            &d_swa_k,
            &d_swa_v,
            &d_kv,
            &d_indices,
            &d_sink,
            &d_n_swa,
            &d_n_topk,
            &d_direct_out,
            H as i32,
            D as i32,
            SWA as i32,
            TOPK as i32,
            N_COMPRESSED as i32,
            1,
        )
        .unwrap();
    };

    run_gathered(&mut gpu);
    run_direct(&mut gpu);
    gpu.hip.device_synchronize().unwrap();
    let gathered = gpu.download_f32(&d_gathered_out).unwrap();
    let direct = gpu.download_f32(&d_direct_out).unwrap();
    let bit_mismatches = gathered
        .iter()
        .zip(&direct)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = gathered
        .iter()
        .zip(&direct)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "parity: active_topk={active_topk} bit_mismatches={bit_mismatches} max_abs={max_abs:.9e}"
    );
    assert_eq!(bit_mismatches, 0, "direct B=1 attention must be bit-exact");

    for _ in 0..10 {
        run_gathered(&mut gpu);
        run_direct(&mut gpu);
    }
    gpu.hip.device_synchronize().unwrap();

    let iterations = 100;
    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..iterations {
        run_gathered(&mut gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let gathered_us =
        gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1_000.0 / iterations as f64;

    let start = gpu.hip.event_create().unwrap();
    let stop = gpu.hip.event_create().unwrap();
    gpu.hip.event_record(&start, None).unwrap();
    for _ in 0..iterations {
        run_direct(&mut gpu);
    }
    gpu.hip.event_record(&stop, None).unwrap();
    gpu.hip.event_synchronize(&stop).unwrap();
    let direct_us =
        gpu.hip.event_elapsed_ms(&start, &stop).unwrap() as f64 * 1_000.0 / iterations as f64;

    println!(
        "timing: active_topk={active_topk} gathered={gathered_us:.3} us direct={direct_us:.3} us speedup={:.3}x",
        gathered_us / direct_us
    );
}
