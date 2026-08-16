// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx90a F16 decode-projection shape benchmark.
//!
//! Compares the portable one-row `gemm_f16_tiled` baseline with the native
//! wave64 two-row DeepSeek V4 kernel at G=1, B=1.  The candidate intentionally
//! preserves the baseline accumulation tree, so every shape is checked with a
//! byte-for-byte F32 output comparison before it is timed.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const G: i32 = 1;
const BATCH: i32 = 1;
const WARMUP: usize = 20;
const SAMPLES: usize = 5;
const TARGET_BYTES_PER_SAMPLE: usize = 8 * 1024 * 1024 * 1024;
const MIN_ITERS: usize = 32;
const MAX_ITERS: usize = 400;

#[derive(Clone, Copy)]
struct Shape {
    label: &'static str,
    m: usize,
    k: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "q_lora_a",
        m: 1024,
        k: 4096,
    },
    Shape {
        label: "indexer_q",
        m: 8192,
        k: 1024,
    },
    Shape {
        label: "proj_4096x8192",
        m: 4096,
        k: 8192,
    },
    Shape {
        label: "proj_2048x4096",
        m: 2048,
        k: 4096,
    },
    Shape {
        label: "proj_4096x2048",
        m: 4096,
        k: 2048,
    },
    Shape {
        label: "compressor_index",
        m: 256,
        k: 4096,
    },
    Shape {
        label: "vocab_shard",
        m: 32320,
        k: 4096,
    },
    Shape {
        label: "odd_m",
        m: 1025,
        k: 4096,
    },
    Shape {
        label: "k_tail_7",
        m: 1024,
        k: 4103,
    },
];

fn deterministic_f16_bits(index: usize) -> u16 {
    let mut x = (index as u32).wrapping_add(17).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    let sign = ((x >> 16) & 0x8000) as u16;
    sign | 0x3000 | (((x >> 5) as u16) & 0x03ff)
}

fn deterministic_x(index: usize) -> f32 {
    let mut x = (index as u32).wrapping_add(29).wrapping_mul(0x85eb_ca6b);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ((x & 0xffff) as f32 / 32_767.5 - 1.0) * 0.25
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn iterations(m: usize, k: usize) -> usize {
    let bytes_per_call = m
        .checked_mul(k)
        .and_then(|v| v.checked_mul(std::mem::size_of::<u16>()))
        .expect("shape byte count overflow");
    (TARGET_BYTES_PER_SAMPLE / bytes_per_call.max(1)).clamp(MIN_ITERS, MAX_ITERS)
}

fn launch_tiled(gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) {
    gpu.gemm_f16_tiled(w, x, y, m, k, 1)
        .expect("gemm_f16_tiled launch");
}

fn launch_wave64(gpu: &mut Gpu, w: &GpuTensor, x: &GpuTensor, y: &GpuTensor, m: usize, k: usize) {
    gpu.wo_per_group_batched_f16_wave64_row2_gfx90a(w, x, y, G, m as i32, k as i32, BATCH)
        .expect("wave64 row2 launch");
}

fn time_tiled(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    iters: usize,
) -> f64 {
    let started = Instant::now();
    for _ in 0..iters {
        launch_tiled(gpu, w, x, y, m, k);
    }
    gpu.hip.device_synchronize().expect("tiled timing sync");
    started.elapsed().as_secs_f64() * 1.0e6 / iters as f64
}

fn time_wave64(
    gpu: &mut Gpu,
    w: &GpuTensor,
    x: &GpuTensor,
    y: &GpuTensor,
    m: usize,
    k: usize,
    iters: usize,
) -> f64 {
    let started = Instant::now();
    for _ in 0..iters {
        launch_wave64(gpu, w, x, y, m, k);
    }
    gpu.hip.device_synchronize().expect("wave64 timing sync");
    started.elapsed().as_secs_f64() * 1.0e6 / iters as f64
}

fn assert_memcmp(reference: &[f32], candidate: &[f32], shape: Shape) {
    assert_eq!(reference.len(), candidate.len());
    let mismatch = reference
        .iter()
        .zip(candidate)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    if let Some(i) = mismatch {
        panic!(
            "{} M={} K={} memcmp failed at row {}: ref={:#010x} wave64={:#010x}",
            shape.label,
            shape.m,
            shape.k,
            i,
            reference[i].to_bits(),
            candidate[i].to_bits()
        );
    }

    let byte_len = std::mem::size_of_val(reference);
    let reference_bytes =
        unsafe { std::slice::from_raw_parts(reference.as_ptr().cast::<u8>(), byte_len) };
    let candidate_bytes =
        unsafe { std::slice::from_raw_parts(candidate.as_ptr().cast::<u8>(), byte_len) };
    assert_eq!(reference_bytes, candidate_bytes, "F32 output memcmp");
}

fn run_shape(gpu: &mut Gpu, shape: Shape) {
    let Shape { label, m, k } = shape;
    let w_bits = (0..m * k).map(deterministic_f16_bits).collect::<Vec<_>>();
    let w_bytes = unsafe {
        std::slice::from_raw_parts(
            w_bits.as_ptr().cast::<u8>(),
            w_bits.len() * std::mem::size_of::<u16>(),
        )
    };
    let w = gpu
        .alloc_tensor(&[1, m, k], DType::F16)
        .expect("allocate W");
    gpu.memcpy_htod_auto(&w.buf, w_bytes).expect("upload W");

    let x_data = (0..k).map(deterministic_x).collect::<Vec<_>>();
    let x = gpu.upload_f32(&x_data, &[1, 1, k]).expect("upload X");
    let y_tiled = gpu.zeros(&[1, 1, m], DType::F32).expect("allocate tiled Y");
    let y_wave64 = gpu
        .zeros(&[1, 1, m], DType::F32)
        .expect("allocate wave64 Y");

    launch_tiled(gpu, &w, &x, &y_tiled, m, k);
    launch_wave64(gpu, &w, &x, &y_wave64, m, k);
    gpu.hip.device_synchronize().expect("parity sync");
    let reference = gpu.download_f32(&y_tiled).expect("download tiled Y");
    let candidate = gpu.download_f32(&y_wave64).expect("download wave64 Y");
    assert_memcmp(&reference, &candidate, shape);

    for _ in 0..WARMUP {
        launch_tiled(gpu, &w, &x, &y_tiled, m, k);
        launch_wave64(gpu, &w, &x, &y_wave64, m, k);
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let iters = iterations(m, k);
    let mut tiled_samples = Vec::with_capacity(SAMPLES);
    let mut wave64_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        if sample % 2 == 0 {
            tiled_samples.push(time_tiled(gpu, &w, &x, &y_tiled, m, k, iters));
            wave64_samples.push(time_wave64(gpu, &w, &x, &y_wave64, m, k, iters));
        } else {
            wave64_samples.push(time_wave64(gpu, &w, &x, &y_wave64, m, k, iters));
            tiled_samples.push(time_tiled(gpu, &w, &x, &y_tiled, m, k, iters));
        }
    }

    let tiled_us = median(tiled_samples);
    let wave64_us = median(wave64_samples);
    println!(
        "{label:20} M={m:5} K={k:5} iters={iters:3} memcmp=PASS tiled={tiled_us:9.3} us wave64={wave64_us:9.3} us speedup={:.3}x delta={:+.1}%",
        tiled_us / wave64_us,
        (tiled_us / wave64_us - 1.0) * 100.0,
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("GPU init");
    assert_eq!(gpu.arch, "gfx90a", "this benchmark targets MI250 gfx90a");
    println!(
        "arch={} G={G} B={BATCH} warmup={WARMUP} samples={SAMPLES} target_bytes_per_sample={} MiB",
        gpu.arch,
        TARGET_BYTES_PER_SAMPLE / (1024 * 1024)
    );
    println!("timing=median-of-{SAMPLES} batched host-wall us/call; correctness=F32 memcmp");
    for &shape in SHAPES {
        run_shape(&mut gpu, shape);
    }
    println!("PASS: all {} shapes are bit-exact", SHAPES.len());
}
