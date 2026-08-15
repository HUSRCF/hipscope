// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx90a parity oracle for the official DeepSeek V4 F16 batched O-LoRA
//! projection shape.

use rdna_compute::{DType, Gpu};

const G: usize = 8;
const M: usize = 1024;
const K: usize = 4096;

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

fn sample(index: usize, salt: u32, scale: f32) -> f32 {
    let mut x = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b9);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ((x & 0xffff) as f32 / 32_767.5 - 1.0) * scale
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx90a", "this oracle targets MI250 wave64");

    let w_bits = (0..G * M * K)
        .map(|i| f32_to_f16_bits(sample(i, 17, 0.03125)))
        .collect::<Vec<_>>();
    let w_bytes =
        unsafe { std::slice::from_raw_parts(w_bits.as_ptr().cast::<u8>(), w_bits.len() * 2) };
    let d_w = gpu.alloc_tensor(&[G, M, K], DType::F16).unwrap();
    gpu.memcpy_htod_auto(&d_w.buf, w_bytes).unwrap();

    for batch in [1usize, 5usize] {
        let x = (0..batch * G * K)
            .map(|i| sample(i, 29, 0.25))
            .collect::<Vec<_>>();
        let d_x = gpu.upload_f32(&x, &[batch, G, K]).unwrap();
        let d_ref = gpu.zeros(&[batch, G, M], DType::F32).unwrap();
        let d_wave = gpu.zeros(&[batch, G, M], DType::F32).unwrap();

        for b in 0..batch {
            for g in 0..G {
                let w = d_w.sub_offset(g * M * K, M * K);
                let x = d_x.sub_offset((b * G + g) * K, K);
                let y = d_ref.sub_offset((b * G + g) * M, M);
                gpu.gemm_f16_tiled(&w, &x, &y, M, K, 1).unwrap();
            }
        }
        gpu.wo_per_group_batched_f16_wave64_row2_gfx90a(
            &d_w,
            &d_x,
            &d_wave,
            G as i32,
            M as i32,
            K as i32,
            batch as i32,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let reference = gpu.download_f32(&d_ref).unwrap();
        let wave = gpu.download_f32(&d_wave).unwrap();
        let mut bit_mismatch = 0usize;
        let mut max_abs = 0.0f32;
        for (&a, &b) in reference.iter().zip(&wave) {
            bit_mismatch += usize::from(a.to_bits() != b.to_bits());
            max_abs = max_abs.max((a - b).abs());
        }
        println!("B={batch} G={G} M={M} K={K}: bit_mismatch={bit_mismatch} max_abs={max_abs:.3e}");
        assert_eq!(bit_mismatch, 0, "wave64 row2 must match scalar F16 bits");
    }

    println!("PASS: DeepSeek V4 F16 batched O-LoRA wave64 parity");
}
