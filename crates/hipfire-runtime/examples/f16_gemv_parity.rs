// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! F16-weight GEMV cross-arch parity probe (#26 ds4 EP divergence hunt).
//!
//! The ds4 compressor projections are F16 and go through
//! `GemvFamily::run_auto` (the same call `gemv_auto` makes in
//! hipfire-arch-deepseek4). On gfx1151 the resolved path matches the
//! validated forward; on gfx1201 the compressor GEMV output came back
//! ~9x too large. This probe isolates the kernel stack from the model:
//! deterministic W[m,k] (F16) and x[k] (F32), CPU f32 reference, then
//! three GPU paths:
//!   1. dispatch `run_auto` (what the compressor actually calls)
//!   2. direct `gemm_f16_batched_lmhead` (batch=1)
//!   3. direct `gemm_f16_tiled` (n=1)
//!
//! Run: cargo run --release -p hipfire-runtime --example f16_gemv_parity

fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let frac = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if frac != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (frac | 0x80_0000) >> (1 - e + 13);
        return sign | m as u16;
    }
    sign | ((e as u16) << 10) | ((frac >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let b = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // subnormal
            let mut e = 127 - 15 - 10;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (((e + 10) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(b)
}

fn main() {
    use rdna_compute::{DType, Gpu};

    let m: usize = 256; // ds4 main-compressor proj_dim ballpark
    let k: usize = 4096; // hidden

    // Deterministic LCG fill, same on every box.
    let mut s: u64 = 0x1234_5678_9abc_def0;
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let w_f32: Vec<f32> = (0..m * k).map(|_| next() * 0.1).collect();
    let x_f32: Vec<f32> = (0..k).map(|_| next()).collect();
    let w_f16_bits: Vec<u16> = w_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let w_bytes: Vec<u8> = w_f16_bits.iter().flat_map(|b| b.to_le_bytes()).collect();

    // CPU reference: f32 accumulate over f16-rounded weights.
    let y_ref: Vec<f32> = (0..m)
        .map(|r| {
            (0..k)
                .map(|c| f16_bits_to_f32(w_f16_bits[r * k + c]) * x_f32[c])
                .sum::<f32>()
        })
        .collect();
    let l2 = |v: &[f32]| v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    let report = |name: &str, y: &[f32], y_ref: &[f32]| {
        let mut max_rel = 0f64;
        for (a, b) in y.iter().zip(y_ref) {
            let d = (*a as f64 - *b as f64).abs() / (b.abs() as f64).max(1e-3);
            if d > max_rel {
                max_rel = d;
            }
        }
        println!(
            "{name:>28}: l2={:.6e} (ref {:.6e}, ratio {:.4}) max_rel_err={:.3e} head={:.5},{:.5},{:.5},{:.5}",
            l2(y),
            l2(y_ref),
            l2(y) / l2(y_ref),
            max_rel,
            y[0],
            y[1],
            y[2],
            y[3]
        );
    };

    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch={} has_wmma_w32={} has_wmma_w32_gfx12={}", gpu.arch, gpu.arch_caps.has_wmma_w32(), gpu.arch_caps.has_wmma_w32_gfx12());

    let mut w_t = gpu.upload_raw(&w_bytes, &[m, k]).expect("upload w");
    w_t.dtype = DType::F16;
    let x_t = gpu.upload_f32(&x_f32, &[k]).expect("upload x");
    let y_t = gpu.upload_f32(&vec![0f32; m], &[m]).expect("upload y");

    // 1. dispatch run_auto — the exact compressor call shape.
    {
        use hipfire_dispatch::context::DispatchCtx;
        use hipfire_dispatch::families::gemv::WeightRef;
        let gemv = hipfire_runtime::llama::gemv_family();
        let ctx = DispatchCtx::new(&gpu);
        let wr = WeightRef { buf: &w_t, dtype: w_t.dtype, m, k, row_stride: 0, rotation: None, awq_scale: None };
        gemv.run_auto(&ctx, &mut gpu, &wr, &x_t, &y_t).expect("run_auto");
        let y = gpu.download_f32(&y_t).expect("dl");
        report("dispatch run_auto", &y, &y_ref);
    }

    // 2. direct gemm_f16_batched_lmhead (batch=1).
    {
        let _ = gpu.hip.memset(&y_t.buf, 0, m * 4);
        gpu.gemm_f16_batched_lmhead(&w_t, &x_t, &y_t, m, k, 1).expect("lmhead");
        let y = gpu.download_f32(&y_t).expect("dl");
        report("gemm_f16_batched_lmhead b1", &y, &y_ref);
    }

    // 3. direct gemm_f16_tiled (n=1).
    {
        let _ = gpu.hip.memset(&y_t.buf, 0, m * 4);
        gpu.gemm_f16_tiled(&w_t, &x_t, &y_t, m, k, 1).expect("tiled");
        let y = gpu.download_f32(&y_t).expect("dl");
        report("gemm_f16_tiled n1", &y, &y_ref);
    }
}
