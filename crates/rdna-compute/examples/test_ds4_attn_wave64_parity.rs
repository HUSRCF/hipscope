// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Focused gfx90a parity probe for the logical-wave32 reductions used by the
//! DeepSeek V4 decode and batched-prefill attention kernels.
//!
//! With q=0, every cache score and the attention sink have exp(score)=1, so
//! the exact reference denominator is `n_total + 1`. This catches accidental
//! wave64 broadcasts at the first 32-lane boundary without involving model
//! weights or autoregressive feedback.

use rdna_compute::{DType, Gpu};

const HEAD_DIM: usize = 512;
const SWA_WINDOW: usize = 128;
const TOPK_WINDOW: usize = 512;

fn i32_bytes(value: i32) -> [u8; 4] {
    value.to_le_bytes()
}

fn check_close(label: &str, got: &[f32], expected: f32) {
    let max_abs = got
        .iter()
        .map(|&x| (x - expected).abs())
        .fold(0.0f32, f32::max);
    println!("{label}: expected={expected:.8} max_abs={max_abs:.3e}");
    assert!(
        max_abs <= 2.0e-5,
        "{label}: GPU output diverged from analytical CPU reference: max_abs={max_abs}"
    );
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    println!("DeepSeek4 attention wave32 parity probe: arch={}", gpu.arch);
    assert_eq!(
        gpu.arch, "gfx90a",
        "this probe targets the MI250 wave64 path"
    );

    let q = vec![0.0f32; HEAD_DIM];
    let k = vec![0.0f32; HEAD_DIM * SWA_WINDOW];
    let mut swa_v = vec![0.0f32; HEAD_DIM * SWA_WINDOW];
    for d in 0..HEAD_DIM {
        for p in 0..SWA_WINDOW {
            swa_v[d * SWA_WINDOW + p] = (p + 1) as f32 / SWA_WINDOW as f32;
        }
    }

    let topk_k = vec![0.0f32; HEAD_DIM * TOPK_WINDOW];
    let mut topk_v = vec![0.0f32; HEAD_DIM * TOPK_WINDOW];
    for d in 0..HEAD_DIM {
        for p in 0..TOPK_WINDOW {
            topk_v[d * TOPK_WINDOW + p] = 1.0 + (p + 1) as f32 / TOPK_WINDOW as f32;
        }
    }

    let d_q = gpu.upload_f32(&q, &[HEAD_DIM]).unwrap();
    let d_k = gpu.upload_f32(&k, &[HEAD_DIM * SWA_WINDOW]).unwrap();
    let d_swa_v = gpu.upload_f32(&swa_v, &[HEAD_DIM * SWA_WINDOW]).unwrap();
    let d_topk_k = gpu.upload_f32(&topk_k, &[HEAD_DIM * TOPK_WINDOW]).unwrap();
    let d_topk_v = gpu.upload_f32(&topk_v, &[HEAD_DIM * TOPK_WINDOW]).unwrap();
    let d_sink = gpu.upload_f32(&[0.0], &[1]).unwrap();
    let d_out = gpu.zeros(&[HEAD_DIM], DType::F32).unwrap();

    for n_valid in [1usize, 31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128] {
        let d_n = gpu.upload_raw(&i32_bytes(n_valid as i32), &[4]).unwrap();
        gpu.deepseek4_attn_swa_buf(
            &d_q,
            &d_k,
            &d_swa_v,
            &d_sink,
            &d_out,
            &d_n,
            1,
            HEAD_DIM as i32,
            8,
            SWA_WINDOW as i32,
        )
        .unwrap();
        let got = gpu.download_f32(&d_out).unwrap();
        let value_sum = (n_valid * (n_valid + 1)) as f32 / (2.0 * SWA_WINDOW as f32);
        let expected = value_sum / (n_valid + 1) as f32;
        check_close(&format!("swa n={n_valid}"), &got, expected);
    }

    for (n_swa, n_topk) in [
        (31usize, 0usize),
        (32, 0),
        (32, 1),
        (63, 1),
        (64, 1),
        (64, 32),
        (64, 33),
        (96, 32),
        (128, 128),
        (128, 384),
    ] {
        let d_n_swa = gpu.upload_raw(&i32_bytes(n_swa as i32), &[4]).unwrap();
        let d_n_topk = gpu.upload_raw(&i32_bytes(n_topk as i32), &[4]).unwrap();
        gpu.deepseek4_attn_swa_topk_f32_buf(
            &d_q,
            &d_k,
            &d_swa_v,
            &d_topk_k,
            &d_topk_v,
            &d_sink,
            &d_out,
            &d_n_swa,
            &d_n_topk,
            1,
            HEAD_DIM as i32,
            SWA_WINDOW as i32,
            TOPK_WINDOW as i32,
        )
        .unwrap();
        let got = gpu.download_f32(&d_out).unwrap();
        let swa_sum = (n_swa * (n_swa + 1)) as f32 / (2.0 * SWA_WINDOW as f32);
        let topk_sum = n_topk as f32 + (n_topk * (n_topk + 1)) as f32 / (2.0 * TOPK_WINDOW as f32);
        let expected = (swa_sum + topk_sum) / (n_swa + n_topk + 1) as f32;
        check_close(
            &format!("swa+topk n_swa={n_swa} n_topk={n_topk}"),
            &got,
            expected,
        );
    }

    check_batched_attention(&mut gpu, &d_sink);

    println!("PASS: decode and batched-prefill logical-wave32 boundaries match CPU");
}
fn i32_slice_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn check_batched_close(label: &str, got: &[f32], expected: &[f32]) {
    let expected_len = expected.len() * HEAD_DIM;
    if got.len() != expected_len {
        println!(
            "RESULT kernel={label} max_abs=NaN rms=NaN pass=false \
             got_len={} expected_len={expected_len}",
            got.len()
        );
        assert_eq!(got.len(), expected_len, "{label}: output shape mismatch");
    }

    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut finite = true;
    for (b, &want) in expected.iter().enumerate() {
        let row = &got[b * HEAD_DIM..(b + 1) * HEAD_DIM];
        for &value in row {
            let diff = f64::from(value) - f64::from(want);
            finite &= diff.is_finite();
            max_abs = max_abs.max(diff.abs());
            sum_sq += diff * diff;
        }
    }

    let rms = (sum_sq / got.len() as f64).sqrt();
    let passed = finite && max_abs <= 2.0e-5 && rms.is_finite();
    println!(
        "RESULT kernel={label} max_abs={max_abs:.6e} rms={rms:.6e} pass={passed} \
         batches={}",
        expected.len(),
    );
    assert!(
        passed,
        "{label}: GPU output diverged from analytical CPU reference: \
         max_abs={max_abs} rms={rms} finite={finite}"
    );
}

fn check_batched_attention(gpu: &mut Gpu, d_sink: &rdna_compute::GpuTensor) {
    let batched_swa_counts = [1i32, 31, 32, 33, 63, 64, 127, 128];
    let batch_size = batched_swa_counts.len();
    let batched_q = vec![0.0f32; batch_size * HEAD_DIM];
    let batched_k = vec![0.0f32; batch_size * HEAD_DIM * SWA_WINDOW];
    let mut batched_swa_v = vec![0.0f32; batch_size * HEAD_DIM * SWA_WINDOW];
    for b in 0..batch_size {
        for d in 0..HEAD_DIM {
            for p in 0..SWA_WINDOW {
                let offset = (b * HEAD_DIM + d) * SWA_WINDOW + p;
                batched_swa_v[offset] = b as f32 * 0.125 + (p + 1) as f32 / SWA_WINDOW as f32;
            }
        }
    }
    let d_batched_q = gpu
        .upload_f32(&batched_q, &[batch_size * HEAD_DIM])
        .unwrap();
    let d_batched_k = gpu
        .upload_f32(&batched_k, &[batch_size * HEAD_DIM * SWA_WINDOW])
        .unwrap();
    let d_batched_swa_v = gpu
        .upload_f32(&batched_swa_v, &[batch_size * HEAD_DIM * SWA_WINDOW])
        .unwrap();
    let d_batched_swa_counts = gpu
        .upload_raw(
            &i32_slice_bytes(&batched_swa_counts),
            &[batch_size * std::mem::size_of::<i32>()],
        )
        .unwrap();
    let d_batched_out = gpu.zeros(&[batch_size * HEAD_DIM], DType::F32).unwrap();
    gpu.deepseek4_attn_swa_batched(
        &d_batched_q,
        &d_batched_k,
        &d_batched_swa_v,
        d_sink,
        &d_batched_swa_counts,
        &d_batched_out,
        1,
        HEAD_DIM as i32,
        8,
        SWA_WINDOW as i32,
        batch_size as i32,
    )
    .unwrap();
    let batched_swa_expected = batched_swa_counts
        .iter()
        .enumerate()
        .map(|(b, &n)| {
            let n = n as usize;
            let value_sum =
                n as f32 * b as f32 * 0.125 + (n * (n + 1)) as f32 / (2.0 * SWA_WINDOW as f32);
            value_sum / (n + 1) as f32
        })
        .collect::<Vec<_>>();
    check_batched_close(
        "deepseek4_attn_swa_batched",
        &gpu.download_f32(&d_batched_out).unwrap(),
        &batched_swa_expected,
    );

    let batched_topk_cases = [
        (31i32, 0i32),
        (32, 0),
        (32, 1),
        (63, 1),
        (64, 32),
        (64, 33),
        (96, 32),
        (128, 384),
    ];
    let topk_batch_size = batched_topk_cases.len();
    let topk_swa_counts = batched_topk_cases.map(|(n_swa, _)| n_swa);
    let topk_counts = batched_topk_cases.map(|(_, n_topk)| n_topk);
    let topk_q = vec![0.0f32; topk_batch_size * HEAD_DIM];
    let topk_swa_k = vec![0.0f32; topk_batch_size * HEAD_DIM * SWA_WINDOW];
    let topk_k = vec![0.0f32; topk_batch_size * HEAD_DIM * TOPK_WINDOW];
    let mut topk_swa_v = vec![0.0f32; topk_batch_size * HEAD_DIM * SWA_WINDOW];
    let mut topk_v = vec![0.0f32; topk_batch_size * HEAD_DIM * TOPK_WINDOW];
    for b in 0..topk_batch_size {
        for d in 0..HEAD_DIM {
            for p in 0..SWA_WINDOW {
                let offset = (b * HEAD_DIM + d) * SWA_WINDOW + p;
                topk_swa_v[offset] = b as f32 * 0.125 + (p + 1) as f32 / SWA_WINDOW as f32;
            }
            for p in 0..TOPK_WINDOW {
                let offset = (b * HEAD_DIM + d) * TOPK_WINDOW + p;
                topk_v[offset] = 1.0 + b as f32 * 0.125 + (p + 1) as f32 / TOPK_WINDOW as f32;
            }
        }
    }
    let d_topk_q = gpu
        .upload_f32(&topk_q, &[topk_batch_size * HEAD_DIM])
        .unwrap();
    let d_topk_swa_k = gpu
        .upload_f32(&topk_swa_k, &[topk_batch_size * HEAD_DIM * SWA_WINDOW])
        .unwrap();
    let d_topk_swa_v = gpu
        .upload_f32(&topk_swa_v, &[topk_batch_size * HEAD_DIM * SWA_WINDOW])
        .unwrap();
    let d_topk_k_batched = gpu
        .upload_f32(&topk_k, &[topk_batch_size * HEAD_DIM * TOPK_WINDOW])
        .unwrap();
    let d_topk_v_batched = gpu
        .upload_f32(&topk_v, &[topk_batch_size * HEAD_DIM * TOPK_WINDOW])
        .unwrap();
    let d_topk_swa_counts = gpu
        .upload_raw(
            &i32_slice_bytes(&topk_swa_counts),
            &[topk_batch_size * std::mem::size_of::<i32>()],
        )
        .unwrap();
    let d_topk_counts = gpu
        .upload_raw(
            &i32_slice_bytes(&topk_counts),
            &[topk_batch_size * std::mem::size_of::<i32>()],
        )
        .unwrap();
    let d_topk_out = gpu
        .zeros(&[topk_batch_size * HEAD_DIM], DType::F32)
        .unwrap();
    gpu.deepseek4_attn_swa_topk_batched_f32(
        &d_topk_q,
        &d_topk_swa_k,
        &d_topk_swa_v,
        &d_topk_k_batched,
        &d_topk_v_batched,
        d_sink,
        &d_topk_swa_counts,
        &d_topk_counts,
        &d_topk_out,
        1,
        HEAD_DIM as i32,
        SWA_WINDOW as i32,
        TOPK_WINDOW as i32,
        topk_batch_size as i32,
    )
    .unwrap();
    let topk_expected = batched_topk_cases
        .iter()
        .enumerate()
        .map(|(b, &(n_swa, n_topk))| {
            let n_swa = n_swa as usize;
            let n_topk = n_topk as usize;
            let swa_sum = n_swa as f32 * b as f32 * 0.125
                + (n_swa * (n_swa + 1)) as f32 / (2.0 * SWA_WINDOW as f32);
            let topk_sum = n_topk as f32 * (1.0 + b as f32 * 0.125)
                + (n_topk * (n_topk + 1)) as f32 / (2.0 * TOPK_WINDOW as f32);
            (swa_sum + topk_sum) / (n_swa + n_topk + 1) as f32
        })
        .collect::<Vec<_>>();
    check_batched_close(
        "deepseek4_attn_swa_topk_batched_f32",
        &gpu.download_f32(&d_topk_out).unwrap(),
        &topk_expected,
    );

    let n_compressed = topk_batch_size * TOPK_WINDOW;
    let mut direct_kv = vec![0.0f32; n_compressed * HEAD_DIM];
    let mut direct_indices = vec![0i32; topk_batch_size * TOPK_WINDOW];
    for b in 0..topk_batch_size {
        for p in 0..TOPK_WINDOW {
            let idx = b * TOPK_WINDOW + p;
            direct_indices[idx] = idx as i32;
            for d in 0..HEAD_DIM {
                direct_kv[idx * HEAD_DIM + d] =
                    1.0 + b as f32 * 0.125 + (p + 1) as f32 / TOPK_WINDOW as f32;
            }
        }
    }
    let d_direct_kv = gpu
        .upload_f32(&direct_kv, &[n_compressed * HEAD_DIM])
        .unwrap();
    let d_direct_indices = gpu
        .upload_raw(
            &i32_slice_bytes(&direct_indices),
            &[direct_indices.len() * std::mem::size_of::<i32>()],
        )
        .unwrap();
    gpu.deepseek4_attn_swa_topk_direct_batched_f32(
        &d_topk_q,
        &d_topk_swa_k,
        &d_topk_swa_v,
        &d_direct_kv,
        &d_direct_indices,
        d_sink,
        &d_topk_swa_counts,
        &d_topk_counts,
        &d_topk_out,
        1,
        HEAD_DIM as i32,
        SWA_WINDOW as i32,
        TOPK_WINDOW as i32,
        n_compressed as i32,
        topk_batch_size as i32,
    )
    .unwrap();
    check_batched_close(
        "deepseek4_attn_swa_topk_direct_batched_f32",
        &gpu.download_f32(&d_topk_out).unwrap(),
        &topk_expected,
    );
}
