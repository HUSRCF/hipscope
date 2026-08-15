// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 hipfire contributors

//! CPU/GPU parity for the MQ4G256 indexed MoE kernels used by DeepSeek V4.
//!
//! This probe deliberately uses the production DeepSeek V4 dimensions:
//! gate+up [4096, 4096], down [4096, 2048], and top-k=6. It covers batched
//! decode/prefill shapes and the EP zero-dummy pointer contract.
//!
//! Run:
//!   cargo run --release -p rdna-compute \
//!     --example test_mq4g256_deepseek4_indexed_parity

use rdna_compute::{DType, Gpu, GpuTensor};

const N_EXPERTS: usize = 6;
const TOP_K: usize = 6;
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 2048;
const GATE_UP_M: usize = INTERMEDIATE * 2;
const GROUP_SIZE: usize = 256;
const GROUP_BYTES: usize = 136;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn build_weight(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert_eq!(k % GROUP_SIZE, 0);
    let groups = k / GROUP_SIZE;
    let row_bytes = groups * GROUP_BYTES;
    let mut out = vec![0_u8; m * row_bytes];
    let mut state = seed;
    for row in 0..m {
        for group in 0..groups {
            let base = row * row_bytes + group * GROUP_BYTES;
            let scale = 0.0025 + (lcg(&mut state) as f64 / u32::MAX as f64) as f32 * 0.0175;
            let zero = -scale * (5.0 + (lcg(&mut state) % 6) as f32);
            out[base..base + 4].copy_from_slice(&scale.to_bits().to_le_bytes());
            out[base + 4..base + 8].copy_from_slice(&zero.to_bits().to_le_bytes());
            for packed in &mut out[base + 8..base + GROUP_BYTES] {
                let lo = (lcg(&mut state) >> 28) as u8;
                let hi = (lcg(&mut state) >> 28) as u8;
                *packed = lo | (hi << 4);
            }
        }
    }
    out
}

fn dot_mq4(weight: &[u8], row: usize, x: &[f32], m: usize, k: usize) -> f32 {
    assert!(row < m);
    assert_eq!(x.len(), k);
    let groups = k / GROUP_SIZE;
    let row_bytes = groups * GROUP_BYTES;
    let mut sum = 0.0_f64;
    for group in 0..groups {
        let base = row * row_bytes + group * GROUP_BYTES;
        let scale = f32::from_bits(u32::from_le_bytes(
            weight[base..base + 4].try_into().unwrap(),
        ));
        let zero = f32::from_bits(u32::from_le_bytes(
            weight[base + 4..base + 8].try_into().unwrap(),
        ));
        for i in 0..GROUP_SIZE {
            let packed = weight[base + 8 + i / 2];
            let q = if i & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            sum += (scale * q as f32 + zero) as f64 * x[group * GROUP_SIZE + i] as f64;
        }
    }
    sum as f32
}

fn upload_raw(gpu: &mut Gpu, bytes: &[u8]) -> GpuTensor {
    let tensor = gpu
        .alloc_tensor(&[bytes.len()], DType::Raw)
        .expect("allocate raw tensor");
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload raw tensor");
    tensor
}

fn upload_f32(gpu: &mut Gpu, values: &[f32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<f32>(),
        )
    };
    let tensor = gpu
        .alloc_tensor(&[values.len()], DType::F32)
        .expect("allocate f32 tensor");
    gpu.hip
        .memcpy_htod(&tensor.buf, bytes)
        .expect("upload f32 tensor");
    tensor
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr().cast::<u8>(),
            values.len() * size_of::<i32>(),
        )
    };
    upload_raw(gpu, bytes)
}

fn upload_ptrs(gpu: &mut Gpu, tensors: &[&GpuTensor]) -> GpuTensor {
    let bytes: Vec<u8> = tensors
        .iter()
        .flat_map(|tensor| (tensor.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    upload_raw(gpu, &bytes)
}

fn download_f32(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Vec<f32> {
    let mut values = vec![0.0_f32; len];
    let bytes = unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            values.len() * size_of::<f32>(),
        )
    };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .expect("download f32 tensor");
    values
}

fn deterministic_values(len: usize, seed: u32, amplitude: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            let unit = lcg(&mut state) as f64 / u32::MAX as f64;
            ((unit * 2.0 - 1.0) as f32) * amplitude
        })
        .collect()
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) -> bool {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut err2 = 0.0_f64;
    let mut ref2 = 0.0_f64;
    let mut bad = 0_usize;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let abs = (actual - expected).abs();
        let rel = abs / expected.abs().max(1.0e-6);
        let tolerance = 5.0e-3 + expected.abs() * 5.0e-4;
        bad += usize::from(!actual.is_finite() || abs > tolerance);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        err2 += (actual - expected) as f64 * (actual - expected) as f64;
        ref2 += expected as f64 * expected as f64;
    }
    let nrmse = (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
    println!(
        "{label}: n={} bad={} max_abs={:.6e} max_rel={:.6e} nrmse={:.6e}",
        actual.len(),
        bad,
        max_abs,
        max_rel,
        nrmse
    );
    bad == 0
}

fn route_indices(batch: usize) -> Vec<i32> {
    (0..batch)
        .flat_map(|bid| (0..TOP_K).map(move |krank| ((bid * 5 + krank * 5) % N_EXPERTS) as i32))
        .collect()
}

fn run_batch(
    gpu: &mut Gpu,
    gate_weights: &[Vec<u8>],
    gate_ptrs: &GpuTensor,
    down_weights: &[Vec<u8>],
    down_ptrs: &GpuTensor,
    batch: usize,
) -> bool {
    let indices = route_indices(batch);
    let indices_gpu = upload_i32(gpu, &indices);
    let x = deterministic_values(batch * HIDDEN, 0x1234_0000 ^ batch as u32, 0.25);
    let x_gpu = upload_f32(gpu, &x);
    let gate_gpu = gpu
        .alloc_tensor(&[batch * TOP_K * INTERMEDIATE], DType::F32)
        .expect("allocate gate output");
    let up_gpu = gpu
        .alloc_tensor(&[batch * TOP_K * INTERMEDIATE], DType::F32)
        .expect("allocate up output");
    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
        gate_ptrs,
        &indices_gpu,
        &x_gpu,
        &gate_gpu,
        &up_gpu,
        GATE_UP_M,
        HIDDEN,
        TOP_K,
        batch,
    )
    .expect("MQ4 indexed gate/up");
    gpu.hip.device_synchronize().expect("gate/up synchronize");

    let mut gate_ref = vec![0.0_f32; batch * TOP_K * INTERMEDIATE];
    let mut up_ref = vec![0.0_f32; batch * TOP_K * INTERMEDIATE];
    for bid in 0..batch {
        let x_row = &x[bid * HIDDEN..(bid + 1) * HIDDEN];
        for krank in 0..TOP_K {
            let expert = indices[bid * TOP_K + krank] as usize;
            let out_base = (bid * TOP_K + krank) * INTERMEDIATE;
            for row in 0..INTERMEDIATE {
                gate_ref[out_base + row] =
                    dot_mq4(&gate_weights[expert], row, x_row, GATE_UP_M, HIDDEN);
                up_ref[out_base + row] = dot_mq4(
                    &gate_weights[expert],
                    row + INTERMEDIATE,
                    x_row,
                    GATE_UP_M,
                    HIDDEN,
                );
            }
        }
    }
    let mut ok = compare(
        &format!("batch{batch}_gate"),
        &download_f32(gpu, &gate_gpu, gate_ref.len()),
        &gate_ref,
    );
    ok &= compare(
        &format!("batch{batch}_up"),
        &download_f32(gpu, &up_gpu, up_ref.len()),
        &up_ref,
    );

    let rot = deterministic_values(
        batch * TOP_K * INTERMEDIATE,
        0x5678_0000 ^ batch as u32,
        0.2,
    );
    let rot_gpu = upload_f32(gpu, &rot);
    let weights: Vec<f32> = (0..batch)
        .flat_map(|bid| {
            let denominator = (21 + bid) as f32;
            (1..=TOP_K).map(move |krank| krank as f32 / denominator)
        })
        .collect();
    let weights_gpu = upload_f32(gpu, &weights);
    let residual = deterministic_values(batch * HIDDEN, 0x9abc_0000 ^ batch as u32, 0.1);
    let residual_gpu = upload_f32(gpu, &residual);
    gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
        down_ptrs,
        &indices_gpu,
        &weights_gpu,
        &rot_gpu,
        &residual_gpu,
        HIDDEN,
        INTERMEDIATE,
        TOP_K,
        batch,
    )
    .expect("MQ4 indexed down");
    gpu.hip.device_synchronize().expect("down synchronize");
    let mut down_ref = residual;
    for bid in 0..batch {
        for krank in 0..TOP_K {
            let expert = indices[bid * TOP_K + krank] as usize;
            let rot_base = (bid * TOP_K + krank) * INTERMEDIATE;
            let rot_row = &rot[rot_base..rot_base + INTERMEDIATE];
            let scale = weights[bid * TOP_K + krank];
            for row in 0..HIDDEN {
                down_ref[bid * HIDDEN + row] +=
                    scale * dot_mq4(&down_weights[expert], row, rot_row, HIDDEN, INTERMEDIATE);
            }
        }
    }
    ok &= compare(
        &format!("batch{batch}_down"),
        &download_f32(gpu, &residual_gpu, down_ref.len()),
        &down_ref,
    );
    ok
}

fn run_ep_dummy(
    gpu: &mut Gpu,
    gate_weights: &[Vec<u8>],
    gate_tensors: &[GpuTensor],
    down_weights: &[Vec<u8>],
    down_tensors: &[GpuTensor],
) -> bool {
    let zero_gate = upload_raw(
        gpu,
        &vec![0_u8; GATE_UP_M * (HIDDEN / GROUP_SIZE) * GROUP_BYTES],
    );
    let zero_down = upload_raw(
        gpu,
        &vec![0_u8; HIDDEN * (INTERMEDIATE / GROUP_SIZE) * GROUP_BYTES],
    );
    let gate_routes: Vec<&GpuTensor> = (0..TOP_K)
        .map(|expert| {
            if expert & 1 == 0 {
                &gate_tensors[expert]
            } else {
                &zero_gate
            }
        })
        .collect();
    let down_routes: Vec<&GpuTensor> = (0..TOP_K)
        .map(|expert| {
            if expert & 1 == 0 {
                &down_tensors[expert]
            } else {
                &zero_down
            }
        })
        .collect();
    let gate_ptrs = upload_ptrs(gpu, &gate_routes);
    let down_ptrs = upload_ptrs(gpu, &down_routes);
    let indices: Vec<i32> = (0..TOP_K as i32).collect();
    let indices_gpu = upload_i32(gpu, &indices);
    let x = deterministic_values(HIDDEN, 0xdead_beef, 0.25);
    let x_gpu = upload_f32(gpu, &x);
    let gate_gpu = gpu
        .alloc_tensor(&[TOP_K * INTERMEDIATE], DType::F32)
        .expect("allocate EP gate");
    let up_gpu = gpu
        .alloc_tensor(&[TOP_K * INTERMEDIATE], DType::F32)
        .expect("allocate EP up");
    gpu.gemv_hfq4g256_moe_gate_up_k8_indexed_batched(
        &gate_ptrs,
        &indices_gpu,
        &x_gpu,
        &gate_gpu,
        &up_gpu,
        GATE_UP_M,
        HIDDEN,
        TOP_K,
        1,
    )
    .expect("EP dummy gate/up");
    gpu.hip
        .device_synchronize()
        .expect("EP gate/up synchronize");
    let mut gate_ref = vec![0.0_f32; TOP_K * INTERMEDIATE];
    let mut up_ref = vec![0.0_f32; TOP_K * INTERMEDIATE];
    for expert in (0..TOP_K).step_by(2) {
        let base = expert * INTERMEDIATE;
        for row in 0..INTERMEDIATE {
            gate_ref[base + row] = dot_mq4(&gate_weights[expert], row, &x, GATE_UP_M, HIDDEN);
            up_ref[base + row] = dot_mq4(
                &gate_weights[expert],
                row + INTERMEDIATE,
                &x,
                GATE_UP_M,
                HIDDEN,
            );
        }
    }
    let mut ok = compare(
        "ep_dummy_gate",
        &download_f32(gpu, &gate_gpu, gate_ref.len()),
        &gate_ref,
    );
    ok &= compare(
        "ep_dummy_up",
        &download_f32(gpu, &up_gpu, up_ref.len()),
        &up_ref,
    );

    let rot = deterministic_values(TOP_K * INTERMEDIATE, 0xcafe_babe, 0.2);
    let rot_gpu = upload_f32(gpu, &rot);
    let scales: Vec<f32> = (1..=TOP_K).map(|slot| slot as f32 / 21.0).collect();
    let scales_gpu = upload_f32(gpu, &scales);
    let down_gpu = upload_f32(gpu, &vec![0.0_f32; HIDDEN]);
    gpu.gemv_hfq4g256_moe_down_residual_scaled_k8_indexed_batched(
        &down_ptrs,
        &indices_gpu,
        &scales_gpu,
        &rot_gpu,
        &down_gpu,
        HIDDEN,
        INTERMEDIATE,
        TOP_K,
        1,
    )
    .expect("EP dummy down");
    gpu.hip.device_synchronize().expect("EP down synchronize");
    let mut down_ref = vec![0.0_f32; HIDDEN];
    for expert in (0..TOP_K).step_by(2) {
        let rot_row = &rot[expert * INTERMEDIATE..(expert + 1) * INTERMEDIATE];
        for row in 0..HIDDEN {
            down_ref[row] +=
                scales[expert] * dot_mq4(&down_weights[expert], row, rot_row, HIDDEN, INTERMEDIATE);
        }
    }
    ok &= compare(
        "ep_dummy_down",
        &download_f32(gpu, &down_gpu, down_ref.len()),
        &down_ref,
    );
    ok
}

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    println!(
        "arch={} gate_up=[{},{}] down=[{},{}] top_k={}",
        gpu.arch, GATE_UP_M, HIDDEN, HIDDEN, INTERMEDIATE, TOP_K
    );

    let gate_weights: Vec<Vec<u8>> = (0..N_EXPERTS)
        .map(|expert| build_weight(GATE_UP_M, HIDDEN, 0x1000_0000 + expert as u32))
        .collect();
    let down_weights: Vec<Vec<u8>> = (0..N_EXPERTS)
        .map(|expert| build_weight(HIDDEN, INTERMEDIATE, 0x2000_0000 + expert as u32))
        .collect();
    let gate_tensors: Vec<GpuTensor> = gate_weights
        .iter()
        .map(|weight| upload_raw(&mut gpu, weight))
        .collect();
    let down_tensors: Vec<GpuTensor> = down_weights
        .iter()
        .map(|weight| upload_raw(&mut gpu, weight))
        .collect();
    let gate_refs: Vec<&GpuTensor> = gate_tensors.iter().collect();
    let down_refs: Vec<&GpuTensor> = down_tensors.iter().collect();
    let gate_ptrs = upload_ptrs(&mut gpu, &gate_refs);
    let down_ptrs = upload_ptrs(&mut gpu, &down_refs);

    let mut ok = true;
    for batch in [1, 2, 7] {
        ok &= run_batch(
            &mut gpu,
            &gate_weights,
            &gate_ptrs,
            &down_weights,
            &down_ptrs,
            batch,
        );
    }
    ok &= run_ep_dummy(
        &mut gpu,
        &gate_weights,
        &gate_tensors,
        &down_weights,
        &down_tensors,
    );
    assert!(ok, "MQ4G256 DeepSeek4 indexed parity failed");
}
