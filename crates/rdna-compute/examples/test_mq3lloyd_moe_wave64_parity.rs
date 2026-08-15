//! CPU/GPU parity channel for the gfx90a MQ3-Lloyd indexed MoE GEMVs.

use half::f16;
use rdna_compute::{DType, Gpu, GpuTensor};

fn make_weights(m: usize, k: usize, seed: u32) -> Vec<u8> {
    assert_eq!(k % 256, 0);
    let mut state = seed;
    let mut out = Vec::with_capacity(m * (k / 256) * 112);
    for row in 0..m {
        for group in 0..k / 256 {
            let scale = 0.125 + ((row + group) % 11) as f32 * 0.015625;
            for value in [-3.0, -1.75, -0.875, -0.25, 0.375, 1.0, 1.875, 3.25] {
                out.extend_from_slice(&f16::from_f32(value * scale).to_bits().to_le_bytes());
            }
            let mut packed = [0_u8; 96];
            for index in 0..256 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let code = ((state >> 29) & 7) as u8;
                let bit = index * 3;
                let byte = bit >> 3;
                let shift = bit & 7;
                packed[byte] |= code << shift;
                if shift > 5 {
                    packed[byte + 1] |= code >> (8 - shift);
                }
            }
            out.extend_from_slice(&packed);
        }
    }
    out
}

fn dequant_dot(weights: &[u8], row: usize, x: &[f32], m: usize, k: usize) -> f32 {
    assert!(row < m);
    let row_bytes = (k / 256) * 112;
    let mut sum = 0.0_f32;
    for group in 0..k / 256 {
        let base = row * row_bytes + group * 112;
        let codebook = std::array::from_fn::<_, 8, _>(|i| {
            f16::from_bits(u16::from_le_bytes([
                weights[base + i * 2],
                weights[base + i * 2 + 1],
            ]))
            .to_f32()
        });
        let packed = &weights[base + 16..base + 112];
        for index in 0..256 {
            let bit = index * 3;
            let byte = bit >> 3;
            let shift = bit & 7;
            let mut bits = packed[byte] as u16;
            if byte + 1 < packed.len() {
                bits |= (packed[byte + 1] as u16) << 8;
            }
            let code = ((bits >> shift) & 7) as usize;
            sum += codebook[code] * x[group * 256 + index];
        }
    }
    sum
}

fn upload(gpu: &mut Gpu, bytes: &[u8], dtype: DType) -> GpuTensor {
    let tensor = gpu.alloc_tensor(&[bytes.len()], dtype).expect("allocate");
    gpu.hip.memcpy_htod(&tensor.buf, bytes).expect("upload");
    tensor
}

fn upload_f32(gpu: &mut Gpu, values: &[f32]) -> GpuTensor {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
    upload(gpu, bytes, DType::F32)
}

fn upload_i32(gpu: &mut Gpu, values: &[i32]) -> GpuTensor {
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) };
    upload(gpu, bytes, DType::Raw)
}

fn download(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Vec<f32> {
    let mut values = vec![0.0_f32; len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len * 4) };
    gpu.hip.memcpy_dtoh(bytes, &tensor.buf).expect("download");
    values
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) -> bool {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut bad = 0_usize;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let abs = (actual - expected).abs();
        let rel = abs / expected.abs().max(1.0e-5);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        bad += usize::from(!actual.is_finite() || (abs > 5.0e-3 && rel > 5.0e-4));
    }
    println!("{label}: max_abs={max_abs:.7e} max_rel={max_rel:.7e} bad={bad}");
    bad == 0
}

fn main() {
    const M: usize = 64;
    const K: usize = 1280;
    const TOP_K: usize = 6;

    let mut gpu = Gpu::init().expect("Gpu::init");
    println!("detected arch={} M={M} K={K} top_k={TOP_K}", gpu.arch);
    assert_eq!(
        gpu.arch, "gfx90a",
        "this channel must witness the gfx90a path"
    );

    let weights: Vec<Vec<u8>> = (0..TOP_K)
        .map(|expert| make_weights(M * 2, K, 0x1234_5678 + expert as u32 * 97))
        .collect();
    let weight_tensors: Vec<GpuTensor> = weights
        .iter()
        .map(|weight| upload(&mut gpu, weight, DType::MQ3G256Lloyd))
        .collect();
    let ptr_bytes: Vec<u8> = weight_tensors
        .iter()
        .flat_map(|tensor| (tensor.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let expert_ptrs = upload(&mut gpu, &ptr_bytes, DType::Raw);
    let all_ids: Vec<i32> = (0..TOP_K as i32).collect();
    let all_indices = upload_i32(&mut gpu, &all_ids);

    let x: Vec<f32> = (0..K)
        .map(|index| (((index * 17 + 5) % 71) as f32 - 35.0) / 29.0)
        .collect();
    let x_gpu = upload_f32(&mut gpu, &x);
    let gate = gpu.alloc_tensor(&[TOP_K * M], DType::F32).expect("gate");
    let up = gpu.alloc_tensor(&[TOP_K * M], DType::F32).expect("up");

    gpu.deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
        &expert_ptrs,
        &all_indices,
        &x_gpu,
        &gate,
        &up,
        M * 2,
        K,
        TOP_K,
    )
    .expect("MQ3 gate/up");
    gpu.hip.device_synchronize().expect("gate/up sync");

    let mut gate_ref = Vec::with_capacity(TOP_K * M);
    let mut up_ref = Vec::with_capacity(TOP_K * M);
    for expert in 0..TOP_K {
        gate_ref.extend((0..M).map(|row| dequant_dot(&weights[expert], row, &x, M * 2, K)));
        up_ref.extend((M..M * 2).map(|row| dequant_dot(&weights[expert], row, &x, M * 2, K)));
    }
    let mut ok = compare("gate", &download(&gpu, &gate, TOP_K * M), &gate_ref);
    ok &= compare("up", &download(&gpu, &up, TOP_K * M), &up_ref);

    let masked_ids = [0, -1, 2, -1, 4, -1];
    let masked_indices = upload_i32(&mut gpu, &masked_ids);
    let masked_gate = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("masked gate");
    let masked_up = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("masked up");
    gpu.deepseek4_gemv_mq3g256_lloyd_moe_gate_up_indexed(
        &expert_ptrs,
        &masked_indices,
        &x_gpu,
        &masked_gate,
        &masked_up,
        M * 2,
        K,
        TOP_K,
    )
    .expect("masked MQ3 gate/up");
    gpu.hip.device_synchronize().expect("masked gate/up sync");
    let mut masked_gate_ref = vec![0.0_f32; TOP_K * M];
    let mut masked_up_ref = vec![0.0_f32; TOP_K * M];
    for slot in [0, 2, 4] {
        masked_gate_ref[slot * M..(slot + 1) * M]
            .copy_from_slice(&gate_ref[slot * M..(slot + 1) * M]);
        masked_up_ref[slot * M..(slot + 1) * M].copy_from_slice(&up_ref[slot * M..(slot + 1) * M]);
    }
    ok &= compare(
        "masked_gate",
        &download(&gpu, &masked_gate, TOP_K * M),
        &masked_gate_ref,
    );
    ok &= compare(
        "masked_up",
        &download(&gpu, &masked_up, TOP_K * M),
        &masked_up_ref,
    );

    let rot_batch: Vec<f32> = (0..TOP_K)
        .flat_map(|slot| {
            (0..K).map(move |index| (((index * 13 + slot * 19 + 11) % 83) as f32 - 41.0) / 31.0)
        })
        .collect();
    let rot_gpu = upload_f32(&mut gpu, &rot_batch);
    let route_weights = [0.07_f32, 0.11, 0.16, 0.19, 0.21, 0.26];
    let route_weights_gpu = upload_f32(&mut gpu, &route_weights);
    let residual = upload_f32(&mut gpu, &vec![0.0_f32; M]);
    gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
        &expert_ptrs,
        &all_indices,
        &route_weights_gpu,
        &rot_gpu,
        &residual,
        M,
        K,
        TOP_K,
    )
    .expect("MQ3 down");
    gpu.hip.device_synchronize().expect("down sync");
    let down_ref: Vec<f32> = (0..M)
        .map(|row| {
            (0..TOP_K)
                .map(|slot| {
                    route_weights[slot]
                        * dequant_dot(
                            &weights[slot],
                            row,
                            &rot_batch[slot * K..(slot + 1) * K],
                            M * 2,
                            K,
                        )
                })
                .sum()
        })
        .collect();
    ok &= compare("down", &download(&gpu, &residual, M), &down_ref);

    let masked_residual = upload_f32(&mut gpu, &vec![0.0_f32; M]);
    gpu.deepseek4_gemv_mq3g256_lloyd_moe_down_residual_scaled_indexed(
        &expert_ptrs,
        &masked_indices,
        &route_weights_gpu,
        &rot_gpu,
        &masked_residual,
        M,
        K,
        TOP_K,
    )
    .expect("masked MQ3 down");
    gpu.hip.device_synchronize().expect("masked down sync");
    let masked_down_ref: Vec<f32> = (0..M)
        .map(|row| {
            [0, 2, 4]
                .into_iter()
                .map(|slot| {
                    route_weights[slot]
                        * dequant_dot(
                            &weights[slot],
                            row,
                            &rot_batch[slot * K..(slot + 1) * K],
                            M * 2,
                            K,
                        )
                })
                .sum()
        })
        .collect();
    ok &= compare(
        "masked_down",
        &download(&gpu, &masked_residual, M),
        &masked_down_ref,
    );

    assert!(ok, "gfx90a MQ3-Lloyd indexed MoE parity failed");
    println!("PASS gfx90a MQ3-Lloyd indexed MoE parity");
}
