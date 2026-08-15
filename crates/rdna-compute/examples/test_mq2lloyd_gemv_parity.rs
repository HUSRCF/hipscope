//! CPU/GPU parity probe for the MQ2-Lloyd GEMV kernels used by DeepSeek V4.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return (sign << 15) | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    if exponent > 142 {
        return (sign << 15) | 0x7c00;
    }
    if exponent >= 113 {
        let half_exp = (exponent - 112) as u16;
        let mut half_mantissa = (mantissa >> 13) as u16;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && half_mantissa & 1 != 0) {
            half_mantissa += 1;
            if half_mantissa == 0x0400 {
                return (sign << 15) | ((half_exp + 1) << 10);
            }
        }
        return (sign << 15) | (half_exp << 10) | half_mantissa;
    }
    if exponent >= 103 {
        let shift = (113 - exponent) as u32;
        let full = mantissa | 0x80_0000;
        let mut half_mantissa = (full >> (shift + 13)) as u16;
        let remainder_mask = (1_u32 << (shift + 13)) - 1;
        let remainder = full & remainder_mask;
        let halfway = 1_u32 << (shift + 12);
        if remainder > halfway || (remainder == halfway && half_mantissa & 1 != 0) {
            half_mantissa += 1;
        }
        return (sign << 15) | half_mantissa;
    }
    sign << 15
}

fn f16_value(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x03ff) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut mantissa = mantissa;
        let mut exponent = -14_i32;
        while mantissa & 0x400 == 0 {
            mantissa <<= 1;
            exponent -= 1;
        }
        return f32::from_bits(
            sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x3ff) << 13),
        );
    }
    if exponent == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | (((exponent as i32 - 15 + 127) as u32) << 23) | (mantissa << 13))
}

fn make_weights(m: usize, k: usize, seed: u32) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(m * (k / 256) * 72);
    for row in 0..m {
        for group in 0..k / 256 {
            let scale = 0.25 + ((row + group) % 7) as f32 * 0.03125;
            for value in [-3.0, -0.75, 0.5, 2.5] {
                out.extend_from_slice(&f16_bits(value * scale).to_le_bytes());
            }
            for _ in 0..64 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                out.push((state >> 24) as u8);
            }
        }
    }
    out
}

fn dequant_dot(weights: &[u8], row: usize, x: &[f32], m: usize, k: usize) -> f32 {
    assert!(row < m);
    let row_bytes = (k / 256) * 72;
    let mut sum = 0.0_f32;
    for group in 0..k / 256 {
        let base = row * row_bytes + group * 72;
        let codebook = std::array::from_fn::<_, 4, _>(|i| {
            f16_value(u16::from_le_bytes([
                weights[base + i * 2],
                weights[base + i * 2 + 1],
            ]))
        });
        for i in 0..256 {
            let packed = weights[base + 8 + i / 4];
            let code = ((packed >> (2 * (i % 4))) & 3) as usize;
            sum += codebook[code] * x[group * 256 + i];
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

fn download(gpu: &Gpu, tensor: &GpuTensor, len: usize) -> Vec<f32> {
    let mut values = vec![0.0_f32; len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), len * 4) };
    gpu.hip.memcpy_dtoh(bytes, &tensor.buf).expect("download");
    values
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) -> bool {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut bad = 0;
    for (&a, &e) in actual.iter().zip(expected) {
        let abs = (a - e).abs();
        let rel = abs / e.abs().max(1.0e-5);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        bad += usize::from(!a.is_finite() || (abs > 2.0e-3 && rel > 2.0e-4));
    }
    println!("{label}: max_abs={max_abs:.6e} max_rel={max_rel:.6e} bad={bad}");
    bad == 0
}

#[allow(non_snake_case)]
fn main() {
    let env_usize = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let M = env_usize("HIPFIRE_PARITY_M", 64);
    let K = env_usize("HIPFIRE_PARITY_K", 512);
    let TOP_K = env_usize("HIPFIRE_PARITY_TOP_K", 2);
    let ITERS = env_usize("HIPFIRE_PARITY_ITERS", 1_000);
    let BATCH = env_usize("HIPFIRE_PARITY_BATCH", 1);
    let N_EXP = env_usize("HIPFIRE_PARITY_N_EXP", 256);
    let GROUPED_ITERS = env_usize("HIPFIRE_PARITY_GROUPED_ITERS", 20);
    assert!(M % 2 == 0 && K % 256 == 0 && TOP_K >= 2 && BATCH >= 1);
    assert!(N_EXP >= TOP_K && GROUPED_ITERS >= 1);

    let mut gpu = Gpu::init().expect("Gpu::init");
    println!(
        "arch={} gate_total_m={} synthetic_down_m={M} K={K} top_k={TOP_K} batch={BATCH} n_exp={N_EXP}",
        gpu.arch,
        M * 2,
    );
    let weights: Vec<Vec<u8>> = (0..TOP_K)
        .map(|expert| make_weights(M * 2, K, 0x1234_5678 + expert as u32))
        .collect();
    let weight_tensors: Vec<GpuTensor> = weights
        .iter()
        .map(|w| upload(&mut gpu, w, DType::Raw))
        .collect();
    let ptr_bytes: Vec<u8> = weight_tensors
        .iter()
        .flat_map(|t| (t.buf.as_ptr() as u64).to_le_bytes())
        .collect();
    let expert_ptrs = upload(&mut gpu, &ptr_bytes, DType::Raw);
    let index_bytes: Vec<u8> = (0..BATCH)
        .flat_map(|_| 0..TOP_K as i32)
        .flat_map(i32::to_le_bytes)
        .collect();
    let indices = upload(&mut gpu, &index_bytes, DType::Raw);
    let x: Vec<f32> = (0..K).map(|i| ((i % 29) as f32 - 14.0) / 17.0).collect();
    let x_batch: Vec<f32> = (0..BATCH).flat_map(|_| x.iter().copied()).collect();
    let x_gpu = upload_f32(&mut gpu, &x_batch);
    let x_batch_f16: Vec<f32> = x_batch
        .iter()
        .map(|&value| f16_value(f16_bits(value)))
        .collect();
    let x_f16_gpu = upload_f32(&mut gpu, &x_batch_f16);

    let route_ptr_bytes: Vec<u8> = (0..N_EXP)
        .flat_map(|expert| {
            let ptr = weight_tensors[expert % TOP_K].buf.as_ptr() as u64;
            ptr.to_le_bytes()
        })
        .collect();
    let route_expert_ptrs = upload(&mut gpu, &route_ptr_bytes, DType::Raw);
    let route_indices_host: Vec<i32> = (0..BATCH * TOP_K)
        .map(|slot| ((slot / TOP_K * 17 + slot % TOP_K * 43) % N_EXP) as i32)
        .collect();
    let route_index_bytes: Vec<u8> = route_indices_host
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let route_indices = upload(&mut gpu, &route_index_bytes, DType::Raw);

    let plain_out = gpu.alloc_tensor(&[M], DType::F32).expect("plain out");
    gpu.gemv_mq2g256_lloyd(&weight_tensors[0], &x_gpu, &plain_out, M, K)
        .expect("plain GEMV");
    gpu.hip.device_synchronize().expect("plain sync");
    let plain_ref: Vec<f32> = (0..M)
        .map(|row| dequant_dot(&weights[0], row, &x, M * 2, K))
        .collect();
    let mut ok = compare("plain", &download(&gpu, &plain_out, M), &plain_ref);

    let gate = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("gate out");
    let up = gpu.alloc_tensor(&[TOP_K * M], DType::F32).expect("up out");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &expert_ptrs,
        &indices,
        &x_gpu,
        &gate,
        &up,
        M * 2,
        K,
        TOP_K,
    )
    .expect("indexed gate/up");
    gpu.hip.device_synchronize().expect("gate/up sync");
    let mut gate_ref = Vec::with_capacity(TOP_K * M);
    let mut up_ref = Vec::with_capacity(TOP_K * M);
    for expert in 0..TOP_K {
        gate_ref.extend((0..M).map(|row| dequant_dot(&weights[expert], row, &x, M * 2, K)));
        up_ref.extend((M..M * 2).map(|row| dequant_dot(&weights[expert], row, &x, M * 2, K)));
    }
    ok &= compare("indexed_gate", &download(&gpu, &gate, TOP_K * M), &gate_ref);
    ok &= compare("indexed_up", &download(&gpu, &up, TOP_K * M), &up_ref);

    // EP owner-mask channel: expert 0 is local, every other selected route is
    // remote. Routing weights stay global; only indices are replaced by -1.
    let ep_indices = upload(&mut gpu, &index_bytes, DType::Raw);
    let mut owned = vec![0_u8; TOP_K];
    owned[0] = 1;
    let owned_gpu = upload(&mut gpu, &owned, DType::Raw);
    gpu.deepseek4_mask_topk_owned(&ep_indices, &owned_gpu, TOP_K, TOP_K)
        .expect("mask EP top-k");
    let ep_gate = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("EP gate out");
    let ep_up = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("EP up out");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
        &expert_ptrs,
        &ep_indices,
        &x_gpu,
        &ep_gate,
        &ep_up,
        M * 2,
        K,
        TOP_K,
    )
    .expect("EP masked gate/up");
    gpu.hip.device_synchronize().expect("EP gate/up sync");
    let mut ep_gate_ref = vec![0.0; TOP_K * M];
    let mut ep_up_ref = vec![0.0; TOP_K * M];
    ep_gate_ref[..M].copy_from_slice(&gate_ref[..M]);
    ep_up_ref[..M].copy_from_slice(&up_ref[..M]);
    ok &= compare(
        "ep_masked_gate",
        &download(&gpu, &ep_gate, TOP_K * M),
        &ep_gate_ref,
    );
    ok &= compare(
        "ep_masked_up",
        &download(&gpu, &ep_up, TOP_K * M),
        &ep_up_ref,
    );

    let gate_k4 = gpu
        .alloc_tensor(&[BATCH * TOP_K * M], DType::F32)
        .expect("K4 gate out");
    let up_k4 = gpu
        .alloc_tensor(&[BATCH * TOP_K * M], DType::F32)
        .expect("K4 up out");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
        &expert_ptrs,
        &indices,
        &x_gpu,
        &gate_k4,
        &up_k4,
        M * 2,
        K,
        TOP_K,
        BATCH,
    )
    .expect("batched K4 gate/up");
    gpu.hip.device_synchronize().expect("K4 gate/up sync");
    let gate_k4_host = download(&gpu, &gate_k4, BATCH * TOP_K * M);
    let up_k4_host = download(&gpu, &up_k4, BATCH * TOP_K * M);
    let gate_k4_ref: Vec<f32> = (0..BATCH).flat_map(|_| gate_ref.iter().copied()).collect();
    let up_k4_ref: Vec<f32> = (0..BATCH).flat_map(|_| up_ref.iter().copied()).collect();
    ok &= compare("batched_k4_gate", &gate_k4_host, &gate_k4_ref);
    ok &= compare("batched_k4_up", &up_k4_host, &up_k4_ref);
    let k4_hash =
        gate_k4_host
            .iter()
            .chain(&up_k4_host)
            .fold(0xcbf29ce484222325_u64, |mut hash, value| {
                for byte in value.to_bits().to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
                }
                hash
            });
    println!("batched_k4_hash: {k4_hash:#018x}");

    let rot: Vec<f32> = (0..TOP_K)
        .flat_map(|expert| x.iter().map(move |v| v * (expert as f32 + 1.0)))
        .collect();
    let rot_gpu = upload_f32(&mut gpu, &rot);
    let scale_sum = (TOP_K * (TOP_K + 1) / 2) as f32;
    let scales: Vec<f32> = (1..=TOP_K).map(|i| i as f32 / scale_sum).collect();
    let scales_gpu = upload_f32(&mut gpu, &scales);
    let down = gpu.alloc_tensor(&[M], DType::F32).expect("down out");
    gpu.hip.memset(&down.buf, 0, M * 4).expect("zero down");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        &expert_ptrs,
        &indices,
        &scales_gpu,
        &rot_gpu,
        &down,
        M,
        K,
        TOP_K,
    )
    .expect("indexed down");
    gpu.hip.device_synchronize().expect("down sync");
    let down_ref: Vec<f32> = (0..M)
        .map(|row| {
            (0..TOP_K)
                .map(|expert| {
                    scales[expert]
                        * dequant_dot(
                            &weights[expert],
                            row,
                            &rot[expert * K..(expert + 1) * K],
                            M * 2,
                            K,
                        )
                })
                .sum()
        })
        .collect();
    ok &= compare("indexed_down", &download(&gpu, &down, M), &down_ref);

    let ep_down = gpu.alloc_tensor(&[M], DType::F32).expect("EP down out");
    gpu.hip
        .memset(&ep_down.buf, 0, M * 4)
        .expect("zero EP down");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed(
        &expert_ptrs,
        &ep_indices,
        &scales_gpu,
        &rot_gpu,
        &ep_down,
        M,
        K,
        TOP_K,
    )
    .expect("EP masked indexed down");
    gpu.hip.device_synchronize().expect("EP down sync");
    let ep_down_ref: Vec<f32> = (0..M)
        .map(|row| scales[0] * dequant_dot(&weights[0], row, &rot[..K], M * 2, K))
        .collect();
    ok &= compare(
        "ep_masked_indexed_down",
        &download(&gpu, &ep_down, M),
        &ep_down_ref,
    );

    // The row2 ranged kernels require even row_base and row_count. Preserve a
    // three-way coverage split while aligning both interior boundaries.
    let split1 = (M / 3) & !1;
    let split2 = (2 * M / 3) & !1;
    assert!(split1 > 0 && split2 > split1 && split2 < M);
    let row_splits = [0, split1, split2, M];
    let down_rows = gpu.alloc_tensor(&[M], DType::F32).expect("ranged down");
    gpu.hip
        .memset(&down_rows.buf, 0, M * 4)
        .expect("zero ranged down");
    for rows in row_splits.windows(2) {
        let row_base = rows[0];
        let row_count = rows[1] - row_base;
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_residual_scaled_indexed_rows(
            &expert_ptrs,
            &indices,
            &scales_gpu,
            &rot_gpu,
            &down_rows,
            M,
            K,
            TOP_K,
            row_base,
            row_count,
        )
        .expect("indexed ranged down");
    }
    gpu.hip.device_synchronize().expect("ranged down sync");
    ok &= compare(
        "indexed_down_rows",
        &download(&gpu, &down_rows, M),
        &down_ref,
    );

    let expanded = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("expanded down");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
        &expert_ptrs,
        &indices,
        &rot_gpu,
        &expanded,
        M,
        K,
        TOP_K,
        1,
    )
    .expect("expanded down");
    gpu.hip.device_synchronize().expect("expanded sync");
    let mut expanded_ref = Vec::with_capacity(TOP_K * M);
    for expert in 0..TOP_K {
        for row in 0..M {
            expanded_ref.push(dequant_dot(
                &weights[expert],
                row,
                &rot[expert * K..(expert + 1) * K],
                M * 2,
                K,
            ));
        }
    }
    ok &= compare(
        "expanded_down",
        &download(&gpu, &expanded, TOP_K * M),
        &expanded_ref,
    );

    let ep_expanded = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("EP expanded down");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
        &expert_ptrs,
        &ep_indices,
        &rot_gpu,
        &ep_expanded,
        M,
        K,
        TOP_K,
        1,
    )
    .expect("EP masked expanded down");
    gpu.hip.device_synchronize().expect("EP expanded sync");
    let mut ep_expanded_ref = vec![0.0; TOP_K * M];
    ep_expanded_ref[..M].copy_from_slice(&expanded_ref[..M]);
    ok &= compare(
        "ep_masked_expanded_down",
        &download(&gpu, &ep_expanded, TOP_K * M),
        &ep_expanded_ref,
    );

    let expanded_rows = gpu
        .alloc_tensor(&[TOP_K * M], DType::F32)
        .expect("ranged expanded down");
    for rows in row_splits.windows(2) {
        let row_base = rows[0];
        let row_count = rows[1] - row_base;
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4_rows(
            &expert_ptrs,
            &indices,
            &rot_gpu,
            &expanded_rows,
            M,
            K,
            TOP_K,
            1,
            row_base,
            row_count,
        )
        .expect("ranged expanded down");
    }
    gpu.hip.device_synchronize().expect("ranged expanded sync");
    ok &= compare(
        "expanded_down_rows",
        &download(&gpu, &expanded_rows, TOP_K * M),
        &expanded_ref,
    );

    let combined_rows = gpu.alloc_tensor(&[M], DType::F32).expect("ranged combine");
    gpu.hip
        .memset(&combined_rows.buf, 0, M * 4)
        .expect("zero ranged combine");
    for rows in row_splits.windows(2) {
        let row_base = rows[0];
        let row_count = rows[1] - row_base;
        gpu.moe_down_combine_k8_batched_rows(
            &expanded_rows,
            &scales_gpu,
            &combined_rows,
            M,
            TOP_K,
            1,
            row_base,
            row_count,
        )
        .expect("ranged combine");
    }
    gpu.hip.device_synchronize().expect("ranged combine sync");
    let combined_host = download(&gpu, &combined_rows, M);
    ok &= compare("expanded_combine_rows", &combined_host, &down_ref);

    let fused_rows = gpu.alloc_tensor(&[M], DType::F32).expect("fused row2");
    gpu.hip
        .memset(&fused_rows.buf, 0, M * 4)
        .expect("zero fused row2");
    for rows in row_splits.windows(2) {
        let row_base = rows[0];
        let row_count = rows[1] - row_base;
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_fused_rows(
            &expert_ptrs,
            &indices,
            &scales_gpu,
            &rot_gpu,
            &fused_rows,
            M,
            K,
            TOP_K,
            row_base,
            row_count,
        )
        .expect("fused row2");
    }
    gpu.hip.device_synchronize().expect("fused row2 sync");
    let fused_host = download(&gpu, &fused_rows, M);
    ok &= compare("fused_down_row2_rows", &fused_host, &down_ref);
    let (bit_bad, max_abs) = fused_host.iter().zip(&combined_host).fold(
        (0usize, 0.0f32),
        |(bad, max_abs), (&fused, &expanded)| {
            (
                bad + usize::from(fused.to_bits() != expanded.to_bits()),
                max_abs.max((fused - expanded).abs()),
            )
        },
    );
    println!("fused_vs_expanded_bits: bad={bit_bad}/{M} max_abs={max_abs:.6e}");

    const SLOTS: usize = 32;
    let grouped_x: Vec<f32> = (0..SLOTS * K)
        .map(|i| ((i % 37) as f32 - 18.0) / 23.0)
        .collect();
    let grouped_x_f16: Vec<f32> = grouped_x
        .iter()
        .map(|&value| f16_value(f16_bits(value)))
        .collect();
    let grouped_x_gpu = upload_f32(&mut gpu, &grouped_x);
    let tile_bytes: Vec<u8> = [0_i32, 1_i32]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    let tile_ids = upload(&mut gpu, &tile_bytes, DType::Raw);
    let slot_bytes: Vec<u8> = (0..SLOTS as i32).flat_map(i32::to_le_bytes).collect();
    let slot_indices = upload(&mut gpu, &slot_bytes, DType::Raw);
    let grouped_out = gpu
        .alloc_tensor(&[SLOTS * M], DType::F32)
        .expect("grouped out");
    gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx90a(
        &expert_ptrs,
        &tile_ids,
        &slot_indices,
        &grouped_x_gpu,
        &grouped_out,
        M,
        K,
        1,
        SLOTS,
        SLOTS,
    )
    .expect("grouped MFMA");
    gpu.hip.device_synchronize().expect("grouped sync");
    let mut grouped_ref = Vec::with_capacity(SLOTS * M);
    for slot in 0..SLOTS {
        let expert = slot / 16;
        let x = &grouped_x_f16[slot * K..(slot + 1) * K];
        for row in 0..M {
            grouped_ref.push(dequant_dot(&weights[expert], row, x, M * 2, K));
        }
    }
    ok &= compare(
        "grouped_mfma",
        &download(&gpu, &grouped_out, SLOTS * M),
        &grouped_ref,
    );

    const GROUP_TILE: usize = 16;
    let total_routes = BATCH * TOP_K;
    let live_expert_bound = total_routes.min(N_EXP);
    let grouped_bound =
        (total_routes + live_expert_bound * (GROUP_TILE - 1)).div_ceil(GROUP_TILE) * GROUP_TILE;
    let mut route_counts = vec![0usize; N_EXP];
    for &expert in &route_indices_host {
        route_counts[expert as usize] += 1;
    }
    let active_experts = route_counts.iter().filter(|&&count| count != 0).count();
    let live_rows: usize = route_counts
        .iter()
        .map(|&count| count.div_ceil(GROUP_TILE) * GROUP_TILE)
        .sum();
    println!(
        "grouped_route_bound: routes={total_routes} active={active_experts} live_rows={live_rows} bound={grouped_bound} old_bound={}",
        total_routes + N_EXP * GROUP_TILE
    );

    let route_gate = gpu
        .alloc_tensor(&[total_routes * M], DType::F32)
        .expect("route gate");
    let route_up = gpu
        .alloc_tensor(&[total_routes * M], DType::F32)
        .expect("route up");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
        &route_expert_ptrs,
        &route_indices,
        &x_f16_gpu,
        &route_gate,
        &route_up,
        M * 2,
        K,
        TOP_K,
        BATCH,
    )
    .expect("route scalar K4");

    let expert_counts = gpu
        .alloc_tensor(&[N_EXP], DType::F32)
        .expect("expert counts");
    let expert_offsets = gpu
        .alloc_tensor(&[N_EXP + 1], DType::F32)
        .expect("expert offsets");
    let sorted_slots = gpu
        .alloc_tensor(&[grouped_bound], DType::F32)
        .expect("sorted slots");
    let grouped_tile_ids = gpu
        .alloc_tensor(&[grouped_bound / GROUP_TILE], DType::F32)
        .expect("grouped tile ids");
    let inverse_perm = gpu
        .alloc_tensor(&[total_routes], DType::F32)
        .expect("inverse perm");
    let route_grouped = gpu
        .alloc_tensor(&[grouped_bound * M * 2], DType::F32)
        .expect("route grouped");
    let grouped_gate = gpu
        .alloc_tensor(&[total_routes * M], DType::F32)
        .expect("grouped gate");
    let grouped_up = gpu
        .alloc_tensor(&[total_routes * M], DType::F32)
        .expect("grouped up");

    gpu.moe_scatter_fused_k8(
        &route_indices,
        &expert_counts,
        &expert_offsets,
        &sorted_slots,
        &grouped_tile_ids,
        &inverse_perm,
        total_routes,
        N_EXP,
        grouped_bound,
        GROUP_TILE,
    )
    .expect("grouped scatter");
    let x_ptr = x_gpu.buf.as_ptr();
    gpu.scratch.invalidate_x_caches_for(x_ptr);
    gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx90a(
        &route_expert_ptrs,
        &grouped_tile_ids,
        &sorted_slots,
        &x_gpu,
        &route_grouped,
        M * 2,
        K,
        TOP_K,
        grouped_bound,
        BATCH,
    )
    .expect("route grouped MFMA");
    gpu.moe_gate_up_unscatter_k8(
        &route_grouped,
        &sorted_slots,
        &grouped_gate,
        &grouped_up,
        M,
        TOP_K,
        grouped_bound,
    )
    .expect("route grouped unscatter");
    gpu.hip.device_synchronize().expect("route grouped sync");
    ok &= compare(
        "route_grouped_gate",
        &download(&gpu, &grouped_gate, total_routes * M),
        &download(&gpu, &route_gate, total_routes * M),
    );
    ok &= compare(
        "route_grouped_up",
        &download(&gpu, &grouped_up, total_routes * M),
        &download(&gpu, &route_up, total_routes * M),
    );

    let started = Instant::now();
    for _ in 0..GROUPED_ITERS {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
            &route_expert_ptrs,
            &route_indices,
            &x_gpu,
            &route_gate,
            &route_up,
            M * 2,
            K,
            TOP_K,
            BATCH,
        )
        .expect("bench route scalar K4");
    }
    gpu.hip
        .device_synchronize()
        .expect("bench route scalar sync");
    println!(
        "bench_route_scalar_k4: {:.3} us/iteration",
        started.elapsed().as_secs_f64() * 1.0e6 / GROUPED_ITERS as f64
    );

    let started = Instant::now();
    for _ in 0..GROUPED_ITERS {
        gpu.moe_scatter_fused_k8(
            &route_indices,
            &expert_counts,
            &expert_offsets,
            &sorted_slots,
            &grouped_tile_ids,
            &inverse_perm,
            total_routes,
            N_EXP,
            grouped_bound,
            GROUP_TILE,
        )
        .expect("bench grouped scatter");
        gpu.scratch.invalidate_x_caches_for(x_ptr);
        gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx90a(
            &route_expert_ptrs,
            &grouped_tile_ids,
            &sorted_slots,
            &x_gpu,
            &route_grouped,
            M * 2,
            K,
            TOP_K,
            grouped_bound,
            BATCH,
        )
        .expect("bench grouped MFMA");
        gpu.moe_gate_up_unscatter_k8(
            &route_grouped,
            &sorted_slots,
            &grouped_gate,
            &grouped_up,
            M,
            TOP_K,
            grouped_bound,
        )
        .expect("bench grouped unscatter");
    }
    gpu.hip
        .device_synchronize()
        .expect("bench grouped route sync");
    println!(
        "bench_route_grouped_mfma: {:.3} us/iteration",
        started.elapsed().as_secs_f64() * 1.0e6 / GROUPED_ITERS as f64
    );

    gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
        &route_expert_ptrs,
        &route_indices,
        &x_gpu,
        &route_gate,
        &route_up,
        M * 2,
        K,
        TOP_K,
        BATCH,
    )
    .expect("prepare route gate/up");
    gpu.deepseek4_silu_mul_clamp_f32_batched(
        &route_gate,
        &route_up,
        &route_gate,
        M,
        total_routes,
        7.0,
    )
    .expect("prepare route silu");
    let route_rot = gpu
        .alloc_tensor(&[total_routes * M], DType::F32)
        .expect("route rot");
    gpu.rotate_x_mq_batched(&route_gate, &route_rot, M, total_routes)
        .expect("prepare route rotate");
    gpu.hip
        .device_synchronize()
        .expect("prepare route rot sync");
    let route_rot_f16_host: Vec<f32> = download(&gpu, &route_rot, total_routes * M)
        .into_iter()
        .map(|value| f16_value(f16_bits(value)))
        .collect();
    let route_rot_f16 = upload_f32(&mut gpu, &route_rot_f16_host);
    let topk_weight_host = vec![1.0f32 / TOP_K as f32; total_routes];
    let route_topk_weights = upload_f32(&mut gpu, &topk_weight_host);
    let route_down_expanded = gpu
        .alloc_tensor(&[total_routes * M], DType::F32)
        .expect("route down expanded");
    let route_down_scalar = gpu
        .alloc_tensor(&[BATCH * M], DType::F32)
        .expect("route down scalar");
    let route_down_grouped = gpu
        .alloc_tensor(&[grouped_bound * M], DType::F32)
        .expect("route down grouped");
    let route_down_mfma = gpu
        .alloc_tensor(&[BATCH * M], DType::F32)
        .expect("route down MFMA");

    gpu.hip
        .memset(&route_down_scalar.buf, 0, BATCH * M * 4)
        .expect("zero route scalar down");
    gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
        &route_expert_ptrs,
        &route_indices,
        &route_rot_f16,
        &route_down_expanded,
        M,
        M,
        TOP_K,
        BATCH,
    )
    .expect("route scalar down");
    gpu.moe_down_combine_k8_batched(
        &route_down_expanded,
        &route_topk_weights,
        &route_down_scalar,
        M,
        TOP_K,
        BATCH,
    )
    .expect("route scalar combine");

    gpu.hip
        .memset(&route_down_mfma.buf, 0, BATCH * M * 4)
        .expect("zero route MFMA down");
    let route_rot_f16_ptr = route_rot_f16.buf.as_ptr();
    gpu.scratch.invalidate_x_caches_for(route_rot_f16_ptr);
    gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx90a(
        &route_expert_ptrs,
        &grouped_tile_ids,
        &sorted_slots,
        &route_rot_f16,
        &route_down_grouped,
        M,
        M,
        1,
        grouped_bound,
        total_routes,
    )
    .expect("route grouped down");
    gpu.moe_down_combine_grouped_k8(
        &route_down_grouped,
        &inverse_perm,
        &route_topk_weights,
        &route_down_mfma,
        M,
        TOP_K,
        BATCH,
    )
    .expect("route grouped combine");
    gpu.hip
        .device_synchronize()
        .expect("route down parity sync");
    ok &= compare(
        "route_grouped_down",
        &download(&gpu, &route_down_mfma, BATCH * M),
        &download(&gpu, &route_down_scalar, BATCH * M),
    );

    let started = Instant::now();
    for _ in 0..GROUPED_ITERS {
        gpu.hip
            .memset(&route_down_scalar.buf, 0, BATCH * M * 4)
            .expect("bench zero scalar down");
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
            &route_expert_ptrs,
            &route_indices,
            &route_rot,
            &route_down_expanded,
            M,
            M,
            TOP_K,
            BATCH,
        )
        .expect("bench route scalar down");
        gpu.moe_down_combine_k8_batched(
            &route_down_expanded,
            &route_topk_weights,
            &route_down_scalar,
            M,
            TOP_K,
            BATCH,
        )
        .expect("bench route scalar combine");
    }
    gpu.hip
        .device_synchronize()
        .expect("bench scalar down sync");
    println!(
        "bench_route_scalar_down: {:.3} us/iteration",
        started.elapsed().as_secs_f64() * 1.0e6 / GROUPED_ITERS as f64
    );

    let route_rot_ptr = route_rot.buf.as_ptr();
    let started = Instant::now();
    for _ in 0..GROUPED_ITERS {
        gpu.hip
            .memset(&route_down_mfma.buf, 0, BATCH * M * 4)
            .expect("bench zero grouped down");
        gpu.scratch.invalidate_x_caches_for(route_rot_ptr);
        gpu.gemm_mq2g256_lloyd_moe_grouped_mfma_gfx90a(
            &route_expert_ptrs,
            &grouped_tile_ids,
            &sorted_slots,
            &route_rot,
            &route_down_grouped,
            M,
            M,
            1,
            grouped_bound,
            total_routes,
        )
        .expect("bench route grouped down");
        gpu.moe_down_combine_grouped_k8(
            &route_down_grouped,
            &inverse_perm,
            &route_topk_weights,
            &route_down_mfma,
            M,
            TOP_K,
            BATCH,
        )
        .expect("bench route grouped combine");
    }
    gpu.hip
        .device_synchronize()
        .expect("bench grouped down sync");
    println!(
        "bench_route_grouped_down: {:.3} us/iteration",
        started.elapsed().as_secs_f64() * 1.0e6 / GROUPED_ITERS as f64
    );

    let started = Instant::now();
    for _ in 0..ITERS {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed(
            &expert_ptrs,
            &indices,
            &x_gpu,
            &gate,
            &up,
            M * 2,
            K,
            TOP_K,
        )
        .expect("bench gate/up");
    }
    gpu.hip.device_synchronize().expect("bench gate/up sync");
    println!(
        "bench_gate_up: {:.3} us/launch",
        started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64
    );

    let started = Instant::now();
    for _ in 0..ITERS {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_gate_up_indexed_batched_k4(
            &expert_ptrs,
            &indices,
            &x_gpu,
            &gate_k4,
            &up_k4,
            M * 2,
            K,
            TOP_K,
            BATCH,
        )
        .expect("bench batched K4 gate/up");
    }
    gpu.hip
        .device_synchronize()
        .expect("bench batched K4 gate/up sync");
    println!(
        "bench_batched_k4_gate_up: {:.3} us/launch",
        started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64
    );
    let started = Instant::now();
    for _ in 0..ITERS {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4(
            &expert_ptrs,
            &indices,
            &rot_gpu,
            &expanded,
            M,
            K,
            TOP_K,
            1,
        )
        .expect("bench expanded down");
    }
    gpu.hip.device_synchronize().expect("bench expanded sync");
    println!(
        "bench_expanded_down: {:.3} us/launch",
        started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64
    );

    let started = Instant::now();
    for _ in 0..ITERS {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_expanded_k4_rows(
            &expert_ptrs,
            &indices,
            &rot_gpu,
            &expanded,
            M,
            K,
            TOP_K,
            1,
            0,
            M,
        )
        .expect("bench deterministic expanded");
        gpu.moe_down_combine_k8_batched_rows(
            &expanded,
            &scales_gpu,
            &combined_rows,
            M,
            TOP_K,
            1,
            0,
            M,
        )
        .expect("bench deterministic combine");
    }
    gpu.hip
        .device_synchronize()
        .expect("bench deterministic sync");
    println!(
        "bench_deterministic_down_pair: {:.3} us/iteration",
        started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64
    );

    let started = Instant::now();
    for _ in 0..ITERS {
        gpu.deepseek4_gemv_mq2g256_lloyd_moe_down_fused_rows(
            &expert_ptrs,
            &indices,
            &scales_gpu,
            &rot_gpu,
            &fused_rows,
            M,
            K,
            TOP_K,
            0,
            M,
        )
        .expect("bench fused row2");
    }
    gpu.hip.device_synchronize().expect("bench fused sync");
    println!(
        "bench_fused_down_row2: {:.3} us/iteration",
        started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64
    );

    assert!(ok, "MQ2-Lloyd gfx90a parity failed");
}
