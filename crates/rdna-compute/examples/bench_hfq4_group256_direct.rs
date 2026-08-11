// SPDX-License-Identifier: MIT OR Apache-2.0
//! Focused gfx11 group128-LDS versus group256-direct activation probe.

use rdna_compute::Gpu;
use std::time::Instant;

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        return (sign << 15) | (0x1f << 10) | if mant != 0 { 0x200 } else { 0 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp < 1 {
        return sign << 15;
    }
    if half_exp > 30 {
        return (sign << 15) | (0x1f << 10);
    }
    let mut half_mant = (mant >> 13) as u16;
    let remainder = mant & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && (half_mant & 1) != 0) {
        half_mant += 1;
    }
    let mut half_exp = half_exp as u16;
    if half_mant == 0x400 {
        half_mant = 0;
        half_exp += 1;
    }
    (sign << 15) | (half_exp << 10) | half_mant
}

fn build_hfq4g256(m: usize, k: usize, zero_metadata: bool) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13) % 97) as f32 * 0.0001;
            let zero = if zero_metadata {
                0.0
            } else {
                ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015
            };
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[off + 8 + byte] = ((row * 29 + group * 19 + byte * 23) & 0xff) as u8;
            }
        }
    }
    out
}

fn build_hfq4g256_planar(m: usize, k: usize, zero_metadata: bool) -> Vec<u8> {
    let groups = k / 256;
    let group_count = m * groups;
    let mut out = vec![0u8; group_count * 136];
    let metadata_base = group_count * 128;
    for row in 0..m {
        for group in 0..groups {
            let index = row * groups + group;
            let payload_off = index * 128;
            let metadata_off = metadata_base + index * 8;
            let scale = 0.01 + ((row * 17 + group * 13) % 97) as f32 * 0.0001;
            let zero = if zero_metadata {
                0.0
            } else {
                ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015
            };
            out[metadata_off..metadata_off + 4].copy_from_slice(&scale.to_le_bytes());
            out[metadata_off + 4..metadata_off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[payload_off + byte] = ((row * 29 + group * 19 + byte * 23) & 0xff) as u8;
            }
        }
    }
    out
}

fn build_hfq4g256_tile64_planar(m: usize, k: usize, zero_metadata: bool) -> Vec<u8> {
    assert!(m % 64 == 0, "tile64 planar weights require M%64=0");
    let groups = k / 256;
    let group_count = m * groups;
    let mut out = vec![0u8; group_count * 136];
    let metadata_base = group_count * 128;
    for row in 0..m {
        for group in 0..groups {
            let index = ((row / 64) * groups + group) * 64 + row % 64;
            let payload_off = index * 128;
            let metadata_off = metadata_base + index * 8;
            let scale = 0.01 + ((row * 17 + group * 13) % 97) as f32 * 0.0001;
            let zero = if zero_metadata {
                0.0
            } else {
                ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015
            };
            out[metadata_off..metadata_off + 4].copy_from_slice(&scale.to_le_bytes());
            out[metadata_off + 4..metadata_off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[payload_off + byte] = ((row * 29 + group * 19 + byte * 23) & 0xff) as u8;
            }
        }
    }
    out
}

fn build_hfq4g256_half_meta(m: usize, k: usize, zero_metadata: bool) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13) % 97) as f32 * 0.0001;
            let zero = if zero_metadata {
                0.0
            } else {
                ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015
            };
            out[off..off + 2].copy_from_slice(&f32_to_f16_bits(scale).to_le_bytes());
            out[off + 2..off + 4].copy_from_slice(&f32_to_f16_bits(zero).to_le_bytes());
            for byte in 0..128 {
                out[off + 8 + byte] = ((row * 29 + group * 19 + byte * 23) & 0xff) as u8;
            }
        }
    }
    out
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn main() {
    let mut m = 17_408usize;
    let mut k = 5_120usize;
    let mut n = 2_048usize;
    let mut pairs = 15usize;
    let mut skip_correctness = false;
    let mut staged = false;
    let mut serial_row = false;
    let mut group128_serial_row = false;
    let mut group128_direct = false;
    let mut group128_direct_x512 = false;
    let mut group128_x128y128 = false;
    let mut group128_x192y96 = false;
    let mut group128_n2_reuse = false;
    let mut group128_n4_reuse = false;
    let mut direct_packed_weight_n2 = false;
    let mut packed_weight_y64 = false;
    let mut packed_weight_x128y64 = false;
    let mut group128_dual_row_u32x2 = false;
    let mut group128_dual_row_scalar2 = false;
    let mut group128_quad_row_u32x2 = false;
    let mut group128_interleave_row_wmma = false;
    let mut group128_quad_row_min1 = false;
    let mut group128_quad_row_vector_activation = false;
    let mut group128_quad_row_vector_activation_batch3 = false;
    let mut group128_warp_specialized_stage = false;
    let mut group128_prefetch_next = false;
    let mut group128_oct_row_u32x2 = false;
    let mut group128_planar_quad_row_uint4 = false;
    let mut group128_tile64_planar_quad_uint4 = false;
    let mut group128_f16_accum = false;
    let mut group128_half_meta = false;
    let mut f32a_k32 = false;
    let mut f32a_k32_unique_decode = false;
    let mut f32a_k32_compact_decode = false;
    let mut f32a_k32_compact_perm_decode = false;
    let mut f32a_k64 = false;
    let mut f32a_k64_compact_decode = false;
    let mut group128_k32_stationary = false;
    let mut skip_zero = false;
    let mut zero_metadata = false;
    let mut stream_k128 = false;
    let mut stream_k128_phased_x256 = false;
    let mut stream_k128_x256y128 = false;
    let mut add = false;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--m" => {
                i += 1;
                m = args[i].parse().expect("valid --m");
            }
            "--k" => {
                i += 1;
                k = args[i].parse().expect("valid --k");
            }
            "--n" => {
                i += 1;
                n = args[i].parse().expect("valid --n");
            }
            "--pairs" => {
                i += 1;
                pairs = args[i].parse().expect("valid --pairs");
            }
            "--skip-correctness" => skip_correctness = true,
            "--staged" => staged = true,
            "--serial-row" => serial_row = true,
            "--group128-serial-row" => group128_serial_row = true,
            "--group128-direct" => group128_direct = true,
            "--group128-direct-x512" => group128_direct_x512 = true,
            "--group128-x128y128" => group128_x128y128 = true,
            "--group128-x192y96" => group128_x192y96 = true,
            "--group128-n2-reuse" => group128_n2_reuse = true,
            "--group128-n4-reuse" => group128_n4_reuse = true,
            "--direct-packed-weight-n2" => direct_packed_weight_n2 = true,
            "--packed-weight-y64" => packed_weight_y64 = true,
            "--packed-weight-x128y64" => packed_weight_x128y64 = true,
            "--group128-dual-row-u32x2" => group128_dual_row_u32x2 = true,
            "--group128-dual-row-scalar2" => group128_dual_row_scalar2 = true,
            "--group128-quad-row-u32x2" => group128_quad_row_u32x2 = true,
            "--group128-interleave-row-wmma" => group128_interleave_row_wmma = true,
            "--group128-quad-row-min1" => group128_quad_row_min1 = true,
            "--group128-quad-row-vector-activation" => group128_quad_row_vector_activation = true,
            "--group128-quad-row-vector-activation-batch3" => {
                group128_quad_row_vector_activation_batch3 = true
            }
            "--group128-warp-specialized-stage" => group128_warp_specialized_stage = true,
            "--group128-prefetch-next" => group128_prefetch_next = true,
            "--group128-oct-row-u32x2" => group128_oct_row_u32x2 = true,
            "--group128-planar-quad-row-uint4" => group128_planar_quad_row_uint4 = true,
            "--group128-tile64-planar-quad-uint4" => group128_tile64_planar_quad_uint4 = true,
            "--group128-f16-accum" => group128_f16_accum = true,
            "--group128-half-meta" => group128_half_meta = true,
            "--f32a-k32" => f32a_k32 = true,
            "--f32a-k32-unique-decode" => f32a_k32_unique_decode = true,
            "--f32a-k32-compact-decode" => f32a_k32_compact_decode = true,
            "--f32a-k32-compact-perm-decode" => f32a_k32_compact_perm_decode = true,
            "--f32a-k64" => f32a_k64 = true,
            "--f32a-k64-compact-decode" => f32a_k64_compact_decode = true,
            "--group128-k32-stationary" => group128_k32_stationary = true,
            "--skip-zero" => skip_zero = true,
            "--zero-metadata" => zero_metadata = true,
            "--stream-k128" => stream_k128 = true,
            "--stream-k128-phased-x256" => stream_k128_phased_x256 = true,
            "--stream-k128-x256y128" => stream_k128_x256y128 = true,
            "--add" => add = true,
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    // The X128 candidate accepts N%128, but this paired benchmark deliberately
    // retains N%256 because its production X256 reference requires that shape.
    assert!(m % 64 == 0 && k % 256 == 0 && n % 256 == 0);
    assert!(
        [
            staged,
            serial_row,
            group128_serial_row,
            group128_direct,
            group128_direct_x512,
            group128_x128y128,
            group128_x192y96,
            group128_n2_reuse,
            group128_n4_reuse,
            direct_packed_weight_n2,
            packed_weight_y64,
            packed_weight_x128y64,
            group128_dual_row_u32x2,
            group128_dual_row_scalar2,
            group128_quad_row_u32x2,
            group128_interleave_row_wmma,
            group128_quad_row_min1,
            group128_quad_row_vector_activation,
            group128_quad_row_vector_activation_batch3,
            group128_warp_specialized_stage,
            group128_prefetch_next,
            group128_oct_row_u32x2,
            group128_planar_quad_row_uint4,
            group128_tile64_planar_quad_uint4,
            group128_f16_accum,
            group128_half_meta,
            f32a_k32,
            f32a_k32_unique_decode,
            f32a_k32_compact_decode,
            f32a_k32_compact_perm_decode,
            f32a_k64,
            f32a_k64_compact_decode,
            group128_k32_stationary,
            skip_zero,
            stream_k128,
            stream_k128_phased_x256,
            stream_k128_x256y128,
        ]
        .into_iter()
        .filter(|enabled| *enabled)
        .count()
            <= 1,
        "choose only one candidate variant"
    );

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!("arch={} M={m} K={k} N={n} pairs={pairs}", gpu.arch);
    let a = gpu
        .upload_raw(&build_hfq4g256(m, k, zero_metadata), &[m, k])
        .expect("upload A");
    let a_half_meta = group128_half_meta.then(|| {
        gpu.upload_raw(&build_hfq4g256_half_meta(m, k, zero_metadata), &[m, k])
            .expect("upload half-meta A")
    });
    let a_planar = group128_planar_quad_row_uint4.then(|| {
        gpu.upload_raw(&build_hfq4g256_planar(m, k, zero_metadata), &[m, k])
            .expect("upload planar A")
    });
    let a_tile64_planar = group128_tile64_planar_quad_uint4.then(|| {
        gpu.upload_raw(&build_hfq4g256_tile64_planar(m, k, zero_metadata), &[m, k])
            .expect("upload tile64 planar A")
    });
    let x_host: Vec<f32> = (0..n * k)
        .map(|idx| ((idx * 17 + idx / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let zeros = vec![0.0f32; n * m];
    let y128 = gpu.upload_f32(&zeros, &[n, m]).expect("group128 output");
    let y256 = gpu.upload_f32(&zeros, &[n, m]).expect("group256 output");
    let xq128_storage = gpu.hip.malloc((k / 128) * n * 144).expect("Xq128");
    let xq256_storage = gpu.hip.malloc((k / 128) * n * 144).expect("Xq256");
    let xq128 = xq128_storage.as_ptr();
    let xq256 = xq256_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, xq128, n, k)
        .expect("quantize group128");
    gpu.quantize_q8_1_mmq_group256_into(&x, xq256, n, k)
        .expect("quantize group256");

    let run128 = |gpu: &mut Gpu| {
        if f32a_k32_unique_decode
            || f32a_k32_compact_decode
            || f32a_k32_compact_perm_decode
            || f32a_k64
            || f32a_k64_compact_decode
        {
            if f32a_k32_compact_perm_decode {
                gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_compact_decode(&a, &x, &y128, m, k, n, add)
            } else if add {
                gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_add(&a, &x, &y128, m, k, n)
            } else {
                gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_set(&a, &x, &y128, m, k, n)
            }
        } else if add {
            gpu.gemm_hfq4g256_mmq_add_prequant_x256y64_perm_group128(&a, xq128, &y128, m, k, n)
        } else {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(&a, xq128, &y128, m, k, n)
        }
    };
    let run_candidate = |gpu: &mut Gpu| {
        if stream_k128_x256y128 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y128_stream_k128(&a, xq128, &y256, m, k, n, add)
        } else if stream_k128_phased_x256 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_stream_k128_phased(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if stream_k128 {
            gpu.gemm_hfq4g256_mmq_prequant_x128y64_stream_k128(&a, xq128, &y256, m, k, n, add)
        } else if skip_zero {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_skip_zero(&a, xq128, &y256, m, k, n, add)
        } else if group128_k32_stationary {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_k32_stationary(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_direct_x512 {
            gpu.gemm_hfq4g256_mmq_prequant_x512y64_group128_direct(&a, xq128, &y256, m, k, n, add)
        } else if group128_x128y128 {
            gpu.gemm_hfq4g256_mmq_prequant_x128y128_perm_group128(&a, xq128, &y256, m, k, n, add)
        } else if group128_x192y96 {
            gpu.gemm_hfq4g256_mmq_prequant_x192y96_group128_quad_row(&a, xq128, &y256, m, k, n, add)
        } else if group128_n2_reuse {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_n_reuse(
                &a, xq128, &y256, m, k, n, add, 2,
            )
        } else if group128_n4_reuse {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_n_reuse(
                &a, xq128, &y256, m, k, n, add, 4,
            )
        } else if direct_packed_weight_n2 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_direct_packed_weight_n2(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if packed_weight_y64 {
            gpu.gemm_hfq4g256_mmq_packed_weight_y64_group128(&a, xq128, &y256, m, k, n, add)
        } else if packed_weight_x128y64 {
            gpu.gemm_hfq4g256_mmq_packed_weight_x128y64_group128(&a, xq128, &y256, m, k, n, add)
        } else if group128_dual_row_u32x2 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_dual_row_u32x2(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_dual_row_scalar2 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_dual_row_scalar2(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_quad_row_u32x2 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_interleave_row_wmma {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_interleave_row_wmma(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_quad_row_min1 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_min1(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_quad_row_vector_activation {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_vector_activation(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_quad_row_vector_activation_batch3 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_vector_activation_batch3(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_warp_specialized_stage {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_warp_specialized_stage(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_prefetch_next {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_prefetch_next(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_oct_row_u32x2 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_oct_row_u32x2(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_planar_quad_row_uint4 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_planar_quad_row_uint4(
                a_planar.as_ref().expect("planar weights"),
                xq128,
                &y256,
                m,
                k,
                n,
                add,
            )
        } else if group128_tile64_planar_quad_uint4 {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_tile64_planar_quad_uint4(
                a_tile64_planar.as_ref().expect("tile64 planar weights"),
                xq128,
                &y256,
                m,
                k,
                n,
                add,
            )
        } else if group128_f16_accum {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_f16_accum(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if group128_half_meta {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_half_meta(
                a_half_meta.as_ref().expect("half-meta weights"),
                xq128,
                &y256,
                m,
                k,
                n,
                add,
            )
        } else if f32a_k32 {
            if add {
                gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_add(&a, &x, &y256, m, k, n)
            } else {
                gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_set(&a, &x, &y256, m, k, n)
            }
        } else if f32a_k32_unique_decode {
            gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_unique_decode(&a, &x, &y256, m, k, n, add)
        } else if f32a_k32_compact_decode {
            gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_compact_decode(&a, &x, &y256, m, k, n, add)
        } else if f32a_k32_compact_perm_decode {
            gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_compact_perm_decode(&a, &x, &y256, m, k, n, add)
        } else if f32a_k64 {
            gpu.gemm_hfq4g256_f32a_wmma_128x64_k64(&a, &x, &y256, m, k, n, add)
        } else if f32a_k64_compact_decode {
            gpu.gemm_hfq4g256_f32a_wmma_128x64_k64_compact_decode(&a, &x, &y256, m, k, n, add)
        } else if group128_direct {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_direct(&a, xq128, &y256, m, k, n, add)
        } else if group128_serial_row {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_serial_row(
                &a, xq128, &y256, m, k, n, add,
            )
        } else if staged {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group256_staged(&a, xq256, &y256, m, k, n, add)
        } else if serial_row {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group256_serial_row(
                &a, xq256, &y256, m, k, n, add,
            )
        } else {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group256_direct(&a, xq256, &y256, m, k, n, add)
        }
    };

    run128(&mut gpu).expect("group128 correctness");
    run_candidate(&mut gpu).expect("candidate correctness");
    gpu.hip.device_synchronize().expect("correctness sync");
    let (max_abs, mean_abs) = if skip_correctness {
        (f32::NAN, f64::NAN)
    } else {
        let ref_host = gpu.download_f32(&y128).expect("download group128");
        let candidate_host = gpu.download_f32(&y256).expect("download group256");
        let mut max_abs = 0.0f32;
        let mut abs_sum = 0.0f64;
        for (a, b) in ref_host.iter().zip(candidate_host.iter()) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            abs_sum += d as f64;
        }
        (max_abs, abs_sum / ref_host.len() as f64)
    };

    for _ in 0..3 {
        run128(&mut gpu).expect("group128 warmup");
        run_candidate(&mut gpu).expect("candidate warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut ms128 = Vec::with_capacity(pairs);
    let mut ms256 = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        for direct_first in [pair % 2 != 0, pair % 2 == 0] {
            let start = Instant::now();
            if direct_first {
                run_candidate(&mut gpu).expect("timed candidate");
            } else {
                run128(&mut gpu).expect("timed group128");
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let ms = start.elapsed().as_secs_f64() * 1_000.0;
            if direct_first {
                ms256.push(ms);
            } else {
                ms128.push(ms);
            }
        }
    }

    let med128 = median(ms128);
    let med256 = median(ms256);
    println!("m={m} k={k} n={n}");
    println!("add={add}");
    println!(
        "reference_mode={}",
        if f32a_k32_unique_decode
            || f32a_k32_compact_decode
            || f32a_k32_compact_perm_decode
            || f32a_k64
            || f32a_k64_compact_decode
        {
            if f32a_k32_compact_perm_decode {
                "f32a-k32-compact-decode"
            } else {
                "f32a-k32"
            }
        } else {
            "group128-lds"
        }
    );
    println!("group128_lds_ms={med128:.4}");
    println!(
        "group256_mode={}",
        if stream_k128_x256y128 {
            "stream-k128-x256y128"
        } else if stream_k128_phased_x256 {
            "stream-k128-phased-x256"
        } else if stream_k128 {
            "stream-k128"
        } else if skip_zero {
            "skip-zero"
        } else if group128_k32_stationary {
            "group128-k32-stationary"
        } else if group128_direct_x512 {
            "group128-direct-x512"
        } else if group128_x128y128 {
            "group128-x128y128"
        } else if group128_x192y96 {
            "group128-x192y96"
        } else if group128_n2_reuse {
            "group128-n2-reuse"
        } else if group128_n4_reuse {
            "group128-n4-reuse"
        } else if direct_packed_weight_n2 {
            "direct-packed-weight-n2"
        } else if packed_weight_y64 {
            "packed-weight-y64"
        } else if packed_weight_x128y64 {
            "packed-weight-x128y64"
        } else if group128_dual_row_u32x2 {
            "group128-dual-row-u32x2"
        } else if group128_dual_row_scalar2 {
            "group128-dual-row-scalar2"
        } else if group128_quad_row_u32x2 {
            "group128-quad-row-u32x2"
        } else if group128_interleave_row_wmma {
            "group128-interleave-row-wmma"
        } else if group128_quad_row_min1 {
            "group128-quad-row-min1"
        } else if group128_quad_row_vector_activation {
            "group128-quad-row-vector-activation"
        } else if group128_quad_row_vector_activation_batch3 {
            "group128-quad-row-vector-activation-batch3"
        } else if group128_warp_specialized_stage {
            "group128-warp-specialized-stage"
        } else if group128_prefetch_next {
            "group128-prefetch-next"
        } else if group128_oct_row_u32x2 {
            "group128-oct-row-u32x2"
        } else if group128_planar_quad_row_uint4 {
            "group128-planar-quad-row-uint4"
        } else if group128_tile64_planar_quad_uint4 {
            "group128-tile64-planar-quad-uint4"
        } else if group128_f16_accum {
            "group128-f16-accum"
        } else if group128_half_meta {
            "group128-half-meta"
        } else if f32a_k32 {
            "f32a-k32"
        } else if f32a_k32_unique_decode {
            "f32a-k32-unique-decode"
        } else if f32a_k32_compact_decode {
            "f32a-k32-compact-decode"
        } else if f32a_k32_compact_perm_decode {
            "f32a-k32-compact-perm-decode"
        } else if f32a_k64 {
            "f32a-k64"
        } else if f32a_k64_compact_decode {
            "f32a-k64-compact-decode"
        } else if group128_direct {
            "group128-direct"
        } else if group128_serial_row {
            "group128-serial-row"
        } else if staged {
            "staged"
        } else if serial_row {
            "serial-row"
        } else {
            "direct"
        }
    );
    println!("group256_ms={med256:.4}");
    println!("group256_speedup={:.4}x", med128 / med256);
    println!("max_abs={max_abs:.8e}");
    println!("mean_abs={mean_abs:.8e}");
}
