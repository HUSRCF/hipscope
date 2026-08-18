// SPDX-License-Identifier: MIT OR Apache-2.0
//! Full-shape gfx11 probe for the 256-column/64-row HFQ4 MMQ topology.

use rdna_compute::Gpu;
use std::time::Instant;

fn build_hfq4g256(m: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 136];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13 + seed) % 97) as f32 * 0.0001;
            let zero = ((row * 7 + group * 11 + seed * 3) % 31) as f32 * 0.001 - 0.015;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[off + 8 + byte] =
                    ((row * 29 + group * 19 + byte * 23 + seed * 37) & 0xff) as u8;
            }
        }
    }
    out
}

fn expand_hfq4g256_i8(src: &[u8], m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; m * groups * 264];
    for row in 0..m {
        for group in 0..groups {
            let src_off = (row * groups + group) * 136;
            let dst_off = (row * groups + group) * 264;
            out[dst_off..dst_off + 8].copy_from_slice(&src[src_off..src_off + 8]);
            for byte in 0..128 {
                let packed = src[src_off + 8 + byte];
                out[dst_off + 8 + 2 * byte] = packed & 0x0f;
                out[dst_off + 8 + 2 * byte + 1] = packed >> 4;
            }
        }
    }
    out
}

fn interleave_gate_up_rows(
    gate: &[u8],
    up: &[u8],
    m: usize,
    k: usize,
    plane_rows: usize,
) -> Vec<u8> {
    let row_bytes = (k / 256) * 136;
    let mut out = vec![0u8; 2 * m * row_bytes];
    for block in 0..m / plane_rows {
        for (plane, src) in [gate, up].into_iter().enumerate() {
            for row in 0..plane_rows {
                let src_row = block * plane_rows + row;
                let dst_row = block * (2 * plane_rows) + plane * plane_rows + row;
                out[dst_row * row_bytes..(dst_row + 1) * row_bytes]
                    .copy_from_slice(&src[src_row * row_bytes..(src_row + 1) * row_bytes]);
            }
        }
    }
    out
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn parse_args() -> (usize, usize, usize, usize, bool, bool, bool) {
    let mut m = 17_408;
    let mut k = 5_120;
    let mut n = 2_048;
    let mut pairs = 10;
    let mut residual = false;
    let mut perm_nibble = false;
    let mut base_x256y64 = false;
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
            "--residual" => residual = true,
            "--perm-nibble" => perm_nibble = true,
            "--base-x256y64" => base_x256y64 = true,
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }
    (m, k, n, pairs, residual, perm_nibble, base_x256y64)
}

fn main() {
    let (m, k, n, pairs, residual, perm_nibble, base_x256y64) = parse_args();
    assert!(!base_x256y64 || perm_nibble);
    assert!(m % 64 == 0 && k % 256 == 0 && n % 256 == 0);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "arch={} mode={} M={m} K={k} N={n} pairs={pairs}",
        gpu.arch,
        if residual {
            "residual"
        } else if perm_nibble {
            "set-perm-nibble"
        } else {
            "set"
        }
    );
    let a_host = build_hfq4g256(m, k, 0);
    let a_up_host = build_hfq4g256(m, k, 101);
    let a_expanded_host = expand_hfq4g256_i8(&a_host, m, k);
    let a_interleaved_host = interleave_gate_up_rows(&a_host, &a_up_host, m, k, 32);
    let a_full_plane_interleaved_host = interleave_gate_up_rows(&a_host, &a_up_host, m, k, 64);
    let a = gpu.upload_raw(&a_host, &[m, k]).expect("upload A");
    let a_up = gpu.upload_raw(&a_up_host, &[m, k]).expect("upload A up");
    let a_expanded = gpu
        .upload_raw(&a_expanded_host, &[m, k])
        .expect("upload expanded A");
    let a_interleaved = gpu
        .upload_raw(&a_interleaved_host, &[2 * m, k])
        .expect("upload interleaved gate/up A");
    let a_full_plane_interleaved = gpu
        .upload_raw(&a_full_plane_interleaved_host, &[2 * m, k])
        .expect("upload full-plane interleaved gate/up A");
    let x_host: Vec<f32> = (0..n * k)
        .map(|i| ((i * 17 + i / k * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let x = gpu.upload_f32(&x_host, &[n, k]).expect("upload X");
    let initial: Vec<f32> = if residual {
        (0..n * m).map(|i| (i % 17) as f32 * 0.001).collect()
    } else {
        vec![0.0; n * m]
    };
    let y_base = gpu.upload_f32(&initial, &[n, m]).expect("upload base");
    let y_wide = gpu.upload_f32(&initial, &[n, m]).expect("upload wide");
    let y_a16 = gpu.upload_f32(&initial, &[n, m]).expect("upload A16");
    let y_a16_wide = gpu.upload_f32(&initial, &[n, m]).expect("upload A16 wide");
    let y_a16_k32 = gpu.upload_f32(&initial, &[n, m]).expect("upload A16 K32");
    let y_f32a_k32 = gpu.upload_f32(&initial, &[n, m]).expect("upload F32A K32");
    let y_q8_k32 = gpu.upload_f32(&initial, &[n, m]).expect("upload Q8 K32");
    let y_q8_x256_k32 = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload Q8 X256 K32");
    let y_q8_k64 = gpu.upload_f32(&initial, &[n, m]).expect("upload Q8 K64");
    let y_regzero = gpu.upload_f32(&initial, &[n, m]).expect("upload regzero");
    let y_meta1 = gpu.upload_f32(&initial, &[n, m]).expect("upload meta1");
    let y_group128 = gpu.upload_f32(&initial, &[n, m]).expect("upload group128");
    let y_group128_row1 = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload group128 row2");
    let y_expanded = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload expanded-i8 output");
    let y_group128_pair_gate = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload group128 pair gate");
    let y_group128_pair_up = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload group128 pair up");
    let y_dual_gate = gpu.upload_f32(&initial, &[n, m]).expect("upload dual gate");
    let y_dual_up = gpu.upload_f32(&initial, &[n, m]).expect("upload dual up");
    let y_dual_interleaved_gate = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload dual interleaved gate");
    let y_dual_interleaved_up = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload dual interleaved up");
    let y_dual_full_plane_gate = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload dual full-plane gate");
    let y_dual_full_plane_up = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload dual full-plane up");
    let y_dual_full_plane_interleaved_gate = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload dual full-plane interleaved gate");
    let y_dual_full_plane_interleaved_up = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload dual full-plane interleaved up");
    let y_pair_ref_gate = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload pair ref gate");
    let y_pair_ref_up = gpu
        .upload_f32(&initial, &[n, m])
        .expect("upload pair ref up");
    let y_pair_gate = gpu.upload_f32(&initial, &[n, m]).expect("upload pair gate");
    let y_pair_up = gpu.upload_f32(&initial, &[n, m]).expect("upload pair up");
    let y_grid_gate = gpu.upload_f32(&initial, &[n, m]).expect("upload grid gate");
    let y_grid_up = gpu.upload_f32(&initial, &[n, m]).expect("upload grid up");
    let xq = gpu.ensure_q8_1_mmq_x(&x, n, k).expect("quantize X");
    let xq_group128_storage = gpu
        .hip
        .malloc((k / 128) * n * 144)
        .expect("allocate group128 X");
    let xq_group128 = xq_group128_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x, xq_group128, n, k)
        .expect("quantize group128 X");

    let run = |gpu: &mut Gpu, wide: bool| {
        if residual {
            if wide && perm_nibble {
                gpu.gemm_hfq4g256_residual_mmq_x256y64_perm(&a, &x, &y_wide, m, k, n)
            } else if wide {
                gpu.gemm_hfq4g256_residual_mmq_x256y64(&a, &x, &y_wide, m, k, n)
            } else if base_x256y64 && perm_nibble {
                gpu.gemm_hfq4g256_residual_mmq_x256y64_perm(&a, &x, &y_base, m, k, n)
            } else if base_x256y64 {
                gpu.gemm_hfq4g256_residual_mmq_x256y64(&a, &x, &y_base, m, k, n)
            } else {
                gpu.gemm_hfq4g256_residual_mmq(&a, &x, &y_base, m, k, n)
            }
        } else if wide && perm_nibble {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&a, xq, &y_wide, m, k, n)
        } else if wide {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64(&a, xq, &y_wide, m, k, n)
        } else if base_x256y64 && perm_nibble {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&a, xq, &y_base, m, k, n)
        } else if base_x256y64 {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64(&a, xq, &y_base, m, k, n)
        } else {
            gpu.gemm_hfq4g256_mmq_set_prequant(&a, xq, &y_base, m, k, n)
        }
    };
    let run_a16_k32 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_a16_wmma_128x64_k32_add(&a, &x, &y_a16_k32, m, k, n)
        } else {
            gpu.gemm_hfq4g256_a16_wmma_128x64_k32_set(&a, &x, &y_a16_k32, m, k, n)
        }
    };
    let run_q8_k32 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_q8_wmma_128x64_k32_add_prequant(&a, xq, &y_q8_k32, m, k, n)
        } else {
            gpu.gemm_hfq4g256_q8_wmma_128x64_k32_set_prequant(&a, xq, &y_q8_k32, m, k, n)
        }
    };
    let run_q8_x256_k32 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_q8_wmma_256x64_k32_add_prequant(
                &a, xq, &y_q8_x256_k32, m, k, n,
            )
        } else {
            gpu.gemm_hfq4g256_q8_wmma_256x64_k32_set_prequant(
                &a, xq, &y_q8_x256_k32, m, k, n,
            )
        }
    };
    let run_f32a_k32 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_add(&a, &x, &y_f32a_k32, m, k, n)
        } else {
            gpu.gemm_hfq4g256_f32a_wmma_128x64_k32_set(&a, &x, &y_f32a_k32, m, k, n)
        }
    };
    let run_q8_k64 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_q8_wmma_128x64_k64_add_prequant(&a, xq, &y_q8_k64, m, k, n)
        } else {
            gpu.gemm_hfq4g256_q8_wmma_128x64_k64_set_prequant(&a, xq, &y_q8_k64, m, k, n)
        }
    };
    let run_regzero = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_regzero(&a, xq, &y_regzero, m, k, n)
    };
    let run_meta1 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_residual_mmq_x256y64_perm_meta1(&a, &x, &y_meta1, m, k, n)
        } else {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_meta1(&a, xq, &y_meta1, m, k, n)
        }
    };
    let run_group128 = |gpu: &mut Gpu| {
        if residual {
            gpu.gemm_hfq4g256_mmq_add_prequant_x256y64_perm_group128(
                &a,
                xq_group128,
                &y_group128,
                m,
                k,
                n,
            )
        } else {
            gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
                &a,
                xq_group128,
                &y_group128,
                m,
                k,
                n,
            )
        }
    };
    let run_group128_row1 = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_perm_group128_row1(
            &a,
            xq_group128,
            &y_group128_row1,
            m,
            k,
            n,
            residual,
        )
    };
    let run_expanded = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_group128_expanded_i8(
            &a_expanded,
            xq_group128,
            &y_expanded,
            m,
            k,
            n,
        )
    };
    let run_dual_interleaved = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfq4g256_mmq_dual_group128(
            &a_interleaved,
            &a_interleaved,
            xq_group128,
            &y_dual_interleaved_gate,
            &y_dual_interleaved_up,
            m,
            k,
            n,
            true,
            false,
        )
    };
    let run_dual_full_plane = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfq4g256_mmq_dual_group128(
            &a,
            &a_up,
            xq_group128,
            &y_dual_full_plane_gate,
            &y_dual_full_plane_up,
            m,
            k,
            n,
            false,
            true,
        )
    };
    let run_dual_full_plane_interleaved = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfq4g256_mmq_dual_group128(
            &a_full_plane_interleaved,
            &a_full_plane_interleaved,
            xq_group128,
            &y_dual_full_plane_interleaved_gate,
            &y_dual_full_plane_interleaved_up,
            m,
            k,
            n,
            true,
            true,
        )
    };
    let run_group128_pair = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &a,
            xq_group128,
            &y_group128_pair_gate,
            m,
            k,
            n,
        )?;
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm_group128(
            &a_up,
            xq_group128,
            &y_group128_pair_up,
            m,
            k,
            n,
        )
    };
    let run_dual = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfq4g256_mmq_dual_group128(
            &a,
            &a_up,
            xq_group128,
            &y_dual_gate,
            &y_dual_up,
            m,
            k,
            n,
            false,
            false,
        )
    };
    let run_pair_ref = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&a, xq, &y_pair_ref_gate, m, k, n)?;
        gpu.gemm_hfq4g256_mmq_set_prequant_x256y64_perm(&a_up, xq, &y_pair_ref_up, m, k, n)
    };
    let run_pair = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfq4g256_q8_wmma_128x64_k32_set_prequant(
            &a,
            &a_up,
            xq,
            &y_pair_gate,
            &y_pair_up,
            m,
            k,
            n,
        )
    };
    let run_grid = |gpu: &mut Gpu| {
        gpu.gemm_gate_up_hfq4g256_mmq_set_prequant_x256y64_perm_grid(
            &a,
            &a_up,
            xq,
            &y_grid_gate,
            &y_grid_up,
            m,
            k,
            n,
        )
    };

    run(&mut gpu, false).expect("baseline correctness");
    run(&mut gpu, true).expect("x256y64 correctness");
    gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_a16, m, k, n)
        .expect("A16 correctness");
    gpu.gemm_hfq4g256_a16_wmma_128x64_set(&a, &x, &y_a16_wide, m, k, n)
        .expect("A16 128x64 correctness");
    run_a16_k32(&mut gpu).expect("A16 128x64 K32 correctness");
    run_f32a_k32(&mut gpu).expect("F32A 128x64 K32 correctness");
    run_q8_k32(&mut gpu).expect("Q8 128x64 K32 correctness");
    run_q8_x256_k32(&mut gpu).expect("Q8 256x64 K32 correctness");
    run_q8_k64(&mut gpu).expect("Q8 128x64 K64 correctness");
    run_regzero(&mut gpu).expect("Q8 regzero correctness");
    run_meta1(&mut gpu).expect("Q8 meta1 correctness");
    run_group128(&mut gpu).expect("Q8 group128 correctness");
    run_group128_row1(&mut gpu).expect("Q8 group128 row1 correctness");
    run_expanded(&mut gpu).expect("expanded-i8 correctness");
    run_group128_pair(&mut gpu).expect("Q8 group128 pair correctness");
    run_dual(&mut gpu).expect("Q8 dual group128 correctness");
    run_dual_interleaved(&mut gpu).expect("Q8 interleaved dual group128 correctness");
    run_dual_full_plane(&mut gpu).expect("Q8 full-plane dual group128 correctness");
    run_dual_full_plane_interleaved(&mut gpu)
        .expect("Q8 full-plane interleaved dual group128 correctness");
    run_pair_ref(&mut gpu).expect("Q8 pair reference correctness");
    run_pair(&mut gpu).expect("Q8 pair correctness");
    run_grid(&mut gpu).expect("Q8 combined-grid correctness");
    gpu.hip.device_synchronize().expect("sync correctness");
    let base = gpu.download_f32(&y_base).expect("download base");
    let wide = gpu.download_f32(&y_wide).expect("download wide");
    let a16 = gpu.download_f32(&y_a16).expect("download A16");
    let a16_wide = gpu.download_f32(&y_a16_wide).expect("download A16 wide");
    let a16_k32 = gpu.download_f32(&y_a16_k32).expect("download A16 K32");
    let f32a_k32 = gpu.download_f32(&y_f32a_k32).expect("download F32A K32");
    let q8_k32 = gpu.download_f32(&y_q8_k32).expect("download Q8 K32");
    let q8_x256_k32 = gpu
        .download_f32(&y_q8_x256_k32)
        .expect("download Q8 X256 K32");
    let q8_k64 = gpu.download_f32(&y_q8_k64).expect("download Q8 K64");
    let regzero = gpu.download_f32(&y_regzero).expect("download regzero");
    let meta1 = gpu.download_f32(&y_meta1).expect("download meta1");
    let group128 = gpu.download_f32(&y_group128).expect("download group128");
    let group128_row1 = gpu
        .download_f32(&y_group128_row1)
        .expect("download group128 row2");
    let expanded = gpu
        .download_f32(&y_expanded)
        .expect("download expanded-i8 output");
    let group128_pair_gate = gpu
        .download_f32(&y_group128_pair_gate)
        .expect("download group128 pair gate");
    let group128_pair_up = gpu
        .download_f32(&y_group128_pair_up)
        .expect("download group128 pair up");
    let dual_gate = gpu.download_f32(&y_dual_gate).expect("download dual gate");
    let dual_up = gpu.download_f32(&y_dual_up).expect("download dual up");
    let dual_interleaved_gate = gpu
        .download_f32(&y_dual_interleaved_gate)
        .expect("download dual interleaved gate");
    let dual_interleaved_up = gpu
        .download_f32(&y_dual_interleaved_up)
        .expect("download dual interleaved up");
    let dual_full_plane_gate = gpu
        .download_f32(&y_dual_full_plane_gate)
        .expect("download dual full-plane gate");
    let dual_full_plane_up = gpu
        .download_f32(&y_dual_full_plane_up)
        .expect("download dual full-plane up");
    let dual_full_plane_interleaved_gate = gpu
        .download_f32(&y_dual_full_plane_interleaved_gate)
        .expect("download dual full-plane interleaved gate");
    let dual_full_plane_interleaved_up = gpu
        .download_f32(&y_dual_full_plane_interleaved_up)
        .expect("download dual full-plane interleaved up");
    let pair_ref_gate = gpu
        .download_f32(&y_pair_ref_gate)
        .expect("download pair ref gate");
    let pair_ref_up = gpu
        .download_f32(&y_pair_ref_up)
        .expect("download pair ref up");
    let pair_gate = gpu.download_f32(&y_pair_gate).expect("download pair gate");
    let pair_up = gpu.download_f32(&y_pair_up).expect("download pair up");
    let grid_gate = gpu.download_f32(&y_grid_gate).expect("download grid gate");
    let grid_up = gpu.download_f32(&y_grid_up).expect("download grid up");
    let max_abs = base
        .iter()
        .zip(&wide)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("correctness max_abs={max_abs:.8e}");
    let a16_max_abs = base
        .iter()
        .zip(&a16)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let a16_mean_abs = base
        .iter()
        .zip(&a16)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    eprintln!("A16 vs Q8 max_abs={a16_max_abs:.8e} mean_abs={a16_mean_abs:.8e}");
    let a16_wide_max_abs = a16
        .iter()
        .zip(&a16_wide)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("A16 128x64 vs A16 16x16 max_abs={a16_wide_max_abs:.8e}");
    let a16_k32_max_abs = a16
        .iter()
        .zip(&a16_k32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let f32a_k32_max_abs = a16_k32
        .iter()
        .zip(&f32a_k32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("A16 128x64 K32 vs A16 16x16 max_abs={a16_k32_max_abs:.8e}");
    let q8_k32_max_abs = base
        .iter()
        .zip(&q8_k32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let q8_k32_mean_abs = base
        .iter()
        .zip(&q8_k32)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    let q8_x256_k32_max_abs = base
        .iter()
        .zip(&q8_x256_k32)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let q8_x256_k32_mean_abs = base
        .iter()
        .zip(&q8_x256_k32)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    eprintln!("Q8 128x64 K32 vs Q8 max_abs={q8_k32_max_abs:.8e} mean_abs={q8_k32_mean_abs:.8e}");
    let q8_k64_max_abs = base
        .iter()
        .zip(&q8_k64)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let q8_k64_mean_abs = base
        .iter()
        .zip(&q8_k64)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    eprintln!("Q8 128x64 K64 vs Q8 max_abs={q8_k64_max_abs:.8e} mean_abs={q8_k64_mean_abs:.8e}");
    let regzero_max_abs = base
        .iter()
        .zip(&regzero)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let regzero_mean_abs = base
        .iter()
        .zip(&regzero)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    eprintln!("Q8 regzero vs Q8 max_abs={regzero_max_abs:.8e} mean_abs={regzero_mean_abs:.8e}");
    let meta1_max_abs = base
        .iter()
        .zip(&meta1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let meta1_mean_abs = base
        .iter()
        .zip(&meta1)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    let group128_max_abs = base
        .iter()
        .zip(&group128)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let group128_mean_abs = base
        .iter()
        .zip(&group128)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / base.len() as f64;
    let group128_row1_max_abs = group128
        .iter()
        .zip(&group128_row1)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let expanded_max_abs = group128
        .iter()
        .zip(&expanded)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_gate_max_abs = group128_pair_gate
        .iter()
        .zip(&dual_gate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_up_max_abs = group128_pair_up
        .iter()
        .zip(&dual_up)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_interleaved_gate_max_abs = group128_pair_gate
        .iter()
        .zip(&dual_interleaved_gate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_interleaved_up_max_abs = group128_pair_up
        .iter()
        .zip(&dual_interleaved_up)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_full_plane_gate_max_abs = group128_pair_gate
        .iter()
        .zip(&dual_full_plane_gate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_full_plane_up_max_abs = group128_pair_up
        .iter()
        .zip(&dual_full_plane_up)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_full_plane_interleaved_gate_max_abs = group128_pair_gate
        .iter()
        .zip(&dual_full_plane_interleaved_gate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let dual_full_plane_interleaved_up_max_abs = group128_pair_up
        .iter()
        .zip(&dual_full_plane_interleaved_up)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Q8 meta1 vs Q8 max_abs={meta1_max_abs:.8e} mean_abs={meta1_mean_abs:.8e}");
    let pair_gate_max_abs = pair_ref_gate
        .iter()
        .zip(&pair_gate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let pair_up_max_abs = pair_ref_up
        .iter()
        .zip(&pair_up)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let grid_gate_max_abs = pair_ref_gate
        .iter()
        .zip(&grid_gate)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let grid_up_max_abs = pair_ref_up
        .iter()
        .zip(&grid_up)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("Q8 gate/up pair max_abs gate={pair_gate_max_abs:.8e} up={pair_up_max_abs:.8e}");

    for _ in 0..3 {
        run(&mut gpu, false).expect("baseline warmup");
        run(&mut gpu, true).expect("x256y64 warmup");
        gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_a16, m, k, n)
            .expect("A16 warmup");
        gpu.gemm_hfq4g256_a16_wmma_128x64_set(&a, &x, &y_a16_wide, m, k, n)
            .expect("A16 128x64 warmup");
        run_a16_k32(&mut gpu).expect("A16 128x64 K32 warmup");
        run_f32a_k32(&mut gpu).expect("F32A 128x64 K32 warmup");
        run_q8_k32(&mut gpu).expect("Q8 128x64 K32 warmup");
        run_q8_x256_k32(&mut gpu).expect("Q8 256x64 K32 warmup");
        run_q8_k64(&mut gpu).expect("Q8 128x64 K64 warmup");
        run_regzero(&mut gpu).expect("Q8 regzero warmup");
        run_meta1(&mut gpu).expect("Q8 meta1 warmup");
        run_group128(&mut gpu).expect("Q8 group128 warmup");
        run_group128_row1(&mut gpu).expect("Q8 group128 row1 warmup");
        run_expanded(&mut gpu).expect("expanded-i8 warmup");
        run_group128_pair(&mut gpu).expect("Q8 group128 pair warmup");
        run_dual(&mut gpu).expect("Q8 dual group128 warmup");
        run_dual_interleaved(&mut gpu).expect("Q8 interleaved dual group128 warmup");
        run_dual_full_plane(&mut gpu).expect("Q8 full-plane dual group128 warmup");
        run_dual_full_plane_interleaved(&mut gpu)
            .expect("Q8 full-plane interleaved dual group128 warmup");
        run_pair_ref(&mut gpu).expect("Q8 pair reference warmup");
        run_pair(&mut gpu).expect("Q8 pair warmup");
        run_grid(&mut gpu).expect("Q8 combined-grid warmup");
    }
    gpu.hip.device_synchronize().expect("sync warmup");

    let mut base_ms = Vec::with_capacity(pairs);
    let mut wide_ms = Vec::with_capacity(pairs);
    let mut a16_ms = Vec::with_capacity(pairs);
    let mut a16_wide_ms = Vec::with_capacity(pairs);
    let mut a16_k32_ms = Vec::with_capacity(pairs);
    let mut f32a_k32_ms = Vec::with_capacity(pairs);
    let mut q8_k32_ms = Vec::with_capacity(pairs);
    let mut q8_x256_k32_ms = Vec::with_capacity(pairs);
    let mut q8_k64_ms = Vec::with_capacity(pairs);
    let mut regzero_ms = Vec::with_capacity(pairs);
    let mut meta1_ms = Vec::with_capacity(pairs);
    let mut group128_ms = Vec::with_capacity(pairs);
    let mut group128_row1_ms = Vec::with_capacity(pairs);
    let mut expanded_ms = Vec::with_capacity(pairs);
    let mut group128_pair_ms = Vec::with_capacity(pairs);
    let mut dual_ms = Vec::with_capacity(pairs);
    let mut dual_interleaved_ms = Vec::with_capacity(pairs);
    let mut dual_full_plane_ms = Vec::with_capacity(pairs);
    let mut dual_full_plane_interleaved_ms = Vec::with_capacity(pairs);
    let mut pair_ref_ms = Vec::with_capacity(pairs);
    let mut pair_ms = Vec::with_capacity(pairs);
    let mut grid_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let modes = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for wide in modes {
            let start = Instant::now();
            run(&mut gpu, wide).expect("timed kernel");
            gpu.hip.device_synchronize().expect("sync timed");
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            if wide {
                wide_ms.push(elapsed);
            } else {
                base_ms.push(elapsed);
            }
        }
        let start = Instant::now();
        gpu.gemm_hfq4g256_residual_wmma(&a, &x, &y_a16, m, k, n)
            .expect("timed A16 kernel");
        gpu.hip.device_synchronize().expect("sync timed A16");
        a16_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        gpu.gemm_hfq4g256_a16_wmma_128x64_set(&a, &x, &y_a16_wide, m, k, n)
            .expect("timed A16 128x64 kernel");
        gpu.hip.device_synchronize().expect("sync timed A16 128x64");
        a16_wide_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_a16_k32(&mut gpu).expect("timed A16 128x64 K32 kernel");
        gpu.hip
            .device_synchronize()
            .expect("sync timed A16 128x64 K32");
        a16_k32_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_f32a_k32(&mut gpu).expect("timed F32A 128x64 K32 kernel");
        gpu.hip
            .device_synchronize()
            .expect("sync timed F32A 128x64 K32");
        f32a_k32_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_q8_k32(&mut gpu).expect("timed Q8 128x64 K32 kernel");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 128x64 K32");
        q8_k32_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_q8_x256_k32(&mut gpu).expect("timed Q8 256x64 K32 kernel");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 256x64 K32");
        q8_x256_k32_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_q8_k64(&mut gpu).expect("timed Q8 128x64 K64 kernel");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 128x64 K64");
        q8_k64_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_regzero(&mut gpu).expect("timed Q8 regzero kernel");
        gpu.hip.device_synchronize().expect("sync timed Q8 regzero");
        regzero_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_meta1(&mut gpu).expect("timed Q8 meta1 kernel");
        gpu.hip.device_synchronize().expect("sync timed Q8 meta1");
        meta1_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let expanded_first = pair % 2 != 0;
        if expanded_first {
            let start = Instant::now();
            run_expanded(&mut gpu).expect("timed expanded-i8 kernel");
            gpu.hip
                .device_synchronize()
                .expect("sync timed expanded-i8");
            expanded_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let row2_first = pair % 2 != 0;
        if row2_first {
            let start = Instant::now();
            run_group128_row1(&mut gpu).expect("timed Q8 group128 row1 kernel");
            gpu.hip
                .device_synchronize()
                .expect("sync timed Q8 group128 row2");
            group128_row1_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let start = Instant::now();
        run_group128(&mut gpu).expect("timed Q8 group128 kernel");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 group128");
        group128_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        if !row2_first {
            let start = Instant::now();
            run_group128_row1(&mut gpu).expect("timed Q8 group128 row1 kernel");
            gpu.hip
                .device_synchronize()
                .expect("sync timed Q8 group128 row2");
            group128_row1_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        if !expanded_first {
            let start = Instant::now();
            run_expanded(&mut gpu).expect("timed expanded-i8 kernel");
            gpu.hip
                .device_synchronize()
                .expect("sync timed expanded-i8");
            expanded_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let dual_first = pair % 2 != 0;
        if dual_first {
            let start = Instant::now();
            run_dual(&mut gpu).expect("timed Q8 dual group128");
            gpu.hip
                .device_synchronize()
                .expect("sync timed Q8 dual group128");
            dual_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let start = Instant::now();
        run_group128_pair(&mut gpu).expect("timed Q8 group128 pair");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 group128 pair");
        group128_pair_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        if !dual_first {
            let start = Instant::now();
            run_dual(&mut gpu).expect("timed Q8 dual group128");
            gpu.hip
                .device_synchronize()
                .expect("sync timed Q8 dual group128");
            dual_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let start = Instant::now();
        run_dual_interleaved(&mut gpu).expect("timed Q8 interleaved dual group128");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 interleaved dual group128");
        dual_interleaved_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_dual_full_plane(&mut gpu).expect("timed Q8 full-plane dual group128");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 full-plane dual group128");
        dual_full_plane_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_dual_full_plane_interleaved(&mut gpu)
            .expect("timed Q8 full-plane interleaved dual group128");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 full-plane interleaved dual group128");
        dual_full_plane_interleaved_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_pair_ref(&mut gpu).expect("timed Q8 gate/up reference");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 gate/up reference");
        pair_ref_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_pair(&mut gpu).expect("timed Q8 gate/up pair");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 gate/up pair");
        pair_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        let start = Instant::now();
        run_grid(&mut gpu).expect("timed Q8 combined-grid");
        gpu.hip
            .device_synchronize()
            .expect("sync timed Q8 combined-grid");
        grid_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let base_median = median(&mut base_ms);
    let wide_median = median(&mut wide_ms);
    let a16_median = median(&mut a16_ms);
    let a16_wide_median = median(&mut a16_wide_ms);
    let a16_k32_median = median(&mut a16_k32_ms);
    let f32a_k32_median = median(&mut f32a_k32_ms);
    let q8_k32_median = median(&mut q8_k32_ms);
    let q8_x256_k32_median = median(&mut q8_x256_k32_ms);
    let q8_k64_median = median(&mut q8_k64_ms);
    let regzero_median = median(&mut regzero_ms);
    let meta1_median = median(&mut meta1_ms);
    let group128_median = median(&mut group128_ms);
    let group128_row1_median = median(&mut group128_row1_ms);
    let expanded_median = median(&mut expanded_ms);
    let group128_pair_median = median(&mut group128_pair_ms);
    let dual_median = median(&mut dual_ms);
    let dual_interleaved_median = median(&mut dual_interleaved_ms);
    let dual_full_plane_median = median(&mut dual_full_plane_ms);
    let dual_full_plane_interleaved_median = median(&mut dual_full_plane_interleaved_ms);
    let pair_ref_median = median(&mut pair_ref_ms);
    let pair_median = median(&mut pair_ms);
    let grid_median = median(&mut grid_ms);
    println!(
        "mode={}",
        if residual {
            "residual"
        } else if perm_nibble {
            "set-perm-nibble"
        } else {
            "set"
        }
    );
    println!("m={m} k={k} n={n}");
    println!("baseline_ms={base_median:.4}");
    println!("x256y64_ms={wide_median:.4}");
    println!("speedup={:.4}x", base_median / wide_median);
    println!("a16_wmma_ms={a16_median:.4}");
    println!("q8_over_a16={:.4}x", a16_median / wide_median);
    println!("a16_128x64_ms={a16_wide_median:.4}");
    println!("a16_topology_speedup={:.4}x", a16_median / a16_wide_median);
    println!("q8_over_a16_128x64={:.4}x", a16_wide_median / wide_median);
    println!("a16_128x64_k32_ms={a16_k32_median:.4}");
    println!("a16_k32_speedup={:.4}x", a16_wide_median / a16_k32_median);
    println!("q8_over_a16_k32={:.4}x", a16_k32_median / wide_median);
    println!("f32a_128x64_k32_ms={f32a_k32_median:.4}");
    println!(
        "f32a_vs_materialized_a16={:.4}x",
        a16_k32_median / f32a_k32_median
    );
    println!("q8_128x64_k32_ms={q8_k32_median:.4}");
    println!("q8_k32_speedup={:.4}x", wide_median / q8_k32_median);
    println!("q8_256x64_k32_ms={q8_x256_k32_median:.4}");
    println!(
        "q8_256x64_k32_speedup={:.4}x",
        wide_median / q8_x256_k32_median
    );
    println!("q8_128x64_k64_ms={q8_k64_median:.4}");
    println!("q8_k64_speedup={:.4}x", wide_median / q8_k64_median);
    println!("q8_regzero_ms={regzero_median:.4}");
    println!("q8_regzero_speedup={:.4}x", wide_median / regzero_median);
    println!("q8_meta1_ms={meta1_median:.4}");
    println!("q8_meta1_speedup={:.4}x", wide_median / meta1_median);
    println!("q8_group128_ms={group128_median:.4}");
    println!("q8_group128_speedup={:.4}x", wide_median / group128_median);
    println!("q8_group128_row1_ms={group128_row1_median:.4}");
    println!(
        "q8_group128_speedup_vs_row1={:.4}x",
        group128_row1_median / group128_median
    );
    println!("q8_group128_expanded_i8_ms={expanded_median:.4}");
    println!(
        "q8_group128_expanded_i8_speedup={:.4}x",
        group128_median / expanded_median
    );
    println!("q8_group128_pair_ms={group128_pair_median:.4}");
    println!("q8_dual_group128_ms={dual_median:.4}");
    println!(
        "q8_dual_group128_speedup={:.4}x",
        group128_pair_median / dual_median
    );
    println!("q8_dual_group128_interleaved_ms={dual_interleaved_median:.4}");
    println!(
        "q8_dual_group128_interleaved_speedup={:.4}x",
        group128_pair_median / dual_interleaved_median
    );
    println!("q8_dual_group128_full_plane_ms={dual_full_plane_median:.4}");
    println!(
        "q8_dual_group128_full_plane_speedup={:.4}x",
        group128_pair_median / dual_full_plane_median
    );
    println!("q8_dual_group128_full_plane_interleaved_ms={dual_full_plane_interleaved_median:.4}");
    println!(
        "q8_dual_group128_full_plane_interleaved_speedup={:.4}x",
        group128_pair_median / dual_full_plane_interleaved_median
    );
    println!("q8_gate_up_ref_ms={pair_ref_median:.4}");
    println!("q8_gate_up_pair_ms={pair_median:.4}");
    println!(
        "q8_gate_up_pair_speedup={:.4}x",
        pair_ref_median / pair_median
    );
    println!("q8_gate_up_grid_ms={grid_median:.4}");
    println!(
        "q8_gate_up_grid_speedup={:.4}x",
        pair_ref_median / grid_median
    );
    println!("max_abs={max_abs:.8e}");
    println!("a16_max_abs={a16_max_abs:.8e}");
    println!("a16_mean_abs={a16_mean_abs:.8e}");
    println!("a16_128x64_max_abs={a16_wide_max_abs:.8e}");
    println!("a16_128x64_k32_max_abs={a16_k32_max_abs:.8e}");
    println!("f32a_128x64_k32_max_abs={f32a_k32_max_abs:.8e}");
    println!("q8_128x64_k32_max_abs={q8_k32_max_abs:.8e}");
    println!("q8_128x64_k32_mean_abs={q8_k32_mean_abs:.8e}");
    println!("q8_256x64_k32_max_abs={q8_x256_k32_max_abs:.8e}");
    println!("q8_256x64_k32_mean_abs={q8_x256_k32_mean_abs:.8e}");
    println!("q8_128x64_k64_max_abs={q8_k64_max_abs:.8e}");
    println!("q8_128x64_k64_mean_abs={q8_k64_mean_abs:.8e}");
    println!("q8_regzero_max_abs={regzero_max_abs:.8e}");
    println!("q8_regzero_mean_abs={regzero_mean_abs:.8e}");
    println!("q8_meta1_max_abs={meta1_max_abs:.8e}");
    println!("q8_meta1_mean_abs={meta1_mean_abs:.8e}");
    println!("q8_group128_max_abs={group128_max_abs:.8e}");
    println!("q8_group128_mean_abs={group128_mean_abs:.8e}");
    println!("q8_group128_row1_max_abs={group128_row1_max_abs:.8e}");
    println!("q8_group128_expanded_i8_max_abs={expanded_max_abs:.8e}");
    println!("q8_dual_group128_gate_max_abs={dual_gate_max_abs:.8e}");
    println!("q8_dual_group128_up_max_abs={dual_up_max_abs:.8e}");
    println!("q8_dual_group128_interleaved_gate_max_abs={dual_interleaved_gate_max_abs:.8e}");
    println!("q8_dual_group128_interleaved_up_max_abs={dual_interleaved_up_max_abs:.8e}");
    println!("q8_dual_group128_full_plane_gate_max_abs={dual_full_plane_gate_max_abs:.8e}");
    println!("q8_dual_group128_full_plane_up_max_abs={dual_full_plane_up_max_abs:.8e}");
    println!(
        "q8_dual_group128_full_plane_interleaved_gate_max_abs={dual_full_plane_interleaved_gate_max_abs:.8e}"
    );
    println!(
        "q8_dual_group128_full_plane_interleaved_up_max_abs={dual_full_plane_interleaved_up_max_abs:.8e}"
    );
    println!("q8_gate_up_pair_gate_max_abs={pair_gate_max_abs:.8e}");
    println!("q8_gate_up_pair_up_max_abs={pair_up_max_abs:.8e}");
    println!("q8_gate_up_grid_gate_max_abs={grid_gate_max_abs:.8e}");
    println!("q8_gate_up_grid_up_max_abs={grid_up_max_abs:.8e}");
}
