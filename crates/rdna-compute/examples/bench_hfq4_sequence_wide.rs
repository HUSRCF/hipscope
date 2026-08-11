// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare chunked and sequence-wide launches of the production gfx1100 MQ4 primitive.

use rdna_compute::Gpu;
use std::time::Instant;

fn parse_arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.windows(2)
        .find_map(|pair| (pair[0] == name).then(|| pair[1].parse().expect("valid integer")))
        .unwrap_or(default)
}

fn has_flag(name: &str) -> bool {
    std::env::args().any(|arg| arg == name)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn checked_product(values: &[usize], label: &str) -> usize {
    values
        .iter()
        .try_fold(1usize, |product, &value| product.checked_mul(value))
        .unwrap_or_else(|| panic!("{label} exceeds usize"))
}

fn build_hfq4g256(m: usize, k: usize) -> Vec<u8> {
    let groups = k / 256;
    let mut out = vec![0u8; checked_product(&[m, groups, 136], "weight allocation")];
    for row in 0..m {
        for group in 0..groups {
            let off = (row * groups + group) * 136;
            let scale = 0.01 + ((row * 17 + group * 13) % 97) as f32 * 0.0001;
            let zero = ((row * 7 + group * 11) % 31) as f32 * 0.001 - 0.015;
            out[off..off + 4].copy_from_slice(&scale.to_le_bytes());
            out[off + 4..off + 8].copy_from_slice(&zero.to_le_bytes());
            for byte in 0..128 {
                out[off + 8 + byte] = ((row * 29 + group * 19 + byte * 23) & 0xff) as u8;
            }
        }
    }
    out
}

fn main() {
    let m = parse_arg("--m", 17_408);
    let k = parse_arg("--k", 5_120);
    let small_n = parse_arg("--small-n", 2_048);
    let chunks = parse_arg("--chunks", 8);
    let pairs = parse_arg("--pairs", 9);
    let residual = has_flag("--residual");
    let wide_n = small_n.checked_mul(chunks).expect("wide N exceeds usize");
    assert!(m % 64 == 0 && k % 256 == 0);
    assert!(small_n % 256 == 0 && wide_n % 256 == 0);
    assert!(chunks > 1 && pairs >= 3);
    assert!(m <= i32::MAX as usize && k <= i32::MAX as usize);
    assert!(small_n <= i32::MAX as usize && wide_n <= i32::MAX as usize);

    let mut gpu = Gpu::init().expect("GPU init");
    eprintln!(
        "arch={} op={} M={m} K={k} small_N={small_n} chunks={chunks} wide_N={wide_n} pairs={pairs}",
        gpu.arch,
        if residual { "add" } else { "set" }
    );

    let weights_host = build_hfq4g256(m, k);
    let weights = gpu
        .upload_raw(&weights_host, &[m, k])
        .expect("upload weights");
    drop(weights_host);

    let small_x_elements = checked_product(&[small_n, k], "small X allocation");
    let wide_x_elements = checked_product(&[wide_n, k], "wide X allocation");
    let x_small_host: Vec<f32> = (0..small_x_elements)
        .map(|i| ((i * 17 + (i / k) * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let mut x_wide_host = Vec::with_capacity(wide_x_elements);
    for _ in 0..chunks {
        x_wide_host.extend_from_slice(&x_small_host);
    }
    let mut x_small = Vec::with_capacity(chunks);
    let mut q8_small_storage = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        let x = gpu
            .upload_f32(&x_small_host, &[small_n, k])
            .expect("upload small X");
        let q8 = gpu
            .hip
            .malloc(checked_product(
                &[k / 128, small_n, 144],
                "small Q8 allocation",
            ))
            .expect("allocate small Q8");
        gpu.quantize_q8_1_mmq_group128_into(&x, q8.as_ptr(), small_n, k)
            .expect("quantize small X");
        x_small.push(x);
        q8_small_storage.push(q8);
    }
    let x_wide = gpu
        .upload_f32(&x_wide_host, &[wide_n, k])
        .expect("upload wide X");
    drop(x_wide_host);

    let q8_wide_storage = gpu
        .hip
        .malloc(checked_product(
            &[k / 128, wide_n, 144],
            "wide Q8 allocation",
        ))
        .expect("allocate wide Q8");
    let q8_wide = q8_wide_storage.as_ptr();
    gpu.quantize_q8_1_mmq_group128_into(&x_wide, q8_wide, wide_n, k)
        .expect("quantize wide X");

    let initial_small: Vec<f32> = if residual {
        (0..checked_product(&[small_n, m], "small Y allocation"))
            .map(|i| (i % 17) as f32 * 0.001)
            .collect()
    } else {
        vec![0.0; checked_product(&[small_n, m], "small Y allocation")]
    };
    let mut initial_wide = Vec::with_capacity(checked_product(&[wide_n, m], "wide Y allocation"));
    for _ in 0..chunks {
        initial_wide.extend_from_slice(&initial_small);
    }
    let mut y_small = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        y_small.push(
            gpu.upload_f32(&initial_small, &[small_n, m])
                .expect("upload small Y"),
        );
    }
    let y_wide = gpu
        .upload_f32(&initial_wide, &[wide_n, m])
        .expect("upload wide Y");
    drop(initial_wide);

    let run_small = |gpu: &mut Gpu| {
        for (q8, y) in q8_small_storage.iter().zip(&y_small) {
            gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
                &weights,
                q8.as_ptr(),
                y,
                m,
                k,
                small_n,
                residual,
            )?;
        }
        Ok::<(), hip_bridge::HipError>(())
    };
    let run_wide = |gpu: &mut Gpu| {
        gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
            &weights, q8_wide, &y_wide, m, k, wide_n, residual,
        )
    };

    for _ in 0..3 {
        run_small(&mut gpu).expect("small warmup");
        run_wide(&mut gpu).expect("wide warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");

    let mut chunked_ms = Vec::with_capacity(pairs);
    let mut wide_ms = Vec::with_capacity(pairs);
    for pair in 0..pairs {
        let wide_first = pair % 2 == 1;
        for wide in [wide_first, !wide_first] {
            let start = Instant::now();
            if wide {
                run_wide(&mut gpu).expect("wide timed");
            } else {
                run_small(&mut gpu).expect("chunked timed");
            }
            gpu.hip.device_synchronize().expect("timed sync");
            let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
            if wide {
                wide_ms.push(elapsed);
            } else {
                chunked_ms.push(elapsed);
            }
        }
    }

    // Use fresh outputs so residual accumulation from timing cannot affect the check.
    let mut check_wide_initial =
        Vec::with_capacity(checked_product(&[wide_n, m], "check Y allocation"));
    for _ in 0..chunks {
        check_wide_initial.extend_from_slice(&initial_small);
    }
    let y_small_check = gpu
        .upload_f32(&initial_small, &[small_n, m])
        .expect("upload small check Y");
    let y_wide_check = gpu
        .upload_f32(&check_wide_initial, &[wide_n, m])
        .expect("upload wide check Y");
    gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
        &weights,
        q8_small_storage[0].as_ptr(),
        &y_small_check,
        m,
        k,
        small_n,
        residual,
    )
    .expect("small correctness launch");
    gpu.gemm_hfq4g256_mmq_prequant_x256y64_group128_quad_row_u32x2(
        &weights,
        q8_wide,
        &y_wide_check,
        m,
        k,
        wide_n,
        residual,
    )
    .expect("wide correctness launch");
    gpu.hip.device_synchronize().expect("correctness sync");
    let small_out = gpu
        .download_f32(&y_small_check)
        .expect("download small output");
    let wide_out = gpu
        .download_f32(&y_wide_check)
        .expect("download wide output");
    let mut max_abs = 0.0f32;
    for chunk in 0..chunks {
        let base = chunk * small_out.len();
        for (index, expected) in small_out.iter().enumerate() {
            max_abs = max_abs.max((wide_out[base + index] - expected).abs());
        }
    }

    let chunked_median = median(&mut chunked_ms);
    let wide_median = median(&mut wide_ms);
    println!("chunked_ms={chunked_median:.6}");
    println!("wide_ms={wide_median:.6}");
    println!("speedup={:.6}", chunked_median / wide_median);
    println!("max_abs={max_abs:.8e}");
}
