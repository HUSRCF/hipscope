// SPDX-License-Identifier: Apache-2.0

//! DeepSeek V4 compressor paired ring-write parity and launch benchmark.

use rdna_compute::{DType, Gpu, GpuTensor};

const WARMUP: usize = 500;
const ITERS: usize = 20_000;
const SAMPLES: usize = 5;

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let tensor = gpu
        .alloc_tensor(&[data.len() * 4], DType::Raw)
        .expect("alloc i32");
    gpu.hip.memcpy_htod(&tensor.buf, bytes).expect("upload i32");
    tensor
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    eprintln!(
        "=== DS4 paired ring write ===\narch={} warmup={WARMUP} iterations={ITERS}",
        gpu.arch
    );

    for (proj_dim, state_rows, slot, label) in [
        (1024usize, 8usize, 5i32, "ratio4 main"),
        (256usize, 8usize, 5i32, "ratio4 index"),
        (512usize, 128usize, 37i32, "ratio128 main"),
    ] {
        let src0_host: Vec<f32> = (0..proj_dim).map(|i| (i as f32 * 0.031_25).sin()).collect();
        let src1_host: Vec<f32> = (0..proj_dim)
            .map(|i| (i as f32 * 0.015_625).cos())
            .collect();
        let src0 = gpu.upload_f32(&src0_host, &[proj_dim]).unwrap();
        let src1 = gpu.upload_f32(&src1_host, &[proj_dim]).unwrap();
        let slot_buf = upload_i32(&mut gpu, &[slot]);

        let ref0 = gpu.zeros(&[state_rows, proj_dim], DType::F32).unwrap();
        let ref1 = gpu.zeros(&[state_rows, proj_dim], DType::F32).unwrap();
        let pair0 = gpu.zeros(&[state_rows, proj_dim], DType::F32).unwrap();
        let pair1 = gpu.zeros(&[state_rows, proj_dim], DType::F32).unwrap();

        gpu.state_ring_write_f32_buf(&src0, &ref0, &slot_buf, proj_dim as i32)
            .unwrap();
        gpu.state_ring_write_f32_buf(&src1, &ref1, &slot_buf, proj_dim as i32)
            .unwrap();
        gpu.state_ring_write_pair_f32_buf(&src0, &src1, &pair0, &pair1, &slot_buf, proj_dim as i32)
            .unwrap();
        gpu.hip.device_synchronize().unwrap();

        let ref0_host = gpu.download_f32(&ref0).unwrap();
        let ref1_host = gpu.download_f32(&ref1).unwrap();
        let pair0_host = gpu.download_f32(&pair0).unwrap();
        let pair1_host = gpu.download_f32(&pair1).unwrap();
        let bad0 = ref0_host
            .iter()
            .zip(&pair0_host)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let bad1 = ref1_host
            .iter()
            .zip(&pair1_host)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!((bad0, bad1), (0, 0), "{label} parity");
        eprintln!("\n--- {label}: proj={proj_dim} rows={state_rows} parity=PASS ---");

        for _ in 0..WARMUP {
            gpu.state_ring_write_f32_buf(&src0, &ref0, &slot_buf, proj_dim as i32)
                .unwrap();
            gpu.state_ring_write_f32_buf(&src1, &ref1, &slot_buf, proj_dim as i32)
                .unwrap();
            gpu.state_ring_write_pair_f32_buf(
                &src0,
                &src1,
                &pair0,
                &pair1,
                &slot_buf,
                proj_dim as i32,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let mut separate_samples = Vec::new();
        let mut pair_samples = Vec::new();
        for sample in 0..SAMPLES {
            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.state_ring_write_f32_buf(&src0, &ref0, &slot_buf, proj_dim as i32)
                    .unwrap();
                gpu.state_ring_write_f32_buf(&src1, &ref1, &slot_buf, proj_dim as i32)
                    .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let separate = started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64;

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.state_ring_write_pair_f32_buf(
                    &src0,
                    &src1,
                    &pair0,
                    &pair1,
                    &slot_buf,
                    proj_dim as i32,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let pair = started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64;
            separate_samples.push(separate);
            pair_samples.push(pair);
            eprintln!("sample={sample} separate={separate:.3}us pair={pair:.3}us");
        }
        let separate = median(separate_samples);
        let pair = median(pair_samples);
        eprintln!(
            "median separate={separate:.3}us pair={pair:.3}us speedup={:.1}%",
            (separate / pair - 1.0) * 100.0
        );
    }

    for (proj_dim, label) in [(1024usize, "ratio4 main"), (256usize, "ratio4 index")] {
        let ratio = 4usize;
        let rows = 2 * ratio;
        let head_dim = proj_dim / 2;
        let elems = rows * proj_dim;
        let state0_host: Vec<f32> = (0..elems)
            .map(|i| (i as f32 * 0.003_906_25).sin())
            .collect();
        let state1_host: Vec<f32> = (0..elems).map(|i| (i as f32 * 0.007_812_5).cos()).collect();

        let ref0 = gpu.upload_f32(&state0_host, &[rows, proj_dim]).unwrap();
        let ref1 = gpu.upload_f32(&state1_host, &[rows, proj_dim]).unwrap();
        let pair0 = gpu.upload_f32(&state0_host, &[rows, proj_dim]).unwrap();
        let pair1 = gpu.upload_f32(&state1_host, &[rows, proj_dim]).unwrap();
        let ref_out0 = gpu.zeros(&[rows, head_dim], DType::F32).unwrap();
        let ref_out1 = gpu.zeros(&[rows, head_dim], DType::F32).unwrap();
        let pair_out0 = gpu.zeros(&[rows, head_dim], DType::F32).unwrap();
        let pair_out1 = gpu.zeros(&[rows, head_dim], DType::F32).unwrap();
        let commit_buf = upload_i32(&mut gpu, &[0]);

        gpu.compressor_overlap_concat_f32(&ref0, &ref_out0, ratio as i32, head_dim as i32)
            .unwrap();
        gpu.compressor_overlap_concat_f32(&ref1, &ref_out1, ratio as i32, head_dim as i32)
            .unwrap();
        gpu.compressor_overlap_concat_pair_f32(
            &pair0,
            &pair1,
            &pair_out0,
            &pair_out1,
            ratio as i32,
            head_dim as i32,
        )
        .unwrap();
        gpu.state_overlap_shift_f32_buf(&ref0, &commit_buf, ratio as i32, proj_dim as i32)
            .unwrap();
        gpu.state_overlap_shift_f32_buf(&ref1, &commit_buf, ratio as i32, proj_dim as i32)
            .unwrap();
        gpu.state_overlap_shift_pair_f32_buf(
            &pair0,
            &pair1,
            &commit_buf,
            ratio as i32,
            proj_dim as i32,
        )
        .unwrap();
        gpu.hip.device_synchronize().unwrap();

        for (a, b) in [
            (&ref_out0, &pair_out0),
            (&ref_out1, &pair_out1),
            (&ref0, &pair0),
            (&ref1, &pair1),
        ] {
            let ah = gpu.download_f32(a).unwrap();
            let bh = gpu.download_f32(b).unwrap();
            assert!(
                ah.iter().zip(&bh).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{label} concat/shift parity"
            );
        }
        eprintln!("\n--- {label}: paired concat/shift parity=PASS ---");

        for _ in 0..WARMUP {
            gpu.compressor_overlap_concat_f32(&ref0, &ref_out0, ratio as i32, head_dim as i32)
                .unwrap();
            gpu.compressor_overlap_concat_f32(&ref1, &ref_out1, ratio as i32, head_dim as i32)
                .unwrap();
            gpu.compressor_overlap_concat_pair_f32(
                &pair0,
                &pair1,
                &pair_out0,
                &pair_out1,
                ratio as i32,
                head_dim as i32,
            )
            .unwrap();
            gpu.state_overlap_shift_f32_buf(&ref0, &commit_buf, ratio as i32, proj_dim as i32)
                .unwrap();
            gpu.state_overlap_shift_f32_buf(&ref1, &commit_buf, ratio as i32, proj_dim as i32)
                .unwrap();
            gpu.state_overlap_shift_pair_f32_buf(
                &pair0,
                &pair1,
                &commit_buf,
                ratio as i32,
                proj_dim as i32,
            )
            .unwrap();
        }
        gpu.hip.device_synchronize().unwrap();

        let mut concat_sep_samples = Vec::new();
        let mut concat_pair_samples = Vec::new();
        let mut shift_sep_samples = Vec::new();
        let mut shift_pair_samples = Vec::new();
        for sample in 0..SAMPLES {
            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.compressor_overlap_concat_f32(&ref0, &ref_out0, ratio as i32, head_dim as i32)
                    .unwrap();
                gpu.compressor_overlap_concat_f32(&ref1, &ref_out1, ratio as i32, head_dim as i32)
                    .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let concat_sep = started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64;

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.compressor_overlap_concat_pair_f32(
                    &pair0,
                    &pair1,
                    &pair_out0,
                    &pair_out1,
                    ratio as i32,
                    head_dim as i32,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let concat_pair = started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64;

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.state_overlap_shift_f32_buf(&ref0, &commit_buf, ratio as i32, proj_dim as i32)
                    .unwrap();
                gpu.state_overlap_shift_f32_buf(&ref1, &commit_buf, ratio as i32, proj_dim as i32)
                    .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let shift_sep = started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64;

            let started = std::time::Instant::now();
            for _ in 0..ITERS {
                gpu.state_overlap_shift_pair_f32_buf(
                    &pair0,
                    &pair1,
                    &commit_buf,
                    ratio as i32,
                    proj_dim as i32,
                )
                .unwrap();
            }
            gpu.hip.device_synchronize().unwrap();
            let shift_pair = started.elapsed().as_secs_f64() * 1.0e6 / ITERS as f64;

            concat_sep_samples.push(concat_sep);
            concat_pair_samples.push(concat_pair);
            shift_sep_samples.push(shift_sep);
            shift_pair_samples.push(shift_pair);
            eprintln!(
                "sample={sample} concat={concat_sep:.3}/{concat_pair:.3}us shift={shift_sep:.3}/{shift_pair:.3}us"
            );
        }

        let concat_sep = median(concat_sep_samples);
        let concat_pair = median(concat_pair_samples);
        let shift_sep = median(shift_sep_samples);
        let shift_pair = median(shift_pair_samples);
        eprintln!(
            "median concat={concat_sep:.3}/{concat_pair:.3}us ({:.1}%) shift={shift_sep:.3}/{shift_pair:.3}us ({:.1}%)",
            (concat_sep / concat_pair - 1.0) * 100.0,
            (shift_sep / shift_pair - 1.0) * 100.0,
        );
    }

    eprintln!("\nALL PASS");
}
