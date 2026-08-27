// SPDX-License-Identifier: MIT OR Apache-2.0
//! Qwen3.6 GDN conv token-parallel correctness and timing probe.

use rdna_compute::Gpu;
use std::time::Instant;

const K_DIM: usize = 2_048;
const V_DIM: usize = 2_048;
const N: usize = 2_048;

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn check_short(gpu: &mut Gpu, n: usize) {
    const KD: usize = 32;
    const VD: usize = 32;
    let channels = 2 * KD + VD;
    let input_host: Vec<f32> = (0..n * channels)
        .map(|i| ((i * 7) % 31) as f32 * 0.01 - 0.15)
        .collect();
    let weight_host: Vec<f32> = (0..channels * 4)
        .map(|i| ((i * 5) % 23) as f32 * 0.003 - 0.03)
        .collect();
    let state_host: Vec<f32> = (0..channels * 3)
        .map(|i| ((i * 11) % 29) as f32 * 0.003 - 0.04)
        .collect();
    let input = gpu.upload_f32(&input_host, &[n, channels]).unwrap();
    let weight = gpu.upload_f32(&weight_host, &[channels, 4]).unwrap();
    let state_seq = gpu.upload_f32(&state_host, &[channels, 3]).unwrap();
    let state_par = gpu.upload_f32(&state_host, &[channels, 3]).unwrap();
    let zero_k = vec![0.0f32; n * KD];
    let zero_v = vec![0.0f32; n * VD];
    let q_seq = gpu.upload_f32(&zero_k, &[n, KD]).unwrap();
    let k_seq = gpu.upload_f32(&zero_k, &[n, KD]).unwrap();
    let v_seq = gpu.upload_f32(&zero_v, &[n, VD]).unwrap();
    let q_par = gpu.upload_f32(&zero_k, &[n, KD]).unwrap();
    let k_par = gpu.upload_f32(&zero_k, &[n, KD]).unwrap();
    let v_par = gpu.upload_f32(&zero_v, &[n, VD]).unwrap();
    gpu.conv1d_silu_split_f32_n(
        &q_seq, &k_seq, &v_seq, &input, &weight, &state_seq, KD, VD, n,
    )
    .unwrap();
    gpu.conv1d_silu_split_f32_n_parallel(
        &q_par, &k_par, &v_par, &input, &weight, &state_par, KD, VD, n,
    )
    .unwrap();
    gpu.hip.device_synchronize().unwrap();
    for (name, a, b) in [
        ("q", &q_seq, &q_par),
        ("k", &k_seq, &k_par),
        ("v", &v_seq, &v_par),
        ("state", &state_seq, &state_par),
    ] {
        let lhs = gpu.download_f32(a).unwrap();
        let rhs = gpu.download_f32(b).unwrap();
        assert_eq!(lhs, rhs, "N={n} tensor={name}");
    }
    println!("short_correctness n={n} exact=true");
}

fn main() {
    let channels = 2 * K_DIM + V_DIM;
    let mut gpu = Gpu::init().expect("GPU init");
    for n in 1..=3 {
        check_short(&mut gpu, n);
    }
    let input_host: Vec<f32> = (0..N * channels)
        .map(|i| ((i * 17 + i / channels * 31) % 101) as f32 * 0.01 - 0.5)
        .collect();
    let weight_host: Vec<f32> = (0..channels * 4)
        .map(|i| ((i * 13) % 37) as f32 * 0.002 - 0.03)
        .collect();
    let state_host: Vec<f32> = (0..channels * 3)
        .map(|i| ((i * 11) % 29) as f32 * 0.003 - 0.04)
        .collect();
    let input = gpu.upload_f32(&input_host, &[N, channels]).expect("input");
    let weight = gpu
        .upload_f32(&weight_host, &[channels, 4])
        .expect("weight");
    let state_seq = gpu
        .upload_f32(&state_host, &[channels, 3])
        .expect("seq state");
    let state_par = gpu
        .upload_f32(&state_host, &[channels, 3])
        .expect("par state");
    let zero_k = vec![0.0f32; N * K_DIM];
    let zero_v = vec![0.0f32; N * V_DIM];
    let q_seq = gpu.upload_f32(&zero_k, &[N, K_DIM]).expect("q seq");
    let k_seq = gpu.upload_f32(&zero_k, &[N, K_DIM]).expect("k seq");
    let v_seq = gpu.upload_f32(&zero_v, &[N, V_DIM]).expect("v seq");
    let q_par = gpu.upload_f32(&zero_k, &[N, K_DIM]).expect("q par");
    let k_par = gpu.upload_f32(&zero_k, &[N, K_DIM]).expect("k par");
    let v_par = gpu.upload_f32(&zero_v, &[N, V_DIM]).expect("v par");

    gpu.conv1d_silu_split_f32_n(
        &q_seq, &k_seq, &v_seq, &input, &weight, &state_seq, K_DIM, V_DIM, N,
    )
    .expect("sequential correctness");
    gpu.conv1d_silu_split_f32_n_parallel(
        &q_par, &k_par, &v_par, &input, &weight, &state_par, K_DIM, V_DIM, N,
    )
    .expect("parallel correctness");
    gpu.hip.device_synchronize().expect("correctness sync");
    for (name, reference, candidate) in [
        ("q", &q_seq, &q_par),
        ("k", &k_seq, &k_par),
        ("v", &v_seq, &v_par),
        ("state", &state_seq, &state_par),
    ] {
        let a = gpu.download_f32(reference).expect("reference");
        let b = gpu.download_f32(candidate).expect("candidate");
        let max_abs = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        println!("correctness tensor={name} max_abs={max_abs:.9e}");
    }

    let reset_state = |gpu: &Gpu, state: &rdna_compute::GpuTensor| {
        let bytes = unsafe {
            std::slice::from_raw_parts(state_host.as_ptr() as *const u8, state_host.len() * 4)
        };
        gpu.hip.memcpy_htod(&state.buf, bytes).expect("reset state");
    };
    for _ in 0..3 {
        reset_state(&gpu, &state_seq);
        gpu.conv1d_silu_split_f32_n(
            &q_seq, &k_seq, &v_seq, &input, &weight, &state_seq, K_DIM, V_DIM, N,
        )
        .expect("sequential warmup");
        reset_state(&gpu, &state_par);
        gpu.conv1d_silu_split_f32_n_parallel(
            &q_par, &k_par, &v_par, &input, &weight, &state_par, K_DIM, V_DIM, N,
        )
        .expect("parallel warmup");
    }
    gpu.hip.device_synchronize().expect("warmup sync");
    let mut seq_ms = Vec::new();
    let mut par_ms = Vec::new();
    for pair in 0..9 {
        for parallel in if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let state = if parallel { &state_par } else { &state_seq };
            reset_state(&gpu, state);
            let start = Instant::now();
            if parallel {
                gpu.conv1d_silu_split_f32_n_parallel(
                    &q_par, &k_par, &v_par, &input, &weight, state, K_DIM, V_DIM, N,
                )
            } else {
                gpu.conv1d_silu_split_f32_n(
                    &q_seq, &k_seq, &v_seq, &input, &weight, state, K_DIM, V_DIM, N,
                )
            }
            .expect("timed run");
            gpu.hip.device_synchronize().expect("timed sync");
            if parallel {
                par_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            } else {
                seq_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
        }
    }
    let seq = median(&mut seq_ms);
    let par = median(&mut par_ms);
    println!(
        "sequential_ms={seq:.4} parallel_ms={par:.4} speedup={:.4}x",
        seq / par
    );
    println!("sequential={seq_ms:?}");
    println!("parallel={par_ms:?}");
}
