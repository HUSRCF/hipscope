// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1201: the decode-only fusion `conv1d_silu_split_qknorm` must produce the
//! same bytes as the unfused pair it replaces (`conv1d_silu_split_f32` then
//! `fused_qk_l2_norm_scale_f32`). The Qwen3.5 lowered decode uses the fusion
//! and the hand decode (`HIPFIRE_FORWARD_LOWERED=0`, DFlash's per-token
//! hidden-extracting forward) uses the pair, so any difference makes the two
//! decode paths diverge. It did: the compiler contracted the conv tap sum and
//! the first Q/K square-sum add into FMAs in a different order in each kernel.
//!
//! Q, K, V and the conv state are compared bit for bit over four chained
//! decode steps (the state ring shifts each step), direct and replayed from a
//! captured hipGraph, at the Qwen3.6/3.8-27B shape (16 key heads, 48 value
//! heads), the Qwen3.5-A3B shape (16 / 32), and head counts one either side
//! of those (a value width of an odd number of 128-wide heads leaves the last
//! 256-thread V block half full).
//!
//! gfx1100 and gfx1151 still run the original fused expressions (their code
//! objects are unchanged), so the test skips there.
//!
//! `#[ignore]`d: needs a GPU with a working HIP toolchain. Run explicitly:
//!
//!   cargo test -p rdna-compute --release --features deltanet \
//!       --test conv_qknorm_parity -- --ignored

#![cfg(feature = "deltanet")]

use rdna_compute::{Gpu, GpuTensor};

const HEAD_DIM: usize = 128;
const EPS: f32 = 1e-6;
const STEPS: usize = 4;

/// (key heads, value heads): production shapes, then ±1 head around them.
const SHAPES: [(usize, usize); 6] = [(16, 48), (16, 32), (15, 47), (17, 49), (16, 33), (1, 1)];

/// Deterministic values in [-scale, scale).
fn fill(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn bits(gpu: &Gpu, t: &GpuTensor) -> Vec<u32> {
    gpu.download_f32(t)
        .expect("download")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

fn write(gpu: &Gpu, t: &GpuTensor, data: &[f32]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.hip.memcpy_htod(&t.buf, bytes).expect("htod");
}

struct Fused<'a> {
    q: &'a GpuTensor,
    k: &'a GpuTensor,
    v: &'a GpuTensor,
    input: &'a GpuTensor,
    weight: &'a GpuTensor,
    state: &'a GpuTensor,
    n_key_heads: usize,
    v_dim: usize,
}

fn launch_fused(gpu: &mut Gpu, f: &Fused) {
    gpu.conv1d_silu_split_qknorm(
        f.q,
        f.k,
        f.v,
        f.input,
        f.weight,
        f.state,
        f.n_key_heads * HEAD_DIM,
        f.v_dim,
        f.n_key_heads,
        HEAD_DIM,
        1.0 / (HEAD_DIM as f32).sqrt(),
        EPS,
    )
    .expect("conv1d_silu_split_qknorm");
}

fn check_shape(gpu: &mut Gpu, n_key_heads: usize, n_value_heads: usize, graph: bool) {
    let k_dim = n_key_heads * HEAD_DIM;
    let v_dim = n_value_heads * HEAD_DIM;
    let channels = 2 * k_dim + v_dim;
    let mut seed = 0x5eed_0000 + (n_key_heads * 1000 + n_value_heads) as u64;
    let mut alloc = |gpu: &mut Gpu, n: usize| gpu.upload_f32(&vec![0.0; n], &[n]).unwrap();

    let weight = alloc(gpu, channels * 4);
    write(gpu, &weight, &fill(&mut seed, channels * 4, 0.5));
    let state0 = fill(&mut seed, channels * 3, 2.0);
    let (state_ref, state_fused) = (alloc(gpu, channels * 3), alloc(gpu, channels * 3));
    write(gpu, &state_ref, &state0);
    write(gpu, &state_fused, &state0);
    let input = alloc(gpu, channels);
    let (q_ref, k_ref, v_ref) = (alloc(gpu, k_dim), alloc(gpu, k_dim), alloc(gpu, v_dim));
    let (q_fused, k_fused, v_fused) = (alloc(gpu, k_dim), alloc(gpu, k_dim), alloc(gpu, v_dim));
    let fused = Fused {
        q: &q_fused,
        k: &k_fused,
        v: &v_fused,
        input: &input,
        weight: &weight,
        state: &state_fused,
        n_key_heads,
        v_dim,
    };

    if graph {
        // JIT outside the capture on throwaway state, then record one launch.
        let scratch = alloc(gpu, channels * 3);
        launch_fused(gpu, &Fused { state: &scratch, ..fused });
        gpu.hip.device_synchronize().unwrap();
        gpu.free_tensor(scratch).unwrap();
        if gpu.active_stream.is_none() {
            gpu.active_stream = Some(gpu.hip.stream_create().unwrap());
        }
        let stream = gpu.active_stream.take().unwrap();
        gpu.graphs
            .begin_graph_capture(&gpu.hip, gpu.device_id, &stream)
            .unwrap();
        gpu.active_stream = Some(stream);
        launch_fused(gpu, &fused);
        let stream = gpu.active_stream.take().unwrap();
        gpu.graphs
            .end_graph_capture(&gpu.hip, gpu.device_id, &stream)
            .unwrap();
        gpu.active_stream = Some(stream);
    }

    for step in 0..STEPS {
        gpu.hip.device_synchronize().unwrap();
        write(gpu, &input, &fill(&mut seed, channels, 2.0));
        gpu.conv1d_silu_split_f32(
            &q_ref, &k_ref, &v_ref, &input, &weight, &state_ref, k_dim, v_dim,
        )
        .unwrap();
        gpu.fused_qk_l2_norm_scale_f32(
            &q_ref,
            &k_ref,
            n_key_heads,
            HEAD_DIM,
            1.0 / (HEAD_DIM as f32).sqrt(),
            EPS,
        )
        .unwrap();
        if graph {
            let stream = gpu.active_stream.as_ref().unwrap();
            gpu.graphs
                .graph_launch(&gpu.hip, gpu.device_id, stream)
                .unwrap();
        } else {
            launch_fused(gpu, &fused);
        }
        gpu.hip.device_synchronize().unwrap();
        for (name, reference, got) in [
            ("q", &q_ref, &q_fused),
            ("k", &k_ref, &k_fused),
            ("v", &v_ref, &v_fused),
            ("conv state", &state_ref, &state_fused),
        ] {
            let (r, f) = (bits(gpu, reference), bits(gpu, got));
            let differing = r.iter().zip(&f).filter(|(a, b)| a != b).count();
            assert_eq!(
                differing,
                0,
                "{n_key_heads}/{n_value_heads} heads, {}, step {step}: {differing} of {} {name} \
                 floats differ between conv1d_silu_split_qknorm and \
                 conv1d_silu_split_f32 + fused_qk_l2_norm_scale_f32",
                if graph { "graph replay" } else { "direct" },
                r.len()
            );
        }
    }
    if graph {
        gpu.graphs.drop_captured_graph(&gpu.hip, gpu.device_id);
        if let Some(stream) = gpu.active_stream.take() {
            gpu.hip.stream_destroy(stream).unwrap();
        }
    }
    for t in [
        weight, state_ref, state_fused, input, q_ref, k_ref, v_ref, q_fused, k_fused, v_fused,
    ] {
        gpu.free_tensor(t).unwrap();
    }
}

#[test]
#[ignore = "needs a GPU"]
fn gfx1201_conv_qknorm_fusion_matches_unfused_pair_bit_for_bit() {
    let mut gpu = Gpu::init().expect("gpu init");
    if !gpu.arch_caps.is_gfx1201() {
        eprintln!("skip: {} keeps the original fused expressions (only gfx1201 is exact)", gpu.arch);
        return;
    }
    for (n_key_heads, n_value_heads) in SHAPES {
        check_shape(&mut gpu, n_key_heads, n_value_heads, false);
        check_shape(&mut gpu, n_key_heads, n_value_heads, true);
    }
}
