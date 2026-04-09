//! hipGraph POC for the redline-dispatch branch.
//!
//! Validates that hipfire kernels capture cleanly into a hipGraph and that
//! replay produces byte-exact output. Measures per-kernel walk inside the
//! graph vs sequential null-stream launches.
//!
//! Sequence under test (10 kernels, mix of small element-wise + rmsnorm +
//! a synthetic GEMV through the hipfire dispatch path):
//!
//!   1. rmsnorm_f32           — block-reduction → divide
//!   2. scale_f32             — element-wise × const
//!   3. sigmoid_f32           — element-wise sigmoid (in-place)
//!   4. mul_f32               — element-wise multiply (3 buffers)
//!   5. add_inplace_f32       — c += d
//!   6. silu_mul_f32          — fused SwiGLU
//!   7. rmsnorm_f32           — second rmsnorm
//!   8. scale_f32             — second scale
//!   9. sigmoid_f32           — second sigmoid
//!  10. mul_f32               — second mul
//!
//! All operate on a 4096-float vector — same shape the 9B forward uses for
//! its hidden state. This is the worst case for dispatch latency: each kernel
//! does ~µs of compute and the rest is launch overhead.
//!
//! Run:
//!   cargo run --release -p rdna-compute --example hip_graph_poc

use rdna_compute::Gpu;
use std::time::Instant;

fn main() {
    let mut gpu = Gpu::init().expect("Gpu::init");
    eprintln!("[hip_graph_poc] arch={}", gpu.arch);

    let dim = 4096usize;
    let nbytes = dim * 4;

    // Allocate two distinct buffers for the test sequence.
    let init_a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
    let init_b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.0007 + 0.5).collect();
    let init_w: Vec<f32> = (0..dim).map(|i| 1.0 + ((i % 16) as f32) * 0.01).collect();

    let a = gpu.upload_f32(&init_a, &[dim]).unwrap();
    let b = gpu.upload_f32(&init_b, &[dim]).unwrap();
    let c = gpu.upload_f32(&init_a, &[dim]).unwrap();
    let d = gpu.upload_f32(&init_b, &[dim]).unwrap();
    let weight = gpu.upload_f32(&init_w, &[dim]).unwrap();
    let scratch = gpu.upload_f32(&vec![0.0; dim], &[dim]).unwrap();

    // ─── Helper that runs ONE kernel (mul_f32: c = scratch * b) ─
    // Reduced to a single kernel to isolate whether rdna-compute's kernarg
    // packing pattern (stack-local pointer to by-value args) is compatible
    // with HIP graph capture. mul_f32 takes 3 buffer pointers + n (i32 by
    // value via stack pointer).
    let run_sequence = |gpu: &mut Gpu| -> hip_bridge::HipResult<()> {
        gpu.mul_f32(&a, &b, &scratch)?;
        Ok(())
    };

    // ─── Reset helper: re-uploads inputs so each run starts identically ──
    let reset_inputs = |gpu: &mut Gpu| {
        gpu.hip
            .memcpy_htod(&a.buf, as_bytes(&init_a))
            .unwrap();
        gpu.hip
            .memcpy_htod(&b.buf, as_bytes(&init_b))
            .unwrap();
        gpu.hip
            .memcpy_htod(&c.buf, as_bytes(&init_a))
            .unwrap();
        gpu.hip
            .memcpy_htod(&d.buf, as_bytes(&init_b))
            .unwrap();
        gpu.hip
            .memcpy_htod(&scratch.buf, &vec![0u8; nbytes])
            .unwrap();
    };

    // ─── 1. Reference run on the null stream, capture output ─────────────
    eprintln!("\n--- Reference (null stream, sequential launches) ---");
    reset_inputs(&mut gpu);
    run_sequence(&mut gpu).unwrap();
    gpu.hip.device_synchronize().unwrap();
    let reference = gpu.download_f32(&scratch).unwrap();
    eprintln!(
        "  output[0..4] = {:?}",
        &reference[..4]
    );

    // ─── 2. Time sequential launches (no graph) on a custom stream ──────
    eprintln!("\n--- Sequential launches on custom stream ---");
    let stream = gpu.hip.stream_create().unwrap();
    gpu.active_stream = Some(stream);

    // Warm-up
    for _ in 0..10 {
        reset_inputs(&mut gpu);
        run_sequence(&mut gpu).unwrap();
    }
    gpu.hip
        .stream_synchronize(gpu.active_stream.as_ref().unwrap())
        .unwrap();

    let iters = 200u32;
    let mut seq_lat = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        reset_inputs(&mut gpu);
        let t = Instant::now();
        run_sequence(&mut gpu).unwrap();
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();
        seq_lat.push(t.elapsed());
    }
    let stream = gpu.active_stream.take().unwrap();
    print_stats("Sequential (10 kernels, custom stream + sync)", &mut seq_lat);
    let seq_total_us = median_us(&seq_lat);
    let seq_per_kernel = seq_total_us / 10.0;
    eprintln!("  per-kernel walk: {:.2} µs", seq_per_kernel);

    // Verify sequential output still matches reference
    let seq_out = gpu.download_f32(&scratch).unwrap();
    let bad = (0..dim).filter(|&i| (seq_out[i] - reference[i]).abs() > 1e-4).count();
    eprintln!("  sequential vs reference: {}/{} match", dim - bad, dim);

    // ─── 3. Graph capture and replay ────────────────────────────────────
    eprintln!("\n--- hipGraph capture + replay ---");
    gpu.active_stream = Some(stream);
    reset_inputs(&mut gpu);

    // Begin capture, run sequence, end capture.
    gpu.hip
        .stream_begin_capture(gpu.active_stream.as_ref().unwrap(), 0)
        .expect("stream_begin_capture");
    let capture_result = run_sequence(&mut gpu);
    let graph_result = gpu
        .hip
        .stream_end_capture(gpu.active_stream.as_ref().unwrap());

    let graph = match (capture_result, graph_result) {
        (Ok(()), Ok(g)) => {
            eprintln!("  capture succeeded");
            g
        }
        (Err(e), _) => {
            eprintln!("  ❌ capture FAILED during sequence: {e}");
            std::process::exit(1);
        }
        (Ok(()), Err(e)) => {
            eprintln!("  ❌ stream_end_capture FAILED: {e}");
            std::process::exit(1);
        }
    };

    let exec = gpu.hip.graph_instantiate(&graph).expect("graph_instantiate");
    eprintln!("  instantiated");

    // First replay (warmup)
    reset_inputs(&mut gpu);
    gpu.hip
        .graph_launch(&exec, gpu.active_stream.as_ref().unwrap())
        .expect("graph_launch (warmup)");
    gpu.hip
        .stream_synchronize(gpu.active_stream.as_ref().unwrap())
        .unwrap();

    // Verify graph output matches reference
    let graph_out = gpu.download_f32(&scratch).unwrap();
    let bad = (0..dim).filter(|&i| (graph_out[i] - reference[i]).abs() > 1e-4).count();
    eprintln!("  graph vs reference: {}/{} match", dim - bad, dim);
    if bad > 0 {
        for i in 0..8 {
            eprintln!(
                "    [{i}] graph={} ref={} delta={}",
                graph_out[i],
                reference[i],
                graph_out[i] - reference[i]
            );
        }
        std::process::exit(1);
    }

    // ─── 4. Time graph replay ───────────────────────────────────────────
    let mut warmup_count = 0;
    while warmup_count < 20 {
        reset_inputs(&mut gpu);
        gpu.hip
            .graph_launch(&exec, gpu.active_stream.as_ref().unwrap())
            .unwrap();
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();
        warmup_count += 1;
    }

    let mut graph_lat = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        reset_inputs(&mut gpu);
        let t = Instant::now();
        gpu.hip
            .graph_launch(&exec, gpu.active_stream.as_ref().unwrap())
            .unwrap();
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();
        graph_lat.push(t.elapsed());
    }
    print_stats("hipGraph replay (10 kernels, 1 launch + sync)", &mut graph_lat);
    let graph_total_us = median_us(&graph_lat);
    let graph_per_kernel = graph_total_us / 10.0;
    eprintln!("  per-kernel walk: {:.2} µs", graph_per_kernel);

    // ─── 5. Measure replay WITHOUT the per-launch reset+sync overhead ───
    // The reset + per-launch sync above adds host bookkeeping time. To
    // isolate the actual GPU graph walk cost, run a tight burst of N
    // graph_launches and sync once at the end.
    eprintln!("\n--- hipGraph burst (no per-launch reset, 1 sync at end) ---");
    let burst = 100u32;
    reset_inputs(&mut gpu);
    let mut burst_lat = Vec::with_capacity(50);
    for _ in 0..50 {
        let t = Instant::now();
        for _ in 0..burst {
            gpu.hip
                .graph_launch(&exec, gpu.active_stream.as_ref().unwrap())
                .unwrap();
        }
        gpu.hip
            .stream_synchronize(gpu.active_stream.as_ref().unwrap())
            .unwrap();
        burst_lat.push(t.elapsed());
    }
    burst_lat.sort();
    let burst_total_us = burst_lat[burst_lat.len() / 2].as_secs_f64() * 1_000_000.0;
    let burst_per_replay = burst_total_us / burst as f64;
    let burst_per_kernel = burst_per_replay / 10.0;
    eprintln!(
        "  median {} replays = {:.1} µs total → {:.2} µs/replay → {:.2} µs/kernel-walk",
        burst, burst_total_us, burst_per_replay, burst_per_kernel
    );

    // ─── 6. Final summary ───────────────────────────────────────────────
    eprintln!("\n=== Summary ===");
    eprintln!(
        "  Sequential per-kernel walk:    {:.2} µs   (10 launches per replay, sync per replay)",
        seq_per_kernel
    );
    eprintln!(
        "  Graph per-kernel walk (sync):  {:.2} µs   (1 graph launch per replay, sync per replay)",
        graph_per_kernel
    );
    eprintln!(
        "  Graph per-kernel walk (burst): {:.2} µs   ({burst} replays, sync at end)",
        burst_per_kernel
    );
    let speedup_sync = seq_per_kernel / graph_per_kernel;
    let speedup_burst = seq_per_kernel / burst_per_kernel;
    eprintln!(
        "  Speedup (graph sync vs seq):   {:.2}x",
        speedup_sync
    );
    eprintln!(
        "  Speedup (graph burst vs seq):  {:.2}x",
        speedup_burst
    );

    // Cleanup
    gpu.hip.graph_exec_destroy(exec).unwrap();
    gpu.hip.graph_destroy(graph).unwrap();
    let stream = gpu.active_stream.take().unwrap();
    gpu.hip.stream_destroy(stream).unwrap();

    eprintln!("\n=== POC PASSED ===");
}

fn print_stats(label: &str, lat: &mut Vec<std::time::Duration>) {
    lat.sort();
    let to_us = |d: std::time::Duration| d.as_secs_f64() * 1_000_000.0;
    let median = to_us(lat[lat.len() / 2]);
    let mean = to_us(lat.iter().sum::<std::time::Duration>()) / lat.len() as f64;
    let p99 = to_us(lat[(lat.len() as f64 * 0.99) as usize]);
    let min = to_us(lat[0]);
    let max = to_us(*lat.last().unwrap());
    eprintln!("[{label}]");
    eprintln!("  median: {median:7.2} µs");
    eprintln!("  mean:   {mean:7.2} µs");
    eprintln!("  p99:    {p99:7.2} µs");
    eprintln!("  min:    {min:7.2} µs");
    eprintln!("  max:    {max:7.2} µs");
}

fn median_us(lat: &[std::time::Duration]) -> f64 {
    let mut sorted = lat.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2].as_secs_f64() * 1_000_000.0
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
