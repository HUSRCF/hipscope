//! Benchmark: Redline dispatch overhead.
//! Measures per-dispatch latency, multi-dispatch throughput, startup time, and memory.

use redline::device::Device;
use redline::dispatch::{DispatchQueue, KernargBuilder, Kernel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000u32);

    eprintln!("=== Redline Dispatch Benchmark ===");
    eprintln!("Iterations: {}\n", iterations);

    // --- Startup time ---
    let t_start = std::time::Instant::now();
    let dev = Device::open(None).unwrap();
    let dq = DispatchQueue::new(&dev).unwrap();
    let startup_device = t_start.elapsed();

    // Compile vector_add with __launch_bounds__ (so no hidden arg overhead)
    let hip_src = r#"
#include <hip/hip_runtime.h>
extern "C" __launch_bounds__(256)
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
"#;
    std::fs::write("/tmp/redline_bench_va.hip", hip_src).unwrap();
    let out = std::process::Command::new("hipcc")
        .args(["--genco", "--offload-arch=gfx1010", "-O3",
               "-o", "/tmp/redline_bench_va.hsaco", "/tmp/redline_bench_va.hip"])
        .output().expect("hipcc");
    assert!(out.status.success(), "hipcc: {}", String::from_utf8_lossy(&out.stderr));

    let module = dev.load_module_file("/tmp/redline_bench_va.hsaco").unwrap();
    let kernel = Kernel::find(&module, "vector_add").expect("kernel not found");
    let startup_total = t_start.elapsed();

    eprintln!("[startup] device+queue: {:.2}ms, total (incl compile): {:.2}ms",
        startup_device.as_secs_f64() * 1000.0, startup_total.as_secs_f64() * 1000.0);

    // Set up buffers (256 elements = 1KB)
    let n = 256u32;
    let nbytes = (n as usize) * 4;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 2.0).collect();

    let a_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let b_buf = dev.alloc_vram(nbytes as u64).unwrap();
    let c_buf = dev.alloc_vram(nbytes as u64).unwrap();
    dev.upload(&a_buf, as_bytes(&a_data)).unwrap();
    dev.upload(&b_buf, as_bytes(&b_data)).unwrap();

    let mut ka = KernargBuilder::new(28);
    ka.write_ptr(0, a_buf.gpu_addr).write_ptr(8, b_buf.gpu_addr)
      .write_ptr(16, c_buf.gpu_addr).write_u32(24, n);

    // Warm up (first dispatch is always slower)
    for _ in 0..10 {
        dq.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1],
            ka.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();
    }

    // --- Per-dispatch latency (includes submit + fence wait) ---
    let mut latencies = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        let t = std::time::Instant::now();
        dq.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1],
            ka.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();
        latencies.push(t.elapsed());
    }

    latencies.sort();
    let to_us = |d: std::time::Duration| d.as_secs_f64() * 1_000_000.0;
    let median = to_us(latencies[latencies.len() / 2]);
    let mean = to_us(latencies.iter().sum::<std::time::Duration>()) / latencies.len() as f64;
    let p99 = to_us(latencies[(latencies.len() as f64 * 0.99) as usize]);
    let min = to_us(latencies[0]);
    let max = to_us(*latencies.last().unwrap());

    eprintln!("\n[per-dispatch] {} iterations, vector_add 256 elements:", iterations);
    eprintln!("  median: {:.1} µs", median);
    eprintln!("  mean:   {:.1} µs", mean);
    eprintln!("  p99:    {:.1} µs", p99);
    eprintln!("  min:    {:.1} µs", min);
    eprintln!("  max:    {:.1} µs", max);

    // --- Multi-dispatch throughput (sequential submits) ---
    let batch = 200u32;
    let t_batch = std::time::Instant::now();
    for _ in 0..batch {
        dq.dispatch(&dev, kernel, [1, 1, 1], [256, 1, 1],
            ka.as_bytes(), &[&module.code_buf, &a_buf, &b_buf, &c_buf]).unwrap();
    }
    let batch_time = t_batch.elapsed();
    let per_kernel = batch_time.as_secs_f64() * 1_000_000.0 / batch as f64;
    eprintln!("\n[{}-dispatch sequential] total: {:.2}ms, per-kernel: {:.1} µs",
        batch, batch_time.as_secs_f64() * 1000.0, per_kernel);

    // --- Memory overhead ---
    let rss = get_rss_kb();
    eprintln!("\n[memory] RSS: {} KB ({:.1} MB)", rss, rss as f64 / 1024.0);

    // --- Verify correctness ---
    let mut c_raw = vec![0u8; nbytes];
    dev.download(&c_buf, &mut c_raw).unwrap();
    let c: &[f32] = unsafe { std::slice::from_raw_parts(c_raw.as_ptr() as *const f32, n as usize) };
    let bad = (0..n as usize).filter(|&i| (c[i] - (i as f32) * 3.0).abs() > 0.001).count();
    eprintln!("\n[verify] vector_add: {}/{} correct", n as usize - bad, n);

    // --- Print machine-readable results ---
    println!("BENCH_REDLINE_MEDIAN_US={:.1}", median);
    println!("BENCH_REDLINE_MEAN_US={:.1}", mean);
    println!("BENCH_REDLINE_P99_US={:.1}", p99);
    println!("BENCH_REDLINE_MIN_US={:.1}", min);
    println!("BENCH_REDLINE_MAX_US={:.1}", max);
    println!("BENCH_REDLINE_BATCH_TOTAL_MS={:.2}", batch_time.as_secs_f64() * 1000.0);
    println!("BENCH_REDLINE_BATCH_PER_KERNEL_US={:.1}", per_kernel);
    println!("BENCH_REDLINE_STARTUP_MS={:.2}", startup_device.as_secs_f64() * 1000.0);
    println!("BENCH_REDLINE_RSS_KB={}", rss);

    dq.destroy(&dev);
}

fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn get_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}
