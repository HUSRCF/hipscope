// dp4a throughput microbenchmark for gfx1100.
use hip_bridge::HipRuntime;
use std::ffi::c_void;

fn main() {
    let hip = HipRuntime::load().unwrap();
    hip.set_device(0).unwrap();
    let arch = hip.get_arch(0).unwrap_or("gfx1100".into());
    eprintln!("GPU: {arch}");

    let hsaco_path = format!("{}/results/dp4a_bench.hsaco", env!("CARGO_MANIFEST_DIR"));
    let module = hip.module_load(&hsaco_path).unwrap();
    let fn_dp4a = hip.module_get_function(&module, "bench_dp4a").unwrap();
    let fn_fp32 = hip.module_get_function(&module, "bench_fp32").unwrap();

    // 16M elements — enough to saturate the GPU
    let n = 16 * 1024 * 1024;
    let threads = 256u32;
    let blocks = 512u32;
    let total_threads = (threads * blocks) as usize;

    // Allocate: dp4a uses packed int32 (4 INT8 per element)
    let buf_a_i = hip.malloc(n * 4).unwrap(); // int32[N]
    let buf_b_i = hip.malloc(n * 4).unwrap();
    let buf_out_i = hip.malloc(total_threads * 4).unwrap();
    hip.memset(&buf_a_i, 0x01, n * 4).unwrap(); // fill with 1s
    hip.memset(&buf_b_i, 0x01, n * 4).unwrap();

    // FP32 uses 4x the elements (4 floats per dp4a element)
    let buf_a_f = hip.malloc(n * 4 * 4).unwrap(); // float[N*4]
    let buf_b_f = hip.malloc(n * 4 * 4).unwrap();
    let buf_out_f = hip.malloc(total_threads * 4).unwrap();
    hip.memset(&buf_a_f, 0x3f, n * 4 * 4).unwrap(); // ~1.0f
    hip.memset(&buf_b_f, 0x3f, n * 4 * 4).unwrap();

    let ev_start = hip.event_create().unwrap();
    let ev_stop = hip.event_create().unwrap();
    let iters = 200;

    // Warmup
    for _ in 0..5 {
        let mut ap = buf_a_i.as_ptr(); let mut bp = buf_b_i.as_ptr();
        let mut op = buf_out_i.as_ptr(); let mut nv = n as i32;
        let mut args: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut bp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut nv as *mut _ as *mut c_void,
        ];
        unsafe { hip.launch_kernel(&fn_dp4a, [blocks, 1, 1], [threads, 1, 1], 0, None, &mut args).unwrap(); }
    }
    hip.device_synchronize().unwrap();

    // Benchmark dp4a
    hip.event_record(&ev_start, None).unwrap();
    for _ in 0..iters {
        let mut ap = buf_a_i.as_ptr(); let mut bp = buf_b_i.as_ptr();
        let mut op = buf_out_i.as_ptr(); let mut nv = n as i32;
        let mut args: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut bp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut nv as *mut _ as *mut c_void,
        ];
        unsafe { hip.launch_kernel(&fn_dp4a, [blocks, 1, 1], [threads, 1, 1], 0, None, &mut args).unwrap(); }
    }
    hip.event_record(&ev_stop, None).unwrap();
    hip.event_synchronize(&ev_stop).unwrap();
    let ms_dp4a = hip.event_elapsed_ms(&ev_start, &ev_stop).unwrap();

    // Benchmark fp32
    hip.event_record(&ev_start, None).unwrap();
    for _ in 0..iters {
        let mut ap = buf_a_f.as_ptr(); let mut bp = buf_b_f.as_ptr();
        let mut op = buf_out_f.as_ptr(); let mut nv = n as i32;
        let mut args: Vec<*mut c_void> = vec![
            &mut ap as *mut _ as *mut c_void, &mut bp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void, &mut nv as *mut _ as *mut c_void,
        ];
        unsafe { hip.launch_kernel(&fn_fp32, [blocks, 1, 1], [threads, 1, 1], 0, None, &mut args).unwrap(); }
    }
    hip.event_record(&ev_stop, None).unwrap();
    hip.event_synchronize(&ev_stop).unwrap();
    let ms_fp32 = hip.event_elapsed_ms(&ev_start, &ev_stop).unwrap();

    let total_macs = n as f64 * 4.0 * iters as f64; // 4 MACs per element
    let gmacs_dp4a = total_macs / (ms_dp4a as f64 * 1e-3) / 1e9;
    let gmacs_fp32 = total_macs / (ms_fp32 as f64 * 1e-3) / 1e9;

    eprintln!("\n=== dp4a vs FP32 throughput ({n} elements × {iters} iterations) ===");
    eprintln!("dp4a (v_dot4_i32_iu8): {:.1} ms → {:.0} GMAC/s", ms_dp4a, gmacs_dp4a);
    eprintln!("FP32 (v_fmac_f32×4):   {:.1} ms → {:.0} GMAC/s", ms_fp32, gmacs_fp32);
    eprintln!("Speedup: {:.2}x", ms_fp32 as f64 / ms_dp4a as f64);
}
