// Bandwidth microbenchmark: measure actual GB/s for turbo4 vs Q8 attention
// at various context lengths matching the 9B Qwen3.5 config.

use hip_bridge::HipRuntime;
use std::ffi::c_void;

fn main() {
    if let Err(e) = run() { eprintln!("FATAL: {e}"); std::process::exit(1); }
}

fn compile(arch: &str, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let src = format!("{}/kernels/{}.hip", env!("CARGO_MANIFEST_DIR"), name);
    let out = format!("{}/results/{}.hsaco", env!("CARGO_MANIFEST_DIR"), name);
    let status = std::process::Command::new("hipcc")
        .args(["--genco", &format!("--offload-arch={arch}"), "-O3",
               "-I", &format!("{}/../kernels/src", env!("CARGO_MANIFEST_DIR")),
               "-o", &out, &src])
        .status()?;
    if !status.success() { return Err(format!("hipcc failed for {name}").into()); }
    Ok(out)
}

fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
    }).collect()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let hip = HipRuntime::load()?;
    hip.set_device(0)?;
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1010".to_string());

    let hsaco = compile(&arch, "bw_bench")?;
    let module = hip.module_load(&hsaco)?;
    let fn_turbo4 = hip.module_get_function(&module, "bw_attention_turbo4")?;
    let fn_q8 = hip.module_get_function(&module, "bw_attention_q8")?;

    // 9B Qwen3.5 config (FullAttn layers)
    let n_heads: i32 = 32;
    let n_kv_heads: i32 = 8;
    let head_dim: i32 = 128;
    let scale_attn: f32 = 1.0 / (head_dim as f32).sqrt();
    let peak_bw: f64 = 448.0; // GB/s theoretical

    // Turbo4: 68 bytes/head, Q8: 136 bytes/head
    let t4_bytes_per_pos = n_kv_heads as usize * 68;   // 544
    let q8_bytes_per_pos = n_kv_heads as usize * 136;   // 1088

    let signs1 = gen_fwht_signs(42, head_dim as usize);
    let signs2 = gen_fwht_signs(1042, head_dim as usize);
    let buf_s1 = hip.malloc(head_dim as usize * 4)?;
    let buf_s2 = hip.malloc(head_dim as usize * 4)?;
    hip.memcpy_htod(&buf_s1, unsafe { std::slice::from_raw_parts(signs1.as_ptr() as *const u8, signs1.len() * 4) })?;
    hip.memcpy_htod(&buf_s2, unsafe { std::slice::from_raw_parts(signs2.as_ptr() as *const u8, signs2.len() * 4) })?;

    let q_size = n_heads as usize * head_dim as usize * 4;
    let out_size = n_heads as usize * head_dim as usize * 4;
    let buf_q = hip.malloc(q_size)?;
    let buf_out = hip.malloc(out_size)?;
    hip.memset(&buf_q, 0x3f, q_size)?; // fill with ~1.0f

    let ev_start = hip.event_create()?;
    let ev_stop = hip.event_create()?;

    eprintln!("=== Attention Bandwidth Benchmark (9B config: {} heads, {} KV heads, dim {}) ===", n_heads, n_kv_heads, head_dim);
    eprintln!("Peak VRAM BW: {peak_bw} GB/s");
    eprintln!("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "seq_len", "T4 us", "T4 GB/s", "T4 %peak", "Q8 us", "Q8 GB/s", "Q8 %peak");

    for &seq_len in &[128, 256, 512, 1024, 2048, 4096] {
        let max_seq = seq_len + 64;

        // Allocate KV caches
        let t4_cache_size = max_seq * t4_bytes_per_pos;
        let q8_cache_size = max_seq * q8_bytes_per_pos;
        let buf_k_t4 = hip.malloc(t4_cache_size)?;
        let buf_v_t4 = hip.malloc(t4_cache_size)?;
        let buf_k_q8 = hip.malloc(q8_cache_size)?;
        let buf_v_q8 = hip.malloc(q8_cache_size)?;
        hip.memset(&buf_k_t4, 0x42, t4_cache_size)?;
        hip.memset(&buf_v_t4, 0x42, t4_cache_size)?;
        hip.memset(&buf_k_q8, 0x42, q8_cache_size)?;
        hip.memset(&buf_v_q8, 0x42, q8_cache_size)?;

        let iters = if seq_len <= 512 { 500 } else if seq_len <= 2048 { 200 } else { 100 };

        // Turbo4
        hip.device_synchronize()?;
        hip.event_record(&ev_start, None)?;
        for _ in 0..iters {
            let mut qp = buf_q.as_ptr();
            let mut kp = buf_k_t4.as_ptr();
            let mut vp = buf_v_t4.as_ptr();
            let mut op = buf_out.as_ptr();
            let mut s1p = buf_s1.as_ptr();
            let mut s2p = buf_s2.as_ptr();
            let mut sl = seq_len as i32;
            let mut nkv = n_kv_heads;
            let mut hd = head_dim;
            let mut sc = scale_attn;
            let mut args: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
            ];
            let shared = (seq_len * 4) as u32;
            unsafe { hip.launch_kernel(&fn_turbo4, [n_heads as u32,1,1], [32,1,1], shared, None, &mut args)?; }
        }
        hip.event_record(&ev_stop, None)?;
        hip.event_synchronize(&ev_stop)?;
        let ms_t4 = hip.event_elapsed_ms(&ev_start, &ev_stop)?;
        let us_t4 = ms_t4 as f64 * 1000.0 / iters as f64;
        // Bytes read: K cache + V cache (both full seq_len, all KV heads)
        // Each head reads seq_len * bytes_per_head for K and V
        let bytes_t4 = (seq_len * t4_bytes_per_pos * 2) as f64; // K + V
        let bw_t4 = bytes_t4 / (us_t4 * 1e-6) / 1e9;

        // Q8
        hip.device_synchronize()?;
        hip.event_record(&ev_start, None)?;
        for _ in 0..iters {
            let mut qp = buf_q.as_ptr();
            let mut kp = buf_k_q8.as_ptr();
            let mut vp = buf_v_q8.as_ptr();
            let mut op = buf_out.as_ptr();
            let mut sl = seq_len as i32;
            let mut nkv = n_kv_heads;
            let mut hd = head_dim;
            let mut sc = scale_attn;
            let mut args: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut op as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut sc as *mut _ as *mut c_void,
            ];
            let shared = (seq_len * 4) as u32;
            unsafe { hip.launch_kernel(&fn_q8, [n_heads as u32,1,1], [32,1,1], shared, None, &mut args)?; }
        }
        hip.event_record(&ev_stop, None)?;
        hip.event_synchronize(&ev_stop)?;
        let ms_q8 = hip.event_elapsed_ms(&ev_start, &ev_stop)?;
        let us_q8 = ms_q8 as f64 * 1000.0 / iters as f64;
        let bytes_q8 = (seq_len * q8_bytes_per_pos * 2) as f64;
        let bw_q8 = bytes_q8 / (us_q8 * 1e-6) / 1e9;

        eprintln!("{:<10} {:>10.1} {:>10.1} {:>9.1}% {:>10.1} {:>10.1} {:>9.1}%",
            seq_len, us_t4, bw_t4, bw_t4/peak_bw*100.0,
            us_q8, bw_q8, bw_q8/peak_bw*100.0);

        hip.free(buf_k_t4)?;
        hip.free(buf_v_t4)?;
        hip.free(buf_k_q8)?;
        hip.free(buf_v_q8)?;
    }

    hip.free(buf_q)?;
    hip.free(buf_out)?;
    hip.free(buf_s1)?;
    hip.free(buf_s2)?;

    Ok(())
}
