// TurboQuant KV cache round-trip diagnosis
// Tests the turbo4 pipeline stage by stage with known vectors.

use hip_bridge::HipRuntime;
use std::ffi::c_void;
use std::f32::consts::PI;

fn main() {
    if let Err(e) = run() {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
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

/// Generate FWHT signs matching the engine's gen_fwht_signs(seed, n)
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
    eprintln!("GPU: {arch}");

    // Compile diagnostic kernel
    let hsaco = compile(&arch, "turbo_roundtrip_test")?;
    let module = hip.module_load(&hsaco)?;
    let fn_test = hip.module_get_function(&module, "turbo4_roundtrip_test")?;

    let head_dim = 128usize;

    // Generate sign tables (same seeds as engine)
    let signs1 = gen_fwht_signs(42, head_dim);
    let signs2 = gen_fwht_signs(1042, head_dim);
    let buf_s1 = hip.malloc(head_dim * 4)?;
    let buf_s2 = hip.malloc(head_dim * 4)?;
    hip.memcpy_htod(&buf_s1, f32_bytes(&signs1))?;
    hip.memcpy_htod(&buf_s2, f32_bytes(&signs2))?;

    // Allocate stage buffers
    let buf_in = hip.malloc(head_dim * 4)?;
    let buf_out = hip.malloc(head_dim * 4)?;
    let buf_norm = hip.malloc(head_dim * 4)?;
    let buf_fwht = hip.malloc(head_dim * 4)?;
    let buf_deq = hip.malloc(head_dim * 4)?;
    let buf_dbg = hip.malloc(8 * 4)?;

    std::fs::create_dir_all(format!("{}/results/turbo_diagnosis", env!("CARGO_MANIFEST_DIR")))?;

    // ── Test vectors ──
    let test_vectors: Vec<(&str, Vec<f32>)> = vec![
        ("all_ones", vec![1.0f32; head_dim]),
        ("counting", (0..head_dim).map(|i| i as f32).collect()),
        ("single_hot_0", {
            let mut v = vec![0.0f32; head_dim]; v[0] = 1.0; v
        }),
        ("single_hot_64", {
            let mut v = vec![0.0f32; head_dim]; v[64] = 1.0; v
        }),
        ("alternating", (0..head_dim).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()),
        ("sin_wave", (0..head_dim).map(|i| (i as f32 * PI / 16.0).sin()).collect()),
        ("gaussian_like", (0..head_dim).map(|i| {
            let x = (i as f32 - 64.0) / 20.0;
            (-0.5 * x * x).exp()
        }).collect()),
    ];

    for (name, input) in &test_vectors {
        eprintln!("\n=== Test vector: {name} ===");

        hip.memcpy_htod(&buf_in, f32_bytes(input))?;

        // Launch roundtrip test
        let mut hd = head_dim as i32;
        {
            let mut inp = buf_in.as_ptr();
            let mut outp = buf_out.as_ptr();
            let mut np = buf_norm.as_ptr();
            let mut fp = buf_fwht.as_ptr();
            let mut dp = buf_deq.as_ptr();
            let mut dbgp = buf_dbg.as_ptr();
            let mut s1p = buf_s1.as_ptr();
            let mut s2p = buf_s2.as_ptr();
            let mut args: Vec<*mut c_void> = vec![
                &mut inp as *mut _ as *mut c_void,
                &mut outp as *mut _ as *mut c_void,
                &mut np as *mut _ as *mut c_void,
                &mut fp as *mut _ as *mut c_void,
                &mut dp as *mut _ as *mut c_void,
                &mut dbgp as *mut _ as *mut c_void,
                &mut s1p as *mut _ as *mut c_void,
                &mut s2p as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let shared = ((head_dim + 32) * 4) as u32;
            unsafe { hip.launch_kernel(&fn_test, [1,1,1], [32,1,1], shared, None, &mut args)?; }
        }
        hip.device_synchronize()?;

        // Download all stages
        let output = download_f32(&hip, &buf_out, head_dim)?;
        let after_norm = download_f32(&hip, &buf_norm, head_dim)?;
        let after_fwht = download_f32(&hip, &buf_fwht, head_dim)?;
        let after_deq = download_f32(&hip, &buf_deq, head_dim)?;
        let debug = download_f32(&hip, &buf_dbg, 8)?;

        let cnorm = debug[0];
        let orig_norm = debug[1];
        let recon_norm = debug[2];

        eprintln!("  orig_norm={orig_norm:.6}, recon_norm={recon_norm:.6}, cnorm={cnorm:.6}");

        // Compare stages
        let (max_ab, mse_ab, cos_ab) = compare(input, &output);
        eprintln!("  A→E (full round-trip): max_err={max_ab:.6e}, MSE={mse_ab:.6e}, cos={cos_ab:.10}");

        // Check normalized vector
        let norm_check: f32 = after_norm.iter().map(|x| x*x).sum::<f32>().sqrt();
        eprintln!("  After norm: ||v||={norm_check:.8} (should be ~1.0)");

        // Check FWHT output range
        let fwht_min = after_fwht.iter().copied().fold(f32::MAX, f32::min);
        let fwht_max = after_fwht.iter().copied().fold(f32::MIN, f32::max);
        let fwht_rms: f32 = (after_fwht.iter().map(|x| x*x).sum::<f32>() / head_dim as f32).sqrt();
        eprintln!("  After FWHT: min={fwht_min:.6}, max={fwht_max:.6}, RMS={fwht_rms:.6}");

        // Check dequant vs FWHT (quantization error only)
        let (max_bd, mse_bd, cos_bd) = compare(&after_fwht, &after_deq);
        eprintln!("  B→D (FWHT→dequant, quant error): max_err={max_bd:.6e}, MSE={mse_bd:.6e}, cos={cos_bd:.10}");

        // Also do CPU reference: input → normalize → CPU FWHT → quantize → dequant → CPU inv FWHT
        let cpu_norm = l2_norm(input);
        let cpu_normed: Vec<f32> = input.iter().map(|x| x / cpu_norm).collect();
        let cpu_fwht = cpu_fwht_forward(&cpu_normed, &signs1, &signs2);
        let cpu_dequant = cpu_turbo_dequant_4bit(&cpu_fwht, cpu_norm);
        let cpu_inv = cpu_fwht_inverse(&cpu_dequant, &signs1, &signs2);

        let (max_cpu, mse_cpu, cos_cpu) = compare(input, &cpu_inv);
        eprintln!("  CPU reference round-trip: max_err={max_cpu:.6e}, MSE={mse_cpu:.6e}, cos={cos_cpu:.10}");

        // GPU vs CPU comparison (are they getting the same result?)
        let (max_gc, _, cos_gc) = compare(&output, &cpu_inv);
        eprintln!("  GPU vs CPU output: max_diff={max_gc:.6e}, cos={cos_gc:.10}");

        // Show first 8 elements at each stage
        eprintln!("  First 8 elements:");
        eprintln!("    Input:     {:?}", &input[..8]);
        eprintln!("    Normed:    {:?}", &after_norm[..8]);
        eprintln!("    FWHT:      {:?}", &after_fwht[..8]);
        eprintln!("    Dequant:   {:?}", &after_deq[..8]);
        eprintln!("    Output:    {:?}", &output[..8]);
        eprintln!("    CPU ref:   {:?}", &cpu_inv[..8]);
    }

    // ── Phase 2: Single-head attention test with actual write+attend kernels ──
    eprintln!("\n=== Phase 2: Single-head turbo attention round-trip ===");

    let attn_hsaco = compile(&arch, "turbo_attention_test")?;
    let attn_module = hip.module_load(&attn_hsaco)?;
    let fn_write = hip.module_get_function(&attn_module, "test_turbo4_write_one")?;
    let fn_attend = hip.module_get_function(&attn_module, "test_turbo4_attend_one")?;

    let bytes_per_head = 4 + head_dim / 2; // 68
    let max_seq = 64usize;
    let seq_len = 8;

    // Allocate KV cache (single head)
    let buf_k_cache = hip.malloc(max_seq * bytes_per_head)?;
    let buf_v_cache = hip.malloc(max_seq * bytes_per_head)?;
    hip.memset(&buf_k_cache, 0, max_seq * bytes_per_head)?;
    hip.memset(&buf_v_cache, 0, max_seq * bytes_per_head)?;

    // Generate distinct K and V vectors for each position
    let mut k_vecs = Vec::new();
    let mut v_vecs = Vec::new();
    for t in 0..seq_len {
        let k: Vec<f32> = (0..head_dim).map(|i| ((t as f32 * 7.0 + i as f32) * 0.31).sin()).collect();
        let v: Vec<f32> = (0..head_dim).map(|i| ((t as f32 * 13.0 + i as f32) * 0.17).cos()).collect();
        k_vecs.push(k);
        v_vecs.push(v);
    }

    // Write each position to cache
    let buf_src = hip.malloc(head_dim * 4)?;
    for t in 0..seq_len {
        hip.memcpy_htod(&buf_src, f32_bytes(&k_vecs[t]))?;
        let mut cp = buf_k_cache.as_ptr();
        let mut sp = buf_src.as_ptr();
        let mut pos_val = t as i32;
        let mut s1p = buf_s1.as_ptr();
        let mut s2p = buf_s2.as_ptr();
        let mut hd = head_dim as i32;
        let mut args: Vec<*mut c_void> = vec![
            &mut cp as *mut _ as *mut c_void,
            &mut sp as *mut _ as *mut c_void,
            &mut pos_val as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let shared = ((head_dim + 32) * 4) as u32;
        unsafe { hip.launch_kernel(&fn_write, [1,1,1], [32,1,1], shared, None, &mut args)?; }

        hip.memcpy_htod(&buf_src, f32_bytes(&v_vecs[t]))?;
        let mut vcp = buf_v_cache.as_ptr();
        args[0] = &mut vcp as *mut _ as *mut c_void;
        unsafe { hip.launch_kernel(&fn_write, [1,1,1], [32,1,1], shared, None, &mut args)?; }
    }
    hip.device_synchronize()?;

    // Q = K[4] (should produce strong attention on position 4)
    let q_vec = k_vecs[4].clone();
    let buf_q = hip.malloc(head_dim * 4)?;
    hip.memcpy_htod(&buf_q, f32_bytes(&q_vec))?;

    let buf_attn_out = hip.malloc(head_dim * 4)?;
    let buf_debug_scores = hip.malloc(seq_len * 4)?;
    let scale_attn = 1.0f32 / (head_dim as f32).sqrt();

    {
        let mut qp = buf_q.as_ptr();
        let mut kp = buf_k_cache.as_ptr();
        let mut vp = buf_v_cache.as_ptr();
        let mut op = buf_attn_out.as_ptr();
        let mut dp = buf_debug_scores.as_ptr();
        let mut sl = seq_len as i32;
        let mut s1p = buf_s1.as_ptr();
        let mut s2p = buf_s2.as_ptr();
        let mut hd = head_dim as i32;
        let mut sc = scale_attn;
        let mut args: Vec<*mut c_void> = vec![
            &mut qp as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut op as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut s1p as *mut _ as *mut c_void,
            &mut s2p as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        let shared = (seq_len * 4) as u32;
        unsafe { hip.launch_kernel(&fn_attend, [1,1,1], [32,1,1], shared, None, &mut args)?; }
    }
    hip.device_synchronize()?;

    let attn_out = download_f32(&hip, &buf_attn_out, head_dim)?;
    let attn_scores = download_f32(&hip, &buf_debug_scores, seq_len)?;

    eprintln!("  Q = K[4], seq_len={seq_len}");
    eprintln!("  Attention scores: {:?}", &attn_scores);

    // Expected: attention should concentrate on position 4 (where Q=K)
    // Output should approximate V[4]
    let (max_err, mse, cos) = compare(&v_vecs[4], &attn_out);
    eprintln!("  Output vs V[4]: max_err={max_err:.6}, MSE={mse:.6e}, cosine={cos:.8}");
    eprintln!("  Output[0..8]:  {:?}", &attn_out[..8]);
    eprintln!("  V[4][0..8]:    {:?}", &v_vecs[4][..8]);

    // CPU reference: compute ideal attention (FP32, no quantization)
    let mut cpu_scores = vec![0.0f64; seq_len];
    for t in 0..seq_len {
        let dot: f64 = q_vec.iter().zip(&k_vecs[t]).map(|(a, b)| *a as f64 * *b as f64).sum();
        cpu_scores[t] = dot * scale_attn as f64;
    }
    let max_score = cpu_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> = cpu_scores.iter().map(|s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    let cpu_weights: Vec<f64> = exp_scores.iter().map(|e| e / sum_exp).collect();
    eprintln!("  CPU ideal weights: {:?}", &cpu_weights.iter().map(|w| *w as f32).collect::<Vec<_>>());

    let mut cpu_out = vec![0.0f64; head_dim];
    for t in 0..seq_len {
        for i in 0..head_dim {
            cpu_out[i] += cpu_weights[t] * v_vecs[t][i] as f64;
        }
    }
    let cpu_out_f32: Vec<f32> = cpu_out.iter().map(|x| *x as f32).collect();
    let (cpu_max, cpu_mse, cpu_cos) = compare(&cpu_out_f32, &attn_out);
    eprintln!("  GPU turbo vs CPU ideal attention: max_err={cpu_max:.6}, MSE={cpu_mse:.6e}, cos={cpu_cos:.8}");

    hip.free(buf_k_cache)?;
    hip.free(buf_v_cache)?;
    hip.free(buf_src)?;
    hip.free(buf_q)?;
    hip.free(buf_attn_out)?;
    hip.free(buf_debug_scores)?;

    eprintln!("\n=== Diagnosis complete ===");
    Ok(())
}

// ── CPU reference implementations ──

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cpu_fwht_forward(input: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut x: Vec<f32> = input.iter().zip(signs1).map(|(a, s)| a * s).collect();

    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }

    let scale = 1.0 / (n as f32).sqrt();
    x.iter_mut().zip(signs2).for_each(|(v, s)| *v *= scale * s);
    x
}

fn cpu_fwht_inverse(input: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut x: Vec<f32> = input.iter().zip(signs2).map(|(a, s)| a * s).collect();

    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }

    let scale = 1.0 / (n as f32).sqrt();
    x.iter_mut().zip(signs1).for_each(|(v, s)| *v *= scale * s);
    x
}

// Lloyd-Max centroids for N(0, 1/128) after unit-norm + FWHT
const TURBO_C4: [f32; 16] = [
    -0.241565, -0.182875, -0.143012, -0.111016, -0.083262, -0.057983, -0.034295, -0.011225,
     0.011225,  0.034295,  0.057983,  0.083262,  0.111016,  0.143012,  0.182875,  0.241565,
];

// Quantize thresholds (same as turbo_quantize_4bit in turbo_common.h)
fn cpu_turbo_quantize_4bit(x: f32) -> usize {
    let mut idx = 0usize;
    if x > -0.212220 { idx += 1; }
    if x > -0.162944 { idx += 1; }
    if x > -0.127014 { idx += 1; }
    if x > -0.097139 { idx += 1; }
    if x > -0.070622 { idx += 1; }
    if x > -0.046139 { idx += 1; }
    if x > -0.022760 { idx += 1; }
    if x > 0.0       { idx += 1; }
    if x > 0.022760  { idx += 1; }
    if x > 0.046139  { idx += 1; }
    if x > 0.070622  { idx += 1; }
    if x > 0.097139  { idx += 1; }
    if x > 0.127014  { idx += 1; }
    if x > 0.162944  { idx += 1; }
    if x > 0.212220  { idx += 1; }
    idx
}

fn cpu_turbo_quantize_4bit_vec(fwht_output: &[f32]) -> (Vec<usize>, f32, f32) {
    let indices: Vec<usize> = fwht_output.iter().map(|&x| cpu_turbo_quantize_4bit(x)).collect();
    let recon_sq: f32 = indices.iter().map(|&i| TURBO_C4[i] * TURBO_C4[i]).sum();
    let recon_norm = recon_sq.sqrt();
    (indices, recon_norm, 0.0)
}

fn cpu_turbo_dequant_4bit(fwht_output: &[f32], orig_norm: f32) -> Vec<f32> {
    let (indices, recon_norm, _) = cpu_turbo_quantize_4bit_vec(fwht_output);
    let cnorm = if recon_norm > 1e-10 { orig_norm / recon_norm } else { orig_norm };
    indices.iter().map(|&i| cnorm * TURBO_C4[i]).collect()
}

// ── Helpers ──

fn f32_bytes(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}

fn download_f32(hip: &HipRuntime, buf: &hip_bridge::DeviceBuffer, n: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut data = vec![0.0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, n * 4) };
    hip.memcpy_dtoh(bytes, buf)?;
    Ok(data)
}

fn compare(a: &[f32], b: &[f32]) -> (f32, f32, f64) {
    let n = a.len() as f64;
    let mut max_err: f32 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut dot: f64 = 0.0;
    let mut na: f64 = 0.0;
    let mut nb: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        max_err = max_err.max(diff);
        sum_sq += (diff as f64).powi(2);
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    let mse = sum_sq / n;
    let cos = if na < 1e-30 && nb < 1e-30 { 1.0 }
        else if na > 0.0 && nb > 0.0 { dot / (na.sqrt() * nb.sqrt()) }
        else { 0.0 };
    (max_err, mse as f32, cos)
}
