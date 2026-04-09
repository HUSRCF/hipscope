// MagnumQuant Phase 1+2 test harness
// Compiles butterfly Givens rotation kernel, runs on GPU, verifies correctness.

use hip_bridge::HipRuntime;
use std::ffi::c_void;
use std::f32::consts::PI;
use std::process::Command;

fn main() {
    if let Err(e) = run() {
        eprintln!("FATAL: {e}");
        std::process::exit(1);
    }
}

fn compile_kernel(arch: &str) -> Result<String, Box<dyn std::error::Error>> {
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/magnum_butterfly.hip");
    let out_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    std::fs::create_dir_all(out_dir)?;
    let out = format!("{out_dir}/magnum_butterfly.hsaco");

    let status = Command::new("hipcc")
        .args(["--genco", &format!("--offload-arch={arch}"), "-O3", "-o", &out, src])
        .status()?;
    if !status.success() {
        return Err(format!("hipcc failed with {status}").into());
    }
    Ok(out)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    // ── Init GPU ──
    let hip = HipRuntime::load()?;
    let count = hip.device_count()?;
    if count == 0 {
        return Err("no GPU found".into());
    }
    hip.set_device(0)?;
    let arch = hip.get_arch(0).unwrap_or_else(|_| "gfx1010".to_string());
    let (_, vtotal) = hip.get_vram_info().unwrap_or((0, 0));
    eprintln!("GPU: {} ({:.1} GB VRAM)", arch, vtotal as f64 / 1e9);

    // ── Compile kernel ──
    let hsaco_path = compile_kernel(&arch)?;
    eprintln!("Compiled: {hsaco_path}");
    let module = hip.module_load(&hsaco_path)?;

    let fn_fwd = hip.module_get_function(&module, "magnum_butterfly_rotate_f32")?;
    let fn_inv = hip.module_get_function(&module, "magnum_butterfly_rotate_inv_f32")?;
    let fn_adaptive = hip.module_get_function(&module, "magnum_butterfly_adaptive")?;
    let fn_adaptive_inv = hip.module_get_function(&module, "magnum_butterfly_adaptive_inv")?;
    eprintln!("Compiled and loaded 4 kernels from magnum_butterfly.hip");

    // ── Test parameters ──
    let num_vectors: u32 = 4096;
    let total = num_vectors as usize * 32;

    // Input: deterministic pseudo-random vectors (sin-based for reproducibility)
    let input: Vec<f32> = (0..total)
        .map(|i| ((i as f32) * 0.7312 + 0.3).sin() * 2.0)
        .collect();

    // Rotation params: 5 angles, varying per round for expressiveness
    let angles: [f32; 5] = [PI / 7.0, PI / 5.0, PI / 11.0, PI / 3.0, PI / 13.0];
    let params: Vec<f32> = angles.iter()
        .flat_map(|&a| [a.cos(), a.sin()])
        .collect();

    // ── Allocate GPU buffers ──
    let buf_in = hip.malloc(total * 4)?;
    let buf_fwd = hip.malloc(total * 4)?;
    let buf_inv = hip.malloc(total * 4)?;
    let buf_params = hip.malloc(10 * 4)?;

    // Upload
    let input_bytes = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, total * 4)
    };
    let param_bytes = unsafe {
        std::slice::from_raw_parts(params.as_ptr() as *const u8, 10 * 4)
    };
    hip.memcpy_htod(&buf_in, input_bytes)?;
    hip.memcpy_htod(&buf_params, param_bytes)?;

    // ── Phase 1: Forward + inverse round-trip ──
    eprintln!("\n=== Phase 1: Full 5-round butterfly rotation ===");

    // Forward: input → buf_fwd
    {
        let mut in_ptr = buf_in.as_ptr();
        let mut out_ptr = buf_fwd.as_ptr();
        let mut p_ptr = buf_params.as_ptr();
        let mut n = num_vectors;
        let mut args: Vec<*mut c_void> = vec![
            &mut in_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let waves = (num_vectors + 7) / 8; // 256 threads = 8 waves per block
        unsafe {
            hip.launch_kernel(&fn_fwd, [waves, 1, 1], [256, 1, 1], 0, None, &mut args)?;
        }
    }
    hip.device_synchronize()?;

    // Inverse: buf_fwd → buf_inv
    {
        let mut in_ptr = buf_fwd.as_ptr();
        let mut out_ptr = buf_inv.as_ptr();
        let mut p_ptr = buf_params.as_ptr();
        let mut n = num_vectors;
        let mut args: Vec<*mut c_void> = vec![
            &mut in_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let waves = (num_vectors + 7) / 8;
        unsafe {
            hip.launch_kernel(&fn_inv, [waves, 1, 1], [256, 1, 1], 0, None, &mut args)?;
        }
    }
    hip.device_synchronize()?;

    // Download and compare
    let mut recovered = vec![0.0f32; total];
    let recv_bytes = unsafe {
        std::slice::from_raw_parts_mut(recovered.as_mut_ptr() as *mut u8, total * 4)
    };
    hip.memcpy_dtoh(recv_bytes, &buf_inv)?;

    let mut rotated = vec![0.0f32; total];
    let rot_bytes = unsafe {
        std::slice::from_raw_parts_mut(rotated.as_mut_ptr() as *mut u8, total * 4)
    };
    hip.memcpy_dtoh(rot_bytes, &buf_fwd)?;

    // Round-trip error
    let (max_err, mse, cos_sim) = compare_vectors(&input, &recovered);
    eprintln!("  Round-trip (fwd→inv) max error: {:.2e}", max_err);
    eprintln!("  Round-trip MSE: {:.2e}", mse);
    eprintln!("  Round-trip cosine similarity: {:.10}", cos_sim);
    assert!(max_err < 1e-4, "Round-trip error too large: {max_err}");

    // Verify rotation actually changed values (not identity)
    let (rot_max_diff, _, _) = compare_vectors(&input, &rotated);
    eprintln!("  Forward rotation max diff from input: {:.6}", rot_max_diff);
    assert!(rot_max_diff > 0.01, "Rotation didn't change values — identity?");

    // Show sample vectors
    eprintln!("\n  Sample vector 0 (first 8 elements):");
    eprintln!("    Input:   {:?}", &input[0..8]);
    eprintln!("    Rotated: {:?}", &rotated[0..8]);
    eprintln!("    Recover: {:?}", &recovered[0..8]);

    // ── Phase 2: Adaptive mode test ──
    eprintln!("\n=== Phase 2: Adaptive mode selection ===");

    // Create mode buffer: cycle through 0, 1, 2
    let modes: Vec<u8> = (0..num_vectors).map(|i| (i % 3) as u8).collect();
    let buf_modes = hip.malloc(num_vectors as usize)?;
    hip.memcpy_htod(&buf_modes, &modes)?;

    let buf_adapt_fwd = hip.malloc(total * 4)?;
    let buf_adapt_inv = hip.malloc(total * 4)?;

    // Adaptive forward
    {
        let mut in_ptr = buf_in.as_ptr();
        let mut out_ptr = buf_adapt_fwd.as_ptr();
        let mut p_ptr = buf_params.as_ptr();
        let mut m_ptr = buf_modes.as_ptr();
        let mut n = num_vectors;
        let mut args: Vec<*mut c_void> = vec![
            &mut in_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut m_ptr as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let waves = (num_vectors + 7) / 8;
        unsafe {
            hip.launch_kernel(&fn_adaptive, [waves, 1, 1], [256, 1, 1], 0, None, &mut args)?;
        }
    }
    hip.device_synchronize()?;

    // Adaptive inverse
    {
        let mut in_ptr = buf_adapt_fwd.as_ptr();
        let mut out_ptr = buf_adapt_inv.as_ptr();
        let mut p_ptr = buf_params.as_ptr();
        let mut m_ptr = buf_modes.as_ptr();
        let mut n = num_vectors;
        let mut args: Vec<*mut c_void> = vec![
            &mut in_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut m_ptr as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let waves = (num_vectors + 7) / 8;
        unsafe {
            hip.launch_kernel(&fn_adaptive_inv, [waves, 1, 1], [256, 1, 1], 0, None, &mut args)?;
        }
    }
    hip.device_synchronize()?;

    // Download adaptive results
    let mut adapt_recovered = vec![0.0f32; total];
    let arb = unsafe {
        std::slice::from_raw_parts_mut(adapt_recovered.as_mut_ptr() as *mut u8, total * 4)
    };
    hip.memcpy_dtoh(arb, &buf_adapt_inv)?;

    let mut adapt_rotated = vec![0.0f32; total];
    let art = unsafe {
        std::slice::from_raw_parts_mut(adapt_rotated.as_mut_ptr() as *mut u8, total * 4)
    };
    hip.memcpy_dtoh(art, &buf_adapt_fwd)?;

    // Check each mode's round-trip independently
    for mode in 0..3u32 {
        let mode_name = match mode { 0 => "Mode 0 (2 rounds)", 1 => "Mode 1 (3 rounds)", _ => "Mode 2 (5 rounds)" };
        let mut orig_elems = Vec::new();
        let mut recv_elems = Vec::new();
        let mut rot_elems = Vec::new();
        for i in 0..num_vectors {
            if (i % 3) as u32 == mode {
                let start = i as usize * 32;
                orig_elems.extend_from_slice(&input[start..start + 32]);
                recv_elems.extend_from_slice(&adapt_recovered[start..start + 32]);
                rot_elems.extend_from_slice(&adapt_rotated[start..start + 32]);
            }
        }
        let (max_e, mse_e, cos_e) = compare_vectors(&orig_elems, &recv_elems);
        let (rot_diff, _, _) = compare_vectors(&orig_elems, &rot_elems);
        eprintln!("  {mode_name}: round-trip max_err={:.2e}, MSE={:.2e}, cos={:.10}, rotation_diff={:.6}",
            max_e, mse_e, cos_e, rot_diff);
        assert!(max_e < 1e-4, "{mode_name} round-trip error too large: {max_e}");
    }

    // Verify mode 2 (5 rounds) matches the full rotation
    let full_mode2: Vec<f32> = (0..num_vectors)
        .filter(|i| (i % 3) == 2)
        .flat_map(|i| {
            let s = i as usize * 32;
            rotated[s..s + 32].to_vec()
        })
        .collect();
    let adapt_mode2: Vec<f32> = (0..num_vectors)
        .filter(|i| (i % 3) == 2)
        .flat_map(|i| {
            let s = i as usize * 32;
            adapt_rotated[s..s + 32].to_vec()
        })
        .collect();
    let (m2_diff, _, m2_cos) = compare_vectors(&full_mode2, &adapt_mode2);
    eprintln!("  Mode 2 vs full rotation: max_diff={:.2e}, cos={:.10}", m2_diff, m2_cos);
    assert!(m2_diff < 1e-6, "Mode 2 should match full rotation exactly");

    // ── Phase 3: Fused dequant + rotation (G32 format) ──
    // G32 = quantization group of 32 elements, matching the rotation block.
    // This is the correct layout: each block gets its own scale/zero,
    // so rotation equalization directly benefits quantization quality.
    eprintln!("\n=== Phase 3: Fused G32 dequant + inverse rotation ===");

    let g32_hsaco = compile_kernel_file(&arch, "magnum_dequant_g32")?;
    let g32_module = hip.module_load(&g32_hsaco)?;
    let fn_dequant = hip.module_get_function(&g32_module, "magnum_dequant_g32")?;
    let fn_dequant_norot = hip.module_get_function(&g32_module, "magnum_dequant_g32_norot")?;
    eprintln!("  Compiled G32 dequant kernels");

    let total_elems: usize = 4096 * 32; // 4096 blocks of 32 elements
    let num_blocks_q = total_elems / 32;

    let orig_data: Vec<f32> = (0..total_elems)
        .map(|i| ((i as f32) * 0.4517 + 0.9).sin() * 3.0)
        .collect();

    let buf_orig = hip.malloc(total_elems * 4)?;
    let buf_rotated_out = hip.malloc(total_elems * 4)?;
    hip.memcpy_htod(&buf_orig, unsafe {
        std::slice::from_raw_parts(orig_data.as_ptr() as *const u8, total_elems * 4)
    })?;

    // All mode 2 for Phase 3 test
    let all_mode2: Vec<u8> = vec![2u8; num_blocks_q];
    let buf_modes3 = hip.malloc(num_blocks_q)?;
    hip.memcpy_htod(&buf_modes3, &all_mode2)?;

    // Forward rotate on GPU
    {
        let mut in_ptr = buf_orig.as_ptr();
        let mut out_ptr = buf_rotated_out.as_ptr();
        let mut p_ptr = buf_params.as_ptr();
        let mut m_ptr = buf_modes3.as_ptr();
        let mut n = num_blocks_q as u32;
        let mut args: Vec<*mut c_void> = vec![
            &mut in_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut m_ptr as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
        ];
        let grid = (n + 7) / 8;
        unsafe { hip.launch_kernel(&fn_adaptive, [grid, 1, 1], [256, 1, 1], 0, None, &mut args)?; }
    }
    hip.device_synchronize()?;

    let mut rotated_data = vec![0.0f32; total_elems];
    hip.memcpy_dtoh(unsafe {
        std::slice::from_raw_parts_mut(rotated_data.as_mut_ptr() as *mut u8, total_elems * 4)
    }, &buf_rotated_out)?;

    // Quantize with G32 (each 32-element block gets its own scale/zero)
    let compressed = quantize_g32(&rotated_data);
    eprintln!("  G32: {} floats → {} bytes ({:.2} bits/element)",
        total_elems, compressed.len(), compressed.len() as f64 * 8.0 / total_elems as f64);

    let buf_compressed = hip.malloc(compressed.len())?;
    hip.memcpy_htod(&buf_compressed, &compressed)?;
    let buf_decompressed = hip.malloc(total_elems * 4)?;

    // Run fused G32 dequant + inverse rotation
    {
        let mut d_ptr = buf_compressed.as_ptr();
        let mut out_ptr = buf_decompressed.as_ptr();
        let mut p_ptr = buf_params.as_ptr();
        let mut m_ptr = buf_modes3.as_ptr();
        let mut nb = num_blocks_q as u32;
        let mut args: Vec<*mut c_void> = vec![
            &mut d_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut p_ptr as *mut _ as *mut c_void,
            &mut m_ptr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let grid = (nb + 7) / 8;
        unsafe { hip.launch_kernel(&fn_dequant, [grid, 1, 1], [256, 1, 1], 0, None, &mut args)?; }
    }
    hip.device_synchronize()?;

    let mut decompressed = vec![0.0f32; total_elems];
    hip.memcpy_dtoh(unsafe {
        std::slice::from_raw_parts_mut(decompressed.as_mut_ptr() as *mut u8, total_elems * 4)
    }, &buf_decompressed)?;

    let (fused_max, fused_mse, fused_cos) = compare_vectors(&orig_data, &decompressed);
    eprintln!("  Fused dequant+rotation vs original:");
    eprintln!("    max_err={:.6}, MSE={:.6e}, cosine={:.8}", fused_max, fused_mse, fused_cos);

    // Dequant-only sanity check: dequant without rotation should match rotated data
    let buf_norot_out = hip.malloc(total_elems * 4)?;
    {
        let mut d_ptr = buf_compressed.as_ptr();
        let mut out_ptr = buf_norot_out.as_ptr();
        let mut nb = num_blocks_q as u32;
        let mut args: Vec<*mut c_void> = vec![
            &mut d_ptr as *mut _ as *mut c_void,
            &mut out_ptr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let grid = (nb + 7) / 8;
        unsafe { hip.launch_kernel(&fn_dequant_norot, [grid, 1, 1], [256, 1, 1], 0, None, &mut args)?; }
    }
    hip.device_synchronize()?;

    let mut norot_out = vec![0.0f32; total_elems];
    hip.memcpy_dtoh(unsafe {
        std::slice::from_raw_parts_mut(norot_out.as_mut_ptr() as *mut u8, total_elems * 4)
    }, &buf_norot_out)?;

    let (norot_max, norot_mse, norot_cos) = compare_vectors(&rotated_data, &norot_out);
    eprintln!("  Dequant-only vs rotated (G32 quant error only):");
    eprintln!("    max_err={:.6}, MSE={:.6e}, cosine={:.8}", norot_max, norot_mse, norot_cos);

    // ── Phase 4: Quality measurement (G32 — rotation block = quant group) ──
    eprintln!("\n=== Phase 4: Quantization quality (G32, matched layout) ===");

    for mode_val in 0..3u8 {
        let mode_name = match mode_val { 0 => "Mode 0 (2 rounds)", 1 => "Mode 1 (3 rounds)", _ => "Mode 2 (5 rounds)" };

        // CPU: rotate → G32 quantize → G32 dequant → inverse rotate
        let mut recon = vec![0.0f32; total_elems];
        for b in 0..num_blocks_q {
            let s = b * 32;
            let block_orig = &orig_data[s..s + 32];
            let rotated = cpu_butterfly_forward(block_orig, &params, mode_val);
            let cmp = quantize_g32(&rotated);
            let deq = dequantize_g32(&cmp, 32);
            let inv = cpu_butterfly_inverse(&deq, &params, mode_val);
            recon[s..s + 32].copy_from_slice(&inv);
        }
        let (max_e, mse_e, cos_e) = compare_vectors(&orig_data, &recon);
        eprintln!("  {mode_name}: max_err={:.6}, MSE={:.6e}, cosine={:.8}", max_e, mse_e, cos_e);
    }

    // Baseline: G32 without rotation
    {
        let cmp = quantize_g32(&orig_data);
        let deq = dequantize_g32(&cmp, total_elems);
        let (b_max, b_mse, b_cos) = compare_vectors(&orig_data, &deq);
        eprintln!("  Baseline G32 (no rotation): max_err={:.6}, MSE={:.6e}, cosine={:.8}", b_max, b_mse, b_cos);
    }
    // Reference: G256 without rotation (original HFQ4-G256 format)
    {
        // Pad to multiple of 256 for G256
        let padded_len = (total_elems + 255) / 256 * 256;
        let mut padded = orig_data.clone();
        padded.resize(padded_len, 0.0);
        let cmp = quantize_hfq4_g256(&padded);
        let deq = dequantize_hfq4_g256(&cmp, padded_len);
        let (r_max, r_mse, r_cos) = compare_vectors(&orig_data, &deq[..total_elems]);
        eprintln!("  Reference G256 (no rotation): max_err={:.6}, MSE={:.6e}, cosine={:.8}", r_max, r_mse, r_cos);
    }

    // ── Phase 5: Encoder (G32 — no cross-block interference) ──
    eprintln!("\n=== Phase 5: Adaptive mode selection encoder (G32) ===");

    // With G32, each block is its own quantization group (own scale/zero).
    // Mode selection for one block cannot affect another. Simple per-block sweep.
    let cos_threshold = 0.995;
    let mut mode_counts = [0u32; 3];
    let mut selected_modes = vec![0u8; num_blocks_q];
    let mut gate_failures = 0u32;

    for b in 0..num_blocks_q {
        let s = b * 32;
        let block_orig = &orig_data[s..s + 32];

        let mut best_mode = 2u8; // fallback
        for mode_val in 0..3u8 {
            let rotated = cpu_butterfly_forward(block_orig, &params, mode_val);
            let cmp = quantize_g32(&rotated);
            let deq = dequantize_g32(&cmp, 32);
            let recon = cpu_butterfly_inverse(&deq, &params, mode_val);

            let (_, _, cos) = compare_vectors(block_orig, &recon);
            if cos >= cos_threshold as f64 {
                best_mode = mode_val;
                break;
            }
        }
        selected_modes[b] = best_mode;
        mode_counts[best_mode as usize] += 1;

        // Check if the chosen mode actually passes (best_mode=2 might still miss)
        if best_mode == 2 {
            let rotated = cpu_butterfly_forward(block_orig, &params, 2);
            let cmp = quantize_g32(&rotated);
            let deq = dequantize_g32(&cmp, 32);
            let recon = cpu_butterfly_inverse(&deq, &params, 2);
            let (_, _, cos) = compare_vectors(block_orig, &recon);
            if cos < cos_threshold as f64 {
                gate_failures += 1;
            }
        }
    }

    // End-to-end quality with selected modes
    let mut final_recon = vec![0.0f32; total_elems];
    for b in 0..num_blocks_q {
        let s = b * 32;
        let rotated = cpu_butterfly_forward(&orig_data[s..s + 32], &params, selected_modes[b]);
        let cmp = quantize_g32(&rotated);
        let deq = dequantize_g32(&cmp, 32);
        let recon = cpu_butterfly_inverse(&deq, &params, selected_modes[b]);
        final_recon[s..s + 32].copy_from_slice(&recon);
    }
    let (enc_max, enc_mse, enc_cos) = compare_vectors(&orig_data, &final_recon);

    let total_b = num_blocks_q as f32;
    eprintln!("  Threshold: cosine >= {cos_threshold}");
    eprintln!("  Mode 0 (2 rounds): {:5} blocks ({:.1}%)", mode_counts[0], mode_counts[0] as f32 / total_b * 100.0);
    eprintln!("  Mode 1 (3 rounds): {:5} blocks ({:.1}%)", mode_counts[1], mode_counts[1] as f32 / total_b * 100.0);
    eprintln!("  Mode 2 (5 rounds): {:5} blocks ({:.1}%)", mode_counts[2], mode_counts[2] as f32 / total_b * 100.0);
    eprintln!("  Gate failures (miss threshold at max mode): {} blocks ({:.1}%)",
        gate_failures, gate_failures as f32 / total_b * 100.0);
    eprintln!("  End-to-end with selected modes: max_err={:.6}, MSE={:.6e}, cosine={:.8}", enc_max, enc_mse, enc_cos);

    // ── Phase 6: Bandwidth benchmark (G32 format) ──
    eprintln!("\n=== Phase 6: Bandwidth benchmark on 5700 XT (G32) ===");
    let peak_bw = 448.0; // GB/s theoretical

    let ev_start = hip.event_create()?;
    let ev_stop = hip.event_create()?;

    for &seq_len_label in &[2048u32, 4096, 8192] {
        // KV per token for ~3B model: 2 * 32 heads * 128 dim = 8192 elements
        let elems_per_token: u32 = 8192;
        let total_elems_bm = seq_len_label * elems_per_token;
        let num_blocks_bm = total_elems_bm / 32;
        let compressed_bytes = num_blocks_bm as usize * 24; // G32: 24 bytes per block
        let output_bytes = total_elems_bm as usize * 4;

        let buf_c = hip.malloc(compressed_bytes)?;
        let buf_o = hip.malloc(output_bytes)?;
        let buf_m = hip.malloc(num_blocks_bm as usize)?;
        hip.memset(&buf_c, 0x42, compressed_bytes)?;
        hip.memset(&buf_m, 2, num_blocks_bm as usize)?;

        let iters = 500u32;

        // Fused (dequant + rotation)
        hip.device_synchronize()?;
        hip.event_record(&ev_start, None)?;
        for _ in 0..iters {
            let mut d_ptr = buf_c.as_ptr();
            let mut out_ptr = buf_o.as_ptr();
            let mut p_ptr = buf_params.as_ptr();
            let mut m_ptr = buf_m.as_ptr();
            let mut nb = num_blocks_bm;
            let mut args: Vec<*mut c_void> = vec![
                &mut d_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut p_ptr as *mut _ as *mut c_void,
                &mut m_ptr as *mut _ as *mut c_void,
                &mut nb as *mut _ as *mut c_void,
            ];
            let grid = (num_blocks_bm + 7) / 8;
            unsafe { hip.launch_kernel(&fn_dequant, [grid, 1, 1], [256, 1, 1], 0, None, &mut args)?; }
        }
        hip.event_record(&ev_stop, None)?;
        hip.event_synchronize(&ev_stop)?;
        let ms_fused = hip.event_elapsed_ms(&ev_start, &ev_stop)?;
        let us_fused = ms_fused as f64 * 1000.0 / iters as f64;
        let bytes_total = (compressed_bytes + output_bytes) as f64;
        let bw_fused = bytes_total / (us_fused * 1e-6) / 1e9;

        // Dequant-only
        hip.device_synchronize()?;
        hip.event_record(&ev_start, None)?;
        for _ in 0..iters {
            let mut d_ptr = buf_c.as_ptr();
            let mut out_ptr = buf_o.as_ptr();
            let mut nb = num_blocks_bm;
            let mut args: Vec<*mut c_void> = vec![
                &mut d_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &mut nb as *mut _ as *mut c_void,
            ];
            let grid = (num_blocks_bm + 7) / 8;
            unsafe { hip.launch_kernel(&fn_dequant_norot, [grid, 1, 1], [256, 1, 1], 0, None, &mut args)?; }
        }
        hip.event_record(&ev_stop, None)?;
        hip.event_synchronize(&ev_stop)?;
        let ms_norot = hip.event_elapsed_ms(&ev_start, &ev_stop)?;
        let us_norot = ms_norot as f64 * 1000.0 / iters as f64;
        let bw_norot = bytes_total / (us_norot * 1e-6) / 1e9;

        let overhead_pct = ((us_fused / us_norot) - 1.0) * 100.0;

        eprintln!("  seq_len={seq_len_label}: fused {:.1} us ({:.1} GB/s, {:.1}% peak) | norot {:.1} us ({:.1} GB/s) | overhead: {:.1}%",
            us_fused, bw_fused, bw_fused / peak_bw * 100.0,
            us_norot, bw_norot, overhead_pct);

        hip.free(buf_c)?;
        hip.free(buf_o)?;
        hip.free(buf_m)?;
    }

    // Cleanup
    hip.free(buf_in)?;
    hip.free(buf_fwd)?;
    hip.free(buf_inv)?;
    hip.free(buf_params)?;
    hip.free(buf_modes)?;
    hip.free(buf_adapt_fwd)?;
    hip.free(buf_adapt_inv)?;
    hip.free(buf_orig)?;
    hip.free(buf_rotated_out)?;
    hip.free(buf_modes3)?;
    hip.free(buf_compressed)?;
    hip.free(buf_decompressed)?;
    hip.free(buf_norot_out)?;

    eprintln!("\n=== All phases complete ===");
    Ok(())
}

fn compile_kernel_file(arch: &str, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let src = format!("{}/kernels/{}.hip", env!("CARGO_MANIFEST_DIR"), name);
    let out_dir = format!("{}/results", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(&out_dir)?;
    let out = format!("{out_dir}/{name}.hsaco");

    let status = std::process::Command::new("hipcc")
        .args(["--genco", &format!("--offload-arch={arch}"), "-O3", "-o", &out, &src])
        .status()?;
    if !status.success() {
        return Err(format!("hipcc failed for {name}").into());
    }
    Ok(out)
}

// ── CPU-side HFQ4-G256 quantization ──

fn quantize_hfq4_g256(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 256 == 0);
    let num_groups = data.len() / 256;
    let mut output = Vec::with_capacity(num_groups * 136);

    for g in 0..num_groups {
        let group = &data[g * 256..(g + 1) * 256];
        let min_val = group.iter().copied().fold(f32::MAX, f32::min);
        let max_val = group.iter().copied().fold(f32::MIN, f32::max);

        let scale = if max_val > min_val { (max_val - min_val) / 15.0 } else { 1.0 };
        let zero = min_val;

        output.extend_from_slice(&scale.to_le_bytes());
        output.extend_from_slice(&zero.to_le_bytes());

        for i in (0..256).step_by(2) {
            let q0 = ((group[i] - zero) / scale).round().clamp(0.0, 15.0) as u8;
            let q1 = ((group[i + 1] - zero) / scale).round().clamp(0.0, 15.0) as u8;
            output.push(q0 | (q1 << 4));
        }
    }
    output
}

fn dequantize_hfq4_g256(data: &[u8], total_elems: usize) -> Vec<f32> {
    let num_groups = total_elems / 256;
    let mut output = Vec::with_capacity(total_elems);

    for g in 0..num_groups {
        let gptr = &data[g * 136..];
        let scale = f32::from_le_bytes([gptr[0], gptr[1], gptr[2], gptr[3]]);
        let zero = f32::from_le_bytes([gptr[4], gptr[5], gptr[6], gptr[7]]);

        for i in 0..128 {
            let byte = gptr[8 + i];
            let low = (byte & 0xF) as f32;
            let high = (byte >> 4) as f32;
            output.push(scale * low + zero);
            output.push(scale * high + zero);
        }
    }
    output
}

// ── CPU-side G32 quantization (group size = 32, matching rotation block) ──

fn quantize_g32(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0);
    let num_blocks = data.len() / 32;
    let mut output = Vec::with_capacity(num_blocks * 24);

    for b in 0..num_blocks {
        let block = &data[b * 32..(b + 1) * 32];
        let min_val = block.iter().copied().fold(f32::MAX, f32::min);
        let max_val = block.iter().copied().fold(f32::MIN, f32::max);

        let scale = if max_val > min_val { (max_val - min_val) / 15.0 } else { 1.0 };
        let zero = min_val;

        output.extend_from_slice(&scale.to_le_bytes());
        output.extend_from_slice(&zero.to_le_bytes());

        for i in (0..32).step_by(2) {
            let q0 = ((block[i] - zero) / scale).round().clamp(0.0, 15.0) as u8;
            let q1 = ((block[i + 1] - zero) / scale).round().clamp(0.0, 15.0) as u8;
            output.push(q0 | (q1 << 4));
        }
    }
    output
}

fn dequantize_g32(data: &[u8], total_elems: usize) -> Vec<f32> {
    let num_blocks = total_elems / 32;
    let mut output = Vec::with_capacity(total_elems);

    for b in 0..num_blocks {
        let bptr = &data[b * 24..];
        let scale = f32::from_le_bytes([bptr[0], bptr[1], bptr[2], bptr[3]]);
        let zero = f32::from_le_bytes([bptr[4], bptr[5], bptr[6], bptr[7]]);

        for i in 0..16 {
            let byte = bptr[8 + i];
            output.push(scale * (byte & 0xF) as f32 + zero);
            output.push(scale * (byte >> 4) as f32 + zero);
        }
    }
    output
}

// ── CPU-side butterfly rotation (for encoder) ──

fn cpu_butterfly_forward(data: &[f32], params: &[f32], mode: u8) -> Vec<f32> {
    let mut v = data.to_vec();
    let rounds: &[(usize, usize)] = match mode {
        0 => &[(1, 0), (2, 1)],
        1 => &[(1, 0), (2, 1), (4, 2)],
        _ => &[(1, 0), (2, 1), (4, 2), (8, 3), (16, 4)],
    };
    for &(stride, pidx) in rounds {
        let c = params[pidx * 2];
        let s = params[pidx * 2 + 1];
        let mut next = v.clone();
        for i in 0..v.len() {
            let partner = i ^ stride;
            if partner < v.len() {
                let sign_s = if (i & stride) != 0 { s } else { -s };
                next[i] = c * v[i] + sign_s * v[partner];
            }
        }
        v = next;
    }
    v
}

fn cpu_butterfly_inverse(data: &[f32], params: &[f32], mode: u8) -> Vec<f32> {
    let mut v = data.to_vec();
    let rounds: Vec<(usize, usize)> = match mode {
        0 => vec![(2, 1), (1, 0)],
        1 => vec![(4, 2), (2, 1), (1, 0)],
        _ => vec![(16, 4), (8, 3), (4, 2), (2, 1), (1, 0)],
    };
    for &(stride, pidx) in &rounds {
        let c = params[pidx * 2];
        let s = -params[pidx * 2 + 1]; // negate sin for inverse
        let mut next = v.clone();
        for i in 0..v.len() {
            let partner = i ^ stride;
            if partner < v.len() {
                let sign_s = if (i & stride) != 0 { s } else { -s };
                next[i] = c * v[i] + sign_s * v[partner];
            }
        }
        v = next;
    }
    v
}

fn compare_vectors(a: &[f32], b: &[f32]) -> (f32, f32, f64) {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let mut max_err: f32 = 0.0;
    let mut sum_sq: f64 = 0.0;
    let mut dot: f64 = 0.0;
    let mut norm_a: f64 = 0.0;
    let mut norm_b: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        max_err = max_err.max(diff);
        sum_sq += (diff as f64) * (diff as f64);
        dot += (*x as f64) * (*y as f64);
        norm_a += (*x as f64) * (*x as f64);
        norm_b += (*y as f64) * (*y as f64);
    }
    let mse = sum_sq / n;
    let cos = if norm_a < 1e-30 && norm_b < 1e-30 {
        1.0 // both vectors effectively zero — identical
    } else if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        0.0 // one zero, one not — genuinely dissimilar
    };
    (max_err, mse as f32, cos)
}
