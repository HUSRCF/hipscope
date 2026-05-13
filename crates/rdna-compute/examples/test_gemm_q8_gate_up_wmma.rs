//! T3-2 correctness — `gemm_gate_up_q8_0_wmma` vs substrate × 2.
//! Gate: gated mean_rel < 1e-3 on Y_gate and Y_up.

use rdna_compute::{DType, Gpu};

fn main() {
    let mut gpu = Gpu::init().expect("gpu init");
    let arch = gpu.arch.clone();
    eprintln!("=== test_gemm_q8_gate_up_wmma ===\n  arch = {arch}");
    if !arch.starts_with("gfx11") && !arch.starts_with("gfx12") {
        eprintln!("  SKIPPED: needs gfx11/12, got {arch}"); std::process::exit(0);
    }

    // (gate_m, up_m, K, label) — gate_m == up_m for Qwen3.5.
    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (128, 128, 128, "tiny"),
        (512, 512, 512, "medium"),
        (11008, 11008, 4096, "9B FFN  (gate=up=11008 K=4096)"),
    ];
    let batches: Vec<usize> = vec![1, 4, 16, 32, 64, 128, 256];
    let mut total_fail = 0usize;

    for (gate_m, up_m, k, label) in &shapes {
        let (gate_m, up_m, k) = (*gate_m, *up_m, *k);
        eprintln!("\n--- {label} ---");

        let w_g = synth_q8(gate_m, k, 0xA1B2C3D4);
        let w_u = synth_q8(up_m, k, 0xE5F60718);
        let d_g = gpu.upload_raw(&w_g, &[w_g.len()]).unwrap();
        let d_u = gpu.upload_raw(&w_u, &[w_u.len()]).unwrap();

        let max_n = *batches.iter().max().unwrap();
        let x_host: Vec<f32> = (0..max_n * k).map(synth_x).collect();
        let d_x = gpu.upload_f32(&x_host, &[max_n * k]).unwrap();

        let d_yg_w = gpu.zeros(&[max_n * gate_m], DType::F32).unwrap();
        let d_yu_w = gpu.zeros(&[max_n * up_m], DType::F32).unwrap();
        let d_yg_r = gpu.zeros(&[max_n * gate_m], DType::F32).unwrap();
        let d_yu_r = gpu.zeros(&[max_n * up_m], DType::F32).unwrap();

        for &n in &batches {
            let x_n = d_x.sub_offset(0, n * k);
            let gw = d_yg_w.sub_offset(0, n * gate_m);
            let uw = d_yu_w.sub_offset(0, n * up_m);
            let gr = d_yg_r.sub_offset(0, n * gate_m);
            let ur = d_yu_r.sub_offset(0, n * up_m);

            gpu.gemm_gate_up_q8_0_wmma(&d_g, &d_u, &x_n, &gw, &uw, gate_m, up_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_g, &x_n, &gr, gate_m, k, n).unwrap();
            gpu.gemm_q8_0_batched_chunked(&d_u, &x_n, &ur, up_m, k, n).unwrap();

            let s = [
                compare(&gpu.download_f32(&gw).unwrap(), &gpu.download_f32(&gr).unwrap()),
                compare(&gpu.download_f32(&uw).unwrap(), &gpu.download_f32(&ur).unwrap()),
            ];
            let pass = s.iter().all(|x| x.mean_rel < 2e-3 && x.max_rel < 5e-2);
            let mark = if pass { "PASS" } else { total_fail += 1; "FAIL" };
            eprintln!(
                "  N={n:4}  {mark}   gate: mean={:.2e}/max={:.2e}  up: {:.2e}/{:.2e}",
                s[0].mean_rel, s[0].max_rel, s[1].mean_rel, s[1].max_rel,
            );
        }
    }
    eprintln!("\n=== {total_fail} failure(s) ===");
    std::process::exit(if total_fail == 0 { 0 } else { 1 });
}

// (helpers identical to test_gemm_q8_qkvza_wmma.rs)
struct Stats { mean_rel: f64, max_rel: f64 }
fn compare(a: &[f32], b: &[f32]) -> Stats {
    let max_ref = b.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let thr = max_ref * 0.01;
    let (mut sum, mut max_r, mut n) = (0.0f64, 0.0f64, 0usize);
    for (x, y) in a.iter().zip(b.iter()) {
        if y.abs() > thr {
            let r = ((x - y).abs() / y.abs()) as f64;
            sum += r; if r > max_r { max_r = r; } n += 1;
        }
    }
    Stats { mean_rel: if n == 0 { 0.0 } else { sum / n as f64 }, max_rel: max_r }
}
fn synth_x(i: usize) -> f32 {
    let v = ((i as i64).wrapping_mul(1103515245).wrapping_add(12345)) as f32;
    (v * 1e-9) % 2.0 - 1.0
}
fn synth_q8(m: usize, k: usize, seed0: u32) -> Vec<u8> {
    let bpr = k / 32;
    let mut out = vec![0u8; m * bpr * 34];
    let mut seed = seed0;
    let mut prng = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); seed };
    for r in 0..m { for b in 0..bpr {
        let off = r * bpr * 34 + b * 34;
        let sf = 0.001 + (prng() as f32 / u32::MAX as f32) * 0.049;
        let sb = f32_to_f16_bits(sf);
        out[off] = (sb & 0xFF) as u8; out[off+1] = (sb >> 8) as u8;
        for j in 0..32 { out[off+2+j] = ((prng() as i32 % 255) - 127) as i8 as u8; }
    }}
    out
}
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp_f32 = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp_f32 == 0 { return sign; }
    if exp_f32 == 0xff { return sign | 0x7c00 | if mant != 0 { 1 } else { 0 }; }
    let exp = exp_f32 - 127 + 15;
    if exp <= 0 { return sign; }
    if exp >= 31 { return sign | 0x7c00; }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}
