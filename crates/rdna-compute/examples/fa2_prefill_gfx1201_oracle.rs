// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// THROWAWAY proof harness for the gfx1201 GQA-fused FA2 prefill kernel
// (Slice D). NOT a permanent test: remove after the model gates and the
// final perf checkpoint are recorded.
//
// MODE=oracle (default): deterministic f32 Q + byte-exact Q8 K/V, CPU-f64
// causal reference (decodes each f16 scale/int8 payload, rounds Q through
// f16 like the kernel), frozen shapes (N,CTX) = (1,1),(7,63),(8,64),
// (9,65),(17,257),(16,4096) each with tail and ragged positions. Requires
// for incumbent-vs-CPU and candidate-vs-CPU (direct S1 and split S2):
// finite outputs, global rel_l2 <= 1e-3, per-(query,head) cosine >=
// 1-1e-6, plus a causal-sentinel check (perturbing future K/V bytes cannot
// change an earlier row).
//
// MODE=bench: GPU-event-only timing (allocation/upload/download outside the
// interval) at N/CTX (defaults 384/7936), WARMUPS (10) untimed + RUNS (21)
// timed runs each for incumbent, QT64 direct, QT64 S2+merge. Prints
// machine-readable RESULT lines with 21-run median/p10/p90, effective
// TFLOPS against the exact causal FLOP count, and the candidate/incumbent
// ratio. Exits nonzero unless the fastest correct candidate is >= 3.00x
// the incumbent (the kill gate).
//
// Env: MODE, N, CTX, WARMUPS, RUNS. Routed to GPU 1 via ROCR_VISIBLE_DEVICES
// by the caller; GPU 0 in-process is that visible device.

use rdna_compute::{DType, Gpu};

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const BPH: usize = HD / 32;
const ROW_STRIDE: usize = NKV * BPH * 34;
const SCALE_ATTN_F32: f32 = 1.0f32 / 16.0f32; // 1/sqrt(256)

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn env_str(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.into())
}

/// f32 -> IEEE binary16 bits, round-to-nearest-even.
fn f32_to_half_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x007F_FFFF;
    if exp == 0 {
        // Subnormal/zero: flush (magnitudes here never need subnormals).
        let v = (mant >> 13) as u16;
        return sign | v;
    }
    if exp == 0xFF {
        return sign | 0x7C00 | if mant == 0 { 0 } else { 0x0200 };
    }
    let e = exp - 127 + 15;
    if e <= 0 {
        return sign;
    }
    if e >= 31 {
        return sign | 0x7C00;
    }
    let keep = mant & !0x1FFF;
    let rem = mant & 0x1FFF;
    let mut m = keep;
    if rem > 0x1000 || (rem == 0x1000 && (keep & 0x2000) != 0) {
        m = keep + 0x2000;
        if m > 0x007F_FFFF {
            return sign | (((e + 1) as u16) << 10);
        }
    }
    sign | ((e as u16) << 10) | ((m >> 13) as u16)
}

/// IEEE binary16 bits -> f32.
fn half_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 14;
            let mut m = mant;
            while m & 0x0400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x03FF) << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (mant << 13)
    } else {
        sign | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Deterministic Q8 cache bytes. Same family as test_q8_flash_prefill:
/// varied scales/codes so a wrong dequant, stride, or GQA head fails.
fn build_cache(ctx: usize, v_bump: u8) -> Vec<u8> {
    let mut kv = vec![0u8; ctx * ROW_STRIDE];
    for (bi, blk) in kv.chunks_mut(34).enumerate() {
        let scale: f32 = 0.02 + ((bi % 13) as f32) * 0.005;
        let h = f32_to_half_bits(scale);
        blk[0] = (h & 0xFF) as u8;
        blk[1] = (h >> 8) as u8;
        for (j, b) in blk[2..].iter_mut().enumerate() {
            *b = (((bi * 31 + j * 17) % 251) as i32 - 125 + v_bump as i32) as i8 as u8;
        }
    }
    kv
}

fn build_q(n: usize) -> Vec<f32> {
    (0..n * NH * HD)
        .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.01)
        .collect()
}

fn build_positions(n: usize, ctx: usize, ragged: bool) -> Vec<i32> {
    if ragged {
        // Deterministic non-monotonic causal windows in [0, ctx).
        (0..n).map(|b| ((b * 7919) % ctx.max(1)) as i32).collect()
    } else {
        (0..n).map(|b| (ctx - n + b) as i32).collect()
    }
}

/// Decode one Q8 value exactly as the kernel does: f16(scale_f32 * code).
fn decode_q8(cache: &[u8], pos: usize, kv_h: usize, d: usize) -> f64 {
    let blk = d / 32;
    let off = pos * ROW_STRIDE + (kv_h * BPH + blk) * 34;
    let su = (cache[off] as u16) | ((cache[off + 1] as u16) << 8);
    let ks = half_bits_to_f32(su);
    let code = cache[off + 2 + (d % 32)] as i8 as f32;
    half_bits_to_f32(f32_to_half_bits(ks * code)) as f64
}

/// CPU-f64 causal reference. Q is rounded through f16 (kernel parity);
/// K/V decode each f16 scale/int8 payload then round through f16.
fn cpu_reference(q: &[f32], k: &[u8], v: &[u8], pos: &[i32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * NH * HD];
    let scale = SCALE_ATTN_F32 as f64;
    for b in 0..n {
        let pb = pos[b] as usize;
        for h in 0..NH {
            let kv_h = h / (NH / NKV);
            // Q row rounded through f16.
            let mut qq = [0.0f64; HD];
            for d in 0..HD {
                qq[d] = half_bits_to_f32(f32_to_half_bits(
                    q[(b * NH + h) * HD + d],
                )) as f64;
            }
            // Scores over valid keys.
            let mut scores = vec![f64::NEG_INFINITY; pb + 1];
            for kk in 0..=pb {
                let mut dot = 0.0f64;
                for d in 0..HD {
                    dot += qq[d] * decode_q8(k, kk, kv_h, d);
                }
                scores[kk] = dot * scale;
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut den = 0.0f64;
            let mut probs = vec![0.0f64; pb + 1];
            if m.is_finite() {
                for kk in 0..=pb {
                    let e = (scores[kk] - m).exp();
                    probs[kk] = e;
                    den += e;
                }
            }
            for d in 0..HD {
                let mut acc = 0.0f64;
                for kk in 0..=pb {
                    acc += probs[kk] * decode_q8(v, kk, kv_h, d);
                }
                out[(b * NH + h) * HD + d] = if den > 0.0 { (acc / den) as f32 } else { 0.0 };
            }
        }
    }
    out
}

struct Metrics {
    rel_l2: f32,
    min_cos: f32,
    finite: bool,
    degenerate: usize,
    compared: usize,
}

fn metrics(reference: &[f32], cand: &[f32], n: usize) -> Metrics {
    let finite = cand.iter().all(|v| v.is_finite());
    let (mut sa, mut sd) = (0.0f64, 0.0f64);
    for (x, y) in reference.iter().zip(cand.iter()) {
        sa += (*x as f64) * (*x as f64);
        sd += ((*x - *y) as f64) * ((*x - *y) as f64);
    }
    let rel_l2 = if sa > 0.0 { (sd.sqrt() / sa.sqrt()) as f32 } else { 0.0 };
    let mut min_cos = 1.0f32;
    let (mut compared, mut degenerate) = (0usize, 0usize);
    for vec_i in 0..(n * NH) {
        let s = vec_i * HD;
        let (mut dot, mut na, mut nb, mut nd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for d in 0..HD {
            dot += (reference[s + d] as f64) * (cand[s + d] as f64);
            na += (reference[s + d] as f64).powi(2);
            nb += (cand[s + d] as f64).powi(2);
            nd += ((reference[s + d] - cand[s + d]) as f64).powi(2);
        }
        let _ = nd;
        if na > 0.0 && nb > 0.0 {
            min_cos = min_cos.min((dot / (na.sqrt() * nb.sqrt())) as f32);
            compared += 1;
        } else if na > 0.0 {
            degenerate += 1;
        }
    }
    Metrics { rel_l2, min_cos, finite, degenerate, compared }
}

fn upload_i32_as_raw(gpu: &Gpu, data: &[i32]) -> rdna_compute::HipResult<rdna_compute::GpuTensor> {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    gpu.upload_raw(bytes, &[data.len()])
}

fn main() {
    let mode = env_str("MODE", "oracle");
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("SKIP: fa2 oracle/bench requires gfx1201, got {}", gpu.arch);
        std::process::exit(2);
    }
    match mode.as_str() {
        "oracle" => oracle(&mut gpu),
        "bench" => bench(&mut gpu),
        _ => {
            eprintln!("unknown MODE={mode} (want oracle|bench)");
            std::process::exit(2);
        }
    }
}

fn check_case(
    gpu: &mut Gpu,
    n: usize,
    ctx: usize,
    ragged: bool,
) -> bool {
    let tag = if ragged { "ragged" } else { "tail" };
    let k = build_cache(ctx, 0);
    let v = build_cache(ctx, 7);
    let qd = build_q(n);
    let pos = build_positions(n, ctx, ragged);
    let k_g = gpu.upload_raw(&k, &[k.len()]).expect("k upload");
    let v_g = gpu.upload_raw(&v, &[v.len()]).expect("v upload");
    let q_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("q upload");
    let p_g = upload_i32_as_raw(gpu, &pos).expect("pos upload");
    let out_inc = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let out_s1 = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let out_s2 = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let partials = gpu
        .zeros(&[2 * n * NH * (HD + 2)], DType::F32)
        .expect("partials");

    gpu.attention_q8_0_flash_prefill_wmma_incumbent_gfx1201_bench(
        &q_g, &k_g, &v_g, &out_inc, &p_g, NH, NKV, HD, ctx, n,
    )
    .expect("incumbent launch");
    gpu.attention_q8_0_fa2_gqa_gfx1201(
        &q_g, &k_g, &v_g, &out_s1, &p_g, NH, NKV, HD, ctx, n,
    )
    .expect("candidate S1 launch");
    gpu.attention_q8_0_fa2_gqa_split_gfx1201_bench(
        &q_g, &k_g, &v_g, &out_s2, &p_g, &partials, NH, NKV, HD, n, 2,
    )
    .expect("candidate S2 launch");

    let ri = gpu.download_f32(&out_inc).expect("dl inc");
    let r1 = gpu.download_f32(&out_s1).expect("dl s1");
    let r2 = gpu.download_f32(&out_s2).expect("dl s2");
    let reference = cpu_reference(&qd, &k, &v, &pos, n);

    let mi = metrics(&reference, &ri, n);
    let m1 = metrics(&reference, &r1, n);
    let m2 = metrics(&reference, &r2, n);
    let ok_arm = |m: &Metrics| {
        m.finite && m.degenerate == 0 && m.compared == n * NH && m.rel_l2 <= 1e-3 && m.min_cos >= 1.0 - 1e-6
    };
    let pass = ok_arm(&mi) && ok_arm(&m1) && ok_arm(&m2);
    println!(
        "RESULT oracle n={n} ctx={ctx} pos={tag} inc_rel_l2={:.3e} inc_cos={:.9} \
         s1_rel_l2={:.3e} s1_cos={:.9} s2_rel_l2={:.3e} s2_cos={:.9} pass={pass}",
        mi.rel_l2, mi.min_cos, m1.rel_l2, m1.min_cos, m2.rel_l2, m2.min_cos
    );
    if !pass {
        eprintln!(
            "FAIL oracle n={n} ctx={ctx} pos={tag}: inc finite={} deg={} cmp={} | \
             s1 finite={} deg={} cmp={} | s2 finite={} deg={} cmp={}",
            mi.finite, mi.degenerate, mi.compared,
            m1.finite, m1.degenerate, m1.compared,
            m2.finite, m2.degenerate, m2.compared
        );
    }

    // Causal sentinel: perturb keys strictly after row 0's position; row 0
    // must come back bit-identical.
    let mut k2 = k.clone();
    let mut v2 = v.clone();
    let p0 = pos[0] as usize;
    if p0 + 1 < ctx {
        for kk in (p0 + 1)..ctx {
            for blk in 0..(NKV * BPH) {
                let off = kk * ROW_STRIDE + blk * 34;
                for j in 2..34 {
                    k2[off + j] = k2[off + j].wrapping_add(13);
                    v2[off + j] = v2[off + j].wrapping_add(29);
                }
            }
        }
        let k2_g = gpu.upload_raw(&k2, &[k2.len()]).expect("k2 upload");
        let v2_g = gpu.upload_raw(&v2, &[v2.len()]).expect("v2 upload");
        let out_s = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
        gpu.attention_q8_0_fa2_gqa_gfx1201(
            &q_g, &k2_g, &v2_g, &out_s, &p_g, NH, NKV, HD, ctx, n,
        )
        .expect("sentinel launch");
        let rs = gpu.download_f32(&out_s).expect("dl sentinel");
        let row0_same = r1[..NH * HD] == rs[..NH * HD];
        println!("RESULT sentinel n={n} ctx={ctx} pos={tag} row0_bitidentical={row0_same}");
        if !row0_same {
            eprintln!("FAIL sentinel n={n} ctx={ctx} pos={tag}: row 0 changed");
            return false;
        }
    } else {
        println!("RESULT sentinel n={n} ctx={ctx} pos={tag} row0_bitidentical=skip");
    }
    pass
}

fn oracle(gpu: &mut Gpu) {
    let shapes: &[(usize, usize)] = &[(1, 1), (7, 63), (8, 64), (9, 65), (17, 257), (16, 4096)];
    let mut all_pass = true;
    for &(n, ctx) in shapes {
        for &ragged in &[false, true] {
            if !check_case(gpu, n, ctx, ragged) {
                all_pass = false;
            }
        }
    }
    println!("RESULT oracle_verdict pass={all_pass}");
    if !all_pass {
        std::process::exit(1);
    }
    println!("PASS");
}

/// Median/p10/p90 of GPU-event times (ms) around `f`, allocations outside.
fn event_times(
    gpu: &mut Gpu,
    warmups: usize,
    runs: usize,
    f: &dyn Fn(&mut Gpu),
) -> Vec<f64> {
    for _ in 0..warmups {
        f(gpu);
    }
    gpu.hip.device_synchronize().unwrap();
    let mut ts = Vec::with_capacity(runs);
    for _ in 0..runs {
        let s = gpu.hip.event_create().unwrap();
        let e = gpu.hip.event_create().unwrap();
        gpu.hip.event_record(&s, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&e, None).unwrap();
        gpu.hip.event_synchronize(&e).unwrap();
        ts.push(gpu.hip.event_elapsed_ms(&s, &e).unwrap() as f64);
        gpu.hip.event_destroy(s).unwrap();
        gpu.hip.event_destroy(e).unwrap();
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts
}

fn bench(gpu: &mut Gpu) {
    let n = env_usize("N", 384);
    let ctx = env_usize("CTX", 7936);
    let warmups = env_usize("WARMUPS", 10);
    let runs = env_usize("RUNS", 21);
    assert!(n >= 1 && n <= 384, "bench N must be 1..=384");

    let k = build_cache(ctx, 0);
    let v = build_cache(ctx, 7);
    let qd = build_q(n);
    let pos = build_positions(n, ctx, false);
    let k_g = gpu.upload_raw(&k, &[k.len()]).expect("k upload");
    let v_g = gpu.upload_raw(&v, &[v.len()]).expect("v upload");
    let q_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("q upload");
    let p_g = upload_i32_as_raw(gpu, &pos).expect("pos upload");
    let out_inc = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let out_s1 = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let out_s2 = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let partials = gpu
        .zeros(&[2 * n * NH * (HD + 2)], DType::F32)
        .expect("partials");

    let t_inc = event_times(gpu, warmups, runs, &|g: &mut Gpu| {
        g.attention_q8_0_flash_prefill_wmma_incumbent_gfx1201_bench(
            &q_g, &k_g, &v_g, &out_inc, &p_g, NH, NKV, HD, ctx, n,
        )
        .expect("incumbent");
    });
    let t_s1 = event_times(gpu, warmups, runs, &|g: &mut Gpu| {
        g.attention_q8_0_fa2_gqa_gfx1201(
            &q_g, &k_g, &v_g, &out_s1, &p_g, NH, NKV, HD, ctx, n,
        )
        .expect("s1");
    });
    let t_s2 = event_times(gpu, warmups, runs, &|g: &mut Gpu| {
        g.attention_q8_0_fa2_gqa_split_gfx1201_bench(
            &q_g, &k_g, &v_g, &out_s2, &p_g, &partials, NH, NKV, HD, n, 2,
        )
        .expect("s2");
    });

    // Exact causal FLOP count: per row 4*H*D keys (QK 2 + PV 2).
    let mut flop = 0u64;
    for b in 0..n {
        flop += 4 * NH as u64 * HD as u64 * (pos[b] as u64 + 1);
    }
    let report = |name: &str, ts: &[f64]| {
        let med = ts[ts.len() / 2];
        let p10 = ts[ts.len() / 10];
        let p90 = ts[(ts.len() * 9) / 10];
        let tflops = flop as f64 / 1e12 / (med / 1e3);
        println!(
            "RESULT bench arm={name} n={n} ctx={ctx} runs={runs} \
             median_ms={med:.4} p10_ms={p10:.4} p90_ms={p90:.4} \
             gflop={:.3} tflops={tflops:.2}",
            flop as f64 / 1e9
        );
        (med, tflops)
    };
    let (m_inc, _) = report("incumbent", &t_inc);
    let (m_s1, t_s1) = report("direct", &t_s1);
    let (m_s2, t_s2) = report("split_s2", &t_s2);

    // Sanity (not a gate): candidate vs incumbent agreement at bench shape.
    let r1 = gpu.download_f32(&out_s1).expect("dl s1");
    let ri = gpu.download_f32(&out_inc).expect("dl inc");
    let m = metrics(&ri, &r1, n);
    println!(
        "RESULT bench_agree s1_vs_inc_rel_l2={:.3e} min_cos={:.9} finite={}",
        m.rel_l2, m.min_cos, m.finite
    );

    let best = m_s1.min(m_s2);
    let best_name = if m_s1 <= m_s2 { "direct" } else { "split_s2" };
    let best_tf = t_s1.max(t_s2);
    let ratio = m_inc / best;
    let pass = ratio >= 3.00 && m.finite;
    println!(
        "RESULT verdict best={best_name} ratio_vs_incumbent={ratio:.3} \
         best_tflops={best_tf:.2} kill_gate_3x={pass}"
    );
    if !pass {
        std::process::exit(1);
    }
    println!("PASS");
}
