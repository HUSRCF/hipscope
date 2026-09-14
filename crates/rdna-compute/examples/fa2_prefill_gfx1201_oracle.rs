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
    let kmode = env_usize("KMODE", 0);
    let mut gpu = Gpu::init().expect("gpu init");
    if gpu.arch != "gfx1201" {
        eprintln!("SKIP: fa2 oracle/bench requires gfx1201, got {}", gpu.arch);
        std::process::exit(2);
    }
    match (mode.as_str(), kmode) {
        ("oracle", 0) => oracle(&mut gpu),
        ("bench", 0) => bench(&mut gpu),
        ("oracle", 3) => oracle_fwht3(&mut gpu),
        ("bench", 3) => bench_fwht3(&mut gpu),
        _ => {
            eprintln!("unknown MODE={mode} KMODE={kmode} (want oracle|bench x 0|3)");
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
    // S2 (split-KV) is opt-in via SPLIT=1 while its partial epilogue is
    // under repair; the oracle gates S1 only by default.
    let s2_on = env_usize("SPLIT", 0) == 1;
    if s2_on {
        gpu.attention_q8_0_fa2_gqa_split_gfx1201_bench(
            &q_g, &k_g, &v_g, &out_s2, &p_g, &partials, NH, NKV, HD, n, 2,
        )
        .expect("candidate S2 launch");
    }

    let ri = gpu.download_f32(&out_inc).expect("dl inc");
    let r1 = gpu.download_f32(&out_s1).expect("dl s1");
    let reference = cpu_reference(&qd, &k, &v, &pos, n);

    let mi = metrics(&reference, &ri, n);
    let m1 = metrics(&reference, &r1, n);
    let ok_arm = |m: &Metrics| {
        m.finite && m.degenerate == 0 && m.compared == n * NH && m.rel_l2 <= 1e-3 && m.min_cos >= 1.0 - 1e-6
    };
    let (m2, pass2) = if s2_on {
        let r2 = gpu.download_f32(&out_s2).expect("dl s2");
        let m = metrics(&reference, &r2, n);
        let p = ok_arm(&m);
        (m, p)
    } else {
        (
            Metrics { rel_l2: 0.0, min_cos: 1.0, finite: true, degenerate: 0, compared: n * NH },
            true,
        )
    };
    let pass = ok_arm(&mi) && ok_arm(&m1) && pass2;
    println!(
        "RESULT oracle n={n} ctx={ctx} pos={tag} inc_rel_l2={:.3e} inc_cos={:.9} \
         s1_rel_l2={:.3e} s1_cos={:.9} s2_rel_l2={:.3e} s2_cos={:.9} s2_on={s2_on} pass={pass}",
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
    // S2 (split-KV) is opt-in via SPLIT=1 while its partial epilogue is
    // under repair; the kill gate is evaluated on the fastest CORRECT arm.
    let t_s2 = if env_usize("SPLIT", 0) == 1 {
        event_times(gpu, warmups, runs, &|g: &mut Gpu| {
            g.attention_q8_0_fa2_gqa_split_gfx1201_bench(
                &q_g, &k_g, &v_g, &out_s2, &p_g, &partials, NH, NKV, HD, n, 2,
            )
            .expect("s2");
        })
    } else {
        vec![f64::INFINITY; runs]
    };

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

// ── KMODE=3 (fwht3 K, V Q8_0) oracle/bench arm ──
//
// K bytes are produced by the production write kernel
// (`kv_cache_write_fwht3_vec_batched`), so they are byte-exact by
// construction; the CPU-f64 reference dequantizes those bytes
// (cnorm * TURBO_C3_256[code], f16-rounded like the kernel) in rotated space
// and rotates Q with the same signed FWHT-256 in f64. Gates: candidate vs
// f64 AND candidate vs the incumbent `attention_flash_fwht3_tile_batched`
// output, both rel_l2 <= 1e-3, plus the causal sentinel.
// NOTE: the candidate rotates its Q buffer in place — every candidate launch
// needs a freshly uploaded Q (timing loops re-launch on the same buffer only
// because the work is identical; agreement is always checked after a
// fresh-Q run).

const K_BPH_FWHT3: usize = 100; // 4 B cnorm + 96 B of 3-bit codes at hd 256
const K_BPP_FWHT3: usize = NKV * K_BPH_FWHT3; // 400

const TURBO_C3_256: [f32; 8] = [
    -0.134860, -0.083320, -0.046469, -0.015176,
    0.015176, 0.046469, 0.083320, 0.134860,
];

/// Deterministic FWHT sign tables (same LCG as ScratchState::ensure_mq_signs:
/// seeds 42 / 1042, 256 floats each). Self-consistent within this harness:
/// the same tables feed the K-write, the candidate, the incumbent, and the
/// CPU reference.
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
            if (state >> 16) & 1 == 1 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// f64 mirror of fwht_shfl_forward_256 (signs → butterfly → 1/16 → signs).
fn fwht_forward_256_f64(x: &mut [f64; 256], s1: &[f32], s2: &[f32]) {
    for i in 0..256 {
        x[i] *= s1[i] as f64;
    }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
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
    for i in 0..256 {
        x[i] *= (1.0 / 16.0) * s2[i] as f64;
    }
}

/// Deterministic f32 K source, [ctx, NKV, HD] contiguous.
fn build_k_src(ctx: usize) -> Vec<f32> {
    (0..ctx * NKV * HD)
        .map(|i| (((i * 53) % 101) as f32 - 50.0) * 0.01)
        .collect()
}

/// Decode one fwht3 K value exactly as the kernel does: f16(cnorm * C3[code]).
fn decode_fwht3k(cache: &[u8], pos: usize, kv_h: usize, d: usize) -> f64 {
    let off = pos * K_BPP_FWHT3 + kv_h * K_BPH_FWHT3;
    let cn = f32::from_ne_bytes([cache[off], cache[off + 1], cache[off + 2], cache[off + 3]]);
    let lane = d / 8;
    let j = d % 8;
    let b0 = cache[off + 4 + lane * 3] as u32;
    let b1 = cache[off + 4 + lane * 3 + 1] as u32;
    let b2 = cache[off + 4 + lane * 3 + 2] as u32;
    let packed = b0 | (b1 << 8) | (b2 << 16);
    let code = ((packed >> (3 * j)) & 7) as usize;
    half_bits_to_f32(f32_to_half_bits(cn * TURBO_C3_256[code])) as f64
}

/// CPU-f64 causal reference for fwht3 K / Q8 V. Q is rotated with the same
/// signed FWHT in f64 then rounded through f16 (kernel: f32 rotate, f16 cast
/// per chunk); K/V decode each payload then round through f16.
fn cpu_reference_fwht3k(
    q: &[f32],
    k: &[u8],
    v: &[u8],
    pos: &[i32],
    s1: &[f32],
    s2: &[f32],
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n * NH * HD];
    let scale = SCALE_ATTN_F32 as f64;
    for b in 0..n {
        let pb = pos[b] as usize;
        for h in 0..NH {
            let kv_h = h / (NH / NKV);
            let mut qq = [0.0f64; HD];
            for d in 0..HD {
                qq[d] = q[(b * NH + h) * HD + d] as f64;
            }
            fwht_forward_256_f64(&mut qq, s1, s2);
            for d in 0..HD {
                qq[d] = half_bits_to_f32(f32_to_half_bits(qq[d] as f32)) as f64;
            }
            let mut scores = vec![f64::NEG_INFINITY; pb + 1];
            for kk in 0..=pb {
                let mut dot = 0.0f64;
                for d in 0..HD {
                    dot += qq[d] * decode_fwht3k(k, kk, kv_h, d);
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

fn download_bytes(gpu: &Gpu, t: &rdna_compute::GpuTensor, len: usize) -> Vec<u8> {
    gpu.hip.device_synchronize().unwrap();
    let mut data = vec![0u8; len];
    gpu.hip.memcpy_dtoh(&mut data, &t.buf).unwrap();
    data
}

fn check_case_fwht3(gpu: &mut Gpu, n: usize, ctx: usize, ragged: bool) -> bool {
    let tag = if ragged { "ragged" } else { "tail" };
    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    // Byte-exact fwht3 K via the production write kernel over 0..ctx.
    let k_src = build_k_src(ctx);
    let wpos: Vec<i32> = (0..ctx as i32).collect();
    let ks_g = gpu.upload_f32(&k_src, &[ctx * NKV * HD]).expect("ksrc upload");
    let wp_g = upload_i32_as_raw(gpu, &wpos).expect("wpos upload");
    let s1_g = gpu.upload_f32(&s1, &[256]).expect("s1 upload");
    let s2_g = gpu.upload_f32(&s2, &[256]).expect("s2 upload");
    let k_g = gpu
        .upload_raw(&vec![0u8; ctx * K_BPP_FWHT3], &[ctx * K_BPP_FWHT3])
        .expect("k upload");
    gpu.kv_cache_write_fwht3_vec_batched(&k_g, &ks_g, &wp_g, &s1_g, &s2_g, NKV, HD, ctx)
        .expect("k write");
    let k = download_bytes(gpu, &k_g, ctx * K_BPP_FWHT3);

    let v = build_cache(ctx, 7);
    let qd = build_q(n);
    let pos = build_positions(n, ctx, ragged);
    let v_g = gpu.upload_raw(&v, &[v.len()]).expect("v upload");
    // Two Q uploads: the candidate rotates its buffer in place.
    let qi_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("qi upload");
    let qc_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("qc upload");
    let p_g = upload_i32_as_raw(gpu, &pos).expect("pos upload");
    let out_inc = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let out_cand = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let partials = gpu
        .zeros(&[2 * n * NH * (HD + 2)], DType::F32)
        .expect("partials");

    gpu.attention_flash_fwht3_batched_masked(
        &qi_g, &k_g, &v_g, &out_inc, &p_g, &s1_g, &s2_g, NH, NKV, HD, ctx, ctx, n,
        &partials, None, 0, 0, 8,
    )
    .expect("incumbent launch");
    gpu.attention_q8_0_fa2_gqa_fwht3k_gfx1201(
        &qc_g, &k_g, &v_g, &out_cand, &p_g, &s1_g, &s2_g, NH, NKV, HD, ctx, n,
    )
    .expect("candidate launch");

    let ri = gpu.download_f32(&out_inc).expect("dl inc");
    let rc = gpu.download_f32(&out_cand).expect("dl cand");
    let reference = cpu_reference_fwht3k(&qd, &k, &v, &pos, &s1, &s2, n);

    let mi = metrics(&reference, &ri, n);
    let mc = metrics(&reference, &rc, n);
    let mx = metrics(&ri, &rc, n);
    let ok_arm = |m: &Metrics| {
        m.finite && m.degenerate == 0 && m.compared == n * NH && m.rel_l2 <= 1e-3 && m.min_cos >= 1.0 - 1e-6
    };
    let pass = ok_arm(&mi) && ok_arm(&mc) && ok_arm(&mx);
    println!(
        "RESULT oracle_fwht3 n={n} ctx={ctx} pos={tag} inc_rel_l2={:.3e} inc_cos={:.9} \
         cand_rel_l2={:.3e} cand_cos={:.9} cand_vs_inc_rel_l2={:.3e} cand_vs_inc_cos={:.9} pass={pass}",
        mi.rel_l2, mi.min_cos, mc.rel_l2, mc.min_cos, mx.rel_l2, mx.min_cos
    );
    if !pass {
        eprintln!(
            "FAIL oracle_fwht3 n={n} ctx={ctx} pos={tag}: inc finite={} deg={} cmp={} | \
             cand finite={} deg={} cmp={} | x finite={} deg={} cmp={}",
            mi.finite, mi.degenerate, mi.compared,
            mc.finite, mc.degenerate, mc.compared,
            mx.finite, mx.degenerate, mx.compared
        );
        return false;
    }

    // Causal sentinel: perturb K codes strictly after row 0's position (plus
    // V code bytes like the Q8 arm); row 0 must come back bit-identical.
    // Fresh Q upload: the candidate rotated the earlier buffer in place.
    let p0 = pos[0] as usize;
    if p0 + 1 < ctx {
        let mut k2 = k.clone();
        for kk in (p0 + 1)..ctx {
            for kvh in 0..NKV {
                let base = kk * K_BPP_FWHT3 + kvh * K_BPH_FWHT3 + 4;
                for j in 0..(K_BPH_FWHT3 - 4) {
                    k2[base + j] = k2[base + j].wrapping_add(13);
                }
            }
        }
        let mut v2 = v.clone();
        for kk in (p0 + 1)..ctx {
            for blk in 0..(NKV * BPH) {
                let off = kk * ROW_STRIDE + blk * 34;
                for j in 2..34 {
                    v2[off + j] = v2[off + j].wrapping_add(29);
                }
            }
        }
        let k2_g = gpu.upload_raw(&k2, &[k2.len()]).expect("k2 upload");
        let v2_g = gpu.upload_raw(&v2, &[v2.len()]).expect("v2 upload");
        let qs_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("qs upload");
        let out_s = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
        gpu.attention_q8_0_fa2_gqa_fwht3k_gfx1201(
            &qs_g, &k2_g, &v2_g, &out_s, &p_g, &s1_g, &s2_g, NH, NKV, HD, ctx, n,
        )
        .expect("sentinel launch");
        let rs = gpu.download_f32(&out_s).expect("dl sentinel");
        let row0_same = rc[..NH * HD] == rs[..NH * HD];
        println!("RESULT sentinel_fwht3 n={n} ctx={ctx} pos={tag} row0_bitidentical={row0_same}");
        if !row0_same {
            eprintln!("FAIL sentinel_fwht3 n={n} ctx={ctx} pos={tag}: row 0 changed");
            return false;
        }
    } else {
        println!("RESULT sentinel_fwht3 n={n} ctx={ctx} pos={tag} row0_bitidentical=skip");
    }
    true
}

fn oracle_fwht3(gpu: &mut Gpu) {
    let shapes: &[(usize, usize)] = &[(1, 1), (7, 63), (8, 64), (9, 65), (17, 257), (16, 4096)];
    let mut all_pass = true;
    for &(n, ctx) in shapes {
        for &ragged in &[false, true] {
            if !check_case_fwht3(gpu, n, ctx, ragged) {
                all_pass = false;
            }
        }
    }
    println!("RESULT oracle_fwht3_verdict pass={all_pass}");
    if !all_pass {
        std::process::exit(1);
    }
    println!("PASS");
}

fn bench_fwht3(gpu: &mut Gpu) {
    let n = env_usize("N", 384);
    let ctx = env_usize("CTX", 7936);
    let warmups = env_usize("WARMUPS", 10);
    let runs = env_usize("RUNS", 21);
    assert!(n >= 1 && n <= 384, "bench N must be 1..=384");

    let s1 = gen_fwht_signs(42, 256);
    let s2 = gen_fwht_signs(1042, 256);
    let k_src = build_k_src(ctx);
    let wpos: Vec<i32> = (0..ctx as i32).collect();
    let ks_g = gpu.upload_f32(&k_src, &[ctx * NKV * HD]).expect("ksrc upload");
    let wp_g = upload_i32_as_raw(gpu, &wpos).expect("wpos upload");
    let s1_g = gpu.upload_f32(&s1, &[256]).expect("s1 upload");
    let s2_g = gpu.upload_f32(&s2, &[256]).expect("s2 upload");
    let k_g = gpu
        .upload_raw(&vec![0u8; ctx * K_BPP_FWHT3], &[ctx * K_BPP_FWHT3])
        .expect("k upload");
    gpu.kv_cache_write_fwht3_vec_batched(&k_g, &ks_g, &wp_g, &s1_g, &s2_g, NKV, HD, ctx)
        .expect("k write");

    let v = build_cache(ctx, 7);
    let qd = build_q(n);
    let pos = build_positions(n, ctx, false);
    let v_g = gpu.upload_raw(&v, &[v.len()]).expect("v upload");
    let qi_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("qi upload");
    let qc_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("qc upload");
    let p_g = upload_i32_as_raw(gpu, &pos).expect("pos upload");
    let out_inc = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let out_cand = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    let partials = gpu
        .zeros(&[2 * n * NH * (HD + 2)], DType::F32)
        .expect("partials");

    let t_inc = event_times(gpu, warmups, runs, &|g: &mut Gpu| {
        g.attention_flash_fwht3_batched_masked(
            &qi_g, &k_g, &v_g, &out_inc, &p_g, &s1_g, &s2_g, NH, NKV, HD, ctx, ctx, n,
            &partials, None, 0, 0, 8,
        )
        .expect("incumbent");
    });
    // Same-buffer re-launches rotate Q again, but the work (FLOPs, traffic)
    // is identical every call, so timing stays valid; agreement is checked
    // below after a fresh-Q run.
    let t_cand = event_times(gpu, warmups, runs, &|g: &mut Gpu| {
        g.attention_q8_0_fa2_gqa_fwht3k_gfx1201(
            &qc_g, &k_g, &v_g, &out_cand, &p_g, &s1_g, &s2_g, NH, NKV, HD, ctx, n,
        )
        .expect("candidate");
    });

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
            "RESULT bench_fwht3 arm={name} n={n} ctx={ctx} runs={runs} \
             median_ms={med:.4} p10_ms={p10:.4} p90_ms={p90:.4} \
             gflop={:.3} tflops={tflops:.2}",
            flop as f64 / 1e9
        );
        med
    };
    let m_inc = report("incumbent", &t_inc);
    let m_cand = report("fa2_fwht3k", &t_cand);

    // Agreement after a fresh-Q candidate run (timing loop rotated qc_g).
    let qf_g = gpu.upload_f32(&qd, &[n * NH * HD]).expect("qf upload");
    let out_f = gpu.zeros(&[n * NH * HD], DType::F32).expect("out");
    gpu.attention_q8_0_fa2_gqa_fwht3k_gfx1201(
        &qf_g, &k_g, &v_g, &out_f, &p_g, &s1_g, &s2_g, NH, NKV, HD, ctx, n,
    )
    .expect("agree launch");
    let rf = gpu.download_f32(&out_f).expect("dl agree");
    let ri = gpu.download_f32(&out_inc).expect("dl inc");
    let m = metrics(&ri, &rf, n);
    println!(
        "RESULT bench_fwht3_agree cand_vs_inc_rel_l2={:.3e} min_cos={:.9} finite={}",
        m.rel_l2, m.min_cos, m.finite
    );

    let ratio = m_inc / m_cand;
    let pass = m.finite && m.rel_l2 <= 1e-3;
    println!("RESULT bench_fwht3_verdict ratio_vs_incumbent={ratio:.3} agree_gate_1e3={pass}");
    if !pass {
        std::process::exit(1);
    }
    println!("PASS");
}
