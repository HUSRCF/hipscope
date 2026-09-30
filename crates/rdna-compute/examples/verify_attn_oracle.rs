// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Byte oracle for VerifyAttn (`Gpu::try_attention_verify_gqa`) against the
// batched flash tile + reduce it replaces
// (`attention_flash_{fp8_e4m3,q8_0}_tile_batched` +
// `attention_flash_asym_reduce_batched`), H2 shape (24 q heads, 4 kv heads,
// head_dim 256).
//
// Every run poisons the whole output and partials allocations (including guard
// regions around the views the launch receives), and the KV cache past the
// last written position holds NaN codes/scales, so a read or write outside the
// reference footprint shows up as a byte difference. The reference and
// candidate must match byte for byte over the full allocations.
//
// Modes:
//   oracle [fp8|q8|both]      B {1,4,16} (+2,5,9,32) x ctx {1,127,128,2K,8K,16K,32K,64K},
//                             eager and graph-capture max_ctx_len, full and
//                             sub-batched partials, cap 262144 capture grid.
//   stress <reps>             poisoned repeats (rotating 0xFF/0x7F/0x00/0xA5),
//                             each compared with the reference and the first
//                             candidate launch. Run it under
//                             ROC_GLOBAL_CU_MASK=0x1 and as two concurrent
//                             processes for the co-resident modes.
//   soak <secs>               back-to-back candidate launches (B16 32K fp8/q8),
//                             every output compared with the reference.
//   bench                     wall-clock per call, reference vs candidate.
// Exit 0 = pass. Run: cargo run --release -p rdna-compute --example verify_attn_oracle -- oracle both

use rdna_compute::attention::VerifyKv;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::time::Instant;

const NH: usize = 24;
const NKV: usize = 4;
const HD: usize = 256;
const TILE: usize = 128;
const STRIDE: usize = 2 + HD;
const QDIM: usize = NH * HD;
/// Guard elements on each side of the output view and after the partials view.
const GUARD: usize = 4096;
const POS_GUARD: i32 = 0x3fff_ffff;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn f16_bits(x: f32) -> u16 {
    half_from_f32(x)
}

// Round-to-nearest-even f32 -> f16 bits (no external crate in examples).
fn half_from_f32(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && (half & 1) == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) { half + 1 } else { half };
    sign | r as u16
}

fn row_bytes(kv: VerifyKv) -> usize {
    match kv {
        VerifyKv::Q8 => NKV * (HD / 32) * 34,
        VerifyKv::Fp8 => NKV * (HD + 2),
    }
}

/// Block/head scale: mostly positive normal f16 in [2^-9, 2^-4], with some
/// negative, zero and subnormal scales to pin signed-zero and denormal paths.
fn scale_f16(rng: &mut Rng) -> u16 {
    match rng.below(64) {
        0 => 0x0000,
        1 => 0x8000,
        2 => 0x0001 + rng.below(0x3ff) as u16,
        3 | 4 => f16_bits(-(0.002 + 0.06 * rng.unit())),
        _ => f16_bits(0.002 + 0.06 * rng.unit()),
    }
}

/// K or V cache: `valid` random rows, then NaN-poisoned rows up to `cap`.
fn fill_kv(kv: VerifyKv, rng: &mut Rng, valid: usize, cap: usize) -> Vec<u8> {
    let rb = row_bytes(kv);
    let mut buf = vec![0u8; cap * rb];
    for t in 0..cap {
        let row = &mut buf[t * rb..(t + 1) * rb];
        let poison = t >= valid;
        match kv {
            VerifyKv::Q8 => {
                for blk in 0..NKV * (HD / 32) {
                    let o = blk * 34;
                    let s = if poison { 0x7e00 } else { scale_f16(rng) };
                    row[o..o + 2].copy_from_slice(&s.to_le_bytes());
                    for j in 0..32 {
                        row[o + 2 + j] = if poison {
                            0x7f
                        } else if rng.below(16) == 0 {
                            [0x80u8, 0x00, 0x7f, 0x81][rng.below(4) as usize]
                        } else {
                            rng.next() as u8
                        };
                    }
                }
            }
            VerifyKv::Fp8 => {
                for i in 0..NKV * HD {
                    row[i] = if poison {
                        0x7f
                    } else {
                        loop {
                            let c = rng.next() as u8;
                            if c & 0x7f != 0x7f {
                                break c;
                            }
                        }
                    };
                }
                for h in 0..NKV {
                    let s = if poison { 0x7e00 } else { scale_f16(rng) };
                    let o = NKV * HD + 2 * h;
                    row[o..o + 2].copy_from_slice(&s.to_le_bytes());
                }
            }
        }
    }
    buf
}

#[derive(Clone, Copy, Debug)]
struct Case {
    kv: VerifyKv,
    b: usize,
    ctx: usize,
    /// Graph-capture shape: `max_ctx_len = cap` instead of the logical context.
    capture: bool,
    cap: usize,
    /// Partials capacity in rows (at this case's max_tiles); < b sub-batches.
    partial_rows: usize,
}

impl Case {
    fn start(&self) -> usize {
        self.ctx - 1
    }
    fn valid(&self) -> usize {
        self.start() + self.b
    }
    fn max_ctx_len(&self) -> usize {
        if self.capture {
            self.cap
        } else {
            self.valid()
        }
    }
    fn max_tiles(&self) -> usize {
        self.max_ctx_len().div_ceil(TILE)
    }
    fn partials_len(&self) -> usize {
        self.partial_rows * NH * self.max_tiles() * STRIDE
    }
}

struct Bufs {
    q: GpuTensor,
    q_full: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    pos: GpuTensor,
    out: GpuTensor,
    out_full: GpuTensor,
    part: GpuTensor,
    part_full: GpuTensor,
}

impl Bufs {
    fn new(gpu: &mut Gpu, c: &Case, seed: u64) -> Bufs {
        let mut rng = Rng(seed);
        let mut qh = vec![0f32; c.b * QDIM + GUARD];
        for (i, x) in qh.iter_mut().enumerate() {
            *x = if i >= c.b * QDIM {
                f32::NAN
            } else {
                // Rows alternate between ordinary and 12x-scaled queries so
                // some tiles run their softmax deep into expf underflow.
                let row = i / QDIM;
                let amp = if row % 3 == 2 { 12.0 } else { 1.0 };
                (rng.unit() * 2.0 - 1.0) * amp
            };
        }
        let q_full = gpu.upload_f32(&qh, &[qh.len()]).expect("q");
        let q = q_full.sub_offset(0, c.b * QDIM);
        let kh = fill_kv(c.kv, &mut rng, c.valid(), c.cap);
        let vh = fill_kv(c.kv, &mut rng, c.valid(), c.cap);
        let k = gpu.upload_raw(&kh, &[kh.len()]).expect("k");
        let v = gpu.upload_raw(&vh, &[vh.len()]).expect("v");
        let mut posh: Vec<i32> = (0..c.b).map(|i| (c.start() + i) as i32).collect();
        posh.extend(std::iter::repeat(POS_GUARD).take(8));
        let pb: Vec<u8> = posh.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let pos = gpu.upload_raw(&pb, &[posh.len()]).expect("pos");
        let out_full = gpu.zeros(&[c.b * QDIM + 2 * GUARD], DType::F32).expect("out");
        let out = out_full.sub_offset(GUARD, c.b * QDIM);
        let part_full = gpu.zeros(&[c.partials_len() + GUARD], DType::F32).expect("partials");
        let part = part_full.sub_offset(0, c.partials_len());
        Bufs { q, q_full, k, v, pos, out, out_full, part, part_full }
    }

    fn free(self, gpu: &mut Gpu) {
        for t in [self.q_full, self.k, self.v, self.pos, self.out_full, self.part_full] {
            let _ = gpu.free_tensor(t);
        }
        let _ = (self.q, self.out, self.part);
    }
}

fn poison(gpu: &Gpu, t: &GpuTensor, byte: u8) {
    gpu.hip.memset(&t.buf, byte as i32, t.buf.size()).expect("memset");
}

fn bytes_of(gpu: &Gpu, t: &GpuTensor) -> Vec<u8> {
    gpu.hip.device_synchronize().expect("sync");
    let mut v = vec![0u8; t.buf.size()];
    gpu.hip.memcpy_dtoh(&mut v, &t.buf).expect("dtoh");
    v
}

/// One launch of the reference (`candidate == false`) or VerifyAttn.
fn launch(gpu: &mut Gpu, c: &Case, bf: &Bufs, candidate: bool) {
    std::sync::Arc::make_mut(&mut gpu.flags).verify_attn = candidate;
    if candidate {
        let ran = gpu
            .try_attention_verify_gqa(
                c.kv, &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.max_ctx_len(), c.b, &bf.part,
                None, 0,
            )
            .expect("verify_attn launch");
        assert!(ran, "VerifyAttn declined an admitted case: {c:?}");
    } else {
        match c.kv {
            VerifyKv::Fp8 => gpu
                .attention_flash_fp8_e4m3_tile_batched(
                    &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.cap, c.max_ctx_len(), c.b,
                    &bf.part, None, 0, 0,
                )
                .expect("reference fp8"),
            VerifyKv::Q8 => gpu
                .attention_flash_q8_0_batched_masked(
                    &bf.q, &bf.k, &bf.v, &bf.out, &bf.pos, NH, NKV, HD, c.cap, c.max_ctx_len(), c.b,
                    &bf.part, None, 0, 0,
                )
                .expect("reference q8"),
        }
    }
}

/// Poison both allocations with `byte`, launch, return (out_full, part_full).
fn run(gpu: &mut Gpu, c: &Case, bf: &Bufs, candidate: bool, byte: u8) -> (Vec<u8>, Vec<u8>) {
    poison(gpu, &bf.out_full, byte);
    poison(gpu, &bf.part_full, byte);
    launch(gpu, c, bf, candidate);
    (bytes_of(gpu, &bf.out_full), bytes_of(gpu, &bf.part_full))
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or(if a.len() != b.len() { Some(a.len().min(b.len())) } else { None })
}

fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn out_view(c: &Case, full: &[u8]) -> Vec<f32> {
    as_f32(&full[GUARD * 4..(GUARD + c.b * QDIM) * 4])
}

/// Byte ranges of the partials the reference writes (tiles below each row's
/// causal bound), in chunk-local rows of the sub-batched launch.
fn written_ranges(c: &Case) -> Vec<(usize, usize)> {
    let mt = c.max_tiles();
    let sub = c.partial_rows.min(c.b);
    let mut r = Vec::new();
    let mut off = 0;
    while off < c.b {
        let chunk = (c.b - off).min(sub);
        for lr in 0..chunk {
            let seq = c.start() + off + lr + 1;
            let nt = seq.div_ceil(TILE).min(mt);
            for h in 0..NH {
                let base = ((lr * NH + h) * mt) * STRIDE * 4;
                r.push((base, base + nt * STRIDE * 4));
            }
        }
        off += chunk;
    }
    r
}

fn kv_name(kv: VerifyKv) -> &'static str {
    match kv {
        VerifyKv::Fp8 => "fp8",
        VerifyKv::Q8 => "q8",
    }
}

fn oracle_cases(kvs: &[VerifyKv]) -> Vec<Case> {
    const CAP: usize = 65536 + 256;
    let mut cases = Vec::new();
    for &kv in kvs {
        for &b in &[1usize, 4, 16] {
            for &ctx in &[1usize, 127, 128, 2048, 8192, 16384, 32768, 65536] {
                for &capture in &[false, true] {
                    cases.push(Case { kv, b, ctx, capture, cap: CAP, partial_rows: 16 });
                }
            }
        }
        // Odd batch sizes, ragged row groups, the 32-row ceiling, sub-batching.
        for &(b, ctx, rows) in &[(2, 129, 16), (5, 2047, 16), (9, 4097, 16), (32, 3000, 32), (16, 8191, 5), (13, 300, 4)] {
            cases.push(Case { kv, b, ctx, capture: false, cap: CAP, partial_rows: rows });
            cases.push(Case { kv, b, ctx, capture: true, cap: CAP, partial_rows: rows });
        }
        // The real graph-capture shape: physical cap 262144 (2048 partial tiles).
        for &(b, ctx) in &[(16, 2048), (4, 20000), (16, 9000)] {
            cases.push(Case { kv, b, ctx, capture: true, cap: 262144, partial_rows: 16 });
        }
    }
    cases
}

fn parse_kvs(arg: Option<&str>) -> Vec<VerifyKv> {
    match arg.unwrap_or("both") {
        "fp8" => vec![VerifyKv::Fp8],
        "q8" => vec![VerifyKv::Q8],
        _ => vec![VerifyKv::Fp8, VerifyKv::Q8],
    }
}

fn oracle(gpu: &mut Gpu, kvs: &[VerifyKv]) -> bool {
    let mut ok = true;
    let mut n = 0;
    for (i, c) in oracle_cases(kvs).iter().enumerate() {
        let bf = Bufs::new(gpu, c, 0x5eed_0000 + i as u64);
        let (ro, rp) = run(gpu, c, &bf, false, 0xA5);
        let (co, cp) = run(gpu, c, &bf, true, 0xA5);
        let od = first_diff(&ro, &co);
        let pd = first_diff(&rp, &cp);
        let rv = out_view(c, &ro);
        let nan = rv.iter().filter(|x| !x.is_finite()).count();
        let pass = od.is_none() && pd.is_none() && nan == 0;
        let (mut abs, mut rel) = (0f32, 0f32);
        if od.is_some() {
            let cv = out_view(c, &co);
            for (a, b) in rv.iter().zip(&cv) {
                let d = (a - b).abs();
                abs = abs.max(d);
                rel = rel.max(d / a.abs().max(1e-30));
            }
        }
        println!(
            "ORACLE kv={} B={} ctx={} capture={} cap={} max_tiles={} partial_rows={} out_bytes={} part_bytes={} \
             ref_nonfinite={} out_first_diff={:?} part_first_diff={:?} max_abs={:e} max_rel={:e} {}",
            kv_name(c.kv), c.b, c.ctx, c.capture, c.cap, c.max_tiles(), c.partial_rows, ro.len(), rp.len(), nan,
            od, pd, abs, rel, if pass { "PASS" } else { "FAIL" }
        );
        ok &= pass;
        n += 1;
        bf.free(gpu);
    }
    println!("ORACLE_SUMMARY cases={n} {}", if ok { "PASS" } else { "FAIL" });
    ok
}

fn stress(gpu: &mut Gpu, reps: usize, kvs: &[VerifyKv]) -> bool {
    let mut ok = true;
    let mut launches = 0usize;
    let mut mism = 0usize;
    for &kv in kvs {
        for &(b, ctx, capture, cap) in &[
            (16usize, 8192usize, true, 65536 + 256),
            (16, 1000, false, 4096),
            (4, 16384, false, 20000),
            (1, 127, true, 4096),
            (9, 2049, true, 4096),
        ] {
            let c = Case { kv, b, ctx, capture, cap, partial_rows: 16 };
            let bf = Bufs::new(gpu, &c, 0xC0FFEE ^ (b * 131 + ctx) as u64);
            let (ro, rp) = run(gpu, &c, &bf, false, 0xA5);
            let ref_out = out_view(&c, &ro);
            let ranges = written_ranges(&c);
            let mut first: Option<Vec<u8>> = None;
            let mut bad = 0usize;
            for r in 0..reps {
                let byte = [0xFFu8, 0x7F, 0x00, 0xA5][r % 4];
                let (co, cp) = run(gpu, &c, &bf, true, byte);
                let cv = out_view(&c, &co);
                let mut same = cv.iter().zip(&ref_out).all(|(a, b)| a.to_bits() == b.to_bits());
                // Guards keep this launch's poison byte.
                same &= co[..GUARD * 4].iter().all(|&x| x == byte);
                same &= co[(GUARD + c.b * QDIM) * 4..].iter().all(|&x| x == byte);
                // Written partials equal the reference's; the rest keeps the poison.
                let mut covered = vec![false; cp.len()];
                for &(s, e) in &ranges {
                    same &= cp[s..e] == rp[s..e];
                    covered[s..e].iter_mut().for_each(|x| *x = true);
                }
                same &= cp.iter().zip(&covered).all(|(&x, &cov)| cov || x == byte);
                let outb = co[GUARD * 4..(GUARD + c.b * QDIM) * 4].to_vec();
                match &first {
                    None => first = Some(outb),
                    Some(f) => same &= *f == outb,
                }
                launches += 1;
                if !same {
                    bad += 1;
                }
            }
            println!(
                "STRESS kv={} B={} ctx={} capture={} reps={} mismatched={} {}",
                kv_name(kv), b, ctx, capture, reps, bad, if bad == 0 { "PASS" } else { "FAIL" }
            );
            mism += bad;
            ok &= bad == 0;
            bf.free(gpu);
        }
    }
    println!(
        "STRESS_SUMMARY launches={launches} mismatched_launches={mism} cu_mask={} {}",
        std::env::var("ROC_GLOBAL_CU_MASK").unwrap_or_else(|_| "none".into()),
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn soak(gpu: &mut Gpu, secs: u64) -> bool {
    let cases = [
        Case { kv: VerifyKv::Fp8, b: 16, ctx: 32768, capture: true, cap: 32768 + 256, partial_rows: 16 },
        Case { kv: VerifyKv::Q8, b: 16, ctx: 32768, capture: true, cap: 32768 + 256, partial_rows: 16 },
    ];
    let bufs: Vec<Bufs> = cases.iter().enumerate().map(|(i, c)| Bufs::new(gpu, c, 0x50A4 + i as u64)).collect();
    let refs: Vec<Vec<f32>> = cases
        .iter()
        .zip(&bufs)
        .map(|(c, bf)| {
            let (o, _) = run(gpu, c, bf, false, 0xA5);
            out_view(c, &o)
        })
        .collect();
    let t0 = Instant::now();
    let (mut launches, mut bad) = (0usize, 0usize);
    while t0.elapsed().as_secs() < secs {
        for (i, c) in cases.iter().enumerate() {
            for _ in 0..16 {
                launch(gpu, c, &bufs[i], true);
                launches += 1;
            }
            let o = bytes_of(gpu, &bufs[i].out_full);
            let cv = out_view(c, &o);
            if !cv.iter().zip(&refs[i]).all(|(a, b)| a.to_bits() == b.to_bits()) {
                bad += 1;
            }
        }
    }
    println!(
        "SOAK secs={} launches={} mismatched_checks={} {}",
        t0.elapsed().as_secs_f64(),
        launches,
        bad,
        if bad == 0 { "PASS" } else { "FAIL" }
    );
    for bf in bufs {
        bf.free(gpu);
    }
    bad == 0
}

fn bench(gpu: &mut Gpu, kvs: &[VerifyKv]) {
    for &kv in kvs {
        for &b in &[4usize, 16] {
            for &ctx in &[2048usize, 8192, 16384, 32768, 65536] {
                for &(capture, cap) in &[(false, ctx + 64), (true, 262144usize)] {
                    let c = Case { kv, b, ctx, capture, cap: cap.max(ctx + 64), partial_rows: 16 };
                    let bf = Bufs::new(gpu, &c, 0xBE4C);
                    let mut ms = [0f64; 2];
                    for (arm, cand) in [(0usize, false), (1, true)] {
                        for _ in 0..3 {
                            launch(gpu, &c, &bf, cand);
                        }
                        gpu.hip.device_synchronize().expect("sync");
                        let iters = 20;
                        let t = Instant::now();
                        for _ in 0..iters {
                            launch(gpu, &c, &bf, cand);
                        }
                        gpu.hip.device_synchronize().expect("sync");
                        ms[arm] = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
                    }
                    println!(
                        "BENCH kv={} B={} ctx={} capture={} ref_ms={:.4} verify_attn_ms={:.4} speedup={:.2}",
                        kv_name(kv), b, ctx, capture, ms[0], ms[1], ms[0] / ms[1]
                    );
                    bf.free(gpu);
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("oracle");
    let mut gpu = Gpu::init().expect("gpu init");
    assert_eq!(gpu.arch, "gfx1201", "VerifyAttn oracle runs on gfx1201");
    let ok = match mode {
        "oracle" => oracle(&mut gpu, &parse_kvs(args.get(2).map(String::as_str))),
        "stress" => {
            let reps = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
            stress(&mut gpu, reps, &parse_kvs(args.get(3).map(String::as_str)))
        }
        "soak" => soak(&mut gpu, args.get(2).and_then(|s| s.parse().ok()).unwrap_or(60)),
        "bench" => {
            bench(&mut gpu, &parse_kvs(args.get(2).map(String::as_str)));
            true
        }
        m => panic!("unknown mode {m}"),
    };
    std::process::exit(if ok { 0 } else { 1 });
}
