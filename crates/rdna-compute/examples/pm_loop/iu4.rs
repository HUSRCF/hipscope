// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! `iu4_gemm` workload: the builder-emitted MQ4V2 x `block_i4_128` GEMM for
//! gfx1201 (`peacemaker-isa emit --kernel iu4_gemm`) against the committed
//! production bundles and a CPU reference.
//!
//! # Shape grammar
//! `M=<rows>,N=<tokens>,K=<k>`: all three required, any order, comma
//! separated (e.g. `M=4096,N=512,K=5120`). `K % 256 == 0`, `M % 4 == 0`,
//! `N >= 1`; partial M/N tiles are handled by the kernel. For `silu` /
//! `silu-bf16` `M` is the number of intermediate rows (the gate and the up
//! matrix each have `M` rows, the kernel runs `2M` virtual rows and writes
//! `H[N x M]`), and `M % 128 == 0` as the production launcher requires
//! (`gemm_gate_up_silu_mq4g256v2_iu4_gfx12`). `qkvzagdn` is rejected: its
//! 120-byte fused ABI (conv state, Q/K/V rows, ...) is not reproduced here.
//!
//! # Knobs (candidate; names as `peacemaker-isa emit --kernel iu4_gemm`)
//! `fold` (k128|k256s|k256shared|k256p|k256pow2), `tile`
//! (128x128x8|256x128x16), `cacc` (1|2), `epi` (set|add|silu|silu-bf16),
//! `alayout` (token|slab). Baseline overrides use the `base-` prefix on
//! `fold|tile|cacc|alayout`; with none the baseline is the committed
//! production bundle for the candidate's activation layout, otherwise it is
//! emitted from the knobs with unspecified ones inherited.
//!
//! # Extra knobs
//! `ref-rows`: `full` (default while `M*N*K <= 2e9` MACs) or `sample:<n>`
//! (default `sample:192` beyond that): rows `0..64`, the last 64, and `n`
//! evenly spread rows, every token column. `max_abs_diff` always covers the
//! whole output. `ref-fold`: `fma` (default) or `muladd`, see below.
//!
//! # Reference numerics
//! Symmetric fold DAG (production `mq4v2_symmetric`): per K128 block
//! `kb = 2g + h` in ascending order, `dot = sum_t (nib ^ 8 as signed) * q`
//! (exact i32), `t1 = RN(sc_h * d)`, then the accumulate. The builder's fold
//! emitters (`iu4_k1.rs::emit_fold_scale`) issue `v_fmac_f32 acc += t1 * c`
//! with `c = float(C) - magic = float(dot)`: a FUSED `fma(t1, dot, acc)` from
//! `acc = +0.0`. The hipcc `_v3` DAG (`iu4_symmetric_fold_rn_pair`,
//! gemm_mq4g256v2_residual_mmq_iu4_v3.gfx12.hip:~280) is `RN(acc + RN(t1*dot))`.
//! The default follows the builder code (`fma`); `ref-fold=muladd` selects the
//! hipcc DAG. Output `Y[col * M + row]`, f32; ADD is `RN(Y0 + sum)`.

use crate::workload::{Arm, Knobs, Launch, Object, RefReport, Result, Workload};
use peacemaker_isa::kernels::iu4_gemm::{self, ALayout, Epi, Spec};
use peacemaker_isa::Arch;
use rdna_compute::Gpu;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cell::OnceCell;
use std::path::Path;

const GROUP_BYTES: usize = 136;
const REC: usize = 72;
const FULL_REF_MACS: u128 = 2_000_000_000;

fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.windows(needle.len()).any(|w| w == needle)
}

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefFold {
    Fma,
    MulAdd,
}

/// Host-side synthetic inputs: everything the reference needs.
pub struct Host {
    /// MQ4G256V2 weights `[M x K/256 x 136]` (gate for silu).
    pub w: Vec<u8>,
    /// Up weights (silu only, empty otherwise).
    pub w_up: Vec<u8>,
    /// `block_i4_128` records, token layout: record `kb * N + col`.
    pub recs: Vec<u8>,
    /// Initial `Y` for ADD `[N x M]`.
    pub y0: Vec<f32>,
}

fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn f16_val(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

fn synth_weights(m: usize, k: usize, seed: u64) -> Vec<u8> {
    let gpr = k / 256;
    let mut rng = Rng::new(seed);
    let mut w = vec![0u8; m * gpr * GROUP_BYTES];
    for blk in w.chunks_exact_mut(GROUP_BYTES) {
        let sc0 = 0.02 + (rng.next() % 16) as f32 * 0.001;
        let sc1 = 0.03 + (rng.next() % 16) as f32 * 0.001;
        for (i, sc) in [sc0, sc1].into_iter().enumerate() {
            let bits = f16_bits(sc);
            blk[4 * i..4 * i + 2].copy_from_slice(&bits.to_le_bytes());
            // zp is ignored by the symmetric fold; keep the asymmetric value.
            let zp = f16_bits(-8.0 * f16_val(bits));
            blk[4 * i + 2..4 * i + 4].copy_from_slice(&zp.to_le_bytes());
        }
        for q in blk[8..].chunks_exact_mut(8) {
            q.copy_from_slice(&rng.next().to_le_bytes());
        }
    }
    w
}

/// One `block_i4_128`: per-block symmetric `d = max|x| / 7`, `q = clamp(round(x/d), -8, 7)`.
fn quantize_block(x: &[f32; 128]) -> [u8; REC] {
    let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
    let d = amax / 7.0;
    let mut q = [0i32; 128];
    if d > 0.0 {
        for (qi, v) in q.iter_mut().zip(x) {
            *qi = ((v / d).round() as i32).clamp(-8, 7);
        }
    }
    let mut rec = [0u8; REC];
    rec[0..4].copy_from_slice(&d.to_le_bytes());
    rec[4..8].copy_from_slice(&q.iter().sum::<i32>().to_le_bytes());
    for j in 0..64 {
        rec[8 + j] = (q[2 * j] as u8 & 0xf) | ((q[2 * j + 1] as u8 & 0xf) << 4);
    }
    rec
}

impl Host {
    pub fn synth(m: usize, n: usize, k: usize, silu: bool) -> Self {
        let w = synth_weights(m, k, 0x9e37_79b9_7f4a_7c15);
        let w_up = if silu {
            synth_weights(m, k, 0xd1b5_4a32_d192_ed03)
        } else {
            Vec::new()
        };
        let mut rng = Rng::new(0x1234_5678_9abc_def1);
        let mut recs = vec![0u8; k / 128 * n * REC];
        for kb in 0..k / 128 {
            for col in 0..n {
                let amp = 1.0 + (col % 4) as f32;
                let mut x = [0f32; 128];
                for v in &mut x {
                    *v = rng.unit() * amp;
                }
                let o = (kb * n + col) * REC;
                recs[o..o + REC].copy_from_slice(&quantize_block(&x));
            }
        }
        let y0 = (0..n * m).map(|_| 0.25 + rng.unit()).collect();
        Self { w, w_up, recs, y0 }
    }
}

/// Token-layout records to the slab planes `[d:4N][s:4N][qs0..32:32N][qs32..64:32N]` per K128 block.
pub fn to_slab(recs: &[u8], n: usize) -> Vec<u8> {
    let mut out = vec![0u8; recs.len()];
    for (kb, blk) in out.chunks_exact_mut(REC * n).enumerate() {
        for col in 0..n {
            let r = &recs[(kb * n + col) * REC..][..REC];
            blk[4 * col..4 * col + 4].copy_from_slice(&r[0..4]);
            blk[4 * n + 4 * col..4 * n + 4 * col + 4].copy_from_slice(&r[4..8]);
            blk[8 * n + 32 * col..8 * n + 32 * col + 32].copy_from_slice(&r[8..40]);
            blk[40 * n + 32 * col..40 * n + 32 * col + 32].copy_from_slice(&r[40..72]);
        }
    }
    out
}

fn unpack_x(recs: &[u8]) -> (Vec<i8>, Vec<f32>) {
    let mut q = Vec::with_capacity(recs.len() / REC * 128);
    let mut d = Vec::with_capacity(recs.len() / REC);
    for r in recs.chunks_exact(REC) {
        d.push(f32::from_le_bytes(r[0..4].try_into().unwrap()));
        for &b in &r[8..] {
            q.push(((b << 4) as i8) >> 4);
            q.push((b as i8) >> 4);
        }
    }
    (q, d)
}

/// Reference `Y` rows (`rows` x N, row-major in `rows` order) for SET/ADD.
pub fn reference(
    host: &Host,
    m: usize,
    n: usize,
    k: usize,
    add: bool,
    rows: &[usize],
    fold: RefFold,
) -> Vec<f32> {
    let (xq, xd) = unpack_x(&host.recs);
    let (gpr, nkb) = (k / 256, k / 128);
    let mut out = vec![0f32; rows.len() * n];
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        for (rs, os) in rows.chunks(per).zip(out.chunks_mut(per * n)) {
            let (xq, xd) = (&xq, &xd);
            s.spawn(move || {
                let mut wq = vec![0i8; k];
                let mut sc = vec![0f32; nkb];
                for (&r, orow) in rs.iter().zip(os.chunks_exact_mut(n)) {
                    for g in 0..gpr {
                        let b = &host.w[(r * gpr + g) * GROUP_BYTES..][..GROUP_BYTES];
                        for h in 0..2 {
                            sc[2 * g + h] = f16_val(u16::from_le_bytes([b[4 * h], b[4 * h + 1]]));
                        }
                        for (j, &byte) in b[8..].iter().enumerate() {
                            // nibble ^ 8 as signed 4-bit (low nibble = even K).
                            wq[256 * g + 2 * j] = ((((byte & 0xf) ^ 8) << 4) as i8) >> 4;
                            wq[256 * g + 2 * j + 1] = (((byte >> 4) ^ 8) << 4) as i8 >> 4;
                        }
                    }
                    for (col, o) in orow.iter_mut().enumerate() {
                        let mut sum = 0f32;
                        for kb in 0..nkb {
                            let x = &xq[(kb * n + col) * 128..][..128];
                            let w = &wq[kb * 128..][..128];
                            let dot: i32 = w
                                .iter()
                                .zip(x)
                                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                                .sum();
                            let t1 = sc[kb] * xd[kb * n + col];
                            sum = match fold {
                                RefFold::Fma => t1.mul_add(dot as f32, sum),
                                RefFold::MulAdd => sum + t1 * dot as f32,
                            };
                        }
                        *o = if add { host.y0[col * m + r] + sum } else { sum };
                    }
                }
            });
        }
    });
    out
}

/// Deterministic sampled rows: first 64, last 64, and `n` evenly spread.
pub fn sample_rows(m: usize, n: usize) -> Vec<usize> {
    let mut s = std::collections::BTreeSet::new();
    s.extend(0..m.min(64));
    s.extend(m.saturating_sub(64)..m);
    for i in 0..n {
        s.insert(i * (m - 1) / n.saturating_sub(1).max(1));
    }
    s.into_iter().collect()
}

fn parse_shape(s: &str) -> Result<(usize, usize, usize)> {
    let (mut m, mut n, mut k) = (None, None, None);
    for part in s.split(',') {
        let (key, v) = part
            .split_once('=')
            .ok_or_else(|| format!("shape `{s}`: expected M=..,N=..,K=.."))?;
        let v: usize = v
            .trim()
            .parse()
            .map_err(|e| format!("shape `{s}`: {key}: {e}"))?;
        let slot = match key.trim() {
            "M" => &mut m,
            "N" => &mut n,
            "K" => &mut k,
            other => return Err(format!("shape `{s}`: unknown dimension `{other}`")),
        };
        if slot.replace(v).is_some() {
            return Err(format!("shape `{s}`: duplicate {key}"));
        }
    }
    match (m, n, k) {
        (Some(m), Some(n), Some(k)) => Ok((m, n, k)),
        _ => Err(format!("shape `{s}`: M, N and K are all required")),
    }
}

fn put_ptr(b: &mut Vec<u8>, p: u64) {
    b.extend_from_slice(&p.to_le_bytes());
}

fn put_i32(b: &mut Vec<u8>, v: i32) {
    b.extend_from_slice(&v.to_le_bytes());
}

/// SET/ADD explicit kernarg: `A Xq Y` at 0/8/16, `M K N add` at 24/28/32/36.
fn pack_set_add(a: u64, xq: u64, y: u64, m: usize, k: usize, n: usize, add: bool) -> Vec<u8> {
    let mut b = Vec::with_capacity(40);
    for p in [a, xq, y] {
        put_ptr(&mut b, p);
    }
    for v in [m, k, n, usize::from(add)] {
        put_i32(&mut b, v as i32);
    }
    b
}

/// Gate/up explicit kernarg: `G U Xq H` at 0/8/16/24, `M K N` at 32/36/40.
fn pack_silu(g: u64, u: u64, xq: u64, h: u64, m: usize, k: usize, n: usize) -> Vec<u8> {
    let mut b = Vec::with_capacity(44);
    for p in [g, u, xq, h] {
        put_ptr(&mut b, p);
    }
    for v in [m, k, n] {
        put_i32(&mut b, v as i32);
    }
    b
}

#[derive(Clone)]
struct Strs {
    fold: String,
    tile: String,
    cacc: String,
    epi: String,
    alayout: String,
}

impl Strs {
    fn json(&self) -> Value {
        json!({"fold": self.fold, "tile": self.tile, "cacc": self.cacc,
               "epi": self.epi, "alayout": self.alayout})
    }
    fn spec(&self) -> Result<Spec> {
        Spec::from_knobs(
            &self.fold,
            &self.tile,
            &self.cacc,
            &self.epi,
            &self.alayout,
            Arch::Gfx1201,
        )
    }
}

struct Dev {
    w: hip_bridge::DeviceBuffer,
    w_up: Option<hip_bridge::DeviceBuffer>,
    xq_token: hip_bridge::DeviceBuffer,
    xq_slab: hip_bridge::DeviceBuffer,
    /// Indexed by [`arm_index`].
    y: [hip_bridge::DeviceBuffer; 2],
}

fn arm_index(arm: Arm) -> usize {
    match arm {
        Arm::Baseline => 0,
        Arm::Candidate => 1,
    }
}

pub struct Iu4 {
    m: usize,
    n: usize,
    k: usize,
    cand: Spec,
    cand_strs: Strs,
    /// Spec the baseline arm runs (shipped: tile 128x128x8 with the candidate's epi/act).
    base: Spec,
    base_strs: Strs,
    /// `false`: the shipped bundle; `true`: emitted from `base`.
    base_emitted: bool,
    rows: Vec<usize>,
    rows_desc: String,
    ref_fold: RefFold,
    host: Option<Host>,
    dev: Option<Dev>,
    reference: OnceCell<Vec<f32>>,
    /// `--rip`: the candidate arm is this source, not `iu4_gemm::emit`.
    rip: Option<RipSource>,
}

struct RipSource {
    path: String,
    text: String,
}

/// ABI of a kernel in an assembled object: everything the launcher and the
/// kernarg packer depend on (symbol, kernarg layout, workgroup size, LDS,
/// wave mode). Code, VGPR/SGPR counts and the like may differ.
fn abi_signature(elf: &[u8], symbol: &str) -> Result<String> {
    use peacemaker_ir::inst::Abi;
    let program = peacemaker_lift::lift_object(
        elf,
        peacemaker_lift::Options {
            frontend: peacemaker_ir::inst::Frontend::Builder,
        },
    )
    .map_err(dev_err)?
    .program;
    let kernel = program
        .kernels
        .iter()
        .find(|k| k.symbol.0 == symbol)
        .ok_or_else(|| format!("object has no kernel {symbol}"))?;
    let Abi::Hsa {
        descriptor,
        metadata,
    } = &kernel.abi
    else {
        return Err(format!("{symbol}: not an HSA kernel"));
    };
    let m = &metadata.parsed;
    Ok(format!(
        "symbol={} args={:?} kernarg_bytes={} kernarg_align={} lds_bytes={} max_flat_workgroup_size={} wavefront_size={} workgroup_processor_mode={}",
        m.symbol,
        m.args,
        descriptor.kernarg_size,
        m.kernarg_segment_align,
        descriptor.group_segment_fixed_size,
        m.max_flat_workgroup_size,
        m.wavefront_size,
        m.workgroup_processor_mode,
    ))
}

/// Compile `.rip` source for `spec`, refuse unless its ABI equals that of
/// `iu4_gemm::emit(spec)`, and return the linked ELF, the symbol and the
/// `candidate.rip` record fields (`variant`, `sha256`, `identical_to_generator`).
fn compile_rip(source: &str, spec: Spec) -> Result<(Vec<u8>, String, Value)> {
    let variant = spec.variant_name();
    let compiled = peacemaker_rip::compile(source, spec.arch, &variant)?;
    let generator = iu4_gemm::emit(spec)?;
    let symbol = spec.symbol();
    let elf = peacemaker_isa::native::assemble(&compiled.s_text, spec.arch)?;
    let gen_elf = peacemaker_isa::native::assemble(&generator.s_text, spec.arch)?;
    let got = abi_signature(&elf, &symbol)
        .map_err(|e| format!("compiled kernel does not expose {symbol}: {e}"))?;
    let want = abi_signature(&gen_elf, &symbol)?;
    if got != want {
        return Err(format!(
            "ABI mismatch with iu4_gemm::emit({variant}):\n  .rip:      {got}\n  generator: {want}"
        ));
    }
    let info = json!({
        "sha256": sha256_hex(source.as_bytes()),
        "variant": variant,
        "identical_to_generator": compiled.s_text == generator.s_text,
    });
    Ok((elf, symbol, info))
}

fn dev_err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

#[cfg(test)]
pub fn new(shape: &str, knobs: &mut Knobs) -> Result<Box<dyn Workload>> {
    new_with_rip(shape, knobs, None)
}

/// [`new`] with the candidate compiled from a `.rip` file (`--rip`).
pub fn new_with_rip(
    shape: &str,
    knobs: &mut Knobs,
    rip: Option<&Path>,
) -> Result<Box<dyn Workload>> {
    let rip = match rip {
        Some(p) => Some(RipSource {
            path: p.display().to_string(),
            text: std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?,
        }),
        None => None,
    };
    let (m, n, k) = parse_shape(shape)?;
    if m == 0 || n == 0 || k == 0 || k % 256 != 0 || m % 4 != 0 {
        return Err(format!(
            "shape M={m},N={n},K={k}: need M%4==0, K%256==0, N>=1"
        ));
    }
    let cand_strs = Strs {
        fold: knobs.take_or("fold", "k128"),
        tile: knobs.take_or("tile", "128x128x8"),
        cacc: knobs.take_or("cacc", "1"),
        epi: knobs.take_or("epi", "set"),
        alayout: knobs.take_or("alayout", "token"),
    };
    let cand = cand_strs.spec()?;
    if cand.epi == Epi::QkvzaGdn {
        return Err("epi qkvzagdn is not supported by pm_loop (120-byte fused GDN ABI)".into());
    }
    if cand.epi.is_silu() && m % 128 != 0 {
        return Err(format!(
            "epi {}: M={m} must be a multiple of 128 (production gate/up launcher)",
            cand.epi.name()
        ));
    }
    let mut base_strs = cand_strs.clone();
    let mut base_emitted = false;
    for (key, slot) in [
        ("base-fold", &mut base_strs.fold),
        ("base-tile", &mut base_strs.tile),
        ("base-cacc", &mut base_strs.cacc),
        ("base-alayout", &mut base_strs.alayout),
    ] {
        if let Some(v) = knobs.take(key) {
            *slot = v;
            base_emitted = true;
        }
    }
    let base = if base_emitted {
        base_strs.spec()?
    } else {
        // The shipped bundle: production tile, K128/cacc1, the candidate's layout and epilogue.
        base_strs = Strs {
            fold: "k128".into(),
            tile: "128x128x8".into(),
            cacc: "1".into(),
            ..cand_strs.clone()
        };
        Spec {
            act: cand.act,
            ..Spec::control(cand.epi)
        }
    };
    let ref_fold = match knobs.take_or("ref-fold", "fma").as_str() {
        "fma" => RefFold::Fma,
        "muladd" => RefFold::MulAdd,
        o => return Err(format!("--ref-fold {o}: expected fma|muladd")),
    };
    let macs = (m as u128) * (n as u128) * (k as u128);
    let mode = knobs.take_or(
        "ref-rows",
        if macs <= FULL_REF_MACS {
            "full"
        } else {
            "sample:192"
        },
    );
    let (rows, rows_desc) = if mode == "full" {
        ((0..m).collect(), "full".to_owned())
    } else if let Some(c) = mode.strip_prefix("sample:") {
        let c: usize = c.parse().map_err(|e| format!("--ref-rows {mode}: {e}"))?;
        (sample_rows(m, c.max(2)), format!("sample:{c}"))
    } else {
        return Err(format!("--ref-rows {mode}: expected full|sample:<n>"));
    };
    Ok(Box::new(Iu4 {
        m,
        n,
        k,
        cand,
        cand_strs,
        base,
        base_strs,
        base_emitted,
        rows,
        rows_desc,
        ref_fold,
        host: None,
        dev: None,
        reference: OnceCell::new(),
        rip,
    }))
}

fn emit_elf(spec: Spec) -> Result<(Vec<u8>, String)> {
    let e = iu4_gemm::emit(spec)?;
    Ok((
        peacemaker_isa::native::assemble(&e.s_text, spec.arch)?,
        spec.symbol(),
    ))
}

impl Iu4 {
    fn spec_of(&self, arm: Arm) -> Spec {
        match arm {
            Arm::Baseline => self.base,
            Arm::Candidate => self.cand,
        }
    }

    fn out_bytes(&self) -> usize {
        self.n
            * self.m
            * if self.cand.epi == Epi::GateUpSiluBf16 {
                2
            } else {
                4
            }
    }

    fn rip_object(&self, rip: &RipSource) -> Result<Object> {
        let (elf, symbol, mut info) = compile_rip(&rip.text, self.cand)?;
        info["path"] = json!(rip.path);
        Ok(Object {
            source: json!({
                "source": format!("peacemaker_rip::compile({}) + native::assemble", rip.path),
                "variant": self.cand.variant_name(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": self.cand_strs.json(), "rip": info}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn emit_object(&self, spec: Spec, knobs: Value) -> Result<Object> {
        let (elf, symbol) = emit_elf(spec)?;
        Ok(Object {
            source: json!({
                "source": "peacemaker_isa::kernels::iu4_gemm::emit + native::assemble (peacemaker-isa emit --co)",
                "variant": spec.variant_name(),
                "sha256": sha256_hex(&elf),
            }),
            desc: json!({"knobs": knobs}),
            exports: vec![symbol.clone()],
            launch_symbol: symbol,
            bytes: elf,
        })
    }

    fn shipped(&self) -> Result<Object> {
        let symbol = self.base.symbol();
        let (bytes, file, selection) = match self.base.act {
            ALayout::Token => (
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1,
                "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1.hxaco",
                "crates/rdna-compute/src/gemm.rs:917-926 g12_iu4_b1_image \
                 (kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1); set/add kernel choice \
                 gemm.rs:21707-21726 (gemm_mq4g256v2_mmq_prequant_iu4), silu gemm.rs:36483-36530 \
                 (gemm_gate_up_silu_mq4g256v2_iu4_gfx12)",
            ),
            ALayout::Slab => (
                rdna_compute::kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S,
                "kernels/gemm_mq4g256v2_residual_mmq_iu4_gfx12_b1s.hxaco",
                "crates/rdna-compute/src/gemm.rs:929-934 g12_iu4_b1s_image \
                 (kernels::GEMM_MQ4G256V2_RESIDUAL_MMQ_IU4_GFX12_B1S); set/add kernel choice \
                 gemm.rs:21707-21716 (slab arm), silu gemm.rs:36483-36500",
            ),
        };
        if !contains(bytes, symbol.as_bytes()) {
            return Err(format!(
                "symbol {symbol} is not in the shipped bundle {file}"
            ));
        }
        Ok(Object {
            bytes: bytes.to_vec(),
            launch_symbol: symbol.clone(),
            exports: vec![symbol],
            desc: json!({"artifact": file, "knobs": self.base_strs.json()}),
            source: json!({
                "source": selection,
                "artifact": file,
                "sha256": sha256_hex(bytes),
            }),
        })
    }

    /// Lift the candidate and, when the emulator accepts every instruction, run a
    /// SET/ADD GEMM (M=128, N=128, K=256) and compare against the CPU reference.
    fn run_emu(&self) -> Result<Value> {
        use peacemaker_ir::emu::{Emulator, Launch as ELaunch, Memory};
        let (elf, symbol) = emit_elf(self.cand)?;
        let lifted = peacemaker_lift::lift_object(
            &elf,
            peacemaker_lift::Options {
                frontend: peacemaker_ir::inst::Frontend::Builder,
            },
        )
        .map_err(|e| format!("lift: {e:?}"))?;
        // Reports the first opcode without semantics, if any.
        let emu = Emulator::new(&lifted.program, &symbol)?;
        if !matches!(self.cand.epi, Epi::Set | Epi::Add) {
            return Err(format!(
                "emulator accepts every {symbol} opcode but pm_loop drives only set/add emulation"
            ));
        }
        let (m, n, k) = (128usize, 128usize, 256usize);
        let add = self.cand.epi == Epi::Add;
        let host = Host::synth(m, n, k, false);
        let (kva, ava, xva, yva) = (
            0x1000_0000u64,
            0x2000_0000u64,
            0x3000_0000u64,
            0x4000_0000u64,
        );
        let xq = match self.cand.act {
            ALayout::Token => host.recs.clone(),
            ALayout::Slab => to_slab(&host.recs, n),
        };
        let mut kern = pack_set_add(ava, xva, yva, m, k, n, add);
        kern.resize(self.cand.kernargs().size as usize, 0);
        let y0: Vec<u8> = if add {
            host.y0.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            vec![0; n * m * 4]
        };
        let mut mem = Memory::default();
        mem.map("kernarg".into(), kva, kern)?;
        for (name, va, bytes) in [("a", ava, host.w.clone()), ("xq", xva, xq), ("y", yva, y0)] {
            let len = bytes.len() as u64;
            mem.map(name.into(), va, bytes)?;
            mem.map(format!("{name}_tail"), va + len, vec![0; 4096])?;
        }
        let tile = self.cand.tile;
        for y in 0..n.div_ceil(128) {
            for x in 0..m.div_ceil(tile.rows() as usize) {
                emu.run_workgroup(
                    &ELaunch {
                        kernarg_ptr: kva,
                        workgroup: [x as u32, y as u32, 0],
                        block: [tile.threads(), 1, 1],
                        dynamic_lds: tile.layout().launch,
                    },
                    &mut mem,
                )?;
            }
        }
        let got = mem.read(yva, n * m * 4)?;
        let rows: Vec<usize> = (0..m).collect();
        let want = reference(&host, m, n, k, add, &rows, self.ref_fold);
        let mismatches = (0..m)
            .flat_map(|r| (0..n).map(move |c| (r, c)))
            .filter(|&(r, c)| {
                let o = (c * m + r) * 4;
                got[o..o + 4] != want[r * n + c].to_le_bytes()
            })
            .count();
        Ok(json!({
            "ran": true,
            "shape": {"M": m, "N": n, "K": k},
            "symbol": symbol,
            "reference": "cpu_iu4_symfold_rn",
            "mismatching_elements": mismatches,
            "bit_exact": mismatches == 0,
        }))
    }
}

fn max_diff(a: &[u8], b: &[u8], width: usize) -> f64 {
    let val = |c: &[u8]| -> f32 {
        if width == 2 {
            f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
        } else {
            f32::from_le_bytes(c.try_into().unwrap())
        }
    };
    a.chunks_exact(width)
        .zip(b.chunks_exact(width))
        .map(|(x, y)| elem_diff(val(x), val(y)))
        .fold(0f64, f64::max)
}

fn elem_diff(x: f32, y: f32) -> f64 {
    if x.is_nan() != y.is_nan() {
        f64::INFINITY
    } else if x.is_nan() || x == y {
        0.0
    } else {
        (f64::from(x) - f64::from(y)).abs()
    }
}

impl Workload for Iu4 {
    fn kernel(&self) -> &'static str {
        "iu4_gemm"
    }
    fn arch(&self) -> &'static str {
        "gfx1201"
    }
    fn shape_json(&self) -> Value {
        json!({"M": self.m, "N": self.n, "K": self.k,
               "launch_rows": if self.cand.epi.is_silu() { 2 * self.m } else { self.m }})
    }
    fn candidate_knobs(&self) -> Value {
        let mut v = self.cand_strs.json();
        v["variant"] = json!(self.cand.variant_name());
        v
    }

    fn candidate(&self) -> Result<Object> {
        if let Some(rip) = &self.rip {
            return self.rip_object(rip);
        }
        self.emit_object(self.cand, self.cand_strs.json())
    }

    fn baseline(&self) -> Result<Object> {
        if self.base_emitted {
            self.emit_object(self.base, self.base_strs.json())
        } else {
            self.shipped()
        }
    }

    fn setup(&mut self, gpu: &Gpu) -> Result<()> {
        let silu = self.cand.epi.is_silu();
        let host = Host::synth(self.m, self.n, self.k, silu);
        let up = |b: &[u8]| -> Result<hip_bridge::DeviceBuffer> {
            let d = gpu.hip.malloc(b.len().max(16)).map_err(dev_err)?;
            if !b.is_empty() {
                gpu.hip.memcpy_htod(&d, b).map_err(dev_err)?;
            }
            Ok(d)
        };
        let out = self.out_bytes().max(16);
        let dev = Dev {
            w: up(&host.w)?,
            w_up: if silu { Some(up(&host.w_up)?) } else { None },
            xq_token: up(&host.recs)?,
            xq_slab: up(&to_slab(&host.recs, self.n))?,
            y: [
                gpu.hip.malloc(out).map_err(dev_err)?,
                gpu.hip.malloc(out).map_err(dev_err)?,
            ],
        };
        self.host = Some(host);
        self.dev = Some(dev);
        for arm in [Arm::Baseline, Arm::Candidate] {
            self.reset(gpu, arm)?;
        }
        Ok(())
    }

    fn launch(&self, arm: Arm) -> Launch {
        let dev = self.dev.as_ref().expect("launch before setup");
        let spec = self.spec_of(arm);
        let ptr = |b: &hip_bridge::DeviceBuffer| b.as_ptr() as u64;
        let xq = ptr(match spec.act {
            ALayout::Token => &dev.xq_token,
            ALayout::Slab => &dev.xq_slab,
        });
        let y = ptr(&dev.y[arm_index(arm)]);
        let rows = if spec.epi.is_silu() {
            2 * self.m
        } else {
            self.m
        };
        let kernarg = match &dev.w_up {
            Some(up) => pack_silu(ptr(&dev.w), ptr(up), xq, y, self.m, self.k, self.n),
            None => pack_set_add(
                ptr(&dev.w),
                xq,
                y,
                self.m,
                self.k,
                self.n,
                spec.epi == Epi::Add,
            ),
        };
        Launch {
            kernarg,
            grid: [
                rows.div_ceil(spec.tile.rows() as usize) as u32,
                self.n.div_ceil(128) as u32,
                1,
            ],
            block: [spec.tile.threads(), 1, 1],
            lds: spec.tile.layout().launch,
        }
    }

    fn reset(&self, gpu: &Gpu, arm: Arm) -> Result<()> {
        let dev = self.dev.as_ref().ok_or("reset before setup")?;
        let y = &dev.y[arm_index(arm)];
        if self.cand.epi == Epi::Add {
            let host = self.host.as_ref().ok_or("reset before setup")?;
            let bytes: Vec<u8> = host.y0.iter().flat_map(|v| v.to_le_bytes()).collect();
            gpu.hip.memcpy_htod(y, &bytes).map_err(dev_err)
        } else {
            gpu.hip.memset(y, 0, y.size()).map_err(dev_err)
        }
    }

    fn output(&self, gpu: &Gpu, arm: Arm) -> Result<Vec<u8>> {
        let dev = self.dev.as_ref().ok_or("output before setup")?;
        // Device buffers are padded to 16 bytes; the payload is the leading `out_bytes`.
        let y = &dev.y[arm_index(arm)];
        let mut v = vec![0u8; y.size()];
        gpu.hip.memcpy_dtoh(&mut v, y).map_err(dev_err)?;
        v.truncate(self.out_bytes());
        Ok(v)
    }

    fn check(&self, out: &[u8]) -> Result<RefReport> {
        if self.cand.epi.is_silu() {
            return Err(format!(
                "no CPU reference for epi {} (v_exp-approximated SiLU region not reproduced); \
                 correctness is candidate-vs-baseline only",
                self.cand.epi.name()
            ));
        }
        if out.len() != self.out_bytes() {
            return Err(format!(
                "output is {} bytes, expected {}",
                out.len(),
                self.out_bytes()
            ));
        }
        let host = self.host.as_ref().ok_or("check before setup")?;
        let add = self.cand.epi == Epi::Add;
        let want = self.reference.get_or_init(|| {
            reference(host, self.m, self.n, self.k, add, &self.rows, self.ref_fold)
        });
        let (mut mismatches, mut worst) = (0usize, 0f64);
        for (ri, &r) in self.rows.iter().enumerate() {
            for col in 0..self.n {
                let o = (col * self.m + r) * 4;
                let got = f32::from_le_bytes(out[o..o + 4].try_into().unwrap());
                let exp = want[ri * self.n + col];
                mismatches += usize::from(got.to_bits() != exp.to_bits());
                worst = worst.max(elem_diff(got, exp));
            }
        }
        Ok(RefReport {
            reference: format!(
                "cpu_iu4_symfold_rn(rows={},fold={},mismatching_elements={mismatches})",
                self.rows_desc,
                match self.ref_fold {
                    RefFold::Fma => "fma",
                    RefFold::MulAdd => "muladd",
                }
            ),
            max_abs_err: worst,
            tolerance: 0.0,
            pass: mismatches == 0,
        })
    }

    fn max_abs_diff(&self, a: &[u8], b: &[u8]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(format!("length mismatch {} vs {}", a.len(), b.len()));
        }
        let width = if self.cand.epi == Epi::GateUpSiluBf16 {
            2
        } else {
            4
        };
        Ok(max_diff(a, b, width))
    }

    fn emu(&self) -> Value {
        match self.run_emu() {
            Ok(v) => v,
            Err(e) => Value::String(format!("refused ({e})")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knobs(kv: &[(&str, &str)]) -> Knobs {
        Knobs(kv.iter().map(|(k, v)| ((*k).into(), (*v).into())).collect())
    }

    #[test]
    fn shape_grammar() {
        assert_eq!(
            parse_shape("M=4096,N=512,K=5120").unwrap(),
            (4096, 512, 5120)
        );
        assert_eq!(parse_shape("K=256,N=1,M=4").unwrap(), (4, 1, 256));
        assert!(parse_shape("M=4,N=1").is_err());
        assert!(parse_shape("M=4,N=1,K=256,K=256").is_err());
        assert!(parse_shape("M=4,N=1,Q=3").is_err());
        assert!(new("M=4,N=1,K=128", &mut knobs(&[])).is_err());
        assert!(new("M=6,N=1,K=256", &mut knobs(&[])).is_err());
        assert!(new("M=64,N=1,K=256", &mut knobs(&[("epi", "silu")])).is_err());
        assert!(new("M=128,N=1,K=256", &mut knobs(&[("epi", "qkvzagdn")])).is_err());
    }

    const TWIN: &str = include_str!("../../../peacemaker-isa/src/kernels/iu4_gemm.rip");

    fn default_spec() -> Spec {
        Spec::from_knobs("k128", "128x128x8", "1", "set", "token", Arch::Gfx1201).unwrap()
    }

    #[test]
    fn rip_twin_is_identical_to_generator() {
        let (_, symbol, info) = compile_rip(TWIN, default_spec()).unwrap();
        assert_eq!(symbol, default_spec().symbol());
        assert_eq!(info["identical_to_generator"], true);
        assert_eq!(info["variant"], default_spec().variant_name());
        assert_eq!(info["sha256"], sha256_hex(TWIN.as_bytes()));
    }

    #[test]
    fn rip_edit_passes_abi_and_reports_difference() {
        let edited = TWIN.replacen(
            "    op(wg.isa(), \"s_setprio 1\", vec![], vec![])?;\n",
            "",
            1,
        );
        assert_ne!(edited, TWIN, "the s_setprio 1 line must exist in the twin");
        let (_, _, info) = compile_rip(&edited, default_spec()).unwrap();
        assert_eq!(info["identical_to_generator"], false);
    }

    #[test]
    fn rip_symbol_change_is_refused() {
        // The twin derives the kernel (and so the symbol) from the variant it
        // parses: make `set` build the `add` kernel.
        let renamed = TWIN.replacen("\"set\" => Epi::Set,", "\"set\" => Epi::Add,", 1);
        assert_ne!(renamed, TWIN, "the epilogue arm must exist in the twin");
        let spec = default_spec();
        let e = compile_rip(&renamed, spec).unwrap_err();
        assert!(
            e.contains("does not expose") || e.contains("ABI mismatch"),
            "{e}"
        );
    }

    #[test]
    fn kernarg_layout() {
        let b = pack_set_add(0x11, 0x22, 0x33, 7, 256, 5, true);
        assert_eq!(b.len(), 40);
        assert_eq!(u64::from_le_bytes(b[0..8].try_into().unwrap()), 0x11);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 0x22);
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 0x33);
        for (off, v) in [(24, 7), (28, 256), (32, 5), (36, 1)] {
            assert_eq!(i32::from_le_bytes(b[off..off + 4].try_into().unwrap()), v);
        }
        let s = pack_silu(1, 2, 3, 4, 128, 256, 9);
        assert_eq!(s.len(), 44);
        assert_eq!(u64::from_le_bytes(s[24..32].try_into().unwrap()), 4);
        for (off, v) in [(32, 128), (36, 256), (40, 9)] {
            assert_eq!(i32::from_le_bytes(s[off..off + 4].try_into().unwrap()), v);
        }
    }

    #[test]
    fn knob_to_spec_and_symbol() {
        let w = new(
            "M=128,N=64,K=256",
            &mut knobs(&[("epi", "add"), ("alayout", "slab")]),
        )
        .unwrap();
        assert_eq!(w.kernel(), "iu4_gemm");
        assert_eq!(
            w.candidate_knobs()["variant"],
            "k128-128x128x8-cacc1-slab-add"
        );
        let spec =
            Spec::from_knobs("k128", "128x128x8", "1", "add", "slab", Arch::Gfx1201).unwrap();
        assert_eq!(
            spec.symbol(),
            "gemm_mq4g256v2_residual_mmq_iu4_full_add_b1s"
        );
        let t = Spec::from_knobs("k128", "256x128x16", "1", "set", "token", Arch::Gfx1201).unwrap();
        assert_eq!(
            t.symbol(),
            "gemm_mq4g256v2_residual_mmq_iu4_full_set_b1t256"
        );
        assert_eq!(t.tile.layout().launch, 30720);
        assert_eq!(t.tile.threads(), 512);
    }

    #[test]
    fn closed_axes_surface_the_emit_error() {
        let w = new("M=128,N=64,K=256", &mut knobs(&[("fold", "k256shared")])).unwrap();
        assert!(w.candidate().unwrap_err().contains("closed"));
        let w = new("M=128,N=64,K=256", &mut knobs(&[("cacc", "2")])).unwrap();
        assert!(w.candidate().unwrap_err().contains("Cacc::Two"));
    }

    #[test]
    fn base_knobs_select_emitted_baseline() {
        let mut k = knobs(&[("tile", "256x128x16"), ("base-tile", "128x128x8")]);
        let w = new("M=128,N=64,K=256", &mut k).unwrap();
        assert!(k.0.is_empty());
        assert_eq!(w.candidate_knobs()["tile"], "256x128x16");
    }

    #[test]
    fn sampled_rows_cover_edges() {
        let r = sample_rows(1000, 192);
        assert!(r.starts_with(&[0, 1, 2]));
        assert_eq!(*r.last().unwrap(), 999);
        assert!(r.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(sample_rows(10, 4), (0..10).collect::<Vec<_>>());
    }

    /// Hand-checkable: every weight nibble 9 (signed +1), every x code 1, so
    /// dot = 128 per K128 block; sc = 1.0, d = 0.5 -> t1 = 0.5, term = 64, two blocks.
    #[test]
    fn reference_hand_case() {
        let (m, n, k) = (1usize, 1usize, 256usize);
        let mut w = vec![0u8; GROUP_BYTES];
        for h in 0..2 {
            w[4 * h..4 * h + 2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        }
        w[8..].fill(0x99);
        let mut recs = Vec::new();
        for _ in 0..2 {
            recs.extend_from_slice(&0.5f32.to_le_bytes());
            recs.extend_from_slice(&128i32.to_le_bytes());
            recs.extend_from_slice(&[0x11; 64]);
        }
        let host = Host {
            w,
            w_up: vec![],
            recs,
            y0: vec![1.5],
        };
        for fold in [RefFold::Fma, RefFold::MulAdd] {
            assert_eq!(reference(&host, m, n, k, false, &[0], fold), vec![128.0]);
            assert_eq!(reference(&host, m, n, k, true, &[0], fold), vec![129.5]);
        }
    }

    #[test]
    fn negative_codes_and_xor_fold() {
        // weight nibble 0 -> signed -8, x code -8 (nibble 0x8): dot = 128 * 64 = 8192 per block, d = 1.
        let mut w = vec![0u8; GROUP_BYTES];
        for h in 0..2 {
            w[4 * h..4 * h + 2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        }
        let mut recs = Vec::new();
        for _ in 0..2 {
            recs.extend_from_slice(&1.0f32.to_le_bytes());
            recs.extend_from_slice(&(-1024i32).to_le_bytes());
            recs.extend_from_slice(&[0x88; 64]);
        }
        let host = Host {
            w,
            w_up: vec![],
            recs,
            y0: vec![0.0],
        };
        assert_eq!(
            reference(&host, 1, 1, 256, false, &[0], RefFold::Fma),
            vec![2.0 * 8192.0]
        );
    }

    #[test]
    fn slab_planes_hold_the_same_bytes() {
        let n = 3;
        let host = Host::synth(4, n, 256, false);
        let slab = to_slab(&host.recs, n);
        assert_eq!(slab.len(), host.recs.len());
        let (kb, col) = (1, 2);
        let rec = &host.recs[(kb * n + col) * REC..][..REC];
        let base = kb * REC * n;
        assert_eq!(&slab[base + 4 * col..][..4], &rec[0..4]);
        assert_eq!(&slab[base + 4 * n + 4 * col..][..4], &rec[4..8]);
        assert_eq!(&slab[base + 8 * n + 32 * col..][..32], &rec[8..40]);
        assert_eq!(&slab[base + 40 * n + 32 * col..][..32], &rec[40..72]);
    }

    #[test]
    fn synthetic_inputs_are_deterministic_and_diff_is_nan_aware() {
        let a = Host::synth(8, 4, 256, false);
        let b = Host::synth(8, 4, 256, false);
        assert_eq!(a.w, b.w);
        assert_eq!(a.recs, b.recs);
        assert_eq!(
            max_diff(&1.0f32.to_le_bytes(), &f32::NAN.to_le_bytes(), 4),
            f64::INFINITY
        );
        assert_eq!(
            max_diff(&1.0f32.to_le_bytes(), &3.0f32.to_le_bytes(), 4),
            2.0
        );
    }
}
