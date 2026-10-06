// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! Developer-only raw-ABI oracle, screen and timing of the live F32 QSA
//! selector pair (`indexed_attention_select_scores_rows16_f32` +
//! `indexed_attention_select_from_scores`) on real `qsa-source-v1` snapshots.
//!
//! `qsa_select_pm_check SNAPSHOT_DIR... --out OUT.jsonl [--time N]
//!     [--score hipcc|pm] [--select hipcc|pm] [--pm-image ABS_FILE]
//!     [--cpu-rows STRIDE] [--no-screen] [--tied]`
//!
//! Every snapshot (pre-prologue dumps skipped) is replayed with the
//! production grouping (64 MiB score scratch, rows16, one select WG per row):
//! - baseline hipcc/hipcc: ordered `selected.i32` and the final-row mirror
//!   must equal the snapshot bytes (else the capture/ABI/route is wrong);
//! - CPU: every `--cpu-rows`-th row (default 64) plus the last row is
//!   rescored with the per-head serial F32 FMA chain and bit-compared with the
//!   hipcc scores; every row's stable (score desc, block asc) CPU selection
//!   over the hipcc scores is compared with the hipcc selection;
//! - screen: the adaptive radix prefix replay (stop when above + boundary
//!   bucket <= 1024) over every row's hipcc scores;
//! - WMMA diagnostic (CPU emulation of F16/BF16 operand rounding with the
//!   serial F32 chain kept: a lower bound on a WMMA port's differences);
//! - with `--score pm` / `--select pm`, the experimental image's symbols
//!   replace the corresponding hipcc launch: valid score bits (and untouched
//!   poison outside them) and the ordered selection/mirror must be identical;
//! - `--time N`: N repetitions of the whole call, per-kernel event time.
//! `--tied` adds the synthetic all-tied rows (16384, 32768 and 65536 blocks)
//! for the PM selector only, against the CPU stable-order oracle; the
//! pre-fix hipcc selector never runs on them.

use hip_bridge::{Function, KernargBlob, Module};
use rayon::prelude::*;
use rdna_compute::{Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};

type Result<T, E = String> = std::result::Result<T, E>;
fn err(e: impl std::fmt::Debug) -> String { format!("{e:?}") }
fn sha256(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }

const TENSOR_OPS_SRC: &str = concat!(
    include_str!("../../../kernels/src/mq_fwht256.h"),
    include_str!("../../../kernels/src/tensor_ops.hip")
);
const HIPCC_SCORE: &str = "indexed_attention_select_scores_rows16_f32";
const HIPCC_SELECT: &str = "indexed_attention_select_from_scores";
const PM_SCORE: &str = "indexed_attention_select_scores_rows16_f32_pm_gfx1151";
const PM_SELECT: &str = "indexed_attention_select_from_scores_pm_gfx1151";
const SCORE_SCRATCH_BYTES: usize = 64 << 20;
const POISON: u8 = 0xa5;

fn snapshots(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(dir).map_err(err)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n == "pre-prologue") { continue }
            snapshots(&p, out)?;
        } else if p.file_name().is_some_and(|n| n == "snapshot.json") {
            out.push(p);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Impl { Hipcc, Pm }

struct Pm { _module: Module, score: Option<Function>, select: Option<Function> }

/// One original select call's geometry.
#[derive(Clone, Copy)]
struct Call { rows: usize, stride: usize, block_count: usize, dim: usize, compress: usize, position_start: usize, budget: usize, capacity: usize }

impl Call {
    fn group(&self) -> usize { (SCORE_SCRATCH_BYTES / (self.block_count * 4) / 16 * 16).max(16).min(self.rows) }
    fn row_blocks(&self, row: usize) -> usize { ((self.position_start + row + 1) / self.compress).min(self.block_count) }
}

struct Bufs { query: GpuTensor, pooled: GpuTensor, scores: GpuTensor, selected: GpuTensor, mirror: GpuTensor }

fn score_blob(c: &Call, b: &Bufs, g0: usize, n: usize) -> KernargBlob {
    let mut a = KernargBlob::new();
    a.push_ptr(unsafe { (b.query.buf.as_ptr() as *const f32).add(g0 * c.stride) } as *const _);
    a.push_ptr(b.pooled.buf.as_ptr());
    a.push_ptr(b.scores.buf.as_ptr());
    a.push_i32(n as i32);
    a.push_i32(c.stride as i32);
    a.push_i32(c.block_count as i32);
    a.push_i32(c.dim as i32);
    a.push_i32(c.compress as i32);
    a.push_i32((c.position_start + g0) as i32);
    a.push_i32(c.block_count as i32);
    a.pad_to(16);
    a
}

fn select_blob(c: &Call, b: &Bufs, g0: usize, n: usize, last: bool) -> KernargBlob {
    let mut a = KernargBlob::new();
    a.push_ptr(b.scores.buf.as_ptr());
    a.push_i32(c.block_count as i32);
    a.push_ptr(unsafe { (b.selected.buf.as_ptr() as *const i32).add(g0 * c.capacity) } as *const _);
    a.push_i32(n as i32);
    a.push_i32(c.block_count as i32);
    a.push_i32(c.budget as i32);
    a.push_i32(c.compress as i32);
    a.push_i32((c.position_start + g0) as i32);
    a.push_i32(c.capacity as i32);
    a.push_ptr(if last { b.mirror.buf.as_ptr() } else { std::ptr::null_mut() });
    a.pad_to(16);
    a
}

fn launch(gpu: &mut Gpu, pm: &Option<Pm>, which: Impl, score: bool, grid: [u32; 3], blob: &mut KernargBlob) -> Result<()> {
    match which {
        Impl::Hipcc => gpu
            .launch_kernel_blob(if score { HIPCC_SCORE } else { HIPCC_SELECT }, grid, [256, 1, 1], 0, blob.as_mut_slice())
            .map_err(err),
        Impl::Pm => {
            let pm = pm.as_ref().ok_or("--pm-image required for a pm arm")?;
            let f = if score { pm.score.as_ref() } else { pm.select.as_ref() }.ok_or("pm image lacks the symbol")?;
            unsafe { gpu.hip.launch_kernel_blob(f, grid, [256, 1, 1], 0, None, blob.as_mut_slice()) }.map_err(err)
        }
    }
}

struct Timing { score_ms: f64, select_ms: f64 }

/// The production pair over all groups. `keep` downloads each group's
/// scratch after its score launch (rows x block_count, raw bytes).
fn run_call(gpu: &mut Gpu, pm: &Option<Pm>, c: &Call, b: &Bufs, arms: (Impl, Impl), mut keep: Option<&mut Vec<u8>>,
            events: Option<&[hip_bridge::Event; 3]>) -> Result<Timing> {
    let group = c.group();
    let (mut score_ms, mut select_ms) = (0.0f64, 0.0f64);
    let mut g0 = 0;
    while g0 < c.rows {
        let n = group.min(c.rows - g0);
        if keep.is_some() {
            gpu.hip.memset(&b.scores.buf, POISON as i32, b.scores.byte_size()).map_err(err)?;
        }
        let mut blob = score_blob(c, b, g0, n);
        if let Some(e) = events { gpu.hip.event_record(&e[0], None).map_err(err)?; }
        launch(gpu, pm, arms.0, true, [c.block_count.div_ceil(256) as u32, n.div_ceil(16) as u32, 1], &mut blob)?;
        if let Some(e) = events { gpu.hip.event_record(&e[1], None).map_err(err)?; }
        if let Some(k) = keep.as_deref_mut() {
            gpu.hip.device_synchronize().map_err(err)?;
            let mut bytes = vec![0u8; n * c.block_count * 4];
            let len = bytes.len();
            gpu.hip.memcpy_dtoh(&mut bytes, &b.scores.buf.byte_view(0, len)).map_err(err)?;
            k.extend_from_slice(&bytes);
        }
        let mut blob = select_blob(c, b, g0, n, g0 + n == c.rows);
        launch(gpu, pm, arms.1, false, [n as u32, 1, 1], &mut blob)?;
        if let Some(e) = events {
            gpu.hip.event_record(&e[2], None).map_err(err)?;
            gpu.hip.event_synchronize(&e[2]).map_err(err)?;
            score_ms += gpu.hip.event_elapsed_ms(&e[0], &e[1]).map_err(err)? as f64;
            select_ms += gpu.hip.event_elapsed_ms(&e[1], &e[2]).map_err(err)? as f64;
        }
        g0 += n;
    }
    gpu.hip.device_synchronize().map_err(err)?;
    Ok(Timing { score_ms, select_ms })
}

fn download(gpu: &Gpu, t: &GpuTensor) -> Result<Vec<u8>> {
    let mut bytes = vec![0u8; t.byte_size()];
    gpu.hip.memcpy_dtoh(&mut bytes, &t.buf).map_err(err)?;
    Ok(bytes)
}

fn poison_outputs(gpu: &Gpu, b: &Bufs) -> Result<()> {
    gpu.hip.memset(&b.selected.buf, POISON as i32, b.selected.byte_size()).map_err(err)?;
    gpu.hip.memset(&b.mirror.buf, POISON as i32, b.mirror.byte_size()).map_err(err)
}

// ---------------------------------------------------------------- CPU side

fn f16_round(x: f32) -> f32 {
    if !x.is_finite() { return x; }
    let a = x.abs();
    if a < 6.103_515_6e-5 {
        return (x * 16_777_216.0).round_ties_even() / 16_777_216.0;
    }
    let b = x.to_bits();
    let r = f32::from_bits((b + 0xfff + ((b >> 13) & 1)) & !0x1fff);
    if r.abs() > 65504.0 { f32::INFINITY.copysign(x) } else { r }
}

fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() { return x; }
    let b = x.to_bits();
    f32::from_bits((b + 0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

/// Per head serial d-order F32 FMA chain from +0, `max(dot / sqrt(dim), 0)`
/// per head, heads added in order: the incumbent's source arithmetic.
/// Also reports per-head pre-ReLU signs (bit h set if dot > 0).
fn cpu_score(q: &[f32], key: &[f32], dim: usize) -> (f32, u8) {
    let scale = (dim as f32).sqrt();
    let mut score = 0.0f32;
    let mut signs = 0u8;
    for h in 0..4 {
        let qh = &q[h * dim..(h + 1) * dim];
        let mut dot = 0.0f32;
        for d in 0..dim { dot = qh[d].mul_add(key[d], dot); }
        let s = dot / scale;
        if s > 0.0 { signs |= 1 << h; }
        score += if s > 0.0 { s } else { 0.0 };
    }
    (score, signs)
}

fn rank(keys: &[u32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| keys[b as usize].cmp(&keys[a as usize]).then(a.cmp(&b)));
    order
}

fn selection_row(order: &[u32], c: &Call, row: usize) -> Vec<i32> {
    let visible = c.position_start + row + 1;
    let blocks = c.row_blocks(row);
    let chosen = c.budget.min(blocks);
    let mut out = vec![-1i32; c.capacity];
    for (slot, &block) in order.iter().take(chosen).enumerate() {
        for o in 0..c.compress {
            let (i, t) = (slot * c.compress + o, block as usize * c.compress + o);
            if i < c.capacity && t < visible { out[i] = t as i32; }
        }
    }
    let mut i = chosen * c.compress;
    for t in blocks * c.compress..visible {
        if i >= c.capacity { break; }
        out[i] = t as i32;
        i += 1;
    }
    out
}

#[derive(Default, Clone)]
struct Screen {
    rows: usize,
    finish: [usize; 5],          // index 0: chosen == visible blocks (no refinement)
    all_equal: usize,
    pad512: usize,
    pad1024: usize,
    over1024_after: [Vec<usize>; 4], // candidates (above + boundary) after byte b
    boundary_after: [Vec<usize>; 4],
}

/// Adaptive exact prefix selection replay over one row's keys.
fn screen_row(keys: &[u32], chosen: usize, s: &mut Screen) {
    s.rows += 1;
    let (mn, mx) = keys.iter().fold((u32::MAX, 0u32), |(a, b), &k| (a.min(k), b.max(k)));
    if mn == mx { s.all_equal += 1; }
    if chosen == keys.len() {
        s.finish[0] += 1;
        s.pad512 += 1;
        return;
    }
    let (mut prefix, mut need, mut above) = (0u32, chosen, 0usize);
    let mut done = false;
    for pass in 0..4 {
        let shift = 24 - 8 * pass;
        let high = if pass == 0 { 0 } else { !0u32 << (shift + 8) };
        let mut hist = [0usize; 256];
        for &k in keys { if k & high == prefix { hist[((k >> shift) & 255) as usize] += 1; } }
        let mut acc = 0usize;
        let mut digit = 255usize;
        loop {
            if acc + hist[digit] >= need { break; }
            acc += hist[digit];
            digit -= 1;
        }
        above += acc;
        need -= acc;
        prefix |= (digit as u32) << shift;
        let boundary = hist[digit];
        let cand = above + boundary;
        s.over1024_after[pass].push(cand);
        s.boundary_after[pass].push(boundary);
        if !done && (cand <= 1024 || pass == 3) {
            done = true;
            s.finish[pass + 1] += 1;
            let final_cand = if pass == 3 { above + need } else { cand };
            if final_cand <= 512 { s.pad512 += 1 } else { s.pad1024 += 1 }
        }
    }
}

fn quant(v: &mut [usize]) -> Value {
    if v.is_empty() { return Value::Null; }
    v.sort_unstable();
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    json!({"n":v.len(),"min":v[0],"p50":q(0.5),"p90":q(0.9),"p99":q(0.99),"max":v[v.len()-1]})
}

fn screen_json(s: &mut Screen) -> Value {
    let rows = s.rows.max(1) as f64;
    let by_byte2 = (s.finish[0] + s.finish[1] + s.finish[2]) as f64 / rows;
    let cands: Vec<Value> = s.over1024_after.iter_mut().map(|v| quant(v)).collect();
    let bounds: Vec<Value> = s.boundary_after.iter_mut().map(|v| quant(v)).collect();
    json!({"rows":s.rows,"finish_no_refine":s.finish[0],"finish_after_byte":[s.finish[1],s.finish[2],s.finish[3],s.finish[4]],
        "frac_finish_by_byte":[ (s.finish[0]+s.finish[1]) as f64/rows, by_byte2,
            (s.finish[0]+s.finish[1]+s.finish[2]+s.finish[3]) as f64/rows, 1.0],
        "all_equal":s.all_equal,"pad512":s.pad512,"pad1024":s.pad1024,
        "candidates_after_byte":cands,"boundary_after_byte":bounds,"by_byte2_frac":by_byte2})
}

#[derive(Default)]
struct Diag { rows: usize, score_bits_differ: u64, scores: u64, relu_sign_changes: u64, ordered_rows_differ: usize, ordered_elems_differ: u64, set_rows_differ: usize, set_elems_differ: u64 }

// ---------------------------------------------------------------- main

fn cpu_checks(c: &Call, query: &[f32], pooled: &[f32], scores: &[u32], hipcc_sel: &[i32], cpu_stride: usize, screen: bool)
    -> (Value, Screen) {
    let bc = c.block_count;
    // Stable CPU selection over the hipcc scores, every row.
    let sel_mismatch: Vec<usize> = (0..c.rows).into_par_iter().filter(|&r| {
        let keys = &scores[r * bc..r * bc + c.row_blocks(r)];
        selection_row(&rank(keys), c, r) != hipcc_sel[r * c.capacity..(r + 1) * c.capacity]
    }).collect();
    // Exact rescoring and WMMA operand-rounding diagnostic on sampled rows.
    let sampled: Vec<usize> = (0..c.rows).filter(|r| r % cpu_stride == 0 || *r == c.rows - 1).collect();
    let per_row: Vec<(u64, u64, [Diag; 2], f64)> = sampled.par_iter().map(|&r| {
        let q = &query[r * c.stride..r * c.stride + 4 * c.dim];
        let blocks = c.row_blocks(r);
        let qs = [q.iter().map(|&x| f16_round(x)).collect::<Vec<_>>(), q.iter().map(|&x| bf16_round(x)).collect::<Vec<_>>()];
        let mut exact = Vec::with_capacity(blocks);
        let mut signs = Vec::with_capacity(blocks);
        let mut alt = [Vec::with_capacity(blocks), Vec::with_capacity(blocks)];
        let mut alt_signs = [Vec::with_capacity(blocks), Vec::with_capacity(blocks)];
        let mut differ = 0u64;
        for b in 0..blocks {
            let key = &pooled[b * c.dim..(b + 1) * c.dim];
            let (s, sg) = cpu_score(q, key, c.dim);
            if s.to_bits() != scores[r * bc + b] { differ += 1; }
            exact.push(s.to_bits());
            signs.push(sg);
            let k16: Vec<f32> = key.iter().map(|&x| f16_round(x)).collect();
            for (v, (qv, kv)) in [(&qs[0], &k16), (&qs[1], &key.to_vec())].into_iter().enumerate() {
                let (s, sg) = cpu_score(qv, kv, c.dim);
                alt[v].push(s.to_bits());
                alt_signs[v].push(sg);
            }
        }
        let chosen = c.budget.min(blocks);
        let order = rank(&exact);
        let margin = if blocks > chosen && chosen > 0 {
            let (a, b) = (f32::from_bits(exact[order[chosen - 1] as usize]), f32::from_bits(exact[order[chosen] as usize]));
            ((a - b) / a.max(f32::MIN_POSITIVE)) as f64
        } else { f64::NAN };
        let mut diags = [Diag::default(), Diag::default()];
        for v in 0..2 {
            let d = &mut diags[v];
            d.rows = 1;
            d.scores = blocks as u64;
            d.score_bits_differ = exact.iter().zip(&alt[v]).filter(|(a, b)| a != b).count() as u64;
            d.relu_sign_changes = signs.iter().zip(&alt_signs[v]).map(|(a, b)| (a ^ b).count_ones() as u64).sum();
            let ao = rank(&alt[v]);
            let e = order[..chosen].iter().zip(&ao[..chosen]).filter(|(a, b)| a != b).count() as u64;
            d.ordered_elems_differ = e;
            d.ordered_rows_differ = usize::from(e > 0);
            let es: std::collections::BTreeSet<u32> = order[..chosen].iter().copied().collect();
            let miss = ao[..chosen].iter().filter(|x| !es.contains(x)).count() as u64;
            d.set_elems_differ = miss;
            d.set_rows_differ = usize::from(miss > 0);
        }
        (differ, blocks as u64, diags, margin)
    }).collect();
    let rescore_differ: u64 = per_row.iter().map(|x| x.0).sum();
    let rescored: u64 = per_row.iter().map(|x| x.1).sum();
    let mut margins: Vec<f64> = per_row.iter().map(|x| x.3).filter(|m| m.is_finite()).collect();
    margins.sort_by(f64::total_cmp);
    let mut wmma = Vec::new();
    for (v, name) in ["f16_operands", "bf16_operands"].iter().enumerate() {
        let mut t = Diag::default();
        for x in &per_row {
            let d = &x.2[v];
            t.rows += d.rows; t.scores += d.scores; t.score_bits_differ += d.score_bits_differ;
            t.relu_sign_changes += d.relu_sign_changes; t.ordered_rows_differ += d.ordered_rows_differ;
            t.ordered_elems_differ += d.ordered_elems_differ; t.set_rows_differ += d.set_rows_differ; t.set_elems_differ += d.set_elems_differ;
        }
        wmma.push(json!({"variant":name,"rows":t.rows,"scores":t.scores,"score_bits_differ":t.score_bits_differ,
            "relu_sign_changes":t.relu_sign_changes,"rows_ordered_selection_differ":t.ordered_rows_differ,
            "ordered_elements_differ":t.ordered_elems_differ,"rows_set_differ":t.set_rows_differ,"set_elements_differ":t.set_elems_differ}));
    }
    let mut s = Screen::default();
    if screen {
        let parts: Vec<Screen> = (0..c.rows).into_par_iter().fold(Screen::default, |mut s, r| {
            let blocks = c.row_blocks(r);
            let chosen = c.budget.min(blocks);
            if chosen > 0 { screen_row(&scores[r * bc..r * bc + blocks], chosen, &mut s); }
            s
        }).collect();
        for p in parts {
            s.rows += p.rows; s.all_equal += p.all_equal; s.pad512 += p.pad512; s.pad1024 += p.pad1024;
            for i in 0..5 { s.finish[i] += p.finish[i]; }
            for i in 0..4 { s.over1024_after[i].extend(&p.over1024_after[i]); s.boundary_after[i].extend(&p.boundary_after[i]); }
        }
    }
    let m = |p: f64| if margins.is_empty() { f64::NAN } else { margins[((margins.len() - 1) as f64 * p) as usize] };
    (json!({"cpu_selection_rows_differ":sel_mismatch.len(),"cpu_selection_first_differ":sel_mismatch.first(),
        "rescored_rows":sampled.len(),"rescored_scores":rescored,"rescore_bits_differ":rescore_differ,
        "kth_rel_margin":{"min":m(0.0),"p10":m(0.1),"p50":m(0.5)},
        "wmma_cpu_emulation":wmma,
        "wmma_note":"CPU emulation: F16/BF16 operand rounding with the incumbent serial F32 FMA chain kept; a real WMMA port also reassociates, so these are lower bounds"}), s)
}

fn main() -> Result<()> {
    if std::env::var_os("HIPFIRE_LOCK_DIR").is_none() { return Err("private HIPFIRE_LOCK_DIR required".into()) }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut dirs, mut out, mut timed, mut image) = (Vec::new(), None, 0usize, None::<PathBuf>);
    let (mut arms, mut cpu_stride, mut screen, mut tied) = ((Impl::Hipcc, Impl::Hipcc), 64usize, true, false);
    let parse_impl = |v: Option<String>| -> Result<Impl> {
        match v.as_deref() { Some("hipcc") => Ok(Impl::Hipcc), Some("pm") => Ok(Impl::Pm), _ => Err("arm must be hipcc|pm".into()) }
    };
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out = it.next(),
            "--time" => timed = it.next().ok_or("--time N")?.parse().map_err(err)?,
            "--pm-image" => image = Some(PathBuf::from(it.next().ok_or("--pm-image FILE")?)),
            "--score" => arms.0 = parse_impl(it.next())?,
            "--select" => arms.1 = parse_impl(it.next())?,
            "--cpu-rows" => cpu_stride = it.next().ok_or("--cpu-rows N")?.parse().map_err(err)?,
            "--no-screen" => screen = false,
            "--tied" => tied = true,
            _ => dirs.push(PathBuf::from(a)),
        }
    }
    let out = out.ok_or("--out OUT.jsonl required")?;
    let mut paths = Vec::new();
    for d in &dirs { snapshots(d, &mut paths)?; }
    if paths.is_empty() && !tied { return Err("no snapshot.json found".into()) }
    let mut gpu = Gpu::init().map_err(err)?;
    if gpu.arch != "gfx1151" { return Err(format!("gfx1151 only, got {}", gpu.arch)) }
    for k in [HIPCC_SCORE, HIPCC_SELECT] { gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, k).map_err(err)?; }
    let pm = match &image {
        Some(p) => {
            if !p.is_absolute() { return Err("--pm-image must be absolute".into()) }
            let bytes = std::fs::read(p).map_err(err)?;
            let module = gpu.hip.module_load_data(&bytes).map_err(err)?;
            let score = gpu.hip.module_get_function(&module, PM_SCORE).ok();
            let select = gpu.hip.module_get_function(&module, PM_SELECT).ok();
            eprintln!("pm image {} sha256 {} score={} select={}", p.display(), sha256(&bytes), score.is_some(), select.is_some());
            Some(Pm { _module: module, score, select })
        }
        None => None,
    };
    let mut out_f = std::fs::File::create(&out).map_err(err)?;
    let events = [gpu.hip.event_create().map_err(err)?, gpu.hip.event_create().map_err(err)?, gpu.hip.event_create().map_err(err)?];
    let mut failures = 0usize;

    for path in &paths {
        let root = path.parent().unwrap();
        let header: Value = serde_json::from_slice(&std::fs::read(path).map_err(err)?).map_err(err)?;
        if header["schema"] != "qsa-source-v1" { return Err(format!("{}: unsupported schema", path.display())) }
        let sel = &header["identity"]["selector"];
        if header["identity"]["arch"] != "gfx1151" { return Err("snapshot arch not gfx1151".into()) }
        if sel["pooled_dtype"] != "F32" || sel["projection_dtype"] != "F32" { return Err(format!("{}: not the F32 pooled route", path.display())) }
        if sel["heads"] != 4 || sel["dim"] != 128 { return Err(format!("{}: uncovered selector geometry", path.display())) }
        let get = |name: &str| -> Result<Vec<u8>> {
            let b = std::fs::read(root.join(name)).map_err(err)?;
            if header["files"][name]["sha256"] != sha256(&b) { return Err(format!("{}: hash mismatch {name}", path.display())) }
            Ok(b)
        };
        let g = &header["geometry"];
        let n = |v: &Value, k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("missing {k}"));
        let (rows, position_start, compress) = (n(g, "rows")?, n(g, "position_start")?, n(g, "compress")?);
        let c = Call { rows, stride: n(sel, "projection_stride")?, block_count: (position_start + rows) / compress, dim: 128,
            compress, position_start, budget: n(g, "budget_blocks")?, capacity: n(g, "capacity")? };
        if c.budget > 512 || c.block_count == 0 { return Err("uncovered budget/block count".into()) }
        let pooled_capacity = n(sel, "pooled_capacity")?;
        let qbytes = get("index-projection.f32")?;
        let pbytes = get("pooled.source")?;
        let snap_sel = get("selected.i32")?;
        let snap_mirror = get("selected-mirror.i32")?;
        if qbytes.len() != rows * c.stride * 4 || pbytes.len() != pooled_capacity * 128 * 4 || snap_sel.len() != rows * c.capacity * 4
            || snap_mirror.len() != c.capacity * 4 || c.block_count > pooled_capacity {
            return Err(format!("{}: geometry/extent mismatch", path.display()))
        }
        let group = c.group();
        let bufs = Bufs {
            query: gpu.upload_raw(&qbytes, &[qbytes.len()]).map_err(err)?,
            pooled: gpu.upload_raw(&pbytes, &[pbytes.len()]).map_err(err)?,
            scores: gpu.upload_raw(&vec![0u8; group * c.block_count * 4], &[group * c.block_count * 4]).map_err(err)?,
            selected: gpu.upload_raw(&vec![0u8; snap_sel.len()], &[snap_sel.len()]).map_err(err)?,
            mirror: gpu.upload_raw(&vec![0u8; snap_mirror.len()], &[snap_mirror.len()]).map_err(err)?,
        };
        let mut rec = json!({"snapshot":path.display().to_string(),"ctx":header["identity"]["ctx"],"layer":header["identity"]["layer"],
            "position_start":position_start,"rows":rows,"block_count":c.block_count,"group":group,
            "groups":rows.div_ceil(group),"budget_blocks":c.budget,"capacity":c.capacity,"stride":c.stride});

        // Baseline hipcc/hipcc against the captured bytes, keeping the scores.
        poison_outputs(&gpu, &bufs)?;
        let mut base_scores = Vec::new();
        run_call(&mut gpu, &pm, &c, &bufs, (Impl::Hipcc, Impl::Hipcc), Some(&mut base_scores), None)?;
        let base_sel = download(&gpu, &bufs.selected)?;
        let base_mirror = download(&gpu, &bufs.mirror)?;
        let baseline_exact = base_sel == snap_sel && base_mirror == snap_mirror;
        rec["baseline_selected_equal"] = json!(base_sel == snap_sel);
        rec["baseline_mirror_equal"] = json!(base_mirror == snap_mirror);
        if !baseline_exact { failures += 1; }
        let score_keys: Vec<u32> = base_scores.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let query: Vec<f32> = qbytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let pooled: Vec<f32> = pbytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let hipcc_sel: Vec<i32> = base_sel.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
        if screen || cpu_stride > 0 {
            let (cpu, mut s) = cpu_checks(&c, &query, &pooled, &score_keys, &hipcc_sel, cpu_stride.max(1), screen);
            if cpu["cpu_selection_rows_differ"] != 0 || cpu["rescore_bits_differ"] != 0 { failures += 1; }
            rec["cpu"] = cpu;
            if screen { rec["screen"] = screen_json(&mut s); }
        }

        // Experimental arm: byte identity against the baseline.
        if arms != (Impl::Hipcc, Impl::Hipcc) {
            poison_outputs(&gpu, &bufs)?;
            let mut arm_scores = Vec::new();
            run_call(&mut gpu, &pm, &c, &bufs, arms, Some(&mut arm_scores), None)?;
            let sel_eq = download(&gpu, &bufs.selected)? == base_sel;
            let mir_eq = download(&gpu, &bufs.mirror)? == base_mirror;
            // Scores: every valid (causal) score bit equal and all poison
            // outside them untouched (the baseline leaves the same poison).
            let score_eq = arm_scores == base_scores;
            rec["arm"] = json!({"score":if arms.0==Impl::Pm {"pm"} else {"hipcc"},"select":if arms.1==Impl::Pm {"pm"} else {"hipcc"},
                "selected_equal":sel_eq,"mirror_equal":mir_eq,"score_bytes_equal_incl_poison":score_eq});
            if !(sel_eq && mir_eq && score_eq) { failures += 1; }
        }

        if timed > 0 {
            for _ in 0..2 { run_call(&mut gpu, &pm, &c, &bufs, arms, None, Some(&events))?; }
            let mut sc = Vec::new();
            let mut se = Vec::new();
            for _ in 0..timed {
                let t = run_call(&mut gpu, &pm, &c, &bufs, arms, None, Some(&events))?;
                sc.push(t.score_ms);
                se.push(t.select_ms);
            }
            sc.sort_by(f64::total_cmp);
            se.sort_by(f64::total_cmp);
            rec["timing"] = json!({"reps":timed,"score_ms_median":sc[sc.len()/2],"select_ms_median":se[se.len()/2],
                "score_ms":sc,"select_ms":se});
        }
        eprintln!("{} baseline_exact={} {}", path.display(), baseline_exact, rec.get("timing").map(|t| t.to_string()).unwrap_or_default());
        writeln!(out_f, "{rec}").map_err(err)?;
        out_f.flush().map_err(err)?;
        for t in [bufs.query, bufs.pooled, bufs.scores, bufs.selected, bufs.mirror] { gpu.free_tensor(t).map_err(err)?; }
    }

    if tied {
        if arms.1 != Impl::Pm { return Err("--tied runs the PM selector only (never the pre-fix hipcc arm)".into()) }
        for (blocks, value) in [(16384usize, 1.0f32), (16384, 0.0), (32768, 1.0), (65536, 1.0), (65536, 0.0)] {
            let compress = 4;
            let rows = 3usize;
            // The last row sees exactly `blocks` complete blocks plus a 3-token tail.
            let position_start = blocks * compress + 3 - rows;
            let c = Call { rows, stride: 0, block_count: blocks, dim: 128, compress, position_start, budget: 512, capacity: 512 * compress + compress - 1 };
            let scores: Vec<u8> = (0..rows * blocks).flat_map(|_| value.to_le_bytes()).collect();
            let sel_len = rows * c.capacity * 4;
            let b = Bufs {
                query: gpu.upload_raw(&[0u8; 16], &[16]).map_err(err)?,
                pooled: gpu.upload_raw(&[0u8; 16], &[16]).map_err(err)?,
                scores: gpu.upload_raw(&scores, &[scores.len()]).map_err(err)?,
                selected: gpu.upload_raw(&vec![0u8; sel_len], &[sel_len]).map_err(err)?,
                mirror: gpu.upload_raw(&vec![0u8; c.capacity * 4], &[c.capacity * 4]).map_err(err)?,
            };
            poison_outputs(&gpu, &b)?;
            let mut blob = select_blob(&c, &b, 0, rows, true);
            launch(&mut gpu, &pm, Impl::Pm, false, [rows as u32, 1, 1], &mut blob)?;
            gpu.hip.device_synchronize().map_err(err)?;
            let got: Vec<i32> = download(&gpu, &b.selected)?.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
            let mirror: Vec<i32> = download(&gpu, &b.mirror)?.chunks_exact(4).map(|x| i32::from_le_bytes(x.try_into().unwrap())).collect();
            let keys = vec![value.to_bits(); blocks];
            let mut ok = true;
            for r in 0..rows {
                let rb = c.row_blocks(r);
                let want = selection_row(&rank(&keys[..rb]), &c, r);
                ok &= got[r * c.capacity..(r + 1) * c.capacity] == want[..];
                if r == rows - 1 { ok &= mirror == want; }
            }
            if !ok { failures += 1; }
            let rec = json!({"tied":true,"blocks":blocks,"value":value,"rows":rows,"position_start":position_start,"pm_select_equal_cpu_oracle":ok});
            eprintln!("{rec}");
            writeln!(out_f, "{rec}").map_err(err)?;
            for t in [b.query, b.pooled, b.scores, b.selected, b.mirror] { gpu.free_tensor(t).map_err(err)?; }
        }
    }
    for e in events { gpu.hip.event_destroy(e).map_err(err)?; }
    if failures > 0 { return Err(format!("{failures} failing checks; see {out}")) }
    Ok(())
}
