// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! VMM continuous-batch state oracle (PLAN §6 Slice2B, §7 acceptance item 2).
//!
//! cb_vmm_state_oracle <model> --ks 1,2,3,4,5,6,7,8 --contexts 512,8192,32768
//!     --steps 256 --out <absolute.json> [--short 256] [--artifacts <absolute dir>]
//!     [--phase all|singleton] [--stop-repeats N] [--drift-cycles N]
//!
//! `--contexts` are test prefix lengths, never max_seq overrides: the model
//! loads through the production `load_qwen35_bundle` with automatic VMM
//! sequence sizing, and the loaded ack (kv_backend/max_seq/K mode) is
//! recorded. Request r of a k-request case uses prefix length
//! `[contexts.., short][r % n]` and a request-unique prompt, so every k>1 case
//! mixes unequal prompts and contexts.
//!
//! Singleton phase (base route, runs first and is a self-test): every
//! distinct fixture is prefilled and decoded greedily `--steps` times on the
//! existing singleton route; logits/hidden per step and full KV prefix,
//! DeltaNet S/scales/EF and conv state at each stage boundary are written as
//! raw bytes. A second independent run must be byte-identical (any differing
//! byte fails). Negative controls must FAIL: wrong row position (row_slot),
//! another request's reference (epoch), and publishing a rejected verify tail
//! (rejected-tail read). A rejected verify with DN restored and the KV tail
//! left poisoned must still MATCH (mask positive control).
//!
//! Comparisons are actual byte comparisons; sha256 digests are receipts only.

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, LayerType, Qwen35Config, StateQuant};
use hipfire_arch_qwen35::forward_slots::vmm::{Qwen35RequestState, Qwen35VmmStore, VmmRequestInit, VmmRoute};
use hipfire_arch_qwen35::{load_qwen35_bundle, Qwen35Bundle};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SequenceHint, SpecLoadCfg};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch, StepOutput,
};
use hipfire_runtime::sampler::SamplerConfig;
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

const CORPUS: &str = "The scheduler admits each request with its own key/value mapping, \
recurrent state and convolution ring. A committed step boundary is the only place where \
new work may join; rejected speculative rows are masked by absolute position and never \
published. Write a careful explanation of how a merge sort splits, recurses and merges, \
then implement it in Rust with tests, and finally discuss its cache behaviour on large \
inputs compared with an in-place quicksort.\n";

struct Args {
    model: String,
    ks: Vec<usize>,
    contexts: Vec<usize>,
    short: usize,
    steps: usize,
    out: PathBuf,
    artifacts: PathBuf,
    /// Run the executor batch phase after the singleton phase.
    batch: bool,
    /// In-process repetitions of the stop_id case (repeatability check).
    stop_repeats: usize,
    /// Sequential 2-request cycles for the memory-drift check (0 = off).
    drift_cycles: usize,
}

fn parse_list(s: &str) -> Result<Vec<usize>> {
    let v: Vec<usize> = s
        .split(',')
        .map(|p| p.trim().parse::<usize>())
        .collect::<std::result::Result<_, _>>()?;
    if v.is_empty() || v.contains(&0) {
        return Err(format!("list {s:?} must be non-empty positive integers").into());
    }
    Ok(v)
}

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: cb_vmm_state_oracle <model> --ks 1,..,8 --contexts 512,8192,32768 \
                 --steps 256 --out <absolute.json> [--short 256] [--artifacts <absolute dir>] \
                 [--phase all|singleton] [--stop-repeats N] [--drift-cycles N]";
    let mut batch = true;
    let mut stop_repeats = 1usize;
    let mut drift_cycles = 0usize;
    let model = raw.first().ok_or(usage)?.clone();
    let (mut ks, mut contexts, mut steps, mut out, mut short, mut artifacts) =
        (None, None, None, None, 256usize, None);
    let mut i = 1;
    while i < raw.len() {
        let val = raw.get(i + 1).ok_or(usage)?;
        match raw[i].as_str() {
            "--ks" => ks = Some(parse_list(val)?),
            "--contexts" => contexts = Some(parse_list(val)?),
            "--steps" => steps = Some(val.parse::<usize>()?),
            "--out" => out = Some(PathBuf::from(val)),
            "--short" => short = val.parse()?,
            "--artifacts" => artifacts = Some(PathBuf::from(val)),
            "--stop-repeats" => stop_repeats = val.parse::<usize>()?.max(1),
            "--drift-cycles" => drift_cycles = val.parse()?,
            "--phase" => {
                batch = match val.as_str() {
                    "all" => true,
                    "singleton" => false,
                    _ => return Err(usage.into()),
                }
            }
            other => return Err(format!("unknown flag {other}; {usage}").into()),
        }
        i += 2;
    }
    let out: PathBuf = out.ok_or(usage)?;
    if !out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    let artifacts = artifacts.unwrap_or_else(|| out.with_extension("artifacts"));
    if !artifacts.is_absolute() {
        return Err("--artifacts must be absolute".into());
    }
    let ks = ks.ok_or(usage)?;
    if ks.iter().any(|&k| k > 8) {
        return Err("--ks values must be 1..=8".into());
    }
    let steps = steps.ok_or(usage)?;
    if steps < 8 {
        return Err("--steps must be >= 8 (negative controls need 8 steps)".into());
    }
    Ok(Args { model, ks, contexts: contexts.ok_or(usage)?, short, steps, out, artifacts, batch, stop_repeats, drift_cycles })
}

/// One isolated request fixture: request-unique prompt at an exact length.
#[derive(Clone)]
struct Fixture {
    request: usize,
    prefix: usize,
    tokens: Vec<u32>,
}

impl Fixture {
    fn name(&self) -> String {
        format!("r{}_p{}", self.request, self.prefix)
    }
}

fn fixtures(args: &Args, tok: &Tokenizer) -> Result<Vec<Fixture>> {
    let lens: Vec<usize> = args.contexts.iter().copied().chain([args.short]).collect();
    let max_k = *args.ks.iter().max().unwrap();
    let mut out = Vec::with_capacity(max_k);
    for r in 0..max_k {
        let prefix = lens[r % lens.len()];
        let head = tok.encode(&format!("Request {r} of the continuous-batch oracle.\n"));
        let body = tok.encode(CORPUS);
        if body.is_empty() || head.is_empty() {
            return Err("tokenizer produced empty fixture".into());
        }
        // Rotate the shared corpus per request so streams differ at every row.
        let tokens: Vec<u32> = head
            .iter()
            .copied()
            .chain(body.iter().copied().cycle().skip(r * 7))
            .take(prefix)
            .collect();
        out.push(Fixture { request: r, prefix, tokens });
    }
    Ok(out)
}

fn read_dev(gpu: &Gpu, t: &GpuTensor, offset: usize, bytes: usize) -> Result<Vec<u8>> {
    if offset + bytes > t.buf.size() {
        return Err(format!("read {offset}+{bytes} exceeds allocation {}", t.buf.size()).into());
    }
    let mut raw = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh_at(&mut raw, &t.buf, offset)?;
    Ok(raw)
}

fn sha(bytes: &[u8]) -> String {
    let d = Sha256::digest(bytes);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn argmax(logits: &[u8]) -> Result<u32> {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, c) in logits.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes(c.try_into().unwrap());
        if !v.is_finite() {
            return Err(format!("nonfinite logit at {i}").into());
        }
        if v > best.1 {
            best = (i, v);
        }
    }
    Ok(best.0 as u32)
}

/// Per-position KV row stride for the loaded cache encoding.
fn kv_row_bytes(kv: &KvCache) -> Result<(usize, usize)> {
    if kv.quant_fp8 {
        let r = KvCache::fp8_row_bytes(kv.n_kv_heads, kv.head_dim)?;
        return Ok((r, r));
    }
    if kv.quant_bf16 {
        let r = KvCache::bf16_row_bytes(kv.n_kv_heads, kv.head_dim)?;
        return Ok((r, r));
    }
    let rotated = kv.quant_asym2 || kv.quant_asym3 || kv.quant_asym4 || kv.quant_fwht;
    if kv.quant_q8 && !rotated {
        let r = kv.n_kv_heads * (kv.head_dim / 32) * 34;
        return Ok((r, r));
    }
    Err("oracle supports the resolved default fp8/q8 (and bf16) KV encodings only".into())
}

/// Named raw state bytes at a committed boundary of `rows` positions.
fn state_bytes(
    gpu: &Gpu,
    config: &Qwen35Config,
    kv: &KvCache,
    dn: &DeltaNetState,
    rows: usize,
) -> Result<Vec<(String, Vec<u8>)>> {
    gpu.hip.device_synchronize()?;
    if dn.quant != StateQuant::Q8 {
        return Err(format!("oracle state layout covers Q8 DeltaNet only (got {:?})", dn.quant).into());
    }
    // Logical Q8 layout (same as mq4_prefill_state_oracle): i8 codes and f16
    // EF over heads×Dv², f32 scales/conv by element count. Allocation sizes
    // may be rounded and are never compared.
    let n = config.linear_num_value_heads * config.linear_value_head_dim.pow(2);
    let (kr, vr) = kv_row_bytes(kv)?;
    let mut out = Vec::new();
    let mut la = 0;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        if *ty == LayerType::LinearAttention {
            out.push((format!("L{layer:02}.dn_s"), read_dev(gpu, &dn.s_matrices[la], 0, n)?));
            if let Some(t) = dn.s_scales.get(la) {
                out.push((format!("L{layer:02}.dn_scales"), read_dev(gpu, t, 0, t.numel() * 4)?));
            }
            if let Some(t) = dn.s_ef_residual.get(la) {
                out.push((format!("L{layer:02}.dn_ef"), read_dev(gpu, t, 0, n * 2)?));
            }
            let c = &dn.conv_states[la];
            out.push((format!("L{layer:02}.conv"), read_dev(gpu, c, 0, c.numel() * 4)?));
            la += 1;
        } else {
            out.push((format!("L{layer:02}.k"), read_dev(gpu, &kv.k_gpu[layer], 0, rows * kr)?));
            out.push((format!("L{layer:02}.v"), read_dev(gpu, &kv.v_gpu[layer], 0, rows * vr)?));
        }
    }
    Ok(out)
}

/// Full trace of one request: per-step logits/hidden + boundary states.
struct Trace {
    /// logits after prefill, then after each decode step.
    logits: Vec<PathBuf>,
    hidden: Vec<PathBuf>,
    committed: Vec<u32>,
    position: usize,
    pending_seed: u32,
    states: Vec<(String, PathBuf)>,
    /// Prompt chunk lengths the route executed, in order.
    prefill_chunks: Vec<usize>,
}

struct Ctx {
    gpu: Gpu,
    b: Qwen35Bundle,
}

impl Ctx {
    fn logits(&self) -> Result<Vec<u8>> {
        self.gpu.hip.device_synchronize()?;
        read_dev(&self.gpu, &self.b.scratch.logits, 0, self.b.config.vocab_size * 4)
    }
    fn hidden(&self) -> Result<Vec<u8>> {
        read_dev(&self.gpu, &self.b.scratch.x, 0, self.b.config.dim * 4)
    }
    fn reset(&mut self) -> Result<()> {
        self.b.dn_state.reset(&mut self.gpu)?;
        self.gpu.hip.device_synchronize()?;
        Ok(())
    }
    fn prefill(&mut self, tokens: &[u32], start: usize) -> Result<()> {
        let b = &mut self.b;
        qwen35::forward_prefill_batch(
            &mut self.gpu, &b.weights, &b.config, tokens, start, &mut b.kv_cache,
            &mut b.dn_state, &b.scratch, None, None, None, None,
        )?;
        Ok(())
    }
    /// Default singleton serve prefill: the same outer chunking as the serve
    /// route (`ordinary_prefill_chunk_limit` + `ordinary_serve_prefill_chunk_len`),
    /// re-evaluated before every chunk. Returns the executed chunk lengths.
    fn prefill_serve(&mut self, tokens: &[u32]) -> Result<Vec<usize>> {
        let mut chunks = Vec::new();
        let mut done = 0;
        while done < tokens.len() {
            let rem = tokens.len() - done;
            let b = &self.b;
            let ceiling = qwen35::ordinary_prefill_chunk_limit(&self.gpu, &b.weights, &b.config, &b.dn_state, &b.kv_cache, None)?;
            let len = qwen35::prefill::ordinary_serve_prefill_chunk_len(rem, ceiling).unwrap_or(rem.min(ceiling).max(1));
            self.prefill(&tokens[done..done + len], done)?;
            chunks.push(len);
            done += len;
        }
        Ok(chunks)
    }
    fn decode(&mut self, token: u32, pos: usize) -> Result<()> {
        let b = &mut self.b;
        qwen35::forward_scratch(
            &mut self.gpu, &b.weights, &b.config, token, pos, &mut b.kv_cache,
            &mut b.dn_state, &b.scratch,
        )?;
        Ok(())
    }
    fn snapshot_dn(&self) -> Result<Vec<Vec<u8>>> {
        self.gpu.hip.device_synchronize()?;
        let dn = &self.b.dn_state;
        dn.s_matrices
            .iter()
            .chain(&dn.s_scales)
            .chain(&dn.s_ef_residual)
            .chain(&dn.conv_states)
            .map(|t| read_dev(&self.gpu, t, 0, t.buf.size()))
            .collect()
    }
    fn restore_dn(&self, snap: &[Vec<u8>]) -> Result<()> {
        let dn = &self.b.dn_state;
        let all: Vec<&GpuTensor> = dn
            .s_matrices
            .iter()
            .chain(&dn.s_scales)
            .chain(&dn.s_ef_residual)
            .chain(&dn.conv_states)
            .collect();
        if all.len() != snap.len() {
            return Err("DN snapshot arity mismatch".into());
        }
        for (t, bytes) in all.iter().zip(snap) {
            self.gpu.hip.memcpy_htod(&t.buf, bytes)?;
        }
        self.gpu.hip.device_synchronize()?;
        Ok(())
    }
    /// Overwrite KV rows [from, to) of every attention layer with 0x7f bytes.
    fn poison_kv(&mut self, from: usize, to: usize) -> Result<()> {
        self.b.kv_cache.ensure_mapped_capacity(&mut self.gpu, to)?;
        let (kr, vr) = kv_row_bytes(&self.b.kv_cache)?;
        for (layer, ty) in self.b.config.layer_types.iter().enumerate() {
            if *ty == LayerType::FullAttention {
                let kv = &self.b.kv_cache;
                self.gpu.hip.memcpy_htod_offset(&kv.k_gpu[layer].buf, from * kr, &vec![0x7f; (to - from) * kr])?;
                self.gpu.hip.memcpy_htod_offset(&kv.v_gpu[layer].buf, from * vr, &vec![0x7f; (to - from) * vr])?;
            }
        }
        self.gpu.hip.device_synchronize()?;
        Ok(())
    }
}

fn write_art(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path, bytes)?;
    Ok(())
}

/// Reference run: record every step to `dir`.
fn record(ctx: &mut Ctx, f: &Fixture, steps: usize, dir: &Path) -> Result<Trace> {
    fs::create_dir_all(dir.parent().ok_or("reference dir has no parent")?)?;
    fs::create_dir(dir)?; // never overwrite a previous reference
    ctx.reset()?;
    let prefill_chunks = ctx.prefill_serve(&f.tokens)?;
    let mut tr = Trace {
        logits: vec![],
        hidden: vec![],
        committed: vec![],
        position: f.prefix,
        pending_seed: 0,
        states: vec![],
        prefill_chunks,
    };
    let save_states = |ctx: &Ctx, tr: &mut Trace, stage: &str, rows: usize| -> Result<()> {
        for (name, bytes) in state_bytes(&ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, rows)? {
            let p = dir.join(stage).join(format!("{name}.bin"));
            write_art(&p, &bytes)?;
            tr.states.push((format!("{stage}/{name}"), p));
        }
        Ok(())
    };
    let l = ctx.logits()?;
    let p = dir.join("logits_0000.bin");
    write_art(&p, &l)?;
    tr.logits.push(p);
    tr.pending_seed = argmax(&l)?;
    save_states(ctx, &mut tr, "prefill", f.prefix)?;
    for s in 1..=steps {
        let tok = tr.pending_seed;
        ctx.decode(tok, tr.position)?;
        tr.committed.push(tok);
        tr.position += 1;
        let l = ctx.logits()?;
        let h = ctx.hidden()?;
        let (pl, ph) = (dir.join(format!("logits_{s:04}.bin")), dir.join(format!("hidden_{s:04}.bin")));
        write_art(&pl, &l)?;
        write_art(&ph, &h)?;
        tr.logits.push(pl);
        tr.hidden.push(ph);
        tr.pending_seed = argmax(&l)?;
    }
    let end = tr.position;
    save_states(ctx, &mut tr, "final", end)?;
    Ok(tr)
}

/// Byte comparison report: first differing (item, byte) and count of items.
#[derive(Default)]
struct Diff {
    compared_items: usize,
    compared_bytes: usize,
    differing_items: Vec<String>,
}

impl Diff {
    fn check(&mut self, item: &str, reference: &[u8], candidate: &[u8]) {
        self.compared_items += 1;
        self.compared_bytes += reference.len();
        if reference.len() != candidate.len() {
            self.differing_items.push(format!("{item}: len {} vs {}", reference.len(), candidate.len()));
        } else if let Some(i) = reference.iter().zip(candidate).position(|(a, b)| a != b) {
            let n = reference.iter().zip(candidate).filter(|(a, b)| a != b).count();
            self.differing_items.push(format!("{item}: first byte {i}, {n} bytes differ"));
        }
    }
    fn ids(&mut self, item: &str, reference: &[u32], candidate: &[u32]) {
        let to = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        self.check(item, &to(reference), &to(candidate));
    }
    fn exact(&self) -> bool {
        self.differing_items.is_empty() && self.compared_items > 0
    }
    fn json(&self) -> Value {
        json!({
            "exact": self.exact(),
            "compared_items": self.compared_items,
            "compared_bytes": self.compared_bytes,
            "differing_items": self.differing_items.len(),
            "first_differences": self.differing_items.iter().take(16).collect::<Vec<_>>(),
        })
    }
}

/// Replay `f` on the singleton route and byte-compare against `reference`.
/// `perturb` lets negative controls corrupt exactly one invariant.
#[derive(Clone, Copy, PartialEq)]
enum Perturb {
    None,
    /// Decode step 1 is written at position+1 (wrong row slot/position).
    RowSlot,
    /// Verify 4 rows at step 1, reject all drafts, restore DN, keep poisoned
    /// KV tail masked, then decode normally (must still match).
    RejectedTailMasked,
    /// Same verify, but publish the rejected tail: DN not restored and the
    /// committed position advances over the rejected rows (must fail).
    RejectedTailRead,
}

fn replay(ctx: &mut Ctx, f: &Fixture, steps: usize, reference: &Trace, perturb: Perturb) -> Result<Diff> {
    let mut d = Diff::default();
    let rd = |p: &Path| fs::read(p);
    ctx.reset()?;
    ctx.prefill_serve(&f.tokens)?;
    let l = ctx.logits()?;
    d.check("logits_0000", &rd(&reference.logits[0])?, &l);
    let mut seed = argmax(&l)?;
    let mut position = f.prefix;
    let mut committed = Vec::with_capacity(steps);
    let pre = state_bytes(&ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, f.prefix)?;
    for (name, bytes) in &pre {
        let (_, p) = reference.states.iter().find(|(n, _)| n == &format!("prefill/{name}")).ok_or("missing ref state")?;
        d.check(&format!("prefill/{name}"), &rd(p)?, bytes);
    }
    for s in 1..=steps {
        if s == 1 && matches!(perturb, Perturb::RejectedTailMasked | Perturb::RejectedTailRead) {
            let snap = ctx.snapshot_dn()?;
            ctx.poison_kv(position, position + 4)?;
            // Drafts that the target will reject: off-by-one token ids.
            let block = [seed, seed.wrapping_add(1) % 1000, 17, 23];
            ctx.prefill(&block, position)?;
            if perturb == Perturb::RejectedTailRead {
                committed.extend_from_slice(&block);
                position += block.len();
                seed = argmax(&ctx.logits()?)?;
            } else {
                ctx.restore_dn(&snap)?;
            }
        }
        let pos = if perturb == Perturb::RowSlot && s == 1 { position + 1 } else { position };
        ctx.decode(seed, pos)?;
        committed.push(seed);
        position += 1;
        let l = ctx.logits()?;
        let h = ctx.hidden()?;
        d.check(&format!("logits_{s:04}"), &rd(&reference.logits[s])?, &l);
        d.check(&format!("hidden_{s:04}"), &rd(&reference.hidden[s - 1])?, &h);
        seed = argmax(&l)?;
    }
    d.ids("committed_ids", &reference.committed, &committed);
    d.ids("position", &[reference.position as u32], &[position as u32]);
    d.ids("pending_seed", &[reference.pending_seed], &[seed]);
    let fin = state_bytes(&ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, reference.position)?;
    for (name, bytes) in &fin {
        let (_, p) = reference.states.iter().find(|(n, _)| n == &format!("final/{name}")).ok_or("missing ref state")?;
        d.check(&format!("final/{name}"), &rd(p)?, bytes);
    }
    Ok(d)
}

fn load(model: &str) -> Result<(Ctx, Tokenizer, Value)> {
    let hfq = HfqFile::open(Path::new(model))?;
    let tok = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| format!("{e:?}"))?;
    let meta: Value = serde_json::from_str(&hfq.metadata_json)?;
    let cfg = meta.get("config").unwrap_or(&meta);
    let model_ctx = cfg
        .get("text_config")
        .unwrap_or(cfg)
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .ok_or("checkpoint has no max_position_embeddings")? as usize;
    drop(hfq);
    let mut gpu = Gpu::init()?;
    let src = ModelSource::from_path(model)?;
    let cask = CaskConfig::default();
    let qwen_default_q8 = !matches!(
        hipfire_config::developer_var("HIPFIRE_QWEN_KV_DEFAULT_Q8").ok().as_deref(),
        Some("0")
    );
    // Automatic sequence sizing exactly as admission's "pending" bound: the
    // carrier re-measures the card after weights. No max_seq/kv_backend override.
    let mut lc = LoadCtx {
        path: model,
        max_seq: model_ctx,
        sequence: Some(SequenceHint { model_ctx, automatic: true, card_cap: 0 }),
        deepseek4_compute_placement: Default::default(),
        deepseek4_experts_per_token: None,
        draft_path: None,
        vision_path: None,
        mtp_path: None,
        vision_mode: "off".to_string(),
        kv_mode_override: None,
        kv_k_override: None,
        kv_v_override: None,
        qwen_default_q8,
        kv_backend: KvBackend::default(),
        kv_adaptive_override: None,
        state_quant_override: None,
        cask: &cask,
        pp: 1,
        spec: SpecLoadCfg::default(),
        gpu: &mut gpu,
        gemma4_drafter_path: None,
        gemma4_draft_len: 3,
        xdna: None,
    };
    let b = load_qwen35_bundle(src, &mut lc).map_err(|e| format!("load_qwen35_bundle: {e}"))?;
    let max_seq = lc.max_seq;
    let kv = &b.kv_cache;
    let ack = json!({
        "kv_backend": if kv.uses_vmm_backend() { "vmm" } else { "legacy" },
        "kv_backend_legacy": !kv.uses_vmm_backend(),
        "max_seq_bound": max_seq,
        "model_ctx": model_ctx,
        "kv_k_mode": format!("{:?}", kv.current_kv_mode().ok()),
        "kv_fp8": kv.quant_fp8,
        "kv_q8": kv.quant_q8,
        "dn_quant": format!("{:?}", b.dn_state.quant),
        "arch": gpu.arch.clone(),
        "qwen_default_q8": qwen_default_q8,
    });
    eprintln!("loaded ack: {ack}");
    if !kv.uses_vmm_backend() {
        return Err("loaded KV is not VMM; oracle refuses a legacy owner".into());
    }
    Ok((Ctx { gpu, b }, tok, ack))
}

// ── Batch phase: Qwen35VmmStore / executor (§4.2) ────────────────────

/// Requests in flight per store. Prefill chunk lengths come from the store's
/// `exact_prefill_chunk_len` (the singleton serve chunking), one prefill
/// request per step so the row budget is width + one chunk.
const WIDTH: usize = 8;

enum Sink {
    /// Isolated executor reference (k=1 on the batch route).
    Record { dir: PathBuf, trace: Trace },
    /// Batch request: byte-compare to the isolated executor reference (state
    /// bleed; must be exact) and to the singleton route (route exactness).
    Compare { b: usize, a: usize, vs_b: Diff, vs_a: Diff },
}

struct Req {
    fx: usize,
    epoch: RequestEpoch,
    slot: usize,
    /// Picks to commit (pick 0 comes from the final prompt chunk).
    max_tokens: usize,
    stop: Option<u32>,
    cancel_at: Option<usize>,
    fed: usize,
    committed: Vec<u32>,
    finish: Option<String>,
    store_watermark_errors: Vec<String>,
    /// Prompt chunk lengths the executor ran for this request.
    chunks: Vec<usize>,
    sink: Sink,
}

impl Req {
    fn new(fx: usize, tag: u64, generation: u64, slot: usize, max_tokens: usize, sink: Sink) -> Self {
        Self {
            fx,
            epoch: RequestEpoch { request_tag: tag, owner_generation: generation },
            slot,
            max_tokens,
            stop: None,
            cancel_at: None,
            fed: 0,
            committed: Vec::new(),
            finish: None,
            store_watermark_errors: Vec::new(),
            chunks: Vec::new(),
            sink,
        }
    }
}

struct Refs<'a> {
    b: &'a [Option<Trace>],
    a: &'a [Trace],
}

fn state_cmp(d: &mut Diff, stage: &str, now: &[(String, Vec<u8>)], reference: &Trace) -> Result<()> {
    for (name, bytes) in now {
        let key = format!("{stage}/{name}");
        match reference.states.iter().find(|(n, _)| *n == key) {
            Some((_, p)) => d.check(&key, &fs::read(p)?, bytes),
            None => d.differing_items.push(format!("{key}: missing in reference")),
        }
    }
    Ok(())
}

/// Record or compare pick `i` (logits after the head row, hidden of that row).
fn sink_pick(sink: &mut Sink, refs: &Refs, i: usize, logits: &[u8], hidden: &[u8]) -> Result<()> {
    match sink {
        Sink::Record { dir, trace } => {
            let pl = dir.join(format!("logits_{i:04}.bin"));
            write_art(&pl, logits)?;
            trace.logits.push(pl);
            if i > 0 {
                let ph = dir.join(format!("hidden_{i:04}.bin"));
                write_art(&ph, hidden)?;
                trace.hidden.push(ph);
            }
        }
        Sink::Compare { b, a, vs_b, vs_a } => {
            for (d, t) in [(vs_b, refs.b[*b].as_ref().ok_or("missing isolated reference")?), (vs_a, &refs.a[*a])] {
                match t.logits.get(i) {
                    Some(p) => d.check(&format!("logits_{i:04}"), &fs::read(p)?, logits),
                    None => d.differing_items.push(format!("logits_{i:04}: beyond reference")),
                }
                if i > 0 {
                    d.check(&format!("hidden_{i:04}"), &fs::read(&t.hidden[i - 1])?, hidden);
                }
            }
        }
    }
    Ok(())
}

fn sink_state(sink: &mut Sink, refs: &Refs, stage: &str, now: Vec<(String, Vec<u8>)>) -> Result<()> {
    match sink {
        Sink::Record { dir, trace } => {
            for (name, bytes) in now {
                let p = dir.join(stage).join(format!("{name}.bin"));
                write_art(&p, &bytes)?;
                trace.states.push((format!("{stage}/{name}"), p));
            }
        }
        Sink::Compare { b, a, vs_b, vs_a } => {
            state_cmp(vs_b, stage, &now, refs.b[*b].as_ref().ok_or("missing isolated reference")?)?;
            state_cmp(vs_a, stage, &now, &refs.a[*a])?;
        }
    }
    Ok(())
}

/// Frontier/terminal comparisons once a request stops committing.
fn sink_finish(sink: &mut Sink, refs: &Refs, r_committed: &[u32], position: usize, pending: Option<u32>) -> Result<()> {
    match sink {
        Sink::Record { trace, .. } => {
            // Reference convention: committed = fed picks, pending = last pick.
            let (fed, last) = r_committed.split_at(r_committed.len() - 1);
            trace.committed = fed.to_vec();
            trace.pending_seed = last[0];
            trace.position = position;
            if pending != Some(last[0]) {
                return Err("store pending_seed disagrees with last commit".into());
            }
        }
        Sink::Compare { b, a, vs_b, vs_a } => {
            for (d, t) in [(vs_b, refs.b[*b].as_ref().ok_or("missing isolated reference")?), (vs_a, &refs.a[*a])] {
                let picks: Vec<u32> = t.committed.iter().copied().chain([t.pending_seed]).collect();
                let n = r_committed.len().min(picks.len());
                d.ids("committed_ids", &picks[..n], r_committed);
                let prefix = t.position - t.committed.len();
                d.ids("position", &[(prefix + r_committed.len() - 1) as u32], &[position as u32]);
                d.ids("pending_seed", &[picks[r_committed.len() - 1]], &[pending.unwrap_or(u32::MAX)]);
            }
        }
    }
    Ok(())
}

/// Prompt chunk sequence the executor ran. Informational, not a parity
/// item: the admitted ceiling follows free VRAM (other live requests), and
/// the inner planner keeps 512-row state/KV commits, so chunking may differ
/// while every state byte matches. Recorded in the reference and case JSON.
fn sink_chunks(sink: &mut Sink, chunks: &[usize]) {
    if let Sink::Record { trace, .. } = sink {
        trace.prefill_chunks = chunks.to_vec();
    }
}

fn free_vram(gpu: &Gpu) -> Result<usize> {
    Ok(gpu.hip.get_vram_info()?.0)
}

fn new_store(ctx: &mut Ctx) -> Result<Qwen35VmmStore> {
    // The singleton phase leaves its optional widened prefill scratch cached
    // (32K prefix); the batch route never uses it, so release it first.
    if let Some(pbs) = ctx.b.scratch.widened_prefill_batch.borrow_mut().take() {
        pbs.free_gpu(&mut ctx.gpu)?;
    }
    // Exact route: prefill rows run on the singleton prefill and do not count
    // against the trunk row budget, so the budget is the decode width.
    // Shared physical KV budget: free VRAM minus a 6 GiB margin for
    // per-request DeltaNet state and transients. This is an admission
    // budget, not a max_seq override.
    let free = free_vram(&ctx.gpu)?;
    let budget = free.saturating_sub(6 << 30);
    eprintln!("batch store: route=Exact width={WIDTH} row_budget={WIDTH} free_vram={free} kv_budget={budget}");
    Ok(Qwen35VmmStore::new(&mut ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, WIDTH, WIDTH, budget, VmmRoute::Exact)?)
}

fn admit(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], r: &Req) -> Result<()> {
    let stop = r.stop.into_iter().collect();
    // Greedy sampler: plain argmax, the singleton reference semantics.
    let init = VmmRequestInit {
        prompt_len: fx[r.fx].prefix,
        stop_ids: stop,
        sampler: SamplerConfig::greedy(),
        rng_state: 0,
        history: vec![],
    };
    let st = Qwen35RequestState::new_like(
        &mut ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, r.epoch, r.slot, init,
    )?;
    if let Err((st, e)) = store.admit(st) {
        st.free_gpu(&mut ctx.gpu)?;
        return Err(e.into());
    }
    Ok(())
}

fn retire(ctx: &mut Ctx, store: &mut Qwen35VmmStore, epoch: &RequestEpoch) -> Result<()> {
    store.retire(epoch)?.free_gpu(&mut ctx.gpu)?;
    Ok(())
}

/// Build the next step: one AR row per decoding request plus one prefill
/// chunk (first prefilling request in slot order) of the exact route's
/// singleton chunk length. Other prefilling requests idle (masked) this step.
fn build_plan(ctx: &Ctx, store: &Qwen35VmmStore, fx: &[Fixture], reqs: &[Req]) -> Result<BatchStepPlan> {
    let mut per: Vec<(usize, Vec<u32>, usize, Option<(RequestEpoch, RequestStepKind)>)> =
        (0..WIDTH).map(|s| (s, vec![], 0, None)).collect();
    let mut live: Vec<&Req> = reqs.iter().filter(|r| r.finish.is_none()).collect();
    live.sort_by_key(|r| r.slot);
    let mut prefill_taken = false;
    for r in live {
        store.request_state(&r.epoch).ok_or("live request missing from store")?;
        let f = &fx[r.fx];
        let (toks, start, kind) = if r.fed < f.prefix {
            if prefill_taken {
                continue;
            }
            prefill_taken = true;
            let len = store.exact_prefill_chunk_len(&ctx.gpu, &ctx.b.weights, &ctx.b.config, &r.epoch, f.prefix - r.fed)?;
            (f.tokens[r.fed..r.fed + len].to_vec(), r.fed, RequestStepKind::Prefill)
        } else {
            let seed = *r.committed.last().ok_or("decode without a pick")?;
            (vec![seed], f.prefix + r.committed.len() - 1, RequestStepKind::Ar)
        };
        per[r.slot] = (r.slot, toks, start, Some((r.epoch, kind)));
    }
    let triples: Vec<(SlotId, &[u32], usize)> =
        per.iter().map(|(s, t, p, _)| (SlotId(*s), t.as_slice(), *p)).collect();
    let batch = SlotBatch::build(&triples);
    let mut requests = Vec::new();
    let (mut begin, mut decode_rows, mut prefill_rows) = (0, 0, 0);
    for (_, t, _, who) in &per {
        if let Some((epoch, kind)) = who {
            requests.push(RequestRows { epoch: *epoch, rows: RowRange { begin, len: t.len() }, kind: *kind });
            match kind {
                RequestStepKind::Ar => decode_rows += t.len(),
                _ => prefill_rows += t.len(),
            }
        }
        begin += t.len();
    }
    Ok(BatchStepPlan { batch, requests, decode_rows, prefill_rows, verify_rows: 0, forced_rows: 0 })
}

/// Run one planned step and publish every head into its request's sink.
fn run_step(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], refs: &Refs, plan: &BatchStepPlan) -> Result<()> {
    let Ctx { gpu, b } = ctx;
    let out = {
        let mut ex = store.executor(&b.weights, &b.config, &b.scratch);
        ex.provision_step(gpu, plan)?;
        ex.forward_step(gpu, plan)?
    };
    let (vocab, dim) = (b.config.vocab_size, b.config.dim);
    // Device heads are valid until the next forward: read before commit.
    let mut heads = Vec::new();
    // Exact route: hidden() holds decode rows in plan order of the AR
    // requests; prefill rows have no hidden (pick 0 records none).
    let mut ar_ordinal = 0;
    for rr in &plan.requests {
        let last = rr.rows.end() - 1;
        let ar_row = (rr.kind == RequestStepKind::Ar).then(|| {
            ar_ordinal += 1;
            ar_ordinal - 1
        });
        if out.target_picks[last] != u32::MAX {
            let slot = plan.batch.row_slot[last] as usize;
            let l = read_dev(gpu, store.logits(), slot * vocab * 4, vocab * 4)?;
            let h = match ar_row {
                Some(row) => read_dev(gpu, store.hidden(), row * dim * 4, dim * 4)?,
                None => Vec::new(),
            };
            if argmax(&l)? != out.target_picks[last] {
                return Err(format!("device pick {} != host argmax of slot logits", out.target_picks[last]).into());
            }
            heads.push((rr.epoch, l, h));
        }
    }
    let advances = store.executor(&b.weights, &b.config, &b.scratch).commit_step(gpu, plan, out)?;
    for rr in &plan.requests {
        let r = reqs.iter_mut().find(|r| r.epoch == rr.epoch).ok_or("plan epoch not in case")?;
        if rr.kind == RequestStepKind::Prefill {
            r.fed += rr.rows.len;
            r.chunks.push(rr.rows.len);
            if r.fed == fx[r.fx].prefix {
                let st = store.request_state(&r.epoch).ok_or("request vanished")?;
                let now = state_bytes(gpu, &b.config, &st.kv, &st.dn, r.fed)?;
                sink_state(&mut r.sink, refs, "prefill", now)?;
                sink_chunks(&mut r.sink, &r.chunks);
            }
        }
    }
    if advances.len() != heads.len() {
        return Err(format!("{} advances for {} heads", advances.len(), heads.len()).into());
    }
    for (adv, (epoch, l, h)) in advances.iter().zip(heads) {
        if adv.epoch != epoch || adv.committed_ids.len() != 1 {
            return Err("advance order/arity differs from plan heads".into());
        }
        let r = reqs.iter_mut().find(|r| r.epoch == epoch).unwrap();
        let i = r.committed.len();
        r.committed.push(adv.committed_ids[0]);
        sink_pick(&mut r.sink, refs, i, &l, &h)?;
        let st = store.request_state(&epoch).ok_or("request vanished")?;
        let expect_pos = fx[r.fx].prefix + r.committed.len() - 1;
        if adv.committed_position != expect_pos || st.position != expect_pos || st.pending_seed != Some(adv.committed_ids[0]) {
            r.store_watermark_errors.push(format!(
                "pick {i}: advance pos {} store pos {} expected {expect_pos}, pending {:?}",
                adv.committed_position, st.position, st.pending_seed
            ));
        }
        if let Some(f) = &adv.finish {
            r.finish = Some(f.clone());
        } else if r.committed.len() >= r.max_tokens {
            r.finish = Some("max_tokens".into());
        }
    }
    Ok(())
}

/// Retire finished/cancelled requests, publishing final comparisons.
fn reap(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], refs: &Refs, steps: usize) -> Result<()> {
    for r in reqs.iter_mut() {
        if r.finish.is_none() && r.cancel_at.is_some_and(|c| r.committed.len() >= c) {
            r.finish = Some("cancelled".into());
        }
        let Some(fin) = r.finish.clone() else { continue };
        let Some(st) = store.request_state(&r.epoch) else { continue };
        if fin != "cancelled" && !r.committed.is_empty() {
            let pos = st.position;
            let pending = st.pending_seed;
            if r.committed.len() == steps + 1 {
                let now = state_bytes(&ctx.gpu, &ctx.b.config, &st.kv, &st.dn, pos)?;
                sink_state(&mut r.sink, refs, "final", now)?;
            }
            sink_finish(&mut r.sink, refs, &r.committed, pos, pending)?;
        }
        let _ = fx;
        retire(ctx, store, &r.epoch)?;
    }
    Ok(())
}

/// Admit every request, step until all finish. On any error every admitted
/// owner is aborted/retired so the store is reusable.
fn drive(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], refs: &Refs, steps: usize, controls: Option<&mut serde_json::Map<String, Value>>) -> Result<()> {
    let mut controls = controls;
    let res = (|| -> Result<()> {
        for r in reqs.iter() {
            admit(ctx, store, fx, r)?;
        }
        loop {
            reap(ctx, store, fx, reqs, refs, steps)?;
            if reqs.iter().all(|r| r.finish.is_some()) {
                return Ok(());
            }
            if let Some(map) = controls.as_deref_mut() {
                let live: Vec<usize> = (0..reqs.len()).filter(|&i| reqs[i].finish.is_none()).collect();
                if live.len() >= 2 && live.iter().all(|&i| reqs[i].fed == fx[reqs[i].fx].prefix && reqs[i].committed.len() >= 4) {
                    executor_controls(ctx, store, fx, reqs, &live, map)?;
                    controls = None;
                    continue;
                }
            }
            let plan = build_plan(ctx, store, fx, reqs)?;
            run_step(ctx, store, fx, reqs, refs, &plan).inspect_err(|_| store.abort_step(&plan))?;
        }
    })();
    if res.is_err() {
        for r in reqs.iter() {
            if store.request_state(&r.epoch).is_some() {
                let _ = retire(ctx, store, &r.epoch);
            }
        }
    }
    res
}

fn expect_err(map: &mut serde_json::Map<String, Value>, name: &str, r: std::result::Result<(), String>) {
    let ok = r.is_err();
    eprintln!("executor control {name}: refused={ok} -> {}", if ok { "OK" } else { "FAIL" });
    map.insert(name.into(), json!({"expected": "refused", "ok": ok, "error": r.err()}));
}

/// Executor-level negative/positive controls with two live decoding
/// requests. Refusals must leave state untouched: the case continues and
/// its byte comparisons cover that.
fn executor_controls(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], reqs: &mut [Req], live: &[usize], map: &mut serde_json::Map<String, Value>) -> Result<()> {
    let (i0, i1) = (live[0], live[1]);
    let base = build_plan(ctx, store, fx, reqs)?;
    let try_plan = |ctx: &mut Ctx, store: &mut Qwen35VmmStore, plan: &BatchStepPlan| -> std::result::Result<(), String> {
        let Ctx { gpu, b } = ctx;
        let r = store.executor(&b.weights, &b.config, &b.scratch).provision_step(gpu, plan);
        if r.is_ok() {
            store.abort_step(plan); // provisioned only: no device writes, no poison
        }
        r
    };
    // row_slot: request0's AR row addressed at request1's slot.
    let mut p = base.clone();
    let row0 = p.requests.iter().find(|q| q.epoch == reqs[i0].epoch).unwrap().rows.begin;
    p.batch.row_slot[row0] = reqs[i1].slot as i32;
    expect_err(map, "neg_row_slot", try_plan(ctx, store, &p));
    // epoch: stale owner generation.
    let mut p = base.clone();
    p.requests[0].epoch.owner_generation += 1;
    expect_err(map, "neg_epoch_generation", try_plan(ctx, store, &p));
    // rejected-tail read: poison rows past request1's frontier (masked
    // positive control, verified by the case's exact comparison), then an
    // AR row whose position skips onto the poisoned tail must be refused.
    let pos1 = store.request_state(&reqs[i1].epoch).unwrap().position;
    {
        let Ctx { gpu, b } = ctx;
        let st = store.request_state_mut(&reqs[i1].epoch).unwrap();
        st.kv.ensure_mapped_capacity(gpu, pos1 + 4)?;
        let (kr, vr) = kv_row_bytes(&st.kv)?;
        for (layer, ty) in b.config.layer_types.iter().enumerate() {
            if *ty == LayerType::FullAttention {
                gpu.hip.memcpy_htod_offset(&st.kv.k_gpu[layer].buf, (pos1 + 1) * kr, &vec![0x7f; 3 * kr])?;
                gpu.hip.memcpy_htod_offset(&st.kv.v_gpu[layer].buf, (pos1 + 1) * vr, &vec![0x7f; 3 * vr])?;
            }
        }
        gpu.hip.device_synchronize()?;
    }
    map.insert("pos_rejected_tail_masked".into(), json!({"poisoned_rows": [pos1 + 1, pos1 + 4], "request": reqs[i1].fx, "verified_by": "case byte comparison"}));
    let mut p = base.clone();
    let row1 = p.requests.iter().find(|q| q.epoch == reqs[i1].epoch).unwrap().rows.begin;
    p.batch.positions[row1] += 2;
    expect_err(map, "neg_rejected_tail_read", try_plan(ctx, store, &p));
    // Stale commit on a step planned for request0 only: a foreign step id
    // must be refused. The step is then aborted after its forward, which
    // must poison request0 (device state written, never committed) while
    // request1 continues and must stay byte-exact over its poisoned tail.
    let solo = build_plan(ctx, store, fx, std::slice::from_ref(&reqs[i0]))?;
    {
        let Ctx { gpu, b } = ctx;
        let mut ex = store.executor(&b.weights, &b.config, &b.scratch);
        ex.provision_step(gpu, &solo)?;
        let out = ex.forward_step(gpu, &solo)?;
        let forged = StepOutput { step_id: out.step_id + 1, target_picks: out.target_picks.clone() };
        let r = ex.commit_step(gpu, &solo, forged).map(|_| ());
        store.abort_step(&solo);
        let ok = r.is_err();
        eprintln!("executor control neg_stale_commit: refused={ok} -> {}", if ok { "OK" } else { "FAIL" });
        map.insert("neg_stale_commit".into(), json!({"expected": "refused", "ok": ok, "error": r.err()}));
    }
    let poisoned = store.request_state(&reqs[i0].epoch).is_some_and(|s| s.poisoned)
        && store.request_state(&reqs[i1].epoch).is_some_and(|s| !s.poisoned);
    eprintln!("executor control poison_after_aborted_forward: {} -> {}", poisoned, if poisoned { "OK" } else { "FAIL" });
    map.insert("poison_after_aborted_forward".into(), json!({"ok": poisoned}));
    expect_err(map, "neg_poisoned_provision", try_plan(ctx, store, &base));
    reqs[i0].finish = Some("cancelled".into());
    Ok(())
}

fn diff_ok(sink: &Sink) -> (bool, bool, Value, Value) {
    match sink {
        Sink::Compare { vs_b, vs_a, .. } => (vs_b.exact(), vs_a.exact(), vs_b.json(), vs_a.json()),
        Sink::Record { .. } => (true, true, Value::Null, Value::Null),
    }
}

/// Per-case receipt; returns whether the batch-vs-isolated comparison is exact.
fn case_json(name: &str, fx: &[Fixture], reqs: &[Req]) -> (bool, bool, Value) {
    let (mut bleed_exact, mut route_exact) = (true, true);
    let rows: Vec<Value> = reqs
        .iter()
        .map(|r| {
            let (b, a, jb, ja) = diff_ok(&r.sink);
            let wm = r.store_watermark_errors.is_empty();
            bleed_exact &= b && wm;
            route_exact &= a;
            json!({
                "fixture": fx[r.fx].name(), "slot": r.slot, "epoch": [r.epoch.request_tag, r.epoch.owner_generation],
                "max_tokens": r.max_tokens, "stop": r.stop, "cancel_at": r.cancel_at, "prefill_chunks": r.chunks,
                "finish": r.finish, "committed": r.committed.len(),
                "watermarks_ok": wm, "watermark_errors": r.store_watermark_errors,
                "vs_isolated_executor": jb, "vs_singleton_route": ja,
            })
        })
        .collect();
    eprintln!("batch {name}: isolated-exact={bleed_exact} singleton-route-exact={route_exact}");
    (bleed_exact, route_exact, json!({"case": name, "bleed_exact": bleed_exact, "route_exact": route_exact, "requests": rows}))
}

fn compare_sink(i: usize) -> Sink {
    Sink::Compare { b: i, a: i, vs_b: Diff::default(), vs_a: Diff::default() }
}

/// Free VRAM and pool counters `(free, pool_new, pool_reused, pool_bytes_new)`.
fn mem_sample(gpu: &Gpu) -> Result<(usize, usize, usize, usize)> {
    gpu.hip.device_synchronize()?;
    let (n, r, b) = gpu.pool_stats();
    Ok((free_vram(gpu)?, n, r, b))
}

/// Memory drift over `cycles` sequential 2-request admit/run/retire cycles
/// (the two shortest fixtures, 8 picks each, byte-compared like every
/// case), plus isolation probes that attribute any drift: DeltaNet state
/// alone and a whole request owner (VMM KV + DN) allocated and freed with
/// no forward. Bounded = the last 10 cycles together lose < 16 MiB.
fn drift_check(ctx: &mut Ctx, store: &mut Qwen35VmmStore, fx: &[Fixture], refs: &Refs, supported: &[usize], cycles: usize) -> Result<(Value, bool)> {
    let mut by_len = supported.to_vec();
    by_len.sort_by_key(|&i| fx[i].prefix);
    let (c0, c1) = (by_len[0], by_len[1]);
    let mut rows = Vec::with_capacity(cycles);
    let mut exact_all = true;
    let start = mem_sample(&ctx.gpu)?;
    let mut refused: Option<(usize, String)> = None;
    for c in 0..cycles {
        let before = mem_sample(&ctx.gpu)?;
        let tag = 20_000 + 10 * c as u64;
        let mut reqs = vec![
            Req::new(c0, tag, 1, c % WIDTH, 8, compare_sink(c0)),
            Req::new(c1, tag + 1, 1, (c + 4) % WIDTH, 8, compare_sink(c1)),
        ];
        // `steps` far above 8 picks: no "final" (256-step) state comparison.
        if let Err(e) = drive(ctx, store, fx, &mut reqs, refs, usize::MAX - 1, None) {
            eprintln!("drift cycle {c}: REFUSED/FAILED: {e}");
            refused = Some((c, e.to_string()));
            break;
        }
        let after = mem_sample(&ctx.gpu)?;
        let exact = reqs.iter().all(|r| matches!(&r.sink, Sink::Compare { vs_b, vs_a, .. } if vs_b.exact() && vs_a.exact()) && r.store_watermark_errors.is_empty());
        exact_all &= exact;
        let d_free = after.0 as i64 - before.0 as i64;
        // Singleton's cached widened prefill scratch (exact prefill runs on it).
        let widened = ctx.b.scratch.widened_prefill_batch.borrow().as_ref().map(|p| p.max_batch);
        let store_mapped = store.mapped_kv_bytes()?;
        rows.push(json!({
            "cycle": c, "free_before": before.0, "free_after": after.0, "free_delta": d_free,
            "pool_new_delta": after.1 - before.1, "pool_reused_delta": after.2 - before.2,
            "pool_bytes_new_delta": after.3 - before.3, "widened_pbs_rows": widened,
            "store_mapped_after_retire": store_mapped, "exact": exact,
        }));
        eprintln!(
            "drift cycle {c}: free_before={} free_delta={d_free} pool_new+={} pool_reused+={} pool_bytes_new+={} widened_pbs={widened:?} store_mapped={store_mapped} exact={exact}",
            before.0, after.1 - before.1, after.2 - before.2, after.3 - before.3
        );
    }
    let end = mem_sample(&ctx.gpu)?;
    let deltas: Vec<i64> = rows.iter().map(|r| r["free_delta"].as_i64().unwrap()).collect();
    let last10: i64 = deltas.iter().rev().take(10).sum();
    let first10: i64 = deltas.iter().take(10).sum();
    // Bounded only if every cycle ran and the last 10 together lose < 16 MiB.
    let bounded = refused.is_none() && rows.len() >= 10 && last10 > -(16 << 20);
    // Attribution probes (no forward, no admit).
    let probe = |ctx: &mut Ctx, kind: &str, n: usize| -> Result<Value> {
        let s0 = mem_sample(&ctx.gpu)?;
        for i in 0..n {
            match kind {
                "dn_state" => {
                    let dn = DeltaNetState::new_with_quant(&mut ctx.gpu, &ctx.b.config, ctx.b.dn_state.quant)?;
                    dn.free_gpu(&mut ctx.gpu);
                }
                _ => {
                    let init = VmmRequestInit { prompt_len: 1, stop_ids: vec![], sampler: SamplerConfig::greedy(), rng_state: 0, history: vec![] };
                    let epoch = RequestEpoch { request_tag: 30_000 + i as u64, owner_generation: 1 };
                    let st = Qwen35RequestState::new_like(&mut ctx.gpu, &ctx.b.config, &ctx.b.kv_cache, &ctx.b.dn_state, epoch, 0, init)?;
                    st.free_gpu(&mut ctx.gpu)?;
                }
            }
        }
        let s1 = mem_sample(&ctx.gpu)?;
        let v = json!({"iterations": n, "free_delta": s1.0 as i64 - s0.0 as i64, "pool_new_delta": s1.1 - s0.1, "pool_bytes_new_delta": s1.3 - s0.3});
        eprintln!("drift probe {kind}: {v}");
        Ok(v)
    };
    let p_dn = probe(ctx, "dn_state", 10).unwrap_or_else(|e| json!({"error": e.to_string()}));
    let p_owner = probe(ctx, "request_owner", 10).unwrap_or_else(|e| json!({"error": e.to_string()}));
    eprintln!(
        "drift: cycles_run={}/{cycles} total_free_delta={} first10={first10} last10={last10} bounded={bounded} exact={exact_all} refused={refused:?}",
        rows.len(),
        end.0 as i64 - start.0 as i64
    );
    let v = json!({
        "cycles_requested": cycles, "cycles_run": rows.len(),
        "refused": refused.as_ref().map(|(c, e)| json!({"cycle": c, "error": e})),
        "fixtures": [fx[c0].name(), fx[c1].name()], "picks_per_request": 8,
        "start": {"free": start.0, "pool_new": start.1, "pool_reused": start.2, "pool_bytes_new": start.3},
        "end": {"free": end.0, "pool_new": end.1, "pool_reused": end.2, "pool_bytes_new": end.3},
        "total_free_delta": end.0 as i64 - start.0 as i64, "first10_free_delta": first10, "last10_free_delta": last10,
        "bounded": bounded, "all_exact": exact_all,
        "probe_dn_state_x10": p_dn, "probe_request_owner_x10": p_owner, "per_cycle": rows,
    });
    Ok((v, bounded && exact_all))
}

/// Whole batch phase. Returns (report, pass).
fn batch_phase(ctx: &mut Ctx, args: &Args, fx: &[Fixture], refs_a: &[Trace]) -> Result<(Value, bool)> {
    let mut store = new_store(ctx)?;
    let steps = args.steps;
    let mut pass = true;
    let mut out = serde_json::Map::new();
    // Isolated executor references (k=1 on the batch route).
    let mut refs_b: Vec<Option<Trace>> = Vec::with_capacity(fx.len());
    let mut isolated = Vec::new();
    for (i, f) in fx.iter().enumerate() {
        let dir = args.artifacts.join("executor_k1").join(f.name());
        fs::create_dir_all(&dir)?;
        let trace = Trace { logits: vec![], hidden: vec![], committed: vec![], position: 0, pending_seed: 0, states: vec![], prefill_chunks: vec![] };
        let mut reqs = [Req::new(i, 1000 + i as u64, 1, i % WIDTH, steps + 1, Sink::Record { dir, trace })];
        let none: [Option<Trace>; 0] = [];
        let r = drive(ctx, &mut store, fx, &mut reqs, &Refs { b: &none, a: refs_a }, steps, None);
        let [req] = reqs;
        match (r, req.sink) {
            (Ok(()), Sink::Record { trace, .. }) => {
                // Isolated executor vs singleton route (route exactness at k=1).
                let mut d = Diff::default();
                let t = &refs_a[i];
                for (j, p) in trace.logits.iter().enumerate() {
                    d.check(&format!("logits_{j:04}"), &fs::read(&t.logits[j])?, &fs::read(p)?);
                }
                for (j, p) in trace.hidden.iter().enumerate() {
                    d.check(&format!("hidden_{:04}", j + 1), &fs::read(&t.hidden[j])?, &fs::read(p)?);
                }
                for (n, p) in &trace.states {
                    let (_, q) = t.states.iter().find(|(m, _)| m == n).ok_or("missing state")?;
                    d.check(n, &fs::read(q)?, &fs::read(p)?);
                }
                d.ids("committed_ids", &t.committed, &trace.committed);
                d.ids("position", &[t.position as u32], &[trace.position as u32]);
                d.ids("pending_seed", &[t.pending_seed], &[trace.pending_seed]);
                eprintln!("executor k1 {}: vs singleton route exact={}", f.name(), d.exact());
                isolated.push(json!({
                    "fixture": f.name(), "supported": true, "vs_singleton_route": d.json(),
                    "prefill_chunks_singleton": t.prefill_chunks, "prefill_chunks_executor": trace.prefill_chunks,
                }));
                refs_b.push(Some(trace));
            }
            (Err(e), _) => {
                eprintln!("executor k1 {}: REFUSED/FAILED: {e}", f.name());
                isolated.push(json!({"fixture": f.name(), "supported": false, "error": e.to_string()}));
                refs_b.push(None);
            }
            _ => unreachable!(),
        }
    }
    out.insert("executor_k1".into(), Value::Array(isolated));
    let refs = Refs { b: &refs_b, a: refs_a };
    let supported: Vec<usize> = (0..fx.len()).filter(|&i| refs_b[i].is_some()).collect();
    let mut cases = Vec::new();
    let mut route_exact_all = true;
    let mut run_case = |ctx: &mut Ctx, store: &mut Qwen35VmmStore, name: String, mut reqs: Vec<Req>, controls: bool, cases: &mut Vec<Value>, pass: &mut bool| -> Result<()> {
        let mut map = serde_json::Map::new();
        let r = drive(ctx, store, fx, &mut reqs, &refs, steps, controls.then_some(&mut map));
        let (bleed, route, mut j) = case_json(&name, fx, &reqs);
        route_exact_all &= route;
        let ctl_ok = map.values().all(|v| v["ok"].as_bool().unwrap_or(true));
        if controls {
            j["executor_controls"] = Value::Object(map);
        }
        if let Err(e) = &r {
            j["error"] = json!(e.to_string());
        }
        *pass &= r.is_ok() && bleed && ctl_ok && (!controls || j["executor_controls"].as_object().is_some_and(|m| m.len() >= 6));
        cases.push(j);
        Ok(())
    };
    // k = 1..8 with unequal prompts/contexts; request r uses fixture r.
    for &k in &args.ks {
        let chosen: Vec<usize> = (0..k).filter(|i| refs_b[*i].is_some()).collect();
        let refused: Vec<String> = (0..k).filter(|i| refs_b[*i].is_none()).map(|i| fx[i].name()).collect();
        if !refused.is_empty() {
            cases.push(json!({"case": format!("k{k}"), "refused_fixtures": refused, "note": "executor refused these fixtures in isolation; case runs the remaining requests"}));
            pass = false;
        }
        if chosen.is_empty() {
            continue;
        }
        // max_tokens clip: unequal exits (request r stops 3r picks early).
        let reqs = chosen.iter().map(|&i| Req::new(i, (k * 100 + i) as u64, 1, i, (steps + 1).saturating_sub(3 * i).max(2), compare_sink(i))).collect();
        run_case(ctx, &mut store, format!("k{k}"), reqs, false, &mut cases, &mut pass)?;
    }
    if supported.len() >= 2 {
        let (s0, s1) = (supported[0], supported[1]);
        // Stop id inside the stream: first pick equal to the reference's 6th.
        let t = refs_b[s0].as_ref().unwrap();
        let picks: Vec<u32> = t.committed.iter().copied().chain([t.pending_seed]).collect();
        let stop = picks[5];
        let stop_at = picks.iter().position(|&p| p == stop).unwrap() + 1;
        // Repeated in this process (`--stop-repeats`): retire of a stopped
        // request while its peer is mid-prefill must reproduce every time.
        for rep in 0..args.stop_repeats {
            let tag = 9001 + 10 * rep as u64;
            let mut a = Req::new(s0, tag, 1, 2, steps + 1, compare_sink(s0));
            a.stop = Some(stop);
            let b = Req::new(s1, tag + 1, 1, 5, steps + 1, compare_sink(s1));
            let name = if rep == 0 { "stop_id".to_string() } else { format!("stop_id_rep{rep}") };
            eprintln!("batch {name}: free_vram before={}", free_vram(&ctx.gpu)?);
            run_case(ctx, &mut store, name.clone(), vec![a, b], false, &mut cases, &mut pass)?;
            let got = cases.last().unwrap()["requests"][0].clone();
            let ok = got["finish"] == json!("stop") && got["committed"] == json!(stop_at);
            eprintln!("batch {name}: finish={} committed={} expected {stop_at} -> {}", got["finish"], got["committed"], if ok { "OK" } else { "FAIL" });
            pass &= ok;
        }
        // Regression: stop_id under memory pressure (the ks56 failure at
        // 34c1ee400b). Ballast lowers the singleton prefill ceiling to <=1024
        // so r1's 8192 prompt runs in several chunks and r0's stop+retire
        // lands between two of them (r0 picks 0..5 on steps 1..6, retired
        // before step 7 = r1's sixth chunk). Exercised only if r1 actually
        // ran more than five chunks of <=1024 rows.
        for rep in 0..args.stop_repeats {
            if fx[s1].prefix <= 5 * 1024 {
                // Needs a peer prompt of >5 chunks at <=1024 rows (the default
                // contexts give 8192); recorded, not silently passed.
                cases.push(json!({"case": "stop_id_pressure", "skipped": format!("peer prompt {} rows <= 5120", fx[s1].prefix)}));
                eprintln!("batch stop_id_pressure: SKIPPED (peer prompt {} rows)", fx[s1].prefix);
                break;
            }
            let name = if rep == 0 { "stop_id_pressure".to_string() } else { format!("stop_id_pressure_rep{rep}") };
            let tag = 9051 + 10 * rep as u64;
            let mut ballast: Vec<GpuTensor> = Vec::new();
            let mut ceilings = Vec::new();
            let limit = |ctx: &Ctx| -> Result<usize> {
                let b = &ctx.b;
                Ok(qwen35::ordinary_prefill_chunk_limit(&ctx.gpu, &b.weights, &b.config, &b.dn_state, &b.kv_cache, None)?)
            };
            let mut ceiling = limit(ctx)?;
            ceilings.push(ceiling);
            while ceiling > 1024 && ballast.len() < 128 && free_vram(&ctx.gpu)? > (1usize << 30) {
                ballast.push(ctx.gpu.zeros(&[256 << 20], DType::Raw)?);
                ceiling = limit(ctx)?;
                ceilings.push(ceiling);
            }
            let ballast_bytes = ballast.len() * (256 << 20);
            eprintln!("batch {name}: ballast={ballast_bytes} ceiling={ceiling} free_vram={}", free_vram(&ctx.gpu)?);
            let mut a = Req::new(s0, tag, 1, 2, steps + 1, compare_sink(s0));
            a.stop = Some(stop);
            let b = Req::new(s1, tag + 1, 1, 5, steps + 1, compare_sink(s1));
            let res = run_case(ctx, &mut store, name.clone(), vec![a, b], false, &mut cases, &mut pass);
            for t in ballast {
                ctx.gpu.free_tensor(t)?;
            }
            res?;
            let j = cases.last_mut().unwrap();
            let chunks: Vec<u64> = j["requests"][1]["prefill_chunks"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default();
            let exercised = chunks.len() > 5 && chunks.iter().all(|&c| c <= 1024);
            let ok = j["requests"][0]["finish"] == json!("stop") && j["requests"][0]["committed"] == json!(stop_at);
            j["ballast_bytes"] = json!(ballast_bytes);
            j["ceilings"] = json!(ceilings);
            j["pressure_exercised"] = json!(exercised);
            eprintln!("batch {name}: r1 chunks={chunks:?} exercised={exercised} stop_ok={ok}");
            pass &= exercised && ok;
        }
        // Cancel mid-stream: the survivor must stay byte-exact.
        let mut a = Req::new(s0, 9101, 1, 0, steps + 1, compare_sink(s0));
        a.cancel_at = Some(9);
        let b = Req::new(s1, 9102, 1, 1, steps + 1, compare_sink(s1));
        run_case(ctx, &mut store, "cancel".into(), vec![a, b], false, &mut cases, &mut pass)?;
        // Executor controls (row_slot / epoch / rejected tail / stale commit /
        // poison) need both requests decoding together: the two shortest.
        let mut by_len = supported.clone();
        by_len.sort_by_key(|&i| fx[i].prefix);
        let (c0, c1) = (by_len[0], by_len[1]);
        let a = Req::new(c0, 9201, 1, 3, steps + 1, compare_sink(c0));
        let b = Req::new(c1, 9202, 1, 6, steps + 1, compare_sink(c1));
        run_case(ctx, &mut store, "controls".into(), vec![a, b], true, &mut cases, &mut pass)?;
        // Slot wrap/reuse: the same slots now hold new epochs (generation 2)
        // with swapped fixtures; must still match their isolated references.
        let a = Req::new(s1, 9201, 2, 3, steps + 1, compare_sink(s1));
        let b = Req::new(s0, 9202, 2, 6, steps + 1, compare_sink(s0));
        run_case(ctx, &mut store, "slot_reuse".into(), vec![a, b], false, &mut cases, &mut pass)?;
    } else {
        pass = false;
    }
    // Verify rows belong to slice 2: record the executor's current answer.
    {
        let probe = supported.first().copied().unwrap_or(0);
        let r = Req::new(probe, 9301, 1, 0, steps + 1, compare_sink(probe));
        admit(ctx, &mut store, fx, &r)?;
        let mut p = build_plan(ctx, &store, fx, std::slice::from_ref(&r))?;
        p.requests[0].kind = RequestStepKind::Verify { draft_len: 0 };
        let Ctx { gpu, b } = ctx;
        let res = store.executor(&b.weights, &b.config, &b.scratch).provision_step(gpu, &p);
        if res.is_ok() {
            store.abort_step(&p);
        }
        retire(ctx, &mut store, &r.epoch)?;
        out.insert("spec_verify_probe".into(), json!({
            "accepted_by_executor": res.is_ok(), "error": res.err(),
            "note": "full/partial accept, EOS-in-draft and spec max_tokens clipping need Verify rows (Slice2A); not certified until accepted",
        }));
    }
    if args.drift_cycles > 0 && supported.len() >= 2 {
        let (drift, ok) = drift_check(ctx, &mut store, fx, &refs, &supported, args.drift_cycles)?;
        out.insert("memory_drift".into(), drift);
        pass &= ok;
    }
    out.insert("cases".into(), Value::Array(cases));
    let k1_route = out["executor_k1"]
        .as_array()
        .is_some_and(|v| v.iter().all(|e| e["vs_singleton_route"]["exact"] == json!(true)));
    out.insert("route_exact_vs_singleton".into(), json!(route_exact_all && k1_route));
    let receipt = store.receipt()?;
    out.insert("store_receipt".into(), json!({
        "kv_backend": receipt.kv_backend, "kv_mode": receipt.kv_mode, "max_seq_bound": receipt.max_seq_bound,
        "mapped_bytes": receipt.mapped_bytes, "mapped_high_water": receipt.mapped_high_water,
    }));
    store.free_gpu(&mut ctx.gpu)?;
    Ok((Value::Object(out), pass))
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if args.artifacts.exists() {
        return Err(format!("artifacts dir {} exists; refusing to overwrite", args.artifacts.display()).into());
    }
    fs::create_dir_all(&args.artifacts)?;
    let (mut ctx, tok, ack) = load(&args.model)?;
    let max_prefix = args.contexts.iter().copied().chain([args.short]).max().unwrap();
    if max_prefix + args.steps + 8 > ack["max_seq_bound"].as_u64().unwrap() as usize {
        return Err(format!("context {max_prefix}+{} exceeds loaded max_seq bound", args.steps).into());
    }
    let fx = fixtures(&args, &tok)?;
    let mut report = json!({
        "model": args.model,
        "ks": args.ks,
        "contexts": args.contexts,
        "short": args.short,
        "steps": args.steps,
        "loaded_ack": ack,
        "artifacts": args.artifacts,
    });

    // ── Singleton phase: references + determinism self-test ──────────
    let mut refs = Vec::with_capacity(fx.len());
    let mut singleton = Vec::new();
    let mut pass = true;
    for f in &fx {
        let t0 = std::time::Instant::now();
        let tr = record(&mut ctx, f, args.steps, &args.artifacts.join("singleton").join(f.name()))?;
        let rec_s = t0.elapsed().as_secs_f64();
        let again = replay(&mut ctx, f, args.steps, &tr, Perturb::None)?;
        pass &= again.exact();
        eprintln!("singleton {}: determinism exact={} ({:.1}s)", f.name(), again.exact(), rec_s);
        singleton.push(json!({
            "fixture": f.name(),
            "prefix": f.prefix,
            "prompt_sha256": sha(&f.tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>()),
            "position": tr.position,
            "pending_seed": tr.pending_seed,
            "committed_ids": tr.committed,
            "final_logits_sha256": sha(&fs::read(tr.logits.last().unwrap())?),
            "record_seconds": rec_s,
            "determinism": again.json(),
        }));
        refs.push(tr);
    }
    report["singleton"] = Value::Array(singleton);

    // ── Negative/positive controls on the shortest fixture ───────────
    let ci = (0..fx.len()).min_by_key(|&i| fx[i].prefix).unwrap();
    let (f, tr) = (&fx[ci], &refs[ci]);
    let mut controls = serde_json::Map::new();
    let mut control = |name: &str, d: Diff, must_match: bool| {
        let ok = d.exact() == must_match;
        pass &= ok;
        eprintln!("control {name}: exact={} expected_exact={must_match} -> {}", d.exact(), if ok { "OK" } else { "FAIL" });
        controls.insert(name.into(), json!({"expected_exact": must_match, "ok": ok, "diff": d.json()}));
    };
    let n = args.steps;
    control("neg_row_slot", replay(&mut ctx, f, n, tr, Perturb::RowSlot)?, false);
    control("neg_rejected_tail_read", replay(&mut ctx, f, n, tr, Perturb::RejectedTailRead)?, false);
    control("pos_rejected_tail_masked", replay(&mut ctx, f, n, tr, Perturb::RejectedTailMasked)?, true);
    let same = (0..fx.len()).find(|&i| i != ci && fx[i].prefix == f.prefix);
    if let Some(other) = same.or_else(|| (0..fx.len()).find(|&i| i != ci)) {
        // Epoch control: request ci replayed against another request's reference.
        let mut d = replay(&mut ctx, f, n, &refs[other], Perturb::None)?;
        if fx[other].prefix != f.prefix {
            d.differing_items.push("prefix length differs (epoch mismatch)".into());
        }
        control("neg_epoch", d, false);
    }
    report["controls"] = Value::Object(controls);
    if args.batch {
        let (batch, ok) = batch_phase(&mut ctx, &args, &fx, &refs)?;
        // Isolation (batch == isolated executor, controls) and route
        // exactness (executor == singleton route, §4.3 exact default) are
        // separate verdicts; both are required for the exact-route gate.
        let route = batch["route_exact_vs_singleton"].as_bool() == Some(true);
        report["pass_batch_isolation"] = json!(ok);
        report["route_exact_vs_singleton"] = json!(route);
        report["batch"] = batch;
        pass &= ok && route;
    }
    report["pass"] = json!(pass);
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("wrote {} pass={pass}", args.out.display());
    if !pass {
        return Err("oracle FAILED (see report)".into());
    }
    Ok(())
}
