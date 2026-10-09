// SPDX-License-Identifier: Apache-2.0
// hipfire — see LICENSE and NOTICE in the project root.
//! VMM continuous-batch state oracle (PLAN §6 Slice2B, §7 acceptance item 2).
//!
//! cb_vmm_state_oracle <model> --ks 1,2,3,4,5,6,7,8 --contexts 512,8192,32768
//!     --steps 256 --out <absolute.json> [--short 256] [--artifacts <absolute dir>]
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

use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, LayerType, Qwen35Config};
use hipfire_arch_qwen35::{load_qwen35_bundle, Qwen35Bundle};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::loader_api::{CaskConfig, LoadCtx, ModelSource, SequenceHint, SpecLoadCfg};
use hipfire_runtime::tokenizer::Tokenizer;
use rdna_compute::{Gpu, GpuTensor};
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
                 --steps 256 --out <absolute.json> [--short 256] [--artifacts <absolute dir>]";
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
    Ok(Args { model, ks, contexts: contexts.ok_or(usage)?, short, steps, out, artifacts })
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
    let (kr, vr) = kv_row_bytes(kv)?;
    let mut out = Vec::new();
    let mut la = 0;
    for (layer, ty) in config.layer_types.iter().enumerate() {
        if *ty == LayerType::LinearAttention {
            let whole = |t: &GpuTensor| read_dev(gpu, t, 0, t.buf.size());
            out.push((format!("L{layer:02}.dn_s"), whole(&dn.s_matrices[la])?));
            if let Some(t) = dn.s_scales.get(la) {
                out.push((format!("L{layer:02}.dn_scales"), whole(t)?));
            }
            if let Some(t) = dn.s_ef_residual.get(la) {
                out.push((format!("L{layer:02}.dn_ef"), whole(t)?));
            }
            out.push((format!("L{layer:02}.conv"), whole(&dn.conv_states[la])?));
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
    ctx.prefill(&f.tokens, 0)?;
    let mut tr = Trace {
        logits: vec![],
        hidden: vec![],
        committed: vec![],
        position: f.prefix,
        pending_seed: 0,
        states: vec![],
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
    ctx.prefill(&f.tokens, 0)?;
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
    report["pass"] = json!(pass);
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    eprintln!("wrote {} pass={pass}", args.out.display());
    if !pass {
        return Err("oracle FAILED (see report)".into());
    }
    Ok(())
}
