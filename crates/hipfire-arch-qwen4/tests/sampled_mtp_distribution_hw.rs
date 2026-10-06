// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Sampled Flash-Next MTP against AR sampling, in distribution.
//!
//! `#[ignore]`d: it needs a real HIP GPU and the canonical
//! `qwen3.8-flash-next-gptq3.mq4` artifact named by
//! `HIPFIRE_MTP_IDENTITY_MODEL`.
//!
//! The serve battery's prose prompt (thinking off, rendered as `hipfire
//! serve` renders one user message) is decoded from a cold state
//! `HIPFIRE_SAMPLED_MTP_TRIALS` times (default 2000) per arm, each trial with
//! its own seed, keeping the first `TOKENS` generated IDs:
//!
//! - `ar`: `forward_chunk_final` / `forward_token_or_argmax` with the
//!   production host sampler (`sampler::sample_cpu`);
//! - `batched`: sampled native MTP, `HIPFIRE_MTP_INCREMENTAL=0`;
//! - `interleaved`: sampled native MTP, `HIPFIRE_MTP_INCREMENTAL=1`.
//!
//! Trial seeds are SplitMix64-scrambled (`trial_seed`), distinct per arm:
//! the host AR sampler's xorshift32 maps consecutive seeds to nearly equal
//! first draws, which production avoids by hashing its request seeds.
//!
//! Each MTP arm's per-position marginals and its (t1, t2) joint are compared
//! with AR's by a two-sample chi-square homogeneity test (cells with fewer
//! than 5 expected draws pooled); every p-value must exceed `ALPHA`. Each MTP
//! arm also replays its first `REPLAYS` seeds and must emit the same IDs.
//! Temperature / top_p / top_k / min_p come from `HIPFIRE_SAMPLED_MTP_TEMP` /
//! `HIPFIRE_SAMPLED_MTP_TOP_P` / `HIPFIRE_SAMPLED_MTP_TOP_K` (unset = absent)
//! / `HIPFIRE_SAMPLED_MTP_MIN_P` (default 1.0 / 0.95 / absent / 0); both arms
//! receive them as a request would. Per-arm IDs and the report go to
//! `HIPFIRE_SAMPLED_MTP_OUT`; an arm whose `<arm>.ids.json` is already there
//! is reused, so the arms can run first as separate time-boxed invocations of
//! `sampled_mtp_distribution_arm` (set `HIPFIRE_SAMPLED_MTP_ARM`,
//! `HIPFIRE_MTP_SAMPLED=1` and the arm's `HIPFIRE_MTP_INCREMENTAL` yourself).
//! The arms inherit `HIPFIRE_MTP_SAMPLED_MODE`, so the test checks whichever
//! sampled verifier that selects (unset: speculative rejection sampling).
//!
//! `naive_sampled_mtp_emits_seeded_ar_ids` checks the stronger property of
//! `HIPFIRE_MTP_SAMPLED_MODE=naive`: with the same seed, every MTP route
//! (`batched`, `interleaved`, `adaptive` = route knob unset) emits the AR
//! arm's exact IDs. Cases: the serve bench's `lru_cache_pep8_strict` and
//! `prose_river_short` prompts × T0.7/top_p 0.8/top_k 20, T1.0/top_p 0.95
//! and T0.8/top_p 0.95/top_k 40/min_p 0.05 × `ID_SEEDS` seeds, `ID_TOKENS`
//! IDs each (cut after EOS). Arms run as fresh processes of
//! `sampled_mtp_identity_arm`; an arm whose `<arm>.identity.json` already
//! exists in `HIPFIRE_SAMPLED_MTP_OUT` is reused.

use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::mtp_spec::Qwen4MtpDrafter;
use hipfire_arch_qwen4::{admit_hfqm_artifact, Qwen4KvBackend};
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::prompt_frame::{JinjaChatFrame, Message, Role};
use hipfire_runtime::sampler::{sample_cpu, SamplerConfig};
use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::{DType, Gpu};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MODEL_ENV: &str = "HIPFIRE_MTP_IDENTITY_MODEL";
const OUT_ENV: &str = "HIPFIRE_SAMPLED_MTP_OUT";
const ARM_ENV: &str = "HIPFIRE_SAMPLED_MTP_ARM";
const TRIALS_ENV: &str = "HIPFIRE_SAMPLED_MTP_TRIALS";
const TEMP_ENV: &str = "HIPFIRE_SAMPLED_MTP_TEMP";
const TOP_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_P";
const TOP_K_ENV: &str = "HIPFIRE_SAMPLED_MTP_TOP_K";
const MIN_P_ENV: &str = "HIPFIRE_SAMPLED_MTP_MIN_P";
const ARM_TEST: &str = "sampled_mtp_distribution_arm";

/// The serve battery's prose prompt (`scripts/serve_harness.py`).
const PROSE: &str = "Write a four-sentence story about a lighthouse keeper who finds something unexpected washed up on the rocks.";
const TOKENS: usize = 4;
const MAX_SEQ: usize = 4096;
const MTP_K: usize = 3;
const REPLAYS: usize = 5;
const ALPHA: f64 = 1e-3;
const ARMS: [&str; 3] = ["ar", "batched", "interleaved"];

/// SplitMix64 of `arm_salt ^ trial`, never zero.
fn trial_seed(arm_salt: u64, trial: usize) -> u64 {
    let mut z = (arm_salt ^ trial as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)).max(1)
}

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn model_path() -> PathBuf {
    let path = std::env::var_os(MODEL_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{MODEL_ENV} must name qwen3.8-flash-next-gptq3.mq4"));
    assert!(
        path.is_file(),
        "{MODEL_ENV}={} is not a file",
        path.display()
    );
    path
}

fn out_dir() -> PathBuf {
    let dir = std::env::var_os(OUT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("qwen4-sampled-mtp-{}", std::process::id()))
        });
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}

/// ln Γ(x), Lanczos (g = 7, n = 9).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized upper incomplete gamma Q(a, x) (series / continued fraction).
fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let ln_pre = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let (mut sum, mut term, mut n) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            n += 1.0;
            term *= x / n;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * ln_pre.exp()
    } else {
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        ln_pre.exp() * h
    }
}

/// Two-sample chi-square homogeneity test of category counts `a` vs `b`.
/// Categories whose expected count is under 5 in either sample are pooled.
/// Returns `(statistic, degrees of freedom, p-value)`.
fn homogeneity<K: std::hash::Hash + Eq + Clone>(a: &[K], b: &[K]) -> (f64, usize, f64) {
    let mut cells: HashMap<K, (f64, f64)> = HashMap::new();
    for key in a {
        cells.entry(key.clone()).or_default().0 += 1.0;
    }
    for key in b {
        cells.entry(key.clone()).or_default().1 += 1.0;
    }
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let n = na + nb;
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut pooled = (0.0, 0.0);
    for &(x, y) in cells.values() {
        let total = x + y;
        if total * na.min(nb) / n < 5.0 {
            pooled.0 += x;
            pooled.1 += y;
        } else {
            kept.push((x, y));
        }
    }
    if pooled.0 + pooled.1 > 0.0 {
        kept.push(pooled);
    }
    let mut stat = 0.0;
    for &(x, y) in &kept {
        let total = x + y;
        let (ea, eb) = (total * na / n, total * nb / n);
        stat += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    let df = kept.len().saturating_sub(1).max(1);
    (stat, df, gamma_q(df as f64 / 2.0, stat / 2.0))
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn sampled_mtp_matches_ar_distribution_on_flash_next() {
    let model = model_path();
    let dir = out_dir();
    let exe = std::env::current_exe().expect("test binary path");
    let mut samples: Vec<Vec<Vec<u32>>> = Vec::new();
    for arm in ARMS {
        let ids_path = dir.join(format!("{arm}.ids.json"));
        if ids_path.exists() {
            eprintln!("reusing {}", ids_path.display());
            samples
                .push(serde_json::from_slice(&fs::read(&ids_path).unwrap()).expect("arm ids json"));
            continue;
        }
        let mut child = Command::new(&exe);
        child
            .args([
                "--exact",
                ARM_TEST,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(ARM_ENV, arm)
            .env(MODEL_ENV, &model)
            .env(OUT_ENV, &dir)
            .env("HIPFIRE_MTP_SAMPLED", "1")
            .env_remove("HIPFIRE_MTP_INCREMENTAL")
            .stdin(Stdio::null());
        match arm {
            "batched" => {
                child.env("HIPFIRE_MTP_INCREMENTAL", "0");
            }
            "interleaved" => {
                child.env("HIPFIRE_MTP_INCREMENTAL", "1");
            }
            _ => {}
        }
        let output = child
            .output()
            .unwrap_or_else(|e| panic!("spawn {arm}: {e}"));
        fs::write(dir.join(format!("{arm}.out")), &output.stdout).unwrap();
        fs::write(dir.join(format!("{arm}.err")), &output.stderr).unwrap();
        assert!(
            output.status.success(),
            "{arm} arm failed ({}); logs in {}",
            output.status,
            dir.display()
        );
        let arm_samples: Vec<Vec<u32>> =
            serde_json::from_slice(&fs::read(&ids_path).expect("arm ids")).expect("arm ids json");
        samples.push(arm_samples);
    }
    let mut report = String::new();
    let mut failures = Vec::new();
    let ar = &samples[0];
    for (arm, mtp) in ARMS.iter().zip(&samples).skip(1) {
        for position in 0..TOKENS {
            let a: Vec<u32> = ar.iter().map(|ids| ids[position]).collect();
            let b: Vec<u32> = mtp.iter().map(|ids| ids[position]).collect();
            let (stat, df, p) = homogeneity(&a, &b);
            let line = format!(
                "{arm} vs ar, token {}: chi2={stat:.2} df={df} p={p:.4} (n={}+{})\n",
                position + 1,
                a.len(),
                b.len()
            );
            if p < ALPHA {
                failures.push(line.clone());
            }
            report.push_str(&line);
        }
        let a: Vec<(u32, u32)> = ar.iter().map(|ids| (ids[0], ids[1])).collect();
        let b: Vec<(u32, u32)> = mtp.iter().map(|ids| (ids[0], ids[1])).collect();
        let (stat, df, p) = homogeneity(&a, &b);
        let line = format!("{arm} vs ar, (t1,t2) joint: chi2={stat:.2} df={df} p={p:.4}\n");
        if p < ALPHA {
            failures.push(line.clone());
        }
        report.push_str(&line);
    }
    eprint!("{report}");
    fs::write(dir.join("report.txt"), &report).unwrap();
    assert!(
        failures.is_empty(),
        "sampled MTP differs from AR in distribution (alpha {ALPHA}; logs in {}):\n{}",
        dir.display(),
        failures.concat()
    );
}

/// One arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of sampled_mtp_matches_ar_distribution_on_flash_next"]
fn sampled_mtp_distribution_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} arm: {error}");
    }
}

/// A loaded Flash-Next target with its tokenizer and chat template.
struct Loaded {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    tokenizer: Tokenizer,
    template: String,
    state: String,
}

fn load(model: &Path) -> Result<Loaded, String> {
    let mut hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
    let receipt = admit_hfqm_artifact(&hfq).map_err(|e| e.to_string())?;
    let mut gpu = Gpu::init().map_err(|e| e.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(|e| e.to_string())?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            source
                .tensor_range(&entry.name)
                .map_err(|e| e.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(|e| e.to_string())?;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        &gpu,
        &receipt.config,
    )?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        transaction,
        &receipt.placements,
        &mut gpu,
        MAX_SEQ,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(|e| e.to_string())?;
    bundle
        .attach_forward(&mut gpu, MAX_SEQ)
        .map_err(|e| e.to_string())?;
    Ok(Loaded {
        gpu,
        bundle,
        tokenizer,
        template,
        state: format!("{state_format:?}"),
    })
}

impl Loaded {
    /// `text` as one user message, thinking off, rendered as `hipfire serve`
    /// renders it.
    fn render(&self, text: &str) -> Result<Vec<u32>, String> {
        let user = Message {
            role: Role::User,
            content: text.to_string(),
            reasoning_content: None,
            name: None,
            rendered_name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_plan: String::new(),
        };
        let rendered = JinjaChatFrame {
            tokenizer: &self.tokenizer,
            template: &self.template,
            system: None,
            user: text,
            enable_thinking: false,
            bos_token: None,
            reasoning_strength: None,
            reasoning_effort: None,
        }
        .render_messages(&[user], None, None)?;
        Ok(self.tokenizer.encode(&rendered))
    }

    /// Seeded AR from a cold state with the production host sampler: up to
    /// `tokens` IDs, cut after EOS.
    fn ar_ids(
        &mut self,
        logits: &rdna_compute::GpuTensor,
        prompt: &[u32],
        cfg: &SamplerConfig,
        seed: u32,
        tokens: usize,
    ) -> Result<Vec<u32>, String> {
        let eos = self.bundle.config.eos_token_id;
        hipfire_runtime::llama::reset_cpu_sampler_rng(seed);
        self.bundle
            .reset(&mut self.gpu)
            .map_err(|e| e.to_string())?;
        self.bundle
            .forward_chunk_final(&mut self.gpu, prompt, logits, None)
            .map_err(|e| e.to_string())?;
        let mut history = prompt.to_vec();
        let mut ids = Vec::with_capacity(tokens);
        loop {
            let mut row = self.gpu.download_f32(logits).map_err(|e| e.to_string())?;
            let token = sample_cpu(&mut row, &history, cfg);
            ids.push(token);
            history.push(token);
            if ids.len() == tokens || token == eos {
                return Ok(ids);
            }
            self.bundle
                .forward_token_or_argmax(&mut self.gpu, Some(token), logits)
                .map_err(|e| e.to_string())?;
        }
    }

    /// Sampled native MTP from a cold state: up to `tokens` IDs, cut after
    /// EOS. Returns the IDs and the windows' (drafts, accepted) totals.
    fn mtp_ids(
        &mut self,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
        cfg: SpecRequestConfig,
        tokens: usize,
    ) -> Result<(Vec<u32>, usize, usize), String> {
        let eos = self.bundle.config.eos_token_id;
        drafter.configure_request(cfg);
        let mut seed_token = drafter.mtp_prefill(
            &mut self.gpu,
            &mut self.bundle,
            prompt,
            prompt,
            0,
            false,
            &|| false,
        )?;
        let mut ids = vec![seed_token];
        let (mut drafted, mut accepted) = (0, 0);
        while ids.len() < tokens && seed_token != eos {
            let position = self.bundle.state.position;
            let window = drafter.mtp_step(
                &mut self.gpu,
                &mut self.bundle,
                position,
                seed_token,
                &ids,
                MTP_K.min(tokens - ids.len()),
                eos,
                None,
            )?;
            drafted += window.drafts_generated;
            accepted += window.accepted;
            seed_token = *window.committed.last().ok_or("empty MTP window")?;
            ids.extend_from_slice(&window.committed);
        }
        ids.truncate(tokens);
        Ok((ids, drafted, accepted))
    }
}

fn ar_sampler(temp: f32, top_p: f32, top_k: Option<u32>, min_p: f32) -> SamplerConfig {
    SamplerConfig {
        temperature: temp,
        top_p,
        repeat_penalty: 1.0,
        repeat_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        blocked_tokens: Vec::new(),
        top_k,
        min_p: (min_p > 0.0).then_some(min_p),
    }
}

fn spec_request(
    temp: f32,
    top_p: f32,
    top_k: Option<u32>,
    min_p: f32,
    seed: u64,
) -> SpecRequestConfig {
    SpecRequestConfig {
        temp,
        top_p,
        top_k,
        min_p,
        cactus_delta: 0.0,
        rng_seed: seed,
        allow_ngram_modifier: false,
    }
}

fn new_drafter(loaded: &mut Loaded) -> Result<Qwen4MtpDrafter, String> {
    loaded
        .bundle
        .attach_mtp(&mut loaded.gpu, MAX_SEQ)
        .map_err(|e| e.to_string())?;
    let drafter = Qwen4MtpDrafter::new(MTP_K, MAX_SEQ, None);
    if !drafter.supports_temp_verify() {
        return Err("HIPFIRE_MTP_SAMPLED did not enable sampled verification".into());
    }
    Ok(drafter)
}

fn run_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let trials: usize = env_or(TRIALS_ENV, 2000);
    let temp: f32 = env_or(TEMP_ENV, 1.0);
    let top_p: f32 = env_or(TOP_P_ENV, 0.95);
    let top_k: Option<u32> = std::env::var(TOP_K_ENV)
        .ok()
        .map(|v| v.parse().map_err(|e| format!("{TOP_K_ENV}={v}: {e}")))
        .transpose()?;
    let min_p: f32 = env_or(MIN_P_ENV, 0.0);
    let mut loaded = load(model)?;
    let prompt = loaded.render(PROSE)?;
    let eos = loaded.bundle.config.eos_token_id;
    println!(
        "ARM {arm} arch={} prompt_tokens={} trials={trials} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} state={}",
        loaded.gpu.arch,
        prompt.len(),
        loaded.state
    );
    let started = std::time::Instant::now();
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(trials);
    if arm == "ar" {
        let logits = loaded
            .gpu
            .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
            .map_err(|e| e.to_string())?;
        let cfg = ar_sampler(temp, top_p, top_k, min_p);
        for trial in 0..trials {
            let seed = trial_seed(0xA11, trial) as u32;
            samples.push(loaded.ar_ids(&logits, &prompt, &cfg, seed, TOKENS)?);
        }
        loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        let mut drafter = new_drafter(&mut loaded)?;
        let mut run = |loaded: &mut Loaded, seed: u64| {
            let cfg = spec_request(temp, top_p, top_k, min_p, seed);
            loaded
                .mtp_ids(&mut drafter, &prompt, cfg, TOKENS)
                .map(|(ids, _, _)| ids)
        };
        for trial in 0..trials {
            samples.push(run(&mut loaded, trial_seed(0x5EC, trial))?);
        }
        for trial in 0..REPLAYS.min(trials) {
            let replay = run(&mut loaded, trial_seed(0x5EC, trial))?;
            if replay != samples[trial] {
                return Err(format!(
                    "seed {} replayed {replay:?}, first run {:?}",
                    trial + 1,
                    samples[trial]
                ));
            }
        }
        println!(
            "ARM {arm} replayed {} seeds identically",
            REPLAYS.min(trials)
        );
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    // Positions after an EOS read as EOS.
    for ids in &mut samples {
        ids.resize(TOKENS, eos);
    }
    println!(
        "ARM {arm} {trials} trials in {:.1}s; first sample: {:?}",
        started.elapsed().as_secs_f64(),
        loaded.tokenizer.decode(&samples[0])
    );
    fs::write(
        dir.join(format!("{arm}.ids.json")),
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}

const ID_ARM_TEST: &str = "sampled_mtp_identity_arm";
const ID_ARMS: [&str; 4] = ["ar", "batched", "interleaved", "adaptive"];
const ID_TOKENS: usize = 128;
const ID_SEEDS: usize = 6;
/// (name, temperature, top_p, top_k, min_p).
const ID_SAMPLINGS: [(&str, f32, f32, Option<u32>, f32); 3] = [
    ("T0.7/p0.8/k20", 0.7, 0.8, Some(20), 0.0),
    ("T1.0/p0.95", 1.0, 0.95, None, 0.0),
    ("T0.8/p0.95/k40/minp0.05", 0.8, 0.95, Some(40), 0.05),
];
const ID_PROMPTS: [(&str, &str); 2] = [
    (
        "lru_cache",
        include_str!("../../../benchmarks/prompts/lru_cache_pep8_strict.txt"),
    ),
    (
        "prose_river",
        include_str!("../../../benchmarks/prompts/prose_river_short.txt"),
    ),
];

/// One identity case's result in an arm.
struct IdCase {
    case: String,
    ids: Vec<u32>,
    drafted: usize,
    accepted: usize,
}

impl IdCase {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "case": self.case,
            "ids": self.ids,
            "drafted": self.drafted,
            "accepted": self.accepted,
        })
    }

    fn from_json(value: &serde_json::Value) -> Self {
        let count = |key: &str| value[key].as_u64().expect("identity count") as usize;
        Self {
            case: value["case"].as_str().expect("identity case").to_string(),
            ids: serde_json::from_value(value["ids"].clone()).expect("identity ids"),
            drafted: count("drafted"),
            accepted: count("accepted"),
        }
    }
}

#[test]
#[ignore = "requires a real HIP GPU and HIPFIRE_MTP_IDENTITY_MODEL (qwen3.8-flash-next-gptq3.mq4)"]
fn naive_sampled_mtp_emits_seeded_ar_ids() {
    let model = model_path();
    let dir = out_dir();
    let exe = std::env::current_exe().expect("test binary path");
    let mut arms: Vec<Vec<IdCase>> = Vec::new();
    for arm in ID_ARMS {
        let ids_path = dir.join(format!("{arm}.identity.json"));
        if !ids_path.exists() {
            let mut child = Command::new(&exe);
            child
                .args([
                    "--exact",
                    ID_ARM_TEST,
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(ARM_ENV, arm)
                .env(MODEL_ENV, &model)
                .env(OUT_ENV, &dir)
                .env("HIPFIRE_MTP_SAMPLED", "1")
                .env("HIPFIRE_MTP_SAMPLED_MODE", "naive")
                .env_remove("HIPFIRE_MTP_INCREMENTAL")
                .stdin(Stdio::null());
            match arm {
                "batched" => {
                    child.env("HIPFIRE_MTP_INCREMENTAL", "0");
                }
                "interleaved" => {
                    child.env("HIPFIRE_MTP_INCREMENTAL", "1");
                }
                _ => {}
            }
            let output = child
                .output()
                .unwrap_or_else(|e| panic!("spawn {arm}: {e}"));
            fs::write(dir.join(format!("{arm}.identity.out")), &output.stdout).unwrap();
            fs::write(dir.join(format!("{arm}.identity.err")), &output.stderr).unwrap();
            assert!(
                output.status.success(),
                "{arm} arm failed ({}); logs in {}",
                output.status,
                dir.display()
            );
        }
        let json: Vec<serde_json::Value> =
            serde_json::from_slice(&fs::read(&ids_path).expect("arm identity ids"))
                .expect("arm identity json");
        arms.push(json.iter().map(IdCase::from_json).collect());
    }
    let mut report = String::new();
    let mut failures = 0usize;
    let ar = &arms[0];
    for (arm, cases) in ID_ARMS.iter().zip(&arms).skip(1) {
        assert_eq!(cases.len(), ar.len(), "{arm}: case count");
        let mut same = 0usize;
        for (a, m) in ar.iter().zip(cases) {
            assert_eq!(a.case, m.case, "{arm}: case order");
            match a.ids.iter().zip(&m.ids).position(|(x, y)| x != y) {
                None if a.ids.len() == m.ids.len() => same += 1,
                first => {
                    failures += 1;
                    report.push_str(&format!(
                        "{arm} {}: first differs at {} (ar {} ids, mtp {} ids)\n",
                        m.case,
                        first.unwrap_or(a.ids.len().min(m.ids.len())),
                        a.ids.len(),
                        m.ids.len()
                    ));
                }
            }
        }
        let (drafted, accepted) = cases
            .iter()
            .fold((0, 0), |(d, a), c| (d + c.drafted, a + c.accepted));
        report.push_str(&format!(
            "{arm}: {same}/{} cases identical to AR; drafts {drafted}, accepted {accepted}\n",
            ar.len()
        ));
    }
    eprint!("{report}");
    fs::write(dir.join("identity-report.txt"), &report).unwrap();
    assert_eq!(
        failures,
        0,
        "naive sampled MTP left AR's seeded IDs (logs in {}):\n{report}",
        dir.display()
    );
}

/// One identity arm in a fresh process; spawned by the test above.
#[test]
#[ignore = "child process of naive_sampled_mtp_emits_seeded_ar_ids"]
fn sampled_mtp_identity_arm() {
    let Ok(arm) = std::env::var(ARM_ENV) else {
        return;
    };
    if let Err(error) = run_identity_arm(&arm, &model_path(), &out_dir()) {
        panic!("{arm} identity arm: {error}");
    }
}

fn run_identity_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let mut loaded = load(model)?;
    let started = std::time::Instant::now();
    let logits = loaded
        .gpu
        .zeros(&[loaded.bundle.config.vocab_size], DType::F32)
        .map_err(|e| e.to_string())?;
    let mut drafter = if arm == "ar" {
        None
    } else {
        Some(new_drafter(&mut loaded)?)
    };
    let mut out = Vec::new();
    for (prompt_name, text) in ID_PROMPTS {
        let prompt = loaded.render(text)?;
        for (sampling, temp, top_p, top_k, min_p) in ID_SAMPLINGS {
            for n in 0..ID_SEEDS {
                let seed = trial_seed(0x1D, n) as u32;
                let case = format!("{prompt_name} {sampling} seed#{n}");
                let (ids, drafted, accepted) = match drafter.as_mut() {
                    None => {
                        let cfg = ar_sampler(temp, top_p, top_k, min_p);
                        let ids = loaded.ar_ids(&logits, &prompt, &cfg, seed, ID_TOKENS)?;
                        (ids, 0, 0)
                    }
                    Some(drafter) => {
                        let cfg = spec_request(temp, top_p, top_k, min_p, seed as u64);
                        loaded.mtp_ids(drafter, &prompt, cfg, ID_TOKENS)?
                    }
                };
                println!(
                    "ARM {arm} {case}: {} ids, drafts {drafted}, accepted {accepted}",
                    ids.len()
                );
                out.push(IdCase {
                    case,
                    ids,
                    drafted,
                    accepted,
                });
            }
        }
    }
    println!(
        "ARM {arm} {} cases in {:.1}s (arch {}, state {})",
        out.len(),
        started.elapsed().as_secs_f64(),
        loaded.gpu.arch,
        loaded.state
    );
    fs::write(
        dir.join(format!("{arm}.identity.json")),
        serde_json::to_vec(&out.iter().map(IdCase::to_json).collect::<Vec<_>>())
            .map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if let Some(drafter) = drafter {
        Box::new(drafter).mtp_free(&mut loaded.gpu);
    }
    loaded.gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    loaded
        .bundle
        .free_gpu(&mut loaded.gpu)
        .map_err(|e| e.to_string())
}
