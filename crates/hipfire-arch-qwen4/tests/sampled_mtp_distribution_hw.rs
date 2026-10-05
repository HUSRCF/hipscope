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

fn run_arm(arm: &str, model: &Path, dir: &Path) -> Result<(), String> {
    let trials: usize = env_or(TRIALS_ENV, 2000);
    let temp: f32 = env_or(TEMP_ENV, 1.0);
    let top_p: f32 = env_or(TOP_P_ENV, 0.95);
    let top_k: Option<u32> = std::env::var(TOP_K_ENV)
        .ok()
        .map(|v| v.parse().map_err(|e| format!("{TOP_K_ENV}={v}: {e}")))
        .transpose()?;
    let min_p: f32 = env_or(MIN_P_ENV, 0.0);
    let mut hfq = HfqFile::open(model).map_err(|e| e.to_string())?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(|e| e.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
    let user = Message {
        role: Role::User,
        content: PROSE.to_string(),
        reasoning_content: None,
        name: None,
        rendered_name: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        tool_plan: String::new(),
    };
    let rendered = JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system: None,
        user: PROSE,
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    }
    .render_messages(&[user], None, None)?;
    let prompt = tokenizer.encode(&rendered);
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
    let vocab = receipt.config.vocab_size;
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
    println!(
        "ARM {arm} arch={} prompt_tokens={} trials={trials} temp={temp} top_p={top_p} top_k={top_k:?} min_p={min_p} state={state_format:?}",
        gpu.arch,
        prompt.len()
    );
    let started = std::time::Instant::now();
    let mut samples: Vec<Vec<u32>> = Vec::with_capacity(trials);
    if arm == "ar" {
        let logits = gpu.zeros(&[vocab], DType::F32).map_err(|e| e.to_string())?;
        let cfg = SamplerConfig {
            temperature: temp,
            top_p,
            repeat_penalty: 1.0,
            repeat_window: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            blocked_tokens: Vec::new(),
            top_k,
            min_p: (min_p > 0.0).then_some(min_p),
        };
        let mut history = prompt.clone();
        for trial in 0..trials {
            hipfire_runtime::llama::reset_cpu_sampler_rng(trial_seed(0xA11, trial) as u32);
            bundle.reset(&mut gpu).map_err(|e| e.to_string())?;
            bundle
                .forward_chunk_final(&mut gpu, &prompt, &logits, None)
                .map_err(|e| e.to_string())?;
            history.truncate(prompt.len());
            let mut ids = Vec::with_capacity(TOKENS);
            loop {
                let mut row = gpu.download_f32(&logits).map_err(|e| e.to_string())?;
                let token = sample_cpu(&mut row, &history, &cfg);
                ids.push(token);
                history.push(token);
                if ids.len() == TOKENS {
                    break;
                }
                bundle
                    .forward_token_or_argmax(&mut gpu, Some(token), &logits)
                    .map_err(|e| e.to_string())?;
            }
            samples.push(ids);
        }
        gpu.free_tensor(logits).map_err(|e| e.to_string())?;
    } else {
        bundle
            .attach_mtp(&mut gpu, MAX_SEQ)
            .map_err(|e| e.to_string())?;
        let mut drafter = Qwen4MtpDrafter::new(MTP_K, MAX_SEQ);
        if !drafter.supports_temp_verify() {
            return Err("HIPFIRE_MTP_SAMPLED did not enable sampled verification".into());
        }
        let eos = bundle.config.eos_token_id;
        let mut run = |drafter: &mut Qwen4MtpDrafter, seed: u64| -> Result<Vec<u32>, String> {
            drafter.configure_request(SpecRequestConfig {
                temp,
                top_p,
                top_k,
                min_p,
                cactus_delta: 0.0,
                rng_seed: seed,
                allow_ngram_modifier: false,
            });
            let mut seed_token =
                drafter
                    .mtp_prefill(&mut gpu, &mut bundle, &prompt, &prompt, 0, false, &|| false)?;
            let mut ids = vec![seed_token];
            while ids.len() < TOKENS && seed_token != eos {
                let position = bundle.state.position;
                let window = drafter.mtp_step(
                    &mut gpu,
                    &mut bundle,
                    position,
                    seed_token,
                    &ids,
                    MTP_K.min(TOKENS - ids.len()),
                    eos,
                    None,
                )?;
                seed_token = *window.committed.last().ok_or("empty MTP window")?;
                ids.extend_from_slice(&window.committed);
            }
            ids.truncate(TOKENS);
            ids.resize(TOKENS, eos);
            Ok(ids)
        };
        for trial in 0..trials {
            samples.push(run(&mut drafter, trial_seed(0x5EC, trial))?);
        }
        for trial in 0..REPLAYS.min(trials) {
            let replay = run(&mut drafter, trial_seed(0x5EC, trial))?;
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
        Box::new(drafter).mtp_free(&mut gpu);
    }
    if arm == "ar" {
        for ids in &mut samples {
            if let Some(end) = ids.iter().position(|&t| t == bundle.config.eos_token_id) {
                ids[end..].fill(bundle.config.eos_token_id);
            }
        }
    }
    println!(
        "ARM {arm} {trials} trials in {:.1}s; first sample: {:?}",
        started.elapsed().as_secs_f64(),
        tokenizer.decode(&samples[0])
    );
    fs::write(
        dir.join(format!("{arm}.ids.json")),
        serde_json::to_vec(&samples).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    bundle.free_gpu(&mut gpu).map_err(|e| e.to_string())
}
