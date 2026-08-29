// SPDX-License-Identifier: Apache-2.0

use hipfire_arch_qwen35::qwen35::{
    self, DeltaNetState, HfqSource, Layout, Qwen35Scratch, StateQuant,
};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::multi_gpu::Gpus;
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use rdna_compute::Gpu;
use std::path::Path;

const KV_MAX: usize = 4096;
fn decode_steps() -> usize {
    std::env::var("HIPFIRE_TP_PARITY_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(16)
}

fn argmax(xs: &[f32]) -> u32 {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn prompt(tokenizer: &Tokenizer) -> Vec<u32> {
    tokenizer
        .encode("<|im_start|>user\nExplain KV cache briefly.<|im_end|>\n<|im_start|>assistant\n")
}

fn load_single(
    hfq: &mut HfqFile,
    config: &qwen35::Qwen35Config,
    gpu: &mut Gpu,
) -> qwen35::Qwen35Weights {
    let mut source = HfqSource::new(hfq, config);
    qwen35::load_weights(
        &mut source,
        std::slice::from_mut(gpu),
        &Layout::single(config.n_layers),
    )
    .expect("load single weights")
}

fn make_kv(gpu: &mut Gpu, config: &qwen35::Qwen35Config) -> KvCache {
    KvCache::new_gpu(
        gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        KV_MAX,
    )
    .expect("fp32 kv")
}

fn run_reference(path: &str, seed: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let decode = decode_steps();
    let mut hfq = HfqFile::open(Path::new(path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let mut gpu = Gpu::init().expect("gpu0");
    let weights = load_single(&mut hfq, &config, &mut gpu);
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).expect("scratch");
    let mut kv = make_kv(&mut gpu, &config);
    let mut dn = DeltaNetState::new_with_quant(&mut gpu, &config, StateQuant::FP32).expect("dn");
    for (pos, &token) in seed.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu, &weights, &config, token, pos, &mut kv, &mut dn, &scratch,
        )
        .expect("reference prefill");
    }
    let mut tokens = Vec::with_capacity(decode);
    let mut logits = Vec::with_capacity(decode);
    let mut next = argmax(&gpu.download_f32(&scratch.logits).unwrap());
    for step in 0..decode {
        tokens.push(next);
        logits.push(gpu.download_f32(&scratch.logits).unwrap());
        if step + 1 < decode {
            qwen35::forward_scratch(
                &mut gpu,
                &weights,
                &config,
                next,
                seed.len() + step,
                &mut kv,
                &mut dn,
                &scratch,
            )
            .expect("reference decode");
            next = argmax(&gpu.download_f32(&scratch.logits).unwrap());
        }
    }
    let _ = scratch.free_gpu(&mut gpu);
    dn.free_gpu(&mut gpu);
    let _ = kv.free_gpu(&mut gpu);
    weights.free_gpu(&mut gpu);
    gpu.drain_pool();
    (tokens, logits)
}

fn run_tp(path: &str, seed: &[u32], forced: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
    let decode = decode_steps();
    let mut hfq = HfqFile::open(Path::new(path)).expect("open model");
    let global = qwen35::config_from_hfq(&hfq).expect("config");
    let shard = ShardConfig::new(2, false, 0, ExpertAssign::Stride).unwrap();
    qwen35::validate_dense_tp(&global, &shard).unwrap();
    let local = qwen35::local_dense_tp_config(&global, &shard);
    let configs = vec![local.clone(), local];
    let mut gpus = Gpus::init_tp(2, global.n_layers).expect("init tp2");
    for gpu in &mut gpus.devices {
        gpu.bind_thread().unwrap();
        gpu.active_stream = Some(gpu.hip.stream_create().unwrap());
    }
    let mut weights = Vec::new();
    let mut scratches = Vec::new();
    let mut kvs = Vec::new();
    let mut dns = Vec::new();
    for rank in 0..2 {
        weights.push(
            qwen35::load_weights_dense_tp_rank(
                &mut hfq,
                &global,
                &mut gpus.devices[rank],
                &shard,
                rank,
            )
            .expect("load TP rank"),
        );
        scratches.push(Qwen35Scratch::new(&mut gpus.devices[rank], &configs[rank], 128).unwrap());
        kvs.push(make_kv(&mut gpus.devices[rank], &configs[rank]));
        dns.push(
            DeltaNetState::new_with_quant(
                &mut gpus.devices[rank],
                &configs[rank],
                StateQuant::FP32,
            )
            .unwrap(),
        );
    }
    qwen35::forward_prefill_dense_tp(
        &mut gpus, &shard, &weights, &configs, seed, 0, &mut kvs, &mut dns, &scratches,
    )
    .expect("TP prefill");
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    for step in 0..decode {
        gpus.devices[0].bind_thread().unwrap();
        let row = gpus.devices[0].download_f32(&scratches[0].logits).unwrap();
        tokens.push(argmax(&row));
        logits.push(row);
        if step + 1 < decode {
            qwen35::forward_scratch_dense_tp(
                &mut gpus,
                &shard,
                &weights,
                &configs,
                forced[step],
                seed.len() + step,
                &mut kvs,
                &mut dns,
                &scratches,
            )
            .expect("TP decode");
        }
    }
    (tokens, logits)
}

fn main() {
    std::env::set_var("HIPFIRE_DETERMINISTIC", "1");
    let path = std::env::args().nth(1).expect("model path");
    let hfq = HfqFile::open(Path::new(&path)).expect("open tokenizer metadata");
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).expect("tokenizer");
    let seed = prompt(&tokenizer);
    drop(hfq);
    let (reference_tokens, reference_logits) = run_reference(&path, &seed);
    let (tp_tokens, tp_logits) = run_tp(&path, &seed, &reference_tokens);
    let mut worst = 0.0f32;
    for (step, (reference, tp)) in reference_logits.iter().zip(&tp_logits).enumerate() {
        let scale = reference
            .iter()
            .fold(0.0f32, |m, x| m.max(x.abs()))
            .max(1e-12);
        let delta = reference
            .iter()
            .zip(tp)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(delta / scale);
        println!(
            "step={step:02} ref={} tp={} rel={:.3e}",
            reference_tokens[step],
            tp_tokens[step],
            delta / scale
        );
    }
    assert_eq!(reference_tokens, tp_tokens, "TP2 argmax divergence");
    assert!(worst < 1e-4, "TP2 relative logit error {worst:.3e}");
    println!(
        "PASS tokens={} worst_rel={worst:.3e}",
        reference_tokens.len()
    );
}
