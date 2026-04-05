//! Profile turbo attention: wall-clock comparison of Q8 vs asymmetric at varying context.
//! Uses the actual model KV cache dispatch path.

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("Build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use engine::hfq::HfqFile;
    use engine::qwen35;
    use engine::llama;
    use rdna_compute::DType;
    use std::path::Path;
    use std::time::Instant;

    let model_path = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: profile_turbo_attention <model.hfq>"); std::process::exit(1); });

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init");
    let hfq = HfqFile::open(Path::new(&model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("config");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("weights");

    eprintln!("Model: {} layers, dim={}, heads={}, kv_heads={}, head_dim={}",
        config.n_layers, config.dim, config.n_heads, config.n_kv_heads, config.head_dim);

    let max_seq = 8192usize;
    let n_kv_layers = config.layer_types.iter()
        .filter(|t| **t == qwen35::LayerType::FullAttention).count();
    eprintln!("FA layers: {n_kv_layers}, max_seq: {max_seq}\n");

    // Test Q8 and asymmetric at different prefilled context lengths
    for mode in ["q8", "asym"] {
        let mut kv = match mode {
            "asym" => llama::KvCache::new_gpu_asym_q8k_turbo4v_boundary(
                &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq, 2, n_kv_layers).unwrap(),
            _ => llama::KvCache::new_gpu_q8(
                &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, max_seq).unwrap(),
        };
        let mut dn = qwen35::DeltaNetState::new(&mut gpu, &config).unwrap();
        let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

        eprintln!("=== {mode} KV ===");
        eprintln!("{:>8} {:>10} {:>10}", "ctx_len", "ms/token", "tok/s");

        // Fill context by running tokens through forward
        let dummy_token = 1u32; // any valid token
        let mut pos = 0usize;

        for &target_ctx in &[64usize, 256, 512, 1024, 2048, 4096] {
            // Fill to target context
            while pos < target_ctx {
                qwen35::forward_scratch(&mut gpu, &weights, &config, dummy_token, pos, &mut kv, &mut dn, &scratch).unwrap();
                pos += 1;
            }
            gpu.hip.device_synchronize().unwrap();

            // Measure decode speed at this context length (10 tokens, median)
            let mut times = Vec::new();
            for _ in 0..10 {
                let t = Instant::now();
                qwen35::forward_scratch(&mut gpu, &weights, &config, dummy_token, pos, &mut kv, &mut dn, &scratch).unwrap();
                gpu.hip.device_synchronize().unwrap();
                times.push(t.elapsed());
                pos += 1;
            }
            times.sort();
            let median = times[5]; // median of 10
            let ms = median.as_secs_f64() * 1000.0;
            let tps = 1000.0 / ms;

            eprintln!("{:>8} {:>10.2} {:>10.1}", target_ctx, ms, tps);
        }

        // Free
        kv.free_gpu(&mut gpu);
        dn.free_gpu(&mut gpu);
        scratch.free_gpu(&mut gpu);
        gpu.drain_pool();
        eprintln!();
    }
}
