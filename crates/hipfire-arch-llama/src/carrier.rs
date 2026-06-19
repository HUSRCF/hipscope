use crate::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::llama::{ForwardScratch, KvCache, LlamaConfig, LlamaWeights};
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

pub struct LlamaBundle {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv: KvCache,
}

/// Build the LLaMA GPU bundle from an HFQ source.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("llama: directory source unsupported".into());
    };
    let config = <Llama as Architecture>::config_from_hfq(&hfq).map_err(|e| e.to_string())?;
    let weights = <Llama as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
    let scratch = <Llama as Architecture>::new_state(ctx.gpu, &config)?;
    let kv = KvCache::new_gpu_q8(
        ctx.gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        ctx.max_seq,
    )
    .map_err(|e| format!("llama: KvCache::new_gpu_q8 failed: {e}"))?;
    Ok(LlamaBundle {
        config,
        weights,
        scratch,
        kv,
    })
}
