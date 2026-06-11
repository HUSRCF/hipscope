use crate::Llama;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::llama::{ForwardScratch, KvCache, LlamaConfig, LlamaWeights};
use hipfire_runtime::loader_api::{Carrier, ModelSource, LoadCtx};

pub struct LlamaBundle {
    pub config: LlamaConfig,
    pub weights: LlamaWeights,
    pub scratch: ForwardScratch,
    pub kv: KvCache,
}

pub struct LlamaCarrier;
impl Carrier for LlamaCarrier {
    type Bundle = LlamaBundle;
    fn name(&self) -> &'static str { "llama" }
    fn probe(&self, src: &ModelSource) -> bool {
        matches!(src.arch_id(), Some(id) if id < 5)
    }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<LlamaBundle, String> {
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
        Ok(LlamaBundle { config, weights, scratch, kv })
    }
}
