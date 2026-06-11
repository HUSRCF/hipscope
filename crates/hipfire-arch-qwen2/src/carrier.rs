use crate::qwen2::{Qwen2Config, Qwen2State, Qwen2Weights};
use crate::Qwen2;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{Carrier, ModelSource, LoadCtx};

pub struct Qwen2Bundle {
    pub config: Qwen2Config,
    pub weights: Qwen2Weights,
    pub state: Qwen2State,
}

pub struct Qwen2Carrier;
impl Carrier for Qwen2Carrier {
    type Bundle = Qwen2Bundle;
    fn name(&self) -> &'static str { "qwen2" }
    fn probe(&self, src: &ModelSource) -> bool { src.arch_id() == Some(7) }
    fn load(&self, src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen2Bundle, String> {
        let ModelSource::Hfq(mut hfq) = src else {
            return Err("qwen2: directory source unsupported".into());
        };
        if ctx.draft_path.is_some() {
            return Err("DFlash not supported on arch_id=7 (qwen2 bring-up). Reload without a draft.".into());
        }
        if ctx.cask.sidecar.is_some() {
            return Err("CASK eviction not supported on arch_id=7 (qwen2 bring-up). Reload without --cask-sidecar.".into());
        }
        let config = <Qwen2 as Architecture>::config_from_hfq(&hfq)?;
        let weights = <Qwen2 as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
        let state = Qwen2State::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
            .map_err(|e| format!("qwen2: Qwen2State::new_with_max_seq failed: {e:?}"))?;
        Ok(Qwen2Bundle { config, weights, state })
    }
}
