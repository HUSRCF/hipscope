use crate::qwen2::{Qwen2Config, Qwen2State, Qwen2Weights};
use crate::Qwen2;
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};

pub struct Qwen2Bundle {
    pub config: Qwen2Config,
    pub weights: Qwen2Weights,
    pub state: Qwen2State,
}

/// Build the Qwen2 GPU bundle from an HFQ source. Refusals owned here.
pub fn load_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Qwen2Bundle, String> {
    let ModelSource::Hfq(mut hfq) = src else {
        return Err("qwen2: directory source unsupported".into());
    };
    if ctx.draft_path.is_some() {
        return Err(
            "DFlash not supported on arch_id=7 (qwen2 bring-up). Reload without a draft.".into(),
        );
    }
    if ctx.cask.sidecar.is_some() {
        return Err("CASK eviction not supported on arch_id=7 (qwen2 bring-up). Reload without --cask-sidecar.".into());
    }
    let config = <Qwen2 as Architecture>::config_from_hfq(&hfq)?;
    let weights = <Qwen2 as Architecture>::load_weights(&mut hfq, &config, ctx.gpu)?;
    let state = Qwen2State::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
        .map_err(|e| format!("qwen2: Qwen2State::new_with_max_seq failed: {e:?}"))?;
    Ok(Qwen2Bundle {
        config,
        weights,
        state,
    })
}
