// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen3.5/3.6/3.8 dense tensor parallelism on the generic mesh executor
//! (#927 item 4).
//!
//! Each TP rank holds an even contiguous block of every head set (query, KV,
//! DeltaNet key and value heads) and of the FFN intermediate
//! ([`tp_rank_layouts`], the hand route's `dense_tp_rank_layouts` restricted
//! to even splits). The split plan ([`qwen35_tp_plan`]) runs the embedding
//! and the head replicated. Per layer the mixer (`SplitRule::HeadMixer`)
//! writes this rank's output-projection partial into
//! [`QWEN35_MIX_OUT_RESOURCE`] and the FFN (`SplitRule::GatedMlp`, `Product`
//! ladder) into [`QWEN35_FFN_OUT_RESOURCE`]; the residual `Accumulate` after
//! each becomes one rank-ordered `AllReduceAdd` into `x`, so a rank's program
//! runs in one segment per collective ([`Qwen35TpBindings::run_ops`]). The
//! mixers run their `MixerOutput::Write` (decode) and batched
//! `BatchEpilogue::Partial` (prefill) bodies, the hand dense-TP route's.
//!
//! One program serves both routes: a decode epoch runs it over one token on
//! the decode scratch, a prefill epoch over up to `max_batch` tokens on the
//! rank's batch scratch (the logits are the last row's). Sessions,
//! checkpoints and prefix repair are refused until they are implemented.

use super::batch::PrefillBatchScratch;
use super::config::{dense_tp_rank_layouts, DenseTpRankLayout, LayerType, Qwen35Config};
use super::forward::{prepare_scratch_inputs, Qwen35Scratch};
use super::prefill::{batch_chunk_embed_tokens, batch_chunk_upload_positions};
use super::program;
use super::weights::{DeltaNetState, LayerWeights, Qwen35Weights};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::batched::{
    q8_prefill_wmma_enabled, BatchEpilogue, BatchSemantics, DenseBatchMath, DflashFusionCtx,
};
use hipfire_dispatch::pipeline::batched_attention::batch_chunk_full_attn_attn;
use hipfire_dispatch::pipeline::batched_deltanet::batch_chunk_delta_net_attn;
use hipfire_dispatch::pipeline::gated_mlp::{MlpLadder, MlpOutput};
use hipfire_dispatch::pipeline::hybrid::MixerOutput;
use hipfire_dispatch::pipeline::step_graph::{
    OpDecl, OpId, OpKind, ResourceClass, ResourceDecl, ResourceId, SplitRule, StepGraph,
};
use hipfire_dispatch::pipeline::{execute_steps, GemvInput, Step};
use hipfire_runtime::device_mesh::{DeviceMesh, DimKind};
use hipfire_runtime::llama;
use hipfire_runtime::mesh_placement::BandPlacement;
use hipfire_runtime::mesh_split::{
    split_program, CollectiveKind, PlannedCollective, RankItem, SplitPlan,
};
use hipfire_runtime::multi_gpu::executor::{
    BoundaryIo, ForwardEpoch, MeshCheckpointId, MeshCheckpointKey, MeshRoute, MeshStepBindings,
};
use hipfire_runtime::multi_gpu::transport::EdgeDescriptor;
use hipfire_runtime::tp_shard::{ExpertAssign, ShardConfig};
use hipfire_runtime::weight_manifest::{plan_manifest, ShardPolicy, WeightEntry};
use rdna_compute::{DType, Gpu};

type Result<T, E = String> = std::result::Result<T, E>;

/// The residual stream every op reads and the `AllReduceAdd` updates.
const X_RESOURCE: ResourceId = ResourceId(1);
/// The dense FFN output: a per-rank partial on a TP mesh.
pub const QWEN35_FFN_OUT_RESOURCE: ResourceId = ResourceId(2);
/// The mixer output projection: a per-rank partial on a TP mesh.
pub const QWEN35_MIX_OUT_RESOURCE: ResourceId = ResourceId(3);
/// Packed-quant block along a row-sharded (input) axis: whole G256 groups, as
/// in the hand dense-TP route.
const ROW_GROUP: usize = 256;

/// What one op of the program runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TpOp {
    /// Token embedding and positions into the scratch.
    Embed,
    /// The layer's gated DeltaNet or gated attention mixer over this rank's
    /// heads, written to `mix_out`; `index` is the layer's DeltaNet state
    /// index or KV layer index.
    Mixer { layer: usize, index: usize },
    /// The layer's dense `GatedMlp`, written to `ffn_out`.
    Ffn { layer: usize },
    /// `x += mix_out` or `x += ffn_out`; replaced by the `AllReduceAdd` on a
    /// TP mesh.
    Residual,
    /// Final norm and LM head into the logits.
    Head,
}

/// One declared weight: manifest name, logical shape, policy, row group.
type WeightDecl = (&'static str, [usize; 2], ShardPolicy, usize);

/// The mixer projections of `kind`, input projections first, output
/// projection last: heads split by `HeadSharded`, the output projection by
/// `RowShard { axis: 1 }` over the same heads.
fn mixer_weights(config: &Qwen35Config, kind: LayerType) -> (Vec<WeightDecl>, usize, usize) {
    let dim = config.dim;
    let heads = |n_heads, head_dim| ShardPolicy::HeadSharded { n_heads, head_dim };
    let row = ShardPolicy::RowShard { axis: 1 };
    match kind {
        LayerType::LinearAttention => {
            let (k, v) = (config.linear_num_key_heads, config.linear_num_value_heads);
            let (kd, vd) = (config.linear_key_head_dim, config.linear_value_head_dim);
            let qkv = 2 * k * kd + v * vd;
            let decls = vec![
                (
                    "linear_attn.in_proj_qkv",
                    [qkv, dim],
                    heads(k, qkv / k.max(1)),
                    1,
                ),
                ("linear_attn.in_proj_z", [v * vd, dim], heads(v, vd), 1),
                ("linear_attn.in_proj_b", [v, dim], heads(v, 1), 1),
                ("linear_attn.in_proj_a", [v, dim], heads(v, 1), 1),
                ("linear_attn.out_proj", [dim, v * vd], row, ROW_GROUP),
            ];
            (decls, v, k)
        }
        LayerType::FullAttention => {
            let (h, kvh, hd) = (config.n_heads, config.n_kv_heads, config.head_dim);
            let decls = vec![
                ("self_attn.q_proj", [2 * h * hd, dim], heads(h, 2 * hd), 1),
                ("self_attn.k_proj", [kvh * hd, dim], heads(kvh, hd), 1),
                ("self_attn.v_proj", [kvh * hd, dim], heads(kvh, hd), 1),
                ("self_attn.o_proj", [dim, h * hd], row, ROW_GROUP),
            ];
            (decls, h, kvh)
        }
    }
}

/// The dense FFN projections, `[gate, up, down]`.
fn ffn_weights(config: &Qwen35Config) -> [WeightDecl; 3] {
    let (dim, inner) = (config.dim, config.hidden_dim);
    let column = ShardPolicy::ColumnShard { axis: 0 };
    [
        ("mlp.gate_proj", [inner, dim], column.clone(), 1),
        ("mlp.up_proj", [inner, dim], column, 1),
        (
            "mlp.down_proj",
            [dim, inner],
            ShardPolicy::RowShard { axis: 1 },
            ROW_GROUP,
        ),
    ]
}

/// The program of a dense model: the step graph and, per op, what it runs.
/// `dtypes[layer]` are the payload dtypes of the layer's declared weights
/// (mixer projections, then gate/up/down) that the manifest entries carry.
/// Built together so the two cannot drift.
fn tp_program(
    config: &Qwen35Config,
    dtypes: &[Vec<DType>],
) -> Result<(StepGraph, Vec<TpOp>, Vec<WeightEntry>)> {
    let dim = config.dim;
    if config.num_experts != 0 {
        return Err("Qwen3.5 TP admits dense models only (MoE uses the EP route)".into());
    }
    if config.n_layers == 0
        || config.layer_types.len() != config.n_layers
        || dtypes.len() != config.n_layers
    {
        return Err("Qwen3.5 TP program: layer count disagrees".into());
    }
    let error = |e| format!("Qwen3.5 TP step graph: {e}");
    let mut graph = StepGraph::new();
    let resource = |graph: &mut StepGraph, class, shape: &[usize]| {
        graph
            .add_resource(ResourceDecl {
                class,
                shape: shape.to_vec(),
                alias: None,
            })
            .map_err(error)
    };
    let token = resource(&mut graph, ResourceClass::Input, &[1])?;
    let x = resource(&mut graph, ResourceClass::Activation, &[dim])?;
    let ffn_out = resource(&mut graph, ResourceClass::Activation, &[dim])?;
    let mix_out = resource(&mut graph, ResourceClass::Activation, &[dim])?;
    debug_assert_eq!(
        (x, ffn_out, mix_out),
        (X_RESOURCE, QWEN35_FFN_OUT_RESOURCE, QWEN35_MIX_OUT_RESOURCE)
    );
    let logits = resource(&mut graph, ResourceClass::Output, &[config.vocab_size])?;

    let mut ops = Vec::with_capacity(4 * config.n_layers + 2);
    let mut entries = Vec::with_capacity(8 * config.n_layers);
    let op = |graph: &mut StepGraph,
              ops: &mut Vec<TpOp>,
              what: TpOp,
              kind: OpKind,
              layer: Option<usize>,
              rule: SplitRule,
              reads: &[ResourceId],
              writes: &[ResourceId]| {
        graph
            .add_op(OpDecl {
                kind,
                layer,
                rule,
                reads: reads.to_vec(),
                writes: writes.to_vec(),
            })
            .map_err(error)?;
        ops.push(what);
        Ok::<_, String>(())
    };
    let opaque = SplitRule::Opaque;
    op(
        &mut graph,
        &mut ops,
        TpOp::Embed,
        OpKind::Embed,
        None,
        opaque,
        &[token],
        &[x],
    )?;
    let (mut delta, mut kv) = (0, 0);
    for (layer, (&kind, dtypes)) in config.layer_types.iter().zip(dtypes).enumerate() {
        let (mixer, heads, kv_heads) = mixer_weights(config, kind);
        let n_mixer = mixer.len();
        let ffn = ffn_weights(config);
        let decls: Vec<&WeightDecl> = mixer.iter().chain(&ffn).collect();
        if dtypes.len() != decls.len() {
            return Err(format!(
                "Qwen3.5 TP program: layer {layer} weight dtypes disagree"
            ));
        }
        let mut weight = Vec::with_capacity(decls.len());
        for (&&(name, shape, ref policy, group), &dtype) in decls.iter().zip(dtypes) {
            let class = ResourceClass::Weight {
                name: name.to_string(),
                layer: Some(layer),
                group,
            };
            weight.push(resource(&mut graph, class, &shape)?);
            entries.push(WeightEntry::layer(
                name,
                layer,
                shape.to_vec(),
                dtype,
                policy.clone(),
            ));
        }
        let (mixer_w, ffn_w) = weight.split_at(n_mixer);
        let (counter, op_kind) = match kind {
            LayerType::LinearAttention => (&mut delta, OpKind::DeltaNetMixer),
            LayerType::FullAttention => (&mut kv, OpKind::GatedAttention),
        };
        let index = *counter;
        *counter += 1;
        let rule = SplitRule::HeadMixer {
            input: x,
            out_proj: mixer_w[n_mixer - 1],
            output: mix_out,
            heads,
            kv_heads,
        };
        let reads: Vec<_> = std::iter::once(x).chain(mixer_w.iter().copied()).collect();
        let what = TpOp::Mixer { layer, index };
        op(
            &mut graph,
            &mut ops,
            what,
            op_kind,
            Some(layer),
            rule,
            &reads,
            &[mix_out],
        )?;
        let residual = |src| SplitRule::Accumulate { dst: x, src };
        let what = TpOp::Residual;
        let kind = OpKind::ResidualAdd;
        op(
            &mut graph,
            &mut ops,
            what,
            kind,
            Some(layer),
            residual(mix_out),
            &[x, mix_out],
            &[x],
        )?;
        let [gate, up, down] = [ffn_w[0], ffn_w[1], ffn_w[2]];
        let rule = SplitRule::GatedMlp {
            input: x,
            gate,
            up,
            down,
            output: ffn_out,
            ladder: MlpLadder::Product,
        };
        let what = TpOp::Ffn { layer };
        op(
            &mut graph,
            &mut ops,
            what,
            OpKind::GatedMlp,
            Some(layer),
            rule,
            &[x, gate, up, down],
            &[ffn_out],
        )?;
        op(
            &mut graph,
            &mut ops,
            TpOp::Residual,
            kind,
            Some(layer),
            residual(ffn_out),
            &[x, ffn_out],
            &[x],
        )?;
    }
    op(
        &mut graph,
        &mut ops,
        TpOp::Head,
        OpKind::Gemv,
        None,
        SplitRule::RowComplete,
        &[x],
        &[logits],
    )?;
    Ok((graph, ops, entries))
}

/// Rank layouts of a `parts`-way dense TP mesh: the hand route's static
/// layouts (`dense_tp_rank_layouts`), admitted only when every split is an
/// even contiguous block (the splitter's `Split` layout): query, KV, key and
/// value heads and the FFN intermediate each divide by `parts`.
pub fn tp_rank_layouts(config: &Qwen35Config, parts: usize) -> Result<Vec<DenseTpRankLayout>> {
    let shard =
        ShardConfig::new(parts, false, 0, ExpertAssign::Stride).map_err(|e| format!("{e:?}"))?;
    let layouts = dense_tp_rank_layouts(config, &shard)?;
    let even = |range: &std::ops::Range<usize>, total: usize, rank: usize| {
        total % parts == 0 && *range == (rank * total / parts..(rank + 1) * total / parts)
    };
    for (rank, l) in layouts.iter().enumerate() {
        if !(even(&l.q_head_range, config.n_heads, rank)
            && even(&l.kv_head_range, config.n_kv_heads, rank)
            && even(&l.delta_key_head_range, config.linear_num_key_heads, rank)
            && even(
                &l.delta_value_head_range,
                config.linear_num_value_heads,
                rank,
            )
            && even(&l.ffn_hidden_range, config.hidden_dim, rank))
        {
            return Err(format!(
                "Qwen3.5 TP{parts}: rank {rank} layout {l:?} is not an even head/FFN split"
            ));
        }
    }
    Ok(layouts)
}

/// The TP split plan of the program on a pure TP `mesh`; `dtypes` as
/// [`layer_dtypes`].
pub fn qwen35_tp_plan(
    config: &Qwen35Config,
    dtypes: &[Vec<DType>],
    mesh: &DeviceMesh,
) -> Result<SplitPlan> {
    if mesh.n_devices() < 2 || mesh.size_of(DimKind::Tp) != mesh.n_devices() {
        return Err("Qwen3.5 TP plan needs a pure TP mesh of two or more devices".into());
    }
    let (graph, _, entries) = tp_program(config, dtypes)?;
    let placement = BandPlacement::uniform(mesh, config.n_layers)?;
    let plan = plan_manifest(&entries, &[], mesh, config.n_layers, &placement)?;
    split_program(&graph, &plan, mesh).map_err(|error| format!("Qwen3.5 TP split: {error}"))
}

/// Per layer, the payload dtypes of the declared weights in [`tp_program`]
/// order: the mixer projections, then gate/up/down.
pub fn layer_dtypes(weights: &Qwen35Weights) -> Result<Vec<Vec<DType>>> {
    weights
        .layers
        .iter()
        .map(|layer| match layer {
            LayerWeights::DeltaNet(l) => Ok([
                &l.wqkv, &l.wz, &l.w_beta, &l.w_alpha, &l.wo, &l.w_gate, &l.w_up, &l.w_down,
            ]
            .iter()
            .map(|w| w.gpu_dtype)
            .collect()),
            LayerWeights::FullAttn(l) => {
                Ok([&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                    .iter()
                    .map(|w| w.gpu_dtype)
                    .collect())
            }
            _ => Err("Qwen3.5 TP admits dense layers only".to_string()),
        })
        .collect()
}

/// One TP rank's model shard and state, resident on the rank's device. Only
/// [`Self::load`] builds one: it derives the rank's layout and loads the
/// weights with exactly that layout, so the layout [`Qwen35TpBindings::new`]
/// checks is the slice the rank holds.
pub struct Qwen35TpRank {
    /// Rank-local config (`local_dense_tp_config` of `layout`): this rank's
    /// heads and FFN block.
    config: Qwen35Config,
    layout: DenseTpRankLayout,
    weights: Qwen35Weights,
    kv: llama::KvCache,
    dn: DeltaNetState,
    /// Decode scratch; its `logits` are the rank's output on both routes.
    scratch: Qwen35Scratch,
    /// Prefill rows (`max_batch` bounds a prefill epoch).
    pbs: PrefillBatchScratch,
    /// F32 `[pbs.max_batch, dim]`: a prefill epoch's mixer and FFN partial.
    partial: rdna_compute::GpuTensor,
    /// Tokens committed so far.
    position: usize,
}

impl Qwen35TpRank {
    /// Rank `rank` of a `parts`-way TP mesh of `global` on `gpu`: derives the
    /// rank's layout, loads its weight slice with that layout and allocates
    /// its state (`kv_mode` KV and `state` DeltaNet state over `max_seq`
    /// positions) and scratch (prefill epochs up to `max_batch` rows). On
    /// error everything allocated so far is freed.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        hfq: &mut hipfire_runtime::hfq::HfqFile,
        global: &Qwen35Config,
        gpu: &mut Gpu,
        rank: usize,
        parts: usize,
        kv_mode: hipfire_runtime::kv_mode::KvMode,
        state: super::weights::StateQuant,
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self> {
        use hipfire_runtime::llama::{KvCacheExt, KvDims, KvLayers, KvTarget};
        let hip = |e: hip_bridge::HipError| e.to_string();
        gpu.bind_thread().map_err(hip)?;
        let mut layouts = tp_rank_layouts(global, parts)?;
        if rank >= layouts.len() {
            return Err(format!("Qwen3.5 TP rank {rank} is outside {parts} parts"));
        }
        let layout = layouts.swap_remove(rank);
        let config = super::config::local_dense_tp_config(global, &layout);
        let weights =
            super::load::load_weights_dense_tp_rank(hfq, global, gpu, &layout).map_err(hip)?;
        let mut weights = Some(weights);
        let mut scratch = None;
        let mut pbs = None;
        let mut partial = None;
        let mut kv = None;
        let built = (|| -> hip_bridge::HipResult<DeltaNetState> {
            scratch = Some(Qwen35Scratch::new_with_kv_max(gpu, &config, 64, max_seq)?);
            pbs = Some(PrefillBatchScratch::new_opt(
                gpu, &config, max_batch, false,
            )?);
            partial = Some(gpu.zeros(&[max_batch * config.dim], DType::F32)?);
            let mask = config
                .layer_types
                .iter()
                .map(|layer| *layer == LayerType::FullAttention)
                .collect();
            let dims = KvDims {
                layers: KvLayers::Mask(mask),
                n_kv_heads: config.n_kv_heads,
                head_dim: config.head_dim,
                max_seq,
                physical_cap: Some(max_seq),
            };
            kv = Some(<llama::KvCache as KvCacheExt>::from_mode(
                kv_mode,
                KvTarget::Single(gpu),
                &dims,
            )?);
            DeltaNetState::new_with_quant(gpu, &config, state)
        })();
        match built {
            Ok(dn) => Ok(Self {
                config,
                layout,
                weights: weights.take().expect("loaded"),
                kv: kv.take().expect("allocated"),
                dn,
                scratch: scratch.take().expect("allocated"),
                pbs: pbs.take().expect("allocated"),
                partial: partial.take().expect("allocated"),
                position: 0,
            }),
            Err(error) => {
                if let Some(kv) = kv {
                    let _ = kv.free_gpu(gpu);
                }
                if let Some(partial) = partial {
                    let _ = gpu.free_tensor(partial);
                }
                if let Some(pbs) = pbs {
                    let _ = pbs.free_gpu(gpu);
                }
                if let Some(scratch) = scratch {
                    let _ = scratch.free_gpu(gpu);
                }
                if let Some(weights) = weights {
                    weights.free_gpu(gpu);
                }
                Err(format!("Qwen3.5 TP rank {rank}: {error}"))
            }
        }
    }

    /// The rank's logits (last row of a prefill epoch, or the decode token).
    pub fn logits(&self) -> &rdna_compute::GpuTensor {
        &self.scratch.logits
    }

    /// Tokens committed so far.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Free everything the rank holds on `gpu`.
    pub fn free_gpu(self, gpu: &mut Gpu) {
        let _ = gpu.bind_thread();
        let _ = self.scratch.free_gpu(gpu);
        let _ = self.pbs.free_gpu(gpu);
        let _ = gpu.free_tensor(self.partial);
        self.dn.free_gpu(gpu);
        let _ = self.kv.free_gpu(gpu);
        self.weights.free_gpu(gpu);
        gpu.drain_pool();
    }
}

/// One epoch adapter over the TP ranks (rank `i` on device `i`): borrows the
/// ranks and the plan they were validated against, never a GPU.
pub struct Qwen35TpBindings<'a> {
    ranks: &'a mut [Qwen35TpRank],
    plan: &'a SplitPlan,
    ops: Vec<TpOp>,
    /// Per rank, the op ranges of its program between collectives.
    segments: Vec<Vec<std::ops::Range<u32>>>,
    /// Per rank, the next segment to run.
    cursor: Vec<usize>,
    tokens: &'a [u32],
    tp_epoch: u64,
    prepared: Option<ForwardEpoch>,
}

impl<'a> Qwen35TpBindings<'a> {
    /// Validate before any submission: two or more ranks of one global config,
    /// rank `i` loaded (by [`Qwen35TpRank::load`]) as the `i`-th even head
    /// and FFN block ([`tp_rank_layouts`]), one weight dtype set, and `plan`
    /// the TP plan of exactly these shards. `tp_epoch` is
    /// `Gpus::mesh_tp_epoch()`.
    pub fn new(
        global: &Qwen35Config,
        ranks: &'a mut [Qwen35TpRank],
        plan: &'a SplitPlan,
        tp_epoch: u64,
    ) -> Result<Self> {
        let parts = ranks.len();
        if parts < 2 || tp_epoch == 0 {
            return Err("Qwen3.5 TP needs two or more ranks and a provisioned TP group".into());
        }
        let dtypes = layer_dtypes(&ranks[0].weights)?;
        let layouts = tp_rank_layouts(global, parts)?;
        for (index, (rank, layout)) in ranks.iter().zip(&layouts).enumerate() {
            // The layout is the one the rank's weights were loaded with.
            if rank.layout != *layout || layer_dtypes(&rank.weights)? != dtypes {
                return Err(format!(
                    "Qwen3.5 TP rank {index} was not loaded as rank {index} of {parts}"
                ));
            }
        }
        let (_, ops, _) = tp_program(global, &dtypes)?;
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, parts)]).map_err(|e| format!("{e:?}"))?;
        let expected = qwen35_tp_plan(global, &dtypes, &mesh)?;
        if expected.ranks != plan.ranks || expected.collectives != plan.collectives {
            return Err("Qwen3.5 TP plan is not the plan of these shards".into());
        }
        let segments = plan
            .ranks
            .iter()
            .map(|rank| {
                let mut out = Vec::new();
                let mut run: Option<std::ops::Range<u32>> = None;
                for item in &rank.items {
                    match *item {
                        RankItem::Op(OpId(id)) => match &mut run {
                            Some(range) if range.end == id => range.end = id + 1,
                            Some(range) => {
                                out.push(range.clone());
                                *range = id..id + 1;
                            }
                            None => run = Some(id..id + 1),
                        },
                        RankItem::Collective(_) => out.extend(run.take()),
                    }
                }
                out.extend(run);
                out
            })
            .collect();
        Ok(Self {
            ranks,
            plan,
            ops,
            segments,
            cursor: vec![0; parts],
            tokens: &[],
            tp_epoch,
            prepared: None,
        })
    }

    /// The tokens of the next epoch: one for decode, the prompt chunk for
    /// prefill.
    pub fn set_tokens(&mut self, tokens: &'a [u32]) {
        self.tokens = tokens;
        self.prepared = None;
    }

    fn require_prepared(&self, epoch: &ForwardEpoch) -> Result<()> {
        if self.prepared != Some(*epoch) {
            return Err("Qwen3.5 TP epoch has not been prepared".into());
        }
        Ok(())
    }

    fn unsupported(what: &str) -> String {
        format!("Qwen3.5 TP does not admit {what} yet")
    }

    fn run_op(
        rank: &mut Qwen35TpRank,
        gpu: &mut Gpu,
        op: TpOp,
        epoch: &ForwardEpoch,
        tokens: &[u32],
    ) -> Result<()> {
        let hip = |e: hip_bridge::HipError| e.to_string();
        let dispatch = |e: hipfire_dispatch::types::DispatchError| e.to_string();
        let ctx = DispatchCtx::new(gpu);
        let dims = program::hybrid_dims(&rank.config);
        let (s, n, start) = (&rank.scratch, epoch.actual_rows, rank.position);
        let prefill = epoch.route == MeshRoute::Prefill;
        let dim = rank.config.dim;
        let q8 = q8_prefill_wmma_enabled(gpu);
        match (op, prefill) {
            (TpOp::Embed, _) => {
                rank.kv
                    .ensure_mapped_capacity(gpu, start + n)
                    .map_err(hip)?;
                if !prefill {
                    let w = &rank.weights;
                    return prepare_scratch_inputs(gpu, w, &rank.config, tokens[0], start, s)
                        .map_err(hip);
                }
                let pbs = &rank.pbs;
                let w = &rank.weights;
                batch_chunk_embed_tokens(
                    gpu,
                    w,
                    tokens,
                    s,
                    pbs,
                    n,
                    dim,
                    dim * 4,
                    true,
                    false,
                    false,
                    None,
                    None,
                )
                .map_err(hip)?;
                batch_chunk_upload_positions(
                    gpu,
                    pbs,
                    BatchSemantics::Sequential,
                    start,
                    n,
                    None,
                    false,
                )
                .map_err(hip)
            }
            (TpOp::Mixer { layer, index }, false) => {
                let binding = program::DecodeBinding {
                    dims,
                    s,
                    kv_cache: &rank.kv,
                    dn_state: &rank.dn,
                    position: start,
                    mrope_section: None,
                };
                let output = MixerOutput::Write(&s.o);
                let layer_weights = &rank.weights.layers[layer];
                let step = binding.mixer(layer_weights, layer, index, false, None, output);
                execute_steps(gpu, &ctx, &[step]).map_err(dispatch)
            }
            // The batched mixers of the hand dense-TP prefill over this rank's
            // heads, their output projection written to the partial rows.
            (TpOp::Mixer { layer, index }, true) => match &rank.weights.layers[layer] {
                LayerWeights::DeltaNet(l) => {
                    let cfg = &rank.config;
                    let k_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
                    let v_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
                    batch_chunk_delta_net_attn(
                        gpu,
                        &l.mixer_view(),
                        &dims,
                        &rank.pbs.view(),
                        &program::deltanet_state_view(&rank.dn),
                        n,
                        dim,
                        k_dim,
                        v_dim,
                        cfg.linear_num_value_heads,
                        cfg.linear_key_head_dim,
                        BatchSemantics::Sequential,
                        None,
                        None,
                        0,
                        index,
                        q8,
                        q8,
                        BatchEpilogue::Partial(&rank.partial),
                        DflashFusionCtx::Off,
                        None,
                        false,
                    )
                    .map_err(hip)
                }
                LayerWeights::FullAttn(l) => {
                    let tap_fn = program::batched_attention_tap(&rank.config);
                    batch_chunk_full_attn_attn(
                        gpu,
                        false,
                        &l.mixer_view(),
                        &dims,
                        &rank.pbs.view(),
                        &program::flash_scratch(s),
                        &program::kv_view(&rank.kv),
                        n,
                        dim,
                        start,
                        start + n,
                        &ctx,
                        BatchSemantics::Sequential,
                        None,
                        q8,
                        q8,
                        index,
                        layer,
                        BatchEpilogue::Partial(&rank.partial),
                        DflashFusionCtx::Off,
                        None,
                        tap_fn.get(),
                    )
                    .map_err(hip)
                }
                _ => Err("Qwen3.5 TP mixer on a MoE layer".into()),
            },
            (TpOp::Ffn { layer }, false) => {
                let output = MlpOutput::Write(&s.ffn_out);
                let ffn = program::dense_ffn(&dims, s, &rank.weights.layers[layer], output)
                    .ok_or("Qwen3.5 TP FFN on a MoE layer")?;
                execute_steps(gpu, &ctx, &[Step::GatedMlp(ffn)]).map_err(dispatch)
            }
            (TpOp::Ffn { layer }, true) => {
                let (eps, batch) = (rank.config.norm_eps, rank.pbs.view());
                let output = MlpOutput::Write(&rank.partial);
                let (off, math) = (DflashFusionCtx::Off, DenseBatchMath::Product);
                match &rank.weights.layers[layer] {
                    LayerWeights::DeltaNet(l) => {
                        l.ffn_rows(gpu, eps, &batch, n, dim, q8, q8, output, off, math)
                    }
                    LayerWeights::FullAttn(l) => {
                        l.ffn_rows(gpu, eps, &batch, n, dim, q8, q8, output, off, math)
                    }
                    _ => return Err("Qwen3.5 TP FFN on a MoE layer".into()),
                }
                .map_err(hip)
            }
            (TpOp::Residual, _) => Err("Qwen3.5 TP residual add must be the AllReduceAdd".into()),
            (TpOp::Head, _) => {
                let last;
                let x = if prefill {
                    last = rank.pbs.x_batch.sub_offset((n - 1) * dim, dim);
                    &last
                } else {
                    &s.x
                };
                gpu.rmsnorm_f32(x, &rank.weights.output_norm, &s.tmp, rank.config.norm_eps)
                    .map_err(hip)?;
                let w = rank.weights.output.dispatch_ref();
                let step = Step::Gemv {
                    w: &w,
                    input: GemvInput::Raw(&s.tmp),
                    out: &s.logits,
                };
                execute_steps(gpu, &ctx, &[step]).map_err(dispatch)
            }
        }
    }
}

impl MeshStepBindings for Qwen35TpBindings<'_> {
    fn prepare_epoch(&mut self, epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let rows = epoch.actual_rows;
        let admitted = match epoch.route {
            MeshRoute::Decode => rows == 1,
            MeshRoute::Prefill => {
                rows >= 1 && self.ranks.iter().all(|rank| rows <= rank.pbs.max_batch)
            }
            _ => false,
        };
        if !admitted || epoch.route_rows != rows {
            return Err(format!(
                "Qwen3.5 TP admits single-token decode and prefill epochs within the batch \
                 scratch only, got {:?} over {rows} rows",
                epoch.route
            ));
        }
        if self.tokens.len() != rows {
            return Err("Qwen3.5 TP epoch tokens disagree with its rows".into());
        }
        for rank in self.ranks.iter() {
            let end = rank.position.checked_add(rows);
            if rank.position != epoch.start_position
                || end.is_none_or(|end| end > rank.kv.physical_cap)
            {
                return Err("Qwen3.5 TP rank position or KV capacity disagrees".into());
            }
        }
        self.cursor.fill(0);
        self.prepared = Some(*epoch);
        Ok(())
    }

    fn run_ops(
        &mut self,
        owner: usize,
        gpu: &mut Gpu,
        epoch: &ForwardEpoch,
        ops: &[OpId],
    ) -> Result<()> {
        self.require_prepared(epoch)?;
        let segment = self
            .segments
            .get(owner)
            .and_then(|segments| segments.get(self.cursor[owner]))
            .ok_or("Qwen3.5 TP rank ran past its program")?;
        if !ops.iter().map(|op| op.0).eq(segment.clone()) {
            return Err("Qwen3.5 TP segment does not continue its rank program in order".into());
        }
        for op in ops {
            let op = self.ops[op.0 as usize];
            Self::run_op(&mut self.ranks[owner], gpu, op, epoch, self.tokens)?;
        }
        self.cursor[owner] += 1;
        Ok(())
    }

    fn boundary(
        &self,
        owner: usize,
        collective: &PlannedCollective,
        epoch: &ForwardEpoch,
    ) -> Result<BoundaryIo<'_>> {
        self.require_prepared(epoch)?;
        if !matches!(collective.kind, CollectiveKind::AllReduceAdd { into, .. } if into == X_RESOURCE)
            || !self.plan.collectives.contains(collective)
        {
            return Err("Qwen3.5 TP only adds mixer and FFN partials into the residual".into());
        }
        let rank = self.ranks.get(owner).ok_or("unknown Qwen3.5 TP rank")?;
        let rows = epoch.actual_rows;
        let bytes = rows * rank.config.dim * std::mem::size_of::<f32>();
        let s = &rank.scratch;
        let (view, into) = match (epoch.route, collective.operand) {
            (MeshRoute::Decode, QWEN35_MIX_OUT_RESOURCE) => (&s.o, &s.x),
            (MeshRoute::Decode, QWEN35_FFN_OUT_RESOURCE) => (&s.ffn_out, &s.x),
            (MeshRoute::Prefill, QWEN35_MIX_OUT_RESOURCE | QWEN35_FFN_OUT_RESOURCE) => {
                (&rank.partial, &rank.pbs.x_batch)
            }
            _ => return Err("Qwen3.5 TP collective operand is not a mixer or FFN partial".into()),
        };
        Ok(BoundaryIo {
            view,
            into: Some(into),
            descriptor: EdgeDescriptor {
                epoch: self.tp_epoch,
                op: epoch.op_id,
                rows,
                dtype_bytes: 4,
                bytes,
            },
        })
    }

    fn apply_rank_marks(&mut self, owner: usize, epoch: &ForwardEpoch) -> Result<()> {
        self.require_prepared(epoch)?;
        if self.cursor.get(owner) != self.segments.get(owner).map(Vec::len).as_ref() {
            return Err("Qwen3.5 TP rank did not run its whole program".into());
        }
        let rank = &mut self.ranks[owner];
        if rank.position != epoch.start_position {
            return Err("Qwen3.5 TP rank position moved during the epoch".into());
        }
        rank.position += epoch.actual_rows;
        Ok(())
    }

    fn reset_rank(&mut self, owner: usize, gpu: &mut Gpu, _epoch: &ForwardEpoch) -> Result<()> {
        self.prepared = None;
        let rank = self.ranks.get_mut(owner).ok_or("unknown Qwen3.5 TP rank")?;
        rank.dn.reset(gpu).map_err(|error| error.to_string())?;
        rank.kv.compact_offset = 0;
        rank.position = 0;
        Ok(())
    }

    /// Refused before the first callback, so the mesh stays healthy.
    fn admits_prefix_repair(&self) -> bool {
        false
    }

    fn repair_prefix_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
        _keep: usize,
        _tokens: &[u32],
    ) -> Result<()> {
        Err(Self::unsupported("prefix repair"))
    }

    fn capture_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
    ) -> Result<MeshCheckpointId> {
        Err(Self::unsupported("session capture"))
    }

    fn probe_session_rank(
        &mut self,
        _owner: usize,
        _epoch: &ForwardEpoch,
        _piece: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("session restore"))
    }

    fn restore_session_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _epoch: &ForwardEpoch,
        _checkpoint: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("session restore"))
    }

    fn checkpoint_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
    ) -> Result<MeshCheckpointId> {
        Err(Self::unsupported("checkpoints"))
    }

    fn restore_rank(
        &mut self,
        _owner: usize,
        _gpu: &mut Gpu,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("checkpoints"))
    }

    fn validate_finalize_rank(
        &mut self,
        _owner: usize,
        _key: &MeshCheckpointKey,
        _ticket: MeshCheckpointId,
    ) -> Result<()> {
        Err(Self::unsupported("checkpoints"))
    }

    /// Unreachable: `validate_finalize_rank` refuses every ticket first.
    fn finalize_rank(&mut self, _: usize, _: &MeshCheckpointKey, _: MeshCheckpointId) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small hybrid: three DeltaNet layers then one full-attention layer;
    /// heads 8, KV heads 4, key heads 4, value heads 8 (G256 value width per
    /// TP4 rank) and an FFN intermediate of four G256 groups.
    fn dense_config(experts: usize) -> Qwen35Config {
        let linear = "linear_attention";
        let inner = serde_json::json!({ "config": {
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 512,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 8,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "num_experts": experts,
            "layer_types": [linear, linear, linear, "full_attention"],
        }});
        super::super::config::config_from_metadata_json(&inner.to_string()).unwrap()
    }

    fn dtypes(config: &Qwen35Config) -> Vec<Vec<DType>> {
        config
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerType::LinearAttention => vec![DType::MQ4G256; 8],
                LayerType::FullAttention => vec![DType::MQ4G256; 7],
            })
            .collect()
    }

    /// Every rank runs the whole program except the residual adds; each
    /// layer's mixer partial and then its FFN partial are added into `x` by
    /// one rank-ordered `AllReduceAdd` each, replacing the residual op that
    /// follows the producer.
    #[test]
    fn tp_plan_adds_each_mixer_and_ffn_partial_into_the_residual() {
        let config = dense_config(0);
        let (graph, ops, _) = tp_program(&config, &dtypes(&config)).unwrap();
        for parts in [2, 4] {
            let mesh = DeviceMesh::rect(&[(DimKind::Tp, parts)]).unwrap();
            let plan = qwen35_tp_plan(&config, &dtypes(&config), &mesh).unwrap();
            assert_eq!(plan.collectives.len(), 2 * config.n_layers);
            let mut expected = Vec::new();
            let mut next = 0;
            for (index, op) in ops.iter().enumerate() {
                let id = OpId(index as u32);
                if *op != TpOp::Residual {
                    expected.push(RankItem::Op(id));
                    continue;
                }
                let c = &plan.collectives[next];
                let layer = next / 2;
                let (operand, kind) = match (next % 2, config.layer_types[layer]) {
                    (0, LayerType::LinearAttention) => {
                        (QWEN35_MIX_OUT_RESOURCE, OpKind::DeltaNetMixer)
                    }
                    (0, LayerType::FullAttention) => {
                        (QWEN35_MIX_OUT_RESOURCE, OpKind::GatedAttention)
                    }
                    _ => (QWEN35_FFN_OUT_RESOURCE, OpKind::GatedMlp),
                };
                let add = CollectiveKind::AllReduceAdd {
                    into: X_RESOURCE,
                    replaces: id,
                };
                assert_eq!((c.kind, c.operand), (add, operand));
                assert_eq!(c.group, (0..parts).collect::<Vec<_>>());
                let producer = &graph.ops()[c.producer.0 as usize];
                assert_eq!((producer.kind, producer.layer), (kind, Some(layer)));
                expected.push(RankItem::Collective(c.id));
                next += 1;
            }
            for rank in &plan.ranks {
                assert_eq!(rank.items, expected, "rank {}", rank.device);
            }
            for (rank, layout) in tp_rank_layouts(&config, parts).unwrap().iter().enumerate() {
                let block = |n: usize| rank * n / parts..(rank + 1) * n / parts;
                assert_eq!(layout.q_head_range, block(8));
                assert_eq!(layout.kv_head_range, block(4));
                assert_eq!(layout.delta_value_head_range, block(8));
                assert_eq!(layout.ffn_hidden_range, block(1024));
            }
        }
    }

    /// Three parts split no head count evenly: the rank layout and the
    /// splitter refuse them. MoE models and a one-device mesh are refused.
    #[test]
    fn tp_plan_refusals() {
        let config = dense_config(0);
        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 3)]).unwrap();
        let error = qwen35_tp_plan(&config, &dtypes(&config), &mesh).unwrap_err();
        assert!(error.contains("not divisible by Tp=3"), "{error}");
        let error = tp_rank_layouts(&config, 3).unwrap_err();
        assert!(error.contains("not an even head/FFN split"), "{error}");

        let mesh = DeviceMesh::rect(&[(DimKind::Tp, 2)]).unwrap();
        let moe = dense_config(8);
        let error = qwen35_tp_plan(&moe, &dtypes(&moe), &mesh).unwrap_err();
        assert!(error.contains("dense models only"), "{error}");
        let single = DeviceMesh::single().unwrap();
        assert!(qwen35_tp_plan(&config, &dtypes(&config), &single).is_err());
    }
}

/// Emulated TP prefill + decode of a dense checkpoint through the generic
/// executor, against the hand dense-TP route over the same shards (byte
/// identity) and against the plain single card (reported). Every arm runs
/// the same prompt and the plain card's greedy tokens (teacher-forced).
#[cfg(test)]
mod hw_tests {
    use super::*;
    use crate::qwen35::{
        config_from_hfq, forward_prefill_batch, forward_prefill_dense_tp, forward_scratch,
        forward_scratch_dense_tp, load_weights, load_weights_dense_tp_rank, local_dense_tp_config,
        HfqSource, Layout, StateQuant,
    };
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::kv_mode::KvMode;
    use hipfire_runtime::llama::{KvCacheExt, KvDims, KvLayers, KvTarget};
    use hipfire_runtime::mesh_placement::{AdmittedMesh, AdmittedRank};
    use hipfire_runtime::multi_gpu::Gpus;
    use hipfire_runtime::tokenizer::Tokenizer;
    use rdna_compute::GpuTensor;

    const MODEL_ENV: &str = "HIPFIRE_QWEN35_TP_MODEL";
    const GREEDY: usize = 16;
    /// Forced tokens that run as 1-row Prefill epochs instead of decode: one
    /// right after the prompt, one after decode epochs.
    const ONE_ROW_PREFILL: [usize; 2] = [0, 8];

    /// `name` as a number, else `default`.
    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn kv(gpu: &mut Gpu, config: &Qwen35Config, max_seq: usize) -> llama::KvCache {
        let mask = config
            .layer_types
            .iter()
            .map(|layer| *layer == LayerType::FullAttention)
            .collect();
        let dims = KvDims {
            layers: KvLayers::Mask(mask),
            n_kv_heads: config.n_kv_heads,
            head_dim: config.head_dim,
            max_seq,
            physical_cap: Some(max_seq),
        };
        <llama::KvCache as KvCacheExt>::from_mode(KvMode::Q8, KvTarget::Single(gpu), &dims).unwrap()
    }

    /// The run's rank shape: prefill epoch cap and KV positions.
    #[derive(Clone, Copy)]
    struct Shape {
        chunk: usize,
        max_seq: usize,
    }

    fn load_rank(
        hfq: &mut HfqFile,
        global: &Qwen35Config,
        gpu: &mut Gpu,
        rank: usize,
        parts: usize,
        shape: Shape,
    ) -> Qwen35TpRank {
        let (kv, state) = (KvMode::Q8, StateQuant::Q8);
        let (max_seq, chunk) = (shape.max_seq, shape.chunk);
        Qwen35TpRank::load(hfq, global, gpu, rank, parts, kv, state, max_seq, chunk).unwrap()
    }

    fn free_rank(rank: Qwen35TpRank, gpu: &mut Gpu) {
        rank.free_gpu(gpu);
    }

    fn download(gpu: &Gpu, t: &GpuTensor, n: usize) -> Vec<f32> {
        gpu.bind_thread().unwrap();
        gpu.hip.device_synchronize().unwrap();
        gpu.download_f32(t).unwrap()[..n].to_vec()
    }

    fn argmax(row: &[f32]) -> u32 {
        (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best }) as u32
    }

    fn fnv(rows: &[Vec<f32>]) -> u64 {
        rows.iter()
            .flatten()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
                (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3)
            })
    }

    fn epoch(
        mesh: &DeviceMesh,
        op_id: u64,
        route: MeshRoute,
        start: usize,
        rows: usize,
    ) -> ForwardEpoch {
        ForwardEpoch {
            mesh_epoch: mesh.epoch(),
            load_id: 1,
            request_id: 1,
            op_id,
            route,
            start_position: start,
            actual_rows: rows,
            route_rows: rows,
            policy_id: 1,
        }
    }

    /// The epochs of the TP arms: the prompt as prefill epochs of at most
    /// `chunk` rows (a `rows % chunk == 1` prompt ends in a 1-row epoch), then
    /// one epoch per forced token, a 1-row Prefill at [`ONE_ROW_PREFILL`],
    /// else a decode. `run(tokens, route, start)` returns the epoch's
    /// logits; the result keeps the last prompt epoch's and every forced
    /// token's, the rows the plain card also produces.
    fn rows_of(
        prompt: &[u32],
        chunk: usize,
        forced: &[u32],
        mut run: impl FnMut(&[u32], MeshRoute, usize) -> Vec<f32>,
    ) -> Vec<Vec<f32>> {
        let mut last = Vec::new();
        for (index, epoch) in prompt.chunks(chunk).enumerate() {
            last = run(epoch, MeshRoute::Prefill, index * chunk);
        }
        let mut rows = vec![last];
        for (step, token) in forced.iter().enumerate() {
            let route = if ONE_ROW_PREFILL.contains(&step) {
                MeshRoute::Prefill
            } else {
                MeshRoute::Decode
            };
            rows.push(run(std::slice::from_ref(token), route, prompt.len() + step));
        }
        rows
    }

    /// `HIPFIRE_QWEN35_TP_ROWS` prompt rows (default 16, the text prompt
    /// cycled) as prefill epochs of at most `HIPFIRE_QWEN35_TP_CHUNK` rows
    /// (default 16), then 16 forced tokens, two of them as 1-row Prefill
    /// epochs (after the prompt and after decode).
    #[test]
    #[ignore = "GPU: HIPFIRE_QWEN35_TP_MODEL=<dense .mq4> with HIPFIRE_EMULATE_GPUS=N (one card) or HIPFIRE_QWEN35_TP_PARTS=N (N cards)"]
    fn tp_prefill_decode_matches_hand_route() {
        let parts = std::env::var("HIPFIRE_QWEN35_TP_PARTS")
            .ok()
            .and_then(|parts| parts.parse().ok())
            .or(hipfire_runtime::config::get().emulate_gpus)
            .expect("set HIPFIRE_EMULATE_GPUS or HIPFIRE_QWEN35_TP_PARTS");
        let path = std::env::var(MODEL_ENV).expect(MODEL_ENV);
        let mut hfq = HfqFile::open(std::path::Path::new(&path)).unwrap();
        let global = config_from_hfq(&hfq).unwrap();
        let (vocab, dim) = (global.vocab_size, global.dim);
        let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).unwrap();
        // Cut inside the sentence: the greedy tokens continue natural text.
        let text = "The Roman Empire was the post-Republican state of ancient Rome, ruling large territories around the Mediterranean";
        let rows = env_usize("HIPFIRE_QWEN35_TP_ROWS", 16);
        let chunk = env_usize("HIPFIRE_QWEN35_TP_CHUNK", 16);
        let shape = Shape {
            chunk,
            max_seq: (rows + GREEDY + 16).next_multiple_of(64),
        };
        let prompt: Vec<u32> = tokenizer
            .encode(text)
            .into_iter()
            .cycle()
            .take(rows)
            .collect();
        let max_seq = shape.max_seq;
        let mut gpu = Gpu::init().unwrap();
        gpu.ensure_capture_stream().unwrap();

        // Plain single card, unsharded: batched prefill, then greedy decode;
        // its greedy tokens drive every arm.
        let weights = {
            let mut source = HfqSource::new(&mut hfq, &global);
            let layout = Layout::single(global.n_layers);
            load_weights(&mut source, std::slice::from_mut(&mut gpu), &layout).unwrap()
        };
        let s = Qwen35Scratch::new_with_kv_max(&mut gpu, &global, 64, max_seq).unwrap();
        let mut plain_kv = kv(&mut gpu, &global, max_seq);
        let mut dn = DeltaNetState::new_with_quant(&mut gpu, &global, StateQuant::Q8).unwrap();
        let w = &weights;
        forward_prefill_batch(
            &mut gpu,
            w,
            &global,
            &prompt,
            0,
            &mut plain_kv,
            &mut dn,
            &s,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let mut plain_rows = vec![download(&gpu, &s.logits, vocab)];
        let mut forced = Vec::new();
        for step in 0..GREEDY {
            let token = argmax(plain_rows.last().unwrap());
            forced.push(token);
            let pos = rows + step;
            forward_scratch(&mut gpu, w, &global, token, pos, &mut plain_kv, &mut dn, &s).unwrap();
            plain_rows.push(download(&gpu, &s.logits, vocab));
        }
        let _ = s.free_gpu(&mut gpu);
        dn.free_gpu(&mut gpu);
        let _ = plain_kv.free_gpu(&mut gpu);
        weights.free_gpu(&mut gpu);
        gpu.drain_pool();

        // The hand dense-TP route (`forward_prefill_dense_tp`,
        // `forward_scratch_dense_tp`) over the same shards on its own mesh.
        drop(gpu);
        let hand_rows = {
            let shard = ShardConfig::new(parts, false, 0, ExpertAssign::Stride).unwrap();
            let layouts = tp_rank_layouts(&global, parts).unwrap();
            let configs: Vec<_> = layouts
                .iter()
                .map(|l| local_dense_tp_config(&global, l))
                .collect();
            let mut gpus = Gpus::init_tp(parts, global.n_layers).unwrap();
            let (mut weights, mut scratches, mut kvs, mut dns) = (vec![], vec![], vec![], vec![]);
            for (rank, gpu) in gpus.devices.iter_mut().enumerate() {
                gpu.bind_thread().unwrap();
                gpu.active_stream = Some(gpu.hip.stream_create().unwrap());
                let config = &configs[rank];
                let w = load_weights_dense_tp_rank(&mut hfq, &global, gpu, &layouts[rank]).unwrap();
                weights.push(w);
                scratches.push(Qwen35Scratch::new_with_kv_max(gpu, config, 64, max_seq).unwrap());
                kvs.push(kv(gpu, config, max_seq));
                dns.push(DeltaNetState::new_with_quant(gpu, config, StateQuant::Q8).unwrap());
            }
            if gpus.can_access_peer_all().unwrap() {
                assert!(gpus.enable_peer_all().unwrap());
            }
            let (w, c, sc) = (&weights, &configs, &scratches);
            let rows = rows_of(&prompt, chunk, &forced, |tokens, route, start| {
                match route {
                    MeshRoute::Decode => forward_scratch_dense_tp(
                        &mut gpus, &shard, w, c, tokens[0], start, &mut kvs, &mut dns, sc,
                    ),
                    _ => forward_prefill_dense_tp(
                        &mut gpus, &shard, w, c, tokens, start, &mut kvs, &mut dns, sc,
                    ),
                }
                .unwrap();
                download(&gpus.devices[0], &scratches[0].logits, vocab)
            });
            gpus.free_peer_reduce_scratch().unwrap();
            let parts = weights.into_iter().zip(scratches).zip(kvs).zip(dns);
            for (rank, (((w, s), kv), dn)) in parts.enumerate() {
                let gpu = &mut gpus.devices[rank];
                gpu.bind_thread().unwrap();
                let _ = s.free_gpu(gpu);
                dn.free_gpu(gpu);
                let _ = kv.free_gpu(gpu);
                w.free_gpu(gpu);
                gpu.drain_pool();
            }
            rows
        };

        // The same shards, one per emulated device, through the executor.
        let mut gpus = Gpus::init_tp(parts, 1).unwrap();
        for gpu in &mut gpus.devices {
            gpu.ensure_capture_stream().unwrap();
        }
        let tp_mesh = gpus.mesh.clone();
        let admitted = AdmittedMesh {
            mesh: tp_mesh.clone(),
            ranks: gpus
                .devices
                .iter()
                .enumerate()
                .map(|(logical, gpu)| AdmittedRank {
                    logical,
                    device_ordinal: gpu.device_id,
                    pci_bus_id: None,
                    arch: gpu.arch.clone(),
                })
                .collect(),
            edges: Vec::new(),
            family: "qwen35-tp".into(),
            numerical_policy: "f32".into(),
        };
        let max_bytes = (chunk * dim * std::mem::size_of::<f32>()).next_multiple_of(16);
        let placement = BandPlacement::uniform(&tp_mesh, 1).unwrap();
        gpus.provision_mesh_edges(&admitted, &placement, max_bytes)
            .unwrap();
        let mut ranks: Vec<_> = (0..parts)
            .map(|r| load_rank(&mut hfq, &global, &mut gpus.devices[r], r, parts, shape))
            .collect();
        let dtypes = layer_dtypes(&ranks[0].weights).unwrap();
        let tp_plan = qwen35_tp_plan(&global, &dtypes, &tp_mesh).unwrap();
        let tp_epoch = gpus.mesh_tp_epoch().unwrap();
        // Refusals before any launch, leaving every rank at position 0 and
        // the mesh healthy: a prefill epoch past the batch scratch, an epoch
        // whose tokens disagree with its rows, session capture and prefix
        // repair.
        {
            let long = vec![prompt[0]; chunk + 1];
            let mut bindings =
                Qwen35TpBindings::new(&global, &mut ranks, &tp_plan, tp_epoch).unwrap();
            bindings.set_tokens(&long);
            let e = epoch(&tp_mesh, 1, MeshRoute::Prefill, 0, chunk + 1);
            let error = gpus
                .submit_mesh_epoch(&tp_plan, &e, &mut bindings)
                .unwrap_err();
            assert!(error.contains("within the batch scratch"), "{error}");
            bindings.set_tokens(&prompt[..chunk]);
            let e = epoch(&tp_mesh, 1, MeshRoute::Prefill, 0, chunk - 1);
            let error = gpus
                .submit_mesh_epoch(&tp_plan, &e, &mut bindings)
                .unwrap_err();
            assert!(error.contains("tokens disagree"), "{error}");
            let e = epoch(&tp_mesh, 1, MeshRoute::Decode, 0, 1);
            let error = bindings
                .capture_session_rank(0, &mut gpus.devices[0], &e)
                .unwrap_err();
            assert!(error.contains("does not admit session capture"), "{error}");
            let e = epoch(&tp_mesh, 1, MeshRoute::Decode, 0, 1);
            let error = gpus
                .repair_mesh_prefix(&mut bindings, &e, 0, &[])
                .unwrap_err();
            assert!(error.contains("not admitted"), "{error}");
        }
        assert!(
            gpus.mesh_tp_epoch().is_some(),
            "a refusal quarantined the mesh"
        );
        assert!(ranks.iter().all(|rank| rank.position == 0));
        let mut op_id = 1;
        let tp_rows = rows_of(&prompt, chunk, &forced, |tokens, route, start| {
            op_id += 1;
            let mut bindings =
                Qwen35TpBindings::new(&global, &mut ranks, &tp_plan, tp_epoch).unwrap();
            bindings.set_tokens(tokens);
            let e = epoch(&tp_mesh, op_id, route, start, tokens.len());
            let ticket = gpus.submit_mesh_epoch(&tp_plan, &e, &mut bindings).unwrap();
            drop(bindings);
            gpus.drain_mesh_epoch(ticket).unwrap();
            let rows: Vec<_> = gpus
                .devices
                .iter()
                .zip(&ranks)
                .map(|(gpu, r)| download(gpu, &r.scratch.logits, vocab))
                .collect();
            assert!(rows.iter().all(|row| row == &rows[0]), "TP ranks disagree");
            rows[0].clone()
        });
        assert!(ranks.iter().all(|rank| rank.position == rows + GREEDY));
        gpus.drain_mesh_edges().unwrap();
        for (rank, gpu) in ranks.into_iter().zip(&mut gpus.devices) {
            free_rank(rank, gpu);
        }

        let ids = |rows: &[Vec<f32>]| rows.iter().map(|row| argmax(row)).collect::<Vec<_>>();
        let max_diff = tp_rows
            .iter()
            .flatten()
            .zip(plain_rows.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let same = ids(&tp_rows)
            .iter()
            .zip(ids(&plain_rows))
            .filter(|(a, b)| **a == *b)
            .count();
        eprintln!(
            "TP{parts} {} rows={rows} chunk={chunk}: {} collectives/epoch; hand fnv64 {:016x}, tp fnv64 {:016x}, plain fnv64 {:016x}; \
             tp argmax {:?}; plain argmax {:?} ({same}/{} equal); max |logit diff| vs plain {max_diff:e}",
            std::path::Path::new(&path).file_name().unwrap().to_string_lossy(),
            tp_plan.collectives.len(),
            fnv(&hand_rows),
            fnv(&tp_rows),
            fnv(&plain_rows),
            ids(&tp_rows),
            ids(&plain_rows),
            tp_rows.len(),
        );
        eprintln!(
            "TP{parts} text: {:?} | plain text: {:?}",
            tokenizer.decode(&ids(&tp_rows)),
            tokenizer.decode(&ids(&plain_rows)),
        );
        let bits = |rows: &[Vec<f32>]| {
            rows.iter()
                .map(|row| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&tp_rows),
            bits(&hand_rows),
            "TP{parts} generic vs hand dense-TP route"
        );
    }
}
