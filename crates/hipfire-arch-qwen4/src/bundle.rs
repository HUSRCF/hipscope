// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 model ownership and teardown boundary.
//!
//! A published bundle owns every model resource: the bounded SSD PLE reader,
//! mutable GPU state, finalized resident weights, and the attached canonical
//! load transaction/census (including external PLE descriptors).  No loader
//! local may outlive publication as a second owner.

use crate::config::Qwen4Config;
use crate::gpu_forward::{
    qwen4_forward_device_bytes, qwen4_prefill_chunk_requested, qwen4_prefill_chunk_rungs,
    qwen4_spec_logit_rows, Qwen4GpuForward, Qwen4OutputRows, QWEN4_FORWARD_HEADROOM_BYTES,
};
use crate::kv_backend::Qwen4KvBackend;
use crate::mtp_gpu::{
    MtpAppendScratch, MtpGpuStateSnapshot, MtpStep, Qwen4HeadCheckpoint, Qwen4MtpGpu,
};
use crate::ple::PleHashMetadata;
use crate::state::Qwen4StateFormat;
use crate::state::{Qwen4State, Qwen4StateSnapshot, Qwen4TargetCheckpoint, StateError};
use crate::weights::{
    ple_valid_rows_for_shard, Qwen4Manifest, Qwen4Placement, Qwen4Weights, WeightError,
    PLE_ROW_WIDTH, PLE_SHARD_COUNT, PLE_SHARD_ROWS,
};
use hipfire_runtime::external_rows::{RowEncoding, RowStore, RowStoreError};
use hipfire_runtime::model_source::{SourceFormat, SourceRangeDescriptor};
use hipfire_runtime::sampler::{
    apply_logit_policy_candidates_cpu, apply_logit_policy_cpu, SamplerConfig,
};
use hipfire_runtime::spec_sampling::{SampleSpec, SparseDist};
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::{WeightLoadTransaction, WeightStoreError};
use hip_bridge::VmmPhysicalId;
use hipfire_runtime::checkpoint_pool::{CheckpointBlob, QwenCheckpointPool};
use hipfire_runtime::prefix_index::{Handle, PinTicket, PrefixIndex};
use hipfire_runtime::serve_contract::{sha256_len_prefixed, CacheDomain, CheckpointId};
use rdna_compute::page_pool::{BlockTable, PagePool, PAGE_TOKENS};
use rdna_compute::{Gpu, GpuTensor, VmmPrefixSet, VmmPrefixSpec, VmmResourceMove};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PLE_RESET_TIMEOUT: Duration = Duration::from_secs(5);

/// Architecture-private owner for the fulfilled manifest census.
///
/// Assembly takes every resident handle into [`Qwen4Weights`], so rollback at
/// unload releases no duplicate device allocations.  The transaction still
/// owns the immutable projections, aliases, and external PLE descriptors and
/// is drained only after the reader, mutable state, and resident weights.
pub(crate) struct AttachedWeightStore {
    transaction: WeightLoadTransaction,
}

impl AttachedWeightStore {
    fn new(transaction: WeightLoadTransaction) -> Self {
        Self { transaction }
    }

    fn drain(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        self.transaction.rollback(gpu)
    }

    fn external_descriptor(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&SourceRangeDescriptor> {
        self.transaction.external_descriptor(name, layer, device)
    }

    pub(crate) fn external_rows_len(&self) -> usize {
        self.transaction.external_rows_len()
    }

    pub(crate) fn inventory_len(&self) -> usize {
        self.transaction.inventory_len()
    }

    pub(crate) fn origin(&self) -> Option<hipfire_runtime::weight_store::WeightOrigin> {
        self.transaction.origin()
    }
}

/// Published Qwen4 architecture owner.
pub struct Qwen4Bundle {
    pub config: Qwen4Config,
    pub weights: Qwen4Weights,
    pub state: Qwen4State,
    /// Bounded model-owned PLE reader/cache.  It must quiesce before source
    /// descriptors and the attached transaction are dropped.
    pub(crate) ple_rows: RowStore,
    pub(crate) ple_metadata: PleHashMetadata,
    /// Canonical load census and external descriptors.  This is deliberately
    /// not left in the loader or carrier after publication.
    pub(crate) weight_store: AttachedWeightStore,
    /// Reusable ordinary-HIP execution resources.  This remains attached to
    /// the published bundle so unload owns the scratch, expert pointer tables,
    /// and all per-layer dispatch state exactly once.
    pub(crate) execution: Option<Qwen4GpuForward>,
    /// Reusable native MTP execution resources, attached only when the
    /// admitted artifact carries the validated one-layer MTP head.
    pub(crate) mtp: Option<Qwen4MtpGpu>,
    /// Fixed target-side output buffers for the arch-generic speculative seam.
    /// These are allocated with the ordinary forward owner and reused by every
    /// verify/advance call.
    pub(crate) spec_logits: Option<GpuTensor>,
    pub(crate) spec_top1: Option<GpuTensor>,
    pub(crate) spec_hidden: Option<GpuTensor>,
    pub(crate) spec_host_top1: Vec<u8>,
    /// Single-session prefix cache (`attach_prefix_cache`); `None` = off.
    prefix: Option<Qwen4PrefixCache>,
}

/// The Qwen4 prefix cache is on by default; `HIPFIRE_QWEN_PROMPT_CACHE=0|1`
/// (the prompt-cache switch every Qwen family reads) overrides it.
pub const QWEN4_PREFIX_CACHE_DEFAULT: bool = true;

/// Whether a load should attach (and charge) the prefix cache.
pub fn prefix_cache_requested() -> bool {
    hipfire_config::developer_bool("HIPFIRE_QWEN_PROMPT_CACHE", QWEN4_PREFIX_CACHE_DEFAULT)
}

/// Device bytes [`Qwen4Bundle::attach_prefix_cache`] allocates: the target
/// state checkpoint plus, with native MTP, the head's selection, device
/// selected length and wide hidden. Charged by the load's VRAM reserve.
pub fn prefix_cache_device_bytes(
    config: &Qwen4Config,
    format: Qwen4StateFormat,
    native_mtp: bool,
) -> Option<u64> {
    let target = Qwen4State::prefix_arena_bytes(config, format)?;
    if !native_mtp {
        return Some(target);
    }
    let mtp = config
        .qsa_selected_capacity()
        .checked_add(1)?
        .checked_add(config.hc_count.checked_mul(config.hidden_size)?)?
        .checked_mul(4)?;
    target.checked_add(u64::try_from(mtp).ok()?)
}

/// Prefill schedule a prefix checkpoint was produced under. AR and native
/// MTP prefill run different forwards (MTP captures wide hidden rows and
/// appends every row to the head), so a checkpoint never crosses modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4PrefixMode {
    Ar,
    NativeMtp,
}

/// Which owner state a prefill continues from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Qwen4PrefixSource {
    /// Reset every owner and prefill the whole prompt.
    #[default]
    Cold,
    /// Continue the committed live state in place: no copy, no reset.
    Live,
    /// Restore the end-of-prompt checkpoint.
    Prompt,
    /// Restore a retained radix checkpoint (in place, or into a fresh bank).
    Radix,
}

/// Where a prefill starts. `start_pos` is the cached-token count: `0` is a
/// cold reset; otherwise the live state or the end-of-prompt checkpoint at
/// exactly `start_pos` tokens continues and `prompt[start_pos..]` is
/// prefilled. A hit plan is a receipt only [`Qwen4Bundle::bind_prefix_plan`]
/// issues: `begin_prefix` re-binds it and rejects it once the cache has
/// moved on. `Default` is an unconditional cold start.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen4PrefixPlan {
    pub start_pos: usize,
    source: Qwen4PrefixSource,
    generation: u64,
}

impl Qwen4PrefixPlan {
    pub fn source(&self) -> Qwen4PrefixSource {
        self.source
    }
}

/// Device positions and the admitted prefill chunk the prefix decisions
/// read. Plain data so bind, publish and lineage decisions run without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PrefixMarks {
    /// Tokens the target state has consumed.
    target_position: usize,
    /// Tokens the MTP head has consumed; `None` without a head.
    mtp_position: Option<usize>,
    /// Position of the target's valid durable checkpoint.
    target_prefix: Option<usize>,
    /// Position of the head's valid durable checkpoint.
    mtp_prefix: Option<usize>,
    /// Admitted prefill chunk now; `None` without forward resources.
    chunk: Option<usize>,
    /// A speculative verify row capture is in flight.
    row_capture_armed: bool,
}

impl PrefixMarks {
    /// Target (and for native MTP, the head) have consumed exactly `end`.
    fn aligned_at(&self, end: usize, mode: Qwen4PrefixMode) -> bool {
        self.target_position == end
            && (mode == Qwen4PrefixMode::Ar || self.mtp_position == Some(end))
    }

    /// The live state is the committed state after `end` tokens.
    fn live_at(&self, end: usize, mode: Qwen4PrefixMode) -> bool {
        end > 0 && !self.row_capture_armed && self.aligned_at(end, mode)
    }

    /// The durable arenas hold the checkpoint after `p` tokens: the target's
    /// always, the head's too for native MTP.
    fn checkpoint_at(&self, p: usize, mode: Qwen4PrefixMode) -> bool {
        p > 0
            && self.target_prefix == Some(p)
            && (mode == Qwen4PrefixMode::Ar || self.mtp_prefix == Some(p))
    }
}

/// Host record of what the device owners hold: the request's end-of-prompt
/// checkpoint (`P = prompt_len`) and, after a committed request, the live state.
///
/// The device bytes live in checkpoint slots (target state plus, for native
/// MTP, the head; see [`Qwen4Checkpoint`]) and in the live state itself; this
/// records which tokens they hold and under which schedule. With a radix store
/// it also owns the shared index, descriptor pool, checkpoint pool and budget
/// ledger.
///
/// The key is only tokens, mode and admitted chunk because everything else
/// is immutable for this owner: model artifact, device, state formats,
/// config and placement are fixed per bundle (a reload builds a new bundle
/// with no checkpoint), and numeric dispatch knobs come from the process
/// config snapshot (`hipfire_config` `OnceLock`), fixed for the process. Any
/// future mutable route knob or cross-bundle transport must join the key.
struct Qwen4PrefixCache {
    /// One reusable record (capacity `max_seq`): the prompt `[0, prompt_len)`
    /// after a stage, the full committed history while `live_len` is set.
    /// The checkpoint key is `tokens[..prompt_len]`, never the vector length.
    tokens: Vec<u32>,
    /// End of the staged prompt: the checkpoint's position.
    prompt_len: usize,
    /// Consumed tokens the live state holds, when published and exact.
    live_len: Option<usize>,
    mode: Qwen4PrefixMode,
    /// Admitted prefill chunk the record was produced under.
    chunk: usize,
    /// Set only by a client-committed request; a running request's
    /// candidate is never planned against.
    published: bool,
    /// Absolute position the running prefill checkpoints at.
    capture_at: Option<usize>,
    /// Advances on every begin, reset, rewind and invalidation; a hit receipt
    /// carries the value it was bound under.
    generation: u64,
    /// Free reusable checkpoint slots.
    slots: Vec<Qwen4CheckpointSlot>,
    /// Slots ever allocated and still owned (free, staged, local, pooled).
    slots_allocated: usize,
    /// This request's captures, published at commit.
    staged: Vec<Qwen4Checkpoint>,
    /// The end-of-prompt checkpoint outside the staged list.
    local: Option<LocalEop>,
    /// Planned, not yet consumed by a begin.
    selection: Option<Qwen4Selection>,
    /// The running request's pins (consumed selection), released at the
    /// terminal commit, reset or a failed begin.
    held: Option<Qwen4Selection>,
    /// Checkpoints host-only paths dropped; freed at the next quiescent point.
    deferred: Vec<Qwen4Checkpoint>,
    /// Id of the live context bank; rows of a checkpoint are restorable in
    /// place only while its bank is this one.
    bank: u64,
    next_bank: u64,
    /// Positions after each `<|im_end|>` of the planned prompt.
    turn_boundaries: Vec<usize>,
    /// The conditional shared-turn anchor armed by the selection.
    anchor: Option<usize>,
    /// Last capture (or the start) the periodic distance counts from.
    periodic_due_from: Option<usize>,
    /// The request's single periodic capture was taken or refused.
    periodic_done: bool,
    radix: Option<Qwen4RadixCache>,
}

impl Qwen4PrefixCache {
    fn new(max_seq_len: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(max_seq_len),
            prompt_len: 0,
            live_len: None,
            mode: Qwen4PrefixMode::Ar,
            chunk: 0,
            published: false,
            capture_at: None,
            generation: 0,
            slots: Vec::new(),
            slots_allocated: 0,
            staged: Vec::new(),
            local: None,
            selection: None,
            held: None,
            deferred: Vec::new(),
            bank: 0,
            next_bank: 1,
            turn_boundaries: Vec::new(),
            anchor: None,
            periodic_due_from: None,
            periodic_done: false,
            radix: None,
        }
    }

    /// Drop every record and unpublish.
    fn clear(&mut self) {
        self.tokens.clear();
        self.prompt_len = 0;
        self.live_len = None;
        self.published = false;
        self.capture_at = None;
        self.anchor = None;
        self.periodic_due_from = None;
        self.periodic_done = false;
        self.generation += 1;
    }

    /// Published for this `mode` under the chunk the device now admits.
    fn current(&self, marks: &PrefixMarks, mode: Qwen4PrefixMode) -> bool {
        self.published && self.mode == mode && marks.chunk == Some(self.chunk)
    }

    /// End `L` of the live state when it is published and still exactly at `L`.
    fn live_end(&self, marks: &PrefixMarks, mode: Qwen4PrefixMode) -> Option<usize> {
        let end = self.live_len?;
        (self.current(marks, mode) && marks.live_at(end, mode)).then_some(end)
    }

    /// End `P` of the durable checkpoint when it is published and valid.
    fn checkpoint_end(&self, marks: &PrefixMarks, mode: Qwen4PrefixMode) -> Option<usize> {
        (self.current(marks, mode) && marks.checkpoint_at(self.prompt_len, mode))
            .then_some(self.prompt_len)
    }

    /// Longest valid published record: the live history, else the prompt.
    fn prefix_lineage(&self, marks: &PrefixMarks, mode: Qwen4PrefixMode) -> Option<&[u32]> {
        let end = self
            .live_end(marks, mode)
            .or_else(|| self.checkpoint_end(marks, mode))?;
        Some(&self.tokens[..end])
    }

    /// Bind the planner-selected `start` for `prompt`.
    fn bind(
        &self,
        marks: &PrefixMarks,
        prompt: &[u32],
        start: usize,
        mode: Qwen4PrefixMode,
    ) -> Result<Qwen4PrefixPlan, String> {
        let plan = |source| Qwen4PrefixPlan {
            start_pos: start,
            source,
            generation: self.generation,
        };
        if start == 0 {
            return Ok(plan(Qwen4PrefixSource::Cold));
        }
        if start >= prompt.len() {
            return Err(format!(
                "Qwen4 prefix start {start} leaves no suffix of a {}-token prompt",
                prompt.len()
            ));
        }
        let record_matches = |end: usize| prompt[..end] == self.tokens[..end];
        if self.live_end(marks, mode) == Some(start) && record_matches(start) {
            return Ok(plan(Qwen4PrefixSource::Live));
        }
        if self.checkpoint_end(marks, mode) == Some(start) && record_matches(start) {
            return Ok(plan(Qwen4PrefixSource::Prompt));
        }
        Err(format!(
            "Qwen4 prefix start {start} matches neither the live state ({:?}) nor the checkpoint ({:?})",
            self.live_end(marks, mode),
            self.checkpoint_end(marks, mode)
        ))
    }

    /// Publish after the client committed the request whose full host history
    /// is `consumed`. The live state is published iff `consumed` extends the
    /// staged prompt, the target (and for native MTP, the head) are exactly at
    /// its end, and no row capture is armed; the record then becomes
    /// `consumed`, extended in place. Otherwise the record falls back to the
    /// prompt. Published = live valid or the checkpoint's arenas valid.
    fn commit(&mut self, marks: &PrefixMarks, consumed: &[u32]) {
        self.capture_at = None;
        let p = self.prompt_len;
        let live = p > 0
            && consumed.len() >= p
            && marks.live_at(consumed.len(), self.mode)
            && consumed[..p] == self.tokens[..p];
        self.tokens.truncate(p);
        if live {
            self.tokens.extend_from_slice(&consumed[p..]);
            self.live_len = Some(consumed.len());
        } else {
            self.live_len = None;
        }
        self.published = live || marks.checkpoint_at(p, self.mode);
    }
}

/// Tokens between periodic checkpoints (`anchor-v1-n8192`).
const PERIODIC_CAPTURE_TOKENS: usize = 8192;
/// Smallest turn boundary an anchor may split at.
const ANCHOR_MIN_TOKENS: usize = 128;
/// Hard bound on radix tree nodes.
const RADIX_INDEX_NODES: usize = 65_536;
/// Descriptor pages minted for published checkpoints (metadata only).
const RADIX_PAGE_POOL_PAGES: usize = 32_768;
/// Stale checkpoint ids skipped by one selection before giving up.
const RADIX_LOOKUP_RETRIES: usize = 8;
/// Host metadata estimate per radix node.
const INDEX_NODE_HOST_BYTES: usize = 256;
/// Alignment of each frontier entry of a retained context's slab.
const FRONTIER_SLAB_ALIGN: usize = 256;

/// Budgets of the shared radix store, fixed at attach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen4RadixLimits {
    /// Device bytes the cache may hold above the baseline: extra checkpoint
    /// slots, packed frontier slabs and retained granules the live bank left.
    pub device_bytes: u64,
    /// Host metadata bytes (exact token records plus index nodes).
    pub host_bytes: usize,
    /// Checkpoint slots (including the baseline one) and pool entries.
    pub checkpoints: usize,
}

/// Counters of the radix store, reported by [`Qwen4Bundle::radix_stats`].
#[derive(Clone, Debug, Default)]
pub struct Qwen4RadixStats {
    pub lookups: u64,
    pub hits_live: u64,
    pub hits_prompt: u64,
    pub hits_radix: u64,
    pub forks: u64,
    pub fresh_banks: u64,
    pub captures: u64,
    pub anchors: u64,
    pub periodic: u64,
    pub published: u64,
    pub duplicates: u64,
    pub evictions: u64,
    /// Sum over forks of the aliased (uncopied) bytes.
    pub alias_bytes: u64,
    /// Sum over forks of the copied frontier bytes.
    pub copied_bytes: u64,
    /// Cache-charged device bytes at the last budget check.
    pub ledger_bytes: u64,
    pub peak_ledger_bytes: u64,
    pub relocations: u64,
    pub relocation_ns_last: u64,
    /// Process-wide retired VA bytes when the stats were read.
    pub retired_va_bytes: u64,
    pub banks_created: u64,
}

/// A reusable bounded checkpoint slot: target state (+ native head) payload.
struct Qwen4CheckpointSlot {
    target: Qwen4TargetCheckpoint,
    head: Option<Qwen4HeadCheckpoint>,
}

impl Qwen4CheckpointSlot {
    fn device_bytes(&self) -> u64 {
        self.target.device_bytes() as u64
            + self
                .head
                .as_ref()
                .map_or(0, |head| head.device_bytes() as u64)
    }

    fn invalidate(&mut self) {
        self.target.invalidate();
        if let Some(head) = self.head.as_mut() {
            head.invalidate();
        }
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let Qwen4CheckpointSlot { target, head } = self;
        let target = target.free_gpu(gpu).map_err(BundleError::State);
        let head = match head {
            Some(head) => head
                .free_gpu(gpu)
                .map_err(|error| BundleError::Forward(error.to_string())),
            None => Ok(()),
        };
        target.and(head)
    }
}

/// One immutable checkpoint at exact position `B = tokens.len()`.
pub(crate) struct Qwen4Checkpoint {
    slot: Qwen4CheckpointSlot,
    /// `None` = local-only (budget refused): same-bank restore only, never
    /// published.
    context: Option<VmmPrefixSet>,
    /// Exactly the consumed prefix `[0, B)`.
    tokens: Arc<[u32]>,
    mode: Qwen4PrefixMode,
    chunk: usize,
    /// Bank id whose rows it was captured from.
    bank: u64,
    /// That bank's rows `[0, B)` are unchanged since capture.
    same_bank: bool,
    /// Captured at the end of its request's prompt.
    eop: bool,
    /// Begins that restored from it.
    hits: u32,
    /// Recompute cost in token equivalents ([`gdsf_cost`]).
    cost: f64,
    /// GDSF eviction priority ([`gdsf_priority`]).
    priority: f64,
}

impl Qwen4Checkpoint {
    fn slab_bytes(&self) -> u64 {
        self.context
            .as_ref()
            .map_or(0, |context| context.slab_bytes() as u64)
    }

    /// Bytes the eviction priority divides by: slot + slab + aliased granules.
    fn gdsf_bytes(&self) -> u64 {
        self.slot
            .device_bytes()
            .saturating_add(self.slab_bytes())
            .saturating_add(
                self.context
                    .as_ref()
                    .map_or(0, |context| context.alias_bytes() as u64),
            )
    }
}

impl CheckpointBlob for Qwen4Checkpoint {
    fn bytes_len(&self) -> u64 {
        self.slot.device_bytes().saturating_add(self.slab_bytes())
    }
}

/// The request's end-of-prompt checkpoint outside the pool.
enum LocalEop {
    /// Owned here (no context, or its publication failed).
    Owned(Qwen4Checkpoint),
    /// A reference to an entry of the radix pool.
    Published(CheckpointId),
}

/// Where a begin restores from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CheckpointRef {
    /// The staged (unpublished) checkpoint at this index.
    Staged(usize),
    LocalOwned,
    Pool(CheckpointId),
}

/// What [`Qwen4Bundle::begin_prefix`] does once its plan validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BeginAction {
    Cold,
    Live,
    Restore(CheckpointRef),
}

/// Outcome of one boundary capture.
enum Capture {
    Staged(Qwen4Checkpoint),
    /// The pool already holds this exact prefix.
    Duplicate(CheckpointId),
    /// Optional capture dropped (no slot / budget refused / copy failed).
    Skipped,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaptureKind {
    EndOfPrompt,
    Anchor,
    Periodic,
}

impl CaptureKind {
    fn label(self) -> &'static str {
        match self {
            Self::EndOfPrompt => "eop",
            Self::Anchor => "anchor",
            Self::Periodic => "periodic",
        }
    }
}

/// What the bank bookkeeping does after a restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BankAction {
    /// Same bank, rewound to `position`: later checkpoints lose their rows.
    Rewound(usize),
    /// A new bank replaced the old one.
    Replaced,
}

/// Counters of one restore, for stats and the begin trace.
#[derive(Default)]
struct RestoreReport {
    fork: bool,
    bank_replaced: bool,
    fresh_bank: bool,
    alias_bytes: usize,
    copied_bytes: usize,
    relocation_ns: u64,
    attach_ns: u64,
}

/// A failed restore: `live_modified` = a live owner was already written.
struct RestoreFailure {
    error: BundleError,
    live_modified: bool,
}

impl RestoreFailure {
    fn clean(error: BundleError) -> Self {
        Self {
            error,
            live_modified: false,
        }
    }

    fn dirty(error: BundleError) -> Self {
        Self {
            error,
            live_modified: true,
        }
    }

    fn from_swap(failure: SwapFailure) -> Self {
        Self {
            error: failure.error,
            live_modified: failure.live_modified,
        }
    }
}

struct BankSwap {
    relocation_ns: u64,
    /// Freeing the replaced banks failed (the swap itself completed).
    cleanup: Option<BundleError>,
}

struct SwapFailure {
    error: BundleError,
    live_modified: bool,
}

/// A pending (unconsumed) or held (running request) plan receipt and its pins.
struct Qwen4Selection {
    /// The pinned pool entry (`NONE` for a local or cold selection).
    id: CheckpointId,
    start: usize,
    mode: Qwen4PrefixMode,
    source: Qwen4PrefixSource,
    /// The index path pin of a radix lookup (empty otherwise).
    pin: PinTicket,
    generation: u64,
}

impl Qwen4Selection {
    fn plan(&self) -> Qwen4PrefixPlan {
        Qwen4PrefixPlan {
            start_pos: self.start,
            source: self.source,
            generation: self.generation,
        }
    }
}

/// The shared, byte-bounded radix store of one bundle.
struct Qwen4RadixCache {
    /// `[Ar, NativeMtp]` (see [`mode_index`]).
    domains: [CacheDomain; 2],
    limits: Qwen4RadixLimits,
    index: PrefixIndex,
    /// Token-page descriptors only: no backing arena.
    pages: PagePool,
    /// Exact-boundary checkpoints. Uncapped: the ledger enforces bytes.
    pool: QwenCheckpointPool<Qwen4Checkpoint>,
    stats: Qwen4RadixStats,
    /// Device bytes of one checkpoint slot.
    slot_bytes: u64,
    /// Physical granules the live banks map (charged by the baseline).
    live_granules: HashSet<VmmPhysicalId>,
    /// Retired ids whose pool entry was still pinned.
    retire_pending: Vec<CheckpointId>,
    /// GDSF aging clock: the priority of the last evicted checkpoint.
    gdsf_clock: f64,
}

impl Qwen4RadixCache {
    /// The pool entry `id` is a usable restore source for `prompt[..b]`.
    fn checkpoint_valid(
        &self,
        id: CheckpointId,
        b: usize,
        prompt: &[u32],
        mode: Qwen4PrefixMode,
        chunk: usize,
    ) -> bool {
        let (Some(entry), Some(tokens)) = (self.pool.peek_id(id), self.pool.exact_tokens(id))
        else {
            return false;
        };
        checkpoint_facts_valid(
            &CheckpointFacts {
                tokens,
                mode: entry.mode,
                chunk: entry.chunk,
                has_context: entry.context.is_some(),
                target_position: entry.slot.target.position(),
                head_position: entry.slot.head.as_ref().and_then(|head| head.position()),
            },
            prompt,
            b,
            mode,
            chunk,
        )
    }
}

// ---- pure decisions over plain data ----

/// Index of `mode`'s domain in [`Qwen4RadixCache::domains`].
fn mode_index(mode: Qwen4PrefixMode) -> usize {
    match mode {
        Qwen4PrefixMode::Ar => 0,
        Qwen4PrefixMode::NativeMtp => 1,
    }
}

/// The winner of the local candidate (`local`, ending at `local_start`) and the
/// deepest valid radix checkpoint `radix`: live wins when it reaches the radix
/// boundary, a local prompt wins ties, otherwise the deeper radix boundary.
fn pick_source(
    local: Qwen4PrefixSource,
    local_start: usize,
    radix: Option<usize>,
) -> Qwen4PrefixSource {
    match (local, radix) {
        (Qwen4PrefixSource::Live, Some(b)) if local_start >= b => Qwen4PrefixSource::Live,
        (Qwen4PrefixSource::Live, None) => Qwen4PrefixSource::Live,
        (Qwen4PrefixSource::Prompt, Some(b)) if local_start >= b => Qwen4PrefixSource::Prompt,
        (Qwen4PrefixSource::Prompt, None) => Qwen4PrefixSource::Prompt,
        (_, Some(_)) => Qwen4PrefixSource::Radix,
        (_, None) => Qwen4PrefixSource::Cold,
    }
}

/// Sorted, deduplicated turn boundaries below `max_seq`.
fn normalized_turn_boundaries(boundaries: &[usize], max_seq: usize) -> Vec<usize> {
    let mut sorted: Vec<usize> = boundaries
        .iter()
        .copied()
        .filter(|&boundary| boundary < max_seq)
        .collect();
    sorted.sort_unstable();
    sorted.dedup();
    sorted
}

/// The conditional shared-turn anchor (design §4.1): the first turn boundary
/// `S` with `128 <= S < prompt_len` and `S > start`, armed only when a
/// committed path already covers `S` (`index_covers`), no checkpoint exists at
/// `S` (the deepest radix checkpoint is below it) and a slot/budget is
/// available. A first-ever cold request finds nothing in the index: no anchor.
fn anchor_choice(
    boundaries: &[usize],
    prompt_len: usize,
    start: usize,
    index_covers: impl Fn(usize) -> bool,
    radix_best: Option<usize>,
    resources_ok: bool,
) -> Option<usize> {
    let s = boundaries
        .iter()
        .copied()
        .find(|&s| s >= ANCHOR_MIN_TOKENS && s < prompt_len && s > start)?;
    (resources_ok && radix_best.is_none_or(|best| best < s) && index_covers(s)).then_some(s)
}

/// Look up the deepest valid checkpoint `B < prompt_len`: a hit that fails
/// `valid` is discarded (its pin released) and the lookup retried below its
/// boundary, at most [`RADIX_LOOKUP_RETRIES`] times. `lookup(ctx, before)`
/// returns the hit and its boundary.
fn lookup_valid_checkpoint<C, H>(
    ctx: &mut C,
    prompt_len: usize,
    lookup: impl Fn(&mut C, usize) -> Option<(H, usize)>,
    valid: impl Fn(&C, &H, usize) -> bool,
    discard: impl Fn(&mut C, H),
) -> Option<(H, usize)> {
    let mut before = prompt_len;
    for _ in 0..RADIX_LOOKUP_RETRIES {
        if before == 0 {
            return None;
        }
        let (hit, boundary) = lookup(ctx, before)?;
        if boundary == 0 || boundary >= before {
            discard(ctx, hit);
            return None;
        }
        if valid(ctx, &hit, boundary) {
            return Some((hit, boundary));
        }
        discard(ctx, hit);
        before = boundary;
    }
    None
}

/// Plain facts of a pool entry [`checkpoint_facts_valid`] checks.
struct CheckpointFacts<'a> {
    tokens: &'a [u32],
    mode: Qwen4PrefixMode,
    chunk: usize,
    has_context: bool,
    target_position: Option<usize>,
    head_position: Option<usize>,
}

/// The entry holds exactly `prompt[..b]`'s state for `mode` under `chunk`:
/// exact tokens, retained context, target (and, for native MTP, head) at `b`.
fn checkpoint_facts_valid(
    facts: &CheckpointFacts<'_>,
    prompt: &[u32],
    b: usize,
    mode: Qwen4PrefixMode,
    chunk: usize,
) -> bool {
    b > 0
        && b < prompt.len()
        && facts.tokens.len() == b
        && facts.tokens == &prompt[..b]
        && facts.mode == mode
        && facts.chunk == chunk
        && facts.has_context
        && facts.target_position == Some(b)
        && (mode == Qwen4PrefixMode::Ar || facts.head_position == Some(b))
}

/// Plain inputs of [`Qwen4Bundle::next_prefix_capture`].
#[derive(Clone, Copy, Debug)]
struct CaptureSchedule {
    /// End of the armed prompt; `None` = nothing armed.
    capture_at: Option<usize>,
    anchor: Option<usize>,
    radix: bool,
    periodic_done: bool,
    /// Last capture (or the start) periodic distance is measured from.
    periodic_due_from: Option<usize>,
    /// A slot and budget exist for an optional capture.
    resources_ok: bool,
}

impl CaptureSchedule {
    /// The boundary the chunk `(position, natural_end]` must end at, if any.
    fn next(&self, position: usize, natural_end: usize) -> Option<usize> {
        let prompt_end = self.capture_at?;
        if natural_end <= position {
            return None;
        }
        if let Some(anchor) = self.anchor {
            if position < anchor && anchor <= natural_end {
                return Some(anchor);
            }
        }
        if natural_end == prompt_end {
            return Some(natural_end);
        }
        if self.radix && !self.periodic_done && self.resources_ok && natural_end < prompt_end {
            let due = self
                .periodic_due_from
                .and_then(|from| natural_end.checked_sub(from))
                .is_some_and(|distance| distance >= PERIODIC_CAPTURE_TOKENS);
            if due {
                return Some(natural_end);
            }
        }
        None
    }
}

/// A slot can be obtained for an optional capture: a free one, a new one
/// within the count and byte budget, or an evictable entry.
fn capture_resources_ok(
    free_slots: usize,
    slots_allocated: usize,
    slot_limit: usize,
    evictable: usize,
    ledger: u64,
    slot_bytes: u64,
    device_bytes: u64,
) -> bool {
    free_slots > 0
        || (slots_allocated < slot_limit && ledger.saturating_add(slot_bytes) <= device_bytes)
        || evictable > 0
}

/// Largest granule boundary `<= limit` over consecutive granule sizes.
fn alias_boundary(granule_sizes: impl IntoIterator<Item = usize>, limit: usize) -> usize {
    let mut end = 0usize;
    for size in granule_sizes {
        match end.checked_add(size) {
            Some(next) if size != 0 && next <= limit => end = next,
            _ => break,
        }
    }
    end
}

/// Slab bytes one arena's frontier copy needs: valid bytes past the aliased
/// granule boundary, rounded to the slab alignment.
fn slab_estimate(
    valid_bytes: usize,
    writable_from: usize,
    granule_sizes: impl IntoIterator<Item = usize>,
) -> u64 {
    let alias = alias_boundary(granule_sizes, valid_bytes.min(writable_from));
    let frontier = (valid_bytes - alias) as u64;
    let align = FRONTIER_SLAB_ALIGN as u64;
    frontier.div_ceil(align) * align
}

fn slab_estimate_total(gpu: &Gpu, specs: &[VmmPrefixSpec<'_>]) -> u64 {
    specs
        .iter()
        .map(|spec| {
            let sizes = gpu.vmm_physical_granules(spec.tensor).unwrap_or_default();
            slab_estimate(
                spec.valid_bytes,
                spec.writable_from,
                sizes.into_iter().map(|(_, bytes)| bytes),
            )
        })
        .fold(0u64, u64::saturating_add)
}

/// Bytes of the unique physical granules `contexts` reference that no live
/// bank maps: each counted once however many checkpoints share it.
fn cache_granule_bytes(
    contexts: impl IntoIterator<Item = (VmmPhysicalId, usize)>,
    live: &HashSet<VmmPhysicalId>,
) -> u64 {
    let mut unique: HashMap<VmmPhysicalId, usize> = HashMap::new();
    for (id, bytes) in contexts {
        if !live.contains(&id) {
            unique.entry(id).or_insert(bytes);
        }
    }
    unique.values().map(|&bytes| bytes as u64).sum()
}

/// Cache-charged device bytes: extra slots, slabs and retained granules.
fn ledger_total(extra_slot_bytes: u64, slab_bytes: u64, granule_bytes: u64) -> u64 {
    extra_slot_bytes
        .saturating_add(slab_bytes)
        .saturating_add(granule_bytes)
}

/// A budget is exceeded: device bytes (only checked with a GPU), entry count
/// or host bytes.
fn over_budget(
    device_over: bool,
    entries: usize,
    max_entries: usize,
    host_bytes: usize,
    max_host_bytes: usize,
) -> bool {
    device_over || entries > max_entries || host_bytes > max_host_bytes
}

/// How one eviction step picks its victim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvictMode {
    /// Evict the lowest-priority unpinned checkpoint.
    Checkpoint,
    /// Prune a pageless/checkpointless tree leaf first (node/host bound).
    PruneFirst,
}

fn evict_mode(device_over: bool, count_over: bool) -> EvictMode {
    if device_over || count_over {
        EvictMode::Checkpoint
    } else {
        EvictMode::PruneFirst
    }
}

/// Recompute cost of a checkpoint at `b` in token equivalents: prefill grows
/// superlinearly with context, and the rebuild replays the whole prefix.
fn gdsf_cost(b: usize) -> f64 {
    let b = b as f64;
    b * (1.0 + b / 65536.0)
}

/// GDSF priority: the aging clock plus the recompute saved per byte, weighted
/// by hits.
fn gdsf_priority(clock: f64, cost: f64, hits: u32, bytes: u64) -> f64 {
    clock + cost * (1.0 + f64::from(hits)) / bytes.max(1) as f64
}

/// The lowest-priority unpinned entry (`(id, priority, pinned)`); ties go to
/// the lowest id.
fn gdsf_victim(entries: &[(CheckpointId, f64, bool)]) -> Option<CheckpointId> {
    entries
        .iter()
        .filter(|(_, _, pinned)| !pinned)
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
        .map(|(id, _, _)| *id)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    out
}

fn prefix_mode_tag(mode: Qwen4PrefixMode) -> &'static str {
    match mode {
        Qwen4PrefixMode::Ar => "ar",
        Qwen4PrefixMode::NativeMtp => "native-mtp",
    }
}

/// Versioned, length-delimited state ABI tag of a radix domain.
fn radix_state_abi_tag(
    base_tag: &str,
    mode: Qwen4PrefixMode,
    chunk: usize,
    gdn_format: &str,
    qsa_format: &str,
) -> String {
    let chunk = (chunk as u64).to_le_bytes();
    let digest = sha256_len_prefixed(&[
        "qwen4-radix-v1".as_bytes(),
        base_tag.as_bytes(),
        prefix_mode_tag(mode).as_bytes(),
        &chunk,
        gdn_format.as_bytes(),
        qsa_format.as_bytes(),
        "head-f32".as_bytes(),
        "anchor-v1-n8192".as_bytes(),
    ]);
    format!("qwen4-radix-v1:{}", hex(&digest))
}

/// The `mode` domain derived from the load's base domain: the state ABI tag
/// binds mode, admitted chunk, state formats and the anchor policy; the layout
/// strides are the per-arena row strides.
fn build_radix_domain(
    base: &CacheDomain,
    mode: Qwen4PrefixMode,
    chunk: usize,
    gdn_format: &str,
    qsa_format: &str,
    arena_strides: &[u64],
) -> CacheDomain {
    let mut domain = base.clone();
    domain.arch_policy.state_abi_tag = radix_state_abi_tag(
        &base.arch_policy.state_abi_tag,
        mode,
        chunk,
        gdn_format,
        qsa_format,
    );
    domain.kv_layout.k_stride_bytes = arena_strides.to_vec();
    domain.kv_layout.v_stride_bytes = arena_strides.to_vec();
    domain
}

/// Row stride of one context arena of `rows` rows.
fn arena_row_stride(tensor: &GpuTensor, rows: usize) -> Result<u64, BundleError> {
    let bytes = tensor.byte_size();
    if rows == 0 || bytes % rows != 0 {
        return Err(BundleError::Forward(format!(
            "Qwen4 context arena of {bytes} bytes does not divide into {rows} rows"
        )));
    }
    Ok((bytes / rows) as u64)
}

/// Logical per-position `(K side, V side)` bytes of the page pool: K rows and
/// raw index rows, V rows and pooled index rows amortized per token. Strides
/// are in arena order (K, V, raw, pooled per layer).
fn logical_page_strides(arena_strides: &[u64], compress: usize) -> (usize, usize) {
    let (mut k, mut v) = (0u64, 0u64);
    for (index, &stride) in arena_strides.iter().enumerate() {
        match index % 4 {
            0 | 2 => k += stride,
            1 => v += stride,
            _ => v += stride.div_ceil(compress.max(1) as u64),
        }
    }
    (k.max(1) as usize, v.max(1) as usize)
}

/// The checkpoint `source` names, borrowed from disjoint cache fields.
fn resolve_checkpoint<'a>(
    staged: &'a [Qwen4Checkpoint],
    local: &'a Option<LocalEop>,
    pool: Option<&'a QwenCheckpointPool<Qwen4Checkpoint>>,
    source: CheckpointRef,
) -> Option<&'a Qwen4Checkpoint> {
    match source {
        CheckpointRef::Staged(index) => staged.get(index),
        CheckpointRef::LocalOwned => match local {
            Some(LocalEop::Owned(checkpoint)) => Some(checkpoint),
            _ => None,
        },
        CheckpointRef::Pool(id) => pool?.peek_id(id),
    }
}

/// Physical granule ids the live banks map.
fn live_granule_ids(
    state: &Qwen4State,
    mtp: Option<&Qwen4MtpGpu>,
    gpu: &Gpu,
) -> HashSet<VmmPhysicalId> {
    let head = mtp.map(Qwen4MtpGpu::context_tensors).unwrap_or_default();
    let mut ids = HashSet::new();
    for tensor in state.context_tensors().into_iter().chain(head) {
        if let Some(granules) = gpu.vmm_physical_granules(tensor) {
            ids.extend(granules.into_iter().map(|(id, _)| id));
        }
    }
    ids
}

/// Free replaced or abandoned context banks; the first error.
fn free_banks(
    gpu: &mut Gpu,
    target: Option<Vec<GpuTensor>>,
    head: Option<Vec<GpuTensor>>,
) -> Option<BundleError> {
    let mut first = None;
    if let Some(bank) = target {
        if let Err(error) = Qwen4State::free_context_bank(gpu, bank) {
            first.get_or_insert(BundleError::State(error));
        }
    }
    if let Some(bank) = head {
        if let Err(error) = Qwen4MtpGpu::free_context_bank(gpu, bank) {
            first.get_or_insert(BundleError::Forward(error.to_string()));
        }
    }
    first
}

/// Install fresh `target` / `head` context banks: pair every old and new
/// tensor, relocate the replay and graph bindings (old and new both alive),
/// install, then free the replaced banks. Failure before the relocation frees
/// the new banks and leaves the owners untouched.
fn swap_context_banks(
    state: &mut Qwen4State,
    mut mtp: Option<&mut Qwen4MtpGpu>,
    gpu: &mut Gpu,
    target: Option<Vec<GpuTensor>>,
    head: Option<Vec<GpuTensor>>,
) -> Result<BankSwap, SwapFailure> {
    let mut moves: Vec<VmmResourceMove> = Vec::new();
    let mut problem: Option<String> = None;
    if let Some(bank) = target.as_ref() {
        let old = state.context_tensors();
        if old.len() != bank.len() {
            problem = Some(format!(
                "Qwen4 target bank has {} arenas, expected {}",
                bank.len(),
                old.len()
            ));
        } else {
            for (old, new) in old.into_iter().zip(bank) {
                match gpu.vmm_resource_move(old, new) {
                    Some(mv) => moves.push(mv),
                    None => {
                        problem = Some("Qwen4 context bank tensor is not a VMM owner".to_string());
                        break;
                    }
                }
            }
        }
    }
    if problem.is_none() {
        if let Some(bank) = head.as_ref() {
            match mtp.as_deref() {
                Some(m) => {
                    let old = m.context_tensors();
                    if old.len() != bank.len() {
                        problem = Some(format!(
                            "Qwen4 head bank has {} arenas, expected {}",
                            bank.len(),
                            old.len()
                        ));
                    } else {
                        for (old, new) in old.into_iter().zip(bank) {
                            match gpu.vmm_resource_move(old, new) {
                                Some(mv) => moves.push(mv),
                                None => {
                                    problem = Some(
                                        "Qwen4 head bank tensor is not a VMM owner".to_string(),
                                    );
                                    break;
                                }
                            }
                        }
                    }
                }
                None => problem = Some("Qwen4 MTP resources are not attached".to_string()),
            }
        }
    }
    if let Some(message) = problem {
        let cleanup = free_banks(gpu, target, head);
        let note = cleanup.map_or(String::new(), |e| format!("; cleanup failed: {e}"));
        return Err(SwapFailure {
            error: BundleError::Forward(format!("{message}{note}")),
            live_modified: false,
        });
    }
    let started = Instant::now();
    if let Err(error) = gpu.relocate_qsa_resources(&moves) {
        let cleanup = free_banks(gpu, target, head);
        let note = cleanup.map_or(String::new(), |e| format!("; cleanup failed: {e}"));
        return Err(SwapFailure {
            error: BundleError::Forward(format!("Qwen4 bank relocation failed: {error}{note}")),
            live_modified: false,
        });
    }
    let relocation_ns = started.elapsed().as_nanos() as u64;
    let mut old_target = None;
    let mut old_head = None;
    if let Some(bank) = target {
        match state.install_context_bank(gpu, bank) {
            Ok(old) => old_target = Some(old),
            Err(error) => {
                let _ = free_banks(gpu, None, head);
                return Err(SwapFailure {
                    error: BundleError::State(error),
                    live_modified: true,
                });
            }
        }
    }
    if let Some(bank) = head {
        let installed = match mtp.as_deref_mut() {
            Some(m) => m
                .install_context_bank(gpu, bank)
                .map_err(|error| BundleError::Forward(error.to_string())),
            None => Err(BundleError::Forward(
                "Qwen4 MTP resources are not attached".to_string(),
            )),
        };
        match installed {
            Ok(old) => old_head = Some(old),
            Err(error) => {
                let _ = free_banks(gpu, old_target, None);
                return Err(SwapFailure {
                    error,
                    live_modified: true,
                });
            }
        }
    }
    let cleanup = free_banks(gpu, old_target, old_head);
    Ok(BankSwap {
        relocation_ns,
        cleanup,
    })
}

/// Release a checkpoint's context now (GPU quiescent) and return its slot.
fn recycle_checkpoint(
    slots: &mut Vec<Qwen4CheckpointSlot>,
    gpu: &mut Gpu,
    checkpoint: Qwen4Checkpoint,
) -> Result<(), BundleError> {
    let Qwen4Checkpoint {
        mut slot, context, ..
    } = checkpoint;
    slot.invalidate();
    slots.push(slot);
    context.map_or(Ok(()), |context| {
        gpu.free_vmm_prefixes(context).map_err(BundleError::Hip)
    })
}

/// Release a checkpoint's context and its slot's device memory (unload).
fn free_checkpoint_final(gpu: &mut Gpu, checkpoint: Qwen4Checkpoint) -> Result<(), BundleError> {
    let Qwen4Checkpoint { slot, context, .. } = checkpoint;
    let context = context.map_or(Ok(()), |context| {
        gpu.free_vmm_prefixes(context).map_err(BundleError::Hip)
    });
    context.and(slot.free_gpu(gpu))
}

fn keep_first(first: &mut Option<BundleError>, result: Result<(), BundleError>) {
    if let Err(error) = result {
        first.get_or_insert(error);
    }
}

impl Qwen4PrefixCache {
    /// Release the pins of a selection.
    fn release_pins(radix: &mut Option<Qwen4RadixCache>, selection: Qwen4Selection) {
        if let Some(radix) = radix.as_mut() {
            radix
                .index
                .release_pin(&radix.domains[mode_index(selection.mode)], selection.pin);
            if selection.id.is_some() {
                radix.pool.unpin_id(selection.id);
            }
        }
    }

    /// Release an unconsumed selection's pins.
    fn release_pending(&mut self) {
        if let Some(selection) = self.selection.take() {
            Self::release_pins(&mut self.radix, selection);
        }
    }

    /// Release the running request's pins (commit, reset, a failed begin).
    fn release_held(&mut self) {
        if let Some(selection) = self.held.take() {
            Self::release_pins(&mut self.radix, selection);
        }
    }

    /// Return a host-only-dropped checkpoint to the free slots, or defer it
    /// while it still retains a context (freed at the next quiescent point).
    fn retire_host(&mut self, checkpoint: Qwen4Checkpoint) {
        if checkpoint.context.is_some() {
            self.deferred.push(checkpoint);
        } else {
            let mut slot = checkpoint.slot;
            slot.invalidate();
            self.slots.push(slot);
        }
    }

    /// Replace the local end-of-prompt checkpoint.
    fn set_local(&mut self, local: LocalEop) {
        if let Some(LocalEop::Owned(old)) = self.local.replace(local) {
            self.retire_host(old);
        }
    }

    /// Drop the local end-of-prompt checkpoint (a pool entry survives).
    fn drop_local(&mut self) {
        if let Some(LocalEop::Owned(old)) = self.local.take() {
            self.retire_host(old);
        }
    }

    /// The end-of-prompt checkpoint a restore may use: this request's staged
    /// one, else the local one.
    fn eop_ref(&self) -> Option<(CheckpointRef, &Qwen4Checkpoint)> {
        if let Some(index) = self.staged.iter().position(|checkpoint| checkpoint.eop) {
            return Some((CheckpointRef::Staged(index), &self.staged[index]));
        }
        match self.local.as_ref()? {
            LocalEop::Owned(checkpoint) => Some((CheckpointRef::LocalOwned, checkpoint)),
            LocalEop::Published(id) => self
                .radix
                .as_ref()?
                .pool
                .peek_id(*id)
                .map(|checkpoint| (CheckpointRef::Pool(*id), checkpoint)),
        }
    }

    /// `(source, B, mode)` of the end-of-prompt checkpoint when it is
    /// restorable under the armed mode: payloads at exactly `B`, and either its
    /// bank's rows still unchanged or a retained context to fork from.
    fn eop_facts(&self) -> Option<(CheckpointRef, usize, Qwen4PrefixMode)> {
        let (source, checkpoint) = self.eop_ref()?;
        let b = checkpoint.tokens.len();
        let restorable = checkpoint.mode == self.mode
            && b > 0
            && checkpoint.slot.target.position() == Some(b)
            && (checkpoint.mode == Qwen4PrefixMode::Ar
                || checkpoint
                    .slot
                    .head
                    .as_ref()
                    .is_some_and(|head| head.position() == Some(b)))
            && ((checkpoint.same_bank && checkpoint.bank == self.bank)
                || checkpoint.context.is_some());
        restorable.then_some((source, b, checkpoint.mode))
    }

    fn capture_schedule(&self) -> CaptureSchedule {
        CaptureSchedule {
            capture_at: self.capture_at,
            anchor: self.anchor,
            radix: self.radix.is_some(),
            periodic_done: self.periodic_done,
            periodic_due_from: self.periodic_due_from,
            resources_ok: self.capture_resources_available(),
        }
    }

    /// Conservative (last ledger) check that an optional capture finds a slot.
    fn capture_resources_available(&self) -> bool {
        self.radix.as_ref().is_some_and(|radix| {
            capture_resources_ok(
                self.slots.len(),
                self.slots_allocated,
                radix.limits.checkpoints,
                radix.pool.len(),
                radix.stats.ledger_bytes,
                radix.slot_bytes,
                radix.limits.device_bytes,
            )
        })
    }

    /// Open a new bank id; returns the previous one.
    fn start_bank(&mut self) -> u64 {
        let old = self.bank;
        self.bank = self.next_bank;
        self.next_bank += 1;
        old
    }

    fn for_each_checkpoint_mut(&mut self, mut apply: impl FnMut(&mut Qwen4Checkpoint)) {
        for checkpoint in &mut self.staged {
            apply(checkpoint);
        }
        if let Some(LocalEop::Owned(checkpoint)) = self.local.as_mut() {
            apply(checkpoint);
        }
        if let Some(radix) = self.radix.as_mut() {
            for id in radix.pool.exact_ids() {
                if let Some(checkpoint) = radix.pool.peek_id_mut(id) {
                    apply(checkpoint);
                }
            }
        }
    }

    /// The rows of bank `old` are gone or overwritten.
    fn bank_gone(&mut self, old: u64) {
        self.for_each_checkpoint_mut(|checkpoint| {
            if checkpoint.bank == old {
                checkpoint.same_bank = false;
            }
        });
    }

    /// Bank `bank` was rewound in place to `position`: checkpoints beyond it
    /// lose their rows (the suffix overwrites them).
    fn bank_rewound(&mut self, bank: u64, position: usize) {
        self.for_each_checkpoint_mut(|checkpoint| {
            if checkpoint.bank == bank && checkpoint.tokens.len() > position {
                checkpoint.same_bank = false;
            }
        });
    }

    /// Every checkpoint the cache holds device memory for.
    fn retained(&self) -> impl Iterator<Item = &Qwen4Checkpoint> + '_ {
        let local = match &self.local {
            Some(LocalEop::Owned(checkpoint)) => Some(checkpoint),
            _ => None,
        };
        let pool = self.radix.as_ref();
        self.staged
            .iter()
            .chain(self.deferred.iter())
            .chain(local)
            .chain(pool.into_iter().flat_map(|radix| {
                radix
                    .pool
                    .exact_ids()
                    .into_iter()
                    .filter_map(move |id| radix.pool.peek_id(id))
            }))
    }

    /// Device bytes the cache is charged: extra slots, slabs, and the unique
    /// retained granules no live bank maps (from the last live refresh).
    fn ledger_bytes(&self) -> u64 {
        let Some(radix) = self.radix.as_ref() else {
            return 0;
        };
        let extra_slots = self.slots_allocated.saturating_sub(1) as u64 * radix.slot_bytes;
        let slabs = self
            .retained()
            .map(Qwen4Checkpoint::slab_bytes)
            .fold(0u64, u64::saturating_add);
        let granules = cache_granule_bytes(
            self.retained()
                .filter_map(|checkpoint| checkpoint.context.as_ref())
                .flat_map(VmmPrefixSet::physical_granules),
            &radix.live_granules,
        );
        ledger_total(extra_slots, slabs, granules)
    }

    /// Host bytes: exact token records and the index node estimate.
    fn host_bytes(&self) -> usize {
        let tokens: usize = self
            .retained()
            .map(|checkpoint| checkpoint.tokens.len() * std::mem::size_of::<u32>())
            .sum();
        let nodes = self.radix.as_ref().map_or(0, |radix| {
            radix.index.total_nodes().saturating_mul(INDEX_NODE_HOST_BYTES)
        });
        tokens.saturating_add(nodes)
    }

    /// Radix half of [`Qwen4Bundle::select_prefix_plan`].
    fn select_radix(
        &mut self,
        prompt: &[u32],
        local: Qwen4PrefixPlan,
        mode: Qwen4PrefixMode,
        chunk: usize,
    ) -> Result<Qwen4PrefixPlan, BundleError> {
        let di = mode_index(mode);
        let resources_ok = self.capture_resources_available();
        let generation = self.generation;
        let Some(radix) = self.radix.as_mut() else {
            return Ok(local);
        };
        radix.stats.lookups += 1;
        let found = lookup_valid_checkpoint(
            &mut *radix,
            prompt.len(),
            |radix: &mut Qwen4RadixCache, before| {
                radix
                    .index
                    .lookup_checkpoint(&radix.domains[di], prompt, before, &radix.pages)
                    .ok()
                    .map(|hit| {
                        let boundary = hit.lookup.resumable_tokens as usize;
                        (hit, boundary)
                    })
            },
            |radix: &Qwen4RadixCache, hit, boundary| {
                radix.checkpoint_valid(hit.checkpoint, boundary, prompt, mode, chunk)
            },
            |radix: &mut Qwen4RadixCache, hit| {
                radix.index.release_pin(&radix.domains[di], hit.pin);
            },
        );
        let mut best = None;
        if let Some((hit, boundary)) = found {
            if radix.pool.pin_id(hit.checkpoint) {
                best = Some((hit, boundary));
            } else {
                radix.index.release_pin(&radix.domains[di], hit.pin);
            }
        }
        let radix_best = best.as_ref().map(|(_, boundary)| *boundary);
        let source = pick_source(local.source, local.start_pos, radix_best);
        let (id, start, pin, source) = match (source, best) {
            (Qwen4PrefixSource::Radix, Some((hit, boundary))) => {
                (hit.checkpoint, boundary, hit.pin, Qwen4PrefixSource::Radix)
            }
            (_, losing) => {
                if let Some((hit, _)) = losing {
                    radix.index.release_pin(&radix.domains[di], hit.pin);
                    radix.pool.unpin_id(hit.checkpoint);
                }
                (
                    CheckpointId::NONE,
                    local.start_pos,
                    PinTicket::none(),
                    local.source,
                )
            }
        };
        let anchor = anchor_choice(
            &self.turn_boundaries,
            prompt.len(),
            start,
            |s| {
                radix
                    .index
                    .inspect(&radix.domains[di], &prompt[..s], &radix.pages)
                    .matched_tokens
                    >= s as u64
            },
            radix_best,
            resources_ok,
        );
        self.anchor = anchor;
        let selection = Qwen4Selection {
            id,
            start,
            mode,
            source,
            pin,
            generation,
        };
        let plan = selection.plan();
        self.selection = Some(selection);
        Ok(plan)
    }

    /// Host-only validation of `plan` for [`Qwen4Bundle::begin_prefix`].
    fn validate_plan(
        &self,
        marks: &PrefixMarks,
        prompt: &[u32],
        plan: Qwen4PrefixPlan,
        mode: Qwen4PrefixMode,
        selection: Option<&Qwen4Selection>,
    ) -> Result<BeginAction, BundleError> {
        let stale = || {
            BundleError::Forward(format!(
                "Qwen4 prefix plan at {} ({:?}) is no longer valid",
                plan.start_pos, plan.source
            ))
        };
        match plan.source {
            Qwen4PrefixSource::Cold => Ok(BeginAction::Cold),
            Qwen4PrefixSource::Live | Qwen4PrefixSource::Prompt => {
                let bound = self
                    .bind(marks, prompt, plan.start_pos, mode)
                    .map_err(BundleError::Forward)?;
                if bound != plan {
                    return Err(stale());
                }
                if plan.source == Qwen4PrefixSource::Live {
                    return Ok(BeginAction::Live);
                }
                match self.eop_facts() {
                    Some((source, b, m)) if b == plan.start_pos && m == mode => {
                        Ok(BeginAction::Restore(source))
                    }
                    _ => Err(stale()),
                }
            }
            Qwen4PrefixSource::Radix => {
                let (Some(selection), Some(radix), Some(chunk)) =
                    (selection, self.radix.as_ref(), marks.chunk)
                else {
                    return Err(stale());
                };
                if plan.generation != self.generation
                    || selection.id.is_none()
                    || radix.pool.pin_count(selection.id) == 0
                    || !radix.checkpoint_valid(selection.id, plan.start_pos, prompt, mode, chunk)
                {
                    return Err(stale());
                }
                Ok(BeginAction::Restore(CheckpointRef::Pool(selection.id)))
            }
        }
    }

    /// Count a begin that restored from `source` and re-prioritise it.
    fn note_restore_hit(&mut self, source: CheckpointRef) {
        if let (Some(radix), CheckpointRef::Pool(id)) = (self.radix.as_mut(), source) {
            let clock = radix.gdsf_clock;
            if let Some(checkpoint) = radix.pool.peek_id_mut(id) {
                checkpoint.hits = checkpoint.hits.saturating_add(1);
                checkpoint.priority = gdsf_priority(
                    clock,
                    checkpoint.cost,
                    checkpoint.hits,
                    checkpoint.gdsf_bytes(),
                );
            }
        }
    }

    /// Stats and trace of a staged checkpoint.
    fn note_capture(&mut self, checkpoint: &Qwen4Checkpoint, kind: CaptureKind, trace: bool) {
        let Some(radix) = self.radix.as_mut() else {
            return;
        };
        radix.stats.captures += 1;
        match kind {
            CaptureKind::Anchor => radix.stats.anchors += 1,
            CaptureKind::Periodic => radix.stats.periodic += 1,
            CaptureKind::EndOfPrompt => {}
        }
        if trace {
            let (alias, copied) = checkpoint
                .context
                .as_ref()
                .map_or((0, 0), |context| (context.alias_bytes(), context.frontier_bytes()));
            eprintln!(
                "[qwen4-radix] stage at={} kind={} alias={alias} copied={copied}",
                checkpoint.tokens.len(),
                kind.label()
            );
        }
    }

    /// Bookkeeping after an optional capture at `b` (taken or dropped).
    fn after_optional_capture(&mut self, b: usize, kind: CaptureKind) {
        self.periodic_due_from = Some(b);
        match kind {
            CaptureKind::Anchor => {
                if self.anchor == Some(b) {
                    self.anchor = None;
                }
            }
            CaptureKind::Periodic => self.periodic_done = true,
            CaptureKind::EndOfPrompt => {}
        }
    }

    /// Host-only publication of this request's staged checkpoints (see
    /// [`Qwen4Bundle::commit_prefix`]): the request's pins are released, every
    /// staged checkpoint is published or kept, retired ids are reclaimed into
    /// `deferred`, and the host/count caps are enforced.
    fn publish_staged(&mut self) {
        self.release_pending();
        self.release_held();
        self.anchor = None;
        let trace = hipfire_config::developer_var("HIPFIRE_QWEN_CACHE_TRACE")
            .is_ok_and(|value| value == "1");
        for checkpoint in std::mem::take(&mut self.staged) {
            let (at, eop) = (checkpoint.tokens.len(), checkpoint.eop);
            let published_before = self.radix.as_ref().map_or(0, |radix| radix.stats.published);
            self.publish_one(checkpoint);
            if trace {
                let published = self.radix.as_ref().map_or(0, |radix| radix.stats.published);
                eprintln!(
                    "[qwen4-radix] publish at={at} eop={eop} published={}",
                    published > published_before
                );
            }
        }
        let mut first_error = None;
        self.reclaim_retired(None, &mut first_error);
        if let Err(error) = self.evict_for(None, 0) {
            eprintln!("[qwen4-radix] host-only eviction failed: {error}");
        }
    }

    /// Publish one staged checkpoint: radix + context → pool and index;
    /// otherwise the end of prompt stays local and optional ones return their
    /// slot.
    fn publish_one(&mut self, checkpoint: Qwen4Checkpoint) {
        let eop = checkpoint.eop;
        if self.radix.is_none() || checkpoint.context.is_none() {
            if eop {
                self.set_local(LocalEop::Owned(checkpoint));
            } else {
                self.retire_host(checkpoint);
            }
            return;
        }
        let mut defer = Vec::new();
        let mut local = None;
        if let Some(radix) = self.radix.as_mut() {
            let di = mode_index(checkpoint.mode);
            let tokens = checkpoint.tokens.clone();
            let bytes = checkpoint.gdsf_bytes();
            let cost = checkpoint.cost;
            let (id, unadopted) = radix.pool.insert_exact(
                radix.domains[di].clone(),
                tokens.clone(),
                checkpoint,
            );
            match (id.is_some(), unadopted) {
                // Pool refused: the end of prompt stays a (context-bearing) local.
                (false, Some(blob)) => {
                    if eop {
                        local = Some(LocalEop::Owned(blob));
                    } else {
                        defer.push(blob);
                    }
                }
                (false, None) => {}
                (true, Some(blob)) => {
                    radix.stats.duplicates += 1;
                    defer.push(blob);
                    if eop {
                        local = Some(LocalEop::Published(id));
                    }
                }
                (true, None) => match publish_pages(radix, di, &tokens, id) {
                    Ok(adopted) if adopted == id => {
                        radix.stats.published += 1;
                        let clock = radix.gdsf_clock;
                        if let Some(entry) = radix.pool.peek_id_mut(id) {
                            entry.priority = gdsf_priority(clock, cost, 0, bytes);
                        }
                        if eop {
                            local = Some(LocalEop::Published(id));
                        }
                    }
                    Ok(adopted) => {
                        // The index already holds this boundary under another id.
                        radix.stats.duplicates += 1;
                        if let Some(blob) = radix.pool.take_id(id) {
                            defer.push(blob);
                        }
                        if eop && radix.pool.peek_id(adopted).is_some() {
                            local = Some(LocalEop::Published(adopted));
                        }
                    }
                    Err(message) => {
                        eprintln!("[qwen4-radix] publish refused: {message}");
                        if let Some(blob) = radix.pool.take_id(id) {
                            if eop {
                                local = Some(LocalEop::Owned(blob));
                            } else {
                                defer.push(blob);
                            }
                        }
                    }
                },
            }
        }
        if let Some(local) = local {
            self.set_local(local);
        }
        self.deferred.extend(defer);
    }

    /// Take the pool entries the index retired (and earlier ones that were
    /// pinned): freed now with a GPU, else deferred. Returns how many left.
    fn reclaim_retired(
        &mut self,
        mut gpu: Option<&mut Gpu>,
        first_error: &mut Option<BundleError>,
    ) -> usize {
        let Qwen4PrefixCache {
            radix,
            slots,
            deferred,
            ..
        } = self;
        let Some(radix) = radix.as_mut() else {
            return 0;
        };
        let mut ids = std::mem::take(&mut radix.retire_pending);
        ids.extend(radix.index.take_retired_checkpoints());
        let mut taken = 0;
        for id in ids {
            match radix.pool.take_id(id) {
                Some(blob) => {
                    taken += 1;
                    radix.stats.evictions += 1;
                    match gpu.as_deref_mut() {
                        Some(gpu) => {
                            keep_first(first_error, recycle_checkpoint(slots, gpu, blob));
                        }
                        None => deferred.push(blob),
                    }
                }
                None => {
                    if radix.pool.peek_id(id).is_some() {
                        radix.retire_pending.push(id);
                    }
                }
            }
        }
        taken
    }

    /// Free the checkpoints host-only paths deferred (GPU quiescent).
    fn drain_deferred(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let mut first = None;
        for checkpoint in std::mem::take(&mut self.deferred) {
            keep_first(&mut first, recycle_checkpoint(&mut self.slots, gpu, checkpoint));
        }
        first.map_or(Ok(()), Err)
    }

    /// One eviction step. Checkpoint victims are GDSF-ranked: the lowest
    /// priority unpinned pool entry leaves index visibility first, the aging
    /// clock moves to its priority, then its payload is reclaimed. A pruning
    /// step removes a pageless/checkpointless tree leaf instead. Returns
    /// whether anything progressed.
    fn evict_one(
        &mut self,
        mut gpu: Option<&mut Gpu>,
        mode: EvictMode,
        free_surplus_slot: bool,
        first_error: &mut Option<BundleError>,
    ) -> bool {
        let mut progressed = false;
        let mut victim = None;
        if let Some(radix) = self.radix.as_mut() {
            if mode == EvictMode::PruneFirst {
                progressed = radix.index.evict_oldest_unpinned_leaf(&mut radix.pages);
            }
            if !progressed {
                let entries: Vec<(CheckpointId, f64, bool)> = radix
                    .pool
                    .exact_ids()
                    .into_iter()
                    .filter_map(|id| {
                        radix
                            .pool
                            .peek_id(id)
                            .map(|entry| (id, entry.priority, radix.pool.pin_count(id) > 0))
                    })
                    .collect();
                if let Some(id) = gdsf_victim(&entries) {
                    if let Some(entry) = radix.pool.peek_id(id) {
                        radix.gdsf_clock = entry.priority;
                    }
                    radix.index.forget_checkpoint(id);
                    victim = Some(id);
                    progressed = true;
                }
            }
        }
        if self.reclaim_retired(gpu.as_deref_mut(), first_error) > 0 {
            progressed = true;
        }
        // A victim the index did not retire is still ours to take.
        if let Some(id) = victim {
            let Qwen4PrefixCache {
                radix,
                slots,
                deferred,
                ..
            } = self;
            if let Some(radix) = radix.as_mut() {
                if radix.pool.pin_count(id) == 0 {
                    if let Some(blob) = radix.pool.take_id(id) {
                        radix.stats.evictions += 1;
                        match gpu.as_deref_mut() {
                            Some(gpu) => {
                                keep_first(first_error, recycle_checkpoint(slots, gpu, blob));
                            }
                            None => deferred.push(blob),
                        }
                    }
                }
            }
        }
        if !progressed && free_surplus_slot && self.slots_allocated > 1 {
            if let Some(gpu) = gpu.as_deref_mut() {
                if let Some(slot) = self.slots.pop() {
                    self.slots_allocated -= 1;
                    keep_first(first_error, slot.free_gpu(gpu));
                    progressed = true;
                }
            }
        }
        progressed
    }

    /// Enforce the budgets: evict until the ledger plus `need` fits the device
    /// budget (only with a GPU, which also frees deferred checkpoints first)
    /// and the entry/host caps hold, or nothing evictable remains.
    fn evict_for(&mut self, mut gpu: Option<&mut Gpu>, need: u64) -> Result<(), BundleError> {
        let mut first_error = None;
        if let Some(gpu) = gpu.as_deref_mut() {
            keep_first(&mut first_error, self.drain_deferred(gpu));
        }
        loop {
            let ledger = self.ledger_bytes();
            let Some(radix) = self.radix.as_ref() else {
                break;
            };
            let device_over =
                gpu.is_some() && ledger.saturating_add(need) > radix.limits.device_bytes;
            let count_over = radix.pool.len() > radix.limits.checkpoints;
            let over = over_budget(
                device_over,
                radix.pool.len(),
                radix.limits.checkpoints,
                self.host_bytes(),
                radix.limits.host_bytes,
            );
            if let Some(radix) = self.radix.as_mut() {
                radix.stats.ledger_bytes = ledger;
                radix.stats.peak_ledger_bytes = radix.stats.peak_ledger_bytes.max(ledger);
            }
            if !over {
                break;
            }
            let mode = evict_mode(device_over, count_over);
            if !self.evict_one(gpu.as_deref_mut(), mode, device_over, &mut first_error) {
                break;
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Unload: release the index, every retained checkpoint and every slot.
    fn free_gpu(self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let Qwen4PrefixCache {
            staged,
            local,
            deferred,
            slots,
            radix,
            ..
        } = self;
        let mut first = None;
        let mut blobs: Vec<Qwen4Checkpoint> = Vec::new();
        if let Some(mut radix) = radix {
            radix.index.release_all(&mut radix.pages);
            blobs.extend(radix.pool.drain_exact());
        }
        blobs.extend(staged);
        blobs.extend(deferred);
        if let Some(LocalEop::Owned(checkpoint)) = local {
            blobs.push(checkpoint);
        }
        for checkpoint in blobs {
            keep_first(&mut first, free_checkpoint_final(gpu, checkpoint));
        }
        for slot in slots {
            keep_first(&mut first, slot.free_gpu(gpu));
        }
        first.map_or(Ok(()), Err)
    }
}

/// Mint `floor(B/128)` sealed descriptor pages and publish the checkpoint into
/// the index; the table's own references are released again (the index holds
/// the cache references). Returns the adopted checkpoint id.
fn publish_pages(
    radix: &mut Qwen4RadixCache,
    di: usize,
    tokens: &[u32],
    id: CheckpointId,
) -> Result<CheckpointId, String> {
    let count = tokens.len() / PAGE_TOKENS;
    let mut table = BlockTable::new();
    let published = match radix.pages.alloc_pages_checked(&mut table, count) {
        Ok(handles) => {
            let mut sealed = Ok(());
            for handle in &handles {
                if let Err(error) = radix.pages.seal(handle.phys) {
                    sealed = Err(error);
                    break;
                }
            }
            match sealed {
                Ok(()) => {
                    let pages: Vec<Handle> = handles
                        .iter()
                        .enumerate()
                        .map(|(page, handle)| Handle {
                            handle: *handle,
                            token_offset: (page * PAGE_TOKENS) as u64,
                        })
                        .collect();
                    radix
                        .index
                        .publish_checkpoint(&radix.domains[di], tokens, &pages, id, &mut radix.pages)
                        .map_err(|error| format!("{error:?}"))
                }
                Err(error) => Err(error),
            }
        }
        Err(error) => Err(error),
    };
    if let Err(error) = radix.pages.release_table(&mut table) {
        eprintln!("[qwen4-radix] descriptor table release failed: {error}");
    }
    published
}

impl Qwen4Bundle {
    /// Assemble a complete Single bundle using metadata parsed from the
    /// artifact's canonical `qwen4_ple` object. `state_format` is the request
    /// state's storage ([`crate::resolve_state_format`]); `backend` the QSA
    /// context storage load admission resolved, shared by target and MTP.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        config: Qwen4Config,
        transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
        state_format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
    ) -> Result<Self, BundleError> {
        Self::assemble_with_metadata(
            config,
            transaction,
            placements,
            gpu,
            max_seq_len,
            metadata,
            state_format,
            backend,
        )
    }

    /// Assemble with validated metadata read from the artifact's exact I64
    /// arrays.  The transaction is consumed so no load-side owner can remain
    /// live after this method publishes the bundle.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble_with_metadata(
        config: Qwen4Config,
        mut transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
        state_format: Qwen4StateFormat,
        backend: Qwen4KvBackend,
    ) -> Result<Self, BundleError> {
        let weights = match Qwen4Weights::assemble(&mut transaction, &config, placements) {
            Ok(weights) => weights,
            Err(error) => {
                return Err(cleanup_transaction(
                    BundleError::Weights(error),
                    transaction.rollback(gpu),
                ));
            }
        };
        let descriptors = match ple_descriptors(&transaction, &weights.manifest, &metadata) {
            Ok(descriptors) => descriptors,
            Err(error) => {
                let weight_result = weights.free_gpu(gpu);
                let cleanup = transaction.rollback(gpu);
                return Err(cleanup_bundle_failure(error, weight_result, cleanup));
            }
        };
        // One prefill chunk prefetches `rows * PLE_HEAD_COUNT` n-gram rows in
        // a single row-store request, so staging holds the requested chunk.
        let staging_rows =
            qwen4_prefill_chunk_requested(&gpu.arch, max_seq_len) * crate::ple::PLE_HEAD_COUNT;
        let ple_rows = match RowStore::with_staging_rows(
            "qwen4-ple-reader",
            descriptors,
            metadata.valid_rows(),
            staging_rows,
            // `HIPFIRE_QWEN4_PLE_WIDE_READERS=0` keeps 16 concurrent reads
            // for every PLE request, including a whole prefill chunk.
            hipfire_config::developer_bool("HIPFIRE_QWEN4_PLE_WIDE_READERS", true),
        ) {
            Ok(rows) => rows,
            Err(error) => {
                let weight_result = weights.free_gpu(gpu);
                let cleanup = transaction.rollback(gpu);
                return Err(cleanup_bundle_failure(
                    BundleError::PleRows(error),
                    weight_result,
                    cleanup,
                ));
            }
        };
        let mut state =
            match Qwen4State::new_with_backend(gpu, &config, max_seq_len, state_format, backend) {
                Ok(state) => state,
                Err(error) => {
                    // `unload` consumes the reader and joins its worker even on a
                    // quiesce error, so source descriptors cannot outlive failure.
                    let _ = ple_rows.unload();
                    let weight_result = weights.free_gpu(gpu);
                    let cleanup = transaction.rollback(gpu);
                    return Err(cleanup_bundle_failure(
                        BundleError::State(error),
                        weight_result,
                        cleanup,
                    ));
                }
            };
        state.bind_transaction_generation(transaction.inventory_len() as u64);
        Ok(Self {
            config,
            weights,
            state,
            ple_rows,
            ple_metadata: metadata,
            weight_store: AttachedWeightStore::new(transaction),
            execution: None,
            mtp: None,
            spec_logits: None,
            spec_top1: None,
            spec_hidden: None,
            spec_host_top1: Vec::new(),
            prefix: None,
        })
    }

    pub fn manifest(&self) -> &Qwen4Manifest {
        &self.weights.manifest
    }

    pub fn external_descriptor(
        &self,
        placement: &Qwen4Placement,
    ) -> Option<&SourceRangeDescriptor> {
        self.weight_store
            .external_descriptor(&placement.name, placement.layer, placement.device)
    }

    /// Return all numerically ordered PLE shard descriptors from the
    /// attached canonical transaction census.
    ///
    /// This strict accessor is useful to admission and diagnostics callers;
    /// the bundle's own reader already owns the same sealed descriptors.
    pub fn ple_descriptors(&self) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
        ple_descriptors(
            &self.weight_store.transaction,
            self.manifest(),
            &self.ple_metadata,
        )
    }

    pub fn ple_rows(&self) -> &RowStore {
        &self.ple_rows
    }

    pub fn ple_rows_mut(&mut self) -> &mut RowStore {
        &mut self.ple_rows
    }

    pub fn attached_origin(&self) -> Option<hipfire_runtime::weight_store::WeightOrigin> {
        self.weight_store.origin()
    }

    pub fn attached_inventory_len(&self) -> usize {
        self.weight_store.inventory_len()
    }

    pub fn attached_external_rows_len(&self) -> usize {
        self.weight_store.external_rows_len()
    }

    /// Attach reusable ordinary-HIP execution resources after the manifest
    /// transaction and model state have assembled successfully.
    pub fn attach_forward(&mut self, gpu: &mut Gpu, max_chunk: usize) -> Result<(), BundleError> {
        if self.execution.is_some() {
            return Err(BundleError::Forward(
                "Qwen4 forward resources are already attached".to_string(),
            ));
        }
        if max_chunk == 0 {
            return Err(BundleError::Forward(
                "Qwen4 forward chunk capacity is zero".to_string(),
            ));
        }
        // A chunk's PLE prefetch is one row-store request: the staging sized
        // at assembly bounds it.
        let ple_rows_cap = self.ple_rows.max_rows_per_prefetch() / crate::ple::PLE_HEAD_COUNT;
        let requested = qwen4_prefill_chunk_requested(&gpu.arch, max_chunk).min(ple_rows_cap);
        if requested == 0 {
            return Err(BundleError::Forward(
                "Qwen4 PLE row store cannot stage one chunk row".to_string(),
            ));
        }
        // The gathered QSA prefill attention (HIPFIRE_QWEN4_QSA_WMMA_GATHER)
        // converts the cache rows it reads into its own workspace. Reserve
        // its address for the whole context now, before any capture or
        // record, so a later longer prefill never moves it. With VMM QSA
        // state only the reservation is made and each forward maps the
        // prefix it reads (`Qwen4GpuForward`); legacy state commits all of
        // it here, as before. A no-op when the route is off for this arch
        // and state format.
        if let Some(qsa) = self.state.qsa.first() {
            let heads = self.config.num_key_value_heads;
            let reserved = match self.state.qsa_backend() {
                Qwen4KvBackend::Legacy => {
                    rdna_compute::tensor_ops::reserve_qsa_gathered_wmma_scratch(
                        gpu,
                        qsa.format,
                        heads,
                        qsa.full_capacity,
                    )
                }
                Qwen4KvBackend::Vmm => {
                    rdna_compute::tensor_ops::reserve_qsa_gathered_wmma_workspace(
                        gpu,
                        qsa.format,
                        heads,
                        qsa.full_capacity,
                    )
                }
            }
            .map_err(BundleError::Hip)?;
            if reserved > 0 {
                eprintln!(
                    "  qwen4 QSA gather workspace: {} MiB reserved for {} context tokens, {} MiB committed",
                    reserved >> 20,
                    qsa.full_capacity,
                    gpu.qsa_gather_scratch_bytes() >> 20
                );
            }
        }
        self.weights
            .requant_from_env(gpu)
            .map_err(BundleError::Forward)?;
        // The largest rung whose chunk-sized resources fit the free device
        // memory beside the kernels' lazily sized workspaces; the smallest
        // rung is attempted regardless and fails at allocation if it must.
        let (free, _) = gpu.hip.get_vram_info().map_err(BundleError::Hip)?;
        let max_chunk = qwen4_prefill_chunk_rungs(requested)
            .find(|&rows| {
                qwen4_forward_device_bytes(&self.config, rows)
                    .and_then(|bytes| bytes.checked_add(QWEN4_FORWARD_HEADROOM_BYTES))
                    .is_some_and(|bytes| bytes <= free as u64)
            })
            .unwrap_or_else(|| requested.min(1536));
        eprintln!("  qwen4 prefill chunk: {max_chunk} rows (requested {requested})");
        let forward = Qwen4GpuForward::new(gpu, self, max_chunk)
            .map_err(|error| BundleError::Forward(error.to_string()))?;
        let spec_rows = qwen4_spec_logit_rows(max_chunk);
        let logits_len = spec_rows
            .checked_mul(self.config.vocab_size)
            .ok_or_else(|| BundleError::Forward("spec logit scratch overflow".to_string()))?;
        let spec_logits = match gpu.zeros(&[logits_len], rdna_compute::DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = forward.free_gpu(gpu);
                return Err(BundleError::Hip(error));
            }
        };
        let top1_len = match spec_rows.checked_mul(std::mem::size_of::<i32>()) {
            Some(len) => len,
            None => {
                let _ = gpu.free_tensor(spec_logits);
                let _ = forward.free_gpu(gpu);
                return Err(BundleError::Forward(
                    "spec argmax scratch overflow".to_string(),
                ));
            }
        };
        let spec_top1 = match gpu.zeros(&[top1_len], rdna_compute::DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(spec_logits);
                let _ = forward.free_gpu(gpu);
                return Err(BundleError::Hip(error));
            }
        };
        self.execution = Some(forward);
        self.spec_logits = Some(spec_logits);
        self.spec_top1 = Some(spec_top1);
        self.spec_host_top1 = vec![0; top1_len];
        Ok(())
    }
    /// Attach the reusable native MTP head and its bounded GPU state.
    pub fn attach_mtp(&mut self, gpu: &mut Gpu, max_seq: usize) -> Result<(), BundleError> {
        if self.mtp.is_some() {
            return Err(BundleError::Forward(
                "Qwen4 MTP resources are already attached".to_string(),
            ));
        }
        let mtp = Qwen4MtpGpu::new_with_backend(
            gpu,
            &self.weights,
            &self.config,
            max_seq,
            self.state.qsa_backend(),
        )
        .map_err(|error| BundleError::Forward(error.to_string()))?;
        self.mtp = Some(mtp);
        Ok(())
    }

    pub(crate) fn ensure_spec_hidden(
        &mut self,
        gpu: &mut Gpu,
        rows: usize,
    ) -> Result<(), BundleError> {
        if rows == 0 {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden capacity is zero".to_string(),
            ));
        }
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
        let elements = rows
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden capacity overflow".to_string()))?;
        if let Some(hidden) = self.spec_hidden.as_ref() {
            if hidden.dtype != rdna_compute::DType::F32 || hidden.numel() < elements {
                return Err(BundleError::Forward(
                    "Qwen4 spec hidden capacity is too small".to_string(),
                ));
            }
            return Ok(());
        }
        self.spec_hidden = Some(
            gpu.zeros(&[elements], rdna_compute::DType::F32)
                .map_err(BundleError::Hip)?,
        );
        Ok(())
    }

    /// Install (or clear) the QSA parity observer on the attached forward.
    #[cfg(feature = "reference-parity")]
    pub fn set_qsa_tap(
        &mut self,
        tap: Option<crate::gpu_forward::Qwen4QsaTap>,
    ) -> Result<(), BundleError> {
        let forward = self.execution.as_mut().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        forward.qsa_tap = tap;
        Ok(())
    }

    /// Install (or clear) the QSA projection hook of this thread's forward:
    /// `(gpu, qsa_slot, op)` right after each QSA step's projections, i.e. on
    /// the raw projected index row / query + gate / K / V rows before the
    /// prologue (the step's cache and pool are still pre-step). Same contract
    /// as [`Self::set_qsa_tap`]: a forward with a hook never records or replays a
    /// retained body, and with none installed nothing runs or allocates.
    #[cfg(feature = "reference-parity")]
    pub fn set_qsa_projection_hook(
        &mut self,
        hook: Option<hipfire_dispatch::pipeline::QsaProjectionHook>,
    ) -> Result<(), BundleError> {
        if self.execution.is_none() {
            return Err(BundleError::Forward(
                "Qwen4 forward resources are not attached".to_string(),
            ));
        }
        hipfire_dispatch::pipeline::set_qsa_projection_hook(hook);
        Ok(())
    }

    /// Rows the attached forward can process in one chunked call.  The MTP
    /// prefill uses this to batch a whole prompt chunk through the shared
    /// forward instead of one single-row forward per prompt token.
    pub(crate) fn spec_chunk_rows(&self) -> Option<usize> {
        self.execution
            .as_ref()
            .map(|forward| forward.scratch.max_chunk)
    }

    pub(crate) fn spec_forward_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<Vec<u32>, BundleError> {
        self.spec_forward_rows_with_output(gpu, tokens, capture_hidden, Qwen4OutputRows::All)
    }

    /// Final-row argmax of one forward over `tokens`: prompt fills and
    /// advances, which read no other row's logits.
    pub(crate) fn spec_prefill_rows(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
    ) -> Result<u32, BundleError> {
        self.spec_forward_rows_with_output(gpu, tokens, capture_hidden, Qwen4OutputRows::Final)?
            .into_iter()
            .next()
            .ok_or_else(|| BundleError::Forward("Qwen4 prefill produced no argmax".into()))
    }

    fn spec_forward_rows_with_output(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        capture_hidden: bool,
        output_rows: Qwen4OutputRows,
    ) -> Result<Vec<u32>, BundleError> {
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "Qwen4 spec forward cannot process an empty block".to_string(),
            ));
        }
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .scratch
            .max_chunk;
        if tokens.len() > max_chunk {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec block length {} exceeds capacity {max_chunk}",
                tokens.len()
            )));
        }
        let vocab = self.config.vocab_size;
        let output_count = output_rows.count(tokens.len());
        let spec_rows = qwen4_spec_logit_rows(max_chunk);
        if output_count > spec_rows {
            return Err(BundleError::Forward(format!(
                "Qwen4 spec block of {output_count} output rows exceeds the {spec_rows}-row verify capacity"
            )));
        }
        let logits_len = output_count
            .checked_mul(vocab)
            .ok_or_else(|| BundleError::Forward("Qwen4 spec logits overflow".to_string()))?;
        let logits = self
            .spec_logits
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec logits are not attached".to_string()))?
            .sub_offset(0, logits_len);
        let top1_len = output_count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| BundleError::Forward("Qwen4 spec argmax overflow".to_string()))?;
        let top1 = self
            .spec_top1
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec argmax is not attached".to_string()))?
            .sub_offset(0, top1_len);
        let hidden = if capture_hidden {
            let width = self
                .config
                .hc_count
                .checked_mul(self.config.hidden_size)
                .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
            let hidden_len = tokens
                .len()
                .checked_mul(width)
                .ok_or_else(|| BundleError::Forward("spec hidden row overflow".to_string()))?;
            Some(
                self.spec_hidden
                    .as_ref()
                    .ok_or_else(|| {
                        BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
                    })?
                    .sub_offset(0, hidden_len),
            )
        } else {
            None
        };
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk(
                self,
                gpu,
                tokens,
                &logits,
                Some(&top1),
                hidden.as_ref(),
                output_rows,
            )
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result?;
        let bytes_len = output_count * std::mem::size_of::<i32>();
        if self.spec_host_top1.len() < bytes_len {
            return Err(BundleError::Forward(
                "Qwen4 spec host argmax capacity is too small".to_string(),
            ));
        }
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..bytes_len], &top1.buf)
            .map_err(BundleError::Hip)?;
        let mut picks = Vec::with_capacity(output_count);
        for bytes in self.spec_host_top1[..bytes_len].chunks_exact(4) {
            picks.push(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        }
        Ok(picks)
    }

    pub(crate) fn copy_spec_hidden_row_to(
        &self,
        gpu: &mut Gpu,
        row: usize,
        destination: &GpuTensor,
    ) -> Result<(), BundleError> {
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
        if destination.dtype != rdna_compute::DType::F32 || destination.numel() != width {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden destination shape mismatch".to_string(),
            ));
        }
        let source = self.spec_hidden.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
        })?;
        let offset = row
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden row offset overflow".to_string()))?;
        if offset
            .checked_add(width)
            .is_none_or(|end| end > source.numel())
        {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden row is outside capture".to_string(),
            ));
        }
        let source = source.sub_offset(offset, width);
        gpu.copy_d2d(&source, destination, destination.byte_size())
            .map_err(BundleError::Hip)
    }

    /// Keep the first `keep` rows of the armed `tokens.len()`-row verify the
    /// active snapshot ticket brackets, without re-running them; the ticket
    /// stays active for the caller's commit or restore.
    pub(crate) fn rollback_verify_rows_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
        keep: usize,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        let ple_normed = &self
            .execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .scratch
            .ple_normed;
        self.state
            .rollback_rows_retain(
                gpu,
                snapshot,
                keep,
                tokens.len(),
                tokens,
                ple_normed,
                self.config.ple_conv_history_rows(),
                self.config.linear_conv_kernel_dim - 1,
                self.config.indexer_compress_ratio,
                self.config.indexer_budget,
            )
            .map_err(BundleError::State)
    }

    /// Start reading the PLE rows `tokens` (the next tokens after the
    /// committed history, in order) will need, so a forward over them later
    /// finds them cached. Best effort: a failure only loses the head start.
    pub(crate) fn warm_ple_rows(&self, tokens: &[u32]) {
        let ids = self.state.ple_history.row_ids(&self.ple_metadata, tokens);
        let _ = self.ple_rows.warm(ids);
    }

    pub(crate) fn spec_capture_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
    ) -> Result<u32, BundleError> {
        self.spec_forward_rows(gpu, std::slice::from_ref(&token), true)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 spec capture returned no argmax".to_string())
            })
    }

    /// One ordinary target-only forward of `token`: a single final row, no
    /// wide-hidden capture, so the replay graph route stays eligible. Returns
    /// the greedy argmax (read into the first 4 bytes of `spec_host_top1`, no
    /// allocation); the row's logits stay in row 0 of `spec_logits` for a
    /// sampled follow-up. Clears any stale PLE lookahead first.
    pub(crate) fn spec_ar_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
    ) -> Result<u32, BundleError> {
        let vocab = self.config.vocab_size;
        let logits = self
            .spec_logits
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec logits are not attached".to_string()))?
            .sub_offset(0, vocab);
        let top1_bytes = std::mem::size_of::<i32>();
        let top1 = self
            .spec_top1
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 spec argmax is not attached".to_string()))?
            .sub_offset(0, top1_bytes);
        if self.spec_host_top1.len() < top1_bytes {
            return Err(BundleError::Forward(
                "Qwen4 spec host argmax capacity is too small".to_string(),
            ));
        }
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        forward.set_ple_lookahead(&[]);
        let result = forward
            .forward_token(self, gpu, token, &logits, Some(&top1))
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result?;
        gpu.hip
            .memcpy_dtoh(&mut self.spec_host_top1[..top1_bytes], &top1.buf)
            .map_err(BundleError::Hip)?;
        let bytes = &self.spec_host_top1[..top1_bytes];
        Ok(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Copy the wide HC stream row the last [`Self::spec_ar_token`] (or any
    /// ordinary one-token forward) left in the forward scratch into
    /// `destination` (F32, exactly `hc_count * hidden_size` elements). Valid
    /// only until the next forward; for calibration, not the hot path.
    pub(crate) fn copy_ar_hidden_to(
        &self,
        gpu: &mut Gpu,
        destination: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.execution
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?
            .copy_last_wide_hidden_to(gpu, &self.config, destination)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_forward_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
    ) -> Result<u32, BundleError> {
        let mtp = mapped_mtp(self.mtp.as_mut(), gpu, position)?;
        mtp.forward_token(
            gpu,
            &self.weights,
            &self.config,
            token,
            backbone_hidden,
            position,
            fresh_qsa_selection,
            MtpStep::Predict,
        )
        .map_err(|error| BundleError::Forward(error.to_string()))?
        .ok_or_else(|| {
            BundleError::Forward("MTP prediction requested but no token produced".into())
        })
    }

    /// Exact logit margin of the last MTP draft over its runner-up.
    pub(crate) fn mtp_draft_margin(&self) -> f32 {
        self.mtp
            .as_ref()
            .map_or(f32::INFINITY, |mtp| mtp.draft.margin())
    }

    /// The draft head's request-local policy (full-vocabulary hold, last
    /// margin). Not part of the MTP state snapshot: a caller that drafts and
    /// then restores the head must save and restore it separately.
    pub(crate) fn mtp_draft_request_state(
        &self,
    ) -> Result<hipfire_dispatch::pipeline::DraftHeadRequestState, BundleError> {
        self.mtp
            .as_ref()
            .map(|mtp| mtp.draft.request_state())
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })
    }

    pub(crate) fn set_mtp_draft_request_state(
        &mut self,
        state: hipfire_dispatch::pipeline::DraftHeadRequestState,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .map(|mtp| mtp.draft.set_request_state(state))
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })
    }

    /// Host copy of row `row` of the last speculative forward's logits into
    /// the reused buffer `host`.
    pub(crate) fn spec_row_logits(
        &self,
        gpu: &Gpu,
        row: usize,
        host: &mut Vec<f32>,
    ) -> Result<(), BundleError> {
        let vocab = self.config.vocab_size;
        let logits = self.spec_logits.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec logits are not attached".to_string())
        })?;
        let offset = row
            .checked_mul(vocab)
            .filter(|offset| offset + vocab <= logits.numel())
            .ok_or_else(|| {
                BundleError::Forward(format!("Qwen4 spec logit row {row} is outside capacity"))
            })?;
        gpu.download_f32_into(&logits.sub_offset(offset, vocab), host)
            .map_err(BundleError::Hip)
    }

    /// Row `row` of the last speculative forward's logits as `spec`'s
    /// truncated distribution, after `policy` (repeat/presence/frequency
    /// penalties and blocked tokens) is applied over `history` — the row's AR
    /// history — to the whole downloaded row, before the pool gather. `host`
    /// and `scratch` are reused buffers.
    pub(crate) fn spec_row_dist(
        &self,
        gpu: &Gpu,
        row: usize,
        spec: SampleSpec,
        history: &[u32],
        policy: &SamplerConfig,
        host: &mut Vec<f32>,
        scratch: &mut Vec<(u32, f32)>,
        out: &mut SparseDist,
    ) -> Result<(), BundleError> {
        self.spec_row_logits(gpu, row, host)?;
        apply_logit_policy_cpu(host, history, policy);
        out.build_from_logits(host, spec, scratch)
            .map_err(BundleError::Forward)
    }

    /// The last MTP prediction's draft distribution under `spec`: its 8
    /// re-scored candidates' exact logits, or the whole draft logit row when
    /// the draft head does not re-score. `policy` is applied over `history`
    /// (the AR history of the row being drafted) to those exact logits, by
    /// token id, before truncation.
    pub(crate) fn mtp_draft_dist(
        &self,
        gpu: &Gpu,
        spec: SampleSpec,
        history: &[u32],
        policy: &SamplerConfig,
        host: &mut Vec<f32>,
        scratch: &mut Vec<(u32, f32)>,
        out: &mut SparseDist,
    ) -> Result<(), BundleError> {
        let draft = &self
            .mtp
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .draft;
        let vocab = self.config.vocab_size as u32;
        match draft.rescored_candidates() {
            Some(candidates) => {
                scratch.clear();
                scratch.extend(
                    candidates
                        .iter()
                        .copied()
                        .filter(|&(token, _)| token < vocab),
                );
                // `rescored_candidates` is a fixed `[_; 8]`, so the filtered
                // count is at most 8.
                let mut ids = [0u32; 8];
                let mut values = [0f32; 8];
                let n = scratch.len();
                for (slot, &(token, logit)) in scratch.iter().enumerate() {
                    ids[slot] = token;
                    values[slot] = logit;
                }
                apply_logit_policy_candidates_cpu(&ids[..n], &mut values[..n], history, policy);
                for (entry, &logit) in scratch.iter_mut().zip(&values[..n]) {
                    entry.1 = logit;
                }
                out.build_from_candidates(scratch, spec)
            }
            None => {
                gpu.download_f32_into(draft.logits(), host)
                    .map_err(BundleError::Hip)?;
                apply_logit_policy_cpu(host, history, policy);
                out.build_from_logits(host, spec, scratch)
            }
        }
        .map_err(BundleError::Forward)
    }

    pub(crate) fn mtp_advance_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
        fresh_qsa_selection: bool,
    ) -> Result<(), BundleError> {
        mapped_mtp(self.mtp.as_mut(), gpu, position)?
            .forward_token(
                gpu,
                &self.weights,
                &self.config,
                token,
                backbone_hidden,
                position,
                fresh_qsa_selection,
                MtpStep::Advance,
            )
            .map(|_| ())
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// The last MTP step of a chain: append only its K/V and index-key cache
    /// rows (see [`MtpStep::Append`]); the next step must bring its own
    /// backbone hidden and a fresh selection.
    pub(crate) fn mtp_append_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        backbone_hidden: Option<&GpuTensor>,
        position: usize,
    ) -> Result<(), BundleError> {
        mapped_mtp(self.mtp.as_mut(), gpu, position)?
            .forward_token(
                gpu,
                &self.weights,
                &self.config,
                token,
                backbone_hidden,
                position,
                true,
                MtpStep::Append,
            )
            .map(|_| ())
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// Whether [`Self::mtp_append_rows`] can run on this GPU for this model.
    pub(crate) fn mtp_append_rows_supported(&self, gpu: &Gpu) -> bool {
        self.mtp.is_some() && Qwen4MtpGpu::append_rows_supported(gpu, &self.weights, &self.config)
    }

    /// Batched prompt fill: [`MtpStep::Append`] for `tokens` at
    /// `position..position + tokens.len()`, row `i` paired with captured spec
    /// hidden row `hidden_row0 + i`; the same (token p, hidden p) rows as
    /// calling [`Self::mtp_append_token`] per row after `copy_spec_hidden_row_to`.
    pub(crate) fn mtp_append_rows(
        &mut self,
        gpu: &mut Gpu,
        scratch: &mut MtpAppendScratch,
        tokens: &[u32],
        hidden_row0: usize,
        position: usize,
    ) -> Result<(), BundleError> {
        if tokens.is_empty() {
            return Err(BundleError::Forward(
                "Qwen4 MTP batched append has no rows".to_string(),
            ));
        }
        let width = self
            .config
            .hc_count
            .checked_mul(self.config.hidden_size)
            .ok_or_else(|| BundleError::Forward("spec hidden width overflow".to_string()))?;
        let source = self.spec_hidden.as_ref().ok_or_else(|| {
            BundleError::Forward("Qwen4 spec hidden is not allocated".to_string())
        })?;
        let offset = hidden_row0
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden row offset overflow".to_string()))?;
        let len = tokens
            .len()
            .checked_mul(width)
            .ok_or_else(|| BundleError::Forward("spec hidden row overflow".to_string()))?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > source.numel())
        {
            return Err(BundleError::Forward(
                "Qwen4 spec hidden rows are outside capture".to_string(),
            ));
        }
        let hidden = source.sub_offset(offset, len);
        let last = position
            .checked_add(tokens.len() - 1)
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP position overflows".to_string()))?;
        mapped_mtp(self.mtp.as_mut(), gpu, last)?
            .append_rows(
                gpu,
                &self.weights,
                &self.config,
                scratch,
                tokens,
                &hidden,
                position,
            )
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    /// Run one native MTP token from its committed state and copy the
    /// production logits into the caller-owned F32 destination.
    pub fn mtp_forward_token_logits(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        position: usize,
        fresh_qsa_selection: bool,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let mtp = mapped_mtp(self.mtp.as_mut(), gpu, position)?;
        mtp.forward_token_with_logits(
            gpu,
            &self.weights,
            &self.config,
            token,
            position,
            fresh_qsa_selection,
            logits,
        )
        .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_snapshot(
        &mut self,
        gpu: &mut Gpu,
    ) -> Result<MtpGpuStateSnapshot, BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .snapshot(gpu)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .restore(gpu, snapshot)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }
    pub(crate) fn mtp_restore_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .restore_retain(gpu, snapshot)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_truncate_retain(
        &mut self,
        snapshot: MtpGpuStateSnapshot,
        keep: usize,
    ) -> Result<(), BundleError> {
        let compress = self.config.indexer_compress_ratio;
        self.mtp
            .as_mut()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .truncate_retain(snapshot, keep, compress)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_validate_commit(
        &self,
        snapshot: MtpGpuStateSnapshot,
    ) -> Result<(), BundleError> {
        self.mtp
            .as_ref()
            .ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?
            .validate_commit(snapshot)
            .map_err(|error| BundleError::Forward(error.to_string()))
    }

    pub(crate) fn mtp_commit_validated(&mut self, snapshot: MtpGpuStateSnapshot) {
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.commit_validated(snapshot);
        }
    }

    pub(crate) fn mtp_position(&self) -> Result<usize, BundleError> {
        self.mtp
            .as_ref()
            .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".to_string()))
            .map(Qwen4MtpGpu::position)
    }

    /// Run one token through the attached execution owner without exposing a
    /// second bundle owner to callers.
    pub fn forward_token(
        &mut self,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_token(self, gpu, token, logits, top1)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// [`Self::forward_token`] of `token`, or (`None`) of the GPU argmax of
    /// `logits` as the previous forward left them; returns the token (see
    /// `Qwen4GpuForward::forward_token_or_argmax`).
    pub fn forward_token_or_argmax(
        &mut self,
        gpu: &mut Gpu,
        token: Option<u32>,
        logits: &GpuTensor,
    ) -> Result<u32, BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_token_or_argmax(self, gpu, token, logits)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Run a token sequence through the shared execution owner.  The forward
    /// owner tiles requests longer than its bounded scratch capacity while
    /// preserving the public all-row logits contract.
    pub fn forward_chunk(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk(self, gpu, tokens, logits, top1, None, Qwen4OutputRows::All)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Run a prompt through the shared execution owner and retain only its
    /// final logits row.  Long prompts are tiled over bounded scratch.
    ///
    /// This is the explicit autoregressive prefill contract; `forward_chunk`
    /// remains the public all-row API for callers that need every row.
    pub fn forward_chunk_final(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk(
                self,
                gpu,
                tokens,
                logits,
                top1,
                None,
                Qwen4OutputRows::Final,
            )
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Run `tokens` through the shared execution owner committing model state
    /// only: no logits, argmax or wide-hidden rows. Tiles exactly as
    /// [`Self::forward_chunk_final`], so a prefill split at a capture boundary
    /// keeps the trunk numerics of the unsplit one.
    pub(crate) fn forward_chunk_silent(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
    ) -> Result<(), BundleError> {
        let mut forward = self.execution.take().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let result = forward
            .forward_chunk_silent(self, gpu, tokens)
            .map_err(|error| BundleError::Forward(error.to_string()));
        self.execution = Some(forward);
        result
    }

    /// Name the tokens the next prefill forward is followed by, so it warms
    /// their PLE rows while its own chunk runs. Best effort; bytes unchanged.
    pub(crate) fn set_ple_lookahead(&mut self, tokens: &[u32]) {
        if let Some(forward) = self.execution.as_mut() {
            forward.set_ple_lookahead(tokens);
        }
    }

    fn invalidate_ple_epoch(&self) -> Result<(), BundleError> {
        self.ple_rows
            .reset_epoch(PLE_RESET_TIMEOUT)
            .map(|_| ())
            .map_err(BundleError::PleRows)
    }

    /// Quiesce request-local PLE work without advancing the current epoch.
    /// Snapshot and commit preserve that epoch; reset and restore call
    /// `invalidate_ple_epoch` instead so work from the discarded state cannot
    /// publish after the state transition.
    fn quiesce_ple(&self) -> Result<(), BundleError> {
        self.ple_rows
            .quiesce(PLE_RESET_TIMEOUT)
            .map(|_| ())
            .map_err(BundleError::PleRows)?;
        self.ple_rows.resume().map_err(BundleError::PleRows)
    }

    /// Reset every owner for a cold prefill. With a prefix cache the request's
    /// pins and staged checkpoints are released (GPU quiescent: freed now), the
    /// live record and local checkpoint dropped, and every bank whose sealed
    /// bytes must not be written is swapped for a fresh empty one before the
    /// owners reset. The shared radix store is never cleared.
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let mut cleanup: Option<BundleError> = None;
        if self.prefix.is_some() {
            if let Some(cache) = self.prefix.as_mut() {
                cache.release_pending();
                cache.release_held();
            }
            if let Err(error) = self.discard_staged(gpu) {
                cleanup.get_or_insert(error);
            }
            if let Err(error) = self.drain_deferred(gpu) {
                cleanup.get_or_insert(error);
            }
            if let Some(cache) = self.prefix.as_mut() {
                cache.clear();
                cache.drop_local();
            }
            self.swap_sealed_banks(gpu)?;
            // The reset below rewrites the rows of an unswapped bank, so no
            // checkpoint may be restored in place into it afterwards.
            if let Some(cache) = self.prefix.as_mut() {
                let old = cache.start_bank();
                cache.bank_gone(old);
            }
        }
        self.invalidate_ple_epoch()?;
        self.state.reset(gpu).map_err(BundleError::State)?;
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.reset(gpu)
                .map_err(|error| BundleError::Forward(error.to_string()))?;
        }
        cleanup.map_or(Ok(()), Err)
    }

    /// Allocate the baseline durable checkpoint slot (target state plus, when
    /// attached, the MTP head's part). Call after `attach_forward` and MTP
    /// attach; its bytes are charged by the load reserve
    /// (`Qwen4State::prefix_arena_bytes`).
    /// Idempotent: call once right after assembly, before `attach_forward`
    /// (so its free-VRAM chunk rung sees the checkpoint already allocated),
    /// and again after MTP attach to add the head's part.
    pub fn attach_prefix_cache(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        if self.prefix.is_none() {
            let target = self
                .state
                .new_checkpoint(gpu)
                .map_err(BundleError::State)?;
            let mut cache = Qwen4PrefixCache::new(self.state.max_seq_len);
            cache.slots.push(Qwen4CheckpointSlot { target, head: None });
            cache.slots_allocated = 1;
            self.prefix = Some(cache);
        }
        if let (Some(mtp), Some(cache)) = (self.mtp.as_ref(), self.prefix.as_mut()) {
            if let Some(slot) = cache.slots.iter_mut().find(|slot| slot.head.is_none()) {
                slot.head = Some(
                    mtp.new_checkpoint(gpu)
                        .map_err(|error| BundleError::Forward(error.to_string()))?,
                );
            }
        }
        Ok(())
    }

    pub fn prefix_cache_attached(&self) -> bool {
        self.prefix.is_some()
    }

    /// Device bytes the QSA context arenas commit now, `(target, MTP head)`:
    /// mapped pages of VMM owners, full allocations of legacy ones.
    pub fn qsa_context_committed_bytes(&self, gpu: &Gpu) -> Result<(usize, usize), BundleError> {
        let target = self
            .state
            .mapped_context_bytes(gpu)
            .map_err(BundleError::State)?;
        let mtp = match self.mtp.as_ref() {
            Some(mtp) => mtp
                .mapped_context_bytes(gpu)
                .map_err(|error| BundleError::Forward(error.to_string()))?,
            None => 0,
        };
        Ok((target, mtp))
    }

    /// Device positions and admitted chunk the prefix decisions read. The
    /// durable checkpoint is the end-of-prompt one: the staged one while the
    /// request is unpublished, else the local one, valid when restorable.
    fn prefix_marks(&self) -> PrefixMarks {
        let eop = self.prefix.as_ref().and_then(|cache| cache.eop_facts());
        PrefixMarks {
            target_position: self.state.position,
            mtp_position: self.mtp.as_ref().map(Qwen4MtpGpu::position),
            target_prefix: eop.map(|(_, position, _)| position),
            mtp_prefix: eop.and_then(|(_, position, mode)| {
                (mode == Qwen4PrefixMode::NativeMtp).then_some(position)
            }),
            chunk: self.spec_chunk_rows(),
            row_capture_armed: self.state.row_capture_armed,
        }
    }

    /// Published token lineage for `mode`, the `conversation_tokens` input of
    /// the shared prompt-cache planner: the full committed record while the
    /// live state still holds it, else the end-of-prompt record while the
    /// durable checkpoint is valid, else `None` (no cache, unpublished, or
    /// mode, admitted chunk or device validity mismatch).
    pub fn prefix_cache_tokens(&self, mode: Qwen4PrefixMode) -> Option<&[u32]> {
        self.prefix
            .as_ref()?
            .prefix_lineage(&self.prefix_marks(), mode)
    }

    /// End-of-prompt position `P` while the durable checkpoint is published
    /// and valid for `mode`; the planner's checkpoint list.
    pub fn prefix_checkpoint_position(&self, mode: Qwen4PrefixMode) -> Option<usize> {
        self.prefix
            .as_ref()?
            .checkpoint_end(&self.prefix_marks(), mode)
    }

    /// Bind a planner-selected `start_pos` for `prompt` (the full canonical
    /// token ids) under `mode`. A pending [`Self::select_prefix_plan`]
    /// selection matching `(start_pos, mode)` returns its plan. Otherwise:
    /// `0` is a cold start; the valid live end `L` (with `prompt[..L]` equal
    /// to the committed record) is a live continuation, preferred when it also
    /// ends at `P`; the valid checkpoint end `P` (with `prompt[..P]` equal to
    /// its record) is a checkpoint restore. A hit must leave a non-empty
    /// suffix; any other start is an error. No GPU or state mutation.
    pub fn bind_prefix_plan(
        &self,
        prompt: &[u32],
        start_pos: usize,
        mode: Qwen4PrefixMode,
    ) -> Result<Qwen4PrefixPlan, BundleError> {
        if let Some(selection) = self
            .prefix
            .as_ref()
            .and_then(|cache| cache.selection.as_ref())
        {
            if selection.start == start_pos && selection.mode == mode {
                return Ok(selection.plan());
            }
        }
        self.bind_prefix_plan_local(prompt, start_pos, mode)
    }

    /// The local (live record / end-of-prompt checkpoint) bind of
    /// [`Self::bind_prefix_plan`], ignoring any pending selection.
    fn bind_prefix_plan_local(
        &self,
        prompt: &[u32],
        start_pos: usize,
        mode: Qwen4PrefixMode,
    ) -> Result<Qwen4PrefixPlan, BundleError> {
        let bound = match self.prefix.as_ref() {
            Some(cache) => cache.bind(&self.prefix_marks(), prompt, start_pos, mode),
            None if start_pos == 0 => Ok(Qwen4PrefixPlan::default()),
            None => Err(format!(
                "Qwen4 prefix cache is not attached (start {start_pos})"
            )),
        };
        bound.map_err(BundleError::Forward)
    }

    /// Whether the radix store is attached ([`Self::attach_radix_cache`]).
    pub fn radix_cache_attached(&self) -> bool {
        self.prefix
            .as_ref()
            .is_some_and(|cache| cache.radix.is_some())
    }

    /// A copy of the radix counters, `None` without a radix store.
    pub fn radix_stats(&self) -> Option<Qwen4RadixStats> {
        let radix = self.prefix.as_ref()?.radix.as_ref()?;
        let mut stats = radix.stats.clone();
        stats.retired_va_bytes = hip_bridge::retired_va_bytes() as u64;
        Some(stats)
    }

    /// Register the positions after every ChatML turn terminator of the
    /// canonical prompt (`<|im_end|>`), the anchor candidates of
    /// [`Self::select_prefix_plan`]. Stored sorted and deduplicated, each
    /// below the admitted context length.
    pub fn set_prefix_turn_boundaries(&mut self, boundaries: &[usize]) {
        let max_seq = self.state.max_seq_len;
        if let Some(cache) = self.prefix.as_mut() {
            cache.turn_boundaries = normalized_turn_boundaries(boundaries, max_seq);
        }
    }

    /// Host-only: release the pins of a selection no prefill consumed and drop
    /// its anchor. A prefill that already began owns its pin until
    /// commit/reset, so this is a no-op then.
    pub fn abandon_prefix_selection(&mut self) {
        if let Some(cache) = self.prefix.as_mut() {
            cache.release_pending();
            cache.anchor = None;
        }
    }

    /// Choose the authoritative prefill start for `prompt`. `local_start` is
    /// the shared planner's start (`0`, or a live/checkpoint end
    /// [`Self::bind_prefix_plan`] accepts). Without a radix store this equals
    /// the local bind. With one, the deepest valid retained checkpoint
    /// `B < prompt.len()` of this mode's domain competes with the local
    /// candidate (live wins when `L >= B`, otherwise the greater of local
    /// prompt `P` and `B`, `P` on ties); the winner is pinned, the pending
    /// selection stored (one at a time) and the conditional turn anchor
    /// decided. Host-only: no GPU or owner mutation.
    pub fn select_prefix_plan(
        &mut self,
        prompt: &[u32],
        local_start: usize,
        mode: Qwen4PrefixMode,
    ) -> Result<Qwen4PrefixPlan, BundleError> {
        if let Some(cache) = self.prefix.as_mut() {
            cache.release_pending();
            cache.anchor = None;
        }
        let local = self.bind_prefix_plan_local(prompt, local_start, mode)?;
        let chunk = self.spec_chunk_rows();
        let Some(cache) = self.prefix.as_mut() else {
            return Ok(local);
        };
        let Some(chunk) = chunk.filter(|_| cache.radix.is_some()) else {
            return Ok(local);
        };
        cache.select_radix(prompt, local, mode, chunk)
    }

    /// The next absolute position the running prefill must end its chunk at
    /// and stage a checkpoint at, if any: the pre-admitted anchor inside
    /// `(position, natural_end]`, else `natural_end` when it is the end of the
    /// armed prompt or a due periodic boundary. `None` without a prefix cache.
    /// `Some(b)` satisfies `position < b <= natural_end`.
    pub fn next_prefix_capture(&self, position: usize, natural_end: usize) -> Option<usize> {
        self.prefix
            .as_ref()?
            .capture_schedule()
            .next(position, natural_end)
    }

    /// Attach the shared radix store: granular context growth for target and
    /// head, the per-mode cache domains derived from `domain`, and the
    /// host-side index, descriptor pool and checkpoint pool. Requires
    /// [`Self::attach_prefix_cache`] (and any MTP head and forward) done and
    /// no context bytes mapped yet.
    pub fn attach_radix_cache(
        &mut self,
        gpu: &mut Gpu,
        domain: CacheDomain,
        limits: Qwen4RadixLimits,
    ) -> Result<(), BundleError> {
        let chunk = self.spec_chunk_rows().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let slot_bytes = {
            let cache = self.prefix.as_ref().ok_or_else(|| {
                BundleError::Forward("Qwen4 prefix cache is not attached".to_string())
            })?;
            if cache.radix.is_some() {
                return Err(BundleError::Forward(
                    "Qwen4 radix cache is already attached".to_string(),
                ));
            }
            cache
                .slots
                .first()
                .map(Qwen4CheckpointSlot::device_bytes)
                .ok_or_else(|| {
                    BundleError::Forward("Qwen4 baseline checkpoint slot is missing".to_string())
                })?
        };
        if self.state.qsa_backend() != Qwen4KvBackend::Vmm {
            return Err(BundleError::Forward(
                "Qwen4 radix cache needs the VMM QSA context backend".to_string(),
            ));
        }
        if self
            .state
            .mapped_context_bytes(gpu)
            .map_err(BundleError::State)?
            != 0
        {
            return Err(BundleError::Forward(
                "Qwen4 radix cache must attach before any target context is mapped".to_string(),
            ));
        }
        if let Some(mtp) = self.mtp.as_ref() {
            if mtp.state.backend() != Qwen4KvBackend::Vmm
                || mtp
                    .mapped_context_bytes(gpu)
                    .map_err(|error| BundleError::Forward(error.to_string()))?
                    != 0
            {
                return Err(BundleError::Forward(
                    "Qwen4 radix cache needs an unmapped VMM MTP head context".to_string(),
                ));
            }
        }
        let target_strides = self.target_arena_strides()?;
        let head_strides = self.head_arena_strides()?;
        let gdn_tag = self
            .state
            .gdn
            .first()
            .map(|gdn| {
                format!(
                    "{:?}:{}:{:?}:{}",
                    gdn.recurrent.dtype,
                    gdn.recurrent.numel(),
                    gdn.conv.dtype,
                    gdn.conv.numel()
                )
            })
            .unwrap_or_default();
        let qsa_tag = self
            .state
            .qsa
            .first()
            .map(|qsa| format!("{:?}", qsa.format))
            .unwrap_or_default();
        let ar_strides = target_strides.clone();
        let mtp_strides: Vec<u64> = target_strides.iter().chain(&head_strides).copied().collect();
        let domains = [
            build_radix_domain(&domain, Qwen4PrefixMode::Ar, chunk, &gdn_tag, &qsa_tag, &ar_strides),
            build_radix_domain(
                &domain,
                Qwen4PrefixMode::NativeMtp,
                chunk,
                &gdn_tag,
                &qsa_tag,
                &mtp_strides,
            ),
        ];
        let compress = self.config.indexer_compress_ratio.max(1);
        let (k_sum, v_sum) = logical_page_strides(&mtp_strides, compress);
        let pages = PagePool::new_external_with_strides(RADIX_PAGE_POOL_PAGES, k_sum, v_sum)
            .map_err(BundleError::Forward)?;
        self.state.set_granular_context(true);
        if let Some(mtp) = self.mtp.as_mut() {
            mtp.set_granular_context(true);
        }
        if let Some(cache) = self.prefix.as_mut() {
            cache.radix = Some(Qwen4RadixCache {
                domains,
                limits,
                index: PrefixIndex::new(RADIX_INDEX_NODES),
                pages,
                pool: QwenCheckpointPool::new(u64::MAX),
                stats: Qwen4RadixStats::default(),
                slot_bytes,
                live_granules: HashSet::new(),
                retire_pending: Vec::new(),
                gdsf_clock: 0.0,
            });
        }
        Ok(())
    }

    /// Row strides of the target's context arenas, arena order (layer by layer:
    /// K, V, raw index, pooled index).
    fn target_arena_strides(&self) -> Result<Vec<u64>, BundleError> {
        let mut strides = Vec::with_capacity(self.state.qsa.len() * 4);
        for layer in &self.state.qsa {
            for (tensor, rows) in [
                (&layer.full_keys, layer.full_capacity),
                (&layer.full_values, layer.full_capacity),
                (&layer.raw_index_keys, layer.raw_capacity),
                (&layer.pooled_keys, layer.pooled_capacity),
            ] {
                strides.push(arena_row_stride(tensor, rows)?);
            }
        }
        Ok(strides)
    }

    /// Row strides of the native head's four context arenas (empty without a head).
    fn head_arena_strides(&self) -> Result<Vec<u64>, BundleError> {
        let Some(mtp) = self.mtp.as_ref() else {
            return Ok(Vec::new());
        };
        let max_seq = self.state.max_seq_len;
        let compress = self.config.indexer_compress_ratio.max(1);
        let tensors = mtp.context_tensors();
        if tensors.len() != 4 {
            return Err(BundleError::Forward(format!(
                "Qwen4 MTP head exposes {} context arenas, expected 4",
                tensors.len()
            )));
        }
        let rows = [max_seq, max_seq, max_seq, max_seq.div_ceil(compress)];
        tensors
            .into_iter()
            .zip(rows)
            .map(|(tensor, rows)| arena_row_stride(tensor, rows))
            .collect()
    }

    /// Discard live decode state the host will never extend, keeping the
    /// end-of-prompt checkpoint: restore it into every owner (in place, or into
    /// a fresh bank when the bank is sealed), as a checkpoint hit's prefill
    /// would. Afterwards no live record remains and earlier receipts are
    /// stale; staged checkpoints stay publishable. Without a valid checkpoint,
    /// or if the restore fails, reset instead. Every later prefill begins with
    /// [`Self::begin_prefix`], so the discarded state is never built on.
    pub fn rewind_to_prefix(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let marks = self.prefix_marks();
        let target = match self.prefix.as_ref() {
            Some(cache) if marks.checkpoint_at(cache.prompt_len, cache.mode) => cache
                .eop_facts()
                .map(|(source, _, mode)| (source, mode)),
            _ => None,
        };
        let Some((source, mode)) = target else {
            return self.reset(gpu);
        };
        if let Err(failure) = self.restore_checkpoint_owners(gpu, source, mode) {
            return match self.reset(gpu) {
                Ok(()) => Ok(()),
                Err(reset) => Err(BundleError::Forward(format!(
                    "{}; reset after the failed prefix rewind also failed: {reset}",
                    failure.error
                ))),
            };
        }
        if let Some(cache) = self.prefix.as_mut() {
            cache.live_len = None;
            cache.tokens.truncate(cache.prompt_len);
            cache.generation += 1;
        }
        Ok(())
    }

    /// Start a prefill of `prompt` under `plan`. The plan is validated against
    /// the pending selection (or the local bind) and the pinned checkpoint
    /// before any device write. A cold plan resets every owner (a sealed bank
    /// is swapped for a fresh one; the radix store survives). A live plan
    /// continues the committed state in place: only the previous request's PLE
    /// work is invalidated. A checkpoint plan restores `P` into target state
    /// and, for native MTP, the head and its draft policy: in place when the
    /// bank is unchanged and unsealed where the restore writes, otherwise into
    /// a fresh bank (forked from the checkpoint's retained granules, with typed
    /// PM4 relocation). A failed restore before any live write returns the
    /// error with the old state intact; after one it resets. The cache is
    /// unpublished until [`Self::commit_prefix`], and the prefill checkpoints
    /// at `prompt.len()`.
    pub fn begin_prefix(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        plan: Qwen4PrefixPlan,
        mode: Qwen4PrefixMode,
    ) -> Result<(), BundleError> {
        // A lookahead left by an aborted request names another prompt's rows.
        self.set_ple_lookahead(&[]);
        let action = self.plan_begin(prompt, plan, mode)?;
        let result = self.apply_begin(gpu, prompt, plan, mode, action);
        if result.is_err() {
            if let Some(cache) = self.prefix.as_mut() {
                cache.release_held();
            }
        }
        result
    }

    /// Host-only validation of `plan`; consumes the pending selection into the
    /// request's held pin.
    fn plan_begin(
        &mut self,
        prompt: &[u32],
        plan: Qwen4PrefixPlan,
        mode: Qwen4PrefixMode,
    ) -> Result<BeginAction, BundleError> {
        let marks = self.prefix_marks();
        let Some(cache) = self.prefix.as_mut() else {
            return if plan.source == Qwen4PrefixSource::Cold && plan.start_pos == 0 {
                Ok(BeginAction::Cold)
            } else {
                Err(BundleError::Forward(format!(
                    "Qwen4 prefix cache is not attached (start {})",
                    plan.start_pos
                )))
            };
        };
        let selection = cache.selection.take();
        let selected = selection.as_ref().is_some_and(|selection| {
            selection.start == plan.start_pos
                && selection.source == plan.source
                && selection.generation == plan.generation
                && selection.mode == mode
        });
        let outcome = cache.validate_plan(
            &marks,
            prompt,
            plan,
            mode,
            selection.as_ref().filter(|_| selected),
        );
        match outcome {
            Ok(action) => {
                cache.release_held();
                match selection {
                    Some(selection) if selected => cache.held = Some(selection),
                    Some(selection) => Qwen4PrefixCache::release_pins(&mut cache.radix, selection),
                    None => {}
                }
                if !selected || cache.anchor.is_some_and(|anchor| anchor <= plan.start_pos) {
                    cache.anchor = None;
                }
                Ok(action)
            }
            Err(error) => {
                if let Some(selection) = selection {
                    Qwen4PrefixCache::release_pins(&mut cache.radix, selection);
                }
                cache.anchor = None;
                Err(error)
            }
        }
    }

    fn apply_begin(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        plan: Qwen4PrefixPlan,
        mode: Qwen4PrefixMode,
        action: BeginAction,
    ) -> Result<(), BundleError> {
        // Quiescent point: free what the previous request left behind.
        self.discard_staged(gpu)?;
        self.drain_deferred(gpu)?;
        let mut report = RestoreReport::default();
        match action {
            BeginAction::Cold => {
                let anchor = self.prefix.as_ref().and_then(|cache| cache.anchor);
                self.reset(gpu)?;
                if let Some(cache) = self.prefix.as_mut() {
                    cache.anchor = anchor;
                }
            }
            BeginAction::Live => {
                if let Some(cache) = self.prefix.as_mut() {
                    cache.published = false;
                    cache.live_len = None;
                    cache.tokens.truncate(cache.prompt_len);
                }
                self.invalidate_ple_epoch()?;
            }
            BeginAction::Restore(source) => {
                report = match self.restore_checkpoint_owners(gpu, source, mode) {
                    Ok(report) => report,
                    Err(failure) => return Err(self.fail_restore(gpu, failure)),
                };
                if let Some(cache) = self.prefix.as_mut() {
                    cache.note_restore_hit(source);
                    cache.live_len = None;
                    cache.tokens.truncate(cache.prompt_len);
                }
            }
        }
        let chunk = self.spec_chunk_rows();
        if let Some(cache) = self.prefix.as_mut() {
            let chunk = chunk.ok_or_else(|| {
                BundleError::Forward("Qwen4 forward resources are not attached".to_string())
            })?;
            cache.published = false;
            cache.mode = mode;
            cache.chunk = chunk;
            cache.capture_at = Some(prompt.len());
            cache.generation += 1;
            cache.periodic_due_from = Some(plan.start_pos);
            cache.periodic_done = false;
            if let Some(radix) = cache.radix.as_mut() {
                match plan.source {
                    Qwen4PrefixSource::Live => radix.stats.hits_live += 1,
                    Qwen4PrefixSource::Prompt => radix.stats.hits_prompt += 1,
                    Qwen4PrefixSource::Radix => radix.stats.hits_radix += 1,
                    Qwen4PrefixSource::Cold => {}
                }
            }
        }
        // Retained granules that left the live bank are cache-charged now.
        if self.radix_cache_attached() && action != BeginAction::Live {
            self.refresh_live_granules(&*gpu);
            self.evict_for(Some(&mut *gpu), 0)?;
        }
        if hipfire_config::developer_var("HIPFIRE_QWEN_CACHE_TRACE").is_ok_and(|value| value == "1")
        {
            match self.prefix.as_ref().and_then(|cache| cache.radix.as_ref()) {
                Some(radix) => eprintln!(
                    "[qwen4-radix] begin source={:?} start={} prompt={} mode={mode:?} fork={} alias={} copied={} reloc_ns={} attach_ns={} ledger={} entries={} banks={} retired_va={}",
                    plan.source,
                    plan.start_pos,
                    prompt.len(),
                    report.fork,
                    report.alias_bytes,
                    report.copied_bytes,
                    report.relocation_ns,
                    report.attach_ns,
                    radix.stats.ledger_bytes,
                    radix.pool.len(),
                    radix.stats.banks_created,
                    hip_bridge::retired_va_bytes()
                ),
                None => eprintln!(
                    "[qwen4-prefix] begin source={:?} start={} prompt={} mode={mode:?}",
                    plan.source,
                    plan.start_pos,
                    prompt.len()
                ),
            }
        }
        Ok(())
    }

    /// A failed restore: before any live write the old state is intact and the
    /// error is returned as is; after one every owner is reset.
    fn fail_restore(&mut self, gpu: &mut Gpu, failure: RestoreFailure) -> BundleError {
        if !failure.live_modified {
            return failure.error;
        }
        match self.reset(gpu) {
            Ok(()) => failure.error,
            Err(reset) => BundleError::Forward(format!(
                "{}; reset after the failed prefix restore also failed: {reset}",
                failure.error
            )),
        }
    }

    /// Restore the checkpoint `source` into every owner. In place iff the
    /// checkpoint's bank is still the live bank with its rows unchanged and no
    /// sealed granule lies above the bytes the restore writes; otherwise a
    /// fresh bank forked from the checkpoint's retained granules replaces the
    /// live one (typed PM4 relocation, old bank freed).
    fn restore_checkpoint_owners(
        &mut self,
        gpu: &mut Gpu,
        source: CheckpointRef,
        mode: Qwen4PrefixMode,
    ) -> Result<RestoreReport, RestoreFailure> {
        self.invalidate_ple_epoch().map_err(RestoreFailure::clean)?;
        let Qwen4Bundle {
            state, mtp, prefix, ..
        } = self;
        let Some(cache) = prefix.as_mut() else {
            return Err(RestoreFailure::clean(BundleError::Forward(
                "Qwen4 prefix cache is not attached".to_string(),
            )));
        };
        let head_needed = mode == Qwen4PrefixMode::NativeMtp;
        let (report, bank_action, cleanup) = {
            let pool = cache.radix.as_ref().map(|radix| &radix.pool);
            let Some(x) = resolve_checkpoint(&cache.staged, &cache.local, pool, source) else {
                return Err(RestoreFailure::clean(BundleError::Forward(
                    "Qwen4 prefix checkpoint is no longer retained".to_string(),
                )));
            };
            let b = x.tokens.len();
            if x.mode != mode || b == 0 || x.slot.target.position() != Some(b) {
                return Err(RestoreFailure::clean(BundleError::Forward(format!(
                    "Qwen4 prefix checkpoint at {b} ({:?}) does not match a {mode:?} restore",
                    x.mode
                ))));
            }
            if head_needed
                && !(mtp.is_some() && x.slot.head.as_ref().is_some_and(|h| h.position() == Some(b)))
            {
                return Err(RestoreFailure::clean(BundleError::Forward(
                    "Qwen4 native MTP checkpoint has no head state".to_string(),
                )));
            }
            let in_place = x.same_bank
                && x.bank == cache.bank
                && state.context_writable_at(gpu, b)
                && (!head_needed
                    || mtp
                        .as_ref()
                        .is_some_and(|mtp| mtp.context_writable_at(gpu, b)));
            let mut report = RestoreReport::default();
            let mut cleanup = None;
            let bank_action;
            if in_place {
                state
                    .ensure_mapped_capacity(gpu, b)
                    .map_err(|e| RestoreFailure::clean(BundleError::State(e)))?;
                if head_needed {
                    mtp.as_mut()
                        .ok_or_else(|| {
                            RestoreFailure::clean(BundleError::Forward(
                                "Qwen4 MTP resources are not attached".to_string(),
                            ))
                        })?
                        .ensure_mapped_capacity(gpu, b)
                        .map_err(|e| RestoreFailure::clean(BundleError::Forward(e.to_string())))?;
                }
                state
                    .restore_checkpoint(gpu, &x.slot.target)
                    .map_err(|e| RestoreFailure::dirty(BundleError::State(e)))?;
                let mut head_swapped = false;
                if head_needed {
                    let head = x.slot.head.as_ref().ok_or_else(|| {
                        RestoreFailure::dirty(BundleError::Forward(
                            "Qwen4 native MTP checkpoint has no head state".to_string(),
                        ))
                    })?;
                    mtp.as_mut()
                        .ok_or_else(|| {
                            RestoreFailure::dirty(BundleError::Forward(
                                "Qwen4 MTP resources are not attached".to_string(),
                            ))
                        })?
                        .restore_checkpoint(gpu, head)
                        .map_err(|e| RestoreFailure::dirty(BundleError::Forward(e.to_string())))?;
                } else if let Some(m) = mtp.as_ref() {
                    // An AR prefill leaves the head cold, as a reset would; a
                    // sealed head bank is swapped, never written.
                    if m.context_sealed(gpu) {
                        let fresh = m.new_context_bank(gpu).map_err(|e| {
                            RestoreFailure::dirty(BundleError::Forward(e.to_string()))
                        })?;
                        let swap = swap_context_banks(state, mtp.as_mut(), gpu, None, Some(fresh))
                            .map_err(RestoreFailure::from_swap)?;
                        report.relocation_ns = swap.relocation_ns;
                        report.bank_replaced = true;
                        report.fresh_bank = true;
                        cleanup = swap.cleanup;
                        head_swapped = true;
                    }
                }
                if !head_needed {
                    if let Some(m) = mtp.as_mut() {
                        m.reset(gpu).map_err(|e| {
                            RestoreFailure::dirty(BundleError::Forward(e.to_string()))
                        })?;
                    }
                }
                bank_action = if head_swapped {
                    BankAction::Replaced
                } else {
                    BankAction::Rewound(b)
                };
            } else {
                let Some(context) = x.context.as_ref() else {
                    return Err(RestoreFailure::clean(BundleError::Forward(format!(
                        "Qwen4 prefix checkpoint at {b} cannot be restored in place and retains no context"
                    ))));
                };
                let started = Instant::now();
                let device = gpu.device_id;
                let mut forked = gpu
                    .fork_vmm_prefixes(context, &[device])
                    .map_err(|e| RestoreFailure::clean(BundleError::Hip(e)))?;
                let target_arenas = state.context_tensors().len();
                let head_arenas = if head_needed {
                    mtp.as_ref().map_or(0, |mtp| mtp.context_tensors().len())
                } else {
                    0
                };
                if forked.len() != target_arenas + head_arenas {
                    let count = forked.len();
                    let cleanup = free_banks(gpu, Some(forked), None);
                    return Err(RestoreFailure::clean(BundleError::Forward(format!(
                        "Qwen4 fork produced {count} arenas, expected {} {}",
                        target_arenas + head_arenas,
                        cleanup.map_or(String::new(), |e| format!("(cleanup failed: {e})"))
                    ))));
                }
                let head_forked = forked.split_off(target_arenas);
                let head_bank = if head_needed {
                    Some(head_forked)
                } else if let Some(m) = mtp.as_ref().filter(|m| m.context_sealed(gpu)) {
                    match m.new_context_bank(gpu) {
                        Ok(fresh) => Some(fresh),
                        Err(error) => {
                            let _ = free_banks(gpu, Some(forked), None);
                            return Err(RestoreFailure::clean(BundleError::Forward(
                                error.to_string(),
                            )));
                        }
                    }
                } else {
                    None
                };
                report.alias_bytes = context.alias_bytes();
                report.copied_bytes = context.frontier_bytes();
                let swap = swap_context_banks(state, mtp.as_mut(), gpu, Some(forked), head_bank)
                    .map_err(RestoreFailure::from_swap)?;
                report.relocation_ns = swap.relocation_ns;
                cleanup = swap.cleanup;
                state
                    .ensure_mapped_capacity(gpu, b)
                    .map_err(|e| RestoreFailure::dirty(BundleError::State(e)))?;
                state
                    .restore_checkpoint(gpu, &x.slot.target)
                    .map_err(|e| RestoreFailure::dirty(BundleError::State(e)))?;
                if head_needed {
                    let head = x.slot.head.as_ref().ok_or_else(|| {
                        RestoreFailure::dirty(BundleError::Forward(
                            "Qwen4 native MTP checkpoint has no head state".to_string(),
                        ))
                    })?;
                    let m = mtp.as_mut().ok_or_else(|| {
                        RestoreFailure::dirty(BundleError::Forward(
                            "Qwen4 MTP resources are not attached".to_string(),
                        ))
                    })?;
                    m.ensure_mapped_capacity(gpu, b)
                        .and_then(|()| m.restore_checkpoint(gpu, head))
                        .map_err(|e| RestoreFailure::dirty(BundleError::Forward(e.to_string())))?;
                } else if let Some(m) = mtp.as_mut() {
                    m.reset(gpu)
                        .map_err(|e| RestoreFailure::dirty(BundleError::Forward(e.to_string())))?;
                }
                report.fork = true;
                report.bank_replaced = true;
                report.attach_ns = started.elapsed().as_nanos() as u64;
                bank_action = BankAction::Replaced;
            }
            (report, bank_action, cleanup)
        };
        match bank_action {
            BankAction::Rewound(position) => {
                let bank = cache.bank;
                cache.bank_rewound(bank, position);
            }
            BankAction::Replaced => {
                let old = cache.start_bank();
                cache.bank_gone(old);
            }
        }
        if let Some(radix) = cache.radix.as_mut() {
            let stats = &mut radix.stats;
            if report.bank_replaced {
                stats.banks_created += 1;
                stats.relocations += 1;
                stats.relocation_ns_last = report.relocation_ns;
            }
            if report.fork {
                stats.forks += 1;
                stats.alias_bytes += report.alias_bytes as u64;
                stats.copied_bytes += report.copied_bytes as u64;
            } else if report.fresh_bank {
                stats.fresh_banks += 1;
            }
        }
        match cleanup {
            Some(error) => Err(RestoreFailure::dirty(error)),
            None => Ok(report),
        }
    }

    /// Absolute position the running prefill must checkpoint at, if any.
    pub fn prefix_capture_at(&self) -> Option<usize> {
        self.prefix.as_ref().and_then(|cache| cache.capture_at)
    }

    /// Checkpoint the live state at the end of the armed prompt (see
    /// [`Self::stage_prefix_boundary`]); `prefix` is the whole prompt.
    pub fn stage_prefix(&mut self, gpu: &mut Gpu, prefix: &[u32]) -> Result<(), BundleError> {
        self.stage_prefix_boundary(gpu, prefix, true)
    }

    /// Checkpoint the live state at exactly `prefix.len()` tokens: target (and
    /// for native MTP, the head) must have consumed exactly `prefix` and no row
    /// capture may be armed. `end_of_prompt` is the armed prompt end: on
    /// success the host record is `prefix` and the live record is dropped; a
    /// failure leaves no valid checkpoint. Any other boundary is an optional
    /// anchor/periodic capture: budget refusal or a capture failure drops it
    /// and the prefill continues. Staged checkpoints are published by
    /// [`Self::commit_prefix`].
    pub fn stage_prefix_boundary(
        &mut self,
        gpu: &mut Gpu,
        prefix: &[u32],
        end_of_prompt: bool,
    ) -> Result<(), BundleError> {
        let (at, mode) = match self.prefix.as_ref() {
            Some(cache) => (cache.capture_at, cache.mode),
            None => return Ok(()),
        };
        let b = prefix.len();
        let marks = self.prefix_marks();
        let armed = match at {
            Some(end) if end_of_prompt => end == b,
            Some(end) => b > 0 && b < end,
            None => false,
        };
        let result = if armed && marks.aligned_at(b, mode) && !marks.row_capture_armed {
            self.capture_boundary(gpu, prefix, mode, end_of_prompt)
        } else {
            Err(BundleError::Forward(format!(
                "Qwen4 prefix checkpoint at {b} is not aligned (armed {at:?}, target {}, mtp {:?})",
                marks.target_position, marks.mtp_position
            )))
        };
        let trace = hipfire_config::developer_var("HIPFIRE_QWEN_CACHE_TRACE")
            .is_ok_and(|value| value == "1");
        let Some(cache) = self.prefix.as_mut() else {
            return result.map(|_| ());
        };
        let anchor = cache.anchor;
        if end_of_prompt {
            cache.capture_at = None;
            cache.published = false;
            cache.live_len = None;
            cache.prompt_len = 0;
            cache.tokens.clear();
            let record = match result {
                Ok(Capture::Staged(checkpoint)) => {
                    cache.note_capture(&checkpoint, CaptureKind::EndOfPrompt, trace);
                    cache.staged.push(checkpoint);
                    Ok(())
                }
                Ok(Capture::Duplicate(id)) => {
                    cache.set_local(LocalEop::Published(id));
                    if let Some(radix) = cache.radix.as_mut() {
                        radix.stats.duplicates += 1;
                    }
                    Ok(())
                }
                Ok(Capture::Skipped) => Err(BundleError::Forward(
                    "Qwen4 end-of-prompt checkpoint was skipped".to_string(),
                )),
                Err(error) => Err(error),
            };
            return match record {
                Ok(()) => {
                    cache.tokens.extend_from_slice(prefix);
                    cache.prompt_len = b;
                    Ok(())
                }
                Err(error) => {
                    self.invalidate_prefix();
                    Err(error)
                }
            };
        }
        match result {
            Ok(Capture::Staged(checkpoint)) => {
                let kind = if anchor == Some(b) {
                    CaptureKind::Anchor
                } else {
                    CaptureKind::Periodic
                };
                cache.note_capture(&checkpoint, kind, trace);
                cache.staged.push(checkpoint);
                cache.after_optional_capture(b, kind);
                Ok(())
            }
            Ok(Capture::Duplicate(_)) | Ok(Capture::Skipped) => {
                let kind = if anchor == Some(b) {
                    CaptureKind::Anchor
                } else {
                    CaptureKind::Periodic
                };
                cache.after_optional_capture(b, kind);
                Ok(())
            }
            Err(error) => {
                // Misaligned optional capture: a caller defect, fail closed.
                Err(error)
            }
        }
    }

    /// Capture the target (and native head) at exactly `prefix.len()`.
    fn capture_boundary(
        &mut self,
        gpu: &mut Gpu,
        prefix: &[u32],
        mode: Qwen4PrefixMode,
        end_of_prompt: bool,
    ) -> Result<Capture, BundleError> {
        self.quiesce_ple()?;
        let di = mode_index(mode);
        if let Some(radix) = self.prefix.as_ref().and_then(|cache| cache.radix.as_ref()) {
            if let Some(id) = radix.pool.find_exact(&radix.domains[di], prefix) {
                return Ok(Capture::Duplicate(id));
            }
        }
        let Some(slot) = self.acquire_slot(gpu, end_of_prompt)? else {
            return if end_of_prompt {
                Err(BundleError::Forward(
                    "Qwen4 prefix cache has no checkpoint slot for the end of prompt".to_string(),
                ))
            } else {
                Ok(Capture::Skipped)
            };
        };
        match self.fill_checkpoint(gpu, prefix, mode, slot, end_of_prompt) {
            Ok(checkpoint) => Ok(Capture::Staged(checkpoint)),
            Err((mut slot, error)) => {
                slot.invalidate();
                if let Some(cache) = self.prefix.as_mut() {
                    cache.slots.push(slot);
                }
                if end_of_prompt {
                    Err(error)
                } else {
                    if hipfire_config::developer_var("HIPFIRE_QWEN_CACHE_TRACE")
                        .is_ok_and(|value| value == "1")
                    {
                        eprintln!(
                            "[qwen4-radix] optional capture at {} dropped: {error}",
                            prefix.len()
                        );
                    }
                    Ok(Capture::Skipped)
                }
            }
        }
    }

    /// Copy the bounded state (and, with a radix store, retain the context) of
    /// the live owners into `slot`.
    fn fill_checkpoint(
        &mut self,
        gpu: &mut Gpu,
        prefix: &[u32],
        mode: Qwen4PrefixMode,
        mut slot: Qwen4CheckpointSlot,
        end_of_prompt: bool,
    ) -> Result<Qwen4Checkpoint, (Qwen4CheckpointSlot, BundleError)> {
        let b = prefix.len();
        if let Err(error) = self.state.capture_checkpoint(gpu, &mut slot.target) {
            return Err((slot, BundleError::State(error)));
        }
        let head_result = match mode {
            Qwen4PrefixMode::NativeMtp => match (self.mtp.as_mut(), slot.head.as_mut()) {
                (Some(mtp), Some(head)) => mtp
                    .capture_checkpoint(gpu, head)
                    .map_err(|error| BundleError::Forward(error.to_string())),
                _ => Err(BundleError::Forward(
                    "Qwen4 MTP checkpoint resources are not attached".to_string(),
                )),
            },
            Qwen4PrefixMode::Ar => {
                if let Some(head) = slot.head.as_mut() {
                    head.invalidate();
                }
                Ok(())
            }
        };
        if let Err(error) = head_result {
            return Err((slot, error));
        }
        let radix_on = self.radix_cache_attached();
        let context = if radix_on {
            match self.capture_context(gpu, mode) {
                Ok(context) => context,
                Err(error) => return Err((slot, error)),
            }
        } else {
            None
        };
        if radix_on && context.is_none() && !end_of_prompt {
            return Err((
                slot,
                BundleError::Forward("Qwen4 radix budget refused the optional capture".to_string()),
            ));
        }
        let chunk = self.spec_chunk_rows().unwrap_or(0);
        let bank = self.prefix.as_ref().map_or(0, |cache| cache.bank);
        Ok(Qwen4Checkpoint {
            slot,
            context,
            tokens: Arc::from(prefix),
            mode,
            chunk,
            bank,
            same_bank: true,
            eop: end_of_prompt,
            hits: 0,
            cost: gdsf_cost(b),
            priority: 0.0,
        })
    }

    /// Retain the live context arenas at the current marks when the ledger can
    /// afford the frontier copies; `None` = local-only (budget refused).
    fn capture_context(
        &mut self,
        gpu: &mut Gpu,
        mode: Qwen4PrefixMode,
    ) -> Result<Option<VmmPrefixSet>, BundleError> {
        let estimate = {
            let specs = self.context_specs(mode)?;
            slab_estimate_total(&*gpu, &specs)
        };
        self.refresh_live_granules(&*gpu);
        self.evict_for(Some(&mut *gpu), estimate)?;
        let affordable = self.prefix.as_ref().is_some_and(|cache| {
            cache.radix.as_ref().is_some_and(|radix| {
                cache.ledger_bytes().saturating_add(estimate) <= radix.limits.device_bytes
            })
        });
        if !affordable {
            return Ok(None);
        }
        let specs = self.context_specs(mode)?;
        let set = gpu
            .capture_vmm_prefixes(&specs)
            .map_err(BundleError::Hip)?;
        Ok(Some(set))
    }

    /// Capture specs of the live context arenas in arena order: the target's
    /// 48, then the native head's 4.
    fn context_specs(&self, mode: Qwen4PrefixMode) -> Result<Vec<VmmPrefixSpec<'_>>, BundleError> {
        let mut specs = self
            .state
            .context_prefix_specs()
            .map_err(BundleError::State)?;
        if mode == Qwen4PrefixMode::NativeMtp {
            let mtp = self.mtp.as_ref().ok_or_else(|| {
                BundleError::Forward("Qwen4 MTP resources are not attached".to_string())
            })?;
            specs.extend(
                mtp.context_prefix_specs()
                    .map_err(|error| BundleError::Forward(error.to_string()))?,
            );
        }
        Ok(specs)
    }

    /// A checkpoint slot for a capture: a free one, else (radix) a new one
    /// within the count and byte budget, else (radix) one freed by evicting
    /// the oldest unpinned entries, else (`steal_local`, the end of prompt)
    /// the superseded local checkpoint's. `None` = no slot.
    fn acquire_slot(
        &mut self,
        gpu: &mut Gpu,
        steal_local: bool,
    ) -> Result<Option<Qwen4CheckpointSlot>, BundleError> {
        let allocation = {
            let Some(cache) = self.prefix.as_mut() else {
                return Ok(None);
            };
            if let Some(slot) = cache.slots.pop() {
                return Ok(Some(slot));
            }
            cache.radix.as_ref().and_then(|radix| {
                (cache.slots_allocated < radix.limits.checkpoints)
                    .then_some((radix.slot_bytes, radix.limits.device_bytes))
            })
        };
        if let Some((slot_bytes, device_bytes)) = allocation {
            self.refresh_live_granules(&*gpu);
            let ledger = self.prefix.as_ref().map_or(0, |cache| cache.ledger_bytes());
            if ledger.saturating_add(slot_bytes) <= device_bytes {
                let slot = self.new_checkpoint_slot(gpu)?;
                if let Some(cache) = self.prefix.as_mut() {
                    cache.slots_allocated += 1;
                }
                return Ok(Some(slot));
            }
        }
        let Some(cache) = self.prefix.as_mut() else {
            return Ok(None);
        };
        if cache.radix.is_some() {
            let mut first_error = None;
            let slot = loop {
                if !cache.evict_one(
                    Some(&mut *gpu),
                    EvictMode::Checkpoint,
                    false,
                    &mut first_error,
                ) {
                    break None;
                }
                if let Some(slot) = cache.slots.pop() {
                    break Some(slot);
                }
            };
            if let Some(error) = first_error {
                if let Some(slot) = slot {
                    cache.slots.push(slot);
                }
                return Err(error);
            }
            if slot.is_some() {
                return Ok(slot);
            }
        }
        if steal_local
            && matches!(&cache.local, Some(LocalEop::Owned(checkpoint)) if checkpoint.context.is_none())
        {
            if let Some(LocalEop::Owned(checkpoint)) = cache.local.take() {
                let mut slot = checkpoint.slot;
                slot.invalidate();
                return Ok(Some(slot));
            }
        }
        Ok(None)
    }

    fn new_checkpoint_slot(&mut self, gpu: &mut Gpu) -> Result<Qwen4CheckpointSlot, BundleError> {
        let target = self
            .state
            .new_checkpoint(gpu)
            .map_err(BundleError::State)?;
        let head = match self.mtp.as_ref() {
            Some(mtp) => match mtp.new_checkpoint(gpu) {
                Ok(head) => Some(head),
                Err(error) => {
                    let _ = target.free_gpu(gpu);
                    return Err(BundleError::Forward(error.to_string()));
                }
            },
            None => None,
        };
        Ok(Qwen4CheckpointSlot { target, head })
    }

    /// Recompute which physical granules the live banks map (the ledger's
    /// "already charged to the baseline" set).
    fn refresh_live_granules(&mut self, gpu: &Gpu) {
        let Qwen4Bundle {
            state, mtp, prefix, ..
        } = self;
        if let Some(radix) = prefix.as_mut().and_then(|cache| cache.radix.as_mut()) {
            radix.live_granules = live_granule_ids(state, mtp.as_ref(), gpu);
        }
    }

    /// Enforce the radix budgets (§8): evict the oldest unpinned leaves until
    /// the ledger plus `need` fits `limits.device_bytes` (only with a GPU: it
    /// needs the live banks) and the entry/host caps hold. With a GPU the
    /// freed checkpoints are released now, else they are deferred.
    fn evict_for(&mut self, gpu: Option<&mut Gpu>, need: u64) -> Result<(), BundleError> {
        match self.prefix.as_mut() {
            Some(cache) => cache.evict_for(gpu, need),
            None => Ok(()),
        }
    }

    /// Free every discarded staged checkpoint now (the GPU is quiescent).
    fn discard_staged(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let Some(cache) = self.prefix.as_mut() else {
            return Ok(());
        };
        let mut first = None;
        for checkpoint in std::mem::take(&mut cache.staged) {
            if let Err(error) = recycle_checkpoint(&mut cache.slots, gpu, checkpoint) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// Free the checkpoints host-only publication deferred (quiescent GPU).
    fn drain_deferred(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        match self.prefix.as_mut() {
            Some(cache) => cache.drain_deferred(gpu),
            None => Ok(()),
        }
    }

    /// Swap a fresh, empty bank in for every sealed one (a sealed bank's lower
    /// bytes may never be written). Returns whether any bank was swapped.
    fn swap_sealed_banks(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let target_sealed = self.state.context_sealed(gpu);
        let head_sealed = self.mtp.as_ref().is_some_and(|mtp| mtp.context_sealed(gpu));
        if !target_sealed && !head_sealed {
            return Ok(());
        }
        let target = if target_sealed {
            Some(
                self.state
                    .new_context_bank(gpu)
                    .map_err(BundleError::State)?,
            )
        } else {
            None
        };
        let head = if head_sealed {
            match self.mtp.as_ref().map(|mtp| mtp.new_context_bank(gpu)) {
                Some(Ok(bank)) => Some(bank),
                Some(Err(error)) => {
                    let _ = free_banks(gpu, target, None);
                    return Err(BundleError::Forward(error.to_string()));
                }
                None => None,
            }
        } else {
            None
        };
        let Qwen4Bundle {
            state, mtp, prefix, ..
        } = self;
        let swap = swap_context_banks(state, mtp.as_mut(), gpu, target, head)
            .map_err(|failure| failure.error)?;
        if let Some(radix) = prefix.as_mut().and_then(|cache| cache.radix.as_mut()) {
            radix.stats.fresh_banks += 1;
            radix.stats.banks_created += 1;
            radix.stats.relocations += 1;
            radix.stats.relocation_ns_last = swap.relocation_ns;
        }
        swap.cleanup.map_or(Ok(()), Err)
    }

    /// Publish after the client committed the request. `consumed_tokens` is
    /// the full host history the device now holds (prompt plus every consumed
    /// generated token). Staged checkpoints are first published host-only
    /// (radix: into the shared index and pool; otherwise the end of prompt
    /// becomes the local checkpoint); GPU frees are deferred to the next
    /// quiescent begin / reset / unload. The live state is then published when
    /// it is exactly that history (see `Qwen4PrefixCache::commit`); the
    /// end-of-prompt checkpoint is published whenever it is valid, so a
    /// terminal that leaves no exact live state still keeps `P`.
    pub fn commit_prefix(&mut self, consumed_tokens: &[u32]) {
        if let Some(cache) = self.prefix.as_mut() {
            cache.publish_staged();
        }
        let marks = self.prefix_marks();
        if let Some(cache) = self.prefix.as_mut() {
            cache.commit(&marks, consumed_tokens);
        }
    }

    /// Drop the live record and the local end-of-prompt checkpoint (radix
    /// entries survive). Host-only.
    pub fn invalidate_prefix(&mut self) {
        if let Some(cache) = self.prefix.as_mut() {
            cache.clear();
            cache.drop_local();
        }
    }

    /// AR prefill of the full canonical `prompt` under `plan`, keeping only
    /// the final logits row: begin (reset, live continuation or checkpoint
    /// restore), then the suffix `prompt[plan.start_pos..]` as ONE final
    /// forward unless a capture (anchor / periodic) is due before the end, in
    /// which case the prefill splits there (silent forward, stage), and the
    /// end-of-prompt checkpoint at `prompt.len()`.
    pub fn prefill_final(
        &mut self,
        gpu: &mut Gpu,
        prompt: &[u32],
        plan: Qwen4PrefixPlan,
        logits: &GpuTensor,
    ) -> Result<(), BundleError> {
        self.begin_prefix(gpu, prompt, plan, Qwen4PrefixMode::Ar)?;
        let chunk = self.spec_chunk_rows().ok_or_else(|| {
            BundleError::Forward("Qwen4 forward resources are not attached".to_string())
        })?;
        let len = prompt.len();
        if plan.start_pos >= len {
            return Err(BundleError::Forward(
                "Qwen4 prefill has no suffix tokens".to_string(),
            ));
        }
        let mut pos = plan.start_pos;
        while pos < len {
            // The first capture strictly inside the prompt over the chunk ends
            // the forward's own tiles land on.
            let mut split = None;
            let mut probe = pos;
            while probe < len {
                let natural = probe.saturating_add(chunk).min(len);
                if let Some(end) = self.next_prefix_capture(probe, natural) {
                    if end < len {
                        split = Some(end);
                        break;
                    }
                }
                probe = natural;
            }
            match split {
                Some(end) => {
                    self.set_ple_lookahead(&prompt[end..]);
                    self.forward_chunk_silent(gpu, &prompt[pos..end])?;
                    self.stage_prefix_boundary(gpu, &prompt[..end], false)?;
                    pos = end;
                }
                None => {
                    self.forward_chunk_final(gpu, &prompt[pos..], logits, None)?;
                    pos = len;
                }
            }
        }
        if self.prefix_capture_at() == Some(len) {
            self.stage_prefix(gpu, prompt)?;
        }
        Ok(())
    }

    pub fn snapshot(&mut self, gpu: &mut Gpu) -> Result<Qwen4StateSnapshot, BundleError> {
        self.quiesce_ple()?;
        self.state.snapshot(gpu).map_err(BundleError::State)
    }

    pub fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.invalidate_ple_epoch()?;
        self.state
            .restore(gpu, snapshot)
            .map_err(BundleError::State)
    }
    pub(crate) fn restore_retain(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.quiesce_ple()?;
        self.state
            .restore_retain(gpu, snapshot)
            .map_err(BundleError::State)
    }

    pub(crate) fn validate_commit(&self, snapshot: Qwen4StateSnapshot) -> Result<(), BundleError> {
        self.state
            .validate_commit(snapshot)
            .map_err(BundleError::State)
    }

    pub(crate) fn commit_validated(&mut self, snapshot: Qwen4StateSnapshot) {
        self.state.commit_validated(snapshot);
    }

    pub fn commit(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.quiesce_ple()?;
        self.state.commit(snapshot, gpu).map_err(BundleError::State)
    }

    /// Teardown is deliberately ordered: stop/quiesce PLE reads and release
    /// leases/page-cache resources, free mutable GPU state, free finalized
    /// resident weights, then drain the attached transaction/census.
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let Qwen4Bundle {
            weights,
            state,
            ple_rows,
            weight_store,
            execution,
            mtp,
            spec_logits,
            spec_top1,
            spec_hidden,
            prefix,
            ..
        } = self;
        let ple_result = ple_rows.unload().map(|_| ()).map_err(BundleError::PleRows);
        // Retained checkpoints release their granule leases before the banks go.
        let prefix_result = prefix
            .map(|cache| cache.free_gpu(gpu))
            .unwrap_or(Ok(()));
        let execution_result = execution
            .map(|forward| forward.free_gpu(gpu).map_err(BundleError::Hip))
            .unwrap_or(Ok(()));
        let mtp_result = mtp
            .map(|mtp| {
                mtp.free_gpu(gpu)
                    .map_err(|error| BundleError::Forward(error.to_string()))
            })
            .unwrap_or(Ok(()));
        let mut spec_error = None;
        for tensor in [spec_logits, spec_top1, spec_hidden].into_iter().flatten() {
            if let Err(error) = gpu.free_tensor(tensor) {
                spec_error.get_or_insert(error);
            }
        }
        let spec_result = spec_error.map_or(Ok(()), |error| Err(BundleError::Hip(error)));
        let state_result = state.free_gpu(gpu).map_err(BundleError::State);
        let weight_result = weights.free_gpu(gpu).map_err(BundleError::Hip);
        let store_result = weight_store.drain(gpu).map_err(BundleError::Hip);
        first_bundle_error([
            ple_result,
            prefix_result,
            execution_result,
            mtp_result,
            spec_result,
            state_result,
            weight_result,
            store_result,
        ])
    }
}

impl hipfire_runtime::arch_model::ArchModel for Qwen4Bundle {
    fn dim(&self) -> usize {
        self.config.hidden_size
    }

    fn n_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn arch_key(&self) -> &'static str {
        "qwen4"
    }

    fn kv_cache_mut(&mut self) -> Option<&mut hipfire_runtime::llama::KvCache> {
        None
    }

    fn reset_session_state(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu).map_err(|error| error.to_string())
    }

    fn free_gpu(self: Box<Self>, gpu: &mut Gpu) {
        if let Err(error) = Qwen4Bundle::free_gpu(*self, gpu) {
            eprintln!("Qwen4 bundle teardown failed: {error}");
        }
    }
}

const PLE_SHARD_NAME_PREFIX: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";

fn ple_shard_index(name: &str) -> Result<usize, WeightError> {
    let suffix = name
        .strip_prefix(PLE_SHARD_NAME_PREFIX)
        .and_then(|name| name.strip_suffix(".weight"))
        .ok_or_else(|| {
            WeightError::DescriptorMismatch(format!("invalid PLE shard name '{name}'"))
        })?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(WeightError::DescriptorMismatch(format!(
            "invalid PLE shard name '{name}'"
        )));
    }
    let index = suffix
        .parse::<usize>()
        .map_err(|_| WeightError::DescriptorMismatch(format!("invalid PLE shard name '{name}'")))?;
    if index >= PLE_SHARD_COUNT || index.to_string() != suffix {
        return Err(WeightError::DescriptorMismatch(format!(
            "PLE shard index {index} is outside canonical range in '{name}'"
        )));
    }
    Ok(index)
}

fn ordered_ple_entries<'a>(
    manifest: &'a Qwen4Manifest,
) -> Result<Vec<&'a WeightEntry>, WeightError> {
    let entries = manifest.external_entries().collect::<Vec<_>>();
    if entries.len() != PLE_SHARD_COUNT {
        return Err(WeightError::PleShardCount {
            expected: PLE_SHARD_COUNT,
            actual: entries.len(),
        });
    }
    let mut ordered = vec![None; PLE_SHARD_COUNT];
    for entry in entries {
        let index = ple_shard_index(&entry.name)?;
        if entry.layer != Some(1)
            || entry.logical_shape.as_slice() != [PLE_SHARD_ROWS, PLE_ROW_WIDTH]
        {
            return Err(WeightError::DescriptorMismatch(entry.name.clone()));
        }
        if ordered[index].replace(entry).is_some() {
            return Err(WeightError::DescriptorMismatch(format!(
                "duplicate PLE shard index {index}"
            )));
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, entry)| {
            entry.ok_or_else(|| {
                WeightError::DescriptorMismatch(format!("missing PLE shard index {index}"))
            })
        })
        .collect()
}

fn validate_ple_metadata(metadata: &PleHashMetadata) -> Result<(), BundleError> {
    let physical_rows = (PLE_SHARD_ROWS as u64)
        .checked_mul(PLE_SHARD_COUNT as u64)
        .ok_or_else(|| {
            BundleError::Weights(WeightError::DescriptorMismatch(
                "PLE physical row count overflow".to_string(),
            ))
        })?;
    if metadata.padded_rows() != physical_rows {
        return Err(BundleError::Weights(WeightError::DescriptorMismatch(
            format!(
                "PLE metadata padded rows {} do not match physical rows {physical_rows}",
                metadata.padded_rows()
            ),
        )));
    }
    let valid_rows = (0..PLE_SHARD_COUNT).try_fold(0u64, |sum, shard| {
        sum.checked_add(ple_valid_rows_for_shard(shard) as u64)
            .ok_or_else(|| {
                BundleError::Weights(WeightError::DescriptorMismatch(
                    "PLE valid row count overflow".to_string(),
                ))
            })
    })?;
    if metadata.valid_rows() != valid_rows {
        return Err(BundleError::Weights(WeightError::DescriptorMismatch(
            format!(
                "PLE metadata valid rows {} do not match canonical rows {valid_rows}",
                metadata.valid_rows()
            ),
        )));
    }
    Ok(())
}
fn ple_descriptors(
    transaction: &WeightLoadTransaction,
    manifest: &Qwen4Manifest,
    metadata: &PleHashMetadata,
) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
    validate_ple_metadata(metadata)?;
    let entries = ordered_ple_entries(manifest).map_err(BundleError::Weights)?;
    let mut descriptors: Vec<SourceRangeDescriptor> = Vec::with_capacity(entries.len());
    let mut valid_rows_total = 0u64;
    for (index, entry) in entries.into_iter().enumerate() {
        let descriptor = transaction
            .external_descriptor(&entry.name, entry.layer, 0)
            .ok_or_else(|| {
                BundleError::Weights(WeightError::MissingExternalDescriptor {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                })
            })?
            .clone();
        let (row_bytes, valid_rows) = match entry.residency {
            WeightResidency::ExternalRows {
                row_bytes,
                valid_rows,
            } => (row_bytes, valid_rows),
            WeightResidency::Resident | WeightResidency::HostMapped => {
                return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                    entry.name.clone(),
                )))
            }
        };
        // The manifest's stride is the tier it was declared for; the descriptor
        // side (length for its own declared dtype, shape, valid rows, HFQ index
        // seal, file bounds) was already checked when the transaction was
        // fulfilled. Here the declaration only has to name a tier the reader
        // decodes, so a sealed BF16-PLE artifact stays loadable next to a
        // Q8F16 one.
        if ![
            RowEncoding::Bf16.encoded_row_bytes(PLE_ROW_WIDTH),
            RowEncoding::Q8F16.encoded_row_bytes(PLE_ROW_WIDTH),
        ]
        .contains(&row_bytes)
            || valid_rows != ple_valid_rows_for_shard(index)
        {
            return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                entry.name.clone(),
            )));
        }
        valid_rows_total = valid_rows_total
            .checked_add(valid_rows as u64)
            .ok_or_else(|| {
                BundleError::Weights(WeightError::DescriptorMismatch(
                    "PLE valid row count overflow".to_string(),
                ))
            })?;
        if let Some(first) = descriptors.first() {
            if first.source_identity() != descriptor.source_identity() {
                return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                    format!("PLE shard {index} source identity differs"),
                )));
            }
        }
        // Fulfillment only enforces the index seal for HFQ sources; PLE rows
        // must come from one.
        if descriptor.source_identity().format != SourceFormat::Hfq {
            return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                entry.name.clone(),
            )));
        }
        descriptors.push(descriptor);
    }
    if valid_rows_total != metadata.valid_rows() {
        return Err(BundleError::Weights(WeightError::DescriptorMismatch(
            format!(
                "PLE metadata valid rows {} do not match manifest rows {valid_rows_total}",
                metadata.valid_rows()
            ),
        )));
    }
    Ok(descriptors)
}

/// The attached MTP head with its QSA context mapped through `position`,
/// the row the next step writes. Mapping happens here, outside any capture
/// or record (MTP steps never run under either), never inside the step.
fn mapped_mtp<'a>(
    mtp: Option<&'a mut Qwen4MtpGpu>,
    gpu: &mut Gpu,
    position: usize,
) -> Result<&'a mut Qwen4MtpGpu, BundleError> {
    let mtp = mtp
        .ok_or_else(|| BundleError::Forward("Qwen4 MTP resources are not attached".to_string()))?;
    let required = position
        .checked_add(1)
        .ok_or_else(|| BundleError::Forward("Qwen4 MTP position overflows".to_string()))?;
    mtp.ensure_mapped_capacity(gpu, required)
        .map_err(|error| BundleError::Forward(error.to_string()))?;
    Ok(mtp)
}

fn cleanup_transaction(primary: BundleError, rollback: hip_bridge::HipResult<()>) -> BundleError {
    match rollback {
        Ok(()) => primary,
        Err(error) => BundleError::Rollback {
            cause: primary.to_string(),
            error,
        },
    }
}

fn cleanup_bundle_failure(
    primary: BundleError,
    weights: hip_bridge::HipResult<()>,
    transaction: hip_bridge::HipResult<()>,
) -> BundleError {
    match (weights, transaction) {
        (Ok(()), Ok(())) => primary,
        (Err(weight), Ok(())) => BundleError::Rollback {
            cause: format!("{primary}; resident weight cleanup failed"),
            error: weight,
        },
        (Ok(()), Err(transaction)) => BundleError::Rollback {
            cause: format!("{primary}; transaction cleanup failed"),
            error: transaction,
        },
        (Err(weight), Err(transaction)) => BundleError::Rollback {
            cause: format!("{primary}; resident and transaction cleanup failed: {transaction}"),
            error: weight,
        },
    }
}

fn first_bundle_error(results: [Result<(), BundleError>; 8]) -> Result<(), BundleError> {
    let mut first = None;
    for result in results {
        if let Err(error) = result {
            if first.is_none() {
                first = Some(error);
            }
        }
    }
    first.map_or(Ok(()), Err)
}

#[derive(Debug)]
pub enum BundleError {
    Config(String),
    Weights(WeightError),
    State(StateError),
    PleRows(RowStoreError),
    Forward(String),
    Hip(hip_bridge::HipError),
    Rollback {
        cause: String,
        error: hip_bridge::HipError,
    },
    Transaction(WeightStoreError),
}

impl From<WeightError> for BundleError {
    fn from(value: WeightError) -> Self {
        Self::Weights(value)
    }
}

impl From<WeightStoreError> for BundleError {
    fn from(value: WeightStoreError) -> Self {
        Self::Transaction(value)
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 bundle config: {message}"),
            Self::Weights(error) => write!(f, "Qwen4 bundle weights: {error}"),
            Self::State(error) => write!(f, "Qwen4 bundle state: {error}"),
            Self::Forward(error) => write!(f, "Qwen4 bundle forward: {error}"),
            Self::PleRows(error) => write!(f, "Qwen4 bundle PLE rows: {error}"),
            Self::Hip(error) => write!(f, "Qwen4 bundle HIP teardown: {error}"),
            Self::Rollback { cause, error } => {
                write!(f, "Qwen4 bundle cleanup after {cause}: {error}")
            }
            Self::Transaction(error) => write!(f, "Qwen4 bundle transaction: {error}"),
        }
    }
}

impl std::error::Error for BundleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ple::PLE_HEAD_COUNT;

    #[test]
    fn rejects_metadata_with_same_padding_but_different_valid_rows() {
        let canonical = PleHashMetadata::qwen4();
        let mut sizes = *canonical.head_vocab_sizes();
        sizes[PLE_HEAD_COUNT - 1] -= 1;
        let mut offsets = [0u64; PLE_HEAD_COUNT];
        for head in 1..PLE_HEAD_COUNT {
            offsets[head] = offsets[head - 1] + sizes[head - 1];
        }
        let metadata = PleHashMetadata::from_stored(
            *canonical.multipliers(),
            sizes,
            offsets,
            canonical.padded_rows(),
        )
        .expect("one valid row removed still rounds to the same physical padding");

        let error = validate_ple_metadata(&metadata).unwrap_err();
        assert!(matches!(
            error,
            BundleError::Weights(WeightError::DescriptorMismatch(message))
                if message.contains("valid rows")
        ));
    }

    const CHUNK: usize = 256;

    /// Device marks of a target (and, for native MTP, head) sitting at `at`
    /// with a durable checkpoint at `checkpoint`.
    fn marks(mode: Qwen4PrefixMode, at: usize, checkpoint: Option<usize>) -> PrefixMarks {
        let mtp = mode == Qwen4PrefixMode::NativeMtp;
        PrefixMarks {
            target_position: at,
            mtp_position: mtp.then_some(at),
            target_prefix: checkpoint,
            mtp_prefix: if mtp { checkpoint } else { None },
            chunk: Some(CHUNK),
            row_capture_armed: false,
        }
    }

    /// A cache whose request staged `prompt` (unpublished), as after `stage_prefix`.
    fn staged(prompt: &[u32], mode: Qwen4PrefixMode) -> Qwen4PrefixCache {
        let mut cache = Qwen4PrefixCache::new(64);
        cache.mode = mode;
        cache.chunk = CHUNK;
        cache.generation = 7;
        cache.tokens.extend_from_slice(prompt);
        cache.prompt_len = prompt.len();
        cache
    }

    fn prompt(len: u32) -> Vec<u32> {
        (1..=len).collect()
    }

    /// `staged` + committed generation: record = prompt + `generated` tokens.
    fn committed(
        prompt_len: u32,
        generated: u32,
        mode: Qwen4PrefixMode,
    ) -> (Qwen4PrefixCache, Vec<u32>) {
        let mut cache = staged(&prompt(prompt_len), mode);
        let consumed = prompt(prompt_len + generated);
        let end = consumed.len();
        cache.commit(&marks(mode, end, Some(prompt_len as usize)), &consumed);
        (cache, consumed)
    }

    const AR: Qwen4PrefixMode = Qwen4PrefixMode::Ar;
    const MTP: Qwen4PrefixMode = Qwen4PrefixMode::NativeMtp;

    #[test]
    fn default_plan_is_unconditional_cold() {
        let plan = Qwen4PrefixPlan::default();
        assert_eq!(plan.start_pos, 0);
        assert_eq!(plan.source(), Qwen4PrefixSource::Cold);
    }

    #[test]
    fn cold_bind_ignores_cache_state() {
        let cache = Qwen4PrefixCache::new(64);
        let m = marks(AR, 0, None);
        let plan = cache.bind(&m, &prompt(10), 0, AR).unwrap();
        assert_eq!(
            (plan.start_pos, plan.source()),
            (0, Qwen4PrefixSource::Cold)
        );
        assert!(cache.bind(&m, &prompt(10), 4, AR).is_err());
    }

    #[test]
    fn live_bind_extends_the_committed_record() {
        let (cache, consumed) = committed(8, 5, AR);
        let m = marks(AR, 13, Some(8));
        let mut next = consumed.clone();
        next.extend([900, 901]);
        let plan = cache.bind(&m, &next, 13, AR).unwrap();
        assert_eq!(
            (plan.start_pos, plan.source()),
            (13, Qwen4PrefixSource::Live)
        );
        // L == N leaves no suffix.
        assert!(cache.bind(&m, &consumed, 13, AR).is_err());
        // The whole committed record must match, not just the prompt.
        let mut diverged = next.clone();
        diverged[10] = 777;
        assert!(cache.bind(&m, &diverged, 13, AR).is_err());
    }

    #[test]
    fn prompt_bind_restores_p_after_divergence() {
        let (cache, consumed) = committed(8, 5, AR);
        let m = marks(AR, 13, Some(8));
        let mut next = consumed[..8].to_vec();
        next.extend([500, 501, 502]);
        let plan = cache.bind(&m, &next, 8, AR).unwrap();
        assert_eq!(
            (plan.start_pos, plan.source()),
            (8, Qwen4PrefixSource::Prompt)
        );
        // Divergence inside the prompt leaves no matching checkpoint.
        let mut early = next.clone();
        early[7] = 999;
        assert!(cache.bind(&m, &early, 8, AR).is_err());
        // Only P (and L) are bindable starts, not an arbitrary LCP.
        assert!(cache.bind(&m, &next, 5, AR).is_err());
        assert!(cache.bind(&m, &next, 10, AR).is_err());
        // The prompt must leave a non-empty suffix.
        assert!(cache.bind(&m, &consumed[..8], 8, AR).is_err());
    }

    #[test]
    fn live_is_preferred_when_it_ends_at_the_checkpoint() {
        let (cache, consumed) = committed(8, 0, AR);
        let m = marks(AR, 8, Some(8));
        let mut next = consumed.clone();
        next.push(42);
        let plan = cache.bind(&m, &next, 8, AR).unwrap();
        assert_eq!(plan.source(), Qwen4PrefixSource::Live);
    }

    #[test]
    fn mode_or_chunk_mismatch_publishes_nothing() {
        let (cache, consumed) = committed(8, 5, MTP);
        let m = marks(MTP, 13, Some(8));
        let mut next = consumed.clone();
        next.push(42);
        assert!(cache.bind(&m, &next, 13, MTP).is_ok());
        assert!(cache.bind(&m, &next, 13, AR).is_err());
        assert!(cache.prefix_lineage(&m, AR).is_none());
        assert_eq!(cache.checkpoint_end(&m, AR), None);
        let wrong_chunk = PrefixMarks {
            chunk: Some(CHUNK / 2),
            ..m
        };
        assert!(cache.bind(&wrong_chunk, &next, 13, MTP).is_err());
        assert!(cache.prefix_lineage(&wrong_chunk, MTP).is_none());
        let detached = PrefixMarks { chunk: None, ..m };
        assert!(cache.prefix_lineage(&detached, MTP).is_none());
    }

    #[test]
    fn device_marks_gate_live_and_checkpoint() {
        let (cache, consumed) = committed(8, 5, MTP);
        let mut next = consumed.clone();
        next.push(42);
        let ok = marks(MTP, 13, Some(8));
        assert_eq!(cache.live_end(&ok, MTP), Some(13));
        // Target moved: live is gone, the durable checkpoint remains.
        let moved = PrefixMarks {
            target_position: 14,
            ..ok
        };
        assert!(cache.bind(&moved, &next, 13, MTP).is_err());
        assert_eq!(cache.prefix_lineage(&moved, MTP), Some(&consumed[..8]));
        // A lagging head is never live-eligible.
        let lagging = PrefixMarks {
            mtp_position: Some(12),
            ..ok
        };
        assert_eq!(cache.live_end(&lagging, MTP), None);
        assert!(cache.bind(&lagging, &next, 13, MTP).is_err());
        // Neither owner's checkpoint valid: no lineage at all.
        let gone = PrefixMarks {
            target_position: 14,
            target_prefix: None,
            ..ok
        };
        assert!(cache.prefix_lineage(&gone, MTP).is_none());
        // Head checkpoint missing invalidates the NativeMtp checkpoint.
        let no_head = PrefixMarks {
            target_position: 14,
            mtp_prefix: None,
            ..ok
        };
        assert_eq!(cache.checkpoint_end(&no_head, MTP), None);
    }

    #[test]
    fn stale_receipt_no_longer_rebinds() {
        let (mut cache, consumed) = committed(8, 5, AR);
        let m = marks(AR, 13, Some(8));
        let mut next = consumed.clone();
        next.push(42);
        let plan = cache.bind(&m, &next, 13, AR).unwrap();
        assert_eq!(cache.bind(&m, &next, 13, AR).unwrap(), plan);
        cache.generation += 1;
        assert_ne!(cache.bind(&m, &next, 13, AR).unwrap(), plan);
    }

    #[test]
    fn commit_publishes_live_and_extends_the_record_in_place() {
        let (cache, consumed) = committed(8, 5, MTP);
        let m = marks(MTP, 13, Some(8));
        assert_eq!(cache.prompt_len, 8);
        assert_eq!(cache.live_len, Some(13));
        assert!(cache.published);
        assert_eq!(cache.capture_at, None);
        assert_eq!(cache.prefix_lineage(&m, MTP), Some(&consumed[..]));
        assert_eq!(cache.checkpoint_end(&m, MTP), Some(8));
    }

    #[test]
    fn commit_with_misaligned_target_publishes_the_checkpoint_only() {
        for target in [12usize, 14] {
            let mut cache = staged(&prompt(8), AR);
            let consumed = prompt(13);
            cache.commit(&marks(AR, target, Some(8)), &consumed);
            assert_eq!(cache.live_len, None);
            assert_eq!(cache.tokens.len(), 8);
            assert!(cache.published);
            let m = marks(AR, target, Some(8));
            assert_eq!(cache.prefix_lineage(&m, AR), Some(&consumed[..8]));
        }
    }

    #[test]
    fn commit_with_lagging_head_or_armed_capture_publishes_the_checkpoint_only() {
        let consumed = prompt(13);
        let ok = marks(MTP, 13, Some(8));
        let lagging = PrefixMarks {
            mtp_position: Some(12),
            ..ok
        };
        let armed = PrefixMarks {
            row_capture_armed: true,
            ..ok
        };
        for m in [lagging, armed] {
            let mut cache = staged(&prompt(8), MTP);
            cache.commit(&m, &consumed);
            assert_eq!(cache.live_len, None);
            assert_eq!(cache.tokens.len(), 8);
            assert!(cache.published);
        }
    }

    #[test]
    fn commit_with_empty_or_foreign_history_publishes_the_checkpoint_only() {
        let m = marks(AR, 13, Some(8));
        let mut empty = staged(&prompt(8), AR);
        empty.commit(&m, &[]);
        assert_eq!(empty.live_len, None);
        assert_eq!(empty.tokens.len(), 8);
        assert!(empty.published);

        let mut shorter = staged(&prompt(8), AR);
        shorter.commit(&marks(AR, 5, Some(8)), &prompt(5));
        assert_eq!(shorter.live_len, None);

        let mut foreign = staged(&prompt(8), AR);
        let mut history = prompt(13);
        history[3] = 4242;
        foreign.commit(&m, &history);
        assert_eq!(foreign.live_len, None);
        assert_eq!(foreign.tokens.len(), 8);
    }

    #[test]
    fn commit_without_a_valid_checkpoint_or_live_state_publishes_nothing() {
        let mut cache = staged(&prompt(8), AR);
        cache.commit(&marks(AR, 12, None), &prompt(13));
        assert!(!cache.published);
        assert_eq!(cache.live_len, None);
        assert!(cache.prefix_lineage(&marks(AR, 12, None), AR).is_none());

        let mut unstaged = Qwen4PrefixCache::new(64);
        unstaged.commit(&marks(AR, 13, None), &prompt(13));
        assert!(!unstaged.published);
        assert_eq!(unstaged.live_len, None);
    }

    #[test]
    fn lineage_is_the_full_record_when_live_and_the_prompt_otherwise() {
        let (cache, consumed) = committed(8, 5, AR);
        let live = marks(AR, 13, Some(8));
        assert_eq!(cache.prefix_lineage(&live, AR), Some(&consumed[..]));
        let moved = marks(AR, 40, Some(8));
        assert_eq!(cache.prefix_lineage(&moved, AR), Some(&consumed[..8]));
        assert_eq!(cache.checkpoint_end(&moved, AR), Some(8));
    }

    #[test]
    fn unpublished_cache_has_no_lineage() {
        let cache = staged(&prompt(8), AR);
        let m = marks(AR, 8, Some(8));
        assert!(cache.prefix_lineage(&m, AR).is_none());
        assert_eq!(cache.checkpoint_end(&m, AR), None);
        assert!(cache.bind(&m, &prompt(12), 8, AR).is_err());
    }

    #[test]
    fn clear_unpublishes_and_advances_the_generation() {
        let (mut cache, _) = committed(8, 5, AR);
        let before = cache.generation;
        cache.clear();
        assert!(cache.tokens.is_empty());
        assert_eq!((cache.prompt_len, cache.live_len), (0, None));
        assert!(!cache.published);
        assert_eq!(cache.capture_at, None);
        assert!(cache.generation > before);
    }

    // ---- radix store: pure decisions ----

    use hipfire_runtime::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity, TokenizerIdentity,
    };

    const MIB: usize = 1 << 20;

    fn base_domain() -> CacheDomain {
        CacheDomain {
            model_content_digest: vec![1],
            model_load_epoch: 1,
            sidecar_digests: vec![],
            tokenizer: TokenizerIdentity {
                vocab_digest: vec![2],
                config_digest: vec![3],
            },
            template: TemplateIdentity {
                template_digest: vec![4],
                normalization_tag: "n".to_string(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "qwen4".to_string(),
                state_abi_tag: "abi".to_string(),
                position_attention_tag: "p".to_string(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![],
                v_stride_bytes: vec![],
                layout_tag: "l".to_string(),
            },
            device: DeviceTopology {
                device_id: "d".to_string(),
                topology_id: "t".to_string(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "ns".to_string(),
            },
        }
    }

    #[test]
    fn live_beats_equal_or_shallower_radix_and_deeper_radix_beats_live() {
        use Qwen4PrefixSource as S;
        assert_eq!(pick_source(S::Live, 100, Some(100)), S::Live);
        assert_eq!(pick_source(S::Live, 100, Some(60)), S::Live);
        assert_eq!(pick_source(S::Live, 100, None), S::Live);
        assert_eq!(pick_source(S::Live, 100, Some(101)), S::Radix);
        assert_eq!(pick_source(S::Live, 100, Some(4096)), S::Radix);
    }

    #[test]
    fn prompt_wins_ties_and_the_greater_boundary_wins_otherwise() {
        use Qwen4PrefixSource as S;
        assert_eq!(pick_source(S::Prompt, 80, Some(80)), S::Prompt);
        assert_eq!(pick_source(S::Prompt, 80, Some(40)), S::Prompt);
        assert_eq!(pick_source(S::Prompt, 80, None), S::Prompt);
        assert_eq!(pick_source(S::Prompt, 80, Some(81)), S::Radix);
        assert_eq!(pick_source(S::Cold, 0, Some(1)), S::Radix);
        assert_eq!(pick_source(S::Cold, 0, None), S::Cold);
    }

    /// A scripted index: `(id, boundary)` entries, the valid ids, and a log.
    struct Script {
        entries: Vec<(u32, usize)>,
        valid: Vec<u32>,
        discarded: Vec<u32>,
        befores: Vec<usize>,
    }

    fn scripted(script: &mut Script, prompt_len: usize) -> Option<(u32, usize)> {
        lookup_valid_checkpoint(
            script,
            prompt_len,
            |s: &mut Script, before: usize| {
                s.befores.push(before);
                s.entries
                    .iter()
                    .filter(|(_, b)| *b < before)
                    .max_by_key(|(_, b)| *b)
                    .copied()
            },
            |s: &Script, id: &u32, _b: usize| s.valid.contains(id),
            |s: &mut Script, id: u32| s.discarded.push(id),
        )
    }

    #[test]
    fn stale_checkpoint_ids_retry_below_their_boundary() {
        let mut script = Script {
            entries: vec![(1, 256), (2, 1024), (3, 2048)],
            valid: vec![1],
            discarded: vec![],
            befores: vec![],
        };
        assert_eq!(scripted(&mut script, 4096), Some((1, 256)));
        assert_eq!(script.befores, vec![4096, 2048, 1024]);
        assert_eq!(script.discarded, vec![3, 2]);

        // A valid deepest hit is taken at once.
        let mut direct = Script {
            entries: vec![(1, 256), (2, 1024)],
            valid: vec![1, 2],
            discarded: vec![],
            befores: vec![],
        };
        assert_eq!(scripted(&mut direct, 4096), Some((2, 1024)));
        assert!(direct.discarded.is_empty());

        // Nothing below the prompt: a miss.
        let mut miss = Script {
            entries: vec![(1, 256)],
            valid: vec![1],
            discarded: vec![],
            befores: vec![],
        };
        assert_eq!(scripted(&mut miss, 256), None);
        assert_eq!(scripted(&mut miss, 0), None);
    }

    #[test]
    fn stale_retries_are_bounded_and_release_every_discarded_pin() {
        let entries: Vec<(u32, usize)> = (1..=20).map(|i| (i, i as usize * 128)).collect();
        let mut script = Script {
            entries,
            valid: vec![],
            discarded: vec![],
            befores: vec![],
        };
        assert_eq!(scripted(&mut script, 10_000), None);
        assert_eq!(script.discarded.len(), RADIX_LOOKUP_RETRIES);
        assert_eq!(script.befores.len(), RADIX_LOOKUP_RETRIES);
    }

    #[test]
    fn anchor_arms_only_on_the_second_observation() {
        let covered = |s: usize| s <= 1000;
        // The index already covers the first eligible boundary: arm it.
        assert_eq!(
            anchor_choice(&[64, 300, 900], 2000, 0, covered, None, true),
            Some(300)
        );
        // First-ever cold request: nothing in the index covers S.
        assert_eq!(
            anchor_choice(&[64, 300, 900], 2000, 0, |_| false, None, true),
            None
        );
        // Only the first eligible boundary is considered.
        assert_eq!(
            anchor_choice(&[300, 900], 2000, 0, |s| s == 900, None, true),
            None
        );
    }

    #[test]
    fn anchor_needs_a_free_boundary_a_deeper_start_and_resources() {
        let covered = |_: usize| true;
        // A checkpoint already exists at or beyond S.
        assert_eq!(anchor_choice(&[300], 2000, 0, covered, Some(300), true), None);
        assert_eq!(anchor_choice(&[300], 2000, 0, covered, Some(512), true), None);
        assert_eq!(
            anchor_choice(&[300], 2000, 0, covered, Some(299), true),
            Some(300)
        );
        // S must lie beyond the chosen start and inside the prompt.
        assert_eq!(
            anchor_choice(&[300, 900], 2000, 300, covered, None, true),
            Some(900)
        );
        assert_eq!(anchor_choice(&[1900], 1900, 0, covered, None, true), None);
        assert_eq!(anchor_choice(&[127], 2000, 0, covered, None, true), None);
        assert_eq!(anchor_choice(&[], 2000, 0, covered, None, true), None);
        // Budget refusal.
        assert_eq!(anchor_choice(&[300], 2000, 0, covered, None, false), None);
    }

    fn schedule() -> CaptureSchedule {
        CaptureSchedule {
            capture_at: Some(20_000),
            anchor: None,
            radix: true,
            periodic_done: false,
            periodic_due_from: Some(0),
            resources_ok: true,
        }
    }

    #[test]
    fn end_of_prompt_is_always_due_and_nothing_is_without_an_armed_prompt() {
        let eop = CaptureSchedule {
            resources_ok: false,
            radix: false,
            ..schedule()
        };
        assert_eq!(eop.next(19_000, 20_000), Some(20_000));
        assert_eq!(eop.next(19_000, 19_500), None);
        let unarmed = CaptureSchedule {
            capture_at: None,
            ..schedule()
        };
        assert_eq!(unarmed.next(19_000, 20_000), None);
        assert_eq!(schedule().next(20_000, 20_000), None);
    }

    #[test]
    fn periodic_is_due_only_after_8192_tokens_and_at_most_once() {
        let s = schedule();
        assert_eq!(s.next(7_680, 7_936), None);
        assert_eq!(s.next(7_936, 8_192), Some(8_192));
        // Measured from the last capture, not the prompt start.
        let later = CaptureSchedule {
            periodic_due_from: Some(8_192),
            ..s
        };
        assert_eq!(later.next(16_000, 16_383), None);
        assert_eq!(later.next(16_128, 16_384), Some(16_384));
        // Once taken (or refused) it never repeats in the request.
        let done = CaptureSchedule {
            periodic_done: true,
            ..s
        };
        assert_eq!(done.next(7_936, 8_192), None);
        // At the prompt end the boundary is the end-of-prompt capture.
        let tail = CaptureSchedule {
            capture_at: Some(8_192),
            ..s
        };
        assert_eq!(tail.next(7_936, 8_192), Some(8_192));
        let unseeded = CaptureSchedule {
            periodic_due_from: None,
            ..s
        };
        assert_eq!(unseeded.next(7_936, 8_192), None);
    }

    #[test]
    fn budget_refusal_skips_optional_captures_but_not_the_end_of_prompt() {
        let refused = CaptureSchedule {
            resources_ok: false,
            anchor: None,
            ..schedule()
        };
        assert_eq!(refused.next(7_936, 8_192), None);
        assert_eq!(refused.next(19_000, 20_000), Some(20_000));
        let no_radix = CaptureSchedule {
            radix: false,
            ..schedule()
        };
        assert_eq!(no_radix.next(7_936, 8_192), None);
    }

    #[test]
    fn anchor_splits_only_a_chunk_that_contains_it() {
        let s = CaptureSchedule {
            anchor: Some(5_000),
            ..schedule()
        };
        assert_eq!(s.next(4_096, 4_352), None);
        assert_eq!(s.next(4_864, 5_120), Some(5_000));
        // The chunk ending exactly at S needs no split.
        assert_eq!(s.next(4_864, 5_000), Some(5_000));
        // Already at S: the anchor is behind this chunk.
        assert_eq!(s.next(5_000, 5_256), None);
        // The anchor beats the end of prompt inside one chunk.
        let short = CaptureSchedule {
            capture_at: Some(5_100),
            ..s
        };
        assert_eq!(short.next(4_864, 5_100), Some(5_000));
        assert_eq!(short.next(5_000, 5_100), Some(5_100));
    }

    #[test]
    fn capture_resources_need_a_slot_a_budgeted_allocation_or_an_evictable_entry() {
        // A free slot.
        assert!(capture_resources_ok(1, 4, 4, 0, u64::MAX, 10, 0));
        // A new slot within count and bytes.
        assert!(capture_resources_ok(0, 2, 4, 0, 100, 50, 200));
        assert!(!capture_resources_ok(0, 2, 4, 0, 160, 50, 200));
        assert!(!capture_resources_ok(0, 4, 4, 0, 0, 50, 200));
        // An evictable pool entry.
        assert!(capture_resources_ok(0, 4, 4, 1, 0, 50, 0));
    }

    #[test]
    fn checkpoint_validity_requires_exact_tokens_mode_chunk_context_and_head() {
        let tokens = prompt(8);
        let long_prompt = prompt(12);
        let facts = |mode, chunk, ctx, target, head| CheckpointFacts {
            tokens: &tokens,
            mode,
            chunk,
            has_context: ctx,
            target_position: target,
            head_position: head,
        };
        let ok = facts(AR, CHUNK, true, Some(8), None);
        assert!(checkpoint_facts_valid(&ok, &long_prompt, 8, AR, CHUNK));
        // B must be strictly below the prompt end.
        assert!(!checkpoint_facts_valid(&ok, &tokens, 8, AR, CHUNK));
        // Token mismatch, mode, chunk, context, position.
        let mut other = long_prompt.clone();
        other[3] = 999;
        assert!(!checkpoint_facts_valid(&ok, &other, 8, AR, CHUNK));
        assert!(!checkpoint_facts_valid(&ok, &long_prompt, 8, MTP, CHUNK));
        assert!(!checkpoint_facts_valid(&ok, &long_prompt, 8, AR, CHUNK / 2));
        assert!(!checkpoint_facts_valid(
            &facts(AR, CHUNK, false, Some(8), None),
            &long_prompt,
            8,
            AR,
            CHUNK
        ));
        assert!(!checkpoint_facts_valid(
            &facts(AR, CHUNK, true, Some(7), None),
            &long_prompt,
            8,
            AR,
            CHUNK
        ));
        assert!(!checkpoint_facts_valid(&ok, &long_prompt, 7, AR, CHUNK));
        // Native MTP needs the head at B as well.
        assert!(!checkpoint_facts_valid(
            &facts(MTP, CHUNK, true, Some(8), None),
            &long_prompt,
            8,
            MTP,
            CHUNK
        ));
        assert!(checkpoint_facts_valid(
            &facts(MTP, CHUNK, true, Some(8), Some(8)),
            &long_prompt,
            8,
            MTP,
            CHUNK
        ));
        assert!(!checkpoint_facts_valid(
            &facts(MTP, CHUNK, true, Some(8), Some(7)),
            &long_prompt,
            8,
            MTP,
            CHUNK
        ));
    }

    fn granule(id: u64) -> VmmPhysicalId {
        VmmPhysicalId { id, generation: 1 }
    }

    #[test]
    fn ledger_counts_shared_granules_once_and_skips_live_ones() {
        let a = vec![(granule(1), 2 * MIB), (granule(2), 2 * MIB)];
        let b = vec![(granule(2), 2 * MIB), (granule(3), 2 * MIB)];
        let contexts = || a.iter().copied().chain(b.iter().copied());
        // Granule 2 is shared by both checkpoints: one charge. Granule 1 is
        // still mapped by the live bank: baseline-charged.
        let live: HashSet<VmmPhysicalId> = [granule(1)].into_iter().collect();
        assert_eq!(cache_granule_bytes(contexts(), &live), 4 * MIB as u64);
        // No live granules: all three unique ones.
        assert_eq!(
            cache_granule_bytes(contexts(), &HashSet::new()),
            6 * MIB as u64
        );
    }

    #[test]
    fn ledger_charges_retained_granules_after_the_live_bank_leaves_them() {
        let retained = vec![(granule(7), 2 * MIB), (granule(8), 2 * MIB)];
        let live_before: HashSet<VmmPhysicalId> =
            [granule(7), granule(8), granule(9)].into_iter().collect();
        let before = cache_granule_bytes(retained.iter().copied(), &live_before);
        assert_eq!(before, 0);
        // The bank swapped away: the still-referenced granules are now
        // charged to the cache.
        let live_after: HashSet<VmmPhysicalId> = [granule(10)].into_iter().collect();
        let after = cache_granule_bytes(retained.iter().copied(), &live_after);
        assert_eq!(after, 4 * MIB as u64);
        assert_eq!(ledger_total(33, 5, after), 38 + 4 * MIB as u64);
        // Saturates instead of wrapping.
        assert_eq!(ledger_total(u64::MAX, 1, 1), u64::MAX);
    }

    #[test]
    fn budget_caps_trigger_eviction_and_pick_the_victim_kind() {
        assert!(!over_budget(false, 4, 4, 100, 100));
        assert!(over_budget(true, 0, 4, 0, 100));
        assert!(over_budget(false, 5, 4, 0, 100));
        assert!(over_budget(false, 0, 4, 101, 100));
        assert_eq!(evict_mode(true, false), EvictMode::Checkpoint);
        assert_eq!(evict_mode(false, true), EvictMode::Checkpoint);
        assert_eq!(evict_mode(false, false), EvictMode::PruneFirst);
    }

    #[test]
    fn slab_estimate_aliases_whole_granules_below_the_writable_end() {
        let granules = [2 * MIB; 3];
        // 5 MiB valid, all writable-safe: two granules aliased, 1 MiB copied.
        assert_eq!(slab_estimate(5 * MIB, 5 * MIB, granules), MIB as u64);
        // The in-place restore tail starts at 3 MiB: only one granule aliased.
        assert_eq!(slab_estimate(5 * MIB, 3 * MIB, granules), 3 * MIB as u64);
        // Shorter than one granule: nothing aliases.
        assert_eq!(slab_estimate(MIB, MIB, granules), MIB as u64);
        assert_eq!(slab_estimate(0, 0, granules), 0);
        // Entries are padded to the slab alignment.
        assert_eq!(slab_estimate(1000, 1000, granules), 1024);
        // No granule segments at all: the whole valid prefix is copied.
        assert_eq!(slab_estimate(4096, 4096, Vec::<usize>::new()), 4096);
        assert_eq!(alias_boundary([2 * MIB, 2 * MIB], 3 * MIB), 2 * MIB);
        assert_eq!(alias_boundary([2 * MIB, 0, 2 * MIB], 8 * MIB), 2 * MIB);
    }

    #[test]
    fn gdsf_cost_grows_superlinearly_and_priority_follows_the_formula() {
        assert!(gdsf_cost(2048) > 2048.0);
        assert_eq!(gdsf_cost(65_536), 131_072.0);
        assert!(gdsf_cost(32_768) / 32_768.0 > gdsf_cost(1024) / 1024.0);
        assert_eq!(gdsf_priority(2.0, 100.0, 1, 50), 6.0);
        assert_eq!(gdsf_priority(0.0, 10.0, 0, 0), 10.0);
        // Hits and cost raise it, bytes lower it.
        assert!(gdsf_priority(0.0, 10.0, 3, 100) > gdsf_priority(0.0, 10.0, 0, 100));
        assert!(gdsf_priority(0.0, 10.0, 0, 200) < gdsf_priority(0.0, 10.0, 0, 100));
    }

    #[test]
    fn gdsf_victim_is_the_lowest_unpinned_priority_with_ties_to_the_lowest_id() {
        let id = CheckpointId;
        assert_eq!(gdsf_victim(&[]), None);
        assert_eq!(
            gdsf_victim(&[(id(5), 2.0, false), (id(3), 1.0, false), (id(4), 9.0, false)]),
            Some(id(3))
        );
        assert_eq!(
            gdsf_victim(&[(id(9), 1.0, false), (id(4), 1.0, false), (id(7), 1.0, false)]),
            Some(id(4))
        );
        // A pinned entry is never chosen, however low its priority.
        assert_eq!(
            gdsf_victim(&[(id(1), 0.0, true), (id(2), 5.0, false)]),
            Some(id(2))
        );
        assert_eq!(gdsf_victim(&[(id(1), 0.0, true), (id(2), 5.0, true)]), None);
    }

    struct SimEntry {
        id: u64,
        priority: f64,
    }

    /// Insert a checkpoint of boundary `b` and evict GDSF victims down to `cap`.
    fn sim_insert(
        entries: &mut Vec<SimEntry>,
        clock: &mut f64,
        cap: usize,
        (id, b, hits): (u64, usize, u32),
        bytes: u64,
    ) -> Vec<u64> {
        entries.push(SimEntry {
            id,
            priority: gdsf_priority(*clock, gdsf_cost(b), hits, bytes),
        });
        let mut evicted = Vec::new();
        while entries.len() > cap {
            let view: Vec<(CheckpointId, f64, bool)> = entries
                .iter()
                .map(|entry| (CheckpointId(entry.id), entry.priority, false))
                .collect();
            let victim = gdsf_victim(&view).expect("an unpinned victim");
            let at = entries
                .iter()
                .position(|entry| CheckpointId(entry.id) == victim)
                .expect("the victim is retained");
            let gone = entries.remove(at);
            *clock = gone.priority;
            evicted.push(gone.id);
        }
        evicted
    }

    #[test]
    fn a_hot_long_checkpoint_survives_a_burst_of_one_off_short_ones() {
        let bytes = 40 * MIB as u64;
        let mut entries = Vec::new();
        let mut clock = 0.0;
        sim_insert(&mut entries, &mut clock, 3, (1, 32_768, 3), bytes);
        for id in 2..=40 {
            let evicted = sim_insert(&mut entries, &mut clock, 3, (id, 512, 0), bytes);
            assert!(!evicted.contains(&1), "long hot checkpoint evicted at {id}");
        }
        assert!(entries.iter().any(|entry| entry.id == 1));
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn aging_lets_an_old_hot_entry_eventually_go() {
        let bytes = 40 * MIB as u64;
        let mut entries = Vec::new();
        let mut clock = 0.0;
        sim_insert(&mut entries, &mut clock, 2, (1, 2048, 5), bytes);
        let mut gone_at = None;
        for id in 2..=200 {
            let evicted = sim_insert(&mut entries, &mut clock, 2, (id, 2048, 0), bytes);
            if evicted.contains(&1) {
                gone_at = Some(id);
                break;
            }
        }
        let gone_at = gone_at.expect("the aging clock must eventually pass a stale hot entry");
        assert!(gone_at > 2, "evicted immediately at {gone_at}");
        assert!(clock > 0.0);
    }

    #[test]
    fn radix_domains_bind_mode_chunk_formats_and_strides() {
        let base = base_domain();
        let strides = vec![52u64, 52, 128, 32];
        let ar = build_radix_domain(&base, AR, 256, "gdn", "qsa", &strides);
        let again = build_radix_domain(&base, AR, 256, "gdn", "qsa", &strides);
        assert_eq!(ar, again);
        assert!(ar.arch_policy.state_abi_tag.starts_with("qwen4-radix-v1:"));
        assert_eq!(
            ar.arch_policy.state_abi_tag.len(),
            "qwen4-radix-v1:".len() + 64
        );
        assert_eq!(ar.kv_layout.k_stride_bytes, strides);
        assert_eq!(ar.kv_layout.v_stride_bytes, strides);
        // Everything but the tag and strides is the base domain's.
        assert_eq!(ar.model_content_digest, base.model_content_digest);
        assert_eq!(ar.namespace, base.namespace);
        assert_eq!(ar.device, base.device);
        for other in [
            build_radix_domain(&base, MTP, 256, "gdn", "qsa", &strides),
            build_radix_domain(&base, AR, 512, "gdn", "qsa", &strides),
            build_radix_domain(&base, AR, 256, "gdn-q8", "qsa", &strides),
            build_radix_domain(&base, AR, 256, "gdn", "qsa-fp8", &strides),
        ] {
            assert_ne!(other.arch_policy.state_abi_tag, ar.arch_policy.state_abi_tag);
        }
        let mut changed_base = base.clone();
        changed_base.arch_policy.state_abi_tag = "abi2".to_string();
        assert_ne!(
            build_radix_domain(&changed_base, AR, 256, "gdn", "qsa", &strides)
                .arch_policy
                .state_abi_tag,
            ar.arch_policy.state_abi_tag
        );
    }

    #[test]
    fn turn_boundaries_are_sorted_deduplicated_and_bounded() {
        assert_eq!(
            normalized_turn_boundaries(&[300, 128, 300, 9000, 64], 8192),
            vec![64, 128, 300]
        );
        assert!(normalized_turn_boundaries(&[], 8192).is_empty());
        assert!(normalized_turn_boundaries(&[8192], 8192).is_empty());
    }

    #[test]
    fn logical_page_strides_sum_k_and_v_sides_per_position() {
        // One layer: K 100, V 200, raw 50, pooled 40 per 4 tokens.
        assert_eq!(logical_page_strides(&[100, 200, 50, 40], 4), (150, 210));
        assert_eq!(logical_page_strides(&[], 4), (1, 1));
    }

    #[test]
    fn a_cache_without_radix_schedules_only_the_end_of_prompt() {
        let mut cache = Qwen4PrefixCache::new(64);
        assert!(!cache.capture_resources_available());
        cache.capture_at = Some(1000);
        cache.anchor = None;
        cache.periodic_due_from = Some(0);
        let schedule = cache.capture_schedule();
        assert!(!schedule.radix);
        assert_eq!(schedule.next(0, 256), None);
        assert_eq!(schedule.next(768, 1000), Some(1000));
        assert_eq!(cache.eop_facts().map(|(_, b, _)| b), None);
    }

    #[test]
    fn clear_drops_the_anchor_and_periodic_state() {
        let mut cache = Qwen4PrefixCache::new(64);
        cache.anchor = Some(300);
        cache.periodic_due_from = Some(10);
        cache.periodic_done = true;
        cache.clear();
        assert_eq!(cache.anchor, None);
        assert_eq!(cache.periodic_due_from, None);
        assert!(!cache.periodic_done);
    }

    #[test]
    fn selection_receipts_carry_start_source_and_generation() {
        let selection = Qwen4Selection {
            id: CheckpointId(3),
            start: 4096,
            mode: MTP,
            source: Qwen4PrefixSource::Radix,
            pin: PinTicket::none(),
            generation: 11,
        };
        let plan = selection.plan();
        assert_eq!(plan.start_pos, 4096);
        assert_eq!(plan.source(), Qwen4PrefixSource::Radix);
        assert_eq!(plan.generation, 11);
        // Releasing pins without a radix store is a no-op.
        let mut radix: Option<Qwen4RadixCache> = None;
        Qwen4PrefixCache::release_pins(&mut radix, selection);
        let mut cache = Qwen4PrefixCache::new(8);
        cache.release_pending();
        cache.release_held();
    }

    #[test]
    fn begin_validation_keeps_live_and_cold_and_refuses_unselected_radix() {
        let (cache, consumed) = committed(8, 5, AR);
        let m = marks(AR, 13, Some(8));
        let mut next = consumed.clone();
        next.push(42);
        let live = cache.bind(&m, &next, 13, AR).unwrap();
        assert_eq!(
            cache.validate_plan(&m, &next, live, AR, None).unwrap(),
            BeginAction::Live
        );
        let cold = cache.bind(&m, &next, 0, AR).unwrap();
        assert_eq!(
            cache.validate_plan(&m, &next, cold, AR, None).unwrap(),
            BeginAction::Cold
        );
        // A checkpoint restore needs the checkpoint itself, not just marks.
        let mut diverged = consumed[..8].to_vec();
        diverged.extend([500, 501]);
        let prompt_plan = cache.bind(&m, &diverged, 8, AR).unwrap();
        assert!(cache
            .validate_plan(&m, &diverged, prompt_plan, AR, None)
            .is_err());
        // A radix plan without its selection is refused.
        let radix_plan = Qwen4PrefixPlan {
            start_pos: 8,
            source: Qwen4PrefixSource::Radix,
            generation: cache.generation,
        };
        assert!(cache
            .validate_plan(&m, &diverged, radix_plan, AR, None)
            .is_err());
        // A stale live receipt is refused too.
        let mut stale = cache.bind(&m, &next, 13, AR).unwrap();
        stale.generation += 1;
        assert!(cache.validate_plan(&m, &next, stale, AR, None).is_err());
    }
}
