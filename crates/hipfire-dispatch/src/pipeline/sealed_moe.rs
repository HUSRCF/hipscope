// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Checked, launch-free boundary for routed MoE execution.
//!
//! This module deliberately does not own a GPU allocation or issue a launch.
//! The model/loader supplies CPU metadata through [`ExpertTable`], binds that
//! metadata once, and then seals the existing family parameter records.  The
//! parameter records remain the sole source of tensor borrows; this layer only
//! proves that they agree with the immutable expert contract before a future
//! executor is allowed to consume them.

use crate::context::DispatchCtx;
use crate::families::moe::{MoeParams, MoePrefillParams, RoutedExpertWeights};
use crate::types::{dtype_rotation_plan, DispatchError, RotationPlan};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

const DEVICE_POINTER_BYTES: usize = 8;
/// Grouped-GEMM rows are emitted in fixed-width tiles. Keep this tied to the
/// dispatch implementation in `pipeline::mod` rather than letting callers
/// choose a second alignment policy.
const GROUPED_BLOCK_M: usize = super::MOE_GROUPED_BLOCK_M;
static NEXT_INVOCATION: AtomicU64 = AtomicU64::new(1);
static NEXT_TABLE_ID: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "serve-fault-inject")]
thread_local! {
    static FAULT_AFTER_EXPERT_MUTATION_ARMED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(feature = "serve-fault-inject")]
pub fn arm_fault_after_expert_mutation(armed: bool) {
    FAULT_AFTER_EXPERT_MUTATION_ARMED.with(|cell| cell.set(armed));
}

#[cfg(feature = "serve-fault-inject")]
pub fn take_fault_after_expert_mutation() -> bool {
    FAULT_AFTER_EXPERT_MUTATION_ARMED.with(|cell| {
        let armed = cell.get();
        cell.set(false);
        armed
    })
}

/// Checked upper bound for padded grouped rows. Every active expert can add
/// at most `block - 1` padding rows; the result is rounded up to a complete
/// tile because grouped kernels index one tile id per block.
pub fn checked_grouped_m_total_bound(
    total_slots: usize,
    n_experts: usize,
    block: usize,
) -> Result<usize, DispatchError> {
    if total_slots == 0 || n_experts == 0 || block == 0 {
        return Err(invalid(
            "grouped row bound requires nonzero slots, experts, and block",
        ));
    }
    let active = total_slots.min(n_experts);
    let padding = active
        .checked_mul(block - 1)
        .ok_or_else(|| invalid("grouped row padding overflows"))?;
    let padded_rows = total_slots
        .checked_add(padding)
        .ok_or_else(|| invalid("grouped row bound overflows"))?;
    let remainder = padded_rows % block;
    if remainder == 0 {
        return Ok(padded_rows);
    }
    padded_rows
        .checked_add(block - remainder)
        .ok_or_else(|| invalid("grouped row alignment overflows"))
}

fn validate_grouped_m_total_max(
    total_slots: usize,
    n_experts: usize,
    m_total_max: usize,
) -> Result<(), DispatchError> {
    let required = checked_grouped_m_total_bound(total_slots, n_experts, GROUPED_BLOCK_M)?;
    if m_total_max % GROUPED_BLOCK_M != 0 {
        return Err(invalid(format!(
            "prefill m_total_max={} is not aligned to grouped block {}",
            m_total_max, GROUPED_BLOCK_M
        )));
    }
    if m_total_max < required {
        return Err(invalid(format!(
            "prefill m_total_max={} is below padded grouped-row bound {required} \
             (slots={total_slots}, active_experts={})",
            m_total_max,
            total_slots.min(n_experts)
        )));
    }
    Ok(())
}

/// Derive the only legal local projection dimensions for a TP shard.
pub fn checked_tp_local_shapes(
    hidden: usize,
    intermediate: usize,
    tp_size: usize,
) -> Result<(usize, usize, usize, usize), DispatchError> {
    if hidden == 0 || intermediate == 0 || tp_size == 0 {
        return Err(invalid("TP dimensions must be nonzero"));
    }
    if intermediate % tp_size != 0 {
        return Err(invalid(
            "intermediate dimension is not divisible by TP size",
        ));
    }
    let local_intermediate = intermediate / tp_size;
    let gate_up_rows = local_intermediate
        .checked_mul(2)
        .ok_or_else(|| invalid("gate/up local dimension overflows"))?;
    Ok((gate_up_rows, hidden, hidden, local_intermediate))
}

/// Complete routed execution grammar selected by the sealed call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoeProtocol {
    /// One token, indexed expert gate/up and expanded down.
    IndexedDecode,
    /// Batched tokens, scatter/grouped projections and unscatter.
    GroupedPrefill,
}

/// The producer that owns the route buffers used by a sealed call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoeRouterInput {
    /// The sealed decode call owns softmax and top-k production.
    SoftmaxTopK,
    /// A softmax/top-k producer already populated the route buffers.
    PrecomputedSoftmaxTopK,
    /// A sigmoid/top-k producer already populated the route buffers.
    PrecomputedSigmoidTopK,
}

/// Where the routed contribution is accumulated before the grammar completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoeContribution {
    /// Combine directly into the residual stream.
    Residual,
    /// Combine into a zeroed rank-local partial for one authorized reduction.
    ZeroedPartial,
}

/// Where the replicated/shared contribution is accumulated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MoeSharedContribution {
    /// No shared expert participates in the call.
    None,
    /// Shared output stays outside the routed reduction on every rank.
    PerRankResidual,
    /// One designated rank contributes the replicated shared output to a partial.
    RootPartial,
}

/// Whether the routed activation is natural or already transformed into the
/// basis required by the expert source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActivationInput {
    Natural,
    PreRotated,
}

/// Activation identity carried by the sealed signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActivationIdentity {
    input: ActivationInput,
    rotation: RotationPlan,
}

impl ActivationIdentity {
    /// Create an activation identity.  `rotation` is the source basis, not a
    /// caller-selected kernel variant.
    pub fn new(input: ActivationInput, rotation: RotationPlan) -> Self {
        Self { input, rotation }
    }

    pub fn input(self) -> ActivationInput {
        self.input
    }

    pub fn rotation(self) -> RotationPlan {
        self.rotation
    }
}

/// A read-only alias into another declared source allocation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceAlias {
    owner: String,
    byte_offset: usize,
    byte_len: usize,
}

impl ResourceAlias {
    pub fn new(
        owner: impl Into<String>,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Self, DispatchError> {
        let owner = owner.into();
        if owner.is_empty() {
            return Err(invalid("alias owner is empty"));
        }
        if byte_len == 0 {
            return Err(invalid("alias byte length is zero"));
        }
        byte_offset
            .checked_add(byte_len)
            .ok_or_else(|| invalid("alias byte range overflows"))?;
        Ok(Self {
            owner,
            byte_offset,
            byte_len,
        })
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }
}

/// CPU-only representation metadata for one projection source.
///
/// The fields are private on purpose: a source can enter an execution plan
/// only through the checked constructors below.  `shape` is the logical
/// (rows, columns) shape, while `encoded_bytes` and `row_stride` describe the
/// actual encoded representation rather than an inferred element count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertResource {
    source_name: String,
    source_fingerprint: String,
    shape: Vec<usize>,
    dtype: DType,
    encoded_bytes: usize,
    row_stride: usize,
    alignment: usize,
    basis: RotationPlan,
    sidecars: Vec<String>,
    alias: Option<ResourceAlias>,
}

impl ExpertResource {
    /// Construct a source with its loader-provided identity and representation.
    pub fn new(
        source_name: impl Into<String>,
        source_fingerprint: impl Into<String>,
        shape: Vec<usize>,
        dtype: DType,
        encoded_bytes: usize,
        row_stride: usize,
        alignment: usize,
        basis: RotationPlan,
    ) -> Result<Self, DispatchError> {
        let source_name = source_name.into();
        let source_fingerprint = source_fingerprint.into();
        validate_resource_header(
            &source_name,
            &source_fingerprint,
            &shape,
            dtype,
            encoded_bytes,
            row_stride,
            alignment,
            basis,
        )?;
        Ok(Self {
            source_name,
            source_fingerprint,
            shape,
            dtype,
            encoded_bytes,
            row_stride,
            alignment,
            basis,
            sidecars: Vec::new(),
            alias: None,
        })
    }

    /// Add the declared read-only sidecar identities.
    pub fn with_sidecars(mut self, sidecars: Vec<String>) -> Result<Self, DispatchError> {
        validate_sidecars(&sidecars)?;
        self.sidecars = sidecars;
        Ok(self)
    }

    /// Mark this source as a read-only range alias of another declared source.
    pub fn with_alias(mut self, alias: ResourceAlias) -> Result<Self, DispatchError> {
        self.alias = Some(alias);
        Ok(self)
    }

    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    pub fn source_fingerprint(&self) -> &str {
        &self.source_fingerprint
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn encoded_bytes(&self) -> usize {
        self.encoded_bytes
    }

    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn basis(&self) -> RotationPlan {
        self.basis
    }

    pub fn sidecars(&self) -> &[String] {
        &self.sidecars
    }

    pub fn alias(&self) -> Option<&ResourceAlias> {
        self.alias.as_ref()
    }
}

/// The three logical projection records for an expert.  A source may be
/// represented as one fused gate/up record or as separate gate and up records,
/// but never both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertResources {
    gate_up: Option<ExpertResource>,
    gate: Option<ExpertResource>,
    up: Option<ExpertResource>,
    down: ExpertResource,
}

impl ExpertResources {
    /// Construct the common fused gate/up representation.
    pub fn new(gate_up: ExpertResource, down: ExpertResource) -> Result<Self, DispatchError> {
        Self::fused(gate_up, down)
    }

    pub fn fused(gate_up: ExpertResource, down: ExpertResource) -> Result<Self, DispatchError> {
        if gate_up.source_name() == down.source_name() {
            return Err(invalid("gate/up and down sources must be distinct"));
        }
        Ok(Self {
            gate_up: Some(gate_up),
            gate: None,
            up: None,
            down,
        })
    }

    /// Construct separate gate, up, and down projection records.
    pub fn separate(
        gate: ExpertResource,
        up: ExpertResource,
        down: ExpertResource,
    ) -> Result<Self, DispatchError> {
        if gate.source_name() == up.source_name()
            || gate.source_name() == down.source_name()
            || up.source_name() == down.source_name()
        {
            return Err(invalid(
                "separate expert projection sources must be distinct",
            ));
        }
        Ok(Self {
            gate_up: None,
            gate: Some(gate),
            up: Some(up),
            down,
        })
    }

    pub fn is_fused(&self) -> bool {
        self.gate_up.is_some()
    }

    pub fn gate_up(&self) -> Option<&ExpertResource> {
        self.gate_up.as_ref()
    }

    pub fn gate(&self) -> Option<&ExpertResource> {
        self.gate.as_ref()
    }

    pub fn up(&self) -> Option<&ExpertResource> {
        self.up.as_ref()
    }

    pub fn down(&self) -> &ExpertResource {
        &self.down
    }

    fn iter(&self) -> impl Iterator<Item = &ExpertResource> {
        self.gate_up
            .iter()
            .chain(self.gate.iter())
            .chain(self.up.iter())
            .chain(std::iter::once(&self.down))
    }
}

/// One global expert and its compact local rank slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertMetadata {
    global_id: usize,
    owner_rank: usize,
    local_slot: usize,
    resources: ExpertResources,
}

impl ExpertMetadata {
    pub fn new(
        global_id: usize,
        owner_rank: usize,
        local_slot: usize,
        resources: ExpertResources,
    ) -> Result<Self, DispatchError> {
        let mut basis = None;
        for resource in resources.iter() {
            validate_resource_header(
                resource.source_name(),
                resource.source_fingerprint(),
                resource.shape(),
                resource.dtype(),
                resource.encoded_bytes(),
                resource.row_stride(),
                resource.alignment(),
                resource.basis(),
            )?;
            validate_sidecars(resource.sidecars())?;
            if let Some(expected) = basis {
                if expected != resource.basis() {
                    return Err(invalid(format!(
                        "expert {global_id} projection sources use different rotation bases"
                    )));
                }
            } else {
                basis = Some(resource.basis());
            }
        }
        Ok(Self {
            global_id,
            owner_rank,
            local_slot,
            resources,
        })
    }

    pub fn global_id(&self) -> usize {
        self.global_id
    }

    pub fn owner_rank(&self) -> usize {
        self.owner_rank
    }

    pub fn local_slot(&self) -> usize {
        self.local_slot
    }

    pub fn resources(&self) -> &ExpertResources {
        &self.resources
    }
}

/// Canonical execution string for expert-parallel decode whose root combine
/// folds selected slots in slot order. Only an EP plan carrying this execution
/// string together with its EP collective schedule may authorize the sealed
/// canonical slot combine owned by a later slice.
pub const CANONICAL_EP_EXECUTION: &str = "indexed-decode-slot-order";

/// Dispatch-owned mirror of the runtime parallelism axis. This crate never
/// depends on the runtime planner; the runtime maps its own enum onto this
/// one when adapting a sealed plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContractParallelism {
    Single,
    TensorParallel,
    ExpertParallel,
}

/// Dispatch-owned mirror of the runtime expert assignment policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContractAssignment {
    Stride,
    Contiguous,
}

/// Dispatch-owned mirror of the mesh axis named by a collective row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContractAxis {
    Pp,
    Tp,
    Ep,
}

/// Dispatch-owned mirror of one collective schedule row hint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContractCollectiveHint {
    AllReduce { kind: ContractAxis },
    BandXfer { src: usize, dst: usize },
}

/// One ordered collective row carried by the execution contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractCollectiveRow {
    pub name: String,
    pub layer: usize,
    pub hint: ContractCollectiveHint,
}

/// Deterministic execution contract adapted from one sealed runtime plan.
///
/// The contract is pure CPU metadata: group/layer identity, source
/// fingerprint, mesh epoch, the physical rank list, parallelism, assignment,
/// the per-global-expert owner/local-slot map, the execution string, and the
/// ordered collective rows. It contains no pointer, so two ranks (or two
/// processes) holding the same sealed plan render the same fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertExecutionContract {
    group: String,
    layer: Option<usize>,
    source_fingerprint: String,
    mesh_epoch: u64,
    physical_devices: Vec<i32>,
    parallelism: ContractParallelism,
    assignment: ContractAssignment,
    owner_ranks: Vec<usize>,
    local_slots: Vec<usize>,
    execution: String,
    collective_rows: Vec<ContractCollectiveRow>,
}

impl ExpertExecutionContract {
    /// Build a contract over every global expert. `owner_ranks` and
    /// `local_slots` are indexed by global expert id and must agree in
    /// length; the table cross-checks them against its own records on
    /// attach.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        group: impl Into<String>,
        layer: Option<usize>,
        source_fingerprint: impl Into<String>,
        mesh_epoch: u64,
        physical_devices: Vec<i32>,
        parallelism: ContractParallelism,
        assignment: ContractAssignment,
        owner_ranks: Vec<usize>,
        local_slots: Vec<usize>,
        execution: impl Into<String>,
        collective_rows: Vec<ContractCollectiveRow>,
    ) -> Result<Self, DispatchError> {
        if owner_ranks.len() != local_slots.len() {
            return Err(invalid(format!(
                "execution contract owner/slot length mismatch: {} owners vs {} slots",
                owner_ranks.len(),
                local_slots.len()
            )));
        }
        if owner_ranks.is_empty() {
            return Err(invalid("execution contract covers no experts"));
        }
        Ok(Self {
            group: group.into(),
            layer,
            source_fingerprint: source_fingerprint.into(),
            mesh_epoch,
            physical_devices,
            parallelism,
            assignment,
            owner_ranks,
            local_slots,
            execution: execution.into(),
            collective_rows,
        })
    }

    pub fn group(&self) -> &str {
        &self.group
    }

    pub fn layer(&self) -> Option<usize> {
        self.layer
    }

    pub fn source_fingerprint(&self) -> &str {
        &self.source_fingerprint
    }

    pub fn mesh_epoch(&self) -> u64 {
        self.mesh_epoch
    }

    pub fn physical_devices(&self) -> &[i32] {
        &self.physical_devices
    }

    pub fn parallelism(&self) -> ContractParallelism {
        self.parallelism
    }

    pub fn assignment(&self) -> ContractAssignment {
        self.assignment
    }

    pub fn owner_ranks(&self) -> &[usize] {
        &self.owner_ranks
    }

    pub fn local_slots(&self) -> &[usize] {
        &self.local_slots
    }

    pub fn execution(&self) -> &str {
        &self.execution
    }

    pub fn collective_rows(&self) -> &[ContractCollectiveRow] {
        &self.collective_rows
    }

    pub fn owner_rank(&self, global_id: usize) -> Option<usize> {
        self.owner_ranks.get(global_id).copied()
    }

    pub fn local_slot(&self, global_id: usize) -> Option<usize> {
        self.local_slots.get(global_id).copied()
    }

    /// Deterministic canonical rendering of the whole contract. Same sealed
    /// plan in, same string out, on every rank and in every process.
    pub fn fingerprint(&self) -> String {
        fn join(values: &[usize]) -> String {
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }
        let phys = self
            .physical_devices
            .iter()
            .map(|device| device.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut rows = String::new();
        for row in &self.collective_rows {
            let hint = match row.hint {
                ContractCollectiveHint::AllReduce { kind } => match kind {
                    ContractAxis::Pp => "AR:Pp",
                    ContractAxis::Tp => "AR:Tp",
                    ContractAxis::Ep => "AR:Ep",
                },
                ContractCollectiveHint::BandXfer { src, dst } => {
                    rows.push_str(&format!("{}:{}:BX:{src}>{dst};", row.name, row.layer));
                    continue;
                }
            };
            rows.push_str(&format!("{}:{}:{hint};", row.name, row.layer));
        }
        let mut out = String::from("sealed-ep/v1");
        out.push_str("|group|");
        out.push_str(&self.group);
        out.push_str("|layer|");
        match self.layer {
            Some(layer) => out.push_str(&layer.to_string()),
            None => out.push_str("none"),
        }
        out.push_str("|src|");
        out.push_str(&self.source_fingerprint);
        out.push_str("|epoch|");
        out.push_str(&self.mesh_epoch.to_string());
        out.push_str("|phys|");
        out.push_str(&phys);
        out.push_str("|par|");
        out.push_str(match self.parallelism {
            ContractParallelism::Single => "single",
            ContractParallelism::TensorParallel => "tp",
            ContractParallelism::ExpertParallel => "ep",
        });
        out.push_str("|assign|");
        out.push_str(match self.assignment {
            ContractAssignment::Stride => "stride",
            ContractAssignment::Contiguous => "contiguous",
        });
        out.push_str("|owners|");
        out.push_str(&join(&self.owner_ranks));
        out.push_str("|slots|");
        out.push_str(&join(&self.local_slots));
        out.push_str("|exec|");
        out.push_str(&self.execution);
        out.push_str("|rows|");
        out.push_str(&rows);
        out
    }

    /// Whether this contract authorizes the canonical EP slot combine: EP
    /// parallelism, the canonical indexed-slot execution string, and a
    /// nonempty schedule consisting only of EP all-reduce rows. The sealed
    /// combine owned by a later slice performs its own preflight on top of
    /// this; this predicate alone does not launch anything.
    pub fn is_canonical_ep(&self) -> bool {
        self.parallelism == ContractParallelism::ExpertParallel
            && self.execution == CANONICAL_EP_EXECUTION
            && !self.collective_rows.is_empty()
            && self.collective_rows.iter().all(|row| {
                matches!(
                    row.hint,
                    ContractCollectiveHint::AllReduce {
                        kind: ContractAxis::Ep
                    }
                )
            })
    }
}

/// Immutable, fully validated expert table. It owns CPU metadata only.
///
/// A process-local generation is part of the identity. It prevents an old
/// cache/binding from becoming valid again after an allocator reuses the
/// table vector's address.
#[derive(Debug, PartialEq, Eq)]
pub struct ExpertTable {
    identity: u64,
    experts: Vec<ExpertMetadata>,
    contract: Option<ExpertExecutionContract>,
}

impl ExpertTable {
    pub fn new(experts: Vec<ExpertMetadata>) -> Result<Self, DispatchError> {
        validate_expert_records(&experts)?;
        let identity = NEXT_TABLE_ID.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            identity,
            experts,
            contract: None,
        })
    }

    /// Attach the deterministic execution contract adapted from the sealed
    /// runtime plan. The contract's per-expert owner/slot map must agree
    /// with the table records exactly; a contract from another plan (or
    /// another group/layer) is rejected here, before any rank binds.
    pub fn with_execution_contract(
        self,
        contract: ExpertExecutionContract,
    ) -> Result<Self, DispatchError> {
        if contract.owner_ranks().len() != self.experts.len()
            || contract.local_slots().len() != self.experts.len()
        {
            return Err(invalid(format!(
                "execution contract covers {} experts, table holds {}",
                contract.owner_ranks().len(),
                self.experts.len()
            )));
        }
        for (index, record) in self.experts.iter().enumerate() {
            if contract.owner_ranks()[index] != record.owner_rank()
                || contract.local_slots()[index] != record.local_slot()
            {
                return Err(invalid(format!(
                    "execution contract owner/slot disagrees with expert {index}"
                )));
            }
        }
        Ok(Self {
            contract: Some(contract),
            ..self
        })
    }

    /// The adapted execution contract, if the runtime sealer attached one.
    /// Tables built without a sealed plan carry no contract.
    pub fn execution_contract(&self) -> Option<&ExpertExecutionContract> {
        self.contract.as_ref()
    }

    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    pub fn experts(&self) -> &[ExpertMetadata] {
        &self.experts
    }

    /// Prepare the rank-local lookup once, during model load.  The returned
    /// cache owns all compact metadata needed by call-time binding.
    pub fn prepare_binding(
        &self,
        local_rank: usize,
        rank_count: usize,
        physical_device: i32,
    ) -> Result<ExpertBindingCache, DispatchError> {
        ExpertBindingCache::new(self, local_rank, rank_count, physical_device)
    }
}

/// Identity of one live GPU tensor allocation.
#[derive(Debug, PartialEq, Eq)]
struct LiveTensorIdentity {
    ptr: usize,
    byte_capacity: usize,
    shape: Box<[usize]>,
    dtype: DType,
}

impl LiveTensorIdentity {
    fn capture(tensor: &GpuTensor, name: &str) -> Result<Self, DispatchError> {
        let tensor_bytes = tensor_capacity_bytes(tensor)?;
        let byte_capacity = tensor.buf.size();
        if byte_capacity != tensor_bytes {
            return Err(invalid(format!(
                "{name} buffer capacity {byte_capacity} does not equal tensor capacity {tensor_bytes}"
            )));
        }
        Ok(Self {
            ptr: tensor.buf.as_ptr() as usize,
            byte_capacity,
            shape: tensor.shape.clone().into_boxed_slice(),
            dtype: tensor.dtype,
        })
    }

    fn matches(&self, tensor: &GpuTensor, name: &str) -> Result<(), DispatchError> {
        let tensor_bytes = tensor_capacity_bytes(tensor)?;
        let byte_capacity = tensor.buf.size();
        if self.ptr != tensor.buf.as_ptr() as usize {
            return Err(invalid(format!("{name} live buffer identity differs")));
        }
        if self.byte_capacity != byte_capacity || byte_capacity != tensor_bytes {
            return Err(invalid(format!(
                "{name} live buffer capacity {byte_capacity} differs from bound {}",
                self.byte_capacity
            )));
        }
        if self.dtype != tensor.dtype
            || self.shape.len() != tensor.shape.len()
            || self
                .shape
                .iter()
                .copied()
                .zip(tensor.shape.iter().copied())
                .any(|(expected, actual)| expected != actual)
        {
            return Err(invalid(format!(
                "{name} live tensor dtype/shape differs from the bound resource"
            )));
        }
        Ok(())
    }
}

/// Identity of one live [`WeightRef`] returned by [`RoutedExpertWeights`].
#[derive(Debug, PartialEq, Eq)]
struct LiveWeightIdentity {
    buffer: LiveTensorIdentity,
    dtype: DType,
    m: usize,
    k: usize,
    row_stride: usize,
}

impl LiveWeightIdentity {
    fn capture(
        weight: &crate::families::gemv::WeightRef<'_>,
        name: &str,
    ) -> Result<Self, DispatchError> {
        Ok(Self {
            buffer: LiveTensorIdentity::capture(weight.buf, name)?,
            dtype: weight.dtype,
            m: weight.m,
            k: weight.k,
            row_stride: weight.row_stride,
        })
    }

    fn matches(
        &self,
        weight: &crate::families::gemv::WeightRef<'_>,
        name: &str,
    ) -> Result<(), DispatchError> {
        self.buffer.matches(weight.buf, name)?;
        if self.dtype != weight.dtype
            || self.m != weight.m
            || self.k != weight.k
            || self.row_stride != weight.row_stride
        {
            return Err(invalid(format!(
                "{name} live dtype/shape/stride differs from the bound resource"
            )));
        }
        Ok(())
    }
}

/// Checked identities for all live resources published by one MoE owner.
///
/// `mapping_fingerprint` retains a deterministic hex digest of the exact
/// global-entry-to-buffer mapping proven at bind time (owned entries plus
/// zero-dummy entries, table identities, and AWQ/tag presence). It travels
/// with the tensor identities so a later slice can prove which mapping a
/// sealed call was authorized under.
#[derive(Debug, PartialEq, Eq)]
struct LiveMoeBinding {
    table_identity: u64,
    table_ptr: usize,
    table_len: usize,
    experts: Box<[(LiveWeightIdentity, LiveWeightIdentity)]>,
    gate_up_ptrs: LiveTensorIdentity,
    down_ptrs: LiveTensorIdentity,
    down_awq_ptrs: Option<LiveTensorIdentity>,
    dtype_tags: Option<LiveTensorIdentity>,
    mapping_fingerprint: String,
}

/// One live expert projection pair borrowed for a compact bind: either an
/// owned local expert (in local-slot order) or an owned zero dummy backing
/// non-owned global entries. The struct borrows the caller's tensors; the
/// bind retains identities only, never the tensors themselves.
pub struct CompactLiveWeight<'a> {
    pub gate_up: crate::families::gemv::WeightRef<'a>,
    pub down: crate::families::gemv::WeightRef<'a>,
}

/// Reject a cache/table generation mismatch before any bind work.
fn check_cache_table(cache: &ExpertBindingCache, table: &ExpertTable) -> Result<(), DispatchError> {
    if cache.table_identity != table.identity
        || cache.table_ptr != table.experts.as_ptr() as usize
        || cache.table_len != table.experts.len()
    {
        return Err(invalid(
            "expert binding cache belongs to a different expert table",
        ));
    }
    Ok(())
}

/// FNV-1a hex digest over one canonical rendering. The rendering (not this
/// hash) carries the determinism; the hash only compacts it for retention.
fn fingerprint_hex(canonical: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in canonical.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Device address of the buffer behind a borrowed live weight.
fn weight_buf_ptr(weight: &crate::families::gemv::WeightRef<'_>) -> usize {
    weight.buf.buf.as_ptr() as usize
}

/// Owned, immutable load-time binding proof for one logical rank.
///
/// The cache owns no GPU allocation. Before a model is published, its owner
/// must call [`ExpertBindingCache::bind_live`] with the exact post-upload
/// routed weights and table tensors. Only then can a forward call borrow it
/// through [`BoundMoeExperts::from_cache`].
#[derive(Debug, PartialEq, Eq)]
pub struct ExpertBindingCache {
    table_identity: u64,
    table_ptr: usize,
    table_len: usize,
    local_rank: usize,
    rank_count: usize,
    physical_device: i32,
    global_to_local: Box<[Option<usize>]>,
    local_expert_ids: Box<[usize]>,
    live: Option<LiveMoeBinding>,
}

impl ExpertBindingCache {
    pub fn new(
        table: &ExpertTable,
        local_rank: usize,
        rank_count: usize,
        physical_device: i32,
    ) -> Result<Self, DispatchError> {
        if rank_count == 0 {
            return Err(invalid("expert rank count is zero"));
        }
        if local_rank >= rank_count {
            return Err(invalid(format!(
                "local expert rank {local_rank} is outside rank count {rank_count}"
            )));
        }
        if physical_device < 0 {
            return Err(invalid(format!(
                "physical device id {physical_device} is invalid"
            )));
        }

        let mut global_to_local = vec![None; table.experts.len()];
        let mut slot_ordered: Vec<(usize, usize)> = Vec::new();
        for record in &table.experts {
            if record.owner_rank() >= rank_count {
                return Err(invalid(format!(
                    "expert {} owner rank {} is outside rank count {rank_count}",
                    record.global_id(),
                    record.owner_rank()
                )));
            }
            if record.owner_rank() == local_rank {
                global_to_local[record.global_id()] = Some(record.local_slot());
                slot_ordered.push((record.local_slot(), record.global_id()));
            }
        }
        // Local experts are ordered by the sealed `local_slot`, never by
        // inferred global id order. The slots must still be compact
        // (`0..owned`), so a plan-mapped local tensor always sits at its
        // slot index.
        slot_ordered.sort();
        for (expected, (slot, global_id)) in slot_ordered.iter().copied().enumerate() {
            if slot != expected {
                return Err(invalid(format!(
                    "local expert slots must be compact and ordered: expert {global_id} has slot {slot}, expected {expected}"
                )));
            }
        }
        let local_expert_ids: Vec<usize> = slot_ordered
            .into_iter()
            .map(|(_, global_id)| global_id)
            .collect();

        Ok(Self {
            table_identity: table.identity,
            table_ptr: table.experts.as_ptr() as usize,
            table_len: table.experts.len(),
            local_rank,
            rank_count,
            physical_device,
            global_to_local: global_to_local.into_boxed_slice(),
            local_expert_ids: local_expert_ids.into_boxed_slice(),
            live: None,
        })
    }
    /// Bind the exact resources published by the model owner after allocation
    /// and pointer-table upload. The descriptor stores identities only; it
    /// does not retain GPU owners or perform a copy/download.
    ///
    /// This operation is intentionally one-shot. A reload must construct a
    /// fresh cache, so an old binding cannot be silently repointed at a new
    /// same-capacity allocation.
    pub fn bind_live(
        &mut self,
        table: &ExpertTable,
        routed_experts: &dyn RoutedExpertWeights,
        gate_up_ptrs: &GpuTensor,
        down_ptrs: &GpuTensor,
        down_awq_ptrs: Option<&GpuTensor>,
        dtype_tags: Option<&GpuTensor>,
    ) -> Result<(), DispatchError> {
        check_cache_table(self, table)?;
        if self.live.is_some() {
            return Err(invalid("expert live resources are already bound"));
        }
        let live = build_live_binding(
            table,
            routed_experts,
            gate_up_ptrs,
            down_ptrs,
            down_awq_ptrs,
            dtype_tags,
        )?;
        self.live = Some(live);
        Ok(())
    }

    /// Bind the compact expert-parallel resources owned by this rank.
    ///
    /// This sits alongside (never replaces) [`ExpertBindingCache::bind_live`],
    /// which remains the Single-owner API. The caller passes the owned local
    /// experts **in local-slot order**, the exact host pointer entries that
    /// were uploaded to each global `[n_experts]` pointer table, descriptors
    /// for the owned zero dummies backing non-owned entries, and the GPU
    /// table tensors themselves.
    ///
    /// The bind verifies, before anything is retained:
    /// * every owned global entry points to its plan-mapped local tensor
    ///   (slot order included: local expert `i` must own the global id at
    ///   local slot `i`);
    /// * every non-owned global entry points to an owned zero dummy whose
    ///   gate/up (respectively down) geometry is layout-compatible with
    ///   that expert's sealed metadata;
    /// * AWQ/tag table presence and capacity match the sealed mixed-dtype
    ///   policy, and the GPU table tensors have exact identity/capacity;
    /// * the executing device equals the cache's sealed physical device.
    ///
    /// Like [`ExpertBindingCache::bind_live`] this is one-shot and stores
    /// identities only. The mapping fingerprint is retained alongside the
    /// tensor identities. This is the residency proof a later slice uses to
    /// replace the Single-rank refusal; it does not itself admit EP
    /// execution.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_live_compact(
        &mut self,
        table: &ExpertTable,
        local_experts: &[CompactLiveWeight<'_>],
        gate_up_ptr_entries: &[usize],
        down_ptr_entries: &[usize],
        down_awq_ptr_entries: Option<&[usize]>,
        gate_up_ptrs: &GpuTensor,
        down_ptrs: &GpuTensor,
        down_awq_ptrs: Option<&GpuTensor>,
        dtype_tags: Option<&GpuTensor>,
        zero_dummies: &[CompactLiveWeight<'_>],
        device_id: i32,
    ) -> Result<(), DispatchError> {
        check_cache_table(self, table)?;
        if self.live.is_some() {
            return Err(invalid("expert live resources are already bound"));
        }
        if device_id != self.physical_device {
            return Err(invalid(format!(
                "compact bind executing device {device_id} differs from sealed physical device {}",
                self.physical_device
            )));
        }
        let live = build_compact_live_binding(
            table,
            self.local_rank,
            &self.local_expert_ids,
            local_experts,
            gate_up_ptr_entries,
            down_ptr_entries,
            down_awq_ptr_entries,
            gate_up_ptrs,
            down_ptrs,
            down_awq_ptrs,
            dtype_tags,
            zero_dummies,
        )?;
        self.live = Some(live);
        Ok(())
    }

    /// Whether this cache has passed the post-allocation live-resource bind.
    pub fn is_live_bound(&self) -> bool {
        self.live.is_some()
    }

    /// The retained mapping fingerprint from the live bind, if bound. See
    /// [`ExpertBindingCache::bind_live_compact`].
    pub fn mapping_fingerprint(&self) -> Option<&str> {
        self.live
            .as_ref()
            .map(|live| live.mapping_fingerprint.as_str())
    }

    pub fn local_rank(&self) -> usize {
        self.local_rank
    }

    pub fn rank_count(&self) -> usize {
        self.rank_count
    }

    pub fn physical_device(&self) -> i32 {
        self.physical_device
    }

    pub fn local_expert_ids(&self) -> &[usize] {
        &self.local_expert_ids
    }

    pub fn local_slot(&self, global_id: usize) -> Option<usize> {
        self.global_to_local.get(global_id).copied().flatten()
    }
}

/// Borrowed expert metadata bound to one logical rank.  The binding owns no
/// per-call lookup vectors; those live in the load-time cache.
#[derive(Clone, Copy, Debug)]
pub struct BoundMoeExperts<'a> {
    table: &'a ExpertTable,
    cache: &'a ExpertBindingCache,
}

impl<'a> BoundMoeExperts<'a> {
    /// Construct a zero-allocation borrowed view over a validated table/cache
    /// pair.  Pairing a cache with a different table is rejected even when
    /// both happen to have the same expert count.
    pub fn from_cache(
        table: &'a ExpertTable,
        cache: &'a ExpertBindingCache,
    ) -> Result<Self, DispatchError> {
        if cache.table_identity != table.identity
            || cache.table_ptr != table.experts.as_ptr() as usize
            || cache.table_len != table.experts.len()
        {
            return Err(invalid(
                "expert binding cache belongs to a different expert table",
            ));
        }
        let Some(live) = cache.live.as_ref() else {
            return Err(invalid("expert binding cache has no live resource binding"));
        };
        if live.table_identity != cache.table_identity
            || live.table_ptr != cache.table_ptr
            || live.table_len != cache.table_len
        {
            return Err(invalid(
                "expert live resource binding belongs to a different expert table",
            ));
        }
        Ok(Self { table, cache })
    }

    pub fn n_experts(&self) -> usize {
        self.table.experts.len()
    }

    pub fn local_rank(&self) -> usize {
        self.cache.local_rank
    }

    pub fn rank_count(&self) -> usize {
        self.cache.rank_count
    }

    pub fn physical_device(&self) -> i32 {
        self.cache.physical_device
    }

    pub fn records(&self) -> &[ExpertMetadata] {
        &self.table.experts
    }

    /// The deterministic execution contract adapted from the sealed plan, if
    /// the runtime sealer attached one. Read-only: a later slice preflights
    /// the canonical EP combine against this without mutating the binding.
    pub fn execution_contract(&self) -> Option<&ExpertExecutionContract> {
        self.table.execution_contract()
    }

    pub fn local_expert_ids(&self) -> &[usize] {
        &self.cache.local_expert_ids
    }

    pub fn local_slot(&self, global_id: usize) -> Option<usize> {
        self.cache.local_slot(global_id)
    }

    pub fn expert(&self, global_id: usize) -> Option<&ExpertMetadata> {
        self.table.experts.get(global_id)
    }
}

/// A route result tied to one invocation of one sealed call.
///
/// The fields are private and the type has no public constructor.  The only
/// production function that creates one is [`produce_prefill_route`], which
/// executes the family-specific router before returning this proof.  A receipt
/// therefore cannot be made by wrapping an arbitrary pair of device buffers.
pub struct MoeRouteReceipt<'a> {
    invocation: u64,
    protocol: MoeProtocol,
    router: MoeRouterInput,
    n_experts: usize,
    k_top: usize,
    /// Identity of the score tensor consumed by the producer.  This is kept as
    /// an opaque pointer identity; no pointer is ever dereferenced here.
    scores: usize,
    normalized: bool,
    indices: &'a GpuTensor,
    weights: &'a GpuTensor,
}

impl MoeRouteReceipt<'_> {
    pub fn protocol(&self) -> MoeProtocol {
        self.protocol
    }

    pub fn router_input(&self) -> MoeRouterInput {
        self.router
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub fn k_top(&self) -> usize {
        self.k_top
    }
}

/// A fully checked decode or grouped-prefill call.  Its fields are private so
/// no caller can inject an alternate expert table, operation order, or combine
/// policy after validation.
pub struct SealedMoeCall<'a> {
    invocation: u64,
    experts: BoundMoeExperts<'a>,
    protocol: MoeProtocol,
    router: MoeRouterInput,
    contribution: MoeContribution,
    shared: MoeSharedContribution,
    activation: ActivationIdentity,
    params: SealedParams<'a>,
    route_receipt: Option<MoeRouteReceipt<'a>>,
}

enum SealedParams<'a> {
    Decode(MoeParams<'a>),
    Prefill(MoePrefillParams<'a>),
}

impl SealedMoeCall<'_> {
    pub fn invocation(&self) -> u64 {
        self.invocation
    }

    pub fn protocol(&self) -> MoeProtocol {
        self.protocol
    }

    pub fn router_input(&self) -> MoeRouterInput {
        self.router
    }

    pub fn contribution(&self) -> MoeContribution {
        self.contribution
    }

    pub fn shared_contribution(&self) -> MoeSharedContribution {
        self.shared
    }

    pub fn activation(&self) -> ActivationIdentity {
        self.activation
    }

    pub fn experts(&self) -> &BoundMoeExperts<'_> {
        &self.experts
    }

    pub fn decode_params(&self) -> Option<&MoeParams<'_>> {
        match &self.params {
            SealedParams::Decode(params) => Some(params),
            SealedParams::Prefill(_) => None,
        }
    }

    pub fn prefill_params(&self) -> Option<&MoePrefillParams<'_>> {
        match &self.params {
            SealedParams::Decode(_) => None,
            SealedParams::Prefill(params) => Some(params),
        }
    }

    /// Whether a producer receipt has been attached to this call.
    pub fn has_route_receipt(&self) -> bool {
        self.route_receipt.is_some()
    }
}

impl<'a> SealedMoeCall<'a> {
    fn expected_route_receipt(&self) -> MoeRouteReceipt<'a> {
        let (indices, weights, n_experts, k_top) = match &self.params {
            SealedParams::Decode(params) => (
                params.topk_indices,
                params.topk_weights,
                params.n_exp,
                params.k,
            ),
            SealedParams::Prefill(params) => (
                params.topk_indices,
                params.topk_weights,
                params.n_exp,
                params.k_top,
            ),
        };
        MoeRouteReceipt {
            invocation: self.invocation,
            protocol: self.protocol,
            router: self.router,
            n_experts,
            k_top,
            scores: 0,
            normalized: false,
            indices,
            weights,
        }
    }

    /// Attach the receipt returned by the actual route producer.  A second
    /// producer or a receipt from another invocation is rejected.
    pub fn attach_route_receipt(
        &mut self,
        receipt: MoeRouteReceipt<'a>,
    ) -> Result<(), DispatchError> {
        if self.protocol != MoeProtocol::GroupedPrefill {
            return Err(invalid(
                "route receipts may only be attached to grouped prefill calls",
            ));
        }
        if self.route_receipt.is_some() {
            return Err(invalid("a route receipt is already attached"));
        }
        let expected = self.expected_route_receipt();
        validate_route_receipt_pair(&expected, &receipt)?;
        self.route_receipt = Some(receipt);
        Ok(())
    }

    /// Validate that a producer receipt belongs to this exact invocation and
    /// route-buffer pair.
    pub fn validate_route_receipt(
        &self,
        receipt: &MoeRouteReceipt<'_>,
    ) -> Result<(), DispatchError> {
        let expected = self.expected_route_receipt();
        validate_route_receipt_pair(&expected, receipt)
    }

    /// Check the dynamic device and route proof immediately before execution.
    /// This method contains no allocation and must run before any launch in a
    /// step list.
    pub(crate) fn validate_for_gpu(&self, gpu: &Gpu) -> Result<(), DispatchError> {
        if self.experts.rank_count() != 1 || self.experts.local_rank() != 0 {
            return Err(invalid(format!(
                "sealed MoE execution requires Single rank (rank={}/{}); refusing before launch",
                self.experts.local_rank(),
                self.experts.rank_count()
            )));
        }
        if self.experts.physical_device() != gpu.device_id {
            return Err(invalid(format!(
                "sealed MoE physical device mismatch: plan={} executing_gpu={}",
                self.experts.physical_device(),
                gpu.device_id
            )));
        }
        if self.protocol == MoeProtocol::GroupedPrefill {
            let receipt = self
                .route_receipt
                .as_ref()
                .ok_or_else(|| invalid("prefill route producer receipt is missing"))?;
            self.validate_route_receipt(receipt)?;
        }
        Ok(())
    }
}

fn validate_route_receipt_pair(
    expected: &MoeRouteReceipt<'_>,
    receipt: &MoeRouteReceipt<'_>,
) -> Result<(), DispatchError> {
    if receipt.invocation != expected.invocation {
        return Err(invalid("route receipt belongs to another invocation"));
    }
    if receipt.protocol != expected.protocol || receipt.router != expected.router {
        return Err(invalid("route receipt grammar or router mode mismatch"));
    }
    if receipt.n_experts != expected.n_experts || receipt.k_top != expected.k_top {
        return Err(invalid("route receipt dimensions mismatch"));
    }
    if !std::ptr::eq(receipt.indices, expected.indices)
        || !std::ptr::eq(receipt.weights, expected.weights)
    {
        return Err(invalid("route receipt buffers mismatch"));
    }
    Ok(())
}

/// Run the family-specific prefill router and mint its invocation-bound
/// receipt.  Qwen's softmax producer and Cohere's sigmoid producer both use
/// this boundary; model callers must not launch those operations themselves.
pub fn produce_prefill_route<'a>(
    call: &SealedMoeCall<'a>,
    gpu: &mut Gpu,
    scores: &GpuTensor,
    normalize: bool,
) -> Result<MoeRouteReceipt<'a>, DispatchError> {
    if call.protocol != MoeProtocol::GroupedPrefill {
        return Err(invalid("prefill route producer requires GroupedPrefill"));
    }
    if !matches!(
        call.router,
        MoeRouterInput::PrecomputedSoftmaxTopK | MoeRouterInput::PrecomputedSigmoidTopK
    ) {
        return Err(invalid("unsupported prefill route producer"));
    }
    let params = call
        .prefill_params()
        .ok_or_else(|| invalid("prefill route producer received a decode call"))?;
    let score_elements = params
        .batch_size
        .checked_mul(params.n_exp)
        .ok_or_else(|| invalid("prefill router score capacity overflows"))?;
    require_elements(scores, score_elements, "prefill router scores")?;

    match call.router {
        MoeRouterInput::PrecomputedSoftmaxTopK => {
            if scores.shape.len() != 2
                || scores.shape[0] != params.batch_size
                || scores.shape[1] != params.n_exp
            {
                return Err(invalid(
                    "prefill softmax producer requires a 2-D [batch,n_experts] score view",
                ));
            }
            gpu.softmax_f32(scores)
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            gpu.moe_topk_renorm_k8_batched(
                scores,
                params.topk_indices,
                params.topk_weights,
                params.n_exp,
                normalize,
                params.batch_size,
            )
            .map_err(|e| DispatchError::Hip(e.to_string()))?;
        }
        MoeRouterInput::PrecomputedSigmoidTopK => {
            #[cfg(feature = "deltanet")]
            {
                gpu.sigmoid_f32(scores)
                    .map_err(|e| DispatchError::Hip(e.to_string()))?;
                gpu.moe_topk_renorm_k8_batched(
                    scores,
                    params.topk_indices,
                    params.topk_weights,
                    params.n_exp,
                    normalize,
                    params.batch_size,
                )
                .map_err(|e| DispatchError::Hip(e.to_string()))?;
            }
            #[cfg(not(feature = "deltanet"))]
            {
                return Err(invalid(
                    "sigmoid prefill routing requires the dispatch deltanet feature",
                ));
            }
        }
        MoeRouterInput::SoftmaxTopK => unreachable!("validated above"),
    }

    let expected = call.expected_route_receipt();
    Ok(MoeRouteReceipt {
        invocation: expected.invocation,
        protocol: expected.protocol,
        router: expected.router,
        n_experts: expected.n_experts,
        k_top: expected.k_top,
        scores: scores.buf.as_ptr() as usize,
        normalized: normalize,
        indices: expected.indices,
        weights: expected.weights,
    })
}

/// Execute a validated sealed call.  The arithmetic helpers remain in
/// `pipeline::mod`; this chokepoint prevents a caller from bypassing the
/// bound-call checks.
pub(crate) fn execute_sealed(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    call.validate_for_gpu(gpu)?;
    match call.protocol() {
        MoeProtocol::IndexedDecode => super::run_moe_decode(ctx, gpu, call),
        MoeProtocol::GroupedPrefill => super::run_moe_prefill(ctx, gpu, call),
    }
}

/// Seal the existing indexed decode parameter record.
pub fn seal_decode<'a>(
    experts: BoundMoeExperts<'a>,
    ctx: &DispatchCtx,
    params: MoeParams<'a>,
) -> Result<SealedMoeCall<'a>, DispatchError> {
    seal_decode_with_router(experts, ctx, params, MoeRouterInput::SoftmaxTopK)
}

/// Checked decode adapter hook.  Decode currently admits only the in-call
/// softmax/top-k producer; other router identities are rejected rather than
/// silently treating precomputed buffers as executable authority.
pub fn seal_decode_with_router<'a>(
    experts: BoundMoeExperts<'a>,
    ctx: &DispatchCtx,
    params: MoeParams<'a>,
    router: MoeRouterInput,
) -> Result<SealedMoeCall<'a>, DispatchError> {
    require_single_binding(&experts)?;
    if router != MoeRouterInput::SoftmaxTopK {
        return Err(invalid("indexed decode requires SoftmaxTopK routing"));
    }
    validate_decode(ctx, &experts, &params)?;
    let basis = expected_basis(&experts, params.dtypes.routed_gate_up)?;
    let (contribution, shared) = decode_combine(&params)?;
    Ok(SealedMoeCall {
        invocation: NEXT_INVOCATION.fetch_add(1, Ordering::Relaxed),
        experts,
        protocol: MoeProtocol::IndexedDecode,
        router,
        contribution,
        shared,
        activation: ActivationIdentity::new(
            if params.x_rot_prerotated {
                ActivationInput::PreRotated
            } else {
                ActivationInput::Natural
            },
            basis,
        ),
        params: SealedParams::Decode(params),
        route_receipt: None,
    })
}

/// Seal the existing grouped-prefill parameter record using the Qwen softmax
/// route producer.  The producer must be run through [`produce_prefill_route`]
/// and attached before the call is executable.
pub fn seal_prefill<'a>(
    experts: BoundMoeExperts<'a>,
    ctx: &DispatchCtx,
    params: MoePrefillParams<'a>,
) -> Result<SealedMoeCall<'a>, DispatchError> {
    seal_prefill_with_router(experts, ctx, params, MoeRouterInput::PrecomputedSoftmaxTopK)
}

/// Checked prefill adapter hook for model families whose route producer has a
/// different normalized score basis (currently Cohere's sigmoid/top-k path).
pub fn seal_prefill_with_router<'a>(
    experts: BoundMoeExperts<'a>,
    ctx: &DispatchCtx,
    params: MoePrefillParams<'a>,
    router: MoeRouterInput,
) -> Result<SealedMoeCall<'a>, DispatchError> {
    require_single_binding(&experts)?;
    if router == MoeRouterInput::SoftmaxTopK {
        return Err(invalid(
            "prefill route buffers must identify a precomputed softmax or sigmoid producer",
        ));
    }
    validate_prefill(ctx, &experts, &params)?;
    let basis = expected_basis(&experts, params.dtypes.routed_gate_up)?;
    let contribution = if params.routed_out.is_some() {
        MoeContribution::ZeroedPartial
    } else {
        MoeContribution::Residual
    };
    Ok(SealedMoeCall {
        invocation: NEXT_INVOCATION.fetch_add(1, Ordering::Relaxed),
        experts,
        protocol: MoeProtocol::GroupedPrefill,
        router,
        contribution,
        // Prefill's shared output is already in each rank's residual and is
        // deliberately outside the routed reduction.
        shared: MoeSharedContribution::PerRankResidual,
        activation: ActivationIdentity::new(ActivationInput::PreRotated, basis),
        params: SealedParams::Prefill(params),
        route_receipt: None,
    })
}
fn require_single_binding(experts: &BoundMoeExperts<'_>) -> Result<(), DispatchError> {
    if experts.rank_count() != 1 || experts.local_rank() != 0 {
        return Err(invalid(format!(
            "sealed MoE only admits Single rank (rank={}/{}); refusing before launch",
            experts.local_rank(),
            experts.rank_count()
        )));
    }
    Ok(())
}

fn validate_decode(
    _ctx: &DispatchCtx,
    experts: &BoundMoeExperts<'_>,
    params: &MoeParams<'_>,
) -> Result<(), DispatchError> {
    if params.batch_size != 1 {
        return Err(invalid(format!(
            "indexed decode requires batch_size=1, got {}",
            params.batch_size
        )));
    }
    if params.hidden == 0 || params.mi == 0 || params.smi == 0 {
        return Err(invalid("decode dimensions must be nonzero"));
    }
    if params.n_exp == 0 || params.k == 0 || params.k > params.n_exp {
        return Err(invalid(format!(
            "decode route width k={} is outside 1..={} ",
            params.k, params.n_exp
        )));
    }
    if params.defer_routed_combine {
        return Err(invalid(
            "sealed indexed decode requires exactly one routed combine",
        ));
    }
    let gate_up_rows = params
        .mi
        .checked_mul(2)
        .ok_or_else(|| invalid("decode gate/up dimension overflows"))?;
    validate_expert_shape_and_dtype(
        experts,
        params.n_exp,
        gate_up_rows,
        params.hidden,
        params.hidden,
        params.mi,
        params.dtypes.routed_gate_up,
        params.dtypes.routed_down,
        params.dtypes.routed_has_mixed_experts,
        params.dtypes.per_expert_gate_up.as_deref(),
        params.dtypes.per_expert_down.as_deref(),
    )?;
    validate_live_binding(
        experts,
        Some(params.routed_experts),
        params.expert_gate_up_ptrs,
        params.expert_down_ptrs,
        params.expert_down_awq_ptrs,
        params.expert_dtype_tags,
    )?;
    validate_dtype_tag_table(
        params.expert_dtype_tags,
        params.n_exp,
        params.dtypes.routed_has_mixed_experts,
    )?;
    validate_pointer_table(params.expert_gate_up_ptrs, params.n_exp, "decode gate/up")?;
    validate_pointer_table(params.expert_down_ptrs, params.n_exp, "decode down")?;
    if let Some(table) = params.expert_down_awq_ptrs {
        validate_pointer_table(table, params.n_exp, "decode down AWQ")?;
    }
    require_elements(params.x_norm, params.hidden, "decode x_norm")?;
    require_elements(params.x_residual, params.hidden, "decode residual")?;
    require_elements(params.router_logits, params.n_exp, "decode router logits")?;
    require_elements(params.topk_indices, params.k, "decode top-k indices")?;
    require_elements(params.topk_weights, params.k, "decode top-k weights")?;
    require_elements(
        params.x_rot_local,
        params.hidden,
        "decode rotated activation",
    )?;
    require_elements(
        params.gate_up_buf,
        2usize
            .checked_mul(params.mi)
            .ok_or_else(|| invalid("decode gate/up scratch overflows"))?,
        "decode gate/up scratch",
    )?;
    require_elements(params.gate_buf, params.mi, "decode gate scratch")?;
    require_elements(params.up_buf, params.mi, "decode up scratch")?;
    require_elements(params.ffn_hidden, params.smi, "decode shared activation")?;
    require_elements(params.ffn_out, params.hidden, "decode shared output")?;
    require_elements(
        params.gate_batch,
        params
            .k
            .checked_mul(params.mi)
            .ok_or_else(|| invalid("decode gate batch capacity overflows"))?,
        "decode gate batch",
    )?;
    require_elements(
        params.up_batch,
        params
            .k
            .checked_mul(params.mi)
            .ok_or_else(|| invalid("decode up batch capacity overflows"))?,
        "decode up batch",
    )?;
    require_elements(
        params.rot_batch,
        params
            .k
            .checked_mul(params.mi)
            .ok_or_else(|| invalid("decode rotation batch capacity overflows"))?,
        "decode rotation batch",
    )?;
    require_elements(
        params.down_expanded,
        params
            .k
            .checked_mul(params.hidden)
            .ok_or_else(|| invalid("decode expanded down capacity overflows"))?,
        "decode expanded down",
    )?;
    if let Some(partial) = params.routed_out {
        require_elements(partial, params.hidden, "decode routed partial")?;
    }
    Ok(())
}

fn validate_prefill(
    _ctx: &DispatchCtx,
    experts: &BoundMoeExperts<'_>,
    params: &MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    if params.batch_size == 0 {
        return Err(invalid("grouped prefill batch_size is zero"));
    }
    if params.mi == 0 || params.down_m == 0 || params.down_k == 0 || params.gate_up_k == 0 {
        return Err(invalid("prefill dimensions must be nonzero"));
    }
    if params.n_exp == 0 || params.k_top == 0 || params.k_top > params.n_exp {
        return Err(invalid(format!(
            "prefill route width k_top={} is outside 1..={} ",
            params.k_top, params.n_exp
        )));
    }
    let total_slots = params
        .batch_size
        .checked_mul(params.k_top)
        .ok_or_else(|| invalid("prefill route slot count overflows"))?;
    validate_grouped_m_total_max(total_slots, params.n_exp, params.m_total_max)?;
    validate_expert_shape_and_dtype(
        experts,
        params.n_exp,
        2usize
            .checked_mul(params.mi)
            .ok_or_else(|| invalid("prefill gate/up dimension overflows"))?,
        params.gate_up_k,
        params.down_m,
        params.down_k,
        params.dtypes.routed_gate_up,
        params.dtypes.routed_down,
        params.dtypes.routed_has_mixed_experts,
        params.dtypes.per_expert_gate_up.as_deref(),
        params.dtypes.per_expert_down.as_deref(),
    )?;
    validate_live_binding(
        experts,
        Some(params.routed_experts),
        params.expert_gate_up_ptrs,
        params.expert_down_ptrs,
        params.expert_down_awq_ptrs,
        params.expert_dtype_tags,
    )?;
    validate_dtype_tag_table(
        params.expert_dtype_tags,
        params.n_exp,
        params.dtypes.routed_has_mixed_experts,
    )?;
    validate_pointer_table(params.expert_gate_up_ptrs, params.n_exp, "prefill gate/up")?;
    validate_pointer_table(params.expert_down_ptrs, params.n_exp, "prefill down")?;
    if let Some(table) = params.expert_down_awq_ptrs {
        validate_pointer_table(table, params.n_exp, "prefill down AWQ")?;
    }
    require_elements(params.topk_indices, total_slots, "prefill top-k indices")?;
    require_elements(params.topk_weights, total_slots, "prefill top-k weights")?;
    require_elements(
        params.x_batch,
        params
            .batch_size
            .checked_mul(params.down_m)
            .ok_or_else(|| invalid("prefill residual capacity overflows"))?,
        "prefill residual",
    )?;
    require_elements(
        params.x_norm_batch,
        params
            .batch_size
            .checked_mul(params.gate_up_k)
            .ok_or_else(|| invalid("prefill norm capacity overflows"))?,
        "prefill normalized activation",
    )?;
    require_elements(
        params.x_rot_batch,
        params
            .batch_size
            .checked_mul(params.gate_up_k)
            .ok_or_else(|| invalid("prefill rotated activation capacity overflows"))?,
        "prefill rotated activation",
    )?;
    let gate_capacity = total_slots
        .checked_mul(params.mi)
        .ok_or_else(|| invalid("prefill gate scratch capacity overflows"))?;
    require_elements(params.gate_batch, gate_capacity, "prefill gate scratch")?;
    require_elements(params.up_batch, gate_capacity, "prefill up scratch")?;
    require_elements(params.rot_batch, gate_capacity, "prefill rotation scratch")?;
    require_elements(
        params.down_expanded,
        total_slots
            .checked_mul(params.down_m)
            .ok_or_else(|| invalid("prefill expanded down capacity overflows"))?,
        "prefill expanded down",
    )?;
    require_elements(
        params.expert_token_counts,
        params.n_exp,
        "prefill expert token counts",
    )?;
    require_elements(
        params.expert_offsets,
        params
            .n_exp
            .checked_add(1)
            .ok_or_else(|| invalid("prefill expert offset capacity overflows"))?,
        "prefill expert offsets",
    )?;
    require_elements(
        params.sorted_slot_index,
        params.m_total_max,
        "prefill sorted slots",
    )?;
    let expert_tile_bytes = (params.m_total_max / GROUPED_BLOCK_M)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| invalid("prefill expert tile capacity overflows"))?;
    require_elements(
        params.expert_tile_ids,
        expert_tile_bytes,
        "prefill expert tiles",
    )?;
    require_elements(
        params.inverse_perm,
        total_slots,
        "prefill inverse permutation",
    )?;
    require_elements(
        params.y_gate_up_grouped,
        params
            .m_total_max
            .checked_mul(
                2usize
                    .checked_mul(params.mi)
                    .ok_or_else(|| invalid("prefill grouped gate/up capacity overflows"))?,
            )
            .ok_or_else(|| invalid("prefill grouped gate/up capacity overflows"))?,
        "prefill grouped gate/up",
    )?;
    require_elements(
        params.y_down_grouped,
        params
            .m_total_max
            .checked_mul(params.down_m)
            .ok_or_else(|| invalid("prefill grouped down capacity overflows"))?,
        "prefill grouped down",
    )?;
    if let Some(partial) = params.routed_out {
        require_elements(
            partial,
            params
                .batch_size
                .checked_mul(params.down_m)
                .ok_or_else(|| invalid("prefill routed partial capacity overflows"))?,
            "prefill routed partial",
        )?;
    }
    if let Some(scale) = params.down_awq_scale {
        require_elements(scale, 1, "prefill down AWQ scale")?;
    }
    Ok(())
}

fn decode_combine(
    params: &MoeParams<'_>,
) -> Result<(MoeContribution, MoeSharedContribution), DispatchError> {
    match (params.routed_out.is_some(), params.skip_shared) {
        (false, false) => Ok((MoeContribution::Residual, MoeSharedContribution::None)),
        (true, true) => Ok((
            MoeContribution::ZeroedPartial,
            MoeSharedContribution::RootPartial,
        )),
        (true, false) => Err(invalid(
            "decode partial must not reduce replicated shared output",
        )),
        (false, true) => Err(invalid(
            "decode skip_shared requires a zeroed routed partial",
        )),
    }
}

fn validate_expert_shape_and_dtype(
    experts: &BoundMoeExperts<'_>,
    n_experts: usize,
    gate_up_rows: usize,
    gate_up_cols: usize,
    down_rows: usize,
    down_cols: usize,
    representative_gate_up: DType,
    representative_down: DType,
    mixed: bool,
    per_gate_up: Option<&[DType]>,
    per_down: Option<&[DType]>,
) -> Result<(), DispatchError> {
    if experts.n_experts() != n_experts {
        return Err(invalid(format!(
            "expert table has {} records, parameter record declares {n_experts}",
            experts.n_experts()
        )));
    }
    let expected_basis = dtype_rotation_plan(representative_gate_up);
    if mixed {
        if per_gate_up.map_or(true, |values| values.len() != n_experts)
            || per_down.map_or(true, |values| values.len() != n_experts)
        {
            return Err(invalid(
                "mixed expert metadata must contain one gate/up and down dtype per expert",
            ));
        }
    } else if per_gate_up.is_some() || per_down.is_some() {
        return Err(invalid(
            "per-expert dtype tables are present for a non-mixed execution",
        ));
    }
    if gate_up_rows % 2 != 0 {
        return Err(invalid(format!(
            "fused gate/up row count {gate_up_rows} is not divisible by two"
        )));
    }
    let separate_gate_up_rows = gate_up_rows / 2;
    for (index, expert) in experts.records().iter().enumerate() {
        let resources = expert.resources();
        let expected_gate_up = per_gate_up.map_or(representative_gate_up, |values| values[index]);
        let gate_up_dtype = if let Some(gate_up) = resources.gate_up() {
            if gate_up.dtype() != expected_gate_up {
                return Err(invalid(format!(
                    "expert {index} gate/up source '{}' dtype differs from the declared source metadata",
                    gate_up.source_name()
                )));
            }
            if dtype_rotation_plan(gate_up.dtype()) != expected_basis {
                return Err(invalid(format!(
                    "expert {index} gate/up source '{}' rotation basis differs from the activation basis",
                    gate_up.source_name()
                )));
            }
            validate_logical_shape(
                gate_up.shape(),
                gate_up_rows,
                gate_up_cols,
                "gate/up",
                index,
            )?;
            gate_up.dtype()
        } else {
            let gate = resources
                .gate()
                .ok_or_else(|| invalid("separate expert resources have no gate source"))?;
            let up = resources
                .up()
                .ok_or_else(|| invalid("separate expert resources have no up source"))?;
            if gate.dtype() != up.dtype() || gate.basis() != up.basis() {
                return Err(invalid(format!(
                    "expert {index} separate gate '{}' and up '{}' dtype or basis mismatch",
                    gate.source_name(),
                    up.source_name()
                )));
            }
            for (role, source) in [("gate", gate), ("up", up)] {
                if source.dtype() != expected_gate_up {
                    return Err(invalid(format!(
                        "expert {index} separate {role} source '{}' dtype differs from the declared source metadata",
                        source.source_name()
                    )));
                }
                if dtype_rotation_plan(source.dtype()) != expected_basis {
                    return Err(invalid(format!(
                        "expert {index} separate {role} source '{}' rotation basis differs from the activation basis",
                        source.source_name()
                    )));
                }
                validate_logical_shape(
                    source.shape(),
                    separate_gate_up_rows,
                    gate_up_cols,
                    role,
                    index,
                )?;
            }
            gate.dtype()
        };
        if gate_up_dtype != expected_gate_up {
            return Err(invalid(format!(
                "expert {index} gate/up dtype differs from the declared source metadata"
            )));
        }
        if dtype_rotation_plan(gate_up_dtype) != expected_basis {
            return Err(invalid(format!(
                "expert {index} gate/up rotation basis differs from the activation basis"
            )));
        }

        let down = resources.down();
        let expected_down = per_down.map_or(representative_down, |values| values[index]);
        if down.dtype() != expected_down {
            return Err(invalid(format!(
                "expert {index} down dtype differs from the declared source metadata"
            )));
        }
        if dtype_rotation_plan(down.dtype()) != expected_basis {
            return Err(invalid(format!(
                "expert {index} down rotation basis differs from the activation basis"
            )));
        }
        validate_logical_shape(down.shape(), down_rows, down_cols, "down", index)?;
    }
    Ok(())
}

fn build_live_binding(
    table: &ExpertTable,
    routed_experts: &dyn RoutedExpertWeights,
    gate_up_ptrs: &GpuTensor,
    down_ptrs: &GpuTensor,
    down_awq_ptrs: Option<&GpuTensor>,
    dtype_tags: Option<&GpuTensor>,
) -> Result<LiveMoeBinding, DispatchError> {
    let n_experts = table.n_experts();
    if routed_experts.len() != n_experts {
        return Err(invalid(format!(
            "live routed expert table has {} records, expected {n_experts}",
            routed_experts.len()
        )));
    }
    validate_pointer_table(gate_up_ptrs, n_experts, "live gate/up")?;
    validate_pointer_table(down_ptrs, n_experts, "live down")?;
    if let Some(table) = down_awq_ptrs {
        validate_pointer_table(table, n_experts, "live down AWQ")?;
    }
    validate_dtype_tag_table(dtype_tags, n_experts, table_has_mixed_dtypes(table)?)?;

    let gate_up_identity = LiveTensorIdentity::capture(gate_up_ptrs, "live gate/up")?;
    let down_identity = LiveTensorIdentity::capture(down_ptrs, "live down")?;
    let down_awq_identity = down_awq_ptrs
        .map(|table| LiveTensorIdentity::capture(table, "live down AWQ"))
        .transpose()?;
    let dtype_tag_identity = dtype_tags
        .map(|table| LiveTensorIdentity::capture(table, "live dtype tags"))
        .transpose()?;

    let mut live_experts = Vec::with_capacity(n_experts);
    for (index, metadata) in table.experts().iter().enumerate() {
        let (gate_up, down) = routed_experts.get(index).ok_or_else(|| {
            invalid(format!(
                "live routed expert table is missing expert {index}"
            ))
        })?;
        let gate_up = bind_live_weight(metadata, &gate_up, true, index, "gate/up")?;
        let down = bind_live_weight(metadata, &down, false, index, "down")?;
        live_experts.push((gate_up, down));
    }

    let mut canonical = format!("single-live/v1|n|{n_experts}|experts|");
    for (index, (gate_up, down)) in live_experts.iter().enumerate() {
        canonical.push_str(&format!(
            "{index}:{:x}:{}:{:x}:{};",
            gate_up.buffer.ptr,
            gate_up.buffer.byte_capacity,
            down.buffer.ptr,
            down.buffer.byte_capacity
        ));
    }
    canonical.push_str(&format!(
        "|tables|{:x}:{}:{:x}:{}|awq|",
        gate_up_identity.ptr,
        gate_up_identity.byte_capacity,
        down_identity.ptr,
        down_identity.byte_capacity
    ));
    match &down_awq_identity {
        Some(identity) => {
            canonical.push_str(&format!("{:x}:{};", identity.ptr, identity.byte_capacity))
        }
        None => canonical.push_str("none;"),
    }
    canonical.push_str("|tags|");
    match &dtype_tag_identity {
        Some(identity) => {
            canonical.push_str(&format!("{:x}:{};", identity.ptr, identity.byte_capacity))
        }
        None => canonical.push_str("none;"),
    }

    Ok(LiveMoeBinding {
        table_identity: table.identity,
        table_ptr: table.experts.as_ptr() as usize,
        table_len: table.experts.len(),
        experts: live_experts.into_boxed_slice(),
        gate_up_ptrs: gate_up_identity,
        down_ptrs: down_identity,
        down_awq_ptrs: down_awq_identity,
        dtype_tags: dtype_tag_identity,
        mapping_fingerprint: fingerprint_hex(&canonical),
    })
}

/// Build the compact live binding proven by
/// [`ExpertBindingCache::bind_live_compact`]. `local_slot_order` holds this
/// rank's owned global ids in local-slot order; `local_experts[i]` is the
/// live tensor pair for the global id at slot `i`. Host pointer entries are
/// the exact values uploaded to the global `[n_experts]` tables. Every
/// non-owned entry must name an owned zero dummy whose matching half is
/// layout-compatible with that expert; AWQ host entries and tag tables are
/// presence/capacity checked like the Single path (scale storage itself is
/// not mapped here).
#[allow(clippy::too_many_arguments)]
fn build_compact_live_binding(
    table: &ExpertTable,
    local_rank: usize,
    local_slot_order: &[usize],
    local_experts: &[CompactLiveWeight<'_>],
    gate_up_ptr_entries: &[usize],
    down_ptr_entries: &[usize],
    down_awq_ptr_entries: Option<&[usize]>,
    gate_up_ptrs: &GpuTensor,
    down_ptrs: &GpuTensor,
    down_awq_ptrs: Option<&GpuTensor>,
    dtype_tags: Option<&GpuTensor>,
    zero_dummies: &[CompactLiveWeight<'_>],
) -> Result<LiveMoeBinding, DispatchError> {
    let n_experts = table.n_experts();
    if gate_up_ptr_entries.len() != n_experts {
        return Err(invalid(format!(
            "compact gate/up host entries cover {} experts, expected {n_experts}",
            gate_up_ptr_entries.len()
        )));
    }
    if down_ptr_entries.len() != n_experts {
        return Err(invalid(format!(
            "compact down host entries cover {} experts, expected {n_experts}",
            down_ptr_entries.len()
        )));
    }
    if local_experts.len() != local_slot_order.len() {
        return Err(invalid(format!(
            "compact local experts cover {} slots, rank {local_rank} owns {}",
            local_experts.len(),
            local_slot_order.len()
        )));
    }
    match (down_awq_ptr_entries, down_awq_ptrs) {
        (Some(entries), Some(_)) => {
            if entries.len() != n_experts {
                return Err(invalid(format!(
                    "compact down AWQ host entries cover {} experts, expected {n_experts}",
                    entries.len()
                )));
            }
        }
        (None, None) => {}
        (Some(_), None) => {
            return Err(invalid(
                "compact down AWQ host entries supplied without a down AWQ pointer table",
            ));
        }
        (None, Some(_)) => {
            return Err(invalid(
                "compact down AWQ pointer table supplied without down AWQ host entries",
            ));
        }
    }
    validate_pointer_table(gate_up_ptrs, n_experts, "compact gate/up")?;
    validate_pointer_table(down_ptrs, n_experts, "compact down")?;
    if let Some(tensor) = down_awq_ptrs {
        validate_pointer_table(tensor, n_experts, "compact down AWQ")?;
    }
    validate_dtype_tag_table(dtype_tags, n_experts, table_has_mixed_dtypes(table)?)?;

    let gate_up_identity = LiveTensorIdentity::capture(gate_up_ptrs, "compact gate/up")?;
    let down_identity = LiveTensorIdentity::capture(down_ptrs, "compact down")?;
    let down_awq_identity = down_awq_ptrs
        .map(|tensor| LiveTensorIdentity::capture(tensor, "compact down AWQ"))
        .transpose()?;
    let dtype_tag_identity = dtype_tags
        .map(|tensor| LiveTensorIdentity::capture(tensor, "compact dtype tags"))
        .transpose()?;

    // Owned globals must point at their plan-mapped local tensor: slot `i`
    // of `local_experts` backs the owned global id at slot `i`.
    let mut live_experts: Vec<Option<(LiveWeightIdentity, LiveWeightIdentity)>> =
        (0..n_experts).map(|_| None).collect();
    for (slot, &global_id) in local_slot_order.iter().enumerate() {
        let metadata = table.experts().get(global_id).ok_or_else(|| {
            invalid(format!(
                "compact local slot {slot} names missing expert {global_id}"
            ))
        })?;
        if metadata.owner_rank() != local_rank || metadata.local_slot() != slot {
            return Err(invalid(format!(
                "compact local slot {slot} names expert {global_id} owned by rank {} at slot {}",
                metadata.owner_rank(),
                metadata.local_slot()
            )));
        }
        let local = local_experts
            .get(slot)
            .ok_or_else(|| invalid(format!("compact local experts are missing slot {slot}")))?;
        let gate_ptr = weight_buf_ptr(&local.gate_up);
        let down_ptr = weight_buf_ptr(&local.down);
        if gate_up_ptr_entries[global_id] != gate_ptr {
            return Err(invalid(format!(
                "compact owned expert {global_id} gate/up entry {:x} does not point at its local tensor {:x}",
                gate_up_ptr_entries[global_id], gate_ptr
            )));
        }
        if down_ptr_entries[global_id] != down_ptr {
            return Err(invalid(format!(
                "compact owned expert {global_id} down entry {:x} does not point at its local tensor {:x}",
                down_ptr_entries[global_id], down_ptr
            )));
        }
        let gate_up = bind_live_weight(metadata, &local.gate_up, true, global_id, "gate/up")?;
        let down = bind_live_weight(metadata, &local.down, false, global_id, "down")?;
        live_experts[global_id] = Some((gate_up, down));
    }
    // Non-owned globals must point at an owned zero dummy whose matching
    // half satisfies the expert's sealed geometry. Each table is matched
    // independently: one dummy pair may back the whole table, or several
    // dummies may cover mixed layouts.
    for (global_id, metadata) in table.experts().iter().enumerate() {
        if metadata.owner_rank() == local_rank {
            continue;
        }
        let dummy_gate = zero_dummies.iter().find(|dummy| {
            weight_buf_ptr(&dummy.gate_up) == gate_up_ptr_entries[global_id]
                && bind_live_weight(metadata, &dummy.gate_up, true, global_id, "gate/up").is_ok()
        });
        let Some(dummy_gate) = dummy_gate else {
            return Err(invalid(format!(
                "compact non-owned expert {global_id} gate/up entry {:x} names no layout-compatible zero dummy",
                gate_up_ptr_entries[global_id]
            )));
        };
        let dummy_down = zero_dummies.iter().find(|dummy| {
            weight_buf_ptr(&dummy.down) == down_ptr_entries[global_id]
                && bind_live_weight(metadata, &dummy.down, false, global_id, "down").is_ok()
        });
        let Some(dummy_down) = dummy_down else {
            return Err(invalid(format!(
                "compact non-owned expert {global_id} down entry {:x} names no layout-compatible zero dummy",
                down_ptr_entries[global_id]
            )));
        };
        let gate_up = bind_live_weight(metadata, &dummy_gate.gate_up, true, global_id, "gate/up")?;
        let down = bind_live_weight(metadata, &dummy_down.down, false, global_id, "down")?;
        live_experts[global_id] = Some((gate_up, down));
    }
    let live_experts: Vec<(LiveWeightIdentity, LiveWeightIdentity)> = live_experts
        .into_iter()
        .enumerate()
        .map(|(global_id, entry)| {
            entry.ok_or_else(|| {
                invalid(format!(
                    "compact bind left expert {global_id} without a live identity"
                ))
            })
        })
        .collect::<Result<_, _>>()?;

    let contract_part = match table.execution_contract() {
        Some(contract) => contract.fingerprint(),
        None => format!("nocontract|n|{n_experts}"),
    };
    let mut canonical =
        format!("compact-live/v1|rank|{local_rank}|contract|{contract_part}|entries|");
    for global_id in 0..n_experts {
        let metadata = &table.experts()[global_id];
        canonical.push_str(&format!(
            "{global_id}:{}:{}:{:x}:{:x};",
            metadata.owner_rank(),
            metadata.local_slot(),
            gate_up_ptr_entries[global_id],
            down_ptr_entries[global_id]
        ));
    }
    canonical.push_str("|dummies|");
    for dummy in zero_dummies {
        canonical.push_str(&format!(
            "{:x}:{:x};",
            weight_buf_ptr(&dummy.gate_up),
            weight_buf_ptr(&dummy.down)
        ));
    }
    canonical.push_str(&format!(
        "|tables|{:x}:{}:{:x}:{}|awq|",
        gate_up_identity.ptr,
        gate_up_identity.byte_capacity,
        down_identity.ptr,
        down_identity.byte_capacity
    ));
    match (&down_awq_identity, down_awq_ptr_entries) {
        (Some(identity), Some(entries)) => canonical.push_str(&format!(
            "{:x}:{}:{};",
            identity.ptr,
            identity.byte_capacity,
            fingerprint_hex(&format!("{entries:x?}"))
        )),
        (None, None) => canonical.push_str("none;"),
        _ => {
            return Err(invalid(
                "compact down AWQ host entries and pointer table disagree",
            ));
        }
    }
    canonical.push_str("|tags|");
    match &dtype_tag_identity {
        Some(identity) => {
            canonical.push_str(&format!("{:x}:{};", identity.ptr, identity.byte_capacity))
        }
        None => canonical.push_str("none;"),
    }

    Ok(LiveMoeBinding {
        table_identity: table.identity,
        table_ptr: table.experts.as_ptr() as usize,
        table_len: table.experts.len(),
        experts: live_experts.into_boxed_slice(),
        gate_up_ptrs: gate_up_identity,
        down_ptrs: down_identity,
        down_awq_ptrs: down_awq_identity,
        dtype_tags: dtype_tag_identity,
        mapping_fingerprint: fingerprint_hex(&canonical),
    })
}

fn bind_live_weight(
    metadata: &ExpertMetadata,
    weight: &crate::families::gemv::WeightRef<'_>,
    gate_up: bool,
    index: usize,
    role: &str,
) -> Result<LiveWeightIdentity, DispatchError> {
    let resources = metadata.resources();
    let (rows, cols, dtype, bytes) = if gate_up {
        if let Some(resource) = resources.gate_up() {
            let (rows, cols) = resource_dims(resource, index, role)?;
            (rows, cols, resource.dtype(), resource.encoded_bytes())
        } else {
            let gate = resources.gate().ok_or_else(|| {
                invalid(format!("expert {index} separate resources have no gate"))
            })?;
            let up = resources
                .up()
                .ok_or_else(|| invalid(format!("expert {index} separate resources have no up")))?;
            let (gate_rows, gate_cols) = resource_dims(gate, index, "gate")?;
            let (up_rows, up_cols) = resource_dims(up, index, "up")?;
            if gate.dtype() != up.dtype() || gate_cols != up_cols {
                return Err(invalid(format!(
                    "expert {index} separate gate/up dtype or shape mismatch"
                )));
            }
            let rows = gate_rows
                .checked_add(up_rows)
                .ok_or_else(|| invalid(format!("expert {index} fused gate/up rows overflow")))?;
            let bytes = gate
                .encoded_bytes()
                .checked_add(up.encoded_bytes())
                .ok_or_else(|| {
                    invalid(format!("expert {index} separate gate/up bytes overflow"))
                })?;
            (rows, gate_cols, gate.dtype(), bytes)
        }
    } else {
        let resource = resources.down();
        let (rows, cols) = resource_dims(resource, index, role)?;
        (rows, cols, resource.dtype(), resource.encoded_bytes())
    };

    if weight.dtype != dtype {
        return Err(invalid(format!(
            "expert {index} live {role} dtype {:?} differs from {:?}",
            weight.dtype, dtype
        )));
    }
    if weight.m != rows || weight.k != cols {
        return Err(invalid(format!(
            "expert {index} live {role} shape [{}, {}] does not match [{rows}, {cols}]",
            weight.m, weight.k
        )));
    }
    let capacity = weight.buf.buf.size();
    if capacity != bytes {
        return Err(invalid(format!(
            "expert {index} live {role} byte capacity {capacity} does not equal {bytes}"
        )));
    }
    LiveWeightIdentity::capture(weight, &format!("expert {index} live {role}"))
}

fn resource_dims(
    resource: &ExpertResource,
    index: usize,
    role: &str,
) -> Result<(usize, usize), DispatchError> {
    if resource.shape().len() != 2 {
        return Err(invalid(format!(
            "expert {index} {role} logical shape {:?} is not two-dimensional",
            resource.shape()
        )));
    }
    Ok((resource.shape()[0], resource.shape()[1]))
}

fn table_has_mixed_dtypes(table: &ExpertTable) -> Result<bool, DispatchError> {
    let mut representative = None;
    let mut mixed = false;
    for (index, metadata) in table.experts().iter().enumerate() {
        let resources = metadata.resources();
        let gate_up = resources
            .gate_up()
            .or_else(|| resources.gate())
            .ok_or_else(|| invalid(format!("expert {index} has no gate/up resource")))?;
        let down = resources.down();
        if let Some(up) = resources.up() {
            if up.dtype() != gate_up.dtype() {
                return Err(invalid(format!(
                    "expert {index} separate gate/up dtype mismatch"
                )));
            }
        }
        let pair = (gate_up.dtype(), down.dtype());
        if representative.is_some_and(|expected| expected != pair) {
            mixed = true;
        } else {
            representative = Some(pair);
        }
    }
    Ok(mixed)
}

fn validate_live_binding(
    experts: &BoundMoeExperts<'_>,
    routed_experts: Option<&dyn RoutedExpertWeights>,
    gate_up_ptrs: &GpuTensor,
    down_ptrs: &GpuTensor,
    down_awq_ptrs: Option<&GpuTensor>,
    dtype_tags: Option<&GpuTensor>,
) -> Result<(), DispatchError> {
    let Some(live) = experts.cache.live.as_ref() else {
        return Err(invalid("expert binding cache has no live resource binding"));
    };
    if live.table_identity != experts.table.identity
        || live.table_ptr != experts.table.experts.as_ptr() as usize
        || live.table_len != experts.table.experts.len()
    {
        return Err(invalid(
            "expert live resource binding belongs to a different expert table",
        ));
    }
    let Some(routed_experts) = routed_experts else {
        return Err(invalid(
            "sealed MoE call has no live routed expert resources",
        ));
    };
    if routed_experts.len() != live.experts.len() {
        return Err(invalid(format!(
            "live routed expert table has {} records, expected {}",
            routed_experts.len(),
            live.experts.len()
        )));
    }
    for (index, (expected_gate_up, expected_down)) in live.experts.iter().enumerate() {
        let (gate_up, down) = routed_experts.get(index).ok_or_else(|| {
            invalid(format!(
                "live routed expert table is missing expert {index}"
            ))
        })?;
        expected_gate_up.matches(&gate_up, "live gate/up")?;
        expected_down.matches(&down, "live down")?;
    }
    live.gate_up_ptrs
        .matches(gate_up_ptrs, "gate/up pointer table")?;
    live.down_ptrs.matches(down_ptrs, "down pointer table")?;
    match (&live.down_awq_ptrs, down_awq_ptrs) {
        (Some(expected), Some(actual)) => expected.matches(actual, "down AWQ pointer table")?,
        (None, None) => {}
        (Some(_), None) => {
            return Err(invalid("down AWQ pointer table is missing from the call"));
        }
        (None, Some(_)) => {
            return Err(invalid("unexpected down AWQ pointer table in the call"));
        }
    }
    match (&live.dtype_tags, dtype_tags) {
        (Some(expected), Some(actual)) => expected.matches(actual, "dtype tag table")?,
        (None, None) => {}
        (Some(_), None) => return Err(invalid("dtype tag table is missing from the call")),
        (None, Some(_)) => return Err(invalid("unexpected dtype tag table in the call")),
    }
    Ok(())
}

fn validate_logical_shape(
    shape: &[usize],
    rows: usize,
    cols: usize,
    role: &str,
    expert: usize,
) -> Result<(), DispatchError> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| invalid(format!("expert {expert} {role} shape overflows")))?;
    let actual = checked_product(shape, &format!("expert {expert} {role} shape"))?;
    if actual != expected || (shape.len() == 2 && shape != [rows, cols]) {
        return Err(invalid(format!(
            "expert {expert} {role} logical shape {:?} does not match [{rows}, {cols}]",
            shape
        )));
    }
    Ok(())
}

fn expected_basis(
    experts: &BoundMoeExperts<'_>,
    representative: DType,
) -> Result<RotationPlan, DispatchError> {
    let basis = dtype_rotation_plan(representative);
    for (index, expert) in experts.records().iter().enumerate() {
        let resource = expert
            .resources()
            .gate_up()
            .or_else(|| expert.resources().gate())
            .ok_or_else(|| invalid(format!("expert {index} has no gate/up resource")))?;
        if resource.basis() != basis {
            return Err(invalid(format!(
                "expert {index} activation basis {:?} differs from {:?}",
                resource.basis(),
                basis
            )));
        }
    }
    Ok(basis)
}

fn validate_expert_records(records: &[ExpertMetadata]) -> Result<(), DispatchError> {
    if records.is_empty() {
        return Err(invalid("expert table is empty"));
    }
    let mut sources = HashMap::new();
    let mut source_bytes = HashMap::new();
    let mut aliases = Vec::new();
    let mut owner_slots = HashSet::new();
    let mut fingerprint: Option<&str> = None;
    for (expected_id, record) in records.iter().enumerate() {
        if record.global_id() != expected_id {
            return Err(invalid(format!(
                "expert records must be ordered by global id: expected {expected_id}, got {}",
                record.global_id()
            )));
        }
        if !owner_slots.insert((record.owner_rank(), record.local_slot())) {
            return Err(invalid(format!(
                "duplicate local expert slot ({}, {})",
                record.owner_rank(),
                record.local_slot()
            )));
        }
        for resource in record.resources().iter() {
            match fingerprint {
                None => fingerprint = Some(resource.source_fingerprint()),
                Some(expected) if expected != resource.source_fingerprint() => {
                    return Err(invalid(format!(
                        "expert source '{}' has inconsistent fingerprint",
                        resource.source_name()
                    )));
                }
                Some(_) => {}
            }
            if let Some(previous) = sources.get(resource.source_name()) {
                if resource.alias().is_some() || *previous != resource {
                    return Err(invalid(format!(
                        "expert source '{}' is repeated with different metadata or alias identity",
                        resource.source_name()
                    )));
                }
            } else {
                sources.insert(resource.source_name().to_owned(), resource);
                source_bytes.insert(resource.source_name().to_owned(), resource.encoded_bytes());
            }
            if let Some(alias) = resource.alias() {
                aliases.push((resource, alias));
            }
            validate_sidecars(resource.sidecars())?;
        }
    }
    for &(resource, alias) in &aliases {
        let owner_bytes = source_bytes.get(alias.owner()).ok_or_else(|| {
            invalid(format!(
                "alias '{}' refers to missing owner '{}'",
                resource.source_name(),
                alias.owner()
            ))
        })?;
        let end = alias
            .byte_offset()
            .checked_add(alias.byte_len())
            .ok_or_else(|| invalid("alias byte range overflows"))?;
        if end > *owner_bytes {
            return Err(invalid(format!(
                "alias '{}' range [{}, {}) exceeds owner '{}' bytes {}",
                resource.source_name(),
                alias.byte_offset(),
                end,
                alias.owner(),
                owner_bytes
            )));
        }
        if alias.owner() == resource.source_name() {
            return Err(invalid(format!(
                "alias '{}' refers to itself",
                resource.source_name()
            )));
        }
        if aliases
            .iter()
            .any(|(other, _)| other.source_name() == alias.owner())
        {
            return Err(invalid(format!(
                "alias '{}' refers to another alias; cyclic/indirect aliases are unsupported",
                resource.source_name()
            )));
        }
    }
    Ok(())
}

fn validate_resource_header(
    source_name: &str,
    source_fingerprint: &str,
    shape: &[usize],
    dtype: DType,
    encoded_bytes: usize,
    row_stride: usize,
    alignment: usize,
    basis: RotationPlan,
) -> Result<(), DispatchError> {
    if source_name.is_empty() {
        return Err(invalid("expert source name is empty"));
    }
    if source_fingerprint.is_empty() {
        return Err(invalid(format!(
            "expert source '{source_name}' fingerprint is empty"
        )));
    }
    if shape.is_empty() || shape.iter().any(|dimension| *dimension == 0) {
        return Err(invalid(format!(
            "expert source '{source_name}' has an empty logical shape"
        )));
    }
    if encoded_bytes == 0 || row_stride == 0 {
        return Err(invalid(format!(
            "expert source '{source_name}' has zero encoded bytes or row stride"
        )));
    }
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(invalid(format!(
            "expert source '{source_name}' alignment {alignment} is not a power of two"
        )));
    }
    if dtype_rotation_plan(dtype) != basis {
        return Err(invalid(format!(
            "expert source '{source_name}' basis {:?} does not match dtype {:?} ({:?})",
            basis,
            dtype,
            dtype_rotation_plan(dtype)
        )));
    }
    let rows = checked_product(&shape[..shape.len() - 1], "expert source row count")?;
    let minimum = rows
        .checked_mul(row_stride)
        .ok_or_else(|| invalid(format!("expert source '{source_name}' row range overflows")))?;
    if encoded_bytes < minimum {
        return Err(invalid(format!(
            "expert source '{source_name}' encoded bytes {encoded_bytes} are below row-stride requirement {minimum}"
        )));
    }
    Ok(())
}

fn validate_sidecars(sidecars: &[String]) -> Result<(), DispatchError> {
    let mut seen = HashSet::new();
    for sidecar in sidecars {
        if sidecar.is_empty() || !seen.insert(sidecar) {
            return Err(invalid(
                "expert sidecar identities must be nonempty and unique",
            ));
        }
    }
    Ok(())
}

fn validate_dtype_tag_table(
    tags: Option<&GpuTensor>,
    n_experts: usize,
    mixed: bool,
) -> Result<(), DispatchError> {
    match (tags, mixed) {
        (Some(tags), true) => {
            if tags.dtype != DType::Raw && tags.dtype != DType::F32 {
                return Err(invalid("expert dtype tags must be byte/raw or f32 slots"));
            }
            let capacity = tensor_capacity_bytes(tags)?;
            if capacity != n_experts {
                return Err(invalid(format!(
                    "expert dtype tag capacity {capacity} does not equal n_experts {n_experts}"
                )));
            }
            Ok(())
        }
        (None, true) => Err(invalid("mixed experts require a dtype tag table")),
        (Some(_), false) => Err(invalid("dtype tag table supplied for uniform experts")),
        (None, false) => Ok(()),
    }
}

fn validate_pointer_table(
    table: &GpuTensor,
    n_experts: usize,
    name: &str,
) -> Result<(), DispatchError> {
    if table.dtype != DType::Raw && table.dtype != DType::F32 {
        return Err(invalid(format!("{name} pointer table must be Raw or F32")));
    }
    let expected = n_experts
        .checked_mul(DEVICE_POINTER_BYTES)
        .ok_or_else(|| invalid(format!("{name} pointer table byte length overflows")))?;
    let capacity = tensor_capacity_bytes(table)?;
    if capacity != expected {
        return Err(invalid(format!(
            "{name} pointer table capacity {capacity} does not equal required {expected}"
        )));
    }
    Ok(())
}

fn require_elements(tensor: &GpuTensor, required: usize, name: &str) -> Result<(), DispatchError> {
    let capacity = checked_product(&tensor.shape, &format!("{name} shape"))?;
    if capacity < required {
        return Err(invalid(format!(
            "{name} capacity {capacity} is below required {required}"
        )));
    }
    Ok(())
}

fn tensor_capacity_bytes(tensor: &GpuTensor) -> Result<usize, DispatchError> {
    let elements = checked_product(&tensor.shape, "tensor capacity")?;
    elements
        .checked_mul(tensor.dtype.size())
        .ok_or_else(|| invalid("tensor byte capacity overflows"))
}

fn checked_product(values: &[usize], label: &str) -> Result<usize, DispatchError> {
    if values.is_empty() {
        return Err(invalid(format!("{label} is empty")));
    }
    values.iter().try_fold(1usize, |product, value| {
        if *value == 0 {
            return Err(invalid(format!("{label} contains a zero dimension")));
        }
        product
            .checked_mul(*value)
            .ok_or_else(|| invalid(format!("{label} overflows")))
    })
}

fn invalid(message: impl Into<String>) -> DispatchError {
    DispatchError::Hip(format!("sealed_moe: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    struct FakeWeight {
        buf: GpuTensor,
        dtype: DType,
        m: usize,
        k: usize,
        row_stride: usize,
    }

    struct FakeExpert {
        gate_up: FakeWeight,
        down: FakeWeight,
    }

    struct FakeRouted {
        experts: Vec<FakeExpert>,
    }

    impl RoutedExpertWeights for FakeRouted {
        fn len(&self) -> usize {
            self.experts.len()
        }

        fn get(
            &self,
            expert_idx: usize,
        ) -> Option<(
            crate::families::gemv::WeightRef<'_>,
            crate::families::gemv::WeightRef<'_>,
        )> {
            self.experts.get(expert_idx).map(|expert| {
                (
                    crate::families::gemv::WeightRef {
                        buf: &expert.gate_up.buf,
                        dtype: expert.gate_up.dtype,
                        m: expert.gate_up.m,
                        k: expert.gate_up.k,
                        row_stride: expert.gate_up.row_stride,
                        rotation: None,
                        awq_scale: None,
                    },
                    crate::families::gemv::WeightRef {
                        buf: &expert.down.buf,
                        dtype: expert.down.dtype,
                        m: expert.down.m,
                        k: expert.down.k,
                        row_stride: expert.down.row_stride,
                        rotation: None,
                        awq_scale: None,
                    },
                )
            })
        }
    }

    struct LiveFixture {
        routed: FakeRouted,
        gate_up_ptrs: GpuTensor,
        down_ptrs: GpuTensor,
        down_awq_ptrs: Option<GpuTensor>,
        dtype_tags: Option<GpuTensor>,
    }

    fn live_tensor(ptr: usize, bytes: usize, shape: &[usize], dtype: DType) -> GpuTensor {
        GpuTensor {
            buf: unsafe { hip_bridge::DeviceBuffer::from_raw(ptr as *mut std::ffi::c_void, bytes) },
            shape: shape.to_vec(),
            dtype,
        }
    }

    fn live_fixture(table: &ExpertTable, base: usize) -> LiveFixture {
        let experts = table
            .experts()
            .iter()
            .enumerate()
            .map(|(index, metadata)| {
                let resources = metadata.resources();
                let (gate_rows, gate_cols, gate_dtype, gate_bytes) =
                    if let Some(gate_up) = resources.gate_up() {
                        (
                            gate_up.shape()[0],
                            gate_up.shape()[1],
                            gate_up.dtype(),
                            gate_up.encoded_bytes(),
                        )
                    } else {
                        let gate = resources.gate().unwrap();
                        let up = resources.up().unwrap();
                        (
                            gate.shape()[0] + up.shape()[0],
                            gate.shape()[1],
                            gate.dtype(),
                            gate.encoded_bytes() + up.encoded_bytes(),
                        )
                    };
                let down = resources.down();
                FakeExpert {
                    gate_up: FakeWeight {
                        buf: live_tensor(
                            base + index * 0x1000,
                            gate_bytes,
                            &[gate_bytes],
                            DType::Raw,
                        ),
                        dtype: gate_dtype,
                        m: gate_rows,
                        k: gate_cols,
                        row_stride: 0,
                    },
                    down: FakeWeight {
                        buf: live_tensor(
                            base + 0x800 + index * 0x1000,
                            down.encoded_bytes(),
                            &[down.encoded_bytes()],
                            DType::Raw,
                        ),
                        dtype: down.dtype(),
                        m: down.shape()[0],
                        k: down.shape()[1],
                        row_stride: 0,
                    },
                }
            })
            .collect();
        let n = table.n_experts();
        LiveFixture {
            routed: FakeRouted { experts },
            gate_up_ptrs: live_tensor(
                base + 0x100000,
                n * DEVICE_POINTER_BYTES,
                &[2 * n],
                DType::F32,
            ),
            down_ptrs: live_tensor(
                base + 0x101000,
                n * DEVICE_POINTER_BYTES,
                &[2 * n],
                DType::F32,
            ),
            down_awq_ptrs: None,
            dtype_tags: None,
        }
    }

    fn bind_fixture(table: &ExpertTable, cache: &mut ExpertBindingCache, fixture: &LiveFixture) {
        cache
            .bind_live(
                table,
                &fixture.routed,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                fixture.down_awq_ptrs.as_ref(),
                fixture.dtype_tags.as_ref(),
            )
            .unwrap();
    }

    fn resource(name: &str, dtype: DType, shape: &[usize]) -> ExpertResource {
        let rows = shape[..shape.len() - 1].iter().product::<usize>();
        let bytes = rows * shape[shape.len() - 1];
        ExpertResource::new(
            name,
            "fixture-fingerprint",
            shape.to_vec(),
            dtype,
            bytes,
            shape[shape.len() - 1],
            16,
            dtype_rotation_plan(dtype),
        )
        .unwrap()
    }

    #[test]
    fn bound_construction_rejects_an_unbound_cache() {
        let table = table(2, DType::MQ4G256);
        let cache = table.prepare_binding(0, 1, 0).unwrap();
        let error = BoundMoeExperts::from_cache(&table, &cache).unwrap_err();
        assert!(format!("{error:?}").contains("no live resource binding"));
    }

    #[test]
    fn bound_construction_rejects_a_swapped_same_capacity_table_generation() {
        let table_a = table(2, DType::MQ4G256);
        let table_b = table(2, DType::MQ4G256);
        let mut cache = table_a.prepare_binding(0, 1, 0).unwrap();
        let fixture = live_fixture(&table_a, 0x30_0000);
        bind_fixture(&table_a, &mut cache, &fixture);
        assert!(BoundMoeExperts::from_cache(&table_b, &cache).is_err());
    }

    #[test]
    fn live_validation_rejects_swapped_same_capacity_pointer_table() {
        let table = table(2, DType::MQ4G256);
        let mut cache = table.prepare_binding(0, 1, 0).unwrap();
        let fixture = live_fixture(&table, 0x40_0000);
        bind_fixture(&table, &mut cache, &fixture);
        let bound = BoundMoeExperts::from_cache(&table, &cache).unwrap();
        let swapped_gate_up = live_tensor(
            0x51_0000,
            2 * DEVICE_POINTER_BYTES,
            &[2 * table.n_experts()],
            DType::F32,
        );
        assert!(validate_live_binding(
            &bound,
            Some(&fixture.routed),
            &swapped_gate_up,
            &fixture.down_ptrs,
            None,
            None,
        )
        .is_err());
    }

    #[test]
    fn live_validation_rejects_swapped_same_capacity_expert_buffer() {
        let table = table(2, DType::MQ4G256);
        let mut cache = table.prepare_binding(0, 1, 0).unwrap();
        let fixture = live_fixture(&table, 0x60_0000);
        bind_fixture(&table, &mut cache, &fixture);
        let bound = BoundMoeExperts::from_cache(&table, &cache).unwrap();
        let replacement = live_fixture(&table, 0x70_0000);
        assert!(validate_live_binding(
            &bound,
            Some(&replacement.routed),
            &fixture.gate_up_ptrs,
            &fixture.down_ptrs,
            None,
            None,
        )
        .is_err());
    }

    #[test]
    fn live_validation_allows_reuse_of_the_exact_bound_resources() {
        let table = table(2, DType::MQ4G256);
        let mut cache = table.prepare_binding(0, 1, 0).unwrap();
        let fixture = live_fixture(&table, 0x80_0000);
        bind_fixture(&table, &mut cache, &fixture);
        let bound = BoundMoeExperts::from_cache(&table, &cache).unwrap();
        for _ in 0..2 {
            validate_live_binding(
                &bound,
                Some(&fixture.routed),
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
            )
            .unwrap();
        }
    }

    fn table(n: usize, dtype: DType) -> ExpertTable {
        let records = (0..n)
            .map(|id| {
                ExpertMetadata::new(
                    id,
                    0,
                    id,
                    ExpertResources::fused(
                        resource(&format!("gate_up_{id}"), dtype, &[8, 4]),
                        resource(&format!("down_{id}"), dtype, &[4, 4]),
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect();
        ExpertTable::new(records).unwrap()
    }

    #[test]
    fn mixed_dtype_with_common_basis_is_admitted_and_basis_mismatch_is_rejected() {
        let records = vec![
            ExpertMetadata::new(
                0,
                0,
                0,
                ExpertResources::fused(
                    resource("gu0", DType::MQ4G256, &[8, 4]),
                    resource("dn0", DType::MQ4G256, &[4, 4]),
                )
                .unwrap(),
            )
            .unwrap(),
            ExpertMetadata::new(
                1,
                0,
                1,
                ExpertResources::fused(
                    resource("gu1", DType::MQ6G256, &[8, 4]),
                    resource("dn1", DType::MQ6G256, &[4, 4]),
                )
                .unwrap(),
            )
            .unwrap(),
        ];
        let table = ExpertTable::new(records).unwrap();
        assert_eq!(table.n_experts(), 2);
        let bad = ExpertResource::new(
            "bad",
            "fp:bad",
            vec![8, 4],
            DType::F32,
            32,
            4,
            16,
            RotationPlan::None,
        )
        .unwrap();
        let bad_record = ExpertMetadata::new(
            0,
            0,
            0,
            ExpertResources::fused(bad, resource("bad_dn", DType::MQ4G256, &[4, 4])).unwrap(),
        );
        assert!(bad_record.is_err());
    }

    #[test]
    fn shape_and_capacity_checks_are_checked_without_gpu_work() {
        let table = table(2, DType::MQ4G256);
        let mut cache = table.prepare_binding(0, 1, 0).unwrap();
        let fixture = live_fixture(&table, 0x10_0000);
        bind_fixture(&table, &mut cache, &fixture);
        let bound = BoundMoeExperts::from_cache(&table, &cache).unwrap();
        assert_eq!(bound.local_expert_ids(), &[0, 1]);
        let too_small = GpuTensor::null_for_test();
        assert!(require_elements(&too_small, 1, "test").is_err());
    }

    #[test]
    fn aliases_are_range_checked_and_combine_policy_is_explicit() {
        let owner = resource("owner", DType::MQ4G256, &[8, 4]);
        let alias = resource("alias", DType::MQ4G256, &[8, 4])
            .with_alias(ResourceAlias::new("owner", 0, 16).unwrap())
            .unwrap();
        let records = vec![
            ExpertMetadata::new(
                0,
                0,
                0,
                ExpertResources::fused(owner.clone(), resource("down", DType::MQ4G256, &[4, 4]))
                    .unwrap(),
            )
            .unwrap(),
            ExpertMetadata::new(
                1,
                0,
                1,
                ExpertResources::fused(alias, resource("down1", DType::MQ4G256, &[4, 4])).unwrap(),
            )
            .unwrap(),
        ];
        assert!(ExpertTable::new(records).is_ok());
        let out_of_range = resource("alias_bad", DType::MQ4G256, &[8, 4])
            .with_alias(ResourceAlias::new("owner", 31, 4).unwrap())
            .unwrap();
        let bad = ExpertTable::new(vec![
            ExpertMetadata::new(
                0,
                0,
                0,
                ExpertResources::fused(owner, resource("down2", DType::MQ4G256, &[4, 4])).unwrap(),
            )
            .unwrap(),
            ExpertMetadata::new(
                1,
                0,
                1,
                ExpertResources::fused(out_of_range, resource("down3", DType::MQ4G256, &[4, 4]))
                    .unwrap(),
            )
            .unwrap(),
        ]);
        assert!(bad.is_err());
    }

    #[test]
    fn paro_separate_gate_up_uses_each_half_of_fused_rows() {
        let gate = resource("paro_gate", DType::ParoQ4G128, &[4, 4]);
        let up = resource("paro_up", DType::ParoQ4G128, &[4, 4]);
        let down = resource("paro_down", DType::ParoQ4G128, &[4, 4]);
        let table = ExpertTable::new(vec![ExpertMetadata::new(
            0,
            0,
            0,
            ExpertResources::separate(gate, up, down).unwrap(),
        )
        .unwrap()])
        .unwrap();
        let mut cache = table.prepare_binding(0, 1, 0).unwrap();
        let fixture = live_fixture(&table, 0x20_0000);
        bind_fixture(&table, &mut cache, &fixture);
        let bound = BoundMoeExperts::from_cache(&table, &cache).unwrap();

        validate_expert_shape_and_dtype(
            &bound,
            1,
            8,
            4,
            4,
            4,
            DType::ParoQ4G128,
            DType::ParoQ4G128,
            false,
            None,
            None,
        )
        .unwrap();
        let resources = bound.expert(0).unwrap().resources();
        assert_eq!(resources.gate().unwrap().source_name(), "paro_gate");
        assert_eq!(resources.up().unwrap().source_name(), "paro_up");
    }

    #[test]
    fn grouped_prefill_capacity_accepts_just_fit_and_rejects_one_row_small() {
        let total_slots = 17;
        let n_experts = 2;
        let bound = checked_grouped_m_total_bound(total_slots, n_experts, GROUPED_BLOCK_M).unwrap();
        assert_eq!(bound, 48);
        assert!(validate_grouped_m_total_max(total_slots, n_experts, bound).is_ok());
        assert!(validate_grouped_m_total_max(total_slots, n_experts, bound - 1).is_err());
        assert!(checked_grouped_m_total_bound(usize::MAX, n_experts, GROUPED_BLOCK_M).is_err());
    }

    #[test]
    fn route_receipt_rejects_another_invocation() {
        let indices = GpuTensor::null_for_test();
        let weights = GpuTensor::null_for_test();
        let expected = MoeRouteReceipt {
            invocation: 17,
            protocol: MoeProtocol::GroupedPrefill,
            router: MoeRouterInput::PrecomputedSoftmaxTopK,
            n_experts: 4,
            k_top: 2,
            scores: 0,
            normalized: false,
            indices: &indices,
            weights: &weights,
        };
        let other_invocation = MoeRouteReceipt {
            invocation: 18,
            protocol: expected.protocol,
            router: expected.router,
            n_experts: expected.n_experts,
            k_top: expected.k_top,
            scores: 0,
            normalized: false,
            indices: expected.indices,
            weights: expected.weights,
        };
        assert!(validate_route_receipt_pair(&expected, &other_invocation).is_err());
    }

    fn ep_table(n: usize, rank_count: usize, dtype: DType) -> ExpertTable {
        let mut owned_so_far = vec![0usize; rank_count];
        let records = (0..n)
            .map(|id| {
                let owner = id % rank_count;
                let slot = owned_so_far[owner];
                owned_so_far[owner] += 1;
                ExpertMetadata::new(
                    id,
                    owner,
                    slot,
                    ExpertResources::fused(
                        resource(&format!("gate_up_{id}"), dtype, &[8, 4]),
                        resource(&format!("down_{id}"), dtype, &[4, 4]),
                    )
                    .unwrap(),
                )
                .unwrap()
            })
            .collect();
        ExpertTable::new(records).unwrap()
    }

    fn compact_weight(
        buf: &GpuTensor,
        dtype: DType,
        m: usize,
        k: usize,
    ) -> crate::families::gemv::WeightRef<'_> {
        crate::families::gemv::WeightRef {
            buf,
            dtype,
            m,
            k,
            row_stride: 0,
            rotation: None,
            awq_scale: None,
        }
    }

    struct CompactFixture {
        local_gate_up: Vec<GpuTensor>,
        local_down: Vec<GpuTensor>,
        dummy_gate_up: GpuTensor,
        dummy_down: GpuTensor,
        gate_entries: Vec<usize>,
        down_entries: Vec<usize>,
        gate_up_ptrs: GpuTensor,
        down_ptrs: GpuTensor,
    }

    /// Fixture for one rank of a stride EP table: owned globals in slot
    /// order get distinct local tensors, every other global entry names the
    /// shared zero dummy pair.
    fn compact_fixture(
        table: &ExpertTable,
        rank: usize,
        base: usize,
        dtype: DType,
    ) -> CompactFixture {
        let owned: Vec<usize> = table
            .experts()
            .iter()
            .filter(|record| record.owner_rank() == rank)
            .map(|record| record.global_id())
            .collect();
        assert!(!owned.is_empty());
        let local_gate_up = owned
            .iter()
            .enumerate()
            .map(|(slot, _)| live_tensor(base + slot * 0x1000, 32, &[32], DType::Raw))
            .collect::<Vec<_>>();
        let local_down = owned
            .iter()
            .enumerate()
            .map(|(slot, _)| live_tensor(base + 0x800 + slot * 0x1000, 16, &[16], DType::Raw))
            .collect::<Vec<_>>();
        let dummy_gate_up = live_tensor(base + 0x9000, 32, &[32], DType::Raw);
        let dummy_down = live_tensor(base + 0x9800, 16, &[16], DType::Raw);
        let _ = dtype;
        let gate_entries = (0..table.n_experts())
            .map(|global| {
                table
                    .experts()
                    .iter()
                    .find(|record| record.global_id() == global)
                    .and_then(|record| {
                        (record.owner_rank() == rank).then(|| {
                            owned
                                .iter()
                                .position(|&id| id == global)
                                .map(|slot| local_gate_up[slot].buf.as_ptr() as usize)
                        })?
                    })
                    .unwrap_or_else(|| dummy_gate_up.buf.as_ptr() as usize)
            })
            .collect::<Vec<_>>();
        let down_entries = (0..table.n_experts())
            .map(|global| {
                table
                    .experts()
                    .iter()
                    .find(|record| record.global_id() == global)
                    .and_then(|record| {
                        (record.owner_rank() == rank).then(|| {
                            owned
                                .iter()
                                .position(|&id| id == global)
                                .map(|slot| local_down[slot].buf.as_ptr() as usize)
                        })?
                    })
                    .unwrap_or_else(|| dummy_down.buf.as_ptr() as usize)
            })
            .collect::<Vec<_>>();
        let n = table.n_experts();
        CompactFixture {
            local_gate_up,
            local_down,
            dummy_gate_up,
            dummy_down,
            gate_entries,
            down_entries,
            gate_up_ptrs: live_tensor(
                base + 0x100000,
                n * DEVICE_POINTER_BYTES,
                &[2 * n],
                DType::F32,
            ),
            down_ptrs: live_tensor(
                base + 0x101000,
                n * DEVICE_POINTER_BYTES,
                &[2 * n],
                DType::F32,
            ),
        }
    }

    fn compact_locals<'a>(fixture: &'a CompactFixture, dtype: DType) -> Vec<CompactLiveWeight<'a>> {
        fixture
            .local_gate_up
            .iter()
            .zip(fixture.local_down.iter())
            .map(|(gate_up, down)| CompactLiveWeight {
                gate_up: compact_weight(gate_up, dtype, 8, 4),
                down: compact_weight(down, dtype, 4, 4),
            })
            .collect()
    }

    fn compact_dummies<'a>(
        fixture: &'a CompactFixture,
        dtype: DType,
    ) -> Vec<CompactLiveWeight<'a>> {
        vec![CompactLiveWeight {
            gate_up: compact_weight(&fixture.dummy_gate_up, dtype, 8, 4),
            down: compact_weight(&fixture.dummy_down, dtype, 4, 4),
        }]
    }

    fn bind_compact_ok(
        table: &ExpertTable,
        cache: &mut ExpertBindingCache,
        fixture: &CompactFixture,
        dtype: DType,
        device: i32,
    ) {
        let locals = compact_locals(fixture, dtype);
        let dummies = compact_dummies(fixture, dtype);
        cache
            .bind_live_compact(
                table,
                &locals,
                &fixture.gate_entries,
                &fixture.down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &dummies,
                device,
            )
            .unwrap();
    }

    #[test]
    fn compact_bind_rank0_and_rank1_succeed_with_slot_ordered_lists() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut cache0 = table.prepare_binding(0, 2, 3).unwrap();
        let mut cache1 = table.prepare_binding(1, 2, 5).unwrap();
        assert_eq!(cache0.local_expert_ids(), &[0, 2]);
        assert_eq!(cache1.local_expert_ids(), &[1, 3]);
        let fix0 = compact_fixture(&table, 0, 0xA0_0000, dtype);
        let fix1 = compact_fixture(&table, 1, 0xB0_0000, dtype);
        bind_compact_ok(&table, &mut cache0, &fix0, dtype, 3);
        bind_compact_ok(&table, &mut cache1, &fix1, dtype, 5);
        assert!(cache0.is_live_bound() && cache1.is_live_bound());
        // Same uploaded mapping shape on both ranks retains the same
        // fingerprint only when the entries agree; here the local bases
        // differ per rank, so the fingerprints must differ.
        assert_ne!(cache0.mapping_fingerprint(), cache1.mapping_fingerprint());
        assert!(cache0.mapping_fingerprint().is_some());
        BoundMoeExperts::from_cache(&table, &cache0).unwrap();
        BoundMoeExperts::from_cache(&table, &cache1).unwrap();
    }

    #[test]
    fn compact_bind_same_mapping_retains_same_fingerprint() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut first = table.prepare_binding(0, 2, 3).unwrap();
        let mut second = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0xC0_0000, dtype);
        bind_compact_ok(&table, &mut first, &fixture, dtype, 3);
        bind_compact_ok(&table, &mut second, &fixture, dtype, 3);
        assert_eq!(first.mapping_fingerprint(), second.mapping_fingerprint());
    }

    #[test]
    fn compact_bind_rejects_missing_local_expert() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut cache = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0xD0_0000, dtype);
        let locals = compact_locals(&fixture, dtype);
        let dummies = compact_dummies(&fixture, dtype);
        let error = cache
            .bind_live_compact(
                &table,
                &locals[..1],
                &fixture.gate_entries,
                &fixture.down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &dummies,
                3,
            )
            .unwrap_err();
        assert!(format!("{error:?}").contains("slots"), "{error:?}");
        assert!(!cache.is_live_bound());
    }

    #[test]
    fn compact_bind_rejects_misordered_local_experts() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut cache = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0xE0_0000, dtype);
        let mut locals = compact_locals(&fixture, dtype);
        locals.swap(0, 1);
        let dummies = compact_dummies(&fixture, dtype);
        let error = cache
            .bind_live_compact(
                &table,
                &locals,
                &fixture.gate_entries,
                &fixture.down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &dummies,
                3,
            )
            .unwrap_err();
        assert!(format!("{error:?}").contains("does not point"), "{error:?}");
        assert!(!cache.is_live_bound());
    }

    #[test]
    fn compact_bind_rejects_owned_pointer_mismatch() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut cache = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0xF0_0000, dtype);
        let locals = compact_locals(&fixture, dtype);
        let dummies = compact_dummies(&fixture, dtype);
        let mut gate_entries = fixture.gate_entries.clone();
        gate_entries[0] = gate_entries[0].wrapping_add(0x40);
        let error = cache
            .bind_live_compact(
                &table,
                &locals,
                &gate_entries,
                &fixture.down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &dummies,
                3,
            )
            .unwrap_err();
        assert!(format!("{error:?}").contains("owned expert 0"), "{error:?}");
        assert!(!cache.is_live_bound());
    }

    #[test]
    fn compact_bind_rejects_non_owned_non_dummy_pointer() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut cache = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0x1_0000, dtype);
        let locals = compact_locals(&fixture, dtype);
        let dummies = compact_dummies(&fixture, dtype);
        let mut down_entries = fixture.down_entries.clone();
        down_entries[1] = 0xdead_beef;
        let error = cache
            .bind_live_compact(
                &table,
                &locals,
                &fixture.gate_entries,
                &down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &dummies,
                3,
            )
            .unwrap_err();
        assert!(format!("{error:?}").contains("zero dummy"), "{error:?}");
        assert!(!cache.is_live_bound());
    }

    #[test]
    fn compact_bind_rejects_wrong_dummy_layout() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut cache = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0x2_0000, dtype);
        let locals = compact_locals(&fixture, dtype);
        let bad_dummy = CompactLiveWeight {
            gate_up: compact_weight(&fixture.dummy_gate_up, dtype, 7, 4),
            down: compact_weight(&fixture.dummy_down, dtype, 4, 4),
        };
        let error = cache
            .bind_live_compact(
                &table,
                &locals,
                &fixture.gate_entries,
                &fixture.down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &[bad_dummy],
                3,
            )
            .unwrap_err();
        assert!(format!("{error:?}").contains("zero dummy"), "{error:?}");
        assert!(!cache.is_live_bound());
    }

    #[test]
    fn compact_bind_rejects_wrong_device_and_stale_table() {
        let dtype = DType::MQ4G256;
        let table = ep_table(4, 2, dtype);
        let mut cache = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0x3_0000, dtype);
        let locals = compact_locals(&fixture, dtype);
        let dummies = compact_dummies(&fixture, dtype);
        let error = cache
            .bind_live_compact(
                &table,
                &locals,
                &fixture.gate_entries,
                &fixture.down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &dummies,
                9,
            )
            .unwrap_err();
        assert!(format!("{error:?}").contains("device"), "{error:?}");
        let other = ep_table(4, 2, dtype);
        let error = cache
            .bind_live_compact(
                &other,
                &locals,
                &fixture.gate_entries,
                &fixture.down_entries,
                None,
                &fixture.gate_up_ptrs,
                &fixture.down_ptrs,
                None,
                None,
                &dummies,
                3,
            )
            .unwrap_err();
        assert!(
            format!("{error:?}").contains("different expert table"),
            "{error:?}"
        );
        assert!(!cache.is_live_bound());
    }

    #[test]
    fn local_expert_ids_follow_local_slot_order_not_global_order() {
        let records = vec![
            ExpertMetadata::new(
                0,
                0,
                1,
                ExpertResources::fused(
                    resource("gu0", DType::MQ4G256, &[8, 4]),
                    resource("dn0", DType::MQ4G256, &[4, 4]),
                )
                .unwrap(),
            )
            .unwrap(),
            ExpertMetadata::new(
                1,
                0,
                0,
                ExpertResources::fused(
                    resource("gu1", DType::MQ4G256, &[8, 4]),
                    resource("dn1", DType::MQ4G256, &[4, 4]),
                )
                .unwrap(),
            )
            .unwrap(),
        ];
        let table = ExpertTable::new(records).unwrap();
        let cache = table.prepare_binding(0, 1, 0).unwrap();
        assert_eq!(cache.local_expert_ids(), &[1, 0]);
        assert_eq!(cache.local_slot(0), Some(1));
        assert_eq!(cache.local_slot(1), Some(0));
    }

    #[test]
    fn execution_contract_fingerprint_and_canonical_predicate() {
        let rows = ["gate", "up", "down"]
            .into_iter()
            .map(|name| ContractCollectiveRow {
                name: name.into(),
                layer: 0,
                hint: ContractCollectiveHint::AllReduce {
                    kind: ContractAxis::Ep,
                },
            })
            .collect::<Vec<_>>();
        let contract = ExpertExecutionContract::new(
            "ffn",
            Some(0),
            "fp-v1",
            7,
            vec![3, 5],
            ContractParallelism::ExpertParallel,
            ContractAssignment::Stride,
            vec![0, 1, 0, 1],
            vec![0, 0, 1, 1],
            CANONICAL_EP_EXECUTION,
            rows,
        )
        .unwrap();
        let fingerprint = contract.fingerprint();
        assert!(fingerprint.contains("sealed-ep/v1"));
        assert!(fingerprint.contains(CANONICAL_EP_EXECUTION));
        assert!(fingerprint.contains("owners|0,1,0,1"));
        assert!(contract.is_canonical_ep());
        let table = ep_table(4, 2, DType::MQ4G256);
        let table = table.with_execution_contract(contract.clone()).unwrap();
        assert_eq!(table.execution_contract(), Some(&contract));
        let mut cache = table.prepare_binding(0, 2, 3).unwrap();
        let fixture = compact_fixture(&table, 0, 0x4_0000, DType::MQ4G256);
        bind_compact_ok(&table, &mut cache, &fixture, DType::MQ4G256, 3);
        let viewed = BoundMoeExperts::from_cache(&table, &cache).unwrap();
        assert_eq!(viewed.execution_contract(), Some(&contract));
        // Mismatched owner/slot maps are rejected at attach time.
        let bad = ExpertExecutionContract::new(
            "ffn",
            Some(0),
            "fp-v1",
            7,
            vec![3, 5],
            ContractParallelism::ExpertParallel,
            ContractAssignment::Stride,
            vec![1, 0, 1, 0],
            vec![0, 0, 1, 1],
            CANONICAL_EP_EXECUTION,
            Vec::new(),
        )
        .unwrap();
        assert!(!bad.is_canonical_ep());
        assert!(table.with_execution_contract(bad).is_err());
        // A Single-style contract never authorizes the canonical combine.
        let single = ExpertExecutionContract::new(
            "ffn",
            Some(0),
            "fp-v1",
            7,
            vec![3],
            ContractParallelism::Single,
            ContractAssignment::Stride,
            vec![0, 0],
            vec![0, 1],
            "indexed-single",
            Vec::new(),
        )
        .unwrap();
        assert!(!single.is_canonical_ep());
    }
}
