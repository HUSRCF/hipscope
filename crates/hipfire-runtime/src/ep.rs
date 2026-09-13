// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Expert-parallel (EP) executor for the Ship 6 super-op substrate.
//!
//! Runs a lowered [`LayerProgram`] **replicated across N ranks** (every rank
//! runs every op on full, replicated attention/dense weights), special-casing
//! the `Moe` super-op with one of two EP combines, selected per binding via
//! [`ForwardBindings::ep_moe_combine_mode`] (all ranks must agree; mixed
//! modes refuse):
//!
//! - **Root-routed partial** (`EpMoeCombineMode::RootRoutedPartial`, Qwen
//!   plan-bound compact EP): the root seals SoftmaxTopK route production,
//!   folds owned experts plus the shared expert once into its zeroed partial,
//!   and returns a sealer-issued route-producer proof; the driver
//!   broadcasts the root top-k IDs **and** weights device-to-device into each
//!   rank's existing route buffers, non-roots seal routed contrib against the
//!   proof (no independent router; zero dummies read 0), then the existing
//!   all-reduce-sum plus residual add completes the combine.
//! - **Rank-partial all-reduce** (`EpMoeCombineMode::RankPartial`, the default
//!   for DeepSeek4/MiniMax):
//!
//!   1. zero each rank's routed partial,
//!   2. each rank computes ONLY its owned experts (+ the shared expert on rank 0)
//!      into its partial via [`ForwardBindings::run_moe_ep`] (non-owned experts
//!      read load-time zero-dummy weights → contribute 0),
//!   3. `all_reduce_sum_f32` the partials across ranks (canonical deterministic
//!      rooted peer reduce; RCCL/legacy-unrooted only via explicit opt-in),
//!   4. each rank adds the reduced partial into its residual stream via
//!      [`ForwardBindings::ep_add_into_residual`].
//!
//! Other super-ops ordinarily run **replicated** and unchanged. An architecture
//! may explicitly opt `Attend` into dense tensor parallelism through the
//! fail-closed `ForwardBindings` attention-TP hooks; the default remains false,
//! so Qwen, MiniMax, and every existing EP route retain replicated attention.
//!
//! Ordering: every op (zero, root/contrib or `run_moe_ep`, the collective, the
//! residual add, and the next layer's ops) is enqueued on each device's
//! `active_stream`, which is FIFO — so the per-rank sequence is correctly
//! ordered without host syncs between ops or layers. The decode driver syncs
//! once at the end before reading logits.
//!
//! This executor drives ONE layer's program across all ranks; the per-arch EP
//! driver loops layers (advancing each rank's per-layer binding state) the same
//! way the single-GPU lowered driver loops `run_layer_program`.

use crate::multi_gpu::{Gpus, PeerReduceScratchLease};
use hip_bridge::{DeviceBuffer, HipError};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::superop::{
    dispatch_super_op, EpMoeCombineMode, ForwardBindings, LayerProgram, SuperOpKind,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::GpuTensor;

fn hip_err(e: HipError) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

/// Ensure every device owns an `active_stream` (the stream the EP collectives
/// and per-rank work run on). Idempotent; safe to call before each layer.
pub fn ensure_rank_streams(gpus: &mut Gpus) -> Result<(), DispatchError> {
    for dev in gpus.devices.iter_mut() {
        dev.bind_thread().map_err(hip_err)?;
        if dev.active_stream.is_none() {
            dev.active_stream = Some(dev.hip.stream_create().map_err(hip_err)?);
        }
    }
    Ok(())
}

/// Decode all-reduce selection. The DEFAULT is the canonical deterministic
/// rooted peer reduce ([`crate::multi_gpu::Gpus::all_reduce_sum_f32_peer_rooted`]:
/// every rank observes the exact same left-associated sum `((p0+p1)+p2)+...`
/// over ranks in index order, regardless of N). Both non-canonical transports
/// stay reachable via explicit opt-in only:
/// - `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1` → legacy unrooted peer diagnostic,
/// - `HIPFIRE_EP_PEER_ALLREDUCE_DECODE=0` → RCCL.
/// Without peer access the canonical path cannot run, so it falls back to
/// RCCL (and the selection log line says so).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeArMode {
    Canonical,
    LegacyPeer,
    Rccl,
}

static DECODE_AR_MODE: std::sync::LazyLock<DecodeArMode> = std::sync::LazyLock::new(|| {
    match hipfire_config::developer_var("HIPFIRE_EP_PEER_ALLREDUCE_DECODE").as_deref() {
        Ok("1") => DecodeArMode::LegacyPeer,
        Ok("0") => DecodeArMode::Rccl,
        _ => DecodeArMode::Canonical,
    }
});

fn all_reduce_sum_f32_decode(
    gpus: &mut Gpus,
    refs: &[&DeviceBuffer],
    count: usize,
    peer_lease: Option<&PeerReduceScratchLease>,
) -> Result<(), DispatchError> {
    // Batch-owned lease: fixed PeerRootedF32 contract, same ascending-rank
    // arithmetic as prefill. Bypasses decode mode selection entirely.
    if let Some(lease) = peer_lease {
        return gpus
            .all_reduce_sum_f32_peer_rooted_leased(lease, refs, count)
            .map_err(hip_err);
    }
    let mode = *DECODE_AR_MODE;
    let use_rooted = mode == DecodeArMode::Canonical && gpus.peer_access_enabled;
    // Selection log line: names the active path (once per process).
    static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOGGED.get_or_init(|| {
        let path = match mode {
            DecodeArMode::Canonical if gpus.peer_access_enabled => {
                "canonical rooted-peer (fixed left fold over ranks; \
                 HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1 selects legacy unrooted peer, \
                 =0 selects RCCL)"
            }
            DecodeArMode::Canonical => {
                "RCCL (canonical rooted-peer unavailable: peer access disabled)"
            }
            DecodeArMode::LegacyPeer => "legacy unrooted peer (HIPFIRE_EP_PEER_ALLREDUCE_DECODE=1)",
            DecodeArMode::Rccl => "RCCL (HIPFIRE_EP_PEER_ALLREDUCE_DECODE=0)",
        };
        eprintln!("EP decode all-reduce: {path}");
    });
    if mode == DecodeArMode::LegacyPeer {
        gpus.all_reduce_sum_f32_peer(refs, count).map_err(hip_err)
    } else if use_rooted {
        gpus.all_reduce_sum_f32_peer_rooted(refs, count)
            .map_err(hip_err)
    } else {
        gpus.all_reduce_sum_f32(refs, count).map_err(hip_err)
    }
}

fn tp_peer_hc4_admitted<B: ForwardBindings>(gpus: &Gpus, bindings: &[B]) -> bool {
    gpus.devices.len() == 4
        && gpus.peer_access_enabled
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
        && bindings.iter().all(ForwardBindings::supports_tp_peer_hc4)
}

fn tp_peer_hc3_admitted<B: ForwardBindings>(gpus: &Gpus, bindings: &[B]) -> bool {
    gpus.devices.len() == 3
        && gpus.peer_access_enabled
        && gpus
            .devices
            .iter()
            .all(|device| device.arch_caps.is_gfx1201())
        && bindings.iter().all(ForwardBindings::supports_tp_peer_hc3)
}

/// Execute one lowered layer program across `gpus.devices.len()` EP ranks.
///
/// - `bindings[r]` drives rank `r`'s forward (it holds that rank's state /
///   weights / per-layer counters by reference, exactly like the single-GPU
///   `ForwardBindings` impl).
/// - `partials[r]` is rank `r`'s zeroed routed-output scratch, a contiguous f32
///   buffer of length `residual_dim` on `gpus.devices[r]`. The executor owns the
///   zero/all-reduce/add lifecycle; the binding only writes its owned-expert
///   contribution into it during `run_moe_ep`.
/// - `residual_dim` is the residual width (= hidden size) used for the partial
///   memset byte size and the all-reduce element count.
///
/// Every device must have an `active_stream` set ([`ensure_rank_streams`]).
pub fn run_layer_program_ep<B: ForwardBindings>(
    gpus: &mut Gpus,
    bindings: &mut [B],
    partials: &[GpuTensor],
    program: &LayerProgram,
    residual_dim: usize,
    peer_lease: Option<&PeerReduceScratchLease>,
) -> Result<(), DispatchError> {
    let n = gpus.devices.len();
    assert_eq!(
        bindings.len(),
        n,
        "run_layer_program_ep: bindings.len() != n_ranks"
    );
    assert_eq!(
        partials.len(),
        n,
        "run_layer_program_ep: partials.len() != n_ranks"
    );

    for op in program {
        if matches!(op.kind, SuperOpKind::Attend)
            && bindings.iter().any(ForwardBindings::attention_tp_enabled)
        {
            if !bindings.iter().all(ForwardBindings::attention_tp_enabled) {
                return Err(DispatchError::Hip(
                    "run_layer_program_ep: mixed attention-TP admission across ranks".into(),
                ));
            }

            // Each rank computes its local head/O-LoRA shard and stops before
            // the residual mix, leaving one hidden-width partial in the
            // architecture-owned attention output tensor.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                bindings[r].run_attend_ep(&mut gpus.devices[r], &ctx, &op.binding)?;
            }

            if tp_peer_hc3_admitted(gpus, bindings) {
                let peer_partials = bindings
                    .iter()
                    .map(|binding| {
                        let partial = binding.ep_attention_partial().ok_or_else(|| {
                            DispatchError::Hip(
                                "run_layer_program_ep: attention TP partial missing".into(),
                            )
                        })?;
                        Ok(GpuTensor {
                            buf: unsafe { partial.buf.alias() },
                            shape: partial.shape.clone(),
                            dtype: partial.dtype,
                        })
                    })
                    .collect::<Result<Vec<_>, DispatchError>>()?;
                let peers = [&peer_partials[0], &peer_partials[1], &peer_partials[2]];
                gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend_peer_hc3(&mut gpus.devices[r], peers)?;
                }
            } else if tp_peer_hc4_admitted(gpus, bindings) {
                // Borrow-independent aliases let the architecture hooks
                // consume all four peer pointers while each binding is
                // mutably advanced through its own HC residual mix.
                let peer_partials = bindings
                    .iter()
                    .map(|binding| {
                        let partial = binding.ep_attention_partial().ok_or_else(|| {
                            DispatchError::Hip(
                                "run_layer_program_ep: attention TP partial missing".into(),
                            )
                        })?;
                        Ok(GpuTensor {
                            buf: unsafe { partial.buf.alias() },
                            shape: partial.shape.clone(),
                            dtype: partial.dtype,
                        })
                    })
                    .collect::<Result<Vec<_>, DispatchError>>()?;
                let peers = [
                    &peer_partials[0],
                    &peer_partials[1],
                    &peer_partials[2],
                    &peer_partials[3],
                ];
                gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend_peer_hc4(&mut gpus.devices[r], peers)?;
                }
            } else {
                // Sum the input-column-sharded output projection directly in
                // its destination tensor. No staging copy or extra scratch.
                let refs: Vec<&DeviceBuffer> = bindings
                    .iter()
                    .map(|binding| {
                        binding
                            .ep_attention_partial()
                            .map(|partial| &partial.buf)
                            .ok_or_else(|| {
                                DispatchError::Hip(
                                    "run_layer_program_ep: attention TP partial missing".into(),
                                )
                            })
                    })
                    .collect::<Result<_, _>>()?;
                all_reduce_sum_f32_decode(gpus, &refs, residual_dim, peer_lease)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend(&mut gpus.devices[r])?;
                }
            }
        } else if matches!(op.kind, SuperOpKind::Moe) {
            // Root-routed vs rank-partial is reported per binding; every rank
            // must agree. Mixed modes return Err — automatic fallback from
            // root-routed to rank partials is a review veto.
            let all_root_routed = bindings
                .iter()
                .all(|b| b.ep_moe_combine_mode() == EpMoeCombineMode::RootRoutedPartial);
            let all_partial = bindings
                .iter()
                .all(|b| b.ep_moe_combine_mode() == EpMoeCombineMode::RankPartial);
            if all_root_routed {
                run_moe_ep_root_routed(
                    gpus,
                    bindings,
                    &op.binding,
                    partials,
                    residual_dim,
                    peer_lease,
                )?;
            } else if all_partial {
                // 1. Zero each rank's routed partial on its own stream.
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    let stream = gpus.devices[r]
                        .active_stream
                        .as_ref()
                        .ok_or_else(|| DispatchError::Hip(format!(
                            "run_layer_program_ep: device {r} has no active_stream (call ensure_rank_streams)"
                        )))?;
                    gpus.devices[r]
                        .hip
                        .memset_async(&partials[r].buf, 0, residual_dim * 4, stream)
                        .map_err(hip_err)?;
                }

                // 2. Each rank computes its owned-expert routed partial (+ shared on
                //    rank 0 via skip_shared=false; ranks>0 skip the shared down).
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    let ctx = DispatchCtx::new(&gpus.devices[r]);
                    bindings[r].run_moe_ep(
                        &mut gpus.devices[r],
                        &ctx,
                        &op.binding,
                        &partials[r],
                        /* skip_shared = */ r != 0,
                    )?;
                }

                if tp_peer_hc3_admitted(gpus, bindings) {
                    let peers = [&partials[0], &partials[1], &partials[2]];
                    gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_moe_peer_hc3(&mut gpus.devices[r], peers)?;
                    }
                } else if tp_peer_hc4_admitted(gpus, bindings) {
                    let peers = [&partials[0], &partials[1], &partials[2], &partials[3]];
                    gpus.barrier_rank_streams_reuse().map_err(hip_err)?;
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_finish_moe_peer_hc4(&mut gpus.devices[r], peers)?;
                    }
                } else {
                    // 3. All-reduce-sum the partials across ranks (in-place, RCCL).
                    let refs: Vec<&DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
                    all_reduce_sum_f32_decode(gpus, &refs, residual_dim, peer_lease)?;

                    // 4. Fold the reduced partial into each residual stream.
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
                    }
                }
            } else {
                return Err(DispatchError::Hip(
                    "run_layer_program_ep: mixed EP MoE combine modes across ranks; refusing (no fallback from root-routed to rank partials)".into(),
                ));
            }
        } else {
            // Replicated op — every rank runs it unchanged on full weights.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                let ctx = DispatchCtx::new(&gpus.devices[r]);
                dispatch_super_op(&mut gpus.devices[r], &ctx, op, &mut bindings[r])?;
            }
        }
    }
    Ok(())
}

/// Maximum EP ranks the root-routed driver stages in host-stack storage.
/// Existing architectural bound: rank identity is a `u8` (the qwen35 loader
/// refuses wider EP shard identity) and rank sets are `u64` masks
/// (`Qwen35EpBatchReceipt` / `Qwen35BatchCompatibility`), so meshes past 64
/// ranks are not admittable upstream. The driver still checks `n` against
/// this bound fail-closed before ANY work. Host stack only (512 B worst
/// case) — never a GPU allocation, never a 4-only mesh assumption.
const MAX_EP_STACK_RANKS: usize = 64;

/// Root-routed EP MoE for one layer program op. Only entered when every rank
/// reports [`EpMoeCombineMode::RootRoutedPartial`].
///
/// State machine (decode; k is the flat top-k width, typically 8):
/// 1. Preflight partial capacities, root/non-root sealed roles, and static
///    contract/layer/hidden/k/n_exp agreement across ranks — fail-closed
///    **before** any scratch mutation. Contract identity is the load-bound
///    `u64` (`contract_id`); no per-token fingerprint String render.
/// 1b. Pure binding/params/proof preflight on EVERY rank (root builds the
///    actual seal inputs and returns a validation proof; each non-root
///    validates its actual seal inputs against it; enqueues NOTHING) —
///    still before zeroing ANY partial.
/// 2. Zero every rank's routed partial on its own stream.
/// 3. Rank 0 runs [`ForwardBindings::ep_run_moe_root`] (router + owned +
///    shared exactly once into its zeroed partial) and returns the opaque
///    [`MoeRouteProducerProof`](hipfire_dispatch::pipeline::sealed_moe::MoeRouteProducerProof).
/// 4. Re-borrow each rank's resident route buffers via a broadcast closure;
///    broadcast root top-k IDs **and** weights into them via
///    [`Gpus::broadcast_ep_route`] (no host D2H/H2D, no new staging, no
///    per-layer heap allocation).
/// 5. Every non-root runs [`ForwardBindings::ep_run_moe_contrib`] against the
///    proof (owned experts only; no independent router).
/// 6. Existing `all_reduce_sum_f32_decode` over the partials, then
///    [`ForwardBindings::ep_add_into_residual`] exactly once per rank.
///
/// Any error is fail-stop for the token: no retry, no fallback to the
/// independent-router rank-partial path.
fn run_moe_ep_root_routed<B: ForwardBindings>(
    gpus: &mut Gpus,
    bindings: &mut [B],
    op: &hipfire_dispatch::pipeline::superop::OpBinding,
    partials: &[GpuTensor],
    residual_dim: usize,
    peer_lease: Option<&PeerReduceScratchLease>,
) -> Result<(), DispatchError> {
    let n = gpus.devices.len();

    // Fail-closed rank bound for the host-stack staging in step 6 (and the
    // per-rank preflight loop in step 1b): checked before ANY work, so no
    // path below can index past the bounded staging.
    if n > MAX_EP_STACK_RANKS {
        return Err(DispatchError::Hip(format!(
            "run_layer_program_ep: root-routed EP rank count {n} exceeds host-stack bound {MAX_EP_STACK_RANKS}"
        )));
    }

    // ── 1. Preflight (no scratch mutation yet) ─────────────────────────────
    let need_bytes = residual_dim
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            DispatchError::Hip(
                "run_layer_program_ep: residual_dim*4 partial byte count overflow".into(),
            )
        })?;
    for r in 0..n {
        if partials[r].buf.size() < need_bytes {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} partial has {} bytes, needs {need_bytes}",
                partials[r].buf.size()
            )));
        }
    }

    let root_view = bindings[0].ep_moe_route_view().ok_or_else(|| {
        DispatchError::Hip(
            "run_layer_program_ep: root-routed EP rank 0 is missing its route view".into(),
        )
    })?;
    let root_contract_id = root_view.contract_id().ok_or_else(|| {
        DispatchError::Hip(
            "run_layer_program_ep: root-routed EP root has no execution contract".into(),
        )
    })?;
    if let Some(contract) = root_view.execution_contract() {
        if !contract.is_root_routed_ep() {
            return Err(DispatchError::Hip(
                "run_layer_program_ep: root-routed EP root contract is not root-routed".into(),
            ));
        }
    }
    if root_view.experts.local_rank() != 0 {
        return Err(DispatchError::Hip(format!(
            "run_layer_program_ep: root-routed EP rank 0 binding reports local_rank {}",
            root_view.experts.local_rank()
        )));
    }
    if root_view.experts.rank_count() != n {
        return Err(DispatchError::Hip(format!(
            "run_layer_program_ep: root-routed EP root rank_count {} != mesh {n}",
            root_view.experts.rank_count()
        )));
    }
    if root_view.k == 0 {
        return Err(DispatchError::Hip(
            "run_layer_program_ep: root-routed EP requires k > 0".into(),
        ));
    }
    if root_view.hidden != residual_dim {
        return Err(DispatchError::Hip(format!(
            "run_layer_program_ep: root-routed EP root hidden {} != residual_dim {residual_dim}",
            root_view.hidden
        )));
    }
    let root_layer = root_view.layer;
    let root_hidden = root_view.hidden;
    let root_k = root_view.k;
    let root_n_exp = root_view.n_exp;
    // End the root view borrow before the experts loop (and before re-borrow).
    drop(root_view);

    for r in 1..n {
        let view = bindings[r].ep_moe_route_view().ok_or_else(|| {
            DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} is missing its route view"
            ))
        })?;
        let contract_id = view.contract_id().ok_or_else(|| {
            DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} has no execution contract"
            ))
        })?;
        if contract_id != root_contract_id
            || view.layer != root_layer
            || view.hidden != root_hidden
            || view.k != root_k
            || view.n_exp != root_n_exp
        {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} disagrees with the root plan/dimensions"
            )));
        }
        if view.experts.local_rank() != r {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} binding reports local_rank {}",
                view.experts.local_rank()
            )));
        }
        if view.experts.rank_count() != n {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} rank_count {} != mesh {n}",
                view.experts.rank_count()
            )));
        }
    }

    // ── 1b. Actual binding/params/proof preflight (pure; enqueues nothing) ──
    // Every rank validates its real MoE seal inputs (bound experts, params,
    // proof adoption) BEFORE any partial is zeroed, so a bad rank fails
    // closed without mutating scratch. The proof below is static load-bound
    // validation metadata only; the authoritative proof still comes from
    // the root compute in step 3.
    {
        let ctx0 = DispatchCtx::new(&gpus.devices[0]);
        let preflight_proof =
            bindings[0].ep_preflight_moe_root(&gpus.devices[0], &ctx0, op, &partials[0])?;
        for r in 1..n {
            let ctx = DispatchCtx::new(&gpus.devices[r]);
            bindings[r].ep_preflight_moe_contrib(
                &gpus.devices[r],
                &ctx,
                op,
                &preflight_proof,
                &partials[r],
            )?;
        }
    }

    // ── 2. Zero every rank partial ─────────────────────────────────────────
    for r in 0..n {
        gpus.devices[r].bind_thread().map_err(hip_err)?;
        let stream = gpus.devices[r].active_stream.as_ref().ok_or_else(|| {
            DispatchError::Hip(format!(
                "run_layer_program_ep: device {r} has no active_stream (call ensure_rank_streams)"
            ))
        })?;
        gpus.devices[r]
            .hip
            .memset_async(&partials[r].buf, 0, need_bytes, stream)
            .map_err(hip_err)?;
    }

    // ── 3. Root: router + owned + shared exactly once → proof ──────────────
    gpus.devices[0].bind_thread().map_err(hip_err)?;
    let ctx0 = DispatchCtx::new(&gpus.devices[0]);
    let proof = bindings[0].ep_run_moe_root(&mut gpus.devices[0], &ctx0, op, &partials[0])?;

    // ── 4. Broadcast root IDs + weights into existing per-rank route bufs ──
    // No host ID pack, no extra staging, no per-layer heap allocation: the
    // broadcast closure re-borrows each rank's resident route buffers
    // directly from its binding (the `&'a GpuTensor` is copied out of the
    // view, so the returned `&DeviceBuffer`s outlive the view temporary).
    // Non-root bindings are untouched by the root compute, but the closure
    // API is infallible, so re-check every view here to stay fail-stop
    // (never a fallback); the expect below is unreachable after that
    // check on this single thread.
    for r in 1..n {
        if bindings[r].ep_moe_route_view().is_none() {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: root-routed EP rank {r} lost its route view after root"
            )));
        }
    }
    let root_view = bindings[0].ep_moe_route_view().ok_or_else(|| {
        DispatchError::Hip(
            "run_layer_program_ep: root-routed EP rank 0 lost its route view after root".into(),
        )
    })?;
    let root_ids: &GpuTensor = root_view.topk_ids;
    let root_weights: &GpuTensor = root_view.topk_weights;
    let root_k = root_view.k;
    gpus.broadcast_ep_route(
        &root_ids.buf,
        &root_weights.buf,
        |r| {
            let view = bindings[r].ep_moe_route_view().expect(
                "run_layer_program_ep: root-routed EP rank lost its route view after re-check",
            );
            let ids: &GpuTensor = view.topk_ids;
            let weights: &GpuTensor = view.topk_weights;
            (&ids.buf, &weights.buf)
        },
        root_k,
    )
    .map_err(hip_err)?;

    // ── 5. Non-root contrib (proof-bound; no independent router) ───────────
    for r in 1..n {
        gpus.devices[r].bind_thread().map_err(hip_err)?;
        let ctx = DispatchCtx::new(&gpus.devices[r]);
        bindings[r].ep_run_moe_contrib(&mut gpus.devices[r], &ctx, op, &proof, &partials[r])?;
    }

    // ── 6. All-reduce partials + residual add, once per rank ───────────────
    // Host-stack slice of buffer refs: no per-layer heap allocation. The
    // rank bound was checked fail-closed at the top of this function, so
    // every staged index below is live and `staged[..n]` is exactly the N
    // ranks. Stack only (64 refs worst case) — never a GPU allocation.
    let mut staged: [&DeviceBuffer; MAX_EP_STACK_RANKS] = [&partials[0].buf; MAX_EP_STACK_RANKS];
    for r in 1..n {
        staged[r] = &partials[r].buf;
    }
    all_reduce_sum_f32_decode(gpus, &staged[..n], residual_dim, peer_lease)?;
    for r in 0..n {
        gpus.devices[r].bind_thread().map_err(hip_err)?;
        bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
    }
    Ok(())
}
