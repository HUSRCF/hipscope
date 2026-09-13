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
//! - **Canonical sealed slot order** (`EpMoeCombineMode::CanonicalSlotOrder`,
//!   Qwen plan-bound compact EP): each rank runs the sealed per-rank `Step::Moe`
//!   leaving raw per-slot rows expanded (root folds shared first); the driver
//!   reads 8 root route IDs, gathers remote-owned raw rows to the root,
//!   runs the sealed `Step::MoeSlotCombine` once on the root (the existing
//!   `moe_down_combine_k8_batched` fold, same association as Single), then
//!   broadcast-overwrites the finalized residual to every rank.
//! - **Rank-partial all-reduce** (`EpMoeCombineMode::RankPartial`, the default
//!   for DeepSeek4/MiniMax and the explicit Qwen
//!   `HIPFIRE_EP_SLOT_COMBINE=0` legacy diagnostic):
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
//! Ordering: every op (zero, run_moe_ep, the collective, the residual add, and
//! the next layer's ops) is enqueued on each device's `active_stream`, which is
//! FIFO — so the per-rank sequence is correctly ordered without host syncs
//! between ops or layers. The decode driver syncs once at the end before
//! reading logits.
//!
//! This executor drives ONE layer's program across all ranks; the per-arch EP
//! driver loops layers (advancing each rank's per-layer binding state) the same
//! way the single-GPU lowered driver loops `run_layer_program`.

use crate::multi_gpu::Gpus;
use hip_bridge::{DeviceBuffer, HipError};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::sealed_moe::{
    seal_slot_combine, MoeRootRouteReceipt, MoeSlotGatherReceipt,
};
use hipfire_dispatch::pipeline::superop::{
    dispatch_super_op, EpMoeCombineMode, EpMoeSlotView, ForwardBindings, LayerProgram, SuperOpKind,
};
use hipfire_dispatch::pipeline::{execute_steps, Step};
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
) -> Result<(), DispatchError> {
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
                all_reduce_sum_f32_decode(gpus, &refs, residual_dim)?;
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    bindings[r].ep_finish_attend(&mut gpus.devices[r])?;
                }
            }
        } else if matches!(op.kind, SuperOpKind::Moe) {
            // Canonical vs rank-partial is reported per binding; every rank
            // must agree. Mixed modes return Err — automatic fallback from
            // canonical slots to rank partials is a review veto.
            let all_canonical = bindings
                .iter()
                .all(|b| b.ep_moe_combine_mode() == EpMoeCombineMode::CanonicalSlotOrder);
            let all_partial = bindings
                .iter()
                .all(|b| b.ep_moe_combine_mode() == EpMoeCombineMode::RankPartial);
            if all_canonical {
                run_moe_ep_canonical(gpus, bindings, &op.binding, partials)?;
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
                    all_reduce_sum_f32_decode(gpus, &refs, residual_dim)?;

                    // 4. Fold the reduced partial into each residual stream.
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
                    }
                }
            } else {
                return Err(DispatchError::Hip(
                    "run_layer_program_ep: mixed EP MoE combine modes across ranks; refusing (no fallback from canonical slots to rank partials)".into(),
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

/// Borrow every rank's canonical slot view. A missing view is fail-stop.
/// Called twice per MoE layer: before the non-root experts (for agreement +
/// root readback) and after (for gather/combine/broadcast). The two borrows
/// never overlap the `&mut` experts loop between them (NLL).
fn collect_slot_views<B: ForwardBindings>(
    bindings: &[B],
) -> Result<Vec<EpMoeSlotView<'_>>, DispatchError> {
    bindings
        .iter()
        .map(|b| {
            b.ep_moe_slot_view().ok_or_else(|| {
                DispatchError::Hip(
                    "run_layer_program_ep: canonical EP rank is missing its slot view".into(),
                )
            })
        })
        .collect::<Result<_, _>>()
}
/// Canonical sealed EP MoE for one layer program op. Only entered when every
/// rank reports [`EpMoeCombineMode::CanonicalSlotOrder`].
///
/// State machine (one token, decode-only):
/// 1. The root runs the sealed full `Step::Moe` (norm/router/top-k/shared +
///    indexed experts, same kernels as Single). Owned selected slots produce
///    raw `v[i]` into the root's `down_expanded`, non-owned slots read
///    load-time zero dummies; the shared expert folds into the root residual
///    first (`x0+s`). Routed combine is deferred. Non-roots run nothing yet.
/// 2. Synchronously read exactly 8 route IDs (32 bytes) from the root,
///    range-check them, and derive each slot's owner from the sealed root
///    plan (never `e % N`). These IDs are authoritative for every rank:
///    per-rank re-routing is never assumed (near-tie order flips across
///    devices are real).
/// 3. Seal the root-route receipt and hand the 8 IDs (as bytes) plus the
///    receipt to every non-root rank. Each rank installs its copy on its
///    OWN stream ahead of its expert kernels (same-stream FIFO is the whole
///    ordering contract — no cross-rank copies, waits, or host syncs).
/// 4. Every non-root runs the sealed experts-only call (norm/rotate plus
///    indexed experts over the distributed IDs; no router runs there) into
///    its own `down_expanded`, skipping shared (stays `x0`).
/// 5. Row-granular gather of remote-owned raw rows into the root's existing
///    `moe_down_expanded`. Gathered rows are RAW `v[i]` — the launcher
///    applies `w[i]` exactly once.
/// 6. Seal the gather receipt and run the root continuation as
///    `Step::MoeSlotCombine` through `execute_steps` (prevalidated before
///    launch): the existing `moe_down_combine_k8_batched` fold once, so the
///    root residual is exactly `(x0+s)+fold_slot_0_to_7(w[i]*v[i])` with the
///    same kernel variant and association as Single.
/// 7. Broadcast-overwrite the finalized root residual to every non-root.
///    Overwrite, never add: `x+(s+r) != (x+s)+r`.
///
/// Any error is fail-stop for the token: no retry, no fallback to rank
/// partials (a review veto), no combine or broadcast after a gather error.
/// `partials`/`residual_dim` are the legacy rank-partial scratch and are
/// untouched on this path.
fn run_moe_ep_canonical<B: ForwardBindings>(
    gpus: &mut Gpus,
    bindings: &mut [B],
    op: &hipfire_dispatch::pipeline::superop::OpBinding,
    partials: &[GpuTensor],
) -> Result<(), DispatchError> {
    let n = gpus.devices.len();
    // 1. Root sealed decode (norm/router/top-k/shared/gate/down-expanded).
    // The root's top-k IDs are authoritative for every rank; non-roots run
    // NOTHING here — their experts-only compute (step 8) runs on the root
    // IDs distributed in step 7, so per-rank re-ranking cannot diverge.
    // Canonical bindings ignore the routed partial; pass the legacy slot
    // through untouched.
    gpus.devices[0].bind_thread().map_err(hip_err)?;
    let ctx0 = DispatchCtx::new(&gpus.devices[0]);
    bindings[0].run_moe_ep(
        &mut gpus.devices[0],
        &ctx0,
        op,
        &partials[0],
        /* skip_shared = */ false,
    )?;
    // 2. Borrow every rank's slot view. A missing view is fail-stop. This
    // borrow ends before the `&mut` experts loop below (NLL); views are
    // re-collected afterwards for gather/combine/broadcast.
    let views = collect_slot_views(bindings)?;
    // 3. Cross-rank agreement: same sealed plan (fingerprint) and same
    // layer/dimensions/width on every rank.
    let root = views[0];
    let root_contract = root.execution_contract().ok_or_else(|| {
        DispatchError::Hip(
            "run_layer_program_ep: canonical EP root has no execution contract".into(),
        )
    })?;
    let root_fingerprint = root_contract.fingerprint();
    for (r, view) in views.iter().enumerate() {
        let fingerprint = view
            .execution_contract()
            .map(|contract| contract.fingerprint())
            .ok_or_else(|| {
                DispatchError::Hip(format!(
                    "run_layer_program_ep: canonical EP rank {r} has no execution contract"
                ))
            })?;
        if fingerprint != root_fingerprint
            || view.layer != root.layer
            || view.hidden != root.hidden
            || view.k != root.k
            || view.n_exp != root.n_exp
        {
            return Err(DispatchError::Hip(
                format!("run_layer_program_ep: canonical EP rank {r} disagrees with the root plan/dimensions"),
            ));
        }
    }
    if root.k != 8 {
        return Err(DispatchError::Hip(format!(
            "run_layer_program_ep: canonical EP requires route width k=8, got {}",
            root.k
        )));
    }
    // 4. Root route-ID readback: exactly 8 i32 (32 bytes). Only the root ran
    // the router, so its IDs are authoritative for every rank by construction.
    // `download_f32` is a bare null-stream D2H with no stream ordering, so the
    // root's active stream (which
    // carries this layer's router/top-k writes to `topk_ids`) must be
    // synchronized first — otherwise the readback races the top-k kernel
    // and gathers a stale route. (Gather/broadcast need no extra sync:
    // `boundary_copy` enqueues on the SRC rank's stream, FIFO behind that
    // rank's kernels, and `wait_boundary` orders the DST stream behind
    // the copy.)
    gpus.devices[0].bind_thread().map_err(hip_err)?;
    // Shared borrows only: the scrutinee holds `&active_stream` while the
    // arms reborrow `&hip`.
    match gpus.devices[0].active_stream.as_ref() {
        Some(stream) => gpus.devices[0]
            .hip
            .stream_synchronize(stream)
            .map_err(hip_err)?,
        None => gpus.devices[0].hip.device_synchronize().map_err(hip_err)?,
    }
    let ids_f32 = gpus.devices[0]
        .download_f32(root.topk_ids)
        .map_err(hip_err)?;
    if ids_f32.len() != 8 {
        return Err(DispatchError::Hip(format!(
            "run_layer_program_ep: canonical EP root route readback is {} IDs, expected 8",
            ids_f32.len()
        )));
    }
    let mut route_ids = [0usize; 8];
    for (slot, bits) in ids_f32.iter().enumerate() {
        // `moe_topk_indices` is `[k]` i32 stored in an f32 tensor (same byte
        // width; the producing kernel casts the buffer to int*).
        let id = bits.to_bits() as i32;
        if id < 0 || (id as usize) >= root.n_exp {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: canonical EP root route ID {id} (slot {slot}) is outside 0..{}",
                root.n_exp
            )));
        }
        route_ids[slot] = id as usize;
    }
    // 5. Owner lookup from the sealed root plan. Any selected expert has
    // exactly one owner; an unowned ID or an owner outside the mesh is a
    // plan violation, not a silent drop.
    let mut slot_owner = [0usize; 8];
    for (slot, &id) in route_ids.iter().enumerate() {
        let owner = root
            .experts
            .expert(id)
            .map(|record| record.owner_rank())
            .ok_or_else(|| {
                DispatchError::Hip(format!(
                    "run_layer_program_ep: canonical EP root route ID {id} names no planned expert"
                ))
            })?;
        if owner >= n {
            return Err(DispatchError::Hip(format!(
                "run_layer_program_ep: canonical EP owner {owner} of expert {id} is outside rank count {n}"
            )));
        }
        slot_owner[slot] = owner;
    }
    // 6. Root-route hash + receipt: binds this token's authoritative route
    // (plan, layer, width, IDs) for the non-root experts-only seals below
    // and for the gather receipt afterwards. The seal checks plan/layer/dim
    // identity; buffers alone are never routing authority.
    let mut route_hash: u64 = 0xcbf29ce484222325;
    for &id in &route_ids {
        route_hash ^= id as u64;
        route_hash = route_hash.wrapping_mul(0x100000001b3);
    }
    let route_receipt = MoeRootRouteReceipt {
        execution_fingerprint: root_fingerprint.clone(),
        layer: root.layer,
        k: root.k,
        n_exp: root.n_exp,
        route_hash,
    };
    // 7. Pack the authoritative IDs as little-endian i32 bytes for the
    // non-root install below (exact inverse of the i32-as-f32 readback:
    // `bits.to_bits() as i32`). Each rank's hook uploads its copy on its
    // OWN stream, FIFO ahead of its expert kernels — same-stream ordering
    // is the whole contract: no cross-rank copies, event waits, or host
    // syncs, and ranks overlap freely.
    let mut route_id_bytes = [0u8; 32];
    for (slot, &id) in route_ids.iter().enumerate() {
        route_id_bytes[4 * slot..4 * slot + 4].copy_from_slice(&(id as i32).to_ne_bytes());
    }
    // 8. Non-root experts-only compute (norm/rotate + ID install + indexed
    // experts; no router runs here). The seal admits only the receipt-bound
    // canonical combination; anything else fail-stops before any launch.
    for r in 1..n {
        gpus.devices[r].bind_thread().map_err(hip_err)?;
        let ctx = DispatchCtx::new(&gpus.devices[r]);
        bindings[r].ep_run_moe_experts(
            &mut gpus.devices[r],
            &ctx,
            op,
            &route_receipt,
            &route_id_bytes,
        )?;
    }
    // 8b. Re-borrow slot views for gather/combine/broadcast below. The
    // pre-experts borrows ended before the `&mut` loop (NLL); these fresh
    // borrows observe the post-experts buffers (installed IDs included).
    let views = collect_slot_views(bindings)?;
    let root = views[0];
    // 9. Gather remote-owned raw rows into the root's existing expanded
    // buffer (copy-only, no arithmetic, root-owned slots skip self-copy).
    gpus.gather_slots_to_root_f32(
        &root.raw_slots.buf,
        |r| &views[r].raw_slots.buf,
        &slot_owner,
        root.hidden,
    )
    .map_err(hip_err)?;
    // 10. Seal the gather receipt and run the root continuation through the
    // sealed Step boundary (prevalidated before launch).
    let receipt = MoeSlotGatherReceipt {
        execution_fingerprint: root_fingerprint,
        layer: root.layer,
        hidden: root.hidden,
        k: root.k,
        n_exp: root.n_exp,
        route_hash,
    };
    let call = seal_slot_combine(
        root.experts,
        receipt,
        &route_id_bytes,
        root.raw_slots,
        root.topk_weights,
        root.residual,
        root.hidden,
        root.k,
    )?;
    gpus.devices[0].bind_thread().map_err(hip_err)?;
    let ctx0 = DispatchCtx::new(&gpus.devices[0]);
    execute_steps(&mut gpus.devices[0], &ctx0, &[Step::MoeSlotCombine(call)])?;
    // 11. Broadcast-overwrite the finalized root residual. Destinations
    // overwrite, never add.
    gpus.broadcast_root_row_f32(&root.residual.buf, |r| &views[r].residual.buf, root.hidden)
        .map_err(hip_err)?;
    Ok(())
}
