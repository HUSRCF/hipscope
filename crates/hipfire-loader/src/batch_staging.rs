// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Continuous-batch staging, moved out of the daemon load handler.
//!
//! This block constructs `Qwen35DecodeBatchState`, `Lfm2DecodeBatchState` and
//! the EP `Qwen35DecodeBatchEpState`, and it was the single largest remaining
//! reason the daemon named architecture types at all (17 of 87 uses).
//!
//! It belongs here rather than in the daemon because `LoadedModel` already
//! owns the typed fields these write (`qwen35_decode_batch`,
//! `lfm2_decode_batch`, `EpArch::Qwen35 { batch, .. }`) — only the
//! *construction* had leaked upward.
//!
//! `ContinuousBatchScheduler` lives in `hipfire-engine`, which is *above* this
//! crate, so staging returns the two numbers the scheduler needs and the
//! caller constructs it. That keeps the dependency edge pointing one way.
//!
//! The body is a verbatim move: identical branch order, identical fail-closed
//! EP invariants, identical operator-visible `[daemon]` / `[daemon][EP]`
//! strings. Only the local `staged_*` variables became [`BatchStaging`]
//! fields.

use crate::LoadedModel;
use rdna_compute::Gpu;

/// What the load handler needs to know after staging.
#[derive(Debug, Default, Clone, Copy)]
pub struct BatchStaging {
    /// Continuous batching is live for this model.
    pub capable: bool,
    /// Scheduler slot count (0 when not staged).
    pub slots: usize,
    /// Per-lane KV capacity, clamped to what independent attention admits.
    pub lane_capacity: usize,
    /// The expert-parallel (TP=4) route was taken rather than single-GPU.
    pub ep: bool,
    /// EP slot count, mirrored for the receipt.
    pub ep_slots: usize,
    /// EP lane capacity, mirrored for the receipt.
    pub ep_lane_cap: usize,
}

/// True when embedding and lm_head formats admit the batched decode kernels.
pub fn qwen_batch_weight_formats_supported(
    weights: &hipfire_arch_qwen35::qwen35::Qwen35Weights,
) -> bool {
    use hipfire_runtime::llama::EmbeddingFormat;
    use rdna_compute::DType;
    let embd_ok = matches!(
        weights.embd_format,
        EmbeddingFormat::HFQ4G256 | EmbeddingFormat::Q8_0
    );
    let lm_ok = matches!(
        weights.output.gpu_dtype,
        DType::Q8_0
            | DType::HFQ4G256
            | DType::MQ4G256
            | DType::HFQ6G256
            | DType::MQ6G256
            | DType::MQ3G256
    );
    embd_ok && lm_ok
}

/// Stage continuous batching for `m`, returning what the caller must publish.
///
/// Returns `Err` when staging leaves no usable pool behind: any device-sync
/// or GPU-free failure during batch publication/cleanup is fatal (the batch
/// is unpublished and the sequential PBS/partial pool is drained or
/// untrusted). The caller must roll back the whole staged model and never
/// emit a `loaded` ack. `Ok` preserves the safe sequential fallback: failed
/// optional allocation, or a failed peer-enable whose cleanup succeeds with
/// the sequential pool intact, simply reports `capable == false`.
pub fn stage_continuous_batch(
    m: &mut LoadedModel,
    gpu: &mut Gpu,
    requested: usize,
) -> Result<BatchStaging, String> {
    let mut out = BatchStaging::default();
    // ── Continuous batch staging (must be before `loaded` ack) ──
    // Stage Qwen35DecodeBatchState / hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState (single-GPU) or
    // Qwen35DecodeBatchEpState (EP TP=4 pure gfx1201) + host scheduler.
    // `continuous_batch_capable` reflects the newly staged state, not the previous.
    // EP is batch-only: TP must be 4 and exactly 4×gfx1201, else fail closed.
    // Allocation failure advertises false and preserves sequential/poison handling.
    if requested > 1 && m.pp == 1 && m.ep.is_none() {
        match crate::continuous_batch_route(m.arch_id) {
            Some(crate::ContinuousBatchRoute::Qwen35) => {
                // Immutable borrow of `m` ends after this extraction; mutable borrow for batch field later is disjoint.
                let qwen_info = m.qwen35().map(|b| {
                    (
                        qwen_batch_weight_formats_supported(&b.weights),
                        b.scratch.repeat_buf.buf.size(),
                        b.config.head_dim,
                        b.config.clone(),
                        b.weights.embd_format,
                        b.weights.output.gpu_dtype,
                    )
                });
                if let Some((
                    weight_ok,
                    scratch_size,
                    head_dim,
                    config_clone,
                    embd_fmt,
                    out_dtype,
                )) = qwen_info
                {
                    if !weight_ok {
                        eprintln!(
                                            "[daemon] continuous batch requested but weight formats unsupported (embd={:?} lm_head={:?}) — fallback to sequential",
                                            embd_fmt, out_dtype
                                        );
                    } else {
                        let repeat_cap = (scratch_size / 4).max(1);
                        let max_attention_lane =
                            gpu.attention_q8_0_kv_independent_max_lane_capacity(head_dim);
                        let batch_lane_capacity = m.max_seq.min(max_attention_lane);
                        if batch_lane_capacity == 0 {
                            eprintln!(
                                                "[daemon] continuous batch unavailable: independent attention admits no lanes — fallback to sequential"
                                            );
                        } else {
                            if batch_lane_capacity < m.max_seq {
                                eprintln!(
                                                    "[daemon] continuous batch lane capacity clamped: requested={} supported={}",
                                                    m.max_seq,
                                                    batch_lane_capacity
                                                );
                            }
                            match hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchState::new(
                                gpu,
                                &config_clone,
                                requested,
                                batch_lane_capacity,
                                repeat_cap,
                            ) {
                                Ok(batch_state) => {
                                    m.qwen35_mut().unwrap().qwen35_decode_batch = Some(batch_state);
                                    out.slots = requested;
                                    out.lane_capacity = batch_lane_capacity;
                                    out.capable = true;
                                    eprintln!(
                                                        "[daemon] continuous batch staged: slots={} lane_cap={} repeat_cap={}",
                                                        requested,
                                                        batch_lane_capacity,
                                                        repeat_cap
                                                    );
                                }
                                Err(e) => {
                                    eprintln!(
                                                        "[daemon] continuous batch allocation failed: {e} — fallback to sequential"
                                                    );
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("[daemon] continuous batch requested but model state not Qwen35 — fallback to sequential");
                }
            }
            Some(crate::ContinuousBatchRoute::Lfm2Moe) => {
                if m.lfm2moe().is_none() {
                    eprintln!("[daemon] continuous batch requested but model state not Lfm2Moe — fallback to sequential");
                } else if !m.lfm2moe().unwrap().config.is_dense() {
                    eprintln!(
                                        "[daemon] continuous batch requested but LFM MoE not supported (dense only) — fallback to sequential"
                                    );
                } else if let Err(reason) = hipfire_arch_lfm2moe::batch_weight_formats_supported(
                    &m.lfm2moe().unwrap().weights,
                ) {
                    eprintln!(
                                        "[daemon] continuous batch requested but weight formats unsupported: {} — fallback to sequential",
                                        reason
                                    );
                } else {
                    let repeat_cap = 2048usize.max(1);
                    let max_attention_lane = {
                        let b = m.lfm2moe().unwrap();
                        gpu.attention_q8_0_kv_independent_max_lane_capacity(b.config.head_dim)
                    };
                    let batch_lane_capacity = m.max_seq.min(max_attention_lane);
                    if batch_lane_capacity == 0 {
                        eprintln!(
                                            "[daemon] continuous batch unavailable: independent attention admits no lanes — fallback to sequential"
                                        );
                    } else {
                        if batch_lane_capacity < m.max_seq {
                            eprintln!(
                                                "[daemon] continuous batch lane capacity clamped: requested={} supported={}",
                                                m.max_seq,
                                                batch_lane_capacity
                                            );
                        }
                        // Clone config for the call so the immutable borrow ends before the mutable one.
                        let cfg = m.lfm2moe().unwrap().config.clone();
                        match hipfire_arch_lfm2moe::batch::Lfm2DecodeBatchState::new(
                            gpu,
                            &cfg,
                            requested,
                            batch_lane_capacity,
                            repeat_cap,
                        ) {
                            Ok(batch_state) => {
                                if let Some(b) = m.lfm2moe_mut() {
                                    b.lfm2_decode_batch = Some(batch_state);
                                    out.slots = requested;
                                    out.lane_capacity = batch_lane_capacity;
                                    out.capable = true;
                                    eprintln!(
                                                        "[daemon] continuous batch staged: slots={} lane_cap={} repeat_cap={}",
                                                        requested,
                                                        batch_lane_capacity,
                                                        repeat_cap
                                                    );
                                } else {
                                    // Should be unreachable (we checked is_some above), but free to avoid leak.
                                    batch_state.free_gpu(gpu);
                                    eprintln!("[daemon] continuous batch requested but model state not Lfm2Moe — fallback to sequential");
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                                    "[daemon] continuous batch allocation failed: {e} — fallback to sequential"
                                                );
                            }
                        }
                    }
                }
            }
            None => {
                eprintln!("[daemon] continuous batch requested but not capable (arch_id={} pp={} ep={:?}) — fallback to sequential", m.arch_id, m.pp, m.ep.is_some());
            }
        }
    } else if requested > 1 && m.pp == 1 && m.ep.is_some() {
        // EP Qwen35 pure expert-parallel batch route: TP=4, 4×gfx1201, batch-only.
        let tp_ok =
            m.ep.as_ref()
                .map(|ep| ep.gpus.devices.len() == 4)
                .unwrap_or(false);
        let gfx_ok =
            m.ep.as_ref()
                .map(|ep| ep.gpus.devices.iter().all(|d| d.arch_caps.is_gfx1201()))
                .unwrap_or(false);
        let arch_ok = matches!(m.arch_id, 5 | 6);
        if !arch_ok || !tp_ok || !gfx_ok {
            eprintln!("[daemon][EP] continuous batch requires arch 5/6, TP=4, 4×gfx1201 (arch_ok={arch_ok} tp_ok={tp_ok} gfx_ok={gfx_ok}) — fail closed");
            out.capable = false;
        } else if let Some(ep) = m.ep.as_mut() {
            if let crate::EpArch::Qwen35 {
                config,
                weights,
                batch,
                prefill_pbs,
                prefill_partials,
                ..
            } = &mut ep.inner
            {
                // Format admission lives solely in
                // `validate_ep_batch_compatibility` below (single authority).
                // Derive capacities similar to single-GPU but via EP Gpus handle when possible.
                let max_attention_lane = ep.gpus.devices[0]
                    .attention_q8_0_kv_independent_max_lane_capacity(config.head_dim);
                let batch_lane_capacity = m.max_seq.min(max_attention_lane).max(1);
                let repeat_cap = 128usize.max(1);
                let prefill_chunk = hipfire_arch_qwen35::qwen35::prefill_max_batch_ep();
                if batch_lane_capacity == 0 || batch_lane_capacity >= m.max_seq + 1 {
                    eprintln!("[daemon][EP] continuous batch lane capacity invalid — fail closed");
                } else {
                    let load_cfg = hipfire_arch_qwen35::qwen35::Qwen35BatchLoadConfig::new(
                        requested,
                        batch_lane_capacity,
                        repeat_cap,
                        prefill_chunk,
                    );
                    // Fail-closed validation before allocation.
                    match hipfire_arch_qwen35::qwen35::validate_ep_batch_compatibility(
                        &ep.gpus, weights, config, &load_cfg,
                    ) {
                        Ok(compat) => {
                            // Enforce frozen invariants.
                            if compat.rank_count() != 4
                                || compat.rank_mask() != 0x0f
                                || compat.reduce()
                                    != hipfire_arch_qwen35::qwen35::Qwen35EpReduce::PeerRootedF32
                                || compat.topology()
                                    != hipfire_arch_qwen35::qwen35::Qwen35EpTopology::ExpertParallel
                            {
                                eprintln!("[daemon][EP] compat invariants violated — fail closed: rank_count={} mask={:#x} reduce={:?} topo={:?}", compat.rank_count(), compat.rank_mask(), compat.reduce(), compat.topology());
                            } else {
                                // Keep load-owned sequential PBS/partials intact through
                                // batch construction and peer enable so allocation or
                                // enable_peer_all failure can still fall back to them.
                                // Free them only at the publication boundary after both
                                // succeed (post-sync), retaining only the batch seed pool.
                                match hipfire_arch_qwen35::qwen35::Qwen35DecodeBatchEpState::new(
                                    &mut ep.gpus,
                                    weights,
                                    config,
                                    &load_cfg,
                                ) {
                                    Ok(ep_batch) => {
                                        // Attest receipt getters work before publishing.
                                        let _ = ep_batch.max_batch();
                                        let _ = ep_batch.lane_capacity();
                                        // Peer access MUST follow every peer-visible batch
                                        // allocation (partials + leased scratch); ROCm may
                                        // not retroactively map late allocs.
                                        match ep.gpus.enable_peer_all() {
                                            Ok(peer_access) => {
                                                // Publication transaction: sync, free unused
                                                // sequential owners, then publish batch. Any
                                                // cleanup failure must not leave batch=None
                                                // with empty PBS as a silent sequential path.
                                                let mut first_cleanup_err = None;
                                                for dev in ep.gpus.devices.iter_mut() {
                                                    if let Err(e) = dev.bind_thread() {
                                                        first_cleanup_err.get_or_insert(e);
                                                        continue;
                                                    }
                                                    if let Err(e) = dev.hip.device_synchronize() {
                                                        first_cleanup_err.get_or_insert(e);
                                                    }
                                                }
                                                if let Some(sync_err) = first_cleanup_err.take() {
                                                    // A failed device sync leaves neither pool
                                                    // trustworthy: free the batch best-effort
                                                    // and fail the whole load (never `loaded`
                                                    // ack on an unusable resident).
                                                    let cleanup_note = match ep_batch
                                                        .free_gpu(&mut ep.gpus)
                                                    {
                                                        Ok(()) => {
                                                            eprintln!("[daemon][EP] device sync failed after peer enable before sequential free: {sync_err:?} — fail closed (batch freed)");
                                                            String::new()
                                                        }
                                                        Err(cleanup_err) => {
                                                            eprintln!("[daemon][EP] device sync failed after peer enable before sequential free: {sync_err:?}; batch cleanup also failed: {cleanup_err:?} — fail closed");
                                                            format!("; batch cleanup also failed: {cleanup_err:?}")
                                                        }
                                                    };
                                                    return Err(format!("device sync failed after peer enable before sequential free: {sync_err:?}{cleanup_note}"));
                                                } else {
                                                    for (r, pbs) in
                                                        prefill_pbs.drain(..).enumerate()
                                                    {
                                                        if let Some(dev) =
                                                            ep.gpus.devices.get_mut(r)
                                                        {
                                                            if let Err(e) = dev.bind_thread() {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                            if let Err(e) = pbs.free_gpu(dev) {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                        }
                                                    }
                                                    for (r, p) in
                                                        prefill_partials.drain(..).enumerate()
                                                    {
                                                        if let Some(dev) =
                                                            ep.gpus.devices.get_mut(r)
                                                        {
                                                            if let Err(e) = dev.bind_thread() {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                            if let Err(e) = dev.free_tensor(p) {
                                                                first_cleanup_err.get_or_insert(e);
                                                            }
                                                        }
                                                    }
                                                    if let Some(free_err) = first_cleanup_err {
                                                        // Sequential pool already drained — do not
                                                        // publish batch, free it, and fail the
                                                        // load as unusable (no silent empty-PBS
                                                        // sequential fallback).
                                                        let cleanup_note = match ep_batch
                                                            .free_gpu(&mut ep.gpus)
                                                        {
                                                            Ok(()) => {
                                                                eprintln!("[daemon][EP] sequential scratch free failed after peer enable: {free_err:?} — fail closed (batch freed; sequential pool unusable)");
                                                                String::new()
                                                            }
                                                            Err(cleanup_err) => {
                                                                eprintln!("[daemon][EP] sequential scratch free failed after peer enable: {free_err:?}; batch cleanup also failed: {cleanup_err:?} — fail closed (sequential pool unusable)");
                                                                format!("; batch cleanup also failed: {cleanup_err:?}")
                                                            }
                                                        };
                                                        return Err(format!("sequential scratch free failed after peer enable: {free_err:?}{cleanup_note}"));
                                                    } else {
                                                        *batch = Some(ep_batch);
                                                        out.slots = requested;
                                                        out.lane_capacity = batch_lane_capacity;
                                                        out.capable = true;
                                                        out.ep = true;
                                                        out.ep_slots = requested;
                                                        out.ep_lane_cap = batch_lane_capacity;
                                                        eprintln!("[daemon][EP] expert-parallel batch staged: slots={} lane_cap={} repeat_cap={} prefill_chunk={} reduce=peer_rooted_f32 rank_count=4 peer_access={}", requested, batch_lane_capacity, repeat_cap, prefill_chunk, peer_access);
                                                    }
                                                }
                                            }
                                            Err(enable_err) => {
                                                // Sequential owners untouched: a successful
                                                // batch free keeps the sequential fallback
                                                // usable. A failed free leaves neither pool
                                                // trustworthy — fail the whole load.
                                                match ep_batch.free_gpu(&mut ep.gpus) {
                                                    Ok(()) => {
                                                        eprintln!("[daemon][EP] enable_peer_all failed after batch alloc: {enable_err:?} — fail closed (batch freed; sequential pool preserved)");
                                                    }
                                                    Err(cleanup_err) => {
                                                        eprintln!("[daemon][EP] enable_peer_all failed after batch alloc: {enable_err:?}; cleanup also failed: {cleanup_err:?} — fail closed");
                                                        return Err(format!("enable_peer_all failed after batch alloc: {enable_err:?}; batch cleanup also failed: {cleanup_err:?}"));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        // Sequential owners untouched — sequential fallback remains usable.
                                        eprintln!("[daemon][EP] expert-parallel batch allocation failed: {e} — fail closed (sequential pool preserved)");
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[daemon][EP] expert-parallel batch compatibility failed: {e} — fail closed");
                        }
                    }
                }
            } else {
                eprintln!(
                    "[daemon][EP] continuous batch requested but EP arch not Qwen35 — fail closed"
                );
            }
        }
    } else if requested > 1 {
        eprintln!("[daemon] continuous batch requested but not capable (arch_id={} pp={} ep={:?}) — fallback to sequential", m.arch_id, m.pp, m.ep.is_some());
    }
    // Qwen EP peer finalization after every optional batch alloc/cleanup path.
    // Load defers enable_peer_all so late batch scratch is peer-mapped; sequential
    // (requested<=1) never hits the batch success enable, so without this the flag
    // stays false and EP falls through to RCCL / HIP host-staging allreduce.
    // Batch success already enabled → skip. Err/false: report truthfully; do not
    // claim peer-rooted when unavailable (sequential owners stay usable).
    if let Some(ep) = m.ep.as_mut() {
        if matches!(&ep.inner, crate::EpArch::Qwen35 { .. }) && !ep.gpus.peer_access_enabled {
            match ep.gpus.enable_peer_all() {
                Ok(true) => {
                    eprintln!(
                        "[daemon][EP] peer access enabled after staging (sequential/fallback path)"
                    );
                }
                Ok(false) => {
                    eprintln!("[daemon][EP] peer access incomplete after staging — peer-rooted path unavailable");
                }
                Err(e) => {
                    eprintln!("[daemon][EP] enable_peer_all failed after staging: {e:?} — peer-rooted path unavailable");
                }
            }
        }
    }
    Ok(out)
}
