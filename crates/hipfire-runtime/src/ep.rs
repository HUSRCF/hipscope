// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Expert-parallel (EP) executor for the Ship 6 super-op substrate.
//!
//! Runs a lowered [`LayerProgram`] **replicated across N ranks** (every rank
//! runs every op on full, replicated attention/dense weights), special-casing
//! the `Moe` super-op with **all-reduce EP**:
//!
//! 1. zero each rank's routed partial,
//! 2. each rank computes ONLY its owned experts (+ the shared expert on rank 0)
//!    into its partial via [`ForwardBindings::run_moe_ep`] (non-owned experts
//!    read load-time zero-dummy weights → contribute 0),
//! 3. `all_reduce_sum_f32` the partials across ranks (RCCL),
//! 4. each rank adds the reduced partial into its residual stream via
//!    [`ForwardBindings::ep_add_into_residual`].
//!
//! All other super-ops (Attend / Norm / Proj / ResidualGemv / Recurrent / Conv
//! / Escape) run **replicated** and unchanged — every rank holds the full
//! weights and full KV, so they are deterministic functions of replicated
//! inputs and stay bit-identical across ranks. This is why EP needs no
//! attention-sharding (FaPhase) seam: the only divergence is at `Moe`.
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
use hipfire_dispatch::pipeline::superop::{
    dispatch_super_op, ForwardBindings, LayerProgram, SuperOpKind,
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
        if matches!(op.kind, SuperOpKind::Moe) {
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

            let overlap_requested =
                hipfire_config::developer_var("HIPFIRE_GFX90A_EP_OVERLAP").as_deref() == Ok("1");
            let use_overlap = overlap_requested
                && n == 2
                && gpus.devices.iter().all(|device| device.arch == "gfx90a")
                && bindings.iter().all(ForwardBindings::supports_moe_ep_rows);

            if use_overlap {
                for r in 0..n {
                    gpus.devices[r].bind_thread().map_err(hip_err)?;
                    let ctx = DispatchCtx::new(&gpus.devices[r]);
                    bindings[r].run_moe_ep_prepare(
                        &mut gpus.devices[r],
                        &ctx,
                        &op.binding,
                        &partials[r],
                    )?;
                }

                let chunk_rows = hipfire_config::developer_var("HIPFIRE_GFX90A_EP_CHUNK_ROWS")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|value| *value > 0)
                    .unwrap_or(1024)
                    .min(residual_dim);
                let refs = [&partials[0].buf, &partials[1].buf];
                let mut done = Vec::with_capacity(residual_dim.div_ceil(chunk_rows));
                for row_base in (0..residual_dim).step_by(chunk_rows) {
                    let row_count = chunk_rows.min(residual_dim - row_base);
                    for r in 0..2 {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        bindings[r].run_moe_ep_rows(
                            &mut gpus.devices[r],
                            &partials[r],
                            row_base,
                            row_count,
                        )?;
                    }
                    gpus.devices[0].bind_thread().map_err(hip_err)?;
                    let ready0 = gpus.devices[0].hip.event_create().map_err(hip_err)?;
                    gpus.devices[0]
                        .hip
                        .event_record(&ready0, gpus.devices[0].active_stream.as_ref())
                        .map_err(hip_err)?;
                    gpus.devices[1].bind_thread().map_err(hip_err)?;
                    let ready1 = gpus.devices[1].hip.event_create().map_err(hip_err)?;
                    gpus.devices[1]
                        .hip
                        .event_record(&ready1, gpus.devices[1].active_stream.as_ref())
                        .map_err(hip_err)?;
                    done.push(
                        gpus.all_reduce_sum_f32_peer_chunk_async(
                            &refs,
                            row_base,
                            row_count,
                            [ready0, ready1],
                        )
                        .map_err(hip_err)?,
                    );
                }
                gpus.finish_peer_chunks(done).map_err(hip_err)?;
            } else {
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

                // 3. All-reduce-sum the partials across ranks in place. Two-rank
                //    gfx90a EP defaults to peer-direct after a byte-identical
                //    32-token DeepSeek V4 A/B showed a decode win together with HIP
                //    Graph. Other configurations retain RCCL; the environment
                //    variable explicitly forces either path for regression bisects.
                static COARSE_PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let coarse_probe = *COARSE_PROBE.get_or_init(|| {
                    hipfire_config::developer_var("HIPFIRE_EP_COARSE_PROBE").as_deref() == Ok("1")
                });
                if coarse_probe {
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        gpus.devices[r].hip.device_synchronize().map_err(hip_err)?;
                    }
                }
                let allreduce_started = std::time::Instant::now();
                let refs: Vec<&DeviceBuffer> = partials.iter().map(|p| &p.buf).collect();
                static PEER_DECODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                let use_peer = *PEER_DECODE.get_or_init(|| {
                    match hipfire_config::developer_var("HIPFIRE_EP_PEER_ALLREDUCE_DECODE")
                        .ok()
                        .as_deref()
                    {
                        Some("1") => true,
                        Some("0") => false,
                        _ => n == 2 && gpus.devices.iter().all(|device| device.arch == "gfx90a"),
                    }
                });
                if use_peer {
                    gpus.all_reduce_sum_f32_peer(&refs, residual_dim)
                        .map_err(hip_err)?;
                } else {
                    gpus.all_reduce_sum_f32(&refs, residual_dim)
                        .map_err(hip_err)?;
                }
                if coarse_probe {
                    for r in 0..n {
                        gpus.devices[r].bind_thread().map_err(hip_err)?;
                        gpus.devices[r].hip.device_synchronize().map_err(hip_err)?;
                    }
                    eprintln!(
                        "EP-COARSE stage=ep_allreduce ranks={n} ms={:.3}",
                        allreduce_started.elapsed().as_secs_f64() * 1000.0,
                    );
                }
            }

            // 4. Each rank adds the reduced partial into its residual stream.
            for r in 0..n {
                gpus.devices[r].bind_thread().map_err(hip_err)?;
                bindings[r].ep_add_into_residual(&mut gpus.devices[r], &partials[r])?;
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
