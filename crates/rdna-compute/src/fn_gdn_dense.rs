// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next's gfx1201 prefill GDN on the dense C64 chunk scan.
//!
//! The fused producer preserves the lab route's serial BF16 Q/K norm,
//! F16 packing and chunk-local gate cumsum. The consumer reads the scan's
//! BF16 output directly. The Q8 state aliases the scan's codes/scales; EF
//! enters as zero and is discarded on exit (FN keeps none). Exit Q8 uses
//! round-to-nearest-even, not the pipe128 recurrence's stochastic rounding.
//! Default on for admitted steps; `HIPFIRE_FN_GDN_DENSE_SCAN=0` opts out.

use std::cell::RefCell;

use hip_bridge::{DeviceBuffer, HipResult, KernargBlob};

use crate::tensor_ops::{
    GatedDeltaConvBatched, GatedDeltaGateBatched, GatedDeltaParamsBatched,
    GatedDeltaStepBatched, GdnStateFormat,
};
use crate::{DType, Gpu, GpuTensor};

const EF_BYTES: usize = 48 * 128 * 128 * 2;

struct Scratch {
    rows: usize,
    a: GpuTensor,
    ef: GpuTensor,
}

thread_local! {
    static SCRATCH: RefCell<Option<Scratch>> = const { RefCell::new(None) };
}

fn raw(t: &GpuTensor, off: usize, len: usize) -> GpuTensor {
    GpuTensor {
        buf: unsafe { DeviceBuffer::from_raw((t.buf.as_ptr() as *mut u8).add(off) as *mut _, len) },
        shape: vec![len],
        dtype: DType::Raw,
    }
}

/// Exact gfx1201 FN geometry, Q8 state, no capture/recording, and the mseg
/// segment geometry. Other steps retain the incumbent recurrence.
pub fn applies(gpu: &Gpu, p: &GatedDeltaStepBatched<'_>) -> bool {
    let n = p.rows;
    gpu.flags.fn_gdn_dense_scan
        && gpu.arch == "gfx1201"
        && matches!(GdnStateFormat::of(p.state), Ok(GdnStateFormat::Q8))
        && p.row_states.is_none()
        && p.key_heads == 16
        && p.value_heads == 48
        && p.key_dim == 128
        && p.value_dim == 128
        && p.qkv_width == 10240
        && p.projection.dtype == DType::F32
        && !gpu.replay.is_recording()
        && !gpu.graphs.capture_mode
        && n >= 64
        && (n % 512 == 0 || (64..512).contains(&(n % 512)))
}

fn launch(gpu: &mut Gpu, sym: &str, grid: [u32; 3], block: u32, args: &mut KernargBlob) -> HipResult<()> {
    // Uses the registered module/pack lookup, with the same compiler options
    // as tensor_ops (no chunk-scan-only floating-point flags).
    gpu.ensure_kernel("fn_gdn_dense", crate::kernels::FN_GDN_DENSE_SRC, sym)?;
    args.pad_to(16);
    gpu.launch_blob_recorded(
        sym, grid, [block, 1, 1], 0, args.as_mut_slice(),
        crate::dispatch::ReplayLaunchBindings::NONE,
    )
}

/// Fused convolution/parameters/packing, then KKT and mseg. Caller checked
/// [`applies`]; the input projection and convolution destination are distinct.
pub fn run(
    gpu: &mut Gpu,
    conv: &GatedDeltaConvBatched<'_>,
    params: &GatedDeltaParamsBatched<'_>,
    p: &GatedDeltaStepBatched<'_>,
) -> HipResult<()> {
    let n = p.rows;
    let n64 = n.div_ceil(64) * 64;
    // KKT clamps K loads to the last real row and guards G/beta; mseg clamps
    // Q/K/V loads, guards its controls, and only stores output for tok<rows.
    // Neither reads Q/K/V/G/beta pad rows. KKT DOES write a full 64-row A
    // block, so only A needs padded scratch. Q/K/V alias the dead F32 conv
    // destination (n64*10240*2 <= n*10240*4 for admitted n>=64); the producer
    // also zeros their pad rows. G aliases gate, beta stays in its original
    // plane, and BF16 scan output aliases the F32 recurrent destination.
    let q_bytes = n64 * 2048 * 2;
    let q = raw(p.projection, 0, q_bytes);
    let k = raw(p.projection, q_bytes, q_bytes);
    let v = raw(p.projection, q_bytes * 2, n64 * 6144 * 2);
    let out = raw(p.output, 0, n * 6144 * 2);
    let mut slot = SCRATCH.with(|s| s.borrow_mut().take());
    if slot.as_ref().is_some_and(|s| s.rows < n64) {
        let s = slot.take().unwrap();
        gpu.free_tensor(s.a)?;
        gpu.free_tensor(s.ef)?;
    }
    let s = match slot {
        Some(s) => s,
        None => Scratch {
            rows: n64,
            a: gpu.zeros(&[n64 * 48 * 64 * 2], DType::Raw)?,
            ef: gpu.zeros(&[EF_BYTES], DType::Raw)?,
        },
    };
    let result = (|| -> HipResult<()> {
        let mut args = KernargBlob::new();
        for t in [
            conv.input, conv.kernel, conv.history, conv.next_history,
            &q, &k, &v, p.gate, p.beta,
            params.a, params.b, params.a_log, params.dt_bias,
        ] {
            args.push_ptr(t.buf.as_ptr());
        }
        args.push_ptr(s.ef.buf.as_ptr());
        args.push_i32(n as i32);
        args.push_i32(n64 as i32);
        args.push_i32(conv.start_cursor as i32);
        args.push_i32(i32::from(conv.input.dtype == DType::BF16));
        args.push_f32((p.key_dim as f32).sqrt().recip());
        args.push_i32((EF_BYTES / 4) as i32);
        launch(gpu, "fn_gdn_dense_producer", [82, (n64 / 32) as u32, 1], 256, &mut args)?;
        gpu.gdn_chunk_kkt_solve_batched(&k, p.gate, p.beta, &s.a, n)?;
        let codes = raw(p.state, 0, 48 * 128 * 128);
        let scales = raw(p.state, 48 * 128 * 128, 48 * 128 * 4);
        gpu.gdn_chunk_scan_layer_mseg(
            &q, &k, &v, &s.a, p.gate, p.beta,
            &codes, &scales, &s.ef, &out, n,
        )
    })();
    SCRATCH.with(|c| *c.borrow_mut() = Some(s));
    result
}

/// Gate the packed BF16 scan output, optionally also producing the output
/// projection's Aligned256 rotation. No later reader needs recurrent_output.
pub fn gate(
    gpu: &mut Gpu,
    p: &GatedDeltaGateBatched<'_>,
    rotated: Option<&GpuTensor>,
) -> HipResult<()> {
    let elements = p.rows * p.value_heads * p.value_dim;
    let mut args = KernargBlob::new();
    for t in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(t.buf.as_ptr());
    }
    if let Some(rotated) = rotated {
        gpu.ensure_mq_signs()?;
        args.push_ptr(rotated.buf.as_ptr());
        args.push_ptr(gpu.scratch.mq_signs1.as_ref().unwrap().buf.as_ptr());
        args.push_ptr(gpu.scratch.mq_signs2.as_ref().unwrap().buf.as_ptr());
        launch(gpu, "fn_gdn_dense_gate_rotate_bf16in", [(elements / 256) as u32, 1, 1], 64, &mut args)?;
        gpu.scratch.prerotated = Some((
            p.output.buf.as_ptr() as usize,
            rotated.buf.as_ptr() as usize,
            elements,
        ));
        Ok(())
    } else {
        args.push_i32(p.rows as i32);
        args.push_i32(p.value_heads as i32);
        args.push_i32(p.value_dim as i32);
        launch(gpu, "fn_gdn_dense_gate_bf16in", [(elements / 128) as u32, 1, 1], 32, &mut args)
    }
}
