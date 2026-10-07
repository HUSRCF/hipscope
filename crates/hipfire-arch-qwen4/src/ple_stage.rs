// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Host staging for one forward's PLE rows, and the lifetime rule that makes
//! an asynchronous upload out of it safe.
//!
//! A deferred (mid-program) PLE upload used to be a blocking `hipMemcpy`,
//! which also waits for every layer enqueued ahead of it. The asynchronous
//! upload returns at once, but the device may read the host bytes any time
//! until the stream reaches the copy, so the single reusable staging buffer is
//! shared state with a hazard: it must not be overwritten, reallocated or
//! freed while a copy still reads it. [`PleHostStage`] is the only owner of
//! that buffer. It records a completion fence behind every asynchronous copy
//! and every write goes through [`PleHostStage::stage_buffer`], which first
//! drains the fence. A forward that fails after the enqueue, or drops the
//! stage without [`PleHostStage::free_gpu`], cannot skip the drain: the next
//! write still finds the state, and [`Drop`] leaks the bytes rather than free
//! memory a copy may still read.

use hip_bridge::{DeviceBuffer, Event, HipError, HipResult};
use rdna_compute::Gpu;

/// `HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD=1` opts in to the fenced asynchronous upload
/// for a deferred PLE stage; unset or `0` keeps the blocking upload. Default
/// off (warm-cache and cache-quality identity unconfirmed on the RC4c gate).
pub const PLE_ASYNC_UPLOAD_ENV: &str = "HIPFIRE_QWEN4_PLE_ASYNC_UPLOAD";

/// Whether an asynchronous copy may still read the staging bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PleUpload {
    /// No enqueued copy reads the bytes.
    Idle,
    /// An asynchronous copy reads the bytes and the fence was recorded right
    /// behind it on the same stream.
    Fenced,
    /// An asynchronous copy may read the bytes and no fence covers it (the
    /// enqueue or the fence record failed). Only draining the device proves
    /// it complete.
    Unfenced,
}

/// What must complete before the staging bytes may be written again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PleDrain {
    Nothing,
    Fence,
    Device,
}

impl PleUpload {
    /// The wait that retires the state.
    pub(crate) fn drain(self) -> PleDrain {
        match self {
            Self::Idle => PleDrain::Nothing,
            Self::Fenced => PleDrain::Fence,
            Self::Unfenced => PleDrain::Device,
        }
    }

    /// State just before an asynchronous copy is enqueued. The copy may
    /// already be reading when the call fails, so this is pessimistic.
    pub(crate) fn enqueue(self) -> Self {
        debug_assert_eq!(self, Self::Idle, "an upload was enqueued over an undrained one");
        Self::Unfenced
    }

    /// State once the fence recorded right behind the copy succeeded.
    pub(crate) fn fence_recorded(self) -> Self {
        match self {
            Self::Unfenced => Self::Fenced,
            other => other,
        }
    }
}

/// Whether a deferred PLE stage uploads asynchronously: the knob is on, the
/// forward holds no device token (that branch carries its own readback
/// ordering), staging is deferred past layer 0 (otherwise nothing is queued
/// ahead of the copy to drain and a retained body may follow), and no graph
/// capture is open (the null stream is forbidden inside one).
pub(crate) fn ple_async_upload(
    enabled: bool,
    device_token: bool,
    deferred: bool,
    capturing: bool,
) -> bool {
    enabled && !device_token && deferred && !capturing
}

/// The reusable host staging buffer of one forward and its upload fence.
pub(crate) struct PleHostStage {
    bytes: Vec<u8>,
    fence: Option<Event>,
    upload: PleUpload,
    enabled: bool,
}

impl PleHostStage {
    pub(crate) fn new(bytes: Vec<u8>, enabled: bool) -> Self {
        Self {
            bytes,
            fence: None,
            upload: PleUpload::Idle,
            enabled,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Wait until no enqueued copy reads the bytes. On failure the state is
    /// unchanged, so the bytes stay unwritable and the next call retries.
    fn drain(&mut self, gpu: &Gpu) -> HipResult<()> {
        match (self.upload.drain(), self.fence.as_ref()) {
            (PleDrain::Nothing, _) => {}
            (PleDrain::Fence, Some(fence)) => gpu.hip.event_synchronize(fence)?,
            // A Fenced state without an event cannot arise; drain the whole
            // device rather than trust it.
            (PleDrain::Fence, None) | (PleDrain::Device, _) => gpu.hip.device_synchronize()?,
        }
        self.upload = PleUpload::Idle;
        Ok(())
    }

    /// The first `len` bytes for the host copy to fill. This is the only
    /// mutable access to the buffer, and it drains any pending upload first.
    pub(crate) fn stage_buffer(&mut self, gpu: &Gpu, len: usize) -> HipResult<&mut [u8]> {
        self.drain(gpu)?;
        Ok(&mut self.bytes[..len])
    }

    /// Blocking upload of the staged bytes (`memcpy_htod_auto`, capture aware).
    pub(crate) fn upload_blocking(
        &self,
        gpu: &Gpu,
        dst: &DeviceBuffer,
        len: usize,
    ) -> HipResult<()> {
        gpu.memcpy_htod_auto(dst, &self.bytes[..len])
    }

    /// The device-token forward's asynchronous upload, unchanged and not
    /// fenced (decode hot path). Its ordering is the readback: the caller has
    /// just synchronized on the event of an argmax enqueued behind the
    /// previous forward's upload, so the buffer was free to write; the next
    /// writer is the next device-token forward behind the same readback, or a
    /// forward that began with a blocking token-id copy on that stream.
    pub(crate) fn upload_unfenced(
        &self,
        gpu: &Gpu,
        dst: &DeviceBuffer,
        len: usize,
    ) -> HipResult<()> {
        gpu.hip.memcpy_htod_async_default(dst, &self.bytes[..len])
    }

    /// Enqueue the staged bytes on the legacy stream without blocking the
    /// host and fence the copy. Every `?` after the enqueue leaves the state
    /// pending, so a failing forward cannot release the buffer early.
    pub(crate) fn upload_async(
        &mut self,
        gpu: &Gpu,
        dst: &DeviceBuffer,
        len: usize,
    ) -> HipResult<()> {
        gpu.bind_thread()?;
        if self.fence.is_none() {
            self.fence = Some(
                gpu.hip
                    .event_create_with_flags(hip_bridge::HIP_EVENT_DISABLE_TIMING)?,
            );
        }
        self.upload = self.upload.enqueue();
        gpu.hip
            .memcpy_htod_async_default(dst, &self.bytes[..len])?;
        // Same (legacy) stream as the copy, so the event completes after it.
        let fence = self.fence.as_ref().expect("fence created above");
        gpu.hip.event_record(fence, None)?;
        self.upload = self.upload.fence_recorded();
        Ok(())
    }

    /// Teardown: retire any pending upload, then destroy the fence.
    pub(crate) fn free_gpu(mut self, gpu: &Gpu) -> Option<HipError> {
        let mut first = self.drain(gpu).err();
        if let Some(fence) = self.fence.take() {
            if let Err(error) = gpu.hip.event_destroy(fence) {
                first.get_or_insert(error);
            }
        }
        first
    }
}

impl Drop for PleHostStage {
    fn drop(&mut self) {
        // A stage dropped with a copy pending (no `free_gpu`, or a failed
        // drain) keeps its bytes alive forever instead of freeing memory the
        // device may still read.
        if self.upload != PleUpload::Idle {
            std::mem::forget(std::mem::take(&mut self.bytes));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_needs_no_wait() {
        assert_eq!(PleUpload::Idle.drain(), PleDrain::Nothing);
    }

    #[test]
    fn fenced_copy_waits_on_the_fence_only() {
        assert_eq!(PleUpload::Fenced.drain(), PleDrain::Fence);
    }

    #[test]
    fn unfenced_copy_waits_on_the_device() {
        assert_eq!(PleUpload::Unfenced.drain(), PleDrain::Device);
    }

    #[test]
    fn enqueue_is_pending_until_the_fence_is_recorded() {
        // Success path: enqueue then fence record.
        let enqueued = PleUpload::Idle.enqueue();
        assert_eq!(enqueued, PleUpload::Unfenced);
        assert_eq!(enqueued.fence_recorded(), PleUpload::Fenced);
        // Failure between the two (enqueue or record error): the state never
        // reaches Idle, so the next write still drains.
        assert_ne!(enqueued.drain(), PleDrain::Nothing);
    }

    #[test]
    fn fence_record_never_downgrades_or_revives() {
        assert_eq!(PleUpload::Idle.fence_recorded(), PleUpload::Idle);
        assert_eq!(PleUpload::Fenced.fence_recorded(), PleUpload::Fenced);
    }

    #[test]
    fn every_pending_state_has_a_drain() {
        for state in [PleUpload::Fenced, PleUpload::Unfenced] {
            assert_ne!(state.drain(), PleDrain::Nothing);
        }
    }

    #[test]
    fn async_upload_only_for_a_deferred_stage_without_a_device_token() {
        // (enabled, device_token, deferred, capturing)
        assert!(ple_async_upload(true, false, true, false));
        assert!(!ple_async_upload(false, false, true, false), "opt-out");
        assert!(!ple_async_upload(true, true, true, false), "device-token branch");
        assert!(!ple_async_upload(true, false, false, false), "not deferred");
        assert!(!ple_async_upload(true, false, true, true), "graph capture");
    }

    #[test]
    fn async_upload_truth_table_requires_every_condition() {
        for bits in 0u8..16 {
            let [enabled, device_token, deferred, capturing] =
                [0, 1, 2, 3].map(|bit| bits >> bit & 1 == 1);
            assert_eq!(
                ple_async_upload(enabled, device_token, deferred, capturing),
                enabled && !device_token && deferred && !capturing,
            );
        }
    }
}
