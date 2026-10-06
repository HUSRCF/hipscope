// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Engine-owned cache of prefill snapshots shared by every session on one
//! loaded model.
//!
//! The cache owns keys, planning, eviction, the memory guard, storage
//! placement and every copy. An architecture only describes its live state
//! through [`SessionState`]: where a cold prefill materializes canonical state
//! ([`SessionState::snapshot_boundaries`]) and which device byte ranges plus
//! host metadata make up that state ([`SessionState::snapshot_parts`]). Drivers
//! call [`SessionCache::begin`] before prefill, split prefill at
//! [`SessionCache::next_boundary`], report each boundary through
//! [`SessionCache::at_boundary`], and [`SessionCache::commit`] once the turn's
//! output reached the client. Snapshots are self-contained: restoring one
//! never depends on what the live state held before.

use std::collections::VecDeque;

use hip_bridge::DeviceBuffer;
use rdna_compute::tensor_ops::{copy_regions, CopyRegion};
use rdna_compute::Gpu;

use crate::checkpoint_pool::{prefix_fingerprint, CheckpointBlob, CheckpointPool};
use crate::serve_contract::CacheDomain;

/// Byte alignment of each part inside a stored snapshot.
const PART_ALIGN: usize = 256;
/// Unified memory: a snapshot is system RAM, and overshoot reaches the global
/// OOM killer, so leave the desktop this much `MemAvailable`.
const UMA_HEADROOM: u64 = 8 << 30;
/// Discrete GPUs: free VRAM left after a snapshot and the state's growth.
const VRAM_HEADROOM: u64 = 256 << 20;

/// Which decode route the snapshot serves. Routes own different state (MTP
/// adds the draft head), so their snapshots never alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SessionRoute {
    Ar,
    Mtp,
}

/// One contiguous device byte range of live model state.
pub struct StatePart<'a> {
    pub buf: &'a DeviceBuffer,
    pub offset: usize,
    pub bytes: usize,
}

/// What an architecture captures at its current position.
pub struct SnapshotParts<'a> {
    pub meta: Vec<u8>,
    pub parts: Vec<StatePart<'a>>,
}

/// Implemented by an architecture's model state. It only describes its state;
/// the cache owns keys, planning, placement, eviction and every copy.
pub trait SessionState {
    /// Everything besides the token prefix that decides whether a snapshot is
    /// reusable for `route` (route, prefill chunk, state formats). `None` = not
    /// cacheable now.
    fn snapshot_scope(&self, route: SessionRoute) -> Option<String>;
    /// Ascending positions p with after < p <= up_to where a cold prefill of
    /// this prompt materializes canonical state.
    fn snapshot_boundaries(&self, route: SessionRoute, after: usize, up_to: usize) -> Vec<usize>;
    /// Device ranges + host metadata that together are the live state, which
    /// must be exactly at `position`.
    fn snapshot_parts(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        position: usize,
    ) -> Result<SnapshotParts<'_>, String>;
    /// Make the live state ready to receive the snapshot described by `meta`
    /// (map capacity, bump epochs) and return destination ranges in capture
    /// order.
    fn restore_parts(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<Vec<StatePart<'_>>, String>;
    /// Apply host metadata after the device bytes were copied.
    fn finish_restore(
        &mut self,
        gpu: &mut Gpu,
        route: SessionRoute,
        meta: &[u8],
    ) -> Result<(), String>;
    /// Bytes the live state may still map or allocate before reaching its
    /// admitted context.
    fn growth_reserve_bytes(&self) -> u64;
    /// Cold-start the live state.
    fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String>;
}

/// Where a snapshot's bytes live. Storage tiers (host RAM, SSD, HDD) and
/// streamed restores are added as variants here, so consumers and
/// [`SessionState`] never change.
enum SnapshotLocation {
    Device(DeviceBuffer),
}

struct StoredSnapshot {
    location: SnapshotLocation,
    part_bytes: Vec<usize>,
    meta: Vec<u8>,
    bytes: u64,
}

impl CheckpointBlob for StoredSnapshot {
    fn bytes_len(&self) -> u64 {
        self.bytes
    }
}

struct Turn {
    domain: CacheDomain,
    route: SessionRoute,
    boundaries: VecDeque<usize>,
}

pub struct SessionCache {
    pool: CheckpointPool<StoredSnapshot>,
    domain: CacheDomain,
    budget: u64,
    /// Captured this turn, published by [`Self::commit`].
    pending: Vec<(CacheDomain, u64, u64, StoredSnapshot)>,
    /// Displaced by `commit` (which has no GPU); freed at the next `begin`/`clear`.
    release: Vec<StoredSnapshot>,
    turn: Option<Turn>,
}

/// Offsets of `part_bytes` packed at [`PART_ALIGN`], and the total size.
fn layout(part_bytes: impl IntoIterator<Item = usize>) -> (Vec<usize>, usize) {
    let mut offsets = Vec::new();
    let mut end = 0usize;
    for bytes in part_bytes {
        let offset = end.next_multiple_of(PART_ALIGN);
        offsets.push(offset);
        end = offset + bytes;
    }
    (offsets, end)
}

fn free_snapshot(gpu: &mut Gpu, snapshot: StoredSnapshot) {
    let SnapshotLocation::Device(buf) = snapshot.location;
    if let Err(error) = gpu.hip.free(buf) {
        eprintln!("  session cache: freeing a snapshot failed: {error}");
    }
}

fn memory_fits(gpu: &mut Gpu, need: u64) -> Result<bool, String> {
    if gpu.is_uma() {
        Ok(rdna_compute::kv_slots::mem_available_bytes()
            .is_some_and(|available| available >= need + UMA_HEADROOM))
    } else {
        let (free, _) = gpu.hip.get_vram_info().map_err(|e| e.to_string())?;
        Ok(free as u64 >= need + VRAM_HEADROOM)
    }
}

impl SessionCache {
    pub fn new(domain: CacheDomain, budget_bytes: u64) -> Self {
        Self {
            pool: CheckpointPool::new(budget_bytes),
            domain,
            budget: budget_bytes,
            pending: Vec::new(),
            release: Vec::new(),
            turn: None,
        }
    }

    /// Longest cached prefix of `prompt` (always shorter than the prompt, so
    /// prefill computes at least the last token); 0 on a miss.
    pub fn plan(&self, state: &dyn SessionState, prompt: &[u32], route: SessionRoute) -> usize {
        let Some(scope) = state.snapshot_scope(route) else {
            return 0;
        };
        if prompt.len() < 2 {
            return 0;
        }
        let domain = self.domain.scoped(&scope);
        state
            .snapshot_boundaries(route, 0, prompt.len() - 1)
            .into_iter()
            .rev()
            .find(|&p| {
                self.pool
                    .contains(&domain, p as u64, prefix_fingerprint(&prompt[..p]))
            })
            .unwrap_or(0)
    }

    /// Start a prefill of `prompt`: restore the `reused`-token snapshot
    /// [`Self::plan`] returned (or cold-start the state for 0) and arm the
    /// boundaries this prefill crosses.
    pub fn begin(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prompt: &[u32],
        route: SessionRoute,
        reused: usize,
    ) -> Result<(), String> {
        for snapshot in self.release.drain(..) {
            free_snapshot(gpu, snapshot);
        }
        for (_, _, _, snapshot) in self.pending.drain(..) {
            free_snapshot(gpu, snapshot);
        }
        self.turn = None;
        let Some(scope) = state.snapshot_scope(route) else {
            if reused > 0 {
                return Err(format!(
                    "session cache: route {route:?} has no snapshot scope"
                ));
            }
            return state.reset(gpu);
        };
        let domain = self.domain.scoped(&scope);
        if reused == 0 {
            state.reset(gpu)?;
        } else {
            let fp = prefix_fingerprint(&prompt[..reused]);
            let Some(snapshot) = self.pool.get(&domain, reused as u64, fp) else {
                return Err(format!(
                    "session cache: planned {reused}-token snapshot is no longer present"
                ));
            };
            let restored = (|| {
                let parts = state.restore_parts(gpu, route, &snapshot.meta)?;
                if parts.len() != snapshot.part_bytes.len()
                    || parts
                        .iter()
                        .zip(&snapshot.part_bytes)
                        .any(|(part, &bytes)| part.bytes != bytes)
                {
                    return Err("session cache: snapshot layout mismatch".to_string());
                }
                let SnapshotLocation::Device(src) = &snapshot.location;
                let (offsets, _) = layout(snapshot.part_bytes.iter().copied());
                let regions: Vec<CopyRegion<'_>> = parts
                    .iter()
                    .zip(offsets)
                    .map(|(part, src_offset)| CopyRegion {
                        dst: part.buf,
                        dst_offset: part.offset,
                        src,
                        src_offset,
                        bytes: part.bytes,
                    })
                    .collect();
                copy_regions(gpu, &regions).map_err(|e| e.to_string())?;
                drop(parts);
                state.finish_restore(gpu, route, &snapshot.meta)
            })();
            if let Err(error) = restored {
                let _ = state.reset(gpu);
                return Err(error);
            }
        }
        self.turn = Some(Turn {
            boundaries: state
                .snapshot_boundaries(route, reused, prompt.len())
                .into(),
            domain,
            route,
        });
        Ok(())
    }

    /// The next position at which the prefill must stop and call
    /// [`Self::at_boundary`].
    pub fn next_boundary(&self) -> Option<usize> {
        self.turn.as_ref()?.boundaries.front().copied()
    }

    /// The live state reached `prefix.len()`, the boundary
    /// [`Self::next_boundary`] returned; capture it (pending until commit).
    ///
    /// ponytail: full snapshot per boundary (rows from 0); delta snapshots
    /// sharing earlier rows would cut memory when budgets bind.
    pub fn at_boundary(
        &mut self,
        gpu: &mut Gpu,
        state: &mut dyn SessionState,
        prefix: &[u32],
    ) -> Result<(), String> {
        let turn = self
            .turn
            .as_mut()
            .ok_or("session cache: no prefill in progress")?;
        if turn.boundaries.front() != Some(&prefix.len()) {
            return Err(format!(
                "session cache: driver reported position {} but next boundary is {:?}",
                prefix.len(),
                turn.boundaries.front()
            ));
        }
        turn.boundaries.pop_front();
        let (domain, route, p) = (turn.domain.clone(), turn.route, prefix.len());
        let fp = prefix_fingerprint(prefix);
        if self.pool.contains(&domain, p as u64, fp) {
            return Ok(());
        }
        let growth = state.growth_reserve_bytes();
        let SnapshotParts { meta, parts } = state.snapshot_parts(gpu, route, p)?;
        let part_bytes: Vec<usize> = parts.iter().map(|part| part.bytes).collect();
        let (offsets, total) = layout(part_bytes.iter().copied());
        let bytes = total as u64;
        let skip =
            |reason: &str| eprintln!("  session cache: skipped {p}-token snapshot ({reason})");
        if bytes > self.budget {
            skip("exceeds budget");
            return Ok(());
        }
        let pending_bytes: u64 = self.pending.iter().map(|entry| entry.3.bytes).sum();
        loop {
            let over_budget = self.pool.total_bytes() + pending_bytes + bytes > self.budget;
            if !over_budget && memory_fits(gpu, bytes + growth)? {
                break;
            }
            match self.pool.pop_lru() {
                Some(evicted) => free_snapshot(gpu, evicted),
                None => {
                    skip(if over_budget {
                        "budget"
                    } else {
                        "memory guard"
                    });
                    return Ok(());
                }
            }
        }
        let Ok(dst) = gpu.bind_thread().and_then(|()| gpu.hip.malloc(total)) else {
            skip("allocation");
            return Ok(());
        };
        let regions: Vec<CopyRegion<'_>> = parts
            .iter()
            .zip(offsets)
            .map(|(part, dst_offset)| CopyRegion {
                dst: &dst,
                dst_offset,
                src: part.buf,
                src_offset: part.offset,
                bytes: part.bytes,
            })
            .collect();
        let copied = copy_regions(gpu, &regions);
        drop(regions);
        drop(parts);
        let snapshot = StoredSnapshot {
            location: SnapshotLocation::Device(dst),
            part_bytes,
            meta,
            bytes,
        };
        if let Err(error) = copied {
            free_snapshot(gpu, snapshot);
            return Err(format!("session cache: snapshot copy failed: {error}"));
        }
        self.pending.push((domain, p as u64, fp, snapshot));
        Ok(())
    }

    /// Publish this turn's snapshots. Uncommitted snapshots are dropped by
    /// the next [`Self::begin`], so a failed turn never publishes state.
    pub fn commit(&mut self) {
        for (domain, p, fp, snapshot) in self.pending.drain(..) {
            let (_, displaced) = self.pool.insert(domain, p, fp, snapshot);
            self.release.extend(displaced);
        }
        self.turn = None;
    }

    /// Free every snapshot.
    pub fn clear(&mut self, gpu: &mut Gpu) {
        let pooled = self.pool.drain_blobs();
        let pending = self.pending.drain(..).map(|entry| entry.3);
        for snapshot in pooled
            .into_iter()
            .chain(pending)
            .chain(self.release.drain(..))
        {
            free_snapshot(gpu, snapshot);
        }
        self.turn = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve_contract::{
        ArchPolicy, DeviceTopology, KvLayout, SharingNamespace, TemplateIdentity, TokenizerIdentity,
    };
    use rdna_compute::{DType, GpuTensor};

    const STATE_BYTES: usize = 4096;
    const STRIDE: usize = 128;

    struct Toy {
        state: GpuTensor,
        position: usize,
    }

    impl SessionState for Toy {
        fn snapshot_scope(&self, _route: SessionRoute) -> Option<String> {
            Some("toy".to_string())
        }
        fn snapshot_boundaries(
            &self,
            _route: SessionRoute,
            after: usize,
            up_to: usize,
        ) -> Vec<usize> {
            (after / STRIDE + 1..=up_to / STRIDE)
                .map(|k| k * STRIDE)
                .collect()
        }
        fn snapshot_parts(
            &mut self,
            _gpu: &mut Gpu,
            _route: SessionRoute,
            position: usize,
        ) -> Result<SnapshotParts<'_>, String> {
            assert_eq!(self.position, position);
            Ok(SnapshotParts {
                meta: (position as u64).to_le_bytes().to_vec(),
                parts: vec![StatePart {
                    buf: &self.state.buf,
                    offset: 0,
                    bytes: STATE_BYTES,
                }],
            })
        }
        fn restore_parts(
            &mut self,
            _gpu: &mut Gpu,
            _route: SessionRoute,
            _meta: &[u8],
        ) -> Result<Vec<StatePart<'_>>, String> {
            Ok(vec![StatePart {
                buf: &self.state.buf,
                offset: 0,
                bytes: STATE_BYTES,
            }])
        }
        fn finish_restore(
            &mut self,
            _gpu: &mut Gpu,
            _route: SessionRoute,
            meta: &[u8],
        ) -> Result<(), String> {
            self.position = u64::from_le_bytes(meta.try_into().unwrap()) as usize;
            Ok(())
        }
        fn growth_reserve_bytes(&self) -> u64 {
            0
        }
        fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
            self.position = 0;
            gpu.hip
                .memcpy_htod(&self.state.buf, &[0; STATE_BYTES])
                .map_err(|e| e.to_string())
        }
    }

    impl Toy {
        /// Stand-in for a prefill chunk: the state becomes a function of the prefix.
        fn advance(&mut self, gpu: &mut Gpu, prefix: &[u32]) {
            self.position = prefix.len();
            gpu.hip.memcpy_htod(&self.state.buf, &fill(prefix)).unwrap();
        }
        fn bytes(&self, gpu: &mut Gpu) -> Vec<u8> {
            gpu.hip.device_synchronize().unwrap();
            let mut out = vec![0; STATE_BYTES];
            gpu.hip.memcpy_dtoh(&mut out, &self.state.buf).unwrap();
            out
        }
    }

    fn fill(prefix: &[u32]) -> Vec<u8> {
        let seed = prefix_fingerprint(prefix).to_le_bytes();
        (0..STATE_BYTES).map(|i| seed[i % 8] ^ i as u8).collect()
    }

    fn domain() -> CacheDomain {
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
                normalization_tag: "jinja".into(),
            },
            arch_policy: ArchPolicy {
                arch_tag: "toy".into(),
                state_abi_tag: String::new(),
                position_attention_tag: String::new(),
            },
            kv_layout: KvLayout {
                k_stride_bytes: vec![],
                v_stride_bytes: vec![],
                layout_tag: String::new(),
            },
            device: DeviceTopology {
                device_id: "gpu-0".into(),
                topology_id: "single".into(),
                allocation_epoch: 1,
            },
            namespace: SharingNamespace {
                domain_id: "default".into(),
            },
        }
    }

    fn prompt(seed: u32, len: usize) -> Vec<u32> {
        (0..len as u32).map(|i| seed * 100_000 + i).collect()
    }

    /// One turn: restore what `plan` offers, prefill to the end capturing at
    /// every boundary, optionally commit. Returns the planned reuse.
    fn turn(
        cache: &mut SessionCache,
        gpu: &mut Gpu,
        toy: &mut Toy,
        prompt: &[u32],
        commit: bool,
    ) -> usize {
        let reused = cache.plan(toy, prompt, SessionRoute::Ar);
        cache
            .begin(gpu, toy, prompt, SessionRoute::Ar, reused)
            .unwrap();
        while let Some(b) = cache.next_boundary() {
            toy.advance(gpu, &prompt[..b]);
            cache.at_boundary(gpu, toy, &prompt[..b]).unwrap();
        }
        toy.advance(gpu, prompt);
        if commit {
            cache.commit();
        }
        reused
    }

    #[test]
    fn snapshots_restore_exactly_publish_on_commit_and_evict_lru() {
        let Ok(mut gpu) = Gpu::init() else {
            eprintln!("skip: session cache tests require a GPU");
            return;
        };
        let gpu = &mut gpu;
        let state = gpu.alloc_tensor(&[STATE_BYTES], DType::Raw).unwrap();
        let mut toy = Toy { state, position: 0 };
        let one = layout([STATE_BYTES]).1 as u64;
        let mut cache = SessionCache::new(domain(), 2 * one);
        let (a, b, c) = (
            prompt(1, STRIDE + 5),
            prompt(2, STRIDE + 7),
            prompt(3, STRIDE + 9),
        );

        // Uncommitted capture is invisible and abandoned by the next begin.
        assert_eq!(turn(&mut cache, gpu, &mut toy, &a, false), 0);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), 0);
        assert_eq!(turn(&mut cache, gpu, &mut toy, &a, true), 0);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), STRIDE);

        // Restore reproduces the boundary state byte for byte.
        let reused = cache.plan(&toy, &a, SessionRoute::Ar);
        cache
            .begin(gpu, &mut toy, &a, SessionRoute::Ar, reused)
            .unwrap();
        assert_eq!(toy.position, STRIDE);
        assert_eq!(toy.bytes(gpu), fill(&a[..STRIDE]));
        assert_eq!(cache.next_boundary(), None);
        cache.commit();

        // Budget holds two snapshots: a third evicts the least recently used.
        turn(&mut cache, gpu, &mut toy, &b, true);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), STRIDE);
        cache
            .begin(gpu, &mut toy, &a, SessionRoute::Ar, STRIDE)
            .unwrap(); // a is now newer than b
        cache.commit();
        turn(&mut cache, gpu, &mut toy, &c, true);
        assert_eq!(cache.plan(&toy, &b, SessionRoute::Ar), 0);
        assert_eq!(cache.plan(&toy, &a, SessionRoute::Ar), STRIDE);
        assert_eq!(cache.plan(&toy, &c, SessionRoute::Ar), STRIDE);
        cache.clear(gpu);

        // A snapshot larger than the whole budget is skipped, nothing published.
        let mut tiny = SessionCache::new(domain(), one - 1);
        turn(&mut tiny, gpu, &mut toy, &a, true);
        assert_eq!(tiny.plan(&toy, &a, SessionRoute::Ar), 0);
        gpu.free_tensor(toy.state).unwrap();
    }
}
