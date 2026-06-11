use crate::plan::ReapPlan;
use hipfire_runtime::hfq::{HfqFile, HfqTensorInfo};

/// Overlay-then-base tensor resolver. SP1: overlay is always None (base only);
/// SP3 adds the overlay HfqFile and prefers it when it holds `name`.
pub struct TensorSource<'a> {
    pub base: &'a HfqFile,
    pub overlay: Option<&'a HfqFile>,
}

impl<'a> TensorSource<'a> {
    pub fn new(base: &'a HfqFile) -> Self {
        TensorSource { base, overlay: None }
    }

    /// Resolve a tensor by name: overlay first (SP3), else base.
    /// Returns `(&HfqTensorInfo, Vec<u8>)` using `tensor_data_vec` for
    /// owned bytes (avoids RefCell borrow and works on all platforms).
    pub fn tensor(&self, name: &str) -> Option<(&'a HfqTensorInfo, Vec<u8>)> {
        if let Some(ov) = self.overlay {
            if let Some(hit) = ov.tensor_data_vec(name) {
                return Some(hit);
            }
        }
        self.base.tensor_data_vec(name)
    }
}

/// Per-(layer, role) plan slice the arch loader consumes at its expert loop.
pub struct ExpertPlan<'a> {
    /// `keep[slot]` = original expert index for compact slot. `None` ⇒ identity.
    keep: Option<&'a [u32]>,
}

impl<'a> ExpertPlan<'a> {
    pub fn keep(&self) -> Option<&'a [u32]> {
        self.keep
    }
    /// Original expert index for a compact slot (identity when no keep map).
    /// Panics if slot >= n_slots(full); callers must iterate 0..n_slots(full).
    pub fn src(&self, slot: usize) -> usize {
        self.keep.map(|k| k[slot] as usize).unwrap_or(slot)
    }
    /// Number of compact expert slots for this layer.
    pub fn n_slots(&self, full: usize) -> usize {
        self.keep.map(|k| k.len()).unwrap_or(full)
    }
}

impl ReapPlan {
    /// Build the per-layer expert plan (routed experts). `None`-keep ⇒ identity.
    pub fn expert_plan(&self, layer: usize) -> ExpertPlan<'_> {
        ExpertPlan {
            keep: self.keep.as_ref().map(|k| k[layer].as_slice()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ExpertPlan::src/n_slots are pure; test them without an HfqFile.
    #[test]
    fn identity_src_when_no_keep() {
        let ep = ExpertPlan { keep: None };
        assert_eq!(ep.src(5), 5);
        assert_eq!(ep.n_slots(8), 8);
    }
    #[test]
    fn remaps_src_with_keep() {
        let k = vec![3u32, 1, 0];
        let ep = ExpertPlan { keep: Some(&k) };
        assert_eq!(ep.src(0), 3);
        assert_eq!(ep.n_slots(8), 3);
    }
}
