// SPDX-License-Identifier: Apache-2.0
//! Static schedule evidence for lifted code objects. Ratios describe an emitted
//! loop trip, not elapsed cycles; without routing data padding is unknown.
use peacemaker_ir::{inst::{Arch, Form}, passes::{cfg::Cfg, waits::replay}, reg::Kind, wait::Counter};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Route {
    /// Real rows per routed expert in this capture; absent for dense kernels.
    #[serde(default)] pub rows: Vec<u32>,
    /// Kernel's executed row tile, from its launch contract.
    #[serde(default)] pub tile_rows: u32,
    /// K128 epochs executed in one hot-loop trip, from the kernel tile rule.
    #[serde(default)] pub k128_epochs: Option<usize>,
}
impl Route {
    pub fn padding_fraction(&self) -> Result<Option<f64>, String> {
        if self.rows.is_empty() { return Ok(None); }
        if self.tile_rows == 0 { return Err("tile_rows must be positive when rows are provided".into()); }
        let real: u64 = self.rows.iter().map(|&n| u64::from(n)).sum();
        let executed: u64 = self.rows.iter().map(|&n| u64::from(n).div_ceil(u64::from(self.tile_rows)) * u64::from(self.tile_rows)).sum();
        if executed == 0 { return Err("route has no executed rows".into()); }
        Ok(Some((executed - real) as f64 / executed as f64))
    }
}

#[derive(Debug, Serialize)]
pub struct LoopCost {
    pub blocks: Vec<usize>, pub wmma_per_trip: usize, pub k128_epochs: Option<usize>,
    pub wmma_per_k128: Option<f64>, pub independent_chains: usize,
    pub valu_per_wmma: f64, pub vmem_per_wmma: f64, pub vmem_bytes_per_wmma: f64,
    /// Minimum pending VMEM units immediately before an epoch's first WMMA.
    pub wait_depth_before_wmma: Option<u64>,
    /// Conservative loop-carried distance: VMEM issued before its consuming wait
    /// in the next epoch. None if epoch boundaries cannot be established.
    pub prefetch_distance_epochs: Option<usize>,
    pub padding_fraction: Option<f64>, pub findings: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct KernelCost { pub symbol: String, pub loops: Vec<LoopCost> }
#[derive(Debug, Serialize)]
pub struct CostReport { pub arch: String, pub kernels: Vec<KernelCost> }

fn vmem_bytes(name: &str) -> u64 {
    if let Some(n) = name.rsplit('_').next().and_then(|s| s.strip_prefix('b')).and_then(|s| s.parse::<u64>().ok()) { return n / 8; }
    for (suffix, bytes) in [("_u8",1),("_i8",1),("_u16",2),("_i16",2),("_f16",2),("_b32",4),("_f32",4)] {
        if name.ends_with(suffix) { return bytes; }
    }
    0 // An unmodeled image/atomic byte width is not guessed.
}
fn vmem_pending(state: &peacemaker_ir::wait::WaitState, arch: Arch) -> u64 {
    let counter = if arch == Arch::Gfx1201 { Counter::Load } else { Counter::Vm };
    state.pending.iter().filter(|p| p.counters.contains(counter) && !p.satisfied.contains(counter))
        .map(|p| u64::from(p.units[counter as usize])).sum()
}

pub fn analyze(program: &peacemaker_ir::Program, routes: &BTreeMap<String, Route>) -> Result<CostReport, String> {
    let arch = program.target.arch;
    let mut kernels = Vec::new();
    for kernel in &program.kernels {
        let body = &kernel.body;
        let graph = Cfg::build(body).map_err(|e| format!("{}: {e}", kernel.symbol.0))?;
        let waits = replay(body, arch).map_err(|e| format!("{}: {e}", kernel.symbol.0))?;
        let mut loops = Vec::new();
        for info in graph.loops() {
            // A natural loop may have multiple exits; no dynamic branch frequency
            // is inferred. Each instruction in its member blocks is counted once.
            let mut positions: Vec<usize> = info.members.iter().flat_map(|id| body.blocks.iter()
                .filter(move |block| block.id == *id)
                .flat_map(|block| block.range.0..block.range.1)).collect();
            positions.sort_unstable(); positions.dedup();
            let mut chains = BTreeSet::new();
            let (mut wmma, mut valu, mut vmem, mut bytes) = (0, 0, 0, 0);
            let mut depth = None::<u64>;
            for &pos in &positions {
                let id = body.layout[pos]; let inst = body.insts.get(id).ok_or("dangling instruction")?;
                let name = inst.op.name(arch).unwrap_or("");
                if name.starts_with("v_wmma_") || name.starts_with("v_swmmac_") {
                    wmma += 1;
                    if let Some(dst) = inst.effects.defs.iter().find(|r| r.kind == Kind::V) {
                        // Re-seeding a destination is not a second chain. The
                        // physical destination is the in-loop recurrence identity.
                        chains.insert((dst.base, dst.len));
                    }
                    if let Some(state) = waits.before.get(&id) {
                        let pending = vmem_pending(state, arch);
                        depth = Some(depth.map_or(pending, |d| d.min(pending)));
                    }
                } else if name.starts_with("v_") { valu += 1; }
                if matches!(inst.form, Form::Vmem(_)) {
                    vmem += 1;
                    bytes += vmem_bytes(name);
                }
            }
            if wmma == 0 { continue; }
            let mut findings = Vec::new();
            if chains.len() == 1 { findings.push("single_accumulator_chain".into()); }
            // A full drain matters here when a single-chain, VMEM-heavy loop
            // has no outstanding work before WMMA. Dense multi-chain kernels
            // and low-VMEM loops can legitimately drain without this finding.
            if depth == Some(0) && chains.len() == 1 && vmem * 2 >= wmma {
                findings.push("prefetch_drained_before_wmma".into());
            }
            let route = routes.get(&kernel.symbol.0);
            let padding = route.map(Route::padding_fraction).transpose()?.flatten();
            let epochs = route.and_then(|r| r.k128_epochs);
            if epochs == Some(0) { return Err(format!("{}: k128_epochs must be positive", kernel.symbol.0)); }
            let per_epoch = epochs.map(|n| wmma as f64 / n as f64);
            loops.push(LoopCost { blocks: info.members.iter().map(|id| id.0).collect(), wmma_per_trip: wmma,
                k128_epochs: epochs, wmma_per_k128: per_epoch, independent_chains: chains.len(),
                valu_per_wmma: valu as f64 / wmma as f64, vmem_per_wmma: vmem as f64 / wmma as f64,
                vmem_bytes_per_wmma: bytes as f64 / wmma as f64,
                wait_depth_before_wmma: depth, prefetch_distance_epochs: None,
                padding_fraction: padding, findings });
        }
        kernels.push(KernelCost { symbol: kernel.symbol.0.clone(), loops });
    }
    Ok(CostReport { arch: format!("{arch:?}"), kernels })
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn padding_on_ragged_experts() {
        assert_eq!(Route { rows: vec![1, 16, 17], tile_rows: 16, ..Default::default() }.padding_fraction().unwrap(), Some(30.0/64.0));
        assert!(Route { rows: vec![1], tile_rows: 0, ..Default::default() }.padding_fraction().is_err());
    }

    /// External historical modules are not checked into the source tree; the
    /// gate runs whenever the pinned CPU-only calibration corpus is mounted.
    #[test] fn wait_delete_flags_six_and_not_the_other_twenty() {
        let root = std::path::Path::new("/home/kaden/qcal/release-0.4.1/pm-wait-delete");
        if !root.join("before/sym1151.co").exists() { return; }
        let mut total = 0;
        let mut newly_flagged = 0;
        for module in ["b1", "b1s", "v2b", "v2c", "sym1151", "sym1201"] {
            let lift = |phase: &str| {
                let object = std::fs::read(root.join(phase).join(format!("{module}.co"))).unwrap();
                let lifted = peacemaker_lift::lift_object(&object, peacemaker_lift::Options {
                    frontend: peacemaker_ir::inst::Frontend::Hipcc,
                }).unwrap();
                analyze(&lifted.program, &BTreeMap::new()).unwrap()
            };
            let before = lift("before"); let after = lift("after");
            for (a,b) in before.kernels.iter().zip(&after.kernels) {
                assert_eq!(a.symbol, b.symbol);
                let previously = a.loops.iter().any(|l| l.findings.iter().any(|f| f == "prefetch_drained_before_wmma"));
                let currently = b.loops.iter().any(|l| l.findings.iter().any(|f| f == "prefetch_drained_before_wmma"));
                let regressed = a.symbol.contains("_nt4") || a.symbol.contains("_nt8");
                assert_eq!(currently && !previously, regressed, "{}", a.symbol);
                newly_flagged += usize::from(currently && !previously);
                total += 1;
            }
        }
        assert_eq!((total, newly_flagged), (26, 6));
    }
}
