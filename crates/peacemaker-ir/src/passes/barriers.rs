//! C5: split-barrier pairing and drain checks.
//!
//! For every `s_barrier_signal` the pass proves, over the CFG, that each
//! path meets exactly one `s_barrier_wait` before the next signal or
//! `s_endpgm` (and symmetrically that every wait is preceded by a signal on
//! every path). The forward/backward scans are memoized per
//! `(BlockId, position)` so loops terminate: a path that cycles without
//! meeting a barrier fails the pairing. Additionally every barrier
//! (signal, wait, or full `s_barrier`) must see drained DS stores —
//! checked against the [`WaitReplay`] states, mirroring the builder's
//! `pending_stores` drain before `barrier_signal` (`lib.rs:111-118`).

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::cfg::{BlockId, Body, InstId};
use crate::effects::Control;
use crate::inst::Arch;
use crate::passes::waits::{self, WaitReplay};
use crate::state::{Obligation, ObligationKind};
use crate::wait::Counter;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum BarrierError {
    #[error("barrier analysis needs the gfx1201 table (got {0:?})")]
    UnsupportedArch(Arch),
    #[error(transparent)]
    Waits(#[from] waits::WaitError),
    #[error("layout refers to a tombstoned or missing instruction")]
    DanglingInst { id: InstId },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BarrierPair {
    pub signal: InstId,
    pub waits: Vec<InstId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BarrierAnalysis {
    pub pairs: Vec<BarrierPair>,
    pub obligations: Vec<Obligation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Scan {
    Good(Vec<InstId>),
    Bad,
}

struct Ctx<'a> {
    body: &'a Body,
    /// Block ranges in layout order plus successor block ids.
    blocks: Vec<(BlockId, (usize, usize), Vec<BlockId>)>,
    range_of: HashMap<BlockId, (usize, usize)>,
    succ_of: HashMap<BlockId, Vec<BlockId>>,
}

impl<'a> Ctx<'a> {
    fn new(body: &'a Body) -> Self {
        let mut blocks = Vec::new();
        if body.blocks.is_empty() {
            blocks.push((BlockId(0), (0, body.layout.len()), Vec::new()));
        } else {
            for block in &body.blocks {
                blocks.push((
                    block.id,
                    block.range,
                    block.succs.iter().copied().collect(),
                ));
            }
        }
        let mut range_of = HashMap::new();
        let mut succ_of = HashMap::new();
        for (id, range, succs) in &blocks {
            range_of.insert(*id, *range);
            succ_of.insert(*id, succs.clone());
        }
        Self { body, blocks, range_of, succ_of }
    }

    fn barrier_at(&self, pos: usize) -> Option<(InstId, bool)> {
        let id = *self.body.layout.get(pos)?;
        let inst = self.body.insts.get(id)?;
        match inst.effects.control {
            Control::Barrier(crate::cfg::BarrierKind::Signal(_)) => Some((id, true)),
            Control::Barrier(crate::cfg::BarrierKind::Wait) => Some((id, false)),
            Control::Barrier(_) => None,
            _ => None,
        }
    }
}

/// Forward scan from `pos` (inclusive): the first barrier on every path
/// must be a wait; collects the waits met.
fn forward(
    ctx: &Ctx,
    block: BlockId,
    pos: usize,
    memo: &mut HashMap<(BlockId, usize), Scan>,
    active: &mut HashSet<(BlockId, usize)>,
) -> Scan {
    let key = (block, pos);
    if let Some(known) = memo.get(&key) {
        return known.clone();
    }
    if !active.insert(key) {
        return Scan::Bad;
    }
    let result = forward_inner(ctx, block, pos, memo, active);
    active.remove(&key);
    memo.insert(key, result.clone());
    result
}

fn forward_inner(
    ctx: &Ctx,
    block: BlockId,
    pos: usize,
    memo: &mut HashMap<(BlockId, usize), Scan>,
    active: &mut HashSet<(BlockId, usize)>,
) -> Scan {
    let range = ctx.range_of[&block];
    for at in pos..range.1 {
        if let Some((id, is_signal)) = ctx.barrier_at(at) {
            if is_signal {
                return Scan::Bad;
            }
            return Scan::Good(vec![id]);
        }
        let id = ctx.body.layout[at];
        if let Some(inst) = ctx.body.insts.get(id) {
            if matches!(inst.effects.control, Control::EndPgm) {
                return Scan::Bad;
            }
        }
    }
    let mut waits = Vec::new();
    let succs = ctx.succ_of[&block].clone();
    if succs.is_empty() {
        return Scan::Bad;
    }
    for succ in succs {
        match forward(ctx, succ, ctx.range_of[&succ].0, memo, active) {
            Scan::Good(mut found) => waits.append(&mut found),
            Scan::Bad => return Scan::Bad,
        }
    }
    waits.sort();
    waits.dedup();
    Scan::Good(waits)
}

/// Backward scan from `pos` (exclusive): the first barrier on every
/// reversed path must be a signal.
fn backward(
    ctx: &Ctx,
    preds: &HashMap<BlockId, Vec<BlockId>>,
    block: BlockId,
    pos: usize,
    memo: &mut HashMap<(BlockId, usize), Scan>,
    active: &mut HashSet<(BlockId, usize)>,
) -> Scan {
    let key = (block, pos);
    if let Some(known) = memo.get(&key) {
        return known.clone();
    }
    if !active.insert(key) {
        return Scan::Bad;
    }
    let result = backward_inner(ctx, preds, block, pos, memo, active);
    active.remove(&key);
    memo.insert(key, result.clone());
    result
}

fn backward_inner(
    ctx: &Ctx,
    preds: &HashMap<BlockId, Vec<BlockId>>,
    block: BlockId,
    pos: usize,
    memo: &mut HashMap<(BlockId, usize), Scan>,
    active: &mut HashSet<(BlockId, usize)>,
) -> Scan {
    let range = ctx.range_of[&block];
    for at in (range.0..pos).rev() {
        if let Some((id, is_signal)) = ctx.barrier_at(at) {
            if is_signal {
                return Scan::Good(vec![id]);
            }
            return Scan::Bad;
        }
    }
    let incoming = preds.get(&block).cloned().unwrap_or_default();
    if incoming.is_empty() {
        return Scan::Bad;
    }
    let mut signals = Vec::new();
    for pred in incoming {
        match backward(ctx, preds, pred, ctx.range_of[&pred].1, memo, active) {
            Scan::Good(mut found) => signals.append(&mut found),
            Scan::Bad => return Scan::Bad,
        }
    }
    signals.sort();
    signals.dedup();
    Scan::Good(signals)
}

fn ds_stores_pending(replay: &WaitReplay, id: InstId) -> bool {
    replay
        .before
        .get(&id)
        .is_some_and(|state| {
            state.pending.iter().any(|event| {
                event.counters.contains(Counter::Ds) && !event.src_locks.0.is_empty()
            })
        })
}

fn drain_obligation(id: InstId, what: &str) -> Obligation {
    Obligation {
        kind: ObligationKind::Hazard,
        insts: vec![id],
        rule_id: "barrier-ds-pending".into(),
        text: format!("{what} with a DS store still holding DScnt"),
    }
}

/// Barrier pairing plus DS-drain checks for one kernel body.
pub fn analyze(body: &Body, arch: Arch) -> Result<BarrierAnalysis, BarrierError> {
    if arch != Arch::Gfx1201 {
        return Err(BarrierError::UnsupportedArch(arch));
    }
    for id in &body.layout {
        if body.insts.get(*id).is_none() {
            return Err(BarrierError::DanglingInst { id: *id });
        }
    }
    let replay = waits::replay(body, arch)?;
    let ctx = Ctx::new(body);
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for (id, _, succs) in &ctx.blocks {
        for succ in succs {
            preds.entry(*succ).or_default().push(*id);
        }
    }
    let mut analysis = BarrierAnalysis::default();
    let mut forward_memo = HashMap::new();
    let mut backward_memo = HashMap::new();
    for (index, (block, range, _)) in ctx.blocks.iter().enumerate() {
        for pos in range.0..range.1 {
            let id = body.layout[pos];
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            match inst.effects.control {
                Control::Barrier(crate::cfg::BarrierKind::Signal(_)) => {
                    let mut active = HashSet::new();
                    match forward(&ctx, *block, pos + 1, &mut forward_memo, &mut active) {
                        Scan::Good(waits) => {
                            analysis.pairs.push(BarrierPair { signal: id, waits });
                        }
                        Scan::Bad => analysis.obligations.push(Obligation {
                            kind: ObligationKind::Hazard,
                            insts: vec![id],
                            rule_id: "barrier-signal-unpaired".into(),
                            text: "s_barrier_signal is not followed by exactly one s_barrier_wait on every path".into(),
                        }),
                    }
                    if ds_stores_pending(&replay, id) {
                        analysis.obligations.push(drain_obligation(id, "s_barrier_signal"));
                    }
                    let _ = index;
                }
                Control::Barrier(crate::cfg::BarrierKind::Wait) => {
                    let mut active = HashSet::new();
                    if matches!(
                        backward(&ctx, &preds, *block, pos, &mut backward_memo, &mut active),
                        Scan::Bad
                    ) {
                        analysis.obligations.push(Obligation {
                            kind: ObligationKind::Hazard,
                            insts: vec![id],
                            rule_id: "barrier-wait-unsignalled".into(),
                            text: "s_barrier_wait is reachable without a preceding s_barrier_signal on some path".into(),
                        });
                    }
                    if ds_stores_pending(&replay, id) {
                        analysis.obligations.push(drain_obligation(id, "s_barrier_wait"));
                    }
                }
                Control::Barrier(_) => {
                    if ds_stores_pending(&replay, id) {
                        analysis.obligations.push(drain_obligation(id, "barrier"));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(analysis)
}

#[cfg(test)]
mod kt48_tests {
    use crate::cfg::Body;
    use crate::inst::Arch;
    use crate::passes::cfg::build_blocks;

    fn kt48_body() -> Body {
        const IMAGE: &[u8] =
            include_bytes!("../../../peacemaker-lift/tests/fixtures/kt48/hipcc.co");
        const START: usize = 0x6f00;
        const SIZE: usize = 10_604;
        let words: Vec<u32> = IMAGE[START..START + SIZE]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let mut body = Body::default();
        let mut index = 0;
        while index < words.len() {
            let (inst, count) = crate::codec::gfx12::decode(&words[index..]).expect("KT48 decodes");
            let id = body.insts.insert(inst);
            body.layout.push(id);
            index += count;
        }
        build_blocks(&mut body).expect("CFG");
        body
    }

    /// Every split barrier in KT48 pairs signal→wait on all paths with DS
    /// stores drained at both ends.
    #[test]
    fn kt48_barriers_pair_cleanly() {
        let body = kt48_body();
        let analysis = super::analyze(&body, Arch::Gfx1201).unwrap();
        assert!(analysis.obligations.is_empty(), "{:?}", analysis.obligations);
        assert!(!analysis.pairs.is_empty(), "KT48 uses split barriers");
        for pair in &analysis.pairs {
            assert_eq!(pair.waits.len(), 1);
        }
    }
}
