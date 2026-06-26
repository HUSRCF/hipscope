// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Grégory D
// hipfire — see LICENSE and NOTICE in the project root.

//! DDTree: tree-structured speculative verification built from DFlash's
//! per-position draft marginals.
//!
//! Port of Ringel & Romano's Algorithm 1 (MIT-licensed reference at
//! `github.com/liranringel/ddtree`, cached locally at `/tmp/ddtree_ref/`).
//! The reference is PyTorch; this is a pure-Rust port of the tree-construction
//! and greedy-walk logic. The target-verify stage is separate (lives in
//! `speculative::spec_step_ddtree`) because the hybrid Qwen3.5 architecture
//! (24 DeltaNet + 8 FullAttention layers) forces a per-branch DFS walk
//! with state snapshot/restore rather than the reference's single-pass
//! batched tree attention — LA layers don't accept an attention mask, so
//! "run the whole tree in one forward" would pollute recurrent state.
//!
//! What this module owns:
//!   - `DdTree` construction from per-position top-K (token, log-prob) pairs
//!   - Visibility matrix (ancestor-only)
//!   - `follow_verified_tree` — greedy walk selecting the longest accepted path
//!
//! What it doesn't own:
//!   - Target forwards (those live in `speculative::spec_step_ddtree`)
//!   - KV compaction (same)
//!   - Draft-side top-K extraction (computed in the caller from DFlash logits)

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// A tree node. Index 0 is implicit (the "root" = seed/anchor token the
/// caller already holds); stored nodes live in `DdTree::nodes[0..N]` and
/// are addressed by their index + 1 in the visibility / child-map tables.
#[derive(Debug, Clone)]
pub struct DdNode {
    /// The token this node proposes. Sourced from the draft's top-K at depth.
    pub token: u32,
    /// 1-indexed depth relative to root. Depth 1 = direct child of the root,
    /// depth D = D layers beneath root (matches the reference's `node_depths`).
    pub depth: u32,
    /// Index in `DdTree.nodes` of this node's parent, or -1 if parent == root.
    /// Note: -1 here is the "root is parent" sentinel, NOT "no parent".
    pub parent_index: i32,
    /// Cumulative draft log-probability of the path root→this node (sum of the
    /// per-depth conditional log-probs along the path). The conditional draft
    /// prob of this node given its parent is `exp(logw − parent.logw)` (with
    /// `parent.logw = 0` when the parent is the root). Not needed by the
    /// distribution-preserving `sample_verified_tree` (which samples the target,
    /// not the draft), but kept as cheap metadata for diagnostics and any
    /// future draft-conditional (typical-acceptance / CACTUS-style) variant.
    pub logw: f32,
}

/// A speculative-verification tree.
///
/// Fields match the reference's Python layout — callers that need to interop
/// with test vectors / debug dumps can read them directly. The convention
/// `0 = root / 1..=N = tree nodes` matches the reference exactly.
pub struct DdTree {
    /// N tree nodes (root is implicit at index 0 and is not stored).
    pub nodes: Vec<DdNode>,
    /// Ancestor visibility. `visibility[i][j] == true` iff node `j` is an
    /// ancestor of node `i` (inclusive, with the convention that root is
    /// ancestor of every node and of itself). Dimensions: (1 + N) × (1 + N).
    /// Row/col 0 refers to the root; row/col i>0 refers to `nodes[i-1]`.
    pub visibility: Vec<Vec<bool>>,
    /// Per-node adjacency: `child_maps[i]` is the map `token → child_index`
    /// (index into `nodes`) for children of the node at index `i`.
    /// `child_maps[0]` holds the root's children. Size: 1 + N.
    pub child_maps: Vec<HashMap<u32, usize>>,
}

impl DdTree {
    /// Number of stored nodes (root-exclusive).
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Walk from a node back to (but not including) root, collecting ancestor
    /// indices in root-to-self order. Used during DFS verify to know which
    /// KV slots / DeltaNet snapshot to rewind to at each branch step.
    pub fn ancestors_of(&self, node_index: usize) -> Vec<usize> {
        let mut chain: Vec<usize> = Vec::new();
        let mut cur = node_index as i32;
        while cur >= 0 {
            chain.push(cur as usize);
            cur = self.nodes[cur as usize].parent_index;
        }
        chain.reverse();
        chain
    }
}

/// Min-heap wrapper for f32 (smaller popped first). Tie-breaks by push order
/// to reproduce Python's heapq stability (which is important because the
/// reference uses a `ranks` tuple as the secondary key, and equal log-weights
/// occur routinely on near-uniform distributions).
#[derive(PartialEq)]
struct HeapEntry {
    neg_logw: f32,   // negated so BinaryHeap (max-heap) pops MIN neg_logw = MAX logw first
    push_order: u64, // FIFO tie-break — earlier pushes win on equal neg_logw
    depth: usize,    // 1-indexed; 1 = child of root
    rank: usize,     // position in the top-K at this depth
    parent_index: i32, // -1 = parent is root; else nodes[parent_index]
    logw: f32,       // cumulative log-weight of the path root→this-candidate
}

impl Eq for HeapEntry {}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-heap. We want MIN neg_logw (= MAX logw) first.
        // NaN treated as equal — shouldn't occur (log-softmax is finite), but
        // if it does we prefer not to panic.
        match other
            .neg_logw
            .partial_cmp(&self.neg_logw)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => other.push_order.cmp(&self.push_order),
            ord => ord,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Build a DDTree from per-position top-K draft marginals (Algorithm 1,
/// Ringel & Romano).
///
/// Arguments:
/// - `top_tokens`: row-major `[depth × topk]` u32 array. `top_tokens[d*topk+k]`
///   is the k-th most likely draft token at position d (0-indexed).
/// - `top_log_probs`: matching `[depth × topk]` f32 array of log-probabilities
///   (normalized, i.e. logits minus per-row log-sum-exp).
/// - `depth`: number of draft positions (usually B - 1 where B is block size).
/// - `topk`: K. Must equal the second dim of the arrays.
/// - `budget`: max nodes in the output tree (paper: 60). Must be ≥ 0.
///
/// Returns a `DdTree` with `min(budget, reachable)` nodes. If `depth == 0`
/// or `budget == 0` the tree is empty (visibility still contains the 1×1
/// root-only row so downstream callers can probe it uniformly).
pub fn build_ddtree_tree(
    top_tokens: &[u32],
    top_log_probs: &[f32],
    depth: usize,
    topk: usize,
    budget: usize,
) -> DdTree {
    build_ddtree_tree_with_cutoff(
        top_tokens,
        top_log_probs,
        depth,
        topk,
        budget,
        f32::NEG_INFINITY,
    )
}

/// Same as `build_ddtree_tree`, but also stops expansion when the next
/// heap-pop candidate's cumulative log-weight falls below `logw_cutoff`.
/// Use `f32::NEG_INFINITY` to disable (= original behaviour — only the
/// `budget` cap applies).
///
/// Rationale: the reference's Algorithm 1 pops candidates in
/// descending-cumulative-logw order, so later pops are strictly lower
/// probability than earlier ones. When a candidate's cumulative log-prob
/// drops below, say, -4.0 (≈ 1.8 % absolute probability), further
/// expansion has diminishing returns — those slots are rarely accepted
/// by the target anyway, yet each costs verify time linear in B.
///
/// This is a zero-training "meta-verifier" pruner: per-cycle dynamic
/// budget that shrinks the tree on high-confidence cycles (where top-1
/// logw is near zero and the heap's tail collapses fast) but keeps full
/// budget on uncertain ones (where many candidates are plausible).
///
/// Measured on 27B MQ4 / 7900XTX (2026-04-24, 3-run median):
/// TODO: populate once the bench lands.
pub fn build_ddtree_tree_with_cutoff(
    top_tokens: &[u32],
    top_log_probs: &[f32],
    depth: usize,
    topk: usize,
    budget: usize,
    logw_cutoff: f32,
) -> DdTree {
    // Thin wrapper: no floor (min_nodes=0), budget is the ceiling.
    build_ddtree_tree_bounded(
        top_tokens,
        top_log_probs,
        depth,
        topk,
        0,
        budget,
        logw_cutoff,
    )
}

/// Best-first DDTree build with a node-count band `[min_nodes, max_nodes]`.
///
/// The `logw_cutoff` (meta-verifier pruner) is the *shape* control — it shrinks
/// the tree on confident steps — but its raw node count swings with the step's
/// distribution (a near-chain when peaked, near-full when flat). This bands it:
/// expand the most-likely nodes *ignoring* the cutoff until `min_nodes` are
/// placed (a floor, so the verify is never wastefully tiny), then let the cutoff
/// prune, and stop at `max_nodes` (the ceiling, so verify cost is capped). The
/// result is always in `[min(min_nodes, reachable), max_nodes]`, with the cutoff
/// steering depth-vs-breadth within the band. `min_nodes=0` ⇒ pure cutoff
/// (the historical `build_ddtree_tree_with_cutoff`).
#[allow(clippy::too_many_arguments)]
pub fn build_ddtree_tree_bounded(
    top_tokens: &[u32],
    top_log_probs: &[f32],
    depth: usize,
    topk: usize,
    min_nodes: usize,
    max_nodes: usize,
    logw_cutoff: f32,
) -> DdTree {
    let budget = max_nodes;
    // Early out: no draft positions or no budget → root-only tree.
    if budget == 0 || depth == 0 {
        return DdTree {
            nodes: Vec::new(),
            visibility: vec![vec![true]],
            child_maps: vec![HashMap::new()],
        };
    }
    assert_eq!(
        top_tokens.len(),
        depth * topk,
        "top_tokens size mismatch: expected {}, got {}",
        depth * topk,
        top_tokens.len()
    );
    assert_eq!(
        top_log_probs.len(),
        depth * topk,
        "top_log_probs size mismatch"
    );

    // Seed heap with the root's best child (depth 1, rank 0). The reference
    // stores a `ranks` tuple to tie-break across otherwise-equal priorities;
    // we use a push-order counter, which is functionally equivalent because
    // ranks monotonically increase along each sibling chain.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::new();
    let mut push_counter: u64 = 0;
    let first_logw = top_log_probs[0];
    heap.push(HeapEntry {
        neg_logw: -first_logw,
        push_order: push_counter,
        depth: 1,
        rank: 0,
        parent_index: -1,
        logw: first_logw,
    });
    push_counter += 1;

    let mut nodes: Vec<DdNode> = Vec::with_capacity(budget);
    let mut child_maps: Vec<HashMap<u32, usize>> = Vec::with_capacity(budget + 1);
    child_maps.push(HashMap::new()); // root

    while let Some(entry) = heap.pop() {
        if nodes.len() >= budget {
            break;
        }
        // Meta-verifier pruner: heap pops in strictly descending logw, so
        // once a candidate falls below the cutoff, every remaining one is
        // also below. Bail early to shrink the tree for high-confidence
        // cycles — verify cost saved ∝ nodes-pruned, acceptance loss ≈ 0
        // (those nodes' target-accept probability is bounded by exp(logw)).
        // The `min_nodes` floor delays the cutoff: always place at least that
        // many highest-probability nodes first, so the band's lower bound holds.
        if nodes.len() >= min_nodes && entry.logw < logw_cutoff {
            break;
        }
        let HeapEntry {
            depth: d,
            rank,
            parent_index,
            logw,
            ..
        } = entry;

        // Add the node at (d, rank). The token and log-prob come from the
        // top-K table. Child index convention matches the reference:
        // nodes are assigned sequential indices, starting at 1 (root is 0).
        let token = top_tokens[(d - 1) * topk + rank];
        let current_index = nodes.len(); // 0-indexed into nodes; +1 in the row/col convention
        nodes.push(DdNode {
            token,
            depth: d as u32,
            parent_index,
            logw,
        });
        child_maps.push(HashMap::new());
        // Register this node as a child of its parent by its draft token.
        // `parent_slot` maps the root-indexed convention (parent = -1 → slot 0).
        let parent_slot = if parent_index < 0 {
            0
        } else {
            (parent_index as usize) + 1
        };
        child_maps[parent_slot].insert(token, current_index);

        // Push sibling at (d, rank+1) if any remain in the top-K at this depth.
        // Sibling's log-weight = parent_logw + top_log_probs[d-1, rank+1] (i.e.
        // replace the current rank's contribution with the next one).
        if rank + 1 < topk {
            let rank_next = rank + 1;
            let sibling_logw = logw - top_log_probs[(d - 1) * topk + rank]
                + top_log_probs[(d - 1) * topk + rank_next];
            heap.push(HeapEntry {
                neg_logw: -sibling_logw,
                push_order: push_counter,
                depth: d,
                rank: rank_next,
                parent_index,
                logw: sibling_logw,
            });
            push_counter += 1;
        }

        // Push child at (d+1, 0) if there's a deeper draft position available.
        if d < depth {
            let child_logw = logw + top_log_probs[d * topk + 0];
            heap.push(HeapEntry {
                neg_logw: -child_logw,
                push_order: push_counter,
                depth: d + 1,
                rank: 0,
                parent_index: current_index as i32,
                logw: child_logw,
            });
            push_counter += 1;
        }
    }

    // Visibility: ancestor-only. Computed bottom-up — node i's row equals
    // its parent's row ∪ {i}. Matches `visibility_np` in the reference.
    let n = nodes.len();
    let len = 1 + n;
    let mut visibility: Vec<Vec<bool>> = vec![vec![false; len]; len];
    visibility[0][0] = true;
    for i in 1..len {
        let parent_slot = {
            let p = nodes[i - 1].parent_index;
            if p < 0 {
                0
            } else {
                (p as usize) + 1
            }
        };
        // Clone parent's ancestor set.
        for j in 0..i {
            visibility[i][j] = visibility[parent_slot][j];
        }
        visibility[i][i] = true;
    }

    DdTree {
        nodes,
        visibility,
        child_maps,
    }
}

/// Greedy walk (Algorithm 2 / `follow_verified_tree`): starting at root,
/// at each step move to the child whose token matches `posterior[current]`
/// (= target's argmax/sampled token AT that tree slot). Stop when no child
/// matches. Returns:
/// - `accepted_indices`: indices into `tree.nodes` of accepted nodes, in order
///   from root's first accepted child down to the deepest accepted descendant.
///   NOTE: root (implicit index 0) is NOT included — just the accepted
///   "tree" nodes we commit to the output stream.
/// - `bonus_token`: the first non-matching posterior token = what target
///   predicts after the accepted path. Committed as the next cycle's seed.
///
/// `posterior` has length 1 + nodes.len(); `posterior[0]` is target's
/// prediction AT the root (i.e. what comes after seed); `posterior[i+1]`
/// is target's prediction AT `tree.nodes[i]`.
pub fn follow_verified_tree(tree: &DdTree, posterior: &[u32]) -> (Vec<usize>, u32) {
    debug_assert_eq!(
        posterior.len(),
        1 + tree.nodes.len(),
        "posterior length must equal 1 + number of tree nodes"
    );
    let mut accepted: Vec<usize> = Vec::new();
    let mut current_slot: usize = 0; // root
    let mut next_token: u32 = posterior[current_slot];
    loop {
        let Some(&child_node_index) = tree.child_maps[current_slot].get(&next_token) else {
            break;
        };
        accepted.push(child_node_index);
        // Advance: new "current" is the accepted child. Its slot = child_node_index + 1.
        current_slot = child_node_index + 1;
        if current_slot >= posterior.len() {
            break;
        }
        next_token = posterior[current_slot];
    }
    (accepted, next_token)
}

/// Xorshift64* → uniform [0, 1). Bit-compatible with the qwen35 spec sampler's
/// RNG (`speculative::xorshift_next_unit`) so the tree accept walk is
/// deterministic given a seed and consistent with the linear DFlash path.
#[inline]
fn xorshift_unit(state: &mut u64) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    ((s >> 40) as f32) * (1.0 / 16_777_216.0)
}

/// Temperature softmax of one logits row into `out`.
fn softmax_temp_into(logits: &[f32], temp: f32, out: &mut Vec<f32>) {
    out.clear();
    out.reserve(logits.len());
    let inv_t = 1.0 / temp;
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let s = v * inv_t;
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v * inv_t - max).exp();
        out.push(e);
        sum += e;
    }
    let inv = 1.0 / sum;
    for p in out.iter_mut() {
        *p *= inv;
    }
}

/// Sample an index from a possibly sub-normalized weight vector (the residual
/// target after rejection subtractions sums to < 1). Falls back to argmax if
/// total positive mass is ≤ 0.
fn sample_unnormalized(w: &[f32], u: f32) -> u32 {
    let mut sum = 0.0f32;
    for &x in w {
        if x > 0.0 {
            sum += x;
        }
    }
    if sum <= 0.0 {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in w.iter().enumerate() {
            if x > bv {
                bv = x;
                bi = i;
            }
        }
        return bi as u32;
    }
    let target = u * sum;
    let mut acc = 0.0f32;
    for (i, &x) in w.iter().enumerate() {
        if x > 0.0 {
            acc += x;
            if target < acc {
                return i as u32;
            }
        }
    }
    (w.len() - 1) as u32
}

/// Phase-0 instrumentation: append one JSON-lines record describing this cycle's
/// tree as a per-slot REDUCED categorical `{child_1..child_k, TAIL}` over both the
/// target `p` (softmax of the verify logits at `temp`) and the draft conditional
/// `q` (`exp(node.logw − parent.logw)`). Children are the only reusable tokens, so
/// accept decisions only ever involve this small support — collapsing all non-child
/// tokens into one TAIL bucket is exact for accept-length simulation and needs no
/// full-vocab draft logits. Consumed by the `ddtree_pq_sim` example to A/B
/// greedy/naive/WR/SWOR acceptance offline (see the q-exploiting-verify plan).
///
/// Record shape (one line):
/// `{"cycle":C,"slots":[{"s":S,"argmax_child":CS,"p_tail":PT,"q_tail":QT,
///    "children":[[child_slot,q,p],...]},...]}`
/// where `child_slot = child_node_index + 1`, `argmax_child` = the child slot whose
/// token is the global argmax of `p` at that slot (or `-1` if the argmax is a
/// non-child / tail token).
pub fn dump_pq_jsonl(
    path: &str,
    tree: &DdTree,
    logits_per_pos: &[f32],
    vocab: usize,
    temp: f32,
    cycle: u64,
) {
    use std::io::Write;
    let n_slots = 1 + tree.nodes.len();
    if logits_per_pos.len() != n_slots * vocab {
        return; // only valid on the temp>0 full-logits path
    }
    let mut p: Vec<f32> = Vec::with_capacity(vocab);
    let mut out = String::new();
    out.push_str(&format!("{{\"cycle\":{cycle},\"slots\":["));
    for s in 0..n_slots {
        let row = &logits_per_pos[s * vocab..(s + 1) * vocab];
        softmax_temp_into(row, if temp > 0.0 { temp } else { 1.0 }, &mut p);
        let argmax_tok = {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        };
        let parent_logw = if s == 0 { 0.0 } else { tree.nodes[s - 1].logw };
        let mut children: Vec<(usize, f32, f32)> = Vec::new();
        let mut p_children = 0.0f32;
        let mut q_children = 0.0f32;
        let mut argmax_child: i64 = -1;
        for (&token, &child_idx) in tree.child_maps[s].iter() {
            let q = (tree.nodes[child_idx].logw - parent_logw)
                .exp()
                .clamp(0.0, 1.0);
            let pt = p[token as usize];
            p_children += pt;
            q_children += q;
            if token == argmax_tok {
                argmax_child = (child_idx + 1) as i64;
            }
            children.push((child_idx + 1, q, pt));
        }
        children.sort_unstable_by_key(|&(cs, _, _)| cs);
        let p_tail = (1.0 - p_children).max(0.0);
        let q_tail = (1.0 - q_children).max(0.0);
        if s > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"s\":{s},\"argmax_child\":{argmax_child},\"p_tail\":{p_tail:.6},\"q_tail\":{q_tail:.6},\"children\":["
        ));
        for (i, (cs, q, pt)) in children.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!("[{cs},{q:.6},{pt:.6}]"));
        }
        out.push_str("]}");
    }
    out.push_str("]}\n");
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = f.write_all(out.as_bytes());
    }
}

/// Distribution-preserving tree verify — the temp>0 acceptance rule, replacing
/// `follow_verified_tree`'s greedy argmax walk.
///
/// This is SpecInfer's "naive sampling" tree verification: at each slot, draw a
/// token `x` from the **target** distribution (softmax of the verify logits at
/// `temp`); if `x` matches one of the drafted children, accept that child and
/// descend; otherwise `x` is the bonus and the walk stops. Because every emitted
/// token is drawn directly from the target, the output distribution is preserved
/// EXACTLY at any temperature — no rejection ratio, no dependence on the draft
/// probabilities (the tree only affects *how many* target draws get reused, i.e.
/// acceptance length, not WHAT is emitted). Acceptance length rises with the
/// target mass the tree covers, so wider trees (more siblings / the banded
/// `MINNODES` floor) raise τ here — the regime the greedy verify could not.
///
/// `logits_per_pos` is the target's full logits, row-major `[(1+N) × vocab]`,
/// row `s` = the target distribution at tree slot `s` (slot 0 = root/seed,
/// slot `i+1` = `nodes[i]`).
///
/// At `temp → 0` the target draw is its argmax, so this reduces EXACTLY to
/// `follow_verified_tree` (descend the child whose token is the argmax, else the
/// argmax is the bonus). Returns `(accepted_node_indices, bonus_token)`, same
/// shape as `follow_verified_tree`.
pub fn sample_verified_tree(
    tree: &DdTree,
    logits_per_pos: &[f32],
    vocab: usize,
    temp: f32,
    rng_state: &mut u64,
) -> (Vec<usize>, u32) {
    let n_slots = 1 + tree.nodes.len();
    debug_assert_eq!(logits_per_pos.len(), n_slots * vocab);

    let mut accepted: Vec<usize> = Vec::new();
    let mut current_slot: usize = 0; // root
    let mut p: Vec<f32> = Vec::with_capacity(vocab);

    loop {
        let row = &logits_per_pos[current_slot * vocab..(current_slot + 1) * vocab];
        // Draw the target's token at this slot (argmax when greedy).
        let x: u32 = if temp > 0.0 {
            softmax_temp_into(row, temp, &mut p);
            let u = xorshift_unit(rng_state);
            sample_unnormalized(&p, u)
        } else {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi as u32
        };

        // Accept iff the target's draw matches a drafted child; else x is the bonus.
        match tree.child_maps[current_slot].get(&x) {
            Some(&child_idx) => {
                accepted.push(child_idx);
                current_slot = child_idx + 1;
            }
            None => return (accepted, x),
        }
    }
}

/// One position's VERBATIM recursive without-replacement speculative-sampling
/// step (SpecTr / Sequoia / SpecInfer §A). Given the target distribution `p_s`
/// (full vocab), the full draft distribution `q_pos` (full vocab), and the draft's
/// `cands` drawn from `q_pos` WITHOUT replacement IN DRAW ORDER, emits one token
/// distributed EXACTLY as `p_s` while accepting a candidate when possible.
///
/// For draw `j` (0-indexed), `Z = 1 − Σ_{i<j} q(c_i)` is the remaining draft mass
/// and the residual draft distribution is `q_j(t) = q_pos(t)/Z` for `t` not yet
/// drawn. Accept `c_j` w.p. `min(1, p_j(c_j)/q_j(c_j))`; on rejection set
/// `p_{j+1} = normalize(relu(p_j − q_j))` (full-vocab) and continue; if all `k`
/// reject, draw the token from the final residual `p_{k+1}`. Returns
/// `(Some(accepted_token), _)` or `(None, residual_token)`.
fn swor_step(
    p_s: &mut [f32],
    q_pos: &[f32],
    cands: &[u32],
    rng_state: &mut u64,
) -> (Option<u32>, u32) {
    let vocab = p_s.len();
    let mut drawn: Vec<u32> = Vec::with_capacity(cands.len());
    let mut drawn_q_sum = 0.0f32;
    for &c in cands {
        let ct = c as usize;
        if ct >= vocab {
            continue;
        }
        let z = (1.0 - drawn_q_sum).max(1e-9);
        let qjc = (q_pos[ct] / z).clamp(0.0, 1.0);
        let ratio = if qjc > 0.0 {
            (p_s[ct] / qjc).min(1.0)
        } else {
            0.0
        };
        let u = xorshift_unit(rng_state);
        if p_s[ct] > 0.0 && u < ratio {
            return (Some(c), c);
        }
        // Reject: p_{j+1} = normalize(relu(p_j − q_j)), q_j(t)=q_pos[t]/z for t∉drawn.
        for (t, pv) in p_s.iter_mut().enumerate() {
            if drawn.iter().any(|&d| d as usize == t) {
                continue;
            }
            *pv = (*pv - q_pos[t] / z).max(0.0);
        }
        let s: f32 = p_s.iter().sum();
        if s > 0.0 {
            for pv in p_s.iter_mut() {
                *pv /= s;
            }
        }
        drawn.push(c);
        drawn_q_sum += q_pos[ct];
    }
    // All candidates rejected: draw from the final residual target.
    let u = xorshift_unit(rng_state);
    (None, sample_unnormalized(p_s, u))
}

/// q-EXPLOITING tree verify — VERBATIM recursive without-replacement speculative
/// sampling (SpecTr / Sequoia / SpecInfer), distribution-EXACT.
///
/// Unlike naive sampling (which ignores `q`), at each tree node this runs the
/// full recursive SWOR step ([`swor_step`]) over the node's draft proposals —
/// the `k` tokens the draft sampled WITHOUT replacement from that position's
/// distribution, in draw order — accepting one when the rejection ratio clears.
/// Because every emitted token is drawn from the (residual) TARGET, the output
/// distribution is preserved EXACTLY at any temperature; `q` only changes WHICH
/// target draws get reused (acceptance), not WHAT is emitted. (Validated by the
/// `swor_preserves_target_distribution` Monte-Carlo test.)
///
/// Inputs: `target_logits` `[(1+N)·vocab]` (per verify slot); `draft_q`
/// `[num_pos·vocab]` the full draft softmax per draft position; `pos_cands`
/// `[num_pos·k]` the draft's draw-ordered samples per position. A node at tree
/// depth `d` draws from position `d` (root = position 0). On accept the walk
/// descends the matching tree child; an accepted-but-pruned candidate (not in the
/// tree) is emitted as the final token. Reduces to `follow_verified_tree` at
/// `temp→0`. Returns the same `(accepted_node_indices, bonus_token)` shape.
#[allow(clippy::too_many_arguments)]
pub fn sample_verified_tree_swor(
    tree: &DdTree,
    target_logits: &[f32],
    draft_q: &[f32],
    pos_cands: &[u32],
    num_pos: usize,
    k: usize,
    vocab: usize,
    temp: f32,
    rng_state: &mut u64,
) -> (Vec<usize>, u32) {
    let n_slots = 1 + tree.nodes.len();
    debug_assert_eq!(target_logits.len(), n_slots * vocab);

    // Greedy (temp≤0): delegate to the argmax walk (exact follow_verified_tree).
    if temp <= 0.0 {
        let argmax: Vec<u32> = (0..n_slots)
            .map(|s| {
                let row = &target_logits[s * vocab..(s + 1) * vocab];
                let mut bi = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, &v) in row.iter().enumerate() {
                    if v > bv {
                        bv = v;
                        bi = i;
                    }
                }
                bi as u32
            })
            .collect();
        return follow_verified_tree(tree, &argmax);
    }

    let mut accepted: Vec<usize> = Vec::new();
    let mut current_slot: usize = 0;
    let mut p: Vec<f32> = Vec::with_capacity(vocab);

    loop {
        // Draft position generating this node's children = the node's tree depth
        // (root depth 0). Past the last draft position there are no children.
        let depth = if current_slot == 0 {
            0
        } else {
            tree.nodes[current_slot - 1].depth as usize
        };
        let row = &target_logits[current_slot * vocab..(current_slot + 1) * vocab];
        softmax_temp_into(row, temp, &mut p);
        if depth >= num_pos {
            // Leaf: no draft proposals → emit a target draw.
            let u = xorshift_unit(rng_state);
            return (accepted, sample_unnormalized(&p, u));
        }

        let q_pos = &draft_q[depth * vocab..(depth + 1) * vocab];
        let cands = &pos_cands[depth * k..depth * k + k];
        let (acc_tok, emit) = swor_step(&mut p, q_pos, cands, rng_state);
        match acc_tok {
            Some(c) => match tree.child_maps[current_slot].get(&c) {
                Some(&child_idx) => {
                    accepted.push(child_idx);
                    current_slot = child_idx + 1;
                }
                // Accepted a candidate the budget pruned from the tree — emit it.
                None => return (accepted, c),
            },
            None => return (accepted, emit),
        }
    }
}

/// Return the **greedy main path** through the tree as a list of node
/// indices: the chain that starts at the root and at each step descends
/// to its highest-cumulative-log-prob child.
///
/// The implementation exploits an invariant of `build_ddtree_tree_with_cutoff`:
/// nodes are pushed into `tree.nodes` in heap-pop order (strictly
/// descending cumulative logw, push_order tie-breaking). For any given
/// parent slot, the child with the *lowest* index in `nodes` was popped
/// earliest and therefore has the highest cumulative logw among that
/// parent's children. Walking from root and always picking the
/// smallest-indexed child yields the greedy main path. See
/// `deeper_tree_maintains_heap_order` for a worked example.
///
/// Returns the chain of node indices in root-to-leaf order. The
/// linearization slot of `chain[i]` (matching `linearize_tree`'s output)
/// is `chain[i] + 1`. An empty tree returns an empty chain.
///
/// Used by Path C (main-path-first lazy verify): the caller forwards the
/// main chain as a flat linear verify (committed RoPE phases, no
/// linearization-slot phase poisoning, no GDN drift), then if a position
/// is rejected, lazily re-verifies any sibling branch at that depth
/// from a tape-restored DeltaNet snapshot. See
/// `docs/plans/ddtree-path-c-main-path-first-from-lucebox.prd`.
///
/// Cost: O(depth × N) linear scan. Tree sizes in production are small
/// (paper budget ≈ 22, hipfire default ≈ 16) so this is negligible vs
/// the verify forward.
pub fn select_main_path(tree: &DdTree) -> Vec<usize> {
    let mut chain: Vec<usize> = Vec::new();
    // -1 is the sentinel `DdNode::parent_index` value for direct children
    // of the root (matches `build_ddtree_tree_with_cutoff`'s seeding).
    let mut current_parent: i32 = -1;
    loop {
        let next = tree
            .nodes
            .iter()
            .enumerate()
            .find(|(_, n)| n.parent_index == current_parent)
            .map(|(i, _)| i);
        match next {
            Some(idx) => {
                chain.push(idx);
                current_parent = idx as i32;
            }
            None => break,
        }
    }
    chain
}

/// A branch off the main path: the smallest-indexed-child chain that
/// descends from a non-main sibling of one of the main-path nodes (or a
/// non-main root child). Produced by [`enumerate_branches`] for Path C
/// Phase 2's lazy FA-only re-verify.
#[derive(Debug, Clone)]
pub struct DdBranch {
    /// Tree depth of the parent (= the fork point). `0` means the branch
    /// forks off the root (seed); `k > 0` means the parent is
    /// `main_path[k-1]`, which sits at tree depth `k`.
    ///
    /// Matches the PRD's "branches off at depth `d`" convention: the
    /// branch spans depths `[fork_depth + 1, ..., fork_depth + chain.len()]`.
    /// Chain element `chain[i]` would land at absolute position
    /// `start_pos + fork_depth + i` if accepted (where `start_pos` is the
    /// position `main_path[0]` would commit to).
    pub fork_depth: u32,
    /// Greedy chain of tree node indices in root-to-leaf order. `chain[0]`
    /// is the forking sibling (a non-main child of the parent at depth
    /// `fork_depth`); subsequent entries descend by smallest-indexed-child,
    /// matching [`select_main_path`]'s greedy rule. Always non-empty.
    pub chain: Vec<usize>,
}

/// Enumerate the branches of `tree` that need lazy FA-only re-verify
/// against an already-committed `main_path` of which the first
/// `accepted_main` nodes were accepted by the target.
///
/// A branch is eligible iff its fork depth `d` satisfies `d ≤ accepted_main`
/// (PRD §"Architecture / Step 2"): the branch's parent must itself be on
/// the accepted main-path prefix (or be root), so the caller has a valid
/// DeltaNet snapshot to restore from before the branch's FA forward.
///
/// The output is in fork-depth-then-heap order (shallowest forks first;
/// within a fork depth, in `tree.nodes` order). For each branch, the
/// `chain` follows the smallest-indexed-child rule at every step — same
/// convention as [`select_main_path`] — so chains are deterministic
/// and consistent with the rest of the linearization.
///
/// `accepted_main` is bounded above by `main_path.len()`; passing a
/// larger value is treated as full acceptance.
pub fn enumerate_branches(
    tree: &DdTree,
    main_path: &[usize],
    accepted_main: usize,
) -> Vec<DdBranch> {
    let mut branches: Vec<DdBranch> = Vec::new();
    let max_d = accepted_main.min(main_path.len());

    for d in 0..=max_d {
        // Parent tree index in DdNode::parent_index convention: `-1` for
        // root (d == 0), `main_path[d-1]` otherwise (a node at depth d).
        let parent_tree_idx: i32 = if d == 0 { -1 } else { main_path[d - 1] as i32 };
        // The main-path child at depth d+1 (if any) is the one to skip;
        // every other child of `parent_tree_idx` is a branch root.
        let main_child: Option<usize> = main_path.get(d).copied();

        for (node_idx, node) in tree.nodes.iter().enumerate() {
            if node.parent_index != parent_tree_idx {
                continue;
            }
            if Some(node_idx) == main_child {
                continue;
            }
            // Greedy descent: at each step take the smallest-indexed child,
            // which by `build_ddtree_tree_with_cutoff`'s heap-pop invariant
            // is the highest-cumulative-logw child of the current node.
            let mut chain = vec![node_idx];
            let mut current_parent: i32 = node_idx as i32;
            loop {
                let next = tree
                    .nodes
                    .iter()
                    .enumerate()
                    .find(|(_, n)| n.parent_index == current_parent)
                    .map(|(i, _)| i);
                match next {
                    Some(idx) => {
                        chain.push(idx);
                        current_parent = idx as i32;
                    }
                    None => break,
                }
            }
            branches.push(DdBranch {
                fork_depth: d as u32,
                chain,
            });
        }
    }

    branches
}

/// Linearize a DDTree into a verify-ready `(tokens, positions, mask_block)`
/// triple suitable for a single batched target forward.
///
/// - `seed_token`: the anchor at tree root (= `block_output_ids[0]` in the
///   DFlash / DDTree papers). Occupies slot 0 in the returned arrays.
/// - `base_pos`: the absolute decode position of the seed (matches the
///   `position` argument of `spec_step_dflash`). Tree nodes live at
///   `base_pos + depth` where `depth` is 1-indexed from `DdNode::depth`.
///
/// Returns:
/// - `tokens: Vec<u32>` of length `1 + tree.num_nodes()`. `tokens[0] =
///   seed_token`; `tokens[i+1] = tree.nodes[i].token`.
/// - `positions: Vec<i32>` matching `tokens`, carrying each slot's logical
///   RoPE position (seed at `base_pos`, node i at `base_pos + depth_i`).
///   Two nodes at the same tree depth get the same logical position — they
///   represent alternative futures at the same time step, not successive
///   tokens in a chain.
/// - `mask_block: Vec<f32>` of shape `[(1+N) × (1+N)]` row-major. Value is
///   `0.0` when `visibility[i][j]` (j is an ancestor-or-self of i), else
///   `f32::NEG_INFINITY`. Attention kernels add this as a bias to qk scores
///   for keys in the tree block; the prompt region (positions
///   `[0, base_pos - 1]`) is always visible and needs no mask.
///
/// Ordering: matches `tree.nodes` exactly, which is heap-pop order. That
/// order is guaranteed topological (every parent appears before its children)
/// but NOT strictly BFS — siblings of different root-children may interleave
/// with their descendants. The kernel doesn't care; what matters is
/// (a) topological order so children see their parents' K/V, and
/// (b) the mask encodes the actual tree, not the linearization index.
pub fn linearize_tree(
    tree: &DdTree,
    seed_token: u32,
    base_pos: u32,
) -> (Vec<u32>, Vec<i32>, Vec<f32>) {
    let (tokens, positions, mask_block, _) =
        linearize_tree_with_parents(tree, seed_token, base_pos);
    (tokens, positions, mask_block)
}

/// Same as `linearize_tree` but also returns `parent_indices`: for each
/// slot in the linearized block, the slot index of its parent in the same
/// linearization, or `-1` for the root slot (slot 0, the seed token).
///
/// Tree-aware kernels (`conv1d_silu_split_tree`, `gated_delta_net_q8_tree`)
/// consume this array to walk per-token ancestor chains instead of the
/// linear-sequence predecessor. For the GDN kernel: `parent_indices[t] < 0`
/// means "read from the pre-block initial state" (s_q8_init); otherwise
/// "read from s_tape[parent]".
///
/// For the conv1d kernel: negative sentinels index the pre-block conv_state
/// ring (-1 → state[0], -2 → state[1], -3 → state[2]); since walking past
/// the block root consumes one sentinel slot per ancestor-chain step, the
/// kernel handles the `-1 → -2 → -3` chain internally. Callers only need
/// `-1` at the slot 0 position.
///
/// Returns `(tokens, positions, mask_block, parent_indices)` all of length
/// `1 + tree.num_nodes()`. `mask_block` is the same `[N×N]` row-major f32
/// additive bias; `parent_indices` is `[N]` i32.
pub fn linearize_tree_with_parents(
    tree: &DdTree,
    seed_token: u32,
    base_pos: u32,
) -> (Vec<u32>, Vec<i32>, Vec<f32>, Vec<i32>) {
    let len = 1 + tree.num_nodes();

    let mut tokens: Vec<u32> = Vec::with_capacity(len);
    tokens.push(seed_token);
    tokens.extend(tree.nodes.iter().map(|n| n.token));

    let mut positions: Vec<i32> = Vec::with_capacity(len);
    positions.push(base_pos as i32);
    positions.extend(tree.nodes.iter().map(|n| base_pos as i32 + n.depth as i32));

    // Row-major flatten of the visibility bool matrix → f32 additive bias.
    // -inf on masked keeps the running-max / logsumexp math exact (exp(-inf)
    // = 0 contributes nothing) and needs no epsilon tuning across dtypes.
    let mut mask_block: Vec<f32> = Vec::with_capacity(len * len);
    for row in &tree.visibility {
        debug_assert_eq!(row.len(), len);
        for &v in row {
            mask_block.push(if v { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    debug_assert_eq!(mask_block.len(), len * len);

    // Parent indices. Slot 0 = seed token = root = -1 sentinel (reads from
    // pre-block state in the GDN tree kernel). Slot i+1 corresponds to
    // tree.nodes[i]; its linearized parent is:
    //   - 0 if nodes[i].parent_index == -1 (direct child of root / seed)
    //   - nodes[i].parent_index + 1 otherwise
    let mut parent_indices: Vec<i32> = Vec::with_capacity(len);
    parent_indices.push(-1);
    for node in &tree.nodes {
        let p = if node.parent_index < 0 {
            0
        } else {
            node.parent_index + 1
        };
        parent_indices.push(p);
    }
    debug_assert_eq!(parent_indices.len(), len);

    (tokens, positions, mask_block, parent_indices)
}

/// CPU top-K per row on a log-softmax-normalized logits matrix. Produces the
/// `(top_tokens, top_log_probs)` arrays expected by `build_ddtree_tree`.
///
/// Inputs:
///   - `logits`: row-major `[rows × vocab]` raw logits (not yet softmaxed)
///   - `rows`: number of draft positions
///   - `vocab`: per-row width
///   - `k`: top-K
///
/// Outputs: `(top_tokens [rows*k], top_log_probs [rows*k])`. Log-probs are
/// computed once per row via log-sum-exp (numerically stable) and then
/// subtracted from each top-K logit.
pub fn topk_from_logits(
    logits: &[f32],
    rows: usize,
    vocab: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(
        logits.len(),
        rows * vocab,
        "topk_from_logits: logits size mismatch"
    );
    assert!(k <= vocab, "topk_from_logits: k > vocab");
    let mut top_tokens = Vec::with_capacity(rows * k);
    let mut top_log_probs = Vec::with_capacity(rows * k);

    // Fast path: k==1 is just argmax + the max value (no heap needed).
    // Common case for spine-only trees — O(vocab) instead of O(vocab log k).
    if k == 1 {
        for r in 0..rows {
            let row = &logits[r * vocab..(r + 1) * vocab];
            let mut best_val = f32::NEG_INFINITY;
            let mut best_idx: u32 = 0;
            let mut max = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i as u32;
                }
                if v > max {
                    max = v;
                }
            }
            let mut sum_exp = 0.0f64;
            for &v in row {
                sum_exp += ((v - max) as f64).exp();
            }
            let log_z = max + sum_exp.ln() as f32;
            top_tokens.push(best_idx);
            top_log_probs.push(best_val - log_z);
        }
        return (top_tokens, top_log_probs);
    }

    // General path: min-heap of size k, O(vocab × log k) per row instead of
    // O(vocab × log vocab) for a full sort. At k=8 over 248K vocab that's
    // ~10× faster than the prior sort-based impl — which on 9B MQ4 was
    // eating 150 ms per cycle (measured via DDTREE_TIMING).
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    // (value, index). BinaryHeap is max-heap; to keep the top-k we want a
    // MIN-heap of k items — wrap in Reverse. Custom Eq/Ord because f32
    // isn't total-ordered; NaN is treated as NEG_INFINITY-equivalent which
    // is safe for softmax-normalized logits (NaN never arises in practice).
    #[derive(Copy, Clone, PartialEq)]
    struct Item(f32, u32);
    impl Eq for Item {}
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(Ordering::Equal)
                .then(self.1.cmp(&other.1))
        }
    }
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut heap: BinaryHeap<std::cmp::Reverse<Item>> = BinaryHeap::with_capacity(k + 1);
    let mut top_k_items: Vec<Item> = Vec::with_capacity(k);
    for r in 0..rows {
        let row = &logits[r * vocab..(r + 1) * vocab];
        // Pass 1: top-k via min-heap + running max for log-sum-exp.
        heap.clear();
        let mut max = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > max {
                max = v;
            }
            let item = std::cmp::Reverse(Item(v, i as u32));
            if heap.len() < k {
                heap.push(item);
            } else if heap.peek().unwrap() > &item {
                // heap's min is GREATER than new item — wait, with Reverse
                // the BinaryHeap max is actually the Reverse-smallest = the
                // f32-SMALLEST. So we want to evict when new f32 > smallest.
                // `peek() > item` via Reverse means "peek's inner < item's
                // inner" which means "stored smallest < new" → push.
                heap.pop();
                heap.push(item);
            }
        }
        // Pass 2: log-sum-exp (needs running max from pass 1).
        let mut sum_exp = 0.0f64;
        for &v in row {
            sum_exp += ((v - max) as f64).exp();
        }
        let log_z = max + sum_exp.ln() as f32;

        // Extract heap contents sorted by value descending.
        top_k_items.clear();
        while let Some(std::cmp::Reverse(item)) = heap.pop() {
            top_k_items.push(item);
        }
        // Heap popped in ascending f32 order (smallest first via Reverse);
        // reverse for descending output.
        top_k_items.reverse();
        for item in &top_k_items {
            top_tokens.push(item.1);
            top_log_probs.push(item.0 - log_z);
        }
    }
    (top_tokens, top_log_probs)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Argmax of one logits row (test helper).
    fn row_argmax(row: &[f32]) -> u32 {
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > bv {
                bv = v;
                bi = i;
            }
        }
        bi as u32
    }

    // A small depth-2, top-2 tree over an 8-token vocab.
    fn small_tree() -> DdTree {
        let depth = 2usize;
        let topk = 2usize;
        // depth0 tokens {1,2}, depth1 tokens {3,4}; arbitrary normalized log-probs.
        let top_tokens = [1u32, 2, 3, 4];
        let p = [0.6f32, 0.4, 0.7, 0.3];
        let top_log_probs: Vec<f32> = p.iter().map(|x| x.ln()).collect();
        build_ddtree_tree_bounded(
            &top_tokens,
            &top_log_probs,
            depth,
            topk,
            0,
            16,
            f32::NEG_INFINITY,
        )
    }

    #[test]
    fn sample_temp0_matches_follow_verified_tree() {
        let tree = small_tree();
        let vocab = 8usize;
        let n = 1 + tree.nodes.len();
        // Build logits whose per-slot argmax is the slot's highest-prob child
        // token where one exists (else token 0) — exercises real descent.
        let mut logits = vec![0.0f32; n * vocab];
        for s in 0..n {
            let pref = tree.child_maps[s]
                .values()
                .copied()
                .min()
                .map(|ci| tree.nodes[ci].token)
                .unwrap_or(0);
            logits[s * vocab + pref as usize] = 10.0;
        }
        let argmax: Vec<u32> = (0..n)
            .map(|s| row_argmax(&logits[s * vocab..(s + 1) * vocab]))
            .collect();
        let (acc_g, bonus_g) = follow_verified_tree(&tree, &argmax);
        let mut rng = 0x1234_5678_9abc_def1u64;
        let (acc_s, bonus_s) = sample_verified_tree(&tree, &logits, vocab, 0.0, &mut rng);
        assert_eq!(acc_g, acc_s, "accepted path diverged at temp=0");
        assert_eq!(bonus_g, bonus_s, "bonus diverged at temp=0");
    }

    #[test]
    fn sample_temp_pos_preserves_target_distribution() {
        // Depth-1, top-2 tree: root children = tokens {0,1}; target mode (token 2)
        // is NOT in the tree, so it can only surface via the bonus. Naive-sampling
        // tree verify draws from the target, so the FIRST emitted token's marginal
        // must equal the target p EXACTLY (within MC noise) — incl. the uncovered
        // mode at token 2.
        let vocab = 3usize;
        let top_tokens = [0u32, 1];
        let q = [0.5f32, 0.3];
        let top_log_probs: Vec<f32> = q.iter().map(|x| x.ln()).collect();
        let tree =
            build_ddtree_tree_bounded(&top_tokens, &top_log_probs, 1, 2, 0, 2, f32::NEG_INFINITY);
        assert_eq!(tree.nodes.len(), 2);
        let n = 1 + tree.nodes.len();

        // Target p at the root (slot 0). Other slots (leaves) only feed bonuses
        // after acceptance; uniform is fine.
        let p = [0.2f32, 0.3, 0.5];
        let mut logits = vec![0.0f32; n * vocab];
        for (t, &pt) in p.iter().enumerate() {
            logits[t] = pt.ln(); // slot 0 = ln(p) → softmax(temp=1) = p
        }

        let n_runs = 400_000u32;
        let mut hist = [0u64; 3];
        let mut rng = 0xdead_beef_cafe_0001u64;
        for _ in 0..n_runs {
            let (accepted, bonus) = sample_verified_tree(&tree, &logits, vocab, 1.0, &mut rng);
            let first = if let Some(&ni) = accepted.first() {
                tree.nodes[ni].token
            } else {
                bonus
            };
            hist[first as usize] += 1;
        }
        let mut tv = 0.0f64;
        for t in 0..3 {
            let emp = hist[t] as f64 / n_runs as f64;
            tv += (emp - p[t] as f64).abs();
        }
        tv *= 0.5;
        assert!(
            tv < 0.01,
            "naive-sampling tree verify should preserve the target distribution; TV={tv:.4} hist={hist:?}"
        );
    }

    // softmax(logits / temp) — the per-slot target the walk must reproduce.
    fn softmax_temp(logits: &[f32], temp: f32) -> Vec<f32> {
        let inv = 1.0 / temp;
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut e: Vec<f32> = logits.iter().map(|&v| ((v - m) * inv).exp()).collect();
        let s: f32 = e.iter().sum();
        for x in e.iter_mut() {
            *x /= s;
        }
        e
    }

    #[test]
    fn swor_temp0_matches_follow_verified_tree() {
        // The q-exploiting SWOR walk must reduce to the greedy argmax walk at
        // temp→0 (same safety guarantee as naive sampling).
        let tree = small_tree();
        let vocab = 8usize;
        let n = 1 + tree.nodes.len();
        let mut logits = vec![0.0f32; n * vocab];
        for s in 0..n {
            let pref = tree.child_maps[s]
                .values()
                .copied()
                .min()
                .map(|ci| tree.nodes[ci].token)
                .unwrap_or(0);
            logits[s * vocab + pref as usize] = 10.0;
        }
        let argmax: Vec<u32> = (0..n)
            .map(|s| row_argmax(&logits[s * vocab..(s + 1) * vocab]))
            .collect();
        let (acc_g, bonus_g) = follow_verified_tree(&tree, &argmax);
        let mut rng = 0x2222_3333_4444_5555u64;
        // temp≤0 delegates to greedy before touching draft_q/pos_cands → empty ok.
        let (acc_s, bonus_s) =
            sample_verified_tree_swor(&tree, &logits, &[], &[], 0, 0, vocab, 0.0, &mut rng);
        assert_eq!(acc_g, acc_s, "SWOR accepted path diverged at temp=0");
        assert_eq!(bonus_g, bonus_s, "SWOR bonus diverged at temp=0");
    }

    // Sample k tokens WITHOUT replacement from `q` (sequential), returning draw
    // order — the proposal sequence the verbatim SWOR step expects.
    fn swor_sample(q: &[f32], k: usize, rng: &mut u64) -> Vec<u32> {
        let mut qr = q.to_vec();
        let mut out = Vec::with_capacity(k);
        for _ in 0..k {
            let s: f32 = qr.iter().sum();
            if s <= 0.0 {
                break;
            }
            let u = xorshift_unit(rng) * s;
            let mut acc = 0.0f32;
            let mut pick = qr.len() - 1;
            for (i, &v) in qr.iter().enumerate() {
                acc += v;
                if u < acc {
                    pick = i;
                    break;
                }
            }
            out.push(pick as u32);
            qr[pick] = 0.0; // without replacement
        }
        out
    }

    #[test]
    fn swor_preserves_target_distribution() {
        // VERBATIM distribution-fidelity proof: draw the candidates WITHOUT
        // replacement from the draft q (as the real Gumbel-top-k path does), run
        // the exact SWOR step, and confirm the emitted token's marginal equals the
        // TARGET p — at several temperatures and for k < vocab (so some target mass
        // is never proposed and must surface via the residual). This is the check
        // that was missing from the approximate implementation.
        let vocab = 6usize;
        // A target and a DISTINCT draft (so q is genuinely exploited / corrected).
        let p = [0.30f32, 0.05, 0.20, 0.10, 0.25, 0.10];
        let q = [0.10f32, 0.40, 0.05, 0.20, 0.05, 0.20];
        for &k in &[2usize, 3, 4] {
            let n_runs = 600_000u32;
            let mut hist = vec![0u64; vocab];
            let mut rng = 0x5eed_1234_abcd_0007u64 ^ (k as u64);
            for _ in 0..n_runs {
                let cands = swor_sample(&q, k, &mut rng);
                let mut p_work = p.to_vec();
                let (_acc, emit) = swor_step(&mut p_work, &q, &cands, &mut rng);
                hist[emit as usize] += 1;
            }
            let tv: f64 = (0..vocab)
                .map(|t| (hist[t] as f64 / n_runs as f64 - p[t] as f64).abs())
                .sum::<f64>()
                * 0.5;
            assert!(
                tv < 0.01,
                "verbatim SWOR must preserve the target distribution; k={k} TV={tv:.4} hist={hist:?}"
            );
        }
    }

    // Exact CPU mirror of the device Gumbel-top-k sampler's per-element RNG
    // (`kernels/src/ddtree_gumbel_topk_batched.hip`): murmur3 fmix32 of (seed,b,i).
    // Keep in sync with the kernel — this is the regression net for that RNG.
    fn murmur3_unit(seed: u32, b: u32, i: u32) -> f32 {
        let mut h = i.wrapping_mul(0x9E3779B1) ^ b.wrapping_mul(0x85EBCA77) ^ seed;
        h ^= h >> 16;
        h = h.wrapping_mul(0x7FEB352D);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846CA68B);
        h ^= h >> 16;
        (h >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    // Mirror of the device Gumbel-top-k SWOR sampler: draw k tokens WITHOUT
    // replacement from softmax(logits/temp) by top-k of `logit/temp + Gumbel`.
    fn gumbel_topk_murmur3(logits: &[f32], k: usize, temp: f32, seed: u32, b: u32) -> Vec<u32> {
        let inv_t = 1.0 / temp.max(1e-4);
        let mut scored: Vec<(f32, u32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let mut u = murmur3_unit(seed, b, i as u32);
                if u <= 1e-7 {
                    u = 1e-7;
                }
                if u >= 1.0 {
                    u = 0.9999999;
                }
                let g = -(-(u.ln())).ln();
                (v * inv_t + g, i as u32)
            })
            .collect();
        scored.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap());
        scored.iter().take(k).map(|&(_, t)| t).collect()
    }

    #[test]
    fn gumbel_swor_composition_preserves_target_distribution() {
        // THE end-to-end gate the prior tests missed: candidates come from the
        // ACTUAL murmur3 Gumbel sampler (not an exact host SWOR draw), fed into the
        // exact swor_step. The emitted marginal must STILL equal the target — this
        // is the only test that would catch a weak/biased draft sampler (which
        // keeps rank-0 correct but biases the without-replacement joint, the
        // recurring over-acceptance trap). Several temps × k, distinct p≠q.
        let vocab = 6usize;
        let plog = [0.4f32, -1.2, 1.1, 0.0, 0.7, -0.3]; // target logits (≠ draft)
        let qlog = [-0.5f32, 1.3, -0.2, 0.6, -1.0, 0.2]; // draft logits
        for &temp in &[0.5f32, 0.7, 1.0, 1.5] {
            let inv_t = 1.0 / temp;
            // target p at this temp.
            let mut p = vec![0f32; vocab];
            {
                let m = plog.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0;
                for (t, &v) in plog.iter().enumerate() {
                    p[t] = ((v - m) * inv_t).exp();
                    s += p[t];
                }
                for x in p.iter_mut() {
                    *x /= s;
                }
            }
            // full draft q at this temp (what swor_step's residual uses).
            let mut q = vec![0f32; vocab];
            {
                let m = qlog.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0;
                for (t, &v) in qlog.iter().enumerate() {
                    q[t] = ((v - m) * inv_t).exp();
                    s += q[t];
                }
                for x in q.iter_mut() {
                    *x /= s;
                }
            }
            for &k in &[2usize, 3, 4] {
                let n_runs = 400_000u32;
                let mut hist = vec![0u64; vocab];
                let mut vrng = 0x1357_9bdf_2468_ace0u64 ^ ((k as u64) << 8);
                for run in 0..n_runs {
                    let cands = gumbel_topk_murmur3(&qlog, k, temp, run.wrapping_add(1), 0);
                    let mut p_work = p.clone();
                    let (_acc, emit) = swor_step(&mut p_work, &q, &cands, &mut vrng);
                    hist[emit as usize] += 1;
                }
                let tv: f64 = (0..vocab)
                    .map(|t| (hist[t] as f64 / n_runs as f64 - p[t] as f64).abs())
                    .sum::<f64>()
                    * 0.5;
                assert!(
                    tv < 0.01,
                    "Gumbel→SWOR composition must preserve the target; temp={temp} k={k} TV={tv:.4} hist={hist:?}"
                );
            }
        }
    }

    #[test]
    fn sample_first_token_marginal_exact_across_temperatures() {
        // The first emitted token is always a draw from the ROOT target (whether
        // it lands on a drafted child or becomes the bonus), so its marginal must
        // equal softmax(root_logits / temp) EXACTLY at every temperature, for any
        // tree shape. This is the core "distribution-exact at any temperature"
        // invariant.
        let vocab = 5usize;
        // Tree covers tokens {1,3} at the root; tokens {0,2,4} reach only via bonus.
        let top_tokens = [1u32, 3];
        let q = [0.55f32, 0.30];
        let lp: Vec<f32> = q.iter().map(|x| x.ln()).collect();
        let tree = build_ddtree_tree_bounded(&top_tokens, &lp, 1, 2, 0, 2, f32::NEG_INFINITY);
        let n = 1 + tree.nodes.len();

        // Arbitrary (non-uniform) root logits; leaf rows unused for first token.
        let root_logits = [0.4f32, -1.2, 2.1, 0.0, -0.5];
        let mut logits = vec![0.0f32; n * vocab];
        logits[..vocab].copy_from_slice(&root_logits);

        let mut rng = 0x0123_4567_89ab_cdefu64;
        for &temp in &[0.3f32, 0.7, 1.0, 1.5] {
            let want = softmax_temp(&root_logits, temp);
            let n_runs = 600_000u32;
            let mut hist = vec![0u64; vocab];
            for _ in 0..n_runs {
                let (accepted, bonus) = sample_verified_tree(&tree, &logits, vocab, temp, &mut rng);
                let first = accepted
                    .first()
                    .map(|&ni| tree.nodes[ni].token)
                    .unwrap_or(bonus);
                hist[first as usize] += 1;
            }
            let tv: f64 = (0..vocab)
                .map(|t| (hist[t] as f64 / n_runs as f64 - want[t] as f64).abs())
                .sum::<f64>()
                * 0.5;
            assert!(
                tv < 0.01,
                "first-token marginal must equal softmax(root/temp); temp={temp} TV={tv:.4} hist={hist:?}"
            );
        }
    }

    #[test]
    fn sample_second_token_conditional_exact() {
        // After the walk descends an accepted child, the NEXT emitted token is a
        // draw from that child slot's target. Depth-1 tree → each child is a leaf,
        // so conditioned on accepting child token 0 (slot 1), the second token is
        // the bonus drawn from softmax(slot1_logits / temp). This proves the
        // descended-slot conditional draw is exact (i.e. the full sequence is
        // autoregressive sampling from the target).
        let vocab = 4usize;
        let top_tokens = [0u32, 1];
        let q = [0.6f32, 0.3];
        let lp: Vec<f32> = q.iter().map(|x| x.ln()).collect();
        let tree = build_ddtree_tree_bounded(&top_tokens, &lp, 1, 2, 0, 2, f32::NEG_INFINITY);
        // child token 0 is node 0 → slot 1.
        let slot_of_child0 = tree.child_maps[0][&0] + 1;
        let n = 1 + tree.nodes.len();

        let root_logits = [1.5f32, 0.2, -0.3, -2.0]; // token 0 likely → frequent descent
        let slot1_logits = [-0.5f32, 0.8, 1.2, 0.1]; // the conditional target after token 0
        let mut logits = vec![0.0f32; n * vocab];
        logits[..vocab].copy_from_slice(&root_logits);
        logits[slot_of_child0 * vocab..(slot_of_child0 + 1) * vocab].copy_from_slice(&slot1_logits);

        let temp = 0.7f32;
        let want = softmax_temp(&slot1_logits, temp);
        let mut rng = 0xfeed_face_0000_1111u64;
        let mut hist = vec![0u64; vocab];
        let mut conditioned = 0u64;
        for _ in 0..2_000_000u32 {
            let (accepted, bonus) = sample_verified_tree(&tree, &logits, vocab, temp, &mut rng);
            // Condition on x0 == token 0 (accepted child 0); then x1 = bonus.
            if accepted.first().map(|&ni| tree.nodes[ni].token) == Some(0) {
                conditioned += 1;
                hist[bonus as usize] += 1;
            }
        }
        assert!(
            conditioned > 100_000,
            "too few descents to estimate ({conditioned})"
        );
        let tv: f64 = (0..vocab)
            .map(|t| (hist[t] as f64 / conditioned as f64 - want[t] as f64).abs())
            .sum::<f64>()
            * 0.5;
        assert!(
            tv < 0.01,
            "second-token | x0=0 must equal softmax(slot1/temp); TV={tv:.4} hist={hist:?} n={conditioned}"
        );
    }

    #[test]
    fn empty_tree_has_root_only_visibility() {
        let t = build_ddtree_tree(&[], &[], 0, 0, 0);
        assert_eq!(t.nodes.len(), 0);
        assert_eq!(t.visibility, vec![vec![true]]);
    }

    #[test]
    fn bounded_keeps_node_count_in_band() {
        let (depth, topk) = (4usize, 3usize);
        // Peaked and flat distributions — raw cutoff would swing the count, but
        // the band must clamp both to [4, 16] at every cutoff.
        for probs in [[0.90f32, 0.06, 0.04], [0.40, 0.32, 0.28]] {
            let mut toks = Vec::new();
            let mut lp = Vec::new();
            for d in 0..depth {
                for r in 0..topk {
                    toks.push((d * 10 + r) as u32);
                    lp.push(probs[r].ln());
                }
            }
            for cut in [f32::NEG_INFINITY, -4.0, -3.0, -2.0] {
                let t = build_ddtree_tree_bounded(&toks, &lp, depth, topk, 4, 16, cut);
                let n = t.num_nodes();
                assert!(
                    (4..=16).contains(&n),
                    "node count {n} out of [4,16] (cut={cut})"
                );
            }
        }
    }

    #[test]
    fn single_depth_single_budget_picks_top() {
        // Depth 1, top-3, best = token 7 with log-prob -0.1.
        let tokens = vec![7, 3, 9];
        let logps = vec![-0.1, -1.0, -2.0];
        let t = build_ddtree_tree(&tokens, &logps, 1, 3, 1);
        assert_eq!(t.nodes.len(), 1);
        assert_eq!(t.nodes[0].token, 7);
        assert_eq!(t.nodes[0].depth, 1);
        assert_eq!(t.nodes[0].parent_index, -1);
        // Visibility: root visible to self + node 1 visible to {0,1}.
        assert_eq!(t.visibility[1][0], true);
        assert_eq!(t.visibility[1][1], true);
        // child_maps[0] (root's children) should have {7 → 0}.
        assert_eq!(t.child_maps[0].get(&7), Some(&0));
    }

    #[test]
    fn deeper_tree_maintains_heap_order() {
        // Two depths, top-2 each. Best path: (d1 rank 0) → (d2 rank 0).
        // Second-best sibling at d1: (d1 rank 1) = alternative root child.
        let tokens = vec![
            10, 20, // depth 1: top-2
            30, 40, // depth 2: top-2
        ];
        let logps = vec![
            -0.1, -1.0, // depth 1 log-probs
            -0.2, -1.5, // depth 2 log-probs
        ];
        let t = build_ddtree_tree(&tokens, &logps, 2, 2, 4);
        // Popped in descending cumulative log-weight:
        //   (d1 r0) logw = -0.1
        //   → push (d1 r1) logw = -1.0, push (d2 r0) logw = -0.1 + -0.2 = -0.3
        //   → pop (d2 r0) logw = -0.3
        //     → push (d2 r1) logw = -0.1 + -1.5 = -1.6
        //   → pop (d1 r1) logw = -1.0
        //     → push (d2 r0 child of d1r1) logw = -1.0 + -0.2 = -1.2
        //   → pop (d2 r0 child of d1r1) logw = -1.2
        // So 4-node tree is [10, 30 under 10, 20 (sibling of 10), 30 under 20]
        assert_eq!(t.nodes.len(), 4);
        assert_eq!(t.nodes[0].token, 10);
        assert_eq!(t.nodes[1].token, 30);
        assert_eq!(t.nodes[1].parent_index, 0);
        assert_eq!(t.nodes[2].token, 20);
        assert_eq!(t.nodes[2].parent_index, -1);
        assert_eq!(t.nodes[3].token, 30);
        assert_eq!(t.nodes[3].parent_index, 2);
    }

    #[test]
    fn follow_accepts_matching_chain() {
        // Tree (4 nodes): node 0 = 10 (child of root), node 1 = 30 (child of
        // node 0), node 2 = 20 (child of root), node 3 = 30 (child of node 2).
        // posterior has 1 + 4 = 5 entries: posterior[0] = target at root slot
        // (predicts after seed), posterior[i+1] = target at node i slot.
        let tokens = vec![10, 20, 30, 40];
        let logps = vec![-0.1, -1.0, -0.2, -1.5];
        let t = build_ddtree_tree(&tokens, &logps, 2, 2, 4);
        // target: root → "10" (matches child 0) → "30" (matches child of node 0)
        // → 99 (bonus, no match at node 1's slot).
        let posterior = vec![10, 30, 99, 99, 99];
        let (accepted, bonus) = follow_verified_tree(&t, &posterior);
        assert_eq!(accepted, vec![0, 1]);
        assert_eq!(bonus, 99);
    }

    #[test]
    fn follow_returns_bonus_on_root_miss() {
        // 2-node tree (depth 1, top 2). posterior length = 1 + 2 = 3.
        let tokens = vec![10, 20];
        let logps = vec![-0.1, -1.0];
        let t = build_ddtree_tree(&tokens, &logps, 1, 2, 2);
        // posterior[0] = 55 (not a child of root) → no acceptance.
        let posterior = vec![55, 0, 0];
        let (accepted, bonus) = follow_verified_tree(&t, &posterior);
        assert_eq!(accepted.len(), 0);
        assert_eq!(bonus, 55);
    }

    #[test]
    fn linearize_empty_tree_is_seed_only() {
        let t = build_ddtree_tree(&[], &[], 0, 0, 0);
        let (toks, pos, mask) = linearize_tree(&t, /*seed=*/ 42, /*base_pos=*/ 100);
        assert_eq!(toks, vec![42]);
        assert_eq!(pos, vec![100]);
        // Single slot; root is visible to itself.
        assert_eq!(mask, vec![0.0]);
    }

    #[test]
    fn linearize_spine_has_causal_mask() {
        // Spine (topk=1): depth 3, top-1. Tree nodes = [t1, t2, t3] all on a
        // single chain. Linearized: [seed, t1, t2, t3]. Each node should see
        // root + every ancestor. Mask should be lower-triangular zeros.
        let tokens = vec![11, 22, 33];
        let logps = vec![-0.1, -0.2, -0.3];
        let t = build_ddtree_tree(&tokens, &logps, 3, 1, 3);
        assert_eq!(t.nodes.len(), 3);
        let (toks, pos, mask) = linearize_tree(&t, 5, 50);
        assert_eq!(toks, vec![5, 11, 22, 33]);
        // Depths: 1, 2, 3 → positions: 50, 51, 52, 53.
        assert_eq!(pos, vec![50, 51, 52, 53]);
        // 4×4 lower-triangular: visible = 0.0, invisible = -inf.
        let expected: Vec<f32> = vec![
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn linearize_bushy_tree_masks_siblings() {
        // Reuse the 4-node test tree: [10, 30 under 10, 20 (sibling of 10),
        // 30 under 20]. Seed = 1, base_pos = 0.
        let tokens = vec![10, 20, 30, 40];
        let logps = vec![-0.1, -1.0, -0.2, -1.5];
        let t = build_ddtree_tree(&tokens, &logps, 2, 2, 4);
        assert_eq!(t.nodes.len(), 4);
        let (toks, pos, mask) = linearize_tree(&t, 1, 0);
        assert_eq!(toks, vec![1, 10, 30, 20, 30]);
        // Depths from construction: 1 (node0=10), 2 (node1=30 under 10),
        // 1 (node2=20), 2 (node3=30 under 20). Base 0 → positions 0,1,2,1,2.
        assert_eq!(pos, vec![0, 1, 2, 1, 2]);
        // 5×5 mask. Slot 0 = root. Slot i+1 = node i.
        // Ancestors:
        //   slot 0 (root): {0}
        //   slot 1 (node 10, parent=root): {0, 1}
        //   slot 2 (node 30 under 10, parent=node 10): {0, 1, 2}
        //   slot 3 (node 20, parent=root): {0, 3}
        //   slot 4 (node 30 under 20, parent=node 20): {0, 3, 4}
        let ni = f32::NEG_INFINITY;
        let expected: Vec<f32> = vec![
            0.0, ni, ni, ni, ni, 0.0, 0.0, ni, ni, ni, 0.0, 0.0, 0.0, ni, ni, 0.0, ni, ni, 0.0, ni,
            0.0, ni, ni, 0.0, 0.0,
        ];
        assert_eq!(mask, expected);
    }

    #[test]
    fn linearize_with_parents_spine() {
        // Spine chain of 3 depths × topk=1: same as linearize_spine_tree.
        let tokens = vec![11, 22, 33];
        let logps = vec![-0.1, -0.2, -0.3];
        let t = build_ddtree_tree(&tokens, &logps, 3, 1, 3);
        let (_toks, _pos, _mask, parents) = linearize_tree_with_parents(&t, 5, 50);
        // Slot 0 = root/seed = -1 sentinel. Slot i (i>=1) = previous slot.
        assert_eq!(parents, vec![-1, 0, 1, 2]);
    }

    #[test]
    fn linearize_with_parents_bushy() {
        // 4-node bushy tree: [node0=10 root-child, node1=30 under 10,
        // node2=20 root-child, node3=30 under 20]. linearize slots:
        //   0 = seed(1)
        //   1 = nodes[0]=10 (parent=-1 → slot 0)
        //   2 = nodes[1]=30 (parent=0 → slot 1)
        //   3 = nodes[2]=20 (parent=-1 → slot 0)
        //   4 = nodes[3]=30 (parent=2 → slot 3)
        let tokens = vec![10, 20, 30, 40];
        let logps = vec![-0.1, -1.0, -0.2, -1.5];
        let t = build_ddtree_tree(&tokens, &logps, 2, 2, 4);
        let (_toks, _pos, _mask, parents) = linearize_tree_with_parents(&t, 1, 0);
        assert_eq!(parents, vec![-1, 0, 1, 0, 3]);
    }

    #[test]
    fn topk_log_probs_are_normalized() {
        // Two rows, vocab=4, k=2. Row 0: logits [2, 1, 0, -1]. Top-2 = [0, 1].
        // log-sum-exp = log(e^2 + e^1 + e^0 + e^-1) ≈ 2.44
        // log-prob(2) ≈ 2 - 2.44 ≈ -0.44, log-prob(1) ≈ 1 - 2.44 ≈ -1.44
        let logits = vec![2.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 2.0];
        let (toks, logps) = topk_from_logits(&logits, 2, 4, 2);
        assert_eq!(toks, vec![0, 1, 3, 2]);
        assert!((logps[0] - (-0.44)).abs() < 0.02);
        assert!((logps[1] - (-1.44)).abs() < 0.02);
    }

    #[test]
    fn select_main_path_empty_tree_is_empty() {
        let t = build_ddtree_tree(&[], &[], 0, 0, 0);
        assert_eq!(select_main_path(&t), Vec::<usize>::new());
    }

    #[test]
    fn select_main_path_single_node_returns_that_node() {
        // depth=1, top=1, budget=1 → one-node tree.
        let t = build_ddtree_tree(&[7], &[-0.1], 1, 1, 1);
        assert_eq!(select_main_path(&t), vec![0]);
    }

    #[test]
    fn select_main_path_picks_first_child_at_each_depth() {
        // Re-uses `deeper_tree_maintains_heap_order`'s tree:
        // nodes[0] = 10 (root child, best)
        // nodes[1] = 30 under nodes[0] (best chain extension)
        // nodes[2] = 20 (alternative root child)
        // nodes[3] = 30 under nodes[2] (best ext of alt branch)
        //
        // Greedy main path = follow root → 10 → 30 = [0, 1].
        let tokens = vec![10, 20, 30, 40];
        let logps = vec![-0.1, -1.0, -0.2, -1.5];
        let t = build_ddtree_tree(&tokens, &logps, 2, 2, 4);
        assert_eq!(select_main_path(&t), vec![0, 1]);
    }

    #[test]
    fn select_main_path_handles_chain_only_tree() {
        // Force a depth-3 chain by giving every alternative very low prob.
        // depth=3, topk=1, budget=3 → linear chain of 3 nodes.
        let tokens = vec![10, 20, 30];
        let logps = vec![-0.1, -0.1, -0.1];
        let t = build_ddtree_tree(&tokens, &logps, 3, 1, 3);
        assert_eq!(t.nodes.len(), 3);
        // Each node is its predecessor's only child → main path is the
        // full chain.
        assert_eq!(select_main_path(&t), vec![0, 1, 2]);
    }

    #[test]
    fn select_main_path_consistent_with_linearization() {
        // The main-path tokens, when read off via `tree.nodes[i].token`,
        // must equal the slot[i+1] tokens in `linearize_tree`'s output
        // for those slots. This guarantees the caller can swap a tree
        // verify for a linear verify on the main chain without resequencing.
        let tokens = vec![10, 20, 30, 40, 50, 60];
        let logps = vec![-0.1, -1.0, -0.2, -1.5, -0.3, -2.0];
        let t = build_ddtree_tree(&tokens, &logps, 3, 2, 6);
        let main = select_main_path(&t);
        let (lin_tokens, _pos, _mask) = linearize_tree(&t, /*seed=*/ 1, 0);
        for &idx in &main {
            assert_eq!(t.nodes[idx].token, lin_tokens[idx + 1]);
        }
    }

    #[test]
    fn enumerate_branches_empty_tree_returns_empty() {
        let t = build_ddtree_tree(&[], &[], 0, 0, 0);
        assert!(enumerate_branches(&t, &[], 0).is_empty());
    }

    #[test]
    fn enumerate_branches_zero_accepted_yields_only_root_siblings() {
        // Same 4-node tree as deeper_tree_maintains_heap_order:
        //   nodes[0] = 10 (main path step 1)
        //   nodes[1] = 30 under nodes[0] (main path step 2)
        //   nodes[2] = 20 (alt root child — branch sibling)
        //   nodes[3] = 30 under nodes[2] (extends the alt branch)
        let tokens = vec![10, 20, 30, 40];
        let logps = vec![-0.1, -1.0, -0.2, -1.5];
        let t = build_ddtree_tree(&tokens, &logps, 2, 2, 4);
        let main = select_main_path(&t);
        assert_eq!(main, vec![0, 1]);

        let bs = enumerate_branches(&t, &main, 0);
        // accepted_main=0 → only forks at depth 0 (root). Sibling = nodes[2]
        // with greedy descent extending into nodes[3].
        assert_eq!(bs.len(), 1);
        assert_eq!(bs[0].fork_depth, 0);
        assert_eq!(bs[0].chain, vec![2, 3]);
    }

    #[test]
    fn enumerate_branches_full_accept_includes_terminal_siblings() {
        // Build a tree where main_path[1] (last main node) has a sibling
        // (= a non-main child of main_path[0]). With full accept, that
        // terminal sibling should appear as a fork at depth 1.
        // Construction: depth=2, topk=2, budget=5 with logps tuned so
        // nodes[0]=A, nodes[1]=A's best child (main path 0,1), nodes[2]=
        // alt root child, nodes[3]=A's second child (= sibling of main 1),
        // nodes[4]=alt-root-child's child.
        let tokens = vec![10, 20, 30, 40];
        let logps = vec![
            -0.1, -1.0, // depth 1: top-2
            -0.2, -0.5, // depth 2: top-2 (close so alt sibling expands)
        ];
        let t = build_ddtree_tree(&tokens, &logps, 2, 2, 5);
        let main = select_main_path(&t);
        // Main path = [0, 1]: nodes[0]=10 → nodes[1]=30
        assert_eq!(main, vec![0, 1]);

        let bs = enumerate_branches(&t, &main, main.len());
        // Forks expected:
        //   - fork_depth=0: nodes[2] (alt root child = 20)
        //   - fork_depth=1: nodes[1]'s parent is nodes[0]; sibling of main_path[1]
        //     under nodes[0] = a 4th node if budget permits. With heap order:
        //     pop (d1 r0)=10 logw=-0.1 → push (d1 r1) logw=-1, (d2 r0 of 10) logw=-0.3
        //     pop (d2 r0 of 10)=30 logw=-0.3 → push (d2 r1 of 10)=40 logw=-0.6
        //     pop (d2 r1 of 10)=40 logw=-0.6 → no further push (d=2 max)
        //     pop (d1 r1)=20 logw=-1 → push (d2 r0 of 20)=30 logw=-1.2
        //     pop (d2 r0 of 20)=30 logw=-1.2
        //   So nodes order: [10@d1, 30@d2 child-of-10, 40@d2 child-of-10, 20@d1, 30@d2 child-of-20]
        // Branches: fork_depth=0 → chain starting at nodes[3]=20, descend to nodes[4]=30
        //           fork_depth=1 → chain = [nodes[2]=40] (no further descent)
        assert_eq!(bs.len(), 2);
        // Sort by fork_depth for deterministic check.
        let mut by_depth: Vec<_> = bs.iter().collect();
        by_depth.sort_by_key(|b| b.fork_depth);
        assert_eq!(by_depth[0].fork_depth, 0);
        assert_eq!(by_depth[0].chain, vec![3, 4]);
        assert_eq!(by_depth[1].fork_depth, 1);
        assert_eq!(by_depth[1].chain, vec![2]);
    }

    #[test]
    fn enumerate_branches_chain_only_tree_has_no_branches() {
        // Pure spine: every node is the only child of its predecessor.
        let tokens = vec![10, 20, 30];
        let logps = vec![-0.1, -0.1, -0.1];
        let t = build_ddtree_tree(&tokens, &logps, 3, 1, 3);
        let main = select_main_path(&t);
        assert_eq!(main, vec![0, 1, 2]);
        assert!(enumerate_branches(&t, &main, main.len()).is_empty());
    }

    #[test]
    fn enumerate_branches_partial_accept_caps_eligible_depth() {
        // Reuse full-accept tree; with accepted_main=1 only fork_depth ∈ [0, 1]
        // are eligible — same as full accept here because main path has
        // length 2 and its full accept also yields fork_depth ∈ [0, 1].
        // To distinguish, build a 3-deep main path tree and accept just 1.
        let tokens = vec![1, 2, 3, 4, 5, 6];
        let logps = vec![-0.1, -1.0, -0.1, -1.0, -0.1, -1.0];
        let t = build_ddtree_tree(&tokens, &logps, 3, 2, 8);
        let main = select_main_path(&t);
        assert!(main.len() >= 2, "test fixture needs main path ≥ 2");

        let bs_zero = enumerate_branches(&t, &main, 0);
        let bs_one = enumerate_branches(&t, &main, 1);
        // accepted_main=0 ⇒ only depth-0 forks; accepted_main=1 ⇒ depths 0 + 1.
        // → bs_one ⊇ bs_zero, and bs_one has at least one fork at depth 1
        //   (sibling of main_path[1] under main_path[0]) iff the tree has one.
        assert!(bs_one.len() >= bs_zero.len());
        assert!(bs_zero.iter().all(|b| b.fork_depth == 0));
        assert!(bs_one.iter().all(|b| b.fork_depth <= 1));
    }

    #[test]
    fn enumerate_branches_chains_descend_smallest_index_first() {
        // Property: each chain follows the smallest-indexed child rule
        // (i.e. greedy / select_main_path-consistent). For every branch,
        // verify chain[i+1] is the smallest-index node whose parent_index
        // equals chain[i].
        let tokens = vec![10, 20, 30, 40, 50, 60];
        let logps = vec![-0.1, -0.5, -0.2, -0.6, -0.3, -0.7];
        let t = build_ddtree_tree(&tokens, &logps, 3, 2, 8);
        let main = select_main_path(&t);
        let bs = enumerate_branches(&t, &main, main.len());
        for branch in &bs {
            assert!(!branch.chain.is_empty());
            for win in branch.chain.windows(2) {
                let parent = win[0] as i32;
                let child = win[1];
                let expected = t
                    .nodes
                    .iter()
                    .enumerate()
                    .find(|(_, n)| n.parent_index == parent)
                    .map(|(i, _)| i)
                    .expect("chain step must have at least one descendant");
                assert_eq!(child, expected);
            }
            // chain[0]'s parent must be at fork_depth (root if 0, else main_path[fork_depth-1]).
            let expected_parent: i32 = if branch.fork_depth == 0 {
                -1
            } else {
                main[branch.fork_depth as usize - 1] as i32
            };
            assert_eq!(t.nodes[branch.chain[0]].parent_index, expected_parent);
        }
    }

    #[test]
    fn select_main_path_strict_descent_in_depth() {
        // Main-path nodes must form a parent-chain: each successor's
        // parent_index must equal its predecessor's tree index, and
        // depth strictly increases by 1 along the chain.
        let tokens = vec![10, 20, 30, 40, 50, 60];
        let logps = vec![-0.1, -1.0, -0.2, -1.5, -0.3, -2.0];
        let t = build_ddtree_tree(&tokens, &logps, 3, 2, 6);
        let main = select_main_path(&t);
        if main.is_empty() {
            return;
        }
        // First node of main path is a direct root child.
        assert_eq!(t.nodes[main[0]].parent_index, -1);
        assert_eq!(t.nodes[main[0]].depth, 1);
        for w in main.windows(2) {
            let parent = w[0];
            let child = w[1];
            assert_eq!(t.nodes[child].parent_index, parent as i32);
            assert_eq!(t.nodes[child].depth, t.nodes[parent].depth + 1);
        }
    }
}
