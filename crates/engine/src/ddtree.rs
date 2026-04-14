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
    neg_logw: f32,     // negated so BinaryHeap (max-heap) pops MIN neg_logw = MAX logw first
    push_order: u64,   // FIFO tie-break — earlier pushes win on equal neg_logw
    depth: usize,      // 1-indexed; 1 = child of root
    rank: usize,       // position in the top-K at this depth
    parent_index: i32, // -1 = parent is root; else nodes[parent_index]
    logw: f32,         // cumulative log-weight of the path root→this-candidate
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
        });
        child_maps.push(HashMap::new());
        // Register this node as a child of its parent by its draft token.
        // `parent_slot` maps the root-indexed convention (parent = -1 → slot 0).
        let parent_slot = if parent_index < 0 { 0 } else { (parent_index as usize) + 1 };
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
            if p < 0 { 0 } else { (p as usize) + 1 }
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
    // Reused per-row index buffer; partial_sort would be ideal but std has
    // no generic partial sort for f32 so we use full sort on indices.
    // Cheap enough at rows × O(vocab log vocab), called once per cycle.
    let mut idx: Vec<usize> = Vec::with_capacity(vocab);
    for r in 0..rows {
        let row = &logits[r * vocab..(r + 1) * vocab];
        // log-sum-exp for normalization.
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f64;
        for &v in row {
            sum_exp += ((v - max) as f64).exp();
        }
        let log_z = max + sum_exp.ln() as f32;

        idx.clear();
        idx.extend(0..vocab);
        // Partial sort would be O(vocab) but std_sort_by_cached_key is plenty
        // fast and keeps the code simple.
        idx.sort_unstable_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for j in 0..k {
            let tok = idx[j] as u32;
            top_tokens.push(tok);
            top_log_probs.push(row[idx[j]] - log_z);
        }
    }
    (top_tokens, top_log_probs)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tree_has_root_only_visibility() {
        let t = build_ddtree_tree(&[], &[], 0, 0, 0);
        assert_eq!(t.nodes.len(), 0);
        assert_eq!(t.visibility, vec![vec![true]]);
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
}
