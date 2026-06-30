// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stage 3b: on-GPU DDTree build + linearize.
//
// Eliminates D4 (attn_bias H2D, ~15 KB), D5 (parent_indices H2D, 244 B),
// D6/D7 (top-K D2H, ~480 B), and D10–D12 (SWOR metadata H2D, ~1.5 KB)
// by constructing the full tree directly from device-resident draft logits.
//
// The kernel is a single-workgroup, single-thread sequential build.  The
// heap at budget=60, k=4, depth=15 runs at most 2×budget heap pushes, each
// O(log budget) ≈ O(6).  Total ~720 ops in LDS: < 1 µs on any RDNA arch.
// Single-thread is load-bearing for FIFO heap tie-break correctness.
//
// ── Byte-identity design ─────────────────────────────────────────────────
//
// This kernel MUST produce bit-identical trees to the Rust host build
// (`build_ddtree_tree_bounded` in ddtree.rs) at temp=0.  Three risks:
//
//   Risk 1 — top-K tie-break: equal logit values broken by token index
//     ASCENDING.  The host `topk_from_logits` uses a min-heap whose `Ord`
//     impl has secondary key `self.1.cmp(&other.1)` (token index, asc).
//     We replicate this exactly in a single linear scan: maintain a sorted
//     array of (val, idx) of size K; on equal val prefer smaller idx.
//
//   Risk 2 — log-sum-exp precision: host accumulates `sum_exp` in f64, then
//     casts to f32 only at the end (`max + sum_exp.ln() as f32`).  We do
//     the same: `double` accumulator, cast to float at write time.
//
//   Risk 3 — cumulative logw arithmetic: sibling/child pushes use sequential
//     f32 add/sub.  We use `volatile float` intermediates to prevent the
//     compiler from fusing them into FMAs, which would change the result.
//
//   Risk 4 — heap FIFO tie-break: equal neg_logw entries broken by
//     push_counter (earlier push wins, matching the host's `push_order`).
//     Safe in a single-thread kernel — push_counter advances deterministically.
//
// ── Algorithm ─────────────────────────────────────────────────────────────
//
// Phase 1: per-row top-K (single pass over logits row, max tracked for lse)
// Phase 2: best-first heap expansion — exact mirror of the Rust host
// Phase 3: write outputs (tokens, positions, parent_indices, slot_depth,
//           child_of_cand) and big_n scalar
// Phase 4: attn_bias mask (same as ddtree_build_attn_mask_f32 but inline)
//
// Grid: [1,1,1].  Block: [1,1,1].
//
// Limits:
//   MAX_BUDGET   ≤ 60   (tree nodes)
//   MAX_TOPK     ≤ 8
//   MAX_DEPTH    ≤ 63
//   MAX_HEAP     = 2 × MAX_BUDGET + 4  (worst-case heap size)
//   MAX_N        = 1 + MAX_BUDGET = 61 (slots including root)

#include <hip/hip_runtime.h>
#include <math.h>

#define BUILD_MAX_BUDGET 60
#define BUILD_MAX_TOPK   8
#define BUILD_MAX_DEPTH  63
#define BUILD_MAX_N      (1 + BUILD_MAX_BUDGET)         // 61
#define BUILD_MAX_HEAP   (2 * BUILD_MAX_BUDGET + 4)     // 124

// ── Heap entry ───────────────────────────────────────────────────────────
struct HeapEntry {
    float  neg_logw;     // BinaryHeap is max-heap; MIN neg_logw = MAX logw
    uint64_t push_order; // FIFO tie-break (smaller wins = earlier push)
    int    depth;        // 1-indexed
    int    rank;         // position in top-K at this depth
    int    parent_index; // -1 = parent is root; else nodes[parent_index]
    float  logw;         // cumulative log-weight
};

// Min-heap by neg_logw (max logw), FIFO tie-break.
// Returns true if a should come before b in heap ordering (a pops first).
__device__ static bool heap_before(const HeapEntry* a, const HeapEntry* b) {
    if (a->neg_logw != b->neg_logw) {
        // Smaller neg_logw = larger logw = higher priority = pops first.
        return a->neg_logw < b->neg_logw;
    }
    // FIFO: smaller push_order pops first.
    return a->push_order < b->push_order;
}

// Max-heap (largest-priority first) via sift operations.
// The entry at index 0 is the one to pop (highest priority).

__device__ static void heap_sift_up(HeapEntry* h, int n) {
    int i = n - 1;
    while (i > 0) {
        int parent = (i - 1) >> 1;
        if (heap_before(&h[i], &h[parent])) {
            HeapEntry tmp = h[i]; h[i] = h[parent]; h[parent] = tmp;
            i = parent;
        } else break;
    }
}

__device__ static void heap_sift_down(HeapEntry* h, int n) {
    int i = 0;
    for (;;) {
        int best = i;
        int l = 2*i + 1, r = 2*i + 2;
        if (l < n && heap_before(&h[l], &h[best])) best = l;
        if (r < n && heap_before(&h[r], &h[best])) best = r;
        if (best == i) break;
        HeapEntry tmp = h[i]; h[i] = h[best]; h[best] = tmp;
        i = best;
    }
}

__device__ static void heap_push(HeapEntry* h, int* sz, const HeapEntry* e) {
    h[*sz] = *e;
    (*sz)++;
    heap_sift_up(h, *sz);
}

__device__ static HeapEntry heap_pop(HeapEntry* h, int* sz) {
    HeapEntry top = h[0];
    (*sz)--;
    if (*sz > 0) {
        h[0] = h[*sz];
        heap_sift_down(h, *sz);
    }
    return top;
}

// ── Kernel ───────────────────────────────────────────────────────────────
//
// Parameters:
//   logits       [depth × vocab] f32  — draft logits, row-major
//   top_tokens   [depth × topk]  i32  — output: token ids
//   top_log_probs[depth × topk]  f32  — output: log-probs (= logit - log_z)
//   attn_bias    [max_n × max_n] f32  — output: row-major mask (0 / -INF)
//   parent_indices[max_n]        i32  — output: parent slot per slot
//   slot_depth   [max_n]         i32  — output: depth per slot
//   child_of_cand[max_n × topk]  i32  — output: child slot per (slot, rank)
//   big_n_out    [1]             i32  — output: actual 1 + num_nodes
//   node_tokens  [max_n]         i32  — output: token per slot (slot 0 = unused/-1)
//   vocab, depth, topk: dimensions
//   max_nodes: tree node budget
//   logw_cutoff: early-stop threshold (NEG_INFINITY to disable)

extern "C" __global__ void ddtree_build_and_linearize_f32(
    const float* __restrict__ logits,        // [depth × vocab]
    int*   __restrict__ top_tokens_out,      // [depth × topk]  i32
    float* __restrict__ top_log_probs_out,   // [depth × topk]  f32
    float* __restrict__ attn_bias,           // [max_n × max_n] f32
    int*   __restrict__ parent_indices,      // [max_n]          i32
    int*   __restrict__ slot_depth,          // [max_n]          i32
    int*   __restrict__ child_of_cand,       // [max_n × topk]   i32
    int*   __restrict__ big_n_out,           // [1]              i32
    int*   __restrict__ node_tokens_out,     // [max_n]          i32
    int vocab, int depth, int topk,
    int max_nodes,
    float logw_cutoff
) {
    // Only thread 0 in the single workgroup runs this; all others skip.
    // (Block is [1,1,1] so this guard is a no-op but makes intent clear.)
    if (threadIdx.x != 0) return;

    // ── Phase 1: per-row top-K + log-sum-exp ─────────────────────────
    //
    // For each row r in 0..depth:
    //   - find the top-K (val, idx) pairs, tie-break: equal val → smaller idx wins
    //   - compute log_z via f64 accumulator (matches host topk_from_logits exactly)
    //   - write top_tokens_out[r*topk..], top_log_probs_out[r*topk..]
    //
    // Storage in registers/stack (depth ≤ 63, topk ≤ 8):
    //   top_tok[MAX_DEPTH][MAX_TOPK], top_lp[MAX_DEPTH][MAX_TOPK]

    int   top_tok[BUILD_MAX_DEPTH][BUILD_MAX_TOPK];
    float top_lp [BUILD_MAX_DEPTH][BUILD_MAX_TOPK];

    for (int r = 0; r < depth; r++) {
        const float* row = logits + (size_t)r * vocab;

        // Running max (for numerically stable log-sum-exp).
        float gmax = -INFINITY;
        for (int i = 0; i < vocab; i++) {
            float v = row[i];
            if (v > gmax) gmax = v;
        }

        // Top-K with ascending token-index tie-break.
        // We maintain a sorted array of size K (descending by val; within
        // equal val, ascending by idx — matching the host's min-heap secondary).
        // Implementation: for each candidate, find insertion point and shift.
        int   tk[BUILD_MAX_TOPK];
        float tv[BUILD_MAX_TOPK];
        for (int j = 0; j < topk; j++) { tk[j] = -1; tv[j] = -INFINITY; }

        for (int i = 0; i < vocab; i++) {
            float v = row[i];
            // Insertion position: first j where v > tv[j], OR
            // first j where v == tv[j] && i < tk[j] (ascending idx tie-break).
            // We want to keep the top-K; tv is stored descending.
            int ins = -1;
            for (int j = 0; j < topk; j++) {
                if (v > tv[j]) { ins = j; break; }
                if (v == tv[j] && i < tk[j]) { ins = j; break; }
            }
            if (ins < 0) continue; // below all current top-K
            // Shift down from topk-1 to ins+1 to make room.
            for (int j = topk - 1; j > ins; j--) {
                tv[j] = tv[j-1];
                tk[j] = tk[j-1];
            }
            tv[ins] = v;
            tk[ins] = i;
        }

        // log-sum-exp with f64 accumulator (matches host exactly).
        double sum_exp = 0.0;
        for (int i = 0; i < vocab; i++) {
            sum_exp += (double)expf(row[i] - gmax);
        }
        float log_z = gmax + (float)log(sum_exp);

        for (int j = 0; j < topk; j++) {
            top_tok[r][j] = tk[j];
            top_lp [r][j] = tv[j] - log_z;
        }
    }

    // Write the top-K outputs (consumed by SWOR path in Rust as pos_cands).
    for (int r = 0; r < depth; r++) {
        for (int j = 0; j < topk; j++) {
            top_tokens_out   [r * topk + j] = top_tok[r][j];
            top_log_probs_out[r * topk + j] = top_lp [r][j];
        }
    }

    // ── Phase 2: best-first heap expansion ────────────────────────────
    //
    // Exact mirror of build_ddtree_tree_bounded in ddtree.rs:216.
    // Uses volatile for logw arithmetic to prevent FMA fusion.
    //
    // Node storage (up to max_nodes, max MAX_BUDGET=60):
    struct Node {
        int   token;
        int   depth;
        int   parent_index; // -1 = root; else index into nodes[]
        float logw;
    };
    Node nodes[BUILD_MAX_BUDGET];
    int  n_nodes = 0;

    // child_map[slot][token] → child node index.  We can't store a HashMap
    // in a kernel; instead we keep a flat array child_cmap[MAX_N][MAX_TOPK]
    // that maps (slot, rank) → child index exactly like child_of_cand, but
    // indexed by the draft token's RANK at its depth (not by arbitrary token
    // id).  This is sufficient because the tree only inserts tokens that are
    // in the top-K, so rank is known at insert time.
    //
    // child_slot_map[parent_slot * MAX_TOPK + rank] = child_node_index+1
    // (0 = empty, index+1 to distinguish from the root).
    // This allows child_of_cand computation in Phase 3.
    //
    // Actually we need child lookup by TOKEN id (not rank) for child_of_cand.
    // child_of_cand[slot * topk + r] = tree.child_maps[slot].get(&pos_cands[depth*k+r]).
    // pos_cands at depth d, rank r = top_tokens[d * topk + r].
    //
    // So child lookup key = token id, not rank.  We store a parallel mapping:
    // node_child_tok[slot][r] = token placed as the r-th child of slot (in insertion order).
    // node_child_idx[slot][r] = child node index.
    // node_child_cnt[slot] = number of children.
    // This is equivalent to the HashMap in the host.

    int node_child_tok[BUILD_MAX_N][BUILD_MAX_TOPK]; // token ids of each child
    int node_child_idx[BUILD_MAX_N][BUILD_MAX_TOPK]; // child node index
    int node_child_cnt[BUILD_MAX_N];
    for (int s = 0; s < BUILD_MAX_N; s++) {
        node_child_cnt[s] = 0;
        for (int j = 0; j < BUILD_MAX_TOPK; j++) {
            node_child_tok[s][j] = -1;
            node_child_idx[s][j] = -1;
        }
    }

    HeapEntry heap[BUILD_MAX_HEAP];
    int heap_sz = 0;
    uint64_t push_counter = 0;

    // Seed: depth=1, rank=0, parent=-1, logw=top_lp[0][0].
    if (depth > 0 && max_nodes > 0 && top_tok[0][0] >= 0) {
        float first_logw = top_lp[0][0];
        HeapEntry seed;
        seed.neg_logw    = -first_logw;
        seed.push_order  = push_counter++;
        seed.depth       = 1;
        seed.rank        = 0;
        seed.parent_index = -1;
        seed.logw        = first_logw;
        heap_push(heap, &heap_sz, &seed);
    }

    while (heap_sz > 0 && n_nodes < max_nodes) {
        HeapEntry e = heap_pop(heap, &heap_sz);

        // logw_cutoff pruner (min_nodes=0 so cutoff always active).
        if (e.logw < logw_cutoff) break;

        int d = e.depth;
        int rank = e.rank;
        int par = e.parent_index;

        int token = top_tok[d - 1][rank];
        int current_index = n_nodes;

        nodes[current_index].token        = token;
        nodes[current_index].depth        = d;
        nodes[current_index].parent_index = par;
        nodes[current_index].logw         = e.logw;
        n_nodes++;

        // Register as child of parent.
        int parent_slot = (par < 0) ? 0 : (par + 1);
        int cnt = node_child_cnt[parent_slot];
        if (cnt < BUILD_MAX_TOPK) {
            node_child_tok[parent_slot][cnt] = token;
            node_child_idx[parent_slot][cnt] = current_index;
            node_child_cnt[parent_slot]++;
        }

        // Push sibling (d, rank+1).
        if (rank + 1 < topk) {
            int rank_next = rank + 1;
            // Use volatile to prevent FMA fusion across the two operations.
            volatile float lw = e.logw;
            volatile float sub = top_lp[d-1][rank];
            volatile float add = top_lp[d-1][rank_next];
            float sibling_logw = (float)lw - (float)sub + (float)add;
            HeapEntry sib;
            sib.neg_logw     = -sibling_logw;
            sib.push_order   = push_counter++;
            sib.depth        = d;
            sib.rank         = rank_next;
            sib.parent_index = par;
            sib.logw         = sibling_logw;
            heap_push(heap, &heap_sz, &sib);
        }

        // Push child (d+1, 0).
        if (d < depth) {
            volatile float lw = e.logw;
            volatile float add = top_lp[d][0];
            float child_logw = (float)lw + (float)add;
            HeapEntry ch;
            ch.neg_logw     = -child_logw;
            ch.push_order   = push_counter++;
            ch.depth        = d + 1;
            ch.rank         = 0;
            ch.parent_index = current_index;
            ch.logw         = child_logw;
            heap_push(heap, &heap_sz, &ch);
        }
    }

    int big_n = 1 + n_nodes;

    // ── Phase 3: write outputs ────────────────────────────────────────

    // parent_indices[0] = -1 (root slot).
    // parent_indices[i+1] = node[i].parent_index < 0 ? 0 : node[i].parent_index + 1.
    parent_indices[0] = -1;
    for (int i = 0; i < n_nodes; i++) {
        int p = nodes[i].parent_index;
        parent_indices[i + 1] = (p < 0) ? 0 : (p + 1);
    }

    // slot_depth[0] = 0 (root).
    // slot_depth[i+1] = node[i].depth.
    slot_depth[0] = 0;
    for (int i = 0; i < n_nodes; i++) {
        slot_depth[i + 1] = nodes[i].depth;
    }

    // node_tokens_out[0] = -1 (root; caller fills seed token).
    // node_tokens_out[i+1] = node[i].token.
    node_tokens_out[0] = -1;
    for (int i = 0; i < n_nodes; i++) {
        node_tokens_out[i + 1] = nodes[i].token;
    }

    // child_of_cand[slot * topk + r]:
    //   For slot s and SWOR candidate rank r at depth slot_depth[s]:
    //     token = top_tokens_out[slot_depth[s] * topk + r]  (= pos_cands[depth*k+r] in SWOR)
    //     look up token in node_child_tok[s] → child node index, or -1.
    //   Output is a node index (into nodes[]), not a slot index.
    //   Matches the host:  child_of_cand[s * k + r] = tree.child_maps[s].get(&token)  (Option<usize>)
    //   which returns the 0-based node index (child_of_cand is -1 if absent).
    for (int s = 0; s < big_n; s++) {
        int d = slot_depth[s];
        for (int r = 0; r < topk; r++) {
            int coc = -1;
            if (d < depth) {
                // token at this position rank
                int tok = top_tokens_out[d * topk + r];
                int nc = node_child_cnt[s];
                for (int c = 0; c < nc; c++) {
                    if (node_child_tok[s][c] == tok) {
                        coc = node_child_idx[s][c];
                        break;
                    }
                }
            }
            child_of_cand[s * topk + r] = coc;
        }
    }

    big_n_out[0] = big_n;

    // ── Phase 4: attn_bias mask (inline ddtree_build_attn_mask_f32) ──
    //
    // For each row i in 0..big_n: fill with -INF, then walk ancestor chain
    // setting 0.0 at each visited slot j.
    for (int i = 0; i < big_n; i++) {
        float* row = attn_bias + (size_t)i * big_n;
        for (int k = 0; k < big_n; k++) row[k] = -INFINITY;
        int j = i;
        while (j >= 0) {
            row[j] = 0.0f;
            j = parent_indices[j];
        }
    }
}
