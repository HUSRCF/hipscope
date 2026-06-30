// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Stage 3c: on-GPU greedy tree follow.
//
// Eliminates ~244 B parents D2H (scratch.parent_indices → host child_maps
// reconstruction) and ~big_n × 4 B argmax D2H (verify_scratch.argmax → host
// follow_verified_tree) from the greedy (temp=0) ddtree cycle.
//
// The kernel is a single-workgroup, single-thread sequential walk.  At
// budget=60, k=4, depth=15 the walk is at most depth=15 steps; each step
// scans big_n≤61 slots to find the matching child.  Total: ≤ 915 operations —
// < 1 µs on any RDNA arch.
//
// ── Algorithm ─────────────────────────────────────────────────────────────
//
// Replicates `follow_verified_tree` (ddtree.rs:387–409) exactly:
//
//   current_slot = 0          // root
//   next_token   = argmax[0]  // target's prediction at root
//   loop:
//     find child: scan slots 1..big_n for parent_indices[s]==current_slot
//                 AND node_tokens[s]==next_token → child_slot s
//     if not found: break
//     accepted[accept_len++] = s - 1   // node index (0-based)
//     current_slot = s
//     if current_slot >= big_n: break
//     next_token   = argmax[current_slot]
//   bonus_token = next_token
//
// Output buffer layout (i32 elements):
//   follow_result[0]              = accept_len
//   follow_result[1]              = bonus_token  (i32)
//   follow_result[2 .. 2+accept_len] = accepted_node_indices (i32, 0-based)
//
// ── Byte-identity guarantee ───────────────────────────────────────────────
//
// The walk is deterministic (no floating-point tie-breaking needed — argmax
// values and token ids are exact integers).  The child scan order matches the
// slot order (1..big_n), which is the same order the host's child_maps were
// built in (loop over nodes in allocation order = insertion order matching
// the heap expansion).  Because `DdTree.child_maps[slot]` is a HashMap and
// `child_maps[slot].get(&token)` returns the unique entry for `token` (tokens
// are unique per slot — no two sibling nodes have the same token), the child
// lookup has no ordering ambiguity.  The GPU linear scan returns the same
// result as the HashMap lookup.
//
// Grid: [1,1,1].  Block: [1,1,1].

#include <hip/hip_runtime.h>

#define FOLLOW_MAX_N 61  // 1 + max_budget (budget ≤ 60)

// Parameters:
//   argmax        [big_n] i32 (stored in f32 buffer — read as int32_t)
//                 argmax[i] = target's greedy token at slot i.
//   parent_indices[big_n] i32
//                 parent_indices[0] = -1 (root).
//                 parent_indices[i] for i≥1: parent slot of node (i-1).
//   node_tokens   [big_n] i32
//                 node_tokens[0] = -1 (unused root sentinel).
//                 node_tokens[i] for i≥1: draft token at node (i-1).
//   follow_result [2 + max_accept] i32
//                 [0] = accept_len, [1] = bonus_token, [2..] = accepted_node_indices.
//   big_n         actual tree size (1 + num_nodes), ≤ FOLLOW_MAX_N.

extern "C" __global__ void ddtree_greedy_follow_f32(
    const int* __restrict__ argmax,          // [big_n] i32 (from verify argmax buf)
    const int* __restrict__ parent_indices,  // [big_n] i32
    const int* __restrict__ node_tokens,     // [big_n] i32
    int*       __restrict__ follow_result,   // [2 + max_accept] i32
    int big_n
) {
    if (threadIdx.x != 0) return;

    int accept_len   = 0;
    int current_slot = 0;
    int next_token   = argmax[0];

    // Accepted node indices (0-based), at most big_n-1 = 60 entries.
    int accepted[FOLLOW_MAX_N];

    for (;;) {
        // Find the child of current_slot whose token == next_token.
        int found_slot = -1;
        for (int s = 1; s < big_n; s++) {
            if (parent_indices[s] == current_slot && node_tokens[s] == next_token) {
                found_slot = s;
                break;
            }
        }
        if (found_slot < 0) break;  // no matching child → stop

        int node_index = found_slot - 1;  // 0-based node index
        accepted[accept_len++] = node_index;

        current_slot = found_slot;
        if (current_slot >= big_n) break;
        next_token = argmax[current_slot];
    }

    // Write output.
    follow_result[0] = accept_len;
    follow_result[1] = next_token;  // bonus token
    for (int i = 0; i < accept_len; i++) {
        follow_result[2 + i] = accepted[i];
    }
}
