// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Upstream-style CPU n-gram modular hash pool (llama.cpp `ngram-mod`).
//!
//! Direct-mapped table: hash(`n_match` prior tokens) → next token. Empty
//! sentinel is [`u32::MAX`]. Production defaults match the daemon-ready
//! primitive: 2²² slots (~16 MiB), `n_match=24`, draft gate `n_min=48`,
//! cap `n_max=64`. Occupancy above 25% and five consecutive low-acceptance
//! attempts (`drafted > 0 && accepted * 4 < drafted`) both clear the table.
//!
//! [`MtpNgramContext`] is the shared request-local lifecycle used by the
//! Qwen3.5 and Qwen4 MTP drafters. At every request boundary
//! ([`MtpNgramContext::begin_request`]) the pool is cleared (table memory
//! reused) and reseeded from the full canonical prompt, so another request's
//! transitions never change this request's candidates. During the request
//! only tokens the emitter retained ([`MtpNgramContext::sync_emitted`]) are
//! learned; speculative candidates and rejected verify tails never are.
//! [`NgramModPool::draft_into`] lets the per-window proposal reuse one
//! candidate buffer instead of allocating.
//!
//! Pure host Rust. Existing [`crate::spec::PldMatcher`] / [`crate::spec::NgramCache`]
//! are unchanged for DFlash and legacy standalone users.

/// Multiplier for the modular rolling hash (wrapping `u64`).
pub const HASH_MUL: u64 = 6_364_136_223_846_793_005;

/// Empty direct-map slot.
pub const EMPTY: u32 = u32::MAX;

/// Configuration for [`NgramModPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NgramModConfig {
    /// Power-of-two direct-map capacity (number of `u32` slots).
    pub capacity: usize,
    /// Exact n-gram window length hashed as the key.
    pub n_match: usize,
    /// Minimum chained draft tokens required for a successful [`NgramModPool::draft`].
    pub n_min: usize,
    /// Hard cap on chained draft tokens (also limited by caller_max).
    pub n_max: usize,
}

impl Default for NgramModConfig {
    fn default() -> Self {
        Self {
            capacity: 1 << 22,
            n_match: 24,
            n_min: 48,
            n_max: 64,
        }
    }
}

/// Direct-mapped n-gram → next-token pool.
pub struct NgramModPool {
    config: NgramModConfig,
    /// `capacity` slots; `EMPTY` means unoccupied.
    table: Vec<u32>,
    /// Exact number of non-empty slots.
    occupied: usize,
    /// Bit mask for capacity (power of two).
    mask: usize,
    /// `HASH_MUL.wrapping_pow((n_match - 1) as u32)` for rolling updates.
    mul_pow: u64,
    /// Consecutive low-acceptance draft attempts; pool clears at 5.
    low_accept_streak: u32,
}

impl NgramModPool {
    /// Build a pool from `config`. Validates power-of-two capacity and draft bounds.
    pub fn new(config: NgramModConfig) -> Result<Self, &'static str> {
        if config.capacity == 0 || !config.capacity.is_power_of_two() {
            return Err("ngram_mod capacity must be a non-zero power of two");
        }
        if config.n_match == 0 {
            return Err("ngram_mod n_match must be >= 1");
        }
        if config.n_max == 0 {
            return Err("ngram_mod n_max must be >= 1");
        }
        if config.n_min > config.n_max {
            return Err("ngram_mod n_min must be <= n_max");
        }
        let mul_pow = if config.n_match <= 1 {
            1
        } else {
            HASH_MUL.wrapping_pow((config.n_match - 1) as u32)
        };
        Ok(Self {
            config,
            table: vec![EMPTY; config.capacity],
            occupied: 0,
            mask: config.capacity - 1,
            mul_pow,
            low_accept_streak: 0,
        })
    }

    /// Borrow the config used to construct this pool.
    #[inline]
    pub fn config(&self) -> &NgramModConfig {
        &self.config
    }

    /// Exact number of non-empty table slots.
    #[inline]
    pub fn occupied(&self) -> usize {
        self.occupied
    }

    /// Zero the table, occupancy, and low-acceptance streak.
    pub fn clear(&mut self) {
        self.table.fill(EMPTY);
        self.occupied = 0;
        self.low_accept_streak = 0;
    }

    /// Index every next-token position `j` in
    /// `max(next_token_start, n_match) .. context.len()` as
    /// `context[j - n_match .. j] → context[j]`.
    ///
    /// Uses a rolling hash across consecutive windows. After the batch, if
    /// occupancy is strictly above `capacity / 4`, the pool is cleared.
    pub fn insert_range(&mut self, context: &[u32], next_token_start: usize) {
        let n = self.config.n_match;
        let start = next_token_start.max(n);
        if start >= context.len() {
            return;
        }

        // Initial key window context[start - n .. start].
        let mut hash = hash_window(&context[start - n..start]);
        let mut j = start;
        loop {
            self.store(hash, context[j]);
            j += 1;
            if j >= context.len() {
                break;
            }
            // Slide: drop context[j - 1 - n], append context[j - 1].
            let old = context[j - 1 - n];
            let newly = context[j - 1];
            hash = roll_hash(hash, old, newly, self.mul_pow);
            debug_assert_eq!(
                hash,
                hash_window(&context[j - n..j]),
                "rolling hash must match full recompute"
            );
        }

        if self.occupied > self.config.capacity / 4 {
            // A storage reset also clears the low-acceptance streak.
            self.clear();
        }
    }

    /// Chain table hits by sliding the exact `n_match`-token window over
    /// `context`'s suffix. Returns `None` unless at least `n_min` tokens
    /// chain; length is capped by `n_max` and `caller_max`.
    ///
    /// Allocating wrapper over [`draft_into`](Self::draft_into); the only heap
    /// allocation is the returned candidate `Vec`.
    pub fn draft(&self, context: &[u32], caller_max: usize) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        if self.draft_into(context, caller_max, &mut out) {
            Some(out)
        } else {
            None
        }
    }

    /// Like [`draft`](Self::draft), but writes into `out` (cleared first,
    /// capacity reused). Returns `true` iff `out.len() >= n_min`; on `false`,
    /// `out` is left empty. No allocation when `out` already has capacity
    /// `>= min(n_max, caller_max)`.
    pub fn draft_into(&self, context: &[u32], caller_max: usize, out: &mut Vec<u32>) -> bool {
        out.clear();
        let n = self.config.n_match;
        let limit = self.config.n_max.min(caller_max);
        if limit == 0 || context.len() < n || self.config.n_min > limit {
            return false;
        }
        out.reserve(limit);

        // The sliding key is tracked purely as a rolling hash; the token that
        // ages out of the window is read from `context` for the first `n`
        // steps and from previously drafted tokens thereafter.
        let base = context.len() - n;
        let mut hash = hash_window(&context[base..]);

        for k in 0..limit {
            let idx = (hash as usize) & self.mask;
            let tok = self.table[idx];
            if tok == EMPTY {
                break;
            }
            out.push(tok);

            let old = if k < n {
                context[base + k]
            } else {
                out[k - n]
            };
            hash = roll_hash(hash, old, tok, self.mul_pow);
        }

        if out.len() < self.config.n_min {
            out.clear();
            false
        } else {
            true
        }
    }

    /// Record one draft attempt's `(drafted, accepted)` counts.
    ///
    /// A low-acceptance attempt is `drafted > 0 && accepted * 4 < drafted`.
    /// Each low attempt increments a consecutive streak; any other attempt
    /// resets the streak to zero. On streak 5 the pool is cleared and this
    /// returns `true`; otherwise it returns `false`.
    pub fn record_draft_result(&mut self, drafted: u32, accepted: u32) -> bool {
        // accepted/drafted < 1/4  ⇔  accepted * 4 < drafted (drafted > 0).
        let low = drafted > 0 && u64::from(accepted).saturating_mul(4) < u64::from(drafted);
        if !low {
            self.low_accept_streak = 0;
            return false;
        }
        self.low_accept_streak = self.low_accept_streak.saturating_add(1);
        if self.low_accept_streak < 5 {
            return false;
        }
        self.clear();
        true
    }

    #[inline]
    fn store(&mut self, hash: u64, next: u32) {
        let idx = (hash as usize) & self.mask;
        if self.table[idx] == EMPTY {
            self.occupied += 1;
        }
        self.table[idx] = next;
    }
}

/// Request-local n-gram proposal owner shared by the Qwen3.5 and Qwen4 MTP
/// drafters.
///
/// Owns one [`NgramModPool`], the request's canonical token history
/// (`prompt ++ emitted`), and a reusable candidate buffer. The pool is cleared
/// and reseeded from the full prompt at every request boundary
/// ([`begin_request`](Self::begin_request)), so another request's transitions
/// never change this request's candidates. Only tokens the emitter retained
/// ([`sync_emitted`](Self::sync_emitted)) are ever learned: speculative
/// candidates and rejected verify tails are not history.
pub struct MtpNgramContext {
    pool: NgramModPool,
    /// `prompt ++ emitted`.
    history: Vec<u32>,
    /// Prefix of `history` whose transitions are already in the pool.
    indexed_until: usize,
    /// Number of `emitted` tokens already appended to `history`.
    emitted_len: usize,
    /// Reusable draft buffer; capacity reserved to `n_max` once in `new`.
    candidate: Vec<u32>,
    /// `begin_request` ran and `reset_request` has not since.
    begun: bool,
}

impl MtpNgramContext {
    /// Build the pool and reserve the candidate buffer (`n_max` tokens).
    pub fn new(config: NgramModConfig) -> Result<Self, &'static str> {
        let pool = NgramModPool::new(config)?;
        Ok(Self {
            pool,
            history: Vec::new(),
            indexed_until: 0,
            emitted_len: 0,
            candidate: Vec::with_capacity(config.n_max),
            begun: false,
        })
    }

    /// Config the underlying pool was built with.
    #[inline]
    pub fn config(&self) -> &NgramModConfig {
        self.pool.config()
    }

    /// Request boundary: clear the pool (table memory is reused; the fill is
    /// skipped when the pool is already empty) and the history, then seed both
    /// from the FULL canonical `prompt` (every transition
    /// `prompt[j-n..j] -> prompt[j]`). Sets `begun`.
    pub fn begin_request(&mut self, prompt: &[u32]) {
        if self.pool.occupied() > 0 {
            self.pool.clear();
        } else {
            // Table already empty; still drop any stale low-acceptance streak.
            self.pool.low_accept_streak = 0;
        }
        self.history.clear();
        self.history.extend_from_slice(prompt);
        self.emitted_len = 0;
        let n_match = self.pool.config().n_match;
        self.pool.insert_range(&self.history, n_match);
        self.indexed_until = self.history.len();
        self.candidate.clear();
        self.begun = true;
    }

    /// Whether [`begin_request`](Self::begin_request) ran for the live request.
    #[inline]
    pub fn is_begun(&self) -> bool {
        self.begun
    }

    /// Append only the not-yet-seen suffix of `emitted` (the emitter-retained
    /// output stream, containing the first token/seed exactly once) to the
    /// history and index the new transitions. No-op before `begin_request`.
    /// Never learns candidates or rejected tails.
    pub fn sync_emitted(&mut self, emitted: &[u32]) {
        if !self.begun {
            return;
        }
        debug_assert!(
            emitted.len() >= self.emitted_len,
            "MTP n-gram emitted history must grow monotonically"
        );
        if emitted.len() <= self.emitted_len {
            return;
        }
        self.history.extend_from_slice(&emitted[self.emitted_len..]);
        self.pool.insert_range(&self.history, self.indexed_until);
        self.indexed_until = self.history.len();
        self.emitted_len = emitted.len();
    }

    /// [`sync_emitted`](Self::sync_emitted)`(emitted)`, then draft from the
    /// history with `caller_max`. `Some(candidates)` iff a non-empty draft of
    /// at least `n_min` tokens exists (length `<= min(n_max, caller_max)`);
    /// the slice borrows the internal candidate buffer. `None` before
    /// `begin_request`.
    pub fn propose(&mut self, emitted: &[u32], caller_max: usize) -> Option<&[u32]> {
        if !self.begun {
            return None;
        }
        self.sync_emitted(emitted);
        if self.pool.draft_into(&self.history, caller_max, &mut self.candidate)
            && !self.candidate.is_empty()
        {
            Some(&self.candidate)
        } else {
            self.candidate.clear();
            None
        }
    }

    /// Record one draft window's `(drafted, accepted)` counts; applies the
    /// five-consecutive-low-acceptance pool clear
    /// ([`NgramModPool::record_draft_result`]).
    pub fn observe_result(&mut self, drafted: usize, accepted: usize) {
        let drafted = u32::try_from(drafted).unwrap_or(u32::MAX);
        let accepted = u32::try_from(accepted).unwrap_or(u32::MAX);
        let _ = self.pool.record_draft_result(drafted, accepted);
    }

    /// Drop history/emitted bookkeeping and set `begun = false`. The pool
    /// table is cleared lazily by the next `begin_request`; allocations are
    /// kept.
    pub fn reset_request(&mut self) {
        self.history.clear();
        self.indexed_until = 0;
        self.emitted_len = 0;
        self.candidate.clear();
        self.begun = false;
    }

    /// Canonical request history (`prompt ++ emitted`).
    #[inline]
    pub fn history(&self) -> &[u32] {
        &self.history
    }
}

/// Full modular hash of an `n_match`-token window.
#[inline]
fn hash_window(tokens: &[u32]) -> u64 {
    let mut h = 0u64;
    for &t in tokens {
        h = h.wrapping_mul(HASH_MUL).wrapping_add(u64::from(t));
    }
    h
}

/// Slide one token: drop `old` from the left, append `new_tok` on the right.
#[inline]
fn roll_hash(hash: u64, old: u32, new_tok: u32, mul_pow: u64) -> u64 {
    // h' = (h - old * M^{n-1}) * M + new
    hash.wrapping_sub(u64::from(old).wrapping_mul(mul_pow))
        .wrapping_mul(HASH_MUL)
        .wrapping_add(u64::from(new_tok))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg(capacity: usize, n_match: usize, n_min: usize, n_max: usize) -> NgramModConfig {
        NgramModConfig {
            capacity,
            n_match,
            n_min,
            n_max,
        }
    }

    #[test]
    fn default_config_matches_production() {
        let d = NgramModConfig::default();
        assert_eq!(d.capacity, 1 << 22);
        assert_eq!(d.n_match, 24);
        assert_eq!(d.n_min, 48);
        assert_eq!(d.n_max, 64);
    }

    #[test]
    fn new_rejects_invalid_config() {
        assert!(NgramModPool::new(small_cfg(0, 2, 1, 2)).is_err());
        assert!(NgramModPool::new(small_cfg(3, 2, 1, 2)).is_err()); // not pow2
        assert!(NgramModPool::new(small_cfg(8, 0, 1, 2)).is_err());
        assert!(NgramModPool::new(small_cfg(8, 2, 3, 2)).is_err()); // n_min > n_max
        assert!(NgramModPool::new(small_cfg(8, 2, 1, 0)).is_err());
    }

    #[test]
    fn rolling_hash_equals_recompute() {
        let n = 5usize;
        let mul_pow = HASH_MUL.wrapping_pow((n - 1) as u32);
        let tokens: Vec<u32> = (1..30).collect();
        let mut h = hash_window(&tokens[0..n]);
        assert_eq!(h, hash_window(&tokens[0..n]));
        for j in n..tokens.len() {
            let old = tokens[j - n];
            let newly = tokens[j];
            h = roll_hash(h, old, newly, mul_pow);
            assert_eq!(
                h,
                hash_window(&tokens[j - n + 1..j + 1]),
                "mismatch at j={j}"
            );
        }
    }

    #[test]
    fn insert_range_rolling_matches_per_window_store() {
        // Capacity large enough that 27 inserts stay under capacity/4.
        let cfg = small_cfg(256, 3, 1, 8);
        let mut a = NgramModPool::new(cfg).unwrap();
        let mut b = NgramModPool::new(cfg).unwrap();
        let ctx: Vec<u32> = (10..40).collect();
        a.insert_range(&ctx, 0);
        let n = cfg.n_match;
        for j in n..ctx.len() {
            let h = hash_window(&ctx[j - n..j]);
            b.store(h, ctx[j]);
        }
        assert!(a.occupied() > 0, "must not have occupancy-reset");
        assert_eq!(a.table, b.table);
        assert_eq!(a.occupied(), b.occupied());
    }

    #[test]
    fn min_gate_closure() {
        let cfg = small_cfg(64, 2, 4, 8);
        let mut pool = NgramModPool::new(cfg).unwrap();
        // Train a short chain: only three next-tokens from [1,2].
        let ctx = vec![1, 2, 3, 4, 5];
        pool.insert_range(&ctx, 0);
        // Draft from [1,2] chains 3,4,5 then empty — length 3 < n_min=4.
        assert_eq!(pool.draft(&[1, 2], 8), None);
        // Extend chain to satisfy n_min=4.
        let ctx2 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        pool.insert_range(&ctx2, 0);
        let d = pool.draft(&[1, 2], 8).expect("min gate should open");
        assert!(d.len() >= 4);
        assert_eq!(&d[..4], &[3, 4, 5, 6]);
    }

    #[test]
    fn exact_chain_and_caller_cap() {
        let cfg = small_cfg(128, 2, 2, 5);
        let mut pool = NgramModPool::new(cfg).unwrap();
        // Linear chain 0,1,2,...,20
        let ctx: Vec<u32> = (0..21).collect();
        pool.insert_range(&ctx, 0);
        // Full n_max=5
        let d = pool.draft(&[0, 1], 100).unwrap();
        assert_eq!(d, vec![2, 3, 4, 5, 6]);
        // caller_max trims below n_max
        let d2 = pool.draft(&[0, 1], 3).unwrap();
        assert_eq!(d2, vec![2, 3, 4]);
        // caller_max below n_min → None
        assert_eq!(pool.draft(&[0, 1], 1), None);
    }

    #[test]
    fn incremental_range_insertion() {
        let cfg = small_cfg(64, 3, 2, 8);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let ctx = vec![10, 11, 12, 13, 14, 15, 16, 17];
        // First half through index 4 inclusive as next-token positions.
        pool.insert_range(&ctx[..5], 0); // j = 3,4 → (10,11,12)->13, (11,12,13)->14
        assert_eq!(pool.occupied(), 2);
        // Resume from next_token_start = 5
        pool.insert_range(&ctx, 5); // j = 5,6,7
        assert_eq!(pool.occupied(), 5);
        let d = pool.draft(&[10, 11, 12], 8).unwrap();
        assert_eq!(&d[..5], &[13, 14, 15, 16, 17]);
    }

    #[test]
    fn collision_occupancy_exact() {
        // capacity 4 → only 4 slots; force many distinct windows into few buckets.
        let cfg = small_cfg(4, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let mut ctx: Vec<u32> = Vec::new();
        for i in 0..50u32 {
            ctx.push(i.wrapping_mul(17).wrapping_add(3));
        }
        pool.insert_range(&ctx, 0);
        // Either still filled (≤ capacity/4 = 1) or cleared by occupancy reset.
        let occ = pool.occupied();
        assert!(occ <= 4);
        let non_empty = pool.table.iter().filter(|&&t| t != EMPTY).count();
        assert_eq!(occ, non_empty);

        // Overwrite path: same keys must keep occupancy exact.
        if occ > 0 {
            pool.insert_range(&ctx[..4], 0);
            let non_empty2 = pool.table.iter().filter(|&&t| t != EMPTY).count();
            assert_eq!(pool.occupied(), non_empty2);
        }
    }

    #[test]
    fn occupancy_reset_above_quarter() {
        // capacity 16 → threshold capacity/4 = 4; reset when occupied > 4.
        let cfg = small_cfg(16, 1, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        for _ in 0..4 {
            assert!(!pool.record_draft_result(10, 0));
        }
        assert_eq!(pool.low_accept_streak, 4);
        // n_match=1: key is single prior token. Distinct keys 0..8 → 9 inserts.
        let ctx: Vec<u32> = (0..10).collect();
        pool.insert_range(&ctx, 0);
        // After batch occupancy would be 9 > 4 → cleared.
        assert_eq!(pool.occupied(), 0);
        assert!(pool.table.iter().all(|&t| t == EMPTY));
        assert_eq!(pool.low_accept_streak, 0);

        // Exactly 4 distinct keys via first four next-token positions only.
        // ctx2[..5] → j=1..4 keys = 0,10,1,11.
        let mut pool2 = NgramModPool::new(cfg).unwrap();
        let ctx2 = vec![0, 10, 1, 11, 2, 12, 3, 13];
        pool2.insert_range(&ctx2[..5], 0);
        assert_eq!(pool2.occupied(), 4);
        assert!(!pool2.table.iter().all(|&t| t == EMPTY));
    }

    #[test]
    fn consecutive_low_accept_streak_clears() {
        let cfg = small_cfg(32, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let ctx = vec![1, 2, 3, 4, 5, 6];
        pool.insert_range(&ctx, 0);
        assert!(pool.occupied() > 0);

        // Four low attempts — no clear yet.
        for _ in 0..4 {
            assert!(!pool.record_draft_result(10, 0));
        }
        assert!(pool.occupied() > 0);
        assert_eq!(pool.low_accept_streak, 4);

        // A healthy attempt resets the streak.
        assert!(!pool.record_draft_result(10, 5));
        assert_eq!(pool.low_accept_streak, 0);
        let filled = pool.occupied();
        assert!(filled > 0);

        // Five subsequent low attempts clear.
        for i in 0..5 {
            let cleared = pool.record_draft_result(8, 0);
            if i < 4 {
                assert!(!cleared);
                assert_eq!(pool.low_accept_streak, i + 1);
            } else {
                assert!(cleared);
            }
        }
        assert_eq!(pool.occupied(), 0);
        assert_eq!(pool.low_accept_streak, 0);
    }

    #[test]
    fn draft_requires_context_suffix() {
        let cfg = small_cfg(16, 4, 1, 4);
        let pool = NgramModPool::new(cfg).unwrap();
        assert_eq!(pool.draft(&[1, 2, 3], 4), None); // shorter than n_match
        assert_eq!(pool.draft(&[], 4), None);
    }

    #[test]
    fn next_token_start_skips_early_positions() {
        let cfg = small_cfg(64, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        let ctx = vec![1, 2, 3, 4, 5, 6];
        // Only index j >= 4.
        pool.insert_range(&ctx, 4);
        // (1,2)->3 and (2,3)->4 must be absent; (3,4)->5 and (4,5)->6 present.
        assert_eq!(pool.draft(&[1, 2], 4), None);
        assert_eq!(pool.draft(&[3, 4], 4).unwrap(), vec![5, 6]);
    }

    #[test]
    fn clear_zeros_table_and_streak() {
        let cfg = small_cfg(16, 2, 1, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        pool.insert_range(&[1, 2, 3, 4], 0);
        assert!(pool.occupied() > 0);
        assert!(!pool.record_draft_result(10, 0));
        assert_eq!(pool.low_accept_streak, 1);
        pool.clear();
        assert_eq!(pool.occupied(), 0);
        assert!(pool.table.iter().all(|&t| t == EMPTY));
        assert_eq!(pool.low_accept_streak, 0);
        // Streak reset: need five more low attempts to clear again.
        pool.insert_range(&[1, 2, 3, 4], 0);
        for _ in 0..4 {
            assert!(!pool.record_draft_result(4, 0));
        }
        assert!(pool.record_draft_result(4, 0));
    }

    // ---- draft_into -------------------------------------------------------

    #[test]
    fn draft_into_matches_draft_including_misses() {
        let ctx_tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        for n_min in [1usize, 2, 3] {
            let cfg = small_cfg(1 << 20, 2, n_min, 4);
            let mut pool = NgramModPool::new(cfg).unwrap();
            pool.insert_range(&ctx_tokens, 0);
            let contexts: [&[u32]; 7] = [
                &[1, 2],
                &[3, 4],
                &[5, 6],
                &[7, 8],
                &[9, 9],
                &[1],
                &[],
            ];
            for context in contexts {
                for caller_max in 0..=6usize {
                    let expected = pool.draft(context, caller_max);
                    // Junk in `out` must be cleared first.
                    let mut out: Vec<u32> = vec![99, 98, 97];
                    let hit = pool.draft_into(context, caller_max, &mut out);
                    match expected {
                        Some(v) => {
                            assert!(hit, "ctx={context:?} max={caller_max} n_min={n_min}");
                            assert_eq!(out, v);
                            assert!(out.len() >= n_min);
                        }
                        None => {
                            assert!(!hit, "ctx={context:?} max={caller_max} n_min={n_min}");
                            assert!(out.is_empty());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn draft_into_n_min_miss_returns_false_and_empty() {
        let cfg = small_cfg(1 << 20, 2, 3, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        pool.insert_range(&[1, 2, 3, 4, 5], 0);
        // (3,4)->5 chains only one token (< n_min = 3).
        let mut out = vec![7u32; 5];
        assert!(!pool.draft_into(&[3, 4], 4, &mut out));
        assert!(out.is_empty());
        assert_eq!(pool.draft(&[3, 4], 4), None);
    }

    #[test]
    fn draft_into_reuses_out_capacity() {
        let cfg = small_cfg(1 << 20, 2, 2, 4);
        let mut pool = NgramModPool::new(cfg).unwrap();
        pool.insert_range(&[1, 2, 3, 4, 5, 6, 7, 8], 0);
        let mut out: Vec<u32> = Vec::with_capacity(8);
        let ptr = out.as_ptr();
        for _ in 0..4 {
            assert!(pool.draft_into(&[1, 2], 4, &mut out));
            assert_eq!(out, vec![3, 4, 5, 6]);
            assert!(!pool.draft_into(&[9, 9], 4, &mut out));
            assert_eq!(out.as_ptr(), ptr);
            assert_eq!(out.capacity(), 8);
        }
    }

    // ---- MtpNgramContext --------------------------------------------------

    fn ctx_cfg() -> NgramModConfig {
        small_cfg(1 << 20, 2, 2, 4)
    }

    #[test]
    fn context_new_propagates_invalid_config() {
        assert!(MtpNgramContext::new(small_cfg(3, 2, 1, 4)).is_err());
        assert!(MtpNgramContext::new(small_cfg(16, 2, 5, 4)).is_err());
        let ctx = MtpNgramContext::new(ctx_cfg()).unwrap();
        assert_eq!(*ctx.config(), ctx_cfg());
        assert!(!ctx.is_begun());
        assert!(ctx.history().is_empty());
    }

    #[test]
    fn context_propose_before_begin_returns_none() {
        let mut ctx = MtpNgramContext::new(ctx_cfg()).unwrap();
        assert!(ctx.propose(&[1, 2, 3], 4).is_none());
        // sync before begin is a no-op.
        ctx.sync_emitted(&[1, 2, 3]);
        assert!(ctx.history().is_empty());
        assert!(!ctx.is_begun());
        // After a reset the context is un-begun again.
        ctx.begin_request(&[1, 2, 3, 4, 5, 6]);
        assert!(ctx.is_begun());
        ctx.reset_request();
        assert!(!ctx.is_begun());
        assert!(ctx.history().is_empty());
        assert!(ctx.propose(&[1, 2], 4).is_none());
    }

    #[test]
    fn context_full_prompt_seed_finds_copied_span() {
        let mut ctx = MtpNgramContext::new(ctx_cfg()).unwrap();
        let prompt: Vec<u32> = (100..110).collect();
        ctx.begin_request(&prompt);
        assert_eq!(ctx.history(), prompt.as_slice());
        // The model starts copying the prompt: first token + one more.
        let emitted = [100u32, 101];
        let cands = ctx.propose(&emitted, 4).expect("copied span must draft");
        assert_eq!(cands, &[102, 103, 104, 105]);
        // caller_max caps the candidate length.
        let capped = ctx.propose(&emitted, 3).expect("capped draft");
        assert_eq!(capped, &[102, 103, 104]);
        // caller_max below n_min yields no proposal.
        assert!(ctx.propose(&emitted, 1).is_none());
    }

    #[test]
    fn context_seed_token_learned_exactly_once() {
        let mut ctx = MtpNgramContext::new(ctx_cfg()).unwrap();
        ctx.begin_request(&[1, 2, 3]);
        // (1,2)->3 only.
        assert_eq!(ctx.pool.occupied(), 1);
        ctx.sync_emitted(&[4]);
        assert_eq!(ctx.history(), &[1, 2, 3, 4]);
        assert_eq!(ctx.pool.occupied(), 2); // + (2,3)->4
        // Re-syncing the same emitted stream (seed included once) is idempotent.
        ctx.sync_emitted(&[4]);
        ctx.sync_emitted(&[4]);
        assert_eq!(ctx.history(), &[1, 2, 3, 4]);
        assert_eq!(ctx.pool.occupied(), 2);
        assert_eq!(ctx.indexed_until, 4);
        assert_eq!(ctx.emitted_len, 1);
        // Only the new suffix is appended on growth.
        ctx.sync_emitted(&[4, 5, 6]);
        assert_eq!(ctx.history(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(ctx.pool.occupied(), 4);
        assert_eq!(ctx.emitted_len, 3);
        assert_eq!(ctx.indexed_until, 6);
    }

    #[test]
    fn context_candidates_and_rejected_tails_never_learned() {
        let mut ctx = MtpNgramContext::new(ctx_cfg()).unwrap();
        let prompt: Vec<u32> = (100..110).collect();
        ctx.begin_request(&prompt);
        let emitted = [100u32, 101];
        let first = ctx.propose(&emitted, 4).expect("draft").to_vec();
        let occupied = ctx.pool.occupied();
        let history_len = ctx.history().len();
        let second = ctx.propose(&emitted, 4).expect("draft").to_vec();
        assert_eq!(first, second);
        assert_eq!(ctx.pool.occupied(), occupied);
        assert_eq!(ctx.history().len(), history_len);
        let mut expected_history = prompt.clone();
        expected_history.extend_from_slice(&emitted);
        assert_eq!(ctx.history(), expected_history.as_slice());
        // A candidate token that the emitter has not retained is not history:
        // the bridge window (first[last], 7777) was never indexed.
        let tail = *first.last().unwrap();
        let mut probe = Vec::new();
        assert!(!ctx.pool.draft_into(&[tail, 7777], 4, &mut probe));
        // A shrunk emitted stream (rejected tail dropped by the emitter) is a
        // release-mode no-op; the history keeps what was already retained.
        #[cfg(not(debug_assertions))]
        {
            ctx.sync_emitted(&emitted[..1]);
            assert_eq!(ctx.history(), expected_history.as_slice());
        }
    }

    #[test]
    fn context_requests_are_isolated() {
        let mut ctx = MtpNgramContext::new(ctx_cfg()).unwrap();
        let prompt_a: Vec<u32> = (1..=10).collect();
        let prompt_b: Vec<u32> = (200..210).collect();

        ctx.begin_request(&prompt_a);
        assert_eq!(ctx.pool.occupied(), prompt_a.len() - 2);
        assert_eq!(
            ctx.propose(&[1, 2], 4).expect("A copies A"),
            &[3, 4, 5, 6]
        );

        ctx.reset_request();
        ctx.begin_request(&prompt_b);
        assert_eq!(ctx.history(), prompt_b.as_slice());
        assert_eq!(ctx.pool.occupied(), prompt_b.len() - 2);
        // Request A's transitions must not draft inside request B.
        assert!(ctx.propose(&[1, 2], 4).is_none());
        ctx.reset_request();
        ctx.begin_request(&prompt_b);
        assert_eq!(
            ctx.propose(&[200, 201], 4).expect("B copies B"),
            &[202, 203, 204, 205]
        );

        // begin_request without an intervening reset_request also isolates.
        ctx.begin_request(&prompt_a);
        assert_eq!(ctx.pool.occupied(), prompt_a.len() - 2);
        assert_eq!(ctx.history(), prompt_a.as_slice());
        assert_eq!(
            ctx.propose(&[1, 2], 4).expect("A again"),
            &[3, 4, 5, 6]
        );
        ctx.begin_request(&prompt_b);
        assert_eq!(ctx.pool.occupied(), prompt_b.len() - 2);
        assert!(ctx.propose(&[1, 2], 4).is_none());
    }

    #[test]
    fn context_candidate_buffer_is_not_reallocated() {
        let cfg = ctx_cfg();
        let mut ctx = MtpNgramContext::new(cfg).unwrap();
        assert!(ctx.candidate.capacity() >= cfg.n_max);
        let ptr = ctx.candidate.as_ptr();
        let cap = ctx.candidate.capacity();
        let prompt: Vec<u32> = (100..110).collect();
        for _ in 0..3 {
            ctx.begin_request(&prompt);
            for caller_max in [4usize, 3, 2, 1, 4] {
                let _ = ctx.propose(&[100, 101], caller_max);
                assert_eq!(ctx.candidate.as_ptr(), ptr);
                assert_eq!(ctx.candidate.capacity(), cap);
            }
            ctx.reset_request();
            assert_eq!(ctx.candidate.as_ptr(), ptr);
        }
    }

    #[test]
    fn context_observe_result_clears_after_five_low_attempts() {
        let cfg = small_cfg(1 << 20, 2, 1, 4);
        let mut ctx = MtpNgramContext::new(cfg).unwrap();
        let prompt: Vec<u32> = (1..=8).collect();
        ctx.begin_request(&prompt);
        assert!(ctx.pool.occupied() > 0);
        for _ in 0..4 {
            ctx.observe_result(4, 0);
            assert!(ctx.pool.occupied() > 0);
        }
        // A healthy attempt resets the streak.
        ctx.observe_result(4, 4);
        for _ in 0..4 {
            ctx.observe_result(8, 1);
            assert!(ctx.pool.occupied() > 0);
        }
        ctx.observe_result(4, 0);
        assert_eq!(ctx.pool.occupied(), 0);
        assert!(ctx.propose(&[1, 2], 4).is_none());
    }

    #[test]
    fn context_begin_request_resets_low_accept_streak() {
        let cfg = small_cfg(1 << 20, 2, 1, 4);
        let mut ctx = MtpNgramContext::new(cfg).unwrap();
        let prompt: Vec<u32> = (1..=8).collect();
        ctx.begin_request(&prompt);
        for _ in 0..4 {
            ctx.observe_result(4, 0);
        }
        ctx.begin_request(&prompt);
        // One more low attempt would have been the fifth without the reset.
        ctx.observe_result(4, 0);
        assert_eq!(ctx.pool.occupied(), prompt.len() - 2);
        // An already-empty pool still resets the streak.
        let mut empty = MtpNgramContext::new(cfg).unwrap();
        empty.begin_request(&[1]);
        assert_eq!(empty.pool.occupied(), 0);
        for _ in 0..4 {
            empty.observe_result(4, 0);
        }
        empty.begin_request(&[1]);
        assert_eq!(empty.pool.low_accept_streak, 0);
    }
}
