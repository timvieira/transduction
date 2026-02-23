//! Peekaboo recursive decomposition ported from Python `peekaboo_recursive.py`.
//!
//! Produces per-symbol quotient/remainder FSAs for the next target symbol,
//! given an FST and a full target string.
//!
//! NFA states are `(fst_state, buf_len, extra_sym, truncated)` packed into u64.
//! The encoding is **step-independent**: the same buffer content always maps
//! to the same u64 regardless of which step we're at.
//!
//!   - On-target: buffer = full_target[:buf_len], `extra_sym = NO_EXTRA`
//!   - Off-target: buffer = full_target[:buf_len-1] + sym, `extra_sym = sym_idx`
//!
//! Packing:
//!   bits [63:32] = fst_state (u32)
//!   bits [31:17] = buf_len   (u15, max 32767)
//!   bits [16:1]  = extra_sym_idx (u16, 0xFFFF = NO_EXTRA for on-target)
//!   bit  [0]     = truncated
//!
//! ## FST-level closure optimization in `compute_all_arcs`
//!
//! The standard epsilon-closure (`eps_closure_single`) operates on packed u64
//! NFA states and is cached by packed state.  For off-target states (those with
//! `extra_sym != NO_EXTRA` and `buf_len > step_n`), the epsilon closure only
//! changes the FST state component — the buffer parameters `(buf_len, extra_sym)`
//! are invariant through epsilon transitions.  This means two packed states that
//! differ only in `extra_sym` produce structurally identical closures (same FST
//! states reachable), but the packed-state cache treats them as distinct entries.
//!
//! With V target-side symbols (e.g. V=50K for BPE), a single DFA state can
//! contain NFA elements for ~V distinct `extra_sym` values, all sharing the same
//! FST state.  Each `eps_closure_single` call does a BFS through O(|NFA|) states,
//! so the total cost is O(V × |NFA|) — the dominant bottleneck at large V.
//!
//! The optimization adds a local FST-level closure cache keyed by FST state
//! (u32) instead of packed NFA state (u64).  For each unique FST state, a
//! two-phase BFS computes:
//!
//!   1. **Non-truncated reachable**: follow epsilon-input arcs with epsilon
//!      output (buffer unchanged, stays non-truncated).
//!   2. **Truncated reachable**: from non-truncated states, follow epsilon-input
//!      arcs with non-epsilon output (causes truncation), then transitively
//!      follow all epsilon-input arcs (truncated states accept all outputs).
//!
//! The FST-level BFS uses pre-extracted `eps_input_arcs` slices (u32 states,
//! no packing/unpacking), which is cheaper per step than the packed-state BFS.
//! The cache collapses V redundant computations into one per unique FST state,
//! reducing the cost from O(V × |NFA|) BFS steps to O(V × |closure|) for
//! repacking + O(|FST states| × avg_eps_closure) for the BFS itself.
//!
//! The FST-level closure results are pre-sorted by FST state.  Since
//! `fst_state` occupies the high bits of the packed u64, the repacked entries
//! are nearly sorted, allowing Rust's pdqsort (`sort_unstable`) to verify
//! sortedness in O(n) instead of a full O(n log n) sort.  For BPE FSTs at
//! V=5000 this reduces the sort+dedup phase from ~550ms to ~100ms.
//!
//! **Remaining bottleneck**: the output of `compute_all_arcs` is O(V × |closure|)
//! packed entries (e.g. 43M at V=5000).  This materialization cost dominates
//! at large V and cannot be reduced without changing the DFA representation
//! (e.g. parameterized states that share a single FST closure across
//! `extra_sym` values).  Measured ~2x speedup at V≥5000 compared to the
//! baseline without this optimization.

use crate::fst::{compute_ip_universal_states, Fst, EPSILON};
// PowersetArena still used by lazy_precover.rs, decompose.rs; not used here.
// use crate::powerset::PowersetArena;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// NFA state packing
// ---------------------------------------------------------------------------

pub(crate) const NO_EXTRA: u16 = 0xFFFF;

#[inline]
pub(crate) fn pack_peekaboo(fst_state: u32, buf_len: u16, extra_sym: u16, truncated: bool) -> u64 {
    ((fst_state as u64) << 32)
        | ((buf_len as u64) << 17)
        | ((extra_sym as u64) << 1)
        | (truncated as u64)
}

#[inline]
pub(crate) fn unpack_peekaboo(packed: u64) -> (u32, u16, u16, bool) {
    let fst_state = (packed >> 32) as u32;
    let buf_len = ((packed >> 17) & 0x7FFF) as u16;
    let extra_sym = ((packed >> 1) & 0xFFFF) as u16;
    let truncated = (packed & 1) != 0;
    (fst_state, buf_len, extra_sym, truncated)
}

// ---------------------------------------------------------------------------
// FstBitset: compact bitset over FST states for fast closure operations
// ---------------------------------------------------------------------------

/// Fixed-capacity bitset over FST state IDs (u32).
///
/// For BPE FSTs with ~1000 states, this is ~128 bytes (16 u64 words).
/// Operations: insert O(1), contains O(1), union O(n_words), iter O(n_words).
/// Replaces FxHashSet<u32> in FST-level closure computations for ~2-3x speedup
/// on insert/contains and eliminates the collect+sort step.
#[derive(Clone)]
pub(crate) struct FstBitset {
    words: Vec<u64>,
}

impl FstBitset {
    /// Create an empty bitset capable of holding states in [0, capacity).
    #[inline]
    fn new(capacity: u32) -> Self {
        let n_words = ((capacity as usize) + 63) / 64;
        FstBitset {
            words: vec![0u64; n_words],
        }
    }

    /// Insert a state. Returns true if newly inserted.
    #[inline]
    fn insert(&mut self, state: u32) -> bool {
        let word = (state / 64) as usize;
        let bit = state % 64;
        let mask = 1u64 << bit;
        let was_set = self.words[word] & mask != 0;
        self.words[word] |= mask;
        !was_set
    }

    /// Test membership.
    #[inline]
    fn contains(&self, state: u32) -> bool {
        let word = (state / 64) as usize;
        let bit = state % 64;
        self.words[word] & (1u64 << bit) != 0
    }

    /// Iterate set bits in ascending order.
    fn iter(&self) -> FstBitsetIter<'_> {
        FstBitsetIter {
            words: &self.words,
            word_idx: 0,
            current: if self.words.is_empty() { 0 } else { self.words[0] },
        }
    }

    /// Collect set bits into a sorted Vec<u32>.
    fn to_sorted_vec(&self) -> Vec<u32> {
        let mut result = Vec::new();
        for (i, &w) in self.words.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let tz = bits.trailing_zeros();
                result.push(i as u32 * 64 + tz);
                bits &= bits - 1; // clear lowest set bit
            }
        }
        result
    }
}

pub(crate) struct FstBitsetIter<'a> {
    words: &'a [u64],
    word_idx: usize,
    current: u64,
}

impl<'a> Iterator for FstBitsetIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        loop {
            if self.current != 0 {
                let tz = self.current.trailing_zeros();
                self.current &= self.current - 1;
                return Some(self.word_idx as u32 * 64 + tz);
            }
            self.word_idx += 1;
            if self.word_idx >= self.words.len() {
                return None;
            }
            self.current = self.words[self.word_idx];
        }
    }
}

// ---------------------------------------------------------------------------
// PeekabooNFA
// ---------------------------------------------------------------------------

/// PeekabooPrecover NFA for a given step.
///
/// This mirrors Python's `PeekabooPrecover(fst, target[:step_n])` with K=1.
/// NFA state encoding is step-independent (same buffer → same packed u64).
pub(crate) struct PeekabooNFAMapped<'a> {
    pub(crate) fst: &'a Fst,
    pub(crate) full_target: &'a [u32],
    pub(crate) step_n: u16,
    pub(crate) sym_to_idx: &'a FxHashMap<u32, u16>,
}

impl<'a> PeekabooNFAMapped<'a> {
    pub(crate) fn new(
        fst: &'a Fst,
        full_target: &'a [u32],
        step_n: u16,
        sym_to_idx: &'a FxHashMap<u32, u16>,
    ) -> Self {
        PeekabooNFAMapped {
            fst,
            full_target,
            step_n,
            sym_to_idx,
        }
    }

    pub(crate) fn start_states(&self) -> Vec<u64> {
        self.fst
            .start_states
            .iter()
            .map(|&s| pack_peekaboo(s, 0, NO_EXTRA, false))
            .collect()
    }

    /// Is an NFA state final? Final when fst_state is final AND buffer has
    /// length > step_n (i.e., buffer = target[:step_n] + sym).
    ///
    /// For the buffer to be valid at this step, it must be prefix-compatible.
    /// A final state has buf_len == step_n + 1 and extra_sym != NO_EXTRA.
    /// Since buf_len-1 == step_n >= step_n, it's always prefix-compatible.
    #[inline]
    pub(crate) fn is_final(&self, packed: u64) -> bool {
        let (fst_state, buf_len, extra_sym, _truncated) = unpack_peekaboo(packed);
        self.fst.is_final[fst_state as usize]
            && buf_len == self.step_n + 1
            && extra_sym != NO_EXTRA
    }

    /// Is an NFA state "informative"?  An NFA element is kept in the
    /// epsilon-closed powerset state if it can produce further arcs,
    /// contributes to DFA finality, carries preimage metadata, or is
    /// truncated (needed for truncation metadata).
    #[inline]
    fn is_productive(&self, packed: u64) -> bool {
        let (fst_state, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);
        // Can produce more transitions
        self.fst.has_non_eps_input[fst_state as usize]
            // NFA-final: contributes to quotient/remainder
            || (self.fst.is_final[fst_state as usize]
                && buf_len == self.step_n + 1
                && extra_sym != NO_EXTRA)
            // Is-preimage: output buffer effectively matches target at a final state.
            // Canonical: buf_len == step_n, extra == NO_EXTRA.
            // Non-canonical: buf_len == step_n, extra == sym_to_idx[target[step_n-1]].
            || (self.fst.is_final[fst_state as usize]
                && buf_len == self.step_n
                && (extra_sym == NO_EXTRA
                    || (self.step_n > 0
                        && (self.step_n - 1) < self.full_target.len() as u16
                        && self.sym_to_idx
                            .get(&self.full_target[(self.step_n - 1) as usize])
                            .map_or(false, |&idx| extra_sym == idx))))
            // Truncated: carries truncation metadata
            || truncated
    }

    /// Compute the "effective" buffer position for this NFA state at the
    /// current step. This determines which branch of the arc logic to use.
    ///
    /// Returns (effective_n, effective_extra, is_valid):
    /// - effective_n: equivalent of `len(ys)` in the Python code
    /// - effective_extra: the extra symbol (or NO_EXTRA)
    /// - is_valid: whether the state is prefix-compatible at this step
    #[inline]
    pub(crate) fn effective_state(
        &self,
        buf_len: u16,
        extra_sym: u16,
        _truncated: bool,
    ) -> (u16, u16, bool) {
        if extra_sym == NO_EXTRA {
            // On-target: buf_len is the true buffer length.
            // In the Python NFA, n = len(ys) = buf_len.
            (buf_len, NO_EXTRA, true)
        } else {
            // Off-target: buffer = target[:buf_len-1] + sym.
            // In the Python NFA, n = len(ys) = buf_len.
            let prefix_len = buf_len - 1;
            if prefix_len >= self.step_n {
                // Buffer extends at or beyond step_n.
                // The Python NFA treats this as n = buf_len, ys[step_n] = extra_sym.
                (buf_len, extra_sym, true)
            } else {
                // prefix_len < step_n: check if extra_sym matches target[prefix_len].
                // If so, the buffer is effectively target[:buf_len] (on-target).
                if (prefix_len as usize) < self.full_target.len() {
                    if let Some(&expected_idx) = self.sym_to_idx.get(&self.full_target[prefix_len as usize]) {
                        if extra_sym == expected_idx {
                            // Re-interpret as on-target with buf_len.
                            (buf_len, NO_EXTRA, true)
                        } else {
                            // Incompatible: buffer diverges before step_n.
                            (buf_len, extra_sym, false)
                        }
                    } else {
                        (buf_len, extra_sym, false)
                    }
                } else {
                    (buf_len, extra_sym, false)
                }
            }
        }
    }

    /// Compute all arcs from an NFA state (including epsilon arcs).
    /// Mirrors Python's `PeekabooPrecover.arcs()`.
    pub(crate) fn arcs(&self, packed: u64) -> Vec<(u32, u64)> {
        let (i, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);
        let step_n = self.step_n;

        // Compute the effective state at this step.
        let (eff_n, eff_extra, is_valid) = self.effective_state(buf_len, extra_sym, truncated);
        if !is_valid {
            return Vec::new(); // Dead state at this step.
        }

        let mut result = Vec::new();

        // m = min(step_n, eff_n)
        // The Python code's branches:
        // if m >= N (target <= ys): step_n <= eff_n
        // else (ys < target): eff_n < step_n
        if eff_n >= step_n {
            // Buffer has reached or passed step_n.
            for arc in self.fst.arcs_from(i) {
                let x = arc.input;
                let y = arc.output;
                let j = arc.dest;

                if y == EPSILON || truncated {
                    // Epsilon output or already truncated: buffer unchanged.
                    result.push((x, pack_peekaboo(j, buf_len, extra_sym, truncated)));
                } else if eff_extra == NO_EXTRA && eff_n == step_n {
                    // On-target at exactly step_n: output y extends to step_n+1.
                    // was = ys + y, which is target[:step_n] + y.
                    // now = was[:step_n+K] = was[:step_n+1] = target[:step_n] + y.
                    // This creates an off-target state with extra = y.
                    if let Some(&y_idx) = self.sym_to_idx.get(&y) {
                        result.push((x, pack_peekaboo(j, step_n + 1, y_idx, false)));
                    }
                } else {
                    // Off-target or on-target with eff_n > step_n (which shouldn't happen
                    // normally, but could for re-interpreted states).
                    // In Python: was = ys + y, now = was[:step_n+K].
                    // Since eff_n >= step_n and we already have extra != NO_EXTRA or eff_n > step_n,
                    // the buffer is already at or past the max length.
                    // Truncate: mark as truncated.
                    result.push((x, pack_peekaboo(j, buf_len, extra_sym, true)));
                }
            }
        } else {
            // Buffer hasn't reached step_n yet (eff_n < step_n).
            // eff_extra must be NO_EXTRA (since off-target states with prefix_len < step_n
            // that are valid must have been re-interpreted as on-target).
            if truncated {
                // Truncated state from a previous step_n: the FST has consumed
                // outputs beyond the truncation point, so we can't continue
                // growing the buffer. Treat as dead.
                return Vec::new();
            }
            assert!(eff_extra == NO_EXTRA);
            // Epsilon-output arcs: buffer unchanged
            for arc in self.fst.arcs_by_output(i, EPSILON) {
                result.push((arc.input, pack_peekaboo(arc.dest, eff_n, NO_EXTRA, false)));
            }
            // Target-matching output arcs: buffer advances
            let target_sym = self.full_target[eff_n as usize];
            for arc in self.fst.arcs_by_output(i, target_sym) {
                result.push((arc.input, pack_peekaboo(arc.dest, eff_n + 1, NO_EXTRA, false)));
            }
        }

        result
    }

    /// Epsilon-input successors from a packed NFA state.
    /// Uses the FST's eps_input_arcs side table instead of full arcs() + filter.
    fn eps_arcs(&self, packed: u64) -> Vec<u64> {
        let (i, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);
        let step_n = self.step_n;

        let (eff_n, eff_extra, is_valid) = self.effective_state(buf_len, extra_sym, truncated);
        if !is_valid {
            return Vec::new();
        }

        let mut result = Vec::new();

        if eff_n >= step_n {
            // Boundary phase: all epsilon-input arcs
            for ea in self.fst.eps_input_arcs(i) {
                let y = ea.output;
                let j = ea.dest;
                if y == EPSILON || truncated {
                    result.push(pack_peekaboo(j, buf_len, extra_sym, truncated));
                } else if eff_extra == NO_EXTRA && eff_n == step_n {
                    if let Some(&y_idx) = self.sym_to_idx.get(&y) {
                        result.push(pack_peekaboo(j, step_n + 1, y_idx, false));
                    }
                } else {
                    result.push(pack_peekaboo(j, buf_len, extra_sym, true));
                }
            }
        } else {
            // Growing phase
            if truncated {
                return Vec::new();
            }
            assert!(eff_extra == NO_EXTRA);
            // Epsilon-output arcs: buffer unchanged
            for ea in self.fst.eps_input_arcs_by_output(i, EPSILON) {
                result.push(pack_peekaboo(ea.dest, eff_n, NO_EXTRA, false));
            }
            // Target-matching output arcs: buffer advances
            let target_sym = self.full_target[eff_n as usize];
            for ea in self.fst.eps_input_arcs_by_output(i, target_sym) {
                result.push(pack_peekaboo(ea.dest, eff_n + 1, NO_EXTRA, false));
            }
        }

        result
    }

    /// Epsilon-closure of a single NFA state (cached).
    /// Cache stores (closure, max_buf_len) for O(1) eviction on prefix extension.
    pub(crate) fn eps_closure_single(&self, state: u64, cache: &mut FxHashMap<u64, (Vec<u64>, u16)>) -> Vec<u64> {
        if let Some(cached) = cache.get(&state) {
            return cached.0.clone();
        }

        let mut all_reachable = Vec::new();
        let mut worklist: VecDeque<u64> = VecDeque::new();
        let mut seen = FxHashSet::default();

        all_reachable.push(state);
        seen.insert(state);
        worklist.push_back(state);

        while let Some(s) = worklist.pop_front() {
            for dest in self.eps_arcs(s) {
                if seen.insert(dest) {
                    all_reachable.push(dest);
                    worklist.push_back(dest);
                }
            }
        }

        let mut result: Vec<u64> = all_reachable
            .into_iter()
            .filter(|&s| self.is_productive(s))
            .collect();
        result.sort_unstable();
        result.dedup();

        let max_bl = result.iter()
            .map(|&s| ((s >> 17) & 0x7FFF) as u16)
            .max()
            .unwrap_or(0);
        cache.insert(state, (result.clone(), max_bl));
        result
    }

    pub(crate) fn eps_closure_set(&self, states: &[u64], cache: &mut FxHashMap<u64, (Vec<u64>, u16)>) -> Vec<u64> {
        let mut result = Vec::new();
        for &s in states {
            let closure = self.eps_closure_single(s, cache);
            result.extend_from_slice(&closure);
        }
        result.sort_unstable();
        result.dedup();
        result
    }

    /// Batch-compute all non-epsilon arcs from an epsilon-closed powerset state.
    ///
    /// For off-target NFA destinations (extra_sym != NO_EXTRA, buf_len > step_n),
    /// the epsilon closure depends only on the FST state — the buffer parameters
    /// (buf_len, extra_sym) affect only the packed representation, not which FST
    /// states are reachable.  A local FST-level closure cache collapses ~|V|
    /// redundant packed-state BFS computations into one per unique FST state.
    pub(crate) fn compute_all_arcs(
        &self,
        states: &[u64],
        cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
    ) -> Vec<(u32, Vec<u64>)> {
        let mut fst_full_cache: FxHashMap<u32, (Vec<u32>, Vec<u32>)> = FxHashMap::default();
        let mut fst_trunc_cache: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        self.compute_all_arcs_cached(states, cache, &mut fst_full_cache, &mut fst_trunc_cache)
    }

    /// Like compute_all_arcs, but with persistent FST-level closure caches.
    ///
    /// When processing multiple DFA states in a BFS, passing the same caches
    /// across calls avoids redundant FST-level closure computations. Typical
    /// BPE FSTs have ~100-1000 unique FST states, so the cache fills quickly
    /// and subsequent calls get near-100% hit rates.
    pub(crate) fn compute_all_arcs_cached(
        &self,
        states: &[u64],
        cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        fst_full_cache: &mut FxHashMap<u32, (Vec<u32>, Vec<u32>)>,
        fst_trunc_cache: &mut FxHashMap<u32, Vec<u32>>,
    ) -> Vec<(u32, Vec<u64>)> {
        let mut by_symbol: FxHashMap<u32, Vec<u64>> = FxHashMap::default();
        let step_n = self.step_n;

        // Deferred off-target groups: (input_sym, dest_fst, is_trunc) → unique (bl, es) params.
        // Instead of immediately expanding V × |closure| packed states, we collect
        // unique parameter combinations first, then expand once per unique param.
        // Key: (input_symbol, dest_fst, is_trunc_source)
        //   is_trunc_source=false → dest came from non-truncated NFA element
        //   is_trunc_source=true  → dest came from truncated NFA element
        let mut deferred_full: FxHashMap<(u32, u32), Vec<(u16, u16)>> = FxHashMap::default();
        let mut deferred_trunc: FxHashMap<(u32, u32), Vec<(u16, u16)>> = FxHashMap::default();

        for &packed in states {
            for (x, dest) in self.arcs(packed) {
                if x != EPSILON {
                    let (dest_fst, dest_bl, dest_es, dest_trunc) = unpack_peekaboo(dest);

                    if dest_es != NO_EXTRA && dest_bl > step_n {
                        // Populate FST closure caches eagerly
                        if !dest_trunc {
                            fst_full_cache
                                .entry(dest_fst)
                                .or_insert_with(|| self.fst_full_closure(dest_fst));
                            deferred_full
                                .entry((x, dest_fst))
                                .or_default()
                                .push((dest_bl, dest_es));
                        } else {
                            fst_trunc_cache
                                .entry(dest_fst)
                                .or_insert_with(|| self.fst_trunc_closure(dest_fst));
                            deferred_trunc
                                .entry((x, dest_fst))
                                .or_default()
                                .push((dest_bl, dest_es));
                        }
                    } else {
                        // Standard path (on-target or small buf_len)
                        let closure = self.eps_closure_single(dest, cache);
                        let bucket = by_symbol.entry(x).or_default();
                        bucket.extend_from_slice(&closure);
                    }
                }
            }
        }

        // Expand deferred non-truncated off-target groups.
        // For each (input_sym, dest_fst), we have unique (bl, es) params and a
        // shared FST closure. Expand: |unique_params| × |closure| instead of
        // |total_hits| × |closure|.
        for ((x, dest_fst), mut params) in deferred_full {
            params.sort_unstable();
            params.dedup();
            let entry = fst_full_cache.get(&dest_fst).unwrap();
            let bucket = by_symbol.entry(x).or_default();
            for &(bl, es) in &params {
                for &r in entry.0.iter() {
                    bucket.push(pack_peekaboo(r, bl, es, false));
                }
                for &t in entry.1.iter() {
                    bucket.push(pack_peekaboo(t, bl, es, true));
                }
            }
        }

        // Expand deferred truncated off-target groups.
        for ((x, dest_fst), mut params) in deferred_trunc {
            params.sort_unstable();
            params.dedup();
            let entry = fst_trunc_cache.get(&dest_fst).unwrap();
            let bucket = by_symbol.entry(x).or_default();
            for &(bl, es) in &params {
                for &t in entry.iter() {
                    bucket.push(pack_peekaboo(t, bl, es, true));
                }
            }
        }

        let mut result: Vec<(u32, Vec<u64>)> = Vec::with_capacity(by_symbol.len());
        for (sym, mut v) in by_symbol {
            v.sort_unstable();
            v.dedup();
            result.push((sym, v));
        }
        result
    }

    /// FST-level epsilon closure for a non-truncated off-target starting state.
    ///
    /// Phase 1: follow eps-input arcs with eps-output (stays non-truncated).
    /// Phase 2: from non-eps-output destinations, follow ALL eps-input arcs (truncated).
    ///
    /// Returns (non_trunc_productive_fst_states, trunc_fst_states).
    /// Non-truncated states are filtered by productivity (has_non_eps_input || is_final).
    /// All truncated states are productive (truncated flag => productive).
    fn fst_full_closure(&self, fst_state: u32) -> (Vec<u32>, Vec<u32>) {
        let n = self.fst.num_states;
        // Phase 1: non-truncated reachable via eps-input/eps-output arcs
        let mut non_trunc = FstBitset::new(n);
        let mut worklist: VecDeque<u32> = VecDeque::new();
        non_trunc.insert(fst_state);
        worklist.push_back(fst_state);

        let mut trunc_seeds: Vec<u32> = Vec::new();

        while let Some(s) = worklist.pop_front() {
            for ea in self.fst.eps_input_arcs(s) {
                if ea.output == EPSILON {
                    if non_trunc.insert(ea.dest) {
                        worklist.push_back(ea.dest);
                    }
                } else {
                    trunc_seeds.push(ea.dest);
                }
            }
        }

        // Phase 2: truncated reachable from trunc_seeds via ALL eps-input arcs
        let mut trunc = FstBitset::new(n);
        for &f in &trunc_seeds {
            if trunc.insert(f) {
                worklist.push_back(f);
            }
        }
        while let Some(s) = worklist.pop_front() {
            for ea in self.fst.eps_input_arcs(s) {
                if trunc.insert(ea.dest) {
                    worklist.push_back(ea.dest);
                }
            }
        }

        // Filter non-truncated by productivity: has_non_eps_input || is_final
        // Bitset iteration yields sorted order automatically.
        let non_trunc_productive: Vec<u32> = non_trunc
            .iter()
            .filter(|&r| {
                self.fst.has_non_eps_input[r as usize]
                    || self.fst.is_final[r as usize]
            })
            .collect();

        // All truncated states are productive; bitset iter yields sorted order.
        let trunc_result: Vec<u32> = trunc.to_sorted_vec();

        (non_trunc_productive, trunc_result)
    }

    /// FST-level epsilon closure for a truncated starting state.
    /// Follows ALL eps-input arcs (any output). All reachable states are productive.
    fn fst_trunc_closure(&self, fst_state: u32) -> Vec<u32> {
        let n = self.fst.num_states;
        let mut visited = FstBitset::new(n);
        let mut worklist: VecDeque<u32> = VecDeque::new();
        visited.insert(fst_state);
        worklist.push_back(fst_state);
        while let Some(s) = worklist.pop_front() {
            for ea in self.fst.eps_input_arcs(s) {
                if visited.insert(ea.dest) {
                    worklist.push_back(ea.dest);
                }
            }
        }
        // Bitset iteration yields sorted order automatically.
        visited.to_sorted_vec()
    }

    /// Batch-compute all non-epsilon arcs from a factored NFA state set.
    ///
    /// Returns factored successor sets instead of flat Vec<u64>.
    /// Core elements are processed via eps_closure_single (same as compute_all_arcs_cached).
    /// Group elements are processed at the FST level, extending deferred maps with
    /// the group's params rather than materializing the cartesian product.
    pub(crate) fn compute_all_arcs_factored(
        &self,
        factored: &FactoredNfaSet,
        cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        fst_full_cache: &mut FxHashMap<u32, (Vec<u32>, Vec<u32>)>,
        fst_trunc_cache: &mut FxHashMap<u32, Vec<u32>>,
    ) -> Vec<(u32, FactoredNfaSet)> {
        let mut by_symbol_core: FxHashMap<u32, Vec<u64>> = FxHashMap::default();
        let step_n = self.step_n;

        // Deferred off-target groups: (input_sym, dest_fst) → (bl, es) params.
        let mut deferred_full: FxHashMap<(u32, u32), Vec<(u16, u16)>> = FxHashMap::default();
        let mut deferred_trunc: FxHashMap<(u32, u32), Vec<(u16, u16)>> = FxHashMap::default();

        // Phase 1: Process core elements (same as compute_all_arcs_cached)
        for &packed in &factored.core {
            for (x, dest) in self.arcs(packed) {
                if x != EPSILON {
                    let (dest_fst, dest_bl, dest_es, dest_trunc) = unpack_peekaboo(dest);

                    if dest_es != NO_EXTRA && dest_bl > step_n {
                        if !dest_trunc {
                            fst_full_cache
                                .entry(dest_fst)
                                .or_insert_with(|| self.fst_full_closure(dest_fst));
                            deferred_full
                                .entry((x, dest_fst))
                                .or_default()
                                .push((dest_bl, dest_es));
                        } else {
                            fst_trunc_cache
                                .entry(dest_fst)
                                .or_insert_with(|| self.fst_trunc_closure(dest_fst));
                            deferred_trunc
                                .entry((x, dest_fst))
                                .or_default()
                                .push((dest_bl, dest_es));
                        }
                    } else {
                        let closure = self.eps_closure_single(dest, cache);
                        let bucket = by_symbol_core.entry(x).or_default();
                        bucket.extend_from_slice(&closure);
                    }
                }
            }
        }

        // Phase 2: Process off-target groups at FST level
        for group in &factored.groups {
            // Collect unique (x, dest_fst) → (goes_to_full, goes_to_trunc)
            let mut group_deferred: FxHashMap<(u32, u32), (bool, bool)> = FxHashMap::default();

            for &f in &group.fst_nontrunc {
                for arc in self.fst.arcs_from(f) {
                    if arc.input == EPSILON { continue; }
                    let entry = group_deferred.entry((arc.input, arc.dest)).or_insert((false, false));
                    if arc.output == EPSILON {
                        entry.0 = true; // non-truncated dest → deferred_full
                    } else {
                        entry.1 = true; // truncated dest → deferred_trunc
                    }
                }
            }

            for &f in &group.fst_trunc {
                for arc in self.fst.arcs_from(f) {
                    if arc.input == EPSILON { continue; }
                    group_deferred.entry((arc.input, arc.dest)).or_insert((false, false)).1 = true;
                }
            }

            // Extend deferred maps once per unique (x, dest_fst) with this group's params
            for ((x, dest_fst), (goes_full, goes_trunc)) in group_deferred {
                if goes_full {
                    fst_full_cache
                        .entry(dest_fst)
                        .or_insert_with(|| self.fst_full_closure(dest_fst));
                    deferred_full
                        .entry((x, dest_fst))
                        .or_default()
                        .extend_from_slice(&group.params);
                }
                if goes_trunc {
                    fst_trunc_cache
                        .entry(dest_fst)
                        .or_insert_with(|| self.fst_trunc_closure(dest_fst));
                    deferred_trunc
                        .entry((x, dest_fst))
                        .or_default()
                        .extend_from_slice(&group.params);
                }
            }
        }

        // Phase 3: Assemble factored successors
        // Group deferred entries by input symbol
        let mut full_by_sym: FxHashMap<u32, Vec<(u32, Vec<(u16, u16)>)>> = FxHashMap::default();
        for ((x, dest_fst), mut params) in deferred_full {
            params.sort_unstable();
            params.dedup();
            full_by_sym.entry(x).or_default().push((dest_fst, params));
        }

        let mut trunc_by_sym: FxHashMap<u32, Vec<(u32, Vec<(u16, u16)>)>> = FxHashMap::default();
        for ((x, dest_fst), mut params) in deferred_trunc {
            params.sort_unstable();
            params.dedup();
            trunc_by_sym.entry(x).or_default().push((dest_fst, params));
        }

        // Collect all input symbols
        let mut all_symbols: FxHashSet<u32> = FxHashSet::default();
        for &sym in by_symbol_core.keys() { all_symbols.insert(sym); }
        for &sym in full_by_sym.keys() { all_symbols.insert(sym); }
        for &sym in trunc_by_sym.keys() { all_symbols.insert(sym); }

        let mut result: Vec<(u32, FactoredNfaSet)> = Vec::with_capacity(all_symbols.len());
        for sym in all_symbols {
            let mut core = by_symbol_core.remove(&sym).unwrap_or_default();
            core.sort_unstable();
            core.dedup();

            let mut groups = Vec::new();

            if let Some(entries) = full_by_sym.get(&sym) {
                for (dest_fst, params) in entries {
                    let (nontrunc, trunc) = fst_full_cache.get(dest_fst).unwrap();
                    groups.push(OffTargetGroup {
                        fst_nontrunc: nontrunc.clone(),
                        fst_trunc: trunc.clone(),
                        params: params.clone(),
                    });
                }
            }

            if let Some(entries) = trunc_by_sym.get(&sym) {
                for (dest_fst, params) in entries {
                    let trunc = fst_trunc_cache.get(dest_fst).unwrap();
                    groups.push(OffTargetGroup {
                        fst_nontrunc: vec![],
                        fst_trunc: trunc.clone(),
                        params: params.clone(),
                    });
                }
            }

            groups.sort(); // canonical order for fingerprint/equality
            result.push((sym, FactoredNfaSet { core, groups }));
        }
        result
    }

    /// Compute the successor factored NFA set for a single input symbol.
    ///
    /// This is the single-symbol version of `compute_all_arcs_factored`:
    /// instead of iterating all input symbols and building successors for each,
    /// it only processes transitions labeled with the given symbol `x`.
    ///
    /// Cost: O(|core| × arcs_per_state + |groups| × fst_arcs) instead of
    /// O(|V| × |closure|) for the full expansion.  For BPE FSTs with V=50K,
    /// this reduces per-symbol work from ~5M operations to ~400.
    pub(crate) fn compute_single_arc_factored(
        &self,
        factored: &FactoredNfaSet,
        x: u32,
        cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        fst_full_cache: &mut FxHashMap<u32, (Vec<u32>, Vec<u32>)>,
        fst_trunc_cache: &mut FxHashMap<u32, Vec<u32>>,
    ) -> Option<FactoredNfaSet> {
        let step_n = self.step_n;
        let mut core_dests: Vec<u64> = Vec::new();
        let mut deferred_full: FxHashMap<u32, Vec<(u16, u16)>> = FxHashMap::default();
        let mut deferred_trunc: FxHashMap<u32, Vec<(u16, u16)>> = FxHashMap::default();

        // Phase 1: Process core elements — filter to arcs with input == x
        for &packed in &factored.core {
            for (inp, dest) in self.arcs(packed) {
                if inp != x { continue; }
                let (dest_fst, dest_bl, dest_es, dest_trunc) = unpack_peekaboo(dest);
                if dest_es != NO_EXTRA && dest_bl > step_n {
                    if !dest_trunc {
                        fst_full_cache
                            .entry(dest_fst)
                            .or_insert_with(|| self.fst_full_closure(dest_fst));
                        deferred_full.entry(dest_fst).or_default().push((dest_bl, dest_es));
                    } else {
                        fst_trunc_cache
                            .entry(dest_fst)
                            .or_insert_with(|| self.fst_trunc_closure(dest_fst));
                        deferred_trunc.entry(dest_fst).or_default().push((dest_bl, dest_es));
                    }
                } else {
                    let closure = self.eps_closure_single(dest, cache);
                    core_dests.extend_from_slice(&closure);
                }
            }
        }

        // Phase 2: Process off-target groups — only arcs with input == x
        for group in &factored.groups {
            for &f in &group.fst_nontrunc {
                for arc in self.fst.arcs_from(f) {
                    if arc.input != x { continue; }
                    if arc.output == EPSILON {
                        fst_full_cache
                            .entry(arc.dest)
                            .or_insert_with(|| self.fst_full_closure(arc.dest));
                        deferred_full.entry(arc.dest).or_default()
                            .extend_from_slice(&group.params);
                    } else {
                        fst_trunc_cache
                            .entry(arc.dest)
                            .or_insert_with(|| self.fst_trunc_closure(arc.dest));
                        deferred_trunc.entry(arc.dest).or_default()
                            .extend_from_slice(&group.params);
                    }
                }
            }
            for &f in &group.fst_trunc {
                for arc in self.fst.arcs_from(f) {
                    if arc.input != x { continue; }
                    fst_trunc_cache
                        .entry(arc.dest)
                        .or_insert_with(|| self.fst_trunc_closure(arc.dest));
                    deferred_trunc.entry(arc.dest).or_default()
                        .extend_from_slice(&group.params);
                }
            }
        }

        // Check if anything was produced
        if core_dests.is_empty() && deferred_full.is_empty() && deferred_trunc.is_empty() {
            return None;
        }

        core_dests.sort_unstable();
        core_dests.dedup();

        // Phase 3: Build factored successor set
        let mut groups = Vec::new();
        for (dest_fst, mut params) in deferred_full {
            params.sort_unstable();
            params.dedup();
            let (nontrunc, trunc) = fst_full_cache.get(&dest_fst).unwrap();
            groups.push(OffTargetGroup {
                fst_nontrunc: nontrunc.clone(),
                fst_trunc: trunc.clone(),
                params,
            });
        }
        for (dest_fst, mut params) in deferred_trunc {
            params.sort_unstable();
            params.dedup();
            let trunc = fst_trunc_cache.get(&dest_fst).unwrap();
            groups.push(OffTargetGroup {
                fst_nontrunc: vec![],
                fst_trunc: trunc.clone(),
                params,
            });
        }
        groups.sort();

        Some(FactoredNfaSet { core: core_dests, groups })
    }
}

// ---------------------------------------------------------------------------
// SymbolIndex: O(1) per-symbol projection over NFA state sets
// ---------------------------------------------------------------------------

/// Pre-built index over an NFA state set for efficient per-symbol projection.
///
/// Partitions NFA elements by `extra_sym`, enabling O(|on_target| + |matching|)
/// projection instead of O(|total_set|) per symbol.  Built once per DFA state,
/// amortized across ~V universality checks.
///
/// At V=5000 with |closure|=100, the full NFA set has ~500K elements.
/// Without the index, each `project_and_refine` call scans all 500K elements.
/// With V symbols, the total cost is O(V × 500K) = O(V² × |closure|).
/// The index reduces this to O(V × |closure|) total (building the index once
/// is O(V × |closure|), then each projection is O(|on_target| + |matching|)).
pub(crate) struct SymbolIndex {
    /// Elements with extra_sym == NO_EXTRA
    pub on_target: Vec<u64>,
    /// Elements indexed by extra_sym (!= NO_EXTRA), each sub-vec in original order
    pub by_extra: FxHashMap<u16, Vec<u64>>,
}

impl SymbolIndex {
    /// Build an index over the given NFA state set.
    pub fn new(nfa_set: &[u64]) -> Self {
        let mut on_target = Vec::new();
        let mut by_extra: FxHashMap<u16, Vec<u64>> = FxHashMap::default();

        for &packed in nfa_set {
            let extra_sym = ((packed >> 1) & 0xFFFF) as u16;
            if extra_sym == NO_EXTRA {
                on_target.push(packed);
            } else {
                by_extra.entry(extra_sym).or_default().push(packed);
            }
        }

        SymbolIndex { on_target, by_extra }
    }

    /// Relevant output symbols at the given step: those with at least one
    /// element having `buf_len > step_n` (frontier elements).
    ///
    /// For off-target elements with `buf_len > step_n`, effective_state
    /// always returns `(buf_len, extra_sym, true)`, so relevance reduces
    /// to the simple `buf_len > step_n` check.
    pub fn relevant_symbols(&self, step_n: u16) -> Vec<u16> {
        let mut result = Vec::with_capacity(self.by_extra.len());
        for (&extra_sym, elements) in &self.by_extra {
            for &packed in elements {
                let buf_len = ((packed >> 17) & 0x7FFF) as u16;
                if buf_len > step_n {
                    result.push(extra_sym);
                    break;
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Factored NFA state sets: O(|closure| + |V|) representation
// ---------------------------------------------------------------------------

/// A group of off-target NFA elements sharing the same FST closure.
/// Represents: {pack(f, bl, es, false) | f in fst_nontrunc, (bl,es) in params}
///           ∪ {pack(f, bl, es, true)  | f in fst_trunc, (bl,es) in params}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct OffTargetGroup {
    pub fst_nontrunc: Vec<u32>,   // sorted FST states (non-truncated)
    pub fst_trunc: Vec<u32>,      // sorted FST states (truncated)
    pub params: Vec<(u16, u16)>,  // sorted (buf_len, extra_sym)
}

/// Factored NFA state set. Boundary states store off-target elements
/// as (closure, params) groups instead of the full cartesian product.
#[derive(Clone, Debug)]
pub(crate) struct FactoredNfaSet {
    /// On-target elements + small off-target below frontier.
    /// Sorted, deduplicated packed u64.
    pub core: Vec<u64>,
    /// Off-target groups sharing FST closures, in canonical order.
    pub groups: Vec<OffTargetGroup>,
}

impl FactoredNfaSet {
    /// True if no groups (flat set, e.g. from sub-BFS projection).
    pub fn is_simple(&self) -> bool {
        self.groups.is_empty()
    }

    /// Max buf_len across core and groups.
    pub fn max_bufpos(&self) -> u16 {
        let core_max = self.core.iter()
            .map(|&e| ((e >> 17) & 0x7FFF) as u16)
            .max()
            .unwrap_or(0);
        let group_max = self.groups.iter()
            .flat_map(|g| g.params.iter())
            .map(|&(bl, _)| bl)
            .max()
            .unwrap_or(0);
        core_max.max(group_max)
    }

    /// Check if any NFA element is final.
    pub fn any_final(&self, nfa: &PeekabooNFAMapped) -> bool {
        for &packed in &self.core {
            if nfa.is_final(packed) {
                return true;
            }
        }
        let target_bl = nfa.step_n + 1;
        for group in &self.groups {
            let has_matching_buf = group.params.iter().any(|&(bl, _)| bl == target_bl);
            if has_matching_buf {
                if group.fst_nontrunc.iter().any(|&f| nfa.fst.is_final[f as usize])
                    || group.fst_trunc.iter().any(|&f| nfa.fst.is_final[f as usize])
                {
                    return true;
                }
            }
        }
        false
    }

    /// Relevant output symbols: those with buf_len > step_n.
    pub fn relevant_symbols(&self, step_n: u16) -> Vec<u16> {
        let mut seen = FxHashSet::default();
        let mut result = Vec::new();
        for &packed in &self.core {
            let extra_sym = ((packed >> 1) & 0xFFFF) as u16;
            let buf_len = ((packed >> 17) & 0x7FFF) as u16;
            if extra_sym != NO_EXTRA && buf_len > step_n && seen.insert(extra_sym) {
                result.push(extra_sym);
            }
        }
        for group in &self.groups {
            for &(bl, es) in &group.params {
                if bl > step_n && seen.insert(es) {
                    result.push(es);
                }
            }
        }
        result
    }

    /// Hash for arena interning.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.core.hash(&mut hasher);
        self.groups.len().hash(&mut hasher);
        for g in &self.groups {
            g.fst_nontrunc.hash(&mut hasher);
            g.fst_trunc.hash(&mut hasher);
            g.params.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Canonical equality.
    pub fn eq_factored(&self, other: &Self) -> bool {
        self.core == other.core && self.groups == other.groups
    }

    /// Project to a single output symbol, returning a flat Vec<u64>.
    /// Replaces project_and_refine_indexed for factored sets.
    pub fn project_for_symbol(&self, y_idx: u16, step_n: u16) -> Vec<u64> {
        let target_len = step_n + 1;
        let mut projected = Vec::new();
        for &packed in &self.core {
            let (fst_state, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);
            if extra_sym == NO_EXTRA {
                let clipped = buf_len.min(target_len);
                projected.push(pack_peekaboo(fst_state, clipped, NO_EXTRA, truncated));
            } else if extra_sym == y_idx {
                let clipped = buf_len.min(target_len);
                projected.push(pack_peekaboo(fst_state, clipped, y_idx, truncated));
            }
        }
        for group in &self.groups {
            // Only include group elements that are truly off-target boundary
            // (bl > step_n). Stale params with bl <= step_n are handled by
            // normalize_for_step moving them to core before the BFS.
            let has_match = group.params.iter().any(|&(bl, es)| bl > step_n && es == y_idx);
            if has_match {
                for &f in &group.fst_nontrunc {
                    projected.push(pack_peekaboo(f, target_len, y_idx, false));
                }
                for &f in &group.fst_trunc {
                    projected.push(pack_peekaboo(f, target_len, y_idx, true));
                }
            }
        }
        projected.sort_unstable();
        projected.dedup();
        projected
    }

    /// Check if projection for y_idx contains any final NFA element.
    pub fn is_projected_final(&self, y_idx: u16, fst: &Fst, step_n: u16) -> bool {
        for &packed in &self.core {
            let fst_state = (packed >> 32) as u32;
            let buf_len = ((packed >> 17) & 0x7FFF) as u16;
            let extra_sym = ((packed >> 1) & 0xFFFF) as u16;
            if extra_sym == y_idx && fst.is_final[fst_state as usize] && buf_len == step_n + 1 {
                return true;
            }
        }
        for group in &self.groups {
            let has_match = group.params.iter().any(|&(bl, es)| bl == step_n + 1 && es == y_idx);
            if has_match {
                if group.fst_nontrunc.iter().any(|&f| fst.is_final[f as usize])
                    || group.fst_trunc.iter().any(|&f| fst.is_final[f as usize])
                {
                    return true;
                }
            }
        }
        false
    }

    /// Check if any element has the truncated flag set.
    pub fn has_truncated(&self) -> bool {
        for &packed in &self.core {
            if (packed & 1) != 0 {
                return true;
            }
        }
        for group in &self.groups {
            if !group.fst_trunc.is_empty() {
                return true;
            }
        }
        false
    }

    /// Check if any element is a preimage element.
    ///
    /// After normalize_for_step(), only core can contain preimage elements.
    /// But when called on un-normalized arena sets (e.g. from DirtyPeekaboo's
    /// compute_preimage_stops), groups may have stale params with bl == step_n
    /// that are also preimage candidates.
    pub fn is_preimage(&self, fst: &Fst, step_n: u16, target: &[u32], sym_to_idx: &FxHashMap<u32, u16>) -> bool {
        // Pre-compute the non-canonical extra_sym that also matches preimage.
        let non_canonical_extra: Option<u16> = if step_n > 0 && (step_n - 1) < target.len() as u16 {
            sym_to_idx.get(&target[(step_n - 1) as usize]).copied()
        } else {
            None
        };

        // Check core elements
        for &packed in &self.core {
            let (fst_state, buf_len, extra_sym, _truncated) = unpack_peekaboo(packed);
            if buf_len == step_n && fst.is_final[fst_state as usize] {
                if extra_sym == NO_EXTRA {
                    return true;
                }
                if let Some(expected) = non_canonical_extra {
                    if extra_sym == expected {
                        return true;
                    }
                }
            }
        }

        // Check groups: stale params with bl == step_n may be preimage candidates.
        // Group elements always have extra_sym != NO_EXTRA, so only the
        // non-canonical match applies.
        if let Some(expected) = non_canonical_extra {
            for group in &self.groups {
                let has_matching_param = group.params.iter().any(|&(bl, es)| bl == step_n && es == expected);
                if has_matching_param {
                    let has_final_fst = group.fst_nontrunc.iter().any(|&f| fst.is_final[f as usize])
                        || group.fst_trunc.iter().any(|&f| fst.is_final[f as usize]);
                    if has_final_fst {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Iterate core elements unpacked (for diagnostic/py.rs).
    pub fn iter_all_unpacked(&self) -> Vec<(u32, u16, u16, bool)> {
        let mut result: Vec<(u32, u16, u16, bool)> = Vec::new();
        for &packed in &self.core {
            result.push(unpack_peekaboo(packed));
        }
        for group in &self.groups {
            for &(bl, es) in &group.params {
                for &f in &group.fst_nontrunc {
                    result.push((f, bl, es, false));
                }
                for &f in &group.fst_trunc {
                    result.push((f, bl, es, true));
                }
            }
        }
        result
    }

    /// Flatten all elements (core + groups) into packed u64s.
    pub fn flatten(&self) -> Vec<u64> {
        let mut result = self.core.clone();
        for group in &self.groups {
            for &(bl, es) in &group.params {
                for &f in &group.fst_nontrunc {
                    result.push(pack_peekaboo(f, bl, es, false));
                }
                for &f in &group.fst_trunc {
                    result.push(pack_peekaboo(f, bl, es, true));
                }
            }
        }
        result
    }

    /// Collect all extra_sym values from truncated elements with prefix_len >= step_n.
    pub fn trunc_output_syms(&self, step_n: u16) -> Vec<u16> {
        let mut seen = FxHashSet::default();
        let mut result = Vec::new();
        for &packed in &self.core {
            let (_fst_state, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);
            if truncated && extra_sym != NO_EXTRA {
                let prefix_len = buf_len - 1;
                if prefix_len >= step_n && seen.insert(extra_sym) {
                    result.push(extra_sym);
                }
            }
        }
        for group in &self.groups {
            if !group.fst_trunc.is_empty() {
                for &(bl, es) in &group.params {
                    // Only include params that are truly off-target boundary
                    let prefix_len = bl - 1;
                    if prefix_len >= step_n && seen.insert(es) {
                        result.push(es);
                    }
                }
            }
        }
        result
    }

    /// Collect all extra_sym values with buf_len == step_n + 1 and final FST state.
    pub fn final_symbols(&self, fst: &Fst, step_n: u16) -> Vec<u16> {
        let mut seen = FxHashSet::default();
        let mut result = Vec::new();
        let target_bl = step_n + 1;
        for &packed in &self.core {
            let fst_state = (packed >> 32) as u32;
            let buf_len = ((packed >> 17) & 0x7FFF) as u16;
            let extra_sym = ((packed >> 1) & 0xFFFF) as u16;
            if extra_sym != NO_EXTRA && buf_len == target_bl
                && fst.is_final[fst_state as usize] && seen.insert(extra_sym)
            {
                result.push(extra_sym);
            }
        }
        for group in &self.groups {
            let has_matching_buf = group.params.iter().any(|&(bl, _)| bl == target_bl);
            if has_matching_buf {
                let has_final_fst = group.fst_nontrunc.iter().any(|&f| fst.is_final[f as usize])
                    || group.fst_trunc.iter().any(|&f| fst.is_final[f as usize]);
                if has_final_fst {
                    for &(bl, es) in &group.params {
                        if bl == target_bl && seen.insert(es) {
                            result.push(es);
                        }
                    }
                }
            }
        }
        result
    }
}

impl FactoredNfaSet {
    /// Move stale group params (bl <= step_n) back to core elements.
    ///
    /// Groups assume all params have bl > step_n (off-target boundary).
    /// When step_n increases (dirty-state incremental), some params become
    /// stale. These need NFA-level re-interpretation (via effective_state),
    /// which only happens for core elements in compute_all_arcs_factored.
    pub fn normalize_for_step(&mut self, step_n: u16) {
        let mut new_groups = Vec::with_capacity(self.groups.len());
        for group in self.groups.drain(..) {
            let mut fresh_params = Vec::new();
            let mut stale_params = Vec::new();
            for (bl, es) in group.params {
                if bl > step_n {
                    fresh_params.push((bl, es));
                } else {
                    stale_params.push((bl, es));
                }
            }
            // Materialize stale params back into core
            for (bl, es) in stale_params {
                for &f in &group.fst_nontrunc {
                    self.core.push(pack_peekaboo(f, bl, es, false));
                }
                for &f in &group.fst_trunc {
                    self.core.push(pack_peekaboo(f, bl, es, true));
                }
            }
            // Keep group only if it has fresh params
            if !fresh_params.is_empty() {
                new_groups.push(OffTargetGroup {
                    fst_nontrunc: group.fst_nontrunc,
                    fst_trunc: group.fst_trunc,
                    params: fresh_params,
                });
            }
        }
        self.groups = new_groups;
        if !self.core.is_empty() {
            self.core.sort_unstable();
            self.core.dedup();
        }
    }
}

/// Interns factored NFA state sets as u32 IDs.
pub(crate) struct FactoredArena {
    single_map: FxHashMap<u64, u32>,
    fingerprint_map: FxHashMap<u64, Vec<u32>>,
    pub sets: Vec<FactoredNfaSet>,
    pub is_final: Vec<bool>,
}

impl FactoredArena {
    pub fn new() -> Self {
        FactoredArena {
            single_map: FxHashMap::default(),
            fingerprint_map: FxHashMap::default(),
            sets: Vec::new(),
            is_final: Vec::new(),
        }
    }

    /// Intern a factored NFA set. Returns the u32 ID.
    pub fn intern(&mut self, set: FactoredNfaSet, any_final: bool) -> u32 {
        if set.core.len() == 1 && set.groups.is_empty() {
            let key = set.core[0];
            if let Some(&id) = self.single_map.get(&key) {
                self.is_final[id as usize] = any_final;
                return id;
            }
            let id = self.sets.len() as u32;
            self.sets.push(set);
            self.is_final.push(any_final);
            self.single_map.insert(key, id);
            return id;
        }

        let fp = set.fingerprint();
        if let Some(candidates) = self.fingerprint_map.get(&fp) {
            for &cid in candidates {
                if self.sets[cid as usize].eq_factored(&set) {
                    self.is_final[cid as usize] = any_final;
                    return cid;
                }
            }
        }

        let id = self.sets.len() as u32;
        self.sets.push(set);
        self.is_final.push(any_final);
        self.fingerprint_map.entry(fp).or_default().push(id);
        id
    }

    /// Intern a flat (no groups) sorted set.
    pub fn intern_flat(&mut self, sorted_set: Vec<u64>, any_final: bool) -> u32 {
        self.intern(FactoredNfaSet { core: sorted_set, groups: vec![] }, any_final)
    }

    pub fn len(&self) -> usize {
        self.sets.len()
    }
}

// ---------------------------------------------------------------------------
// Universality filter for peekaboo (per target-symbol)
// ---------------------------------------------------------------------------

pub(crate) struct PeekabooUniversalityFilter {
    witnesses: FxHashSet<u64>,
    pos_index: FxHashMap<u64, Vec<u32>>,
    pos_sizes: Vec<usize>,
    neg_index: FxHashMap<u64, Vec<u32>>,
    neg_next: u32,
}

impl PeekabooUniversalityFilter {
    pub(crate) fn new(_fst: &Fst, step_n: u16, y_idx: u16, ip_universal_states: &[bool]) -> Self {
        let mut witnesses = FxHashSet::default();

        for (q, &is_univ) in ip_universal_states.iter().enumerate() {
            if is_univ {
                witnesses.insert(pack_peekaboo(q as u32, step_n + 1, y_idx, false));
            }
        }

        PeekabooUniversalityFilter {
            witnesses,
            pos_index: FxHashMap::default(),
            pos_sizes: Vec::new(),
            neg_index: FxHashMap::default(),
            neg_next: 0,
        }
    }

    pub(crate) fn add_pos(&mut self, nfa_set: &[u64]) {
        let eid = self.pos_sizes.len() as u32;
        self.pos_sizes.push(nfa_set.len());
        for &e in nfa_set {
            self.pos_index.entry(e).or_default().push(eid);
        }
    }

    pub(crate) fn add_neg(&mut self, nfa_set: &[u64]) {
        let eid = self.neg_next;
        self.neg_next += 1;
        for &e in nfa_set {
            self.neg_index.entry(e).or_default().push(eid);
        }
    }

    fn has_pos_subset(&self, nfa_set: &[u64]) -> bool {
        let mut hits: FxHashMap<u32, usize> = FxHashMap::default();
        for &e in nfa_set {
            if let Some(eids) = self.pos_index.get(&e) {
                for &eid in eids {
                    let h = hits.entry(eid).or_insert(0);
                    *h += 1;
                    if *h == self.pos_sizes[eid as usize] {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn has_neg_superset(&self, nfa_set: &[u64]) -> bool {
        if nfa_set.is_empty() {
            return self.neg_next > 0;
        }
        let mut candidates: Option<FxHashSet<u32>> = None;
        for &e in nfa_set {
            match self.neg_index.get(&e) {
                None => return false,
                Some(eids) => {
                    let eid_set: FxHashSet<u32> = eids.iter().copied().collect();
                    candidates = Some(match candidates {
                        None => eid_set,
                        Some(prev) => prev.intersection(&eid_set).copied().collect(),
                    });
                    if candidates.as_ref().unwrap().is_empty() {
                        return false;
                    }
                }
            }
        }
        candidates.map_or(false, |c| !c.is_empty())
    }

    /// Project the full DFA state to y-compatible NFA states, applying refine.
    pub(crate) fn project_and_refine(&self, full_nfa_set: &[u64], y_idx: u16, step_n: u16) -> Vec<u64> {
        let mut projected = Vec::new();
        let target_len = step_n + 1;

        for &packed in full_nfa_set {
            let (fst_state, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);

            if extra_sym == NO_EXTRA {
                let clipped_len = buf_len.min(target_len);
                projected.push(pack_peekaboo(fst_state, clipped_len, NO_EXTRA, truncated));
            } else if extra_sym == y_idx {
                let clipped_len = buf_len.min(target_len);
                projected.push(pack_peekaboo(fst_state, clipped_len, extra_sym, truncated));
            }
        }

        projected.sort_unstable();
        projected.dedup();
        projected
    }

    /// Indexed projection: O(|on_target| + |matching|) instead of O(|total_set|).
    ///
    /// Uses a pre-built SymbolIndex to avoid scanning the full NFA set for each
    /// symbol. At V=5000, this is ~2500x faster per projection call.
    pub(crate) fn project_and_refine_indexed(
        &self,
        index: &SymbolIndex,
        y_idx: u16,
        step_n: u16,
    ) -> Vec<u64> {
        let target_len = step_n + 1;
        let match_count = index.by_extra.get(&y_idx).map_or(0, |v| v.len());
        let mut projected = Vec::with_capacity(index.on_target.len() + match_count);

        for &packed in &index.on_target {
            let fst_state = (packed >> 32) as u32;
            let buf_len = ((packed >> 17) & 0x7FFF) as u16;
            let truncated = (packed & 1) != 0;
            let clipped_len = buf_len.min(target_len);
            projected.push(pack_peekaboo(fst_state, clipped_len, NO_EXTRA, truncated));
        }

        if let Some(matching) = index.by_extra.get(&y_idx) {
            for &packed in matching {
                let fst_state = (packed >> 32) as u32;
                let buf_len = ((packed >> 17) & 0x7FFF) as u16;
                let truncated = (packed & 1) != 0;
                let clipped_len = buf_len.min(target_len);
                projected.push(pack_peekaboo(fst_state, clipped_len, y_idx, truncated));
            }
        }

        projected.sort_unstable();
        projected.dedup();
        projected
    }

    pub(crate) fn is_universal(
        &mut self,
        full_nfa_set: &[u64],
        y_idx: u16,
        nfa: &PeekabooNFAMapped,
        arena: &mut FactoredArena,
        eps_cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        num_source_symbols: usize,
        step_n: u16,
    ) -> bool {
        let projected = self.project_and_refine(full_nfa_set, y_idx, step_n);

        if projected.is_empty() {
            return false;
        }

        let any_final = projected.iter().any(|&s| {
            let (fst_state, buf_len, extra_sym, _) = unpack_peekaboo(s);
            nfa.fst.is_final[fst_state as usize]
                && buf_len == step_n + 1
                && extra_sym == y_idx
        });

        if !any_final {
            return false;
        }

        if projected.iter().any(|e| self.witnesses.contains(e)) {
            self.add_pos(&projected);
            return true;
        }

        if self.has_pos_subset(&projected) {
            return true;
        }

        if self.has_neg_superset(&projected) {
            return false;
        }

        let result = self.bfs_universal(
            &projected, y_idx, nfa, arena, eps_cache, num_source_symbols, step_n,
        );
        if result {
            self.add_pos(&projected);
        } else {
            self.add_neg(&projected);
        }
        result
    }

    /// Indexed universality check: uses SymbolIndex for O(|closure|) initial
    /// projection instead of O(V × |closure|). The sub-BFS (bfs_universal)
    /// continues to use flat project_and_refine since projected sets are small.
    pub(crate) fn is_universal_indexed(
        &mut self,
        index: &SymbolIndex,
        y_idx: u16,
        nfa: &PeekabooNFAMapped,
        arena: &mut FactoredArena,
        eps_cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        num_source_symbols: usize,
        step_n: u16,
    ) -> bool {
        let projected = self.project_and_refine_indexed(index, y_idx, step_n);

        if projected.is_empty() {
            return false;
        }

        let any_final = projected.iter().any(|&s| {
            let (fst_state, buf_len, extra_sym, _) = unpack_peekaboo(s);
            nfa.fst.is_final[fst_state as usize]
                && buf_len == step_n + 1
                && extra_sym == y_idx
        });

        if !any_final {
            return false;
        }

        if projected.iter().any(|e| self.witnesses.contains(e)) {
            self.add_pos(&projected);
            return true;
        }

        if self.has_pos_subset(&projected) {
            return true;
        }

        if self.has_neg_superset(&projected) {
            return false;
        }

        // Sub-BFS uses flat project_and_refine (projected sets are small)
        let result = self.bfs_universal(
            &projected, y_idx, nfa, arena, eps_cache, num_source_symbols, step_n,
        );
        if result {
            self.add_pos(&projected);
        } else {
            self.add_neg(&projected);
        }
        result
    }

    /// Indexed finality check: O(|matching|) instead of O(|total_set|).
    pub(crate) fn is_projected_final_indexed(
        &self,
        index: &SymbolIndex,
        y_idx: u16,
        fst: &Fst,
        step_n: u16,
    ) -> bool {
        if let Some(matching) = index.by_extra.get(&y_idx) {
            matching.iter().any(|&packed| {
                let fst_state = (packed >> 32) as u32;
                let buf_len = ((packed >> 17) & 0x7FFF) as u16;
                fst.is_final[fst_state as usize] && buf_len == step_n + 1
            })
        } else {
            false
        }
    }

    fn bfs_universal(
        &self,
        projected_set: &[u64],
        y_idx: u16,
        nfa: &PeekabooNFAMapped,
        arena: &mut FactoredArena,
        eps_cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        num_source_symbols: usize,
        step_n: u16,
    ) -> bool {
        let any_final = projected_set.iter().any(|&s| nfa.is_final(s));
        let start_id = arena.intern_flat(projected_set.to_vec(), any_final);

        // Use directly computed finality, not arena.is_final, which may be
        // stale when the same NFA set was interned under a different step_n.
        if !any_final {
            return false;
        }

        let mut sub_visited: FxHashSet<u32> = FxHashSet::default();
        let mut sub_worklist: VecDeque<u32> = VecDeque::new();

        sub_visited.insert(start_id);
        sub_worklist.push_back(start_id);

        while let Some(cur) = sub_worklist.pop_front() {
            // Sub-BFS states are always flat (projected, no groups)
            let cur_core = arena.sets[cur as usize].core.clone();

            // Compute finality directly from NFA set rather than arena.is_final
            let cur_final = cur_core.iter().any(|&s| nfa.is_final(s));
            if !cur_final {
                return false;
            }

            let all_arcs = nfa.compute_all_arcs(&cur_core, eps_cache);

            if all_arcs.len() < num_source_symbols {
                return false;
            }

            for (_sym, successor) in &all_arcs {
                let projected_succ = self.project_and_refine(successor, y_idx, step_n);

                if projected_succ.is_empty() {
                    return false;
                }

                let succ_final = projected_succ.iter().any(|&s| nfa.is_final(s));
                let dest_id = arena.intern_flat(projected_succ, succ_final);

                if sub_visited.insert(dest_id) {
                    sub_worklist.push_back(dest_id);
                }
            }
        }

        true
    }

    /// Universality check with a pre-computed projected set.
    /// Used with FactoredNfaSet.project_for_symbol() which does the projection
    /// externally instead of using SymbolIndex.
    pub(crate) fn is_universal_from_projected(
        &mut self,
        projected: Vec<u64>,
        y_idx: u16,
        nfa: &PeekabooNFAMapped,
        arena: &mut FactoredArena,
        eps_cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        num_source_symbols: usize,
        step_n: u16,
    ) -> bool {
        if projected.is_empty() {
            return false;
        }

        let any_final = projected.iter().any(|&s| {
            let (fst_state, buf_len, extra_sym, _) = unpack_peekaboo(s);
            nfa.fst.is_final[fst_state as usize]
                && buf_len == step_n + 1
                && extra_sym == y_idx
        });

        if !any_final {
            return false;
        }

        if projected.iter().any(|e| self.witnesses.contains(e)) {
            self.add_pos(&projected);
            return true;
        }

        if self.has_pos_subset(&projected) {
            return true;
        }

        if self.has_neg_superset(&projected) {
            return false;
        }

        let result = self.bfs_universal(
            &projected, y_idx, nfa, arena, eps_cache, num_source_symbols, step_n,
        );
        if result {
            self.add_pos(&projected);
        } else {
            self.add_neg(&projected);
        }
        result
    }

    pub(crate) fn is_projected_final(
        &self,
        full_nfa_set: &[u64],
        y_idx: u16,
        fst: &Fst,
        step_n: u16,
    ) -> bool {
        full_nfa_set.iter().any(|&packed| {
            let (fst_state, buf_len, extra_sym, _) = unpack_peekaboo(packed);
            fst.is_final[fst_state as usize]
                && buf_len == step_n + 1
                && extra_sym == y_idx
        })
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

use crate::decompose::FsaResult;

pub struct PeekabooProfileStats {
    pub total_ms: f64,
    pub init_ms: f64,
    pub bfs_ms: f64,
    pub extract_ms: f64,

    // Per-step data
    pub num_steps: u32,
    pub per_step_visited: Vec<u32>,
    pub per_step_frontier_size: Vec<u32>,

    // BFS aggregates
    pub total_bfs_visited: u64,
    pub compute_arcs_ms: f64,
    pub compute_arcs_calls: u64,
    pub intern_ms: f64,
    pub intern_calls: u64,
    pub universal_ms: f64,
    pub universal_calls: u64,
    pub universal_true: u64,
    pub universal_false: u64,

    // Arena stats
    pub arena_size: u32,
    pub max_powerset_size: usize,
    pub avg_powerset_size: f64,

    // Merged incoming stats
    pub merged_incoming_states: u32,
    pub merged_incoming_arcs: u64,

    // Eps cache
    pub eps_cache_clears: u32,

    // Per-symbol result sizes
    pub per_symbol_q_stops: Vec<(u32, u32)>,  // (sym, count)
    pub per_symbol_r_stops: Vec<(u32, u32)>,
}

pub struct PeekabooResult {
    pub per_symbol: FxHashMap<u32, (FsaResult, FsaResult)>,
    pub stats: PeekabooProfileStats,
}

// ---------------------------------------------------------------------------
// DirtyPeekaboo: true dirty-state incremental peekaboo decomposition
// ---------------------------------------------------------------------------
// State status constants for DirtyPeekaboo
pub(crate) const STATUS_NEW: u8 = 0;       // needs full expansion
pub(crate) const STATUS_INTERIOR: u8 = 1;  // non-final, expanded (has cached arcs)
pub(crate) const STATUS_QSTOP: u8 = 2;     // universal final, no outgoing arcs
pub(crate) const STATUS_RSTOP: u8 = 3;     // non-universal final, expanded (has cached arcs)

/// Evict stale eps_cache entries for a peekaboo prefix extension.
/// An entry is stale if the key NFA state has buf_len >= frontier,
/// or the closure result contains states with buf_len >= frontier.
pub(crate) fn evict_peekaboo_eps_cache(cache: &mut FxHashMap<u64, (Vec<u64>, u16)>, frontier: u16) {
    cache.retain(|&key, (_value, max_bl)| {
        let key_bl = ((key >> 17) & 0x7FFF) as u16;
        key_bl < frontier && *max_bl < frontier
    });
}

/// True dirty-state incremental peekaboo decomposition.
///
/// Uses a single-pass BFS with `PeekabooNFAMapped(step_n=target.len())`
/// instead of N+1 sequential steps. Persists the full DFA structure
/// (per-state arcs, status, reverse_arcs) across calls. On prefix extension,
/// only re-expands "dirty" states (whose NFA elements have buf_len >= frontier)
/// and their "border" predecessors.
pub struct DirtyPeekaboo {
    // FST metadata (computed once in new())
    output_alphabet: Vec<u32>,
    sym_to_idx: FxHashMap<u32, u16>,
    idx_to_sym: Vec<u32>,
    ip_universal_states: Vec<bool>,
    num_source_symbols: usize,

    // Persistent DFA structure
    arena: FactoredArena,
    global_start_id: u32,
    arcs_from: Vec<Vec<(u32, u32)>>,       // [sid] → [(label, dest_sid)]
    state_status: Vec<u8>,                  // [sid] → STATUS_*
    max_bufpos: Vec<u16>,                   // [sid] → max buf_len in NFA set
    reverse_arcs: Vec<Vec<u32>>,            // [sid] → [predecessor sids]
    reachable: Vec<u32>,                    // BFS-order reachable state list
    reachable_flags: Vec<bool>,             // dense membership for O(1) lookup
    needs_reexpand: Vec<bool>,              // scratch buffer for dirty+border marking

    // Per-symbol Q/R
    decomp_q: FxHashMap<u16, Vec<u32>>,
    decomp_r: FxHashMap<u16, Vec<u32>>,

    // Caches
    eps_cache: FxHashMap<u64, (Vec<u64>, u16)>,
    fst_univ_cache: FxHashMap<Vec<u32>, bool>,

    // Target tracking
    prev_target: Vec<u32>,

    // Generation counter: incremented on every full_reset() so callers can
    // detect when DFA state IDs have been invalidated.
    generation: u64,

    // Arena mode: true = factored (groups), false = flat (intern_flat only)
    use_factored: bool,
}

impl DirtyPeekaboo {
    pub fn new(fst: &Fst, use_factored: bool) -> Self {
        // Build the output alphabet.
        let mut output_alphabet: Vec<u32> = Vec::new();
        {
            let mut seen = FxHashSet::default();
            for arc in &fst.arcs {
                if arc.output != EPSILON && seen.insert(arc.output) {
                    output_alphabet.push(arc.output);
                }
            }
            output_alphabet.sort_unstable();
        }

        // Build sym_to_idx map.
        let mut sym_to_idx: FxHashMap<u32, u16> = FxHashMap::default();
        let mut idx_to_sym: Vec<u32> = Vec::new();
        for &sym in &output_alphabet {
            let idx = idx_to_sym.len() as u16;
            sym_to_idx.insert(sym, idx);
            idx_to_sym.push(sym);
        }

        assert!(
            idx_to_sym.len() < NO_EXTRA as usize,
            "Too many output symbols for u16 encoding"
        );

        let ip_universal_states = compute_ip_universal_states(fst);
        let num_source_symbols = fst.source_alphabet.len();

        DirtyPeekaboo {
            output_alphabet,
            sym_to_idx,
            idx_to_sym,
            ip_universal_states,
            num_source_symbols,
            arena: FactoredArena::new(),
            global_start_id: 0,
            arcs_from: Vec::new(),
            state_status: Vec::new(),
            max_bufpos: Vec::new(),
            reverse_arcs: Vec::new(),
            reachable: Vec::new(),
            reachable_flags: Vec::new(),
            needs_reexpand: Vec::new(),
            decomp_q: FxHashMap::default(),
            decomp_r: FxHashMap::default(),
            eps_cache: FxHashMap::default(),
            fst_univ_cache: FxHashMap::default(),
            prev_target: Vec::new(),
            generation: 0,
            use_factored,
        }
    }

    fn is_prefix_extension(&self, target: &[u32]) -> bool {
        if target.len() <= self.prev_target.len() {
            return false;
        }
        target[..self.prev_target.len()] == self.prev_target[..]
    }

    fn full_reset(&mut self) {
        self.generation += 1;
        self.arena = FactoredArena::new();
        self.global_start_id = 0;
        self.arcs_from.clear();
        self.state_status.clear();
        self.max_bufpos.clear();
        self.reverse_arcs.clear();
        self.reachable.clear();
        self.reachable_flags.clear();
        self.needs_reexpand.clear();
        self.decomp_q.clear();
        self.decomp_r.clear();
        self.eps_cache.clear();
        // NOTE: fst_univ_cache is NOT cleared — it's target-independent
    }

    /// Ensure all per-state arrays are sized to cover `needed` entries,
    /// populating max_bufpos for any newly-added arena states.
    fn ensure_capacity(&mut self, needed: usize) {
        let old_len = self.arcs_from.len();
        if needed > old_len {
            self.arcs_from.resize_with(needed, Vec::new);
            self.state_status.resize(needed, STATUS_NEW);
            self.reverse_arcs.resize_with(needed, Vec::new);
            self.max_bufpos.resize(needed, 0);
            for sid in old_len..needed {
                self.max_bufpos[sid] = self.arena.sets[sid].max_bufpos();
            }
        }
    }

    /// Remove `sid` from reverse_arcs of all its outgoing arc destinations,
    /// and clear its arcs_from.
    fn remove_outgoing_reverse_arcs(&mut self, sid: u32) {
        let arcs = std::mem::take(&mut self.arcs_from[sid as usize]);
        for &(_lbl, dst) in &arcs {
            let dst_usize = dst as usize;
            if dst_usize < self.reverse_arcs.len() {
                let ra = &mut self.reverse_arcs[dst_usize];
                if let Some(pos) = ra.iter().position(|&s| s == sid) {
                    ra.swap_remove(pos);
                }
            }
        }
    }

    /// Collect trimmed arcs via backward BFS from stop states through reverse_arcs.
    fn collect_arcs_trimmed(&self, stops: &[u32]) -> FsaResult {
        if stops.is_empty() {
            return FsaResult {
                num_states: 0,
                start: vec![],
                stop: vec![],
                arc_src: vec![],
                arc_lbl: vec![],
                arc_dst: vec![],
            };
        }

        let n = self.arena.len();

        // Backward BFS from stops through reverse_arcs, intersected with reachable_flags
        let mut backward = vec![false; n];
        let mut bfs: VecDeque<u32> = VecDeque::new();
        for &sid in stops {
            if (sid as usize) < n {
                backward[sid as usize] = true;
                bfs.push_back(sid);
            }
        }
        while let Some(sid) = bfs.pop_front() {
            let sid_usize = sid as usize;
            if sid_usize < self.reverse_arcs.len() {
                for &src in &self.reverse_arcs[sid_usize] {
                    let src_usize = src as usize;
                    if src_usize < n
                        && !backward[src_usize]
                        && self.reachable_flags[src_usize]
                    {
                        backward[src_usize] = true;
                        bfs.push_back(src);
                    }
                }
            }
        }

        let start = if (self.global_start_id as usize) < n
            && backward[self.global_start_id as usize]
        {
            vec![self.global_start_id]
        } else {
            vec![]
        };

        let mut arc_src = Vec::new();
        let mut arc_lbl = Vec::new();
        let mut arc_dst = Vec::new();
        for &sid in &self.reachable {
            let sid_usize = sid as usize;
            if backward[sid_usize] && self.state_status[sid_usize] != STATUS_QSTOP {
                for &(l, d) in &self.arcs_from[sid_usize] {
                    if (d as usize) < n && backward[d as usize] {
                        arc_src.push(sid);
                        arc_lbl.push(l);
                        arc_dst.push(d);
                    }
                }
            }
        }

        FsaResult {
            num_states: n as u32,
            start,
            stop: stops.to_vec(),
            arc_src,
            arc_lbl,
            arc_dst,
        }
    }

    /// Extract per-symbol FSA results from decomp_q/decomp_r.
    fn extract_results(&self) -> FxHashMap<u32, (FsaResult, FsaResult)> {
        let mut per_symbol_result: FxHashMap<u32, (FsaResult, FsaResult)> = FxHashMap::default();

        for &sym in &self.output_alphabet {
            let y_idx = self.sym_to_idx[&sym];

            // Filter stops to reachable states only
            let q_stops: Vec<u32> = self.decomp_q.get(&y_idx)
                .map(|v| v.iter().filter(|&&sid| {
                    (sid as usize) < self.reachable_flags.len()
                        && self.reachable_flags[sid as usize]
                }).copied().collect())
                .unwrap_or_default();
            let r_stops: Vec<u32> = self.decomp_r.get(&y_idx)
                .map(|v| v.iter().filter(|&&sid| {
                    (sid as usize) < self.reachable_flags.len()
                        && self.reachable_flags[sid as usize]
                }).copied().collect())
                .unwrap_or_default();

            let q_fsa = self.collect_arcs_trimmed(&q_stops);
            let r_fsa = self.collect_arcs_trimmed(&r_stops);

            per_symbol_result.insert(sym, (q_fsa, r_fsa));
        }

        per_symbol_result
    }

    // --- Accessors for py.rs beam view ---

    pub fn global_start_id(&self) -> u32 {
        self.global_start_id
    }

    /// Generation counter — incremented on every full_reset().
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn arcs_from(&self, sid: u32) -> &[(u32, u32)] {
        let sid_usize = sid as usize;
        if sid_usize < self.arcs_from.len() {
            &self.arcs_from[sid_usize]
        } else {
            &[]
        }
    }

    /// Follow a single arc labeled `x` from `sid`. Returns `Some(dest)` or `None`.
    pub fn step(&self, sid: u32, x: u32) -> Option<u32> {
        for &(lbl, dst) in self.arcs_from(sid) {
            if lbl == x {
                return Some(dst);
            }
        }
        None
    }

    /// Run a source path from the global start state, returning the reached
    /// DFA state or `None` if any arc is missing.
    pub fn run(&self, source_path: &[u32]) -> Option<u32> {
        let mut state = self.global_start_id;
        for &x in source_path {
            state = self.step(state, x)?;
        }
        Some(state)
    }

    pub fn idx_to_sym(&self) -> &[u32] {
        &self.idx_to_sym
    }

    pub fn decomp_q(&self) -> &FxHashMap<u16, Vec<u32>> {
        &self.decomp_q
    }

    pub fn decomp_r(&self) -> &FxHashMap<u16, Vec<u32>> {
        &self.decomp_r
    }

    pub fn reachable_flags(&self) -> &[bool] {
        &self.reachable_flags
    }

    pub fn arena_sets(&self, sid: u32) -> &FactoredNfaSet {
        &self.arena.sets[sid as usize]
    }

    /// Compute preimage stop states: DFA states where any NFA element
    /// effectively has buf == target[:step_n] and fst_state is final.
    /// These represent source strings that produce exactly the current target.
    ///
    /// An NFA element matches preimage in two encodings:
    /// - Canonical: buf_len == step_n, extra_sym == NO_EXTRA
    /// - Non-canonical: buf_len == step_n, extra_sym == sym_to_idx[target[step_n-1]]
    ///   (i.e. buffer = target[:step_n-1] + target[step_n-1] = target[:step_n])
    ///
    /// The non-canonical case arises for resume-frontier states carried from a
    /// parent step where the extra symbol was beyond the parent's target but
    /// now matches the extended target.
    pub fn compute_preimage_stops(&self, fst: &Fst, step_n: u16) -> Vec<u32> {
        // Pre-compute the non-canonical extra_sym that also matches preimage.
        let non_canonical_extra: Option<u16> = if step_n > 0 {
            let last_target_sym = self.prev_target[(step_n - 1) as usize];
            self.sym_to_idx.get(&last_target_sym).copied()
        } else {
            None
        };

        let mut stops = Vec::new();
        for &sid in &self.reachable {
            let sid_usize = sid as usize;
            if sid_usize >= self.reachable_flags.len() || !self.reachable_flags[sid_usize] {
                continue;
            }
            let factored = &self.arena.sets[sid_usize];
            if factored.is_preimage(fst, step_n, &self.prev_target, &self.sym_to_idx) {
                stops.push(sid);
            }
        }
        stops
    }

    /// Compute resume frontiers: for each output symbol y, collect non-truncated
    /// reachable states that sit on the truncation boundary for y.
    ///
    /// A state is on the resume frontier for y if:
    /// - It has no truncated NFA elements, AND one of:
    ///   (a) any successor contains a truncated NFA element with extra_sym == y_idx, OR
    ///   (b) the state is in decomp_q[y_idx] or decomp_r[y_idx]
    pub fn compute_resume_frontiers(&self) -> FxHashMap<u16, Vec<u32>> {
        let mut frontiers: FxHashMap<u16, Vec<u32>> = FxHashMap::default();

        for &sid in &self.reachable {
            let sid_usize = sid as usize;
            if sid_usize >= self.reachable_flags.len() || !self.reachable_flags[sid_usize] {
                continue;
            }
            let factored = &self.arena.sets[sid_usize];

            // Check if this state has any truncated NFA element
            if factored.has_truncated() {
                continue;  // Only non-truncated states can be resume frontiers
            }

            // Check successors for truncated elements
            let mut frontier_syms: FxHashSet<u16> = FxHashSet::default();
            for &(_lbl, dst) in &self.arcs_from[sid_usize] {
                let dst_usize = dst as usize;
                if dst_usize < self.arena.sets.len() {
                    let dst_factored = &self.arena.sets[dst_usize];
                    // Check core for truncated elements
                    for &packed in &dst_factored.core {
                        let extra_sym = ((packed >> 1) & 0xFFFF) as u16;
                        let truncated = (packed & 1) != 0;
                        if truncated && extra_sym != NO_EXTRA {
                            frontier_syms.insert(extra_sym);
                        }
                    }
                    // Check groups: truncated FST states carry truncated elements
                    for group in &dst_factored.groups {
                        if !group.fst_trunc.is_empty() {
                            for &(_bl, es) in &group.params {
                                frontier_syms.insert(es);
                            }
                        }
                    }
                }
            }

            for &y_idx in &frontier_syms {
                frontiers.entry(y_idx).or_default().push(sid);
            }

            // Also add if state is in decomp_q or decomp_r for any symbol
            for (&y_idx, q_stops) in &self.decomp_q {
                if q_stops.contains(&sid) {
                    frontiers.entry(y_idx).or_default().push(sid);
                }
            }
            for (&y_idx, r_stops) in &self.decomp_r {
                if r_stops.contains(&sid) {
                    frontiers.entry(y_idx).or_default().push(sid);
                }
            }
        }

        // Deduplicate
        for (_y_idx, sids) in frontiers.iter_mut() {
            sids.sort_unstable();
            sids.dedup();
        }

        frontiers
    }

    /// Decompose without extracting FSA results — just runs the BFS and
    /// populates the DFA structure. Used by decompose_for_beam().
    pub fn decompose_bfs_only(&mut self, fst: &Fst, target: &[u32]) {
        // Run the full decompose but discard the extracted results.
        // We need the BFS side effects (arena, arcs_from, decomp_q/r, etc.)
        let _ = self.decompose(fst, target);
    }

    /// Main entry point: decompose the FST for the given target.
    /// Uses a single-pass BFS with step_n=target.len(). On prefix extension,
    /// only re-expands dirty and border states.
    pub fn decompose(&mut self, fst: &Fst, target: &[u32]) -> PeekabooResult {
        let total_start = Instant::now();
        let target_len = target.len();

        // Same target → extract cached results
        if target == self.prev_target.as_slice() && self.arena.len() > 0 {
            let extract_start = Instant::now();
            let per_symbol = self.extract_results();
            let extract_ms = extract_start.elapsed().as_secs_f64() * 1000.0;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            return PeekabooResult {
                per_symbol,
                stats: PeekabooProfileStats {
                    total_ms,
                    init_ms: 0.0,
                    bfs_ms: 0.0,
                    extract_ms,
                    num_steps: 0,
                    per_step_visited: Vec::new(),
                    per_step_frontier_size: Vec::new(),
                    total_bfs_visited: 0,
                    compute_arcs_ms: 0.0,
                    compute_arcs_calls: 0,
                    intern_ms: 0.0,
                    intern_calls: 0,
                    universal_ms: 0.0,
                    universal_calls: 0,
                    universal_true: 0,
                    universal_false: 0,
                    arena_size: self.arena.len() as u32,
                    max_powerset_size: 0,
                    avg_powerset_size: 0.0,
                    merged_incoming_states: 0,
                    merged_incoming_arcs: 0,
                    eps_cache_clears: 0,
                    per_symbol_q_stops: Vec::new(),
                    per_symbol_r_stops: Vec::new(),
                },
            };
        }

        // Determine if prefix extension
        let is_extension = self.is_prefix_extension(target);
        let mut dirty_border: Vec<u32> = Vec::new();

        if is_extension {
            let frontier = self.prev_target.len() as u16;

            // Evict stale eps_cache entries
            evict_peekaboo_eps_cache(&mut self.eps_cache, frontier);

            // Ensure needs_reexpand covers arena
            let n = self.arena.len();
            if self.needs_reexpand.len() < n {
                self.needs_reexpand.resize(n, false);
            }

            // Step 1: Mark dirty states (max_bufpos >= frontier)
            for &sid in &self.reachable {
                let sid_usize = sid as usize;
                if sid_usize < self.max_bufpos.len() && self.max_bufpos[sid_usize] >= frontier {
                    self.needs_reexpand[sid_usize] = true;
                    dirty_border.push(sid);
                }
            }

            // Step 2: Mark border states using reverse_arcs
            let dirty_count = dirty_border.len();
            for i in 0..dirty_count {
                let dirty_sid = dirty_border[i];
                let dirty_usize = dirty_sid as usize;
                if dirty_usize < self.reverse_arcs.len() {
                    for j in 0..self.reverse_arcs[dirty_usize].len() {
                        let src = self.reverse_arcs[dirty_usize][j];
                        let src_usize = src as usize;
                        if src_usize < n
                            && !self.needs_reexpand[src_usize]
                            && self.state_status[src_usize] != STATUS_NEW
                        {
                            self.needs_reexpand[src_usize] = true;
                            dirty_border.push(src);
                        }
                    }
                }
            }

            // Step 3: Reset dirty+border states
            for &sid in &dirty_border {
                let sid_usize = sid as usize;
                self.remove_outgoing_reverse_arcs(sid);
                self.state_status[sid_usize] = STATUS_NEW;
                self.needs_reexpand[sid_usize] = false;
            }
        } else {
            // Non-extension → full reset
            self.full_reset();
        }

        // Create NFA with step_n = target_len (single-pass)
        // Clone sym_to_idx to avoid holding an immutable borrow on self
        let sym_to_idx = self.sym_to_idx.clone();
        let step_n = target_len as u16;
        let nfa = PeekabooNFAMapped::new(fst, target, step_n, &sym_to_idx);

        // Compute start state (flat: eps closure is always flat)
        let raw_starts = nfa.start_states();
        let init_closed = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);
        let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
        let start_id = self.arena.intern_flat(init_closed, any_final);
        self.global_start_id = start_id;

        // Ensure capacity for all arena entries
        self.ensure_capacity(self.arena.len());

        // Ensure reachable_flags covers arena
        let needed_flags = self.arena.len().max(start_id as usize + 1);
        if self.reachable_flags.len() < needed_flags {
            self.reachable_flags.resize(needed_flags, false);
        }

        // Seed worklist
        let mut worklist: VecDeque<u32> = VecDeque::new();

        if self.state_status[start_id as usize] == STATUS_NEW {
            worklist.push_back(start_id);
            if !self.reachable_flags[start_id as usize] {
                self.reachable_flags[start_id as usize] = true;
                self.reachable.push(start_id);
            }
        }

        // Dirty+border states are already STATUS_NEW and in reachable
        for &sid in &dirty_border {
            worklist.push_back(sid);
        }

        // Clear per-symbol Q/R
        self.decomp_q.clear();
        self.decomp_r.clear();

        // Create fresh universality filters (per y_idx, cheap to construct)
        let mut univ_filters: FxHashMap<u16, PeekabooUniversalityFilter> = FxHashMap::default();

        // Persistent FST-level closure caches shared across all BFS states.
        // Avoids redundant closure computations when multiple DFA states share
        // common FST destination states (typical for BPE).
        let mut fst_full_cache: FxHashMap<u32, (Vec<u32>, Vec<u32>)> = FxHashMap::default();
        let mut fst_trunc_cache: FxHashMap<u32, Vec<u32>> = FxHashMap::default();

        let bfs_start = Instant::now();

        // BFS loop
        while let Some(sid) = worklist.pop_front() {
            if self.state_status[sid as usize] != STATUS_NEW {
                continue;
            }

            let mut factored = self.arena.sets[sid as usize].clone();

            // Move stale group params (bl <= step_n) back to core so they
            // get correct NFA-level re-interpretation via effective_state.
            factored.normalize_for_step(step_n);

            // Use factored methods for relevant symbols and projection
            let relevant_syms = factored.relevant_symbols(step_n);

            let mut continuous: Option<u16> = None;
            let mut has_final_syms = false;

            for &y_idx in &relevant_syms {
                if !univ_filters.contains_key(&y_idx) {
                    univ_filters.insert(
                        y_idx,
                        PeekabooUniversalityFilter::new(
                            fst, step_n, y_idx, &self.ip_universal_states,
                        ),
                    );
                }

                {
                    // Project using factored set: O(|core| + |closure|) per symbol
                    let projected = factored.project_for_symbol(y_idx, step_n);
                    let cache_hit = if !projected.is_empty() {
                        let all_frontier = projected.iter().all(|&packed| {
                            let (_, buf_len, extra_sym, _) = unpack_peekaboo(packed);
                            buf_len == step_n + 1 && extra_sym == y_idx
                        });
                        if all_frontier {
                            let mut fst_states: Vec<u32> = projected.iter()
                                .map(|&packed| (packed >> 32) as u32)
                                .collect();
                            fst_states.sort_unstable();
                            fst_states.dedup();
                            self.fst_univ_cache.get(&fst_states).copied()
                                .map(|result| (result, fst_states))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let is_univ = if let Some((cached_result, fst_states)) = cache_hit {
                        let filter = univ_filters.get_mut(&y_idx).unwrap();
                        if cached_result {
                            filter.add_pos(&projected);
                        } else {
                            filter.add_neg(&projected);
                        }
                        let _ = fst_states;
                        cached_result
                    } else {
                        let filter = univ_filters.get_mut(&y_idx).unwrap();
                        let result = filter.is_universal_from_projected(
                            projected.clone(),
                            y_idx,
                            &nfa,
                            &mut self.arena,
                            &mut self.eps_cache,
                            self.num_source_symbols,
                            step_n,
                        );

                        // Ensure capacity after universality sub-BFS may grow arena
                        self.ensure_capacity(self.arena.len());
                        if self.arena.len() > self.reachable_flags.len() {
                            self.reachable_flags.resize(self.arena.len(), false);
                        }

                        // Cache if pure-frontier
                        if !projected.is_empty() {
                            let all_frontier = projected.iter().all(|&packed| {
                                let (_, buf_len, extra_sym, _) = unpack_peekaboo(packed);
                                buf_len == step_n + 1 && extra_sym == y_idx
                            });
                            if all_frontier {
                                let mut fst_states: Vec<u32> = projected.iter()
                                    .map(|&packed| (packed >> 32) as u32)
                                    .collect();
                                fst_states.sort_unstable();
                                fst_states.dedup();
                                self.fst_univ_cache.insert(fst_states, result);
                            }
                        }

                        result
                    };

                    if is_univ {
                        if let Some(prev) = continuous {
                            panic!(
                                "State is universal for both symbol {} and {} — \
                                 FST is likely non-functional",
                                prev, y_idx
                            );
                        }
                        self.decomp_q.entry(y_idx).or_default().push(sid);
                        continuous = Some(y_idx);
                        continue;
                    }
                }

                if factored.is_projected_final(y_idx, fst, step_n) {
                    self.decomp_r.entry(y_idx).or_default().push(sid);
                    has_final_syms = true;
                }
            }

            if continuous.is_some() {
                self.state_status[sid as usize] = STATUS_QSTOP;
                continue;
            }

            // Expand arcs: factored mode uses grouped sets, flat mode uses plain sorted vecs
            let mut interned_arcs: Vec<(u32, u32)>;
            let mut unique_dests: FxHashSet<u32> = FxHashSet::default();

            if self.use_factored {
                let all_arcs = nfa.compute_all_arcs_factored(
                    &factored, &mut self.eps_cache,
                    &mut fst_full_cache, &mut fst_trunc_cache,
                );
                interned_arcs = Vec::with_capacity(all_arcs.len());
                for (x, succ_factored) in all_arcs {
                    let any_final = succ_factored.any_final(&nfa);
                    let dest_id = self.arena.intern(succ_factored, any_final);
                    let needed = dest_id as usize + 1;
                    self.ensure_capacity(needed);
                    if needed > self.reachable_flags.len() {
                        self.reachable_flags.resize(needed, false);
                    }
                    interned_arcs.push((x, dest_id));
                    unique_dests.insert(dest_id);
                }
            } else {
                let all_arcs = nfa.compute_all_arcs_cached(
                    &factored.core, &mut self.eps_cache,
                    &mut fst_full_cache, &mut fst_trunc_cache,
                );
                interned_arcs = Vec::with_capacity(all_arcs.len());
                for (x, succ_flat) in all_arcs {
                    let mut sorted = succ_flat;
                    sorted.sort_unstable();
                    sorted.dedup();
                    let any_final = sorted.iter().any(|&p| {
                        let (fst_state, _, _, _) = unpack_peekaboo(p);
                        nfa.fst.is_final[fst_state as usize]
                    });
                    let dest_id = self.arena.intern_flat(sorted, any_final);
                    let needed = dest_id as usize + 1;
                    self.ensure_capacity(needed);
                    if needed > self.reachable_flags.len() {
                        self.reachable_flags.resize(needed, false);
                    }
                    interned_arcs.push((x, dest_id));
                    unique_dests.insert(dest_id);
                }
            }

            // Store all arcs explicitly
            self.arcs_from[sid as usize] = interned_arcs;

            // Add reverse arcs and enqueue successors for ALL unique destinations
            for &dest_id in &unique_dests {
                self.reverse_arcs[dest_id as usize].push(sid);
                if self.state_status[dest_id as usize] == STATUS_NEW {
                    worklist.push_back(dest_id);
                    if !self.reachable_flags[dest_id as usize] {
                        self.reachable_flags[dest_id as usize] = true;
                        self.reachable.push(dest_id);
                    }
                }
            }

            // Classify state
            self.state_status[sid as usize] = if has_final_syms {
                STATUS_RSTOP
            } else {
                STATUS_INTERIOR
            };
        }

        let bfs_ms = bfs_start.elapsed().as_secs_f64() * 1000.0;

        // Update prev_target
        self.prev_target.clear();
        self.prev_target.extend_from_slice(target);

        // Extract results
        let extract_start = Instant::now();
        let per_symbol = self.extract_results();
        let extract_ms = extract_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let arena_size = self.arena.len() as u32;

        PeekabooResult {
            per_symbol,
            stats: PeekabooProfileStats {
                total_ms,
                init_ms: 0.0,
                bfs_ms,
                extract_ms,
                num_steps: 1,
                per_step_visited: Vec::new(),
                per_step_frontier_size: Vec::new(),
                total_bfs_visited: 0,
                compute_arcs_ms: 0.0,
                compute_arcs_calls: 0,
                intern_ms: 0.0,
                intern_calls: 0,
                universal_ms: 0.0,
                universal_calls: 0,
                universal_true: 0,
                universal_false: 0,
                arena_size,
                max_powerset_size: 0,
                avg_powerset_size: 0.0,
                merged_incoming_states: 0,
                merged_incoming_arcs: 0,
                eps_cache_clears: 0,
                per_symbol_q_stops: Vec::new(),
                per_symbol_r_stops: Vec::new(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// LazyPeekabooDFA: per-step lazy DFA for FusedTransducedLM
// ---------------------------------------------------------------------------

/// Cheap metadata extracted from NFA set without arc computation.
pub struct StateMeta {
    pub relevant_symbols: Vec<u16>,
    pub final_symbols: Vec<u16>,
    pub is_preimage: bool,
    pub has_truncated: bool,
    pub trunc_output_syms: Vec<u16>,
}

impl Default for StateMeta {
    fn default() -> Self {
        StateMeta {
            relevant_symbols: Vec::new(),
            final_symbols: Vec::new(),
            is_preimage: false,
            has_truncated: false,
            trunc_output_syms: Vec::new(),
        }
    }
}

/// Expensive classification including universality check.
pub struct ClassifyResult {
    pub quotient_sym: Option<u16>,
    pub remainder_syms: Vec<u16>,
}

impl Default for ClassifyResult {
    fn default() -> Self {
        ClassifyResult {
            quotient_sym: None,
            remainder_syms: Vec::new(),
        }
    }
}

/// Lazy DFA over the peekaboo NFA for FusedTransducedLM.
///
/// The arena persists across steps so that carry-forward beam items'
/// u32 DFA state IDs remain valid.  Each `new_step()` call clears the
/// per-step caches (arcs, meta, classify) and computes new start states,
/// but the arena retains all previously interned NFA sets.
pub struct LazyPeekabooDFA {
    sym_to_idx: FxHashMap<u32, u16>,
    idx_to_sym: Vec<u32>,
    ip_universal_states: Vec<bool>,
    all_final_universal: bool,
    num_source_symbols: usize,
    use_factored: bool,

    // Per-step (cleared on new_step)
    target: Vec<u32>,
    step_n: u16,

    // Persistent across steps
    arena: FactoredArena,
    pub fst_univ_cache: FxHashMap<Vec<u32>, bool>,
    // FST-level closure caches: persistent across steps and states since
    // closures depend only on FST structure, not target or step.
    fst_full_cache: FxHashMap<u32, (Vec<u32>, Vec<u32>)>,
    fst_trunc_cache: FxHashMap<u32, Vec<u32>>,

    // Per-step caches (cleared on new_step)
    eps_cache: FxHashMap<u64, (Vec<u64>, u16)>,
    arcs_computed: Vec<bool>,
    arcs_from: Vec<Vec<(u32, u32)>>,       // [sid] → [(label, dest_sid)]
    meta_computed: Vec<bool>,
    meta: Vec<StateMeta>,
    classify_computed: Vec<bool>,
    classify: Vec<ClassifyResult>,
    univ_filters: FxHashMap<u16, PeekabooUniversalityFilter>,

    start_ids: Vec<u32>,
}

impl LazyPeekabooDFA {
    /// Create a new LazyPeekabooDFA with per-FST data.  Call `new_step()`
    /// before using any query methods.
    pub fn new(
        sym_to_idx: FxHashMap<u32, u16>,
        idx_to_sym: Vec<u32>,
        ip_universal_states: Vec<bool>,
        all_final_universal: bool,
        num_source_symbols: usize,
        use_factored: bool,
    ) -> Self {
        LazyPeekabooDFA {
            sym_to_idx,
            idx_to_sym,
            ip_universal_states,
            all_final_universal,
            num_source_symbols,
            use_factored,
            target: Vec::new(),
            step_n: 0,
            arena: FactoredArena::new(),
            fst_univ_cache: FxHashMap::default(),
            fst_full_cache: FxHashMap::default(),
            fst_trunc_cache: FxHashMap::default(),
            eps_cache: FxHashMap::default(),
            arcs_computed: Vec::new(),
            arcs_from: Vec::new(),
            meta_computed: Vec::new(),
            meta: Vec::new(),
            classify_computed: Vec::new(),
            classify: Vec::new(),
            univ_filters: FxHashMap::default(),
            start_ids: Vec::new(),
        }
    }

    /// Reset for a new target prefix.  The arena is preserved (old state IDs
    /// remain valid for carry-forward), but all per-step caches are cleared.
    pub fn new_step(&mut self, fst: &Fst, target: Vec<u32>) {
        let step_n = target.len() as u16;
        self.target = target;
        self.step_n = step_n;

        // Clear per-step caches
        self.eps_cache.clear();
        self.univ_filters.clear();

        // Reset computed flags for all existing arena states
        let n = self.arena.len();
        // Re-use vec capacity, fill with false
        self.arcs_computed.clear();
        self.arcs_computed.resize(n, false);
        self.arcs_from.iter_mut().for_each(|v| v.clear());
        self.arcs_from.resize_with(n, Vec::new);
        self.meta_computed.clear();
        self.meta_computed.resize(n, false);
        self.meta.clear();
        self.meta.resize_with(n, StateMeta::default);
        self.classify_computed.clear();
        self.classify_computed.resize(n, false);
        self.classify.clear();
        self.classify.resize_with(n, ClassifyResult::default);

        // Compute new start states
        let sym_to_idx = self.sym_to_idx.clone();
        let nfa = PeekabooNFAMapped::new(fst, &self.target, step_n, &sym_to_idx);
        let raw_starts = nfa.start_states();
        let init_closed = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);
        let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
        let start_id = self.arena.intern_flat(init_closed, any_final);
        self.start_ids = vec![start_id];

        // Ensure capacity for any new arena entries
        self.ensure_capacity(self.arena.len());
    }

    pub fn start_ids(&self) -> &[u32] {
        &self.start_ids
    }

    pub fn idx_to_sym(&self) -> &[u32] {
        &self.idx_to_sym
    }

    fn ensure_capacity(&mut self, needed: usize) {
        let old_len = self.arcs_computed.len();
        if needed > old_len {
            self.arcs_computed.resize(needed, false);
            self.arcs_from.resize_with(needed, Vec::new);
            self.meta_computed.resize(needed, false);
            self.meta.resize_with(needed, StateMeta::default);
            self.classify_computed.resize(needed, false);
            self.classify.resize_with(needed, ClassifyResult::default);
        }
    }

    /// Lazily compute StateMeta from NFA set (cheap, no arc computation).
    pub fn ensure_meta(&mut self, fst: &Fst, sid: u32) {
        let sid_usize = sid as usize;
        if sid_usize < self.meta_computed.len() && self.meta_computed[sid_usize] {
            return;
        }
        self.ensure_capacity(sid_usize + 1);

        let mut factored = self.arena.sets[sid_usize].clone();
        let step_n = self.step_n;
        factored.normalize_for_step(step_n);

        let relevant_symbols = factored.relevant_symbols(step_n);
        let final_symbols = factored.final_symbols(fst, step_n);
        let is_preimage = factored.is_preimage(fst, step_n, &self.target, &self.sym_to_idx);
        let has_truncated = factored.has_truncated();
        let trunc_output_syms = factored.trunc_output_syms(step_n);

        self.meta[sid_usize] = StateMeta {
            relevant_symbols,
            final_symbols,
            is_preimage,
            has_truncated,
            trunc_output_syms,
        };
        self.meta_computed[sid_usize] = true;
    }

    /// Lazily compute ClassifyResult via universality check.
    pub fn ensure_classify(&mut self, fst: &Fst, sid: u32) {
        let sid_usize = sid as usize;
        if sid_usize < self.classify_computed.len() && self.classify_computed[sid_usize] {
            return;
        }
        self.ensure_meta(fst, sid);
        self.ensure_capacity(sid_usize + 1);

        // Fast path: when all FST states with non-eps input are ip-universal
        // (e.g. BPE), universality reduces to finality — no BFS/cache needed.
        if self.all_final_universal {
            let relevant = &self.meta[sid_usize].relevant_symbols;
            let final_syms = &self.meta[sid_usize].final_symbols;
            let mut quotient_sym: Option<u16> = None;
            let mut remainder_syms: Vec<u16> = Vec::new();
            for &y_idx in relevant {
                if final_syms.contains(&y_idx) {
                    if quotient_sym.is_none() {
                        quotient_sym = Some(y_idx);
                    } else {
                        remainder_syms.push(y_idx);
                    }
                }
            }
            self.classify[sid_usize] = ClassifyResult { quotient_sym, remainder_syms };
            self.classify_computed[sid_usize] = true;
            return;
        }

        let relevant = self.meta[sid_usize].relevant_symbols.clone();
        let final_syms_set: FxHashSet<u16> =
            self.meta[sid_usize].final_symbols.iter().copied().collect();
        let mut factored = self.arena.sets[sid_usize].clone();
        let step_n = self.step_n;
        factored.normalize_for_step(step_n);

        let sym_to_idx = self.sym_to_idx.clone();
        let target = self.target.clone();
        let num_source_symbols = self.num_source_symbols;

        // Create needed universality filters (needs &self.ip_universal_states)
        for &y_idx in &relevant {
            if !self.univ_filters.contains_key(&y_idx) {
                self.univ_filters.insert(
                    y_idx,
                    PeekabooUniversalityFilter::new(
                        fst, step_n, y_idx, &self.ip_universal_states,
                    ),
                );
            }
        }

        // Take filters out to avoid borrow conflicts with arena/eps_cache
        let mut univ_filters = std::mem::take(&mut self.univ_filters);
        let nfa = PeekabooNFAMapped::new(fst, &target, step_n, &sym_to_idx);

        let mut quotient_sym: Option<u16> = None;
        let mut remainder_syms = Vec::new();

        for &y_idx in &relevant {
            if quotient_sym.is_some() {
                if final_syms_set.contains(&y_idx) {
                    remainder_syms.push(y_idx);
                }
                continue;
            }

            // Use factored projection: O(|core| + |closure|) per symbol
            let projected = factored.project_for_symbol(y_idx, step_n);

            let cache_hit = if !projected.is_empty() {
                let all_frontier = projected.iter().all(|&packed| {
                    let (_, buf_len, extra_sym, _) = unpack_peekaboo(packed);
                    buf_len == step_n + 1 && extra_sym == y_idx
                });
                if all_frontier {
                    let mut fst_states: Vec<u32> = projected
                        .iter()
                        .map(|&packed| (packed >> 32) as u32)
                        .collect();
                    fst_states.sort_unstable();
                    fst_states.dedup();
                    self.fst_univ_cache
                        .get(&fst_states)
                        .copied()
                        .map(|result| (result, fst_states))
                } else {
                    None
                }
            } else {
                None
            };

            let is_univ = if let Some((cached_result, _)) = cache_hit {
                let filter_mut = univ_filters.get_mut(&y_idx).unwrap();
                if cached_result {
                    filter_mut.add_pos(&projected);
                } else {
                    filter_mut.add_neg(&projected);
                }
                cached_result
            } else {
                let filter_mut = univ_filters.get_mut(&y_idx).unwrap();
                let result = filter_mut.is_universal_from_projected(
                    projected.clone(), y_idx, &nfa,
                    &mut self.arena, &mut self.eps_cache,
                    num_source_symbols, step_n,
                );
                self.ensure_capacity(self.arena.len());

                if !projected.is_empty() {
                    let all_frontier = projected.iter().all(|&packed| {
                        let (_, buf_len, extra_sym, _) = unpack_peekaboo(packed);
                        buf_len == step_n + 1 && extra_sym == y_idx
                    });
                    if all_frontier {
                        let mut fst_states: Vec<u32> = projected
                            .iter()
                            .map(|&packed| (packed >> 32) as u32)
                            .collect();
                        fst_states.sort_unstable();
                        fst_states.dedup();
                        self.fst_univ_cache.insert(fst_states, result);
                    }
                }
                result
            };

            if is_univ {
                quotient_sym = Some(y_idx);
            } else if final_syms_set.contains(&y_idx) {
                remainder_syms.push(y_idx);
            }
        }

        self.univ_filters = univ_filters;

        self.classify[sid_usize] = ClassifyResult { quotient_sym, remainder_syms };
        self.classify_computed[sid_usize] = true;
    }

    /// Lazily compute DFA arcs from a state.
    pub fn ensure_arcs(&mut self, fst: &Fst, sid: u32) {
        let sid_usize = sid as usize;
        if sid_usize < self.arcs_computed.len() && self.arcs_computed[sid_usize] {
            return;
        }
        self.ensure_capacity(sid_usize + 1);

        let mut factored = self.arena.sets[sid_usize].clone();
        let sym_to_idx = self.sym_to_idx.clone();
        let target = self.target.clone();
        let step_n = self.step_n;
        factored.normalize_for_step(step_n);

        let nfa = PeekabooNFAMapped::new(fst, &target, step_n, &sym_to_idx);

        let mut interned_arcs: Vec<(u32, u32)>;
        if self.use_factored {
            let all_arcs = nfa.compute_all_arcs_factored(
                &factored, &mut self.eps_cache,
                &mut self.fst_full_cache, &mut self.fst_trunc_cache,
            );
            interned_arcs = Vec::with_capacity(all_arcs.len());
            for (x, succ_factored) in all_arcs {
                let any_final = succ_factored.any_final(&nfa);
                let dest_id = self.arena.intern(succ_factored, any_final);
                self.ensure_capacity(self.arena.len());
                interned_arcs.push((x, dest_id));
            }
        } else {
            let all_arcs = nfa.compute_all_arcs_cached(
                &factored.core, &mut self.eps_cache,
                &mut self.fst_full_cache, &mut self.fst_trunc_cache,
            );
            interned_arcs = Vec::with_capacity(all_arcs.len());
            for (x, succ_flat) in all_arcs {
                let mut sorted = succ_flat;
                sorted.sort_unstable();
                sorted.dedup();
                let any_final = sorted.iter().any(|&p| {
                    let (fst_state, _, _, _) = unpack_peekaboo(p);
                    nfa.fst.is_final[fst_state as usize]
                });
                let dest_id = self.arena.intern_flat(sorted, any_final);
                self.ensure_capacity(self.arena.len());
                interned_arcs.push((x, dest_id));
            }
        }

        self.arcs_from[sid_usize] = interned_arcs;
        self.arcs_computed[sid_usize] = true;
    }

    /// Return the arcs for a state.
    pub fn get_arcs(&mut self, fst: &Fst, sid: u32) -> Vec<(u32, u32)> {
        self.ensure_arcs(fst, sid);
        self.arcs_from[sid as usize].clone()
    }

    /// Return arcs from a state (same as get_arcs).
    pub fn get_arcs_expanded(&mut self, fst: &Fst, sid: u32) -> Vec<(u32, u32)> {
        self.get_arcs(fst, sid)
    }

    /// Follow a single arc labeled `x` from `sid`. Lazily computes arcs.
    pub fn step(&mut self, fst: &Fst, sid: u32, x: u32) -> Option<u32> {
        self.ensure_arcs(fst, sid);
        let sid_usize = sid as usize;
        for &(lbl, dst) in &self.arcs_from[sid_usize] {
            if lbl == x {
                return Some(dst);
            }
        }
        None
    }

    /// Run a source path from the start state, returning the reached
    /// DFA state or `None` if any arc is missing. Lazily computes arcs.
    pub fn run(&mut self, fst: &Fst, source_path: &[u32]) -> Option<u32> {
        if self.start_ids.is_empty() {
            return None;
        }
        let mut state = self.start_ids[0];
        for &x in source_path {
            state = self.step(fst, state, x)?;
        }
        Some(state)
    }

    /// Compute the DFA destination for a single input symbol from a state.
    ///
    /// If arcs have already been computed for `sid`, this is a simple lookup.
    /// Otherwise, it computes only the arc for symbol `x` without materializing
    /// all other arcs — O(|NFA_set|) instead of O(|V| × |closure|).
    pub fn single_arc(&mut self, fst: &Fst, sid: u32, x: u32) -> Option<u32> {
        let sid_usize = sid as usize;

        // Fast path: arcs already computed
        if sid_usize < self.arcs_computed.len() && self.arcs_computed[sid_usize] {
            return self.arcs_from[sid_usize].iter()
                .find(|&&(lbl, _)| lbl == x)
                .map(|&(_, dst)| dst);
        }

        self.ensure_capacity(sid_usize + 1);

        // Slow path: compute single arc from NFA set
        let mut factored = self.arena.sets[sid_usize].clone();
        let sym_to_idx = self.sym_to_idx.clone();
        let target = self.target.clone();
        let step_n = self.step_n;
        factored.normalize_for_step(step_n);

        let nfa = PeekabooNFAMapped::new(fst, &target, step_n, &sym_to_idx);
        let result = nfa.compute_single_arc_factored(
            &factored, x, &mut self.eps_cache,
            &mut self.fst_full_cache, &mut self.fst_trunc_cache,
        );

        match result {
            None => None,
            Some(succ_factored) => {
                let any_final = succ_factored.any_final(&nfa);
                let dest_id = if self.use_factored {
                    self.arena.intern(succ_factored, any_final)
                } else {
                    let mut flat = succ_factored.flatten();
                    flat.sort_unstable();
                    flat.dedup();
                    self.arena.intern_flat(flat, any_final)
                };
                self.ensure_capacity(self.arena.len());
                Some(dest_id)
            }
        }
    }

    pub fn get_classify(&mut self, fst: &Fst, sid: u32) -> &ClassifyResult {
        self.ensure_classify(fst, sid);
        &self.classify[sid as usize]
    }

    pub fn get_meta(&mut self, fst: &Fst, sid: u32) -> &StateMeta {
        self.ensure_meta(fst, sid);
        &self.meta[sid as usize]
    }

    pub fn arena_sets(&self, sid: u32) -> &FactoredNfaSet {
        &self.arena.sets[sid as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_peekaboo() {
        let cases = vec![
            (0, 0, NO_EXTRA, false),
            (100, 50, 42, true),
            (u32::MAX >> 1, 32767, NO_EXTRA, true),
            (0, 0, 0, false),
            (1, 1, 1, true),
        ];
        for (fs, bl, es, tr) in cases {
            let packed = pack_peekaboo(fs, bl, es, tr);
            let (fs2, bl2, es2, tr2) = unpack_peekaboo(packed);
            assert_eq!(fs, fs2, "fst_state mismatch");
            assert_eq!(bl, bl2, "buf_len mismatch");
            assert_eq!(es, es2, "extra_sym mismatch");
            assert_eq!(tr, tr2, "truncated mismatch");
        }
    }

}
