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
use crate::powerset::PowersetArena;
use crate::rho::RHO;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// NFA state packing
// ---------------------------------------------------------------------------

const NO_EXTRA: u16 = 0xFFFF;

#[inline]
fn pack_peekaboo(fst_state: u32, buf_len: u16, extra_sym: u16, truncated: bool) -> u64 {
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
// PeekabooNFA
// ---------------------------------------------------------------------------

/// PeekabooPrecover NFA for a given step.
///
/// This mirrors Python's `PeekabooPrecover(fst, target[:step_n])` with K=1.
/// NFA state encoding is step-independent (same buffer → same packed u64).
struct PeekabooNFAMapped<'a> {
    fst: &'a Fst,
    full_target: &'a [u32],
    step_n: u16,
    sym_to_idx: &'a FxHashMap<u32, u16>,
}

impl<'a> PeekabooNFAMapped<'a> {
    fn new(
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

    fn start_states(&self) -> Vec<u64> {
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
    fn is_final(&self, packed: u64) -> bool {
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
    fn effective_state(
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
    fn arcs(&self, packed: u64) -> Vec<(u32, u64)> {
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
            for arc in self.fst.arcs_from(i) {
                let x = arc.input;
                let y = arc.output;
                let j = arc.dest;

                if y == EPSILON {
                    // Epsilon output: buffer unchanged.
                    result.push((x, pack_peekaboo(j, eff_n, NO_EXTRA, false)));
                } else if y == self.full_target[eff_n as usize] {
                    // Output matches next target symbol: extend buffer.
                    result.push((x, pack_peekaboo(j, eff_n + 1, NO_EXTRA, false)));
                }
                // else: output doesn't match → dead end.
            }
        }

        result
    }

    /// Epsilon-closure of a single NFA state (cached).
    /// Cache stores (closure, max_buf_len) for O(1) eviction on prefix extension.
    fn eps_closure_single(&self, state: u64, cache: &mut FxHashMap<u64, (Vec<u64>, u16)>) -> Vec<u64> {
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
            for (x, dest) in self.arcs(s) {
                if x == EPSILON && seen.insert(dest) {
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

    fn eps_closure_set(&self, states: &[u64], cache: &mut FxHashMap<u64, (Vec<u64>, u16)>) -> Vec<u64> {
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
    fn compute_all_arcs(
        &self,
        states: &[u64],
        cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
    ) -> Vec<(u32, Vec<u64>)> {
        let mut by_symbol: FxHashMap<u32, Vec<u64>> = FxHashMap::default();

        // FST-level closure caches for off-target states (keyed by FST state).
        // fst_full_cache: non-truncated off-target dest -> (productive_non_trunc, trunc)
        // fst_trunc_cache: truncated dest -> all reachable FST states
        let mut fst_full_cache: FxHashMap<u32, (Vec<u32>, Vec<u32>)> = FxHashMap::default();
        let mut fst_trunc_cache: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
        let step_n = self.step_n;

        for &packed in states {
            for (x, dest) in self.arcs(packed) {
                if x != EPSILON {
                    let (dest_fst, dest_bl, dest_es, dest_trunc) = unpack_peekaboo(dest);

                    if dest_es != NO_EXTRA && dest_bl > step_n {
                        if !dest_trunc {
                            // Fast path: non-truncated off-target dest.
                            // FST-level closure is (non_trunc_productive, trunc) by FST state.
                            let entry = fst_full_cache
                                .entry(dest_fst)
                                .or_insert_with(|| self.fst_full_closure(dest_fst));
                            let bucket = by_symbol.entry(x).or_default();
                            for &r in entry.0.iter() {
                                bucket.push(pack_peekaboo(r, dest_bl, dest_es, false));
                            }
                            for &t in entry.1.iter() {
                                bucket.push(pack_peekaboo(t, dest_bl, dest_es, true));
                            }
                        } else {
                            // Fast path: truncated off-target dest.
                            let entry = fst_trunc_cache
                                .entry(dest_fst)
                                .or_insert_with(|| self.fst_trunc_closure(dest_fst));
                            let bucket = by_symbol.entry(x).or_default();
                            for &t in entry.iter() {
                                bucket.push(pack_peekaboo(t, dest_bl, dest_es, true));
                            }
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
        // Phase 1: non-truncated reachable via eps-input/eps-output arcs
        let mut non_trunc: FxHashSet<u32> = FxHashSet::default();
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
        let mut trunc: FxHashSet<u32> = FxHashSet::default();
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
        // (for off-target with buf_len == step_n + 1, extra_sym != NO_EXTRA)
        let mut non_trunc_productive: Vec<u32> = non_trunc
            .into_iter()
            .filter(|&r| {
                self.fst.has_non_eps_input[r as usize]
                    || self.fst.is_final[r as usize]
            })
            .collect();
        non_trunc_productive.sort_unstable();

        // All truncated states are productive
        let mut trunc_result: Vec<u32> = trunc.into_iter().collect();
        trunc_result.sort_unstable();

        (non_trunc_productive, trunc_result)
    }

    /// FST-level epsilon closure for a truncated starting state.
    /// Follows ALL eps-input arcs (any output). All reachable states are productive.
    fn fst_trunc_closure(&self, fst_state: u32) -> Vec<u32> {
        let mut visited: FxHashSet<u32> = FxHashSet::default();
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
        let mut result: Vec<u32> = visited.into_iter().collect();
        result.sort_unstable();
        result
    }
}

// ---------------------------------------------------------------------------
// Universality filter for peekaboo (per target-symbol)
// ---------------------------------------------------------------------------

struct PeekabooUniversalityFilter {
    witnesses: FxHashSet<u64>,
    pos_index: FxHashMap<u64, Vec<u32>>,
    pos_sizes: Vec<usize>,
    neg_index: FxHashMap<u64, Vec<u32>>,
    neg_next: u32,
}

impl PeekabooUniversalityFilter {
    fn new(_fst: &Fst, step_n: u16, y_idx: u16, ip_universal_states: &[bool]) -> Self {
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

    fn add_pos(&mut self, nfa_set: &[u64]) {
        let eid = self.pos_sizes.len() as u32;
        self.pos_sizes.push(nfa_set.len());
        for &e in nfa_set {
            self.pos_index.entry(e).or_default().push(eid);
        }
    }

    fn add_neg(&mut self, nfa_set: &[u64]) {
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
    fn project_and_refine(&self, full_nfa_set: &[u64], y_idx: u16, step_n: u16) -> Vec<u64> {
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

    fn is_universal(
        &mut self,
        full_nfa_set: &[u64],
        y_idx: u16,
        nfa: &PeekabooNFAMapped,
        arena: &mut PowersetArena,
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

    fn bfs_universal(
        &self,
        projected_set: &[u64],
        y_idx: u16,
        nfa: &PeekabooNFAMapped,
        arena: &mut PowersetArena,
        eps_cache: &mut FxHashMap<u64, (Vec<u64>, u16)>,
        num_source_symbols: usize,
        step_n: u16,
    ) -> bool {
        let any_final = projected_set.iter().any(|&s| nfa.is_final(s));
        let start_id = arena.intern(projected_set.to_vec(), any_final);

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
            let cur_set = arena.sets[cur as usize].clone();

            // Compute finality directly from NFA set rather than arena.is_final
            let cur_final = cur_set.iter().any(|&s| nfa.is_final(s));
            if !cur_final {
                return false;
            }

            let all_arcs = nfa.compute_all_arcs(&cur_set, eps_cache);

            if all_arcs.len() < num_source_symbols {
                return false;
            }

            for (_sym, successor) in &all_arcs {
                let projected_succ = self.project_and_refine(successor, y_idx, step_n);

                if projected_succ.is_empty() {
                    return false;
                }

                let succ_final = projected_succ.iter().any(|&s| nfa.is_final(s));
                let dest_id = arena.intern(projected_succ, succ_final);

                if sub_visited.insert(dest_id) {
                    sub_worklist.push_back(dest_id);
                }
            }
        }

        true
    }

    fn is_projected_final(
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
const STATUS_NEW: u8 = 0;       // needs full expansion
const STATUS_INTERIOR: u8 = 1;  // non-final, expanded (has cached arcs)
const STATUS_QSTOP: u8 = 2;     // universal final, no outgoing arcs
const STATUS_RSTOP: u8 = 3;     // non-universal final, expanded (has cached arcs)

/// Evict stale eps_cache entries for a peekaboo prefix extension.
/// An entry is stale if the key NFA state has buf_len >= frontier,
/// or the closure result contains states with buf_len >= frontier.
fn evict_peekaboo_eps_cache(cache: &mut FxHashMap<u64, (Vec<u64>, u16)>, frontier: u16) {
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
    source_alphabet: Vec<u32>,
    sym_to_idx: FxHashMap<u32, u16>,
    idx_to_sym: Vec<u32>,
    ip_universal_states: Vec<bool>,
    num_source_symbols: usize,

    // Persistent DFA structure
    arena: PowersetArena,
    global_start_id: u32,
    arcs_from: Vec<Vec<(u32, u32)>>,       // [sid] → [(label, dest_sid)]; may contain (RHO, dest)
    has_rho: Vec<bool>,                    // [sid] → true if arcs contain a RHO entry
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
}

impl DirtyPeekaboo {
    pub fn new(fst: &Fst) -> Self {
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
        let source_alphabet = fst.source_alphabet.clone();

        DirtyPeekaboo {
            output_alphabet,
            source_alphabet,
            sym_to_idx,
            idx_to_sym,
            ip_universal_states,
            num_source_symbols,
            arena: PowersetArena::new(),
            global_start_id: 0,
            arcs_from: Vec::new(),
            has_rho: Vec::new(),
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
        }
    }

    fn is_prefix_extension(&self, target: &[u32]) -> bool {
        if target.len() <= self.prev_target.len() {
            return false;
        }
        target[..self.prev_target.len()] == self.prev_target[..]
    }

    fn full_reset(&mut self) {
        self.arena = PowersetArena::new();
        self.global_start_id = 0;
        self.arcs_from.clear();
        self.has_rho.clear();
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
            self.has_rho.resize(needed, false);
            self.state_status.resize(needed, STATUS_NEW);
            self.reverse_arcs.resize_with(needed, Vec::new);
            self.max_bufpos.resize(needed, 0);
            for sid in old_len..needed {
                let nfa_set = &self.arena.sets[sid];
                let mbp = nfa_set.iter()
                    .map(|&e| ((e >> 17) & 0x7FFF) as u16)
                    .max()
                    .unwrap_or(0);
                self.max_bufpos[sid] = mbp;
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
                if sid_usize < self.has_rho.len() && self.has_rho[sid_usize] {
                    // Expand RHO arc: emit explicit arcs for all rho-class symbols
                    let explicit_syms: FxHashSet<u32> = self.arcs_from[sid_usize]
                        .iter()
                        .filter(|&&(l, _)| l != RHO)
                        .map(|&(l, _)| l)
                        .collect();
                    let rho_dest = self.arcs_from[sid_usize]
                        .iter()
                        .find(|&&(l, _)| l == RHO)
                        .map(|&(_, d)| d);

                    // Emit non-RHO arcs
                    for &(l, d) in &self.arcs_from[sid_usize] {
                        if l != RHO && (d as usize) < n && backward[d as usize] {
                            arc_src.push(sid);
                            arc_lbl.push(l);
                            arc_dst.push(d);
                        }
                    }
                    // Expand RHO arc for all symbols not in explicit set
                    if let Some(rd) = rho_dest {
                        if (rd as usize) < n && backward[rd as usize] {
                            for &sym in &self.source_alphabet {
                                if !explicit_syms.contains(&sym) {
                                    arc_src.push(sid);
                                    arc_lbl.push(sym);
                                    arc_dst.push(rd);
                                }
                            }
                        }
                    }
                } else {
                    for &(l, d) in &self.arcs_from[sid_usize] {
                        if (d as usize) < n && backward[d as usize] {
                            arc_src.push(sid);
                            arc_lbl.push(l);
                            arc_dst.push(d);
                        }
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

    pub fn arcs_from(&self, sid: u32) -> &[(u32, u32)] {
        let sid_usize = sid as usize;
        if sid_usize < self.arcs_from.len() {
            &self.arcs_from[sid_usize]
        } else {
            &[]
        }
    }

    /// Follow a single arc labeled `x` from `sid`. Returns `Some(dest)` or `None`.
    /// Handles rho-compressed states: if `x` is not found in explicit arcs,
    /// falls back to the rho destination.
    pub fn step(&self, sid: u32, x: u32) -> Option<u32> {
        let sid_usize = sid as usize;
        // Look for explicit arc
        for &(lbl, dst) in self.arcs_from(sid) {
            if lbl == x {
                return Some(dst);
            }
        }
        // Fall back to rho destination
        if sid_usize < self.has_rho.len() && self.has_rho[sid_usize] {
            return self.arcs_from(sid).iter()
                .find(|&&(lbl, _)| lbl == RHO)
                .map(|&(_, dst)| dst);
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

    pub fn arena_sets(&self, sid: u32) -> &[u64] {
        &self.arena.sets[sid as usize]
    }

    /// Whether the arcs for `sid` include a RHO entry (complete state).
    pub fn state_has_rho(&self, sid: u32) -> bool {
        let sid_usize = sid as usize;
        sid_usize < self.has_rho.len() && self.has_rho[sid_usize]
    }

    /// Return (has_rho, rho_dest, explicit_arcs) for a DFA state.
    pub fn rho_arcs(&self, sid: u32) -> (bool, Option<u32>, Vec<(u32, u32)>) {
        let sid_usize = sid as usize;
        if sid_usize >= self.has_rho.len() || !self.has_rho[sid_usize] {
            // Not a rho state — return all arcs as explicit
            return (false, None, self.arcs_from(sid).to_vec());
        }
        let mut explicit = Vec::new();
        let mut rho_dest = None;
        for &(lbl, dst) in self.arcs_from(sid) {
            if lbl == RHO {
                rho_dest = Some(dst);
            } else {
                explicit.push((lbl, dst));
            }
        }
        (true, rho_dest, explicit)
    }

    /// Return the source alphabet.
    pub fn source_alphabet(&self) -> &[u32] {
        &self.source_alphabet
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
            let nfa_set = &self.arena.sets[sid_usize];
            let is_preimage = nfa_set.iter().any(|&packed| {
                let (fst_state, buf_len, extra_sym, _truncated) = unpack_peekaboo(packed);
                if buf_len != step_n || !fst.is_final[fst_state as usize] {
                    return false;
                }
                // Canonical encoding
                if extra_sym == NO_EXTRA {
                    return true;
                }
                // Non-canonical: extra_sym matches target[step_n - 1]
                if let Some(expected) = non_canonical_extra {
                    extra_sym == expected
                } else {
                    false
                }
            });
            if is_preimage {
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
            let nfa_set = &self.arena.sets[sid_usize];

            // Check if this state has any truncated NFA element
            let has_truncated = nfa_set.iter().any(|&packed| {
                let (_fst_state, _buf_len, _extra_sym, truncated) = unpack_peekaboo(packed);
                truncated
            });

            if has_truncated {
                continue;  // Only non-truncated states can be resume frontiers
            }

            // Check successors for truncated elements
            let mut frontier_syms: FxHashSet<u16> = FxHashSet::default();
            for &(_lbl, dst) in &self.arcs_from[sid_usize] {
                let dst_usize = dst as usize;
                if dst_usize < self.arena.sets.len() {
                    for &packed in &self.arena.sets[dst_usize] {
                        let (_fst_state, _buf_len, extra_sym, truncated) = unpack_peekaboo(packed);
                        if truncated && extra_sym != NO_EXTRA {
                            frontier_syms.insert(extra_sym);
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

        // Compute start state
        let raw_starts = nfa.start_states();
        let init_closed = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);
        let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
        let start_id = self.arena.intern(init_closed, any_final);
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

        let bfs_start = Instant::now();

        // BFS loop
        while let Some(sid) = worklist.pop_front() {
            if self.state_status[sid as usize] != STATUS_NEW {
                continue;
            }

            let nfa_set = self.arena.sets[sid as usize].clone();

            // Find relevant symbols
            let mut relevant_syms = FxHashSet::default();
            for &packed in &nfa_set {
                let (_, buf_len, extra_sym, _truncated) = unpack_peekaboo(packed);
                if extra_sym != NO_EXTRA {
                    let (eff_n, eff_extra, is_valid) = nfa.effective_state(buf_len, extra_sym, false);
                    if is_valid && eff_n > step_n && eff_extra != NO_EXTRA {
                        relevant_syms.insert(eff_extra);
                    }
                }
            }

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
                    // Check fst_univ_cache for pure-frontier states
                    let projected = univ_filters.get(&y_idx).unwrap()
                        .project_and_refine(&nfa_set, y_idx, step_n);
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
                        let result = filter.is_universal(
                            &nfa_set,
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

                let filter = univ_filters.get(&y_idx).unwrap();
                if filter.is_projected_final(&nfa_set, y_idx, fst, step_n) {
                    self.decomp_r.entry(y_idx).or_default().push(sid);
                    has_final_syms = true;
                }
            }

            if continuous.is_some() {
                self.state_status[sid as usize] = STATUS_QSTOP;
                continue;
            }

            // Expand arcs
            let all_arcs = nfa.compute_all_arcs(&nfa_set, &mut self.eps_cache);

            // Intern all destinations first
            let mut interned_arcs: Vec<(u32, u32)> = Vec::with_capacity(all_arcs.len());
            let mut unique_dests: FxHashSet<u32> = FxHashSet::default();
            for (x, successor) in all_arcs {
                let succ_final = successor.iter().any(|&s| nfa.is_final(s));
                let dest_id = self.arena.intern(successor, succ_final);

                // Ensure capacity for new states
                let needed = dest_id as usize + 1;
                self.ensure_capacity(needed);
                if needed > self.reachable_flags.len() {
                    self.reachable_flags.resize(needed, false);
                }

                interned_arcs.push((x, dest_id));
                unique_dests.insert(dest_id);
            }

            // Check completeness and apply rho compression
            let is_complete = self.num_source_symbols > 0
                && interned_arcs.len() == self.num_source_symbols;

            if is_complete {
                // Group by destination to find the most common one
                let mut dest_counts: FxHashMap<u32, usize> = FxHashMap::default();
                for &(_, dest) in &interned_arcs {
                    *dest_counts.entry(dest).or_insert(0) += 1;
                }
                let rho_dest = *dest_counts.iter()
                    .max_by_key(|(_, &count)| count)
                    .unwrap()
                    .0;

                if dest_counts.len() == 1 {
                    self.arcs_from[sid as usize] = vec![(RHO, rho_dest)];
                } else {
                    let mut result_arcs: Vec<(u32, u32)> = Vec::new();
                    for &(x, dest) in &interned_arcs {
                        if dest != rho_dest {
                            result_arcs.push((x, dest));
                        }
                    }
                    result_arcs.push((RHO, rho_dest));
                    self.arcs_from[sid as usize] = result_arcs;
                }
                self.has_rho[sid as usize] = true;
            } else {
                for &(x, dest_id) in &interned_arcs {
                    self.arcs_from[sid as usize].push((x, dest_id));
                }
                self.has_rho[sid as usize] = false;
            }

            // Add reverse arcs and enqueue successors for ALL unique destinations
            // (reverse arcs use sid, not the compressed label)
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
    num_source_symbols: usize,

    // Per-step (cleared on new_step)
    target: Vec<u32>,
    step_n: u16,

    // Persistent across steps
    arena: PowersetArena,
    pub fst_univ_cache: FxHashMap<Vec<u32>, bool>,

    // Per-step caches (cleared on new_step)
    eps_cache: FxHashMap<u64, (Vec<u64>, u16)>,
    arcs_computed: Vec<bool>,
    arcs_from: Vec<Vec<(u32, u32)>>,       // may contain (RHO, dest) entries
    has_rho: Vec<bool>,                    // [sid] -> true if arcs contain a RHO entry
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
        num_source_symbols: usize,
    ) -> Self {
        LazyPeekabooDFA {
            sym_to_idx,
            idx_to_sym,
            ip_universal_states,
            num_source_symbols,
            target: Vec::new(),
            step_n: 0,
            arena: PowersetArena::new(),
            fst_univ_cache: FxHashMap::default(),
            eps_cache: FxHashMap::default(),
            arcs_computed: Vec::new(),
            arcs_from: Vec::new(),
            has_rho: Vec::new(),
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
        self.has_rho.clear();
        self.has_rho.resize(n, false);
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
        let start_id = self.arena.intern(init_closed, any_final);
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
            self.has_rho.resize(needed, false);
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

        let nfa_set = self.arena.sets[sid_usize].clone();
        let step_n = self.step_n;
        let target = &self.target;
        let sym_to_idx = &self.sym_to_idx;

        let mut relevant_symbols = Vec::new();
        let mut final_symbols = Vec::new();
        let mut is_preimage = false;
        let mut has_truncated = false;
        let mut trunc_output_syms = Vec::new();
        let mut seen_relevant: FxHashSet<u16> = FxHashSet::default();
        let mut seen_final: FxHashSet<u16> = FxHashSet::default();
        let mut seen_trunc: FxHashSet<u16> = FxHashSet::default();

        for &packed in &nfa_set {
            let (fst_state, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);

            if buf_len == step_n && extra_sym == NO_EXTRA
                && fst.is_final[fst_state as usize]
            {
                is_preimage = true;
            }

            if truncated {
                has_truncated = true;
                if extra_sym != NO_EXTRA {
                    let prefix_len = buf_len - 1;
                    if prefix_len >= step_n && seen_trunc.insert(extra_sym) {
                        trunc_output_syms.push(extra_sym);
                    }
                }
            }

            if extra_sym != NO_EXTRA {
                let prefix_len = buf_len - 1;
                if prefix_len >= step_n {
                    if seen_relevant.insert(extra_sym) {
                        relevant_symbols.push(extra_sym);
                    }
                    if fst.is_final[fst_state as usize]
                        && buf_len == step_n + 1
                        && seen_final.insert(extra_sym)
                    {
                        final_symbols.push(extra_sym);
                    }
                } else if (prefix_len as usize) < target.len() {
                    if let Some(&expected_idx) = sym_to_idx.get(&target[prefix_len as usize]) {
                        if extra_sym == expected_idx
                            && buf_len == step_n
                            && fst.is_final[fst_state as usize]
                        {
                            is_preimage = true;
                        }
                    }
                }
            }
        }

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

        let relevant = self.meta[sid_usize].relevant_symbols.clone();
        let final_syms_set: FxHashSet<u16> =
            self.meta[sid_usize].final_symbols.iter().copied().collect();
        let nfa_set = self.arena.sets[sid_usize].clone();

        let sym_to_idx = self.sym_to_idx.clone();
        let target = self.target.clone();
        let step_n = self.step_n;
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

            let filter = univ_filters.get(&y_idx).unwrap();
            let projected = filter.project_and_refine(&nfa_set, y_idx, step_n);

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
                let result = filter_mut.is_universal(
                    &nfa_set, y_idx, &nfa,
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

    /// Lazily compute DFA arcs from a state, with rho-arc compression.
    ///
    /// After computing all NFA-side arcs, checks if the state is complete
    /// (has arcs for every source symbol).  If so, finds the most common
    /// destination and replaces all arcs to that destination with a single
    /// RHO arc, keeping only the exception arcs explicitly.
    pub fn ensure_arcs(&mut self, fst: &Fst, sid: u32) {
        let sid_usize = sid as usize;
        if sid_usize < self.arcs_computed.len() && self.arcs_computed[sid_usize] {
            return;
        }
        self.ensure_capacity(sid_usize + 1);

        let nfa_set = self.arena.sets[sid_usize].clone();
        let sym_to_idx = self.sym_to_idx.clone();
        let target = self.target.clone();
        let step_n = self.step_n;
        let num_source_symbols = self.num_source_symbols;

        let nfa = PeekabooNFAMapped::new(fst, &target, step_n, &sym_to_idx);
        let all_arcs = nfa.compute_all_arcs(&nfa_set, &mut self.eps_cache);

        // Intern all destinations first
        let mut interned_arcs: Vec<(u32, u32)> = Vec::with_capacity(all_arcs.len());
        for (x, successor) in all_arcs {
            let succ_final = successor.iter().any(|&s| nfa.is_final(s));
            let dest_id = self.arena.intern(successor, succ_final);
            self.ensure_capacity(self.arena.len());
            interned_arcs.push((x, dest_id));
        }

        // Check completeness and apply rho compression
        let is_complete = num_source_symbols > 0
            && interned_arcs.len() == num_source_symbols;

        if is_complete {
            // Group by destination to find the most common one
            let mut dest_counts: FxHashMap<u32, usize> = FxHashMap::default();
            for &(_, dest) in &interned_arcs {
                *dest_counts.entry(dest).or_insert(0) += 1;
            }

            // Find the destination with the most arcs (rho destination)
            let rho_dest = *dest_counts.iter()
                .max_by_key(|(_, &count)| count)
                .unwrap()
                .0;

            if dest_counts.len() == 1 {
                // All arcs go to the same destination — single RHO arc
                self.arcs_from[sid_usize] = vec![(RHO, rho_dest)];
            } else {
                // Store exception arcs + RHO arc
                let mut result_arcs: Vec<(u32, u32)> = Vec::new();
                for &(x, dest) in &interned_arcs {
                    if dest != rho_dest {
                        result_arcs.push((x, dest));
                    }
                }
                result_arcs.push((RHO, rho_dest));
                self.arcs_from[sid_usize] = result_arcs;
            }
            self.has_rho[sid_usize] = true;
        } else {
            // Incomplete state: store all arcs explicitly
            self.arcs_from[sid_usize] = interned_arcs;
            self.has_rho[sid_usize] = false;
        }

        self.arcs_computed[sid_usize] = true;
    }

    /// Return the raw arcs (may include RHO entries) for a state.
    pub fn get_arcs(&mut self, fst: &Fst, sid: u32) -> Vec<(u32, u32)> {
        self.ensure_arcs(fst, sid);
        self.arcs_from[sid as usize].clone()
    }

    /// Return arcs with RHO expanded to explicit arcs for all rho-class symbols.
    /// Used for backward compatibility by callers that don't handle RHO.
    pub fn get_arcs_expanded(&mut self, fst: &Fst, sid: u32) -> Vec<(u32, u32)> {
        self.ensure_arcs(fst, sid);
        let sid_usize = sid as usize;
        if !self.has_rho[sid_usize] {
            return self.arcs_from[sid_usize].clone();
        }
        // Expand RHO
        let mut result = Vec::new();
        let mut explicit_syms = FxHashSet::default();
        let mut rho_dest = None;
        for &(lbl, dst) in &self.arcs_from[sid_usize] {
            if lbl == RHO {
                rho_dest = Some(dst);
            } else {
                explicit_syms.insert(lbl);
                result.push((lbl, dst));
            }
        }
        if let Some(rd) = rho_dest {
            for &sym in &fst.source_alphabet {
                if !explicit_syms.contains(&sym) {
                    result.push((sym, rd));
                }
            }
        }
        result
    }

    /// Follow a single arc labeled `x` from `sid`. Lazily computes arcs.
    /// Handles rho-compressed states: if `x` is not found in explicit arcs,
    /// falls back to the rho destination.
    pub fn step(&mut self, fst: &Fst, sid: u32, x: u32) -> Option<u32> {
        self.ensure_arcs(fst, sid);
        let sid_usize = sid as usize;
        // Look for an explicit arc
        for &(lbl, dst) in &self.arcs_from[sid_usize] {
            if lbl == x {
                return Some(dst);
            }
        }
        // If not found and state has rho, return the rho destination
        if self.has_rho[sid_usize] {
            return self.rho_dest(sid);
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

    pub fn get_classify(&mut self, fst: &Fst, sid: u32) -> &ClassifyResult {
        self.ensure_classify(fst, sid);
        &self.classify[sid as usize]
    }

    pub fn get_meta(&mut self, fst: &Fst, sid: u32) -> &StateMeta {
        self.ensure_meta(fst, sid);
        &self.meta[sid as usize]
    }

    pub fn arena_sets(&self, sid: u32) -> &[u64] {
        &self.arena.sets[sid as usize]
    }

    /// Whether the arcs for `sid` include a RHO entry (complete state).
    pub fn has_rho(&self, sid: u32) -> bool {
        let sid_usize = sid as usize;
        sid_usize < self.has_rho.len() && self.has_rho[sid_usize]
    }

    /// Return the rho destination for `sid`, or None if not a rho state.
    /// Scans arcs for the RHO label.
    pub fn rho_dest(&self, sid: u32) -> Option<u32> {
        let sid_usize = sid as usize;
        if sid_usize >= self.has_rho.len() || !self.has_rho[sid_usize] {
            return None;
        }
        self.arcs_from[sid_usize]
            .iter()
            .find(|&&(lbl, _)| lbl == RHO)
            .map(|&(_, dest)| dest)
    }

    /// Return explicit arcs (non-RHO) for `sid`.
    pub fn explicit_arcs(&self, sid: u32) -> Vec<(u32, u32)> {
        let sid_usize = sid as usize;
        if sid_usize >= self.arcs_from.len() {
            return Vec::new();
        }
        self.arcs_from[sid_usize]
            .iter()
            .filter(|&&(lbl, _)| lbl != RHO)
            .copied()
            .collect()
    }

    /// Return the number of source symbols in the FST's input alphabet.
    pub fn num_source_symbols(&self) -> usize {
        self.num_source_symbols
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
