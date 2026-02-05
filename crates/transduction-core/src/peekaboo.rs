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

use crate::fst::{compute_ip_universal_states, Fst, EPSILON};
use crate::powerset::PowersetArena;
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
fn unpack_peekaboo(packed: u64) -> (u32, u16, u16, bool) {
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

    /// Is an NFA state "productive"?
    #[inline]
    fn is_productive(&self, packed: u64) -> bool {
        let (fst_state, buf_len, extra_sym, _truncated) = unpack_peekaboo(packed);
        self.fst.has_non_eps_input[fst_state as usize]
            || (self.fst.is_final[fst_state as usize]
                && buf_len == self.step_n + 1
                && extra_sym != NO_EXTRA)
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
            assert!(!truncated);
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
    fn eps_closure_single(&self, state: u64, cache: &mut FxHashMap<u64, Vec<u64>>) -> Vec<u64> {
        if let Some(cached) = cache.get(&state) {
            return cached.clone();
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

        cache.insert(state, result.clone());
        result
    }

    fn eps_closure_set(&self, states: &[u64], cache: &mut FxHashMap<u64, Vec<u64>>) -> Vec<u64> {
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
    fn compute_all_arcs(
        &self,
        states: &[u64],
        cache: &mut FxHashMap<u64, Vec<u64>>,
    ) -> Vec<(u32, Vec<u64>)> {
        let mut by_symbol: FxHashMap<u32, Vec<u64>> = FxHashMap::default();

        for &packed in states {
            for (x, dest) in self.arcs(packed) {
                if x != EPSILON {
                    let closure = self.eps_closure_single(dest, cache);
                    let bucket = by_symbol.entry(x).or_default();
                    bucket.extend_from_slice(&closure);
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
        eps_cache: &mut FxHashMap<u64, Vec<u64>>,
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
        eps_cache: &mut FxHashMap<u64, Vec<u64>>,
        num_source_symbols: usize,
        step_n: u16,
    ) -> bool {
        let any_final = projected_set.iter().any(|&s| nfa.is_final(s));
        let start_id = arena.intern(projected_set.to_vec(), any_final);

        if !arena.is_final[start_id as usize] {
            return false;
        }

        let mut sub_visited: FxHashSet<u32> = FxHashSet::default();
        let mut sub_worklist: VecDeque<u32> = VecDeque::new();

        sub_visited.insert(start_id);
        sub_worklist.push_back(start_id);

        while let Some(cur) = sub_worklist.pop_front() {
            if !arena.is_final[cur as usize] {
                return false;
            }

            let cur_set = arena.sets[cur as usize].clone();
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
// Main algorithm
// ---------------------------------------------------------------------------

pub fn peekaboo_decompose(fst: &Fst, target: &[u32]) -> PeekabooResult {
    let total_start = Instant::now();
    let target_len = target.len();

    let init_start = Instant::now();

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

    let init_ms = init_start.elapsed().as_secs_f64() * 1000.0;

    let mut arena = PowersetArena::new();
    let num_source_symbols = fst.source_alphabet.len();
    let mut eps_cache: FxHashMap<u64, Vec<u64>> = FxHashMap::default();

    // Profiling accumulators
    let mut per_step_visited: Vec<u32> = Vec::new();
    let mut per_step_frontier_size: Vec<u32> = Vec::new();
    let mut total_bfs_visited: u64 = 0;
    let mut compute_arcs_ms: f64 = 0.0;
    let mut compute_arcs_calls: u64 = 0;
    let mut intern_ms: f64 = 0.0;
    let mut intern_calls: u64 = 0;
    let mut universal_ms: f64 = 0.0;
    let mut universal_calls: u64 = 0;
    let mut universal_true: u64 = 0;
    let mut universal_false: u64 = 0;
    let mut eps_cache_clears: u32 = 0;

    // Per-symbol decomp data (only the LAST step's results matter).
    let mut decomp_q: FxHashMap<u16, Vec<u32>> = FxHashMap::default();
    let mut decomp_r: FxHashMap<u16, Vec<u32>> = FxHashMap::default();
    let mut resume_frontiers: FxHashMap<u16, FxHashSet<u32>> = FxHashMap::default();
    let mut univ_filters: FxHashMap<u16, PeekabooUniversalityFilter> = FxHashMap::default();

    // Merged incoming arcs across all steps.
    let mut merged_incoming: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();

    let mut global_start_id: u32 = 0;

    let bfs_start = Instant::now();

    for step in 0..=target_len {
        let step_n = step as u16;

        let nfa = PeekabooNFAMapped::new(fst, target, step_n, &sym_to_idx);

        // Must clear eps_cache because NFA arcs change with step_n.
        eps_cache.clear();
        eps_cache_clears += 1;

        let mut worklist: VecDeque<u32> = VecDeque::new();
        let mut incoming: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
        let mut step_visited: u32 = 0;

        if step == 0 {
            let raw_starts = nfa.start_states();
            let init_closed = nfa.eps_closure_set(&raw_starts, &mut eps_cache);
            let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
            let start_id = arena.intern(init_closed, any_final);
            global_start_id = start_id;

            worklist.push_back(start_id);
            incoming.insert(start_id, Vec::new());
            per_step_frontier_size.push(1);
        } else {
            // Resume from previous step's frontiers.
            let prev_target_sym = target[step - 1];
            let prev_y_idx = sym_to_idx[&prev_target_sym];

            let mut frontier_sz: u32 = 0;
            if let Some(frontier) = resume_frontiers.remove(&prev_y_idx) {
                frontier_sz = frontier.len() as u32;
                for sid in frontier {
                    worklist.push_back(sid);
                    incoming.insert(sid, Vec::new());
                }
            }
            per_step_frontier_size.push(frontier_sz);
        }

        // Reset per-symbol data for this step.
        decomp_q.clear();
        decomp_r.clear();
        resume_frontiers.clear();
        univ_filters.clear();

        // BFS for this step.
        while let Some(sid) = worklist.pop_front() {
            step_visited += 1;
            let nfa_set = arena.sets[sid as usize].clone();

            // Find relevant symbols.
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

            for &y_idx in &relevant_syms {
                if !univ_filters.contains_key(&y_idx) {
                    univ_filters.insert(
                        y_idx,
                        PeekabooUniversalityFilter::new(
                            fst, step_n, y_idx, &ip_universal_states,
                        ),
                    );
                }

                if continuous.is_none() {
                    let uni_start = Instant::now();
                    universal_calls += 1;
                    let filter = univ_filters.get_mut(&y_idx).unwrap();
                    let is_univ = filter.is_universal(
                        &nfa_set,
                        y_idx,
                        &nfa,
                        &mut arena,
                        &mut eps_cache,
                        num_source_symbols,
                        step_n,
                    );
                    universal_ms += uni_start.elapsed().as_secs_f64() * 1000.0;

                    if is_univ {
                        universal_true += 1;
                        decomp_q.entry(y_idx).or_default().push(sid);
                        continuous = Some(y_idx);
                        continue;
                    } else {
                        universal_false += 1;
                    }
                }

                let filter = univ_filters.get(&y_idx).unwrap();
                if filter.is_projected_final(&nfa_set, y_idx, fst, step_n) {
                    decomp_r.entry(y_idx).or_default().push(sid);
                }
            }

            if continuous.is_some() {
                continue;
            }

            // Expand arcs.
            let arcs_start = Instant::now();
            let all_arcs = nfa.compute_all_arcs(&nfa_set, &mut eps_cache);
            compute_arcs_ms += arcs_start.elapsed().as_secs_f64() * 1000.0;
            compute_arcs_calls += 1;

            for (x, successor) in all_arcs {
                let intern_start = Instant::now();
                let succ_final = successor.iter().any(|&s| nfa.is_final(s));
                let dest_id = arena.intern(successor.clone(), succ_final);
                intern_ms += intern_start.elapsed().as_secs_f64() * 1000.0;
                intern_calls += 1;

                if !incoming.contains_key(&dest_id) {
                    worklist.push_back(dest_id);
                    incoming.insert(dest_id, Vec::new());
                }

                incoming.get_mut(&dest_id).unwrap().push((x, sid));

                // Check for truncation boundary.
                let sid_has_truncated = nfa_set.iter().any(|&packed| {
                    let (_, _, _, truncated) = unpack_peekaboo(packed);
                    truncated
                });

                if !sid_has_truncated {
                    let dest_set = &arena.sets[dest_id as usize];
                    for &packed in dest_set {
                        let (_, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);
                        if truncated && extra_sym != NO_EXTRA {
                            let prefix_len = if buf_len > 0 { buf_len - 1 } else { 0 };
                            let eff_extra = if prefix_len >= step_n {
                                extra_sym
                            } else {
                                continue;
                            };
                            if !univ_filters.contains_key(&eff_extra) {
                                univ_filters.insert(
                                    eff_extra,
                                    PeekabooUniversalityFilter::new(
                                        fst, step_n, eff_extra, &ip_universal_states,
                                    ),
                                );
                            }
                            resume_frontiers.entry(eff_extra).or_default().insert(sid);
                        }
                    }
                }
            }
        }

        per_step_visited.push(step_visited);
        total_bfs_visited += step_visited as u64;

        // Q and R states that are non-truncated also sit on the boundary.
        for (&y_idx, q_states) in &decomp_q {
            for &sid in q_states {
                let nfa_set = &arena.sets[sid as usize];
                let has_truncated = nfa_set.iter().any(|&packed| {
                    let (_, _, _, truncated) = unpack_peekaboo(packed);
                    truncated
                });
                if !has_truncated {
                    resume_frontiers.entry(y_idx).or_default().insert(sid);
                }
            }
        }
        for (&y_idx, r_states) in &decomp_r {
            for &sid in r_states {
                let nfa_set = &arena.sets[sid as usize];
                let has_truncated = nfa_set.iter().any(|&packed| {
                    let (_, _, _, truncated) = unpack_peekaboo(packed);
                    truncated
                });
                if !has_truncated {
                    resume_frontiers.entry(y_idx).or_default().insert(sid);
                }
            }
        }

        // Merge incoming into global.
        for (state, arcs) in incoming {
            let entry = merged_incoming.entry(state).or_default();
            for arc in arcs {
                if !entry.contains(&arc) {
                    entry.push(arc);
                }
            }
        }
    }

    let bfs_ms = bfs_start.elapsed().as_secs_f64() * 1000.0;

    // Compute arena stats.
    let arena_size = arena.len() as u32;
    let mut max_powerset_size: usize = 0;
    let total_nfa_states: usize = arena.sets.iter().map(|s| {
        let sz = s.len();
        if sz > max_powerset_size {
            max_powerset_size = sz;
        }
        sz
    }).sum();
    let avg_powerset_size = if arena_size > 0 {
        total_nfa_states as f64 / arena_size as f64
    } else {
        0.0
    };

    // Merged incoming stats.
    let merged_incoming_states = merged_incoming.len() as u32;
    let merged_incoming_arcs: u64 = merged_incoming.values().map(|v| v.len() as u64).sum();

    // Extract trimmed FSAs via backward BFS from Q/R stops.
    let extract_start = Instant::now();
    let start_id = global_start_id;

    let mut per_symbol_result: FxHashMap<u32, (FsaResult, FsaResult)> = FxHashMap::default();
    let mut per_symbol_q_stops: Vec<(u32, u32)> = Vec::new();
    let mut per_symbol_r_stops: Vec<(u32, u32)> = Vec::new();

    for &sym in &output_alphabet {
        let y_idx = sym_to_idx[&sym];

        let q_stops: Vec<u32> = decomp_q.get(&y_idx).cloned().unwrap_or_default();
        let r_stops: Vec<u32> = decomp_r.get(&y_idx).cloned().unwrap_or_default();

        per_symbol_q_stops.push((sym, q_stops.len() as u32));
        per_symbol_r_stops.push((sym, r_stops.len() as u32));

        let q_fsa = trimmed_fsa(start_id, &q_stops, &merged_incoming);
        let r_fsa = trimmed_fsa(start_id, &r_stops, &merged_incoming);

        per_symbol_result.insert(sym, (q_fsa, r_fsa));
    }

    let extract_ms = extract_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    PeekabooResult {
        per_symbol: per_symbol_result,
        stats: PeekabooProfileStats {
            total_ms,
            init_ms,
            bfs_ms,
            extract_ms,
            num_steps: (target_len + 1) as u32,
            per_step_visited,
            per_step_frontier_size,
            total_bfs_visited,
            compute_arcs_ms,
            compute_arcs_calls,
            intern_ms,
            intern_calls,
            universal_ms,
            universal_calls,
            universal_true,
            universal_false,
            arena_size,
            max_powerset_size,
            avg_powerset_size,
            merged_incoming_states,
            merged_incoming_arcs,
            eps_cache_clears,
            per_symbol_q_stops,
            per_symbol_r_stops,
        },
    }
}

/// Build a trimmed FSA by backward BFS from stop states through the incoming graph.
fn trimmed_fsa(
    start_id: u32,
    stop_ids: &[u32],
    incoming: &FxHashMap<u32, Vec<(u32, u32)>>,
) -> FsaResult {
    if stop_ids.is_empty() {
        return FsaResult {
            num_states: 0,
            start: Vec::new(),
            stop: Vec::new(),
            arc_src: Vec::new(),
            arc_lbl: Vec::new(),
            arc_dst: Vec::new(),
        };
    }

    let mut backward_reachable: FxHashSet<u32> = FxHashSet::default();
    let mut worklist: VecDeque<u32> = VecDeque::new();

    for &s in stop_ids {
        if backward_reachable.insert(s) {
            worklist.push_back(s);
        }
    }

    while let Some(state) = worklist.pop_front() {
        if let Some(arcs) = incoming.get(&state) {
            for &(_x, pred) in arcs {
                if backward_reachable.insert(pred) {
                    worklist.push_back(pred);
                }
            }
        }
    }

    let mut arc_src = Vec::new();
    let mut arc_lbl = Vec::new();
    let mut arc_dst = Vec::new();

    let mut state_map: FxHashMap<u32, u32> = FxHashMap::default();
    let mut next_id: u32 = 0;

    let get_id = |state: u32, map: &mut FxHashMap<u32, u32>, next: &mut u32| -> u32 {
        if let Some(&id) = map.get(&state) {
            id
        } else {
            let id = *next;
            *next += 1;
            map.insert(state, id);
            id
        }
    };

    for &state in &backward_reachable {
        if let Some(arcs) = incoming.get(&state) {
            for &(x, pred) in arcs {
                if backward_reachable.contains(&pred) {
                    let src = get_id(pred, &mut state_map, &mut next_id);
                    let dst = get_id(state, &mut state_map, &mut next_id);
                    arc_src.push(src);
                    arc_lbl.push(x);
                    arc_dst.push(dst);
                }
            }
        }
    }

    let start: Vec<u32> = if backward_reachable.contains(&start_id) {
        vec![get_id(start_id, &mut state_map, &mut next_id)]
    } else {
        Vec::new()
    };

    let stop: Vec<u32> = stop_ids
        .iter()
        .filter(|s| backward_reachable.contains(s))
        .map(|&s| get_id(s, &mut state_map, &mut next_id))
        .collect();

    let num_states = next_id;

    FsaResult {
        num_states,
        start,
        stop,
        arc_src,
        arc_lbl,
        arc_dst,
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

    #[test]
    fn test_simple_replace_peekaboo() {
        let fst = Fst::new(
            1,
            vec![0],
            &[0],
            &[0, 0],
            &[1, 2],
            &[10, 11],
            &[0, 0],
            vec![1, 2],
        );

        let result = peekaboo_decompose(&fst, &[]);

        assert!(result.per_symbol.contains_key(&10));
        assert!(result.per_symbol.contains_key(&11));

        let (q10, r10) = &result.per_symbol[&10];
        let (q11, r11) = &result.per_symbol[&11];

        assert!(!q10.stop.is_empty(), "Q for symbol 10 should have stops");
        assert!(!q11.stop.is_empty(), "Q for symbol 11 should have stops");
        assert!(r10.stop.is_empty(), "R for symbol 10 should be empty");
        assert!(r11.stop.is_empty(), "R for symbol 11 should be empty");
    }
}
