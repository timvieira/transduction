//! Position-set-quotiented peekaboo DFA for token-decomposable FSTs.
//!
//! Instead of tracking full NFA state sets (O(|fst_states| × target_len)),
//! DFA states are quotiented by "position keys" — `(buf_len, extra_sym, truncated)`
//! tuples that discard the `fst_state` component.  For token-decomposable FSTs
//! (BPE, PTB), states sharing the same position-key set have identical finality
//! and universality, so this quotienting is exact.
//!
//! Result: ~45 DFA states for BPE vs ~7,000 with the generic approach, while
//! running at Rust speed (~1ms/step vs ~3s for the pure-Python version).

use crate::fst::{compute_ip_universal_states, Fst, EPSILON};
use crate::peekaboo::{
    pack_peekaboo, unpack_peekaboo, evict_peekaboo_eps_cache,
    PeekabooNFAMapped, PeekabooUniversalityFilter, NO_EXTRA,
};
use crate::powerset::PowersetArena;
use crate::token_decompose::{extract_token_bytes, ByteTrie};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Position-key extraction
// ---------------------------------------------------------------------------

/// Extract position key from a packed NFA state: discard fst_state, keep
/// (buf_len, extra_sym, truncated) packed into the lower 32 bits.
#[inline]
fn pos_key(packed: u64) -> u32 {
    (packed & 0xFFFFFFFF) as u32
}

/// Extract sorted, deduped position keys from an NFA state set.
fn extract_pos_keys(nfa_set: &[u64]) -> Vec<u32> {
    let mut keys: Vec<u32> = nfa_set.iter().map(|&p| pos_key(p)).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

// ---------------------------------------------------------------------------
// Position-key arena (interning Vec<u32> → u32 state IDs)
// ---------------------------------------------------------------------------

struct PosKeyArena {
    map: FxHashMap<Vec<u32>, u32>,
    single_map: FxHashMap<u32, u32>,
    sets: Vec<Vec<u32>>,
}

impl PosKeyArena {
    fn new() -> Self {
        PosKeyArena {
            map: FxHashMap::default(),
            single_map: FxHashMap::default(),
            sets: Vec::new(),
        }
    }

    fn intern(&mut self, sorted_keys: Vec<u32>) -> u32 {
        if sorted_keys.len() == 1 {
            let key = sorted_keys[0];
            if let Some(&id) = self.single_map.get(&key) {
                return id;
            }
            let id = self.sets.len() as u32;
            self.sets.push(sorted_keys);
            self.single_map.insert(key, id);
            return id;
        }
        if let Some(&id) = self.map.get(&sorted_keys) {
            return id;
        }
        let id = self.sets.len() as u32;
        self.map.insert(sorted_keys.clone(), id);
        self.sets.push(sorted_keys);
        id
    }

    fn len(&self) -> usize {
        self.sets.len()
    }
}

// ---------------------------------------------------------------------------
// Classification result per DFA state
// ---------------------------------------------------------------------------

pub(crate) struct TokenClassify {
    pub(crate) quotient_sym: Option<u16>,
    pub(crate) remainder_syms: Vec<u16>,
    pub(crate) is_preimage: bool,
    pub(crate) has_truncated: bool,
    pub(crate) trunc_output_syms: Vec<u16>,
}

// ---------------------------------------------------------------------------
// TokenPeekabooDFA — per-FST persistent + per-step eager DFA
// ---------------------------------------------------------------------------

pub struct TokenPeekabooDFA {
    // Per-FST data (computed once in new())
    sym_to_idx: FxHashMap<u32, u16>,
    idx_to_sym: Vec<u32>,
    ip_universal_states: Vec<bool>,
    all_input_universal: bool,
    num_source_symbols: usize,

    // Per-step data (rebuilt in new_step())
    target: Vec<u32>,
    step_n: u16,
    arena: PosKeyArena,
    nfa_reps: Vec<Vec<u64>>,              // representative NFA set per DFA state
    arcs_from: Vec<Vec<(u32, u32)>>,      // [sid] → [(label, dest_sid)]
    start_ids: Vec<u32>,
    classify_results: Vec<TokenClassify>,

    // Persistent caches
    eps_cache: FxHashMap<u64, (Vec<u64>, u16)>,
    fst_univ_cache: FxHashMap<Vec<u32>, bool>,
}

impl TokenPeekabooDFA {
    pub fn new(fst: &Fst) -> Self {
        // Build output alphabet and sym_to_idx mapping
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

        // Check all_input_universal: every state with non-eps input arcs is ip-universal
        let all_input_universal = (0..fst.num_states as usize).all(|q| {
            !fst.has_non_eps_input[q] || ip_universal_states[q]
        });

        TokenPeekabooDFA {
            sym_to_idx,
            idx_to_sym,
            ip_universal_states,
            all_input_universal,
            num_source_symbols,
            target: Vec::new(),
            step_n: 0,
            arena: PosKeyArena::new(),
            nfa_reps: Vec::new(),
            arcs_from: Vec::new(),
            start_ids: Vec::new(),
            classify_results: Vec::new(),
            eps_cache: FxHashMap::default(),
            fst_univ_cache: FxHashMap::default(),
        }
    }

    /// Reset for a new target prefix.  Builds the entire quotiented DFA eagerly
    /// (it's small enough — typically ~45 states for BPE).
    pub fn new_step(&mut self, fst: &Fst, target: Vec<u32>) {
        let step_n = target.len() as u16;
        self.target = target;
        self.step_n = step_n;

        // Reset per-step state
        self.arena = PosKeyArena::new();
        self.nfa_reps.clear();
        self.arcs_from.clear();
        self.start_ids.clear();
        self.classify_results.clear();
        self.eps_cache.clear();

        // Build NFA
        let sym_to_idx = self.sym_to_idx.clone();
        let nfa = PeekabooNFAMapped::new(fst, &self.target, step_n, &sym_to_idx);

        // Compute start state
        let raw_starts = nfa.start_states();
        let init_closed = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);

        let start_keys = extract_pos_keys(&init_closed);
        let start_id = self.arena.intern(start_keys);
        // Ensure nfa_reps has capacity
        while self.nfa_reps.len() <= start_id as usize {
            self.nfa_reps.push(Vec::new());
        }
        self.nfa_reps[start_id as usize] = init_closed;
        self.start_ids = vec![start_id];

        // BFS — single-representative per position-key set.
        //
        // Position-key quotienting stores ONE representative NFA set per DFA
        // state.  Classification (Q/R/preimage) is correct for the
        // representative.  Arcs are approximate: different representatives
        // may have different transitions.  Sentinel replay uses run_nfa()
        // which simulates the NFA directly, avoiding the DFA arc limitation.
        let mut worklist: VecDeque<u32> = VecDeque::new();
        let mut visited: FxHashSet<u32> = FxHashSet::default();
        worklist.push_back(start_id);
        visited.insert(start_id);

        // Universality filters for non-AIU FSTs
        let mut univ_filters: FxHashMap<u16, PeekabooUniversalityFilter> = FxHashMap::default();
        // We need a PowersetArena for universality sub-BFS
        let mut univ_arena = PowersetArena::new();

        while let Some(sid) = worklist.pop_front() {
            let nfa_set = self.nfa_reps[sid as usize].clone();

            // --- Classify this state ---
            let mut relevant_symbols: Vec<u16> = Vec::new();
            let mut final_symbols: FxHashSet<u16> = FxHashSet::default();
            let mut state_has_truncated = false;
            let mut state_is_preimage = false;
            let mut trunc_output_syms: Vec<u16> = Vec::new();
            let mut seen_relevant: FxHashSet<u16> = FxHashSet::default();
            let mut seen_trunc: FxHashSet<u16> = FxHashSet::default();

            // Pre-compute non-canonical preimage extra_sym
            let non_canonical_extra: Option<u16> = if step_n > 0 {
                let last_target_sym = self.target[(step_n - 1) as usize];
                self.sym_to_idx.get(&last_target_sym).copied()
            } else {
                None
            };

            for &packed in &nfa_set {
                let (fst_state, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);

                // Preimage check
                if buf_len == step_n && fst.is_final[fst_state as usize] {
                    if extra_sym == NO_EXTRA {
                        state_is_preimage = true;
                    } else if let Some(expected) = non_canonical_extra {
                        if extra_sym == expected {
                            state_is_preimage = true;
                        }
                    }
                }

                if truncated {
                    state_has_truncated = true;
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
                        {
                            final_symbols.insert(extra_sym);
                        }
                    } else if (prefix_len as usize) < self.target.len() {
                        if let Some(&expected_idx) = self.sym_to_idx.get(&self.target[prefix_len as usize]) {
                            if extra_sym == expected_idx
                                && buf_len == step_n
                                && fst.is_final[fst_state as usize]
                            {
                                state_is_preimage = true;
                            }
                        }
                    }
                }
            }

            // Check universality
            let mut quotient_sym: Option<u16> = None;
            let mut remainder_syms: Vec<u16> = Vec::new();

            for &y_idx in &relevant_symbols {
                if self.all_input_universal {
                    // AIU fast path: finality implies universality
                    if final_symbols.contains(&y_idx) {
                        if quotient_sym.is_some() {
                            panic!(
                                "State is universal for multiple symbols — FST is likely non-functional"
                            );
                        }
                        quotient_sym = Some(y_idx);
                    }
                } else {
                    // Non-AIU: full universality check
                    if !univ_filters.contains_key(&y_idx) {
                        univ_filters.insert(
                            y_idx,
                            PeekabooUniversalityFilter::new(
                                fst, step_n, y_idx, &self.ip_universal_states,
                            ),
                        );
                    }

                    let filter = univ_filters.get_mut(&y_idx).unwrap();
                    let is_univ = filter.is_universal(
                        &nfa_set, y_idx, &nfa,
                        &mut univ_arena, &mut self.eps_cache,
                        self.num_source_symbols, step_n,
                    );

                    if is_univ {
                        if quotient_sym.is_some() {
                            panic!(
                                "State is universal for multiple symbols — FST is likely non-functional"
                            );
                        }
                        quotient_sym = Some(y_idx);
                    } else if final_symbols.contains(&y_idx) {
                        remainder_syms.push(y_idx);
                    }
                }
            }

            // For AIU: non-quotient final symbols go to remainder
            if self.all_input_universal {
                for &y_idx in &relevant_symbols {
                    if final_symbols.contains(&y_idx) && quotient_sym != Some(y_idx) {
                        remainder_syms.push(y_idx);
                    }
                }
            }

            // Store classification
            while self.classify_results.len() <= sid as usize {
                self.classify_results.push(TokenClassify {
                    quotient_sym: None,
                    remainder_syms: Vec::new(),
                    is_preimage: false,
                    has_truncated: false,
                    trunc_output_syms: Vec::new(),
                });
            }
            self.classify_results[sid as usize] = TokenClassify {
                quotient_sym,
                remainder_syms,
                is_preimage: state_is_preimage,
                has_truncated: state_has_truncated,
                trunc_output_syms,
            };

            // If Q-absorbed, skip arc expansion (search never expands Q states)
            while self.arcs_from.len() <= sid as usize {
                self.arcs_from.push(Vec::new());
            }
            if quotient_sym.is_some() {
                self.arcs_from[sid as usize] = Vec::new();
                continue;
            }

            // --- Expand: compute NFA arcs per input symbol ---
            let all_arcs = nfa.compute_all_arcs(&nfa_set, &mut self.eps_cache);

            let mut arcs_list: Vec<(u32, u32)> = Vec::with_capacity(all_arcs.len());
            let mut closure_cache: FxHashMap<Vec<u32>, u32> = FxHashMap::default();

            for (x, successor) in all_arcs {
                let dest_keys = extract_pos_keys(&successor);
                if dest_keys.is_empty() {
                    continue;
                }

                let dest_id = if let Some(&existing_id) = closure_cache.get(&dest_keys) {
                    existing_id
                } else {
                    let id = self.arena.intern(dest_keys.clone());
                    while self.nfa_reps.len() <= id as usize {
                        self.nfa_reps.push(Vec::new());
                    }
                    if self.nfa_reps[id as usize].is_empty() {
                        self.nfa_reps[id as usize] = successor;
                    }
                    closure_cache.insert(dest_keys, id);
                    id
                };

                arcs_list.push((x, dest_id));

                if visited.insert(dest_id) {
                    worklist.push_back(dest_id);
                }
            }

            self.arcs_from[sid as usize] = arcs_list;
        }
    }

    pub fn start_ids(&self) -> &[u32] {
        &self.start_ids
    }

    pub fn idx_to_sym(&self) -> &[u32] {
        &self.idx_to_sym
    }

    pub fn arcs(&self, sid: u32) -> &[(u32, u32)] {
        let sid_usize = sid as usize;
        if sid_usize < self.arcs_from.len() {
            &self.arcs_from[sid_usize]
        } else {
            &[]
        }
    }

    pub fn classify(&self, sid: u32) -> &TokenClassify {
        &self.classify_results[sid as usize]
    }

    /// Run a source path by simulating the peekaboo NFA directly, then
    /// look up the resulting position-key set in the arena.  This avoids the
    /// DFA's single-representative arc limitation and always finds the
    /// correct DFA state for sentinel replay.
    pub fn run_nfa(&mut self, fst: &Fst, source_path: &[u32]) -> Option<u32> {
        if self.start_ids.is_empty() {
            return None;
        }
        let sym_to_idx = self.sym_to_idx.clone();
        let nfa = PeekabooNFAMapped::new(fst, &self.target, self.step_n, &sym_to_idx);

        // Start from the initial NFA set
        let raw_starts = nfa.start_states();
        let mut current = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);

        // Follow each source symbol through the NFA
        for &x in source_path {
            let mut next_raw: Vec<u64> = Vec::new();
            for &packed in &current {
                // Use the NFA's arcs() to get all transitions (handles buffer
                // management, truncation, etc. correctly)
                for (label, dest) in nfa.arcs(packed) {
                    if label == x {
                        next_raw.push(dest);
                    }
                }
            }
            if next_raw.is_empty() {
                return None;
            }
            current = nfa.eps_closure_set(&next_raw, &mut self.eps_cache);
        }

        // Look up position-key set in the arena
        let keys = extract_pos_keys(&current);
        if keys.is_empty() {
            return None;
        }

        // Check arena for this exact key set
        if keys.len() == 1 {
            self.arena.single_map.get(&keys[0]).copied()
        } else {
            self.arena.map.get(&keys).copied()
        }
    }

    /// Run a source path from start using DFA arcs. Falls back to None if
    /// an arc is missing (approximate — single-representative DFA).
    pub fn run(&self, source_path: &[u32]) -> Option<u32> {
        if self.start_ids.is_empty() {
            return None;
        }
        let mut state = self.start_ids[0];
        for &x in source_path {
            let mut found = false;
            for &(lbl, dst) in self.arcs(state) {
                if lbl == x {
                    state = dst;
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
        Some(state)
    }
}
