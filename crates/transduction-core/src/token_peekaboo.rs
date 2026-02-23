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
//!
//! **Lazy + dirty-state persistence:** The arena, NFA reps, arcs, and
//! classification persist across decode steps.  On prefix extension, only
//! "dirty" states (max_bufpos >= frontier) and their "border" predecessors
//! are invalidated.  Clean states keep cached arcs/classification, avoiding
//! the expensive `compute_all_arcs()` call on most states.

use crate::fst::{compute_ip_universal_states, Fst, EPSILON};
use crate::peekaboo::{
    evict_peekaboo_eps_cache, unpack_peekaboo, FactoredArena,
    PeekabooNFAMapped, PeekabooUniversalityFilter, NO_EXTRA,
};
use rustc_hash::{FxHashMap, FxHashSet};

// ---------------------------------------------------------------------------
// Position-key extraction
// ---------------------------------------------------------------------------

#[inline]
fn pos_key(packed: u64) -> u32 {
    (packed & 0xFFFFFFFF) as u32
}

fn extract_pos_keys(nfa_set: &[u64]) -> Vec<u32> {
    let mut keys: Vec<u32> = nfa_set.iter().map(|&p| pos_key(p)).collect();
    keys.sort_unstable();
    keys.dedup();
    keys
}

#[inline]
fn max_bufpos_from_nfa(nfa_set: &[u64]) -> u16 {
    nfa_set
        .iter()
        .map(|&e| ((e >> 17) & 0x7FFF) as u16)
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Position-key arena
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
// Classification result
// ---------------------------------------------------------------------------

pub(crate) struct TokenClassify {
    pub(crate) quotient_sym: Option<u16>,
    pub(crate) remainder_syms: Vec<u16>,
    pub(crate) is_preimage: bool,
    pub(crate) has_truncated: bool,
    pub(crate) trunc_output_syms: Vec<u16>,
}

// ---------------------------------------------------------------------------
// TokenPeekabooDFA
// ---------------------------------------------------------------------------

pub struct TokenPeekabooDFA {
    sym_to_idx: FxHashMap<u32, u16>,
    idx_to_sym: Vec<u32>,
    ip_universal_states: Vec<bool>,
    all_input_universal: bool,
    num_source_symbols: usize,

    target: Vec<u32>,
    step_n: u16,

    // Persistent DFA structure
    arena: PosKeyArena,
    nfa_reps: Vec<Vec<u64>>,
    arcs_from: Vec<Vec<(u32, u32)>>,
    start_ids: Vec<u32>,
    classify_results: Vec<TokenClassify>,
    arcs_computed: Vec<bool>,
    classify_computed: Vec<bool>,

    // Dirty-state tracking
    max_bufpos: Vec<u16>,
    reverse_arcs: Vec<Vec<u32>>,
    prev_target: Vec<u32>,
    needs_reexpand: Vec<bool>,

    // Per-step caches
    univ_filters: FxHashMap<u16, PeekabooUniversalityFilter>,
    univ_arena: FactoredArena,

    // Persistent caches
    eps_cache: FxHashMap<u64, (Vec<u64>, u16)>,
    #[allow(dead_code)]
    fst_univ_cache: FxHashMap<Vec<u32>, bool>,
}

impl TokenPeekabooDFA {
    pub fn new(fst: &Fst) -> Self {
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
        let all_input_universal = (0..fst.num_states as usize)
            .all(|q| !fst.has_non_eps_input[q] || ip_universal_states[q]);

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
            arcs_computed: Vec::new(),
            classify_computed: Vec::new(),
            max_bufpos: Vec::new(),
            reverse_arcs: Vec::new(),
            prev_target: Vec::new(),
            needs_reexpand: Vec::new(),
            univ_filters: FxHashMap::default(),
            univ_arena: FactoredArena::new(),
            eps_cache: FxHashMap::default(),
            fst_univ_cache: FxHashMap::default(),
        }
    }

    fn is_prefix_extension(&self, target: &[u32]) -> bool {
        target.len() > self.prev_target.len()
            && target[..self.prev_target.len()] == self.prev_target[..]
    }

    fn full_reset(&mut self) {
        self.arena = PosKeyArena::new();
        self.nfa_reps.clear();
        self.arcs_from.clear();
        self.arcs_computed.clear();
        self.classify_computed.clear();
        self.classify_results.clear();
        self.max_bufpos.clear();
        self.reverse_arcs.clear();
        self.needs_reexpand.clear();
        self.eps_cache.clear();
    }

    fn remove_outgoing_reverse_arcs(&mut self, sid: u32) {
        let sid_usize = sid as usize;
        if sid_usize >= self.arcs_from.len() {
            return;
        }
        let arcs = std::mem::take(&mut self.arcs_from[sid_usize]);
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

    fn selective_invalidate(&mut self, frontier: u16) {
        let n = self.arena.len();
        if n == 0 {
            return;
        }
        self.needs_reexpand.resize(n, false);

        let mut dirty_border: Vec<u32> = Vec::new();
        for sid in 0..n {
            if sid < self.max_bufpos.len() && self.max_bufpos[sid] >= frontier {
                self.needs_reexpand[sid] = true;
                dirty_border.push(sid as u32);
            }
        }

        let dirty_count = dirty_border.len();
        for i in 0..dirty_count {
            let dirty_sid = dirty_border[i] as usize;
            if dirty_sid < self.reverse_arcs.len() {
                for j in 0..self.reverse_arcs[dirty_sid].len() {
                    let src = self.reverse_arcs[dirty_sid][j] as usize;
                    if src < n
                        && !self.needs_reexpand[src]
                        && self.arcs_computed.get(src).copied().unwrap_or(false)
                    {
                        self.needs_reexpand[src] = true;
                        dirty_border.push(src as u32);
                    }
                }
            }
        }

        for &sid in &dirty_border {
            let sid_usize = sid as usize;
            self.remove_outgoing_reverse_arcs(sid);
            if sid_usize < self.arcs_computed.len() {
                self.arcs_computed[sid_usize] = false;
            }
            if sid_usize < self.classify_computed.len() {
                self.classify_computed[sid_usize] = false;
            }
            self.needs_reexpand[sid_usize] = false;
        }
    }

    pub fn new_step(&mut self, fst: &Fst, target: Vec<u32>) {
        if target == self.prev_target && !self.prev_target.is_empty() {
            return;
        }

        let step_n = target.len() as u16;
        let is_extension =
            !self.prev_target.is_empty() && self.is_prefix_extension(&target);

        if is_extension {
            let frontier = self.prev_target.len() as u16;
            evict_peekaboo_eps_cache(&mut self.eps_cache, frontier);
            self.selective_invalidate(frontier);
        } else if !self.prev_target.is_empty() {
            self.full_reset();
        }

        self.target = target;
        self.step_n = step_n;
        self.univ_filters.clear();
        self.univ_arena = FactoredArena::new();

        let sym_to_idx = self.sym_to_idx.clone();
        let nfa = PeekabooNFAMapped::new(fst, &self.target, step_n, &sym_to_idx);
        let raw_starts = nfa.start_states();
        let init_closed = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);
        let start_keys = extract_pos_keys(&init_closed);
        let start_id = self.arena.intern(start_keys);
        self.ensure_capacity(start_id as usize + 1);
        if self.nfa_reps[start_id as usize].is_empty() {
            self.max_bufpos[start_id as usize] = max_bufpos_from_nfa(&init_closed);
            self.nfa_reps[start_id as usize] = init_closed;
        }
        self.start_ids = vec![start_id];
        self.prev_target = self.target.clone();
    }

    fn ensure_capacity(&mut self, needed: usize) {
        if needed > self.nfa_reps.len() {
            self.nfa_reps.resize_with(needed, Vec::new);
            self.arcs_from.resize_with(needed, Vec::new);
            self.classify_results.resize_with(needed, || TokenClassify {
                quotient_sym: None,
                remainder_syms: Vec::new(),
                is_preimage: false,
                has_truncated: false,
                trunc_output_syms: Vec::new(),
            });
            self.arcs_computed.resize(needed, false);
            self.classify_computed.resize(needed, false);
            self.max_bufpos.resize(needed, 0);
            self.reverse_arcs.resize_with(needed, Vec::new);
        }
    }

    pub fn ensure_classify(&mut self, fst: &Fst, sid: u32) {
        let sid_usize = sid as usize;
        if sid_usize < self.classify_computed.len() && self.classify_computed[sid_usize] {
            return;
        }
        self.ensure_capacity(sid_usize + 1);

        let nfa_set = self.nfa_reps[sid_usize].clone();
        let step_n = self.step_n;

        let mut relevant_symbols: Vec<u16> = Vec::new();
        let mut final_symbols: FxHashSet<u16> = FxHashSet::default();
        let mut state_has_truncated = false;
        let mut state_is_preimage = false;
        let mut trunc_output_syms: Vec<u16> = Vec::new();
        let mut seen_relevant: FxHashSet<u16> = FxHashSet::default();
        let mut seen_trunc: FxHashSet<u16> = FxHashSet::default();

        let non_canonical_extra: Option<u16> = if step_n > 0 {
            let last_target_sym = self.target[(step_n - 1) as usize];
            self.sym_to_idx.get(&last_target_sym).copied()
        } else {
            None
        };

        for &packed in &nfa_set {
            let (fst_state, buf_len, extra_sym, truncated) = unpack_peekaboo(packed);

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
                    if fst.is_final[fst_state as usize] && buf_len == step_n + 1 {
                        final_symbols.insert(extra_sym);
                    }
                } else if (prefix_len as usize) < self.target.len() {
                    if let Some(&expected_idx) =
                        self.sym_to_idx.get(&self.target[prefix_len as usize])
                    {
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

        let mut quotient_sym: Option<u16> = None;
        let mut remainder_syms: Vec<u16> = Vec::new();

        if self.all_input_universal {
            for &y_idx in &relevant_symbols {
                if final_symbols.contains(&y_idx) {
                    if quotient_sym.is_some() {
                        panic!(
                            "State is universal for multiple symbols — FST is likely non-functional"
                        );
                    }
                    quotient_sym = Some(y_idx);
                }
            }
            for &y_idx in &relevant_symbols {
                if final_symbols.contains(&y_idx) && quotient_sym != Some(y_idx) {
                    remainder_syms.push(y_idx);
                }
            }
        } else {
            for &y_idx in &relevant_symbols {
                if !self.univ_filters.contains_key(&y_idx) {
                    self.univ_filters.insert(
                        y_idx,
                        PeekabooUniversalityFilter::new(
                            fst,
                            step_n,
                            y_idx,
                            &self.ip_universal_states,
                        ),
                    );
                }
            }

            let mut univ_filters = std::mem::take(&mut self.univ_filters);
            let sym_to_idx = self.sym_to_idx.clone();
            let target = self.target.clone();
            let nfa = PeekabooNFAMapped::new(fst, &target, step_n, &sym_to_idx);

            for &y_idx in &relevant_symbols {
                let filter = univ_filters.get_mut(&y_idx).unwrap();
                let is_univ = filter.is_universal(
                    &nfa_set,
                    y_idx,
                    &nfa,
                    &mut self.univ_arena,
                    &mut self.eps_cache,
                    self.num_source_symbols,
                    step_n,
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

            self.univ_filters = univ_filters;
        }

        self.classify_results[sid_usize] = TokenClassify {
            quotient_sym,
            remainder_syms,
            is_preimage: state_is_preimage,
            has_truncated: state_has_truncated,
            trunc_output_syms,
        };
        self.classify_computed[sid_usize] = true;
    }

    pub fn ensure_arcs(&mut self, fst: &Fst, sid: u32) {
        let sid_usize = sid as usize;
        if sid_usize < self.arcs_computed.len() && self.arcs_computed[sid_usize] {
            return;
        }
        self.ensure_capacity(sid_usize + 1);

        let nfa_set = self.nfa_reps[sid_usize].clone();
        let sym_to_idx = self.sym_to_idx.clone();
        let target = self.target.clone();
        let step_n = self.step_n;

        let nfa = PeekabooNFAMapped::new(fst, &target, step_n, &sym_to_idx);
        let all_arcs = nfa.compute_all_arcs(&nfa_set, &mut self.eps_cache);

        let mut arcs_list: Vec<(u32, u32)> = Vec::with_capacity(all_arcs.len());
        for (x, successor) in all_arcs {
            let dest_keys = extract_pos_keys(&successor);
            if dest_keys.is_empty() {
                continue;
            }
            let dest_id = self.arena.intern(dest_keys);
            self.ensure_capacity(dest_id as usize + 1);
            if self.nfa_reps[dest_id as usize].is_empty() {
                self.max_bufpos[dest_id as usize] = max_bufpos_from_nfa(&successor);
                self.nfa_reps[dest_id as usize] = successor;
            }
            arcs_list.push((x, dest_id));
        }

        self.arcs_from[sid_usize] = arcs_list;

        for &(_label, dest_id) in &self.arcs_from[sid_usize] {
            self.reverse_arcs[dest_id as usize].push(sid);
        }

        self.arcs_computed[sid_usize] = true;
    }

    /// Compute the DFA destination for a single input symbol from a state.
    /// Lazily computes all arcs from the state on first access.
    pub fn single_arc(&mut self, fst: &Fst, sid: u32, x: u32) -> Option<u32> {
        self.ensure_arcs(fst, sid);
        for &(lbl, dst) in &self.arcs_from[sid as usize] {
            if lbl == x {
                return Some(dst);
            }
        }
        None
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

    pub fn run(&mut self, fst: &Fst, source_path: &[u32]) -> Option<u32> {
        if self.start_ids.is_empty() {
            return None;
        }
        let mut state = self.start_ids[0];
        for &x in source_path {
            self.ensure_arcs(fst, state);
            let mut found = false;
            for &(lbl, dst) in &self.arcs_from[state as usize] {
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

    pub fn run_nfa(&mut self, fst: &Fst, source_path: &[u32]) -> Option<u32> {
        if self.start_ids.is_empty() {
            return None;
        }
        let sym_to_idx = self.sym_to_idx.clone();
        let nfa = PeekabooNFAMapped::new(fst, &self.target, self.step_n, &sym_to_idx);

        let raw_starts = nfa.start_states();
        let mut current = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);

        for &x in source_path {
            let mut next_raw: Vec<u64> = Vec::new();
            for &packed in &current {
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

        let keys = extract_pos_keys(&current);
        if keys.is_empty() {
            return None;
        }

        let id = self.arena.intern(keys);
        self.ensure_capacity(id as usize + 1);
        if self.nfa_reps[id as usize].is_empty() {
            self.max_bufpos[id as usize] = max_bufpos_from_nfa(&current);
            self.nfa_reps[id as usize] = current;
        }
        Some(id)
    }
}
