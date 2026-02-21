//! Position-set peekaboo with dirty-state incremental reuse.
//!
//! Exploits token-decomposability (TD): for TD FSTs, DFA states with the same
//! set of position descriptors have identical transitions, regardless of the FST
//! states in their NFA constituents.  This collapses the DFA dramatically.
//!
//! Position descriptors:
//! - **Non-truncated** elements: `(buf_len, extra_sym, truncated=0)` — the lower
//!   32 bits of the packed u64.  TD guarantees that all FST states at the same
//!   position produce identical successors and finality.
//! - **Truncated** elements: the full packed u64 (includes FST state in bits
//!   63:32).  Truncation breaks TD because all output arcs are followed regardless
//!   of target match, so the FST state determines transitions and finality.
//!
//! Combines: Rust speed + position-set compression + dirty-state reuse.

use crate::decompose::FsaResult;
use crate::fst::{compute_ip_universal_states, Fst, EPSILON};
use crate::peekaboo::{
    evict_peekaboo_eps_cache, unpack_peekaboo, PeekabooNFAMapped,
    PeekabooUniversalityFilter, NO_EXTRA, STATUS_INTERIOR, STATUS_NEW, STATUS_QSTOP, STATUS_RSTOP,
};
use crate::powerset::PowersetArena;
use crate::rho::RHO;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Position descriptor
//   Non-truncated (bit 0 == 0): lower 32 bits only (position portion).
//     TD guarantees equivalence across FST states.
//   Truncated (bit 0 == 1): full packed u64 (includes FST state in bits 63:32).
//     FST state determines transitions and finality for truncated elements.
// ---------------------------------------------------------------------------

type PosDescriptor = u64;

#[inline]
fn pos_descriptor(packed_nfa: u64) -> PosDescriptor {
    if packed_nfa & 1 == 1 {
        // Truncated: include FST state — it determines transitions and finality
        packed_nfa
    } else {
        // Non-truncated: position only — TD guarantees equivalence
        (packed_nfa as u32) as u64
    }
}

// ---------------------------------------------------------------------------
// PositionSetArena: hash-consing arena for sorted Vec<PosDescriptor> → u32 IDs
// ---------------------------------------------------------------------------

struct PositionSetArena {
    sets: Vec<Vec<PosDescriptor>>,
    set_to_id: FxHashMap<Vec<PosDescriptor>, u32>,
    /// One NFA state set (the first encountered) whose position descriptors
    /// map to this psid.  Used as the canonical representative for arc computation.
    canonical_rep: Vec<Vec<u64>>,
    is_final: Vec<bool>,
}

impl PositionSetArena {
    fn new() -> Self {
        PositionSetArena {
            sets: Vec::new(),
            set_to_id: FxHashMap::default(),
            canonical_rep: Vec::new(),
            is_final: Vec::new(),
        }
    }

    /// Intern an NFA state set by its position descriptors.
    /// Returns the position-set ID (psid).
    fn intern(&mut self, nfa_states: Vec<u64>, is_final: bool) -> u32 {
        let mut descriptors: Vec<PosDescriptor> = nfa_states
            .iter()
            .map(|&s| pos_descriptor(s))
            .collect();
        descriptors.sort_unstable();
        descriptors.dedup();

        if let Some(&psid) = self.set_to_id.get(&descriptors) {
            // TD sanity check: finality must match
            debug_assert_eq!(
                self.is_final[psid as usize], is_final,
                "TD violation: position set {:?} has conflicting finality",
                descriptors
            );
            return psid;
        }

        let psid = self.sets.len() as u32;
        self.sets.push(descriptors.clone());
        self.set_to_id.insert(descriptors, psid);
        self.canonical_rep.push(nfa_states);
        self.is_final.push(is_final);
        psid
    }

    fn len(&self) -> usize {
        self.sets.len()
    }
}

// ---------------------------------------------------------------------------
// PositionSetPeekaboo: dirty-state incremental position-set peekaboo
// ---------------------------------------------------------------------------

pub struct PositionSetPeekaboo {
    // FST metadata (computed once in new())
    output_alphabet: Vec<u32>,
    source_alphabet: Vec<u32>,
    sym_to_idx: FxHashMap<u32, u16>,
    idx_to_sym: Vec<u32>,
    ip_universal_states: Vec<bool>,
    num_source_symbols: usize,

    // Position-set DFA
    arena: PositionSetArena,
    global_start_id: u32,
    arcs_from: Vec<Vec<(u32, u32)>>,       // [psid] → [(label, dest_psid)]
    has_rho: Vec<bool>,                    // [psid] → true if arcs contain a RHO entry
    state_status: Vec<u8>,                  // [psid] → STATUS_*
    max_bufpos: Vec<u16>,                   // [psid] → max buf_len in the position descriptors
    reverse_arcs: Vec<Vec<u32>>,
    reachable: Vec<u32>,
    reachable_flags: Vec<bool>,
    needs_reexpand: Vec<bool>,

    // Per-symbol Q/R
    decomp_q: FxHashMap<u16, Vec<u32>>,
    decomp_r: FxHashMap<u16, Vec<u32>>,

    // Caches
    eps_cache: FxHashMap<u64, (Vec<u64>, u16)>,
    fst_univ_cache: FxHashMap<Vec<u32>, bool>,

    // We also need a PowersetArena for universality sub-BFS
    univ_arena: PowersetArena,

    // Target tracking
    prev_target: Vec<u32>,

    // Generation counter: incremented on every full_reset() so callers can
    // detect when DFA state IDs have been invalidated.
    generation: u64,
}

impl PositionSetPeekaboo {
    pub fn new(fst: &Fst) -> Self {
        // Build the output alphabet
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

        // Build sym_to_idx map
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

        PositionSetPeekaboo {
            output_alphabet,
            source_alphabet,
            sym_to_idx,
            idx_to_sym,
            ip_universal_states,
            num_source_symbols,
            arena: PositionSetArena::new(),
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
            univ_arena: PowersetArena::new(),
            prev_target: Vec::new(),
            generation: 0,
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
        self.arena = PositionSetArena::new();
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
        self.univ_arena = PowersetArena::new();
        // NOTE: fst_univ_cache is NOT cleared — it's target-independent
    }

    /// Ensure all per-state arrays are sized to cover `needed` entries.
    fn ensure_capacity(&mut self, needed: usize) {
        let old_len = self.arcs_from.len();
        if needed > old_len {
            self.arcs_from.resize_with(needed, Vec::new);
            self.has_rho.resize(needed, false);
            self.state_status.resize(needed, STATUS_NEW);
            self.reverse_arcs.resize_with(needed, Vec::new);
            self.max_bufpos.resize(needed, 0);
            for psid in old_len..needed {
                let pos_set = &self.arena.sets[psid];
                // Extract buf_len from LOWER 32 bits (works for both truncated and non-truncated)
                let mbp = pos_set
                    .iter()
                    .map(|&d| {
                        let lower = d as u32;
                        ((lower >> 17) & 0x7FFF) as u16
                    })
                    .max()
                    .unwrap_or(0);
                self.max_bufpos[psid] = mbp;
            }
        }
    }

    /// Remove `psid` from reverse_arcs of all its outgoing arc destinations.
    fn remove_outgoing_reverse_arcs(&mut self, psid: u32) {
        let arcs = std::mem::take(&mut self.arcs_from[psid as usize]);
        for &(_lbl, dst) in &arcs {
            let dst_usize = dst as usize;
            if dst_usize < self.reverse_arcs.len() {
                let ra = &mut self.reverse_arcs[dst_usize];
                if let Some(pos) = ra.iter().position(|&s| s == psid) {
                    ra.swap_remove(pos);
                }
            }
        }
    }

    /// Collect trimmed arcs via backward BFS from stop states.
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
                    let explicit_syms: FxHashSet<u32> = self.arcs_from[sid_usize]
                        .iter()
                        .filter(|&&(l, _)| l != RHO)
                        .map(|&(l, _)| l)
                        .collect();
                    let rho_dest = self.arcs_from[sid_usize]
                        .iter()
                        .find(|&&(l, _)| l == RHO)
                        .map(|&(_, d)| d);

                    for &(l, d) in &self.arcs_from[sid_usize] {
                        if l != RHO && (d as usize) < n && backward[d as usize] {
                            arc_src.push(sid);
                            arc_lbl.push(l);
                            arc_dst.push(d);
                        }
                    }
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

            let q_stops: Vec<u32> = self
                .decomp_q
                .get(&y_idx)
                .map(|v| {
                    v.iter()
                        .filter(|&&sid| {
                            (sid as usize) < self.reachable_flags.len()
                                && self.reachable_flags[sid as usize]
                        })
                        .copied()
                        .collect()
                })
                .unwrap_or_default();
            let r_stops: Vec<u32> = self
                .decomp_r
                .get(&y_idx)
                .map(|v| {
                    v.iter()
                        .filter(|&&sid| {
                            (sid as usize) < self.reachable_flags.len()
                                && self.reachable_flags[sid as usize]
                        })
                        .copied()
                        .collect()
                })
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

    pub fn step(&self, sid: u32, x: u32) -> Option<u32> {
        let sid_usize = sid as usize;
        for &(lbl, dst) in self.arcs_from(sid) {
            if lbl == x {
                return Some(dst);
            }
        }
        if sid_usize < self.has_rho.len() && self.has_rho[sid_usize] {
            return self
                .arcs_from(sid)
                .iter()
                .find(|&&(lbl, _)| lbl == RHO)
                .map(|&(_, dst)| dst);
        }
        None
    }

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

    pub fn state_has_rho(&self, sid: u32) -> bool {
        let sid_usize = sid as usize;
        sid_usize < self.has_rho.len() && self.has_rho[sid_usize]
    }

    pub fn rho_arcs(&self, sid: u32) -> (bool, Option<u32>, Vec<(u32, u32)>) {
        let sid_usize = sid as usize;
        if sid_usize >= self.has_rho.len() || !self.has_rho[sid_usize] {
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

    pub fn source_alphabet(&self) -> &[u32] {
        &self.source_alphabet
    }

    /// Decode a position-set DFA state via its canonical NFA representative.
    pub fn canonical_rep(&self, psid: u32) -> &[u64] {
        &self.arena.canonical_rep[psid as usize]
    }

    /// Compute preimage stop states.
    pub fn compute_preimage_stops(&self, fst: &Fst, step_n: u16) -> Vec<u32> {
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
            // Use the canonical NFA state set for preimage classification
            let nfa_set = &self.arena.canonical_rep[sid_usize];
            let is_preimage = nfa_set.iter().any(|&packed| {
                let (fst_state, buf_len, extra_sym, _truncated) = unpack_peekaboo(packed);
                if buf_len != step_n || !fst.is_final[fst_state as usize] {
                    return false;
                }
                if extra_sym == NO_EXTRA {
                    return true;
                }
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

    /// Compute resume frontiers.
    pub fn compute_resume_frontiers(&self) -> FxHashMap<u16, Vec<u32>> {
        let mut frontiers: FxHashMap<u16, Vec<u32>> = FxHashMap::default();

        for &sid in &self.reachable {
            let sid_usize = sid as usize;
            if sid_usize >= self.reachable_flags.len() || !self.reachable_flags[sid_usize] {
                continue;
            }
            // Use canonical rep to check for truncated elements
            let nfa_set = &self.arena.canonical_rep[sid_usize];

            let has_truncated = nfa_set.iter().any(|&packed| {
                let (_fst_state, _buf_len, _extra_sym, truncated) = unpack_peekaboo(packed);
                truncated
            });

            if has_truncated {
                continue;
            }

            // Check successors for truncated elements
            let mut frontier_syms: FxHashSet<u16> = FxHashSet::default();
            for &(_lbl, dst) in &self.arcs_from[sid_usize] {
                let dst_usize = dst as usize;
                if dst_usize < self.arena.canonical_rep.len() {
                    for &packed in &self.arena.canonical_rep[dst_usize] {
                        let (_fst_state, _buf_len, extra_sym, truncated) =
                            unpack_peekaboo(packed);
                        if truncated && extra_sym != NO_EXTRA {
                            frontier_syms.insert(extra_sym);
                        }
                    }
                }
            }

            for &y_idx in &frontier_syms {
                frontiers.entry(y_idx).or_default().push(sid);
            }

            // Also add if state is in decomp_q or decomp_r
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

    /// Decompose without extracting FSA results — just runs the BFS.
    pub fn decompose_bfs_only(&mut self, fst: &Fst, target: &[u32]) {
        let _ = self.decompose(fst, target);
    }

    /// Main entry point: decompose the FST for the given target.
    pub fn decompose(
        &mut self,
        fst: &Fst,
        target: &[u32],
    ) -> crate::peekaboo::PeekabooResult {
        let total_start = Instant::now();
        let target_len = target.len();

        // Same target → extract cached results
        if target == self.prev_target.as_slice() && self.arena.len() > 0 {
            let extract_start = Instant::now();
            let per_symbol = self.extract_results();
            let extract_ms = extract_start.elapsed().as_secs_f64() * 1000.0;
            let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            return crate::peekaboo::PeekabooResult {
                per_symbol,
                stats: crate::peekaboo::PeekabooProfileStats {
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

            let n = self.arena.len();
            if self.needs_reexpand.len() < n {
                self.needs_reexpand.resize(n, false);
            }

            // Mark dirty states
            for &sid in &self.reachable {
                let sid_usize = sid as usize;
                if sid_usize < self.max_bufpos.len() && self.max_bufpos[sid_usize] >= frontier {
                    self.needs_reexpand[sid_usize] = true;
                    dirty_border.push(sid);
                }
            }

            // Mark border states
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

            // Reset dirty+border states
            for &sid in &dirty_border {
                let sid_usize = sid as usize;
                self.remove_outgoing_reverse_arcs(sid);
                self.state_status[sid_usize] = STATUS_NEW;
                self.needs_reexpand[sid_usize] = false;
            }
        } else {
            self.full_reset();
        }

        // Create NFA with step_n = target_len (single-pass)
        let sym_to_idx = self.sym_to_idx.clone();
        let step_n = target_len as u16;
        let nfa = PeekabooNFAMapped::new(fst, target, step_n, &sym_to_idx);

        // Compute start state
        let raw_starts = nfa.start_states();
        let init_closed = nfa.eps_closure_set(&raw_starts, &mut self.eps_cache);
        let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
        let start_id = self.arena.intern(init_closed, any_final);
        self.global_start_id = start_id;

        // Ensure capacity
        self.ensure_capacity(self.arena.len());

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

        for &sid in &dirty_border {
            worklist.push_back(sid);
        }

        // Clear per-symbol Q/R
        self.decomp_q.clear();
        self.decomp_r.clear();

        // Create fresh universality filters
        let mut univ_filters: FxHashMap<u16, PeekabooUniversalityFilter> = FxHashMap::default();

        let bfs_start = Instant::now();

        // BFS loop
        while let Some(psid) = worklist.pop_front() {
            if self.state_status[psid as usize] != STATUS_NEW {
                continue;
            }

            // Get the canonical NFA state set for this position-set state
            let nfa_set = self.arena.canonical_rep[psid as usize].clone();

            // Find relevant symbols
            let mut relevant_syms = FxHashSet::default();
            for &packed in &nfa_set {
                let (_, buf_len, extra_sym, _truncated) = unpack_peekaboo(packed);
                if extra_sym != NO_EXTRA {
                    let (eff_n, eff_extra, is_valid) =
                        nfa.effective_state(buf_len, extra_sym, false);
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
                            fst,
                            step_n,
                            y_idx,
                            &self.ip_universal_states,
                        ),
                    );
                }

                {
                    // Check fst_univ_cache for pure-frontier states
                    let projected = univ_filters
                        .get(&y_idx)
                        .unwrap()
                        .project_and_refine(&nfa_set, y_idx, step_n);
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
                            &mut self.univ_arena,
                            &mut self.eps_cache,
                            self.num_source_symbols,
                            step_n,
                        );

                        // Cache if pure-frontier
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
                        if let Some(prev) = continuous {
                            panic!(
                                "State is universal for both symbol {} and {} — \
                                 FST is likely non-functional",
                                prev, y_idx
                            );
                        }
                        self.decomp_q.entry(y_idx).or_default().push(psid);
                        continuous = Some(y_idx);
                        continue;
                    }
                }

                let filter = univ_filters.get(&y_idx).unwrap();
                if filter.is_projected_final(&nfa_set, y_idx, fst, step_n) {
                    self.decomp_r.entry(y_idx).or_default().push(psid);
                    has_final_syms = true;
                }
            }

            if continuous.is_some() {
                self.state_status[psid as usize] = STATUS_QSTOP;
                continue;
            }

            // Expand arcs using the canonical NFA representative
            let all_arcs = nfa.compute_all_arcs(&nfa_set, &mut self.eps_cache);

            // Intern all destinations via position-set
            let mut interned_arcs: Vec<(u32, u32)> = Vec::with_capacity(all_arcs.len());
            let mut unique_dests: FxHashSet<u32> = FxHashSet::default();
            for (x, successor) in all_arcs {
                let succ_final = successor.iter().any(|&s| nfa.is_final(s));
                let dest_id = self.arena.intern(successor, succ_final);

                let needed = dest_id as usize + 1;
                self.ensure_capacity(needed);
                if needed > self.reachable_flags.len() {
                    self.reachable_flags.resize(needed, false);
                }

                interned_arcs.push((x, dest_id));
                unique_dests.insert(dest_id);
            }

            // Rho compression
            let is_complete = self.num_source_symbols > 0
                && interned_arcs.len() == self.num_source_symbols;

            if is_complete {
                let mut dest_counts: FxHashMap<u32, usize> = FxHashMap::default();
                for &(_, dest) in &interned_arcs {
                    *dest_counts.entry(dest).or_insert(0) += 1;
                }
                let rho_dest = *dest_counts
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .unwrap()
                    .0;

                if dest_counts.len() == 1 {
                    self.arcs_from[psid as usize] = vec![(RHO, rho_dest)];
                } else {
                    let mut result_arcs: Vec<(u32, u32)> = Vec::new();
                    for &(x, dest) in &interned_arcs {
                        if dest != rho_dest {
                            result_arcs.push((x, dest));
                        }
                    }
                    result_arcs.push((RHO, rho_dest));
                    self.arcs_from[psid as usize] = result_arcs;
                }
                self.has_rho[psid as usize] = true;
            } else {
                for &(x, dest_id) in &interned_arcs {
                    self.arcs_from[psid as usize].push((x, dest_id));
                }
                self.has_rho[psid as usize] = false;
            }

            // Add reverse arcs and enqueue successors
            for &dest_id in &unique_dests {
                self.reverse_arcs[dest_id as usize].push(psid);
                if self.state_status[dest_id as usize] == STATUS_NEW {
                    worklist.push_back(dest_id);
                    if !self.reachable_flags[dest_id as usize] {
                        self.reachable_flags[dest_id as usize] = true;
                        self.reachable.push(dest_id);
                    }
                }
            }

            // Classify state
            self.state_status[psid as usize] = if has_final_syms {
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

        crate::peekaboo::PeekabooResult {
            per_symbol,
            stats: crate::peekaboo::PeekabooProfileStats {
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
