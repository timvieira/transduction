use crate::fst::{compute_ip_universal_states, Fst};
use crate::powerset::PowersetArena;
use crate::precover::{pack, PrecoverNFA};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::time::Instant;

/// Result FSA represented as parallel arrays (easy to send across PyO3 boundary).
pub struct FsaResult {
    pub num_states: u32,
    pub start: Vec<u32>,
    pub stop: Vec<u32>,
    pub arc_src: Vec<u32>,
    pub arc_lbl: Vec<u32>,
    pub arc_dst: Vec<u32>,
}

/// Profiling statistics for a decompose() call.
pub struct ProfileStats {
    pub total_ms: f64,
    pub init_ms: f64,
    pub bfs_ms: f64,

    // BFS breakdown
    pub compute_arcs_ms: f64,
    pub compute_arcs_calls: u64,
    pub intern_ms: f64,
    pub intern_calls: u64,

    // is_universal
    pub universal_ms: f64,
    pub universal_calls: u64,
    pub universal_true: u64,
    pub universal_false: u64,
    pub universal_sub_bfs_states: u64,    // total states visited in sub-BFS
    pub universal_compute_arcs_calls: u64,

    // arena stats
    pub dfa_states: u32,
    pub total_arcs: u64,
    pub q_stops: u32,
    pub r_stops: u32,

    // powerset state sizes
    pub max_powerset_size: usize,
    pub avg_powerset_size: f64,

    // eps closure
    pub eps_cache_hits: u64,
    pub eps_cache_misses: u64,
}

/// Result of decomposition: quotient Q and remainder R, plus profiling stats.
pub struct DecompResult {
    pub quotient: FsaResult,
    pub remainder: FsaResult,
    pub stats: ProfileStats,
}

/// Universality short-circuit optimizer.
///
/// Encapsulates multiple strategies to determine whether a DFA state (powerset
/// state) accepts Σ*, avoiding expensive sub-BFS whenever possible:
///
/// 1. Witness check: if any NFA element in the powerset state is a known
///    ip-universal witness, the state is universal.
/// 3. Superset monotonicity: if a known-universal set is a subset of the
///    current set, the current set is universal too.
/// 4. Subset monotonicity: if the current set is a subset of a known
///    non-universal set, it's non-universal too.
/// 5. BFS fallback: full sub-BFS. Results are added to the positive or
///    negative cache for future lookups.
struct UniversalityFilter {
    /// Packed NFA states `(q, target_len)` for ip-universal FST states q.
    witnesses: FxHashSet<u64>,

    /// Element-indexed positive cache (known universal NFA-state sets).
    /// pos_index[nfa_element] = list of entry IDs whose stored set contains that element.
    pos_index: FxHashMap<u64, Vec<u32>>,
    /// entry_id -> size of the stored set
    pos_sizes: Vec<usize>,

    /// Element-indexed negative cache (known non-universal NFA-state sets).
    neg_index: FxHashMap<u64, Vec<u32>>,
    neg_next: u32,

}

impl UniversalityFilter {
    fn new(fst: &Fst, target_len: u32) -> Self {
        let mut witnesses = FxHashSet::default();

        let ip_univ = compute_ip_universal_states(fst);
        for (q, &is_univ) in ip_univ.iter().enumerate() {
            if is_univ {
                witnesses.insert(pack(q as u32, target_len, target_len));
            }
        }

        UniversalityFilter {
            witnesses,
            pos_index: FxHashMap::default(),
            pos_sizes: Vec::new(),
            neg_index: FxHashMap::default(),
            neg_next: 0,
        }
    }

    /// Add a set to the positive (known universal) cache.
    fn add_pos(&mut self, nfa_set: &[u64]) {
        let eid = self.pos_sizes.len() as u32;
        self.pos_sizes.push(nfa_set.len());
        for &e in nfa_set {
            self.pos_index.entry(e).or_default().push(eid);
        }
    }

    /// Add a set to the negative (known non-universal) cache.
    fn add_neg(&mut self, nfa_set: &[u64]) {
        let eid = self.neg_next;
        self.neg_next += 1;
        for &e in nfa_set {
            self.neg_index.entry(e).or_default().push(eid);
        }
    }

    /// Is there a known-universal set u such that u ⊆ nfa_set?
    /// Uses hit-counting: for each element in nfa_set, increment hit count
    /// for each entry containing that element. If any entry's count reaches
    /// its stored size, that entry is a subset.
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

    /// Is there a known-non-universal set nu such that nfa_set ⊆ nu?
    /// Uses intersection of entry-ID sets: for each element of nfa_set,
    /// look up which negative entries contain it. If the intersection across
    /// all elements is non-empty, some negative entry is a superset.
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

    /// BFS universality check: does the DFA state accept Σ*?
    fn bfs_universal(
        &self,
        sid: u32,
        nfa: &PrecoverNFA,
        arena: &mut PowersetArena,
        num_source_symbols: usize,
        stats: &mut ProfileStats,
    ) -> bool {
        if !arena.is_final[sid as usize] {
            return false;
        }

        let mut sub_visited: FxHashSet<u32> = FxHashSet::default();
        let mut sub_worklist: VecDeque<u32> = VecDeque::new();

        sub_visited.insert(sid);
        sub_worklist.push_back(sid);

        while let Some(cur) = sub_worklist.pop_front() {
            if !arena.is_final[cur as usize] {
                return false;
            }

            stats.universal_sub_bfs_states += 1;

            let cur_set = arena.sets[cur as usize].clone();

            let all_arcs = nfa.compute_all_arcs(&cur_set);
            stats.universal_compute_arcs_calls += 1;

            if all_arcs.len() < num_source_symbols {
                return false;
            }

            for (_sym, successor) in &all_arcs {
                let any_final = successor.iter().any(|&s| nfa.is_final(s));
                let dest_id = arena.intern(successor.clone(), any_final);

                if sub_visited.insert(dest_id) {
                    sub_worklist.push_back(dest_id);
                }
            }
        }

        true
    }

    /// Main entry point: determine if a DFA state is universal.
    fn is_universal(
        &mut self,
        sid: u32,
        nfa: &PrecoverNFA,
        arena: &mut PowersetArena,
        num_source_symbols: usize,
        stats: &mut ProfileStats,
    ) -> bool {
        let nfa_set = &arena.sets[sid as usize];

        // 1. Witness check: any element is an ip-universal witness
        if nfa_set.iter().any(|e| self.witnesses.contains(e)) {
            let set_clone = nfa_set.to_vec();
            self.add_pos(&set_clone);
            return true;
        }

        // 2. Superset monotonicity: is nfa_set ⊇ some known-universal set?
        if self.has_pos_subset(nfa_set) {
            return true;
        }

        // 3. Subset monotonicity: is nfa_set ⊆ some known-non-universal set?
        if self.has_neg_superset(nfa_set) {
            return false;
        }

        // 5. BFS fallback
        let nfa_set_clone = arena.sets[sid as usize].to_vec();
        let result = self.bfs_universal(sid, nfa, arena, num_source_symbols, stats);
        if result {
            self.add_pos(&nfa_set_clone);
        } else {
            self.add_neg(&nfa_set_clone);
        }
        result
    }
}

/// Fused BFS that performs determinization + universality detection + Q/R partitioning.
pub fn decompose(fst: &Fst, target: &[u32]) -> DecompResult {
    let total_start = Instant::now();

    let mut stats = ProfileStats {
        total_ms: 0.0,
        init_ms: 0.0,
        bfs_ms: 0.0,
        compute_arcs_ms: 0.0,
        compute_arcs_calls: 0,
        intern_ms: 0.0,
        intern_calls: 0,
        universal_ms: 0.0,
        universal_calls: 0,
        universal_true: 0,
        universal_false: 0,
        universal_sub_bfs_states: 0,
        universal_compute_arcs_calls: 0,
        dfa_states: 0,
        total_arcs: 0,
        q_stops: 0,
        r_stops: 0,
        max_powerset_size: 0,
        avg_powerset_size: 0.0,
        eps_cache_hits: 0,
        eps_cache_misses: 0,
    };

    let init_start = Instant::now();

    let nfa = PrecoverNFA::new(fst, target);
    let mut arena = PowersetArena::new();
    let num_source_symbols = fst.source_alphabet.len();

    // 1. Compute epsilon-closed initial powerset state
    let raw_starts = nfa.start_states();
    let mut init_closed = Vec::new();
    nfa.eps_closure_set(&raw_starts, &mut init_closed);

    let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
    let start_id = arena.intern(init_closed, any_final);

    let mut worklist: VecDeque<u32> = VecDeque::new();
    let mut visited: FxHashSet<u32> = FxHashSet::default();

    let mut arc_src: Vec<u32> = Vec::new();
    let mut arc_lbl: Vec<u32> = Vec::new();
    let mut arc_dst: Vec<u32> = Vec::new();

    let mut q_stop: Vec<u32> = Vec::new();
    let mut r_stop: Vec<u32> = Vec::new();

    let mut filter = UniversalityFilter::new(fst, target.len() as u32);

    worklist.push_back(start_id);
    visited.insert(start_id);

    stats.init_ms = init_start.elapsed().as_secs_f64() * 1000.0;

    let bfs_start = Instant::now();
    let mut bfs_iterations: u64 = 0;

    // 2. BFS
    while let Some(sid) = worklist.pop_front() {
        bfs_iterations += 1;

        // Log progress every 1000 iterations
        if bfs_iterations % 1000 == 0 {
            let elapsed = bfs_start.elapsed().as_secs_f64();
            eprintln!(
                "[decompose] iter={}, visited={}, arena={}, arcs={}, q_stops={}, r_stops={}, elapsed={:.1}s",
                bfs_iterations,
                visited.len(),
                arena.len(),
                arc_src.len(),
                q_stop.len(),
                r_stop.len(),
                elapsed,
            );
        }

        if arena.is_final[sid as usize] {
            let uni_start = Instant::now();
            stats.universal_calls += 1;
            let is_uni = filter.is_universal(sid, &nfa, &mut arena, num_source_symbols, &mut stats);
            stats.universal_ms += uni_start.elapsed().as_secs_f64() * 1000.0;

            if is_uni {
                stats.universal_true += 1;
                q_stop.push(sid);
                continue; // don't expand universal states
            } else {
                stats.universal_false += 1;
                r_stop.push(sid); // fall through to expand
            }
        }

        // 3. Expand arcs using batch computation
        let nfa_set = arena.sets[sid as usize].clone();

        // Track powerset sizes
        let pset_size = nfa_set.len();
        if pset_size > stats.max_powerset_size {
            stats.max_powerset_size = pset_size;
        }

        let arcs_start = Instant::now();
        let all_arcs = nfa.compute_all_arcs(&nfa_set);
        stats.compute_arcs_ms += arcs_start.elapsed().as_secs_f64() * 1000.0;
        stats.compute_arcs_calls += 1;

        for (x, successor) in all_arcs {
            let intern_start = Instant::now();
            let succ_final = successor.iter().any(|&s| nfa.is_final(s));
            let dest_id = arena.intern(successor, succ_final);
            stats.intern_ms += intern_start.elapsed().as_secs_f64() * 1000.0;
            stats.intern_calls += 1;

            arc_src.push(sid);
            arc_lbl.push(x);
            arc_dst.push(dest_id);

            if visited.insert(dest_id) {
                worklist.push_back(dest_id);
            }
        }
    }

    stats.bfs_ms = bfs_start.elapsed().as_secs_f64() * 1000.0;

    // Get eps closure stats from NFA
    let (hits, misses) = nfa.eps_cache_stats();
    stats.eps_cache_hits = hits;
    stats.eps_cache_misses = misses;

    // Compute avg powerset size
    let total_pset: usize = arena.sets.iter().map(|s| s.len()).sum();
    stats.avg_powerset_size = if arena.len() > 0 {
        total_pset as f64 / arena.len() as f64
    } else {
        0.0
    };

    // 4. Build Q and R (same arcs, different stop states)
    let num_states = arena.len() as u32;

    stats.dfa_states = num_states;
    stats.total_arcs = arc_src.len() as u64;
    stats.q_stops = q_stop.len() as u32;
    stats.r_stops = r_stop.len() as u32;
    stats.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    let quotient = FsaResult {
        num_states,
        start: vec![start_id],
        stop: q_stop,
        arc_src: arc_src.clone(),
        arc_lbl: arc_lbl.clone(),
        arc_dst: arc_dst.clone(),
    };

    let remainder = FsaResult {
        num_states,
        start: vec![start_id],
        stop: r_stop,
        arc_src,
        arc_lbl,
        arc_dst,
    };

    DecompResult {
        quotient,
        remainder,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_replace_fst(pairs: &[(u32, u32)]) -> Fst {
        let mut arc_src = Vec::new();
        let mut arc_in = Vec::new();
        let mut arc_out = Vec::new();
        let mut arc_dst = Vec::new();
        let mut source_alpha = Vec::new();

        for &(x, y) in pairs {
            arc_src.push(0);
            arc_in.push(x);
            arc_out.push(y);
            arc_dst.push(0);
            source_alpha.push(x);
        }

        source_alpha.sort_unstable();
        source_alpha.dedup();

        Fst::new(1, vec![0], &[0], &arc_src, &arc_in, &arc_out, &arc_dst, source_alpha)
    }

    #[test]
    fn test_replace_decompose_empty_target() {
        let fst = make_replace_fst(&[(1, 10), (2, 11)]);
        let result = decompose(&fst, &[]);
        assert!(!result.quotient.stop.is_empty());
    }

    #[test]
    fn test_replace_decompose_single_char() {
        let fst = make_replace_fst(&[(1, 10), (2, 11)]);
        let result = decompose(&fst, &[10]);
        assert_eq!(result.quotient.start, vec![0]);
        assert_eq!(result.remainder.start, vec![0]);
    }
}
