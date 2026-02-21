//! Token-level decomposition for BPE-like FSTs.
//!
//! Instead of tracking `(fst_state, buf_pos)` pairs through intermediate states,
//! this module collapses each token into a single transition that advances buf_pos
//! by the token's byte length. NFA states are just position indices `0..=target_len`,
//! and powerset states are compact bitsets over those positions.
//!
//! For a target of length N, the resulting DFA typically has O(N) states instead of
//! the O(7000*N) states produced by the generic approach with intermediate states.

use crate::fst::{Fst, EPSILON};
use crate::decompose::{FsaResult, DecompResult, ProfileStats};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Token byte extraction
// ---------------------------------------------------------------------------

/// Extract `(token_id, byte_sequence)` pairs from a BPE-like FST.
///
/// Assumes the FST has a hub structure: start state(s) with non-epsilon input
/// arcs leading into chains of epsilon-input arcs that return to a start state.
pub(crate) fn extract_token_bytes(fst: &Fst) -> Vec<(u32, Vec<u32>)> {
    let start_set: FxHashSet<u32> = fst.start_states.iter().copied().collect();
    let mut tokens = Vec::new();

    for &start in &fst.start_states {
        for arc in fst.arcs_from(start) {
            if arc.input == EPSILON {
                continue;
            }

            let token_id = arc.input;
            let mut bytes = Vec::new();
            if arc.output != EPSILON {
                bytes.push(arc.output);
            }

            // Follow the epsilon-input chain back to a start state.
            let mut current = arc.dest;
            while !start_set.contains(&current) {
                let mut found = false;
                for a in fst.arcs_from(current) {
                    if a.input == EPSILON {
                        if a.output != EPSILON {
                            bytes.push(a.output);
                        }
                        current = a.dest;
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
            }

            tokens.push((token_id, bytes));
        }
    }

    tokens
}

// ---------------------------------------------------------------------------
// Byte trie for fast prefix matching
// ---------------------------------------------------------------------------

pub(crate) struct ByteTrie {
    pub(crate) children: Vec<FxHashMap<u32, u32>>,
    /// (token_id, byte_length) for tokens completing at each node.
    pub(crate) completions: Vec<Vec<(u32, u32)>>,
}

impl ByteTrie {
    pub(crate) fn new() -> Self {
        ByteTrie {
            children: vec![FxHashMap::default()],
            completions: vec![Vec::new()],
        }
    }

    pub(crate) fn insert(&mut self, token_id: u32, bytes: &[u32]) {
        let mut node = 0usize;
        for &b in bytes {
            if let Some(&child) = self.children[node].get(&b) {
                node = child as usize;
            } else {
                let next = self.children.len() as u32;
                self.children.push(FxHashMap::default());
                self.completions.push(Vec::new());
                self.children[node].insert(b, next);
                node = next as usize;
            }
        }
        self.completions[node].push((token_id, bytes.len() as u32));
    }

    /// Collect all tokens whose byte sequences match `target[pos..]`.
    ///
    /// Two cases:
    /// 1. **Full match**: the token's entire byte sequence fits within
    ///    `target[pos..]`. Advance = byte_length.
    /// 2. **Partial match**: the token's byte sequence extends beyond the
    ///    target. The first `target_len - pos` bytes match, and the
    ///    remaining bytes are consumed "post-target". Advance = target_len - pos.
    pub(crate) fn matches_at(&self, target: &[u32], pos: usize) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        let mut node = 0usize;
        let target_len = target.len();

        for i in pos..target_len {
            match self.children[node].get(&target[i]) {
                Some(&child) => {
                    node = child as usize;
                    // Full match: token byte sequence ends here.
                    result.extend_from_slice(&self.completions[node]);
                }
                None => return result,
            }
        }

        // We've consumed all remaining target bytes. Tokens in the subtree
        // below `node` have byte sequences that START with target[pos..target_len]
        // but extend further. Their extra bytes are consumed post-target.
        // Advance for these = target_len - pos.
        let advance_cap = (target_len - pos) as u32;
        self.collect_subtree(&node, advance_cap, &mut result);

        result
    }

    /// Collect all tokens in the subtree below (not including) `node`,
    /// with advance capped at `advance_cap`.
    fn collect_subtree(&self, node: &usize, advance_cap: u32, result: &mut Vec<(u32, u32)>) {
        for (&_byte, &child) in &self.children[*node] {
            let child = child as usize;
            for &(tid, _byte_len) in &self.completions[child] {
                result.push((tid, advance_cap));
            }
            self.collect_subtree(&child, advance_cap, result);
        }
    }
}

// ---------------------------------------------------------------------------
// Bitset-based position sets
// ---------------------------------------------------------------------------

const BITS: usize = 64;

#[derive(Clone)]
struct PosSet {
    words: Vec<u64>,
}

impl PosSet {
    #[inline]
    fn new(num_positions: usize) -> Self {
        PosSet {
            words: vec![0u64; (num_positions + BITS - 1) / BITS],
        }
    }

    #[inline]
    fn set(&mut self, pos: usize) {
        self.words[pos / BITS] |= 1u64 << (pos % BITS);
    }

    #[inline]
    fn test(&self, pos: usize) -> bool {
        (self.words[pos / BITS] >> (pos % BITS)) & 1 != 0
    }

    fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }
}

// ---------------------------------------------------------------------------
// Position-set arena (interning)
// ---------------------------------------------------------------------------

struct PosSetArena {
    map: FxHashMap<Vec<u64>, u32>,
    sets: Vec<PosSet>,
    is_final: Vec<bool>,
}

impl PosSetArena {
    fn new() -> Self {
        PosSetArena {
            map: FxHashMap::default(),
            sets: Vec::new(),
            is_final: Vec::new(),
        }
    }

    fn intern(&mut self, set: PosSet, final_flag: bool) -> u32 {
        if let Some(&id) = self.map.get(&set.words) {
            return id;
        }
        let id = self.sets.len() as u32;
        self.map.insert(set.words.clone(), id);
        self.sets.push(set);
        self.is_final.push(final_flag);
        id
    }

    fn len(&self) -> usize {
        self.sets.len()
    }
}

// ---------------------------------------------------------------------------
// Token-level decomposition
// ---------------------------------------------------------------------------

/// Decompose using token-level position tracking.
///
/// This is equivalent to the generic `decompose()` but dramatically faster for
/// BPE-like FSTs because it eliminates intermediate states entirely.  The DFA
/// states are subsets of `{0, 1, …, target_len}` rather than subsets of the
/// `O(fst_states × target_len)` NFA state space.
pub fn decompose_token_level(fst: &Fst, target: &[u32]) -> DecompResult {
    let total_start = Instant::now();
    let target_len = target.len();
    let num_positions = target_len + 1;

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

    // 1. Extract token byte sequences and build trie.
    let token_list = extract_token_bytes(fst);
    let mut trie = ByteTrie::new();
    for &(token_id, ref bytes) in &token_list {
        if !bytes.is_empty() {
            trie.insert(token_id, bytes);
        }
    }

    // 2. Precompute matches at each target position.
    //    matches[p] = Vec<(token_id, advance_by)>
    let mut matches: Vec<Vec<(u32, u32)>> = Vec::with_capacity(target_len);
    for p in 0..target_len {
        matches.push(trie.matches_at(target, p));
    }

    // Collect zero-length token IDs (e.g., delete_b's b→ε).
    // These create self-loops: they don't advance the buffer position.
    let zero_len_tokens: Vec<u32> = token_list.iter()
        .filter(|(_, bytes)| bytes.is_empty())
        .map(|(tid, _)| *tid)
        .collect();

    // 3. Seed BFS.
    let mut arena = PosSetArena::new();

    let mut start_set = PosSet::new(num_positions);
    start_set.set(0);
    let start_final = target_len == 0;
    let start_id = arena.intern(start_set, start_final);

    let mut worklist: VecDeque<u32> = VecDeque::new();
    let mut visited: FxHashSet<u32> = FxHashSet::default();

    let mut arc_src: Vec<u32> = Vec::new();
    let mut arc_lbl: Vec<u32> = Vec::new();
    let mut arc_dst: Vec<u32> = Vec::new();

    let mut q_stop: Vec<u32> = Vec::new();
    let r_stop: Vec<u32> = Vec::new();

    worklist.push_back(start_id);
    visited.insert(start_id);

    stats.init_ms = init_start.elapsed().as_secs_f64() * 1000.0;
    let bfs_start = Instant::now();

    // 4. BFS
    while let Some(sid) = worklist.pop_front() {
        // Final states are universal (since all_input_universal).
        if arena.is_final[sid as usize] {
            stats.universal_calls += 1;
            stats.universal_true += 1;
            q_stop.push(sid);
            continue;
        }

        let cur_set = &arena.sets[sid as usize];
        let pset_size = cur_set.count() as usize;
        if pset_size > stats.max_powerset_size {
            stats.max_powerset_size = pset_size;
        }

        let arcs_start = Instant::now();

        // Group successors by token_id.
        let mut by_token: FxHashMap<u32, PosSet> = FxHashMap::default();

        for p in 0..target_len {
            if !cur_set.test(p) {
                continue;
            }
            for &(tid, advance) in &matches[p] {
                let new_pos = p + advance as usize;
                if new_pos <= target_len {
                    by_token
                        .entry(tid)
                        .or_insert_with(|| PosSet::new(num_positions))
                        .set(new_pos);
                }
            }
        }

        stats.compute_arcs_ms += arcs_start.elapsed().as_secs_f64() * 1000.0;
        stats.compute_arcs_calls += 1;

        // Zero-length tokens create self-loops (same position set → same DFA state).
        for &tid in &zero_len_tokens {
            arc_src.push(sid);
            arc_lbl.push(tid);
            arc_dst.push(sid);
        }

        for (token_id, succ_set) in by_token {
            if succ_set.is_empty() {
                continue;
            }

            let intern_start = Instant::now();
            let is_final = succ_set.test(target_len);
            let dest_id = arena.intern(succ_set, is_final);
            stats.intern_ms += intern_start.elapsed().as_secs_f64() * 1000.0;
            stats.intern_calls += 1;

            arc_src.push(sid);
            arc_lbl.push(token_id);
            arc_dst.push(dest_id);

            if visited.insert(dest_id) {
                worklist.push_back(dest_id);
            }
        }
    }

    stats.bfs_ms = bfs_start.elapsed().as_secs_f64() * 1000.0;

    // 5. Compute summary stats.
    let total_pset: usize = arena.sets.iter().map(|s| s.count() as usize).sum();
    stats.avg_powerset_size = if arena.len() > 0 {
        total_pset as f64 / arena.len() as f64
    } else {
        0.0
    };

    let num_states = arena.len() as u32;
    stats.dfa_states = num_states;
    stats.total_arcs = arc_src.len() as u64;
    stats.q_stops = q_stop.len() as u32;
    stats.r_stops = r_stop.len() as u32;
    stats.total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    // 6. Build Q and R.
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

    #[test]
    fn test_extract_simple() {
        // Simple replace: token 0 -> byte 10, token 1 -> byte 11
        let fst = Fst::new(
            1,
            vec![0],
            &[0],
            &[0, 0],
            &[0, 1],
            &[10, 11],
            &[0, 0],
            vec![0, 1],
        );

        let tokens = extract_token_bytes(&fst);
        assert_eq!(tokens.len(), 2);
        // Both should be single-byte tokens
        for &(_, ref bytes) in &tokens {
            assert_eq!(bytes.len(), 1);
        }
    }

    #[test]
    fn test_trie_matches() {
        let mut trie = ByteTrie::new();
        trie.insert(0, &[10]);        // "a"
        trie.insert(1, &[10, 11]);    // "ab"
        trie.insert(2, &[11]);        // "b"

        let target = vec![10, 11, 10];

        // At position 0: "a" (len 1) and "ab" (len 2) match
        let m0 = trie.matches_at(&target, 0);
        assert_eq!(m0.len(), 2);

        // At position 1: "b" (len 1) matches
        let m1 = trie.matches_at(&target, 1);
        assert_eq!(m1.len(), 1);

        // At position 2: "a" (len 1) full match, "ab" (len 2) partial match
        // (first byte 10 matches target[2], second byte 11 is post-target)
        let m2 = trie.matches_at(&target, 2);
        assert_eq!(m2.len(), 2);
    }

    #[test]
    fn test_decompose_simple_replace() {
        // token 0 -> byte 10, token 1 -> byte 11
        let fst = Fst::new(
            1,
            vec![0],
            &[0],
            &[0, 0],
            &[0, 1],
            &[10, 11],
            &[0, 0],
            vec![0, 1],
        );

        let result = decompose_token_level(&fst, &[10]);
        // Q should accept token 0 (then anything)
        assert!(!result.quotient.stop.is_empty());
        // R might be empty (token 0 exactly produces [10])
        assert_eq!(result.remainder.stop.len(), 0);
    }

    #[test]
    fn test_posset_basics() {
        let mut s = PosSet::new(100);
        assert!(s.is_empty());
        s.set(0);
        s.set(50);
        s.set(99);
        assert!(!s.is_empty());
        assert!(s.test(0));
        assert!(s.test(50));
        assert!(s.test(99));
        assert!(!s.test(1));
        assert_eq!(s.count(), 3);
    }
}
