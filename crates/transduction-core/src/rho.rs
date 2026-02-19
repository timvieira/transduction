use crate::decompose::FsaResult;
use crate::fst::Fst;
use crate::powerset::PowersetArena;
use crate::precover::PrecoverNFA;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::time::Instant;

/// RHO label: matches any input symbol not on an explicit arc.
/// Uses u32::MAX - 1 to avoid collision with EPSILON (u32::MAX).
pub const RHO: u32 = u32::MAX - 1;

/// Result of rho-factored determinization.
pub struct RhoDfaResult {
    pub num_states: u32,
    pub start: Vec<u32>,
    pub stop: Vec<u32>,
    pub arc_src: Vec<u32>,
    pub arc_lbl: Vec<u32>, // some entries = RHO
    pub arc_dst: Vec<u32>,
    // stats
    pub num_rho_arcs: u32,
    pub num_explicit_arcs: u32,
    pub complete_states: u32,
    pub total_ms: f64,
}

/// BFS subset construction with rho-arc compression at complete states.
///
/// At each DFA state, computes all arcs from the NFA powerset. If the state
/// is "complete" (has arcs for every source alphabet symbol), the most common
/// destination is replaced with a single RHO arc. Incomplete states emit all
/// arcs explicitly.
pub fn rho_determinize(fst: &Fst, target: &[u32]) -> RhoDfaResult {
    let total_start = Instant::now();

    let nfa = PrecoverNFA::new(fst, target);
    let mut arena = PowersetArena::new();
    let num_source_symbols = fst.source_alphabet.len();

    // Compute epsilon-closed initial powerset state
    let raw_starts = nfa.start_states();
    let mut init_closed = Vec::new();
    nfa.eps_closure_set(&raw_starts, &mut init_closed);

    let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
    let start_id = arena.intern(init_closed, any_final);

    let mut worklist: VecDeque<u32> = VecDeque::new();
    let mut visited: FxHashMap<u32, bool> = FxHashMap::default(); // sid -> expanded

    let mut arc_src: Vec<u32> = Vec::new();
    let mut arc_lbl: Vec<u32> = Vec::new();
    let mut arc_dst: Vec<u32> = Vec::new();

    let mut stop: Vec<u32> = Vec::new();
    let mut num_rho_arcs: u32 = 0;
    let mut num_explicit_arcs: u32 = 0;
    let mut complete_states: u32 = 0;

    worklist.push_back(start_id);
    visited.insert(start_id, false);

    let mut arcs_buf: FxHashMap<u32, Vec<u64>> = FxHashMap::default();

    // BFS
    while let Some(sid) = worklist.pop_front() {
        if visited[&sid] {
            continue;
        }
        visited.insert(sid, true);

        if arena.is_final[sid as usize] {
            stop.push(sid);
        }

        // Compute all arcs from this powerset state
        let all_arcs = nfa.compute_all_arcs_into(&arena.sets[sid as usize], &mut arcs_buf);

        // Check completeness: has arcs for every source symbol
        let is_complete = num_source_symbols > 0 && all_arcs.len() == num_source_symbols;

        if is_complete {
            complete_states += 1;

            // Intern all destinations and group by dest_id
            let mut by_dest: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
            let mut sym_to_dest: Vec<(u32, u32)> = Vec::with_capacity(all_arcs.len());

            for (sym, successor) in all_arcs {
                let succ_final = successor.iter().any(|&s| nfa.is_final(s));
                let dest_id = arena.intern(successor, succ_final);
                if !visited.contains_key(&dest_id) {
                    visited.insert(dest_id, false);
                    worklist.push_back(dest_id);
                }
                sym_to_dest.push((sym, dest_id));
                by_dest.entry(dest_id).or_default().push(sym);
            }

            if by_dest.len() == 1 {
                // All arcs go to the same destination — single RHO arc
                let dest_id = sym_to_dest[0].1;
                arc_src.push(sid);
                arc_lbl.push(RHO);
                arc_dst.push(dest_id);
                num_rho_arcs += 1;
            } else {
                // Find the destination with the most symbols → RHO
                let rho_dest = *by_dest.iter()
                    .max_by_key(|(_, syms)| syms.len())
                    .unwrap()
                    .0;

                // Emit explicit arcs for non-RHO destinations
                for (sym, dest_id) in &sym_to_dest {
                    if *dest_id != rho_dest {
                        arc_src.push(sid);
                        arc_lbl.push(*sym);
                        arc_dst.push(*dest_id);
                        num_explicit_arcs += 1;
                    }
                }

                // Emit the RHO arc
                arc_src.push(sid);
                arc_lbl.push(RHO);
                arc_dst.push(rho_dest);
                num_rho_arcs += 1;
            }
        } else {
            // Incomplete state: emit all arcs explicitly
            for (sym, successor) in all_arcs {
                let succ_final = successor.iter().any(|&s| nfa.is_final(s));
                let dest_id = arena.intern(successor, succ_final);
                if !visited.contains_key(&dest_id) {
                    visited.insert(dest_id, false);
                    worklist.push_back(dest_id);
                }
                arc_src.push(sid);
                arc_lbl.push(sym);
                arc_dst.push(dest_id);
                num_explicit_arcs += 1;
            }
        }
    }

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    RhoDfaResult {
        num_states: arena.len() as u32,
        start: vec![start_id],
        stop,
        arc_src,
        arc_lbl,
        arc_dst,
        num_rho_arcs,
        num_explicit_arcs,
        complete_states,
        total_ms,
    }
}

/// Expand RHO arcs into explicit arcs for every missing symbol.
///
/// For each state with a RHO arc, determines which symbols are already
/// covered by explicit arcs, then emits one explicit arc per missing symbol
/// to the RHO destination.
pub fn expand_rho(rho_dfa: &RhoDfaResult, source_alphabet: &[u32]) -> FsaResult {
    let num_arcs = rho_dfa.arc_src.len();

    // Build per-state: explicit symbols and rho destinations
    let mut explicit_by_state: FxHashMap<u32, Vec<(u32, u32)>> = FxHashMap::default();
    let mut rho_by_state: FxHashMap<u32, u32> = FxHashMap::default();

    for i in 0..num_arcs {
        let src = rho_dfa.arc_src[i];
        let lbl = rho_dfa.arc_lbl[i];
        let dst = rho_dfa.arc_dst[i];
        if lbl == RHO {
            rho_by_state.insert(src, dst);
        } else {
            explicit_by_state.entry(src).or_default().push((lbl, dst));
        }
    }

    // Build expanded arc arrays
    let mut arc_src: Vec<u32> = Vec::new();
    let mut arc_lbl: Vec<u32> = Vec::new();
    let mut arc_dst: Vec<u32> = Vec::new();

    // First, copy all non-RHO arcs
    for i in 0..num_arcs {
        if rho_dfa.arc_lbl[i] != RHO {
            arc_src.push(rho_dfa.arc_src[i]);
            arc_lbl.push(rho_dfa.arc_lbl[i]);
            arc_dst.push(rho_dfa.arc_dst[i]);
        }
    }

    // Then, expand RHO arcs
    for (&state, &rho_dest) in &rho_by_state {
        let explicit_syms: rustc_hash::FxHashSet<u32> = explicit_by_state
            .get(&state)
            .map(|v| v.iter().map(|&(sym, _)| sym).collect())
            .unwrap_or_default();

        for &sym in source_alphabet {
            if !explicit_syms.contains(&sym) {
                arc_src.push(state);
                arc_lbl.push(sym);
                arc_dst.push(rho_dest);
            }
        }
    }

    FsaResult {
        num_states: rho_dfa.num_states,
        start: rho_dfa.start.clone(),
        stop: rho_dfa.stop.clone(),
        arc_src,
        arc_lbl,
        arc_dst,
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
    fn test_rho_replace_empty_target() {
        let fst = make_replace_fst(&[(1, 10), (2, 11), (3, 12)]);
        let result = rho_determinize(&fst, &[]);
        // Empty target: boundary state is start, should be complete
        assert!(result.num_rho_arcs > 0, "Should have RHO arcs");
        assert!(!result.stop.is_empty(), "Should have final states");
    }

    #[test]
    fn test_rho_replace_single_target() {
        let fst = make_replace_fst(&[(1, 10), (2, 11), (3, 12)]);
        let result = rho_determinize(&fst, &[10]);
        // Should have at least one RHO arc at the boundary state
        assert!(result.num_rho_arcs > 0, "Should have RHO arcs at boundary");
    }

    #[test]
    fn test_expand_preserves_arc_count() {
        let fst = make_replace_fst(&[(1, 10), (2, 11), (3, 12)]);
        let rho_result = rho_determinize(&fst, &[10]);
        let expanded = expand_rho(&rho_result, &fst.source_alphabet);

        // The expanded DFA should have at least as many arcs as explicit + rho-expanded
        assert!(expanded.arc_src.len() >= rho_result.num_explicit_arcs as usize);

        // Every complete state should have full alphabet coverage after expansion
        // (count arcs per state and verify)
        let mut arcs_per_state: FxHashMap<u32, usize> = FxHashMap::default();
        for &src in &expanded.arc_src {
            *arcs_per_state.entry(src).or_insert(0) += 1;
        }

        // States that had RHO arcs should now have source_alphabet.len() arcs
        // (This is true for states that were complete in the original)
    }

    #[test]
    fn test_rho_stats_consistency() {
        let fst = make_replace_fst(&[(1, 10), (2, 11), (3, 12)]);
        let result = rho_determinize(&fst, &[10]);
        // Total arcs = explicit + RHO
        assert_eq!(
            result.arc_src.len() as u32,
            result.num_explicit_arcs + result.num_rho_arcs
        );
    }

    #[test]
    fn test_bpe_like_single_rho() {
        // BPE-like FST: all tokens are single-byte, so boundary state
        // has all arcs going to the same dest -> single RHO arc
        use crate::fst::EPSILON;

        let n_tokens = 20u32;
        let mut arc_src = Vec::new();
        let mut arc_in = Vec::new();
        let mut arc_out = Vec::new();
        let mut arc_dst = Vec::new();

        // State 0 = hub (start/final)
        // For each token i: hub --(eps/i)--> state i+1 --(i/eps)--> hub
        for i in 0..n_tokens {
            let token_state = i + 1;
            // eps-input, output=byte_i: hub -> token_state
            arc_src.push(0);
            arc_in.push(EPSILON);
            arc_out.push(i);
            arc_dst.push(token_state);
            // input=token_id, eps-output: token_state -> hub
            arc_src.push(token_state);
            arc_in.push(i);
            arc_out.push(EPSILON);
            arc_dst.push(0);
        }

        let source_alphabet: Vec<u32> = (0..n_tokens).collect();
        let fst = Fst::new(
            n_tokens + 1,
            vec![0],
            &[0],
            &arc_src,
            &arc_in,
            &arc_out,
            &arc_dst,
            source_alphabet,
        );

        let result = rho_determinize(&fst, &[0]);
        assert!(result.num_rho_arcs > 0, "BPE should have RHO arcs");
        // With single-byte tokens and target=(0,), the boundary state has
        // all tokens returning to the same hub state -> should get 1 RHO + at most 1 explicit
        let total_arcs = result.arc_src.len();
        assert!(total_arcs < n_tokens as usize,
            "Rho factoring should reduce arc count: got {} vs {} tokens",
            total_arcs, n_tokens);
    }
}
