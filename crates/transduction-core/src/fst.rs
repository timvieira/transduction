use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

pub const EPSILON: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub struct FstArc {
    pub input: u32,
    pub output: u32,
    pub dest: u32,
}

/// FST stored in CSR format with four auxiliary indexes mirroring the Python
/// `LazyPrecoverNFA` indexes.
pub struct Fst {
    pub num_states: u32,
    pub start_states: Vec<u32>,
    pub is_final: Vec<bool>,

    // CSR arc storage (sorted by source state)
    pub offsets: Vec<u32>, // length num_states+1
    pub arcs: Vec<FstArc>,

    // index_iy_xj: (state, output) → [(input, dest)]
    pub index_iy_xj: FxHashMap<(u32, u32), Vec<(u32, u32)>>,
    // index_i_xj: state → [(input, dest)]
    pub index_i_xj: Vec<Vec<(u32, u32)>>,
    // index_ix_j: (state, input) → [dest]
    pub index_ix_j: FxHashMap<(u32, u32), Vec<u32>>,
    // index_ixy_j: (state, input, output) → [dest]
    pub index_ixy_j: FxHashMap<(u32, u32, u32), Vec<u32>>,

    pub source_alphabet: Vec<u32>,

    /// Per-state flag: true if the state has at least one arc with non-epsilon input.
    /// Used by precover epsilon-closure filtering to identify "productive" NFA states.
    pub has_non_eps_input: Vec<bool>,
}

/// Epsilon closure over input-side epsilon arcs (for the input projection).
fn ip_eps_close(states: &[u32], fst: &Fst) -> Vec<u32> {
    let mut visited: FxHashSet<u32> = FxHashSet::default();
    let mut worklist: VecDeque<u32> = VecDeque::new();
    for &s in states {
        if visited.insert(s) {
            worklist.push_back(s);
        }
    }
    while let Some(s) = worklist.pop_front() {
        if let Some(dests) = fst.index_ix_j.get(&(s, EPSILON)) {
            for &j in dests {
                if visited.insert(j) {
                    worklist.push_back(j);
                }
            }
        }
    }
    let mut result: Vec<u32> = visited.into_iter().collect();
    result.sort_unstable();
    result
}

/// Greatest-fixpoint computation of ip-universal FST states.
///
/// A state q is ip-universal if the input projection of the FST, started from
/// ip_eps_close({q}), accepts Σ*. Returns `Vec<bool>` indexed by state ID.
pub fn compute_ip_universal_states(fst: &Fst) -> Vec<bool> {
    let n = fst.num_states as usize;
    let source_alphabet: Vec<u32> = fst.source_alphabet.clone();

    if source_alphabet.is_empty() {
        // No non-eps symbols: universal iff eps-closure contains a final state
        let mut result = vec![false; n];
        for q in 0..n {
            let closure = ip_eps_close(&[q as u32], fst);
            if closure.iter().any(|&s| fst.is_final[s as usize]) {
                result[q] = true;
            }
        }
        return result;
    }

    // Precompute closures[q] = ip_eps_close({q}) for all states
    let closures: Vec<Vec<u32>> = (0..n)
        .map(|q| ip_eps_close(&[q as u32], fst))
        .collect();

    // Precompute by_symbol[q]: symbol -> set of raw destinations from closure[q]
    let mut by_symbol: Vec<FxHashMap<u32, FxHashSet<u32>>> = Vec::with_capacity(n);
    for q in 0..n {
        let mut sym_map: FxHashMap<u32, FxHashSet<u32>> = FxHashMap::default();
        for &s in &closures[q] {
            for &(x, j) in &fst.index_i_xj[s as usize] {
                if x != EPSILON {
                    sym_map.entry(x).or_default().insert(j);
                }
            }
        }
        by_symbol.push(sym_map);
    }

    // Greatest fixpoint iteration
    let mut candidates = vec![true; n];
    let mut changed = true;
    while changed {
        changed = false;
        for q in 0..n {
            if !candidates[q] {
                continue;
            }

            // Must contain a final state
            if !closures[q].iter().any(|&s| fst.is_final[s as usize]) {
                candidates[q] = false;
                changed = true;
                continue;
            }

            // Must be complete on source alphabet
            if !source_alphabet.iter().all(|a| by_symbol[q].contains_key(a)) {
                candidates[q] = false;
                changed = true;
                continue;
            }

            // For each symbol, successor eps-closure must contain >= 1 candidate
            let mut ok = true;
            for a in &source_alphabet {
                if let Some(dests) = by_symbol[q].get(a) {
                    let raw: Vec<u32> = dests.iter().copied().collect();
                    let succ_closure = ip_eps_close(&raw, fst);
                    if !succ_closure.iter().any(|&s| candidates[s as usize]) {
                        ok = false;
                        break;
                    }
                } else {
                    ok = false;
                    break;
                }
            }
            if !ok {
                candidates[q] = false;
                changed = true;
            }
        }
    }

    candidates
}

impl Fst {
    /// Build an FST from parallel arc arrays.
    pub fn new(
        num_states: u32,
        start_states: Vec<u32>,
        final_states: &[u32],
        arc_src: &[u32],
        arc_in: &[u32],
        arc_out: &[u32],
        arc_dst: &[u32],
        source_alphabet: Vec<u32>,
    ) -> Self {
        let n = num_states as usize;
        let num_arcs = arc_src.len();

        // Build is_final
        let mut is_final = vec![false; n];
        for &s in final_states {
            is_final[s as usize] = true;
        }

        // Sort arcs by source state for CSR
        let mut indices: Vec<usize> = (0..num_arcs).collect();
        indices.sort_unstable_by_key(|&i| arc_src[i]);

        let mut offsets = vec![0u32; n + 1];
        let mut arcs = Vec::with_capacity(num_arcs);

        for &idx in &indices {
            let src = arc_src[idx] as usize;
            offsets[src + 1] += 1;
            arcs.push(FstArc {
                input: arc_in[idx],
                output: arc_out[idx],
                dest: arc_dst[idx],
            });
        }

        // Prefix sum
        for i in 1..=n {
            offsets[i] += offsets[i - 1];
        }

        // Build all four indexes in a single pass
        let mut index_iy_xj: FxHashMap<(u32, u32), Vec<(u32, u32)>> = FxHashMap::default();
        let mut index_i_xj: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
        let mut index_ix_j: FxHashMap<(u32, u32), Vec<u32>> = FxHashMap::default();
        let mut index_ixy_j: FxHashMap<(u32, u32, u32), Vec<u32>> = FxHashMap::default();

        for &idx in &indices {
            let i = arc_src[idx];
            let x = arc_in[idx];
            let y = arc_out[idx];
            let j = arc_dst[idx];

            index_iy_xj.entry((i, y)).or_default().push((x, j));
            index_i_xj[i as usize].push((x, j));
            index_ix_j.entry((i, x)).or_default().push(j);
            index_ixy_j.entry((i, x, y)).or_default().push(j);
        }

        let has_non_eps_input: Vec<bool> = (0..n)
            .map(|i| index_i_xj[i].iter().any(|&(x, _)| x != EPSILON))
            .collect();

        Fst {
            num_states,
            start_states,
            is_final,
            offsets,
            arcs,
            index_iy_xj,
            index_i_xj,
            index_ix_j,
            index_ixy_j,
            source_alphabet,
            has_non_eps_input,
        }
    }

    /// Iterate arcs from a given source state.
    #[inline]
    pub fn arcs_from(&self, state: u32) -> &[FstArc] {
        let lo = self.offsets[state as usize] as usize;
        let hi = self.offsets[state as usize + 1] as usize;
        &self.arcs[lo..hi]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_fst() {
        // Simple replace FST: {(1,a), (2,b)}
        let fst = Fst::new(
            1,
            vec![0],
            &[0],
            &[0, 0],
            &[0, 1],
            &[2, 3],
            &[0, 0],
            vec![0, 1],
        );

        assert_eq!(fst.num_states, 1);
        assert!(fst.is_final[0]);
        assert_eq!(fst.arcs_from(0).len(), 2);
    }
}
