use crate::decompose::FsaResult;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Minimize a DFA represented as an FsaResult.
///
/// Performs trim (remove unreachable/dead states) followed by
/// partition refinement (Moore's algorithm) to merge equivalent states.
pub fn minimize(fsa: &FsaResult) -> FsaResult {
    let trimmed = trim(fsa);
    if trimmed.num_states <= 1 {
        return trimmed;
    }
    partition_refine(&trimmed)
}

/// Remove states not reachable from start or not co-reachable to any final state.
/// Renumbers states contiguously.
fn trim(fsa: &FsaResult) -> FsaResult {
    let n = fsa.num_states as usize;
    if n == 0 {
        return FsaResult {
            num_states: 0,
            start: vec![],
            stop: vec![],
            arc_src: vec![],
            arc_lbl: vec![],
            arc_dst: vec![],
        };
    }

    let num_arcs = fsa.arc_src.len();

    // Build forward and reverse adjacency lists
    let mut fwd: Vec<Vec<u32>> = vec![vec![]; n];
    let mut rev: Vec<Vec<u32>> = vec![vec![]; n];
    for i in 0..num_arcs {
        let s = fsa.arc_src[i] as usize;
        let d = fsa.arc_dst[i] as usize;
        fwd[s].push(d as u32);
        rev[d].push(s as u32);
    }

    // Forward BFS from start
    let mut fwd_reach = vec![false; n];
    let mut queue: VecDeque<u32> = VecDeque::new();
    for &s in &fsa.start {
        let si = s as usize;
        if si < n && !fwd_reach[si] {
            fwd_reach[si] = true;
            queue.push_back(s);
        }
    }
    while let Some(s) = queue.pop_front() {
        for &d in &fwd[s as usize] {
            if !fwd_reach[d as usize] {
                fwd_reach[d as usize] = true;
                queue.push_back(d);
            }
        }
    }

    // Backward BFS from stop
    let mut bwd_reach = vec![false; n];
    for &s in &fsa.stop {
        let si = s as usize;
        if si < n && !bwd_reach[si] {
            bwd_reach[si] = true;
            queue.push_back(s);
        }
    }
    while let Some(s) = queue.pop_front() {
        for &d in &rev[s as usize] {
            if !bwd_reach[d as usize] {
                bwd_reach[d as usize] = true;
                queue.push_back(d);
            }
        }
    }

    // Keep states reachable from start AND co-reachable to stop
    let mut old_to_new = vec![u32::MAX; n];
    let mut new_id: u32 = 0;
    for i in 0..n {
        if fwd_reach[i] && bwd_reach[i] {
            old_to_new[i] = new_id;
            new_id += 1;
        }
    }

    if new_id == n as u32 {
        // Nothing to trim — return a clone
        return FsaResult {
            num_states: fsa.num_states,
            start: fsa.start.clone(),
            stop: fsa.stop.clone(),
            arc_src: fsa.arc_src.clone(),
            arc_lbl: fsa.arc_lbl.clone(),
            arc_dst: fsa.arc_dst.clone(),
        };
    }

    let new_start: Vec<u32> = fsa.start.iter()
        .filter(|&&s| old_to_new[s as usize] != u32::MAX)
        .map(|&s| old_to_new[s as usize])
        .collect();

    let new_stop: Vec<u32> = fsa.stop.iter()
        .filter(|&&s| old_to_new[s as usize] != u32::MAX)
        .map(|&s| old_to_new[s as usize])
        .collect();

    let mut new_src = Vec::new();
    let mut new_lbl = Vec::new();
    let mut new_dst = Vec::new();

    for i in 0..num_arcs {
        let s = fsa.arc_src[i] as usize;
        let d = fsa.arc_dst[i] as usize;
        if old_to_new[s] != u32::MAX && old_to_new[d] != u32::MAX {
            new_src.push(old_to_new[s]);
            new_lbl.push(fsa.arc_lbl[i]);
            new_dst.push(old_to_new[d]);
        }
    }

    FsaResult {
        num_states: new_id,
        start: new_start,
        stop: new_stop,
        arc_src: new_src,
        arc_lbl: new_lbl,
        arc_dst: new_dst,
    }
}

/// Partition refinement (Moore's algorithm) to merge equivalent DFA states.
fn partition_refine(fsa: &FsaResult) -> FsaResult {
    let n = fsa.num_states as usize;

    // Collect alphabet
    let mut alphabet_set: FxHashSet<u32> = FxHashSet::default();
    for &lbl in &fsa.arc_lbl {
        alphabet_set.insert(lbl);
    }
    let mut alphabet: Vec<u32> = alphabet_set.into_iter().collect();
    alphabet.sort_unstable();
    let k = alphabet.len();

    if k == 0 {
        // No transitions — after trim, all remaining states are start∩stop.
        // Collapse to a single state.
        return FsaResult {
            num_states: 1,
            start: vec![0],
            stop: if fsa.stop.is_empty() { vec![] } else { vec![0] },
            arc_src: vec![],
            arc_lbl: vec![],
            arc_dst: vec![],
        };
    }

    let sym_to_idx: FxHashMap<u32, usize> =
        alphabet.iter().enumerate().map(|(i, &s)| (s, i)).collect();

    // Build transition table: delta[state * k + sym_idx] -> dest (or SINK)
    const SINK: u32 = u32::MAX;
    let mut delta = vec![SINK; n * k];
    for i in 0..fsa.arc_src.len() {
        let s = fsa.arc_src[i] as usize;
        let sym_idx = sym_to_idx[&fsa.arc_lbl[i]];
        delta[s * k + sym_idx] = fsa.arc_dst[i];
    }

    // Initial partition: final vs non-final
    let final_set: FxHashSet<u32> = fsa.stop.iter().copied().collect();
    let mut class = vec![0u32; n];
    let has_non_final = (0..n).any(|i| !final_set.contains(&(i as u32)));
    if has_non_final {
        for i in 0..n {
            class[i] = if final_set.contains(&(i as u32)) { 0 } else { 1 };
        }
    }

    // Iterative refinement
    loop {
        let prev_num_classes = *class.iter().max().unwrap() + 1;

        let mut new_class = vec![0u32; n];
        let mut next_id = 0u32;
        let mut sig_to_class: FxHashMap<Vec<u32>, u32> = FxHashMap::default();

        for s in 0..n {
            // Signature: (current_class, class[delta[s][a0]], class[delta[s][a1]], ...)
            let mut sig = Vec::with_capacity(k + 1);
            sig.push(class[s]);
            for sym_idx in 0..k {
                let dest = delta[s * k + sym_idx];
                if dest == SINK {
                    sig.push(SINK);
                } else {
                    sig.push(class[dest as usize]);
                }
            }

            let cls = *sig_to_class.entry(sig).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            new_class[s] = cls;
        }

        class = new_class;

        if next_id == prev_num_classes {
            // No new splits — partition is stable
            break;
        }
    }

    build_minimized(fsa, &class)
}

/// Build a minimized FsaResult from original FSA and class assignments.
fn build_minimized(fsa: &FsaResult, class: &[u32]) -> FsaResult {
    // Determine number of classes
    let num_classes = *class.iter().max().unwrap() + 1;

    // Map start states
    let mut new_start: Vec<u32> = fsa.start.iter()
        .map(|&s| class[s as usize])
        .collect();
    new_start.sort_unstable();
    new_start.dedup();

    // Map stop states
    let mut new_stop: Vec<u32> = fsa.stop.iter()
        .map(|&s| class[s as usize])
        .collect();
    new_stop.sort_unstable();
    new_stop.dedup();

    // Map arcs and deduplicate
    let mut arc_set: FxHashSet<(u32, u32, u32)> = FxHashSet::default();
    let mut new_src = Vec::new();
    let mut new_lbl = Vec::new();
    let mut new_dst = Vec::new();

    for i in 0..fsa.arc_src.len() {
        let s = class[fsa.arc_src[i] as usize];
        let d = class[fsa.arc_dst[i] as usize];
        let lbl = fsa.arc_lbl[i];
        if arc_set.insert((s, lbl, d)) {
            new_src.push(s);
            new_lbl.push(lbl);
            new_dst.push(d);
        }
    }

    FsaResult {
        num_states: num_classes,
        start: new_start,
        stop: new_stop,
        arc_src: new_src,
        arc_lbl: new_lbl,
        arc_dst: new_dst,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimize_empty() {
        let fsa = FsaResult {
            num_states: 0,
            start: vec![],
            stop: vec![],
            arc_src: vec![],
            arc_lbl: vec![],
            arc_dst: vec![],
        };
        let m = minimize(&fsa);
        assert_eq!(m.num_states, 0);
        assert!(m.start.is_empty());
        assert!(m.stop.is_empty());
    }

    #[test]
    fn test_minimize_single_state() {
        // Single self-loop state that is both start and final
        let fsa = FsaResult {
            num_states: 1,
            start: vec![0],
            stop: vec![0],
            arc_src: vec![0],
            arc_lbl: vec![42],
            arc_dst: vec![0],
        };
        let m = minimize(&fsa);
        assert_eq!(m.num_states, 1);
        assert_eq!(m.start, vec![0]);
        assert_eq!(m.stop, vec![0]);
        assert_eq!(m.arc_src.len(), 1);
    }

    #[test]
    fn test_minimize_already_minimal() {
        // 0 --a--> 1(final), already minimal
        let fsa = FsaResult {
            num_states: 2,
            start: vec![0],
            stop: vec![1],
            arc_src: vec![0],
            arc_lbl: vec![1],
            arc_dst: vec![1],
        };
        let m = minimize(&fsa);
        assert_eq!(m.num_states, 2);
        assert_eq!(m.arc_src.len(), 1);
    }

    #[test]
    fn test_minimize_mergeable_states() {
        // Two equivalent final states: 0 --a--> 1(F), 0 --b--> 2(F)
        // States 1 and 2 are both final with no outgoing arcs, so they should merge.
        let fsa = FsaResult {
            num_states: 3,
            start: vec![0],
            stop: vec![1, 2],
            arc_src: vec![0, 0],
            arc_lbl: vec![1, 2],
            arc_dst: vec![1, 2],
        };
        let m = minimize(&fsa);
        // States 1 and 2 should merge into one state
        assert_eq!(m.num_states, 2);
        assert_eq!(m.arc_src.len(), 2); // two arcs with different labels
    }

    #[test]
    fn test_minimize_removes_unreachable() {
        // State 2 is unreachable from start
        let fsa = FsaResult {
            num_states: 3,
            start: vec![0],
            stop: vec![1],
            arc_src: vec![0, 2],
            arc_lbl: vec![1, 1],
            arc_dst: vec![1, 1],
        };
        let m = minimize(&fsa);
        assert_eq!(m.num_states, 2);
    }

    #[test]
    fn test_minimize_removes_dead_states() {
        // State 2 is reachable but has no path to any final state
        let fsa = FsaResult {
            num_states: 3,
            start: vec![0],
            stop: vec![1],
            arc_src: vec![0, 0],
            arc_lbl: vec![1, 2],
            arc_dst: vec![1, 2],
        };
        let m = minimize(&fsa);
        // State 2 is dead (no path to final), should be trimmed
        assert_eq!(m.num_states, 2);
        assert_eq!(m.arc_src.len(), 1); // only arc 0--a-->1 remains
    }

    #[test]
    fn test_minimize_chain_merge() {
        // 0 --a--> 1 --b--> 2(F)
        // 0 --c--> 3 --b--> 4(F)
        // States (1,3) should merge and (2,4) should merge
        let fsa = FsaResult {
            num_states: 5,
            start: vec![0],
            stop: vec![2, 4],
            arc_src: vec![0, 1, 0, 3],
            arc_lbl: vec![1, 2, 3, 2],
            arc_dst: vec![1, 2, 3, 4],
        };
        let m = minimize(&fsa);
        // 0, {1,3}, {2,4} -> 3 states
        assert_eq!(m.num_states, 3);
    }

    #[test]
    fn test_minimize_preserves_language() {
        // DFA accepting {a, b}* with two copies of the accept state that should merge.
        // 0(F) --a--> 1(F) --a--> 0, 0 --b--> 1, 1 --b--> 0
        // States 0 and 1 are equivalent: both final, same transition structure.
        let fsa = FsaResult {
            num_states: 2,
            start: vec![0],
            stop: vec![0, 1],
            arc_src: vec![0, 0, 1, 1],
            arc_lbl: vec![1, 2, 1, 2],
            arc_dst: vec![1, 1, 0, 0],
        };
        let m = minimize(&fsa);
        assert_eq!(m.num_states, 1);
        assert_eq!(m.start, vec![0]);
        assert_eq!(m.stop, vec![0]);
        assert_eq!(m.arc_src.len(), 2); // self-loops on a and b
    }
}
