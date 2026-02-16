use crate::decompose::FsaResult;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Minimize a DFA represented as an FsaResult.
///
/// Performs trim (remove unreachable/dead states) followed by
/// Hopcroft's partition-refinement algorithm (O(kn log n)) to merge
/// equivalent states.
pub fn minimize(fsa: &FsaResult) -> FsaResult {
    let trimmed = trim(fsa);
    if trimmed.num_states <= 1 {
        return trimmed;
    }
    hopcroft_refine(&trimmed)
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

/// Hopcroft's partition-refinement algorithm for DFA minimization.
///
/// O(kn log n) where k = |alphabet|, n = |states|.  Uses a `find` index
/// for O(1) block lookup and groups pre-images by block to turn the
/// superset check into an O(1) length comparison (matching the Python
/// `min_faster` implementation).
fn hopcroft_refine(fsa: &FsaResult) -> FsaResult {
    let n = fsa.num_states as usize;

    // Collect alphabet
    let mut alphabet_set: FxHashSet<u32> = FxHashSet::default();
    for &lbl in &fsa.arc_lbl {
        alphabet_set.insert(lbl);
    }
    let alphabet: Vec<u32> = alphabet_set.into_iter().collect();

    if alphabet.is_empty() {
        // No transitions — after trim, all remaining states are start∩stop.
        return FsaResult {
            num_states: 1,
            start: vec![0],
            stop: if fsa.stop.is_empty() { vec![] } else { vec![0] },
            arc_src: vec![],
            arc_lbl: vec![],
            arc_dst: vec![],
        };
    }

    // Build reverse transition index: inv[(j, sym_idx)] = {i : i --a--> j}
    let sym_to_idx: FxHashMap<u32, usize> =
        alphabet.iter().enumerate().map(|(i, &s)| (s, i)).collect();
    let k = alphabet.len();
    // inv is stored as a flat Vec of Vecs: inv[j * k + sym_idx] = predecessors
    let mut inv: Vec<Vec<u32>> = vec![vec![]; n * k];
    for arc_i in 0..fsa.arc_src.len() {
        let s = fsa.arc_src[arc_i];
        let d = fsa.arc_dst[arc_i] as usize;
        let sym_idx = sym_to_idx[&fsa.arc_lbl[arc_i]];
        inv[d * k + sym_idx].push(s);
    }

    // Initial partition: final vs non-final
    let final_set: FxHashSet<u32> = fsa.stop.iter().copied().collect();
    let mut final_block: Vec<u32> = Vec::new();
    let mut nonfinal_block: Vec<u32> = Vec::new();
    for i in 0..n {
        if final_set.contains(&(i as u32)) {
            final_block.push(i as u32);
        } else {
            nonfinal_block.push(i as u32);
        }
    }

    // P[block_id] = list of states in that block
    let mut blocks: Vec<Vec<u32>> = Vec::new();
    // find[state] = block_id
    let mut find: Vec<u32> = vec![0; n];
    // in_worklist[block_id] = whether this block is in W
    let mut in_worklist: Vec<bool> = Vec::new();

    // Set up initial blocks
    let final_id = 0u32;
    blocks.push(final_block.clone());
    in_worklist.push(true);
    for &s in &final_block {
        find[s as usize] = final_id;
    }

    // Worklist
    let mut worklist: Vec<u32> = vec![final_id];

    if !nonfinal_block.is_empty() {
        let nonfinal_id = 1u32;
        for &s in &nonfinal_block {
            find[s as usize] = nonfinal_id;
        }
        blocks.push(nonfinal_block);
        in_worklist.push(true);
        worklist.push(nonfinal_id);
    }

    // Temporary storage reused across iterations
    let mut block_members: FxHashMap<u32, Vec<u32>> = FxHashMap::default();

    while let Some(a_id) = worklist.pop() {
        in_worklist[a_id as usize] = false;
        let a_block = std::mem::take(&mut blocks[a_id as usize]);

        for sym_idx in 0..k {
            // Group pre-images of a_block on this symbol by their current block
            block_members.clear();
            for &j in &a_block {
                for &i in &inv[j as usize * k + sym_idx] {
                    block_members
                        .entry(find[i as usize])
                        .or_default()
                        .push(i);
                }
            }

            for (&block_id, y_and_x) in &block_members {
                let y = &blocks[block_id as usize];
                // Deduplicate y_and_x (pre-images may have duplicates)
                // Use a bitset approach: mark which states in Y are in X
                let y_len = y.len();
                if y_len == 0 {
                    continue;
                }

                // Count distinct elements of y_and_x that are actually in this block
                // (states may have moved to other blocks between iterations,
                //  but find[] is kept current so this is already correct)
                let mut seen: FxHashSet<u32> = FxHashSet::default();
                for &s in y_and_x {
                    if find[s as usize] == block_id {
                        seen.insert(s);
                    }
                }
                let yx_count = seen.len();

                if yx_count == 0 || yx_count == y_len {
                    // No overlap or X >= Y — no split needed
                    continue;
                }

                // Split: YX stays in block_id, Y_X goes to a new block
                let mut yx = Vec::with_capacity(yx_count);
                let mut y_minus_x = Vec::with_capacity(y_len - yx_count);
                for &s in y {
                    if seen.contains(&s) {
                        yx.push(s);
                    } else {
                        y_minus_x.push(s);
                    }
                }

                // Replace block_id with YX (find[] stays correct for YX elements)
                blocks[block_id as usize] = yx;

                // Create new block for Y \ X
                let new_id = blocks.len() as u32;
                for &s in &y_minus_x {
                    find[s as usize] = new_id;
                }
                blocks.push(y_minus_x);
                in_worklist.push(false);

                // Add the smaller half to the worklist (Hopcroft's trick)
                let yx_len = blocks[block_id as usize].len();
                let ymx_len = blocks[new_id as usize].len();
                if in_worklist[block_id as usize] {
                    // block_id already in worklist — add new_id too
                    in_worklist[new_id as usize] = true;
                    worklist.push(new_id);
                } else if yx_len <= ymx_len {
                    in_worklist[block_id as usize] = true;
                    worklist.push(block_id);
                } else {
                    in_worklist[new_id as usize] = true;
                    worklist.push(new_id);
                }
            }
        }

        // Restore a_block
        blocks[a_id as usize] = a_block;
    }

    // Build class assignment from find[]
    // Renumber blocks contiguously (some may be empty after splits replaced them)
    let mut block_to_class: Vec<u32> = vec![u32::MAX; blocks.len()];
    let mut next_class = 0u32;
    let mut class = vec![0u32; n];
    for i in 0..n {
        let bid = find[i] as usize;
        if block_to_class[bid] == u32::MAX {
            block_to_class[bid] = next_class;
            next_class += 1;
        }
        class[i] = block_to_class[bid];
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
