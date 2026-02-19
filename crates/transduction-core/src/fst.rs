use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

pub const EPSILON: u32 = u32::MAX;

#[derive(Debug, Clone, Copy)]
pub struct FstArc {
    pub input: u32,
    pub output: u32,
    pub dest: u32,
}

/// Per-state output-group directory entry.
/// Within a state's arc range, arcs are sorted by (output, input).
/// Each OutputGroup records an output label and the *end* offset (absolute
/// index into `self.arcs`) for the contiguous sub-slice of arcs with that
/// output label.  The *start* is the previous group's end (or the state's
/// CSR start offset for the first group).
#[derive(Debug, Clone, Copy)]
pub struct OutputGroup {
    pub label: u32,
    pub end: u32,
}

/// Epsilon-input arc (pre-extracted side table for fast epsilon closure).
#[derive(Debug, Clone, Copy)]
pub struct EpsArc {
    pub output: u32,
    pub dest: u32,
}

/// FST stored in CSR format with output-group directory and epsilon side table.
///
/// Arcs are sorted by `(src, output, input)`.  Two lightweight overlays
/// replace the previous 4 hash-map indexes:
///
/// 1. **Output-group directory** — per-state sorted list of
///    `(output_label, end_offset)` for O(log k) lookup by output.
/// 2. **Epsilon-input side table** — per-state list of `(output, dest)` for
///    arcs with input=EPSILON, enabling O(1) epsilon closure without scanning
///    the main arc array.
pub struct Fst {
    pub num_states: u32,
    pub start_states: Vec<u32>,
    pub is_final: Vec<bool>,

    // CSR arc storage (sorted by (src, output, input))
    pub offsets: Vec<u32>, // length num_states+1
    pub arcs: Vec<FstArc>,

    // Output-group directory: per-state sorted list of (output_label, end_offset)
    pub group_offsets: Vec<u32>,       // length num_states+1, indexes into output_groups
    pub output_groups: Vec<OutputGroup>,

    // Epsilon-input side table: per-state list of (output, dest) for input=EPSILON arcs
    pub eps_offsets: Vec<u32>,         // length num_states+1, indexes into eps_arcs
    pub eps_arcs: Vec<EpsArc>,         // sorted by output within each state

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
        for ea in fst.eps_input_arcs(s) {
            if visited.insert(ea.dest) {
                worklist.push_back(ea.dest);
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
            for arc in fst.arcs_from(s) {
                if arc.input != EPSILON {
                    sym_map.entry(arc.input).or_default().insert(arc.dest);
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

        // Sort arcs by (src, output, input) for CSR + output-group directory
        let mut indices: Vec<usize> = (0..num_arcs).collect();
        indices.sort_unstable_by_key(|&i| (arc_src[i], arc_out[i], arc_in[i]));

        // Build CSR offsets + arcs
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

        // Build output-group directory + epsilon side table in a single pass
        let mut group_offsets = vec![0u32; n + 1];
        let mut output_groups: Vec<OutputGroup> = Vec::new();
        let mut eps_offsets = vec![0u32; n + 1];
        let mut eps_arcs: Vec<EpsArc> = Vec::new();
        let mut has_non_eps_input = vec![false; n];

        for state in 0..n {
            let lo = offsets[state] as usize;
            let hi = offsets[state + 1] as usize;

            group_offsets[state] = output_groups.len() as u32;
            eps_offsets[state] = eps_arcs.len() as u32;

            if lo < hi {
                // Build output groups - arcs are sorted by (output, input)
                let mut prev_output = arcs[lo].output;
                for pos in lo..hi {
                    let arc = &arcs[pos];
                    if arc.output != prev_output {
                        output_groups.push(OutputGroup {
                            label: prev_output,
                            end: pos as u32,
                        });
                        prev_output = arc.output;
                    }
                    // Collect epsilon-input arcs
                    if arc.input == EPSILON {
                        eps_arcs.push(EpsArc {
                            output: arc.output,
                            dest: arc.dest,
                        });
                    } else {
                        has_non_eps_input[state] = true;
                    }
                }
                // Close the last group
                output_groups.push(OutputGroup {
                    label: prev_output,
                    end: hi as u32,
                });
            }
        }
        group_offsets[n] = output_groups.len() as u32;
        eps_offsets[n] = eps_arcs.len() as u32;

        Fst {
            num_states,
            start_states,
            is_final,
            offsets,
            arcs,
            group_offsets,
            output_groups,
            eps_offsets,
            eps_arcs,
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

    /// Return the sub-slice of arcs from `state` with output label `output`.
    /// Uses binary search on the output-group directory.
    #[inline]
    pub fn arcs_by_output(&self, state: u32, output: u32) -> &[FstArc] {
        let glo = self.group_offsets[state as usize] as usize;
        let ghi = self.group_offsets[state as usize + 1] as usize;
        let groups = &self.output_groups[glo..ghi];

        // Binary search for the output label
        match groups.binary_search_by_key(&output, |g| g.label) {
            Ok(idx) => {
                let arc_start = if idx == 0 {
                    self.offsets[state as usize] as usize
                } else {
                    groups[idx - 1].end as usize
                };
                let arc_end = groups[idx].end as usize;
                &self.arcs[arc_start..arc_end]
            }
            Err(_) => &[],
        }
    }

    /// All epsilon-input arcs from a state.
    #[inline]
    pub fn eps_input_arcs(&self, state: u32) -> &[EpsArc] {
        let lo = self.eps_offsets[state as usize] as usize;
        let hi = self.eps_offsets[state as usize + 1] as usize;
        &self.eps_arcs[lo..hi]
    }

    /// Epsilon-input arcs from a state with a specific output label.
    /// Uses binary search within the eps_arcs range (sorted by output).
    #[inline]
    pub fn eps_input_arcs_by_output(&self, state: u32, output: u32) -> &[EpsArc] {
        let eps = self.eps_input_arcs(state);
        if eps.is_empty() {
            return &[];
        }
        // eps_arcs are sorted by output within each state.
        // Find the range [lo, hi) where ea.output == output.
        let lo = eps.partition_point(|ea| ea.output < output);
        let hi = eps[lo..].partition_point(|ea| ea.output <= output) + lo;
        &eps[lo..hi]
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

    #[test]
    fn test_arcs_by_output() {
        // State 0: arcs with output=10, output=20, output=EPSILON
        let fst = Fst::new(
            2,
            vec![0],
            &[1],
            &[0, 0, 0, 0],       // all from state 0
            &[1, 2, 3, 4],       // input labels
            &[10, 20, 10, EPSILON], // output labels
            &[1, 1, 1, 1],       // all to state 1
            vec![1, 2, 3, 4],
        );

        // output=10 should have 2 arcs
        let arcs10 = fst.arcs_by_output(0, 10);
        assert_eq!(arcs10.len(), 2);

        // output=20 should have 1 arc
        let arcs20 = fst.arcs_by_output(0, 20);
        assert_eq!(arcs20.len(), 1);
        assert_eq!(arcs20[0].input, 2);

        // output=EPSILON should have 1 arc
        let arcs_eps = fst.arcs_by_output(0, EPSILON);
        assert_eq!(arcs_eps.len(), 1);
        assert_eq!(arcs_eps[0].input, 4);

        // output=99 should be empty
        assert_eq!(fst.arcs_by_output(0, 99).len(), 0);

        // state 1 has no arcs
        assert_eq!(fst.arcs_by_output(1, 10).len(), 0);
    }

    #[test]
    fn test_eps_input_arcs() {
        // State 0: input=EPSILON with output=10 and output=20
        // State 0: input=5 with output=10
        let fst = Fst::new(
            2,
            vec![0],
            &[1],
            &[0, 0, 0],
            &[EPSILON, EPSILON, 5],    // input labels
            &[10, 20, 10],             // output labels
            &[1, 1, 1],
            vec![5],
        );

        let eps = fst.eps_input_arcs(0);
        assert_eq!(eps.len(), 2);

        let eps10 = fst.eps_input_arcs_by_output(0, 10);
        assert_eq!(eps10.len(), 1);
        assert_eq!(eps10[0].dest, 1);

        let eps20 = fst.eps_input_arcs_by_output(0, 20);
        assert_eq!(eps20.len(), 1);

        let eps99 = fst.eps_input_arcs_by_output(0, 99);
        assert_eq!(eps99.len(), 0);

        // state 1 has no eps arcs
        assert_eq!(fst.eps_input_arcs(1).len(), 0);
    }
}
