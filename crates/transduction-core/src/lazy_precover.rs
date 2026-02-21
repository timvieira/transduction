use crate::fst::Fst;
use crate::powerset::PowersetArena;
use crate::precover::PrecoverNFA;
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// Lazy DFA over the precover NFA.
///
/// Wraps PrecoverNFA + PowersetArena and exposes on-demand state expansion.
/// States are interned as u32 IDs via the powerset arena. Arcs are computed
/// lazily on first access and cached for subsequent queries.
///
/// The epsilon-closure cache from PrecoverNFA is preserved across calls to
/// avoid redundant work.
pub struct LazyPrecoverDFA {
    target: Vec<u32>,
    arena: PowersetArena,
    /// Per-state arc cache: arcs_cache[sid] = Some(arcs) once expanded.
    arcs_cache: Vec<Option<Vec<(u32, u32)>>>,
    start_id: u32,
    /// Reusable buffer for PrecoverNFA::compute_all_arcs_into.
    arcs_buf: FxHashMap<u32, Vec<u64>>,
    /// Epsilon-closure cache, preserved across temporary PrecoverNFA instances.
    eps_cache: FxHashMap<u64, (Rc<Vec<u64>>, u32)>,
    stride: u64,
}

impl LazyPrecoverDFA {
    /// Create a new lazy precover DFA for the given FST and target.
    pub fn new(fst: &Fst, target: Vec<u32>) -> Self {
        let stride = target.len() as u64 + 1;
        let nfa = PrecoverNFA::new(fst, &target);

        let mut arena = PowersetArena::new();

        // Compute epsilon-closed initial powerset state
        let raw_starts = nfa.start_states();
        let mut init_closed = Vec::new();
        nfa.eps_closure_set(&raw_starts, &mut init_closed);

        let any_final = init_closed.iter().any(|&s| nfa.is_final(s));
        let start_id = arena.intern(init_closed, any_final);

        let eps_cache = nfa.take_eps_cache();

        let arcs_cache = vec![None; arena.len()];

        LazyPrecoverDFA {
            target,
            arena,
            arcs_cache,
            start_id,
            arcs_buf: FxHashMap::default(),
            eps_cache,
            stride,
        }
    }

    /// Start state ID.
    pub fn start_id(&self) -> u32 {
        self.start_id
    }

    /// Whether a DFA state is final.
    pub fn is_final(&self, sid: u32) -> bool {
        self.arena.is_final[sid as usize]
    }

    /// Number of NFA states in the powerset for a DFA state.
    pub fn powerset_size(&self, sid: u32) -> usize {
        self.arena.sets[sid as usize].len()
    }

    /// Total number of interned DFA states so far.
    pub fn num_states(&self) -> usize {
        self.arena.len()
    }

    /// Lazily compute and return arcs from a DFA state.
    /// Returns a slice of (input_label, dest_sid) pairs.
    pub fn arcs(&mut self, fst: &Fst, sid: u32) -> &[(u32, u32)] {
        self.ensure_arcs_for(fst, sid);
        self.arcs_cache[sid as usize].as_ref().unwrap()
    }

    /// Internal: ensure arcs for sid are computed and cached.
    fn ensure_arcs_for(&mut self, fst: &Fst, sid: u32) {
        if (sid as usize) < self.arcs_cache.len()
            && self.arcs_cache[sid as usize].is_some()
        {
            return;
        }

        // Create a temporary PrecoverNFA, transferring our eps_cache in
        let eps_cache = std::mem::take(&mut self.eps_cache);
        let nfa = PrecoverNFA::with_stride_and_cache(
            fst, &self.target, self.stride, eps_cache,
        );

        let states = &self.arena.sets[sid as usize];
        let all_arcs = nfa.compute_all_arcs_into(states, &mut self.arcs_buf);

        let mut result: Vec<(u32, u32)> = Vec::with_capacity(all_arcs.len());
        for (sym, successor) in all_arcs {
            let any_final = successor.iter().any(|&s| nfa.is_final(s));
            let dest_id = self.arena.intern(successor, any_final);
            result.push((sym, dest_id));
        }

        // Take eps_cache back
        self.eps_cache = nfa.take_eps_cache();

        // Grow arcs_cache if arena grew
        while self.arcs_cache.len() < self.arena.len() {
            self.arcs_cache.push(None);
        }

        self.arcs_cache[sid as usize] = Some(result);
    }

    /// Traverse a full source path from the start state.
    /// Returns the reached DFA state, or None if any arc is missing.
    pub fn run(&mut self, fst: &Fst, path: &[u32]) -> Option<u32> {
        let mut state = self.start_id;
        for &sym in path {
            self.ensure_arcs_for(fst, state);
            let arcs = self.arcs_cache[state as usize].as_ref().unwrap();
            match arcs.iter().find(|&&(lbl, _)| lbl == sym) {
                Some(&(_, dest)) => state = dest,
                None => return None,
            }
        }
        Some(state)
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
    fn test_lazy_precover_empty_target() {
        let fst = make_replace_fst(&[(1, 10), (2, 11)]);
        let mut dfa = LazyPrecoverDFA::new(&fst, vec![]);

        // Start state should be final (empty target = preimage)
        assert!(dfa.is_final(dfa.start_id()));

        // Should have arcs for both input symbols
        let arcs = dfa.arcs(&fst, dfa.start_id()).to_vec();
        assert_eq!(arcs.len(), 2);
    }

    #[test]
    fn test_lazy_precover_single_target() {
        let fst = make_replace_fst(&[(1, 10), (2, 11)]);
        let mut dfa = LazyPrecoverDFA::new(&fst, vec![10]);

        // Start state should NOT be final
        assert!(!dfa.is_final(dfa.start_id()));

        // Walk: consuming input symbol 1 (which maps to output 10)
        // should reach a final state
        let arcs = dfa.arcs(&fst, dfa.start_id()).to_vec();
        assert!(!arcs.is_empty());
    }

    #[test]
    fn test_lazy_precover_run() {
        let fst = make_replace_fst(&[(1, 10), (2, 11)]);
        let mut dfa = LazyPrecoverDFA::new(&fst, vec![10]);

        // Run with input 1 should reach some state
        let result = dfa.run(&fst, &[1]);
        assert!(result.is_some());

        // Run with non-existent input should fail
        let result = dfa.run(&fst, &[99]);
        assert!(result.is_none());
    }
}
