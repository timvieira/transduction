use crate::fst::{Fst, EPSILON};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

/// Pack (fst_state, buffer_pos) into a single u64.
#[inline]
pub fn pack(fst_state: u32, buf_pos: u32, target_len: u32) -> u64 {
    fst_state as u64 * (target_len as u64 + 1) + buf_pos as u64
}

/// Unpack a u64 back to (fst_state, buffer_pos).
#[inline]
pub fn unpack(packed: u64, target_len: u32) -> (u32, u32) {
    let tl = target_len as u64 + 1;
    let fst_state = (packed / tl) as u32;
    let buf_pos = (packed % tl) as u32;
    (fst_state, buf_pos)
}

/// Precover NFA logic. Mirrors the Python `LazyPrecoverNFA` class.
///
/// States are `(fst_state, buffer_pos)` packed as u64.
pub struct PrecoverNFA<'a> {
    pub fst: &'a Fst,
    pub target: &'a [u32],
    pub target_len: u32,
    /// Stride used for packing: fst_state * stride + buf_pos.
    /// For non-incremental use, stride = target_len + 1.
    /// For incremental use, stride is fixed across calls (>= max target_len + 1).
    pub stride: u64,
    /// Cache: eps closure per individual NFA state (mirrors Python's _closure_cache).
    /// Value is (closure, max_buf_pos) where max_buf_pos is the maximum buf_pos
    /// in the closure, used for fast eviction in incremental mode.
    eps_cache: RefCell<FxHashMap<u64, (Rc<Vec<u64>>, u32)>>,
    eps_cache_hits: RefCell<u64>,
    eps_cache_misses: RefCell<u64>,
}

impl<'a> PrecoverNFA<'a> {
    pub fn new(fst: &'a Fst, target: &'a [u32]) -> Self {
        let target_len = target.len() as u32;
        PrecoverNFA {
            fst,
            target,
            target_len,
            stride: target_len as u64 + 1,
            eps_cache: RefCell::new(FxHashMap::default()),
            eps_cache_hits: RefCell::new(0),
            eps_cache_misses: RefCell::new(0),
        }
    }

    /// Create a PrecoverNFA with a fixed stride (for incremental use).
    /// The stride must be >= target_len + 1.
    pub fn with_stride(fst: &'a Fst, target: &'a [u32], stride: u64) -> Self {
        let target_len = target.len() as u32;
        assert!(stride >= target_len as u64 + 1,
            "stride {} too small for target_len {}", stride, target_len);
        PrecoverNFA {
            fst,
            target,
            target_len,
            stride,
            eps_cache: RefCell::new(FxHashMap::default()),
            eps_cache_hits: RefCell::new(0),
            eps_cache_misses: RefCell::new(0),
        }
    }

    /// Create a PrecoverNFA with a fixed stride and a pre-populated eps_cache.
    pub fn with_stride_and_cache(
        fst: &'a Fst,
        target: &'a [u32],
        stride: u64,
        eps_cache: FxHashMap<u64, (Rc<Vec<u64>>, u32)>,
    ) -> Self {
        let target_len = target.len() as u32;
        assert!(stride >= target_len as u64 + 1,
            "stride {} too small for target_len {}", stride, target_len);
        PrecoverNFA {
            fst,
            target,
            target_len,
            stride,
            eps_cache: RefCell::new(eps_cache),
            eps_cache_hits: RefCell::new(0),
            eps_cache_misses: RefCell::new(0),
        }
    }

    /// Pack (fst_state, buf_pos) using this NFA's stride.
    #[inline]
    pub fn pack_state(&self, fst_state: u32, buf_pos: u32) -> u64 {
        fst_state as u64 * self.stride + buf_pos as u64
    }

    /// Unpack a u64 back to (fst_state, buf_pos) using this NFA's stride.
    #[inline]
    pub fn unpack_state(&self, packed: u64) -> (u32, u32) {
        let fst_state = (packed / self.stride) as u32;
        let buf_pos = (packed % self.stride) as u32;
        (fst_state, buf_pos)
    }

    /// Extract the eps_cache for reuse in a subsequent PrecoverNFA.
    pub fn take_eps_cache(&self) -> FxHashMap<u64, (Rc<Vec<u64>>, u32)> {
        std::mem::take(&mut *self.eps_cache.borrow_mut())
    }

    #[inline]
    pub fn is_final(&self, packed: u64) -> bool {
        let (fst_state, buf_pos) = self.unpack_state(packed);
        self.fst.is_final[fst_state as usize] && buf_pos == self.target_len
    }

    /// All arcs from a state: (input_symbol, destination).
    pub fn arcs(&self, packed: u64) -> Vec<(u32, u64)> {
        let (i, n) = self.unpack_state(packed);
        let mut result = Vec::new();

        if n == self.target_len {
            // Boundary phase: all arcs from FST state i
            for arc in self.fst.arcs_from(i) {
                result.push((arc.input, self.pack_state(arc.dest, self.target_len)));
            }
        } else {
            // Growing phase: arcs with output=EPSILON (buffer unchanged)
            for arc in self.fst.arcs_by_output(i, EPSILON) {
                result.push((arc.input, self.pack_state(arc.dest, n)));
            }
            // Growing phase: arcs with output=target[n] (buffer advances)
            for arc in self.fst.arcs_by_output(i, self.target[n as usize]) {
                result.push((arc.input, self.pack_state(arc.dest, n + 1)));
            }
        }

        result
    }

    /// Successors for a given input symbol x.
    /// Note: this is only ever called with x=EPSILON (by eps_closure_single_cached).
    #[inline]
    pub fn arcs_x(&self, packed: u64, x: u32) -> Vec<u64> {
        debug_assert!(x == EPSILON, "arcs_x called with non-EPSILON x={}", x);
        let (i, n) = self.unpack_state(packed);
        let mut result = Vec::new();

        if n == self.target_len {
            // Boundary phase: epsilon-input arcs from FST state i
            for ea in self.fst.eps_input_arcs(i) {
                result.push(self.pack_state(ea.dest, self.target_len));
            }
        } else {
            // Growing phase: epsilon-input arcs with output=EPSILON
            for ea in self.fst.eps_input_arcs_by_output(i, EPSILON) {
                result.push(self.pack_state(ea.dest, n));
            }
            // Growing phase: epsilon-input arcs with output=target[n]
            for ea in self.fst.eps_input_arcs_by_output(i, self.target[n as usize]) {
                result.push(self.pack_state(ea.dest, n + 1));
            }
        }

        result
    }

    pub fn start_states(&self) -> Vec<u64> {
        self.fst
            .start_states
            .iter()
            .map(|&s| self.pack_state(s, 0))
            .collect()
    }

    /// Is an NFA state "productive"?
    ///
    /// A state is productive if its FST state has at least one non-epsilon input
    /// arc, or if the state is NFA-final. Transit-only states (epsilon-input-only,
    /// non-final) are not productive — they serve only as intermediaries in
    /// epsilon chains and don't contribute to DFA transitions or finality.
    #[inline]
    fn is_productive(&self, packed: u64) -> bool {
        let (fst_state, buf_pos) = self.unpack_state(packed);
        self.fst.has_non_eps_input[fst_state as usize]
            || (self.fst.is_final[fst_state as usize] && buf_pos == self.target_len)
    }

    /// Epsilon closure of a single NFA state, cached.
    /// This mirrors Python's `EpsilonRemove._closure_cache`.
    /// Returns Rc to avoid cloning on cache hits.
    ///
    /// The BFS follows all epsilon arcs to find reachable states, then filters
    /// to keep only productive states. This collapses transit-only epsilon chains,
    /// dramatically reducing powerset state sizes for BPE-like FSTs.
    fn eps_closure_single_cached(&self, state: u64) -> Rc<Vec<u64>> {
        // Check cache first
        if let Some((cached, _max_bp)) = self.eps_cache.borrow().get(&state) {
            *self.eps_cache_hits.borrow_mut() += 1;
            return Rc::clone(cached);
        }
        *self.eps_cache_misses.borrow_mut() += 1;

        // BFS: explore all reachable states via epsilon arcs
        let mut visited = FxHashSet::default();
        let mut worklist: VecDeque<u64> = VecDeque::new();

        visited.insert(state);
        worklist.push_back(state);

        while let Some(s) = worklist.pop_front() {
            for dest in self.arcs_x(s, EPSILON) {
                if visited.insert(dest) {
                    worklist.push_back(dest);
                }
            }
        }

        // Filter to productive states only
        let mut result: Vec<u64> = visited
            .into_iter()
            .filter(|&s| self.is_productive(s))
            .collect();
        result.sort_unstable();

        // Compute max_buf_pos for fast eviction
        let max_buf_pos = result.iter()
            .map(|&s| (s % self.stride) as u32)
            .max()
            .unwrap_or(0);

        // Wrap in Rc, cache, and return
        let rc = Rc::new(result);
        self.eps_cache.borrow_mut().insert(state, (Rc::clone(&rc), max_buf_pos));
        rc
    }

    /// Epsilon closure of a set of states (used for initial state construction).
    pub fn eps_closure_set(&self, states: &[u64], buf: &mut Vec<u64>) {
        buf.clear();
        for &s in states {
            let closure = self.eps_closure_single_cached(s);
            for &cs in closure.iter() {
                buf.push(cs);
            }
        }
        buf.sort_unstable();
        buf.dedup();
    }

    /// Batch-compute all non-epsilon arcs from an epsilon-closed powerset state.
    ///
    /// For each NFA state in the set, iterates its arcs, epsilon-closes each
    /// destination using the cache, and groups results by input symbol.
    ///
    /// Complexity: O(total NFA arcs from the set), NOT O(|alphabet|).
    /// Return (hits, misses) for the eps closure cache.
    pub fn eps_cache_stats(&self) -> (u64, u64) {
        (*self.eps_cache_hits.borrow(), *self.eps_cache_misses.borrow())
    }

    /// Buffer-reuse variant of `compute_all_arcs`. The caller allocates
    /// `by_symbol` once and passes it in; inner Vecs are cleared but their
    /// allocations are kept across calls.
    pub fn compute_all_arcs_into(
        &self,
        states: &[u64],
        by_symbol: &mut FxHashMap<u32, Vec<u64>>,
    ) -> Vec<(u32, Vec<u64>)> {
        // Clear values but keep allocated buckets
        for v in by_symbol.values_mut() {
            v.clear();
        }

        for &packed in states {
            for (x, dest) in self.arcs(packed) {
                if x != EPSILON {
                    let closure = self.eps_closure_single_cached(dest);
                    let bucket = by_symbol.entry(x).or_default();
                    bucket.extend_from_slice(&closure);
                }
            }
        }

        // Collect non-empty entries into result, sort+dedup each bucket.
        // Use replace instead of take so the buffer vecs keep their allocated
        // capacity for reuse on subsequent calls.
        let mut result: Vec<(u32, Vec<u64>)> = Vec::with_capacity(by_symbol.len());
        for (&sym, v) in by_symbol.iter_mut() {
            if !v.is_empty() {
                v.sort_unstable();
                v.dedup();
                let cap = v.capacity();
                result.push((sym, std::mem::replace(v, Vec::with_capacity(cap))));
            }
        }

        result
    }

    pub fn compute_all_arcs(&self, states: &[u64]) -> Vec<(u32, Vec<u64>)> {
        let mut buf = FxHashMap::default();
        self.compute_all_arcs_into(states, &mut buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack() {
        let target_len = 5;
        for s in 0..10 {
            for p in 0..=target_len {
                let packed = pack(s, p, target_len);
                let (s2, p2) = unpack(packed, target_len);
                assert_eq!(s, s2);
                assert_eq!(p, p2);
            }
        }
    }
}
