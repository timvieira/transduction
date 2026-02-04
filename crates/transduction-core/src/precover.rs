use crate::fst::{Fst, EPSILON};
use rustc_hash::FxHashMap;
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
    /// Cache: eps closure per individual NFA state (mirrors Python's _closure_cache)
    eps_cache: RefCell<FxHashMap<u64, Rc<Vec<u64>>>>,
    eps_cache_hits: RefCell<u64>,
    eps_cache_misses: RefCell<u64>,
}

impl<'a> PrecoverNFA<'a> {
    pub fn new(fst: &'a Fst, target: &'a [u32]) -> Self {
        PrecoverNFA {
            fst,
            target,
            target_len: target.len() as u32,
            eps_cache: RefCell::new(FxHashMap::default()),
            eps_cache_hits: RefCell::new(0),
            eps_cache_misses: RefCell::new(0),
        }
    }

    #[inline]
    pub fn is_final(&self, packed: u64) -> bool {
        let (fst_state, buf_pos) = unpack(packed, self.target_len);
        self.fst.is_final[fst_state as usize] && buf_pos == self.target_len
    }

    /// All arcs from a state: (input_symbol, destination).
    pub fn arcs(&self, packed: u64) -> Vec<(u32, u64)> {
        let (i, n) = unpack(packed, self.target_len);
        let mut result = Vec::new();

        if n == self.target_len {
            for &(x, j) in &self.fst.index_i_xj[i as usize] {
                result.push((x, pack(j, self.target_len, self.target_len)));
            }
        } else {
            if let Some(eps_arcs) = self.fst.index_iy_xj.get(&(i, EPSILON)) {
                for &(x, j) in eps_arcs {
                    result.push((x, pack(j, n, self.target_len)));
                }
            }
            if let Some(match_arcs) = self.fst.index_iy_xj.get(&(i, self.target[n as usize])) {
                for &(x, j) in match_arcs {
                    result.push((x, pack(j, n + 1, self.target_len)));
                }
            }
        }

        result
    }

    /// Successors for a given input symbol x.
    #[inline]
    pub fn arcs_x(&self, packed: u64, x: u32) -> Vec<u64> {
        let (i, n) = unpack(packed, self.target_len);
        let mut result = Vec::new();

        if n == self.target_len {
            if let Some(dests) = self.fst.index_ix_j.get(&(i, x)) {
                for &j in dests {
                    result.push(pack(j, self.target_len, self.target_len));
                }
            }
        } else {
            if let Some(dests) = self.fst.index_ixy_j.get(&(i, x, EPSILON)) {
                for &j in dests {
                    result.push(pack(j, n, self.target_len));
                }
            }
            if let Some(dests) = self.fst.index_ixy_j.get(&(i, x, self.target[n as usize])) {
                for &j in dests {
                    result.push(pack(j, n + 1, self.target_len));
                }
            }
        }

        result
    }

    pub fn start_states(&self) -> Vec<u64> {
        self.fst
            .start_states
            .iter()
            .map(|&s| pack(s, 0, self.target_len))
            .collect()
    }

    /// Epsilon closure of a single NFA state, cached.
    /// This mirrors Python's `EpsilonRemove._closure_cache`.
    /// Returns Rc to avoid cloning on cache hits.
    fn eps_closure_single_cached(&self, state: u64) -> Rc<Vec<u64>> {
        // Check cache first
        if let Some(cached) = self.eps_cache.borrow().get(&state) {
            *self.eps_cache_hits.borrow_mut() += 1;
            return Rc::clone(cached);
        }
        *self.eps_cache_misses.borrow_mut() += 1;

        // Compute using a simple Vec as visited set (states are sorted at the end anyway)
        let mut result = Vec::new();
        let mut worklist: VecDeque<u64> = VecDeque::new();

        result.push(state);
        worklist.push_back(state);

        while let Some(s) = worklist.pop_front() {
            for dest in self.arcs_x(s, EPSILON) {
                if !result.contains(&dest) {
                    result.push(dest);
                    worklist.push_back(dest);
                }
            }
        }

        result.sort_unstable();

        // Wrap in Rc, cache, and return
        let rc = Rc::new(result);
        self.eps_cache.borrow_mut().insert(state, Rc::clone(&rc));
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

    pub fn compute_all_arcs(&self, states: &[u64]) -> Vec<(u32, Vec<u64>)> {
        let mut by_symbol: FxHashMap<u32, Vec<u64>> = FxHashMap::default();

        for &packed in states {
            for (x, dest) in self.arcs(packed) {
                if x != EPSILON {
                    // Use cached eps closure for each individual destination
                    let closure = self.eps_closure_single_cached(dest);
                    let bucket = by_symbol.entry(x).or_default();
                    bucket.extend_from_slice(&closure);
                }
            }
        }

        // Sort+dedup each bucket to get canonical form for interning
        let mut result: Vec<(u32, Vec<u64>)> = Vec::with_capacity(by_symbol.len());
        for (sym, mut v) in by_symbol {
            v.sort_unstable();
            v.dedup();
            result.push((sym, v));
        }

        result
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
