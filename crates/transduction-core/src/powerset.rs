use rustc_hash::FxHashMap;

/// Interns sorted `Vec<u64>` powerset states as `u32` IDs.
/// After interning, all DFA operations use cheap u32 state IDs.
pub struct PowersetArena {
    map: FxHashMap<Vec<u64>, u32>,
    /// Fast path for single-element sets (99% of cases in BPE).
    /// Avoids hashing a Vec when the set contains exactly one element.
    single_map: FxHashMap<u64, u32>,
    pub sets: Vec<Vec<u64>>,
    pub is_final: Vec<bool>,
}

impl PowersetArena {
    pub fn new() -> Self {
        PowersetArena {
            map: FxHashMap::default(),
            single_map: FxHashMap::default(),
            sets: Vec::new(),
            is_final: Vec::new(),
        }
    }

    /// Intern a sorted set of NFA states. Returns the u32 ID.
    /// `any_final` indicates whether any NFA state in the set is final.
    pub fn intern(&mut self, sorted_set: Vec<u64>, any_final: bool) -> u32 {
        if sorted_set.len() == 1 {
            // Fast path: single-element set — hash a u64 instead of a Vec
            let key = sorted_set[0];
            if let Some(&id) = self.single_map.get(&key) {
                self.is_final[id as usize] = any_final;
                return id;
            }
            let id = self.sets.len() as u32;
            self.sets.push(sorted_set);
            self.is_final.push(any_final);
            self.single_map.insert(key, id);
            return id;
        }

        if let Some(&id) = self.map.get(&sorted_set) {
            self.is_final[id as usize] = any_final;
            return id;
        }
        let id = self.sets.len() as u32;
        self.sets.push(sorted_set.clone());
        self.is_final.push(any_final);
        self.map.insert(sorted_set, id);
        id
    }

    /// Look up an existing set. Returns None if not interned.
    pub fn lookup(&self, sorted_set: &[u64]) -> Option<u32> {
        if sorted_set.len() == 1 {
            return self.single_map.get(&sorted_set[0]).copied();
        }
        self.map.get(sorted_set).copied()
    }

    pub fn len(&self) -> usize {
        self.sets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern() {
        let mut arena = PowersetArena::new();

        let id0 = arena.intern(vec![1, 2, 3], false);
        let id1 = arena.intern(vec![4, 5], true);
        let id2 = arena.intern(vec![1, 2, 3], false);

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 0); // same set, same ID

        assert!(!arena.is_final[id0 as usize]);
        assert!(arena.is_final[id1 as usize]);
    }
}
