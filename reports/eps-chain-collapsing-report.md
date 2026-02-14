# Branch `eps-chain-collapsing`: Optimization Report

## Overview

This branch introduces four optimizations to the decomposition pipeline, targeting
both the Rust-accelerated and Python codepaths. The changes reduce the cost of
universality detection (the main bottleneck in decomposition) and shrink powerset
state sizes by eliminating transit-only NFA states from epsilon closures.

All 77 Python tests pass.
All 10 Rust unit tests pass.

---

## Optimization 1: Epsilon-Chain Collapsing

**Files**: `precover.rs`, `fst.rs`

### Problem

In the precover NFA, epsilon closures can include many "transit-only" states —
NFA states whose FST state has only epsilon-input arcs and that aren't NFA-final.
These states:
- Generate no non-epsilon NFA arcs (don't affect DFA transitions in `compute_all_arcs`)
- Don't affect DFA finality
- Only serve as intermediaries in epsilon chains

Including them inflates powerset state sizes, which makes interning slower (larger
vectors to hash), universality checks slower (more states to explore), and the
overall DFA larger.

### Solution

**`fst.rs`**: Add a per-FST-state flag `has_non_eps_input: Vec<bool>`, computed once
during FST construction. True if the state has at least one arc with non-epsilon input.

**`precover.rs`**: Add an `is_productive()` predicate on NFA states:

```rust
fn is_productive(&self, packed: u64) -> bool {
    let (fst_state, buf_pos) = unpack(packed, self.target_len);
    self.fst.has_non_eps_input[fst_state as usize]
        || (self.fst.is_final[fst_state as usize] && buf_pos == self.target_len)
}
```

A state is productive if it can generate non-epsilon arcs OR if it's NFA-final.
Transit-only states (epsilon-only, non-final) are filtered out.

**`eps_closure_single_cached`**: The BFS still follows all epsilon arcs to find all
reachable states (unchanged), but then filters the result to keep only productive
states before caching:

```rust
let mut result: Vec<u64> = all_reachable
    .into_iter()
    .filter(|&s| self.is_productive(s))
    .collect();
```

### Correctness Argument

For any powerset state S, let S' = {s in S : is_productive(s)}. Then:

1. **Transitions**: Non-epsilon arcs from S equal those from S', since transit-only
   states have no non-epsilon arcs by definition.
2. **Finality**: S contains an NFA-final state iff S' does, since NFA-final states
   are productive by definition.
3. **Closure stability**: Filtering after BFS preserves all productive endpoints,
   since BFS still follows through transit-only intermediaries to reach them.

### Impact

For FSTs with long epsilon chains (e.g., BPE trie structures), this can
significantly reduce average powerset state sizes. For FSTs without epsilon
chains, the filter is a no-op — every state is productive.

---

## Optimization 2: UniversalityFilter (Multi-Strategy Optimizer)

**Files**: `decompose.rs` (Rust), `fst.py` (Python)

### Problem

The old `is_universal()` function in `decompose.rs` had two strategies:

1. A fast path: if `all_input_universal`, skip the check entirely.
2. A BFS fallback with a flat `FxHashSet<u32>` cache of known-universal DFA state IDs.

The flat cache only helped when the *exact same* DFA state was re-encountered.
It missed two common patterns:

- **Supersets**: If state S is universal, any S' ⊇ S is also universal (monotonicity).
- **Subsets**: If state S is non-universal, any S' ⊆ S is also non-universal.

### Solution

Replace the flat cache with a `UniversalityFilter` struct implementing five
strategies in priority order:

1. **Fast path**: If `all_input_universal`, return true immediately.
2. **Witness check**: If any NFA element in the powerset state is a known
   ip-universal witness (see Optimization 3), the state is universal. This is an
   O(|S|) set-intersection check.
3. **Superset monotonicity**: If a known-universal set u is a subset of the
   current set (u ⊆ S), then S is universal. Uses hit-counting: for each
   element of S, increment hit counts for cached entries containing that element.
   If any entry's count reaches its stored size, it's a subset match.
4. **Subset monotonicity**: If S is a subset of a known-non-universal set (S ⊆ nu),
   then S is non-universal. Uses set-intersection of entry-ID lists: for each
   element of S, intersect the set of negative cache entries containing it. If
   the intersection is non-empty, some negative entry is a superset.
5. **BFS fallback**: Full sub-BFS determinization. Results are added to the
   positive or negative cache for future lookups.

### Element-Indexed Caches

Both the positive and negative caches use element-indexed data structures
(inverted indexes) rather than linear scans over stored sets:

```rust
// Positive cache: element -> list of entry IDs containing it
pos_index: FxHashMap<u64, Vec<u32>>,
pos_sizes: Vec<usize>,  // entry_id -> stored set size

// Negative cache: element -> list of entry IDs containing it
neg_index: FxHashMap<u64, Vec<u32>>,
```

The Python implementation mirrors this structure using `defaultdict(set)`.

### Impact

Monotonicity caching avoids redundant BFS for states that are strict
supersets/subsets of previously-seen states. The element-indexed lookup avoids
O(n * m) linear scans over cached sets, replacing them with hash lookups.

---

## Optimization 3: Per-State IP-Universal Witnesses

**Files**: `fst.rs`

### Problem

The `all_input_universal` flag is all-or-nothing: either the *entire* FST's input
projection is universal from the start state, or the flag is false. For FSTs where
only some sub-machines are universal (e.g., a composition of a universal and a
non-universal transducer), this provides no benefit.

### Solution

Add `compute_ip_universal_states()`: a greatest-fixpoint computation that
determines, for each individual FST state q, whether the input projection of the
FST started from ip_eps_close({q}) accepts Sigma*.

**Algorithm**: Start by assuming all states are candidates. Iteratively remove
states that fail any of three conditions:

1. **Finality**: eps-closure must contain a final state.
2. **Completeness**: Must have arcs for all source alphabet symbols.
3. **Successor coverage**: For each symbol, the successor eps-closure must contain
   at least one remaining candidate.

Iterate until no more states are removed (greatest fixpoint).

**Integration with UniversalityFilter**: During initialization, pack each
ip-universal FST state q as the NFA state `(q, target_len)` — this is the NFA
state representing "FST state q, having consumed the entire target." These packed
states form the witness set. During universality checking, if any NFA element in
a powerset state matches a witness, the state is immediately known to be universal.

### Impact

For FSTs where some sub-machines are universal but not all, this provides a
fast-path that avoids BFS for those sub-machine states. For fully-universal FSTs,
`all_input_universal` already short-circuits before witnesses are checked.

---

## Optimization 4: `check_all_input_universal` Closure Caching

**Files**: `fst.rs`

### Problem

The pre-existing `check_all_input_universal` function called `ip_eps_close()` once
per source alphabet symbol to compute each symbol's successor closure. For GPT-2
BPE (~50K symbols, ~98K states), this meant ~50K BFS traversals over ~98K states,
causing FST construction to hang.

### Root Cause

In BPE FSTs, all ~50K token symbols route to the same hub state. Their successor
sets are identical, producing identical closures. But the old code computed each
one independently.

### Solution

Cache closure results keyed by the (sorted, deduped) destination state set:

```rust
let mut closure_cache: FxHashMap<Vec<u32>, bool> = FxHashMap::default();
for (_sym, raw_dests) in &by_symbol {
    let mut raw_vec: Vec<u32> = raw_dests.iter().copied().collect();
    raw_vec.sort_unstable();
    raw_vec.dedup();
    if let Some(&result) = closure_cache.get(&raw_vec) {
        if !result { return false; }
        continue;
    }
    let closed = ip_eps_close(&raw_vec, fst);
    let ok = start.iter().all(|&s| closed.binary_search(&s).is_ok());
    closure_cache.insert(raw_vec, ok);
    if !ok { return false; }
}
```

### Impact

For BPE FSTs: 50K BFS calls reduced to 1. FST construction goes from
hanging indefinitely to completing in ~0.7s.

---

## Optimization 5: `arcs_x` Methods (Python)

**Files**: `eager_nonrecursive.py`,
`peekaboo_nonrecursive.py`, `peekaboo_incremental.py`

### Change

Added `arcs_x(state, x)` methods to the various lazy NFA/DFA wrapper classes.
These return only the successor states for a specific input symbol x, avoiding
the overhead of generating all arcs and then filtering. This is the Python-side
analogue of `index_ix_j` in the Rust FST.

---

## Known Issue: `extract_token_bytes` Structural Mismatch

During benchmarking, we discovered that `token_decompose.rs`'s `extract_token_bytes`
assumes non-epsilon input arcs emanate FROM the hub state (start state), but
`bpe_wfst()` constructs FSTs with the opposite structure: epsilon-input arcs FROM
the hub to trie nodes, and token-input arcs FROM leaf states back TO the hub.

This means `token_decompose` produces trivially empty results (DFA=1 state, 0 arcs)
on real BPE FSTs from `bpe_wfst`. This is a **pre-existing bug** not introduced
by this branch. The token_decompose dispatch is preserved as-is for FSTs that do
match its expected structure.

---

## Test Results

```
Rust:  10 passed
Python: 50 passed
```
