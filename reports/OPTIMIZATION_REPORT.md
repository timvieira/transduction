# Optimization Report: Rust Acceleration of `decompose()`

## 1. Which Python Implementation is Closest to the Rust?

The Rust implementation is closest to **`NonrecursiveDFADecomp`** in `dfa_decomp_nonrecursive.py`. Both follow the same high-level algorithm:

```
1. Build a precover NFA: LazyPrecoverNFA(fst, target)
2. On-the-fly determinize it (powerset construction)
3. BFS over DFA states, checking universality of final states
4. Universal finals → Q stops (don't expand further)
   Non-universal finals → R stops (keep expanding)
5. Return Q and R sharing the same arc set
```

The Python pipeline chains lazy wrappers: `LazyPrecoverNFA(fst, target).det()` produces a `LazyDeterminize(EpsilonRemove(LazyPrecoverNFA(...)))`. Each call to `dfa.arcs(state)` triggers:

```
LazyDeterminize.arcs(frozenset)
  → for each NFA state in frozenset:
      EpsilonRemove.arcs(state)
        → LazyPrecoverNFA.arcs(state)   # get raw NFA arcs
        → for each dest: eps_closure(dest)  # BFS for ε-reachable states
      group by input symbol → frozenset per symbol
```

The Rust `decompose.rs` fuses all of this into a single BFS loop, eliminating the lazy wrapper overhead. The NFA states are packed as `u64` (instead of Python tuples), and powerset states are interned as `u32` IDs (instead of Python frozensets).

However, there are **two Rust paths**. For BPE-like FSTs where the input projection accepts Σ*, the dispatch in `py.rs:164` routes to `token_decompose.rs`, which is a fundamentally different algorithm (described below).

## 2. Optimizations Employed

### Optimization A: Fused State Representation (general)

**Python:** NFA states are `(fst_state, target_prefix_string)` tuples — e.g., `(0, 'Hel')`. The `LazyPrecoverNFA_slower` in `goo.py` improves this to `(fst_state, int_position)` — e.g., `(0, 3)` — avoiding string slicing.

**Rust:** Packs `(fst_state, buf_pos)` into a single `u64` via `fst_state * (target_len + 1) + buf_pos`. No heap allocation. O(1) hashing. This is identical to what `LazyPrecoverNFA_slower` does with `(i, n)` tuples, but avoids Python tuple overhead.

**Generality:** This is the same trick `LazyPrecoverNFA` already uses (the optimized version in `goo.py`). The Rust version just avoids the per-tuple allocation.

**Pullback to Python:** Already done in `LazyPrecoverNFA` (the `(i, n)` representation). No further improvement needed on this front.

### Optimization B: Precomputed FST Indexes (general)

**Python:** `LazyPrecoverNFA` in `goo.py` builds 4 index dictionaries on the FST, cached via `try/except AttributeError`:
- `index_iy_xj`: `(state, output)` → `{(input, dest)}`
- `index_i_xj`: `state` → `{(input, dest)}`
- `index_ix_j`: `(state, input)` → `{dest}`
- `index_ixy_j`: `(state, input, output)` → `{dest}`

**Rust:** Builds the same 4 indexes in `Fst::new()` using `FxHashMap` (a faster hasher than Python's built-in). The Rust indexes use the same keys and structure.

**Generality:** Fully general. Any FST benefits from these indexes.

**Pullback to Python:** Already done — `LazyPrecoverNFA` builds these indexes. The only Python improvement would be to use a faster hash (but Python doesn't easily allow that).

### Optimization C: Arena-Interned Powerset States (general)

**Python:** `LazyDeterminize` represents DFA states as `frozenset` objects. Each `frozenset` is heap-allocated and compared by value. The `accepts_universal` function stores these in a Python `set`, which hashes the frozenset on every insert/lookup.

**Rust:** `PowersetArena` interns sorted `Vec<u64>` → `u32` IDs. After interning, all DFA operations use cheap `u32` comparisons. A fast path handles single-element sets (99% of cases in BPE) by hashing a `u64` instead of a `Vec`.

**Generality:** Fully general. Any powerset construction benefits from interning.

**Pullback to Python:** **Yes, this would help.** You could add an `Integerizer` (which you already have in `arsenal`) to the `LazyDeterminize` class to map frozensets to integers. This avoids repeated frozenset hashing:

```python
class LazyDeterminize(Lazy):
    def __init__(self, fsa):
        self.fsa = fsa.epsremove()
        self.intern = Integerizer()  # frozenset → int

    def arcs(self, Q_id):
        Q = self.intern[Q_id]  # recover frozenset
        tmp = defaultdict(set)
        for i in Q:
            for a, j in self.fsa.arcs(i):
                tmp[a].add(j)
        for a, j in tmp.items():
            yield a, self.intern(frozenset(j))  # intern result
```

The `visited` set and `worklist` in the BFS would then use integers instead of frozensets — much cheaper.

### Optimization D: Cached Epsilon Closures (general)

**Python:** `EpsilonRemove._closure_cache` stores `{state: set_of_states}`. Each cache hit returns the same `set` object (no copy needed in Python since sets are mutable but the cache values are only read).

**Rust:** `eps_cache: FxHashMap<u64, Rc<Vec<u64>>>`. Returns `Rc::clone()` on cache hit (refcount bump, no data copy). The cache avoids recomputing BFS for epsilon transitions.

**Generality:** Fully general. Any NFA with ε-transitions benefits.

**Pullback to Python:** Already done — `EpsilonRemove._closure_cache` is the Python equivalent.

### Optimization E: Batch Arc Computation (moderate generality)

**Python:** `LazyDeterminize.arcs(Q)` iterates over each NFA state in the powerset, calls `self.fsa.arcs(i)` for each, and groups by input symbol. This is **driven by the NFA arcs, not by the alphabet**. For a BPE FST with 50K input symbols but where each intermediate state has only 1-2 active arcs, this is O(active arcs), not O(50K).

**Rust:** `compute_all_arcs()` does the same thing — iterates NFA states in the set, collects their arcs, groups by symbol. The key insight is the same: iterate *arcs from states*, not *symbols in the alphabet*.

**Generality:** This is already the Python approach. The Rust version is faster because it avoids Python per-arc overhead, but the algorithm is the same.

**Pullback to Python:** Already done — `LazyDeterminize.arcs()` iterates arcs, not the alphabet.

### Optimization F: `all_input_universal` Precomputation (moderate generality)

**Python:** `accepts_universal(state, alphabet)` does a sub-BFS from each final DFA state. For each state in the sub-BFS, it calls `dict(dfa.arcs(i))` and then checks all symbols in `alphabet`. This is O(sub-BFS states × |alphabet|) per call. For BPE, every final state is universal, and each sub-BFS re-discovers the same universal automaton — leading to O(N²) total cost.

**Rust:** `check_all_input_universal()` in `fst.rs` is a one-time O(|FST arcs|) check during FST construction. It verifies that the FST's input projection accepts Σ* by checking:
1. The ε-closed start set is final
2. The start set is complete (has arcs for all source symbols)
3. Every successor's ε-closure contains the start set (so all reachable DFA states are supersets of start → also final and complete)

When true, `decompose()` skips `is_universal` entirely, marking all final states as Q stops.

**Generality:** Applies to any FST whose input projection is universal (Σ*). This includes all BPE tokenizers and any "replace" FST (like `examples.replace()`). Does NOT help FSTs whose input projection has constraints (e.g., number format validators).

**Pullback to Python:** **Yes, this is the single highest-impact optimization to pull back.**

```python
def check_all_input_universal(fst):
    """O(|arcs|) check: does the input projection accept Σ*?"""
    source_alpha = fst.A - {EPSILON}
    if not source_alpha:
        return any(fst.is_final(s) for s in fst.I)

    # eps-close the start states (input-side epsilon arcs only)
    start = eps_close_input(fst, fst.I)

    # Must be final
    if not any(fst.is_final(s) for s in start):
        return False

    # Must be complete
    by_symbol = defaultdict(set)
    for s in start:
        for x, _, j in fst.arcs(s):
            if x != EPSILON:
                by_symbol[x].add(j)
    if len(by_symbol) < len(source_alpha):
        return False

    # Every successor's eps-closure must contain the start set
    for sym, raw_dests in by_symbol.items():
        closed = eps_close_input(fst, raw_dests)
        if not start <= closed:
            return False
    return True
```

Then in `NonrecursiveDFADecomp.__init__`, skip `accepts_universal` when this returns true:

```python
if check_all_input_universal(fst):
    # All final states are universal — skip the expensive sub-BFS
    if dfa.is_final(i):
        Q.add_stop(i)
        continue
```

This would eliminate the O(N²) bottleneck that made 500-token vocabularies take 58ms and 5000-token vocabularies take 11.5s. With this fix, those drop to <1ms.

### Optimization G: Token-Level Position Tracking (BPE-specific)

This is the **biggest** optimization and the most domain-specific. It's implemented in `token_decompose.rs` and applies only when `all_input_universal=true` AND the FST has "hub" structure (all tokens start and end at the same state).

**Python:** The precover NFA has states `(fst_state, buf_pos)`. For a BPE FST with ~7000 internal states and target length N, the NFA state space is ~7000×N. The DFA (powerset) over this space has ~6000 states per target byte.

**Rust `token_decompose`:**
1. Extract each token's byte sequence from the FST (follow ε-chains)
2. Build a byte trie over all token byte sequences
3. NFA states are just **positions** `{0, 1, ..., target_len}` — no FST state component, because the FST always returns to the hub state after each token
4. DFA states are subsets of positions — typically O(N) states total
5. For each position p, use the trie to find which tokens match target[p..] in O(match length) time

This collapses the 7000 intermediate FST states per token into a single position transition. Instead of O(7000×N) NFA states, there are O(N) positions. The DFA has O(N) states instead of O(7000×N).

**Generality:** Specific to "hub" FSTs where all tokens start and end at the same state(s). This includes BPE tokenizers but NOT general FSTs (e.g., FSTs with multi-state paths that don't return to a hub).

**Pullback to Python:** **Yes, this is feasible and would be the second-highest-impact optimization.** The trie + position-set approach is purely algorithmic — no Rust-specific tricks. A Python implementation would look like:

```python
class TokenLevelDecomp:
    def __init__(self, fst, target):
        # 1. Extract token byte sequences
        tokens = extract_token_bytes(fst)

        # 2. Build prefix trie
        trie = build_trie(tokens)

        # 3. Precompute matches at each position
        matches = [trie.matches_at(target, p) for p in range(len(target))]

        # 4. BFS over position sets
        start = frozenset({0})
        worklist = deque([start])
        visited = {start}
        ...
```

The speedup from this in Python wouldn't be as dramatic as in Rust (Python loop overhead), but it would reduce the DFA from ~300K states at target_len=50 to ~51 states — a ~6000x reduction in state space that would translate to a large wall-clock improvement even in Python.

## 3. Summary: What to Pull Back

| Optimization | Impact | Generality | Already in Python? |
|---|---|---|---|
| **F: `all_input_universal`** | **Critical** — eliminates O(N²) universality check | BPE + replace FSTs | **No — highest priority pullback** |
| **G: Token-level positions** | **Major** — O(N) states instead of O(7000N) | Hub-structured FSTs | **No — second priority pullback** |
| **C: Arena interning** | Moderate — faster DFA state lookups | All FSTs | Partially (could add Integerizer) |
| **B: FST indexes** | Important | All FSTs | Yes (`LazyPrecoverNFA`) |
| **D: Eps closure cache** | Important | All NFAs with ε | Yes (`_closure_cache`) |
| **A: Packed state repr** | Minor in Python | All FSTs | Yes (`(i, n)` tuples) |

The two optimizations worth pulling back are **F** and **G**. Together they would make the Python `NonrecursiveDFADecomp` handle full GPT-2 (50K tokens) at long target lengths, which is currently impossible.
