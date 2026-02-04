# Algorithm Assessment: Computing Next-Symbol Probabilities of a Transduced Language Model

## The Core Problem

Given an FST `f` and a target prefix **y**, compute for each next symbol `z`:

$$P(z \mid \mathbf{y}) \propto \sum_{x \in \mathcal{Q}(\mathbf{y}z)} P_{\text{LM}}(x) + \sum_{x \in \mathcal{R}(\mathbf{y}z)} P_{\text{LM}}(x \cdot \text{EOS})$$

where $\mathcal{P}(\mathbf{y}) = \mathcal{Q}(\mathbf{y})\Sigma^* \sqcup \mathcal{R}(\mathbf{y})$ is the precover decomposition.

The precover $\mathcal{P}(\mathbf{y})$ is the set of all source strings whose transduction
through `f` begins with **y**. The quotient $\mathcal{Q}$ captures source strings that
can continue producing more output; the remainder $\mathcal{R}$ captures those that
have terminated. Both are represented as finite automata.

---

## Inventory of Current Implementations

### Decomposition Algorithms (Python)

| Algorithm | File | Tested | Terminates | Notes |
|-----------|------|--------|------------|-------|
| `Precover` | `eager_nonrecursive.py` | Reference impl | Yes | Materializes full DFA, correct |
| `NonrecursiveDFADecomp` | `dfa_decomp_nonrecursive.py` | Yes | Yes | Same approach, cleaner interface |
| `RecursiveDFADecomp` | `dfa_decomp_recursive.py` | Yes (xfail) | **No** | Hangs on `triplets_of_doom` |
| Peekaboo (nonrecursive) | `peekaboo_nonrecursive.py` | Yes | Yes | Batches all next-symbols into one DFA |
| Peekaboo (recursive) | `peekaboo_recursive.py` | Yes | Yes | Recursive + truncation markers |
| `TokenDecompose` | `token_decompose.py` | Yes | Yes | BPE-only fast path |
| `EagerNonrecursive` | `eager_nonrecursive.py` | No | Yes | `AbstractAlgorithm` subclass, string-level worklist |
| `LazyNonrecursive` | `lazy_nonrecursive.py` | No | Yes | `AbstractAlgorithm` subclass, string-level worklist |
| `BuggyLazyRecursive` | `lazy_recursive.py` | No | Yes | Self-described as suboptimal |
| `LazyRecursive` | `lazy_recursive.py` | No | Yes | Fix for the above |

### Decomposition Algorithms (Rust)

| Algorithm | File | Dispatch condition |
|-----------|------|--------------------|
| `decompose` (generic) | `decompose.rs` | `all_input_universal = false` |
| `token_decompose` (BPE-optimized) | `token_decompose.rs` | `all_input_universal = true` |

### Inference Algorithms (Python)

| Algorithm | File | Approach |
|-----------|------|----------|
| `prioritized_enumeration` | `enumeration.py` | Best-first search weighted by LM log-probs |
| `importance_sampling` | `enumeration.py` | Sample paths, accumulate partition function |
| `crude_importance_sampling` | `enumeration.py` | Same, without Q/R decomposition |

---

## Assessment of Each Approach

### `prioritized_enumeration`

The most practical approach for neural LMs today. Best-first search over the
precover DFA, weighted by LM log-probabilities. Can terminate early after
`max_steps` and still produce a useful approximation.

**Strength**: Anytime algorithm; high-probability paths explored first.

**Weakness 1**: Rebuilds the entire `Precover` from scratch for each target,
with no incremental reuse across successive target symbols.

**Weakness 2**: Two-phase architecture — fully materializes the precover DFA
(powerset determinization + universality classification for all states), *then*
traverses it with LM weights. In the anytime setting (`max_steps` << total
DFA paths), most of the upfront DFA construction is wasted: you build the
entire DFA but only traverse the high-probability region. See "Fused lazy
DFA construction + LM-weighted enumeration" below for the fix.

### `importance_sampling`

Theoretically elegant: sample paths proportional to LM probability, accumulate
the partition function. In practice, **hung on GPT-2 BPE** because `Precover(fst, target)`
determinizes the full precover NFA (~98K states for BPE). Would work if it
used the Rust `token_decompose` fast path.

### Peekaboo (both variants)

The only algorithms designed specifically for the next-symbol prediction problem.
Instead of computing `Precover(target + z)` separately for each next symbol `z`,
they build a *single* DFA for $f^{-1}(\mathbf{y} \cdot \mathcal{Y} \cdot \mathcal{Y}^*)$
and extract all $|\mathcal{Y}|$ decompositions from it.

**This is the right idea**: amortize the DFA construction across all next symbols.

The **nonrecursive** variant is simpler: it takes a full target string and builds the
Peekaboo DFA from scratch each time. Sufficient for one-shot evaluation.

The **recursive** variant adds truncation markers and a `resume_frontiers` frontier-tracking
mechanism that enables **incremental computation** via the `>>` operator. When
generating a target string one symbol at a time (ancestral sampling, beam search),
advancing from $\mathbf{y}$ to $\mathbf{y}z$ only processes the frontier states
recorded in `parent.resume_frontiers[z]`, rather than re-exploring the entire DFA from scratch.
This is the correct algorithm for autoregressive decoding, where the access pattern
is a sequence of single-symbol extensions of a growing prefix.

Note: the current `__call__` API processes the full target string internally (via
iterated `>>=`), but does not expose the intermediate `PeekabooState` for reuse
across decoding steps. Wiring this up is straightforward -- keep the `PeekabooState`
alive between calls and advance it with `>>=` as each new target symbol is sampled.

### `RecursiveDFADecomp`

Has the right incremental structure (`state >> y` advances one symbol), but uses
an unbounded target-side buffer and doesn't terminate in general.
The truncation strategy in Peekaboo was the fix for this.

### `Precover` / `NonrecursiveDFADecomp`

Correct reference implementations. Rebuild everything from scratch for each
target string. Fine for testing, wrong for production.

### `AbstractAlgorithm` family

`EagerNonrecursive`, `LazyNonrecursive`, `BuggyLazyRecursive`, `LazyRecursive`
operate at the **string level** (worklists of source strings). The earlier
assessment dismissed this as "the wrong abstraction," but that conflates two
issues. The string level is actually the *right* interface for LM interaction
— the LM scores source string prefixes, not DFA states. What's wrong is the
BFS traversal order (which ignores LM probability) and the lack of DFA-level
bookkeeping for state deduplication and Q/R classification.

The fix is not to abandon strings but to fuse string-level enumeration with
DFA-level state management under LM-priority ordering. See "Fused lazy DFA
construction + LM-weighted enumeration" below. The `EagerNonrecursive`
implementation already maintains a `self.state` dict mapping source strings to
DFA states — this is exactly the bookkeeping needed for the fused algorithm,
just with BFS replaced by a priority queue.

### `TokenDecompose` / `token_decompose` (Rust)

For BPE tokenizers where `all_input_universal = true`, this gives 5000x+
speedup by collapsing each token into a single transition. DFA states are
position subsets `{0..target_len}` instead of full NFA powersets.

---

## Recommended Strategy

The best architecture combines four insights:

### 1. Peekaboo's batching

Build one DFA for all next-symbol extensions, not $|\mathcal{Y}|$ separate
decompositions. This is an asymptotic win when the target alphabet is large
(e.g., 256 bytes).

### 2. Incremental (recursive) computation across decoding steps

During autoregressive generation, target symbols are produced one at a time:
$\mathbf{y}_1, \mathbf{y}_1\mathbf{y}_2, \mathbf{y}_1\mathbf{y}_2\mathbf{y}_3, \ldots$
The recursive Peekaboo's `>>` operator advances from step $k$ to step $k+1$ by
resuming only from the frontier states (`resume_frontiers`) rather than rebuilding the entire
DFA. This avoids redundant work proportional to the length of the already-generated
prefix. The nonrecursive variant cannot do this -- it must start from scratch at
every step.

The truncation strategy (bounding the output buffer at $N+K$) is what makes this
terminate where `RecursiveDFADecomp` does not. The truncation policy is a
cost-benefit knob: aggressive truncation (small $K$) minimizes work per step but
may defer more work to later steps; lazy truncation allows more sharing across
steps but risks larger intermediate state spaces.

### 3. Rust acceleration

The DFA construction (powerset determinization + universality detection) should
happen in Rust. `decompose.rs` already does this for a single target. Extending
it to the Peekaboo construction (target + one extra symbol, with per-symbol
Q/R extraction) would be the natural next step. The incremental state (`resume_frontiers`
frontier) would live on the Rust side and be advanced per decoding step.

### 4. Fused lazy DFA construction + LM-weighted enumeration

The current `prioritized_enumeration` is two-phase: (1) fully materialize
the precover DFA, (2) traverse it with LM weights. This is wasteful in the
anytime setting — you build the entire DFA upfront but only traverse the
high-probability region.

The fix is to fuse DFA construction with LM-weighted string enumeration:

1. Start with a **lazy** DFA (no upfront materialization; `LazyDeterminize`
   already supports this)
2. Priority queue holds `(log_prob, dfa_state, lm_state)` triples
3. Pop the highest-priority item
4. Compute DFA arcs **on-demand** (lazy powerset determinization)
5. For each successor DFA state, check if final:
   - If final and universal (Q): accumulate weight contribution, stop expanding
   - If final and not universal (R): accumulate weight $\times P(\text{EOS})$,
     continue expanding
   - If not final: push children weighted by `log_prob + lm_logp(x)`
6. Universality is checked lazily per-state using `UniversalityFilter`
   (cached; amortized sub-linear cost via monotonicity)
7. Stop after `max_steps` — only the high-probability region of the DFA has
   been materialized

There is no ordering conflict between LM-priority traversal and universality
detection. The main loop pops items in LM-priority order; universality
checking is a per-state subroutine (BFS from the state in question) that runs
when a final state is first encountered. These operate at different levels
and do not interfere.

The DFA still provides essential bookkeeping: state identity (multiple source
strings converge to the same DFA state and share futures), Q/R classification
(per-state, cached), and termination guarantee (finite DFA state space).

**Design choice**: the current `LocatorMaxHeap` deduplicates by DFA state
(keeps the max-weight path). For exact probability computation, you'd need
to accumulate *all* paths reaching each state. This is a tractable extension
— replace max-deduplication with weight accumulation per `(dfa_state, lm_state)`
pair.

**Connection to `AbstractAlgorithm` family**: `EagerNonrecursive` already
maintains a `self.state` dict mapping source strings → DFA states. The fused
algorithm is essentially the same idea with BFS replaced by a priority queue
and eager DFA construction replaced by lazy. The string level is the right
interface for LM interaction; the DFA level is the right interface for state
management. The fusion combines both.

### Proposed Architecture

```
       ┌──────────────────────────────────────────────┐
       │  Rust: IncrementalPeekaboo                    │
       │  - Maintains PeekabooState across steps       │
       │  - advance(z): extend target by one symbol    │
       │    using resume_frontiers (not full rebuild)    │
       │  - Powerset determinize (incremental)          │
       │  - UniversalityFilter with caching             │
       │  - Return per-symbol Q/R stop states           │
       └─────────────────────┬────────────────────────┘
                             │
       ┌─────────────────────▼────────────────────────┐
       │  Python: priority-weighted                     │
       │  traversal of the DFA                          │
       │  - Best-first by LM logprob                    │
       │  - Q terms: sum weight                         │
       │  - R terms: sum weight·P(EOS)                  │
       │  - Anytime: stop when good enough              │
       │  - Feed sampled symbol back to Rust for        │
       │    next incremental step                       │
       └───────────────────────────────────────────────┘
```

### The BPE Special Case

For BPE tokenizers, `all_input_universal = true` and `token_decompose` gives
5000x+ speedup. The Peekaboo approach should also have a token-level
specialization: since every state is ip-universal, the Peekaboo DFA has very
simple structure (position sets), and computing next-symbol decompositions
should be nearly free. This would be the highest-impact optimization for the
neural LM use case.

---

## Consolidation Recommendation

Collapse from ~10 Python decomposition implementations to ~3:

### Keep

- **`Precover`** (`eager_nonrecursive.py`) -- reference implementation for testing
- **`RustDecomp`** (`rust_bridge.py`) -- production single-target decomposition
- **`prioritized_enumeration`** (`enumeration.py`) -- end-user inference API

### Keep and refactor

- **Peekaboo recursive** (`peekaboo_recursive.py`) -- the right algorithm for autoregressive next-symbol prediction; supports incremental `>>` advancement via `resume_frontiers` frontier tracking; port the incremental DFA construction to Rust
- **Peekaboo nonrecursive** (`peekaboo_nonrecursive.py`) -- simpler variant for one-shot (non-incremental) next-symbol prediction; useful as a reference and for cases where the full target is known upfront

### Remove or archive

- `BuggyLazyRecursive` -- self-described as buggy
- `LazyRecursive` -- fix for the above, but still string-level BFS
- `EagerNonrecursive` -- string-level BFS; however, its `self.state` dict pattern (mapping source strings to DFA states) is the right bookkeeping for the fused algorithm; useful as design reference
- `LazyNonrecursive` -- same pattern with lazy DFA; closest existing code to the fused lazy approach
- `RecursiveDFADecomp` -- doesn't terminate; Peekaboo recursive is the fix (same incremental idea, with truncation to guarantee termination)
- `PeekabooStrings` -- string-level version of state-level Peekaboo
- `crude_importance_sampling` -- strictly worse than `importance_sampling`
- `NonrecursiveDFADecomp` -- redundant with `Precover` (same algorithm, slightly different interface)

### Keep but may not need long-term

- `TokenDecompose` (Python) -- superseded by Rust `token_decompose`, but useful as fallback
- `importance_sampling` -- theoretically interesting but needs Rust backend to be practical

---

## Performance Reference

### Rust `token_decompose` (GPT-2 BPE, 50K tokens)

| target_len | total | init | BFS | DFA states |
|-----------|-------|------|-----|-----------|
| 100 | 32ms | 32ms | 0.1ms | 101 |
| 500 | 26ms | 25ms | 0.7ms | 501 |
| 1000 | 35ms | 32ms | 2.4ms | 1001 |
| 4000 | 57ms | 29ms | 28ms | 4001 |

### vs old generic decompose (before `token_decompose`)

| target_len | old total | new total | speedup |
|-----------|----------|----------|---------|
| 50 | 1502ms | 0.3ms | ~5000x |
| 80 | 5500ms | 0.2ms | ~27000x |
| 1000 | impossible | 2.4ms | -- |

Init cost (~28ms) is token extraction + trie building; could be cached across calls.

---

## Efficiency Analysis: PeekabooRecursive

The PeekabooRecursive algorithm (`peekaboo_recursive.py`) has the right algorithmic
structure — incremental `>>` advancement via `resume_frontiers` frontier tracking, truncation to
guarantee termination, batched next-symbol computation. This section catalogs
implementation-level efficiency issues: places where the current Python code does
redundant work, misses caching opportunities, or makes unnecessary allocations.
The algorithm is *correct* (passes all tests); these are purely performance concerns.

### Issue 1: TruncatedDFA rebuilt per (state, symbol) pair ✅ FIXED

**Location**: `PeekabooState.__init__`

Previously, a new `TruncatedDFA` was constructed inside the BFS inner loop for each
`(worklist_state, relevant_symbol)` pair. Now pre-built as
`{y: TruncatedDFA(...) for y in target_alphabet}` once before the BFS loop.

### Issue 2: No universality caching ✅ FIXED

**Location**: `PeekabooState.__init__`; `UniversalityFilter` in `fst.py`

Previously, each universality check called `Lazy.accepts_universal`, which created a
`StartAt` → `LazyDeterminize` → `EpsilonRemove` wrapper chain and did a fresh BFS
from scratch — *every* time, for *every* worklist state.

Now uses `UniversalityFilter` (one per target symbol per PeekabooState), which provides:

- **`all_input_universal` fast path**: For BPE-like FSTs, every final state is
  universal — the BFS is eliminated entirely. This is the single biggest win.
- **ip-universal witness check**: O(|state|) set-intersection test using precomputed
  witnesses `{(q, target+y, False) for q in ip_universal_states}`.
- **Superset monotonicity cache**: If `S` is known universal and `S ⊆ T`, then `T`
  is universal (element-indexed lookup, avoids BFS).
- **Subset monotonicity cache**: If `S` is known non-universal and `T ⊆ S`, then `T`
  is non-universal (element-indexed lookup, avoids BFS).
- **BFS fallback**: Only reached when all fast paths miss. Results are cached for
  future monotonicity lookups.

The `all_input_universal` flag and `ip_universal_states` set are computed once in
`Peekaboo.__init__` and propagated through the `>>` chain (parent → child), so
these O(|FST|) computations happen exactly once per FST, not per step.

**Changes to `UniversalityFilter`** (`fst.py`):
- Added `all_input_universal` and `witnesses` keyword parameters to `__init__`,
  allowing pre-computed values to be injected instead of recomputed.
- Added `is_final` guard at the top of `is_universal`: the `all_input_universal`
  fast path previously returned `True` unconditionally (safe in its original call
  context where only final states were tested, but incorrect in general). Now
  explicitly checks `dfa.is_final(dfa_state)` first.

### Issue 3: `TruncatedDFA.refine` allocates a new frozenset on every arc traversal

**Location**: `TruncatedDFA.arcs`, line ~360; `TruncatedDFA.refine`, line ~348

```python
def arcs(self, state):
    for x, next_state in self.dfa.arcs(state):
        yield x, self.refine(next_state)

def refine(self, frontier):
    N = len(self.target)
    return frozenset(
        (i, ys[:N], truncated) for i, ys, truncated in frontier
        if ys[:min(N, len(ys))] == self.target[:min(N, len(ys))]
    )
```

Every arc traversal in the `TruncatedDFA` calls `refine`, which:
1. Iterates over every NFA state in the powerset state
2. Slices the `ys` string (O(N) for target length N)
3. Compares prefixes (O(N))
4. Constructs a new `frozenset` (O(|powerset state|) hashing)

This is called during `accepts_universal`'s BFS, so the total cost is
`O(|reachable states| × |alphabet| × |powerset state| × N)` per universality check.

**Fix**: Memoize `refine` results (the input is a frozenset, which is hashable).
Or better, avoid `refine` entirely by using NFA states that already track the
clipped target position as an integer index rather than a string prefix.

**Impact**: Low. The main BFS expands each state exactly once, so `refine` is
never called from the main loop. It's only called inside `UniversalityFilter._bfs_universal`
sub-BFS calls. With the UniversalityFilter's own caching (monotonicity, witnesses,
`all_input_universal`), most universality checks short-circuit before reaching
`_bfs_universal`, so few sub-BFS calls happen and overlap between their explored
regions is small. A dict cache is trivial to add but unlikely to matter much.

### Issue 4: No epsilon closure sharing across recursive `>>` steps

**Location**: `PeekabooState.__init__`, line ~151

```python
dfa = PeekabooPrecover(self.fst, target).det()
```

Each `PeekabooState` constructs a fresh `PeekabooPrecover` NFA and wraps it in
a fresh `LazyDeterminize`, which internally creates a fresh `EpsilonRemove` with
an empty `_closure_cache`. The epsilon closures computed in step $k$ are discarded
at step $k+1$.

For the FST's NFA states that haven't changed between steps (i.e., states where
$|ys| < N_{k-1}$, which remain identical in the extended NFA), the epsilon
closures are the same. Re-computing them is wasted work.

**Fix**: Share the `EpsilonRemove._closure_cache` across PeekabooState instances
(pass it through from parent to child). Or, pre-compute and cache epsilon closures
on the FST object itself (similar to how `LazyPrecoverNFA` in `goo.py` indexes
arcs on the FST).

**Impact**: Modest for small FSTs; significant for large FSTs with many
epsilon transitions (e.g., BPE tokenizers with ~50K tokens, each introducing
epsilon arcs).

### Issue 5: `LazyDeterminize` has no arc cache — powerset construction is re-done on every `arcs()` call

**Location**: `lazy.py:172`, `LazyDeterminize.arcs`

```python
def arcs(self, Q):
    tmp = defaultdict(set)
    for i in Q:
        for a, j in self.fsa.arcs(i):
            tmp[a].add(j)
    for a, j in tmp.items():
        yield a, frozenset(j)
```

`LazyDeterminize` stores no cache. Every call to `dfa.arcs(state)` iterates over
all NFA states in the powerset state and collects transitions from scratch. In the
main PeekabooState BFS, each state is visited once (via the `_arcs` dict as visited
set), so this isn't a problem there. But during `accepts_universal`, the
`TruncatedDFA.arcs` method calls `self.dfa.arcs(state)` and then `refine`, and the
underlying `dfa.arcs(state)` is the uncached `LazyDeterminize`. If the same DFA
state is queried via the TruncatedDFA for multiple symbols `y`, the arc computation
is repeated.

**Fix**: Add a `dict` cache to `LazyDeterminize` keyed by the frozenset state.
Since states are already frozensets (hashable), this is straightforward.

**Impact**: Proportional to how many times the same DFA state appears in
universality BFS across different symbols. In the worst case, a state relevant
to $k$ symbols has its arcs computed $k$ times instead of once.

### Issue 6: ARCS list in `__call__` grows without bound ✅ FIXED

**Location**: `Peekaboo.__call__`

Previously, a flat `ARCS` list accumulated every arc from all $N+1$ depth levels.
Many of these arcs were unreachable from start states to Q/R stops, and the arc
data was stored twice (once per-state in `incoming`, once in the flat list).

Now the per-depth `incoming` (reverse-arc) dicts are merged into a single shared
graph, and per-symbol trimmed FSAs are extracted via backward BFS from Q/R stop
states through the merged graph.  Since all states are forward-reachable (found by
BFS) and the backward BFS selects only states that can reach the stops, the
resulting FSAs are trimmed by construction — no unreachable arcs, no redundant
storage.  The same merged graph serves all target symbols; each symbol's Q and R
pull out only their reachable slice.

### Issue 7: NFA states use string prefixes — integer positions would be cheaper

**Location**: `PeekabooPrecover` NFA states are `(i, ys, truncated)` where `ys`
is a string prefix.

The NFA states carry the actual string prefix `ys` (e.g., `'hel'` for target
`'hello'`). These strings are:
- Compared frequently (prefix checks in `PeekabooPrecover.arcs`)
- Hashed as part of frozenset DFA states
- Sliced and concatenated (O(N) per operation)

The string `ys` only needs to encode "how far along the target we are" plus
"what's the extra symbol beyond the target" (for the truncated case). This
could be represented as `(position: int, extra: Optional[symbol])` instead
of a full string, reducing hashing and comparison from O(N) to O(1).

Compare with `LazyPrecoverNFA` in `goo.py`, which already uses integer positions
`(i, n)` instead of string prefixes. The PeekabooPrecover needs slightly more
(the extra symbol beyond target), but `(i, n, extra_sym, truncated)` would still
be much cheaper than `(i, ys_string, truncated)`.

**Fix**: Replace `ys` string with `(int position, Optional[symbol] extra)` in the
NFA state representation.

**Impact**: Reduces hashing cost from O(N) to O(1) per NFA state. Since NFA
states are elements of frozenset DFA states, this compounds: hashing a DFA state
of size $k$ goes from O(k × N) to O(k).

### Issue 8: The `>>` operator creates a full PeekabooState — no lightweight advancement

**Location**: `PeekabooState.__rshift__`, line ~247

```python
def __rshift__(self, y):
    return PeekabooState(self.fst, self.target + y, parent=self)
```

Each `>>` call constructs a complete `PeekabooState`, which includes:
- A new `PeekabooPrecover` NFA (Issue 4)
- A new `LazyDeterminize` DFA (Issue 5)
- A full BFS over resume-frontier states
- Universality checks for all relevant symbols (Issue 2)
- A new `incoming` dict, `decomp` dict, and `resume_frontiers` dict

There is no way to do a "lightweight" advancement that, e.g., only computes
the resume_frontiers for a specific symbol of interest rather than all symbols. In the
autoregressive decoding use case, you typically sample one symbol and advance
— but the PeekabooState computes decompositions for *all* next symbols.

This is by design (it's the "batched" property), but it means you can't
cheaply advance without computing everything. A two-phase API — first compute
the batched decompositions, then advance cheaply using only the selected
symbol's resume frontier — would avoid wasted work in the advance step.

### Summary: Priority-ordered fix list

| Priority | Issue | Status | Effort | Impact |
|----------|-------|--------|--------|--------|
| 1 | Cache `TruncatedDFA` per symbol (Issue 1) | ✅ Done | Low | Medium |
| 2 | Add universality caching (Issue 2) | ✅ Done | Medium | High |
| 3 | Use integer positions instead of string prefixes (Issue 7) | Open | Medium | High |
| 4 | Share epsilon closure cache across steps (Issue 4) | Open | Low | Medium |
| 5 | Memoize `TruncatedDFA.refine` (Issue 3) | Open | Low | Low |
| 6 | Add arc cache to `LazyDeterminize` (Issue 5) | Open | Low | Low-Medium |
| 7 | Trim or lazily collect ARCS (Issue 6) | ✅ Done | Low | Low |
| 8 | Two-phase advance API (Issue 8) | Open | Medium | Context-dependent |

Issues 1, 2, and 6 are now fixed. Issue 2 (universality caching via `UniversalityFilter`)
is the single biggest win for general FSTs, and for BPE FSTs the `all_input_universal`
fast path makes the universality check essentially free. Issue 6 (trimmed FSA extraction
via backward BFS over merged `incoming` graph) eliminates redundant arc storage and
produces minimal machines per symbol. Issue 7 (integer positions) is the highest-impact
remaining item — a cross-cutting improvement that speeds up every operation touching
NFA states.
