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

---

## Revised Efficiency Analysis (2025-02-04)

The earlier issue list (above) was written before a careful analysis of where computation
actually occurs in the BFS. Several impact assessments were overstated. This section
provides a corrected analysis grounded in the traversal structure.

### Key Structural Property: Each DFA State Is Expanded Exactly Once

The main BFS in `PeekabooState.__init__` uses the `incoming` dict as its visited set.
A state is added to the worklist only when `next_state not in incoming` (line 341).
Therefore `dfa.arcs(state)` at line 339 is called **exactly once per DFA state** per
depth. There is no redundancy in the main BFS arc computations.

Furthermore, the PeekabooPrecover NFA is genuinely different at each depth — the
truncation boundary moves from $N+K$ to $(N+1)+K$ — so resume frontier states produce
different successors at depth $d+1$ than at depth $d$. Re-expanding them is not
redundant work; it discovers new states that were beyond the truncation boundary at the
previous depth.

### Where Additional `arcs()` Calls Actually Happen

The only place where `arcs()` is called outside the main BFS is inside
`UniversalityFilter._bfs_universal` (fst.py:897-915). This sub-BFS checks whether a
DFA state accepts $\Sigma^*$ by traversing the `TruncatedDFA` graph. It explores
**refined** states — subsets of the main BFS states produced by `TruncatedDFA.refine()`,
which filters to NFA states compatible with a specific target extension $y$ and clips
buffer strings to length $N$.

These refined states are distinct from the main BFS states, so the sub-BFS is genuinely
exploring new territory. However, it is bounded by the `UniversalityFilter` cascade:

1. **`all_input_universal` fast path** (BPE case): `_bfs_universal` is **never called**.
   Every final state is immediately classified as universal. This is the primary use case.

2. **Witness check**: ip-universal witness intersection, O(min(|witnesses|, |state|)).

3. **Monotonicity caches**: Subset/superset lookups against known-universal and
   known-non-universal states.

4. **`_bfs_universal`**: Only reached when all fast paths miss.

A subtlety: `_bfs_universal` caches the **starting state's** result (via `_add_pos` /
`_add_neg`), but intermediate states visited during the sub-BFS are NOT individually
cached. Two different main-BFS states triggering `_bfs_universal` could explore
overlapping sub-BFS regions. The monotonicity caches partially mitigate this (a known
universal/non-universal starting state can short-circuit via subset/superset), but
they don't cover intermediate states.

### Corrected Impact of Issues 3 and 5

**Issue 3 (refine allocations)** and **Issue 5 (LazyDeterminize arc cache)** were
originally motivated by the assumption that `arcs()` is called multiple times per state.
Given that the main BFS calls `arcs()` once per state, these are relevant only inside
`_bfs_universal` sub-BFS calls, which:

- Never fire for BPE FSTs (`all_input_universal = true`)
- Are bounded by the monotonicity cache for general FSTs
- Operate on refined states distinct from the main BFS

Their actual impact is **low** unless the FST triggers many `_bfs_universal` fallbacks
(e.g., non-BPE FSTs with few ip-universal witnesses and poor monotonicity cache hit
rates). In that specific scenario, caching the underlying `LazyDeterminize.arcs()` and
memoizing `refine()` would help reduce the cost of overlapping sub-BFS graphs.

### Corrected Impact of Issue 4 (Epsilon Closure Sharing) — Empirical Study

Each depth creates a fresh `EpsilonRemove` with an empty `_closure_cache`. The
`PeekabooPrecover` NFA has epsilon transitions only when the FST has **input-side**
epsilon arcs (`EPSILON in fst.A`). For FSTs without them (the common case: BPE
tokenizers, simple replacement FSTs), `EpsilonRemove` is a pure no-op wrapper.

#### Empirical hit/miss study

Instrumented all test examples to measure:
- **LD arcs**: `LazyDeterminize.arcs()` calls from main BFS vs universality sub-BFS
- **Cross-depth eps closure**: correctness of sharing (would cached result be right?)
- **Universality filter**: which cascade level resolves each check
- **Time in eps closure**: fraction of total runtime

```
Example                eps? | XDepth closure (correct/WRONG/miss)     | Univ filter (AUI/other)
abc                       N |    2461 correct /    0 WRONG /  781 miss |  2930 AUI /     0 other
delete_b                  N |      47 correct /    0 WRONG /    7 miss |    21 AUI /     0 other
samuel                    Y |     124 correct /    7 WRONG /   14 miss |     0 AUI /    42 other
small                     N |     192 correct /    0 WRONG /   34 miss |     0 AUI /   130 other
duplicate                 Y |     567 correct /  214 WRONG /  156 miss |   815 AUI /     0 other
number_comma_sep          Y |    1328 correct /   45 WRONG /  254 miss |     0 AUI /   951 other
lookahead                 N |     872 correct /    0 WRONG /   72 miss |     0 AUI /   358 other
weird_copy                Y |     257 correct /    0 WRONG /   63 miss |   258 AUI /     0 other
triplets_of_doom          N |    4241 correct /    0 WRONG /  349 miss |     0 AUI /   512 other
infinite_quotient         Y |      12 correct /    0 WRONG /    2 miss |     0 AUI /     2 other
parity                    Y |      33 correct /    3 WRONG /    2 miss |     0 AUI /     6 other
```

**Key finding 1: `_bfs_universal` is never reached.** The "univ" (from-universality)
column was **all zeros** across every example. The UniversalityFilter always short-circuits
via `all_input_universal`, witnesses, or monotonicity caches before reaching the BFS
fallback. This means **DFA transition caching (Issue 5) has zero benefit** for these
examples, and **refine memoization (Issue 3) has zero benefit** since `refine` is only
called inside `_bfs_universal` sub-BFS.

**Key finding 2: Cross-depth eps closure sharing is unsafe for eps FSTs.** The WRONG
column shows that naively sharing the cache would produce incorrect results for FSTs
with input-epsilon arcs (duplicate: 214 WRONG = 22.8%). The incorrect entries are for
NFA states at the truncation boundary (`len(ys) >= N_previous`), whose transitions
change when the target length grows. For no-eps FSTs, WRONG is always 0 — but closures
are trivially `{state}`, so there's nothing meaningful to cache.

**Key finding 3: Eps closure is 11-19% of total runtime.**

```
Example                eps? |  total_ms  closure_ms    pct
abc                       N |     51 ms     8.3 ms  13.7%
triplets_of_doom          N |     43 ms     8.1 ms  14.5%
lookahead                 N |     13 ms     1.7 ms  11.7%
duplicate                 Y |     19 ms     5.1 ms  19.1%
number_comma_sep          Y |     25 ms     4.8 ms  13.7%
```

For no-eps FSTs, this is pure function call overhead on trivial `{state}` closures
(~2μs each). The fix is not caching — it's **bypassing `EpsilonRemove` entirely**.

#### Fix applied: bypass EpsilonRemove for no-eps FSTs ✅

Added `PeekabooPrecover.epsremove()` override that returns `self` (no wrapper) when
`EPSILON not in fst.A`. Benchmark results (best of 5 runs):

```
Example                has_eps |  old_ms  new_ms  speedup
abc                          N |   47.5    40.4    1.18x
delete_b                     N |    0.6     0.5    1.11x
small                        N |    3.4     3.1    1.09x
lookahead                    N |   12.0     8.9    1.35x
triplets_of_doom             N |   38.3    31.2    1.23x
samuel                       Y |    1.7     1.9    1.00x  (unaffected)
duplicate                    Y |   16.6    17.0    1.00x  (unaffected)
```

**1.09x–1.35x speedup** for FSTs without input-epsilon arcs. No effect on eps FSTs.

#### Cross-depth sharing: attempted and abandoned

Also implemented cross-depth sharing with selective invalidation (copy parent's cache,
remove entries where `len(ys) >= N_previous`). Benchmarked separately: the dict
comprehension overhead offset the savings — neutral to slightly negative on all eps-FST
examples. The closure computations are cheap enough (~2-5μs) that the copy cost dominates.
Reverted.

### Remaining Optimizations

The following are the recommendations that survive the corrected analysis and empirical
study.  They are ordered by expected impact, but note the caveats raised by the study.

#### 1. Integer-Encode NFA States (replaces strings with arithmetic)

**Current**: NFA states are `(fst_state, ys_string, truncated)`. The `ys` string is
always either a prefix of the target (when `len(ys) <= N`) or `target + one_symbol`
(when `len(ys) == N+1`). The first $N$ characters are redundant — they're always
`target[:len(ys)]`.

**Proposed**: Replace with `(fst_state, progress: int, ext_sym: Optional[symbol], truncated: bool)`:
- `progress` ∈ {0, ..., N}: how many target symbols have been matched
- `ext_sym`: the one lookahead symbol (meaningful only when `progress == N`; `None` otherwise)
- `truncated`: bool (meaningful only when `progress == N`)

This turns:
- String hashing → integer/tuple hashing (O(1) instead of O(N))
- String slicing and concatenation → integer increment
- Prefix comparison in `PeekabooPrecover.arcs()` → impossible by construction (progress always tracks a valid prefix position)
- `refine()` → simple filter on `ext_sym == y` (no string slicing)

Since NFA states are elements of frozenset DFA states, the improvement compounds:
hashing a DFA state of size $k$ goes from O($k \times N$) to O($k$).

**Caveat**: The empirical study shows the main BFS visits each state exactly once.
The hashing cost is real but may be modest relative to the arc-computation cost in
`LazyDeterminize.arcs()` (iterating NFA states, grouping by symbol, building frozensets).
The improvement is asymptotic in $N$ (target length), so it matters more for long targets.

#### 2. Integer-Intern DFA States (frozenset → int ID)

**Current**: DFA states are `frozenset`s. Every `incoming` lookup, worklist membership
check, and dict key operation hashes the full frozenset — O(|state|) per operation.

**Proposed**: Assign each unique frozenset an integer ID via an interner (cf. the
`Integerizer` from arsenal, or the approach in Rust `powerset.rs`). Benefits:
- O(1) hash/compare for all DFA state operations
- Enables integer-indexed transition tables
- Metadata (is_truncated, relevant_symbols, is_final_for_symbol) can be stored in
  arrays indexed by ID, computed once at interning time

The interning cost is paid once per new DFA state. All subsequent operations are O(1).

**Caveat**: Every operation that needs the NFA states inside a DFA state (arc computation,
metadata extraction, universality checks) must unpack via a reverse lookup. The main BFS
calls `LazyDeterminize.arcs(Q)` once per state, which iterates `for i in Q:` — this
would become a table lookup + iteration over the stored frozenset. Whether the net effect
is positive depends on how many hash/compare operations are saved vs how many unpack
operations are added. The study suggests caution: the main BFS is the dominant cost, and
arc computation (not hashing) is its bottleneck.

#### 3. Single Backward BFS for All Symbols in `_trimmed_fsa`

**Current**: `Peekaboo.__call__` runs a separate `_trimmed_fsa` backward BFS for each
symbol $y$ in the target alphabet (line 150-151). For byte-level alphabets (|Σ| = 256),
this means 256 separate backward BFS traversals through the same `merged_incoming` graph.

**Proposed**: Run one backward BFS from the union of all Q/R stop states, tagging each
backward-reachable state with which symbols it serves. Then partition the tagged graph
into per-symbol FSAs. This replaces O(|Σ| × |states|) work with O(|states| + |Σ|).

In practice, many symbols may have empty Q/R (no decomposition), so the savings depend
on how many symbols are active. But for large alphabets this could be significant.

#### 4. Adaptive Truncation Policy (Algorithmic Change)

**Current**: K=1 always. Every DFA state that has output beyond the target by more than
one symbol is truncated, generating resume frontier entries for the next depth.

The comment at lines 11-26 of `peekaboo_recursive.py` already identifies this as a
cost-benefit knob. Aggressive truncation (small K) minimizes per-depth work but
maximizes the number of truncated states (larger resume frontiers, more work deferred
to later depths). Lazy truncation (larger K) allows more resolution at the current
depth but risks a larger intermediate state space.

Ideas for smarter policies:
- **Adaptive K**: If the sub-automaton reachable from a state is small (few states,
  no cycles), don't truncate — resolve it entirely now.
- **Cycle detection**: Only truncate states on cycles. Acyclic sub-automata are finite
  and will terminate without truncation.
- **Depth-varying K**: Use larger K at later depths where the target prefix constrains
  the state space more tightly.

This is the deepest algorithmic lever — potentially changing total work across depths —
but also the hardest to evaluate without empirical measurement on real FSTs.

### Revised Priority Table (Empirically Informed)

| Priority | Recommendation | Status | Impact |
|----------|---------------|--------|--------|
| 1 | Bypass EpsilonRemove for no-eps FSTs | ✅ Done | 1.09x–1.35x on no-eps FSTs |
| 2 | Integer-encode NFA states | Open | Scales with N; unclear constant factor |
| 3 | Integer-intern DFA states | Open | Saves hashing; adds unpack overhead |
| 4 | Single backward BFS for all symbols | Open | Scales with |Σ| |
| 5 | Adaptive truncation policy | Open | Unknown; needs experiments |
| — | ~~DFA transition cache (Issue 5)~~ | Rejected | `_bfs_universal` never called |
| — | ~~Refine memoization (Issue 3)~~ | Rejected | `_bfs_universal` never called |
| — | ~~Epsilon closure sharing across depths~~ | Rejected | Unsafe for eps FSTs; overhead ≥ savings |
| — | ~~Persist UniversalityFilter caches~~ | Rejected | `_bfs_universal` never called |
