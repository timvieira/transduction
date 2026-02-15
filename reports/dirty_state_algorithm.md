# Dirty-State Incremental Decomposition

## 1. Background and Problem Setting

Given a finite-state transducer $T$ over input alphabet $\mathcal{X}$ and output
alphabet $\mathcal{Y}$, and a target string $\boldsymbol{y} \in \mathcal{Y}^*$, the
**decomposition problem** partitions the source precover
$\mathcal{P}(\boldsymbol{y}) = \text{proj}_\mathcal{X}(T \circ
\boldsymbol{y}\mathcal{Y}^*)$ into a **quotient** $Q(\boldsymbol{y})$ and
**remainder** $R(\boldsymbol{y})$:

$$\mathcal{P}(\boldsymbol{y}) = Q(\boldsymbol{y}) \cdot \mathcal{X}^* \sqcup R(\boldsymbol{y})$$

$Q$ contains source prefixes whose every continuation produces output beginning
with $\boldsymbol{y}$ (i.e., the prefix is *universal* over $\mathcal{X}$).
$R$ contains source prefixes that produce $\boldsymbol{y}$ exactly but are not
universal. Both are represented as finite-state automata (FSAs).

In **autoregressive decoding**, the target is built one symbol at a time:
$\varepsilon \to y_1 \to y_1 y_2 \to \cdots$. Each step requires a fresh
decomposition. Recomputing from scratch at each step is wasteful. The
dirty-state algorithm exploits the incremental structure of prefix extensions
to avoid redundant work.

## 2. The Precover NFA and Powerset DFA

### 2.1 Precover NFA

The decomposition is computed via the **Precover NFA**, an NFA whose states
are pairs $(i, \boldsymbol{b})$ where $i$ is an FST state and $\boldsymbol{b}$
is a **target-side buffer** — a prefix of the target $\boldsymbol{y}$ that
has been produced so far. The NFA is formally defined by:

- **States:** $S \times \text{Prefixes}(\boldsymbol{y})$, where $S$ is the FST
  state set.
- **Start states:** $\{(i, \varepsilon) \mid i \in I_T\}$ where $I_T$ is the
  FST's start set.
- **Final states:** $\{(i, \boldsymbol{y}) \mid i \in F_T\}$ where $F_T$ is
  the FST's final set.
- **Transitions from $(i, \boldsymbol{b})$ with $|\boldsymbol{b}| < |\boldsymbol{y}|$:**
  For each FST arc $i \xrightarrow{x:o} j$:
  - If $o = \varepsilon$ (output epsilon): emit arc $(i, \boldsymbol{b}) \xrightarrow{x} (j, \boldsymbol{b})$.
  - If $o = y_{|\boldsymbol{b}|+1}$ (matches next target symbol): emit $(i, \boldsymbol{b}) \xrightarrow{x} (j, \boldsymbol{b} \cdot o)$.
  - Otherwise: dead end (no arc).
- **Transitions from $(i, \boldsymbol{b})$ with $|\boldsymbol{b}| = |\boldsymbol{y}|$:**
  (Buffer has consumed the full target.) For each FST arc $i \xrightarrow{x:o} j$
  with *any* output $o$: emit $(i, \boldsymbol{y}) \xrightarrow{x} (j, \boldsymbol{y})$.

The language of this NFA is exactly
$\text{proj}_\mathcal{X}(T \circ \boldsymbol{y}\mathcal{Y}^*)$.

### 2.2 Powerset Determinization

The Precover NFA is determinized on-the-fly via the **powerset construction**.
Each DFA state is a set (frozenset) of NFA states:
$d = \{(i_1, \boldsymbol{b}_1), (i_2, \boldsymbol{b}_2), \ldots\}$.

A DFA state $d$ is **final** if any NFA element is final:
$\exists (i, \boldsymbol{b}) \in d : i \in F_T \wedge \boldsymbol{b} = \boldsymbol{y}$.

A final DFA state $d$ is **universal** if
$L(\text{DFA rooted at } d) = \mathcal{X}^*$ — every continuation of the
source string still produces output beginning with $\boldsymbol{y}$.

### 2.3 State Classification

Each DFA state is classified into one of four categories:

| Status | Meaning |
|--------|---------|
| `INTERIOR` | Non-final, or final-but-universality-not-yet-checked. Has outgoing arcs. |
| `QSTOP` | Final and universal. Quotient stop state. No outgoing arcs (the BFS prunes here). |
| `RSTOP` | Final but not universal. Remainder stop state. Has outgoing arcs (expansion continues). |
| `NEW` | Not yet expanded. Placeholder before BFS visit. |

The quotient FSA $Q$ uses `QSTOP` states as its accepting states; the
remainder FSA $R$ uses `RSTOP` states. Both share the same transition
structure (the DFA arcs), with pruning applied during materialization.

## 3. Non-Incremental Decomposition

The baseline algorithm performs a full BFS over the powerset DFA for each
target:

```
function DECOMPOSE(T, y):
    nfa ← PrecoverNFA(T, y)
    arena ← new PowersetArena
    start ← arena.intern(eps_closure(nfa.start()))
    worklist ← {start}
    while worklist not empty:
        d ← worklist.pop()
        if d is final:
            if IS_UNIVERSAL(d, nfa, arena):
                classify d as QSTOP; continue
            else:
                classify d as RSTOP
        for each input symbol x:
            d' ← arena.intern(eps_closure(nfa.step(d, x)))
            add arc d →x d'
            if d' is new: worklist.push(d')
        classify d as INTERIOR
    return (Q from QSTOP, R from RSTOP)
```

The **PowersetArena** is a hash-consing structure that maps sorted
`Vec<u64>` (packed NFA state sets) to `u32` IDs, avoiding duplicate
DFA states. It includes a fast path for single-element sets (the common
case in BPE tokenization, where 99%+ of DFA states contain exactly one NFA
state).

The universality check itself uses a **sub-BFS**: starting from state $d$,
verify that every reachable state is final and has arcs for all input symbols.
This is wrapped in a multi-stage **UniversalityFilter** that short-circuits
the expensive sub-BFS:

1. **Witness check:** If any NFA element $(i, \boldsymbol{y})$ is at an
   *input-universal* FST state $i$ (a state where all input symbols lead back
   to states that can produce any output), the state is immediately universal.
2. **Superset monotonicity:** If a previously-certified universal set $U$ is a
   subset of the current set ($U \subseteq d$), then $d$ is universal.
3. **Subset monotonicity:** If the current set $d$ is a subset of a known
   non-universal set $N$ ($d \subseteq N$), then $d$ is non-universal.
4. **Sub-BFS fallback:** Full exploration. Result is cached in the positive or
   negative index for future lookups.

The positive/negative caches use **element-indexed hit counting** for
efficient subset/superset testing: for each element $e \in d$, increment a
counter for each cache entry containing $e$; if any entry's counter reaches
its stored size, a subset (or superset) relationship has been found.

## 4. The Dirty-State Algorithm

### 4.1 Key Insight

When the target extends from $\boldsymbol{y}$ to $\boldsymbol{y} \cdot
y_{n+1}$, most of the DFA is unchanged. The Precover NFA's transitions
only change for states whose buffer position $|\boldsymbol{b}|$ is at the
**frontier** — the boundary where the old target ended. States with
$|\boldsymbol{b}| < |\boldsymbol{y}|$ have arcs that depend only on earlier
target symbols, which are unchanged.

A DFA state $d$ is **dirty** if it contains any NFA element at the frontier:
$\exists (i, \boldsymbol{b}) \in d : |\boldsymbol{b}| \geq |\boldsymbol{y}|$.
A state is **border** if it is clean (not dirty) but has an outgoing arc to a
dirty state. Border states must be re-expanded because their successor sets
may change.

All other states are **clean** and can reuse their cached arcs, status, and
classifications from the previous step.

### 4.2 Persistent DFA Structure

The algorithm persists the entire DFA structure across calls:

- **PowersetArena:** Hash-consing of NFA state sets → `u32` IDs. Append-only;
  states created during prior decompositions remain valid.
- **`arcs_from[sid]`:** Per-state outgoing arcs as `Vec<(label, dest_id)>`.
- **`state_status[sid]`:** Classification (`INTERIOR`, `QSTOP`, `RSTOP`, `NEW`).
- **`reverse_arcs[dst]`:** Reverse arc index mapping each state to its
  predecessors. Enables O(|dirty| × |in_degree|) border detection.
- **`max_bufpos[sid]`:** Maximum buffer position in the NFA set for state
  `sid`. Enables O(1) dirty detection instead of scanning the full NFA set.
- **`eps_cache`:** Epsilon closure cache mapping individual NFA states to their
  closure sets (as `Rc<Vec<u64>>` to avoid cloning). Persisted across calls
  with selective eviction.
- **`fst_univ_cache`:** Maps sorted FST state sets (without buffer
  information) to universality results. For DFA states where *all* NFA
  elements are at the frontier (pure frontier states), the universality
  sub-BFS explores only frontier states, making the result independent of the
  target content. This cache never needs eviction.
- **`reachable`:** Forward-reachable state IDs from the last BFS, used for
  dirty-state scanning and arc materialization.

### 4.3 Stable NFA State Packing

A critical design requirement is **stable state identity across target
extensions**. In the non-incremental setting, NFA states $(i,
|\boldsymbol{b}|)$ are packed as:

$$\text{packed} = i \times (|\boldsymbol{y}| + 1) + |\boldsymbol{b}|$$

This packing changes when $|\boldsymbol{y}|$ changes, invalidating the
PowersetArena's hash-consing (the same logical NFA state gets a different
`u64` key).

The dirty-state algorithm uses a **fixed stride**:

$$\text{packed} = i \times S + |\boldsymbol{b}|$$

where $S$ is chosen large enough for any future target length (e.g., $S =
4096$). This ensures that the packed representation of state $(i, k)$ is the
same regardless of the current target length, allowing PowersetArena entries
from previous calls to remain valid.

### 4.4 Incremental Update Algorithm

```
function DECOMPOSE_DIRTY(T, y_new):
    if y_new is not a prefix extension of y_prev:
        full reset; fall back to non-incremental
        return

    frontier ← |y_prev|

    // Phase 1: Identify dirty states — O(|reachable|)
    dirty ← {sid ∈ reachable : max_bufpos[sid] ≥ frontier}

    // Phase 2: Identify border states — O(|dirty| × avg_in_degree)
    border ← {}
    for each sid in dirty:
        for each src in reverse_arcs[sid]:
            if src ∉ dirty and status[src] ≠ NEW:
                border ← border ∪ {src}

    // Phase 3: Reset dirty ∪ border
    for each sid in dirty ∪ border:
        remove sid from reverse_arcs of its destinations
        status[sid] ← NEW
        arcs_from[sid] ← []

    // Phase 4: Evict stale caches
    evict eps_cache entries with buf_pos ≥ frontier
    evict universality filter entries touching frontier states

    // Phase 5: Local BFS from dirty ∪ border ∪ {start}
    nfa ← PrecoverNFA(T, y_new, stride=S, eps_cache)
    worklist ← dirty ∪ border
    if status[start] = NEW: worklist ← worklist ∪ {start}
    // ... BFS identical to non-incremental, but only visits
    // dirty/border states and their NEW successors.
    // Clean states reached as successors are not re-expanded.

    // Phase 6: Save state
    eps_cache ← nfa.take_eps_cache()
    prev_target ← y_new
```

The BFS in Phase 5 is identical to the non-incremental BFS except that it
starts from the dirty/border seed set rather than the full DFA. When the
BFS encounters a successor whose status is not `NEW` (i.e., a clean state
from the previous decomposition), it does not re-expand it — the cached
arcs and classification are still valid.

### 4.5 Cache Eviction

Three caches require selective eviction on prefix extension:

1. **Epsilon closure cache:** An entry $(s, \text{closure})$ is stale if the
   NFA state $s$ has $|\boldsymbol{b}| \geq \text{frontier}$ (its arcs
   changed), or if any state in the closure has $|\boldsymbol{b}| \geq
   \text{frontier}$ (the closure result changed). The `max_buf_pos` stored
   alongside each entry enables the second check without scanning the full
   closure vector.

2. **Universality filter:** Positive and negative cache entries are
   element-indexed. Entries containing any NFA element at or beyond the
   frontier are removed. Witnesses are rebuilt for the new target length.

3. **`fst_univ_cache`:** *Never evicted.* For pure frontier DFA states (all
   NFA elements at $|\boldsymbol{b}| = |\boldsymbol{y}|$), the universality
   sub-BFS explores only states at the frontier. These states' arcs do not
   depend on target content (they have consumed the full target and accept any
   further output). The universality result depends only on the set of FST
   states, not the target. The cache maps `sorted(fst_states) → bool`.

### 4.6 Lazy Materialization and Backward-BFS Trimming

The `decompose_dirty` call performs the BFS and state classification but does
**not** construct the quotient and remainder FSAs. Arc materialization is
deferred to `materialize_quotient()` and `materialize_remainder()`, which are
called only when the FSAs are actually needed.

Materialization proceeds in two passes:

1. **Forward BFS** from the start state through non-`QSTOP` arcs, collecting
   reachable states and classifying them as `QSTOP` or `RSTOP`.

2. **Backward BFS (trimming):** Starting from the relevant stop set
   (`QSTOP` for $Q$, `RSTOP` for $R$), traverse `reverse_arcs` backward to
   find states that can actually reach a stop state. Only arcs where both
   source and destination are backward-reachable are included in the output
   FSA. This eliminates dead-end branches that are forward-reachable but lead
   nowhere useful, reducing the size of the output FSAs.

## 5. Per-Symbol Branching with Overlays

### 5.1 Motivation

In autoregressive decoding, the `decompose_next()` operation must produce
$Q(\boldsymbol{y} \cdot \gamma)$ and $R(\boldsymbol{y} \cdot \gamma)$ for
*every* $\gamma \in \mathcal{Y}$ simultaneously. A naive approach calls
`decompose_dirty` independently for each symbol, repeating the dirty/border
identification and paying the full BFS cost $|\mathcal{Y}|$ times.

### 5.2 Overlay Architecture

The `decompose_next_all` method shares the dirty/border identification
across all symbols and uses lightweight **overlays** for per-symbol branching:

```
function DECOMPOSE_NEXT_ALL(T, y, 𝒴):
    // Shared Phase: Identify dirty ∪ border (once for all symbols)
    dirty_border ← identify_dirty_border(reachable, max_bufpos, reverse_arcs)
    base_q_stops ← {sid ∈ reachable : status[sid] = QSTOP, sid ∉ dirty_border}
    base_r_stops ← {sid ∈ reachable : status[sid] = RSTOP, sid ∉ dirty_border}
    evicted_eps_cache ← clone_and_evict(eps_cache)

    results ← {}
    for each γ ∈ 𝒴:
        // Per-symbol overlay: small dicts, not full DFA copies
        overlay_arcs ← {}      // sid → [(label, dest), ...]
        overlay_status ← {}    // sid → status
        overlay_reverse_add ← {}    // dst → [src, ...]
        overlay_reverse_remove ← {} // dst → [src, ...]

        nfa ← PrecoverNFA(T, y·γ, stride=S, evicted_eps_cache.clone())
        start ← arena.intern(eps_closure(nfa.start()))

        // Mark dirty_border as NEW in overlay
        for sid in dirty_border:
            overlay_status[sid] ← NEW
            record reverse_arcs removals in overlay_reverse_remove

        // BFS (reads base arcs for clean states, writes to overlay)
        worklist ← dirty_border ∪ {start if NEW}
        while worklist not empty:
            d ← worklist.pop()
            status ← overlay_status[d] ?? base_status[d]
            if status ≠ NEW: continue

            // NFA-based finality (not arena.is_final, which may be stale)
            is_final ← any(nfa.is_final(s) for s in arena.sets[d])

            if is_final and IS_UNIVERSAL(d):
                overlay_status[d] ← QSTOP; q_stops ← q_stops ∪ {d}
            else:
                compute arcs → store in overlay_arcs[d]
                overlay_status[d] ← RSTOP if is_final else INTERIOR

        // Materialization with overlay view
        Q ← collect_arcs_overlay_trimmed(start, q_stops, overlay)
        R ← collect_arcs_overlay_trimmed(start, r_stops, overlay)
        results[γ] ← (Q, R)

    return results
```

Each overlay is a set of small `HashMap`s (typically containing only the
dirty/border states plus a few newly discovered states). Clean states are
read directly from the base DFA's `arcs_from` and `state_status` arrays.
The arena is shared and append-only (new states interned during per-symbol
BFS persist, but this is harmless since `intern` is idempotent).

### 5.3 Stale Finality

A subtlety arises from the PowersetArena's hash-consing. When the same NFA
state set is interned for different target lengths, the `is_final` flag
reflects whichever target length was seen *first*. For example, the start
state $\{(i_0, 0)\}$ is final when $|\boldsymbol{y}| = 0$ (buffer position
equals target length) but not when $|\boldsymbol{y}| > 0$.

The overlay BFS addresses this by computing finality from the NFA rather
than reading the arena's `is_final`:

```rust
let is_final_nfa = arena.sets[sid].iter().any(|&s| nfa.is_final(s));
```

This ensures correct classification even when the arena contains stale
finality flags from a previous target. The base `decompose_dirty` avoids
this issue by explicitly resetting `arena.is_final` for dirty states before
re-expansion.

### 5.4 Overlay Materialization

The `collect_arcs_overlay_trimmed` method performs the same forward-BFS +
backward-BFS trimming as the base materialization, but reads from a merged
view:

- **Arcs:** `overlay_arcs[sid]` if present, else `arcs_from[sid]`.
- **Status:** `overlay_status[sid]` if present, else `state_status[sid]`.
- **Reverse arcs (for backward BFS):** `base_reverse_arcs[dst] - overlay_reverse_remove[dst] + overlay_reverse_add[dst]`.

This avoids copying the full DFA for each symbol.

## 6. Complexity Analysis

Let $n = |\text{reachable DFA states}|$, $D = |\text{dirty states}|$,
$B = |\text{border states}|$, $k = |\mathcal{X}|$ (input alphabet), and
$m = |\mathcal{Y}|$ (output alphabet).

### 6.1 Non-Incremental Decomposition

- **BFS:** $O(n \cdot k)$ arc computations (each state expanded once).
- **Intern:** $O(n \cdot k)$ hash lookups in the PowersetArena.
- **Universality:** Each final state triggers a sub-BFS; amortized by the
  UniversalityFilter's caches.

### 6.2 Incremental Update (`decompose_dirty`)

- **Dirty identification:** $O(n)$ scan of `max_bufpos`.
- **Border identification:** $O(D \cdot \bar{d}_{\text{in}})$ where
  $\bar{d}_{\text{in}}$ is average in-degree.
- **Reset:** $O((D + B) \cdot \bar{d}_{\text{out}})$ for reverse-arc cleanup.
- **BFS:** $O((D + B + N_{\text{new}}) \cdot k)$ where $N_{\text{new}}$ is
  newly discovered states. Clean states are never re-expanded.
- **Cache eviction:** $O(|\text{eps\_cache}|)$ scan for stale entries.

The savings are proportional to $1 - (D + B)/n$: the fraction of clean
states that are skipped. In BPE tokenization, the DFA typically has
thousands of states but only a small fraction touch the frontier, yielding
significant speedups at longer target lengths.

### 6.3 Per-Symbol Branching (`decompose_next_all`)

- **Shared dirty/border:** $O(n + D \cdot \bar{d}_{\text{in}})$, paid once.
- **Per-symbol BFS:** $O((D + B + N_{\text{new}}^{(\gamma)}) \cdot k)$ for each $\gamma$.
- **Per-symbol materialization:** Forward + backward BFS, $O(n_\gamma)$ per symbol.
- **Total:** $O(n + m \cdot (D + B) \cdot k)$, vs. $O(m \cdot n \cdot k)$ for
  independent decompositions.

## 7. Implementation Details

### 7.1 NFA State Packing

NFA states $(i, |\boldsymbol{b}|)$ are packed into `u64` as:

```
packed = fst_state × stride + buf_pos
```

where `stride` is a fixed constant (e.g., 4096). The `u64` packing enables
efficient hashing and comparison in the PowersetArena. Unpacking extracts
`fst_state = packed / stride` and `buf_pos = packed % stride`.

### 7.2 Epsilon Closure

For FSTs with epsilon-output arcs, the NFA has epsilon transitions. The
epsilon closure of each individual NFA state is computed by BFS over
epsilon arcs, filtered to retain only **productive** states — those with at
least one non-epsilon input arc or that are NFA-final. This filtering
collapses transit-only epsilon chains, dramatically reducing powerset state
sizes for BPE-like FSTs where long epsilon chains are common.

Closures are cached as `Rc<Vec<u64>>` to avoid cloning when multiple
powerset states share the same individual closure. The cache is persisted
across incremental calls with selective eviction of entries touching the
frontier.

### 7.3 Python Implementation

`TruncatedIncrementalDFADecomp` (in `dfa_decomp_incremental_truncated.py`)
mirrors the Rust `DirtyDecomp`. DFA states are Python `frozenset` objects
containing `(fst_state, buffer_string)` tuples. Key differences from the
Rust version:

- Uses Python `dict` and `set` for `_dfa_trans`, `_incoming`, `_dfa_status`.
- The `__rshift__` operator transfers ownership of parent dicts via reference
  (O(1)) rather than copying.
- `decompose_next()` creates `_OverlayChild` objects holding `_overlay_trans`,
  `_overlay_status`, and incoming diff sets, reading clean arcs from the
  parent's dicts.
- Overlay children flatten to full `TruncatedIncrementalDFADecomp` objects on
  demand — calling `>>` or `decompose_next()` on an overlay child triggers
  `_flatten()`.

### 7.4 Rust Implementation

`DirtyDecomp` (in `incremental.rs`) stores all persistent state in flat `Vec`
arrays indexed by `u32` state IDs. The `PowersetArena` (in `powerset.rs`) uses
`FxHashMap` for intern lookups, with a single-element fast path that hashes a
`u64` directly instead of a `Vec` — this covers 99% of BPE DFA states.

The Python-Rust bridge (`rust_bridge.py`) converts Python `FST` objects to
Rust `RustFst` via integer-mapped symbols and state IDs, delegates to
`DirtyDecomp.decompose_dirty()` and `decompose_next_all()` via PyO3 bindings
(in `py.rs`), and converts the resulting `FsaResult` parallel arrays back to
Python `FSA` objects.

## 8. Incremental Peekaboo (`DirtyPeekaboo`)

### 8.1 Background: Peekaboo Decomposition

**Peekaboo** computes per-symbol quotient and remainder — for each next output
symbol $\gamma \in \mathcal{Y}$, it produces $Q(\boldsymbol{y} \cdot \gamma)$
and $R(\boldsymbol{y} \cdot \gamma)$ simultaneously. Instead of a single BFS
over a fixed-target Precover NFA, peekaboo runs a sequence of **steps**
$0, 1, \ldots, |\boldsymbol{y}|$, each using a `PeekabooNFAMapped`
parameterized by `step_n`:

- NFA states are $(i, \text{buf\_len}, \text{extra\_sym}, \text{truncated})$
  packed into `u64`.
- At step $k$, the NFA reads `target[0..k]` and allows one symbol beyond
  the target prefix (`extra_sym`), which becomes the next-symbol index.
- Each step resumes from the previous step's **resume frontiers** — states
  on the truncation boundary for each next symbol.
- The final step ($k = |\boldsymbol{y}|$) produces the per-symbol Q/R
  classification.

A non-incremental approach rebuilds everything from scratch: running all
$|\boldsymbol{y}| + 1$ steps and constructing the arena and merged incoming
arcs anew. This is wasteful in autoregressive decoding where the target grows
by one symbol at a time.

### 8.2 Key Insight

When the target extends from $\boldsymbol{y}[0..N]$ to $\boldsymbol{y}[0..N+1]$:

- Steps $0$ through $N$ produce **identical** NFA arcs because
  `PeekabooNFAMapped` for step $k$ only reads `target[0..k]`, and the
  prefix $\boldsymbol{y}[0..k]$ is unchanged for $k \leq N$.
- Only step $N+1$ is new.
- The arena, merged incoming arcs, and resume frontiers from steps
  $0..N$ are all still valid.

This gives **one new step per prefix extension** instead of $N+1$ steps
from scratch.

### 8.3 Comparison with `DirtyDecomp`

The incremental peekaboo case is simpler than `DirtyDecomp`:

| Aspect | `DirtyDecomp` | `DirtyPeekaboo` |
|--------|---------------|-----------------|
| NFA changes | Frontier states get new arcs | Entire NFA changes per step |
| Cache reuse | eps\_cache persisted with selective eviction | eps\_cache cleared per step (NFA changes) |
| Dirty detection | Scan `max_bufpos`, find border states | Not needed — steps are sequential |
| Work skipping | Skip clean states, re-expand dirty+border | Skip entire completed steps |
| Arena | Shared across all targets | Shared across all steps |
| Reverse arcs | Maintained for border detection | Not needed |

The simplicity comes from the step-sequential structure: there is no
interleaving of clean and dirty regions within a step. The dirty/border
machinery of `DirtyDecomp` is unnecessary — we simply skip steps that
have already been computed.

### 8.4 `DirtyPeekaboo` Struct

```rust
pub struct DirtyPeekaboo {
    // FST metadata (computed once in new())
    output_alphabet: Vec<u32>,
    sym_to_idx: FxHashMap<u32, u16>,
    ip_universal_states: Vec<bool>,
    num_source_symbols: usize,

    // Persistent BFS state (grows monotonically)
    arena: PowersetArena,
    merged_incoming: FxHashMap<u32, Vec<(u32, u32)>>,
    global_start_id: u32,

    // Incremental tracking
    prev_target: Vec<u32>,
    completed_steps: usize,
    decomp_q: FxHashMap<u16, Vec<u32>>,
    decomp_r: FxHashMap<u16, Vec<u32>>,
    resume_frontiers: FxHashMap<u16, FxHashSet<u32>>,

    // FST-level universality cache (target-independent)
    fst_univ_cache: FxHashMap<Vec<u32>, bool>,
}
```

### 8.5 Algorithm

```
function DIRTY_PEEKABOO_DECOMPOSE(fst, target):
    if target = prev_target:
        return cached results

    if target is prefix extension of prev_target:
        start_step ← completed_steps  // skip steps 0..N
    else:
        full_reset()
        start_step ← 0

    for step in start_step..=|target|:
        run_step(fst, target, step)
        // Each step uses its own fresh eps_cache (NFA changes per step)
        // but reuses the persistent arena and merged_incoming

    completed_steps ← |target| + 1
    prev_target ← target
    return extract_results()
```

Each `run_step` constructs a `PeekabooNFAMapped`,
resumes from the previous step's frontiers, runs BFS with universality
checking, and merges new incoming arcs into the global `merged_incoming`.

### 8.6 `fst_univ_cache` Integration

For **pure-frontier projected states** — DFA states where all NFA elements
have `buf_len == step_n + 1` and `extra_sym == y_idx` — the universality
result depends only on the FST state set, not the target content or step
number. This mirrors the `fst_univ_cache` in `DirtyDecomp` (§4.5):

```
projected = filter NFA set to elements matching y_idx
if all elements are at buf_len == step_n + 1:
    key = sorted(fst_states extracted from projected)
    if key in fst_univ_cache:
        return cached result
    // ... fall through to full universality check ...
    fst_univ_cache[key] = result
```

This cache is never evicted since the result is target-independent.

### 8.7 PTB Benchmark Results

On the PTB tokenizer FST (296 states, 23,723 arcs, 256 output symbols),
processing the first 200 symbols of a WikiText test paragraph:

```
pos=  0   366.87ms  steps=1  arena=2076    (cold start)
pos=  1    11.75ms  steps=1  arena=2540
pos=  5    10.49ms  steps=1  arena=4344
pos=  7   151.40ms  steps=1  arena=6820    (large arena growth)
pos= 20    15.66ms  steps=1  arena=15837
pos= 40    12.44ms  steps=1  arena=29162
pos= 60    13.43ms  steps=1  arena=42918
pos= 80     0.11ms  steps=1  arena=44722   (arena stabilized)
pos=100     0.12ms  steps=1  arena=44722
pos=160     0.11ms  steps=1  arena=44722
```

Key observations:

- **`steps=1`** on every incremental call confirms only the new step runs.
- **Arena stabilization:** The arena converges to ~44,700 states by position
  ~70, after which new steps are essentially free (0.1ms) because all
  reachable powerset states have already been interned.
- **Cold start cost:** The first call (pos=0) runs step 0 from scratch,
  building the initial arena (~367ms). Subsequent steps are 10-15ms while
  the arena is growing, then drop to 0.1ms once stable.
- **Aggregate:** 200 symbols processed in 1.8 seconds total (111 sym/sec),
  with median per-step cost of 0.12ms.

For comparison, a from-scratch approach at position $N$ would run
$N+1$ steps, each costing roughly the same as one incremental step. At
position 100, this means ~100× more work than the incremental variant.

On smaller test FSTs (triplets\_of\_doom, number\_comma\_separator), the
incremental variant achieves 1.5-1.8× speedup in tree traversal benchmarks,
with the advantage growing at deeper recursion depths.
