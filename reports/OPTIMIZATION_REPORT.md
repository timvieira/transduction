# Optimization Report: Decomposition Algorithms

This report documents the optimizations employed across Python and Rust implementations
of the precover decomposition algorithm.

## Current State (2026-02-13)

### Implementations

| Algorithm | Language | File | Use Case |
|-----------|----------|------|----------|
| `Precover` | Python | `eager_nonrecursive.py` | Reference implementation |
| `NonrecursiveDFADecomp` | Python | `dfa_decomp_nonrecursive.py` | Single-target decomposition |
| `TruncatedIncrementalDFADecomp` | Python | `dfa_decomp_incremental_truncated.py` | Dirty-state incremental decomposition |
| `PeekabooState` | Python | `peekaboo_incremental.py` | Batched next-symbol (recommended) |
| `DirtyPeekaboo` | Python | `peekaboo_dirty.py` | Dirty-state incremental peekaboo |
| `TokenDecompose` | Python | `token_decompose.py` | BPE fast path (5000x+ speedup) |
| `RustDecomp` | Rust | `decompose.rs` | Single-target (3-10x faster) |
| `RustDirtyState` | Rust | `rust_bridge.py` | Dirty-state incremental (Rust-backed) |
| `RustDirtyPeekaboo` | Rust | `rust_bridge.py` | Dirty-state incremental peekaboo (Rust-backed) |

### Test Status

- `test_general.py`: 148 passed, 14 skipped
- `test_finite.py`: 40 passed
- `test_enumeration.py`: 12 passed (including BPE-scale GPT-2 integration)

### Dependencies

Library: `numpy`, `torch`, `transformers`, `arsenal`
Test-only: `genparse`
Eliminated: `genlm`, `tokenization` (inlined into `transduction/lm/statelm.py`)

---

## 1. Algorithm Correspondence

The Rust `decompose.rs` is closest to Python `NonrecursiveDFADecomp`:

```
1. Build precover NFA: LazyPrecoverNFA(fst, target)
2. On-the-fly determinize (powerset construction)
3. BFS over DFA states, check universality of finals
4. Universal finals → Q stops (don't expand)
   Non-universal finals → R stops (keep expanding)
5. Return Q and R sharing the same arc set
```

The Python pipeline chains lazy wrappers:
```
LazyDeterminize.arcs(frozenset)
  → for each NFA state in frozenset:
      EpsilonRemove.arcs(state)
        → LazyPrecoverNFA.arcs(state)
        → for each dest: eps_closure(dest)
      group by input symbol → frozenset per symbol
```

Rust fuses this into a single BFS loop with packed `u64` NFA states and interned `u32` DFA states.

---

## 2. Optimizations Catalog

### A: Fused State Representation

| | Python | Rust |
|---|---|---|
| NFA state | `(fst_state, position)` tuple | `u64` packed as `fst_state * (N+1) + pos` |
| Status | Done (`LazyPrecoverNFA` uses `(i, n)`) | Done |

### B: Precomputed FST Indexes

Both Python and Rust build 4 index dictionaries:
- `index_iy_xj`: `(state, output)` → `{(input, dest)}`
- `index_i_xj`: `state` → `{(input, dest)}`
- `index_ix_j`: `(state, input)` → `{dest}`
- `index_ixy_j`: `(state, input, output)` → `{dest}`

| | Python | Rust |
|---|---|---|
| Location | `LazyPrecoverNFA.__init__` | `Fst::new()` |
| Hasher | Python built-in | FxHashMap |
| Status | Done | Done |

### C: Arena-Interned Powerset States

| | Python | Rust |
|---|---|---|
| DFA state repr | `frozenset` | `u32` ID via `PowersetArena` |
| Single-element fast path | No | Yes (hashes `u64` not `Vec`) |
| Status | Could add `Integerizer` | Done |

### D: Cached Epsilon Closures

| | Python | Rust |
|---|---|---|
| Cache | `EpsilonRemove._closure_cache` | `Rc<Vec<u64>>` in `eps_cache` |
| Status | Done | Done |

### E: Batch Arc Computation

Both iterate arcs from states (not symbols in alphabet):
```python
# O(active arcs), not O(|alphabet|)
for i in powerset_state:
    for a, j in nfa.arcs(i):
        tmp[a].add(j)
```

| | Python | Rust |
|---|---|---|
| Location | `LazyDeterminize.arcs()` | `compute_all_arcs()` |
| Status | Done | Done |

### F: `all_input_universal` Precomputation ✅

**The single highest-impact optimization for BPE/replace FSTs.**

Instead of O(N²) universality sub-BFS per final state, do one O(|arcs|) check upfront:
1. ε-close start states (input-side only)
2. Check start is final and complete (has arcs for all symbols)
3. Check every successor's ε-closure contains start

When true, all final states are universal → skip `accepts_universal` entirely.

| | Python | Rust |
|---|---|---|
| Location | `check_all_input_universal()` in `fst.py` | `check_all_input_universal()` in `fst.rs` |
| Status | Done | Done |

**Impact:** 500-token vocab: 58ms → <1ms. 5000-token vocab: 11.5s → <1ms.

### G: Token-Level Position Tracking (BPE-specific) ✅

For hub-structured FSTs (BPE tokenizers), collapse FST states:

| Approach | NFA states | DFA states |
|----------|-----------|-----------|
| Generic | `(fst_state, position)` — O(7000×N) | O(6000) per byte |
| Token-level | `position` only — O(N) | O(N) total |

Algorithm:
1. Extract token byte sequences from FST
2. Build byte trie
3. NFA states are positions `{0..N}`
4. Match tokens via trie at each position

| | Python | Rust |
|---|---|---|
| Implementation | `TokenDecompose` in `token_decompose.py` | Removed (was `token_decompose.rs`) |
| Status | Done | N/A |

**Impact:** 5000x+ speedup for BPE FSTs at long target lengths.

### H: Dirty-State Persistence ✅

**Key optimization for autoregressive (incremental) decoding.**

When the target extends from **y** to **y**·z, most of the DFA is unchanged. The
dirty-state algorithm identifies which DFA states are affected by the extension
(dirty states at the frontier + border states with arcs into the dirty region)
and only re-expands those, reusing cached arcs and classifications for all clean
states.

| | Python | Rust |
|---|---|---|
| DFA decomp | `TruncatedIncrementalDFADecomp` in `dfa_decomp_incremental_truncated.py` | `RustDirtyState` in `rust_bridge.py` |
| Peekaboo | `DirtyPeekaboo` in `peekaboo_dirty.py` | `RustDirtyPeekaboo` in `rust_bridge.py` |
| Status | Done | Done |

**Impact:** After arena stabilization (~70 symbols on PTB), per-step cost drops
to ~0.1ms. See `reports/dirty_state_algorithm.md` for full details.

---

## 3. Rust-Specific Optimizations

### Peekaboo (batched next-symbol)

`RustDirtyPeekaboo` in `peekaboo.rs` implements the dirty-state incremental Peekaboo algorithm:
- Computes Q/R for all next symbols in one pass
- Persists DFA state across decoding steps, avoiding redundant recomputation
- Uses same PowersetArena and eps-caching as generic decompose

| Example | Rust | Python | Speedup |
|---------|------|--------|---------|
| newspeak2 (depth=3) | 3.0 ms | 67.0 ms | 22x |
| triplets_of_doom (depth=13) | 27 µs | 278 µs | 10x |
| parity (depth=5) | 13 µs | 318 µs | 25x |

### Single-Element Intern Fast Path

99% of BPE DFA states are single-element sets. `PowersetArena` hashes a `u64`
directly instead of a `Vec` for these, avoiding allocation and Vec hashing.

### Rc<Vec<u64>> for Epsilon Cache

Epsilon closure cache returns `Rc::clone()` on hit (refcount bump, no data copy).

---

## 4. Summary Table

| Optimization | Impact | Generality | Python | Rust |
|--------------|--------|------------|--------|------|
| F: `all_input_universal` | Critical | BPE + replace | ✅ | ✅ |
| H: Dirty-state persistence | Major | Incremental decoding | ✅ | ✅ |
| G: Token-level positions | Major | Hub FSTs | ✅ | Removed |
| C: Arena interning | Moderate | All FSTs | Partial | ✅ |
| B: FST indexes | Important | All FSTs | ✅ | ✅ |
| D: Eps closure cache | Important | NFAs with ε | ✅ | ✅ |
| A: Packed state repr | Minor | All FSTs | ✅ | ✅ |
| Peekaboo batching | Major | All FSTs | ✅ | ✅ |

---

## 5. Choosing an Implementation

| Use Case | Recommendation |
|----------|----------------|
| Autoregressive decoding | `RustDirtyPeekaboo` or `RustDirtyState` (Python: `PeekabooState` or `DirtyPeekaboo`) |
| BPE tokenizer, long targets | `TokenDecompose` (Python) — check `check_all_input_universal` first |
| One-shot decomposition | `RustDecomp` or `NonrecursiveDFADecomp` |
| Reference/testing | `Precover` |

---

## 6. LM Integration

The enumeration algorithms (`prioritized_enumeration`, `importance_sampling`) combine
decomposition with language model scoring:

```python
from transduction.lm import StateLM
from transduction.enumeration import prioritized_enumeration

lm = StateLM.initial('gpt2')
pe = prioritized_enumeration(lm, fst, target='the', max_steps=20)

for item in pe.quotient_terms:
    print(f"{item.source}: {item.weight:.3f}")
```

The `StateLM` class (in `transduction/lm/statelm.py`) wraps HuggingFace causal LMs
with KV-cache-based incremental decoding. All tokenization dependencies have been
inlined — no external `genlm` or `tokenization` packages required.
