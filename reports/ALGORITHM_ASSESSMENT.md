# Algorithm Assessment: Computing Next-Symbol Probabilities of a Transduced Language Model

## The Core Problem

Given an FST `f` and a target prefix **y**, compute for each next symbol `z`:

$$P(z \mid \mathbf{y}) \propto \sum_{x \in \mathcal{Q}(\mathbf{y}z)} P_{\text{LM}}(x) + \sum_{x \in \mathcal{R}(\mathbf{y}z)} P_{\text{LM}}(x \cdot \text{EOS})$$

where $\mathcal{P}(\mathbf{y}) = \mathcal{Q}(\mathbf{y})\mathcal{X}^* \sqcup \mathcal{R}(\mathbf{y})$ is the precover decomposition.

The precover $\mathcal{P}(\mathbf{y})$ is the set of all source strings whose transduction
through `f` begins with **y**. The quotient $\mathcal{Q}$ captures source strings that
can continue producing more output; the remainder $\mathcal{R}$ captures those that
have terminated. Both are represented as finite automata.

---

## Current Implementations (2026-02-13)

### General-Case Decomposition Algorithms

These algorithms use target-buffer truncation to guarantee termination on all FSTs,
including those with infinite quotient/remainder languages.

| Algorithm | Language | File | Incremental | Notes |
|-----------|----------|------|:-----------:|-------|
| `Precover` | Python | `eager_nonrecursive.py` | No | Reference implementation |
| `NonrecursiveDFADecomp` | Python | `dfa_decomp_nonrecursive.py` | No | Same algorithm, cleaner interface |
| `TruncatedIncrementalDFADecomp` | Python | `dfa_decomp_incremental_truncated.py` | Yes | Dirty-state incremental DFA decomp |
| `PeekabooState` | Python | `peekaboo_incremental.py` | Yes | **Recommended.** Batched next-symbol |
| `Peekaboo` | Python | `peekaboo_nonrecursive.py` | No | Non-incremental peekaboo |
| `DirtyPeekaboo` | Python | `peekaboo_dirty.py` | Yes | Dirty-state incremental peekaboo |
| `TokenDecompose` | Python | `token_decompose.py` | No | BPE fast path (`all_input_universal` FSTs only) |
| `RustDecomp` | Rust | `rust_bridge.py` → `decompose.rs` | No | 3-10x faster than Python |
| `RustDirtyState` | Rust | `rust_bridge.py` | Yes | Rust-backed dirty-state incremental |
| `RustDirtyPeekaboo` | Rust | `rust_bridge.py` | Yes | Rust-backed dirty-state peekaboo |

### Finite-Only Decomposition Algorithms

These algorithms lack target-buffer truncation and may diverge on FSTs with infinite
quotients. Tested separately in `test_finite.py`.

| Algorithm | Language | File | Incremental | Notes |
|-----------|----------|------|:-----------:|-------|
| `LazyIncremental` | Python | `lazy_incremental.py` | Yes | Finite-language FSTs only |
| `LazyNonrecursive` | Python | `lazy_nonrecursive.py` | No | Finite-language FSTs only |
| `PrioritizedLazyIncremental` | Python | `prioritized_lazy_incremental.py` | Yes | Heuristic-guided BFS; finite only |

### Inference Algorithms

| Algorithm | File | Approach |
|-----------|------|----------|
| `prioritized_enumeration` | `enumeration.py` | Best-first search weighted by LM log-probs |
| `importance_sampling` | `enumeration.py` | Sample paths, accumulate partition function |
| `crude_importance_sampling` | `enumeration.py` | Same, without Q/R decomposition |

### LM Integration

| Class | File | Description |
|-------|------|-------------|
| `LMState` | `lm/base.py` | ABC: `logp_next`, `eos`, `>>`, `__call__`, `greedy_decode`, `sample_decode` |
| `ByteNgramLM` / `CharNgramLM` | `lm/ngram.py` | Lightweight n-gram LMs for testing |
| `StateLM` | `lm/statelm.py` | Incremental LM state with KV-cache |
| `TokenizedLLM` | `lm/statelm.py` | Wraps HuggingFace causal LMs |
| `load_model_by_name` | `lm/statelm.py` | Load `'gpt2'`, `'meta-llama/...'`, etc. |
| `TransducedLM` | `lm/transduced.py` | Pushforward of an inner LM through an FST |

Self-contained (no external tokenization deps). Example:
```python
from transduction.lm import StateLM
from transduction.enumeration import prioritized_enumeration
lm = StateLM.initial('gpt2')
pe = prioritized_enumeration(lm, fst, target, max_steps=20)
```

---

## Recommended Algorithms

### For Autoregressive Decoding

Use **`RustDirtyPeekaboo`** or **`RustDirtyState`** (or Python `PeekabooState` if Rust unavailable).

Peekaboo builds a *single* DFA for all next-symbol extensions and extracts all $|\mathcal{Y}|$
decompositions from it. This amortizes DFA construction across all next symbols — an
asymptotic win when the target alphabet is large (e.g., 256 bytes).

The dirty-state variants (`RustDirtyPeekaboo`, `DirtyPeekaboo`) additionally persist
DFA state across decoding steps, avoiding redundant recomputation as the target grows.

```python
from transduction.rust_bridge import RustDirtyPeekaboo
from transduction import examples

fst = examples.newspeak2()
peekaboo = RustDirtyPeekaboo(fst)
decomps = peekaboo.decompose_next()  # Q/R for all next symbols
```

The Python incremental variant (`peekaboo_incremental.PeekabooState`) also supports incremental
computation via the `>>` operator for step-by-step decoding.

### For BPE Tokenizers

Check `check_all_input_universal(fst)` first. If true, use **`TokenDecompose`**:

```python
from transduction.universality import check_all_input_universal
from transduction.token_decompose import TokenDecompose

if check_all_input_universal(fst):
    decomp = TokenDecompose(fst, target)  # 5000x+ faster
else:
    decomp = NonrecursiveDFADecomp(fst, target)
```

### For One-Shot Decomposition

Use **`RustDecomp`** or **`NonrecursiveDFADecomp`**.

---

## General vs Finite-Only: The Truncation Distinction

The key mechanism that separates general-case algorithms from finite-only ones is
**target-buffer truncation**. Algorithms that truncate the target buffer
(`NonrecursiveDFADecomp`, `TruncatedIncrementalDFADecomp`, Peekaboo variants, Rust
backends) terminate on all inputs. Those that don't (`LazyIncremental`)
may diverge on FSTs with infinite quotients.

When adding new algorithms or test cases, classify them as general vs finite-only
and put them in the appropriate test file (`test_general.py` vs `test_finite.py`).

---

## Performance Summary

### Rust Peekaboo vs Python

| Example | Rust | Python | Speedup |
|---------|------|--------|---------|
| newspeak2 (depth=3) | 3.0 ms | 67.0 ms | 22x |
| triplets_of_doom (depth=13) | 27 µs | 278 µs | 10x |
| parity (depth=5) | 13 µs | 318 µs | 25x |

### TokenDecompose (BPE FSTs)

| target_len | Generic | TokenDecompose | Speedup |
|-----------|---------|----------------|---------|
| 50 | 1502 ms | 0.3 ms | ~5000x |
| 1000 | impossible | 2.4 ms | — |

---

## Key Optimizations Applied

1. **`all_input_universal` precomputation** — O(|arcs|) check; skips universality BFS entirely for BPE/replace FSTs

2. **Token-level position tracking** — For hub-structured FSTs, NFA states are just positions `{0..N}` instead of `(fst_state, position)`. Collapses 7000 intermediate states per token to 1.

3. **Peekaboo batching** — Build one DFA for all next symbols, not $|\mathcal{Y}|$ separate decompositions

4. **Dirty-state persistence** — Incremental algorithms (`TruncatedIncrementalDFADecomp`, `DirtyPeekaboo`, Rust dirty variants) persist DFA state across decoding steps, avoiding redundant recomputation as the target prefix grows.

5. **Rust acceleration** — Packed `u64` NFA states, interned `u32` DFA states, `Rc<Vec>` eps cache, single-element intern fast path

6. **UniversalityFilter cascade** — AUI fast path → witness check → monotonicity caches → BFS fallback

---

## Test Status

- **`test_general.py`**: 148 passed, 14 skipped
- **`test_finite.py`**: 40 passed
- **`test_enumeration.py`**: 12 passed (including BPE-scale GPT-2 integration)
- **`test_push_labels.py`**: 30 passed
- **`test_transduced.py`**: 23 passed

---

## Dependencies

**Library:** `numpy`, `torch`, `transformers`, `arsenal`

**Test-only:** `genparse`

**Eliminated:** `genlm`, `tokenization` (inlined into `transduction/lm/statelm.py`)
