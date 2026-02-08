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

## Current Implementations (2026-02-04)

### Decomposition Algorithms

| Algorithm | Language | File | Incremental | Notes |
|-----------|----------|------|:-----------:|-------|
| `Precover` | Python | `eager_nonrecursive.py` | No | Reference implementation |
| `NonrecursiveDFADecomp` | Python | `dfa_decomp_nonrecursive.py` | No | Same algorithm, cleaner interface |
| `RecursiveDFADecomp` | Python | `dfa_decomp_recursive.py` | Yes | Diverges on some inputs (xfail) |
| `Peekaboo` (recursive) | Python | `peekaboo_recursive.py` | Yes | **Recommended.** Batched next-symbol |
| `Peekaboo` (nonrecursive) | Python | `peekaboo_nonrecursive.py` | No | Simpler variant |
| `TokenDecompose` | Python | `token_decompose.py` | No | BPE fast path (5000x+ speedup) |
| `RustDecomp` | Rust | `decompose.rs` | No | 3-10x faster than Python |
| `RustPeekaboo` | Rust | `peekaboo.rs` | No | **Recommended.** 3-25x faster |

### Inference Algorithms

| Algorithm | File | Approach |
|-----------|------|----------|
| `prioritized_enumeration` | `enumeration.py` | Best-first search weighted by LM log-probs |
| `importance_sampling` | `enumeration.py` | Sample paths, accumulate partition function |
| `crude_importance_sampling` | `enumeration.py` | Same, without Q/R decomposition |

### LM Integration

| Class | File | Description |
|-------|------|-------------|
| `LMState` | `lm/base.py` | ABC: `logp_next`, `eos`, `<<`, `__call__`, `greedy_decode`, `sample_decode` |
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

Use **`RustPeekaboo`** (or Python `Peekaboo` if Rust unavailable).

Peekaboo builds a *single* DFA for all next-symbol extensions and extracts all $|\mathcal{Y}|$
decompositions from it. This amortizes DFA construction across all next symbols — an
asymptotic win when the target alphabet is large (e.g., 256 bytes).

```python
from transduction.rust_bridge import RustPeekaboo
from transduction import examples

fst = examples.newspeak2()
peekaboo = RustPeekaboo(fst)
decomps = peekaboo('ba')  # Q/R for all next symbols
```

The Python recursive variant (`peekaboo_recursive.Peekaboo`) also supports incremental
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

4. **Rust acceleration** — Packed `u64` NFA states, interned `u32` DFA states, `Rc<Vec>` eps cache, single-element intern fast path

5. **UniversalityFilter cascade** — AUI fast path → witness check → monotonicity caches → BFS fallback

---

## Test Status

- **102/103 tests pass** (1 expected xfail: `RecursiveDFADecomp` timeout)
- 7 implementations tested in `test_general.py`
- 12 enumeration tests including BPE-scale GPT-2 integration

---

## Dependencies

**Library:** `numpy`, `torch`, `transformers`, `arsenal`

**Test-only:** `genparse`

**Eliminated:** `genlm`, `tokenization` (inlined into `transduction/lm/statelm.py`)
