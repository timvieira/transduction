# Project Assessment: Delivering on the Transduced LM Mission

## 2026-02-15

---

## Executive Summary

The transduction library efficiently computes next-symbol probabilities for a
language model composed with a finite-state transducer — enabling constrained
and transformed autoregressive generation. The algorithmic core is strong:
Peekaboo decomposition, dirty-state persistence, BPE-specific fast paths, and
Rust acceleration form a layered optimization stack that achieves real-time
per-step costs on production-scale FSTs. The user-facing `TransducedLM`
implements particle-based approximate inference with two modes: beam-sum
(deterministic top-K, consistent as K -> inf) and SIR (sequential importance
resampling, unbiased prefix probability estimates). Quotient states provide
exact marginalization over infinitely many source continuations — the key
variance reduction mechanism. CI, exports, and core documentation are in place.
The remaining path to production readiness: wire in the BPE fast path
(TokenDecompose), batch LM calls for GPU utilization, and fix multi-character
symbol handling.

---

## What's Working Well

### 1. Algorithmic Depth

12 decomposition algorithms spanning the full design space: reference
implementations, incremental variants, batched variants, BPE-specific fast
paths, and Rust backends. Each variant serves a distinct niche:

| Layer | Algorithm | Purpose |
|-------|-----------|---------|
| Reference | `Precover`, `NonrecursiveDFADecomp` | Correctness oracle |
| Batched | `PeekabooState`, `Peekaboo` (nonrecursive) | Amortize DFA across all next symbols |
| Incremental | `DirtyPeekaboo`, `TruncatedIncrementalDFADecomp` | Reuse DFA state across decode steps |
| BPE fast path | `TokenDecompose` | Exploit hub structure (5000x speedup) |
| Rust | `RustDecomp`, `RustDirtyState`, `RustDirtyPeekaboo` | 3-25x over Python |
| Finite-only | `LazyIncremental`, `LazyNonrecursive`, `PrioritizedLazy` | Finite-language FSTs |

The parametrized test suite (`test_general.py`: 307 tests) ensures all
general-case algorithms agree.

### 2. Optimizations With Measured Impact

| Optimization | Impact | Mechanism |
|-------------|--------|-----------|
| `all_input_universal` | 11,500x on BPE | Skip universality BFS entirely |
| Token-level position tracking | 5000x on BPE | Collapse FST-state x position to just position |
| Peekaboo batching | ~\|Y\|x amortization | One DFA for all next symbols |
| Dirty-state persistence | Per-step -> 0.1ms | Reuse clean DFA states across steps |
| Rust backend | 3-25x | Packed u64 NFA states, interned DFA states |

### 3. Clean Internal Architecture

~11K lines of Python across 32 modules. No circular dependencies. Well-layered:

```
FST / FSA  (data structures)
   |
base.py  (DecompositionResult, abstract interfaces)
   |
algorithm implementations  (peekaboo, dirty, token_decompose, ...)
   |
lm/transduced.py  (TransducedLM — user-facing)
```

The Rust bridge is optional and degrades gracefully. The LM submodule (`lm/`)
is self-contained with no pollution of core algorithms.

### 4. Solid Test Infrastructure

- **663 tests** across 10 test files
- Parametrized cross-algorithm validation catches disagreements automatically
- Reference implementations (`Precover`) serve as correctness oracles
- Real-model integration tests (GPT-2 + BPE FST in `test_enumeration.py`)
- TransducedLM tests: multi-state FSTs, brute-force comparison, consistency
  convergence, SIR unbiasedness
- Clear general vs. finite-only test separation
- CI via GitHub Actions (`.github/workflows/test.yml`)

### 5. Elegant User-Facing API

The `TransducedLM` API mirrors the inner LM interface:

```python
tlm = TransducedLM(inner_lm, fst)
state = tlm >> 'h'          # advance by target symbol
p = state.logp_next['e']    # query next-symbol probability
```

Users don't need to understand precovers, peekaboo, or dirty states. The
`>>` operator and `logp_next` property are the entire surface area. Two
inference modes are supported: beam-sum (deterministic) and SIR (stochastic,
unbiased).

---

## Open Issues

### High: Multi-Character Symbol Handling Broken

All PrecoverNFA implementations index into the output buffer by character
position (`ys[N]`, `ys[:N+1]`). This clips multi-character symbols (e.g.,
PTB byte-value strings like '84', '104'). Blocks use cases beyond byte-level
FSTs.

### High: TokenDecompose Not Wired Into TransducedLM

`TransducedLM` does not automatically use the BPE fast path when
`check_all_input_universal(fst)` is true. The 5000x speedup requires manual
dispatch.

### High: No End-to-End Example

No standalone script showing: load LM, build FST, create TransducedLM,
decode a sentence.

### Medium: No Batched LM Inference

`TransducedLM` processes one sequence at a time. The LM state advance
(`lm_state >> x`) consumes 30-40% of `_compute_logp_next` time. Batching
multiple source-symbol expansions into a single forward pass would improve
GPU utilization for neural LMs.

### Low: K and max_expansions Must Scale Together

Beam-sum consistency requires both `K` (carry-forward budget) and
`max_expansions` (per-step expansion budget) to grow to infinity. The
defaults (K=100, max_expansions=1000) work for most FSTs, but
high-branching-factor FSTs may need a larger ratio. This coupling is not
documented in the user-facing API.

### Low: No Type Annotations on Public API

`base.py`, `fst.py`, `lm/base.py`, `lm/transduced.py` lack type annotations.

---

## Codebase Health Metrics

### Size and Complexity

| Component | Files | Lines | Notes |
|-----------|-------|-------|-------|
| Core (FST/FSA/base) | 3 | ~1,800 | Stable, well-tested |
| Algorithms | 9 | ~2,900 | Active development |
| LM integration | 6 | ~1,700 | Self-contained (includes FusedTransducedLM) |
| Rust backend | 9 | ~3,000 | Well-optimized |
| Applications | 3 | ~590 | BPE, PTB, WikiText |
| Utilities | 3 | ~2,400 | viz, examples, lazy |
| **Total Python** | **32** | **~11,200** | |
| Tests | 10 | ~3,350 | 663 tests |

### Technical Debt

| Item | Severity | Location | Effort |
|------|----------|----------|--------|
| Multi-char symbol handling | High | `precover_nfa.py` | Medium |
| No end-to-end example | High | — | Small |
| TokenDecompose not wired into TransducedLM | High | `lm/transduced.py` | Small |
| No batched LM calls | Medium | `lm/transduced.py` | Medium |
| K/max_expansions coupling undocumented | Low | `lm/transduced.py` | Small |
| No type annotations | Medium | Public API modules | Large |
| Utility modules untested | Low | `util.py` | Small |

---

## Roadmap

### Phase 1: Make It Fast for Real LMs

**Goal:** TransducedLM with GPT-2 + BPE FST runs at interactive speed.

1. **Wire TokenDecompose into TransducedLM.** When
   `check_all_input_universal(fst)` is true, use the BPE fast path
   automatically. The 5000x speedup should be transparent to the user.

2. **Profile the TransducedLM hot loop with a neural LM.** Establish a
   baseline and identify the current bottleneck.

3. **Batch source-symbol expansions.** Group particles by DFA state, batch
   `lm_state >> x` calls into a single forward pass. This is the biggest
   remaining performance win for GPU-backed LMs.

4. **Write a "hello world" example.** A single script showing: load n-gram LM,
   build FST, create TransducedLM, decode a sentence.

### Phase 2: Make It Usable by Others

**Goal:** A new contributor can understand and extend the library.

5. **Fix multi-character symbol handling.** Switch PrecoverNFA buffers from
   string indexing to tuple-of-symbols. This unblocks PTB and other
   non-byte FSTs.

6. **Add type annotations to public API.** Start with `base.py`, `fst.py`,
   `lm/base.py`, `lm/transduced.py`.

7. **Benchmark regression tracking.** Run PTB benchmarks in CI, store timing
   in `output/`, alert on regressions > 20%.

### Phase 3: Scale (ongoing)

8. **FusedTransducedLM evaluation.** Benchmark against standard TransducedLM
   on GPT-2 + BPE to determine when the fused single-pass approach wins.

9. **Streaming / multi-sequence support.** Enable batched inference across
   multiple decode sequences (e.g., for beam search at the TransducedLM level).

10. **FST construction toolkit.** Add utilities for regex-to-FST,
    grammar-to-FST without requiring pynini (GPL dependency risk).

---

## Summary

| Dimension | Grade | Key Finding |
|-----------|-------|-------------|
| **Algorithms** | A | 12 implementations, well-tested, genuine innovations (Peekaboo, dirty-state) |
| **Performance** | A- | 5000x BPE speedup, 25x Rust acceleration, 0.1ms per-step dirty-state |
| **Architecture** | A- | Clean layering, no circular deps, optional Rust, self-contained LM module |
| **Testing** | A | 663 tests, parametrized cross-validation, CI via GitHub Actions |
| **End-to-End Product** | A- | Particle-based beam-sum/SIR inference; quotient exact marginalization |
| **API/Packaging** | B | Exports correct; no end-to-end example; TokenDecompose not auto-wired |
| **Documentation** | B- | Core concepts documented; function-level docs still sparse |

**Bottom line:** The hard parts — fast, correct FST decomposition AND a working
`TransducedLM` with particle-based approximate inference — are done. The remaining
work is performance (TokenDecompose integration, batched LM calls) and usability
(multi-char symbols, examples, types).
