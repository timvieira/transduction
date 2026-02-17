# Project Assessment: Delivering on the Transduced LM Mission

## 2026-02-16

---

## Executive Summary

The transduction library efficiently computes next-symbol probabilities for a
language model composed with a finite-state transducer — enabling constrained
and transformed autoregressive generation. The algorithmic core is strong:
Peekaboo decomposition, dirty-state persistence, and
Rust acceleration form a layered optimization stack that achieves real-time
per-step costs on production-scale FSTs. The user-facing `TransducedLM`
now defaults to the Rust backend (`RustPeekabooState`) and implements
particle-based beam-sum approximate inference (deterministic top-K, consistent
as K -> inf). Quotient states provide exact marginalization over infinitely many
source continuations — the key variance reduction mechanism. A carry-forward
prefix-domination bug was fixed in both `TransducedLM` and `FusedTransducedLM`.
Rich notebook display (`_repr_html_`) enables interactive exploration.
Test coverage expanded significantly (660 tests, up from 493). CI, exports,
and core documentation are in place. The remaining path to production readiness:
batch LM calls for GPU utilization.

---

## What's Working Well

### 1. Algorithmic Depth

11 decomposition algorithms spanning the full design space: reference
implementations, incremental variants, batched variants, and Rust backends.
Each variant serves a distinct niche:

| Layer | Algorithm | Purpose |
|-------|-----------|---------|
| Reference | `Precover`, `NonrecursiveDFADecomp` | Correctness oracle |
| Batched | `PeekabooState`, `Peekaboo` (nonrecursive) | Amortize DFA across all next symbols |
| Incremental | `DirtyPeekaboo`, `TruncatedIncrementalDFADecomp` | Reuse DFA state across decode steps |
| Rust | `RustDecomp`, `RustDirtyState`, `RustDirtyPeekaboo` | 3-25x over Python |
| Finite-only | `LazyIncremental`, `LazyNonrecursive`, `PrioritizedLazy` | Finite-language FSTs |

The parametrized test suite (`test_general.py`: 352 tests) ensures all
general-case algorithms agree.

### 2. Optimizations With Measured Impact

| Optimization | Impact | Mechanism |
|-------------|--------|-----------|
| `all_input_universal` | 11,500x on BPE | Skip universality BFS entirely |
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
algorithm implementations  (peekaboo, dirty, ...)
   |
lm/transduced.py  (TransducedLM — user-facing)
```

The Rust bridge is optional and degrades gracefully. The LM submodule (`lm/`)
is self-contained with no pollution of core algorithms.

### 4. Solid Test Infrastructure

- **660 tests** across 13 test files, all passing (36 skipped for optional deps)
- Parametrized cross-algorithm validation catches disagreements automatically
- Reference implementations (`Precover`) serve as correctness oracles
- Real-model integration tests (GPT-2 + BPE FST in `test_enumeration.py`)
- TransducedLM tests: multi-state FSTs, brute-force comparison, consistency
  convergence, carry-forward prefix-domination regression tests
- `test_fst.py`: 50 tests covering FST methods (99% coverage)
- `test_lazy_peekaboo_dfa.py`: Rust lazy DFA integration tests
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
`>>` operator and `logp_next` property are the entire surface area.
`TransducedLM` now defaults to the Rust backend for decomposition. Both
`TransducedState` and `FusedTransducedState` support rich notebook display
via `_repr_html_` with unified visualization.

---

## Open Issues

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
| Core (FST/FSA/base) | 4 | ~2,000 | Stable, well-tested (precover.py split out) |
| Algorithms | 9 | ~2,900 | Active development |
| LM integration | 6 | ~1,700 | Self-contained (includes FusedTransducedLM) |
| Rust backend | 9 | ~5,500 | Well-optimized (expanded py.rs bindings) |
| Applications | 3 | ~590 | BPE, PTB, WikiText |
| Utilities | 3 | ~2,400 | viz, examples, lazy |
| **Total Python** | **33** | **~11,600** | |
| Tests | 13 | ~4,650 | 660 tests |

### Technical Debt

| Item | Severity | Location | Effort |
|------|----------|----------|--------|
| No end-to-end example | High | — | Small |
| No batched LM calls | Medium | `lm/transduced.py` | Medium |
| K/max_expansions coupling undocumented | Low | `lm/transduced.py` | Small |
| No type annotations | Medium | Public API modules | Large |

### Recently Resolved

| Item | Date | Notes |
|------|------|-------|
| Carry-forward prefix-domination bug | 2026-02-16 | Fixed in TransducedLM and FusedTransducedLM; regression tests added |
| fst.py low test coverage (59%) | 2026-02-16 | Raised to 99% with 50 tests in test_fst.py |
| TransducedLM used Python decomposition | 2026-02-16 | Now defaults to Rust `RustPeekabooState` |
| No rich notebook display | 2026-02-16 | `_repr_html_` on TransducedState and FusedTransducedState |
| Precover mixed into fst.py | 2026-02-16 | Split into dedicated `precover.py` module |

---

## Roadmap

### Phase 1: Make It Fast for Real LMs

**Goal:** TransducedLM with GPT-2 + BPE FST runs at interactive speed.

1. **Profile the TransducedLM hot loop with a neural LM.** Establish a
   baseline and identify the current bottleneck.

2. **Batch source-symbol expansions.** Group particles by DFA state, batch
   `lm_state >> x` calls into a single forward pass. This is the biggest
   remaining performance win for GPU-backed LMs.

3. **Write a "hello world" example.** A single script showing: load n-gram LM,
   build FST, create TransducedLM, decode a sentence.

### Phase 2: Make It Usable by Others

**Goal:** A new contributor can understand and extend the library.

5. ~~**Fix multi-character symbol handling.**~~ Done. All output buffers
   now use tuples of symbols; `FSA.language()` always returns tuples.

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
| **Algorithms** | A | 11 implementations, well-tested, genuine innovations (Peekaboo, dirty-state) |
| **Performance** | A | 25x Rust acceleration, 0.1ms per-step dirty-state; Rust now default backend |
| **Architecture** | A- | Clean layering, no circular deps, optional Rust, self-contained LM module |
| **Testing** | A+ | 660 tests across 13 files, fst.py at 99% coverage, carry-forward regression tests |
| **End-to-End Product** | A- | Particle-based beam-sum inference; quotient exact marginalization; rich notebook display |
| **API/Packaging** | B+ | Exports correct; Rust default; `_repr_html_`; no end-to-end example yet |
| **Documentation** | B- | Core concepts documented; function-level docs still sparse |

**Bottom line:** The hard parts — fast, correct FST decomposition AND a working
`TransducedLM` with particle-based approximate inference — are done. Multi-character
symbol support is complete (tuple-based buffers throughout). TransducedLM now defaults
to Rust acceleration. A carry-forward prefix-domination bug was found and fixed in both
TransducedLM and FusedTransducedLM. Test coverage jumped from 493 to 660 tests.
The remaining work is performance (batched LM calls) and usability (examples, types).
