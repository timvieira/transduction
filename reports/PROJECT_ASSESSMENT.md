# Project Assessment: Delivering on the Transduced LM Mission

## 2026-02-14

---

## Executive Summary

The transduction library aims to efficiently compute next-symbol probabilities
for a language model composed with a finite-state transducer — enabling
constrained and transformed autoregressive generation. The algorithmic core is
strong: Peekaboo decomposition, dirty-state persistence, BPE-specific fast
paths, and Rust acceleration form a layered optimization stack that achieves
real-time per-step costs on production-scale FSTs. However, the end-to-end
product — `TransducedLM`, the thing users actually call — has a known
carry-forward bug that limits it to single-state FSTs, the public API exports
the wrong algorithms, and there is no CI to catch regressions. The path from
"impressive research prototype" to "library others can use" requires fixing a
small number of high-impact issues.

---

## What's Working Well

### 1. Algorithmic Depth

The project implements 12 decomposition algorithms spanning the full design
space: reference implementations, incremental variants, batched variants,
BPE-specific fast paths, and Rust backends. This is not redundant — each
variant serves a distinct niche:

| Layer | Algorithm | Purpose |
|-------|-----------|---------|
| Reference | `Precover`, `NonrecursiveDFADecomp` | Correctness oracle |
| Batched | `PeekabooState`, `PeekabooNonrecursive` | Amortize DFA across all next symbols |
| Incremental | `DirtyPeekaboo`, `TruncatedIncrementalDFADecomp` | Reuse DFA state across decode steps |
| BPE fast path | `TokenDecompose` | Exploit hub structure (5000x speedup) |
| Rust | `RustDecomp`, `RustDirtyState`, `RustDirtyPeekaboo` | 3-25x over Python |
| Finite-only | `LazyIncremental`, `LazyNonrecursive`, `PrioritizedLazy` | Finite-language FSTs |

The parametrized test suite (`test_general.py`: 18 FST examples x 9
implementations) ensures all general-case algorithms agree. When they don't,
the bug is found immediately.

### 2. Optimizations With Measured Impact

These aren't speculative — they're benchmarked:

| Optimization | Impact | Mechanism |
|-------------|--------|-----------|
| `all_input_universal` | 11,500x on BPE | Skip universality BFS entirely |
| Token-level position tracking | 5000x on BPE | Collapse FST-state x position to just position |
| Peekaboo batching | ~|Y|x amortization | One DFA for all next symbols |
| Dirty-state persistence | Per-step → 0.1ms | Reuse clean DFA states across steps |
| Rust backend | 3-25x | Packed u64 NFA states, interned DFA states |

### 3. Clean Internal Architecture

10.6K lines of Python across 32 modules. No circular dependencies. Well-layered:

```
FST / FSA  (data structures)
   ↓
base.py  (DecompositionResult, abstract interfaces)
   ↓
algorithm implementations  (peekaboo, dirty, token_decompose, ...)
   ↓
lm/transduced.py  (TransducedLM — user-facing)
```

The Rust bridge is optional and degrades gracefully. The LM submodule
(`lm/`) is self-contained with no pollution of core algorithms.

### 4. Solid Test Infrastructure

- **253+ passing tests** across 10 test files
- Parametrized cross-algorithm validation catches disagreements automatically
- Reference implementations (`Precover`) serve as correctness oracles
- Real-model integration tests (GPT-2 + BPE FST in `test_enumeration.py`)
- Clear general vs. finite-only test separation

### 5. Elegant User-Facing API

The `TransducedLM` API mirrors the inner LM interface exactly:

```python
tlm = TransducedLM(inner_lm, fst)
state = tlm >> 'h'          # advance by target symbol
p = state.logp_next['e']    # query next-symbol probability
```

Users don't need to understand precovers, peekaboo, or dirty states. The
`>>` operator and `logp_next` property are the entire surface area.

---

## What's Not Working

### Critical: TransducedLM Carry-Forward Bug

**Location:** `lm/transduced.py`, `_compute_logp_next()` beam carry-forward

**Symptom:** "Out of vocabulary" errors on multi-state FSTs. Works for
single-state FSTs (identity, lowercase, copy) but fails on multi-state FSTs
(duplicate, small, most real transducers).

**Root cause:** The `carry_forward` dict only captures beam items at Q/R
states (during expansion) and resume-frontier states (during drain).
Intermediate DFA states that sit on the resume frontier but are expanded
before the drain phase are lost. The next step's beam can't reach Q/R states
for some target symbols.

**Impact:** This is the #1 blocker for the project's mission. TransducedLM is
the product — the thing users call to get constrained generation. If it only
works on trivial FSTs, the library doesn't deliver.

### Critical: Wrong Algorithms Exported — RESOLVED

`__init__.py` now exports `PeekabooState`, `Peekaboo`, `DirtyPeekaboo`,
`TokenDecompose`, and `TransducedLM` as recommended algorithms. Finite-only
algorithms (`LazyIncremental`, etc.) are still exported for backward compat
but grouped separately. Rust backends remain at `transduction.rust_bridge`
(optional dependency).

### Critical: No CI/CD — RESOLVED

GitHub Actions workflow added (`.github/workflows/test.yml`): builds the Rust
crate, installs the wheel, runs `test_general.py`, `test_finite.py`, and
`test_push_labels.py` on Python 3.10. Also eliminated the `arsenal` external
dependency by inlining needed utilities into `util.py`.

### High: Core Concepts Undocumented — RESOLVED

`base.py` now has a module-level docstring defining precover, quotient,
remainder, universality, truncation, and dirty state in plain English.
`PeekabooState` has a comprehensive docstring documenting all five lazy BFS
attributes (`decomp`, `dfa`, `incoming`, `resume_frontiers`, `preimage_stops`)
and the incremental chain mechanism.

### High: Multi-Character Symbol Handling Broken

All PrecoverNFA implementations index into the output buffer by character
position (`ys[N]`, `ys[:N+1]`). This clips multi-character symbols (e.g.,
PTB byte-value strings like '84', '104'). Blocks use cases beyond byte-level
FSTs.

### Medium: No Batched Inference

`TransducedLM` processes one sequence at a time. The LM state advance
(`lm_state >> x`) consumes 30-40% of `_compute_logp_next` time. Batching
multiple source-symbol expansions would improve GPU utilization dramatically
for neural LMs.

---

## Codebase Health Metrics

### Size and Complexity

| Component | Files | Lines | Notes |
|-----------|-------|-------|-------|
| Core (FST/FSA/base) | 3 | 1,861 | Stable, well-tested |
| Algorithms | 9 | 2,895 | Active development |
| LM integration | 5 | 1,594 | Self-contained |
| Rust backend | 6 | ~3,000 | Well-optimized |
| Applications | 3 | 589 | BPE, PTB, WikiText |
| Utilities | 3 | 1,801 | vibes, examples, lazy |
| **Total Python** | **32** | **10,619** | |
| Tests | 10 | 2,810 | Good coverage |

### Technical Debt

| Item | Severity | Location | Effort |
|------|----------|----------|--------|
| TransducedLM carry-forward bug | Critical | `lm/transduced.py` | Medium |
| Wrong exports in `__init__.py` | Resolved | `__init__.py` | Done |
| No CI | Resolved | `.github/workflows/` | Done |
| Undocumented "precover" concept | Resolved | `base.py` | Done |
| Multi-char symbol handling | High | `precover_nfa.py` | Medium |
| No end-to-end example | High | — | Small |
| No `__all__` declarations | Declined | `__init__.py` | — |
| No type annotations | Medium | All modules | Large |
| Utility modules untested | Low | `util.py`, `vibes.py` | Small |
| 27 inline TODOs | Low | Various | Ongoing |

---

## Roadmap: Delivering on the Mission

The mission is *efficiently support transduced LM*. Here's what to do, in
priority order.

### Phase 1: Make TransducedLM Work (1-2 weeks)

**Goal:** A user can run `TransducedLM(lm, fst)` on arbitrary FSTs and get
correct next-symbol probabilities.

1. **Fix the carry-forward bug.** Save beam items at resume-frontier states
   during expansion, not just during the drain phase. Add test coverage for
   multi-state FSTs (duplicate, small, newspeak2) in `test_transduced.py`.

2. **Fix `__init__.py` exports.** Done — recommended algorithms exported.

3. **Add CI.** Done — `.github/workflows/test.yml`.

4. **Write a "hello world" example.** A single script showing: load n-gram LM,
   build FST, create TransducedLM, decode a sentence. Put it in `examples/`
   or in the README.

### Phase 2: Make It Fast for Real LMs (2-4 weeks)

**Goal:** TransducedLM with GPT-2 + BPE FST runs at interactive speed.

5. **Wire TokenDecompose into TransducedLM.** When
   `check_all_input_universal(fst)` is true, use the BPE fast path
   automatically. The 5000x speedup should be transparent to the user.

6. **Profile the TransducedLM hot loop with a neural LM.** The current
   profiling (`benchmark/profile_transduced.py`) is thorough but predates
   some optimizations. Establish a baseline and identify the current
   bottleneck.

7. **Batch source-symbol expansions.** Group beam items by DFA state,
   batch `lm_state >> x` calls into a single forward pass. This is the
   biggest remaining performance win for GPU-backed LMs.

8. **Benchmark regression tracking.** Run PTB benchmarks in CI, store timing
   in `output/`, alert on regressions > 20%.

### Phase 3: Make It Usable by Others (2-4 weeks)

**Goal:** A new contributor can understand and extend the library.

9. **Document the precover concept.** Done — `base.py` module docstring.

10. **Document PeekabooState internals.** Done — `PeekabooState` class docstring.

11. **Fix multi-character symbol handling.** Switch PrecoverNFA buffers from
    string indexing to tuple-of-symbols. This unblocks PTB and other
    non-byte FSTs.

12. **Add type annotations to public API.** Start with `base.py`, `fst.py`,
    `lm/base.py`, `lm/transduced.py`. Internal modules can follow later.

### Phase 4: Scale (ongoing)

13. **FusedTransducedLM evaluation.** It avoids materializing the full DFA,
    which should help when the inner LM has a steep distribution. Benchmark
    against standard TransducedLM on GPT-2 + BPE.

14. **Streaming / multi-sequence support.** Enable batched inference across
    multiple decode sequences (e.g., for beam search at the TransducedLM
    level).

15. **FST construction toolkit.** `bpe_wfst()` is great but covers one use
    case. Add utilities for regex-to-FST, grammar-to-FST without requiring
    pynini (GPL dependency risk).

---

## Summary

| Dimension | Grade | Key Finding |
|-----------|-------|-------------|
| **Algorithms** | A | 12 implementations, well-tested, genuine innovations (Peekaboo, dirty-state) |
| **Performance** | A- | 5000x BPE speedup, 25x Rust acceleration, 0.1ms per-step dirty-state |
| **Architecture** | A- | Clean layering, no circular deps, optional Rust, self-contained LM module |
| **Testing** | A- | 382+ tests, parametrized cross-validation, CI via GitHub Actions |
| **End-to-End Product** | C | TransducedLM carry-forward bug blocks multi-state FSTs |
| **API/Packaging** | C+ | Exports fixed; no end-to-end example yet |
| **Documentation** | C+ | Core concepts now documented; function-level docs still sparse |

**Bottom line:** The hard part — fast, correct FST decomposition — is done.
The easy part — wiring it into a working TransducedLM that users can import
and call — is what remains. Phase 1 is small, high-leverage work that
transforms this from a research prototype into a usable library.
