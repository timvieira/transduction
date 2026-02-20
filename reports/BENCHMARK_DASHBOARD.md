# Benchmark Dashboard

**Last updated:** 2026-02-20
**Test suite:** 881 tests, 0 skipped

---

## Status at a Glance

### Table A: Decomposition Backends (raw Q/R speed, no LM)

| Method | BPE | PTB | Speedup vs Standard | Notes |
|--------|-----|-----|---------------------|-------|
| Standard (Python) | **OK** 3–7 ms | **OK** 1.65 s | 1x | `NonrecursiveDFADecomp`; powerset + universality |
| Pynini | **OK** 0.1–0.2 ms | **WIP** | 41x (BPE) | `PyniniNonrecursiveDecomp`; OpenFST composition |
| Rust | **OK** 0.8–1.5 ms | **WIP** | 4.7x (BPE) | `RustDecomp`; Rust powerset + universality |
| PositionSet | **OK** ~same | **OK** 5 ms | 331x (PTB) | `GeneralTokenDecompose`; exploits token-decomposability |

### Table B: TransducedLM Variants (end-to-end with LM)

| Variant | BPE Status | BPE ms/step | PTB Status | PTB ms/step | Notes |
|---------|------------|-------------|------------|-------------|-------|
| TransducedLM | **WIP** | — | **OK** | 169 | Rust peekaboo decomp + beam search; BPE not yet run |
| TransducedLM+PosSet | **WIP** | — | **WIP** | — | Commented out in PTB notebook |
| FusedTransducedLM | **WIP** | — | **OK** | 71 | Single-pass interleaved; 2.4x faster than TransducedLM on PTB |
| PyniniTransducedLM | **WIP** | — | **FAIL** | — | PTB: no steps completed (timeout or silent failure) |

---

## FST Characteristics

| Property | BPE (subsampled) | PTB |
|----------|------------------|-----|
| States | 140 | 296 |
| Input symbols (\|A\|) | 44 | 257 |
| Output symbols (\|B\|) | 34 | 256 |
| Arcs | ~180 | 23,723 |
| Topology | Star (token → byte chain → start) | CDRewrite rules + identity transducer |
| Build time | <0.001 s | 36.4 s |
| Token-decomposable | Yes (trivially; compression 1.0x) | Yes (compression 35.2x: 634 → 18 DFA states) |
| Vocab used | 43 tokens (from training data) | 257 byte symbols |

---

## Performance Detail

### Decomposition Backends on BPE

Target prefix length → best-of-3 time (ms). BPE FST: 140 states, 43 tokens.

| Length | Standard | Pynini | Rust |
|-------:|---------:|-------:|-----:|
| 3 | 3.8 | 0.1 | 1.0 |
| 5 | 3.7 | 0.2 | 1.5 |
| 8 | 3.3 | 0.1 | 0.8 |
| 10 | 5.2 | 0.1 | 0.9 |
| 15 | 3.6 | 0.1 | 0.8 |
| 20 | 5.7 | 0.1 | 0.9 |
| 30 | 4.6 | 0.1 | 0.9 |
| 40 | 6.8 | 0.2 | 0.9 |

**Geometric-mean speedup vs Standard:** Pynini 41.1x, Rust 4.7x.

Source: `notes/bpe-lm-benchmark.ipynb` cell `r1ontowex3b`

### Decomposition Backends on PTB

No direct backend-comparison timing data in the PTB notebook yet (the
section header exists but the benchmark code cell is missing).

Position-set results from `reports/general_token_decompose_report.md`:

| Method | Time | DFA States | Speedup |
|--------|-----:|-----------:|--------:|
| Standard (`NonrecursiveDFADecomp`) | 1.651 s | 634 | 1x |
| PositionSet (`GeneralTokenDecompose`) | 0.005 s | 18 | 331x |

### TransducedLM Variants on PTB

Config: K=20, max_expansions=200, 3-gram CharNgramLM, 60 s timeout/step.
Target: "The quick brown fox jumps over the lazy dog." (45 symbols).

| Variant | Total (s) | Avg/step (ms) | Steps | Notes |
|---------|----------:|:-------------:|------:|-------|
| TransducedLM | 7.6 | 169 | 45 | Completes all steps |
| FusedTransducedLM | 3.2 | 71 | 45 | 2.4x faster overall |
| PyniniTransducedLM | — | — | 0 | Header printed, no steps (timeout?) |

Source: `notes/ptb-lm-benchmark.ipynb` cell `cb1e75ae`

### TransducedLM Variants on BPE

BPE TransducedLM benchmark cells (`a7-benchmark`, `a8-summary`) have **no
saved outputs** — the benchmark hasn't been run on BPE yet.

---

## Correctness: logp Agreement

### PTB (45 steps)

| Method A | Method B | Max \|logp\| diff | Verdict |
|----------|----------|------------------:|---------|
| FusedTransducedLM | TransducedLM | **2.03** | Significant disagreement |

The 2.03 max absolute logp difference on PTB is large enough to indicate
a real correctness issue (not floating-point noise). Investigation needed
to determine which method is more accurate.

### BPE

No pairwise data yet (benchmark not run).

---

## Method-by-Method Status

### TransducedLM (Rust peekaboo)

- **Status:** OK on PTB; BPE not yet benchmarked
- **Architecture:** Two-phase — PeekabooState BFS decomposition (Rust), then
  beam-weighted search over Q/R
- **PTB performance:** 169 ms/step avg (45/45 steps)
- **Blockers:** None
- **Next steps:** Run BPE benchmark; investigate logp disagreement with Fused

### FusedTransducedLM

- **Status:** OK on PTB; BPE not yet benchmarked
- **Architecture:** Single-pass — interleaves decomposition and LM search in
  one priority queue, no separate BFS phase
- **PTB performance:** 71 ms/step avg (45/45 steps, 2.4x faster than TransducedLM)
- **Blockers:** logp disagreement vs TransducedLM (max diff 2.03)
- **Next steps:** Run BPE benchmark; root-cause logp discrepancy

### TransducedLM + PositionSet

- **Status:** WIP
- **Architecture:** TransducedLM using `PositionSetPeekabooState` +
  `_PositionSetPeekabooUniv` for decomposition
- **PTB performance:** Not tested (commented out in PTB notebook config)
- **BPE performance:** Not tested
- **Blockers:** Needs to be uncommented and run
- **Next steps:** Enable in PTB notebook; expected to dramatically speed up the
  decomposition phase given PTB's 35.2x compression ratio

### PyniniTransducedLM

- **Status:** FAIL on PTB; BPE not yet benchmarked
- **Architecture:** Pynini/OpenFST DFA construction + particle tracking
- **PTB performance:** No steps completed — printed header line but produced no
  timing data (likely timeout on step 1)
- **Blockers:** Silent failure on PTB (no error message visible)
- **Next steps:** Debug why step 1 produces no output; check if pynini
  composition is too expensive for 257-symbol alphabet + 296-state FST

### Decomposition: Pynini Backend

- **Status:** OK on BPE (41x speedup); PTB not benchmarked
- **Architecture:** OpenFST composition for precover construction
- **Blockers:** Missing PTB timing data in notebook
- **Next steps:** Add decomposition backend comparison to PTB notebook

### Decomposition: Rust Backend

- **Status:** OK on BPE (4.7x speedup); PTB not benchmarked
- **Architecture:** Rust powerset determinization + universality
- **Blockers:** Missing PTB timing data in notebook
- **Next steps:** Add decomposition backend comparison to PTB notebook

### Decomposition: PositionSet

- **Status:** OK; verified correct on PTB, BPE, and 5 other FSTs
- **Architecture:** BFS with position-set canonicalization; requires
  token-decomposability
- **PTB speedup:** 331x (634 → 18 DFA states)
- **BPE speedup:** ~1x (compression 1.0x — no benefit, every DFA state
  already unique)
- **Blockers:** Not yet integrated into TransducedLM benchmark (see
  TransducedLM+PosSet above)
- **Next steps:** Uncomment PosSet config in notebooks; measure end-to-end impact

### Infrastructure Issues

1. **BPE benchmark not run** — Cells `a7-benchmark` and `a8-summary` in
   `notes/bpe-lm-benchmark.ipynb` have no saved outputs. Need to re-run the
   notebook.

2. **PTB decomposition backend comparison missing** — Markdown section header
   exists in `notes/ptb-lm-benchmark.ipynb` but no code cell with timing data.
   Need to add Standard/Pynini/Rust comparison on PTB.

3. **logp disagreement (PTB)** — Max |logp| diff of 2.03 between TransducedLM
   and FusedTransducedLM. Need `ReferenceTransducedLM` ground-truth comparison
   to identify which is wrong (note: Reference only works on finite-relation
   FSTs, so this may require a truncated test).

---

## Change Log

| Date | Change |
|------|--------|
| 2026-02-20 | Created dashboard; consolidated data from BPE and PTB notebooks |
| 2026-02-20 | Added `PyniniNonrecursiveDecomp` to benchmark notebooks (`d5cf54a`) |
| 2026-02-20 | Added position-set decomposition report (`faa592d`) |
| 2026-02-20 | Added Rust token-level decomposition using position-set bitsets (`3366dfe`) |
| 2026-02-20 | Added `GeneralTokenDecompose` and token-decomposability analysis (`f6d3a07`) |
| 2026-02-20 | Added pynini-based FST decomposition (`4415308`) |
| 2026-02-19 | Add rho-arc compression, FST-level closure cache, int-token LM API (`38deccb`) |
| 2026-02-18 | Fix EOS double-counting for Q-absorbed preimage particles (`3df259c`) |
| 2026-02-17 | Add `ReferenceTransducedLM` for ground-truth validation (`2719d39`) |
| 2026-02-16 | Rewrite `TransducedLM` with particle-based approximate inference (`68648b3`) |
| 2026-02-08 | Add `FusedTransducedLM`, Lazy.cache(), and TransducedLM improvements (`346b047`) |
