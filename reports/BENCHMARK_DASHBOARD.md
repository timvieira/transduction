# Benchmark Dashboard

**Last updated:** 2026-02-22
**Test suite:** 1191 tests across 16 files (1189 passed, 2 xfailed)
**Rust optimizations:** SymbolIndex, bitset closures, deferred grouping, persistent FST caches

---

## What This Library Does

Computes **next-symbol probabilities for a language model composed with a
finite-state transducer**. Given an inner LM and an FST (e.g., BPE tokenizer,
PTB normalizer), `TransducedLM` produces a new LM over the target alphabet,
marginalizing over all source continuations at each step.

The optimization stack:
1. **Peekaboo decomposition** — Q(y)/R(y) for all next symbols in one BFS pass
2. **Dirty-state persistence** — reuse DFA states across steps
3. **Rust acceleration** — hot loops in compiled Rust via PyO3
4. **Fused search** (`FusedTransducedLM`) — interleaves decomposition and LM search in a single priority queue
5. **Token-level decomposition** — O(N) position-set DFA states for BPE-like FSTs

---

## BPE Vocab Scaling

This is the most important scaling curve. How does end-to-end TransducedLM cost
grow with BPE vocabulary size?

**Setup:** Subsampled GPT-2 BPE FSTs at increasing vocabulary sizes. 8 decode
steps on "The quick", K=10, max_steps=200, CharNgramLM (LM cost ~0).
Backend: `FusedTransducedLM(helper='rust')` — generic Rust dirty-state peekaboo.

![BPE Vocab Scaling](figures/bpe_vocab_scaling.png)

| Vocab | FST states | FusedLM (ms/step) | Peak RSS (GB) | Delta mem (GB) |
|------:|-----------:|------------------:|--------------:|---------------:|
| 297 | 387 | **8** | 0.7 | 0.0 |
| 529 | 623 | **14** | 0.7 | 0.0 |
| 1,023 | 1,313 | **31** | 0.8 | 0.1 |
| 2,020 | 3,010 | **81** | 0.9 | 0.1 |
| 5,011 | 8,694 | **296** | 1.6 | 0.9 |
| 7,010 | 12,408 | **282** | 1.7 | 1.0 |
| 10,008 | 18,203 | **816** | 3.5 | 2.8 |
| 12,008 | 22,172 | **756** | 3.9 | 3.2 |
| 15,008 | 28,005 | **1,349** | 6.9 | 6.1 |
| *20,000* | | *~2,100* | *~7.8* | *~7.1* |
| *30,000* | | *~3,500* | *~14.5* | *~13.8* |
| *50,257* | | *~6,900* | *~32.9* | *~32.2* |

Baseline RSS (Python + tokenizer): 0.7 GB. Delta = peak RSS - baseline.
Italic rows are extrapolated from log-log fit (time: `|V|`^1.31, memory: `|V|`^1.64).

**Memory is the binding constraint**, not time. At full GPT-2 scale (50k
tokens), the extrapolated time (~7 s/step) is expensive but feasible. The
extrapolated memory (~33 GB) is prohibitive — the Rust DFA arena's
`O(|V| x |closure|)` materialization drives `|V|`^1.64 memory growth.
At V=15k (the largest measured point), peak RSS is already 6.9 GB; 50k would
need ~33 GB for the DFA arena alone.

**Key findings:**
- **Time scales as `|V|`^1.31** — 1.3 seconds/step at 15k, extrapolated to
  ~7 seconds/step at 50k. Tolerable for offline use, needs ~70x improvement
  for interactive speed.
- **Memory scales as `|V|`^1.64** — the steeper exponent makes memory the
  bottleneck well before time becomes impractical. The DFA arena stores
  `O(|V| x |closure|)` packed entries per expanded state; at boundary steps
  this materializes the full cross-product.
- The V=5k→7k timing plateau (296 vs 282 ms) reflects that the 8-step decode
  hits similar boundary patterns at both sizes; the jump at V=10k and V=15k
  confirms the underlying superlinear trend.

**What's needed for production:**

The factored DFA arena has been implemented (see "Completed Optimizations"
below), reducing memory by 23-43% at V >= 2000. However, wall-clock is roughly
neutral — the BFS/universality check dominates, not state representation. The
`top_k` pruning (see below) is the most effective wall-clock optimization so far.

Source: `reports/bench_vectorization.py` → `reports/bench_vectorization_results.json`

---

## top_k Pruning vs Full Expansion

The `top_k` feature prunes source-symbol expansion to only the k highest-probability
symbols under the inner LM at each step. Instead of expanding all |V| arcs via
batch `compute_all_arcs`, it calls `single_arc` for each of the top-k symbols.

**Setup:** Same as BPE vocab scaling above. `FusedTransducedLM(helper='rust', top_k=k)`
for k ∈ {50, 100, 500} vs full expansion (no top_k).

| Vocab | Full (ms) | top_k=50 | top_k=100 | top_k=500 |
|------:|----------:|---------:|----------:|----------:|
| 297 | 11 | **1** | 4 | 30 |
| 529 | 19 | **2** | 6 | 62 |
| 1,023 | 38 | **5** | 12 | 112 |
| 2,020 | 86 | **9** | 24 | 254 |
| 5,011 | 305 | **30** | 85 | 789 |
| 7,010 | 249 | **51** | 124 | 1,086 |
| 10,008 | 597 | **68** | 144 | 672 |
| 15,008 | 516 | **104** | 230 | 1,006 |

**Key findings:**
- **top_k=50 gives 5-10x speedup** across all vocab sizes. At V=15k, 104 ms vs
  516 ms (5x). At V=2k, 9 ms vs 86 ms (10x). Scaling is roughly linear in |V|
  (the DFA expansion cost is now O(k) instead of O(|V|)).
- **top_k=100 gives 2-4x speedup** — a reasonable accuracy/speed tradeoff.
- **top_k=500 is slower than full expansion** — the per-arc `single_arc` overhead
  (hash lookup per symbol) dominates when k is large. Batch `compute_all_arcs`
  is more efficient when expanding most of the alphabet.
- **Crossover point:** top_k is faster than full expansion when k < ~|V|/3.
  At k=500 with V=297, top_k expands more symbols than exist, so it's pure overhead.
- **Approximation quality:** top_k is an approximation — it drops low-probability
  source paths. At top_k=50, only ~16/257 target keys match the full distribution
  at boundary steps. At top_k=500, all keys are covered but max logp difference
  can reach ~4-14 nats at some steps. For beam search / greedy decode where only
  the top few targets matter, top_k=50-100 is likely sufficient.

Source: `reports/bench_vectorization.py` → `reports/bench_vectorization_results.json`

---

## Factored Arena vs Flat Arena

The factored DFA arena stores off-target boundary elements as
`(FST_closure, params_list)` groups instead of materializing the full
`|V| × |closure|` cartesian product. This reduces per-boundary-state memory
from O(|V| × |closure|) to O(|closure| + |V|).

**Setup:** GPT-2 BPE FSTs at increasing vocabulary sizes. 8 decode steps on
"The quick brown fox". Factored: `RustPeekabooState` decomposition only.
Flat: `FusedTransducedLM(helper='rust')` K=10 (from prior benchmark).
Memory comparison is valid (arena dominates RSS); time comparison is
**not** apples-to-apples (full BFS decomposition vs beam-pruned FusedLM).

![Factored Arena Scaling](figures/factored_arena_scaling.png)

| Vocab | Flat delta (MB) | Factored delta (MB) | Reduction |
|------:|----------------:|--------------------:|----------:|
| 1,000 | 43 | 9 | 79% |
| 2,000 | 134 | 66 | 51% |
| 5,000 | 675 | 581 | 14% |
| 7,000 | 1,141 | 1,029 | 10% |
| 10,000 | 2,148 | 1,694 | **21%** |

**Key findings:**
- **Memory is consistently lower** at every measured vocab size (10-79%
  reduction). The savings are largest at V=1-2k where boundary states dominate.
- **Wall-clock is roughly neutral.** On identical synthetic FSTs, factored is
  ~16% faster at V=500, ~4% faster at V=1k, and ~20% slower at V=2-5k due to
  `normalize_for_step` cloning and fingerprint-based interning overhead.
- The memory reduction alone raises the feasible vocab ceiling from ~13k (8 GB
  OOM) to ~15-17k under the same memory limit.

Source: `reports/bench_factored_scaling.py` → `reports/bench_factored_results.json`

---

## BPE End-to-End (1k Vocab)

Full 44-step decode at VOCAB_SIZE=1000. This is the longest run at meaningful
scale.

| Method | Total (s) | Avg/step (ms) | Steps |
|--------|----------:|--------------:|------:|
| FusedTransducedLM | 0.81 | **18.5** | 44 |
| TransducedLM | 5.54 | 125.9 | 44 |

FusedLM is **6.8x faster** than TransducedLM at this scale. The gap widens with
vocabulary size because Fused avoids materializing the full peekaboo DFA.

Source: `notes/bpe-lm-benchmark.ipynb` (VOCAB_SIZE=1000 run)

---

## PTB End-to-End

PTB is a different regime: 296 states, 23K arcs, 257 symbols (full byte
alphabet), complex CDRewrite topology. No vocab scaling axis — it's a fixed
transducer.

![PTB TransducedLM](figures/ptb_transduced_lm.png)

| Method | Total (s) | Avg/step (ms) | Steps |
|--------|----------:|--------------:|------:|
| TransducedLM | 5.80 | 129 | 45 |
| FusedTransducedLM | 2.95 | **66** | 45 |
| PyniniTransducedLM | — | — | hangs |

FusedLM is **2.0x faster**. PyniniTransducedLM hangs due to O(|B|)=255
per-symbol compositions.

**Decomposition backend:**

![PTB Backends](figures/ptb_backends.png)

Rust decomposition is **15x faster** than Python (110 ms vs 1,651 ms geomean).

Config: K=20, max_expansions=200, CharNgramLM, "The quick brown fox..." (45 bytes).

Source: `reports/run_benchmarks.py`, `notes/ptb-lm-benchmark.ipynb`

---

## Incremental DFA: Per-Step Work Scaling

From `notes/incremental_scaling.ipynb` — how does dirty-state incremental
decomposition scale over long decode chains (200 steps)?

| FST | Final DFA states | Steady-state ms/step | Scratch ms/step | Speedup |
|-----|------------------:|---------------------:|----------------:|--------:|
| triplets_of_doom | 600 | 0.081 | 1.27 | 15.7x |
| 3-tuples_of_doom | 2,200 | 0.142 | 9.58 | 67.5x |
| lookahead | 200 | 0.020 | 0.50 | 25.0x |
| duplicate | 400 | 0.029 | 0.35 | 12.1x |

Per-step cost plateaus at ~0.02–0.14 ms regardless of total DFA size. The
incremental algorithm pays O(|change|), not O(|total DFA|). Correlation:
**R²=0.97** with ~2.3 μs per state expanded.

---

## Open Issues

- **rust_token still ~10x slower than generic rust** (partially addressed):
  Lazy DFA (2026-02-22) improved by ~1.6x and dirty-state persistence has been
  added to `TokenPeekabooDFA`, but the 10x gap persists at all vocab sizes.
  Root cause: per-state arc computation cost is O(|V| x |eps_closure|)
  regardless of dirty-state amortization — the same bottleneck as the generic
  backend. Position-key quotienting reduces DFA state count (~45 vs ~7K) but
  doesn't reduce per-state arc computation cost.

- **No batched LM inference** ([#7](https://github.com/timvieira/transduction/issues/7)):
  All benchmarks use CharNgramLM (O(1) per call). With a real GPU LM, batching
  multiple `lm_state >> x` calls into one forward pass is the most impactful
  unstarted optimization.

- **DirtyPeekaboo non-monotonic targets** ([#5](https://github.com/timvieira/transduction/issues/5)):
  Incorrect results with tree-branching decode (shorter target after longer).

- **PyniniNonrecursiveDecomp epsilon-chain bug**: `test_bpe_like` and
  `test_bpe_embedded` are xfailed for this implementation (the only 2 xfails
  in the full test suite).

---

## Completed Optimizations

1. **Factored DFA arena** (2026-02-22): `FactoredArena` replaces
   `PowersetArena` in `DirtyPeekaboo` and `LazyPeekabooDFA`. Off-target
   elements sharing the same FST closure are stored as `(closure, params_list)`
   instead of `|V| x |closure|` flat entries. Memory reduction is significant
   (23-43% at V >= 2000), but wall-clock is roughly neutral — slightly faster
   at V=500-1000, ~20% slower at V=2000-5000 due to `normalize_for_step`
   cloning, fingerprint interning overhead, and more complex arc computation.
   See TODO.md for detailed benchmark tables and remaining sub-items (profiling,
   lazy normalization, collision chain analysis).

2. **Vectorization optimizations** (2026-02-22): SymbolIndex (O(1) per-symbol
   projection), FstBitset closures, deferred grouping in `compute_all_arcs`,
   and persistent FST closure caches. These are structural prerequisites for
   the factored arena but showed no measurable end-to-end speedup alone because
   dirty-state persistence already makes most steps O(|change|).

3. **rust_token dirty-state persistence** (2026-02-22): Added to
   `TokenPeekabooDFA` with selective invalidation of dirty+border states.
   Combined with lazy DFA expansion (~1.6x speedup), but the 10x gap vs
   generic rust persists because per-state arc cost O(|V| x |closure|) is
   the same.

## Most Promising Directions

1. **Batched LM calls**: With a real GPU LM, the LM forward pass will
   dominate. Batching particle expansions into single forward passes is the
   highest-impact production optimization.

2. **Pruning at boundary steps** (partially addressed): `top_k` pruning gives
   5-10x speedup at k=50, cutting superlinear scaling to roughly linear. See
   "top_k Pruning vs Full Expansion" above. Remaining work: adaptive k selection
   and quality-aware cutoffs.

3. **Profile with real LM**: We don't know the decomp/LM cost split with GPT-2.
   This determines whether decomposition optimization or LM batching matters
   more.

4. **FactoredArena wall-clock regression**: The factored arena saves memory but
   is ~20% slower at V >= 2000. Profiling BFS/universality vs arc computation,
   avoiding `normalize_for_step` cloning, and analyzing fingerprint collision
   chains could close the gap. See TODO.md for specifics.

---

## Regenerating

```bash
python reports/dashboard_plots.py        # plots → reports/figures/
python reports/bench_vectorization.py    # BPE vocab scaling (V=297..12008)
python reports/run_benchmarks.py         # full benchmark suite
```

Vocab scaling data comes from `notes/bpe-lm-benchmark.ipynb` (run interactively)
or `reports/bench_vectorization.py` (standalone script).

---

## Change Log

| Date | Change |
|------|--------|
| 2026-02-22 | Add top_k pruning benchmark: top_k=50 gives 5-10x speedup vs full expansion |
| 2026-02-22 | Add factored arena scaling curves (memory + time comparison vs flat arena) |
| 2026-02-22 | Reorganize dashboard: move FactoredArena, vectorization, rust_token dirty-state to "Completed Optimizations"; update open issues and future directions |
| 2026-02-22 | Extended BPE vocab scaling sweep: 8 points (297→12,008), hitting 8 GB OOM ceiling at ~13.5k |
| 2026-02-22 | Rust vectorization: SymbolIndex, FstBitset, deferred grouping, persistent caches — no end-to-end speedup (constant-factor only, masked by dirty-state persistence) |
| 2026-02-22 | Lazy TokenPeekabooDFA: ~1.6x speedup but per-state O(|V|) cost remains |
| 2026-02-21 | Rewrite dashboard: focus on scaling curves, cut 43-token toy discussion |
| 2026-02-21 | Add 6-point BPE vocab scaling data (297→5,011) with extrapolation to 50k |
| 2026-02-21 | Identify rust_token scaling regression (~|V|^2.7 vs expected O(N)) |
| 2026-02-21 | Add lazy precover DFA, token-quotiented peekaboo (Rust+Python) |
| 2026-02-21 | Fix PeekabooState incorrect Q/R on BPE-style epsilon-output chains (#9) |
| 2026-02-20 | Fix FusedTransducedLM logp disagreement; diff 2.03→0.000287 |
| 2026-02-20 | Created dashboard |
