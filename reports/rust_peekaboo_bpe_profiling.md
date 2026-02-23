# Rust Peekaboo FusedTransducedLM: BPE Profiling Report

## Setup

- **FST**: BPE (GPT-2 subsets), V in {59..1006}
- **Inner LM**: 3-gram CharNgramLM over token IDs
- **Search**: max_steps=200, max_beam=10
- **Target**: "The quick brown fox" (19 bytes, 4 BPE tokens)
- **Helper**: `rust` (RustLazyPeekabooDFA, factored representation)

## Key Finding: Two types of steps

BPE decoding has two kinds of steps:

1. **Boundary steps** (byte aligns with token boundary): the DFA state is a
   "big" powerset with ~V relevant output symbols. classify is called ~V
   times, arcs produces V transitions. These dominate.

2. **Interior steps** (within a multi-byte token): only ~1-10 relevant
   symbols, classify/arcs are negligible (<1ms). These are fast.

At V=1000, boundary steps take ~50-90ms while interior steps take <1ms.

## Time breakdown at V=1000 (first 4 boundary steps, ~220ms total)

| Component     | Time (ms) | % of total | Calls  |
|---------------|-----------|------------|--------|
| classify      | 114.5     | 52%        | 4083   |
| new_step      | 45.1      | 20%        | 7      |
| python search | 48.9      | 22%        | -      |
| arcs          | 10.5      | 5%         | 4      |
| run           | 2.0       | 1%         | 34     |

## Per-step detail at V=1000

```
step  byte  total_ms |  new_step  classify      arcs   py_srch | #cls #arcs #exp
  1    84      92.1 |      0.11     72.47      4.45     15.09 | 1008     1     1
  2   104      56.7 |     11.73     30.81      2.70     11.27 | 1010     1     1
  3   101      54.9 |     12.01     29.21      2.83     10.57 | 1016     1     1
  4    32      52.2 |     11.55     27.46      2.91      9.98 | 1017     1     1
  5   113      14.6 |     12.38      0.89      0.00      0.24 |   10     0     0
  6   117       0.9 |      0.57      0.18      0.00      0.14 |    1     0     0
```

Step 1 is slowest because classify processes all ~1008 symbols from scratch
(no universality cache). Steps 2-4 benefit from the `fst_univ_cache` built
during prior steps. Step 5+ are interior steps (few relevant symbols).

## Scaling analysis (first 4 boundary steps)

```
     V   new_step(ms)  classify(ms)  arcs(ms)  total(ms)
    59          0.18          1.05      0.12        2.6
   109          1.45          7.09      0.39       16.8
   209          5.71         19.69      0.75       35.3
   408         16.74         45.34      2.23       84.2
   607         17.51         60.69      4.12      104.4
   806         26.41        102.78      6.71      172.3
  1006         29.40        112.13      9.44      192.3
```

**Scaling exponents** (log-log slope between consecutive V values):

```
  V1    V2   new_step  classify  arcs   total
  59   109      3.38      3.12   1.89    3.02
 109   209      2.10      1.57   1.01    1.14
 209   408      1.61      1.25   1.63    1.30
 408   607      0.11      0.73   1.54    0.54
 607   806      1.45      1.86   1.72    1.77
 806  1006      0.48      0.39   1.54    0.50
```

At small V, the exponents are high (2-3x) due to one-time setup costs.
At larger V (400+), the dominant scaling is:

- **classify**: ~O(V^1.0 to V^1.3) — this is the #1 hotspot
- **new_step**: ~O(V^1.0 to V^1.5) — #2 hotspot
- **arcs**: ~O(V^1.5) — consistent but smaller in absolute terms
- **python search**: ~O(V^1.0) — scales linearly with #classify calls

## Where is the super-linear scaling coming from?

### 1. classify (~52% of time): O(V) calls × O(V_relevant) work per call

`classify(sid)` calls `ensure_classify`, which iterates over all *relevant*
output symbols for the DFA state. For a boundary state, there are ~V relevant
symbols. For each symbol, it:

1. Projects the NFA set to get elements for that symbol — O(|core| + |closure|)
2. Checks the `fst_univ_cache` — O(|fst_states|)
3. If cache miss, runs `bfs_universal` — O(|NFA| × |source_alphabet|)

The `fst_univ_cache` is effective (most calls hit cache after step 1), which
is why classify scales closer to O(V) than O(V^2) at large V.

**However**, the first time a DFA state is classified (step 1), every symbol
requires a full BFS universality check, which explains the ~72ms for step 1
vs ~28ms for steps 2-4.

The remaining super-linearity in classify comes from the *size of the NFA
sets* growing with V. Each BPE token adds NFA elements with unique
`extra_sym` values, so the factored NFA set has O(V) parameters. The
`project_for_symbol` call is O(|core|) per symbol, and building the
SymbolIndex for non-factored states is O(V).

### 2. new_step (~20% of time): O(V) eps-closure size

`new_step(target)` creates a fresh `PeekabooNFAMapped` and computes the
epsilon closure of the start states. For BPE at a boundary, the start
state's epsilon closure includes elements for all V tokens (each token
creates a chain of eps-input arcs from the start state). The closure is
O(V × avg_token_length) in size.

The `eps_cache` is cleared on each new_step, so the first closure is
always computed from scratch. Sorting and deduplication of the O(V)
closure elements is O(V log V).

The new_step cost is amplified because it's called once per FusedTransducedLM
step (7 calls for 6 decode steps), and each call after the first must also
compute a new start state with the extended target prefix.

### 3. arcs (~5% of time): O(V) output arcs

`arcs(sid)` calls `compute_all_arcs_factored`, which produces one successor
per source symbol. For boundary states, this yields V transitions. The
factored representation helps — instead of materializing V × |closure|
packed states, it stores (closure, params) groups. But the final
sort/dedup/intern step is still O(V).

The arcs cost at V=1000 is 3ms/call, which is O(V^1.5) scaling. The
super-linearity comes from the interning step: the `FactoredArena` must
hash-cons each successor set, and successor sets grow with V.

### 4. Python search overhead (~22% of time)

The Python search calls `score_item` ~V times per boundary step (once per
DFA state popped from the queue). Each `score_item` call:
- Calls `classify(sid)` in Rust (~0.028ms/call, cached)
- Does Python dict/set operations for carry-forward

The Python-only overhead in `score_item` is ~6-8ms per boundary step,
dominated by:
- `logaddexp` calls on `LogVector` (V symbol accumulations)
- `carry_forward` dict operations (256 byte-values × particles)
- Heap operations (V pops from priority queue)

The `_expand` function is called only ~1 time per boundary step (since
the first particle pops a Q-absorbed state, which is not expanded — it
only gets scored). The expand cost is 3-5ms, dominated by the Rust
`arcs()` call plus Python heap pushes.

## Summary of hotspots (ranked by time at V=1000)

1. **classify / universality checks** (52%): O(V) symbols checked per
   boundary DFA state. Each check is fast (~0.028ms) due to
   `fst_univ_cache`, but V of them add up. First-time checks (no cache)
   are ~5x more expensive.

2. **Python search overhead** (22%): Linear in #classify calls (which is
   linear in V). Mostly dict/heap operations — not easy to push to Rust
   without redesigning the search.

3. **new_step** (20%): Epsilon closure of O(V) NFA elements + sorting.
   Called once per step. Could potentially be amortized across steps
   since the BPE FST's start-state closure changes in a structured way.

4. **arcs** (5%): Factored arc computation is efficient. V output arcs
   produced, each interned. Minor contributor.

## Conclusion

The dominant term is **classify at ~O(V^1.0-1.3)**, not O(V^2). The
apparent super-linearity in the total likely comes from:

1. Growing NFA set sizes (more `extra_sym` parameters as V increases)
2. First-step universality cache misses (cold cache costs ~3x more)
3. Epsilon closure costs in new_step scaling ~O(V log V)

The factored representation and `fst_univ_cache` already provide
significant speedups. The main remaining opportunity would be to avoid
the per-step `new_step` epsilon-closure recomputation (incrementalize
it across steps) and to batch the ~V classify calls into a single Rust
call that avoids V Python→Rust round-trips.
