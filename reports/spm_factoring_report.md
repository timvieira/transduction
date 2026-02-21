# SPM Regime Analysis and Factored Decomposition

## 1. Background: Strict Prefix Monotonicity (SPM)

An FST is **strictly prefix monotone (SPM)** if, for every accepted
source string, the output grows monotonically with the input — each input
symbol produces at least one output symbol, and no "buffering" or "lookahead"
is needed. Formally (cf. arXiv 2412.03719, Section 3.3): an FST is SPM if
the remainder is always empty and the quotient requires no lookahead.

### Connection to the Precover

The precover NFA `P(y)` for target prefix `y` has states `(q, buf)` where `q`
is an FST state and `buf` is a prefix of the target that has been produced so
far. The **position** of an NFA state is `|buf|` — how far through the target
this path has advanced.

For SPM FSTs:
- All NFA states at any given BFS frontier share the same position (they've
  all consumed the target up to the same point).
- The remainder is empty (no FST state is "stuck" behind the target cursor).
- The quotient generalizes the covering: `Q(y) = {x : f(x) starts with y}`.

In principle, an SPM-aware algorithm could track DFA states as sets of
*positions* `{0, ..., |y|}` instead of sets of `(fst_state, position)` pairs.
This drops the FST-state dimension entirely, yielding O(N) states instead of
O(|fst_states| * N).

### The `all_input_universal` property

An FST has `all_input_universal = True` when its input projection from the
start set accepts `Sigma*`. This is a sufficient condition for SPM behavior in
the decomposition: every final DFA state is universal, so the remainder is
always empty. BPE tokenizer FSTs have this property.

## 2. Locally SPM: Definition and Measurement

Not all FSTs are globally SPM, but many are **locally SPM** at most DFA states.

**Definition.** A DFA state `S = frozenset({(q1, buf1), (q2, buf2), ...})` is
**locally SPM** (position-uniform) if all NFA elements share a single position:
`|{|buf| : (q, buf) in S}| = 1`.

When a state is locally SPM:
- The state simplifies to `(position, frozenset_of_fst_states)`.
- No cross-position tracking is needed — cheaper hashing and smaller memory
  footprint.
- Successor computation can skip buffer reconstruction.

**Position-factored** is a weaker property: the state is factorable into
`{position -> frozenset_of_fst_states}` as a Cartesian product. The
**factoring ratio** `|S| / (|positions| * |fst_states|)` measures this:
1.0 = fully factored, <1.0 = sparse (some position-state combinations are
missing).

## 3. The Factored Representation

### Standard vs. Factored

The standard DFA state representation in `NonrecursiveDFADecomp` is a flat
frozenset of NFA states:

```
Standard:  frozenset({(q1, y[:3]), (q2, y[:5]), (q3, y[:3]), ...})
```

The factored representation groups by position:

```
Factored:  {3: frozenset({q1, q3}), 5: frozenset({q2})}
```

### Properties

- **Exact**: The factored representation is lossless — it can be converted back
  to the flat frozenset given the target string (since `buf = target[:pos]`).
- **Compact**: When position-uniform (SPM regime), the state is just
  `(pos, frozenset_of_fst_states)`. Hash computation is cheaper since we avoid
  hashing buffer tuples.
- **Canonical**: The `FState` class sorts entries by position for deterministic
  hashing and equality.

### When it helps (theory vs. practice)

In theory, the factored representation should help proportionally to the fraction
of position-uniform states. In practice (see Section 5), the overhead of
constructing `FState` projections offsets the theoretical savings in Python.
The factored representation is most useful as a **caching key** for universality
checks, not as a replacement for the standard DFA state representation.

Position-uniformity rates across example FSTs:
- **100% position-uniform** (replacement, doom, delete_b, newspeak2, parity_copy):
  all states reduce to `(position, fst_state_set)`.
- **Mixed regime** (samuel_example 85.2%, backticks_to_quote 79.2%): most states
  are position-uniform with occasional multi-position states.
- **Heavily non-uniform**: rare for small example FSTs (anbn at 65.6% is worst).

**Important caveat — BPE is NOT position-uniform**: Despite being globally SPM
(`all_input_universal = True`), BPE's epsilon-heavy structure creates
multi-position DFA states during powerset construction — only **11.1%** of DFA
states are position-uniform for BPE vocab=500-1000 (see Section 5.4).

This was a key incorrect assumption in the original plan. The `all_input_universal`
property guarantees trivial universality (every final state is universal), but
says nothing about position-uniformity of the powerset DFA states. BPE tokens
are represented as chains of epsilon-input/byte-output arcs:
```
() --eps/b1--> (b1,) --eps/b2--> (b1,b2) --token_id/eps--> ()
```
Epsilon closure from any DFA state simultaneously reaches NFA elements at
different depths in different token chains, mixing positions within a single
DFA state. A position-only algorithm could avoid this entirely by bypassing
`PrecoverNFA` and tracking positions directly against the BPE trie — a
fundamentally different approach that the factored representation cannot
replicate within the `PrecoverNFA` framework.

## 4. Empirical Results

### Summary Table

Results from `reports/spm_regime_analysis.py` running over exhaustive short
target strings (length 3-5, depending on alphabet size):

| FST | DFA States | Uniform% | Factored% | Avg Ratio | Max |S| |
|-----|-----------|----------|-----------|-----------|---------|
| newspeak2 | 131,484 | **100.0%** | 100.0% | 1.0000 | 4 |
| samuel_example | 1,085 | 85.2% | 85.2% | 0.9258 | 2 |
| number_comma_separator | 1,457 | **99.1%** | 99.1% | 0.9955 | 2 |
| doom(K=3) | 262 | **100.0%** | 100.0% | 1.0000 | 2 |
| delete_b | 119 | **100.0%** | 100.0% | 1.0000 | 1 |
| mystery1 | 112 | **100.0%** | 100.0% | 1.0000 | 1 |
| mystery7 | 112 | **100.0%** | 100.0% | 1.0000 | 1 |
| replace(a->x,b->y,c->z) | 2,004 | **100.0%** | 100.0% | 1.0000 | 1 |
| anbn | 224 | 65.6% | 65.6% | 0.8229 | 3 |
| backticks_to_quote | 1,068 | 79.2% | 79.2% | 0.8961 | 2 |
| parity_copy | 90 | **100.0%** | 100.0% | 1.0000 | 4 |

### Key observations

1. **Most FSTs are 100% position-uniform.** 7 out of 11 FSTs tested have every
   DFA state in the SPM regime. These include copy-like FSTs (replace, doom),
   deletion FSTs (delete_b), and context-dependent rewriting (newspeak2).

2. **Non-uniform states are rare even in non-SPM FSTs.** The samuel_example
   (85.2%), backticks_to_quote (79.2%), and number_comma_separator (99.1%)
   all have the vast majority of states in the SPM regime.

3. **The worst case is anbn at 65.6%.** This is an inherently ambiguous FST
   where different source paths produce different amounts of output at different
   rates. Even here, two-thirds of states are position-uniform.

4. **Position-uniform == position-factored.** In all tested cases, these two
   metrics are identical. When a state has multiple positions, the factoring
   ratio drops below 1.0 (the position-state combinations are sparse, not a
   full Cartesian product). This means the factored representation has no
   overhead vs. the flat representation in non-SPM states.

5. **Powerset sizes are small.** Max |S| ranges from 1 to 4 across all FSTs.
   The factored representation's hashing advantage is modest for such small
   sets.

6. **These results do not extend to BPE.** The example FSTs above have simple
   epsilon structure. BPE's deep epsilon chains (one per token byte) create
   multi-position DFA states during powerset construction — see Section 5.4
   for BPE-specific measurements showing only 11.1% position-uniformity.

### Factoring ratio distribution

For non-uniform FSTs, the factoring ratio clusters at two values:
- **1.0** for position-uniform states (the majority)
- **0.5** for 2-position states with 2 FST states but only 2 elements
  (one per position, not the full 2x2=4 product)

No intermediate values were observed, suggesting that multi-position states
arise from genuine ambiguity (different source paths producing different output
amounts) rather than incomplete coverage.

## 5. Factored Decomposition: Implementation and Benchmarks

The `FactoredDecomp` class in `transduction/factored_decompose.py` implements
a hybrid approach combining the standard `PrecoverNFA.det()` pipeline with
factored universality caching and dirty-state incremental extension.

### Design

1. **`FState` class**: Lightweight projection of flat frozenset DFA states into
   `{position -> frozenset_of_fst_states}` for universality caching only.

2. **Hybrid arc computation**: Uses the standard `PrecoverNFA.det()` pipeline
   (which benefits from `EpsilonRemove` closure caching) rather than custom
   position-level arc computation. This avoids reinventing optimized BFS.

3. **`FactoredUniversalityFilter`**: Wraps `UniversalityFilter` with frontier
   caching by FST state set. For DFA states where all NFA elements are at
   position N (pure frontier), universality depends only on the FST state set.
   This cache avoids redundant BFS universality checks across different targets
   that share the same frontier state sets.

4. **Incremental `>>` operator**: Dirty-state reuse across target extensions,
   following `TruncatedIncrementalDFADecomp`'s pattern. Only frontier states
   (those touching the target boundary) and their border predecessors are
   re-expanded. Epsilon closure caches and frontier universality caches persist
   across extensions.

5. **`decompose_next()`**: Builds each branch independently (non-incremental)
   to avoid mutating shared parent state. The `>>` operator is available for
   sequential single-branch extension.

6. **Correctness**: Verified to produce identical Q/R on 12 example FSTs across
   18,689 non-incremental test cases, 786 sequential `>>` cases, and 786
   recursive `decompose_next` cases — all passing.

### Design iterations

Three versions were implemented and benchmarked:

**v1 (initial prototype)**: Pure factored BFS with `FState` as the primary state
representation. Used `to_frozenset_with_target()` for universality checks.
Correct but no performance benefit — the overhead of factored state construction
cancelled out the savings.

**v2 (position-level epsilon closure)**: Custom epsilon closure BFS operating
directly on `(position, fst_state_set)` pairs. Skipped buffer reconstruction
for position-uniform states. **0.81x geomean speedup** (slower) — the custom
BFS was less efficient than the well-optimized `LazyDeterminize`/`EpsilonRemove`
caching in the standard pipeline.

**v3 (hybrid — current)**: Standard `PrecoverNFA.det()` for arc computation +
`FState` for frontier universality caching + dirty-state incremental `>>`.
This separates the "arc computation" concern (where the standard pipeline excels)
from the "universality caching" concern (where factored representation helps).

### Benchmark results

#### Non-incremental (fresh build per target)

Each target built independently via `FactoredDecomp(fst, target)`:

| FST | Targets | Standard | Factored | Speedup |
|-----|---------|----------|----------|---------|
| replace(xyz) | 120 | 0.004s | 0.008s | 0.50x |
| delete_b | 30 | 0.001s | 0.001s | 0.60x |
| samuel_example | 120 | 0.008s | 0.014s | 0.60x |
| doom(K=3) | 30 | 0.003s | 0.004s | 0.65x |
| mystery1 | 30 | 0.001s | 0.002s | 0.67x |
| mystery7 | 30 | 0.001s | 0.002s | 0.67x |
| newspeak2 | 18,278 | 17.3s | 18.6s | 0.93x |
| anbn | 30 | 0.003s | 0.004s | 0.71x |
| backticks_to_quote | 39 | 0.005s | 0.007s | 0.74x |
| parity_copy | 14 | 0.001s | 0.002s | 0.74x |
| **Geometric mean** | | | | **0.67x** |

Non-incremental `FactoredDecomp` is **slower** than `NonrecursiveDFADecomp`
due to per-state `FState` construction overhead. The frontier universality cache
doesn't help when each target is built independently (no cache sharing).

#### Incremental (`>>` operator)

Sequential extension: `root >> y1 >> y2 >> y3`, building depth-3 target trees:

| FST | Targets | Standard | Fac (fresh) | Fac (>>) | >>/std |
|-----|---------|----------|-------------|----------|--------|
| replace(xyz) | 39 | 0.001s | 0.002s | 0.002s | 0.50x |
| delete_b | 14 | 0.000s | 0.001s | 0.000s | 0.72x |
| samuel_example | 39 | 0.003s | 0.004s | 0.002s | **1.71x** |
| doom(K=3) | 14 | 0.002s | 0.002s | 0.003s | 0.44x |
| mystery1 | 14 | 0.001s | 0.001s | 0.001s | 0.76x |
| mystery7 | 14 | 0.001s | 0.001s | 0.001s | 0.74x |
| **newspeak2** | **18,278** | **16.2s** | **17.1s** | **2.0s** | **8.22x** |
| anbn | 14 | 0.002s | 0.002s | 0.010s | 0.21x |
| backticks_to_quote | 39 | 0.005s | 0.007s | 0.006s | 0.90x |
| parity_copy | 14 | 0.002s | 0.002s | 0.002s | 1.06x |

The incremental `>>` provides **8.22x speedup on newspeak2** (26-letter alphabet,
131K DFA states). The dirty-state approach re-expands only frontier and border
states, avoiding redundant DFA construction for the vast interior.

**Key observations**:

1. **Incremental >> is the main optimization.** The 8.22x speedup on newspeak2
   dwarfs any benefit from factored state representation. The dirty-state reuse
   pattern is what matters for autoregressive decoding.

2. **Small FSTs don't benefit.** For FSTs with small alphabets and few DFA states,
   the per-state overhead of FState construction dominates. The frontier cache
   has nothing to amortize.

3. **anbn is adversarial for incremental.** At 0.21x, the incremental approach
   hurts anbn because most states are non-uniform (65.6%), creating large dirty
   sets that negate the reuse benefit. The invalidation/re-expansion overhead
   exceeds the savings from incremental construction.

4. **samuel_example benefits modestly.** At 1.71x, the combination of moderate
   alphabet size and non-trivial DFA provides enough work to amortize the
   overhead.

5. **Frontier universality cache helps newspeak2.** With 131K DFA states and
   deep target trees, the same frontier FST state sets recur across many targets,
   making the cache effective.

### 5.4 BPE and PTB benchmark

Benchmark on real tokenizer FSTs (`reports/bpe_ptb_benchmark.py`), using
subsampled BPE (GPT-2) at various vocab sizes and PTB (pynini-built, 296
states). Targets are byte-encoded English text at various lengths, with a
30-second per-call timeout.

#### BPE results

| FST | Len | #Tgt | Standard | Fac(fresh) | Fac(>>) | Fac/Std | >>/Std |
|-----|-----|------|----------|------------|---------|---------|--------|
| BPE(100) | 3 | 4 | 0.022s | 0.021s | 0.037s | 1.04x | 0.59x |
| BPE(500) | 3 | 10 | 1.46s | 1.53s | 2.00s | 0.96x | 0.73x |
| BPE(500) | 5 | 10 | 1.71s | 1.56s | 2.57s | 1.09x | 0.67x |
| BPE(500) | 8 | 10 | 1.40s | 1.48s | 3.44s | 0.94x | 0.41x |
| BPE(500) | 10 | 10 | 1.50s | 1.34s | 3.67s | 1.12x | 0.41x |
| BPE(500) | 15 | 10 | 1.47s | 1.61s | 5.74s | 0.91x | 0.26x |
| BPE(1000) | 3 | 10 | 8.11s | 8.35s | 10.29s | 0.97x | 0.79x |
| BPE(1000) | 5 | 10 | 7.91s | 7.16s | 13.31s | 1.10x | 0.59x |
| BPE(1000) | 8 | 10 | 7.24s | 6.98s | 16.90s | 1.04x | 0.43x |
| BPE(1000) | 10 | 10 | 7.80s | 7.44s | 19.45s | 1.05x | 0.40x |
| BPE(1000) | 15 | 10 | 6.56s | 6.02s | 21.26s | 1.09x | 0.31x |

Geomean speedups:

| FST | Fac(fresh) | Fac(>>) |
|-----|------------|---------|
| BPE(100) | 1.04x | 0.59x |
| BPE(500) | 1.00x | 0.46x |
| BPE(1000) | 1.05x | 0.48x |

#### PTB results

| FST | Len | #Tgt | Standard | Fac(fresh) | Fac(>>) | Fac/Std | >>/Std |
|-----|-----|------|----------|------------|---------|---------|--------|
| PTB(296 st) | 3 | 4 | 1.59s | 1.55s | 4.31s | 1.03x | 0.37x |

Geomean: Fac(fresh) 1.03x, Fac(>>) 0.37x.

#### SPM regime for BPE

| FST | DFA states | Uniform | Uniform% |
|-----|-----------|---------|----------|
| BPE(500) | 27 | 3 | **11.1%** |
| BPE(1000) | 27 | 3 | **11.1%** |

**This is a key finding**: BPE's epsilon-heavy structure (each token is a chain
of epsilon-output arcs followed by a token-consuming arc) creates DFA states
where NFA elements sit at *different* positions. Despite BPE being globally SPM
(`all_input_universal = True`), the powerset construction mixes positions within
DFA states. Only 11.1% of DFA states are position-uniform, making the frontier
universality cache nearly useless.

#### Analysis

**Why the incremental `>>` hurts on BPE/PTB:**

1. **N separate DFA constructions**: Each `>>` step creates a new
   `PrecoverNFA.det()` call. A length-15 target requires 15 separate DFA
   constructions vs. 1 fresh build. The overhead of creating `PrecoverNFA`,
   `LazyDeterminize`, and managing dirty-state bookkeeping far exceeds any
   savings from incremental reuse.

2. **Trivial universality**: BPE has `all_input_universal = True`, so
   universality is O(1) — just check `is_final`. The frontier universality
   cache (the main novel optimization) provides **zero benefit**.

3. **Low position-uniformity**: At 11.1%, almost no DFA states are
   position-uniform. The cache key (`frozenset_of_fst_states` at frontier
   position) rarely applies.

4. **Incremental overhead scales with target length**: The `>>/Std` ratio
   degrades from 0.79x at length 3 to 0.26-0.31x at length 15 for BPE(500-1000).
   Each additional byte adds another full DFA construction step.

**Why fresh factored is at parity**: The per-state `FState.from_nfa_states()`
overhead is small (a `defaultdict` grouping + frozenset construction), roughly
offsetting the occasional frontier cache hit. The net effect is noise around
1.0x.

**Contrast with newspeak2 (8.22x)**: newspeak2's speedup came from a specific
confluence of factors absent in BPE/PTB:
- 26-letter target alphabet → dense depth-3 target tree with 18,278 branches
  sharing DFA interior
- Non-trivial universality checks (BFS required, not just `is_final`)
- 100% position-uniform → frontier cache maximally effective
- Large DFA (131K states) → dirty-state reuse avoids significant work

## 6. Conclusions

### The factored decomposition: mixed results

The `FactoredDecomp` and its incremental `>>` operator provide **no speedup**
on real tokenizer FSTs (BPE, PTB). Fresh factored is at parity with
`NonrecursiveDFADecomp`; incremental `>>` is 2-4x slower. The position-uniformity
hypothesis (Section 5) was refuted — only 11.1% of BPE DFA states are
position-uniform due to epsilon-heavy trie structure. The incremental `>>`
operator does help for context-dependent rewriting FSTs (8.22x on newspeak2).

### Recommendations

1. **For context-dependent rewriting FSTs** (newspeak-like):
   `FactoredDecomp` with `>>` is worthwhile for large-alphabet, large-DFA cases.

2. **For BPE/PTB**: The factored representation provides no benefit. Use the
   standard pipeline with Rust acceleration instead.
