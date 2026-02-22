# Periodic Table of Decomposition Algorithms

## Feature Matrix

| | Incremental (`>>`) | Batched next-sym | Dirty-state | Buffer truncation | State-based | UnivFilter | General (inf quotients) | Rust | Special req |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Reference** | | | | | | | | | |
| `Precover` | | | | N | ✓ | ✓ | ✓ | | |
| **DFA Decomposition** | | | | | | | | | |
| `NonrecursiveDFADecomp` | | | | N | ✓ | ✓ | ✓ | | |
| `TruncatedIncrementalDFADecomp` | ✓ | ✓† | ✓ | N | ✓ | ✓ | ✓ | | |
| **Peekaboo** | | | | | | | | | |
| `PeekabooState` | ✓ | ✓ | resume-frontiers | N+K | ✓ | ✓ | ✓ | | |
| `Peekaboo` (nonrecursive) | | ✓ | | N+1 | ✓ | | ✓ | | |
| `DirtyPeekaboo` | ✓ | ✓ | skip-steps | N+K | ✓ | ✓ | ✓ | | |
| **String-enumerating** | | | | | | | | | |
| `LazyIncremental` | ✓ | | | | | BFS | **no** | | finite-only |
| `PrioritizedLazyIncremental` | ✓ | | | | | BFS+mono | **no** | | finite-only; heuristic |
| `LazyNonrecursive` | | | | | | BFS | borderline | | `max_steps` |
| **Rust** | | | | | | | | | |
| `RustDecomp` | | | | N | ✓ | ✓ | ✓ | ✓ | rust ext |
| `RustDirtyState` | ✓ | | ✓ | N | ✓ | ✓ | ✓ | ✓ | rust ext |
| `RustDirtyPeekaboo` | ✓ | ✓ | ✓ | N+K | ✓ | ✓ | ✓ | ✓ | rust ext |
| `RustLazyPeekabooDFA` | ✓ | ✓ | ✓ | N+K | ✓ | ✓ | ✓ | ✓ | rust ext; lazy DFA for FusedTransducedLM |
| **Token-level / Lazy** | | | | | | | | | |
| `LazyPrecoverDFA` | | | | N | ✓ | ✓ | ✓ | ✓† | on-demand DFA, int packing, hash-consing |
| `TokenDecompose` | | | | N | ✓ | ✓ | ✓ | ✓ | O(N) position-set states; BPE-like only |
| `TokenPeekabooHelper` | ✓ | ✓ | | N+K | ✓ | ✓ | ✓ | ✓ | FusedLM `helper="token"`; BPE+PTB |
| `TrieDispatchDFADecomp` | | | | N | ✓ | ✓ | ✓ | | trie-based dispatch |
| **Pynini reference** | | | | | | | | | |
| `PyniniNonrecursiveDecomp` | | | | N | ✓ | pynini | ✓ | | pynini dep |

†`TruncatedIncrementalDFADecomp.decompose_next()` creates per-symbol overlay children sharing the parent's clean arcs — batched via overlays, not via a single BFS pass like Peekaboo.

## Key columns explained

- **Incremental (`>>`)** — Extends by one target symbol, reusing prior computation (vs rebuilding from scratch).
- **Batched next-sym** — Computes Q/R for *all* possible next symbols in one pass (the peekaboo idea).
- **Dirty-state** — On `>>`, only re-expands DFA states affected by the extension (frontier/border states). "Resume-frontiers" is the peekaboo variant; "skip-steps" means the DirtyPeekaboo approach of skipping already-completed peekaboo steps.
- **Buffer truncation** — Truncates the target-side buffer at depth N, N+1, or N+K to bound the state space. This is *the* mechanism that separates algorithms that terminate on general FSTs from those that don't.
- **State-based** — Explores automaton states (finite) vs source strings (potentially infinite).
- **UnivFilter** — Uses the cascading `UniversalityFilter` (all-input-universal fast path $\to$ ip-universal witnesses $\to$ monotonicity caches $\to$ BFS fallback) vs a simpler bare BFS.
- **General (inf quotients)** — Terminates on FSTs where the quotient language is infinite (e.g., `triplets_of_doom`).

## Natural groupings

The two most important axes are **buffer truncation** and **state-based exploration** — together they determine whether an algorithm terminates on general FSTs. Every algorithm in the "General = ✓" column has both. The string-enumerating algorithms (`Lazy*`) lack both and are restricted to finite languages.

Within the general-case algorithms, the key design choices are:
1. **Incremental vs non-incremental** — amortizes work across decoding steps
2. **Batched vs per-symbol** — the peekaboo insight that one BFS can classify all next symbols
3. **Dirty-state vs full rebuild** — avoids re-expanding the stable interior of the DFA

## Precover NFA variants

Each algorithm builds over a specific NFA variant defined in `precover_nfa.py`. The NFA variant determines the buffer structure, truncation behavior, and state format.

| NFA Variant | Class in `precover_nfa.py` | Buffer | Truncation | State format | Used by |
|---|---|---|---|---|---|
| `PrecoverNFA` | ✓ | tuple push, grows to target prefix | at N (bounded) | `(i, ys)` | Precover, NonrecursiveDFADecomp, LazyNonrecursive |
| `TruncationMarkerPrecoverNFA` | ✓ | tuple push, tracks info loss | at N; `truncated` flag distinguishes eps vs non-eps overflow | `(i, ys, truncated)` | Precover (push-truncated mode) |
| `TargetSideBuffer` | ✓ | unconditional tuple accumulation | none (unbounded) | `(i, ys)` | (archived) |
| `PeekabooLookaheadNFA` | ✓ | tuple push, K-lookahead | at N+K with truncation bit | `(i, ys, truncated)` | PeekabooState, DirtyPeekaboo |
| `PeekabooFixedNFA` | ✓ | tuple push, fixed N+1 | at N+1, no truncation bit | `(i, ys)` | Peekaboo (nonrecursive) |

All NFA variants use **tuples of symbols** for the target-side buffer `ys`, supporting multi-character output symbols (e.g., PTB byte-value strings like `'84'`). Buffer operations use tuple slicing and concatenation (`ys + (a,)`, `ys[:n]`), which correctly preserves symbol boundaries.

Note: `TruncatedIncrementalDFADecomp` and `RustDirtyState` use a frontier-marker approach conceptually similar to `TruncationMarkerPrecoverNFA` — tracking whether each NFA state is at the frontier (`|ys| == N`) to enable dirty-state detection. In the Rust implementation, this is handled directly in `incremental.rs` via `max_bufpos` tracking rather than a separate NFA class.

## Transduced Language Models

The decomposition algorithms above compute *structural* Q/R decompositions. The transduced LM layer sits on top, combining a decomposition with an inner LM to compute the pushforward distribution $P(\text{target}) = \sum_{x \in T^{-1}(\text{target})} P_{\text{inner}}(x)$. All implementations conform to the `LM`/`LMState` interface: `state >> y` advances by one target symbol, `state.logp_next[y]` returns $\log P(y \mid \text{target so far})$.

### Feature Matrix

| | Incremental | Inference | Decomp strategy | Carry-forward | General (inf quotients) | Rust | Finite-only |
|---|:---:|---|---|:---:|:---:|:---:|:---:|
| `TransducedLM` | ✓ | particle beam + best-first search | pre-computed peekaboo | ✓ | ✓ | ✓ (default) | |
| `FusedTransducedLM` | ✓ | particle beam + best-first search | lazy DFA built during search | ✓ | ✓ | ✓ (required) | |
| `ReferenceTransducedLM` | ✓ | exact enumeration of Q/R languages | Precover (full materialization) | | | | ✓ |

### How they work

**`TransducedLM`** (`lm/transduced.py`) — The primary approximate inference engine. Maintains K particles (source-prefix hypotheses), each tracking a DFA state and an inner LM state. Per target step:
1. The peekaboo decomposition classifies DFA states as Q(y), R(y), or preimage.
2. Best-first search pops particles by weight. Quotient particles are *absorbed* — their full weight contributes to y's score (exact marginalization over all continuations). Remainder particles contribute $\text{weight} \times P_{\text{inner}}(\text{EOS})$. Non-classified particles are expanded by source symbols weighted by $P_{\text{inner}}$.
3. After the expansion budget (`max_expansions`) is exhausted, remaining queued particles are scored without expansion.
4. Carry-forward passes particles at Q/R/resume-frontier states to the next step; top-K pruning bounds the beam.

**`FusedTransducedLM`** (`lm/fused_transduced.py`) — Same particle-beam search as `TransducedLM`, but the decomposition DFA is not pre-computed. Instead, a lazy helper builds the powerset DFA during search — only states reachable via high-probability source paths are materialized. The `helper=` parameter selects the backend: `"rust"` (default, `RustLazyPeekabooDFA`), `"python"` (`PythonLazyPeekabooDFAHelper`), or `"token"` (`TokenPeekabooHelper`, position-set-quotiented for BPE-like FSTs). Carry-forward uses sentinel states (dfa_state=None) that are resolved via source-path replay in the next step. logp agreement with TransducedLM: max |diff| = 0.000287 (PTB), 0.000000 (BPE).

**`ReferenceTransducedLM`** (`lm/reference_transduced.py`) — Ground-truth implementation for validation. Uses the `Precover` decomposition to enumerate the Q and R languages exactly, then sums inner LM probabilities over all source strings. Only terminates when Q and R are finite (finite-relation FSTs). Not suitable for production use — exponential in the size of the Q/R languages.

### Key design dimensions

- **Pre-computed vs fused decomposition** — `TransducedLM` computes the full peekaboo decomposition (DFA + Q/R classification) before running LM search; `FusedTransducedLM` interleaves both, building only the DFA fragment the LM actually explores. Pre-computed is simpler and reusable across LMs; fused avoids wasted work on unreachable states.
- **Carry-forward** — Particles at Q/R/resume-frontier states survive across target steps, avoiding redundant re-exploration. This is the key to amortizing work across the autoregressive decode. `ReferenceTransducedLM` lacks carry-forward: it recomputes from scratch at each step.
- **Quotient absorption** — When a particle reaches a quotient state for symbol y, its entire weight (marginalizing over all continuations) contributes to y's score. This is the main source of variance reduction over naive sampling.

## Prefix Probability Estimation (enumeration.py)

These are non-incremental baselines for estimating the prefix probability P(output starts with target) for a *fixed* target string. Unlike the transduced LMs above, they do not advance one symbol at a time — they take a complete target prefix and estimate its probability under the inner LM. Included primarily for pedagogical purposes.

### Feature Matrix

| | Method | Uses decomposition | Early termination at Q | Bounded | Finite-only |
|---|---|---|:---:|:---:|:---:|
| `prioritized_enumeration` | best-first search | ✓ | ✓ (absorbed) | `max_steps` | |
| `importance_sampling` | LM-guided sampling | ✓ | ✓ (absorbed) | `max_length` | |
| `crude_importance_sampling` | LM-guided sampling | | | `max_length` | |

### How they work

All three search through the precover DFA (the automaton recognizing source strings whose output starts with the target prefix), weighted by the inner LM.

**`prioritized_enumeration`** — Best-first search (max-heap by LM weight). Pops the highest-weight item and classifies it: Q states are absorbed (full prefix probability), R states contribute $\text{weight} \times P(\text{EOS})$. Non-terminal states are expanded by source symbols. Equivalent to the per-step search in `TransducedLM`, but for a single fixed target rather than incrementally.

**`importance_sampling`** — Samples a single source path through the precover DFA, proposing transitions proportional to the inner LM. At Q states the sample is absorbed; at R states, EOS can be chosen. The return value carries a log importance weight (the sum of proposal normalizers) for use in Monte Carlo estimation.

**`crude_importance_sampling`** — Same sampling strategy but without the Q/R decomposition. Operates on the raw (determinized) precover NFA — the sample terminates only at final states (where the source has produced exactly the target prefix and stopped). This is the naive baseline that demonstrates why decomposition matters: without quotient absorption, many more samples are needed.
