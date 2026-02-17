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
| **Specialized** | | | | | | | | | |
| `TokenDecompose` | | | | position-set | ✓ | structural | ✓ | | `all_input_universal` |
| **String-enumerating** | | | | | | | | | |
| `LazyIncremental` | ✓ | | | | | BFS | **no** | | finite-only |
| `PrioritizedLazyIncremental` | ✓ | | | | | BFS+mono | **no** | | finite-only; heuristic |
| `LazyNonrecursive` | | | | | | BFS | borderline | | `max_steps` |
| **Rust** | | | | | | | | | |
| `RustDecomp` | | | | N | ✓ | ✓ | ✓ | ✓ | rust ext |
| `RustDirtyState` | ✓ | | ✓ | N | ✓ | ✓ | ✓ | ✓ | rust ext |
| `RustDirtyPeekaboo` | ✓ | ✓ | ✓ | N+K | ✓ | ✓ | ✓ | ✓ | rust ext |
| `RustLazyPeekabooDFA` | ✓ | ✓ | ✓ | N+K | ✓ | ✓ | ✓ | ✓ | rust ext; lazy DFA for TransducedLM |

†`TruncatedIncrementalDFADecomp.decompose_next()` creates per-symbol overlay children sharing the parent's clean arcs — batched via overlays, not via a single BFS pass like Peekaboo.

## Key columns explained

- **Incremental (`>>`)** — Extends by one target symbol, reusing prior computation (vs rebuilding from scratch).
- **Batched next-sym** — Computes Q/R for *all* possible next symbols in one pass (the peekaboo idea).
- **Dirty-state** — On `>>`, only re-expands DFA states affected by the extension (frontier/border states). "Resume-frontiers" is the peekaboo variant; "skip-steps" means the DirtyPeekaboo approach of skipping already-completed peekaboo steps.
- **Buffer truncation** — Truncates the target-side buffer at depth N, N+1, or N+K to bound the state space. This is *the* mechanism that separates algorithms that terminate on general FSTs from those that don't.
- **State-based** — Explores automaton states (finite) vs source strings (potentially infinite).
- **UnivFilter** — Uses the cascading `UniversalityFilter` (all-input-universal fast path -> ip-universal witnesses -> monotonicity caches -> BFS fallback) vs a simpler bare BFS.
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
