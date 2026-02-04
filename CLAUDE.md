# Rust Acceleration: Profiling & Optimization Results

## Architecture

Two Rust decomposition paths:

1. **`token_decompose`** (`token_decompose.rs`) — For BPE-like FSTs where `all_input_universal=true`.
   Collapses each token into a single transition advancing buf_pos by the token's byte length.
   DFA states are position subsets {0..target_len} instead of NFA state subsets.
   Typically O(N) DFA states (N = target length).

2. **`decompose`** (`decompose.rs`) — Generic path for arbitrary FSTs.
   Full precover NFA → powerset determinization → universality detection → Q/R partitioning.

Dispatch in `py.rs:164`: uses `token_decompose` when `all_input_universal`, else `decompose`.

## Performance (Full GPT-2, 50K tokens)

### token_decompose path (current)

| target_len | total | init | BFS | DFA states | arcs |
|-----------|-------|------|-----|-----------|------|
| 100 | 32ms | 32ms | 0.1ms | 101 | 503 |
| 500 | 26ms | 25ms | 0.7ms | 501 | 1756 |
| 1000 | 35ms | 32ms | 2.4ms | 1001 | 4018 |
| 2000 | 38ms | 29ms | 8.8ms | 2001 | 7811 |
| 4000 | 57ms | 29ms | 28ms | 4001 | 12530 |

Init cost (~28ms) is token extraction + trie building. Could be cached across calls.

### vs old generic decompose (before token_decompose)

| target_len | old total | new total | speedup |
|-----------|----------|----------|---------|
| 50 | 1502ms | 0.3ms BFS | ~5000x |
| 80 | 5500ms | 0.2ms BFS | ~27000x |
| 1000 | impossible | 2.4ms BFS | ∞ |

## Key Fixes Applied

1. **`all_input_universal` precomputation** (`fst.rs`): O(N) check during FST construction.
   For BPE FSTs, input projection always accepts Σ*. Eliminates O(N²) `is_universal` sub-BFS.

2. **Partial match in trie** (`token_decompose.rs`): Tokens whose byte sequences extend beyond
   the remaining target now correctly match (excess bytes consumed post-target).

3. **Zero-length token handling** (`token_decompose.rs`): Tokens with ε output (e.g., delete_b)
   create self-loops in the DFA.

4. **Rc<Vec<u64>> in eps cache** (`precover.rs`): Avoids cloning on cache hits.

5. **Single-element intern fast path** (`powerset.rs`): Hashes a u64 instead of Vec for
   single-element powerset states (99% of cases in BPE).

## Key Files

- `crates/transduction-core/src/token_decompose.rs` — BPE-optimized decomposition
- `crates/transduction-core/src/decompose.rs` — Generic decomposition
- `crates/transduction-core/src/fst.rs` — FST + `check_all_input_universal()`
- `crates/transduction-core/src/precover.rs` — Precover NFA + eps closure cache
- `crates/transduction-core/src/powerset.rs` — Powerset arena with single-element fast path
- `crates/transduction-core/src/py.rs` — PyO3 bindings + dispatch logic
- `transduction/rust_bridge.py` — Python ↔ Rust conversion layer

## Test Status

50/51 tests pass. The one failure (`test_triplets_of_doom[recursive_dfa_decomp]`) is a
pre-existing timeout in the Python `RecursiveDFADecomp` implementation, not related to Rust.

## Reports

Generated reports go in `reports/` at the project root.

## Next Optimization Targets

1. **Cache init cost**: The ~28ms init (token extraction + trie) is rebuilt each call.
   Cache it on the RustFst object for repeated decompositions.
2. **Incremental decomposition**: Process target symbols one at a time, reusing previous
   DFA structure (like `RecursiveDFADecomp` pattern).
