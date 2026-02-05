# Rust Acceleration: Profiling & Optimization Results

## Architecture

One Rust decomposition path:

- **`decompose`** (`decompose.rs`) — Generic path for arbitrary FSTs.
  Full precover NFA → powerset determinization → universality detection → Q/R partitioning.

## Key Fixes Applied

1. **Rc<Vec<u64>> in eps cache** (`precover.rs`): Avoids cloning on cache hits.

2. **Single-element intern fast path** (`powerset.rs`): Hashes a u64 instead of Vec for
   single-element powerset states (99% of cases in BPE).

## Key Files

- `crates/transduction-core/src/decompose.rs` — Generic decomposition
- `crates/transduction-core/src/fst.rs` — FST struct + indexes
- `crates/transduction-core/src/precover.rs` — Precover NFA + eps closure cache
- `crates/transduction-core/src/powerset.rs` — Powerset arena with single-element fast path
- `crates/transduction-core/src/py.rs` — PyO3 bindings
- `transduction/rust_bridge.py` — Python ↔ Rust conversion layer

## Test Status

90/91 tests pass. The one xfail (`test_triplets_of_doom[recursive_dfa_decomp]`) is a
pre-existing timeout in the Python `RecursiveDFADecomp` implementation, not related to Rust.

## Reports

Generated reports go in `reports/` at the project root.
