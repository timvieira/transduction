# Peekaboo Decomposition: Rust vs Python Benchmark

## Summary

The Rust implementation of `peekaboo_decompose` achieves **3x-25x speedup** over the
Python `peekaboo_recursive.Peekaboo` across all test examples.

- Typical speedup: **7-15x** for most examples
- Best speedup: **25x** (parity, target_len=5)
- Most complex example (newspeak2): **10-22x** speedup
- Absolute times: **5-80 microseconds** for most examples, **1-3ms** for newspeak2

## Timing Comparison

| Example | target_len | Rust | Python | Speedup |
|---------|-----------|------|--------|---------|
| replace_5 | 0 | 19 us | 79 us | 4.2x |
| replace_5 | 4 | 83 us | 283 us | 3.4x |
| delete_b | 10 | 27 us | 249 us | 9.3x |
| samuel | 5 | 17 us | 255 us | 14.7x |
| sdd1 | 5 | 24 us | 573 us | 24.1x |
| duplicate_5 | 5 | 81 us | 283 us | 3.5x |
| number_comma | 3 | 78 us | 951 us | 12.2x |
| newspeak2 | 0 | 1.5 ms | 27.3 ms | 17.7x |
| newspeak2 | 3 | 3.0 ms | 67.0 ms | 22.1x |
| lookahead | 6 | 15 us | 122 us | 8.4x |
| weird_copy | 3 | 18 us | 381 us | 21.6x |
| triplets_of_doom | 13 | 27 us | 278 us | 10.3x |
| parity_ab | 5 | 13 us | 318 us | 25.0x |

## Incremental Sequence (Autoregressive Simulation)

Simulates decoding: call peekaboo for targets '', 'a', 'ab', 'abc', ...

| Example | Calls | Rust total | Python total | Speedup |
|---------|-------|-----------|-------------|---------|
| triplets_of_doom | 14 | 400 us | 3.6 ms | 8.9x |
| duplicate_5 | 6 | 241 us | 851 us | 3.5x |
| parity_ab | 6 | 76 us | 1.0 ms | 13.8x |
| lookahead | 7 | 106 us | 833 us | 7.8x |

## Where Time is Spent (Rust)

### newspeak2 (most complex example, target_len=3)

| Phase | Time | % |
|-------|------|---|
| init (alphabet, ip_universal) | 0.08 ms | 2.6% |
| BFS (all steps) | 2.99 ms | 96.1% |
| extract (backward BFS) | 0.03 ms | 1.0% |

BFS breakdown:
| Component | Time | % of BFS |
|-----------|------|----------|
| universality checks | 2.71 ms | 90.6% |
| compute_arcs | 0.12 ms | 4.0% |
| intern | 0.03 ms | 1.0% |

**Universality checking dominates BFS time** (90%+), which is expected since
each DFA state requires checking projected universality for each relevant symbol.

### Small examples (< 100 us total)

For most examples, total time is 10-80 microseconds. The init phase (computing
`ip_universal_states`) takes 0-20 us, BFS takes 10-60 us, and extraction is negligible.

## DFA Structure

| Example | target_len | Steps | BFS visited | Arena states | Max pset | Avg pset |
|---------|-----------|-------|-------------|-------------|---------|---------|
| newspeak2 | 3 | 4 | 126 | 429 | 2 | 1.5 |
| triplets_of_doom | 13 | 14 | 20 | 19 | 1 | 1.0 |
| duplicate_5 | 5 | 6 | 8 | 7 | 1 | 1.0 |
| replace_5 | 4 | 5 | 30 | 26 | 1 | 1.0 |
| number_comma | 3 | 4 | 20 | 17 | 2 | 1.3 |

Key observations:
- **Powerset states are tiny**: max size 1-2 NFA states per DFA state
- **triplets_of_doom stabilizes**: only 20 states visited regardless of target length (steps 3+ add 0 new states)
- **newspeak2 grows linearly**: ~31 states per step, 429 arena states at depth 3

## Per-Step BFS Patterns

Most examples show one of two patterns:

1. **Constant per step** (replace_5, delete_b): Each step visits the same number of states.
   The DFA grows linearly with target length.

2. **Convergent** (triplets_of_doom, lookahead, samuel): After a few steps, the frontier
   becomes empty (no resume states). Later steps add no work.

## Optimization Opportunities

1. **Cache `ip_universal_states` across calls**: Currently recomputed each call (~0.02-0.13ms).
   For repeated peekaboo calls on the same FST, this could be precomputed once.

2. **Incremental step reuse**: The current API recomputes all steps from scratch.
   An incremental API that reuses the arena and incoming graph across calls would
   avoid redundant work for the autoregressive use case.

3. **Universality fast path**: When `all_input_universal` (BPE-like FSTs), every
   final-containing projected state is automatically universal. This avoids the witness
   lookup and BFS entirely.

## Usage

```python
from transduction.rust_bridge import RustPeekaboo
from transduction import examples

fst = examples.newspeak2()
peekaboo = RustPeekaboo(fst)

# Get Q(y·z) and R(y·z) for every next symbol z
decomps = peekaboo('ba')
for symbol, decomp in decomps.items():
    print(f"'{symbol}': Q={len(decomp.quotient.states)}, R={len(decomp.remainder.states)}")
```
