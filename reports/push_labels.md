# FST Output Label Pushing

## Overview

`push_labels()` pushes output labels toward initial states to reduce output delay.
This means the precover NFA advances buffer positions earlier, which can reduce
powerset state diversity during decomposition.

## DFA State Comparison (Rust `rust_decompose`)

### Standard test FSTs (sorted alphabet target)

| FST | target_len | DFA states (before) | DFA states (after) | reduction |
|-----|-----------|--------------------|--------------------|-----------|
| mystery1 | 5 | 6 | 6 | 0% |
| mystery1 | 10 | 6 | 6 | 0% |
| mystery1 | 20 | 6 | 6 | 0% |
| mystery7 | 5 | 6 | 6 | 0% |
| mystery7 | 10 | 6 | 6 | 0% |
| mystery7 | 20 | 6 | 6 | 0% |
| mystery8 | 5 | 5 | 5 | 0% |
| mystery8 | 10 | 5 | 5 | 0% |
| mystery8 | 20 | 5 | 5 | 0% |
| samuel_example | 5 | 3 | 3 | 0% |
| samuel_example | 10 | 3 | 3 | 0% |
| samuel_example | 20 | 3 | 3 | 0% |
| lookahead | 5 | 2 | 1 | 50% |
| lookahead | 10 | 2 | 1 | 50% |
| lookahead | 20 | 2 | 1 | 50% |
| small | 5 | 1 | 1 | 0% |
| small | 10 | 1 | 1 | 0% |
| small | 20 | 1 | 1 | 0% |

### Domain-appropriate targets

| FST | target | DFA states (before) | DFA states (after) | arcs (before) | arcs (after) | max powerset (before) | max powerset (after) |
|-----|--------|--------------------|--------------------|--------------|-------------|----------------------|---------------------|
| mystery1 | cxxxx | 12 | 12 | 19 | 19 | 1 | 1 |
| mystery1 | cxxxxxxxxx | 22 | 22 | 39 | 39 | 1 | 1 |
| mystery7 | cxxxx | 12 | 12 | 19 | 19 | 1 | 1 |
| mystery7 | cxxxxxxxxx | 22 | 22 | 39 | 39 | 1 | 1 |
| mystery8 | cxxxx | 8 | 8 | 12 | 12 | 1 | 1 |
| lookahead | xxxxx | 5 | 5 | 4 | 4 | 1 | 1 |

### Custom gap_creator FSTs

| FST | target_len | DFA states (before) | DFA states (after) | reduction |
|-----|-----------|--------------------|--------------------|-----------|
| gap_creator(1) | 10 | 12 | 12 | 0% |
| gap_creator(2) | 10 | 13 | 13 | 0% |
| gap_creator(3) | 10 | 14 | 14 | 0% |

## Analysis

**`lookahead`** shows a consistent 50% DFA state reduction with sorted-alphabet
targets. This FST has genuine output delay: `0 --a/eps--> 1 --a/x--> ...` where
the output `x` is delayed until the second input symbol. After pushing, the output
appears earlier, reducing the number of distinct DFA states the decomposition tracks.

For most other test FSTs, the Rust decomposition already achieves
`max_powerset_size=1` (every DFA state is a singleton NFA state set), so output
pushing has no additional effect — the powerset is already minimal.

The benefit of `push_labels()` is expected to be more significant for:
- Larger FSTs (e.g., real BPE tokenizers) where output delay creates genuine gaps
  between buffer positions across different NFA paths
- FSTs with long epsilon-output chains that delay buffer advancement
- Cases where the powerset construction produces states with `max_powerset_size > 1`

## Correctness

All 12 example FSTs pass relation-preservation tests (up to length 8).
Idempotence verified for mystery1, mystery7, mystery8, samuel_example, lookahead.
Full push-labels test suite: 35 tests pass (`test_push_labels.py`).
