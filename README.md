# Transduction

A library for computing **precover decompositions** of finite state transducers (FSTs), enabling constrained decoding for language models.

Given an FST `f` and a target prefix **y** already generated, transduction computes for each possible next output symbol `z` the set of source strings that transduce through `f` to produce **y**`z`... — partitioned into a **quotient** (sources that can continue) and a **remainder** (sources that have terminated):

$$\mathcal{P}(\mathbf{y}) = \mathcal{Q}(\mathbf{y})\Sigma^* \sqcup \mathcal{R}(\mathbf{y})$$

Both Q and R are returned as finite state automata (FSAs).

## Use cases

- **Constrained decoding** — force a language model to respect output format constraints (JSON, CSV, regex, etc.)
- **Tokenizer-aware constraints** — handle BPE/WordPiece tokenization within the constraint framework
- **Structured generation** — enforce grammars or schemas on LM output
- **LM-weighted enumeration** — best-first search over valid source strings, weighted by LM log-probabilities

## Installation

Requires Python >= 3.10.

```bash
pip install -e .
```

### Rust acceleration (optional)

For 3-25x speedups, build the Rust extension with [maturin](https://www.maturin.rs/):

```bash
pip install maturin
maturin build --release -m crates/transduction-core/Cargo.toml --interpreter python3.10
pip install --force-reinstall crates/transduction-core/target/wheels/transduction_core-*.whl
```

## Quick start

### Define an FST

```python
from transduction import FST, EPSILON

fst = FST()
fst.add_start(0)
fst.add_stop(0)

# An FST that replaces 'bad' with 'ungood' (a la 1984)
fst.add_arc(0, 'a', 'a', 0)  # copy most characters
fst.add_arc(0, 'b', 'u', 1)  # 'b' might start 'bad' -> 'ungood'
fst.add_arc(1, 'a', 'n', 2)
fst.add_arc(2, 'd', 'g', 3)
fst.add_arc(3, EPSILON, 'o', 4)
fst.add_arc(4, EPSILON, 'o', 5)
fst.add_arc(5, EPSILON, 'd', 0)
# ... (plus identity arcs for other characters)
```

### Compute a decomposition

```python
from transduction import Precover

result = Precover(fst, target='ab')
print(result.quotient)   # FSA: source strings that can continue after producing 'ab'
print(result.remainder)  # FSA: source strings that terminate after producing 'ab'
```

### Peekaboo: batched next-symbol prediction

The **peekaboo** algorithm computes decompositions for *all* possible next symbols in a single pass — the key primitive for autoregressive constrained decoding:

```python
from transduction.peekaboo_recursive import Peekaboo

peekaboo = Peekaboo(fst)

# Get Q(y·z) and R(y·z) for every possible next symbol z, in one call
decomps = peekaboo('ab')

for symbol, decomp in decomps.items():
    print(f"Next symbol '{symbol}': Q has {len(decomp.quotient.states)} states")
```

Peekaboo supports incremental extension via the `>>` operator, reusing computation across decoding steps.

### Rust backend

Drop-in replacements for the Python algorithms:

```python
from transduction.rust_bridge import RustDecomp, RustPeekaboo

# Generic decomposition (Rust)
result = RustDecomp(fst, target='ab')

# Peekaboo (Rust) — 3-25x faster than Python
rust_peekaboo = RustPeekaboo(fst)
decomps = rust_peekaboo('ab')
```

### LM-weighted enumeration

Combine decomposition with a language model to enumerate or sample valid source strings:

```python
from transduction.lm import StateLM
from transduction.enumeration import prioritized_enumeration, importance_sampling

# Load GPT-2 (or any HuggingFace causal LM)
lm = StateLM.initial('gpt2')

# Best-first search weighted by LM log-probabilities
pe = prioritized_enumeration(lm, fst, target='the', max_steps=20)
for item in pe.quotient_terms:
    print(f"Source: {item.source}, weight: {item.weight:.3f}")

# Or sample paths proportional to LM probability
sampler = importance_sampling(lm, fst, target='the')
sample = sampler.sample(max_length=50)
```

## Algorithms

| Algorithm | Module | Incremental | Notes |
|-----------|--------|:-----------:|-------|
| `Precover` | `eager_nonrecursive.py` | No | Reference implementation; full powerset determinization |
| `NonrecursiveDFADecomp` | `dfa_decomp_nonrecursive.py` | No | Clean reference; rebuilds from scratch each call |
| `RecursiveDFADecomp` | `dfa_decomp_recursive.py` | Yes | Supports `>>` but diverges on unbounded-buffer FSTs |
| `Peekaboo` | `peekaboo_recursive.py` | Yes | **Recommended.** Batches all next-symbol decompositions; truncation ensures termination |
| `PeekabooNonrecursive` | `peekaboo_nonrecursive.py` | No | Simpler peekaboo variant without `>>` |
| `TokenDecompose` | `token_decompose.py` | No | BPE-optimized fast path (requires `all_input_universal` FST) |
| `RustDecomp` | `rust_bridge.py` | No | Rust generic decomposition |
| `RustPeekaboo` | `rust_bridge.py` | No | Rust peekaboo (3-25x speedup) |

### Choosing an algorithm

- **Autoregressive decoding (token by token):** Use `RustPeekaboo` for best performance, or `Peekaboo` (Python) if the Rust extension is unavailable.
- **BPE tokenizers:** Check `check_all_input_universal(fst)` first — if true, `TokenDecompose` gives massive speedups (5000x+).
- **One-shot decomposition:** Use `RustDecomp` or `Precover`.

## Project structure

```
transduction/              Python package
  fst.py                   FST class, UniversalityFilter
  fsa.py                   FSA class
  base.py                  PrecoverDecomp, AbstractAlgorithm base classes
  peekaboo_recursive.py    Peekaboo algorithm (recommended)
  peekaboo_nonrecursive.py Non-incremental peekaboo
  eager_nonrecursive.py    Reference Precover implementation
  dfa_decomp_*.py          DFA-based decomposition variants
  token_decompose.py       BPE-optimized fast path
  enumeration.py           LM-weighted path enumeration
  goo.py                   BPE WFST builder, LazyPrecoverNFA
  lazy.py                  Lazy automaton framework (on-demand determinization)
  rust_bridge.py           Python <-> Rust conversion layer
  examples.py              Example FSTs for testing
  lm/                      Language model integration
    base.py                LMState ABC (logp_next, eos, <<, advance, greedy/sample decode)
    ngram.py               ByteNgramLM, CharNgramLM (lightweight n-gram LMs)
    statelm.py             StateLM: incremental LM state with KV caching
    transduced.py          TransducedLM: pushforward of an inner LM through an FST

crates/transduction-core/  Rust acceleration (PyO3)
  src/
    fst.rs                 FST struct with CSR storage + auxiliary indexes
    precover.rs            Precover NFA with epsilon closure caching
    powerset.rs            PowersetArena (hash-consing DFA states)
    decompose.rs           Generic decomposition
    peekaboo.rs            Peekaboo recursive (Rust port)
    py.rs                  PyO3 bindings

tests/                     Test suite
  test_general.py          Decomposition correctness (90 tests, 7 implementations)
  test_enumeration.py      Enumeration + BPE-scale integration tests (12 tests)
reports/                   Algorithm analysis and benchmarks
```

## Testing

```bash
# Core decomposition tests (90 pass, 1 xfail across 7 implementations)
python -m pytest tests/test_general.py -v

# Enumeration + BPE-scale integration tests (12 pass)
python -m pytest tests/test_enumeration.py -v
```

102/103 tests pass across 7 decomposition implementations. The single expected failure is a pre-existing timeout in `RecursiveDFADecomp` on an adversarial input (`triplets_of_doom`).

### Dependencies for full test suite

- `genparse` — required for grammar-based LM tests in `test_enumeration.py`
- `transformers` — required for GPT-2 BPE-scale tests

## Built-in examples

The `examples` module provides FSTs useful for testing and exploration:

```python
from transduction import examples

examples.lookahead()        # FST with epsilon-output lookahead arcs
examples.triplets_of_doom() # Adversarial case: (aaa|bbb)* copy transducer
examples.newspeak2()        # Orwellian: replaces 'bad' -> 'ungood'
examples.delete_b()         # Deletes 'b', replaces 'a' -> 'A' (infinite quotients)
examples.parity({'a','b'})  # Outputs parity bit of input length
examples.togglecase()       # Swaps upper/lowercase
examples.duplicate({'a','b'}, K=3)  # Triplicates each symbol
examples.number_comma_separator(Domain={'0','1',',',' '})  # Comma disambiguation
```

## Authors

Tim Vieira, Samuel Kiegeland, Vésteinn Snæbjarnarson, Ryan Cotterell
