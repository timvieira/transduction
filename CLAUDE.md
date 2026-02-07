# Transduction — Developer Guide

## Architecture

### Python Package

- `transduction/fst.py` — FST class, UniversalityFilter
- `transduction/fsa.py` — FSA class
- `transduction/peekaboo_recursive.py` — Peekaboo algorithm (recommended for autoregressive decoding)
- `transduction/peekaboo_nonrecursive.py` — Non-incremental peekaboo
- `transduction/eager_nonrecursive.py` — Reference Precover implementation
- `transduction/dfa_decomp_nonrecursive.py` — NonrecursiveDFADecomp
- `transduction/dfa_decomp_recursive.py` — RecursiveDFADecomp (diverges on some inputs)
- `transduction/token_decompose.py` — BPE-optimized fast path
- `transduction/enumeration.py` — LM-weighted path enumeration (prioritized_enumeration, importance_sampling)
- `transduction/lazy.py` — Lazy automaton framework (LazyDeterminize, EpsilonRemove)
- `transduction/goo.py` — BPE WFST builder (`bpe_wfst`), LazyPrecoverNFA, NonrecursiveDFADecomp
- `transduction/rust_bridge.py` — Python ↔ Rust conversion layer
- `transduction/examples.py` — Example FSTs for testing

### LM Integration (`transduction/lm/`)

Self-contained language model interface for use with enumeration/sampling:

- `transduction/lm/base.py` — `LMState` ABC (defines `logp_next`, `eos`, `__lshift__`, plus `advance`, `greedy_decode`, `sample_decode`)
- `transduction/lm/ngram.py` — `ByteNgramLM`, `CharNgramLM` (lightweight n-gram LMs for testing)
- `transduction/lm/statelm.py` — `StateLM`, `TokenizedLLM`, `load_model_by_name`
  - Wraps HuggingFace causal LMs with KV-cache-based incremental decoding
  - `StateLM.initial('gpt2')` loads a model and returns the initial state
  - `state << token` advances by one token (bytes), caching KV pairs
  - `state.logp_next[token]` returns log-probability (accepts bytes or int token ID)
  - `state.eos` returns the EOS token (bytes)
  - Includes inlined dependencies: `decode_hf_tokenizer`, `LazyProb`, `flatten`/`unflatten`
  - External deps: `torch`, `transformers`, `arsenal`
- `transduction/lm/transduced.py` — `TransducedLM`, `TransducedState` (pushforward of an inner LM through an FST)

### Rust Acceleration (`crates/transduction-core/`)

- `decompose.rs` — Generic FST decomposition (powerset det + universality)
- `peekaboo.rs` — Peekaboo recursive decomposition (per-symbol Q/R via step-wise BFS)
- `fst.rs` — FST struct, indexes, universality checks
- `precover.rs` — PrecoverNFA with eps closure caching (Rc<Vec<u64>> avoids cloning)
- `powerset.rs` — PowersetArena (hash-consing DFA states; single-element fast path for 99% of BPE cases)
- `py.rs` — PyO3 bindings (RustFst, RustFsa, DecompResult, PeekabooDecompResult)

## Dependencies

Library code depends only on:
- `numpy`, `torch`, `transformers` — for LM integration (`transduction/lm/`)
- `arsenal` — utility library (data structures, profiling, maths)

Test-only deps:
- `genparse` — EarleyLM grammar-based LM (used in `test_enumeration.py` small tests)

Eliminated deps (previously external, now inlined):
- `genlm` — `get_byte_vocab` replaced with local `decode_hf_tokenizer`
- `tokenization` — `StateLM`, `LazyProb`, `logsumexp` all copied/inlined

## Test Status

- `test_general.py`: 90/91 pass, 1 xfail (`test_triplets_of_doom[recursive_dfa_decomp]` — pre-existing Python timeout)
- `test_enumeration.py`: 12/12 pass (9 small grammar tests + 3 BPE-scale GPT-2 integration tests)
- 7 decomposition implementations tested: recursive_dfa_decomp, nonrecursive_dfa_decomp, peekaboo_recursive, peekaboo_nonrecursive, token_decompose, rust_decomp, rust_peekaboo

## Build Pipeline

```bash
# Build Rust extension
maturin build --release -m crates/transduction-core/Cargo.toml --interpreter python3.10

# Install
pip install --force-reinstall crates/transduction-core/target/wheels/transduction_core-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl

# Test
python -m pytest tests/test_general.py -v
python -m pytest tests/test_enumeration.py -v
```

## Reports

Generated reports go in `reports/` at the project root.
Generated data goes in `output/` at the project root (gitignored).
