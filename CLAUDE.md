# Transduction — Developer Guide

## Architecture

### Python Package

- `transduction/fst.py` — FST class
- `transduction/universality.py` — UniversalityFilter, check_all_input_universal, compute_ip_universal_states
- `transduction/fsa.py` — FSA class
- `transduction/base.py` — DecompositionResult (concrete base), DecompositionFunction (ABC), IncrementalDecomposition (ABC), AbstractAlgorithm
- `transduction/precover_nfa.py` — All PrecoverNFA variants (PrecoverNFA, TruncationMarkerPrecoverNFA, PopPrecoverNFA, TargetSideBuffer, Relevance, PeekabooLookaheadNFA, PeekabooFixedNFA)
- `transduction/peekaboo_incremental.py` — Peekaboo algorithm (recommended for autoregressive decoding)
- `transduction/peekaboo_nonrecursive.py` — Non-incremental peekaboo
- `transduction/peekaboo_dirty.py` — DirtyPeekaboo (dirty-state incremental peekaboo)
- `transduction/eager_nonrecursive.py` — Reference Precover implementation
- `transduction/dfa_decomp_nonrecursive.py` — NonrecursiveDFADecomp
- `transduction/dfa_decomp_incremental_truncated.py` — TruncatedIncrementalDFADecomp (dirty-state incremental with truncation)
- `transduction/lazy_incremental.py` — LazyIncremental decomposition (finite-language FSTs only; diverges on infinite quotients)
- `transduction/lazy_nonrecursive.py` — LazyNonrecursive decomposition (finite-language FSTs only)
- `transduction/prioritized_lazy_incremental.py` — PrioritizedLazyIncremental (finite-language, heuristic BFS)
- `transduction/token_decompose.py` — BPE-optimized fast path
- `transduction/vibes.py` — Visualization/display utilities for automata (used in notebooks)
- `transduction/enumeration.py` — LM-weighted path enumeration (prioritized_enumeration, importance_sampling)
- `transduction/lazy.py` — Lazy automaton framework (LazyDeterminize, EpsilonRemove)
- `transduction/applications/bpe.py` — BPE WFST builder (`bpe_wfst`)
- `transduction/applications/ptb.py` — PTB tokenizer FST built with pynini
- `transduction/applications/wikitext.py` — WikiText data loading (`load_wikitext`, `wikitext_detokenize`)
- `transduction/rust_bridge.py` — Python ↔ Rust conversion layer
- `transduction/examples.py` — Example FSTs for testing

### LM Integration (`transduction/lm/`)

Self-contained language model interface for use with enumeration/sampling:

- `transduction/lm/base.py` — `LMState` ABC (defines `logp_next`, `eos`, `__rshift__`, `__call__`, `greedy_decode`, `sample_decode`)
- `transduction/lm/ngram.py` — `ByteNgramLM`, `CharNgramLM` (lightweight n-gram LMs for testing)
- `transduction/lm/statelm.py` — `StateLM`, `TokenizedLLM`, `load_model_by_name`
  - Wraps HuggingFace causal LMs with KV-cache-based incremental decoding
  - `StateLM.initial('gpt2')` loads a model and returns the initial state
  - `state >> token` advances by one token (bytes), caching KV pairs
  - `state.logp_next[token]` returns log-probability (accepts bytes or int token ID)
  - `state.eos` returns the EOS token (bytes)
  - Includes inlined dependencies: `decode_hf_tokenizer`, `LazyProb`, `flatten`/`unflatten`
  - External deps: `torch`, `transformers`, `arsenal`
- `transduction/lm/transduced.py` — `TransducedLM`, `TransducedState` (pushforward of an inner LM through an FST)

### Rust Acceleration (`crates/transduction-core/`)

- `decompose.rs` — Generic FST decomposition (powerset det + universality)
- `peekaboo.rs` — DirtyPeekaboo (dirty-state incremental per-symbol Q/R via single-pass BFS)
- `incremental.rs` — DirtyDecomp (dirty-state incremental decomposition)
- `fst.rs` — FST struct, indexes, universality checks
- `precover.rs` — PrecoverNFA with eps closure caching (Rc<Vec<u64>> avoids cloning)
- `powerset.rs` — PowersetArena (hash-consing DFA states; single-element fast path for 99% of BPE cases)
- `minimize.rs` — DFA minimization
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

- `test_general.py`: 172 passed, 17 skipped
- `test_finite.py`: 40 pass
- `test_enumeration.py`: 12/12 pass (9 small grammar tests + 3 BPE-scale GPT-2 integration tests)
- `test_push_labels.py`: 30 pass
- `test_transduced.py`: 23 pass
- `test_general.py` tests the **general-case** algorithms (handle infinite quotients/remainders):
  NonrecursiveDFADecomp, TruncatedIncrementalDFADecomp, PeekabooState, PeekabooNonrecursive,
  DirtyPeekaboo, TokenDecompose, RustDecomp, RustDirtyState, RustDirtyPeekaboo.
- **Finite-only algorithms are excluded from test_general.py** and tested in
  `test_finite.py`. These diverge on FSTs with infinite quotients because they
  don't truncate the target buffer:
  - `LazyIncremental` — enumerates source *strings* (not states); diverges on infinite quotients.
  - `LazyNonrecursive` — same limitation.
  - `PrioritizedLazyIncremental` — heuristic-guided BFS; finite-only.
  - The distinguishing mechanism is **target-buffer truncation**: algorithms that
    truncate (NonrecursiveDFADecomp, TruncatedIncrementalDFADecomp, Peekaboo
    variants, Rust backends) terminate on all inputs; those that don't may diverge.
  - When adding new algorithms or test cases, classify them as general vs finite-only
    and put them in the appropriate test file.

## CRITICAL: Memory Limits

**Always set a memory limit when running scripts that may consume unbounded
memory** (benchmarks, determinization, materialization, enumeration, etc.).
Use `resource.setrlimit(resource.RLIMIT_AS, (limit, limit))` at the top of
the script or use `ulimit -v` from the shell. A reasonable default is 4 GB.
Without this, a runaway process can exhaust RAM and crash the machine.

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

## TODO

See `TODO.md` in the project root for documentation and naming improvements.

## Reports

Generated reports go in `reports/` at the project root.
Generated data goes in `output/` at the project root (gitignored).

## Important: Notebooks

Jupyter notebooks (`*.ipynb`) in this project are actively used and important.
When searching for usage of modules, functions, or code, ALWAYS include
notebooks in the search — not just `.py` files.
