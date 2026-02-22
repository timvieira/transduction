# Transduction — Developer Guide

## CRITICAL: Do NOT commit or push unless explicitly asked

Never run `git commit` or `git push` unless the user explicitly requests it.
Do not auto-commit after finishing work.

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
- `transduction/precover.py` — Precover (reference decomposition implementation)
- `transduction/eager_nonrecursive.py` — EagerNonrecursive algorithm
- `transduction/dfa_decomp_nonrecursive.py` — NonrecursiveDFADecomp
- `transduction/dfa_decomp_incremental_truncated.py` — TruncatedIncrementalDFADecomp (dirty-state incremental with truncation)
- `transduction/lazy_incremental.py` — LazyIncremental decomposition (finite-language FSTs only; diverges on infinite quotients)
- `transduction/lazy_nonrecursive.py` — LazyNonrecursive decomposition (finite-language FSTs only)
- `transduction/prioritized_lazy_incremental.py` — PrioritizedLazyIncremental (finite-language, heuristic BFS)
- `transduction/viz.py` — Visualization/display utilities for automata (used in notebooks)
- `transduction/enumeration.py` — LM-weighted path enumeration (prioritized_enumeration, importance_sampling)
- `transduction/pynini_ops.py` — Pynini/OpenFST-backed reference P(y)/Q(y)/R(y) via composition and projection
- `transduction/lazy.py` — Lazy automaton framework (LazyDeterminize, EpsilonRemove)
- `transduction/lazy_precover_dfa.py` — LazyPrecoverDFA: on-demand DFA with integer packing, hash-consing, eps-closure caching
- `transduction/token_decompose.py` — TokenDecompose, TokenPeekabooHelper: position-set DFA states O(N) for BPE-like FSTs; FusedTransducedLM `helper="token"` backend
- `transduction/trie_dispatch.py` — TrieDispatchDFADecomp: trie-based dispatch for decomposition
- `transduction/applications/bpe.py` — BPE WFST builder (`bpe_wfst`)
- `transduction/applications/ptb.py` — PTB tokenizer FST built with pynini
- `transduction/applications/wikitext.py` — WikiText data loading (`load_wikitext`, `wikitext_detokenize`)
- `transduction/rust_bridge.py` — Python ↔ Rust conversion layer; also provides `RustPeekabooState`, `RustLazyPeekabooDFA`, `RustTokenPeekabooDFA`
- `transduction/util.py` — Shared utilities: `Integerizer`, `logsumexp`, `LogVector` (mutable log-space accumulator), `LogDistr` (immutable log-probability distribution), `memoize`, `timelimit`, `set_memory_limit`, `memory_limit`, `sample`, `colors`
- `transduction/examples.py` — Example FSTs for testing

### LM Integration (`transduction/lm/`)

Self-contained language model interface for use with enumeration/sampling:

- `transduction/lm/base.py` — `LM` (ABC), `LMState` (ABC; defines `logp_next`, `eos`, `__rshift__`, `__call__`, `greedy_decode`, `sample_decode`)
- `transduction/lm/ngram.py` — `ByteNgramLM`, `CharNgramLM` (lightweight n-gram LMs for testing)
- `transduction/lm/statelm.py` — `StateLM`, `TokenizedLLM`, `load_model_by_name`
  - Wraps HuggingFace causal LMs with KV-cache-based incremental decoding
  - `StateLM.initial('gpt2')` loads a model and returns the initial state
  - `state >> token` advances by one token (bytes), caching KV pairs
  - `state.logp_next[token]` returns log-probability (accepts bytes or int token ID)
  - `state.eos` returns the EOS token (bytes)
  - Includes inlined dependencies: `HfTokenizerVocab`, `LazyProb`, `flatten`/`unflatten`
  - External deps: `torch`, `transformers`, `arsenal`
- `transduction/lm/transduced.py` — `TransducedLM`, `TransducedState` (pushforward of an inner LM through an FST; defaults to Rust backend)
- `transduction/lm/fused_transduced.py` — `FusedTransducedLM`, `FusedTransducedState` (single-pass interleaved decomposition + LM search; `helper=` for pluggable backends: `"rust"`, `"python"`, `"token"`)
- `transduction/lm/reference_transduced.py` — `ReferenceTransducedLM` (ground-truth transduced LM via Precover; enumerates Q/R languages exactly; finite-relation FSTs only)
- `transduction/lm/pynini_transduced.py` — `PyniniTransducedLM` (pynini-backed transduced LM with particle-based inference)

### Rust Acceleration (`crates/transduction-core/`)

- `decompose.rs` — Generic FST decomposition (powerset det + universality)
- `peekaboo.rs` — DirtyPeekaboo (dirty-state incremental per-symbol Q/R via single-pass BFS)
- `incremental.rs` — DirtyDecomp (dirty-state incremental decomposition)
- `fst.rs` — FST struct, indexes, universality checks
- `precover.rs` — PrecoverNFA with eps closure caching (Rc<Vec<u64>> avoids cloning)
- `powerset.rs` — PowersetArena (hash-consing DFA states; single-element fast path for 99% of BPE cases)
- `minimize.rs` — DFA minimization
- `lazy_precover.rs` — LazyPrecoverDFA: lazy DFA over precover NFA with on-demand expansion and arc caching
- `token_decompose.rs` — Token-level decomposition using compact bitsets for position sets
- `token_peekaboo.rs` — Position-set-quotiented peekaboo DFA for FusedTransducedLM
- `py.rs` — PyO3 bindings (RustFst, RustFsa, DecompResult, PeekabooDecompResult, TokenPeekabooDFA)

## Dependencies

Core library (`transduction/`):
- `numpy` — numerical operations (enumeration, LM integration)
- `graphviz` — FSA/FST visualization
- `IPython` — notebook display utilities (`display_table`)

LM integration (`transduction/lm/`):
- `torch`, `transformers` — HuggingFace causal LM wrappers

Visualization / benchmarking (notebooks, `notes/`):
- `matplotlib` — plotting

Test-only deps:
- `pytest` — test runner

Optional deps:
- `pynini` — PTB tokenizer FST construction (`applications/ptb.py`)
- `nltk` — PTB tokenizer testing
- `datasets` — WikiText data loading
- `tqdm` — progress bars in benchmarks

Eliminated deps (previously external, now inlined):
- `arsenal` — `Integerizer`, `colors`, `memoize`, `timelimit`, `timeit`, `sample`, `set_memory_limit`, `memory_limit` inlined into `util.py`
- `genlm` — `get_byte_vocab` replaced with local `HfTokenizerVocab`
- `tokenization` — `StateLM`, `LazyProb`, `logsumexp` all copied/inlined
- `LogpNext` (formerly in `lm/base.py`) — replaced by `LogDistr` in `util.py`

## Test Status

- `test_general.py`: 470 passed (10 implementations × 47 test cases)
- `test_finite.py`: 119 passed
- `test_pynini_ops.py`: 115 passed
- `test_transduced.py`: 106 passed
- `test_lazy.py`: 100 passed
- `test_fst.py`: 56 passed
- `test_enumeration.py`: 55 passed
- `test_push_labels.py`: 35 passed
- `test_fsa.py`: 28 passed
- `test_lazy_precover_dfa.py`: 26 passed
- `test_is_functional.py`: 26 passed
- `test_lazy_peekaboo_dfa.py`: 23 passed
- `test_ngram.py`: 22 passed
- `test_ptb_nltk.py`: 4 passed
- `test_make_total.py`: 3 passed
- `test_statelm_kv_cache.py`: 3 passed
- **Total: 1191 tests across 16 files (1189 passed, 2 xfailed)**
- `test_general.py` tests the **general-case** algorithms (handle infinite quotients/remainders):
  NonrecursiveDFADecomp, TruncatedIncrementalDFADecomp, TrieDispatchDFADecomp,
  PeekabooState, PeekabooNonrecursive, DirtyPeekaboo, RustDecomp, RustDirtyState,
  RustDirtyPeekaboo, PyniniNonrecursiveDecomp.
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

## Style Conventions

- **`Str[T]` for symbol strings**: Use `Str[Token]`, `Str[int]`, etc. instead of
  `tuple[Token, ...]`, `tuple[int, ...]` when the type represents a string
  (sequence of symbols) in the automata-theory sense.  `Str` is defined in
  `transduction/util.py` as `Str = tuple[_T, ...]`.  This does **not** apply
  to non-string tuples (arc lists, cons-cell histories, etc.).

- **`LogDistr` and `LogVector` over raw dicts**: Prefer `LogDistr[T]` for
  immutable log-probability distributions and `LogVector[T]` for mutable
  log-space accumulators instead of plain `dict` or `defaultdict`.  Both are
  defined in `transduction/util.py`.

- **`State` for automaton states**: Use the `State` type alias (defined in
  `transduction/util.py` as `State: TypeAlias = Any`) in type annotations for
  automaton states, even though it resolves to `Any`.  This conveys intent and
  makes signatures more readable.

## CRITICAL: Memory Limits

**Always set a memory limit when running scripts that may consume unbounded
memory** (benchmarks, determinization, materialization, enumeration, etc.).
Use `from transduction.util import set_memory_limit; set_memory_limit(4)` at
the top of the script (argument is in GB), or use `ulimit -v` from the shell.
For interactive/notebook use, `memory_limit(gb)` is a context manager that
restores the original limits on exit. A reasonable default is 4 GB. Without
this, a runaway process can exhaust RAM and crash the machine.

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
