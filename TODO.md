# Documentation & Naming TODO

## Critical — would confuse a new contributor

- [ ] **Explain "precover" concept.** It appears in class names, docstrings, and
  variable names across the entire codebase (`PrecoverDecomp`, `PrecoverNFA`,
  `Precover`, `LazyPrecoverNFA`) but nowhere is there a plain-English
  explanation of what a precover is. The LaTeX formula in `PrecoverDecomp`
  assumes the reader already knows. Add a conceptual explanation (e.g., in
  `base.py` or a top-level docstring).

- [ ] **Document `PeekabooState` internals (`peekaboo_incremental.py`).** Has
  ~10 undocumented internal state variables (`decomp`, `resume_frontiers`,
  `incoming`, `_univ`, etc.). The incremental computation logic is complex and
  there's no overview of how the pieces fit together.

## High — causes confusion



## Medium — naming issues

- [ ] **Document `extract_token_bytes()` and `ByteTrie`
  (`token_decompose.py`).** "Hub structure" comment is cryptic. What is a
  "hub"? When does this work vs fail?

- [ ] **Document `enumeration.py` classes.** `prioritized_enumeration` and
  `importance_sampling` lack algorithmic explanations. `Item.weight` units
  (log probability?) undocumented.

## Bugs / Limitations

### precover_nfa.py — Multi-character symbol support

- [ ] **PeekabooPrecover NFA assumes single-character symbols.** All Precover
  NFA implementations (`PrecoverNFA`, `PeekabooLookaheadNFA`,
  `PeekabooFixedNFA`, etc.) use string concatenation for the output buffer
  `ys` and index into it by character position (`ys[N]`, `ys[:N+1]`).
  This breaks for FSTs with multi-character symbol names (e.g., PTB's
  byte-value strings '84', '104', '258').  Truncation at `N+K` characters
  clips multi-character symbols, producing wrong buffer contents.

  Options: (a) switch buffer to tuple-of-symbols (preserves symbol
  boundaries but changes hashing/comparison), (b) require single-char
  symbols and remap at the caller level, (c) store a separate length
  counter alongside the string buffer.

### lm/transduced.py — Beam carry-forward for multi-state FSTs

- [ ] **TransducedLM carry-forward drops intermediate DFA states.** In
  `_compute_logp_next`, the `carry_forward` dict only captures beam items
  at Q/R states (during the expansion phase) and at resume-frontier states
  (during the drain phase).  Intermediate DFA states that are on the
  resume frontier but are expanded before the drain phase are lost from the
  carry-forward beam.  This causes `<< y` to fail with "Out of vocabulary"
  on multi-state FSTs where the beam can't reach Q/R states for some target
  symbols.  Symptom: works for 1-state FSTs (identity/copy, lowercase) but
  fails for most multi-state FSTs (duplicate, small, etc.).

  See the existing TODO comment at line 125 of transduced.py.

## Inline TODOs/XXXs from source

### fst.py

- [ ] Ensure epsilon sentinel objects (`ε_1`, `ε_2`) are truly unique (line 10)
- [ ] Add tests for `dump()` method (line 52)
- [ ] Refactor `delta` data structure to separate input/output labels (line 69)
- [ ] Add tests for `make_total()` (line 228)
- [ ] Guard against state-renaming collisions in `make_total()` (line 236)
- [ ] Add assertions for bad epsilon cases in `_compose()` (line 294)
- [ ] Convert `_compose()` to lazy machine pattern (line 295)
- [ ] Convert `_augment_epsilon_transitions()` to lazy pattern (line 346)
- [ ] Tighten `reachable()`/`coreachable()` — no need to materialize `adj`
  (line 554)
- [ ] Consider implementing `coreachable()` as `self.reverse().reachable()`
  (line 569)

### peekaboo_incremental.py

- [ ] **Incremental `_merged_incoming`**: Build the merged incoming dict
  cumulatively during BFS (at each depth, append into a single running
  dict) rather than walking the parent chain on demand.  Saves O(depth)
  work per `_qr` call.
- [ ] **Skip `_merged_incoming` in TransducedLM path**: TransducedLM never
  accesses `.quotient`/`.remainder` — it reads `.decomp`, `.dfa`,
  `.resume_frontiers`, `.preimage_stops` directly.  The `_merged_incoming`
  + `_trimmed_fsa` backward BFS is wasted work when the caller is
  TransducedLM.  Consider making Q/R FSA construction opt-in (e.g., only
  run when `.quotient` or `.remainder` is accessed).
- [ ] **Fuse `_qr` into `decompose_next()`**: When `decompose_next()` is
  called, all children's Q/R could be computed in a single pass over the
  merged incoming dict (one backward BFS from the union of all stop
  states), avoiding per-child redundant walks.
- [ ] **Truncation policy exploration**: The current policy truncates at
  N+1 (buffer length = len(target)+1).  Smarter policies (N+2, adaptive
  based on DFA size) could reduce the number of resume-frontier states
  and total BFS work across iterations.  See the note at the top of the
  file.
- [ ] Revisit whether merging `incoming` dicts across depths is correct
  (line 135)
- [ ] Fix graphviz visualization: plates show precover for next-symbol
  extension, not current target (line 159)
- [ ] Use `Integerizer` in graphviz so nodes aren't misidentified by string
  repr (line 165)
- [ ] Color active vs inactive nodes in graphviz (line 168)
- [ ] Add output ports between graphviz plates (line 174)

### peekaboo_nonrecursive.py

- [ ] Hook up `Peekaboo`/`PeekabooStrings` to the finite test suite (line 14)

### fsa.py

- [ ] Use indexing to find nonempty set difference in `min_fast()` (line 441)
- [ ] Support NFA/epsilon arcs in `epsremove()` (line 627)

### lazy_incremental.py

- [ ] `frontier()` state depends on target used for filtering — document or fix
  (line 90)
- [ ] Document the "lazy frontier machine" arcs (line 108)
- [ ] `candidates` vs inner loop duplication (line 155)

### lm/statelm.py

- [ ] Add documentation for `decode_hf_tokenizer` assumptions (line 36)
- [ ] Handle tokenizers with multiple byte representations for the same token
  (line 162)

### tests/test_finite.py

- [ ] Unify test frameworks or bring recursive testing strategy to finite tests
  (line 6)

### examples.py

- [x] Dump pynini-based machine to Python code to remove pynini dependency
  (done — `newspeak2()` is the hand-coded version; pynini source kept as reference)
