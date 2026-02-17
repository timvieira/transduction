# Documentation & Naming TODO

- [x] Improve code coverage (fst.py: 59% → 99%; test_fst.py: 50 tests; test_general.py: 286→352; test_finite.py: 70→113; test_transduced.py: 47→55)

## Inline TODOs/XXXs from source

## Completed (2026-02-16)

- [x] **Split Precover into its own module** (`precover.py`)
- [x] **Expose Rust DirtyPeekaboo DFA for TransducedLM** — `RustPeekabooState`, `RustLazyPeekabooDFA` in `rust_bridge.py`; TransducedLM now defaults to Rust backend
- [x] **Fix carry-forward prefix-domination bug** in both `TransducedLM` and `FusedTransducedLM` — root-family tracking prevents duplicates when carry-forward particles are prefix-dominated
- [x] **Rich notebook display** — `_repr_html_` on `TransducedState` and `FusedTransducedState`; unified visualization in `viz.py`
- [x] **Hopcroft minimization improvement** — find-index + block_members grouping (Python & Rust)
- [x] **N-gram LM EOS from training data** — `ByteNgramLM`/`CharNgramLM` learn EOS from per-instance training data
- [x] **Rename `RustFusedHelper` → `RustLazyPeekabooDFA`** — the lazy DFA is a general-purpose interface
- [x] **Centralize memory/time limits** in `util.py`; add example FST tests
- [x] **PeekabooStrings `__call__` interface** backed by `decompose_next`
- [x] **Remove noisy `eprintln!` progress logging** from Rust decompose BFS

### fst.py

- [ ] Refactor `delta` data structure to separate input/output labels (line 66)
- [ ] Convert `_compose()` to lazy machine pattern (line 320)
- [ ] Convert `_augment_epsilon_transitions()` to lazy pattern (line 379)

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
- [ ] Fix graphviz visualization: plates show precover for next-symbol
  extension, not current target (line 110)
- [ ] Use `Integerizer` in graphviz so nodes aren't misidentified by string
  repr (line 116)
- [ ] Color active vs inactive nodes in graphviz (line 119)
- [ ] Add output ports between graphviz plates (line 125)


### fsa.py

- [ ] Support NFA/epsilon arcs in `epsremove()` (line 553)

### lm/transduced.py

- [x] **Check fused_transduced.py for prefix-dominated carry-forward bug**: The
  same carry-forward prefix-domination issue fixed in `TransducedLM._compute_logp_next`
  also existed in `FusedTransducedLM`.  Fixed with root-family tracking in
  `_FusedSearch` (`_add_carry`, `_root_of`, `_carried`).  Tested by
  `TestFusedCarryForwardNoDuplicates`.

### lm/statelm.py

- [ ] Handle tokenizers with multiple byte representations for the same token
  (line 185)
- [ ] Implement immutable-tuple KV cache fix (line 246)
