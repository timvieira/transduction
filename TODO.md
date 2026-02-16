# Documentation & Naming TODO

## Inline TODOs/XXXs from source

### fst.py

- [ ] Refactor `delta` data structure to separate input/output labels (line 66)
- [ ] Guard against state-renaming collisions in `make_total()` (line 271)
- [ ] Add assertions for bad epsilon cases in `_compose()` (line 321)
- [ ] Convert `_compose()` to lazy machine pattern (line 322)
- [ ] Convert `_augment_epsilon_transitions()` to lazy pattern (line 373)

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

### peekaboo_nonrecursive.py

- [ ] Hook up `Peekaboo`/`PeekabooStrings` to the finite test suite (line 15)

### fsa.py

- [ ] Use indexing to find nonempty set difference in `min_fast()` (line 398)
- [ ] Support NFA/epsilon arcs in `epsremove()` (line 553)

### lm/statelm.py

- [ ] Add documentation for `decode_hf_tokenizer` assumptions (line 38)
- [ ] Handle tokenizers with multiple byte representations for the same token
  (line 164)
- [ ] Implement immutable-tuple KV cache fix (line 225)
