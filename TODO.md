# Documentation & Naming TODO

- [ ] 100% code coverage

## Refactoring

- [ ] **Extract dirty-state infrastructure**: `dfa_decomp_incremental_truncated.py`
  and `peekaboo_dirty.py` share near-identical dirty-state management (frontier
  tracking, border identification, invalidation, epsilon-cache eviction,
  `_incoming` dict maintenance).  Extract a shared `DirtyStateManager` to reduce
  duplication and ensure bug fixes apply to both.

- [ ] **Extract peekaboo symbol classification helper**: The "extract relevant
  symbols → check universality → classify Q/R" loop is repeated across
  `peekaboo_incremental.py`, `peekaboo_nonrecursive.py`, and `peekaboo_dirty.py`
  with near-identical code.  A shared `_classify_symbols(frontier, target, N, fst)`
  helper would be clean and low-risk.

## Inline TODOs/XXXs from source

### fst.py

- [ ] Convert `_compose()` to lazy machine pattern (line 381)
- [ ] Convert `_augment_epsilon_transitions()` to lazy pattern (line 440)

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
  extension, not current target (line 114)
- [ ] Use `Integerizer` in graphviz so nodes aren't misidentified by string
  repr (line 120)
- [ ] Color active vs inactive nodes in graphviz (line 123)
  - [ ] Add output ports between graphviz plates (line 129)

### lm/reference_transduced.py

- [ ] Structure Q/R strings into a trie to reduce redundant inner LM state
  updates (line 84)

### fsa.py & fst.py

- [ ] Rename `stop` → `accepting` in FSA/FST classes ("final" and "stop" suggest
  you can't continue past the state, which is misleading; "accepting" is standard)

### lm/huggingface_lm.py

- [ ] Handle tokenizers with multiple byte representations for the same token
  (line 174)
- [x] ~~Implement immutable-tuple KV cache fix for DynamicCache~~ (resolved via `_clone_dynamic_cache()` deep-cloning)

### peekaboo.rs — FactoredArena

The `FactoredArena` replaces `PowersetArena` in `DirtyPeekaboo` and
`LazyPeekabooDFA`.  It stores boundary DFA states (off-target NFA elements) as
`(FST_closure, params_list)` groups instead of the full cartesian product,
reducing per-state memory from O(|V| × |closure|) to O(|closure| + |V|).

**Benchmark results** (synthetic BPE FSTs, 5 decomposition steps on
"The quick brown", 2026-02-22):

```
Memory (peak RSS after 5 steps):

  V       OLD         NEW       Reduction
  100     448 MB      448 MB      0%
  250     462 MB      460 MB      0%
  500     485 MB      478 MB      1%
 1000     551 MB      516 MB      6%
 2000     813 MB      626 MB     23%
 5000    2324 MB     1334 MB     43%

Wall-clock (avg ms/step):

  V       OLD         NEW       Change
  100     0.4         0.8       +100% (noise at <1ms)
  250     8.7         8.9         +2%
  500    26.7        22.5        -16%
 1000   124         119          -4%
 2000   826         972         +18%
 5000  3428        4241         +24%
```

Memory wins are significant at V >= 2000 (23-43% reduction).  Wall-clock is
roughly neutral: slightly faster at V=500-1000 but ~20% slower at V=2000-5000
due to `normalize_for_step` cloning, fingerprint-based interning overhead, and
more complex arc computation.

- [ ] **Profiling**: The BFS/universality check dominates wall-clock at large V,
  not state representation.  Profile `compute_all_arcs_factored` vs
  `bfs_universal` to identify the actual bottleneck.
- [ ] **Avoid `normalize_for_step` cloning**: Instead of cloning + normalizing
  on every access, store a `max_bufpos` in the arena per state and lazily
  normalize only when `max_bufpos <= step_n`.  Or normalize in-place in the
  arena (mutation) when step_n changes.
- [ ] **Fingerprint collisions**: `FactoredArena` uses fingerprint hashing +
  equality check.  For large arenas, profile whether fingerprint collision
  chains become a bottleneck.
