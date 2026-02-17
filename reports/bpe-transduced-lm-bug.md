# TransducedLM fails on BPE-style FSTs

## Summary

`TransducedLM` and `FusedTransducedLM` fail at step 2 when the FST has
the arc structure used by `bpe_wfst`: epsilon-input arcs emit output,
and the non-epsilon input arc (consuming the source symbol) produces
epsilon output.  There are two separate bugs triggered by this structure.

## Affected components

### Bug 1: `PeekabooState` incremental BFS (`peekaboo_incremental.py`)

When `resume_frontiers` is empty at step N, `PeekabooState.__rshift__`
produces a completely empty decomposition at step N+1 because the child
BFS has no seed states.

The Rust `RustPeekabooState` (backed by `RustDirtyPeekabooDecomp`) does
**not** have this bug — it recomputes from scratch via dirty-state
tracking and correctly produces Q/R at every step regardless of empty
resume_frontiers.

### Bug 2: `TransducedLM` / `FusedTransducedLM` carry-forward (`transduced.py`, `fused_transduced.py`)

`TransducedState.__rshift__` (line 315) seeds the next step's beam
**only** from carry-forward particles.  These particles sit at DFA state
IDs from step N's Q states, but in step N+1's DFA those state IDs are
dead ends (self-loops).  No new particles are seeded from the new DFA
start states, so the beam search finds nothing.

This bug is independent of Bug 1: even with the correct Rust
decomposition backend, TransducedLM fails because the carry-forward
particles can never reach the new step's Q/R states.

### Not affected

All non-incremental decomposition algorithms produce correct results on
BPE-style FSTs:

- `Precover`
- `NonrecursiveDFADecomp`
- `PeekabooNonrecursive`
- `RustDirtyPeekaboo` / `RustDecomp`
- `DirtyPeekaboo`
- `TruncatedIncrementalDFADecomp`

The Rust incremental decomposition (`RustPeekabooState` via
`RustDirtyPeekabooDecomp`) also produces correct Q/R/resume_frontiers
at every step.

## Minimal reproduction

Two FSTs that compute the *same* transduction `{x->(a,a), y->(b,b)}`
but differ in where the non-epsilon input arc sits:

**Duplicate-style** (works) -- non-epsilon input arc produces the first
output symbol; epsilon-input arcs produce the rest:

```
(0) --x:a--> (x,0)      # source symbol consumed, first output emitted
(x,0) --eps:a--> (0)     # remaining output via epsilon chain
```

**BPE-style** (fails) -- epsilon-input arcs produce all output; the
non-epsilon input arc produces epsilon:

```
(0) --eps:a--> (a,)         # output emitted on epsilon-input arc
(a,) --eps:a--> (a,a)       # more output on epsilon-input arc
(a,a) --x:eps--> (0)        # source symbol consumed, no output
```

Both transduce `"xy"` to `['a','a','b','b']`.  TransducedLM decodes all
4 steps on the duplicate-style FST but fails at step 2 on the BPE-style
FST with `ValueError: Out of vocabulary: 'a'`.

## Diagnosis

### Step 0 (target=`()`)

Both backends agree on the decomposition at step 0:

| Property | Duplicate-style | BPE-style |
|---|---|---|
| DFA starts | `[0]` | `[0]` |
| Q('a') | `{2}` | `{2}` |
| Q('b') | `{1}` | `{1}` |
| preimage_stops | `{0}` | `{0}` |

The difference is in `resume_frontiers`:

| Backend | Duplicate-style | BPE-style |
|---|---|---|
| Python `PeekabooState` | `{'a': 1, 'b': 1}` | `{'a': 0, 'b': 0}` |
| Rust `RustPeekabooState` | `{'a': 1, 'b': 1}` | `{}` |

Both backends agree that resume_frontiers is empty for BPE-style.
(The Python version reports 0-element sets; Rust omits empty entries.)

### Step 1 (target=`('a',)`)

| Property | Duplicate-style | BPE-style (Rust) | BPE-style (Python) |
|---|---|---|---|
| DFA starts | `[0]` | `[3]` | (empty BFS) |
| Q('a') states | `{5}` | `{4}` | `{}` (empty) |
| resume_frontiers | `{'a': 1}` | `{'a': 1}` | `{}` |

**Python PeekabooState** (Bug 1): The child BFS has no seed states
(resume_frontiers was empty), so the decomposition is completely empty.

**Rust RustPeekabooState**: Correctly finds DFA start state 3 and
Q('a') at state 4 via dirty-state recomputation.

### DFA state analysis (Rust backend, BPE-style)

The carry-forward particle from step 0 sits at DFA state 2 (Q('a') in
step 0).  In step 1's DFA:

```
State 2 (old Q('a')):
  NFA elements: {(('a',), ('a',), T), (('a','a'), ('a',), T),
                  (('b',), ('a',), T), (('b','b'), ('a',), T),
                  (0, ('a',), T)}
  All elements are truncated (T=True).
  Arcs: ('x', 2), ('y', 2)  <-- self-loops, dead end

State 3 (new DFA start):
  NFA elements: {(('a','a'), ('a','a'), F)}
  Non-truncated.
  Arcs: ('x', 4)  <-- leads to Q('a')

State 4 (new Q('a')):
  NFA elements: {(0, ('a','a'), F), (('a',), ('a','a'), T), ...}
```

The carry-forward particle at state 2 self-loops and can never reach
states 3 or 4.  Meanwhile, no particle is seeded at the new DFA start
state 3, so Q('a') at state 4 is unreachable.

## Root cause

### Why resume_frontiers is empty for BPE-style

In BPE-style FSTs, the epsilon closure from the start state traverses
all epsilon-input arcs, accumulating output in the buffer.  Any output
beyond `K=1` symbols is truncated.  Since the non-epsilon input arc
produces no output (epsilon), consuming a source symbol doesn't create
any non-truncated NFA states.  All NFA elements in the Q DFA states
are truncated, so the `resume_frontiers` check
(`not any(truncated for ... in state)`) rejects them all.

In duplicate-style FSTs, the non-epsilon input arc itself produces the
first output symbol.  This creates a non-truncated NFA element
`(fst_state, (first_output,), False)` in the Q DFA state.  The DFA
start state sits at the truncation boundary (non-truncated with
truncated successors), so it's added to `resume_frontiers`.

### Why carry-forward particles are dead ends (Bug 2)

`TransducedState.__rshift__` (line 315) creates the new state:

```python
new_peekaboo = self._peekaboo_state >> y
cf_particles = self._carry_forward_cache.get(y, [])
new_particles = _select_top_k(cf_particles, K)
return TransducedState(self.tlm, new_peekaboo, new_particles, ...)
```

The new state's beam consists entirely of carry-forward particles.
Their DFA state IDs come from step N's Q states.  The new
decomposition (step N+1) has different DFA start states and Q/R
states, but no particles are seeded there.  When `_compute_logp_next`
runs the beam search, the carry-forward particles expand via
`dfa.arcs(particle.dfa_state)` but hit self-loops (all NFA elements
are truncated, so source transitions cycle back to the same powerset
state).

### Why PeekabooState produces empty decomps (Bug 1)

`PeekabooState.__rshift__` uses resume_frontiers from step N to seed
the child BFS for step N+1.  When resume_frontiers is empty, the
child BFS worklist is empty, producing an empty decomposition.  The
Rust DirtyPeekaboo avoids this by always recomputing from the full
dirty-state set rather than relying on resume_frontiers for seeding.

## Impact

`bpe_wfst()` constructs FSTs with the BPE-style arc structure: chains
of `eps:byte` arcs followed by `tokenID:eps`.  Neither `TransducedLM`
nor `FusedTransducedLM` can decode more than one output symbol on these
FSTs.

Bug 2 affects both Python and Rust backends (since the carry-forward
logic is in `transduced.py` / `fused_transduced.py`, not in the
decomposition).  Bug 1 additionally breaks the Python `PeekabooState`
backend, but this is masked by the default use of the Rust backend.

## Failing test case

```python
def test_transduced_lm_bpe_style_fst():
    """TransducedLM fails on FSTs where epsilon-input arcs produce all
    output and the non-epsilon input arc produces epsilon (BPE pattern)."""
    from transduction.fst import FST, EPSILON
    from transduction.lm.ngram import CharNgramLM
    from transduction.lm.transduced import TransducedLM

    # BPE-style: eps:output chains, then source:eps back to start
    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, EPSILON, 'a', ('a',))
    fst.add_arc(('a',), EPSILON, 'a', ('a','a'))
    fst.add_arc(('a','a'), 'x', EPSILON, 0)
    fst.add_arc(0, EPSILON, 'b', ('b',))
    fst.add_arc(('b',), EPSILON, 'b', ('b','b'))
    fst.add_arc(('b','b'), 'y', EPSILON, 0)

    inner_lm = CharNgramLM.train([list('xxyxy')] * 5, n=2, alpha=0.5)
    target = list(fst.transduce(list('xy')))  # ['a','a','b','b']

    tlm = TransducedLM(inner_lm, fst, max_beam=20)
    state = tlm.initial()

    # Step 1 works
    lp = state.logp_next['a']
    state = state >> 'a'
    assert lp < 0

    # Step 2 fails: 'a' is not in logp_next
    assert 'a' in state.logp_next, (
        f"logp_next keys: {sorted(state.logp_next.keys())}"
    )
```

For comparison, the equivalent duplicate-style FST passes:

```python
def test_transduced_lm_duplicate_style_fst():
    """Duplicate-style FST works: source:output first, eps:output after."""
    from transduction.fst import FST, EPSILON
    from transduction.lm.ngram import CharNgramLM
    from transduction.lm.transduced import TransducedLM

    fst = FST()
    fst.add_start(0)
    fst.add_stop(0)
    fst.add_arc(0, 'x', 'a', ('x',0))
    fst.add_arc(('x',0), EPSILON, 'a', 0)
    fst.add_arc(0, 'y', 'b', ('y',0))
    fst.add_arc(('y',0), EPSILON, 'b', 0)

    inner_lm = CharNgramLM.train([list('xxyxy')] * 5, n=2, alpha=0.5)
    target = list(fst.transduce(list('xy')))  # ['a','a','b','b']

    tlm = TransducedLM(inner_lm, fst, max_beam=20)
    state = tlm.initial()
    for y in target:
        assert y in state.logp_next
        state = state >> y
```
