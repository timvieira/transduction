# Carry-Forward Root-Family Deduplication

**Status: Removed.**  The root-family deduplication described below was
implemented and tested, but ultimately removed from both `TransducedLM` and
`FusedTransducedLM`.  The `_select_top_k` beam pruning implicitly limits
redundant carry-forward entries, making the explicit dedup unnecessary.
This document is preserved as a reference for the analysis.

## Background: The Per-Step BFS

TransducedLM estimates P(y | target_so_far) by maintaining a beam of K
**particles** — source-side hypotheses that evolve as each target symbol is
consumed.  Each particle tracks three things:

    particle = (dfa_state, lm_state, log_weight)

where `dfa_state` is the current state in the decomposition DFA (a powerset
construction over the precover NFA), `lm_state` is the inner LM state after
consuming the particle's source prefix, and `log_weight` is the accumulated
log-probability.

When the user requests `state.logp_next`, a **layered BFS** runs through the
DFA to score all possible next target symbols.  Here is the full structure:

```
function compute_logp_next(particles, decomposition):
    # The decomposition provides, for each next target symbol y:
    #   Q(y)      — quotient states (exact marginalization; particle absorbed)
    #   R(y)      — remainder states (weight * P_inner(EOS); particle absorbed)
    #   resume(y) — resume-frontier states (particle survives to next step)
    #   preimage  — states where source produced exactly the target prefix

    scores = {}           # y -> [log-weights]  (current-step scores)
    carry_forward = {}    # y -> [particles]    (next-step seeds)

    candidates = particles   # initial beam from previous step's carry-forward

    for layer = 0, 1, 2, ... :      # each layer = one more source symbol
        if candidates is empty: break

        to_expand = []
        for particle in candidates:
            classify_and_score(particle, scores, carry_forward)
            if particle.dfa_state not in Q:
                to_expand.append(particle)

        if to_expand is empty: break
        if early_termination_bound_met: break

        beam = select_top_k(to_expand, K)

        candidates = []
        for particle in beam:
            for (x, next_dfa_state) in dfa.arcs(particle.dfa_state):
                w = particle.log_weight + particle.lm_state.logp_next[x]
                if w > -inf:
                    candidates.append(Particle(next_dfa_state,
                                               particle.lm_state >> x, w))

    return normalize(scores), carry_forward
```

The key operation is `classify_and_score`, which does two things for each
particle:

1. **Score** — contributes to `scores[y]` for the *current* target step
   (Q states contribute the full weight; R states contribute weight × P(EOS)).

2. **Carry forward** — saves the particle into `carry_forward[y]` for the
   *next* target step, if the particle is at a Q, R, or resume-frontier state
   for symbol y.

```
function classify_and_score(particle, scores, carry_forward):
    d = particle.dfa_state
    w = particle.log_weight

    # Score: accumulate weights for the current target step
    for y where d in Q(y):    scores[y].append(w)
    for y where d in R(y):    scores[y].append(w + lm_state.logp_next[EOS])
    if d in preimage:         eos_scores.append(w + lm_state.logp_next[EOS])

    # Carry forward: save for the next target step
    carry_syms = {y : d in Q(y)} | {y : d in R(y)} | {y : d in resume(y)}
    for y in carry_syms:
        carry_forward[y].append(particle)     # <-- THE BUG IS HERE
```

After the BFS completes, the caller advances by target symbol y:

```
function advance(state, y):
    logp_next, carry_forward = compute_logp_next(state)
    new_particles = select_top_k(carry_forward[y], K)
    return TransducedState(new_particles, logp=state.logp + logp_next[y])
```

## The Bug

The carry-forward append on the last line of `classify_and_score` collects
particles from **all** BFS layers unconditionally.  Consider this scenario
with the `delete_b` FST (maps a→A, b→ε):

```
Layer 0:  particle P₀ at DFA start
             ↓ expand by source symbol 'a'
Layer 1:  particle P₁ at DFA state d₁  ← d₁ is in Q('A') and resume('A')
             ↓ expand by source symbol 'b'
Layer 2:  particle P₂ at DFA state d₂  ← d₂ is also in Q('A') and resume('A')
             ↓ expand by source symbol 'b'
Layer 3:  particle P₃ at DFA state d₃  ← d₃ is also in Q('A') and resume('A')
          ...
```

All of P₁, P₂, P₃ get added to `carry_forward['A']`.  Their source paths
are `a`, `ab`, `abb`, ... — each a strict extension of the previous.

## Why This Is Wrong

The DFA is **deterministic** with a **single start state** (powerset
construction over the precover NFA).  This means: the same source-symbol
sequence always reaches the same DFA state with the same transition weights.

Therefore, when P₁ (source path `a`) is expanded in the next step's BFS, it
will reproduce P₂ at the **exact same DFA state d₂** with the **exact same
weight**.  Continuing expansion reproduces P₃ at d₃, and so on.

```
Next step's BFS, starting from carry_forward['A']:
  P₁ (path=a, state=d₁) → expand by 'b' → reaches d₂ with same weight as P₂
                         → expand by 'b' → reaches d₃ with same weight as P₃
  P₂ (path=ab, state=d₂)  ← REDUNDANT: already reproduced by P₁'s expansion
  P₃ (path=abb, state=d₃) ← REDUNDANT: already reproduced by P₁'s expansion
```

Consequences:
- **Double-counting**: P₂'s and P₃'s weights are counted in scores twice
  (once directly, once via P₁'s expansion).
- **Wasted beam slots**: P₂ and P₃ occupy beam slots that could hold genuinely
  different hypotheses.

## The Invariant

**No particle's source path should be a strict prefix of another particle's
source path within the same carry-forward set (for a given target symbol).**

If particle P has source path that is a prefix of particle P', then:
- P's future BFS expansion will deterministically reproduce P'
- P' contributes no information that P doesn't already provide
- Keeping both wastes a beam slot and double-counts P's contribution

## Root-Family Structure

A key structural observation makes enforcing this invariant simple and
efficient:

**Prefix domination within carry-forward can only occur among descendants of
the same root particle.**

*Proof:*  Let A and B be two carry-forward particles for target symbol y,
where B's total source path is a strict prefix of A's.  We show A and B must
descend from the same initial particle (root).

The initial particles entering the BFS are the carry-forward from the previous
target step.  By the inductive invariant, no initial particle's source path is
a prefix of another's.  Let rₐ and r_b be the root paths of A and B
(the source paths of their respective initial particles).  Then:

    A's total path = rₐ + extₐ
    B's total path = r_b + ext_b

where extₐ and ext_b are the source symbols consumed during this BFS.  Since
B's path is a prefix of A's path:

    r_b + ext_b  is a prefix of  rₐ + extₐ

If |r_b + ext_b| ≤ |rₐ|, then r_b is a prefix of rₐ.  By the no-prefix
invariant on roots, rₐ = r_b (same root).

If |r_b + ext_b| > |rₐ|, then rₐ is a prefix of r_b + ext_b, and since
|rₐ| ≥ |r_b| would imply r_b is a prefix of rₐ (same root), while
|rₐ| < |r_b| would imply rₐ is a prefix of r_b (same root by invariant).

In all cases, A and B share the same root.  ∎

This gives the BFS a **disjoint-set structure**: each initial particle defines
a "root family" of descendants.  Within each family, for a given target symbol,
the **shallowest** carry-forward entry is the canonical representative — all
deeper ones are redundant.  Particles from different families can never have
prefix-related paths, so they never interact.

## The Fix

Tag each particle with the index of the initial particle it descended from.
Since BFS processes layers shallowest-first, the first carry-forward entry
for a `(root_id, target_symbol)` pair is always the shallowest.  "First one
wins" is sufficient — no eviction, no path comparison, no LM-side dependencies:

```
function compute_logp_next(particles, decomposition):
    scores = {}
    carry_forward = {}
    root_of = {}
    carried = {}          # set of (root_id, y) pairs already added

    # Tag initial particles with their root index
    for i, p in enumerate(particles):
        root_of[p] = i

    function add_carry(y, particle):
        rid = root_of[particle]
        if (rid, y) in carried:
            return                              # deeper descendant — skip
        carried.add((rid, y))
        carry_forward[y].append(particle)

    function classify_and_score(particle, scores, carry_forward):
        d = particle.dfa_state
        w = particle.log_weight

        # Score: same as before (unchanged)
        for y where d in Q(y):    scores[y].append(w)
        for y where d in R(y):    scores[y].append(w + lm_state.logp_next[EOS])
        if d in preimage:         eos_scores.append(w + lm_state.logp_next[EOS])

        # Carry forward: with root-family dedup
        carry_syms = {y : d in Q(y)} | {y : d in R(y)} | {y : d in resume(y)}
        for y in carry_syms:
            add_carry(y, particle)              # <-- FIXED

    candidates = particles

    for layer = 0, 1, 2, ... :
        if candidates is empty: break

        to_expand = []
        for particle in candidates:
            classify_and_score(particle, scores, carry_forward)
            if particle.dfa_state not in Q:
                to_expand.append(particle)

        if to_expand is empty: break
        if early_termination_bound_met: break

        beam = select_top_k(to_expand, K)

        candidates = []
        for particle in beam:
            rid = root_of[particle]             # <-- propagate root
            for (x, next_dfa_state) in dfa.arcs(particle.dfa_state):
                w = particle.log_weight + particle.lm_state.logp_next[x]
                if w > -inf:
                    child = Particle(next_dfa_state,
                                     particle.lm_state >> x, w)
                    root_of[child] = rid        # <-- inherit root
                    candidates.append(child)

    return normalize(scores), carry_forward
```

Properties of this approach:
- **O(1) per carry-forward insertion** (set lookup on `(rid, y)`)
- **No dependency on `LMState.path()`** — works with any LM backend
  (HuggingFace models, n-gram LMs, etc.)
- **No eviction logic needed** — BFS layer ordering guarantees shallowest-first
- **Inductive**: the output carry-forward satisfies the no-prefix invariant,
  so the next step's BFS can rely on the same property for its initial particles

## Concrete Example: `delete_b`

The `delete_b` FST over alphabet {a, b} deletes every 'b' and maps a → A:

```
FST:  state 0 --a/A--> 0
      state 0 --b/ε--> 0
      state 0 is start and final
```

With an inner LM that assigns P(a)=0.6, P(b)=0.3, P(EOS)=0.1:

**Without root-family dedup (the bug):**
```
Layer 0: P₀(dfa=start, path=[], w=0.0)
  → not Q/R/resume, expand

Layer 1: P₁(dfa=d₁, path=[a], w=log(0.6))    ← Q('A'), resume('A')
          P₂(dfa=d₂, path=[b], w=log(0.3))    ← Q(???), depends on DFA
  carry_forward['A'] = [P₁]
  → P₁ is Q, skip; P₂ continues

Layer 2: P₃(dfa=d₃, path=[b,a], w=log(0.18)) ← Q('A'), resume('A')
          P₄(dfa=d₄, path=[b,b], w=log(0.09))
  carry_forward['A'] = [P₁, P₃]                ← P₃ IS PREFIX-DOMINATED BY P₁? No!
  ...but from root P₀, path [a] and path [b,a] are different roots' descendants?

Actually, P₁ and P₃ both descend from root P₀. P₁'s source path [a] is NOT
a prefix of P₃'s path [b,a]. So this is fine — they're genuinely different.
```

A clearer trigger: after advancing by 'A', the carry-forward particle P₁ (path
[a]) enters the next BFS.  During that BFS:

```
Layer 0: P₁(dfa=d₁, path=[a], w=w₁)           ← root 0
  → d₁ is Q('A') and resume('A')
  carry_forward['A'] = [P₁]                     root 0, first entry
  → not purely Q, expand

Layer 1: child₁(dfa=d_ab, path=[a,b], w=w₁+log(0.3))  ← root 0
         child₂(dfa=d_aa, path=[a,a], w=w₁+log(0.6))  ← root 0

Layer 1 classify: d_ab might be Q('A')/resume('A') again (b is deleted)
  add_carry('A', child₁):
    root_of[child₁] = 0, (0, 'A') already in carried → SKIP ✓
```

The root-family check correctly skips child₁ — P₁ already covers this
root family for symbol 'A'.

## Why Not Just Stop Expanding Carried-Forward Particles?

The tempting structural alternative: don't expand particles that were carried
forward.  If they're saved for the next step, why explore deeper?

This doesn't work because carry-forward and scoring serve different purposes:

- **Carry-forward** is for the **next** target step (seeding the next BFS).
- **Scoring** is for the **current** target step (accumulating Q/R weights).

A particle carried forward for symbol `y` still needs expansion in the current
BFS because its descendants may reach Q/R states that contribute scores for
**other** symbols `z ≠ y`.  Stopping expansion would lose those scores.

```
Example: particle P at DFA state d
  d is in resume('A')  → P is carried forward for 'A'
  d is NOT in Q        → P is expanded
  P's child P' at d' where d' is in Q('B')  → contributes score for 'B'

If we had stopped expanding P after carrying it forward for 'A',
we would have missed P's contribution to the score for 'B'.
```

## Implications for FusedTransducedLM

Both `TransducedLM` and `FusedTransducedLM` had the dedup implemented and
tested, then removed.  Neither implementation performs root-family tracking.

## Key Properties Used

1. **DFA determinism**: Same source path → same DFA state.  Guaranteed by
   powerset construction.
2. **Single start state**: The DFA has exactly one start state, so there's a
   unique DFA state for each source path.
3. **Monotone weights**: Particle weights only decrease with source depth
   (each step multiplies by P(x) ≤ 1).  So the shallower particle has
   weight ≥ the deeper one.
4. **Inductive invariant**: The previous step's carry-forward already satisfies
   the no-prefix property, so initial particles entering each BFS have
   non-prefix source paths.  This ensures the root-family partition is sound.

## Observable Invariant (Tested)

After `_compute_logp_next`, for each target symbol `y`, no two carry-forward
particles should share a DFA state.  This follows from:
- Within a root family: only one entry per (root, y), at a unique DFA state.
- Across root families: different roots → different DFA states (by DFA
  determinism + non-prefix initial paths).

See `TestCarryForwardNoDuplicates` in `tests/test_transduced.py`, which
verifies this on `delete_b` (the primary trigger — source prefixes `a`, `ab`,
`abb`, ... all produce the same output), `duplicate`, `infinite_quotient`,
`lookahead`, `small`, and `newspeak2`.
