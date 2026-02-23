# CharacterBeam vs FusedTransducedLM: Side-by-Side Comparison

Both algorithms solve the same problem: given an inner LM $P(x)$ over source
tokens and an FST $f: \mathcal{X}^* \to \mathcal{Y}^*$, compute the
conditional distribution $P(y \mid \boldsymbol{y}_{\text{so far}})$ over next
target symbols by marginalizing over source continuations.  Both implement the
`LM`/`LMState` interface and expose the result as `state.logp_next[y]`.

This report presents them in parallel, using a common vocabulary, to make the
structural correspondence explicit.

---

## 1. Architecture Overview

```
                 CharacterBeam                      FusedTransducedLM
                 ─────────────                      ──────────────────
LM factory       CharacterBeam(LM)                  FusedTransducedLM(LM)
LM state         CharacterBeamState                 FusedTransducedState
Internal hyp     TrieState                          Particle
Hyp collection   _Bundle (list of TrieStates)       list[Particle]
State space      TokenCharacterTrie (static)         Lazy powerset DFA (on demand)
Search           Beam (top-K weight pruning)         Best-first (priority queue)
Output level     Bytes (characters)                  Arbitrary FST output symbols
Applicability    BPE/SPM tokenizers only              Any FST
```

---

## 2. Hypothesis Representation

A **hypothesis** is a single weighted guess about what the source has done so
far.  Both algorithms maintain a set of hypotheses and update them as each
target symbol is observed.

### CharacterBeam: TrieState

```python
class TrieState:
    lm_state: LMState       # inner LM state (after completed tokens)
    node: int                # position in the token-character trie
    log_mass: np.ndarray     # log P(subtree) at each trie node for this LM state
    weight: float            # log importance weight
```

The hypothesis tracks where it is *within a token's byte decomposition*.  The
`log_mass` array — precomputed once per LM state via a single O(|V|) vectorized
scatter — enables O(1) conditional probability lookups for every byte.  The LM
state only changes at token boundaries.

### FusedTransducedLM: Particle

```python
class Particle:
    dfa_state: int           # position in the powerset DFA
    lm_state: LMState        # inner LM state (after each source symbol)
    log_weight: float        # log importance weight
    source_path: tuple       # explicit source-symbol history (for replay)
```

The hypothesis tracks where it is in the *FST's powerset DFA*.  The DFA state
encodes which precover-NFA states are reachable given the target observed so
far.  The source path is recorded so the particle can be replayed through a
fresh DFA at the next step (the lazy DFA is rebuilt per target symbol).

### Correspondence

```
TrieState.lm_state      ↔  Particle.lm_state
TrieState.node           ↔  Particle.dfa_state
TrieState.log_mass       ↔  (no direct analog — FusedTransducedLM computes scores
                              incrementally via the search loop)
TrieState.weight         ↔  Particle.log_weight
(implicit in LM state)   ↔  Particle.source_path
```

---

## 3. State Space

### CharacterBeam: TokenCharacterTrie

Built **once** at construction time from the BPE vocabulary.

```
                        root
                       / | \
                    'H' 'e' 'l' ...
                    /     |
                 'e'     'l'
                 /         \
              [Hello]    [ello]     ← leaf (end-of-token sentinel)
```

Each leaf corresponds to a complete token.  The trie is annotated with a COO
reachability index mapping each leaf (token) to all its ancestor nodes.  This
enables the key operation — `log_mass_sum` — to propagate token-level
log-probabilities to every ancestor in one `np.logaddexp.at` call:

```python
def log_mass_sum(self, logp_next: LogDistr) -> np.ndarray:
    logp = np.array([logp_next[tok] for tok in self._tokens])    # |V| token probs
    log_mass = np.full(num_nodes, -np.inf)
    np.logaddexp.at(log_mass, self._coo_nodes, logp[self._coo_token_ids])
    return log_mass
```

### FusedTransducedLM: Lazy Powerset DFA

Built **on demand** per target step via `new_step(target)`.

The DFA states are powerset states of the precover NFA — each one is a set of
`(fst_state, output_buffer, truncated)` tuples.  States and arcs are
materialized only as particles explore them, so the cost is proportional to
search effort rather than FST size.

The DFA supports three key operations:
- `classify(state)` → quotient symbol, remainder symbols, preimage flag
- `arcs(state)` → list of (source_sym, dest_state) transitions
- `single_arc(state, sym)` → dest_state for one source symbol

---

## 4. Step-by-Step Operation

Both algorithms follow the same outer loop:

```
state₀ = alg.initial()          # create initial hypothesis set
p = state₀.logp_next            # compute P(y | empty context) for all y
state₁ = state₀ >> y₁           # advance by observed symbol y₁
p = state₁.logp_next            # compute P(y | y₁)
state₂ = state₁ >> y₂           # advance by y₂
...
```

The inner workings of `logp_next` and `>> y` differ substantially.

### 4.1 Computing logp_next

#### CharacterBeam

```
Given: beam = [TrieState₁, TrieState₂, ..., TrieStateₖ]

Step 1: EXTEND
  For each hypothesis h at an end-of-token leaf:
    if h passes the extend threshold:
      lm' = h.lm_state >> token_id           ← LM call
      mass' = trie.log_mass_sum(lm'.logp_next)  ← O(|V|) scatter
      add TrieState(lm', root, mass', w') to beam

Step 2: SCORE
  For each byte value b:
    logp_next[b] = logsumexp over all h in beam:
      h.weight + h.log_mass[child_b(h.node)] - h.log_mass[h.node]

Step 3: NORMALIZE
  logp_next -= logsumexp(logp_next)
```

**Key property**: Scoring is pure array arithmetic — no priority queue, no DFA
classification.  Each hypothesis contributes via its precomputed mass array.

#### FusedTransducedLM

```
Given: particles = [Particle₁, Particle₂, ..., Particleₖ]

Step 1: RESOLVE sentinels
  For each particle with dfa_state=None:
    replay source_path through fresh DFA → resolved dfa_state

Step 2: SEARCH (best-first, up to max_steps iterations)
  priority_queue = particles
  while queue not empty and steps < max_steps:
    pop highest-weight particle p
    classify(p.dfa_state):

    case QUOTIENT for y:
      scores[y] += p.log_weight       ← exact marginalization
      carry_forward[y].add(p)
      (particle absorbed — not expanded)

    case REMAINDER for y:
      scores[y] += p.log_weight + logp(EOS)
      carry_forward[y].add(p)

    case PREIMAGE:
      eos_score += p.log_weight + logp(EOS)

    case NONE:
      for each source symbol x:
        w' = p.log_weight + p.lm_state.logp_next[x]   ← LM lookup
        lm' = p.lm_state >> x                          ← LM call
        dfa' = arcs(p.dfa_state, x)                    ← DFA transition
        push Particle(dfa', lm', w', path+(x,)) onto queue

Step 3: DRAIN remaining queue (score without expanding)

Step 4: NORMALIZE
  scores[eos] += eos_score
  logp_next = scores.normalize()
```

**Key property**: The Q/R/preimage classification drives the search.  Quotient
states provide exact marginalization (the particle's full weight counts for all
continuations), eliminating expansion.  Non-classified states require per-symbol
LM calls to expand.

### 4.2 Advancing by symbol y (the >> operator)

#### CharacterBeam

```
Step 1: PRUNE — keep top-K hypotheses by weight

Step 2: ADVANCE
  For each surviving hypothesis h:
    child = trie.children[h.node].get(y)
    if child exists:
      emit TrieState(h.lm_state, child, h.log_mass,
                      h.weight + log_mass[child] - log_mass[h.node])
```

No DFA rebuild, no replay.  Each hypothesis simply steps one node deeper in the
(static) trie.

#### FusedTransducedLM

```
Step 1: SELECT carry_forward[y]

Step 2: PRUNE — keep top max_beam particles by weight

Step 3: (logp_next computation at next step will call new_step(target ++ y),
         rebuilding the lazy DFA, and resolve sentinel particles by replaying
         their source_path through the fresh DFA)
```

---

## 5. When the Inner LM Is Called

| | CharacterBeam | FusedTransducedLM |
|---|---|---|
| **Trigger** | End-of-token (EOT sentinel in trie) | Expansion of a non-Q/R particle |
| **What happens** | `lm_state >> token` for the completed token | `lm_state >> x` for one source symbol |
| **Frequency** | Once per completed token per hypothesis | Once per expansion per source symbol |
| **Between calls** | Hypotheses coast through trie using cached `log_mass` | — |

CharacterBeam amortizes LM calls across $\bar{L}$ byte steps (where $\bar{L}$
is average token length).  Between token boundaries, the precomputed mass array
provides exact conditional probabilities with zero LM interaction.

FusedTransducedLM calls the LM on every expansion, but compensates by absorbing
particles at quotient states — avoiding expansion of particles that would
contribute the same information.

---

## 6. Exact Marginalization

Both algorithms perform exact marginalization — but at different granularities.

### CharacterBeam: via trie mass

At the trie root, `log_mass[child_b]` is the exact log-sum of
$P_{\text{inner}}(t)$ over all tokens $t$ whose byte decomposition starts with
byte $b$.  This marginalizes over the *entire vocabulary subtree* in one
vectorized operation.

For a hypothesis mid-trie at node $n$, the ratio
`log_mass[child_b(n)] - log_mass[n]` is the exact conditional probability of
byte $b$ given that the token's prefix matches the path to $n$.

### FusedTransducedLM: via Q-absorption

When a particle reaches a quotient state $Q(y)$, its full weight is credited
to $y$ — marginalizing over *all source continuations from that DFA state*
(which, by definition, all produce $y$).  This is an infinite summation
performed in O(1).

The two are equivalent for BPE: the trie-mass computation at the root is
exactly the sum that FusedTransducedLM would compute by expanding all source
symbols from the hub and Q-absorbing every one.

---

## 7. Carry-Forward and Pruning

### CharacterBeam

The carry-forward is the beam itself.  After pruning to top-K and advancing by
one byte, the surviving `TrieState`s are the hypotheses for the next step.

No deduplication is needed: Q-absorption at the hub means particles are not
expanded, so no two hypotheses create overlapping descendants.

**Pruning controls**:
- `K`: hard beam width (max hypotheses)
- `relative_score_threshold`: ratio-based pruning
- `eot_immunity`: protect non-EOT hypotheses from pruning
- `extend_threshold`: skip extending low-weight hypotheses at EOT

### FusedTransducedLM

Carry-forward is a per-symbol dict.  When the user advances by symbol $y$,
`carry_forward[y]` provides the particle set for the next step.

Particles at Q states are carried forward without prefix-domination checks
(they're absorbed, not expanded).  Particles at R/resume states use
prefix-domination checks to avoid double-counting when two particles reach the
same DFA state via different source paths.

**Pruning controls**:
- `max_beam`: limit on particles carried forward per symbol
- `max_steps`: total expansion budget per step
- `top_k`: only expand the k highest-probability source symbols per particle

---

## 8. Construction Cost

### CharacterBeam

```python
cb = CharacterBeam(lm, vocab, K=5)
```

- Builds `TokenCharacterTrie` from vocab: O(|V| * avg_token_length)
- Computes initial `log_mass_sum`: O(|V| * avg_token_length)
- No DFA construction

### FusedTransducedLM

```python
tlm = FusedTransducedLM(inner_lm, fst, max_steps=1000)
```

- Converts FST to Rust representation: O(|arcs|)
- Creates `RustLazyPeekabooDFA`: O(|states| + |arcs|)
- DFA states are built lazily during search (no upfront cost)
- But each step rebuilds the lazy DFA via `new_step(target)`

---

## 9. What Enables CharacterBeam's Specialization

CharacterBeam works because BPE tokenizers satisfy four properties that the
general FusedTransducedLM cannot assume:

### 9.1 All accepting states are IP-universal → Q-only scoring

The BPE WFST has a single hub state that is both start and sole accept state.
Every DFA state containing the hub is a quotient state.  There are **no
remainder contributions**: the R term vanishes entirely.

**CharacterBeam**: no EOS-probability logic anywhere.
**FusedTransducedLM**: must handle R scoring (`weight + logp(EOS)`) and
preimage scoring at every step.

### 9.2 Monoid homomorphism → single shared trie

Each token always emits the same byte sequence regardless of FST state.  This
means a single trie works for all hypotheses.  The mass array depends only on
the LM state, not the FST state.

**CharacterBeam**: one `TokenCharacterTrie` built once.
**FusedTransducedLM**: the DFA encodes state-dependent output, requiring
per-state classification.

### 9.3 Single hub → no carry-forward bookkeeping

All completed tokens return to the same hub.  Beam propagation is the
carry-forward: no source-path tracking, no prefix-domination checks.

**CharacterBeam**: beam *is* carry-forward.
**FusedTransducedLM**: per-symbol `carry_forward` dict with dedup checks.

### 9.4 Token boundaries → deferred LM calls

The FST returns to the hub after each token, so the LM state changes only at
token boundaries.  Between boundaries, the precomputed mass array provides all
necessary probabilities.

**CharacterBeam**: O(1) LM calls per token boundary.
**FusedTransducedLM**: O(k) LM calls per expansion (one per source symbol).

---

## 10. Pseudocode: Side-by-Side

### Initialization

```
CharacterBeam                          FusedTransducedLM
─────────────                          ──────────────────
s₀ = lm.initial()                     s₀ = inner_lm.initial()
mass₀ = trie.scatter(s₀.logp_next)    dfa.new_step([])
beam = [TrieState(s₀, root, mass₀,    start_ids = dfa.start_ids()
                   weight=0)]          particles = [Particle(sid, s₀, 0, ())
                                                    for sid in start_ids]
```

### Scoring (one target step)

```
CharacterBeam                          FusedTransducedLM
─────────────                          ──────────────────
# EXTEND (advance LM at token         # RESOLVE sentinels
# boundaries)                          for p with dfa_state=None:
for h in beam:                           p.dfa_state = dfa.run(p.source_path)
  for token t at EOT(h.node):
    w' = h.w + mass[eot(t)] - mass[n]
    s' = h.lm >> t          ← LM      # SEARCH (best-first)
    mass' = scatter(s'.logp_next)      queue = heap(particles)
    beam += TrieState(s', root,        while queue and steps < budget:
                      mass', w')         p = queue.pop_max()
                                         classify(p.dfa_state):
# SCORE (trie mass ratios)
for each byte b:                         QUOTIENT(y):
  logp[b] = logsumexp over beam:           scores[y] ⊕= p.weight
    h.w + mass[child_b] - mass[node]       carry[y].add(p)    ← absorbed

                                         REMAINDER(y):
                                           scores[y] ⊕= p.weight + logp(EOS)
                                           carry[y].add(p)

                                         PREIMAGE:
                                           eos ⊕= p.weight + logp(EOS)

                                         ELSE:          ← expand
                                           for x in source_symbols:
                                             w' = p.weight + logp(x)
                                             s' = p.lm >> x    ← LM
                                             d' = dfa.arc(p.dfa, x)
                                             queue.push(Particle(d',s',w',path+(x,)))

# NORMALIZE                            # DRAIN + NORMALIZE
logp_next = normalize(logp)            scores[EOS] ⊕= eos_score
                                       logp_next = scores.normalize()
```

### Advancing by symbol y

```
CharacterBeam                          FusedTransducedLM
─────────────                          ──────────────────
beam = top_K(beam)                     particles = carry_forward[y]
beam' = []                             particles = top_K(particles, max_beam)
for h in beam:
  if y in children(h.node):            # (DFA rebuilt at next logp_next call
    c = children(h.node)[y]            #  via new_step(target ++ y);
    w' = h.w + mass[c] - mass[h.node]  #  sentinel particles replayed via
    beam' += TrieState(h.lm,           #  source_path)
             c, h.log_mass, w')
```

---

## 11. Complexity Comparison

| | CharacterBeam | FusedTransducedLM |
|---|---|---|
| **Per-step LM calls** | $\leq K$ (at EOT boundaries) | $\leq$ max_steps (at expansion) |
| **Per-step scoring** | O(K * |children|) array ops | O(max_steps) heap ops + DFA classify |
| **Mass computation** | O(|V| * L̄) vectorized scatter | — (no analog) |
| **DFA cost** | — (no DFA) | O(expanded states) lazy construction |
| **Memory** | O(|V| * L̄) for trie + mass | O(max_beam) particles + lazy DFA cache |
| **Carry-forward cost** | O(K) (beam = carry) | O(max_beam) + dedup overhead |

Where $K$ = beam width, $|V|$ = vocabulary size, $\bar{L}$ = avg token length in
bytes, max_steps = expansion budget.

---

## 12. Why CharacterBeam Is So Much Faster: The Implicit Decomposition

A natural question: CharacterBeam must be doing something equivalent to the
incremental peekaboo decomposition that FusedTransducedLM relies on — how can
it be so much more efficient?

The answer is that CharacterBeam doesn't have a precover NFA or powerset DFA
at all.  **The trie replaces the entire decomposition machinery.**  This is
possible because the BPE structure makes the decomposition trivial — and the
trie encodes that trivial decomposition in a form that admits vectorized
computation.

### 12.1 What the powerset DFA is doing (and why it's unnecessary for BPE)

FusedTransducedLM's powerset DFA tracks: "given the target output observed so
far, which `(fst_state, output_buffer)` pairs are still live?"  This is needed
for a general FST because different source symbols from different states can
produce different outputs, and the algorithm must track which source paths are
*consistent* with the observed target.

For a BPE WFST, this tracking collapses.  The FST has a single hub state, and
every token always returns to that hub.  So after processing any complete token,
the decomposition state is always the same: `{(hub, target_prefix, False)}`.
There is only one "live" FST state at token boundaries — the hub — and the
output buffer is just the target seen so far.  The powerset never has more than
one FST state in it (at token boundaries).

Mid-token, the DFA state would encode "I've matched bytes $b_1 \ldots b_j$ of
the current token — which tokens could I still be building?"  But this is
exactly what the trie encodes: the set of tokens sharing a given byte prefix
corresponds to the subtree rooted at the matching trie node.  The trie is a
static, precomputed representation of what the DFA would discover dynamically.

### 12.2 The trie replaces the DFA

The trie encodes the mid-token part of the decomposition as a static prefix
tree:

| DFA concept | Trie equivalent |
|---|---|
| DFA state at token boundary | Trie root (all tokens reachable) |
| DFA state mid-token (after bytes $b_1 \ldots b_j$) | Trie node at depth $j$ (subtree = compatible tokens) |
| DFA arc for source symbol $x$ | Trie EOT leaf for token $x$ + return to root |
| DFA arc for output byte $b$ | Trie child edge for byte $b$ |
| `classify(state) → Q(y)` | `log_mass[child_y] - log_mass[node]` (trie mass ratio) |

Three specific advantages flow from this replacement:

1. **Precomputed once vs rebuilt per step.**  The BPE structure means the set
   of "which tokens match this byte prefix" is the same regardless of target
   position or LM state.  The trie is built once at construction.
   FusedTransducedLM rebuilds its lazy DFA via `new_step()` at every target
   position.

2. **A trie, not a powerset.**  The trie has O(|V| * $\bar{L}$) nodes with
   simple integer children maps.  The powerset DFA, even for BPE, must
   construct states as sets-of-NFA-states and hash-cons them.  The trie avoids
   all set operations, hashing, and Q/R/preimage classification overhead.

3. **O(1) scoring via the mass array.**  Each trie node's score is precomputed
   by `log_mass_sum` — a single `np.logaddexp.at` scatter.  FusedTransducedLM
   scores by expanding particles through the DFA one source symbol at a time,
   pushing each onto a heap, classifying each DFA state.  The trie replaces all
   of that with an array subtraction.

### 12.3 The core efficiency gap: vectorized scatter vs Python heap

Both algorithms need to answer the same question at token boundaries: "what's
the probability of each next byte, marginalizing over all tokens?"

**FusedTransducedLM**: For each particle at the hub, expand by each source
symbol (up to |V| or `top_k`), compute the DFA destination, classify it, push
onto the heap, eventually pop and Q-absorb.  That's O(|V|) or O(`top_k`)
*heap operations + DFA lookups + LM lookups* per particle, and there may be K
particles.

**CharacterBeam**: For each hypothesis at the root, call
`trie.log_mass_sum(lm_state.logp_next)` — one vectorized `np.logaddexp.at`
scatter over the COO index.  This does the same O(|V|) summation but as a
single NumPy operation over contiguous arrays.  Then read off
`log_mass[child] - log_mass[root]` for each byte — O(256) array lookups.

Critically, the mass array is **reused for all subsequent byte positions** until
the next token boundary ($\bar{L} \approx 4$ bytes on average for GPT-2), so
the O(|V|) scatter cost is amortized across $\bar{L}$ steps.

In concrete terms for GPT-2 (|V| = 50K, $\bar{L}$ ≈ 4, K = 10):

```
FusedTransducedLM per token:
  K particles × |V| expansions × (heap push + DFA lookup + LM lookup)
  = 10 × 50K × (Python heap push + hash lookup + dict lookup)
  ≈ 500K Python-level operations

CharacterBeam per token:
  K hypotheses × 1 scatter(|V|) + K × L̄ × |byte_children| array lookups
  = 10 × 1 NumPy scatter(50K) + 10 × 4 × 256 array reads
  ≈ 10 vectorized operations + 10K array reads
```

The constant-factor gap between a Python heap push and a NumPy vectorized
scatter over contiguous memory is enormous — easily 100× for |V| > 1000.
This is the fundamental source of CharacterBeam's speed advantage.

### 12.4 Why FusedTransducedLM can't use this trick in general

For a non-BPE FST, the trie replacement breaks down:

- **State-dependent output**: Different source symbols may produce different
  output sequences from different FST states — there is no single trie that
  works for all hypotheses at all states.

- **Target-dependent state**: The set of live `(fst_state, buffer)` pairs
  changes with each target symbol — you can't precompute a static structure.

- **Non-universal states**: When Q-absorption doesn't apply, particles must
  be expanded and scored via the R/preimage paths, which require tracking
  individual source paths and computing $P(\text{EOS})$ weights.

- **Genuine powerset complexity**: The DFA has multiple live FST states
  simultaneously — the powerset is doing real work tracking which NFA states
  are reachable, not just replaying a trie.

`GeneralizedBeam` recovers the trie trick at **hub states** where the BPE-like
structure holds locally (IP-universal, accepting, deterministic output per
source symbol), and falls back to FusedTransducedLM's particle expansion
elsewhere.  For BPE FSTs, 100% of the mass flows through the single hub, so
the full speedup applies.

---

## 13. Empirical Scaling (from Benchmark Dashboard)

Measured on BPE tokenizer FSTs with a bigram inner LM, K=10 beam / max_steps=1000:

| Vocab Size | FusedTransducedLM (ms/step) | CharacterBeam (ms/step) |
|-----------:|----------------------------:|------------------------:|
|        297 |                           2 |                       7 |
|      1,023 |                           9 |                      14 |
|      5,011 |                          70 |                      33 |
|     10,008 |                         171 |                      69 |
|     15,008 |                         213 |                     103 |

At small vocab, FusedTransducedLM's all-final-universal fast path is faster.
At V > ~2000, CharacterBeam's vectorized trie-mass scatter dominates.
CharacterBeam scales as ~|V|^0.7 while FusedTransducedLM scales more steeply.

---

## 14. GeneralizedBeam: The Unified Algorithm

`GeneralizedBeam` (in `generalized_beam.py`) subsumes both algorithms:

- At **IP-universal accepting hub states**: use trie-mass scoring (CharacterBeam path)
- At **non-hub DFA states**: use particle expansion (FusedTransducedLM path)

It maintains two hypothesis types — `HubHyp` (analogous to `TrieState`) and
`Particle` — and converts between them at hub boundaries:

```
Hub hyp completes token → destination is hub?
  Yes → new HubHyp at destination hub (stays on fast path)
  No  → spawn Particle (falls to slow path)

Particle reaches Q-absorbed hub?
  → scored and carried forward (could be promoted to HubHyp at next step)
```

Special cases:
- **BPE WFST** (single hub, all tokens return): only HubHyps, no Particles.
  Degenerates to CharacterBeam.
- **triplets_of_doom** (no hubs): only Particles, no HubHyps.
  Degenerates to FusedTransducedLM.
- **Most NLP tokenizers**: mostly HubHyps with occasional Particle excursions.

| Vocab Size | FusedTransducedLM | CharacterBeam | GeneralizedBeam |
|-----------:|------------------:|---------------:|----------------:|
|        297 |              2 ms |           7 ms |            1 ms |
|      1,023 |              9 ms |          14 ms |            3 ms |
|      5,011 |             70 ms |          33 ms |            8 ms |
|     10,008 |            171 ms |          69 ms |           18 ms |
|     15,008 |            213 ms |         103 ms |           30 ms |

GeneralizedBeam is fastest at all vocab sizes, with scaling ~|V|^0.55.

---

## 15. Summary

| Dimension | CharacterBeam | FusedTransducedLM |
|---|---|---|
| **Applicability** | BPE/SPM only | Any FST |
| **State space** | Static trie | Lazy powerset DFA |
| **Marginalization** | Trie mass scatter (vectorized) | Q-absorption (per-particle) |
| **LM calls** | At token boundaries only | At every expansion |
| **Carry-forward** | Beam = carry (no dedup) | Per-symbol dict (with dedup) |
| **EOS/remainder** | Not needed (all-Q) | Required (general FSTs have R) |
| **Scaling** | ~|V|^0.7 | Steeper (DFA + heap overhead) |
| **Advantage** | Fast for large BPE vocabs | Works for arbitrary FSTs |

CharacterBeam is FusedTransducedLM specialized to monoid-homomorphism FSTs
where all accepting states are IP-universal.  The specialization yields three
linked efficiency gains: Q-only scoring (no remainder logic), trie-mass
precomputation (replacing priority-queue search), and deferred LM calls
(amortized across byte steps).  GeneralizedBeam recovers these gains at hub
states while falling back to the general algorithm elsewhere.
