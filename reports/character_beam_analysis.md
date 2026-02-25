# CharacterBeam as a Special Case of FusedTransducedLM

## 1. Overview

CharacterBeam and FusedTransducedLM compute the same quantity: $P(y \mid
\boldsymbol{y}_{\text{so far}})$ for each next target symbol $y$, marginalizing
over all source (token) continuations under an inner LM composed with an FST.
CharacterBeam is a specialization that exploits the **monoid-homomorphism
structure** of BPE tokenizers to replace the general DFA-based search with a
single vectorized trie-mass computation. This report traces the exact
correspondence, identifies what makes the specialization possible, and discusses
how to recover the efficient algorithm in the general setting.

## 2. Setup and Notation

- **Inner LM**: $P_{\text{inner}}$ over source alphabet $\mathcal{X}$ (token
  IDs). Accessed via the `LM`/`LMState` interface: `state.logp_next[x]` gives
  $\log P(x \mid \text{history})$, and `state >> x` advances by one token.

- **FST**: $f : \mathcal{X}^* \to \mathcal{Y}^*$ mapping token sequences to
  byte sequences.

- **Transduced LM**: $P(y \mid \boldsymbol{y}) = \sum_{\boldsymbol{x}}
  P_{\text{inner}}(\boldsymbol{x}) \cdot \mathbb{1}[f(\boldsymbol{x}) \text{
  starts with } \boldsymbol{y} \cdot y] / Z(\boldsymbol{y})$.

- **BPE WFST** (`bpe_wfst`): A single hub state $()$ that is both start and
  sole accept state. Each token $t$ with bytes $b_1 \ldots b_k$ creates a chain
  $() \xrightarrow{\varepsilon:b_1} (b_1,) \xrightarrow{\varepsilon:b_2}
  \cdots \xrightarrow{t:\varepsilon} ()$. This is a **monoid homomorphism**:
  $f(t_1 t_2 \cdots t_n) = f(t_1) \cdot f(t_2) \cdots f(t_n)$.

## 3. FusedTransducedLM: The General Algorithm

FusedTransducedLM (`fused_transduced.py`) computes $P(y \mid \boldsymbol{y})$
via best-first search over source paths. Each **particle** carries:

- A **DFA state** (powerset of precover NFA states), tracking which source
  prefixes are consistent with the target seen so far.
- An **inner LM state**, encoding the source-side history.
- A **log weight** $\log P_{\text{inner}}(\boldsymbol{x}_{\text{prefix}})$.
- An **explicit source path** $\boldsymbol{x}_{\text{prefix}}$ (for
  carry-forward deduplication).

At each target step, the search:

1. **Pops** the highest-weight particle.
2. **Classifies** its DFA state via peekaboo decomposition:
   - **Quotient** ($Q$) for symbol $y$: all continuations produce $y$. The
     particle's full weight scores $y$. The particle is *absorbed* (not
     expanded further).
   - **Remainder** ($R$) for symbol $y$: the source can stop here. Scores $y$
     with weight $\times P_{\text{inner}}(\text{EOS})$.
   - **Preimage**: the source has produced exactly the target at a final state.
     Scores EOS with weight $\times P_{\text{inner}}(\text{EOS})$.
   - **None of the above**: expand by each source symbol, weighted by LM
     probability.
3. **Carry-forward**: particles at Q/R/resume-frontier states are passed to the
   next target step. Source-path prefix-domination checks prevent duplication.

This is fully general: it works for any FST, handles both Q and R
contributions, and terminates via truncation for FSTs with infinite quotients.

## 4. CharacterBeam: The Specialized Algorithm

CharacterBeam (`character_beam.py`) maintains a beam of **TrieState**
hypotheses, each carrying:

- An **inner LM state** (after completed tokens).
- A **position in a token-character trie** (partial match within the current
  token's byte decomposition).
- A **precomputed log-mass array** over all trie nodes.
- A **log weight**.

At each byte position:

1. **Extend**: hypotheses at end-of-token advance the LM (consume the token
   ID, return to trie root with a fresh mass array).
2. **Score**: the next-byte distribution is the mixture over hypotheses,
   each contributing $\text{mass}(\text{child}_b) / \text{mass}(\text{node})$.
3. **Prune**: top-$K$ hypotheses by weight.
4. **Advance**: surviving hypotheses move to `trie.children[node][b]`.

### 4.1 The Trie Mass Computation

The central operation is `TokenCharacterTrie.log_mass_sum`:

```python
def log_mass_sum(self, logp_next: LogDistr) -> torch.Tensor:
    logp = torch.tensor([logp_next[tok] for tok in self._tokens],
                        dtype=torch.float64)
    logp_max = logp.max()
    prob = torch.exp(logp - logp_max)
    mass = torch.mv(self._reach, prob)      # sparse CSR matvec
    log_mass = torch.log(mass) + logp_max
    return log_mass
```

This computes, for every trie node $n$:

$$\text{mass}(n) = \bigoplus_{t : \text{path}(t) \ni n} P_{\text{inner}}(t \mid \text{LM state})$$

where $\oplus$ is logaddexp. The `_reach` sparse matrix maps each token to all
its ancestor nodes, so the matvec computes all masses in a single
$O(|\mathcal{V}| \cdot \bar{L})$ pass ($\bar{L}$ = avg token length in bytes).
The log-sum-exp trick (factor out `logp_max`) avoids numerical underflow.

The conditional probability of byte $b$ at node $n$ is then:

$$P(b \mid n) = \exp\!\big(\text{mass}(\text{child}_b(n)) - \text{mass}(n)\big)$$

This is **exact marginalization** over all tokens whose byte decomposition
passes through `child_b(n)`.

## 5. The Correspondence

### 5.1 State-level mapping

| CharacterBeam | FusedTransducedLM |
|---|---|
| `TrieState` at trie root | Particle at hub DFA state (Q-absorbed) |
| `TrieState` at mid-trie node for prefix $(b_1, \ldots, b_j)$ | Particle at DFA state "mid-token: first $j$ bytes matched" |
| End-of-token extend (`>> None`) | Source-symbol expansion (consume token ID → return to hub) |
| `log_mass[child] - log_mass[node]` | Q-absorption: particle weight contributes directly to next-byte score |
| Beam pruning (top-$K$ TrieStates) | `max_beam` / `_select_top_k` on particles |
| Beam = carry-forward | Carry-forward dict + prefix-domination checks |

### 5.2 The trie mass *is* the Q-absorption sum

In FusedTransducedLM, when a particle at DFA state $d$ is classified as
$Q(y)$, its full weight $w$ is added to $\text{scores}[y]$. Over many
particles, this accumulates:

$$\text{score}(y) = \bigoplus_{\text{particle } p : d_p \in Q(y)} w_p$$

In CharacterBeam, the equivalent happens via the mixture over hypotheses.
Consider a hypothesis $h$ at the trie root with weight $w_h$ and LM state
$s_h$. Its contribution to $P(\text{byte } b)$ is:

$$w_h + \text{mass}_{s_h}(\text{child}_b(\text{root})) - \text{mass}_{s_h}(\text{root})$$

The mass ratio marginalizes over all tokens starting with byte $b$, each
weighted by $P_{\text{inner}}(t \mid s_h)$. This is precisely the sum that
FusedTransducedLM would compute by expanding each source symbol (token) from
the hub and Q-absorbing those that emit $b$ — but computed in one vectorized
pass instead of iterating over tokens in a priority queue.

### 5.3 What about hypotheses mid-trie?

A hypothesis mid-trie at node $n$ (having matched bytes $b_1, \ldots, b_j$ of
the current token) corresponds to a particle at an intermediate DFA state — one
where the FST is partway through a token's epsilon-output chain. In
FusedTransducedLM, this state would be expanded by source symbols, but most
expansions would lead to dead ends (only tokens whose byte decomposition
*continues* from $b_1 \ldots b_j$ are viable). The trie structure handles this
implicitly: only children of node $n$ are reachable, and the mass array already
accounts for this filtering.

## 6. What Enables the Specialization

Three properties of BPE WFSTs, acting together, enable CharacterBeam's
efficiency gains over the general algorithm:

### 6.1 All accepting states are universal (no remainder contributions)

The BPE WFST has a single hub state $()$ that is both start and sole accept
state. `check_all_input_universal` returns `True`: the input projection from
the hub accepts $\mathcal{X}^*$.

**Consequence**: every DFA state containing the hub with buffer $\geq$ target
is a **quotient state**. There are no remainder contributions — the $R$ term
vanishes entirely. This eliminates:

- The `P_{\text{inner}}(\text{EOS})$ weighting of remainder states.
- The preimage scoring path.
- The distinction between Q carry-forward (no prefix check) and R
  carry-forward (with prefix check) in `_FusedSearch._score_item`.

In CharacterBeam, this manifests as the absence of any EOS-probability logic.
Every completed tokenization prefix contributes its full weight.

### 6.2 Fixed output per source symbol (monoid homomorphism)

Each token $t$ always emits the same byte sequence $f(t)$, regardless of the
FST state. Combined with the single-hub structure, this means:

- A **single trie** represents all reachable tokens from any hypothesis.
- The mass array depends only on the LM state (through `logp_next`), not on
  the FST state or target position.
- Two hypotheses at the same trie node with the same LM state share the same
  mass array object.

This is what makes the O(|V|) vectorized scatter feasible: the trie is built
once and the mass array is computed once per LM state transition, then reused
for all subsequent byte positions until the next end-of-token.

### 6.3 Carry-forward degenerates to beam propagation

In FusedTransducedLM, carry-forward requires explicit source-path tracking and
prefix-domination checks (`_is_prefix_dominated`, `_cf_paths`). These prevent
double-counting when two particles at the same DFA state arrive via different
source paths.

In CharacterBeam, the beam *is* the carry-forward. After pruning and advancing
by one byte, the surviving hypotheses are exactly the particles for the next
step. No deduplication is needed because:

- Q-absorption means particles at the hub are not expanded further for the
  current target symbol. They contribute their weight and are carried forward
  directly.
- The only state at which "convergence" could happen (different source paths
  reaching the same DFA state) is the hub — but hub particles are Q-absorbed
  before expansion, so no two particles create overlapping descendants.

Note that CharacterBeam *does* track source paths — the LM state encodes the
full token history (via KV cache), and the trie node encodes the partial
current token. What's eliminated is the *explicit* source-path tuple and the
deduplication machinery.

### 6.4 LM calls deferred to token boundaries

Because the FST always returns to the hub after consuming a token ID, the LM
state only changes at end-of-token boundaries. Between boundaries, hypotheses
advance through the trie using only the precomputed mass array — no `state >> x`
calls needed. This amortizes the expensive LM transition cost ($O(1)$ LM call
per token vs. $O(\bar{L})$ byte steps per token).

In FusedTransducedLM, every source-symbol expansion calls `lm_state >> x` to
advance the inner LM, even when the resulting DFA state might be immediately
Q-absorbed. The trie-mass approach avoids these intermediate LM calls entirely.

## 7. Recovering Trie-Mass Efficiency in the General Setting

### 7.1 When is the trick applicable?

The trie-mass computation is recoverable at any DFA state where:

1. **Universal** (Q-absorbed): the full continuation probability contributes
   without EOS weighting, so the mass ratio is exact.

2. **Fixed output per source symbol at this state**: from this state, each
   source symbol produces a deterministic byte sequence, so a trie can
   represent the reachable tokens.

3. **Hub-like**: many hypotheses converge to this state (after completing
   tokens), so the cost of computing the mass array is amortized across
   multiple hypotheses and byte steps.

These conditions need not hold globally. They define a class of **hub states**
at which the fast path applies.

### 7.2 Multi-hub generalization

For an FST with $k$ hub states $h_1, \ldots, h_k$, each universal and
accepting, with deterministic output per source symbol from each hub:

- Build **$k$ tries**, one per hub, containing the tokens reachable from that
  hub with their byte decompositions from that hub.
- A hypothesis becomes $(s_{\text{LM}}, h_i, n_{\text{trie}},
  \text{mass}_{h_i})$ — the hub identity selects which trie to use.
- At end-of-token from hub $h_i$ consuming token $t$: transition to hub $h_j$
  (determined by the FST), switch to $h_j$'s trie, recompute mass array for
  the new LM state using $h_j$'s trie.
- Between token boundaries: advance through the current hub's trie using
  precomputed ratios, as in CharacterBeam.

Cost per token boundary: $O(|\mathcal{V}_{h_j}| \cdot \bar{L}_{h_j})$ for the
mass computation, where $|\mathcal{V}_{h_j}|$ is the vocabulary reachable from
hub $h_j$. This remains efficient when $k$ is small.

### 7.3 Hybrid: trie-mass at hubs, particle expansion elsewhere

When some FST states are not universal (have $R$ contributions), the trie-mass
trick doesn't apply because the ratio structure requires Q-absorption. But a
hybrid approach is natural:

- **At universal hub states**: use trie-mass computation. These hypotheses
  advance byte-by-byte using precomputed ratios, with LM calls only at token
  boundaries.

- **At non-universal states**: fall back to particle expansion via the standard
  FusedTransducedLM search. These hypotheses expand by source symbols, get
  classified, and scored via the Q/R/preimage logic.

The hybrid doesn't change the mathematics — it uses a faster computation for
the Q component at hub states. The practical payoff depends on what fraction of
probability mass flows through hub states:

| FST type | Hub fraction | Expected benefit |
|---|---|---|
| BPE tokenizer | 100% (single hub) | Full CharacterBeam speedup |
| Unigram tokenizer | 100% (single hub) | Full CharacterBeam speedup |
| PTB normalizer | High (few hub states) | Most mass uses fast path |
| Complex multi-state FST | Variable | Depends on topology |

### 7.4 Amortization economics

The trie mass costs $O(|\mathcal{V}|)$ per distinct LM state per token
boundary. FusedTransducedLM with `top_k` pruning costs $O(k)$ per particle per
byte step. The trie approach wins when the $O(|\mathcal{V}|)$ cost is amortized
across enough byte steps:

$$\frac{|\mathcal{V}| \cdot \bar{L}}{\bar{L}} = |\mathcal{V}| \quad \text{(trie: once per token boundary)}$$

$$K \cdot k \cdot \bar{L} \quad \text{(expansion: beam} \times \text{top-k} \times \text{bytes per token)}$$

For GPT-2 ($|\mathcal{V}| = 50\text{K}$, $\bar{L} \approx 4$, $K = 10$,
$k = 500$): trie = 50K vectorized ops once; expansion = $10 \times 500 \times 4
= 20\text{K}$ heap + LM ops per token. The trie operations are numpy scatter
(cache-friendly, SIMD); the expansion operations involve Python heap pushes,
LM state copies, and DFA lookups. Empirically this yields 3–12× speedup
(see Benchmark Dashboard).

## 8. Pseudocode

### 8.1 CharacterBeam (BPE-specialized)

Single hub, all-input-universal, monoid homomorphism.  Every hypothesis is a
trie hypothesis; there are no particles and no remainder logic.

```
Hypothesis = (lm_state, trie_node, log_mass, weight)

─── Trie mass ───

trie.scatter(logp_next) → log_mass[node] for all nodes:
    # O(|V|·L̄) vectorized scatter via COO index
    for each token t at leaf position idx:
        logp[idx] ← logp_next[t]
    log_mass[:] ← -∞
    logaddexp.at(log_mass, coo_nodes, logp[coo_token_ids])

─── Init ───

function INIT(lm, trie):
    s₀ ← lm.initial()
    mass₀ ← trie.scatter(s₀.logp_next)
    return beam = { (s₀, root, mass₀, 0.0) }

─── Scoring (one target step) ───

function SCORE(beam) → logp_next:

    # EXTEND: hypotheses at end-of-token complete the token, advance LM,
    # return to trie root with a fresh mass array.  Extended hypotheses
    # join the beam for this step's scoring.

    for (s, n, mass, w) ∈ beam:
        for token t reachable at end-of-token from n:
            w' ← w + mass[eot_child(n,t)] - mass[n]
            s' ← s >> t                              ← LM call (expensive)
            mass' ← trie.scatter(s'.logp_next)       ← O(|V|) scatter
            beam ← beam ∪ { (s', root, mass', w') }

    # SCORE: for each next target symbol y, accumulate the marginal
    # probability across all hypotheses.  The mass ratio at each
    # hypothesis is the exact marginalization over all tokens whose
    # byte decomposition continues with y from the current trie node.

    for each target symbol y:
        logp_next[y] ← logsumexp over (s, n, mass, w) ∈ beam
                        where y ∈ children(n):
                            w + mass[child_y(n)] - mass[n]

    return normalize(logp_next)

─── Advance (move to next target position) ───

function ADVANCE(beam, symbol y) → beam':
    beam ← top_K(beam, by weight)
    beam' ← ∅
    for (s, n, mass, w) ∈ beam:
        if y ∈ children(n):
            c ← children(n)[y]
            beam' ← beam' ∪ { (s, c, mass, w + mass[c] - mass[n]) }
    return beam'
```

Key properties:
- **No R/preimage/EOS logic.** All accepting states are universal, so every
  completed token returns to the hub and contributes via Q.  (EOS is handled
  by including it as a token in the trie with special sentinel bytes.)
- **No carry-forward dict.** The beam *is* the carry-forward.  ADVANCE moves
  each hypothesis through the trie; no DFA re-resolution needed.
- **No source-path tracking.** Q-absorbed hypotheses aren't expanded, so
  no two hypotheses create overlapping descendants.  No dedup needed.
- **LM calls only at token boundaries.** Between boundaries, hypotheses
  advance through the trie using the precomputed mass array.

### 8.2 Generalized CharacterBeam (arbitrary FST)

Zero, one, or many hubs.  Two hypothesis types: **TrieHyp** at IP-universal
accepting states (fast path) and **Particle** at non-hub DFA states
(slow path with Q/R/preimage classification).

```
TrieHyp  = (lm_state, hub, trie_node, log_mass, weight)
Particle = (dfa_state, lm_state, weight, source_path)

─── Precomputation ───

hubs ← { q ∈ fst.states : q is ip-universal AND q is accepting }

for each hub h:
    trie[h] ← build_trie_from_hub(fst, h)
        # Contains all tokens reachable from h.
        # Each token t records:
        #   bytes(h,t)  — output byte sequence when consuming t from h
        #   dest(h,t)   — FST state reached after consuming t from h
        # Precondition: bytes(h,t) is deterministic (fixed output per
        # token from this hub).  This holds for hub states where the
        # FST's output along each token's arc chain is unambiguous.

─── Init ───

function INIT(lm, fst):
    s₀ ← lm.initial()
    beam ← ∅
    for q₀ ∈ fst.start:
        if q₀ ∈ hubs:
            mass ← trie[q₀].scatter(s₀.logp_next)
            beam ← beam ∪ { TrieHyp(s₀, q₀, root, mass, 0.0) }
        else:
            dfa ← initial_dfa_state(q₀)
            beam ← beam ∪ { Particle(dfa, s₀, 0.0, ()) }
    return beam

─── Scoring ───

function SCORE(beam, target) → logp_next, carry_forward:

    scores ← LogVector()
    eos_score ← -∞
    carry_forward ← { }            # symbol → list of hypotheses

    # ──────────────────────────────────────────────────────────
    # PHASE 1: Trie hypotheses (hub fast path)
    #
    # Identical to CharacterBeam except:
    #   (a) each hub has its own trie
    #   (b) end-of-token may land at a non-hub → spawn Particle
    # ──────────────────────────────────────────────────────────

    new_particles ← []

    for h = (s, hub, n, mass, w) ∈ beam where h is TrieHyp:

        # EXTEND
        for token t reachable at end-of-token from n:
            w' ← w + mass[eot_child(n,t)] - mass[n]
            s' ← s >> t
            d ← dest(hub, t)

            if d ∈ hubs:                                     ← stays on fast path
                mass' ← trie[d].scatter(s'.logp_next)
                beam ← beam ∪ { TrieHyp(s', d, root, mass', w') }
            else:                                            ← falls off hub
                dfa ← resolve_dfa_state(d, target)
                new_particles ← new_particles ∪
                    { Particle(dfa, s', w', (t,)) }

        # SCORE (Q contribution via trie mass, same as CharacterBeam)
        for each symbol y ∈ children(n):
            c ← children(n)[y]
            scores[y] ⊕= w + mass[c] - mass[n]

    # ──────────────────────────────────────────────────────────
    # PHASE 2: Particle expansion (non-hub slow path)
    #
    # Standard FusedTransducedLM best-first search.
    # Handles Q-absorption, R scoring (with EOS), preimage,
    # and source-symbol expansion.
    # ──────────────────────────────────────────────────────────

    queue ← max_heap(
        { p ∈ beam : p is Particle } ∪ new_particles
    )
    steps ← 0

    while queue ≠ ∅ AND steps < max_steps:
        steps ← steps + 1
        item ← queue.pop_max()
        (q_sym, r_syms, is_preimage) ← classify(item.dfa_state)

        if q_sym ≠ None:                             ← Q-absorbed
            scores[q_sym] ⊕= item.weight
            carry_forward[q_sym].add(item)            # no dedup (Q)
            continue                                  # don't expand

        if is_preimage:                               ← preimage/EOS
            eos_score ⊕= item.weight
                         + item.lm_state.logp_next[EOS]

        for y ∈ r_syms:                              ← remainder
            scores[y] ⊕= item.weight
                         + item.lm_state.logp_next[EOS]
            carry_forward[y].add_checked(item)        # with dedup

        # Expand by source symbols
        for (x, dest_dfa) ∈ dfa_arcs(item.dfa_state):
            w' ← item.weight + item.lm_state.logp_next[x]
            if w' > -∞:
                queue.push(Particle(
                    dest_dfa, item.lm_state >> x,
                    w', item.source_path + (x,)
                ))

    # Drain: score remaining without expanding
    while queue ≠ ∅:
        score_and_carry_forward(queue.pop())

    scores[EOS] ⊕= eos_score
    return scores.normalize(), carry_forward

─── Advance ───

function ADVANCE(beam, carry_forward, symbol y) → beam':
    beam ← top_K(beam, by weight)
    beam' ← ∅

    # Trie hypotheses: advance through trie (same as CharacterBeam)
    for h = (s, hub, n, mass, w) ∈ beam where h is TrieHyp:
        if y ∈ children(n):
            c ← children(n)[y]
            beam' ← beam' ∪ { TrieHyp(s, hub, c, mass,
                                        w + mass[c] - mass[n]) }

    # Particles: bring in from carry-forward, re-resolve in new DFA
    for p ∈ carry_forward.get(y, []):
        dfa' ← resolve_in_new_dfa(p.source_path, target ++ y)
        if dfa' ≠ None:
            beam' ← beam' ∪ { Particle(dfa', p.lm_state,
                                        p.weight, p.source_path) }

    return top_K(beam', K)
```

### 8.3 Special cases

The generalized algorithm degenerates to the existing algorithms at the
endpoints of the hub spectrum:

**When `hubs = ∅`** (e.g., `triplets_of_doom`): Phase 1 is empty.
All hypotheses are Particles.  SCORE reduces to pure FusedTransducedLM
best-first search.  ADVANCE uses only the carry-forward dict.  The trie
machinery is never invoked.

**When `hubs = {single hub}` and all tokens return to that hub** (e.g.,
BPE WFST): Phase 2 is empty.  `new_particles` is always empty because
`dest(hub, t) = hub` for all tokens $t$.  SCORE reduces to the CharacterBeam
EXTEND + trie-mass computation.  ADVANCE uses only the trie path.  The
DFA/classify/carry-forward machinery is never invoked.

**When `hubs ≠ ∅` but some tokens leave hubs** (e.g., `backticks_to_quote`):
Both phases are active.  A hypothesis at `START` (hub) that extends with
token `` ` `` lands at `Quote` (IP-universal but not accepting, so not a hub);
this spawns a Particle.  Most mass stays on the trie path through `START`;
the Particle-expansion path handles the `1_Quote` remainder residual.

### 8.4 Conversion between hypothesis types

The two hypothesis types convert at well-defined boundaries:

**TrieHyp → Particle** (hub exit): occurs in Phase 1 EXTEND when
`dest(hub, t) ∉ hubs`.  The TrieHyp completes a token whose destination is
a non-hub FST state.  The new Particle carries a minimal source path `(t,)` —
only the single token that departed the hub.

**Particle → TrieHyp** (hub entry): not shown in the pseudocode above.
Could occur when a Particle arrives at a DFA state that is Q-absorbed and
the underlying FST state is a hub.  In principle, the particle could be
"promoted" to a TrieHyp to benefit from trie-mass scoring at the next step.
In practice, Q-absorption already handles the scoring correctly — the particle
contributes its full weight to the Q symbol and is carried forward.  The
promotion would save work at the *next* target step by enabling trie-mass
scoring there.  Whether this is worthwhile depends on how often particles
reach hubs; for most practical FSTs, the hub-exit path is rare enough that
the added complexity isn't justified.

## 9. Examples: multi-hub FSTs with remainders

To ground the discussion in Sections 7–8, we trace the hybrid trie-mass / particle
expansion approach through three FSTs from `examples.py` with increasing
structural complexity.

### 9.1 Triplets of doom: single hub, no IP-universal states

```
State 0 (start, accept):  a:a → 1,  b:b → 2
State 1:                   a:a → 3
State 2:                   b:b → 4
State 3:                   a:a → 0
State 4:                   b:b → 0
```

**Structure.** A copy transducer for $(aaa \mid bbb)^*$.  State 0 is the sole
hub (start and only accepting state).  States 1–4 are transient waypoints
inside a triplet.

**Universality.**  `check_all_input_universal` = False.
`compute_ip_universal_states` = $\emptyset$.  No individual FST state is
IP-universal because each of states 1–4 has arcs for only one source symbol.
State 0 has arcs for both `a` and `b`, but after `a` → state 1, only `a`
continues — so the start-set containment invariant fails.

**Consequence for the hybrid.**  No state qualifies as a trie-mass hub.  Every
DFA state must use the general particle-expansion path.  Universality can only
be detected via BFS on composite DFA states (e.g., a powerset state that covers
both `a` and `b` branches simultaneously — which never arises for this FST
since the branches are disjoint).

**Remainder behavior.**  Consider target `"a"`.  A source prefix `"aaa"` returns
to state 0 (accepting) and has produced `"aaa"`.  But from state 0, the source
can continue with `"bbb"` producing output `"bbb"` — so `"aaa"` is NOT
universal for any next target symbol (e.g., for next symbol `"a"`, the
continuation `"b"` at state 0 doesn't produce `"a"`).  Source paths at state 0
contribute via R (weighted by $P(\text{EOS})$), not Q.

This is the worst case for CharacterBeam-style optimization: **no hubs, no
trie-mass, full particle expansion required.**

### 9.2 Number-comma separator: two IP-universal hubs

```
State 0 (start, accept):  d:d → 0 (∀ digit d),  x:x → 0 (∀ non-comma x),  ',':', ' → 1
State 1 (accept):         d:d → 0 (∀ digit d),  ε:'|' → 2
State 2:                  ',':', ' → 1,  x:x → 0 (∀ non-digit non-comma x)
```

(`number_comma_separator` with `Domain = Digit ∪ {',', ' '}`)

**Structure.**  State 0 is the main hub: identity pass-through for all non-comma
symbols, with comma triggering a transition to state 1.  State 1 is a secondary
hub: if a digit follows the comma, it was a numeric comma (return to 0);
otherwise, epsilon-emit `|` and enter state 2.  State 2 routes back to the hubs.

**Universality.**  `check_all_input_universal` = False.
`compute_ip_universal_states` = $\{0, 1\}$.  Both states 0 and 1 are
IP-universal: their epsilon closures have arcs for every symbol in the domain,
and all successors' closures contain at least one IP-universal state.  State 2
is NOT IP-universal (missing digit arcs).

**Consequence for the hybrid.**  The trie-mass trick applies at **two hubs**:

- **Hub 0 trie**: tokens = all symbols in Domain.  Each "token" is a single
  symbol producing itself (a 1-to-1 map).  Hypotheses at hub 0 use trie-mass
  ratios for O(1) per-symbol scoring.

- **Hub 1 trie**: tokens reachable from state 1's epsilon-closure
  $\{1, 2\}$.  This includes digits (via state 1 → state 0, output = digit)
  and non-digit-non-comma symbols (via state 2 → state 0).  A separate trie
  for hub 1 captures this different vocabulary.

- **State 2 paths**: when a hypothesis transits through state 2 (after the
  `ε:'|'` arc), it's in a non-universal non-accepting state.  This requires
  particle expansion — but only for one step, until the source enters the
  next hub.

**Remainder behavior.**  State 1 is accepting and IP-universal, so when it
appears in a DFA state, it contributes via Q (not R).  However, consider a
DFA state containing only state 2 elements (e.g., after the epsilon transition
from state 1).  State 2 is not accepting — so no R contribution there.  The R
term is empty for most target prefixes because both accepting states are
IP-universal.  This FST is in the "mostly Q, trie-mass dominates" regime.

### 9.3 Backticks to quote: hub-dominated FST with remainder residual

```
START (accept):     a:ε → CHAR_a,   '`':ε → Quote
CHAR_a:             ε:b → START
Quote:              ε:'`' → 1_Quote,   '`':'"' → 2_quotes
1_Quote (accept):   a:ε → CHAR_a
2_quotes (accept):  ε:ε → START
```

(`backticks_to_quote`: single backtick passes through, double backtick →
`"`, `a` → `b`)

**Structure.**  `START` is the main hub.  `Quote` is a "lookahead" state
(consumed the first backtick, waiting to see if a second follows).
`1_Quote` is the state after committing to a single backtick (output `` ` ``).
`2_quotes` is the state after consuming both backticks (output `"`).

**Universality.**  `check_all_input_universal` = False.
`compute_ip_universal_states` = $\{\text{START}, \text{Quote}, \text{CHAR\_a},
\text{2\_quotes}\}$.  Four of five states are IP-universal.  The exception is
**`1_Quote`**, which is accepting but has only one input arc (`a`).

**IP-universal states as hubs.**

| State | Accepting | IP-universal | Hub candidate |
|---|---|---|---|
| `START` | yes | yes | yes — main hub |
| `Quote` | no | yes | no — not accepting |
| `CHAR_a` | no | yes | no — not accepting |
| `1_Quote` | yes | **no** | no — remainder source |
| `2_quotes` | yes | yes | yes — secondary hub |

**Remainder behavior.**  `1_Quote` is the interesting state: it's accepting
(the source can stop after a single backtick, producing `` ` `` as output) but
NOT IP-universal (only `a` continues from here — missing `` ` `` arcs).  When
a particle reaches a DFA state containing `1_Quote` elements:

- The state is **not Q-absorbed** (not universal — continuation `` ` `` is
  impossible from `1_Quote`).
- The state **is a preimage/remainder contributor**: `1_Quote` is accepting,
  so $P(\text{EOS})$ weight scores the EOS/target symbols.
- The particle **must be expanded** by source symbols to discover further
  contributions.

In the hybrid approach:
- Hypotheses at `START` and `2_quotes` use **trie-mass computation** (fast
  path, no EOS weighting).
- Hypotheses reaching `1_Quote` require **particle expansion** (slow path,
  with $P(\text{EOS})$ scoring for the R contribution).
- Since most probability mass flows through `START` (the main hub), the
  trie-mass fast path dominates.

### 9.4 Scaled newspeak: all states IP-universal but not AIU

```
State 0 (accept):  non-trigger x → x:x → 0,  trigger i → i:ε → (i+1)
State k (accept, k=1..n_patterns):  completion symbol → out_k → 0,  other x → x:x → k
```

(`scaled_newspeak(n_patterns=3, alpha_size=5)`)

**Universality.**  `check_all_input_universal` = False (trigger symbols break
start-set containment).  But `compute_ip_universal_states` = $\{0, 1, 2, 3\}$
— **all states are IP-universal**.  Each state individually accepts
$\mathcal{X}^*$ because it has arcs for every symbol.

**Consequence.**  Despite AIU being false, **every DFA state that contains any
FST state will be detected as universal** via the IP-universality witness check.
This means Q-absorption applies everywhere — the R term vanishes, just like BPE.
The hybrid degenerates to pure trie-mass at $k+1$ hubs (one per state).

This is the intermediate regime: **structurally richer than BPE** (multiple hubs,
context-dependent output — trigger symbols produce different output from
different states) **but behaviorally equivalent** (all Q, no R, trie-mass
dominates).

### 9.5 Summary of examples

| FST | Hubs | IP-univ states | R contributions | Trie-mass applicable |
|---|---|---|---|---|
| BPE WFST | 1 | all (AIU) | none | 100% (CharacterBeam) |
| `scaled_newspeak` | $k{+}1$ | all | none | 100% ($k{+}1$ tries) |
| `number_comma_separator` | 2 | 2 of 3 | rare (transient) | most mass |
| `backticks_to_quote` | 2 | 4 of 5 | at `1_Quote` | most mass |
| `triplets_of_doom` | 0 | 0 of 5 | all scoring | 0% (full expansion) |

The spectrum runs from "pure trie-mass" (BPE) through "hybrid with small R
residual" (backticks, comma separator) to "pure particle expansion" (triplets
of doom).  The practical payoff of the hybrid depends on how much probability
mass flows through IP-universal hub states — which, for the FSTs arising in
NLP tokenization, is typically the dominant case.

## 10. Summary

CharacterBeam is FusedTransducedLM specialized to **monoid-homomorphism FSTs
where all accepting states are universal**. The specialization yields three
linked efficiency gains:

1. **Q-only scoring** → no remainder logic, no EOS weighting, carry-forward
   degenerates to beam propagation (no source-path deduplication needed because
   Q-absorbed particles at the hub aren't expanded, preventing the convergence
   that would require dedup).

2. **Trie-mass precomputation** → the Q-absorption marginalization is computed
   as a single $O(|\mathcal{V}|)$ vectorized scatter, replacing per-step
   priority-queue search.

3. **Deferred LM calls** → LM state transitions happen only at token
   boundaries, amortized across $\bar{L}$ byte steps.

These gains are recoverable in the general setting at any **universal hub state
with fixed output per source symbol**. The multi-hub generalization ($k$ tries)
and the hybrid approach (trie-mass at hubs, particle expansion elsewhere)
provide a smooth path from the fully general algorithm to the fully specialized
one, with the specialization triggered by local structural properties of the
FST rather than requiring global assumptions.

The examples in Section 9 show the spectrum: from FSTs where the trie-mass
trick covers 100% of the computation (BPE, scaled newspeak) through FSTs where
it covers most of the mass with a small particle-expansion residual (backticks,
comma separator) to FSTs where it cannot help at all (triplets of doom).  For
the tokenization FSTs that motivate this work, the "mostly hub" regime is the
common case.
