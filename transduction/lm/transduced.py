"""
Incremental transduced language model with particle-based approximate inference.

Given an inner LM P_inner(source) and an FST mapping source -> target,
TransducedLM estimates P(y | target_so_far) by maintaining a beam of K
particles (source-side hypotheses) that evolve as each target symbol is added.
Each particle tracks a DFA state (powerset of precover NFA states) and an
inner LM state; at quotient states, exact marginalization sums over infinitely
many source continuations, providing dramatic variance reduction.

Per-step expansion uses beam search through the decomposition's DFA with beam
width K.  Layers proceed by source-symbol depth: at each layer, all beam
particles are classified (Q/R/preimage), scored, and non-Q particles are
expanded.  Top K candidates form the next beam.  Beam search terminates when:
  - The beam is empty (all particles absorbed by Q/R), or
  - The monotone weight bound is satisfied: the best expandable particle has
    weight <= the K-th highest carry-forward weight.  Since particle weights
    only decrease with source-symbol depth, no descendant can displace the
    current top-K carry-forward set.

Two inference modes:
  'beam_sum' -- deterministic top-K pruning; consistent as K -> inf.
  'sir'      -- sequential importance resampling; provides unbiased estimates
                of the target prefix probability p(y_{1:t}) for any K.

Usage:
    from transduction import examples
    from transduction.lm.ngram import CharNgramLM
    from transduction.lm.transduced import TransducedLM

    inner = CharNgramLM.train("hello world", n=2)
    fst = examples.lowercase()
    tlm = TransducedLM(inner, fst)

    state = tlm >> 'h'
    print(state.logp_next['e'])
    print(state.logp)
"""

import heapq
import numpy as np

from transduction.lm.base import LM, LMState, LogpNext

# Default incremental decomposition; can be overridden via TransducedLM constructor.
from transduction.peekaboo_incremental import PeekabooState as _DefaultDecompState
from transduction.peekaboo_incremental import FstUniversality as _DefaultUniv


def logsumexp(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = arr[arr > -np.inf]
    if len(arr) == 0:
        return -np.inf
    vmax = arr.max()
    arr -= vmax
    np.exp(arr, out=arr)
    out = np.log(arr.sum())
    out += vmax
    return out


def _to_key(token):
    """Coerce token to a string key for output symbols."""
    if isinstance(token, (bytes, bytearray)):
        return token.decode('latin-1')
    if isinstance(token, int):
        return chr(token)
    if isinstance(token, str):
        return token
    return None


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

class Particle:
    """A weighted source-prefix hypothesis.

    Tracks a DFA state (powerset of precover NFA states) that captures the
    transducer's progress, the inner LM state, and a log importance weight.
    """
    __slots__ = ('dfa_state', 'lm_state', 'log_weight')

    def __init__(self, dfa_state, lm_state, log_weight):
        self.dfa_state = dfa_state
        self.lm_state = lm_state
        self.log_weight = log_weight

    def __lt__(self, other):
        # max-heap: higher weight = higher priority
        return self.log_weight > other.log_weight

    def __repr__(self):
        return f'Particle(w={self.log_weight:.3f})'


# Backward-compat alias for FusedTransducedLM and benchmarks.
class BeamItem:
    __slots__ = ('dfa_state', 'lm_state', 'weight')
    def __init__(self, dfa_state, lm_state, weight):
        self.dfa_state = dfa_state
        self.lm_state = lm_state
        self.weight = weight
    def __lt__(self, other):
        return self.weight > other.weight


# ---------------------------------------------------------------------------
# Selection / resampling
# ---------------------------------------------------------------------------

def _select_top_k(particles, k):
    """Select the k highest-weight particles in O(n) expected time."""
    n = len(particles)
    if n <= k:
        return list(particles)
    weights = np.array([p.log_weight for p in particles])
    # argpartition: after partitioning, the k largest are at indices[n-k:]
    indices = np.argpartition(weights, n - k)[n - k:]
    return [particles[i] for i in indices]


def _resample(particles, k, rng):
    """Multinomial resampling: draw k particles proportional to exp(weight).

    Returns (resampled_particles, log_normalizer) where log_normalizer =
    logsumexp(weights), the total evidence before normalization.
    """
    if not particles:
        return [], -np.inf
    weights = np.array([p.log_weight for p in particles])
    log_Z = logsumexp(weights)
    if log_Z == -np.inf:
        return [], -np.inf
    probs = np.exp(weights - log_Z)
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    indices = rng.choice(len(particles), size=k, replace=True, p=probs)
    resampled = [
        Particle(particles[i].dfa_state, particles[i].lm_state, 0.0)
        for i in indices
    ]
    return resampled, log_Z


# ---------------------------------------------------------------------------
# TransducedState
# ---------------------------------------------------------------------------

class TransducedState(LMState):
    """Immutable state for the TransducedLM.

    Supports:
        state >> y         -> new state (advance by target symbol y)
        state.logp_next[y] -> log P(y | target_so_far)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token

    Each state holds K particles (source-prefix hypotheses). Computing
    logp_next runs beam search (width K) through the decomposition's DFA:
    at each layer, particles are classified, scored at Q/R/preimage states,
    and non-Q particles are expanded by source symbols. The carry-forward
    mechanism passes particles at Q/R/resume_frontier states to the next
    target step.
    """

    def __init__(self, tlm, peekaboo_state, particles, logp, history=()):
        self.tlm = tlm
        self.eos = tlm.eos
        self._peekaboo_state = peekaboo_state
        self._particles = particles
        self.logp = logp
        self.history = history
        self._logp_next_cache = None
        self._carry_forward_cache = None
        self._raw_scores_cache = None

    def _ensure_computed(self):
        if self._logp_next_cache is None:
            self._compute_logp_next()

    @property
    def logp_next(self):
        self._ensure_computed()
        return self._logp_next_cache

    def __rshift__(self, y):
        if y == self.eos or y not in self.tlm.fst.B:
            raise ValueError(f"Out of vocabulary: {y!r}")
        self._ensure_computed()

        key = _to_key(y)
        if key is None or key not in self._logp_next_cache:
            raise ValueError(f"Out of vocabulary: {y!r}")

        lp_y = self._logp_next_cache[y]

        # Advance peekaboo decomposition state
        new_peekaboo = self._peekaboo_state >> key

        # Get carry-forward particles for this symbol
        cf_particles = self._carry_forward_cache.get(key, [])

        K = self.tlm.K

        if self.tlm.method == 'sir':
            # SIR prefix probability:
            #   log p-hat(y_t | y_{<t}) = log[ (1/K) * sum_k C_k(y_t) ]
            #                           = raw_score(y_t) - log(K_input)
            K_input = max(len(self._particles), 1)
            raw_y = self._raw_scores_cache.get(key, -np.inf)
            new_logp = self.logp + raw_y - np.log(K_input)
            new_particles, _ = _resample(cf_particles, K, self.tlm._rng)
        else:
            # Beam-sum prefix probability: product of normalized conditionals
            new_logp = self.logp + lp_y
            new_particles = _select_top_k(cf_particles, K)

        return TransducedState(
            self.tlm, new_peekaboo, new_particles,
            new_logp, history=(self.history, y),
        )

    def _compute_logp_next(self):
        """Beam search through the DFA to score next target symbols.

        Layer-by-layer beam search with beam width K:
        1. Classify all beam particles: Q particles score and exit (exact
           marginalization), R particles score (weight * P(EOS)), preimage
           particles score outer EOS.
        2. Carry-forward: particles at Q/R/resume_frontier states are saved
           (deduplicated by object identity to prevent double-counting).
        3. Expand non-Q particles by source symbols.
        4. Select top K candidates -> new beam.
        5. Terminate when beam is empty or monotone weight bound is met:
           best expandable weight <= K-th highest carry-forward weight.
           Since weights only decrease with depth, no descendant can
           displace the current top-K carry-forward set.
        6. Normalize scores -> conditional distribution.
        """
        decomp = self._peekaboo_state.decomp
        dfa = self._peekaboo_state.dfa

        # --- Per-state classification lookups ---
        q_of = {}       # dfa_state -> set of y where state in Q(y)
        r_of = {}       # dfa_state -> set of y where state in R(y)
        resume_of = {}  # dfa_state -> set of y where state in resume(y)

        for y, d in decomp.items():
            for state in d.quotient:
                q_of.setdefault(state, set()).add(y)
            for state in d.remainder:
                r_of.setdefault(state, set()).add(y)
        for y, states in self._peekaboo_state.resume_frontiers.items():
            for state in states:
                resume_of.setdefault(state, set()).add(y)

        preimage_stops = self._peekaboo_state.preimage_stops

        # Inner LM EOS token
        EOS = (self.tlm.inner_lm.eos if hasattr(self.tlm.inner_lm, 'eos')
               else self.tlm.inner_lm.initial().eos)

        K = self.tlm.K

        # --- Accumulators ---
        scores = {}          # y -> [log-weights]
        eos_scores = []      # [log-weights] for outer EOS
        # carry_forward: y -> {id(particle): particle}
        # Dict-keyed by id() to prevent the same particle object from being
        # added multiple times (a state can be Q(y) AND resume_frontier(y)).
        carry_forward = {}

        def _add_carry(y, particle):
            cf = carry_forward.setdefault(y, {})
            cf[id(particle)] = particle

        # Top-K carry-forward weight tracker (min-heap of size K).
        # cf_top_k[0] is the K-th highest carry-forward weight when full.
        cf_top_k = []

        def classify_and_score(particle):
            """Classify particle's DFA state, accumulate scores, save carry-forward.

            Returns True if the particle is at a quotient state (skip expansion).
            """
            d = particle.dfa_state
            w = particle.log_weight

            q_syms = q_of.get(d)
            r_syms = r_of.get(d)
            is_preimage = d in preimage_stops
            is_quotient = q_syms is not None

            # Only trigger an LM forward pass (logp_next[EOS]) when needed:
            # preimage scoring, or remainder for symbols not covered by quotient.
            needs_eos = is_preimage or (
                r_syms is not None
                and (not is_quotient or not r_syms.issubset(q_syms))
            )
            eos_lp = None
            if needs_eos:
                eos_lp = particle.lm_state.logp_next[EOS]

            # Preimage: source produced exactly the target prefix at a final state
            if is_preimage and eos_lp is not None and eos_lp > -np.inf:
                eos_scores.append(w + eos_lp)

            # Quotient: full weight contributes (exact marginalization over futures)
            if is_quotient:
                for y in q_syms:
                    scores.setdefault(y, []).append(w)

            # Remainder: weight * P(EOS), skipping symbols covered by quotient
            if r_syms is not None and eos_lp is not None and eos_lp > -np.inf:
                eos_w = w + eos_lp
                for y in r_syms:
                    if not is_quotient or y not in q_syms:
                        scores.setdefault(y, []).append(eos_w)

            # Carry-forward: save particle at resume_frontier / Q / R states.
            # The id()-keyed dict ensures each particle is saved at most once
            # per symbol, even when a state is in multiple sets.
            carry_syms = set()
            rf = resume_of.get(d)
            if rf is not None:
                carry_syms |= rf
            if q_syms is not None:
                carry_syms |= q_syms
            if r_syms is not None:
                carry_syms |= r_syms
            for y in carry_syms:
                _add_carry(y, particle)

            # Track carry-forward weights for early termination.
            if carry_syms:
                if len(cf_top_k) < K:
                    heapq.heappush(cf_top_k, w)
                elif w > cf_top_k[0]:
                    heapq.heapreplace(cf_top_k, w)

            return is_quotient

        # --- Beam search ---
        # Each iteration: classify candidates -> prune non-Q to K -> expand.
        # Initial "candidates" are the carry-forward particles from the parent.
        # Classification happens BEFORE pruning so that all Q/R candidates
        # are scored regardless of beam width.
        candidates = list(self._particles)

        for _ in range(self.tlm.max_layers):
            if not candidates:
                break

            # Classify all candidates: Q/R score and exit, non-Q continue.
            to_expand = []
            for particle in candidates:
                is_q = classify_and_score(particle)
                if not is_q:
                    to_expand.append(particle)

            if not to_expand:
                break

            # Early termination: monotone weight bound.
            # Particle weights only decrease with source-symbol depth, so if
            # the best non-Q candidate can't beat the K-th highest carry-
            # forward weight, no descendant can either.
            if len(cf_top_k) >= K:
                best_w = max(p.log_weight for p in to_expand)
                if best_w <= cf_top_k[0]:
                    break

            # Prune non-Q candidates to K (beam width).
            beam = _select_top_k(to_expand, K)

            # Expand beam particles by source symbols.
            candidates = []
            for particle in beam:
                lm_logp = particle.lm_state.logp_next
                for x, next_dfa_state in dfa.arcs(particle.dfa_state):
                    child_w = particle.log_weight + lm_logp[x]
                    if child_w > -np.inf:
                        candidates.append(Particle(
                            next_dfa_state,
                            particle.lm_state >> x,
                            child_w,
                        ))

        # --- Normalize ---
        raw_scores = {}
        for y, w_list in scores.items():
            raw_scores[y] = logsumexp(w_list)

        eos_raw = logsumexp(eos_scores)

        Z_parts = list(raw_scores.values())
        if eos_raw > -np.inf:
            Z_parts.append(eos_raw)
        Z = logsumexp(Z_parts)

        normalized = {}
        for y, raw in raw_scores.items():
            normalized[y] = raw - Z
        normalized[self.tlm.eos] = eos_raw - Z if Z > -np.inf else -np.inf

        self._logp_next_cache = LogpNext(normalized)
        self._raw_scores_cache = raw_scores
        # Convert carry_forward id-dicts to lists
        self._carry_forward_cache = {
            y: list(d.values()) for y, d in carry_forward.items()
        }

    def path(self):
        tokens = []
        h = self.history
        while h:
            h, token = h
            tokens.append(token)
        tokens.reverse()
        return tokens

    def __repr__(self):
        return (f'TransducedState(target={self._peekaboo_state.target!r},'
                f' K={len(self._particles)})')


# ---------------------------------------------------------------------------
# TransducedLM
# ---------------------------------------------------------------------------

class TransducedLM(LM):
    """Incremental transduced language model with particle-based inference.

    Computes the pushforward of an inner LM through an FST using a beam of
    K particles.  Each particle tracks a DFA state (powerset of precover NFA
    states) and an inner LM state.  At quotient states, exact marginalization
    sums over infinitely many source continuations.

    Per-step expansion uses beam search with beam width K through the
    decomposition's DFA.  Layers proceed by source-symbol depth: at each
    layer, all beam particles are classified (Q/R/preimage), scored, and
    non-Q particles are expanded.  Top K candidates form the next beam.
    Termination is by the monotone weight bound: since particle weights only
    decrease with depth, expansion stops when no beam particle can displace
    the current top-K carry-forward set.

    Args:
        inner_lm: LM with StateLM interface (has .initial(), state >> x,
            state.logp_next).
        fst: FST instance mapping source -> target.
        K: Beam width — number of particles maintained during both per-step
            expansion and carry-forward between target steps.  This is the
            primary knob controlling approximation quality.
        max_layers: Maximum beam search depth (source-symbol layers) per
            target step.  The primary termination criterion is the monotone
            weight bound (automatic); this is a safety valve for pathological
            FSTs where weight decay is very slow.
        eos: Outer EOS token.
        method: 'beam_sum' (deterministic top-K, consistent) or 'sir'
            (sequential importance resampling, unbiased prefix probability).
        decomp_state_cls: Incremental decomposition class (default: PeekabooState).
        univ_cls: Universality precomputation class (default: FstUniversality).
        seed: Random seed for SIR resampling.
        max_beam: Backward-compat alias for K.
        max_steps: Accepted for backward compat (unused).
        max_expansions: Accepted for backward compat (unused).
    """

    def __init__(self, inner_lm, fst, K=100, max_layers=100,
                 max_expansions=None, eos='<EOS>',
                 method='beam_sum', decomp_state_cls=None, univ_cls=None,
                 seed=None,
                 max_beam=None, max_steps=None):
        self.inner_lm = inner_lm
        self.fst = fst
        self.K = max_beam if max_beam is not None else K
        self.max_layers = max_layers
        # Backward-compat attribute (no longer controls beam search).
        self.max_expansions = max_steps or max_expansions or 1000
        self.eos = eos
        self.method = method
        self._decomp_state_cls = decomp_state_cls or _DefaultDecompState
        self._univ_cls = univ_cls or _DefaultUniv
        self._univ = self._univ_cls(fst)
        self._rng = np.random.default_rng(seed)

    def initial(self):
        """Return the initial TransducedState (empty target prefix)."""
        peekaboo = self._decomp_state_cls(
            self.fst, '', parent=None, univ=self._univ,
        )
        dfa = peekaboo.dfa
        start_states = list(dfa.start())

        inner_initial = self.inner_lm.initial()
        particles = [
            Particle(s, inner_initial, 0.0)
            for s in start_states
        ]

        return TransducedState(self, peekaboo, particles, 0.0)

    def __repr__(self):
        return (f'TransducedLM(inner={self.inner_lm!r},'
                f' K={self.K}, method={self.method!r})')
