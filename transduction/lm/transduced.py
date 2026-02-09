"""
Incremental transduced language model.

Given an inner LM P_inner(source) and an FST mapping source → target,
TransducedLM computes P(y | target_so_far) by marginalizing over all
source strings that produce the target prefix, carrying a beam of
source-side search items forward across target steps.

Usage:
    from transduction import examples
    from transduction.lm.ngram import ByteNgramLM
    from transduction.lm.transduced import TransducedLM

    inner = ByteNgramLM.train(b"hello world", n=2)
    fst = examples.lowercase()
    tlm = TransducedLM(inner, fst)

    state = tlm.initial()
    state = state << 'h'
    print(state.logp_next['e'])
    print(state.logp)
"""

import heapq

import numpy as np

from transduction.lm.base import LMState, LogpNext

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


class BeamItem:
    """Search frontier element for source-side expansion."""
    __slots__ = ('dfa_state', 'lm_state', 'weight')

    def __init__(self, dfa_state, lm_state, weight):
        self.dfa_state = dfa_state
        self.lm_state = lm_state
        self.weight = weight

    def __lt__(self, other):
        return self.weight > other.weight  # higher weight = higher priority


class TransducedState(LMState):
    """Immutable state for the TransducedLM.

    Supports:
        state << y         -> new state (advance by target symbol y)
        state.logp_next[y] -> log P(y | target_so_far)
        state.logp         -> cumulative log probability
        state.eos          -> EOS token
    """

    def __init__(self, tlm, peekaboo_state, beam, logp, history=()):
        self.tlm = tlm
        self.eos = tlm.eos
        self._peekaboo_state = peekaboo_state
        self._beam = beam
        self.logp = logp
        self.history = history
        self._logp_next_cache = None
        self._carry_forward_cache = None

    def _ensure_computed(self):
        if self._logp_next_cache is None:
            self._compute_logp_next()

    @property
    def logp_next(self):
        self._ensure_computed()
        return self._logp_next_cache

    def __lshift__(self, y):
        if y == self.eos or y not in self.tlm.fst.B:
            raise ValueError(f"Out of vocabulary: {y!r}")
        self._ensure_computed()

        # Resolve y to its string key
        key = _to_key(y)
        if key is None or key not in self._logp_next_cache:
            raise ValueError(f"Out of vocabulary: {y!r}")

        lp_y = self._logp_next_cache[y]

        # Advance peekaboo state
        new_peekaboo = self._peekaboo_state >> key

        # Get carry-forward beam for this symbol
        new_beam = self._carry_forward_cache.get(key, [])

        # Prune to max_beam (keep highest-weight items)
        if len(new_beam) > self.tlm.max_beam:
            new_beam = sorted(new_beam, key=lambda it: it.weight, reverse=True)
            new_beam = new_beam[:self.tlm.max_beam]

        return TransducedState(
            self.tlm, new_peekaboo, new_beam,
            self.logp + lp_y,
            history=(self.history, y),
        )

    # TODO (timv): I still need to review and test this method carefully.
    def _compute_logp_next(self):
        """Core algorithm: bounded best-first search over source-side paths.

        Computes P(y | target_so_far) for each next target symbol y by
        searching over source-side paths weighted by the inner LM.

        The search is guided by the PeekabooState's decomposition, which
        classifies DFA states into four roles for each target symbol y:

        - Quotient Q(y): all source continuations from this state produce y
          as the next target symbol.  Contributes the full beam weight to
          scores[y].  These states are *not* expanded further — they already
          account for all continuations.

        - Remainder R(y): the source can stop here (EOS) and still produce y.
          Contributes weight + log P_inner(EOS) to scores[y].  Only counted
          when the state is not already a quotient for y (to avoid
          double-counting).

        - Preimage: the source has produced exactly the current target prefix
          and the FST is in a final state.  Same as remainder, but for the
          outer EOS symbol rather than a regular target symbol.

        - Resume frontier: states at the truncation boundary that should be
          carried forward to seed the next step's search (after the user
          advances by y).  These are saved in carry_forward[y] but do not
          contribute to scores.

        Q and R states are also added to carry_forward[y] so they can seed
        the next step.

        After the search, scores are normalized via logsumexp to produce a
        proper conditional distribution.
        """
        decomp = self._peekaboo_state.decomp
        dfa = self._peekaboo_state.dfa

        # The decomposition is keyed by symbol (decomp[y].quotient = set of
        # DFA states), but the search visits states one at a time, so we
        # invert to state → set of symbols for efficient lookup.
        q_lookup = {}   # dfa_state → {y: state is a quotient stop for y}
        r_lookup = {}   # dfa_state → {y: state is a remainder stop for y}
        for y, d in decomp.items():
            for state in d.quotient:
                q_lookup.setdefault(state, set()).add(y)
            for state in d.remainder:
                r_lookup.setdefault(state, set()).add(y)

        resume_states = {}  # dfa_state → {y: state is a resume frontier for y}
        for y, states in self._peekaboo_state.resume_frontiers.items():
            for state in states:
                resume_states.setdefault(state, set()).add(y)

        # Accumulators populated by score_stops
        scores = {}         # y → [log-weights] for regular target symbols
        eos_scores = []     # [log-weights] for outer EOS
        carry_forward = {}  # y → [BeamItems] to seed the next step

        EOS = self.tlm.inner_lm.eos if hasattr(self.tlm.inner_lm, 'eos') else self.tlm.inner_lm.initial().eos
        preimage_lookup = self._peekaboo_state.preimage_stops

        def score_stops(item):
            """Classify item's DFA state and accumulate scores. Returns True if quotient."""
            dfa_state = item.dfa_state
            weight = item.weight

            # Preimage: source completed the target prefix at a final FST state
            if dfa_state in preimage_lookup:
                eos_lp = item.lm_state.logp_next._scores.get(EOS)
                eos_scores.append(weight + (eos_lp if eos_lp is not None else -np.inf))

            # Resume frontier: carry forward for the next target step
            if dfa_state in resume_states:
                for y in resume_states[dfa_state]:
                    carry_forward.setdefault(y, []).append(item)

            # Quotient: full weight, no further expansion needed
            q_syms = q_lookup.get(dfa_state)
            if q_syms is not None:
                for y in q_syms:
                    scores.setdefault(y, []).append(weight)
                    carry_forward.setdefault(y, []).append(item)

            # Remainder: weight + P(EOS), skipping symbols already covered by Q
            r_syms = r_lookup.get(dfa_state)
            if r_syms is not None:
                eos_lp = item.lm_state.logp_next._scores.get(EOS)
                eos_weight = weight + (eos_lp if eos_lp is not None else -np.inf)
                for y in r_syms:
                    if q_syms is None or y not in q_syms:
                        scores.setdefault(y, []).append(eos_weight)
                    carry_forward.setdefault(y, []).append(item)

            return bool(q_syms)

        # Best-first search: pop highest-weight item, score it, expand if not quotient
        heap = list(self._beam)
        heapq.heapify(heap)

        steps = 0
        while heap and steps < self.tlm.max_steps:
            steps += 1
            item = heapq.heappop(heap)

            is_quotient = score_stops(item)
            if is_quotient:
                continue

            # Expand: advance the inner LM by each source symbol x, push children
            lm_logp_next = item.lm_state.logp_next
            for x, next_dfa_state in dfa.arcs(item.dfa_state):
                next_weight = item.weight + lm_logp_next[x]
                if next_weight == -np.inf:
                    continue
                heapq.heappush(heap, BeamItem(next_dfa_state, item.lm_state << x, next_weight))

        # Budget exhausted: score remaining items without expanding so they
        # still contribute to scores and carry_forward.
        for item in heap:
            score_stops(item)

        # Normalize to a proper conditional: P(y | target) = exp(score(y)) / Z
        all_raw = []
        for y, s_list in scores.items():
            all_raw.append(logsumexp(s_list))

        eos_raw = logsumexp(eos_scores)
        if eos_raw > -np.inf:
            all_raw.append(eos_raw)

        Z = logsumexp(all_raw)

        normalized = {}
        for y, s_list in scores.items():
            normalized[y] = logsumexp(s_list) - Z

        eos_logp = eos_raw - Z if Z > -np.inf else -np.inf
        normalized[self.tlm.eos] = eos_logp

        self._logp_next_cache = LogpNext(normalized)
        self._carry_forward_cache = carry_forward

    def path(self):
        tokens = []
        h = self.history
        while h:
            h, token = h
            tokens.append(token)
        tokens.reverse()
        return tokens

    def __repr__(self):
        return f'TransducedState(target={self._peekaboo_state.target!r})'


class TransducedLM:
    """Incremental transduced language model.

    Computes the pushforward of an inner LM through an FST, providing
    the StateLM-compatible interface for autoregressive decoding on
    the target side.
    """

    def __init__(self, inner_lm, fst, max_steps=1000, max_beam=100, eos='<EOS>',
                 decomp_state_cls=None, univ_cls=None):
        """
        Args:
            inner_lm: LM with StateLM interface (has .initial(), state << x, state.logp_next)
            fst: FST instance mapping source → target
            max_steps: budget per logp_next computation
            max_beam: max items carried forward between steps
            eos: outer EOS token
            decomp_state_cls: incremental decomposition class (default: PeekabooState).
                Must support ``state >> y``, and expose ``.decomp``, ``.dfa``,
                ``.resume_frontiers``, ``.preimage_stops``, ``.target``.
            univ_cls: universality precomputation class (default: FstUniversality).
                Must accept ``(fst)`` and be passable as ``univ=`` to decomp_state_cls.
        """
        self.inner_lm = inner_lm
        self.fst = fst
        self.max_steps = max_steps
        self.max_beam = max_beam
        self.eos = eos
        self._decomp_state_cls = decomp_state_cls or _DefaultDecompState
        self._univ_cls = univ_cls or _DefaultUniv
        self._univ = self._univ_cls(fst)

    def initial(self):
        """Return the initial TransducedState (empty target prefix)."""
        # Create initial decomposition state
        peekaboo = self._decomp_state_cls(self.fst, '', parent=None, univ=self._univ)

        # Create initial DFA state (start states of the powerset DFA)
        dfa = peekaboo.dfa
        start_states = list(dfa.start())

        # Initial beam: one item per DFA start state, with the inner LM initial state
        inner_initial = self.inner_lm.initial()
        beam = [
            BeamItem(
                dfa_state=s,
                lm_state=inner_initial,
                weight=0.0,
            )
            for s in start_states
        ]

        return TransducedState(self, peekaboo, beam, 0.0)

    def __repr__(self):
        return f'TransducedLM(inner={self.inner_lm!r}, fst={self.fst!r})'
