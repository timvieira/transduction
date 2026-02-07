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

import numpy as np
from dataclasses import dataclass
from arsenal.datastructures import LocatorMaxHeap

from transduction.lm.base import LMState, LogpNext

# Default incremental decomposition; can be overridden via TransducedLM constructor.
from transduction.peekaboo_recursive import PeekabooState as _DefaultDecompState
from transduction.peekaboo_recursive import FstUniversality as _DefaultUniv


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


@dataclass(frozen=False, eq=True, unsafe_hash=True)
class BeamItem:
    """Search frontier element for source-side expansion."""
    dfa_state: object       # frozenset (powerset DFA state)
    lm_state: object        # inner LM state
    weight: float           # cumulative log P_inner of source prefix


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
        if y == self.eos:
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

        Uses the PeekabooState's decomposition to identify which DFA states
        correspond to quotient (Q) or remainder (R) stops for each next
        target symbol, and accumulates log-probability scores.
        """
        decomp = self._peekaboo_state.decomp
        dfa = self._peekaboo_state.dfa
        _target = self._peekaboo_state.target

        # Build reverse lookups: dfa_state → set of symbols y
        q_lookup = {}   # states that are quotient stops for symbol y
        r_lookup = {}   # states that are remainder stops for symbol y
        for y, d in decomp.items():
            for state in d.quotient:
                q_lookup.setdefault(state, set()).add(y)
            for state in d.remainder:
                r_lookup.setdefault(state, set()).add(y)

        # Resume frontiers for carry-forward
        resume_states = {}  # dfa_state → set of symbols y
        for y, states in self._peekaboo_state.resume_frontiers.items():
            for state in states:
                resume_states.setdefault(state, set()).add(y)

        # scores[y] = list of log-weights contributing to P(y | target)
        scores = {}
        eos_scores = []
        # carry_forward[y] = list of BeamItems to seed the next step
        carry_forward = {}

        EOS = self.tlm.inner_lm.eos if hasattr(self.tlm.inner_lm, 'eos') else self.tlm.inner_lm.initial().eos

        # DFA states where the source has fully produced `target` and FST is final
        preimage_lookup = self._peekaboo_state.preimage_stops

        # Initialize priority queue from beam
        queue = LocatorMaxHeap()
        for item in self._beam:
            queue[item] = item.weight

        steps = 0
        while queue and steps < self.tlm.max_steps:
            steps += 1
            item, _ = queue.pop()
            dfa_state = item.dfa_state
            lm_state = item.lm_state
            weight = item.weight

            lm_logp_next = lm_state.logp_next

            # --- Preimage stop check (EOS) ---
            if dfa_state in preimage_lookup:
                eos_scores.append(weight + lm_logp_next[EOS])

            # --- Quotient check ---
            is_quotient = False
            q_syms = q_lookup.get(dfa_state, set())
            if q_syms:
                for y in q_syms:
                    scores.setdefault(y, []).append(weight)
                    carry_forward.setdefault(y, []).append(item)
                is_quotient = True

            # --- Remainder check ---
            # Skip symbols that already have a quotient at this state:
            # the quotient covers all continuations (including stopped),
            # so remainder would double-count.
            if dfa_state in r_lookup:
                for y in r_lookup[dfa_state]:
                    if y not in q_syms:
                        scores.setdefault(y, []).append(weight + lm_logp_next[EOS])
                    carry_forward.setdefault(y, []).append(item)

            # Quotient states accept all continuations — don't expand
            if is_quotient:
                continue

            # --- Expand ---
            for x, next_dfa_state in dfa.arcs(dfa_state):
                next_weight = float(weight + lm_logp_next[x])
                if next_weight == -np.inf:
                    continue
                next_item = BeamItem(
                    dfa_state=next_dfa_state,
                    lm_state=lm_state << x,
                    weight=next_weight,
                )
                queue[next_item] = next_weight

        # Drain remaining queue items: record items at resume frontiers
        while queue:
            item, _ = queue.pop()
            dfa_state = item.dfa_state
            if dfa_state in preimage_lookup:
                lm_logp_next_drain = item.lm_state.logp_next
                eos_scores.append(item.weight + lm_logp_next_drain[EOS])
            if dfa_state in resume_states:
                for y in resume_states[dfa_state]:
                    carry_forward.setdefault(y, []).append(item)
            # Also check Q/R for drained items
            q_syms = q_lookup.get(dfa_state, set())
            if q_syms:
                for y in q_syms:
                    scores.setdefault(y, []).append(item.weight)
                    carry_forward.setdefault(y, []).append(item)
            if dfa_state in r_lookup:
                lm_logp_next = item.lm_state.logp_next
                for y in r_lookup[dfa_state]:
                    if y not in q_syms:
                        scores.setdefault(y, []).append(item.weight + lm_logp_next[EOS])
                    carry_forward.setdefault(y, []).append(item)

        # Normalize: logp_next[y] = logsumexp(scores[y]) - logsumexp(all_scores)
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
